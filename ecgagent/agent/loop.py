"""The bounded phase loop.

Not free-form ReAct: an unbounded tool loop is unbudgetable and unauditable,
and neither is acceptable for something that adjudicates a clinical rule
engine. Not the fixed pipeline the existing layered path uses either — that
one decides in advance what evidence the model sees. This is the middle: fixed
phases with hard per-phase call budgets, and free choice of evidence inside
each phase.

Phase 5 is deterministic code, not a model turn. Asking the model that wrote
the verdict whether the verdict is sound is not verification.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..backends.base import (
    BackendCapabilities,
    LLMBackend,
    LLMResponse,
    ToolOutcome,
    backend_capabilities,
    model_visible_outcome_citations,
)
from ..evidence.briefing import build_chart_briefing
from ..evidence.ledger import evidence_set_id
from ..evidence.pointer import PointerError, display_precision
from ..evidence.store import EvidenceStore
from ..tools.registry import ToolRegistry, build_default_registry
from ..verify import (
    VerificationPolicy,
    VerificationReport,
    claim_is_qualified,
    verify_structured,
)
from . import prompts
from .runtime import (
    DEFAULT_MAX_CONSECUTIVE_MAX_TOKENS,
    DEFAULT_MAX_MODEL_TURNS,
    DEFAULT_MAX_REVISIONS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MAX_WALL_SECONDS,
    DEFAULT_PHASE_STATE_RETRIES,
    phase_turn_limit as _max_turns,
    positive_float_env as _positive_float_env,
    positive_int_env as _positive_int_env,
)

# Local/provider tool protocols permit several calls in one model turn.  The
# old ``budget + 4`` bound silently assumed one call per turn and allowed a
# nominally bounded phase to run for dozens of expensive generations.  Backends
# declare their real batch width; legacy/mock backends default to one.
AGENT_PROTOCOL_VERSION = "ecgagent.v10"


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    instruction: str
    tool_budget: int
    structured: bool = False
    # Evidence-gathering phases may need a compact structured state without
    # using the final verdict schema. ``structured`` continues to mean that
    # the phase returns the final diagnosis contract and therefore must not be
    # folded into rolling phase memory.
    response_schema: dict[str, Any] | None = None
    max_tokens: int | None = None
    allowed_tools: tuple[str, ...] | None = None
    minimum_tool_calls: int = 0
    # Evidence phases should normally finish against named clinical coverage
    # requirements rather than a global call count.  Each row is
    # ``(coverage_id, allowed_tool_names, required_distinct_views)``.  One
    # successful call may satisfy several requirements when it is genuinely a
    # shared high-density view; repeated identical calls never count twice.
    coverage_requirements: tuple[
        tuple[str, tuple[str, ...], int], ...
    ] = ()
    # Challenge can require a genuinely new slice of evidence without
    # pretending that contradictory evidence must exist.  A call is novel when
    # its canonical tool+arguments signature was not successfully read in an
    # earlier phase.
    require_novel_tool_views: bool = False
    required_tool_counts: tuple[tuple[str, int], ...] = ()
    # Argument-aware requirements are used when calling a tool name alone is
    # not enough to prove coverage.  For example, two morphology-map calls do
    # not establish that both QRS and T/U were reviewed if the model requested
    # the QRS profile twice.  Each row is ``(tool, arguments, count)``; only the
    # listed arguments have to match, so callers may still add optional
    # narrowing arguments such as lead subsets.
    required_tool_calls: tuple[
        tuple[str, tuple[tuple[str, Any], ...], int], ...
    ] = ()
    # Local text-tool backends may execute an invariant baseline packet before
    # the first model turn. Calls still pass through ToolRegistry, consume
    # budget and remain fully auditable. Provider-native backends ignore it.
    prefetch_tool_calls: tuple[
        tuple[str, tuple[tuple[str, Any], ...]], ...
    ] = ()
    # A deterministic resolver may need the complete result without exposing
    # it to the model. These calls are dispatched and audited like every other
    # prefetch, but are omitted from the next-model evidence packet.
    program_only_prefetch_tool_calls: tuple[
        tuple[str, tuple[tuple[str, Any], ...]], ...
    ] = ()


# Sized against a real record: 8 matched rules and 36 abstentions on the
# sample payload. Opening one rule detail each already costs 8, so a budget of
# 10 forced synthesis from a half-read chart — the model retreated to a single
# verifiable diagnosis and abstained on everything else.
DEFAULT_PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec(
        "orient",
        prompts.ORIENT_INSTRUCTION,
        tool_budget=0,
        max_tokens=2048,
    ),
    PhaseSpec(
        "test",
        prompts.TEST_INSTRUCTION,
        tool_budget=18,
        max_tokens=3072,
    ),
    PhaseSpec(
        "adjudicate",
        prompts.ADJUDICATE_INSTRUCTION,
        tool_budget=6,
        max_tokens=2048,
    ),
    PhaseSpec(
        "synthesize",
        prompts.SYNTHESIZE_INSTRUCTION,
        tool_budget=0,
        structured=True,
        max_tokens=6144,
    ),
)


def adaptive_phases(store: EvidenceStore) -> tuple[PhaseSpec, ...]:
    """Scale evidence budgets to chart complexity, within audited hard caps.

    A chart with one matched rule should not spend the same 24 evidence calls
    as a chart with eight diagnoses and several suppressed candidates. Ten
    Test calls and three Adjudicate calls are the conservative floor; the
    original 18/6 limits remain the ceiling for complex or quality-limited
    records.
    """
    clinical = store.clinical()
    matched = len(clinical.get("final_statements") or [])
    change_candidates = len(clinical.get("suppressed_statements") or []) + len(
        clinical.get("borderline_statements") or []
    )
    conflicts = len(clinical.get("conflicts") or [])
    gate_state = str(
        store.raw("/metadata/diagnostic_gate/state", "unknown") or "unknown"
    ).lower()
    quality_penalty = 0 if gate_state == "pass" else 2

    test_budget = min(
        18,
        max(
            10,
            2
            + (2 * matched)
            + min(4, change_candidates)
            + min(2, conflicts)
            + quality_penalty,
        ),
    )
    adjudicate_budget = min(
        6,
        max(
            3,
            2
            + min(2, change_candidates)
            + min(2, conflicts)
            + (1 if quality_penalty else 0),
        ),
    )
    return tuple(
        PhaseSpec(
            key=phase.key,
            instruction=phase.instruction,
            tool_budget=(
                test_budget
                if phase.key == "test"
                else adjudicate_budget
                if phase.key == "adjudicate"
                else phase.tool_budget
            ),
            structured=phase.structured,
            response_schema=phase.response_schema,
            max_tokens=phase.max_tokens,
            allowed_tools=phase.allowed_tools,
        )
        for phase in DEFAULT_PHASES
    )


def _accepts_keyword(function: Any, name: str) -> bool:
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        return False


def _item_citation_tokens(item: Mapping[str, Any]) -> list[str]:
    """Every citation an evidence item carries, primary and supporting alike.

    A pointer cited only by a `supporting_values` entry is cited just as
    bindingly as one in `citations`; missing it would let the caveat and
    provenance machinery treat a comparative claim as if half of it were
    unsourced.
    """
    tokens = [str(token) for token in (item.get("citations") or [])]
    for extra in item.get("supporting_values") or []:
        if isinstance(extra, Mapping):
            token = str(extra.get("citation") or "").strip()
            if token:
                tokens.append(token)
    return tokens


def _prompt_fingerprint() -> str:
    from ..tools import orient, query

    prompt_text = "\n".join(
        [
            prompts.SYSTEM_PROMPT,
            *(phase.instruction for phase in DEFAULT_PHASES),
            prompts.REVISION_INSTRUCTION,
            json.dumps(prompts.OUTPUT_SCHEMA, sort_keys=True),
            json.dumps(
                [spec.to_openai() for spec in (*orient.SPECS, *query.SPECS)],
                sort_keys=True,
            ),
        ]
    )
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


PROMPT_FINGERPRINT = _prompt_fingerprint()


@dataclass
class PhaseRecord:
    key: str
    tool_budget: int = 0
    exposed_tools: list[str] = field(default_factory=list)
    prefetched_tool_calls: int = 0
    text: str = ""
    tool_calls: int = 0        # successful; these are what the budget counts
    rejected_calls: int = 0
    turns: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = ""
    duplicate_batches: int = 0
    tool_loop_fused: bool = False
    phase_guard_attempts: int = 0
    phase_guard_passed: bool | None = None
    phase_guard_problems: list[str] = field(default_factory=list)
    phase_sanitizations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.key,
            "tool_budget": self.tool_budget,
            "exposed_tools": list(self.exposed_tools),
            "prefetched_tool_calls": self.prefetched_tool_calls,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "rejected_calls": self.rejected_calls,
            "elapsed_s": round(self.elapsed_s, 2),
            "stop_reason": self.stop_reason,
            "duplicate_batches": self.duplicate_batches,
            "tool_loop_fused": self.tool_loop_fused,
            "phase_guard_attempts": self.phase_guard_attempts,
            "phase_guard_passed": self.phase_guard_passed,
            "phase_guard_problems": list(self.phase_guard_problems),
            "phase_sanitizations": list(self.phase_sanitizations),
            "text": self.text,
        }


@dataclass
class AgentResult:
    record_id: str
    verdict: dict[str, Any] | None = None
    # General knowledge may be retrieved after the measurement survey to plan
    # the provisional differential.  It is kept as audit metadata only and is
    # never a source of patient evidence.
    knowledge_navigation: dict[str, Any] | None = None
    # When the optional detached post-hoc challenger is enabled, this is the
    # already evidence-verified verdict that existed before that second review.
    pre_knowledge_verdict: dict[str, Any] | None = None
    knowledge_review: dict[str, Any] | None = None
    knowledge_revisions: int = 0
    phases: list[PhaseRecord] = field(default_factory=list)
    verification: VerificationReport | None = None
    revisions: int = 0
    refused: str | None = None
    error: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)
    _trace_tool_calls: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    _trace_briefing_text: str = field(default="", repr=False)
    _trace_intermediate_contexts: list[dict[str, str]] = field(
        default_factory=list,
        repr=False,
    )

    @property
    def ok(self) -> bool:
        return self.verdict is not None and self.error is None and self.refused is None

    @property
    def verified(self) -> bool:
        return self.verification is not None and self.verification.passed

    @property
    def human_report(self) -> str:
        """Clinician-readable rendering of the same structured verdict."""
        from ..report import render_human_report

        return render_human_report(
            self.verdict,
            record_id=self.record_id,
            verified=(
                True
                if self.verified
                else False
                if self.verdict is not None
                else None
            ),
            knowledge_navigation=self.knowledge_navigation,
            knowledge_review=self.knowledge_review,
            error=self.error,
        )

    @property
    def brief_report(self) -> str:
        """Short clinician-readable Markdown derived from the same verdict."""
        from ..report import render_brief_report

        return render_brief_report(
            self.verdict,
            record_id=self.record_id,
            verified=(
                True
                if self.verified
                else False
                if self.verdict is not None
                else None
            ),
            error=self.error,
        )

    def summary(self) -> str:
        if self.refused:
            return f"{self.record_id}: model declined the request ({self.refused})"
        if self.error:
            return f"{self.record_id}: failed — {self.error}"
        if self.verdict is None:
            return f"{self.record_id}: no verdict produced"

        diagnoses = self.verdict.get("diagnoses") or []
        changed = [d for d in diagnoses if d.get("status") in {"added", "withdrawn", "downgraded"}]
        verdict_line = "verified" if self.verified else "VERIFICATION FAILED"
        if any("status" in diagnosis for diagnosis in diagnoses if isinstance(diagnosis, dict)):
            headline = (
                f"{self.record_id}: {len(diagnoses)} diagnoses "
                f"({len(changed)} changed vs the rule engine), {verdict_line}"
            )
        else:
            headline = (
                f"{self.record_id}: {len(diagnoses)} independent diagnosis/diagnoses, "
                f"{verdict_line}"
            )
        lines = [
            headline + (f" after {self.revisions} revision(s)" if self.revisions else "")
        ]
        if self.verdict.get("summary"):
            lines.append(f"  {self.verdict['summary']}")
        for diagnosis in changed:
            lines.append(
                f"  [{diagnosis.get('status')}] {diagnosis.get('code')}: "
                f"{diagnosis.get('adjudication', '')[:120]}"
            )
        for abstention in self.verdict.get("abstentions") or []:
            lines.append(
                f"  [abstained] {abstention.get('topic')} — needs: "
                f"{abstention.get('what_would_resolve_it', '')[:90]}"
            )
        if self.verdict.get("human_review", {}).get("required"):
            reasons = ", ".join(self.verdict["human_review"].get("reasons") or [])
            lines.append(f"  human review required: {reasons}")
        return "\n".join(lines)

    def render_trace(self, payload: dict[str, Any] | None = None) -> str:
        """Render model-visible excerpts beside full raw tool audit results."""

        from ..trace import render_agent_trace

        return render_agent_trace(
            payload or self.to_dict(),
            tool_calls=self._trace_tool_calls or None,
            briefing_text=self._trace_briefing_text or None,
            intermediate_contexts=self._trace_intermediate_contexts or None,
        )

    @property
    def trace_report(self) -> str:
        return self.render_trace()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "ok": self.ok,
            "verified": self.verified,
            "revisions": self.revisions,
            "refused": self.refused,
            "error": self.error,
            "verdict": self.verdict,
            "knowledge_navigation": self.knowledge_navigation,
            "pre_knowledge_verdict": self.pre_knowledge_verdict,
            "knowledge_review": self.knowledge_review,
            "knowledge_revisions": self.knowledge_revisions,
            "brief_report": self.brief_report,
            "human_report": self.human_report,
            "verification": self.verification.to_dict() if self.verification else None,
            "phases": [phase.to_dict() for phase in self.phases],
            "audit": self.audit,
        }


@dataclass
class ECGAgent:
    """Runs a configured bounded phase loop for one record.

    The defaults retain the legacy rule-adjudication profile.  New diagnosis
    workflows use :class:`ECGDiagnosticAgent`, which supplies a neutral
    briefing, independent diagnostic prompts and measurement-only phase tools.
    """

    store: EvidenceStore
    backend: LLMBackend
    registry: ToolRegistry | None = None
    phases: tuple[PhaseSpec, ...] = DEFAULT_PHASES
    max_revisions: int = DEFAULT_MAX_REVISIONS
    max_model_turns: int = DEFAULT_MAX_MODEL_TURNS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS
    max_consecutive_max_tokens: int = DEFAULT_MAX_CONSECUTIVE_MAX_TOKENS
    phase_state_retries: int = DEFAULT_PHASE_STATE_RETRIES
    verify_policy: VerificationPolicy | None = None
    on_event: Any = None  # optional callable(str) for progress reporting
    on_checkpoint: Any = None  # optional callable(dict) after every completed phase
    adaptive_budgets: bool = True
    prompt_module: Any = field(default=prompts, repr=False)
    briefing_builder: Any = field(default=build_chart_briefing, repr=False)
    protocol_version: str = AGENT_PROTOCOL_VERSION
    prompt_fingerprint: str = PROMPT_FINGERPRINT
    mode: str = "adjudicate"
    briefing_intro: str = (
        "Record briefing. Everything below was produced by ecgfeat; "
        "the pointers are citable as-is."
    )
    revision_allowed_tools: tuple[str, ...] | None = None
    _briefing_chars: int = field(default=0, repr=False)
    _context_compactions: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    _verdict_normalizations: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    _run_started: float = field(default=0.0, repr=False)
    _model_turns: int = field(default=0, repr=False)
    _prompt_tokens: int = field(default=0, repr=False)
    _completion_tokens: int = field(default=0, repr=False)
    _total_tokens: int = field(default=0, repr=False)
    _consecutive_max_tokens: int = field(default=0, repr=False)
    _last_phase_state_text: str = field(default="", repr=False)
    _backend_capabilities: BackendCapabilities = field(
        default_factory=BackendCapabilities,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._backend_capabilities = backend_capabilities(self.backend)
        self.max_model_turns = _positive_int_env(
            "ECG_AGENT_MAX_MODEL_TURNS", self.max_model_turns
        )
        self.max_total_tokens = _positive_int_env(
            "ECG_AGENT_MAX_TOTAL_TOKENS", self.max_total_tokens
        )
        self.max_wall_seconds = _positive_float_env(
            "ECG_AGENT_MAX_WALL_SECONDS", self.max_wall_seconds
        )
        self.max_consecutive_max_tokens = _positive_int_env(
            "ECG_AGENT_MAX_CONSECUTIVE_MAX_TOKENS",
            self.max_consecutive_max_tokens,
        )
        self.phase_state_retries = _positive_int_env(
            "ECG_AGENT_PHASE_STATE_ATTEMPTS", self.phase_state_retries + 1
        ) - 1
        if (
            self.mode == "adjudicate"
            and self.adaptive_budgets
            and self.phases is DEFAULT_PHASES
        ):
            self.phases = adaptive_phases(self.store)
        if self.registry is None:
            # Budget is applied per phase below, so the registry starts unbounded.
            self.registry = build_default_registry(self.store, budget=None)
        if self.verify_policy is None:
            # The verdict is structured, so an evidence item with no citations
            # is unambiguously unsupported and must block.
            self.verify_policy = VerificationPolicy.for_structured()

    def _emit(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)

    def _runtime_snapshot(self) -> dict[str, Any]:
        elapsed = (
            time.perf_counter() - self._run_started if self._run_started else 0.0
        )
        return {
            "elapsed_seconds": round(elapsed, 3),
            "model_turns": self._model_turns,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "consecutive_max_tokens": self._consecutive_max_tokens,
            "limits": {
                "max_model_turns": self.max_model_turns,
                "max_total_tokens": self.max_total_tokens,
                "max_wall_seconds": self.max_wall_seconds,
                "max_consecutive_max_tokens": self.max_consecutive_max_tokens,
                "phase_state_retries": self.phase_state_retries,
            },
        }

    def _check_runtime_budget(self) -> None:
        snapshot = self._runtime_snapshot()
        if self._model_turns >= self.max_model_turns:
            raise RuntimeError(
                "agent model-turn budget exhausted before the next request "
                f"({self._model_turns}/{self.max_model_turns})"
            )
        if self._total_tokens >= self.max_total_tokens:
            raise RuntimeError(
                "agent cumulative token budget exhausted before the next request "
                f"({self._total_tokens}/{self.max_total_tokens})"
            )
        if float(snapshot["elapsed_seconds"]) >= self.max_wall_seconds:
            raise RuntimeError(
                "agent wall-clock budget exhausted before the next request "
                f"({snapshot['elapsed_seconds']:.1f}/{self.max_wall_seconds:.1f}s)"
            )

    def _record_model_response(self, response: LLMResponse) -> None:
        self._model_turns += 1
        usage = response.usage if isinstance(response.usage, Mapping) else {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if isinstance(prompt, int):
            self._prompt_tokens += prompt
        if isinstance(completion, int):
            self._completion_tokens += completion
        if isinstance(total, int):
            self._total_tokens += total
        elif isinstance(prompt, int) or isinstance(completion, int):
            self._total_tokens += int(prompt or 0) + int(completion or 0)

        if response.stop_reason == "max_tokens" and not response.wants_tools:
            self._consecutive_max_tokens += 1
        else:
            self._consecutive_max_tokens = 0
        self._emit(
            "[model] completed turn "
            f"{self._model_turns}/{self.max_model_turns}; "
            f"stop={response.stop_reason}; request_tokens={usage.get('total_tokens', '?')}; "
            f"cumulative_tokens={self._total_tokens}/{self.max_total_tokens}"
        )
        if self._consecutive_max_tokens >= self.max_consecutive_max_tokens:
            raise RuntimeError(
                "agent stopped after repeated non-tool completions reached "
                f"max_tokens ({self._consecutive_max_tokens} consecutive turns)"
            )

    def _complete_turn(self, **kwargs: Any) -> LLMResponse:
        self._check_runtime_budget()
        self._emit(
            "[model] starting turn "
            f"{self._model_turns + 1}/{self.max_model_turns}; "
            f"elapsed={self._runtime_snapshot()['elapsed_seconds']:.1f}s"
        )
        response = self.backend.complete(**kwargs)
        self._record_model_response(response)
        return response

    def _backend_usage_snapshot(self) -> dict[str, int]:
        getter = getattr(self.backend, "usage_total", None)
        if not callable(getter):
            return {}
        usage = getter()
        return {
            key: int(value)
            for key, value in (usage.items() if isinstance(usage, Mapping) else [])
            if key in {"turns", "prompt_tokens", "completion_tokens", "total_tokens"}
            and isinstance(value, int)
        }

    def _absorb_external_backend_usage(self, before: Mapping[str, int]) -> None:
        """Account for isolated helper turns that call the backend directly."""

        after = self._backend_usage_snapshot()
        turns = max(0, int(after.get("turns", 0)) - int(before.get("turns", 0)))
        prompt = max(
            0,
            int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0)),
        )
        completion = max(
            0,
            int(after.get("completion_tokens", 0))
            - int(before.get("completion_tokens", 0)),
        )
        total = max(
            0,
            int(after.get("total_tokens", 0)) - int(before.get("total_tokens", 0)),
        )
        self._model_turns += turns
        self._prompt_tokens += prompt
        self._completion_tokens += completion
        self._total_tokens += total if total else prompt + completion

    def _checkpoint(self, result: AgentResult, *, status: str) -> None:
        if self.on_checkpoint is None:
            return
        payload = {
            "record_id": result.record_id,
            "status": status,
            "ok": result.ok,
            "verified": result.verified,
            "error": result.error,
            "revisions": result.revisions,
            "runtime": self._runtime_snapshot(),
            "phases": [phase.to_dict() for phase in result.phases],
            "tools": self._tools_audit(result.verdict),
            "updated_at_epoch": time.time(),
        }
        payload.update(self._checkpoint_payload_extra(result, status=status))
        try:
            self.on_checkpoint(payload)
        except Exception as exc:
            # A progress file is operational telemetry, not patient evidence;
            # failure to write it must not invalidate an otherwise sound read.
            self._emit(
                f"[checkpoint] warning: {type(exc).__name__}: {exc}"
            )

    def _checkpoint_payload_extra(
        self,
        result: AgentResult,
        *,
        status: str,
    ) -> dict[str, Any]:
        """Profile-owned crash-recovery state persisted with each checkpoint."""

        return {}

    # -- main entry ---------------------------------------------------------
    def run(self) -> AgentResult:
        self._run_started = time.perf_counter()
        self._model_turns = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._consecutive_max_tokens = 0
        self._last_phase_state_text = ""
        result = AgentResult(record_id=self.store.record_id)
        briefing = self.briefing_builder(self.store)
        self._briefing_chars = len(briefing.text)
        self.registry.seed_citations(
            briefing.citations,
            source="chart_briefing",
            model_text=briefing.text,
        )
        tools = self.registry.anthropic_tools()

        messages: list[dict[str, Any]] = [
            self.backend.user_turn(
                self.briefing_intro + "\n\n" + briefing.text
            )
        ]
        result._trace_briefing_text = (
            self.briefing_intro + "\n\n" + briefing.text
        )

        candidate: dict[str, Any] | None = None
        try:
            for configured_spec in self.phases:
                spec = self._runtime_phase_spec(configured_spec, result)
                record = self._run_phase(spec, messages, tools)
                result.phases.append(record)
                if record.stop_reason == "refusal":
                    result.refused = record.text
                    return self._finish(result)
                self._postprocess_phase(
                    result,
                    record=record,
                    messages=messages,
                    tools=tools,
                )
                self._checkpoint(result, status=f"phase_completed:{record.key}")
                if record.phase_guard_passed is False:
                    result.error = (
                        f"phase `{record.key}` failed the hard state/evidence guard: "
                        + "; ".join(record.phase_guard_problems[:6])
                    )
                    return self._finish(result)

            candidate = self._parse_verdict(result.phases[-1].text)
            if candidate is None:
                candidate = self._request_revision(
                    messages,
                    tools,
                    (
                        "The synthesis response was not a parseable JSON "
                        "object. Return the complete verdict as one JSON "
                        "object only, matching the schema."
                    ),
                    result,
                    reason="synthesis was not parseable JSON",
                    measurement_reread=False,
                )
                if candidate is None:
                    result.error = (
                        "synthesis and its revisions did not return "
                        "parseable JSON"
                    )
                    return self._finish(result)
            else:
                candidate = self._normalize_verdict(candidate)

            unchanged_revisions = 0
            while True:
                structural = self._validate_verdict_contract(candidate)
                if structural:
                    if result.revisions >= self.max_revisions:
                        result.verdict = candidate
                        result.error = (
                            "verdict failed deterministic contract validation: "
                            + "; ".join(structural[:8])
                        )
                        break
                    feedback = self._contract_feedback(candidate, structural)
                    previous_candidate = candidate
                    revised_candidate = self._request_revision(
                        messages,
                        tools,
                        feedback,
                        result,
                        reason=(
                            f"contract failed ({len(structural)} problem(s))"
                        ),
                        current_candidate=candidate,
                        measurement_reread=True,
                    )
                    if revised_candidate is None:
                        result.error = (
                            "contract revisions did not return parseable JSON"
                        )
                        break
                    if self._verdict_signature(
                        revised_candidate
                    ) == self._verdict_signature(previous_candidate):
                        unchanged_revisions += 1
                    else:
                        unchanged_revisions = 0
                    candidate = revised_candidate
                    if unchanged_revisions >= 2:
                        result.verdict = candidate
                        result.error = (
                            "contract revision stalled: the model returned an "
                            "unchanged invalid verdict twice"
                        )
                        break
                    continue

                result.verdict = candidate
                result.verification = self._verify(candidate)
                if result.verification.passed or result.revisions >= self.max_revisions:
                    break

                previous_candidate = candidate
                revised_candidate = self._request_revision(
                    messages,
                    tools,
                    result.verification.feedback(),
                    result,
                    reason=(
                        f"verification failed "
                        f"({len(result.verification.blocking)} blocking)"
                    ),
                    current_candidate=candidate,
                    measurement_reread=True,
                )
                if revised_candidate is None:
                    result.error = "verification revisions did not return parseable JSON"
                    break
                if self._verdict_signature(
                    revised_candidate
                ) == self._verdict_signature(previous_candidate):
                    unchanged_revisions += 1
                else:
                    unchanged_revisions = 0
                candidate = revised_candidate
                if unchanged_revisions >= 2:
                    result.verdict = candidate
                    result.error = (
                        "evidence revision stalled: the model returned an "
                        "unchanged unverified verdict twice"
                    )
                    break

            # A diagnosis profile may add a separate post-hoc reviewer here.
            # The default adjudication profile is a no-op.  The reviewer is
            # isolated from the earlier survey-to-hypothesis navigator and may
            # revise a verdict only after returning to patient measurements.
            if result.verified:
                self._postprocess_verified_result(
                    result,
                    messages=messages,
                    tools=tools,
                )

        except Exception as exc:  # surfaced in the result, never crashes a batch
            result.error = f"{type(exc).__name__}: {exc}"
            # Preserve an already synthesized candidate as explicitly
            # unverified output. The runtime error remains blocking and visible.
            if result.verdict is None and isinstance(candidate, Mapping):
                result.verdict = dict(candidate)

        return self._finish(result)

    def _postprocess_phase(
        self,
        result: AgentResult,
        *,
        record: PhaseRecord,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> None:
        """Optional hook between bounded phases.

        The diagnosis-first profile uses this boundary after ``survey`` to
        retrieve governed general knowledge before its tool-free provisional
        hypothesis phase.  The generic Agent deliberately does nothing.
        """

        return None

    def _canonicalize_phase_state(
        self,
        spec: PhaseSpec,
        record: PhaseRecord,
        *,
        calls_at_start: int,
    ) -> str:
        """Return the state stored in rolling model memory.

        Generic profiles keep the model response verbatim. Diagnosis-first
        profiles override this hook to apply typed model patches to a central
        program-owned ledger before any backend builds its compact phase
        memory. The raw patch remains in ``record.text`` for audit.
        """

        return record.text

    def _additional_phase_state_problems(
        self,
        spec: PhaseSpec,
        state: Mapping[str, Any],
        *,
        calls_at_start: int,
    ) -> list[str]:
        """Profile-owned semantic checks beyond the prompt JSON contract."""

        return []

    def _runtime_phase_spec(
        self,
        spec: PhaseSpec,
        result: AgentResult,
    ) -> PhaseSpec:
        """Return the per-record phase contract used for this run.

        The generic Agent keeps its configured phase unchanged. Diagnostic
        profiles may narrow a later phase after reading the preceding
        hypothesis state, so irrelevant tool definitions never enter that
        model request.
        """

        return spec

    def _validate_verdict_contract(
        self,
        candidate: Mapping[str, Any],
    ) -> list[str]:
        """Run the profile contract with any deterministic phase-state context.

        Diagnosis-first synthesis is not allowed to reinterpret the final
        Challenge ledger.  The generic adjudication profile has no such state,
        so the optional argument is passed only when its validator declares it.
        """

        validator = self.prompt_module.validate_verdict
        kwargs: dict[str, Any] = {"evidence_document": self.store.document}
        if _accepts_keyword(validator, "hypothesis_state"):
            kwargs["hypothesis_state"] = getattr(
                self,
                "_latest_phase_state",
                None,
            )
        return list(validator(candidate, **kwargs))

    def _revision_allowed_tool_names(self, feedback: str) -> tuple[str, ...] | None:
        """Return tools exposed for a measurement-driven verdict repair."""

        return self.revision_allowed_tools

    def _postprocess_verified_result(
        self,
        result: AgentResult,
        *,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> None:
        """Optional post-verification extension point.

        The generic Agent deliberately does nothing.  The diagnosis-first
        profile overrides this for the detached knowledge challenge.
        """

        return None

    def _contract_feedback(
        self,
        candidate: dict[str, Any],
        structural: Sequence[str],
    ) -> str:
        """Add exact, safe repair routes to structural verifier feedback.

        Live failures showed the model repeatedly replacing a missing pointer
        with prose such as "as stated in the briefing". The briefing already
        exposes the baseline statement pointer, but generic feedback did not
        tell the model which claim that pointer can legitimately support.
        """
        lines = [
            "The verdict violates the deterministic output contract. "
            "Fix every item:",
            *[f"- {problem}" for problem in structural[:16]],
        ]
        if self.mode != "adjudicate":
            lines.extend(
                [
                    "",
                    "Repair diagnostic evidence with measurement tools only. "
                    "A rule-engine conclusion is not evidence for an independent diagnosis.",
                ]
            )
            return "\n".join(lines)
        diagnoses = candidate.get("diagnoses")
        diagnoses = diagnoses if isinstance(diagnoses, list) else []
        clinical = self.store.clinical()
        final_statements = clinical.get("final_statements") or []
        baseline_by_code: dict[str, tuple[int, dict[str, Any]]] = {}
        for index, row in enumerate(final_statements):
            if not isinstance(row, dict):
                continue
            code = str(row.get("statement_code") or "")
            if code:
                baseline_by_code[code] = (index, row)

        hints: list[str] = []
        for index, diagnosis in enumerate(diagnoses):
            if not isinstance(diagnosis, dict):
                continue
            evidence = diagnosis.get("evidence") or []
            counterevidence = diagnosis.get("counterevidence") or []
            missing_support = any(
                isinstance(item, dict) and not item.get("citations")
                for item in evidence
            )
            missing_counter = any(
                isinstance(item, dict) and not item.get("citations")
                for item in counterevidence
            )
            if not missing_support and not missing_counter:
                continue
            code = str(diagnosis.get("code") or "")
            baseline = baseline_by_code.get(code)
            rule_ids = [
                str(rule_id)
                for rule_id in (diagnosis.get("rule_ids") or [])
                if rule_id
            ]
            if missing_support and baseline is not None:
                statement_index, row = baseline
                pointer = (
                    "/clinical_interpretation/final_statements/"
                    f"{statement_index}/statement"
                )
                if pointer in self.registry.whitelist:
                    hints.append(
                        f"diagnoses[{index}] `{code}`: the baseline statement is "
                        f"already citable as ev:{pointer}. It may support only an "
                        "explicit claim that the rule engine emitted that statement; "
                        "do not use it to support a measurement."
                    )
                rule_id = str(row.get("rule_id") or "")
                if rule_id and rule_id not in rule_ids:
                    rule_ids.append(rule_id)
            if rule_ids and (missing_support or missing_counter):
                rendered = ", ".join(f"`{rule_id}`" for rule_id in rule_ids[:3])
                hints.append(
                    f"diagnoses[{index}] `{code}`: call get_rule_detail for "
                    f"{rendered} and copy the exact ev:/ pointers returned for "
                    "any measurement or counterevidence claim. Delete an optional "
                    "unsupported item rather than leaving citations empty."
                )
            if len(hints) >= 10:
                break
        if hints:
            lines.extend(
                [
                    "",
                    "Exact repair routes for the empty-citation failures:",
                    *[f"- {hint}" for hint in hints[:10]],
                ]
            )
        return "\n".join(lines)

    def _finish(self, result: AgentResult) -> AgentResult:
        result.audit = {
            "agent_protocol": self.protocol_version,
            "prompt_fingerprint": self.prompt_fingerprint,
            "mode": self.mode,
            "input_fingerprint": self._input_fingerprint(),
            "evidence_access_contract": dict(self.store.access_contract),
            "phase_config": [
                {
                    "key": phase.key,
                    "tool_budget": phase.tool_budget,
                    "structured": phase.structured,
                    "has_response_schema": phase.response_schema is not None,
                    "max_tokens": phase.max_tokens,
                    "max_turns": _max_turns(
                        phase.tool_budget,
                        max_tools_per_turn=self._backend_capabilities.max_parallel_tool_calls,
                        phase_state_retries=(
                            self.phase_state_retries
                            if self._backend_capabilities.hard_phase_guards
                            else 0
                        ),
                    ),
                    "minimum_tool_calls": phase.minimum_tool_calls,
                    "coverage_requirements": [
                        {
                            "id": coverage_id,
                            "tools": list(tool_names),
                            "required_distinct_views": count,
                        }
                        for coverage_id, tool_names, count in phase.coverage_requirements
                    ],
                    "require_novel_tool_views": phase.require_novel_tool_views,
                    "required_tool_counts": [
                        [name, count]
                        for name, count in phase.required_tool_counts
                    ],
                    "required_tool_calls": [
                        {
                            "tool": name,
                            "arguments": dict(arguments),
                            "count": count,
                        }
                        for name, arguments, count in phase.required_tool_calls
                    ],
                    "prefetch_tool_calls": [
                        {
                            "tool": name,
                            "arguments": dict(arguments),
                        }
                        for name, arguments in phase.prefetch_tool_calls
                    ],
                    "allowed_tools": (
                        list(phase.allowed_tools)
                        if phase.allowed_tools is not None
                        else None
                    ),
                }
                for phase in self.phases
            ],
            "tools": self._tools_audit(result.verdict),
            "model": {
                "backend": getattr(self.backend, "name", "unknown"),
                "capabilities": {
                    "native_tool_calls": self._backend_capabilities.native_tool_calls,
                    "structured_output_level": (
                        self._backend_capabilities.structured_output_level
                    ),
                    "context_compaction": self._backend_capabilities.context_compaction,
                    "citation_aliases": self._backend_capabilities.citation_aliases,
                    "phase_memory": self._backend_capabilities.phase_memory,
                    "orchestrated_prefetch": (
                        self._backend_capabilities.orchestrated_prefetch
                    ),
                    "enforce_phase_coverage": (
                        self._backend_capabilities.enforce_phase_coverage
                    ),
                    "max_parallel_tool_calls": (
                        self._backend_capabilities.max_parallel_tool_calls
                    ),
                    "hard_phase_guards": self._backend_capabilities.hard_phase_guards,
                },
                "usage": self.backend.usage_total()
                if hasattr(self.backend, "usage_total")
                else {},
            },
            "briefing_chars": getattr(self, "_briefing_chars", 0),
            "context_compactions": list(self._context_compactions),
            "verdict_normalizations": list(self._verdict_normalizations),
            "runtime_controls": self._runtime_snapshot(),
        }
        if hasattr(self.backend, "audit_config"):
            result.audit["model"]["config"] = self.backend.audit_config()
        if hasattr(self.backend, "cache_report"):
            result.audit["model"]["cache"] = self.backend.cache_report()
        trace_calls = [
            record.to_trace_dict() for record in self.registry.calls
        ]
        context_trace_builder = getattr(self.backend, "tool_context_trace", None)
        if callable(context_trace_builder):
            context_rows = list(context_trace_builder() or [])
            used: set[int] = set()
            for call in trace_calls:
                for index, context_row in enumerate(context_rows):
                    if index in used or not isinstance(context_row, Mapping):
                        continue
                    if str(context_row.get("phase") or "") != str(
                        call.get("phase") or ""
                    ):
                        continue
                    if str(context_row.get("tool") or "") != str(
                        call.get("tool") or ""
                    ):
                        continue
                    if self.registry.call_signature(
                        str(context_row.get("tool") or ""),
                        dict(context_row.get("arguments") or {}),
                    ) != self.registry.call_signature(
                        str(call.get("tool") or ""),
                        dict(call.get("args") or {}),
                    ):
                        continue
                    call.update(
                        {
                            "model_context_text": context_row.get(
                                "model_context_text"
                            ),
                            "original_chars": context_row.get("original_chars"),
                            "model_context_chars": context_row.get(
                                "model_context_chars"
                            ),
                            "truncated_for_model": bool(
                                context_row.get("truncated_for_model")
                            ),
                            "rendered_candidate_citations": list(
                                context_row.get("candidate_citations") or []
                            ),
                            "atomic_view_citations": list(
                                context_row.get("atomic_view_citations") or []
                            ),
                            "packed_visible_citations": list(
                                context_row.get("visible_citations") or []
                            ),
                            "omitted_atom_count": int(
                                context_row.get("omitted_atom_count") or 0
                            ),
                        }
                    )
                    used.add(index)
                    break
        result._trace_tool_calls = trace_calls
        if result.verification is not None:
            self._emit("[verify] " + result.verification.summary().splitlines()[0])
        self._checkpoint(result, status="finished")
        return result

    def _input_fingerprint(self) -> str:
        clinical_fingerprint = self.store.raw(
            "/clinical_interpretation/artifact_fingerprint"
        )
        if clinical_fingerprint:
            return str(clinical_fingerprint)
        return self.store.fingerprint()

    @staticmethod
    def _verdict_signature(verdict: Mapping[str, Any]) -> str:
        """Canonical representation used to detect no-op revision loops."""
        return json.dumps(
            verdict,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _phase_state_problems(
        self,
        spec: PhaseSpec,
        text: str,
        *,
        calls_at_start: int,
        stop_reason: str,
    ) -> list[str] | None:
        """Return hard transition problems for grammar-constrained phase states.

        ``None`` means the backend does not promise a strict phase-state
        contract, so the legacy/provider path remains best effort.  The local
        vLLM backend opts in because its guided JSON grammar makes this a safe
        deterministic gate rather than a prose heuristic.
        """

        if not self._backend_capabilities.hard_phase_guards:
            return None
        if spec.structured or spec.response_schema is None:
            return None
        problems: list[str] = []
        try:
            parsed = json.loads(str(text or ""))
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if not isinstance(parsed, Mapping):
            return ["phase response is not a structured JSON object"]

        normalizer = getattr(self.backend, "normalize_verdict", None)
        normalized: Mapping[str, Any] = parsed
        if callable(normalizer):
            candidate = normalizer(dict(parsed))
            if isinstance(candidate, Mapping):
                normalized = candidate
        validator = getattr(self.prompt_module, "validate_phase_state", None)
        if callable(validator):
            phase_citations = {
                citation
                for call in self.registry.calls[calls_at_start:]
                if call.ok
                for citation in call.visible_citations
            }
            validator_kwargs: dict[str, Any] = {
                "phase_citations": tuple(phase_citations),
            }
            if _accepts_keyword(validator, "ledger_state"):
                validator_kwargs["ledger_state"] = getattr(
                    self,
                    "_latest_phase_state",
                    None,
                )
            problems.extend(validator(spec.key, normalized, **validator_kwargs))
        problems.extend(
            self._additional_phase_state_problems(
                spec,
                normalized,
                calls_at_start=calls_at_start,
            )
        )
        if stop_reason == "max_tokens":
            problems.append(
                "phase response ended at max_tokens and may be semantically truncated"
            )

        return problems

    def _fatal_phase_problems(self, problems: Sequence[str]) -> list[str]:
        """Guard problems that must end the run rather than be committed.

        A profile may declare that some problem has a deterministic safe
        outcome, in which case exhausting the repair budget should commit the
        patch and let that outcome apply. Profiles without the hook keep every
        problem fatal.
        """

        predicate = getattr(self.prompt_module, "is_remediated_phase_problem", None)
        if not callable(predicate):
            return list(problems)
        return [problem for problem in problems if not predicate(problem)]

    def _phase_citation_tokens(self, calls_at_start: int) -> set[str]:
        pointers = {
            citation
            for call in self.registry.calls[calls_at_start:]
            if call.ok
            for citation in call.visible_citations
        }
        tokens = {token for pointer in pointers for token in (pointer, f"ev:{pointer}")}
        alias_getter = (
            getattr(self.backend, "citation_aliases", None)
            if self._backend_capabilities.citation_aliases
            else None
        )
        aliases = alias_getter() if callable(alias_getter) else {}
        if isinstance(aliases, Mapping):
            for alias, pointer in aliases.items():
                if str(pointer) in pointers:
                    tokens.add(str(alias))
                    tokens.add(f"ev:{alias}")
        for evidence_id, pointer in self._state_citation_aliases().items():
            if str(pointer) in pointers:
                tokens.add(str(evidence_id))
        return tokens

    def _active_state_codes(self) -> Sequence[str]:
        """Diagnosis codes a program-owned state already carries a candidate for."""

        return ()

    def _state_citation_aliases(self) -> Mapping[str, str]:
        """Identifiers a program-owned state snapshot published to the model.

        Empty for profiles without such a snapshot. A profile that shows the
        model content ids instead of pointers must declare them here, or the
        deterministic sanitizer reads a citation the model copied straight out
        of the supplied state as citing nothing.
        """

        return {}

    def _sanitize_phase_response(
        self,
        spec: PhaseSpec,
        text: str,
        *,
        calls_at_start: int,
    ) -> tuple[str, list[str]]:
        sanitizer = getattr(self.prompt_module, "sanitize_phase_state", None)
        if not callable(sanitizer) or spec.structured or spec.response_schema is None:
            return text, []
        try:
            parsed = json.loads(str(text or ""))
        except (json.JSONDecodeError, TypeError):
            return text, []
        if not isinstance(parsed, Mapping):
            return text, []
        sanitizer_kwargs: dict[str, Any] = {
            "phase_citations": tuple(self._phase_citation_tokens(calls_at_start)),
        }
        if _accepts_keyword(sanitizer, "active_diagnosis_codes"):
            sanitizer_kwargs["active_diagnosis_codes"] = tuple(
                self._active_state_codes()
            )
        if _accepts_keyword(sanitizer, "expected_base_version"):
            latest_state = getattr(self, "_latest_phase_state", None)
            expected_version = (
                0
                if spec.key == "survey"
                else latest_state.get("version")
                if isinstance(latest_state, Mapping)
                else None
            )
            sanitizer_kwargs["expected_base_version"] = expected_version
        sanitized, notes = sanitizer(spec.key, parsed, **sanitizer_kwargs)
        if not isinstance(sanitized, Mapping):
            return text, []
        rendered = json.dumps(
            sanitized,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return rendered, [str(note) for note in (notes or [])]

    def _authorize_outcomes(
        self,
        outcomes: Sequence[ToolOutcome],
        messages: Sequence[Mapping[str, Any]],
        *,
        source: str,
    ) -> None:
        """Commit only citations present in the exact next-model context."""

        by_call = model_visible_outcome_citations(self.backend, outcomes, messages)
        for outcome in outcomes:
            if outcome.is_error:
                continue
            shown = by_call.get(outcome.call_id, ())
            accepted = self.registry.authorize_model_visible(
                tool=outcome.name,
                arguments=dict(outcome.arguments),
                citations=shown,
                source=f"{source}:{outcome.call_id}",
            )
            hidden = max(0, len(outcome.citations) - len(accepted))
            if hidden:
                self._emit(
                    f"  -> {outcome.name}: {len(accepted)} visible citation(s), "
                    f"{hidden} omitted before model context"
                )

    # -- phases -------------------------------------------------------------
    def _run_phase(
        self,
        spec: PhaseSpec,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> PhaseRecord:
        record = PhaseRecord(
            key=spec.key,
            tool_budget=spec.tool_budget,
            exposed_tools=list(spec.allowed_tools or ()),
        )
        started = time.perf_counter()
        self._emit(f"[{spec.key}] starting (tool budget {spec.tool_budget})")

        calls_at_start = len(self.registry.calls)
        self.registry.begin_phase(spec.key, spec.tool_budget)
        phase_message_start = len(messages)

        # Phase instructions may contain verifier feedback copied from an
        # evidence caveat, including literal braces such as
        # ``{aVF=qt_outside_path_bounds}``. ``str.format`` would treat those as
        # fields on a revision pass and raise KeyError. Only the one supported
        # placeholder is substituted.
        instruction = spec.instruction.replace("{budget}", str(spec.tool_budget))
        messages.append(self.backend.user_turn(instruction))
        # A phase with no tool budget gets no tool schemas: an unusable tool in
        # context is an invitation to try it and burn a turn on the refusal.
        if spec.tool_budget <= 0:
            phase_tools = []
        elif spec.allowed_tools is None:
            phase_tools = list(tools)
        else:
            allowed = set(spec.allowed_tools)
            phase_tools = [
                tool for tool in tools if str(tool.get("name") or "") in allowed
            ]
        record.exposed_tools = [
            str(tool.get("name") or "")
            for tool in phase_tools
            if str(tool.get("name") or "")
        ]

        # The diagnostic Survey has a small invariant baseline packet. On the
        # local text protocol, dispatching it directly avoids several expensive
        # generations whose only purpose was to restate predetermined function
        # names and arguments. Provider-native protocols keep the ordinary
        # assistant/tool-call handshake.
        prefetch_terminal = False
        can_prefetch = bool(
            spec.prefetch_tool_calls
            and self._backend_capabilities.orchestrated_prefetch
            and self._backend_capabilities.enforce_phase_coverage
        )
        if can_prefetch:
            prefetched: list[ToolOutcome] = []
            successful_prefetches = 0
            program_only_signatures = {
                self.registry.call_signature(name, dict(argument_items))
                for name, argument_items in spec.program_only_prefetch_tool_calls
            }
            for index, (name, argument_items) in enumerate(
                spec.prefetch_tool_calls,
                start=1,
            ):
                arguments = dict(argument_items)
                result = self.registry.call(name, arguments)
                successful_prefetches += int(result.ok)
                program_only = self.registry.call_signature(
                    name, arguments
                ) in program_only_signatures
                model_view = (
                    None
                    if program_only
                    else self.registry.model_evidence_view(
                        result,
                        tool=name,
                        arguments=arguments,
                    )
                )
                if not program_only:
                    prefetched.append(
                        ToolOutcome(
                            call_id=f"prefetch-{spec.key}-{index}",
                            text=result.render(),
                            is_error=not result.ok,
                            name=name,
                            citations=tuple(result.citations),
                            model_text=(
                                model_view.text if model_view is not None else None
                            ),
                            model_citations=(
                                model_view.citations if model_view is not None else ()
                            ),
                            model_candidate_citations=(
                                model_view.candidate_citations
                                if model_view is not None
                                else ()
                            ),
                            model_omitted_count=(
                                model_view.omitted_count
                                if model_view is not None
                                else 0
                            ),
                            arguments=arguments,
                        )
                    )
                self._emit(
                    f"  -> {name}({_brief_args(arguments)}) "
                    f"ok={result.ok} "
                    f"[{'program-only prefetch' if program_only else 'prefetch'}]"
                )
            record.prefetched_tool_calls = successful_prefetches
            prefetch_result_turns = (
                self.backend.tool_result_turn(prefetched) if prefetched else []
            )
            messages.extend(prefetch_result_turns)
            authorization_messages: Sequence[Mapping[str, Any]] = (
                prefetch_result_turns
            )
            incremental_compactor = (
                getattr(self.backend, "compact_phase_context", None)
                if self._backend_capabilities.context_compaction
                else None
            )
            if callable(incremental_compactor):
                transient = messages[phase_message_start:]
                before_chars = len(
                    json.dumps(
                        transient,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                )
                compacted = incremental_compactor(spec.key, transient)
                compacted = (
                    [compacted]
                    if isinstance(compacted, Mapping)
                    else list(compacted or [])
                )
                messages[:] = [
                    *messages[:phase_message_start],
                    *compacted,
                ]
                authorization_messages = compacted
                after_chars = len(
                    json.dumps(
                        compacted,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                )
                self._context_compactions.append(
                    {
                        "phase": spec.key,
                        "kind": "orchestrated_prefetch",
                        "before_chars": before_chars,
                        "after_chars": after_chars,
                        "saved_chars": max(0, before_chars - after_chars),
                        "tool_calls": record.prefetched_tool_calls,
                    }
                )
            self._authorize_outcomes(
                prefetched,
                authorization_messages,
                source=f"{spec.key}:prefetch",
            )
            coverage_gap = self._phase_completion_feedback(spec, calls_at_start)
            # The packet used known, program-owned arguments. A legitimate
            # unavailable result is itself a Survey limitation; asking the
            # model to retry that same mandatory view cannot create evidence.
            # Close Survey tools even when one packet view was unavailable.
            phase_tools = []
            record.exposed_tools = []
            prefetch_terminal = True
            messages.append(
                self.backend.user_turn(
                    (
                        "BASELINE INTAKE PREFETCH COMPLETE. The neutral overview "
                        "has already been attempted and its Qn facts are visible. "
                        "Treat unavailable values as limitations. Return the "
                        "compact candidate/check plan now; targeted modalities "
                        "will be executed in the next stage."
                        if spec.key == "plan"
                        else
                        "BASELINE SURVEY PREFETCH COMPLETE. The required neutral "
                        "measurement packet has been attempted and is already in "
                        "the evidence ledger. Treat any unavailable view as a "
                        "limitation. Do not request more tools in Survey; return "
                        "the compact systematic Survey state now. Targeted "
                        "modalities remain available after provisional hypotheses "
                        "are formed."
                        if spec.key == "survey"
                        else
                        "TARGETED EVIDENCE PREFETCH COMPLETE. The orchestrator "
                        "has attempted every bounded view selected for this phase "
                        "with validated arguments. Treat unavailable values as "
                        "limitations. No more tools are available; adjudicate the "
                        "validated candidate plan from the visible Qn packets and "
                        "return the requested JSON now."
                    )
                )
            )
            self._emit(
                f"[{spec.key}] baseline prefetch complete; finishing in one "
                "model turn"
                + (" with unavailable view(s)" if coverage_gap else "")
            )

        response: LLMResponse | None = None
        tool_loop_fused = False
        phase_generation_max_tokens = spec.max_tokens
        hard_guards = self._backend_capabilities.hard_phase_guards
        max_tools_per_turn = self._backend_capabilities.max_parallel_tool_calls
        turn_limit = _max_turns(
            spec.tool_budget,
            max_tools_per_turn=max_tools_per_turn,
            phase_state_retries=(self.phase_state_retries if hard_guards else 0),
        )
        for _turn in range(turn_limit):
            response_schema = (
                spec.response_schema
                if spec.response_schema is not None
                else self.prompt_module.OUTPUT_SCHEMA
                if spec.structured
                else None
            )
            coverage_before_turn = (
                self._phase_completion_feedback(spec, calls_at_start)
                if phase_tools
                else None
            )
            require_tool_call = bool(coverage_before_turn and phase_tools)
            response = self._complete_turn(
                system=self.prompt_module.SYSTEM_PROMPT,
                messages=messages,
                tools=phase_tools,
                max_tokens=phase_generation_max_tokens,
                response_schema=response_schema,
                require_tool_call=require_tool_call,
            )
            record.turns += 1
            record.stop_reason = response.stop_reason

            if response.refused:
                record.text = response.refusal or "refused"
                break

            messages.append(self.backend.assistant_turn(response))
            if response.wants_tools and prefetch_terminal:
                # Program-prefetched phases deliberately close tool schemas.
                # If a provider nevertheless emits a remembered tool call, do
                # not execute or count it a second time; ask only for the
                # terminal structured decision.
                record.text = response.text
                messages.append(
                    self.backend.user_turn(
                        "TOOLS ARE CLOSED. The bounded program prefetch is "
                        "complete; do not request or repeat a tool. Return the "
                        "requested terminal JSON from the visible Qn evidence."
                    )
                )
                self._emit(
                    f"[{spec.key}] ignored provider tool call after terminal "
                    "prefetch; requesting JSON only"
                )
                continue
            if not response.wants_tools:
                coverage_feedback = (
                    None
                    if tool_loop_fused or prefetch_terminal
                    else self._phase_completion_feedback(
                        spec,
                        calls_at_start,
                    )
                )
                if coverage_feedback and _turn + 1 < turn_limit:
                    record.text = response.text
                    messages.append(self.backend.user_turn(coverage_feedback))
                    self._emit(f"[{spec.key}] continuing: minimum evidence coverage")
                    continue
                sanitized_text, sanitizations = self._sanitize_phase_response(
                    spec,
                    response.text,
                    calls_at_start=calls_at_start,
                )
                record.text = sanitized_text
                record.phase_sanitizations.extend(sanitizations)
                if sanitizations:
                    messages.append(
                        self.backend.user_turn(
                            "DETERMINISTIC FAIL-CLOSED PHASE SANITIZATION APPLIED. "
                            "Copy-forward/no-op mutations were removed and new "
                            "identities were kept at raised status. Do not restore "
                            "a removed mutation without newly read patient evidence:\n- "
                            + "\n- ".join(sanitizations[:6])
                        )
                    )
                    self._emit(
                        f"[{spec.key}] sanitized {len(sanitizations)} "
                        "fail-closed state projection(s) without model repair"
                    )
                guard_problems = self._phase_state_problems(
                    spec,
                    record.text,
                    calls_at_start=calls_at_start,
                    stop_reason=response.stop_reason,
                )
                if guard_problems is not None:
                    if guard_problems:
                        record.phase_guard_attempts += 1
                        record.phase_guard_problems = list(guard_problems)
                        if (
                            record.phase_guard_attempts <= self.phase_state_retries
                            and _turn + 1 < turn_limit
                        ):
                            # Evidence coverage is already complete. Close tools
                            # and repair only the compact state transition.
                            phase_tools = []
                            # A response that ended exactly at max_tokens needs
                            # room to close the complete guided JSON object on
                            # its one permitted repair.  Repeating the same cap
                            # simply reproduces the truncation and wastes a
                            # full model turn.  Keep the expansion bounded.
                            if response.stop_reason == "max_tokens":
                                phase_generation_max_tokens = min(
                                    max(
                                        int(phase_generation_max_tokens or 1) * 2,
                                        int(spec.max_tokens or 1),
                                    ),
                                    6144,
                                )
                            feedback = (
                                f"HARD PHASE STATE GUARD for `{spec.key}` failed. "
                                "Do not gather more evidence and do not copy the "
                                "previous state. Return one corrected phase-state "
                                "JSON object matching the supplied schema:\n- "
                                + "\n- ".join(guard_problems[:8])
                            )
                            messages.append(self.backend.user_turn(feedback))
                            self._emit(
                                f"[{spec.key}] hard state repair "
                                f"{record.phase_guard_attempts}/{self.phase_state_retries}"
                            )
                            continue
                        # The repair budget is spent. Only defects the program
                        # cannot resolve on its own justify discarding the run.
                        fatal = self._fatal_phase_problems(guard_problems)
                        if fatal:
                            record.phase_guard_problems = list(fatal)
                            record.phase_guard_passed = False
                            record.stop_reason = "phase_guard_failed"
                        else:
                            record.phase_guard_passed = True
                            self._emit(
                                f"[{spec.key}] {len(guard_problems)} guard "
                                "problem(s) left to deterministic remediation"
                            )
                    else:
                        record.phase_guard_problems = []
                        record.phase_guard_passed = True
                break

            outcomes = self._dispatch(
                response,
                record,
                allowed_tools=spec.allowed_tools,
            )
            outcome_result_turns = self.backend.tool_result_turn(outcomes)
            messages.extend(outcome_result_turns)
            authorization_messages = outcome_result_turns
            # Local text-only tool protocols can safely replace the growing
            # assistant/result transcript with a single bounded evidence
            # ledger after every batch.  Provider-native backends omit this
            # optional hook because their call-id histories must round-trip.
            incremental_compactor = (
                getattr(self.backend, "compact_phase_context", None)
                if self._backend_capabilities.context_compaction
                else None
            )
            if callable(incremental_compactor):
                transient = messages[phase_message_start:]
                before_chars = len(
                    json.dumps(
                        transient,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                )
                compacted = incremental_compactor(spec.key, transient)
                compacted = (
                    [compacted]
                    if isinstance(compacted, Mapping)
                    else list(compacted or [])
                )
                messages[:] = [
                    *messages[:phase_message_start],
                    *compacted,
                ]
                authorization_messages = compacted
                after_chars = len(
                    json.dumps(
                        compacted,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                )
                self._context_compactions.append(
                    {
                        "phase": spec.key,
                        "kind": "incremental_tool_batch",
                        "before_chars": before_chars,
                        "after_chars": after_chars,
                        "saved_chars": max(0, before_chars - after_chars),
                    }
                )
            self._authorize_outcomes(
                outcomes,
                authorization_messages,
                source=f"{spec.key}:tool_result",
            )
            duplicate_only_batch = bool(outcomes) and all(
                outcome.is_error
                and "duplicate call suppressed" in str(outcome.text or "")
                for outcome in outcomes
            )
            if duplicate_only_batch:
                record.duplicate_batches += 1
                record.tool_loop_fused = True
                tool_loop_fused = True
                phase_tools = []
                messages.append(
                    self.backend.user_turn(
                        "DUPLICATE TOOL LOOP STOPPED. This entire batch repeated "
                        "evidence views already read in the current phase, so no "
                        "new patient evidence was added. Tool access is now closed "
                        "for this phase. Finish the structured phase state from "
                        "the existing evidence ledger; preserve unresolved items "
                        "instead of requesting another tool."
                    )
                )
                self._emit(
                    f"[{spec.key}] duplicate-only tool batch fused; finishing phase"
                )
            # Once the successful-call budget is exhausted, remove tool schemas
            # for the next turn. Otherwise local models can spend several turns
            # requesting the same tools and receiving only budget errors.
            if (
                spec.tool_budget > 0
                and self.registry.phase_successes >= spec.tool_budget
            ):
                phase_tools = []
                messages.append(
                    self.backend.user_turn(
                        "TOOL BUDGET COMPLETE. Do not request another tool. "
                        "Finish this phase now from the evidence already read, "
                        "preserving candidate and reliability limitations."
                    )
                )
        else:
            record.text = (response.text if response else "") or ""
            record.stop_reason = "turn_limit"
            if hard_guards and spec.response_schema is not None and not spec.structured:
                record.phase_guard_passed = False
                if not record.phase_guard_problems:
                    record.phase_guard_problems = [
                        "phase reached its compact turn limit without a valid state"
                    ]

        record.tool_calls = self.registry.phase_successes
        record.rejected_calls = (
            len(self.registry.calls) - calls_at_start - record.tool_calls
        )
        record.elapsed_s = time.perf_counter() - started
        self._emit(
            f"[{spec.key}] done: {record.turns} turn(s), {record.tool_calls} tool call(s)"
            + (f", {record.rejected_calls} rejected" if record.rejected_calls else "")
        )
        canonical_phase_text = record.text
        if record.phase_guard_passed is not False and spec.response_schema is not None:
            canonical_phase_text = self._canonicalize_phase_state(
                spec,
                record,
                calls_at_start=calls_at_start,
            )
        if record.phase_guard_passed is not False and spec.response_schema is not None:
            if not spec.structured:
                self._last_phase_state_text = canonical_phase_text
        # Replace transient call envelopes with one rolling evidence ledger.
        # Unlike the original aggressive compaction, the local backend now
        # retains all ten domain summaries plus selected compressed raw output
        # from decisive tools. Older rolling-memory turns are replaced rather
        # than accumulated, so richer evidence does not create phase-by-phase
        # duplication.
        memory_builder = (
            getattr(self.backend, "phase_memory_turn", None)
            if self._backend_capabilities.phase_memory
            else None
        )
        if not spec.structured and callable(memory_builder):
            transient = messages[phase_message_start:]
            memory_predicate = getattr(
                self.backend,
                "is_phase_memory_turn",
                None,
            )
            prefix_messages = messages[:phase_message_start]
            previous_memory = (
                [
                    message
                    for message in prefix_messages
                    if memory_predicate(message)
                ]
                if callable(memory_predicate)
                else []
            )
            retained_prefix = (
                [
                    message
                    for message in prefix_messages
                    if not memory_predicate(message)
                ]
                if callable(memory_predicate)
                else prefix_messages
            )
            replaced_messages = [*previous_memory, *transient]
            before_chars = len(
                json.dumps(
                    replaced_messages,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            memory_turns = memory_builder(
                spec.key,
                canonical_phase_text,
                transient,
            )
            if isinstance(memory_turns, dict):
                memory_turns = [memory_turns]
            memory_turns = list(memory_turns or [])
            messages[:] = [*retained_prefix, *memory_turns]
            after_chars = len(
                json.dumps(
                    memory_turns,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            self._context_compactions.append(
                {
                    "phase": spec.key,
                    "kind": "phase_boundary",
                    "before_chars": before_chars,
                    "after_chars": after_chars,
                    "saved_chars": max(0, before_chars - after_chars),
                }
            )
        return record

    def _phase_completion_feedback(
        self,
        spec: PhaseSpec,
        calls_at_start: int,
    ) -> str | None:
        """Prevent the local model from ending evidence phases prematurely."""

        if not self._backend_capabilities.enforce_phase_coverage:
            return None
        if spec.tool_budget <= 0:
            return None
        successful_calls = [
            call
            for call in self.registry.calls[calls_at_start:]
            if call.ok
        ]
        phase_calls: list[Any] = []
        seen_signatures: set[str] = set()
        for call in successful_calls:
            signature = self.registry.call_signature(call.tool, call.args)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            phase_calls.append(call)
        if len(phase_calls) >= spec.tool_budget:
            return None
        eligible_calls = phase_calls
        if spec.require_novel_tool_views:
            prior_signatures = {
                self.registry.call_signature(call.tool, call.args)
                for call in self.registry.calls[:calls_at_start]
                if call.ok
            }
            eligible_calls = [
                call
                for call in phase_calls
                if self.registry.call_signature(call.tool, call.args)
                not in prior_signatures
            ]
        counts: dict[str, int] = {}
        for call in phase_calls:
            counts[call.tool] = counts.get(call.tool, 0) + 1
        missing = [
            f"{name} x{required - counts.get(name, 0)}"
            for name, required in spec.required_tool_counts
            if counts.get(name, 0) < required
        ]
        for name, required_arguments, required in spec.required_tool_calls:
            expected = dict(required_arguments)
            matched = sum(
                1
                for call in phase_calls
                if call.tool == name
                and all(call.args.get(key) == value for key, value in expected.items())
            )
            if matched >= required:
                continue
            rendered_arguments = ", ".join(
                f"{key}={value!r}" for key, value in expected.items()
            )
            label = f"{name}({rendered_arguments})" if rendered_arguments else name
            missing.append(f"{label} x{required - matched}")
        missing_coverage: list[str] = []
        for coverage_id, tool_names, required in spec.coverage_requirements:
            allowed = set(tool_names)
            matched = sum(1 for call in eligible_calls if call.tool in allowed)
            if matched < required:
                suffix = " novel" if spec.require_novel_tool_views else ""
                missing_coverage.append(
                    f"{coverage_id} needs {required - matched} more{suffix} "
                    f"view(s) from [{', '.join(tool_names)}]"
                )
        remaining_to_minimum = max(
            0,
            int(spec.minimum_tool_calls) - len(phase_calls),
        )
        if not missing and not missing_coverage and remaining_to_minimum <= 0:
            return None
        requirements = []
        if missing:
            requirements.append("missing required calls: " + ", ".join(missing))
        if missing_coverage:
            requirements.append(
                "unresolved evidence coverage: " + "; ".join(missing_coverage)
            )
        if remaining_to_minimum:
            requirements.append(
                f"make at least {remaining_to_minimum} additional distinct, "
                "clinically relevant measurement view(s); exact repeats do "
                "not count"
            )
        return (
            "EVIDENCE COVERAGE IS NOT YET MET. "
            + "; ".join(requirements)
            + ". Continue using tools before finishing this phase. Complete "
            "the ten-domain memory only after these measurements have been "
            "reviewed; this is a coverage requirement, not a diagnostic rule."
        )

    def _dispatch(
        self,
        response: LLMResponse,
        record: PhaseRecord,
        *,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> list[ToolOutcome]:
        outcomes: list[ToolOutcome] = []
        allowed = set(allowed_tools) if allowed_tools is not None else None
        parallel_limit = max(
            1,
            int(self._backend_capabilities.max_parallel_tool_calls or 1),
        )
        for call_index, call in enumerate(response.tool_calls):
            if call_index >= parallel_limit:
                message = (
                    "parallel tool-call limit reached for this model turn "
                    f"({parallel_limit}); request this still-needed view in the "
                    "next turn instead of repeating completed calls"
                )
                self._emit(
                    f"  -> {call.name} deferred by parallel-call limit "
                    f"{parallel_limit}"
                )
                outcomes.append(
                    ToolOutcome(
                        call_id=call.id,
                        text=message,
                        is_error=True,
                        name=call.name,
                        arguments=dict(call.arguments),
                    )
                )
                continue
            if allowed is not None and call.name not in allowed:
                message = (
                    f"tool {call.name!r} is not available in this phase. "
                    f"Available: {', '.join(sorted(allowed)) or '(none)'}"
                )
                self._emit(f"  -> {call.name} rejected by phase tool policy")
                outcomes.append(
                    ToolOutcome(
                        call_id=call.id,
                        text=message,
                        is_error=True,
                        name=call.name,
                        arguments=dict(call.arguments),
                    )
                )
                continue
            result = self.registry.call(call.name, call.arguments)
            self._emit(f"  -> {call.name}({_brief_args(call.arguments)}) ok={result.ok}")
            model_view = self.registry.model_evidence_view(
                result,
                tool=call.name,
                arguments=dict(call.arguments),
            )
            outcomes.append(
                ToolOutcome(
                    call_id=call.id,
                    text=result.render(),
                    is_error=not result.ok,
                    name=call.name,
                    citations=tuple(result.citations),
                    model_text=(model_view.text if model_view is not None else None),
                    model_citations=(
                        model_view.citations if model_view is not None else ()
                    ),
                    model_candidate_citations=(
                        model_view.candidate_citations
                        if model_view is not None
                        else ()
                    ),
                    model_omitted_count=(
                        model_view.omitted_count if model_view is not None else 0
                    ),
                    arguments=dict(call.arguments),
                )
            )
        return outcomes

    def _revise(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        feedback: str,
        current_candidate: Mapping[str, Any] | None = None,
        *,
        measurement_reread: bool,
    ) -> PhaseRecord:
        revision_compactor = (
            getattr(self.backend, "compact_revision_context", None)
            if self._backend_capabilities.context_compaction
            else None
        )
        if callable(revision_compactor):
            before_chars = len(
                json.dumps(
                    messages,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            compacted = revision_compactor(messages, current_candidate)
            compacted = list(compacted or [])
            messages[:] = compacted
            after_chars = len(
                json.dumps(
                    messages,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            self._context_compactions.append(
                {
                    "phase": "revise",
                    "kind": "revision_history",
                    "before_chars": before_chars,
                    "after_chars": after_chars,
                    "saved_chars": max(0, before_chars - after_chars),
                }
            )
        revision_tools = (
            self._revision_allowed_tool_names(feedback)
            if measurement_reread
            else ()
        )
        spec = PhaseSpec(
            key="revise",
            instruction=self._revision_instruction(
                feedback,
                current_candidate=current_candidate,
            ),
            # Evidence revisions may re-read a value, but a purely structural
            # or caveat repair must not manufacture a redundant tool call.
            tool_budget=3 if measurement_reread else 0,
            structured=True,
            response_schema=self._revision_response_schema(
                current_candidate=current_candidate,
            ),
            max_tokens=6144,
            allowed_tools=revision_tools,
            minimum_tool_calls=0,
        )
        return self._run_phase(spec, messages, tools)

    def _revision_instruction(
        self,
        feedback: str,
        *,
        current_candidate: Mapping[str, Any] | None,
    ) -> str:
        """Return the model instruction for one bounded verdict repair."""

        return self.prompt_module.REVISION_INSTRUCTION.format(feedback=feedback)

    def _revision_response_schema(
        self,
        *,
        current_candidate: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Optional profile-specific repair schema.

        A ``None`` schema retains the historical complete-verdict repair.
        Diagnosis-first profiles use a section patch when a candidate exists.
        """

        return None

    def _request_revision(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        feedback: str,
        result: AgentResult,
        *,
        reason: str,
        current_candidate: Mapping[str, Any] | None = None,
        measurement_reread: bool,
    ) -> dict[str, Any] | None:
        """Use the bounded revision allowance until one reply parses as JSON."""
        current_feedback = feedback
        while result.revisions < self.max_revisions:
            result.revisions += 1
            self._emit(
                f"[revise] {reason}; "
                f"revision {result.revisions}/{self.max_revisions}"
            )
            revised = self._revise(
                messages,
                tools,
                current_feedback,
                current_candidate=current_candidate,
                measurement_reread=measurement_reread,
            )
            result.phases.append(revised)
            self._checkpoint(
                result,
                status=f"revision_completed:{result.revisions}",
            )
            parsed = self._parse_verdict(revised.text)
            if parsed is not None:
                return self._normalize_verdict(parsed)
            current_feedback = (
                "The previous revision was not a parseable JSON object. "
                "Return the complete verdict as one JSON object only, matching "
                "the schema; do not explain the error or discuss the feedback."
            )
            reason = "previous revision was not parseable JSON"
        return None

    def _normalize_verdict(self, verdict: dict[str, Any]) -> dict[str, Any]:
        """Apply provider-local lossless normalization before validation.

        The local backend uses short citation aliases in model context to
        avoid repeating long JSON Pointers thousands of times.  Expansion is
        deterministic and happens before the output contract or evidence
        verifier sees the verdict.  Provider backends without a normalizer
        keep the original object unchanged.
        """
        normalizer = getattr(self.backend, "normalize_verdict", None)
        if not callable(normalizer):
            normalized = verdict
        else:
            backend_normalized = normalizer(verdict)
            normalized = (
                backend_normalized
                if isinstance(backend_normalized, dict)
                else verdict
            )

        # The model selects evidence and explains its relevance, but it is not
        # trusted to transcribe measurements. Resolve any unambiguous numeric
        # evidence item from the immutable store before presentation cleanup
        # and verification. This prevents a correct pointer paired with a
        # hallucinated or stale copied value from leaking into the result.
        normalized = self._materialize_evidence_values(normalized)

        presentation_normalizer = getattr(
            self.prompt_module,
            "normalize_verdict",
            None,
        )
        if not callable(presentation_normalizer):
            return self._qualify_caveated_evidence_items(normalized)
        # Checked by signature rather than by catching TypeError, which would
        # also swallow one raised inside the normalizer and silently fall back
        # to the ungrounded path.
        if _accepts_keyword(presentation_normalizer, "resolve_cited_value"):
            presentation_result = presentation_normalizer(
                normalized,
                resolve_cited_value=self._cited_display_value,
            )
        else:
            presentation_result = presentation_normalizer(normalized)
        if (
            isinstance(presentation_result, tuple)
            and len(presentation_result) == 2
            and isinstance(presentation_result[0], dict)
        ):
            normalized, paths = presentation_result
            changed_paths = [
                str(path) for path in paths
            ] if isinstance(paths, (list, tuple)) else []
            if changed_paths:
                self._verdict_normalizations.append(
                    {
                        "kind": "narrative_quantity_placement",
                        "paths": changed_paths,
                    }
                )
            return self._qualify_caveated_evidence_items(normalized)
        normalized = (
            presentation_result
            if isinstance(presentation_result, dict)
            else normalized
        )
        return self._qualify_caveated_evidence_items(normalized)

    def _cited_display_value(
        self,
        token: str,
    ) -> tuple[float, str | None] | None:
        """Resolve one citation to the number the model was shown for it.

        Presentation cleanup needs the *store* to decide whether a number in a
        claim is grounded. Judging that against the item's single `value` field
        instead makes the schema, not the evidence, the arbiter: a claim citing
        both `r_amp_mv` and `s_amp_mv` had its second number scrubbed to
        "corresponding measured value" even though the pointer for it was right there in the same
        item. Comparative morphology reads as exactly that kind of claim.
        """
        evidence = self.store.try_resolve(str(token))
        if evidence is None or evidence.pointer not in self.registry.whitelist:
            return None
        value = evidence.display_value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return float(value), evidence.unit

    @staticmethod
    def _verdict_evidence_items(
        verdict: Mapping[str, Any] | None,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Return final evidence items with their semantic use."""

        if not isinstance(verdict, Mapping):
            return []
        found: list[tuple[str, str, str, dict[str, Any]]] = []

        for diagnosis_index, diagnosis in enumerate(verdict.get("diagnoses") or []):
            if not isinstance(diagnosis, Mapping):
                continue
            subject = str(
                diagnosis.get("code") or diagnosis.get("statement") or "diagnosis"
            )
            for key, role in (("evidence", "supporting"), ("counterevidence", "opposing")):
                for item_index, item in enumerate(diagnosis.get(key) or []):
                    if isinstance(item, dict):
                        found.append(
                            (
                                f"diagnoses[{diagnosis_index}].{key}[{item_index}]",
                                role,
                                subject,
                                item,
                            )
                        )

        for differential_index, differential in enumerate(
            verdict.get("differential_diagnoses") or []
        ):
            if not isinstance(differential, Mapping):
                continue
            subject = str(
                differential.get("code")
                or differential.get("statement")
                or "differential"
            )
            for key, role in (
                ("supporting_evidence", "differential_support"),
                ("counterevidence", "differential_opposition"),
            ):
                for item_index, item in enumerate(differential.get(key) or []):
                    if isinstance(item, dict):
                        found.append(
                            (
                                "differential_diagnoses"
                                f"[{differential_index}].{key}[{item_index}]",
                                role,
                                subject,
                                item,
                            )
                        )

        for context_index, context in enumerate(
            verdict.get("interval_measurement_contexts") or []
        ):
            if not isinstance(context, Mapping):
                continue
            subject = str(context.get("interval") or "interval_context")
            for item_index, item in enumerate(context.get("residual_evidence") or []):
                if isinstance(item, dict):
                    found.append(
                        (
                            "interval_measurement_contexts"
                            f"[{context_index}].residual_evidence[{item_index}]",
                            "residual_context",
                            subject,
                            item,
                        )
                    )
        return found

    def _materialize_evidence_values(
        self,
        verdict: dict[str, Any],
    ) -> dict[str, Any]:
        """Fill direct numeric evidence values from authorised source pointers.

        Correcting a wrong value and un-rounding a right one look identical to
        a naive `!=` comparison, and only the first is wanted. A value the
        model transcribed at the precision the tool displayed already agrees
        with the store; replacing it with the underlying float desynchronised
        the number from the claim sentence that reports it, and verification
        then charged the model for a digit string it had never seen.
        """

        changed: list[str] = []
        for path, _role, _subject, item in self._verdict_evidence_items(verdict):
            citation_tokens = [
                str(token)
                for token in (item.get("citations") or [])
                if str(token).strip()
            ]
            # A multi-pointer item is a qualitative relation/context. Turning
            # one of its pointers into the primary scalar would violate the
            # atomic evidence contract and silently choose which fact the item
            # is "about". Numeric materialization is therefore limited to a
            # single-citation item.
            if len(citation_tokens) != 1:
                continue
            numeric_sources: dict[str, Any] = {}
            for token in citation_tokens:
                evidence = self.store.try_resolve(str(token))
                if evidence is None or evidence.pointer not in self.registry.whitelist:
                    continue
                if isinstance(evidence.value, (int, float)) and not isinstance(
                    evidence.value, bool
                ):
                    numeric_sources[evidence.pointer] = evidence
            # One direct numeric source behind that one citation is unambiguous.
            if len(numeric_sources) != 1:
                continue
            evidence = next(iter(numeric_sources.values()))
            # The immutable store retains full machine precision for audit and
            # calculation. The verdict is a clinical presentation object, so
            # materialize the same canonical precision used by tool rendering
            # (for example 50.083... bpm -> 50 bpm and -0.13296 mV ->
            # -0.133 mV). Injecting raw floats beside an already rounded claim
            # creates two textual quantities for one citation.
            precision = display_precision(evidence.unit)
            rounded = round(float(evidence.value), precision)
            display_value: int | float = (
                int(rounded) if precision == 0 else rounded
            )
            if item.get("value") != display_value:
                item["value"] = display_value
                changed.append(f"{path}.value")
            if item.get("unit") != evidence.unit:
                item["unit"] = evidence.unit
                changed.append(f"{path}.unit")

        if changed:
            self._verdict_normalizations.append(
                {
                    "kind": "deterministic_evidence_value_materialization",
                    "paths": changed,
                }
            )
        return verdict

    def _effective_evidence_snapshot(
        self,
        verdict: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Materialise only evidence actually selected by the final verdict."""

        selected: dict[str, dict[str, Any]] = {}
        for path, role, subject, item in self._verdict_evidence_items(verdict):
            use = {
                "path": path,
                "role": role,
                "subject": subject,
                "claim": str(item.get("claim") or ""),
            }
            for token in _item_citation_tokens(item):
                citation = str(token).strip()
                evidence = self.store.try_resolve(citation)
                provenance = (
                    self.registry.evidence_provenance(evidence.pointer)
                    if evidence is not None
                    else {}
                )
                authorised = bool(
                    evidence is not None
                    and evidence.pointer in self.registry.whitelist
                )
                model_visible = bool(provenance.get("model_visible"))
                program_authorized = bool(provenance.get("program_authorized"))
                key = evidence.pointer if evidence is not None else citation
                row = selected.setdefault(
                    key,
                    {
                        "pointer": evidence.pointer if evidence is not None else None,
                        "citation": (
                            evidence.citation if evidence is not None else citation
                        ),
                        "status": (
                            "resolved_model_visible"
                            if model_visible
                            else "resolved_program_deterministic"
                            if program_authorized
                            else "not_model_visible"
                            if evidence is not None
                            else "unresolvable"
                        ),
                        "model_visible": model_visible,
                        "program_authorized": program_authorized,
                        "program_sources": list(
                            provenance.get("program_sources") or []
                        ),
                        # Never turn a guessed but resolvable pointer into new
                        # evidence during serialization. Values are materialized
                        # only when the original value-bearing row was visible.
                        "value": evidence.value if authorised else None,
                        "unit": evidence.unit if authorised else None,
                        "reliability": (
                            "unavailable"
                            if authorised and evidence.is_null
                            else "limited"
                            if authorised and evidence.caveats
                            else "reliable"
                            if authorised
                            else "unverified"
                        ),
                        "caveats": (
                            list(evidence.caveats) if authorised else []
                        ),
                        "source": evidence.source if authorised else None,
                        "uses": [],
                    },
                )
                if use not in row["uses"]:
                    row["uses"].append(use)

        rows = list(selected.values())
        resolved_pointers = [
            str(row["pointer"])
            for row in rows
            if row.get("pointer") is not None
        ]
        return {
            "selection": "final_verdict_citations_only",
            "count": len(rows),
            "evidence_set_id": evidence_set_id(resolved_pointers),
            "items": rows,
        }

    def _tools_audit(
        self,
        verdict: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build compact tool audit plus human-meaningful final evidence."""

        audit = self.registry.audit()
        audit["effective_evidence"] = self._effective_evidence_snapshot(verdict)
        return audit

    def _qualify_caveated_evidence_items(
        self,
        verdict: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach reliability language to the exact claim that cites it.

        Reliability is a property of an ecgfeat measurement, not of the report
        paragraph around it.  The verifier deliberately checks each structured
        evidence item in isolation.  Adding this conservative qualifier is a
        lossless deterministic operation: the observation and citation remain,
        while the claim can no longer present a flagged value as clean fact.
        """

        item_groups: list[tuple[str, Any]] = []
        for diagnosis_index, diagnosis in enumerate(verdict.get("diagnoses") or []):
            if not isinstance(diagnosis, Mapping):
                continue
            for key in ("evidence", "counterevidence"):
                item_groups.append(
                    (f"diagnoses[{diagnosis_index}].{key}", diagnosis.get(key))
                )
        for differential_index, differential in enumerate(
            verdict.get("differential_diagnoses") or []
        ):
            if not isinstance(differential, Mapping):
                continue
            for key in ("supporting_evidence", "counterevidence"):
                item_groups.append(
                    (
                        f"differential_diagnoses[{differential_index}].{key}",
                        differential.get(key),
                    )
                )
        for context_index, context in enumerate(
            verdict.get("interval_measurement_contexts") or []
        ):
            if isinstance(context, Mapping):
                item_groups.append(
                    (
                        f"interval_measurement_contexts[{context_index}].residual_evidence",
                        context.get("residual_evidence"),
                    )
                )

        changed_paths: list[str] = []
        for group_path, items in item_groups:
            if not isinstance(items, list):
                continue
            for item_index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim") or "").strip()
                if not claim or claim_is_qualified(claim):
                    continue
                has_caveated_citation = False
                for token in _item_citation_tokens(item):
                    citation = str(token).strip()
                    if not citation.startswith("ev:/"):
                        continue
                    try:
                        evidence = self.store.resolve(citation[3:])
                    except (PointerError, TypeError, ValueError):
                        continue
                    if evidence.caveats:
                        has_caveated_citation = True
                        break
                if not has_caveated_citation:
                    continue
                is_chinese = any("\u4e00" <= char <= "\u9fff" for char in claim)
                qualifier = (
                    " (this measurement carries an ecgfeat reliability flag and is interpreted only as limited evidence)"
                    if is_chinese
                    else " (ecgfeat reliability-flagged; interpret as limited evidence)"
                )
                item["claim"] = claim.rstrip(".\u3002") + qualifier
                changed_paths.append(f"{group_path}[{item_index}].claim")

        if changed_paths:
            self._verdict_normalizations.append(
                {
                    "kind": "caveated_evidence_qualification",
                    "paths": changed_paths,
                }
            )
        return verdict

    # -- verification -------------------------------------------------------
    def _verify(self, verdict: dict[str, Any]) -> VerificationReport:
        return verify_structured(
            verdict,
            self.store,
            whitelist=self.registry.whitelist,
            policy=self.verify_policy,
        )

    @staticmethod
    def _parse_verdict(text: str) -> dict[str, Any] | None:
        from ..backends.anthropic_api import parse_json_text

        return parse_json_text(text) if text else None


def _brief_args(arguments: dict[str, Any]) -> str:
    rendered = json.dumps(arguments, ensure_ascii=False, default=str)
    return rendered if len(rendered) <= 70 else rendered[:67] + "..."


def run_agent(
    features_path: str,
    *,
    backend: LLMBackend,
    record_id: str | None = None,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    on_event: Any = None,
) -> AgentResult:
    """Convenience entry point: load a payload and run the loop over it."""
    store = EvidenceStore.from_path(features_path, record_id=record_id)
    return ECGAgent(
        store=store, backend=backend, max_revisions=max_revisions, on_event=on_event
    ).run()
