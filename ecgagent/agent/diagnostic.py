"""Diagnosis-first ECG agent profile.

The compact workflow combines a blind model-generated measurement plan with a
bounded ecgfeat rule second opinion.  Rules can add candidates after planning,
but only independently cited measurement tools can support a final diagnosis.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import re
import time
from typing import Any, Mapping, Sequence

from ..age import resolve_patient_age
from ..backends.base import LLMBackend
from ..evidence.diagnostic_briefing import (
    build_compact_diagnostic_briefing,
    build_diagnostic_briefing,
)
from ..evidence.store import EvidenceStore
from ..quality_gate import (
    QUALITY_GATE_VALIDATION_VERSION,
    validate_diagnostic_gate,
)
from ..tools.registry import ToolRegistry, build_default_registry
from ..verify import VerificationPolicy
from . import compact_diagnostic_prompts, diagnostic_prompts
from .diagnostic_ledger import DiagnosticLedger, LedgerPatchError
from .diagnostic_pathways import (
    build_diagnostic_pathway,
    semantic_candidate_family,
)
from .deterministic_pathways import (
    DEFAULT_DETERMINISTIC_PATHWAY_POLICY,
    deterministic_nodes_shadowed as _deterministic_nodes_shadowed,
    resolve_additional_deterministic_step,
)
from .loop import AgentResult, ECGAgent, PhaseRecord, PhaseSpec
from .protocol import (
    DEFAULT_DIAGNOSTIC_PROTOCOL,
    DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
    DIAGNOSTIC_TOOLS,
    DOMAIN_TOOL_MAP,
)
from .rule_second_opinion import (
    RULE_SECOND_OPINION_VERSION,
    build_rule_second_opinion,
    rule_candidate_blueprint,
)
from .safety_policy import DEFAULT_CLINICAL_SAFETY_POLICY
from .tool_planner import build_tool_plan
from .urgent_review import assess_urgent_review

# Backward-compatible import name.  The tuple now also contains modality tools,
# all of which still return measurement evidence rather than diagnoses.
MEASUREMENT_TOOLS = DIAGNOSTIC_TOOLS


# Survey has one provider-independent intake surface. Native tool-call
# backends therefore cannot replace the compact overview with a large table.
SURVEY_TOOLS: tuple[str, ...] = ("get_diagnostic_overview",)

_INTRAVENTRICULAR_CONDUCTION_CODES = frozenset(
    {
        "bifascicular_block_pattern",
        "incomplete_rbbb_pattern",
        "lafb_pattern",
        "lbbb_pattern",
        "lpfb_pattern",
        "nonspecific_ivcd",
        "probable_bifascicular_block_pattern",
        "probable_lafb_pattern",
        "probable_lbbb_pattern",
        "probable_nonspecific_ivcd",
        "probable_rbbb_pattern",
        "rbbb_pattern",
        "right_bundle_branch_block",
        "left_bundle_branch_block",
        "trifascicular_block_pattern",
    }
)


_REQUIRED_NODES_PREFIX = "Complete the following required nodes: "


def _fit_required_node_questions(questions: Sequence[str]) -> str:
    """Join unresolved pathway questions without overrunning the verdict guard.

    This string is copied verbatim into `key_uncertainty`, which
    ``validate_verdict_contract`` caps at ``KEY_UNCERTAINTY_MAX_CHARS``. Pathway
    questions are whole clinical sentences, so joining three of them routinely
    produced 180-290 characters and the program then failed its own record on
    text no model ever wrote -- the single largest cause of lost records.

    Whole questions are kept rather than truncated: a question cut mid-clause
    tells a reviewer less than one fewer complete question does.
    """

    budget = (
        diagnostic_prompts.KEY_UNCERTAINTY_MAX_CHARS
        - len(_REQUIRED_NODES_PREFIX)
    )
    selected: list[str] = []
    used = 0
    for question in dict.fromkeys(str(item).strip() for item in questions if item):
        # "; " between entries, so every item after the first costs two more.
        cost = len(question) + (2 if selected else 0)
        if used + cost > budget:
            continue
        selected.append(question)
        used += cost
        if len(selected) == 3:
            break
    return "; ".join(selected)


def _diagnostic_complexity_flags(store: EvidenceStore) -> tuple[str, ...]:
    """Return measurement-only flags used solely to size hard call ceilings."""

    flags: list[str] = []
    gate = str(
        store.raw("/metadata/diagnostic_gate/state", "unknown") or "unknown"
    ).lower()
    if gate != "pass":
        flags.append("quality_limited")
    group_count = store.raw("/metadata/diagnostic_gate/group_count", None)
    if isinstance(group_count, (int, float)) and int(group_count) > 1:
        flags.append("multiple_morphologies")
    if bool(store.raw("/metadata/diagnostic_gate/morphology_complex", False)):
        flags.append("morphology_complex")
    pacing_state = str(
        store.raw("/rhythm_inputs/pacing/state", "off") or "off"
    ).lower()
    if pacing_state not in {"", "off", "none", "false"}:
        flags.append("pacing_possible")
    global_features = store.document.get("global_features")
    global_features = (
        global_features if isinstance(global_features, Mapping) else {}
    )
    qt_reliability = str(
        global_features.get("qt_reliability") or "unavailable"
    ).lower()
    if (
        global_features.get("qt_ms") is None
        or global_features.get("qt_reportable") is False
        or qt_reliability
        in {"rescued", "fallback", "low_confidence", "unreliable", "unavailable"}
    ):
        flags.append("qt_limited")
    availability = store.raw("/rhythm_inputs/record/availability", {})
    if isinstance(availability, Mapping):
        if availability.get("pr_available") is False:
            flags.append("pr_limited")
        if availability.get("atrial_rhythm_available") is False:
            flags.append("atrial_activity_uncertain")
    return tuple(dict.fromkeys(flags))


def _abnormal_domain_count(store: EvidenceStore) -> int:
    """Count domains whose own measurement is abnormal or unavailable.

    Sizes the plan's candidate budget. A fixed three-candidate cap guarantees a
    miss on any record carrying more than three findings, and the PTB-XL
    reference labels routinely carry five or six; measured over four repeats the
    agent's deficit is recall by 2x, with whole families never raised at all.

    Every pointer read here is already displayed to the model by
    ``get_diagnostic_overview`` in the same phase, so sizing on them tells the
    model nothing it is not shown and keeps the plan blind. Rule-engine verdicts
    are deliberately NOT consulted: they would leak how many abnormalities exist
    into a phase whose independence is the point.
    """

    def number(pointer: str) -> float | None:
        value = store.raw(pointer, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    patient = store.raw("/metadata/patient_meta", {})
    age = resolve_patient_age(patient if isinstance(patient, Mapping) else {})
    domains = 0
    if age.adult is True:
        rate = number("/global_features/heart_rate_bpm")
        if rate is not None and not (60.0 <= rate <= 100.0):
            domains += 1
        pr = number("/global_features/pr_ms")
        if pr is None or pr > 200.0 or pr < 120.0:
            domains += 1
        qrs = number("/global_features/qrs_ms")
        if qrs is not None and qrs >= 110.0:
            domains += 1
        axis = number("/global_features/qrs_axis_deg")
        if axis is not None and not (-30.0 <= axis <= 90.0):
            domains += 1
        p_axis = number("/global_features/p_axis_deg")
        if p_axis is None or not (0.0 <= p_axis <= 90.0):
            domains += 1
    else:
        # Adult reference intervals cannot classify pediatric or unknown-age
        # records.  Count those domains as unresolved for budget sizing so the
        # model gets enough measurement checks without calling them abnormal.
        domains += 5
    if store.raw("/global_features/qt_reportable", None) is False:
        domains += 1
    if store.raw("/rhythm_inputs/background/background_rr_regular", None) is False:
        domains += 1
    if bool(store.raw("/rhythm_inputs/af_afl/af_afl_indeterminate", False)):
        domains += 1
    pacing = str(store.raw("/rhythm_inputs/pacing/state", "off") or "off").lower()
    if pacing not in {"", "off", "none", "false"}:
        domains += 1
    gate = str(store.raw("/metadata/diagnostic_gate/state", "unknown") or "unknown")
    if gate.lower() != "pass":
        domains += 1
    return domains


def plan_candidate_limit(store: EvidenceStore) -> int:
    """Per-record candidate budget for the blind plan."""

    policy = compact_diagnostic_prompts
    return max(
        policy.INDEPENDENT_PLAN_MIN_CANDIDATES,
        min(
            policy.INDEPENDENT_PLAN_MAX_CANDIDATES,
            policy.INDEPENDENT_PLAN_MIN_CANDIDATES + _abnormal_domain_count(store) // 2,
        ),
    )


def legacy_diagnostic_phases(store: EvidenceStore) -> tuple[PhaseSpec, ...]:
    """Build the v31 five-stage workflow for compatibility and comparison."""

    policy = DEFAULT_DIAGNOSTIC_PROTOCOL.phases
    complexity_flags = _diagnostic_complexity_flags(store)
    if complexity_flags:
        investigate_budget = min(
            policy.investigate_ceiling,
            policy.investigate_default + max(0, len(complexity_flags) - 2),
        )
        challenge_budget = min(
            policy.challenge_ceiling,
            policy.challenge_default + max(0, len(complexity_flags) - 3),
        )
    else:
        investigate_budget = policy.investigate_floor
        challenge_budget = policy.challenge_floor
    return (
        PhaseSpec(
            "survey",
            diagnostic_prompts.SURVEY_INSTRUCTION,
            tool_budget=policy.survey_budget,
            response_schema=diagnostic_prompts.SURVEY_STATE_SCHEMA,
            max_tokens=policy.survey_max_tokens,
            allowed_tools=SURVEY_TOOLS,
            minimum_tool_calls=1,
            required_tool_counts=(("get_diagnostic_overview", 1),),
            prefetch_tool_calls=(("get_diagnostic_overview", ()),),
        ),
        PhaseSpec(
            "hypothesize",
            diagnostic_prompts.HYPOTHESIZE_INSTRUCTION,
            tool_budget=0,
            response_schema=diagnostic_prompts.HYPOTHESIS_STATE_SCHEMA,
            # The compact hypothesis object still carries support,
            # counterevidence and a targeted plan.  Real Qwen generations for
            # difficult records require about 1.6K-2.1K tokens; 1.2K caused a
            # deterministic JSON-tail truncation and then repeated the same
            # failure during hard-state repair.
            max_tokens=policy.hypothesize_max_tokens,
            allowed_tools=(),
        ),
        PhaseSpec(
            "investigate",
            diagnostic_prompts.INVESTIGATE_INSTRUCTION,
            tool_budget=investigate_budget,
            response_schema=diagnostic_prompts.INVESTIGATION_STATE_SCHEMA,
            # Difficult records can fill four evidence-backed hypotheses even
            # without the removed ten-domain ledger. Historical Qwen outputs
            # reached about 2.6K tokens, so 2.8K prevents JSON-tail truncation.
            max_tokens=policy.investigate_max_tokens,
            allowed_tools=DIAGNOSTIC_TOOLS,
            minimum_tool_calls=0,
        ),
        PhaseSpec(
            "challenge",
            diagnostic_prompts.CHALLENGE_INSTRUCTION,
            tool_budget=challenge_budget,
            response_schema=diagnostic_prompts.CHALLENGE_STATE_SCHEMA,
            max_tokens=policy.challenge_max_tokens,
            allowed_tools=DIAGNOSTIC_TOOLS,
            minimum_tool_calls=0,
            require_novel_tool_views=True,
        ),
        PhaseSpec(
            "synthesize",
            diagnostic_prompts.SYNTHESIZE_INSTRUCTION,
            tool_budget=0,
            structured=True,
            max_tokens=policy.synthesize_max_tokens,
            allowed_tools=(),
        ),
    )


def compact_diagnostic_phases(store: EvidenceStore) -> tuple[PhaseSpec, ...]:
    """Build the bounded small-model plan/adjudicate workflow.

    The diagnosis-neutral overview is program-prefetched inside ``plan``, so
    the usual path has two model decisions rather than five state-producing
    phases. Targeted views are also prefetched before the final decision.
    """

    policy = DEFAULT_DIAGNOSTIC_PROTOCOL.phases
    candidate_limit = plan_candidate_limit(store)
    return (
        PhaseSpec(
            "plan",
            compact_diagnostic_prompts.PLAN_INSTRUCTION.format(
                candidate_limit=candidate_limit
            ),
            tool_budget=1,
            response_schema=compact_diagnostic_prompts.PLAN_SCHEMA,
            structured=True,
            max_tokens=policy.compact_plan_max_tokens,
            allowed_tools=SURVEY_TOOLS,
            minimum_tool_calls=1,
            required_tool_counts=(("get_diagnostic_overview", 1),),
            prefetch_tool_calls=(("get_diagnostic_overview", ()),),
        ),
        PhaseSpec(
            "adjudicate",
            compact_diagnostic_prompts.ADJUDICATE_INSTRUCTION,
            tool_budget=policy.compact_tool_ceiling,
            response_schema=compact_diagnostic_prompts.COMPACT_VERDICT_SCHEMA,
            structured=True,
            max_tokens=policy.compact_adjudicate_max_tokens,
            allowed_tools=DIAGNOSTIC_TOOLS,
            minimum_tool_calls=0,
            require_novel_tool_views=True,
        ),
    )


def diagnostic_phases(store: EvidenceStore) -> tuple[PhaseSpec, ...]:
    """Return the default compact diagnostic workflow."""

    return compact_diagnostic_phases(store)


def _diagnostic_prompt_fingerprint() -> str:
    from ..tools import modalities, query, survey
    from ..evidence.model_view import MODEL_EVIDENCE_VIEW_VERSION
    from ..knowledge.challenger import (
        CHALLENGE_VERSION,
        KNOWLEDGE_CHALLENGE_SCHEMA,
        KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT,
        RUNTIME_CATEGORIES,
    )
    from ..knowledge.navigator import NAVIGATION_VERSION

    phase_templates = diagnostic_phases(
        EvidenceStore.from_dict(
            {
                "beats": [],
                "global_features": {},
                "metadata": {"diagnostic_gate": {"state": "pass"}},
            },
            record_id="fingerprint",
        )
    )
    prompt_text = "\n".join(
        [
            compact_diagnostic_prompts.SYSTEM_PROMPT,
            json.dumps(DEFAULT_DIAGNOSTIC_PROTOCOL.to_dict(), sort_keys=True),
            json.dumps(DEFAULT_CLINICAL_SAFETY_POLICY.to_dict(), sort_keys=True),
            QUALITY_GATE_VALIDATION_VERSION,
            *(phase.instruction for phase in phase_templates),
            json.dumps(
                [
                    {
                        "key": phase.key,
                        "tool_budget": phase.tool_budget,
                        "minimum_tool_calls": phase.minimum_tool_calls,
                        "coverage_requirements": phase.coverage_requirements,
                        "require_novel_tool_views": phase.require_novel_tool_views,
                        "required_tool_calls": phase.required_tool_calls,
                        "prefetch_tool_calls": phase.prefetch_tool_calls,
                        "max_tokens": phase.max_tokens,
                    }
                    for phase in phase_templates
                ],
                sort_keys=True,
            ),
            compact_diagnostic_prompts.REVISION_INSTRUCTION,
            json.dumps(compact_diagnostic_prompts.PLAN_SCHEMA, sort_keys=True),
            json.dumps(
                compact_diagnostic_prompts.COMPACT_VERDICT_SCHEMA,
                sort_keys=True,
            ),
            json.dumps(diagnostic_prompts.SURVEY_STATE_SCHEMA, sort_keys=True),
            json.dumps(diagnostic_prompts.HYPOTHESIS_STATE_SCHEMA, sort_keys=True),
            json.dumps(diagnostic_prompts.INVESTIGATION_STATE_SCHEMA, sort_keys=True),
            json.dumps(diagnostic_prompts.CHALLENGE_STATE_SCHEMA, sort_keys=True),
            json.dumps(diagnostic_prompts.OUTPUT_SCHEMA, sort_keys=True),
            json.dumps(
                diagnostic_prompts.VERDICT_SECTION_PATCH_SCHEMA,
                sort_keys=True,
            ),
            diagnostic_prompts.VERDICT_SECTION_PATCH_INSTRUCTION,
            json.dumps(
                [
                    spec.to_openai()
                    for spec in (*survey.SPECS, *query.SPECS, *modalities.SPECS)
                    if spec.name in DIAGNOSTIC_TOOLS
                ],
                sort_keys=True,
            ),
            CHALLENGE_VERSION,
            KNOWLEDGE_CHALLENGE_SYSTEM_PROMPT,
            json.dumps(KNOWLEDGE_CHALLENGE_SCHEMA, sort_keys=True),
            json.dumps(RUNTIME_CATEGORIES),
            NAVIGATION_VERSION,
            RULE_SECOND_OPINION_VERSION,
            MODEL_EVIDENCE_VIEW_VERSION,
        ]
    )
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


DIAGNOSTIC_PROMPT_FINGERPRINT = _diagnostic_prompt_fingerprint()


@lru_cache(maxsize=1)
def _shared_knowledge_base() -> Any:
    """Build the immutable governed corpus once per batch process."""
    from ..knowledge import KnowledgeBase

    return KnowledgeBase.from_project()


class ECGDiagnosticAgent(ECGAgent):
    """Dual-channel ECG diagnostician backed by auditable measurements."""

    def __init__(
        self,
        *,
        store: EvidenceStore,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        phases: tuple[PhaseSpec, ...] | None = None,
        max_revisions: int = DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_revisions,
        verify_policy: VerificationPolicy | None = None,
        on_event: Any = None,
        on_checkpoint: Any = None,
        knowledge_guidance: bool = True,
        knowledge_challenge: bool = False,
        knowledge_max_chunks: int = (
            DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.default_max_chunks
        ),
        knowledge_base: Any = None,
        workflow: str = "legacy",
    ) -> None:
        workflow_name = str(workflow or "compact").strip().lower()
        if workflow_name not in {"compact", "legacy"}:
            raise ValueError("workflow must be `compact` or `legacy`")
        self.workflow = "custom" if phases is not None else workflow_name
        self._compact_workflow = self.workflow == "compact"
        # Capture a small conclusion-only routing packet before constructing
        # the physically isolated diagnosis document.  It is not placed in
        # model context until after the blind compact plan has completed.
        self._compact_rule_second_opinion = (
            build_rule_second_opinion(store)
            if self._compact_workflow
            else {}
        )
        self._compact_rule_merge_audit: dict[str, Any] = {}
        diagnostic_store = store.diagnostic_view()
        self._compact_plan: dict[str, Any] = {
            "candidates": [],
            "review_tools": [],
            "quality_limitations": [],
        }
        self._compact_decision_audit: dict[str, Any] = {}
        self._quality_gate = validate_diagnostic_gate(
            diagnostic_store.raw("/metadata/diagnostic_gate", None)
        )
        metadata = diagnostic_store.document.setdefault("metadata", {})
        if isinstance(metadata, dict):
            # Keep every downstream contract, briefing and tool view aligned
            # with the same fail-closed gate rather than retaining the invalid
            # persisted object beside a separate validated copy.
            metadata["diagnostic_gate"] = copy.deepcopy(self._quality_gate)
        self.urgent_review_assessment = assess_urgent_review(diagnostic_store)
        selected_phases = phases or (
            compact_diagnostic_phases(diagnostic_store)
            if self._compact_workflow
            else legacy_diagnostic_phases(diagnostic_store)
        )
        requested_knowledge_guidance = bool(knowledge_guidance)
        if self._compact_workflow:
            knowledge_guidance = False
        self.knowledge_guidance = bool(
            knowledge_guidance
            and any(phase.key == "hypothesize" for phase in selected_phases)
        )
        self.knowledge_challenge = bool(knowledge_challenge)
        self.knowledge_max_chunks = max(
            1,
            min(
                int(knowledge_max_chunks),
                DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.challenge_ceiling,
            ),
        )
        self.knowledge_base = knowledge_base
        self._knowledge_navigation_turn: dict[str, Any] | None = None
        self._knowledge_navigation_audit: dict[str, Any] = {
            "enabled": self.knowledge_guidance,
            "requested": requested_knowledge_guidance,
            "status": "pending" if self.knowledge_guidance else "disabled",
            "max_chunks": min(
                self.knowledge_max_chunks,
                DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.navigation_ceiling,
            ),
            "patient_evidence": False,
            "injected_after_phase": "survey",
            "removed_after_phase": "hypothesize",
        }
        self._knowledge_challenge_audit: dict[str, Any] = {
            "enabled": self.knowledge_challenge,
            # Kept for older audit readers: the full diagnostic pass is only
            # knowledge-isolated when navigation is explicitly disabled.
            "first_pass_isolated": not self.knowledge_guidance,
            "first_pass_knowledge_navigation": self.knowledge_guidance,
            "posthoc_excerpts_isolated": True,
            "max_chunks": self.knowledge_max_chunks,
        }
        self._state_update_audit: list[dict[str, Any]] = []
        self._latest_phase_state: Mapping[str, Any] | None = None
        self._survey_state: Mapping[str, Any] | None = None
        self._tool_selection_audit: list[dict[str, Any]] = []
        self._runtime_phase_specs: dict[str, PhaseSpec] = {}
        self._ledger_commit_audit: list[dict[str, Any]] = []
        if registry is None:
            diagnostic_registry = build_default_registry(
                diagnostic_store,
                budget=None,
                include=DIAGNOSTIC_TOOLS,
            )
            diagnostic_registry.deduplicate_within_phase = True
        else:
            # Only the explicit diagnosis-tool allowlist crosses this boundary.
            # Rebinding protects against a caller accidentally supplying a
            # registry attached to the unredacted adjudication document.
            diagnostic_registry = ToolRegistry(
                store=diagnostic_store,
                budget=registry.budget,
                dispatch_ratio=registry.dispatch_ratio,
                deduplicate_within_phase=True,
            )
            for spec in registry.specs.values():
                if spec.name in DIAGNOSTIC_TOOLS:
                    diagnostic_registry.register(spec)
        self._diagnostic_ledger = DiagnosticLedger(
            store=diagnostic_store,
            diagnosis_codes=diagnostic_prompts.DIAGNOSIS_CATALOG,
            domains=diagnostic_prompts.DIAGNOSTIC_DOMAINS,
            tools=DIAGNOSTIC_TOOLS,
            provenance_lookup=diagnostic_registry.evidence_provenance,
        )
        if self._compact_workflow and hasattr(
            backend,
            "max_retained_evidence_chars",
        ):
            # Full raw results remain in the audit trace. Only the next-model
            # evidence packet is narrowed for the 27B checkpoint.
            backend.max_retained_evidence_chars = min(
                int(
                    getattr(
                        backend,
                        "max_retained_evidence_chars",
                        DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_model_evidence_chars,
                    )
                ),
                DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_model_evidence_chars,
            )
        super().__init__(
            store=diagnostic_store,
            backend=backend,
            registry=diagnostic_registry,
            phases=selected_phases,
            max_revisions=(
                0
                if self._compact_workflow
                else max_revisions
            ),
            max_model_turns=DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_model_turns,
            max_total_tokens=DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_total_tokens,
            max_wall_seconds=DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_wall_seconds,
            max_consecutive_max_tokens=(
                DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_consecutive_max_tokens
            ),
            phase_state_retries=(
                0
                if self._compact_workflow
                else DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.phase_state_retries
            ),
            verify_policy=verify_policy,
            on_event=on_event,
            on_checkpoint=on_checkpoint,
            adaptive_budgets=False,
            prompt_module=(
                compact_diagnostic_prompts
                if self._compact_workflow
                else diagnostic_prompts
            ),
            briefing_builder=(
                build_compact_diagnostic_briefing
                if self._compact_workflow
                else build_diagnostic_briefing
            ),
            protocol_version=DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
            prompt_fingerprint=DIAGNOSTIC_PROMPT_FINGERPRINT,
            mode="diagnose",
            briefing_intro=(
                "Neutral ECG intake. ecgfeat supplies measurements and quality "
                "metadata as tools; no rule conclusion or dataset label is shown."
                + (
                    "\nDETERMINISTIC ROUTING FLAG: measurement-only screening "
                    "requires urgent human review. This is not a confirmed "
                    "diagnosis; preserve it in human_review and do not delay "
                    "waveform review."
                    if self.urgent_review_assessment.get(
                        "do_not_delay_human_review"
                    )
                    else ""
                )
            ),
            revision_allowed_tools=DIAGNOSTIC_TOOLS,
        )

    def run(self) -> AgentResult:
        """Short-circuit records the deterministic quality gate marked stop."""

        if str(self._quality_gate.get("state") or "pass").lower() != "stop":
            return super().run()

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
        result._trace_briefing_text = self.briefing_intro + "\n\n" + briefing.text
        reasons = [
            str(value)
            for value in (self._quality_gate.get("stop_reasons") or [])
            if str(value).strip()
        ] or ["diagnostic_quality_prerequisites_not_met"]
        rendered_reasons = "; ".join(reasons)
        verdict = {
            "summary": (
                "No positive diagnosis or complete ECG interpretation was confirmed: "
                "technical quality did not meet the diagnostic threshold. Reassess using a technically adequate recording."
            ),
            "ranked_complete_interpretations": [
                {
                    "rank": 1,
                    "interpretation_type": "PRIMARY",
                    "complete_diagnosis": (
                        "This ECG is non-diagnostic. No positive diagnosis was confirmed, and abnormality cannot be excluded."
                    ),
                    "confidence": "LOW",
                    "basis_codes": [],
                    "key_uncertainty": "Signal or acquisition conditions were inadequate for reliable ECG assessment.",
                }
            ],
            "diagnoses": [],
            "differential_diagnoses": [],
            "interval_measurement_contexts": [],
            "abstentions": [
                {
                    "topic": "Complete ECG interpretation",
                    "reason": f"Diagnostic-quality hard gate stopped interpretation: {rendered_reasons}",
                    "what_would_resolve_it": (
                        "Verify electrode and lead connections, acquire a technically adequate standard ECG, and have a clinician review the original waveforms."
                    ),
                }
            ],
            "quality_assessment": {
                "interpretability": "non_diagnostic",
                "limitations": reasons,
            },
            "human_review": {
                "required": True,
                "reasons": [
                    "The diagnostic-quality hard gate stopped automated interpretation; acquisition quality requires immediate human confirmation and repeat recording."
                ],
            },
        }
        result.verdict = verdict
        contract_problems = self._validate_verdict_contract(verdict)
        if contract_problems:
            result.error = (
                "deterministic quality-stop verdict failed contract: "
                + "; ".join(contract_problems[:8])
            )
        else:
            result.verification = self._verify(verdict)
        result.phases.append(
            PhaseRecord(
                key="quality_gate",
                tool_budget=0,
                text=json.dumps(verdict, ensure_ascii=False),
                turns=0,
                stop_reason="diagnostic_gate_stop",
                phase_guard_passed=True,
            )
        )
        self._knowledge_navigation_audit.update(
            {"status": "skipped_quality_stop", "enabled": False}
        )
        self._emit("[quality_gate] stop; deterministic non-diagnostic output")
        return self._finish(result)

    def _normalized_phase_patch(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        normalizer = getattr(self.backend, "normalize_verdict", None)
        if not callable(normalizer):
            return value
        normalized = normalizer(dict(value))
        return normalized if isinstance(normalized, Mapping) else value

    @staticmethod
    def _compact_humanize_limitation(value: Any) -> str:
        token = str(value or "").strip()
        translations = {
            "precordial_lead_placement_suspected": "Suspected precordial lead-placement abnormality",
            "limb_lead_reversal_suspected": "Suspected limb-lead connection abnormality",
            "qt_limited": "Limited QT/QTc measurement reliability",
            "pr_limited": "PR measurement unavailable or limited",
            "atrial_activity_uncertain": "Uncertain atrial-activity identification",
            "quality_limited": "Technical quality limits part of the ECG interpretation",
            "t_fusion_reliable=false": "Insufficient reliability of the T-wave fusion endpoint",
            "p_assessment_rejected": "Some P-wave boundary assessments are unavailable",
            "pr_unavailable": "PR interval unavailable",
            "p_axis_unavailable": "P-wave axis unavailable",
            "t_axis_unavailable": "T-wave axis unavailable",
            "atrial_rhythm_unavailable": "Atrial-rhythm assessment unavailable",
        }
        if token.startswith("t_confidence_reason="):
            return "Limited T-wave endpoint confidence in some leads"
        if token.startswith("qt_reliability="):
            return "Limited QT/QTc measurement reliability"
        normalized = token.lower().replace(" ", "_")
        return translations.get(token, translations.get(normalized, token.replace("_", " ")))

    @staticmethod
    def _compact_clean_text(value: Any, fallback: str = "") -> str:
        """Remove model citations and quantities from program-facing prose.

        Patient values are materialized from cited evidence items later. Free
        prose from a small model must never carry an extra threshold, range or
        copied measurement that can bypass the evidence binding contract.
        """

        text = " ".join(str(value or "").split()).strip()
        text = re.sub(
            r"\s*[\uFF08(][^()\uFF08\uFF09]*Q\d+[^()\uFF08\uFF09]*[)\uFF09]",
            "",
            text,
        )
        text = re.sub(r"Q\d+", "", text)
        text = re.sub(
            r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?"
            r"(?:\s*[-–—]\s*\d+(?:\.\d+)?)?\s*"
            r"(?:ms|msec|s|bpm|mv|μv|uv|hz|deg|°|%)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*([,\uFF0C\u3001])\s*(?=[,\uFF0C\u3001\u3002\uFF1B;])", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" \uFF0C,\uFF1B;")
        return text or fallback

    def _compact_evidence_items(self, value: Any) -> list[dict[str, Any]]:
        """Convert citation/claim pairs to the full atomic evidence shape."""

        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in value if isinstance(value, list) else []:
            if not isinstance(row, Mapping):
                continue
            citation = str(row.get("citation") or "").strip()
            if not citation or citation in seen:
                continue
            pointer = self._compact_pointer(citation)
            if pointer is None:
                continue
            claim = self._compact_clean_text(row.get("claim"))
            if not claim:
                continue
            # A 27B model can correctly select a Q id while copying the lead
            # name from an adjacent atom.  Bind lead prose to the selected
            # pointer programmatically so a V2 citation can never be rendered
            # as V3/aVF evidence.  Numeric values remain program-owned.
            lead_match = re.match(r"^/representative_leads/([^/]+)/", pointer)
            if lead_match is not None:
                claim = re.sub(
                    r"(?<![A-Za-z0-9])(?:aVR|aVL|aVF|V[1-6]|III|II|I)"
                    r"(?![A-Za-z0-9])\s*(?:\u5BFC\u8054|lead)?",
                    "",
                    claim,
                    flags=re.IGNORECASE,
                )
                claim = re.sub(r"\s{2,}", " ", claim).strip(" \uFF1A:\uFF0C,\uFF1B;")
                claim = f"Lead {lead_match.group(1)}: {claim or 'morphology measurement'}"
            seen.add(citation)
            items.append(
                {
                    "claim": claim,
                    "value": None,
                    "unit": None,
                    "citations": [citation],
                }
            )
        return items

    def _compact_new_evidence_pointers(self) -> set[str]:
        return {
            str(pointer)
            for call in self.registry.calls
            if call.ok and call.phase == "adjudicate"
            for pointer in call.visible_citations
        }

    def _compact_program_evidence(
        self,
        pointer: str,
        claim: str,
    ) -> dict[str, Any]:
        return {
            "claim": claim,
            "value": None,
            "unit": None,
            "citations": [f"ev:{pointer}"],
        }

    def _compact_deterministic_pathway_step(
        self,
        *,
        code: str,
        step_id: str,
        expected_tool: str,
        pointer_tools: Mapping[str, set[str]],
    ) -> tuple[Any, ...] | None:
        """Resolve definition-level nodes that need no model interpretation."""

        patient = self.store.raw("/metadata/patient_meta", {})
        if resolve_patient_age(
            patient if isinstance(patient, Mapping) else {}
        ).adult is not True:
            # The program-owned thresholds are explicitly adult.  Pediatric
            # and unknown-age pathway steps remain model-owned so raw age and
            # measurements can be interpreted with the appropriate norms.
            return None

        def visible(pointer: str) -> bool:
            return expected_tool in pointer_tools.get(pointer, set())

        def number(pointer: str) -> float | None:
            if not visible(pointer):
                return None
            value = self.store.raw(pointer, None)
            return float(value) if isinstance(value, (int, float)) else None

        if step_id == "rate_threshold":
            pointer = "/global_features/heart_rate_bpm"
            rate = number(pointer)
            age = self.store.raw("/metadata/patient_meta/age", None)
            if rate is None or (
                isinstance(age, (int, float)) and float(age) < 18.0
            ):
                return "unknown", []
            evidence = [
                self._compact_program_evidence(
                    pointer,
                    "Global heart rate used for the definition-level rate assessment",
                )
            ]
            brady_cutoff = float(
                DEFAULT_CLINICAL_SAFETY_POLICY.bradycardia_upper_exclusive_bpm
            )
            tachy_cutoff = float(
                DEFAULT_CLINICAL_SAFETY_POLICY.tachycardia_lower_exclusive_bpm
            )
            if code in {"bradycardia", "sinus_bradycardia"}:
                return ("pass" if rate < brady_cutoff else "fail"), evidence
            if code in {"tachycardia", "sinus_tachycardia"}:
                return ("pass" if rate > tachy_cutoff else "fail"), evidence
            if code == "rate_abnormality":
                return (
                    "pass" if rate < brady_cutoff or rate > tachy_cutoff else "fail"
                ), evidence

        if step_id == "axis_threshold":
            pointer = "/global_features/qrs_axis_deg"
            axis = number(pointer)
            if axis is None:
                return "unknown", []
            evidence = [
                self._compact_program_evidence(
                    pointer,
                    "Global QRS axis used for the definition-level axis-range assessment",
                )
            ]
            if code == "left_axis_deviation":
                return ("pass" if -90.0 < axis < -30.0 else "fail"), evidence
            if code == "right_axis_deviation":
                return ("pass" if 90.0 < axis <= 180.0 else "fail"), evidence
            if code == "extreme_axis_deviation":
                return ("pass" if -180.0 <= axis <= -90.0 else "fail"), evidence

        if code == "ventricular_preexcitation_pattern" and step_id == "preexcitation_components":
            interval_pointer = "/rhythm_inputs/preexcitation/short_pr_interval"
            segment_pointer = "/rhythm_inputs/preexcitation/short_pr_segment"
            count_pointer = "/rhythm_inputs/preexcitation/delta_lead_count"
            if not visible(count_pointer) or not (
                visible(interval_pointer) or visible(segment_pointer)
            ):
                return "unknown", []
            short_interval = self.store.raw(interval_pointer, None)
            short_segment = self.store.raw(segment_pointer, None)
            delta_count = self.store.raw(count_pointer, None)
            if not isinstance(delta_count, (int, float)):
                return "unknown", []
            short_present = short_interval is True or short_segment is True
            short_known = isinstance(short_interval, bool) or isinstance(short_segment, bool)
            if not short_known:
                return "unknown", []
            short_pointer = (
                interval_pointer if visible(interval_pointer) else segment_pointer
            )
            evidence = [
                self._compact_program_evidence(
                    short_pointer,
                    "Short-PR component measurement used in the pre-excitation pathway",
                ),
                self._compact_program_evidence(
                    count_pointer,
                    "Cross-lead delta-candidate count used in the pre-excitation pathway",
                ),
            ]
            if short_present and float(delta_count) >= 2.0:
                return "pass", evidence
            if not short_present and float(delta_count) == 0.0:
                return "fail", evidence
            return "unknown", evidence

        if step_id in {"dominant_qrs_wide", "wide_qrs_representative"}:
            groups = self.store.raw("/groups", {})
            groups = groups if isinstance(groups, Mapping) else {}
            dominant_id = ""
            for group_id, row in groups.items():
                if isinstance(row, Mapping) and bool(
                    (row.get("flags") or {}).get("dominant_group")
                    if isinstance(row.get("flags"), Mapping)
                    else False
                ):
                    dominant_id = str(group_id)
                    break
            if not dominant_id:
                dominant_id = str(
                    self.store.raw("/metadata/representative_group_id", "") or ""
                )
            qrs_pointer = f"/groups/{dominant_id}/mean_qrs_ms"
            pct_pointer = f"/groups/{dominant_id}/member_pct"
            qrs_ms = number(qrs_pointer)
            member_pct = number(pct_pointer)
            if step_id == "dominant_qrs_wide":
                if qrs_ms is None:
                    return "unknown", []
                evidence = [
                    self._compact_program_evidence(
                        qrs_pointer,
                        "Dominant morphology-group QRS duration used for the definition-level width assessment",
                    )
                ]
                if qrs_ms >= 120.0:
                    return "pass", evidence
                if qrs_ms < 110.0:
                    return "fail", evidence
                return "unknown", evidence
            if qrs_ms is None or member_pct is None:
                return "unknown", []
            evidence = [
                self._compact_program_evidence(
                    qrs_pointer,
                    "Dominant morphology-group QRS duration used in the representativeness assessment",
                ),
                self._compact_program_evidence(
                    pct_pointer,
                    "Dominant morphology-group prevalence used in the representativeness assessment",
                ),
            ]
            if qrs_ms >= 120.0 and member_pct >= 50.0:
                return "pass", evidence
            if member_pct < 50.0:
                return "fail", evidence
            return "unknown", evidence

        if step_id == "preexcitation_excluded":
            interval_pointer = "/rhythm_inputs/preexcitation/short_pr_interval"
            segment_pointer = "/rhythm_inputs/preexcitation/short_pr_segment"
            count_pointer = "/rhythm_inputs/preexcitation/delta_lead_count"
            availability_pointer = "/rhythm_inputs/record/availability/pr_available"
            # PR shortening has two constituents and they are not
            # interchangeable. The PR *segment* is the interval minus P
            # duration, so a wide P wave shortens the segment while the interval
            # stays normal -- which is not pre-excitation. The segment is
            # therefore only admissible as substitute evidence when the interval
            # itself could not be measured: reading it unconditionally vetoed a
            # true LAFB whose PR measured 141 ms, while reading only the interval
            # missed a true WPW whose PR was unmeasurable.
            if not visible(count_pointer) or not (
                visible(interval_pointer) or visible(segment_pointer)
            ):
                return "unknown", []
            short_interval = self.store.raw(interval_pointer, None)
            short_segment = self.store.raw(segment_pointer, None)
            delta_count = self.store.raw(count_pointer, None)
            pr_measured = (
                self.store.raw(availability_pointer, None)
                if visible(availability_pointer)
                else None
            )
            if not isinstance(delta_count, (int, float)):
                return "unknown", []
            if short_interval is True:
                short_present = True
            elif pr_measured is True:
                # A measured, non-short PR interval refutes the short-PR
                # constituent outright; the segment cannot override it.
                short_present = False
            else:
                short_present = short_segment is True
            short_known = isinstance(short_interval, bool) or isinstance(
                short_segment, bool
            )
            if not short_known:
                return "unknown", []
            short_pointer = (
                interval_pointer
                if short_interval is True
                or pr_measured is True
                or not visible(segment_pointer)
                else segment_pointer
            )
            evidence = [
                self._compact_program_evidence(
                    short_pointer,
                    "Short-PR component evidence used to exclude pre-excitation",
                ),
                self._compact_program_evidence(
                    count_pointer,
                    "Number of leads with delta-wave candidates used to exclude pre-excitation",
                ),
            ]
            if short_present and float(delta_count) >= 2.0:
                return "fail", evidence
            if not short_present and float(delta_count) == 0.0:
                return "pass", evidence
            # A delta detector without any independent PR shortening is
            # explicitly candidate-only. It prevents exclusion but cannot
            # refute the conduction candidate.
            return "unknown", evidence

        if code == "lvh_voltage_criteria" and step_id == "cross_lead_voltage_criterion":
            component_pointers = {
                "r_avl": "/representative_leads/aVL/params/r_amp_mv",
                "s_v2": "/representative_leads/V2/params/s_amp_mv",
                "s_v3": "/representative_leads/V3/params/s_amp_mv",
                "s_v4": "/representative_leads/V4/params/s_amp_mv",
            }
            values = {name: number(pointer) for name, pointer in component_pointers.items()}
            cornell_available = values["r_avl"] is not None and values["s_v3"] is not None
            peguero_available = values["s_v2"] is not None and values["s_v4"] is not None
            if not cornell_available and not peguero_available:
                return "unknown", []
            # Matches clinical_rules/hypertrophy.py::_adult_lvh exactly: same
            # sex-string set and same per-criterion threshold and comparison
            # operator, so this program node cannot diverge from the rule
            # engine's own definition of the criteria it is replicating.
            sex = str(self.store.raw("/metadata/patient_meta/sex", "") or "").lower()
            is_female = sex in {"female", "f", "woman"}
            cornell_threshold = 2.0 if is_female else 2.8
            cornell_pass = bool(
                cornell_available
                and float(values["r_avl"] or 0.0) + abs(float(values["s_v3"] or 0.0))
                > cornell_threshold
            )
            peguero_threshold = 2.3 if is_female else 2.8
            peguero_pass = bool(
                peguero_available
                and abs(float(values["s_v2"] or 0.0))
                + abs(float(values["s_v4"] or 0.0))
                >= peguero_threshold
            )
            used = (
                ("r_avl", "s_v3")
                if cornell_pass
                else ("s_v2", "s_v4")
                if peguero_pass
                else tuple(name for name, value in values.items() if value is not None)
            )
            component_claims = {
                "r_avl": "Lead aVL R wave is a component measurement of the cross-lead voltage criterion",
                "s_v2": "Lead V2 S wave is a component measurement of the cross-lead voltage criterion",
                "s_v3": "Lead V3 S wave is a component measurement of the cross-lead voltage criterion",
                "s_v4": "Lead V4 S wave is a component measurement of the cross-lead voltage criterion",
            }
            evidence = [
                self._compact_program_evidence(
                    component_pointers[name], component_claims[name]
                )
                for name in used
            ]
            return ("pass" if cornell_pass or peguero_pass else "fail"), evidence

        fact = resolve_additional_deterministic_step(
            self.store,
            code=code,
            step_id=step_id,
            available_pointers={
                pointer
                for pointer, tools in pointer_tools.items()
                if expected_tool in tools
            },
        )
        if fact is not None:
            return (
                fact.status,
                [
                    self._compact_program_evidence(pointer, claim)
                    for pointer, claim in fact.evidence
                ],
                fact.reason_code,
                dict(fact.metrics),
                list(fact.input_pointers),
            )
        return None

    def _compact_program_urgency(self, code: str) -> str:
        if self.urgent_review_assessment.get("do_not_delay_human_review"):
            return "URGENT"
        if code in {
            "acute_occlusion_pattern",
            "left_main_pattern",
            "de_winter_pattern",
            "sgarbossa_positive",
            "complete_av_block",
            "complete_av_block_pattern",
            "wide_complex_tachycardia",
        }:
            return "URGENT"
        if code in {"sinus_rhythm", "sinus_mechanism"}:
            return "NONE"
        return "ROUTINE"

    def _expand_compact_verdict(
        self,
        compact: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compute placements from fixed disease-pathway node results."""

        plan_by_id = {
            str(row.get("id") or ""): row
            for row in self._compact_plan.get("candidates") or []
            if isinstance(row, Mapping) and row.get("id")
        }
        plan_ids_by_code: dict[str, list[str]] = {}
        for planned_id, planned in plan_by_id.items():
            plan_ids_by_code.setdefault(str(planned.get("code") or ""), []).append(
                planned_id
            )
        new_pointers = self._compact_new_evidence_pointers()
        pointer_tools: dict[str, set[str]] = {}
        visible_pointer_tools: dict[str, set[str]] = {}
        for call in self.registry.calls:
            if not call.ok or call.phase != "adjudicate":
                continue
            for pointer in call.citations:
                pointer_tools.setdefault(str(pointer), set()).add(str(call.tool))
            for pointer in call.visible_citations:
                visible_pointer_tools.setdefault(str(pointer), set()).add(str(call.tool))

        model_by_id: dict[str, Mapping[str, Any]] = {}
        rejected_model_rows: list[dict[str, Any]] = []
        repaired_model_rows: list[dict[str, Any]] = []
        for index, raw in enumerate(compact.get("decisions") or []):
            if not isinstance(raw, Mapping):
                continue
            candidate_id = str(raw.get("id") or "")
            code = str(raw.get("code") or "")
            planned = plan_by_id.get(candidate_id)
            if not isinstance(planned, Mapping) or str(planned.get("code") or "") != code:
                matching_ids = plan_ids_by_code.get(code, [])
                if len(matching_ids) == 1 and matching_ids[0] not in model_by_id:
                    repaired_id = matching_ids[0]
                    repaired_model_rows.append(
                        {
                            "index": index,
                            "code": code,
                            "submitted_id": candidate_id,
                            "repaired_id": repaired_id,
                            "reason": "unique_code_to_validated_candidate_id",
                        }
                    )
                    candidate_id = repaired_id
                    planned = plan_by_id.get(candidate_id)
            if (
                not isinstance(planned, Mapping)
                or str(planned.get("code") or "") != code
                or candidate_id in model_by_id
            ):
                rejected_model_rows.append(
                    {"index": index, "code": code, "reason": "not_in_validated_plan"}
                )
                continue
            model_by_id[candidate_id] = raw

        positives: list[dict[str, Any]] = []
        differentials: list[dict[str, Any]] = []
        abstention_codes: list[str] = []
        decision_audit: list[dict[str, Any]] = []
        unresolved_pathways = False

        for candidate_id, planned in plan_by_id.items():
            code = str(planned.get("code") or "")
            raw = model_by_id.get(candidate_id, {})
            pathway = planned.get("diagnostic_pathway")
            pathway = pathway if isinstance(pathway, Mapping) else {}
            expected_steps = [
                step
                for step in (pathway.get("steps") or [])
                if isinstance(step, Mapping) and step.get("id")
            ]
            raw_steps: dict[str, Mapping[str, Any]] = {}
            duplicate_step_ids: list[str] = []
            for row in raw.get("pathway_steps") or []:
                if not isinstance(row, Mapping):
                    continue
                step_id = str(row.get("id") or "")
                if step_id in raw_steps:
                    duplicate_step_ids.append(step_id)
                    continue
                raw_steps[step_id] = row

            support: list[dict[str, Any]] = []
            counter: list[dict[str, Any]] = []
            uncertain_context: list[dict[str, Any]] = []
            step_audit: list[dict[str, Any]] = []
            required_statuses: list[str] = []
            invalidator_statuses: list[str] = []
            unknown_questions: list[str] = []
            support_tools: set[str] = set()
            for expected in expected_steps:
                step_id = str(expected.get("id") or "")
                expected_tool = str(expected.get("tool") or "")
                gate = str(expected.get("gate") or "required")
                submitted = raw_steps.get(step_id, {})
                requested_status = str(submitted.get("status") or "unknown")
                evidence = self._compact_evidence_items(submitted.get("evidence"))
                authorized: list[dict[str, Any]] = []
                # A citation still has to name a measurement fetched in this
                # phase, which is what keeps a remembered or invented number
                # out of a pathway node.  Which view rendered it is a weaker
                # question: sibling views of the same namespace return the same
                # measurement, so accepting them stops a step from collapsing
                # to `unknown` purely because its named view lost a prefetch
                # slot to another candidate.
                authorization = "none"
                for item in evidence:
                    pointers = {
                        pointer
                        for token in (item.get("citations") or [])
                        if (pointer := self._compact_pointer(token)) is not None
                    }
                    fresh = {
                        pointer for pointer in pointers if pointer in new_pointers
                    }
                    if any(
                        expected_tool in visible_pointer_tools.get(pointer, set())
                        for pointer in fresh
                    ):
                        authorized.append(item)
                        authorization = "exact_view"
                    elif any(
                        compact_diagnostic_prompts.namespace_authorizes_step(
                            pointer,
                            expected_tool,
                        )
                        for pointer in fresh
                    ):
                        authorized.append(item)
                        if authorization == "none":
                            authorization = "sibling_view_same_namespace"
                effective_status = (
                    requested_status
                    if requested_status in {"pass", "fail"} and authorized
                    else "unknown"
                )
                resolution_owner = "model_with_tool_bound_evidence"
                deterministic_reason_code = None
                deterministic_metrics: dict[str, Any] = {}
                # Shadow accounting: the model's own label is preserved even
                # when the program answers the node, so every run keeps
                # producing a model-vs-program agreement series for the nodes
                # that have already been migrated and for those still queued.
                model_status = effective_status
                program_status: str | None = None
                deterministic = self._compact_deterministic_pathway_step(
                    code=code,
                    step_id=step_id,
                    expected_tool=expected_tool,
                    pointer_tools=pointer_tools,
                )
                if deterministic is not None:
                    program_status = str(deterministic[0])
                if deterministic is not None and not _deterministic_nodes_shadowed():
                    effective_status = str(deterministic[0])
                    authorized = list(deterministic[1])
                    deterministic_reason_code = (
                        str(deterministic[2])
                        if len(deterministic) >= 3
                        else "legacy_definition_level_measurement_gate"
                    )
                    deterministic_metrics = (
                        dict(deterministic[3])
                        if len(deterministic) >= 4
                        and isinstance(deterministic[3], Mapping)
                        else {}
                    )
                    input_pointers = (
                        list(deterministic[4])
                        if len(deterministic) >= 5
                        else [
                            resolved.pointer
                            for item in authorized
                            for token in (item.get("citations") or [])
                            if str(token).startswith("ev:")
                            and (
                                resolved := self.store.try_resolve(
                                    str(token).removeprefix("ev:")
                                )
                            )
                            is not None
                        ]
                    )
                    self.registry.authorize_program_evidence(
                        tool=expected_tool,
                        citations=input_pointers,
                        source=(
                            f"deterministic_pathway:{candidate_id}:{step_id}:"
                            f"{deterministic_reason_code}"
                        ),
                    )
                    resolution_owner = "deterministic_measurement_gate"
                    authorization = "deterministic_measurement_gate"
                if gate == "required":
                    required_statuses.append(effective_status)
                elif gate == "invalidator":
                    invalidator_statuses.append(effective_status)
                if effective_status == "pass":
                    support.extend(authorized)
                    support_tools.add(expected_tool)
                elif effective_status == "fail":
                    counter.extend(authorized)
                else:
                    # An unknown node may still carry directly measured,
                    # tool-authorized context (for example repeated blocked
                    # atrial candidates whose sequence is not yet diagnostic).
                    # Preserve it for an explicitly unconfirmed differential;
                    # it never contributes a `pass` and can never promote the
                    # candidate to the positive diagnosis list.
                    uncertain_context.extend(authorized)
                    unknown_questions.append(str(expected.get("question") or step_id))
                step_audit.append(
                    {
                        "id": step_id,
                        "gate": gate,
                        "tool": expected_tool,
                        "requested_status": requested_status,
                        "effective_status": effective_status,
                        "model_status": model_status,
                        "program_status": program_status,
                        "resolution_owner": resolution_owner,
                        "authorization": authorization,
                        "authorized_evidence_count": len(authorized),
                        "deterministic_reason_code": deterministic_reason_code,
                        "deterministic_metrics": deterministic_metrics,
                    }
                )

            support = self._deduplicate_full_evidence(support)
            counter = self._deduplicate_full_evidence(counter)
            uncertain_context = self._deduplicate_full_evidence(
                uncertain_context
            )
            quality_allowed = self._compact_gate_allows_category(
                diagnostic_prompts.DIAGNOSIS_CATALOG[code][1]
            )
            # An invalidator reports that the criterion itself does not apply to
            # this record, so it rejects on `fail` alone. `unknown` deliberately
            # does not stall the candidate: an unmeasurable precondition must
            # not cost a true positive.
            if any(status == "fail" for status in invalidator_statuses):
                placement = "rejected"
            elif required_statuses and any(status == "fail" for status in required_statuses):
                placement = "rejected"
            elif (
                required_statuses
                and all(status == "pass" for status in required_statuses)
                and support
                and quality_allowed
            ):
                placement = "confirmed"
            # A partially satisfied pathway deliberately stays a differential.
            # Confirming it at LOW confidence was tried and reverted: the
            # verdict contract (diagnostic_prompts.py:215) requires every
            # positive to be HIGH or MEDIUM precisely so that a weaker basis
            # cannot enter the positive list, and 7 of 16 records failed
            # validation on that rule. The 31-in-287 misses that stop at a
            # differential are better recovered by making the missing node
            # measurable -- see the prefetch lead-set fix -- than by lowering
            # what "confirmed" means.
            else:
                placement = "unresolved"
                unresolved_pathways = True

            display_name = str(pathway.get("display_name") or code)
            promotion_gate_problems: list[str] = []
            if placement == "confirmed":
                confidence = "HIGH" if len(support_tools) >= 2 else "MEDIUM"
                proposed_positive = {
                    "code": code,
                    "statement": display_name,
                    "category": diagnostic_prompts.DIAGNOSIS_CATALOG[code][1],
                    "confidence": confidence,
                    "urgency": self._compact_program_urgency(code),
                    "evidence": support,
                    "counterevidence": counter,
                    "reasoning": "Every required node in the diagnosis-specific pathway is supported by its corresponding measurement view.",
                }
                promotion_gate_problems = (
                    diagnostic_prompts.candidate_evidence_hierarchy_problems(
                        [proposed_positive],
                        self.store.document,
                    )
                )
                if promotion_gate_problems:
                    # The positive evidence hierarchy is stricter than an
                    # individual pathway's node completion.  Preserve all
                    # measurements, but keep a same-chain/candidate-only
                    # result explicitly unconfirmed instead of letting the
                    # final contract reject the whole record.
                    placement = "unresolved"
                    unresolved_pathways = True
                else:
                    positives.append(proposed_positive)
            elif placement == "unresolved":
                if not support:
                    support = uncertain_context
                if not support:
                    support = self._compact_evidence_items(
                        [
                            {
                                "citation": token,
                                "claim": "The blinded overview raised this candidate, but its diagnosis-specific pathway remains incomplete",
                            }
                            for token in (planned.get("support") or [])
                        ]
                    )
                if support:
                    resolving = _fit_required_node_questions(unknown_questions)
                    differentials.append(
                        {
                            "code": code,
                            "statement": display_name + " (unconfirmed)",
                            "confidence": "MEDIUM" if any(
                                status == "pass" for status in required_statuses
                            ) else "LOW",
                            "supporting_evidence": support,
                            "counterevidence": counter,
                            "what_would_resolve_it": (
                                "Complete the following required nodes: " + resolving
                                if resolving
                                else "Complete the diagnosis-specific pathway and review the original waveforms"
                            ),
                        }
                    )
                else:
                    abstention_codes.append(code)

            # A promotion gate can turn a just-completed pathway into an
            # unresolved differential.  Handle that transition after the
            # original branch without duplicating the normal unresolved path.
            if placement == "unresolved" and promotion_gate_problems:
                resolving = _fit_required_node_questions(unknown_questions)
                differentials.append(
                    {
                        "code": code,
                        "statement": display_name + " (unconfirmed)",
                        "confidence": "MEDIUM",
                        "supporting_evidence": support,
                        "counterevidence": counter,
                        "what_would_resolve_it": (
                            resolving
                            or "Independent corroborating evidence is required before this candidate can be promoted"
                        ),
                    }
                )

            decision_audit.append(
                {
                    "id": candidate_id,
                    "code": code,
                    "model_decision_present": bool(raw),
                    "final_placement": placement,
                    "quality_scope_allowed": quality_allowed,
                    "supporting_tool_count": len(support_tools),
                    "uncertain_context_evidence_count": len(uncertain_context),
                    "promotion_gate_problems": promotion_gate_problems,
                    "duplicate_step_ids": duplicate_step_ids,
                    "unexpected_step_ids": sorted(
                        set(raw_steps)
                        - {str(step.get("id") or "") for step in expected_steps}
                    ),
                    "pathway_steps": step_audit,
                }
            )

        limitations = [
            self._compact_clean_text(self._compact_humanize_limitation(value))
            for value in (
                *list(self._quality_gate.get("partial_reasons") or []),
                *list(self._quality_gate.get("stop_reasons") or []),
                *list(self._compact_plan.get("quality_limitations") or []),
            )
            if str(value).strip()
        ]
        if unresolved_pathways:
            limitations.append("Required nodes remain incomplete in some diagnosis-specific pathways")
        if abstention_codes:
            limitations.append("Some rule-based second-opinion candidates lack support from an independent measurement pathway")
        limitations = list(dict.fromkeys(value for value in limitations if value))[:6]
        interval_contexts = self._expand_compact_interval_contexts(
            compact.get("interval_contexts"),
            new_pointers=new_pointers,
        )

        positive_statements = [str(row.get("statement") or "") for row in positives]
        differential_statements = [str(row.get("statement") or "") for row in differentials]
        summary = (
            "ECG conclusion: " + "; ".join(positive_statements) + ". "
            if positives
            else "ECG conclusion: No positive diagnosis was confirmed on this ECG. "
        )
        if differential_statements:
            summary += "Unconfirmed or limited: " + "; ".join(differential_statements) + "."
        elif limitations:
            summary += "Unconfirmed or limited: " + "; ".join(limitations[:3]) + "."

        positive_codes = [str(row.get("code") or "") for row in positives]
        if positives:
            primary_text = "; ".join(positive_statements)
            primary_confidence = (
                "MEDIUM"
                if any(row.get("confidence") == "MEDIUM" for row in positives)
                else "HIGH"
            )
        else:
            primary_text = "No positive diagnosis confirmed"
            if differential_statements:
                primary_text += "; unconfirmed differential interpretations remain"
            primary_confidence = (
                "MEDIUM"
                if any(row.get("confidence") == "MEDIUM" for row in differentials)
                else "LOW"
            )
        ranked = [
            {
                "rank": 1,
                "interpretation_type": "PRIMARY",
                "complete_diagnosis": primary_text,
                "confidence": primary_confidence,
                "basis_codes": positive_codes,
                "key_uncertainty": (
                    limitations[0]
                    if limitations
                    else differentials[0]["what_would_resolve_it"]
                    if differentials
                    else "Review against the original waveforms and clinical context is still required"
                ),
            }
        ]
        confidence_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        for row in sorted(
            differentials,
            key=lambda item: confidence_rank.get(str(item.get("confidence") or "LOW"), 2),
        )[:2]:
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "interpretation_type": "ALTERNATIVE",
                    "complete_diagnosis": str(row.get("statement") or ""),
                    "confidence": str(row.get("confidence") or "LOW"),
                    "basis_codes": [str(row.get("code") or "")],
                    "key_uncertainty": str(row.get("what_would_resolve_it") or ""),
                }
            )

        abstentions = [
            {
                "topic": diagnostic_prompts.DIAGNOSIS_CATALOG[code][0],
                "reason": "The candidate lacks sufficient independent measurement evidence to complete its diagnosis-specific pathway",
                "what_would_resolve_it": "Complete the pathway-specified measurement views and review the original 12-lead waveforms",
            }
            for code in dict.fromkeys(abstention_codes)
            if code in diagnostic_prompts.DIAGNOSIS_CATALOG
        ]
        if not positives and not differentials and not abstentions:
            abstentions.append(
                {
                    "topic": "Complete ECG interpretation",
                    "reason": "Current measurements do not establish a confirmable positive diagnosis or a stable differential mechanism",
                    "what_would_resolve_it": "Have a qualified clinician review the original 12-lead waveforms and repeat the ECG if necessary",
                }
            )

        review_reasons: list[str] = []
        if self.urgent_review_assessment.get("do_not_delay_human_review"):
            review_reasons.append("Immediate or urgent human review of the original waveforms is required")
        if positives:
            review_reasons.append("Review the program-confirmed diagnostic pathways against the original 12-lead evidence")
        if differentials or abstentions:
            review_reasons.append("Incomplete diagnostic pathways require review with the original waveforms and clinical context")
        for context in interval_contexts[:2]:
            review_reasons.append(
                f"{context.get('interval')} measurement is limited; component waves and interval endpoints require human review"
            )
        if not review_reasons:
            review_reasons = ["Verify automated measurements, lead placement and the original 12-lead waveforms"]

        program_status = (
            "confirmed_diagnosis"
            if positives
            else "no_confirmed_positive_diagnosis"
            if differentials
            else "insufficient_evidence"
        )
        self._compact_decision_audit = {
            "workflow": "compact_pathway",
            "model_overall_status": compact.get("overall_status"),
            "program_overall_status": program_status,
            "rejected_model_rows": rejected_model_rows,
            "repaired_model_rows": repaired_model_rows,
            "decisions": decision_audit,
            "pathway_placement_is_program_owned": True,
            "rule_candidates_without_measurement_adjudication": list(
                dict.fromkeys(abstention_codes)
            ),
        }
        gate_state = str(self._quality_gate.get("state") or "pass").lower()
        return {
            "summary": summary,
            "ranked_complete_interpretations": ranked,
            "diagnoses": positives,
            "differential_diagnoses": differentials,
            "interval_measurement_contexts": interval_contexts,
            "abstentions": abstentions,
            "quality_assessment": {
                "interpretability": (
                    "non_diagnostic"
                    if gate_state == "stop"
                    else "limited"
                    if gate_state != "pass" or limitations
                    else "adequate"
                ),
                "limitations": limitations,
            },
            "human_review": {
                "required": True,
                "reasons": list(dict.fromkeys(review_reasons))[:4],
            },
        }

    @staticmethod
    def _deduplicate_full_evidence(
        items: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for item in items:
            citations = tuple(str(value) for value in (item.get("citations") or []))
            if not citations or citations in seen:
                continue
            seen.add(citations)
            kept.append(dict(item))
        return kept

    def _expand_compact_interval_contexts(
        self,
        value: Any,
        *,
        new_pointers: set[str],
    ) -> list[dict[str, Any]]:
        rows_by_interval = {
            str(row.get("interval") or ""): row
            for row in (value if isinstance(value, list) else [])
            if isinstance(row, Mapping)
        }
        contexts: list[dict[str, Any]] = []
        for interval in sorted(self._required_component_intervals()):
            row = rows_by_interval.get(interval, {})
            evidence = self._compact_evidence_items(row.get("evidence"))
            evidence = [
                item
                for item in evidence
                if any(
                    (pointer := self._compact_pointer(token)) is not None
                    and pointer in new_pointers
                    and self._is_direct_component_pointer(interval, pointer)
                    for token in item.get("citations") or []
                )
            ]
            if not evidence:
                direct = [
                    pointer
                    for pointer in new_pointers
                    if self._is_direct_component_pointer(interval, pointer)
                ]
                direct.sort(
                    key=lambda pointer: (
                        0
                        if pointer.startswith("/representative_leads/")
                        and not (self.store.try_resolve(pointer).caveats)
                        else 1
                        if pointer.startswith("/representative_leads/")
                        else 2,
                        pointer,
                    )
                )
                pointer = direct[0] if direct else None
                if pointer:
                    evidence = [
                        {
                            "claim": (
                                "T/U-wave component evidence when QT/QTc is limited"
                                if interval == "QT_QTc"
                                else "P-wave or AV-association component evidence when PR is limited"
                            ),
                            "value": None,
                            "unit": None,
                            "citations": [f"ev:{pointer}"],
                        }
                    ]
            if not evidence:
                continue
            if interval == "QT_QTc":
                assessment = "T/U-wave morphology remains informative, but QT/QTc endpoint reliability is limited"
                interval_available = bool(
                    self.store.raw("/global_features/qt_ms", None) is not None
                    and self.store.raw(
                        "/global_features/qt_reportable", False
                    )
                    is True
                )
                impact = "Limited QT/QTc measurement cannot confirm or exclude a QT abnormality"
                resolving = "Review T-wave endpoints across leads and repeat measurement under improved technical conditions if needed"
            else:
                assessment = "P-wave and AV-association information remains informative, but the global PR interval cannot be reported reliably"
                interval_available = bool(
                    self.store.raw("/global_features/pr_ms", None) is not None
                    and self.store.raw(
                        "/rhythm_inputs/record/availability/pr_available", False
                    )
                    is True
                )
                impact = "Limited PR measurement cannot confirm or exclude an AV conduction abnormality"
                resolving = "Review P-wave onset and QRS association, and repeat the measurement if needed"
            contexts.append(
                {
                    "interval": interval,
                    "status": "limited" if interval_available else "unavailable",
                    "interval_conclusion": (
                        "This interval has reliability limitations and cannot serve as unconditional diagnostic evidence"
                    ),
                    "component_waveform_assessment": assessment,
                    "residual_evidence": evidence,
                    "interpretive_impact": impact,
                    "what_would_resolve_it": resolving,
                }
            )
        return contexts

    def _normalize_verdict(self, verdict: dict[str, Any]) -> dict[str, Any]:
        if self._compact_workflow and isinstance(verdict.get("decisions"), list):
            verdict = self._expand_compact_verdict(verdict)
        return super()._normalize_verdict(verdict)

    def _validate_verdict_contract(
        self,
        candidate: Mapping[str, Any],
    ) -> list[str]:
        problems = super()._validate_verdict_contract(candidate)
        if self.urgent_review_assessment.get("do_not_delay_human_review"):
            review = candidate.get("human_review")
            reasons = (
                review.get("reasons")
                if isinstance(review, Mapping)
                else []
            )
            rendered = " ".join(str(value) for value in (reasons or [])).lower()
            if not any(token in rendered for token in ("\u7d27\u6025", "\u7acb\u5373", "urgent", "immediate")):
                problems.append(
                    "deterministic urgent-review routing is active; human_review.reasons "
                    "must explicitly require urgent/immediate waveform review"
                )
        return problems

    def _phase_visible_citations(self, calls_at_start: int) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                citation
                for call in self.registry.calls[calls_at_start:]
                if call.ok
                for citation in call.visible_citations
            )
        )

    def _additional_phase_state_problems(
        self,
        spec: PhaseSpec,
        state: Mapping[str, Any],
        *,
        calls_at_start: int,
    ) -> list[str]:
        if spec.key not in {"survey", "hypothesize", "investigate", "challenge"}:
            return []
        if not any(
            spec.response_schema is schema
            for schema in (
                diagnostic_prompts.SURVEY_STATE_SCHEMA,
                diagnostic_prompts.HYPOTHESIS_STATE_SCHEMA,
                diagnostic_prompts.INVESTIGATION_STATE_SCHEMA,
                diagnostic_prompts.CHALLENGE_STATE_SCHEMA,
            )
        ):
            return []
        normalized = self._normalized_phase_patch(state)
        problems = self._diagnostic_ledger.validate_phase_response(
            spec.key,
            normalized,
            authorized_citations=self.registry.whitelist,
            new_phase_citations=self._phase_visible_citations(calls_at_start),
        )
        problems.extend(self._new_hypothesis_semantic_problems(normalized))
        problems.extend(self._quality_gate_phase_problems(spec.key, normalized))
        if spec.key == "investigate":
            problems.extend(
                self._component_wave_coverage_problems(
                    normalized,
                    new_phase_citations=self._phase_visible_citations(calls_at_start),
                )
            )
        return list(dict.fromkeys(problems))

    def _quality_gate_phase_problems(
        self,
        phase: str,
        state: Mapping[str, Any],
    ) -> list[str]:
        """Prevent a partial gate from becoming an assessed-normal domain."""

        if str(self._quality_gate.get("state") or "pass").lower() != "partial":
            return []
        suppressed = {
            str(value) for value in (self._quality_gate.get("suppressed_domains") or [])
        }
        if "axis_quantitative" in suppressed:
            suppressed.add("axis")
        allowed = {
            str(value) for value in (self._quality_gate.get("allowed_domains") or [])
        }
        if allowed and "all" not in allowed:
            internal_allowed: set[str] = set()
            if "quality" in allowed:
                internal_allowed.add("quality")
            if "rhythm" in allowed:
                internal_allowed.update(("rhythm_rate", "p_av"))
            if "ectopy" in allowed:
                internal_allowed.add("ectopy_pauses")
            suppressed.update(
                set(diagnostic_prompts.DIAGNOSTIC_DOMAINS) - internal_allowed
            )

        rows: list[tuple[str, Mapping[str, Any], str]] = []
        if phase == "survey":
            domains = state.get("domains")
            if isinstance(domains, Mapping):
                rows.extend(
                    (str(domain), row, f"domains.{domain}")
                    for domain, row in domains.items()
                    if isinstance(row, Mapping)
                )
        else:
            rows.extend(
                (
                    str(row.get("domain") or ""),
                    row,
                    f"domain_updates[{index}]",
                )
                for index, row in enumerate(state.get("domain_updates") or [])
                if isinstance(row, Mapping)
            )
        problems: list[str] = []
        for domain, row, where in rows:
            if domain not in suppressed:
                continue
            if str(row.get("status") or "") not in {"limited", "not_assessed"}:
                problems.append(
                    f"{where} is suppressed by the partial quality gate and must "
                    "remain limited or not_assessed"
                )
            if str(row.get("finding") or "") not in {
                "indeterminate",
                "not_assessed",
            }:
                problems.append(
                    f"{where} is suppressed by the partial quality gate and may "
                    "not be labeled normal or abnormal"
                )
        return problems

    def _new_hypothesis_semantic_problems(
        self,
        state: Mapping[str, Any],
    ) -> list[str]:
        """Reject impossible or visibly mis-scoped new diagnosis identities.

        This gate never infers a positive diagnosis. It only stops a weak model
        from binding an observation to a registered code whose definition is
        already contradicted by the record, while the identity is still cheap
        to repair. Otherwise a late bad code becomes immutable and is carried
        into the final differential by the frozen ledger.
        """

        problems: list[str] = []
        for index, row in enumerate(state.get("create_hypotheses") or []):
            if not isinstance(row, Mapping):
                continue
            code = str(row.get("diagnosis_code") or "")
            domains = {str(value) for value in (row.get("domains") or [])}
            if (
                code in _INTRAVENTRICULAR_CONDUCTION_CODES
                and "conduction_preexcitation" not in domains
            ):
                problems.append(
                    f"create_hypotheses[{index}] diagnosis_code `{code}` is an "
                    "intraventricular QRS-conduction identity, but its domains "
                    "omit `conduction_preexcitation`; do not use it as a label "
                    "for a P/AV-association observation"
                )

            pseudo_verdict = {
                "summary": "",
                "diagnoses": [
                    {
                        "code": code,
                        "statement": str(row.get("phenotype") or ""),
                        "reasoning": str(row.get("phenotype") or ""),
                        "evidence": [],
                        "counterevidence": [],
                    }
                ],
            }
            semantic = diagnostic_prompts.validate_diagnosis_measurement_semantics(
                pseudo_verdict,
                self.store.document,
            )
            for problem in semantic:
                # Missing corroboration is expected for a newly raised
                # question. A hard measurement contradiction is not.
                if "[diagnosis_measurement_conflict]" not in str(problem):
                    continue
                problems.append(
                    f"create_hypotheses[{index}] is definitionally "
                    f"contradicted: {problem}"
                )
        return problems

    @staticmethod
    def _patch_citations(value: Any) -> tuple[str, ...]:
        citations: list[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, Mapping):
                for key, child in item.items():
                    if key == "citations" and isinstance(child, (list, tuple)):
                        citations.extend(str(token) for token in child)
                    else:
                        collect(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)

        collect(value)
        return tuple(dict.fromkeys(citations))

    @staticmethod
    def _is_direct_component_pointer(interval: str, pointer: str) -> bool:
        """Use component morphology/association, not a global interval proxy."""

        token = str(pointer).removeprefix("ev:")
        if interval == "QT_QTc":
            return (
                (
                    token.startswith("/representative_leads/")
                    and any(part in token for part in ("/params/t_", "/params/u_"))
                )
                or (
                    token.startswith("/beat_features/")
                    and any(part in token for part in ("/t_", "/qt_", "/u_"))
                )
                or (
                    token.startswith("/rhythm_inputs/native_beat_profiles/")
                    and any(part in token for part in ("/t_", "/qt_", "/u_"))
                )
                or (
                    token.startswith("/quality/")
                    and token.rsplit("/", 1)[-1]
                    in {"reliable_for_t", "reliable_for_qt"}
                )
            )
        if interval == "PR":
            return (
                (
                    token.startswith("/representative_leads/")
                    and "/params/p_" in token
                )
                or token.startswith("/p_wave_assessments/")
                or token.startswith("/rhythm_inputs/p_events/")
                or token.startswith("/rhythm_inputs/av_block/")
                or (
                    token.startswith("/beat_features/")
                    and any(part in token for part in ("/p_", "/pr_"))
                )
            )
        return False

    def _component_wave_coverage_problems(
        self,
        state: Mapping[str, Any],
        *,
        new_phase_citations: Sequence[str],
    ) -> list[str]:
        """Materialize a component-wave atom already read in Investigation.

        Final interval contexts deliberately require a P/AV or T/U observation,
        not just the failed interval value. If Qwen called the correct view but
        forgot to select one of its atoms, repair that omission while the
        current compact evidence packet and its Q ids are still visible.
        """

        required = self._required_component_intervals()
        if not required:
            return []

        existing = [
            str(row.get("pointer") or "")
            for row in self._diagnostic_ledger.evidence.values()
            if isinstance(row, Mapping) and row.get("pointer")
        ]
        proposed = [
            resolved.pointer
            for token in self._patch_citations(state)
            if (resolved := self.store.try_resolve(str(token))) is not None
        ]
        newly_visible = [
            resolved.pointer
            for token in new_phase_citations
            if (resolved := self.store.try_resolve(str(token))) is not None
        ]
        alias_by_pointer = {
            str(pointer): str(alias)
            for alias, pointer in self.backend.citation_aliases().items()
        } if callable(getattr(self.backend, "citation_aliases", None)) else {}

        problems: list[str] = []
        for interval in sorted(required):
            if any(
                self._is_direct_component_pointer(interval, pointer)
                for pointer in existing
            ):
                continue
            available = [
                pointer
                for pointer in newly_visible
                if self._is_direct_component_pointer(interval, pointer)
            ]
            # Do not demand an unreachable citation. The Challenge instruction
            # remains responsible for acquiring a missing component view.
            if not available:
                continue
            if any(
                self._is_direct_component_pointer(interval, pointer)
                for pointer in proposed
            ):
                continue
            examples = [
                alias_by_pointer.get(pointer, f"ev:{pointer}")
                for pointer in available[:4]
            ]
            component = "T/U" if interval == "QT_QTc" else "P/AV"
            problems.append(
                f"limited `{interval}` has newly read {component} component-wave "
                "evidence, but the patch materializes none of it. Add one exact "
                "component citation to an evidence or domain update while it is "
                f"still visible (for example: {', '.join(examples)})"
            )
        return problems

    def _required_component_intervals(self) -> set[str]:
        """Return intervals whose final wording needs component evidence."""

        required = set(
            diagnostic_prompts._required_interval_contexts(self.store.document)
        )
        global_features = self.store.document.get("global_features")
        if isinstance(global_features, Mapping):
            qt_reliability = str(
                global_features.get("qt_reliability") or ""
            ).lower()
            if (
                global_features.get("t_fusion_reliable") is False
                or qt_reliability
                in {"rescued", "fallback", "low_confidence", "unreliable"}
            ):
                required.add("QT_QTc")
        return required

    def _state_citation_aliases(self) -> Mapping[str, str]:
        """Content ids the ledger snapshot publishes for each evidence atom."""

        return {
            evidence_id: str(row.get("pointer") or "")
            for evidence_id, row in self._diagnostic_ledger.evidence.items()
        }

    def _active_state_codes(self) -> Sequence[str]:
        """Codes already carried by a live hypothesis in the ledger."""

        return [
            str(hypothesis.get("diagnosis_code") or "")
            for hypothesis in self._diagnostic_ledger.hypotheses.values()
            if hypothesis.get("status") not in {"rejected", "abstain"}
        ]

    def _canonicalize_phase_state(
        self,
        spec: PhaseSpec,
        record: PhaseRecord,
        *,
        calls_at_start: int,
    ) -> str:
        """Atomically reduce a model patch and return the canonical snapshot."""

        if spec.key not in {"survey", "hypothesize", "investigate", "challenge"}:
            return record.text
        if not any(
            spec.response_schema is schema
            for schema in (
                diagnostic_prompts.SURVEY_STATE_SCHEMA,
                diagnostic_prompts.HYPOTHESIS_STATE_SCHEMA,
                diagnostic_prompts.INVESTIGATION_STATE_SCHEMA,
                diagnostic_prompts.CHALLENGE_STATE_SCHEMA,
            )
        ):
            return record.text
        try:
            parsed = json.loads(str(record.text or ""))
        except (json.JSONDecodeError, TypeError) as exc:
            record.phase_guard_passed = False
            record.phase_guard_problems.append(
                f"phase patch is not parseable JSON: {exc}"
            )
            return record.text
        if not isinstance(parsed, Mapping):
            record.phase_guard_passed = False
            record.phase_guard_problems.append("phase patch is not a JSON object")
            return record.text
        before_version = self._diagnostic_ledger.version
        try:
            snapshot = self._diagnostic_ledger.apply_phase_response(
                spec.key,
                self._normalized_phase_patch(parsed),
                authorized_citations=self.registry.whitelist,
                new_phase_citations=self._phase_visible_citations(calls_at_start),
            )
        except LedgerPatchError as exc:
            record.phase_guard_passed = False
            record.phase_guard_problems.extend(exc.problems)
            return record.text
        self._latest_phase_state = snapshot
        if spec.key == "survey":
            self._survey_state = copy.deepcopy(snapshot)
        self._ledger_commit_audit.append(
            {
                "phase": spec.key,
                "base_version": before_version,
                "committed_version": self._diagnostic_ledger.version,
                "snapshot_hash": snapshot.get("snapshot_hash"),
                "created_hypotheses": len(parsed.get("create_hypotheses") or []),
                "evidence_updates": len(parsed.get("evidence_updates") or []),
                "domain_updates": len(parsed.get("domain_updates") or []),
                "transitions": len(parsed.get("transitions") or []),
                "frozen": self._diagnostic_ledger.frozen,
            }
        )
        return json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))

    def _checkpoint_payload_extra(
        self,
        result: AgentResult,
        *,
        status: str,
    ) -> dict[str, Any]:
        return {
            "diagnostic_ledger": self._diagnostic_ledger.audit_state(),
            "diagnostic_ledger_commits": list(self._ledger_commit_audit),
        }

    def _compact_pointer(self, token: Any) -> str | None:
        """Resolve one model-visible Qn id without accepting invented paths."""

        value = str(token or "").strip()
        aliases = getattr(self.backend, "citation_aliases", None)
        alias_map = aliases() if callable(aliases) else {}
        pointer = str(alias_map.get(value) or "") if isinstance(alias_map, Mapping) else ""
        if not pointer:
            return None
        resolved = self.store.try_resolve(pointer)
        if resolved is None or resolved.pointer not in self.registry.whitelist:
            return None
        return resolved.pointer

    @staticmethod
    def _compact_even_sample(values: Sequence[int], limit: int = 4) -> list[int]:
        ordered = list(dict.fromkeys(int(value) for value in values))
        if len(ordered) <= limit:
            return ordered
        if limit <= 1:
            return ordered[:1]
        indexes = [
            round(index * (len(ordered) - 1) / (limit - 1))
            for index in range(limit)
        ]
        return [ordered[index] for index in dict.fromkeys(indexes)]

    def _compact_row_ids(
        self,
        pointer: str,
        field: str,
        *,
        limit: int = 4,
    ) -> list[int]:
        rows = self.store.raw(pointer, [])
        identifiers = [
            int(row.get(field, index))
            for index, row in enumerate(rows if isinstance(rows, list) else [])
            if isinstance(row, Mapping)
            and isinstance(row.get(field, index), (int, float))
        ]
        return self._compact_even_sample(identifiers, limit)

    @staticmethod
    def _compact_candidate_domains(
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[set[str], set[str]]:
        codes = {str(row.get("code") or "") for row in candidates}
        domains = {
            str(domain)
            for row in candidates
            for domain in (row.get("domains") or [])
        }
        return codes, domains

    def _compact_prefetch_arguments(
        self,
        tool: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build a validated, bounded argument set for one planned view."""

        codes, domains = self._compact_candidate_domains(candidates)
        if tool == "get_global_table":
            fields: list[str] = []
            if "axis" in domains or any("axis" in code for code in codes):
                fields.append("qrs_axis_deg")
            if "rhythm_rate" in domains or any(
                token in code
                for code in codes
                for token in ("brady", "tachy", "rate")
            ):
                fields.extend(["heart_rate_bpm", "rr_cv"])
            return {"fields": list(dict.fromkeys(fields or ["heart_rate_bpm"]))}
        if tool == "get_rhythm_profile":
            return {
                "sections": [
                    "background",
                    "atrial_signal",
                    "av_association",
                    "preexcitation",
                    "aberrancy",
                ]
            }
        if tool == "get_atrial_event_table":
            arguments: dict[str, Any] = {
                "fields": [
                    "association_type",
                    "confidence",
                    "pr_ms",
                    "template_similarity",
                    "source_leads",
                ],
                "limit": 12,
            }
            event_ids = self._compact_row_ids(
                "/rhythm_inputs/p_events", "p_event_id", limit=4
            )
            if event_ids:
                arguments["event_ids"] = event_ids
            return arguments
        if tool == "get_p_assessment_table":
            arguments = {
                "fields": [
                    "accepted",
                    "ta_ambiguous",
                    "reject_reasons",
                    "onset_confidence",
                    "offset_confidence",
                    "valid_leads",
                ],
                "limit": 12,
            }
            beat_ids = self._compact_row_ids(
                "/p_wave_assessments", "beat_id", limit=4
            )
            if beat_ids:
                arguments["beat_ids"] = beat_ids
            return arguments
        if tool == "get_morphology_groups":
            return {"include_beats": "ectopy_pauses" in domains}
        if tool == "get_beat_table":
            arguments = {
                "fields": ["rr_prev_ms", "rr_next_ms", "group_id", "paced"]
            }
            beat_ids = self._compact_row_ids(
                "/beat_features", "beat_id", limit=4
            )
            if beat_ids:
                arguments["beat_ids"] = beat_ids
            return arguments
        if tool == "get_qrs_measurement_bundle":
            fascicular = any(
                token in code
                for code in codes
                for token in ("lafb", "lpfb", "fascicular")
            )
            return {
                "leads": (
                    ["I", "II", "III", "aVL", "aVF", "V1", "V6"]
                    if fascicular
                    else ["I", "aVL", "V1", "V2", "V5", "V6"]
                )
            }
        if tool in {"get_morphology_map", "get_lead_table"}:
            atrial = any(
                token in code
                for code in codes
                for token in ("atrial", "p_wave", "interatrial")
            )
            repolarization = "q_st_t_u" in domains or any(
                token in code
                for code in codes
                for token in ("t_wave", "qt", "repolarization")
            )
            st_focused = any(
                token in code for code in codes for token in ("st_", "injury")
            )
            qrs_focused = any(
                token in code
                for code in codes
                for token in (
                    "q_wave",
                    "infarct",
                    "r_wave",
                    "rotation",
                    "dextrocardia",
                    "voltage",
                    "hypertrophy",
                )
            )
            conduction_focused = any(
                token in code
                for code in codes
                for token in (
                    "ivcd",
                    "bundle",
                    "rbbb",
                    "lbbb",
                    "lafb",
                    "lpfb",
                    "fascicular",
                    "preexcitation",
                )
            )
            voltage_focused = any(
                token in code
                for code in codes
                for token in ("lvh", "rvh", "voltage", "hypertrophy")
            )
            profile = (
                "p"
                if atrial
                else "st"
                if st_focused
                else "qrs"
                if qrs_focused
                else "t_u"
                if repolarization
                else "qrs"
            )
            if tool == "get_morphology_map":
                arguments: dict[str, Any] = {"profile": profile}
                if profile == "qrs" and conduction_focused:
                    # I/V1/V5/V6 answer the BUNDLE-branch question. The
                    # fascicular blocks are defined in the frontal plane
                    # instead -- qR in aVL with rS in II, III and aVF for LAFB,
                    # and the mirror pattern for LPFB -- so with only the
                    # bundle leads on screen `required_lead_pattern` resolved
                    # `unknown` and LAFB/LPFB scored 0 TP across four repeats.
                    fascicular = any(
                        token in code
                        for code in codes
                        for token in ("lafb", "lpfb", "fascicular")
                    )
                    arguments["leads"] = (
                        ["I", "II", "III", "aVL", "aVF", "V1", "V6"]
                        if fascicular
                        else ["I", "V1", "V5", "V6"]
                    )
                return arguments
            if profile == "qrs" and voltage_focused:
                # aVL/V2/V3/V4 are the Cornell and Peguero LEFT-sided leads.
                # Sending them for a right-sided candidate fetched no lead the
                # RVH criteria are defined on, so `rvh_qrs_distribution`
                # resolved `unknown` with the note "V1 R' amplitude is
                # unavailable" and RVH scored 0 TP across four repeats. Right
                # ventricular hypertrophy is read in V1 (tall R / R') against
                # the left precordial S waves, so fetch those instead.
                right_sided = any(
                    token in code for code in codes for token in ("rvh", "right_ventric")
                )
                left_sided = any(
                    token in code for code in codes for token in ("lvh", "left_ventric")
                )
                leads = ["aVL", "V2", "V3", "V4"] if left_sided else []
                if right_sided:
                    leads = list(dict.fromkeys(["V1", "V5", "V6"] + leads))
                return {
                    "fields": ["r_amp_mv", "s_amp_mv"],
                    "leads": leads or ["aVL", "V2", "V3", "V4"],
                }
            fields = {
                "p": [
                    "p_dur_ms",
                    "p_amp_mv",
                    "p_notched",
                    "p_biphasic",
                    "p_terminal_duration_ms",
                    "p_terminal_amp_mv",
                ],
                "st": [
                    "st_hybrid_j_mv",
                    "st_hybrid_60ms_mv",
                    "st_hybrid_shape",
                    "st_hybrid_reliable",
                ],
                "t_u": [
                    "t_amp_mv",
                    "t_polarity",
                    "t_symmetry",
                    "t_sqi_score",
                    "u_prominence_mv",
                ],
                "qrs": [
                    "qrs_ms",
                    "q_duration_ms",
                    "q_amp_mv",
                    "r_amp_mv",
                    "s_amp_mv",
                    "qrs_slur_flag",
                ],
            }
            return {"fields": fields[profile]}
        if tool == "get_native_beat_profile":
            qrs_focused = any(
                token in code
                for code in codes
                for token in (
                    "q_wave",
                    "infarct",
                    "r_wave",
                    "rotation",
                    "dextrocardia",
                    "conduction",
                    "bundle",
                    "ivcd",
                )
            )
            return {
                "profile": (
                    "qrs_infarct"
                    if qrs_focused or "q_st_t_u" not in domains
                    else "repolarization"
                ),
                "max_beats": 3,
            }
        if tool == "get_pacing_profile":
            return {"include_beats": False}
        if tool == "get_interval_waveform_context":
            pr_focused = any(
                token in code
                for code in codes
                for token in (
                    "first_degree",
                    "short_pr",
                    "av_block",
                    "av_delay",
                )
            )
            return {
                "interval": "pr" if pr_focused else "qt",
                "include_beat_flags": False,
            }
        return {}

    def _compact_candidate_semantic_conflict(self, code: str) -> str | None:
        if code == "wide_complex_tachycardia":
            patient = self.store.raw("/metadata/patient_meta", {})
            adult = resolve_patient_age(
                patient if isinstance(patient, Mapping) else {}
            ).adult
            maximum_rate = self.store.raw(
                "/global_features/heart_rate_max_bpm", None
            )
            if (
                adult is True
                and isinstance(maximum_rate, (int, float))
                and float(maximum_rate)
                <= float(
                    DEFAULT_CLINICAL_SAFETY_POLICY.tachycardia_lower_exclusive_bpm
                )
            ):
                return (
                    "[diagnosis_measurement_conflict] wide-complex tachycardia "
                    "is contradicted by the record-wide maximum measured heart "
                    "rate; a wide QRS without a tachycardic sequence is not WCT"
                )
        conflicts = [
            problem
            for problem in diagnostic_prompts.validate_diagnosis_measurement_semantics(
                {
                    "diagnoses": [
                        {
                            "code": code,
                            "statement": diagnostic_prompts.DIAGNOSIS_CATALOG[code][0],
                            "reasoning": diagnostic_prompts.DIAGNOSIS_CATALOG[code][0],
                            "evidence": [],
                            "counterevidence": [],
                        }
                    ]
                },
                self.store.document,
            )
            if "[diagnosis_measurement_conflict]" in str(problem)
        ]
        return str(conflicts[0]) if conflicts else None

    @staticmethod
    def _compact_model_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        """Project the validated plan to only what adjudication must retain."""

        return {
            "candidates": [
                {
                    key: copy.deepcopy(row.get(key))
                    for key in (
                        "id",
                        "code",
                        "sources",
                        "rule_second_opinion",
                        "support",
                        "counter",
                        "diagnostic_pathway",
                    )
                    if row.get(key) not in (None, [], {})
                }
                for row in (plan.get("candidates") or [])
                if isinstance(row, Mapping)
            ],
            "quality_limitations": list(plan.get("quality_limitations") or []),
        }

    @staticmethod
    def _compact_runtime_verdict_schema(
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Compile the static verdict contract to the exact model-owned work.

        The static schema must permit an empty decision list because some
        records have only program-owned pathway nodes.  Reusing that permissive
        shape after a concrete plan was known let a truncated or inattentive
        adjudication silently omit model-owned candidates and steps.  The
        runtime schema fixes the required row count and narrows candidate and
        step ids to the validated plan; deterministic-only candidates remain
        legitimately absent from the model response.
        """

        schema = copy.deepcopy(
            compact_diagnostic_prompts.COMPACT_VERDICT_SCHEMA
        )
        decisions_schema = schema["properties"]["decisions"]
        base_decision = copy.deepcopy(decisions_schema["items"])
        decision_options: list[dict[str, Any]] = []

        for candidate in candidates:
            pathway = candidate.get("diagnostic_pathway")
            pathway = pathway if isinstance(pathway, Mapping) else {}
            model_steps = [
                step
                for step in (pathway.get("steps") or [])
                if isinstance(step, Mapping)
                and step.get("id")
                and str(step.get("owner") or "model") != "program"
            ]
            if not model_steps:
                continue

            candidate_schema = copy.deepcopy(base_decision)
            candidate_schema["properties"]["id"] = {
                "type": "string",
                "enum": [str(candidate.get("id") or "candidate")],
            }
            candidate_schema["properties"]["code"] = {
                "type": "string",
                "enum": [str(candidate.get("code") or "")],
            }
            pathway_steps = candidate_schema["properties"]["pathway_steps"]
            step_options: list[dict[str, Any]] = []
            for step in model_steps:
                step_schema = copy.deepcopy(
                    compact_diagnostic_prompts._PATHWAY_STEP_RESULT
                )
                step_schema["properties"]["id"] = {
                    "type": "string",
                    "enum": [str(step.get("id"))],
                }
                step_options.append(step_schema)
            pathway_steps["minItems"] = len(step_options)
            pathway_steps["maxItems"] = len(step_options)
            # vLLM's structured-output grammar rejects ``uniqueItems``.  A
            # positional tuple is both supported and stronger here: every
            # planned step must occur exactly once and in plan order.
            pathway_steps["prefixItems"] = step_options
            pathway_steps["items"] = False
            decision_options.append(candidate_schema)

        decisions_schema["minItems"] = len(decision_options)
        decisions_schema["maxItems"] = len(decision_options)
        if decision_options:
            decisions_schema["prefixItems"] = decision_options
            decisions_schema["items"] = False
        return schema

    def _compact_candidate_pathway_calls(
        self,
        candidate: Mapping[str, Any],
    ) -> list[tuple[str, dict[str, Any], str]]:
        """Compile one fixed pathway to exact local view calls."""

        pathway = candidate.get("diagnostic_pathway")
        pathway = pathway if isinstance(pathway, Mapping) else {}
        calls: list[tuple[str, dict[str, Any], str]] = []
        for step in pathway.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            tool = str(step.get("tool") or "")
            if tool not in compact_diagnostic_prompts.COMPACT_DIRECT_TOOLS:
                continue
            explicit = step.get("arguments")
            arguments = (
                copy.deepcopy(dict(explicit))
                if isinstance(explicit, Mapping)
                else self._compact_prefetch_arguments(tool, [candidate])
            )
            calls.append((tool, arguments, str(step.get("id") or "step")))
        return calls

    def _compact_bound_candidates_by_view_budget(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Keep only candidates whose complete path fits the local-view cap.

        This executes before the adjudication plan is shown to Qwen.  A path is
        therefore either fully retrievable or absent; no candidate can remain
        in model context while one of its named views is silently truncated.
        """

        ceiling = DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_tool_ceiling
        signatures: set[str] = set()
        for interval in sorted(self._required_component_intervals()):
            arguments = {
                "interval": "qt" if interval == "QT_QTc" else "pr",
                "include_beat_flags": False,
            }
            signatures.add(
                self.registry.call_signature(
                    "get_interval_waveform_context",
                    arguments,
                )
            )

        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for raw in candidates:
            candidate = copy.deepcopy(dict(raw))
            calls = self._compact_candidate_pathway_calls(candidate)
            candidate_signatures = {
                self.registry.call_signature(tool, arguments)
                for tool, arguments, _ in calls
            }
            projected = signatures | candidate_signatures
            if len(projected) > ceiling:
                dropped.append(
                    {
                        "id": str(candidate.get("id") or ""),
                        "code": str(candidate.get("code") or ""),
                        "sources": list(candidate.get("sources") or []),
                        "required_new_views": len(projected - signatures),
                        "projected_view_count": len(projected),
                    }
                )
                continue
            signatures = projected
            kept.append(candidate)
        return kept, dropped

    def _merge_compact_rule_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        notes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Union blind-plan candidates with a bounded rule second opinion."""

        sanitizations = notes if notes is not None else []
        merged: list[dict[str, Any]] = []
        seen_codes: dict[str, dict[str, Any]] = {}
        seen_families: dict[str, dict[str, Any]] = {}
        seen_ids: set[str] = set()
        for raw in candidates:
            candidate = copy.deepcopy(dict(raw))
            code = str(candidate.get("code") or "")
            family = semantic_candidate_family(code)
            if family in seen_families:
                sanitizations.append(
                    f"independent candidate `{code}` dropped: semantic family "
                    f"`{family}` already represented"
                )
                continue
            candidate["sources"] = list(
                dict.fromkeys(
                    [
                        *list(candidate.get("sources") or []),
                        "independent_measurement_plan",
                    ]
                )
            )
            merged.append(candidate)
            seen_codes[code] = candidate
            seen_families[family] = candidate
            seen_ids.add(str(candidate.get("id") or ""))

        added: list[str] = []
        annotated: list[str] = []
        annotated_family: list[str] = []
        excluded: list[dict[str, str]] = []
        deferred: list[str] = []
        rule_rows = self._compact_rule_second_opinion.get("candidate_rows") or []
        for rule_index, raw in enumerate(rule_rows):
            if not isinstance(raw, Mapping):
                continue
            code = str(raw.get("code") or "")
            if not code or code not in diagnostic_prompts.DIAGNOSIS_CATALOG:
                continue
            rule_metadata = {
                key: copy.deepcopy(raw.get(key))
                for key in (
                    "code",
                    "label",
                    "publication",
                    "status",
                    "confidence",
                    "priority",
                )
            }
            existing = seen_codes.get(code)
            family = semantic_candidate_family(code)
            if existing is None:
                existing = seen_families.get(family)
            if existing is not None:
                existing["sources"] = list(
                    dict.fromkeys(
                        [
                            *list(existing.get("sources") or []),
                            "ecgfeat_rule_second_opinion",
                        ]
                    )
                )
                existing.setdefault("rule_second_opinions", []).append(rule_metadata)
                if str(existing.get("code") or "") == code:
                    existing["rule_second_opinion"] = rule_metadata
                    annotated.append(code)
                else:
                    annotated_family.append(code)
                continue
            if len(merged) >= compact_diagnostic_prompts.MERGED_CANDIDATE_MAX:
                deferred.append(code)
                continue
            conflict = self._compact_candidate_semantic_conflict(code)
            if conflict:
                excluded.append({"code": code, "reason": conflict})
                sanitizations.append(
                    f"rule second-opinion candidate `{code}` excluded: {conflict}"
                )
                continue

            blueprint = rule_candidate_blueprint(code)
            slug = re.sub(r"[^a-z0-9_]+", "_", code.lower()).strip("_")
            candidate_id = f"r{rule_index + 1}_{slug}"[:32]
            suffix = 2
            while candidate_id in seen_ids:
                candidate_id = f"r{rule_index + 1}_{slug[:25]}_{suffix}"[:32]
                suffix += 1
            candidate = {
                "id": candidate_id,
                "code": code,
                "domains": list(blueprint["domains"])[:3],
                "support": [],
                "counter": [],
                "checks": list(blueprint["checks"])[:2],
                "uncertainty": (
                    "The rule-based second opinion only raises a candidate for review; it requires independent confirmation from measurement views in this run."
                ),
                "sources": ["ecgfeat_rule_second_opinion"],
                "rule_second_opinion": rule_metadata,
            }
            merged.append(candidate)
            seen_codes[code] = candidate
            seen_families[family] = candidate
            seen_ids.add(candidate_id)
            added.append(code)
        patient_meta = self.store.raw("/metadata/patient_meta", {})
        adult_program_pathways = resolve_patient_age(
            patient_meta if isinstance(patient_meta, Mapping) else {}
        ).adult is True
        for candidate in merged:
            candidate["diagnostic_pathway"] = build_diagnostic_pathway(
                str(candidate.get("code") or ""),
                fallback_checks=[
                    row
                    for row in (candidate.get("checks") or [])
                    if isinstance(row, Mapping)
                ],
                program_owned=adult_program_pathways,
            )
        independent_rows = [
            row
            for row in merged
            if "independent_measurement_plan" in (row.get("sources") or [])
        ]
        rule_only_rows = [
            row
            for row in merged
            if "independent_measurement_plan" not in (row.get("sources") or [])
            and "ecgfeat_rule_second_opinion" in (row.get("sources") or [])
        ]
        other_rows = [
            row
            for row in merged
            if row not in independent_rows and row not in rule_only_rows
        ]
        # Preserve the leading blind differential while guaranteeing that a
        # specific rule-audit candidate is considered before a third, often
        # redundant small-model hypothesis consumes the remaining views.
        budget_order = [
            *independent_rows[:2],
            *rule_only_rows,
            *independent_rows[2:],
            *other_rows,
        ]
        bounded, dropped_by_views = self._compact_bound_candidates_by_view_budget(
            budget_order
        )
        retained_codes = {str(row.get("code") or "") for row in bounded}
        added = [code for code in added if code in retained_codes]
        deferred.extend(
            str(row.get("code") or "")
            for row in dropped_by_views
            if "ecgfeat_rule_second_opinion" in (row.get("sources") or [])
        )
        for row in dropped_by_views:
            sanitizations.append(
                f"candidate `{row.get('code')}` dropped before adjudication: its "
                "complete pathway exceeds the compact exact-view budget"
            )
        self._compact_rule_merge_audit = {
            "independent_candidate_count": len(candidates),
            "merged_candidate_count": len(bounded),
            "rule_added_codes": added,
            "rule_annotated_codes": annotated,
            "rule_annotated_family_codes": annotated_family,
            "rule_excluded": excluded,
            "rule_deferred_by_cap": list(dict.fromkeys(deferred)),
            "candidates_dropped_by_view_budget": dropped_by_views,
            "merged_candidate_cap": compact_diagnostic_prompts.MERGED_CANDIDATE_MAX,
            "exact_view_cap": DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_tool_ceiling,
        }
        return bounded

    def _capture_compact_plan(
        self,
        record: PhaseRecord,
        messages: list[dict[str, Any]],
    ) -> None:
        """Validate a small plan once; never ask Qwen to manage workflow state."""

        try:
            parsed = json.loads(str(record.text or ""))
        except (json.JSONDecodeError, TypeError) as exc:
            record.phase_guard_passed = False
            record.phase_guard_problems = [f"compact plan is not JSON: {exc}"]
            return
        if not isinstance(parsed, Mapping):
            record.phase_guard_passed = False
            record.phase_guard_problems = ["compact plan is not an object"]
            return
        parsed = copy.deepcopy(dict(parsed))
        notes: list[str] = []

        # Qwen occasionally switches a free-text plan field to Chinese even
        # under guided JSON.  Re-running the whole model turn cannot improve the
        # structured routing signal and caused 2/50 records per repeat to fail.
        # These prose fields do not select diagnosis identities, citations,
        # tools or arguments; fixed diagnostic pathways replace the questions
        # before adjudication.  Canonicalize only those non-authoritative fields
        # and retain an explicit audit note.  Unexpected Han text in any
        # identity/routing field remains a hard failure below.
        for candidate_index, candidate in enumerate(parsed.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            uncertainty = str(candidate.get("uncertainty") or "")
            if diagnostic_prompts.non_english_text_paths(uncertainty):
                candidate["uncertainty"] = "Targeted measurement review is required."
                notes.append(
                    f"candidate[{candidate_index}].uncertainty replaced with an "
                    "English audit placeholder; routing fields were preserved"
                )
            for check_index, check in enumerate(candidate.get("checks") or []):
                if not isinstance(check, dict):
                    continue
                question = str(check.get("question") or "")
                if diagnostic_prompts.non_english_text_paths(question):
                    check["question"] = (
                        "Review the selected measurement view for direct support "
                        "or counterevidence."
                    )
                    notes.append(
                        f"candidate[{candidate_index}].checks[{check_index}].question "
                        "replaced with an English audit placeholder; tool and purpose "
                        "were preserved"
                    )
        limitations: list[Any] = []
        for limitation_index, limitation in enumerate(
            parsed.get("quality_limitations") or []
        ):
            text = str(limitation or "").strip()
            if diagnostic_prompts.non_english_text_paths(text):
                text = "The overview indicates a quality limitation requiring human review."
                notes.append(
                    f"quality_limitations[{limitation_index}] replaced with an "
                    "English audit placeholder"
                )
            limitations.append(text)
        parsed["quality_limitations"] = limitations

        language_paths = diagnostic_prompts.non_english_text_paths(
            parsed,
            "compact_plan",
        )
        if language_paths:
            record.phase_guard_passed = False
            record.phase_guard_problems = [
                f"{path} must be English-only and must not contain Han-script text"
                for path in language_paths[:8]
            ]
            return

        candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_codes: set[str] = set()
        seen_families: set[str] = set()
        # The schema's maxItems is the global ceiling; this record's own limit
        # is what the instruction stated, so enforce it here rather than
        # trusting the model to have honoured a number in prose.
        candidate_limit = plan_candidate_limit(self.store)
        for index, raw in enumerate(parsed.get("candidates") or []):
            if len(candidates) >= candidate_limit:
                notes.append(
                    f"candidate[{index}] dropped: exceeds this record's "
                    f"plan limit of {candidate_limit}"
                )
                continue
            if not isinstance(raw, Mapping):
                notes.append(f"candidate[{index}] dropped: not an object")
                continue
            candidate_id = str(raw.get("id") or f"h{index + 1}").strip()
            code = str(raw.get("code") or "").strip()
            family = semantic_candidate_family(code)
            if (
                not candidate_id
                or candidate_id in seen_ids
                or code in seen_codes
                or family in seen_families
                or code not in diagnostic_prompts.DIAGNOSIS_CATALOG
                or code in diagnostic_prompts.NON_HYPOTHESIS_CODES
            ):
                notes.append(
                    f"candidate[{index}] dropped: duplicate or invalid identity/code"
                )
                continue
            semantic_conflict = self._compact_candidate_semantic_conflict(code)
            if semantic_conflict:
                notes.append(
                    f"candidate[{index}] `{code}` dropped: "
                    + semantic_conflict
                )
                continue
            support = [
                str(token)
                for token in (raw.get("support") or [])
                if self._compact_pointer(token) is not None
            ]
            counter = [
                str(token)
                for token in (raw.get("counter") or [])
                if self._compact_pointer(token) is not None
            ]
            if not support:
                notes.append(
                    f"candidate[{index}] `{code}` dropped: no authorized overview evidence"
                )
                continue
            checks = [
                {
                    "tool": str(check.get("tool") or ""),
                    "purpose": str(check.get("purpose") or "support"),
                    "question": str(check.get("question") or "").strip(),
                }
                for check in (raw.get("checks") or [])
                if isinstance(check, Mapping)
                and str(check.get("tool") or "")
                in compact_diagnostic_prompts.COMPACT_DIRECT_TOOLS
            ][:2]
            if not checks:
                notes.append(f"candidate[{index}] `{code}` dropped: no valid check")
                continue
            if not any(check["purpose"] == "falsify" for check in checks):
                checks[-1]["purpose"] = "falsify"
                notes.append(
                    f"candidate[{index}] `{code}` final check reclassified as "
                    "falsification so absence of contradiction cannot pass as support"
                )
            seen_ids.add(candidate_id)
            seen_codes.add(code)
            seen_families.add(family)
            candidates.append(
                {
                    "id": candidate_id,
                    "code": code,
                    "domains": [
                        str(value)
                        for value in (raw.get("domains") or [])
                        if str(value) in diagnostic_prompts.DIAGNOSTIC_DOMAINS
                    ][:3],
                    "support": list(dict.fromkeys(support))[:4],
                    "counter": list(dict.fromkeys(counter))[:3],
                    "checks": checks,
                    "uncertainty": str(raw.get("uncertainty") or "").strip(),
                }
            )

        review_tools = [
            str(tool)
            for tool in (parsed.get("review_tools") or [])
            if str(tool) in compact_diagnostic_prompts.COMPACT_DIRECT_TOOLS
        ]
        independent_candidate_count = len(candidates)
        merged_candidates = self._merge_compact_rule_candidates(candidates, notes)
        self._compact_plan = {
            "candidates": merged_candidates,
            "review_tools": list(dict.fromkeys(review_tools))[:2],
            "quality_limitations": [
                str(value).strip()
                for value in (parsed.get("quality_limitations") or [])
                if str(value).strip()
                and str(value).strip().lower()
                not in {"none", "none.", "no limitation", "no limitations", "\u65e0"}
            ][:3],
            "independent_candidate_count": independent_candidate_count,
            "rule_second_opinion_version": self._compact_rule_second_opinion.get(
                "version"
            ),
        }
        record.phase_guard_passed = True
        record.phase_sanitizations.extend(notes)
        messages.append(
            self.backend.user_turn(
                "PROGRAM-VALIDATED COMPACT PLAN. This replaces any invalid or "
                "duplicate plan rows. Fill only these disease-pathway nodes:\n"
                + json.dumps(
                    self._compact_model_plan(self._compact_plan),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
        self._tool_selection_audit.append(
            {
                "phase": "plan",
                "workflow": "compact",
                "independent_candidate_count": independent_candidate_count,
                "candidate_count": len(merged_candidates),
                "rule_merge": copy.deepcopy(self._compact_rule_merge_audit),
                "sanitizations": list(notes),
            }
        )

    def _compact_runtime_phase_spec(self, spec: PhaseSpec) -> PhaseSpec:
        """Compile the plan to deterministic prefetches plus one verdict turn."""

        if spec.key != "adjudicate":
            return spec

        compact_ceiling = DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_tool_ceiling
        routing_audit: list[dict[str, Any]] = []
        required_intervals = sorted(self._required_component_intervals())
        prefetches: list[tuple[str, dict[str, Any]]] = []
        signatures: set[str] = set()
        model_visible_signatures: set[str] = set()

        def add_view(
            tool: str,
            arguments: Mapping[str, Any],
            *,
            coverage_id: str,
            required: bool,
            model_required: bool,
        ) -> bool:
            normalized = copy.deepcopy(dict(arguments))
            signature = self.registry.call_signature(tool, normalized)
            if signature in signatures:
                if model_required:
                    model_visible_signatures.add(signature)
                routing_audit.append(
                    {
                        "coverage_id": coverage_id,
                        "tool": tool,
                        "arguments": normalized,
                        "required": required,
                        "consumer": "model" if model_required else "program",
                        "deduplicated": True,
                    }
                )
                return True
            if len(prefetches) >= compact_ceiling:
                routing_audit.append(
                    {
                        "coverage_id": coverage_id,
                        "tool": tool,
                        "arguments": normalized,
                        "required": required,
                        "omitted_by_cap": True,
                    }
                )
                return False
            signatures.add(signature)
            if model_required:
                model_visible_signatures.add(signature)
            prefetches.append((tool, normalized))
            routing_audit.append(
                {
                    "coverage_id": coverage_id,
                    "tool": tool,
                    "arguments": normalized,
                    "required": required,
                    "consumer": "model" if model_required else "program",
                    "deduplicated": False,
                }
            )
            return True

        # Limited interval context is an independent reporting requirement.
        for interval in required_intervals:
            add_view(
                "get_interval_waveform_context",
                {
                    "interval": "qt" if interval == "QT_QTc" else "pr",
                    "include_beat_flags": False,
                },
                coverage_id=f"required_interval:{interval}",
                required=True,
                model_required=True,
            )

        # Exact pathway views are compiled candidate by candidate.  The plan
        # was already bounded by the same signature accounting, so every path
        # node must fit.  A mismatch is audited and fails loudly in tests
        # instead of creating a misleading unknown node.
        missing_required_views: list[str] = []
        candidates = [
            row
            for row in (self._compact_plan.get("candidates") or [])
            if isinstance(row, Mapping)
        ]
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "candidate")
            for tool, arguments, step_id in self._compact_candidate_pathway_calls(
                candidate
            ):
                pathway = candidate.get("diagnostic_pathway")
                pathway = pathway if isinstance(pathway, Mapping) else {}
                step_owner = next(
                    (
                        str(step.get("owner") or "model")
                        for step in pathway.get("steps") or []
                        if isinstance(step, Mapping)
                        and str(step.get("id") or "") == step_id
                    ),
                    "model",
                )
                if not add_view(
                    tool,
                    arguments,
                    coverage_id=f"{candidate_id}:pathway:{step_id}",
                    required=True,
                    model_required=step_owner != "program",
                ):
                    missing_required_views.append(f"{candidate_id}:{step_id}")

        # Free-form planning checks are used only to construct a fallback path
        # for an uncatalogued family. Once a fixed path exists, replaying those
        # checks would duplicate evidence and dilute the 8K model packet.
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "candidate")
            for index, check in enumerate(candidate.get("checks") or []):
                if not isinstance(check, Mapping):
                    continue
                tool = str(check.get("tool") or "")
                if tool not in compact_diagnostic_prompts.COMPACT_DIRECT_TOOLS:
                    continue
                routing_audit.append(
                    {
                        "coverage_id": (
                            f"{candidate_id}:{check.get('purpose') or 'review'}:"
                            f"{index + 1}"
                        ),
                        "tool": tool,
                        "required": False,
                        "superseded_by_fixed_pathway": True,
                    }
                )
        for index, tool in enumerate(
            (self._compact_plan.get("review_tools") or []) if not candidates else []
        ):
            name = str(tool)
            if name not in compact_diagnostic_prompts.COMPACT_DIRECT_TOOLS:
                continue
            add_view(
                name,
                self._compact_prefetch_arguments(name, candidates),
                coverage_id=f"domain_review:{index + 1}",
                required=False,
                model_required=True,
            )

        if not prefetches:
            add_view(
                "get_rhythm_profile",
                {"sections": ["background", "atrial_signal"]},
                coverage_id="fallback:rhythm",
                required=False,
                model_required=True,
            )
            add_view(
                "get_qrs_measurement_bundle",
                {"leads": ["I", "V1", "V5", "V6"]},
                coverage_id="fallback:morphology",
                required=False,
                model_required=True,
            )

        budget = len(prefetches)
        prefetch_tool_calls = tuple(
            (name, tuple(arguments.items()))
            for name, arguments in prefetches
        )
        program_only_prefetch_tool_calls = tuple(
            (name, tuple(arguments.items()))
            for name, arguments in prefetches
            if self.registry.call_signature(name, arguments)
            not in model_visible_signatures
        )
        required_tool_calls = tuple(
            (name, tuple(arguments.items()), 1)
            for name, arguments in prefetches
        )
        runtime = replace(
            spec,
            tool_budget=budget,
            minimum_tool_calls=0,
            allowed_tools=tuple(dict.fromkeys(name for name, _ in prefetches)),
            coverage_requirements=(),
            required_tool_calls=required_tool_calls,
            prefetch_tool_calls=prefetch_tool_calls,
            program_only_prefetch_tool_calls=program_only_prefetch_tool_calls,
            response_schema=self._compact_runtime_verdict_schema(candidates),
        )
        self._runtime_phase_specs[spec.key] = runtime
        self._tool_selection_audit.append(
            {
                "phase": spec.key,
                "workflow": "compact",
                "configured_budget": spec.tool_budget,
                "effective_budget": budget,
                "exposed_tools": [],
                "program_prefetch": [
                    {"tool": name, "arguments": arguments}
                    for name, arguments in prefetches
                ],
                "program_only_prefetch": [
                    {"tool": name, "arguments": dict(arguments)}
                    for name, arguments in program_only_prefetch_tool_calls
                ],
                "plan_routing": routing_audit,
                "required_interval_views": list(required_intervals),
                "missing_required_pathway_views": missing_required_views,
            }
        )
        if missing_required_views:
            self._emit(
                "[adjudicate] ERROR missing required compact pathway views: "
                + ",".join(missing_required_views)
            )
        self._emit(
            "[adjudicate] compact prefetch="
            + ",".join(name for name, _ in prefetches)
            + f"; calls={budget}/{compact_ceiling}; model_tool_rounds=0"
        )
        return runtime

    def _compact_gate_allows_category(self, category: str) -> bool:
        """Apply partial quality scope before a compact positive can be emitted."""

        if str(self._quality_gate.get("state") or "pass").lower() != "partial":
            return True
        allowed = {
            str(value) for value in (self._quality_gate.get("allowed_domains") or [])
        }
        if allowed and "all" not in allowed:
            allowed_categories: set[str] = set()
            if "quality" in allowed:
                allowed_categories.add("quality")
            if "rhythm" in allowed:
                allowed_categories.update(("rhythm", "rate"))
            if "ectopy" in allowed:
                allowed_categories.add("ectopy")
            if category not in allowed_categories:
                return False
        suppressed = {
            str(value) for value in (self._quality_gate.get("suppressed_domains") or [])
        }
        suppressed_map = {
            "axis": {"axis"},
            "axis_quantitative": {"axis"},
            "intervals": {"interval"},
            "conduction_preexcitation": {"conduction"},
            "voltage_chamber_r_progression": {"chamber"},
            "q_st_t_u": {"ischemia_repolarization"},
            "pacing_high_risk": {"pacing"},
        }
        return category not in {
            item
            for domain in suppressed
            for item in suppressed_map.get(domain, set())
        }

    def _expand_compact_verdict_v2(self, compact: Mapping[str, Any]) -> dict[str, Any]:
        """Expand Qwen's bounded decision into the full audited report contract."""

        plan_by_id = {
            str(row.get("id") or ""): row
            for row in (self._compact_plan.get("candidates") or [])
            if isinstance(row, Mapping) and row.get("id")
        }
        new_pointers = {
            pointer
            for call in self.registry.calls
            if call.ok and call.phase == "adjudicate"
            for pointer in call.visible_citations
        }
        diagnoses: list[dict[str, Any]] = []
        differentials: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        audit_rows: list[dict[str, Any]] = []
        for index, raw in enumerate(compact.get("decisions") or []):
            if not isinstance(raw, Mapping):
                continue
            candidate_id = str(raw.get("id") or "")
            code = str(raw.get("code") or "")
            planned = plan_by_id.get(candidate_id)
            if (
                planned is None
                or str(planned.get("code") or "") != code
                or candidate_id in seen_ids
                or code not in diagnostic_prompts.DIAGNOSIS_CATALOG
            ):
                audit_rows.append(
                    {"index": index, "accepted": False, "reason": "not_in_validated_plan"}
                )
                continue
            seen_ids.add(candidate_id)
            support = self._compact_evidence_items(raw.get("support"))
            counter = self._compact_evidence_items(raw.get("counter"))
            falsification = raw.get("falsification")
            falsification = falsification if isinstance(falsification, Mapping) else {}
            falsification_evidence = self._compact_evidence_items(
                falsification.get("evidence")
            )
            falsification_pointers = {
                str(citation).removeprefix("ev:")
                for item in falsification_evidence
                for citation in item.get("citations") or []
            }
            category = diagnostic_prompts.DIAGNOSIS_CATALOG[code][1]
            requested_placement = str(raw.get("placement") or "unresolved")
            challenge_passed = bool(
                falsification.get("outcome") == "not_refuted"
                and falsification.get("direction") == "supports"
                and falsification_pointers.intersection(new_pointers)
            )
            confirmed = bool(
                requested_placement == "confirmed"
                and raw.get("confidence") in {"HIGH", "MEDIUM"}
                and support
                and challenge_passed
                and self._compact_gate_allows_category(category)
            )
            audit_rows.append(
                {
                    "id": candidate_id,
                    "code": code,
                    "requested_placement": requested_placement,
                    "final_placement": (
                        "confirmed"
                        if confirmed
                        else "rejected"
                        if requested_placement == "rejected"
                        else "differential"
                    ),
                    "challenge_passed": challenge_passed,
                    "quality_scope_allowed": self._compact_gate_allows_category(category),
                }
            )
            statement = str(raw.get("statement") or "This ECG candidate requires waveform review.")
            reasoning = str(raw.get("reasoning") or "The available evidence still requires human synthesis.")
            if confirmed:
                diagnoses.append(
                    {
                        "code": code,
                        "statement": statement,
                        "category": category,
                        "confidence": str(raw.get("confidence")),
                        "urgency": str(raw.get("urgency") or "NONE"),
                        "evidence": support,
                        "counterevidence": counter,
                        "reasoning": reasoning,
                    }
                )
            elif requested_placement != "rejected":
                differentials.append(
                    {
                        "code": code,
                        "statement": statement,
                        "confidence": (
                            "MEDIUM" if raw.get("confidence") == "MEDIUM" else "LOW"
                        ),
                        "supporting_evidence": support,
                        "counterevidence": [*counter, *falsification_evidence],
                        "what_would_resolve_it": str(
                            raw.get("what_would_resolve_it")
                            or "Independent discriminative waveform evidence is required."
                        ),
                    }
                )

        # A planned candidate cannot disappear silently from adjudication.
        for candidate_id, planned in plan_by_id.items():
            if candidate_id in seen_ids:
                continue
            support = [
                {
                    "claim": "Overview evidence raises this candidate, but targeted adjudication is incomplete.",
                    "value": None,
                    "unit": None,
                    "citations": [f"ev:{pointer}"],
                }
                for token in (planned.get("support") or [])
                if (pointer := self._compact_pointer(token)) is not None
            ]
            differentials.append(
                {
                    "code": str(planned.get("code") or ""),
                    "statement": "This candidate did not complete targeted falsification and cannot be a positive conclusion.",
                    "confidence": "LOW",
                    "supporting_evidence": support,
                    "counterevidence": [],
                    "what_would_resolve_it": "Complete the planned independent support and falsification views and review the original waveforms.",
                }
            )
            audit_rows.append(
                {
                    "id": candidate_id,
                    "code": planned.get("code"),
                    "requested_placement": "omitted",
                    "final_placement": "differential",
                    "challenge_passed": False,
                }
            )

        interval_contexts: list[dict[str, Any]] = []
        for raw in compact.get("interval_contexts") or []:
            if not isinstance(raw, Mapping):
                continue
            interval_contexts.append(
                {
                    "interval": str(raw.get("interval") or ""),
                    "status": str(raw.get("status") or "limited"),
                    "interval_conclusion": str(
                        raw.get("assessment") or "This interval cannot be interpreted reliably as a quantitative measurement."
                    ),
                    "component_waveform_assessment": str(
                        raw.get("assessment") or "The component waveform still requires independent review."
                    ),
                    "residual_evidence": self._compact_evidence_items(
                        raw.get("evidence")
                    ),
                    "interpretive_impact": str(
                        raw.get("impact") or "A missing value cannot establish normality or abnormality."
                    ),
                    "what_would_resolve_it": str(
                        raw.get("what_would_resolve_it") or "Review the component waveform and its boundaries."
                    ),
                }
            )

        limitations = list(
            dict.fromkeys(
                str(value)
                for value in (
                    *(self._compact_plan.get("quality_limitations") or []),
                    *(compact.get("limitations") or []),
                    *(self._quality_gate.get("partial_reasons") or []),
                )
                if str(value).strip()
            )
        )
        review_reasons = [
            str(value)
            for value in (compact.get("human_review_reasons") or [])
            if str(value).strip()
        ] or ["A clinician must review the original 12-lead waveforms."]
        if self.urgent_review_assessment.get("do_not_delay_human_review"):
            review_reasons.insert(0, "A deterministic screening flag requires immediate urgent human waveform review.")

        positive_codes = [row["code"] for row in diagnoses]
        if diagnoses:
            statements = "; ".join(str(row["statement"]) for row in diagnoses)
            summary = "Confirmed positive ECG interpretations: " + statements
            primary = statements
            primary_confidence = (
                "MEDIUM"
                if any(row["confidence"] == "MEDIUM" for row in diagnoses)
                else "HIGH"
            )
        else:
            summary = "No positive diagnosis was confirmed; current candidates remain unconfirmed or insufficiently supported."
            primary = "No positive diagnosis was confirmed; this ECG interpretation remains unconfirmed or limited."
            primary_confidence = "LOW"
        self._compact_decision_audit = {
            "model_overall_status": compact.get("overall_status"),
            "program_overall_status": (
                "confirmed_diagnosis" if diagnoses else "no_confirmed_positive_diagnosis"
            ),
            "decisions": audit_rows,
        }
        return {
            "summary": summary,
            "ranked_complete_interpretations": [
                {
                    "rank": 1,
                    "interpretation_type": "PRIMARY",
                    "complete_diagnosis": primary,
                    "confidence": primary_confidence,
                    "basis_codes": positive_codes,
                    "key_uncertainty": (
                        limitations[0]
                        if limitations
                        else "Human confirmation against the original waveforms and clinical data remains required."
                    ),
                }
            ],
            "diagnoses": diagnoses,
            "differential_diagnoses": differentials,
            "interval_measurement_contexts": interval_contexts,
            "abstentions": [],
            "quality_assessment": {
                "interpretability": (
                    "limited"
                    if limitations
                    or str(self._quality_gate.get("state") or "pass") == "partial"
                    else "adequate"
                ),
                "limitations": limitations,
            },
            "human_review": {"required": True, "reasons": review_reasons},
        }

    def _normalize_verdict_v2(self, verdict: dict[str, Any]) -> dict[str, Any]:
        if self._compact_workflow and "overall_status" in verdict:
            verdict = self._expand_compact_verdict(verdict)
        return super()._normalize_verdict(verdict)

    def _postprocess_phase(
        self,
        result: AgentResult,
        *,
        record: PhaseRecord,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> None:
        """Insert knowledge after survey, then discard excerpts after planning."""

        if self._compact_workflow:
            if record.key == "plan":
                self._capture_compact_plan(record, messages)
                result._trace_intermediate_contexts.append(
                    {
                        "stage": "program_validated_compact_plan",
                        "text": json.dumps(
                            self._compact_plan,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
            elif record.key == "adjudicate":
                runtime_spec = self._runtime_phase_specs.get(record.key)
                if runtime_spec is not None and record.phase_guard_passed is not False:
                    missing = self._unmet_runtime_coverage(runtime_spec)
                    if missing:
                        record.phase_guard_passed = False
                        record.phase_guard_problems.extend(missing)
            return

        runtime_spec = self._runtime_phase_specs.get(record.key)
        if runtime_spec is not None and record.phase_guard_passed is not False:
            missing = self._unmet_runtime_coverage(runtime_spec)
            if missing:
                record.phase_guard_passed = False
                record.phase_guard_problems.extend(missing)

        self._audit_phase_state(result, record=record, messages=messages)
        if (
            record.key in {"survey", "hypothesize", "investigate", "challenge"}
            and record.phase_guard_passed is not False
            and not self._backend_capabilities.phase_memory
            and isinstance(self._latest_phase_state, Mapping)
        ):
            ledger_prefix = "PROGRAM-OWNED DIAGNOSTIC LEDGER SNAPSHOT."
            messages[:] = [
                message
                for message in messages
                if not str(message.get("content") or "").startswith(ledger_prefix)
            ]
            messages.append(
                self.backend.user_turn(
                    ledger_prefix + " This is the sole "
                    "current state; prior model patches are audit history only:\n"
                    + json.dumps(
                        self._latest_phase_state,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            )

        if record.key == "survey" and self.knowledge_guidance:
            from ..knowledge import KnowledgeNavigator

            self._emit("[knowledge_navigate] retrieving provisional-diagnosis guidance")
            try:
                navigator = KnowledgeNavigator(
                    self.knowledge_base or _shared_knowledge_base(),
                    max_chunks=min(
                        self.knowledge_max_chunks,
                        DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.navigation_ceiling,
                    ),
                )
                survey_call_summaries = [
                    call.result_summary
                    for call in self.registry.calls
                    if call.phase == "survey" and call.ok and call.result_summary
                ]
                # The model's survey synthesis is the primary query. Compact
                # tool-result summaries provide a deterministic fallback when
                # a weaker model ends the phase with only "survey complete".
                # They are used for retrieval only and are not copied into the
                # knowledge payload.
                prompt, audit = navigator.navigate(
                    "\n".join([record.text, *survey_call_summaries])
                )
                audit["survey_tool_summaries_used"] = len(
                    survey_call_summaries
                )
            except Exception as exc:
                prompt = (
                    "INTERIM MEDICAL KNOWLEDGE NAVIGATION: retrieval failed. "
                    "Form a provisional differential from the surveyed ecgfeat "
                    "observations and request targeted patient measurements. "
                    "General knowledge is not patient evidence."
                )
                audit = {
                    **self._knowledge_navigation_audit,
                    "status": "error",
                    "reference_count": 0,
                    "references": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._knowledge_navigation_audit.update(audit)
            result.knowledge_navigation = dict(self._knowledge_navigation_audit)
            result._trace_intermediate_contexts.append(
                {
                    "stage": "knowledge_navigation_after_survey",
                    "text": prompt,
                }
            )
            turn = self.backend.user_turn(prompt)
            self._knowledge_navigation_turn = turn
            messages.append(turn)
            self._emit(
                "[knowledge_navigate] ready: "
                f"{self._knowledge_navigation_audit.get('reference_count', 0)} "
                "general reference(s); patient_evidence=false"
            )
            return

        if record.key == "hypothesize" and self._knowledge_navigation_turn is not None:
            # Preserve the model's provisional hypothesis/plan as phase memory,
            # but do not carry the full retrieved excerpts through every later
            # tool turn. The selected tool names are already captured in the
            # structured state above.
            navigation_turn = self._knowledge_navigation_turn
            messages[:] = [
                message for message in messages if message is not navigation_turn
            ]
            self._knowledge_navigation_turn = None
            self._knowledge_navigation_audit["prompt_removed_after_planning"] = True
            result.knowledge_navigation = dict(self._knowledge_navigation_audit)
            self._emit(
                "[knowledge_navigate] excerpts removed; provisional plan retained"
            )
            return

        if record.key == "challenge" and record.phase_guard_passed is not False:
            skeleton = self._diagnostic_ledger.synthesis_instruction()
            messages.append(self.backend.user_turn(skeleton))
            result._trace_intermediate_contexts.append(
                {"stage": "frozen_diagnostic_ledger_before_synthesis", "text": skeleton}
            )
            self._emit("[ledger] frozen; deterministic verdict placement injected")

    def _unmet_runtime_coverage(self, spec: PhaseSpec) -> list[str]:
        """Turn named evidence coverage into a hard, provider-independent gate."""

        phase_calls: list[Any] = []
        seen: set[str] = set()
        for call in self.registry.phase_calls:
            if not call.ok:
                continue
            signature = self.registry.call_signature(call.tool, call.args)
            if signature in seen:
                continue
            seen.add(signature)
            phase_calls.append(call)
        eligible = phase_calls
        if spec.require_novel_tool_views:
            current_ids = {id(call) for call in self.registry.phase_calls}
            prior = {
                self.registry.call_signature(call.tool, call.args)
                for call in self.registry.calls
                if id(call) not in current_ids and call.ok
            }
            eligible = [
                call
                for call in phase_calls
                if self.registry.call_signature(call.tool, call.args) not in prior
            ]
        problems: list[str] = []
        for coverage_id, tool_names, required in spec.coverage_requirements:
            count = sum(1 for call in eligible if call.tool in set(tool_names))
            if count < required:
                problems.append(
                    f"coverage `{coverage_id}` incomplete: needs {required - count} "
                    "additional distinct measurement view(s)"
                )
        for name, arguments, required in spec.required_tool_calls:
            expected = dict(arguments)
            count = sum(
                1
                for call in phase_calls
                if call.tool == name
                and all(call.args.get(key) == value for key, value in expected.items())
            )
            if count < required:
                problems.append(
                    f"required measurement `{name}{expected}` incomplete"
                )
        return problems

    def _runtime_phase_spec(
        self,
        spec: PhaseSpec,
        result: AgentResult,
    ) -> PhaseSpec:
        """Expose only tools justified by the preceding hypothesis state."""

        if self._compact_workflow:
            return self._compact_runtime_phase_spec(spec)

        if spec.key not in {"investigate", "challenge"}:
            return spec
        plan = build_tool_plan(
            self._latest_phase_state,
            survey_state=self._survey_state,
            include_targeted_checks=(spec.key == "investigate"),
            repair_survey_domains=(spec.key == "investigate"),
            max_tools=DEFAULT_DIAGNOSTIC_PROTOCOL.phases.max_planned_tools,
        )
        selected = plan.tools
        planned_views = plan.planned_views
        required_interval_views: list[str] = []
        if spec.key == "investigate":
            existing_pointers = [
                str(row.get("pointer") or "")
                for row in self._diagnostic_ledger.evidence.values()
                if isinstance(row, Mapping) and row.get("pointer")
            ]
            for interval in sorted(
                self._required_component_intervals()
            ):
                if any(
                    self._is_direct_component_pointer(interval, pointer)
                    for pointer in existing_pointers
                ):
                    continue
                required_interval_views.append(
                    "qt" if interval == "QT_QTc" else "pr"
                )
            if required_interval_views:
                selected = tuple(
                    dict.fromkeys(("get_interval_waveform_context", *selected))
                )[: DEFAULT_DIAGNOSTIC_PROTOCOL.phases.max_planned_tools]
        if not selected:
            selected = (
                ("get_morphology_map", "get_measurement", "search_measurements")
                if spec.key == "challenge"
                else (
                    "get_measurement",
                    "search_measurements",
                    "get_morphology_map",
                )
            )
        selected_set = set(selected)
        active_hypotheses = [
            row
            for row in (
                self._latest_phase_state.get("hypotheses") or []
                if isinstance(self._latest_phase_state, Mapping)
                else []
            )
            if isinstance(row, Mapping)
            and str(row.get("status") or "") not in {"rejected", "abstain"}
        ]

        def hypothesis_tools(row: Mapping[str, Any]) -> tuple[str, ...]:
            requested = [str(name) for name in (row.get("next_tools") or [])]
            domain_tools = [
                name
                for domain in (row.get("domains") or [])
                for name in DOMAIN_TOOL_MAP.get(str(domain), ())
            ]
            covered = tuple(
                name
                for name in dict.fromkeys((*requested, *domain_tools))
                if name in selected_set
            )
            return covered or tuple(selected)

        coverage_requirements: list[tuple[str, tuple[str, ...], int]] = []
        if spec.key == "investigate":
            for index, row in enumerate(active_hypotheses):
                hypothesis_id = str(
                    row.get("hypothesis_id")
                    or row.get("id")
                    or f"candidate_{index + 1}"
                )
                coverage_requirements.append(
                    (f"hypothesis:{hypothesis_id}", hypothesis_tools(row), 1)
                )
            for domain in plan.repaired_domains:
                tools = tuple(
                    name
                    for name in DOMAIN_TOOL_MAP.get(domain, ())
                    if name in selected_set
                )
                if tools:
                    coverage_requirements.append(
                        (f"survey_caveat:{domain}", tools, 1)
                    )
        else:
            leading_id = ""
            for row in active_hypotheses:
                if str(row.get("status") or "") == "supported":
                    leading_id = str(
                        row.get("hypothesis_id") or row.get("id") or ""
                    )
                    break
            if not leading_id and active_hypotheses:
                leading_id = str(
                    active_hypotheses[0].get("hypothesis_id")
                    or active_hypotheses[0].get("id")
                    or ""
                )
            for index, row in enumerate(active_hypotheses):
                hypothesis_id = str(
                    row.get("hypothesis_id")
                    or row.get("id")
                    or f"candidate_{index + 1}"
                )
                tools = hypothesis_tools(row)
                if hypothesis_id != leading_id:
                    coverage_requirements.append(
                        (f"falsify:{hypothesis_id}", tools, 1)
                    )
                    continue
                requested = tuple(
                    name
                    for name in (str(value) for value in (row.get("next_tools") or []))
                    if name in tools
                )
                primary = requested or tools[:1]
                orthogonal = tuple(name for name in tools if name not in set(primary))
                if orthogonal:
                    coverage_requirements.extend(
                        (
                            (f"falsify:{hypothesis_id}:planned", primary, 1),
                            (f"falsify:{hypothesis_id}:orthogonal", orthogonal, 1),
                        )
                    )
                else:
                    coverage_requirements.append(
                        (f"falsify:{hypothesis_id}", tools, 2)
                    )

        required_views = sum(row[2] for row in coverage_requirements)
        required_views += len(required_interval_views)
        if required_views:
            absolute_ceiling = (
                DEFAULT_DIAGNOSTIC_PROTOCOL.phases.investigate_ceiling
                if spec.key == "investigate"
                else DEFAULT_DIAGNOSTIC_PROTOCOL.phases.challenge_ceiling
            )
            # Candidate count is learned only after Hypothesize. Allow the
            # runtime ceiling to expand within the protocol's audited absolute
            # cap so named coverage can never be mathematically impossible.
            budget = min(
                absolute_ceiling,
                max(required_views, required_views + 1),
            )
        else:
            budget = 0
        instruction = spec.instruction
        required_tool_calls = spec.required_tool_calls
        if required_interval_views:
            calls = ", ".join(
                f'get_interval_waveform_context(interval="{interval}")'
                for interval in required_interval_views
            )
            instruction += (
                "\n\nDETERMINISTIC INTERVAL COVERAGE: Before finishing, call "
                f"{calls}. These component views are required because the "
                "corresponding interval is limited in this record."
            )
            required_tool_calls = (
                *required_tool_calls,
                *(
                    (
                        "get_interval_waveform_context",
                        (("interval", interval),),
                        1,
                    )
                    for interval in required_interval_views
                ),
            )
        runtime = replace(
            spec,
            instruction=instruction,
            tool_budget=budget,
            minimum_tool_calls=0,
            allowed_tools=tuple(selected),
            coverage_requirements=tuple(coverage_requirements),
            required_tool_calls=required_tool_calls,
        )
        self._runtime_phase_specs[spec.key] = runtime
        audit = {
            "phase": spec.key,
            "configured_budget": spec.tool_budget,
            "effective_budget": budget,
            "minimum_tool_calls": 0,
            "coverage_requirements": [
                {
                    "coverage_id": coverage_id,
                    "tools": list(tool_names),
                    "distinct_views": count,
                }
                for coverage_id, tool_names, count in coverage_requirements
            ],
            "exposed_tools": list(selected),
            "planned_views": planned_views,
            "domains": list(plan.domains),
            "repaired_domains": list(plan.repaired_domains),
            "rejected_tool_names": list(plan.rejected_tools),
            "required_interval_views": list(required_interval_views),
            "complexity_flags": list(_diagnostic_complexity_flags(self.store)),
        }
        self._tool_selection_audit.append(audit)
        self._emit(
            f"[{spec.key}] selective tools={','.join(selected)}; "
            f"budget={budget}/{spec.tool_budget}"
        )
        return runtime

    def _revision_allowed_tool_names(self, feedback: str) -> tuple[str, ...]:
        """Return schema-planned tools plus exact lookup/discovery fallbacks."""

        plan = build_tool_plan(
            self._latest_phase_state,
            survey_state=self._survey_state,
            include_targeted_checks=False,
            max_tools=DEFAULT_DIAGNOSTIC_PROTOCOL.phases.max_planned_tools,
        )
        selected = ["get_measurement", "search_measurements", *plan.tools]
        return tuple(dict.fromkeys(selected))[
            : DEFAULT_DIAGNOSTIC_PROTOCOL.phases.max_planned_tools
        ]

    @staticmethod
    def _candidate_hash(candidate: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _revision_instruction(
        self,
        feedback: str,
        *,
        current_candidate: Mapping[str, Any] | None,
    ) -> str:
        if not isinstance(current_candidate, Mapping):
            return super()._revision_instruction(
                feedback,
                current_candidate=current_candidate,
            )
        return diagnostic_prompts.VERDICT_SECTION_PATCH_INSTRUCTION.format(
            candidate_hash=self._candidate_hash(current_candidate),
            feedback=feedback,
        )

    def _revision_response_schema(
        self,
        *,
        current_candidate: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(current_candidate, Mapping):
            return None
        return diagnostic_prompts.VERDICT_SECTION_PATCH_SCHEMA

    def _apply_verdict_section_patch(
        self,
        candidate: Mapping[str, Any],
        patch: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_hash = self._candidate_hash(candidate)
        if str(patch.get("base_candidate_hash") or "") != expected_hash:
            raise ValueError("stale verdict patch: base_candidate_hash does not match")
        changed = [str(value) for value in (patch.get("changed_sections") or [])]
        if not changed or len(set(changed)) != len(changed):
            raise ValueError("changed_sections must contain unique section names")
        allowed = set(diagnostic_prompts.OUTPUT_SCHEMA["required"])
        if any(section not in allowed for section in changed):
            raise ValueError("verdict patch names an unknown top-level section")
        replacements = patch.get("replacement_sections")
        if not isinstance(replacements, Mapping):
            raise ValueError("replacement_sections must be an object")
        if set(str(key) for key in replacements) != set(changed):
            raise ValueError(
                "replacement_sections keys must exactly equal changed_sections"
            )
        merged = copy.deepcopy(dict(candidate))
        for section in changed:
            merged[section] = copy.deepcopy(replacements[section])
        return merged

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
        """Repair an existing verdict with an atomic top-level section patch."""

        if not isinstance(current_candidate, Mapping):
            return super()._request_revision(
                messages,
                tools,
                feedback,
                result,
                reason=reason,
                current_candidate=current_candidate,
                measurement_reread=measurement_reread,
            )
        current_feedback = feedback
        while result.revisions < self.max_revisions:
            result.revisions += 1
            self._emit(
                f"[revise] {reason}; section patch "
                f"{result.revisions}/{self.max_revisions}"
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
            try:
                parsed = json.loads(str(revised.text or ""))
                if not isinstance(parsed, Mapping):
                    raise ValueError("revision patch is not a JSON object")
                normalized_patch = self._normalized_phase_patch(parsed)
                merged = self._apply_verdict_section_patch(
                    current_candidate,
                    normalized_patch,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                current_feedback = (
                    "The previous section patch was rejected atomically: "
                    f"{type(exc).__name__}: {exc}. Return a corrected section "
                    "patch against the unchanged current candidate."
                )
                reason = "invalid verdict section patch"
                continue
            revised.phase_sanitizations.append(
                "applied verdict section patch: "
                + ", ".join(normalized_patch.get("changed_sections") or [])
            )
            return self._normalize_verdict(merged)
        return None

    def _semantic_phase_state(self, text: str) -> str:
        """Compare typed mutations, ignoring cosmetic phase-summary prose."""

        try:
            parsed = json.loads(str(text or ""))
        except (json.JSONDecodeError, TypeError):
            return " ".join(str(text or "").split())
        if not isinstance(parsed, Mapping):
            return " ".join(str(text or "").split())
        normalizer = getattr(self.backend, "normalize_verdict", None)
        if callable(normalizer):
            normalized = normalizer(dict(parsed))
            if isinstance(normalized, Mapping):
                parsed = normalized

        semantic = {
            key: parsed.get(key) or []
            for key in (
                "create_hypotheses",
                "evidence_updates",
                "domain_updates",
                "transitions",
                "plan_updates",
                "refuting_tests",
                "targeted_checks",
            )
        }
        if isinstance(parsed.get("domains"), Mapping):
            semantic["domains"] = parsed.get("domains")
        return json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _audit_phase_state(
        self,
        result: AgentResult,
        *,
        record: PhaseRecord,
        messages: list[dict[str, Any]],
    ) -> None:
        """Record whether each clinical phase actually updated its state."""
        tracked = {"survey", "hypothesize", "investigate", "challenge"}
        if record.key not in tracked:
            return
        parsed: Mapping[str, Any] | None = None
        try:
            candidate = json.loads(record.text)
            if isinstance(candidate, dict):
                parsed = candidate
        except (json.JSONDecodeError, TypeError):
            pass
        transitions = parsed.get("transitions") if parsed is not None else None
        mutations = sum(
            len(parsed.get(key) or [])
            for key in (
                "create_hypotheses",
                "evidence_updates",
                "domain_updates",
                "transitions",
                "plan_updates",
                "refuting_tests",
            )
        ) if parsed is not None else 0
        # Reported for a phase that passed as well: a problem left to
        # deterministic remediation still changed the committed state, and the
        # model has to see that its support was downgraded rather than kept.
        guard_problems = list(record.phase_guard_problems)
        entry: dict[str, Any] = {
            "phase": record.key,
            "structured_state": parsed is not None,
            "declared_state_changes": (
                len(transitions) if isinstance(transitions, list) else 0
            ),
            "declared_mutations": mutations,
            "status": "initial" if len(result.phases) < 2 else "updated",
            "guard_passed": record.phase_guard_passed is not False,
            "guard_problems": list(guard_problems),
        }
        if len(result.phases) >= 2:
            previous = result.phases[-2]
            unchanged = mutations == 0 and record.tool_calls > 0
            entry.update({"previous_phase": previous.key, "unchanged": unchanged})
            if unchanged:
                entry["status"] = "unchanged"
                warning = (
                    f"STATE UPDATE REQUIRED: phase `{record.key}` read patient "
                    "evidence but committed no typed ledger mutation. In the next "
                    "phase, explicitly add support/counterevidence or explain the "
                    "unchanged status through a valid refuting test."
                )
                messages.append(self.backend.user_turn(warning))
                result._trace_intermediate_contexts.append(
                    {"stage": f"unchanged_state_after_{record.key}", "text": warning}
                )
                self._emit(f"[{record.key}] warning: unchanged phase state")
        if guard_problems:
            warning = (
                f"PHASE EVIDENCE GUARD for `{record.key}`: the following "
                "transitions are not established and must not be promoted in "
                "the next phase:\n- "
                + "\n- ".join(guard_problems[:8])
            )
            messages.append(self.backend.user_turn(warning))
            result._trace_intermediate_contexts.append(
                {"stage": f"phase_guard_after_{record.key}", "text": warning}
            )
            entry["status"] = "guard_warning"
            self._emit(
                f"[{record.key}] evidence guard: {len(guard_problems)} warning(s)"
            )
        self._state_update_audit.append(entry)

    def _postprocess_verified_result(
        self,
        result: AgentResult,
        *,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> None:
        if not self.knowledge_challenge or result.verdict is None:
            return

        from ..knowledge import KnowledgeChallenger

        # This copy is made before the separate post-hoc retrieval/review.  It
        # is persisted beside the final verdict so any second-review change is
        # directly auditable.  Earlier navigation, when enabled, is separately
        # recorded in ``knowledge_navigation``.
        result.pre_knowledge_verdict = copy.deepcopy(result.verdict)
        self._emit("[knowledge] starting detached post-hoc challenge")
        challenger = KnowledgeChallenger(
            self.knowledge_base or _shared_knowledge_base(),
            max_chunks=self.knowledge_max_chunks,
        )
        self._check_runtime_budget()
        usage_before_review = self._backend_usage_snapshot()
        try:
            review = challenger.review(result.pre_knowledge_verdict, backend=self.backend)
        except Exception as exc:
            review = {
                "version": "ecg-knowledge-challenge.v1",
                "status": "error",
                "summary": "The knowledge challenge failed; the initially verified diagnosis remains unchanged.",
                "issues": [],
                "references": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self._absorb_external_backend_usage(usage_before_review)
        result.knowledge_review = review
        self._knowledge_challenge_audit.update(
            {
                "status": review.get("status"),
                "issue_count": len(review.get("issues") or []),
                "corpus_fingerprint": review.get("corpus_fingerprint"),
                "runtime_categories": review.get("runtime_categories"),
                "excluded_categories": review.get("excluded_categories"),
            }
        )
        feedback = challenger.neutral_revision_feedback(review)
        if not feedback:
            self._emit(
                "[knowledge] completed: no evidence re-read requested"
                if review.get("status") != "error"
                else "[knowledge] failed safely; retained first-pass verdict"
            )
            return

        # The knowledge model and excerpts are not appended to ``messages``.
        # Only neutral questions reach this diagnostic conversation, and at
        # least one successful ecgfeat call is required before a revision can
        # be accepted.
        calls_at_start = len(self.registry.calls)
        result.knowledge_revisions += 1
        result.revisions += 1
        self._emit("[knowledge_revise] re-reading patient measurements")
        spec = PhaseSpec(
            key="knowledge_revise",
            instruction=diagnostic_prompts.VERDICT_SECTION_PATCH_INSTRUCTION.format(
                candidate_hash=self._candidate_hash(result.pre_knowledge_verdict),
                feedback=feedback,
            ),
            tool_budget=4,
            structured=True,
            response_schema=diagnostic_prompts.VERDICT_SECTION_PATCH_SCHEMA,
            max_tokens=5120,
            allowed_tools=self._revision_allowed_tool_names(feedback),
            minimum_tool_calls=1,
        )
        record = self._run_phase(spec, messages, tools)
        result.phases.append(record)
        self._checkpoint(result, status="phase_completed:knowledge_revise")
        successful_rereads = sum(
            1 for call in self.registry.calls[calls_at_start:] if call.ok
        )
        revision_audit: dict[str, Any] = {
            "attempted": True,
            "applied": False,
            "successful_measurement_calls": successful_rereads,
            "phase_stop_reason": record.stop_reason,
        }
        try:
            parsed_patch = json.loads(str(record.text or ""))
            if not isinstance(parsed_patch, Mapping):
                raise ValueError("knowledge revision patch is not an object")
            normalized_patch = self._normalized_phase_patch(parsed_patch)
            parsed = self._apply_verdict_section_patch(
                result.pre_knowledge_verdict,
                normalized_patch,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            parsed = None
            revision_audit["rejection_reason"] = (
                f"invalid atomic section patch: {type(exc).__name__}: {exc}"
            )
        if parsed is None:
            pass
        elif successful_rereads < 1:
            revision_audit["rejection_reason"] = (
                "knowledge-driven revision made no successful ecgfeat call"
            )
        else:
            candidate = self._normalize_verdict(parsed)
            structural = self._validate_verdict_contract(candidate)
            if structural:
                revision_audit["rejection_reason"] = (
                    "revised verdict failed deterministic contract"
                )
                revision_audit["contract_problems"] = list(structural[:12])
            else:
                verification = self._verify(candidate)
                if verification.passed:
                    result.verdict = candidate
                    result.verification = verification
                    revision_audit["applied"] = True
                else:
                    revision_audit["rejection_reason"] = (
                        "revised verdict failed patient-evidence verification"
                    )
                    revision_audit["verification_counts"] = verification.counts()
        review["revision"] = revision_audit
        self._knowledge_challenge_audit["revision"] = dict(revision_audit)
        if revision_audit["applied"]:
            self._emit("[knowledge_revise] accepted after evidence verification")
        else:
            self._emit(
                "[knowledge_revise] rejected safely; retained first-pass verdict"
            )

    def _finish(self, result: AgentResult) -> AgentResult:
        finished = super()._finish(result)
        finished.knowledge_navigation = dict(self._knowledge_navigation_audit)
        finished.audit["knowledge_navigation"] = dict(
            self._knowledge_navigation_audit
        )
        finished.audit["knowledge_challenge"] = dict(
            self._knowledge_challenge_audit
        )
        finished.audit["phase_state_updates"] = list(self._state_update_audit)
        finished.audit["diagnostic_ledger"] = self._diagnostic_ledger.audit_state()
        finished.audit["diagnostic_ledger_commits"] = list(
            self._ledger_commit_audit
        )
        finished.audit["dynamic_tool_selection"] = list(
            self._tool_selection_audit
        )
        finished.audit["diagnostic_protocol_config"] = (
            DEFAULT_DIAGNOSTIC_PROTOCOL.to_dict()
        )
        finished.audit["clinical_safety_policy"] = (
            DEFAULT_CLINICAL_SAFETY_POLICY.to_dict()
        )
        finished.audit["diagnostic_gate_validation"] = {
            "version": QUALITY_GATE_VALIDATION_VERSION,
            "state": self._quality_gate.get("state"),
            "source": self._quality_gate.get("source", "validated_artifact"),
            "stop_reasons": list(self._quality_gate.get("stop_reasons") or []),
        }
        finished.audit["deterministic_pathway_policy"] = {
            key: getattr(DEFAULT_DETERMINISTIC_PATHWAY_POLICY, key)
            for key in DEFAULT_DETERMINISTIC_PATHWAY_POLICY.__dataclass_fields__
        }
        finished.audit["urgent_review"] = copy.deepcopy(
            self.urgent_review_assessment
        )
        finished.audit["final_hypothesis_state"] = copy.deepcopy(
            self._latest_phase_state
        )
        finished.audit["diagnostic_workflow"] = self.workflow
        if self._compact_workflow:
            finished.audit["compact_plan"] = copy.deepcopy(self._compact_plan)
            finished.audit["compact_decision"] = copy.deepcopy(
                self._compact_decision_audit
            )
            finished.audit["rule_second_opinion"] = {
                **copy.deepcopy(self._compact_rule_second_opinion),
                "merge": copy.deepcopy(self._compact_rule_merge_audit),
                "patient_evidence": False,
                "injected_after_blind_plan": True,
            }
        return finished


def run_diagnostic_agent(
    features_path: str,
    *,
    backend: LLMBackend,
    record_id: str | None = None,
    max_revisions: int = DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_revisions,
    on_event: Any = None,
    knowledge_guidance: bool = True,
    knowledge_challenge: bool = False,
    knowledge_max_chunks: int = (
        DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.default_max_chunks
    ),
    workflow: str = "compact",
) -> AgentResult:
    store = EvidenceStore.from_path(features_path, record_id=record_id)
    return ECGDiagnosticAgent(
        store=store,
        backend=backend,
        max_revisions=max_revisions,
        on_event=on_event,
        knowledge_guidance=knowledge_guidance,
        knowledge_challenge=knowledge_challenge,
        knowledge_max_chunks=knowledge_max_chunks,
        workflow=workflow,
    ).run()
