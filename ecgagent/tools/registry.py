"""Tool registration, dispatch, budgeting and audit.

Two invariants make this layer worth having over plain function calls:

1. Every tool return declares the pointers it touched, while a separate
   model-visible ledger records only pointers whose value-bearing citation
   tokens survive rendering and backend compression.  Only that latter set is
   eligible for downstream verification.
2. Every call is budgeted and logged.  An agent that cannot exhaust its budget
   cannot loop, and a run that is fully logged can be audited afterwards.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..evidence.ledger import (
    EvidenceLedger,
    evidence_set_id,
    evidence_set_summary,
    visible_citations,
)
from ..evidence.model_view import ModelEvidenceView, build_model_evidence_view
from ..evidence.store import EvidenceStore


class ToolBudgetExceeded(RuntimeError):
    """Raised when a phase runs out of tool calls."""


@dataclass
class ToolResult:
    """What a tool hands back to the model, plus what the verifier needs."""

    ok: bool
    text: str
    citations: tuple[str, ...] = ()
    payload: Any = None
    cost: str = "cheap"
    truncated: bool = False
    note: str | None = None

    def render(self) -> str:
        """Text form injected into the model's context."""
        body = self.text if self.ok else f"ERROR: {self.text}"
        # Never append a pointer-only manifest here.  A citation grants
        # provenance only when the value-bearing row itself renders the exact
        # token.  Large results may later be compressed again by a backend.
        if self.note:
            body += f"\n(note: {self.note})"
        if self.truncated:
            body += "\n(output truncated; narrow the request to see more)"
        return body

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "text": self.text,
            "citations": list(self.citations),
            "cost": self.cost,
            "truncated": self.truncated,
            "note": self.note,
        }

    @classmethod
    def error(cls, message: str, note: str | None = None) -> "ToolResult":
        return cls(ok=False, text=message, note=note)

    @classmethod
    def unavailable(cls, message: str) -> "ToolResult":
        """A valid read with no measurement; not an execution/schema error."""
        return cls.error(message, note="measurement_unavailable")


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, with the schema needed to call it."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    cost: str = "cheap"
    required: tuple[str, ...] = ()

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": self.parameters,
            "required": list(self.required),
            "additionalProperties": False,
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema(),
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema(),
            },
        }


@dataclass
class CallRecord:
    call_id: str
    tool: str
    args: dict[str, Any]
    ok: bool
    citations: tuple[str, ...]
    elapsed_ms: float
    phase: str
    visible_citations: tuple[str, ...] = ()
    result_summary: str = ""
    # Kept in memory for the standalone Markdown trajectory. This is the raw
    # tool render; backend context traces separately record the exact compressed
    # model-visible excerpt.
    result_text: str = ""
    result_sha256: str = ""
    error_note: str | None = None
    program_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Compact call metadata for the primary result JSON.

        Pointer manifests belong to :meth:`to_trace_dict`.  Keeping two full
        copies of every touched and visible pointer in the primary result made
        large morphology-table calls dominate the file while saying nothing
        about the values that mattered to the verdict.
        """

        return {
            "call_id": self.call_id,
            "program_only": self.program_only,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "citation_manifest": {
                "touched_count": len(set(self.citations)),
                "touched_evidence_set_id": evidence_set_id(self.citations),
                "visible_count": len(set(self.visible_citations)),
                "visible_evidence_set_id": evidence_set_id(
                    self.visible_citations
                ),
                "hidden_count": max(
                    0,
                    len(set(self.citations) - set(self.visible_citations)),
                ),
            },
            "elapsed_ms": round(self.elapsed_ms, 2),
            "phase": self.phase,
            "result_summary": self.result_summary,
            "result_sha256": self.result_sha256,
            "error_note": self.error_note,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        """Full raw call record for the separate trace document."""

        return {
            **self.to_dict(),
            "citations": list(self.citations),
            "visible_citations": list(self.visible_citations),
            "hidden_citation_count": max(
                0,
                len(set(self.citations) - set(self.visible_citations)),
            ),
            "result_text": self.result_text,
        }


@dataclass
class ToolRegistry:
    """Holds the tool set bound to one EvidenceStore, with budget and audit."""

    store: EvidenceStore
    budget: int | None = None
    specs: dict[str, ToolSpec] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    phase: str = "test"
    deduplicate_within_phase: bool = False
    _ledger: EvidenceLedger = field(default_factory=EvidenceLedger, repr=False)
    _seeded_provenance: dict[str, set[str]] = field(default_factory=dict, repr=False)
    _phase_start: int = field(default=0, repr=False)
    # Exact successful calls are memoized only for the current phase.  A later
    # challenge is allowed to re-read a survey measurement, but asking for the
    # same tool with byte-equivalent normalized arguments twice in one phase
    # must not consume budget or duplicate a large result in model context.
    _phase_success_results: dict[str, ToolResult] = field(
        default_factory=dict,
        repr=False,
    )
    # Guards against a model thrashing on rejected calls forever. Generous
    # because a rejected call costs nothing but a round trip.
    dispatch_ratio: int = 3

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self.specs[spec.name] = spec

    def seed_citations(
        self,
        citations: Any,
        *,
        source: str,
        model_text: str | None = None,
    ) -> None:
        """Whitelist evidence already shown outside a tool-response turn.

        The initial chart briefing is pushed before the first phase and does
        not spend a tool call, but it is still evidence the model actually
        read. Keep that provenance explicit in the audit instead of forcing
        the model to re-read every briefing value solely to satisfy the
        session whitelist.
        """
        resolved: list[str] = []
        for reference in citations:
            evidence = self.store.try_resolve(str(reference))
            if evidence is None:
                raise ValueError(
                    f"{source} tried to seed unresolvable citation {reference!r}"
                )
            resolved.append(evidence.pointer)
        self._ledger.record_touched(resolved)
        shown = visible_citations(model_text or "", resolved)
        seeded = self._seeded_provenance.setdefault(str(source), set())
        seeded.update(shown)
        self._ledger.authorize(shown, source=str(source))

    def authorize_model_visible(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        citations: Any,
        source: str,
    ) -> tuple[str, ...]:
        """Authorize citations after final backend rendering/compaction.

        The loop calls this immediately before the next model request, using
        the citations the backend confirms are present in that exact message.
        """

        accepted: list[str] = []
        for reference in citations:
            evidence = self.store.try_resolve(str(reference))
            if evidence is not None:
                accepted.append(evidence.pointer)
        visible = self._ledger.authorize(accepted, source=source)
        signature = self.call_signature(tool, arguments)
        for record in reversed(self.phase_calls):
            if self.call_signature(record.tool, record.args) != signature:
                continue
            record.visible_citations = tuple(
                dict.fromkeys((*record.visible_citations, *visible))
            )
            break
        return visible

    def authorize_program_evidence(
        self,
        *,
        tool: str,
        citations: Any,
        source: str,
        include_previous_batches: bool = False,
    ) -> tuple[str, ...]:
        """Authorize inputs consumed by a deterministic pathway resolver.

        A program resolver may use the complete bounded tool result even when
        backend context compression hid some rows from the model.  Only
        pointers actually touched by a successful call of ``tool`` in the
        current batch are accepted by default. Compact finalization may include
        earlier batches with the same phase name; other phases remain excluded.
        """

        phase_calls = ([record for record in self.calls if record.phase == self.phase]
                       if include_previous_batches else self.phase_calls)
        touched_by_tool = {
            pointer
            for record in phase_calls
            if record.ok and record.tool == str(tool)
            for pointer in record.citations
        }
        resolved: list[str] = []
        for reference in citations:
            evidence = self.store.try_resolve(str(reference))
            if evidence is not None and evidence.pointer in touched_by_tool:
                resolved.append(evidence.pointer)
        return self._ledger.authorize_program(resolved, source=str(source))

    def evidence_provenance(self, pointer: str) -> dict[str, Any]:
        """Return model-visible and program-deterministic provenance."""

        canonical = str(pointer).removeprefix("ev:")
        records = [
            record
            for record in self.calls
            if record.ok and canonical in record.visible_citations
        ]
        return {
            "tool_call_ids": [record.call_id for record in records],
            "tool_names": list(dict.fromkeys(record.tool for record in records)),
            "tool_phases": list(dict.fromkeys(record.phase for record in records)),
            "seed_sources": [
                source
                for source, pointers in self._seeded_provenance.items()
                if canonical in pointers
            ],
            "program_sources": [
                source
                for source, pointers in self._ledger.program_by_source.items()
                if canonical in pointers
            ],
            "model_visible": canonical in self._ledger.visible,
            "program_authorized": canonical in self._ledger.program_authorized,
        }

    def begin_phase(self, name: str, budget: int | None) -> None:
        """Start a budgeted phase. Budget counts *successful* calls only.

        A call that returned no evidence — an unknown field name, a bad rule id —
        should not consume the evidence budget. The budget exists to bound how
        much evidence is gathered, not to charge for a typo, and charging for
        one pushes the model into synthesising from whatever it managed to read
        before its allowance ran out.
        """
        self.phase = name
        self._phase_start = len(self.calls)
        self.budget = budget
        self._phase_success_results.clear()

    @property
    def phase_calls(self) -> list[CallRecord]:
        return self.calls[self._phase_start :]

    @property
    def phase_successes(self) -> int:
        return sum(1 for record in self.phase_calls if record.ok)

    @staticmethod
    def call_signature(name: str, arguments: dict[str, Any]) -> str:
        """Canonical identity for argument-aware coverage and de-duplication."""

        return str(name) + ":" + json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @property
    def phase_distinct_successes(self) -> int:
        return len(self._phase_success_results)

    # -- dispatch -----------------------------------------------------------
    def call(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        started = time.perf_counter()
        arguments = dict(args or {})
        spec = self.specs.get(name)
        if spec is None:
            known = ", ".join(sorted(self.specs)) or "(none)"
            return self._record(
                name,
                arguments,
                ToolResult.error(f"unknown tool {name!r}. Available: {known}"),
                started,
            )

        signature = self.call_signature(name, arguments)
        previous = self._phase_success_results.get(signature)
        if self.deduplicate_within_phase and previous is not None:
            return self._record(
                name,
                arguments,
                ToolResult(
                    ok=False,
                    text=(
                        "duplicate call suppressed: this exact tool and argument "
                        "set was already read successfully in the current phase. "
                        "Use the earlier evidence ledger and its citations; choose "
                        "a narrower or different view only if another question "
                        "remains unresolved."
                    ),
                    citations=previous.citations,
                    note="duplicate",
                ),
                started,
            )

        if self.budget is not None:
            if self.phase_successes >= self.budget:
                return self._record(
                    name,
                    arguments,
                    ToolResult.error(
                        f"tool budget exhausted ({self.budget} successful calls used). "
                        "Conclude with the evidence already gathered, and state what "
                        "remains unresolved.",
                        note="budget",
                    ),
                    started,
                )
            if len(self.phase_calls) >= self.budget * self.dispatch_ratio:
                return self._record(
                    name,
                    arguments,
                    ToolResult.error(
                        f"too many rejected calls in this phase "
                        f"({len(self.phase_calls)} attempts for {self.phase_successes} "
                        "successes). Read the error messages and stop guessing; "
                        "use search_measurements to find exact field names.",
                        note="thrash",
                    ),
                    started,
                )

        missing = [key for key in spec.required if key not in arguments]
        if missing:
            return self._record(
                name,
                arguments,
                ToolResult.error(
                    f"{name} is missing required argument(s): {', '.join(missing)}"
                ),
                started,
            )
        unknown = [key for key in arguments if key not in spec.parameters]
        if unknown:
            return self._record(
                name,
                arguments,
                ToolResult.error(
                    f"{name} got unknown argument(s): {', '.join(unknown)}. "
                    f"Accepted: {', '.join(spec.parameters)}"
                ),
                started,
            )

        try:
            result = spec.handler(self.store, **arguments)
        except Exception as exc:  # surfaced to the model, never crashes the run
            result = ToolResult.error(f"{type(exc).__name__}: {exc}")
        if result.ok:
            self._phase_success_results.setdefault(signature, result)
        return self._record(name, arguments, result, started)

    def _record(
        self,
        name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        started: float,
    ) -> ToolResult:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._ledger.record_touched(result.citations)
        rendered = result.render()
        compact = " ".join(rendered.split())
        if len(compact) > 500:
            compact = compact[:497] + "..."
        self.calls.append(
            CallRecord(
                call_id=f"T{len(self.calls) + 1:04d}",
                tool=name,
                args=arguments,
                ok=result.ok,
                citations=result.citations,
                elapsed_ms=elapsed_ms,
                phase=self.phase,
                result_summary=compact,
                result_text=rendered,
                result_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                error_note=result.note if not result.ok else None,
            )
        )
        return result

    def call_json(self, raw: str | dict[str, Any]) -> ToolResult:
        """Dispatch from a `{"tool": ..., "args": {...}}` object or its JSON text.

        Gemma-class models have no native tool calling, so the vLLM backend
        drives them with guided JSON in exactly this shape.
        """
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                started = time.perf_counter()
                return self._record(
                    "<invalid_json>",
                    {"raw": raw[:500]},
                    ToolResult.error(f"tool call is not valid JSON: {exc}"),
                    started,
                )
        if not isinstance(raw, dict):
            started = time.perf_counter()
            return self._record(
                "<invalid_call>",
                {"raw_type": type(raw).__name__},
                ToolResult.error("tool call must be a JSON object"),
                started,
            )
        name = raw.get("tool") or raw.get("name")
        if not name:
            started = time.perf_counter()
            return self._record(
                "<missing_tool>",
                dict(raw),
                ToolResult.error('tool call must carry a "tool" field'),
                started,
            )
        args = raw.get("args") or raw.get("input") or raw.get("arguments") or {}
        if not isinstance(args, dict):
            started = time.perf_counter()
            return self._record(
                str(name),
                {"raw_args_type": type(args).__name__},
                ToolResult.error('"args" must be a JSON object'),
                started,
            )
        return self.call(str(name), args)

    def model_evidence_view(
        self,
        result: ToolResult,
        *,
        tool: str,
        arguments: dict[str, Any] | None = None,
        required_by: tuple[str, ...] = (),
    ) -> ModelEvidenceView | None:
        """Return the canonical model/audit view for a successful result.

        Raw renderer text remains in :class:`CallRecord` for human audit.  The
        model receives this atomized view so context budgeting can never split
        a measurement away from its value, unit, reliability or citation.
        """

        if not result.ok:
            return None
        return build_model_evidence_view(
            self.store,
            tool=str(tool),
            arguments=dict(arguments or {}),
            rendered_text=result.render(),
            citations=result.citations,
            required_by=required_by,
        )

    # -- introspection ------------------------------------------------------
    @property
    def whitelist(self) -> frozenset[str]:
        """Pointers authorized by model visibility or deterministic execution."""
        return self._ledger.authorized

    @property
    def model_visible_whitelist(self) -> frozenset[str]:
        """Pointers whose value-bearing atoms were present in model context."""

        return frozenset(self._ledger.visible)

    @property
    def remaining(self) -> int | None:
        """Successful calls still allowed in the current phase."""
        return None if self.budget is None else max(0, self.budget - self.phase_successes)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [spec.to_anthropic() for spec in self.specs.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.to_openai() for spec in self.specs.values()]

    def guided_json_schema(self) -> dict[str, Any]:
        """One schema constraining a model to emit exactly one valid tool call.

        Used with vLLM `guided_json` for backends without native tool calling
        (Gemma has none).  The union is discriminated on the tool name so each
        branch also constrains that tool's arguments; a schema that left `args`
        an open object would only guarantee well-formed JSON, not a callable
        request, and the malformed calls would surface as wasted turns.
        """
        branches = [
            {
                "type": "object",
                "properties": {
                    "tool": {"const": name},
                    "args": spec.input_schema(),
                },
                "required": ["tool", "args"],
                "additionalProperties": False,
            }
            for name, spec in sorted(self.specs.items())
        ]
        return {"anyOf": branches} if branches else {"type": "object"}

    def describe_for_prompt(self) -> str:
        """Compact human/model-readable tool catalogue."""
        lines: list[str] = []
        for spec in self.specs.values():
            params = ", ".join(
                f"{key}{'' if key in spec.required else '?'}"
                for key in spec.parameters
            )
            lines.append(f"- {spec.name}({params})")
            lines.append(f"    {spec.description.strip().splitlines()[0]}")
        return "\n".join(lines)

    def audit(self) -> dict[str, Any]:
        return {
            "format": "tool-audit.compact.v1",
            "full_pointer_manifest": "standalone_agent_trace",
            "record_id": self.store.record_id,
            "n_calls": len(self.calls),
            "budget": self.budget,
            "calls": [record.to_dict() for record in self.calls],
            "duplicate_calls_suppressed": sum(
                1 for record in self.calls if record.error_note == "duplicate"
            ),
            "whitelist_size": len(self._ledger.authorized),
            "model_visible_whitelist_size": len(self._ledger.visible),
            "program_authorized_whitelist_size": len(
                self._ledger.program_authorized
            ),
            "provenance": self._ledger.audit(),
            "seeded_provenance": {
                source: evidence_set_summary(citations)
                for source, citations in sorted(self._seeded_provenance.items())
            },
        }


def build_default_registry(
    store: EvidenceStore,
    budget: int | None = 20,
    *,
    include: tuple[str, ...] | None = None,
) -> ToolRegistry:
    """Registry with the MVP tool set bound to `store`."""
    from . import modalities, orient, query, survey, waveform_review

    registry = ToolRegistry(store=store, budget=budget)
    for spec in (*orient.SPECS, *survey.SPECS, *query.SPECS, *modalities.SPECS, *waveform_review.SPECS):
        if include is None or spec.name in include:
            registry.register(spec)
    return registry
