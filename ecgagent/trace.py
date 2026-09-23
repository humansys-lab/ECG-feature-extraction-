"""Human-readable, analysis-oriented rendering of one Agent execution trace."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


TRACE_FORMAT_VERSION = "ecgagent.trace.v4"

_PHASE_LABELS = {
    "plan": "Candidate diagnoses and support/falsification plan",
    "survey": "Global measurement survey",
    "hypothesize": "Provisional diagnoses and targeted-check plan",
    "investigate": "Targeted ecgfeat checks",
    "challenge": "Active falsification",
    "synthesize": "Final structured synthesis",
    "revise": "Revision after deterministic validation",
    "knowledge_revise": "Evidence reacquisition after the post-diagnosis knowledge challenge",
    "orient": "Orientation",
    "test": "Testing",
    "adjudicate": "Targeted evidence acquisition, active falsification and final adjudication",
}


def trace_filename(record_id: str) -> str:
    """Return a filesystem-safe filename for a record trace."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record_id or "unknown")).strip(
        "._"
    )
    return f"{safe or 'unknown'}_agent_trace.md"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _compact(value: Any, default: str = "-") -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text or default


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _fenced(lines: list[str], text: Any, language: str = "text") -> None:
    body = str(text or "").rstrip()
    fence = "````" if "```" in body else "```"
    lines.extend([f"{fence}{language}", body or "(empty)", fence])


def _phase_timeline(lines: list[str], phases: Sequence[Mapping[str, Any]]) -> None:
    lines.extend(
        [
            "## Phase Timeline",
            "",
            "| Order | Phase | Meaning | Model turns | Successful tool calls | Rejected calls | State gate | Elapsed | Stop reason |",
            "|---:|---|---|---:|---:|---:|---|---:|---|",
        ]
    )
    if not phases:
        lines.append("| - | - | No phase records | 0 | 0 | 0 | - | 0 s | - |")
        lines.append("")
        return
    for index, phase in enumerate(phases, start=1):
        key = _compact(phase.get("phase"), "unknown")
        guard = phase.get("phase_guard_passed")
        guard_text = (
            "pass"
            if guard is True
            else "failed"
            if guard is False
            else "n/a"
        )
        sanitizations = phase.get("phase_sanitizations")
        if isinstance(sanitizations, Sequence) and not isinstance(
            sanitizations, (str, bytes)
        ) and sanitizations:
            guard_text += f"; sanitized={len(sanitizations)}"
        tool_text = str(int(phase.get("tool_calls") or 0))
        prefetched = int(phase.get("prefetched_tool_calls") or 0)
        if prefetched:
            tool_text += f" (prefetched {prefetched})"
        lines.append(
            f"| {index} | `{key}` | {_PHASE_LABELS.get(key, key)} | "
            f"{int(phase.get('turns') or 0)} | {tool_text} | "
            f"{int(phase.get('rejected_calls') or 0)} | "
            f"{guard_text} | "
            f"{float(phase.get('elapsed_s') or 0):.2f} s | "
            f"{_compact(phase.get('stop_reason'))} |"
        )
    lines.append("")


def _knowledge_navigation(lines: list[str], payload: Mapping[str, Any]) -> None:
    navigation = _mapping(
        payload.get("knowledge_navigation")
        or _mapping(payload.get("audit")).get("knowledge_navigation")
    )
    if not navigation:
        return
    lines.extend(
        [
            "## Knowledge Navigation After Survey",
            "",
            f"- Enabled: `{bool(navigation.get('enabled'))}`",
            f"- Status: `{_compact(navigation.get('status'))}`",
            f"- Retrieved excerpts: {int(navigation.get('reference_count') or 0)}",
            f"- Knowledge is patient evidence: `{bool(navigation.get('patient_evidence'))}`",
            "- Boundary: general knowledge is used only to form provisional diagnoses and check plans; final patient conclusions must return to ecgfeat measurements.",
        ]
    )
    references = _items(navigation.get("references"))
    if references:
        lines.extend(
            [
                "",
                "| Reference | Category | Source | Lines | Title | Retrieval score |",
                "|---|---|---|---:|---|---:|",
            ]
        )
        for row in references:
            line_range = f"{row.get('line_start', '-')}-{row.get('line_end', '-')}"
            lines.append(
                f"| {_compact(row.get('ref'))} | {_compact(row.get('category'))} | "
                f"`{_compact(row.get('source'))}` | {line_range} | "
                f"{_compact(row.get('title'))} | {_compact(row.get('score'))} |"
            )
    lines.append("")


def _model_visible_inputs(
    lines: list[str],
    briefing_text: str | None,
    intermediate_contexts: Sequence[Mapping[str, Any]] | None,
) -> None:
    lines.extend(["## Model-Visible Initial Input and Interphase Context", ""])
    if briefing_text:
        lines.extend(["### Neutral Initial ECG Briefing", ""])
        _fenced(lines, briefing_text, "text")
        lines.append("")
    else:
        lines.extend(
            [
                "The full initial briefing was not saved in this historical result; it can be reconstructed from the input fingerprint and original feature file.",
                "",
            ]
        )
    for index, context in enumerate(intermediate_contexts or [], start=1):
        lines.extend(
            [
                f"### Interphase Context {index} · `{_compact(context.get('stage'))}`",
                "",
            ]
        )
        _fenced(lines, context.get("text"), "text")
        lines.append("")


def _phase_outputs(lines: list[str], phases: Sequence[Mapping[str, Any]]) -> None:
    lines.extend(["## Model Output by Phase", ""])
    if not phases:
        lines.extend(["No phase model output.", ""])
        return
    for index, phase in enumerate(phases, start=1):
        key = _compact(phase.get("phase"), "unknown")
        lines.extend(
            [
                f"### {index}. `{key}` · {_PHASE_LABELS.get(key, key)}",
                "",
                (
                    f"Turns: {int(phase.get('turns') or 0)}; successful tool calls: "
                    f"{int(phase.get('tool_calls') or 0)}; stop reason: "
                    f"`{_compact(phase.get('stop_reason'))}`."
                ),
                "",
            ]
        )
        for attempt in _items(phase.get("response_history")):
            lines.append(f"#### Public response · turn {attempt.get('turn')} · {attempt.get('stop_reason')}")
            lines.append("")
            _fenced(lines, attempt.get("text"), "text")
            if attempt.get("guard_problems"):
                lines.append("Guard problems: " + _json(attempt["guard_problems"]))
            if attempt.get("repair_feedback"):
                _fenced(lines, attempt["repair_feedback"], "text")
            lines.append("")
        lines.extend(["Final committed phase output:", ""])
        _fenced(lines, phase.get("text"), "text")
        lines.append("")


def _tool_calls(
    lines: list[str],
    payload: Mapping[str, Any],
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> None:
    audit = _mapping(payload.get("audit"))
    tools = _mapping(audit.get("tools"))
    calls = list(tool_calls) if tool_calls is not None else _items(tools.get("calls"))
    lines.extend(
        [
            "## ecgfeat Tool-Call Trace",
            "",
            f"Total calls: {len(calls)}; final citable patient pointers: {int(tools.get('whitelist_size') or 0)}.",
            "",
        ]
    )
    if not calls:
        lines.extend(["No tool-call records.", ""])
        return
    for index, call in enumerate(calls, start=1):
        tool = _compact(call.get("tool"), "unknown")
        phase = _compact(call.get("phase"), "unknown")
        full_result = call.get("result_text")
        model_context_result = call.get("model_context_text")
        result_kind = (
            "Program-only result; not sent to model"
            if call.get("program_only")
            else "Model excerpt plus complete raw audit result"
            if model_context_result is not None
            else "Complete model-visible result"
            if full_result is not None
            else "Audit summary"
        )
        lines.extend(
            [
                f"### {index}. `{tool}`",
                "",
                f"- Phase: `{phase}`",
                f"- Successful: `{bool(call.get('ok'))}`",
                f"- Elapsed: {float(call.get('elapsed_ms') or 0):.2f} ms",
                f"- Arguments: `{_json(call.get('args') or {})}`",
                f"- Result detail level: {result_kind}",
            ]
        )
        if model_context_result is not None:
            lines.extend(
                [
                    f"- Original characters: {int(call.get('original_chars') or 0)}",
                    f"- Model-context characters: {int(call.get('model_context_chars') or 0)}",
                    f"- Truncated for model: `{bool(call.get('truncated_for_model'))}`",
                ]
            )
        citations = [str(item) for item in (call.get("citations") or [])]
        visible = [
            str(item) for item in (call.get("visible_citations") or [])
        ]
        rendered_candidates = [
            str(item)
            for item in (call.get("rendered_candidate_citations") or [])
        ]
        atomic_candidates = [
            str(item) for item in (call.get("atomic_view_citations") or [])
        ]
        packed_visible = [
            str(item) for item in (call.get("packed_visible_citations") or [])
        ]
        if citations:
            lines.append(f"- Citations touched internally by the tool: {len(citations)}")
            if rendered_candidates:
                lines.append(
                    f"- Citations actually rendered in the raw tool result: {len(rendered_candidates)}"
                )
                lines.append(f"- Single-tool atomic-view citations: {len(atomic_candidates)}")
                lines.append(f"- Citations after cross-tool compaction: {len(packed_visible)}")
                lines.append(
                    f"- Omitted by atomic budget: {int(call.get('omitted_atom_count') or 0)}"
                )
            lines.append(f"- Cumulative visible and authorized citations in this run: {len(visible)}")
            if visible:
                lines.extend(
                    f"  - `ev:{item.removeprefix('ev:')}`" for item in visible
                )
            hidden = len(set(citations) - set(visible))
            if hidden:
                lines.append(f"- Citations not authorized after rendering/compaction: {hidden}")
        else:
            manifest = _mapping(call.get("citation_manifest"))
            touched_count = int(manifest.get("touched_count") or 0)
            visible_count = int(manifest.get("visible_count") or 0)
            if touched_count:
                lines.extend(
                    [
                        f"- Candidate citations touched by tool: {touched_count}",
                        f"- Model-visible and authorized citations: {visible_count}",
                        "- Pointer manifest: the primary result stores only a set hash; see the separate raw trace for the complete list",
                    ]
                )
            else:
                lines.append("- Citations: none")
        if model_context_result is not None:
            lines.extend(["", "Result excerpt actually seen by the model:", ""])
            _fenced(lines, model_context_result, "text")
            lines.extend(
                [
                    "",
                    "Complete raw result (audit only; not all content necessarily entered model context):",
                    "",
                ]
            )
            _fenced(lines, full_result, "text")
        else:
            lines.extend(["", "Result:", ""])
            _fenced(
                lines,
                full_result if full_result is not None else call.get("result_summary"),
                "text",
            )
        lines.append("")


def _verification(lines: list[str], payload: Mapping[str, Any]) -> None:
    verification = _mapping(payload.get("verification"))
    lines.extend(["## Deterministic Validation and Revision", ""])
    lines.append(f"- Final verified: `{bool(payload.get('verified'))}`")
    lines.append(f"- Standard revisions: {int(payload.get('revisions') or 0)}")
    lines.append(f"- Knowledge-challenge revisions: {int(payload.get('knowledge_revisions') or 0)}")
    if not verification:
        lines.extend(["- No deterministic validation report.", ""])
        return
    lines.extend(
        [
            f"- Patient claims: {int(verification.get('n_claims') or 0)}",
            f"- Citations: {int(verification.get('n_citations') or 0)}",
            f"- Numeric claims: {int(verification.get('n_quantities') or 0)}",
            f"- Supported quantities: {int(verification.get('n_supported_quantities') or 0)}",
            f"- Problem counts: `{_json(verification.get('counts') or {})}`",
        ]
    )
    findings = _items(verification.get("findings"))
    if findings:
        lines.extend(["", "Validation findings:", ""])
        for finding in findings:
            lines.append(
                f"- `{_compact(finding.get('severity'))}` / "
                f"`{_compact(finding.get('code'))}`: "
                f"{_compact(finding.get('detail') or finding.get('excerpt'))}"
            )
    lines.append("")


def _knowledge_review(lines: list[str], payload: Mapping[str, Any]) -> None:
    review = _mapping(payload.get("knowledge_review"))
    if not review:
        return
    lines.extend(
        [
            "## Post-Diagnosis Medical Knowledge Challenge",
            "",
            f"- Status: `{_compact(review.get('status'))}`",
            f"- Summary: {_compact(review.get('summary'))}",
        ]
    )
    for index, issue in enumerate(_items(review.get("issues")), start=1):
        lines.append(
            f"- {index}. `{_compact(issue.get('type'))}` / "
            f"`{_compact(issue.get('priority'))}` · {_compact(issue.get('topic'))}: "
            f"{_compact(issue.get('evidence_review_question'))}"
        )
    revision = _mapping(review.get("revision"))
    if revision:
        lines.append(f"- Revision audit: `{_json(revision)}`")
    lines.append("")


def _runtime(lines: list[str], payload: Mapping[str, Any]) -> None:
    audit = _mapping(payload.get("audit"))
    model = _mapping(audit.get("model"))
    batch = _mapping(payload.get("batch"))
    lines.extend(
        [
            "## Runtime, Model and Context",
            "",
            f"- Agent protocol: `{_compact(audit.get('agent_protocol'))}`",
            f"- Prompt fingerprint: `{_compact(audit.get('prompt_fingerprint'))}`",
            f"- Input fingerprint: `{_compact(audit.get('input_fingerprint'))}`",
            f"- Backend: `{_compact(model.get('backend') or batch.get('backend'))}`",
            f"- Model: `{_compact(batch.get('model') or _mapping(model.get('config')).get('model'))}`",
            f"- Token/call usage: `{_json(model.get('usage') or {})}`",
            f"- Whole-record runtime circuit breaker: `{_json(audit.get('runtime_controls') or {})}`",
            (f"- Total batch runtime: {float(batch['runtime_seconds']):.2f} s"
             if isinstance(batch.get("runtime_seconds"), (int, float))
             else "- Total batch runtime: N/A (not recorded for this run)"),
        ]
    )
    dynamic_tools = audit.get("dynamic_tool_selection")
    if isinstance(dynamic_tools, list) and dynamic_tools:
        lines.append(f"- Dynamic tool routing: `{_json(dynamic_tools)}`")
    attempts = _items(batch.get("attempts"))
    if attempts:
        lines.extend(["", "Runtime attempts:", ""])
        for attempt in attempts:
            lines.append(
                f"- Attempt {int(attempt.get('attempt') or 0)}: "
                f"ok=`{bool(attempt.get('ok'))}`, "
                f"verified=`{bool(attempt.get('verified'))}`, "
                f"elapsed={float(attempt.get('runtime_seconds') or 0):.2f} s, "
                f"error={_compact(attempt.get('error'))}"
            )
    compactions = _items(audit.get("context_compactions"))
    if compactions:
        lines.extend(["", "Interphase context compaction:", ""])
        for row in compactions:
            lines.append(
                f"- `{_compact(row.get('phase'))}`: "
                f"{int(row.get('before_chars') or 0)} → "
                f"{int(row.get('after_chars') or 0)} characters, "
                f"saved {int(row.get('saved_chars') or 0)}."
            )
    lines.append("")


def _final_outcome(lines: list[str], payload: Mapping[str, Any]) -> None:
    verdict = _mapping(payload.get("verdict"))
    lines.extend(["## Final Outcome Index", ""])
    if not verdict:
        lines.extend(["No usable final structured diagnosis.", ""])
        return
    lines.append(f"- Summary: {_compact(verdict.get('summary'))}")
    diagnoses = _items(verdict.get("diagnoses"))
    differentials = _items(verdict.get("differential_diagnoses"))
    lines.append(
        "- Positive diagnoses: "
        + (
            "; ".join(
                f"`{_compact(item.get('code'))}` {_compact(item.get('statement'), '')}"
                for item in diagnoses
            )
            if diagnoses
            else "None"
        )
    )
    lines.append(
        "- Differential diagnoses: "
        + (
            "; ".join(
                f"`{_compact(item.get('code'))}` {_compact(item.get('statement'), '')}"
                for item in differentials
            )
            if differentials
            else "None"
        )
    )
    lines.append("")


def render_agent_trace(
    payload: Mapping[str, Any],
    *,
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
    briefing_text: str | None = None,
    intermediate_contexts: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render one Agent result/audit payload into a standalone Markdown trace."""

    record_id = _compact(payload.get("record_id"), "unknown")
    phases = _items(payload.get("phases"))
    lines = [
        "# ECG Agent Execution Trace",
        "",
        f"- Record: `{record_id}`",
        f"- Trace format: `{TRACE_FORMAT_VERSION}`",
        f"- Run outcome: ok=`{bool(payload.get('ok'))}`; verified=`{bool(payload.get('verified'))}`",
        f"- Error: {_compact(payload.get('error'))}",
        f"- Refusal: {_compact(payload.get('refused'))}",
        "- Note: this trace stores model-visible phase output and ecgfeat tool results. It does not contain vendor-hidden internal reasoning fields.",
        "",
    ]
    _phase_timeline(lines, phases)
    _model_visible_inputs(lines, briefing_text, intermediate_contexts)
    _knowledge_navigation(lines, payload)
    _phase_outputs(lines, phases)
    _tool_calls(lines, payload, tool_calls)
    _verification(lines, payload)
    _knowledge_review(lines, payload)
    _runtime(lines, payload)
    _final_outcome(lines, payload)
    lines.extend(
        [
            "---",
            "",
            "This trace is for research audit and error analysis. It does not replace clinician waveform review or a formal clinical diagnosis.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["TRACE_FORMAT_VERSION", "render_agent_trace", "trace_filename"]
