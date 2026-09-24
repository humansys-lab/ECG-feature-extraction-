from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


_APPROXIMATION_LABELS = {
    "public_standard_approximation": "[APPROXIMATION: PUBLIC STANDARD]",
    "existing_dxl_approximation": "[APPROXIMATION: EXISTING DXL]",
    "not_reproducible_from_guide": "[UNAVAILABLE: GUIDE METHOD NOT REPRODUCIBLE]",
}


def _suffix(fidelity: str) -> str:
    return _APPROXIMATION_LABELS.get(str(fidelity), "")


def _display_row(item: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(item)
    suffix = _suffix(str(row.get("fidelity") or ""))
    row["display_statement"] = " ".join(
        part for part in (str(row.get("statement") or ""), suffix) if part
    )
    row["approximation_label"] = suffix
    return row


def _compact(value: Any, limit: int = 180) -> str:
    if value in (None, {}, [], ""):
        return ""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _audit_row(item: Dict[str, Any]) -> Dict[str, Any]:
    missing = item.get("missing_inputs") or []
    suppressed = item.get("suppressed_by") or []
    evidence_parts = []
    if item.get("evidence"):
        evidence_parts.append(f"evidence={_compact(item['evidence'])}")
    if item.get("thresholds"):
        evidence_parts.append(f"thresholds={_compact(item['thresholds'])}")
    if missing:
        evidence_parts.append(f"missing={','.join(str(value) for value in missing)}")
    if suppressed:
        evidence_parts.append(f"suppressed_by={','.join(str(value) for value in suppressed)}")
    if item.get("evaluation_error"):
        evidence_parts.append(f"error={item['evaluation_error']}")
    return {
        **_display_row(item),
        "audit_evidence": "; ".join(evidence_parts) or "no additional evidence",
    }


def _selected_thresholds(evaluations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    for item in evaluations:
        if item.get("thresholds"):
            selected[str(item.get("rule_id"))] = dict(item["thresholds"])
    return selected


def build_display_model(analysis: Dict[str, Any]) -> Dict[str, Any]:
    if analysis.get("schema_version") != "glasgow_rules.v2":
        raise ValueError("Glasgow v2 analysis is required")
    resolution = analysis.get("statement_resolution") or {}
    evaluations = list(analysis.get("rule_evaluations") or [])
    approximations = [
        _display_row(item)
        for item in evaluations
        if item.get("fidelity") in _APPROXIMATION_LABELS
    ]
    pending = [
        chapter
        for chapter, detail in (analysis.get("coverage", {}).get("chapters", {}) or {}).items()
        if detail.get("status") == "pending_phase"
    ]
    limitations: List[str] = [
        "Automated interpretation requires clinician review; this is not certified Glasgow equivalence."
    ]
    if pending:
        limitations.append(f"Guide chapters pending rule-by-rule implementation: {', '.join(pending)}.")
    if approximations:
        limitations.append(
            f"{len(approximations)} registered rules or measurements use explicitly labelled approximations."
        )
    return {
        "schema_version": analysis["schema_version"],
        "summary_code": dict(resolution.get("summary_code") or {}),
        "preliminary": [_display_row(item) for item in resolution.get("preliminary", [])],
        "final_statements": [_display_row(item) for item in resolution.get("final", [])],
        "suppressed": [_display_row(item) for item in resolution.get("suppressed", [])],
        "unavailable": [_display_row(item) for item in resolution.get("unavailable", [])],
        "restricted_analysis": dict(resolution.get("restricted_analysis") or {}),
        "lead_exclusions": dict(resolution.get("lead_exclusions") or {}),
        "audit_rows": [_audit_row(item) for item in evaluations],
        "measurement_matrix": dict(analysis.get("measurement_matrix") or {}),
        "selected_thresholds": _selected_thresholds(evaluations),
        "approximations": approximations,
        "limitations": limitations,
        "config": dict(analysis.get("config") or {}),
        "patient_route": dict(analysis.get("patient_route") or {}),
    }

