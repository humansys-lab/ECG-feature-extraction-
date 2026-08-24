"""Strict validation for persisted diagnostic-quality gate artifacts."""
from __future__ import annotations

from typing import Any, Mapping


QUALITY_GATE_VALIDATION_VERSION = "ecgagent.quality-gate-validation.v1"
_LIST_FIELDS = (
    "stop_reasons",
    "partial_reasons",
    "allowed_domains",
    "suppressed_domains",
)


def _invalid_gate(reason: str) -> dict[str, Any]:
    return {
        "state": "stop",
        "stop_reasons": [reason],
        "partial_reasons": [],
        "allowed_domains": [],
        "suppressed_domains": ["all"],
        "source": "fail_closed_validation",
        "validation_version": QUALITY_GATE_VALIDATION_VERSION,
    }


def validate_diagnostic_gate(value: Any) -> dict[str, Any]:
    """Return a normalized gate, synthesizing STOP for incomplete artifacts."""

    if not isinstance(value, Mapping):
        return _invalid_gate("diagnostic_gate_missing_or_not_object")
    state = str(value.get("state") or "").strip().lower()
    if state not in {"pass", "partial", "stop"}:
        return _invalid_gate("diagnostic_gate_state_invalid")

    normalized_lists: dict[str, list[str]] = {}
    for field in _LIST_FIELDS:
        raw = value.get(field)
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            return _invalid_gate(f"diagnostic_gate_{field}_invalid")
        normalized_lists[field] = [item.strip() for item in raw]

    stop_reasons = normalized_lists["stop_reasons"]
    partial_reasons = normalized_lists["partial_reasons"]
    allowed = normalized_lists["allowed_domains"]
    suppressed = normalized_lists["suppressed_domains"]
    if state == "pass" and (
        stop_reasons or partial_reasons or allowed != ["all"] or suppressed
    ):
        return _invalid_gate("diagnostic_gate_pass_contract_invalid")
    if state == "partial" and (
        stop_reasons or not partial_reasons or not allowed
    ):
        return _invalid_gate("diagnostic_gate_partial_contract_invalid")
    if state == "stop" and (not stop_reasons or allowed):
        return _invalid_gate("diagnostic_gate_stop_contract_invalid")

    normalized = dict(value)
    normalized["state"] = state
    normalized.update(normalized_lists)
    normalized["validation_version"] = QUALITY_GATE_VALIDATION_VERSION
    return normalized


__all__ = ["QUALITY_GATE_VALIDATION_VERSION", "validate_diagnostic_gate"]
