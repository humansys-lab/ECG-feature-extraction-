"""Deterministic, measurement-only urgent human-review routing.

This module does not emit ECG diagnoses.  It exposes a small set of explicitly
versioned conditions that should be visible to a clinician before the language
model finishes its interpretation.  Categories without a safe deterministic
measurement contract remain listed as unassessed instead of being guessed.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..evidence.store import EvidenceStore
from .safety_policy import DEFAULT_CLINICAL_SAFETY_POLICY


URGENT_REVIEW_POLICY_VERSION = "ecgagent.urgent-review.v1"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def assess_urgent_review(store: EvidenceStore) -> dict[str, Any]:
    """Return a conservative routing flag from already exported measurements."""

    policy = DEFAULT_CLINICAL_SAFETY_POLICY
    triggers: list[dict[str, Any]] = []

    def add(code: str, *pointers: str) -> None:
        triggers.append(
            {
                "code": code,
                "evidence_pointers": list(pointers),
                "patient_evidence": True,
            }
        )

    heart_rate = _number(store.raw("/global_features/heart_rate_bpm", None))
    qrs_ms = _number(store.raw("/global_features/qrs_ms", None))
    if heart_rate is not None and heart_rate < policy.urgent_bradycardia_below_bpm:
        add("extreme_bradycardia_measurement", "/global_features/heart_rate_bpm")
    if heart_rate is not None and heart_rate > policy.urgent_tachycardia_above_bpm:
        add("extreme_tachycardia_measurement", "/global_features/heart_rate_bpm")
        if qrs_ms is not None and qrs_ms >= policy.urgent_wide_qrs_at_least_ms:
            add(
                "wide_complex_tachycardia_measurements",
                "/global_features/heart_rate_bpm",
                "/global_features/qrs_ms",
            )

    if store.raw("/rhythm_inputs/av_block/complete_av_block", False) is True:
        add(
            "complete_av_block_measurement_flag",
            "/rhythm_inputs/av_block/complete_av_block",
        )

    qtc = _number(store.raw("/global_features/qtc_bazett_ms", None))
    qt_reportable = store.raw("/global_features/qt_reportable", None)
    qt_reliability = str(
        store.raw("/global_features/qt_reliability", "unavailable") or "unavailable"
    ).lower()
    if (
        qtc is not None
        and qtc > policy.urgent_qtc_bazett_above_ms
        and qt_reportable is True
        and qt_reliability not in {
            "limited",
            "low_confidence",
            "unreliable",
            "unavailable",
            "fallback",
            "rescued",
        }
    ):
        add(
            "marked_qtc_prolongation_measurement",
            "/global_features/qtc_bazett_ms",
            "/global_features/qt_reportable",
            "/global_features/qt_reliability",
        )

    pacing_base = "/rhythm_inputs/pacing"
    if (
        store.raw(f"{pacing_base}/capture_failure_suspected", False) is True
        and store.raw(f"{pacing_base}/supports_measurement_routing", False) is True
        and store.raw(f"{pacing_base}/evidence_conflicted", True) is False
    ):
        add(
            "pacing_capture_failure_measurement_flag",
            f"{pacing_base}/capture_failure_suspected",
            f"{pacing_base}/supports_measurement_routing",
            f"{pacing_base}/evidence_conflicted",
        )

    gate_state = str(
        store.raw("/metadata/diagnostic_gate/state", "pass") or "pass"
    ).lower()
    technical_stop = gate_state == "stop"
    return {
        "policy_version": URGENT_REVIEW_POLICY_VERSION,
        "safety_policy_version": policy.version,
        "urgent_review_required": bool(triggers or technical_stop),
        "critical_pattern_possible": bool(triggers),
        "do_not_delay_human_review": bool(triggers),
        "technical_quality_stop": technical_stop,
        "triggers": triggers,
        "unassessed_categories": [
            "acute_st_injury_pattern",
            "pacing_sensing_failure",
        ],
        "scope_note": (
            "Routing flag only; it neither confirms a diagnosis nor represents "
            "a complete urgent-ECG detector."
        ),
    }


__all__ = ["URGENT_REVIEW_POLICY_VERSION", "assess_urgent_review"]
