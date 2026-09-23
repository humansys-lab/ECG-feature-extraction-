"""Measurement stage: representative, group and global measurement assembly.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..._engine.foundation.numeric import _finite_float, _median_or_none
from ..._engine.measurement.features import (
    _axis_from_amplitudes,
    _low_support_limb_t_axis_coverage_values,
    _st_supported_limb_t_axis_coverage_values,
)
from ..context import PipelineContext, StageResult

_REGULAR_NARROW_PR_CORE_RR_CV_MAX = 0.08
_REGULAR_NARROW_PR_CORE_QRS_MAX_MS = 125.0
_REGULAR_NARROW_PR_CORE_MIN_MS = 120.0
_REGULAR_NARROW_PR_CORE_MAX_MS = 240.0


def _backfill_t_axis_after_measurement_profile(
    global_features: object,
    representative_leads: Dict[str, Any],
    r_locs: np.ndarray,
    fs: int,
) -> bool:
    """Recover stable low-support T axis after 12SL profile fields are attached."""
    if getattr(global_features, "t_axis_deg", None) is not None:
        return False
    qrs_ms = _finite_float(getattr(global_features, "qrs_ms", None))
    values = _low_support_limb_t_axis_coverage_values(
        representative_leads,
        qrs_ms,
        r_locs,
        fs,
        atrial_invalid=False,
    )
    if values is None:
        values = _st_supported_limb_t_axis_coverage_values(
            representative_leads,
            qrs_ms,
            r_locs,
            fs,
            atrial_invalid=False,
        )
    if values is None:
        return False
    confidences = {
        lead: getattr(rep, "params", {}).get("qt_confidence_mean")
        for lead, rep in (representative_leads or {}).items()
    }
    t_axis = _axis_from_amplitudes(values, confidences)
    if t_axis is None:
        return False
    global_features.t_axis_deg = t_axis
    return True



def _build_rhythm_beat_rows(
    beats: List[object],
    beat_features: List[object],
) -> List[Dict[str, object]]:
    qrs_by_beat: Dict[int, List[float]] = {}
    for bf in beat_features:
        qrs_ms = getattr(bf, "qrs_ms", None)
        if qrs_ms is None:
            continue
        qrs_value = float(qrs_ms)
        if not np.isfinite(qrs_value):
            continue
        qrs_by_beat.setdefault(int(getattr(bf, "beat_id")), []).append(qrs_value)

    rows: List[Dict[str, object]] = []
    for beat in beats:
        beat_id = int(getattr(beat, "beat_id"))
        qrs_values = qrs_by_beat.get(beat_id, [])
        rows.append({
            "beat_id": beat_id,
            "paced": bool(getattr(beat, "paced", False)),
            "group_id": int(getattr(beat, "group_id")),
            "qrs_duration_ms": float(np.median(qrs_values)) if qrs_values else None,
            "is_ventricular_ectopic": False,
        })
    return rows



def _delta_evidence(beat_features: List[object]) -> Dict[str, Any]:
    delta_leads = set()
    delta_beat_ids = set()
    confidence_by_lead: Dict[str, float] = {}
    for bf in beat_features:
        if not bool(getattr(bf, "delta_present", False)):
            continue
        lead = str(getattr(bf, "lead"))
        beat_id = int(getattr(bf, "beat_id"))
        delta_leads.add(lead)
        delta_beat_ids.add(beat_id)
        confidence = _finite_float(getattr(bf, "qrs_confidence", None))
        if confidence is not None:
            confidence_by_lead[lead] = max(confidence_by_lead.get(lead, 0.0), confidence)
    return {
        "delta_leads": sorted(delta_leads),
        "delta_beat_ids": sorted(delta_beat_ids),
        "delta_confidence_by_lead": confidence_by_lead,
    }



def _pr_series_ms(beat_features: List[object], n_beats: int) -> List[Optional[float]]:
    by_beat: Dict[int, List[float]] = {}
    for bf in beat_features:
        pr_ms = _finite_float(getattr(bf, "pr_ms", None))
        if pr_ms is None or not (60.0 <= pr_ms <= 600.0):
            continue
        by_beat.setdefault(int(getattr(bf, "beat_id")), []).append(pr_ms)
    return [
        _median_or_none(by_beat.get(beat_id, []))
        for beat_id in range(n_beats)
    ]



def _atrial_events_per_rr(
    atrial_events: List[Dict[str, Any]],
    r_locs: np.ndarray,
) -> List[int]:
    counts = [0 for _ in range(max(0, len(r_locs) - 1))]
    if not counts:
        return counts
    for event in atrial_events:
        sample = _finite_float(event.get("sample"))
        if sample is None:
            continue
        for idx in range(len(r_locs) - 1):
            if float(r_locs[idx]) <= sample < float(r_locs[idx + 1]):
                counts[idx] += 1
                break
    return counts



@dataclass(frozen=True, slots=True)
class MeasurementBundle:
    representative: Mapping[int, Any] = field(default_factory=dict)
    groups: Mapping[int, Any] = field(default_factory=dict)
    global_features: Any = None
    qt_decision: Any = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)


def run(context: PipelineContext) -> StageResult:
    raise NotImplementedError("Independent measurement stage is not migrated; use ecg_measure/ecg_emit with the compatibility engine")
