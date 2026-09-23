from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, Mapping, Optional

from ..foundation.models import RepresentativeLeadFeatures


HEXAXIAL_ANGLES: Dict[str, float] = {
    "I": 0.0,
    "II": 60.0,
    "III": 120.0,
    "aVR": -150.0,
    "aVL": -30.0,
    "aVF": 90.0,
}
_PROJECTION_NEUTRAL_COS = 0.15


@dataclass
class AxisResult:
    axis_deg: Optional[float]
    used_leads: Dict[str, float] = field(default_factory=dict)
    excluded_leads: Dict[str, str] = field(default_factory=dict)
    reason: Optional[str] = None


def _finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _polarity(value: float) -> int:
    return 1 if value > 0.0 else -1


def _axis_from_signed_values(
    signed_values: Mapping[str, float],
    confidences: Mapping[str, float],
) -> Optional[float]:
    x = 0.0
    y = 0.0
    for lead, value in signed_values.items():
        angle_deg = HEXAXIAL_ANGLES.get(lead)
        if angle_deg is None:
            continue
        confidence = confidences.get(lead, 1.0)
        angle_rad = math.radians(angle_deg)
        x += value * math.cos(angle_rad) * confidence
        y += value * math.sin(angle_rad) * confidence
    if math.hypot(x, y) < 1e-6:
        return None
    return math.degrees(math.atan2(y, x))


def compute_t_axis_from_cluster(
    representative_leads: Mapping[str, RepresentativeLeadFeatures],
    min_leads: int = 2,
    min_abs_amp_mv: float = 0.05,
    min_confidence: float = 0.45,
    allowed_leads: Optional[Iterable[str]] = None,
) -> AxisResult:
    raw_values: Dict[str, float] = {}
    confidences: Dict[str, float] = {}
    excluded: Dict[str, str] = {}
    allowed = set(allowed_leads) if allowed_leads is not None else None

    for lead in HEXAXIAL_ANGLES:
        if allowed is not None and lead not in allowed:
            excluded[lead] = "filtered_by_reporting"
            continue

        rep = representative_leads.get(lead)
        if rep is None:
            excluded[lead] = "missing"
            continue
        if not rep.params.get("reliable_for_t", False):
            excluded[lead] = "not_reliable_for_t"
            continue

        t_amp = _finite_float(rep.params.get("t_amp_mv"))
        if t_amp is None or abs(t_amp) < min_abs_amp_mv:
            excluded[lead] = "low_t_amplitude"
            continue

        confidence = _finite_float(rep.params.get("qt_confidence_mean"))
        if confidence is None or confidence < min_confidence:
            excluded[lead] = "low_t_confidence"
            continue

        signed_area = _finite_float(rep.params.get("t_signed_area"))
        if signed_area is None or abs(signed_area) < 1e-12:
            excluded[lead] = "missing_signed_area"
            continue

        raw_values[lead] = signed_area
        confidences[lead] = confidence

    if len(raw_values) < min_leads:
        return AxisResult(
            axis_deg=None,
            used_leads={},
            excluded_leads=excluded,
            reason="insufficient_t_axis_leads",
        )

    provisional_axis = _axis_from_signed_values(raw_values, confidences)
    if provisional_axis is None:
        return AxisResult(
            axis_deg=None,
            used_leads={},
            excluded_leads=excluded,
            reason="insufficient_clustered_t_axis_leads",
        )

    clustered_values: Dict[str, float] = {}
    clustered_confidences: Dict[str, float] = {}
    for lead, value in raw_values.items():
        rep = representative_leads[lead]
        lead_angle = HEXAXIAL_ANGLES[lead]
        expected_projection = math.cos(math.radians(provisional_axis - lead_angle))
        sign_conflict = (
            abs(expected_projection) >= _PROJECTION_NEUTRAL_COS
            and _polarity(value) != _polarity(expected_projection)
        )
        if rep.params.get("st_t_confusion", False) or sign_conflict:
            excluded[lead] = "polarity_conflict_or_st_t_confusion"
            continue
        clustered_values[lead] = value
        clustered_confidences[lead] = confidences[lead]

    if len(clustered_values) < min_leads:
        return AxisResult(
            axis_deg=None,
            used_leads=clustered_values,
            excluded_leads=excluded,
            reason="insufficient_clustered_t_axis_leads",
        )

    axis = _axis_from_signed_values(clustered_values, clustered_confidences)
    if axis is None:
        return AxisResult(
            axis_deg=None,
            used_leads=clustered_values,
            excluded_leads=excluded,
            reason="insufficient_clustered_t_axis_leads",
        )
    return AxisResult(
        axis_deg=axis,
        used_leads=clustered_values,
        excluded_leads=excluded,
        reason=None,
    )
