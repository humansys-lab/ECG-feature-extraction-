"""Pacing policy: pacing context, capture, QRS rescue and paced-measurement decisions.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
Thresholds and decisions are unchanged; the policy objects in this module wrap
each decision cluster so the staged extractor can record a ``PolicyEvent`` next
to the value the helper returned.

Placement note: ``_robust_rr_profile`` is assigned to ``pipeline/stages/measurement.py``
by the doc-01 table but to this module by doc 04.  Its only callers are the pacing
QRS-rescue helpers below, and the doc-01 placement would create a pacing-policy <->
measurement-stage import cycle, so it lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..._engine.detection.qrs import DEFAULT_QRS_LEADS, detect_qrs_multilead_with_meta
from ..._engine.foundation.numeric import _finite_float
from ..._engine.preprocess import lowpass_filter
from ..._engine.quality.signal import (
    detect_pacing_spikes,
    remove_pacing_spikes,
    validate_pacing_spikes_against_qrs,
)
from ...models import STANDARD_12_LEADS
from ..context import PipelineContext, PolicyDecision, PolicyEvent
from ..stages.atrial import _av_block_evidence
from ._events import event

_PACING_LIKE_NEAR_WIDE_QRS_MS = 115.0
_PACING_LIKE_ATRIAL_EVENTS_PER_RR_MIN = 3.0
_PACING_LIKE_WIDE_GROUP_QRS_MS = 150.0
_PACING_LIKE_OVERWIDE_CONSENSUS_MS = 160.0
_PACING_LIKE_NARROW_RAW_GROUP_MS = 90.0
_PACING_LIKE_RAW_QRS_MIN_MS = 90.0
_PACING_LIKE_RAW_QRS_MAX_MS = 180.0
_PACING_LIKE_QRS_UNDERESTIMATE_MARGIN_MS = 25.0
_PACING_LIKE_RAW_QRS_UNDERESTIMATE_MARGIN_MS = 20.0
_PACING_DOMINANT_WIDE_QRS_MIN_MS = 160.0
_PACING_DOMINANT_WIDE_GROUP_MIN_MS = 145.0
_PACING_DOMINANT_WIDE_GROUP_MAX_MS = 220.0
_PACING_DOMINANT_WIDE_MIN_DELTA_MS = 8.0
_PACING_DOMINANT_WIDE_MAX_DELTA_MS = 25.0
_PACING_INTERMITTENT_NARROW_QRS_MAX_MS = 120.0
_PACING_INTERMITTENT_WIDE_CONSENSUS_MIN_MS = 130.0
_PACING_INTERMITTENT_WIDE_GROUP_MIN_MS = 180.0
_PACING_INTERMITTENT_WIDE_GROUP_MAX_MS = 260.0
_PACING_INTERMITTENT_MIN_OVERRIDE_DELTA_MS = 25.0
_PACING_BORDERLINE_QRS_MIN_MS = 115.0
_PACING_BORDERLINE_QRS_MAX_MS = 125.0
_PACING_BORDERLINE_WIDE_OFFSET_MIN_MS = 128.0
_PACING_BORDERLINE_WIDE_OFFSET_MAX_MS = 160.0
_PACING_BORDERLINE_WIDE_OFFSET_MIN_DELTA_MS = 8.0
_PACING_SECONDARY_OVERWIDE_QRS_MIN_MS = 170.0
_PACING_SECONDARY_OVERWIDE_QRS_MAX_MS = 190.0
_PACING_SECONDARY_WIDE_OFFSET_MIN_MS = 190.0
_PACING_SECONDARY_DOMINANT_GROUP_MIN_MS = 110.0
_PACING_SECONDARY_DOMINANT_GROUP_MAX_MS = 140.0
_PACING_SECONDARY_GROUP_MIN_MS = 145.0
_PACING_SECONDARY_GROUP_MAX_MS = 180.0
_PACING_CAPTURE_MIN_CONFIRMED_BEATS = 2
_PACING_CAPTURE_MAX_MEDIAN_ABS_OFFSET_MS = 55.0
_PACING_QRS_RESCUE_MIN_RAW_BEATS = 5
_PACING_QRS_RESCUE_MIN_ADDITIONAL_BEATS = 2
_PACING_QRS_RESCUE_MIN_COUNT_RATIO = 1.35
_PACING_QRS_RESCUE_MIN_CAPTURE_BEATS = 3
_PACING_QRS_RESCUE_MIN_RAW_CAPTURE_FRACTION = 0.50
_PACING_QRS_RESCUE_MIN_SPIKE_CAPTURE_FRACTION = 0.60
_PACING_QRS_RESCUE_MAX_RAW_RR_CV = 0.15
_PACING_QRS_RESCUE_MAX_RAW_RR_DEVIATION = 0.35
_PACING_QRS_RESCUE_MIN_RR_MS = 240.0
_PACING_QRS_RESCUE_MAX_RR_MS = 2400.0
_PACING_QRS_RESCUE_MATCH_TOLERANCE_MS = 120.0
_PACING_NEAR_WIDE_RAW_QRS_MIN_MS = 115.0
_PACING_NEAR_WIDE_RAW_QRS_MAX_MS = 125.0
_PACING_NEAR_WIDE_OFFSET_MIN_MS = 160.0
_PACING_NEAR_WIDE_OFFSET_MAX_MS = 190.0
_PACING_NEAR_WIDE_GROUP_MIN_MS = 145.0
_PACING_NEAR_WIDE_GROUP_MAX_MS = 180.0
_PACING_OVERWIDE_ESTIMATE_MIN_MS = 150.0
_PACING_OVERWIDE_ESTIMATE_MAX_MS = 170.0
_PACING_OVERWIDE_RAW_MIN_MS = 115.0
_PACING_OVERWIDE_RAW_MAX_MS = 140.0
_PACING_OVERWIDE_OFFSET_MIN_MS = 190.0
_PACING_OVERWIDE_DOMINANT_GROUP_MIN_MS = 110.0
_PACING_OVERWIDE_DOMINANT_GROUP_MAX_MS = 140.0
_PACING_ATTENUATED_RESCUE_PROMINENCE_UV = (150.0, 100.0, 25.0)
_PACING_ATTENUATED_RESCUE_CLASSES = {
    "ventricular_capture",
    "mixed_pacing_and_qrs_edge",
}
_PACING_RESCUE_STOP_CLASSES = {
    "atrial_or_nonventricular",
    "qrs_edge_artifact",
}


def _pacing_like_av_evidence(rule_summary: Dict[str, Any]) -> Tuple[bool, bool]:
    pauses = rule_summary.get("pauses") if isinstance(rule_summary, dict) else {}
    av_evidence = _av_block_evidence(rule_summary)
    av_relation_evidence = bool(
        rule_summary.get("av_dissociation") or av_evidence.get("dropped_p_evidence")
    )
    atrial_events_per_rr_max = _finite_float(
        pauses.get("atrial_events_per_rr_max") if isinstance(pauses, dict) else None
    )
    rich_atrial_activity = bool(
        rule_summary.get("atrial_measurements_invalid")
        or (
            atrial_events_per_rr_max is not None
            and atrial_events_per_rr_max >= _PACING_LIKE_ATRIAL_EVENTS_PER_RR_MIN
        )
    )
    return av_relation_evidence, rich_atrial_activity



def _dominant_group_qrs_ms(groups: Dict[int, Any]) -> Optional[float]:
    fallback: List[float] = []
    for group in (groups or {}).values():
        group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
        if group_qrs is None:
            continue
        fallback.append(group_qrs)
        flags = getattr(group, "flags", {})
        if isinstance(flags, dict) and bool(flags.get("dominant_group")):
            return group_qrs
    return float(np.median(fallback)) if fallback else None



def _wide_qrs_pacing_like_context(qrs_duration_ms: object, rule_summary: Dict[str, Any]) -> bool:
    qrs_ms = _finite_float(qrs_duration_ms)
    if qrs_ms is None or qrs_ms < _PACING_LIKE_NEAR_WIDE_QRS_MS:
        return False
    av_relation_evidence, rich_atrial_activity = _pacing_like_av_evidence(rule_summary)
    if qrs_ms >= 120.0:
        return bool(av_relation_evidence and rich_atrial_activity)
    return bool(av_relation_evidence and rich_atrial_activity)



def _overwide_qrs_pacing_like_context(
    qrs_duration_ms: object,
    groups: Dict[int, Any],
    rule_summary: Dict[str, Any],
) -> bool:
    qrs_ms = _finite_float(qrs_duration_ms)
    if qrs_ms is None or qrs_ms < _PACING_LIKE_OVERWIDE_CONSENSUS_MS:
        return False
    raw_group_qrs = _dominant_group_qrs_ms(groups)
    if raw_group_qrs is None or raw_group_qrs > _PACING_LIKE_NARROW_RAW_GROUP_MS:
        return False
    return bool(_av_block_evidence(rule_summary).get("dropped_p_evidence"))



def _overwide_pacing_qrs_override_ms(
    qrs_duration_ms: object,
    representative_leads: Dict[str, Any],
    groups: Dict[int, Any],
    rule_summary: Dict[str, Any],
) -> Optional[float]:
    if not _overwide_qrs_pacing_like_context(qrs_duration_ms, groups, rule_summary):
        return None
    candidates: List[float] = []
    for rep in (representative_leads or {}).values():
        params = getattr(rep, "params", {})
        if not isinstance(params, dict) or not params.get("reliable_for_qrs"):
            continue
        raw_qrs = _finite_float(params.get("qrs_ms"))
        if (
            raw_qrs is not None
            and _PACING_LIKE_RAW_QRS_MIN_MS <= raw_qrs <= _PACING_LIKE_RAW_QRS_MAX_MS
        ):
            candidates.append(raw_qrs)
    if len(candidates) < 2:
        return None
    return float(np.median(candidates))



def _raw_pacing_qrs_underestimate_override_ms(
    qrs_duration_ms: object,
    representative_leads: Dict[str, Any],
    rule_summary: Dict[str, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    if qrs_ms is None or not _wide_qrs_pacing_like_context(qrs_ms, rule_summary):
        return None
    candidates: List[float] = []
    for rep in (representative_leads or {}).values():
        params = getattr(rep, "params", {})
        if not isinstance(params, dict) or not params.get("reliable_for_qrs"):
            continue
        raw_qrs = _finite_float(params.get("qrs_ms"))
        if (
            raw_qrs is not None
            and _PACING_LIKE_RAW_QRS_MIN_MS <= raw_qrs <= _PACING_LIKE_RAW_QRS_MAX_MS
        ):
            candidates.append(raw_qrs)
    if len(candidates) < 2:
        return None
    override = float(np.median(candidates))
    if override - qrs_ms >= _PACING_LIKE_RAW_QRS_UNDERESTIMATE_MARGIN_MS:
        return override
    return None



def _near_wide_pacing_qrs_override_ms(
    qrs_duration_ms: object,
    groups: Dict[int, Any],
    rule_summary: Dict[str, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    if qrs_ms is None:
        return None
    if not _wide_qrs_pacing_like_context(qrs_ms, rule_summary):
        return None
    dominant_qrs = _dominant_group_qrs_ms(groups)
    if dominant_qrs is not None and dominant_qrs >= 120.0:
        return None
    candidates: List[float] = []
    for group in (groups or {}).values():
        group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
        if group_qrs is not None and group_qrs >= _PACING_LIKE_WIDE_GROUP_QRS_MS:
            candidates.append(group_qrs)
    if not candidates:
        for group in (groups or {}).values():
            group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
            if group_qrs is not None and group_qrs >= 120.0:
                candidates.append(group_qrs)
    if not candidates:
        return None
    override = float(np.median(candidates))
    if qrs_ms < 120.0 or override - qrs_ms >= _PACING_LIKE_QRS_UNDERESTIMATE_MARGIN_MS:
        return override
    return None



def _dominant_paced_wide_qrs_override_ms(
    qrs_duration_ms: object,
    groups: Dict[int, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    if qrs_ms is None or qrs_ms < _PACING_DOMINANT_WIDE_QRS_MIN_MS:
        return None
    for group in (groups or {}).values():
        flags = getattr(group, "flags", {})
        if not isinstance(flags, dict):
            continue
        if not (flags.get("dominant_group") and flags.get("wide_qrs")):
            continue
        group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
        if group_qrs is None:
            continue
        if not (_PACING_DOMINANT_WIDE_GROUP_MIN_MS <= group_qrs <= _PACING_DOMINANT_WIDE_GROUP_MAX_MS):
            continue
        delta = qrs_ms - group_qrs
        if _PACING_DOMINANT_WIDE_MIN_DELTA_MS <= delta <= _PACING_DOMINANT_WIDE_MAX_DELTA_MS:
            return group_qrs
    return None



def _intermittent_paced_wide_qrs_override_ms(
    qrs_duration_ms: object,
    representative_leads: Dict[str, Any],
    groups: Dict[int, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    if qrs_ms is None or qrs_ms >= _PACING_INTERMITTENT_NARROW_QRS_MAX_MS:
        return None

    qrs_wide_values: List[float] = []
    for rep in (representative_leads or {}).values():
        params = getattr(rep, "params", {})
        if not isinstance(params, dict) or not params.get("reliable_for_qrs"):
            continue
        qrs_wide = _finite_float(params.get("qrs_wide_ms"))
        if qrs_wide is not None:
            qrs_wide_values.append(qrs_wide)
    if not qrs_wide_values:
        return None
    qrs_wide_ms = float(np.median(qrs_wide_values))
    if (
        qrs_wide_ms < _PACING_INTERMITTENT_WIDE_CONSENSUS_MIN_MS
        or qrs_wide_ms - qrs_ms < _PACING_INTERMITTENT_MIN_OVERRIDE_DELTA_MS
    ):
        return None

    wide_group_values: List[float] = []
    for group in (groups or {}).values():
        flags = getattr(group, "flags", {})
        if not isinstance(flags, dict) or not flags.get("wide_qrs"):
            continue
        group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
        if (
            group_qrs is not None
            and _PACING_INTERMITTENT_WIDE_GROUP_MIN_MS <= group_qrs <= _PACING_INTERMITTENT_WIDE_GROUP_MAX_MS
        ):
            wide_group_values.append(group_qrs)
    if not wide_group_values:
        return None

    override = float(np.median([qrs_wide_ms, float(np.median(wide_group_values))]))
    if override - qrs_ms < _PACING_INTERMITTENT_MIN_OVERRIDE_DELTA_MS:
        return None
    return override



def _qrs_wide_values(representative_leads: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for rep in (representative_leads or {}).values():
        params = getattr(rep, "params", {})
        if not isinstance(params, dict) or not params.get("reliable_for_qrs"):
            continue
        qrs_wide = _finite_float(params.get("qrs_wide_ms"))
        if qrs_wide is not None:
            values.append(qrs_wide)
    return values



def _intermittent_pacing_wide_measurement_context(
    representative_leads: Dict[str, Any],
) -> bool:
    raw_values: List[float] = []
    for rep in (representative_leads or {}).values():
        params = getattr(rep, "params", {})
        if not isinstance(params, dict) or not params.get("reliable_for_qrs"):
            continue
        raw_qrs = _finite_float(params.get("qrs_ms"))
        if raw_qrs is not None:
            raw_values.append(raw_qrs)
    wide_values = _qrs_wide_values(representative_leads)
    if len(raw_values) < 3 or len(wide_values) < 3:
        return False
    raw_median = float(np.median(raw_values))
    wide_median = float(np.median(wide_values))
    return bool(
        raw_median < _PACING_INTERMITTENT_NARROW_QRS_MAX_MS
        and wide_median >= _PACING_INTERMITTENT_WIDE_CONSENSUS_MIN_MS
        and wide_median - raw_median >= _PACING_INTERMITTENT_MIN_OVERRIDE_DELTA_MS
    )



def _borderline_paced_qrs_wide_offset_override_ms(
    qrs_duration_ms: object,
    representative_leads: Dict[str, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    if (
        qrs_ms is None
        or qrs_ms < _PACING_BORDERLINE_QRS_MIN_MS
        or qrs_ms > _PACING_BORDERLINE_QRS_MAX_MS
    ):
        return None
    values = _qrs_wide_values(representative_leads)
    if len(values) < 3:
        return None
    qrs_wide_ms = float(np.median(values))
    if not (
        _PACING_BORDERLINE_WIDE_OFFSET_MIN_MS
        <= qrs_wide_ms
        <= _PACING_BORDERLINE_WIDE_OFFSET_MAX_MS
    ):
        return None
    if qrs_wide_ms - qrs_ms < _PACING_BORDERLINE_WIDE_OFFSET_MIN_DELTA_MS:
        return None
    return qrs_wide_ms



def _secondary_paced_wide_group_qrs_override_ms(
    qrs_duration_ms: object,
    representative_leads: Dict[str, Any],
    groups: Dict[int, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    floor_censored_consensus_ms: Optional[float] = None
    if (
        qrs_ms is not None
        and _PACING_NEAR_WIDE_RAW_QRS_MIN_MS
        <= qrs_ms
        <= _PACING_NEAR_WIDE_RAW_QRS_MAX_MS
    ):
        raw_values: List[float] = []
        consensus_values: List[float] = []
        for rep in (representative_leads or {}).values():
            params = getattr(rep, "params", {})
            if not isinstance(params, dict) or not params.get("reliable_for_qrs"):
                continue
            raw_qrs = _finite_float(params.get("qrs_ms"))
            consensus_qrs = _finite_float(params.get("qrs_consensus_ms"))
            if raw_qrs is not None:
                raw_values.append(raw_qrs)
            if consensus_qrs is not None:
                consensus_values.append(consensus_qrs)
        floor_support = sum(
            abs(value - qrs_ms) <= 2.0
            for value in raw_values
        )
        minimum_floor_support = max(3, int(np.ceil(0.50 * len(raw_values))))
        if len(consensus_values) >= 6 and floor_support >= minimum_floor_support:
            consensus_center = float(np.median(consensus_values))
            consensus_spread = float(
                np.percentile(consensus_values, 90.0)
                - np.percentile(consensus_values, 10.0)
            )
            if (
                _PACING_SECONDARY_OVERWIDE_QRS_MIN_MS
                <= consensus_center
                <= _PACING_SECONDARY_OVERWIDE_QRS_MAX_MS
                and consensus_center - qrs_ms >= 35.0
                and consensus_spread <= 20.0
            ):
                floor_censored_consensus_ms = consensus_center
                qrs_ms = consensus_center
    if (
        qrs_ms is None
        or qrs_ms < _PACING_SECONDARY_OVERWIDE_QRS_MIN_MS
        or qrs_ms > _PACING_SECONDARY_OVERWIDE_QRS_MAX_MS
    ):
        return None
    qrs_wide_values = _qrs_wide_values(representative_leads)
    if len(qrs_wide_values) < 3 or float(np.median(qrs_wide_values)) < _PACING_SECONDARY_WIDE_OFFSET_MIN_MS:
        return None

    dominant_group_qrs: Optional[float] = None
    secondary_wide_values: List[float] = []
    for group in (groups or {}).values():
        flags = getattr(group, "flags", {})
        if not isinstance(flags, dict):
            continue
        group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
        if group_qrs is None:
            continue
        if flags.get("dominant_group"):
            dominant_group_qrs = group_qrs
            continue
        if (
            flags.get("wide_qrs")
            and _PACING_SECONDARY_GROUP_MIN_MS
            <= group_qrs
            <= _PACING_SECONDARY_GROUP_MAX_MS
        ):
            secondary_wide_values.append(group_qrs)

    if dominant_group_qrs is None and floor_censored_consensus_ms is None:
        return None
    if dominant_group_qrs is not None and not (
        _PACING_SECONDARY_DOMINANT_GROUP_MIN_MS
        <= dominant_group_qrs
        <= _PACING_SECONDARY_DOMINANT_GROUP_MAX_MS
    ):
        return None
    if not secondary_wide_values:
        return None
    return float(np.median(secondary_wide_values))



def _near_wide_paced_qrs_offset_override_ms(
    qrs_duration_ms: object,
    representative_leads: Dict[str, Any],
    groups: Dict[int, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    if (
        qrs_ms is None
        or qrs_ms < _PACING_NEAR_WIDE_RAW_QRS_MIN_MS
        or qrs_ms > _PACING_NEAR_WIDE_RAW_QRS_MAX_MS
    ):
        return None
    qrs_wide_values = _qrs_wide_values(representative_leads)
    if len(qrs_wide_values) < 3:
        return None
    qrs_wide_ms = float(np.median(qrs_wide_values))
    if not (
        _PACING_NEAR_WIDE_OFFSET_MIN_MS
        <= qrs_wide_ms
        <= _PACING_NEAR_WIDE_OFFSET_MAX_MS
    ):
        return None

    wide_group_values: List[float] = []
    for group in (groups or {}).values():
        flags = getattr(group, "flags", {})
        if not isinstance(flags, dict) or not flags.get("wide_qrs"):
            continue
        group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
        if (
            group_qrs is not None
            and _PACING_NEAR_WIDE_GROUP_MIN_MS
            <= group_qrs
            <= _PACING_NEAR_WIDE_GROUP_MAX_MS
        ):
            wide_group_values.append(group_qrs)
    if wide_group_values:
        return float(np.median(wide_group_values))
    return qrs_wide_ms



def _overwide_paced_qrs_raw_consensus_override_ms(
    qrs_duration_ms: object,
    representative_leads: Dict[str, Any],
    groups: Dict[int, Any],
) -> Optional[float]:
    qrs_ms = _finite_float(qrs_duration_ms)
    if (
        qrs_ms is None
        or qrs_ms < _PACING_OVERWIDE_ESTIMATE_MIN_MS
        or qrs_ms > _PACING_OVERWIDE_ESTIMATE_MAX_MS
    ):
        return None

    qrs_wide_values = _qrs_wide_values(representative_leads)
    if len(qrs_wide_values) < 3 or float(np.median(qrs_wide_values)) < _PACING_OVERWIDE_OFFSET_MIN_MS:
        return None

    raw_values: List[float] = []
    raw_consensus_count = 0
    for rep in (representative_leads or {}).values():
        params = getattr(rep, "params", {})
        if not isinstance(params, dict) or not params.get("reliable_for_qrs"):
            continue
        raw_qrs = _finite_float(params.get("qrs_ms"))
        if raw_qrs is None:
            continue
        raw_values.append(raw_qrs)
        if _PACING_OVERWIDE_RAW_MIN_MS <= raw_qrs <= _PACING_OVERWIDE_RAW_MAX_MS:
            raw_consensus_count += 1
    min_raw_consensus = max(3, int(np.ceil(0.50 * len(raw_values))))
    if len(raw_values) < 3 or raw_consensus_count < min_raw_consensus:
        return None

    dominant_group_qrs: Optional[float] = None
    for group in (groups or {}).values():
        flags = getattr(group, "flags", {})
        if not isinstance(flags, dict) or not flags.get("dominant_group"):
            continue
        dominant_group_qrs = _finite_float(getattr(group, "mean_qrs_ms", None))
        break
    if dominant_group_qrs is None or not (
        _PACING_OVERWIDE_DOMINANT_GROUP_MIN_MS
        <= dominant_group_qrs
        <= _PACING_OVERWIDE_DOMINANT_GROUP_MAX_MS
    ):
        return None
    return float(np.median(raw_values))



def _paced_beat_fraction(paced_beat_ids: List[int], total_beats: int) -> float:
    if total_beats <= 0:
        return 0.0
    return float(len({int(beat_id) for beat_id in paced_beat_ids})) / float(total_beats)



def _selected_group_paced_majority(
    measurement_beat_ids: List[int],
    paced_beat_ids: List[int],
) -> bool:
    if not measurement_beat_ids:
        return False
    paced_set = {int(beat_id) for beat_id in paced_beat_ids}
    paced_count = sum(1 for beat_id in measurement_beat_ids if int(beat_id) in paced_set)
    return paced_count > (len(measurement_beat_ids) / 2.0)



def _pacing_capture_alignment_confirmed(
    pacing_result: Dict[str, Any],
    spike_offsets_samples: List[int],
    fs: int,
    evidence_quality: Optional[Dict[str, Any]] = None,
) -> bool:
    if not bool(pacing_result.get("paced", False)):
        return False
    if evidence_quality is not None:
        return bool(evidence_quality.get("supports_measurement_routing", False))
    if len(spike_offsets_samples) < _PACING_CAPTURE_MIN_CONFIRMED_BEATS:
        return False
    if fs <= 0:
        return False
    offsets_ms = [
        abs(float(offset) * 1000.0 / float(fs))
        for offset in spike_offsets_samples
    ]
    return float(np.median(offsets_ms)) <= _PACING_CAPTURE_MAX_MEDIAN_ABS_OFFSET_MS



def _confirmed_attenuated_pacing_rescue(pacing_result: Dict[str, Any]) -> bool:
    if not bool(pacing_result.get("paced", False)):
        return False
    if str(pacing_result.get("state") or "off") != "on":
        return False
    if len(pacing_result.get("spike_times", []) or []) < 3:
        return False
    return str(pacing_result.get("pacing_event_class") or "") in _PACING_ATTENUATED_RESCUE_CLASSES



def _pacing_capture_beat_ids(
    spike_times: List[int],
    r_locs: np.ndarray,
    fs: int,
) -> List[int]:
    if fs <= 0 or len(r_locs) == 0 or not spike_times:
        return []
    spike_lo_margin = int(round(0.080 * fs))
    spike_hi_margin = int(round(0.020 * fs))
    beat_ids: List[int] = []
    for beat_id, r in enumerate(np.asarray(r_locs, dtype=int)):
        spike_lo = int(r) - spike_lo_margin
        spike_hi = int(r) + spike_hi_margin
        if any(spike_lo <= int(spike) <= spike_hi for spike in spike_times):
            beat_ids.append(int(beat_id))
    return beat_ids



def _robust_rr_profile(r_locs: np.ndarray, fs: int) -> Optional[Dict[str, float]]:
    if fs <= 0 or len(r_locs) < 3:
        return None
    rr_ms = np.diff(np.asarray(r_locs, dtype=float)) * 1000.0 / float(fs)
    rr_ms = rr_ms[np.isfinite(rr_ms) & (rr_ms > 0.0)]
    if len(rr_ms) < 2:
        return None
    median_rr_ms = float(np.median(rr_ms))
    if median_rr_ms <= 0.0:
        return None
    mad_ms = float(np.median(np.abs(rr_ms - median_rr_ms)))
    return {
        "median_rr_ms": median_rr_ms,
        "robust_rr_cv": float(1.4826 * mad_ms / median_rr_ms),
        "max_relative_rr_deviation": float(
            np.max(np.abs(rr_ms - median_rr_ms)) / median_rr_ms
        ),
    }



def _pacing_qrs_rescue_evidence(
    *,
    pacing_result: Dict[str, Any],
    pre_despike_qrs_result: Any,
    despiked_qrs_result: Any,
    fs: int,
) -> Dict[str, Any]:
    """Decide whether de-spiking removed genuine, regularly captured QRS beats."""
    evidence: Dict[str, Any] = {
        "applied": False,
        "reason": None,
        "pre_despike_n_beats": 0,
        "despiked_n_beats": 0,
        "pre_despike_median_rr_ms": None,
        "pre_despike_robust_rr_cv": None,
        "pre_despike_max_relative_rr_deviation": None,
        "despiked_robust_rr_cv": None,
        "capture_beats": 0,
        "pre_despike_capture_fraction": 0.0,
        "spike_capture_fraction": 0.0,
    }
    if (
        fs <= 0
        or not bool(pacing_result.get("paced", False))
        or str(pacing_result.get("state") or "off") != "on"
    ):
        return evidence

    spike_times = [int(s) for s in pacing_result.get("spike_times", []) or []]
    raw_r_locs = np.asarray(
        getattr(pre_despike_qrs_result, "r_locs", []) or [],
        dtype=int,
    )
    cleaned_r_locs = np.asarray(
        getattr(despiked_qrs_result, "r_locs", []) or [],
        dtype=int,
    )
    raw_count = int(len(raw_r_locs))
    cleaned_count = int(len(cleaned_r_locs))
    evidence["pre_despike_n_beats"] = raw_count
    evidence["despiked_n_beats"] = cleaned_count
    if (
        len(spike_times) < _PACING_QRS_RESCUE_MIN_CAPTURE_BEATS
        or raw_count < _PACING_QRS_RESCUE_MIN_RAW_BEATS
        or raw_count < cleaned_count + _PACING_QRS_RESCUE_MIN_ADDITIONAL_BEATS
        or raw_count / float(max(cleaned_count, 1)) < _PACING_QRS_RESCUE_MIN_COUNT_RATIO
    ):
        return evidence

    raw_rr = _robust_rr_profile(raw_r_locs, fs)
    cleaned_rr = _robust_rr_profile(cleaned_r_locs, fs)
    if raw_rr is None:
        return evidence
    evidence["pre_despike_median_rr_ms"] = raw_rr["median_rr_ms"]
    evidence["pre_despike_robust_rr_cv"] = raw_rr["robust_rr_cv"]
    evidence["pre_despike_max_relative_rr_deviation"] = raw_rr[
        "max_relative_rr_deviation"
    ]
    evidence["despiked_robust_rr_cv"] = (
        cleaned_rr["robust_rr_cv"] if cleaned_rr is not None else None
    )
    if not (
        _PACING_QRS_RESCUE_MIN_RR_MS
        <= raw_rr["median_rr_ms"]
        <= _PACING_QRS_RESCUE_MAX_RR_MS
        and raw_rr["robust_rr_cv"] <= _PACING_QRS_RESCUE_MAX_RAW_RR_CV
        and raw_rr["max_relative_rr_deviation"]
        <= _PACING_QRS_RESCUE_MAX_RAW_RR_DEVIATION
    ):
        return evidence

    capture_beat_ids = _pacing_capture_beat_ids(spike_times, raw_r_locs, fs)
    capture_count = int(len(capture_beat_ids))
    raw_capture_fraction = capture_count / float(raw_count)
    spike_capture_fraction = capture_count / float(len(spike_times))
    evidence["capture_beats"] = capture_count
    evidence["pre_despike_capture_fraction"] = raw_capture_fraction
    evidence["spike_capture_fraction"] = spike_capture_fraction
    if (
        capture_count < _PACING_QRS_RESCUE_MIN_CAPTURE_BEATS
        or raw_capture_fraction < _PACING_QRS_RESCUE_MIN_RAW_CAPTURE_FRACTION
        or spike_capture_fraction < _PACING_QRS_RESCUE_MIN_SPIKE_CAPTURE_FRACTION
    ):
        return evidence

    evidence["applied"] = True
    evidence["reason"] = "despiking_removed_regular_pacing_aligned_qrs"
    return evidence



def _pacing_qrs_cleaned_single_lead_rescue(
    *,
    ecg_detect: np.ndarray,
    fs: int,
    quality: Dict[str, Any],
    pacing_result: Dict[str, Any],
    pre_despike_qrs_result: Any,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Recover missing paced beats on a reliable de-spiked single lead."""
    raw_r_locs = np.asarray(
        getattr(pre_despike_qrs_result, "r_locs", []) or [],
        dtype=int,
    )
    if fs <= 0 or len(raw_r_locs) < _PACING_QRS_RESCUE_MIN_RAW_BEATS:
        return None, {}

    spike_times = [int(s) for s in pacing_result.get("spike_times", []) or []]
    tolerance = max(1, int(round(_PACING_QRS_RESCUE_MATCH_TOLERANCE_MS * fs / 1000.0)))
    best: Optional[Tuple[Tuple[float, float, float, float], Any, Dict[str, Any]]] = None
    for lead_index in DEFAULT_QRS_LEADS:
        if lead_index >= ecg_detect.shape[0]:
            continue
        lead_name = STANDARD_12_LEADS[lead_index]
        lead_quality = quality.get(lead_name)
        if lead_quality is not None and not bool(
            getattr(lead_quality, "reliable_for_qrs", getattr(lead_quality, "reliable", True))
        ):
            continue

        candidate = detect_qrs_multilead_with_meta(
            ecg_detect,
            fs,
            leads=(lead_index,),
            # Compare single-lead energy candidates on the same II timing
            # reference as the original multi-lead detections. This is an
            # explicit pacing-capture reference, not an implicit excluded lead.
            fiducial_lead=1,
        )
        candidate_r_locs = np.asarray(candidate.r_locs, dtype=int)
        if len(candidate_r_locs) < len(raw_r_locs):
            continue
        rr_profile = _robust_rr_profile(candidate_r_locs, fs)
        if rr_profile is None or not (
            _PACING_QRS_RESCUE_MIN_RR_MS
            <= rr_profile["median_rr_ms"]
            <= _PACING_QRS_RESCUE_MAX_RR_MS
            and rr_profile["robust_rr_cv"] <= _PACING_QRS_RESCUE_MAX_RAW_RR_CV
            and rr_profile["max_relative_rr_deviation"]
            <= _PACING_QRS_RESCUE_MAX_RAW_RR_DEVIATION
        ):
            continue

        matched = sum(
            1
            for raw_r in raw_r_locs
            if np.any(np.abs(candidate_r_locs - int(raw_r)) <= tolerance)
        )
        match_fraction = matched / float(len(raw_r_locs))
        capture_count = len(
            _pacing_capture_beat_ids(spike_times, candidate_r_locs, fs)
        )
        raw_capture_fraction = capture_count / float(len(candidate_r_locs))
        spike_capture_fraction = capture_count / float(max(len(spike_times), 1))
        if (
            match_fraction < 0.90
            or capture_count < _PACING_QRS_RESCUE_MIN_CAPTURE_BEATS
            or raw_capture_fraction < _PACING_QRS_RESCUE_MIN_RAW_CAPTURE_FRACTION
            or spike_capture_fraction < _PACING_QRS_RESCUE_MIN_SPIKE_CAPTURE_FRACTION
        ):
            continue

        confidences = np.asarray(
            getattr(candidate, "qrs_detector_confidence", []) or [],
            dtype=float,
        )
        mean_confidence = (
            float(np.mean(confidences[np.isfinite(confidences)]))
            if np.any(np.isfinite(confidences))
            else 0.0
        )
        detail = {
            "rescue_lead": lead_name,
            "cleaned_single_lead_n_beats": int(len(candidate_r_locs)),
            "cleaned_single_lead_match_fraction": match_fraction,
            "cleaned_single_lead_capture_fraction": raw_capture_fraction,
            "cleaned_single_lead_robust_rr_cv": rr_profile["robust_rr_cv"],
        }
        score = (
            float(len(candidate_r_locs)),
            match_fraction,
            mean_confidence,
            -rr_profile["robust_rr_cv"],
        )
        bundle = (score, candidate, detail)
        if best is None or bundle[0] > best[0]:
            best = bundle

    if best is None:
        return None, {}
    _, candidate, detail = best
    candidate.fallback_used = True
    candidate.fallback_reason = "paced_cleaned_single_lead_qrs_rescue"
    return candidate, detail



def _try_attenuated_pacing_rescue(
    *,
    ecg_rs: np.ndarray,
    ecg_an: np.ndarray,
    fs_run: int,
    lp_hz: float,
    current_pacing_result: Dict[str, Any],
    current_ecg_measure: np.ndarray,
    current_ecg_detect: np.ndarray,
    current_qrs_result: Any,
    current_r_locs: np.ndarray,
    pacing_detection_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Any, np.ndarray]:
    """Recover attenuated pacer spikes only when QRS-validated capture is present."""
    if (
        bool(current_pacing_result.get("paced", False))
        and str(current_pacing_result.get("state") or "off") == "on"
        and current_pacing_result.get("spike_times")
    ):
        return (
            current_pacing_result,
            current_ecg_measure,
            current_ecg_detect,
            current_qrs_result,
            current_r_locs,
        )
    if str(current_pacing_result.get("pacing_event_class") or "") in _PACING_RESCUE_STOP_CLASSES:
        return (
            current_pacing_result,
            current_ecg_measure,
            current_ecg_detect,
            current_qrs_result,
            current_r_locs,
        )
    if fs_run <= 0 or len(current_r_locs) < 3:
        return (
            current_pacing_result,
            current_ecg_measure,
            current_ecg_detect,
            current_qrs_result,
            current_r_locs,
        )

    existing_spikes = [int(s) for s in current_pacing_result.get("spike_times", []) or []]
    best: Optional[
        Tuple[
            Tuple[float, float, float, float, float],
            Dict[str, Any],
            np.ndarray,
            np.ndarray,
            Any,
            np.ndarray,
        ]
    ] = None
    for prominence_uv in _PACING_ATTENUATED_RESCUE_PROMINENCE_UV:
        candidate = detect_pacing_spikes(
            ecg_rs,
            fs_run,
            min_prominence_uv=float(prominence_uv),
            detection_cache=pacing_detection_cache,
        )
        raw_spike_times = [int(s) for s in candidate.get("spike_times", []) or []]
        if len(raw_spike_times) < 3 or raw_spike_times == existing_spikes:
            continue

        candidate_measure = remove_pacing_spikes(ecg_an, raw_spike_times, fs_run)
        candidate_detect = lowpass_filter(candidate_measure, fs_run, cutoff_hz=lp_hz, order=4)
        candidate_qrs = detect_qrs_multilead_with_meta(candidate_detect, fs_run)
        candidate_r_locs = np.asarray(candidate_qrs.r_locs, dtype=int)
        if len(candidate_r_locs) < 3:
            continue

        validated = validate_pacing_spikes_against_qrs(
            ecg_rs,
            fs_run,
            raw_spike_times,
            candidate_r_locs,
            candidate,
        )
        if not _confirmed_attenuated_pacing_rescue(validated):
            continue

        validated_spikes = [int(s) for s in validated.get("spike_times", []) or []]
        capture_beat_ids = _pacing_capture_beat_ids(validated_spikes, candidate_r_locs, fs_run)
        capture_fraction = float(len(capture_beat_ids)) / float(len(candidate_r_locs))
        if len(capture_beat_ids) < _PACING_CAPTURE_MIN_CONFIRMED_BEATS or capture_fraction < 0.50:
            continue

        validated = dict(validated)
        validated["rescue_applied"] = True
        validated["rescue_prominence_uv"] = float(prominence_uv)
        validated["rescue_reason"] = "attenuated_multilead_qrs_validated_pacing"
        validated["rescue_capture_fraction"] = capture_fraction
        score = (
            float(len(capture_beat_ids)),
            capture_fraction,
            float(len(validated_spikes)),
            float(prominence_uv),
            float(validated.get("lead_vote_count", 0) or 0),
        )
        candidate_bundle = (
            score,
            validated,
            candidate_measure,
            candidate_detect,
            candidate_qrs,
            candidate_r_locs,
        )
        if best is None or candidate_bundle[0] > best[0]:
            best = candidate_bundle

    if best is not None:
        _, validated, candidate_measure, candidate_detect, candidate_qrs, candidate_r_locs = best
        return validated, candidate_measure, candidate_detect, candidate_qrs, candidate_r_locs

    return (
        current_pacing_result,
        current_ecg_measure,
        current_ecg_detect,
        current_qrs_result,
        current_r_locs,
    )



def _pacing_measurement_effect(
    *,
    pacing_enabled: bool,
    pacing_result: Dict[str, Any],
    measurement_group_paced: bool,
    global_measurement_paced: bool,
    pacing_capture_confirmed: bool,
    paced_qrs_floor_beat_ids: List[int],
) -> str:
    if not pacing_enabled:
        return "disabled"
    if not pacing_result.get("spike_times"):
        return "none"
    if measurement_group_paced and paced_qrs_floor_beat_ids:
        return "segmentation_floor_and_global_paced_route"
    if measurement_group_paced:
        return "global_paced_route"
    if global_measurement_paced:
        return "intermittent_wide_qrs_paced_route"
    if pacing_capture_confirmed and paced_qrs_floor_beat_ids:
        return "per_beat_segmentation_floor"
    return "metadata_only"



def _pacing_segmentation_effect(
    *,
    pacing_enabled: bool,
    pacing_result: Dict[str, Any],
    measurement_group_paced: bool,
    segmentation_paced_beat_ids: List[int],
    paced_qrs_floor_beat_ids: List[int],
) -> str:
    if not pacing_enabled:
        return "disabled"
    if not pacing_result.get("spike_times"):
        return "none"
    if paced_qrs_floor_beat_ids:
        return (
            "segmentation_floor"
            if measurement_group_paced
            else "per_beat_segmentation_floor"
        )
    if segmentation_paced_beat_ids:
        return (
            "paced_segmentation"
            if measurement_group_paced
            else "per_beat_segmentation"
        )
    return "metadata_only"



def _measurement_pacing_state(
    *,
    pacing_enabled: bool,
    detection_state: object,
    confirmed_pacing_context: bool,
    pacing_measurement_effect: str,
) -> str:
    if not pacing_enabled:
        return "off"
    raw_state = str(detection_state or "off")
    if confirmed_pacing_context or pacing_measurement_effect in {
        "segmentation_floor_and_global_paced_route",
        "global_paced_route",
        "intermittent_wide_qrs_paced_route",
        "per_beat_segmentation_floor",
    }:
        return "on"
    if raw_state in {"on", "unknown"}:
        return "unknown"
    return "off"



def _build_pacing_evidence_by_beat(
    *,
    n_beats: int,
    pacing_result: Dict[str, Any],
    pacing_spike_beat_ids: List[int],
    pacing_spike_offsets_samples: List[int],
    pacing_capture_confirmed: bool,
    paced_beat_ids: List[int],
    segmentation_paced_beat_ids: List[int],
    paced_qrs_floor_beat_ids: List[int],
    measurement_beat_ids: List[int],
    measurement_group_paced: bool,
    fs: int,
) -> List[Dict[str, Any]]:
    spike_offsets_by_beat = {
        int(beat_id): int(offset)
        for beat_id, offset in zip(pacing_spike_beat_ids, pacing_spike_offsets_samples)
    }
    spike_set = {int(beat_id) for beat_id in pacing_spike_beat_ids}
    paced_set = {int(beat_id) for beat_id in paced_beat_ids}
    segmentation_set = {int(beat_id) for beat_id in segmentation_paced_beat_ids}
    floor_set = {int(beat_id) for beat_id in paced_qrs_floor_beat_ids}
    measurement_set = {int(beat_id) for beat_id in measurement_beat_ids}
    fs_value = float(fs) if fs else 0.0
    evidence: List[Dict[str, Any]] = []
    for beat_id in range(max(0, int(n_beats))):
        offset_samples = spike_offsets_by_beat.get(beat_id)
        offset_ms = (
            float(offset_samples) * 1000.0 / fs_value
            if offset_samples is not None and fs_value > 0.0
            else None
        )
        used_for_global = bool(
            measurement_group_paced
            and beat_id in measurement_set
            and beat_id in paced_set
        )
        evidence.append({
            "beat_id": beat_id,
            "spike_present": beat_id in spike_set,
            "spike_offset_ms": offset_ms,
            "ventricular_capture_confirmed": bool(
                pacing_capture_confirmed and beat_id in paced_set
            ),
            "used_for_segmentation": beat_id in segmentation_set,
            "used_for_representative_segmentation": used_for_global,
            "used_for_qrs_floor": beat_id in floor_set,
            "used_for_global_paced_route": used_for_global,
            "pacing_state": pacing_result.get("state", "off"),
            "lead_vote_count": int(pacing_result.get("lead_vote_count", 0) or 0),
        })
    return evidence



@dataclass(frozen=True, slots=True)
class PacingEvidence:
    spike_indices: tuple[int, ...] = ()
    validated_spike_indices: tuple[int, ...] = ()
    pacing_by_beat: Mapping[int, bool] = field(default_factory=dict)
    qrs_widths_ms: Mapping[int, float] = field(default_factory=dict)
    capture_alignment: Mapping[int, bool] = field(default_factory=dict)
    rr_ms: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PacingDecision:
    context: str
    paced_beat_ids: tuple[int, ...]
    qrs_override_ms: float | None
    selected_qrs_source: str
    events: tuple[PolicyEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class PacingPolicy:
    name: str = "default_pacing_policy"
    version: str = "1"

    def decide(self, evidence: PacingEvidence, *, context: PipelineContext) -> PolicyDecision:
        raise NotImplementedError(
            "a target-architecture pacing decision from abstract PacingEvidence is not implemented; "
            "the legacy-parity decisions are the decision-cluster policies in this module"
        )


# --------------------------------------------------------------------------- #
# Decision-cluster policy objects.
#
# Each ``decide`` takes exactly the arguments of the legacy helper(s) it wraps,
# returns exactly the value the legacy code computed, and describes the outcome
# in a PolicyEvent.  Event construction only reads the evidence.  The staged
# extractor applies the value exactly as the pre-decomposition code did.
# --------------------------------------------------------------------------- #

_POLICY = "pacing"


def _qrs_source_ids(*names: str) -> tuple[str, ...]:
    return tuple(f"qrs_detector:{name}" for name in names)


@dataclass(frozen=True, slots=True)
class PacingQRSRescueEvidencePolicy:
    """Does de-spiking look like it removed genuine, regularly captured QRS beats?"""

    name: str = f"{_POLICY}.qrs_rescue_evidence"

    def decide(
        self,
        *,
        pacing_result: Dict[str, Any],
        pre_despike_qrs_result: Any,
        despiked_qrs_result: Any,
        fs: int,
    ) -> PolicyDecision:
        evidence = _pacing_qrs_rescue_evidence(
            pacing_result=pacing_result,
            pre_despike_qrs_result=pre_despike_qrs_result,
            despiked_qrs_result=despiked_qrs_result,
            fs=fs,
        )
        supported = bool(evidence.get("applied"))
        return PolicyDecision(evidence, event(
            self.name,
            "accept" if supported else "no_change",
            str(evidence.get("reason")) if supported else "rescue_evidence_insufficient",
            _qrs_source_ids("pre_despike", "despiked"),
            pre_despike_n_beats=evidence.get("pre_despike_n_beats"),
            despiked_n_beats=evidence.get("despiked_n_beats"),
            capture_beats=evidence.get("capture_beats"),
        ))


@dataclass(frozen=True, slots=True)
class CleanedSingleLeadQRSRescuePolicy:
    """Recover missed paced beats from one reliable de-spiked lead, if one confirms them."""

    name: str = f"{_POLICY}.cleaned_single_lead_qrs_rescue"

    def decide(
        self,
        *,
        ecg_detect: np.ndarray,
        fs: int,
        quality: Dict[str, Any],
        pacing_result: Dict[str, Any],
        pre_despike_qrs_result: Any,
    ) -> PolicyDecision:
        candidate, detail = _pacing_qrs_cleaned_single_lead_rescue(
            ecg_detect=ecg_detect,
            fs=fs,
            quality=quality,
            pacing_result=pacing_result,
            pre_despike_qrs_result=pre_despike_qrs_result,
        )
        if candidate is None:
            outcome = event(self.name, "reject", "no_regular_cleaned_single_lead_confirmation",
                            _qrs_source_ids("pre_despike", "despiked"))
        else:
            outcome = event(
                self.name, "rescue", "paced_cleaned_single_lead_qrs_rescue",
                _qrs_source_ids("despiked") + (f"lead:{detail.get('rescue_lead')}",),
                n_beats=detail.get("cleaned_single_lead_n_beats"),
                match_fraction=detail.get("cleaned_single_lead_match_fraction"),
            )
        return PolicyDecision((candidate, detail), outcome)


@dataclass(frozen=True, slots=True)
class AttenuatedPacingRescuePolicy:
    """Accept attenuated pacer spikes only when QRS-validated capture is present."""

    name: str = f"{_POLICY}.attenuated_spike_rescue"

    def decide(
        self,
        *,
        ecg_rs: np.ndarray,
        ecg_an: np.ndarray,
        fs_run: int,
        lp_hz: float,
        current_pacing_result: Dict[str, Any],
        current_ecg_measure: np.ndarray,
        current_ecg_detect: np.ndarray,
        current_qrs_result: Any,
        current_r_locs: np.ndarray,
        pacing_detection_cache: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        result = _try_attenuated_pacing_rescue(
            ecg_rs=ecg_rs,
            ecg_an=ecg_an,
            fs_run=fs_run,
            lp_hz=lp_hz,
            current_pacing_result=current_pacing_result,
            current_ecg_measure=current_ecg_measure,
            current_ecg_detect=current_ecg_detect,
            current_qrs_result=current_qrs_result,
            current_r_locs=current_r_locs,
            pacing_detection_cache=pacing_detection_cache,
        )
        pacing_result = result[0]
        if pacing_result is current_pacing_result:
            outcome = event(self.name, "no_change", "attenuated_rescue_not_applied", ("pacing_detection",))
        else:
            outcome = event(
                self.name, "rescue", str(pacing_result.get("rescue_reason")),
                ("pacing_detection", "qrs_detector:attenuated_spike_cleaned"),
                prominence_uv=pacing_result.get("rescue_prominence_uv"),
                capture_fraction=pacing_result.get("rescue_capture_fraction"),
            )
        return PolicyDecision(result, outcome)


@dataclass(frozen=True, slots=True)
class PacingCaptureAlignmentPolicy:
    """Is ventricular capture by the detected spikes confirmed?"""

    name: str = f"{_POLICY}.capture_alignment"

    def decide(
        self,
        pacing_result: Dict[str, Any],
        spike_offsets_samples: List[int],
        fs: int,
        evidence_quality: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        confirmed = _pacing_capture_alignment_confirmed(
            pacing_result,
            spike_offsets_samples,
            fs,
            evidence_quality,
        )
        if confirmed:
            outcome = event(self.name, "accept", "ventricular_capture_alignment_confirmed",
                            ("pacing_detection",), n_aligned_spikes=len(spike_offsets_samples))
        elif bool(pacing_result.get("paced", False)):
            outcome = event(self.name, "reject", "capture_alignment_not_confirmed",
                            ("pacing_detection",), n_aligned_spikes=len(spike_offsets_samples))
        else:
            outcome = event(self.name, "no_change", "pacing_not_detected", ("pacing_detection",))
        return PolicyDecision(confirmed, outcome)


@dataclass(frozen=True, slots=True)
class PacedMeasurementGroupPolicy:
    """Is the selected measurement group made mostly of paced beats?"""

    name: str = f"{_POLICY}.paced_measurement_group"

    def decide(
        self,
        measurement_beat_ids: List[int],
        paced_beat_ids: List[int],
    ) -> PolicyDecision:
        paced_majority = _selected_group_paced_majority(measurement_beat_ids, paced_beat_ids)
        return PolicyDecision(paced_majority, event(
            self.name,
            "accept" if paced_majority else "no_change",
            "selected_group_paced_majority" if paced_majority else "selected_group_not_paced_majority",
            tuple(f"beat:{int(beat_id)}" for beat_id in measurement_beat_ids),
        ))


@dataclass(frozen=True, slots=True)
class IntermittentPacingWideContextPolicy:
    """Route intermittent pacing with a wide QRS offset consensus to paced measurement."""

    name: str = f"{_POLICY}.intermittent_wide_measurement_context"

    def decide(self, representative_leads: Dict[str, Any]) -> PolicyDecision:
        wide = _intermittent_pacing_wide_measurement_context(representative_leads)
        return PolicyDecision(wide, event(
            self.name,
            "accept" if wide else "no_change",
            "intermittent_pacing_wide_qrs_consensus" if wide else "no_intermittent_wide_qrs_consensus",
            ("representative_leads",),
        ))


@dataclass(frozen=True, slots=True)
class PacingLikeContextPolicy:
    """Wide or over-wide QRS with AV-relation evidence that behaves like pacing."""

    name: str = f"{_POLICY}.pacing_like_context"

    def decide(
        self,
        qrs_duration_ms: object,
        groups: Dict[int, Any],
        rule_summary: Dict[str, Any],
    ) -> PolicyDecision:
        wide = _wide_qrs_pacing_like_context(qrs_duration_ms, rule_summary)
        context = wide or _overwide_qrs_pacing_like_context(qrs_duration_ms, groups, rule_summary)
        if wide:
            reason = "wide_qrs_pacing_like_context"
        elif context:
            reason = "overwide_qrs_pacing_like_context"
        else:
            reason = "no_pacing_like_context"
        return PolicyDecision(context, event(
            self.name, "accept" if context else "no_change", reason,
            ("global_features.qrs_ms",), qrs_ms=_finite_float(qrs_duration_ms),
        ))


@dataclass(frozen=True, slots=True)
class PacingLikeQRSOverridePolicy:
    """QRS-duration override in a pacing-like context (first matching rule wins)."""

    name: str = f"{_POLICY}.pacing_like_qrs_override"

    def decide(
        self,
        qrs_duration_ms: object,
        representative_leads: Dict[str, Any],
        groups: Dict[int, Any],
        rule_summary: Dict[str, Any],
    ) -> PolicyDecision:
        rule = "raw_pacing_qrs_underestimate"
        override = _raw_pacing_qrs_underestimate_override_ms(
            qrs_duration_ms,
            representative_leads,
            rule_summary,
        )
        if override is None:
            rule = "near_wide_pacing_group"
            override = _near_wide_pacing_qrs_override_ms(
                qrs_duration_ms,
                groups,
                rule_summary,
            )
        if override is None:
            rule = "overwide_pacing_raw_consensus"
            override = _overwide_pacing_qrs_override_ms(
                qrs_duration_ms,
                representative_leads,
                groups,
                rule_summary,
            )
        return PolicyDecision(override, _override_event(self.name, rule, qrs_duration_ms, override))


@dataclass(frozen=True, slots=True)
class PacedQRSOverridePolicy:
    """QRS-duration override on the paced measurement route (first matching rule wins)."""

    name: str = f"{_POLICY}.paced_qrs_override"

    def decide(
        self,
        qrs_duration_ms: object,
        representative_leads: Dict[str, Any],
        groups: Dict[int, Any],
    ) -> PolicyDecision:
        rules = (
            ("overwide_paced_raw_consensus", _overwide_paced_qrs_raw_consensus_override_ms,
             (qrs_duration_ms, representative_leads, groups)),
            ("dominant_paced_wide_group", _dominant_paced_wide_qrs_override_ms,
             (qrs_duration_ms, groups)),
            ("near_wide_paced_qrs_offset", _near_wide_paced_qrs_offset_override_ms,
             (qrs_duration_ms, representative_leads, groups)),
            ("intermittent_paced_wide_qrs", _intermittent_paced_wide_qrs_override_ms,
             (qrs_duration_ms, representative_leads, groups)),
            ("borderline_paced_qrs_wide_offset", _borderline_paced_qrs_wide_offset_override_ms,
             (qrs_duration_ms, representative_leads)),
            ("secondary_paced_wide_group", _secondary_paced_wide_group_qrs_override_ms,
             (qrs_duration_ms, representative_leads, groups)),
        )
        override: Optional[float] = None
        rule = rules[-1][0]
        for rule, helper, arguments in rules:
            override = helper(*arguments)
            if override is not None:
                break
        return PolicyDecision(override, _override_event(self.name, rule, qrs_duration_ms, override))


def _override_event(policy: str, rule: str, measured: object, override: Optional[float]) -> PolicyEvent:
    if override is None:
        return event(policy, "no_change", "no_override_rule_matched", ("global_features.qrs_ms",),
                     measured_ms=_finite_float(measured))
    return event(policy, "override", rule, ("global_features.qrs_ms",),
                 measured_ms=_finite_float(measured), override_ms=_finite_float(override))


@dataclass(frozen=True, slots=True)
class PacingMeasurementEffectPolicy:
    """How pacing affected the global measurement route."""

    name: str = f"{_POLICY}.measurement_effect"

    def decide(
        self,
        *,
        pacing_enabled: bool,
        pacing_result: Dict[str, Any],
        measurement_group_paced: bool,
        global_measurement_paced: bool,
        pacing_capture_confirmed: bool,
        paced_qrs_floor_beat_ids: List[int],
    ) -> PolicyDecision:
        effect = _pacing_measurement_effect(
            pacing_enabled=pacing_enabled,
            pacing_result=pacing_result,
            measurement_group_paced=measurement_group_paced,
            global_measurement_paced=global_measurement_paced,
            pacing_capture_confirmed=pacing_capture_confirmed,
            paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
        )
        return PolicyDecision(effect, _effect_event(self.name, effect))


@dataclass(frozen=True, slots=True)
class PacingSegmentationEffectPolicy:
    """How pacing affected beat segmentation."""

    name: str = f"{_POLICY}.segmentation_effect"

    def decide(
        self,
        *,
        pacing_enabled: bool,
        pacing_result: Dict[str, Any],
        measurement_group_paced: bool,
        segmentation_paced_beat_ids: List[int],
        paced_qrs_floor_beat_ids: List[int],
    ) -> PolicyDecision:
        effect = _pacing_segmentation_effect(
            pacing_enabled=pacing_enabled,
            pacing_result=pacing_result,
            measurement_group_paced=measurement_group_paced,
            segmentation_paced_beat_ids=segmentation_paced_beat_ids,
            paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
        )
        return PolicyDecision(effect, _effect_event(self.name, effect))


@dataclass(frozen=True, slots=True)
class MeasurementPacingStatePolicy:
    """The pacing state reported for measurement (off / unknown / on)."""

    name: str = f"{_POLICY}.measurement_state"

    def decide(
        self,
        *,
        pacing_enabled: bool,
        detection_state: object,
        confirmed_pacing_context: bool,
        pacing_measurement_effect: str,
    ) -> PolicyDecision:
        state = _measurement_pacing_state(
            pacing_enabled=pacing_enabled,
            detection_state=detection_state,
            confirmed_pacing_context=confirmed_pacing_context,
            pacing_measurement_effect=pacing_measurement_effect,
        )
        return PolicyDecision(state, event(
            self.name, "accept", f"measurement_pacing_state_{state}", ("pacing_detection",),
            detection_state=str(detection_state or "off"),
        ))


def _effect_event(policy: str, effect: str) -> PolicyEvent:
    inert = effect in {"disabled", "none", "metadata_only"}
    return event(policy, "no_change" if inert else "accept", effect, ("pacing_detection",))


PACING_QRS_RESCUE_EVIDENCE = PacingQRSRescueEvidencePolicy()
CLEANED_SINGLE_LEAD_QRS_RESCUE = CleanedSingleLeadQRSRescuePolicy()
ATTENUATED_PACING_RESCUE = AttenuatedPacingRescuePolicy()
PACING_CAPTURE_ALIGNMENT = PacingCaptureAlignmentPolicy()
PACED_MEASUREMENT_GROUP = PacedMeasurementGroupPolicy()
INTERMITTENT_PACING_WIDE_CONTEXT = IntermittentPacingWideContextPolicy()
PACING_LIKE_CONTEXT = PacingLikeContextPolicy()
PACING_LIKE_QRS_OVERRIDE = PacingLikeQRSOverridePolicy()
PACED_QRS_OVERRIDE = PacedQRSOverridePolicy()
PACING_MEASUREMENT_EFFECT = PacingMeasurementEffectPolicy()
PACING_SEGMENTATION_EFFECT = PacingSegmentationEffectPolicy()
MEASUREMENT_PACING_STATE = MeasurementPacingStatePolicy()


__all__ = [
    "PacingEvidence", "PacingDecision", "PacingPolicy",
    "PacingQRSRescueEvidencePolicy", "CleanedSingleLeadQRSRescuePolicy", "AttenuatedPacingRescuePolicy",
    "PacingCaptureAlignmentPolicy", "PacedMeasurementGroupPolicy", "IntermittentPacingWideContextPolicy",
    "PacingLikeContextPolicy", "PacingLikeQRSOverridePolicy", "PacedQRSOverridePolicy",
    "PacingMeasurementEffectPolicy", "PacingSegmentationEffectPolicy", "MeasurementPacingStatePolicy",
    "PACING_QRS_RESCUE_EVIDENCE", "CLEANED_SINGLE_LEAD_QRS_RESCUE", "ATTENUATED_PACING_RESCUE",
    "PACING_CAPTURE_ALIGNMENT", "PACED_MEASUREMENT_GROUP", "INTERMITTENT_PACING_WIDE_CONTEXT",
    "PACING_LIKE_CONTEXT", "PACING_LIKE_QRS_OVERRIDE", "PACED_QRS_OVERRIDE",
    "PACING_MEASUREMENT_EFFECT", "PACING_SEGMENTATION_EFFECT", "MEASUREMENT_PACING_STATE",
]
