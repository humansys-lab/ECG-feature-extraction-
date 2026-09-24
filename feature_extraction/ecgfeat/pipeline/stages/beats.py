"""Beats stage: grouping, representatives and measurement-group selection.

``refine_measurement_group`` is the stage-owned refinement the extractor runs after
primary delineation (doc 04: the beats stage keeps ownership of group selection).

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.

Private helpers below were moved verbatim from ``ecgfeat/api.py``
(docs/library_design/01_architecture.md, "Destination of all 49 private helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..._engine.beats.grouping import build_beat_annotations, cluster_beats
from ..._engine.beats.representative import build_representative_beats_with_meta
from ..._engine.foundation.numeric import _finite_float
from ..context import (
    PipelineContext,
    PolicyEvent,
    StageResult,
    record_decision,
    require,
    with_policy_events,
)
from ..policies.pacing import PACED_MEASUREMENT_GROUP, _paced_beat_fraction


_MEASUREMENT_RESELECT_MIN_NATIVE_BEATS = 2
_MEASUREMENT_RESELECT_NATIVE_MAX_QRS_MS = 115.0
_MEASUREMENT_RESELECT_NATIVE_MAX_SPREAD_MS = 25.0
_MEASUREMENT_RESELECT_SELECTED_MIN_QRS_MS = 115.0
_MEASUREMENT_RESELECT_MIN_QRS_DELTA_MS = 25.0
_MEASUREMENT_RESELECT_STABLE_TEMPLATE_CORR = 0.95
_MEASUREMENT_RESELECT_STABLE_TEMPLATE_MAX_OUTLIER_FRAC = 0.25


def _select_measurement_group(
    beat_groups: Dict[int, List[int]],
    paced_beat_ids: List[int],
) -> Tuple[int, List[int]]:
    if not beat_groups:
        return 1, []

    paced_set = {int(beat_id) for beat_id in paced_beat_ids}
    candidates: List[Tuple[int, int, int, List[int]]] = []
    fallback_gid = 1
    fallback_members: List[int] = []
    all_members: List[int] = []

    for gid, members in beat_groups.items():
        members = [int(beat_id) for beat_id in members]
        all_members.extend(members)
        if not fallback_members:
            fallback_gid = int(gid)
            fallback_members = members
    paced_majority = bool(all_members) and sum(1 for beat_id in all_members if beat_id in paced_set) > (
        len(all_members) / 2.0
    )

    for gid, members in beat_groups.items():
        members = [int(beat_id) for beat_id in members]
        selected_members = (
            [beat_id for beat_id in members if beat_id in paced_set]
            if paced_majority
            else [beat_id for beat_id in members if beat_id not in paced_set]
        )
        if selected_members:
            candidates.append((len(selected_members), len(members), -int(gid), selected_members))

    if not candidates:
        return fallback_gid, fallback_members

    candidates.sort(reverse=True)
    _, _, neg_gid, selected_members = candidates[0]
    return -neg_gid, selected_members


def _beat_qrs_profiles(
    beat_features: List[Any],
) -> Tuple[Dict[int, float], Dict[int, bool]]:
    by_beat: Dict[int, List[float]] = {}
    paced_floor_by_beat: Dict[int, bool] = {}
    for feature in beat_features:
        beat_id_raw = getattr(feature, "beat_id", None)
        if beat_id_raw is None:
            continue
        beat_id = int(beat_id_raw)
        flags = set(getattr(feature, "flags", []) or [])
        if "paced_floor_applied" in flags:
            paced_floor_by_beat[beat_id] = True
        qrs_ms = _finite_float(getattr(feature, "qrs_ms", None))
        if qrs_ms is None or not (20.0 <= qrs_ms <= 280.0):
            continue
        by_beat.setdefault(beat_id, []).append(float(qrs_ms))

    qrs_by_beat = {
        beat_id: float(np.median(values))
        for beat_id, values in by_beat.items()
        if values
    }
    return qrs_by_beat, paced_floor_by_beat


def _group_qrs_profile(
    members: List[int],
    *,
    paced_set: set[int],
    qrs_by_beat: Dict[int, float],
    paced_floor_by_beat: Dict[int, bool],
) -> Optional[Dict[str, float]]:
    values = [
        float(qrs_by_beat[int(beat_id)])
        for beat_id in members
        if int(beat_id) in qrs_by_beat
    ]
    if not values:
        return None
    floor_count = sum(1 for beat_id in members if paced_floor_by_beat.get(int(beat_id), False))
    paced_count = sum(1 for beat_id in members if int(beat_id) in paced_set)
    spread = (
        float(np.percentile(values, 75) - np.percentile(values, 25))
        if len(values) >= 2
        else 0.0
    )
    return {
        "n": float(len(values)),
        "qrs_median": float(np.median(values)),
        "qrs_spread": spread,
        "paced_fraction": float(paced_count) / float(len(members)) if members else 0.0,
        "floor_fraction": float(floor_count) / float(len(members)) if members else 0.0,
    }


def _refine_measurement_group_after_delineation(
    beat_groups: Dict[int, List[int]],
    paced_beat_ids: List[int],
    selected_group_id: int,
    selected_members: List[int],
    beat_features: List[Any],
    representative_meta: Optional[Dict[int, Any]] = None,
) -> Tuple[int, List[int], Optional[str]]:
    paced_set = {int(beat_id) for beat_id in paced_beat_ids}
    if not paced_set or not selected_members:
        return selected_group_id, selected_members, None

    qrs_by_beat, paced_floor_by_beat = _beat_qrs_profiles(beat_features)
    selected_profile = _group_qrs_profile(
        [int(beat_id) for beat_id in selected_members],
        paced_set=paced_set,
        qrs_by_beat=qrs_by_beat,
        paced_floor_by_beat=paced_floor_by_beat,
    )
    if selected_profile is None or selected_profile["paced_fraction"] <= 0.50:
        return selected_group_id, selected_members, None
    if (
        selected_profile["qrs_median"] < _MEASUREMENT_RESELECT_SELECTED_MIN_QRS_MS
        and selected_profile["floor_fraction"] < 0.50
    ):
        return selected_group_id, selected_members, None
    selected_meta = (representative_meta or {}).get(int(selected_group_id))
    selected_corr = _finite_float(getattr(selected_meta, "mean_template_corr", None))
    selected_member_count = _finite_float(getattr(selected_meta, "member_count", None))
    selected_outlier_count = _finite_float(getattr(selected_meta, "outlier_count", None))
    if (
        selected_corr is not None
        and selected_corr >= _MEASUREMENT_RESELECT_STABLE_TEMPLATE_CORR
        and selected_member_count is not None
        and selected_member_count > 0
        and selected_outlier_count is not None
        and selected_outlier_count / selected_member_count
        <= _MEASUREMENT_RESELECT_STABLE_TEMPLATE_MAX_OUTLIER_FRAC
    ):
        return selected_group_id, selected_members, None

    best: Optional[Tuple[float, int, List[int]]] = None
    for gid, members_raw in beat_groups.items():
        gid_int = int(gid)
        if gid_int == int(selected_group_id):
            continue
        members = [int(beat_id) for beat_id in members_raw]
        profile = _group_qrs_profile(
            members,
            paced_set=paced_set,
            qrs_by_beat=qrs_by_beat,
            paced_floor_by_beat=paced_floor_by_beat,
        )
        if profile is None:
            continue
        if profile["n"] < _MEASUREMENT_RESELECT_MIN_NATIVE_BEATS:
            continue
        if profile["paced_fraction"] > 0.0:
            continue
        if profile["qrs_median"] > _MEASUREMENT_RESELECT_NATIVE_MAX_QRS_MS:
            continue
        if profile["qrs_spread"] > _MEASUREMENT_RESELECT_NATIVE_MAX_SPREAD_MS:
            continue
        if (
            selected_profile["qrs_median"] - profile["qrs_median"]
            < _MEASUREMENT_RESELECT_MIN_QRS_DELTA_MS
            and selected_profile["floor_fraction"] < 0.50
        ):
            continue
        score = (
            profile["n"] * 10.0
            - profile["qrs_spread"]
            - max(0.0, profile["qrs_median"] - 90.0) * 0.25
        )
        candidate = (score, -gid_int, members)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return selected_group_id, selected_members, None

    _, neg_gid, members = best
    return -neg_gid, members, "stable_native_family"



@dataclass(frozen=True, slots=True)
class BeatBundle:
    """Engine state produced by the beats stage (legacy variable names).

    Fields reference live engine objects; see the ownership-transfer rule in
    ``ecgfeat.pipeline.context``.
    """

    beat_groups: Any
    dominant_group_id: Any
    measurement_beat_ids: Any
    measurement_group_paced: Any
    confirmed_pacing_context: Any
    paced_fraction: Any
    segmentation_paced_beat_ids: Any
    paced_qrs_floor_beat_ids: Any
    beats: Any
    _representative: Any
    representative_meta: Any
    initial_measurement_group_id: Any = None
    initial_measurement_beat_ids: Any = None
    measurement_reselect_reason: Any = None


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 2078-2116 (``self`` is ``context.config``)."""
    context = require(context, 'beats', 'config', 'ventricular', 'input', 'quality')
    settings = context.config
    events: List[PolicyEvent] = []
    ecg_measure = context.ventricular.ecg_measure
    r_locs = context.ventricular.r_locs
    fs_run = context.input.fs_run
    paced_beat_ids = context.ventricular.paced_beat_ids
    pacing_result = context.ventricular.pacing_result
    pacing_capture_confirmed = context.ventricular.pacing_capture_confirmed
    representative_options = context.quality.representative_options
    beat_groups = (
        cluster_beats(ecg_measure, r_locs, fs_run, paced_beat_ids=paced_beat_ids,
                      **({"preserve_outliers": True} if settings.refinement.grouping_outliers else {}))
        if settings.compute_grouping
        else {1: list(range(len(r_locs)))}
    )
    dominant_group_id, measurement_beat_ids = _select_measurement_group(beat_groups, paced_beat_ids)
    measurement_group_paced = bool(
        pacing_result.get("paced", False)
        and record_decision(events, PACED_MEASUREMENT_GROUP.decide(measurement_beat_ids, paced_beat_ids))
    )
    confirmed_pacing_context = bool(measurement_group_paced or pacing_capture_confirmed)
    paced_fraction = _paced_beat_fraction(paced_beat_ids, len(r_locs))
    localized_pacing_segmentation = bool(
        pacing_capture_confirmed
        and (
            measurement_group_paced
            or paced_fraction <= 0.50
        )
    )
    segmentation_paced_beat_ids = (
        list(paced_beat_ids)
        if localized_pacing_segmentation
        else []
    )
    paced_qrs_floor_beat_ids = (
        list(segmentation_paced_beat_ids)
        if localized_pacing_segmentation and pacing_capture_confirmed
        else []
    )
    beats = build_beat_annotations(r_locs, fs_run, beat_groups, paced_beat_ids=paced_beat_ids)
    _representative, representative_meta = build_representative_beats_with_meta(
        ecg_measure,
        r_locs,
        beat_groups,
        fs_run,
        paced_beat_ids=paced_beat_ids,
        **representative_options,
    )
    bundle = BeatBundle(
        beat_groups=beat_groups,
        dominant_group_id=dominant_group_id,
        measurement_beat_ids=measurement_beat_ids,
        measurement_group_paced=measurement_group_paced,
        confirmed_pacing_context=confirmed_pacing_context,
        paced_fraction=paced_fraction,
        segmentation_paced_beat_ids=segmentation_paced_beat_ids,
        paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
        beats=beats,
        _representative=_representative,
        representative_meta=representative_meta,
    )
    context = replace(context, beats=bundle)
    context = with_policy_events(context, events)
    return StageResult(context)


def refine_measurement_group(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 2151-2169 (``self`` is ``context.config``)."""
    context = require(context, 'beats.refine_measurement_group', 'beats', 'ventricular', 'delineation')
    events: List[PolicyEvent] = []
    dominant_group_id = context.beats.dominant_group_id
    measurement_beat_ids = context.beats.measurement_beat_ids
    beat_groups = context.beats.beat_groups
    paced_beat_ids = context.ventricular.paced_beat_ids
    beat_features = context.delineation.beat_features
    representative_meta = context.beats.representative_meta
    pacing_result = context.ventricular.pacing_result
    pacing_capture_confirmed = context.ventricular.pacing_capture_confirmed
    initial_measurement_group_id = dominant_group_id
    initial_measurement_beat_ids = list(measurement_beat_ids)
    (
        dominant_group_id,
        measurement_beat_ids,
        measurement_reselect_reason,
    ) = _refine_measurement_group_after_delineation(
        beat_groups,
        paced_beat_ids,
        dominant_group_id,
        measurement_beat_ids,
        beat_features,
        representative_meta,
    )
    measurement_group_paced = bool(
        pacing_result.get("paced", False)
        and record_decision(events, PACED_MEASUREMENT_GROUP.decide(measurement_beat_ids, paced_beat_ids))
    )
    confirmed_pacing_context = bool(measurement_group_paced or pacing_capture_confirmed)
    bundle = replace(
        context.beats,
        initial_measurement_group_id=initial_measurement_group_id,
        initial_measurement_beat_ids=initial_measurement_beat_ids,
        dominant_group_id=dominant_group_id,
        measurement_beat_ids=measurement_beat_ids,
        measurement_reselect_reason=measurement_reselect_reason,
        measurement_group_paced=measurement_group_paced,
        confirmed_pacing_context=confirmed_pacing_context,
    )
    context = replace(context, beats=bundle)
    context = with_policy_events(context, events)
    return StageResult(context)


__all__ = ['BeatBundle', 'run', 'refine_measurement_group']
