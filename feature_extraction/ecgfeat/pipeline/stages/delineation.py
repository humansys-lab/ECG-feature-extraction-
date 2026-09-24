"""Delineation stage: fiducial delineation and opt-in boundary refinements.

``run`` is primary delineation; ``settle_qrs_tail_and_refine_t`` runs after the beats
stage refined the measurement group (QRS tail-settling rescue, then T refinement).

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List

from ..._engine.delineation.core import apply_systematic_qrs_tail_settling_rescue, delineate_beats
from ..._engine.delineation.r_localization import apply_hybrid_r_localization
from ..._engine.delineation.t_refinement import refine_t_wave_boundaries
from ..._engine.delineation.wave_localization import apply_hybrid_wave_localization
from ..._engine.measurement.st_localization import apply_hybrid_st_measurement
from ..context import PipelineContext, StageResult, require


@dataclass(frozen=True, slots=True)
class DelineationBundle:
    """Engine state produced by the delineation stage (legacy variable names).

    Fields reference live engine objects; see the ownership-transfer rule in
    ``ecgfeat.pipeline.context``.
    """

    beat_features: Any
    qrs_tail_settling_rescue: Any = None
    t_fusion_audit: Any = None


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 2117-2150 (``self`` is ``context.config``)."""
    context = require(context, 'delineation', 'config', 'ventricular', 'input', 'beats', 'quality')
    settings = context.config
    ecg_measure = context.ventricular.ecg_measure
    fs_run = context.input.fs_run
    r_locs = context.ventricular.r_locs
    beat_groups = context.beats.beat_groups
    _representative = context.beats._representative
    segmentation_paced_beat_ids = context.beats.segmentation_paced_beat_ids
    paced_qrs_floor_beat_ids = context.beats.paced_qrs_floor_beat_ids
    pacing_result = context.ventricular.pacing_result
    quality = context.quality.quality
    qrs_result = context.ventricular.qrs_result
    rep_left_ms = context.quality.rep_left_ms
    refinement_kwargs = context.quality.refinement_kwargs
    ecg_detect = context.ventricular.ecg_detect
    beat_features = delineate_beats(
        ecg_measure, fs_run, r_locs,
        beat_groups=beat_groups,
        rep_beats=_representative,
        paced_beat_ids=segmentation_paced_beat_ids,
        paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
        pacing_spike_times=pacing_result["spike_times"],
        quality=quality,
        qrs_low_slope_guard=qrs_result.fallback_used,
        rep_left_ms=rep_left_ms,
        **refinement_kwargs,
    )
    if settings.enable_hybrid_r_localization:
        apply_hybrid_r_localization(
            beat_features,
            localization_ecg=ecg_detect,
            r_locs=r_locs,
            fs=fs_run,
        )
    if settings.enable_hybrid_wave_localization:
        apply_hybrid_wave_localization(
            beat_features,
            localization_ecg=ecg_measure,
            r_locs=r_locs,
            fs=fs_run,
        )
    if settings.enable_hybrid_st_measurement:
        apply_hybrid_st_measurement(
            beat_features,
            measurement_ecg=ecg_measure,
            r_locs=r_locs,
            fs=fs_run,
            quality=quality,
        )
    bundle = DelineationBundle(
        beat_features=beat_features,
    )
    context = replace(context, delineation=bundle)
    return StageResult(context)


def settle_qrs_tail_and_refine_t(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 2170-2233 (``self`` is ``context.config``)."""
    context = require(context, 'delineation.settle_qrs_tail_and_refine_t', 'config', 'beats', 'delineation', 'ventricular', 'input', 'quality')
    settings = context.config
    confirmed_pacing_context = context.beats.confirmed_pacing_context
    beat_features = context.delineation.beat_features
    ecg_measure = context.ventricular.ecg_measure
    fs_run = context.input.fs_run
    quality = context.quality.quality
    r_locs = context.ventricular.r_locs
    measurement_beat_ids = context.beats.measurement_beat_ids
    ecg_detect = context.ventricular.ecg_detect
    refinement_kwargs = context.quality.refinement_kwargs
    qrs_tail_settling_rescue: Dict[str, Any] = {
        "applied": False,
        "reason": "confirmed_pacing_context"
        if confirmed_pacing_context
        else "not_evaluated",
    }
    if not confirmed_pacing_context:
        beat_features, qrs_tail_settling_rescue = (
            apply_systematic_qrs_tail_settling_rescue(
                beat_features,
                ecg=ecg_measure,
                fs=fs_run,
                quality=quality,
                r_locs=r_locs,
                selected_beat_ids=measurement_beat_ids,
            )
        )
        if qrs_tail_settling_rescue.get("applied"):
            if settings.enable_hybrid_r_localization:
                apply_hybrid_r_localization(
                    beat_features,
                    localization_ecg=ecg_detect,
                    r_locs=r_locs,
                    fs=fs_run,
                )
            if settings.enable_hybrid_wave_localization:
                apply_hybrid_wave_localization(
                    beat_features,
                    localization_ecg=ecg_measure,
                    r_locs=r_locs,
                    fs=fs_run,
                )
            if settings.enable_hybrid_st_measurement:
                apply_hybrid_st_measurement(
                    beat_features,
                    measurement_ecg=ecg_measure,
                    r_locs=r_locs,
                    fs=fs_run,
                    quality=quality,
                )
    # Refine T only after the QRS terminal-tail decision.  T onset is a
    # guard for that rescue, so changing it earlier creates a circular
    # dependency and can suppress a valid late QRS offset.
    t_fusion_audit: List[Dict[str, Any]] = []
    if settings.enable_t_wave_refinement:
        refine_t_wave_boundaries(
            beat_features,
            measurement_ecg=ecg_measure,
            r_locs=r_locs,
            fs=fs_run,
            quality=quality,
            **({"fusion_audit": t_fusion_audit} if (settings.refinement.t_correlated_fusion
                or settings.refinement.t_boundary_projection or settings.refinement.t_sequence_selection
                or settings.refinement.t_projection_offset_only or settings.refinement.t_sequence_offset_only) else {}),
            **refinement_kwargs,
        )
        if settings.enable_hybrid_st_measurement:
            apply_hybrid_st_measurement(
                beat_features,
                measurement_ecg=ecg_measure,
                r_locs=r_locs,
                fs=fs_run,
                quality=quality,
            )
    bundle = replace(
        context.delineation,
        qrs_tail_settling_rescue=qrs_tail_settling_rescue,
        beat_features=beat_features,
        t_fusion_audit=t_fusion_audit,
    )
    context = replace(context, delineation=bundle)
    return StageResult(context)


__all__ = ['DelineationBundle', 'run', 'settle_qrs_tail_and_refine_t']
