"""Atrial stage: P-wave assessments, atrial events and measurement-side AV evidence.

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.

Private helpers below were moved verbatim from ``ecgfeat/api.py``
(docs/library_design/01_architecture.md, "Destination of all 49 private helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict

from ..._engine.atrial.core import (
    _classify_af_afl,
    build_qrst_subtracted_residual,
    compute_organized_p_ratio,
    compute_pr_dispersion_ms,
    extract_atrial_events,
)
from ..._engine.atrial.p_wave import (
    PWaveConfig,
    backfill_missing_p_from_robust_engine,
    build_p_wave_assessments,
    finalize_p_wave_states,
)
from ..._engine.delineation.refinement import correct_p_boundaries
from ..._engine.measurement.st_localization import apply_hybrid_st_measurement
from ..context import PipelineContext, StageResult, require


def _av_block_evidence(rule_summary: Dict[str, Any]) -> Dict[str, Any]:
    pauses = rule_summary.get("pauses") if isinstance(rule_summary, dict) else {}
    evidence = pauses.get("av_block_evidence", {}) if isinstance(pauses, dict) else {}
    return evidence if isinstance(evidence, dict) else {}



@dataclass(frozen=True, slots=True)
class AtrialBundle:
    """Engine state produced by the atrial stage (legacy variable names).

    Fields reference live engine objects; see the ownership-transfer rule in
    ``ecgfeat.pipeline.context``.
    """

    p_wave_assessments: Any
    atrial_events: Any
    atrial_validation_audit: Any
    atrial_residual: Any
    rr_ms: Any
    af_afl_summary: Any
    pristine_p_wave_states: Any
    p_corrections: Any


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 2234-2324 (``self`` is ``context.config``)."""
    context = require(context, 'atrial', 'config', 'input', 'ventricular', 'delineation', 'quality', 'beats')
    settings = context.config
    available_leads = context.input.available_leads
    ecg_measure = context.ventricular.ecg_measure
    fs_run = context.input.fs_run
    r_locs = context.ventricular.r_locs
    beat_features = context.delineation.beat_features
    quality = context.quality.quality
    beat_groups = context.beats.beat_groups
    acquisition_qc = context.ventricular.acquisition_qc
    p_config_kwargs = dict(model_arbitration=settings.refinement.p_model_arbitration,
                           multiple_candidates=settings.refinement.p_multiple_candidates)
    if available_leads is not None:
        p_config_kwargs.update(minimum_informative_leads=min(2, len(available_leads)),
                               minimum_independent_groups=1)
    p_wave_assessments = build_p_wave_assessments(
        ecg_measure,
        fs_run,
        r_locs,
        beat_features,
        quality,
        beat_groups=beat_groups,
        acquisition_qc=acquisition_qc,
        **({"config": PWaveConfig(**p_config_kwargs)}
           if available_leads is not None or settings.refinement.p_model_arbitration or settings.refinement.p_multiple_candidates else {}),
    )
    atrial_events = extract_atrial_events(beat_features, quality, r_locs, fs_run, ecg=ecg_measure)
    atrial_validation_audit: Dict[str, Any] = {}
    if settings.refinement.atrial_event_validation:
        from ..._engine.atrial.validation import validate_atrial_events
        atrial_events = validate_atrial_events(atrial_events, ecg_measure, fs_run, beat_features,
                                              quality, audit=atrial_validation_audit)
    atrial_residual = build_qrst_subtracted_residual(
        ecg_measure,
        fs_run,
        r_locs,
        beat_features,
        quality,
    )
    rr_ms = [
        float(r_locs[idx + 1] - r_locs[idx]) * 1000.0 / float(fs_run)
        for idx in range(max(0, len(r_locs) - 1))
    ]
    organized_p_ratio = compute_organized_p_ratio(atrial_events, len(r_locs))
    pr_dispersion_ms = compute_pr_dispersion_ms(atrial_events)
    af_afl_summary = _classify_af_afl(
        rr_ms, atrial_residual, organized_p_ratio, pr_dispersion_ms
    )
    # Snapshot the boundary-quality verdict before the rhythm verdict is
    # stamped onto it.  `finalize_p_wave_states` is destructive: on an AF or
    # flutter call it forces `accepted=False` on every beat with no pristine
    # copy retained, so it cannot be re-run to a correct answer later (a
    # second call finds `accepted` already False and lands every beat in
    # OVERLAP_UNCERTAIN).  The flutter verdict read here can still be
    # retracted further down (`flutter_suppressed_by`), and that retraction
    # has to be able to give these beats their P waves back -- hence the
    # snapshot.  See the restore below.
    pristine_p_wave_states = [
        (
            assessment.accepted,
            list(assessment.reject_reasons),
            assessment.p_state,
        )
        for assessment in p_wave_assessments
    ]
    finalize_p_wave_states(p_wave_assessments, af_afl_summary)
    p_corrections = []
    if settings.refinement.p_boundary_correction:
        p_corrections = correct_p_boundaries(beat_features, p_wave_assessments, ecg_measure, fs_run)
        if p_corrections:
            if settings.enable_hybrid_st_measurement:
                # Hybrid ST uses the accepted P offset to choose its PR
                # baseline. Correcting P invalidates that measurement.
                apply_hybrid_st_measurement(
                    beat_features, measurement_ecg=ecg_measure,
                    r_locs=r_locs, fs=fs_run, quality=quality,
                )
            # Rebuild dependent atrial evidence once, then reapply the
            # rhythm gate to the pristine boundary verdict (not a verdict
            # already destructively finalized by the previous rhythm).
            atrial_events = extract_atrial_events(beat_features, quality, r_locs, fs_run, ecg=ecg_measure)
            if settings.refinement.atrial_event_validation:
                atrial_events = validate_atrial_events(atrial_events, ecg_measure, fs_run, beat_features,
                                                      quality, audit=atrial_validation_audit)
            organized_p_ratio = compute_organized_p_ratio(atrial_events, len(r_locs))
            pr_dispersion_ms = compute_pr_dispersion_ms(atrial_events)
            af_afl_summary = _classify_af_afl(rr_ms, atrial_residual, organized_p_ratio, pr_dispersion_ms)
            for assessment, (accepted, reasons, state) in zip(p_wave_assessments, pristine_p_wave_states):
                assessment.accepted, assessment.reject_reasons, assessment.p_state = accepted, list(reasons), state
            finalize_p_wave_states(p_wave_assessments, af_afl_summary)
    # Ordering note: the backfill must stay downstream of a finalize, not be
    # deferred to the restore below.  Its AF-safety rests entirely on the
    # `assessment.accepted` gate, which only excludes AF_LIKE /
    # ORGANIZED_ATRIAL_ACTIVITY beats once a finalize has stamped them; run
    # against the raw boundary verdict it would backfill P waves onto
    # fibrillating beats.  It also writes `feature.p`, which feeds
    # `_physiologic_pr_core_from_beats` and therefore the retraction decision
    # itself, so moving it would make that decision circular.
    backfill_missing_p_from_robust_engine(
        beat_features, p_wave_assessments, ecg_measure, fs_run
    )
    bundle = AtrialBundle(
        p_wave_assessments=p_wave_assessments,
        atrial_events=atrial_events,
        atrial_validation_audit=atrial_validation_audit,
        atrial_residual=atrial_residual,
        rr_ms=rr_ms,
        af_afl_summary=af_afl_summary,
        pristine_p_wave_states=pristine_p_wave_states,
        p_corrections=p_corrections,
    )
    context = replace(context, atrial=bundle)
    return StageResult(context)


__all__ = ['AtrialBundle', 'run']
