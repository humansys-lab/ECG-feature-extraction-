"""Measurement stage: representative, group and global measurement assembly.

The legacy call order interleaves rhythm-context and pacing decisions with measurement
(flutter retraction, P-state re-finalization, pacing-like QRS overrides, availability);
they execute here, in that order.  The rhythm statement-evidence hook is invoked at its
exact legacy point when the context carries interpretation hooks.

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.

Private helpers below were moved verbatim from ``ecgfeat/api.py``
(docs/library_design/01_architecture.md, "Destination of all 49 private helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np

from ..._engine.atrial.p_wave import finalize_p_wave_states
from ..._engine.beats.representative import build_representative_beats_with_meta
from ..._engine.delineation.core import delineate_beats
from ..._engine.delineation.t_refinement import refine_t_wave_boundaries
from ..._engine.foundation.numeric import _finite_float, _median_or_none
from ..._engine.measurement.features import (
    _axis_delta_deg,
    _axis_from_amplitudes,
    _low_support_limb_t_axis_coverage_values,
    _physiologic_pr_core_from_beats,
    _select_reliable_qt_leads,
    _st_supported_limb_t_axis_coverage_values,
    _stable_limb_signed_t_axis_deg,
    build_representative_lead_features,
    compute_global_features,
    compute_group_features,
    estimate_initial_qrs_axis_deg,
    estimate_pr_segment_ms,
)
from ..._engine.measurement.profiles.twelve_sl import apply_twelve_sl_measurement_profile
from ..._engine.measurement.st_baseline import adaptive_st_signal, calibrated_st_signal
from ..._engine.measurement.st_localization import apply_hybrid_st_measurement
from ..._engine.quality.acquisition import apply_channel_delay_compensation
from ..._engine.quality.signal import (
    compute_adjacent_precordial_correlations,
    remove_pacing_spikes,
)
from ...models import STANDARD_12_LEADS
from ...rhythm_rules import (
    build_measurement_availability,
    classify_pacing_context,
    classify_post_pause_or_interpolated_beats,
    detect_av_block_availability_flags,
    detect_pauses_and_av_block,
    detect_preexcitation,
    select_measurement_beat_ids,
)
from ..context import (
    PipelineContext,
    PolicyEvent,
    StageResult,
    record_decision,
    require,
    with_policy_events,
)
from ..policies.applicability import ATRIAL_MEASUREMENT_VALIDITY, PR_AVAILABILITY
from ..policies.lead_integrity import LIMB_LEAD_REVERSAL, PRECORDIAL_REVERSAL_FLAGS
from ..policies.pacing import (
    INTERMITTENT_PACING_WIDE_CONTEXT,
    MEASUREMENT_PACING_STATE,
    PACED_QRS_OVERRIDE,
    PACING_LIKE_CONTEXT,
    PACING_LIKE_QRS_OVERRIDE,
    PACING_MEASUREMENT_EFFECT,
    PACING_SEGMENTATION_EFFECT,
    _build_pacing_evidence_by_beat,
)
from ..policies.qt import INTERMITTENT_PACED_QT_RESCUE, QT_TAIL_SETTLING_RESCUE


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
    """Engine state produced by the measurement stage (legacy variable names).

    Fields reference live engine objects; see the ownership-transfer rule in
    ``ecgfeat.pipeline.context``.
    """

    measurement_representative_meta: Any
    st_baseline_metadata: Any
    representative_leads: Any
    precordial_reversal: Any
    groups: Any
    global_measurement_paced: Any
    global_features: Any
    pacing_qt_rescue_source: Any
    af_afl_summary: Any
    twelve_sl_profile: Any
    pacing_context: Any
    policy_measurement_beat_ids: Any
    rule_summary: Any
    availability: Any
    t_axis_profile_backfill: Any
    reliable_qt_leads: Any
    pacing_measurement_effect: Any
    pacing_segmentation_effect: Any
    pacing_evidence_by_beat: Any
    pacing_detection_state: Any
    measurement_pacing_state: Any


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 2325-2861 (``self`` is ``context.config``)."""
    context = require(context, 'measurement', 'config', 'ventricular', 'beats', 'delineation', 'input', 'quality', 'atrial')
    settings = context.config
    events: List[PolicyEvent] = []
    hooks = context.hooks
    paced_beat_ids = context.ventricular.paced_beat_ids
    measurement_group_paced = context.beats.measurement_group_paced
    beat_features = context.delineation.beat_features
    measurement_beat_ids = context.beats.measurement_beat_ids
    dominant_group_id = context.beats.dominant_group_id
    ecg_measure = context.ventricular.ecg_measure
    r_locs = context.ventricular.r_locs
    fs_run = context.input.fs_run
    representative_options = context.quality.representative_options
    rep_left_ms = context.quality.rep_left_ms
    pacing_capture_confirmed = context.ventricular.pacing_capture_confirmed
    refinement_kwargs = context.quality.refinement_kwargs
    ecg_rs = context.input.ecg_rs
    pacing_result = context.ventricular.pacing_result
    acquisition_qc = context.ventricular.acquisition_qc
    approved_delays = context.ventricular.approved_delays
    mains_freq_run = context.input.mains_freq_run
    quality = context.quality.quality
    available_leads = context.input.available_leads
    beat_groups = context.beats.beat_groups
    confirmed_pacing_context = context.beats.confirmed_pacing_context
    measurement_reselect_reason = context.beats.measurement_reselect_reason
    qrs_tail_settling_rescue = context.delineation.qrs_tail_settling_rescue
    paced_fraction = context.beats.paced_fraction
    af_afl_summary = context.atrial.af_afl_summary
    atrial_residual = context.atrial.atrial_residual
    p_wave_assessments = context.atrial.p_wave_assessments
    pristine_p_wave_states = context.atrial.pristine_p_wave_states
    beats = context.beats.beats
    atrial_events = context.atrial.atrial_events
    pacing_evidence_quality = context.ventricular.pacing_evidence_quality
    rr_ms = context.atrial.rr_ms
    lead_reversal = context.quality.lead_reversal
    paced_qrs_floor_beat_ids = context.beats.paced_qrs_floor_beat_ids
    segmentation_paced_beat_ids = context.beats.segmentation_paced_beat_ids
    pacing_spike_beat_ids = context.ventricular.pacing_spike_beat_ids
    pacing_spike_offsets_samples = context.ventricular.pacing_spike_offsets_samples
    paced_beat_id_set = {int(beat_id) for beat_id in paced_beat_ids}
    measurement_beat_features = (
        [
            bf
            for bf in beat_features
            if int(getattr(bf, "beat_id", -1)) not in paced_beat_id_set
        ]
        if paced_beat_id_set and not measurement_group_paced
        else beat_features
    )
    measurement_rep_groups = {dominant_group_id: measurement_beat_ids} if measurement_beat_ids else {}
    measurement_representative, measurement_representative_meta = build_representative_beats_with_meta(
        ecg_measure,
        r_locs,
        measurement_rep_groups,
        fs_run,
        paced_beat_ids=paced_beat_ids,
        **representative_options,
    ) if measurement_rep_groups else ({}, {})
    representative_beat_features = []
    rep_center = int(rep_left_ms * fs_run / 1000)
    if dominant_group_id in measurement_representative:
        # Propagate pacing flag so the representative beat uses the T024
        # minimum-QRS-width constraint (Part 2).  Spike times are omitted
        # because the representative waveform is already de-spiked; Part 1
        # (onset anchor) is unnecessary and spike coordinates would not map
        # into the representative-beat time axis anyway.
        rep_paced_ids = [0] if measurement_group_paced else []
        rep_paced_floor_ids = [0] if measurement_group_paced and pacing_capture_confirmed else []
        representative_beat_features = delineate_beats(
            measurement_representative[dominant_group_id],
            fs_run,
            np.asarray([rep_center], dtype=int),
            paced_beat_ids=rep_paced_ids,
            paced_qrs_floor_beat_ids=rep_paced_floor_ids,
            **refinement_kwargs,
        )
        if settings.enable_t_wave_refinement:
            refine_t_wave_boundaries(
                representative_beat_features,
                measurement_ecg=measurement_representative[dominant_group_id],
                r_locs=np.asarray([rep_center], dtype=int),
                fs=fs_run,
                **refinement_kwargs,
            )
        if settings.enable_hybrid_st_measurement:
            apply_hybrid_st_measurement(
                representative_beat_features,
                measurement_ecg=measurement_representative[dominant_group_id],
                r_locs=np.asarray([rep_center], dtype=int),
                fs=fs_run,
            )
    st_baseline_metadata: Dict[str, Any] = {"source": settings.st_amplitude_source}
    if settings.st_amplitude_source in {"calibrated_pr", "adaptive_pr_tp"}:
        # This explicitly selected amplitude path uses the calibrated
        # signal, with the same accepted timing/spike transformations.
        raw_st = ecg_rs
        if pacing_result["spike_times"]:
            raw_st = remove_pacing_spikes(raw_st, pacing_result["spike_times"], fs_run)
        if acquisition_qc.get("automatic_compensation_applied"):
            raw_st, _ = apply_channel_delay_compensation(raw_st, approved_delays)
        baseline_builder = adaptive_st_signal if settings.st_amplitude_source == "adaptive_pr_tp" else calibrated_st_signal
        st_baseline = baseline_builder(
            raw_st, fs_run, beat_features, mains_hz=mains_freq_run
        )
        apply_hybrid_st_measurement(
            beat_features, measurement_ecg=st_baseline.signal,
            r_locs=r_locs, fs=fs_run, quality=quality,
        )
        st_representatives, _ = build_representative_beats_with_meta(
            st_baseline.signal, r_locs, measurement_rep_groups, fs_run,
            paced_beat_ids=paced_beat_ids,
            **representative_options,
        ) if measurement_rep_groups else ({}, {})
        if dominant_group_id in st_representatives and representative_beat_features:
            apply_hybrid_st_measurement(
                representative_beat_features,
                measurement_ecg=st_representatives[dominant_group_id],
                r_locs=np.asarray([rep_center], dtype=int), fs=fs_run,
            )
        for feature in [*beat_features, *representative_beat_features]:
            if feature.lead in st_baseline.unavailable_leads:
                feature.st_hybrid_reliable = False
                feature.st_hybrid_unreliable_reason = "insufficient_calibrated_pr_anchors"
                feature.st_pattern_class = "unknown"
        if st_baseline.valid_mask is not None:
            for feature in beat_features:
                li = STANDARD_12_LEADS.index(feature.lead)
                j = feature.st_hybrid_j_index if feature.st_hybrid_j_index is not None else feature.qrs.offset
                # All amplitude samples through J+80 ms (including the
                # local averaging radius) require supported baseline.
                st_end = int(j) + int(round(.086 * fs_run)) + 1 if j is not None else 0
                if (j is None or not 0 <= int(j) < st_end <= ecg_rs.shape[1]
                        or not np.all(st_baseline.valid_mask[li, int(j):st_end])):
                    feature.st_hybrid_reliable = False
                    feature.st_hybrid_unreliable_reason = "baseline_anchor_support_insufficient"
                    feature.st_pattern_class = "unknown"
            # Representative time coordinates are not original-record
            # coordinates; report only if every selected donor supports ST.
            for feature in representative_beat_features:
                donors = [b for b in beat_features if b.lead == feature.lead and b.beat_id in measurement_beat_ids]
                if not donors or any(not b.st_hybrid_reliable for b in donors):
                    feature.st_hybrid_reliable = False
                    feature.st_hybrid_unreliable_reason = "representative_donor_baseline_uncertain"
                    feature.st_pattern_class = "unknown"
            st_baseline_metadata.update(anchor_sources=st_baseline.anchor_sources,
                valid_sample_fraction={lead: float(np.mean(st_baseline.valid_mask[i])) for i, lead in enumerate(STANDARD_12_LEADS)},
                uncertainty_semantics="heuristic_mv_envelope_not_calibrated_probability")
        st_baseline_metadata.update(
            anchor_counts=st_baseline.anchor_counts,
            unavailable_leads=list(st_baseline.unavailable_leads),
            validation_status="experimental_not_default",
        )
    adjacent_precordial_correlations: Dict[str, float] = {}
    if settings.enable_lead_reversal and available_leads is None and representative_beat_features and dominant_group_id in measurement_representative:
        lead_qrs_bounds = {
            bf.lead: (bf.qrs.onset, bf.qrs.offset) for bf in representative_beat_features
        }
        adjacent_precordial_correlations = compute_adjacent_precordial_correlations(
            measurement_representative[dominant_group_id],
            lead_qrs_bounds,
            {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)},
        )
    representative_leads = build_representative_lead_features(
        measurement_beat_features,
        quality,
        beat_groups=beat_groups,
        representative_beat_features=representative_beat_features,
        dominant_group_id=dominant_group_id,
        selected_beat_ids=set(measurement_beat_ids),
        adjacent_precordial_correlations=adjacent_precordial_correlations,
    )
    if not settings.enable_lead_reversal:
        record_decision(events, PRECORDIAL_REVERSAL_FLAGS.decide(representative_leads))
    precordial_reversal = {}
    if settings.enable_lead_reversal:
        for lead in [f"V{i}" for i in range(1, 7)]:
            lead_detail = representative_leads.get(lead)
            if lead_detail is None:
                continue
            detail = lead_detail.params.get("precordial_reversal_detail")
            if isinstance(detail, dict):
                precordial_reversal = detail
                break
    groups = compute_group_features(beat_features, beat_groups, len(r_locs), fs_run, r_locs=r_locs)
    global_measurement_paced = bool(
        measurement_group_paced
        or (
            confirmed_pacing_context
            and measurement_reselect_reason is None
            and record_decision(events, INTERMITTENT_PACING_WIDE_CONTEXT.decide(representative_leads))
        )
    )
    global_features = compute_global_features(
        representative_leads, measurement_beat_features, r_locs, fs_run,
        paced=global_measurement_paced,
    )
    qrs_tail_qt_rescue_source = record_decision(events, QT_TAIL_SETTLING_RESCUE.decide(
        global_features=global_features,
        representative_leads=representative_leads,
        r_locs=r_locs,
        fs=fs_run,
        qrs_tail_settling_rescue=qrs_tail_settling_rescue,
    ))
    qrs_tail_settling_rescue["qt_rescue_source"] = qrs_tail_qt_rescue_source
    pacing_qt_rescue_source = (
        record_decision(events, INTERMITTENT_PACED_QT_RESCUE.decide(
            global_features,
            representative_leads,
            measurement_beat_features,
            r_locs,
            fs_run,
            paced_fraction,
        ))
        if global_measurement_paced
        else None
    )
    pr_core_ms = _physiologic_pr_core_from_beats(beat_features)
    rr_cv_for_pr_core = _finite_float(af_afl_summary.get("rr_cv"))
    qrs_for_pr_core = _finite_float(global_features.qrs_ms)
    regular_narrow_pr_core_context = bool(
        pr_core_ms is not None
        and _REGULAR_NARROW_PR_CORE_MIN_MS
        <= pr_core_ms
        <= _REGULAR_NARROW_PR_CORE_MAX_MS
        and qrs_for_pr_core is not None
        and qrs_for_pr_core <= _REGULAR_NARROW_PR_CORE_QRS_MAX_MS
        and rr_cv_for_pr_core is not None
        and rr_cv_for_pr_core <= _REGULAR_NARROW_PR_CORE_RR_CV_MAX
        and not confirmed_pacing_context
    )
    # A regular narrow-complex rhythm carrying a normal-range PR is
    # ordinarily better explained by sinus/SVT than by flutter, which is
    # what this suppression is for. But fixed-ratio flutter *is* a regular
    # narrow-complex rhythm, and the interval measured from one of its F
    # waves to the QRS lands in the same range -- so on 2:1 flutter, the
    # single most typical presentation, the guard fires on precisely the
    # records it should leave alone. Measured on PTB-XL 10752/5252/1173
    # (all labelled AFLT): raw spectral narrowness 0.955-0.967 at a
    # conduction ratio of 1.98-2.00, i.e. unambiguous flutter evidence,
    # every one of them suppressed to `none`. Positive spectral evidence
    # of a single narrow F-wave peak therefore outranks the PR-shape
    # heuristic; without such evidence the suppression stands unchanged.
    flutter_spectral_evidence = bool(
        atrial_residual.get("flutter_raw_narrow_peak")
        or atrial_residual.get("flutter_narrow_peak_consensus")
    )
    if (
        regular_narrow_pr_core_context
        and bool(af_afl_summary.get("probable_flutter"))
        and not flutter_spectral_evidence
    ):
        af_afl_summary = dict(af_afl_summary)
        af_afl_summary["probable_flutter"] = False
        af_afl_summary["atrial_rhythm_classification"] = "none"
        af_afl_summary["diagnostic_confidence"] = 0.0
        af_afl_summary["flutter_wave_confidence"] = min(
            _finite_float(af_afl_summary.get("flutter_wave_confidence")) or 0.0,
            0.59,
        )
        af_afl_summary["F_wave_confidence"] = min(
            _finite_float(af_afl_summary.get("F_wave_confidence")) or 0.0,
            0.59,
        )
        af_afl_summary["flutter_suppressed_by"] = "regular_narrow_pr_core"
    # The P-wave contract was finalized against the *pre*-retraction verdict,
    # so on a suppressed record every beat is still carrying
    # ORGANIZED_ATRIAL_ACTIVITY / accepted=False / RHYTHM_ORGANIZED_ATRIAL_ACTIVITY
    # from a flutter call that no longer stands.  Every other consumer of
    # `af_afl_summary` reads it after this point and therefore sees the
    # correction; the P contract was the one place the dead verdict lived on,
    # which is why the output could report a numeric PR interval and "no beat
    # has a usable P wave" at the same time.  Measured on 300 PTB-XL records:
    # the retraction fired on 40, all 40 reported 0 of N beats usable, and
    # none of the 40 carried an AFIB or AFLT label -- 36 were labelled sinus.
    # Restoring the boundary verdict and re-finalizing is unconditional: when
    # nothing was retracted it reproduces the same states it just replaced.
    for assessment, (accepted, reject_reasons, p_state) in zip(
        p_wave_assessments, pristine_p_wave_states
    ):
        assessment.accepted = accepted
        assessment.reject_reasons = list(reject_reasons)
        assessment.p_state = p_state
    finalize_p_wave_states(p_wave_assessments, af_afl_summary)
    flutter_confidence = _finite_float(af_afl_summary.get("flutter_wave_confidence"))
    rhythm_allows_pr_rescue = not bool(af_afl_summary.get("probable_af")) and (
        flutter_confidence is None or flutter_confidence < 0.60
    )
    current_pr_ms = _finite_float(global_features.pr_ms)
    if (
        rhythm_allows_pr_rescue
        and pr_core_ms is not None
        and (
            current_pr_ms is None
            or current_pr_ms < 120.0
            or current_pr_ms > 240.0
        )
    ):
        global_features.pr_ms = pr_core_ms
    twelve_sl_profile = apply_twelve_sl_measurement_profile(
        representative_leads=representative_leads,
        beat_features=measurement_beat_features,
        r_locs=r_locs,
        fs=fs_run,
    )
    rhythm_beats = _build_rhythm_beat_rows(beats, beat_features)
    pacing_context = classify_pacing_context(
        rhythm_beats,
        atrial_events,
        pacing_result["spike_times"],
        evidence_quality=pacing_evidence_quality,
    )
    background_rr_ms = float(np.median(np.asarray(rr_ms, dtype=float))) if rr_ms else None
    post_pause_or_interpolated_beats = classify_post_pause_or_interpolated_beats(
        beats=rhythm_beats,
        background_rr_ms=background_rr_ms,
    )
    policy_measurement_beat_ids = select_measurement_beat_ids(rhythm_beats)
    pr_segment_ms = estimate_pr_segment_ms(beat_features, fs_run)
    delta_evidence = _delta_evidence(beat_features)
    pr_ms_for_rules = _finite_float(global_features.pr_ms)
    preexcitation = detect_preexcitation(
        short_pr_interval=bool(pr_ms_for_rules is not None and pr_ms_for_rules < 120.0),
        short_pr_segment=bool(pr_segment_ms is not None and pr_segment_ms < 50.0),
        delta_leads=delta_evidence["delta_leads"],
        mean_qrs_duration_ms=global_features.qrs_ms,
        initial_qrs_axis_deg=estimate_initial_qrs_axis_deg(representative_leads),
    )
    preexcitation["pr_segment_ms"] = pr_segment_ms
    preexcitation["delta_beat_ids"] = delta_evidence["delta_beat_ids"]
    preexcitation["delta_confidence_by_lead"] = delta_evidence["delta_confidence_by_lead"]

    pr_series_ms = _pr_series_ms(beat_features, len(r_locs))
    atrial_counts = _atrial_events_per_rr(atrial_events, r_locs)
    if not any(atrial_counts):
        atrial_counts = [1 if pr is not None else 0 for pr in pr_series_ms[:-1]]
    pauses = detect_pauses_and_av_block(
        rr_ms=rr_ms,
        pr_series_ms=pr_series_ms,
        atrial_events_per_rr=atrial_counts,
        qrs_duration_ms=global_features.qrs_ms,
    )
    rhythm_summary: Dict[str, Any] = {
        "primary_statement": None,
        "additional_statements": [],
        "statements": [],
        "bypass_remaining_algorithm": False,
    }
    if preexcitation.get("wpw_pattern"):
        rhythm_summary = {
            "primary_statement": "wpw_pattern",
            "additional_statements": [],
            "statements": [{"code": "wpw_pattern", "confidence": 1.0}],
            "bypass_remaining_algorithm": True,
        }
    atrial_measurements_invalid = record_decision(events, ATRIAL_MEASUREMENT_VALIDITY.decide(
        global_features,
        af_afl_summary,
    ))
    av_block_flags = detect_av_block_availability_flags(
        beats,
        beat_features,
        global_features.heart_rate_bpm,
        atrial_rate_bpm=getattr(global_features, "atrial_rate_bpm", None),
    )
    narrow_qrs_pr_core_rescue = bool(
        rhythm_allows_pr_rescue
        and pr_core_ms is not None
        and (_finite_float(global_features.qrs_ms) or 999.0) <= 125.0
    )
    if narrow_qrs_pr_core_rescue and av_block_flags.get("av_dissociation"):
        av_block_flags = dict(av_block_flags)
        av_block_flags["av_dissociation"] = False
        # complete_av_block may have been supported solely by the
        # PR-variability evidence just rescinded above; keep it only if
        # the atrial-rate criterion independently still supports it.
        if not av_block_flags.get("atrial_faster_than_ventricular"):
            av_block_flags["complete_av_block"] = False
    rule_summary = {
        "preexcitation": preexcitation,
        "pauses": pauses,
        "post_pause_or_interpolated_beats": post_pause_or_interpolated_beats,
        "atrial_measurements_invalid": atrial_measurements_invalid,
        **av_block_flags,
    }
    pacing_like_context = bool(
        not narrow_qrs_pr_core_rescue
        and not bool(pacing_evidence_quality.get("evidence_conflicted"))
        and (
            record_decision(events, PACING_LIKE_CONTEXT.decide(global_features.qrs_ms, groups, rule_summary))
        )
    )
    if narrow_qrs_pr_core_rescue and bool(pacing_context.get("intermittent_pacing")):
        pacing_context = dict(pacing_context)
        pacing_context["suppress_further_rhythm_interpretation"] = False
        pacing_context["wide_qrs_pacing_like_context"] = False
    if narrow_qrs_pr_core_rescue and global_features.t_axis_deg is None:
        stable_t_axis_deg = _stable_limb_signed_t_axis_deg(representative_leads)
        if stable_t_axis_deg is not None:
            global_features.t_axis_deg = stable_t_axis_deg
    if pacing_like_context:
        qrs_override_ms = record_decision(events, PACING_LIKE_QRS_OVERRIDE.decide(
            global_features.qrs_ms,
            representative_leads,
            groups,
            rule_summary,
        ))
        if qrs_override_ms is not None:
            global_features.qrs_ms = qrs_override_ms
        global_features.pr_ms = None
        if record_decision(events, LIMB_LEAD_REVERSAL.decide(lead_reversal)) and not pacing_result.get("paced"):
            global_features.qrs_axis_deg = None
        pacing_context = dict(pacing_context)
        pacing_context["wide_qrs_pacing_like_context"] = True
        pacing_context["suppress_further_rhythm_interpretation"] = True
        if (
            preexcitation.get("wpw_pattern")
            and not preexcitation.get("strong_multilead_fallback")
        ):
            preexcitation = dict(preexcitation)
            preexcitation["wpw_pattern"] = False
            preexcitation["accessory_pathway_side"] = None
            preexcitation["suppressed_by"] = "wide_qrs_pacing_like_context"
            rule_summary["preexcitation"] = preexcitation
            rhythm_summary = {
                "primary_statement": None,
                "additional_statements": [],
                "statements": [],
                "bypass_remaining_algorithm": False,
            }
    if global_measurement_paced:
        paced_qrs_override_ms: Optional[float] = record_decision(events, PACED_QRS_OVERRIDE.decide(
            global_features.qrs_ms,
            representative_leads,
            groups,
        ))
        if paced_qrs_override_ms is not None:
            global_features.qrs_ms = paced_qrs_override_ms
            pacing_context = dict(pacing_context)
            pacing_context["wide_qrs_pacing_like_context"] = True
            pacing_context["suppress_further_rhythm_interpretation"] = True
    availability = build_measurement_availability(
        pacing_context,
        af_afl_summary,
        rule_summary,
    )
    record_decision(events, PR_AVAILABILITY.decide(
        representative_leads,
        global_features,
        availability,
        af_afl_summary,
    ))
    t_axis_profile_backfill = False
    has_reliable_qt_lead = bool(
        _select_reliable_qt_leads(representative_leads)
    )
    if (
        not global_measurement_paced
        and not pacing_like_context
        and has_reliable_qt_lead
        and not bool(af_afl_summary.get("probable_af"))
    ):
        t_axis_profile_backfill = _backfill_t_axis_after_measurement_profile(
            global_features,
            representative_leads,
            r_locs,
            fs_run,
        )
    if getattr(global_features, "t_axis_deg", None) is not None:
        stable_t_axis_deg = _stable_limb_signed_t_axis_deg(representative_leads)
        t_axis_delta = _axis_delta_deg(getattr(global_features, "t_axis_deg", None), stable_t_axis_deg)
        if stable_t_axis_deg is not None and t_axis_delta is not None and t_axis_delta >= 60.0:
            global_features.t_axis_deg = stable_t_axis_deg
    if hooks is not None:
        rule_summary["statement_evidence"] = hooks.rhythm_statement_evidence(
            rhythm_summary=rhythm_summary,
            preexcitation=preexcitation,
            pacing_context=pacing_context,
            availability=availability,
            af_afl_summary=af_afl_summary,
            atrial_residual=atrial_residual,
        )

    # T024: Pacing spike detection
    if settings.enable_pacing:
        global_features.pacing_spikes = pacing_result["spike_times"]
        global_features.paced_rhythm  = confirmed_pacing_context

    reliable_qt_leads = _select_reliable_qt_leads(representative_leads)
    pacing_measurement_effect = record_decision(events, PACING_MEASUREMENT_EFFECT.decide(
        pacing_enabled=settings.enable_pacing,
        pacing_result=pacing_result,
        measurement_group_paced=measurement_group_paced,
        global_measurement_paced=global_measurement_paced,
        pacing_capture_confirmed=pacing_capture_confirmed,
        paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
    ))
    pacing_segmentation_effect = record_decision(events, PACING_SEGMENTATION_EFFECT.decide(
        pacing_enabled=settings.enable_pacing,
        pacing_result=pacing_result,
        measurement_group_paced=measurement_group_paced,
        segmentation_paced_beat_ids=segmentation_paced_beat_ids,
        paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
    ))
    pacing_evidence_by_beat = _build_pacing_evidence_by_beat(
        n_beats=len(r_locs),
        pacing_result=pacing_result,
        pacing_spike_beat_ids=pacing_spike_beat_ids,
        pacing_spike_offsets_samples=pacing_spike_offsets_samples,
        pacing_capture_confirmed=pacing_capture_confirmed,
        paced_beat_ids=paced_beat_ids,
        segmentation_paced_beat_ids=segmentation_paced_beat_ids,
        paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
        measurement_beat_ids=measurement_beat_ids,
        measurement_group_paced=measurement_group_paced,
        fs=fs_run,
    )
    pacing_detection_state = pacing_result.get("state", "off")
    measurement_pacing_state = record_decision(events, MEASUREMENT_PACING_STATE.decide(
        pacing_enabled=settings.enable_pacing,
        detection_state=pacing_detection_state,
        confirmed_pacing_context=confirmed_pacing_context,
        pacing_measurement_effect=pacing_measurement_effect,
    ))
    bundle = MeasurementBundle(
        measurement_representative_meta=measurement_representative_meta,
        st_baseline_metadata=st_baseline_metadata,
        representative_leads=representative_leads,
        precordial_reversal=precordial_reversal,
        groups=groups,
        global_measurement_paced=global_measurement_paced,
        global_features=global_features,
        pacing_qt_rescue_source=pacing_qt_rescue_source,
        af_afl_summary=af_afl_summary,
        twelve_sl_profile=twelve_sl_profile,
        pacing_context=pacing_context,
        policy_measurement_beat_ids=policy_measurement_beat_ids,
        rule_summary=rule_summary,
        availability=availability,
        t_axis_profile_backfill=t_axis_profile_backfill,
        reliable_qt_leads=reliable_qt_leads,
        pacing_measurement_effect=pacing_measurement_effect,
        pacing_segmentation_effect=pacing_segmentation_effect,
        pacing_evidence_by_beat=pacing_evidence_by_beat,
        pacing_detection_state=pacing_detection_state,
        measurement_pacing_state=measurement_pacing_state,
    )
    context = replace(context, measurements=bundle)
    context = with_policy_events(context, events)
    return StageResult(context)


__all__ = ['MeasurementBundle', 'run']
