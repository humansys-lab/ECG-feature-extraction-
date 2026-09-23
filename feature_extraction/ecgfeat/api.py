from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from ._engine.atrial.p_wave import finalize_p_wave_states, summarize_p_wave_assessments
from ._engine.beats.representative import build_representative_beats_with_meta
from ._engine.delineation.core import delineate_beats
from ._engine.delineation.t_refinement import refine_t_wave_boundaries
from ._engine.foundation.numeric import _finite_float
from ._engine.measurement.features import (
    _axis_delta_deg,
    _physiologic_pr_core_from_beats,
    _select_reliable_qt_leads,
    _stable_limb_signed_t_axis_deg,
    build_representative_lead_features,
    compute_global_features,
    compute_group_features,
    estimate_initial_qrs_axis_deg,
    estimate_pr_segment_ms,
)
from ._engine.measurement.profiles.twelve_sl import apply_twelve_sl_measurement_profile
from ._engine.measurement.st_baseline import adaptive_st_signal, calibrated_st_signal
from ._engine.measurement.st_localization import apply_hybrid_st_measurement
from ._engine.quality.acquisition import apply_channel_delay_compensation
from ._engine.quality.signal import (
    build_diagnostic_gate,
    compute_adjacent_precordial_correlations,
    compute_qrs_detector_agreement,
    remove_pacing_spikes,
)
from .clinical_rules.engine import analyze_clinical
from .interpret import interpret
from .models import (
    ECGFeatures,
    PatientMeta,
    STANDARD_12_LEADS,
    resolve_patient_age,
)
from .pipeline.policies.applicability import (
    _apply_measurement_availability_to_representatives,
    _atrial_measurements_invalid_for_availability,
)
from .pipeline.policies.lead_integrity import (
    _clear_precordial_reversal_flags,
    _probable_limb_lead_reversal,
)
from .pipeline.policies.pacing import (
    _borderline_paced_qrs_wide_offset_override_ms,
    _build_pacing_evidence_by_beat,
    _dominant_paced_wide_qrs_override_ms,
    _intermittent_paced_wide_qrs_override_ms,
    _intermittent_pacing_wide_measurement_context,
    _measurement_pacing_state,
    _near_wide_paced_qrs_offset_override_ms,
    _near_wide_pacing_qrs_override_ms,
    _overwide_paced_qrs_raw_consensus_override_ms,
    _overwide_pacing_qrs_override_ms,
    _overwide_qrs_pacing_like_context,
    _pacing_measurement_effect,
    _pacing_segmentation_effect,
    _raw_pacing_qrs_underestimate_override_ms,
    _secondary_paced_wide_group_qrs_override_ms,
    _wide_qrs_pacing_like_context,
)
from .pipeline.policies.qt import (
    _apply_qt_reject_gate,
    _rescue_intermittent_paced_qt_from_native,
    _rescue_qt_after_qrs_tail_settling,
)
from .pipeline.stages.measurement import (
    _REGULAR_NARROW_PR_CORE_MAX_MS,
    _REGULAR_NARROW_PR_CORE_MIN_MS,
    _REGULAR_NARROW_PR_CORE_QRS_MAX_MS,
    _REGULAR_NARROW_PR_CORE_RR_CV_MAX,
    _atrial_events_per_rr,
    _backfill_t_axis_after_measurement_profile,
    _build_rhythm_beat_rows,
    _delta_evidence,
    _pr_series_ms,
)
from .refinement import RefinementConfig
from .rhythm_rules import (
    build_measurement_availability,
    classify_pacing_context,
    classify_post_pause_or_interpolated_beats,
    detect_av_block_availability_flags,
    detect_pauses_and_av_block,
    detect_preexcitation,
    select_measurement_beat_ids,
)
from .rhythm_statements import build_rhythm_statement_candidates


class ECGFeatureExtractor:
    """
    DXL-inspired 12-lead ECG feature extractor.

    This is a research/engineering scaffold, not a validated medical device.
    The pipeline mirrors the public DXL architecture at a high level:
    signal standardization -> quality -> multilead QRS -> beat grouping ->
    representative beat logic -> comprehensive measurements -> global features.
    """

    def __init__(
        self,
        fs_internal: int | None = None,
        mains_freq: int | str | None = 50,
        lp_hz: float = 40.0,
        enable_pacing: bool = True,
        enable_lead_reversal: bool = True,
        compute_grouping: bool = True,
        enable_hybrid_r_localization: bool = True,
        enable_hybrid_wave_localization: bool = True,
        enable_hybrid_st_measurement: bool = True,
        enable_t_wave_refinement: bool = True,
        st_amplitude_source: str = "analysis",
        refinement: Optional[RefinementConfig] = None,
        input_mode: str = "standard",
    ) -> None:
        if input_mode not in {"standard", "limited"}:
            raise ValueError("input_mode must be 'standard' or 'limited'")
        self.input_mode = input_mode
        if st_amplitude_source not in {"analysis", "calibrated_pr", "adaptive_pr_tp"}:
            raise ValueError("st_amplitude_source must be 'analysis', 'calibrated_pr' or 'adaptive_pr_tp'")
        if refinement is not None and not isinstance(refinement, RefinementConfig):
            raise TypeError("refinement must be a RefinementConfig")
        self.refinement = refinement or RefinementConfig()
        if st_amplitude_source != "analysis" and not enable_hybrid_st_measurement:
            raise ValueError("calibrated_pr requires enable_hybrid_st_measurement=True")
        self.st_amplitude_source = st_amplitude_source
        self.fs_internal = fs_internal  # None = use native input fs
        self.mains_freq = mains_freq
        self.lp_hz = lp_hz
        self.enable_pacing = enable_pacing
        self.enable_lead_reversal = enable_lead_reversal
        self.compute_grouping = compute_grouping
        self.enable_hybrid_r_localization = enable_hybrid_r_localization
        self.enable_hybrid_wave_localization = enable_hybrid_wave_localization
        self.enable_hybrid_st_measurement = enable_hybrid_st_measurement
        self.enable_t_wave_refinement = enable_t_wave_refinement

    def extract(
        self,
        ecg_12lead: np.ndarray,
        fs: float,
        meta: Optional[PatientMeta] = None,
        *,
        lead_names: Optional[List[str]] = None,
        amplitude_unit: Optional[str] = None,
        gain_uv_per_lsb: Optional[float] = None,
        prior_features: Optional[ECGFeatures] = None,
    ) -> ECGFeatures:
        from .pipeline.extractor import run_legacy_pipeline

        context = run_legacy_pipeline(
            self,
            ecg_12lead,
            fs,
            meta,
            lead_names=lead_names,
            amplitude_unit=amplitude_unit,
            gain_uv_per_lsb=gain_uv_per_lsb,
            prior_features=prior_features,
            stop_after='atrial',
        ).context
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
        ecg_detect = context.ventricular.ecg_detect
        record_quality = context.quality.record_quality
        input_contract = context.input.input_contract
        raw_record_quality = context.quality.raw_record_quality
        qrs_result = context.ventricular.qrs_result
        pacing_qrs_rescue = context.ventricular.pacing_qrs_rescue
        representative_meta = context.beats.representative_meta
        initial_measurement_group_id = context.beats.initial_measurement_group_id
        initial_measurement_beat_ids = context.beats.initial_measurement_beat_ids
        pacing_failures = context.ventricular.pacing_failures
        p_corrections = context.atrial.p_corrections
        atrial_validation_audit = context.atrial.atrial_validation_audit
        t_fusion_audit = context.delineation.t_fusion_audit
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
            if self.enable_t_wave_refinement:
                refine_t_wave_boundaries(
                    representative_beat_features,
                    measurement_ecg=measurement_representative[dominant_group_id],
                    r_locs=np.asarray([rep_center], dtype=int),
                    fs=fs_run,
                    **refinement_kwargs,
                )
            if self.enable_hybrid_st_measurement:
                apply_hybrid_st_measurement(
                    representative_beat_features,
                    measurement_ecg=measurement_representative[dominant_group_id],
                    r_locs=np.asarray([rep_center], dtype=int),
                    fs=fs_run,
                )
        st_baseline_metadata: Dict[str, Any] = {"source": self.st_amplitude_source}
        if self.st_amplitude_source in {"calibrated_pr", "adaptive_pr_tp"}:
            # This explicitly selected amplitude path uses the calibrated
            # signal, with the same accepted timing/spike transformations.
            raw_st = ecg_rs
            if pacing_result["spike_times"]:
                raw_st = remove_pacing_spikes(raw_st, pacing_result["spike_times"], fs_run)
            if acquisition_qc.get("automatic_compensation_applied"):
                raw_st, _ = apply_channel_delay_compensation(raw_st, approved_delays)
            baseline_builder = adaptive_st_signal if self.st_amplitude_source == "adaptive_pr_tp" else calibrated_st_signal
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
        if self.enable_lead_reversal and available_leads is None and representative_beat_features and dominant_group_id in measurement_representative:
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
        if not self.enable_lead_reversal:
            _clear_precordial_reversal_flags(representative_leads)
        precordial_reversal = {}
        if self.enable_lead_reversal:
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
                and _intermittent_pacing_wide_measurement_context(representative_leads)
            )
        )
        global_features = compute_global_features(
            representative_leads, measurement_beat_features, r_locs, fs_run,
            paced=global_measurement_paced,
        )
        qrs_tail_qt_rescue_source = _rescue_qt_after_qrs_tail_settling(
            global_features=global_features,
            representative_leads=representative_leads,
            r_locs=r_locs,
            fs=fs_run,
            qrs_tail_settling_rescue=qrs_tail_settling_rescue,
        )
        qrs_tail_settling_rescue["qt_rescue_source"] = qrs_tail_qt_rescue_source
        pacing_qt_rescue_source = (
            _rescue_intermittent_paced_qt_from_native(
                global_features,
                representative_leads,
                measurement_beat_features,
                r_locs,
                fs_run,
                paced_fraction,
            )
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
        atrial_measurements_invalid = _atrial_measurements_invalid_for_availability(
            global_features,
            af_afl_summary,
        )
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
                _wide_qrs_pacing_like_context(global_features.qrs_ms, rule_summary)
                or _overwide_qrs_pacing_like_context(global_features.qrs_ms, groups, rule_summary)
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
            qrs_override_ms = _raw_pacing_qrs_underestimate_override_ms(
                global_features.qrs_ms,
                representative_leads,
                rule_summary,
            )
            if qrs_override_ms is None:
                qrs_override_ms = _near_wide_pacing_qrs_override_ms(
                    global_features.qrs_ms,
                    groups,
                    rule_summary,
                )
            if qrs_override_ms is None:
                qrs_override_ms = _overwide_pacing_qrs_override_ms(
                    global_features.qrs_ms,
                    representative_leads,
                    groups,
                    rule_summary,
                )
            if qrs_override_ms is not None:
                global_features.qrs_ms = qrs_override_ms
            global_features.pr_ms = None
            if _probable_limb_lead_reversal(lead_reversal) and not pacing_result.get("paced"):
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
            raw_consensus_qrs_override_ms = _overwide_paced_qrs_raw_consensus_override_ms(
                global_features.qrs_ms,
                representative_leads,
                groups,
            )
            paced_qrs_override_ms: Optional[float] = None
            if raw_consensus_qrs_override_ms is not None:
                paced_qrs_override_ms = raw_consensus_qrs_override_ms
            if paced_qrs_override_ms is None:
                dominant_wide_qrs_override_ms = _dominant_paced_wide_qrs_override_ms(
                    global_features.qrs_ms,
                    groups,
                )
                if dominant_wide_qrs_override_ms is not None:
                    paced_qrs_override_ms = dominant_wide_qrs_override_ms
            if paced_qrs_override_ms is None:
                near_wide_qrs_override_ms = _near_wide_paced_qrs_offset_override_ms(
                    global_features.qrs_ms,
                    representative_leads,
                    groups,
                )
                if near_wide_qrs_override_ms is not None:
                    paced_qrs_override_ms = near_wide_qrs_override_ms
            if paced_qrs_override_ms is None:
                intermittent_wide_qrs_override_ms = _intermittent_paced_wide_qrs_override_ms(
                    global_features.qrs_ms,
                    representative_leads,
                    groups,
                )
                if intermittent_wide_qrs_override_ms is not None:
                    paced_qrs_override_ms = intermittent_wide_qrs_override_ms
            if paced_qrs_override_ms is None:
                borderline_qrs_override_ms = _borderline_paced_qrs_wide_offset_override_ms(
                    global_features.qrs_ms,
                    representative_leads,
                )
                if borderline_qrs_override_ms is not None:
                    paced_qrs_override_ms = borderline_qrs_override_ms
            if paced_qrs_override_ms is None:
                secondary_wide_qrs_override_ms = _secondary_paced_wide_group_qrs_override_ms(
                    global_features.qrs_ms,
                    representative_leads,
                    groups,
                )
                if secondary_wide_qrs_override_ms is not None:
                    paced_qrs_override_ms = secondary_wide_qrs_override_ms
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
        _apply_measurement_availability_to_representatives(
            representative_leads,
            global_features,
            availability,
            af_afl_summary,
        )
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
        rule_summary["statement_evidence"] = build_rhythm_statement_candidates(
            rhythm_summary=rhythm_summary,
            preexcitation=preexcitation,
            pacing_context=pacing_context,
            availability=availability,
            af_afl_summary=af_afl_summary,
            atrial_residual=atrial_residual,
        )

        # T024: Pacing spike detection
        if self.enable_pacing:
            global_features.pacing_spikes = pacing_result["spike_times"]
            global_features.paced_rhythm  = confirmed_pacing_context

        reliable_qt_leads = _select_reliable_qt_leads(representative_leads)
        pacing_measurement_effect = _pacing_measurement_effect(
            pacing_enabled=self.enable_pacing,
            pacing_result=pacing_result,
            measurement_group_paced=measurement_group_paced,
            global_measurement_paced=global_measurement_paced,
            pacing_capture_confirmed=pacing_capture_confirmed,
            paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
        )
        pacing_segmentation_effect = _pacing_segmentation_effect(
            pacing_enabled=self.enable_pacing,
            pacing_result=pacing_result,
            measurement_group_paced=measurement_group_paced,
            segmentation_paced_beat_ids=segmentation_paced_beat_ids,
            paced_qrs_floor_beat_ids=paced_qrs_floor_beat_ids,
        )
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
        measurement_pacing_state = _measurement_pacing_state(
            pacing_enabled=self.enable_pacing,
            detection_state=pacing_detection_state,
            confirmed_pacing_context=confirmed_pacing_context,
            pacing_measurement_effect=pacing_measurement_effect,
        )
        duration_sec = float(ecg_rs.shape[-1]) / float(fs_run)
        qrs_detector_agreement = compute_qrs_detector_agreement(
            ecg_detect,
            fs_run,
            r_locs,
            quality,
        )
        diagnostic_gate = build_diagnostic_gate(
            record_quality=record_quality,
            duration_sec=duration_sec,
            n_beats=int(len(r_locs)),
            beat_groups=beat_groups,
            limb_reversal=lead_reversal,
            precordial_reversal=precordial_reversal,
            limb_lead_consistency=input_contract.get("limb_lead_consistency"),
            qrs_detector_agreement=qrs_detector_agreement,
            amplitude_calibration=input_contract.get("amplitude_calibration"),
        )
        patient_age = resolve_patient_age(meta).age_years
        patient_sex = (
            str(getattr(meta, "sex", "") or "").strip().lower()
            if meta is not None
            else ""
        )
        metadata_missing = []
        if patient_age is None:
            metadata_missing.append("patient.age")
        if patient_sex not in {"male", "m", "female", "f"}:
            metadata_missing.append("patient.sex")
        metadata_warnings = []
        if meta is None or getattr(meta, "acquisition_time", None) is None:
            metadata_warnings.append("patient.acquisition_time")
        if meta is None or getattr(meta, "device", None) is None:
            metadata_warnings.append("patient.device")
        if meta is None or getattr(meta, "filter_highpass_hz", None) is None:
            metadata_warnings.append("patient.filter_highpass_hz")
        if meta is None or getattr(meta, "filter_lowpass_hz", None) is None:
            metadata_warnings.append("patient.filter_lowpass_hz")
        if metadata_missing and diagnostic_gate["state"] != "stop":
            diagnostic_gate["state"] = "partial"
            diagnostic_gate["partial_reasons"] = sorted(set(
                list(diagnostic_gate["partial_reasons"])
                + ["required_patient_metadata_missing"]
            ))
        metadata_contract = {
            "complete_for_age_sex_rules": not metadata_missing,
            "missing_required": metadata_missing,
            "missing_audit_metadata": metadata_warnings,
            "amplitude_unit": input_contract.get("amplitude_output_unit"),
            "raw_sampling_rate_500_hz_required": False,
        }
        metadata = {
            "input_fs": input_contract["input_fs"],
            "internal_fs": fs_run,
            "mains_frequency_hz": mains_freq_run,
            "duration_sec": duration_sec,
            "input_contract": input_contract,
            "lead_order": STANDARD_12_LEADS,
            "lead_reversal": {
                "limb": lead_reversal,
                "precordial": precordial_reversal,
            },
            "record_quality": record_quality,
            "diagnostic_gate": diagnostic_gate,
            "metadata_contract": metadata_contract,
            "quality_stages": {
                "raw": raw_record_quality,
                "processed": record_quality,
            },
            "acquisition_qc": acquisition_qc,
            "p_wave_contract": summarize_p_wave_assessments(
                p_wave_assessments
            ),
            "n_beats": int(len(r_locs)),
            "r_localization": {
                "enabled": bool(self.enable_hybrid_r_localization),
                "method": (
                    "hybrid_prominence"
                    if self.enable_hybrid_r_localization
                    else None
                ),
                "affects_measurements": False,
                "affects_interpretation": False,
            },
            "wave_localization": {
                "enabled": bool(self.enable_hybrid_wave_localization),
                "methods": {
                    "p": (
                        "native_peak_bipolar_morphology"
                        if self.enable_hybrid_wave_localization
                        else None
                    ),
                    "t": (
                        "hybrid_bipolar_prominence"
                        if self.enable_hybrid_wave_localization
                        else None
                    ),
                    "s": (
                        "hybrid_s_morphology"
                        if self.enable_hybrid_wave_localization
                        else None
                    ),
                },
                "affects_measurements": False,
                "affects_interpretation": False,
            },
            "st_measurement": {
                "enabled": bool(self.enable_hybrid_st_measurement),
                "method": (
                    "robust_pr_baseline_local_settling_multilead_consensus"
                    if self.enable_hybrid_st_measurement
                    else None
                ),
                "sampling": (
                    "window_median_j_j20_j40_j60_j80_adaptive_rr_ton_guard"
                ),
                "morphology": "robust_linear_slope_plus_quadratic_curvature",
                "affects_native_measurements": False,
                "affects_interpretation": False,
            },
            "t_wave_refinement": {
                "enabled": bool(self.enable_t_wave_refinement),
                "filter_hz": 22.5 if self.enable_t_wave_refinement else None,
                "candidate_methods": (
                    ["dxl_chord", "mallat_quadratic_spline", "trapezium_area"]
                    if self.enable_t_wave_refinement
                    else []
                ),
                "fusion": (
                    "weighted_median_mad_huber_center_and_p85"
                    if self.enable_t_wave_refinement
                    else None
                ),
                "derived_leads": (
                    ["RMS", "PC1"] if self.enable_t_wave_refinement else []
                ),
                "affects_local_t_onset": bool(self.enable_t_wave_refinement),
                "affects_local_t_offset": "strict_late_tail_rescue_only"
                if self.enable_t_wave_refinement
                else False,
                "affects_qt_consensus": "only_when_fusion_reliable"
                if self.enable_t_wave_refinement
                else False,
            },
            "twelve_sl_measurement_profile": twelve_sl_profile,
            "t_axis_profile_backfill": t_axis_profile_backfill,
            "reliable_qt_leads": reliable_qt_leads,
            "n_reliable_qt_leads": len(reliable_qt_leads),
            "global_p_duration": {
                "source": getattr(global_features, "p_duration_source", None),
                "used_leads": list(
                    getattr(global_features, "p_duration_used_leads", []) or []
                ),
                "support": int(
                    getattr(global_features, "p_duration_support", 0) or 0
                ),
                "spread_ms": getattr(global_features, "p_duration_spread_ms", None),
                "reliability": getattr(
                    global_features,
                    "p_duration_reliability",
                    "unavailable",
                ),
            },
            "global_qt": {
                "source": getattr(global_features, "qt_source", None),
                "used_leads": list(getattr(global_features, "qt_used_leads", []) or []),
                "reliability": getattr(global_features, "qt_reliability", "unavailable"),
                "path": getattr(global_features, "qt_path", None),
                "confidence_reason": getattr(global_features, "qt_confidence_reason", None),
                "excluded_leads": dict(getattr(global_features, "qt_excluded_leads", {}) or {}),
                "lead_weights": dict(getattr(global_features, "qt_lead_weights", {}) or {}),
                "consensus_vs_independent_per_lead": {
                    lead: dict(values)
                    for lead, values in (
                        getattr(global_features, "consensus_vs_independent_per_lead", {}) or {}
                    ).items()
                },
            },
            "st_amplitude_signal": st_baseline_metadata,
            "confidence_semantics": {
                "qrs_detector_confidence": "uncalibrated_energy_margin_score",
                "qt_confidence": "uncalibrated_local_evidence_score_not_accuracy_probability",
                "qt_reporting_gate": "global_features.qt_reportable",
            },
            "qrs_detector": qrs_result,
            "qrs_detector_agreement": qrs_detector_agreement,
            "pacing_qrs_rescue": pacing_qrs_rescue,
            "qrs_tail_settling_rescue": qrs_tail_settling_rescue,
            "representative_group_id": dominant_group_id,
            "representative_beat_meta": representative_meta,
            "measurement_representative_beat_meta": measurement_representative_meta,
            "measurement_beat_ids": measurement_beat_ids,
            "measurement_group_reselected": bool(measurement_reselect_reason),
            "measurement_group_reselect_reason": measurement_reselect_reason,
            "initial_measurement_group_id": initial_measurement_group_id,
            "initial_measurement_beat_ids": initial_measurement_beat_ids,
            "pacing_spike_beat_ids": pacing_spike_beat_ids,
            "pacing_spike_offsets_ms": [
                float(offset) * 1000.0 / float(fs_run)
                for offset in pacing_spike_offsets_samples
            ],
            "pacing_capture_confirmed": pacing_capture_confirmed,
            "pacing_evidence_quality": pacing_evidence_quality,
            "paced_beat_ids": paced_beat_ids,
            "pacing_detection_state": pacing_detection_state,
            "measurement_pacing_state": measurement_pacing_state,
            "pacing_state": measurement_pacing_state,
            "pacing_rescue_applied": bool(pacing_result.get("rescue_applied", False)),
            "pacing_rescue_prominence_uv": pacing_result.get("rescue_prominence_uv"),
            "pacing_rescue_reason": pacing_result.get("rescue_reason"),
            "pacing_rescue_capture_fraction": pacing_result.get("rescue_capture_fraction"),
            "pacing_qt_rescue_source": pacing_qt_rescue_source,
            "paced_beat_fraction": paced_fraction,
            "measurement_group_paced": measurement_group_paced,
            "global_measurement_paced": global_measurement_paced,
            "confirmed_pacing_context": confirmed_pacing_context,
            "pacing_segmentation_effect": pacing_segmentation_effect,
            "pacing_measurement_effect": pacing_measurement_effect,
            "pacing_evidence_by_beat": pacing_evidence_by_beat,
            "rhythm_analysis": {
                "atrial_events": atrial_events,
                "atrial_residual": atrial_residual,
                "af_afl_summary": af_afl_summary,
                "pacing_context": pacing_context,
                "pacing_failures": pacing_failures,
                "policy_measurement_beat_ids": policy_measurement_beat_ids,
                "rule_summary": rule_summary,
                "availability": availability,
            },
        }
        if meta is not None:
            metadata["patient_meta"] = meta
        if self.refinement.enabled:
            metadata["refinement"] = {"config": asdict(self.refinement),
                "status": "experimental_requires_cohort_validation", "p_boundary_corrections": p_corrections,
                "atrial_event_validation": atrial_validation_audit,
                "t_interval_status": "heuristic_uncalibrated", "native_t_fusion": t_fusion_audit}

        _apply_qt_reject_gate(global_features, record_quality.get("record_grade"))

        ecg_features = ECGFeatures(
            fs=fs_run,
            quality=quality,
            beats=beats,
            beat_features=beat_features,
            representative_leads=representative_leads,
            groups=groups,
            global_features=global_features,
            p_wave_assessments=p_wave_assessments,
            metadata=metadata,
        )
        if available_leads is not None:
            # A limited-lead extraction is a measurement contract, not a
            # synthetic 12-lead diagnostic report. Keep physical channel names
            # in the input contract and remove empty computational slots.
            ecg_features.quality = {k: v for k, v in quality.items() if k in available_leads}
            ecg_features.beat_features = [b for b in beat_features if b.lead in available_leads]
            ecg_features.representative_leads = {k: v for k, v in representative_leads.items() if k in available_leads}
            for name in ("p_axis_deg", "qrs_axis_deg", "t_axis_deg", "st_axis_deg", "qrs_t_angle_deg", "transition_zone"):
                setattr(global_features, name, None)
            global_features.t_axis_reliable = False
            global_features.qt_reportable = False
            global_features.qt_unreliable_reasons = list(dict.fromkeys(
                [*global_features.qt_unreliable_reasons, "limited_lead_measurement_only"]))
            if not input_contract["anatomical_names_known"] or "V1" not in available_leads:
                global_features.ptf_v1_mv_ms = None
            if not input_contract["anatomical_names_known"]:
                for feature in ecg_features.beat_features:
                    feature.ptf_v1_mv_ms = None
                for representative in ecg_features.representative_leads.values():
                    representative.params["ptf_v1_mv_ms"] = None
                global_features.t_fusion_reliable = False
                global_features.t_fusion_lead_groups = ["supplied_channels"]
            metadata["lead_order"] = list(available_leads)
            metadata["diagnostic_gate"] = {
                **metadata["diagnostic_gate"], "state": "stop", "allowed_domains": [],
                "stop_reasons": ["limited_lead_measurement_only"], "suppressed_domains": ["all_diagnosis"],
            }
            metadata["limited_lead_capabilities"] = {
                "mode": "measurements_only", "available_leads": available_leads,
                "channel_to_slot": input_contract["lead_slot_map"],
                "available": ["beat_detection", "per_channel_delineation", "atrial_events", "intervals"],
                "unavailable": ["frontal_axes", "twelve_lead_diagnosis", "lead_reversal_diagnosis"],
                "p_acceptance": "quality_and_boundary_agreement_on_supplied_channels",
                "qt_reportability": "raw_intervals_only_no_twelve_lead_reportability_claim",
            }
            metadata["clinical_interpretation"] = {"status": "not_applicable_limited_lead_measurements"}
            return ecg_features
        ecg_features.interpretation = interpret(ecg_features)
        ecg_features.metadata["clinical_interpretation"] = analyze_clinical(
            ecg_features,
            prior_features=prior_features,
        ).to_dict()
        return ecg_features
