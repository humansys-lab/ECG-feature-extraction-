"""Finalize stage: reportability gates, legacy metadata and ``ECGFeatures`` assembly.

Ends at the completed legacy ``ECGFeatures`` (``context.legacy_features``).  Record
emission stays in ``ecgfeat.pipeline.extractor.ecg_emit``; interpretation runs only
through injected hooks.

Stage functions execute contiguous segments of the pre-decomposition
``ECGFeatureExtractor.extract`` (api.py at e7c26f7) in their original order; the
generic rewrites are ``self`` -> ``context.config`` and legacy helper calls ->
``record_decision(events, POLICY.decide(...))``, which returns the helper's value.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, List

from ..._engine.atrial.p_wave import summarize_p_wave_assessments
from ..._engine.quality.signal import build_diagnostic_gate, compute_qrs_detector_agreement
from ...models import ECGFeatures, STANDARD_12_LEADS, resolve_patient_age
from ..context import (
    PipelineContext,
    PolicyEvent,
    StageResult,
    record_decision,
    require,
    with_policy_events,
)
from ..policies.qt import QT_REJECT_GATE


def _finalized(context: PipelineContext, features: Any, events: List[PolicyEvent]) -> StageResult:
    return StageResult(with_policy_events(replace(context, legacy_features=features), events))


def run(context: PipelineContext) -> StageResult:
    """Legacy source: api.py@e7c26f7 lines 2862-3155 (``self`` is ``context.config``)."""
    context = require(context, 'finalize', 'config', 'input', 'ventricular', 'quality', 'beats', 'measurements', 'request', 'atrial', 'delineation')
    settings = context.config
    events: List[PolicyEvent] = []
    hooks = context.hooks
    ecg_rs = context.input.ecg_rs
    fs_run = context.input.fs_run
    ecg_detect = context.ventricular.ecg_detect
    r_locs = context.ventricular.r_locs
    quality = context.quality.quality
    record_quality = context.quality.record_quality
    beat_groups = context.beats.beat_groups
    lead_reversal = context.quality.lead_reversal
    precordial_reversal = context.measurements.precordial_reversal
    input_contract = context.input.input_contract
    meta = context.request.meta
    mains_freq_run = context.input.mains_freq_run
    raw_record_quality = context.quality.raw_record_quality
    acquisition_qc = context.ventricular.acquisition_qc
    p_wave_assessments = context.atrial.p_wave_assessments
    twelve_sl_profile = context.measurements.twelve_sl_profile
    t_axis_profile_backfill = context.measurements.t_axis_profile_backfill
    reliable_qt_leads = context.measurements.reliable_qt_leads
    global_features = context.measurements.global_features
    st_baseline_metadata = context.measurements.st_baseline_metadata
    qrs_result = context.ventricular.qrs_result
    pacing_qrs_rescue = context.ventricular.pacing_qrs_rescue
    qrs_tail_settling_rescue = context.delineation.qrs_tail_settling_rescue
    dominant_group_id = context.beats.dominant_group_id
    representative_meta = context.beats.representative_meta
    measurement_representative_meta = context.measurements.measurement_representative_meta
    measurement_beat_ids = context.beats.measurement_beat_ids
    measurement_reselect_reason = context.beats.measurement_reselect_reason
    initial_measurement_group_id = context.beats.initial_measurement_group_id
    initial_measurement_beat_ids = context.beats.initial_measurement_beat_ids
    pacing_spike_beat_ids = context.ventricular.pacing_spike_beat_ids
    pacing_spike_offsets_samples = context.ventricular.pacing_spike_offsets_samples
    pacing_capture_confirmed = context.ventricular.pacing_capture_confirmed
    pacing_evidence_quality = context.ventricular.pacing_evidence_quality
    paced_beat_ids = context.ventricular.paced_beat_ids
    pacing_detection_state = context.measurements.pacing_detection_state
    measurement_pacing_state = context.measurements.measurement_pacing_state
    pacing_result = context.ventricular.pacing_result
    pacing_qt_rescue_source = context.measurements.pacing_qt_rescue_source
    paced_fraction = context.beats.paced_fraction
    measurement_group_paced = context.beats.measurement_group_paced
    global_measurement_paced = context.measurements.global_measurement_paced
    confirmed_pacing_context = context.beats.confirmed_pacing_context
    pacing_segmentation_effect = context.measurements.pacing_segmentation_effect
    pacing_measurement_effect = context.measurements.pacing_measurement_effect
    pacing_evidence_by_beat = context.measurements.pacing_evidence_by_beat
    atrial_events = context.atrial.atrial_events
    atrial_residual = context.atrial.atrial_residual
    af_afl_summary = context.measurements.af_afl_summary
    pacing_context = context.measurements.pacing_context
    pacing_failures = context.ventricular.pacing_failures
    policy_measurement_beat_ids = context.measurements.policy_measurement_beat_ids
    rule_summary = context.measurements.rule_summary
    availability = context.measurements.availability
    p_corrections = context.atrial.p_corrections
    atrial_validation_audit = context.atrial.atrial_validation_audit
    t_fusion_audit = context.delineation.t_fusion_audit
    beats = context.beats.beats
    beat_features = context.delineation.beat_features
    representative_leads = context.measurements.representative_leads
    groups = context.measurements.groups
    available_leads = context.input.available_leads
    prior_features = context.request.prior_features
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
            "enabled": bool(settings.enable_hybrid_r_localization),
            "method": (
                "hybrid_prominence"
                if settings.enable_hybrid_r_localization
                else None
            ),
            "affects_measurements": False,
            "affects_interpretation": False,
        },
        "wave_localization": {
            "enabled": bool(settings.enable_hybrid_wave_localization),
            "methods": {
                "p": (
                    "native_peak_bipolar_morphology"
                    if settings.enable_hybrid_wave_localization
                    else None
                ),
                "t": (
                    "hybrid_bipolar_prominence"
                    if settings.enable_hybrid_wave_localization
                    else None
                ),
                "s": (
                    "hybrid_s_morphology"
                    if settings.enable_hybrid_wave_localization
                    else None
                ),
            },
            "affects_measurements": False,
            "affects_interpretation": False,
        },
        "st_measurement": {
            "enabled": bool(settings.enable_hybrid_st_measurement),
            "method": (
                "robust_pr_baseline_local_settling_multilead_consensus"
                if settings.enable_hybrid_st_measurement
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
            "enabled": bool(settings.enable_t_wave_refinement),
            "filter_hz": 22.5 if settings.enable_t_wave_refinement else None,
            "candidate_methods": (
                ["dxl_chord", "mallat_quadratic_spline", "trapezium_area"]
                if settings.enable_t_wave_refinement
                else []
            ),
            "fusion": (
                "weighted_median_mad_huber_center_and_p85"
                if settings.enable_t_wave_refinement
                else None
            ),
            "derived_leads": (
                ["RMS", "PC1"] if settings.enable_t_wave_refinement else []
            ),
            "affects_local_t_onset": bool(settings.enable_t_wave_refinement),
            "affects_local_t_offset": "strict_late_tail_rescue_only"
            if settings.enable_t_wave_refinement
            else False,
            "affects_qt_consensus": "only_when_fusion_reliable"
            if settings.enable_t_wave_refinement
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
    if settings.refinement.enabled:
        metadata["refinement"] = {"config": asdict(settings.refinement),
            "status": "experimental_requires_cohort_validation", "p_boundary_corrections": p_corrections,
            "atrial_event_validation": atrial_validation_audit,
            "t_interval_status": "heuristic_uncalibrated", "native_t_fusion": t_fusion_audit}

    record_decision(events, QT_REJECT_GATE.decide(global_features, record_quality.get("record_grade")))

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
        return _finalized(context, ecg_features, events)
    if hooks is not None:
        ecg_features.interpretation = hooks.interpret(ecg_features)
        ecg_features.metadata["clinical_interpretation"] = hooks.clinical_interpretation(
            ecg_features,
            prior_features=prior_features,
        )
    return _finalized(context, ecg_features, events)


__all__ = ['run']
