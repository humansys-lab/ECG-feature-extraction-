from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np

from .atrial import (
    _classify_af_afl,
    build_qrst_subtracted_residual,
    compute_organized_p_ratio,
    compute_pr_dispersion_ms,
    extract_atrial_events,
)
from .acquisition_qc import (
    apply_channel_delay_compensation,
    assess_acquisition_chain,
)
from .delineate import (
    apply_systematic_qrs_tail_settling_rescue,
    delineate_beats,
)
from .features import (
    _axis_from_amplitudes,
    _low_support_limb_t_axis_coverage_values,
    _st_supported_limb_t_axis_coverage_values,
    build_representative_lead_features,
    compute_global_features,
    compute_group_features,
    estimate_initial_qrs_axis_deg,
    estimate_pr_segment_ms,
    _axis_delta_deg,
    _physiologic_pr_core_from_beats,
    _select_reliable_qt_leads,
    _stable_limb_signed_t_axis_deg,
)
from .grouping import build_beat_annotations, cluster_beats
from .clinical_rules.engine import analyze_clinical
from .interpret import interpret
from .models import (
    ECGFeatures,
    PatientMeta,
    STANDARD_12_LEADS,
    resolve_patient_age,
)
from .preprocess import analysis_signal, lowpass_filter, resample_ecg
from .p_wave_engine import (
    PWaveConfig,
    backfill_missing_p_from_robust_engine,
    build_p_wave_assessments,
    finalize_p_wave_states,
    summarize_p_wave_assessments,
)
from .qrs import DEFAULT_QRS_LEADS, detect_qrs_multilead_with_meta
from .quality import (
    build_diagnostic_gate,
    compute_adjacent_precordial_correlations,
    compute_qrs_detector_agreement,
    compute_quality,
    detect_limb_lead_reversal,
    detect_pacing_spikes,
    prepare_pacing_detection_cache,
    remove_pacing_spikes,
    summarize_record_quality,
    validate_pacing_spikes_against_qrs,
)
from .r_localization import apply_hybrid_r_localization
from .wave_localization import apply_hybrid_wave_localization
from .st_localization import apply_hybrid_st_measurement
from .st_baseline import calibrated_st_signal, adaptive_st_signal
from .refinement import RefinementConfig
from .boundary_refinement import correct_p_boundaries
from .t_wave_refinement import refine_t_wave_boundaries
from .representative import build_representative_beats_with_meta
from .rhythm_rules import (
    assess_pacing_evidence_quality,
    build_measurement_availability,
    classify_pacing_context,
    classify_post_pause_or_interpolated_beats,
    detect_av_block_availability_flags,
    detect_pacing_failures,
    detect_pauses_and_av_block,
    detect_preexcitation,
    select_measurement_beat_ids,
)
from .rhythm_statements import build_rhythm_statement_candidates
from .twelve_sl import apply_twelve_sl_measurement_profile
from .validation import validate_ecg_input


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


def _clear_precordial_reversal_flags(representative_leads: Dict[str, object]) -> None:
    for lead in [f"V{i}" for i in range(1, 7)]:
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        rep.params.pop("precordial_reversal_detail", None)
        rep.params.pop("probable_precordial_reversal", None)


def _apply_measurement_availability_to_representatives(
    representative_leads: Dict[str, object],
    global_features: object,
    availability: Dict[str, Any],
    af_afl_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Remove native AV-conduction measurements when rhythm context invalidates them."""
    if not isinstance(availability, dict):
        return
    if bool(availability.get("pr_available", True)):
        return
    reasons = set(availability.get("reasons") or [])
    # Every reason here is a positive finding that AV conduction cannot be
    # measured: the atria are fibrillating, fluttering, paced, or dissociated,
    # so a PR interval would be an artifact of pairing unrelated waves.
    #
    # `af_afl_indeterminate` is deliberately NOT in this set. It is the
    # detector explicitly declining to decide, and deleting an independently
    # measured PR on the strength of a non-decision converts uncertainty into
    # destruction of evidence. Measured on the diverse50 cohort: it fired on
    # four sinus-labelled records (09786, 09741, 09249, 09466), erasing a
    # physiologic PR of 145-162 ms from the global feature AND from every
    # representative lead. Downstream that removed the interval the agent needs
    # for first-degree AV block and pre-excitation, and the loss propagated
    # into QT conclusions through shared interval-reportability gating.
    # Consumers still see `pr_available=False` and can weigh the value
    # accordingly; they can no longer be silently denied it.
    hard_pr_unavailable = {
        "probable_af",
        "probable_flutter",
        "continuous_pacing",
        "wide_qrs_pacing_like_context",
        "complete_av_block",
    }
    rr_cv_raw = (af_afl_summary or {}).get("rr_cv") if isinstance(af_afl_summary, dict) else None
    try:
        rr_cv = float(rr_cv_raw)
    except (TypeError, ValueError):
        rr_cv = None
    irregular_atrial_unavailable = (
        "atrial_measurements_unavailable" in reasons
        and rr_cv is not None
        and np.isfinite(rr_cv)
        and rr_cv >= 0.15
    )
    if not reasons.intersection(hard_pr_unavailable) and not irregular_atrial_unavailable:
        return
    if hasattr(global_features, "pr_ms"):
        global_features.pr_ms = None
    for rep in representative_leads.values():
        params = getattr(rep, "params", None)
        if not isinstance(params, dict):
            continue
        params["pr_ms"] = None
        params["pr_consensus_ms"] = None


def _atrial_measurements_invalid_for_availability(
    global_features: object,
    af_afl_summary: Dict[str, Any],
) -> bool:
    rr_cv_raw = af_afl_summary.get("rr_cv") if isinstance(af_afl_summary, dict) else None
    try:
        rr_cv = float(rr_cv_raw)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(rr_cv) or rr_cv < 0.15:
        return False
    return (
        getattr(global_features, "pr_ms", None) is None
        and getattr(global_features, "p_axis_deg", None) is None
    )


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
_PACING_INTERMITTENT_QT_NATIVE_RESCUE_MAX_PACED_FRACTION = 0.80
_PACING_INTERMITTENT_QT_NATIVE_RESCUE_MIN_DELTA_MS = 40.0
_REGULAR_NARROW_PR_CORE_RR_CV_MAX = 0.08
_REGULAR_NARROW_PR_CORE_QRS_MAX_MS = 125.0
_REGULAR_NARROW_PR_CORE_MIN_MS = 120.0
_REGULAR_NARROW_PR_CORE_MAX_MS = 240.0
_PACING_ATTENUATED_RESCUE_PROMINENCE_UV = (150.0, 100.0, 25.0)
_PACING_ATTENUATED_RESCUE_CLASSES = {
    "ventricular_capture",
    "mixed_pacing_and_qrs_edge",
}
_PACING_RESCUE_STOP_CLASSES = {
    "atrial_or_nonventricular",
    "qrs_edge_artifact",
}
_MEASUREMENT_RESELECT_MIN_NATIVE_BEATS = 2
_MEASUREMENT_RESELECT_NATIVE_MAX_QRS_MS = 115.0
_MEASUREMENT_RESELECT_NATIVE_MAX_SPREAD_MS = 25.0
_MEASUREMENT_RESELECT_SELECTED_MIN_QRS_MS = 115.0
_MEASUREMENT_RESELECT_MIN_QRS_DELTA_MS = 25.0
_MEASUREMENT_RESELECT_STABLE_TEMPLATE_CORR = 0.95
_MEASUREMENT_RESELECT_STABLE_TEMPLATE_MAX_OUTLIER_FRAC = 0.25
_QRS_TAIL_QT_RESCUE_MIN_LEADS = 6
_QRS_TAIL_QT_RESCUE_MAX_SPREAD_MS = 25.0
_QRS_TAIL_QT_RESCUE_MIN_JT_MS = 180.0
_QRS_TAIL_QT_RESCUE_MAX_JT_MS = 500.0


def _resolve_mains_frequency(
    ecg: np.ndarray,
    fs: int,
    configured: int | str | None,
) -> int:
    if configured not in {None, "auto"}:
        value = int(configured)
        if value not in {50, 60}:
            raise ValueError("mains_freq must be 50, 60, 'auto', or None")
        return value
    if fs <= 122:
        return 50
    sample = np.asarray(ecg, dtype=float)
    spectrum = np.abs(np.fft.rfft(sample - np.mean(sample, axis=1, keepdims=True), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(sample.shape[1], d=1.0 / float(fs))
    powers = {}
    for candidate in (50, 60):
        mask = np.abs(frequencies - candidate) <= 1.0
        powers[candidate] = float(np.median(np.sum(spectrum[:, mask], axis=1)))
    return max(powers, key=powers.get)


def _av_block_evidence(rule_summary: Dict[str, Any]) -> Dict[str, Any]:
    pauses = rule_summary.get("pauses") if isinstance(rule_summary, dict) else {}
    evidence = pauses.get("av_block_evidence", {}) if isinstance(pauses, dict) else {}
    return evidence if isinstance(evidence, dict) else {}


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


def _copy_qt_measurements(target: Any, source: Any) -> None:
    for attr in (
        "qt_ms",
        "qtc_bazett_ms",
        "qtc_fridericia_ms",
        "qt_dispersion_ms",
        "qt_dispersion_independent_ms",
        "qt_dispersion_p90_p10_ms",
        "qt_dispersion_source",
        "qt_dispersion_used_leads",
        "qt_dispersion_legacy_excluded_leads",
        "qt_source",
        "qt_used_leads",
        "qt_reliability",
        "qt_reportable",
        "qt_unreliable_reasons",
        "qt_path",
        "qt_confidence_reason",
        "qt_excluded_leads",
        "qt_lead_weights",
        "consensus_vs_independent_per_lead",
        "qt_robust_center_ms",
        "qt_latest_p85_ms",
        "t_fusion_support",
        "t_fusion_mad_ms",
        "t_fusion_ci_half_width_ms",
        "t_fusion_reliable",
        "t_fusion_cluster_count",
        "t_fusion_selected_cluster_score",
        "t_fusion_selected_cluster_support",
        "t_fusion_lead_groups",
        "t_fusion_methods",
        "t_fusion_reliability_reasons",
        "t_global_tpte_ms",
        "t_derived_disagreement_ms",
        "t_tail_incomplete_leads",
        "t_tail_incomplete_fraction",
        "t_systematic_early_risk",
        "t_systematic_early_fraction",
        "t_morphology_guard_pass",
    ):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


_QT_REJECT_RELIABILITY_TIERS = ("fallback", "low_confidence")
_QT_REJECT_RECORD_GRADES = ("Q2", "Q3")


def _apply_qt_reject_gate(global_features: Any, record_grade: Optional[str]) -> None:
    """Null QT/QTc rather than report a number when the evidence is too weak.

    quality.py's diagnostic gate deliberately does not suppress measurements
    (it only gates diagnostic *statements*), so a record can reach here with
    qt_reliability already at its weakest tiers ("fallback" — a single
    preferred-lead value with no cross-validation, or "low_confidence" — too
    few reliable QT leads). That alone doesn't justify rejecting a number
    (an otherwise-clean record can still fall back for incidental reasons),
    but combined with a poor overall record grade (Q2/Q3: reduced reliable
    lead coverage or other technical issues) it is — reject and record a
    reason rather than emit a plausible-looking but unsupported QT/QTc.
    """
    reliability = getattr(global_features, "qt_reliability", "unavailable")
    if reliability not in _QT_REJECT_RELIABILITY_TIERS:
        return
    if record_grade not in _QT_REJECT_RECORD_GRADES:
        return
    global_features.qt_rejected = True
    global_features.qt_reject_reason = f"qt_reliability={reliability},record_grade={record_grade}"
    for attr in ("qt_ms", "qtc_bazett_ms", "qtc_fridericia_ms", "qtc_framingham_ms", "qtc_hodges_ms"):
        setattr(global_features, attr, None)


def _rescue_intermittent_paced_qt_from_native(
    global_features: Any,
    representative_leads: Dict[str, Any],
    measurement_beat_features: List[Any],
    r_locs: np.ndarray,
    fs: int,
    paced_fraction: float,
) -> Optional[str]:
    if paced_fraction >= _PACING_INTERMITTENT_QT_NATIVE_RESCUE_MAX_PACED_FRACTION:
        return None
    if str(getattr(global_features, "qt_source", "") or "") != "irregular_raw_beat_qt_core":
        return None
    current_qt = _finite_float(getattr(global_features, "qt_ms", None))
    native_global = compute_global_features(
        representative_leads,
        measurement_beat_features,
        r_locs,
        fs,
        paced=False,
    )
    native_qt = _finite_float(getattr(native_global, "qt_ms", None))
    if native_qt is None:
        return None
    if current_qt is not None and native_qt - current_qt < _PACING_INTERMITTENT_QT_NATIVE_RESCUE_MIN_DELTA_MS:
        return None
    _copy_qt_measurements(global_features, native_global)
    return str(getattr(native_global, "qt_source", None) or "native_global")


def _probable_limb_lead_reversal(lead_reversal: Dict[str, Any]) -> bool:
    if not isinstance(lead_reversal, dict):
        return False
    return bool(
        lead_reversal.get("probable_extremity_reversal")
        or lead_reversal.get("probable_ra_la")
        or lead_reversal.get("probable_ra_ll")
        or lead_reversal.get("probable_la_ll")
    )


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


def _paced_beat_fraction(paced_beat_ids: List[int], total_beats: int) -> float:
    if total_beats <= 0:
        return 0.0
    return float(len({int(beat_id) for beat_id in paced_beat_ids})) / float(total_beats)


def _rescue_qt_after_qrs_tail_settling(
    *,
    global_features: Any,
    representative_leads: Dict[str, Any],
    r_locs: np.ndarray,
    fs: int,
    qrs_tail_settling_rescue: Dict[str, Any],
) -> Optional[str]:
    if (
        not bool(qrs_tail_settling_rescue.get("applied", False))
        or getattr(global_features, "qt_ms", None) is not None
        or fs <= 0
        or len(r_locs) < 2
    ):
        return None

    values: List[Tuple[str, float]] = []
    for lead, representative in representative_leads.items():
        params = getattr(representative, "params", {}) or {}
        if not bool(params.get("reliable_for_global", False)):
            continue
        value = _finite_float(params.get("qt_consensus_ms"))
        if value is None or not (250.0 <= value <= 650.0):
            continue
        values.append((lead, value))
    if len(values) < _QRS_TAIL_QT_RESCUE_MIN_LEADS:
        return None

    qt_values = np.asarray([value for _lead, value in values], dtype=float)
    if float(np.ptp(qt_values)) > _QRS_TAIL_QT_RESCUE_MAX_SPREAD_MS:
        return None
    qt_ms = float(np.median(qt_values))
    qrs_ms = _finite_float(getattr(global_features, "qrs_ms", None))
    if qrs_ms is None or not (
        _QRS_TAIL_QT_RESCUE_MIN_JT_MS
        <= qt_ms - qrs_ms
        <= _QRS_TAIL_QT_RESCUE_MAX_JT_MS
    ):
        return None

    rr_sec = float(np.median(np.diff(np.asarray(r_locs, dtype=float)))) / float(fs)
    if not np.isfinite(rr_sec) or rr_sec <= 0.0:
        return None
    global_features.qt_ms = qt_ms
    global_features.qtc_bazett_ms = float(qt_ms / np.sqrt(rr_sec))
    global_features.qtc_fridericia_ms = float(qt_ms / np.cbrt(rr_sec))
    global_features.qt_source = "qrs_tail_settling_consensus_rescue"
    global_features.qt_used_leads = [lead for lead, _value in values]
    global_features.qt_reliability = "rescued"
    global_features.qt_reportable = True
    global_features.qt_unreliable_reasons = []
    global_features.qt_path = "qrs_tail_settling_consensus"
    global_features.qt_confidence_reason = (
        "systematic_multilead_qrs_tail_and_t_end_consensus"
    )
    return "qrs_tail_settling_consensus_rescue"


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


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _median_or_none(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


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
        input_unit = (
            amplitude_unit
            or (getattr(meta, "amplitude_unit", None) if meta is not None else None)
            or "mV"
        )
        input_gain = (
            gain_uv_per_lsb
            if gain_uv_per_lsb is not None
            else (getattr(meta, "gain_uv_per_lsb", None) if meta is not None else None)
        )
        ecg, fs_run, input_contract = validate_ecg_input(
            ecg_12lead,
            fs,
            self.fs_internal,
            lead_names=lead_names,
            amplitude_unit=input_unit,
            gain_uv_per_lsb=input_gain,
            **({"allow_limited_leads": True} if self.input_mode == "limited" else {}),
        )
        available_leads = input_contract.get("available_leads")
        quality_options = {"available_leads": available_leads} if available_leads is not None else {}

        ecg_rs = resample_ecg(ecg, fs, fs_run)
        mains_freq_run = _resolve_mains_frequency(ecg_rs, fs_run, self.mains_freq)
        # Measurement signal: baseline removal + mains notch (no LP).
        # Used for all amplitude/boundary measurements to preserve peak heights.
        ecg_an = analysis_signal(ecg_rs, fs_run, mains_hz=mains_freq_run)
        # Detection signal: ecg_an + low-pass filter.
        # Used only for QRS R-peak detection to suppress muscle noise.
        ecg_det = lowpass_filter(ecg_an, fs_run, cutoff_hz=self.lp_hz, order=4)

        adc_full_scale_mv = (
            getattr(meta, "adc_full_scale_mv", None)
            if meta is not None
            else None
        )
        raw_quality = compute_quality(
            ecg_rs,
            fs_run,
            mains_hz=mains_freq_run,
            adc_full_scale_mv=adc_full_scale_mv,
        )
        raw_record_quality = summarize_record_quality(raw_quality, **quality_options)
        quality = compute_quality(
            ecg_an,
            fs_run,
            mains_hz=mains_freq_run,
            adc_full_scale_mv=adc_full_scale_mv,
        )
        record_quality = summarize_record_quality(quality, **quality_options)
        refinement_kwargs = {"refinement": self.refinement} if self.refinement.enabled else {}
        qrs_options = ({"quality": quality, "quality_reference": True}
                       if self.refinement.qrs_quality_reference else {})
        if available_leads is not None:
            qrs_options["leads"] = tuple(STANDARD_12_LEADS.index(name) for name in available_leads)
        if self.refinement.qrs_adaptive_consensus:
            qrs_options["adaptive_consensus"] = True
            qrs_options["quality"] = quality
        rep_left_ms = 450 if self.refinement.representative_robust else 300
        representative_options = ({"left_ms": rep_left_ms, "robust_alignment": True}
                                  if self.refinement.representative_robust else {})
        lead_reversal = detect_limb_lead_reversal(ecg_an) if self.enable_lead_reversal and available_leads is None else {}
        pacing_detection_cache = (
            prepare_pacing_detection_cache(ecg_rs, fs_run)
            if self.enable_pacing
            else None
        )
        pacing_result = detect_pacing_spikes(
            ecg_rs,
            fs_run,
            detection_cache=pacing_detection_cache,
        ) if self.enable_pacing else {
            "spike_times": [],
            "paced": False,
            "state": "off",
            "lead_vote_count": 0,
        }
        ecg_measure = ecg_an
        ecg_detect = ecg_det
        pre_despike_qrs_result = None
        if pacing_result["spike_times"]:
            if (
                bool(pacing_result.get("paced", False))
                and str(pacing_result.get("state") or "off") == "on"
                and len(pacing_result["spike_times"]) >= _PACING_QRS_RESCUE_MIN_CAPTURE_BEATS
            ):
                pre_despike_qrs_result = detect_qrs_multilead_with_meta(ecg_det, fs_run, **qrs_options)
            ecg_measure = remove_pacing_spikes(ecg_an, pacing_result["spike_times"], fs_run)
            ecg_detect = lowpass_filter(ecg_measure, fs_run, cutoff_hz=self.lp_hz, order=4)

        qrs_result = detect_qrs_multilead_with_meta(ecg_detect, fs_run, **qrs_options)
        r_locs = np.asarray(qrs_result.r_locs, dtype=int)
        pacing_qrs_rescue = _pacing_qrs_rescue_evidence(
            pacing_result=pacing_result,
            pre_despike_qrs_result=pre_despike_qrs_result,
            despiked_qrs_result=qrs_result,
            fs=fs_run,
        )
        if pacing_qrs_rescue["applied"]:
            rescued_qrs_result, rescue_detail = _pacing_qrs_cleaned_single_lead_rescue(
                ecg_detect=ecg_detect,
                fs=fs_run,
                quality=quality,
                pacing_result=pacing_result,
                pre_despike_qrs_result=pre_despike_qrs_result,
            )
            if rescued_qrs_result is None:
                pacing_qrs_rescue = {
                    **pacing_qrs_rescue,
                    "applied": False,
                    "reason": "no_regular_cleaned_single_lead_confirmation",
                }
            else:
                pacing_qrs_rescue = {**pacing_qrs_rescue, **rescue_detail}
                qrs_result = rescued_qrs_result
                r_locs = np.asarray(qrs_result.r_locs, dtype=int)
        if pacing_result["spike_times"]:
            raw_spike_times = [int(s) for s in pacing_result["spike_times"]]
            pacing_result = validate_pacing_spikes_against_qrs(
                ecg_rs,
                fs_run,
                raw_spike_times,
                r_locs,
                pacing_result,
            )
            validated_spike_times = [int(s) for s in pacing_result.get("spike_times", [])]
            if (
                pacing_result.get("rejection_reason") == "qrs_edge_artifact"
                and validated_spike_times != raw_spike_times
            ):
                ecg_measure = ecg_an
                ecg_detect = ecg_det
                if validated_spike_times:
                    ecg_measure = remove_pacing_spikes(ecg_an, validated_spike_times, fs_run)
                    ecg_detect = lowpass_filter(ecg_measure, fs_run, cutoff_hz=self.lp_hz, order=4)
                qrs_result = detect_qrs_multilead_with_meta(ecg_detect, fs_run, **qrs_options)
                r_locs = np.asarray(qrs_result.r_locs, dtype=int)
                if not validated_spike_times:
                    pacing_qrs_rescue = {
                        **pacing_qrs_rescue,
                        "applied": False,
                        "reason": "pacing_validation_rejected_qrs_edge_artifact",
                    }
        if self.enable_pacing:
            (
                pacing_result,
                ecg_measure,
                ecg_detect,
                qrs_result,
                r_locs,
            ) = _try_attenuated_pacing_rescue(
                ecg_rs=ecg_rs,
                ecg_an=ecg_an,
                fs_run=fs_run,
                lp_hz=self.lp_hz,
                current_pacing_result=pacing_result,
                current_ecg_measure=ecg_measure,
                current_ecg_detect=ecg_detect,
                current_qrs_result=qrs_result,
                current_r_locs=r_locs,
                pacing_detection_cache=pacing_detection_cache,
            )
        acquisition_qc = assess_acquisition_chain(
            ecg_rs,
            fs_run,
            r_locs,
            quality,
            # The swap detector has no `severely_inconsistent` key, so passing
            # it here silently disabled the LEAD_CONFIGURATION_INVALID reject.
            # The limb-equation residual lives in the input contract.
            limb_lead_consistency=input_contract.get("limb_lead_consistency"),
        )
        acquisition_qc["automatic_compensation_applied"] = False
        acquisition_qc["automatic_compensation_rejected_reason"] = None
        approved_delays = dict(
            acquisition_qc.get("approved_delay_samples") or {}
        )
        if approved_delays and not pacing_result.get("spike_times"):
            compensated_measure, applied_measure = apply_channel_delay_compensation(
                ecg_measure,
                approved_delays,
            )
            compensated_detect, _ = apply_channel_delay_compensation(
                ecg_detect,
                approved_delays,
            )
            compensated_qrs = detect_qrs_multilead_with_meta(
                compensated_detect,
                fs_run,
                **qrs_options,
            )
            compensated_r = np.asarray(compensated_qrs.r_locs, dtype=int)
            qrs_alignment_ms = None
            if compensated_r.size == r_locs.size and r_locs.size:
                qrs_alignment_ms = float(
                    np.percentile(
                        np.abs(compensated_r - r_locs),
                        95,
                    )
                    * 1000.0
                    / float(fs_run)
                )
            if (
                applied_measure
                and compensated_r.size == r_locs.size
                and qrs_alignment_ms is not None
                and qrs_alignment_ms <= 12.0
            ):
                ecg_measure = compensated_measure
                ecg_detect = compensated_detect
                qrs_result = compensated_qrs
                r_locs = compensated_r
                acquisition_qc["automatic_compensation_applied"] = True
                acquisition_qc["automatic_compensation_leads"] = sorted(
                    applied_measure
                )
                acquisition_qc["post_compensation_qrs_p95_shift_ms"] = (
                    qrs_alignment_ms
                )
            else:
                acquisition_qc["automatic_compensation_rejected_reason"] = (
                    "qrs_detection_not_stable_after_compensation"
                )
                acquisition_qc["post_compensation_qrs_p95_shift_ms"] = (
                    qrs_alignment_ms
                )
        elif approved_delays and pacing_result.get("spike_times"):
            acquisition_qc["automatic_compensation_rejected_reason"] = (
                "pacing_spikes_present"
            )
        pacing_failures = detect_pacing_failures(
            spike_times=[int(s) for s in pacing_result["spike_times"]],
            qrs_times=[int(r) for r in r_locs],
            fs=fs_run,
        )
        pacing_spike_beat_ids = []
        pacing_spike_offsets_samples = []
        if pacing_result["spike_times"]:
            for beat_id, r in enumerate(r_locs):
                spike_lo = int(r - 0.08 * fs_run)
                spike_hi = int(r + 0.02 * fs_run)
                nearby_spikes = [
                    int(s)
                    for s in pacing_result["spike_times"]
                    if spike_lo <= int(s) <= spike_hi
                ]
                if nearby_spikes:
                    nearest_spike = min(nearby_spikes, key=lambda s: abs(int(s) - int(r)))
                    pacing_spike_beat_ids.append(int(beat_id))
                    pacing_spike_offsets_samples.append(int(nearest_spike) - int(r))
        pacing_evidence_quality = assess_pacing_evidence_quality(
            pacing_result=pacing_result,
            spike_times=pacing_result["spike_times"],
            qrs_count=len(r_locs),
            pacing_spike_beat_ids=pacing_spike_beat_ids,
            spike_offsets_samples=pacing_spike_offsets_samples,
            pacing_failures=pacing_failures,
            fs=fs_run,
        )
        pacing_capture_confirmed = _pacing_capture_alignment_confirmed(
            pacing_result,
            pacing_spike_offsets_samples,
            fs_run,
            pacing_evidence_quality,
        )
        paced_beat_ids = (
            list(pacing_spike_beat_ids)
            if pacing_capture_confirmed or not bool(pacing_result.get("paced", False))
            else []
        )
        beat_groups = (
            cluster_beats(ecg_measure, r_locs, fs_run, paced_beat_ids=paced_beat_ids,
                          **({"preserve_outliers": True} if self.refinement.grouping_outliers else {}))
            if self.compute_grouping
            else {1: list(range(len(r_locs)))}
        )
        dominant_group_id, measurement_beat_ids = _select_measurement_group(beat_groups, paced_beat_ids)
        measurement_group_paced = bool(
            pacing_result.get("paced", False)
            and _selected_group_paced_majority(measurement_beat_ids, paced_beat_ids)
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
        if self.enable_hybrid_r_localization:
            apply_hybrid_r_localization(
                beat_features,
                localization_ecg=ecg_detect,
                r_locs=r_locs,
                fs=fs_run,
            )
        if self.enable_hybrid_wave_localization:
            apply_hybrid_wave_localization(
                beat_features,
                localization_ecg=ecg_measure,
                r_locs=r_locs,
                fs=fs_run,
            )
        if self.enable_hybrid_st_measurement:
            apply_hybrid_st_measurement(
                beat_features,
                measurement_ecg=ecg_measure,
                r_locs=r_locs,
                fs=fs_run,
                quality=quality,
            )
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
            and _selected_group_paced_majority(measurement_beat_ids, paced_beat_ids)
        )
        confirmed_pacing_context = bool(measurement_group_paced or pacing_capture_confirmed)
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
                if self.enable_hybrid_r_localization:
                    apply_hybrid_r_localization(
                        beat_features,
                        localization_ecg=ecg_detect,
                        r_locs=r_locs,
                        fs=fs_run,
                    )
                if self.enable_hybrid_wave_localization:
                    apply_hybrid_wave_localization(
                        beat_features,
                        localization_ecg=ecg_measure,
                        r_locs=r_locs,
                        fs=fs_run,
                    )
                if self.enable_hybrid_st_measurement:
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
        if self.enable_t_wave_refinement:
            refine_t_wave_boundaries(
                beat_features,
                measurement_ecg=ecg_measure,
                r_locs=r_locs,
                fs=fs_run,
                quality=quality,
                **({"fusion_audit": t_fusion_audit} if (self.refinement.t_correlated_fusion
                    or self.refinement.t_boundary_projection or self.refinement.t_sequence_selection
                    or self.refinement.t_projection_offset_only or self.refinement.t_sequence_offset_only) else {}),
                **refinement_kwargs,
            )
            if self.enable_hybrid_st_measurement:
                apply_hybrid_st_measurement(
                    beat_features,
                    measurement_ecg=ecg_measure,
                    r_locs=r_locs,
                    fs=fs_run,
                    quality=quality,
                )
        p_config_kwargs = dict(model_arbitration=self.refinement.p_model_arbitration,
                               multiple_candidates=self.refinement.p_multiple_candidates)
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
               if available_leads is not None or self.refinement.p_model_arbitration or self.refinement.p_multiple_candidates else {}),
        )
        atrial_events = extract_atrial_events(beat_features, quality, r_locs, fs_run, ecg=ecg_measure)
        atrial_validation_audit: Dict[str, Any] = {}
        if self.refinement.atrial_event_validation:
            from .atrial_validation import validate_atrial_events
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
        if self.refinement.p_boundary_correction:
            p_corrections = correct_p_boundaries(beat_features, p_wave_assessments, ecg_measure, fs_run)
            if p_corrections:
                if self.enable_hybrid_st_measurement:
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
                if self.refinement.atrial_event_validation:
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
