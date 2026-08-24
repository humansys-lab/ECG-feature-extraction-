from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Union


STANDARD_12_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
]


@dataclass
class BoundaryEstimate:
    """Unified boundary descriptor with confidence. Used internally by delineation."""
    onset:      Optional[int]
    peak1:      Optional[int]
    peak2:      Optional[int]   # None for single-peak waves
    offset:     Optional[int]
    confidence: float           # 0.0 – 1.0
    method:     str             # "geometric" | "threshold" | "threshold_fallback"
    flags:      List[str] = field(default_factory=list)


@dataclass
class PatientMeta:
    age: Optional[float] = None
    age_days: Optional[float] = None
    sex: Optional[str] = None
    meds: Optional[List[str]] = None
    clinical_classifications: Optional[List[str]] = None
    device: Optional[str] = None
    acquisition_time: Optional[str] = None
    amplitude_unit: str = "mV"
    gain_uv_per_lsb: Optional[float] = None
    adc_full_scale_mv: Optional[float] = None
    filter_highpass_hz: Optional[float] = None
    filter_lowpass_hz: Optional[float] = None
    muscle_filter_enabled: Optional[bool] = None
    # Kept at the end to preserve the positional constructor contract.
    patient_id: Optional[str] = None


DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class ResolvedPatientAge:
    """One internally consistent age pair for all interpretation layers."""

    age_years: Optional[float]
    age_days: Optional[float]
    source: str

    @property
    def known(self) -> bool:
        return self.age_days is not None


def _patient_field(patient: Any, name: str) -> Any:
    if isinstance(patient, Mapping):
        return patient.get(name)
    return getattr(patient, name, None)


def _finite_nonnegative(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def resolve_patient_age(patient: Any) -> ResolvedPatientAge:
    """Resolve demographics with exact-day precedence and no invalid fallback.

    ``age_days`` is used when supplied because neonatal thresholds cannot be
    selected reliably from a rounded year-age. If that explicit field is
    malformed, a potentially conflicting ``age`` value is not substituted.
    """

    raw_days = _patient_field(patient, "age_days")
    if raw_days is not None:
        days = _finite_nonnegative(raw_days)
        if days is None:
            return ResolvedPatientAge(None, None, "invalid_age_days")
        return ResolvedPatientAge(days / DAYS_PER_YEAR, days, "age_days")

    raw_years = _patient_field(patient, "age")
    if raw_years is not None:
        years = _finite_nonnegative(raw_years)
        if years is None:
            return ResolvedPatientAge(None, None, "invalid_age_years")
        return ResolvedPatientAge(years, years * DAYS_PER_YEAR, "age_years")

    return ResolvedPatientAge(None, None, "missing")


@dataclass
class LeadQuality:
    lead: str
    baseline_wander_score: float
    muscle_noise_score: float
    powerline_score: float
    clipping_score: float
    flatline_score: float
    missing: bool
    reliable: bool
    flags: List[str] = field(default_factory=list)
    grade: str = "Q0"
    reason_codes: List[str] = field(default_factory=list)
    # Wave-specific reliability flags (T003)
    reliable_for_p:   Optional[bool] = None
    reliable_for_qrs: Optional[bool] = None
    reliable_for_t:   Optional[bool] = None
    reliable_for_qt:  Optional[bool] = None
    flatline_fraction: Optional[float] = None
    saturation_fraction: Optional[float] = None
    p_sqi: Optional[float] = None
    k_sqi: Optional[float] = None
    b_sqi: Optional[float] = None

    def __post_init__(self) -> None:
        if self.reliable_for_p is None:
            self.reliable_for_p = self.reliable
        if self.reliable_for_qrs is None:
            self.reliable_for_qrs = self.reliable
        if self.reliable_for_t is None:
            self.reliable_for_t = self.reliable
        if self.reliable_for_qt is None:
            self.reliable_for_qt = self.reliable


@dataclass
class BeatAnnotation:
    beat_id: int
    r_index: int
    paced: bool
    group_id: int
    rr_prev_ms: Optional[float]
    rr_next_ms: Optional[float]


@dataclass
class WaveBounds:
    onset: Optional[int]
    peak: Optional[int]
    offset: Optional[int]


@dataclass
class QRSCandidateWindow:
    detector_peak_index: int
    search_start_index: int
    search_end_index: int
    refined_r_index: int
    detector_height: float
    confidence: float


@dataclass
class QRSDetectorResult:
    r_locs: List[int]
    qrs_candidate_windows: List[QRSCandidateWindow]
    qrs_detector_confidence: List[float]
    energy_threshold: float
    detection_leads: List[str]
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class PWaveLeadBoundary:
    """Auditable lead-local P-wave boundary produced by the robust P pipeline."""

    lead: str
    onset: Optional[int]
    peak: Optional[int]
    offset: Optional[int]
    onset_ci_low: Optional[int] = None
    onset_ci_high: Optional[int] = None
    offset_ci_low: Optional[int] = None
    offset_ci_high: Optional[int] = None
    onset_confidence: float = 0.0
    offset_confidence: float = 0.0
    onset_sigma_ms: Optional[float] = None
    offset_sigma_ms: Optional[float] = None
    quality_score: float = 0.0
    local_snr: Optional[float] = None
    informative: bool = False
    hard_quality_pass: bool = False
    baseline_mode: str = "unavailable"
    baseline_confidence: float = 0.0
    baseline_uncertainty_mv: Optional[float] = None
    t_reconstructed: bool = False
    t_reconstruction_validated: bool = False
    ta_ambiguous: bool = False
    on_t_overlap: bool = False
    branch_disagreement_ms: Optional[float] = None
    methods: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


@dataclass
class PWaveBeatAssessment:
    """Strict and robust multi-lead P-wave boundary contract for one beat."""

    beat_id: int
    strict_onset: Optional[int]
    strict_offset: Optional[int]
    robust_onset: Optional[int]
    robust_offset: Optional[int]
    onset_ci_low: Optional[int] = None
    onset_ci_high: Optional[int] = None
    offset_ci_low: Optional[int] = None
    offset_ci_high: Optional[int] = None
    onset_confidence: float = 0.0
    offset_confidence: float = 0.0
    p_state: str = "OVERLAP_UNCERTAIN"
    accepted: bool = False
    reject_reasons: List[str] = field(default_factory=list)
    valid_leads: List[str] = field(default_factory=list)
    valid_lead_groups: List[str] = field(default_factory=list)
    per_lead: Dict[str, PWaveLeadBoundary] = field(default_factory=dict)
    baseline_mode: str = "unavailable"
    t_reconstructed: bool = False
    ta_ambiguous: bool = False
    global_detector_method: str = "whitened_spatial_activity"
    svd_dimension: int = 1
    coarse_window_start: Optional[int] = None
    coarse_window_end: Optional[int] = None
    morphology_cluster_id: Optional[int] = None
    temporal_jitter_ms: Optional[float] = None
    ci_calibration_status: str = "uncalibrated"


@dataclass
class LeadBeatFeatures:
    lead: str
    beat_id: int
    p: WaveBounds
    qrs: WaveBounds
    t: WaveBounds
    qt_ms: Optional[float]
    pr_ms: Optional[float]
    qrs_ms: Optional[float]
    p_amp_mv: Optional[float]
    qrs_area: Optional[float]
    q_amp_mv: Optional[float]
    r_amp_mv: Optional[float]
    s_amp_mv: Optional[float]
    st_on_mv: Optional[float]
    st_mid_mv: Optional[float]
    st_80ms_mv: Optional[float]
    t_amp_mv: Optional[float]
    j_index: Optional[int]
    beat_window_start_index: Optional[int] = None
    glasgow_measurements: Dict[str, object] = field(default_factory=dict)
    st_j_original_index: Optional[int] = None
    st_j_remeasured_index: Optional[int] = None
    st_j_remeasure_delta_ms: Optional[float] = None
    st_j_remeasure_reason: Optional[str] = None
    st_j_remeasure_confidence: Optional[float] = None
    st_j_remeasure_consensus_index: Optional[int] = None
    st_j_remeasure_support: Optional[int] = None
    st_j_remeasure_used_leads: Optional[str] = None
    st_j_remeasure_excluded_leads: Optional[str] = None
    # 12SL measurement profile fields. These are parallel outputs and do not
    # replace the native DXL-style measurement fields above.
    twelve_sl_stj_mv: Optional[float] = None
    twelve_sl_stm_mv: Optional[float] = None
    twelve_sl_ste_mv: Optional[float] = None
    twelve_sl_stm_offset_ms: Optional[float] = None
    twelve_sl_ste_offset_ms: Optional[float] = None
    twelve_sl_qrs_area_uv_ms: Optional[float] = None
    twelve_sl_qrs_signed_area_uv_ms: Optional[float] = None
    twelve_sl_qrs_balance_uv: Optional[float] = None
    twelve_sl_qrs_deflection_uv: Optional[float] = None
    twelve_sl_minimum_st_uv: Optional[float] = None
    twelve_sl_special_t_uv: Optional[float] = None
    twelve_sl_t_prime_uv: Optional[float] = None
    twelve_sl_t_prime_area_uv_ms: Optional[float] = None
    twelve_sl_st_confidence: Optional[float] = None
    twelve_sl_st_confidence_reason: Optional[str] = None
    twelve_sl_qrs_significant: bool = False
    baseline_source: str = "pr_segment"
    baseline_confidence: float = 0.0
    qrs_signed_area: Optional[float] = None
    q_onset: Optional[int] = None
    q_offset: Optional[int] = None
    q_duration_ms: Optional[float] = None
    q_area_mv_ms: Optional[float] = None
    q_r_ratio: Optional[float] = None
    initial_qrs_area_mv_ms: Optional[float] = None
    initial_qrs_net_mv: Optional[float] = None
    p_signed_area: Optional[float] = None
    t_signed_area: Optional[float] = None
    delta_present: bool = False
    qrs_notch_sign: int = 0
    flags: List[str] = field(default_factory=list)
    # Extended fields (T022)
    jt_ms:         Optional[float] = None
    qt_confidence: float           = 0.0
    t_end_method:  str             = "threshold"
    t_end_repair_reason: Optional[str] = None
    t_offset_original_index: Optional[int] = None
    t_offset_repaired_index: Optional[int] = None
    t_offset_repair_delta_ms: Optional[float] = None
    t_offset_repair_reason: Optional[str] = None
    t_offset_repair_confidence: Optional[float] = None
    t_offset_repair_consensus_index: Optional[int] = None
    t_offset_repair_support: Optional[int] = None
    t_offset_repair_used_leads: Optional[str] = None
    t_offset_repair_excluded_leads: Optional[str] = None
    t_offset_dual_original_index: Optional[int] = None
    t_offset_dual_rescued_index: Optional[int] = None
    t_offset_dual_delta_ms: Optional[float] = None
    t_offset_dual_reason: Optional[str] = None
    t_offset_dual_confidence: Optional[float] = None
    t_offset_dual_consensus_index: Optional[int] = None
    t_offset_dual_support: Optional[int] = None
    t_offset_dual_used_leads: Optional[str] = None
    t_offset_dual_excluded_leads: Optional[str] = None
    t_offset_dual_chord_index: Optional[int] = None
    t_offset_dual_slope_index: Optional[int] = None
    t_offset_dual_tangent_index: Optional[int] = None
    t_offset_dual_method_spread_ms: Optional[float] = None
    # T-wave specific refinement and robust multi-lead fusion.  The lead-local
    # ``t`` bounds remain independent measurements; the consensus fields below
    # make the record-aligned centre/latest-repolarisation definitions explicit.
    t_wavelet_onset_index: Optional[int] = None
    t_wavelet_offset_index: Optional[int] = None
    t_trapezium_offset_index: Optional[int] = None
    t_candidate_spread_ms: Optional[float] = None
    t_local_snr_db: Optional[float] = None
    t_boundary_stability_ms: Optional[float] = None
    t_sqi_score: Optional[float] = None
    t_sqi_pass: bool = False
    t_fusion_weight: Optional[float] = None
    t_offset_robust_center_index: Optional[int] = None
    t_offset_latest_p85_index: Optional[int] = None
    t_offset_fusion_ci_low_index: Optional[int] = None
    t_offset_fusion_ci_high_index: Optional[int] = None
    t_offset_fusion_support: Optional[int] = None
    t_offset_fusion_mad_ms: Optional[float] = None
    t_offset_fusion_ci_half_width_ms: Optional[float] = None
    t_offset_fusion_used_leads: Optional[str] = None
    t_offset_fusion_excluded_leads: Optional[str] = None
    t_offset_fusion_reliable: bool = False
    t_offset_statistical_fusion_reliable: bool = False
    t_offset_cluster_count: Optional[int] = None
    t_offset_selected_cluster_score: Optional[float] = None
    t_offset_selected_cluster_support: Optional[int] = None
    t_offset_selected_cluster_lead_groups: Optional[str] = None
    t_offset_selected_cluster_methods: Optional[str] = None
    t_offset_fusion_reliability_reason: Optional[str] = None
    t_peak_consensus_index: Optional[int] = None
    t_global_tpte_ms: Optional[float] = None
    t_offset_derived_disagreement_ms: Optional[float] = None
    t_offset_tail_incomplete_leads: Optional[str] = None
    t_offset_systematic_early_risk: bool = False
    t_offset_morphology_guard_pass: bool = False
    qt_latest_p85_ms: Optional[float] = None
    t_rms_onset_index: Optional[int] = None
    t_rms_offset_index: Optional[int] = None
    t_pc1_onset_index: Optional[int] = None
    t_pc1_offset_index: Optional[int] = None
    t_derived_spread_ms: Optional[float] = None
    # Beat-level quality (T004)
    beat_noise_score:          float = 0.0
    beat_baseline_shift:       float = 0.0
    beat_template_corr:        float = 0.0
    beat_measurement_reliable: bool  = True
    # Boundary confidence scores (T020)
    p_confidence:   float = 0.0
    qrs_confidence: float = 0.0
    p_onset_confidence: Optional[float] = None
    p_offset_confidence: Optional[float] = None
    # P-specific local signal quality and deterministic boundary stability.
    # ``p_confidence`` mainly describes wave presence/amplitude.  These fields
    # keep that concept separate from how stable the onset/offset locations are
    # under small deterministic threshold/method perturbations.
    p_local_noise_rms_mv: Optional[float] = None
    p_local_noise_source: Optional[str] = None
    p_local_snr: Optional[float] = None
    p_informative: Optional[bool] = None
    # Baseline context is explicit because the PR segment contains atrial
    # repolarisation (Ta), and at high rates there may be no isoelectric TP
    # interval before the next P wave.  ``p_tp_gap_ms`` is measured from the
    # previous T offset to the current P onset when both are available.
    p_tp_gap_ms: Optional[float] = None
    p_quiet_window_available: Optional[bool] = None
    p_on_t_overlap_risk: Optional[bool] = None
    p_ta_overlap_risk: Optional[bool] = None
    p_baseline_method: Optional[str] = None
    # Where this beat's limb-lead P amplitude was sampled: "limb_simultaneous"
    # once the six limb leads were aligned to one instant (so the Einthoven /
    # Goldberger identities hold), otherwise the reason they were not.
    p_amp_source: Optional[str] = None
    # Residual baseline drift across this beat's P search window (mV).  A P wave
    # no larger than the drift it sits on is not a measurement, which is what
    # withholds the record-level amplitude in `features.py`.
    p_window_drift_mv: Optional[float] = None
    p_baseline_confidence: Optional[float] = None
    p_onset_sigma_ms: Optional[float] = None
    p_offset_sigma_ms: Optional[float] = None
    p_boundary_stability_method: Optional[str] = None
    # QRS onset/offset precision confidence (T006)
    qrs_on_confidence:  float = 0.0
    qrs_off_confidence: float = 0.0
    # QRS component amplitudes (T013)
    r_prime_amp_mv: Optional[float] = None
    s_prime_amp_mv: Optional[float] = None
    r_duration_ms: Optional[float] = None
    r_prime_duration_ms: Optional[float] = None
    s_duration_ms: Optional[float] = None
    s_prime_duration_ms: Optional[float] = None
    qrs_num_peaks:  int = 1
    qrs_notch_count: int = 0
    # Ventricular activation time (T014)
    vat_ms: Optional[float] = None
    # Additional morphology fields (T026)
    p_dur_ms:          Optional[float] = None
    p_area:            Optional[float] = None
    p_notched:         bool = False
    p_biphasic:        bool = False
    p_notch_interval_ms: Optional[float] = None
    p_initial_duration_ms: Optional[float] = None
    p_initial_amp_mv: Optional[float] = None
    p_terminal_duration_ms: Optional[float] = None
    p_terminal_amp_mv: Optional[float] = None
    p_terminal_area_mv_ms: Optional[float] = None
    t_dur_ms:          Optional[float] = None
    t_area:            Optional[float] = None
    t_polarity:        int  = 0
    u_wave_flag:       bool = False
    # ``u_amp_mv`` is the legacy unsigned window maximum kept for the T-offset
    # fusion path. Diagnostic rules must use the signed measurement below --
    # an inverted U wave is invisible to a magnitude.
    u_amp_mv:          Optional[float] = None
    u_amp_signed_mv:   Optional[float] = None
    u_polarity:        int = 0
    u_prominence_mv:   Optional[float] = None
    u_dur_ms:          Optional[float] = None
    u_isoelectric_gap_ms: Optional[float] = None
    u_peak_index:      Optional[int] = None
    u_measurement_reliable: bool = False
    u_reject_reason:   Optional[str] = None
    t_symmetry:        Optional[float] = None
    pr_segment_level_mv: Optional[float] = None
    qrs_slur_flag:     bool = False
    st_slope_mv_per_ms: Optional[float] = None
    # Extended measurements (T027)
    tpe_ms:          Optional[float] = None   # T-peak to T-end (ms)
    st_morphology:   Optional[str]   = None   # "upsloping" | "horizontal" | "downsloping"
    fqrs_score:      float           = 0.0    # fragmented-QRS normalised score [0, 1]
    ptf_v1_mv_ms:    Optional[float] = None   # P-terminal force V1 (V1 lead only, mV·ms)
    # Multi-lead consensus intervals (T025)
    qt_consensus_ms:  Optional[float] = None   # QT from global QRS-onset to global T-end
    pr_consensus_ms:  Optional[float] = None   # PR from global P-onset to global QRS-onset
    p_onset_consensus_index: Optional[int] = None
    p_offset_consensus_index: Optional[int] = None
    p_dur_consensus_ms: Optional[float] = None
    pr_segment_consensus_ms: Optional[float] = None
    p_onset_consensus_support: Optional[int] = None
    p_offset_consensus_support: Optional[int] = None
    p_onset_consensus_spread_ms: Optional[float] = None
    p_offset_consensus_spread_ms: Optional[float] = None
    p_boundary_consensus_source: Optional[str] = None
    p_duration_guard_target_ms: Optional[float] = None
    p_duration_guard_delta_ms: Optional[float] = None
    p_duration_guard_support: Optional[int] = None
    p_duration_guard_source: Optional[str] = None
    # Three-layer P-boundary contract:
    #   raw       — lead-local delineation before multi-lead correction
    #   corrected — lead-local boundary after consensus correction/suppression
    #   consensus — record-aligned multi-lead boundary (fields above)
    # ``p`` remains the corrected lead-local boundary for backward compatibility.
    p_onset_raw_index: Optional[int] = None
    p_offset_raw_index: Optional[int] = None
    p_onset_corrected_index: Optional[int] = None
    p_offset_corrected_index: Optional[int] = None
    p_offset_correction_fraction: Optional[float] = None
    p_boundary_source: Optional[str] = None  # raw_local | lead_consensus[_blend] | suppressed
    p_onset_cluster_center_index: Optional[int] = None
    p_onset_cluster_support: Optional[int] = None
    p_onset_cluster_spread_ms: Optional[float] = None
    p_onset_cluster_pr_ms: Optional[float] = None
    p_onset_cluster_limb_support: Optional[int] = None
    p_onset_cluster_precordial_support: Optional[int] = None
    p_onset_cluster_leads: Optional[str] = None
    p_onset_consensus_reason: Optional[str] = None
    p_template_corr: Optional[float] = None
    p_multilead_support: Optional[int] = None
    p_multilead_support_score: Optional[float] = None
    p_pr_consistency_score: Optional[float] = None
    p_pp_consistency_score: Optional[float] = None
    p_candidate_context_score: Optional[float] = None
    p_context_reason: Optional[str] = None
    p_candidate_alternative_count: Optional[int] = None
    p_reselected_from_peak_index: Optional[int] = None
    p_reselected_to_peak_index: Optional[int] = None
    p_reselected_old_context_score: Optional[float] = None
    p_reselected_new_context_score: Optional[float] = None
    p_reselection_reason: Optional[str] = None
    qrs_consensus_ms: Optional[float] = None   # QRS from global QRS-onset (P75 offset) — measurement
    qrs_wide_ms:      Optional[float] = None   # QRS from global QRS-onset (P90 offset) — wide-QRS classification
    qrs_terminal_path: Optional[str] = None
    qrs_offset_used_leads: Optional[str] = None
    qrs_offset_excluded_leads: Optional[str] = None
    qrs_offset_exclusion_reason: Optional[str] = None
    qrs_late_cluster_support: Optional[int] = None
    qrs_terminal_confidence: Optional[float] = None
    qrs_offset_original_index: Optional[int] = None
    qrs_offset_repaired_index: Optional[int] = None
    qrs_offset_repair_delta_ms: Optional[float] = None
    qrs_offset_repair_reason: Optional[str] = None
    qrs_offset_repair_confidence: Optional[float] = None
    qrs_offset_repair_consensus_index: Optional[int] = None
    qrs_offset_repair_support: Optional[int] = None
    t_peak_path: Optional[str] = None
    t_polarity_expected: Optional[int] = None
    t_polarity_observed: Optional[int] = None
    st_t_confusion: bool = False
    t_confidence_reason: Optional[str] = None
    # Sample index of the true positive R-wave peak (max positive deflection
    # within [qrs.onset, qrs.offset]), matching r_amp_mv. Distinct from
    # qrs.peak, which is a beat-detection/alignment fiducial and may sit on
    # the S trough for negative-dominant complexes (aVR, V1-V3 rS/QS beats).
    r_peak_index: Optional[int] = None
    # Parallel lead-specific R/QRS localization for plotting and detector
    # evaluation. It must not replace global r_locs, qrs.peak, or boundaries.
    r_localized_index: Optional[int] = None
    r_localization_method: Optional[str] = None
    r_localization_confidence: Optional[float] = None
    r_localization_prominence_mv: Optional[float] = None
    r_localization_polarity: Optional[int] = None
    # Parallel P/T/S localizations inspired by NeuroKit2 prominence
    # delineation. These fields are for evaluation and visualization and must
    # not replace native bounds, amplitudes, intervals, or interpretation.
    p_localized_index: Optional[int] = None
    p_localized_onset: Optional[int] = None
    p_localized_offset: Optional[int] = None
    p_localization_method: Optional[str] = None
    p_localization_confidence: Optional[float] = None
    p_localization_prominence_mv: Optional[float] = None
    p_localization_polarity: Optional[int] = None
    p_localization_morphology: Optional[str] = None
    p_localization_secondary_index: Optional[int] = None
    p_localization_candidate_count: Optional[int] = None
    p_localization_absent_probability: Optional[float] = None
    p_localization_retrograde: bool = False
    t_localized_index: Optional[int] = None
    t_localized_onset: Optional[int] = None
    t_localized_offset: Optional[int] = None
    t_localization_method: Optional[str] = None
    t_localization_confidence: Optional[float] = None
    t_localization_prominence_mv: Optional[float] = None
    t_localization_polarity: Optional[int] = None
    t_localization_morphology: Optional[str] = None
    t_localization_secondary_index: Optional[int] = None
    s_peak_index: Optional[int] = None
    s_amplitude_index: Optional[int] = None
    s_prime_peak_index: Optional[int] = None
    qs_nadir_index: Optional[int] = None
    s_localization_method: Optional[str] = None
    s_localization_confidence: Optional[float] = None
    s_localization_prominence_mv: Optional[float] = None
    s_localization_morphology: Optional[str] = None
    # Parallel robust ST/J profile. The native ST fields remain the validated
    # compatibility path; these values use a PR-segment baseline, local
    # QRS-to-ST settling, multi-lead J consensus, and window medians.
    st_hybrid_j_index: Optional[int] = None
    st_hybrid_j_mv: Optional[float] = None
    st_hybrid_20ms_mv: Optional[float] = None
    st_hybrid_40ms_mv: Optional[float] = None
    st_hybrid_60ms_mv: Optional[float] = None
    st_hybrid_80ms_mv: Optional[float] = None
    st_hybrid_adaptive_index: Optional[int] = None
    st_hybrid_adaptive_mv: Optional[float] = None
    st_hybrid_mean_mv: Optional[float] = None
    st_hybrid_area_mv_ms: Optional[float] = None
    st_hybrid_slope_mv_per_ms: Optional[float] = None
    st_hybrid_curvature_mv_per_ms2: Optional[float] = None
    st_hybrid_trend: Optional[str] = None
    st_hybrid_shape: Optional[str] = None
    # Named clinical ST morphology class; see st_localization.classify_st_pattern.
    st_pattern_class: Optional[str] = None
    st_hybrid_baseline_mv: Optional[float] = None
    st_hybrid_baseline_source: Optional[str] = None
    st_hybrid_baseline_confidence: Optional[float] = None
    st_hybrid_j_method: Optional[str] = None
    st_hybrid_j_confidence: Optional[float] = None
    st_hybrid_consensus_index: Optional[int] = None
    st_hybrid_consensus_support: Optional[int] = None
    st_hybrid_noise_mv: Optional[float] = None
    st_hybrid_reliable: bool = False
    st_hybrid_unreliable_reason: Optional[str] = None


@dataclass
class RepresentativeLeadFeatures:
    lead: str
    params: Dict[str, float | int | str | bool | None]
    variance: Dict[str, float]


@dataclass
class GroupFeatures:
    group_id: int
    member_count: int
    member_pct: float
    longest_run: int
    mean_rr_ms: Optional[float]
    mean_pr_ms: Optional[float]
    mean_qrs_ms: Optional[float]
    mean_qt_ms: Optional[float]
    mean_ventr_rate_bpm: Optional[float]
    flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class GlobalFeatures:
    heart_rate_bpm: Optional[float]
    atrial_rate_bpm: Optional[float]
    pr_ms: Optional[float]
    qrs_ms: Optional[float]
    qt_ms: Optional[float]
    qtc_bazett_ms: Optional[float]
    qtc_fridericia_ms: Optional[float]
    p_axis_deg: Optional[float]
    qrs_axis_deg: Optional[float]
    t_axis_deg: Optional[float]
    st_axis_deg: Optional[float]
    qt_dispersion_ms: Optional[float]
    # Formal record-level P duration and its measurement provenance.
    p_duration_ms: Optional[float] = None
    p_duration_source: Optional[str] = None
    p_duration_used_leads: List[str] = field(default_factory=list)
    p_duration_support: int = 0
    p_duration_spread_ms: Optional[float] = None
    p_duration_reliability: str = "unavailable"
    # T024: Pacing spike detection
    pacing_spikes: Optional[List[int]] = None
    paced_rhythm: bool = False
    # T027: PTF-V1 global representative value
    ptf_v1_mv_ms: Optional[float] = None
    # T-axis amplitude reliability flag (DXL: ≥2 limb leads with |T_amp| > 150µV)
    t_axis_reliable: bool = False
    # Global QT provenance. Per-lead QT values remain raw/lead-local diagnostics.
    qt_source: Optional[str] = None
    qt_used_leads: List[str] = field(default_factory=list)
    qt_reliability: str = "unavailable"       # "reliable" | "rescued" | "fallback" | "low_confidence" | "unreliable" | "unavailable"
    qt_reportable: bool = False
    qt_unreliable_reasons: List[str] = field(default_factory=list)
    qt_path: Optional[str] = None
    qt_confidence_reason: Optional[str] = None
    qt_excluded_leads: Dict[str, str] = field(default_factory=dict)
    qt_lead_weights: Dict[str, float] = field(default_factory=dict)
    consensus_vs_independent_per_lead: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    qt_robust_center_ms: Optional[float] = None
    qt_latest_p85_ms: Optional[float] = None
    t_fusion_support: int = 0
    t_fusion_mad_ms: Optional[float] = None
    t_fusion_ci_half_width_ms: Optional[float] = None
    t_fusion_reliable: bool = False
    t_fusion_cluster_count: int = 0
    t_fusion_selected_cluster_score: Optional[float] = None
    t_fusion_selected_cluster_support: int = 0
    t_fusion_lead_groups: List[str] = field(default_factory=list)
    t_fusion_methods: List[str] = field(default_factory=list)
    t_fusion_reliability_reasons: List[str] = field(default_factory=list)
    t_global_tpte_ms: Optional[float] = None
    t_derived_disagreement_ms: Optional[float] = None
    t_tail_incomplete_leads: List[str] = field(default_factory=list)
    t_tail_incomplete_fraction: float = 0.0
    t_systematic_early_risk: bool = False
    t_systematic_early_fraction: float = 0.0
    t_morphology_guard_pass: bool = False
    heart_rate_min_bpm: Optional[float] = None
    heart_rate_max_bpm: Optional[float] = None
    qtc_framingham_ms: Optional[float] = None
    qtc_hodges_ms: Optional[float] = None
    rr_mean_ms: Optional[float] = None
    rr_sd_ms: Optional[float] = None
    rr_cv: Optional[float] = None
    rmssd_ms: Optional[float] = None
    pnn50: Optional[float] = None
    poincare_sd1_ms: Optional[float] = None
    poincare_sd2_ms: Optional[float] = None
    rr_entropy: Optional[float] = None
    qrs_t_angle_deg: Optional[float] = None
    transition_zone: Optional[float] = None
    # P90-based QRS width (record-level counterpart of per-lead qrs_wide_ms):
    # a looser consensus used to catch BBB/IVCD that the P75 measurement
    # channel (qrs_ms) under-calls when only a minority of leads corroborate
    # a late offset (e.g. a single lead in a low-reliable-lead-count record).
    qrs_wide_ms: Optional[float] = None
    # Set when QT/QTc were nulled below because the reliable-lead QT pipeline
    # degraded to its weakest tiers on a record whose overall quality is also
    # weak — reject rather than report a plausible-looking but unsupported
    # number (see api._apply_qt_reject_gate).
    qt_rejected: bool = False
    qt_reject_reason: Optional[str] = None


@dataclass
class ECGInterpretation:
    """Clinical interpretation layer – derived from ECGFeatures measurements.

    All thresholds sourced from DXL_Threshold_Reference.xlsx.
    Research scaffold only; not a validated medical device.
    """
    # --- Rhythm ---------------------------------------------------------------
    rr_cv: Optional[float]                    # RR coefficient of variation
    rr_irregularity_class: str                # "regular" | "mildly_irregular" | "irregular" | "indeterminate"
    probable_af: bool                          # irregular rhythm + absent P waves
    heart_rate_class: str                      # "extreme_bradycardia" | "bradycardia" | "normal" | "tachycardia"

    # --- Axis -----------------------------------------------------------------
    qrs_axis_class: str                        # "normal" | "LAD" | "LAFB" | "RAD" | "LPFB" | "ERAD" | "indeterminate"
    p_axis_normal: Optional[bool]              # P axis in sinus range (0–75°)
    t_axis_class: str                          # "normal" | "abnormal" | "indeterminate"
    qrs_t_angle_deg: Optional[float]          # degrees between QRS and T axes

    # --- Conduction / Intervals -----------------------------------------------
    pr_class: str                              # "short" | "normal" | "avb1" | "indeterminate"
    avb_grade: Optional[int]                   # 1, 2, 3 or None
    qrs_width_class: str                       # "normal" | "borderline_ivcd" | "nonspecific_ivcd" | "bbb" | "indeterminate"
    bundle_branch_block: Optional[str]         # None | "LBBB" | "RBBB" | "incomplete_RBBB" | "IVCD"
    qtc_class: str                             # "short" | "normal" | "borderline_prolonged" | "prolonged" | "significantly_prolonged" | "indeterminate"
    wpw_pattern: bool                          # short PR + delta + wide QRS

    # --- P wave / Atrial enlargement ------------------------------------------
    p_morphology_class: Optional[str]          # None | "normal" | "probable_rae" | "rae" | "probable_lae" | "lae" | "bae"
    rae_leads: List[str]                       # limb leads with P ≥ 0.24 mV
    lae_suspected: bool
    lae_definite: bool
    ptf_v1_class: Optional[str]               # None | "normal" | "probable_lae" | "definite_lae"

    # --- Q waves --------------------------------------------------------------
    pathological_q_leads: Dict[str, bool]      # per-lead flag
    q_wave_territories: List[str]              # ["inferior", "anterior", "lateral"]

    # --- R-wave progression ---------------------------------------------------
    r_progression_class: str                   # "normal" | "poor" | "reverse" | "indeterminate"
    r_s_transition_lead: Optional[str]         # first lead where R ≥ |S|

    # --- ST segment -----------------------------------------------------------
    st_elevation_leads: Dict[str, float]       # lead → J-point elevation (mV)
    st_depression_leads: Dict[str, float]      # lead → J-point depression (mV, negative)
    st_territories_elevated: List[str]         # e.g. ["inferior", "lateral"]
    st_territories_depressed: List[str]
    stemi_suspected_codes: List[str]           # DXL-style codes e.g. "IMIA", "AMIA"
    reciprocal_change_detected: bool
    reciprocal_pairs: List[List[str]]          # ["+elevated_leads", "-depressed_leads"]

    # --- LVH ------------------------------------------------------------------
    lvh_voltage_criteria: List[str]            # criteria names that exceeded threshold
    lvh_class: Optional[str]                   # None | "by_voltage" | "consider" | "probable" | "definite"

    # --- Low voltage ----------------------------------------------------------
    low_voltage_class: Optional[str]           # None | "frontal_borderline" | "frontal_definite" | "precordial_definite"

    # --- RVH ------------------------------------------------------------------
    rvh_suspected: bool

    # --- Lead reversal --------------------------------------------------------
    limb_reversal_suspected: Optional[str]     # description string or None
    precordial_reversal_suspected: bool

    # --- Tall T ---------------------------------------------------------------
    tall_t_leads: List[str]

    # --- Advanced rhythm detection (DXL Chapter 2) ----------------------------
    premature_complexes: List[str] = field(default_factory=list)
    bigeminy: Optional[str] = None
    trigeminy: bool = False
    non_sustained_vt: bool = False
    pauses_detected: bool = False
    pause_longest_ms: Optional[float] = None
    complete_av_block: bool = False
    av_dissociation: bool = False
    second_degree_avb: Optional[str] = None
    pacemaker_like_artifact: bool = False

    # --- Morphology extensions (DXL Chapter 3) --------------------------------
    dextrocardia_suspected: bool = False
    rvh_class: Optional[str] = None        # "consider" | "probable" | "definitive"
    copd_pattern: bool = False
    qtc_electrolyte_hint: Optional[str] = None  # "hypercalcemia" | "hypokalemia" | "hypocalcemia"
    posterior_mi_suspected: bool = False   # R-dominant V1-V3 pattern
    mi_evidence: Dict[str, object] = field(default_factory=dict)
    mi_statement_candidates: List[Dict[str, object]] = field(default_factory=list)
    st_rate_related: bool = False          # ST dep probably rate-related
    lvh_secondary_repol_abnormality: bool = False  # LVH + anterolateral LV strain
    extreme_tachycardia_critical: bool = False     # HR > 220-age bpm (critical value)
    digitalis_effect_suspected: bool = False       # short QTc + repol abnormality
    nonspecific_t_abnormality: bool = False        # QRS-T angle > 90 deg
    # True when qrs_width_class/bundle_branch_block were escalated from the
    # measurement-channel qrs_ms call using the looser qrs_wide_ms consensus,
    # because the two channels disagreed and morphology corroborated BBB.
    qrs_width_measurement_disagreement: bool = False

    # --- Pediatric extensions (DXL Chapter 4) ---------------------------------
    is_pediatric: bool = False             # True when algorithm routed through Chapter 4
    lsh_suspected: Optional[str] = None   # None | "consider_lsh" | "lsh"
    bvh_suspected: bool = False            # Biventricular Hypertrophy
    pediatric_hypertrophy_evidence: Dict[str, object] = field(default_factory=dict)
    pericarditis_suspected: bool = False   # Diffuse ST elevation all territories
    early_repolarization_suspected: bool = False  # Nonspecific ST ele, no T inversion


@dataclass
class ECGFeatures:
    fs: int
    quality: Dict[str, LeadQuality]
    beats: List[BeatAnnotation]
    beat_features: List[LeadBeatFeatures]
    representative_leads: Dict[str, RepresentativeLeadFeatures]
    groups: Dict[int, GroupFeatures]
    global_features: GlobalFeatures
    p_wave_assessments: List[PWaveBeatAssessment] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    interpretation: Optional[ECGInterpretation] = None
