from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict


@dataclass(frozen=True)
class SignalQualityThresholds:
    flatline_peak_to_peak_mv: float = 0.05
    flatline_fraction_max: float = 0.10
    saturation_fraction_max: float = 0.005
    baseline_drift_ratio_max: float = 0.30
    powerline_ratio_max: float = 0.15
    emg_ratio_max: float = 0.40
    psqi_min: float = 0.40
    psqi_max: float = 0.90
    ksqi_min: float = 5.0
    bsqi_min: float = 0.85
    bsqi_record_stop: float = 0.70
    minimum_usable_leads: int = 6
    full_coverage_leads: int = 10


@dataclass(frozen=True)
class IntervalThresholds:
    adult_bradycardia_bpm: float = 60.0
    adult_tachycardia_bpm: float = 100.0
    pr_short_ms: float = 120.0
    pr_prolonged_ms: float = 200.0
    qrs_borderline_ms: float = 110.0
    qrs_wide_ms: float = 120.0
    qtc_prolonged_male_ms: float = 450.0
    qtc_prolonged_female_ms: float = 460.0
    # Escalation tier between "prolonged" and the >500 ms critical alert.
    qtc_markedly_prolonged_ms: float = 480.0
    qtc_severe_ms: float = 500.0
    qtc_short_ms: float = 340.0
    qtc_possibly_short_ms: float = 360.0
    qtc_borderline_short_ms: float = 390.0
    p_duration_prolonged_ms: float = 120.0
    # Frontal-plane P axis accepted as a sinus origin.
    sinus_p_axis_min_deg: float = 0.0
    sinus_p_axis_max_deg: float = 75.0


@dataclass(frozen=True)
class RhythmThresholds:
    rr_irregular_cv: float = 0.15
    rr_regular_cv: float = 0.05
    p_absent_ratio: float = 0.20
    p_uniform_correlation: float = 0.85
    minimum_rhythm_duration_seconds: float = 8.0
    minimum_diagnostic_duration_seconds: float = 10.0
    minimum_detected_beats: int = 3
    flutter_rate_min_bpm: float = 240.0
    flutter_rate_max_bpm: float = 340.0
    flutter_review_rate_min_bpm: float = 140.0
    flutter_review_rate_max_bpm: float = 160.0
    # Minimum organized_p_ratio for a P-wave measurement to support a chamber
    # or AV-conduction statement. Lower than the 0.70 the sinus rule needs:
    # the question here is only "is this really a P wave", not "is it sinus".
    # Separated empirically on WFDBRecords/01/019, where flutter records ran
    # 0.05-0.16 and records with genuine P-wave findings ran 0.53-1.00.
    # Wider validation still outstanding.
    organized_p_ratio_min_for_morphology: float = 0.35


@dataclass(frozen=True)
class MorphologyThresholds:
    dominant_group_fraction_min: float = 0.30
    maximum_template_classes: int = 5
    template_correlation_min: float = 0.90
    template_merge_correlation_min: float = 0.95
    premature_rr_ratio: float = 0.85
    low_voltage_limb_mv: float = 0.50
    low_voltage_precordial_mv: float = 1.00
    # Precordial R-wave progression / rotation.
    poor_r_progression_v3_mv: float = 0.30
    normal_transition_leads: tuple = ("V3", "V4")
    # A transition lead needs a real R wave. Without a floor, a noise-level R
    # sitting next to an equally small S reads as an R/S crossover, which turns
    # a QS pattern in V1-V2 into a spurious counterclockwise rotation.
    #
    # The floor scales with the record's own precordial voltage so that
    # low-voltage tracings are not stripped of their real transition: it is
    # `transition_min_r_fraction` of the median precordial QRS peak-to-peak,
    # clamped into [transition_min_r_floor_mv, transition_min_r_mv].
    transition_min_r_mv: float = 0.10
    transition_min_r_floor_mv: float = 0.04
    transition_min_r_fraction: float = 0.08
    # LPFB axis floor. AHA/ACCF/HRS Part III; also the textbook value. The
    # previous +90 deg floor fired on ordinary vertical hearts.
    lpfb_axis_min_deg: float = 110.0
    lpfb_axis_max_deg: float = 180.0
    lafb_axis_min_deg: float = -90.0
    lafb_axis_max_deg: float = -45.0
    # Textbooks also recognise a <= -30 deg variant. Kept as a lower-confidence
    # tier rather than widening the definite window, because the -30 to -45 deg
    # band is where LVH, horizontal heart and prior inferior infarction cluster.
    lafb_axis_probable_max_deg: float = -30.0
    # Sokolow-Lyon limb-lead criterion R(I) + S(III).
    lvh_sokolow_limb_mv: float = 2.5
    # RVH precordial sum R(V1) + S(V5 or V6).
    rvh_precordial_sum_mv: float = 1.05
    # Right precordial strain: ST depression with T inversion in V1-V3.
    rvh_strain_st_mv: float = -0.05
    rvh_strain_t_mv: float = 0.0


@dataclass(frozen=True)
class IschemiaThresholds:
    st_other_leads_mv: float = 0.10
    st_v2_v3_male_40_plus_mv: float = 0.20
    st_v2_v3_male_under_40_mv: float = 0.25
    st_v2_v3_female_mv: float = 0.15
    st_depression_borderline_mv: float = -0.05
    st_depression_mv: float = -0.075
    t_inversion_mv: float = -0.10
    sgarbossa_smith_ratio: float = -0.25
    # Pathological Q wave. UDMI 2018 uses >=30 ms generally and >=20 ms in
    # V2-V3; the >=40 ms figure in the textbook literature is the pre-2018
    # convention. Raise pathological_q_duration_ms to 40.0 to switch.
    pathological_q_duration_ms: float = 30.0
    pathological_q_duration_v2_v3_ms: float = 20.0
    pathological_q_amplitude_mv: float = 0.10
    pathological_q_r_ratio: float = 0.25


@dataclass(frozen=True)
class DiagnosticConfig:
    """Single versioned source for public-rule thresholds.

    The signal extractor can continue to accept lower-rate research data.  The
    raw sampling-rate minimum is intentionally not part of this configuration,
    per project policy.
    """

    version: str = "2026.07.14"
    quality: SignalQualityThresholds = field(default_factory=SignalQualityThresholds)
    intervals: IntervalThresholds = field(default_factory=IntervalThresholds)
    rhythm: RhythmThresholds = field(default_factory=RhythmThresholds)
    morphology: MorphologyThresholds = field(default_factory=MorphologyThresholds)
    ischemia: IschemiaThresholds = field(default_factory=IschemiaThresholds)
    diagnostic_threshold_sources: Dict[str, str] = field(
        default_factory=lambda: {
            "quality": "docs/流程 sections 4 and 13.2",
            "intervals": "AHA/ACCF/HRS ECG standardization recommendations",
            "rhythm": "AHA/ACC/HRS rhythm guidance and docs/流程 section 10.6",
            "morphology": "AHA/ACCF/HRS chamber hypertrophy and docs/流程",
            "ischemia": "Fourth Universal Definition of Myocardial Infarction",
        }
    )
    # See clinical_rules.sources.BASELINE_STANDARD for which standards family
    # arbitrates these numbers, and BASELINE_DIVERGENCES for the conflicts
    # that were decided rather than left open.
    baseline_standard: str = (
        "AHA/ACCF/HRS ECG standardization (2009) + UDMI 4th edition (2018)"
    )

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_DIAGNOSTIC_CONFIG = DiagnosticConfig()
