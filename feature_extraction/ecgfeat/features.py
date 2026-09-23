from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .dispersion import QTDispersion, summarize_qt_dispersion
from .measurement_paths import build_qt_path_decision
from .models import (
    GlobalFeatures,
    GroupFeatures,
    LeadBeatFeatures,
    RepresentativeLeadFeatures,
    STANDARD_12_LEADS,
)
from .quality import detect_precordial_reversal
from .vector_axis import compute_t_axis_from_cluster


HEXAXIAL_ANGLES = {
    "I": 0.0,
    "II": 60.0,
    "III": 120.0,
    "aVR": -150.0,
    "aVL": -30.0,
    "aVF": 90.0,
}

_MEASUREMENT_LIMITS_MS = {
    "pr_ms": (60.0, 350.0),
    "pr_consensus_ms": (60.0, 350.0),
    "p_dur_ms": (35.0, 180.0),
    "p_dur_consensus_ms": (35.0, 180.0),
    "pr_segment_consensus_ms": (0.0, 220.0),
    "qrs_ms": (40.0, 220.0),
    "qrs_consensus_ms": (40.0, 220.0),
    "qt_ms": (240.0, 700.0),
    "qt_consensus_ms": (240.0, 700.0),
    "jt_ms": (120.0, 650.0),
}

_P_MIN_ABS_AMP_MV = 0.02
_P_MIN_CONFIDENCE = 0.20
_P_AXIS_CONTEXT_STRONG = 0.55
_P_AXIS_CONTEXT_WEAK = 0.35
_P_AXIS_SEED_MIN_CONFIDENCE = 0.55
_P_AXIS_SEED_MIN_ABS_AREA = 0.30
_P_AXIS_CONFLICT_PROJECTION_NEUTRAL = 0.15
_P_AXIS_MIN_BOUNDARY_CONFIDENCE = 0.20
_P_AXIS_CONFLICT_MIN_BOUNDARY_CONFIDENCE = 0.45
_P_AXIS_ROBUST_MIN_RETAINED_LEADS = 3
_P_AXIS_ROBUST_PROJECTION_NEUTRAL = 0.20
_P_AXIS_ROBUST_DOMINANT_RATIO = 2.8
_P_AXIS_ROBUST_MIN_SCORE_IMPROVEMENT = 0.30
_P_AXIS_ROBUST_MAX_FINAL_SCORE = 0.95
_P_AXIS_INVERSION_MIN_PR_SUPPORT = 3
_P_AXIS_INVERSION_MIN_PR_MS = 120.0
_P_AXIS_INVERSION_MAX_PR_MS = 230.0
_P_AXIS_INVERSION_MAX_PR_SPREAD_MS = 60.0
_P_AXIS_INVERSION_EXTREME_NEG_DEG = -75.0
_P_AXIS_INVERSION_EXTREME_POS_DEG = 150.0
_P_AXIS_MIN_FRONTAL_LEADS = 2
_P_AXIS_MIN_FRONTAL_LEADS_WITHOUT_PR_SUPPORT = 3
_T_MIN_AXIS_ABS_AMP_MV = 0.05
_T_AXIS_RELAXED_MIN_ABS_AMP_MV = 0.02
_T_AXIS_RELAXED_MIN_CONFIDENCE = 0.30
_T_AXIS_RELAXED_MIN_LEADS = 4
_T_MIN_QT_ABS_AMP_MV = 0.05
_T_MIN_CONFIDENCE = 0.45
_T_ST_AXIS_MIN_ABS_AMP_MV = 0.02
_T_ST_AXIS_MIN_LEADS = 2
_T_ST_AXIS_MIN_SHIFT_DEG = 30.0
_T_ST_AXIS_MAX_POLARITY_DISAGREEMENTS = 1
_T_ST_AXIS_MIN_QRS_MS = 145.0
_T_AXIS_MIN_SIGNED_AREA_RATIO = 0.15
_T_AXIS_MAX_AMP_SD_MV = 0.08
_T_AXIS_MAX_AMP_SD_RATIO = 0.80
_T_AXIS_MIN_LEADS = 2
_T_AXIS_WIDE_MIN_LEADS = 3
_T_AXIS_WIDE_HIGH_CONFIDENCE = 0.60
_T_AXIS_WIDE_QRS_SUPPRESSION_MS = 145.0
_T_AXIS_WIDE_QRS_T_ANGLE_SUPPRESS_DEG = 125.0
_T_AXIS_WIDE_EXTREME_SUPPRESS_DEG = 150.0
_T_AXIS_POST_ST_MIN_ABS_MV = 0.03
_T_AXIS_POST_ST_MIN_AMP_RATIO = 0.30
_ATRIAL_IRREGULAR_RR_CV = 0.15
_T_AXIS_IRREGULAR_RR_CV = 0.18
_T_AXIS_IRREGULAR_MAX_VISIBLE_P_LEADS = 2
_QRS_AXIS_MIN_SHIFT_DEG = 20.0
_QRS_AXIS_MIN_POLARITY_COMPARISONS = 4
_QRS_AXIS_MAX_POLARITY_DISAGREEMENTS = 1
_QT_CLUSTER_OUTLIER_MS = 80.0
_WIDE_QRS_MS = 120.0
_WIDE_QRS_MIN_JT_MS = 160.0
_WIDE_QRS_MAX_JT_MS = 550.0
_QRS_CONSENSUS_MAX_BEYOND_RAW_MS = 12.0
_QRS_CONSENSUS_STRONG_WIDE_MS = 145.0
_QRS_CONSENSUS_MODERATE_WIDE_MAX_MS = 190.0
_QRS_CONSENSUS_MIN_SUPPORT_LEADS = 6
_QRS_CONSENSUS_MIN_RAW_WIDE_SUPPORT = 2
_QRS_CONSENSUS_RAW_CORE_MIN_DELTA_MS = 35.0
_QRS_CONSENSUS_RAW_CORE_MIN_SUPPORT = 5
_QRS_CONSENSUS_RAW_CORE_CLUSTER_MS = 30.0
_QRS_CONSENSUS_RAW_CORE_PERCENTILE = 60.0
_QRS_SPLIT_MIN_LIMB_LEADS = 4
_QRS_SPLIT_MIN_PRECORDIAL_LEADS = 3
_QRS_SPLIT_MIN_GAP_MS = 70.0
_QRS_SPLIT_RAW_PERCENTILE = 60.0
_SPLIT_QT_RESCUE_MIN_DELTA_MS = 35.0
_SPLIT_QT_RESCUE_SOFT_MIN_DELTA_MS = 15.0
_SPLIT_QT_RESCUE_MIN_LEADS = 2
_SPLIT_QT_RESCUE_PERCENTILE = 25.0
_WIDE_QRS_RAW_QT_RESCUE_MIN_DELTA_MS = 15.0
_WIDE_QRS_RAW_QT_RESCUE_MAX_SPREAD_MS = 40.0
_WIDE_QRS_LONG_QT_RAW_CORE_MIN_QT_MS = 500.0
_WIDE_QRS_LONG_QT_RAW_CORE_MIN_DELTA_MS = 25.0
_WIDE_QRS_LONG_QT_RAW_CORE_CLUSTER_MS = 45.0
_WIDE_QRS_LONG_QT_RAW_CORE_MIN_SUPPORT = 4
_WIDE_QRS_LONG_QT_RAW_CORE_MIN_ABS_AMP_MV = 0.03
_WIDE_QRS_LONG_QT_RAW_CORE_MIN_CONFIDENCE = 0.25
_AF_QT_RESCUE_RR_CV = 0.12
_RAW_QT_CORE_CLUSTER_MS = 45.0
_RAW_QT_CORE_MIN_LEADS = 3
_RAW_QT_CORE_REGULAR_MIN_LEADS = 5
_RAW_QT_CORE_IRREGULAR_MIN_ABS_AMP_MV = 0.03
_RAW_QT_CORE_MIN_CONFIDENCE = 0.20
_RAW_QT_CORE_MIN_DELTA_MS = 25.0
_RAW_QT_CORE_ACTIVATION_DELTA_MS = 35.0
_LATE_RAW_QT_CLUSTER_MIN_DELTA_MS = 20.0
_LATE_RAW_QT_CLUSTER_MS = 35.0
_LATE_RAW_QT_CLUSTER_MIN_SUPPORT = 4
_LATE_RAW_QT_CLUSTER_MIN_FRACTION = 0.50
_LOW_SUPPORT_LATE_RAW_QT_MIN_DELTA_MS = 35.0
_LOW_SUPPORT_LATE_RAW_QT_CLUSTER_MS = 35.0
_LOW_SUPPORT_LATE_RAW_QT_MIN_SUPPORT = 4
_LOW_SUPPORT_LATE_RAW_QT_MIN_CONFIDENCE = 0.10
_TAIL_CONFIRMED_LATE_QT_MIN_DELTA_MS = 45.0
_TAIL_CONFIRMED_LATE_QT_CLUSTER_MIN_DELTA_MS = 40.0
_TAIL_CONFIRMED_LATE_QT_CLUSTER_MS = 35.0
_TAIL_CONFIRMED_LATE_QT_MIN_SUPPORT = 2
_TAIL_CONFIRMED_LATE_QT_MIN_CONFIDENCE = 0.25
_TAIL_CONFIRMED_LATE_QT_MIN_ABS_AMP_MV = 0.02
_TAIL_CONFIRMED_LATE_QT_PAIR_MAX_MS = 480.0
_TAIL_CONFIRMED_LATE_QT_SMALL_CLUSTER_MAX_MS = 500.0
_TAIL_CONFIRMED_LATE_QT_WEAK_PRECORDIAL_PAIR_MIN_QT_MS = 450.0
_TAIL_CONFIRMED_LATE_QT_PRECORDIAL_PAIR_MIN_ABS_AMP_MV = 0.04
_PACED_LOW_SUPPORT_RAW_QT_MIN_CONFIDENCE = 0.0
_PACED_LOW_SUPPORT_RAW_QT_CLUSTER_MS = 90.0
_PHYSIOLOGIC_PR_CORE_MIN_MS = 120.0
_PHYSIOLOGIC_PR_CORE_MAX_MS = 220.0
_PHYSIOLOGIC_PR_CORE_MIN_CONFIDENCE = 0.5
_PHYSIOLOGIC_PR_CORE_UPPER_PERCENTILE = 78.0
_PHYSIOLOGIC_PR_CORE_SPLIT_GAP_MS = 18.0
_PHYSIOLOGIC_PR_CORE_SPLIT_GAP_RATIO = 2.0
_PHYSIOLOGIC_PR_CORE_SPLIT_MIN_LOWER_BEATS = 3
_PHYSIOLOGIC_PR_CORE_SPLIT_MIN_UPPER_BEATS = 2
_PHYSIOLOGIC_PR_CORE_LOWER_CLUSTER_PERCENTILE = 75.0
_PR_CONSENSUS_MIN_MS = 120.0
_PR_CONSENSUS_MAX_MS = 300.0
_PR_CONSENSUS_MIN_SUPPORT = 3
_PR_CONSENSUS_MAX_ONSET_SPREAD_MS = 200.0
_PR_CONSENSUS_RAW_SPLIT_MS = 60.0
_PR_CONSENSUS_MIN_RAW_MEDIAN_DELTA_MS = 25.0
_PR_CONSENSUS_CLUSTER_MAX_SPREAD_MS = 45.0
_PR_CONSENSUS_CLUSTER_MIN_LIMB_SUPPORT = 3
_PR_CONSENSUS_CLUSTER_MIN_RAW_MEDIAN_DELTA_MS = 5.0
_PR_CONSENSUS_SHORT_OUTLIER_GAP_MS = 50.0
_PR_CONSENSUS_CLUSTER_RAW_SUPPORT_MS = 35.0
_PR_CONSENSUS_CLUSTER_MIN_RAW_SUPPORT = 2
_PR_CONSENSUS_LONG_RAW_RESCUE_MIN_DELTA_MS = 25.0
_T_AXIS_STABLE_LIMB_MAX_RR_CV = 0.08
_T_AXIS_STABLE_LIMB_MAX_QRS_MS = 120.0
_T_AXIS_STABLE_LIMB_MIN_LEADS = 5
_T_AXIS_STABLE_LIMB_MIN_SIGNED_AREA = 0.50
_T_AXIS_HARD_EXCLUSION_REASONS = {"polarity_conflict_or_st_t_confusion"}
_T_AXIS_T_PRIME_CONFLICT_MIN_AREA_UV_MS = 160.0
_T_AXIS_SPECIAL_T_CONFLICT_MIN_UV = 50.0
_T_AXIS_LIMB_SEED_MIN_CONFIDENCE = 0.60
_T_AXIS_LIMB_SEED_MIN_ABS_MV = 0.05
_T_AXIS_LIMB_SEED_MIN_AREA = 0.30
_T_AXIS_LIMB_SEED_PROJECTION_NEUTRAL = 0.15
_T_AXIS_ST_CONFIDENCE_MIN = 0.50
_T_AXIS_LOW_SUPPORT_COVERAGE_MIN_LEADS = 4
_T_AXIS_LOW_SUPPORT_COVERAGE_MIN_CONFIDENCE = 0.10
_T_AXIS_LOW_SUPPORT_COVERAGE_MIN_AREA = 0.50
_T_AXIS_LOW_SUPPORT_COVERAGE_MIN_ABS_MV = 0.05
_T_AXIS_LOW_SUPPORT_COVERAGE_MAX_ABS_AXIS_DEG = 120.0
_T_AXIS_LOW_SUPPORT_ST_CONFIDENCE_MIN = 0.80
_T_AXIS_LOW_SUPPORT_POST_ST_MIN_ABS_MV = 0.08
_T_AXIS_DETECTED_COVERAGE_MIN_LEADS = 3
_T_AXIS_DETECTED_COVERAGE_MIN_CONFIDENCE = 0.20
_T_AXIS_DETECTED_COVERAGE_MIN_AREA = 0.75
_T_AXIS_DETECTED_COVERAGE_MIN_ABS_MV = 0.08
_T_AXIS_DETECTED_COVERAGE_MAX_ABS_AXIS_DEG = 120.0
_T_AXIS_DETECTED_COVERAGE_ST_CONFIDENCE_MIN = 0.85
_T_AXIS_DETECTED_COVERAGE_POST_ST_MIN_ABS_MV = 0.08
_T_AXIS_ST_SUPPORTED_COVERAGE_MIN_LEADS = 2
_T_AXIS_ST_SUPPORTED_COVERAGE_MIN_CONFIDENCE = 0.60
_T_AXIS_ST_SUPPORTED_COVERAGE_MIN_AREA = 1.00
_T_AXIS_ST_SUPPORTED_COVERAGE_MIN_ABS_MV = 0.05
_T_AXIS_ST_SUPPORTED_COVERAGE_MIN_SIGNED_RATIO = 0.50
_T_AXIS_ST_SUPPORTED_COVERAGE_ST_CONFIDENCE_MIN = 0.90
_T_AXIS_ST_SUPPORTED_COVERAGE_MAX_ABS_AXIS_DEG = 130.0
_T_REP_TEMPLATE_TINY_AMP_MV = 0.020
_T_REP_POOLED_MIN_AMP_MV = 0.020
_T_REP_POOLED_MIN_QT_CONFIDENCE = 0.50
_T_REP_POOLED_MIN_SUPPORT = 2
_REPRESENTATIVE_QT_CORE_BACKFILL_MIN_DELTA_MS = 60.0


def _valid_measurement(attr: str, value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    value = float(value)
    limits = _MEASUREMENT_LIMITS_MS.get(attr)
    if limits is None:
        return value
    lo, hi = limits
    return value if lo <= value <= hi else None


def _nanmedian(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not vals:
        return None
    return float(np.median(vals))


def _nanmean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not vals:
        return None
    return float(np.mean(vals))


def estimate_pr_segment_ms(
    beat_features: List[LeadBeatFeatures],
    fs: int,
) -> Optional[float]:
    """Estimate PR-segment duration as QRS onset minus P offset."""
    if fs <= 0:
        return None
    values: List[float] = []
    for bf in beat_features:
        if bf.qrs.onset is None or bf.p.offset is None:
            continue
        pr_segment_ms = float(bf.qrs.onset - bf.p.offset) * 1000.0 / float(fs)
        if 0.0 <= pr_segment_ms <= 200.0:
            values.append(pr_segment_ms)
    if not values:
        return None
    return float(np.median(values))


def estimate_initial_qrs_axis_deg(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Optional[float]:
    """Estimate a coarse initial/frontal QRS axis from limb-lead QRS vectors."""
    try:
        values = _qrs_net_amplitude(representative_leads)
        usable = [
            lead for lead in HEXAXIAL_ANGLES
            if values.get(lead) is not None and np.isfinite(float(values[lead]))
        ]
    except (TypeError, ValueError):
        return None
    if len(usable) < 2:
        return None
    confidences = {
        lead: rep.params.get("qrs_confidence_mean")
        for lead, rep in representative_leads.items()
    }
    try:
        return _axis_from_amplitudes(values, confidences)
    except (TypeError, ValueError):
        return None


def _mode_str(xs: Iterable[Optional[str]]) -> Optional[str]:
    """Return the most common non-None string value, or None if empty."""
    from collections import Counter
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def _mode_int(xs: Iterable[Optional[int]]) -> Optional[int]:
    from collections import Counter
    vals = [int(x) for x in xs if x is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def _rep_and_pooled_values(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
    attr: str,
):
    if rep_feature is not None:
        yield getattr(rep_feature, attr)
    for item in pooled_items:
        yield getattr(item, attr)


def _pick_representative_value(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
    attr: str,
) -> Optional[float]:
    rep_val = _valid_measurement(attr, getattr(rep_feature, attr)) if rep_feature is not None else None
    if rep_val is not None:
        return rep_val
    return _nanmedian(_valid_measurement(attr, getattr(i, attr)) for i in pooled_items)


_P_REP_POLARITY_MIN_SUPPORT = 3      # beats needed before a polarity is "dominant"
_P_REP_POLARITY_MIN_FRACTION = 0.60  # ...and the share of valid beats it must hold


def _p_polarity_consensus_value(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
    attr: str,
    *,
    polarity_attr: str = "p_amp_mv",
) -> Optional[float]:
    """Record-level value for a P-wave measurement, decided across beats.

    `_pick_representative_value` reports whatever the single representative beat
    measured.  The representative beat is chosen on QRS grounds, and the P wave is
    small enough that a single beat's P can be mis-located while the beat majority
    is right: measured across 120 PTB-XL records, the representative `p_amp_mv`
    carried the *opposite sign* to the median of its own lead's per-beat values on
    23-37% of records (aVL worst at 36.7%) and fell outside the entire per-beat
    range on 10-23%, by a median 12-25 µV -- as large as the P wave itself.  That
    is what put a positive P in aVR on 19.5% of sinus records and a negative P in
    lead II on 16.5% of them, neither of which a sinus vector can produce, and it
    reached every amplitude-based atrial criterion (RAE 0.24/0.25 mV, the LAE
    0.10 mV condition, BAE 0.30 mV) plus the P axis.

    So the estimator is a cross-beat one: take the beats whose P shares the
    dominant polarity, require that group to be a real majority, and return its
    median.  Without a majority, fall back to the median of all valid beats --
    still never a single beat.  This mirrors what this module already does for
    the T wave in `_t_consensus_items_for_representative`.
    """
    paired: List[Tuple[float, float]] = []
    for item in pooled_items:
        value = _valid_measurement(attr, getattr(item, attr, None))
        polarity = _valid_measurement(polarity_attr, getattr(item, polarity_attr, None))
        if value is None or polarity is None or polarity == 0.0:
            continue
        paired.append((float(value), float(polarity)))
    if not paired:
        return _pick_representative_value(rep_feature, pooled_items, attr)

    positive = [v for v, pol in paired if pol > 0.0]
    negative = [v for v, pol in paired if pol < 0.0]
    dominant = positive if len(positive) >= len(negative) else negative
    if (
        len(dominant) >= _P_REP_POLARITY_MIN_SUPPORT
        and len(dominant) >= _P_REP_POLARITY_MIN_FRACTION * len(paired)
    ):
        return float(np.median(dominant))
    return float(np.median([v for v, _ in paired]))


def _p_amplitude_if_measurable(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
) -> Optional[float]:
    """Record-level P amplitude, withheld when it is not a measurement.

    A P wave no larger than the residual baseline drift across its own search
    window carries no reliable amplitude and no reliable *sign*: measured across
    500 sinus PTB-XL records, records whose aVR P polarity is physiologically
    impossible have drift/|P| at a median 0.99 (50% above 1.0) against 0.38 (6%)
    for the correct ones, while the absolute drift does not separate them at all
    (AUC 0.51).  So the criterion is the ratio, and the threshold is not tuned:
    the P has to exceed the drift.

    Withholding costs the 6% of correct records that sit in the same regime -- on
    those the sign came out right by luck, which is exactly the accident
    `_p_polarity_consensus_value` documents on the beat-selection side.
    """
    amp = _p_polarity_consensus_value(rep_feature, pooled_items, "p_amp_mv")
    if amp is None:
        return None
    drifts = [
        float(value)
        for item in pooled_items
        if (value := getattr(item, "p_window_drift_mv", None)) is not None
        and np.isfinite(float(value))
    ]
    if not drifts:
        return amp
    if float(np.median(drifts)) >= abs(float(amp)):
        return None
    return amp


def _t_consensus_items_for_representative(
    pooled_items: List[LeadBeatFeatures],
) -> List[LeadBeatFeatures]:
    candidates: List[LeadBeatFeatures] = []
    for item in pooled_items:
        if bool(getattr(item, "st_t_confusion", False)):
            continue
        amp = _valid_measurement("t_amp_mv", getattr(item, "t_amp_mv", None))
        if amp is None or abs(float(amp)) < _T_REP_POOLED_MIN_AMP_MV:
            continue
        confidence = getattr(item, "qt_confidence", None)
        if confidence is None or not np.isfinite(float(confidence)):
            continue
        if float(confidence) < _T_REP_POOLED_MIN_QT_CONFIDENCE:
            continue
        candidates.append(item)

    positive = [
        item for item in candidates
        if float(getattr(item, "t_amp_mv")) > 0.0
    ]
    negative = [
        item for item in candidates
        if float(getattr(item, "t_amp_mv")) < 0.0
    ]
    dominant = positive if len(positive) >= len(negative) else negative
    if len(dominant) < _T_REP_POOLED_MIN_SUPPORT:
        return []
    if len(dominant) < int(np.ceil(0.60 * max(len(candidates), 1))):
        return []
    return dominant


def _pick_representative_t_value(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
    attr: str,
) -> Optional[float]:
    rep_val = _valid_measurement(attr, getattr(rep_feature, attr)) if rep_feature is not None else None
    rep_amp = (
        _valid_measurement("t_amp_mv", getattr(rep_feature, "t_amp_mv"))
        if rep_feature is not None
        else None
    )
    if rep_amp is not None and abs(float(rep_amp)) < _T_REP_TEMPLATE_TINY_AMP_MV:
        consensus_items = _t_consensus_items_for_representative(pooled_items)
        if consensus_items:
            consensus = _nanmedian(_valid_measurement(attr, getattr(item, attr)) for item in consensus_items)
            if consensus is not None:
                return consensus
    if rep_val is not None:
        return rep_val
    return _nanmedian(_valid_measurement(attr, getattr(i, attr)) for i in pooled_items)


def _pick_pooled_interval_value(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
    attr: str,
) -> Optional[float]:
    pooled = _nanmedian(_valid_measurement(attr, getattr(i, attr)) for i in pooled_items)
    if pooled is not None:
        return pooled
    return _valid_measurement(attr, getattr(rep_feature, attr)) if rep_feature is not None else None


def _has_p_fine_measurement(item: LeadBeatFeatures) -> bool:
    bounds = item.p
    if bounds.onset is not None and bounds.peak is not None and bounds.offset is not None:
        return True
    if bool(item.p_notched) or bool(item.p_biphasic):
        return True
    numeric_attrs = (
        "p_dur_ms",
        "p_dur_consensus_ms",
        "p_area",
        "p_notch_interval_ms",
        "p_initial_duration_ms",
        "p_initial_amp_mv",
        "p_terminal_duration_ms",
        "p_terminal_amp_mv",
        "p_terminal_area_mv_ms",
    )
    return any(_valid_measurement(attr, getattr(item, attr)) is not None for attr in numeric_attrs)


def _u_wave_params(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
) -> Dict[str, object]:
    """Aggregate the signed U-wave measurement across a lead's beats.

    Pooling the signed amplitude directly would cancel a genuine polarity
    disagreement into a near-zero mean, so the majority polarity is resolved
    first and only the beats that agree with it contribute to the amplitude.
    A lead whose beats disagree about polarity is reported as unreliable
    rather than as a small U wave -- inverted-U calls are high-specificity and
    must not rest on an averaging artefact.
    """
    candidates = [
        item
        for item in (
            [rep_feature] if rep_feature is not None else []
        ) + list(pooled_items)
        if bool(getattr(item, "u_measurement_reliable", False))
        and getattr(item, "u_amp_signed_mv", None) is not None
    ]
    total = 1 if rep_feature is not None else 0
    total += len(pooled_items)
    if not candidates:
        return {
            "u_amp_signed_mv": None,
            "u_polarity": 0,
            "u_dur_ms": None,
            "u_isoelectric_gap_ms": None,
            "u_prominence_mv": None,
            "u_measurement_reliable": False,
            "u_beats_measured": 0,
            "u_beats_total": int(total),
            "u_polarity_agreement": None,
        }
    polarities = [int(getattr(item, "u_polarity", 0) or 0) for item in candidates]
    positive = sum(1 for value in polarities if value > 0)
    negative = sum(1 for value in polarities if value < 0)
    if positive == negative:
        # A tie carries no majority; refuse rather than pick arbitrarily.
        return {
            "u_amp_signed_mv": None,
            "u_polarity": 0,
            "u_dur_ms": None,
            "u_isoelectric_gap_ms": None,
            "u_prominence_mv": None,
            "u_measurement_reliable": False,
            "u_beats_measured": len(candidates),
            "u_beats_total": int(total),
            "u_polarity_agreement": 0.5,
        }
    majority = 1 if positive > negative else -1
    agreeing = [
        item
        for item in candidates
        if int(getattr(item, "u_polarity", 0) or 0) == majority
    ]
    agreement = len(agreeing) / float(len(candidates))
    return {
        "u_amp_signed_mv": _nanmedian(
            float(getattr(item, "u_amp_signed_mv")) for item in agreeing
        ),
        "u_polarity": majority,
        "u_dur_ms": _nanmedian(
            getattr(item, "u_dur_ms", None) for item in agreeing
        ),
        "u_isoelectric_gap_ms": _nanmedian(
            getattr(item, "u_isoelectric_gap_ms", None) for item in agreeing
        ),
        "u_prominence_mv": _nanmedian(
            getattr(item, "u_prominence_mv", None) for item in agreeing
        ),
        # One beat can be noise; two agreeing beats is the minimum a
        # morphology claim rests on.
        "u_measurement_reliable": bool(len(agreeing) >= 2 or total <= 1),
        "u_beats_measured": len(candidates),
        "u_beats_total": int(total),
        "u_polarity_agreement": float(agreement),
    }


def _pick_representative_bool(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
    attr: str,
) -> Optional[bool]:
    measured_items = [item for item in pooled_items if _has_p_fine_measurement(item)]
    rep_measured = rep_feature is not None and _has_p_fine_measurement(rep_feature)
    if not rep_measured and not measured_items:
        return None
    return bool(
        (rep_feature is not None and bool(getattr(rep_feature, attr)))
        or any(bool(getattr(item, attr)) for item in measured_items)
    )


def _pick_representative_consensus_bool(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
    attr: str,
) -> bool:
    """Use the representative beat's consensus verdict, or a beat majority."""

    if rep_feature is not None:
        return bool(getattr(rep_feature, attr, False))
    values = [bool(getattr(item, attr, False)) for item in pooled_items]
    return bool(values and sum(values) >= (len(values) + 1) // 2)


def _st_j_is_guarded(item: LeadBeatFeatures) -> bool:
    return "st_j_unreliable" in item.flags


def _valid_st_j_value(item: LeadBeatFeatures) -> Optional[float]:
    if _st_j_is_guarded(item):
        return None
    return _valid_measurement("st_on_mv", item.st_on_mv)


def _representative_st_j_params(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
) -> Dict[str, object]:
    guarded_count = sum(1 for item in pooled_items if _st_j_is_guarded(item))
    pooled_value = _nanmedian(_valid_st_j_value(item) for item in pooled_items)
    if pooled_items and pooled_value is None and guarded_count:
        return {
            "st_on_mv": None,
            "st_j_source": "qrs_tail_guard",
            "st_j_reliable": False,
            "st_j_unreliable": True,
            "st_j_unreliable_reason": "qrs_tail_guard",
        }

    rep_guarded = rep_feature is not None and _st_j_is_guarded(rep_feature)
    if rep_guarded:
        guarded_count += 1

    rep_value = _valid_st_j_value(rep_feature) if rep_feature is not None else None
    value = rep_value if rep_value is not None else pooled_value

    if value is not None:
        return {
            "st_on_mv": value,
            "st_j_source": "mixed_reliable_beats" if guarded_count else "qrs_offset",
            "st_j_reliable": True,
            "st_j_unreliable": False,
            "st_j_unreliable_reason": None,
        }

    if guarded_count:
        return {
            "st_on_mv": None,
            "st_j_source": "qrs_tail_guard",
            "st_j_reliable": False,
            "st_j_unreliable": True,
            "st_j_unreliable_reason": "qrs_tail_guard",
        }

    return {
        "st_on_mv": None,
        "st_j_source": "unavailable",
        "st_j_reliable": False,
        "st_j_unreliable": True,
        "st_j_unreliable_reason": None,
    }


def _representative_hybrid_st_params(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
) -> Dict[str, object]:
    """Aggregate the robust profile across beats instead of trusting one medoid."""

    reliable_items = [
        item
        for item in pooled_items
        if bool(getattr(item, "st_hybrid_reliable", False))
    ]
    source = "pooled_reliable_beats"
    if not reliable_items and rep_feature is not None and bool(
        getattr(rep_feature, "st_hybrid_reliable", False)
    ):
        reliable_items = [rep_feature]
        source = "representative_template"

    all_items = list(pooled_items)
    if rep_feature is not None:
        all_items.append(rep_feature)
    if not reliable_items:
        return {
            "st_hybrid_j_mv": None,
            "st_hybrid_20ms_mv": None,
            "st_hybrid_40ms_mv": None,
            "st_hybrid_60ms_mv": None,
            "st_hybrid_80ms_mv": None,
            "st_hybrid_adaptive_mv": None,
            "st_hybrid_mean_mv": None,
            "st_hybrid_area_mv_ms": None,
            "st_hybrid_slope_mv_per_ms": None,
            "st_hybrid_curvature_mv_per_ms2": None,
            "st_hybrid_trend": "unknown",
            "st_hybrid_shape": "unknown",
            "st_pattern_class": "unknown",
            "st_hybrid_baseline_mv": None,
            "st_hybrid_baseline_source": _mode_str(
                getattr(item, "st_hybrid_baseline_source", None) for item in all_items
            ),
            "st_hybrid_baseline_confidence": _nanmedian(
                getattr(item, "st_hybrid_baseline_confidence", None) for item in all_items
            ),
            "st_hybrid_j_method": _mode_str(
                getattr(item, "st_hybrid_j_method", None) for item in all_items
            ),
            "st_hybrid_j_confidence": _nanmedian(
                getattr(item, "st_hybrid_j_confidence", None) for item in all_items
            ),
            "st_hybrid_consensus_support": 0,
            "st_hybrid_beat_support": 0,
            "st_hybrid_source": "unavailable",
            "st_hybrid_reliable": False,
            "st_hybrid_unreliable_reason": _mode_str(
                getattr(item, "st_hybrid_unreliable_reason", None) for item in all_items
            ),
        }

    def median(attr: str) -> Optional[float]:
        return _nanmedian(getattr(item, attr, None) for item in reliable_items)

    return {
        "st_hybrid_j_mv": median("st_hybrid_j_mv"),
        "st_hybrid_20ms_mv": median("st_hybrid_20ms_mv"),
        "st_hybrid_40ms_mv": median("st_hybrid_40ms_mv"),
        "st_hybrid_60ms_mv": median("st_hybrid_60ms_mv"),
        "st_hybrid_80ms_mv": median("st_hybrid_80ms_mv"),
        "st_hybrid_adaptive_mv": median("st_hybrid_adaptive_mv"),
        "st_hybrid_mean_mv": median("st_hybrid_mean_mv"),
        "st_hybrid_area_mv_ms": median("st_hybrid_area_mv_ms"),
        "st_hybrid_slope_mv_per_ms": median("st_hybrid_slope_mv_per_ms"),
        "st_hybrid_curvature_mv_per_ms2": median(
            "st_hybrid_curvature_mv_per_ms2"
        ),
        "st_hybrid_trend": _mode_str(
            getattr(item, "st_hybrid_trend", None) for item in reliable_items
        ) or "unknown",
        "st_hybrid_shape": _mode_str(
            getattr(item, "st_hybrid_shape", None) for item in reliable_items
        ) or "unknown",
        "st_pattern_class": _mode_str(
            getattr(item, "st_pattern_class", None) for item in reliable_items
        ) or "unknown",
        "st_hybrid_baseline_mv": median("st_hybrid_baseline_mv"),
        "st_hybrid_baseline_source": _mode_str(
            getattr(item, "st_hybrid_baseline_source", None) for item in reliable_items
        ),
        "st_hybrid_baseline_confidence": median("st_hybrid_baseline_confidence"),
        "st_hybrid_j_method": _mode_str(
            getattr(item, "st_hybrid_j_method", None) for item in reliable_items
        ),
        "st_hybrid_j_confidence": median("st_hybrid_j_confidence"),
        "st_hybrid_consensus_support": int(
            round(median("st_hybrid_consensus_support") or 0.0)
        ),
        "st_hybrid_beat_support": len(reliable_items),
        "st_hybrid_source": source,
        "st_hybrid_reliable": True,
        "st_hybrid_unreliable_reason": None,
    }


def _cap_qrs_consensus_to_raw_distribution(
    reps: Dict[str, RepresentativeLeadFeatures],
) -> None:
    raw_vals: List[float] = []
    for rep in reps.values():
        if not rep.params.get("reliable_for_qrs"):
            continue
        raw = _valid_measurement("qrs_ms", rep.params.get("qrs_ms"))
        if raw is not None:
            raw_vals.append(raw)
    if len(raw_vals) < 3:
        return

    raw_max = float(np.max(raw_vals))
    for rep in reps.values():
        for key in ("qrs_consensus_ms", "qrs_wide_ms"):
            consensus = _valid_measurement("qrs_consensus_ms", rep.params.get(key))
            if consensus is not None and consensus > raw_max + _QRS_CONSENSUS_MAX_BEYOND_RAW_MS:
                rep.params[key] = raw_max


_GLASGOW_PROFILE_METADATA_KEYS = {
    "measurement_source", "representative_beat_id",
    "unavailable_reasons", "provenance",
}
_GLASGOW_PROFILE_DISCRETE_KEYS = {"t_morphology", "r_wave_notch_count", "beat_window_start_index"}


def _glasgow_field_provenance(key: str, profiles: List[Dict[str, object]]) -> Optional[str]:
    fidelities = [
        profile.get("provenance", {}).get(key)
        for profile in profiles
        if isinstance(profile.get("provenance"), dict)
    ]
    return next((str(value) for value in fidelities if value), None)


def _glasgow_field_unavailable_reason(key: str, profiles: List[Dict[str, object]]) -> Optional[str]:
    reasons = [
        profile.get("unavailable_reasons", {}).get(key)
        for profile in profiles
        if isinstance(profile.get("unavailable_reasons"), dict)
    ]
    return next((str(reason) for reason in reasons if reason), None)


def _glasgow_field_median(key: str, profiles: List[Dict[str, object]]) -> Optional[float]:
    values = [profile.get(key) for profile in profiles]
    numeric = [
        float(value)
        for value in values
        if not isinstance(value, bool)
        and value is not None
        and np.isfinite(float(value))
    ]
    if not numeric:
        return None
    return _mode_int(numeric) if key in _GLASGOW_PROFILE_DISCRETE_KEYS else float(np.median(numeric))


def _representative_glasgow_measurements(
    rep_feature: Optional[LeadBeatFeatures],
    pooled_items: List[LeadBeatFeatures],
) -> Dict[str, object]:
    profiles = [
        item.glasgow_measurements
        for item in pooled_items
        if isinstance(item.glasgow_measurements, dict) and item.glasgow_measurements
    ]

    if rep_feature is not None and rep_feature.glasgow_measurements:
        profile = dict(rep_feature.glasgow_measurements)
        profile["measurement_source"] = "dominant_representative_beat"
        profile["representative_beat_id"] = int(rep_feature.beat_id)

        # A field the representative beat itself couldn't measure falls back to
        # the cross-beat median/mode, mirroring _pick_representative_value's
        # per-field fallback in the main measurement pipeline. Without this, a
        # field missing only on this one beat (e.g. no R' on the chosen beat
        # while other beats show one) reads as permanently unavailable here
        # even though the rest of the interpretation pipeline used the
        # cross-beat value just fine.
        unavailable = dict(profile.get("unavailable_reasons") or {})
        provenance = dict(profile.get("provenance") or {})
        for key in list(profile.keys()):
            if key in _GLASGOW_PROFILE_METADATA_KEYS or profile.get(key) is not None:
                continue
            fallback_value = _glasgow_field_median(key, profiles)
            if fallback_value is None:
                continue
            profile[key] = fallback_value
            unavailable.pop(key, None)
            fallback_provenance = _glasgow_field_provenance(key, profiles)
            if fallback_provenance:
                provenance[key] = fallback_provenance
        profile["unavailable_reasons"] = unavailable
        profile["provenance"] = provenance
        return profile

    if not profiles:
        return {
            "measurement_source": "unavailable",
            "representative_beat_id": None,
            "unavailable_reasons": {},
            "provenance": {},
        }

    keys = sorted(set().union(*(profile.keys() for profile in profiles)) - _GLASGOW_PROFILE_METADATA_KEYS)
    aggregate: Dict[str, object] = {key: _glasgow_field_median(key, profiles) for key in keys}

    unavailable = {}
    provenance = {}
    for key in keys:
        if aggregate.get(key) is None:
            unavailable[key] = _glasgow_field_unavailable_reason(key, profiles) or f"{key}_unavailable"
        provenance[key] = _glasgow_field_provenance(key, profiles) or "glasgow_explicit"
    aggregate.update(
        {
            "measurement_source": "measurement_beat_median_fallback",
            "representative_beat_id": None,
            "unavailable_reasons": unavailable,
            "provenance": provenance,
        }
    )
    return aggregate


def build_representative_lead_features(
    beat_features: List[LeadBeatFeatures],
    quality: Dict[str, object],
    beat_groups: Optional[Dict[int, List[int]]] = None,
    representative_beat_features: Optional[List[LeadBeatFeatures]] = None,
    dominant_group_id: int = 1,
    selected_beat_ids: Optional[Set[int]] = None,
    adjacent_precordial_correlations: Optional[Dict[str, float]] = None,
) -> Dict[str, RepresentativeLeadFeatures]:
    by_lead: Dict[str, List[LeadBeatFeatures]] = defaultdict(list)
    for bf in beat_features:
        by_lead[bf.lead].append(bf)

    dominant_members = set(int(bid) for bid in (selected_beat_ids or []))
    if not dominant_members and beat_groups:
        dominant_members = set(beat_groups.get(dominant_group_id, []))

    if dominant_members:
        pooled_by_lead: Dict[str, List[LeadBeatFeatures]] = defaultdict(list)
        for bf in beat_features:
            if bf.beat_id in dominant_members:
                pooled_by_lead[bf.lead].append(bf)
    else:
        pooled_by_lead = by_lead

    rep_by_lead = {
        bf.lead: bf
        for bf in (representative_beat_features or [])
    }

    reps: Dict[str, RepresentativeLeadFeatures] = {}
    for lead in STANDARD_12_LEADS:
        items = pooled_by_lead.get(lead, [])
        rep_feature = rep_by_lead.get(lead)
        st_j_params = _representative_st_j_params(rep_feature, items)
        st_hybrid_params = _representative_hybrid_st_params(rep_feature, items)
        params = {
            "pr_ms": _pick_pooled_interval_value(rep_feature, items, "pr_ms"),
            "qrs_ms": _pick_pooled_interval_value(rep_feature, items, "qrs_ms"),
            "qt_ms": _pick_pooled_interval_value(rep_feature, items, "qt_ms"),
            "jt_ms": _pick_pooled_interval_value(rep_feature, items, "jt_ms"),
            # Cross-beat polarity consensus, not one beat's sample -- see
            # `_p_polarity_consensus_value` for the sign-error measurement -- and
            # withheld outright when the P is no larger than the drift it sits on.
            "p_amp_mv": _p_amplitude_if_measurable(rep_feature, items),
            "p_area": _p_polarity_consensus_value(rep_feature, items, "p_area"),
            "p_dur_ms": _pick_pooled_interval_value(rep_feature, items, "p_dur_ms"),
            "p_dur_consensus_ms": _nanmedian(
                _valid_measurement("p_dur_consensus_ms", i.p_dur_consensus_ms) for i in items
            ),
            "pr_segment_consensus_ms": _nanmedian(
                _valid_measurement("pr_segment_consensus_ms", i.pr_segment_consensus_ms) for i in items
            ),
            "p_notched": _pick_representative_bool(rep_feature, items, "p_notched"),
            "p_biphasic": _pick_representative_bool(rep_feature, items, "p_biphasic"),
            "p_notch_interval_ms": _p_polarity_consensus_value(rep_feature, items, "p_notch_interval_ms"),
            "p_initial_duration_ms": _p_polarity_consensus_value(rep_feature, items, "p_initial_duration_ms"),
            "p_initial_amp_mv": _p_polarity_consensus_value(rep_feature, items, "p_initial_amp_mv"),
            "p_terminal_duration_ms": _p_polarity_consensus_value(rep_feature, items, "p_terminal_duration_ms"),
            "p_terminal_amp_mv": _p_polarity_consensus_value(rep_feature, items, "p_terminal_amp_mv"),
            "p_terminal_area_mv_ms": _p_polarity_consensus_value(rep_feature, items, "p_terminal_area_mv_ms"),
            "p_template_corr": _nanmedian(i.p_template_corr for i in items),
            "p_multilead_support_score": _nanmedian(i.p_multilead_support_score for i in items),
            "p_pr_consistency_score": _nanmedian(i.p_pr_consistency_score for i in items),
            "p_pp_consistency_score": _nanmedian(i.p_pp_consistency_score for i in items),
            "p_candidate_context_score": _nanmedian(i.p_candidate_context_score for i in items),
            "q_amp_mv": _pick_representative_value(rep_feature, items, "q_amp_mv"),
            "r_amp_mv": _pick_representative_value(rep_feature, items, "r_amp_mv"),
            "r_prime_amp_mv": _pick_representative_value(rep_feature, items, "r_prime_amp_mv"),
            "s_amp_mv": _pick_representative_value(rep_feature, items, "s_amp_mv"),
            "s_prime_amp_mv": _pick_representative_value(rep_feature, items, "s_prime_amp_mv"),
            "q_duration_ms": _pick_representative_value(rep_feature, items, "q_duration_ms"),
            "r_duration_ms": _pick_representative_value(rep_feature, items, "r_duration_ms"),
            "r_prime_duration_ms": _pick_representative_value(rep_feature, items, "r_prime_duration_ms"),
            "s_duration_ms": _pick_representative_value(rep_feature, items, "s_duration_ms"),
            "s_prime_duration_ms": _pick_representative_value(rep_feature, items, "s_prime_duration_ms"),
            "q_area_mv_ms": _pick_representative_value(rep_feature, items, "q_area_mv_ms"),
            "q_r_ratio": _pick_representative_value(rep_feature, items, "q_r_ratio"),
            "initial_qrs_area_mv_ms": _pick_representative_value(rep_feature, items, "initial_qrs_area_mv_ms"),
            "initial_qrs_net_mv": _pick_representative_value(rep_feature, items, "initial_qrs_net_mv"),
            "qrs_area": _pick_representative_value(rep_feature, items, "qrs_area"),
            "qrs_signed_area": _pick_representative_value(rep_feature, items, "qrs_signed_area"),
            "qrs_signed_area_uv_ms": _pick_representative_value(
                rep_feature, items, "twelve_sl_qrs_signed_area_uv_ms"
            ),
            "st_on_mv": st_j_params["st_on_mv"],
            "st_j_source": st_j_params["st_j_source"],
            "st_j_reliable": st_j_params["st_j_reliable"],
            "st_j_unreliable": st_j_params["st_j_unreliable"],
            "st_j_unreliable_reason": st_j_params["st_j_unreliable_reason"],
            "st_mid_mv": _pick_representative_value(rep_feature, items, "st_mid_mv"),
            "st_80ms_mv": _pick_representative_value(rep_feature, items, "st_80ms_mv"),
            **st_hybrid_params,
            "t_amp_mv": _pick_representative_t_value(rep_feature, items, "t_amp_mv"),
            "t_area": _pick_representative_t_value(rep_feature, items, "t_area"),
            "t_signed_area": _pick_representative_t_value(rep_feature, items, "t_signed_area"),
            "t_dur_ms": _pick_representative_value(rep_feature, items, "t_dur_ms"),
            "t_polarity": _mode_int(_rep_and_pooled_values(rep_feature, items, "t_polarity")),
            "t_symmetry": _pick_representative_value(rep_feature, items, "t_symmetry"),
            "u_wave_flag": _pick_representative_bool(rep_feature, items, "u_wave_flag"),
            "u_amp_mv": _pick_representative_value(rep_feature, items, "u_amp_mv"),
            **_u_wave_params(rep_feature, items),
            "t_wavelet_onset_index": _pick_representative_value(
                rep_feature, items, "t_wavelet_onset_index"
            ),
            "t_wavelet_offset_index": _pick_representative_value(
                rep_feature, items, "t_wavelet_offset_index"
            ),
            "t_trapezium_offset_index": _pick_representative_value(
                rep_feature, items, "t_trapezium_offset_index"
            ),
            "t_candidate_spread_ms": _pick_representative_value(
                rep_feature, items, "t_candidate_spread_ms"
            ),
            "t_local_snr_db": _pick_representative_value(
                rep_feature, items, "t_local_snr_db"
            ),
            "t_boundary_stability_ms": _pick_representative_value(
                rep_feature, items, "t_boundary_stability_ms"
            ),
            "t_sqi_score": _pick_representative_value(
                rep_feature, items, "t_sqi_score"
            ),
            "t_sqi_pass": _pick_representative_bool(
                rep_feature, items, "t_sqi_pass"
            ),
            "t_fusion_weight": _pick_representative_value(
                rep_feature, items, "t_fusion_weight"
            ),
            "t_offset_fusion_support": _mode_int(
                _rep_and_pooled_values(rep_feature, items, "t_offset_fusion_support")
            ),
            "t_offset_fusion_mad_ms": _pick_representative_value(
                rep_feature, items, "t_offset_fusion_mad_ms"
            ),
            "t_offset_fusion_ci_half_width_ms": _pick_representative_value(
                rep_feature, items, "t_offset_fusion_ci_half_width_ms"
            ),
            "t_offset_fusion_used_leads": _mode_str(
                _rep_and_pooled_values(
                    rep_feature, items, "t_offset_fusion_used_leads"
                )
            ),
            "t_offset_fusion_excluded_leads": _mode_str(
                _rep_and_pooled_values(
                    rep_feature, items, "t_offset_fusion_excluded_leads"
                )
            ),
            "t_offset_fusion_reliable": _pick_representative_bool(
                rep_feature, items, "t_offset_fusion_reliable"
            ),
            "t_offset_statistical_fusion_reliable": _pick_representative_consensus_bool(
                rep_feature, items, "t_offset_statistical_fusion_reliable"
            ),
            "t_offset_cluster_count": _mode_int(
                _rep_and_pooled_values(rep_feature, items, "t_offset_cluster_count")
            ),
            "t_offset_selected_cluster_score": _pick_representative_value(
                rep_feature, items, "t_offset_selected_cluster_score"
            ),
            "t_offset_selected_cluster_support": _mode_int(
                _rep_and_pooled_values(
                    rep_feature, items, "t_offset_selected_cluster_support"
                )
            ),
            "t_offset_selected_cluster_lead_groups": _mode_str(
                _rep_and_pooled_values(
                    rep_feature, items, "t_offset_selected_cluster_lead_groups"
                )
            ),
            "t_offset_selected_cluster_methods": _mode_str(
                _rep_and_pooled_values(
                    rep_feature, items, "t_offset_selected_cluster_methods"
                )
            ),
            "t_offset_fusion_reliability_reason": _mode_str(
                _rep_and_pooled_values(
                    rep_feature, items, "t_offset_fusion_reliability_reason"
                )
            ),
            "t_global_tpte_ms": _pick_representative_value(
                rep_feature, items, "t_global_tpte_ms"
            ),
            "t_offset_derived_disagreement_ms": _pick_representative_value(
                rep_feature, items, "t_offset_derived_disagreement_ms"
            ),
            "t_offset_tail_incomplete_leads": _mode_str(
                _rep_and_pooled_values(
                    rep_feature, items, "t_offset_tail_incomplete_leads"
                )
            ),
            "t_offset_systematic_early_risk": _pick_representative_consensus_bool(
                rep_feature, items, "t_offset_systematic_early_risk"
            ),
            "t_offset_morphology_guard_pass": _pick_representative_consensus_bool(
                rep_feature, items, "t_offset_morphology_guard_pass"
            ),
            "t_rms_offset_index": _pick_representative_value(
                rep_feature, items, "t_rms_offset_index"
            ),
            "t_pc1_offset_index": _pick_representative_value(
                rep_feature, items, "t_pc1_offset_index"
            ),
            "t_derived_spread_ms": _pick_representative_value(
                rep_feature, items, "t_derived_spread_ms"
            ),
            "pr_segment_level_mv": _pick_representative_value(
                rep_feature, items, "pr_segment_level_mv"
            ),
            "qrs_notch_count": _mode_int(
                _rep_and_pooled_values(rep_feature, items, "qrs_notch_count")
            ),
            "qrs_slur_flag": _pick_representative_bool(
                rep_feature, items, "qrs_slur_flag"
            ),
            "vat_ms": _pick_representative_value(rep_feature, items, "vat_ms"),
            "st_slope_mv_per_ms": _pick_representative_value(
                rep_feature, items, "st_slope_mv_per_ms"
            ),
            "t_peak_path": _mode_str(_rep_and_pooled_values(rep_feature, items, "t_peak_path")),
            "t_polarity_expected": _mode_int(_rep_and_pooled_values(rep_feature, items, "t_polarity_expected")),
            "t_polarity_observed": _mode_int(_rep_and_pooled_values(rep_feature, items, "t_polarity_observed")),
            "st_t_confusion": any(bool(v) for v in _rep_and_pooled_values(rep_feature, items, "st_t_confusion")),
            "t_confidence_reason": _mode_str(_rep_and_pooled_values(rep_feature, items, "t_confidence_reason")),
            "reliable_for_global": bool(getattr(quality[lead], "reliable", False)),
            "reliable_for_p":      bool(getattr(quality[lead], "reliable_for_p",   False)),
            "reliable_for_qrs":    bool(getattr(quality[lead], "reliable_for_qrs", False)),
            "reliable_for_qt":     bool(getattr(quality[lead], "reliable_for_qt", False)),
            "reliable_for_t":      bool(getattr(quality[lead], "reliable_for_t",  False)),
            "qt_confidence_mean":  _nanmean(i.qt_confidence  for i in items),
            "p_confidence_mean":   _nanmean(i.p_confidence   for i in items),
            "p_onset_confidence_mean": _nanmean(i.p_onset_confidence for i in items),
            "p_offset_confidence_mean": _nanmean(i.p_offset_confidence for i in items),
            "p_local_noise_rms_mv_median": _nanmedian(
                i.p_local_noise_rms_mv for i in items
            ),
            "p_local_snr_median": _nanmedian(i.p_local_snr for i in items),
            "p_informative_fraction": _nanmean(
                float(i.p_informative)
                for i in items
                if i.p_informative is not None
            ),
            "p_tp_gap_ms_median": _nanmedian(i.p_tp_gap_ms for i in items),
            "p_quiet_window_available_fraction": _nanmean(
                float(i.p_quiet_window_available)
                for i in items
                if i.p_quiet_window_available is not None
            ),
            "p_on_t_overlap_risk": any(
                i.p_on_t_overlap_risk is True for i in items
            ),
            "p_ta_overlap_risk": any(
                i.p_ta_overlap_risk is True for i in items
            ),
            "p_baseline_method": _mode_str(
                _rep_and_pooled_values(rep_feature, items, "p_baseline_method")
            ),
            "p_baseline_confidence_mean": _nanmean(
                i.p_baseline_confidence for i in items
            ),
            "p_onset_sigma_ms_median": _nanmedian(
                i.p_onset_sigma_ms for i in items
            ),
            "p_offset_sigma_ms_median": _nanmedian(
                i.p_offset_sigma_ms for i in items
            ),
            "p_boundary_stability_method": _mode_str(
                _rep_and_pooled_values(
                    rep_feature,
                    items,
                    "p_boundary_stability_method",
                )
            ),
            "qrs_confidence_mean": _nanmean(i.qrs_confidence for i in items),
            "qrs_off_confidence_mean": _nanmean(i.qrs_off_confidence for i in items),
            "qrs_terminal_path": _mode_str(_rep_and_pooled_values(rep_feature, items, "qrs_terminal_path")),
            "qrs_offset_used_leads": _mode_str(_rep_and_pooled_values(rep_feature, items, "qrs_offset_used_leads")),
            "qrs_offset_excluded_leads": _mode_str(_rep_and_pooled_values(rep_feature, items, "qrs_offset_excluded_leads")),
            "qrs_offset_exclusion_reason": _mode_str(_rep_and_pooled_values(rep_feature, items, "qrs_offset_exclusion_reason")),
            "qrs_late_cluster_support": _mode_int(_rep_and_pooled_values(rep_feature, items, "qrs_late_cluster_support")),
            "qrs_terminal_confidence": _nanmean(i.qrs_terminal_confidence for i in items),
            "p_onset_consensus_support": _mode_int(_rep_and_pooled_values(rep_feature, items, "p_onset_consensus_support")),
            "p_offset_consensus_support": _mode_int(_rep_and_pooled_values(rep_feature, items, "p_offset_consensus_support")),
            "p_onset_consensus_spread_ms": _nanmedian(i.p_onset_consensus_spread_ms for i in items),
            "p_offset_consensus_spread_ms": _nanmedian(i.p_offset_consensus_spread_ms for i in items),
            "p_onset_cluster_support": _mode_int(_rep_and_pooled_values(rep_feature, items, "p_onset_cluster_support")),
            "p_onset_cluster_spread_ms": _nanmedian(i.p_onset_cluster_spread_ms for i in items),
            "p_onset_cluster_pr_ms": _nanmedian(
                _valid_measurement("pr_consensus_ms", i.p_onset_cluster_pr_ms) for i in items
            ),
            "p_onset_cluster_limb_support": _mode_int(
                _rep_and_pooled_values(rep_feature, items, "p_onset_cluster_limb_support")
            ),
            "p_onset_cluster_precordial_support": _mode_int(
                _rep_and_pooled_values(rep_feature, items, "p_onset_cluster_precordial_support")
            ),
            "p_onset_cluster_leads": _mode_str(_rep_and_pooled_values(rep_feature, items, "p_onset_cluster_leads")),
            "p_onset_consensus_reason": _mode_str(_rep_and_pooled_values(rep_feature, items, "p_onset_consensus_reason")),
            "p_boundary_consensus_source": _mode_str(_rep_and_pooled_values(rep_feature, items, "p_boundary_consensus_source")),
            "p_duration_guard_target_ms": _nanmedian(
                i.p_duration_guard_target_ms for i in items
            ),
            "p_duration_guard_delta_ms": _nanmedian(
                i.p_duration_guard_delta_ms for i in items
            ),
            "p_duration_guard_support": _mode_int(
                _rep_and_pooled_values(rep_feature, items, "p_duration_guard_support")
            ),
            "p_duration_guard_source": _mode_str(
                _rep_and_pooled_values(rep_feature, items, "p_duration_guard_source")
            ),
            "beat_count": len(items),
            "representative_group_id": dominant_group_id,
            "measurement_source": "dominant_representative_beat" if rep_feature is not None else "beat_median_fallback",
            "glasgow_measurements": _representative_glasgow_measurements(rep_feature, items),
            # T027 extended measurements
            "tpe_ms":          _pick_pooled_interval_value(rep_feature, items, "tpe_ms"),
            "st_morphology":   _mode_str(i.st_morphology for i in items),
            "fqrs_score":      _nanmedian(i.fqrs_score for i in items),
            "ptf_v1_mv_ms":    _nanmedian(i.ptf_v1_mv_ms for i in items),
            # T025 multi-lead consensus intervals
            "qt_consensus_ms":  _nanmedian(_valid_measurement("qt_consensus_ms", i.qt_consensus_ms) for i in items),
            "qt_latest_p85_ms": _nanmedian(
                _valid_measurement("qt_consensus_ms", i.qt_latest_p85_ms)
                for i in items
            ),
            "pr_consensus_ms":  _nanmedian(_valid_measurement("pr_consensus_ms", i.pr_consensus_ms) for i in items),
            "qrs_consensus_ms": _nanmedian(_valid_measurement("qrs_consensus_ms", i.qrs_consensus_ms) for i in items),
            "qrs_wide_ms":      _nanmedian(_valid_measurement("qrs_consensus_ms", i.qrs_wide_ms)      for i in items),
        }
        _t_fusion_support = int(params.get("t_offset_fusion_support") or 0)
        _t_fusion_mad = params.get("t_offset_fusion_mad_ms")
        params["t_offset_fusion_reliable"] = bool(
            params.get("t_offset_statistical_fusion_reliable")
            and params.get("t_offset_morphology_guard_pass")
            and _t_fusion_support >= 4
            and _t_fusion_mad is not None
            and np.isfinite(float(_t_fusion_mad))
            and float(_t_fusion_mad) <= 20.0
        )
        if not _params_have_visible_p(params):
            params["pr_ms"] = None
            params["pr_consensus_ms"] = None
            params["p_dur_ms"] = None
            params["p_dur_consensus_ms"] = None
            params["pr_segment_consensus_ms"] = None
            params["p_measurement_suppressed"] = "low_p_support"
        def _var(attr: str) -> float:
            vals = [
                v for v in (_valid_measurement(attr, getattr(i, attr)) for i in items)
                if v is not None
            ]
            return float(np.std(vals)) if vals else np.nan
        variance = {
            "pr_ms_sd":    _var("pr_ms"),
            "qrs_ms_sd":   _var("qrs_ms"),
            "qt_ms_sd":    _var("qt_ms"),
            "jt_ms_sd":    _var("jt_ms"),    # T022
            "st_on_mv_sd": float(np.std([v for v in (_valid_st_j_value(i) for i in items) if v is not None]))
            if any(_valid_st_j_value(i) is not None for i in items) else np.nan,
            "t_amp_mv_sd": _var("t_amp_mv"),
        }
        reps[lead] = RepresentativeLeadFeatures(lead=lead, params=params, variance=variance)

    _cap_qrs_consensus_to_raw_distribution(reps)

    precordial_lead_metrics = {
        lead: {
            "r_amp_mv": reps[lead].params.get("r_amp_mv"),
            "s_amp_mv": reps[lead].params.get("s_amp_mv"),
            "q_amp_mv": reps[lead].params.get("q_amp_mv"),
            "qrs_ms": reps[lead].params.get("qrs_ms"),
        }
        for lead in [f"V{i}" for i in range(1, 7)]
    }
    precordial_reversal = detect_precordial_reversal(
        precordial_lead_metrics, adjacent_precordial_correlations
    )
    if precordial_reversal.get("suspected"):
        for lead in [f"V{i}" for i in range(1, 7)]:
            reps[lead].params["probable_precordial_reversal"] = True
            reps[lead].params["precordial_reversal_detail"] = precordial_reversal
    return reps


def compute_group_features(
    beat_features: List[LeadBeatFeatures],
    beat_groups: Dict[int, List[int]],
    total_beats: int,
    fs: int,
    r_locs: Optional[np.ndarray] = None,
) -> Dict[int, GroupFeatures]:
    by_beat: Dict[int, List[LeadBeatFeatures]] = defaultdict(list)
    for bf in beat_features:
        by_beat[bf.beat_id].append(bf)

    groups: Dict[int, GroupFeatures] = {}
    for gid, members in beat_groups.items():
        member_feats = [f for bid in members for f in by_beat.get(bid, []) if f.lead == "II"]
        rr = []
        for a, b in zip(members[:-1], members[1:]):
            if r_locs is not None and b < len(r_locs) and a < len(r_locs):
                rr.append((float(r_locs[b]) - float(r_locs[a])) * 1000.0 / fs)
        qrs = [v for m in member_feats if (v := _valid_measurement("qrs_ms", m.qrs_ms)) is not None]
        pr = [v for m in member_feats if (v := _valid_measurement("pr_ms", m.pr_ms)) is not None]
        qt = [v for m in member_feats if (v := _valid_measurement("qt_ms", m.qt_ms)) is not None]
        longest_run = 1
        cur = 1
        for a, b in zip(members[:-1], members[1:]):
            if b == a + 1:
                cur += 1
                longest_run = max(longest_run, cur)
            else:
                cur = 1
        mean_rr = float(np.mean(rr)) if rr else None
        groups[gid] = GroupFeatures(
            group_id=gid,
            member_count=len(members),
            member_pct=100.0 * len(members) / max(1, total_beats),
            longest_run=longest_run,
            mean_rr_ms=mean_rr,
            mean_pr_ms=_nanmean(pr),
            mean_qrs_ms=_nanmean(qrs),
            mean_qt_ms=_nanmean(qt),
            mean_ventr_rate_bpm=(60000.0 / mean_rr) if mean_rr and mean_rr > 0 else None,
            flags={
                "dominant_group": gid == 1,
                "wide_qrs": bool(_nanmean(qrs) and _nanmean(qrs) > 120),
            },
        )
    return groups


def _estimate_atrial_rate_bpm(
    beat_features: List[LeadBeatFeatures],
    fs: int,
) -> Optional[float]:
    by_beat: Dict[int, List[LeadBeatFeatures]] = defaultdict(list)
    for bf in beat_features:
        by_beat[bf.beat_id].append(bf)

    p_peaks = []
    for beat_id in sorted(by_beat):
        candidates = [
            bf for bf in by_beat[beat_id]
            if bf.p.peak is not None
            and bf.p_confidence > 0.0
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda bf: (
                bool(bf.beat_measurement_reliable),
                float(bf.p_confidence),
                abs(float(bf.p_amp_mv or 0.0)),
            ),
            reverse=True,
        )
        p_peaks.append(int(candidates[0].p.peak))

    if len(p_peaks) < 2:
        return None

    pp_ms = np.diff(np.asarray(p_peaks, dtype=float)) * 1000.0 / fs
    pp_ms = pp_ms[(pp_ms >= 250.0) & (pp_ms <= 2000.0)]
    if len(pp_ms) == 0:
        return None

    return float(60000.0 / np.median(pp_ms))


def _axis_from_amplitudes(
    lead_values: Dict[str, Optional[float]],
    confidences: Optional[Dict[str, Optional[float]]] = None,
) -> Optional[float]:
    """
    Compute electrical axis (degrees) from limb-lead amplitudes.
    T020: optional per-lead confidence weights improve axis when some leads are noisy.
    """
    xs = []
    ys = []
    ws = []
    for lead, angle_deg in HEXAXIAL_ANGLES.items():
        val = lead_values.get(lead)
        if val is None or not np.isfinite(val):
            continue
        conf = 1.0
        if confidences is not None:
            c = confidences.get(lead)
            if c is not None and np.isfinite(c):
                conf = max(0.1, float(c))   # floor at 0.1 to avoid zero-weight
        angle = np.deg2rad(angle_deg)
        w = abs(val) * conf
        xs.append(val * np.cos(angle) * conf)
        ys.append(val * np.sin(angle) * conf)
        ws.append(w)
    if not ws or np.sum(ws) < 1e-6:
        return None
    x = float(np.sum(xs))
    y = float(np.sum(ys))
    return float(np.rad2deg(np.arctan2(y, x)))


def _axis_delta_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return None
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def _axis_projection(axis_deg: float, lead: str) -> float:
    return float(np.cos(np.deg2rad(float(axis_deg) - HEXAXIAL_ANGLES[lead])))


def _axis_fit_score(
    lead_values: Dict[str, Optional[float]],
    confidences: Optional[Dict[str, Optional[float]]] = None,
) -> tuple[Optional[float], float, int]:
    axis = _axis_from_amplitudes(lead_values, confidences)
    if axis is None:
        return None, float("inf"), 0

    rows: List[tuple[float, float, float]] = []
    for lead in HEXAXIAL_ANGLES:
        value = lead_values.get(lead)
        if value is None or not np.isfinite(float(value)):
            continue
        confidence = 1.0
        if confidences is not None:
            conf = confidences.get(lead)
            if conf is not None and np.isfinite(float(conf)):
                confidence = max(0.1, float(conf))
        rows.append((float(value), _axis_projection(axis, lead), confidence))
    if len(rows) < 2:
        return axis, float("inf"), 0

    den = float(sum(conf * proj * proj for _value, proj, conf in rows))
    if den < 1e-9:
        return axis, float("inf"), 0
    scale = float(sum(conf * value * proj for value, proj, conf in rows) / den)
    if scale <= 0.0 or not np.isfinite(scale):
        return axis, float("inf"), 0

    residual_sum = 0.0
    weight_sum = 0.0
    sign_conflicts = 0
    for value, projection, confidence in rows:
        expected = scale * projection
        norm = max(abs(value), abs(expected), 1e-6)
        residual = abs(value - expected) / norm
        residual_sum += confidence * residual
        weight_sum += confidence
        if (
            abs(projection) >= _P_AXIS_ROBUST_PROJECTION_NEUTRAL
            and value * projection < 0.0
        ):
            sign_conflicts += 1
    if weight_sum <= 0.0:
        return axis, float("inf"), sign_conflicts
    score = float(residual_sum / weight_sum + 0.55 * sign_conflicts / len(rows))
    return axis, score, sign_conflicts


def _p_axis_can_remove_physical_outlier(
    lead: str,
    value: float,
    retained_values: Dict[str, Optional[float]],
    retained_axis: float,
) -> bool:
    retained_abs = [
        abs(float(v))
        for v in retained_values.values()
        if v is not None and np.isfinite(float(v))
    ]
    if len(retained_abs) < _P_AXIS_ROBUST_MIN_RETAINED_LEADS:
        return False
    projection = _axis_projection(retained_axis, lead)
    if abs(projection) < _P_AXIS_ROBUST_PROJECTION_NEUTRAL:
        return False
    sign_conflict = float(value) * projection < 0.0
    dominant = abs(float(value)) >= _P_AXIS_ROBUST_DOMINANT_RATIO * max(
        float(np.median(retained_abs)),
        _P_AXIS_SEED_MIN_ABS_AREA,
    )
    return bool(sign_conflict and dominant)


def _robust_p_axis_from_values(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    p_axis_values: Dict[str, Optional[float]],
    confidences: Optional[Dict[str, Optional[float]]] = None,
) -> Optional[float]:
    active = {
        lead: float(value)
        for lead, value in p_axis_values.items()
        if lead in HEXAXIAL_ANGLES
        and value is not None
        and np.isfinite(float(value))
    }
    if len(active) < _P_AXIS_ROBUST_MIN_RETAINED_LEADS:
        return _axis_from_amplitudes(p_axis_values, confidences)

    all_axis, all_score, _ = _axis_fit_score(active, confidences)
    if all_axis is None:
        return None

    best_axis = all_axis
    best_score = all_score
    for removed_lead, removed_value in active.items():
        retained = {
            lead: value
            for lead, value in active.items()
            if lead != removed_lead
        }
        if len(retained) < _P_AXIS_ROBUST_MIN_RETAINED_LEADS:
            continue
        if not _p_axis_values_have_local_pr_support(representative_leads, retained):
            continue
        retained_axis, retained_score, _ = _axis_fit_score(retained, confidences)
        if retained_axis is None:
            continue
        if retained_score + _P_AXIS_ROBUST_MIN_SCORE_IMPROVEMENT >= all_score:
            continue
        if not _p_axis_can_remove_physical_outlier(
            removed_lead,
            removed_value,
            retained,
            retained_axis,
        ):
            continue
        best_axis = retained_axis
        best_score = retained_score

    if best_score > _P_AXIS_ROBUST_MAX_FINAL_SCORE:
        return None
    return best_axis


def _signed_wave_area(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    area_key: str,
    amp_key: str,
    reliable_key: Optional[str] = None,
    confidence_key: Optional[str] = None,
    min_confidence: float = 0.0,
    min_abs_amp_mv: float = 0.0,
    signed_area_key: Optional[str] = None,
) -> "Dict[str, Optional[float]]":
    """
    Signed waveform area for axis calculation (DXL convention).

    Unsigned area (stored as absolute integral) × sign(peak amplitude) gives
    the net signed area — correct for axis when the wave is predominantly one
    polarity (normal P and T waves).

    reliable_key: if set (e.g. "reliable_for_p"), leads where that flag is
    False are excluded so false detections don't corrupt the axis vector.
    """
    result: Dict[str, Optional[float]] = {}
    for k, v in representative_leads.items():
        if reliable_key and not v.params.get(reliable_key, False):
            result[k] = None
            continue
        if confidence_key is not None:
            confidence = v.params.get(confidence_key)
            if min_confidence > 0.0 and (confidence is None or not np.isfinite(float(confidence))):
                result[k] = None
                continue
            if confidence is not None and np.isfinite(float(confidence)) and float(confidence) < min_confidence:
                result[k] = None
                continue
        amp  = v.params.get(amp_key)
        if min_abs_amp_mv > 0.0 and (amp is None or not np.isfinite(float(amp))):
            result[k] = None
            continue
        if amp is not None and np.isfinite(float(amp)) and abs(float(amp)) < min_abs_amp_mv:
            result[k] = None
            continue
        if signed_area_key is not None:
            signed_area = v.params.get(signed_area_key)
            if signed_area is not None and np.isfinite(float(signed_area)):
                result[k] = float(signed_area)
                continue
        area = v.params.get(area_key)
        if area is None or not np.isfinite(float(area)):
            result[k] = None
        else:
            sign = 1.0 if (amp is not None and np.isfinite(float(amp)) and float(amp) >= 0.0) else -1.0
            result[k] = sign * float(area)
    return result


def _lead_context_score(rep: "RepresentativeLeadFeatures", key: str) -> Optional[float]:
    value = rep.params.get(key)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _p_axis_boundary_confidence(rep: "RepresentativeLeadFeatures") -> Optional[float]:
    values: List[float] = []
    for key in ("p_onset_confidence_mean", "p_offset_confidence_mean"):
        value = rep.params.get(key)
        if value is None:
            continue
        value_f = float(value)
        if np.isfinite(value_f):
            values.append(value_f)
    if not values:
        return None
    return float(min(values))


def _p_axis_has_sinus_like_pr_support(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
) -> bool:
    prs: List[float] = []
    for lead in HEXAXIAL_ANGLES:
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        pr = _valid_measurement("pr_ms", rep.params.get("pr_ms"))
        if pr is None:
            pr = _valid_measurement("pr_consensus_ms", rep.params.get("pr_consensus_ms"))
        if pr is None:
            continue
        if _P_AXIS_INVERSION_MIN_PR_MS <= float(pr) <= _P_AXIS_INVERSION_MAX_PR_MS:
            prs.append(float(pr))
    if len(prs) < _P_AXIS_INVERSION_MIN_PR_SUPPORT:
        return False
    if float(np.max(prs) - np.min(prs)) > _P_AXIS_INVERSION_MAX_PR_SPREAD_MS:
        return False
    return True


def _p_axis_values_have_local_pr_support(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    p_axis_values: Dict[str, Optional[float]],
) -> bool:
    prs: List[float] = []
    for lead in HEXAXIAL_ANGLES:
        value = p_axis_values.get(lead)
        rep = representative_leads.get(lead)
        if value is None or rep is None or not np.isfinite(float(value)):
            continue
        pr = _valid_measurement("pr_ms", rep.params.get("pr_ms"))
        if pr is None:
            pr = _valid_measurement("pr_consensus_ms", rep.params.get("pr_consensus_ms"))
        if pr is None:
            continue
        if _P_AXIS_INVERSION_MIN_PR_MS <= float(pr) <= _P_AXIS_INVERSION_MAX_PR_MS:
            prs.append(float(pr))
    if len(prs) < _P_AXIS_MIN_FRONTAL_LEADS:
        return False
    if float(np.max(prs) - np.min(prs)) > _P_AXIS_INVERSION_MAX_PR_SPREAD_MS:
        return False
    return True


def _should_flip_sinus_like_p_axis_inversion(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    p_axis_values: Dict[str, Optional[float]],
) -> bool:
    confidences = {
        lead: rep.params.get("p_confidence_mean")
        for lead, rep in representative_leads.items()
    }
    axis = _axis_from_amplitudes(p_axis_values, confidences)
    if axis is None:
        return False
    axis_std = ((float(axis) + 180.0) % 360.0) - 180.0
    if not (
        axis_std <= _P_AXIS_INVERSION_EXTREME_NEG_DEG
        or axis_std >= _P_AXIS_INVERSION_EXTREME_POS_DEG
    ):
        return False
    if not _p_axis_has_sinus_like_pr_support(representative_leads):
        return False

    def _sign(lead: str) -> Optional[int]:
        value = p_axis_values.get(lead)
        if value is None or not np.isfinite(float(value)) or abs(float(value)) < _P_AXIS_SEED_MIN_ABS_AREA:
            return None
        return 1 if float(value) > 0.0 else -1

    inversion_votes = 0
    if _sign("I") == -1:
        inversion_votes += 1
    if _sign("II") == -1:
        inversion_votes += 1
    if _sign("aVF") == -1:
        inversion_votes += 1
    if _sign("aVR") == 1:
        inversion_votes += 1
    if inversion_votes < 3:
        return False

    context_values = [
        context for lead, rep in representative_leads.items()
        if lead in HEXAXIAL_ANGLES
        and (context := _lead_context_score(rep, "p_candidate_context_score")) is not None
    ]
    if context_values and float(np.median(context_values)) < _P_AXIS_CONTEXT_STRONG:
        return False
    return True


def _p_axis_values_with_polarity_consensus(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
) -> Dict[str, Optional[float]]:
    values = _signed_wave_area(
        representative_leads,
        "p_area",
        "p_amp_mv",
        reliable_key="reliable_for_p",
        confidence_key="p_confidence_mean",
        min_confidence=_P_MIN_CONFIDENCE,
        min_abs_amp_mv=_P_MIN_ABS_AMP_MV,
    )

    result: Dict[str, Optional[float]] = {
        lead: (
            float(value)
            if value is not None and np.isfinite(float(value))
            else None
        )
        for lead, value in values.items()
    }
    boundary_confidences: Dict[str, Optional[float]] = {}
    for lead in HEXAXIAL_ANGLES:
        rep = representative_leads.get(lead)
        if rep is None:
            boundary_confidences[lead] = None
            continue
        boundary_confidence = _p_axis_boundary_confidence(rep)
        boundary_confidences[lead] = boundary_confidence
        if (
            result.get(lead) is not None
            and boundary_confidence is not None
            and boundary_confidence < _P_AXIS_MIN_BOUNDARY_CONFIDENCE
        ):
            result[lead] = None

    has_candidate_context = any(
        _lead_context_score(rep, "p_candidate_context_score") is not None
        for lead, rep in representative_leads.items()
        if lead in HEXAXIAL_ANGLES
    )
    if not has_candidate_context:
        return result

    strong_abs_values: List[float] = []
    for lead in HEXAXIAL_ANGLES:
        value = result.get(lead)
        rep = representative_leads.get(lead)
        if value is None or rep is None:
            continue
        context = _lead_context_score(rep, "p_candidate_context_score")
        confidence = _lead_context_score(rep, "p_confidence_mean") or 0.0
        boundary_confidence = boundary_confidences.get(lead)
        if (
            (context is None or context >= _P_AXIS_CONTEXT_STRONG)
            and confidence >= _P_AXIS_SEED_MIN_CONFIDENCE
            and (
                boundary_confidence is None
                or boundary_confidence >= _P_AXIS_CONFLICT_MIN_BOUNDARY_CONFIDENCE
            )
            and abs(float(value)) >= _P_AXIS_SEED_MIN_ABS_AREA
        ):
            strong_abs_values.append(abs(float(value)))

    if strong_abs_values:
        strong_scale = max(float(np.median(strong_abs_values)), _P_AXIS_SEED_MIN_ABS_AREA)
        for lead in HEXAXIAL_ANGLES:
            value = result.get(lead)
            rep = representative_leads.get(lead)
            if value is None or rep is None:
                continue
            context = _lead_context_score(rep, "p_candidate_context_score")
            confidence = _lead_context_score(rep, "p_confidence_mean") or 0.0
            boundary_confidence = boundary_confidences.get(lead)
            weak_context = context is not None and context < _P_AXIS_CONTEXT_WEAK
            weak_boundary = (
                boundary_confidence is not None
                and boundary_confidence < _P_AXIS_CONFLICT_MIN_BOUNDARY_CONFIDENCE
            )
            if (
                (weak_context or weak_boundary)
                and confidence < _P_AXIS_SEED_MIN_CONFIDENCE
                and abs(float(value)) > 2.5 * strong_scale
            ):
                result[lead] = None

    if _should_flip_sinus_like_p_axis_inversion(representative_leads, result):
        return {
            lead: (-float(value) if value is not None and np.isfinite(float(value)) else None)
            for lead, value in result.items()
        }

    seed_leads = ("I", "II", "aVF")
    seed_values: Dict[str, float] = {}
    seed_confidences: Dict[str, float] = {}
    seed_signs: List[int] = []
    for lead in seed_leads:
        value = result.get(lead)
        rep = representative_leads.get(lead)
        if value is None or rep is None:
            return result
        context = _lead_context_score(rep, "p_candidate_context_score")
        confidence = _lead_context_score(rep, "p_confidence_mean") or 0.0
        boundary_confidence = boundary_confidences.get(lead)
        if context is not None and context < _P_AXIS_CONTEXT_STRONG:
            return result
        if (
            boundary_confidence is not None
            and boundary_confidence < _P_AXIS_CONFLICT_MIN_BOUNDARY_CONFIDENCE
        ):
            return result
        if confidence < _P_AXIS_SEED_MIN_CONFIDENCE or abs(float(value)) < _P_AXIS_SEED_MIN_ABS_AREA:
            return result
        seed_values[lead] = float(value)
        seed_confidences[lead] = float(confidence)
        seed_signs.append(1 if float(value) >= 0.0 else -1)

    if len(set(seed_signs)) != 1:
        return result
    seed_axis = _axis_from_amplitudes(seed_values, seed_confidences)
    if seed_axis is None:
        return result

    for lead in ("aVR", "aVL", "III"):
        value = result.get(lead)
        rep = representative_leads.get(lead)
        if value is None or rep is None:
            continue
        projection = float(np.cos(np.deg2rad(float(seed_axis) - HEXAXIAL_ANGLES[lead])))
        if abs(projection) < _P_AXIS_CONFLICT_PROJECTION_NEUTRAL:
            continue
        if np.sign(float(value)) == np.sign(projection):
            continue
        context = _lead_context_score(rep, "p_candidate_context_score")
        confidence = _lead_context_score(rep, "p_confidence_mean") or 0.0
        boundary_confidence = boundary_confidences.get(lead)
        weak_context = context is not None and context < _P_AXIS_CONTEXT_STRONG
        weak_boundary = (
            boundary_confidence is not None
            and boundary_confidence < _P_AXIS_CONFLICT_MIN_BOUNDARY_CONFIDENCE
        )
        if weak_context or weak_boundary or confidence < _P_AXIS_SEED_MIN_CONFIDENCE:
            result[lead] = None

    if _should_flip_sinus_like_p_axis_inversion(representative_leads, result):
        result = {
            lead: (-float(value) if value is not None and np.isfinite(float(value)) else None)
            for lead, value in result.items()
        }

    return result


def _st_corrected_t_amplitudes(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    signed_t_values: Dict[str, Optional[float]],
) -> "Dict[str, Optional[float]]":
    result: Dict[str, Optional[float]] = {}
    for lead, rep in representative_leads.items():
        if lead not in HEXAXIAL_ANGLES or signed_t_values.get(lead) is None:
            result[lead] = None
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        st_mid = _lead_float(rep, "st_mid_mv")
        if t_amp is None or st_mid is None:
            result[lead] = None
            continue
        st_confidence = _lead_float(rep, "twelve_sl_st_confidence")
        if st_confidence is not None and st_confidence < _T_AXIS_ST_CONFIDENCE_MIN:
            result[lead] = None
            continue
        value = t_amp - st_mid
        signed_value = signed_t_values.get(lead)
        if (
            signed_value is not None
            and np.isfinite(float(signed_value))
            and abs(float(signed_value)) >= 1e-6
            and abs(value) >= _T_ST_AXIS_MIN_ABS_AMP_MV
            and np.sign(value) != np.sign(float(signed_value))
        ):
            result[lead] = None
            continue
        if _has_t_prime_special_t_axis_conflict(rep, value):
            result[lead] = None
            continue
        result[lead] = value if abs(value) >= _T_ST_AXIS_MIN_ABS_AMP_MV else None
    return result


def _filter_limb_seed_t_axis_conflicts(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    t_axis_values: Dict[str, Optional[float]],
    qrs_ms: Optional[float],
    rr_cv: Optional[float],
) -> Dict[str, Optional[float]]:
    if qrs_ms is None or qrs_ms > _T_AXIS_STABLE_LIMB_MAX_QRS_MS:
        return t_axis_values
    if rr_cv is not None and rr_cv > _T_AXIS_STABLE_LIMB_MAX_RR_CV:
        return t_axis_values

    seed_leads = ("I", "II", "aVF")
    seed_values: Dict[str, float] = {}
    seed_confidences: Dict[str, float] = {}
    seed_signs: List[int] = []
    for lead in seed_leads:
        value = t_axis_values.get(lead)
        rep = representative_leads.get(lead)
        if value is None or rep is None or not np.isfinite(float(value)):
            return t_axis_values
        t_amp = _lead_float(rep, "t_amp_mv")
        conf = _lead_float(rep, "qt_confidence_mean")
        if t_amp is None or abs(t_amp) < _T_AXIS_LIMB_SEED_MIN_ABS_MV:
            return t_axis_values
        if conf is None or conf < _T_AXIS_LIMB_SEED_MIN_CONFIDENCE:
            return t_axis_values
        if abs(float(value)) < _T_AXIS_LIMB_SEED_MIN_AREA:
            return t_axis_values
        seed_values[lead] = float(value)
        seed_confidences[lead] = float(conf)
        seed_signs.append(1 if float(value) >= 0.0 else -1)

    if len(set(seed_signs)) != 1:
        return t_axis_values
    seed_axis = _axis_from_amplitudes(seed_values, seed_confidences)
    if seed_axis is None:
        return t_axis_values

    result = dict(t_axis_values)
    for lead in ("aVR", "aVL", "III"):
        value = result.get(lead)
        rep = representative_leads.get(lead)
        if value is None or rep is None or not np.isfinite(float(value)):
            continue
        conf = _lead_float(rep, "qt_confidence_mean") or 0.0
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _T_AXIS_LIMB_SEED_MIN_ABS_MV:
            continue
        projection = float(np.cos(np.deg2rad(float(seed_axis) - HEXAXIAL_ANGLES[lead])))
        if abs(projection) < _T_AXIS_LIMB_SEED_PROJECTION_NEUTRAL:
            continue
        if np.sign(float(value)) == np.sign(projection):
            continue
        if lead == "aVR" or conf < 0.95:
            result[lead] = None
    return result


def _has_t_prime_special_t_axis_conflict(
    rep: "RepresentativeLeadFeatures",
    axis_value: float,
) -> bool:
    special_t_uv = _lead_float(rep, "twelve_sl_special_t_uv")
    t_prime_area_uv_ms = _lead_float(rep, "twelve_sl_t_prime_area_uv_ms")
    return (
        special_t_uv is not None
        and t_prime_area_uv_ms is not None
        and abs(float(t_prime_area_uv_ms)) >= _T_AXIS_T_PRIME_CONFLICT_MIN_AREA_UV_MS
        and abs(float(special_t_uv)) >= _T_AXIS_SPECIAL_T_CONFLICT_MIN_UV
        and float(axis_value) * float(special_t_uv) < 0.0
    )


def _should_use_st_corrected_t_axis(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    signed_t_values: Dict[str, Optional[float]],
    st_corrected_values: Dict[str, Optional[float]],
    qrs_ms: Optional[float],
) -> bool:
    if qrs_ms is None or qrs_ms < _T_ST_AXIS_MIN_QRS_MS:
        return False

    usable = [
        lead for lead in HEXAXIAL_ANGLES
        if st_corrected_values.get(lead) is not None
    ]
    if len(usable) < _T_ST_AXIS_MIN_LEADS:
        return False

    disagreements = 0
    comparisons = 0
    for lead in HEXAXIAL_ANGLES:
        signed_value = signed_t_values.get(lead)
        if signed_value is None or not np.isfinite(float(signed_value)):
            continue
        t_amp = _lead_float(representative_leads[lead], "t_amp_mv")
        if t_amp is None or abs(t_amp) < _T_MIN_AXIS_ABS_AMP_MV or abs(float(signed_value)) < 1e-6:
            continue
        comparisons += 1
        if np.sign(float(signed_value)) != np.sign(t_amp):
            disagreements += 1

    return comparisons > 0 and disagreements <= _T_ST_AXIS_MAX_POLARITY_DISAGREEMENTS


def _filter_t_axis_values_for_reporting(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    t_axis_values: Dict[str, Optional[float]],
) -> "Dict[str, Optional[float]]":
    result: Dict[str, Optional[float]] = {}
    for lead in HEXAXIAL_ANGLES:
        value = t_axis_values.get(lead)
        if value is None or not np.isfinite(float(value)):
            result[lead] = None
            continue

        rep = representative_leads.get(lead)
        if rep is None:
            result[lead] = None
            continue

        area = _lead_float(rep, "t_area")
        if area is not None and abs(area) >= 1e-6:
            if abs(float(value)) / abs(area) < _T_AXIS_MIN_SIGNED_AREA_RATIO:
                result[lead] = None
                continue

        if _has_t_prime_special_t_axis_conflict(rep, float(value)):
            result[lead] = None
            continue

        t_amp = _lead_float(rep, "t_amp_mv")
        t_amp_sd = rep.variance.get("t_amp_mv_sd", np.nan)
        if t_amp is not None and np.isfinite(float(t_amp_sd)):
            max_allowed_sd = max(_T_AXIS_MAX_AMP_SD_MV, _T_AXIS_MAX_AMP_SD_RATIO * abs(t_amp))
            if float(t_amp_sd) > max_allowed_sd:
                result[lead] = None
                continue

        result[lead] = float(value)
    return result


def _mask_hard_excluded_t_axis_values(
    t_axis_values: Dict[str, Optional[float]],
    excluded_leads: Dict[str, str],
) -> "Dict[str, Optional[float]]":
    result: Dict[str, Optional[float]] = {}
    for lead in HEXAXIAL_ANGLES:
        if excluded_leads.get(lead) in _T_AXIS_HARD_EXCLUSION_REASONS:
            result[lead] = None
            continue
        value = t_axis_values.get(lead)
        result[lead] = float(value) if value is not None and np.isfinite(float(value)) else None
    return result


def _axis_value_count(values: Dict[str, Optional[float]]) -> int:
    return sum(
        1
        for lead in HEXAXIAL_ANGLES
        if values.get(lead) is not None and np.isfinite(float(values[lead]))
    )


def _high_confidence_t_axis_value_count(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    values: Dict[str, Optional[float]],
) -> int:
    count = 0
    for lead in HEXAXIAL_ANGLES:
        if values.get(lead) is None:
            continue
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        confidence = rep.params.get("qt_confidence_mean")
        if confidence is not None and np.isfinite(float(confidence)):
            if float(confidence) >= _T_AXIS_WIDE_HIGH_CONFIDENCE:
                count += 1
    return count


def _has_post_st_t_axis_support(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    t_axis_values: Dict[str, Optional[float]],
    min_leads: int,
) -> bool:
    supported = 0
    for lead in HEXAXIAL_ANGLES:
        if t_axis_values.get(lead) is None:
            continue
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        st_mid = _lead_float(rep, "st_mid_mv")
        if t_amp is None or st_mid is None:
            continue
        post_st = abs(t_amp - st_mid)
        if (
            post_st >= _T_AXIS_POST_ST_MIN_ABS_MV
            and post_st >= _T_AXIS_POST_ST_MIN_AMP_RATIO * max(abs(t_amp), 1e-6)
        ):
            supported += 1
    return supported >= min_leads


def _rr_cv(r_locs: np.ndarray, fs: int) -> float:
    if len(r_locs) < 4:
        return 0.0
    rr_ms = np.diff(r_locs) * 1000.0 / fs
    if len(rr_ms) == 0:
        return 0.0
    return float(np.std(rr_ms) / (np.median(rr_ms) + 1e-6))


def _should_suppress_t_axis(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    t_axis_values: Dict[str, Optional[float]],
    t_axis: Optional[float],
    qrs_axis: Optional[float],
    qrs_ms: Optional[float],
    r_locs: np.ndarray,
    fs: int,
    used_st_corrected_axis: bool = False,
    paced: bool = False,
) -> bool:
    if t_axis is None:
        return False

    value_count = _axis_value_count(t_axis_values)
    if value_count < _T_AXIS_MIN_LEADS:
        return True

    high_conf_count = _high_confidence_t_axis_value_count(representative_leads, t_axis_values)
    if high_conf_count < 1:
        return True

    if (
        _rr_cv(r_locs, fs) > _T_AXIS_IRREGULAR_RR_CV
        and sum(1 for rep in representative_leads.values() if _has_p_support(rep))
            < _T_AXIS_IRREGULAR_MAX_VISIBLE_P_LEADS
    ):
        return True

    if qrs_ms is not None and qrs_ms >= _T_AXIS_WIDE_QRS_SUPPRESSION_MS:
        wide_min_leads = _T_ST_AXIS_MIN_LEADS if used_st_corrected_axis else _T_AXIS_WIDE_MIN_LEADS
        if value_count < wide_min_leads:
            return True
        # Pacing spikes systematically lower QT confidence.  Require only 2
        # high-confidence leads (instead of wide_min_leads=3) in paced rhythm.
        high_conf_min = _T_ST_AXIS_MIN_LEADS if paced else wide_min_leads
        if high_conf_count < high_conf_min:
            return True
        if not _has_post_st_t_axis_support(
            representative_leads,
            t_axis_values,
            min_leads=wide_min_leads,
        ):
            return True
        # In confirmed paced rhythm the T-wave is expected to be discordant with the
        # QRS (LBBB-like secondary repolarisation), so extreme T-axis and large
        # QRS-T angle are normal findings rather than artefacts.
        if not paced:
            if abs(((float(t_axis) + 180.0) % 360.0) - 180.0) >= _T_AXIS_WIDE_EXTREME_SUPPRESS_DEG:
                return True
            qrs_t_angle = _axis_delta_deg(qrs_axis, t_axis)
            if (
                qrs_t_angle is not None
                and qrs_t_angle >= _T_AXIS_WIDE_QRS_T_ANGLE_SUPPRESS_DEG
            ):
                return True

    return False


def _stable_limb_t_axis_fallback_values(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    qrs_ms: Optional[float],
    r_locs: np.ndarray,
    fs: int,
    atrial_invalid: bool,
) -> Optional[Dict[str, Optional[float]]]:
    """Use limb signed T area when QT-confidence gates hide a stable T vector."""
    if atrial_invalid:
        return None
    if qrs_ms is None or qrs_ms > _T_AXIS_STABLE_LIMB_MAX_QRS_MS:
        return None
    if _rr_cv(r_locs, fs) > _T_AXIS_STABLE_LIMB_MAX_RR_CV:
        return None

    raw_values = _signed_wave_area(
        representative_leads,
        "t_area",
        "t_amp_mv",
        reliable_key="reliable_for_t",
        confidence_key="qt_confidence_mean",
        min_confidence=0.0,
        min_abs_amp_mv=_T_AXIS_RELAXED_MIN_ABS_AMP_MV,
        signed_area_key="t_signed_area",
    )
    values: Dict[str, Optional[float]] = {}
    for lead in HEXAXIAL_ANGLES:
        value = raw_values.get(lead)
        if (
            value is not None
            and np.isfinite(float(value))
            and abs(float(value)) >= _T_AXIS_STABLE_LIMB_MIN_SIGNED_AREA
        ):
            values[lead] = float(value)
        else:
            values[lead] = None
    values = _filter_t_axis_values_for_reporting(representative_leads, values)
    if _axis_value_count(values) < _T_AXIS_STABLE_LIMB_MIN_LEADS:
        return None
    clustered = compute_t_axis_from_cluster(
        representative_leads,
        min_leads=_T_AXIS_STABLE_LIMB_MIN_LEADS,
        min_abs_amp_mv=_T_AXIS_RELAXED_MIN_ABS_AMP_MV,
        min_confidence=0.0,
        allowed_leads={lead for lead, value in values.items() if value is not None},
    )
    if clustered.axis_deg is None:
        return None
    return {
        lead: clustered.used_leads.get(lead)
        for lead in HEXAXIAL_ANGLES
    }


def _low_support_limb_t_axis_coverage_values(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    qrs_ms: Optional[float],
    r_locs: np.ndarray,
    fs: int,
    atrial_invalid: bool,
) -> Optional[Dict[str, Optional[float]]]:
    if atrial_invalid:
        return None
    if qrs_ms is None or qrs_ms > _T_AXIS_STABLE_LIMB_MAX_QRS_MS:
        return None
    if _rr_cv(r_locs, fs) > _T_AXIS_STABLE_LIMB_MAX_RR_CV:
        return None

    raw_values = _signed_wave_area(
        representative_leads,
        "t_area",
        "t_amp_mv",
        reliable_key="reliable_for_t",
        confidence_key="qt_confidence_mean",
        min_confidence=0.0,
        min_abs_amp_mv=_T_AXIS_LOW_SUPPORT_COVERAGE_MIN_ABS_MV,
        signed_area_key="t_signed_area",
    )
    values: Dict[str, Optional[float]] = {}
    for lead in HEXAXIAL_ANGLES:
        rep = representative_leads.get(lead)
        value = raw_values.get(lead)
        if rep is None or value is None or not np.isfinite(float(value)):
            values[lead] = None
            continue
        confidence = _lead_float(rep, "qt_confidence_mean")
        if confidence is None or confidence < _T_AXIS_LOW_SUPPORT_COVERAGE_MIN_CONFIDENCE:
            values[lead] = None
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _T_AXIS_LOW_SUPPORT_COVERAGE_MIN_ABS_MV:
            values[lead] = None
            continue
        if bool(rep.params.get("st_t_confusion", False)):
            st_confidence = _lead_float(rep, "twelve_sl_st_confidence") or 0.0
            st_mid = _lead_float(rep, "st_mid_mv")
            if (
                st_confidence < _T_AXIS_LOW_SUPPORT_ST_CONFIDENCE_MIN
                or st_mid is None
                or abs(float(t_amp) - float(st_mid)) < _T_AXIS_LOW_SUPPORT_POST_ST_MIN_ABS_MV
            ):
                values[lead] = None
                continue
        t_amp_sd = rep.variance.get("t_amp_mv_sd", np.nan)
        if np.isfinite(float(t_amp_sd)):
            max_allowed_sd = max(_T_AXIS_MAX_AMP_SD_MV, _T_AXIS_MAX_AMP_SD_RATIO * abs(t_amp))
            if float(t_amp_sd) > max_allowed_sd:
                values[lead] = None
                continue
        if abs(float(value)) < _T_AXIS_LOW_SUPPORT_COVERAGE_MIN_AREA:
            values[lead] = None
            continue
        values[lead] = float(value)

    values = _filter_t_axis_values_for_reporting(representative_leads, values)
    if _axis_value_count(values) < _T_AXIS_LOW_SUPPORT_COVERAGE_MIN_LEADS:
        return None
    confidences = {
        lead: representative_leads[lead].params.get("qt_confidence_mean")
        for lead in HEXAXIAL_ANGLES
        if lead in representative_leads
    }
    axis = _axis_from_amplitudes(values, confidences)
    if axis is None:
        return None
    axis_std = ((float(axis) + 180.0) % 360.0) - 180.0
    if abs(axis_std) > _T_AXIS_LOW_SUPPORT_COVERAGE_MAX_ABS_AXIS_DEG:
        return None
    return values


def _detected_limb_t_axis_coverage_values(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    qrs_ms: Optional[float],
    r_locs: np.ndarray,
    fs: int,
    atrial_invalid: bool,
) -> Optional[Dict[str, Optional[float]]]:
    if atrial_invalid:
        return None
    if qrs_ms is None or qrs_ms > _T_AXIS_STABLE_LIMB_MAX_QRS_MS:
        return None
    if _rr_cv(r_locs, fs) > _T_AXIS_STABLE_LIMB_MAX_RR_CV:
        return None

    raw_values = _signed_wave_area(
        representative_leads,
        "t_area",
        "t_amp_mv",
        reliable_key="reliable_for_t",
        confidence_key="qt_confidence_mean",
        min_confidence=0.0,
        min_abs_amp_mv=_T_AXIS_DETECTED_COVERAGE_MIN_ABS_MV,
        signed_area_key="t_signed_area",
    )
    values: Dict[str, Optional[float]] = {}
    for lead in HEXAXIAL_ANGLES:
        rep = representative_leads.get(lead)
        value = raw_values.get(lead)
        if rep is None or value is None or not np.isfinite(float(value)):
            values[lead] = None
            continue
        confidence = _lead_float(rep, "qt_confidence_mean")
        if confidence is None or confidence < _T_AXIS_DETECTED_COVERAGE_MIN_CONFIDENCE:
            values[lead] = None
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _T_AXIS_DETECTED_COVERAGE_MIN_ABS_MV:
            values[lead] = None
            continue
        if abs(float(value)) < _T_AXIS_DETECTED_COVERAGE_MIN_AREA:
            values[lead] = None
            continue
        if bool(rep.params.get("st_t_confusion", False)):
            st_confidence = _lead_float(rep, "twelve_sl_st_confidence") or 0.0
            st_mid = _lead_float(rep, "st_mid_mv")
            if (
                st_confidence < _T_AXIS_DETECTED_COVERAGE_ST_CONFIDENCE_MIN
                or st_mid is None
                or abs(float(t_amp) - float(st_mid)) < _T_AXIS_DETECTED_COVERAGE_POST_ST_MIN_ABS_MV
            ):
                values[lead] = None
                continue
        t_amp_sd = rep.variance.get("t_amp_mv_sd", np.nan)
        if np.isfinite(float(t_amp_sd)):
            max_allowed_sd = max(_T_AXIS_MAX_AMP_SD_MV, _T_AXIS_MAX_AMP_SD_RATIO * abs(t_amp))
            if float(t_amp_sd) > max_allowed_sd:
                values[lead] = None
                continue
        values[lead] = float(value)

    values = _filter_t_axis_values_for_reporting(representative_leads, values)
    values = _filter_limb_seed_t_axis_conflicts(
        representative_leads,
        values,
        qrs_ms,
        _rr_cv(r_locs, fs),
    )
    if _axis_value_count(values) < _T_AXIS_DETECTED_COVERAGE_MIN_LEADS:
        return None
    confidences = {
        lead: representative_leads[lead].params.get("qt_confidence_mean")
        for lead in HEXAXIAL_ANGLES
        if lead in representative_leads
    }
    axis = _axis_from_amplitudes(values, confidences)
    if axis is None:
        return None
    axis_std = ((float(axis) + 180.0) % 360.0) - 180.0
    if abs(axis_std) > _T_AXIS_DETECTED_COVERAGE_MAX_ABS_AXIS_DEG:
        return None
    return values


def _st_supported_limb_t_axis_coverage_values(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
    qrs_ms: Optional[float],
    r_locs: np.ndarray,
    fs: int,
    atrial_invalid: bool,
) -> Optional[Dict[str, Optional[float]]]:
    if atrial_invalid:
        return None
    if qrs_ms is None or qrs_ms > _T_AXIS_STABLE_LIMB_MAX_QRS_MS:
        return None
    if _rr_cv(r_locs, fs) > _T_AXIS_STABLE_LIMB_MAX_RR_CV:
        return None

    raw_values = _signed_wave_area(
        representative_leads,
        "t_area",
        "t_amp_mv",
        reliable_key="reliable_for_t",
        confidence_key="qt_confidence_mean",
        min_confidence=0.0,
        min_abs_amp_mv=_T_AXIS_ST_SUPPORTED_COVERAGE_MIN_ABS_MV,
        signed_area_key="t_signed_area",
    )
    values: Dict[str, Optional[float]] = {}
    for lead in HEXAXIAL_ANGLES:
        rep = representative_leads.get(lead)
        value = raw_values.get(lead)
        if rep is None or value is None or not np.isfinite(float(value)):
            values[lead] = None
            continue
        confidence = _lead_float(rep, "qt_confidence_mean")
        if confidence is None or confidence < _T_AXIS_ST_SUPPORTED_COVERAGE_MIN_CONFIDENCE:
            values[lead] = None
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        area = _lead_float(rep, "t_area")
        if t_amp is None or abs(t_amp) < _T_AXIS_ST_SUPPORTED_COVERAGE_MIN_ABS_MV:
            values[lead] = None
            continue
        if area is None or abs(area) < 1e-6:
            values[lead] = None
            continue
        if abs(float(value)) < _T_AXIS_ST_SUPPORTED_COVERAGE_MIN_AREA:
            values[lead] = None
            continue
        if abs(float(value)) / abs(float(area)) < _T_AXIS_ST_SUPPORTED_COVERAGE_MIN_SIGNED_RATIO:
            values[lead] = None
            continue
        if _has_t_prime_special_t_axis_conflict(rep, float(value)):
            values[lead] = None
            continue
        if bool(rep.params.get("st_t_confusion", False)):
            st_confidence = _lead_float(rep, "twelve_sl_st_confidence") or 0.0
            if st_confidence < _T_AXIS_ST_SUPPORTED_COVERAGE_ST_CONFIDENCE_MIN:
                values[lead] = None
                continue
        t_amp_sd = rep.variance.get("t_amp_mv_sd", np.nan)
        if np.isfinite(float(t_amp_sd)):
            max_allowed_sd = max(_T_AXIS_MAX_AMP_SD_MV, _T_AXIS_MAX_AMP_SD_RATIO * abs(t_amp))
            if float(t_amp_sd) > max_allowed_sd:
                values[lead] = None
                continue
        values[lead] = float(value)

    values = _filter_limb_seed_t_axis_conflicts(
        representative_leads,
        values,
        qrs_ms,
        _rr_cv(r_locs, fs),
    )
    if _axis_value_count(values) < _T_AXIS_ST_SUPPORTED_COVERAGE_MIN_LEADS:
        return None
    confidences = {
        lead: representative_leads[lead].params.get("qt_confidence_mean")
        for lead in HEXAXIAL_ANGLES
        if lead in representative_leads
    }
    axis = _axis_from_amplitudes(values, confidences)
    if axis is None:
        return None
    axis_std = ((float(axis) + 180.0) % 360.0) - 180.0
    if abs(axis_std) > _T_AXIS_ST_SUPPORTED_COVERAGE_MAX_ABS_AXIS_DEG:
        return None
    return values


def _stable_limb_signed_t_axis_deg(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
) -> Optional[float]:
    raw_values = _signed_wave_area(
        representative_leads,
        "t_area",
        "t_amp_mv",
        reliable_key="reliable_for_t",
        confidence_key="qt_confidence_mean",
        min_confidence=0.0,
        min_abs_amp_mv=_T_AXIS_RELAXED_MIN_ABS_AMP_MV,
        signed_area_key="t_signed_area",
    )
    values = {
        lead: raw_values.get(lead)
        for lead in HEXAXIAL_ANGLES
    }
    values = _filter_t_axis_values_for_reporting(representative_leads, values)
    finite_values = [
        float(value)
        for value in values.values()
        if value is not None and np.isfinite(float(value))
    ]
    if len(finite_values) < _T_AXIS_STABLE_LIMB_MIN_LEADS:
        return None
    if not (any(value > 0.0 for value in finite_values) and any(value < 0.0 for value in finite_values)):
        return None
    t_confs = {lead: rep.params.get("qt_confidence_mean") for lead, rep in representative_leads.items()}
    return _axis_from_amplitudes(values, t_confs)


def _qrs_net_amplitude(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
) -> "Dict[str, Optional[float]]":
    """
    Net QRS deflection (signed) for QRS axis calculation (DXL convention).

    r_amp_mv (always ≥ 0) + q_amp_mv (≤ 0) + s_amp_mv (≤ 0) gives the signed
    net QRS deflection, which is positive in leads where R is dominant and
    negative where S/Q dominate (e.g. aVR, V1 in normal axis).
    Replaces the unsigned qrs_area which incorrectly treats all leads as
    contributing in the positive direction.
    """
    result: Dict[str, Optional[float]] = {}
    for k, v in representative_leads.items():
        signed_area = v.params.get("qrs_signed_area")
        if signed_area is not None and np.isfinite(float(signed_area)):
            result[k] = float(signed_area)
            continue
        r = v.params.get("r_amp_mv")
        q = v.params.get("q_amp_mv")
        s = v.params.get("s_amp_mv")
        if r is None or not np.isfinite(float(r)):
            result[k] = None
        else:
            net = float(r) + (float(q) if (q is not None and np.isfinite(float(q))) else 0.0) \
                           + (float(s) if (s is not None and np.isfinite(float(s))) else 0.0)
            result[k] = net
    return result


def _qrs_peak_net_amplitude(
    representative_leads: "Dict[str, RepresentativeLeadFeatures]",
) -> "Dict[str, Optional[float]]":
    result: Dict[str, Optional[float]] = {}
    for lead, rep in representative_leads.items():
        r = rep.params.get("r_amp_mv")
        if r is None or not np.isfinite(float(r)):
            result[lead] = None
            continue
        q = rep.params.get("q_amp_mv")
        s = rep.params.get("s_amp_mv")
        result[lead] = (
            float(r)
            + (float(q) if q is not None and np.isfinite(float(q)) else 0.0)
            + (float(s) if s is not None and np.isfinite(float(s)) else 0.0)
        )
    return result


def _should_use_peak_net_qrs_axis(
    signed_values: Dict[str, Optional[float]],
    peak_values: Dict[str, Optional[float]],
    qrs_ms: Optional[float],
) -> bool:
    if qrs_ms is None or qrs_ms < _WIDE_QRS_MS:
        return False

    comparisons = 0
    disagreements = 0
    for lead in HEXAXIAL_ANGLES:
        signed_value = signed_values.get(lead)
        peak_value = peak_values.get(lead)
        if signed_value is None or peak_value is None:
            continue
        if not np.isfinite(float(signed_value)) or not np.isfinite(float(peak_value)):
            continue
        if abs(float(signed_value)) < 1e-6 or abs(float(peak_value)) < 0.02:
            continue
        comparisons += 1
        if np.sign(float(signed_value)) != np.sign(float(peak_value)):
            disagreements += 1

    return (
        comparisons >= _QRS_AXIS_MIN_POLARITY_COMPARISONS
        and disagreements <= _QRS_AXIS_MAX_POLARITY_DISAGREEMENTS
    )


def _select_reliable_qt_leads(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qt_variance_threshold_ms: float = 25.0,
    qt_confidence_threshold: float  = 0.20,
) -> List[str]:
    """
    Return lead names reliable enough for global QT computation. (T021)

    Conditions (all must hold):
    1. Wave-specific quality gate: reliable_for_qt == True
    2. Beat-to-beat QT SD < qt_variance_threshold_ms  (DXL: "low variance" ≤ 25 ms)
    3. Mean geometric T-end confidence > qt_confidence_threshold
    4. Median QT within physiological range [240, 700] ms

    Rescue pass: when no lead survives strict criteria (wide QRS, ischemia,
    pacing), a relaxed second pass (SD < 50 ms, confidence > 0.05, sorted by
    lowest SD) returns up to 2 best-available leads so QT is never fully NA.
    """
    reliable: List[str] = []
    for lead, rep in representative_leads.items():
        if not rep.params.get("reliable_for_qt", False):
            continue
        if not _has_visible_t_for_qt(rep):
            continue
        qt_sd = rep.variance.get("qt_ms_sd", np.nan)
        if np.isnan(qt_sd) or qt_sd > qt_variance_threshold_ms:
            continue
        conf = rep.params.get("qt_confidence_mean") or 0.0
        if conf < qt_confidence_threshold:
            continue
        qt_val = (
            _valid_measurement("qt_consensus_ms", rep.params.get("qt_consensus_ms"))
            or _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        )
        if qt_val is None:
            continue
        reliable.append(lead)

    if reliable:
        return _filter_qt_cluster(representative_leads, reliable)

    # Rescue pass: complex morphologies (BBB, ischemia, pacing) often yield 0
    # reliable leads under strict criteria.  Return only the single best lead
    # (lowest beat-to-beat QT SD) with tightened thresholds so a genuinely poor
    # measurement never overrides the existing preferred-lead fallback.
    rescue_leads: List[str] = []
    rescue_sds: List[float] = []
    for lead, rep in representative_leads.items():
        if not _has_visible_t_for_qt(rep):
            continue
        qt_val = (
            _valid_measurement("qt_consensus_ms", rep.params.get("qt_consensus_ms"))
            or _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        )
        if qt_val is None:
            continue
        qt_sd = rep.variance.get("qt_ms_sd", np.nan)
        if np.isnan(qt_sd) or qt_sd > 30.0:
            continue
        conf = rep.params.get("qt_confidence_mean") or 0.0
        if conf < 0.10:
            continue
        rescue_leads.append(lead)
        rescue_sds.append(float(qt_sd))

    if rescue_leads:
        order = np.argsort(rescue_sds)
        return [rescue_leads[int(order[0])]]
    return []


def _lead_float(rep: "RepresentativeLeadFeatures", key: str) -> Optional[float]:
    value = rep.params.get(key)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _has_visible_t_for_qt(rep: "RepresentativeLeadFeatures") -> bool:
    t_amp = _lead_float(rep, "t_amp_mv")
    return t_amp is not None and abs(t_amp) >= _T_MIN_QT_ABS_AMP_MV


def _rep_qt_value(rep: "RepresentativeLeadFeatures") -> Optional[float]:
    return (
        _valid_measurement("qt_consensus_ms", rep.params.get("qt_consensus_ms"))
        or _valid_measurement("qt_ms", rep.params.get("qt_ms"))
    )


def _qt_value_allowed_for_qrs_width(qt_ms: Optional[float], qrs_ms: Optional[float]) -> bool:
    if qt_ms is None:
        return False
    if qrs_ms is None or qrs_ms < _WIDE_QRS_MS:
        return True
    jt = float(qt_ms) - float(qrs_ms)
    return _WIDE_QRS_MIN_JT_MS <= jt <= _WIDE_QRS_MAX_JT_MS


def _rep_qt_value_for_qrs_width(
    rep: "RepresentativeLeadFeatures",
    qrs_ms: Optional[float],
) -> Optional[float]:
    candidates = [
        _valid_measurement("qt_consensus_ms", rep.params.get("qt_consensus_ms")),
        _valid_measurement("qt_ms", rep.params.get("qt_ms")),
    ]
    for qt in candidates:
        if _qt_value_allowed_for_qrs_width(qt, qrs_ms):
            return qt
    return None


def _rep_qrs_value(rep: "RepresentativeLeadFeatures") -> Optional[float]:
    return (
        _valid_measurement("qrs_consensus_ms", rep.params.get("qrs_consensus_ms"))
        or _valid_measurement("qrs_ms", rep.params.get("qrs_ms"))
    )


def _rep_jt_value(rep: "RepresentativeLeadFeatures") -> Optional[float]:
    jt = _valid_measurement("jt_ms", rep.params.get("jt_ms"))
    if jt is not None:
        return jt
    qt = _rep_qt_value(rep)
    qrs = _rep_qrs_value(rep)
    if qt is None or qrs is None:
        return None
    return qt - qrs


def _filter_qt_cluster(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    leads: List[str],
) -> List[str]:
    if len(leads) < 3:
        return leads
    values: Dict[str, float] = {}
    for lead in leads:
        qt = _rep_qt_value(representative_leads[lead])
        if qt is not None:
            values[lead] = qt
    if len(values) < 3:
        return leads
    center = float(np.median(list(values.values())))
    kept = [
        lead
        for lead in leads
        if lead in values and abs(values[lead] - center) <= _QT_CLUSTER_OUTLIER_MS
    ]
    return kept if len(kept) >= 2 else leads


def _filter_wide_qrs_qt_leads(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    leads: List[str],
    qrs_ms: Optional[float],
) -> List[str]:
    if qrs_ms is None or qrs_ms < _WIDE_QRS_MS:
        return leads
    kept: List[str] = []
    for lead in leads:
        if _qt_allowed_for_qrs_width(representative_leads[lead], qrs_ms):
            kept.append(lead)
    return kept


def _qt_allowed_for_qrs_width(
    rep: "RepresentativeLeadFeatures",
    qrs_ms: Optional[float],
) -> bool:
    return _rep_qt_value_for_qrs_width(rep, qrs_ms) is not None


def _weighted_median_qt(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    reliable_leads: List[str],
    qrs_ms: Optional[float] = None,
) -> Optional[float]:
    """
    T017: Amplitude-weighted QT fusion across reliable leads.

    Weight = |T_amp| × qt_confidence × (1 / (qt_sd + 5))
    Produces weighted median: leads with larger T-wave and lower variance
    dominate the final global QT estimate.
    """
    qt_vals: List[float] = []
    weights: List[float] = []
    for lead in reliable_leads:
        rep = representative_leads[lead]
        qt = _rep_qt_value_for_qrs_width(rep, qrs_ms)
        if qt is None:
            continue
        t_amp = rep.params.get("t_amp_mv")
        t_amp_w = abs(float(t_amp)) if t_amp is not None and np.isfinite(t_amp) else 0.05
        conf = float(rep.params.get("qt_confidence_mean") or 0.0)
        qt_sd = rep.variance.get("qt_ms_sd", np.nan)
        sd_w = 1.0 / (float(qt_sd) + 5.0) if np.isfinite(qt_sd) else 0.1
        weight = max(1e-6, t_amp_w * (conf + 0.1) * sd_w)
        qt_vals.append(float(qt))
        weights.append(weight)

    if not qt_vals:
        return None

    # Weighted median: sort by value, find cumulative weight midpoint
    order = np.argsort(qt_vals)
    sorted_qt = np.array(qt_vals)[order]
    sorted_w  = np.array(weights)[order]
    cumw = np.cumsum(sorted_w)
    mid  = cumw[-1] / 2.0
    idx  = int(np.searchsorted(cumw, mid))
    return float(sorted_qt[min(idx, len(sorted_qt) - 1)])


def _wide_qrs_raw_qt_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    reliable_leads: List[str],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
) -> Optional[float]:
    if qrs_ms is None or qrs_ms < _WIDE_QRS_MS or consensus_qt_ms is None:
        return None
    raw_qt_values: List[float] = []
    for lead in reliable_leads:
        rep = representative_leads[lead]
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        raw_qt_values.append(float(raw_qt))
    if len(raw_qt_values) < 2:
        return None
    spread = float(np.max(raw_qt_values) - np.min(raw_qt_values))
    if spread > _WIDE_QRS_RAW_QT_RESCUE_MAX_SPREAD_MS:
        return None
    raw_median = float(np.median(raw_qt_values))
    if raw_median - float(consensus_qt_ms) < _WIDE_QRS_RAW_QT_RESCUE_MIN_DELTA_MS:
        return None
    return raw_median


def _wide_qrs_long_raw_qt_core_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
) -> Optional[tuple[float, List[str]]]:
    if (
        qrs_ms is None
        or qrs_ms < _WIDE_QRS_MS
        or consensus_qt_ms is None
        or consensus_qt_ms < _WIDE_QRS_LONG_QT_RAW_CORE_MIN_QT_MS
    ):
        return None

    values: List[tuple[str, float]] = []
    for lead in STANDARD_12_LEADS:
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        if not rep.params.get("reliable_for_qt", False):
            continue
        if bool(rep.params.get("st_t_confusion", False)):
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _WIDE_QRS_LONG_QT_RAW_CORE_MIN_ABS_AMP_MV:
            continue
        confidence = _lead_float(rep, "qt_confidence_mean") or 0.0
        if confidence < _WIDE_QRS_LONG_QT_RAW_CORE_MIN_CONFIDENCE:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        values.append((lead, float(raw_qt)))

    if len(values) < _WIDE_QRS_LONG_QT_RAW_CORE_MIN_SUPPORT:
        return None

    values.sort(key=lambda item: item[1])
    candidates: List[List[tuple[str, float]]] = []
    for idx, (_lead, start) in enumerate(values):
        cluster = [
            item
            for item in values[idx:]
            if item[1] - start <= _WIDE_QRS_LONG_QT_RAW_CORE_CLUSTER_MS
        ]
        if len(cluster) < _WIDE_QRS_LONG_QT_RAW_CORE_MIN_SUPPORT:
            continue
        median = float(np.median([value for _lead_name, value in cluster]))
        if float(consensus_qt_ms) - median < _WIDE_QRS_LONG_QT_RAW_CORE_MIN_DELTA_MS:
            continue
        candidates.append(cluster)

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda cluster: (
            len(cluster),
            float(np.median([value for _lead, value in cluster])),
            -float(np.ptp([value for _lead, value in cluster])),
        ),
    )
    lead_order = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    return (
        float(np.median([value for _lead, value in best])),
        [lead for lead, _value in sorted(best, key=lambda item: lead_order.get(item[0], 999))],
    )


def _paced_low_support_raw_qt_fallback(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
) -> Optional[tuple[float, List[str]]]:
    if qrs_ms is None or qrs_ms < _WIDE_QRS_MS:
        return None

    values: List[tuple[str, float]] = []
    for lead, rep in representative_leads.items():
        if not rep.params.get("reliable_for_qt", False):
            continue
        if not rep.params.get("reliable_for_global", False):
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _RAW_QT_CORE_IRREGULAR_MIN_ABS_AMP_MV:
            continue
        conf = float(rep.params.get("qt_confidence_mean") or 0.0)
        if conf < _PACED_LOW_SUPPORT_RAW_QT_MIN_CONFIDENCE:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        values.append((lead, float(raw_qt)))

    if len(values) < _RAW_QT_CORE_MIN_LEADS:
        return None

    values.sort(key=lambda item: item[1])
    candidates: List[List[tuple[str, float]]] = []
    for idx, (_, start) in enumerate(values):
        cluster = [
            item for item in values[idx:]
            if item[1] - start <= _PACED_LOW_SUPPORT_RAW_QT_CLUSTER_MS
        ]
        if len(cluster) >= _RAW_QT_CORE_MIN_LEADS:
            candidates.append(cluster)
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda cluster: (len(cluster), -float(np.ptp([value for _, value in cluster]))),
    )
    return float(np.median([value for _, value in best])), [lead for lead, _ in best]


def _rr_cv_from_locs(r_locs: np.ndarray) -> Optional[float]:
    rr = np.diff(r_locs).astype(float) if len(r_locs) > 2 else np.asarray([])
    if len(rr) < 2:
        return None
    median = float(np.median(rr))
    if median <= 0.0 or not np.isfinite(median):
        return None
    return float(np.std(rr) / median)


def _raw_qt_core_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
    rr_cv: Optional[float],
) -> Optional[float]:
    if consensus_qt_ms is None:
        return None

    is_irregular = rr_cv is not None and rr_cv >= _AF_QT_RESCUE_RR_CV
    min_abs_t_amp = (
        _RAW_QT_CORE_IRREGULAR_MIN_ABS_AMP_MV
        if is_irregular
        else _T_MIN_QT_ABS_AMP_MV
    )
    raw_values: List[float] = []
    for rep in representative_leads.values():
        if not rep.params.get("reliable_for_qt", False):
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < min_abs_t_amp:
            continue
        conf = float(rep.params.get("qt_confidence_mean") or 0.0)
        if conf < _RAW_QT_CORE_MIN_CONFIDENCE:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        raw_values.append(float(raw_qt))

    if len(raw_values) < _RAW_QT_CORE_MIN_LEADS:
        return None

    values = sorted(raw_values)
    candidates: List[tuple[int, float]] = []
    for i, start in enumerate(values):
        cluster = [
            value
            for value in values[i:]
            if value - start <= _RAW_QT_CORE_CLUSTER_MS
        ]
        if len(cluster) < _RAW_QT_CORE_MIN_LEADS:
            continue
        if not is_irregular and len(cluster) < _RAW_QT_CORE_REGULAR_MIN_LEADS:
            continue
        median = float(np.median(cluster))
        delta = float(consensus_qt_ms) - median
        if delta < _RAW_QT_CORE_MIN_DELTA_MS:
            continue
        if not is_irregular and delta < _RAW_QT_CORE_ACTIVATION_DELTA_MS:
            continue
        candidates.append((len(cluster), median))

    if not candidates:
        return None

    # Prefer the best-supported lower/core cluster, breaking ties toward the
    # less-short QT to avoid overcorrecting toward an early T-end outlier.
    _, median = max(candidates, key=lambda item: (item[0], item[1]))
    return median


def _reliable_raw_qt_cluster_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    reliable_leads: List[str],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
) -> Optional[float]:
    if consensus_qt_ms is None:
        return None
    is_wide_qrs = qrs_ms is not None and qrs_ms >= _WIDE_QRS_MS
    min_reliable_leads = 2 if is_wide_qrs else 3
    if len(reliable_leads) < min_reliable_leads:
        return None
    if len(reliable_leads) > 6:
        return None

    consensus_values = [
        value
        for lead in reliable_leads
        if (value := _valid_measurement(
            "qt_consensus_ms",
            representative_leads[lead].params.get("qt_consensus_ms"),
        )) is not None
    ]
    if len(consensus_values) < min_reliable_leads:
        return None
    if float(np.max(consensus_values) - np.min(consensus_values)) > 6.0:
        return None

    raw_values: List[float] = []
    for lead in reliable_leads:
        raw_qt = _valid_measurement("qt_ms", representative_leads[lead].params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        raw_values.append(float(raw_qt))
    if len(raw_values) < min_reliable_leads:
        return None

    values = sorted(raw_values)
    min_support = max(min_reliable_leads, int(np.ceil(0.60 * len(values))))
    candidates: List[tuple[int, float]] = []
    for i, start in enumerate(values):
        cluster = [
            value
            for value in values[i:]
            if value - start <= 25.0
        ]
        if len(cluster) < min_support:
            continue
        median = float(np.median(cluster))
        if float(consensus_qt_ms) - median < 10.0:
            continue
        candidates.append((len(cluster), median))

    if not candidates:
        return None
    _, median = max(candidates, key=lambda item: (item[0], item[1]))
    return median


def _late_raw_qt_cluster_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    reliable_leads: List[str],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
    rr_cv: Optional[float],
) -> Optional[tuple[float, List[str]]]:
    if consensus_qt_ms is None:
        return None
    if qrs_ms is not None and qrs_ms >= _WIDE_QRS_MS:
        return None
    if rr_cv is not None and rr_cv >= _AF_QT_RESCUE_RR_CV:
        return None

    values: List[tuple[str, float]] = []
    for lead in reliable_leads:
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        if bool(rep.params.get("st_t_confusion", False)):
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _T_MIN_QT_ABS_AMP_MV:
            continue
        conf = float(rep.params.get("qt_confidence_mean") or 0.0)
        if conf < _T_MIN_CONFIDENCE:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        values.append((lead, float(raw_qt)))

    if len(values) < _LATE_RAW_QT_CLUSTER_MIN_SUPPORT:
        return None

    values.sort(key=lambda item: item[1])
    min_support = max(
        _LATE_RAW_QT_CLUSTER_MIN_SUPPORT,
        int(np.ceil(_LATE_RAW_QT_CLUSTER_MIN_FRACTION * len(values))),
    )
    candidates: List[List[tuple[str, float]]] = []
    for idx, (_lead, start) in enumerate(values):
        cluster = [
            item for item in values[idx:]
            if item[1] - start <= _LATE_RAW_QT_CLUSTER_MS
        ]
        if len(cluster) < min_support:
            continue
        median = float(np.median([value for _lead_name, value in cluster]))
        if median - float(consensus_qt_ms) < _LATE_RAW_QT_CLUSTER_MIN_DELTA_MS:
            continue
        candidates.append(cluster)

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda cluster: (
            len(cluster),
            float(np.median([value for _lead, value in cluster])),
            -float(np.ptp([value for _lead, value in cluster])),
        ),
    )
    lead_order = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    return (
        float(np.median([value for _lead, value in best])),
        [lead for lead, _value in sorted(best, key=lambda item: lead_order.get(item[0], 999))],
    )


def _tail_confirmed_late_raw_qt_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
    rr_cv: Optional[float],
) -> Optional[tuple[float, List[str]]]:
    if consensus_qt_ms is None:
        return None
    if qrs_ms is not None and qrs_ms >= _WIDE_QRS_MS:
        return None
    if rr_cv is not None and rr_cv >= _AF_QT_RESCUE_RR_CV:
        return None

    values: List[tuple[str, float]] = []
    for lead in STANDARD_12_LEADS:
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        if not rep.params.get("reliable_for_t", False):
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _TAIL_CONFIRMED_LATE_QT_MIN_ABS_AMP_MV:
            continue
        confidence = _lead_float(rep, "qt_confidence_mean") or 0.0
        if confidence < _TAIL_CONFIRMED_LATE_QT_MIN_CONFIDENCE:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        if float(raw_qt) - float(consensus_qt_ms) < _TAIL_CONFIRMED_LATE_QT_CLUSTER_MIN_DELTA_MS:
            continue
        values.append((lead, float(raw_qt)))

    if len(values) < _TAIL_CONFIRMED_LATE_QT_MIN_SUPPORT:
        return None

    values.sort(key=lambda item: item[1])
    candidates: List[List[tuple[str, float]]] = []
    for idx, (_lead, start) in enumerate(values):
        cluster = [
            item for item in values[idx:]
            if item[1] - start <= _TAIL_CONFIRMED_LATE_QT_CLUSTER_MS
        ]
        if len(cluster) < _TAIL_CONFIRMED_LATE_QT_MIN_SUPPORT:
            continue
        candidates.append(cluster)
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda cluster: (
            len(cluster),
            float(np.median([value for _lead, value in cluster])),
            -float(np.ptp([value for _lead, value in cluster])),
        ),
    )
    if not any(
        value - float(consensus_qt_ms) >= _TAIL_CONFIRMED_LATE_QT_MIN_DELTA_MS
        for _lead, value in best
    ):
        return None
    best_median = float(np.median([value for _lead, value in best]))
    if (
        len(best) < 4
        and best_median > _TAIL_CONFIRMED_LATE_QT_SMALL_CLUSTER_MAX_MS
    ):
        return None
    if (
        len(best) == _TAIL_CONFIRMED_LATE_QT_MIN_SUPPORT
        and best_median > _TAIL_CONFIRMED_LATE_QT_PAIR_MAX_MS
    ):
        return None
    if (
        len(best) < 4
        and best_median >= _TAIL_CONFIRMED_LATE_QT_WEAK_PRECORDIAL_PAIR_MIN_QT_MS
        and all(lead.startswith("V") for lead, _value in best)
    ):
        for lead, _value in best:
            rep = representative_leads.get(lead)
            t_amp = _lead_float(rep, "t_amp_mv") if rep is not None else None
            if (
                t_amp is None
                or abs(t_amp) < _TAIL_CONFIRMED_LATE_QT_PRECORDIAL_PAIR_MIN_ABS_AMP_MV
            ):
                return None
    lead_order = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    return (
        best_median,
        [lead for lead, _value in sorted(best, key=lambda item: lead_order.get(item[0], 999))],
    )


def _low_support_late_raw_qt_cluster_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
    rr_cv: Optional[float],
) -> Optional[tuple[float, List[str]]]:
    if consensus_qt_ms is None:
        return None
    if qrs_ms is not None and qrs_ms >= _WIDE_QRS_MS:
        return None
    if rr_cv is not None and rr_cv >= _AF_QT_RESCUE_RR_CV:
        return None

    values: List[tuple[str, float]] = []
    for lead, rep in representative_leads.items():
        if lead not in STANDARD_12_LEADS:
            continue
        if not rep.params.get("reliable_for_global", False):
            continue
        if not rep.params.get("reliable_for_qt", False):
            continue
        if bool(rep.params.get("st_t_confusion", False)):
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < _T_MIN_QT_ABS_AMP_MV:
            continue
        confidence = _lead_float(rep, "qt_confidence_mean") or 0.0
        if confidence < _LOW_SUPPORT_LATE_RAW_QT_MIN_CONFIDENCE:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        values.append((lead, float(raw_qt)))

    if len(values) < _LOW_SUPPORT_LATE_RAW_QT_MIN_SUPPORT:
        return None

    values.sort(key=lambda item: item[1])
    candidates: List[List[tuple[str, float]]] = []
    for idx, (_lead, start) in enumerate(values):
        cluster = [
            item for item in values[idx:]
            if item[1] - start <= _LOW_SUPPORT_LATE_RAW_QT_CLUSTER_MS
        ]
        if len(cluster) < _LOW_SUPPORT_LATE_RAW_QT_MIN_SUPPORT:
            continue
        median = float(np.median([value for _lead_name, value in cluster]))
        if median - float(consensus_qt_ms) < _LOW_SUPPORT_LATE_RAW_QT_MIN_DELTA_MS:
            continue
        candidates.append(cluster)

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda cluster: (
            len(cluster),
            float(np.median([value for _lead, value in cluster])),
            -float(np.ptp([value for _lead, value in cluster])),
        ),
    )
    lead_order = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    return (
        float(np.median([value for _lead, value in best])),
        [lead for lead, _value in sorted(best, key=lambda item: lead_order.get(item[0], 999))],
    )


def _raw_qt_late_core_rescue(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
    rr_cv: Optional[float],
) -> Optional[float]:
    if consensus_qt_ms is None:
        return None
    if qrs_ms is not None and qrs_ms >= _WIDE_QRS_MS:
        return None
    if rr_cv is not None and rr_cv >= _AF_QT_RESCUE_RR_CV:
        return None

    raw_values: List[float] = []
    for rep in representative_leads.values():
        if not rep.params.get("reliable_for_global", False):
            continue
        t_amp = _lead_float(rep, "t_amp_mv")
        if t_amp is None or abs(t_amp) < 0.015:
            continue
        conf = float(rep.params.get("qt_confidence_mean") or 0.0)
        if conf < _RAW_QT_CORE_MIN_CONFIDENCE:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        raw_values.append(float(raw_qt))

    if len(raw_values) < _RAW_QT_CORE_MIN_LEADS:
        return None

    values = sorted(raw_values)
    candidates: List[tuple[int, float]] = []
    for i, start in enumerate(values):
        cluster = [
            value
            for value in values[i:]
            if value - start <= _RAW_QT_CORE_CLUSTER_MS
        ]
        if len(cluster) < _RAW_QT_CORE_MIN_LEADS:
            continue
        median = float(np.median(cluster))
        if median - float(consensus_qt_ms) < _RAW_QT_CORE_ACTIVATION_DELTA_MS:
            continue
        candidates.append((len(cluster), median))

    if not candidates:
        return None

    _, median = max(candidates, key=lambda item: (item[0], item[1]))
    return median


def _irregular_beat_qt_core(
    beat_features: List[LeadBeatFeatures],
    rr_cv: Optional[float],
) -> Optional[float]:
    if rr_cv is None or rr_cv < _AF_QT_RESCUE_RR_CV:
        return None

    by_beat: Dict[int, List[float]] = defaultdict(list)
    for bf in beat_features:
        qt = _valid_measurement("qt_consensus_ms", bf.qt_consensus_ms)
        if qt is not None:
            by_beat[int(bf.beat_id)].append(float(qt))

    per_beat = [
        float(np.median(values))
        for values in by_beat.values()
        if values
    ]
    per_beat = [value for value in per_beat if 300.0 <= value <= 520.0]
    if len(per_beat) < 5:
        return None

    q1, q3 = np.percentile(per_beat, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        kept = per_beat
    else:
        lo = float(q1 - 1.5 * iqr)
        hi = float(q3 + 1.5 * iqr)
        kept = [value for value in per_beat if lo <= value <= hi]
    if len(kept) < 5:
        return None

    return float(np.median(kept))


def _irregular_raw_beat_qt_core(
    beat_features: List[LeadBeatFeatures],
    rr_cv: Optional[float],
) -> Optional[float]:
    if rr_cv is None or rr_cv < _AF_QT_RESCUE_RR_CV:
        return None

    by_beat: Dict[int, List[float]] = defaultdict(list)
    for bf in beat_features:
        qt = _valid_measurement("qt_ms", bf.qt_ms)
        if qt is not None:
            by_beat[int(bf.beat_id)].append(float(qt))

    per_beat = [
        float(np.median(values))
        for values in by_beat.values()
        if values
    ]
    core_values = [value for value in per_beat if 240.0 <= value <= 420.0]
    if len(core_values) < 4:
        return None
    return float(np.median(core_values))


def _backfill_representative_qt_from_core(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    *,
    beat_features: List[LeadBeatFeatures],
    qt_ms: Optional[float],
    qrs_ms: Optional[float],
    rr_cv: Optional[float],
    source: Optional[str],
) -> None:
    if qt_ms is None or source != "irregular_raw_beat_qt_core":
        return
    per_lead_core: Dict[str, float] = {}
    if rr_cv is not None and rr_cv >= _AF_QT_RESCUE_RR_CV:
        by_lead: Dict[str, List[float]] = defaultdict(list)
        for bf in beat_features:
            qt = _valid_measurement("qt_ms", bf.qt_ms)
            if qt is not None and 240.0 <= qt <= 430.0:
                by_lead[bf.lead].append(float(qt))
        for lead, values in by_lead.items():
            if len(values) >= 3:
                per_lead_core[lead] = float(np.median(values))

    global_qt_value = float(qt_ms)
    for rep in representative_leads.values():
        if not (
            rep.params.get("reliable_for_qt", False)
            or rep.params.get("reliable_for_global", False)
        ):
            continue
        qt_value = per_lead_core.get(rep.lead, global_qt_value)
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        consensus_qt = _valid_measurement("qt_consensus_ms", rep.params.get("qt_consensus_ms"))
        current_qt = raw_qt if raw_qt is not None else consensus_qt
        if (
            current_qt is not None
            and current_qt - qt_value < _REPRESENTATIVE_QT_CORE_BACKFILL_MIN_DELTA_MS
        ):
            continue

        rep.params["representative_qt_original_ms"] = raw_qt
        rep.params["representative_qt_consensus_original_ms"] = consensus_qt
        rep.params["qt_ms"] = qt_value
        rep.params["qt_consensus_ms"] = qt_value

        qrs_value = _rep_qrs_value(rep)
        if qrs_value is None:
            qrs_value = qrs_ms
        if qrs_value is not None and qt_value > float(qrs_value):
            rep.params["jt_ms"] = qt_value - float(qrs_value)

        rep.params["representative_qt_source"] = source
        rep.params["representative_qt_core_scope"] = (
            "lead_raw_beat_core" if rep.lead in per_lead_core else "global_raw_beat_core"
        )
        rep.params["representative_qt_backfilled"] = True


def _qt_dispersion_summary(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
) -> QTDispersion:
    values: Dict[str, float] = {}
    for lead, rep in representative_leads.items():
        if not rep.params.get("reliable_for_qt", False):
            continue
        if not _has_visible_t_for_qt(rep):
            continue
        conf = rep.params.get("qt_confidence_mean") or 0.0
        if float(conf) < 0.20:
            continue
        qt_sd = rep.variance.get("qt_ms_sd", np.nan)
        if np.isfinite(qt_sd) and float(qt_sd) > 50.0:
            continue
        qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if qt is None or not _qt_value_allowed_for_qrs_width(qt, qrs_ms):
            continue
        values[lead] = float(qt)
    return summarize_qt_dispersion(values)


def _raw_qt_dispersion(
    representative_leads: Dict[str, "RepresentativeLeadFeatures"],
    qrs_ms: Optional[float],
) -> Optional[float]:
    """Full independent-lead range, without compact-cluster truncation."""
    return _qt_dispersion_summary(representative_leads, qrs_ms).independent_ms


_LIMB_LEADS = {"I", "II", "III", "aVR", "aVL", "aVF"}
_PRECORDIAL_LEADS = {"V1", "V2", "V3", "V4", "V5", "V6"}


def _raw_qrs_by_region(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> tuple[List[float], List[float], List[float]]:
    limb_values: List[float] = []
    precordial_values: List[float] = []
    all_values: List[float] = []
    for lead, rep in representative_leads.items():
        if not rep.params.get("reliable_for_qrs"):
            continue
        raw = _valid_measurement("qrs_ms", rep.params.get("qrs_ms"))
        if raw is None:
            continue
        value = float(raw)
        all_values.append(value)
        if lead in _LIMB_LEADS:
            limb_values.append(value)
        elif lead in _PRECORDIAL_LEADS:
            precordial_values.append(value)
    return limb_values, precordial_values, all_values


def _has_limb_precordial_qrs_split(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> bool:
    limb_values, precordial_values, _ = _raw_qrs_by_region(representative_leads)
    if (
        len(limb_values) < _QRS_SPLIT_MIN_LIMB_LEADS
        or len(precordial_values) < _QRS_SPLIT_MIN_PRECORDIAL_LEADS
    ):
        return False
    limb_center = float(np.median(limb_values))
    precordial_center = float(np.median(precordial_values))
    return (
        limb_center <= _WIDE_QRS_MS
        and precordial_center >= _QRS_CONSENSUS_STRONG_WIDE_MS
        and precordial_center - limb_center >= _QRS_SPLIT_MIN_GAP_MS
    )


def _split_qrs_raw_core(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Optional[float]:
    if not _has_limb_precordial_qrs_split(representative_leads):
        return None
    _, precordial_values, all_values = _raw_qrs_by_region(representative_leads)
    if not all_values or not precordial_values:
        return None
    raw_core = float(np.percentile(all_values, _QRS_SPLIT_RAW_PERCENTILE))
    precordial_center = float(np.median(precordial_values))
    return float(max(_WIDE_QRS_MS, min(raw_core, precordial_center)))


def _late_consensus_raw_qrs_core(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    consensus_qrs_ms: Optional[float],
    *,
    paced: bool = False,
) -> Optional[float]:
    if consensus_qrs_ms is None:
        return None

    raw_vals: List[float] = []
    has_st_t_confusion = False
    has_late_offset_low_support = False
    for rep in representative_leads.values():
        has_st_t_confusion = has_st_t_confusion or bool(rep.params.get("st_t_confusion"))
        has_late_offset_low_support = (
            has_late_offset_low_support
            or rep.params.get("qrs_offset_exclusion_reason") == "late_offset_low_support"
        )
        if not rep.params.get("reliable_for_qrs"):
            continue
        raw = _valid_measurement("qrs_ms", rep.params.get("qrs_ms"))
        if raw is not None:
            raw_vals.append(float(raw))
    if not (has_st_t_confusion or has_late_offset_low_support):
        return None
    if len(raw_vals) < _QRS_CONSENSUS_MIN_SUPPORT_LEADS:
        return None

    raw_median = float(np.median(raw_vals))
    wide_raw_count = sum(1 for value in raw_vals if value >= _WIDE_QRS_MS)
    if (
        not paced
        and float(consensus_qrs_ms) >= _QRS_CONSENSUS_STRONG_WIDE_MS
        and float(consensus_qrs_ms) <= _QRS_CONSENSUS_MODERATE_WIDE_MAX_MS
        and wide_raw_count >= _QRS_CONSENSUS_MIN_RAW_WIDE_SUPPORT
    ):
        return None
    if float(consensus_qrs_ms) - raw_median < _QRS_CONSENSUS_RAW_CORE_MIN_DELTA_MS:
        return None

    core_support = [
        value
        for value in raw_vals
        if value <= raw_median + _QRS_CONSENSUS_RAW_CORE_CLUSTER_MS
    ]
    if len(core_support) < _QRS_CONSENSUS_RAW_CORE_MIN_SUPPORT:
        return None
    if len(core_support) < int(np.ceil(len(raw_vals) / 2.0)):
        return None

    raw_core = float(np.percentile(raw_vals, _QRS_CONSENSUS_RAW_CORE_PERCENTILE))
    return min(raw_core, float(consensus_qrs_ms))


def _split_raw_qt_rescue(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    qrs_ms: Optional[float],
    consensus_qt_ms: Optional[float],
) -> Optional[tuple[float, List[str]]]:
    if (
        consensus_qt_ms is None
        or qrs_ms is None
        or qrs_ms < _WIDE_QRS_MS
        or not _has_limb_precordial_qrs_split(representative_leads)
    ):
        return None

    values: List[float] = []
    used_leads: List[str] = []
    for lead in STANDARD_12_LEADS:
        if lead not in _PRECORDIAL_LEADS:
            continue
        rep = representative_leads.get(lead)
        if rep is None or not rep.params.get("reliable_for_global", False):
            continue
        if not _has_visible_t_for_qt(rep):
            continue
        conf = float(rep.params.get("qt_confidence_mean") or 0.0)
        if conf < 0.10:
            continue
        raw_qt = _valid_measurement("qt_ms", rep.params.get("qt_ms"))
        if raw_qt is None or not _qt_value_allowed_for_qrs_width(raw_qt, qrs_ms):
            continue
        values.append(float(raw_qt))
        used_leads.append(lead)

    if len(values) < _SPLIT_QT_RESCUE_MIN_LEADS:
        return None
    rescued = float(np.percentile(values, _SPLIT_QT_RESCUE_PERCENTILE))
    min_delta = (
        _SPLIT_QT_RESCUE_MIN_DELTA_MS
        if float(consensus_qt_ms) < 430.0
        else _SPLIT_QT_RESCUE_SOFT_MIN_DELTA_MS
    )
    if rescued - float(consensus_qt_ms) < min_delta:
        return None
    return rescued, used_leads


def _has_p_support(rep: RepresentativeLeadFeatures) -> bool:
    p_amp = rep.params.get("p_amp_mv")
    # Legacy callers/tests may omit P amplitude. In that case keep the previous
    # behavior; when amplitude is available, require a visible P wave.
    if p_amp is None:
        return True
    confidence = rep.params.get("p_confidence_mean")
    if confidence is not None and np.isfinite(float(confidence)) and float(confidence) < _P_MIN_CONFIDENCE:
        return False
    if not rep.params.get("reliable_for_p", False):
        return False
    if np.isfinite(float(p_amp)) and abs(float(p_amp)) < _P_MIN_ABS_AMP_MV:
        return False
    return True


def _params_have_visible_p(params: Dict[str, object]) -> bool:
    p_amp = params.get("p_amp_mv")
    if p_amp is None:
        return True
    if not np.isfinite(float(p_amp)):
        return False
    confidence = params.get("p_confidence_mean")
    if confidence is not None and np.isfinite(float(confidence)) and float(confidence) < _P_MIN_CONFIDENCE:
        return False
    return abs(float(p_amp)) >= _P_MIN_ABS_AMP_MV


def _supported_pr_consensus(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    raw_vals: List[float],
) -> Optional[float]:
    if len(raw_vals) < 3:
        return None
    raw_spread = float(max(raw_vals) - min(raw_vals))
    if raw_spread < _PR_CONSENSUS_RAW_SPLIT_MS:
        return None

    def _param_int(params: Dict[str, object], key: str) -> int:
        value = params.get(key)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    def _param_float(
        params: Dict[str, object],
        key: str,
        default: float,
    ) -> float:
        value = params.get(key)
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    candidates: List[float] = []
    strong_cluster_candidates: List[float] = []
    for rep in representative_leads.values():
        if not rep.params.get("reliable_for_global"):
            continue
        if not _has_p_support(rep):
            continue
        consensus = _valid_measurement("pr_consensus_ms", rep.params.get("pr_consensus_ms"))
        if consensus is None:
            continue
        if not (_PR_CONSENSUS_MIN_MS <= consensus <= _PR_CONSENSUS_MAX_MS):
            continue
        support_int = _param_int(rep.params, "p_onset_consensus_support")
        if support_int < _PR_CONSENSUS_MIN_SUPPORT:
            continue
        spread_float = _param_float(
            rep.params,
            "p_onset_consensus_spread_ms",
            _PR_CONSENSUS_MAX_ONSET_SPREAD_MS + 1.0,
        )
        if spread_float > _PR_CONSENSUS_MAX_ONSET_SPREAD_MS:
            continue
        candidates.append(float(consensus))

        cluster_support = _param_int(rep.params, "p_onset_cluster_support")
        cluster_spread = _param_float(
            rep.params,
            "p_onset_cluster_spread_ms",
            _PR_CONSENSUS_CLUSTER_MAX_SPREAD_MS + 1.0,
        )
        cluster_limb_support = _param_int(rep.params, "p_onset_cluster_limb_support")
        cluster_pr = _valid_measurement(
            "pr_consensus_ms",
            rep.params.get("p_onset_cluster_pr_ms"),
        )
        if cluster_pr is None:
            cluster_pr = float(consensus)
        if not (_PR_CONSENSUS_MIN_MS <= cluster_pr <= _PR_CONSENSUS_MAX_MS):
            continue
        if cluster_support < _PR_CONSENSUS_MIN_SUPPORT:
            continue
        if cluster_spread > _PR_CONSENSUS_CLUSTER_MAX_SPREAD_MS:
            continue
        if cluster_limb_support < _PR_CONSENSUS_CLUSTER_MIN_LIMB_SUPPORT:
            continue
        strong_cluster_candidates.append(float(cluster_pr))

    if len(candidates) < _PR_CONSENSUS_MIN_SUPPORT:
        return None
    consensus = float(np.median(candidates))
    raw_median = float(np.median(raw_vals))
    if consensus - raw_median >= _PR_CONSENSUS_MIN_RAW_MEDIAN_DELTA_MS:
        return consensus

    if len(strong_cluster_candidates) >= _PR_CONSENSUS_MIN_SUPPORT:
        cluster_consensus = float(np.median(strong_cluster_candidates))
        cluster_raw_support = sum(
            1
            for value in raw_vals
            if abs(float(value) - cluster_consensus) <= _PR_CONSENSUS_CLUSTER_RAW_SUPPORT_MS
        )
        has_short_raw_outlier = (
            raw_median - float(min(raw_vals)) >= _PR_CONSENSUS_SHORT_OUTLIER_GAP_MS
        )
        if (
            has_short_raw_outlier
            and cluster_raw_support >= _PR_CONSENSUS_CLUSTER_MIN_RAW_SUPPORT
            and cluster_consensus - raw_median >= _PR_CONSENSUS_CLUSTER_MIN_RAW_MEDIAN_DELTA_MS
        ):
            return cluster_consensus
        if (
            cluster_raw_support >= _PR_CONSENSUS_CLUSTER_MIN_RAW_SUPPORT
            and raw_median - cluster_consensus >= _PR_CONSENSUS_LONG_RAW_RESCUE_MIN_DELTA_MS
            and raw_spread >= _PR_CONSENSUS_RAW_SPLIT_MS
        ):
            return cluster_consensus
    return None


def _strong_p_onset_cluster_pr(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Optional[float]:
    candidates: List[float] = []
    for rep in representative_leads.values():
        if not rep.params.get("reliable_for_global"):
            continue
        try:
            cluster_support = int(rep.params.get("p_onset_cluster_support") or 0)
            cluster_limb_support = int(rep.params.get("p_onset_cluster_limb_support") or 0)
            cluster_spread = float(
                rep.params.get("p_onset_cluster_spread_ms")
                if rep.params.get("p_onset_cluster_spread_ms") is not None
                else _PR_CONSENSUS_CLUSTER_MAX_SPREAD_MS + 1.0
            )
        except (TypeError, ValueError):
            continue
        if cluster_support < _PR_CONSENSUS_MIN_SUPPORT:
            continue
        if cluster_limb_support < _PR_CONSENSUS_CLUSTER_MIN_LIMB_SUPPORT:
            continue
        if cluster_spread > _PR_CONSENSUS_CLUSTER_MAX_SPREAD_MS:
            continue
        cluster_pr = _valid_measurement(
            "pr_consensus_ms",
            rep.params.get("p_onset_cluster_pr_ms"),
        )
        if cluster_pr is None:
            continue
        if not (_PR_CONSENSUS_MIN_MS <= cluster_pr <= _PR_CONSENSUS_MAX_MS):
            continue
        candidates.append(float(cluster_pr))
    if len(candidates) < _PR_CONSENSUS_MIN_SUPPORT:
        return None
    return float(np.median(candidates))


def _global_pr_median(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Optional[float]:
    # P-wave morphology is more reliable in limb leads; precordial leads frequently
    # produce false P detections close to the QRS, biasing the median downward.
    # Primary: limb leads that are globally reliable.
    vals: List[float] = []
    for lead, rep in representative_leads.items():
        if lead not in _LIMB_LEADS:
            continue
        if not rep.params.get("reliable_for_global"):
            continue
        if not _has_p_support(rep):
            continue
        v = _valid_measurement("pr_ms", rep.params.get("pr_ms"))
        if v is not None:
            vals.append(v)
    if vals:
        if len(vals) < _PR_CONSENSUS_MIN_SUPPORT:
            cluster = _strong_p_onset_cluster_pr(representative_leads)
            if cluster is not None:
                return cluster
        consensus = _supported_pr_consensus(representative_leads, vals)
        if consensus is not None:
            return consensus
        return float(np.median(vals))
    # Fallback: all reliable leads (if no limb lead is reliable).
    for rep in representative_leads.values():
        if not rep.params.get("reliable_for_global"):
            continue
        if not _has_p_support(rep):
            continue
        v = _valid_measurement("pr_ms", rep.params.get("pr_ms"))
        if v is not None:
            vals.append(v)
    if vals:
        if len(vals) < _PR_CONSENSUS_MIN_SUPPORT:
            cluster = _strong_p_onset_cluster_pr(representative_leads)
            if cluster is not None:
                return cluster
        consensus = _supported_pr_consensus(representative_leads, vals)
        if consensus is not None:
            return consensus
        return float(np.median(vals))
    return None


def _physiologic_pr_core_from_beats(
    beat_features: List[LeadBeatFeatures],
) -> Optional[float]:
    by_beat: Dict[object, List[float]] = defaultdict(list)
    for idx, feature in enumerate(beat_features or []):
        pr_ms = _valid_measurement("pr_ms", getattr(feature, "pr_ms", None))
        if pr_ms is None:
            continue
        if not (_PHYSIOLOGIC_PR_CORE_MIN_MS <= pr_ms <= _PHYSIOLOGIC_PR_CORE_MAX_MS):
            continue
        confidence = getattr(feature, "p_confidence", 0.0)
        try:
            confidence_float = float(confidence)
        except (TypeError, ValueError):
            confidence_float = 0.0
        if confidence_float < _PHYSIOLOGIC_PR_CORE_MIN_CONFIDENCE:
            continue
        flags = set(getattr(feature, "flags", []) or [])
        if "p_unreliable" in flags:
            continue
        beat_key = getattr(feature, "beat_id", idx)
        by_beat[beat_key].append(float(pr_ms))

    values = [float(np.median(items)) for items in by_beat.values() if items]
    if len(values) < 2:
        return None
    if len(values) >= 5:
        ordered = sorted(values)
        gaps = [ordered[idx + 1] - ordered[idx] for idx in range(len(ordered) - 1)]
        if gaps:
            max_gap = max(gaps)
            split_idx = gaps.index(max_gap) + 1
            typical_gap = float(np.median(gaps))
            lower_cluster = ordered[:split_idx]
            upper_cluster = ordered[split_idx:]
            if (
                max_gap >= _PHYSIOLOGIC_PR_CORE_SPLIT_GAP_MS
                and max_gap >= typical_gap * _PHYSIOLOGIC_PR_CORE_SPLIT_GAP_RATIO
                and len(lower_cluster) >= _PHYSIOLOGIC_PR_CORE_SPLIT_MIN_LOWER_BEATS
                and len(upper_cluster) >= _PHYSIOLOGIC_PR_CORE_SPLIT_MIN_UPPER_BEATS
            ):
                return float(np.percentile(
                    lower_cluster,
                    _PHYSIOLOGIC_PR_CORE_LOWER_CLUSTER_PERCENTILE,
                ))
        return float(np.percentile(values, _PHYSIOLOGIC_PR_CORE_UPPER_PERCENTILE))
    return float(np.median(values))


def _global_qrs_median(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    *,
    paced: bool = False,
) -> Optional[float]:
    vals: List[float] = []
    raw_vals: List[float] = []
    for rep in representative_leads.values():
        if not rep.params.get("reliable_for_qrs"):
            continue
        raw = _valid_measurement("qrs_ms", rep.params.get("qrs_ms"))
        if raw is not None:
            raw_vals.append(raw)
        consensus = _valid_measurement("qrs_consensus_ms", rep.params.get("qrs_consensus_ms"))
        v = consensus if consensus is not None else raw
        if v is not None:
            vals.append(v)

    if vals:
        split_qrs = _split_qrs_raw_core(representative_leads)
        if split_qrs is not None:
            return split_qrs
        qrs = float(np.median(vals))
        raw_core_qrs = _late_consensus_raw_qrs_core(representative_leads, qrs, paced=paced)
        if raw_core_qrs is not None:
            return raw_core_qrs
        if len(raw_vals) >= 3:
            raw_max = float(np.max(raw_vals))
            if qrs > raw_max + _QRS_CONSENSUS_MAX_BEYOND_RAW_MS:
                return raw_max
            raw_median = float(np.median(raw_vals))
            wide_raw_count = sum(1 for value in raw_vals if value >= _WIDE_QRS_MS)
            strong_wide_consensus = (
                qrs >= _QRS_CONSENSUS_STRONG_WIDE_MS
                and len(vals) >= _QRS_CONSENSUS_MIN_SUPPORT_LEADS
                and wide_raw_count >= _QRS_CONSENSUS_MIN_RAW_WIDE_SUPPORT
            )
            if (
                qrs >= _WIDE_QRS_MS
                and raw_median < _WIDE_QRS_MS
                and wide_raw_count < 3
                and not strong_wide_consensus
            ):
                return raw_median
        return qrs
    return None


def _global_qrs_wide(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    *,
    paced: bool = False,
) -> Optional[float]:
    """P90-based QRS for wide-QRS classification (T-axis branches). Falls back to _global_qrs_median."""
    vals: List[float] = []
    raw_vals: List[float] = []
    for rep in representative_leads.values():
        if not rep.params.get("reliable_for_qrs"):
            continue
        raw = _valid_measurement("qrs_ms", rep.params.get("qrs_ms"))
        if raw is not None:
            raw_vals.append(raw)
        wide = _valid_measurement("qrs_consensus_ms", rep.params.get("qrs_wide_ms"))
        if wide is not None:
            vals.append(wide)
    if vals:
        split_qrs = _split_qrs_raw_core(representative_leads)
        if split_qrs is not None:
            return split_qrs
        qrs = float(np.median(vals))
        if len(raw_vals) >= 3:
            raw_max = float(np.max(raw_vals))
            if qrs > raw_max + _QRS_CONSENSUS_MAX_BEYOND_RAW_MS:
                return raw_max
            raw_median = float(np.median(raw_vals))
            wide_raw_count = sum(1 for value in raw_vals if value >= _WIDE_QRS_MS)
            strong_wide_consensus = (
                qrs >= _QRS_CONSENSUS_STRONG_WIDE_MS
                and len(vals) >= _QRS_CONSENSUS_MIN_SUPPORT_LEADS
                and wide_raw_count >= _QRS_CONSENSUS_MIN_RAW_WIDE_SUPPORT
            )
            if (
                qrs >= _WIDE_QRS_MS
                and raw_median < _WIDE_QRS_MS
                and wide_raw_count < 3
                and not strong_wide_consensus
            ):
                return raw_median
        return qrs
    return None


def _global_p_duration(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Tuple[
    Optional[float],
    Optional[str],
    List[str],
    int,
    Optional[float],
    str,
]:
    """Build one auditable record-level P duration.

    Prefer the aligned multi-lead boundary duration already produced by the
    delineator. If that is unavailable, retain a clearly labelled median of
    reliable lead-local durations rather than silently mixing both paths.
    """

    consensus_values: List[float] = []
    consensus_leads: List[str] = []
    boundary_support: List[int] = []
    raw_values: List[float] = []
    raw_leads: List[str] = []

    for lead in STANDARD_12_LEADS:
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        if not rep.params.get("reliable_for_p", False) or not _has_p_support(rep):
            continue

        consensus = _valid_measurement(
            "p_dur_consensus_ms",
            rep.params.get("p_dur_consensus_ms"),
        )
        if consensus is not None:
            consensus_values.append(float(consensus))
            consensus_leads.append(lead)
            try:
                onset_support = int(rep.params.get("p_onset_consensus_support") or 0)
                offset_support = int(rep.params.get("p_offset_consensus_support") or 0)
            except (TypeError, ValueError):
                onset_support = 0
                offset_support = 0
            if onset_support > 0 and offset_support > 0:
                boundary_support.append(min(onset_support, offset_support))

        raw = _valid_measurement("p_dur_ms", rep.params.get("p_dur_ms"))
        if raw is not None:
            raw_values.append(float(raw))
            raw_leads.append(lead)

    if consensus_values:
        support = max(boundary_support) if boundary_support else len(consensus_values)
        spread = float(np.max(consensus_values) - np.min(consensus_values))
        reliability = "reliable" if support >= 3 else "low_support"
        guarded = any(
            rep.params.get("p_duration_guard_source")
            for lead in consensus_leads
            if (rep := representative_leads.get(lead)) is not None
        )
        return (
            float(np.median(consensus_values)),
            (
                "representative_multilead_consensus_paired_duration_guard"
                if guarded
                else "representative_multilead_consensus"
            ),
            consensus_leads,
            int(support),
            spread,
            reliability,
        )

    if raw_values:
        support = len(raw_values)
        reliability = "fallback" if support >= 2 else "low_support"
        return (
            float(np.median(raw_values)),
            "reliable_lead_median_fallback",
            raw_leads,
            support,
            float(np.max(raw_values) - np.min(raw_values)),
            reliability,
        )

    return None, None, [], 0, None, "unavailable"


def _p_wave_measurements_invalid(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    beat_features: List[LeadBeatFeatures],
    r_locs: np.ndarray,
    fs: int,
) -> bool:
    """The P waves themselves cannot be trusted.

    Distinct from `_pr_association_invalid`: this covers the cases where there
    is no usable P wave to measure at all, so every P-derived quantity --
    including the frontal P axis -- has to be withheld.
    """
    by_beat: Dict[int, List[LeadBeatFeatures]] = defaultdict(list)
    for bf in beat_features:
        by_beat[int(bf.beat_id)].append(bf)

    if by_beat:
        paced_count = sum(
            1
            for items in by_beat.values()
            if any("paced_beat" in bf.flags for bf in items)
        )
        if paced_count == len(by_beat):
            return True

    if len(r_locs) >= 6:
        rr_ms = np.diff(r_locs) * 1000.0 / fs
        rr_cv = float(np.std(rr_ms) / (np.median(rr_ms) + 1e-6)) if len(rr_ms) else 0.0
        visible_p_leads = sum(1 for rep in representative_leads.values() if _has_p_support(rep))
        if rr_cv >= _ATRIAL_IRREGULAR_RR_CV and visible_p_leads < 2:
            return True
    return False


def _pr_association_invalid(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    r_locs: np.ndarray,
    fs: int,
) -> bool:
    """The P-to-QRS pairing is unreliable, so PR-derived values are not usable.

    Deliberately separate from `_p_wave_measurements_invalid`. Inconsistent PR
    across leads says the algorithm cannot tell which P belongs to which QRS;
    it says nothing about the P waves themselves, so the frontal P axis -- a
    pure P-area quantity that never references the QRS -- stays valid. Folding
    the two together previously discarded a perfectly good P axis whenever PR
    was noisy.
    """
    if len(r_locs) < 6:
        return False
    rr_ms = np.diff(r_locs) * 1000.0 / fs
    rr_cv = float(np.std(rr_ms) / (np.median(rr_ms) + 1e-6)) if len(rr_ms) else 0.0
    if rr_cv < _ATRIAL_IRREGULAR_RR_CV:
        return False
    visible_p_leads = sum(1 for rep in representative_leads.values() if _has_p_support(rep))
    if visible_p_leads < 2:
        return False
    pr_values = [
        float(value)
        for rep in representative_leads.values()
        if _has_p_support(rep)
        for value in [_valid_measurement("pr_ms", rep.params.get("pr_ms"))]
        if value is not None
    ]
    if not pr_values:
        return False
    pr_median = float(np.median(pr_values))
    pr_spread = float(np.max(pr_values) - np.min(pr_values))
    return pr_median >= 240.0 or pr_spread >= 120.0


def _atrial_measurements_invalid(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    beat_features: List[LeadBeatFeatures],
    r_locs: np.ndarray,
    fs: int,
) -> bool:
    return _p_wave_measurements_invalid(
        representative_leads, beat_features, r_locs, fs
    ) or _pr_association_invalid(representative_leads, r_locs, fs)


def _t_fusion_quality_summary(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    beat_features: Sequence[LeadBeatFeatures] = (),
) -> Dict[str, object]:
    def finite_values(key: str) -> List[float]:
        values: List[float] = []
        for rep in representative_leads.values():
            value = rep.params.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(numeric):
                values.append(numeric)
        return values

    def string_union(key: str) -> List[str]:
        values: Set[str] = set()
        for rep in representative_leads.values():
            raw = rep.params.get(key)
            if raw is None:
                continue
            values.update(
                token.strip()
                for token in str(raw).split(",")
                if token.strip() and token.strip() != "reliable"
            )
        return sorted(values)

    support_values = finite_values("t_offset_fusion_support")
    mad_values = finite_values("t_offset_fusion_mad_ms")
    ci_values = finite_values("t_offset_fusion_ci_half_width_ms")
    cluster_count_values = finite_values("t_offset_cluster_count")
    cluster_score_values = finite_values("t_offset_selected_cluster_score")
    cluster_support_values = finite_values("t_offset_selected_cluster_support")
    tpte_values = finite_values("t_global_tpte_ms")
    derived_disagreement_values = finite_values(
        "t_offset_derived_disagreement_ms"
    )
    evidence_present = bool(support_values or mad_values or cluster_count_values)
    reliable_votes = [
        bool(rep.params.get("t_offset_statistical_fusion_reliable"))
        for rep in representative_leads.values()
        if rep.params.get("t_offset_fusion_support") is not None
    ]
    representative_systematic_early = any(
        bool(rep.params.get("t_offset_systematic_early_risk"))
        for rep in representative_leads.values()
    )
    beat_risk_by_id: Dict[int, bool] = {}
    beat_tail_by_id: Dict[int, bool] = {}
    for feature in beat_features:
        beat_id = int(feature.beat_id)
        beat_risk_by_id[beat_id] = bool(
            beat_risk_by_id.get(beat_id, False)
            or feature.t_offset_systematic_early_risk
        )
        beat_tail_by_id[beat_id] = bool(
            beat_tail_by_id.get(beat_id, False)
            or "multilead_residual_t_tail"
            in str(feature.t_offset_fusion_reliability_reason or "")
        )
    support = (
        int(round(float(np.median(support_values)))) if support_values else 0
    )
    cluster_count = (
        int(round(float(np.median(cluster_count_values))))
        if cluster_count_values
        else 0
    )
    derived_disagreement_ms = (
        float(np.median(derived_disagreement_values))
        if derived_disagreement_values
        else None
    )
    risk_count = sum(beat_risk_by_id.values())
    systematic_early_fraction = (
        float(risk_count / len(beat_risk_by_id))
        if beat_risk_by_id
        else (1.0 if representative_systematic_early else 0.0)
    )
    tail_count = sum(beat_tail_by_id.values())
    tail_incomplete_fraction = (
        float(tail_count / len(beat_tail_by_id))
        if beat_tail_by_id
        else 0.0
    )
    recurrent_combined_risk = bool(
        risk_count >= 2
        and systematic_early_fraction >= 0.25
    )
    recurrent_tail_with_low_support = bool(
        tail_count >= 2
        and tail_incomplete_fraction >= 0.70
        and support <= 6
    )
    ambiguous_multicluster_vector_disagreement = bool(
        support <= 5
        and cluster_count >= 3
        and derived_disagreement_ms is not None
        and derived_disagreement_ms >= 50.0
    )
    systematic_early = (
        bool(
            recurrent_combined_risk
            or recurrent_tail_with_low_support
            or ambiguous_multicluster_vector_disagreement
        )
        if beat_risk_by_id
        else representative_systematic_early
    )
    mad_ms = float(np.median(mad_values)) if mad_values else None
    base_reliable = bool(
        evidence_present
        and reliable_votes
        and sum(reliable_votes) >= (len(reliable_votes) + 1) // 2
        and support >= 4
        and mad_ms is not None
        and mad_ms <= 20.0
    )
    morphology_guard_pass = bool(
        base_reliable
        and not systematic_early
    )
    reasons = string_union("t_offset_fusion_reliability_reason")
    if evidence_present and support < 4:
        reasons.append("support_below_4")
    if mad_ms is not None and mad_ms > 20.0:
        reasons.append("dispersion_above_20ms")
    if systematic_early:
        reasons.append("systematic_early_t_end_risk")
    reasons = list(dict.fromkeys(reasons))
    return {
        "evidence_present": evidence_present,
        "support": support,
        "mad_ms": mad_ms,
        "ci_half_width_ms": (
            float(np.median(ci_values)) if ci_values else None
        ),
        "reliable": morphology_guard_pass,
        "cluster_count": cluster_count,
        "selected_cluster_score": (
            float(np.median(cluster_score_values))
            if cluster_score_values
            else None
        ),
        "selected_cluster_support": (
            int(round(float(np.median(cluster_support_values))))
            if cluster_support_values
            else 0
        ),
        "lead_groups": string_union(
            "t_offset_selected_cluster_lead_groups"
        ),
        "methods": string_union("t_offset_selected_cluster_methods"),
        "reasons": reasons,
        "tpte_ms": float(np.median(tpte_values)) if tpte_values else None,
        "derived_disagreement_ms": derived_disagreement_ms,
        "tail_incomplete_leads": string_union(
            "t_offset_tail_incomplete_leads"
        ),
        "systematic_early_risk": systematic_early,
        "systematic_early_fraction": systematic_early_fraction,
        "tail_incomplete_fraction": tail_incomplete_fraction,
        "morphology_guard_pass": morphology_guard_pass,
    }


def _apply_t_fusion_qt_reliability(
    *,
    qt_ms: Optional[float],
    qt_source: Optional[str],
    qt_reliability: str,
    t_fusion: Mapping[str, object],
) -> tuple[str, bool, List[str]]:
    """Apply record-level morphology validity without discarding audit values."""

    if qt_ms is None:
        return "unavailable", False, []
    reportable = qt_reliability in {"reliable", "rescued"}
    if not bool(t_fusion.get("evidence_present")):
        return qt_reliability, reportable, []

    fusion_reasons = [
        str(reason) for reason in (t_fusion.get("reasons") or []) if reason
    ]
    # This path has an independent, multi-lead late-tail proof and is
    # deliberately retained when the earlier generic fusion is scattered.
    independent_late_tail_rescue = (
        qt_source == "tail_confirmed_late_raw_qt_rescue"
        and qt_reliability == "rescued"
    )
    if independent_late_tail_rescue:
        return qt_reliability, True, []

    if bool(t_fusion.get("systematic_early_risk")):
        reasons = fusion_reasons or ["systematic_early_t_end_risk"]
        return "unreliable", False, reasons
    if not bool(t_fusion.get("reliable")):
        reasons = fusion_reasons or ["t_fusion_not_reliable"]
        return "unreliable", False, reasons
    return qt_reliability, reportable, []


def compute_global_features(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    beat_features: List[LeadBeatFeatures],
    r_locs: np.ndarray,
    fs: int,
    paced: bool = False,
) -> GlobalFeatures:
    rr_ms = np.diff(r_locs) * 1000.0 / fs if len(r_locs) > 1 else np.array([])
    hr = float(60000.0 / np.median(rr_ms)) if len(rr_ms) > 0 else None
    atrial_rate = _estimate_atrial_rate_bpm(beat_features, fs) or hr

    # Global PR stays on the per-lead median path for stable interval policy.
    p_wave_invalid = _p_wave_measurements_invalid(
        representative_leads, beat_features, r_locs, fs
    )
    atrial_invalid = p_wave_invalid or _pr_association_invalid(
        representative_leads, r_locs, fs
    )
    pr  = None if atrial_invalid else _global_pr_median(representative_leads)
    if atrial_invalid:
        p_duration = None
        p_duration_source: Optional[str] = "suppressed_atrial_measurements"
        p_duration_used_leads: List[str] = []
        p_duration_support = 0
        p_duration_spread = None
        p_duration_reliability = "unavailable"
    else:
        (
            p_duration,
            p_duration_source,
            p_duration_used_leads,
            p_duration_support,
            p_duration_spread,
            p_duration_reliability,
        ) = _global_p_duration(representative_leads)
    # Global QRS: P75-based consensus for measurement; P90-based for wide-QRS classification
    qrs      = _global_qrs_median(representative_leads, paced=paced)
    qrs_wide = _global_qrs_wide(representative_leads, paced=paced) or qrs   # fallback to measurement if P90 unavailable
    t_fusion_quality = _t_fusion_quality_summary(
        representative_leads,
        beat_features,
    )
    # T021/T023: reliable-lead QT pipeline with physiological range guard
    selected_reliable_qt_leads = _select_reliable_qt_leads(representative_leads)
    reliable_qt_leads = _filter_wide_qrs_qt_leads(
        representative_leads,
        selected_reliable_qt_leads,
        qrs_wide,
    )

    qt_consensus_candidates: List[float] = []
    qt_consensus_used_leads: List[str] = []
    for lead in reliable_qt_leads:
        qt_cons = _rep_qt_value_for_qrs_width(representative_leads[lead], qrs)
        if qt_cons is not None:
            qt_consensus_candidates.append(qt_cons)
            qt_consensus_used_leads.append(lead)

    # Prefer the consensus QT path because it reuses a global reliable-lead T-end.
    qt = float(np.median(qt_consensus_candidates)) if qt_consensus_candidates else None
    qt_source: Optional[str] = "reliable_lead_median" if qt is not None else None
    qt_used_leads: List[str] = list(qt_consensus_used_leads) if qt is not None else []
    qt_reliability = "reliable" if qt is not None else "unavailable"
    qt_raw_rescue = _wide_qrs_raw_qt_rescue(representative_leads, reliable_qt_leads, qrs, qt)
    wide_raw_rescue_applied = qt_raw_rescue is not None
    if qt_raw_rescue is not None:
        qt = qt_raw_rescue
        qt_consensus_candidates = [qt]
        qt_source = "wide_qrs_raw_qt_rescue"
        qt_used_leads = list(reliable_qt_leads)
        qt_reliability = "rescued"
    if not wide_raw_rescue_applied:
        wide_long_raw_core_rescue = _wide_qrs_long_raw_qt_core_rescue(
            representative_leads,
            qrs,
            qt,
        )
        if wide_long_raw_core_rescue is not None:
            qt, qt_used_leads = wide_long_raw_core_rescue
            qt_consensus_candidates = [qt]
            qt_source = "wide_qrs_long_raw_qt_core_rescue"
            qt_reliability = "rescued"
            wide_raw_rescue_applied = True
    rr_cv = _rr_cv_from_locs(r_locs)
    qt_core_rescue = (
        None
        if wide_raw_rescue_applied
        else _raw_qt_core_rescue(representative_leads, qrs, qt, rr_cv)
    )
    if qt_core_rescue is not None:
        qt = qt_core_rescue
        qt_consensus_candidates = [qt]
        qt_source = "raw_qt_core_rescue"
        qt_used_leads = [
            lead
            for lead, rep in representative_leads.items()
            if rep.params.get("reliable_for_qt", False)
        ]
        qt_reliability = "rescued"
    if not wide_raw_rescue_applied:
        qt_raw_cluster_rescue = _reliable_raw_qt_cluster_rescue(
            representative_leads,
            reliable_qt_leads,
            qrs,
            qt,
        )
        if qt_raw_cluster_rescue is not None:
            qt = qt_raw_cluster_rescue
            qt_consensus_candidates = [qt]
            qt_source = "reliable_raw_qt_cluster_rescue"
            qt_used_leads = list(reliable_qt_leads)
            qt_reliability = "rescued"
    late_raw_qt_cluster_rescue = _late_raw_qt_cluster_rescue(
        representative_leads,
        reliable_qt_leads,
        qrs,
        qt,
        rr_cv,
    )
    if late_raw_qt_cluster_rescue is not None:
        qt, rescue_used_leads = late_raw_qt_cluster_rescue
        qt_consensus_candidates = [qt]
        qt_source = "late_raw_qt_cluster_rescue"
        qt_used_leads = rescue_used_leads
        qt_reliability = "rescued"
    tail_confirmed_late_qt_rescue = _tail_confirmed_late_raw_qt_rescue(
        representative_leads,
        qrs,
        qt,
        rr_cv,
    )
    if tail_confirmed_late_qt_rescue is not None:
        qt, rescue_used_leads = tail_confirmed_late_qt_rescue
        qt_consensus_candidates = [qt]
        qt_source = "tail_confirmed_late_raw_qt_rescue"
        qt_used_leads = rescue_used_leads
        qt_reliability = "rescued"
    split_qt_rescue = _split_raw_qt_rescue(
        representative_leads,
        qrs,
        qt,
    )
    if split_qt_rescue is not None:
        qt, rescue_used_leads = split_qt_rescue
        qt_consensus_candidates = [qt]
        qt_source = "split_raw_qt_rescue"
        qt_used_leads = rescue_used_leads
        qt_reliability = "rescued"

    qt_candidates = list(qt_consensus_candidates)

    # T017: amplitude-weighted median remains the fallback for per-lead QT values.
    if qt is None:
        qt = _weighted_median_qt(representative_leads, reliable_qt_leads, qrs)
        qt_candidates = []
        for lead in reliable_qt_leads:
            qt_v = _rep_qt_value_for_qrs_width(representative_leads[lead], qrs)
            if qt_v is not None:
                qt_candidates.append(qt_v)
        if qt is not None:
            qt_source = "amplitude_weighted_reliable_leads"
            qt_used_leads = list(reliable_qt_leads)
            qt_reliability = "fallback"

    # Fallback: if weighted-median QT failed, use global-reliable leads with physiological range guard
    if qt is None and not qt_candidates:
        fallback_used_leads: List[str] = []
        for rep in representative_leads.values():
            qt_v = _rep_qt_value_for_qrs_width(rep, qrs)
            if (rep.params.get("reliable_for_global")
                    and qt_v is not None):
                qt_candidates.append(qt_v)
                fallback_used_leads.append(rep.lead)
        if qt_candidates:
            qt_used_leads = fallback_used_leads
    if qt is None and qt_candidates:
        qt = float(np.median(qt_candidates))
        qt_source = "global_reliable_lead_fallback"
        qt_reliability = "fallback"
    qt_beat_core = _irregular_beat_qt_core(beat_features, rr_cv)
    if qt_beat_core is not None and (
        qt is None
        or (
            not wide_raw_rescue_applied
            and qt_beat_core < qt - 5.0
        )
        or (
            paced
            and not wide_raw_rescue_applied
            and qt_beat_core > qt + 20.0
        )
    ):
        qt = qt_beat_core
        qt_candidates = [qt]
        qt_source = "irregular_beat_qt_core"
        qt_used_leads = []
        qt_reliability = "rescued"
    if qt is not None and len(reliable_qt_leads) <= 1:
        qt_late_core_rescue = _raw_qt_late_core_rescue(
            representative_leads,
            qrs,
            qt,
            rr_cv,
        )
        if qt_late_core_rescue is not None:
            qt = qt_late_core_rescue
            qt_candidates = [qt]
            qt_source = "raw_qt_late_core_rescue"
            qt_used_leads = [
                lead
                for lead, rep in representative_leads.items()
                if rep.params.get("reliable_for_global", False)
            ]
            qt_reliability = "rescued"

    # DXL Fig 1-23: leads II and V5 are most often closest to the 5-cardiologist
    # reference QT — use as final fallback when the reliable-lead pipeline yields nothing.
    _PREFERRED_QT_LEADS = ["II", "V5", "V4", "V6", "III", "aVF"]
    if qt is None:
        for _fl in _PREFERRED_QT_LEADS:
            _rep = representative_leads.get(_fl)
            if _rep is None:
                continue
            _qt_v = _rep_qt_value_for_qrs_width(_rep, qrs)
            if _qt_v is not None:
                qt = _qt_v
                if not qt_candidates:
                    qt_candidates = [qt]
                qt_source = "preferred_lead_fallback"
                qt_used_leads = [_fl]
                qt_reliability = "fallback"
                break

    qrs_offset_confidence = _nanmedian(
        rep.params.get("qrs_off_confidence_mean")
        for rep in representative_leads.values()
    )
    lead_qt_values: Dict[str, Optional[float]] = {}
    lead_qt_consensus_values: Dict[str, Optional[float]] = {}
    lead_qrs_values: Dict[str, Optional[float]] = {}
    for _lead, _rep in representative_leads.items():
        lead_qt_values[_lead] = _valid_measurement("qt_ms", _rep.params.get("qt_ms"))
        lead_qt_consensus_values[_lead] = _valid_measurement(
            "qt_consensus_ms",
            _rep.params.get("qt_consensus_ms"),
        )
        lead_qrs_values[_lead] = _rep_qrs_value(_rep)
    qt_path_decision = build_qt_path_decision(
        qrs_ms=qrs_wide,
        reliable_qt_leads=selected_reliable_qt_leads,
        lead_qt_values=lead_qt_values,
        lead_qrs_values=lead_qrs_values,
        lead_qt_consensus_values=lead_qt_consensus_values,
        paced=paced,
        qrs_offset_confidence=qrs_offset_confidence,
    )
    def _low_support_qt_fallback() -> tuple[Optional[float], List[str]]:
        if qt_path_decision.used_leads:
            candidate_leads = list(qt_path_decision.used_leads)
            min_candidates = 1
        elif (
            not paced
            and qrs_wide is not None
            and qrs_wide < _WIDE_QRS_MS
        ):
            candidate_leads = list(qt_path_decision.consensus_vs_independent_per_lead.keys())
            min_candidates = 3
        else:
            return None, []
        values: List[float] = []
        used: List[str] = []
        for lead in candidate_leads:
            detail = qt_path_decision.consensus_vs_independent_per_lead.get(lead, {})
            value = _valid_measurement("qt_consensus_ms", detail.get("qt_consensus_ms"))
            if value is None:
                value = _valid_measurement("qt_ms", detail.get("qt_ms"))
            if value is not None:
                values.append(value)
                used.append(lead)
        if len(values) < min_candidates:
            return None, []
        return float(np.median(values)), used

    if qt_path_decision.path == "low_qt_support":
        prior_qt_source = qt_source
        qt_source = "low_qt_support"
        qt_reliability = qt_path_decision.reliability
        qt_used_leads = list(qt_path_decision.used_leads)
        raw_beat_core_fallback = (
            _irregular_raw_beat_qt_core(beat_features, rr_cv)
            if qt_path_decision.used_leads
            else None
        )
        if (
            raw_beat_core_fallback is not None
            and (qt is None or qt - raw_beat_core_fallback >= 60.0)
        ):
            qt = raw_beat_core_fallback
            qt_source = "irregular_raw_beat_qt_core"
            qt_reliability = "low_confidence"
            qt_used_leads = []
        elif paced and (
            paced_raw_fallback := _paced_low_support_raw_qt_fallback(representative_leads, qrs_wide)
        ) is not None:
            qt, qt_used_leads = paced_raw_fallback
            qt_source = "paced_low_support_raw_qt_fallback"
            qt_reliability = "low_confidence"
        elif (
            qt is not None
            and qrs_wide is not None
            and qrs_wide >= _WIDE_QRS_MS
            and qt_path_decision.used_leads
            and "qt_outside_path_bounds" not in set(qt_path_decision.excluded_leads.values())
        ):
            qt_source = "wide_qrs_low_support_qt_fallback"
            qt_reliability = "low_confidence"
        elif (
            qt is not None
            and prior_qt_source == "irregular_beat_qt_core"
            and not qt_path_decision.used_leads
        ):
            qt_source = "irregular_beat_qt_core"
            qt_reliability = "low_confidence"
        elif qt is not None and prior_qt_source == "preferred_lead_fallback":
            qt_source = "preferred_lead_fallback"
            qt_reliability = "low_confidence"
        else:
            qt, low_support_used_leads = _low_support_qt_fallback()
            if qt is not None:
                qt_source = "low_qt_support_fallback"
                qt_reliability = "low_confidence"
                qt_used_leads = low_support_used_leads
            else:
                qt = None
        if qt is not None and qt_source in {
            "low_qt_support",
            "low_qt_support_fallback",
            "preferred_lead_fallback",
        }:
            low_support_late_rescue = _low_support_late_raw_qt_cluster_rescue(
                representative_leads,
                qrs_wide,
                qt,
                rr_cv,
            )
            if low_support_late_rescue is not None:
                qt, qt_used_leads = low_support_late_rescue
                qt_candidates = [qt]
                qt_source = "low_support_late_raw_qt_cluster_rescue"
                qt_reliability = "rescued"
        if qt is not None and qt_source in {
            "low_qt_support",
            "low_qt_support_fallback",
            "preferred_lead_fallback",
        }:
            tail_confirmed_low_support_rescue = _tail_confirmed_late_raw_qt_rescue(
                representative_leads,
                qrs_wide,
                qt,
                rr_cv,
            )
            if tail_confirmed_low_support_rescue is not None:
                qt, qt_used_leads = tail_confirmed_low_support_rescue
                qt_candidates = [qt]
                qt_source = "tail_confirmed_late_raw_qt_rescue"
                qt_reliability = "rescued"

    _backfill_representative_qt_from_core(
        representative_leads,
        beat_features=beat_features,
        qt_ms=qt,
        qrs_ms=qrs,
        rr_cv=rr_cv,
        source=qt_source,
    )

    (
        qt_reliability,
        qt_reportable,
        qt_unreliable_reasons,
    ) = _apply_t_fusion_qt_reliability(
        qt_ms=qt,
        qt_source=qt_source,
        qt_reliability=qt_reliability,
        t_fusion=t_fusion_quality,
    )
    qt_confidence_parts = [
        str(qt_path_decision.reason)
        if qt_path_decision.reason is not None
        else ""
    ]
    qt_confidence_parts.extend(qt_unreliable_reasons)
    qt_confidence_reason = ",".join(
        dict.fromkeys(part for part in qt_confidence_parts if part)
    ) or None

    qt_dispersion = _qt_dispersion_summary(representative_leads, qrs)
    qt_disp = qt_dispersion.legacy_ms
    rr_sec = float(np.median(rr_ms) / 1000.0) if len(rr_ms) > 0 else None
    qtc_b = float(qt / np.sqrt(rr_sec)) if qt and rr_sec and rr_sec > 0 else None   # Bazett:     QTcB = QT / √RR
    qtc_f = float(qt / np.cbrt(rr_sec)) if qt and rr_sec and rr_sec > 0 else None   # Fridericia: QTcF = QT / RR^(1/3)  (cbrt = exact ⅓, more precise than 0.33)
    qtc_framingham = (
        float(qt + 154.0 * (1.0 - rr_sec))
        if qt is not None and rr_sec is not None and rr_sec > 0.0
        else None
    )
    qtc_hodges = (
        float(qt + 1.75 * (hr - 60.0))
        if qt is not None and hr is not None
        else None
    )
    instantaneous_hr = 60000.0 / rr_ms if len(rr_ms) else np.asarray([], dtype=float)
    heart_rate_min = float(np.min(instantaneous_hr)) if instantaneous_hr.size else None
    heart_rate_max = float(np.max(instantaneous_hr)) if instantaneous_hr.size else None
    rr_mean = float(np.mean(rr_ms)) if len(rr_ms) else None
    rr_sd = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else None
    rr_cv_metric = (
        float(np.std(rr_ms) / np.mean(rr_ms))
        if len(rr_ms) > 1 and float(np.mean(rr_ms)) > 0.0
        else None
    )
    rr_diffs = np.diff(rr_ms) if len(rr_ms) > 1 else np.asarray([], dtype=float)
    rmssd = float(np.sqrt(np.mean(rr_diffs**2))) if rr_diffs.size else None
    pnn50 = float(np.mean(np.abs(rr_diffs) > 50.0)) if rr_diffs.size else None
    poincare_sd1 = (
        float(np.std(rr_diffs, ddof=1) / np.sqrt(2.0))
        if rr_diffs.size > 1
        else None
    )
    poincare_sd2 = (
        float(np.sqrt(max(0.0, 2.0 * float(np.var(rr_ms, ddof=1)) - float(poincare_sd1 or 0.0) ** 2)))
        if len(rr_ms) > 2 and poincare_sd1 is not None
        else None
    )
    rr_entropy = None
    if len(rr_ms) >= 3:
        bin_width_ms = 20.0
        edges = np.arange(
            np.floor(float(np.min(rr_ms)) / bin_width_ms) * bin_width_ms,
            np.ceil(float(np.max(rr_ms)) / bin_width_ms) * bin_width_ms + 2 * bin_width_ms,
            bin_width_ms,
        )
        counts, _ = np.histogram(rr_ms, bins=edges)
        probabilities = counts[counts > 0].astype(float) / float(np.sum(counts))
        rr_entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    # T020: pass per-lead confidence weights to axis computation
    p_confs   = {k: v.params.get("p_confidence_mean")   for k, v in representative_leads.items()}
    qrs_confs = {k: v.params.get("qrs_confidence_mean") for k, v in representative_leads.items()}
    t_confs   = {k: v.params.get("qt_confidence_mean")  for k, v in representative_leads.items()}

    # DXL: use signed waveform areas for axis (not unsigned absolute areas or peak amplitudes)
    p_axis_values = _p_axis_values_with_polarity_consensus(representative_leads)
    p_axis_support_count = _axis_value_count(p_axis_values)
    p_axis_has_enough_support = bool(
        p_axis_support_count >= _P_AXIS_MIN_FRONTAL_LEADS
        and (
            p_axis_support_count >= _P_AXIS_MIN_FRONTAL_LEADS_WITHOUT_PR_SUPPORT
            or _p_axis_values_have_local_pr_support(representative_leads, p_axis_values)
        )
    )
    # Gated on P-wave validity only, not on PR validity: the frontal P axis is
    # computed from P-wave signed areas and never references the QRS, so an
    # unreliable P-to-QRS pairing must not discard it.
    p_axis   = None if p_wave_invalid or not p_axis_has_enough_support else _robust_p_axis_from_values(
        representative_leads,
        p_axis_values,
        p_confs,
    )
    qrs_axis_values = _qrs_net_amplitude(representative_leads)
    qrs_axis = _axis_from_amplitudes(qrs_axis_values, qrs_confs)
    qrs_peak_values = _qrs_peak_net_amplitude(representative_leads)
    qrs_peak_axis = _axis_from_amplitudes(qrs_peak_values, qrs_confs)
    if (
        _should_use_peak_net_qrs_axis(qrs_axis_values, qrs_peak_values, qrs)
        and qrs_peak_axis is not None
        and (
            qrs_axis is None
            or (_axis_delta_deg(qrs_axis, qrs_peak_axis) or 0.0) >= _QRS_AXIS_MIN_SHIFT_DEG
        )
    ):
        qrs_axis = qrs_peak_axis
    raw_t_axis_values = _signed_wave_area(
        representative_leads,
        "t_area",
        "t_amp_mv",
        reliable_key="reliable_for_t",
        confidence_key="qt_confidence_mean",
        min_confidence=_T_MIN_CONFIDENCE,
        min_abs_amp_mv=_T_MIN_AXIS_ABS_AMP_MV,
        signed_area_key="t_signed_area",
    )
    legacy_t_axis_values = _filter_t_axis_values_for_reporting(representative_leads, raw_t_axis_values)
    legacy_t_axis_values = _filter_limb_seed_t_axis_conflicts(
        representative_leads,
        legacy_t_axis_values,
        qrs,
        rr_cv,
    )
    legacy_t_axis = _axis_from_amplitudes(legacy_t_axis_values, t_confs)
    st_corrected_t_values = _st_corrected_t_amplitudes(representative_leads, legacy_t_axis_values)
    st_corrected_t_axis = _axis_from_amplitudes(st_corrected_t_values, t_confs)
    legacy_used_st_corrected_t_axis = False
    if (
        _should_use_st_corrected_t_axis(representative_leads, legacy_t_axis_values, st_corrected_t_values, qrs_wide)
        and st_corrected_t_axis is not None
        and (
            legacy_t_axis is None
            or (_axis_delta_deg(legacy_t_axis, st_corrected_t_axis) or 0.0) >= _T_ST_AXIS_MIN_SHIFT_DEG
        )
    ):
        legacy_t_axis = st_corrected_t_axis
        legacy_t_axis_values = st_corrected_t_values
        legacy_used_st_corrected_t_axis = True

    cluster_allowed_t_axis_leads = {
        lead for lead in HEXAXIAL_ANGLES
        if legacy_t_axis_values.get(lead) is not None
    }
    clustered_t_axis = compute_t_axis_from_cluster(
        representative_leads,
        allowed_leads=cluster_allowed_t_axis_leads,
    )
    used_clustered_t_axis = False
    used_low_support_t_axis_coverage = False
    if legacy_used_st_corrected_t_axis:
        t_axis_values = _mask_hard_excluded_t_axis_values(
            legacy_t_axis_values,
            clustered_t_axis.excluded_leads,
        )
        t_axis = _axis_from_amplitudes(t_axis_values, t_confs)
        used_st_corrected_t_axis = True
    elif clustered_t_axis.axis_deg is not None:
        t_axis_values = {
            lead: clustered_t_axis.used_leads.get(lead)
            for lead in HEXAXIAL_ANGLES
        }
        t_axis = clustered_t_axis.axis_deg
        used_st_corrected_t_axis = False
        used_clustered_t_axis = True
    else:
        t_axis_values = _mask_hard_excluded_t_axis_values(
            legacy_t_axis_values,
            clustered_t_axis.excluded_leads,
        )
        t_axis = _axis_from_amplitudes(t_axis_values, t_confs)
        used_st_corrected_t_axis = legacy_used_st_corrected_t_axis

    if _should_suppress_t_axis(
        representative_leads,
        t_axis_values,
        t_axis,
        qrs_axis,
        qrs,
        r_locs,
        fs,
        used_st_corrected_axis=used_st_corrected_t_axis,
        paced=paced,
    ):
        t_axis = None
        if used_clustered_t_axis:
            t_axis_values = _mask_hard_excluded_t_axis_values(
                legacy_t_axis_values,
                clustered_t_axis.excluded_leads,
            )
            t_axis = _axis_from_amplitudes(t_axis_values, t_confs)
            used_st_corrected_t_axis = legacy_used_st_corrected_t_axis
            if _should_suppress_t_axis(
                representative_leads,
                t_axis_values,
                t_axis,
                qrs_axis,
                qrs,
                r_locs,
                fs,
                used_st_corrected_axis=used_st_corrected_t_axis,
                paced=paced,
            ):
                t_axis = None
    if t_axis is None:
        relaxed_t_axis_values = _filter_t_axis_values_for_reporting(
            representative_leads,
            _signed_wave_area(
                representative_leads,
                "t_area",
                "t_amp_mv",
                reliable_key="reliable_for_t",
                confidence_key="qt_confidence_mean",
                min_confidence=_T_AXIS_RELAXED_MIN_CONFIDENCE,
                min_abs_amp_mv=_T_AXIS_RELAXED_MIN_ABS_AMP_MV,
                signed_area_key="t_signed_area",
            ),
        )
        relaxed_t_axis_values = _filter_limb_seed_t_axis_conflicts(
            representative_leads,
            relaxed_t_axis_values,
            qrs,
            rr_cv,
        )
        relaxed_clustered_t_axis = compute_t_axis_from_cluster(
            representative_leads,
            min_leads=_T_AXIS_RELAXED_MIN_LEADS,
            min_abs_amp_mv=_T_AXIS_RELAXED_MIN_ABS_AMP_MV,
            min_confidence=_T_AXIS_RELAXED_MIN_CONFIDENCE,
            allowed_leads={
                lead for lead, value in relaxed_t_axis_values.items()
                if value is not None
            },
        )
        relaxed_t_axis_values = _mask_hard_excluded_t_axis_values(
            relaxed_t_axis_values,
            relaxed_clustered_t_axis.excluded_leads,
        )
        relaxed_t_axis = _axis_from_amplitudes(relaxed_t_axis_values)
        if (
            relaxed_t_axis is not None
            and _axis_value_count(relaxed_t_axis_values) >= _T_AXIS_RELAXED_MIN_LEADS
            and not _should_suppress_t_axis(
                representative_leads,
                relaxed_t_axis_values,
                relaxed_t_axis,
                qrs_axis,
                qrs,
                r_locs,
                fs,
                used_st_corrected_axis=False,
                paced=paced,
            )
        ):
            t_axis = relaxed_t_axis
            t_axis_values = relaxed_t_axis_values
    if t_axis is None:
        stable_limb_t_axis_values = _stable_limb_t_axis_fallback_values(
            representative_leads,
            qrs,
            r_locs,
            fs,
            atrial_invalid,
        )
        if stable_limb_t_axis_values is not None:
            stable_limb_t_axis = _axis_from_amplitudes(stable_limb_t_axis_values)
            if (
                stable_limb_t_axis is not None
                and not _should_suppress_t_axis(
                    representative_leads,
                    stable_limb_t_axis_values,
                    stable_limb_t_axis,
                    qrs_axis,
                    qrs,
                    r_locs,
                    fs,
                    used_st_corrected_axis=False,
                    paced=paced,
                )
            ):
                t_axis = stable_limb_t_axis
                t_axis_values = stable_limb_t_axis_values
    if t_axis is None:
        low_support_t_axis_values = _low_support_limb_t_axis_coverage_values(
            representative_leads,
            qrs,
            r_locs,
            fs,
            atrial_invalid,
        )
        if low_support_t_axis_values is not None:
            low_support_t_axis = _axis_from_amplitudes(low_support_t_axis_values)
            if low_support_t_axis is not None:
                t_axis = low_support_t_axis
                t_axis_values = low_support_t_axis_values
                used_low_support_t_axis_coverage = True
    if t_axis is None:
        detected_t_axis_values = _detected_limb_t_axis_coverage_values(
            representative_leads,
            qrs,
            r_locs,
            fs,
            atrial_invalid,
        )
        if detected_t_axis_values is not None:
            detected_t_axis = _axis_from_amplitudes(detected_t_axis_values, t_confs)
            if detected_t_axis is not None:
                t_axis = detected_t_axis
                t_axis_values = detected_t_axis_values
                used_low_support_t_axis_coverage = True
    if t_axis is None:
        st_supported_t_axis_values = _st_supported_limb_t_axis_coverage_values(
            representative_leads,
            qrs,
            r_locs,
            fs,
            atrial_invalid,
        )
        if st_supported_t_axis_values is not None:
            st_supported_t_axis = _axis_from_amplitudes(st_supported_t_axis_values, t_confs)
            if st_supported_t_axis is not None:
                t_axis = st_supported_t_axis
                t_axis_values = st_supported_t_axis_values
                used_low_support_t_axis_coverage = True
    if (
        bool(beat_features)
        and
        qt_source == "low_qt_support_fallback"
        and qt_path_decision.reason == "insufficient_reliable_qt_leads"
        and not used_low_support_t_axis_coverage
    ):
        t_axis = None
    if atrial_invalid and _rr_cv(r_locs, fs) > _T_AXIS_IRREGULAR_RR_CV:
        t_axis = None
    if (
        bool(beat_features)
        and not selected_reliable_qt_leads
        and _rr_cv(r_locs, fs) > _T_AXIS_IRREGULAR_RR_CV
    ):
        # In an irregular record with no lead reliable enough for QT/T-end,
        # a plausible frontal T axis from relaxed amplitude fallbacks is not
        # independently supportable.
        t_axis = None
    st_axis  = _axis_from_amplitudes({k: v.params.get("st_mid_mv") for k, v in representative_leads.items()})
    qrs_t_angle = _axis_delta_deg(qrs_axis, t_axis)
    transition_zone: Optional[float] = None
    precordial_balance: List[tuple[int, float]] = []
    for index in range(1, 7):
        rep = representative_leads.get(f"V{index}")
        if rep is None:
            continue
        r_value = _valid_measurement("r_amp_mv", rep.params.get("r_amp_mv"))
        s_value = _valid_measurement("s_amp_mv", rep.params.get("s_amp_mv"))
        if r_value is None or s_value is None:
            continue
        precordial_balance.append((index, float(r_value - abs(s_value))))
    for (left_index, left_value), (right_index, right_value) in zip(
        precordial_balance,
        precordial_balance[1:],
    ):
        if left_value == 0.0:
            transition_zone = float(left_index)
            break
        if left_value < 0.0 <= right_value:
            denominator = right_value - left_value
            fraction = (-left_value / denominator) if denominator else 0.0
            transition_zone = float(left_index + fraction * (right_index - left_index))
            break

    # T027: PTF-V1 from V1 representative lead
    ptf_v1: Optional[float] = None
    v1_rep = representative_leads.get("V1")
    if v1_rep is not None:
        _ptf = v1_rep.params.get("ptf_v1_mv_ms")
        if _ptf is not None and np.isfinite(float(_ptf)):
            ptf_v1 = float(_ptf)

    # T-axis amplitude reliability: ≥2 limb leads with |T_amp| > 150µV (DXL spec)
    _T_AXIS_RELIABLE_AMP_MV = 0.15
    _T_AXIS_RELIABLE_MIN_LEADS = 2
    _limb_leads = {"I", "II", "III", "aVR", "aVL", "aVF"}
    _limb_t_amp_count = sum(
        1 for lead, rep in representative_leads.items()
        if lead in _limb_leads
        and rep.params.get("reliable_for_t", True)
        and (t_amp_val := rep.params.get("t_amp_mv")) is not None
        and abs(float(t_amp_val)) >= _T_AXIS_RELIABLE_AMP_MV
    )
    t_axis_reliable = _limb_t_amp_count >= _T_AXIS_RELIABLE_MIN_LEADS

    _qt_center_values = [
        float(rep.params["qt_consensus_ms"])
        for rep in representative_leads.values()
        if bool(rep.params.get("t_offset_fusion_reliable"))
        and rep.params.get("qt_consensus_ms") is not None
        and np.isfinite(float(rep.params["qt_consensus_ms"]))
    ]
    _qt_latest_values = [
        float(rep.params["qt_latest_p85_ms"])
        for rep in representative_leads.values()
        if bool(rep.params.get("t_offset_fusion_reliable"))
        and rep.params.get("qt_latest_p85_ms") is not None
        and np.isfinite(float(rep.params["qt_latest_p85_ms"]))
    ]
    qt_robust_center = (
        float(np.median(_qt_center_values)) if _qt_center_values else None
    )
    qt_latest_p85 = (
        float(np.median(_qt_latest_values)) if _qt_latest_values else None
    )
    t_fusion_support = int(t_fusion_quality["support"])
    t_fusion_mad = t_fusion_quality["mad_ms"]
    t_fusion_ci = t_fusion_quality["ci_half_width_ms"]
    t_fusion_reliable = bool(t_fusion_quality["reliable"])

    return GlobalFeatures(
        heart_rate_bpm=hr,
        atrial_rate_bpm=atrial_rate,
        pr_ms=pr,
        qrs_ms=qrs,
        qrs_wide_ms=qrs_wide,
        qt_ms=qt,
        qtc_bazett_ms=qtc_b,
        qtc_fridericia_ms=qtc_f,
        p_axis_deg=p_axis,
        qrs_axis_deg=qrs_axis,
        t_axis_deg=t_axis,
        st_axis_deg=st_axis,
        qt_dispersion_ms=qt_disp,
        qt_dispersion_independent_ms=qt_dispersion.independent_ms,
        qt_dispersion_p90_p10_ms=qt_dispersion.p90_p10_ms,
        qt_dispersion_source=qt_dispersion.legacy_source,
        qt_dispersion_used_leads=list(qt_dispersion.used_leads),
        qt_dispersion_legacy_excluded_leads=list(qt_dispersion.legacy_excluded_leads),
        p_duration_ms=p_duration,
        p_duration_source=p_duration_source,
        p_duration_used_leads=p_duration_used_leads,
        p_duration_support=p_duration_support,
        p_duration_spread_ms=p_duration_spread,
        p_duration_reliability=p_duration_reliability,
        ptf_v1_mv_ms=ptf_v1,
        t_axis_reliable=t_axis_reliable,
        qt_source=qt_source,
        qt_used_leads=qt_used_leads,
        qt_reliability=qt_reliability,
        qt_reportable=qt_reportable,
        qt_unreliable_reasons=qt_unreliable_reasons,
        qt_path=qt_path_decision.path,
        qt_confidence_reason=qt_confidence_reason,
        qt_excluded_leads=qt_path_decision.excluded_leads,
        qt_lead_weights=qt_path_decision.weights,
        consensus_vs_independent_per_lead=qt_path_decision.consensus_vs_independent_per_lead,
        qt_robust_center_ms=qt_robust_center,
        qt_latest_p85_ms=qt_latest_p85,
        t_fusion_support=t_fusion_support,
        t_fusion_mad_ms=t_fusion_mad,
        t_fusion_ci_half_width_ms=t_fusion_ci,
        t_fusion_reliable=t_fusion_reliable,
        t_fusion_cluster_count=int(t_fusion_quality["cluster_count"]),
        t_fusion_selected_cluster_score=t_fusion_quality[
            "selected_cluster_score"
        ],
        t_fusion_selected_cluster_support=int(
            t_fusion_quality["selected_cluster_support"]
        ),
        t_fusion_lead_groups=list(t_fusion_quality["lead_groups"]),
        t_fusion_methods=list(t_fusion_quality["methods"]),
        t_fusion_reliability_reasons=list(t_fusion_quality["reasons"]),
        t_global_tpte_ms=t_fusion_quality["tpte_ms"],
        t_derived_disagreement_ms=t_fusion_quality[
            "derived_disagreement_ms"
        ],
        t_tail_incomplete_leads=list(
            t_fusion_quality["tail_incomplete_leads"]
        ),
        t_tail_incomplete_fraction=float(
            t_fusion_quality["tail_incomplete_fraction"]
        ),
        t_systematic_early_risk=bool(
            t_fusion_quality["systematic_early_risk"]
        ),
        t_systematic_early_fraction=float(
            t_fusion_quality["systematic_early_fraction"]
        ),
        t_morphology_guard_pass=bool(
            t_fusion_quality["morphology_guard_pass"]
        ),
        heart_rate_min_bpm=heart_rate_min,
        heart_rate_max_bpm=heart_rate_max,
        qtc_framingham_ms=qtc_framingham,
        qtc_hodges_ms=qtc_hodges,
        rr_mean_ms=rr_mean,
        rr_sd_ms=rr_sd,
        rr_cv=rr_cv_metric,
        rmssd_ms=rmssd,
        pnn50=pnn50,
        poincare_sd1_ms=poincare_sd1,
        poincare_sd2_ms=poincare_sd2,
        rr_entropy=rr_entropy,
        qrs_t_angle_deg=qrs_t_angle,
        transition_zone=transition_zone,
    )
