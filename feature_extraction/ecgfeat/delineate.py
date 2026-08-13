from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .atrial import _composite_raw_p_detection
from .glasgow_measurements import measure_glasgow_profile
from .models import LeadBeatFeatures, WaveBounds, STANDARD_12_LEADS
from .numeric import trapezoid
from .p_morphology import measure_p_components
from .preprocess import bandpass_filter, lowpass_filter
from .qrs import _qrs_fiducial_from_local_waveform
from .repolarization import (
    TWaveMeasurement,
    candidates_from_triplets,
    detect_t_wave,
    infer_cluster_polarity,
)
from .twelve_sl import twelve_sl_wave_measurements_from_signal
from .u_wave import measure_u_wave


_EXPECTED_POSITIVE_T_LEADS = {"I", "II", "aVF", "V5", "V6"}
_T_FULL_WINDOW_RESCUE_EARLY_MAX_MS = 170.0
_T_FULL_WINDOW_RESCUE_LATE_MIN_MS = 180.0
_T_FULL_WINDOW_RESCUE_MIN_GAP_MS = 60.0
_T_FULL_WINDOW_RESCUE_MIN_CONF_RATIO = 0.45
_T_FULL_WINDOW_RESCUE_MAX_QRS_MS = 120.0
_T_LATE_ST_RESCUE_MAX_QRS_MS = 150.0
_PACED_QRS_FLOOR_MS = 120.0
_PACED_QRS_FLOOR_MIN_INTRINSIC_MS = 110.0
_QRS_TERMINAL_LATE_GAP_MS = 35.0
_QRS_TERMINAL_LOW_OFF_CONF = 0.45
_QRS_TERMINAL_MIN_SUPPORTED_LATE = 2
_QRS_TERMINAL_MORPH_AMP_MV = 0.04
_QRS_TERMINAL_P75_MIN_OFFSETS = 6
_QRS_OFFSET_REPAIR_MIN_SUPPORT = 4
_QRS_OFFSET_REPAIR_OUTLIER_MS = 35.0
_QRS_OFFSET_REPAIR_SEARCH_MARGIN_MS = 25.0
_QRS_OFFSET_REPAIR_LOW_CONF = 0.45
_QRS_OFFSET_REPAIR_HIGH_CONF = 0.70
_QRS_OFFSET_REPAIR_MIN_QRS_MS = 40.0
_QRS_OFFSET_REPAIR_MAX_QRS_MS = 220.0
_QRS_OFFSET_REPAIR_MAX_SHORTEN_RATIO = 0.65
# Clinical R selection in negative-dominant complexes; see
# _clinical_r_peak_local. The fraction sits on the flat 0.70-0.80 optimum of a
# LUDB sweep; the absolute floor keeps a pure QS complex from promoting noise.
_R_FIRST_POSITIVE_FRACTION = 0.80
_R_FIRST_POSITIVE_MIN_MV = 0.05
_QRS_OFFSET_REPAIR_PLAUSIBLE_SHORTEN_MAX_MS = 180.0
_QRS_TAIL_RESCUE_MAX_CURRENT_MS = 110.0
_QRS_TAIL_RESCUE_MIN_SELECTED_BEATS = 3
_QRS_TAIL_RESCUE_MIN_BEAT_FRACTION = 0.60
_QRS_TAIL_RESCUE_MIN_LEADS_PER_BEAT = 8
_QRS_TAIL_RESCUE_PLATEAU_START_MS = 105.0
_QRS_TAIL_RESCUE_PLATEAU_END_MS = 155.0
_QRS_TAIL_RESCUE_SEARCH_START_MS = 35.0
_QRS_TAIL_RESCUE_SEARCH_END_MS = 150.0
_QRS_TAIL_RESCUE_SETTLE_MS = 12.0
_QRS_TAIL_RESCUE_MIN_DELTA_MS = 20.0
_QRS_TAIL_RESCUE_MAX_DELTA_MS = 80.0
_QRS_TAIL_RESCUE_MIN_WIDTH_MS = 105.0
_QRS_TAIL_RESCUE_MAX_WIDTH_MS = 170.0
_P_BOUNDARY_CLUSTER_MS = 25.0
_P_BOUNDARY_MIN_SUPPORT = 3
_P_ONSET_CLUSTER_MS = 25.0
_P_ONSET_CLUSTER_MIN_SUPPORT = 3
_P_ONSET_CLUSTER_MAX_SPREAD_MS = 45.0
_P_ONSET_PHYSIOLOGIC_MIN_PR_MS = 120.0
_P_ONSET_PHYSIOLOGIC_MAX_PR_MS = 260.0
_P_DURATION_MIN_MS = 35.0
_P_DURATION_MAX_MS = 180.0
_PR_SEGMENT_MIN_MS = 8.0
_PR_SEGMENT_MAX_MS = 220.0
_P_BOUNDARY_CORRECTION_MIN_SHIFT_MS = 20.0
_P_BOUNDARY_CORRECTION_MAX_SPREAD_MS = 45.0
_P_OFFSET_CORRECTION_FRACTION = 0.50
_P_OFFSET_LATE_CLUSTER_MIN_PRIMARY_SUPPORT = 2
_P_OFFSET_LATE_CLUSTER_PRIMARY_RATIO = 0.75
_P_OFFSET_LATE_CLUSTER_MIN_SEPARATION_MS = 20.0
_P_PAIRED_DURATION_GUARD_PERCENTILE = 90.0
_P_PAIRED_DURATION_GUARD_MAX_EXTENSION_MS = 25.0
_P_PAIRED_DURATION_GUARD_MIN_SUPPORT = 3
_P_ONSET_TO_PEAK_MAX_MS = 125.0
_P_ONSET_EDGE_EXTENSION_MS = 12.0
_P_OFFSET_EDGE_EXTENSION_MS = 8.0
_P_FALSE_CANDIDATE_SHORT_PR_MS = 100.0
_P_FALSE_CANDIDATE_MIN_CLUSTER_DELTA_MS = 50.0
# Plausible P-peak-to-QRS-ONSET gap (note: onset, not R peak -- callers pass
# `qrs_on`). Calibrated against all 16790 expert P annotations in LUDB, whose
# distribution is p1 52 ms, p5 66 ms, p10 74 ms, median 102 ms, p99 180 ms.
# The former 80 ms floor sat above the 10th percentile and so declared 15% of
# genuine P waves implausible: in `_fuse_peak_anchor` they lose the 1.25x
# physiologic bonus *and* take the 0.02x short-gap penalty, a 62x swing that
# hands the anchor to a spurious candidate near the far edge of the window.
# Because every lead sees the same geometry the whole record then agrees on
# the wrong anchor, and the +/-40 ms window it imposes excludes the real P in
# all 12 leads at once. 45 ms keeps 99.5% of real P waves while still
# rejecting deflections that sit on the QRS itself (<40 ms is 0.1%).
_P_PEAK_QRS_GAP_MIN_MS = 60.0
_P_PEAK_QRS_GAP_MAX_MS = 300.0
_P_SHORT_P_PEAK_QRS_GAP_MS = 60.0
_P_FUSED_PEAK_MIN_AMP_RATIO = 0.35
_P_FUSED_PEAK_CLOSE_AMP_RATIO = 0.85
_P_FUSED_ANCHOR_SUPPORT_POWER = 1.5
_P_FUSED_ANCHOR_PHYSIOLOGIC_BONUS = 1.25
_P_FUSED_ANCHOR_SHORT_GAP_PENALTY = 0.02
_P_FUSED_ANCHOR_SINGLE_LEAD_PENALTY = 0.15
# Minimum number of corroborating leads before the fused P anchor is allowed to
# steer a per-lead decision.  The scoring already demotes single-lead clusters
# (_P_FUSED_ANCHOR_SINGLE_LEAD_PENALTY), but a demoted cluster can still be the
# winner, and the winner is then used to clamp the per-lead search window to
# +/-P_PRIOR_MARGIN.  An uncorroborated anchor that lands on the wrong
# deflection therefore removes the true P wave from the searchable range
# entirely, leaving the SNR-gated refined engine no way to recover.
#
# The requirement is deliberately *relative*: it applies only when at least this
# many leads were eligible to corroborate in the first place.  On genuinely
# single-lead input one supporting lead is the maximum achievable, so demanding
# two would disable the anchor exactly where it is the only available evidence.
# What the check targets is one lead disagreeing with several others, not one
# lead being alone.
_P_ANCHOR_MIN_LEAD_SUPPORT = 2
_P_CONTEXT_NEUTRAL_SCORE = 0.5
_P_CONTEXT_TEMPLATE_HALF_MS = 80.0
_P_CONTEXT_MIN_TEMPLATE_SEEDS = 3
_P_CONTEXT_SUPPORT_RADIUS_MS = 25.0
_P_CONTEXT_STRONG_SUPPORT = 4
_P_CONTEXT_MODERATE_SUPPORT = 2
_P_CONTEXT_MIN_PR_VALUES = 4
_P_CONTEXT_MIN_PP_VALUES = 4
_P_CONTEXT_PR_STABLE_MAD_MS = 25.0
_P_CONTEXT_PP_STABLE_MAD_MS = 35.0
_P_CONTEXT_SCORE_LOW = 0.35
_P_CONTEXT_COMPONENT_LOW = 0.35
_P_CONTEXT_SHORT_PR_NO_LIMB_MS = 100.0
_P_CONTEXT_LIMB_LEADS = {"I", "II", "III", "aVR", "aVL", "aVF"}
_P_RESELECT_REVIEW_SCORE = 0.50
_P_RESELECT_MIN_NEW_SCORE = 0.55
_P_RESELECT_MIN_SCORE_DELTA = 0.20
_P_RESELECT_TEMPLATE_GAIN = 0.25
_P_RESELECT_TIMING_GAIN = 0.25
_P_RAW_ATRIAL_EVENT_MIN_CONFIDENCE = 0.35
_P_RAW_ATRIAL_EVENT_DUPLICATE_RADIUS_MS = 25.0
_P_RAW_ATRIAL_EVENT_REVIEW_SCORE = 0.65
_P_DETECTION_LOWPASS_HZ = 35.0
_P_LOCAL_SNR_MIN = 2.0
_P_LOCAL_NOISE_FLOOR_MV = 0.002
_P_TP_QUIET_GUARD_MS = 8.0
_P_TP_QUIET_MIN_MS = 16.0
_P_TP_CONTEXT_MIN_MS = 30.0
_P_TA_VISIBLE_PR_MIN_MS = 16.0
# A Ta (atrial repolarisation) deflection is present in the PR segment of every
# ECG and is always opposite in polarity to P, so its mere presence says nothing
# about P-offset reliability. Only a Ta comparable to the P wave itself can drag
# the offset. Scale the threshold with P amplitude; the absolute floor exists
# only to stop division-by-noise on flat leads, and is deliberately far below
# the fraction so it never binds on a normally-sized P.
_P_TA_OVERLAP_P_FRACTION = 0.25
_P_TA_OVERLAP_FLOOR_MV = 0.010
_P_NO_TP_BOUNDARY_CONFIDENCE_MAX = 0.45
_P_TA_OFFSET_CONFIDENCE_MAX = 0.55
_P_BOUNDARY_SIGMA_REFERENCE_MS = 12.0
_P_BOUNDARY_SIGMA_ORDER_STAT_MAX_MS = 30.0
_P_BOUNDARY_PERTURBATION_FRACTIONS = (0.04, 0.06, 0.08, 0.10, 0.12)


@dataclass
class BeatWindow:
    start: int
    r:     int
    end:   int


@dataclass
class PCandidateAlternative:
    peak: int
    amp: float
    area: float


@dataclass
class PCandidateMeasurement:
    onset: int
    peak: int
    offset: int
    pr_ms: Optional[float]
    p_dur_ms: float
    p_amp_mv: float
    p_area: float
    p_signed_area: float
    p_confidence: float
    p_components: Dict[str, object]
    ptf_v1_mv_ms: Optional[float]
    context_score: float
    template_corr: float
    multilead_support: int
    multilead_support_score: float
    pr_consistency_score: float
    pp_consistency_score: float


@dataclass
class QRSOffsetRepairTarget:
    offset: int
    support: int
    used_leads: str
    excluded_leads: str


@dataclass
class STJRepairTarget:
    j_index: int
    support: int
    used_leads: str
    excluded_leads: str


@dataclass
class TEndRepairTarget:
    offset: int
    support: int
    used_leads: str
    excluded_leads: str
    early_outlier_leads: str
    late_outlier_leads: str


@dataclass
class TDualEndpointEvidence:
    chord_index: Optional[int]
    slope_index: Optional[int]
    tangent_index: Optional[int]
    best_index: Optional[int]
    method_spread_ms: Optional[float]
    confidence: float
    reason: str
    possible_t_u_fusion: bool = False
    flat_t_wave: bool = False


@dataclass
class TDualRescueTarget:
    offset: int
    support: int
    used_leads: str
    excluded_leads: str
    early_outlier_leads: str


# ─────────────────────────────────────────────────────────────────────────────
# Low-level primitives
# ─────────────────────────────────────────────────────────────────────────────

def _edge_preserving_moving_average(
    signal: np.ndarray,
    window: int,
) -> np.ndarray:
    """Smooth without introducing a zero-padding step at a beat boundary.

    ``np.convolve(..., mode="same")`` implicitly pads with zeros. Beat
    fragments generally start at a non-zero ECG level, so that padding creates
    a large, perfectly time-aligned turning point at half a smoothing window
    from sample zero. Cross-lead fusion can then mistake the shared numerical
    edge for a strongly corroborated P wave. Extending the endpoint value
    keeps the output length and moving-average response while avoiding a
    boundary that is not present in the signal.
    """
    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        return values.copy()
    width = max(1, int(window))
    if width == 1:
        return values.copy()
    pad_left = (width - 1) // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _find_peak(
    sig: np.ndarray, lo: int, hi: int, mode: str = "abs"
) -> Optional[int]:
    lo = max(0, lo)
    hi = min(len(sig), hi)
    if hi <= lo:
        return None
    x = sig[lo:hi]
    if mode == "abs":
        idx = int(np.argmax(np.abs(x)))
    elif mode == "pos":
        idx = int(np.argmax(x))
    else:
        idx = int(np.argmin(x))
    return lo + idx


def _p_context_snippet(
    sig: np.ndarray,
    center: Optional[int],
    half_width: int,
) -> Optional[np.ndarray]:
    if center is None or half_width <= 0:
        return None
    center_i = int(center)
    lo = center_i - int(half_width)
    hi = center_i + int(half_width) + 1
    if lo < 0 or hi > len(sig) or hi <= lo:
        return None
    snippet = sig[lo:hi].astype(float)
    snippet = snippet - float(np.median(snippet))
    scale = float(np.max(np.abs(snippet)))
    if scale <= 1e-9:
        return None
    return snippet / scale


def _p_context_corr(candidate: Optional[np.ndarray], template: Optional[np.ndarray]) -> float:
    if candidate is None or template is None:
        return _P_CONTEXT_NEUTRAL_SCORE
    if len(candidate) != len(template) or len(candidate) < 3:
        return _P_CONTEXT_NEUTRAL_SCORE
    cand = np.asarray(candidate, dtype=float)
    tmpl = np.asarray(template, dtype=float)
    cand = cand - float(np.mean(cand))
    tmpl = tmpl - float(np.mean(tmpl))
    denom = float(np.linalg.norm(cand) * np.linalg.norm(tmpl))
    if denom <= 1e-9:
        return _P_CONTEXT_NEUTRAL_SCORE
    corr = float(np.dot(cand, tmpl) / denom)
    return float(np.clip(max(corr, -corr), 0.0, 1.0))


def _p_context_robust_score(
    value: Optional[float],
    center: Optional[float],
    scale: Optional[float],
    *,
    floor_scale: float = 12.0,
) -> float:
    if value is None or center is None or scale is None:
        return _P_CONTEXT_NEUTRAL_SCORE
    if not (np.isfinite(value) and np.isfinite(center) and np.isfinite(scale)):
        return _P_CONTEXT_NEUTRAL_SCORE
    sigma = max(float(scale), float(floor_scale), 1e-6)
    z = abs(float(value) - float(center)) / sigma
    return float(np.clip(np.exp(-0.5 * z * z), 0.0, 1.0))


def _p_boundary_confidence_scores(
    *,
    p_on: Optional[int],
    p_peak: Optional[int],
    p_off: Optional[int],
    qrs_ref: Optional[int],
    fs: int,
    p_confidence: float,
    support_score: float = 0.0,
    onset_sigma_ms: Optional[float] = None,
    offset_sigma_ms: Optional[float] = None,
    informative: Optional[bool] = None,
    quiet_window_available: Optional[bool] = None,
    on_t_overlap_risk: Optional[bool] = None,
    ta_overlap_risk: Optional[bool] = None,
) -> Tuple[float, float]:
    if fs <= 0 or p_on is None or p_peak is None or p_off is None:
        return 0.0, 0.0
    if not (int(p_on) < int(p_peak) < int(p_off)):
        return 0.0, 0.0
    base = float(np.clip(float(p_confidence or 0.0), 0.0, 1.0))
    support = float(np.clip(float(support_score or 0.0), 0.0, 1.0))
    p_dur_ms = (int(p_off) - int(p_on)) * 1000.0 / float(fs)
    dur_score = 1.0 if _P_DURATION_MIN_MS <= p_dur_ms <= _P_DURATION_MAX_MS else 0.0
    onset_to_peak_ms = (int(p_peak) - int(p_on)) * 1000.0 / float(fs)
    onset_arm_score = 1.0 if onset_to_peak_ms <= _P_ONSET_TO_PEAK_MAX_MS else 0.0
    if qrs_ref is None:
        pr_segment_score = 0.65
    else:
        pr_segment_ms = (int(qrs_ref) - int(p_off)) * 1000.0 / float(fs)
        pr_segment_score = 1.0 if _PR_SEGMENT_MIN_MS <= pr_segment_ms <= _PR_SEGMENT_MAX_MS else 0.0
    onset_conf = float(np.clip(
        0.65 * base + 0.20 * dur_score + 0.15 * onset_arm_score,
        0.0,
        1.0,
    ))
    offset_conf = float(np.clip(
        0.55 * base + 0.20 * dur_score + 0.20 * pr_segment_score + 0.05 * support,
        0.0,
        1.0,
    ))

    def _stability_factor(sigma_ms: Optional[float]) -> float:
        if sigma_ms is None:
            return 1.0
        try:
            sigma = float(sigma_ms)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(sigma) or sigma < 0.0:
            return 1.0
        relative = sigma / _P_BOUNDARY_SIGMA_REFERENCE_MS
        stability = 1.0 / (1.0 + relative * relative)
        # Stability is an independent modifier, but do not erase otherwise
        # useful morphology solely because two deterministic methods disagree.
        return float(0.60 + 0.40 * stability)

    onset_conf *= _stability_factor(onset_sigma_ms)
    offset_conf *= _stability_factor(offset_sigma_ms)
    if informative is False:
        onset_conf = min(onset_conf, 0.20)
        offset_conf = min(offset_conf, 0.20)
    # No-TP/P-on-T cases have no observable isoelectric reference.  Keep the
    # P-presence score intact, but never advertise a high-confidence boundary.
    if quiet_window_available is False or on_t_overlap_risk is True:
        onset_conf = min(onset_conf, _P_NO_TP_BOUNDARY_CONFIDENCE_MAX)
        offset_conf = min(offset_conf, _P_NO_TP_BOUNDARY_CONFIDENCE_MAX)
    # Ta begins after atrial depolarisation and can visibly bend the exposed PR
    # interval.  This is chiefly an offset/baseline limitation, not evidence
    # that the P wave itself is absent.
    if ta_overlap_risk is True:
        offset_conf = min(offset_conf, _P_TA_OFFSET_CONFIDENCE_MAX)
    return onset_conf, offset_conf


def _robust_p_noise_rms(segment: np.ndarray) -> Optional[float]:
    """Estimate local noise after removing a linear quiet-segment trend."""
    values = np.asarray(segment, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return None
    axis = np.arange(values.size, dtype=float)
    try:
        slope, intercept = np.polyfit(axis, values, 1)
        residual = values - (slope * axis + intercept)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        residual = values - float(np.median(values))
    center = float(np.median(residual))
    mad_noise = 1.4826 * float(np.median(np.abs(residual - center)))
    diffs = np.diff(residual)
    diff_center = float(np.median(diffs))
    diff_noise = (
        1.4826 * float(np.median(np.abs(diffs - diff_center))) / np.sqrt(2.0)
        if diffs.size
        else 0.0
    )
    return float(max(mad_noise, diff_noise, _P_LOCAL_NOISE_FLOOR_MV))


def _p_local_noise_estimate(
    sig: np.ndarray,
    *,
    p_on: int,
    previous_t_off: Optional[int],
    allow_unverified_pre_p: bool,
    fs: int,
) -> Tuple[Optional[float], Optional[str], Optional[bool]]:
    """Estimate P-local noise only where a P-free interval is defensible.

    The PR segment is intentionally excluded: it contains atrial
    repolarisation (Ta) and is not a true noise-only reference.  When the
    previous T offset is known, only the observable TP interval is eligible.
    The first beat may use an explicitly labelled, unverified pre-P fallback
    because its preceding T wave is outside the record context.
    """
    if fs <= 0:
        return None, None, None
    n_samples = len(sig)
    guard = max(1, int(round(_P_TP_QUIET_GUARD_MS * fs / 1000.0)))
    minimum = max(4, int(round(_P_TP_QUIET_MIN_MS * fs / 1000.0)))
    windows: List[Tuple[str, int, int]] = []

    if previous_t_off is not None:
        lo = max(
            0,
            int(previous_t_off) + guard,
            int(p_on) - int(round(0.080 * fs)),
        )
        hi = min(n_samples, int(p_on) - guard)
        if hi - lo >= minimum:
            windows.append(("tp_quiet", lo, hi))
        else:
            return None, "no_tp_quiet_p_on_t", False
    elif allow_unverified_pre_p:
        pre_near_lo = max(0, int(p_on) - int(round(0.070 * fs)))
        pre_near_hi = min(n_samples, int(p_on) - int(round(0.015 * fs)))
        if pre_near_hi - pre_near_lo >= minimum:
            windows.append(
                ("pre_p_unverified_previous_t_out_of_context", pre_near_lo, pre_near_hi)
            )
        pre_far_lo = max(0, int(p_on) - int(round(0.140 * fs)))
        pre_far_hi = min(n_samples, int(p_on) - int(round(0.085 * fs)))
        if pre_far_hi - pre_far_lo >= minimum:
            windows.append(
                ("far_pre_p_unverified_previous_t_out_of_context", pre_far_lo, pre_far_hi)
            )
    else:
        return None, "previous_t_offset_unavailable", None

    estimates: List[Tuple[float, str]] = []
    for source, lo, hi in windows:
        estimate = _robust_p_noise_rms(sig[lo:hi])
        if estimate is not None and np.isfinite(estimate):
            estimates.append((float(estimate), source))
    if not estimates:
        return None, "no_valid_p_local_noise_window", None

    # Multiple unverified first-beat windows can be available.  Preserve the
    # conservative physical floor while choosing the least contaminated one.
    estimate, source = min(estimates, key=lambda item: item[0])
    quiet_available = True if source == "tp_quiet" else None
    return float(estimate), source, quiet_available


def _robust_polynomial_trend(
    sig: np.ndarray,
    sample_indices: np.ndarray,
    *,
    center: int,
    degree: int = 2,
) -> Optional[np.ndarray]:
    """Fit a continuous low-order trend with deterministic Huber reweighting."""
    x_idx = np.asarray(sample_indices, dtype=int)
    x_idx = x_idx[(x_idx >= 0) & (x_idx < len(sig))]
    if x_idx.size < max(8, degree + 3):
        return None
    scale = max(1.0, float(np.max(np.abs(x_idx - int(center)))))
    x = (x_idx.astype(float) - float(center)) / scale
    y = np.asarray(sig[x_idx], dtype=float)
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if y.size < max(8, degree + 3):
        return None
    design = np.vander(x, N=degree + 1, increasing=True)
    weights = np.ones(y.size, dtype=float)
    coefficients: Optional[np.ndarray] = None
    for _ in range(5):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        try:
            coefficients, *_ = np.linalg.lstsq(
                weighted_design,
                weighted_y,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return None
        residual = y - design @ coefficients
        residual_center = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - residual_center)))
        if sigma <= 1e-9:
            break
        huber_limit = 1.5 * sigma
        absolute = np.abs(residual - residual_center)
        weights = np.ones_like(absolute)
        outliers = absolute > huber_limit
        weights[outliers] = huber_limit / np.maximum(absolute[outliers], 1e-12)
    if coefficients is None:
        return None
    full_x = (np.arange(len(sig), dtype=float) - float(center)) / scale
    full_design = np.vander(full_x, N=degree + 1, increasing=True)
    return np.asarray(full_design @ coefficients, dtype=float)


def _p_baseline_view(
    sig: np.ndarray,
    *,
    p_on: int,
    p_peak: int,
    p_off: int,
    qrs_on: Optional[int],
    previous_t_off: Optional[int],
    quiet_window_available: Optional[bool],
    fs: int,
) -> Tuple[np.ndarray, str, float]:
    """Return a P-detection view with an explicit baseline provenance.

    With a real TP interval the view uses its median.  When TP has disappeared,
    a smooth robust quadratic is fitted to the two sidebands around the already
    detected P candidate.  The latter is a *low-confidence reconstruction*: it
    is suitable for stability diagnostics but never upgrades/overwrites the
    raw P boundary on its own.
    """
    values = np.asarray(sig, dtype=float)
    guard = max(1, int(round(_P_TP_QUIET_GUARD_MS * fs / 1000.0)))
    minimum = max(4, int(round(_P_TP_QUIET_MIN_MS * fs / 1000.0)))
    if quiet_window_available is True and previous_t_off is not None:
        lo = max(
            0,
            int(previous_t_off) + guard,
            int(p_on) - int(round(0.080 * fs)),
        )
        hi = min(len(values), int(p_on) - guard)
        if hi - lo >= minimum:
            baseline = float(np.median(values[lo:hi]))
            return values - baseline, "tp_median", 0.90

    left_lo = max(0, int(p_on) - int(round(0.080 * fs)))
    left_hi = max(left_lo, int(p_on) - guard)
    right_lo = min(len(values), int(p_off) + guard)
    right_cap = (
        min(len(values), int(qrs_on) - guard)
        if qrs_on is not None
        else len(values)
    )
    right_hi = min(right_cap, int(p_off) + int(round(0.060 * fs)))
    sidebands: List[np.ndarray] = []
    if left_hi - left_lo >= minimum:
        sidebands.append(np.arange(left_lo, left_hi, dtype=int))
    if right_hi - right_lo >= minimum:
        sidebands.append(np.arange(right_lo, right_hi, dtype=int))
    if len(sidebands) == 2:
        indices = np.concatenate(sidebands)
        trend = _robust_polynomial_trend(
            values,
            indices,
            center=int(p_peak),
            degree=2,
        )
        if trend is not None:
            return (
                values - trend,
                "robust_quadratic_overlap_context",
                0.25,
            )

    baseline_ref = int(qrs_on) if qrs_on is not None else int(p_off)
    baseline, _source, _confidence = _baseline_contract(values, baseline_ref, fs)
    return values - baseline, "legacy_window_fallback", 0.10


_P_AMP_REF_GUARD_MS = 6.0      # stay clear of the P feet and of QRS onset
_P_AMP_REF_MIN_MS = 16.0       # shortest usable isoelectric stretch
_P_AMP_REF_SPAN_MS = 60.0      # how far to look on each side


def _p_isoelectric_reference(
    sig: np.ndarray,
    *,
    p_on: Optional[int],
    p_off: Optional[int],
    qrs_on: Optional[int],
    fs: int,
    fallback: float,
) -> Tuple[float, str]:
    """Isoelectric level for P *amplitude*, measured off the P wave itself.

    `_baseline_contract` medians the window ref-220 ms .. ref-80 ms, which in
    sinus rhythm is where the P wave lives: measured across PTB-XL 09000 that
    window is 51-56% occupied by the P in every lead, so its median is dragged
    toward the P's own deflection and every `peak - baseline` amplitude is biased
    toward zero.  Lead II measured 81 µV against 136 µV from a P-free reference
    (-40%), aVR -67 against -115 µV, and only 0.7% of records reached the 0.24 mV
    RAE bar against 3.3% -- which is why RAE effectively never fired.

    The reference is the isoelectric tissue on either side of the P wave: the PR
    segment (P offset -> QRS onset) and the TP tail (before P onset).  Both are
    used when both exist, so a baseline drifting through the beat cancels instead
    of biasing the level, mirroring the two-sideband fit in `_p_baseline_view`.

    Boundary detection deliberately keeps using the passed-in `baseline`: that
    path is tuned and benchmarked against LUDB, and amplitude referencing is a
    separate question from where the tangent crossing lands.
    """
    values = np.asarray(sig, dtype=float)
    if p_on is None or p_off is None or fs <= 0:
        return float(fallback), "legacy_window_fallback"
    guard = max(1, int(round(_P_AMP_REF_GUARD_MS * fs / 1000.0)))
    minimum = max(2, int(round(_P_AMP_REF_MIN_MS * fs / 1000.0)))
    span = max(minimum, int(round(_P_AMP_REF_SPAN_MS * fs / 1000.0)))

    windows: List[np.ndarray] = []
    left_hi = max(0, int(p_on) - guard)
    left_lo = max(0, left_hi - span)
    if left_hi - left_lo >= minimum:
        windows.append(values[left_lo:left_hi])
    right_lo = min(len(values), int(p_off) + guard)
    right_cap = min(len(values), int(qrs_on) - guard) if qrs_on is not None else len(values)
    right_hi = min(right_cap, right_lo + span)
    if right_hi - right_lo >= minimum:
        windows.append(values[right_lo:right_hi])
    if not windows:
        return float(fallback), "legacy_window_fallback"
    level = float(np.median(np.concatenate(windows)))
    if not np.isfinite(level):
        return float(fallback), "legacy_window_fallback"
    source = "pr_and_tp_sidebands" if len(windows) == 2 else "single_sideband"
    return level, source


def _robust_boundary_sigma_ms(values: List[int], fs: int) -> Optional[float]:
    if fs <= 0 or len(values) < 3:
        return None
    samples = np.asarray(values, dtype=float)
    center = float(np.median(samples))
    sigma_samples = 1.4826 * float(np.median(np.abs(samples - center)))
    return float(sigma_samples * 1000.0 / float(fs))


def _p_boundary_stability(
    p_signal: np.ndarray,
    *,
    p_peak: int,
    p_on: int,
    p_off: int,
    search_lo: int,
    p_off_limit: int,
    baseline: float,
    fs: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Measure deterministic boundary dispersion across nearby methods."""
    if fs <= 0 or not (0 <= p_on < p_peak < p_off < len(p_signal)):
        return None, None
    onset_values = [int(p_on)]
    offset_values = [int(p_off)]

    tangent_on = _t_onset_tangent(
        p_signal,
        int(p_peak),
        max(0, int(search_lo)),
        float(baseline),
        fs,
    )
    tangent_off = _p_offset_tangent(
        p_signal,
        int(p_peak),
        min(len(p_signal) - 1, int(p_off_limit)),
        float(baseline),
        fs,
    )
    if tangent_on is not None:
        onset_values.append(int(tangent_on))
    if tangent_off is not None:
        offset_values.append(int(tangent_off))

    for fraction in _P_BOUNDARY_PERTURBATION_FRACTIONS:
        candidate_on, candidate_off = _find_wave_bounds(
            p_signal,
            int(p_peak),
            float(baseline),
            thresh_frac=float(fraction),
        )
        if (
            candidate_on is not None
            and int(search_lo) <= int(candidate_on) < int(p_peak)
            and (int(p_peak) - int(candidate_on)) * 1000.0 / fs
            <= _P_ONSET_TO_PEAK_MAX_MS
        ):
            onset_values.append(int(candidate_on))
        if (
            candidate_off is not None
            and int(p_peak) < int(candidate_off) <= int(p_off_limit)
            and (int(candidate_off) - int(p_peak)) * 1000.0 / fs <= _P_DURATION_MAX_MS
        ):
            offset_values.append(int(candidate_off))

    return (
        _robust_boundary_sigma_ms(onset_values, fs),
        _robust_boundary_sigma_ms(offset_values, fs),
    )


def _p_boundary_fusion_weight(
    feature: "LeadBeatFeatures",
    *,
    boundary: str,
) -> Tuple[float, bool]:
    """Return fusion weight and whether P-specific diagnostics are available."""
    if getattr(feature, "p_informative", None) is False:
        return 0.0, True
    if (
        getattr(feature, "p_on_t_overlap_risk", None) is True
        or getattr(feature, "p_quiet_window_available", None) is False
    ):
        # When no TP reference exists, deterministic methods can agree on the
        # same T-tail feature.  Agreement is therefore not enough to justify
        # an extreme onset/offset order statistic; retain the median fallback.
        return 0.0, True
    confidence = getattr(feature, f"p_{boundary}_confidence", None)
    if confidence is None:
        confidence = getattr(feature, "p_confidence", 0.0)
    confidence_value = float(np.clip(float(confidence or 0.0), 0.0, 1.0))

    sigma = getattr(feature, f"p_{boundary}_sigma_ms", None)
    has_sigma = sigma is not None
    if sigma is None:
        stability = 1.0
    else:
        sigma_value = max(0.0, float(sigma))
        relative = sigma_value / _P_BOUNDARY_SIGMA_REFERENCE_MS
        stability = 1.0 / (1.0 + relative * relative)

    snr = getattr(feature, "p_local_snr", None)
    has_snr = snr is not None
    if snr is None:
        snr_gate = 1.0
    else:
        snr_gate = float(np.clip((float(snr) - 1.0) / 3.0, 0.0, 1.0))
    ta_gate = (
        0.60
        if boundary == "offset"
        and getattr(feature, "p_ta_overlap_risk", None) is True
        else 1.0
    )
    return (
        float(confidence_value * stability * snr_gate * ta_gate),
        bool(has_sigma or has_snr),
    )


def _confidence_gated_boundary_order_stat(
    candidates: List[Tuple["LeadBeatFeatures", int]],
    *,
    boundary: str,
) -> Tuple[Optional[int], Optional[str]]:
    """Choose a protected early onset or late offset when diagnostics agree."""
    if not candidates:
        return None, None
    eligible: List[Tuple[int, float]] = []
    for feature, value in candidates:
        weight, has_diagnostics = _p_boundary_fusion_weight(
            feature,
            boundary=boundary,
        )
        sigma = getattr(feature, f"p_{boundary}_sigma_ms", None)
        if (
            has_diagnostics
            and weight > 0.0
            and (
                sigma is None
                or float(sigma) <= _P_BOUNDARY_SIGMA_ORDER_STAT_MAX_MS
            )
        ):
            eligible.append((int(value), float(weight)))

    if len(eligible) < 4:
        values = [int(value) for _feature, value in candidates]
        return int(round(float(np.median(values)))), "cluster_median"

    ordered = sorted(value for value, _weight in eligible)
    if boundary == "onset":
        return int(ordered[1]), "confidence_gated_second_earliest"
    return int(ordered[-2]), "confidence_gated_second_latest"


def _refresh_p_boundary_diagnostics(
    beat_features: List["LeadBeatFeatures"],
    *,
    ecg: np.ndarray,
    fs: int,
) -> List["LeadBeatFeatures"]:
    """Populate local P SNR and boundary dispersion after candidate re-selection."""
    ecg_arr = np.asarray(ecg, dtype=float)
    if fs <= 0 or ecg_arr.ndim != 2 or not beat_features:
        return beat_features
    cutoff = min(_P_DETECTION_LOWPASS_HZ, 0.45 * float(fs))
    if cutoff <= 0.0:
        return beat_features
    p_detection_ecg = lowpass_filter(ecg_arr, fs, cutoff_hz=cutoff, order=3)
    lead_to_index = {
        lead: index
        for index, lead in enumerate(STANDARD_12_LEADS)
        if index < ecg_arr.shape[0]
    }
    t_offsets_by_beat: Dict[int, int] = {}
    for beat_id in sorted({int(feature.beat_id) for feature in beat_features}):
        offsets = [
            int(feature.t.offset)
            for feature in beat_features
            if int(feature.beat_id) == beat_id and feature.t.offset is not None
        ]
        if offsets:
            t_offsets_by_beat[beat_id] = int(round(float(np.median(offsets))))

    for feature in beat_features:
        if (
            feature.p.onset is None
            or feature.p.peak is None
            or feature.p.offset is None
        ):
            continue
        lead_index = lead_to_index.get(feature.lead)
        if lead_index is None:
            continue
        p_on = int(feature.p.onset)
        p_peak = int(feature.p.peak)
        p_off = int(feature.p.offset)
        if not (0 <= p_on < p_peak < p_off < ecg_arr.shape[1]):
            continue
        qrs_on = (
            int(feature.qrs.onset)
            if feature.qrs.onset is not None
            else None
        )
        previous_t_off = t_offsets_by_beat.get(int(feature.beat_id) - 1)
        tp_gap_ms = (
            (int(p_on) - int(previous_t_off)) * 1000.0 / float(fs)
            if previous_t_off is not None
            else None
        )
        on_t_overlap_risk = (
            bool(tp_gap_ms < _P_TP_CONTEXT_MIN_MS)
            if tp_gap_ms is not None
            else None
        )
        raw_signal = ecg_arr[lead_index]
        p_signal = p_detection_ecg[lead_index]
        noise, noise_source, quiet_window_available = _p_local_noise_estimate(
            raw_signal,
            p_on=p_on,
            previous_t_off=previous_t_off,
            allow_unverified_pre_p=int(feature.beat_id) == 0,
            fs=fs,
        )
        raw_p_view, baseline_method, baseline_confidence = _p_baseline_view(
            raw_signal,
            p_on=p_on,
            p_peak=p_peak,
            p_off=p_off,
            qrs_on=qrs_on,
            previous_t_off=previous_t_off,
            quiet_window_available=quiet_window_available,
            fs=fs,
        )
        p_detection_view, _method, _confidence = _p_baseline_view(
            p_signal,
            p_on=p_on,
            p_peak=p_peak,
            p_off=p_off,
            qrs_on=qrs_on,
            previous_t_off=previous_t_off,
            quiet_window_available=quiet_window_available,
            fs=fs,
        )
        p_amplitude = abs(float(raw_p_view[p_peak]))
        ta_overlap_risk: Optional[bool] = None
        if (
            quiet_window_available is True
            and qrs_on is not None
            and (int(qrs_on) - int(p_off)) * 1000.0 / float(fs)
            >= _P_TA_VISIBLE_PR_MIN_MS
        ):
            guard = max(
                1,
                int(round(_P_TP_QUIET_GUARD_MS * fs / 1000.0)),
            )
            ta_lo = min(len(raw_p_view), int(p_off) + guard)
            ta_hi = min(len(raw_p_view), int(qrs_on) - guard)
            if ta_hi - ta_lo >= max(
                4,
                int(round(_P_TP_QUIET_MIN_MS * fs / 1000.0)),
            ):
                p_polarity = 1.0 if float(raw_p_view[p_peak]) >= 0.0 else -1.0
                opposite_excursion = float(
                    np.max(-p_polarity * raw_p_view[ta_lo:ta_hi])
                )
                ta_threshold = max(
                    _P_TA_OVERLAP_P_FRACTION * p_amplitude,
                    _P_TA_OVERLAP_FLOOR_MV,
                    2.0 * float(noise)
                    if noise is not None
                    else _P_LOCAL_NOISE_FLOOR_MV * 2.0,
                )
                ta_overlap_risk = bool(opposite_excursion >= ta_threshold)
        p_snr = (
            float(p_amplitude / max(float(noise), _P_LOCAL_NOISE_FLOOR_MV))
            if noise is not None
            else None
        )
        informative = (
            bool(p_snr >= _P_LOCAL_SNR_MIN)
            if p_snr is not None
            else None
        )
        search_lo = max(
            0,
            p_peak - int(round(0.140 * fs)),
            p_on - int(round(0.080 * fs)),
        )
        p_off_limit = (
            min(len(p_signal) - 1, qrs_on - 1)
            if qrs_on is not None
            else min(len(p_signal) - 1, p_off + int(round(0.080 * fs)))
        )
        onset_sigma, offset_sigma = _p_boundary_stability(
            p_detection_view,
            p_peak=p_peak,
            p_on=p_on,
            p_off=p_off,
            search_lo=search_lo,
            p_off_limit=p_off_limit,
            baseline=0.0,
            fs=fs,
        )

        feature.p_local_noise_rms_mv = noise
        feature.p_local_noise_source = noise_source
        feature.p_local_snr = p_snr
        feature.p_informative = informative
        feature.p_tp_gap_ms = tp_gap_ms
        feature.p_quiet_window_available = quiet_window_available
        feature.p_on_t_overlap_risk = on_t_overlap_risk
        feature.p_ta_overlap_risk = ta_overlap_risk
        feature.p_baseline_method = baseline_method
        feature.p_baseline_confidence = baseline_confidence
        feature.p_onset_sigma_ms = onset_sigma
        feature.p_offset_sigma_ms = offset_sigma
        feature.p_boundary_stability_method = (
            f"lp35_{baseline_method}_tangent_threshold_perturbation"
        )
        if informative is False:
            if "p_isoelectric_uninformative" not in feature.flags:
                feature.flags.append("p_isoelectric_uninformative")
        if quiet_window_available is False:
            if "p_no_tp_quiet_window" not in feature.flags:
                feature.flags.append("p_no_tp_quiet_window")
        if on_t_overlap_risk is True:
            if "p_on_t_overlap_risk" not in feature.flags:
                feature.flags.append("p_on_t_overlap_risk")
        if ta_overlap_risk is True:
            if "p_ta_offset_risk" not in feature.flags:
                feature.flags.append("p_ta_offset_risk")

        onset_conf, offset_conf = _p_boundary_confidence_scores(
            p_on=p_on,
            p_peak=p_peak,
            p_off=p_off,
            qrs_ref=qrs_on,
            fs=fs,
            p_confidence=float(feature.p_confidence or 0.0),
            onset_sigma_ms=onset_sigma,
            offset_sigma_ms=offset_sigma,
            informative=informative,
            quiet_window_available=quiet_window_available,
            on_t_overlap_risk=on_t_overlap_risk,
            ta_overlap_risk=ta_overlap_risk,
        )
        feature.p_onset_confidence = onset_conf
        feature.p_offset_confidence = offset_conf
    return beat_features


def _p_should_reselect_candidate(
    *,
    old_score: Optional[float],
    new_score: Optional[float],
    old_support_score: float,
    new_support_score: float,
    old_template_corr: float,
    new_template_corr: float,
    old_pr_score: float,
    new_pr_score: float,
    old_pp_score: float,
    new_pp_score: float,
    review_score: float = _P_RESELECT_REVIEW_SCORE,
) -> bool:
    if old_score is None or new_score is None:
        return False
    old = float(old_score)
    new = float(new_score)
    if old >= float(review_score):
        return False
    if new < _P_RESELECT_MIN_NEW_SCORE:
        return False
    if new - old < _P_RESELECT_MIN_SCORE_DELTA:
        return False
    if old_support_score >= 1.0 and new_support_score < old_support_score:
        return False
    support_better = new_support_score > old_support_score
    template_better = new_template_corr >= old_template_corr + _P_RESELECT_TEMPLATE_GAIN
    timing_better = max(new_pr_score - old_pr_score, new_pp_score - old_pp_score) >= _P_RESELECT_TIMING_GAIN
    return bool(support_better or (template_better and timing_better))


def _select_p_peak_with_fused_anchor(
    sig_sm: np.ndarray,
    *,
    baseline: float,
    p_peak: Optional[int],
    fused_p_peak: Optional[int],
    search_lo: int,
    search_hi: int,
    qrs_ref: Optional[int],
    fs: int,
    min_amp: float,
) -> Optional[int]:
    if p_peak is None or fused_p_peak is None:
        return p_peak
    if p_peak == fused_p_peak:
        return p_peak
    if not (search_lo <= fused_p_peak < search_hi):
        return p_peak
    if not (0 <= p_peak < len(sig_sm) and 0 <= fused_p_peak < len(sig_sm)):
        return p_peak

    peak_amp = abs(float(sig_sm[p_peak]) - baseline)
    fused_amp = abs(float(sig_sm[fused_p_peak]) - baseline)
    if fused_amp < min_amp:
        return p_peak
    if np.isclose(peak_amp, fused_amp, rtol=1e-6, atol=1e-12):
        return fused_p_peak
    if qrs_ref is None or qrs_ref <= 0:
        return p_peak

    peak_gap_ms = (qrs_ref - p_peak) * 1000.0 / fs
    fused_gap_ms = (qrs_ref - fused_p_peak) * 1000.0 / fs
    fused_gap_plausible = (
        _P_PEAK_QRS_GAP_MIN_MS <= fused_gap_ms <= _P_PEAK_QRS_GAP_MAX_MS
    )
    peak_too_close_to_qrs = peak_gap_ms < _P_SHORT_P_PEAK_QRS_GAP_MS
    if (fused_gap_plausible
            and peak_too_close_to_qrs
            and fused_amp >= peak_amp * _P_FUSED_PEAK_MIN_AMP_RATIO):
        return fused_p_peak
    if (fused_gap_plausible
            and fused_gap_ms > peak_gap_ms + 20.0
            and fused_amp >= peak_amp * _P_FUSED_PEAK_CLOSE_AMP_RATIO):
        return fused_p_peak
    return p_peak


def _find_wave_bounds(
    sig: np.ndarray,
    peak: Optional[int],
    baseline: float,
    thresh_frac: float = 0.05,
) -> Tuple[Optional[int], Optional[int]]:
    if peak is None:
        return None, None
    amp = sig[peak] - baseline
    thr = max(abs(amp) * thresh_frac, 1e-4)
    onset = peak
    while onset > 1 and abs(sig[onset] - baseline) > thr:
        onset -= 1
    offset = peak
    while offset < len(sig) - 2 and abs(sig[offset] - baseline) > thr:
        offset += 1
    return onset, offset


def _baseline_contract(sig: np.ndarray, r_index: int, fs: int) -> Tuple[float, str, float]:
    bl_lo = max(0, r_index - int(0.22 * fs))
    bl_hi = max(bl_lo + 1, r_index - int(0.08 * fs))
    if bl_hi > bl_lo:
        return float(np.median(sig[bl_lo:bl_hi])), "pr_segment", 0.8
    return 0.0, "fallback_zero", 0.1


def _t_end_geometric(
    sig: np.ndarray,
    t_peak_local: int,
    search_end_local: int,
    baseline: float,
    fs: int,
) -> Tuple[Optional[int], float, str]:
    """
    DXL chord-distance T-end detector (replaces Laguna tangent).

    Draws a chord from T-peak to search_end; the maximum vertical distance
    (chord − signal, in the T polarity direction) locates T-end.  This mirrors
    the chord-distance method used for T-onset and is mathematically equivalent
    to what the Philips DXL algorithm describes for finding the end of the T-wave.

    Why the Laguna tangent underestimates: it intersects the tangent with the
    baseline at the steepest descent, typically 20–30 ms before the signal
    actually returns to baseline.  The chord method has no such bias.
    """
    if t_peak_local is None:
        return None, 0.0, "missing_peak"

    arm_start = t_peak_local
    arm_end   = min(len(sig) - 1, search_end_local)
    min_arm   = max(4, int(0.040 * fs))

    if arm_end - arm_start < min_arm:
        _, t_off = _find_wave_bounds(sig, t_peak_local, baseline, thresh_frac=0.06)
        return t_off, 0.10, "threshold_fallback"

    arm = (sig[arm_start : arm_end + 1] - baseline).astype(float)
    polarity      = 1.0 if arm[0] >= 0.0 else -1.0
    arm_amplitude = float(np.max(np.abs(arm))) + 1e-6
    # T-peak amplitude (arm[0] is signal at T-peak – baseline).  This is distinct
    # from arm_amplitude which scans the full arm and may be inflated by downstream
    # P-waves or baseline drift well past T-end.
    t_peak_amp = abs(float(arm[0]))

    if arm_amplitude < 0.015:
        _, t_off = _find_wave_bounds(sig, t_peak_local, baseline, thresh_frac=0.06)
        return t_off, 0.10, "threshold_fallback"

    # Correct for baseline wander in the search window:
    # arm[-1] should be 0 at the isoelectric line; any residual is drift.
    # Subtract the linear ramp so the arm ends exactly at 0.
    # (DXL: "baseline wander slope is subtracted from max/min slopes used".)
    bl_drift         = float(arm[-1])
    drift_correction = np.linspace(0.0, bl_drift, len(arm))
    arm_dt           = arm - drift_correction   # detrended; arm_dt[-1] == 0

    # 20 ms smoothed arm — used both for slope constraints and the
    # low-amplitude threshold path below.
    _slp_win = max(3, int(0.020 * fs))
    _arm_sm  = np.convolve(arm_dt, np.ones(_slp_win) / _slp_win, mode="same")

    # ── Small-T-wave flag ──────────────────────────────────────────────────
    # For T-peaks < 120 µV the bc-clamp below (raw 10 % threshold) fires on
    # noise spikes (noise ≈ 15 µV raw; 10 % × 80 µV = 8 µV < noise floor),
    # causing premature T-end truncation.  Gate on t_peak_amp (T-peak only),
    # NOT arm_amplitude (max over full arm) which is inflated by downstream
    # P-waves and baseline drift even when the T-wave itself is tiny.
    # When this flag is set the bc-clamp is skipped entirely: the chord argmax
    # already approximates the inflection-point (tangent-method) T-end.
    _use_smooth_bc = (t_peak_amp < 0.12)
    # ──────────────────────────────────────────────────────────────────────

    # Chord from detrended T-peak to 0 (true isoelectric baseline)
    chord = np.linspace(float(arm_dt[0]), 0.0, len(arm_dt))

    # Deviation in the expected direction:
    #   upright T  (polarity=+1): chord > arm_dt after T-end → deviation = chord − arm_dt > 0
    #   inverted T (polarity=-1): arm_dt > chord after T-end → deviation = arm_dt − chord > 0
    deviation = polarity * (chord - arm_dt)

    # Smooth over 20 ms to suppress noise peaks
    win    = _slp_win
    kernel = np.ones(win) / win
    deviation_smooth = np.convolve(deviation, kernel, mode="same")

    # Skip first 30 ms (T-peak shoulder where deviation rises monotonically from 0)
    excl_start = max(1, int(0.030 * fs))
    if excl_start >= len(deviation_smooth):
        excl_start = max(1, len(deviation_smooth) // 4)

    search_dev = deviation_smooth[excl_start:]
    if len(search_dev) < 2:
        _, t_off = _find_wave_bounds(sig, t_peak_local, baseline, thresh_frac=0.06)
        return t_off, 0.10, "chord_fallback"

    t_end_local = excl_start + int(np.argmax(search_dev))

    # ── DXL slope constraints ──────────────────────────────────────────────
    # Slopes are computed on arm_dt (linear-detrended), so baseline drift is
    # automatically compensated — no separate correction needed here.
    # (DXL: "baseline wander slope is subtracted from max/min slopes used.")
    _slopes   = np.gradient(_arm_sm)

    # Peak descent rate after the 30-ms T-peak shoulder.
    _desc_after = -polarity * _slopes[excl_start:]
    _peak_desc  = float(np.max(_desc_after)) if len(_desc_after) > 0 else 0.0

    # 坡度下限 (slope lower bound): T-wave too flat for reliable chord detection
    # → chord deviation may anchor on a noise peak.  Fall back to threshold.
    if _peak_desc < arm_amplitude * 0.003:
        _, t_off = _find_wave_bounds(sig, t_peak_local, baseline, thresh_frac=0.06)
        return t_off, 0.10, "threshold_slope_flat"

    # 坡度上限 (slope upper bound): if the chord-selected T-end is still on the
    # steep descending limb (slope > 40% of peak descent), the inflection fell
    # too early.  Advance T-end forward up to 30 ms until slope has levelled off.
    _steep_limit = _peak_desc * 0.40
    _max_adv     = max(1, int(0.030 * fs))
    if -polarity * _slopes[t_end_local] > _steep_limit:
        _adv_ceil = min(len(_slopes), t_end_local + _max_adv + 1)
        for _adv in range(t_end_local + 1, _adv_ceil):
            if -polarity * _slopes[_adv] <= _steep_limit:
                t_end_local = _adv
                break
    # ──────────────────────────────────────────────────────────────────────

    max_dev = float(deviation_smooth[t_end_local])
    # Treat sub-epsilon deviations as zero — they are FP noise from detrending arithmetic
    if max_dev < np.finfo(float).eps * arm_amplitude * 100:
        max_dev = 0.0
    confidence = float(np.clip(max_dev / arm_amplitude, 0.0, 1.0))

    # Cross-check: if the chord T-end is far past the signal's own baseline
    # crossing, the chord was likely dragged by a subsequent waveform (P-wave or
    # U-wave). Clamp to the baseline-crossing estimate in that case.
    #
    # For small T-waves (t_peak_amp < 120 µV) the raw 10 % threshold fires on
    # noise spikes (noise ≈ 15 µV; 10 % × 80 µV = 8 µV < noise floor).
    # Skip the bc-clamp entirely for small T-waves: the chord argmax already
    # approximates the inflection-point (tangent-method) T-end, and the
    # clamp would only corrupt it with noise-driven premature truncation.
    if not _use_smooth_bc:
        _, t_off_bc = _find_wave_bounds(sig, arm_start, baseline, thresh_frac=0.10)
        t_end_candidate = arm_start + t_end_local
        if (t_off_bc is not None
                and t_end_candidate > t_off_bc + int(0.10 * fs)):
            return t_off_bc, min(confidence, 0.45), "dxl_chord_bc_clamped"

    return arm_start + t_end_local, confidence, "dxl_chord"


def _t_wave_complex_extend(
    sig: np.ndarray,
    sig_sm: np.ndarray,
    t_peak_local: int,
    t_off: int,
    search_end_local: int,
    baseline: float,
    fs: int,
) -> Tuple[int, str]:
    """
    Extend T-end for biphasic and notched T waves.

    After the chord method delivers a preliminary T-end, scan forward for a
    secondary T-wave feature:
      - Opposite polarity (biphasic): positive-then-negative or vice-versa.
      - Same polarity (notched): two humps of the same sign with a dip between.

    A secondary peak on the smoothed signal that is ≥ 30 % of the primary
    T-peak amplitude triggers re-detection from that secondary peak using the
    standard chord method.  Smaller deflections (< 30 %) are treated as U-waves
    and left alone.  T-end is only ever extended, never shortened.

    Returns (t_off_final, label) where label is "" (unchanged), "biphasic", or
    "notched".
    """
    SECONDARY_THRESH  = 0.50   # ≥ 50 % of primary amplitude
    SCAN_PAST_MS      = 0.060  # look up to 60 ms past preliminary T-end
    MIN_SKIP_MS       = 0.040  # skip the 40 ms immediately after T-peak
    MAX_EXTENSION_MS  = 0.080  # cap extension: notched humps are typically 20–60 ms apart

    primary_amp = float(sig[t_peak_local]) - baseline
    # Flat T-waves (< 50 µV) are dominated by noise; any secondary feature
    # detected there would be unreliable.
    if abs(primary_amp) < 0.050:
        return t_off, ""

    primary_polarity = 1.0 if primary_amp >= 0.0 else -1.0

    scan_lo = min(len(sig) - 1, t_peak_local + int(MIN_SKIP_MS * fs))
    scan_hi = min(len(sig) - 1, search_end_local,
                  t_off + int(SCAN_PAST_MS * fs))
    if scan_hi <= scan_lo:
        return t_off, ""

    seg = sig_sm[scan_lo : scan_hi + 1] - baseline
    abs_seg = np.abs(seg)
    if abs_seg.max() < abs(primary_amp) * SECONDARY_THRESH:
        return t_off, ""   # nothing significant beyond preliminary T-end

    sec_local    = int(np.argmax(abs_seg))
    sec_amp      = float(seg[sec_local])
    sec_peak     = scan_lo + sec_local
    sec_polarity = 1.0 if sec_amp >= 0.0 else -1.0

    if abs(sec_amp) < abs(primary_amp) * SECONDARY_THRESH:
        return t_off, ""

    if sec_polarity != primary_polarity:
        compact_biphasic = sec_peak - t_peak_local <= int(0.180 * fs)
        near_prelim_end = sec_peak <= t_off + int(0.030 * fs)
        if not (compact_biphasic and near_prelim_end):
            return t_off, ""
        t_off_sec, _, _ = _t_end_geometric(sig, sec_peak, search_end_local, baseline, fs)
        if t_off_sec is None or t_off_sec <= t_off:
            return t_off, ""
        if t_off_sec > t_off + int(0.140 * fs):
            return t_off, ""
        return t_off_sec, "biphasic"

    # Only late same-polarity secondary peaks are U/f-wave contamination, not
    # true notched T humps.
    if sec_peak - t_peak_local > int(0.160 * fs):
        return t_off, ""

    # For a genuine notched T the second hump must lie AFTER the preliminary
    # T-end (which the chord method placed at the notch bottom).  If sec_peak
    # is before t_off, it is merely a point on the descending arm of the
    # primary T, not a secondary hump.
    if sec_peak <= t_off:
        return t_off, ""

    # Re-run chord from secondary peak; only accept if it extends T-end
    t_off_sec, _, _ = _t_end_geometric(sig, sec_peak, search_end_local, baseline, fs)
    if t_off_sec is None or t_off_sec <= t_off:
        return t_off, ""

    # Extension cap: genuine notched T second humps produce modest extensions.
    # Very large extensions (> 80 ms) indicate the chord latched onto an
    # unrelated feature (P-wave, artefact, or far-out U-wave tail).
    if t_off_sec > t_off + int(MAX_EXTENSION_MS * fs):
        return t_off, ""

    return t_off_sec, "notched"


def _find_t_peak_candidate(
    sig_sm: np.ndarray,
    lo: int,
    hi: int,
    baseline: float,
    fs: int,
) -> Optional[int]:
    """Choose a T peak using local wave area, not just the tallest sample."""
    lo = max(0, lo)
    hi = min(len(sig_sm), hi)
    if hi <= lo:
        return None

    seg = np.abs(sig_sm[lo:hi] - baseline)
    if len(seg) == 0:
        return None
    max_amp = float(np.max(seg))
    if max_amp <= 0.0:
        return lo + int(np.argmax(seg))

    min_amp = max(0.010, 0.30 * max_amp)
    candidates: List[int] = []
    for idx in range(1, len(seg) - 1):
        if seg[idx] >= min_amp and seg[idx] >= seg[idx - 1] and seg[idx] >= seg[idx + 1]:
            candidates.append(idx)
    argmax_idx = int(np.argmax(seg))
    if argmax_idx not in candidates:
        candidates.append(argmax_idx)

    area_half = max(4, int(0.045 * fs))
    width_half = max(4, int(0.070 * fs))
    best_idx = argmax_idx
    best_score = -1.0
    for idx in candidates:
        area_lo = max(0, idx - area_half)
        area_hi = min(len(seg), idx + area_half + 1)
        local_area = float(trapezoid(seg[area_lo:area_hi]))
        width_lo = max(0, idx - width_half)
        width_hi = min(len(seg), idx + width_half + 1)
        local_width = float(np.sum(seg[width_lo:width_hi] >= max(0.006, 0.40 * float(seg[idx]))))
        score = local_area * (1.0 + min(local_width, float(width_half)) / max(float(width_half), 1.0))
        if score > best_score:
            best_score = score
            best_idx = idx

    return lo + int(best_idx)


def _clamp_t_end_to_tu_nadir(
    sig: np.ndarray,
    sig_sm: np.ndarray,
    t_peak: Optional[int],
    t_off: Optional[int],
    baseline: float,
    local_t_cap: int,
    fs: int,
) -> Tuple[Optional[int], bool]:
    """If a late same-polarity U-like wave follows T, use the T-U nadir."""
    if t_peak is None or t_off is None:
        return t_off, False

    primary_amp = float(sig_sm[t_peak]) - baseline
    if abs(primary_amp) < 0.030:
        return t_off, False
    polarity = 1.0 if primary_amp >= 0.0 else -1.0
    dev = (sig_sm - baseline) * polarity

    search_lo = min(len(sig_sm) - 1, t_peak + int(0.070 * fs))
    search_hi = min(len(sig_sm) - 1, local_t_cap, t_peak + int(0.320 * fs))
    if search_hi <= search_lo + int(0.040 * fs):
        return t_off, False

    seg = dev[search_lo : search_hi + 1]
    min_u_amp = max(0.040, 0.45 * abs(primary_amp))
    if float(np.max(seg)) < min_u_amp:
        return t_off, False

    u_peak = search_lo + int(np.argmax(seg))
    if u_peak <= t_peak + int(0.090 * fs):
        return t_off, False

    valley_lo = min(len(sig_sm) - 1, t_peak + int(0.045 * fs))
    valley_hi = max(valley_lo + 1, u_peak - int(0.020 * fs))
    if valley_hi <= valley_lo + 2:
        return t_off, False
    valley_seg = dev[valley_lo : valley_hi + 1]
    nadir = valley_lo + int(np.argmin(valley_seg))
    nadir_value = float(dev[nadir])

    near_baseline = nadir_value <= max(0.020, 0.30 * abs(primary_amp))
    separated = (u_peak - nadir) >= int(0.035 * fs)
    meaningful_u = float(dev[u_peak] - nadir_value) >= max(0.030, 0.35 * abs(primary_amp))
    if not (near_baseline and separated and meaningful_u):
        return t_off, False
    if t_off <= nadir + int(0.010 * fs):
        return t_off, True
    return int(nadir), True


def _t_onset_tangent(
    sig: np.ndarray,
    t_peak_local: Optional[int],
    search_start_local: int,
    baseline: float,
    fs: int,
) -> Optional[int]:
    """
    Tangent-intersection T-onset detector (ascending-limb mirror of Laguna T-end).

    Finds the steepest ascending slope on the T-wave upstroke, draws a tangent,
    and returns its intersection with the isoelectric baseline.

    The result is always constrained to be >= search_start_local (typically
    QRS offset) so that T onset cannot overlap with the QRS complex.

    Falls back to a higher-threshold amplitude walk (15 %) when the upstroke
    arm is too short or the slope is flat.
    """
    if t_peak_local is None:
        return None

    arm_start = max(0, search_start_local)
    arm_end   = t_peak_local
    min_arm   = max(4, int(0.020 * fs))

    def _fallback() -> Optional[int]:
        # 10 % threshold: lower than before (was 15 %) so T onset is found
        # closer to the wave foot rather than the visible upstroke mid-point.
        t_on, _ = _find_wave_bounds(sig, t_peak_local, baseline, thresh_frac=0.10)
        return max(t_on, arm_start) if t_on is not None else arm_start

    if arm_end - arm_start < min_arm:
        return _fallback()

    arm           = (sig[arm_start : arm_end + 1] - baseline).astype(float)
    polarity      = 1.0 if arm[-1] >= 0.0 else -1.0
    arm_amplitude = float(np.max(np.abs(arm))) + 1e-6

    # Smooth first derivative over 20 ms
    win    = max(3, int(0.020 * fs))
    kernel = np.ones(win) / win
    d1     = np.convolve(np.gradient(arm), kernel, mode="same")

    # Search only the ascending portion — exclude the last 30 ms near the T peak
    # where the slope is again near-zero (peak shoulder).
    excl_end   = max(1, len(arm) - int(0.030 * fs))
    search_seg = d1[:excl_end]

    if len(search_seg) < 2:
        return _fallback()

    # Flat-signal fallback
    max_slope = float(np.max(search_seg * polarity))
    if max_slope < 0.005 * arm_amplitude:
        return _fallback()

    # Steepest ascending point
    steep_local = int(np.argmax(search_seg * polarity))
    slope       = float(d1[steep_local])
    arm_at_sp   = float(arm[steep_local])

    if abs(slope) < 1e-9:
        return _fallback()

    # Tangent–baseline intersection:
    # arm(x) = slope * (x − steep_local) + arm_at_sp = 0
    # → x_cross = steep_local − arm_at_sp / slope
    x_cross = steep_local - arm_at_sp / slope

    # If the tangent intersection is inside the ascending arm, use it directly.
    # If x_cross < 0, the ascending limb starts already above baseline (e.g.,
    # elevated ST, inverted T) — the tangent extrapolates past the arm start.
    # Use the inflection point (steep_local) which corresponds to the visible
    # "foot" of the T-wave upstroke and is within the arm bounds.
    if x_cross >= 0:
        t_on_local = int(np.clip(round(x_cross), 0, steep_local))
    else:
        t_on_local = steep_local

    return arm_start + t_on_local


def _t_onset_chord(
    sig: np.ndarray,
    t_peak_local: Optional[int],
    search_start_local: int,
    baseline: float,
    fs: int,
) -> Optional[int]:
    """
    DXL chord-distance T-onset detector.

    Draws a chord from arm_start (QRS-offset / isoelectric region) to T-peak.
    The point with maximum deviation of the signal BELOW the chord (upright T)
    or ABOVE the chord (inverted T) is T-onset.

    Mathematical property: before T-onset the signal stays at baseline while the
    chord linearly rises toward the T-peak, creating monotonically increasing
    deviation. At T-onset the signal starts rising and catches the chord,
    decreasing the deviation. The argmax is therefore exactly at T-onset.

    No tangent extrapolation → eliminates the x_cross < 0 failure mode that
    afflicts the Laguna tangent when the ST segment is elevated.
    """
    if t_peak_local is None:
        return None

    arm_start = max(0, search_start_local)
    arm_end   = t_peak_local
    min_arm   = max(4, int(0.020 * fs))

    if arm_end - arm_start < min_arm:
        t_on, _ = _find_wave_bounds(sig, t_peak_local, baseline, thresh_frac=0.10)
        return max(t_on, arm_start) if t_on is not None else arm_start

    arm           = (sig[arm_start : arm_end + 1] - baseline).astype(float)
    polarity      = 1.0 if arm[-1] >= 0.0 else -1.0
    arm_amplitude = float(np.max(np.abs(arm))) + 1e-6

    # Very flat T-wave: threshold walk is more reliable
    if arm_amplitude < 0.025:
        t_on, _ = _find_wave_bounds(sig, t_peak_local, baseline, thresh_frac=0.10)
        return max(t_on, arm_start) if t_on is not None else arm_start

    # Chord: linear from arm_start value to T-peak value
    chord = np.linspace(arm[0], arm[-1], len(arm))

    # Deviation in expected direction:
    #   upright T  (polarity=+1): chord > arm before onset → deviation = chord - arm > 0
    #   inverted T (polarity=-1): arm  > chord before onset → deviation = arm - chord > 0
    deviation = polarity * (chord - arm)

    # Smooth over 20 ms to suppress noise peaks
    win    = max(3, int(0.020 * fs))
    kernel = np.ones(win) / win
    deviation_smooth = np.convolve(deviation, kernel, mode="same")

    # Exclude the last 30 ms near T-peak (shoulder where slope reversal distorts deviation)
    excl_end   = max(1, len(arm) - int(0.030 * fs))
    search_dev = deviation_smooth[:excl_end]
    if len(search_dev) < 2:
        return arm_start

    t_on_local = int(np.argmax(search_dev))
    return arm_start + t_on_local


def _candidate_triplets(
    sig: np.ndarray,
    lo: int,
    hi: int,
    baseline: float,
    min_amp: float = 0.008,
    max_candidates: int = 3,
) -> List[Tuple[int, float, float]]:
    """Return (peak_index, amplitude, local_area) for up to max_candidates turning points."""
    lo = max(1, lo)
    hi = min(len(sig) - 1, hi)
    if hi - lo < 4:
        return []
    x  = sig[lo:hi].astype(float)
    d1 = np.diff(x)
    turn = np.where(np.sign(d1[:-1]) != np.sign(d1[1:]))[0] + 1
    out: List[Tuple[int, float, float]] = []
    for local_idx in turn:
        peak = lo + int(local_idx)
        amp  = float(sig[peak] - baseline)
        if abs(amp) < min_amp:
            continue
        area_lo = max(lo, peak - 4)
        area_hi = min(hi, peak + 5)
        area    = float(trapezoid(np.abs(sig[area_lo:area_hi] - baseline)))
        out.append((peak, amp, area))
    out.sort(key=lambda item: (abs(item[1]), item[2]), reverse=True)
    return out[:max_candidates]


def _t_candidate_triplets(
    sig: np.ndarray,
    lo: int,
    hi: int,
    baseline: float,
    fs: int,
    min_amp: float = 0.015,
    max_candidates: int = 4,
) -> List[Tuple[int, float, float]]:
    """Return T candidates with an area window wide enough for broad T waves."""
    lo = max(1, lo)
    hi = min(len(sig) - 1, hi)
    if hi - lo < 4:
        return []

    dev = np.abs(sig[lo:hi].astype(float) - baseline)
    if len(dev) == 0:
        return []

    d1 = np.diff(dev)
    turn = np.where(np.sign(d1[:-1]) != np.sign(d1[1:]))[0] + 1
    argmax_idx = int(np.argmax(dev))
    candidate_idx = set(int(idx) for idx in turn if dev[int(idx)] >= min_amp)
    if dev[argmax_idx] >= min_amp:
        candidate_idx.add(argmax_idx)

    area_half = max(4, int(0.045 * fs))
    width_half = max(4, int(0.070 * fs))
    out: List[Tuple[int, float, float]] = []
    for local_idx in candidate_idx:
        peak = lo + int(local_idx)
        amp = float(sig[peak] - baseline)
        if abs(amp) < min_amp:
            continue
        area_lo = max(lo, peak - area_half)
        area_hi = min(hi, peak + area_half + 1)
        area = float(trapezoid(np.abs(sig[area_lo:area_hi] - baseline)))
        width_lo = max(lo, peak - width_half)
        width_hi = min(hi, peak + width_half + 1)
        width = float(np.sum(np.abs(sig[width_lo:width_hi] - baseline) >= max(0.006, 0.40 * abs(amp))))
        weighted_area = area * (1.0 + min(width, float(width_half)) / max(float(width_half), 1.0))
        out.append((peak, amp, weighted_area))

    out.sort(key=lambda item: (abs(item[1]) * item[2], item[2]), reverse=True)
    return out[:max_candidates]


def _fuse_peak_anchor(
    candidates_by_lead: Dict[str, List[Tuple[int, float, float]]],
    quality: Dict[str, object],
    reliable_attr: str,
    prior_peak: Optional[int],
    cluster_radius: int,
    normalize_per_lead: bool = False,
    qrs_ref: Optional[int] = None,
    fs: Optional[int] = None,
    physiologic_peak_gap_ms: Optional[Tuple[float, float]] = None,
    short_peak_gap_ms: Optional[float] = None,
) -> Tuple[Optional[int], int]:
    """
    Fuse per-lead peak candidates into a single beat-level anchor.

    By default, weights each candidate by |amplitude| × area × prior proximity.
    For low-amplitude P waves, callers can enable per-lead normalization so a
    small multi-lead cluster is not dominated by a high-amplitude outlier lead.

    Returns ``(anchor_index, lead_support)`` where ``lead_support`` is the number
    of distinct leads backing the winning cluster.  Callers need it because the
    anchor's trustworthiness is not visible from its position: a single lead can
    produce a confident-looking anchor that no other lead corroborates.
    """
    weighted: List[Tuple[int, float, str]] = []
    for lead, candidates in candidates_by_lead.items():
        if not getattr(quality.get(lead), reliable_attr, False):
            continue
        raw_weights: List[Tuple[int, float]] = []
        for peak, amp, area in candidates:
            raw_weights.append((int(peak), abs(float(amp)) * max(float(area), 1e-6)))
        lead_scale = max((weight for _peak, weight in raw_weights), default=0.0)
        if lead_scale <= 0.0:
            continue
        for peak, raw_weight in raw_weights:
            prior_weight = 1.0
            if prior_peak is not None:
                dist = abs(int(peak) - int(prior_peak))
                prior_weight = max(0.25, 1.0 - dist / max(cluster_radius * 3, 1))
            candidate_weight = raw_weight / lead_scale if normalize_per_lead else raw_weight
            weighted.append((int(peak), candidate_weight * prior_weight, lead))

    if not weighted:
        return None, 0

    weighted.sort(key=lambda item: item[0])
    best_cluster: List[Tuple[int, float, str]] = []
    best_score = -1.0
    for center, _weight, _lead in weighted:
        cluster = [(p, w, lead) for p, w, lead in weighted if abs(p - center) <= cluster_radius]
        if normalize_per_lead:
            by_lead: Dict[str, Tuple[int, float]] = {}
            for p, w, lead in cluster:
                if lead not in by_lead or w > by_lead[lead][1]:
                    by_lead[lead] = (p, w)
            score_items = [(p, w, lead) for lead, (p, w) in by_lead.items()]
        else:
            score_items = cluster

        score = float(sum(w for _p, w, _lead_name in score_items))
        lead_support = len({lead for _p, _w, lead in score_items})
        if normalize_per_lead and lead_support > 1:
            score *= float(lead_support) ** _P_FUSED_ANCHOR_SUPPORT_POWER
        elif normalize_per_lead:
            score *= _P_FUSED_ANCHOR_SINGLE_LEAD_PENALTY

        if (qrs_ref is not None
                and fs is not None
                and fs > 0
                and score_items):
            peaks_for_gap = np.asarray([p for p, _w, _lead_name in score_items], dtype=float)
            weights_for_gap = np.asarray([w for _p, w, _lead_name in score_items], dtype=float)
            center_for_gap = float(np.average(peaks_for_gap, weights=weights_for_gap))
            gap_ms = (float(qrs_ref) - center_for_gap) * 1000.0 / float(fs)
            if physiologic_peak_gap_ms is not None:
                lo_ms, hi_ms = physiologic_peak_gap_ms
                if lo_ms <= gap_ms <= hi_ms:
                    score *= _P_FUSED_ANCHOR_PHYSIOLOGIC_BONUS
            if short_peak_gap_ms is not None and gap_ms < short_peak_gap_ms:
                score *= _P_FUSED_ANCHOR_SHORT_GAP_PENALTY

        if score > best_score:
            best_cluster = score_items
            best_score   = score

    if not best_cluster:
        return None, 0

    peaks   = np.asarray([p for p, _w, _lead in best_cluster], dtype=float)
    weights = np.asarray([w for _p, w, _lead in best_cluster], dtype=float)
    lead_support = len({lead for _p, _w, lead in best_cluster})
    return int(round(np.average(peaks, weights=weights))), lead_support


def _median_prior_peak(prior: Dict[str, Dict], field: str) -> Optional[int]:
    vals = [int(ld[field]) for ld in prior.values() if ld.get(field) is not None]
    return int(np.median(vals)) if vals else None


def _should_use_full_t_window_measurement(
    narrow: TWaveMeasurement,
    wide: Optional[TWaveMeasurement],
    *,
    qrs_on: Optional[int],
    qrs_off: Optional[int],
    prior_qrs_ms: Optional[float],
    fs: int,
) -> bool:
    if wide is None or wide.peak is None:
        return False
    if narrow.peak is None:
        return True
    if fs <= 0:
        return False
    if qrs_on is not None and qrs_off is not None and int(qrs_off) > int(qrs_on):
        qrs_ms = (int(qrs_off) - int(qrs_on)) * 1000.0 / fs
        if qrs_ms >= _T_FULL_WINDOW_RESCUE_MAX_QRS_MS:
            return False
    if prior_qrs_ms is not None and prior_qrs_ms >= _T_FULL_WINDOW_RESCUE_MAX_QRS_MS:
        return False
    min_gap = int(round(_T_FULL_WINDOW_RESCUE_MIN_GAP_MS * fs / 1000.0))
    if int(wide.peak) <= int(narrow.peak) + min_gap:
        return False
    if qrs_off is not None:
        narrow_dt_ms = (int(narrow.peak) - int(qrs_off)) * 1000.0 / fs
        wide_dt_ms = (int(wide.peak) - int(qrs_off)) * 1000.0 / fs
        if narrow_dt_ms > _T_FULL_WINDOW_RESCUE_EARLY_MAX_MS:
            return False
        if wide_dt_ms < _T_FULL_WINDOW_RESCUE_LATE_MIN_MS:
            return False
    if not bool(getattr(wide, "st_t_confusion", False)):
        return False
    narrow_conf = float(getattr(narrow, "confidence", 0.0) or 0.0)
    wide_conf = float(getattr(wide, "confidence", 0.0) or 0.0)
    return wide_conf >= max(0.15, _T_FULL_WINDOW_RESCUE_MIN_CONF_RATIO * narrow_conf)


def _p_offset_tangent(
    sig: np.ndarray,
    p_peak_local: Optional[int],
    search_end_local: int,
    baseline: float,
    fs: int,
) -> Optional[int]:
    """
    Tangent-intersection P-wave offset detector (descending limb).

    Mirror of _t_end_geometric for P waves: finds the steepest descending
    slope after the P peak, draws a tangent, and returns its intersection
    with the isoelectric baseline.

    Uses a shorter post-tangent extension (20 ms vs 60 ms for T waves) to
    avoid drifting into the PR segment.
    Falls back to an 8 % threshold walk when the arm is too short or flat.
    """
    if p_peak_local is None:
        return None

    arm_start = p_peak_local
    arm_end   = min(len(sig) - 1, search_end_local)
    min_arm   = max(3, int(0.020 * fs))

    def _fallback() -> Optional[int]:
        _, p_off = _find_wave_bounds(sig, p_peak_local, baseline, thresh_frac=0.08)
        return p_off

    if arm_end - arm_start < min_arm:
        return _fallback()

    arm      = (sig[arm_start : arm_end + 1] - baseline).astype(float)
    polarity = 1.0 if arm[0] >= 0.0 else -1.0
    arm_amplitude = float(np.max(np.abs(arm))) + 1e-6

    win    = max(3, int(0.015 * fs))
    kernel = np.ones(win) / win
    d1     = np.convolve(np.gradient(arm), kernel, mode="same")

    excl      = max(1, int(0.020 * fs))
    level_80  = 0.80 * arm[0]
    start_srch = excl
    for i in range(excl, len(arm)):
        if (polarity > 0 and arm[i] < level_80) or \
           (polarity < 0 and arm[i] > level_80):
            start_srch = i
            break

    if start_srch >= len(d1) - 1:
        start_srch = excl

    steep_local = int(np.argmin(d1[start_srch:] * polarity)) + start_srch
    slope       = float(d1[steep_local])
    arm_at_sp   = float(arm[steep_local])

    if abs(slope) < 0.005 * arm_amplitude:
        return _fallback()

    x_cross     = steep_local - arm_at_sp / slope
    p_end_local = int(np.clip(round(x_cross), steep_local, len(arm) - 1))

    # Short post-tangent extension (max 20 ms) — P wave is brief
    arm_len  = len(arm)
    base_thr = 0.04 * arm_amplitude
    ext_limit = p_end_local + max(1, int(0.020 * fs))
    prev_abs  = abs(arm[p_end_local]) if p_end_local < arm_len else 0.0
    p_ext     = p_end_local
    for i in range(p_end_local + 1, min(arm_len, ext_limit)):
        curr_abs = abs(arm[i])
        if curr_abs <= base_thr:
            p_ext = i
            break
        if curr_abs > prev_abs * 1.20:
            break
        prev_abs = curr_abs
        p_ext    = i

    return arm_start + p_ext


def _rescue_long_pr_p_onset(
    sig: np.ndarray,
    p_peak: Optional[int],
    p_on: Optional[int],
    qrs_on: Optional[int],
    baseline: float,
    fs: int,
) -> Tuple[Optional[int], bool]:
    """Recover a visible long-PR P foot when tangent onset lands on the upstroke."""
    if p_peak is None or p_on is None or qrs_on is None:
        return p_on, False
    if (qrs_on - p_on) * 1000.0 / fs >= 190.0:
        return p_on, False
    if (qrs_on - p_peak) * 1000.0 / fs < 90.0:
        return p_on, False

    p_amp = abs(float(sig[p_peak]) - baseline)
    if p_amp < 0.020:
        return p_on, False

    # Bound the rescue to the plausible P-wave foot; do not scan the whole PR
    # segment, where prior T tails or noise could masquerade as baseline.
    rescue_lo = max(0, p_peak - int(0.090 * fs))
    rescue_seg = sig[rescue_lo : p_peak + 1] - baseline
    baseline_tol = max(0.006, 0.08 * p_amp)
    crossings = np.where(np.abs(rescue_seg) <= baseline_tol)[0]
    if not crossings.size:
        return p_on, False

    earlier_crossings = crossings[(rescue_lo + crossings) < p_on]
    if not earlier_crossings.size:
        return p_on, False

    # Use the baseline point nearest the current tangent onset; choosing the
    # earliest crossing can jump to the far-left edge of a quiet PR segment.
    p_on_candidate = rescue_lo + int(earlier_crossings[-1])
    if p_on_candidate < p_on:
        return p_on_candidate, True
    return p_on, False


def _extend_p_candidate_edges(
    *,
    sig: Optional[np.ndarray] = None,
    p_on: int,
    p_peak: int,
    p_off: int,
    baseline: float = 0.0,
    search_lo: int,
    p_off_limit: int,
    qrs_on: Optional[int],
    fs: int,
) -> Tuple[int, int]:
    """Apply a small physiologic P-boundary expansion after tangent detection."""
    raw_on = int(p_on)
    peak = int(p_peak)
    raw_off = int(p_off)
    if not (raw_on < peak < raw_off):
        return raw_on, raw_off
    if sig is None or not (0 <= peak < len(sig)):
        return raw_on, raw_off

    peak_delta = float(sig[peak]) - float(baseline)
    peak_amp = abs(peak_delta)
    if peak_amp <= 1e-6:
        return raw_on, raw_off
    polarity = 1.0 if peak_delta >= 0.0 else -1.0
    support_thr = max(0.003, 0.025 * peak_amp)

    onset_shift = max(0, int(round(_P_ONSET_EDGE_EXTENSION_MS * fs / 1000.0)))
    offset_shift = max(0, int(round(_P_OFFSET_EDGE_EXTENSION_MS * fs / 1000.0)))
    min_on = max(int(search_lo), raw_on - onset_shift)

    max_off = int(p_off_limit)
    if qrs_on is not None:
        min_pr_segment = max(1, int(np.ceil(_PR_SEGMENT_MIN_MS * fs / 1000.0)))
        max_off = min(max_off, int(qrs_on) - min_pr_segment)
    max_off = min(max_off, raw_off + offset_shift, len(sig) - 1)

    cand_on = raw_on
    for idx in range(raw_on - 1, min_on - 1, -1):
        residual = float(sig[idx]) - float(baseline)
        if residual * polarity <= 0.0 or abs(residual) < support_thr:
            break
        if abs(residual) > 0.85 * peak_amp:
            break
        cand_on = idx

    cand_off = raw_off
    prev_abs = abs(float(sig[raw_off]) - float(baseline)) if 0 <= raw_off < len(sig) else 0.0
    for idx in range(raw_off + 1, max_off + 1):
        residual = float(sig[idx]) - float(baseline)
        residual_abs = abs(residual)
        if residual * polarity <= 0.0 or residual_abs < support_thr:
            break
        if prev_abs > 0.0 and residual_abs > max(0.85 * peak_amp, 1.35 * prev_abs):
            break
        cand_off = idx
        prev_abs = residual_abs

    if not (cand_on < peak < cand_off):
        return raw_on, raw_off

    p_dur_ms = (cand_off - cand_on) * 1000.0 / fs
    if not (_P_DURATION_MIN_MS <= p_dur_ms <= _P_DURATION_MAX_MS):
        return raw_on, raw_off

    onset_to_peak_ms = (peak - cand_on) * 1000.0 / fs
    if onset_to_peak_ms > _P_ONSET_TO_PEAK_MAX_MS:
        return raw_on, raw_off

    if qrs_on is not None:
        pr_segment_ms = (int(qrs_on) - cand_off) * 1000.0 / fs
        if not (_PR_SEGMENT_MIN_MS <= pr_segment_ms <= _PR_SEGMENT_MAX_MS):
            return raw_on, raw_off

    return int(cand_on), int(cand_off)


def _p_wave_qrs_geometry_mode(
    p_on: Optional[int],
    p_peak: Optional[int],
    qrs_on: Optional[int],
) -> Optional[str]:
    """Classify P/QRS geometry failures without discarding useful P morphology."""
    if p_peak is None:
        return None
    if p_on is not None and p_on >= p_peak:
        return "unreliable_pr"
    if qrs_on is None:
        return None
    if p_peak >= qrs_on:
        return "invalid_p"
    return None


def _refine_sustained_qrs_foot(
    sig: np.ndarray,
    left: int,
    right: int,
    r: int,
    fs: int,
    noise_lo: int,
    noise_hi: int,
) -> Tuple[int, bool]:
    """Recover a true low-slope QRS foot that derivative energy can miss."""
    if left <= 1 or right <= left:
        return left, False

    qrs_ms = (right - left) * 1000.0 / fs
    if qrs_ms > 180.0:
        return left, False

    smooth_win = max(3, int(0.008 * fs))
    smooth = np.convolve(sig.astype(float), np.ones(smooth_win) / smooth_win, mode="same")

    noise_seg = smooth[noise_lo:noise_hi] if noise_hi > noise_lo + 2 else smooth[: max(1, left)]
    if len(noise_seg) < 3:
        return left, False
    baseline = float(np.median(noise_seg))
    mad = float(np.median(np.abs(noise_seg - baseline)))
    noise = max(1.4826 * mad, 1e-6)

    amp_hi = min(len(smooth), max(right, r) + int(0.040 * fs))
    amp_seg = smooth[max(0, left):amp_hi] - baseline
    if len(amp_seg) < 3:
        return left, False
    amp_idx = int(np.argmax(np.abs(amp_seg)))
    qrs_amp = abs(float(amp_seg[amp_idx]))
    if qrs_amp < max(0.050, 8.0 * noise):
        return left, False
    polarity = 1.0 if float(amp_seg[amp_idx]) >= 0.0 else -1.0
    dev = (smooth - baseline) * polarity

    lookback = int(0.070 * fs)
    search_lo = max(1, left - lookback)
    sustain = max(4, int(0.016 * fs))
    pre = max(3, int(0.012 * fs))
    if left - search_lo < sustain:
        return left, False

    foot_thr = max(0.008, 3.0 * noise, 0.010 * qrs_amp)
    min_rise = max(0.025, 3.0 * foot_thr, 0.030 * qrs_amp)

    for cand in range(search_lo + pre, left - sustain + 1):
        before = dev[cand - pre:cand]
        after = dev[cand:cand + sustain]
        if len(before) < pre or len(after) < sustain:
            continue
        if float(np.max(before)) > foot_thr:
            continue
        if float(np.median(after)) < foot_thr:
            continue
        if float(dev[left] - dev[cand]) < min_rise:
            continue

        trend_end = min(left + 1, cand + max(sustain, int(0.040 * fs)))
        trend = np.diff(dev[cand:trend_end])
        if len(trend) == 0:
            continue
        if float(np.mean(trend >= -0.25 * foot_thr)) < 0.65:
            continue
        return int(cand), True

    return left, False


def _qrs_bounds(
    sig: np.ndarray,
    r: int,
    fs: int,
    rr_prev_ms: Optional[float] = None,
    enable_low_slope_guard: bool = False,
) -> Tuple[Optional[int], Optional[int], int, float, float]:
    """
    QRS onset/offset via squared first-derivative energy (T006: two-stage).

    Stage 1: squared-derivative energy walk for rough boundaries.
    Stage 2: second-derivative zero-crossings to refine onset/offset.

    Returns: (onset, offset, notch_count, onset_confidence, offset_confidence)
    """
    bp = bandpass_filter(sig[None, :], fs, 5.0, min(30.0, fs / 2 - 1))[0]

    # Squared first derivative, smoothed over 12 ms
    d1     = np.gradient(bp)
    d1sq   = d1 ** 2
    win    = max(3, int(0.012 * fs))
    energy = np.convolve(d1sq, np.ones(win) / win, mode="same")

    # Fiducial R can sit on a plateau or low-slope part of a wide/paced QRS.
    # Use the strongest nearby QRS energy as the threshold anchor so slow
    # pre-QRS drift does not get swallowed into the QRS bounds.
    if enable_low_slope_guard:
        peak_lo = max(0, r - int(0.060 * fs))
        peak_hi = min(len(energy), r + int(0.060 * fs) + 1)
        peak_val = float(np.max(energy[peak_lo:peak_hi])) if peak_hi > peak_lo else float(energy[r])
        left_peak = peak_lo + int(np.argmax(energy[peak_lo : r + 1])) if r + 1 > peak_lo else r
        right_peak = r + int(np.argmax(energy[r:peak_hi])) if peak_hi > r else r
        low_slope_fiducial = float(energy[r]) < 0.15 * peak_val
    else:
        peak_val = float(energy[r])
        left_peak = r
        right_peak = r
        low_slope_fiducial = False

    # Noise floor: P75 of energy in an isoelectric window before QRS.
    # Window is placed adaptively: at ~55-65% of the preceding RR so it stays
    # in the TP segment and avoids the previous beat's T wave during tachycardia.
    if rr_prev_ms is not None and rr_prev_ms > 300:
        noise_end_ms   = int(np.clip(rr_prev_ms * 0.55, 120, 300))
        noise_start_ms = noise_end_ms + 100
    else:
        noise_start_ms, noise_end_ms = 300, 200
    noise_lo    = max(0, r - int(noise_start_ms * fs / 1000))
    noise_hi    = max(noise_lo + 2, r - int(noise_end_ms * fs / 1000))
    noise_level = float(np.percentile(energy[noise_lo:noise_hi], 75)) \
                  if noise_hi > noise_lo + 2 else 1e-10

    # Threshold: 4 % of QRS peak, minimum 1.5× noise floor.
    # Previously 6 % / 2×, which caused systematic QRS underestimation (~18 ms)
    # because the energy tail of wide or gradual-slope QRS complexes was cut off.
    qrs_frac = 0.10 if low_slope_fiducial else 0.04
    thr = max(qrs_frac * peak_val, 1.5 * noise_level)

    left = left_peak
    while left > 1 and energy[left] > thr:
        left -= 1
    right = right_peak
    while right < len(sig) - 2 and energy[right] > thr:
        right += 1

    left, sustained_foot_found = _refine_sustained_qrs_foot(
        sig,
        left,
        right,
        r,
        fs,
        noise_lo,
        noise_hi,
    )

    _first_pass_ms = (right - left) * 1000.0 / fs

    notch = int(np.sum(
        np.diff(np.sign(np.diff(sig[max(0, left) : min(len(sig), right)]))) != 0
    ))

    # ── T006: Stage 2 — second-derivative zero-crossing precision ────────────
    d2 = np.gradient(np.gradient(bp))
    ms5  = max(1, int(0.005 * fs))
    ms15 = max(1, int(0.015 * fs))
    ms20 = max(1, int(0.020 * fs))

    # Noise std for confidence (pre-QRS isoelectric region)
    noise_seg = bp[noise_lo:noise_hi] if noise_hi > noise_lo else np.array([0.0])
    noise_std = float(np.std(noise_seg)) if len(noise_seg) > 1 else 1e-9

    # Onset: search [left-5ms, left+15ms] for last neg→pos zero-crossing
    on_lo = max(0, left - ms5)
    on_hi = min(len(d2) - 1, left + ms15)
    refined_left = left
    if not sustained_foot_found and on_hi > on_lo + 1:
        seg = d2[on_lo:on_hi + 1]
        for i in range(len(seg) - 2, 0, -1):
            if seg[i] <= 0.0 and seg[i + 1] > 0.0:
                cand = on_lo + i
                if abs(cand - left) <= ms20:
                    refined_left = cand
                break

    # Offset: search [right-15ms, right+20ms] for first pos→neg zero-crossing.
    off_lo = max(0, right - ms15)
    off_hi = min(len(d2) - 1, right + ms20)
    refined_right = right
    if off_hi > off_lo + 1:
        seg = d2[off_lo:off_hi + 1]
        for i in range(len(seg) - 1):
            if seg[i] >= 0.0 and seg[i + 1] < 0.0:
                cand = off_lo + i
                if abs(cand - right) <= ms20:
                    refined_right = cand
                break

    # Confidence scores based on SNR at the boundary
    onset_conf  = float(min(1.0, abs(float(bp[refined_left]))  / (noise_std + 1e-9)))
    offset_conf = float(min(1.0, abs(float(bp[refined_right])) / (noise_std + 1e-9)))

    return refined_left, refined_right, notch, onset_conf, offset_conf


def _clinical_r_peak_local(seg_bl: np.ndarray) -> int:
    """Index of the R wave inside a baseline-corrected QRS segment.

    Standard nomenclature defines R as the *first* positive deflection of the
    QRS; a later positive component is R'. Taking the global positive maximum
    instead silently relabels R' as R whenever the later component is the
    taller one. That happens routinely in the rSR' and rS-with-overshoot
    morphologies of V1/V2/aVR, and it showed up on LUDB as R markers placed
    20-40 ms late in exactly those leads.

    The rule is applied conservatively: an earlier lobe must reach
    ``_R_FIRST_POSITIVE_FRACTION`` of the tallest positive deflection before it
    displaces it. Implementing the nomenclature more literally (a much lower
    fraction) measured *worse* on LUDB across a 0.10-1.00 sweep, because small
    early bumps in a negative-dominant complex are more often noise than a
    genuine r wave. 0.70-0.80 is a flat optimum rather than a sharp one, which
    is why a value in that band is used rather than a fitted one.

    Callers apply this only to negative-dominant complexes; in a
    positive-dominant complex the tallest positive deflection *is* the R wave
    and the global maximum is already correct.
    """
    if seg_bl.size == 0:
        return 0
    positive_max = float(np.max(seg_bl))
    if positive_max <= 0.0:
        return int(np.argmax(seg_bl))
    threshold = max(_R_FIRST_POSITIVE_MIN_MV, _R_FIRST_POSITIVE_FRACTION * positive_max)
    above = seg_bl >= threshold
    if not np.any(above):
        return int(np.argmax(seg_bl))
    # Peak of the first contiguous lobe clearing the threshold. Using a run
    # rather than a local-maximum test keeps sample-level noise inside the
    # lobe from splitting it into several spurious candidates.
    start = int(np.argmax(above))
    end = start
    while end + 1 < above.size and above[end + 1]:
        end += 1
    return start + int(np.argmax(seg_bl[start : end + 1]))


def _qrs_points(
    sig: np.ndarray,
    qrs_on: Optional[int],
    r: int,
    qrs_off: Optional[int],
    baseline: float = 0.0,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
    """Return (q_amp, r_amp, s_amp, r_peak_pos) relative to baseline.

    r_amp = maximum positive deflection in the entire QRS (≥ 0).
    This matches clinical convention: even when per_lead_r sits on an S
    trough (inverted leads like aVR, V1), R is the peak positive excursion.
    q_amp = minimum deflection in the pre-R (pre-max-positive) portion.
    s_amp = minimum deflection in the post-R portion.
    All values are baseline-corrected (subtracted isoelectric level).
    r_peak_pos is the sample index of that positive peak, in the same
    (sig-relative) frame as qrs_on/qrs_off — distinct from the `r` fiducial
    argument, which is a beat-detection/alignment point and may sit on the
    S trough for negative-dominant complexes.
    """
    if qrs_on is None or qrs_off is None or qrs_off <= qrs_on:
        return None, None, None, None
    seg = sig[qrs_on : qrs_off + 1]
    if len(seg) == 0:
        return None, None, None, None
    seg_bl = seg - baseline

    # R: max positive deflection in QRS (0 if purely negative complex)
    r_amp = float(max(0.0, float(np.max(seg_bl))))

    # Position of the positive R peak (argmax of seg_bl, but clamp to ≥ 0)
    r_pos_local = int(np.argmax(seg_bl))

    # Q: minimum deflection before (and including) the R peak position
    pre_r = seg_bl[: r_pos_local + 1]
    q_amp = float(np.min(pre_r)) if len(pre_r) > 0 else 0.0

    # S: minimum deflection from R peak position onward
    post_r = seg_bl[r_pos_local:]
    s_amp = float(np.min(post_r)) if len(post_r) > 0 else 0.0

    return q_amp, r_amp, s_amp, qrs_on + r_pos_local


def _q_component_metrics(
    sig: np.ndarray,
    qrs_on: Optional[int],
    r_pos: Optional[int],
    qrs_off: Optional[int],
    baseline: float,
    fs: int,
    r_amp_mv: Optional[float],
) -> Dict[str, Optional[float | int]]:
    result: Dict[str, Optional[float | int]] = {
        "q_onset": None,
        "q_offset": None,
        "q_duration_ms": None,
        "q_area_mv_ms": None,
        "q_r_ratio": None,
        "initial_qrs_area_mv_ms": None,
        "initial_qrs_net_mv": None,
    }
    if fs <= 0 or qrs_on is None or r_pos is None or qrs_off is None:
        return result
    if qrs_on < 0 or r_pos <= qrs_on or qrs_on >= len(sig):
        return result

    qrs_hi = min(len(sig) - 1, int(qrs_off))
    initial_hi = min(qrs_hi, int(qrs_on + round(0.04 * fs)))
    if initial_hi > qrs_on:
        initial = sig[qrs_on : initial_hi + 1] - baseline
        result["initial_qrs_area_mv_ms"] = float(trapezoid(initial, dx=1000.0 / fs))
        result["initial_qrs_net_mv"] = float(np.mean(initial))

    pre_hi = min(len(sig) - 1, int(r_pos))
    seg = sig[int(qrs_on) : pre_hi + 1] - baseline
    if len(seg) < 3:
        return result
    q_local = int(np.argmin(seg))
    q_amp = float(seg[q_local])
    if r_amp_mv is not None and np.isfinite(float(r_amp_mv)) and float(r_amp_mv) > 0:
        result["q_r_ratio"] = abs(q_amp) / float(r_amp_mv)
    if q_amp >= -0.015:
        return result

    baseline_tol = 0.005
    left = q_local
    while left > 0 and float(seg[left - 1]) < -baseline_tol:
        left -= 1
    if left == 0 and abs(float(seg[left])) > baseline_tol:
        return result
    if left > 0 and abs(float(seg[left - 1])) <= baseline_tol:
        left -= 1

    right = q_local
    while right + 1 < len(seg) and float(seg[right + 1]) < -baseline_tol:
        right += 1
    if right + 1 >= len(seg):
        return result
    if abs(float(seg[right + 1])) <= baseline_tol:
        right += 1

    q_on = int(qrs_on) + left
    q_off = int(qrs_on) + right
    q_duration_ms = (q_off - q_on + 1) * 1000.0 / fs
    if q_duration_ms < 6.0 or q_duration_ms > 80.0:
        return result

    q_segment = sig[q_on : q_off + 1] - baseline
    result["q_onset"] = q_on
    result["q_offset"] = q_off
    result["q_duration_ms"] = float(q_duration_ms)
    result["q_area_mv_ms"] = float(trapezoid(np.abs(np.minimum(q_segment, 0.0)), dx=1000.0 / fs))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# T013: QRS component extraction (Q/R/R'/S/S', morphology counts)
# ─────────────────────────────────────────────────────────────────────────────

def _qrs_components(
    sig: np.ndarray,
    qrs_on: Optional[int],
    r: int,
    qrs_off: Optional[int],
    baseline: float,
    fs: int,
) -> Tuple[Optional[float], Optional[float], int, int, Dict[str, Optional[float]]]:
    """
    Extract R'/S' amplitudes and morphology peak/notch counts (T013).

    Returns:
        (r_prime_amp_mv, s_prime_amp_mv, qrs_num_peaks, qrs_notch_count, durations)
    """
    if qrs_on is None or qrs_off is None or qrs_off <= qrs_on:
        return None, None, 1, 0, {
            "r_duration_ms": None,
            "r_prime_duration_ms": None,
            "s_duration_ms": None,
            "s_prime_duration_ms": None,
        }

    seg = sig[qrs_on : qrs_off + 1] - baseline
    n = len(seg)
    if n < 3:
        return None, None, 1, 0, {
            "r_duration_ms": None,
            "r_prime_duration_ms": None,
            "s_duration_ms": None,
            "s_prime_duration_ms": None,
        }

    # Find all local maxima and minima
    d = np.diff(seg)
    # Peak = sign change from + to -; trough = sign change from - to +
    sign_d = np.sign(d)
    # Remove zero-crossings (flat regions)
    sign_d_nz = sign_d.copy()
    last_nz = 0
    for i in range(len(sign_d_nz)):
        if sign_d_nz[i] == 0:
            sign_d_nz[i] = last_nz
        else:
            last_nz = sign_d_nz[i]

    peaks_idx = []    # local maxima
    troughs_idx = []  # local minima
    for i in range(1, len(sign_d_nz)):
        if sign_d_nz[i - 1] > 0 and sign_d_nz[i] < 0:
            peaks_idx.append(i)
        elif sign_d_nz[i - 1] < 0 and sign_d_nz[i] > 0:
            troughs_idx.append(i)

    # qrs_num_peaks: total peaks + troughs
    qrs_num_peaks = max(1, len(peaks_idx) + len(troughs_idx))

    # qrs_notch_count: inflection points in QRS (reversals from dominant direction)
    # A notch is a secondary deflection with amplitude > 10% of QRS range
    qrs_range = float(np.max(np.abs(seg))) + 1e-9
    notch_thr = 0.10 * qrs_range
    qrs_notch_count = 0
    for idx in peaks_idx + troughs_idx:
        if abs(float(seg[idx])) > notch_thr:
            qrs_notch_count += 1
    # Subtract 1 for the primary R peak itself (at least one peak should exist)
    qrs_notch_count = max(0, qrs_notch_count - 1)

    # R' = second positive peak after the primary R, if any
    r_local = r - qrs_on   # local index of R within segment
    r_prime_amp_mv = None
    s_prime_amp_mv = None
    r_prime_idx = None
    s_prime_idx = None

    # Post-R peaks (positive → R')
    post_r_peaks = [i for i in peaks_idx if i > r_local and seg[i] > notch_thr]
    if post_r_peaks:
        r_prime_idx = post_r_peaks[0]
        r_prime_amp_mv = float(seg[r_prime_idx])

    # Post-R troughs (negative → S')
    post_r_troughs = [i for i in troughs_idx if i > r_local and seg[i] < -notch_thr]
    s_idx = post_r_troughs[0] if post_r_troughs else None
    if len(post_r_troughs) >= 2:
        # S' is the second trough after R (first is S)
        s_prime_idx = post_r_troughs[1]
        s_prime_amp_mv = float(seg[s_prime_idx])
    elif r_prime_amp_mv is not None:
        # If there's an R', look for an S' after it
        sp_list = [i for i in troughs_idx if i > (post_r_peaks[0] if post_r_peaks else r_local) and seg[i] < -notch_thr]
        if sp_list:
            s_prime_idx = sp_list[0]
            s_prime_amp_mv = float(seg[s_prime_idx])

    primary_r_idx = int(np.argmax(seg))
    duration_thr = max(0.008, 0.05 * qrs_range)

    def _duration_around(idx: Optional[int], polarity: int) -> Optional[float]:
        if idx is None or idx < 0 or idx >= n:
            return None
        if polarity > 0 and float(seg[idx]) <= duration_thr:
            return None
        if polarity < 0 and float(seg[idx]) >= -duration_thr:
            return None
        left = int(idx)
        right = int(idx)
        if polarity > 0:
            while left > 0 and float(seg[left - 1]) > duration_thr:
                left -= 1
            while right + 1 < n and float(seg[right + 1]) > duration_thr:
                right += 1
        else:
            while left > 0 and float(seg[left - 1]) < -duration_thr:
                left -= 1
            while right + 1 < n and float(seg[right + 1]) < -duration_thr:
                right += 1
        duration_ms = (right - left + 1) * 1000.0 / fs
        if duration_ms < 4.0 or duration_ms > 160.0:
            return None
        return float(duration_ms)

    durations = {
        "r_duration_ms": _duration_around(primary_r_idx, 1),
        "r_prime_duration_ms": _duration_around(r_prime_idx, 1),
        "s_duration_ms": _duration_around(s_idx, -1),
        "s_prime_duration_ms": _duration_around(s_prime_idx, -1),
    }
    return r_prime_amp_mv, s_prime_amp_mv, qrs_num_peaks, qrs_notch_count, durations


# ─────────────────────────────────────────────────────────────────────────────
# T004: Beat-level quality scoring
# ─────────────────────────────────────────────────────────────────────────────

def _beat_quality(
    sig: np.ndarray,
    local_r: int,
    qrs_on: Optional[int],
    qrs_off: Optional[int],
    baseline: float,
    fs: int,
    rep_qrs_snippet: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, bool]:
    """
    Compute per-beat per-lead quality scores (T004).

    Returns:
        beat_noise_score          – HF noise relative to R amplitude (0=clean, 1=noisy)
        beat_baseline_shift       – |mean PR segment − baseline| in signal units
        beat_template_corr        – cosine similarity with group representative QRS (0 if unavailable)
        beat_measurement_reliable – combined reliability flag
    """
    r_amp = abs(float(sig[local_r]) - baseline) + 1e-6

    # Noise score: std of signal in a 60ms pre-P window (nominally isoelectric)
    noise_lo = max(0, local_r - int(0.32 * fs))
    noise_hi = max(noise_lo + 1, local_r - int(0.26 * fs))
    noise_seg = sig[noise_lo:noise_hi] - baseline
    beat_noise_score = float(np.clip(np.std(noise_seg) / r_amp, 0.0, 1.0))

    # Baseline shift: deviation of PR segment from local baseline estimate
    bl_lo = max(0, local_r - int(0.22 * fs))
    bl_hi = max(bl_lo + 1, local_r - int(0.08 * fs))
    pr_seg = sig[bl_lo:bl_hi]
    beat_baseline_shift = float(abs(np.mean(pr_seg) - baseline)) if len(pr_seg) > 0 else 0.0

    # Template correlation: cosine similarity of beat QRS with representative beat QRS
    beat_template_corr = 0.0
    if (rep_qrs_snippet is not None and len(rep_qrs_snippet) > 2
            and qrs_on is not None and qrs_off is not None and qrs_off > qrs_on):
        beat_snip = sig[qrs_on : qrs_off + 1] - baseline
        if len(beat_snip) > 2:
            rep_interp = np.interp(
                np.linspace(0.0, 1.0, len(beat_snip)),
                np.linspace(0.0, 1.0, len(rep_qrs_snippet)),
                rep_qrs_snippet,
            )
            denom = np.linalg.norm(beat_snip) * np.linalg.norm(rep_interp)
            if denom > 1e-8:
                beat_template_corr = float(np.clip(np.dot(beat_snip, rep_interp) / denom, -1.0, 1.0))

    # Combined reliability gate
    noise_ok    = beat_noise_score < 0.25
    baseline_ok = beat_baseline_shift < 0.15 * r_amp
    corr_ok     = (beat_template_corr > 0.70) if beat_template_corr > 0.0 else True
    beat_measurement_reliable = noise_ok and baseline_ok and corr_ok

    return beat_noise_score, beat_baseline_shift, beat_template_corr, beat_measurement_reliable


# ─────────────────────────────────────────────────────────────────────────────
# T018: Representative-beat boundary priors
# ─────────────────────────────────────────────────────────────────────────────

def _delineate_rep_prior(
    rep_lead_sig: np.ndarray,
    fs: int,
    left_ms: int = 300,
    qrs_low_slope_guard: bool = False,
) -> Dict[str, Optional[int]]:
    """
    Delineate one lead of a representative beat and return boundary offsets
    from the R-peak position.  Used as priors for individual-beat correction.
    (T018)

    All returned values are sample offsets from local_r:
        negative  → before R
        positive  → after R
    """
    local_r = int(left_ms * fs / 1000)
    n = len(rep_lead_sig)
    if local_r >= n:
        return {}

    baseline, _, _ = _baseline_contract(rep_lead_sig, local_r, fs)

    qrs_on, qrs_off, _, _on_conf, _off_conf = _qrs_bounds(
        rep_lead_sig,
        local_r,
        fs,
        enable_low_slope_guard=qrs_low_slope_guard,
    )

    p_peak = _find_peak(
        rep_lead_sig,
        max(0, local_r - int(0.25 * fs)),
        max(0, local_r - int(0.04 * fs)),
        mode="abs",
    )
    p_on, p_off = _find_wave_bounds(rep_lead_sig, p_peak, baseline, thresh_frac=0.08)

    t_peak = _find_t_peak_candidate(
        rep_lead_sig,
        min(n, local_r + int(0.04 * fs)),
        min(n, local_r + int(0.50 * fs)),
        baseline,
        fs,
    )
    _t_on_start = qrs_off if qrs_off is not None else max(0, local_r + int(0.04 * fs))
    t_on = _t_onset_chord(rep_lead_sig, t_peak, _t_on_start, baseline, fs)
    if t_peak is not None:
        search_end = min(n - 1, local_r + int(0.55 * fs))
        t_off, t_conf, _ = _t_end_geometric(rep_lead_sig, t_peak, search_end, baseline, fs)
    else:
        t_off, t_conf = None, 0.0

    def _off(idx: Optional[int]) -> Optional[int]:
        return (idx - local_r) if idx is not None else None

    return {
        "qrs_on":   _off(qrs_on),
        "qrs_off":  _off(qrs_off),
        "p_on":     _off(p_on),
        "p_peak":   _off(p_peak),
        "p_off":    _off(p_off),
        "t_peak":   _off(t_peak),
        "t_on":     _off(t_on),
        "t_off":    _off(t_off),
        "t_conf":   t_conf,
        "baseline": baseline,
    }


def build_group_priors(
    rep_beats: Dict[int, np.ndarray],
    fs: int,
    left_ms: int = 300,
    qrs_low_slope_guard: bool = False,
) -> Dict[int, Dict[str, Dict[str, Optional[int]]]]:
    """
    Build boundary priors for every group × lead from representative beats.
    Returns: {group_id: {lead_name: {field: offset_from_r}}}
    """
    priors: Dict[int, Dict[str, Dict[str, Optional[int]]]] = {}
    for gid, rep in rep_beats.items():
        priors[gid] = {}
        for li, lead in enumerate(STANDARD_12_LEADS):
            if li < rep.shape[0]:
                priors[gid][lead] = _delineate_rep_prior(
                    rep[li],
                    fs,
                    left_ms,
                    qrs_low_slope_guard=qrs_low_slope_guard,
                )
            else:
                priors[gid][lead] = {}
    return priors


# ─────────────────────────────────────────────────────────────────────────────
# T024 paced-beat branch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _paced_retrograde_p(
    sig: np.ndarray,
    qrs_off_local: Optional[int],
    baseline: float,
    fs: int,
) -> Tuple[Optional[int], Optional[int], Optional[int], float]:
    """
    Search for a retrograde P-wave after a paced QRS (T024 paced-beat branch).

    In RV-paced or DDD rhythms the retrograde P (if present) appears in the
    ST segment as a small, typically negative deflection.
    Search window: [QRS_off + 40 ms, QRS_off + 250 ms].

    Returns: (p_on, p_peak, p_off, p_confidence)
    """
    if qrs_off_local is None:
        return None, None, None, 0.0
    lo = min(len(sig) - 1, qrs_off_local + int(0.040 * fs))
    hi = min(len(sig),     qrs_off_local + int(0.250 * fs))
    if hi <= lo + 2:
        return None, None, None, 0.0
    sm_win = max(3, int(0.015 * fs))
    seg_sm = np.convolve(sig[lo:hi], np.ones(sm_win) / sm_win, mode="same")
    pk_local = int(np.argmax(np.abs(seg_sm - baseline)))
    p_peak = lo + pk_local
    amp = float(sig[p_peak] - baseline)
    if abs(amp) < 0.03:            # < 0.3 mm — too small, skip
        return None, None, None, 0.0
    p_on,  _  = _find_wave_bounds(sig, p_peak, baseline, thresh_frac=0.08)
    _,  p_off = _find_wave_bounds(sig, p_peak, baseline, thresh_frac=0.08)
    if p_on is None or p_off is None or p_off <= p_on:
        return None, None, None, 0.0
    # Retrograde paced P waves are small and relatively narrow.  A broad
    # post-QRS deflection is usually the paced ST/T complex, not an atrial wave.
    if (p_off - p_on) > int(0.120 * fs):
        return None, None, None, 0.0
    confidence = float(np.clip(abs(amp) / 0.12, 0.0, 1.0))
    return p_on, p_peak, p_off, confidence


# ─────────────────────────────────────────────────────────────────────────────
# T027: Extended measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_st_morphology(slope_mv_per_ms: Optional[float]) -> Optional[str]:
    """
    Classify ST segment shape from the J-to-J+80 ms slope (T027).

    Thresholds (clinical convention, ±0.5 mV/s = ±0.0005 mV/ms):
        upsloping   : slope >  +0.0005 mV/ms
        downsloping : slope <  −0.0005 mV/ms
        horizontal  : |slope| ≤  0.0005 mV/ms
    """
    if slope_mv_per_ms is None or not np.isfinite(slope_mv_per_ms):
        return None
    if slope_mv_per_ms > 0.0005:
        return "upsloping"
    if slope_mv_per_ms < -0.0005:
        return "downsloping"
    return "horizontal"


def _st_j_tail_outlier(
    j_mv: float,
    st40_mv: float,
    st80_mv: float,
    qrs_ms: float,
) -> bool:
    return (
        qrs_ms >= 120.0
        and abs(j_mv - st40_mv) > 0.20
        and abs(j_mv - st80_mv) > 0.20
    )


def _measure_st_j_with_guard(
    *,
    sig,
    qrs_on,
    qrs_off,
    baseline,
    fs,
) -> Tuple[Optional[float], str, bool]:
    if qrs_on is None or qrs_off is None or fs <= 0:
        return None, "missing_qrs_bounds", False
    qrs_ms = (qrs_off - qrs_on) * 1000.0 / fs
    j = min(len(sig) - 1, qrs_off)
    st40 = min(len(sig) - 1, qrs_off + int(0.040 * fs))
    st80 = min(len(sig) - 1, qrs_off + int(0.080 * fs))
    j_mv = float(sig[j] - baseline)
    st40_mv = float(sig[st40] - baseline)
    st80_mv = float(sig[st80] - baseline)
    if _st_j_tail_outlier(j_mv, st40_mv, st80_mv, qrs_ms):
        return None, "qrs_tail_guard", False
    return j_mv, "qrs_offset", True


def _fqrs_score(qrs_notch_count: int, qrs_num_peaks: int) -> float:
    """
    Normalised fragmented-QRS score [0, 1] (T027).

    Clinical fQRS = ≥2 notches or additional R'/S' deflections.
    Raw score = notch count + surplus peaks beyond 3; normalised by 4.
    """
    raw = qrs_notch_count + max(0, qrs_num_peaks - 3)
    return float(np.clip(raw / 4.0, 0.0, 1.0))


def _ptf_v1(
    sig: np.ndarray,
    p_on: Optional[int],
    p_off: Optional[int],
    baseline: float,
    fs: int,
) -> Optional[float]:
    """
    P-terminal force in V1 (PTF-V1) (T027).

    Computes: amplitude_of_negative_terminal_P (mV) × its_duration (ms).
    A negative result indicates left atrial enlargement.
    Returns None when the P wave is absent, all-positive, or too short.

    Classic cut-off for LAE: PTF-V1 < −4 mV·ms (Morris: 0.04 mm·s = 0.004 mV·s = 4 mV·ms).
    """
    if p_on is None or p_off is None or p_off <= p_on + 2:
        return None
    seg = sig[p_on : p_off + 1] - baseline
    n = len(seg)
    if n < 4:
        return None
    # Find last positive→negative zero-crossing (terminal inflection)
    last_cross: Optional[int] = None
    for i in range(n - 2, 0, -1):
        if seg[i] >= 0.0 and seg[i + 1] < 0.0:
            last_cross = i + 1
            break
    if last_cross is None or last_cross >= n - 1:
        return None   # no terminal negative component
    terminal = seg[last_cross:]
    if len(terminal) < 2:
        return None
    neg_amp = float(np.min(terminal))
    if neg_amp >= 0.0:
        return None
    # A terminal negative *component* reaches its trough inside the P wave.  A
    # baseline drifting down through the P window keeps falling to the last
    # sample, and multiplying that trough by the whole run length reports it as a
    # terminal force (record 09048: a monotonic 210 -> 0 µV drift with no visible
    # P read as PTF -15 mV·ms).  Plotting the V1 P waves showed this is the only
    # artifact class here -- the deep components are real deflections -- so the
    # guard is deliberately narrow: reject only a trough at the very end, which
    # is 1.7% of records at the Morris bar and 11.1% of the shallow tail.
    if int(np.argmin(terminal)) >= len(terminal) - 2:
        return None
    dur_ms = len(terminal) * 1000.0 / fs
    return float(neg_amp * dur_ms)   # mV·ms (negative = left atrial force)


# ─────────────────────────────────────────────────────────────────────────────
# T025: Multi-lead P-onset / T-end consensus (post-processing pass)
# ─────────────────────────────────────────────────────────────────────────────

def _mad(values: List[float]) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _p_context_support_score(support: int) -> float:
    if support >= _P_CONTEXT_STRONG_SUPPORT:
        return 1.0
    if support >= _P_CONTEXT_MODERATE_SUPPORT:
        return 0.75
    if support == 1:
        return 0.25
    return 0.0


def _measure_p_candidate_from_peak(
    *,
    sig: np.ndarray,
    peak: int,
    search_lo: int,
    qrs_on: Optional[int],
    baseline: float,
    fs: int,
    p_noise_floor: float,
    lead: str,
) -> Optional[PCandidateMeasurement]:
    p_on_raw = _t_onset_tangent(sig, int(peak), int(search_lo), baseline, fs)
    if p_on_raw is None or int(p_on_raw) >= int(peak):
        return None
    p_on_rescued, _p_on_rescued = _rescue_long_pr_p_onset(
        sig,
        int(peak),
        p_on_raw,
        qrs_on,
        baseline,
        fs,
    )
    min_rescue_shift = max(2, int(round(0.008 * fs)))
    if (
        p_on_rescued is not None
        and int(p_on_rescued) < int(p_on_raw) - min_rescue_shift
    ):
        p_on = p_on_rescued
    else:
        p_on = p_on_raw
    p_off_limit = (
        int(qrs_on) - 1
        if qrs_on is not None
        else min(len(sig) - 1, int(peak + round(0.16 * fs)))
    )
    p_off = _p_offset_tangent(sig, int(peak), p_off_limit, baseline, fs)
    if p_on is None or p_off is None:
        return None
    p_on, p_off = _extend_p_candidate_edges(
        sig=sig,
        p_on=int(p_on),
        p_peak=int(peak),
        p_off=int(p_off),
        baseline=baseline,
        search_lo=int(search_lo),
        p_off_limit=int(p_off_limit),
        qrs_on=qrs_on,
        fs=fs,
    )
    if not (int(p_on) < int(peak) < int(p_off)):
        return None
    p_dur_ms = (int(p_off) - int(p_on)) * 1000.0 / fs
    if not (_P_DURATION_MIN_MS <= p_dur_ms <= _P_DURATION_MAX_MS):
        return None
    onset_to_peak_ms = (int(peak) - int(p_on)) * 1000.0 / fs
    if onset_to_peak_ms > _P_ONSET_TO_PEAK_MAX_MS:
        return None
    pr_ms = None
    if qrs_on is not None:
        pr_ms = (int(qrs_on) - int(p_on)) * 1000.0 / fs
        pr_segment_ms = (int(qrs_on) - int(p_off)) * 1000.0 / fs
        if not (_PR_SEGMENT_MIN_MS <= pr_segment_ms <= _PR_SEGMENT_MAX_MS):
            return None
    # Amplitude/area reference: the isoelectric tissue beside the P wave, not the
    # boundary-detection baseline (whose window the P wave sits inside).
    p_amp_baseline, _p_amp_baseline_source = _p_isoelectric_reference(
        sig,
        p_on=int(p_on),
        p_off=int(p_off),
        qrs_on=qrs_on,
        fs=fs,
        fallback=baseline,
    )
    p_amp = float(sig[int(peak)] - p_amp_baseline)
    p_seg = sig[int(p_on): int(p_off) + 1] - p_amp_baseline
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    p_area = float(trapz(np.abs(p_seg)))
    p_signed_area = float(trapz(p_seg))
    p_confidence = float(np.clip(abs(p_amp) / (float(p_noise_floor) + 1e-4), 0.0, 1.0))
    p_components = measure_p_components(
        sig,
        p_on=int(p_on),
        p_peak=int(peak),
        p_off=int(p_off),
        baseline=p_amp_baseline,
        fs=fs,
    )
    ptf_v1 = (
        _ptf_v1(sig, int(p_on), int(p_off), p_amp_baseline, fs)
        if lead == "V1" and p_confidence > 0.30
        else None
    )
    return PCandidateMeasurement(
        onset=int(p_on),
        peak=int(peak),
        offset=int(p_off),
        pr_ms=pr_ms,
        p_dur_ms=float(p_dur_ms),
        p_amp_mv=p_amp,
        p_area=p_area,
        p_signed_area=p_signed_area,
        p_confidence=p_confidence,
        p_components=p_components,
        ptf_v1_mv_ms=ptf_v1,
        context_score=0.0,
        template_corr=_P_CONTEXT_NEUTRAL_SCORE,
        multilead_support=0,
        multilead_support_score=0.0,
        pr_consistency_score=_P_CONTEXT_NEUTRAL_SCORE,
        pp_consistency_score=_P_CONTEXT_NEUTRAL_SCORE,
    )


def _repair_lead_indices(
    beat_features: List["LeadBeatFeatures"],
    ecg: np.ndarray,
) -> Dict[str, int]:
    standard = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    if ecg.shape[0] >= len(STANDARD_12_LEADS):
        return standard
    ordered: List[str] = []
    for feature in beat_features:
        if feature.lead not in ordered:
            ordered.append(feature.lead)
    if len(ordered) == ecg.shape[0]:
        return {lead: idx for idx, lead in enumerate(ordered)}
    return standard


def _apply_p_candidate_context(
    beat_features: List["LeadBeatFeatures"],
    *,
    ecg: np.ndarray,
    fs: int,
    quality: Optional[Dict[str, object]],
    beat_to_group: Optional[Dict[int, int]],
    suppress: bool = True,
) -> List["LeadBeatFeatures"]:
    if not beat_features or ecg is None or fs <= 0:
        return beat_features
    quality = quality or {}
    beat_to_group = beat_to_group or {}
    lead_to_index = _repair_lead_indices(beat_features, ecg)
    half_width = max(2, int(round(_P_CONTEXT_TEMPLATE_HALF_MS * fs / 1000.0)))

    def _usable_p(feature: "LeadBeatFeatures") -> bool:
        flags = set(getattr(feature, "flags", []) or [])
        return (
            feature.p.onset is not None
            and feature.p.peak is not None
            and feature.p.offset is not None
            and "p_unreliable" not in flags
            and "paced_beat" not in flags
            and "retrograde_p" not in flags
        )

    def _snippet_for(feature: "LeadBeatFeatures") -> Optional[np.ndarray]:
        li = lead_to_index.get(feature.lead)
        if li is None or li >= ecg.shape[0] or feature.p.peak is None:
            return None
        return _p_context_snippet(ecg[li], int(feature.p.peak), half_width)

    seeds: Dict[Tuple[int, str], List[np.ndarray]] = {}
    for feature in beat_features:
        if not _usable_p(feature):
            continue
        if float(getattr(feature, "p_confidence", 0.0) or 0.0) < 0.40:
            continue
        p_dur = getattr(feature, "p_dur_ms", None)
        if p_dur is None and feature.p.onset is not None and feature.p.offset is not None:
            p_dur = (int(feature.p.offset) - int(feature.p.onset)) * 1000.0 / fs
        if p_dur is None or not (_P_DURATION_MIN_MS <= float(p_dur) <= _P_DURATION_MAX_MS):
            continue
        pr_ms = getattr(feature, "pr_ms", None)
        if pr_ms is not None and not (60.0 <= float(pr_ms) <= 350.0):
            continue
        snippet = _snippet_for(feature)
        if snippet is None:
            continue
        group_id = int(beat_to_group.get(int(feature.beat_id), 1))
        seeds.setdefault((group_id, feature.lead), []).append(snippet)

    templates: Dict[Tuple[int, str], np.ndarray] = {
        key: np.median(np.vstack(snippets), axis=0)
        for key, snippets in seeds.items()
        if len(snippets) >= _P_CONTEXT_MIN_TEMPLATE_SEEDS
    }

    by_beat: Dict[int, List["LeadBeatFeatures"]] = {}
    for feature in beat_features:
        by_beat.setdefault(int(feature.beat_id), []).append(feature)

    pr_values = [
        float(feature.pr_ms)
        for feature in beat_features
        if _usable_p(feature)
        and feature.pr_ms is not None
        and float(getattr(feature, "p_confidence", 0.0) or 0.0) >= 0.40
    ]
    pr_center = float(np.median(pr_values)) if len(pr_values) >= _P_CONTEXT_MIN_PR_VALUES else None
    pr_mad = _mad(pr_values) if pr_center is not None else None
    pr_enabled = bool(pr_center is not None and pr_mad is not None and pr_mad <= _P_CONTEXT_PR_STABLE_MAD_MS)

    p_events: List[Tuple[int, int]] = []
    for feature in beat_features:
        if _usable_p(feature) and feature.p.peak is not None and float(getattr(feature, "p_confidence", 0.0) or 0.0) >= 0.40:
            p_events.append((int(feature.beat_id), int(feature.p.peak)))
    p_events.sort()
    beat_peak_by_id: Dict[int, float] = {}
    for beat_id in sorted({beat for beat, _peak in p_events}):
        peaks = [peak for beat, peak in p_events if beat == beat_id]
        if peaks:
            beat_peak_by_id[beat_id] = float(np.median(peaks))
    ordered_beats = sorted(beat_peak_by_id)
    pp_values = [
        (beat_peak_by_id[b2] - beat_peak_by_id[b1]) * 1000.0 / fs
        for b1, b2 in zip(ordered_beats, ordered_beats[1:])
    ]
    pp_center = float(np.median(pp_values)) if len(pp_values) >= _P_CONTEXT_MIN_PP_VALUES - 1 else None
    pp_mad = _mad(pp_values) if pp_center is not None else None
    pp_enabled = bool(pp_center is not None and pp_mad is not None and pp_mad <= _P_CONTEXT_PP_STABLE_MAD_MS)

    radius = max(1, int(round(_P_CONTEXT_SUPPORT_RADIUS_MS * fs / 1000.0)))
    to_suppress: List["LeadBeatFeatures"] = []
    for feature in beat_features:
        if not _usable_p(feature):
            continue
        group_id = int(beat_to_group.get(int(feature.beat_id), 1))
        snippet = _snippet_for(feature)
        template = templates.get((group_id, feature.lead))
        template_corr = _p_context_corr(snippet, template)

        support_leads = set()
        if feature.p.peak is not None:
            for peer in by_beat.get(int(feature.beat_id), []):
                if peer.p.peak is None:
                    continue
                if not getattr(quality.get(peer.lead), "reliable_for_p", False):
                    continue
                peer_flags = set(getattr(peer, "flags", []) or [])
                if "paced_beat" in peer_flags or "retrograde_p" in peer_flags:
                    continue
                if abs(int(peer.p.peak) - int(feature.p.peak)) <= radius:
                    support_leads.add(peer.lead)
        support = len(support_leads)
        limb_support = len(support_leads & _P_CONTEXT_LIMB_LEADS)
        support_score = _p_context_support_score(support)

        pr_score = (
            _p_context_robust_score(feature.pr_ms, pr_center, pr_mad)
            if pr_enabled else _P_CONTEXT_NEUTRAL_SCORE
        )
        pp_score = _P_CONTEXT_NEUTRAL_SCORE
        if pp_enabled and feature.p.peak is not None:
            beat_id = int(feature.beat_id)
            prev_beats = [b for b in ordered_beats if b < beat_id]
            next_beats = [b for b in ordered_beats if b > beat_id]
            predictions: List[float] = []
            if prev_beats:
                predictions.append(beat_peak_by_id[prev_beats[-1]] + pp_center * fs / 1000.0)
            if next_beats:
                predictions.append(beat_peak_by_id[next_beats[0]] - pp_center * fs / 1000.0)
            if predictions:
                expected = float(np.median(predictions))
                pp_score = _p_context_robust_score(
                    float(feature.p.peak),
                    expected,
                    max((pp_mad or 0.0) * fs / 1000.0, 1.0),
                    floor_scale=max(1.0, 0.025 * fs),
                )

        geometry_score = 1.0
        if feature.p.onset is None or feature.p.offset is None or feature.p.peak is None:
            geometry_score = 0.0
        elif not (int(feature.p.onset) < int(feature.p.peak) < int(feature.p.offset)):
            geometry_score = 0.0

        context_score = float(np.clip(
            0.25 * float(getattr(feature, "p_confidence", 0.0) or 0.0)
            + 0.25 * template_corr
            + 0.20 * support_score
            + 0.15 * pr_score
            + 0.10 * pp_score
            + 0.05 * geometry_score,
            0.0,
            1.0,
        ))
        feature.p_template_corr = template_corr
        feature.p_multilead_support = support
        feature.p_multilead_support_score = support_score
        feature.p_pr_consistency_score = pr_score
        feature.p_pp_consistency_score = pp_score
        feature.p_candidate_context_score = context_score
        feature.p_context_reason = "context_scored"

        weak_components = 0
        if template_corr < _P_CONTEXT_COMPONENT_LOW and template is not None:
            weak_components += 1
        if support_score < _P_CONTEXT_COMPONENT_LOW:
            weak_components += 1
        if pr_enabled and pr_score < _P_CONTEXT_COMPONENT_LOW:
            weak_components += 1
        if pp_enabled and pp_score < _P_CONTEXT_COMPONENT_LOW:
            weak_components += 1

        flags = set(getattr(feature, "flags", []) or [])
        template_weak = template is not None and template_corr < _P_CONTEXT_COMPONENT_LOW
        support_weak = support_score < _P_CONTEXT_COMPONENT_LOW
        timing_weak = (
            (pr_enabled and pr_score < _P_CONTEXT_COMPONENT_LOW)
            or (pp_enabled and pp_score < _P_CONTEXT_COMPONENT_LOW)
        )
        short_pr_without_limb_support = (
            feature.pr_ms is not None
            and float(feature.pr_ms) < _P_CONTEXT_SHORT_PR_NO_LIMB_MS
            and limb_support == 0
            and template_weak
            and timing_weak
            and float(getattr(feature, "p_confidence", 0.0) or 0.0) < 0.45
        )
        can_suppress = (
            suppress
            and (
                (
                    context_score < _P_CONTEXT_SCORE_LOW
                    and weak_components >= 2
                    and template_weak
                    and support_weak
                    and timing_weak
                )
                or short_pr_without_limb_support
            )
            and "paced_beat" not in flags
            and "retrograde_p" not in flags
        )
        if can_suppress:
            to_suppress.append(feature)

    for feature in to_suppress:
        feature.p = WaveBounds(onset=None, peak=None, offset=None)
        feature.pr_ms = None
        feature.p_dur_ms = None
        feature.p_amp_mv = None
        feature.p_area = None
        feature.p_signed_area = None
        feature.ptf_v1_mv_ms = None
        feature.p_notched = False
        feature.p_biphasic = False
        feature.p_notch_interval_ms = None
        feature.p_initial_duration_ms = None
        feature.p_initial_amp_mv = None
        feature.p_terminal_duration_ms = None
        feature.p_terminal_amp_mv = None
        feature.p_terminal_area_mv_ms = None
        feature.p_confidence = 0.0
        feature.p_context_reason = "context_suppressed"
        if "p_candidate_suppressed_by_context" not in feature.flags:
            feature.flags.append("p_candidate_suppressed_by_context")
        if "p_unreliable" not in feature.flags:
            feature.flags.append("p_unreliable")

    return beat_features


def _p_candidate_peak(candidate: object) -> Optional[int]:
    if hasattr(candidate, "peak"):
        return int(getattr(candidate, "peak"))
    if isinstance(candidate, dict):
        for key in ("peak", "sample", "peak_sample"):
            if candidate.get(key) is not None:
                return int(candidate[key])
    if isinstance(candidate, (tuple, list)) and candidate:
        return int(candidate[0])
    return None


def _p_candidate_reselection_reason(candidate: object) -> str:
    if isinstance(candidate, dict):
        method = str(candidate.get("detection_method") or "")
        if method == "composite_qrst_residual_derivative":
            return "raw_atrial_event_context"
    return "higher_context_alternative"


def _augment_p_alternatives_from_raw_atrial_events(
    beat_features: List["LeadBeatFeatures"],
    raw_events: List[object],
    *,
    fs: int,
    quality: Optional[Dict[str, object]],
    alternatives_by_key: Dict[Tuple[int, str], List[object]],
) -> None:
    if not beat_features or not raw_events or fs <= 0:
        return
    quality = quality or {}
    by_beat: Dict[int, List["LeadBeatFeatures"]] = {}
    for feature in beat_features:
        by_beat.setdefault(int(feature.beat_id), []).append(feature)

    duplicate_radius = max(1, int(round(_P_RAW_ATRIAL_EVENT_DUPLICATE_RADIUS_MS * fs / 1000.0)))

    def _event_value(event: object, key: str) -> Optional[object]:
        return event.get(key) if isinstance(event, dict) else getattr(event, key, None)

    def _candidate_beat_ids(sample: int, event: object) -> List[int]:
        assoc = _event_value(event, "association_type")
        beat_id = _event_value(event, "associated_qrs_beat_id")
        if beat_id is not None and assoc in (None, "conducted", "unknown"):
            return [int(beat_id)]
        ids: List[int] = []
        for current_beat, features in by_beat.items():
            qrs_on_values = [
                int(feature.qrs.onset)
                for feature in features
                if feature.qrs.onset is not None
            ]
            if not qrs_on_values:
                continue
            qrs_on = int(round(float(np.median(qrs_on_values))))
            pr_ms = (qrs_on - int(sample)) * 1000.0 / float(fs)
            if 60.0 <= pr_ms <= 280.0:
                ids.append(int(current_beat))
        return ids

    for event in raw_events:
        confidence = float(_event_value(event, "confidence") or 0.0)
        if confidence < _P_RAW_ATRIAL_EVENT_MIN_CONFIDENCE:
            continue
        sample = _p_candidate_peak(event)
        if sample is None:
            continue
        sample = int(sample)
        for beat_id in _candidate_beat_ids(sample, event):
            for feature in by_beat.get(int(beat_id), []):
                if quality and not getattr(quality.get(feature.lead), "reliable_for_p", False):
                    continue
                if feature.qrs.onset is None:
                    continue
                pr_ms = (int(feature.qrs.onset) - sample) * 1000.0 / float(fs)
                if not (60.0 <= pr_ms <= 280.0):
                    continue
                if feature.p.peak is not None and abs(int(feature.p.peak) - sample) <= duplicate_radius:
                    continue
                key = (int(feature.beat_id), feature.lead)
                existing = alternatives_by_key.setdefault(key, [])
                if any(_p_candidate_peak(candidate) == sample for candidate in existing):
                    continue
                alt = dict(event) if isinstance(event, dict) else {"sample": sample}
                alt["sample"] = sample
                alt.setdefault("detection_method", "composite_qrst_residual_derivative")
                existing.append(alt)


def _apply_p_measurement_to_feature(
    feature: "LeadBeatFeatures",
    measurement: PCandidateMeasurement,
    *,
    old_peak: int,
    old_score: Optional[float],
    reason: str,
) -> None:
    feature.p = WaveBounds(
        onset=int(measurement.onset),
        peak=int(measurement.peak),
        offset=int(measurement.offset),
    )
    feature.pr_ms = measurement.pr_ms
    feature.p_dur_ms = measurement.p_dur_ms
    feature.p_amp_mv = measurement.p_amp_mv
    feature.p_area = measurement.p_area
    feature.p_signed_area = measurement.p_signed_area
    feature.p_confidence = measurement.p_confidence
    feature.ptf_v1_mv_ms = measurement.ptf_v1_mv_ms
    feature.p_notched = bool(measurement.p_components["is_notched"])
    feature.p_biphasic = bool(measurement.p_components["is_biphasic"])
    feature.p_notch_interval_ms = measurement.p_components["notch_interval_ms"]
    feature.p_initial_duration_ms = measurement.p_components["initial_duration_ms"]
    feature.p_initial_amp_mv = measurement.p_components["initial_amplitude_mV"]
    feature.p_terminal_duration_ms = measurement.p_components["terminal_duration_ms"]
    feature.p_terminal_amp_mv = measurement.p_components["terminal_amplitude_mV"]
    feature.p_terminal_area_mv_ms = measurement.p_components["terminal_area_mv_ms"]
    feature.p_template_corr = measurement.template_corr
    feature.p_multilead_support = measurement.multilead_support
    feature.p_multilead_support_score = measurement.multilead_support_score
    feature.p_pr_consistency_score = measurement.pr_consistency_score
    feature.p_pp_consistency_score = measurement.pp_consistency_score
    feature.p_candidate_context_score = measurement.context_score
    feature.p_context_reason = "context_reselected"
    feature.p_reselected_from_peak_index = int(old_peak)
    feature.p_reselected_to_peak_index = int(measurement.peak)
    feature.p_reselected_old_context_score = old_score
    feature.p_reselected_new_context_score = measurement.context_score
    feature.p_reselection_reason = reason
    if "p_candidate_reselected_by_context" not in feature.flags:
        feature.flags.append("p_candidate_reselected_by_context")
    if "p_unreliable" in feature.flags:
        feature.flags.remove("p_unreliable")


def _apply_p_candidate_reselection(
    beat_features: List["LeadBeatFeatures"],
    *,
    ecg: np.ndarray,
    fs: int,
    quality: Optional[Dict[str, object]],
    beat_to_group: Optional[Dict[int, int]],
    alternatives_by_key: Dict[Tuple[int, str], List[object]],
) -> List["LeadBeatFeatures"]:
    if not beat_features or not alternatives_by_key or ecg is None or fs <= 0:
        return beat_features
    quality = quality or {}
    beat_to_group = beat_to_group or {}
    scored = _apply_p_candidate_context(
        beat_features,
        ecg=ecg,
        fs=fs,
        quality=quality,
        beat_to_group=beat_to_group,
        suppress=False,
    )
    lead_to_index = _repair_lead_indices(scored, ecg)
    half_width = max(2, int(round(_P_CONTEXT_TEMPLATE_HALF_MS * fs / 1000.0)))
    radius = max(1, int(round(_P_CONTEXT_SUPPORT_RADIUS_MS * fs / 1000.0)))

    by_key = {(int(feature.beat_id), feature.lead): feature for feature in scored}
    by_beat: Dict[int, List["LeadBeatFeatures"]] = {}
    for feature in scored:
        by_beat.setdefault(int(feature.beat_id), []).append(feature)
        feature.p_candidate_alternative_count = len(
            alternatives_by_key.get((int(feature.beat_id), feature.lead), [])
        )

    def _usable_p(feature: "LeadBeatFeatures") -> bool:
        flags = set(getattr(feature, "flags", []) or [])
        return (
            feature.p.onset is not None
            and feature.p.peak is not None
            and feature.p.offset is not None
            and "p_unreliable" not in flags
            and "paced_beat" not in flags
            and "retrograde_p" not in flags
        )

    def _snippet_for_peak(lead: str, peak: Optional[int]) -> Optional[np.ndarray]:
        li = lead_to_index.get(lead)
        if li is None or li >= ecg.shape[0] or peak is None:
            return None
        return _p_context_snippet(ecg[li], int(peak), half_width)

    seeds: Dict[Tuple[int, str], List[np.ndarray]] = {}
    for feature in scored:
        if not _usable_p(feature):
            continue
        if float(getattr(feature, "p_confidence", 0.0) or 0.0) < 0.40:
            continue
        p_dur = getattr(feature, "p_dur_ms", None)
        if p_dur is None and feature.p.onset is not None and feature.p.offset is not None:
            p_dur = (int(feature.p.offset) - int(feature.p.onset)) * 1000.0 / fs
        if p_dur is None or not (_P_DURATION_MIN_MS <= float(p_dur) <= _P_DURATION_MAX_MS):
            continue
        pr_ms = getattr(feature, "pr_ms", None)
        if pr_ms is not None and not (60.0 <= float(pr_ms) <= 350.0):
            continue
        snippet = _snippet_for_peak(feature.lead, feature.p.peak)
        if snippet is None:
            continue
        group_id = int(beat_to_group.get(int(feature.beat_id), 1))
        seeds.setdefault((group_id, feature.lead), []).append(snippet)

    templates: Dict[Tuple[int, str], np.ndarray] = {
        key: np.median(np.vstack(snippets), axis=0)
        for key, snippets in seeds.items()
        if len(snippets) >= _P_CONTEXT_MIN_TEMPLATE_SEEDS
    }

    pr_values = [
        float(feature.pr_ms)
        for feature in scored
        if _usable_p(feature)
        and feature.pr_ms is not None
        and float(getattr(feature, "p_confidence", 0.0) or 0.0) >= 0.40
    ]
    pr_center = float(np.median(pr_values)) if len(pr_values) >= _P_CONTEXT_MIN_PR_VALUES else None
    pr_mad = _mad(pr_values) if pr_center is not None else None
    pr_enabled = bool(pr_center is not None and pr_mad is not None and pr_mad <= _P_CONTEXT_PR_STABLE_MAD_MS)

    p_events = [
        (int(feature.beat_id), int(feature.p.peak))
        for feature in scored
        if _usable_p(feature)
        and feature.p.peak is not None
        and float(getattr(feature, "p_confidence", 0.0) or 0.0) >= 0.40
    ]
    p_events.sort()
    beat_peak_by_id: Dict[int, float] = {}
    for beat_id in sorted({beat for beat, _peak in p_events}):
        peaks = [peak for beat, peak in p_events if beat == beat_id]
        if peaks:
            beat_peak_by_id[beat_id] = float(np.median(peaks))
    ordered_beats = sorted(beat_peak_by_id)
    pp_values = [
        (beat_peak_by_id[b2] - beat_peak_by_id[b1]) * 1000.0 / fs
        for b1, b2 in zip(ordered_beats, ordered_beats[1:])
    ]
    pp_center = float(np.median(pp_values)) if len(pp_values) >= _P_CONTEXT_MIN_PP_VALUES - 1 else None
    pp_mad = _mad(pp_values) if pp_center is not None else None
    pp_enabled = bool(pp_center is not None and pp_mad is not None and pp_mad <= _P_CONTEXT_PP_STABLE_MAD_MS)

    for key, candidates in alternatives_by_key.items():
        feature = by_key.get((int(key[0]), key[1]))
        if feature is None:
            continue
        flags = set(getattr(feature, "flags", []) or [])
        if "paced_beat" in flags or "retrograde_p" in flags:
            continue
        has_raw_atrial_candidate = any(
            _p_candidate_reselection_reason(candidate) == "raw_atrial_event_context"
            for candidate in candidates
        )
        old_peak = feature.p.peak
        missing_original_p = old_peak is None
        if missing_original_p and not has_raw_atrial_candidate:
            continue
        old_score = feature.p_candidate_context_score
        effective_old_score = 0.0 if missing_original_p and old_score is None else old_score
        review_score = (
            _P_RAW_ATRIAL_EVENT_REVIEW_SCORE
            if has_raw_atrial_candidate
            else _P_RESELECT_REVIEW_SCORE
        )
        if effective_old_score is None or float(effective_old_score) >= review_score:
            continue
        li = lead_to_index.get(feature.lead)
        if li is None or li >= ecg.shape[0]:
            continue
        qrs_on = feature.qrs.onset
        if qrs_on is None:
            continue
        qrs_ref = feature.qrs.peak if feature.qrs.peak is not None else qrs_on
        baseline, _baseline_source, _baseline_confidence = _baseline_contract(ecg[li], int(qrs_ref), fs)
        p_conf = max(float(feature.p_confidence or 0.0), 1e-3)
        p_noise_floor = abs(float(feature.p_amp_mv or 0.0)) / p_conf
        p_noise_floor = max(float(p_noise_floor), 1e-4)
        best: Optional[PCandidateMeasurement] = None
        best_reason = "higher_context_alternative"
        for candidate in candidates:
            peak = _p_candidate_peak(candidate)
            reason = _p_candidate_reselection_reason(candidate)
            if peak is None:
                continue
            if old_peak is not None and peak == old_peak:
                continue
            if missing_original_p and reason != "raw_atrial_event_context":
                continue
            measurement = _measure_p_candidate_from_peak(
                sig=ecg[li],
                peak=int(peak),
                search_lo=max(0, int(peak - round(0.18 * fs))),
                qrs_on=int(qrs_on),
                baseline=baseline,
                fs=fs,
                p_noise_floor=p_noise_floor,
                lead=feature.lead,
            )
            if measurement is None:
                continue

            support_leads = set()
            for peer in by_beat.get(int(feature.beat_id), []):
                if not getattr(quality.get(peer.lead), "reliable_for_p", False):
                    continue
                peer_flags = set(getattr(peer, "flags", []) or [])
                if "paced_beat" in peer_flags or "retrograde_p" in peer_flags:
                    continue
                peer_peak = int(measurement.peak) if peer is feature else peer.p.peak
                if peer_peak is None:
                    continue
                if abs(int(peer_peak) - int(measurement.peak)) <= radius:
                    support_leads.add(peer.lead)
            measurement.multilead_support = len(support_leads)
            measurement.multilead_support_score = _p_context_support_score(measurement.multilead_support)

            group_id = int(beat_to_group.get(int(feature.beat_id), 1))
            snippet = _snippet_for_peak(feature.lead, measurement.peak)
            measurement.template_corr = _p_context_corr(snippet, templates.get((group_id, feature.lead)))
            measurement.pr_consistency_score = (
                _p_context_robust_score(measurement.pr_ms, pr_center, pr_mad)
                if pr_enabled else _P_CONTEXT_NEUTRAL_SCORE
            )
            measurement.pp_consistency_score = _P_CONTEXT_NEUTRAL_SCORE
            if pp_enabled:
                beat_id = int(feature.beat_id)
                prev_beats = [b for b in ordered_beats if b < beat_id]
                next_beats = [b for b in ordered_beats if b > beat_id]
                predictions: List[float] = []
                if prev_beats:
                    predictions.append(beat_peak_by_id[prev_beats[-1]] + pp_center * fs / 1000.0)
                if next_beats:
                    predictions.append(beat_peak_by_id[next_beats[0]] - pp_center * fs / 1000.0)
                if predictions:
                    expected = float(np.median(predictions))
                    measurement.pp_consistency_score = _p_context_robust_score(
                        float(measurement.peak),
                        expected,
                        max((pp_mad or 0.0) * fs / 1000.0, 1.0),
                        floor_scale=max(1.0, 0.025 * fs),
                    )
            measurement.context_score = float(np.clip(
                0.25 * float(measurement.p_confidence)
                + 0.25 * measurement.template_corr
                + 0.20 * measurement.multilead_support_score
                + 0.15 * measurement.pr_consistency_score
                + 0.10 * measurement.pp_consistency_score
                + 0.05,
                0.0,
                1.0,
            ))

            candidate_review_score = (
                _P_RAW_ATRIAL_EVENT_REVIEW_SCORE
                if reason == "raw_atrial_event_context"
                else _P_RESELECT_REVIEW_SCORE
            )
            if not _p_should_reselect_candidate(
                old_score=effective_old_score,
                new_score=measurement.context_score,
                old_support_score=0.0 if missing_original_p else float(feature.p_multilead_support_score or 0.0),
                new_support_score=measurement.multilead_support_score,
                old_template_corr=0.0 if missing_original_p else float(feature.p_template_corr or _P_CONTEXT_NEUTRAL_SCORE),
                new_template_corr=measurement.template_corr,
                old_pr_score=0.0 if missing_original_p else float(feature.p_pr_consistency_score or _P_CONTEXT_NEUTRAL_SCORE),
                new_pr_score=measurement.pr_consistency_score,
                old_pp_score=0.0 if missing_original_p else float(feature.p_pp_consistency_score or _P_CONTEXT_NEUTRAL_SCORE),
                new_pp_score=measurement.pp_consistency_score,
                review_score=candidate_review_score,
            ):
                continue
            if best is None or measurement.context_score > best.context_score:
                best = measurement
                best_reason = reason
        if best is not None:
            _apply_p_measurement_to_feature(
                feature,
                best,
                old_peak=int(old_peak) if old_peak is not None else int(qrs_on),
                old_score=effective_old_score,
                reason=best_reason,
            )
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# ST/J raw remeasurement (runs before T-end repair and multilead consensus)
# ─────────────────────────────────────────────────────────────────────────────
_ST_J_REMEASURE_MIN_SUPPORT = 4
_ST_J_REMEASURE_MIN_CONF = 0.10
_ST_J_REMEASURE_MIN_QRS_MS = 40.0
_ST_J_REMEASURE_MAX_QRS_MS = 220.0
_ST_J_REMEASURE_OFFSET_OUTLIER_MS = 25.0
_ST_J_REMEASURE_STABLE_SPREAD_MV = 0.12
_ST_J_REMEASURE_TAIL_JUMP_MV = 0.18
_ST_J_REMEASURE_MAX_SHIFT_MS = 60.0


def _st_j_csv(items: List[Tuple["LeadBeatFeatures", int]]) -> str:
    return ",".join(feature.lead for feature, _index in items)


def _st_j_remeasurement_target(
    items: List["LeadBeatFeatures"],
    *,
    fs: int,
    quality: Dict[str, object],
) -> Optional[STJRepairTarget]:
    if fs <= 0:
        return None

    used: List[Tuple["LeadBeatFeatures", int]] = []
    excluded: List[Tuple["LeadBeatFeatures", int]] = []
    for feature in items:
        j_index = feature.qrs.offset if feature.qrs.offset is not None else feature.j_index
        if j_index is None or feature.qrs.onset is None:
            excluded.append((feature, int(j_index or 0)))
            continue
        if not getattr(quality.get(feature.lead), "reliable_for_qrs", False):
            excluded.append((feature, int(j_index)))
            continue
        if "qrs_unreliable" in (getattr(feature, "flags", []) or []):
            excluded.append((feature, int(j_index)))
            continue
        qrs_conf = float(getattr(feature, "qrs_confidence", 0.0) or 0.0)
        off_conf = float(getattr(feature, "qrs_off_confidence", 0.0) or 0.0)
        if qrs_conf <= _ST_J_REMEASURE_MIN_CONF or off_conf <= _ST_J_REMEASURE_MIN_CONF:
            excluded.append((feature, int(j_index)))
            continue
        qrs_ms = (int(j_index) - int(feature.qrs.onset)) * 1000.0 / fs
        if not (_ST_J_REMEASURE_MIN_QRS_MS <= qrs_ms <= _ST_J_REMEASURE_MAX_QRS_MS):
            excluded.append((feature, int(j_index)))
            continue
        used.append((feature, int(j_index)))

    if len(used) < _ST_J_REMEASURE_MIN_SUPPORT:
        return None
    target_index = int(np.floor(float(np.median([index for _feature, index in used])) + 0.5))
    return STJRepairTarget(
        j_index=target_index,
        support=len(used),
        used_leads=_st_j_csv(used),
        excluded_leads=_st_j_csv(excluded),
    )


def _st_j_sample_values(
    sig: np.ndarray,
    *,
    anchor: int,
    baseline: float,
    fs: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    if fs <= 0 or len(sig) == 0:
        return None, None, None, None, None
    j = int(anchor)
    st40 = j + int(round(0.040 * fs))
    st80 = j + int(round(0.080 * fs))
    if j < 0 or j >= len(sig) or st40 < 0 or st40 >= len(sig) or st80 < 0 or st80 >= len(sig):
        return None, None, None, None, None
    st_on = float(sig[j] - baseline)
    st_mid = float(sig[st40] - baseline)
    st_80 = float(sig[st80] - baseline)
    slope = (st_80 - st_on) / 80.0
    return st_on, st_mid, st_80, slope, _classify_st_morphology(slope)


def _st_j_values_finite(feature: "LeadBeatFeatures") -> Optional[Tuple[float, float, float]]:
    values = (feature.st_on_mv, feature.st_mid_mv, feature.st_80ms_mv)
    if any(value is None for value in values):
        return None
    out = tuple(float(value) for value in values)
    if not all(np.isfinite(value) for value in out):
        return None
    return out


def _st_j_is_stable(feature: "LeadBeatFeatures") -> bool:
    values = _st_j_values_finite(feature)
    if values is None:
        return False
    if "st_j_unreliable" in (getattr(feature, "flags", []) or []):
        return False
    spread = max(values) - min(values)
    return spread <= _ST_J_REMEASURE_STABLE_SPREAD_MV


def _st_j_has_tail_jump(feature: "LeadBeatFeatures") -> bool:
    values = _st_j_values_finite(feature)
    if values is None:
        return False
    st_on, st_mid, st_80 = values
    return (
        abs(st_on - st_mid) >= _ST_J_REMEASURE_TAIL_JUMP_MV
        and abs(st_on - st_80) >= _ST_J_REMEASURE_TAIL_JUMP_MV
    )


def _st_j_remeasurement_candidate(
    feature: "LeadBeatFeatures",
    *,
    sig: np.ndarray,
    beat_start: int,
    fs: int,
    target: STJRepairTarget,
    next_qrs_guard: Optional[int],
) -> Tuple[Optional[int], Optional[str]]:
    if fs <= 0 or sig is None or len(sig) == 0:
        return None, None
    if feature.qrs.onset is None or feature.qrs.offset is None:
        return None, None

    old_j = int(feature.j_index if feature.j_index is not None else feature.qrs.offset)
    target_j = int(target.j_index)
    delta_ms = (old_j - target_j) * 1000.0 / fs
    flags = getattr(feature, "flags", []) or []
    tail_risk = "st_j_unreliable" in flags or _st_j_has_tail_jump(feature)
    offset_risk = abs(delta_ms) >= _ST_J_REMEASURE_OFFSET_OUTLIER_MS
    beat_risk = "beat_unreliable" in flags
    if _st_j_is_stable(feature) and not offset_risk:
        return None, None
    if not (tail_risk or offset_risk or beat_risk):
        return None, None
    if abs(delta_ms) > _ST_J_REMEASURE_MAX_SHIFT_MS and not tail_risk:
        return None, None

    anchor_local = target_j - int(beat_start)
    if anchor_local < 0 or anchor_local >= len(sig):
        return None, None
    qrs_on = int(feature.qrs.onset)
    if target_j <= qrs_on:
        return None, None
    if feature.t.peak is not None and target_j + int(round(0.080 * fs)) >= int(feature.t.peak):
        return None, None
    if next_qrs_guard is not None and target_j + int(round(0.080 * fs)) >= int(next_qrs_guard):
        return None, None

    st_on, _st_mid, _st_80, _slope, _morphology = _st_j_sample_values(
        sig,
        anchor=anchor_local,
        baseline=0.0,
        fs=fs,
    )
    if st_on is None:
        return None, None
    if tail_risk:
        return target_j, "tail_guard"
    if offset_risk:
        return target_j, "offset_outlier"
    if beat_risk:
        return target_j, "beat_unreliable"
    return None, None


def _apply_st_j_remeasurement(
    feature: "LeadBeatFeatures",
    *,
    anchor_global: int,
    sig: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
    reason: str,
    target: STJRepairTarget,
    confidence: float,
) -> None:
    old_j = int(feature.j_index if feature.j_index is not None else feature.qrs.offset)
    anchor_local = int(anchor_global) - int(beat_start)
    st_on, st_mid, st_80, slope, morphology = _st_j_sample_values(
        sig,
        anchor=anchor_local,
        baseline=baseline,
        fs=fs,
    )
    if st_on is None:
        return

    feature.st_on_mv = st_on
    feature.st_mid_mv = st_mid
    feature.st_80ms_mv = st_80
    feature.st_slope_mv_per_ms = slope
    feature.st_morphology = morphology
    feature.st_j_original_index = old_j
    feature.st_j_remeasured_index = int(anchor_global)
    feature.st_j_remeasure_delta_ms = (int(anchor_global) - old_j) * 1000.0 / fs
    feature.st_j_remeasure_reason = reason
    feature.st_j_remeasure_confidence = float(confidence)
    feature.st_j_remeasure_consensus_index = int(target.j_index)
    feature.st_j_remeasure_support = int(target.support)
    feature.st_j_remeasure_used_leads = target.used_leads
    feature.st_j_remeasure_excluded_leads = target.excluded_leads
    if "st_j_remeasured_by_consensus" not in feature.flags:
        feature.flags.append("st_j_remeasured_by_consensus")
    reason_flag = f"st_j_{reason}_remeasured"
    if reason_flag not in feature.flags:
        feature.flags.append(reason_flag)
    if "st_j_unreliable" in feature.flags:
        feature.flags = [flag for flag in feature.flags if flag != "st_j_unreliable"]


# ─────────────────────────────────────────────────────────────────────────────
# T-end consensus outlier repair (runs before _apply_multilead_consensus)
# ─────────────────────────────────────────────────────────────────────────────
_T_END_REPAIR_MIN_SUPPORT = 4
_T_END_REPAIR_LOW_CONF = 0.35
_T_END_REPAIR_MARGIN_MS = 30.0
_T_END_REPAIR_QT_FLOOR_MS = 280.0
_T_END_REPAIR_HIGH_CONF = 0.70
_T_END_REPAIR_TARGET_MIN_CONF = 0.20
_T_END_REPAIR_OUTLIER_MS = 35.0
_T_END_REPAIR_TARGET_MAD_K = 3.5
_T_END_LATE_TAIL_MIN_SUPPORT = 4
_T_END_LATE_TAIL_MIN_EXTENSION_MS = 30.0
_T_END_LATE_TAIL_MAX_EXTENSION_MS = 170.0
_T_END_LATE_TAIL_CLUSTER_MS = 30.0
_T_END_LATE_TAIL_SHORT_TPE_MS = 70.0
_T_END_LATE_TAIL_STRONG_MIN_SUPPORT = 3
_T_END_LATE_TAIL_STRONG_MIN_AMP_MV = 0.05
_T_END_LATE_TAIL_STRONG_MIN_CONF = 0.50
_T_END_LATE_TAIL_MAX_METHOD_SPREAD_MS = 45.0
_T_DUAL_RESCUE_MIN_SUPPORT = 4
_T_DUAL_RESCUE_LOW_CONF = 0.35
_T_DUAL_RESCUE_HIGH_CONF = 0.70
_T_DUAL_RESCUE_OUTLIER_MS = 35.0
_T_DUAL_RESCUE_MAD_K = 3.5
_T_DUAL_RESCUE_METHOD_DISAGREE_MS = 30.0
_T_DUAL_RESCUE_MAX_EXTENSION_MS = 160.0
_T_DUAL_RESCUE_STABLE_AFTER_MS = 20.0
_T_DUAL_RESCUE_QT_FLOOR_MS = 280.0
_T_DUAL_RESCUE_FLAT_T_AMP_MV = 0.05


def _t_offset_method_spread_ms(
    values: List[Optional[int]],
    fs: int,
) -> Optional[float]:
    finite = [int(value) for value in values if value is not None]
    if len(finite) < 2 or fs <= 0:
        return None
    return (max(finite) - min(finite)) * 1000.0 / fs


def _t_offset_slope_return(
    sig: np.ndarray,
    *,
    t_peak: int,
    search_end: int,
    baseline: float,
    fs: int,
) -> Optional[int]:
    if fs <= 0 or sig is None or len(sig) == 0:
        return None
    peak = int(t_peak)
    hi = min(len(sig) - 1, int(search_end))
    if hi <= peak:
        return None
    tail = sig[max(0, peak): hi + 1].astype(float) - float(baseline)
    if tail.size < max(4, int(0.040 * fs)):
        return None
    win = max(3, int(round(0.016 * fs)))
    smooth = np.convolve(tail, np.ones(win) / win, mode="same")
    slopes = np.gradient(smooth)
    amp = float(np.max(np.abs(smooth)))
    if amp < _T_DUAL_RESCUE_FLAT_T_AMP_MV:
        return None
    amp_thr = max(0.008, 0.08 * amp)
    slope_thr = max(0.0008, 0.12 * float(np.max(np.abs(slopes))))
    stable = max(2, int(round(_T_DUAL_RESCUE_STABLE_AFTER_MS * fs / 1000.0)))
    start = max(1, int(round(0.040 * fs)))
    for local in range(start, len(smooth) - stable):
        amp_close = np.all(np.abs(smooth[local: local + stable]) <= amp_thr)
        slope_close = np.all(np.abs(slopes[local: local + stable]) <= slope_thr)
        if amp_close and slope_close:
            return peak + int(local)
    return None


def _t_offset_tangent(
    sig: np.ndarray,
    *,
    t_peak: int,
    search_end: int,
    baseline: float,
    fs: int,
) -> Optional[int]:
    if fs <= 0 or sig is None or len(sig) == 0:
        return None
    peak = int(t_peak)
    hi = min(len(sig) - 1, int(search_end))
    if hi <= peak + max(3, int(0.020 * fs)):
        return None
    arm = sig[peak: hi + 1].astype(float) - float(baseline)
    polarity = 1.0 if float(arm[0]) >= 0.0 else -1.0
    amp = float(np.max(np.abs(arm)))
    if amp < _T_DUAL_RESCUE_FLAT_T_AMP_MV:
        return None
    win = max(3, int(round(0.016 * fs)))
    smooth = np.convolve(arm, np.ones(win) / win, mode="same")
    slopes = np.gradient(smooth)
    start = max(1, int(round(0.030 * fs)))
    if start >= len(slopes) - 1:
        start = 1
    terminal = slopes[start:]
    if terminal.size == 0:
        return None
    steep = start + int(np.argmin(terminal) if polarity > 0 else np.argmax(terminal))
    slope = float(slopes[steep])
    if abs(slope) < 1e-9:
        return None
    x_cross = steep - float(smooth[steep]) / slope
    if not np.isfinite(x_cross):
        return None
    local = int(round(np.clip(x_cross, steep, len(arm) - 1)))
    return peak + local


def _t_offset_dual_evidence(
    sig: np.ndarray,
    *,
    t_peak: int,
    search_end: int,
    baseline: float,
    fs: int,
) -> TDualEndpointEvidence:
    chord, chord_conf, chord_method = _t_end_geometric(sig, int(t_peak), int(search_end), baseline, fs)
    slope = _t_offset_slope_return(sig, t_peak=int(t_peak), search_end=int(search_end), baseline=baseline, fs=fs)
    tangent = _t_offset_tangent(sig, t_peak=int(t_peak), search_end=int(search_end), baseline=baseline, fs=fs)
    finite = [int(value) for value in (chord, slope, tangent) if value is not None]
    spread = _t_offset_method_spread_ms([chord, slope, tangent], fs)
    flat = False
    if 0 <= int(t_peak) < len(sig):
        flat = abs(float(sig[int(t_peak)] - baseline)) < _T_DUAL_RESCUE_FLAT_T_AMP_MV
    best = int(max(finite)) if finite else None
    agreement_bonus = 0.20 if spread is not None and spread <= _T_DUAL_RESCUE_METHOD_DISAGREE_MS else 0.0
    confidence = float(np.clip(max(float(chord_conf or 0.0), 0.35 if slope is not None else 0.0, 0.35 if tangent is not None else 0.0) + agreement_bonus, 0.0, 1.0))
    return TDualEndpointEvidence(
        chord_index=None if chord is None else int(chord),
        slope_index=None if slope is None else int(slope),
        tangent_index=None if tangent is None else int(tangent),
        best_index=best,
        method_spread_ms=spread,
        confidence=confidence,
        reason=str(chord_method or "dual_method"),
        flat_t_wave=flat,
    )


def _t_dual_is_risky(feature: "LeadBeatFeatures") -> bool:
    flags = set(getattr(feature, "flags", []) or [])
    method = str(getattr(feature, "t_end_method", "") or "")
    return (
        float(getattr(feature, "qt_confidence", 0.0) or 0.0) < _T_DUAL_RESCUE_LOW_CONF
        or "threshold" in method
        or "fallback" in method
        or "discarded_short_qt" in method
        or bool(flags & {"t_end_fallback", "st_t_confusion", "beat_unreliable", "t_unreliable"})
    )


def _t_dual_rescue_target(
    items: List["LeadBeatFeatures"],
    *,
    fs: int,
    quality: Dict[str, object],
) -> Optional[TDualRescueTarget]:
    if fs <= 0:
        return None
    pool: List[Tuple["LeadBeatFeatures", int]] = []
    excluded: List[Tuple["LeadBeatFeatures", int]] = []
    for feature in items:
        if feature.t.offset is None or feature.t.peak is None:
            continue
        if feature.qrs.onset is None or feature.qrs.offset is None:
            continue
        offset = int(feature.t.offset)
        if not getattr(quality.get(feature.lead), "reliable_for_qt", False):
            excluded.append((feature, offset))
            continue
        if float(getattr(feature, "qt_confidence", 0.0) or 0.0) < _T_DUAL_RESCUE_LOW_CONF:
            excluded.append((feature, offset))
            continue
        if _t_dual_is_risky(feature):
            excluded.append((feature, offset))
            continue
        pool.append((feature, offset))
    if len(pool) < _T_DUAL_RESCUE_MIN_SUPPORT:
        return None

    offsets = np.asarray([offset for _feature, offset in pool], dtype=float)
    center = float(np.median(offsets))
    mad = float(np.median(np.abs(offsets - center)))
    tol = max(_T_DUAL_RESCUE_OUTLIER_MS * fs / 1000.0, _T_DUAL_RESCUE_MAD_K * mad)
    early: List[Tuple["LeadBeatFeatures", int]] = []
    pool_ids = {id(feature) for feature, _offset in pool}
    for feature in items:
        if id(feature) in pool_ids:
            continue
        if feature.t.offset is None:
            continue
        offset = int(feature.t.offset)
        if float(offset) < center - tol and _t_dual_is_risky(feature):
            early.append((feature, offset))
    if not early:
        return None

    target_offset = int(np.floor(center + 0.5))
    return TDualRescueTarget(
        offset=target_offset,
        support=len(pool),
        used_leads=_t_end_csv(pool),
        excluded_leads=_t_end_csv(excluded),
        early_outlier_leads=_t_end_csv(early),
    )


def _t_dual_rescue_candidate(
    feature: "LeadBeatFeatures",
    *,
    sig: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
    target: TDualRescueTarget,
    next_qrs_guard: Optional[int],
) -> Tuple[Optional[int], Optional[str], TDualEndpointEvidence]:
    t_peak = feature.t.peak
    old_off = feature.t.offset
    if fs <= 0 or t_peak is None:
        evidence = TDualEndpointEvidence(None, None, None, None, None, 0.0, "missing_peak")
        return None, None, evidence
    if feature.qrs.onset is None or feature.qrs.offset is None:
        evidence = TDualEndpointEvidence(None, None, None, None, None, 0.0, "missing_qrs")
        return None, None, evidence

    peak_local = int(t_peak) - int(beat_start)
    if peak_local < 0 or peak_local >= len(sig):
        evidence = TDualEndpointEvidence(None, None, None, None, None, 0.0, "peak_out_of_window")
        return None, None, evidence

    target_off = int(target.offset)
    old_off_i = int(old_off) if old_off is not None else int(t_peak)
    target_local = int(target_off) - int(beat_start)
    margin = int(round(0.040 * fs))
    search_end = min(len(sig) - 1, max(target_local, old_off_i - int(beat_start)) + margin)
    if next_qrs_guard is not None:
        search_end = min(search_end, int(next_qrs_guard) - int(beat_start))
    evidence = _t_offset_dual_evidence(
        sig,
        t_peak=peak_local,
        search_end=search_end,
        baseline=baseline,
        fs=fs,
    )
    if evidence.flat_t_wave:
        return None, None, evidence
    if (
        evidence.method_spread_ms is not None
        and evidence.method_spread_ms > _T_DUAL_RESCUE_METHOD_DISAGREE_MS
    ):
        return None, None, evidence
    if old_off is not None and int(old_off) >= target_off - int(round(_T_DUAL_RESCUE_OUTLIER_MS * fs / 1000.0)):
        return None, None, evidence
    if feature.lead not in _t_end_lead_set(target.early_outlier_leads):
        return None, None, evidence
    if float(getattr(feature, "qt_confidence", 0.0) or 0.0) >= _T_DUAL_RESCUE_HIGH_CONF and not _t_dual_is_risky(feature):
        return None, None, evidence
    if evidence.best_index is None:
        return None, None, evidence

    candidate_global = int(evidence.best_index) + int(beat_start)
    if old_off is not None and candidate_global <= int(old_off):
        return None, None, evidence
    if old_off is not None and (candidate_global - int(old_off)) * 1000.0 / fs > _T_DUAL_RESCUE_MAX_EXTENSION_MS:
        return None, None, evidence
    if (candidate_global - int(feature.qrs.onset)) * 1000.0 / fs < _T_DUAL_RESCUE_QT_FLOOR_MS:
        return None, None, evidence
    if next_qrs_guard is not None and candidate_global >= int(next_qrs_guard):
        return None, None, evidence
    if candidate_global < int(target.offset) - int(round(_T_DUAL_RESCUE_OUTLIER_MS * fs / 1000.0)):
        return None, None, evidence
    return candidate_global, "early_truncation_dual_method", evidence


def _apply_t_offset_dual_rescue(
    bf: "LeadBeatFeatures",
    *,
    new_off_global: int,
    sig_beat: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
    reason: str,
    target: TDualRescueTarget,
    evidence: TDualEndpointEvidence,
) -> None:
    old_off = bf.t.offset
    bf.t.offset = int(new_off_global)
    qrs_on = bf.qrs.onset
    qrs_off = bf.qrs.offset
    t_on = bf.t.onset
    t_peak = bf.t.peak
    bf.qt_ms = (new_off_global - int(qrs_on)) * 1000.0 / fs if qrs_on is not None else None
    bf.jt_ms = (new_off_global - int(qrs_off)) * 1000.0 / fs if qrs_off is not None else None
    bf.tpe_ms = (
        (new_off_global - int(t_peak)) * 1000.0 / fs
        if t_peak is not None and new_off_global > int(t_peak) else None
    )
    if t_on is not None and new_off_global > int(t_on):
        lo = max(0, int(t_on) - int(beat_start))
        hi = min(len(sig_beat) - 1, int(new_off_global) - int(beat_start))
        if hi > lo:
            seg = sig_beat[lo: hi + 1] - baseline
            trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            bf.t_area = float(trapz(np.abs(seg)))
            bf.t_signed_area = float(trapz(seg))
            bf.t_dur_ms = (int(new_off_global) - int(t_on)) * 1000.0 / fs

    if old_off is not None:
        bf.t_offset_dual_original_index = int(old_off)
        bf.t_offset_dual_delta_ms = (int(new_off_global) - int(old_off)) * 1000.0 / fs
    bf.t_offset_dual_rescued_index = int(new_off_global)
    bf.t_offset_dual_reason = reason
    bf.t_offset_dual_confidence = float(evidence.confidence)
    bf.t_offset_dual_consensus_index = int(target.offset)
    bf.t_offset_dual_support = int(target.support)
    bf.t_offset_dual_used_leads = target.used_leads
    bf.t_offset_dual_excluded_leads = target.excluded_leads
    bf.t_offset_dual_chord_index = evidence.chord_index
    bf.t_offset_dual_slope_index = evidence.slope_index
    bf.t_offset_dual_tangent_index = evidence.tangent_index
    bf.t_offset_dual_method_spread_ms = evidence.method_spread_ms

    for flag in ("t_offset_dual_method_rescued", "t_offset_early_truncation_dual_rescued"):
        if flag not in bf.flags:
            bf.flags.append(flag)
    if evidence.method_spread_ms is not None and evidence.method_spread_ms >= _T_DUAL_RESCUE_METHOD_DISAGREE_MS:
        if "t_offset_method_disagreement" not in bf.flags:
            bf.flags.append("t_offset_method_disagreement")
    if evidence.possible_t_u_fusion and "possible_t_u_fusion" not in bf.flags:
        bf.flags.append("possible_t_u_fusion")
    if evidence.flat_t_wave and "flat_t_wave" not in bf.flags:
        bf.flags.append("flat_t_wave")
    if not str(bf.t_end_method).endswith("_dual_rescued"):
        bf.t_end_method = f"{bf.t_end_method}_dual_rescued"


def _t_end_csv(items: List[Tuple["LeadBeatFeatures", int]]) -> str:
    return ",".join(feature.lead for feature, _offset in items)


def _t_end_is_risky(feature: "LeadBeatFeatures") -> bool:
    flags = getattr(feature, "flags", []) or []
    return (
        bool(getattr(feature, "st_t_confusion", False))
        or "st_t_confusion" in flags
        or "t_end_fallback" in flags
        or "beat_unreliable" in flags
    )


def _t_end_repair_target(
    items: List["LeadBeatFeatures"],
    *,
    fs: int,
    quality: Dict[str, object],
) -> Optional[TEndRepairTarget]:
    if fs <= 0:
        return None

    diagnostic_candidates: List[Tuple["LeadBeatFeatures", int]] = []
    measurement_pool: List[Tuple["LeadBeatFeatures", int]] = []
    for feature in items:
        if not getattr(quality.get(feature.lead), "reliable_for_qt", False):
            continue
        if feature.t.offset is None or feature.t.peak is None:
            continue
        if feature.qrs.onset is None or feature.qrs.offset is None:
            continue
        offset = int(feature.t.offset)
        diagnostic_candidates.append((feature, offset))
        if float(getattr(feature, "qt_confidence", 0.0) or 0.0) > _T_END_REPAIR_TARGET_MIN_CONF:
            measurement_pool.append((feature, offset))
    if len(measurement_pool) < _T_END_REPAIR_MIN_SUPPORT:
        return None

    core = [
        (feature, offset)
        for feature, offset in measurement_pool
        if not _t_end_is_risky(feature)
        and float(getattr(feature, "qt_confidence", 0.0) or 0.0) >= _T_END_REPAIR_LOW_CONF
    ]
    if len(core) < _T_END_REPAIR_MIN_SUPPORT:
        core = measurement_pool[:]
    if len(core) < _T_END_REPAIR_MIN_SUPPORT:
        return None

    core_offsets = np.asarray([offset for _feature, offset in core], dtype=float)
    center = float(np.median(core_offsets))
    mad = float(np.median(np.abs(core_offsets - center)))
    tol = max(
        _T_END_REPAIR_OUTLIER_MS * fs / 1000.0,
        _T_END_REPAIR_TARGET_MAD_K * mad,
    )

    used: List[Tuple["LeadBeatFeatures", int]] = []
    excluded: List[Tuple["LeadBeatFeatures", int]] = []
    early: List[Tuple["LeadBeatFeatures", int]] = []
    late: List[Tuple["LeadBeatFeatures", int]] = []
    core_ids = {id(feature) for feature, _offset in core}
    for feature, offset in diagnostic_candidates:
        delta = float(offset) - center
        low_conf_or_risky = (
            float(getattr(feature, "qt_confidence", 0.0) or 0.0) < _T_END_REPAIR_LOW_CONF
            or _t_end_is_risky(feature)
        )
        if abs(delta) > tol and low_conf_or_risky:
            excluded.append((feature, offset))
            if delta < 0.0:
                early.append((feature, offset))
            else:
                late.append((feature, offset))
        elif id(feature) in core_ids and not _t_end_is_risky(feature):
            used.append((feature, offset))
        else:
            excluded.append((feature, offset))

    if len(used) < _T_END_REPAIR_MIN_SUPPORT:
        return None
    target_offset = int(np.floor(float(np.median([offset for _feature, offset in used])) + 0.5))
    return TEndRepairTarget(
        offset=target_offset,
        support=len(used),
        used_leads=_t_end_csv(used),
        excluded_leads=_t_end_csv(excluded),
        early_outlier_leads=_t_end_csv(early),
        late_outlier_leads=_t_end_csv(late),
    )


def _t_end_lead_set(csv: str) -> set[str]:
    return {lead for lead in str(csv or "").split(",") if lead}


def _t_end_local_tail_present(
    sig: np.ndarray,
    *,
    current_off: int,
    target_off: int,
    baseline: float,
    t_peak: int,
) -> bool:
    lo = max(0, min(int(current_off), int(target_off)))
    hi = min(len(sig) - 1, max(int(current_off), int(target_off)))
    if hi <= lo:
        return False
    peak_amp = abs(float(sig[int(t_peak)] - baseline)) if 0 <= int(t_peak) < len(sig) else 0.0
    threshold = max(0.015, 0.08 * peak_amp)
    return bool(np.max(np.abs(sig[lo: hi + 1] - baseline)) >= threshold)


def _t_end_late_tail_candidate(
    feature: "LeadBeatFeatures",
    *,
    sig: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
    next_qrs_guard: Optional[int],
) -> Optional[int]:
    if fs <= 0 or feature.t.peak is None or feature.t.offset is None:
        return None
    if feature.qrs.onset is None or feature.qrs.offset is None:
        return None

    old_off = int(feature.t.offset)
    t_peak = int(feature.t.peak)
    if old_off <= t_peak:
        return None

    peak_local = t_peak - int(beat_start)
    old_local = old_off - int(beat_start)
    if not (0 <= peak_local < len(sig)) or not (0 <= old_local < len(sig)):
        return None

    max_extension = int(round(_T_END_LATE_TAIL_MAX_EXTENSION_MS * fs / 1000.0))
    min_extension = int(round(_T_END_LATE_TAIL_MIN_EXTENSION_MS * fs / 1000.0))
    search_end = min(len(sig) - 1, old_local + max_extension)
    if next_qrs_guard is not None:
        search_end = min(search_end, int(next_qrs_guard) - int(beat_start))
    if search_end <= old_local + min_extension:
        return None

    if not _t_end_local_tail_present(
        sig,
        current_off=old_local,
        target_off=search_end,
        baseline=baseline,
        t_peak=peak_local,
    ):
        return None

    evidence = _t_offset_dual_evidence(
        sig,
        t_peak=peak_local,
        search_end=search_end,
        baseline=baseline,
        fs=fs,
    )
    if evidence.flat_t_wave or evidence.best_index is None:
        return None
    if (
        evidence.method_spread_ms is not None
        and evidence.method_spread_ms > _T_END_LATE_TAIL_MAX_METHOD_SPREAD_MS
    ):
        return None

    candidate_global = int(evidence.best_index) + int(beat_start)
    if candidate_global <= old_off + min_extension:
        return None
    if candidate_global - old_off > max_extension:
        return None
    if (candidate_global - int(feature.qrs.onset)) * 1000.0 / fs < _T_END_REPAIR_QT_FLOOR_MS:
        return None
    if next_qrs_guard is not None and candidate_global >= int(next_qrs_guard):
        return None
    return candidate_global


def _t_end_late_tail_confirmation_target(
    items: List["LeadBeatFeatures"],
    candidates: List[Tuple["LeadBeatFeatures", int]],
    *,
    fs: int,
) -> Optional[TEndRepairTarget]:
    if fs <= 0 or len(candidates) < _T_END_LATE_TAIL_STRONG_MIN_SUPPORT:
        return None
    candidate_offsets = np.asarray([offset for _feature, offset in candidates], dtype=float)
    center = float(np.median(candidate_offsets))
    cluster_tol = _T_END_LATE_TAIL_CLUSTER_MS * fs / 1000.0
    cluster = [
        (feature, int(offset))
        for feature, offset in candidates
        if abs(float(offset) - center) <= cluster_tol
    ]
    if len(cluster) < _T_END_LATE_TAIL_STRONG_MIN_SUPPORT:
        return None
    if len(cluster) < _T_END_LATE_TAIL_MIN_SUPPORT:
        strong = []
        for feature, offset in cluster:
            if feature.t.peak is None or feature.t.offset is None:
                continue
            tpe_ms = (int(feature.t.offset) - int(feature.t.peak)) * 1000.0 / fs
            extension_ms = (int(offset) - int(feature.t.offset)) * 1000.0 / fs
            t_amp = getattr(feature, "t_amp_mv", None)
            qt_conf = float(getattr(feature, "qt_confidence", 0.0) or 0.0)
            if (
                tpe_ms <= _T_END_LATE_TAIL_SHORT_TPE_MS
                and extension_ms >= _T_END_LATE_TAIL_MIN_EXTENSION_MS
                and t_amp is not None
                and abs(float(t_amp)) >= _T_END_LATE_TAIL_STRONG_MIN_AMP_MV
                and qt_conf >= _T_END_LATE_TAIL_STRONG_MIN_CONF
            ):
                strong.append((feature, offset))
        if len(strong) < _T_END_LATE_TAIL_STRONG_MIN_SUPPORT:
            return None
        cluster = strong

    current_offsets = [
        int(feature.t.offset)
        for feature in items
        if feature.t.offset is not None
    ]
    if not current_offsets:
        return None
    current_center = float(np.median(np.asarray(current_offsets, dtype=float)))
    target_offset = int(np.floor(float(np.median([offset for _feature, offset in cluster])) + 0.5))
    if (target_offset - current_center) * 1000.0 / fs < _T_END_LATE_TAIL_MIN_EXTENSION_MS:
        return None

    cluster_ids = {id(feature) for feature, _offset in cluster}
    excluded = [
        (feature, int(feature.t.offset))
        for feature in items
        if feature.t.offset is not None and id(feature) not in cluster_ids
    ]
    return TEndRepairTarget(
        offset=target_offset,
        support=len(cluster),
        used_leads=_t_end_csv(cluster),
        excluded_leads=_t_end_csv(excluded),
        early_outlier_leads=_t_end_csv(cluster),
        late_outlier_leads="",
    )


def _t_end_repair_candidate(
    feature: "LeadBeatFeatures",
    *,
    sig: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
    target: TEndRepairTarget,
    next_qrs_guard: Optional[int],
) -> Tuple[Optional[int], Optional[str]]:
    if fs <= 0:
        return None, None
    if feature.t.peak is None or feature.t.offset is None:
        return None, None
    if feature.qrs.onset is None or feature.qrs.offset is None:
        return None, None

    old_off = int(feature.t.offset)
    target_off = int(target.offset)
    delta_ms = (old_off - target_off) * 1000.0 / fs
    if abs(delta_ms) < _T_END_REPAIR_OUTLIER_MS:
        return None, None

    risky = _t_end_is_risky(feature)
    if float(getattr(feature, "qt_confidence", 0.0) or 0.0) >= _T_END_REPAIR_HIGH_CONF and not risky:
        return None, None

    early_leads = _t_end_lead_set(getattr(target, "early_outlier_leads", ""))
    late_leads = _t_end_lead_set(getattr(target, "late_outlier_leads", ""))
    if delta_ms < 0.0:
        if feature.lead not in early_leads and not risky:
            return None, None
        reason = "early_truncation"
    else:
        if feature.lead not in late_leads and not risky:
            return None, None
        reason = "late_outlier"

    t_peak_local = int(feature.t.peak) - int(beat_start)
    old_off_local = int(feature.t.offset) - int(beat_start)
    target_local = int(target.offset) - int(beat_start)
    if not (0 <= t_peak_local < len(sig)):
        return None, None
    if not (0 <= old_off_local < len(sig)):
        return None, None

    margin = int(round(_T_END_REPAIR_MARGIN_MS * fs / 1000.0))
    # Bound the geometric re-search to the consensus target, not the outlier
    # itself. For a late_outlier, old_off_local sits beyond target_local; using
    # max() here (as if extending toward whichever endpoint is furthest) lets
    # the re-search window still cover the very late artifact that produced
    # the outlier, so it just re-finds the same wrong feature.
    search_end = min(len(sig) - 1, target_local + margin)
    if next_qrs_guard is not None:
        search_end = min(search_end, int(next_qrs_guard) - int(beat_start))
    if search_end <= t_peak_local:
        return None, None

    if reason == "early_truncation" and not _t_end_local_tail_present(
        sig,
        current_off=old_off_local,
        target_off=min(search_end, target_local),
        baseline=baseline,
        t_peak=t_peak_local,
    ):
        return None, None

    geom_off, _geom_conf, _geom_method = _t_end_geometric(
        sig,
        t_peak_local,
        search_end,
        baseline,
        fs,
    )
    if geom_off is None:
        candidate_local = min(max(target_local, t_peak_local + 1), search_end)
    else:
        candidate_local = int(geom_off)

    candidate_global = int(candidate_local) + int(beat_start)
    if candidate_global <= int(feature.t.peak):
        return None, None
    qrs_on = int(feature.qrs.onset)
    if (candidate_global - qrs_on) * 1000.0 / fs < _T_END_REPAIR_QT_FLOOR_MS:
        return None, None
    if next_qrs_guard is not None and candidate_global >= int(next_qrs_guard):
        return None, None
    return candidate_global, reason


def _apply_t_end_repair(
    bf: "LeadBeatFeatures",
    new_off_global: int,
    sig_beat: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
    *,
    reason: str = "consensus_outlier",
    target: Optional[TEndRepairTarget] = None,
    confidence: float = 0.80,
) -> None:
    """Write a repaired T-end back and recompute all T-end-derived fields."""
    old_off = bf.t.offset
    bf.t.offset = int(new_off_global)
    qrs_on = bf.qrs.onset
    qrs_off = bf.qrs.offset
    t_on = bf.t.onset
    t_peak = bf.t.peak
    bf.qt_ms = (
        (new_off_global - int(qrs_on)) * 1000.0 / fs if qrs_on is not None else None
    )
    bf.jt_ms = (
        (new_off_global - int(qrs_off)) * 1000.0 / fs if qrs_off is not None else None
    )
    bf.tpe_ms = (
        (new_off_global - int(t_peak)) * 1000.0 / fs
        if (t_peak is not None and new_off_global > int(t_peak)) else None
    )
    if t_on is not None and new_off_global > int(t_on):
        lo = max(0, int(t_on) - beat_start)
        hi = min(len(sig_beat) - 1, new_off_global - beat_start)
        if hi > lo:
            seg = sig_beat[lo : hi + 1] - baseline
            trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            bf.t_area = float(trapz(np.abs(seg)))
            bf.t_signed_area = float(trapz(seg))
            bf.t_dur_ms = (new_off_global - int(t_on)) * 1000.0 / fs

    if old_off is not None:
        bf.t_offset_original_index = int(old_off)
        bf.t_offset_repaired_index = int(new_off_global)
        bf.t_offset_repair_delta_ms = (int(new_off_global) - int(old_off)) * 1000.0 / fs
    bf.t_offset_repair_reason = reason
    bf.t_offset_repair_confidence = float(confidence)
    if target is not None:
        bf.t_offset_repair_consensus_index = int(target.offset)
        bf.t_offset_repair_support = int(target.support)
        bf.t_offset_repair_used_leads = target.used_leads
        bf.t_offset_repair_excluded_leads = target.excluded_leads

    if "t_end_consensus_repaired" not in bf.flags:
        bf.flags.append("t_end_consensus_repaired")
    if "t_offset_repaired_by_consensus" not in bf.flags:
        bf.flags.append("t_offset_repaired_by_consensus")
    reason_flag = f"t_offset_{reason}_repaired"
    if reason_flag not in bf.flags:
        bf.flags.append(reason_flag)
    bf.t_end_repair_reason = "consensus_outlier"
    if not str(bf.t_end_method).endswith("_consensus_repaired"):
        bf.t_end_method = f"{bf.t_end_method}_consensus_repaired"


def _apply_st_j_raw_remeasurement(
    beat_features: List["LeadBeatFeatures"],
    *,
    ecg: np.ndarray,
    fs: int,
    quality: Dict[str, object],
    r_locs: np.ndarray,
) -> List["LeadBeatFeatures"]:
    if (
        not beat_features
        or ecg is None
        or fs <= 0
        or r_locs is None
        or len(r_locs) == 0
    ):
        return beat_features

    n = ecg.shape[1]
    lead_index = _repair_lead_indices(beat_features, ecg)
    by_beat: Dict[int, List["LeadBeatFeatures"]] = {}
    for feature in beat_features:
        by_beat.setdefault(feature.beat_id, []).append(feature)

    for beat_id, items in by_beat.items():
        target = _st_j_remeasurement_target(items, fs=fs, quality=quality)
        if target is None:
            continue
        if beat_id < 0 or beat_id >= len(r_locs):
            continue

        r = int(r_locs[beat_id])
        beat_start = max(0, int(r - 0.35 * fs))
        beat_end = min(n, int(r + 0.55 * fs))
        if beat_end - beat_start < 4:
            continue
        local_r = r - beat_start
        next_qrs_guard = None
        if beat_id + 1 < len(r_locs):
            next_qrs_guard = int(r_locs[beat_id + 1] - 0.06 * fs)

        for feature in items:
            li = lead_index.get(feature.lead)
            if li is None or li >= ecg.shape[0]:
                continue
            sig_beat = ecg[li, beat_start:beat_end].astype(float)
            if sig_beat.size < 4:
                continue
            baseline, _, _ = _baseline_contract(sig_beat, local_r, fs)
            anchor, reason = _st_j_remeasurement_candidate(
                feature,
                sig=sig_beat,
                beat_start=beat_start,
                fs=fs,
                target=target,
                next_qrs_guard=next_qrs_guard,
            )
            if anchor is None or reason is None:
                continue
            _apply_st_j_remeasurement(
                feature,
                anchor_global=int(anchor),
                sig=sig_beat,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
                reason=reason,
                target=target,
                confidence=0.80,
            )

    return beat_features


def _apply_t_dual_method_raw_rescue(
    beat_features: List["LeadBeatFeatures"],
    *,
    ecg: np.ndarray,
    fs: int,
    quality: Dict[str, object],
    r_locs: np.ndarray,
) -> List["LeadBeatFeatures"]:
    if not beat_features or ecg is None or fs <= 0 or r_locs is None or len(r_locs) == 0:
        return beat_features

    n = ecg.shape[1]
    lead_index = _repair_lead_indices(beat_features, ecg)
    by_beat: Dict[int, List["LeadBeatFeatures"]] = {}
    for feature in beat_features:
        by_beat.setdefault(feature.beat_id, []).append(feature)

    for beat_id, items in by_beat.items():
        target = _t_dual_rescue_target(items, fs=fs, quality=quality)
        if target is None:
            continue
        if beat_id < 0 or beat_id >= len(r_locs):
            continue

        r = int(r_locs[beat_id])
        beat_start = max(0, int(r - 0.35 * fs))
        beat_end = min(n, int(r + 0.55 * fs))
        if beat_end - beat_start < 4:
            continue
        local_r = r - beat_start
        next_qrs_guard = None
        if beat_id + 1 < len(r_locs):
            next_qrs_guard = int(r_locs[beat_id + 1] - 0.06 * fs)

        for feature in items:
            li = lead_index.get(feature.lead)
            if li is None or li >= ecg.shape[0]:
                continue
            sig_beat = ecg[li, beat_start:beat_end].astype(float)
            if sig_beat.size < 4:
                continue
            baseline, _, _ = _baseline_contract(sig_beat, local_r, fs)
            rescued, reason, evidence = _t_dual_rescue_candidate(
                feature,
                sig=sig_beat,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
                target=target,
                next_qrs_guard=next_qrs_guard,
            )
            if rescued is None or reason is None:
                continue
            _apply_t_offset_dual_rescue(
                feature,
                new_off_global=int(rescued),
                sig_beat=sig_beat,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
                reason=reason,
                target=target,
                evidence=evidence,
            )
    return beat_features


def _repair_t_end_outliers(
    beat_features: List["LeadBeatFeatures"],
    ecg: np.ndarray,
    fs: int,
    quality: Dict[str, object],
    r_locs: np.ndarray,
) -> List["LeadBeatFeatures"]:
    """
    Repair gross low-confidence per-lead T-end outliers toward the robust
    multi-lead consensus, re-detecting on each outlier's own signal and
    recomputing T-end-derived fields.  High-confidence and consensus-consistent
    leads are left untouched.  Operates in global sample coordinates.
    """
    if (
        not beat_features
        or ecg is None
        or fs <= 0
        or r_locs is None
        or len(r_locs) == 0
    ):
        return beat_features

    n = ecg.shape[1]
    lead_index = _repair_lead_indices(beat_features, ecg)

    by_beat: Dict[int, List["LeadBeatFeatures"]] = {}
    for bf in beat_features:
        by_beat.setdefault(bf.beat_id, []).append(bf)

    for beat_id, items in by_beat.items():
        target = _t_end_repair_target(items, fs=fs, quality=quality)
        if target is None:
            continue
        if beat_id < 0 or beat_id >= len(r_locs):
            continue

        r = int(r_locs[beat_id])
        beat_start = max(0, int(r - 0.35 * fs))
        beat_end = min(n, int(r + 0.55 * fs))
        if beat_end - beat_start < 4:
            continue
        local_r = r - beat_start
        next_qrs_guard = None
        if beat_id + 1 < len(r_locs):
            next_qrs_guard = int(r_locs[beat_id + 1] - 0.06 * fs)

        for bf in items:
            li = lead_index.get(bf.lead)
            if li is None or li >= ecg.shape[0]:
                continue
            sig_beat = ecg[li, beat_start:beat_end].astype(float)
            if sig_beat.size < 4:
                continue
            baseline, _, _ = _baseline_contract(sig_beat, local_r, fs)
            repaired, reason = _t_end_repair_candidate(
                bf,
                sig=sig_beat,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
                target=target,
                next_qrs_guard=next_qrs_guard,
            )
            if repaired is None or reason is None or bf.t.offset is None:
                continue
            if int(repaired) == int(bf.t.offset):
                continue
            _apply_t_end_repair(
                bf,
                int(repaired),
                sig_beat,
                beat_start,
                baseline,
                fs,
                reason=reason,
                target=target,
                confidence=0.80,
            )

        late_tail_candidates: List[Tuple["LeadBeatFeatures", int]] = []
        late_tail_context: Dict[int, Tuple[np.ndarray, float]] = {}
        for bf in items:
            if not getattr(quality.get(bf.lead), "reliable_for_qt", False):
                continue
            if "t_offset_repaired_by_consensus" in getattr(bf, "flags", []):
                continue
            li = lead_index.get(bf.lead)
            if li is None or li >= ecg.shape[0]:
                continue
            sig_beat = ecg[li, beat_start:beat_end].astype(float)
            if sig_beat.size < 4:
                continue
            baseline, _, _ = _baseline_contract(sig_beat, local_r, fs)
            late_tail = _t_end_late_tail_candidate(
                bf,
                sig=sig_beat,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
                next_qrs_guard=next_qrs_guard,
            )
            if late_tail is None:
                continue
            late_tail_candidates.append((bf, int(late_tail)))
            late_tail_context[id(bf)] = (sig_beat, baseline)

        late_tail_target = _t_end_late_tail_confirmation_target(
            items,
            late_tail_candidates,
            fs=fs,
        )
        if late_tail_target is not None:
            used_late_tail_leads = _t_end_lead_set(late_tail_target.early_outlier_leads)
            for bf, _candidate in late_tail_candidates:
                if bf.lead not in used_late_tail_leads or bf.t.offset is None:
                    continue
                if int(late_tail_target.offset) <= int(bf.t.offset):
                    continue
                context = late_tail_context.get(id(bf))
                if context is None:
                    continue
                sig_beat, baseline = context
                _apply_t_end_repair(
                    bf,
                    int(late_tail_target.offset),
                    sig_beat,
                    beat_start,
                    baseline,
                    fs,
                    reason="systematic_early_tail",
                    target=late_tail_target,
                    confidence=0.75,
                )

    return beat_features


def _qrs_has_terminal_morphology(bf: "LeadBeatFeatures") -> bool:
    if int(getattr(bf, "qrs_notch_count", 0) or 0) > 0:
        return True
    if bool(getattr(bf, "qrs_slur_flag", False)):
        return True
    if int(getattr(bf, "qrs_num_peaks", 0) or 0) >= 3:
        return True
    for amp_attr in ("r_prime_amp_mv", "s_prime_amp_mv"):
        amp = getattr(bf, amp_attr, None)
        if amp is not None and np.isfinite(float(amp)) and abs(float(amp)) >= _QRS_TERMINAL_MORPH_AMP_MV:
            return True
    return False


def _qrs_offset_consensus_target(
    items: List["LeadBeatFeatures"],
    *,
    fs: int,
    quality: Dict[str, object],
    paced: bool,
) -> Optional[QRSOffsetRepairTarget]:
    if fs <= 0:
        return None
    candidates: List[Tuple["LeadBeatFeatures", int]] = []
    for bf in items:
        if not getattr(quality.get(bf.lead), "reliable_for_qrs", False):
            continue
        if float(getattr(bf, "qrs_confidence", 0.0) or 0.0) <= 0.10:
            continue
        if bf.qrs.onset is None or bf.qrs.offset is None:
            continue
        per_lead_ms = (int(bf.qrs.offset) - int(bf.qrs.onset)) * 1000.0 / fs
        if per_lead_ms > 250.0:
            continue
        candidates.append((bf, int(bf.qrs.offset)))
    if len(candidates) < _QRS_OFFSET_REPAIR_MIN_SUPPORT:
        return None

    offsets = [off for _bf, off in candidates]
    center = float(np.median(offsets))
    late_items = [
        (bf, off, _qrs_has_terminal_morphology(bf))
        for bf, off in candidates
        if (float(off) - center) * 1000.0 / float(fs) >= _QRS_TERMINAL_LATE_GAP_MS
    ]
    supported_late_count = sum(
        1
        for bf, _off, has_morphology in late_items
        if has_morphology or not bool(getattr(bf, "st_t_confusion", False))
    )
    late_cluster_supported = supported_late_count >= _QRS_TERMINAL_MIN_SUPPORTED_LATE

    measurement: List[Tuple["LeadBeatFeatures", int]] = []
    classification_only: List[Tuple["LeadBeatFeatures", int]] = []
    for bf, off in candidates:
        late_gap_ms = (float(off) - center) * 1000.0 / float(fs)
        is_late = late_gap_ms >= _QRS_TERMINAL_LATE_GAP_MS
        if paced or not is_late:
            measurement.append((bf, off))
            continue
        likely_st_t_tail = (
            bool(getattr(bf, "st_t_confusion", False))
            and not _qrs_has_terminal_morphology(bf)
            and float(getattr(bf, "qrs_off_confidence", 0.0) or 0.0) < _QRS_TERMINAL_LOW_OFF_CONF
        )
        if likely_st_t_tail or not late_cluster_supported:
            classification_only.append((bf, off))
        else:
            measurement.append((bf, off))

    if len(measurement) < _QRS_OFFSET_REPAIR_MIN_SUPPORT:
        return None
    target_offset = int(np.floor(float(np.median([off for _bf, off in measurement])) + 0.5))
    return QRSOffsetRepairTarget(
        offset=target_offset,
        support=len(measurement),
        used_leads=",".join(bf.lead for bf, _off in measurement),
        excluded_leads=",".join(bf.lead for bf, _off in classification_only),
    )


def _qrs_offset_local_candidate(
    sig: np.ndarray,
    *,
    qrs_on: int,
    qrs_peak: int,
    consensus_offset: int,
    baseline: float,
    fs: int,
) -> Optional[int]:
    margin = max(1, int(round(_QRS_OFFSET_REPAIR_SEARCH_MARGIN_MS * fs / 1000.0)))
    lo = max(int(qrs_peak) + 1, int(consensus_offset) - margin)
    hi = min(len(sig) - 1, int(consensus_offset) + margin)
    if hi <= lo:
        return None
    qrs_amp = max(abs(float(sig[int(qrs_peak)] - baseline)), 1e-6)
    tolerance = max(0.025, 0.08 * qrs_amp)
    segment = np.abs(sig[lo: hi + 1] - baseline)
    close = np.where(segment <= tolerance)[0]
    if close.size:
        close_indices = lo + close
        return int(close_indices[int(np.argmin(np.abs(close_indices - int(consensus_offset))))])
    return int(consensus_offset)


def _qrs_offset_repair_candidate(
    feature: "LeadBeatFeatures",
    *,
    sig: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
    target: QRSOffsetRepairTarget,
    paced: bool,
) -> Optional[int]:
    if paced or fs <= 0:
        return None
    if feature.qrs.onset is None or feature.qrs.peak is None or feature.qrs.offset is None:
        return None
    qrs_on_global = int(feature.qrs.onset)
    qrs_peak_global = int(feature.qrs.peak)
    qrs_off_global = int(feature.qrs.offset)
    delta_ms = (qrs_off_global - int(target.offset)) * 1000.0 / fs
    if abs(delta_ms) < _QRS_OFFSET_REPAIR_OUTLIER_MS:
        return None
    off_conf = float(getattr(feature, "qrs_off_confidence", 0.0) or 0.0)
    if off_conf >= _QRS_OFFSET_REPAIR_HIGH_CONF:
        return None
    excluded_leads = {
        lead for lead in str(getattr(target, "excluded_leads", "") or "").split(",") if lead
    }
    excluded_by_consensus = feature.lead in excluded_leads
    if (
        delta_ms > 0.0
        and _qrs_has_terminal_morphology(feature)
        and not (excluded_by_consensus and off_conf <= _QRS_OFFSET_REPAIR_LOW_CONF)
    ):
        return None

    qrs_on_local = qrs_on_global - int(beat_start)
    qrs_peak_local = qrs_peak_global - int(beat_start)
    if not (0 <= qrs_on_local < len(sig) and 0 <= qrs_peak_local < len(sig)):
        return None
    candidate_local = _qrs_offset_local_candidate(
        sig,
        qrs_on=qrs_on_local,
        qrs_peak=qrs_peak_local,
        consensus_offset=int(target.offset) - int(beat_start),
        baseline=baseline,
        fs=fs,
    )
    if candidate_local is None:
        return None
    candidate_global = int(candidate_local) + int(beat_start)
    if candidate_global <= qrs_on_global:
        return None
    qrs_ms = (candidate_global - qrs_on_global) * 1000.0 / fs
    original_qrs_ms = (qrs_off_global - qrs_on_global) * 1000.0 / fs
    has_tail_confusion_evidence = (
        bool(getattr(feature, "st_t_confusion", False))
        or "beat_unreliable" in (getattr(feature, "flags", []) or [])
        or "st_t_confusion" in (getattr(feature, "flags", []) or [])
    )
    if (
        candidate_global < qrs_off_global
        and _QRS_OFFSET_REPAIR_MIN_QRS_MS <= original_qrs_ms <= _QRS_OFFSET_REPAIR_PLAUSIBLE_SHORTEN_MAX_MS
        and qrs_ms < _QRS_OFFSET_REPAIR_MAX_SHORTEN_RATIO * original_qrs_ms
        and not has_tail_confusion_evidence
    ):
        return None
    if not (_QRS_OFFSET_REPAIR_MIN_QRS_MS <= qrs_ms <= _QRS_OFFSET_REPAIR_MAX_QRS_MS):
        return None
    return candidate_global


def _recompute_qrs_offset_dependent_fields(
    feature: "LeadBeatFeatures",
    *,
    sig: np.ndarray,
    beat_start: int,
    baseline: float,
    fs: int,
) -> None:
    if feature.qrs.onset is None or feature.qrs.peak is None or feature.qrs.offset is None:
        return
    qrs_on = int(feature.qrs.onset) - int(beat_start)
    qrs_peak = int(feature.qrs.peak) - int(beat_start)
    qrs_off = int(feature.qrs.offset) - int(beat_start)
    if not (0 <= qrs_on < qrs_peak < qrs_off < len(sig)):
        return

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    feature.qrs_ms = (qrs_off - qrs_on) * 1000.0 / fs
    feature.qrs_area = float(trapz(np.abs(sig[qrs_on: qrs_off + 1] - baseline)))
    feature.qrs_signed_area = float(trapz(sig[qrs_on: qrs_off + 1] - baseline))
    feature.j_index = qrs_off + int(beat_start)
    feature.st_on_mv, _st_source, _st_reliable = _measure_st_j_with_guard(
        sig=sig,
        qrs_on=qrs_on,
        qrs_off=qrs_off,
        baseline=baseline,
        fs=fs,
    )
    st_mid = min(len(sig) - 1, qrs_off + int(0.04 * fs))
    st_80 = min(len(sig) - 1, qrs_off + int(0.08 * fs))
    feature.st_mid_mv = float(sig[st_mid] - baseline)
    feature.st_80ms_mv = float(sig[st_80] - baseline)
    if feature.t.offset is not None:
        feature.qt_ms = (int(feature.t.offset) - int(feature.qrs.onset)) * 1000.0 / fs
        feature.jt_ms = (int(feature.t.offset) - int(feature.qrs.offset)) * 1000.0 / fs

    q_amp, r_amp, s_amp, r_peak_pos = _qrs_points(sig, qrs_on, qrs_peak, qrs_off, baseline)
    feature.q_amp_mv = q_amp
    feature.r_amp_mv = r_amp
    feature.s_amp_mv = s_amp
    feature.r_peak_index = None if r_peak_pos is None else r_peak_pos + int(beat_start)
    r_prime_amp, s_prime_amp, qrs_num_peaks, qrs_notch_count, durations = _qrs_components(
        sig,
        qrs_on,
        qrs_peak,
        qrs_off,
        baseline,
        fs,
    )
    feature.r_prime_amp_mv = r_prime_amp
    feature.s_prime_amp_mv = s_prime_amp
    feature.qrs_num_peaks = qrs_num_peaks
    feature.qrs_notch_count = qrs_notch_count
    feature.qrs_slur_flag = qrs_notch_count >= 2
    feature.r_duration_ms = durations["r_duration_ms"]
    feature.r_prime_duration_ms = durations["r_prime_duration_ms"]
    feature.s_duration_ms = durations["s_duration_ms"]
    feature.s_prime_duration_ms = durations["s_prime_duration_ms"]


def _apply_qrs_offset_raw_repair(
    beat_features: List["LeadBeatFeatures"],
    *,
    ecg: np.ndarray,
    fs: int,
    quality: Dict[str, object],
    r_locs: np.ndarray,
) -> List["LeadBeatFeatures"]:
    if not beat_features or ecg is None or fs <= 0 or r_locs is None:
        return beat_features
    lead_to_index = _repair_lead_indices(beat_features, ecg)
    by_beat: Dict[int, List["LeadBeatFeatures"]] = {}
    for feature in beat_features:
        by_beat.setdefault(int(feature.beat_id), []).append(feature)

    n = ecg.shape[1]
    for beat_id, items in by_beat.items():
        paced = any("paced_beat" in (getattr(feature, "flags", []) or []) for feature in items)
        target = _qrs_offset_consensus_target(items, fs=fs, quality=quality, paced=paced)
        if target is None:
            continue
        if beat_id < 0 or beat_id >= len(r_locs):
            continue
        r = int(r_locs[beat_id])
        beat_start = max(0, int(r - 0.35 * fs))
        beat_end = min(n, int(r + 0.55 * fs))
        if beat_end <= beat_start:
            continue
        local_r = r - beat_start
        for feature in items:
            li = lead_to_index.get(feature.lead)
            if li is None or li >= ecg.shape[0]:
                continue
            sig_beat = ecg[li, beat_start:beat_end].astype(float)
            if sig_beat.size < 4 or not (0 <= local_r < len(sig_beat)):
                continue
            baseline, _baseline_source, _baseline_confidence = _baseline_contract(sig_beat, local_r, fs)
            old_offset = feature.qrs.offset
            repaired = _qrs_offset_repair_candidate(
                feature,
                sig=sig_beat,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
                target=target,
                paced=paced,
            )
            if repaired is None or old_offset is None or int(repaired) == int(old_offset):
                continue
            feature.qrs_offset_original_index = int(old_offset)
            feature.qrs_offset_repaired_index = int(repaired)
            feature.qrs_offset_repair_delta_ms = (int(repaired) - int(old_offset)) * 1000.0 / fs
            feature.qrs_offset_repair_reason = "measurement_consensus_terminal_outlier"
            feature.qrs_offset_repair_confidence = 0.80
            feature.qrs_offset_repair_consensus_index = int(target.offset)
            feature.qrs_offset_repair_support = int(target.support)
            feature.qrs.offset = int(repaired)
            feature.qrs_off_confidence = max(float(feature.qrs_off_confidence or 0.0), 0.80)
            feature.qrs_terminal_confidence = feature.qrs_off_confidence
            if "qrs_offset_repaired_by_consensus" not in feature.flags:
                feature.flags.append("qrs_offset_repaired_by_consensus")
            _recompute_qrs_offset_dependent_fields(
                feature,
                sig=sig_beat,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
            )
    return beat_features


def _qrs_tail_settling_candidate(
    smoothed_signal: np.ndarray,
    *,
    r: int,
    qrs_on: int,
    qrs_off: int,
    fs: int,
) -> Optional[int]:
    """Find a delayed J point where a slow terminal QRS tail reaches its ST trend."""
    signal = np.asarray(smoothed_signal, dtype=float)
    if (
        fs <= 0
        or signal.ndim != 1
        or len(signal) < 4
        or not (0 <= qrs_on < r < qrs_off < len(signal))
    ):
        return None

    plateau_lo = min(
        len(signal) - 1,
        int(r + _QRS_TAIL_RESCUE_PLATEAU_START_MS * fs / 1000.0),
    )
    plateau_hi = min(
        len(signal),
        int(r + _QRS_TAIL_RESCUE_PLATEAU_END_MS * fs / 1000.0),
    )
    if plateau_hi - plateau_lo < max(6, int(round(0.020 * fs))):
        return None

    plateau_x = np.arange(plateau_lo, plateau_hi, dtype=float)
    plateau_y = signal[plateau_lo:plateau_hi]
    if not np.all(np.isfinite(plateau_y)):
        return None
    slope, intercept = np.polyfit(plateau_x, plateau_y, 1)

    amplitude_hi = min(
        len(signal),
        int(r + 0.060 * fs) + 1,
    )
    qrs_segment = signal[qrs_on:amplitude_hi]
    if len(qrs_segment) < 3:
        return None
    qrs_amplitude = max(float(np.ptp(qrs_segment)), 0.050)

    noise_lo = max(0, int(r - 0.300 * fs))
    noise_hi = max(noise_lo + 1, int(r - 0.200 * fs))
    noise_segment = signal[noise_lo:noise_hi]
    if len(noise_segment) < 3:
        return None
    noise_center = float(np.median(noise_segment))
    noise = max(
        1.4826 * float(np.median(np.abs(noise_segment - noise_center))),
        0.001,
    )

    indices = np.arange(len(signal), dtype=float)
    deviation = np.abs(signal - (slope * indices + intercept))
    current_min_deviation = max(0.020, 0.08 * qrs_amplitude, 5.0 * noise)
    if float(deviation[qrs_off]) < current_min_deviation:
        return None

    search_lo = max(
        qrs_off,
        int(r + _QRS_TAIL_RESCUE_SEARCH_START_MS * fs / 1000.0),
    )
    search_hi = min(
        len(signal) - 1,
        int(r + _QRS_TAIL_RESCUE_SEARCH_END_MS * fs / 1000.0),
    )
    sustain = max(3, int(round(_QRS_TAIL_RESCUE_SETTLE_MS * fs / 1000.0)))
    tolerance = max(0.012, 0.04 * qrs_amplitude, 3.0 * noise)
    for candidate in range(search_lo, search_hi - sustain + 1):
        if float(np.max(deviation[candidate:candidate + sustain])) > tolerance:
            continue
        delta_ms = (candidate - qrs_off) * 1000.0 / fs
        qrs_ms = (candidate - qrs_on) * 1000.0 / fs
        if (
            _QRS_TAIL_RESCUE_MIN_DELTA_MS
            <= delta_ms
            <= _QRS_TAIL_RESCUE_MAX_DELTA_MS
            and _QRS_TAIL_RESCUE_MIN_WIDTH_MS
            <= qrs_ms
            <= _QRS_TAIL_RESCUE_MAX_WIDTH_MS
        ):
            return int(candidate)
        return None
    return None


def apply_systematic_qrs_tail_settling_rescue(
    beat_features: List["LeadBeatFeatures"],
    *,
    ecg: np.ndarray,
    fs: int,
    quality: Dict[str, object],
    r_locs: np.ndarray,
    selected_beat_ids: List[int],
) -> Tuple[List["LeadBeatFeatures"], Dict[str, object]]:
    """Extend a systematically early QRS offset using multilead ST settling."""
    selected = {int(beat_id) for beat_id in selected_beat_ids}
    evidence: Dict[str, object] = {
        "applied": False,
        "reason": None,
        "selected_beats": len(selected),
        "qualifying_beats": 0,
        "qualifying_beat_fraction": 0.0,
        "current_qrs_consensus_ms": None,
        "median_rescued_qrs_ms": None,
        "median_extension_ms": None,
        "repaired_entries": 0,
    }
    ecg_arr = np.asarray(ecg, dtype=float)
    if (
        not beat_features
        or fs <= 0
        or ecg_arr.ndim != 2
        or r_locs is None
        or len(selected) < _QRS_TAIL_RESCUE_MIN_SELECTED_BEATS
    ):
        evidence["reason"] = "insufficient_input"
        return beat_features, evidence

    selected_features = [
        feature
        for feature in beat_features
        if int(getattr(feature, "beat_id", -1)) in selected
    ]
    if any(
        "paced_beat" in (getattr(feature, "flags", []) or [])
        for feature in selected_features
    ):
        evidence["reason"] = "paced_context_excluded"
        return beat_features, evidence

    consensus_by_beat: Dict[int, List[float]] = {}
    for feature in selected_features:
        value = getattr(feature, "qrs_consensus_ms", None)
        if value is None or not np.isfinite(float(value)):
            continue
        consensus_by_beat.setdefault(int(feature.beat_id), []).append(float(value))
    beat_consensus = [
        float(np.median(values))
        for values in consensus_by_beat.values()
        if values
    ]
    if not beat_consensus:
        evidence["reason"] = "qrs_consensus_unavailable"
        return beat_features, evidence
    current_qrs_ms = float(np.median(beat_consensus))
    evidence["current_qrs_consensus_ms"] = current_qrs_ms
    if current_qrs_ms > _QRS_TAIL_RESCUE_MAX_CURRENT_MS:
        evidence["reason"] = "current_qrs_not_narrow"
        return beat_features, evidence

    lead_to_index = _repair_lead_indices(beat_features, ecg_arr)
    smooth_window = max(3, int(round(0.008 * fs)))
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    smoothed_by_lead = {
        lead: np.convolve(ecg_arr[index], kernel, mode="same")
        for lead, index in lead_to_index.items()
        if index < ecg_arr.shape[0]
    }

    candidates_by_beat: Dict[
        int,
        List[Tuple["LeadBeatFeatures", int, float, float]],
    ] = {}
    for feature in selected_features:
        lead_quality = quality.get(feature.lead)
        if not bool(getattr(lead_quality, "reliable_for_qrs", False)):
            continue
        if float(getattr(feature, "qrs_confidence", 0.0) or 0.0) <= 0.10:
            continue
        if (
            feature.qrs.onset is None
            or feature.qrs.peak is None
            or feature.qrs.offset is None
        ):
            continue
        beat_id = int(feature.beat_id)
        if beat_id < 0 or beat_id >= len(r_locs):
            continue
        smoothed = smoothed_by_lead.get(feature.lead)
        if smoothed is None:
            continue
        candidate = _qrs_tail_settling_candidate(
            smoothed,
            r=int(r_locs[beat_id]),
            qrs_on=int(feature.qrs.onset),
            qrs_off=int(feature.qrs.offset),
            fs=fs,
        )
        if candidate is None:
            continue
        if feature.t.onset is not None and candidate >= int(feature.t.onset):
            continue
        delta_ms = (candidate - int(feature.qrs.offset)) * 1000.0 / fs
        qrs_ms = (candidate - int(feature.qrs.onset)) * 1000.0 / fs
        candidates_by_beat.setdefault(beat_id, []).append(
            (feature, int(candidate), float(delta_ms), float(qrs_ms))
        )

    qualifying = {
        beat_id: candidates
        for beat_id, candidates in candidates_by_beat.items()
        if len(candidates) >= _QRS_TAIL_RESCUE_MIN_LEADS_PER_BEAT
    }
    qualifying_fraction = len(qualifying) / float(len(selected))
    evidence["qualifying_beats"] = len(qualifying)
    evidence["qualifying_beat_fraction"] = qualifying_fraction
    if (
        len(qualifying) < _QRS_TAIL_RESCUE_MIN_SELECTED_BEATS
        or qualifying_fraction < _QRS_TAIL_RESCUE_MIN_BEAT_FRACTION
    ):
        evidence["reason"] = "insufficient_systematic_multilead_support"
        return beat_features, evidence

    repaired_qrs_ms: List[float] = []
    extension_ms: List[float] = []
    repaired_entries = 0
    n_samples = ecg_arr.shape[1]
    for beat_id, candidates in qualifying.items():
        support = len(candidates)
        target_offset = int(round(float(np.median([item[1] for item in candidates]))))
        r = int(r_locs[beat_id])
        beat_start = max(0, int(r - 0.35 * fs))
        beat_end = min(n_samples, int(r + 0.55 * fs))
        for feature, candidate, delta_ms, qrs_ms in candidates:
            lead_index = lead_to_index.get(feature.lead)
            if lead_index is None:
                continue
            signal = ecg_arr[lead_index, beat_start:beat_end]
            local_r = r - beat_start
            if len(signal) < 4 or not (0 <= local_r < len(signal)):
                continue
            baseline, _source, _confidence = _baseline_contract(
                signal,
                local_r,
                fs,
            )
            old_offset = int(feature.qrs.offset)
            feature.qrs_offset_original_index = old_offset
            feature.qrs_offset_repaired_index = int(candidate)
            feature.qrs_offset_repair_delta_ms = float(delta_ms)
            feature.qrs_offset_repair_reason = "systematic_multilead_tail_settling"
            feature.qrs_offset_repair_confidence = 0.85
            feature.qrs_offset_repair_consensus_index = target_offset
            feature.qrs_offset_repair_support = support
            feature.qrs.offset = int(candidate)
            feature.qrs_off_confidence = max(
                float(getattr(feature, "qrs_off_confidence", 0.0) or 0.0),
                0.85,
            )
            if "qrs_offset_repaired_by_tail_settling" not in feature.flags:
                feature.flags.append("qrs_offset_repaired_by_tail_settling")
            _recompute_qrs_offset_dependent_fields(
                feature,
                sig=signal,
                beat_start=beat_start,
                baseline=baseline,
                fs=fs,
            )
            repaired_qrs_ms.append(qrs_ms)
            extension_ms.append(delta_ms)
            repaired_entries += 1

    if repaired_entries == 0:
        evidence["reason"] = "no_eligible_entries"
        return beat_features, evidence

    beat_features = _apply_multilead_consensus(
        beat_features,
        fs,
        quality,
        r_locs=r_locs,
        ecg=ecg_arr,
    )
    evidence.update(
        {
            "applied": True,
            "reason": "systematic_multilead_tail_reached_st_trend",
            "median_rescued_qrs_ms": float(np.median(repaired_qrs_ms)),
            "median_extension_ms": float(np.median(extension_ms)),
            "repaired_entries": repaired_entries,
        }
    )
    return beat_features, evidence


_LIMB_P_ALIGN_MIN_LEADS = 3        # leads needed before a common instant is defined
_LIMB_P_ALIGN_MAX_SPREAD_MS = 60.0  # beyond this the leads disagree about where the P is
_P_DRIFT_WINDOW_LEAD_MS = 260.0     # P search window, back from QRS onset
_P_DRIFT_WINDOW_TRAIL_MS = 30.0
_P_DRIFT_GRID_POINTS = 12           # coarse grid for the robust slope estimate


def _p_window_drift_mv(
    sig: np.ndarray,
    qrs_on: Optional[int],
    fs: int,
) -> Optional[float]:
    """Residual baseline drift across one beat's P search window, in mV.

    Global preprocessing already median-filters at 200/600 ms, so what is left is
    the drift the P wave has to be distinguished *from*.  The slope is estimated
    Theil-Sen style -- the median of pairwise slopes over a coarse grid -- because
    a least-squares fit would be dragged by the P bump itself, which occupies
    roughly 100 ms of this ~230 ms window.

    Measured across 500 sinus PTB-XL records: the *absolute* drift does not
    distinguish records whose P polarity is wrong from those whose is right (AUC
    0.51).  What does is the drift relative to the P amplitude -- median 0.99 for
    the wrong ones (50% above 1.0) against 0.38 for the right ones (6%) -- i.e.
    the failing records are the ones whose P is no bigger than the drift.
    """
    if qrs_on is None or fs <= 0:
        return None
    lo = max(0, int(qrs_on) - int(_P_DRIFT_WINDOW_LEAD_MS * fs / 1000.0))
    hi = max(lo + 1, int(qrs_on) - int(_P_DRIFT_WINDOW_TRAIL_MS * fs / 1000.0))
    if hi - lo < int(0.10 * fs) or hi > len(sig):
        return None
    seg = np.asarray(sig[lo:hi], dtype=float)
    step = max(1, len(seg) // _P_DRIFT_GRID_POINTS)
    idx = list(range(0, len(seg), step))
    slopes = [
        (seg[b] - seg[a]) / (b - a)
        for i, a in enumerate(idx) for b in idx[i + 1:]
    ]
    if not slopes:
        return None
    drift = abs(float(np.median(slopes))) * len(seg)
    return drift if np.isfinite(drift) else None


def _align_limb_p_amplitudes(
    by_beat: Dict[int, list],
    ecg_arr: Optional[np.ndarray],
    lead_to_index: Dict[str, int],
    fs: int,
) -> None:
    """Measure the six limb-lead P amplitudes at one instant per beat.

    I, II, III, aVR, aVL and aVF are linear combinations of two independent
    measurements, so their amplitudes are only a valid vector if they are
    simultaneous: aVR is *identically* -(I+II)/2 sample by sample.  Measured at
    each lead's own peak they are not simultaneous, and the resulting vector can
    be one no dipole can produce -- a positive P in lead II together with a
    positive P in aVR, which is what put an impossible aVR polarity on 8.6% of
    sinus-labelled PTB-XL records.  Taking every limb lead at the cross-lead
    median peak makes the Einthoven/Goldberger identities hold by construction,
    so those combinations can no longer be reported at all.

    The per-lead peaks sit 0-17 ms apart (median 0-5 ms), and the P is a ~100 ms
    wave, so the amplitude cost of moving to the common instant is small.  When
    the limb leads disagree by more than 60 ms they are not describing one wave,
    and each lead keeps its own measurement.  Boundaries, areas and the
    precordial leads are untouched: V1-V6 satisfy no such identity, and the
    terminal-component measurements are defined inside each lead's own window.
    """
    if ecg_arr is None or ecg_arr.ndim != 2 or not lead_to_index:
        return
    for items in by_beat.values():
        limb = [
            bf for bf in items
            if bf.lead in _P_CONTEXT_LIMB_LEADS
            and bf.p.peak is not None
            and bf.lead in lead_to_index
        ]
        if len(limb) < _LIMB_P_ALIGN_MIN_LEADS:
            continue
        peaks = [int(bf.p.peak) for bf in limb]
        spread_ms = (max(peaks) - min(peaks)) * 1000.0 / fs
        if spread_ms > _LIMB_P_ALIGN_MAX_SPREAD_MS:
            for bf in limb:
                bf.p_amp_source = "lead_local_peaks_disagree"
            continue
        common = int(round(float(np.median(peaks))))
        for bf in limb:
            sig = ecg_arr[lead_to_index[bf.lead]]
            if not (0 <= common < len(sig)):
                continue
            reference, _source = _p_isoelectric_reference(
                sig,
                p_on=bf.p.onset,
                p_off=bf.p.offset,
                qrs_on=bf.qrs.onset,
                fs=fs,
                fallback=0.0,
            )
            bf.p_amp_mv = float(sig[common] - reference)
            bf.p_amp_source = "limb_simultaneous"
            bf.p_window_drift_mv = _p_window_drift_mv(sig, bf.qrs.onset, fs)


def _apply_multilead_consensus(
    beat_features: List["LeadBeatFeatures"],
    fs: int,
    quality: Dict[str, object],
    r_locs: Optional[np.ndarray] = None,
    ecg: Optional[np.ndarray] = None,
) -> List["LeadBeatFeatures"]:
    """
    T025 — Fuse wave boundaries across reliable leads per beat (DXL multi-lead).

    Physiology / DXL alignment:
        • Global QRS-onset  = P10  across reliable_for_qrs leads
          (earliest ventricular activation across all leads).
        • Global QRS-offset = P90  across reliable_for_qrs leads
          (latest offset catches delayed activation in bundle-branch patterns).
        • Global P-onset    = physiologic lead cluster; when at least four
          P-local SNR/stability-qualified leads agree, use the protected second
          earliest onset, otherwise retain the cluster median/P10 fallback.
        • Global P-offset   = supported lead cluster; use the protected second
          latest offset when boundary diagnostics are sufficiently populated.
        • Global T-end      = median  across reliable_for_qt leads.

    Derived fields written into each LeadBeatFeatures:
        qrs_consensus_ms — QRS from global QRS-onset  to global QRS-offset
        qt_consensus_ms  — QT  from global QRS-onset  to global T-end
        pr_consensus_ms  — PR  from global P-onset    to global QRS-onset

    Using global QRS-onset for QT (not per-lead) matches the DXL convention and
    removes the +6.5 ms systematic late-onset bias that inflates per-lead QT.

    ``p`` keeps the corrected lead-local P boundary for backward compatibility.
    The original lead-local boundary is preserved in ``p_*_raw_index``;
    ``p_*_corrected_index`` and ``p_*_consensus_index`` expose the other two
    layers explicitly.
    """
    from collections import defaultdict as _dd
    by_beat: Dict[int, list] = _dd(list)
    for bf in beat_features:
        # ``p_boundary_source`` doubles as the capture sentinel so a legitimate
        # raw ``None`` is not mistaken for an uncaptured value on a repeated call.
        if bf.p_boundary_source is None:
            bf.p_onset_raw_index = (
                int(bf.p.onset) if bf.p.onset is not None else None
            )
            bf.p_offset_raw_index = (
                int(bf.p.offset) if bf.p.offset is not None else None
            )
            bf.p_onset_corrected_index = bf.p_onset_raw_index
            bf.p_offset_corrected_index = bf.p_offset_raw_index
            bf.p_boundary_source = "raw_local"
        by_beat[bf.beat_id].append(bf)
    ecg_arr = np.asarray(ecg, dtype=float) if ecg is not None else None
    lead_to_index = _repair_lead_indices(beat_features, ecg_arr) if ecg_arr is not None and ecg_arr.ndim == 2 else {}
    continuous_pacing = bool(by_beat) and all(
        any("paced_beat" in bf.flags for bf in items)
        for items in by_beat.values()
    )

    def _pct(vals: List[int], p: float) -> Optional[int]:
        if len(vals) >= 4:
            return int(np.percentile(vals, p))
        if len(vals) >= 2:
            return int(np.median(vals))
        return vals[0] if vals else None

    def _spread_ms(vals: List[int]) -> Optional[float]:
        if len(vals) < 2:
            return None
        return float(max(vals) - min(vals)) * 1000.0 / float(fs)

    def _supported_boundary_cluster(
        vals: List[int],
        *,
        radius_ms: float = _P_BOUNDARY_CLUSTER_MS,
        min_support: int = _P_BOUNDARY_MIN_SUPPORT,
    ) -> Tuple[Optional[int], int, Optional[float]]:
        if len(vals) < min_support:
            return None, len(vals), _spread_ms(vals)
        radius_samples = max(1, int(round(radius_ms * fs / 1000.0)))
        ordered = sorted(int(v) for v in vals)
        clusters: List[List[int]] = []
        cur = [ordered[0]]
        for value in ordered[1:]:
            if value - cur[-1] <= radius_samples:
                cur.append(value)
            else:
                clusters.append(cur)
                cur = [value]
        clusters.append(cur)

        supported = [cluster for cluster in clusters if len(cluster) >= min_support]
        if not supported:
            return None, max((len(cluster) for cluster in clusters), default=0), _spread_ms(vals)
        max_support = max(len(cluster) for cluster in supported)
        max_supported = [cluster for cluster in supported if len(cluster) == max_support]
        if len(max_supported) > 1:
            return None, max_support, _spread_ms(vals)

        all_center = float(np.median(ordered))
        supported.sort(
            key=lambda cluster: (
                -len(cluster),
                max(cluster) - min(cluster),
                abs(float(np.median(cluster)) - all_center),
            )
        )
        selected = supported[0]
        return int(np.median(selected)), len(selected), _spread_ms(selected)

    def _select_p_offset_cluster(
        candidates: List[Tuple["LeadBeatFeatures", int, bool]],
        *,
        p_onset: Optional[int],
        qrs_onset: Optional[int],
    ) -> Tuple[Optional[int], int, Optional[float], Optional[str]]:
        """Select a primary cluster, with a guarded later-cluster rescue.

        Unreliable leads may corroborate a later cluster but cannot establish
        it: the rescue still requires multiple reliable leads, cross-family
        support, and support close to the dominant primary cluster.
        """

        primary_values = [
            int(offset)
            for _, offset, is_primary in candidates
            if is_primary
        ]
        fallback_offset, fallback_support, fallback_spread = (
            _supported_boundary_cluster(primary_values)
        )

        def _interval_valid(offset: Optional[int]) -> bool:
            if offset is None:
                return False
            if p_onset is not None:
                p_duration_ms = (int(offset) - int(p_onset)) * 1000.0 / fs
                if not (_P_DURATION_MIN_MS <= p_duration_ms <= _P_DURATION_MAX_MS):
                    return False
            if qrs_onset is not None:
                pr_segment_ms = (int(qrs_onset) - int(offset)) * 1000.0 / fs
                if not (_PR_SEGMENT_MIN_MS <= pr_segment_ms <= _PR_SEGMENT_MAX_MS):
                    return False
            return True

        if not candidates:
            return fallback_offset, fallback_support, fallback_spread, (
                "lead_boundary_cluster" if fallback_offset is not None else None
            )

        radius_samples = max(1, int(round(_P_BOUNDARY_CLUSTER_MS * fs / 1000.0)))
        ordered = sorted(candidates, key=lambda item: int(item[1]))
        clusters: List[List[Tuple["LeadBeatFeatures", int, bool]]] = [[ordered[0]]]
        for item in ordered[1:]:
            if int(item[1]) - int(clusters[-1][-1][1]) <= radius_samples:
                clusters[-1].append(item)
            else:
                clusters.append([item])

        fallback_selected = fallback_offset
        if fallback_offset is not None:
            fallback_clusters = [
                cluster
                for cluster in clusters
                if any(
                    is_primary
                    and abs(int(offset) - int(fallback_offset)) <= radius_samples
                    for _bf, offset, is_primary in cluster
                )
            ]
            if fallback_clusters:
                fallback_cluster = max(
                    fallback_clusters,
                    key=lambda cluster: sum(
                        int(is_primary) for _bf, _offset, is_primary in cluster
                    ),
                )
                selected, _selection_method = (
                    _confidence_gated_boundary_order_stat(
                        [
                            (bf, int(offset))
                            for bf, offset, is_primary in fallback_cluster
                            if is_primary
                        ],
                        boundary="offset",
                    )
                )
                if selected is not None:
                    fallback_selected = int(selected)

        max_primary_support = max(
            (
                sum(1 for _, _, is_primary in cluster if is_primary)
                for cluster in clusters
            ),
            default=0,
        )
        required_primary_support = max(
            _P_OFFSET_LATE_CLUSTER_MIN_PRIMARY_SUPPORT,
            int(np.ceil(
                _P_OFFSET_LATE_CLUSTER_PRIMARY_RATIO
                * float(max_primary_support)
            )),
        )
        late_candidates: List[
            Tuple[int, int, Optional[float]]
        ] = []
        for cluster in clusters:
            if len(cluster) < _P_BOUNDARY_MIN_SUPPORT:
                continue
            primary_support = sum(
                1 for _, _, is_primary in cluster if is_primary
            )
            if primary_support < required_primary_support:
                continue
            leads = [bf.lead for bf, _, _ in cluster]
            limb_support = sum(lead in _P_CONTEXT_LIMB_LEADS for lead in leads)
            precordial_support = len(leads) - limb_support
            if limb_support < 1 or precordial_support < 1:
                continue
            values = [int(offset) for _, offset, _ in cluster]
            center = int(np.median(values))
            selected_offset, _selection_method = (
                _confidence_gated_boundary_order_stat(
                    [(bf, int(offset)) for bf, offset, _primary in cluster],
                    boundary="offset",
                )
            )
            if selected_offset is None:
                selected_offset = center
            if not _interval_valid(selected_offset):
                continue
            late_candidates.append((
                int(selected_offset),
                len(cluster),
                _spread_ms(values),
            ))

        if late_candidates:
            late_offset, late_support, late_spread = max(
                late_candidates,
                key=lambda item: item[0],
            )
            separation_ms = (
                (late_offset - int(fallback_selected)) * 1000.0 / fs
                if fallback_selected is not None
                else None
            )
            if (
                fallback_selected is None
                or (
                    separation_ms is not None
                    and separation_ms >= _P_OFFSET_LATE_CLUSTER_MIN_SEPARATION_MS
                )
            ):
                return (
                    late_offset,
                    late_support,
                    late_spread,
                    "lead_boundary_late_supported_cluster",
                )

        if _interval_valid(fallback_selected):
            return (
                fallback_selected,
                fallback_support,
                fallback_spread,
                "lead_boundary_cluster",
            )
        return None, fallback_support, fallback_spread, None

    def _select_p_onset_cluster(
        candidates: List[Tuple["LeadBeatFeatures", int, int]],
        global_qrs_on: Optional[int],
    ) -> Optional[Dict[str, object]]:
        if global_qrs_on is None or len(candidates) < _P_ONSET_CLUSTER_MIN_SUPPORT:
            return None

        valid: List[Tuple["LeadBeatFeatures", int, int]] = []
        for bf, onset, qrs_ref in candidates:
            pr_ms = (float(qrs_ref) - float(onset)) * 1000.0 / float(fs)
            if 50.0 <= pr_ms <= 400.0:
                valid.append((bf, int(onset), int(qrs_ref)))
        if len(valid) < _P_ONSET_CLUSTER_MIN_SUPPORT:
            return None
        informative_valid = [
            item
            for item in valid
            if getattr(item[0], "p_informative", None) is not False
        ]
        used_low_snr_fallback = (
            len(informative_valid) < _P_ONSET_CLUSTER_MIN_SUPPORT
        )
        if not used_low_snr_fallback:
            valid = informative_valid

        radius_samples = max(1, int(round(_P_ONSET_CLUSTER_MS * fs / 1000.0)))
        lead_rank = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
        ordered = sorted(valid, key=lambda item: (item[1], lead_rank.get(item[0].lead, 999)))
        clusters: List[List[Tuple["LeadBeatFeatures", int, int]]] = []
        cur = [ordered[0]]
        for item in ordered[1:]:
            if item[1] - cur[-1][1] <= radius_samples:
                cur.append(item)
            else:
                clusters.append(cur)
                cur = [item]
        clusters.append(cur)

        supported: List[Dict[str, object]] = []
        for cluster in clusters:
            if len(cluster) < _P_ONSET_CLUSTER_MIN_SUPPORT:
                continue
            onsets = [onset for _, onset, _ in cluster]
            center = int(round(float(np.median(onsets))))
            selected_onset, selection_method = (
                _confidence_gated_boundary_order_stat(
                    [(bf, onset) for bf, onset, _qrs_ref in cluster],
                    boundary="onset",
                )
            )
            if selected_onset is None:
                selected_onset = center
                selection_method = "cluster_median"
            spread_ms = _spread_ms(onsets)
            if (
                spread_ms is not None
                and spread_ms > _P_ONSET_CLUSTER_MAX_SPREAD_MS
            ):
                continue
            cluster_pr_ms = (
                float(global_qrs_on) - float(selected_onset)
            ) * 1000.0 / float(fs)
            if not (
                _P_ONSET_PHYSIOLOGIC_MIN_PR_MS
                <= cluster_pr_ms
                <= _P_ONSET_PHYSIOLOGIC_MAX_PR_MS
            ):
                continue
            fusion_weights = [
                _p_boundary_fusion_weight(bf, boundary="onset")
                for bf, _onset, _qrs_ref in cluster
            ]
            diagnostic_count = sum(
                int(has_diagnostics and weight > 0.0)
                for weight, has_diagnostics in fusion_weights
            )
            weight_sum = float(sum(weight for weight, _has in fusion_weights))
            ordered_cluster = sorted(
                cluster,
                key=lambda item: (item[1], lead_rank.get(item[0].lead, 999)),
            )
            leads = [bf.lead for bf, _, _ in ordered_cluster]
            limb_support = sum(1 for lead in leads if lead in {"I", "II", "III", "aVR", "aVL", "aVF"})
            precordial_support = sum(1 for lead in leads if lead.startswith("V"))
            supported.append(
                {
                    "center": center,
                    "selected": int(selected_onset),
                    "selection_method": selection_method,
                    "support": len(cluster),
                    "spread_ms": spread_ms,
                    "pr_ms": float(cluster_pr_ms),
                    "limb_support": limb_support,
                    "precordial_support": precordial_support,
                    "leads": ",".join(leads),
                    "diagnostic_count": diagnostic_count,
                    "weight_sum": weight_sum,
                    "used_low_snr_fallback": used_low_snr_fallback,
                }
            )

        if not supported:
            return None

        supported.sort(
            key=lambda cluster: (
                -int(int(cluster["diagnostic_count"]) >= _P_ONSET_CLUSTER_MIN_SUPPORT),
                -(
                    float(cluster["weight_sum"])
                    if int(cluster["diagnostic_count"]) >= _P_ONSET_CLUSTER_MIN_SUPPORT
                    else float(cluster["support"])
                ),
                -int(cluster["support"]),
                -int(cluster["limb_support"]),
                float(cluster["spread_ms"]) if cluster["spread_ms"] is not None else 0.0,
                -int(cluster["precordial_support"]),
                -float(cluster["pr_ms"]),
            )
        )
        return supported[0]

    def _qrs_measurement_onset(vals: List[int], allow_central_onset: bool) -> Optional[int]:
        if allow_central_onset and len(vals) >= 6:
            p10 = int(np.percentile(vals, 10))
            p50 = int(np.percentile(vals, 50))
            if (p50 - p10) * 1000.0 / fs >= 20.0:
                return p50
        return _pct(vals, 10)

    def _local_rr_ms(beat_id: int) -> Optional[float]:
        if r_locs is None or len(r_locs) < 2:
            return None
        rr_values: List[float] = []
        if beat_id > 0 and beat_id < len(r_locs):
            rr_values.append((float(r_locs[beat_id]) - float(r_locs[beat_id - 1])) * 1000.0 / fs)
        if beat_id + 1 < len(r_locs):
            rr_values.append((float(r_locs[beat_id + 1]) - float(r_locs[beat_id])) * 1000.0 / fs)
        if not rr_values:
            return None
        return float(np.median(rr_values))

    def _has_terminal_qrs_morphology(bf: "LeadBeatFeatures") -> bool:
        return _qrs_has_terminal_morphology(bf)

    def _select_qrs_terminal_offsets(
        candidates: List[Tuple["LeadBeatFeatures", int]],
        *,
        paced: bool,
    ) -> Tuple[List[int], List[int]]:
        def _lead_csv(candidate_items: List[Tuple["LeadBeatFeatures", int]]) -> Optional[str]:
            leads = [bf.lead for bf, _ in candidate_items]
            return ",".join(leads) if leads else None

        def _mark_candidates(
            *,
            measurement_ids: set[int],
            classification_only_ids: set[int],
            support_count: Optional[int],
            reason_by_id: Optional[Dict[int, str]] = None,
            fallback_path: Optional[str] = None,
        ) -> None:
            reason_by_id = reason_by_id or {}
            used_items = [(bf, off) for bf, off in candidates if id(bf) in measurement_ids]
            excluded_items = [(bf, off) for bf, off in candidates if id(bf) in classification_only_ids]
            used_leads = _lead_csv(used_items)
            excluded_leads = _lead_csv(excluded_items)
            for bf, _ in candidates:
                bf.qrs_offset_used_leads = used_leads
                bf.qrs_offset_excluded_leads = excluded_leads
                bf.qrs_late_cluster_support = support_count
                bf.qrs_terminal_confidence = float(getattr(bf, "qrs_off_confidence", 0.0) or 0.0)
                if fallback_path is not None:
                    bf.qrs_terminal_path = fallback_path
                    bf.qrs_offset_exclusion_reason = None
                elif id(bf) in measurement_ids:
                    bf.qrs_terminal_path = "measurement"
                    bf.qrs_offset_exclusion_reason = None
                elif id(bf) in classification_only_ids:
                    bf.qrs_terminal_path = "classification_only"
                    bf.qrs_offset_exclusion_reason = reason_by_id.get(id(bf), "late_offset_low_support")

        if paced or len(candidates) < 4:
            offsets = [off for _, off in candidates]
            _mark_candidates(
                measurement_ids={id(bf) for bf, _ in candidates},
                classification_only_ids=set(),
                support_count=None,
                fallback_path="paced_consensus" if paced else "measurement",
            )
            return offsets, offsets

        offsets = [off for _, off in candidates]
        center = float(np.median(offsets))
        late_items: List[Tuple["LeadBeatFeatures", int, bool]] = []
        for bf, off in candidates:
            late_gap_ms = (float(off) - center) * 1000.0 / float(fs)
            if late_gap_ms >= _QRS_TERMINAL_LATE_GAP_MS:
                late_items.append((bf, off, _has_terminal_qrs_morphology(bf)))

        supported_late_count = sum(
            1
            for bf, _, has_morphology in late_items
            if has_morphology or not bool(getattr(bf, "st_t_confusion", False))
        )
        late_cluster_supported = supported_late_count >= _QRS_TERMINAL_MIN_SUPPORTED_LATE

        measurement_offsets: List[int] = []
        classification_offsets: List[int] = []
        measurement_ids: set[int] = set()
        classification_only_ids: set[int] = set()
        reason_by_id: Dict[int, str] = {}
        for bf, off in candidates:
            classification_offsets.append(off)
            late_gap_ms = (float(off) - center) * 1000.0 / float(fs)
            is_late = late_gap_ms >= _QRS_TERMINAL_LATE_GAP_MS
            if not is_late:
                measurement_offsets.append(off)
                measurement_ids.add(id(bf))
                if "qrs_terminal_measurement" not in bf.flags:
                    bf.flags.append("qrs_terminal_measurement")
                continue

            st_t_confused = bool(getattr(bf, "st_t_confusion", False))
            has_morphology = _has_terminal_qrs_morphology(bf)
            off_conf = float(getattr(bf, "qrs_off_confidence", 0.0) or 0.0)
            likely_st_t_tail = (
                st_t_confused
                and not has_morphology
                and off_conf < _QRS_TERMINAL_LOW_OFF_CONF
            )
            if likely_st_t_tail or not late_cluster_supported:
                classification_only_ids.add(id(bf))
                reason_by_id[id(bf)] = "st_t_tail" if likely_st_t_tail else "late_offset_low_support"
                if "qrs_terminal_classification_only" not in bf.flags:
                    bf.flags.append("qrs_terminal_classification_only")
                if likely_st_t_tail and "qrs_terminal_excluded_st_t_tail" not in bf.flags:
                    bf.flags.append("qrs_terminal_excluded_st_t_tail")
                continue

            measurement_offsets.append(off)
            measurement_ids.add(id(bf))
            if "qrs_terminal_measurement" not in bf.flags:
                bf.flags.append("qrs_terminal_measurement")

        if len(measurement_offsets) < 2:
            _mark_candidates(
                measurement_ids={id(bf) for bf, _ in candidates},
                classification_only_ids=set(),
                support_count=supported_late_count,
                fallback_path="measurement_fallback",
            )
            return offsets, offsets
        _mark_candidates(
            measurement_ids=measurement_ids,
            classification_only_ids=classification_only_ids,
            support_count=supported_late_count,
            reason_by_id=reason_by_id,
        )
        return measurement_offsets, classification_offsets

    for beat_id, items in by_beat.items():
        paced_items = continuous_pacing and any("paced_beat" in bf.flags for bf in items)
        # --- global QRS onset / offset ---
        qrs_ons:  List[int] = []
        qrs_offs: List[int] = []
        qrs_off_candidates: List[Tuple["LeadBeatFeatures", int]] = []
        _QRS_MAX_MS = 250.0  # reject leads where per-lead QRS > 250 ms (T-wave overlap)
        for bf in items:
            if not (bf.qrs_confidence > 0.10
                    and getattr(quality.get(bf.lead), "reliable_for_qrs", False)):
                continue
            on  = bf.qrs.onset
            off = bf.qrs.offset
            # Sanity-check per-lead QRS width before admitting to consensus vote.
            if on is not None and off is not None:
                per_lead_ms = (off - on) * 1000.0 / fs
                if per_lead_ms > _QRS_MAX_MS:
                    continue
            if on is not None:
                qrs_ons.append(int(on))
            if off is not None:
                qrs_offs.append(int(off))
                qrs_off_candidates.append((bf, int(off)))

        # Compute global QRS onset early so the T-end collector can use it.
        allow_central_qrs_onset = bool(
            (local_rr := _local_rr_ms(int(beat_id))) is not None
            and local_rr >= 1200.0
        )
        global_qrs_on      = _qrs_measurement_onset(qrs_ons, allow_central_qrs_onset)
        # Paced complexes often have a long discordant ST/T shoulder in lateral
        # leads.  For measurement, use the median offset so the QRS interval is
        # not dragged into early repolarization; keep a later percentile only for
        # wide-QRS classification.
        qrs_measurement_offs, qrs_classification_offs = _select_qrs_terminal_offsets(
            qrs_off_candidates,
            paced=paced_items,
        )
        qrs_measurement_pct = (
            50
            if paced_items or len(qrs_measurement_offs) < _QRS_TERMINAL_P75_MIN_OFFSETS
            else 75
        )
        global_qrs_off     = _pct(qrs_measurement_offs, qrs_measurement_pct)
        global_qrs_off_p90 = _pct(qrs_classification_offs, 75 if paced_items else 90)

        # Pre-compute per-beat RR for QTcB guard (used in both T-end filter and
        # the post-hoc global QTcB sanity check below).
        _rr_sec_beat = (
            (r_locs[beat_id + 1] - r_locs[beat_id]) / fs
            if r_locs is not None and beat_id + 1 < len(r_locs)
            else None
        )

        # --- global P-onset ---
        p_onsets: List[int] = []
        p_onset_candidates: List[Tuple["LeadBeatFeatures", int, int]] = []
        p_offset_candidates: List[Tuple["LeadBeatFeatures", int, bool]] = []
        for bf in items:
            if "paced_beat" in bf.flags or "retrograde_p" in bf.flags:
                continue
            if (bf.p.onset is not None
                    and bf.p_confidence > 0.25
                    and getattr(quality.get(bf.lead), "reliable_for_p", False)):
                p_on = int(bf.p.onset)
                p_onsets.append(p_on)
                qrs_ref = global_qrs_on if global_qrs_on is not None else bf.qrs.onset
                if qrs_ref is not None:
                    p_onset_candidates.append((bf, p_on, int(qrs_ref)))
            if not (
                bf.p.onset is not None
                and bf.p.peak is not None
                and bf.p.offset is not None
                and bf.p_confidence > 0.25
            ):
                continue
            p_on = int(bf.p.onset)
            p_peak = int(bf.p.peak)
            p_off = int(bf.p.offset)
            if not (p_on < p_peak < p_off):
                continue
            p_dur_ms = (p_off - p_on) * 1000.0 / fs
            if not (_P_DURATION_MIN_MS <= p_dur_ms <= _P_DURATION_MAX_MS):
                continue
            qrs_ref = global_qrs_on if global_qrs_on is not None else bf.qrs.onset
            if qrs_ref is not None:
                pr_segment_ms = (int(qrs_ref) - p_off) * 1000.0 / fs
                if not (_PR_SEGMENT_MIN_MS <= pr_segment_ms <= _PR_SEGMENT_MAX_MS):
                    continue
            p_offset_candidates.append(
                (
                    bf,
                    p_off,
                    bool(
                        getattr(quality.get(bf.lead), "reliable_for_p", False)
                        and getattr(bf, "p_informative", None) is not False
                    ),
                )
            )
        if (
            sum(int(is_primary) for _bf, _offset, is_primary in p_offset_candidates)
            < _P_BOUNDARY_MIN_SUPPORT
        ):
            # Preserve a low-confidence coverage fallback when every visible P
            # lead is close to the local noise floor. Such leads cannot trigger
            # confidence-gated order statistics, but the legacy quality-gated
            # cluster remains available instead of forcing a missing boundary.
            p_offset_candidates = [
                (
                    bf,
                    offset,
                    bool(getattr(quality.get(bf.lead), "reliable_for_p", False)),
                )
                for bf, offset, _is_primary in p_offset_candidates
            ]

        # --- global T-end ---
        # Per-lead QTcB guard: exclude individual T-end candidates that would
        # imply QTcB > 590 ms.  This stops one or two leads that mis-detected
        # a U-wave as T-end from dragging the median to the U-wave region,
        # particularly in records with inverted T followed by a prominent U-wave.
        t_ends: List[int] = []
        for bf in items:
            if (bf.t.offset is not None
                    and bf.qt_confidence > 0.20
                    and getattr(quality.get(bf.lead), "reliable_for_qt", False)):
                if (_rr_sec_beat is not None
                        and global_qrs_on is not None):
                    qt_lead_ms = (bf.t.offset - global_qrs_on) * 1000.0 / fs
                    if qt_lead_ms > 590.0 * float(np.sqrt(_rr_sec_beat)):
                        continue  # skip this lead — T-end likely misdetected
                t_ends.append(int(bf.t.offset))

        p_onset_cluster = _select_p_onset_cluster(p_onset_candidates, global_qrs_on)
        if p_onset_cluster is not None:
            global_p_onset = int(
                p_onset_cluster.get("selected", p_onset_cluster["center"])
            )
            p_onset_support = int(p_onset_cluster["support"])
            p_onset_spread_ms = p_onset_cluster["spread_ms"]
            selection_method = str(
                p_onset_cluster.get("selection_method") or "cluster_median"
            )
            p_onset_reason: Optional[str] = (
                "physiologic_onset_cluster"
                if selection_method == "cluster_median"
                else selection_method
            )
        else:
            global_p_onset = _pct(p_onsets, 10)
            p_onset_support = len(p_onsets)
            p_onset_spread_ms = _spread_ms(p_onsets)
            p_onset_reason = "onset_percentile_fallback" if global_p_onset is not None else None
        (
            global_p_offset,
            p_offset_support,
            p_offset_spread_ms,
            p_boundary_source,
        ) = _select_p_offset_cluster(
            p_offset_candidates,
            p_onset=global_p_onset,
            qrs_onset=global_qrs_on,
        )
        if (
            global_p_offset is None
            and p_onset_cluster is not None
            and int(global_p_onset) != int(p_onset_cluster["center"])
        ):
            # A protected early onset can occasionally make every otherwise
            # supported offset violate the broad P-duration guard. Preserve
            # coverage by reverting this beat to the cluster median as a paired
            # geometry fallback rather than dropping P offset altogether.
            paired_onset = int(p_onset_cluster["center"])
            (
                paired_offset,
                paired_offset_support,
                paired_offset_spread_ms,
                paired_boundary_source,
            ) = _select_p_offset_cluster(
                p_offset_candidates,
                p_onset=paired_onset,
                qrs_onset=global_qrs_on,
            )
            if paired_offset is not None:
                global_p_onset = paired_onset
                global_p_offset = paired_offset
                p_offset_support = paired_offset_support
                p_offset_spread_ms = paired_offset_spread_ms
                p_boundary_source = paired_boundary_source
                p_onset_reason = "cluster_median_paired_geometry_fallback"
        if global_p_offset is None:
            # After the physiology-aware baseline/SNR pass, confidence gating
            # can make onset and offset clusters come from slightly different
            # lead subsets and form an invalidly short global P even though
            # each lead has valid paired geometry.  Recover only with the joint
            # medians of the *same* eligible lead-pairs; this preserves coverage
            # without reintroducing an earliest/latest extreme.  Requiring the
            # diagnostic provenance also keeps legacy/ambiguous callers from
            # forcing a cluster that was intentionally left unresolved.
            paired_candidates = [
                (bf, int(bf.p.onset), int(offset))
                for bf, offset, is_primary in p_offset_candidates
                if (
                    is_primary
                    and bf.p.onset is not None
                    and int(bf.p.onset) < int(offset)
                )
            ]
            diagnostic_support = sum(
                int(
                    getattr(bf, "p_baseline_method", None) is not None
                    or getattr(bf, "p_local_noise_source", None) is not None
                )
                for bf, _onset, _offset in paired_candidates
            )
            if (
                len(paired_candidates) >= _P_BOUNDARY_MIN_SUPPORT
                and diagnostic_support >= _P_BOUNDARY_MIN_SUPPORT
            ):
                paired_onset = int(round(float(np.median(
                    [onset for _bf, onset, _offset in paired_candidates]
                ))))
                paired_offset = int(round(float(np.median(
                    [offset for _bf, _onset, offset in paired_candidates]
                ))))
                paired_duration_ms = (
                    paired_offset - paired_onset
                ) * 1000.0 / fs
                paired_pr_segment_ms = (
                    (int(global_qrs_on) - paired_offset) * 1000.0 / fs
                    if global_qrs_on is not None
                    else None
                )
                paired_geometry_valid = (
                    _P_DURATION_MIN_MS
                    <= paired_duration_ms
                    <= _P_DURATION_MAX_MS
                    and (
                        paired_pr_segment_ms is None
                        or _PR_SEGMENT_MIN_MS
                        <= paired_pr_segment_ms
                        <= _PR_SEGMENT_MAX_MS
                    )
                )
                if paired_geometry_valid:
                    global_p_onset = paired_onset
                    global_p_offset = paired_offset
                    p_offset_support = len(paired_candidates)
                    p_offset_spread_ms = _spread_ms(
                        [
                            offset
                            for _bf, _onset, offset in paired_candidates
                        ]
                    )
                    p_boundary_source = (
                        "paired_lead_joint_median_geometry_fallback"
                    )
                    p_onset_reason = (
                        "paired_lead_joint_median_geometry_fallback"
                    )
        # Keep the cluster center as the lead-local correction target. The
        # global measurement may be extended below by paired-duration evidence,
        # but that extension must not overwrite local morphology.
        p_offset_correction_target = global_p_offset
        p_dur_consensus_ms: Optional[float] = None
        pr_segment_consensus_ms: Optional[float] = None
        p_duration_guard_target_ms: Optional[float] = None
        p_duration_guard_delta_ms: Optional[float] = None
        p_duration_guard_support: Optional[int] = None
        p_duration_guard_source: Optional[str] = None
        if global_p_onset is not None and global_p_offset is not None:
            p_dur_candidate = (global_p_offset - global_p_onset) * 1000.0 / fs
            if _P_DURATION_MIN_MS <= p_dur_candidate <= _P_DURATION_MAX_MS:
                paired_durations_ms = [
                    (int(offset) - int(bf.p.onset)) * 1000.0 / fs
                    for bf, offset, is_primary in p_offset_candidates
                    if (
                        is_primary
                        and bf.p.onset is not None
                        and int(bf.p.onset) < int(offset)
                    )
                ]
                if len(paired_durations_ms) >= _P_PAIRED_DURATION_GUARD_MIN_SUPPORT:
                    paired_target_ms = float(np.percentile(
                        paired_durations_ms,
                        _P_PAIRED_DURATION_GUARD_PERCENTILE,
                    ))
                    requested_extension_ms = min(
                        _P_PAIRED_DURATION_GUARD_MAX_EXTENSION_MS,
                        max(0.0, paired_target_ms - p_dur_candidate),
                    )
                    extension_samples = int(round(
                        requested_extension_ms * fs / 1000.0
                    ))
                    guarded_offset = int(global_p_offset) + extension_samples
                    guarded_duration_ms = (
                        guarded_offset - int(global_p_onset)
                    ) * 1000.0 / fs
                    guarded_pr_segment_ms = (
                        (int(global_qrs_on) - guarded_offset) * 1000.0 / fs
                        if global_qrs_on is not None
                        else None
                    )
                    guard_valid = (
                        extension_samples > 0
                        and _P_DURATION_MIN_MS
                        <= guarded_duration_ms
                        <= _P_DURATION_MAX_MS
                        and (
                            guarded_pr_segment_ms is None
                            or _PR_SEGMENT_MIN_MS
                            <= guarded_pr_segment_ms
                            <= _PR_SEGMENT_MAX_MS
                        )
                    )
                    if guard_valid:
                        global_p_offset = guarded_offset
                        p_dur_candidate = guarded_duration_ms
                        p_duration_guard_target_ms = paired_target_ms
                        p_duration_guard_delta_ms = (
                            extension_samples * 1000.0 / fs
                        )
                        p_duration_guard_support = len(paired_durations_ms)
                        p_duration_guard_source = (
                            "reliable_paired_lead_duration_p90"
                        )
                        p_boundary_source = "paired_lead_duration_guard"
                p_dur_consensus_ms = float(p_dur_candidate)
                if global_qrs_on is not None:
                    pr_segment_candidate = (global_qrs_on - global_p_offset) * 1000.0 / fs
                    if _PR_SEGMENT_MIN_MS <= pr_segment_candidate <= _PR_SEGMENT_MAX_MS:
                        pr_segment_consensus_ms = float(pr_segment_candidate)
                    else:
                        global_p_offset = None
                        p_dur_consensus_ms = None
                        p_boundary_source = None
        global_t_end   = _pct(t_ends,   50)   # DXL: median minimises SD (Fig 1-22)

        # Global QTcB sanity: reject global_t_end if the implied QT still
        # requires QTcB > 590 ms after per-lead filtering.
        if (global_t_end is not None
                and global_qrs_on is not None
                and _rr_sec_beat is not None):
            qt_cand_ms = (global_t_end - global_qrs_on) * 1000.0 / fs
            if qt_cand_ms > 590.0 * float(np.sqrt(_rr_sec_beat)):
                global_t_end = None

        def _corrected_p_geometry_valid(
            p_on: Optional[int],
            p_peak: Optional[int],
            p_off: Optional[int],
            qrs_ref: Optional[int],
        ) -> bool:
            if p_on is None or p_peak is None or p_off is None:
                return False
            if not (int(p_on) < int(p_peak) < int(p_off)):
                return False
            p_dur_ms = (int(p_off) - int(p_on)) * 1000.0 / fs
            if not (_P_DURATION_MIN_MS <= p_dur_ms <= _P_DURATION_MAX_MS):
                return False
            onset_to_peak_ms = (int(p_peak) - int(p_on)) * 1000.0 / fs
            if onset_to_peak_ms > _P_ONSET_TO_PEAK_MAX_MS:
                return False
            if qrs_ref is not None:
                pr_segment_ms = (int(qrs_ref) - int(p_off)) * 1000.0 / fs
                if not (_PR_SEGMENT_MIN_MS <= pr_segment_ms <= _PR_SEGMENT_MAX_MS):
                    return False
            return True

        def _maybe_update_p_duration(bf: "LeadBeatFeatures") -> None:
            if bf.p.onset is None or bf.p.offset is None:
                return
            bf.p_dur_ms = (int(bf.p.offset) - int(bf.p.onset)) * 1000.0 / fs

        def _update_p_boundary_confidence(bf: "LeadBeatFeatures") -> None:
            qrs_ref = global_qrs_on if global_qrs_on is not None else bf.qrs.onset
            support_score = _p_context_support_score(int(p_offset_support or 0))
            onset_conf, offset_conf = _p_boundary_confidence_scores(
                p_on=bf.p.onset,
                p_peak=bf.p.peak,
                p_off=bf.p.offset,
                qrs_ref=qrs_ref,
                fs=fs,
                p_confidence=float(getattr(bf, "p_confidence", 0.0) or 0.0),
                support_score=support_score,
                onset_sigma_ms=getattr(bf, "p_onset_sigma_ms", None),
                offset_sigma_ms=getattr(bf, "p_offset_sigma_ms", None),
                informative=getattr(bf, "p_informative", None),
            )
            bf.p_onset_confidence = onset_conf
            bf.p_offset_confidence = offset_conf

        def _remeasure_p_after_boundary_change(bf: "LeadBeatFeatures") -> None:
            if ecg_arr is None or ecg_arr.ndim != 2:
                return
            if bf.p.onset is None or bf.p.peak is None or bf.p.offset is None:
                return
            li = lead_to_index.get(bf.lead)
            if li is None or li >= ecg_arr.shape[0]:
                return
            sig = ecg_arr[li]
            if not (0 <= int(bf.p.onset) < int(bf.p.peak) < int(bf.p.offset) < len(sig)):
                return
            qrs_ref = global_qrs_on if global_qrs_on is not None else bf.qrs.onset
            baseline_ref = int(qrs_ref) if qrs_ref is not None else int(bf.p.peak)
            baseline, _baseline_source, _baseline_confidence = _baseline_contract(sig, baseline_ref, fs)
            p_on = int(bf.p.onset)
            p_peak = int(bf.p.peak)
            p_off = int(bf.p.offset)
            p_amp_baseline, _p_amp_baseline_source = _p_isoelectric_reference(
                sig,
                p_on=p_on,
                p_off=p_off,
                qrs_on=int(qrs_ref) if qrs_ref is not None else None,
                fs=fs,
                fallback=baseline,
            )
            p_seg = sig[p_on: p_off + 1] - p_amp_baseline
            trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            bf.p_amp_mv = float(sig[p_peak] - p_amp_baseline)
            bf.p_area = float(trapz(np.abs(p_seg)))
            bf.p_signed_area = float(trapz(p_seg))
            components = measure_p_components(
                sig,
                p_on=p_on,
                p_peak=p_peak,
                p_off=p_off,
                baseline=p_amp_baseline,
                fs=fs,
            )
            bf.p_notched = bool(components["is_notched"])
            bf.p_biphasic = bool(components["is_biphasic"])
            bf.p_notch_interval_ms = components["notch_interval_ms"]
            bf.p_initial_duration_ms = components["initial_duration_ms"]
            bf.p_initial_amp_mv = components["initial_amplitude_mV"]
            bf.p_terminal_duration_ms = components["terminal_duration_ms"]
            bf.p_terminal_amp_mv = components["terminal_amplitude_mV"]
            bf.p_terminal_area_mv_ms = components["terminal_area_mv_ms"]
            bf.ptf_v1_mv_ms = (
                _ptf_v1(sig, p_on, p_off, p_amp_baseline, fs)
                if bf.lead == "V1" and float(getattr(bf, "p_confidence", 0.0) or 0.0) > 0.30
                else None
            )

        def _suppress_p_candidate(bf: "LeadBeatFeatures") -> None:
            bf.p = WaveBounds(onset=None, peak=None, offset=None)
            bf.pr_ms = None
            bf.p_dur_ms = None
            bf.p_amp_mv = None
            bf.p_area = None
            bf.p_signed_area = None
            bf.ptf_v1_mv_ms = None
            bf.p_notched = False
            bf.p_biphasic = False
            bf.p_notch_interval_ms = None
            bf.p_initial_duration_ms = None
            bf.p_initial_amp_mv = None
            bf.p_terminal_duration_ms = None
            bf.p_terminal_amp_mv = None
            bf.p_terminal_area_mv_ms = None
            bf.p_confidence = 0.0
            if "p_candidate_suppressed_by_consensus" not in bf.flags:
                bf.flags.append("p_candidate_suppressed_by_consensus")
            if "p_unreliable" not in bf.flags:
                bf.flags.append("p_unreliable")

        def _maybe_suppress_false_p_candidate(bf: "LeadBeatFeatures") -> None:
            if p_onset_cluster is None or global_p_onset is None:
                return
            if bf.p.onset is None or bf.p.peak is None:
                return
            qrs_ref = global_qrs_on if global_qrs_on is not None else bf.qrs.onset
            if qrs_ref is None:
                return
            raw_pr_ms = (int(qrs_ref) - int(bf.p.onset)) * 1000.0 / fs
            cluster_pr_ms = (int(qrs_ref) - int(global_p_onset)) * 1000.0 / fs
            onset_to_peak_ms = (int(bf.p.peak) - int(global_p_onset)) * 1000.0 / fs
            if raw_pr_ms >= _P_FALSE_CANDIDATE_SHORT_PR_MS:
                return
            if cluster_pr_ms - raw_pr_ms < _P_FALSE_CANDIDATE_MIN_CLUSTER_DELTA_MS:
                return
            if onset_to_peak_ms <= _P_ONSET_TO_PEAK_MAX_MS:
                return
            _suppress_p_candidate(bf)

        def _maybe_correct_raw_p_boundaries(bf: "LeadBeatFeatures") -> None:
            if "paced_beat" in bf.flags or "retrograde_p" in bf.flags:
                return
            qrs_ref = global_qrs_on if global_qrs_on is not None else bf.qrs.onset
            if (
                p_onset_cluster is not None
                and global_p_onset is not None
                and bf.p.onset is not None
                and abs(int(bf.p.onset) - int(global_p_onset)) * 1000.0 / fs
                >= _P_BOUNDARY_CORRECTION_MIN_SHIFT_MS
                and _corrected_p_geometry_valid(
                    int(global_p_onset),
                    bf.p.peak,
                    bf.p.offset,
                    qrs_ref,
                )
            ):
                bf.p.onset = int(global_p_onset)
                if qrs_ref is not None:
                    bf.pr_ms = (int(qrs_ref) - int(global_p_onset)) * 1000.0 / fs
                _maybe_update_p_duration(bf)
                _remeasure_p_after_boundary_change(bf)
                if "p_onset_corrected_by_consensus" not in bf.flags:
                    bf.flags.append("p_onset_corrected_by_consensus")

            if (
                p_offset_correction_target is not None
                and bf.p.offset is not None
                and p_offset_support >= _P_BOUNDARY_MIN_SUPPORT
                and (
                    p_offset_spread_ms is None
                    or p_offset_spread_ms <= _P_BOUNDARY_CORRECTION_MAX_SPREAD_MS
                )
                and abs(
                    int(bf.p.offset) - int(p_offset_correction_target)
                ) * 1000.0 / fs
                >= _P_BOUNDARY_CORRECTION_MIN_SHIFT_MS
            ):
                # The global boundary is a measurement target, not a reason to
                # erase valid lead-local morphology. Move only halfway toward
                # it; raw and global remain available as independent layers.
                raw_offset = int(bf.p.offset)
                corrected_offset = int(round(
                    raw_offset
                    + _P_OFFSET_CORRECTION_FRACTION
                    * (int(p_offset_correction_target) - raw_offset)
                ))
                if _corrected_p_geometry_valid(
                    bf.p.onset,
                    bf.p.peak,
                    corrected_offset,
                    qrs_ref,
                ):
                    bf.p.offset = corrected_offset
                    bf.p_offset_correction_fraction = _P_OFFSET_CORRECTION_FRACTION
                    _maybe_update_p_duration(bf)
                    _remeasure_p_after_boundary_change(bf)
                    if "p_offset_corrected_by_consensus" not in bf.flags:
                        bf.flags.append("p_offset_corrected_by_consensus")

            _maybe_suppress_false_p_candidate(bf)

        for bf in items:
            # QRS consensus duration (P75 for measurement, P90 for classification)
            if global_qrs_on is not None and global_qrs_off is not None:
                qrs_c = (global_qrs_off - global_qrs_on) * 1000.0 / fs
                if 20.0 <= qrs_c <= 300.0:
                    bf.qrs_consensus_ms = float(qrs_c)
            if global_qrs_on is not None and global_qrs_off_p90 is not None:
                qrs_c_p90 = (global_qrs_off_p90 - global_qrs_on) * 1000.0 / fs
                if 20.0 <= qrs_c_p90 <= 300.0:
                    bf.qrs_wide_ms = float(qrs_c_p90)
                    if (
                        bf.st_on_mv is not None
                        and bf.st_mid_mv is not None
                        and bf.st_80ms_mv is not None
                        and _st_j_tail_outlier(
                            float(bf.st_on_mv),
                            float(bf.st_mid_mv),
                            float(bf.st_80ms_mv),
                            bf.qrs_wide_ms,
                        )
                    ):
                        bf.st_on_mv = None
                        if "st_j_unreliable" not in bf.flags:
                            bf.flags.append("st_j_unreliable")

            # PR: global P-onset → global QRS-onset
            if global_p_onset is not None and global_qrs_on is not None:
                pr_c = (global_qrs_on - global_p_onset) * 1000.0 / fs
                if 50.0 <= pr_c <= 400.0:
                    bf.pr_consensus_ms = float(pr_c)
                    bf.p_onset_consensus_index = int(global_p_onset)
                    bf.p_onset_consensus_support = int(p_onset_support)
                    bf.p_onset_consensus_spread_ms = p_onset_spread_ms
                    bf.p_onset_consensus_reason = p_onset_reason
                    if p_onset_cluster is not None:
                        bf.p_onset_cluster_center_index = int(p_onset_cluster["center"])
                        bf.p_onset_cluster_support = int(p_onset_cluster["support"])
                        spread = p_onset_cluster["spread_ms"]
                        bf.p_onset_cluster_spread_ms = (
                            float(spread) if spread is not None else None
                        )
                        bf.p_onset_cluster_pr_ms = float(p_onset_cluster["pr_ms"])
                        bf.p_onset_cluster_limb_support = int(p_onset_cluster["limb_support"])
                        bf.p_onset_cluster_precordial_support = int(
                            p_onset_cluster["precordial_support"]
                        )
                        bf.p_onset_cluster_leads = str(p_onset_cluster["leads"])

            if global_p_offset is not None:
                bf.p_offset_consensus_index = int(global_p_offset)
                bf.p_offset_consensus_support = int(p_offset_support)
                bf.p_offset_consensus_spread_ms = p_offset_spread_ms
                bf.p_dur_consensus_ms = p_dur_consensus_ms
                bf.pr_segment_consensus_ms = pr_segment_consensus_ms
                bf.p_boundary_consensus_source = p_boundary_source
                bf.p_duration_guard_target_ms = p_duration_guard_target_ms
                bf.p_duration_guard_delta_ms = p_duration_guard_delta_ms
                bf.p_duration_guard_support = p_duration_guard_support
                bf.p_duration_guard_source = p_duration_guard_source

            _maybe_correct_raw_p_boundaries(bf)
            bf.p_onset_corrected_index = (
                int(bf.p.onset) if bf.p.onset is not None else None
            )
            bf.p_offset_corrected_index = (
                int(bf.p.offset) if bf.p.offset is not None else None
            )
            if "p_candidate_suppressed_by_consensus" in bf.flags:
                bf.p_boundary_source = "suppressed"
            elif (
                "p_onset_corrected_by_consensus" in bf.flags
                or "p_offset_corrected_by_consensus" in bf.flags
            ):
                bf.p_boundary_source = (
                    "lead_consensus"
                    if "p_onset_corrected_by_consensus" in bf.flags
                    else "lead_consensus_blend"
                )
            else:
                bf.p_boundary_source = "raw_local"
            _update_p_boundary_confidence(bf)

            # QT: global QRS-onset → global T-end  (DXL: removes per-lead onset bias)
            if global_qrs_on is not None and global_t_end is not None:
                qt_c = (global_t_end - global_qrs_on) * 1000.0 / fs
                if 200.0 <= qt_c <= 700.0:
                    bf.qt_consensus_ms = float(qt_c)

    _align_limb_p_amplitudes(by_beat, ecg_arr, lead_to_index, fs)
    return beat_features


# ─────────────────────────────────────────────────────────────────────────────
# Main delineation entry point
# ─────────────────────────────────────────────────────────────────────────────

def _detect_delta_wave(
    sig: np.ndarray,
    qrs_on: Optional[int],
    r: int,
    baseline: float,
    fs: int,
) -> bool:
    """
    Heuristic delta-wave detector for WPW pre-excitation.

    A delta wave is characterised by a gradual initial QRS upstroke
    (slurred onset) followed by the main rapid deflection.  We measure:
      - delta_fraction : |peak of first 40 ms| / |R amplitude|
      - slope_ratio    : mean slope of first 40 ms / mean slope of full QRS

    Returns True when both conditions hold:
      delta_fraction >= 0.15  (initial slur carries ≥15 % of R amplitude)
      slope_ratio    <  0.60  (initial slope significantly shallower than full)
    """
    if qrs_on is None or r <= qrs_on:
        return False
    r_amp = abs(float(sig[r]) - baseline)
    if r_amp < 0.05:
        return False
    delta_end = min(r, qrs_on + max(2, int(0.040 * fs)))
    if delta_end <= qrs_on:
        return False
    initial = sig[qrs_on:delta_end] - baseline
    full    = sig[qrs_on:r + 1] - baseline
    if len(initial) < 2 or len(full) < 2:
        return False
    delta_fraction = float(np.max(np.abs(initial))) / r_amp
    initial_slope  = float(np.max(np.abs(initial))) / max(1, len(initial))
    full_slope     = float(np.max(np.abs(full)))    / max(1, len(full))
    slope_ratio    = initial_slope / (full_slope + 1e-9)
    return delta_fraction >= 0.15 and slope_ratio < 0.60


def delineate_beats(
    ecg: np.ndarray,
    fs: int,
    r_locs: np.ndarray,
    beat_groups: Optional[Dict[int, List[int]]] = None,
    rep_beats: Optional[Dict[int, np.ndarray]] = None,
    rep_left_ms: int = 300,
    paced_beat_ids: Optional[List[int]] = None,
    paced_qrs_floor_beat_ids: Optional[List[int]] = None,
    pacing_spike_times: Optional[List[int]] = None,
    quality: Optional[Dict[str, object]] = None,
    qrs_low_slope_guard: bool = False,
) -> List[LeadBeatFeatures]:
    """
    Delineate P/QRS/T boundaries for every beat × lead.

    T010 — Adaptive search regions:
        T-end search is capped so it cannot overlap the next beat's QRS region.

    T018 — Representative-beat priors:
        If rep_beats provided, priors are computed from group representative beats.

    T019 — Beat-wise local correction:
        If a prior exists for this lead/group, P/QRS/T searches are nudged into
        small local windows around the representative prediction, reducing
        beat-to-beat variance without hard-clamping the final boundary.
    """
    # Build beat → group lookup
    beat_to_group: Dict[int, int] = {}
    if beat_groups:
        for gid, members in beat_groups.items():
            for m in members:
                beat_to_group[m] = gid

    # Build representative priors per group × lead (T018)
    group_priors: Dict[int, Dict[str, Dict]] = {}
    if rep_beats:
        group_priors = build_group_priors(
            rep_beats,
            fs,
            left_ms=rep_left_ms,
            qrs_low_slope_guard=qrs_low_slope_guard,
        )
    paced_set = set(paced_beat_ids or [])
    paced_floor_set = (
        set(paced_qrs_floor_beat_ids)
        if paced_qrs_floor_beat_ids is not None
        else paced_set
    )
    spike_times = np.asarray(pacing_spike_times or [], dtype=int)

    LOCAL_MARGIN = int(0.040 * fs)   # QRS correction window
    P_PRIOR_MARGIN = int(0.040 * fs)
    T_PRIOR_MARGIN = int(0.060 * fs)
    T_END_PRIOR_MARGIN = int(0.030 * fs)

    out: List[LeadBeatFeatures] = []
    p_alternatives_by_key: Dict[Tuple[int, str], List[PCandidateAlternative]] = {}
    # Previous beat's measured T-offset per lead (absolute sample index), used
    # to keep the next beat's P search window from picking up the decaying
    # T-tail when the TP interval is short (fast rate / P-on-T).
    prev_t_off_by_lead: Dict[str, Optional[int]] = {}
    P_T_TAIL_GUARD = int(0.020 * fs)
    n = ecg.shape[1]
    all_rr_ms = np.diff(r_locs) * 1000.0 / fs if len(r_locs) > 1 else np.asarray([], dtype=float)

    for beat_id, r in enumerate(r_locs):
        beat_start = max(0, int(r - 0.35 * fs))
        beat_end   = min(n, int(r + 0.55 * fs))

        # T010: compute adaptive T-end cap from next R peak.
        # At lower HR the TP segment is long, so 28% of RR pushes the cap further
        # from next_R (keeping it out of the P-wave region). At high HR the 150 ms
        # fixed floor dominates, preserving sensitivity for physiologically short QT.
        if beat_id + 1 < len(r_locs):
            next_r       = int(r_locs[beat_id + 1])
            rr_samp      = next_r - int(r)
            t_cap_global = next_r - max(int(0.15 * fs), int(round(0.28 * rr_samp)))
        else:
            t_cap_global = n - 1

        # Adaptive P-search lookback based on preceding RR interval.
        # For bradycardia (long RR), the P-wave can be > 250 ms before R;
        # fixed 250 ms was missing P onset in those cases.
        if beat_id > 0:
            rr_prev        = int(r_locs[beat_id] - r_locs[beat_id - 1])
            rr_prev_ms     = rr_prev * 1000.0 / fs
            p_max_lookback = min(int(0.45 * fs), int(0.65 * rr_prev))
        else:
            rr_prev_ms     = None
            p_max_lookback = int(0.40 * fs)
        rr_for_12sl: List[float] = []
        if rr_prev_ms is not None:
            rr_for_12sl.append(float(rr_prev_ms))
        if beat_id + 1 < len(r_locs):
            rr_for_12sl.append(float((int(r_locs[beat_id + 1]) - int(r)) * 1000.0 / fs))
        avg_rr_12sl_ms = (
            float(np.mean(rr_for_12sl))
            if rr_for_12sl
            else (float(np.median(all_rr_ms)) if len(all_rr_ms) else None)
        )

        gid   = beat_to_group.get(beat_id, 1)
        prior = group_priors.get(gid, {})   # {lead: {field: offset_from_r}}

        # ── Beat-level pre-pass: multi-lead P / T anchor fusion ───────────────
        # Collect peak candidates from every lead; fuse into a single anchor
        # that narrows the per-lead search window below.
        _local_r_bp      = int(r - beat_start)
        _sm_win_bp       = max(3, int(0.015 * fs))
        _p_prior_peak_offset_bp = _median_prior_peak(prior, "p_peak")
        _t_prior_peak_bp = _median_prior_peak(prior, "t_peak")
        _prior_qrs_off_bp = _median_prior_peak(prior, "qrs_off")
        _prior_qrs_on_bp  = _median_prior_peak(prior, "qrs_on")
        # Representative priors are stored as offsets from R, whereas fusion
        # candidates use coordinates local to this beat fragment. Convert only
        # physiologically valid P priors: representative delineation can itself
        # lock onto a QRS-adjacent deflection (for example P=-30, QRS_on=-38
        # ms), and turning that result into a strong beat-local prior would
        # replace one false consensus with another.
        _p_prior_qrs_on_offset_bp = (
            int(_prior_qrs_on_bp)
            if _prior_qrs_on_bp is not None
            else -int(0.020 * fs)
        )
        _p_prior_gap_ms_bp = (
            (_p_prior_qrs_on_offset_bp - int(_p_prior_peak_offset_bp))
            * 1000.0 / fs
            if _p_prior_peak_offset_bp is not None else None
        )
        _p_prior_peak_bp = (
            _local_r_bp + int(_p_prior_peak_offset_bp)
            if _p_prior_peak_offset_bp is not None
            and _p_prior_gap_ms_bp is not None
            and _P_PEAK_QRS_GAP_MIN_MS <= _p_prior_gap_ms_bp <= _P_PEAK_QRS_GAP_MAX_MS
            else None
        )
        _prior_qrs_off_local_bp = (
            _local_r_bp + int(_prior_qrs_off_bp)
            if _prior_qrs_off_bp is not None else None
        )
        _prior_qrs_ms_bp = (
            (int(_prior_qrs_off_bp) - int(_prior_qrs_on_bp)) * 1000.0 / fs
            if _prior_qrs_on_bp is not None and _prior_qrs_off_bp is not None
            and int(_prior_qrs_off_bp) > int(_prior_qrs_on_bp)
            else None
        )
        _p_anchor_qrs_ref_bp = (
            _local_r_bp + int(_prior_qrs_on_bp)
            if _prior_qrs_on_bp is not None
            else max(0, _local_r_bp - int(0.020 * fs))
        )
        _qual_map         = quality if quality is not None else {}

        _p_cands_by_lead: Dict[str, List[Tuple[int, float, float]]] = {}
        _t_cands_by_lead: Dict[str, List[Tuple[int, float, float]]] = {}

        _beat_len_bp = beat_end - beat_start
        for _li_bp, _lead_bp in enumerate(STANDARD_12_LEADS):
            _sig_bp = ecg[_li_bp, beat_start:beat_end].astype(float)
            if _local_r_bp < 0 or _local_r_bp >= len(_sig_bp):
                continue
            _bl_bp, _, _ = _baseline_contract(_sig_bp, _local_r_bp, fs)
            _sm_bp = _edge_preserving_moving_average(_sig_bp, _sm_win_bp)

            _p_lo_bp = max(0, _local_r_bp - p_max_lookback)
            _p_hi_bp = max(0, _local_r_bp - int(0.04 * fs))
            if _p_hi_bp > _p_lo_bp:
                _p_cands_by_lead[_lead_bp] = _candidate_triplets(
                    _sm_bp, lo=_p_lo_bp, hi=_p_hi_bp, baseline=_bl_bp, min_amp=0.008
                )

            # Push T candidates past R+80 ms so the fused anchor does not lock
            # onto the R-wave tail / J-wave region (common in BBB and high
            # R-wave amplitude morphologies where a large peak sits at R+40-70 ms).
            _t_lo_bp = min(_beat_len_bp, _local_r_bp + int(0.08 * fs))
            _t_hi_bp = min(
                _beat_len_bp,
                _local_r_bp + int(0.50 * fs),
                max(_t_lo_bp + 1, (t_cap_global - beat_start) - int(0.04 * fs)),
            )
            if _t_hi_bp > _t_lo_bp:
                _t_cands_by_lead[_lead_bp] = _t_candidate_triplets(
                    _sm_bp,
                    lo=_t_lo_bp,
                    hi=_t_hi_bp,
                    baseline=_bl_bp,
                    fs=fs,
                    min_amp=0.015,
                )

        _fused_p_peak, _fused_p_lead_support = _fuse_peak_anchor(
            _p_cands_by_lead, _qual_map,
            reliable_attr="reliable_for_p",
            prior_peak=_p_prior_peak_bp,
            cluster_radius=max(3, int(0.012 * fs)),
            normalize_per_lead=True,
            qrs_ref=_p_anchor_qrs_ref_bp,
            fs=fs,
            physiologic_peak_gap_ms=(_P_PEAK_QRS_GAP_MIN_MS, _P_PEAK_QRS_GAP_MAX_MS),
            short_peak_gap_ms=_P_SHORT_P_PEAK_QRS_GAP_MS,
        )
        # An anchor that other eligible leads decline to corroborate is not
        # evidence about where this beat's P wave is, so it is withheld from
        # every per-lead use below rather than being gated at one call site
        # only.  Leads with no candidates cannot corroborate anything, so they
        # are excluded from the eligibility count.
        _p_eligible_lead_count = sum(
            1
            for _lead_name, _lead_cands in _p_cands_by_lead.items()
            if _lead_cands
            and getattr(_qual_map.get(_lead_name), "reliable_for_p", False)
        )
        if (_p_eligible_lead_count >= _P_ANCHOR_MIN_LEAD_SUPPORT
                and _fused_p_lead_support < _P_ANCHOR_MIN_LEAD_SUPPORT):
            _fused_p_peak = None
        # T is deliberately left ungated here: its failure mode on QTDB is an
        # ST/T component merge, not an uncorroborated anchor, and mixing the two
        # changes into one run would make neither attributable.
        _fused_t_peak, _ = _fuse_peak_anchor(
            _t_cands_by_lead, _qual_map,
            reliable_attr="reliable_for_qt",
            prior_peak=_t_prior_peak_bp,
            cluster_radius=max(4, int(0.020 * fs)),
        )
        _t_cluster_candidates = candidates_from_triplets(
            _t_cands_by_lead,
            qrs_off=_prior_qrs_off_local_bp,
            fs=fs,
        )
        _cluster_t_polarity = infer_cluster_polarity(
            _t_cluster_candidates,
            expected=None,
        )
        # ─────────────────────────────────────────────────────────────────────

        for li, lead in enumerate(STANDARD_12_LEADS):
            sig     = ecg[li, beat_start:beat_end]
            local_r = int(r - beat_start)
            t_end_conf, t_end_method = 0.0, "threshold"

            if local_r < 0 or local_r >= len(sig):
                continue

            p_on_before_rescue: Optional[int] = None
            p_on_rescued = False
            paced_p_candidate = False
            paced_floor_applied = False
            paced_floor_suppressed_narrow_qrs = False

            # Baseline: median of PR segment
            baseline, baseline_source, baseline_confidence = _baseline_contract(sig, local_r, fs)

            # ── QRS ──────────────────────────────────────────────────────────
            lead_prior = prior.get(lead, {})

            if lead_prior.get("qrs_on") is not None and lead_prior.get("qrs_off") is not None:
                # T019: QRS bounds constrained by rep prior ± LOCAL_MARGIN
                pr_qrs_on  = local_r + lead_prior["qrs_on"]
                pr_qrs_off = local_r + lead_prior["qrs_off"]
                # Run energy-based search, then clip to prior window
                qrs_on_raw, qrs_off_raw, notch_sign, qrs_on_conf, qrs_off_conf = _qrs_bounds(
                    sig,
                    local_r,
                    fs,
                    rr_prev_ms,
                    enable_low_slope_guard=qrs_low_slope_guard,
                )
                qrs_on_clipped  = int(np.clip(qrs_on_raw,  pr_qrs_on  - LOCAL_MARGIN, pr_qrs_on  + LOCAL_MARGIN)) if qrs_on_raw  is not None else None
                qrs_off_clipped = int(np.clip(qrs_off_raw, pr_qrs_off - LOCAL_MARGIN, pr_qrs_off + LOCAL_MARGIN)) if qrs_off_raw is not None else None
                # Safety: clamp to valid signal indices
                qrs_on  = int(np.clip(qrs_on_clipped,  0, len(sig) - 1)) if qrs_on_clipped  is not None else None
                qrs_off = int(np.clip(qrs_off_clipped, 0, len(sig) - 1)) if qrs_off_clipped is not None else None
            else:
                qrs_on, qrs_off, notch_sign, qrs_on_conf, qrs_off_conf = _qrs_bounds(
                    sig,
                    local_r,
                    fs,
                    rr_prev_ms,
                    enable_low_slope_guard=qrs_low_slope_guard,
                )

            # T024 paced-beat branch — Part 1: clamp QRS onset to be no earlier
            # than the spike.  The energy walk on the de-spiked signal should
            # already land after the spike, but if residual interpolation energy
            # pulls it back we prevent it from drifting before the stimulus.
            if beat_id in paced_set and spike_times.size > 0:
                spike_lo = int(r - 0.08 * fs)
                spike_hi = int(r + 0.02 * fs)
                near_spikes = spike_times[(spike_times >= spike_lo) & (spike_times <= spike_hi)]
                if near_spikes.size > 0:
                    spike_local = int(near_spikes[np.argmin(np.abs(near_spikes - r))] - beat_start)
                    if 0 <= spike_local < len(sig):
                        qrs_on = spike_local if qrs_on is None else max(qrs_on, spike_local)
                        qrs_on_conf = max(qrs_on_conf, 0.85)

            # T024 paced-beat branch — Part 2: enforce minimum QRS width (≥ 120 ms).
            # Ventricular paced beats should be wide. If the intrinsic delineation
            # is clearly narrow, the paced label is more likely a QRS-edge artifact,
            # so do not pin that beat to 120 ms.
            if beat_id in paced_set and beat_id in paced_floor_set and qrs_on is not None:
                min_paced_samp = int(_PACED_QRS_FLOOR_MS * fs / 1000.0)
                intrinsic_qrs_ms = (
                    (qrs_off - qrs_on) * 1000.0 / fs
                    if qrs_off is not None and qrs_off > qrs_on
                    else None
                )
                floor_explicitly_confirmed = paced_qrs_floor_beat_ids is not None
                can_apply_paced_floor = (
                    floor_explicitly_confirmed
                    or
                    intrinsic_qrs_ms is None
                    or intrinsic_qrs_ms >= _PACED_QRS_FLOOR_MIN_INTRINSIC_MS
                )
                if qrs_off is None or (qrs_off - qrs_on) < min_paced_samp:
                    if can_apply_paced_floor:
                        qrs_off = min(len(sig) - 1, qrs_on + min_paced_samp)
                        qrs_off_conf = min(qrs_off_conf, 0.60)   # confidence reduced
                        paced_floor_applied = True
                    else:
                        paced_floor_suppressed_narrow_qrs = True

            # Per-lead QRS fiducial within [QRS_on, QRS_off].
            # The global fiducial (local_r) is kept for rhythm/interval analysis;
            # per_lead_r is used for visualization and annotation matching. In
            # negative-dominant QS-like beats, avoid placing it on the late S trough.
            if qrs_on is not None and qrs_off is not None and qrs_off > qrs_on:
                qrs_seg  = sig[qrs_on : qrs_off + 1] - baseline
                per_lead_r = qrs_on + _qrs_fiducial_from_local_waveform(qrs_seg)
                # Clinical R peak (for VAT, Q-wave metrics and r_peak_index).
                # In a positive-dominant complex the tallest positive
                # deflection is the R wave. In a negative-dominant complex it
                # may instead be a terminal R', so the first sufficiently large
                # positive lobe is preferred there -- see _clinical_r_peak_local.
                _positive_max = float(np.max(qrs_seg))
                _negative_max = float(-np.min(qrs_seg))
                if _positive_max > 0.0 and _negative_max > _positive_max:
                    r_pos_local = qrs_on + _clinical_r_peak_local(qrs_seg)
                else:
                    r_pos_local = qrs_on + int(np.argmax(qrs_seg))
            else:
                per_lead_r = local_r
                r_pos_local = local_r

            q_amp, r_amp, s_amp, _ = _qrs_points(sig, qrs_on, per_lead_r, qrs_off, baseline)
            q_component = _q_component_metrics(
                sig,
                qrs_on,
                r_pos_local,
                qrs_off,
                baseline,
                fs,
                r_amp,
            )
            delta_present = _detect_delta_wave(sig, qrs_on, per_lead_r, baseline, fs)

            # Smooth signal (15 ms) used for peak detection to avoid noise spikes
            sm_win  = max(3, int(0.015 * fs))
            sig_sm = _edge_preserving_moving_average(sig, sm_win)

            # ── P wave ────────────────────────────────────────────────────────
            if beat_id in paced_set:
                # T024 paced-beat branch — Part 3: search for retrograde P
                # after the wide QRS instead of before it.
                p_on, p_peak, p_off, _paced_p_conf = _paced_retrograde_p(
                    sig, qrs_off, baseline, fs
                )
                paced_p_candidate = p_peak is not None
            else:
                # Normal path: adaptive lookback so bradycardia P-waves are
                # not missed.
                p_search_lo_default = max(0, local_r - p_max_lookback)
                p_search_hi_default = max(0, local_r - int(0.04 * fs))
                # If the previous beat's T-offset (this lead) is known and
                # falls inside the window, push the floor past it so a
                # decaying T-tail cannot win the abs-peak search ahead of a
                # genuine, smaller P wave (short TP interval / fast rate).
                _prev_t_off_abs = prev_t_off_by_lead.get(lead)
                if _prev_t_off_abs is not None:
                    _masked_p_lo = (_prev_t_off_abs - beat_start) + P_T_TAIL_GUARD
                    if _masked_p_lo > p_search_lo_default:
                        p_search_lo_default = _masked_p_lo
                p_search_lo = p_search_lo_default
                p_search_hi = p_search_hi_default
                if _fused_p_peak is not None:
                    p_search_lo = max(p_search_lo_default, _fused_p_peak - P_PRIOR_MARGIN)
                    p_search_hi = min(p_search_hi_default, _fused_p_peak + P_PRIOR_MARGIN)
                elif lead_prior.get("p_peak") is not None:
                    prior_p = local_r + int(lead_prior["p_peak"])
                    local_lo = max(p_search_lo_default, prior_p - P_PRIOR_MARGIN)
                    local_hi = min(p_search_hi_default, prior_p + P_PRIOR_MARGIN)
                    if local_hi > local_lo:
                        p_search_lo = local_lo
                        p_search_hi = local_hi
                _r_amp_local = abs(float(sig[local_r]) - baseline) + 1e-6
                # Cap the R-relative floor: leads with large QRS (precordial)
                # would otherwise get a proportionally higher P-detection
                # floor than limb leads for no physiological reason, silently
                # suppressing genuine small P waves.
                _min_p_amp_for_anchor = max(0.008, min(_r_amp_local * 0.015, 0.020))
                p_alt_triplets = _candidate_triplets(
                    sig_sm,
                    lo=p_search_lo,
                    hi=p_search_hi,
                    baseline=baseline,
                    min_amp=_min_p_amp_for_anchor,
                    max_candidates=8,
                )
                p_alternatives: List[PCandidateAlternative] = []
                if p_alt_triplets:
                    seen_alt_peaks: set[int] = set()
                    refine_radius = max(1, sm_win // 2)
                    for peak, amp, area in p_alt_triplets:
                        raw_lo = max(0, int(peak) - refine_radius)
                        raw_hi = min(len(sig), int(peak) + refine_radius + 1)
                        raw_peak = int(peak)
                        raw_amp = float(amp)
                        if raw_hi > raw_lo:
                            raw_seg = np.abs(sig[raw_lo:raw_hi] - baseline)
                            raw_peak = raw_lo + int(np.argmax(raw_seg))
                            raw_amp = float(sig[raw_peak] - baseline)
                        if raw_peak in seen_alt_peaks:
                            continue
                        seen_alt_peaks.add(raw_peak)
                        p_alternatives.append(PCandidateAlternative(
                            peak=int(raw_peak + beat_start),
                            amp=raw_amp,
                            area=float(area),
                        ))
                    if p_alternatives:
                        p_alternatives_by_key[(int(beat_id), lead)] = p_alternatives

                # Primary pick: score turning-point candidates by amplitude
                # minus a penalty for deviating from the HR-adjusted expected
                # P-to-R gap, instead of taking the single largest deviation
                # (NeuroKit2's dwt delineator does the analogous thing on its
                # wavelet-scale candidates). A same-lead-and-beat T-tail is
                # often the single largest raw deflection in the window but
                # sits far from the expected PR gap, so a timing-aware score
                # lets a smaller, correctly-timed P candidate win instead of
                # only being reachable via the post-hoc range check below.
                if p_alternatives:
                    _expected_gap_ms = 160.0
                    if rr_prev_ms is not None:
                        _expected_gap_ms = min(max(200.0 * (rr_prev_ms / 1000.0), 60.0), 220.0)
                    _timing_penalty_per_ms = 0.001  # mV-equivalent per ms of gap deviation

                    def _p_alt_score(alt: "PCandidateAlternative") -> float:
                        gap_ms = (local_r - (alt.peak - beat_start)) * 1000.0 / fs
                        return abs(alt.amp) - _timing_penalty_per_ms * abs(gap_ms - _expected_gap_ms)

                    _best_alt = max(p_alternatives, key=_p_alt_score)
                    p_peak = _best_alt.peak - beat_start
                else:
                    p_peak = _find_peak(sig_sm, p_search_lo, p_search_hi, mode="abs")
                p_peak = _select_p_peak_with_fused_anchor(
                    sig_sm,
                    baseline=baseline,
                    p_peak=p_peak,
                    fused_p_peak=_fused_p_peak,
                    search_lo=p_search_lo,
                    search_hi=p_search_hi,
                    qrs_ref=qrs_on if qrs_on is not None else local_r,
                    fs=fs,
                    min_amp=_min_p_amp_for_anchor,
                )

                # ── P-peak position validation ─────────────────────────────────
                # P peak should lie within the physiological PR range
                # [60 ms, 280 ms] before the R peak.  Peaks outside this window
                # are likely the preceding beat's T wave (too far) or J-wave
                # artefact (too close).  When the initial abs-mode pick is out of
                # range, try a constrained search within the valid window.
                if p_peak is not None:
                    _p_off_ms = (local_r - p_peak) * 1000.0 / fs
                    if _p_off_ms < 60 or _p_off_ms > 350:
                        _pr_lo = max(0, local_r - int(0.280 * fs))
                        _pr_hi = max(0, local_r - int(0.060 * fs))
                        if _prev_t_off_abs is not None:
                            _masked_pr_lo = (_prev_t_off_abs - beat_start) + P_T_TAIL_GUARD
                            if _masked_pr_lo > _pr_lo:
                                _pr_lo = _masked_pr_lo
                        if _pr_hi > _pr_lo:
                            _pk_in = _find_peak(sig_sm, _pr_lo, _pr_hi, mode="abs")
                            _min_amp_in = max(0.008, min(_r_amp_local * 0.015, 0.020))
                            _pk_in = _select_p_peak_with_fused_anchor(
                                sig_sm,
                                baseline=baseline,
                                p_peak=_pk_in,
                                fused_p_peak=_fused_p_peak,
                                search_lo=_pr_lo,
                                search_hi=_pr_hi,
                                qrs_ref=qrs_on if qrs_on is not None else local_r,
                                fs=fs,
                                min_amp=_min_amp_in,
                            )
                            if (_pk_in is not None
                                    and abs(float(sig_sm[_pk_in]) - baseline) >= _min_amp_in):
                                p_peak = _pk_in
                            else:
                                p_peak = None
                        else:
                            p_peak = None

                # ── P-peak amplitude validation ───────────────────────────────
                # Reject peaks smaller than 1.5 % of R amplitude (min 8 µV).
                # This catches residual noise spikes without being so aggressive
                # that it silences small-but-real P waves in V1/aVL/III.
                if p_peak is not None:
                    _min_p_amp = max(0.008, min(_r_amp_local * 0.015, 0.020))
                    if abs(float(sig_sm[p_peak]) - baseline) < _min_p_amp:
                        p_peak = None

                # Tangent-intersection boundary detection for P wave.
                # Onset: ascending-limb tangent (reuses _t_onset_tangent).
                # Offset: descending-limb tangent (_p_offset_tangent).
                # Both fall back to threshold walk internally when the arm is too short.
                p_on = _t_onset_tangent(sig, p_peak, p_search_lo, baseline, fs)
                p_on_before_rescue = p_on
                p_on, p_on_rescued = _rescue_long_pr_p_onset(
                    sig, p_peak, p_on, qrs_on, baseline, fs
                )
                p_off_limit = (qrs_on - 1) if qrs_on is not None else max(0, local_r - int(0.04 * fs))
                p_off = _p_offset_tangent(sig, p_peak, p_off_limit, baseline, fs)
                # NOT re-solved against the P-free isoelectric reference: measured
                # on LUDB that makes the onset *worse* (bias +11.8 -> +14.9 ms,
                # MAE 25.9 -> 27.3).  Lowering the level the tangent must reach
                # pushes `x_cross` negative more often, and that branch falls back
                # to the steepest-slope point, which sits later than the tangent
                # crossing it replaces.  The +11.8 ms late onset is therefore not a
                # baseline-offset artifact; the x_cross<0 fallback is the candidate
                # mechanism and needs its own validation against LUDB.
                if p_on is not None and p_peak is not None and p_off is not None:
                    p_on, p_off = _extend_p_candidate_edges(
                        sig=sig,
                        p_on=int(p_on),
                        p_peak=int(p_peak),
                        p_off=int(p_off),
                        baseline=baseline,
                        search_lo=int(p_search_lo),
                        p_off_limit=int(p_off_limit),
                        qrs_on=qrs_on,
                        fs=fs,
                    )
                p_geometry_mode = _p_wave_qrs_geometry_mode(
                    p_on, p_peak, qrs_on
                )
                if p_geometry_mode == "invalid_p":
                    p_on = None
                    p_peak = None
                    p_off = None
                    p_on_rescued = False
                elif p_geometry_mode == "unreliable_pr":
                    p_on = None
                    p_on_rescued = False

            # ── T wave (T010 + T018) ──────────────────────────────────────────
            # Adaptive search end: earlier of default (550 ms) and next-beat cap
            local_t_cap = min(
                len(sig) - 1,
                local_r + int(0.55 * fs),
                t_cap_global - beat_start,   # convert global cap to local coords
            )

            # Ensure T-wave search starts after the QRS offset to prevent the
            # R-wave tail / terminal slur from being mistaken for a T-wave in
            # wide-QRS morphologies (BBB, hypertrophy).  Paced beats use the
            # same 40 ms guard because the paced QRS tail often blends into ST/T.
            # The R+40 ms floor is kept as a failsafe when qrs_off is None.
            _t_lo_base = local_r + int(0.04 * fs)
            if beat_id in paced_set and qrs_off is not None:
                _t_lo_base = max(_t_lo_base, qrs_off + int(0.040 * fs))
            elif qrs_off is not None:
                # Wide-QRS morphologies (LBBB, RBBB, IVCD) have a prolonged
                # S-wave tail that extends past the P75 QRS offset.  Use a
                # 40 ms gap (vs 20 ms for narrow QRS) to prevent the T-search
                # from landing in the S-wave and detecting a wrong-sign T-peak.
                _per_lead_qrs_ms = (
                    (qrs_off - qrs_on) * 1000.0 / fs
                    if qrs_on is not None else 0.0
                )
                _t_gap_ms = 0.040 if _per_lead_qrs_ms > 120.0 else 0.020
                _t_lo_base = max(_t_lo_base, qrs_off + int(_t_gap_ms * fs))
            t_search_lo_default = min(len(sig), _t_lo_base)
            # T-peak must be found before the arm end (leave ≥40 ms descending arm)
            t_search_hi_default = min(
                len(sig),
                local_r + int(0.50 * fs),
                max(t_search_lo_default + 1, local_t_cap - int(0.04 * fs)),
            )
            t_search_lo = t_search_lo_default
            t_search_hi = t_search_hi_default
            if _fused_t_peak is not None:
                t_search_lo = max(t_search_lo_default, _fused_t_peak - T_PRIOR_MARGIN)
                t_search_hi = min(t_search_hi_default, _fused_t_peak + T_PRIOR_MARGIN)
            elif lead_prior.get("t_peak") is not None:
                prior_t = local_r + int(lead_prior["t_peak"])
                local_lo = max(t_search_lo_default, prior_t - T_PRIOR_MARGIN)
                local_hi = min(t_search_hi_default, prior_t + T_PRIOR_MARGIN)
                if local_hi > local_lo:
                    t_search_lo = local_lo
                    t_search_hi = local_hi
            _expected_t_polarity = 1 if lead in _EXPECTED_POSITIVE_T_LEADS else None
            _per_lead_qrs_ms_for_t = (
                (qrs_off - qrs_on) * 1000.0 / fs
                if qrs_on is not None and qrs_off is not None and qrs_off > qrs_on
                else None
            )
            _allow_late_same_polarity_st_rescue = (
                beat_id not in paced_set
                and (
                    _per_lead_qrs_ms_for_t is None
                    or _per_lead_qrs_ms_for_t < _T_LATE_ST_RESCUE_MAX_QRS_MS
                )
                and (
                    _prior_qrs_ms_bp is None
                    or _prior_qrs_ms_bp < _T_LATE_ST_RESCUE_MAX_QRS_MS
                )
            )
            t_measurement = detect_t_wave(
                sig=sig,
                sig_sm=sig_sm,
                baseline=baseline,
                search_lo=t_search_lo,
                search_hi=t_search_hi,
                qrs_off=qrs_off,
                local_t_cap=local_t_cap,
                fs=fs,
                lead=lead,
                expected_polarity=_expected_t_polarity,
                cluster_polarity=_cluster_t_polarity,
                allow_late_same_polarity_st_rescue=_allow_late_same_polarity_st_rescue,
            )
            _t_window_measurement: Optional[TWaveMeasurement] = None
            if (_fused_t_peak is not None
                    and _expected_t_polarity is not None
                    and (t_search_lo != t_search_lo_default
                         or t_search_hi != t_search_hi_default)):
                _t_window_measurement = detect_t_wave(
                    sig=sig,
                    sig_sm=sig_sm,
                    baseline=baseline,
                    search_lo=t_search_lo_default,
                    search_hi=t_search_hi_default,
                    qrs_off=qrs_off,
                    local_t_cap=local_t_cap,
                    fs=fs,
                    lead=lead,
                    expected_polarity=_expected_t_polarity,
                    cluster_polarity=_cluster_t_polarity,
                    allow_late_same_polarity_st_rescue=_allow_late_same_polarity_st_rescue,
                )
            if _should_use_full_t_window_measurement(
                t_measurement,
                _t_window_measurement,
                qrs_on=qrs_on,
                qrs_off=qrs_off,
                prior_qrs_ms=_prior_qrs_ms_bp,
                fs=fs,
            ):
                t_measurement = _t_window_measurement
            t_peak = t_measurement.peak
            if (_fused_t_peak is not None
                    and t_search_lo <= _fused_t_peak < t_search_hi):
                if t_peak is None:
                    t_peak = _fused_t_peak
                else:
                    _t_peak_amp = abs(float(sig_sm[t_peak]) - baseline)
                    _fused_t_amp = abs(float(sig_sm[_fused_t_peak]) - baseline)
                    if _t_peak_amp < 0.015 or np.isclose(
                        _t_peak_amp,
                        _fused_t_amp,
                        rtol=1e-6,
                        atol=1e-12,
                    ):
                        t_peak = _fused_t_peak
            if t_peak is not None:
                _same_complex_tol = max(2, int(0.012 * fs))
                _confusion_reason = None
                for _candidate_measurement in (t_measurement, _t_window_measurement):
                    if (_candidate_measurement is not None
                            and _candidate_measurement.st_t_confusion
                            and _candidate_measurement.peak is not None
                            and abs(_candidate_measurement.peak - t_peak) <= _same_complex_tol):
                        _confusion_reason = _candidate_measurement.confidence_reason
                        break
                if t_measurement.peak != t_peak:
                    _t_amp_for_polarity = float(sig_sm[t_peak] - baseline)
                    if _t_amp_for_polarity > 0.015:
                        _observed_t_polarity = 1
                    elif _t_amp_for_polarity < -0.015:
                        _observed_t_polarity = -1
                    else:
                        _observed_t_polarity = None
                    _, _fused_t_offset = _find_wave_bounds(
                        sig,
                        t_peak,
                        baseline,
                        thresh_frac=0.08,
                    )
                    if _fused_t_offset is not None:
                        _fused_t_offset = min(_fused_t_offset, local_t_cap)
                    _fused_t_abs_amp = abs(_t_amp_for_polarity)
                    _fused_t_conf = (
                        min(1.0, max(0.15, _fused_t_abs_amp / (_fused_t_abs_amp + 0.05)))
                        if _fused_t_abs_amp >= 0.015
                        else 0.0
                    )
                    t_measurement = TWaveMeasurement(
                        peak=t_peak,
                        offset=_fused_t_offset,
                        confidence=_fused_t_conf,
                        method="polarity_cluster_fused_anchor",
                        t_prime_peak=getattr(t_measurement, "t_prime_peak", None),
                        polarity_expected=_expected_t_polarity,
                        polarity_observed=_observed_t_polarity,
                        st_t_confusion=_confusion_reason is not None,
                        confidence_reason=_confusion_reason,
                    )
                elif _confusion_reason is not None and not t_measurement.st_t_confusion:
                    t_measurement = TWaveMeasurement(
                        peak=t_peak,
                        offset=t_measurement.offset,
                        confidence=t_measurement.confidence,
                        method=t_measurement.method,
                        t_prime_peak=getattr(t_measurement, "t_prime_peak", None),
                        polarity_expected=t_measurement.polarity_expected,
                        polarity_observed=t_measurement.polarity_observed,
                        st_t_confusion=True,
                        confidence_reason=_confusion_reason,
                    )

            # T-onset search starts at QRS offset, but never earlier than R+70ms.
            # Without this floor, a correctly-placed (earlier) QRS offset gives the
            # ascending-limb tangent a long arm that can overshoot into the ST
            # segment and push T onset artificially early.
            _t_on_min   = local_r + int(0.080 * fs)
            _t_on_start = max(_t_on_min, qrs_off) if qrs_off is not None else _t_on_min
            t_on = _t_onset_chord(sig, t_peak, _t_on_start, baseline, fs)
            tu_nadir_seen = False

            if t_peak is not None:
                t_off = t_measurement.offset
                if t_off is not None:
                    t_end_conf = t_measurement.confidence
                    t_end_method = t_measurement.method
                _geom_t_off, _geom_t_conf, _geom_t_method = _t_end_geometric(
                    sig, t_peak, local_t_cap, baseline, fs
                )
                if _geom_t_off is not None:
                    t_off, t_end_conf, t_end_method = _geom_t_off, _geom_t_conf, _geom_t_method
                # Biphasic / notched T-wave extension: if the chord method
                # stopped at an intermediate feature (zero-crossing for biphasic,
                # notch bottom for notched), look for a secondary T-wave component
                # and re-run the chord from its peak.
                if t_off is not None:
                    _t_off_ext, _t_label = _t_wave_complex_extend(
                        sig, sig_sm, t_peak, t_off, local_t_cap, baseline, fs
                    )
                    if _t_label:
                        t_off = _t_off_ext
                        t_end_method = f"{t_end_method}_{_t_label}"
                t_off, tu_nadir_seen = _clamp_t_end_to_tu_nadir(
                    sig,
                    sig_sm,
                    t_peak,
                    t_off,
                    baseline,
                    local_t_cap,
                    fs,
                )
                if tu_nadir_seen:
                    t_end_method = f"{t_end_method}_tu_nadir"
                if lead_prior.get("t_off") is not None and t_off is not None:
                    prior_t_off = local_r + int(lead_prior["t_off"])
                    local_lo = max(t_peak + 1, prior_t_off - T_END_PRIOR_MARGIN)
                    local_hi = min(local_t_cap, prior_t_off + T_END_PRIOR_MARGIN)
                    if local_hi > local_lo:
                        local_thr = max(abs(float(sig[t_peak] - baseline)) * 0.04, 1e-4)
                        local_tail = np.abs(sig[local_lo : local_hi + 1] - baseline)
                        crossings = np.where(local_tail <= local_thr)[0]
                        if crossings.size > 0:
                            t_off_local = int(local_lo + crossings[0])
                            # Only apply prior when it extends T-end beyond the chord
                            # result. When the template prior is too early (e.g. because
                            # it was built from another beat with a short-QT error), the
                            # prior window lies entirely before the true T-end and the
                            # 4 % crossing found inside it would shorten QT — the
                            # opposite of the intended correction.
                            if (t_off_local >= t_off and
                                    (qrs_on is None or
                                     (t_off_local - qrs_on) >= int(0.200 * fs))):
                                t_off = t_off_local
                                t_end_conf = max(t_end_conf, 0.30)
                                t_end_method = f"{t_end_method}_prior_local"
                # Physiological sanity: QT < 200 ms is impossible in sinus rhythm;
                # discard the T-end if it gives an unrealistically short interval.
                if (t_off is not None and qrs_on is not None and
                        (t_off - qrs_on) < int(0.200 * fs)):
                    t_off        = None
                    t_end_conf   = 0.0
                    t_end_method = "discarded_short_qt"
            else:
                t_off = None

            if paced_p_candidate:
                def _overlaps(
                    a_on: Optional[int],
                    a_off: Optional[int],
                    b_on: Optional[int],
                    b_off: Optional[int],
                ) -> bool:
                    if a_on is None or a_off is None or b_on is None or b_off is None:
                        return False
                    return min(a_off, b_off) > max(a_on, b_on)

                p_overlaps_qrs = _overlaps(p_on, p_off, qrs_on, qrs_off)
                p_overlaps_t = _overlaps(p_on, p_off, t_on, t_off)
                if p_overlaps_qrs or p_overlaps_t:
                    p_on = None
                    p_peak = None
                    p_off = None
                    p_on_rescued = False

            # ── Interval measurements ─────────────────────────────────────────
            pr_ms     = None
            qrs_ms    = None
            qt_ms     = None
            jt_ms     = None
            qrs_area  = None
            qrs_signed_area = None
            j_index   = None
            st_on_mv  = None
            st_mid_mv = None
            st_80ms_mv= None
            twelve_sl_measurements: Dict[str, object] = {}
            st_j_source = "missing_qrs_bounds"
            st_j_reliable = False
            p_amp     = None
            t_amp     = None

            # P amplitude is referenced to the isoelectric tissue beside the P
            # wave, not to a window the P wave itself sits inside.
            p_amp_baseline, p_amp_baseline_source = _p_isoelectric_reference(
                sig,
                p_on=p_on,
                p_off=p_off,
                qrs_on=qrs_on,
                fs=fs,
                fallback=baseline,
            )
            if p_peak is not None:
                p_amp = float(sig[p_peak] - p_amp_baseline)
            if t_peak is not None:
                t_amp = float(sig[t_peak] - baseline)

            # ── DXL T-U nadir: if a prominent U wave exists, T-end is the ─────
            # trough between T-peak and U-peak, not the Laguna tangent crossing.
            if t_off is not None and t_peak is not None and t_amp is not None:
                _tpol  = 1.0 if t_amp >= 0.0 else -1.0
                _u_lo  = t_off + 1
                _u_hi  = min(len(sig), local_t_cap + 1)
                if _u_hi > _u_lo + int(0.04 * fs):
                    _u_bl   = (sig[_u_lo:_u_hi] - baseline) * _tpol
                    _u_pk_v = float(np.max(_u_bl))
                    # Prominent U wave: amplitude > 40 % of |T-peak|
                    if _u_pk_v > max(0.04, 0.40 * abs(t_amp)):
                        _u_pk_abs = _u_lo + int(np.argmax(_u_bl))
                        # Trough between current t_off and U-peak
                        _trough = (sig[t_off : _u_pk_abs + 1] - baseline) * _tpol
                        if len(_trough) > 2:
                            _nadir_local = int(np.argmin(_trough))
                            _t_off_nadir = t_off + _nadir_local
                            # Accept only if it keeps QT physiologically plausible
                            if (qrs_on is None or
                                    (_t_off_nadir - qrs_on) >= int(0.200 * fs)):
                                t_off = _t_off_nadir
                                tu_nadir_seen = True

            if qrs_on is not None and qrs_off is not None:
                qrs_ms    = (qrs_off - qrs_on) * 1000.0 / fs
                qrs_area  = float(trapezoid(np.abs(sig[qrs_on : qrs_off + 1] - baseline)))
                qrs_signed_area = float(trapezoid(sig[qrs_on : qrs_off + 1] - baseline))
                j_index   = qrs_off + beat_start
                st_on_mv, st_j_source, st_j_reliable = _measure_st_j_with_guard(
                    sig=sig,
                    qrs_on=qrs_on,
                    qrs_off=qrs_off,
                    baseline=baseline,
                    fs=fs,
                )
                st_mid    = min(len(sig) - 1, qrs_off + int(0.04 * fs))
                st_80     = min(len(sig) - 1, qrs_off + int(0.08 * fs))
                st_mid_mv = float(sig[st_mid] - baseline)
                st_80ms_mv= float(sig[st_80] - baseline)
                twelve_sl_measurements = twelve_sl_wave_measurements_from_signal(
                    sig=sig,
                    qrs_on=qrs_on,
                    qrs_off=qrs_off,
                    t_peak=t_peak,
                    t_off=t_off,
                    avg_rr_ms=avg_rr_12sl_ms,
                    fs=fs,
                    t_prime_peak=getattr(t_measurement, "t_prime_peak", None),
                    qrs_offset_confidence=qrs_off_conf,
                    qrs_offset_repaired=False,
                )
            if p_on is not None and qrs_on is not None and not paced_p_candidate:
                pr_ms = (qrs_on - p_on) * 1000.0 / fs
            if qrs_on is not None and t_off is not None:
                qt_ms = (t_off - qrs_on) * 1000.0 / fs
            if qrs_off is not None and t_off is not None:
                jt_ms = (t_off - qrs_off) * 1000.0 / fs

            # ── T013: QRS components ──────────────────────────────────────────
            r_prime_amp, s_prime_amp, qrs_num_peaks, qrs_notch_count, qrs_component_durations = _qrs_components(
                sig, qrs_on, per_lead_r, qrs_off, baseline, fs
            )

            # ── T014: Ventricular activation time ────────────────────────────
            # VAT = QRS onset → positive R peak (clinical convention).
            # r_pos_local is argmax of positive deflection within [qrs_on, qrs_off],
            # so it correctly picks the r wave even in predominantly negative leads.
            vat_ms = (r_pos_local - qrs_on) * 1000.0 / fs if qrs_on is not None else None

            # ── T026: Additional morphology fields ────────────────────────────
            p_dur_ms = None
            p_area   = None
            p_signed_area = None
            if p_on is not None and p_off is not None and p_off > p_on:
                p_dur_ms = (p_off - p_on) * 1000.0 / fs
                p_area   = float(trapezoid(np.abs(sig[p_on : p_off + 1] - p_amp_baseline)))
                p_signed_area = float(trapezoid(sig[p_on : p_off + 1] - p_amp_baseline))

            t_dur_ms = None
            t_area   = None
            t_signed_area = None
            if t_on is not None and t_off is not None and t_off > t_on:
                t_dur_ms = (t_off - t_on) * 1000.0 / fs
                t_area   = float(trapezoid(np.abs(sig[t_on : t_off + 1] - baseline)))
                t_signed_area = float(trapezoid(sig[t_on : t_off + 1] - baseline))

            t_polarity = 0
            if t_amp is not None:
                if t_amp > 0.05:
                    t_polarity = 1
                elif t_amp < -0.05:
                    t_polarity = -1

            t_symmetry = None
            if (
                t_on is not None
                and t_peak is not None
                and t_off is not None
                and t_on < t_peak < t_off
            ):
                rise = float(t_peak - t_on)
                decay = float(t_off - t_peak)
                if decay > 0.0:
                    t_symmetry = rise / decay

            pr_segment_level_mv = None
            if (
                p_on is not None
                and p_off is not None
                and qrs_on is not None
                and p_off < qrs_on
            ):
                pr_lo = int(p_off)
                pr_hi = int(qrs_on)
                tp_hi = max(0, int(p_on) - max(1, int(0.010 * fs)))
                tp_lo = max(0, tp_hi - max(2, int(0.040 * fs)))
                if pr_hi - pr_lo >= 2 and tp_hi - tp_lo >= 2:
                    pr_segment_level_mv = float(
                        np.median(sig[pr_lo:pr_hi])
                        - np.median(sig[tp_lo:tp_hi])
                    )

            # U-wave flag: secondary peak in [t_off+1, t_off+200ms]
            u_wave_flag = tu_nadir_seen
            u_peak_val = None
            if t_off is not None:
                u_lo = t_off + 1
                u_hi = min(len(sig), t_off + int(0.20 * fs))
                if u_hi > u_lo + 2:
                    u_seg = sig[u_lo:u_hi]
                    u_peak_val = float(np.max(np.abs(u_seg - baseline)))
                    u_wave_flag = u_wave_flag or u_peak_val > 0.05

            # Signed U-wave measurement (separate from the unsigned flag above,
            # which stays as-is because the T-offset fusion path depends on it).
            u_measurement = measure_u_wave(
                sig,
                t_off=t_off,
                baseline=baseline,
                fs=fs,
                search_cap=local_t_cap,
            )

            # QRS slur flag
            qrs_slur_flag = qrs_notch_count >= 2

            # ST slope: J-point to J+80ms
            st_slope_mv_per_ms = None
            if qrs_off is not None:
                j_local = qrs_off
                j_80 = j_local + int(0.08 * fs)
                if j_80 < len(sig):
                    st_slope_mv_per_ms = (float(sig[j_80]) - float(sig[j_local])) / 80.0

            # ── T027: Extended measurements ───────────────────────────────────
            # Tpe (T-peak to T-end): marker of transmural repolarisation dispersion
            tpe_ms = (
                (t_off - t_peak) * 1000.0 / fs
                if (t_peak is not None and t_off is not None and t_off > t_peak)
                else None
            )
            # ST morphology classification
            st_morphology_cls = _classify_st_morphology(st_slope_mv_per_ms)
            # Fragmented-QRS score
            fqrs_val = _fqrs_score(qrs_notch_count, qrs_num_peaks)

            # ── T004: Beat-level quality ──────────────────────────────────────
            rep_qrs_snippet = None
            if rep_beats and gid in rep_beats and li < rep_beats[gid].shape[0]:
                rep_sig_lead = rep_beats[gid][li]
                rep_local_r  = int(rep_left_ms * fs / 1000)
                pr_on  = lead_prior.get("qrs_on")
                pr_off = lead_prior.get("qrs_off")
                if pr_on is not None and pr_off is not None:
                    rq_on  = max(0, rep_local_r + pr_on)
                    rq_off = min(len(rep_sig_lead) - 1, rep_local_r + pr_off)
                    rep_bl = float(lead_prior.get("baseline", 0.0))
                    rep_qrs_snippet = rep_sig_lead[rq_on : rq_off + 1] - rep_bl

            b_noise, b_bshift, b_tcorr, b_reliable = _beat_quality(
                sig, per_lead_r, qrs_on, qrs_off, baseline, fs, rep_qrs_snippet
            )

            # ── T020: P and QRS boundary confidence ──────────────────────────
            # P confidence: amplitude-to-noise ratio, capped at 1.0
            if p_peak is not None and p_amp is not None:
                noise_floor = b_noise * (abs(float(sig[per_lead_r]) - baseline) + 1e-6)
                p_confidence = float(np.clip(abs(p_amp) / (noise_floor + 1e-4), 0.0, 1.0))
            else:
                p_confidence = 0.0

            if p_on_rescued and p_confidence < 0.30:
                p_on = p_on_before_rescue
                pr_ms = (qrs_on - p_on) * 1000.0 / fs if p_on is not None and qrs_on is not None else None
                p_dur_ms = None
                p_area = None
                p_signed_area = None
                if p_on is not None and p_off is not None and p_off > p_on:
                    p_dur_ms = (p_off - p_on) * 1000.0 / fs
                    p_area = float(trapezoid(np.abs(sig[p_on : p_off + 1] - p_amp_baseline)))
                    p_signed_area = float(trapezoid(sig[p_on : p_off + 1] - p_amp_baseline))

            # QRS confidence: template correlation × (1 - noise_score)
            if b_tcorr > 0.0:
                qrs_confidence = float(np.clip(b_tcorr * (1.0 - b_noise), 0.0, 1.0))
            else:
                qrs_confidence = float(np.clip(1.0 - b_noise, 0.0, 1.0)) if qrs_on is not None else 0.0

            # P-terminal force (V1 lead only); gated on final P-wave confidence
            # to avoid inflated values from mis-detected P boundaries.
            ptf_v1 = (
                _ptf_v1(sig, p_on, p_off, p_amp_baseline, fs)
                if lead == "V1" and p_confidence > 0.30
                else None
            )
            p_components = measure_p_components(
                sig,
                p_on=p_on,
                p_peak=p_peak,
                p_off=p_off,
                baseline=p_amp_baseline,
                fs=fs,
            )
            glasgow_measurements = measure_glasgow_profile(
                sig,
                fs=fs,
                beat_window_start_index=beat_start,
                p_on=p_on,
                p_off=p_off,
                qrs_on=qrs_on,
                qrs_off=qrs_off,
                t_on=t_on,
                t_off=t_off,
                r_peak=per_lead_r,
                baseline_mv=baseline,
                r_prime_amp_baseline_mv=r_prime_amp,
                s_prime_amp_baseline_mv=s_prime_amp,
                component_durations={
                    "q_duration_ms": q_component["q_duration_ms"],
                    "r_duration_ms": qrs_component_durations["r_duration_ms"],
                    "s_duration_ms": qrs_component_durations["s_duration_ms"],
                    "r_prime_duration_ms": qrs_component_durations["r_prime_duration_ms"],
                    "s_prime_duration_ms": qrs_component_durations["s_prime_duration_ms"],
                },
                vat_ms=vat_ms,
                qt_ms=qt_ms,
                delta_present=delta_present,
                delta_confidence=qrs_confidence if delta_present else None,
                qrs_notch_count=qrs_notch_count,
            )
            glasgow_measurements["representative_beat_id"] = int(beat_id)
            glasgow_measurements["measurement_source"] = "beat_delineation"

            # ── Flags ─────────────────────────────────────────────────────────
            flags: List[str] = []
            if p_peak is None:
                flags.append("p_unreliable")
            if t_peak is None:
                flags.append("t_unreliable")
            if t_peak is not None and t_measurement.st_t_confusion:
                flags.append("st_t_confusion")
            if qrs_on is None or qrs_off is None:
                flags.append("qrs_unreliable")
            if t_end_method == "threshold_fallback":
                flags.append("t_end_fallback")
            if not b_reliable:
                flags.append("beat_unreliable")
            if beat_id in paced_set:
                flags.append("paced_beat")
                if paced_floor_applied:
                    flags.append("paced_floor_applied")
                if paced_floor_suppressed_narrow_qrs:
                    flags.append("paced_floor_suppressed_narrow_qrs")
                if p_peak is not None:
                    flags.append("retrograde_p")
            if st_j_source == "qrs_tail_guard" and not st_j_reliable:
                flags.append("st_j_unreliable")

            prev_t_off_by_lead[lead] = (
                int(t_off + beat_start) if t_off is not None else None
            )

            out.append(LeadBeatFeatures(
                lead=lead,
                beat_id=int(beat_id),
                p=WaveBounds(
                    onset=None if p_on    is None else p_on    + beat_start,
                    peak= None if p_peak  is None else p_peak  + beat_start,
                    offset=None if p_off  is None else p_off   + beat_start,
                ),
                qrs=WaveBounds(
                    onset=None if qrs_on  is None else qrs_on  + beat_start,
                    peak=per_lead_r + beat_start,   # beat-detection/alignment fiducial; may sit on the S trough for negative-dominant complexes
                    offset=None if qrs_off is None else qrs_off + beat_start,
                ),
                t=WaveBounds(
                    onset=None if t_on    is None else t_on    + beat_start,
                    peak= None if t_peak  is None else t_peak  + beat_start,
                    offset=None if t_off  is None else t_off   + beat_start,
                ),
                qt_ms=qt_ms,
                pr_ms=pr_ms,
                qrs_ms=qrs_ms,
                p_amp_mv=p_amp,
                qrs_area=qrs_area,
                baseline_source=baseline_source,
                baseline_confidence=baseline_confidence,
                qrs_signed_area=qrs_signed_area,
                q_onset=None if q_component["q_onset"] is None else int(q_component["q_onset"]) + beat_start,
                q_offset=None if q_component["q_offset"] is None else int(q_component["q_offset"]) + beat_start,
                q_duration_ms=q_component["q_duration_ms"],
                q_area_mv_ms=q_component["q_area_mv_ms"],
                q_r_ratio=q_component["q_r_ratio"],
                initial_qrs_area_mv_ms=q_component["initial_qrs_area_mv_ms"],
                initial_qrs_net_mv=q_component["initial_qrs_net_mv"],
                p_signed_area=p_signed_area,
                t_signed_area=t_signed_area,
                q_amp_mv=q_amp,
                r_amp_mv=r_amp,
                s_amp_mv=s_amp,
                r_peak_index=r_pos_local + beat_start,
                st_on_mv=st_on_mv,
                st_mid_mv=st_mid_mv,
                st_80ms_mv=st_80ms_mv,
                t_amp_mv=t_amp,
                j_index=j_index,
                beat_window_start_index=beat_start,
                glasgow_measurements=glasgow_measurements,
                twelve_sl_stj_mv=twelve_sl_measurements.get("stj_mv"),
                twelve_sl_stm_mv=twelve_sl_measurements.get("stm_mv"),
                twelve_sl_ste_mv=twelve_sl_measurements.get("ste_mv"),
                twelve_sl_stm_offset_ms=twelve_sl_measurements.get("stm_offset_ms"),
                twelve_sl_ste_offset_ms=twelve_sl_measurements.get("ste_offset_ms"),
                twelve_sl_qrs_area_uv_ms=twelve_sl_measurements.get("qrs_area_uv_ms"),
                twelve_sl_qrs_signed_area_uv_ms=twelve_sl_measurements.get("qrs_signed_area_uv_ms"),
                twelve_sl_qrs_balance_uv=twelve_sl_measurements.get("qrs_balance_uv"),
                twelve_sl_qrs_deflection_uv=twelve_sl_measurements.get("qrs_deflection_uv"),
                twelve_sl_minimum_st_uv=twelve_sl_measurements.get("minimum_st_uv"),
                twelve_sl_special_t_uv=twelve_sl_measurements.get("special_t_uv"),
                twelve_sl_t_prime_uv=twelve_sl_measurements.get("t_prime_amp_uv"),
                twelve_sl_t_prime_area_uv_ms=twelve_sl_measurements.get("t_prime_area_uv_ms"),
                twelve_sl_st_confidence=twelve_sl_measurements.get("st_confidence"),
                twelve_sl_st_confidence_reason=twelve_sl_measurements.get("st_confidence_reason"),
                twelve_sl_qrs_significant=bool(twelve_sl_measurements.get("qrs_significant", False)),
                delta_present=delta_present,
                qrs_notch_sign=notch_sign if qrs_on is not None else 0,
                flags=flags,
                jt_ms=jt_ms,
                qt_confidence=t_end_conf,
                t_end_method=t_end_method,
                beat_noise_score=b_noise,
                beat_baseline_shift=b_bshift,
                beat_template_corr=b_tcorr,
                beat_measurement_reliable=b_reliable,
                p_confidence=p_confidence,
                qrs_confidence=qrs_confidence,
                qrs_on_confidence=qrs_on_conf,
                qrs_off_confidence=qrs_off_conf,
                r_prime_amp_mv=r_prime_amp,
                s_prime_amp_mv=s_prime_amp,
                r_duration_ms=qrs_component_durations["r_duration_ms"],
                r_prime_duration_ms=qrs_component_durations["r_prime_duration_ms"],
                s_duration_ms=qrs_component_durations["s_duration_ms"],
                s_prime_duration_ms=qrs_component_durations["s_prime_duration_ms"],
                qrs_num_peaks=qrs_num_peaks,
                qrs_notch_count=qrs_notch_count,
                vat_ms=vat_ms,
                p_dur_ms=p_dur_ms,
                p_area=p_area,
                p_notched=bool(p_components["is_notched"]),
                p_biphasic=bool(p_components["is_biphasic"]),
                p_notch_interval_ms=p_components["notch_interval_ms"],
                p_initial_duration_ms=p_components["initial_duration_ms"],
                p_initial_amp_mv=p_components["initial_amplitude_mV"],
                p_terminal_duration_ms=p_components["terminal_duration_ms"],
                p_terminal_amp_mv=p_components["terminal_amplitude_mV"],
                p_terminal_area_mv_ms=p_components["terminal_area_mv_ms"],
                t_dur_ms=t_dur_ms,
                t_area=t_area,
                t_polarity=t_polarity,
                u_wave_flag=u_wave_flag,
                u_amp_mv=u_peak_val,
                u_amp_signed_mv=u_measurement.amplitude_mv,
                u_polarity=u_measurement.polarity,
                u_prominence_mv=u_measurement.prominence_mv,
                u_dur_ms=u_measurement.duration_ms,
                u_isoelectric_gap_ms=u_measurement.isoelectric_gap_ms,
                u_peak_index=(
                    None
                    if u_measurement.peak_index is None
                    else int(u_measurement.peak_index + beat_start)
                ),
                u_measurement_reliable=u_measurement.reliable,
                u_reject_reason=u_measurement.reject_reason,
                t_symmetry=t_symmetry,
                pr_segment_level_mv=pr_segment_level_mv,
                qrs_slur_flag=qrs_slur_flag,
                st_slope_mv_per_ms=st_slope_mv_per_ms,
                tpe_ms=tpe_ms,
                st_morphology=st_morphology_cls,
                fqrs_score=fqrs_val,
                ptf_v1_mv_ms=ptf_v1,
                qt_consensus_ms=None,   # filled by _apply_multilead_consensus
                pr_consensus_ms=None,   # filled by _apply_multilead_consensus
                t_peak_path=t_measurement.method if t_peak is not None else None,
                t_polarity_expected=t_measurement.polarity_expected,
                t_polarity_observed=t_measurement.polarity_observed,
                st_t_confusion=t_measurement.st_t_confusion,
                t_confidence_reason=t_measurement.confidence_reason,
            ))

    if out:
        out = _apply_p_candidate_context(
            out,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group=beat_to_group,
            suppress=False,
        )
        if len(r_locs) >= 3:
            qrs_offsets_for_raw: List[int] = []
            for beat_id in range(len(r_locs)):
                offsets = [
                    int(feature.qrs.offset)
                    for feature in out
                    if int(feature.beat_id) == int(beat_id)
                    and feature.qrs.offset is not None
                ]
                qrs_offsets_for_raw.append(
                    int(round(float(np.median(offsets))))
                    if offsets else int(r_locs[beat_id])
                )
            raw_p_events, raw_p_meta = _composite_raw_p_detection(
                np.asarray(ecg, dtype=float),
                fs,
                np.asarray(r_locs, dtype=int),
                np.asarray(qrs_offsets_for_raw, dtype=int),
                quality or {},
            )
            if raw_p_meta.get("available") and raw_p_events:
                _augment_p_alternatives_from_raw_atrial_events(
                    out,
                    raw_p_events,
                    fs=fs,
                    quality=quality,
                    alternatives_by_key=p_alternatives_by_key,
                )
        out = _apply_p_candidate_reselection(
            out,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group=beat_to_group,
            alternatives_by_key=p_alternatives_by_key,
        )
        out = _apply_p_candidate_context(
            out,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group=beat_to_group,
            suppress=True,
        )
        out = _refresh_p_boundary_diagnostics(
            out,
            ecg=ecg,
            fs=fs,
        )

    # T025: Multi-lead P-onset / T-end consensus (second pass)
    consensus_quality = quality
    if consensus_quality is None and out and len(r_locs) == 1:
        class _RepresentativeConsensusQuality:
            reliable_for_p = True
            reliable_for_qrs = True
            reliable_for_t = True
            reliable_for_qt = True

        consensus_quality = {
            lead: _RepresentativeConsensusQuality()
            for lead in {bf.lead for bf in out}
        }
    if consensus_quality is not None and out:
        out = _apply_qrs_offset_raw_repair(
            out,
            ecg=ecg,
            fs=fs,
            quality=consensus_quality,
            r_locs=r_locs,
        )
        out = _apply_st_j_raw_remeasurement(
            out,
            ecg=ecg,
            fs=fs,
            quality=consensus_quality,
            r_locs=r_locs,
        )
        out = _apply_t_dual_method_raw_rescue(
            out,
            ecg=ecg,
            fs=fs,
            quality=consensus_quality,
            r_locs=r_locs,
        )
        out = _repair_t_end_outliers(out, ecg, fs, consensus_quality, r_locs)
        out = _apply_multilead_consensus(out, fs, consensus_quality, r_locs=r_locs, ecg=ecg)

    return out
