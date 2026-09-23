from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .models import LeadBeatFeatures, STANDARD_12_LEADS
from .numeric import trapezoid


@dataclass(frozen=True)
class STLocalization:
    """Parallel, robust ST/J measurement for one lead and one beat."""

    j_index: int | None
    j_mv: float | None
    st20_mv: float | None
    st40_mv: float | None
    st60_mv: float | None
    st80_mv: float | None
    adaptive_index: int | None
    adaptive_mv: float | None
    mean_mv: float | None
    area_mv_ms: float | None
    slope_mv_per_ms: float | None
    curvature_mv_per_ms2: float | None
    trend: str
    shape: str
    baseline_mv: float | None
    baseline_source: str
    baseline_confidence: float
    j_method: str
    j_confidence: float
    consensus_index: int | None
    consensus_support: int
    noise_mv: float | None
    reliable: bool
    unreliable_reason: str | None


def _median_1d(values: np.ndarray) -> float:
    """Exact one-dimensional median without NumPy's generic reducer overhead."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    midpoint = arr.size // 2
    if arr.size % 2:
        return float(np.partition(arr, midpoint)[midpoint])
    partitioned = np.partition(arr, (midpoint - 1, midpoint))
    return float(0.5 * (partitioned[midpoint - 1] + partitioned[midpoint]))


def _finite_index(value: Any, n_samples: int) -> int | None:
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return index if 0 <= index < n_samples else None


def _smooth(signal: np.ndarray, fs: int, width_ms: float = 6.0) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    width = max(1, int(round(width_ms * fs / 1000.0)))
    if width <= 1:
        return values.copy()
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def _mad_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    center = _median_1d(finite)
    return max(0.0, 1.4826 * _median_1d(np.abs(finite - center)))


def _window_median(
    signal: np.ndarray,
    center: int,
    *,
    radius: int,
) -> float | None:
    lo = max(0, int(center) - max(0, int(radius)))
    hi = min(len(signal), int(center) + max(0, int(radius)) + 1)
    if hi <= lo:
        return None
    values = np.asarray(signal[lo:hi], dtype=float)
    values = values[np.isfinite(values)]
    return _median_1d(values) if values.size else None


def _baseline_candidate(
    signal: np.ndarray,
    lo: int,
    hi: int,
    *,
    fs: int,
    source: str,
    source_confidence: float,
) -> tuple[float, str, float, float] | None:
    lo = max(0, int(lo))
    hi = min(len(signal), int(hi))
    minimum = max(3, int(round(0.012 * fs)))
    if hi - lo < minimum:
        return None
    segment = np.asarray(signal[lo:hi], dtype=float)
    segment = segment[np.isfinite(segment)]
    if segment.size < minimum:
        return None
    baseline = _median_1d(segment)
    noise = _mad_sigma(segment)
    noise_score = float(np.clip(1.0 - noise / 0.060, 0.0, 1.0))
    confidence = float(np.clip(0.75 * source_confidence + 0.25 * noise_score, 0.0, 1.0))
    return baseline, source, confidence, noise


def _st_baseline(
    signal: np.ndarray,
    *,
    fs: int,
    r_index: int,
    qrs_onset: int,
    p_offset: int | None,
) -> tuple[float, str, float, float]:
    """Prefer the electrically quiet PR segment and avoid the P-wave itself."""

    gap = max(1, int(round(0.008 * fs)))
    if p_offset is not None and int(p_offset) < int(qrs_onset) - 2 * gap:
        candidate = _baseline_candidate(
            signal,
            int(p_offset) + gap,
            int(qrs_onset) - gap,
            fs=fs,
            source="post_p_pr_segment",
            source_confidence=0.95,
        )
        if candidate is not None:
            return candidate

    candidate = _baseline_candidate(
        signal,
        int(qrs_onset) - int(round(0.070 * fs)),
        int(qrs_onset) - int(round(0.015 * fs)),
        fs=fs,
        source="pre_qrs_segment",
        source_confidence=0.72,
    )
    if candidate is not None:
        return candidate

    candidate = _baseline_candidate(
        signal,
        int(r_index) - int(round(0.220 * fs)),
        int(r_index) - int(round(0.080 * fs)),
        fs=fs,
        source="pre_r_fallback",
        source_confidence=0.45,
    )
    if candidate is not None:
        return candidate
    return 0.0, "zero_fallback", 0.10, 0.0


def _effective_native_j(feature: LeadBeatFeatures) -> int | None:
    repaired = getattr(feature, "st_j_remeasured_index", None)
    if repaired is not None:
        return int(repaired)
    qrs_offset = getattr(getattr(feature, "qrs", None), "offset", None)
    if qrs_offset is not None:
        return int(qrs_offset)
    j_index = getattr(feature, "j_index", None)
    if j_index is not None:
        return int(j_index)
    return None


def _localize_j(
    signal: np.ndarray,
    *,
    fs: int,
    r_index: int,
    qrs_onset: int,
    native_j: int,
    consensus_j: int | None,
    t_onset: int | None,
    smoothed_signal: np.ndarray | None = None,
    signal_derivative: np.ndarray | None = None,
) -> tuple[int | None, float, str]:
    """Find QRS-to-ST settling while keeping native and consensus anchors."""

    n_samples = len(signal)
    margin = max(2, int(round(0.030 * fs)))
    lo = max(
        1,
        int(r_index) + int(round(0.010 * fs)),
        int(native_j) - margin,
        int(qrs_onset) + int(round(0.035 * fs)),
    )
    hi = min(n_samples - 2, int(native_j) + margin)
    if consensus_j is not None:
        lo = max(lo, int(consensus_j) - margin)
        hi = min(hi, int(consensus_j) + margin)
    if t_onset is not None:
        hi = min(hi, int(t_onset) - int(round(0.020 * fs)))
    if hi <= lo:
        return int(native_j), 0.20, "native_consensus_disagreement"

    smoothed = (
        np.asarray(smoothed_signal, dtype=float)
        if smoothed_signal is not None
        else _smooth(signal, fs)
    )
    derivative = (
        np.asarray(signal_derivative, dtype=float)
        if signal_derivative is not None
        else np.diff(smoothed)
    )
    local_derivative = derivative[max(0, lo - margin) : min(len(derivative), hi + margin)]
    derivative_floor = max(_mad_sigma(local_derivative), 1e-6)
    pre_width = max(2, int(round(0.014 * fs)))
    post_gap = max(1, int(round(0.004 * fs)))
    post_width = max(3, int(round(0.024 * fs)))
    alignment_scale = max(1.0, float(0.014 * fs))

    best_index: int | None = None
    best_score = -1.0
    for candidate in range(lo, hi + 1):
        pre = derivative[max(0, candidate - pre_width) : max(1, candidate - 1)]
        post = derivative[
            min(len(derivative), candidate + post_gap) :
            min(len(derivative), candidate + post_gap + post_width)
        ]
        if pre.size < 2 or post.size < 2:
            continue
        pre_energy = _median_1d(np.abs(pre))
        post_energy = _median_1d(np.abs(post))
        settling = float(np.clip(pre_energy / max(pre_energy + post_energy, 1e-9), 0.0, 1.0))
        flatness = float(
            np.clip(
                1.0 - post_energy / max(post_energy + 3.0 * derivative_floor, 1e-9),
                0.0,
                1.0,
            )
        )
        native_alignment = float(np.exp(-abs(candidate - native_j) / alignment_scale))
        consensus_alignment = (
            float(np.exp(-abs(candidate - consensus_j) / alignment_scale))
            if consensus_j is not None
            else native_alignment
        )
        score = (
            0.34 * native_alignment
            + 0.26 * consensus_alignment
            + 0.25 * settling
            + 0.15 * flatness
        )
        if score > best_score:
            best_score = score
            best_index = candidate

    if best_index is None:
        return int(native_j), 0.20, "native_fallback"
    method = "local_settling_consensus" if consensus_j is not None else "local_settling_native"
    return int(best_index), float(np.clip(best_score, 0.0, 1.0)), method


def _robust_linear_slope(
    signal: np.ndarray,
    *,
    lo: int,
    hi: int,
    baseline: float,
    fs: int,
) -> float | None:
    lo = max(0, int(lo))
    hi = min(len(signal), int(hi))
    if hi - lo < max(4, int(round(0.028 * fs))):
        return None
    y = np.asarray(signal[lo:hi], dtype=float) - float(baseline)
    x = np.arange(y.size, dtype=float) * (1000.0 / float(fs))
    finite = np.isfinite(y)
    if int(np.sum(finite)) < 4:
        return None
    x = x[finite]
    y = y[finite]
    median = _median_1d(y)
    sigma = _mad_sigma(y)
    if sigma > 0:
        keep = np.abs(y - median) <= 3.5 * sigma
        if int(np.sum(keep)) >= 4:
            x = x[keep]
            y = y[keep]
    if y.size < 4 or float(np.ptp(x)) <= 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _robust_quadratic_curvature(
    signal: np.ndarray,
    *,
    lo: int,
    hi: int,
    baseline: float,
    fs: int,
) -> float | None:
    """Return the second derivative of a robust quadratic ST fit."""

    lo = max(0, int(lo))
    hi = min(len(signal), int(hi))
    if hi - lo < max(5, int(round(0.036 * fs))):
        return None
    y = np.asarray(signal[lo:hi], dtype=float) - float(baseline)
    x = np.arange(y.size, dtype=float) * (1000.0 / float(fs))
    finite = np.isfinite(y)
    if int(np.sum(finite)) < 5:
        return None
    x = x[finite]
    y = y[finite]
    keep = np.ones(y.size, dtype=bool)
    coefficients: np.ndarray | None = None
    for _ in range(3):
        if int(np.sum(keep)) < 5:
            break
        coefficients = np.polyfit(x[keep], y[keep], 2)
        residual = y - np.polyval(coefficients, x)
        sigma = _mad_sigma(residual[keep])
        if sigma <= 0.0:
            break
        updated = np.abs(residual) <= 3.5 * sigma
        if np.array_equal(updated, keep):
            break
        keep = updated
    if coefficients is None:
        return None
    return float(2.0 * coefficients[0])


def _st_trend(slope_mv_per_ms: float | None) -> str:
    if slope_mv_per_ms is None:
        return "unknown"
    if slope_mv_per_ms > 0.0005:
        return "upsloping"
    if slope_mv_per_ms < -0.0005:
        return "downsloping"
    return "horizontal"


def _st_shape(
    j_mv: float | None,
    st40_mv: float | None,
    st80_mv: float | None,
    *,
    noise_mv: float,
) -> str:
    if j_mv is None or st40_mv is None or st80_mv is None:
        return "unknown"
    curvature = float(st80_mv - 2.0 * st40_mv + j_mv)
    threshold = max(0.020, 3.0 * float(noise_mv))
    if curvature > threshold:
        return "concave-up"
    if curvature < -threshold:
        return "concave-down"
    return "straight"


# ── Named ST-segment morphology classes ──────────────────────────────────────
# `trend` and `shape` above are geometric primitives. The four classes below
# are the named categories clinical texts actually reason about, and they are
# not a relabelling of `trend`: the J-point class cuts across trend (it is
# defined by an elevated, notched/slurred J point that may then run flat or
# upsloping), and separating it from the plain upsloping class is what keeps
# benign early repolarisation from reading as an ischemic ST segment.
#
#   j_point_elevation  J型     elevated notched/slurred J point, benign
#                              early-repolarisation morphology
#   slow_upsloping     緩徐上行型 ST rises gradually into a normal T wave;
#                              usually normal variant or rate-related
#   horizontal         水平型   ST parallel to baseline; ischemia-suspicious,
#                              especially with symmetric T inversion
#   downsloping        下行傾斜型 ST descends from the J point; highest
#                              specificity for ischemia, also digitalis effect
ST_PATTERN_J_POINT = "j_point_elevation"
ST_PATTERN_UPSLOPING = "slow_upsloping"
ST_PATTERN_HORIZONTAL = "horizontal"
ST_PATTERN_DOWNSLOPING = "downsloping"
ST_PATTERN_UNKNOWN = "unknown"

# J-point elevation required before the benign early-repolarisation class can
# be assigned (1 mm, the conventional threshold).
ST_J_POINT_ELEVATION_MV = 0.10


def classify_st_pattern(
    *,
    j_mv: float | None,
    trend: str,
    shape: str,
    notched_or_slurred: bool,
    reliable: bool = True,
) -> str:
    """Name the ST segment as one of the four clinical morphology classes."""
    if not reliable or trend == "unknown":
        return ST_PATTERN_UNKNOWN
    # Checked first and unconditionally: a descending ST segment is never
    # benign early repolarisation, however notched the J point looks.
    if trend == "downsloping":
        return ST_PATTERN_DOWNSLOPING
    if (
        j_mv is not None
        and j_mv >= ST_J_POINT_ELEVATION_MV
        and (notched_or_slurred or shape == "concave-up")
    ):
        return ST_PATTERN_J_POINT
    if trend == "upsloping":
        return ST_PATTERN_UPSLOPING
    return ST_PATTERN_HORIZONTAL


def _unavailable(
    *,
    consensus_index: int | None,
    consensus_support: int,
    reason: str,
) -> STLocalization:
    return STLocalization(
        j_index=None,
        j_mv=None,
        st20_mv=None,
        st40_mv=None,
        st60_mv=None,
        st80_mv=None,
        adaptive_index=None,
        adaptive_mv=None,
        mean_mv=None,
        area_mv_ms=None,
        slope_mv_per_ms=None,
        curvature_mv_per_ms2=None,
        trend="unknown",
        shape="unknown",
        baseline_mv=None,
        baseline_source="unavailable",
        baseline_confidence=0.0,
        j_method="unavailable",
        j_confidence=0.0,
        consensus_index=consensus_index,
        consensus_support=int(consensus_support),
        noise_mv=None,
        reliable=False,
        unreliable_reason=reason,
    )


def localize_st_robust(
    signal: np.ndarray,
    *,
    fs: int,
    r_index: int,
    qrs_onset: int | None,
    qrs_offset: int | None,
    p_offset: int | None,
    t_onset: int | None,
    next_r_index: int | None,
    consensus_j_index: int | None,
    consensus_support: int,
    flags: Sequence[str] = (),
    smoothed_signal: np.ndarray | None = None,
    signal_derivative: np.ndarray | None = None,
) -> STLocalization:
    """Measure ST using a robust baseline, local J settling, and window medians."""

    values = np.asarray(signal, dtype=float)
    n_samples = len(values)
    if fs <= 0 or n_samples < 4:
        return _unavailable(
            consensus_index=consensus_j_index,
            consensus_support=consensus_support,
            reason="invalid_signal",
        )
    qrs_on = _finite_index(qrs_onset, n_samples)
    native_j = _finite_index(qrs_offset, n_samples)
    if qrs_on is None or native_j is None or native_j <= qrs_on:
        return _unavailable(
            consensus_index=consensus_j_index,
            consensus_support=consensus_support,
            reason="missing_qrs_bounds",
        )
    p_off = _finite_index(p_offset, n_samples)
    t_on = _finite_index(t_onset, n_samples)
    consensus_j = _finite_index(consensus_j_index, n_samples)

    baseline, baseline_source, baseline_confidence, baseline_noise = _st_baseline(
        values,
        fs=fs,
        r_index=int(r_index),
        qrs_onset=qrs_on,
        p_offset=p_off,
    )
    j_index, j_confidence, j_method = _localize_j(
        values,
        fs=fs,
        r_index=int(r_index),
        qrs_onset=qrs_on,
        native_j=native_j,
        consensus_j=consensus_j,
        t_onset=t_on,
        smoothed_signal=smoothed_signal,
        signal_derivative=signal_derivative,
    )
    if j_index is None:
        return _unavailable(
            consensus_index=consensus_j,
            consensus_support=consensus_support,
            reason="j_localization_failed",
        )

    radius = max(1, int(round(0.006 * fs)))
    st20_index = int(j_index + round(0.020 * fs))
    st40_index = int(j_index + round(0.040 * fs))
    st60_index = int(j_index + round(0.060 * fs))
    st80_index = int(j_index + round(0.080 * fs))
    next_guard = (
        int(next_r_index) - int(round(0.060 * fs))
        if next_r_index is not None
        else n_samples
    )
    j_raw = _window_median(values, j_index + radius, radius=radius)
    st20_raw = (
        _window_median(values, st20_index, radius=radius)
        if st20_index + radius < min(n_samples, next_guard)
        else None
    )
    st40_raw = (
        _window_median(values, st40_index, radius=radius)
        if st40_index + radius < min(n_samples, next_guard)
        else None
    )
    st60_raw = (
        _window_median(values, st60_index, radius=radius)
        if st60_index + radius < min(n_samples, next_guard)
        else None
    )
    st80_raw = (
        _window_median(values, st80_index, radius=radius)
        if st80_index + radius < min(n_samples, next_guard)
        else None
    )
    rr_samples = (
        max(1, int(next_r_index) - int(r_index))
        if next_r_index is not None
        else int(round(0.800 * fs))
    )
    adaptive_index = int(
        j_index + min(int(round(0.080 * fs)), int(round(0.10 * rr_samples)))
    )
    if t_on is not None:
        adaptive_index = min(adaptive_index, int(t_on - round(0.010 * fs)))
    adaptive_raw = (
        _window_median(values, adaptive_index, radius=radius)
        if adaptive_index > j_index + radius
        and adaptive_index + radius < min(n_samples, next_guard)
        else None
    )
    j_mv = None if j_raw is None else float(j_raw - baseline)
    st20_mv = None if st20_raw is None else float(st20_raw - baseline)
    st40_mv = None if st40_raw is None else float(st40_raw - baseline)
    st60_mv = None if st60_raw is None else float(st60_raw - baseline)
    st80_mv = None if st80_raw is None else float(st80_raw - baseline)
    adaptive_mv = None if adaptive_raw is None else float(adaptive_raw - baseline)

    fit_lo = int(j_index + round(0.012 * fs))
    fit_hi = int(j_index + round(0.086 * fs))
    early_t_overlap = False
    if t_on is not None:
        t_guard = int(t_on - round(0.010 * fs))
        if t_guard < fit_hi:
            fit_hi = t_guard
        early_t_overlap = t_on <= st80_index + radius
    fit_hi = min(fit_hi, next_guard, n_samples)
    slope = _robust_linear_slope(
        values,
        lo=fit_lo,
        hi=fit_hi,
        baseline=baseline,
        fs=fs,
    )
    curvature = _robust_quadratic_curvature(
        values,
        lo=fit_lo,
        hi=fit_hi,
        baseline=baseline,
        fs=fs,
    )

    mean_lo = int(j_index + round(0.020 * fs))
    mean_hi = min(int(j_index + round(0.086 * fs)), fit_hi)
    mean_mv = None
    area_mv_ms = None
    if mean_hi - mean_lo >= max(3, int(round(0.020 * fs))):
        segment = np.asarray(values[mean_lo:mean_hi], dtype=float) - baseline
        finite_segment = segment[np.isfinite(segment)]
        if finite_segment.size:
            mean_mv = _median_1d(finite_segment)
            area_mv_ms = float(trapezoid(segment, dx=1000.0 / float(fs)))

    st_noise_segment = values[
        max(0, int(j_index + round(0.012 * fs))) :
        min(n_samples, int(j_index + round(0.086 * fs)))
    ]
    st_noise = _mad_sigma(np.diff(st_noise_segment)) / np.sqrt(2.0) if st_noise_segment.size >= 3 else 0.0
    noise_mv = max(float(baseline_noise), float(st_noise))
    support_score = float(np.clip(int(consensus_support) / 6.0, 0.0, 1.0))
    stability_score = float(np.clip(1.0 - noise_mv / 0.060, 0.0, 1.0))
    overall_confidence = float(
        np.clip(
            0.35 * j_confidence
            + 0.25 * baseline_confidence
            + 0.20 * support_score
            + 0.20 * stability_score,
            0.0,
            1.0,
        )
    )

    flag_set = set(flags or ())
    reason = None
    if "paced_beat" in flag_set:
        reason = "paced_beat"
    elif "qrs_unreliable" in flag_set:
        reason = "qrs_unreliable"
    elif "st_hybrid_low_qrs_quality" in flag_set:
        reason = "lead_qrs_quality"
    elif consensus_support < 3:
        reason = "insufficient_consensus_support"
    elif j_method == "native_consensus_disagreement":
        reason = "j_consensus_disagreement"
    elif early_t_overlap:
        reason = "early_t_overlap"
    elif j_mv is None or st40_mv is None or st80_mv is None or slope is None:
        reason = "incomplete_st_window"
    elif baseline_confidence < 0.50:
        reason = "low_baseline_confidence"
    elif overall_confidence < 0.50:
        reason = "low_confidence"

    return STLocalization(
        j_index=int(j_index),
        j_mv=j_mv,
        st20_mv=st20_mv,
        st40_mv=st40_mv,
        st60_mv=st60_mv,
        st80_mv=st80_mv,
        adaptive_index=int(adaptive_index) if adaptive_mv is not None else None,
        adaptive_mv=adaptive_mv,
        mean_mv=mean_mv,
        area_mv_ms=area_mv_ms,
        slope_mv_per_ms=slope,
        curvature_mv_per_ms2=curvature,
        trend=_st_trend(slope),
        shape=_st_shape(j_mv, st40_mv, st80_mv, noise_mv=noise_mv),
        baseline_mv=float(baseline),
        baseline_source=baseline_source,
        baseline_confidence=float(baseline_confidence),
        j_method=j_method,
        j_confidence=overall_confidence,
        consensus_index=consensus_j,
        consensus_support=int(consensus_support),
        noise_mv=float(noise_mv),
        reliable=reason is None,
        unreliable_reason=reason,
    )


def _quality_allows_qrs(quality: Mapping[str, object] | None, lead: str) -> bool:
    if quality is None:
        return True
    lead_quality = quality.get(lead)
    return bool(getattr(lead_quality, "reliable_for_qrs", False))


def apply_hybrid_st_measurement(
    beat_features: Sequence[LeadBeatFeatures],
    *,
    measurement_ecg: np.ndarray,
    r_locs: np.ndarray,
    fs: int,
    quality: Mapping[str, object] | None = None,
) -> None:
    """Attach robust ST side fields without replacing native ST/interpretation."""

    ecg = np.asarray(measurement_ecg, dtype=float)
    r_values = np.asarray(r_locs, dtype=int)
    if ecg.ndim != 2 or fs <= 0 or r_values.size == 0:
        return
    lead_indices = {
        lead: index
        for index, lead in enumerate(STANDARD_12_LEADS)
        if index < ecg.shape[0]
    }
    # J localization previously smoothed the same complete lead once per beat.
    # A 10-second record can contain hundreds of lead-beat rows, while the
    # filtered lead and its derivative are invariant across all of them.
    smoothed_by_index = {
        index: _smooth(ecg[index], fs)
        for index in lead_indices.values()
    }
    derivative_by_index = {
        index: np.diff(smoothed)
        for index, smoothed in smoothed_by_index.items()
    }
    by_beat: dict[int, list[LeadBeatFeatures]] = {}
    for feature in beat_features:
        by_beat.setdefault(int(feature.beat_id), []).append(feature)

    for beat_id, items in by_beat.items():
        if beat_id < 0 or beat_id >= len(r_values):
            continue
        anchors = [
            int(anchor)
            for feature in items
            if _quality_allows_qrs(quality, feature.lead)
            and "qrs_unreliable" not in (getattr(feature, "flags", []) or [])
            and (anchor := _effective_native_j(feature)) is not None
        ]
        # Global J represents the latest reliable ventricular depolarisation;
        # P85 is robust to one noisy-late lead while preserving that definition.
        consensus_j = (
            int(round(float(np.percentile(anchors, 85)))) if anchors else None
        )
        support = len(anchors)
        current_r = int(r_values[beat_id])
        next_r = int(r_values[beat_id + 1]) if beat_id + 1 < len(r_values) else None

        for feature in items:
            lead_index = lead_indices.get(feature.lead)
            native_j = _effective_native_j(feature)
            if lead_index is None or native_j is None:
                continue
            localization_flags = list(getattr(feature, "flags", []) or [])
            if quality is not None and not _quality_allows_qrs(quality, feature.lead):
                localization_flags.append("st_hybrid_low_qrs_quality")
            result = localize_st_robust(
                ecg[lead_index],
                fs=fs,
                r_index=current_r,
                qrs_onset=getattr(getattr(feature, "qrs", None), "onset", None),
                qrs_offset=native_j,
                p_offset=getattr(getattr(feature, "p", None), "offset", None),
                t_onset=getattr(getattr(feature, "t", None), "onset", None),
                next_r_index=next_r,
                consensus_j_index=consensus_j,
                consensus_support=support,
                flags=localization_flags,
                smoothed_signal=smoothed_by_index[lead_index],
                signal_derivative=derivative_by_index[lead_index],
            )
            feature.st_hybrid_j_index = result.j_index
            feature.st_hybrid_j_mv = result.j_mv
            feature.st_hybrid_20ms_mv = result.st20_mv
            feature.st_hybrid_40ms_mv = result.st40_mv
            feature.st_hybrid_60ms_mv = result.st60_mv
            feature.st_hybrid_80ms_mv = result.st80_mv
            feature.st_hybrid_adaptive_index = result.adaptive_index
            feature.st_hybrid_adaptive_mv = result.adaptive_mv
            feature.st_hybrid_mean_mv = result.mean_mv
            feature.st_hybrid_area_mv_ms = result.area_mv_ms
            feature.st_hybrid_slope_mv_per_ms = result.slope_mv_per_ms
            feature.st_hybrid_curvature_mv_per_ms2 = (
                result.curvature_mv_per_ms2
            )
            feature.st_hybrid_trend = result.trend
            feature.st_hybrid_shape = result.shape
            feature.st_hybrid_baseline_mv = result.baseline_mv
            feature.st_hybrid_baseline_source = result.baseline_source
            feature.st_hybrid_baseline_confidence = result.baseline_confidence
            feature.st_hybrid_j_method = result.j_method
            feature.st_hybrid_j_confidence = result.j_confidence
            feature.st_hybrid_consensus_index = result.consensus_index
            feature.st_hybrid_consensus_support = result.consensus_support
            feature.st_hybrid_noise_mv = result.noise_mv
            feature.st_hybrid_reliable = result.reliable
            feature.st_hybrid_unreliable_reason = result.unreliable_reason
            feature.st_pattern_class = classify_st_pattern(
                j_mv=result.j_mv,
                trend=result.trend,
                shape=result.shape,
                notched_or_slurred=bool(
                    getattr(feature, "qrs_slur_flag", False)
                    or int(getattr(feature, "qrs_notch_count", 0) or 0) >= 1
                ),
                reliable=result.reliable,
            )
