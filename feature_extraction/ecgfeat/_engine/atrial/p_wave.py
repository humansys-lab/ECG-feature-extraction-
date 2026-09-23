from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..foundation.models import (
    PWaveBeatAssessment,
    PWaveLeadBoundary,
    STANDARD_12_LEADS,
    WaveBounds,
)
from ..preprocess import lowpass_filter


P_PRESENT = "P_PRESENT"
AF_LIKE = "AF_LIKE"
ORGANIZED_ATRIAL_ACTIVITY = "ORGANIZED_ATRIAL_ACTIVITY"
OVERLAP_UNCERTAIN = "OVERLAP_UNCERTAIN"

BASELINE_UNOBSERVABLE = "BASELINE_UNOBSERVABLE"
P_ON_T_UNRESOLVED = "P_ON_T_UNRESOLVED"
P_OFFSET_TA_AMBIGUOUS = "P_OFFSET_TA_AMBIGUOUS"
INSUFFICIENT_INFORMATIVE_LEADS = "INSUFFICIENT_INFORMATIVE_LEADS"
CROSS_LEAD_DISAGREEMENT = "CROSS_LEAD_DISAGREEMENT"
CHANNEL_DESYNCHRONIZED = "CHANNEL_DESYNCHRONIZED"
LEAD_CONFIGURATION_INVALID = "LEAD_CONFIGURATION_INVALID"
MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"

_LEAD_GROUPS = {
    "limb": {"I", "II", "III", "aVR", "aVL", "aVF"},
    "right_precordial": {"V1", "V2", "V3"},
    "left_precordial": {"V4", "V5", "V6"},
}


@dataclass(frozen=True)
class PWaveConfig:
    mode: str = "offline"
    detection_lowpass_hz: float = 35.0
    minimum_informative_snr: float = 2.0
    minimum_informative_leads: int = 3
    minimum_independent_groups: int = 2
    cluster_radius_ms: float = 25.0
    maximum_cluster_spread_ms: float = 45.0
    strict_robust_disagreement_ms: float = 35.0
    branch_disagreement_ms: float = 30.0
    baseline_unobservable_ms: float = 1000.0
    minimum_ci_half_width_ms: float = 4.0
    ci_scale: float = 1.0
    update_legacy_consensus: bool = True
    model_arbitration: bool = False
    multiple_candidates: bool = False


def _lead_group(lead: str) -> str:
    for group, leads in _LEAD_GROUPS.items():
        if lead in leads:
            return group
    return "unknown"


def _median_1d(values: np.ndarray) -> float:
    """Fast exact median for a finite, non-empty one-dimensional array."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    midpoint = arr.size // 2
    if arr.size % 2:
        return float(np.partition(arr, midpoint)[midpoint])
    partitioned = np.partition(arr, (midpoint - 1, midpoint))
    return float(0.5 * (partitioned[midpoint - 1] + partitioned[midpoint]))


def _robust_sigma(values: Sequence[float], floor: float = 0.0) -> float:
    # Most hot-path callers already have a NumPy array.  Converting it to a
    # Python list and immediately back to an array accounted for thousands of
    # allocations per record in the P-wave boundary engine.  Preserve support
    # for arbitrary sequences/generators without copying array inputs.
    arr = (
        np.asarray(values, dtype=float)
        if isinstance(values, np.ndarray)
        else np.asarray(list(values), dtype=float)
    )
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(floor)
    center = _median_1d(arr)
    sigma = 1.4826 * _median_1d(np.abs(arr - center))
    if sigma <= 1e-12 and arr.size >= 2:
        sigma = float(np.std(arr, ddof=1))
    return float(max(floor, sigma))


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> Optional[float]:
    if not values or len(values) != len(weights):
        return None
    ordered = sorted(
        (float(value), max(0.0, float(weight)))
        for value, weight in zip(values, weights)
        if np.isfinite(value) and np.isfinite(weight)
    )
    if not ordered:
        return None
    total = float(sum(weight for _, weight in ordered))
    if total <= 1e-12:
        return float(np.median([value for value, _ in ordered]))
    target = float(np.clip(quantile, 0.0, 1.0)) * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _features_by_beat(beat_features: Sequence[object]) -> Dict[int, List[object]]:
    out: Dict[int, List[object]] = {}
    for feature in beat_features:
        out.setdefault(int(getattr(feature, "beat_id")), []).append(feature)
    return out


def _features_by_beat_lead(
    beat_features: Sequence[object],
) -> Dict[Tuple[int, str], object]:
    return {
        (int(getattr(feature, "beat_id")), str(getattr(feature, "lead"))): feature
        for feature in beat_features
    }


def _median_index(values: Iterable[Optional[int]]) -> Optional[int]:
    finite = [int(value) for value in values if value is not None]
    return int(round(float(np.median(finite)))) if finite else None


def _wave_exclusion_mask(
    n_samples: int,
    fs: int,
    r_locs: np.ndarray,
    beat_features: Sequence[object],
) -> np.ndarray:
    quiet = np.ones(n_samples, dtype=bool)
    guard = max(1, int(round(0.008 * fs)))
    by_beat = _features_by_beat(beat_features)
    for beat_id, r_sample in enumerate(np.asarray(r_locs, dtype=int)):
        items = by_beat.get(beat_id, [])
        intervals: List[Tuple[int, int]] = []
        for wave_name in ("p", "qrs", "t"):
            onsets = [
                getattr(getattr(item, wave_name, None), "onset", None)
                for item in items
            ]
            offsets = [
                getattr(getattr(item, wave_name, None), "offset", None)
                for item in items
            ]
            onset = _median_index(onsets)
            offset = _median_index(offsets)
            if onset is not None and offset is not None and offset > onset:
                intervals.append((onset - guard, offset + guard))
        if not any(
            start <= int(r_sample) <= stop for start, stop in intervals
        ):
            intervals.append(
                (
                    int(r_sample) - int(round(0.100 * fs)),
                    int(r_sample) + int(round(0.120 * fs)),
                )
            )
        for start, stop in intervals:
            quiet[max(0, start) : min(n_samples, stop + 1)] = False
    return quiet


def _quiet_observation_mask(
    ecg: np.ndarray,
    fs: int,
    r_locs: np.ndarray,
    beat_features: Sequence[object],
) -> np.ndarray:
    values = np.asarray(ecg, dtype=float)
    structural = _wave_exclusion_mask(
        values.shape[1],
        fs,
        r_locs,
        beat_features,
    )
    derivative = np.gradient(values, axis=1)
    spatial_slope = np.median(np.abs(derivative), axis=0)
    fine_energy = gaussian_filter1d(
        np.median(derivative * derivative, axis=0),
        sigma=max(0.5, 0.006 * fs),
        mode="nearest",
    )
    candidate = structural & np.isfinite(spatial_slope) & np.isfinite(fine_energy)
    if int(np.sum(candidate)) < max(16, int(round(0.10 * fs))):
        return structural
    slope_limit = float(np.quantile(spatial_slope[candidate], 0.60))
    energy_limit = float(np.quantile(fine_energy[candidate], 0.60))
    return candidate & (spatial_slope <= slope_limit) & (fine_energy <= energy_limit)


def _kalman_baseline_pass(
    observations: np.ndarray,
    observed: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(observations, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    n = y.size
    if n == 0:
        return y.copy(), np.zeros_like(y)
    quiet_values = y[mask & np.isfinite(y)]
    initial = float(np.median(quiet_values)) if quiet_values.size else float(np.median(y))
    noise = _robust_sigma(
        np.diff(quiet_values) if quiet_values.size > 2 else quiet_values,
        floor=1e-4,
    )
    measurement_variance = max(noise * noise, 1e-8)
    from ..foundation.kalman import run_kalman
    return run_kalman(y, mask, fs, initial, measurement_variance)


def _state_space_baseline(
    ecg: np.ndarray,
    quiet_mask: np.ndarray,
    fs: int,
    *,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(ecg, dtype=float)
    lead_offsets = np.asarray(
        [
            float(np.median(lead[quiet_mask]))
            if int(np.sum(quiet_mask)) >= 4
            else float(np.median(lead))
            for lead in values
        ],
        dtype=float,
    )
    centered = values - lead_offsets[:, None]
    common_observation = np.median(centered, axis=0)
    common, common_sigma = _kalman_baseline_pass(
        common_observation,
        quiet_mask,
        fs,
    )
    if mode == "offline":
        backward, backward_sigma = _kalman_baseline_pass(
            common_observation[::-1],
            quiet_mask[::-1],
            fs,
        )
        common = 0.5 * (common + backward[::-1])
        common_sigma = np.sqrt(
            0.5 * (common_sigma * common_sigma + backward_sigma[::-1] ** 2)
        )

    baselines = np.zeros_like(values)
    uncertainties = np.zeros_like(values)
    for lead_index, signal in enumerate(centered):
        residual_observation = signal - common
        residual, residual_sigma = _kalman_baseline_pass(
            residual_observation,
            quiet_mask,
            fs,
        )
        if mode == "offline":
            backward, backward_sigma = _kalman_baseline_pass(
                residual_observation[::-1],
                quiet_mask[::-1],
                fs,
            )
            residual = 0.5 * (residual + backward[::-1])
            residual_sigma = np.sqrt(
                0.5 * (residual_sigma * residual_sigma + backward_sigma[::-1] ** 2)
            )
        baselines[lead_index] = lead_offsets[lead_index] + common + residual
        uncertainties[lead_index] = np.sqrt(
            common_sigma * common_sigma + residual_sigma * residual_sigma
        )
    return baselines, uncertainties


def _distance_to_observation(mask: np.ndarray) -> np.ndarray:
    observed = np.flatnonzero(mask)
    if observed.size == 0:
        return np.full(mask.size, np.inf, dtype=float)
    indices = np.arange(mask.size)
    positions = np.searchsorted(observed, indices)
    before_pos = np.clip(positions - 1, 0, observed.size - 1)
    after_pos = np.clip(positions, 0, observed.size - 1)
    before = np.abs(indices - observed[before_pos])
    after = np.abs(indices - observed[after_pos])
    return np.minimum(before, after).astype(float)


def _whitened_spatial_activity(
    detection_signal: np.ndarray,
    quiet_mask: np.ndarray,
    quality: Dict[str, object],
    excluded_leads: set[str],
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    values = np.asarray(detection_signal, dtype=float)
    usable_indices = [
        index
        for index, lead in enumerate(STANDARD_12_LEADS)
        if index < values.shape[0]
        and lead not in excluded_leads
        and bool(getattr(quality.get(lead), "reliable_for_p", True))
    ]
    if len(usable_indices) < 2:
        usable_indices = [
            index
            for index, lead in enumerate(STANDARD_12_LEADS)
            if index < values.shape[0] and lead not in excluded_leads
        ]
    selected = values[usable_indices]
    noise_samples = selected[:, quiet_mask]
    if noise_samples.shape[1] < max(8, len(usable_indices) + 1):
        noise_samples = np.diff(selected, axis=1) / np.sqrt(2.0)
    covariance = np.cov(noise_samples)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.75 * covariance + 0.25 * diagonal
    scale = float(np.median(np.diag(covariance))) if covariance.size else 1.0
    covariance += np.eye(covariance.shape[0]) * max(scale * 1e-4, 1e-9)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, max(scale * 1e-6, 1e-9)))
    ) @ eigenvectors.T
    derivative = np.gradient(selected, axis=1)
    whitened = inverse_root @ derivative
    activity = np.sqrt(np.mean(whitened * whitened, axis=0))
    activity = gaussian_filter1d(
        activity,
        sigma=max(0.5, 0.006 * fs),
        mode="nearest",
    )
    return activity, inverse_root, [STANDARD_12_LEADS[index] for index in usable_indices]


def _connected_regions(mask: np.ndarray, lo: int) -> List[Tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    regions: List[Tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for index in indices[1:]:
        value = int(index)
        if value > previous + 1:
            regions.append((lo + start, lo + previous))
            start = value
        previous = value
    regions.append((lo + start, lo + previous))
    return regions


def _coarse_p_window(
    beat_id: int,
    r_locs: np.ndarray,
    items: Sequence[object],
    activity: np.ndarray,
    quiet_mask: np.ndarray,
    fs: int,
    activity_thresholds: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int, int]:
    qrs_on = _median_index(
        getattr(getattr(item, "qrs", None), "onset", None) for item in items
    )
    r_sample = int(r_locs[beat_id])
    qrs_reference = qrs_on if qrs_on is not None else r_sample - int(round(0.020 * fs))
    if beat_id > 0:
        rr = r_sample - int(r_locs[beat_id - 1])
        lookback = min(int(round(0.450 * fs)), int(round(0.65 * rr)))
    else:
        lookback = int(round(0.400 * fs))
    search_lo = max(0, qrs_reference - lookback)
    search_hi = max(search_lo + 2, qrs_reference - int(round(0.008 * fs)))
    search_hi = min(activity.size - 1, search_hi)
    segment = activity[search_lo : search_hi + 1]
    low_threshold, high_threshold = (
        _activity_thresholds(activity, quiet_mask)
        if activity_thresholds is None else activity_thresholds
    )
    low_threshold = max(low_threshold, float(np.median(segment)))
    regions = _connected_regions(segment >= low_threshold, search_lo)
    seeds = [
        int(getattr(getattr(item, "p", None), "peak"))
        for item in items
        if getattr(getattr(item, "p", None), "peak", None) is not None
        and float(getattr(item, "p_confidence", 0.0) or 0.0) >= 0.25
    ]
    seed = int(round(float(np.median(seeds)))) if seeds else None
    eligible: List[Tuple[float, int, int]] = []
    for start, stop in regions:
        local = activity[start : stop + 1]
        if local.size == 0 or float(np.max(local)) < high_threshold:
            continue
        energy = float(np.sum(local))
        if seed is not None:
            distance = 0 if start <= seed <= stop else min(abs(seed - start), abs(seed - stop))
            energy /= 1.0 + distance
        eligible.append((energy, start, stop))
    if eligible:
        _, start, stop = max(eligible, key=lambda item: item[0])
    elif seed is not None:
        start = seed - int(round(0.100 * fs))
        stop = seed + int(round(0.100 * fs))
    else:
        peak = search_lo + int(np.argmax(segment))
        start = peak - int(round(0.100 * fs))
        stop = peak + int(round(0.100 * fs))
    margin = int(round(0.020 * fs))
    start = max(search_lo, start - margin)
    stop = min(search_hi, stop + margin)
    minimum_width = int(round(0.080 * fs))
    if stop - start < minimum_width:
        center = seed if seed is not None else (start + stop) // 2
        start = max(search_lo, center - minimum_width // 2)
        stop = min(search_hi, start + minimum_width)

    usable = np.asarray(
        [
            index
            for index, item in enumerate(items)
            if getattr(getattr(item, "p", None), "peak", None) is not None
        ],
        dtype=int,
    )
    svd_dimension = 1
    if usable.size >= 2:
        # This is a provenance-only fallback. The actual SVD is recomputed from
        # the baseline-corrected multi-lead window in the caller.
        svd_dimension = 2 if usable.size >= 6 and len(set(seeds)) >= 2 else 1
    return int(start), int(stop), int(svd_dimension)


def _activity_thresholds(activity: np.ndarray, quiet_mask: np.ndarray) -> Tuple[float, float]:
    quiet = activity[quiet_mask]
    if quiet.size >= 16:
        return float(np.quantile(quiet, 0.95)), float(np.quantile(quiet, 0.995))
    return float(np.quantile(activity, 0.60)), float(np.quantile(activity, 0.90))


def _svd_dimension(
    corrected_detection: np.ndarray,
    start: int,
    stop: int,
    usable_leads: Sequence[str],
) -> int:
    indices = [
        STANDARD_12_LEADS.index(lead)
        for lead in usable_leads
        if lead in STANDARD_12_LEADS
        and STANDARD_12_LEADS.index(lead) < corrected_detection.shape[0]
    ]
    if len(indices) < 2 or stop - start < 4:
        return 1
    window = corrected_detection[indices, start : stop + 1]
    window = window - np.mean(window, axis=1, keepdims=True)
    singular = np.linalg.svd(window, full_matrices=False, compute_uv=False)
    if singular.size < 2 or singular[0] <= 1e-9:
        return 1
    return 2 if float(singular[1] / singular[0]) >= 0.55 else 1


def _interpolate_masked_region(
    segment: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(segment, dtype=float).copy()
    masked = np.asarray(mask, dtype=bool)
    if not np.any(masked):
        return values
    known = np.flatnonzero(~masked)
    missing = np.flatnonzero(masked)
    if known.size < 2:
        return values
    values[:, missing] = np.vstack(
        [
            np.interp(missing, known, lead[known])
            for lead in values
        ]
    )
    return values


def _robust_affine_fit(
    model: np.ndarray,
    observation: np.ndarray,
    fit_mask: np.ndarray,
) -> Tuple[float, float, float]:
    x = np.asarray(model, dtype=float)[fit_mask]
    y = np.asarray(observation, dtype=float)[fit_mask]
    if x.size < 8:
        return 1.0, 0.0, np.inf
    weights = np.ones_like(x)
    coefficients = np.asarray([1.0, 0.0], dtype=float)
    for _ in range(5):
        # Closed-form weighted simple linear regression is equivalent to the
        # former generic least-squares solve for the fixed [x, 1] design, but
        # avoids five small SVDs for every lead/template candidate.
        weight_sum = float(np.sum(weights))
        weighted_x = float(np.dot(weights, x))
        weighted_y = float(np.dot(weights, y))
        weighted_xx = float(np.dot(weights, x * x))
        weighted_xy = float(np.dot(weights, x * y))
        denominator = weighted_xx * weight_sum - weighted_x * weighted_x
        scale = max(abs(weighted_xx * weight_sum), abs(weighted_x * weighted_x), 1.0)
        if abs(denominator) <= np.finfo(float).eps * scale:
            break
        alpha = (
            weighted_xy * weight_sum - weighted_x * weighted_y
        ) / denominator
        beta = (
            weighted_xx * weighted_y - weighted_x * weighted_xy
        ) / denominator
        coefficients[0] = alpha
        coefficients[1] = beta
        residual = y - (alpha * x + beta)
        center = _median_1d(residual)
        sigma = _robust_sigma(residual, floor=1e-6)
        limit = 1.5 * sigma
        absolute = np.abs(residual - center)
        weights = np.ones_like(absolute)
        outliers = absolute > limit
        weights[outliers] = limit / np.maximum(absolute[outliers], 1e-12)
    alpha = float(np.clip(coefficients[0], 0.50, 1.50))
    beta_limit = max(0.02, 0.25 * float(np.ptp(y)))
    beta = float(np.clip(coefficients[1], -beta_limit, beta_limit))
    error = float(np.sqrt(np.mean((y - (alpha * x + beta)) ** 2)))
    return alpha, beta, error


def _robust_affine_errors_multilead(
    model: np.ndarray,
    observation: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of ``_robust_affine_fit`` for candidate scoring."""
    x = np.asarray(model, dtype=float)[:, fit_mask]
    y = np.asarray(observation, dtype=float)[:, fit_mask]
    if x.shape != y.shape or x.shape[1] < 8:
        return np.full(x.shape[0], np.inf, dtype=float)
    weights = np.ones_like(x)
    alpha = np.ones(x.shape[0], dtype=float)
    beta = np.zeros(x.shape[0], dtype=float)
    active = np.ones(x.shape[0], dtype=bool)
    x_squared = x * x
    for _ in range(5):
        weight_sum = np.sum(weights, axis=1)
        weighted_x = np.sum(weights * x, axis=1)
        weighted_y = np.sum(weights * y, axis=1)
        weighted_xx = np.sum(weights * x_squared, axis=1)
        weighted_xy = np.sum(weights * x * y, axis=1)
        denominator = weighted_xx * weight_sum - weighted_x * weighted_x
        scale = np.maximum.reduce(
            (
                np.abs(weighted_xx * weight_sum),
                np.abs(weighted_x * weighted_x),
                np.ones_like(weight_sum),
            )
        )
        valid = active & (np.abs(denominator) > np.finfo(float).eps * scale)
        if not np.any(valid):
            break
        alpha[valid] = (
            weighted_xy[valid] * weight_sum[valid]
            - weighted_x[valid] * weighted_y[valid]
        ) / denominator[valid]
        beta[valid] = (
            weighted_xx[valid] * weighted_y[valid]
            - weighted_x[valid] * weighted_xy[valid]
        ) / denominator[valid]
        residual = y - (alpha[:, None] * x + beta[:, None])
        center = np.median(residual, axis=1)
        absolute_deviation = np.abs(residual - center[:, None])
        sigma = 1.4826 * np.median(absolute_deviation, axis=1)
        degenerate = sigma <= 1e-12
        if np.any(degenerate):
            sigma[degenerate] = np.std(
                residual[degenerate], axis=1, ddof=1
            )
        sigma = np.maximum(sigma, 1e-6)
        limit = 1.5 * sigma
        weights = np.ones_like(absolute_deviation)
        outliers = absolute_deviation > limit[:, None]
        weights[outliers] = np.broadcast_to(
            limit[:, None], absolute_deviation.shape
        )[outliers] / np.maximum(absolute_deviation[outliers], 1e-12)
        active &= valid
    alpha = np.clip(alpha, 0.50, 1.50)
    beta_limit = np.maximum(0.02, 0.25 * np.ptp(y, axis=1))
    beta = np.clip(beta, -beta_limit, beta_limit)
    return np.sqrt(np.mean((y - (alpha[:, None] * x + beta[:, None])) ** 2, axis=1))


def _resample_template_variant(
    template: np.ndarray,
    shift_samples: int,
    stretch: float,
) -> np.ndarray:
    values = np.asarray(template, dtype=float)
    n = values.shape[1]
    positions = np.arange(n, dtype=float)
    center = 0.5 * float(n - 1)
    source = (positions - center - float(shift_samples)) / float(stretch) + center
    return np.vstack(
        [
            np.interp(source, positions, lead, left=lead[0], right=lead[-1])
            for lead in values
        ]
    )


def _constrained_t_reconstruction(
    corrected_precision: np.ndarray,
    *,
    beat_id: int,
    r_locs: np.ndarray,
    qrs_on: int,
    coarse_start: int,
    coarse_stop: int,
    feature_map: Dict[Tuple[int, str], object],
    fs: int,
) -> Tuple[Optional[np.ndarray], Dict[str, bool], Dict[str, object]]:
    """Reconstruct the previous T tail while excluding every source/target P mask."""
    if beat_id <= 0 or beat_id >= len(r_locs):
        return None, {}, {"available": False, "reason": "previous_beat_unavailable"}
    previous_r = int(r_locs[beat_id - 1])
    rr_target = int(r_locs[beat_id]) - previous_r
    support_start = max(
        0,
        previous_r + int(round(0.060 * fs)),
        coarse_start - int(round(0.160 * fs)),
    )
    support_stop = min(
        corrected_precision.shape[1] - 1,
        int(qrs_on) - int(round(0.008 * fs)),
        coarse_stop + int(round(0.080 * fs)),
    )
    if support_stop - support_start < int(round(0.100 * fs)):
        return None, {}, {"available": False, "reason": "insufficient_t_support"}
    target_indices = np.arange(support_start, support_stop + 1, dtype=int)
    relative = target_indices - previous_r
    target_p_mask = (
        (target_indices >= int(coarse_start))
        & (target_indices <= int(coarse_stop))
    )
    fit_mask = ~target_p_mask
    if int(np.sum(fit_mask)) < max(12, int(round(0.040 * fs))):
        return None, {}, {"available": False, "reason": "insufficient_unmasked_fit_samples"}

    source_segments: List[np.ndarray] = []
    source_ids: List[int] = []
    for source_previous in range(max(0, len(r_locs) - 1)):
        source_beat = source_previous + 1
        if source_previous == beat_id - 1 or source_beat >= len(r_locs):
            continue
        rr_source = int(r_locs[source_beat]) - int(r_locs[source_previous])
        if abs(rr_source - rr_target) / max(float(rr_target), 1.0) > 0.20:
            continue
        source_indices = int(r_locs[source_previous]) + relative
        if source_indices[0] < 0 or source_indices[-1] >= corrected_precision.shape[1]:
            continue
        segment = corrected_precision[:, source_indices].copy()
        source_p_mask = np.zeros(source_indices.size, dtype=bool)
        for lead in STANDARD_12_LEADS:
            feature = feature_map.get((source_beat, lead))
            if feature is None:
                continue
            p_wave = getattr(feature, "p", None)
            onset = getattr(p_wave, "onset", None)
            offset = getattr(p_wave, "offset", None)
            if onset is None or offset is None:
                continue
            source_p_mask |= (
                (source_indices >= int(onset) - int(round(0.010 * fs)))
                & (source_indices <= int(offset) + int(round(0.010 * fs)))
            )
        segment = _interpolate_masked_region(segment, source_p_mask)
        source_segments.append(segment)
        source_ids.append(source_previous)
    if len(source_segments) < 2:
        return None, {}, {
            "available": False,
            "reason": "insufficient_similar_t_templates",
            "template_count": len(source_segments),
        }

    template = np.median(np.stack(source_segments, axis=0), axis=0)
    target = corrected_precision[:, target_indices]
    shifts = range(
        -max(1, int(round(0.008 * fs))),
        max(1, int(round(0.008 * fs))) + 1,
        max(1, int(round(0.004 * fs))),
    )
    best_variant = template
    best_error = np.inf
    best_parameters = (0, 1.0)
    for stretch in (0.97, 1.0, 1.03):
        for shift in shifts:
            variant = _resample_template_variant(template, shift, stretch)
            errors = _robust_affine_errors_multilead(
                variant,
                target,
                fit_mask,
            )
            finite_errors = errors[np.isfinite(errors)]
            score = (
                float(np.median(finite_errors))
                if finite_errors.size
                else np.inf
            )
            if score < best_error:
                best_error = score
                best_variant = variant
                best_parameters = (shift, stretch)

    reconstructed = target.copy()
    validated: Dict[str, bool] = {}
    residual_candidate = np.zeros_like(target)
    hf_ratios: List[float] = []
    residual_ratios: List[float] = []
    for lead_index, lead in enumerate(STANDARD_12_LEADS[: target.shape[0]]):
        alpha, beta, _error = _robust_affine_fit(
            best_variant[lead_index],
            target[lead_index],
            fit_mask,
        )
        model = alpha * best_variant[lead_index] + beta
        edge = min(
            max(2, int(round(0.020 * fs))),
            max(2, model.size // 3),
        )
        taper = np.ones(model.size, dtype=float)
        phase = np.linspace(0.0, 1.0, edge)
        ramp = 0.5 - 0.5 * np.cos(np.pi * phase)
        taper[:edge] = ramp
        taper[-edge:] = ramp[::-1]
        residual = target[lead_index] - taper * model
        residual_candidate[lead_index] = residual
        original_energy = float(np.mean(target[lead_index, fit_mask] ** 2))
        residual_energy = float(np.mean(residual[fit_mask] ** 2))
        energy_ratio = residual_energy / max(original_energy, 1e-12)
        original_hf = np.diff(target[lead_index], n=2)
        residual_hf = np.diff(residual, n=2)
        hf_ratio = float(
            np.sqrt(np.mean(residual_hf * residual_hf))
            / (np.sqrt(np.mean(original_hf * original_hf)) + 1e-12)
        )
        is_valid = bool(energy_ratio < 0.85 and hf_ratio <= 1.25)
        validated[lead] = is_valid
        residual_ratios.append(energy_ratio)
        hf_ratios.append(hf_ratio)
        if is_valid:
            reconstructed[lead_index] = residual

    candidate_slice = slice(
        int(coarse_start - support_start),
        int(coarse_stop - support_start + 1),
    )
    return reconstructed[:, candidate_slice], validated, {
        "available": True,
        "method": "p_masked_rr_matched_t_template_affine_shift_stretch",
        "template_count": len(source_segments),
        "template_source_beats": source_ids,
        "shift_samples": int(best_parameters[0]),
        "stretch": float(best_parameters[1]),
        "median_fit_error": float(best_error),
        "median_residual_energy_ratio": float(np.median(residual_ratios)),
        "median_high_frequency_ratio": float(np.median(hf_ratios)),
    }


def _t_spatial_subspace_suppression(
    corrected_detection: np.ndarray,
    *,
    previous_r: Optional[int],
    coarse_start: int,
    coarse_stop: int,
    fs: int,
) -> Optional[np.ndarray]:
    if previous_r is None:
        return None
    learn_start = max(0, int(previous_r) + int(round(0.080 * fs)))
    learn_stop = min(
        corrected_detection.shape[1],
        coarse_start - int(round(0.010 * fs)),
    )
    if learn_stop - learn_start < max(8, int(round(0.040 * fs))):
        return None
    learning = corrected_detection[:, learn_start:learn_stop]
    learning = learning - np.mean(learning, axis=1, keepdims=True)
    try:
        directions, singular, _ = np.linalg.svd(learning, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular.size == 0 or singular[0] <= 1e-9:
        return None
    rank = 2 if singular.size > 1 and singular[1] / singular[0] >= 0.70 else 1
    t_basis = directions[:, :rank]
    candidate = corrected_detection[:, coarse_start : coarse_stop + 1]
    return candidate - t_basis @ (t_basis.T @ candidate)


def _threshold_bounds(
    signal: np.ndarray,
    peak: int,
    fraction: float,
) -> Tuple[Optional[int], Optional[int]]:
    values = np.asarray(signal, dtype=float)
    if not (0 <= peak < values.size):
        return None, None
    amplitude = abs(float(values[peak]))
    if amplitude <= 1e-9:
        return None, None
    threshold = max(0.001, float(fraction) * amplitude)
    onset = int(peak)
    while onset > 0 and abs(float(values[onset])) > threshold:
        onset -= 1
    offset = int(peak)
    while offset < values.size - 1 and abs(float(values[offset])) > threshold:
        offset += 1
    return onset, offset


def _activity_bounds(
    signal: np.ndarray,
    peak: int,
    fs: int,
) -> Tuple[Optional[int], Optional[int]]:
    values = np.asarray(signal, dtype=float)
    if values.size < 8 or not (0 <= peak < values.size):
        return None, None
    envelopes: List[np.ndarray] = []
    for scale_ms in (4.0, 8.0, 16.0):
        sigma = max(0.5, scale_ms * fs / 1000.0)
        smooth = gaussian_filter1d(values, sigma=sigma, mode="nearest")
        derivative = np.abs(np.gradient(smooth))
        dog = np.abs(
            smooth
            - gaussian_filter1d(
                values,
                sigma=max(sigma * 1.8, sigma + 0.5),
                mode="nearest",
            )
        )
        derivative /= float(np.percentile(derivative, 95) + 1e-9)
        dog /= float(np.percentile(dog, 95) + 1e-9)
        envelopes.append(0.65 * derivative + 0.35 * dog)
    activity = np.median(np.vstack(envelopes), axis=0)
    edge = max(2, int(round(0.15 * values.size)))
    quiet = np.concatenate((activity[:edge], activity[-edge:]))
    low = max(float(np.quantile(quiet, 0.90)), 0.08 * float(np.max(activity)))
    high = max(float(np.quantile(quiet, 0.98)), 0.18 * float(np.max(activity)))
    active = activity >= low
    regions = _connected_regions(active, 0)
    candidates = [
        (start, stop)
        for start, stop in regions
        if start <= peak <= stop
        and float(np.max(activity[start : stop + 1])) >= high
    ]
    if not candidates:
        # Multi-phasic P waves can have a quiet notch at the amplitude peak.
        radius = max(2, int(round(0.030 * fs)))
        candidates = [
            (start, stop)
            for start, stop in regions
            if start - radius <= peak <= stop + radius
            and float(np.max(activity[start : stop + 1])) >= high
        ]
    if not candidates:
        return None, None
    onset = min(start for start, _ in candidates)
    offset = max(stop for _, stop in candidates)
    return int(onset), int(offset)


def _tangent_boundary(
    signal: np.ndarray,
    peak: int,
    fs: int,
    *,
    side: str,
    smooth: Optional[np.ndarray] = None,
    derivative: Optional[np.ndarray] = None,
) -> Optional[int]:
    values = np.asarray(signal, dtype=float)
    if not (0 <= peak < values.size):
        return None
    if smooth is None:
        smooth = gaussian_filter1d(
            values,
            sigma=max(0.5, 0.006 * fs),
            mode="nearest",
        )
    if derivative is None:
        derivative = np.gradient(smooth)
    if side == "onset":
        lo = max(1, peak - int(round(0.140 * fs)))
        hi = max(lo + 1, peak - max(1, int(round(0.010 * fs))))
        candidates = np.arange(lo, min(hi, peak), dtype=int)
    else:
        lo = min(values.size - 2, peak + max(1, int(round(0.010 * fs))))
        hi = min(values.size - 1, peak + int(round(0.160 * fs)))
        candidates = np.arange(lo, hi, dtype=int)
    if candidates.size < 2:
        return None
    polarity = 1.0 if float(smooth[peak]) >= 0.0 else -1.0
    signed = derivative[candidates] * polarity
    steep = (
        int(candidates[int(np.argmax(signed))])
        if side == "onset"
        else int(candidates[int(np.argmin(signed))])
    )
    half = max(2, int(round(0.006 * fs)))
    fit_lo = max(0, steep - half)
    fit_hi = min(values.size, steep + half + 1)
    axis = np.arange(fit_lo, fit_hi, dtype=float)
    if axis.size < 3:
        return None
    try:
        slope, intercept = np.polyfit(axis, smooth[fit_lo:fit_hi], 1)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if abs(float(slope)) <= 1e-9:
        return None
    crossing = int(round(-float(intercept) / float(slope)))
    if side == "onset":
        return int(np.clip(crossing, 0, peak - 1))
    return int(np.clip(crossing, peak + 1, values.size - 1))


def _change_point_boundary(
    signal: np.ndarray,
    peak: int,
    fs: int,
    *,
    side: str,
    energy: Optional[np.ndarray] = None,
) -> Optional[int]:
    values = np.asarray(signal, dtype=float)
    if energy is None:
        energy = gaussian_filter1d(
            np.gradient(values) ** 2,
            sigma=max(0.5, 0.006 * fs),
            mode="nearest",
        )
    window = max(3, int(round(0.012 * fs)))
    if side == "onset":
        candidates = np.arange(
            max(window, peak - int(round(0.140 * fs))),
            peak - window,
            dtype=int,
        )
    else:
        candidates = np.arange(
            peak + window,
            min(values.size - window, peak + int(round(0.160 * fs))),
            dtype=int,
        )
    if candidates.size == 0:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(energy, window)
    before = windows[candidates - window]
    after = windows[candidates]
    before_mean = np.mean(before, axis=1)
    after_mean = np.mean(after, axis=1)
    score = (
        after_mean - before_mean
        if side == "onset"
        else before_mean - after_mean
    )
    score /= 1.0 + np.std(before, axis=1) + np.std(after, axis=1)
    return int(candidates[int(np.argmax(score))])


def _area_bounds(
    signal: np.ndarray,
    peak: int,
) -> Tuple[Optional[int], Optional[int]]:
    values = np.abs(np.asarray(signal, dtype=float))
    if values.size < 4 or not (0 <= peak < values.size):
        return None, None
    total = float(np.sum(values))
    if total <= 1e-9:
        return None, None
    cumulative = np.cumsum(values)
    onset = int(np.searchsorted(cumulative, 0.02 * total))
    offset = int(np.searchsorted(cumulative, 0.98 * total))
    if not (onset < peak < offset):
        return None, None
    return onset, min(offset, values.size - 1)


def _method_boundaries(
    signal: np.ndarray,
    peak: int,
    fs: int,
    *,
    ta_aware: bool,
) -> Tuple[List[int], List[int], List[str]]:
    onsets: List[int] = []
    offsets: List[int] = []
    methods: List[str] = []
    for fraction in (0.04, 0.06, 0.08, 0.10, 0.12):
        onset, offset = _threshold_bounds(signal, peak, fraction)
        if onset is not None and onset < peak:
            onsets.append(int(onset))
        if not ta_aware and offset is not None and offset > peak:
            offsets.append(int(offset))
        methods.append(f"amplitude_hysteresis_{fraction:.2f}")
    activity_on, activity_off = _activity_bounds(signal, peak, fs)
    if activity_on is not None and activity_on < peak:
        onsets.append(int(activity_on))
    if activity_off is not None and activity_off > peak:
        offsets.append(int(activity_off))
    methods.append("multiscale_derivative_dog")
    values = np.asarray(signal, dtype=float)
    smooth = gaussian_filter1d(
        values,
        sigma=max(0.5, 0.006 * fs),
        mode="nearest",
    )
    derivative = np.gradient(smooth)
    tangent_on = _tangent_boundary(
        values,
        peak,
        fs,
        side="onset",
        smooth=smooth,
        derivative=derivative,
    )
    tangent_off = _tangent_boundary(
        values,
        peak,
        fs,
        side="offset",
        smooth=smooth,
        derivative=derivative,
    )
    if tangent_on is not None and tangent_on < peak:
        onsets.append(int(tangent_on))
    if tangent_off is not None and tangent_off > peak:
        offsets.append(int(tangent_off))
    methods.append("least_squares_tangent")
    energy = gaussian_filter1d(
        np.gradient(values) ** 2,
        sigma=max(0.5, 0.006 * fs),
        mode="nearest",
    )
    change_on = _change_point_boundary(
        values,
        peak,
        fs,
        side="onset",
        energy=energy,
    )
    change_off = _change_point_boundary(
        values,
        peak,
        fs,
        side="offset",
        energy=energy,
    )
    if change_on is not None and change_on < peak:
        onsets.append(int(change_on))
    if change_off is not None and change_off > peak:
        offsets.append(int(change_off))
    methods.append("local_energy_change_point")
    if not ta_aware:
        area_on, area_off = _area_bounds(signal, peak)
        if area_on is not None and area_on < peak:
            onsets.append(int(area_on))
        if area_off is not None and area_off > peak:
            offsets.append(int(area_off))
        methods.append("conditional_area")
    return onsets, offsets, methods


def _local_boundary_estimate(
    branches: Dict[str, np.ndarray],
    *,
    coarse_start: int,
    seed_peak: Optional[int],
    seed_onset: Optional[int],
    seed_offset: Optional[int],
    baseline_uncertainty: float,
    fs: int,
    ta_aware: bool,
    on_t_overlap: bool,
    t_reconstruction_validated: bool,
) -> Dict[str, object]:
    primary = branches.get("original_precision")
    if primary is None or primary.size < 8:
        return {"onset": None, "peak": None, "offset": None}
    local_seed = (
        int(seed_peak) - int(coarse_start)
        if seed_peak is not None
        and coarse_start <= int(seed_peak) < coarse_start + primary.size
        else None
    )
    smoothed = gaussian_filter1d(
        primary,
        sigma=max(0.5, 0.008 * fs),
        mode="nearest",
    )
    peak = local_seed if local_seed is not None else int(np.argmax(np.abs(smoothed)))
    if not (0 < peak < primary.size - 1):
        return {"onset": None, "peak": None, "offset": None}

    onset_candidates: List[int] = []
    offset_candidates: List[int] = []
    branch_onsets: List[int] = []
    branch_offsets: List[int] = []
    method_names: List[str] = []
    if seed_onset is not None and coarse_start <= int(seed_onset) < coarse_start + primary.size:
        onset_candidates.append(int(seed_onset) - coarse_start)
        method_names.append("legacy_wideband_seed")
    if seed_offset is not None and coarse_start <= int(seed_offset) < coarse_start + primary.size:
        offset_candidates.append(int(seed_offset) - coarse_start)

    for branch_name, branch_signal in branches.items():
        if branch_signal.size != primary.size:
            continue
        branch_peak = int(
            np.argmax(
                np.abs(
                    gaussian_filter1d(
                        branch_signal,
                        sigma=max(0.5, 0.008 * fs),
                        mode="nearest",
                    )
                )
            )
        )
        if abs(branch_peak - peak) > int(round(0.035 * fs)):
            branch_peak = peak
        onsets, offsets, methods = _method_boundaries(
            branch_signal,
            branch_peak,
            fs,
            ta_aware=bool(ta_aware and branch_name != "t_reconstruction"),
        )
        if onsets:
            branch_onset = int(round(float(np.median(onsets))))
            onset_candidates.extend(onsets)
            branch_onsets.append(branch_onset)
        if offsets:
            branch_offset = int(round(float(np.median(offsets))))
            offset_candidates.extend(offsets)
            branch_offsets.append(branch_offset)
        method_names.extend(f"{branch_name}:{method}" for method in methods)

    # Explicit baseline perturbation transfers state-space uncertainty into the
    # empirical endpoint distribution.
    for sign in (-1.0, 1.0):
        perturbed = primary - sign * float(baseline_uncertainty)
        for fraction in (0.05, 0.10):
            onset, offset = _threshold_bounds(perturbed, peak, fraction)
            if onset is not None and onset < peak:
                onset_candidates.append(int(onset))
            if offset is not None and offset > peak and not ta_aware:
                offset_candidates.append(int(offset))
    if not onset_candidates or not offset_candidates:
        return {
            "onset": None,
            "peak": coarse_start + peak,
            "offset": None,
            "methods": sorted(set(method_names)),
        }

    onset = int(round(float(np.median(onset_candidates))))
    offset = int(round(float(np.median(offset_candidates))))
    if not (0 <= onset < peak < offset < primary.size):
        return {
            "onset": None,
            "peak": coarse_start + peak,
            "offset": None,
            "methods": sorted(set(method_names)),
        }
    duration_ms = (offset - onset) * 1000.0 / float(fs)
    if not 30.0 <= duration_ms <= 200.0:
        return {
            "onset": None,
            "peak": coarse_start + peak,
            "offset": None,
            "methods": sorted(set(method_names)),
        }
    onset_sigma = _robust_sigma(
        [value * 1000.0 / fs for value in onset_candidates],
        floor=1.0,
    )
    offset_sigma = _robust_sigma(
        [value * 1000.0 / fs for value in offset_candidates],
        floor=1.0,
    )
    branch_spreads = []
    if len(branch_onsets) >= 2:
        branch_spreads.append(max(branch_onsets) - min(branch_onsets))
    if len(branch_offsets) >= 2:
        branch_spreads.append(max(branch_offsets) - min(branch_offsets))
    branch_disagreement_ms = (
        float(max(branch_spreads)) * 1000.0 / fs
        if branch_spreads
        else None
    )
    minimum_half = max(1, int(round(0.004 * fs)))
    onset_ci_low = min(onset_candidates) if onset_candidates else onset - minimum_half
    onset_ci_high = max(onset_candidates) if onset_candidates else onset + minimum_half
    offset_ci_low = min(offset_candidates) if offset_candidates else offset - minimum_half
    offset_ci_high = max(offset_candidates) if offset_candidates else offset + minimum_half
    if onset_ci_high - onset_ci_low < 2 * minimum_half:
        onset_ci_low = onset - minimum_half
        onset_ci_high = onset + minimum_half
    if offset_ci_high - offset_ci_low < 2 * minimum_half:
        offset_ci_low = offset - minimum_half
        offset_ci_high = offset + minimum_half
    model_disagreement = bool(
        on_t_overlap
        and (
            not t_reconstruction_validated
            or (
                branch_disagreement_ms is not None
                and branch_disagreement_ms > 30.0
            )
        )
    )
    return {
        "onset": coarse_start + onset,
        "peak": coarse_start + peak,
        "offset": coarse_start + offset,
        "onset_ci_low": coarse_start + int(onset_ci_low),
        "onset_ci_high": coarse_start + int(onset_ci_high),
        "offset_ci_low": coarse_start + int(offset_ci_low),
        "offset_ci_high": coarse_start + int(offset_ci_high),
        "onset_sigma_ms": float(onset_sigma),
        "offset_sigma_ms": float(offset_sigma),
        "branch_disagreement_ms": branch_disagreement_ms,
        "model_disagreement": model_disagreement,
        "methods": sorted(set(method_names)),
    }


def _local_quiet_noise(
    signal: np.ndarray,
    quiet_mask: np.ndarray,
    *,
    coarse_start: int,
    fs: int,
) -> Tuple[Optional[float], str]:
    """Measure P noise only in verified quiet samples, never in the PR segment."""
    lo = max(0, int(coarse_start) - int(round(0.180 * fs)))
    hi = max(lo, int(coarse_start) - int(round(0.012 * fs)))
    local_mask = np.asarray(quiet_mask[lo:hi], dtype=bool)
    segment = np.asarray(signal[lo:hi], dtype=float)
    if segment.size and int(np.sum(local_mask)) >= max(6, int(round(0.020 * fs))):
        quiet = segment[local_mask]
        sigma = _robust_sigma(
            np.diff(quiet) if quiet.size > 2 else quiet,
            floor=1e-5,
        ) / np.sqrt(2.0)
        return float(sigma), "verified_local_quiet_samples"
    return None, "unavailable_no_verified_quiet_samples"


def _lead_boundary(
    *,
    lead: str,
    feature: Optional[object],
    branches: Dict[str, np.ndarray],
    corrected_precision_lead: np.ndarray,
    quiet_mask: np.ndarray,
    coarse_start: int,
    coarse_stop: int,
    baseline_uncertainty: np.ndarray,
    observation_distance: np.ndarray,
    quality: Dict[str, object],
    excluded_leads: set[str],
    fs: int,
    config: PWaveConfig,
    on_t_overlap: bool,
    t_reconstruction_validated: bool,
    ta_ambiguous: bool,
) -> PWaveLeadBoundary:
    q = quality.get(lead)
    hard_quality_pass = bool(
        lead not in excluded_leads
        and q is not None
        and bool(getattr(q, "reliable_for_p", getattr(q, "reliable", False)))
    )
    midpoint = int(np.clip((coarse_start + coarse_stop) // 2, 0, observation_distance.size - 1))
    distance_ms = float(observation_distance[midpoint]) * 1000.0 / float(fs)
    local_observed = bool(distance_ms <= 120.0)
    baseline_mode = (
        "tp_observed_state_space"
        if local_observed
        else "state_space_prediction"
        if distance_ms <= config.baseline_unobservable_ms
        else "baseline_unobservable"
    )
    baseline_confidence = float(
        np.clip(1.0 - distance_ms / max(config.baseline_unobservable_ms, 1.0), 0.0, 1.0)
    )
    uncertainty_start = max(0, coarse_start)
    uncertainty_stop = min(baseline_uncertainty.size, coarse_stop + 1)
    uncertainty_window = baseline_uncertainty[uncertainty_start:uncertainty_stop]
    if uncertainty_window.size:
        local_uncertainty = float(np.median(uncertainty_window))
    elif baseline_uncertainty.size:
        # Paced/overlap guards can collapse the coarse interval.  Preserve a
        # local uncertainty estimate without asking NumPy for the median of an
        # empty slice (which silently creates NaN and leaks into confidence).
        uncertainty_index = int(
            np.clip(midpoint, 0, baseline_uncertainty.size - 1)
        )
        local_uncertainty = float(baseline_uncertainty[uncertainty_index])
    else:
        local_uncertainty = 0.0
    seed_p = getattr(feature, "p", None)
    estimate = _local_boundary_estimate(
        branches,
        coarse_start=coarse_start,
        seed_peak=getattr(seed_p, "peak", None),
        seed_onset=(
            getattr(feature, "p_onset_raw_index", None)
            or getattr(seed_p, "onset", None)
        ),
        seed_offset=(
            getattr(feature, "p_offset_raw_index", None)
            or getattr(seed_p, "offset", None)
        ),
        baseline_uncertainty=local_uncertainty,
        fs=fs,
        ta_aware=ta_ambiguous,
        on_t_overlap=on_t_overlap,
        t_reconstruction_validated=t_reconstruction_validated,
    )
    onset = estimate.get("onset")
    peak = estimate.get("peak")
    offset = estimate.get("offset")
    noise, noise_source = _local_quiet_noise(
        corrected_precision_lead,
        quiet_mask,
        coarse_start=coarse_start,
        fs=fs,
    )
    local_snr: Optional[float] = None
    if noise is not None and peak is not None:
        peak_value = abs(float(corrected_precision_lead[int(peak)]))
        local_snr = float(peak_value / max(noise, 1e-9))

    legacy_presence = float(getattr(feature, "p_confidence", 0.0) or 0.0)
    template_corr = float(getattr(feature, "p_template_corr", 0.0) or 0.0)
    onset_sigma = estimate.get("onset_sigma_ms")
    offset_sigma = estimate.get("offset_sigma_ms")
    onset_stability = (
        float(np.exp(-float(onset_sigma) / 24.0))
        if onset_sigma is not None
        else 0.0
    )
    offset_stability = (
        float(np.exp(-float(offset_sigma) / 24.0))
        if offset_sigma is not None
        else 0.0
    )
    snr_score = (
        float(np.clip((float(local_snr) - 1.0) / 5.0, 0.0, 1.0))
        if local_snr is not None
        else 0.35 if t_reconstruction_validated else 0.0
    )
    quality_score = float(
        np.clip(
            0.28 * legacy_presence
            + 0.20 * max(0.0, template_corr)
            + 0.20 * snr_score
            + 0.16 * onset_stability
            + 0.16 * offset_stability,
            0.0,
            1.0,
        )
    )
    has_boundaries = bool(
        onset is not None
        and peak is not None
        and offset is not None
        and int(onset) < int(peak) < int(offset)
    )
    snr_pass = bool(
        local_snr is not None
        and float(local_snr) >= float(config.minimum_informative_snr)
    )
    prediction_pass = bool(
        local_snr is None
        and baseline_mode != "baseline_unobservable"
        and t_reconstruction_validated
        and estimate.get("branch_disagreement_ms") is not None
        and float(estimate["branch_disagreement_ms"]) <= config.branch_disagreement_ms
    )
    informative = bool(
        hard_quality_pass
        and has_boundaries
        and not bool(estimate.get("model_disagreement"))
        and (snr_pass or prediction_pass)
    )
    onset_confidence = float(
        np.clip(
            quality_score
            * baseline_confidence
            * (0.65 + 0.35 * onset_stability),
            0.0,
            1.0,
        )
    )
    offset_confidence = float(
        np.clip(
            quality_score
            * baseline_confidence
            * (0.65 + 0.35 * offset_stability),
            0.0,
            1.0,
        )
    )
    if on_t_overlap:
        onset_confidence = min(onset_confidence, 0.55 if t_reconstruction_validated else 0.35)
        offset_confidence = min(offset_confidence, 0.55 if t_reconstruction_validated else 0.35)
    if ta_ambiguous:
        offset_confidence = min(offset_confidence, 0.45)

    flags: List[str] = []
    if not hard_quality_pass:
        flags.append("hard_quality_failed")
    if baseline_mode == "baseline_unobservable":
        flags.append(BASELINE_UNOBSERVABLE)
    if noise is None:
        flags.append("local_snr_unavailable")
    elif not snr_pass:
        flags.append("local_snr_below_threshold")
    if on_t_overlap and not t_reconstruction_validated:
        flags.append(P_ON_T_UNRESOLVED)
    if ta_ambiguous:
        flags.append(P_OFFSET_TA_AMBIGUOUS)
    if bool(estimate.get("model_disagreement")):
        flags.append(MODEL_DISAGREEMENT)
    if not has_boundaries:
        flags.append("boundary_unavailable")
    flags.append(f"noise_source:{noise_source}")
    return PWaveLeadBoundary(
        lead=lead,
        onset=int(onset) if onset is not None else None,
        peak=int(peak) if peak is not None else None,
        offset=int(offset) if offset is not None else None,
        onset_ci_low=(
            int(estimate["onset_ci_low"])
            if estimate.get("onset_ci_low") is not None
            else None
        ),
        onset_ci_high=(
            int(estimate["onset_ci_high"])
            if estimate.get("onset_ci_high") is not None
            else None
        ),
        offset_ci_low=(
            int(estimate["offset_ci_low"])
            if estimate.get("offset_ci_low") is not None
            else None
        ),
        offset_ci_high=(
            int(estimate["offset_ci_high"])
            if estimate.get("offset_ci_high") is not None
            else None
        ),
        onset_confidence=onset_confidence,
        offset_confidence=offset_confidence,
        onset_sigma_ms=float(onset_sigma) if onset_sigma is not None else None,
        offset_sigma_ms=float(offset_sigma) if offset_sigma is not None else None,
        quality_score=quality_score,
        local_snr=local_snr,
        informative=informative,
        hard_quality_pass=hard_quality_pass,
        baseline_mode=baseline_mode,
        baseline_confidence=baseline_confidence,
        baseline_uncertainty_mv=local_uncertainty,
        t_reconstructed="t_reconstruction" in branches,
        t_reconstruction_validated=t_reconstruction_validated,
        ta_ambiguous=ta_ambiguous,
        on_t_overlap=on_t_overlap,
        branch_disagreement_ms=(
            float(estimate["branch_disagreement_ms"])
            if estimate.get("branch_disagreement_ms") is not None
            else None
        ),
        methods=list(estimate.get("methods") or []),
        flags=sorted(set(flags)),
    )


def _clusters(
    rows: Sequence[PWaveLeadBoundary],
    *,
    boundary: str,
    fs: int,
    radius_ms: float,
) -> List[List[PWaveLeadBoundary]]:
    ordered = sorted(
        (
            row
            for row in rows
            if getattr(row, boundary) is not None
        ),
        key=lambda row: int(getattr(row, boundary)),
    )
    radius = max(1, int(round(radius_ms * fs / 1000.0)))
    out: List[List[PWaveLeadBoundary]] = []
    for row in ordered:
        value = int(getattr(row, boundary))
        placed = False
        for cluster in out:
            # Anchor-to-candidate comparison prevents single-linkage chaining.
            anchor = int(getattr(cluster[0], boundary))
            if abs(value - anchor) <= radius:
                cluster.append(row)
                placed = True
                break
        if not placed:
            out.append([row])
    return out


def _eligible_cluster(
    rows: Sequence[PWaveLeadBoundary],
    *,
    minimum_leads: int,
    minimum_groups: int,
) -> bool:
    return bool(
        len({row.lead for row in rows}) >= minimum_leads
        and len({_lead_group(row.lead) for row in rows}) >= minimum_groups
    )


def _group_capped_weights(
    rows: Sequence[PWaveLeadBoundary],
    *,
    boundary: str,
) -> List[float]:
    raw = []
    for row in rows:
        sigma = (
            row.onset_sigma_ms
            if boundary == "onset"
            else row.offset_sigma_ms
        )
        confidence = (
            row.onset_confidence
            if boundary == "onset"
            else row.offset_confidence
        )
        raw.append(float(confidence) / max(float(sigma or 20.0), 4.0))
    weights = list(raw)
    for group in set(_lead_group(row.lead) for row in rows):
        indices = [
            index
            for index, row in enumerate(rows)
            if _lead_group(row.lead) == group
        ]
        total = float(sum(raw[index] for index in indices))
        scale = 1.0 / total if total > 1.0 else 1.0
        for index in indices:
            weights[index] = raw[index] * scale
    return weights


def _fuse_boundary(
    rows: Sequence[PWaveLeadBoundary],
    *,
    boundary: str,
    fs: int,
    config: PWaveConfig,
) -> Dict[str, object]:
    clusters = _clusters(
        rows,
        boundary=boundary,
        fs=fs,
        radius_ms=config.cluster_radius_ms,
    )
    eligible = [
        cluster
        for cluster in clusters
        if _eligible_cluster(
            cluster,
            minimum_leads=config.minimum_informative_leads,
            minimum_groups=config.minimum_independent_groups,
        )
    ]
    if not eligible:
        return {"value": None, "cluster": [], "spread_ms": None}
    selected = eligible[0] if boundary == "onset" else eligible[-1]
    values = [float(getattr(row, boundary)) for row in selected]
    weights = _group_capped_weights(selected, boundary=boundary)
    quantile = 0.35 if boundary == "onset" else 0.65
    robust = _weighted_quantile(values, weights, quantile)
    strict = min(values) if boundary == "onset" else max(values)
    ci_lows = [
        getattr(row, f"{boundary}_ci_low")
        for row in selected
        if getattr(row, f"{boundary}_ci_low") is not None
    ]
    ci_highs = [
        getattr(row, f"{boundary}_ci_high")
        for row in selected
        if getattr(row, f"{boundary}_ci_high") is not None
    ]
    distribution_low = _weighted_quantile(values, weights, 0.10)
    distribution_high = _weighted_quantile(values, weights, 0.90)
    minimum_half = max(
        1,
        int(round(config.minimum_ci_half_width_ms * fs / 1000.0)),
    )
    center = int(round(float(robust)))
    ci_low = int(
        min(
            list(ci_lows or [center - minimum_half])
            + [distribution_low if distribution_low is not None else center]
        )
    )
    ci_high = int(
        max(
            list(ci_highs or [center + minimum_half])
            + [distribution_high if distribution_high is not None else center]
        )
    )
    ci_low = min(ci_low, center - minimum_half)
    ci_high = max(ci_high, center + minimum_half)
    spread_ms = float(max(values) - min(values)) * 1000.0 / float(fs)
    support_score = min(
        1.0,
        len(selected) / max(float(config.minimum_informative_leads + 2), 1.0),
    )
    compactness = float(
        np.clip(1.0 - spread_ms / max(config.maximum_cluster_spread_ms, 1.0), 0.0, 1.0)
    )
    row_confidence = float(
        np.average(
            [
                row.onset_confidence
                if boundary == "onset"
                else row.offset_confidence
                for row in selected
            ],
            weights=np.maximum(np.asarray(weights), 1e-9),
        )
    )
    confidence = float(
        np.clip(
            row_confidence * (0.55 + 0.25 * support_score + 0.20 * compactness),
            0.0,
            1.0,
        )
    )
    return {
        "value": center,
        "strict": int(round(strict)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "cluster": selected,
        "spread_ms": spread_ms,
    }


def _fuse_beat(
    *,
    beat_id: int,
    per_lead: Dict[str, PWaveLeadBoundary],
    fs: int,
    coarse_start: int,
    coarse_stop: int,
    svd_dimension: int,
    acquisition_qc: Dict[str, object],
    t_reconstruction_available: bool,
    config: PWaveConfig,
) -> PWaveBeatAssessment:
    informative = [row for row in per_lead.values() if row.informative]
    onset_result = _fuse_boundary(
        informative,
        boundary="onset",
        fs=fs,
        config=config,
    )
    offset_result = _fuse_boundary(
        informative,
        boundary="offset",
        fs=fs,
        config=config,
    )
    reject_reasons = list(acquisition_qc.get("reject_reasons") or [])
    valid_groups = sorted({_lead_group(row.lead) for row in informative})
    if (
        len(informative) < config.minimum_informative_leads
        or len(valid_groups) < config.minimum_independent_groups
        or onset_result["value"] is None
        or offset_result["value"] is None
    ):
        reject_reasons.append(INSUFFICIENT_INFORMATIVE_LEADS)
    if (
        onset_result.get("spread_ms") is not None
        and float(onset_result["spread_ms"]) > config.maximum_cluster_spread_ms
    ) or (
        offset_result.get("spread_ms") is not None
        and float(offset_result["spread_ms"]) > config.maximum_cluster_spread_ms
    ):
        reject_reasons.append(CROSS_LEAD_DISAGREEMENT)
    baseline_modes = [row.baseline_mode for row in informative]
    if not baseline_modes or all(mode == "baseline_unobservable" for mode in baseline_modes):
        reject_reasons.append(BASELINE_UNOBSERVABLE)
    on_t_rows = [row for row in informative if row.on_t_overlap]
    if on_t_rows and not any(row.t_reconstruction_validated for row in on_t_rows):
        reject_reasons.append(P_ON_T_UNRESOLVED)
    ta_rows = [row for row in informative if row.ta_ambiguous]
    unstable_ta_rows = [
        row
        for row in ta_rows
        if row.branch_disagreement_ms is None
        or row.branch_disagreement_ms > config.branch_disagreement_ms
    ]
    # A single flagged lead out of twelve is not beat-level ambiguity: consumers
    # treat `ta_ambiguous` as blocking for P-morphology conclusions, so it must
    # mean the offset is genuinely at risk on this beat, not that one lead saw a
    # Ta deflection. Require the same quorum of *unstable* rows the rejection
    # path uses, minus its cross-lead-spread condition, so an ambiguous beat can
    # still be accepted.
    ta_ambiguous_beat = len(unstable_ta_rows) >= max(2, len(informative) // 2)
    if (
        unstable_ta_rows
        and len(unstable_ta_rows) >= max(2, len(informative) // 2)
        and (
            offset_result.get("spread_ms") is None
            or float(offset_result["spread_ms"]) > 0.60 * config.maximum_cluster_spread_ms
        )
    ):
        reject_reasons.append(P_OFFSET_TA_AMBIGUOUS)
    if any(MODEL_DISAGREEMENT in row.flags for row in informative):
        reject_reasons.append(MODEL_DISAGREEMENT)

    robust_onset = onset_result.get("value")
    robust_offset = offset_result.get("value")
    strict_onset = onset_result.get("strict")
    strict_offset = offset_result.get("strict")
    if robust_onset is not None and robust_offset is not None:
        if int(robust_onset) >= int(robust_offset):
            reject_reasons.append(MODEL_DISAGREEMENT)
        strict_robust = max(
            abs(int(strict_onset) - int(robust_onset))
            if strict_onset is not None
            else 0,
            abs(int(strict_offset) - int(robust_offset))
            if strict_offset is not None
            else 0,
        ) * 1000.0 / float(fs)
        if strict_robust > config.strict_robust_disagreement_ms:
            reject_reasons.append(MODEL_DISAGREEMENT)
    reject_reasons = sorted(set(str(reason) for reason in reject_reasons))
    accepted = bool(
        robust_onset is not None
        and robust_offset is not None
        and not reject_reasons
    )
    baseline_mode = (
        "tp_observed_state_space"
        if any(mode == "tp_observed_state_space" for mode in baseline_modes)
        else "state_space_prediction"
        if any(mode == "state_space_prediction" for mode in baseline_modes)
        else "baseline_unobservable"
    )
    valid_leads = sorted(
        {
            row.lead
            for row in list(onset_result.get("cluster") or [])
            + list(offset_result.get("cluster") or [])
        },
        key=lambda lead: STANDARD_12_LEADS.index(lead)
        if lead in STANDARD_12_LEADS
        else len(STANDARD_12_LEADS),
    )
    return PWaveBeatAssessment(
        beat_id=int(beat_id),
        strict_onset=int(strict_onset) if strict_onset is not None else None,
        strict_offset=int(strict_offset) if strict_offset is not None else None,
        robust_onset=int(robust_onset) if robust_onset is not None else None,
        robust_offset=int(robust_offset) if robust_offset is not None else None,
        onset_ci_low=(
            int(onset_result["ci_low"])
            if onset_result.get("ci_low") is not None
            else None
        ),
        onset_ci_high=(
            int(onset_result["ci_high"])
            if onset_result.get("ci_high") is not None
            else None
        ),
        offset_ci_low=(
            int(offset_result["ci_low"])
            if offset_result.get("ci_low") is not None
            else None
        ),
        offset_ci_high=(
            int(offset_result["ci_high"])
            if offset_result.get("ci_high") is not None
            else None
        ),
        onset_confidence=float(onset_result.get("confidence") or 0.0),
        offset_confidence=float(offset_result.get("confidence") or 0.0),
        p_state=P_PRESENT if accepted else OVERLAP_UNCERTAIN,
        accepted=accepted,
        reject_reasons=reject_reasons,
        valid_leads=valid_leads,
        valid_lead_groups=sorted({_lead_group(lead) for lead in valid_leads}),
        per_lead=per_lead,
        baseline_mode=baseline_mode,
        t_reconstructed=t_reconstruction_available,
        ta_ambiguous=ta_ambiguous_beat,
        global_detector_method="whitened_spatial_derivative_hysteresis",
        svd_dimension=int(svd_dimension),
        coarse_window_start=int(coarse_start),
        coarse_window_end=int(coarse_stop),
        ci_calibration_status="empirical_perturbation_uncalibrated",
    )


def _previous_t_offset(
    beat_id: int,
    by_beat: Dict[int, List[object]],
) -> Optional[int]:
    if beat_id <= 0:
        return None
    return _median_index(
        getattr(getattr(item, "t", None), "offset", None)
        for item in by_beat.get(beat_id - 1, [])
    )


def _assign_temporal_clusters(
    assessments: Sequence[PWaveBeatAssessment],
    r_locs: np.ndarray,
    fs: int,
) -> None:
    prototypes: List[np.ndarray] = []
    members: Dict[int, List[PWaveBeatAssessment]] = {}
    radius = max(12.0, 0.025 * 1000.0)
    for assessment in assessments:
        if not assessment.accepted:
            continue
        r_sample = int(r_locs[assessment.beat_id])
        vector = np.asarray(
            [
                (int(assessment.robust_onset) - r_sample) * 1000.0 / fs,
                (int(assessment.robust_offset) - r_sample) * 1000.0 / fs,
            ],
            dtype=float,
        )
        distances = [
            float(np.max(np.abs(vector - prototype)))
            for prototype in prototypes
        ]
        if distances and min(distances) <= radius:
            cluster_id = int(np.argmin(distances))
            existing = members[cluster_id]
            n = len(existing)
            prototypes[cluster_id] = (n * prototypes[cluster_id] + vector) / (n + 1)
        else:
            cluster_id = len(prototypes)
            prototypes.append(vector)
            members[cluster_id] = []
        assessment.morphology_cluster_id = cluster_id
        members.setdefault(cluster_id, []).append(assessment)

    for cluster_id, rows in members.items():
        if len(rows) < 2:
            rows[0].temporal_jitter_ms = None
            continue
        onset_offsets = [
            (int(row.robust_onset) - int(r_locs[row.beat_id])) * 1000.0 / fs
            for row in rows
        ]
        offset_offsets = [
            (int(row.robust_offset) - int(r_locs[row.beat_id])) * 1000.0 / fs
            for row in rows
        ]
        jitter = max(
            _robust_sigma(onset_offsets, floor=0.0),
            _robust_sigma(offset_offsets, floor=0.0),
        )
        pad = int(round(jitter * fs / 1000.0))
        for row in rows:
            row.temporal_jitter_ms = float(jitter)
            if pad > 0:
                if row.onset_ci_low is not None:
                    row.onset_ci_low -= pad
                if row.onset_ci_high is not None:
                    row.onset_ci_high += pad
                if row.offset_ci_low is not None:
                    row.offset_ci_low -= pad
                if row.offset_ci_high is not None:
                    row.offset_ci_high += pad


def _update_legacy_consensus(
    assessment: PWaveBeatAssessment,
    items: Sequence[object],
    fs: int,
) -> None:
    if not assessment.accepted:
        return
    onset_cluster = [
        row.onset
        for row in assessment.per_lead.values()
        if row.lead in assessment.valid_leads and row.onset is not None
    ]
    offset_cluster = [
        row.offset
        for row in assessment.per_lead.values()
        if row.lead in assessment.valid_leads and row.offset is not None
    ]
    onset_spread = (
        float(max(onset_cluster) - min(onset_cluster)) * 1000.0 / fs
        if len(onset_cluster) >= 2
        else 0.0
    )
    offset_spread = (
        float(max(offset_cluster) - min(offset_cluster)) * 1000.0 / fs
        if len(offset_cluster) >= 2
        else 0.0
    )
    duration = (
        float(int(assessment.robust_offset) - int(assessment.robust_onset))
        * 1000.0
        / fs
    )
    for feature in items:
        feature.p_onset_consensus_index = int(assessment.robust_onset)
        feature.p_offset_consensus_index = int(assessment.robust_offset)
        feature.p_dur_consensus_ms = duration
        feature.p_onset_consensus_support = len(onset_cluster)
        feature.p_offset_consensus_support = len(offset_cluster)
        feature.p_onset_consensus_spread_ms = onset_spread
        feature.p_offset_consensus_spread_ms = offset_spread
        feature.p_boundary_consensus_source = "robust_independent_group_fusion"
        feature.p_onset_consensus_reason = "accepted_p_wave_contract"


def _reconcile_parallel_consensus_models(
    assessment: PWaveBeatAssessment,
    *,
    legacy_onset: Optional[int],
    legacy_offset: Optional[int],
    fs: int,
    evidence_arbitration: bool = False,
    legacy_confidence: float = 0.5,
) -> None:
    """Require the established and robust models to corroborate one another.

    The established consensus is strong at avoiding gross false offsets, while
    the perturbation/SVD model supplies independent boundary evidence and
    uncertainty. Onset is averaged; offset deliberately gives 87.5% weight to
    the later of the two estimates to avoid the known P-duration contraction.
    """
    if legacy_onset is None or legacy_offset is None:
        # No second opinion to reconcile against -- this is not the same as
        # the two models disagreeing. Leave _fuse_beat's own cross-lead
        # verdict (>=3 informative leads, >=2 independent groups, no cluster
        # disagreement, observed baseline, resolved P-on-T) as the answer
        # instead of forcing a reject; that verdict is exactly the case this
        # legacy-required veto was silently overriding whenever the older
        # single/multi-lead consensus missed (short RR / short TP interval,
        # where it is known to be unreliable -- see
        # project_ecgfeat_p_sensitivity_root_cause in the repo's engineering
        # memory).
        assessment.global_detector_method += "+legacy_consensus_unavailable"
        return
    if not assessment.accepted:
        return
    robust_onset = int(assessment.robust_onset)
    robust_offset = int(assessment.robust_offset)
    if evidence_arbitration:
        # The interval remains an uncalibrated model-disagreement envelope;
        # moving a point estimate must never leave it outside its own bounds.
        for boundary, robust, legacy in (("onset", robust_onset, int(legacy_onset)),
                                         ("offset", robust_offset, int(legacy_offset))):
            low = getattr(assessment, boundary + "_ci_low")
            high = getattr(assessment, boundary + "_ci_high")
            setattr(assessment, boundary + "_ci_low", min(robust, legacy, low if low is not None else robust))
            setattr(assessment, boundary + "_ci_high", max(robust, legacy, high if high is not None else robust))
        assessment.strict_onset = min(robust_onset, int(legacy_onset),
                                     assessment.strict_onset if assessment.strict_onset is not None else robust_onset)
        assessment.strict_offset = max(robust_offset, int(legacy_offset),
                                      assessment.strict_offset if assessment.strict_offset is not None else robust_offset)
        gap = max(abs(robust_onset - int(legacy_onset)),
                  abs(robust_offset - int(legacy_offset))) * 1000.0 / fs
        robust_confidence = min(assessment.onset_confidence, assessment.offset_confidence)
        if gap > 30.0:
            strong = (robust_confidence >= 0.65 and len(assessment.valid_leads) >= 4
                      and len(assessment.valid_lead_groups) >= 2)
            if strong and legacy_confidence < 0.45:
                assessment.global_detector_method += "+robust_evidence_selected"
                return
            if legacy_confidence >= 0.65 and robust_confidence < 0.45:
                assessment.robust_onset, assessment.robust_offset = int(legacy_onset), int(legacy_offset)
                assessment.global_detector_method += "+legacy_evidence_selected"
                return
            # Distinct wave hypotheses must not be averaged into a third,
            # unsupported location. Retain evidence but decline acceptance.
            assessment.accepted = False
            assessment.p_state = OVERLAP_UNCERTAIN
            assessment.reject_reasons = sorted(set(assessment.reject_reasons + [MODEL_DISAGREEMENT]))
            assessment.global_detector_method += "+unresolved_model_disagreement"
            return
        weight = robust_confidence / max(robust_confidence + legacy_confidence, 1e-9)
        assessment.robust_onset = int(round(weight * robust_onset + (1 - weight) * legacy_onset))
        assessment.robust_offset = int(round(weight * robust_offset + (1 - weight) * legacy_offset))
        assessment.global_detector_method += "+evidence_weighted_agreement"
        return
    reconciled_onset = int(round(0.50 * robust_onset + 0.50 * int(legacy_onset)))
    early_offset = min(robust_offset, int(legacy_offset))
    late_offset = max(robust_offset, int(legacy_offset))
    # 25% of the equal-weight model average plus 75% of the late-preserving
    # estimate is equivalent to 12.5% early / 87.5% late.
    model_average_offset = 0.50 * (early_offset + late_offset)
    reconciled_offset = int(
        round(0.25 * model_average_offset + 0.75 * late_offset)
    )
    duration_ms = (
        reconciled_offset - reconciled_onset
    ) * 1000.0 / float(fs)
    if not (
        reconciled_onset < reconciled_offset
        and 30.0 <= duration_ms <= 200.0
    ):
        assessment.accepted = False
        assessment.p_state = OVERLAP_UNCERTAIN
        assessment.reject_reasons = sorted(
            set(assessment.reject_reasons + [MODEL_DISAGREEMENT])
        )
        return
    onset_disagreement_ms = (
        abs(robust_onset - int(legacy_onset)) * 1000.0 / float(fs)
    )
    offset_disagreement_ms = (
        abs(robust_offset - int(legacy_offset)) * 1000.0 / float(fs)
    )
    assessment.robust_onset = reconciled_onset
    assessment.robust_offset = reconciled_offset
    assessment.strict_onset = min(
        int(assessment.strict_onset)
        if assessment.strict_onset is not None
        else robust_onset,
        int(legacy_onset),
    )
    assessment.strict_offset = max(
        int(assessment.strict_offset)
        if assessment.strict_offset is not None
        else robust_offset,
        int(legacy_offset),
    )
    assessment.onset_ci_low = min(
        assessment.onset_ci_low
        if assessment.onset_ci_low is not None
        else reconciled_onset,
        int(legacy_onset),
        reconciled_onset,
    )
    assessment.onset_ci_high = max(
        assessment.onset_ci_high
        if assessment.onset_ci_high is not None
        else reconciled_onset,
        int(legacy_onset),
        reconciled_onset,
    )
    assessment.offset_ci_low = min(
        assessment.offset_ci_low
        if assessment.offset_ci_low is not None
        else reconciled_offset,
        int(legacy_offset),
        reconciled_offset,
    )
    assessment.offset_ci_high = max(
        assessment.offset_ci_high
        if assessment.offset_ci_high is not None
        else reconciled_offset,
        int(legacy_offset),
        reconciled_offset,
    )
    assessment.onset_confidence *= float(
        np.exp(-onset_disagreement_ms / 120.0)
    )
    assessment.offset_confidence *= float(
        np.exp(-offset_disagreement_ms / 120.0)
    )
    assessment.global_detector_method += "+parallel_legacy_model_reconciliation"


def _assessment_score(assessment: PWaveBeatAssessment) -> float:
    return (2.0 * float(assessment.accepted)
            + min(assessment.onset_confidence, assessment.offset_confidence)
            + 0.05 * min(len(assessment.valid_leads), 8)
            - 0.1 * len(assessment.reject_reasons))


def _alternate_p_windows(activity, r_locs, beat_id, fs, thresholds, original):
    r = int(r_locs[beat_id])
    rr = r - int(r_locs[beat_id - 1]) if beat_id else int(.65 * fs)
    lo, hi = max(0, r - min(int(.45 * fs), int(.65 * rr))), max(0, r - int(.04 * fs))
    regions = _connected_regions(activity[lo:hi] >= thresholds[0], lo)
    ranked = []
    for start, stop in regions:
        if np.max(activity[start:stop + 1]) < thresholds[1]:
            continue
        start, stop = max(lo, start - int(.025 * fs)), min(hi, stop + int(.025 * fs))
        if stop - start < int(.05 * fs):
            continue
        overlap = max(0, min(stop, original[1]) - max(start, original[0]))
        if overlap >= .6 * min(stop - start, original[1] - original[0]):
            continue
        ranked.append((float(np.sum(activity[start:stop + 1])), start, stop))
    return [(start, stop) for _, start, stop in sorted(ranked, reverse=True)[:2]]


def build_p_wave_assessments(
    ecg: np.ndarray,
    fs: int,
    r_locs: np.ndarray,
    beat_features: Sequence[object],
    quality: Dict[str, object],
    *,
    beat_groups: Optional[Dict[int, Sequence[int]]] = None,
    acquisition_qc: Optional[Dict[str, object]] = None,
    config: Optional[PWaveConfig] = None,
) -> List[PWaveBeatAssessment]:
    """Run the auditable P-wave pipeline and return one contract per beat."""
    del beat_groups  # P morphology is clustered independently below.
    settings = config or PWaveConfig()
    if settings.mode not in {"offline", "online"}:
        raise ValueError("PWaveConfig.mode must be 'offline' or 'online'")
    values = np.asarray(ecg, dtype=float)
    r_values = np.asarray(r_locs, dtype=int)
    if values.ndim != 2 or values.shape[1] == 0 or r_values.size == 0:
        return []
    acquisition = dict(acquisition_qc or {})
    acquisition.setdefault("excluded_leads", [])
    acquisition.setdefault("reject_reasons", [])
    quiet_mask = _quiet_observation_mask(values, fs, r_values, beat_features)
    baseline, baseline_uncertainty = _state_space_baseline(
        values,
        quiet_mask,
        fs,
        mode=settings.mode,
    )
    corrected_precision = values - baseline
    corrected_detection = lowpass_filter(
        corrected_precision,
        fs,
        cutoff_hz=settings.detection_lowpass_hz,
        order=4,
    )
    excluded_leads = set(str(lead) for lead in acquisition["excluded_leads"])
    activity, _whitener, usable_leads = _whitened_spatial_activity(
        corrected_detection,
        quiet_mask,
        quality,
        excluded_leads,
        fs,
    )
    observation_distance = _distance_to_observation(quiet_mask)
    activity_thresholds = _activity_thresholds(activity, quiet_mask)
    by_beat = _features_by_beat(beat_features)
    feature_map = _features_by_beat_lead(beat_features)
    def assess_window(beat_id, items, coarse_start, coarse_stop, svd_dimension):
        qrs_on = _median_index(
            getattr(getattr(item, "qrs", None), "onset", None)
            for item in items
        )
        qrs_on = (
            int(qrs_on)
            if qrs_on is not None
            else int(r_values[beat_id]) - int(round(0.020 * fs))
        )
        previous_t = _previous_t_offset(beat_id, by_beat)
        on_t_overlap = bool(
            previous_t is not None
            and coarse_start - int(previous_t) <= int(round(0.030 * fs))
        )
        t_candidate, t_validated, t_meta = _constrained_t_reconstruction(
            corrected_precision,
            beat_id=beat_id,
            r_locs=r_values,
            qrs_on=qrs_on,
            coarse_start=coarse_start,
            coarse_stop=coarse_stop,
            feature_map=feature_map,
            fs=fs,
        )
        spatial_candidate = _t_spatial_subspace_suppression(
            corrected_detection,
            previous_r=int(r_values[beat_id - 1]) if beat_id > 0 else None,
            coarse_start=coarse_start,
            coarse_stop=coarse_stop,
            fs=fs,
        )
        per_lead: Dict[str, PWaveLeadBoundary] = {}
        for lead_index, lead in enumerate(STANDARD_12_LEADS):
            if lead_index >= values.shape[0]:
                continue
            feature = feature_map.get((beat_id, lead))
            branches = {
                "original_precision": corrected_precision[
                    lead_index, coarse_start : coarse_stop + 1
                ],
                "original_detection": corrected_detection[
                    lead_index, coarse_start : coarse_stop + 1
                ],
            }
            if spatial_candidate is not None:
                branches["t_spatial_subspace"] = spatial_candidate[lead_index]
            reconstruction_validated = bool(t_validated.get(lead, False))
            if t_candidate is not None and reconstruction_validated:
                branches["t_reconstruction"] = t_candidate[lead_index]
            ta_ambiguous = bool(
                getattr(feature, "p_ta_overlap_risk", False)
                or (
                    getattr(getattr(feature, "p", None), "offset", None) is not None
                    and qrs_on
                    - int(getattr(getattr(feature, "p", None), "offset"))
                    < int(round(0.030 * fs))
                )
            )
            per_lead[lead] = _lead_boundary(
                lead=lead,
                feature=feature,
                branches=branches,
                corrected_precision_lead=corrected_precision[lead_index],
                quiet_mask=quiet_mask,
                coarse_start=coarse_start,
                coarse_stop=coarse_stop,
                baseline_uncertainty=baseline_uncertainty[lead_index],
                observation_distance=observation_distance,
                quality=quality,
                excluded_leads=excluded_leads,
                fs=fs,
                config=settings,
                on_t_overlap=on_t_overlap,
                t_reconstruction_validated=reconstruction_validated,
                ta_ambiguous=ta_ambiguous,
            )
        return _fuse_beat(
            beat_id=beat_id,
            per_lead=per_lead,
            fs=fs,
            coarse_start=coarse_start,
            coarse_stop=coarse_stop,
            svd_dimension=svd_dimension,
            acquisition_qc=acquisition,
            t_reconstruction_available=bool(t_meta.get("available")),
            config=settings,
        )

    assessments: List[PWaveBeatAssessment] = []
    for beat_id in range(len(r_values)):
        items = by_beat.get(beat_id, [])
        coarse_start, coarse_stop, _fallback_dimension = _coarse_p_window(
            beat_id,
            r_values,
            items,
            activity,
            quiet_mask,
            fs,
            activity_thresholds=activity_thresholds,
        )
        svd_dimension = _svd_dimension(
            corrected_detection,
            coarse_start,
            coarse_stop,
            usable_leads,
        )
        assessment = assess_window(beat_id, items, coarse_start, coarse_stop, svd_dimension)
        if settings.multiple_candidates and (not assessment.accepted or min(
                assessment.onset_confidence, assessment.offset_confidence) < 0.55):
            alternatives = _alternate_p_windows(
                activity, r_values, beat_id, fs, activity_thresholds,
                (coarse_start, coarse_stop),
            )
            for alt_start, alt_stop in alternatives:
                alternate_dimension = _svd_dimension(
                    corrected_detection, alt_start, alt_stop, usable_leads,
                )
                candidate = assess_window(beat_id, items, alt_start, alt_stop, alternate_dimension)
                if _assessment_score(candidate) > _assessment_score(assessment) + 0.15:
                    assessment = candidate
                    assessment.global_detector_method += "+alternative_window"
        legacy_onset = _median_index(
            getattr(item, "p_onset_consensus_index", None)
            for item in items
        )
        legacy_offset = _median_index(
            getattr(item, "p_offset_consensus_index", None)
            for item in items
        )
        _reconcile_parallel_consensus_models(
            assessment,
            legacy_onset=legacy_onset,
            legacy_offset=legacy_offset,
            fs=fs,
            evidence_arbitration=settings.model_arbitration,
            legacy_confidence=float(np.median([
                min(float(getattr(item, "p_onset_confidence", None) or 0.0),
                    float(getattr(item, "p_offset_confidence", None) or 0.0))
                for item in items
            ])) if items else 0.0,
        )
        assessments.append(assessment)
        if settings.update_legacy_consensus:
            _update_legacy_consensus(assessment, items, fs)
    _assign_temporal_clusters(assessments, r_values, fs)
    return assessments


def finalize_p_wave_states(
    assessments: Sequence[PWaveBeatAssessment],
    af_afl_summary: Optional[Dict[str, object]],
) -> None:
    """Finalize the four-state atrial contract after rhythm evidence exists."""
    summary = af_afl_summary or {}
    probable_af = bool(summary.get("probable_af"))
    probable_flutter = bool(summary.get("probable_flutter"))
    indeterminate = bool(summary.get("af_afl_indeterminate"))
    for assessment in assessments:
        if probable_af:
            assessment.p_state = AF_LIKE
            assessment.accepted = False
            assessment.reject_reasons = sorted(
                set(assessment.reject_reasons + ["RHYTHM_AF_LIKE"])
            )
        elif probable_flutter:
            assessment.p_state = ORGANIZED_ATRIAL_ACTIVITY
            assessment.accepted = False
            assessment.reject_reasons = sorted(
                set(assessment.reject_reasons + ["RHYTHM_ORGANIZED_ATRIAL_ACTIVITY"])
            )
        elif indeterminate and not assessment.accepted:
            assessment.p_state = OVERLAP_UNCERTAIN
        elif assessment.accepted:
            assessment.p_state = P_PRESENT
        else:
            assessment.p_state = OVERLAP_UNCERTAIN


def backfill_missing_p_from_robust_engine(
    beat_features: Sequence[object],
    assessments: Sequence[PWaveBeatAssessment],
    ecg: np.ndarray,
    fs: int,
) -> None:
    """Fill P-wave gaps the single-lead delineator missed, using the robust
    multi-lead engine's per-lead boundary evidence.

    Additive only: never touches a lead-beat that already has a `p` result.
    Gated on `assessment.accepted` (excludes AF_LIKE / ORGANIZED_ATRIAL_ACTIVITY
    / OVERLAP_UNCERTAIN beats) and on the lead being part of the accepted
    fused cluster (`valid_leads`), so this cannot resurrect the T-wave/AF
    false-positive P class of bug: a beat only donates evidence here once the
    engine's own cross-lead corroboration gate has already passed.
    """
    values = np.asarray(ecg, dtype=float)
    for feature in beat_features:
        if getattr(getattr(feature, "p", None), "peak", None) is not None:
            continue
        beat_id = int(getattr(feature, "beat_id", -1))
        if beat_id < 0 or beat_id >= len(assessments):
            continue
        assessment = assessments[beat_id]
        if not assessment.accepted:
            continue
        lead = getattr(feature, "lead", None)
        if lead not in assessment.valid_leads:
            continue
        boundary = assessment.per_lead.get(lead)
        if boundary is None or boundary.onset is None or boundary.offset is None:
            continue
        onset, offset = int(boundary.onset), int(boundary.offset)
        if offset <= onset or lead not in STANDARD_12_LEADS:
            continue
        lead_index = STANDARD_12_LEADS.index(lead)
        if lead_index >= values.shape[0] or offset >= values.shape[1]:
            continue
        seg = values[lead_index, onset : offset + 1]
        if seg.size == 0:
            continue
        local_baseline = (float(values[lead_index, onset]) + float(values[lead_index, offset])) / 2.0
        peak = onset + int(np.argmax(np.abs(seg - local_baseline)))
        feature.p = WaveBounds(onset=onset, peak=peak, offset=offset)
        feature.flags.append("p_backfilled_from_robust_engine")


def summarize_p_wave_assessments(
    assessments: Sequence[PWaveBeatAssessment],
) -> Dict[str, object]:
    accepted = [assessment for assessment in assessments if assessment.accepted]
    reasons: Dict[str, int] = {}
    for assessment in assessments:
        for reason in assessment.reject_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    state_counts: Dict[str, int] = {}
    for assessment in assessments:
        state_counts[assessment.p_state] = state_counts.get(assessment.p_state, 0) + 1
    return {
        "contract_version": "p_wave_boundary.v1",
        "n_beats": len(assessments),
        "n_accepted": len(accepted),
        "accepted_fraction": (
            float(len(accepted) / len(assessments))
            if assessments
            else 0.0
        ),
        "state_counts": state_counts,
        "reject_reason_counts": reasons,
        "baseline_methods": sorted(
            {assessment.baseline_mode for assessment in assessments}
        ),
        "strict_and_robust_boundaries": True,
        "confidence_intervals": "empirical_perturbation_uncalibrated",
        "independent_lead_group_fusion": True,
    }
