from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import savgol_filter

from .models import LeadBeatFeatures, LeadQuality, STANDARD_12_LEADS, WaveBounds
from .numeric import trapezoid
from .preprocess import lowpass_filter


_INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
_LEAD_PRIOR = {
    "V2": 1.00,
    "V3": 1.00,
    "V4": 0.95,
    "V5": 0.95,
    "II": 0.90,
    "V6": 0.85,
    "I": 0.80,
    "aVF": 0.75,
    "aVR": 0.70,
    "V1": 0.60,
    "III": 0.55,
    "aVL": 0.55,
}


@dataclass
class _LeadTCandidates:
    feature: LeadBeatFeatures
    signal: np.ndarray
    filtered: np.ndarray
    baseline: float
    amplitude_mv: float
    snr_db: float
    onset_candidates: dict[str, int]
    offset_candidates: dict[str, int]
    selected_onset: int | None
    selected_offset: int | None
    selected_method: str
    suspect: bool


@dataclass(frozen=True)
class TFusionResult:
    center_index: int | None
    latest_p85_index: int | None
    ci_low_index: int | None
    ci_high_index: int | None
    support: int
    mad_ms: float | None
    used_sources: tuple[str, ...]
    excluded_sources: tuple[str, ...]
    reliable: bool
    cluster_count: int = 0
    selected_cluster_score: float | None = None
    selected_cluster_support: int = 0
    selected_cluster_lead_groups: tuple[str, ...] = ()
    selected_cluster_methods: tuple[str, ...] = ()
    reliability_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TFusionMorphologyAssessment:
    t_peak_consensus_index: int | None
    tpte_ms: float | None
    derived_disagreement_ms: float | None
    tail_incomplete_leads: tuple[str, ...]
    systematic_early_risk: bool
    reliable: bool
    reasons: tuple[str, ...]


def _finite_index(value: object, n: int) -> int | None:
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return index if 0 <= index < n else None


def _mad_sigma(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    center = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - center)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    midpoint = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), midpoint, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _source_group(source: str) -> str:
    if source in {"I", "II", "III", "aVR", "aVL", "aVF"}:
        return "limb"
    if source.startswith("V") and source[1:].isdigit():
        return "precordial"
    if source.upper() in {"RMS", "PC1"}:
        return "derived"
    return "other"


def _method_family(method: str, source: str) -> str:
    text = str(method or "").lower()
    if source.upper() in {"RMS", "PC1"}:
        return "derived"
    for family in ("mallat", "trapezium", "tangent", "slope", "chord"):
        if family in text:
            return family
    return "lead_local"


def _temporal_clusters(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    radius_samples: float,
) -> list[np.ndarray]:
    """Form compact one-dimensional clusters without allowing chain bridging."""

    order = np.argsort(values)
    clusters: list[list[int]] = []
    for raw_index in order:
        index = int(raw_index)
        if not clusters:
            clusters.append([index])
            continue
        current = clusters[-1]
        current_values = values[current]
        current_weights = weights[current]
        center = _weighted_median(current_values, current_weights)
        proposed_span = max(
            float(np.max(current_values)) - float(np.min(current_values)),
            abs(float(values[index]) - center),
        )
        if (
            abs(float(values[index]) - center) <= radius_samples
            and proposed_span <= 2.0 * radius_samples
        ):
            current.append(index)
        else:
            clusters.append([index])
    return [np.asarray(cluster, dtype=int) for cluster in clusters]


def _cluster_score(
    members: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    names: Sequence[str],
    method_names: Sequence[str],
    group_names: Sequence[str],
    *,
    fs: int,
) -> float:
    member_weights = weights[members]
    weight_fraction = float(np.sum(member_weights) / max(np.sum(weights), 1e-9))
    support_score = min(float(len(members)) / 4.0, 1.0)
    lead_groups = {
        group_names[int(index)]
        for index in members
        if group_names[int(index)] in {"limb", "precordial"}
    }
    group_score = 1.0 if lead_groups == {"limb", "precordial"} else (
        0.45 if lead_groups else 0.0
    )
    methods = {method_names[int(index)] for index in members}
    method_score = min(float(len(methods)) / 3.0, 1.0)
    derived_score = min(
        sum(group_names[int(index)] == "derived" for index in members) / 2.0,
        1.0,
    )
    member_values = values[members]
    span_ms = float(np.ptp(member_values) * 1000.0 / fs) if len(members) > 1 else 0.0
    compactness = float(np.clip(1.0 - span_ms / 50.0, 0.0, 1.0))
    # Support and total SQI weight dominate.  Lead-system, algorithm and
    # derived-vector diversity prevent a high-weight isolated endpoint from
    # winning merely because it comes from a visually prominent lead.
    return float(
        0.30 * support_score
        + 0.30 * weight_fraction
        + 0.15 * group_score
        + 0.10 * method_score
        + 0.10 * derived_score
        + 0.05 * compactness
    )


def robust_t_offset_fusion(
    indices: Sequence[int],
    weights: Sequence[float],
    sources: Sequence[str],
    *,
    fs: int,
    methods: Sequence[str] | None = None,
    lead_groups: Sequence[str] | None = None,
    cluster_radius_ms: float = 25.0,
) -> TFusionResult:
    """Temporal clustering → cluster scoring → MAD/Huber T-offset fusion."""

    method_values = list(methods) if methods is not None else [""] * len(indices)
    group_values = list(lead_groups) if lead_groups is not None else [""] * len(indices)
    rows = [
        (
            int(index),
            float(weight),
            str(source),
            _method_family(method, str(source)),
            str(group) if str(group) else _source_group(str(source)),
        )
        for index, weight, source, method, group in zip(
            indices,
            weights,
            sources,
            method_values,
            group_values,
        )
        if np.isfinite(index) and np.isfinite(weight) and float(weight) > 0.0
    ]
    if fs <= 0 or len(rows) < 2:
        return TFusionResult(
            center_index=None,
            latest_p85_index=None,
            ci_low_index=None,
            ci_high_index=None,
            support=0,
            mad_ms=None,
            used_sources=(),
            excluded_sources=tuple(str(s) for s in sources),
            reliable=False,
            reliability_reasons=("insufficient_candidates",),
        )

    values = np.asarray([item[0] for item in rows], dtype=float)
    raw_weights = np.asarray([item[1] for item in rows], dtype=float)
    names = [item[2] for item in rows]
    method_names = [item[3] for item in rows]
    group_names = [item[4] for item in rows]

    radius_samples = max(1.0, float(cluster_radius_ms) * float(fs) / 1000.0)
    clusters = _temporal_clusters(
        values,
        raw_weights,
        radius_samples=radius_samples,
    )
    cluster_scores = [
        _cluster_score(
            members,
            values,
            raw_weights,
            names,
            method_names,
            group_names,
            fs=fs,
        )
        for members in clusters
    ]
    # A later cluster does not receive an intrinsic timing bonus.  It wins only
    # through independent lead/method support, avoiding systematic U/P capture.
    selected_cluster_index = max(
        range(len(clusters)),
        key=lambda index: (
            cluster_scores[index],
            len(clusters[index]),
            float(np.sum(raw_weights[clusters[index]])),
        ),
    )
    selected_members = clusters[selected_cluster_index]
    cluster_mask = np.zeros(len(values), dtype=bool)
    cluster_mask[selected_members] = True

    selected_values = values[selected_members]
    selected_weights = raw_weights[selected_members]
    initial = _weighted_median(selected_values, selected_weights)
    residual_ms = (selected_values - initial) * 1000.0 / float(fs)
    mad_ms = max(_mad_sigma(residual_ms), 3.0)
    inlier_limit_ms = max(3.0 * mad_ms, 15.0)
    selected_inlier_mask = np.abs(residual_ms) <= inlier_limit_ms
    if int(np.sum(selected_inlier_mask)) < 2:
        closest = np.argsort(np.abs(residual_ms))[:2]
        selected_inlier_mask = np.zeros_like(selected_inlier_mask, dtype=bool)
        selected_inlier_mask[closest] = True

    inlier_mask = np.zeros(len(values), dtype=bool)
    inlier_mask[selected_members[selected_inlier_mask]] = True

    inlier_values = values[inlier_mask]
    inlier_weights = raw_weights[inlier_mask]
    center = _weighted_median(inlier_values, inlier_weights)
    huber_c_samples = max(1e-6, 1.345 * mad_ms * float(fs) / 1000.0)
    for _ in range(5):
        residual = inlier_values - center
        huber = np.ones_like(residual)
        large = np.abs(residual) > huber_c_samples
        huber[large] = huber_c_samples / np.abs(residual[large])
        effective = inlier_weights * huber
        updated = float(np.sum(effective * inlier_values) / np.sum(effective))
        if abs(updated - center) * 1000.0 / float(fs) < 0.5:
            center = updated
            break
        center = updated

    support = int(np.sum(inlier_mask))
    latest = float(np.percentile(inlier_values, 85))
    standard_error_ms = mad_ms / np.sqrt(max(float(np.sum(inlier_weights)), 1e-6))
    half_width_samples = 1.96 * standard_error_ms * float(fs) / 1000.0
    used = tuple(name for name, keep in zip(names, inlier_mask) if keep)
    excluded = tuple(name for name, keep in zip(names, inlier_mask) if not keep)
    used_groups = tuple(
        sorted(
            {
                group
                for group, keep in zip(group_names, inlier_mask)
                if keep and group in {"limb", "precordial", "derived"}
            }
        )
    )
    used_methods = tuple(
        sorted({method for method, keep in zip(method_names, inlier_mask) if keep})
    )
    standard_groups = {group for group in used_groups if group != "derived"}
    reasons: list[str] = []
    if support < 4:
        reasons.append("support_below_4")
    if mad_ms > 20.0:
        reasons.append("dispersion_above_20ms")
    if standard_groups != {"limb", "precordial"}:
        reasons.append("missing_limb_or_precordial_support")
    reliable = not reasons
    return TFusionResult(
        center_index=int(round(center)),
        latest_p85_index=int(round(latest)),
        ci_low_index=int(round(center - half_width_samples)),
        ci_high_index=int(round(center + half_width_samples)),
        support=support,
        mad_ms=float(mad_ms),
        used_sources=used,
        excluded_sources=excluded,
        reliable=reliable,
        cluster_count=len(clusters),
        selected_cluster_score=float(cluster_scores[selected_cluster_index]),
        selected_cluster_support=len(selected_members),
        selected_cluster_lead_groups=used_groups,
        selected_cluster_methods=used_methods,
        reliability_reasons=tuple(reasons),
    )


def _tail_remains_after_center(
    candidate: _LeadTCandidates,
    *,
    center_index: int,
    fs: int,
) -> bool:
    peak = candidate.feature.t.peak
    if peak is None or center_index <= int(peak):
        return False
    start = center_index + max(1, int(round(0.006 * fs)))
    end = min(len(candidate.filtered), center_index + int(round(0.070 * fs)))
    if end - start < max(3, int(round(0.020 * fs))):
        return False
    residual = np.abs(
        np.asarray(candidate.filtered[start:end], dtype=float) - candidate.baseline
    )
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return False
    amplitude = max(float(candidate.amplitude_mv), 1e-6)
    # Both a peak and a sustained-residual test are required.  The latter
    # avoids treating one noisy sample or the beginning of a U wave as an
    # unfinished T tail.
    return bool(
        float(np.max(residual)) >= max(0.020, 0.12 * amplitude)
        and float(np.median(residual)) >= max(0.010, 0.05 * amplitude)
    )


def _assess_t_fusion_morphology(
    fusion: TFusionResult,
    candidates: Sequence[_LeadTCandidates],
    derived: Mapping[str, tuple[int | None, int | None, float]],
    *,
    fs: int,
) -> TFusionMorphologyAssessment:
    if fusion.center_index is None or fs <= 0:
        return TFusionMorphologyAssessment(
            t_peak_consensus_index=None,
            tpte_ms=None,
            derived_disagreement_ms=None,
            tail_incomplete_leads=(),
            systematic_early_risk=False,
            reliable=False,
            reasons=("fusion_center_unavailable",),
        )

    peak_rows = [
        (
            int(candidate.feature.t.peak),
            max(float(candidate.feature.t_fusion_weight or 0.0), 1e-3),
        )
        for candidate in candidates
        if candidate.feature.t.peak is not None
        and (
            candidate.feature.t_sqi_pass
            or float(candidate.feature.t_sqi_score or 0.0) >= 0.35
        )
    ]
    peak_center = (
        int(
            round(
                _weighted_median(
                    np.asarray([row[0] for row in peak_rows], dtype=float),
                    np.asarray([row[1] for row in peak_rows], dtype=float),
                )
            )
        )
        if len(peak_rows) >= 3
        else None
    )
    tpte_ms = (
        float((fusion.center_index - peak_center) * 1000.0 / fs)
        if peak_center is not None
        else None
    )

    derived_gaps_ms = [
        float((offset - fusion.center_index) * 1000.0 / fs)
        for _name, (_onset, offset, _weight) in derived.items()
        if offset is not None
    ]
    derived_disagreement_ms = (
        float(max(abs(value) for value in derived_gaps_ms))
        if derived_gaps_ms
        else None
    )
    independently_late_derived = [
        value for value in derived_gaps_ms if value >= 25.0
    ]

    tail_eligible = [
        candidate
        for candidate in candidates
        if candidate.feature.t_sqi_pass
        or float(candidate.feature.t_sqi_score or 0.0) >= 0.35
    ]
    tail_incomplete = tuple(
        candidate.feature.lead
        for candidate in tail_eligible
        if _tail_remains_after_center(
            candidate,
            center_index=fusion.center_index,
            fs=fs,
        )
    )
    tail_incomplete_consensus = bool(
        len(tail_incomplete) >= 3
        and len(tail_incomplete) / max(len(tail_eligible), 1) >= 0.30
    )
    short_tpte = bool(tpte_ms is not None and tpte_ms < 45.0)
    derived_late_consensus = len(independently_late_derived) >= 2

    reasons = list(fusion.reliability_reasons)
    if short_tpte:
        reasons.append("tpte_below_45ms")
    if derived_late_consensus:
        reasons.append("rms_pc1_later_than_fusion")
    if tail_incomplete_consensus:
        reasons.append("multilead_residual_t_tail")
    reasons = list(dict.fromkeys(reasons))
    # None of these morphology cues is specific enough to veto a QT alone:
    # broad normal T waves often retain a small residual after a clinically
    # accepted tangent endpoint, and one derived vector can latch onto U/P.
    # A systemic-early veto therefore requires residual tail evidence plus an
    # independent timing contradiction.
    systematic_early_risk = bool(
        tail_incomplete_consensus
        and (short_tpte or derived_late_consensus)
    )
    return TFusionMorphologyAssessment(
        t_peak_consensus_index=peak_center,
        tpte_ms=tpte_ms,
        derived_disagreement_ms=derived_disagreement_ms,
        tail_incomplete_leads=tail_incomplete,
        systematic_early_risk=systematic_early_risk,
        reliable=bool(fusion.reliable and not systematic_early_risk),
        reasons=tuple(reasons),
    )


def _baseline(signal: np.ndarray, qrs_onset: int | None, r_index: int, fs: int) -> float:
    if qrs_onset is not None:
        lo = max(0, int(qrs_onset) - int(round(0.070 * fs)))
        hi = max(lo + 1, int(qrs_onset) - int(round(0.015 * fs)))
    else:
        lo = max(0, int(r_index) - int(round(0.220 * fs)))
        hi = max(lo + 1, int(r_index) - int(round(0.080 * fs)))
    segment = np.asarray(signal[lo:hi], dtype=float)
    segment = segment[np.isfinite(segment)]
    return float(np.median(segment)) if segment.size else 0.0


def _dilated_kernel(coefficients: Sequence[float], step: int) -> np.ndarray:
    kernel = np.zeros((len(coefficients) - 1) * step + 1, dtype=float)
    kernel[::step] = np.asarray(coefficients, dtype=float)
    return kernel


def _mallat_detail(signal: np.ndarray, fs: int) -> np.ndarray:
    """Quadratic-spline à trous detail at the T-wave scale."""

    level = int(np.clip(round(np.log2(max(float(fs), 1.0) / 31.25)), 2, 6))
    smooth = np.asarray(signal, dtype=float)
    h = (1.0, 3.0, 3.0, 1.0)
    for current_level in range(1, level + 1):
        step = 2 ** (current_level - 1)
        smooth = np.convolve(
            smooth,
            _dilated_kernel([value / 8.0 for value in h], step),
            mode="same",
        )
    derivative_step = 2 ** max(0, level - 1)
    return np.convolve(
        smooth,
        _dilated_kernel((2.0, -2.0), derivative_step),
        mode="same",
    )


def mallat_t_boundaries(
    signal: np.ndarray,
    *,
    fs: int,
    peak: int,
    search_start: int,
    search_end: int,
    low_amplitude: bool = False,
) -> tuple[int | None, int | None]:
    """Return deterministic Mallat-style T onset/offset candidates."""

    values = np.asarray(signal, dtype=float)
    n = len(values)
    peak_i = _finite_index(peak, n)
    lo = max(0, int(search_start))
    hi = min(n - 1, int(search_end))
    if peak_i is None or fs <= 0 or not (lo + 3 < peak_i < hi - 3):
        return None, None

    detail = np.abs(_mallat_detail(values, fs))
    window = detail[lo : hi + 1]
    if window.size < 8:
        return None, None
    epsilon = max(
        0.25 * float(np.sqrt(np.mean(np.square(window)))),
        0.125 * float(np.max(window)),
    )

    left = detail[lo : peak_i + 1]
    right_limit = min(hi, peak_i + int(round(0.200 * fs)))
    right = detail[peak_i : right_limit + 1]
    if left.size < 3 or right.size < 3:
        return None, None
    left_max = float(np.max(left))
    right_max = float(np.max(right))
    if left_max <= epsilon or right_max <= epsilon:
        return None, None

    left_extremum = lo + int(np.argmax(left))
    right_extremum = peak_i + int(np.argmax(right))
    onset_threshold = 0.25 * left_max
    offset_threshold = (0.50 if low_amplitude else 0.40) * right_max

    onset = left_extremum
    while onset > lo and detail[onset] > onset_threshold:
        onset -= 1
    offset = right_extremum
    max_offset = min(hi, right_extremum + int(round(0.160 * fs)))
    while offset < max_offset and detail[offset] > offset_threshold:
        offset += 1
    if offset >= max_offset and detail[offset] > offset_threshold:
        local = detail[right_extremum : max_offset + 1]
        if local.size:
            offset = right_extremum + int(np.argmin(local))
    return int(onset), int(offset)


def trapezium_t_offset(
    signal: np.ndarray,
    *,
    fs: int,
    peak: int,
    search_end: int,
) -> int | None:
    """Low-amplitude T-end candidate using the trapezium-area construction."""

    values = np.asarray(signal, dtype=float)
    n = len(values)
    peak_i = _finite_index(peak, n)
    hi = min(n - 1, int(search_end))
    if peak_i is None or fs <= 0 or hi - peak_i < max(8, int(round(0.060 * fs))):
        return None

    derivative_hi = min(hi, peak_i + int(round(0.200 * fs)))
    segment = values[peak_i : derivative_hi + 1]
    window = min(9, len(segment) if len(segment) % 2 else len(segment) - 1)
    if window < 5:
        return None
    derivative = savgol_filter(
        segment,
        window_length=window,
        polyorder=2,
        deriv=1,
        delta=1.0 / float(fs),
        mode="interp",
    )
    slope_exclusion = min(len(derivative) - 1, max(1, int(round(0.010 * fs))))
    x_m = peak_i + slope_exclusion + int(np.argmax(np.abs(derivative[slope_exclusion:])))
    x_r = min(hi, x_m + int(round(0.160 * fs)))
    candidate_lo = x_m + max(1, int(round(0.010 * fs)))
    candidate_hi = x_r - max(1, int(round(0.010 * fs)))
    if candidate_hi <= candidate_lo:
        return None
    positions = np.arange(candidate_lo, candidate_hi + 1, dtype=int)
    areas = 0.5 * np.abs(values[x_m] - values[positions]) * (
        2.0 * float(x_r) - positions.astype(float) - float(x_m)
    )
    return int(positions[int(np.argmax(areas))])


def _rr_samples(r_locs: np.ndarray, beat_id: int, fs: int) -> int:
    if len(r_locs) >= 2:
        lo = max(0, int(beat_id) - 5)
        prior = np.diff(r_locs[lo : int(beat_id) + 1])
        if prior.size:
            return max(1, int(round(float(np.median(prior)))))
        if int(beat_id) + 1 < len(r_locs):
            return max(1, int(r_locs[int(beat_id) + 1] - r_locs[int(beat_id)]))
    return max(1, int(fs))


def _search_window(
    feature: LeadBeatFeatures,
    r_locs: np.ndarray,
    fs: int,
    n: int,
) -> tuple[int, int]:
    beat_id = int(feature.beat_id)
    r_index = int(r_locs[beat_id])
    rr = _rr_samples(r_locs, beat_id, fs)
    qrs_offset = feature.qrs.offset if feature.qrs.offset is not None else r_index
    start = max(
        int(qrs_offset) + max(1, int(round(0.010 * fs))),
        r_index + int(round(0.040 * fs)),
    )
    end = r_index + min(int(round(0.62 * rr)), int(round(0.500 * fs)))
    if beat_id + 1 < len(r_locs):
        end = min(end, int(r_locs[beat_id + 1]) - int(round(0.120 * fs)))
    return max(0, start), min(n - 1, max(start + 1, end))


def _local_snr_db(
    raw: np.ndarray,
    filtered: np.ndarray,
    *,
    baseline: float,
    peak: int,
    lo: int,
    hi: int,
) -> tuple[float, float]:
    amplitude = abs(float(filtered[peak] - baseline))
    residual = np.asarray(raw[lo : hi + 1] - filtered[lo : hi + 1], dtype=float)
    noise = max(_mad_sigma(residual), 0.002)
    return amplitude, float(20.0 * np.log10(max(amplitude, 1e-6) / noise))


def _select_boundaries(
    feature: LeadBeatFeatures,
    *,
    amplitude_mv: float,
    snr_db: float,
    onset_candidates: Mapping[str, int],
    offset_candidates: Mapping[str, int],
    fs: int,
) -> tuple[int | None, int | None, str, bool]:
    native_onset = onset_candidates.get("chord")
    wavelet_onset = onset_candidates.get("mallat")
    native_offset = offset_candidates.get("chord")
    wavelet_offset = offset_candidates.get("mallat")
    trapezium_offset = offset_candidates.get("trapezium")
    suspect = bool(
        feature.st_t_confusion
        or "fallback" in str(feature.t_end_method or "")
        or feature.qt_confidence < 0.25
    )

    selected_onset = native_onset
    if wavelet_onset is not None and snr_db >= 6.0:
        if native_onset is None or abs(wavelet_onset - native_onset) <= int(round(0.080 * fs)):
            selected_onset = wavelet_onset

    selected_offset = native_offset
    method = "chord"
    if amplitude_mv < 0.080 and trapezium_offset is not None:
        selected_offset = trapezium_offset
        method = "trapezium_low_amplitude"
    elif amplitude_mv < 0.200:
        low_candidates = [
            value for value in (wavelet_offset, trapezium_offset) if value is not None
        ]
        if low_candidates:
            selected_offset = int(round(float(np.median(low_candidates))))
            method = "mallat_trapezium_low_amplitude"
    elif wavelet_offset is not None:
        if native_offset is None:
            selected_offset = wavelet_offset
            method = "mallat"
        elif abs(wavelet_offset - native_offset) <= int(round(0.040 * fs)):
            selected_offset = int(round(0.5 * (wavelet_offset + native_offset)))
            method = "chord_mallat_consensus"
        elif suspect and wavelet_offset > native_offset:
            selected_offset = wavelet_offset
            method = "mallat_late_tail_rescue"
    return selected_onset, selected_offset, method, suspect


def _update_t_measurements(
    candidate: _LeadTCandidates,
    *,
    fs: int,
    rescue: bool = False,
) -> None:
    feature = candidate.feature
    onset = candidate.selected_onset
    peak = feature.t.peak
    offset = candidate.selected_offset
    if peak is None or onset is None or offset is None or not (onset < peak < offset):
        return
    feature.t = WaveBounds(onset=int(onset), peak=int(peak), offset=int(offset))
    feature.t_end_method = candidate.selected_method
    if rescue:
        feature.flags.append("t_end_late_tail_rescue")
        feature.t_end_repair_reason = "cross_lead_late_tail_support"
    if "t_wave_refined" not in feature.flags:
        feature.flags.append("t_wave_refined")
    if feature.qrs.onset is not None:
        feature.qt_ms = float((offset - feature.qrs.onset) * 1000.0 / fs)
    if feature.qrs.offset is not None:
        feature.jt_ms = float((offset - feature.qrs.offset) * 1000.0 / fs)
    feature.tpe_ms = float((offset - peak) * 1000.0 / fs)
    feature.t_dur_ms = float((offset - onset) * 1000.0 / fs)
    segment = candidate.signal[onset : offset + 1] - candidate.baseline
    if segment.size >= 2:
        feature.t_area = float(trapezoid(np.abs(segment), dx=1000.0 / fs))
        feature.t_signed_area = float(trapezoid(segment, dx=1000.0 / fs))


def _update_t_onset_only(candidate: _LeadTCandidates, *, fs: int) -> None:
    """Adopt a high-confidence Mallat onset without changing local T offset."""

    feature = candidate.feature
    onset = candidate.selected_onset
    peak = feature.t.peak
    offset = feature.t.offset
    native_onset = candidate.onset_candidates.get("chord")
    if (
        onset is None
        or peak is None
        or offset is None
        or not (onset < peak < offset)
        or candidate.snr_db < 6.0
        # Below 120 µV the onset wavelet coefficient is too sensitive to noise
        # for an authoritative replacement; keep it as an audited candidate.
        or candidate.amplitude_mv < 0.120
        or (
            native_onset is not None
            and abs(onset - native_onset) > int(round(0.060 * fs))
        )
    ):
        return
    feature.t = WaveBounds(onset=int(onset), peak=int(peak), offset=int(offset))
    feature.t_dur_ms = float((offset - onset) * 1000.0 / fs)
    segment = candidate.signal[onset : offset + 1] - candidate.baseline
    if segment.size >= 2:
        feature.t_area = float(trapezoid(np.abs(segment), dx=1000.0 / fs))
        feature.t_signed_area = float(trapezoid(segment, dx=1000.0 / fs))
    if onset != native_onset and "t_onset_mallat_refined" not in feature.flags:
        feature.flags.append("t_onset_mallat_refined")


def _derived_candidates(
    ecg_t: np.ndarray,
    *,
    features: Sequence[LeadBeatFeatures],
    r_locs: np.ndarray,
    beat_id: int,
    fs: int,
    quality: Mapping[str, LeadQuality] | None,
) -> dict[str, tuple[int | None, int | None, float]]:
    by_lead = {feature.lead: feature for feature in features}
    lead_indices = {lead: index for index, lead in enumerate(STANDARD_12_LEADS)}
    usable = [
        lead
        for lead in _INDEPENDENT_LEADS
        if lead in by_lead
        and lead_indices[lead] < ecg_t.shape[0]
        and (
            quality is None
            or bool(getattr(quality.get(lead), "reliable_for_t", False))
        )
    ]
    if len(usable) < 4:
        return {}

    reference = by_lead[usable[0]]
    lo, hi = _search_window(reference, r_locs, fs, ecg_t.shape[1])
    r_index = int(r_locs[beat_id])
    qrs_offsets = [
        int(by_lead[lead].qrs.offset)
        for lead in usable
        if by_lead[lead].qrs.offset is not None
    ]
    peak_lo = max(
        lo,
        int(np.median(qrs_offsets)) + max(
            int(round(0.060 * fs)),
            int(round(0.04 * _rr_samples(r_locs, beat_id, fs))),
        )
        if qrs_offsets
        else r_index + int(round(0.080 * fs)),
    )
    if hi - peak_lo < max(5, int(round(0.060 * fs))):
        return {}

    rows = []
    for lead in usable:
        index = lead_indices[lead]
        feature = by_lead[lead]
        baseline = _baseline(
            ecg_t[index],
            feature.qrs.onset,
            r_index,
            fs,
        )
        rows.append(np.asarray(ecg_t[index, lo : hi + 1] - baseline, dtype=float))
    matrix = np.vstack(rows)
    rms = np.sqrt(np.mean(np.square(matrix), axis=0))
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        pc1 = singular_values[0] * vh[0]
    except np.linalg.LinAlgError:
        pc1 = np.mean(centered, axis=0)

    results: dict[str, tuple[int | None, int | None, float]] = {}
    for name, derived, prior in (("RMS", rms, 1.0), ("PC1", pc1, 1.0)):
        local_peak_lo = max(0, peak_lo - lo)
        peak = local_peak_lo + int(np.argmax(np.abs(derived[local_peak_lo:])))
        onset, offset = mallat_t_boundaries(
            derived,
            fs=fs,
            peak=peak,
            search_start=0,
            search_end=len(derived) - 1,
            low_amplitude=False,
        )
        if onset is not None:
            onset += lo
        if offset is not None:
            offset += lo
        energy = float(np.sqrt(np.mean(np.square(derived)))) if derived.size else 0.0
        weight = prior * float(np.clip(energy / 0.20, 0.20, 1.0))
        results[name] = (onset, offset, weight)
    return results


def _t_sqi(
    candidate: _LeadTCandidates,
    *,
    stability_ms: float,
    quality: Mapping[str, LeadQuality] | None,
) -> tuple[float, bool, float]:
    spread_values = list(candidate.offset_candidates.values())
    spread_ms = (
        float(np.ptp(spread_values) * 1000.0 / 500.0)
        if len(spread_values) >= 2
        else 10.0
    )
    # The caller replaces the 500 Hz normalisation below with the stored,
    # sampling-rate-correct spread before invoking this helper.
    spread_ms = float(candidate.feature.t_candidate_spread_ms or spread_ms)
    amp_score = float(np.clip(candidate.amplitude_mv / 0.200, 0.0, 1.0))
    snr_score = float(np.clip((candidate.snr_db - 3.0) / 12.0, 0.0, 1.0))
    consistency = float(min(1.0, 10.0 / max(spread_ms, 10.0)))
    stability = float(min(1.0, 10.0 / max(stability_ms, 10.0)))
    u_score = 0.70 if candidate.feature.u_wave_flag else 1.0
    lead_quality = quality.get(candidate.feature.lead) if quality is not None else None
    general_pass = (
        lead_quality is None
        or bool(getattr(lead_quality, "reliable_for_t", False))
    )
    hard_pass = bool(
        general_pass
        and candidate.amplitude_mv >= 0.050
        and candidate.snr_db >= 6.0
        and candidate.selected_offset is not None
    )
    score = float(
        np.clip(
            (amp_score * max(snr_score, 0.05) * consistency * stability * u_score)
            ** 0.2,
            0.0,
            1.0,
        )
    )
    weight = _LEAD_PRIOR.get(candidate.feature.lead, 0.60) * score
    if not hard_pass:
        weight *= 0.20
    return score, hard_pass, float(weight)


def refine_t_wave_boundaries(
    beat_features: Sequence[LeadBeatFeatures],
    *,
    measurement_ecg: np.ndarray,
    r_locs: np.ndarray,
    fs: int,
    quality: Mapping[str, LeadQuality] | None = None,
) -> None:
    """Apply T-specific candidates, SQI, derived leads and robust fusion in place."""

    ecg = np.asarray(measurement_ecg, dtype=float)
    r_values = np.asarray(r_locs, dtype=int)
    if ecg.ndim != 2 or fs <= 0 or r_values.size == 0:
        return
    cutoff = min(22.5, 0.42 * float(fs))
    if cutoff <= 0.0:
        return
    ecg_t = lowpass_filter(ecg, fs, cutoff_hz=cutoff, order=4)
    lead_indices = {
        lead: index
        for index, lead in enumerate(STANDARD_12_LEADS)
        if index < ecg.shape[0]
    }
    by_beat: dict[int, list[LeadBeatFeatures]] = {}
    for feature in beat_features:
        beat_id = int(feature.beat_id)
        if beat_id < len(r_values):
            by_beat.setdefault(beat_id, []).append(feature)

    candidates: dict[tuple[int, str], _LeadTCandidates] = {}
    for beat_id, items in by_beat.items():
        r_index = int(r_values[beat_id])
        for feature in items:
            lead_index = lead_indices.get(feature.lead)
            peak = _finite_index(feature.t.peak, ecg.shape[1])
            if lead_index is None or peak is None:
                continue
            raw = ecg[lead_index]
            filtered = ecg_t[lead_index]
            baseline = _baseline(raw, feature.qrs.onset, r_index, fs)
            lo, hi = _search_window(feature, r_values, fs, ecg.shape[1])
            if not (lo < peak < hi):
                continue
            amplitude, snr_db = _local_snr_db(
                raw,
                filtered,
                baseline=baseline,
                peak=peak,
                lo=lo,
                hi=hi,
            )
            wave_on, wave_off = mallat_t_boundaries(
                filtered,
                fs=fs,
                peak=peak,
                search_start=lo,
                search_end=hi,
                low_amplitude=amplitude < 0.200,
            )
            trap_off = trapezium_t_offset(
                filtered,
                fs=fs,
                peak=peak,
                search_end=hi,
            )
            onset_candidates = {
                name: int(value)
                for name, value in (
                    ("chord", feature.t.onset),
                    ("mallat", wave_on),
                )
                if value is not None and lo <= int(value) < peak
            }
            offset_candidates = {
                name: int(value)
                for name, value in (
                    ("chord", feature.t.offset),
                    ("mallat", wave_off),
                    ("trapezium", trap_off),
                )
                if value is not None and peak < int(value) <= hi
            }
            selected_on, selected_off, method, suspect = _select_boundaries(
                feature,
                amplitude_mv=amplitude,
                snr_db=snr_db,
                onset_candidates=onset_candidates,
                offset_candidates=offset_candidates,
                fs=fs,
            )
            candidate = _LeadTCandidates(
                feature=feature,
                signal=raw,
                filtered=filtered,
                baseline=baseline,
                amplitude_mv=amplitude,
                snr_db=snr_db,
                onset_candidates=onset_candidates,
                offset_candidates=offset_candidates,
                selected_onset=selected_on,
                selected_offset=selected_off,
                selected_method=method,
                suspect=suspect,
            )
            candidates[(beat_id, feature.lead)] = candidate
            feature.t_wavelet_onset_index = wave_on
            feature.t_wavelet_offset_index = wave_off
            feature.t_trapezium_offset_index = trap_off
            feature.t_local_snr_db = snr_db
            feature.t_candidate_spread_ms = (
                float(np.ptp(list(offset_candidates.values())) * 1000.0 / fs)
                if len(offset_candidates) >= 2
                else 0.0
            )
            _update_t_onset_only(candidate, fs=fs)

    # Cross-beat stability is measured on lead-local offset relative to the
    # shared R fiducial, so rate changes do not masquerade as boundary jitter.
    for lead in STANDARD_12_LEADS:
        lead_candidates = [
            candidate
            for (beat_id, candidate_lead), candidate in candidates.items()
            if candidate_lead == lead and candidate.selected_offset is not None
        ]
        offsets_ms = [
            (float(candidate.selected_offset) - float(r_values[candidate.feature.beat_id]))
            * 1000.0
            / fs
            for candidate in lead_candidates
        ]
        stability_ms = _mad_sigma(offsets_ms) if len(offsets_ms) >= 2 else 10.0
        for candidate in lead_candidates:
            score, hard_pass, weight = _t_sqi(
                candidate,
                stability_ms=stability_ms,
                quality=quality,
            )
            feature = candidate.feature
            feature.t_boundary_stability_ms = float(stability_ms)
            feature.t_sqi_score = score
            feature.t_sqi_pass = hard_pass
            feature.t_fusion_weight = weight

    for beat_id, items in by_beat.items():
        beat_candidates = [
            candidates[(beat_id, feature.lead)]
            for feature in items
            if (beat_id, feature.lead) in candidates
        ]
        derived = _derived_candidates(
            ecg_t,
            features=items,
            r_locs=r_values,
            beat_id=beat_id,
            fs=fs,
            quality=quality,
        )

        def fuse() -> TFusionResult:
            hard = [
                candidate
                for candidate in beat_candidates
                if candidate.selected_offset is not None
                and candidate.feature.t_sqi_pass
            ]
            usable = hard if len(hard) >= 4 else [
                candidate
                for candidate in beat_candidates
                if candidate.selected_offset is not None
                and float(candidate.feature.t_sqi_score or 0.0) >= 0.15
            ]
            indices = [int(candidate.selected_offset) for candidate in usable]
            weights = [
                max(float(candidate.feature.t_fusion_weight or 0.0), 1e-3)
                for candidate in usable
            ]
            sources = [candidate.feature.lead for candidate in usable]
            methods = [candidate.selected_method for candidate in usable]
            lead_groups = [_source_group(candidate.feature.lead) for candidate in usable]
            for name, (_onset, offset, weight) in derived.items():
                if offset is not None:
                    indices.append(int(offset))
                    weights.append(float(weight))
                    sources.append(name)
                    methods.append("derived")
                    lead_groups.append("derived")
            return robust_t_offset_fusion(
                indices,
                weights,
                sources,
                fs=fs,
                methods=methods,
                lead_groups=lead_groups,
            )

        fusion = fuse()
        # A suspect early local endpoint can only move to a later independent
        # candidate when at least two other lead/derived endpoints support it.
        if (
            fusion.center_index is not None
            and fusion.reliable
            and fusion.mad_ms is not None
            and fusion.mad_ms <= 12.0
        ):
            for candidate in beat_candidates:
                selected = candidate.selected_offset
                if not candidate.suspect or selected is None:
                    continue
                if selected >= fusion.center_index - int(round(0.020 * fs)):
                    continue
                later = [
                    value
                    for value in candidate.offset_candidates.values()
                    if value > selected + int(round(0.020 * fs))
                    and abs(value - fusion.center_index) <= int(round(0.030 * fs))
                ]
                support = sum(
                    1
                    for other in beat_candidates
                    if other is not candidate
                    and other.selected_offset is not None
                    and abs(other.selected_offset - fusion.center_index)
                    <= int(round(0.030 * fs))
                )
                support += sum(
                    1
                    for _name, (_onset, offset, _weight) in derived.items()
                    if offset is not None
                    and abs(offset - fusion.center_index) <= int(round(0.030 * fs))
                )
                if later and support >= 2:
                    candidate.selected_offset = int(round(float(np.median(later))))
                    candidate.selected_method = (
                        f"{candidate.selected_method}_cross_lead_late_tail"
                    )
                    _update_t_measurements(candidate, fs=fs, rescue=True)
            fusion = fuse()

        morphology = _assess_t_fusion_morphology(
            fusion,
            beat_candidates,
            derived,
            fs=fs,
        )
        statistical_fusion_reliable = fusion.reliable
        fusion = replace(
            fusion,
            reliable=morphology.reliable,
            reliability_reasons=morphology.reasons,
        )
        derived_offsets = [
            offset for _onset, offset, _weight in derived.values() if offset is not None
        ]
        rms_on, rms_off, _ = derived.get("RMS", (None, None, 0.0))
        pc1_on, pc1_off, _ = derived.get("PC1", (None, None, 0.0))
        qrs_onsets = [
            int(feature.qrs.onset)
            for feature in items
            if feature.qrs.onset is not None
            and (
                quality is None
                or bool(getattr(quality.get(feature.lead), "reliable_for_qrs", False))
            )
        ]
        global_qrs_on = (
            int(round(float(np.percentile(qrs_onsets, 15)))) if qrs_onsets else None
        )
        for feature in items:
            feature.t_offset_robust_center_index = fusion.center_index
            feature.t_offset_latest_p85_index = fusion.latest_p85_index
            feature.t_offset_fusion_ci_low_index = fusion.ci_low_index
            feature.t_offset_fusion_ci_high_index = fusion.ci_high_index
            feature.t_offset_fusion_support = fusion.support
            feature.t_offset_fusion_mad_ms = fusion.mad_ms
            feature.t_offset_fusion_ci_half_width_ms = (
                float(
                    (fusion.ci_high_index - fusion.ci_low_index)
                    * 500.0
                    / fs
                )
                if fusion.ci_low_index is not None
                and fusion.ci_high_index is not None
                else None
            )
            feature.t_offset_fusion_used_leads = ",".join(fusion.used_sources) or None
            feature.t_offset_fusion_excluded_leads = (
                ",".join(fusion.excluded_sources) or None
            )
            feature.t_offset_fusion_reliable = fusion.reliable
            feature.t_offset_statistical_fusion_reliable = (
                statistical_fusion_reliable
            )
            feature.t_offset_cluster_count = fusion.cluster_count
            feature.t_offset_selected_cluster_score = fusion.selected_cluster_score
            feature.t_offset_selected_cluster_support = (
                fusion.selected_cluster_support
            )
            feature.t_offset_selected_cluster_lead_groups = (
                ",".join(fusion.selected_cluster_lead_groups) or None
            )
            feature.t_offset_selected_cluster_methods = (
                ",".join(fusion.selected_cluster_methods) or None
            )
            feature.t_offset_fusion_reliability_reason = (
                ",".join(fusion.reliability_reasons) or "reliable"
            )
            feature.t_peak_consensus_index = morphology.t_peak_consensus_index
            feature.t_global_tpte_ms = morphology.tpte_ms
            feature.t_offset_derived_disagreement_ms = (
                morphology.derived_disagreement_ms
            )
            feature.t_offset_tail_incomplete_leads = (
                ",".join(morphology.tail_incomplete_leads) or None
            )
            feature.t_offset_systematic_early_risk = (
                morphology.systematic_early_risk
            )
            feature.t_offset_morphology_guard_pass = morphology.reliable
            feature.t_rms_onset_index = rms_on
            feature.t_rms_offset_index = rms_off
            feature.t_pc1_onset_index = pc1_on
            feature.t_pc1_offset_index = pc1_off
            feature.t_derived_spread_ms = (
                float(np.ptp(derived_offsets) * 1000.0 / fs)
                if len(derived_offsets) >= 2
                else 0.0 if derived_offsets else None
            )
            # An unreliable refinement never erases the existing consensus
            # measurement.  It simply declines to overwrite it and marks the
            # record non-reportable downstream, preserving a numeric audit
            # trail for comparison and later rescue paths.
            if (
                fusion.reliable
                and global_qrs_on is not None
                and fusion.center_index is not None
            ):
                feature.qt_consensus_ms = float(
                    (fusion.center_index - global_qrs_on) * 1000.0 / fs
                )
            if (
                fusion.reliable
                and global_qrs_on is not None
                and fusion.latest_p85_index is not None
            ):
                feature.qt_latest_p85_ms = float(
                    (fusion.latest_p85_index - global_qrs_on) * 1000.0 / fs
                )
