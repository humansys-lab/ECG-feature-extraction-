from __future__ import annotations

from math import atan2, degrees, isfinite
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


GLASGOW_PROFILE_KEYS = (
    "beat_window_start_index",
    "reference_mv",
    "p_onset_ms",
    "p_duration_ms",
    "qrs_onset_ms",
    "qrs_duration_ms",
    "q_duration_ms",
    "r_duration_ms",
    "s_duration_ms",
    "r_prime_duration_ms",
    "s_prime_duration_ms",
    "t_onset_ms",
    "p_positive_duration_ms",
    "vat_ms",
    "p_positive_amp_mv",
    "p_negative_amp_mv",
    "qrs_peak_to_peak_mv",
    "q_amp_mv",
    "r_amp_mv",
    "s_amp_mv",
    "r_prime_amp_mv",
    "s_prime_amp_mv",
    "st_amp_mv",
    "st_2_8_amp_mv",
    "st_3_8_amp_mv",
    "t_positive_amp_mv",
    "t_negative_amp_mv",
    "qrs_area_uv_ms",
    "qrs_area_matrix",
    "qrs_signed_area_uv_ms",
    "qrs_signed_area_matrix",
    "t_morphology",
    "r_wave_notch_count",
    "delta_confidence_pct",
    "st_slope_deg",
    "qt_ms",
    "qrs_notch_slur_amp_mv",
    "pr_amp_mv",
    "st_adjusted_amp_mv",
    "st_2_8_index",
    "st_3_8_index",
)


def _valid_index(index: Optional[int], size: int) -> Optional[int]:
    if index is None:
        return None
    try:
        result = int(index)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if 0 <= result < size else None


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _duration_ms(start: Optional[int], end: Optional[int], fs: float) -> Optional[float]:
    if start is None or end is None or end < start:
        return None
    return float(end - start) * 1000.0 / fs


def _elapsed_ms(index: Optional[int], fs: float) -> Optional[float]:
    return None if index is None else float(index) * 1000.0 / fs


def _component_amplitudes(segment: np.ndarray, reference: float) -> tuple[Optional[float], Optional[float]]:
    if segment.size == 0:
        return None, None
    relative = np.asarray(segment, dtype=float) - reference
    positive = float(max(float(np.max(relative)), 0.0))
    negative = float(min(float(np.min(relative)), 0.0))
    return positive, negative


def _positive_duration_ms(segment: np.ndarray, reference: float, fs: float) -> Optional[float]:
    if segment.size == 0:
        return None
    positive = np.asarray(segment, dtype=float) > reference
    if not np.any(positive):
        return 0.0
    padded = np.pad(positive.astype(int), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    longest = int(np.max(ends - starts))
    return longest * 1000.0 / fs


def _t_morphology(segment: np.ndarray, reference: float) -> Optional[int]:
    """Encode the guide's simple T component order as -2, -1, +1, or +2."""

    if segment.size < 2:
        return None
    relative = np.asarray(segment, dtype=float) - reference
    scale = float(np.max(np.abs(relative)))
    if not isfinite(scale) or scale < 0.005:
        return None
    threshold = max(0.005, 0.05 * scale)
    positive = np.flatnonzero(relative >= threshold)
    negative = np.flatnonzero(relative <= -threshold)
    if positive.size and negative.size:
        positive_peak = int(np.argmax(relative))
        negative_peak = int(np.argmin(relative))
        return 2 if positive_peak < negative_peak else -2
    if positive.size:
        return 1
    if negative.size:
        return -1
    return None


def _notch_slur_amplitude(
    signal: np.ndarray,
    qrs_on: Optional[int],
    qrs_off: Optional[int],
    r_peak: Optional[int],
    reference: Optional[float],
    notch_count: int,
) -> Optional[float]:
    if (
        reference is None
        or qrs_on is None
        or qrs_off is None
        or qrs_off <= qrs_on + 2
        or int(notch_count) <= 0
    ):
        return None
    start = max(qrs_on + 1, (r_peak or qrs_on) + 1)
    if start >= qrs_off:
        return None
    tail = signal[start : qrs_off + 1]
    if tail.size < 3:
        return None
    gradient = np.diff(tail)
    changes = np.flatnonzero(gradient[:-1] * gradient[1:] <= 0.0) + 1
    if changes.size == 0:
        return None
    index = int(changes[0])
    return float(tail[index] - reference)


def measure_glasgow_profile(
    signal_mv: Sequence[float] | np.ndarray,
    *,
    fs: float,
    beat_window_start_index: Optional[int],
    p_on: Optional[int],
    p_off: Optional[int],
    qrs_on: Optional[int],
    qrs_off: Optional[int],
    t_on: Optional[int],
    t_off: Optional[int],
    r_peak: Optional[int],
    baseline_mv: float,
    r_prime_amp_baseline_mv: Optional[float],
    s_prime_amp_baseline_mv: Optional[float],
    component_durations: Mapping[str, Optional[float]],
    vat_ms: Optional[float],
    qt_ms: Optional[float],
    delta_present: bool,
    delta_confidence: Optional[float],
    qrs_notch_count: int,
    paper_speed_mm_per_s: float = 25.0,
    gain_mm_per_mv: float = 10.0,
) -> Dict[str, object]:
    """Measure a per-lead Glasgow-compatible representative-beat profile.

    Indices are local to ``signal_mv``. Amplitudes use the raw sample at QRS
    onset as the guide-defined horizontal reference. Missing bounds remain
    explicit rather than being imputed.
    """

    sig = np.asarray(signal_mv, dtype=float).reshape(-1)
    fs_value = _finite(fs)
    if fs_value is None or fs_value <= 0.0:
        raise ValueError("fs must be finite and greater than zero")
    fs = fs_value
    n = int(sig.size)
    p_on = _valid_index(p_on, n)
    p_off = _valid_index(p_off, n)
    qrs_on = _valid_index(qrs_on, n)
    qrs_off = _valid_index(qrs_off, n)
    t_on = _valid_index(t_on, n)
    t_off = _valid_index(t_off, n)
    r_peak = _valid_index(r_peak, n)
    if p_on is not None and p_off is not None and p_off < p_on:
        p_off = None
    if qrs_on is not None and qrs_off is not None and qrs_off < qrs_on:
        qrs_off = None
    if t_on is not None and t_off is not None and t_off < t_on:
        t_off = None

    profile: Dict[str, object] = {key: None for key in GLASGOW_PROFILE_KEYS}
    unavailable: Dict[str, str] = {}
    provenance: Dict[str, str] = {
        key: "glasgow_explicit" for key in GLASGOW_PROFILE_KEYS
    }
    profile["beat_window_start_index"] = (
        None if beat_window_start_index is None else int(beat_window_start_index)
    )

    reference = float(sig[qrs_on]) if qrs_on is not None else None
    profile["reference_mv"] = reference
    profile["p_onset_ms"] = _elapsed_ms(p_on, fs)
    profile["p_duration_ms"] = _duration_ms(p_on, p_off, fs)
    profile["qrs_onset_ms"] = _elapsed_ms(qrs_on, fs)
    profile["qrs_duration_ms"] = _duration_ms(qrs_on, qrs_off, fs)
    profile["t_onset_ms"] = _elapsed_ms(t_on, fs)
    for key in (
        "q_duration_ms",
        "r_duration_ms",
        "s_duration_ms",
        "r_prime_duration_ms",
        "s_prime_duration_ms",
    ):
        profile[key] = _finite(component_durations.get(key))
    profile["vat_ms"] = _finite(vat_ms)
    profile["qt_ms"] = _finite(qt_ms)
    profile["r_wave_notch_count"] = int(max(0, int(qrs_notch_count)))
    if delta_present and _finite(delta_confidence) is not None:
        profile["delta_confidence_pct"] = float(np.clip(float(delta_confidence), 0.0, 1.0) * 100.0)
    elif delta_present:
        unavailable["delta_confidence_pct"] = "delta_confidence_unavailable"
    else:
        profile["delta_confidence_pct"] = 0.0

    if reference is None:
        for key in (
            "p_positive_amp_mv", "p_negative_amp_mv", "qrs_peak_to_peak_mv",
            "q_amp_mv", "r_amp_mv", "s_amp_mv", "r_prime_amp_mv",
            "s_prime_amp_mv", "st_amp_mv", "st_2_8_amp_mv",
            "st_3_8_amp_mv", "t_positive_amp_mv", "t_negative_amp_mv",
            "qrs_area_uv_ms", "qrs_area_matrix", "t_morphology",
            "qrs_signed_area_uv_ms", "qrs_signed_area_matrix",
            "st_slope_deg", "qrs_notch_slur_amp_mv", "pr_amp_mv",
            "st_adjusted_amp_mv",
        ):
            unavailable[key] = "qrs_onset_reference_unavailable"
        for key in (
            "qrs_peak_to_peak_mv", "q_amp_mv", "r_amp_mv", "s_amp_mv",
            "qrs_area_uv_ms", "qrs_area_matrix",
            "qrs_signed_area_uv_ms", "qrs_signed_area_matrix",
        ):
            unavailable[key] = "qrs_bounds_unavailable"
    else:
        if p_on is not None and p_off is not None:
            p_positive, p_negative = _component_amplitudes(sig[p_on : p_off + 1], reference)
            profile["p_positive_amp_mv"] = p_positive
            profile["p_negative_amp_mv"] = p_negative
            profile["p_positive_duration_ms"] = _positive_duration_ms(
                sig[p_on : p_off + 1], reference, fs
            )
            profile["pr_amp_mv"] = float(sig[qrs_on] - sig[p_on])
        else:
            for key in (
                "p_positive_amp_mv", "p_negative_amp_mv",
                "p_positive_duration_ms", "pr_amp_mv",
            ):
                unavailable[key] = "p_bounds_unavailable"

        if qrs_on is not None and qrs_off is not None:
            qrs_segment = sig[qrs_on : qrs_off + 1]
            relative_qrs = qrs_segment - reference
            profile["qrs_peak_to_peak_mv"] = float(np.ptp(qrs_segment))
            # Q/S windows are split at the true local positive peak within this
            # lead's own QRS segment, not at the caller-supplied r_peak — that
            # value is a cross-lead beat-alignment fiducial and need not sit on
            # this lead's actual R peak (see _qrs_points in delineate.py, which
            # this mirrors for consistency between the two measurement paths).
            peak = int(qrs_on + np.argmax(qrs_segment))
            before_r = sig[qrs_on : peak + 1] - reference
            after_r = sig[peak : qrs_off + 1] - reference
            profile["q_amp_mv"] = float(min(float(np.min(before_r)), 0.0))
            profile["r_amp_mv"] = float(max(float(np.max(relative_qrs)), 0.0))
            profile["s_amp_mv"] = float(min(float(np.min(after_r)), 0.0))
            dx_ms = 1000.0 / fs
            area_uv_ms = float(_trapezoid(np.abs(relative_qrs), dx=dx_ms) * 1000.0)
            profile["qrs_area_uv_ms"] = area_uv_ms
            profile["qrs_area_matrix"] = area_uv_ms / 20.0
            signed_area_uv_ms = float(_trapezoid(relative_qrs, dx=dx_ms) * 1000.0)
            profile["qrs_signed_area_uv_ms"] = signed_area_uv_ms
            profile["qrs_signed_area_matrix"] = signed_area_uv_ms / 20.0
        else:
            for key in (
                "qrs_peak_to_peak_mv", "q_amp_mv", "r_amp_mv", "s_amp_mv",
                "qrs_area_uv_ms", "qrs_area_matrix",
                "qrs_signed_area_uv_ms", "qrs_signed_area_matrix",
            ):
                unavailable[key] = "qrs_bounds_unavailable"

        baseline = _finite(baseline_mv) or 0.0
        for key, amplitude in (
            ("r_prime_amp_mv", r_prime_amp_baseline_mv),
            ("s_prime_amp_mv", s_prime_amp_baseline_mv),
        ):
            value = _finite(amplitude)
            if value is None:
                unavailable[key] = f"{key.removesuffix('_amp_mv')}_unavailable"
            else:
                profile[key] = value + baseline - reference

        if qrs_off is not None:
            profile["st_amp_mv"] = float(sig[qrs_off] - reference)
            if profile["pr_amp_mv"] is not None:
                profile["st_adjusted_amp_mv"] = float(profile["st_amp_mv"]) - float(profile["pr_amp_mv"])
                provenance["st_adjusted_amp_mv"] = "existing_dxl_approximation"
            else:
                unavailable["st_adjusted_amp_mv"] = "pr_amplitude_unavailable"
        else:
            unavailable["st_amp_mv"] = "qrs_offset_unavailable"
            unavailable["st_adjusted_amp_mv"] = "qrs_offset_unavailable"

        if qrs_off is not None and t_off is not None and t_off >= qrs_off:
            st_t_span = t_off - qrs_off
            index_2_8 = int(round(qrs_off + 2.0 * st_t_span / 8.0))
            index_3_8 = int(round(qrs_off + 3.0 * st_t_span / 8.0))
            index_2_8 = min(max(index_2_8, 0), n - 1)
            index_3_8 = min(max(index_3_8, 0), n - 1)
            profile["st_2_8_index"] = index_2_8
            profile["st_3_8_index"] = index_3_8
            profile["st_2_8_amp_mv"] = float(sig[index_2_8] - reference)
            profile["st_3_8_amp_mv"] = float(sig[index_3_8] - reference)
            delta_ms = float(index_3_8 - qrs_off) * 1000.0 / fs
            horizontal_mm = delta_ms * float(paper_speed_mm_per_s) / 1000.0
            vertical_mm = float(sig[index_3_8] - sig[qrs_off]) * float(gain_mm_per_mv)
            profile["st_slope_deg"] = (
                degrees(atan2(vertical_mm, horizontal_mm))
                if horizontal_mm > 0.0
                else None
            )
        else:
            for key in (
                "st_2_8_index", "st_3_8_index", "st_2_8_amp_mv",
                "st_3_8_amp_mv", "st_slope_deg",
            ):
                unavailable[key] = "st_t_bounds_unavailable"

        if t_on is not None and t_off is not None:
            t_segment = sig[t_on : t_off + 1]
            t_positive, t_negative = _component_amplitudes(t_segment, reference)
            profile["t_positive_amp_mv"] = t_positive
            profile["t_negative_amp_mv"] = t_negative
            profile["t_morphology"] = _t_morphology(t_segment, reference)
            if profile["t_morphology"] is None:
                unavailable["t_morphology"] = "t_shape_not_classifiable"
        else:
            for key in ("t_positive_amp_mv", "t_negative_amp_mv", "t_morphology"):
                unavailable[key] = "t_bounds_unavailable"

        profile["qrs_notch_slur_amp_mv"] = _notch_slur_amplitude(
            sig, qrs_on, qrs_off, r_peak, reference, qrs_notch_count
        )
        if profile["qrs_notch_slur_amp_mv"] is None:
            unavailable["qrs_notch_slur_amp_mv"] = "qrs_notch_slur_unavailable"

    for key in GLASGOW_PROFILE_KEYS:
        if profile[key] is None and key not in unavailable:
            unavailable[key] = f"{key}_unavailable"
    profile["unavailable_reasons"] = unavailable
    profile["provenance"] = provenance
    return profile
