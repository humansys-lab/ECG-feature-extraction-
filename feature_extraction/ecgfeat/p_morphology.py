from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .numeric import trapezoid

PComponentMeasurements = Dict[str, Optional[float] | bool]


def _defaults() -> PComponentMeasurements:
    return {
        "is_notched": False,
        "notch_interval_ms": None,
        "is_biphasic": False,
        "initial_duration_ms": None,
        "initial_amplitude_mV": None,
        "terminal_duration_ms": None,
        "terminal_amplitude_mV": None,
        "terminal_area_mv_ms": None,
    }


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _positive_local_peaks(seg: np.ndarray, threshold: float) -> list[int]:
    peaks: list[int] = []
    for idx in range(1, len(seg) - 1):
        if seg[idx] >= threshold and seg[idx] > seg[idx - 1] and seg[idx] >= seg[idx + 1]:
            peaks.append(idx)
    return peaks


def _trapezoid(y: np.ndarray, *, dx: float) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    return float(trapezoid(y, dx=dx))


def measure_p_components(
    sig,
    *,
    p_on: Optional[int],
    p_peak: Optional[int],
    p_off: Optional[int],
    baseline: float,
    fs: int,
) -> PComponentMeasurements:
    out = _defaults()
    if p_on is None or p_peak is None or p_off is None:
        return out

    try:
        p_on_i = int(p_on)
        p_peak_i = int(p_peak)
        p_off_i = int(p_off)
        fs_val = float(fs)
        baseline_val = float(baseline)
    except (TypeError, ValueError, OverflowError):
        return out
    if not np.isfinite(fs_val) or fs_val <= 0.0 or not np.isfinite(baseline_val):
        return out

    try:
        arr = np.asarray(sig, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return out
    if (
        arr.ndim != 1
        or p_on_i < 0
        or p_off_i >= len(arr)
        or p_off_i <= p_on_i + 2
        or not (p_on_i <= p_peak_i <= p_off_i)
    ):
        return out

    seg = arr[p_on_i : p_off_i + 1] - baseline_val
    if not np.all(np.isfinite(seg)):
        return out

    max_abs = float(np.max(np.abs(seg))) if len(seg) else 0.0
    if max_abs <= 1e-8:
        return out

    max_pos = float(np.max(seg))

    # Notching: two positive local maxima separated by at least 20 ms with a
    # real intervening dip, so broad single-hump plateaus do not split.
    if max_pos > 0.0:
        pos_threshold = max(0.02, 0.20 * max_pos)
        min_separation = max(1, int(round(0.020 * fs_val)))
        peaks = _positive_local_peaks(seg, pos_threshold)
        for left, right in zip(peaks, peaks[1:]):
            if right - left < min_separation:
                continue
            smaller_peak = min(float(seg[left]), float(seg[right]))
            valley = float(np.min(seg[left : right + 1]))
            required_dip = max(0.01, 0.10 * smaller_peak)
            if valley <= smaller_peak - required_dip:
                out["is_notched"] = True
                out["notch_interval_ms"] = (right - left) * 1000.0 / fs_val
                break

    positive_threshold = max(0.02, 0.05 * max_abs)
    terminal_amp_threshold = max(0.02, 0.05 * max_abs)
    neg_threshold = max(1e-6, 0.01 * max_abs)
    neg_runs = _contiguous_true_runs(seg < -neg_threshold)
    terminal_start: Optional[int] = None
    terminal_end: Optional[int] = None
    initial_amp: Optional[float] = None
    for run_start, run_end in reversed(neg_runs):
        initial_candidate = seg[:run_start]
        if len(initial_candidate) < 2:
            continue
        candidate_amp = float(np.max(initial_candidate))
        if candidate_amp <= positive_threshold:
            continue

        candidate = seg[run_start:run_end]
        candidate_extreme = float(candidate[int(np.argmax(np.abs(candidate)))])
        candidate_duration_ms = (run_end - run_start) * 1000.0 / fs_val
        candidate_area = _trapezoid(candidate, dx=1000.0 / fs_val)
        if (
            candidate_extreme < -terminal_amp_threshold
            and candidate_duration_ms >= 30.0
            and candidate_area < 0.0
        ):
            terminal_start = run_start
            terminal_end = run_end - 1
            initial_amp = candidate_amp
            break

    initial_end = p_peak_i - p_on_i if terminal_start is None else max(0, terminal_start - 1)
    initial = seg[: initial_end + 1]
    if len(initial) >= 2:
        initial_amp = float(np.max(initial))
        out["initial_duration_ms"] = initial_end * 1000.0 / fs_val
        out["initial_amplitude_mV"] = initial_amp
    else:
        initial_amp = None

    if terminal_start is None or terminal_end is None or terminal_end <= terminal_start:
        return out

    terminal = seg[terminal_start : terminal_end + 1]
    terminal_extreme = float(terminal[int(np.argmax(np.abs(terminal)))])
    terminal_duration_ms = (terminal_end - terminal_start + 1) * 1000.0 / fs_val
    terminal_area = _trapezoid(terminal, dx=1000.0 / fs_val)

    out["terminal_duration_ms"] = terminal_duration_ms
    out["terminal_amplitude_mV"] = terminal_extreme
    out["terminal_area_mv_ms"] = terminal_area

    if (
        initial_amp is not None
        and initial_amp > max(0.02, 0.05 * max_abs)
        and terminal_extreme < -terminal_amp_threshold
        and terminal_duration_ms >= 30.0
        and terminal_area < 0.0
    ):
        out["is_biphasic"] = True

    return out
