"""Optional ST amplitude signal using measured PR baseline anchors.

This path works from calibrated samples, before median baseline subtraction.
It does not replace the established signal used for detection/delineation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .models import LeadBeatFeatures, STANDARD_12_LEADS
from .preprocess import notch_filter


@dataclass(frozen=True)
class STBaselineResult:
    signal: np.ndarray
    anchor_counts: dict[str, int]
    unavailable_leads: tuple[str, ...]
    uncertainty_mv: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    anchor_sources: dict[str, dict[str, int]] = field(default_factory=dict)


def calibrated_st_signal(
    ecg: np.ndarray,
    fs: int,
    beat_features: Sequence[LeadBeatFeatures],
    *,
    mains_hz: int = 50,
) -> STBaselineResult:
    """Remove drift interpolated between quiet, measured post-P PR segments.

    At least two distinct quiet PR anchors are required per lead. Unsupported
    leads retain notch-filtered samples and are explicitly marked unavailable;
    callers must not treat their ST amplitudes as validated measurements.
    """
    values = np.asarray(ecg, dtype=float)
    if values.ndim != 2 or values.shape[0] != 12 or values.shape[1] == 0:
        raise ValueError("calibrated ST input must have shape [12, n_samples]")
    if fs <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("calibrated ST input must be finite with fs > 0")
    filtered = notch_filter(values, fs, mains_hz=mains_hz)
    corrected = filtered.copy()
    gap = max(1, int(round(0.008 * fs)))
    minimum = max(3, int(round(0.012 * fs)))
    maximum = max(minimum, int(round(0.060 * fs)))
    anchors: dict[str, dict[float, float]] = {lead: {} for lead in STANDARD_12_LEADS}
    for feature in beat_features:
        if feature.lead not in anchors or "p_unreliable" in feature.flags:
            continue
        p_off, qrs_on = feature.p.offset, feature.qrs.onset
        if p_off is None or qrs_on is None:
            continue
        hi = min(values.shape[1], int(qrs_on) - gap)
        lo = max(0, int(p_off) + gap, hi - maximum)
        if hi - lo < minimum:
            continue
        segment = filtered[STANDARD_12_LEADS.index(feature.lead), lo:hi]
        center = float(np.median(segment))
        noise = 1.4826 * float(np.median(np.abs(segment - center)))
        if noise > 0.025:
            continue
        anchors[feature.lead][(lo + hi - 1) / 2.0] = center
    counts: dict[str, int] = {}
    unavailable = []
    positions = np.arange(values.shape[1], dtype=float)
    for index, lead in enumerate(STANDARD_12_LEADS):
        ordered = sorted(anchors[lead].items())
        counts[lead] = len(ordered)
        if len(ordered) < 2:
            unavailable.append(lead)
            continue
        times = np.asarray([time for time, _ in ordered])
        levels = np.asarray([level for _, level in ordered])
        baseline = np.interp(positions, times, levels)
        # Linear continuation at the short record edges preserves a linear
        # drift without imposing a zero baseline at the first/last beat.
        for first, second, mask in (
            (0, 1, positions < times[0]),
            (-1, -2, positions > times[-1]),
        ):
            slope = (levels[second] - levels[first]) / (times[second] - times[first])
            baseline[mask] = levels[first] + slope * (positions[mask] - times[first])
        corrected[index] -= baseline
    return STBaselineResult(corrected, counts, tuple(unavailable))


def adaptive_st_signal(ecg, fs, beat_features, *, mains_hz=50,
                       max_anchor_gap_seconds=2.0, max_extrapolation_seconds=.5):
    """PR/TP anchors with bounded support and an uncalibrated error envelope.

    Unsupported samples retain an audit voltage but are explicitly invalid.
    The envelope is a noise/drift heuristic, not a confidence probability.
    """
    values = np.asarray(ecg, dtype=float)
    if (values.ndim != 2 or values.shape[0] != 12 or not values.shape[1]
            or fs <= 0 or not np.all(np.isfinite(values))):
        raise ValueError("adaptive ST requires finite [12, n] samples and positive fs")
    if max_anchor_gap_seconds <= 0 or max_extrapolation_seconds < 0:
        raise ValueError("anchor gap must be positive and extrapolation nonnegative")
    signal = notch_filter(values, fs, mains_hz=mains_hz)
    corrected = signal.copy()
    uncertainty = np.full_like(signal, np.inf)
    valid = np.zeros_like(signal, dtype=bool)
    counts, sources, unavailable = {}, {}, []
    positions = np.arange(values.shape[1])
    guard, minimum = max(1, int(.012 * fs)), max(4, int(.016 * fs))
    for li, lead in enumerate(STANDARD_12_LEADS):
        features = sorted((b for b in beat_features if b.lead == lead), key=lambda b: b.beat_id)
        anchors = []
        previous_t = None
        for beat in features:
            p_on, p_off, q_on = beat.p.onset, beat.p.offset, beat.qrs.onset
            intervals = []
            if "p_unreliable" not in beat.flags:
                if p_off is not None and q_on is not None:
                    intervals.append((p_off + guard, q_on - guard, "PR"))
                if previous_t is not None and p_on is not None:
                    intervals.append((previous_t + guard, p_on - guard, "TP"))
            previous_t = beat.t.offset
            for lo, hi, source in intervals:
                lo, hi = max(0, int(lo)), min(len(positions), int(hi))
                if hi - lo < minimum:
                    continue
                # Limit wide TP windows: evaluate their middle, away from feet.
                center = (lo + hi) // 2
                lo, hi = max(lo, center - int(.030 * fs)), min(hi, center + int(.030 * fs))
                segment = signal[li, lo:hi]
                level = float(np.median(segment))
                noise = float(1.4826 * np.median(np.abs(segment - level)))
                half = max(1, len(segment) // 2)
                change = abs(float(np.median(segment[:half]) - np.median(segment[-half:])))
                if noise <= .020 and change <= .025:
                    anchors.append(((lo + hi - 1) / 2, level, max(noise, .001), source))
        # Combine coincident windows without counting them twice.
        anchors = sorted({a[0]: a for a in anchors}.values())
        if len(anchors) >= 5:
            times = np.asarray([a[0] for a in anchors])
            levels = np.asarray([a[1] for a in anchors])
            keep = np.ones(len(anchors), dtype=bool)
            for i in range(1, len(anchors) - 1):
                predicted = np.interp(times[i], times[[i - 1, i + 1]], levels[[i - 1, i + 1]])
                if abs(levels[i] - predicted) > max(.040, 4 * anchors[i][2]):
                    keep[i] = False
            anchors = [a for a, use in zip(anchors, keep) if use]
        counts[lead] = len(anchors)
        sources[lead] = {name: sum(a[3] == name for a in anchors) for name in ("PR", "TP")}
        if len(anchors) < 2:
            unavailable.append(lead)
            continue
        times, levels, noises = (np.asarray([a[k] for a in anchors]) for k in range(3))
        baseline = np.interp(positions, times, levels)
        for a, b, mask in ((0, 1, positions < times[0]), (-1, -2, positions > times[-1])):
            slope = (levels[b] - levels[a]) / (times[b] - times[a])
            baseline[mask] = levels[a] + slope * (positions[mask] - times[a])
        right = np.clip(np.searchsorted(times, positions), 1, len(times) - 1)
        left = right - 1
        distance = np.minimum(abs(positions - times[left]), abs(positions - times[right])) / fs
        gap = (times[right] - times[left]) / fs
        edge_distance = np.maximum(np.maximum(times[0] - positions, positions - times[-1]), 0) / fs
        envelope = np.interp(positions, times, noises) + .010 * distance + .010 * edge_distance
        uncertainty[li] = envelope
        valid[li] = ((gap <= max_anchor_gap_seconds) & (edge_distance <= max_extrapolation_seconds)
                     & (envelope <= .030))
        corrected[li] -= baseline
        if not np.any(valid[li]):
            unavailable.append(lead)
    return STBaselineResult(corrected, counts, tuple(unavailable), uncertainty, valid, sources)
