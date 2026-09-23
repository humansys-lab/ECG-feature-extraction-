"""U-wave localisation and signed measurement.

Until now the U wave existed in this package only as a by-product of T-offset
fusion: ``_clamp_t_end_to_tu_nadir`` needed to know whether a late same-polarity
hump followed the T wave so it could stop the T wave at the T-U nadir instead of
running through it.  The value it left behind (``u_amp_mv``) was
``max(|signal - baseline|)`` over a fixed 200 ms window -- an unsigned magnitude
taken without any peak test.

That is enough to answer "should T-end stop here", and not enough to answer
either of the two questions the clinical literature actually asks of the U wave:

  * prominent U wave  -- |U| above ~0.1-0.2 mV, or above 25 % of the T wave in
    the same lead (bradycardia, hypokalemia, hypomagnesemia, LVH, drug effect);
  * inverted U wave   -- a negative U wave in a lead whose T wave is upright.
    Rare, easy to miss, and highly specific for myocardial ischemia.

Both need a *signed* amplitude, and the inverted case needs it to be trustworthy
in the presence of baseline wander -- exactly the failure mode an unsigned
window maximum is blind to.  So this module re-derives the U wave as a genuine
local extremum with a prominence test and edge guards, and reports polarity.

Reference: AHA/ACCF/HRS 2009 Part IV (ST segment, T and U waves, QT interval);
LITFL ECG Library "U wave".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Search window after T-offset.  The U wave peaks roughly 90-110 ms after
# T-end at normal rates; 240 ms is generous enough for bradycardia while still
# ending before the next P wave in all but extreme cases (the caller also
# passes a next-beat cap).
U_SEARCH_START_MS = 10.0
U_SEARCH_END_MS = 240.0

# A candidate must be a turning point that rises and falls by at least this
# much relative to the flanking troughs.  Baseline wander across a 240 ms
# window is typically monotonic, so it fails the two-sided prominence test.
U_MIN_PROMINENCE_MV = 0.020

# The extremum may not sit against either window edge: a peak at the far edge
# is usually the next P wave, and a peak at the near edge is usually T-wave
# tail that the T-offset estimate cut short.
U_EDGE_GUARD_MS = 15.0

# Absolute floor on the reported amplitude.  Below this the deflection is not
# separable from noise on a typical recording.
U_MIN_AMPLITUDE_MV = 0.020

# The U wave must be a *separate* wave, which means an isoelectric T-U segment
# has to exist between the T offset and the U onset.  Without this test the
# terminal negative lobe of a biphasic T wave is indistinguishable from an
# inverted U wave: T-offset estimators place the offset at the polarity
# crossing, so the negative lobe then opens exactly where the U-wave search
# window starts.  A biphasic T crosses the isoelectric line and keeps going; a
# real T-U nadir rests near it.  Requiring a minimum *duration* near baseline,
# rather than merely touching it, is what separates the two.
U_ISOELECTRIC_MIN_MS = 20.0
U_ISOELECTRIC_BAND_MV = 0.030
U_ISOELECTRIC_BAND_FRACTION = 0.30

# The U wave is the slowest deflection on the surface ECG -- 150-250 ms wide at
# its base, which is roughly 90-120 ms between the 25 %-of-peak points used
# here.  A narrow spike after the T wave is noise, a filter ripple or an
# artefact, never a U wave, and no amplitude or prominence test excludes one:
# a 30 ms blip can be both deep and locally prominent.  Duration is the only
# thing that separates them, so it is checked directly.  The floor is set well
# below the physiological width so that a partially-delineated U wave still
# survives.
U_MIN_DURATION_MS = 60.0


@dataclass(frozen=True)
class UWaveMeasurement:
    """One lead-beat U-wave measurement.

    ``amplitude_mv`` is signed relative to the beat baseline; ``polarity`` is
    +1/-1/0.  ``reliable`` is False whenever any acceptance test failed, in
    which case every other numeric field is None -- callers must never treat a
    missing U wave as an absent U wave (it may simply be unmeasurable).
    """

    present: bool = False
    reliable: bool = False
    peak_index: Optional[int] = None
    amplitude_mv: Optional[float] = None
    polarity: int = 0
    prominence_mv: Optional[float] = None
    onset_index: Optional[int] = None
    offset_index: Optional[int] = None
    duration_ms: Optional[float] = None
    # Length of the isoelectric T-U segment preceding the U wave. Short values
    # mean the deflection runs straight on from the T wave.
    isoelectric_gap_ms: Optional[float] = None
    reject_reason: Optional[str] = None


def _smooth(values: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase moving average, ~10 ms wide."""
    width = max(1, int(round(0.010 * fs)))
    if width <= 1 or values.size < width:
        return values
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def _interior_extremum(dev: np.ndarray, guard: int) -> Optional[int]:
    """Index of the largest-magnitude interior turning point, or None."""
    if dev.size < 2 * guard + 3:
        return None
    best_index: Optional[int] = None
    best_magnitude = 0.0
    for index in range(guard + 1, dev.size - guard - 1):
        value = float(dev[index])
        rises = value > dev[index - 1] and value >= dev[index + 1]
        falls = value < dev[index - 1] and value <= dev[index + 1]
        if not (rises or falls):
            continue
        magnitude = abs(value)
        if magnitude > best_magnitude:
            best_magnitude = magnitude
            best_index = index
    return best_index


def _prominence(dev: np.ndarray, peak: int) -> float:
    """Two-sided prominence of ``peak`` against the flanking extremes."""
    value = float(dev[peak])
    polarity = 1.0 if value >= 0.0 else -1.0
    signed = dev * polarity
    left = float(np.min(signed[: peak + 1]))
    right = float(np.min(signed[peak:]))
    return float(signed[peak]) - max(left, right)


def _isoelectric_gap_samples(dev: np.ndarray, onset: int, amplitude: float) -> int:
    """Longest run of near-baseline samples before ``onset``.

    Measured as a run rather than a single crossing so that a waveform which
    merely passes through the isoelectric line on its way somewhere else -- the
    biphasic T wave -- does not qualify.
    """
    if onset <= 0:
        return 0
    band = max(U_ISOELECTRIC_BAND_MV, U_ISOELECTRIC_BAND_FRACTION * abs(amplitude))
    near = np.abs(dev[:onset]) <= band
    best = current = 0
    for flag in near:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _boundaries(dev: np.ndarray, peak: int) -> tuple[int, int]:
    """Walk outward from the peak to the 25 %-of-peak return points."""
    value = float(dev[peak])
    polarity = 1.0 if value >= 0.0 else -1.0
    signed = dev * polarity
    threshold = 0.25 * float(signed[peak])
    onset = peak
    while onset > 0 and signed[onset - 1] > threshold:
        onset -= 1
    offset = peak
    last = dev.size - 1
    while offset < last and signed[offset + 1] > threshold:
        offset += 1
    return onset, offset


def measure_u_wave(
    signal: np.ndarray,
    *,
    t_off: Optional[int],
    baseline: float,
    fs: int,
    search_cap: Optional[int] = None,
) -> UWaveMeasurement:
    """Locate the U wave following ``t_off`` and measure it with sign.

    ``search_cap`` is the caller's next-beat guard (in the same index space as
    ``signal``); the window never extends past it.
    """
    if t_off is None:
        return UWaveMeasurement(reject_reason="no_t_offset")
    try:
        t_off_i = int(t_off)
        fs_val = float(fs)
        baseline_val = float(baseline)
    except (TypeError, ValueError, OverflowError):
        return UWaveMeasurement(reject_reason="invalid_inputs")
    if not np.isfinite(fs_val) or fs_val <= 0.0 or not np.isfinite(baseline_val):
        return UWaveMeasurement(reject_reason="invalid_inputs")

    try:
        arr = np.asarray(signal, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return UWaveMeasurement(reject_reason="invalid_inputs")
    if arr.ndim != 1:
        return UWaveMeasurement(reject_reason="invalid_inputs")

    lo = t_off_i + max(1, int(round(U_SEARCH_START_MS / 1000.0 * fs_val)))
    hi = t_off_i + int(round(U_SEARCH_END_MS / 1000.0 * fs_val))
    hi = min(hi, arr.size - 1)
    if search_cap is not None:
        try:
            hi = min(hi, int(search_cap))
        except (TypeError, ValueError, OverflowError):
            pass
    if lo < 0 or hi <= lo + 2:
        return UWaveMeasurement(reject_reason="window_too_short")

    window = arr[lo : hi + 1]
    if not np.all(np.isfinite(window)):
        return UWaveMeasurement(reject_reason="non_finite_window")

    dev = _smooth(window - baseline_val, fs_val)
    guard = max(1, int(round(U_EDGE_GUARD_MS / 1000.0 * fs_val)))
    peak = _interior_extremum(dev, guard)
    if peak is None:
        return UWaveMeasurement(reject_reason="no_interior_extremum")

    amplitude = float(dev[peak])
    if abs(amplitude) < U_MIN_AMPLITUDE_MV:
        return UWaveMeasurement(reject_reason="below_amplitude_floor")

    prominence = _prominence(dev, peak)
    if prominence < U_MIN_PROMINENCE_MV:
        return UWaveMeasurement(reject_reason="insufficient_prominence")

    onset, offset = _boundaries(dev, peak)
    duration_ms = (offset - onset) * 1000.0 / fs_val
    if duration_ms < U_MIN_DURATION_MS:
        return UWaveMeasurement(
            reject_reason="too_narrow_for_a_u_wave",
            duration_ms=float(duration_ms),
        )

    gap_samples = _isoelectric_gap_samples(dev, onset, amplitude)
    gap_ms = gap_samples * 1000.0 / fs_val
    if gap_ms < U_ISOELECTRIC_MIN_MS:
        return UWaveMeasurement(
            reject_reason="no_isoelectric_t_u_segment",
            isoelectric_gap_ms=float(gap_ms),
        )

    return UWaveMeasurement(
        present=True,
        reliable=True,
        peak_index=int(lo + peak),
        amplitude_mv=amplitude,
        polarity=1 if amplitude > 0.0 else -1,
        prominence_mv=float(prominence),
        onset_index=int(lo + onset),
        offset_index=int(lo + offset),
        duration_ms=float(duration_ms),
        isoelectric_gap_ms=float(gap_ms),
    )
