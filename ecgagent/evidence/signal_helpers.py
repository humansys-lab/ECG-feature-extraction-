"""Raw-signal helpers used by waveform review.

ecgagent consumes ``(raw signal, measurement record)`` and must not import the
measurement engine's private modules (docs/library_design/01_architecture.md).
These two helpers are verbatim copies of the engine's zero-phase Butterworth
low-pass and local P-wave chord evidence; ``tests/test_ecgagent_signal_helpers.py``
proves they stay numerically identical to the engine during the migration.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, filtfilt


@lru_cache(maxsize=32)
def _butter_coefficients(order: int, cutoff: float, kind: str):
    return butter(order, cutoff, btype=kind)


def lowpass_filter(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    nyq = fs / 2.0
    if nyq <= 0.0:
        raise ValueError("fs must be greater than zero")
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than zero")
    cutoff = min(max(cutoff_hz / nyq, 1e-6), 0.99)
    b, a = _butter_coefficients(order, cutoff, "lowpass")
    if x.shape[-1] < max(len(a), len(b)) * 3:
        return x.copy()
    return filtfilt(b, a, x)


def p_waveform_evidence(signal, filtered, peak, fs):
    """Local chord prominence and width; returns evidence, not a probability."""
    half = max(3, round(.08*fs))
    if peak-half < 0 or peak+half >= len(signal):
        return None
    lo, hi = peak-half, peak+half+1
    segment = filtered[lo:hi]
    edge = max(1, round(.01*fs))
    baseline = np.linspace(np.median(segment[:edge]), np.median(segment[-edge:]), len(segment))
    wave = segment-baseline
    center = half
    radius = max(1, round(.02*fs))
    local = abs(wave[center-radius:center+radius+1])
    local_peak = center-radius+int(np.argmax(local))
    amplitude = abs(float(wave[local_peak]))
    residual = signal[lo:hi]-filtered[lo:hi]
    noise = max(.004, 1.4826*float(np.median(abs(residual-np.median(residual)))))
    if amplitude < max(.018, 2.5*noise):
        return None
    polarity = 1 if wave[local_peak] >= 0 else -1
    mask = polarity*wave > .5*amplitude
    left = right = local_peak
    while left > 0 and mask[left-1]:
        left -= 1
    while right+1 < len(wave) and mask[right+1]:
        right += 1
    width = (right-left+1)/fs
    if not .020 <= width <= .120:
        return None
    return {"peak": int(lo+local_peak), "amplitude_mv": amplitude,
            "snr": float(amplitude/noise), "half_height_width_ms": width*1000,
            "snippet": wave / max(float(np.linalg.norm(wave)), 1e-12)}
