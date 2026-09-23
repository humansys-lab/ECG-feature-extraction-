from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


@lru_cache(maxsize=128)
def _butter_coefficients(order: int, cutoff: float | tuple[float, float], kind: str):
    # Coefficients depend only on filter parameters, never on patient samples.
    return butter(order, cutoff, btype=kind)


@lru_cache(maxsize=32)
def _notch_coefficients(mains_hz: float, q: float, fs: float):
    return iirnotch(w0=mains_hz, Q=q, fs=fs)


def _safe_filtfilt(b, a, x: np.ndarray) -> np.ndarray:
    if x.shape[-1] < max(len(a), len(b)) * 3:
        return x.copy()
    return filtfilt(b, a, x)


def resample_ecg(ecg: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    fs_in_float = float(fs_in)
    fs_out_float = float(fs_out)
    if not np.isfinite(fs_in_float) or fs_in_float <= 0.0:
        raise ValueError("fs_in must be finite and greater than zero")
    if not np.isfinite(fs_out_float) or fs_out_float <= 0.0:
        raise ValueError("fs_out must be finite and greater than zero")
    if np.isclose(fs_in_float, fs_out_float, rtol=0.0, atol=1e-12):
        return ecg.copy()

    ratio = (
        Fraction(str(fs_out_float)) / Fraction(str(fs_in_float))
    ).limit_denominator(100_000)
    result = resample_poly(ecg, ratio.numerator, ratio.denominator, axis=-1)
    target_samples = max(
        1,
        int(round(ecg.shape[-1] * fs_out_float / fs_in_float)),
    )
    if result.shape[-1] > target_samples:
        result = result[..., :target_samples]
    elif result.shape[-1] < target_samples:
        pad = [(0, 0)] * result.ndim
        pad[-1] = (0, target_samples - result.shape[-1])
        result = np.pad(result, pad, mode="edge")
    return result


def bandpass_filter(x: np.ndarray, fs: int, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    b, a = _butter_coefficients(order, (low, high), "bandpass")
    return _safe_filtfilt(b, a, x)


def highpass_filter(x: np.ndarray, fs: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    nyq = fs / 2.0
    if nyq <= 0.0:
        raise ValueError("fs must be greater than zero")
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than zero")
    cutoff = min(max(cutoff_hz / nyq, 1e-6), 0.99)
    b, a = _butter_coefficients(order, cutoff, "highpass")
    return _safe_filtfilt(b, a, x)


def lowpass_filter(x: np.ndarray, fs: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    nyq = fs / 2.0
    if nyq <= 0.0:
        raise ValueError("fs must be greater than zero")
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be greater than zero")
    cutoff = min(max(cutoff_hz / nyq, 1e-6), 0.99)
    b, a = _butter_coefficients(order, cutoff, "lowpass")
    return _safe_filtfilt(b, a, x)


def notch_filter(x: np.ndarray, fs: int, mains_hz: int = 50, q: float = 30.0) -> np.ndarray:
    nyq = float(fs) / 2.0
    if nyq <= 0.0:
        raise ValueError("fs must be greater than zero")
    if mains_hz <= 0.0:
        raise ValueError("mains_hz must be greater than zero")
    if float(mains_hz) >= 0.99 * nyq:
        return x.copy()
    b, a = _notch_coefficients(mains_hz, q, fs)
    return _safe_filtfilt(b, a, x)


def zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x - mu) / sigma


def remove_baseline_median(x: np.ndarray, fs: int, win1_ms: int = 200, win2_ms: int = 600) -> np.ndarray:
    # Simple baseline estimate inspired by classic two-stage median filtering.
    from scipy.ndimage import median_filter

    win1 = max(1, int(fs * win1_ms / 1000))
    win2 = max(1, int(fs * win2_ms / 1000))
    if win1 % 2 == 0:
        win1 += 1
    if win2 % 2 == 0:
        win2 += 1
    baseline = median_filter(x, size=(1, win1), mode="nearest")
    baseline = median_filter(baseline, size=(1, win2), mode="nearest")
    return x - baseline


def analysis_signal(ecg: np.ndarray, fs: int, mains_hz: int = 50) -> np.ndarray:
    """Established delineation signal: median baseline removal and mains notch.

    No low-pass is applied. Median subtraction can still attenuate prolonged
    ST/T deflections; calibrated_st_signal provides a separate opt-in amplitude
    path using measured PR anchors. Keep this default stable until a replacement
    has passed both delineation and independent amplitude validation.
    """
    x = remove_baseline_median(ecg, fs)
    x = notch_filter(x, fs, mains_hz=mains_hz)
    return x


def detection_signal(ecg: np.ndarray, fs: int, mains_hz: int = 50, lp_hz: float = 40.0) -> np.ndarray:
    """Detection-grade signal: analysis_signal + low-pass for QRS detection.

    The additional low-pass (default 40 Hz, order=4) suppresses muscle noise
    so that R-peak locations are found reliably.  This signal is used ONLY for
    QRS detector input; all amplitude measurements use analysis_signal instead.
    """
    x = analysis_signal(ecg, fs, mains_hz=mains_hz)
    x = lowpass_filter(x, fs, cutoff_hz=lp_hz, order=4)
    return x
