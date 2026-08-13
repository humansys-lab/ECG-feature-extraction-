from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

from .models import QRSDetectorResult, QRSCandidateWindow, STANDARD_12_LEADS
from .preprocess import bandpass_filter, zscore


DEFAULT_QRS_LEADS: Sequence[int] = (0, 1, 7, 8, 9, 10, 11)  # I, II, V2-V6
_NEGATIVE_QS_DOMINANCE_RATIO = 1.20
_NEGATIVE_QS_SLOPE_FRACTION = 0.20


def make_vector_magnitude(ecg: np.ndarray, leads: Iterable[int] = DEFAULT_QRS_LEADS) -> np.ndarray:
    xs = []
    for idx in leads:
        xs.append(zscore(ecg[idx]))
    stack = np.vstack(xs)
    vm = np.sqrt(np.mean(stack**2, axis=0))
    return vm


def _detector_confidence(height: float, threshold: float, mwa: np.ndarray) -> float:
    dynamic_range = max(float(np.percentile(mwa, 99) - threshold), float(np.std(mwa)), 1e-8)
    return float(np.clip((height - threshold) / dynamic_range, 0.0, 1.0))


def _qrs_fiducial_from_local_waveform(local: np.ndarray) -> int:
    """Return a QRS fiducial within a local waveform window.

    For positive-dominant complexes this is the positive R peak. For strongly
    negative QS-like complexes, use the onset of the steep negative deflection
    rather than the late S/QS trough; this better matches annotation fiducials in
    paced/LBBB-like beats while leaving amplitude measurements to the Q/R/S code.
    """
    if local.size == 0:
        return 0

    dev = local.astype(float) - float(np.median(local))
    pos_pk = float(np.max(dev))
    neg_pk = float(-np.min(dev))
    if pos_pk >= neg_pk:
        return int(np.argmax(dev))
    if neg_pk <= max(pos_pk * _NEGATIVE_QS_DOMINANCE_RATIO, 1e-8):
        return int(np.argmin(dev))

    trough = int(np.argmin(dev))
    positive_peak = int(np.argmax(dev))
    if trough <= positive_peak:
        return trough
    if trough <= 1:
        return trough

    pre = dev[: trough + 1]
    slope = np.gradient(pre)
    min_slope = float(np.min(slope))
    if min_slope >= 0.0:
        return trough

    steep = np.where(slope <= _NEGATIVE_QS_SLOPE_FRACTION * min_slope)[0]
    if len(steep) == 0:
        return trough

    return int(steep[0])


def _energy_trace(ecg: np.ndarray, fs: int, leads: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    vm = make_vector_magnitude(ecg, leads)
    bp = bandpass_filter(vm, fs, 5.0, min(25.0, fs / 2 - 1))
    diff = np.diff(bp, prepend=bp[0])
    energy = diff**2
    win = max(1, int(0.120 * fs))
    mwa = np.convolve(energy, np.ones(win) / win, mode="same")
    return vm, mwa


def _standard_energy_threshold(mwa: np.ndarray) -> float:
    return float(max(np.percentile(mwa, 95) * 0.30, np.mean(mwa) + 0.5 * np.std(mwa)))


def _robust_energy_threshold(mwa: np.ndarray) -> float:
    median = float(np.median(mwa))
    mad = float(np.median(np.abs(mwa - median)))
    robust_sigma = 1.4826 * mad
    return float(max(np.percentile(mwa, 95) * 0.30, median + 8.0 * robust_sigma, 1e-12))


def _empty_result(
    threshold: float,
    leads: Sequence[int],
    *,
    fallback_used: bool = False,
    fallback_reason: Optional[str] = None,
) -> QRSDetectorResult:
    return QRSDetectorResult(
        r_locs=[],
        qrs_candidate_windows=[],
        qrs_detector_confidence=[],
        energy_threshold=float(threshold),
        detection_leads=[STANDARD_12_LEADS[i] for i in leads if 0 <= i < len(STANDARD_12_LEADS)],
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def _filter_edge_candidates(
    candidates: Sequence[QRSCandidateWindow],
    n_samples: int,
    fs: int,
) -> list[QRSCandidateWindow]:
    """Drop QRS candidates too close to record edges for complete measurement."""
    if n_samples <= 0 or fs <= 0:
        return list(candidates)
    margin = max(1, int(0.160 * fs))
    return [
        candidate
        for candidate in candidates
        if margin <= int(candidate.refined_r_index) <= int(n_samples) - margin
    ]


def _detect_with_threshold(
    ecg: np.ndarray,
    fs: int,
    leads: Sequence[int],
    threshold: float,
    *,
    fallback_used: bool = False,
    fallback_reason: Optional[str] = None,
) -> QRSDetectorResult:
    vm, mwa = _energy_trace(ecg, fs, leads)
    distance = max(1, int(0.22 * fs))
    peaks, props = find_peaks(mwa, distance=distance, height=threshold)
    if len(peaks) == 0:
        return _empty_result(
            threshold,
            leads,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    # Local R refinement using lead II if available, otherwise vector magnitude.
    ref = ecg[1] if ecg.shape[0] > 1 else vm
    peak_heights = props.get("peak_heights", np.zeros(len(peaks), dtype=float))
    candidates = []
    search = int(0.05 * fs)
    for p, detector_height in zip(peaks, peak_heights):
        lo = max(0, p - search)
        hi = min(len(ref), p + search + 1)
        local = ref[lo:hi]
        if local.size == 0:
            continue
        r = lo + _qrs_fiducial_from_local_waveform(local)
        confidence = _detector_confidence(float(detector_height), float(threshold), mwa)
        candidates.append(QRSCandidateWindow(
            detector_peak_index=int(p),
            search_start_index=int(lo),
            search_end_index=int(hi - 1),
            refined_r_index=int(r),
            detector_height=float(detector_height),
            confidence=confidence,
        ))

    if not candidates:
        return _empty_result(
            threshold,
            leads,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    candidates.sort(key=lambda item: item.refined_r_index)
    # De-duplicate too-close peaks by keeping the higher-confidence candidate.
    kept = [candidates[0]]
    refractory = int(0.20 * fs)
    for cand in candidates[1:]:
        if cand.refined_r_index - kept[-1].refined_r_index >= refractory:
            kept.append(cand)
        elif cand.confidence > kept[-1].confidence:
            kept[-1] = cand
    kept = _filter_edge_candidates(kept, ecg.shape[-1], fs)
    if not kept:
        return _empty_result(
            threshold,
            leads,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    return QRSDetectorResult(
        r_locs=[int(item.refined_r_index) for item in kept],
        qrs_candidate_windows=kept,
        qrs_detector_confidence=[float(item.confidence) for item in kept],
        energy_threshold=float(threshold),
        detection_leads=[STANDARD_12_LEADS[i] for i in leads if 0 <= i < len(STANDARD_12_LEADS)],
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def _should_use_robust_fallback(
    primary: QRSDetectorResult,
    robust: QRSDetectorResult,
    n_samples: int,
    fs: int,
) -> bool:
    if len(robust.r_locs) < 3:
        return False
    if len(robust.r_locs) <= len(primary.r_locs) + 2:
        return False

    duration_sec = float(n_samples) / float(fs) if fs > 0 else 0.0
    if duration_sec < 4.0:
        return False

    rr_ms = np.diff(np.asarray(robust.r_locs, dtype=float)) * 1000.0 / float(fs)
    if len(rr_ms) == 0:
        return False
    median_rr = float(np.median(rr_ms))
    if median_rr < 300.0 or median_rr > 2200.0:
        return False

    primary_rate = len(primary.r_locs) / max(duration_sec, 1e-6)
    robust_rate = len(robust.r_locs) / max(duration_sec, 1e-6)
    if robust_rate > 3.5:
        return False
    return len(primary.r_locs) < 4 or primary_rate < 0.55


def detect_qrs_multilead_with_meta(
    ecg: np.ndarray,
    fs: int,
    leads: Iterable[int] = DEFAULT_QRS_LEADS,
) -> QRSDetectorResult:
    leads = tuple(int(i) for i in leads)
    _, mwa = _energy_trace(ecg, fs, leads)
    primary = _detect_with_threshold(ecg, fs, leads, _standard_energy_threshold(mwa))

    robust_threshold = _robust_energy_threshold(mwa)
    if robust_threshold < primary.energy_threshold:
        robust = _detect_with_threshold(
            ecg,
            fs,
            leads,
            robust_threshold,
            fallback_used=True,
            fallback_reason="robust_threshold_sparse_primary",
        )
        if _should_use_robust_fallback(primary, robust, ecg.shape[-1], fs):
            return robust
    return primary


def detect_qrs_multilead(ecg: np.ndarray, fs: int, leads: Iterable[int] = DEFAULT_QRS_LEADS) -> np.ndarray:
    result = detect_qrs_multilead_with_meta(ecg, fs, leads=leads)
    return np.asarray(result.r_locs, dtype=int)
