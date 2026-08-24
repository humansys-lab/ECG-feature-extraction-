from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .models import STANDARD_12_LEADS


_MIN_DELAY_TO_CORRECT_MS = 8.0
_MAX_AUTOMATIC_DELAY_MS = 20.0
_MAX_DELAY_MAD_MS = 2.5
_MIN_CORRECTED_ENVELOPE_CORRELATION = 0.75
_MIN_CORRELATION_IMPROVEMENT = 0.08


def _finite_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n = min(x.size, y.size)
    if n < 4:
        return 0.0
    x = x[:n] - float(np.mean(x[:n]))
    y = y[:n] - float(np.mean(y[:n]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


def _shift_1d(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Remove a positive acquisition delay with edge-preserving interpolation."""
    values = np.asarray(signal, dtype=float)
    if values.size == 0 or abs(float(delay_samples)) < 1e-9:
        return values.copy()
    positions = np.arange(values.size, dtype=float)
    # A channel observed ``delay`` samples late is evaluated at t + delay to
    # move its contents left onto the common time base.
    source = positions + float(delay_samples)
    return np.interp(source, positions, values, left=values[0], right=values[-1])


def apply_channel_delay_compensation(
    ecg: np.ndarray,
    delays_samples: Dict[str, float],
    *,
    approved_leads: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    values = np.asarray(ecg, dtype=float)
    corrected = values.copy()
    approved = set(approved_leads or delays_samples)
    applied: List[str] = []
    for lead_index, lead in enumerate(STANDARD_12_LEADS):
        if lead_index >= corrected.shape[0] or lead not in approved:
            continue
        delay = float(delays_samples.get(lead, 0.0) or 0.0)
        if abs(delay) < 1e-9:
            continue
        corrected[lead_index] = _shift_1d(corrected[lead_index], delay)
        applied.append(lead)
    return corrected, applied


def _qrs_envelope(signal: np.ndarray, fs: int) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    derivative = np.gradient(values)
    sigma = max(0.5, 0.004 * float(fs))
    return gaussian_filter1d(np.abs(derivative), sigma=sigma, mode="nearest")


def _fractional_peak_offset(scores: Dict[int, float], center: int) -> float:
    left = float(scores.get(center - 1, scores.get(center, 0.0)))
    middle = float(scores.get(center, 0.0))
    right = float(scores.get(center + 1, scores.get(center, 0.0)))
    denom = left - 2.0 * middle + right
    if abs(denom) <= 1e-12:
        return float(center)
    fraction = 0.5 * (left - right) / denom
    return float(center + np.clip(fraction, -0.5, 0.5))


def _lead_delay_evidence(
    envelope: np.ndarray,
    reference: np.ndarray,
    r_locs: np.ndarray,
    fs: int,
) -> Dict[str, object]:
    search = max(2, int(round(0.080 * fs)))
    beat_offsets: List[float] = []
    for r_sample in np.asarray(r_locs, dtype=int):
        lo = max(0, int(r_sample) - search)
        hi = min(envelope.size, int(r_sample) + search + 1)
        if hi - lo < 5:
            continue
        lead_peak = lo + int(np.argmax(envelope[lo:hi]))
        ref_peak = lo + int(np.argmax(reference[lo:hi]))
        beat_offsets.append(float(lead_peak - ref_peak))

    if len(beat_offsets) < 3:
        return {
            "available": False,
            "delay_samples": 0.0,
            "delay_ms": 0.0,
            "mad_ms": None,
            "beat_count": len(beat_offsets),
            "automatic_compensation_approved": False,
        }

    center = float(np.median(beat_offsets))
    mad_samples = 1.4826 * float(
        np.median(np.abs(np.asarray(beat_offsets, dtype=float) - center))
    )
    integer_center = int(round(center))
    scores: Dict[int, float] = {}
    for candidate in range(integer_center - 1, integer_center + 2):
        compensated = _shift_1d(envelope, float(candidate))
        scores[candidate] = _finite_correlation(compensated, reference)
    best_integer = max(scores, key=scores.get)
    fractional = _fractional_peak_offset(scores, best_integer)
    raw_corr = _finite_correlation(envelope, reference)
    corrected_corr = _finite_correlation(
        _shift_1d(envelope, fractional),
        reference,
    )
    delay_ms = fractional * 1000.0 / float(fs)
    mad_ms = mad_samples * 1000.0 / float(fs)
    approved = bool(
        abs(delay_ms) >= _MIN_DELAY_TO_CORRECT_MS
        and abs(delay_ms) <= _MAX_AUTOMATIC_DELAY_MS
        and mad_ms <= _MAX_DELAY_MAD_MS
        and corrected_corr >= _MIN_CORRECTED_ENVELOPE_CORRELATION
        and corrected_corr - raw_corr >= _MIN_CORRELATION_IMPROVEMENT
    )
    return {
        "available": True,
        "delay_samples": float(fractional),
        "delay_ms": float(delay_ms),
        "mad_ms": float(mad_ms),
        "beat_count": len(beat_offsets),
        "raw_envelope_correlation": float(raw_corr),
        "corrected_envelope_correlation": float(corrected_corr),
        "correlation_improvement": float(corrected_corr - raw_corr),
        "automatic_compensation_approved": approved,
    }


def _duplicate_channel_pairs(ecg: np.ndarray) -> List[Dict[str, object]]:
    values = np.asarray(ecg, dtype=float)
    out: List[Dict[str, object]] = []
    for first in range(min(len(STANDARD_12_LEADS), values.shape[0])):
        x = values[first] - float(np.median(values[first]))
        x_scale = float(np.std(x))
        if x_scale <= 1e-8:
            continue
        for second in range(first + 1, min(len(STANDARD_12_LEADS), values.shape[0])):
            y = values[second] - float(np.median(values[second]))
            y_scale = float(np.std(y))
            if y_scale <= 1e-8:
                continue
            corr = _finite_correlation(x, y)
            alpha = float(np.dot(x, y) / (np.dot(y, y) + 1e-12))
            residual = x - alpha * y
            normalized_residual = float(np.sqrt(np.mean(residual * residual)) / x_scale)
            if abs(corr) >= 0.9995 and normalized_residual <= 0.01:
                out.append(
                    {
                        "lead_a": STANDARD_12_LEADS[first],
                        "lead_b": STANDARD_12_LEADS[second],
                        "correlation": corr,
                        "normalized_residual": normalized_residual,
                    }
                )
    return out


def _gain_and_filter_outliers(ecg: np.ndarray, fs: int) -> Tuple[List[str], List[str]]:
    values = np.asarray(ecg, dtype=float)
    amplitudes: Dict[str, float] = {}
    hf_ratios: Dict[str, float] = {}
    for index, lead in enumerate(STANDARD_12_LEADS):
        if index >= values.shape[0]:
            continue
        signal = values[index]
        amplitudes[lead] = float(np.percentile(signal, 99) - np.percentile(signal, 1))
        derivative = np.diff(signal)
        fast = np.diff(derivative) if derivative.size > 1 else derivative
        hf_ratios[lead] = float(
            np.sqrt(np.mean(fast * fast))
            / (np.sqrt(np.mean(derivative * derivative)) + 1e-12)
        )

    gain_outliers: List[str] = []
    filter_outliers: List[str] = []
    for group in (
        list(STANDARD_12_LEADS[:6]),
        ["V1", "V2", "V3"],
        ["V4", "V5", "V6"],
    ):
        group_amplitudes = [
            amplitudes[lead]
            for lead in group
            if amplitudes.get(lead, 0.0) > 1e-6
        ]
        group_hf = [
            hf_ratios[lead]
            for lead in group
            if np.isfinite(hf_ratios.get(lead, np.nan))
        ]
        if not group_amplitudes:
            continue
        amp_center = float(np.median(group_amplitudes))
        hf_center = float(np.median(group_hf)) if group_hf else 0.0
        for lead in group:
            amplitude = amplitudes.get(lead)
            if amplitude is not None and amp_center > 1e-6:
                ratio = amplitude / amp_center
                if ratio > 8.0 or ratio < 0.08:
                    gain_outliers.append(lead)
            hf = hf_ratios.get(lead)
            if hf is not None and hf_center > 1e-6:
                ratio = hf / hf_center
                if ratio > 3.5 or ratio < 0.25:
                    filter_outliers.append(lead)
    return sorted(set(gain_outliers)), sorted(set(filter_outliers))


def assess_acquisition_chain(
    ecg: np.ndarray,
    fs: int,
    r_locs: np.ndarray,
    quality: Optional[Dict[str, object]] = None,
    *,
    limb_lead_consistency: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Audit synchronization/configuration before authoritative P fusion.

    Delay correction is intentionally conservative: a lead must show a stable
    fixed QRS-envelope lag and a material correlation improvement after the
    proposed shift. This avoids erasing real inter-lead activation timing.
    """

    values = np.asarray(ecg, dtype=float)
    if fs <= 0 or values.ndim != 2:
        return {
            "available": False,
            "fusion_allowed": False,
            "reject_reasons": ["LEAD_CONFIGURATION_INVALID"],
            "excluded_leads": list(STANDARD_12_LEADS),
        }

    envelopes = np.vstack(
        [_qrs_envelope(values[index], fs) for index in range(values.shape[0])]
    )
    reliable_indices = [
        index
        for index, lead in enumerate(STANDARD_12_LEADS)
        if index < envelopes.shape[0]
        and (
            quality is None
            or bool(getattr(quality.get(lead), "reliable_for_qrs", True))
        )
    ]
    if not reliable_indices:
        reliable_indices = list(range(envelopes.shape[0]))
    normalized = []
    for index in reliable_indices:
        env = envelopes[index]
        scale = float(np.percentile(env, 95))
        normalized.append(env / max(scale, 1e-9))
    reference = np.median(np.vstack(normalized), axis=0)

    delay_evidence: Dict[str, Dict[str, object]] = {}
    approved_delays: Dict[str, float] = {}
    suspicious_delay_leads: List[str] = []
    for index, lead in enumerate(STANDARD_12_LEADS):
        if index >= envelopes.shape[0]:
            continue
        evidence = _lead_delay_evidence(
            envelopes[index],
            reference,
            np.asarray(r_locs, dtype=int),
            fs,
        )
        delay_evidence[lead] = evidence
        delay_ms = abs(float(evidence.get("delay_ms") or 0.0))
        mad_value = evidence.get("mad_ms")
        # A perfectly stable fixed channel delay legitimately has zero MAD.
        # Do not use truthiness here: treating 0.0 as a missing value would
        # turn it into infinity and let the most repeatable large delays bypass
        # the desynchronization guard.
        mad_ms = float(mad_value) if mad_value is not None else np.inf
        if bool(evidence.get("automatic_compensation_approved")):
            approved_delays[lead] = float(evidence["delay_samples"])
        elif (
            delay_ms > _MAX_AUTOMATIC_DELAY_MS
            and mad_ms <= _MAX_DELAY_MAD_MS
            and float(evidence.get("corrected_envelope_correlation") or 0.0)
            >= _MIN_CORRECTED_ENVELOPE_CORRELATION
            and float(evidence.get("correlation_improvement") or 0.0)
            >= _MIN_CORRELATION_IMPROVEMENT
        ):
            suspicious_delay_leads.append(lead)

    duplicates = _duplicate_channel_pairs(values)
    gain_outliers, filter_outliers = _gain_and_filter_outliers(values, fs)
    severe_limb_error = bool(
        (limb_lead_consistency or {}).get("severely_inconsistent", False)
    )
    configuration_invalid = bool(severe_limb_error or duplicates)
    channel_desynchronized = bool(len(suspicious_delay_leads) >= 1)
    excluded_leads = sorted(
        set(suspicious_delay_leads) | set(gain_outliers) | set(filter_outliers)
    )
    reject_reasons: List[str] = []
    if channel_desynchronized:
        reject_reasons.append("CHANNEL_DESYNCHRONIZED")
    if configuration_invalid:
        reject_reasons.append("LEAD_CONFIGURATION_INVALID")

    return {
        "available": True,
        "method": "qrs_envelope_fixed_delay_with_correlation_guard",
        "channel_delay_estimates": delay_evidence,
        "approved_delay_samples": approved_delays,
        "approved_delay_ms": {
            lead: float(delay) * 1000.0 / float(fs)
            for lead, delay in approved_delays.items()
        },
        "automatic_compensation_leads": sorted(approved_delays),
        "suspicious_delay_leads": sorted(suspicious_delay_leads),
        "duplicate_channel_pairs": duplicates,
        "gain_outlier_leads": gain_outliers,
        "filter_mismatch_leads": filter_outliers,
        "limb_lead_consistency": dict(limb_lead_consistency or {}),
        "configuration_invalid": configuration_invalid,
        "channel_desynchronized": channel_desynchronized,
        "excluded_leads": excluded_leads,
        "fusion_allowed": not bool(reject_reasons),
        "reject_reasons": reject_reasons,
    }
