from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.signal import find_peaks

from .models import LeadBeatFeatures, STANDARD_12_LEADS


@dataclass(frozen=True)
class WaveLocalization:
    index: int | None
    onset: int | None
    offset: int | None
    method: str
    confidence: float
    prominence_mv: float | None
    polarity: int
    morphology: str
    secondary_index: int | None = None
    candidate_count: int = 0


@dataclass(frozen=True)
class SLocalization:
    peak_index: int | None
    amplitude_index: int | None
    prime_index: int | None
    qs_nadir_index: int | None
    method: str
    confidence: float
    prominence_mv: float | None
    morphology: str


@dataclass(frozen=True)
class _Candidate:
    index: int
    onset: int
    offset: int
    prominence: float
    polarity: int


def _finite_index(value: Any, n_samples: int) -> int | None:
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return index if 0 <= index < n_samples else None


def _smooth_signal(signal: np.ndarray, fs: int, width_ms: float = 8.0) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    width = max(1, int(round(width_ms * fs / 1000.0)))
    if width <= 1:
        return values.copy()
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def _noise_sigma(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=float)
    if values.size < 3:
        return 0.0
    differences = np.diff(values)
    center = float(np.median(differences))
    mad = float(np.median(np.abs(differences - center)))
    return max(0.0, 1.4826 * mad / np.sqrt(2.0))


def _baseline(signal: np.ndarray, r_index: int, fs: int) -> float:
    lo = max(0, int(r_index) - int(round(0.220 * fs)))
    hi = max(lo + 1, int(r_index) - int(round(0.080 * fs)))
    hi = min(len(signal), hi)
    if hi > lo:
        return float(np.median(signal[lo:hi]))
    return 0.0


def _extrema_candidates(
    signal: np.ndarray,
    *,
    lo: int,
    hi: int,
) -> list[_Candidate]:
    if lo < 0 or hi >= len(signal) or hi - lo < 2:
        return []
    segment = np.asarray(signal[lo : hi + 1], dtype=float)
    if segment.size < 3 or not np.all(np.isfinite(segment)):
        return []

    candidates: list[_Candidate] = []
    for polarity, values in ((1, segment), (-1, -segment)):
        peaks, properties = find_peaks(
            values,
            prominence=(None, None),
        )
        prominences = properties.get("prominences", np.asarray([], dtype=float))
        left_bases = properties.get("left_bases", np.asarray([], dtype=int))
        right_bases = properties.get("right_bases", np.asarray([], dtype=int))
        for position, peak in enumerate(peaks):
            if (
                position >= len(prominences)
                or position >= len(left_bases)
                or position >= len(right_bases)
            ):
                continue
            candidates.append(
                _Candidate(
                    index=int(lo + int(peak)),
                    onset=int(lo + int(left_bases[position])),
                    offset=int(lo + int(right_bases[position])),
                    prominence=float(prominences[position]),
                    polarity=int(polarity),
                )
            )
    return candidates


def _candidate_confidence(
    candidate: _Candidate,
    *,
    signal: np.ndarray,
    lo: int,
    hi: int,
    noise: float,
) -> float:
    dynamic_range = max(float(np.ptp(signal[lo : hi + 1])), 1e-9)
    range_score = float(np.clip(candidate.prominence / dynamic_range, 0.0, 1.0))
    noise_score = float(
        np.clip(
            candidate.prominence / max(candidate.prominence + 3.0 * noise, 1e-9),
            0.0,
            1.0,
        )
    )
    return float(np.clip(0.45 * range_score + 0.55 * noise_score, 0.0, 1.0))


def _choose_candidate(
    candidates: Sequence[_Candidate],
    *,
    prior: int | None,
) -> _Candidate | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.prominence,
            -abs(item.index - prior) if prior is not None else 0,
        ),
    )


def _secondary_opposite_candidate(
    candidates: Sequence[_Candidate],
    *,
    primary: _Candidate,
    fs: int,
    minimum_ratio: float,
    maximum_distance_ms: float = 160.0,
) -> _Candidate | None:
    radius = max(1, int(round(maximum_distance_ms * fs / 1000.0)))
    eligible = [
        item
        for item in candidates
        if item.polarity != primary.polarity
        and abs(item.index - primary.index) <= radius
        and item.prominence >= minimum_ratio * primary.prominence
    ]
    return max(eligible, key=lambda item: item.prominence) if eligible else None


def _biphasic_morphology(primary: _Candidate, secondary: _Candidate | None) -> str:
    if secondary is None:
        return "positive" if primary.polarity > 0 else "negative"
    first, second = sorted((primary, secondary), key=lambda item: item.index)
    return (
        "biphasic-positive-negative"
        if first.polarity > 0 and second.polarity < 0
        else "biphasic-negative-positive"
    )


def localize_t_prominence(
    signal: np.ndarray,
    *,
    fs: int,
    r_index: int,
    next_r_index: int | None,
    qrs_offset: int | None,
    t_onset: int | None,
    t_peak: int | None,
    t_offset: int | None,
) -> WaveLocalization:
    """Localize positive, negative, or biphasic T waves as a side annotation."""

    values = np.asarray(signal, dtype=float)
    n_samples = len(values)
    if fs <= 0 or n_samples < 3:
        return WaveLocalization(None, None, None, "unavailable", 0.0, None, 0, "uncertain")

    qrs_off = _finite_index(qrs_offset, n_samples)
    prior = _finite_index(t_peak, n_samples)
    existing_on = _finite_index(t_onset, n_samples)
    existing_off = _finite_index(t_offset, n_samples)
    margin = max(1, int(round(0.030 * fs)))

    if existing_on is not None and existing_off is not None and existing_off > existing_on:
        lo = max(0, existing_on - margin)
        hi = min(n_samples - 1, existing_off + margin)
    elif prior is not None:
        lo = max(0, (qrs_off if qrs_off is not None else r_index) + int(round(0.020 * fs)), prior - int(round(0.120 * fs)))
        hi = min(n_samples - 1, prior + int(round(0.180 * fs)))
    else:
        lo = max(0, (qrs_off if qrs_off is not None else r_index) + int(round(0.040 * fs)))
        hi = min(n_samples - 1, int(r_index) + int(round(0.600 * fs)))

    if next_r_index is not None:
        hi = min(hi, int(next_r_index) - int(round(0.080 * fs)))
    if hi - lo < max(3, int(round(0.040 * fs))):
        return WaveLocalization(prior, existing_on, existing_off, "native_fallback", 0.0, None, 0, "uncertain")

    smooth = _smooth_signal(values, fs)
    candidates = _extrema_candidates(smooth, lo=lo, hi=hi)
    noise = _noise_sigma(smooth[lo : hi + 1])
    minimum_prominence = max(0.010, 2.5 * noise)
    reference = _baseline(smooth, int(r_index), fs)
    significant = [
        item
        for item in candidates
        if item.prominence >= minimum_prominence
        and item.polarity * float(smooth[item.index] - reference) >= 0.010
    ]
    primary = _choose_candidate(significant, prior=prior)
    if primary is None:
        return WaveLocalization(prior, existing_on, existing_off, "native_fallback", 0.0, None, 0, "uncertain")

    secondary = _secondary_opposite_candidate(
        significant,
        primary=primary,
        fs=fs,
        minimum_ratio=0.30,
    )
    onset = primary.onset
    offset = primary.offset
    if secondary is not None:
        onset = min(onset, secondary.onset)
        offset = max(offset, secondary.offset)
    confidence = _candidate_confidence(
        primary,
        signal=smooth,
        lo=lo,
        hi=hi,
        noise=noise,
    )
    return WaveLocalization(
        index=primary.index,
        onset=onset,
        offset=offset,
        method="hybrid_bipolar_prominence",
        confidence=confidence,
        prominence_mv=primary.prominence,
        polarity=primary.polarity,
        morphology=_biphasic_morphology(primary, secondary),
        secondary_index=None if secondary is None else secondary.index,
        candidate_count=len(significant),
    )


def localize_p_prominence(
    signal: np.ndarray,
    *,
    fs: int,
    r_index: int,
    previous_r_index: int | None,
    qrs_onset: int | None,
    qrs_offset: int | None,
    p_onset: int | None,
    p_peak: int | None,
    p_offset: int | None,
    retrograde: bool = False,
) -> WaveLocalization:
    """Localize bipolar P candidates while allowing an explicit no-P result."""

    values = np.asarray(signal, dtype=float)
    n_samples = len(values)
    if fs <= 0 or n_samples < 3:
        return WaveLocalization(None, None, None, "unavailable", 0.0, None, 0, "uncertain")

    prior = _finite_index(p_peak, n_samples)
    existing_on = _finite_index(p_onset, n_samples)
    existing_off = _finite_index(p_offset, n_samples)
    qrs_on = _finite_index(qrs_onset, n_samples)
    qrs_off = _finite_index(qrs_offset, n_samples)
    margin = max(1, int(round(0.020 * fs)))

    if existing_on is not None and existing_off is not None and existing_off > existing_on:
        lo = max(0, existing_on - margin)
        hi = min(n_samples - 1, existing_off + margin)
    elif retrograde and qrs_off is not None:
        lo = min(n_samples - 1, qrs_off + int(round(0.030 * fs)))
        hi = min(n_samples - 1, qrs_off + int(round(0.250 * fs)))
    else:
        qrs_anchor = qrs_on if qrs_on is not None else int(r_index)
        lo = max(0, qrs_anchor - int(round(0.350 * fs)))
        hi = min(n_samples - 1, qrs_anchor - int(round(0.030 * fs)))
        if previous_r_index is not None:
            lo = max(lo, int(previous_r_index) + int(round(0.200 * fs)))

    if hi - lo < max(3, int(round(0.025 * fs))):
        return WaveLocalization(prior, existing_on, existing_off, "native_fallback", 0.0, None, 0, "uncertain")

    smooth = _smooth_signal(values, fs, width_ms=10.0)
    candidates = _extrema_candidates(smooth, lo=lo, hi=hi)
    noise = _noise_sigma(smooth[lo : hi + 1])
    minimum_prominence = max(0.008, 3.0 * noise)
    reference = _baseline(smooth, int(r_index), fs)
    significant = [
        item
        for item in candidates
        if item.prominence >= minimum_prominence
        and item.polarity * float(smooth[item.index] - reference) >= 0.008
    ]
    near_prior = (
        [
            item
            for item in significant
            if abs(item.index - prior) <= int(round(0.040 * fs))
        ]
        if prior is not None
        else []
    )
    primary = _choose_candidate(near_prior or significant, prior=prior)
    if primary is None:
        if prior is not None:
            prior_polarity = int(np.sign(float(smooth[prior] - reference)))
            return WaveLocalization(
                prior,
                existing_on,
                existing_off,
                "native_peak_fallback",
                0.0,
                None,
                prior_polarity,
                "uncertain",
                candidate_count=0,
            )
        return WaveLocalization(
            None,
            None,
            None,
            "no_significant_candidate",
            0.0,
            None,
            0,
            "absent",
            candidate_count=0,
        )

    secondary = _secondary_opposite_candidate(
        significant,
        primary=primary,
        fs=fs,
        minimum_ratio=0.45,
        maximum_distance_ms=120.0,
    )
    onset = primary.onset
    offset = primary.offset
    if secondary is not None:
        onset = min(onset, secondary.onset)
        offset = max(offset, secondary.offset)
    confidence = _candidate_confidence(
        primary,
        signal=smooth,
        lo=lo,
        hi=hi,
        noise=noise,
    )
    localized_index = primary.index
    method = "hybrid_bipolar_prominence"
    polarity = primary.polarity
    if prior is not None:
        # Native P timing is already strong on LUDB. Use bipolar extrema for
        # polarity/morphology, but do not move an existing native P marker.
        localized_index = prior
        method = "native_peak_bipolar_morphology"
        prior_deflection = float(smooth[prior] - reference)
        if abs(prior_deflection) >= 0.008:
            polarity = 1 if prior_deflection > 0.0 else -1
    return WaveLocalization(
        index=localized_index,
        onset=existing_on if prior is not None and existing_on is not None else onset,
        offset=existing_off if prior is not None and existing_off is not None else offset,
        method=method,
        confidence=confidence,
        prominence_mv=primary.prominence,
        polarity=polarity,
        morphology=_biphasic_morphology(primary, secondary),
        secondary_index=None if secondary is None else secondary.index,
        candidate_count=len(significant),
    )


def localize_s_morphology(
    signal: np.ndarray,
    *,
    fs: int,
    qrs_onset: int | None,
    qrs_offset: int | None,
    r_peak_index: int | None,
) -> SLocalization:
    """Separate the first morphological S from the deepest S-amplitude point."""

    values = np.asarray(signal, dtype=float)
    n_samples = len(values)
    qrs_on = _finite_index(qrs_onset, n_samples)
    qrs_off = _finite_index(qrs_offset, n_samples)
    if fs <= 0 or qrs_on is None or qrs_off is None or qrs_off - qrs_on < 3:
        return SLocalization(None, None, None, None, "unavailable", 0.0, None, "uncertain")

    smooth = _smooth_signal(values, fs, width_ms=6.0)
    reference = float(np.median(values[max(0, qrs_on - int(round(0.040 * fs))) : qrs_on + 1]))
    qrs = smooth[qrs_on : qrs_off + 1] - reference
    qrs_range = max(float(np.ptp(qrs)), 1e-9)
    r_peak = _finite_index(r_peak_index, n_samples)
    if r_peak is None or not (qrs_on <= r_peak <= qrs_off):
        r_peak = qrs_on + int(np.argmax(qrs))
    positive_r = float(smooth[r_peak] - reference)
    positive_threshold = max(0.015, 0.08 * qrs_range)
    if positive_r < positive_threshold:
        nadir = qrs_on + int(np.argmin(qrs))
        return SLocalization(
            None,
            None,
            None,
            nadir,
            "hybrid_s_morphology",
            float(np.clip(abs(float(qrs[nadir - qrs_on])) / qrs_range, 0.0, 1.0)),
            None,
            "QS",
        )

    lo = min(qrs_off, r_peak + 1)
    if qrs_off - lo < 2:
        return SLocalization(None, None, None, None, "hybrid_s_morphology", 0.0, None, "no-S")
    post = smooth[lo : qrs_off + 1] - reference
    minima, properties = find_peaks(-post, prominence=(None, None))
    prominences = properties.get("prominences", np.asarray([], dtype=float))
    minimum_prominence = max(0.008, 0.05 * qrs_range)
    significant: list[tuple[int, float]] = []
    for position, local_index in enumerate(minima):
        if position >= len(prominences):
            continue
        index = lo + int(local_index)
        prominence = float(prominences[position])
        if prominence >= minimum_prominence and float(smooth[index] - reference) < -0.008:
            significant.append((index, prominence))

    deepest = lo + int(np.argmin(post))
    if float(smooth[deepest] - reference) >= -0.008 or not significant:
        return SLocalization(None, None, None, None, "hybrid_s_morphology", 0.0, None, "no-S")

    first_index, first_prominence = min(significant, key=lambda item: item[0])
    amplitude_index = min(significant, key=lambda item: float(smooth[item[0]]))[0]
    prime_index = None
    for candidate_index, _candidate_prominence in significant:
        if candidate_index <= first_index:
            continue
        between = smooth[first_index : candidate_index + 1] - reference
        if between.size and float(np.max(between)) >= positive_threshold:
            prime_index = candidate_index
            break
    morphology = "S-prime" if prime_index is not None else "S"
    confidence = float(np.clip(first_prominence / qrs_range, 0.0, 1.0))
    return SLocalization(
        first_index,
        amplitude_index,
        prime_index,
        None,
        "hybrid_s_morphology",
        confidence,
        first_prominence,
        morphology,
    )


def apply_hybrid_wave_localization(
    beat_features: Sequence[LeadBeatFeatures],
    *,
    localization_ecg: np.ndarray,
    r_locs: np.ndarray,
    fs: int,
) -> None:
    """Attach P/T/S localization side fields without replacing native fields."""

    ecg = np.asarray(localization_ecg, dtype=float)
    if ecg.ndim != 2 or fs <= 0:
        return
    lead_indices = {
        lead: index
        for index, lead in enumerate(STANDARD_12_LEADS)
        if index < ecg.shape[0]
    }
    r_values = np.asarray(r_locs, dtype=int)
    for feature in beat_features:
        lead_index = lead_indices.get(feature.lead)
        beat_id = int(feature.beat_id)
        if lead_index is None or beat_id < 0 or beat_id >= len(r_values):
            continue
        signal = ecg[lead_index]
        current_r = int(r_values[beat_id])
        previous_r = int(r_values[beat_id - 1]) if beat_id > 0 else None
        next_r = int(r_values[beat_id + 1]) if beat_id + 1 < len(r_values) else None

        t_result = localize_t_prominence(
            signal,
            fs=fs,
            r_index=current_r,
            next_r_index=next_r,
            qrs_offset=feature.qrs.offset,
            t_onset=feature.t.onset,
            t_peak=feature.t.peak,
            t_offset=feature.t.offset,
        )
        feature.t_localized_index = t_result.index
        feature.t_localized_onset = t_result.onset
        feature.t_localized_offset = t_result.offset
        feature.t_localization_method = t_result.method
        feature.t_localization_confidence = t_result.confidence
        feature.t_localization_prominence_mv = t_result.prominence_mv
        feature.t_localization_polarity = t_result.polarity
        feature.t_localization_morphology = t_result.morphology
        feature.t_localization_secondary_index = t_result.secondary_index

        flags = getattr(feature, "flags", []) or []
        retrograde = (
            "retrograde_p" in flags
            or (
                feature.p.peak is not None
                and feature.qrs.offset is not None
                and int(feature.p.peak) > int(feature.qrs.offset)
            )
        )
        p_result = localize_p_prominence(
            signal,
            fs=fs,
            r_index=current_r,
            previous_r_index=previous_r,
            qrs_onset=feature.qrs.onset,
            qrs_offset=feature.qrs.offset,
            p_onset=feature.p.onset,
            p_peak=feature.p.peak,
            p_offset=feature.p.offset,
            retrograde=retrograde,
        )
        feature.p_localized_index = p_result.index
        feature.p_localized_onset = p_result.onset
        feature.p_localized_offset = p_result.offset
        feature.p_localization_method = p_result.method
        feature.p_localization_confidence = p_result.confidence
        feature.p_localization_prominence_mv = p_result.prominence_mv
        feature.p_localization_polarity = p_result.polarity
        feature.p_localization_morphology = p_result.morphology
        feature.p_localization_secondary_index = p_result.secondary_index
        feature.p_localization_candidate_count = p_result.candidate_count
        if feature.p.peak is not None:
            feature.p_localization_absent_probability = 0.0
        elif p_result.index is None:
            feature.p_localization_absent_probability = 1.0
        else:
            feature.p_localization_absent_probability = float(
                1.0 - p_result.confidence
            )
        feature.p_localization_retrograde = bool(retrograde and p_result.index is not None)

        s_result = localize_s_morphology(
            signal,
            fs=fs,
            qrs_onset=feature.qrs.onset,
            qrs_offset=feature.qrs.offset,
            r_peak_index=getattr(feature, "r_peak_index", None),
        )
        feature.s_peak_index = s_result.peak_index
        feature.s_amplitude_index = s_result.amplitude_index
        feature.s_prime_peak_index = s_result.prime_index
        feature.qs_nadir_index = s_result.qs_nadir_index
        feature.s_localization_method = s_result.method
        feature.s_localization_confidence = s_result.confidence
        feature.s_localization_prominence_mv = s_result.prominence_mv
        feature.s_localization_morphology = s_result.morphology
