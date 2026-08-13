from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import find_peaks

from .models import LeadBeatFeatures, STANDARD_12_LEADS


@dataclass(frozen=True)
class RLocalization:
    index: int
    method: str
    confidence: float
    prominence_mv: float | None
    polarity: int


def localize_r_prominence(
    signal: np.ndarray,
    *,
    fs: int,
    global_r: int,
    qrs_onset: int | None,
    qrs_offset: int | None,
    lead_fiducial: int | None,
    global_radius_ms: float = 90.0,
    fiducial_radius_ms: float = 35.0,
) -> RLocalization:
    """Localize a lead-specific R/QRS peak without changing beat detection.

    The multilead ``global_r`` determines beat presence. A tight per-lead
    window is then formed from the final QRS bounds and the existing local
    fiducial. Within it, the most prominent positive peak is selected, matching
    NeuroKit2's final localization convention. A negative peak is used only
    when no positive local maximum exists.

    This result is intentionally a parallel annotation. It does not replace
    ``global_r``, ``qrs.peak``, QRS boundaries, or clinical amplitude fields.
    """

    values = np.asarray(signal, dtype=float)
    n_samples = int(values.size)
    fallback = (
        int(lead_fiducial)
        if lead_fiducial is not None
        else int(global_r)
    )
    fallback = int(np.clip(fallback, 0, max(0, n_samples - 1)))
    if fs <= 0 or n_samples < 3:
        return RLocalization(fallback, "fiducial_fallback", 0.0, None, 0)

    global_radius = max(1, int(round(global_radius_ms * fs / 1000.0)))
    fiducial_radius = max(1, int(round(fiducial_radius_ms * fs / 1000.0)))
    center = fallback
    lo = max(0, int(global_r) - global_radius, center - fiducial_radius)
    hi = min(
        n_samples - 1,
        int(global_r) + global_radius,
        center + fiducial_radius,
    )
    qrs_margin = max(1, int(round(0.006 * fs)))
    if qrs_onset is not None:
        lo = max(lo, int(qrs_onset) - qrs_margin)
    if qrs_offset is not None:
        hi = min(hi, int(qrs_offset) + qrs_margin)
    if hi - lo < 2:
        return RLocalization(fallback, "fiducial_fallback", 0.0, None, 0)

    segment = values[lo : hi + 1]
    if not np.all(np.isfinite(segment)):
        return RLocalization(fallback, "fiducial_fallback", 0.0, None, 0)

    baseline = float(np.median(np.concatenate((segment[:2], segment[-2:]))))
    centered = segment - baseline
    positive_extent = float(max(0.0, np.max(centered)))
    negative_extent = float(max(0.0, -np.min(centered)))

    positive_peaks, positive_props = find_peaks(
        centered,
        prominence=(None, None),
    )
    negative_peaks, negative_props = find_peaks(
        -centered,
        prominence=(None, None),
    )

    candidates = positive_peaks
    prominences = positive_props.get("prominences", np.asarray([]))
    polarity = 1

    if len(candidates) == 0:
        candidates = negative_peaks
        prominences = negative_props.get("prominences", np.asarray([]))
        polarity = -1
    if len(candidates) == 0 or len(prominences) != len(candidates):
        return RLocalization(fallback, "fiducial_fallback", 0.0, None, 0)

    candidate_indices = lo + np.asarray(candidates, dtype=int)
    selected_position = int(np.argmax(np.asarray(prominences, dtype=float)))
    selected = int(candidate_indices[selected_position])
    prominence = float(prominences[selected_position])

    dynamic_range = max(
        positive_extent + negative_extent,
        float(np.ptp(centered)),
        1e-9,
    )
    confidence = float(np.clip(prominence / dynamic_range, 0.0, 1.0))
    return RLocalization(
        selected,
        "hybrid_prominence",
        confidence,
        prominence,
        polarity,
    )


def apply_hybrid_r_localization(
    beat_features: Sequence[LeadBeatFeatures],
    *,
    localization_ecg: np.ndarray,
    r_locs: np.ndarray,
    fs: int,
) -> None:
    """Attach parallel, lead-specific prominence locations in place."""

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
        result = localize_r_prominence(
            ecg[lead_index],
            fs=fs,
            global_r=int(r_values[beat_id]),
            qrs_onset=feature.qrs.onset,
            qrs_offset=feature.qrs.offset,
            lead_fiducial=feature.qrs.peak,
        )
        feature.r_localized_index = result.index
        feature.r_localization_method = result.method
        feature.r_localization_confidence = result.confidence
        feature.r_localization_prominence_mv = result.prominence_mv
        feature.r_localization_polarity = result.polarity
