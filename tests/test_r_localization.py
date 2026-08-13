from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from feature_extraction.ecgfeat.r_localization import (
    apply_hybrid_r_localization,
    localize_r_prominence,
)


def _triangular_peak(
    signal: np.ndarray,
    center: int,
    amplitude: float,
) -> None:
    signal[center - 2 : center + 3] += amplitude * np.asarray(
        [0.15, 0.55, 1.0, 0.55, 0.15]
    )


def test_localize_r_prominence_selects_prominent_positive_peak() -> None:
    signal = np.zeros(500)
    _triangular_peak(signal, 246, 0.25)
    _triangular_peak(signal, 260, 1.00)

    result = localize_r_prominence(
        signal,
        fs=500,
        global_r=250,
        qrs_onset=225,
        qrs_offset=280,
        lead_fiducial=246,
    )

    assert result.index == 260
    assert result.method == "hybrid_prominence"
    assert result.polarity == 1
    assert result.confidence > 0


def test_localize_r_prominence_prefers_positive_peak_like_neurokit() -> None:
    signal = np.zeros(500)
    _triangular_peak(signal, 245, 0.20)
    _triangular_peak(signal, 257, -1.00)

    result = localize_r_prominence(
        signal,
        fs=500,
        global_r=250,
        qrs_onset=225,
        qrs_offset=280,
        lead_fiducial=248,
    )

    assert result.index == 245
    assert result.polarity == 1


def test_localize_r_prominence_ignores_peak_outside_qrs_window() -> None:
    signal = np.zeros(500)
    _triangular_peak(signal, 252, 0.60)
    _triangular_peak(signal, 330, 3.00)

    result = localize_r_prominence(
        signal,
        fs=500,
        global_r=250,
        qrs_onset=225,
        qrs_offset=280,
        lead_fiducial=250,
    )

    assert result.index == 252


def test_apply_hybrid_localization_does_not_replace_existing_fiducial() -> None:
    ecg = np.zeros((12, 500))
    _triangular_peak(ecg[1], 254, 1.00)
    feature = SimpleNamespace(
        lead="II",
        beat_id=0,
        qrs=SimpleNamespace(onset=225, peak=248, offset=280),
        r_localized_index=None,
        r_localization_method=None,
        r_localization_confidence=None,
        r_localization_prominence_mv=None,
        r_localization_polarity=None,
    )

    apply_hybrid_r_localization(
        [feature],
        localization_ecg=ecg,
        r_locs=np.asarray([250]),
        fs=500,
    )

    assert feature.qrs.peak == 248
    assert feature.r_localized_index == 254
