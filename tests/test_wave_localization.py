from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from feature_extraction.ecgfeat.wave_localization import (
    apply_hybrid_wave_localization,
    localize_p_prominence,
    localize_s_morphology,
    localize_t_prominence,
)


def _triangular_peak(signal: np.ndarray, center: int, amplitude: float) -> None:
    signal[center - 2 : center + 3] += amplitude * np.asarray(
        [0.15, 0.55, 1.0, 0.55, 0.15]
    )


def test_t_localizer_detects_inverted_t_wave() -> None:
    signal = np.zeros(1000)
    _triangular_peak(signal, 620, -0.50)

    result = localize_t_prominence(
        signal,
        fs=500,
        r_index=400,
        next_r_index=850,
        qrs_offset=450,
        t_onset=560,
        t_peak=610,
        t_offset=680,
    )

    assert result.index == 620
    assert result.polarity == -1
    assert result.morphology == "negative"


def test_t_localizer_preserves_biphasic_order() -> None:
    signal = np.zeros(1000)
    _triangular_peak(signal, 600, 0.35)
    _triangular_peak(signal, 640, -0.25)

    result = localize_t_prominence(
        signal,
        fs=500,
        r_index=400,
        next_r_index=850,
        qrs_offset=450,
        t_onset=560,
        t_peak=600,
        t_offset=680,
    )

    assert result.morphology == "biphasic-positive-negative"
    assert result.secondary_index is not None


def test_p_localizer_annotates_inverted_p_without_moving_native_peak() -> None:
    signal = np.zeros(1000)
    _triangular_peak(signal, 315, -0.12)

    result = localize_p_prominence(
        signal,
        fs=500,
        r_index=400,
        previous_r_index=None,
        qrs_onset=370,
        qrs_offset=440,
        p_onset=285,
        p_peak=310,
        p_offset=340,
    )

    assert result.index == 310
    assert result.method == "native_peak_bipolar_morphology"
    assert result.polarity == -1
    assert result.morphology == "negative"


def test_p_localizer_allows_absent_p_wave() -> None:
    result = localize_p_prominence(
        np.zeros(1000),
        fs=500,
        r_index=400,
        previous_r_index=None,
        qrs_onset=370,
        qrs_offset=440,
        p_onset=None,
        p_peak=None,
        p_offset=None,
    )

    assert result.index is None
    assert result.morphology == "absent"


def test_s_localizer_separates_first_s_and_deepest_amplitude() -> None:
    signal = np.zeros(1000)
    _triangular_peak(signal, 400, 1.00)
    _triangular_peak(signal, 420, -0.35)
    _triangular_peak(signal, 435, 0.45)
    _triangular_peak(signal, 450, -0.70)

    result = localize_s_morphology(
        signal,
        fs=500,
        qrs_onset=380,
        qrs_offset=470,
        r_peak_index=400,
    )

    assert result.peak_index == 420
    assert result.amplitude_index == 450
    assert result.prime_index == 450
    assert result.morphology == "S-prime"


def test_s_localizer_classifies_qs_without_forcing_s() -> None:
    signal = np.zeros(1000)
    _triangular_peak(signal, 410, -0.80)

    result = localize_s_morphology(
        signal,
        fs=500,
        qrs_onset=380,
        qrs_offset=450,
        r_peak_index=380,
    )

    assert result.peak_index is None
    assert result.amplitude_index is None
    assert result.qs_nadir_index == 410
    assert result.morphology == "QS"


def test_apply_wave_localization_does_not_replace_native_bounds() -> None:
    ecg = np.zeros((12, 1000))
    _triangular_peak(ecg[1], 315, -0.12)
    _triangular_peak(ecg[1], 400, 1.00)
    _triangular_peak(ecg[1], 425, -0.60)
    _triangular_peak(ecg[1], 620, -0.50)
    p = SimpleNamespace(onset=285, peak=310, offset=340)
    qrs = SimpleNamespace(onset=380, peak=400, offset=450)
    t = SimpleNamespace(onset=560, peak=610, offset=680)
    feature = SimpleNamespace(
        lead="II",
        beat_id=0,
        p=p,
        qrs=qrs,
        t=t,
        r_peak_index=400,
        flags=[],
    )

    apply_hybrid_wave_localization(
        [feature],
        localization_ecg=ecg,
        r_locs=np.asarray([400]),
        fs=500,
    )

    assert (feature.p.onset, feature.p.peak, feature.p.offset) == (285, 310, 340)
    assert (feature.qrs.onset, feature.qrs.peak, feature.qrs.offset) == (380, 400, 450)
    assert (feature.t.onset, feature.t.peak, feature.t.offset) == (560, 610, 680)
    assert feature.p_localized_index == 310
    assert feature.p_localization_polarity == -1
    assert feature.p_localization_absent_probability == 0.0
    assert feature.s_peak_index == 425
    assert feature.t_localized_index == 620
