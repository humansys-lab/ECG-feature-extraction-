"""Signed U-wave localisation (feature_extraction/ecgfeat/u_wave.py)."""
from __future__ import annotations

import numpy as np

from feature_extraction.ecgfeat.u_wave import measure_u_wave


FS = 500


def _gaussian(centre_s: float, amplitude_mv: float, width_s: float) -> np.ndarray:
    t = np.arange(0.0, 1.0, 1.0 / FS)
    return amplitude_mv * np.exp(-((t - centre_s) ** 2) / (2.0 * width_s**2))


def _t_wave() -> np.ndarray:
    return _gaussian(0.30, 0.35, 0.035)


T_OFF = int(0.39 * FS)


def _measure(signal: np.ndarray):
    return measure_u_wave(
        signal,
        t_off=T_OFF,
        baseline=0.0,
        fs=FS,
        search_cap=len(signal) - 1,
    )


def test_upright_u_wave_is_measured_with_positive_sign() -> None:
    result = _measure(_t_wave() + _gaussian(0.48, 0.12, 0.030))

    assert result.reliable
    assert result.polarity == 1
    assert result.amplitude_mv > 0.10
    assert result.duration_ms > 0.0


def test_inverted_u_wave_keeps_its_negative_sign() -> None:
    result = _measure(_t_wave() - _gaussian(0.48, 0.09, 0.030))

    assert result.reliable
    assert result.polarity == -1
    assert result.amplitude_mv < -0.05


def test_baseline_drift_is_not_a_u_wave() -> None:
    drift = 0.30 * np.arange(0.0, 1.0, 1.0 / FS)

    result = _measure(_t_wave() + drift)

    assert not result.reliable
    assert result.amplitude_mv is None


def test_absent_u_wave_reports_no_measurement() -> None:
    result = _measure(_t_wave())

    assert not result.reliable
    assert result.reject_reason == "no_interior_extremum"


def test_biphasic_t_terminal_lobe_is_rejected_as_a_u_wave() -> None:
    """The negative lobe of a biphasic T is continuous with the T wave.

    T-offset estimators put the offset at the polarity crossing, so the lobe
    opens exactly where the U-wave window starts. Only the absence of an
    isoelectric T-U segment distinguishes it from a genuine inverted U wave.
    """
    biphasic = _t_wave() - _gaussian(0.45, 0.30, 0.045)

    result = _measure(biphasic)

    assert not result.reliable
    assert result.reject_reason == "no_isoelectric_t_u_segment"


def test_genuine_inverted_u_survives_the_isoelectric_test() -> None:
    """Same polarity and depth as the biphasic lobe, but separated in time."""
    separated = _t_wave() - _gaussian(0.50, 0.30, 0.028)

    result = _measure(separated)

    assert result.reliable
    assert result.polarity == -1
    assert result.isoelectric_gap_ms >= 20.0


def test_search_cap_prevents_reading_into_the_next_beat() -> None:
    signal = _t_wave() + _gaussian(0.48, 0.12, 0.030)

    capped = measure_u_wave(
        signal, t_off=T_OFF, baseline=0.0, fs=FS, search_cap=int(0.42 * FS)
    )

    assert not capped.reliable


def test_missing_t_offset_is_reported_rather_than_guessed() -> None:
    result = measure_u_wave(
        _t_wave(), t_off=None, baseline=0.0, fs=FS, search_cap=None
    )

    assert not result.reliable
    assert result.reject_reason == "no_t_offset"


def test_a_narrow_spike_after_the_t_wave_is_not_a_u_wave() -> None:
    """The U wave is the slowest deflection on the ECG; a fast blip is noise.

    Amplitude and prominence cannot reject this -- a 30 ms spike is both deep
    and locally prominent. Only its duration gives it away.
    """
    spike = _t_wave() - _gaussian(0.50, 0.10, 0.006)

    result = _measure(spike)

    assert not result.reliable
    assert result.reject_reason == "too_narrow_for_a_u_wave"


def test_a_physiologically_wide_u_wave_is_accepted() -> None:
    result = _measure(_t_wave() + _gaussian(0.50, 0.10, 0.040))

    assert result.reliable
    assert result.duration_ms >= 60.0
