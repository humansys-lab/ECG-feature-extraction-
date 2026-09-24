from __future__ import annotations

from math import atan2, degrees

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1.26 has only trapz
import pytest

from feature_extraction.ecgfeat.glasgow_measurements import measure_glasgow_profile


def _synthetic_signal() -> np.ndarray:
    sig = np.zeros(500, dtype=float)
    sig[100:126] = np.linspace(0.0, 0.20, 26)
    sig[126:138] = np.linspace(0.20, -0.10, 12)
    sig[138:151] = np.linspace(-0.10, 0.0, 13)
    sig[200] = 0.02
    sig[201:206] = np.linspace(0.0, -0.20, 5)
    sig[206:221] = np.linspace(-0.20, 0.90, 15)
    sig[221:241] = np.linspace(0.90, -0.30, 20)
    sig[241:251] = np.linspace(-0.30, 0.05, 10)
    sig[250:301] = np.linspace(0.05, 0.10, 51)
    sig[301:326] = np.linspace(0.10, 0.16, 25)
    sig[326:376] = np.linspace(0.16, 0.30, 50)
    sig[376:421] = np.linspace(0.30, -0.15, 45)
    sig[421:451] = np.linspace(-0.15, 0.0, 30)
    return sig


def _profile() -> dict[str, object]:
    return measure_glasgow_profile(
        _synthetic_signal(),
        fs=500,
        beat_window_start_index=1000,
        p_on=100,
        p_off=150,
        qrs_on=200,
        qrs_off=250,
        t_on=300,
        t_off=450,
        r_peak=220,
        baseline_mv=0.0,
        r_prime_amp_baseline_mv=None,
        s_prime_amp_baseline_mv=None,
        component_durations={
            "q_duration_ms": 12.0,
            "r_duration_ms": 30.0,
            "s_duration_ms": 40.0,
            "r_prime_duration_ms": None,
            "s_prime_duration_ms": None,
        },
        vat_ms=40.0,
        qt_ms=500.0,
        delta_present=True,
        delta_confidence=0.8,
        qrs_notch_count=1,
        paper_speed_mm_per_s=25.0,
        gain_mm_per_mv=10.0,
    )


def test_profile_uses_qrs_onset_as_amplitude_reference() -> None:
    profile = _profile()

    assert profile["reference_mv"] == pytest.approx(0.02)
    assert profile["p_positive_amp_mv"] == pytest.approx(0.18)
    assert profile["p_negative_amp_mv"] == pytest.approx(-0.12)
    assert profile["q_amp_mv"] == pytest.approx(-0.22)
    assert profile["r_amp_mv"] == pytest.approx(0.88)
    assert profile["s_amp_mv"] == pytest.approx(-0.32)
    assert profile["t_positive_amp_mv"] == pytest.approx(0.28)
    assert profile["t_negative_amp_mv"] == pytest.approx(-0.17)


def test_profile_onsets_are_elapsed_ms_from_beat_window_start() -> None:
    profile = _profile()

    assert profile["beat_window_start_index"] == 1000
    assert profile["p_onset_ms"] == pytest.approx(200.0)
    assert profile["qrs_onset_ms"] == pytest.approx(400.0)
    assert profile["t_onset_ms"] == pytest.approx(600.0)


def test_profile_qrs_area_has_explicit_uv_ms_and_matrix_scale() -> None:
    sig = _synthetic_signal()
    expected = _trapezoid(
        np.abs(sig[200:251] - sig[200]),
        dx=1000.0 / 500.0,
    ) * 1000.0
    profile = _profile()

    assert profile["qrs_area_uv_ms"] == pytest.approx(expected)
    assert profile["qrs_area_matrix"] == pytest.approx(expected / 20.0)


def test_profile_preserves_signed_qrs_area_for_polarity_rules() -> None:
    sig = _synthetic_signal()
    expected = _trapezoid(
        sig[200:251] - sig[200],
        dx=1000.0 / 500.0,
    ) * 1000.0

    profile = _profile()

    assert profile["qrs_signed_area_uv_ms"] == pytest.approx(expected)
    assert profile["qrs_signed_area_matrix"] == pytest.approx(expected / 20.0)


def test_profile_uses_fractional_st_t_points_and_physical_st_angle() -> None:
    sig = _synthetic_signal()
    profile = _profile()
    reference = sig[200]
    expected_two_eighths = sig[300] - reference
    expected_three_eighths = sig[325] - reference
    horizontal_mm = (325 - 250) * 1000.0 / 500.0 * 25.0 / 1000.0
    vertical_mm = (sig[325] - sig[250]) * 10.0
    expected_angle = degrees(atan2(vertical_mm, horizontal_mm))

    assert profile["st_2_8_index"] == 300
    assert profile["st_3_8_index"] == 325
    assert profile["st_2_8_amp_mv"] == pytest.approx(expected_two_eighths)
    assert profile["st_3_8_amp_mv"] == pytest.approx(expected_three_eighths)
    assert profile["st_slope_deg"] == pytest.approx(expected_angle)


def test_profile_exports_delta_confidence_and_biphasic_t_morphology() -> None:
    profile = _profile()

    assert profile["delta_confidence_pct"] == pytest.approx(80.0)
    assert profile["t_morphology"] == 2


def test_profile_returns_explicit_none_values_when_bounds_are_missing() -> None:
    profile = measure_glasgow_profile(
        np.zeros(100),
        fs=500,
        beat_window_start_index=0,
        p_on=None,
        p_off=None,
        qrs_on=None,
        qrs_off=None,
        t_on=None,
        t_off=None,
        r_peak=None,
        baseline_mv=0.0,
        r_prime_amp_baseline_mv=None,
        s_prime_amp_baseline_mv=None,
        component_durations={},
        vat_ms=None,
        qt_ms=None,
        delta_present=False,
        delta_confidence=None,
        qrs_notch_count=0,
    )

    assert profile["p_onset_ms"] is None
    assert profile["qrs_area_uv_ms"] is None
    assert profile["t_morphology"] is None
    assert profile["unavailable_reasons"]["qrs_area_uv_ms"] == "qrs_bounds_unavailable"
