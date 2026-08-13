"""Per-lead pooling of the signed U-wave measurement (features._u_wave_params)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from feature_extraction.ecgfeat.features import _u_wave_params


def _beat(*, amplitude, polarity=None, reliable=True, duration=90.0):
    return SimpleNamespace(
        u_amp_signed_mv=amplitude,
        u_polarity=(
            polarity
            if polarity is not None
            else (0 if amplitude is None else (1 if amplitude >= 0 else -1))
        ),
        u_measurement_reliable=reliable,
        u_dur_ms=duration,
        u_prominence_mv=0.08,
        u_isoelectric_gap_ms=40.0,
    )


def test_agreeing_beats_pool_to_their_median() -> None:
    beats = [_beat(amplitude=0.10), _beat(amplitude=0.14), _beat(amplitude=0.12)]

    params = _u_wave_params(None, beats)

    assert params["u_amp_signed_mv"] == 0.12
    assert params["u_polarity"] == 1
    assert params["u_measurement_reliable"]
    assert params["u_polarity_agreement"] == 1.0


def test_minority_polarity_beats_do_not_dilute_the_amplitude() -> None:
    """Signed pooling across disagreeing polarities would cancel toward zero."""
    beats = [
        _beat(amplitude=-0.20),
        _beat(amplitude=-0.24),
        _beat(amplitude=-0.22),
        _beat(amplitude=0.18),
    ]

    params = _u_wave_params(None, beats)

    assert params["u_polarity"] == -1
    assert params["u_amp_signed_mv"] == -0.22
    assert params["u_polarity_agreement"] == 0.75


def test_a_polarity_tie_refuses_to_report_a_u_wave() -> None:
    beats = [_beat(amplitude=0.12), _beat(amplitude=-0.12)]

    params = _u_wave_params(None, beats)

    assert params["u_polarity"] == 0
    assert params["u_amp_signed_mv"] is None
    assert not params["u_measurement_reliable"]
    assert params["u_polarity_agreement"] == 0.5


def test_a_single_agreeing_beat_among_many_is_not_reliable() -> None:
    beats = [_beat(amplitude=0.12), _beat(amplitude=None, reliable=False)] + [
        _beat(amplitude=None, reliable=False) for _ in range(3)
    ]

    params = _u_wave_params(None, beats)

    assert params["u_amp_signed_mv"] == 0.12
    assert not params["u_measurement_reliable"]
    assert params["u_beats_measured"] == 1


def test_unreliable_beats_are_excluded_entirely() -> None:
    beats = [
        _beat(amplitude=0.12),
        _beat(amplitude=0.14),
        _beat(amplitude=0.90, reliable=False),
    ]

    params = _u_wave_params(None, beats)

    assert params["u_amp_signed_mv"] == 0.13
    assert params["u_beats_measured"] == 2
    assert params["u_beats_total"] == 3


def test_no_measurable_beats_reports_nothing_rather_than_zero() -> None:
    params = _u_wave_params(None, [_beat(amplitude=None, reliable=False)])

    assert params["u_amp_signed_mv"] is None
    assert params["u_polarity"] == 0
    assert not params["u_measurement_reliable"]
    assert params["u_beats_measured"] == 0


def test_the_representative_beat_participates_in_the_pool() -> None:
    params = _u_wave_params(_beat(amplitude=0.20), [_beat(amplitude=0.10)])

    assert params["u_amp_signed_mv"] == pytest.approx(0.15)
    assert params["u_beats_measured"] == 2
    assert params["u_measurement_reliable"]
