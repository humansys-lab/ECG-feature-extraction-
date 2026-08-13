from __future__ import annotations

import pytest

from compare_ludb_detectors import WaveEvent
from plot_ecgfeat_neurokit_comparison import (
    _events_in_window,
    calculate_window_metrics,
    resolve_time_window,
)


def event(
    peak: int,
    onset: int | None = None,
    offset: int | None = None,
) -> WaveEvent:
    return WaveEvent(onset=onset, peak=peak, offset=offset)


def test_resolve_time_window_can_zoom_around_one_based_beat() -> None:
    first, last = resolve_time_window(
        n_samples=5000,
        fs=500,
        start_sec=None,
        end_sec=None,
        beat=2,
        beat_range=None,
        reference_qrs=[event(500), event(1500), event(2500)],
        before_sec=0.4,
        after_sec=0.6,
    )

    assert first == 1300
    assert last == 1800


def test_resolve_time_window_rejects_conflicting_modes() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_time_window(
            n_samples=5000,
            fs=500,
            start_sec=1.0,
            end_sec=None,
            beat=1,
            beat_range=None,
            reference_qrs=[event(500)],
            before_sec=0.4,
            after_sec=0.6,
        )


def test_resolve_time_window_can_show_consecutive_beats() -> None:
    first, last = resolve_time_window(
        n_samples=5000,
        fs=500,
        start_sec=None,
        end_sec=None,
        beat=None,
        beat_range=(2, 4),
        reference_qrs=[event(500), event(1200), event(1900), event(2600)],
        before_sec=0.4,
        after_sec=0.6,
    )

    assert first == 1000
    assert last == 2900


def test_resolve_time_window_rejects_reversed_beat_range() -> None:
    with pytest.raises(ValueError, match="FIRST must not exceed LAST"):
        resolve_time_window(
            n_samples=5000,
            fs=500,
            start_sec=None,
            end_sec=None,
            beat=None,
            beat_range=(4, 2),
            reference_qrs=[event(500), event(1200), event(1900), event(2600)],
            before_sec=0.4,
            after_sec=0.6,
        )


def test_events_in_window_uses_peak_as_visibility_anchor() -> None:
    selected = _events_in_window(
        [event(99), event(100), event(200), event(201)],
        100,
        200,
    )

    assert [item.peak for item in selected] == [100, 200]


def test_window_metrics_report_detection_and_peak_mae() -> None:
    rows = calculate_window_metrics(
        ground_truth={
            "P": [event(80)],
            "QRS": [event(100), event(200)],
            "T": [event(140)],
        },
        detected={
            "P": [event(81)],
            "QRS": [event(102), event(300)],
            "T": [event(142)],
        },
        first_sample=0,
        last_sample=400,
        fs=500,
        r_tolerance_ms=20.0,
        wave_tolerance_ms=20.0,
    )
    qrs = next(row for row in rows if row["wave"] == "QRS")

    assert (qrs["tp"], qrs["fp"], qrs["fn"]) == (1, 1, 1)
    assert qrs["f1"] == 0.5
    assert qrs["peak_mae_ms"] == 4.0
