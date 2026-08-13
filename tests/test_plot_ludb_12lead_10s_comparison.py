from __future__ import annotations

import numpy as np

from compare_ludb_detectors import WaveEvent
from plot_ludb_12lead_10s_comparison import (
    _signal_limits,
    _visible_events,
)


def test_visible_events_keeps_only_in_record_peaks() -> None:
    events = [
        WaveEvent(None, -1, None),
        WaveEvent(None, 0, None),
        WaveEvent(None, 999, None),
        WaveEvent(None, 1000, None),
        WaveEvent(None, None, None),
    ]

    assert [event.peak for event in _visible_events(events, 1000)] == [0, 999]


def test_signal_limits_are_finite_for_flat_signal() -> None:
    low, high = _signal_limits(np.zeros(5000))

    assert np.isfinite(low)
    assert np.isfinite(high)
    assert low < 0 < high
