from __future__ import annotations

from types import SimpleNamespace

import pytest

from compare_ludb_detectors import (
    WaveEvent,
    _select_ecgfeat_qrs_peak,
    _select_ecgfeat_wave_event,
    aggregate_metrics,
    compare_wave,
    match_events,
)


def event(peak: int, onset: int | None = None, offset: int | None = None) -> WaveEvent:
    return WaveEvent(onset=onset, peak=peak, offset=offset)


def test_match_events_maximizes_matches_before_total_error() -> None:
    # A nearest-pair-first greedy matcher takes 5->3 and leaves only one TP.
    # The optimal ordered assignment is 0->3 and 5->8.
    pairs, false_negatives, false_positives = match_events(
        [event(0), event(5)],
        [event(3), event(8)],
        tolerance_samples=3,
    )

    assert [(gt.peak, detected.peak) for gt, detected in pairs] == [(0, 3), (5, 8)]
    assert false_negatives == 0
    assert false_positives == 0


def test_match_events_minimizes_error_after_maximizing_count() -> None:
    pairs, _, false_positives = match_events(
        [event(100)],
        [event(95), event(101)],
        tolerance_samples=10,
    )

    assert [(gt.peak, detected.peak) for gt, detected in pairs] == [(100, 101)]
    assert false_positives == 1


def test_compare_and_aggregate_metrics_use_milliseconds() -> None:
    detection, matches = compare_wave(
        record_id="1",
        lead="II",
        method="example",
        wave="QRS",
        gt_events=[event(100, 90, 110), event(200, 190, 210)],
        detected_events=[event(102, 88, 114), event(300, 290, 310)],
        fs=500,
        tolerance_ms=20.0,
    )
    summary = aggregate_metrics([detection], matches, ("method", "wave"))[0]

    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 1, 1)
    assert summary["sensitivity"] == 0.5
    assert summary["precision"] == 0.5
    assert summary["f1"] == 0.5
    assert summary["onset_bias_ms"] == -4.0
    assert summary["peak_mae_ms"] == 4.0
    assert summary["offset_bias_ms"] == 8.0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("global", 100),
        ("lead-fiducial", 104),
        ("positive-r", 108),
        ("hybrid-prominence", 106),
    ],
)
def test_select_ecgfeat_qrs_peak_source(source: str, expected: int) -> None:
    beat_feature = SimpleNamespace(
        qrs=SimpleNamespace(peak=104),
        r_peak_index=108,
        r_localized_index=106,
    )

    assert _select_ecgfeat_qrs_peak(beat_feature, 100, source) == expected


def test_positive_r_source_falls_back_to_lead_fiducial() -> None:
    beat_feature = SimpleNamespace(
        qrs=SimpleNamespace(peak=104),
        r_peak_index=None,
    )

    assert _select_ecgfeat_qrs_peak(
        beat_feature,
        100,
        "positive-r",
    ) == 104


def test_select_ecgfeat_hybrid_wave_peak_keeps_native_boundaries() -> None:
    beat_feature = SimpleNamespace(
        p=SimpleNamespace(onset=70, peak=80, offset=90),
        p_localized_index=84,
    )

    event = _select_ecgfeat_wave_event(
        beat_feature,
        "p",
        "hybrid-peaks",
    )

    assert (event.onset, event.peak, event.offset) == (70, 80, 90)


def test_select_ecgfeat_hybrid_wave_peak_does_not_rescue_unvalidated_p() -> None:
    beat_feature = SimpleNamespace(
        p=SimpleNamespace(onset=None, peak=None, offset=None),
        p_localized_index=84,
    )

    event = _select_ecgfeat_wave_event(
        beat_feature,
        "p",
        "hybrid-peaks",
    )

    assert event.peak is None


def test_restrict_to_annotated_span_drops_only_unmatchable_detections() -> None:
    """The filter must remove exactly the detections that could never match.

    LUDB annotates a middle window of each record, so beats outside it are
    real but unreferenced. A detection within the match tolerance of the span
    still has a chance to pair with a real annotation and must survive.
    """
    from compare_ludb_detectors import WaveEvent, restrict_to_annotated_span

    gt = [WaveEvent(None, 1000, None), WaveEvent(None, 2000, None)]
    detected = [
        WaveEvent(None, 100, None),    # far before the span -> unmatchable
        WaveEvent(None, 960, None),    # inside the tolerance of the first GT
        WaveEvent(None, 2040, None),   # inside the tolerance of the last GT
        WaveEvent(None, 5000, None),   # far after the span -> unmatchable
    ]

    kept, dropped = restrict_to_annotated_span(gt, detected, tolerance_samples=50)

    assert [event.peak for event in kept] == [960, 2040]
    assert dropped == 2


def test_restrict_to_annotated_span_drops_everything_when_nothing_is_annotated() -> None:
    """A lead with no annotation cannot score any detection as correct."""
    from compare_ludb_detectors import WaveEvent, restrict_to_annotated_span

    detected = [WaveEvent(None, 100, None), WaveEvent(None, 200, None)]

    kept, dropped = restrict_to_annotated_span([], detected, tolerance_samples=50)

    assert kept == []
    assert dropped == 2


def test_restrict_to_annotated_span_preserves_matching_and_sensitivity() -> None:
    """Restricting the span must not change TP/FN, only FP."""
    from compare_ludb_detectors import WaveEvent, compare_wave

    gt = [WaveEvent(None, 1000, None), WaveEvent(None, 2000, None)]
    detected = [
        WaveEvent(None, 100, None),
        WaveEvent(None, 1005, None),
        WaveEvent(None, 2005, None),
    ]
    common = dict(
        record_id="1", lead="II", method="ecgfeat", wave="QRS",
        gt_events=gt, detected_events=detected, fs=500, tolerance_ms=75.0,
    )

    plain, plain_matches = compare_wave(**common)
    span, span_matches = compare_wave(**common, restrict_span=True)

    assert plain["tp"] == span["tp"] == 2
    assert plain["fn"] == span["fn"] == 0
    assert plain["fp"] == 1 and span["fp"] == 0
    assert span["det_outside_annotated_span"] == 1
    # Boundary errors come from the matched pairs, which are unchanged.
    assert len(plain_matches) == len(span_matches)
