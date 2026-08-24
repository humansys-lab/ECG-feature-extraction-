from __future__ import annotations

from types import SimpleNamespace

import pytest

import evaluate_ludb
from evaluate_ludb import (
    AnnotationReadError,
    _det_p_by_gt_p_timing,
    match_beats,
    parse_ludb_annotations,
)


def _det(qrs_on: int, qrs_peak: int, p_on: int | None = None, p_peak: int | None = None, p_off: int | None = None):
    return SimpleNamespace(
        qrs=SimpleNamespace(onset=qrs_on, peak=qrs_peak),
        p=SimpleNamespace(onset=p_on, peak=p_peak, offset=p_off),
    )


def test_det_p_by_gt_p_timing_uses_unpaired_next_detected_qrs() -> None:
    gt_beats = [
        {"r_sample": 3344, "p_on": 3773, "p_peak": 3794, "p_off": 3819},
    ]
    current = _det(3333, 3358, 3280, 3298, 3308)
    unpaired_next = _det(3835, 3859, 3783, 3797, 3809)
    far_next = _det(4330, 4353, 4277, 4291, 4304)

    mapping = _det_p_by_gt_p_timing(
        gt_beats,
        [current, unpaired_next, far_next],
        fs=500,
    )

    assert mapping[3344] is unpaired_next


def test_det_p_by_gt_p_timing_does_not_cross_implausible_gap() -> None:
    gt_beats = [
        {"r_sample": 100, "p_on": 240, "p_peak": 250, "p_off": 260},
    ]
    far_next = _det(500, 520, 460, 470, 480)

    mapping = _det_p_by_gt_p_timing(
        gt_beats,
        [far_next],
        fs=500,
    )

    assert 100 not in mapping


def test_match_beats_finds_maximum_one_to_one_assignment() -> None:
    ground_truth = [{"r_sample": 0}, {"r_sample": 5}]
    first = _det(0, 4)
    second = _det(75, 79)

    pairs = match_beats(ground_truth, [first, second], fs=1000)

    assert [(gt["r_sample"], det.qrs.peak) for gt, det in pairs] == [
        (0, 4),
        (5, 79),
    ]


def test_annotation_read_error_is_not_silently_treated_as_empty(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("missing annotation")

    monkeypatch.setattr(evaluate_ludb.wfdb, "rdann", fail)

    with pytest.raises(AnnotationReadError, match="missing annotation"):
        parse_ludb_annotations("record", "ii")
