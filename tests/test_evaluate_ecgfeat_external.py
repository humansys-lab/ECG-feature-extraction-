"""Protocol regressions: protect denominators, timing and independent matching."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts.evaluate_ecgfeat_external import (
    WaveEvent, adapt_signal, interval_match, load_annotations, match_events,
    noise_phase, score, sparse_match, windows,
)


def event(peak):
    return WaveEvent(None, peak, None)


def test_sparse_components_preserve_full_dp_objective():
    rng = np.random.default_rng(1984)
    for _ in range(100):
        gt = [event(int(x)) for x in sorted(rng.choice(1000, 25, replace=False))]
        det = [event(int(x)) for x in sorted(rng.choice(1000, 35, replace=False))]
        full = match_events(gt, det, 30)[0]
        sparse = sparse_match(gt, det, 30)
        assert len(full) == len(sparse)
        assert sum(abs(g.peak-d.peak) for g, d in full) == sum(abs(g.peak-d.peak) for g, d in sparse)


@pytest.mark.parametrize("length,fs", [(43201, 360), (15360, 128), (650000, 360), (9999, 1000)])
def test_windows_cover_exactly_once_and_merge_short_tail(length, fs):
    items = list(windows(length, fs))
    assert items[0][0] == 0 and items[-1][1] == length
    assert all(a[1] == b[0] for a, b in zip(items, items[1:]))
    assert all(a <= start < end <= b for start, end, a, b in items)
    assert all(end-start >= 5*fs for start, end, _, _ in items)


def test_noise_phases_are_half_open():
    assert [noise_phase(s, 360) for s in (0, 108000-1, 108000, 151200-1, 151200, 194400)] == [
        "clean", "clean", "noisy", "noisy", "clean", "noisy"]


def test_interval_matching_does_not_invent_peak_truth():
    gt = [WaveEvent(100, None, 200), WaveEvent(300, None, 400)]
    det = [WaveEvent(110, 150, 190), WaveEvent(310, 350, 410), WaveEvent(None, 600, None)]
    row, matched = score("test", "default", "P", "II", gt, det, 1000, "ok", intervals=True, length=1000)
    assert (row["tp"], row["fp"], row["fn"]) == (2, 1, 0)
    assert all(m["peak_error_ms"] is None for m in matched)
    assert row["sample_intersection"] == 170
    assert row["sample_gt"] == 200
    assert row["sample_det"] == 180


def test_interval_matching_is_one_to_one():
    gt = [WaveEvent(100, None, 200)]
    det = [WaveEvent(100, 150, 200), WaveEvent(101, 151, 201)]
    assert interval_match(gt, det) == [(gt[0], det[0])]


def test_failed_extraction_keeps_every_annotation_as_fn():
    row, matches = score("test", "default", "P", "II", [event(0), event(100)], [], 1000, "partial_failure")
    assert row["fn"] == row["gt_count"] == 2
    assert not matches


def test_but_sentinel_and_nonbeat_symbols_are_excluded():
    anns = [SimpleNamespace(sample=np.array([-1, 0, 100]), symbol=["N"]*3),
            SimpleNamespace(sample=np.array([1, 40, 80]), symbol=["+", "~", "R"])]
    with patch("scripts.evaluate_ecgfeat_external.wfdb.rdann", side_effect=anns):
        truth, excluded = load_annotations("but-pdb", {"path": "unused"}, 360, 1000)
    assert [e.peak for e in truth["P"]] == [0, 100]
    assert [e.peak for e in truth["QRS"]] == [80]
    assert len(excluded) == 3


def test_isp_censored_endpoint_and_zero_length_annotation_are_audited():
    truth, audit = load_annotations("isp", {"target": [(2, 900, 1100), (2, 500, 500)]}, 1000, 1000)
    assert truth["T"] == [WaveEvent(900, None, 1000)]
    assert [e["action"] for e in audit] == ["clip_right_censored", "exclude_zero_length"]


def test_sparse_adapter_never_fabricates_einthoven_leads():
    record = SimpleNamespace(p_signal=np.array([[1., 2.], [3., 4.]]), n_sig=2, sig_len=2, units=["mV"]*2)
    result = adapt_signal(record, "but-pdb")
    assert np.array_equal(result[1], [1, 3])
    assert np.array_equal(result[7], [2, 4])
    assert np.count_nonzero(result) == 4


def test_isp_calibration_and_named_lead_order():
    names = ["v6", "v5", "v4", "v3", "v2", "v1", "avf", "avl", "avr", "iii", "ii", "i"]
    record = SimpleNamespace(p_signal=np.tile(np.arange(12)*1000., (3, 1)), sig_name=names, units=["mkv"]*12)
    result = adapt_signal(record, "isp")
    assert np.array_equal(result[:, 0], np.arange(11, -1, -1))
