from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import evaluate_ludb as evaluation
from feature_extraction.ecgfeat.features import _raw_qt_dispersion
from feature_extraction.ecgfeat.dispersion import summarize_qt_dispersion
from feature_extraction.ecgfeat.st_baseline import calibrated_st_signal
from feature_extraction.ecgfeat.qrs import detect_qrs_multilead_with_meta
from feature_extraction.ecgfeat.st_localization import localize_st_robust
from feature_extraction.ecgfeat.repolarization import _trapz
from tests.test_st_hybrid_measurement import _st_signal


def _qrs_signal():
    t = np.arange(5000)
    locations = np.arange(500, 4501, 500)
    pulse = sum(np.exp(-0.5 * ((t - r) / 8) ** 2) for r in locations)
    return np.vstack([(1 + k / 10) * pulse for k in range(12)]), locations


@pytest.mark.parametrize("corruption", ["shift", "flat", "noise"])
def test_excluded_lead_cannot_change_qrs_locations_or_confidence(corruption):
    ecg, locations = _qrs_signal()
    selected = (0, 7, 8, 9, 10, 11)
    reference = detect_qrs_multilead_with_meta(ecg, 500, selected)
    if corruption == "shift":
        ecg[1] = np.roll(ecg[1], 20)
    elif corruption == "flat":
        ecg[1] = 0.0
    else:
        ecg[1] = np.random.default_rng(21).normal(size=ecg.shape[1])
    actual = detect_qrs_multilead_with_meta(ecg, 500, selected)
    np.testing.assert_array_equal(actual.r_locs, locations)
    assert actual == reference


def test_flat_ii_falls_back_to_an_active_participating_lead():
    ecg, locations = _qrs_signal()
    ecg[1] = 0.0
    actual = detect_qrs_multilead_with_meta(ecg, 500)
    np.testing.assert_array_equal(actual.r_locs, locations)


def test_pacing_candidates_can_explicitly_share_a_fiducial_reference():
    ecg, locations = _qrs_signal()
    ecg[1] = np.roll(ecg[1], 20)
    actual = detect_qrs_multilead_with_meta(ecg, 500, (0,), fiducial_lead=1)
    np.testing.assert_array_equal(actual.r_locs, locations + 20)


@pytest.mark.parametrize("leads", [(), (-1,), (12,)])
def test_invalid_detection_leads_fail_explicitly(leads):
    with pytest.raises(ValueError, match="participating ECG"):
        detect_qrs_multilead_with_meta(np.zeros((12, 500)), 500, leads)


def test_st_area_works_without_numpy_2_api(monkeypatch):
    signal = _st_signal()
    kwargs = dict(fs=500, r_index=400, qrs_onset=380, qrs_offset=450,
                  p_offset=345, t_onset=540, next_r_index=850,
                  consensus_j_index=450, consensus_support=10)
    expected = localize_st_robust(signal, **kwargs)
    monkeypatch.delattr(np, "trapezoid", raising=False)
    actual = localize_st_robust(signal, **kwargs)
    assert actual.area_mv_ms == expected.area_mv_ms
    assert actual.j_mv == 0.18


def test_t_wave_integral_works_without_numpy_2_api(monkeypatch):
    monkeypatch.delattr(np, "trapezoid", raising=False)
    assert _trapz(np.asarray([0.0, 1.0, 0.0])) == 1.0


def test_independent_qt_range_does_not_collapse_at_80_ms():
    reps = {
        str(index): SimpleNamespace(
            params={"qt_ms": qt, "reliable_for_qt": True,
                    "qt_confidence_mean": 0.9, "t_amp_mv": 0.3},
            variance={"qt_ms_sd": 2.0},
        )
        for index, qt in enumerate([400, 410, 420, 479])
    }
    assert _raw_qt_dispersion(reps, 90.0) == 79.0
    reps["3"].params["qt_ms"] = 480
    assert _raw_qt_dispersion(reps, 90.0) == 80.0
    reps["3"].params["reliable_for_qt"] = False
    assert _raw_qt_dispersion(reps, 90.0) == 20.0


def test_legacy_dispersion_is_identified_separately_from_the_full_range():
    result = summarize_qt_dispersion({"I": 400, "II": 410, "V5": 420, "V6": 480})
    assert result.independent_ms == 80.0
    assert result.legacy_ms == 20.0
    assert result.legacy_source == "legacy_compact_cluster_range"
    assert result.legacy_excluded_leads == ("V6",)
    assert result.p90_p10_ms == pytest.approx(59.0)


def _st_pr_anchors():
    return [
        SimpleNamespace(lead=lead, flags=[],
                        p=SimpleNamespace(offset=start + 345),
                        qrs=SimpleNamespace(onset=start + 380))
        for start in range(0, 5000, 1000)
        for lead in evaluation.STANDARD_12_LEADS
    ]


@pytest.mark.parametrize("drift_per_second", [0.0, 0.04, -0.04])
@pytest.mark.parametrize("amplitude", [0.18, -0.18])
def test_calibrated_st_preserves_known_voltage_with_dc_and_linear_drift(drift_per_second, amplitude):
    raw = np.tile(_st_signal(st_j_mv=amplitude, st80_mv=amplitude), 5)
    signal = raw + 0.4 + drift_per_second * np.arange(raw.size) / 500
    ecg = np.vstack([signal] * 12)
    original = ecg.copy()
    result = calibrated_st_signal(ecg, 500, _st_pr_anchors())
    assert not result.unavailable_leads
    np.testing.assert_array_equal(ecg, original)
    measured = localize_st_robust(
        result.signal[1], fs=500, r_index=2400, qrs_onset=2380,
        qrs_offset=2450, p_offset=2345, t_onset=2540, next_r_index=3400,
        consensus_j_index=2450, consensus_support=10,
    )
    assert measured.j_mv == pytest.approx(amplitude, abs=0.003)
    assert measured.st80_mv == pytest.approx(amplitude, abs=0.003)


def test_calibrated_st_marks_missing_pr_anchors_unavailable():
    result = calibrated_st_signal(np.zeros((12, 1000)), 500, [])
    assert result.unavailable_leads == tuple(evaluation.STANDARD_12_LEADS)
    assert all(value == 0 for value in result.anchor_counts.values())


def _evaluation_fixture(monkeypatch, *, fail=False, missing=False):
    gt = [{"r_sample": 210, "qrs_on": 200, "qrs_off": 240,
           "t_on": 300, "t_peak": 350, "t_off": 400,
           "p_on": None, "p_peak": None, "p_off": None}]
    monkeypatch.setattr(evaluation, "STANDARD_12_LEADS", ["II"])
    monkeypatch.setattr(evaluation.wfdb, "rdrecord", lambda _path: SimpleNamespace(
        p_signal=np.zeros((1000, 12)), fs=500))
    monkeypatch.setattr(evaluation, "parse_ludb_annotations", lambda *_args: gt)
    feature = SimpleNamespace(
        lead="II", qrs=SimpleNamespace(onset=100, peak=105, offset=120),
        p=SimpleNamespace(onset=None, peak=None, offset=None),
        t=SimpleNamespace(onset=150, peak=175, offset=None if missing else 200),
        qt_ms=None if missing else 400.0, qt_confidence=0.8,
        beat_measurement_reliable=True,
    )

    def extract(*_args):
        if fail:
            raise ValueError("test extraction failure")
        return SimpleNamespace(fs=250, beat_features=[feature])

    return SimpleNamespace(extract=extract)


def test_evaluation_matches_and_scores_on_detector_timebase(monkeypatch):
    extractor = _evaluation_fixture(monkeypatch)
    coverage = []
    rows = evaluation.evaluate_record("test", extractor, coverage_rows=coverage)
    assert len(rows) == 1
    for field in ("qrs_on_err", "qrs_off_err", "t_on_err", "t_off_err", "qt_err"):
        assert rows[0][field] == 0.0
    assert coverage[0]["matched_qrs"] == 1


def test_resampling_references_preserves_fractional_sample_times():
    gt = [{"r_sample": 211, "qrs_on": 201, "p_on": None}]
    scaled = evaluation.rescale_annotations(gt, 500, 250)
    assert scaled == [{"r_sample": 105.5, "qrs_on": 100.5, "p_on": None}]
    assert evaluation.boundary_error_ms(scaled[0]["qrs_on"], 100, 250) == -2.0
    assert gt[0]["r_sample"] == 211


def test_empty_annotation_lead_remains_in_coverage_ledger(monkeypatch):
    extractor = _evaluation_fixture(monkeypatch)
    monkeypatch.setattr(evaluation, "parse_ludb_annotations", lambda *_args: [])
    coverage = []
    assert evaluation.evaluate_record("test", extractor, coverage_rows=coverage) == []
    assert len(coverage) == 1
    assert coverage[0]["gt_qrs"] == 0
    assert coverage[0]["unmatched_det_in_gt_span"] == 0


@pytest.mark.parametrize("failure", [False, True])
def test_coverage_counts_missing_measurements_and_extraction_failures(monkeypatch, failure):
    extractor = _evaluation_fixture(monkeypatch, fail=failure, missing=True)
    coverage, failures = [], []
    rows = evaluation.evaluate_record("test", extractor,
                                      coverage_rows=coverage, failure_details=failures)
    summary = evaluation.compute_summary(rows, coverage_rows=coverage)
    t_offset = summary["coverage"]["boundaries"]["t_off"]
    assert t_offset == {"annotated": 1, "scored": 0, "missing_or_unmatched": 1, "coverage": 0.0}
    assert summary["coverage"]["unmatched_gt_qrs"] == int(failure)
    assert bool(failures) is failure
