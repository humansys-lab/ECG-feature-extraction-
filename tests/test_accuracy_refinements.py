from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extraction.ecgfeat import ECGFeatureExtractor, RefinementConfig
from feature_extraction.ecgfeat.boundary_refinement import correct_p_boundaries, t_onset_change_point, sustained_qrs_offset
from feature_extraction.ecgfeat.grouping import _morphology_cluster
from feature_extraction.ecgfeat.models import PWaveBeatAssessment, PWaveLeadBoundary
from feature_extraction.ecgfeat.p_wave_engine import _reconcile_parallel_consensus_models
from feature_extraction.ecgfeat.st_baseline import adaptive_st_signal
from feature_extraction.ecgfeat.st_localization import localize_st_robust
from feature_extraction.ecgfeat.t_wave_refinement import robust_t_offset_fusion
from tests.test_extraction_regression_fixes import _st_pr_anchors
from tests.test_st_hybrid_measurement import _st_signal
from tests.test_t_wave_refinement import _feature


def test_qt_calibration_weights_records_not_repeated_beats():
    from feature_extraction.ecgfeat.calibration import QTErrorCalibration
    rows = [{"record": "a", "qt_conf": .9, "qt_err": 80},
            {"record": "b", "qt_conf": .7, "qt_err": 4}]
    a = QTErrorCalibration.fit(rows)
    b = QTErrorCalibration.fit(rows[:1] * 100 + rows[1:])
    assert a.global_risk == pytest.approx(.5)
    assert b.global_risk == pytest.approx(.5)
    assert a.risk_by_bin == pytest.approx(b.risk_by_bin)
    assert a.predict(.9)["probability_abs_qt_error_gt_threshold"] > a.predict(.7)["probability_abs_qt_error_gt_threshold"]
    assert QTErrorCalibration.from_dict(a.to_dict()).to_dict() == a.to_dict()
    assert a.predict(float("nan"))["confidence_bin"] == "missing"


def test_qt_calibration_rejects_empty_errors_and_invalid_model():
    from feature_extraction.ecgfeat.calibration import QTErrorCalibration
    with pytest.raises(ValueError):
        QTErrorCalibration.fit([{"record": "a", "qt_err": float("nan")}])
    model = QTErrorCalibration.fit([{"record": "a", "qt_err": 30}]).to_dict()
    model["global_risk"] = 2
    with pytest.raises(ValueError):
        QTErrorCalibration.from_dict(model)


@pytest.mark.parametrize("profile", ["summary", "audit", "debug"])
def test_direct_export_is_exactly_equivalent_and_does_not_mutate(profile):
    from feature_extraction.ecgfeat.export import to_dict, prepare_json_export
    from tests.test_export_contract import _make_features
    from copy import deepcopy
    features = _make_features()
    before = deepcopy(features)
    expected = prepare_json_export(to_dict(features), profile=profile, round_ndigits=None)
    actual = prepare_json_export(to_dict(features, profile=profile), profile=profile, round_ndigits=None)
    assert actual == expected
    assert features == before


def test_sparse_st_adapter_preserves_unsupported_baseline_rejection():
    from analyze_physionet_st import _assess_sparse_hybrid
    for reason in ("baseline_anchor_support_insufficient", "insufficient_calibrated_pr_anchors"):
        feature = SimpleNamespace(st_hybrid_unreliable_reason=reason, st_hybrid_consensus_support=2, flags=[])
        result = _assess_sparse_hybrid(feature, physical_lead_count=2, fs=500)
        assert not result.usable and result.reason == reason


def test_refinement_options_are_explicit_and_validated():
    assert not RefinementConfig().enabled
    assert RefinementConfig.experimental().enabled
    with pytest.raises(ValueError):
        RefinementConfig(t_bidirectional="yes")
    with pytest.raises(TypeError):
        ECGFeatureExtractor(refinement={})
    with pytest.raises(ValueError):
        ECGFeatureExtractor(st_amplitude_source="adaptive_pr_tp", enable_hybrid_st_measurement=False)


def test_correlated_fusion_is_invariant_to_weight_units_and_duplicate_evidence():
    indices, weights, sources = [200, 203, 206, 209], [1, .9, .8, 1], ["I", "II", "V2", "V5"]
    a = robust_t_offset_fusion(indices, weights, sources, fs=500, account_for_correlation=True)
    b = robust_t_offset_fusion(indices, [w * 10 for w in weights], sources, fs=500, account_for_correlation=True)
    c = robust_t_offset_fusion(indices + [200] * 8, weights + [1] * 8, sources + ["I"] * 8,
                              fs=500, account_for_correlation=True)
    for result in (b, c):
        assert (result.center_index, result.ci_low_index, result.ci_high_index) == (a.center_index, a.ci_low_index, a.ci_high_index)
        assert result.effective_support == pytest.approx(a.effective_support)
    assert a.interval_status == "heuristic_uncalibrated"


def test_distinct_p_models_are_not_averaged_without_evidence():
    a = PWaveBeatAssessment(0, 100, 150, 100, 150, accepted=True,
                           onset_confidence=.5, offset_confidence=.5)
    _reconcile_parallel_consensus_models(a, legacy_onset=140, legacy_offset=190,
                                        fs=500, evidence_arbitration=True, legacy_confidence=.5)
    assert not a.accepted
    assert a.robust_onset == 100
    assert "MODEL_DISAGREEMENT" in a.reject_reasons


def test_strong_p_model_can_win_over_weak_legacy():
    a = PWaveBeatAssessment(0, 100, 150, 100, 150, accepted=True,
        onset_confidence=.9, offset_confidence=.9, valid_leads=["I", "II", "V2", "V5"],
        valid_lead_groups=["limb", "precordial"])
    _reconcile_parallel_consensus_models(a, legacy_onset=140, legacy_offset=190,
                                        fs=500, evidence_arbitration=True, legacy_confidence=.2)
    assert a.accepted and a.robust_onset == 100 and a.robust_offset == 150


@pytest.mark.parametrize("state", ["AF_LIKE", "ORGANIZED_ATRIAL_ACTIVITY", "OVERLAP_UNCERTAIN"])
def test_p_correction_cannot_bypass_atrial_state_gate(state):
    feature = _feature("II")
    assessment = PWaveBeatAssessment(0, 110, 160, 110, 160, accepted=True, p_state=state)
    assert correct_p_boundaries([feature], [assessment], np.zeros((12, 800)), 500) == []
    assert feature.p.onset == 120


def test_p_correction_refreshes_dependent_measurements():
    feature = _feature("II")
    feature.p_onset_confidence = feature.p_offset_confidence = .2
    row = PWaveLeadBoundary("II", 110, 135, 160, onset_confidence=.9, offset_confidence=.9,
                           informative=True, hard_quality_pass=True)
    a = PWaveBeatAssessment(0, 110, 160, 110, 160, accepted=True, p_state="P_PRESENT",
                           valid_leads=["II"], per_lead={"II": row})
    t = np.arange(800)
    ecg = np.tile(.1 * np.exp(-.5 * ((t - 135)/10)**2), (12, 1))
    audit = correct_p_boundaries([feature], [a], ecg, 500)
    assert len(audit) == 1
    assert feature.p_dur_ms == 100
    assert feature.pr_ms == (225 - 110) * 2
    assert feature.p_area > 0 and feature.p_amp_mv > .09
    assert feature.p_onset_corrected_index == 110


@pytest.mark.parametrize("polarity", [-1, 1])
def test_t_change_point_preserves_nonzero_st_baseline(polarity):
    t = np.arange(101)
    signal = .18 + polarity * .004 * np.maximum(t - 40, 0)
    onset = t_onset_change_point(signal, 0, 100, 500)
    assert onset is not None and abs(onset - 40) <= 2
    assert t_onset_change_point(.18 + .004 * t, 0, 100, 500) is None


def test_qrs_candidate_does_not_extend_narrow_qrs_into_t():
    x = np.zeros(500)
    x[80:100] = 1
    x[130:170] = .2
    assert sustained_qrs_offset(x, 80, 100, 90, 500) == 100


def test_group_budget_does_not_force_an_unrelated_beat_into_dominant_group():
    vectors = [np.array([1., 0.]), np.array([.99, .01]), np.array([0., 1.])]
    old = _morphology_cluster([0, 1, 2], vectors, 1)
    new = _morphology_cluster([0, 1, 2], vectors, 1, preserve_outliers=True)
    assert old[0].members == [0, 1, 2]
    assert new[0].members == [0, 1] and new[1].members == [2]


@pytest.mark.parametrize("amplitude", [-.18, .18])
@pytest.mark.parametrize("drift", [-.04, 0, .04])
def test_adaptive_st_preserves_known_offset_and_linear_drift(amplitude, drift):
    raw = np.tile(_st_signal(st_j_mv=amplitude, st80_mv=amplitude), 5)
    raw += .4 + drift * np.arange(len(raw)) / 500
    anchors = _st_pr_anchors()
    for b in anchors:
        b.p.onset = None
        b.t = SimpleNamespace(offset=None)
        b.beat_id = b.qrs.onset // 1000
    result = adaptive_st_signal(np.tile(raw, (12, 1)), 500, anchors)
    measured = localize_st_robust(result.signal[1], fs=500, r_index=2400, qrs_onset=2380,
        qrs_offset=2450, p_offset=2345, t_onset=2540, next_r_index=3400,
        consensus_j_index=2450, consensus_support=10)
    assert measured.j_mv == pytest.approx(amplitude, abs=.003)
    assert measured.st80_mv == pytest.approx(amplitude, abs=.003)
    assert result.valid_mask[1, 2450]
    assert result.anchor_sources["II"]["PR"] == 5
    assert not result.valid_mask[1, -1]  # far beyond last observed anchor


def test_adaptive_st_does_not_invent_support_without_anchors():
    result = adaptive_st_signal(np.zeros((12, 500)), 500, [])
    assert not result.valid_mask.any()
    assert len(result.unavailable_leads) == 12
