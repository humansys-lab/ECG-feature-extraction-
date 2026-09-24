import numpy as np
import pytest

from feature_extraction.ecgfeat import ECGFeatureExtractor, ECGInputError
from feature_extraction.ecgfeat.validation import validate_ecg_input
from feature_extraction.ecgfeat.quality import compute_quality, summarize_record_quality
from tests.optional import needs_interpretation


def signal(fs=500):
    t = np.arange(10*fs)/fs
    x = np.zeros_like(t)
    for r in np.arange(.6, 9.7, .8):
        x += .12*np.exp(-.5*((t-r+.18)/.025)**2)
        x += 1.2*np.exp(-.5*((t-r)/.012)**2)
        x -= .2*np.exp(-.5*((t-r-.025)/.014)**2)
        x += .25*np.exp(-.5*((t-r-.28)/.045)**2)
    second = .8*x
    for r in np.arange(.6, 9.7, .8):
        second -= .15*np.exp(-.5*((t-r-.28)/.045)**2)
    return np.array([x, second])


def test_limited_input_requires_explicit_opt_in_and_names():
    x = signal()
    with pytest.raises(ECGInputError):
        validate_ecg_input(x, 500, lead_names=["channel_0", "channel_1"])
    with pytest.raises(ECGInputError):
        validate_ecg_input(x, 500, allow_limited_leads=True)
    with pytest.raises(ECGInputError):
        validate_ecg_input(x, 500, lead_names=["II", "II"], allow_limited_leads=True)
    with pytest.raises(ECGInputError):
        validate_ecg_input(x, 500, lead_names=["II", " "], allow_limited_leads=True)


def test_limited_calibration_mapping_and_no_derived_leads():
    x, fs, contract = validate_ecg_input(signal().T*1000, 500, lead_names=["V1", "unknown"],
                                        amplitude_unit="uV", allow_limited_leads=True)
    assert np.allclose(x[6], signal()[0], atol=1e-12)
    assert np.allclose(x[1], signal()[1])
    assert set(contract["available_leads"]) == {"II", "V1"}
    assert not contract["derived_leads"]
    assert contract["limb_lead_consistency"] is None
    assert np.count_nonzero(np.std(x, axis=1)) == 2


def test_explicit_absence_is_distinct_from_bad_measured_channel():
    x, _, _ = validate_ecg_input(signal(), 500, lead_names=["II", "V2"], allow_limited_leads=True)
    quality = compute_quality(x, 500)
    assert summarize_record_quality(quality)["record_grade"] == "Q3"
    limited = summarize_record_quality(quality, available_leads=["II", "V2"])
    assert "record" not in limited["rejected_functions"]
    assert "record" in summarize_record_quality(compute_quality(x*0, 500), available_leads=["II", "V2"])["rejected_functions"]


@needs_interpretation
def test_limited_pipeline_accepts_clean_p_without_fabricating_twelve_lead_report():
    result = ECGFeatureExtractor(input_mode="limited", fs_internal=500).extract(
        signal(), fs=500, lead_names=["channel_0", "channel_1"])
    assert len(result.beats) >= 10
    assert any(p.accepted for p in result.p_wave_assessments)
    assert set(result.quality) == {"II", "V2"}
    assert {b.lead for b in result.beat_features} == {"II", "V2"}
    assert result.global_features.qrs_axis_deg is None
    assert result.global_features.qrs_t_angle_deg is None
    assert result.global_features.transition_zone is None
    assert not result.global_features.t_axis_reliable
    assert result.global_features.ptf_v1_mv_ms is None
    assert not result.global_features.qt_reportable
    assert result.interpretation is None
    assert result.metadata["limited_lead_capabilities"]["mode"] == "measurements_only"
    from feature_extraction.ecgfeat.export import to_dict, build_structured_payload
    for exported in (to_dict(result), to_dict(result, profile="summary"), build_structured_payload(result)):
        assert exported["morphology_inputs"]["available"] is False
        assert exported["statement_engine"]["final_statements"] == []
        assert exported["clinical_interpretation"]["status"] == "not_applicable_limited_lead_measurements"
    assert result.metadata["diagnostic_gate"]["allowed_domains"] == []
