from __future__ import annotations

import numpy as np
import pytest

from feature_extraction.ecgfeat.api import ECGFeatureExtractor
from feature_extraction.ecgfeat.preprocess import resample_ecg
from feature_extraction.ecgfeat.quality import _rolling_mad_sigma, detect_pacing_spikes
from feature_extraction.ecgfeat.validation import ECGInputError, validate_ecg_input
from tests.optional import needs_interpretation


@pytest.mark.parametrize("fs", [0.0, -500.0, np.nan, np.inf, "invalid"])
def test_extract_rejects_invalid_sampling_rate_with_reason_code(fs: object) -> None:
    with pytest.raises(ECGInputError) as caught:
        ECGFeatureExtractor().extract(np.zeros((12, 500)), fs=fs)

    assert caught.value.code == "invalid_sampling_rate"
    assert caught.value.to_dict()["details"]["field"] == "fs"


def test_validate_input_rejects_nonfinite_samples() -> None:
    ecg = np.zeros((12, 500), dtype=float)
    ecg[3, 100] = np.nan
    ecg[4, 200] = np.inf

    with pytest.raises(ECGInputError) as caught:
        validate_ecg_input(ecg, fs=500)

    assert caught.value.code == "nonfinite_signal"
    assert caught.value.details["nonfinite_count"] == 2


def test_validate_input_rejects_subsecond_record() -> None:
    with pytest.raises(ECGInputError) as caught:
        validate_ecg_input(np.zeros((12, 499)), fs=500)

    assert caught.value.code == "record_too_short"
    assert caught.value.details["minimum_duration_sec"] == 1.0


def test_validate_input_normalizes_samples_by_leads_layout() -> None:
    ecg, fs_run, contract = validate_ecg_input(np.zeros((500, 12)), fs=500)

    assert ecg.shape == (12, 500)
    assert fs_run == 500
    assert contract["transposed"] is True
    assert contract["duration_sec"] == 1.0


def test_noninteger_input_rate_is_resampled_to_declared_internal_timebase() -> None:
    ecg = np.zeros((12, 2575), dtype=float)

    result = resample_ecg(ecg, fs_in=257.5, fs_out=258)

    assert result.shape == (12, 2580)


@pytest.mark.parametrize("fs", [100, 125, 200])
def test_pacing_detection_is_nyquist_safe_at_supported_low_rates(fs: int) -> None:
    result = detect_pacing_spikes(np.zeros((12, fs * 2), dtype=float), fs=fs)

    assert result["state"] == "off"
    assert result["spike_times"] == []


@needs_interpretation
def test_full_pipeline_handles_minimum_supported_sampling_rate() -> None:
    result = ECGFeatureExtractor(enable_lead_reversal=False).extract(
        np.zeros((12, 200), dtype=float),
        fs=100,
    )

    assert result.fs == 100
    assert result.metadata["duration_sec"] == 2.0
    assert result.metadata["record_quality"]["record_grade"] == "Q3"
    assert result.metadata["quality_stages"]["raw"]["record_grade"] == "Q3"


def test_vectorized_rolling_mad_matches_reference_implementation() -> None:
    values = np.random.default_rng(42).normal(size=101)
    window = 10
    win = 11
    padded = np.pad(values, (win // 2, win // 2), mode="edge")
    expected = []
    for idx in range(values.size):
        segment = padded[idx:idx + win]
        median = float(np.median(segment))
        expected.append(1.4826 * float(np.median(np.abs(segment - median))))
    expected_arr = np.asarray(expected)
    positive = expected_arr[expected_arr > 1e-9]
    fallback = float(np.median(positive)) if positive.size else 1e-6
    expected_arr = np.maximum(expected_arr, fallback)

    np.testing.assert_allclose(_rolling_mad_sigma(values, window), expected_arr)
