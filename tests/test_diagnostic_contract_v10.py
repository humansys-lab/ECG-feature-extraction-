from types import SimpleNamespace

import numpy as np
import pytest

from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation
from feature_extraction.ecgfeat.clinical_rules.resolver import ClinicalStatementResolver
from feature_extraction.ecgfeat.clinical_rules.serial import evaluate_serial_comparison
from feature_extraction.ecgfeat.quality import build_diagnostic_gate
from feature_extraction.ecgfeat.validation import ECGInputError, validate_ecg_input


def test_independent_eight_leads_are_reconstructed_in_standard_order() -> None:
    fs = 250
    lead_i = np.linspace(-0.2, 0.2, fs)
    lead_ii = np.linspace(0.1, 0.5, fs)
    chest = [np.full(fs, 0.1 * index) for index in range(1, 7)]
    signal = np.stack([lead_i, lead_ii, *chest])

    normalized, _, contract = validate_ecg_input(
        signal,
        fs,
        lead_names=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    )

    assert normalized.shape == (12, fs)
    np.testing.assert_allclose(normalized[2], lead_ii - lead_i)
    np.testing.assert_allclose(normalized[3], -(lead_i + lead_ii) / 2.0)
    assert contract["derived_leads"] == ["III", "aVR", "aVL", "aVF"]
    assert contract["requires_original_500_hz"] is False


def test_amplitude_units_are_normalized_to_millivolts() -> None:
    signal_uv = np.full((12, 250), 1000.0)
    normalized, _, contract = validate_ecg_input(
        signal_uv,
        250,
        amplitude_unit="uV",
    )
    np.testing.assert_allclose(normalized, 1.0)
    assert contract["amplitude_scale_to_mv"] == pytest.approx(0.001)


def test_digital_samples_require_gain() -> None:
    with pytest.raises(ECGInputError) as raised:
        validate_ecg_input(
            np.ones((12, 250)),
            250,
            amplitude_unit="digital",
        )
    assert raised.value.code == "missing_digital_gain"


def test_diagnostic_gate_does_not_require_original_500_hz() -> None:
    gate = build_diagnostic_gate(
        record_quality={"diagnostic_gate": "pass"},
        duration_sec=10.0,
        n_beats=10,
        beat_groups={1: list(range(10))},
    )
    assert gate["state"] == "pass"
    assert gate["requires_original_500_hz"] is False


def test_resolver_emits_fixed_confidence_priority_review_and_snomed() -> None:
    result = ClinicalStatementResolver().resolve(
        evaluations=[
            RuleEvaluation(
                rule_id="TEST-QT",
                domain="intervals",
                status="matched",
                statement_code="prolonged_qt",
                statement="Prolonged QT interval",
                severity="abnormal",
                confidence="moderate",
            )
        ],
        required_domains={"intervals"},
        available_domains={"intervals"},
    )
    statement = result.final_statements[0]
    assert statement["confidence"] == "MEDIUM"
    assert statement["priority"] == "P2"
    assert statement["human_review_required"] is True
    assert statement["standard_codes"]["SNOMED_CT"] == "111975006"
    assert statement["report_statement"].startswith("Consider ")


def _serial_features(*, qrs: float, rhythm_af: bool = False):
    return SimpleNamespace(
        global_features=SimpleNamespace(
            pr_ms=160.0,
            qrs_ms=qrs,
            qt_ms=400.0,
            qrs_axis_deg=30.0,
            p_axis_deg=None if rhythm_af else 45.0,
            paced_rhythm=False,
        ),
        metadata={
            "rhythm_analysis": {
                "af_afl_summary": {
                    "probable_af": rhythm_af,
                    "probable_flutter": False,
                }
            }
        },
        representative_leads={},
    )


def test_serial_comparison_detects_rhythm_and_qrs_change() -> None:
    prior = _serial_features(qrs=90.0)
    current = _serial_features(qrs=125.0, rhythm_af=True)
    result = evaluate_serial_comparison(current, prior)[0]
    assert result.status == "matched"
    assert result.priority == "P1"
    assert result.evidence["rhythm_changed"] is True
    assert result.evidence["significant_metric_changes"]["qrs_ms"]["delta"] == 35.0
