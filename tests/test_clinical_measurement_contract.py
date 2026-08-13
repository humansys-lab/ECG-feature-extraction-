from __future__ import annotations

import pytest

from feature_extraction.ecgfeat.clinical_rules.hypertrophy import evaluate_hypertrophy
from feature_extraction.ecgfeat.clinical_rules.ischemia import evaluate_sgarbossa
from feature_extraction.ecgfeat.features import build_representative_lead_features
from feature_extraction.ecgfeat.models import STANDARD_12_LEADS
from tests.test_lead_reversal import _make_beat_feature, _make_quality


def _quality_map():
    return {lead: _make_quality(lead) for lead in STANDARD_12_LEADS}


class ContractContext:
    def __init__(self, leads):
        self._leads = leads
        self._global = {"qrs_ms": 90.0, "qrs_axis_deg": 20.0}
        self.age_years = 50.0
        self.sex = "female"

    def global_value(self, name):
        return self._global.get(name)

    def lead_available(self, lead, reliability):
        return lead in self._leads

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        if not self.lead_available(lead, reliability):
            return None
        value = self._leads[lead].get(name)
        return value if isinstance(value, (int, float)) else None

    def lead_raw_value(self, lead, name, reliability="reliable_for_qrs"):
        if not self.lead_available(lead, reliability):
            return None
        return self._leads[lead].get(name)


def _by_code(evaluations, code):
    return next(
        item
        for item in evaluations
        if item.statement_code == code or item.evidence.get("evaluates_code") == code
    )


def test_representative_contract_exports_r_prime_amplitude() -> None:
    beat = _make_beat_feature("V1", 0, 0.7)
    beat.r_prime_amp_mv = 0.42
    beat.r_prime_duration_ms = 46.0

    representatives = build_representative_lead_features([beat], _quality_map())

    assert representatives["V1"].params["r_prime_amp_mv"] == pytest.approx(0.42)


def test_representative_contract_exports_twelve_sl_signed_qrs_area() -> None:
    beat = _make_beat_feature("I", 0, 0.7)
    beat.twelve_sl_qrs_signed_area_uv_ms = 125.0

    representatives = build_representative_lead_features([beat], _quality_map())

    assert representatives["I"].params["qrs_signed_area_uv_ms"] == pytest.approx(125.0)


def test_rae_consumes_consensus_p_duration() -> None:
    context = ContractContext(
        {
            "II": {"p_amp_mv": 0.30, "p_dur_consensus_ms": 105.0},
            "V1": {"p_terminal_amp_mv": -0.02, "p_terminal_duration_ms": 20.0},
        }
    )

    rae = _by_code(evaluate_hypertrophy(context, []), "right_atrial_abnormality")

    assert rae.status == "matched"


def test_sgarbossa_accepts_native_signed_area_polarity_fallback() -> None:
    context = ContractContext({"I": {"st_on_mv": 0.12, "qrs_signed_area": 0.25}})
    context._global["qrs_ms"] = 150.0

    result = evaluate_sgarbossa(context, paced=False)

    assert result.status == "matched"
    assert result.statement_code == "sgarbossa_positive"
