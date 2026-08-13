import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARTIFACTS = (
    ROOT / "JS00059_features.json",
    ROOT / "JS00059_report.txt",
)
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_ARTIFACTS),
    reason="run `python demo_feature_extraction.py JS00059` to generate regression artifacts",
)


def _payload() -> dict:
    return json.loads((ROOT / "JS00059_features.json").read_text(encoding="utf-8"))


def _final_codes(clinical: dict) -> set[str]:
    return {
        item.get("statement_code")
        for item in clinical.get("final_statements", [])
        if isinstance(item, dict) and item.get("statement_code")
    }


def _rule(clinical: dict, domain: str, rule_id: str) -> dict:
    return next(
        item
        for item in clinical["domains"][domain]
        if item.get("rule_id") == rule_id
    )


def test_js00059_unified_result_has_no_stale_contradictions() -> None:
    clinical = _payload()["clinical_interpretation"]
    qt = _rule(clinical, "intervals", "CLIN-INTERVAL-QT-01")

    assert qt["evidence"]["primary_formula"] == "hodges"
    assert qt["statement_code"] is None
    assert qt["evidence"]["formula_values_ms"]["bazett_ms"] > 480.0
    assert qt["evidence"]["selected_qtc_ms"] < 460.0
    assert "rvh_pattern" not in _final_codes(clinical)
    assert "posterior_ischemia_pattern" not in _final_codes(clinical)
    assert "prior_infarct_q_wave_pattern" not in _final_codes(clinical)
    assert "atrial_flutter_pattern" not in _final_codes(clinical)
    assert "tachycardia" in _final_codes(clinical)
    assert clinical["overall_status"] != "normal"


def test_js00059_artifacts_share_fingerprint() -> None:
    clinical = _payload()["clinical_interpretation"]
    fingerprint = clinical["artifact_fingerprint"]
    report = (ROOT / "JS00059_report.txt").read_text(encoding="utf-8")

    assert fingerprint.startswith("sha256:")
    assert fingerprint in report
    assert "STALE ARTIFACT" not in report
    assert "UNIFIED CLINICAL INTERPRETATION" in report
