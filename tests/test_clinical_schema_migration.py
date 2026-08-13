from __future__ import annotations

from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation


def test_indeterminate_rule_can_explain_missing_decisive_input() -> None:
    result = RuleEvaluation(
        rule_id="CLIN-TEST-INDET",
        domain="intervals",
        status="indeterminate",
        missing_inputs=["global.qt_ms"],
        coverage="unavailable",
    )

    assert result.status == "indeterminate"
    assert result.coverage == "unavailable"


def test_legacy_unavailable_row_projects_to_new_dimensions() -> None:
    result = RuleEvaluation.from_dict(
        {
            "rule_id": "CLIN-TEST-LEGACY",
            "domain": "intervals",
            "status": "unavailable",
        }
    )

    assert result.status == "indeterminate"
    assert result.coverage == "unavailable"
    assert result.confidence == "unavailable"


def test_legacy_optional_rule_projects_normality_role() -> None:
    result = RuleEvaluation.from_dict(
        {
            "rule_id": "CLIN-TEST-OPTIONAL",
            "domain": "high_risk_patterns",
            "status": "suppressed",
            "normality_required": False,
        }
    )

    assert result.normality_role == "optional_screen"
    assert result.status == "indeterminate"
    assert result.coverage == "partial"
