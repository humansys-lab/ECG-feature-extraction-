from __future__ import annotations

import pytest

from feature_extraction.ecgfeat.glasgow_rules.models import (
    GlasgowConfig,
    RuleEvaluation,
    RuleSpec,
)
from feature_extraction.ecgfeat.glasgow_rules.registry import RuleRegistry


def _never_matches(_context):
    return RuleEvaluation(
        rule_id="GAN-05.01-01",
        evaluation_status="not_matched",
        resolution_status="no_statement",
        statement="Tachycardia",
    )


def _spec(rule_id: str = "GAN-05.01-01", order: int = 50101) -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id,
        chapter="5.1",
        pdf_page=15,
        category="rate",
        order=order,
        statement="Tachycardia",
        required_inputs=("global.heart_rate_bpm",),
        fidelity="glasgow_explicit",
        source_kind="glasgow_guide",
        source_reference="Physician's Guide section 5.1, PDF page 15",
        evaluator=_never_matches,
        summary_code=2,
    )


def test_config_defaults_match_approved_foundation_contract() -> None:
    config = GlasgowConfig()

    assert config.qtc_formula == "hodges"
    assert config.adult_tachycardia_bpm == 100.0
    assert config.adult_bradycardia_bpm == 50.0
    assert config.paper_speed_mm_per_s == 25.0
    assert config.gain_mm_per_mv == 10.0
    assert config.strict is False


def test_registry_rejects_duplicate_rule_ids() -> None:
    registry = RuleRegistry()
    registry.register(_spec())

    with pytest.raises(ValueError, match="duplicate Glasgow rule id"):
        registry.register(_spec())


def test_registry_sorts_by_order_then_rule_id() -> None:
    registry = RuleRegistry(
        [
            _spec("GAN-05.01-02", order=50102),
            _spec("GAN-05.01-01", order=50101),
        ]
    )

    assert [rule.rule_id for rule in registry.rules] == [
        "GAN-05.01-01",
        "GAN-05.01-02",
    ]


def test_registry_requires_source_metadata() -> None:
    registry = RuleRegistry()
    invalid = RuleSpec(
        rule_id="GAN-05.01-01",
        chapter="5.1",
        pdf_page=15,
        category="rate",
        order=50101,
        statement="Tachycardia",
        required_inputs=(),
        fidelity="glasgow_explicit",
        source_kind="glasgow_guide",
        source_reference="",
        evaluator=_never_matches,
    )

    with pytest.raises(ValueError, match="source_reference"):
        registry.register(invalid)


def test_registry_rejects_invalid_fidelity() -> None:
    registry = RuleRegistry()
    invalid = RuleSpec(
        rule_id="GAN-05.01-01",
        chapter="5.1",
        pdf_page=15,
        category="rate",
        order=50101,
        statement="Tachycardia",
        required_inputs=(),
        fidelity="exactish",
        source_kind="glasgow_guide",
        source_reference="section 5.1",
        evaluator=_never_matches,
    )

    with pytest.raises(ValueError, match="invalid fidelity"):
        registry.register(invalid)


def test_evaluation_serialization_keeps_fidelity_separate_from_availability() -> None:
    result = RuleEvaluation(
        rule_id="GAN-06.02-01",
        evaluation_status="unavailable",
        resolution_status="no_statement",
        statement="Borderline prolonged QT interval",
        required_inputs=["global.qt_ms"],
        missing_inputs=["global.qt_ms"],
        fidelity="glasgow_explicit",
        source={"kind": "glasgow_guide", "reference": "section 6.2"},
    ).to_dict()

    assert result["evaluation_status"] == "unavailable"
    assert result["fidelity"] == "glasgow_explicit"
    assert result["missing_inputs"] == ["global.qt_ms"]
