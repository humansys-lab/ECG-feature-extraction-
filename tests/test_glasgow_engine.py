from __future__ import annotations

from collections import Counter

import pytest

from feature_extraction.ecgfeat.glasgow import build_glasgow_payload
from feature_extraction.ecgfeat.glasgow_rules import rate
from feature_extraction.ecgfeat.glasgow_rules.engine import (
    analyze_glasgow,
    build_foundation_registry,
)
from feature_extraction.ecgfeat.glasgow_rules.models import GlasgowConfig
from feature_extraction.ecgfeat.models import PatientMeta
from tests.glasgow_test_helpers import make_features


def test_foundation_registry_has_complete_chapter_counts() -> None:
    registry = build_foundation_registry()
    counts = Counter(rule.chapter.split(".")[0] for rule in registry.rules)

    assert counts == {"4": 23, "5": 3, "6": 6, "18": 1, "19": 34}
    assert len(registry.rules) == 67


def test_engine_keeps_evaluation_and_resolution_separate() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.global_features.heart_rate_bpm = 110.0
    features.global_features.qt_ms = 350.0

    analysis = analyze_glasgow(features).to_dict()
    tachy = next(item for item in analysis["rule_evaluations"] if item["rule_id"] == "GAN-05.01-01")

    assert tachy["evaluation_status"] == "matched"
    assert tachy["resolution_status"] == "final"
    assert analysis["statement_resolution"]["summary_code"]["code"] == 2
    assert len(analysis["measurement_matrix"]["rows"]) == 34


def test_engine_fail_closed_marks_evaluator_error_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate, "evaluate_tachycardia", lambda _context: 1 / 0)

    analysis = analyze_glasgow(
        make_features(PatientMeta(age=59, sex="Female")),
        GlasgowConfig(strict=False),
    ).to_dict()
    item = next(row for row in analysis["rule_evaluations"] if row["rule_id"] == "GAN-05.01-01")

    assert item["evaluation_status"] == "unavailable"
    assert item["evaluation_error"] == "ZeroDivisionError"
    assert analysis["statement_resolution"]["summary_code"]["code"] == 6


def test_strict_engine_reraises_evaluator_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate, "evaluate_tachycardia", lambda _context: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        analyze_glasgow(
            make_features(PatientMeta(age=59, sex="Female")),
            GlasgowConfig(strict=True),
        )


def test_facade_uses_cached_v2_analysis_and_preserves_compatibility_projection() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    cached = analyze_glasgow(features).to_dict()
    features.metadata["glasgow_analysis"] = cached

    payload = build_glasgow_payload(features)

    assert payload["schema_version"] == "glasgow_rules.v2"
    assert payload["summary_code"] == payload["statement_resolution"]["summary_code"]
    assert payload["qtc_statement_guard"]["source"] == "glasgow_qtc_omission_rule"
    assert payload["statement_catalog"]["summary_codes"]


def test_coverage_marks_deferred_chapters_without_claiming_implementation() -> None:
    coverage = analyze_glasgow(make_features(PatientMeta(age=59, sex="Female"))).to_dict()["coverage"]

    assert coverage["foundation"]["registered_rule_total"] == 67
    assert coverage["chapters"]["3"]["linked_measurement_definitions"] == 34
    assert all(coverage["chapters"][str(chapter)]["status"] == "pending_phase" for chapter in range(7, 18))

