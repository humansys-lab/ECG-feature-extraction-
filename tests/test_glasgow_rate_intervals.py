from __future__ import annotations

import pytest

from feature_extraction.ecgfeat.glasgow_rules.context import build_context
from feature_extraction.ecgfeat.glasgow_rules.intervals import interval_rules, qtc_values
from feature_extraction.ecgfeat.glasgow_rules.models import GlasgowConfig
from feature_extraction.ecgfeat.glasgow_rules.rate import (
    bradycardia_limit,
    rate_rules,
    tachycardia_limit,
)
from feature_extraction.ecgfeat.models import PatientMeta
from tests.glasgow_test_helpers import make_features


@pytest.mark.parametrize(
    ("age_days", "tachy", "brady"),
    [
        (0, 163.0, 88.0),
        (14, 171.5, 96.5),
        (28, 180.0, 105.0),
        (29, 180.0, 105.0),
    ],
)
def test_neonatal_rate_limits(age_days: float, tachy: float, brady: float) -> None:
    config = GlasgowConfig()

    assert tachycardia_limit(age_days, config) == pytest.approx(tachy)
    assert bradycardia_limit(age_days, config) == pytest.approx(brady)


def test_adult_rate_limits_use_config_defaults() -> None:
    config = GlasgowConfig()

    assert tachycardia_limit(18 * 365.25, config) == 100.0
    assert bradycardia_limit(18 * 365.25, config) == 50.0


def test_all_four_qtc_formulae_use_consistent_units() -> None:
    values = qtc_values(qt_ms=400.0, hr_bpm=100.0)

    assert values["bazett_ms"] == pytest.approx(516.3978, abs=0.01)
    assert values["fridericia_ms"] == pytest.approx(474.2516, abs=0.01)
    assert values["hodges_ms"] == pytest.approx(470.0, abs=0.01)
    assert values["framingham_ms"] == pytest.approx(461.6, abs=0.01)


def test_qtc_at_sixty_bpm_equals_measured_qt_for_all_formulae() -> None:
    values = qtc_values(400.0, 60.0)

    assert all(value == pytest.approx(400.0) for value in values.values())


def test_rate_rules_match_tachycardia_and_assign_rate_only_summary() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.global_features.heart_rate_bpm = 101.0
    context = build_context(features)

    result = next(rule.evaluator(context) for rule in rate_rules() if rule.rule_id == "GAN-05.01-01")

    assert result.evaluation_status == "matched"
    assert result.resolution_status == "final"
    assert result.summary_code == 2
    assert result.thresholds["tachycardia_limit_bpm"] == 100.0


def test_marked_sinus_bradycardia_requires_sinus_rhythm() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.global_features.heart_rate_bpm = 39.0
    features.metadata["rhythm_analysis"]["rule_summary"]["primary_statement"] = "junctional_rhythm"
    context = build_context(features)

    result = next(rule.evaluator(context) for rule in rate_rules() if rule.rule_id == "GAN-05.03-01")

    assert result.evaluation_status == "not_applicable"
    assert result.evidence["reason"] == "non_sinus_rhythm"


def test_pr_rules_apply_exact_age_equations_and_priority() -> None:
    features = make_features(PatientMeta(age=10.0, age_days=3650, sex="Female"))
    features.global_features.pr_ms = 180.0
    context = build_context(features)
    results = {
        rule.rule_id: rule.evaluator(context)
        for rule in interval_rules()
        if rule.chapter == "6.1"
    }

    assert results["GAN-06.01-01"].thresholds["lower_limit_ms"] == pytest.approx(96.9)
    assert results["GAN-06.01-02"].evaluation_status == "not_matched"
    assert results["GAN-06.01-03"].evaluation_status == "matched"
    assert results["GAN-06.01-03"].thresholds["borderline_limit_ms"] == pytest.approx(174.755)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("p", "p_wave_flag_not_set"),
        ("sinus", "non_sinus_rhythm"),
        ("wpw", "wpw_pattern"),
    ],
)
def test_pr_omission_is_not_reported_as_normal(mutation: str, reason: str) -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    if mutation == "p":
        for quality in features.quality.values():
            quality.reliable_for_p = False
    elif mutation == "sinus":
        features.metadata["rhythm_analysis"]["rule_summary"]["primary_statement"] = "junctional_rhythm"
    else:
        features.metadata["rhythm_analysis"]["rule_summary"]["preexcitation"] = {"wpw_pattern": True}
    context = build_context(features)

    results = [rule.evaluator(context) for rule in interval_rules() if rule.chapter == "6.1"]

    assert all(item.evaluation_status == "not_applicable" for item in results)
    assert all(item.evidence["omitted_by"] == [reason] for item in results)


@pytest.mark.parametrize(
    ("qrs_ms", "hr_bpm", "reason"),
    [
        (120.0, 80.0, "qrs_duration_ge_120"),
        (90.0, 125.1, "heart_rate_gt_125"),
    ],
)
def test_qtc_guard_keeps_values_but_omits_statements(
    qrs_ms: float,
    hr_bpm: float,
    reason: str,
) -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.global_features.qrs_ms = qrs_ms
    features.global_features.heart_rate_bpm = hr_bpm
    features.global_features.qt_ms = 430.0
    context = build_context(features)

    results = [rule.evaluator(context) for rule in interval_rules() if rule.chapter == "6.2"]

    assert all(item.evaluation_status == "not_applicable" for item in results)
    assert all(reason in item.evidence["omitted_by"] for item in results)
    assert all("qtc_values_ms" in item.evidence for item in results)


def test_missing_sex_uses_approved_neutral_thresholds() -> None:
    features = make_features(PatientMeta(age=60.0, sex=None))
    features.global_features.qt_ms = 465.0
    features.global_features.heart_rate_bpm = 60.0
    context = build_context(features)

    result = next(
        rule.evaluator(context)
        for rule in interval_rules()
        if rule.rule_id == "GAN-06.02-01"
    )

    assert result.evaluation_status == "matched"
    assert result.thresholds == {
        "borderline_min_ms": 465.0,
        "prolonged_min_ms": 485.0,
    }
    assert result.fidelity == "existing_dxl_approximation"
