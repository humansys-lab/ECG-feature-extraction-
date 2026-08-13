from __future__ import annotations

from feature_extraction.ecgfeat.glasgow_rules.context import build_context
from feature_extraction.ecgfeat.glasgow_rules.preliminary import (
    preliminary_base_rules,
    restricted_analysis,
)
from feature_extraction.ecgfeat.models import PatientMeta
from tests.glasgow_test_helpers import make_features


def _results(context):
    return {rule.rule_id: rule.evaluator(context) for rule in preliminary_base_rules()}


def test_missing_age_and_sex_emit_only_combined_warning() -> None:
    results = _results(build_context(make_features(PatientMeta())))

    assert results["GAN-04.03-05"].evaluation_status == "matched"
    assert results["GAN-04.03-05"].resolution_status == "advisory"
    assert results["GAN-04.03-03"].evaluation_status == "not_applicable"
    assert results["GAN-04.03-04"].evaluation_status == "not_applicable"


def test_pediatric_route_emits_advisory_below_18() -> None:
    context = build_context(make_features(PatientMeta(age=17.99, sex="Female")))
    result = _results(context)["GAN-04.03-06"]

    assert result.statement == "--- Interpretation based on pediatric criteria ---"
    assert result.evaluation_status == "matched"


def test_missing_lead_names_every_excluded_lead() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    for lead in ("V3", "V6"):
        features.quality[lead].missing = True
        features.quality[lead].reliable = False
    result = _results(build_context(features))["GAN-04.01-07"]

    assert result.evidence["leads"] == ["V3", "V6"]
    assert result.statement == "Lead(s) unsuitable for analysis: V3, V6"


def test_measurement_error_uses_absolute_p_component_over_one_mv() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.representative_leads["II"].params["p_positive_amp_mv"] = 1.01
    result = _results(build_context(features))["GAN-04.01-08"]

    assert result.evaluation_status == "matched"
    assert result.evidence["max_abs_p_component_mv"] == 1.01
    assert result.evidence["lead"] == "II"


def test_pacemaker_stops_qrs_t_morphology_but_not_rate() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.global_features.paced_rhythm = True
    results = list(_results(build_context(features)).values())

    restricted = restricted_analysis(results)

    assert restricted["stop_categories"] == ["qrs_t_morphology"]
    assert restricted["stopped_by"] == ["GAN-04.04-01"]


def test_invalid_medication_requires_unknown_plus_another_value() -> None:
    invalid = build_context(
        make_features(PatientMeta(age=59, sex="Female", meds=["unknown", "digoxin"]))
    )
    valid = build_context(
        make_features(PatientMeta(age=59, sex="Female", meds=["unknown"]))
    )

    assert _results(invalid)["GAN-04.03-02"].evaluation_status == "matched"
    assert _results(valid)["GAN-04.03-02"].evaluation_status == "not_matched"


def test_q3_record_quality_emits_technical_error_advisory() -> None:
    features = make_features(PatientMeta(age=59, sex="Female"))
    features.metadata["record_quality"]["record_grade"] = "Q3"
    result = _results(build_context(features))["GAN-04.04-04"]

    assert result.evaluation_status == "matched"
    assert result.summary_code == 6


def test_similar_precordial_qrs_is_explicitly_unavailable_without_validated_fact() -> None:
    result = _results(
        build_context(make_features(PatientMeta(age=59, sex="Female")))
    )["GAN-04.04-03"]

    assert result.evaluation_status == "unavailable"
    assert result.fidelity == "not_reproducible_from_guide"
    assert result.missing_inputs == ["metadata.similar_precordial_qrs"]
