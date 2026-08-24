from __future__ import annotations

from copy import deepcopy

from feature_extraction.ecgfeat.glasgow_rules.context import build_context
from feature_extraction.ecgfeat.glasgow_rules.models import GlasgowConfig
from feature_extraction.ecgfeat.models import PatientMeta
from tests.glasgow_test_helpers import make_features


def test_exact_age_days_takes_precedence_over_year_conversion() -> None:
    features = make_features(meta=PatientMeta(age=0.04, age_days=14, sex="Male"))

    context = build_context(features, GlasgowConfig())

    assert context.patient.age_days == 14.0
    assert context.patient.age_years == 14.0 / 365.25
    assert context.patient.age_days_source == "provided"
    assert context.patient.pediatric is True


def test_invalid_explicit_age_days_does_not_use_conflicting_year_age() -> None:
    features = make_features(meta=PatientMeta(age=40.0, age_days=-1.0, sex="Male"))

    context = build_context(features, GlasgowConfig())

    assert context.patient.age_days is None
    assert context.patient.age_years is None
    assert context.patient.age_missing is True


def test_age_years_are_converted_to_days_and_marked_approximate() -> None:
    features = make_features(meta=PatientMeta(age=10.0, sex="Female"))

    context = build_context(features, GlasgowConfig())

    assert context.patient.age_days == 3652.5
    assert context.patient.age_days_source == "derived_from_age_years"
    assert context.patient.age_approximation is True


def test_missing_demographics_use_confirmed_fallback_without_mutating_features() -> None:
    features = make_features(meta=PatientMeta())
    before = deepcopy(features.metadata)

    context = build_context(features, GlasgowConfig())

    assert context.patient.algorithm_age_group == "adult"
    assert context.patient.age_missing is True
    assert context.patient.sex_missing is True
    assert context.patient.qtc_demographic_route == "sex_neutral_existing_dxl"
    assert features.metadata == before


def test_context_exposes_reliable_lead_facts_and_exclusions_separately() -> None:
    features = make_features(meta=PatientMeta(age=59, sex="Female"))
    features.quality["V2"].reliable = False

    context = build_context(features, GlasgowConfig())

    assert context.leads["I"].available is True
    assert context.leads["V2"].available is False
    assert context.excluded_leads == {"V2": ["lead_quality_unreliable"]}


def test_with_exclusions_returns_new_context_and_keeps_original_unchanged() -> None:
    context = build_context(
        make_features(meta=PatientMeta(age=59, sex="Female")),
        GlasgowConfig(),
    )

    updated = context.with_exclusions({"V3": ["GAN-04.01-01"]})

    assert context.excluded_leads == {}
    assert updated.excluded_leads == {"V3": ["GAN-04.01-01"]}
