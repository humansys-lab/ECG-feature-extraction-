import pytest

from feature_extraction.ecgfeat.clinical_rules.intervals import (
    classify_adult_pr,
    classify_adult_qt,
    classify_pediatric_pr,
    evaluate_qt_values,
    qtc_values,
)


def test_js00059_rate_does_not_use_bazett_as_final() -> None:
    values = qtc_values(qt_ms=376.5, hr_bpm=102.040816)
    result = classify_adult_qt(values, sex="female", reliable=True)

    assert values["bazett_ms"] > 480
    assert values["hodges_ms"] < 460
    assert result.statement_code is None
    assert result.status == "not_matched"
    assert result.evidence["primary_formula"] == "hodges"


def test_qtc_values_include_all_named_formulas() -> None:
    values = qtc_values(qt_ms=400.0, hr_bpm=60.0)

    assert values == pytest.approx(
        {
            "bazett_ms": 400.0,
            "fridericia_ms": 400.0,
            "hodges_ms": 400.0,
            "framingham_ms": 400.0,
        }
    )


@pytest.mark.parametrize(
    ("qtc_ms", "code", "severity"),
    [
        (340.0, "markedly_short_qt", "abnormal"),
        (350.0, "possible_short_qt_pattern", "borderline"),
        (380.0, "borderline_short_qt", "borderline"),
    ],
)
def test_short_qt_uses_evidence_tiers(qtc_ms, code, severity) -> None:
    values = {
        "bazett_ms": qtc_ms,
        "fridericia_ms": qtc_ms,
        "hodges_ms": qtc_ms,
        "framingham_ms": qtc_ms,
    }

    result = classify_adult_qt(values, sex="female", reliable=True)

    assert result.statement_code == code
    assert result.severity == severity


def test_qtc_391_is_not_short() -> None:
    values = {
        "bazett_ms": 391.0,
        "fridericia_ms": 391.0,
        "hodges_ms": 391.0,
        "framingham_ms": 391.0,
    }

    assert classify_adult_qt(values, sex="female", reliable=True).status == "not_matched"


def test_wide_qrs_uses_jtc_review_instead_of_unavailable_qt() -> None:
    result = evaluate_qt_values(
        450.0,
        60.0,
        sex="male",
        reliable=True,
        qrs_ms=140.0,
        jt_ms=310.0,
        rr_cv=0.02,
    )

    assert result.status == "matched"
    assert result.statement_code == "wide_qrs_repolarization_review"
    assert result.severity == "observation"
    assert result.evidence["jtc_hodges_ms"] == pytest.approx(310.0)


def test_missing_qt_is_unavailable() -> None:
    result = evaluate_qt_values(None, 70.0, sex="male", reliable=True)

    assert result.status == "unavailable"
    assert "global.qt_ms" in result.missing_inputs


def test_unreliable_qt_is_unavailable() -> None:
    result = evaluate_qt_values(400.0, 70.0, sex="male", reliable=False)

    assert result.status == "unavailable"
    assert "global.qt_reliability" in result.missing_inputs


def test_adult_first_degree_av_delay_boundary() -> None:
    assert classify_adult_pr(200.0).status == "not_matched"
    assert classify_adult_pr(201.0).statement_code == "first_degree_av_delay"


def test_tachycardic_pr_over_half_rr_requires_same_cycle_confirmation() -> None:
    result = classify_adult_pr(310.0, heart_rate_bpm=110.0)

    assert result.status == "matched"
    assert result.statement_code == "possible_first_degree_av_delay"
    assert result.confidence == "low"
    assert result.coverage == "partial"
    assert "beat_level.same_cycle_p_qrs_association" in result.missing_inputs
    assert result.evidence["pr_rr_ratio"] > 0.50
    assert result.evidence["association_gate"] == "tachycardia_pr_exceeds_half_rr"


def test_tachycardic_pr_below_half_rr_can_match_first_degree_delay() -> None:
    result = classify_adult_pr(214.0, heart_rate_bpm=102.0)

    assert result.status == "matched"
    assert result.statement_code == "first_degree_av_delay"
    assert result.evidence["pr_rr_ratio"] < 0.50


def test_slow_rate_long_pr_is_not_rejected_by_tachycardia_gate() -> None:
    result = classify_adult_pr(296.0, heart_rate_bpm=79.0)

    assert result.status == "matched"
    assert result.statement_code == "first_degree_av_delay"


def test_unreliable_atrial_association_downgrades_long_pr() -> None:
    result = classify_adult_pr(
        319.0,
        heart_rate_bpm=84.0,
        atrial_association_unreliable=True,
    )

    assert result.status == "matched"
    assert result.statement_code == "possible_first_degree_av_delay"
    assert (
        "atrial_rhythm_evidence_invalidates_definite_pr_association"
        in result.evidence["association_gate_reasons"]
    )


def test_technical_limitation_downgrades_borderline_tachycardic_pr_ratio() -> None:
    result = classify_adult_pr(
        236.0,
        heart_rate_bpm=124.5,
        technically_limited=True,
    )

    assert result.statement_code == "possible_first_degree_av_delay"
    assert (
        "technical_limitation_with_tachycardic_high_pr_rr_ratio"
        in result.evidence["association_gate_reasons"]
    )


def test_clean_borderline_tachycardic_pr_ratio_remains_definite() -> None:
    result = classify_adult_pr(
        236.0,
        heart_rate_bpm=124.5,
        technically_limited=False,
    )

    assert result.statement_code == "first_degree_av_delay"


def test_adult_short_pr_boundary() -> None:
    assert classify_adult_pr(119.0).statement_code == "short_pr"
    assert classify_adult_pr(120.0).status == "not_matched"


def test_pediatric_pr_without_supported_table_is_unavailable() -> None:
    result = classify_pediatric_pr(180.0, age_years=8.0)

    assert result.status == "unavailable"
    assert "pediatric_pr_public_reference_table" in result.missing_inputs
