from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.hypertrophy import evaluate_hypertrophy


class Context:
    def __init__(self, *, sex="female", age=50.0, qrs_ms=90.0, leads=None, af_afl=None):
        self.sex = sex
        self.age_years = age
        self._global = {"qrs_ms": qrs_ms}
        self._leads = leads or {}
        self.features = SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "af_afl_summary": dict(af_afl or {})
                }
            }
        )

    def global_value(self, name):
        return self._global.get(name)

    def lead_available(self, lead, reliability):
        return lead in self._leads

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        if not self.lead_available(lead, reliability):
            return None
        return self._leads[lead].get(name)


def by_code(evaluations, code):
    return next(
        item
        for item in evaluations
        if item.statement_code == code or item.evidence.get("evaluates_code") == code
    )


def test_cornell_reports_voltage_criteria_not_anatomic_lvh() -> None:
    context = Context(
        sex="female",
        age=50,
        leads={"aVL": {"r_amp_mv": 1.2}, "V3": {"s_amp_mv": -1.2}},
    )

    result = by_code(evaluate_hypertrophy(context, []), "lvh_voltage_criteria")

    assert result.status == "matched"
    assert "ECG voltage criteria" in result.statement
    assert "anatomic" not in result.statement.lower()


def test_pediatric_lvh_without_public_percentile_table_is_unavailable() -> None:
    context = Context(age=8, sex="male")

    result = by_code(evaluate_hypertrophy(context, []), "pediatric_lvh_voltage")

    assert result.status == "unavailable"
    assert "pediatric_lvh_percentile_table" in result.missing_inputs


def test_missing_rprime_duration_cannot_match_rvh() -> None:
    context = Context(
        leads={"V1": {"r_prime_amp_mv": 0.5, "r_prime_duration_ms": None}}
    )

    result = by_code(evaluate_hypertrophy(context, []), "rvh_pattern")

    assert result.status == "unavailable"


def test_missing_rprime_duration_does_not_block_rvh_when_core_criteria_present() -> None:
    context = Context(
        leads={
            "V1": {
                "r_amp_mv": 0.8,
                "s_amp_mv": -0.2,
                "r_prime_amp_mv": 0.5,
                "r_prime_duration_ms": None,
            }
        },
    )
    context._global["qrs_axis_deg"] = 120.0

    result = by_code(evaluate_hypertrophy(context, []), "rvh_pattern")

    assert result.status == "matched"


def test_atrial_abnormality_rules_use_dedicated_domain() -> None:
    context = Context(
        leads={
            "II": {"p_amp_mv": 0.3, "p_duration_ms": 100.0},
            "V1": {"p_terminal_amp_mv": -0.15, "p_terminal_duration_ms": 50.0},
        }
    )

    evaluations = evaluate_hypertrophy(context, [])
    rae = by_code(evaluations, "right_atrial_abnormality")
    lae = by_code(evaluations, "left_atrial_abnormality")
    lvh = by_code(evaluations, "lvh_voltage_criteria")

    assert rae.domain == "atrial_abnormality"
    assert lae.domain == "atrial_abnormality"
    assert lvh.domain == "hypertrophy"


def test_sokolow_lyon_is_reported_as_voltage_criterion() -> None:
    context = Context(
        leads={"V1": {"s_amp_mv": -2.0}, "V5": {"r_amp_mv": 1.6}}
    )

    result = by_code(evaluate_hypertrophy(context, []), "lvh_voltage_criteria")

    assert result.status == "matched"
    assert "Sokolow-Lyon" in result.evidence["matched_criteria"]
    assert result.evidence["matched_criterion_count"] == 1
    assert result.confidence == "single_voltage_criterion"


def test_multiple_lvh_voltage_criteria_get_higher_confidence() -> None:
    context = Context(
        sex="female",
        qrs_ms=130.0,
        leads={
            "aVL": {"r_amp_mv": 1.3},
            "V3": {"s_amp_mv": -1.2},
            "V1": {"s_amp_mv": -2.0},
            "V5": {"r_amp_mv": 1.6},
        },
    )

    result = by_code(
        evaluate_hypertrophy(context, []), "lvh_voltage_criteria"
    )

    assert result.status == "matched"
    assert result.evidence["matched_criterion_count"] >= 2
    assert result.confidence == "multiple_voltage_criteria"
    assert "multiple ECG voltage criteria" in result.statement


def test_lvh_is_suppressed_by_lbbb() -> None:
    context = Context(
        leads={"aVL": {"r_amp_mv": 1.5}, "V3": {"s_amp_mv": -1.5}}
    )
    conduction = [
        type(
            "Evaluation",
            (),
            {
                "statement_code": "lbbb_pattern",
                "status": "matched",
                "evidence": {"evaluates_code": "lbbb_pattern"},
            },
        )()
    ]

    result = by_code(evaluate_hypertrophy(context, conduction), "lvh_voltage_criteria")

    assert result.status == "suppressed"
    assert "lbbb_pattern" in result.suppressed_by


def test_atrial_morphology_is_not_applicable_during_af() -> None:
    context = Context(
        af_afl={"probable_af": True, "probable_flutter": False},
        leads={
            "II": {"p_amp_mv": 0.30, "p_dur_consensus_ms": 100.0},
            "V1": {"p_terminal_amp_mv": -0.12, "p_terminal_duration_ms": 50.0},
        },
    )

    evaluations = evaluate_hypertrophy(context, [])
    rae = by_code(evaluations, "right_atrial_abnormality")
    lae = by_code(evaluations, "left_atrial_abnormality")

    assert rae.status == "not_applicable"
    assert lae.status == "not_applicable"
    assert rae.evidence["atrial_morphology_invalid_by"] == "atrial_fibrillation_pattern"
    # Must stay grouped under atrial_abnormality (not fall back to the
    # default "hypertrophy" domain), otherwise the domain disappears from
    # `domains` entirely whenever AF/flutter is active and gets miscounted
    # as an unavailable required domain.
    assert rae.domain == "atrial_abnormality"
    assert lae.domain == "atrial_abnormality"
