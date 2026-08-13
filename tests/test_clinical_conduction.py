from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.conduction import evaluate_conduction


class Context:
    def __init__(self, *, axis=0.0, qrs_ms=100.0, leads=None, excluded=()):
        self._global = {"qrs_axis_deg": axis, "qrs_ms": qrs_ms}
        self._leads = leads or {}
        self.excluded_leads = frozenset(excluded)
        self.features = SimpleNamespace(interpretation=None)

    def global_value(self, name):
        return self._global.get(name)

    def lead_available(self, lead, reliability):
        return lead not in self.excluded_leads and lead in self._leads

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        if not self.lead_available(lead, reliability):
            return None
        return self._leads[lead].get(name)


def by_code(evaluations, code):
    return next(item for item in evaluations if item.statement_code == code or item.evidence.get("evaluates_code") == code)


def matched_code(evaluations, code):
    return any(item.statement_code == code and item.status == "matched" for item in evaluations)


def test_axis_alone_never_diagnoses_lafb() -> None:
    result = by_code(evaluate_conduction(Context(axis=-60, qrs_ms=100)), "lafb_pattern")

    assert result.status == "unavailable"


def test_lafb_requires_qr_avl_and_rs_inferior() -> None:
    context = Context(
        axis=-60,
        qrs_ms=100,
        leads={
            # R(aVL) > R(I) and S(III) deeper than S(II) are part of the
            # criteria, so lead I has to be present for the rule to resolve.
            "I": {"q_amp_mv": 0.0, "r_amp_mv": 0.3, "s_amp_mv": -0.2},
            "aVL": {"q_amp_mv": -0.05, "r_amp_mv": 0.7, "s_amp_mv": -0.05},
            "II": {"q_amp_mv": 0.0, "r_amp_mv": 0.1, "s_amp_mv": -0.5},
            "III": {"q_amp_mv": 0.0, "r_amp_mv": 0.1, "s_amp_mv": -0.6},
            "aVF": {"q_amp_mv": 0.0, "r_amp_mv": 0.1, "s_amp_mv": -0.5},
        },
    )

    assert matched_code(evaluate_conduction(context), "lafb_pattern")


def test_qrs_108_without_bbb_morphology_is_not_ivcd() -> None:
    context = Context(
        axis=0,
        qrs_ms=108,
        leads={
            "V1": {"r_amp_mv": 0.2, "s_amp_mv": -0.6},
            "I": {"r_amp_mv": 0.8, "s_amp_mv": -0.05},
            "V6": {"r_amp_mv": 0.9, "s_amp_mv": -0.05},
        },
    )

    assert not matched_code(evaluate_conduction(context), "nonspecific_ivcd")


def test_rbbb_without_duration_yields_probable_tier_not_unavailable() -> None:
    context = Context(
        qrs_ms=130,
        leads={
            "V1": {"r_prime_amp_mv": 0.5, "r_prime_duration_ms": None},
            "I": {"s_amp_mv": -0.3, "s_duration_ms": None},
            "V6": {"s_amp_mv": -0.3, "s_duration_ms": None},
        },
    )

    result = by_code(evaluate_conduction(context), "rbbb_pattern")
    assert result.status == "matched"
    assert result.statement_code == "probable_rbbb_pattern"
    assert result.confidence == "probable_amplitude_only"


def test_rbbb_unavailable_without_any_amplitude_evidence() -> None:
    context = Context(qrs_ms=130, leads={})

    result = by_code(evaluate_conduction(context), "rbbb_pattern")
    assert result.status == "unavailable"


def test_rbbb_matches_with_duration_and_morphology() -> None:
    context = Context(
        qrs_ms=130,
        leads={
            "V1": {"r_prime_amp_mv": 0.5, "r_prime_duration_ms": 45.0},
            "I": {"s_amp_mv": -0.3, "s_duration_ms": 45.0},
            "V6": {"s_amp_mv": -0.2, "s_duration_ms": 42.0},
        },
    )

    result = by_code(evaluate_conduction(context), "rbbb_pattern")
    assert result.statement_code == "rbbb_pattern"
    assert result.confidence == "criteria_met"
    assert matched_code(evaluate_conduction(context), "rbbb_pattern")


def test_lbbb_without_r_duration_yields_probable_tier_not_unavailable() -> None:
    context = Context(
        qrs_ms=140,
        leads={
            "V1": {"r_amp_mv": 0.05},
            "I": {"q_amp_mv": 0.0, "r_amp_mv": 0.5, "r_duration_ms": None},
        },
    )

    result = by_code(evaluate_conduction(context), "lbbb_pattern")
    assert result.status == "matched"
    assert result.statement_code == "probable_lbbb_pattern"
    assert result.confidence == "probable_amplitude_only"


def test_lbbb_matches_with_duration_and_morphology() -> None:
    context = Context(
        qrs_ms=140,
        leads={
            "V1": {"r_amp_mv": 0.05},
            "I": {"q_amp_mv": 0.0, "r_amp_mv": 0.5, "r_duration_ms": 70.0},
        },
    )

    result = by_code(evaluate_conduction(context), "lbbb_pattern")
    assert result.status == "matched"
    assert result.statement_code == "lbbb_pattern"
    assert result.confidence == "criteria_met"


def test_lbbb_unavailable_without_any_amplitude_evidence() -> None:
    context = Context(qrs_ms=140, leads={})

    result = by_code(evaluate_conduction(context), "lbbb_pattern")
    assert result.status == "unavailable"


def test_wide_qrs_with_incomplete_bbb_exclusion_yields_probable_ivcd() -> None:
    context = Context(qrs_ms=130, leads={})

    result = by_code(evaluate_conduction(context), "nonspecific_ivcd")

    assert result.status == "matched"
    assert result.statement_code == "probable_nonspecific_ivcd"
    assert result.confidence == "incomplete_bbb_exclusion"
    assert result.evidence["manual_confirmation_required"] is True


def test_borderline_qrs_with_incomplete_bbb_exclusion_stays_unavailable() -> None:
    context = Context(qrs_ms=116, leads={})

    result = by_code(evaluate_conduction(context), "nonspecific_ivcd")

    assert result.status == "unavailable"
    assert "complete_bbb_morphology_exclusion" in result.missing_inputs


def test_wide_endpoint_rescues_probable_rbbb_only_with_terminal_morphology() -> None:
    context = Context(
        qrs_ms=96,
        leads={
            "V1": {
                "qrs_wide_ms": 112.0,
                "r_prime_amp_mv": 0.5,
                "r_prime_duration_ms": 70.0,
            },
            "V6": {
                "qrs_wide_ms": 112.0,
                "r_duration_ms": 30.0,
                "s_amp_mv": -0.3,
                "s_duration_ms": 20.0,
            },
        },
    )

    result = by_code(evaluate_conduction(context), "rbbb_pattern")

    assert result.status == "matched"
    assert result.statement_code == "probable_rbbb_pattern"
    assert result.confidence == "probable_wide_endpoint_morphology"
    assert result.evidence["v1_peak_delay_with_normal_lateral"] is True


def test_wide_endpoint_without_bbb_morphology_yields_probable_ivcd() -> None:
    context = Context(
        qrs_ms=105,
        leads={
            "V1": {
                "qrs_wide_ms": 126.0,
                "r_prime_amp_mv": 0.05,
                "r_amp_mv": 0.3,
            },
            "I": {
                "qrs_wide_ms": 126.0,
                "q_amp_mv": -0.1,
                "r_amp_mv": 0.2,
                "s_amp_mv": -0.05,
            },
            "V6": {
                "qrs_wide_ms": 126.0,
                "q_amp_mv": -0.1,
                "r_amp_mv": 0.2,
                "s_amp_mv": -0.05,
            },
        },
    )

    result = by_code(evaluate_conduction(context), "nonspecific_ivcd")

    assert result.status == "matched"
    assert result.statement_code == "probable_nonspecific_ivcd"
    assert result.confidence == "probable_wide_endpoint_consensus"


def test_incomplete_lbbb_requires_lateral_r_peak_time_support() -> None:
    context = Context(
        qrs_ms=114,
        leads={
            "V1": {"r_amp_mv": 0.05},
            "I": {
                "q_amp_mv": 0.0,
                "r_amp_mv": 0.7,
                "r_duration_ms": 45.0,
            },
        },
    )

    result = by_code(evaluate_conduction(context), "lbbb_pattern")

    assert result.status == "not_matched"
    assert (
        result.evidence["incomplete_lbbb_rejected"]
        == "lateral_r_peak_time_not_supported"
    )


def test_wide_endpoint_rbbb_rescue_rejects_lateral_delay_without_v1_delay() -> None:
    context = Context(
        qrs_ms=105,
        leads={
            "V1": {
                "qrs_wide_ms": 126.0,
                "r_prime_amp_mv": 0.2,
                "r_prime_duration_ms": 20.0,
                "r_amp_mv": 0.2,
            },
            "I": {
                "qrs_wide_ms": 126.0,
                "q_amp_mv": -0.1,
                "r_amp_mv": 0.2,
                "r_duration_ms": 20.0,
                "s_amp_mv": -0.3,
                "s_duration_ms": 50.0,
            },
            "V6": {
                "qrs_wide_ms": 126.0,
                "q_amp_mv": -0.1,
                "r_amp_mv": 0.2,
                "r_duration_ms": 20.0,
                "s_amp_mv": -0.3,
                "s_duration_ms": 50.0,
            },
        },
    )

    rbbb = by_code(evaluate_conduction(context), "rbbb_pattern")
    ivcd = by_code(evaluate_conduction(context), "nonspecific_ivcd")

    assert rbbb.status == "not_matched"
    assert ivcd.statement_code == "probable_nonspecific_ivcd"


def test_matched_lbbb_excludes_ivcd_even_when_rbbb_is_unavailable() -> None:
    context = Context(
        qrs_ms=140,
        leads={
            "V1": {"r_amp_mv": 0.05},
            "I": {
                "q_amp_mv": 0.0,
                "r_amp_mv": 0.5,
                "r_duration_ms": 70.0,
            },
        },
    )

    result = by_code(evaluate_conduction(context), "nonspecific_ivcd")

    assert result.status == "not_matched"
    assert result.evidence["excluded_by"] == ["lbbb_pattern"]


def test_lbbb_suppresses_fascicular_block_from_same_wide_qrs() -> None:
    context = Context(
        axis=-60,
        qrs_ms=140,
        leads={
            "V1": {"r_amp_mv": 0.05},
            "I": {
                "q_amp_mv": 0.0,
                "r_amp_mv": 0.5,
                "r_duration_ms": 70.0,
            },
            "aVL": {
                "q_amp_mv": -0.03,
                "r_amp_mv": 0.7,
                "r_duration_ms": 70.0,
            },
            "II": {"r_amp_mv": 0.1, "s_amp_mv": -0.5},
            "III": {"r_amp_mv": 0.1, "s_amp_mv": -0.6},
            "aVF": {"r_amp_mv": 0.1, "s_amp_mv": -0.5},
        },
    )

    evaluations = evaluate_conduction(context)
    lbbb = by_code(evaluations, "lbbb_pattern")
    lafb = by_code(evaluations, "lafb_pattern")

    assert lbbb.status == "matched"
    assert lafb.status == "suppressed"
    assert "lbbb_pattern" in lafb.suppressed_by
