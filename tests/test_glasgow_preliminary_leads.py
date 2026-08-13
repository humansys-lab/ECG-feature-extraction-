from __future__ import annotations

from typing import Callable

import pytest

from feature_extraction.ecgfeat.glasgow_rules.context import GlasgowContext, build_context
from feature_extraction.ecgfeat.glasgow_rules.preliminary_leads import (
    lead_exclusions,
    preliminary_lead_rules,
)
from feature_extraction.ecgfeat.models import PatientMeta, STANDARD_12_LEADS
from tests.glasgow_test_helpers import make_features


def _context() -> GlasgowContext:
    features = make_features(PatientMeta(age=59, sex="Female"))
    for index, lead in enumerate(STANDARD_12_LEADS):
        profile = {
            "p_positive_amp_mv": 0.10,
            "p_negative_amp_mv": -0.02,
            "qrs_peak_to_peak_mv": 1.0,
            "q_amp_mv": -0.02,
            "r_amp_mv": 0.20 + 0.10 * index,
            "s_amp_mv": -0.40,
            "r_prime_amp_mv": 0.0,
            "s_prime_amp_mv": 0.0,
            "st_amp_mv": 0.01,
            "t_positive_amp_mv": 0.20,
            "t_negative_amp_mv": 0.0,
            "qrs_area_matrix": 100.0 + 50.0 * index,
            "q_duration_ms": 20.0,
            "r_duration_ms": 30.0,
        }
        features.representative_leads[lead].params["glasgow_measurements"] = profile
    return build_context(features)


def _faulty_v3(context: GlasgowContext) -> None:
    context.leads["V2"].measurements["qrs_peak_to_peak_mv"] = 1.5
    context.leads["V3"].measurements.update(
        qrs_peak_to_peak_mv=0.30,
        t_positive_amp_mv=0.05,
        t_negative_amp_mv=-0.05,
    )
    context.leads["V4"].measurements["qrs_peak_to_peak_mv"] = 1.5


def _faulty_v6(context: GlasgowContext) -> None:
    context.leads["V5"].measurements["qrs_peak_to_peak_mv"] = 1.2
    context.leads["V6"].measurements["qrs_peak_to_peak_mv"] = 0.2


def _sequence_v1_v2(context: GlasgowContext) -> None:
    context.leads["V1"].measurements.update(r_amp_mv=0.80, t_positive_amp_mv=0.30, q_amp_mv=-0.02)
    context.leads["V2"].measurements.update(r_amp_mv=0.30, t_positive_amp_mv=0.10)
    context.leads["V3"].measurements["t_positive_amp_mv"] = 0.30


def _sequence_v2_v3(context: GlasgowContext) -> None:
    context.leads["V1"].measurements["qrs_area_matrix"] = -600.0
    context.leads["V2"].measurements.update(qrs_area_matrix=600.0, r_amp_mv=0.60, q_amp_mv=0.0)
    context.leads["V3"].measurements.update(qrs_area_matrix=-600.0, r_amp_mv=0.20)
    context.leads["V4"].measurements["qrs_area_matrix"] = 600.0


def _sequence_v4_v5(context: GlasgowContext) -> None:
    context.leads["V3"].measurements["qrs_area_matrix"] = 600.0
    context.leads["V4"].measurements["qrs_area_matrix"] = -600.0
    context.leads["V5"].measurements["qrs_area_matrix"] = 600.0


def _v1_v2_too_high(context: GlasgowContext) -> None:
    for lead in ("V1", "V2"):
        context.leads[lead].measurements.update(
            p_negative_amp_mv=-0.10,
            r_amp_mv=0.10,
            r_prime_amp_mv=0.20,
            t_negative_amp_mv=-0.10,
        )


def _arm_reversal(context: GlasgowContext) -> None:
    context.global_measurements.update(p_axis_deg=120.0, qrs_axis_deg=120.0)
    context.leads["I"].measurements.update(qrs_area_matrix=-600.0, r_amp_mv=0.10)
    context.leads["V6"].measurements.update(
        qrs_peak_to_peak_mv=1.0,
        qrs_area_matrix=600.0,
        p_positive_amp_mv=0.20,
        p_negative_amp_mv=-0.05,
    )


def _dextrocardia(context: GlasgowContext) -> None:
    context.global_measurements.update(p_axis_deg=120.0, qrs_axis_deg=70.0)
    for index, lead in enumerate(("V3", "V4", "V5", "V6")):
        context.leads[lead].measurements.update(
            r_amp_mv=0.09 - 0.01 * index,
            qrs_area_matrix=10.0 + 10.0 * index,
        )
    context.leads["V6"].measurements["qrs_peak_to_peak_mv"] = 0.70


def _limb_reversal(context: GlasgowContext) -> None:
    context.leads["I"].measurements["qrs_area_matrix"] = 10.0
    context.leads["III"].measurements["qrs_area_matrix"] = -10.0
    context.leads["II"].measurements.update(
        r_amp_mv=0.10,
        r_prime_amp_mv=0.0,
        s_amp_mv=-0.04,
        s_prime_amp_mv=0.0,
        q_amp_mv=-0.02,
        t_positive_amp_mv=0.03,
        t_negative_amp_mv=-0.02,
        p_positive_amp_mv=0.05,
        qrs_area_matrix=0.0,
    )


def _arm_leg_interchange(context: GlasgowContext) -> None:
    context.global_measurements.update(
        p_axis_deg=-120.0,
        qrs_axis_deg=-60.0,
        t_axis_deg=-45.0,
        heart_rate_bpm=80.0,
    )
    for lead in ("I", "II", "III"):
        context.leads[lead].measurements["p_negative_amp_mv"] = -0.08


@pytest.mark.parametrize(
    ("rule_id", "mutation", "expected_leads"),
    [
        ("GAN-04.01-01", _faulty_v3, ["V3"]),
        ("GAN-04.01-02", _faulty_v6, ["V6"]),
        ("GAN-04.01-03", _sequence_v1_v2, ["V1", "V2"]),
        ("GAN-04.01-04", _sequence_v2_v3, ["V2", "V3"]),
        ("GAN-04.01-05", _sequence_v4_v5, ["V4", "V5"]),
        ("GAN-04.01-06", _v1_v2_too_high, ["V1", "V2"]),
        ("GAN-04.01-09", _arm_reversal, ["I", "II", "III", "aVR", "aVL"]),
        ("GAN-04.01-10", _dextrocardia, []),
        ("GAN-04.01-11", _limb_reversal, ["I", "II", "III", "aVR", "aVL", "aVF"]),
        ("GAN-04.01-12", _arm_leg_interchange, ["I", "II", "III", "aVR", "aVL", "aVF"]),
    ],
)
def test_lead_rule_matches_exact_fixture(
    rule_id: str,
    mutation: Callable[[GlasgowContext], None],
    expected_leads: list[str],
) -> None:
    context = _context()
    mutation(context)

    result = next(rule.evaluator(context) for rule in preliminary_lead_rules() if rule.rule_id == rule_id)

    assert result.evaluation_status == "matched"
    assert result.evidence.get("excluded_leads", []) == expected_leads
    assert result.evidence["criteria"]


def test_normal_fixture_does_not_match_any_lead_pattern() -> None:
    results = [rule.evaluator(_context()) for rule in preliminary_lead_rules()]

    assert all(result.evaluation_status == "not_matched" for result in results)


def test_lead_exclusions_merge_rule_ids_without_mutating_context() -> None:
    context = _context()
    _faulty_v3(context)
    results = [rule.evaluator(context) for rule in preliminary_lead_rules()]

    exclusions = lead_exclusions(results)

    assert exclusions == {"V3": ["GAN-04.01-01"]}
    assert context.excluded_leads == {}

