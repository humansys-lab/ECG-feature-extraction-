from __future__ import annotations

from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.ischemia import evaluate_ischemia
from feature_extraction.ecgfeat.clinical_rules.quality import evaluate_quality
from tests.test_clinical_ischemia import Context, by_code


def test_q0_does_not_override_unreliable_st_measurement() -> None:
    context = Context(
        sex="female",
        leads={
            "V2": {"st_on_mv": 0.20, "st_j_reliable": False},
            "V3": {"st_on_mv": 0.20, "st_j_reliable": False},
        },
    )

    result = by_code(evaluate_ischemia(context, []), "acute_occlusion_pattern")

    assert result.status == "unavailable"
    assert "two_reliable_contiguous_st_leads" in result.missing_inputs


def test_brugada_is_optional_screen_for_general_normality() -> None:
    context = Context(leads={})

    result = by_code(evaluate_ischemia(context, []), "brugada_type1_screening")

    assert result.normality_role == "optional_screen"


def test_possible_reversal_is_quality_advisory_not_technical_failure() -> None:
    context = SimpleNamespace(
        excluded_leads=frozenset(),
        precordial_reversal_state="possible",
        precordial_reversal_leads=frozenset({"V5", "V6"}),
        features=SimpleNamespace(
            metadata={
                "record_quality": {
                    "record_grade": "Q0",
                    "rejected_functions": [],
                    "n_reliable_qrs_leads": 12,
                    "n_reliable_p_leads": 12,
                    "n_reliable_qt_leads": 12,
                }
            }
        ),
    )

    rows = evaluate_quality(context)

    advisory = next(row for row in rows if row.statement_code == "possible_precordial_lead_reversal")
    assert advisory.status == "matched"
    assert advisory.severity == "observation"
    assert not any(row.status == "matched" and row.severity == "technical" for row in rows)
