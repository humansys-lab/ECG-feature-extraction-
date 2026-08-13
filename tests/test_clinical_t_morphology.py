"""Hyperacute and giant-negative T-wave statements (clinical_rules/t_morphology.py)."""
from __future__ import annotations

from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation
from feature_extraction.ecgfeat.clinical_rules.t_morphology import (
    evaluate_t_morphology,
)


ALL_LEADS = ("I", "II", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


class Context:
    def __init__(self, *, leads=None, metadata=None):
        self._leads = leads or {}
        self.features = SimpleNamespace(metadata=metadata or {})

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        value = self._leads.get(lead, {}).get(name)
        return value if isinstance(value, (int, float)) else None

    def lead_raw_value(self, lead, name, reliability="reliable_for_qrs"):
        return self._leads.get(lead, {}).get(name)

    def global_value(self, name):
        return None


def by_code(rows, code):
    return next(row for row in rows if row.evidence["evaluates_code"] == code)


def _normal_lead(**overrides):
    lead = {
        "t_amp_mv": 0.25,
        "t_dur_ms": 160.0,
        "t_symmetry": 1.6,
        "r_amp_mv": 0.80,
        "s_amp_mv": -0.20,
        "q_amp_mv": None,
    }
    lead.update(overrides)
    return lead


def _baseline_leads(**per_lead):
    leads = {lead: _normal_lead() for lead in ALL_LEADS}
    for lead, overrides in per_lead.items():
        leads[lead] = _normal_lead(**overrides)
    return leads


# ── hyperacute T waves ───────────────────────────────────────────────────────

# qrs peak-to-peak is r_amp - s_amp = 1.0 mV, so t_amp doubles as the ratio.
_HYPERACUTE = {"t_amp_mv": 0.90, "t_dur_ms": 210.0, "t_symmetry": 1.7}


def test_hyperacute_t_matches_in_two_contiguous_leads() -> None:
    context = Context(leads=_baseline_leads(V2=_HYPERACUTE, V3=_HYPERACUTE))

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.status == "matched"
    assert result.evidence["qualifying_contiguous_pairs"] == [["V2", "V3"]]
    assert result.priority == "P1"


def test_hyperacute_t_needs_two_contiguous_leads_not_two_scattered_ones() -> None:
    context = Context(leads=_baseline_leads(V2=_HYPERACUTE, V6=_HYPERACUTE))

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.status == "not_matched"


def test_tall_t_in_proportion_to_a_large_qrs_is_not_hyperacute() -> None:
    """The finding is disproportion, not height -- an LVH-sized QRS earns a tall T."""
    tall_but_proportionate = {
        **_HYPERACUTE,
        "r_amp_mv": 2.60,
        "s_amp_mv": -0.40,
    }
    context = Context(
        leads=_baseline_leads(V2=tall_but_proportionate, V3=tall_but_proportionate)
    )

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.status == "not_matched"


def test_narrow_symmetric_tall_t_reads_as_hyperkalemic_not_hyperacute() -> None:
    hyperkalemic = {"t_amp_mv": 0.90, "t_dur_ms": 175.0, "t_symmetry": 1.0}
    context = Context(leads=_baseline_leads(V2=hyperkalemic, V3=hyperkalemic))

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.status == "not_matched"


def test_a_broad_symmetric_t_is_too_wide_for_hyperkalemia_and_still_matches() -> None:
    broad_symmetric = {"t_amp_mv": 0.90, "t_dur_ms": 240.0, "t_symmetry": 1.0}
    context = Context(leads=_baseline_leads(V2=broad_symmetric, V3=broad_symmetric))

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.status == "matched"


def test_narrow_t_is_not_hyperacute_however_tall() -> None:
    narrow = {"t_amp_mv": 1.20, "t_dur_ms": 120.0, "t_symmetry": 1.8}
    context = Context(leads=_baseline_leads(V2=narrow, V3=narrow))

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.status == "not_matched"


def test_hyperacute_t_is_suppressed_by_a_bundle_branch_block() -> None:
    context = Context(leads=_baseline_leads(V2=_HYPERACUTE, V3=_HYPERACUTE))
    lbbb = RuleEvaluation(
        rule_id="CLIN-CONDUCTION-LBBB-01",
        domain="conduction",
        status="matched",
        statement_code="lbbb_pattern",
        evidence={"evaluates_code": "lbbb_pattern"},
    )

    result = by_code(
        evaluate_t_morphology(context, conduction=[lbbb]),
        "hyperacute_t_wave_pattern",
    )

    assert result.status == "suppressed"
    assert "lbbb_pattern" in result.suppressed_by


def test_hyperacute_screen_abstains_without_enough_measured_leads() -> None:
    context = Context(leads={"V2": _normal_lead(**_HYPERACUTE)})

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.status == "unavailable"
    assert result.missing_inputs


def test_hyperacute_screen_is_supporting_so_it_cannot_block_normality() -> None:
    context = Context(leads={})

    result = by_code(evaluate_t_morphology(context), "hyperacute_t_wave_pattern")

    assert result.normality_role == "supporting"


# ── giant negative T waves ───────────────────────────────────────────────────


def test_giant_negative_t_matches_on_a_deep_precordial_inversion() -> None:
    context = Context(
        leads=_baseline_leads(
            V3={"t_amp_mv": -1.20},
            V4={"t_amp_mv": -0.90},
            V5={"t_amp_mv": -0.60},
        )
    )

    result = by_code(evaluate_t_morphology(context), "giant_negative_t_wave")

    assert result.status == "matched"
    assert result.evidence["qualifying_leads"] == ["V3"]
    assert result.human_review_required


def test_giant_negative_t_needs_a_second_supporting_lead() -> None:
    context = Context(leads=_baseline_leads(V3={"t_amp_mv": -1.20}))

    result = by_code(evaluate_t_morphology(context), "giant_negative_t_wave")

    assert result.status == "not_matched"


def test_ordinary_strain_pattern_inversion_is_not_giant() -> None:
    context = Context(
        leads=_baseline_leads(
            V4={"t_amp_mv": -0.30},
            V5={"t_amp_mv": -0.40},
            V6={"t_amp_mv": -0.35},
        )
    )

    result = by_code(evaluate_t_morphology(context), "giant_negative_t_wave")

    assert result.status == "not_matched"


def test_lvh_is_reported_beside_a_giant_negative_t_not_used_to_suppress_it() -> None:
    context = Context(
        leads=_baseline_leads(
            V3={"t_amp_mv": -1.20},
            V4={"t_amp_mv": -0.90},
        )
    )
    lvh = RuleEvaluation(
        rule_id="CLIN-HYPERTROPHY-LVH-01",
        domain="hypertrophy",
        status="matched",
        statement_code="lvh_voltage_criteria",
        evidence={"evaluates_code": "lvh_voltage_criteria"},
    )

    result = by_code(
        evaluate_t_morphology(context, hypertrophy=[lvh]), "giant_negative_t_wave"
    )

    assert result.status == "matched"
    assert "lvh_voltage_criteria" in result.evidence["differential_confounders"]
