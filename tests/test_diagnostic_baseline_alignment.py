"""Coverage for the four alignment changes made against
docs/心电图诊断规则系统综述.md:

1. a single declared baseline standard with the numeric conflicts resolved,
2. Mobitz II second-degree AV block actually reachable,
3. precordial R-wave progression promoted to a diagnostic statement,
4. bifascicular / trifascicular block synthesised from the component rules.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from feature_extraction.ecgfeat.clinical_rules.conduction import (
    _fascicular_combinations,
)
from feature_extraction.ecgfeat.clinical_rules.config import (
    DEFAULT_DIAGNOSTIC_CONFIG,
)
from feature_extraction.ecgfeat.clinical_rules.intervals import (
    classify_adult_qt,
    qtc_values,
)
from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation
from feature_extraction.ecgfeat.clinical_rules.r_progression import (
    evaluate_r_progression,
)
from feature_extraction.ecgfeat.clinical_rules.sources import (
    BASELINE_DIVERGENCES,
    BASELINE_STANDARD,
)
from feature_extraction.ecgfeat.rhythm_rules import detect_pauses_and_av_block


class _Context:
    """Minimal stand-in for ClinicalContext."""

    def __init__(self, *, leads=None, globals_=None, precordial_reversal=False):
        self._leads = leads or {}
        self._globals = globals_ or {}
        self.precordial_reversal = precordial_reversal

    def global_value(self, name):
        return self._globals.get(name)

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        return self._leads.get(lead, {}).get(name)

    def lead_raw_value(self, lead, name, reliability="reliable_for_qrs"):
        return self._leads.get(lead, {}).get(name)


def _by_code(rows, code):
    for row in rows:
        if row.evidence.get("evaluates_code") == code:
            return row
    raise AssertionError(f"no evaluation for {code}")


def _matched(rule_id, code):
    return RuleEvaluation(
        rule_id=rule_id,
        domain="conduction",
        status="matched",
        statement_code=code,
        evidence={"evaluates_code": code},
    )


# --- 1. baseline standard -------------------------------------------------


def test_baseline_standard_names_the_authoritative_layer() -> None:
    assert BASELINE_STANDARD["authoritative_layer"].endswith("clinical_rules")
    assert len(BASELINE_STANDARD["non_arbitrating_layers"]) == 2


def test_every_recorded_divergence_states_a_resolution() -> None:
    assert BASELINE_DIVERGENCES
    for entry in BASELINE_DIVERGENCES:
        assert entry["resolution"] in {
            "baseline",
            "adopted_alternative",
            "adopted_alternative_as_additional_tier",
            # The textbook criterion was adopted but did not stand up on its
            # own -- extra guards were needed to stop it false-positiving.
            # Distinct from a plain adoption, because the implementation is
            # deliberately stricter than the cited source.
            "adopted_alternative_with_added_guards",
        }
        assert entry["rationale"].strip()


def test_conflicting_thresholds_live_in_one_config() -> None:
    ischemia = DEFAULT_DIAGNOSTIC_CONFIG.ischemia
    intervals = DEFAULT_DIAGNOSTIC_CONFIG.intervals
    assert ischemia.pathological_q_duration_ms == 30.0
    assert ischemia.pathological_q_duration_v2_v3_ms == 20.0
    assert intervals.qtc_markedly_prolonged_ms == 480.0
    assert DEFAULT_DIAGNOSTIC_CONFIG.morphology.lpfb_axis_min_deg == 110.0


def test_qtc_480_now_reaches_the_markedly_prolonged_tier() -> None:
    values = {name: 485.0 for name in ("bazett_ms", "fridericia_ms", "hodges_ms", "framingham_ms")}

    result = classify_adult_qt(values, sex="male", reliable=True)

    assert result.statement_code == "markedly_prolonged_qt"
    assert result.priority == "P2"


def test_bazett_over_500_keeps_the_p1_critical_alert() -> None:
    values = {name: 460.0 for name in ("fridericia_ms", "hodges_ms", "framingham_ms")}
    values["bazett_ms"] = 530.0

    result = classify_adult_qt(values, sex="male", reliable=True)

    assert result.statement_code == "markedly_prolonged_qt"
    assert result.priority == "P1"


def test_tachycardic_bazett_inflation_still_does_not_trigger() -> None:
    """Regression guard for JS00059: Bazett >480 but Hodges normal."""
    values = qtc_values(qt_ms=376.5, hr_bpm=102.040816)

    result = classify_adult_qt(values, sex="female", reliable=True)

    assert values["bazett_ms"] > 480.0
    assert result.status == "not_matched"


# --- 2. Mobitz II ---------------------------------------------------------


def test_constant_pr_across_a_dropped_beat_is_mobitz_ii() -> None:
    result = detect_pauses_and_av_block(
        rr_ms=[800.0, 810.0, 1180.0, 800.0],
        pr_series_ms=[160.0, 162.0, None, 161.0],
        atrial_events_per_rr=[1, 1, 2, 1],
        qrs_duration_ms=140.0,
    )

    assert result["second_degree_avb"] == "mobitz_ii"
    assert result["av_block_level_hint"] == "infranodal_suspected"
    assert (
        result["second_degree_avb_basis"]["discriminator"]
        == "constant_pr_across_drop"
    )


def test_lengthening_pr_still_wins_over_the_constant_pr_branch() -> None:
    result = detect_pauses_and_av_block(
        rr_ms=[800.0, 810.0, 1180.0, 800.0],
        pr_series_ms=[180.0, 220.0, None, 180.0],
        atrial_events_per_rr=[1, 1, 2, 1],
        qrs_duration_ms=90.0,
    )

    assert result["second_degree_avb"] == "mobitz_i"
    assert result["av_block_level_hint"] == "av_nodal_probable"


def test_pr_jitter_below_the_increment_is_not_called_wenckebach() -> None:
    result = detect_pauses_and_av_block(
        rr_ms=[800.0, 810.0, 1180.0, 800.0],
        pr_series_ms=[160.0, 164.0, None, 161.0],
        atrial_events_per_rr=[1, 1, 2, 1],
        qrs_duration_ms=90.0,
    )

    assert result["second_degree_avb"] == "mobitz_ii"


def test_two_to_one_conduction_is_flagged_without_asserting_a_type() -> None:
    result = detect_pauses_and_av_block(
        rr_ms=[800.0, 1600.0, 810.0, 790.0],
        pr_series_ms=[160.0, None, 162.0, 161.0],
        atrial_events_per_rr=[2, 2, 2, 2],
        qrs_duration_ms=90.0,
    )

    assert result["two_to_one_conduction_suspected"] is True
    assert result["second_degree_avb"] is None


# --- 3. R-wave progression ------------------------------------------------


def _precordial(r_values, s_values=None):
    s_values = s_values or {}
    return _Context(
        leads={
            lead: {
                "r_amp_mv": r_values.get(lead),
                "s_amp_mv": s_values.get(lead, -0.2),
            }
            for lead in ("V1", "V2", "V3", "V4", "V5", "V6")
        }
    )


def test_low_r_in_v3_reports_poor_progression() -> None:
    context = _precordial(
        {"V1": 0.05, "V2": 0.10, "V3": 0.20, "V4": 0.60, "V5": 1.10, "V6": 1.00},
        {"V1": -1.0, "V2": -1.2, "V3": -0.9, "V4": -0.3, "V5": -0.1, "V6": -0.05},
    )

    row = _by_code(evaluate_r_progression(context, []), "poor_r_wave_progression")

    assert row.status == "matched"
    assert "Poor precordial R-wave progression" in row.statement
    assert "prior_anterior_infarction" in row.evidence["confounders_require_review"]


def test_monotonic_decline_reports_reversed_progression() -> None:
    context = _precordial(
        {"V1": 0.80, "V2": 0.60, "V3": 0.40, "V4": 0.20, "V5": 0.15, "V6": 0.10},
        {lead: -0.5 for lead in ("V1", "V2", "V3", "V4", "V5", "V6")},
    )

    row = _by_code(evaluate_r_progression(context, []), "poor_r_wave_progression")

    assert row.status == "matched"
    assert row.statement_code == "reversed_r_wave_progression"
    assert "dextrocardia" in row.statement


def test_normal_progression_reports_nothing() -> None:
    context = _precordial(
        {"V1": 0.15, "V2": 0.45, "V3": 0.85, "V4": 1.40, "V5": 1.60, "V6": 1.30},
        {"V1": -1.0, "V2": -0.9, "V3": -0.5, "V4": -0.2, "V5": -0.1, "V6": -0.05},
    )

    rows = evaluate_r_progression(context, [])

    assert all(row.status == "not_matched" for row in rows)


def test_lbbb_suppresses_progression_statements() -> None:
    context = _precordial(
        {"V1": 0.05, "V2": 0.05, "V3": 0.10, "V4": 0.30, "V5": 0.90, "V6": 1.00},
        {"V1": -1.5, "V2": -1.6, "V3": -1.2, "V4": -0.6, "V5": -0.1, "V6": -0.05},
    )

    row = _by_code(
        evaluate_r_progression(
            context, [_matched("CLIN-CONDUCTION-LBBB-01", "lbbb_pattern")]
        ),
        "poor_r_wave_progression",
    )

    assert row.status == "suppressed"
    assert row.suppressed_by == ["lbbb_pattern"]


def test_precordial_reversal_suppresses_progression_statements() -> None:
    context = _Context(precordial_reversal=True)

    rows = evaluate_r_progression(context, [])

    assert all(row.suppressed_by == ["precordial_lead_reversal"] for row in rows)


def test_late_transition_is_reported_as_clockwise_rotation() -> None:
    context = _precordial(
        {"V1": 0.10, "V2": 0.20, "V3": 0.30, "V4": 0.50, "V5": 0.70, "V6": 1.20},
        {"V1": -1.0, "V2": -1.0, "V3": -0.9, "V4": -0.8, "V5": -0.8, "V6": -0.3},
    )

    row = _by_code(evaluate_r_progression(context, []), "precordial_rotation")

    assert row.status == "matched"
    assert row.evidence["rotation"] == "clockwise"
    assert row.statement_code == "clockwise_rotation"


def test_early_transition_is_reported_as_counterclockwise_rotation() -> None:
    context = _precordial(
        {"V1": 0.90, "V2": 1.40, "V3": 1.60, "V4": 1.70, "V5": 1.50, "V6": 1.20},
        {lead: -0.1 for lead in ("V1", "V2", "V3", "V4", "V5", "V6")},
    )

    row = _by_code(evaluate_r_progression(context, []), "precordial_rotation")

    assert row.status == "matched"
    assert row.evidence["rotation"] == "counterclockwise"
    assert row.statement_code == "counterclockwise_rotation"


def test_noise_level_v1_r_is_not_a_transition_zone() -> None:
    """Regression for JS00948-series: R(V1)=0.054 mV beside S(V1)=-0.028 mV
    read as a V1 crossover even though V2 was a QS complex."""
    context = _precordial(
        {"V1": 0.054, "V2": 0.0, "V3": 0.226, "V4": 0.27, "V5": 0.778, "V6": 1.287},
        {"V1": -0.028, "V2": -1.17, "V3": -1.913, "V4": -1.24, "V5": -0.363, "V6": -0.265},
    )

    rows = evaluate_r_progression(context, [])
    rotation = _by_code(rows, "precordial_rotation")
    progression = _by_code(rows, "poor_r_wave_progression")

    assert rotation.evidence["r_s_transition_lead"] == "V5"
    assert rotation.evidence["rotation"] == "clockwise"
    assert progression.statement_code == "poor_r_wave_progression"


def test_transition_must_persist_into_the_next_lead() -> None:
    context = _precordial(
        {"V1": 0.15, "V2": 0.20, "V3": 0.90, "V4": 1.40, "V5": 1.60, "V6": 1.30},
        {"V1": -0.10, "V2": -1.20, "V3": -0.50, "V4": -0.20, "V5": -0.10, "V6": -0.05},
    )

    rotation = _by_code(evaluate_r_progression(context, []), "precordial_rotation")

    # V1 alone satisfies R >= |S| but V2 reverts, so V3 is the real transition.
    assert rotation.evidence["r_s_transition_lead"] == "V3"
    assert rotation.status == "not_matched"


def test_low_voltage_record_keeps_its_real_transition() -> None:
    """A fixed 0.10 mV floor would erase the transition on a low-voltage
    tracing; the floor scales with the record's own precordial amplitude."""
    context = _precordial(
        {"V1": 0.02, "V2": 0.05, "V3": 0.09, "V4": 0.16, "V5": 0.20, "V6": 0.18},
        {"V1": -0.18, "V2": -0.15, "V3": -0.11, "V4": -0.05, "V5": -0.02, "V6": -0.01},
    )

    rotation = _by_code(evaluate_r_progression(context, []), "precordial_rotation")

    assert rotation.evidence["transition_min_r_mv"] < 0.10
    assert rotation.evidence["r_s_transition_lead"] == "V4"
    assert rotation.evidence["rotation"] == "normal"


def test_missing_anterior_leads_make_progression_unavailable() -> None:
    context = _precordial({"V1": 0.10, "V2": None, "V3": 0.30, "V4": 0.60})

    row = _by_code(evaluate_r_progression(context, []), "poor_r_wave_progression")

    assert row.status == "unavailable"
    assert "lead.V2.r_amp_mv" in row.missing_inputs


# --- 4. bifascicular / trifascicular -------------------------------------


def _fascicular(rbbb_status, lafb_status, lpfb_status, pr_ms=None):
    def row(rule_id, code, status):
        return RuleEvaluation(
            rule_id=rule_id,
            domain="conduction",
            status=status,
            statement_code=code if status == "matched" else None,
            evidence={"evaluates_code": code},
        )

    return _fascicular_combinations(
        _Context(globals_={"pr_ms": pr_ms}),
        row("CLIN-CONDUCTION-RBBB-01", "rbbb_pattern", rbbb_status),
        row("CLIN-CONDUCTION-LAFB-01", "lafb_pattern", lafb_status),
        row("CLIN-CONDUCTION-LPFB-01", "lpfb_pattern", lpfb_status),
    )


def test_rbbb_with_lafb_is_bifascicular() -> None:
    row = _by_code(
        _fascicular("matched", "matched", "not_matched", pr_ms=160.0),
        "bifascicular_block_pattern",
    )

    assert row.status == "matched"
    assert "left anterior fascicular block" in row.statement
    assert row.priority == "P2"


def test_rbbb_with_lpfb_carries_the_higher_priority() -> None:
    row = _by_code(
        _fascicular("matched", "not_matched", "matched", pr_ms=160.0),
        "bifascicular_block_pattern",
    )

    assert row.status == "matched"
    assert row.priority == "P1"


def test_bifascicular_plus_prolonged_pr_is_trifascicular() -> None:
    rows = _fascicular("matched", "matched", "not_matched", pr_ms=240.0)

    assert _by_code(rows, "bifascicular_block_pattern").status == "matched"
    trifascicular = _by_code(rows, "trifascicular_block_pattern")
    assert trifascicular.status == "matched"
    assert trifascicular.priority == "P1"
    assert "cannot be localised" in trifascicular.statement


def test_normal_pr_leaves_trifascicular_unmatched() -> None:
    row = _by_code(
        _fascicular("matched", "matched", "not_matched", pr_ms=160.0),
        "trifascicular_block_pattern",
    )

    assert row.status == "not_matched"


def test_missing_pr_leaves_trifascicular_unavailable_not_negative() -> None:
    row = _by_code(
        _fascicular("matched", "matched", "not_matched", pr_ms=None),
        "trifascicular_block_pattern",
    )

    assert row.status == "unavailable"
    assert row.missing_inputs == ["global.pr_ms"]


def test_fascicular_block_without_rbbb_is_not_bifascicular() -> None:
    row = _by_code(
        _fascicular("not_matched", "matched", "not_matched", pr_ms=160.0),
        "bifascicular_block_pattern",
    )

    assert row.status == "not_matched"


def test_unresolved_component_keeps_the_combination_unavailable() -> None:
    row = _by_code(
        _fascicular("unavailable", "matched", "not_matched", pr_ms=160.0),
        "bifascicular_block_pattern",
    )

    assert row.status == "unavailable"
    assert "CLIN-CONDUCTION-RBBB-01" in row.missing_inputs


@pytest.mark.parametrize("axis", [95.0, 105.0])
def test_vertical_heart_axis_no_longer_reaches_lpfb(axis) -> None:
    from feature_extraction.ecgfeat.clinical_rules.conduction import _lpfb

    result = _lpfb(_Context(), axis)

    assert result.status == "not_matched"
