"""U-wave diagnostic statements (clinical_rules/u_wave.py)."""
from __future__ import annotations

from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation
from feature_extraction.ecgfeat.clinical_rules.u_wave import evaluate_u_wave


class Context:
    def __init__(self, *, leads=None, globals_=None, metadata=None):
        self._leads = leads or {}
        self._globals = globals_ or {"heart_rate_bpm": 62.0, "qrs_ms": 92.0}
        self.features = SimpleNamespace(metadata=metadata or {})

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        value = self._leads.get(lead, {}).get(name)
        return value if isinstance(value, (int, float)) else None

    def lead_raw_value(self, lead, name, reliability="reliable_for_qrs"):
        return self._leads.get(lead, {}).get(name)

    def global_value(self, name):
        return self._globals.get(name)


def by_code(rows, code):
    return next(row for row in rows if row.evidence["evaluates_code"] == code)


def _lead(*, u_amp, t_amp, prominence=0.08, polarity=None):
    return {
        "u_amp_signed_mv": u_amp,
        "u_polarity": polarity if polarity is not None else (1 if u_amp >= 0 else -1),
        "u_measurement_reliable": True,
        "u_prominence_mv": prominence,
        "u_dur_ms": 90.0,
        "u_polarity_agreement": 1.0,
        "t_amp_mv": t_amp,
    }


def _leads(spec):
    return {lead: _lead(**kwargs) for lead, kwargs in spec.items()}


# ── prominent U wave ─────────────────────────────────────────────────────────


def test_prominent_u_wave_matches_on_absolute_amplitude() -> None:
    context = Context(
        leads=_leads(
            {
                "V2": {"u_amp": 0.14, "t_amp": 0.40},
                "V3": {"u_amp": 0.12, "t_amp": 0.45},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "matched"
    assert result.evidence["qualifying_leads"] == ["V2", "V3"]
    assert result.severity == "observation"


def test_prominent_u_wave_matches_on_the_quarter_of_t_criterion() -> None:
    context = Context(
        leads=_leads(
            {
                "V2": {"u_amp": 0.09, "t_amp": 0.30},
                "V3": {"u_amp": 0.08, "t_amp": 0.28},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "matched"


def test_ratio_criterion_needs_a_t_wave_worth_taking_a_quarter_of() -> None:
    """A 1 mm T wave makes every U wave 'disproportionate'; that is not a finding."""
    context = Context(
        leads=_leads(
            {
                "V2": {"u_amp": 0.07, "t_amp": 0.10},
                "V3": {"u_amp": 0.06, "t_amp": 0.09},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "not_matched"


def test_single_lead_prominent_u_is_not_enough() -> None:
    context = Context(
        leads=_leads(
            {
                "V2": {"u_amp": 0.16, "t_amp": 0.40},
                "V3": {"u_amp": 0.02, "t_amp": 0.45},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "not_matched"


def test_tachycardia_blocks_the_prominent_u_call() -> None:
    context = Context(
        leads=_leads(
            {
                "V2": {"u_amp": 0.14, "t_amp": 0.40},
                "V3": {"u_amp": 0.12, "t_amp": 0.45},
            }
        ),
        globals_={"heart_rate_bpm": 112.0, "qrs_ms": 92.0},
    )

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "indeterminate"
    assert result.evidence["rate_confounded"]


def test_inverted_u_is_not_reported_as_a_prominent_u() -> None:
    context = Context(
        leads=_leads(
            {
                "V2": {"u_amp": -0.14, "t_amp": 0.40},
                "V3": {"u_amp": -0.12, "t_amp": 0.45},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "not_matched"


# ── inverted U wave ──────────────────────────────────────────────────────────


def test_inverted_u_wave_matches_against_upright_t_waves() -> None:
    context = Context(
        leads=_leads(
            {
                "V4": {"u_amp": -0.08, "t_amp": 0.35},
                "V5": {"u_amp": -0.07, "t_amp": 0.30},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "inverted_u_wave")

    assert result.status == "matched"
    assert result.severity == "abnormal"
    assert result.human_review_required
    assert result.evidence["qualifying_leads"] == ["V4", "V5"]


def test_inverted_u_needs_prominence_not_just_amplitude() -> None:
    """Post-T baseline drift clears the amplitude gate with no wave under it."""
    context = Context(
        leads=_leads(
            {
                "V4": {"u_amp": -0.08, "t_amp": 0.35, "prominence": 0.03},
                "V5": {"u_amp": -0.07, "t_amp": 0.30, "prominence": 0.03},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "inverted_u_wave")

    assert result.status == "not_matched"


def test_inverted_u_requires_an_upright_t_to_invert_against() -> None:
    context = Context(
        leads=_leads(
            {
                "V4": {"u_amp": -0.08, "t_amp": -0.35},
                "V5": {"u_amp": -0.07, "t_amp": -0.30},
            }
        )
    )

    result = by_code(evaluate_u_wave(context), "inverted_u_wave")

    assert result.status == "not_matched"


def test_inverted_u_is_suppressed_under_a_wide_qrs() -> None:
    context = Context(
        leads=_leads(
            {
                "V4": {"u_amp": -0.08, "t_amp": 0.35},
                "V5": {"u_amp": -0.07, "t_amp": 0.30},
            }
        ),
        globals_={"heart_rate_bpm": 62.0, "qrs_ms": 148.0},
    )
    lbbb = RuleEvaluation(
        rule_id="CLIN-CONDUCTION-LBBB-01",
        domain="conduction",
        status="matched",
        statement_code="lbbb_pattern",
        evidence={"evaluates_code": "lbbb_pattern"},
    )

    result = by_code(
        evaluate_u_wave(context, conduction=[lbbb]), "inverted_u_wave"
    )

    assert result.status == "suppressed"
    assert "lbbb_pattern" in result.suppressed_by
    assert "qrs_duration_ge_120_ms" in result.suppressed_by


def test_unmeasured_u_waves_abstain_rather_than_deny() -> None:
    context = Context(leads={})

    rows = evaluate_u_wave(context)

    assert all(row.status == "unavailable" for row in rows)
    assert all(row.missing_inputs for row in rows)


def test_u_measurement_marked_unreliable_is_not_used() -> None:
    leads = _leads(
        {
            "V2": {"u_amp": 0.14, "t_amp": 0.40},
            "V3": {"u_amp": 0.12, "t_amp": 0.45},
        }
    )
    for lead in leads.values():
        lead["u_measurement_reliable"] = False
    context = Context(leads=leads)

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "unavailable"


def test_atrial_fibrillation_suppresses_the_u_wave_screens() -> None:
    """Fibrillatory waves fill the T-U segment and mimic U waves."""
    context = Context(
        leads=_leads(
            {
                "V2": {"u_amp": 0.14, "t_amp": 0.40},
                "V3": {"u_amp": 0.12, "t_amp": 0.45},
            }
        ),
        metadata={"rhythm_analysis": {"af_afl_summary": {"probable_af": True}}},
    )

    result = by_code(evaluate_u_wave(context), "prominent_u_wave")

    assert result.status == "suppressed"
    assert "probable_atrial_fibrillation" in result.suppressed_by


def test_atrial_flutter_suppresses_the_inverted_u_call() -> None:
    context = Context(
        leads=_leads(
            {
                "V4": {"u_amp": -0.08, "t_amp": 0.35},
                "V5": {"u_amp": -0.07, "t_amp": 0.30},
            }
        ),
        metadata={"rhythm_analysis": {"af_afl_summary": {"probable_flutter": True}}},
    )

    result = by_code(evaluate_u_wave(context), "inverted_u_wave")

    assert result.status == "suppressed"
    assert "probable_atrial_flutter" in result.suppressed_by
