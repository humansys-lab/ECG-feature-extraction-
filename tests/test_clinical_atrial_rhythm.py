"""Atrial origin and multiplicity statements (clinical_rules/atrial_rhythm.py)."""
from __future__ import annotations

from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.atrial_rhythm import (
    evaluate_atrial_rhythm,
)


class Context:
    def __init__(
        self,
        *,
        leads=None,
        globals_=None,
        beats=None,
        beat_features=None,
        metadata=None,
        limb_reversal=False,
    ):
        self._leads = leads or {}
        self._globals = globals_ or {}
        self.features = SimpleNamespace(
            beats=list(beats or []),
            beat_features=list(beat_features or []),
            metadata=metadata or {},
        )
        self.limb_reversal = limb_reversal

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        value = self._leads.get(lead, {}).get(name)
        return value if isinstance(value, (int, float)) else None

    def lead_raw_value(self, lead, name, reliability="reliable_for_qrs"):
        return self._leads.get(lead, {}).get(name)

    def global_value(self, name):
        return self._globals.get(name)


def by_rule(rows, rule_id):
    return next(row for row in rows if row.rule_id == rule_id)


ORIGIN = "CLIN-RHYTHM-P-ORIGIN-01"
MULTIFOCAL = "CLIN-RHYTHM-MULTIFOCAL-01"


def _inferior_p(amplitude):
    return {
        lead: {"p_amp_mv": amplitude, "p_dur_ms": 95.0}
        for lead in ("II", "III", "aVF")
    }


# ── inverted P origin ────────────────────────────────────────────────────────


def test_short_pr_with_inverted_inferior_p_is_junctional() -> None:
    context = Context(
        leads={**_inferior_p(-0.12), "aVR": {"p_amp_mv": 0.10}},
        globals_={"pr_ms": 96.0},
    )

    result = by_rule(evaluate_atrial_rhythm(context), ORIGIN)

    assert result.status == "matched"
    assert result.statement_code == "junctional_rhythm_pattern"
    assert result.evidence["avr_p_upright"]


def test_normal_pr_with_inverted_inferior_p_is_ectopic_atrial() -> None:
    context = Context(
        leads={**_inferior_p(-0.12), "aVR": {"p_amp_mv": 0.10}},
        globals_={"pr_ms": 158.0},
    )

    result = by_rule(evaluate_atrial_rhythm(context), ORIGIN)

    assert result.status == "matched"
    assert result.statement_code == "ectopic_atrial_rhythm_pattern"


def test_upright_inferior_p_is_not_an_ectopic_origin() -> None:
    context = Context(
        leads=_inferior_p(0.14),
        globals_={"pr_ms": 158.0},
    )

    result = by_rule(evaluate_atrial_rhythm(context), ORIGIN)

    assert result.status == "not_matched"


def test_limb_lead_reversal_suppresses_the_origin_call() -> None:
    """Reversed limb leads invert the inferior P waves with no change of origin."""
    context = Context(
        leads=_inferior_p(-0.12),
        globals_={"pr_ms": 96.0},
        limb_reversal=True,
    )

    result = by_rule(evaluate_atrial_rhythm(context), ORIGIN)

    assert result.status == "suppressed"
    assert "limb_lead_reversal" in result.suppressed_by


def test_implausibly_short_pr_is_indeterminate_not_junctional() -> None:
    context = Context(
        leads=_inferior_p(-0.12),
        globals_={"pr_ms": 40.0},
    )

    result = by_rule(evaluate_atrial_rhythm(context), ORIGIN)

    assert result.status == "indeterminate"
    assert result.statement_code is None


def test_missing_pr_abstains_rather_than_naming_a_mechanism() -> None:
    context = Context(leads=_inferior_p(-0.12), globals_={})

    result = by_rule(evaluate_atrial_rhythm(context), ORIGIN)

    assert result.status == "unavailable"
    assert "global.pr_ms" in result.missing_inputs


def test_one_inverted_inferior_lead_is_not_enough() -> None:
    context = Context(
        leads={
            "II": {"p_amp_mv": -0.12},
            "III": {"p_amp_mv": 0.10},
            "aVF": {"p_amp_mv": 0.08},
        },
        globals_={"pr_ms": 96.0},
    )

    result = by_rule(evaluate_atrial_rhythm(context), ORIGIN)

    assert result.status == "not_matched"


# ── multifocal atrial rhythm ─────────────────────────────────────────────────

# Three P morphologies, cycled, with varying PR and irregular RR.
_MORPHOLOGIES = [
    {"amp": 0.15, "dur": 90.0, "pr": 120.0},
    {"amp": 0.30, "dur": 90.0, "pr": 170.0},
    {"amp": 0.15, "dur": 140.0, "pr": 145.0},
]
_RR_CYCLE = (600.0, 900.0, 750.0)


ORGANIZED_P = {"rhythm_analysis": {"af_afl_summary": {"organized_p_ratio": 0.92}}}


def _multifocal_inputs(*, beat_count=9, morphologies=None):
    morphologies = morphologies or _MORPHOLOGIES
    beats = [
        SimpleNamespace(
            beat_id=index,
            paced=False,
            group_id=1,
            rr_prev_ms=_RR_CYCLE[index % len(_RR_CYCLE)] if index else None,
        )
        for index in range(beat_count)
    ]
    beat_features = [
        SimpleNamespace(
            lead="II",
            beat_id=index,
            beat_measurement_reliable=True,
            p_amp_mv=morphologies[index % len(morphologies)]["amp"],
            p_dur_ms=morphologies[index % len(morphologies)]["dur"],
            pr_ms=morphologies[index % len(morphologies)]["pr"],
        )
        for index in range(beat_count)
    ]
    return beats, beat_features


def test_three_p_morphologies_below_100_bpm_is_a_multifocal_atrial_rhythm() -> None:
    beats, beat_features = _multifocal_inputs()
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 88.0},
        metadata=ORGANIZED_P,
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "matched"
    assert result.statement_code == "multifocal_atrial_rhythm"
    assert result.evidence["qualifying_morphology_count"] == 3


def test_three_p_morphologies_at_or_above_100_bpm_is_mat() -> None:
    beats, beat_features = _multifocal_inputs()
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata=ORGANIZED_P,
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "matched"
    assert result.statement_code == "multifocal_atrial_tachycardia"


def test_two_morphologies_read_as_ectopy_not_multifocal_rhythm() -> None:
    beats, beat_features = _multifocal_inputs(morphologies=_MORPHOLOGIES[:2])
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata=ORGANIZED_P,
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "not_matched"
    assert not result.evidence["criteria"]["distinct_morphologies"]


def test_measurement_scatter_within_one_morphology_does_not_split_it() -> None:
    """Beat-to-beat noise must not manufacture three foci out of one."""
    jittered = [
        {"amp": 0.15, "dur": 90.0, "pr": 120.0},
        {"amp": 0.17, "dur": 96.0, "pr": 170.0},
        {"amp": 0.13, "dur": 84.0, "pr": 145.0},
    ]
    beats, beat_features = _multifocal_inputs(morphologies=jittered)
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata=ORGANIZED_P,
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "not_matched"
    assert result.evidence["qualifying_morphology_count"] == 1


def test_a_fixed_pr_interval_blocks_the_multifocal_call() -> None:
    fixed_pr = [dict(item, pr=150.0) for item in _MORPHOLOGIES]
    beats, beat_features = _multifocal_inputs(morphologies=fixed_pr)
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata=ORGANIZED_P,
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "not_matched"
    assert not result.evidence["criteria"]["varying_pr"]


def test_a_regular_ventricular_response_blocks_the_multifocal_call() -> None:
    beats, beat_features = _multifocal_inputs()
    for beat in beats:
        if beat.rr_prev_ms is not None:
            beat.rr_prev_ms = 750.0
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata=ORGANIZED_P,
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "not_matched"
    assert not result.evidence["criteria"]["irregular_rr"]


def test_atrial_fibrillation_suppresses_the_multifocal_call() -> None:
    beats, beat_features = _multifocal_inputs()
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata={
            "rhythm_analysis": {
                "af_afl_summary": {"probable_af": True, "organized_p_ratio": 0.92}
            }
        },
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "suppressed"
    assert "probable_atrial_fibrillation" in result.suppressed_by


def test_too_few_measured_beats_abstains() -> None:
    beats, beat_features = _multifocal_inputs(beat_count=6)
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata=ORGANIZED_P,
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "unavailable"


def test_both_atrial_screens_are_supporting_so_they_cannot_block_normality() -> None:
    rows = evaluate_atrial_rhythm(Context())

    assert all(row.normality_role == "supporting" for row in rows)


def test_disorganised_atrial_activity_blocks_the_multifocal_call() -> None:
    """The AF differential: fibrillatory waves cluster like P morphologies do."""
    beats, beat_features = _multifocal_inputs()
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
        metadata={
            "rhythm_analysis": {"af_afl_summary": {"organized_p_ratio": 0.21}}
        },
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "not_matched"
    assert not result.evidence["criteria"]["organized_p_waves"]


def test_missing_organised_p_ratio_abstains() -> None:
    beats, beat_features = _multifocal_inputs()
    context = Context(
        beats=beats,
        beat_features=beat_features,
        globals_={"heart_rate_bpm": 118.0},
    )

    result = by_rule(evaluate_atrial_rhythm(context), MULTIFOCAL)

    assert result.status == "unavailable"
