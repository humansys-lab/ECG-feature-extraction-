from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.other_patterns import (
    evaluate_other_patterns,
)


class Context:
    def __init__(self, *, leads=None, globals_=None, beats=None, beat_features=None):
        self._leads = leads or {}
        self._globals = globals_ or {}
        self.features = SimpleNamespace(
            beats=list(beats or []),
            beat_features=list(beat_features or []),
            metadata={},
        )
        self.limb_reversal = False

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        value = self._leads.get(lead, {}).get(name)
        return value if isinstance(value, (int, float)) else None

    def lead_raw_value(self, lead, name, reliability="reliable_for_qrs"):
        return self._leads.get(lead, {}).get(name)

    def global_value(self, name):
        return self._globals.get(name)


def by_code(rows, code):
    return next(row for row in rows if row.evidence["evaluates_code"] == code)


def _alternating_features(*, groups=(1,) * 8, leads=("II", "V1")):
    beats = [
        SimpleNamespace(
            beat_id=index,
            paced=False,
            group_id=groups[index],
            rr_prev_ms=800.0 if index else None,
        )
        for index in range(8)
    ]
    features = [
        SimpleNamespace(
            lead=lead,
            beat_id=index,
            beat_measurement_reliable=True,
            qrs_area=1.20 if index % 2 == 0 else 0.80,
        )
        for lead in leads
        for index in range(8)
    ]
    return beats, features


def test_electrical_alternans_requires_two_phase_concordant_leads() -> None:
    beats, features = _alternating_features()
    context = Context(beats=beats, beat_features=features)

    result = by_code(evaluate_other_patterns(context), "electrical_alternans")

    assert result.status == "matched"
    assert result.evidence["supporting_leads"] == ["II", "V1"]
    assert result.evidence["phase_concordant"]


def test_electrical_alternans_rejects_multiple_qrs_groups() -> None:
    beats, features = _alternating_features(groups=(1, 2, 1, 2, 1, 2, 1, 2))
    context = Context(beats=beats, beat_features=features)

    result = by_code(evaluate_other_patterns(context), "electrical_alternans")

    assert result.status != "matched"
    assert "multiple_qrs_morphology_groups" in result.evidence["excluded_reasons"]


def test_electrical_alternans_rejects_single_lead_pattern() -> None:
    beats, features = _alternating_features(leads=("II",))
    context = Context(beats=beats, beat_features=features)

    result = by_code(evaluate_other_patterns(context), "electrical_alternans")

    assert result.status != "matched"


def test_hyperkalemia_like_screen_is_low_confidence_and_requires_wide_qrs() -> None:
    leads = {
        lead: {
            "t_amp_mv": 0.80,
            "t_symmetry": 1.0,
            "t_dur_ms": 180.0,
        }
        for lead in ("V2", "V3", "V4")
    }
    leads["II"] = {"p_amp_mv": 0.05}
    narrow = Context(leads=leads, globals_={"qrs_ms": 98.0})
    wide = Context(leads=leads, globals_={"qrs_ms": 115.0})

    narrow_result = by_code(
        evaluate_other_patterns(narrow), "hyperkalemia_pattern"
    )
    wide_result = by_code(
        evaluate_other_patterns(wide), "hyperkalemia_pattern"
    )

    assert narrow_result.status == "not_matched"
    assert wide_result.status == "matched"
    assert wide_result.confidence == "low"
    assert wide_result.priority == "P4"
    assert wide_result.severity == "observation"


def test_hyperkalemia_screen_handles_unavailable_p_amplitude() -> None:
    leads = {
        lead: {
            "t_amp_mv": 0.80,
            "t_symmetry": 1.0,
            "t_dur_ms": 180.0,
        }
        for lead in ("V2", "V3", "V4")
    }
    context = Context(leads=leads, globals_={"qrs_ms": 120.0})

    result = by_code(evaluate_other_patterns(context), "hyperkalemia_pattern")

    assert result.status == "unavailable"


def test_hypokalemia_screen_requires_three_persistent_multilead_patterns() -> None:
    leads = {
        lead: {
            "t_amp_mv": 0.04,
            "u_amp_mv": 0.06,
            "st_on_mv": -0.08,
            "st_mid_mv": -0.07,
            "st_80ms_mv": -0.06,
            "st_morphology": "horizontal",
        }
        for lead in ("V4", "V5")
    }
    context = Context(leads=leads)

    result = by_code(evaluate_other_patterns(context), "hypokalemia_pattern")

    assert result.status != "matched"
