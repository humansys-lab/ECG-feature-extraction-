from __future__ import annotations

from feature_extraction.ecgfeat.clinical_rules.context import build_context
from tests.test_clinical_export_consumers import _features


def _features_with_precordial_state(state: str, implicated_leads: list[str]):
    features = _features(qt_ms=400.0)
    features.metadata["lead_reversal"] = {
        "limb": {},
        "precordial": {
            "state": state,
            "suspected": state != "not_suspected",
            "implicated_leads": implicated_leads,
        },
    }
    features.interpretation.precordial_reversal_suspected = state != "not_suspected"
    return features


def test_possible_reversal_does_not_exclude_precordial_leads() -> None:
    context = build_context(_features_with_precordial_state("possible", ["V5", "V6"]))

    assert not ({"V1", "V2", "V3", "V4", "V5", "V6"} & context.excluded_leads)
    assert context.precordial_reversal_state == "possible"


def test_confirmed_reversal_excludes_only_implicated_pair() -> None:
    context = build_context(_features_with_precordial_state("confirmed", ["V2", "V3"]))

    assert context.excluded_leads == frozenset({"V2", "V3"})
    assert context.precordial_reversal_state == "confirmed"


def test_legacy_boolean_reversal_is_advisory_not_confirmed() -> None:
    features = _features(qt_ms=400.0)
    features.metadata.pop("lead_reversal", None)
    features.interpretation.precordial_reversal_suspected = True

    context = build_context(features)

    assert context.precordial_reversal_state == "possible"
    assert not context.excluded_leads
