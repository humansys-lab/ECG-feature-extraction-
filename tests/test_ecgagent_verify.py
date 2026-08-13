"""Contract tests for the deterministic verification layer.

Two jobs are tested separately because they answer different questions:

* `verify_output` gates *cited* output — it is the loop's exit condition.
* `audit_untraceable_numbers` measures *uncited* output — it is how the
  existing pipeline gets a baseline without being rewritten first.
"""
from __future__ import annotations

import pytest

from ecgagent.evidence.store import EvidenceStore
from ecgagent.evidence.ledger import visible_citations
from ecgagent.tools.registry import build_default_registry
from ecgagent.verify import (
    VerificationPolicy,
    audit_untraceable_numbers,
    build_numeric_index,
    extract_quantities,
    verify_output,
    verify_structured,
)
from ecgagent.verify.context import parse_claim
from tests.test_ecgagent_tools import _payload


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore.from_dict(_payload(), record_id="TEST001")


@pytest.fixture
def whitelist(store: EvidenceStore):
    registry = build_default_registry(store, budget=None)
    for name, arguments in (
        ("get_measurement", {"pointer": "/global_features/pr_ms"}),
        ("get_measurement", {"pointer": "/global_features/qrs_ms"}),
        ("get_measurement", {"pointer": "/global_features/qtc_bazett_ms"}),
        ("get_lead_table", {"fields": ["q_duration_ms", "r_amp_mv", "t_amp_mv"]}),
    ):
        result = registry.call(name, arguments)
        registry.authorize_model_visible(
            tool=name,
            arguments=arguments,
            citations=visible_citations(result.render(), result.citations),
            source=f"test:{name}",
        )
    return registry.whitelist


# ---------------------------------------------------------------------------
# Quantity extraction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("QRS is 126 ms", [(126.0, "ms")]),
        ("QRS is 126ms wide", [(126.0, "ms")]),
        ("amplitude -0.29 mV", [(-0.29, "mV")]),
        ("axis -40 degrees", [(-40.0, "deg")]),
        ("axis -40°", [(-40.0, "deg")]),
        ("rate 62 bpm", [(62.0, "bpm")]),
        ("duration 10.0 s", [(10.0, "s")]),
        ("QT 400 milliseconds", [(400.0, "ms")]),
        ("PR candidates 114-116 ms", [(114.0, "ms"), (116.0, "ms")]),
        ("PR candidates 114–116 ms", [(114.0, "ms"), (116.0, "ms")]),
        ("axis -116 ms", [(-116.0, "ms")]),
        ("PR 115 ± 1 ms", [(115.0, "ms"), (1.0, "ms")]),
    ],
)
def test_quantity_extraction(text, expected):
    found = [(item.value, item.unit) for item in extract_quantities(text)]
    assert found == expected


def test_extraction_ignores_identifiers_leads_and_pointers():
    """Rule ids, priorities, lead names and pointers are not measurements."""
    text = (
        "CLIN-INTERVAL-PR-01 at priority P3 in leads V1 and V2 "
        "cites ev:/clinical_interpretation/domains/intervals/0/status"
    )
    assert extract_quantities(text) == []


def test_extraction_requires_a_unit():
    """A bare number is usually a count or a ratio; flagging it would drown the signal."""
    assert extract_quantities("Q/R ratio is 2.15 across 3 leads") == []


def test_lead_name_next_to_wave_label_is_not_parsed_as_seconds():
    found = extract_quantities("Lead V3 S wave amplitude is -0.002 mV.")
    assert [(item.value, item.unit) for item in found] == [(-0.002, "mV")]


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------
def test_correct_and_qualified_output_passes(store, whitelist):
    text = (
        "- PR measures 214 ms  ev:/global_features/pr_ms , above the 200 ms limit.\n"
        "- QTcB reads 407 ms  ev:/global_features/qtc_bazett_ms , but ecgfeat marks it "
        "unreliable, so QT is not interpretable here.\n"
    )
    report = verify_output(text, store, whitelist)
    assert report.passed, report.summary()
    assert report.n_supported_quantities == 2


def test_hallucinated_value_is_blocking(store, whitelist):
    text = "- PR measures 168 ms  ev:/global_features/pr_ms , which is normal."
    report = verify_output(text, store, whitelist)
    assert not report.passed
    codes = report.counts()
    assert codes["value_mismatch"] == 1
    finding = next(item for item in report.blocking if item.code == "value_mismatch")
    assert "214" in finding.detail          # feedback names the correct value


def test_unqualified_unreliable_measurement_is_blocking(store, whitelist):
    """The flagship check: an unreportable QT quoted as fact must fail."""
    text = "- QTcB is 407 ms  ev:/global_features/qtc_bazett_ms , normal for a male patient."
    report = verify_output(text, store, whitelist)
    assert not report.passed
    assert report.counts()["missing_qualifier"] == 1


@pytest.mark.parametrize(
    "hedge",
    ["but this is unreliable", "然而该测量不可靠", "ただし信頼できない", "though it is indeterminate"],
)
def test_a_hedged_claim_satisfies_qualification(store, whitelist, hedge):
    text = f"- QTcB is 407 ms  ev:/global_features/qtc_bazett_ms , {hedge}."
    report = verify_output(text, store, whitelist)
    assert report.counts().get("missing_qualifier") is None


def test_invented_pointer_is_blocking(store, whitelist):
    text = "- Dispersion index 44 ms  ev:/global_features/qtc_dispersion_index_ms ."
    report = verify_output(text, store, whitelist)
    assert not report.passed
    assert report.counts()["unresolvable_citation"] == 1


def test_pointer_never_read_by_a_tool_is_blocking(store, whitelist):
    """Resolvable is not enough: the model must have actually looked it up."""
    text = "- P axis is 62 deg  ev:/global_features/p_axis_deg ."
    report = verify_output(text, store, whitelist)
    assert not report.passed
    assert report.counts()["pointer_not_from_tool"] == 1

    relaxed = verify_output(text, store, whitelist, VerificationPolicy(require_provenance=False))
    assert relaxed.counts().get("pointer_not_from_tool") is None


def test_uncited_number_is_a_warning_by_default_and_can_be_escalated(store, whitelist):
    text = "- V2 shows a Q wave of 50 ms."
    report = verify_output(text, store, whitelist)
    assert report.passed                       # warning only
    assert report.counts()["unsupported_number"] == 1

    strict = verify_output(
        text, store, whitelist, VerificationPolicy(unsupported_number_severity="blocking")
    )
    assert not strict.passed


def test_a_threshold_beside_a_measurement_is_not_a_contradiction(store, whitelist):
    """"214 ms, above the 200 ms limit" cites one value and quotes one cutoff."""
    text = "- PR measures 214 ms  ev:/global_features/pr_ms , above the 200 ms limit."
    report = verify_output(text, store, whitelist)
    assert report.passed
    assert report.counts() == {}


def test_threshold_wording_never_excuses_a_contradicted_citation(store, whitelist):
    """Suppression applies to uncited numbers only, never to a wrong cited value."""
    text = "- PR measures 168 ms  ev:/global_features/pr_ms , above the 200 ms limit."
    report = verify_output(text, store, whitelist)
    assert not report.passed
    assert report.counts()["value_mismatch"] == 1


def test_feedback_is_specific_enough_to_act_on(store, whitelist):
    text = "- PR measures 168 ms  ev:/global_features/pr_ms , which is normal."
    feedback = verify_output(text, store, whitelist).feedback()
    assert "line 1" in feedback
    assert "214" in feedback
    assert "168 ms" in feedback


def test_dropping_a_citation_cannot_downgrade_a_failure(store, whitelist):
    """Regression: observed live on DeepSeek.

    A wrong cited value is blocking (`value_mismatch`). If an uncited number were
    only a warning in structured mode, the cheapest way to pass would be to
    delete the citation — which is exactly what the model did, going from 15
    blocking findings to zero by citing less.
    """
    cited_but_wrong = {"diagnoses": [{"evidence": [
        {"claim": "PR interval", "value": 168, "unit": "ms",
         "citations": ["ev:/global_features/pr_ms"]}]}]}
    citation_removed = {"diagnoses": [{"evidence": [
        {"claim": "PR interval", "value": 168, "unit": "ms", "citations": []}]}]}

    wrong = verify_structured(cited_but_wrong, store, whitelist)
    stripped = verify_structured(citation_removed, store, whitelist)

    assert not wrong.passed
    assert not stripped.passed, "removing the citation must not make the verdict pass"
    assert stripped.counts()["unsupported_number"] == 1


def test_free_text_keeps_the_lenient_default(store, whitelist):
    """Prose legitimately carries uncited numbers; structured evidence does not."""
    text = "- V2 shows a Q wave of 50 ms."
    assert verify_output(text, store, whitelist).passed
    assert not verify_structured(
        {"diagnoses": [{"evidence": [
            {"claim": "V2 Q wave", "value": 50, "unit": "ms", "citations": []}]}]},
        store,
        whitelist,
    ).passed


def test_structured_output_is_verified_through_the_same_checks(store, whitelist):
    good = {"diagnoses": [{"evidence": [
        {"claim": "PR interval", "value": 214, "unit": "ms",
         "citations": ["ev:/global_features/pr_ms"]}]}]}
    assert verify_output != None and verify_structured(good, store, whitelist).passed

    bad = {"diagnoses": [{"evidence": [
        {"claim": "PR interval", "value": 168, "unit": "ms",
         "citations": ["ev:/global_features/pr_ms"]}]}]}
    assert not verify_structured(bad, store, whitelist).passed


def test_structured_raw_value_accepts_equivalent_clinical_rounding():
    raw_rate = 50.08347245409015
    store = EvidenceStore.from_dict(
        {"global_features": {"heart_rate_bpm": raw_rate}},
        record_id="ROUNDED",
    )
    pointer = "/global_features/heart_rate_bpm"
    output = {
        "diagnoses": [
            {
                "evidence": [
                    {
                        "claim": "Background ventricular rate is 50 bpm",
                        "value": raw_rate,
                        "unit": "bpm",
                        "citations": [f"ev:{pointer}"],
                    }
                ]
            }
        ]
    }

    report = verify_structured(output, store, frozenset({pointer}))

    assert report.passed, report.summary()
    assert report.n_quantities == 1
    assert report.n_supported_quantities == 1


def test_structured_dimensionless_value_must_come_from_a_direct_pointer(store, whitelist):
    derived = {"diagnoses": [{"evidence": [
        {
            "claim": "Computed Q/R ratio",
            "value": 1.33,
            "unit": "ratio",
            "citations": [
                "ev:/representative_leads/V2/params/q_amp_mv",
                "ev:/representative_leads/V2/params/r_amp_mv",
            ],
        }
    ]}]}
    registry = build_default_registry(store, budget=None)
    result = registry.call(
        "get_lead_table",
        {"fields": ["q_amp_mv", "r_amp_mv"], "leads": ["V2"]},
    )
    registry.authorize_model_visible(
        tool="get_lead_table",
        arguments={"fields": ["q_amp_mv", "r_amp_mv"], "leads": ["V2"]},
        citations=visible_citations(result.render(), result.citations),
        source="test:get_lead_table",
    )

    report = verify_structured(derived, store, registry.whitelist)

    assert not report.passed
    assert report.counts()["derived_or_unsupported_value"] == 1


def test_structured_output_cannot_pass_with_diagnoses_but_no_evidence(store):
    output = {
        "summary": "Unsupported diagnosis.",
        "diagnoses": [
            {
                "statement": "Definite acute myocardial infarction",
                "evidence": [],
                "counterevidence": [],
                "adjudication": "Added without evidence.",
            }
        ],
    }
    report = verify_structured(output, store, frozenset())
    assert not report.passed
    assert report.counts()["empty_diagnostic_evidence"] == 1


# ---------------------------------------------------------------------------
# Traceability audit
# ---------------------------------------------------------------------------
def test_index_holds_only_dimensioned_values(store):
    """A confidence of 0.25 must never be quotable as 0.25 mV of ST shift."""
    index = build_numeric_index(store)
    assert index
    assert all(leaf.unit is not None for leaf in index)


def test_audit_traces_a_number_to_the_right_field(store):
    report = audit_untraceable_numbers("- PR interval is 214 ms.", store)
    assert len(report.results) == 1
    result = report.results[0]
    assert result.status == "traceable"
    assert result.matches[0] == "/global_features/pr_ms"


def test_audit_flags_a_number_that_matches_only_an_unrelated_field(store):
    """Magnitude alone is not provenance: the subject has to agree too."""
    report = audit_untraceable_numbers("- QRS duration is 214 ms.", store)
    result = report.results[0]
    assert result.status == "coincidental"
    assert result.term == "qrs"
    assert "/global_features/pr_ms" in result.loose_matches


def test_audit_flags_a_number_absent_from_the_payload(store):
    report = audit_untraceable_numbers("- QRS duration is 777 ms.", store)
    assert report.results[0].status == "untraceable"
    assert report.unsourced


def test_audit_respects_the_lead_a_claim_names(store):
    """V2's Q duration is 50 ms; I's is null, so the same number in lead I is not sourced."""
    in_v2 = audit_untraceable_numbers("- Lead V2 shows a Q wave of 50 ms.", store)
    assert in_v2.results[0].status == "traceable"

    in_i = audit_untraceable_numbers("- Lead I shows a Q wave of 50 ms.", store)
    assert in_i.results[0].status != "traceable"


def test_audit_summary_reports_the_three_way_split(store):
    text = (
        "- PR interval is 214 ms.\n"
        "- QRS duration is 777 ms.\n"
        "- QRS duration is 214 ms.\n"
    )
    report = audit_untraceable_numbers(text, store)
    assert len(report.traceable) == 1
    assert len(report.untraceable) == 1
    assert len(report.coincidental) == 1
    assert "traceable 1" in report.summary()


# ---------------------------------------------------------------------------
# Claim context
# ---------------------------------------------------------------------------
def test_term_binding_uses_the_nearest_preceding_subject():
    context = parse_claim("QRS duration measures 126 ms and the PR interval is 214 ms")
    positions = [item.start for item in extract_quantities(
        "QRS duration measures 126 ms and the PR interval is 214 ms")]
    assert context.term_for(positions[0]) == "qrs"
    assert context.term_for(positions[1]) == "pr"


def test_qtc_beats_qt_when_both_could_match():
    context = parse_claim("QTcB is 407 ms")
    assert context.term_for(len("QTcB is ")) == "qtc"


def test_lead_mentions_are_collected():
    assert parse_claim("V1 and V2 with aVF").leads == {"V1", "V2", "aVF"}
