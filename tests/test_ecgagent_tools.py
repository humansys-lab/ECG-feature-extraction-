"""Contract tests for the ecgagent tool layer.

The load-bearing property is that a tool answer is *citable*: every pointer a
tool reports must resolve back to the value it showed.  If that breaks, a
downstream verifier cannot tell a real measurement from an invented one, so
those checks are treated as contract, not nicety.
"""
from __future__ import annotations

import json

import pytest

from ecgagent.evidence.briefing import build_chart_briefing
from ecgagent.evidence.diagnostic_briefing import build_diagnostic_briefing
from ecgagent.evidence.pointer import (
    PointerError,
    infer_unit,
    parse_pointer,
    resolve_alias,
)
from ecgagent.evidence.store import EvidenceStore
from ecgagent.tools.registry import build_default_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _payload() -> dict:
    """A small payload shaped like a real ecgfeat export."""
    return {
        "fs": 500,
        "quality": {
            "I": {"lead": "I", "reliable": True, "flags": [], "grade": "Q0",
                  "reliable_for_p": True, "reliable_for_qrs": True,
                  "reliable_for_t": True, "reliable_for_qt": True},
            "V2": {"lead": "V2", "reliable": False, "flags": ["baseline_wander"], "grade": "Q2",
                   "reliable_for_p": True, "reliable_for_qrs": True,
                   "reliable_for_t": False, "reliable_for_qt": False},
        },
        "beats": [
            {"beat_id": 0, "r_index": 100, "paced": False, "group_id": 1,
             "rr_prev_ms": None, "rr_next_ms": 800.0},
            {"beat_id": 1, "r_index": 500, "paced": False, "group_id": 1,
             "rr_prev_ms": 800.0, "rr_next_ms": 1200.0},
        ],
        "beat_features": [
            {"lead": "I", "beat_id": 0, "qrs_ms": 96.0, "t_amp_mv": 0.30},
            {"lead": "I", "beat_id": 1, "qrs_ms": 98.0, "t_amp_mv": 0.28},
            {"lead": "V2", "beat_id": 0, "qrs_ms": 97.0, "t_amp_mv": -0.10},
        ],
        "representative_leads": {
            "I": {"lead": "I", "params": {
                "qrs_ms": 96.0, "q_duration_ms": None, "q_amp_mv": -0.03,
                "r_amp_mv": 0.74, "t_amp_mv": 0.30, "p_terminal_duration_ms": 40.0,
                "reliable_for_p": True, "reliable_for_qrs": True,
                "reliable_for_t": True, "reliable_for_qt": True,
            }},
            "V2": {"lead": "V2", "params": {
                "qrs_ms": 97.0, "q_duration_ms": 50.0, "q_amp_mv": -0.29,
                "r_amp_mv": 0.13, "t_amp_mv": -0.10, "p_terminal_duration_ms": 94.0,
                "reliable_for_p": True, "reliable_for_qrs": True,
                "reliable_for_t": False, "reliable_for_qt": False,
                "st_j_reliable": False, "st_j_unreliable_reason": "baseline_unstable",
            }},
        },
        "global_features": {
            "heart_rate_bpm": 62.0,
            "pr_ms": 214.0,
            "qrs_ms": 108.0,
            "qrs_wide_ms": 124.0,
            "qt_ms": 400.0,
            "qtc_bazett_ms": 407.0,
            "qt_reliability": "unreliable",
            "qt_reportable": False,
            "qt_unreliable_reasons": ["multilead_residual_t_tail"],
            "qt_path": "normal_qt",
            "qt_source": "consensus",
            "qt_excluded_leads": {},
            "qt_rejected": False,
            "qt_reject_reason": None,
            "t_fusion_reliable": False,
            "p_axis_deg": 62.0,
            "qrs_axis_deg": 18.0,
            "t_axis_deg": None,
            "t_axis_reliable": False,
        },
        "metadata": {
            "input_fs": 500.0,
            "duration_sec": 10.0,
            "n_beats": 2,
            "patient_meta": {"age": 58, "sex": "M"},
            "record_quality": {"record_grade": "Q1", "rejected_functions": []},
            "diagnostic_gate": {"state": "partial", "stop_reasons": [],
                                "partial_reasons": ["limb_lead_equation_residual_high"],
                                "allowed_domains": ["all"],
                                "suppressed_domains": []},
        },
        "interpretation": {"probable_af": False, "bundle_branch_block": None},
        "clinical_interpretation": {
            "domains": {
                "intervals": [
                    {"rule_id": "CLIN-INTERVAL-PR-01", "domain": "intervals", "status": "matched",
                     "statement_code": "first_degree_av_delay",
                     "statement": "First-degree AV delay",
                     "confidence": "pr_and_association_evidence", "priority": None,
                     "severity": "abnormal", "coverage": "full",
                     "evidence": {"pr_ms": 214.0,
                                  "criteria": {"pr_upper_ms": 200.0},
                                  "lead_facts": {"II": {"pr_ms": 214.0}}},
                     "thresholds": {"pr_upper_ms": 200.0},
                     "missing_inputs": [], "suppressed_by": [],
                     "source": {"authority": "AHA/ACCF/HRS", "version": "2009"}},
                ],
                "rhythm": [
                    {"rule_id": "CLIN-RHYTHM-AFL-01", "domain": "rhythm", "status": "unavailable",
                     "statement_code": None, "statement": None,
                     "confidence": None, "priority": None, "coverage": "unavailable",
                     "evidence": {},
                     "missing_inputs": ["rhythm.multilead_F_wave_morphology"],
                     "suppressed_by": []},
                ],
            },
            "final_statements": [
                {"rule_id": "CLIN-INTERVAL-PR-01", "domain": "intervals", "status": "matched",
                 "statement_code": "first_degree_av_delay",
                 "statement": "First-degree AV delay",
                 "confidence": "MEDIUM", "priority": "P3",
                 "severity": "abnormal", "coverage": "full",
                 "evidence": {"pr_ms": 214.0,
                              "criteria": {"pr_upper_ms": 200.0},
                              "lead_facts": {"II": {"pr_ms": 214.0}}},
                 "thresholds": {"pr_upper_ms": 200.0},
                 "missing_inputs": [], "suppressed_by": [],
                 "source": {"authority": "AHA/ACCF/HRS", "version": "2009"}},
            ],
            "abstentions": [
                {"rule_id": "CLIN-RHYTHM-AFL-01", "domain": "rhythm", "status": "unavailable",
                 "missing_inputs": ["rhythm.multilead_F_wave_morphology"],
                 "reason": "missing_or_indeterminate_required_evidence"},
            ],
            "review_required": True,
            "review_reasons": ["first_degree_av_delay"],
        },
    }


def test_tool_calls_publish_stable_ids_and_pointer_provenance():
    store = EvidenceStore.from_dict(_payload(), record_id="PROVENANCE")
    registry = build_default_registry(store, include=("get_measurement",))
    registry.begin_phase("investigate", budget=2)
    arguments = {"pointer": "/global_features/heart_rate_bpm"}
    result = registry.call("get_measurement", arguments)
    registry.authorize_model_visible(
        tool="get_measurement",
        arguments=arguments,
        citations=result.citations,
        source="investigate:tool_result",
    )

    assert registry.calls[0].call_id == "T0001"
    provenance = registry.evidence_provenance(
        "/global_features/heart_rate_bpm"
    )
    assert provenance["tool_call_ids"] == ["T0001"]
    assert provenance["tool_names"] == ["get_measurement"]
    assert provenance["tool_phases"] == ["investigate"]


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore.from_dict(_payload(), record_id="TEST001")


@pytest.fixture
def registry(store: EvidenceStore):
    return build_default_registry(store, budget=None)


# ---------------------------------------------------------------------------
# Pointer layer
# ---------------------------------------------------------------------------
def test_pointer_accepts_citation_and_dotted_forms():
    for text in ("/global_features/qt_ms", "ev:/global_features/qt_ms", "global_features.qt_ms"):
        assert parse_pointer(text).raw == "/global_features/qt_ms"


def test_pointer_rejects_empty_and_root():
    for bad in ("", "   ", "/"):
        with pytest.raises(PointerError):
            parse_pointer(bad)


@pytest.mark.parametrize(
    "field, unit",
    [
        ("qt_ms", "ms"),
        ("q_amp_mv", "mV"),
        ("q_area_mv_ms", "mV*ms"),        # must beat both _ms and _mv
        ("qrs_axis_deg", "deg"),
        ("heart_rate_bpm", "bpm"),
        ("st_slope_mv_per_ms", "mV/ms"),
        ("rr_cv", None),
        ("t_sqi_score", None),
    ],
)
def test_unit_inference(field, unit):
    assert infer_unit(field) == unit


def test_alias_resolution_is_case_and_punctuation_tolerant():
    assert resolve_alias("QTc Bazett") == "/global_features/qtc_bazett_ms"
    assert resolve_alias("qtc(bazett)") == "/global_features/qtc_bazett_ms"
    assert (
        resolve_alias("ventricular rate")
        == "/global_features/heart_rate_bpm"
    )
    assert (
        resolve_alias("/global_features/ventricular_rate_bpm")
        == "/global_features/heart_rate_bpm"
    )
    assert resolve_alias("/rr_mean_ms") == "/global_features/rr_mean_ms"
    assert resolve_alias("global.qt_reliability") == "/global_features/qt_reliability"
    assert (
        resolve_alias("lead.V1.r_prime_amp_mv")
        == "/representative_leads/V1/params/r_prime_amp_mv"
    )
    assert resolve_alias("not a real alias") is None


# ---------------------------------------------------------------------------
# Evidence store
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, unit, rendered",
    [
        (50.0, "ms", "50"),        # regression: an unguarded rstrip("0") gave "5"
        (420.0, "ms", "420"),
        (1080.0, "ms", "1080"),
        (100.0, "bpm", "100"),
        (0.0, "mV", "0"),
        (0.130, "mV", "0.13"),     # fractions still lose their trailing zero
        (-0.382, "mV", "-0.382"),
    ],
)
def test_integer_valued_measurements_keep_their_trailing_zeros(value, unit, rendered):
    """A value ending in zero must not be silently truncated."""
    from ecgagent.evidence.store import EvidenceValue

    assert EvidenceValue(pointer="/p", value=value, unit=unit).format_value() == rendered


def test_get_measurement_reports_the_real_value(store):
    """End-to-end guard on the same bug, through the tool the model actually calls."""
    registry = build_default_registry(store, budget=None)
    store.document["global_features"]["qrs_ms"] = 120.0
    result = registry.call("get_measurement", {"pointer": "/global_features/qrs_ms"})
    assert "120" in result.text and " = 12 " not in result.text


def test_bare_global_field_name_resolves_without_guessing_the_prefix(store):
    evidence = store.resolve("pr_ms")
    assert evidence.pointer == "/global_features/pr_ms"
    assert evidence.value == 214.0


def test_unreliable_qt_cannot_be_read_as_a_bare_number(store):
    """The whole point of the caveat layer: qt_reportable=false must travel with the value."""
    evidence = store.resolve("/global_features/qt_ms")
    assert evidence.value == 400.0
    assert evidence.unit == "ms"
    assert not evidence.reliable
    joined = " ".join(evidence.caveats)
    assert "qt_reportable=false" in joined
    assert "qt_reliability=unreliable" in joined


def test_qtc_inherits_the_qt_reliability_story(store):
    evidence = store.resolve("QTc")
    assert evidence.pointer == "/global_features/qtc_bazett_ms"
    assert any("qt_reportable" in note for note in evidence.caveats)


def test_consensus_qrs_disagreement_is_surfaced(store):
    evidence = store.resolve("/global_features/qrs_ms")
    assert any("qrs_wide_ms" in note for note in evidence.caveats)


def test_null_value_is_flagged_rather_than_silently_returned(store):
    evidence = store.resolve("/global_features/t_axis_deg")
    assert evidence.value is None
    assert evidence.caveats and "null" in evidence.caveats[0]


def test_lead_quality_flag_reaches_a_per_lead_value(store):
    evidence = store.resolve("/representative_leads/V2/params/t_amp_mv")
    assert any("reliable_for_t" in note for note in evidence.caveats)
    clean = store.resolve("/representative_leads/I/params/t_amp_mv")
    assert clean.reliable


def test_resolve_refuses_an_oversized_container_but_allows_a_small_one(store):
    """The guard is about rendered size, so small structured values stay citable."""
    payload = _payload()
    payload["representative_leads"]["I"]["params"].update(
        {f"filler_{index}_ms": float(index) for index in range(200)}
    )
    fat = EvidenceStore.from_dict(payload, record_id="FAT")
    with pytest.raises(PointerError) as excinfo:
        fat.resolve("/representative_leads/I/params")
    assert "get_lead_table" in str(excinfo.value)

    small = store.resolve("/clinical_interpretation/final_statements/0/thresholds")
    assert small.value == {"pr_upper_ms": 200.0}


def test_search_matches_reordered_tokens(store):
    hits = store.search("terminal p")
    assert any(hit.field == "p_terminal_duration_ms" for hit in hits)


def test_rule_index_prefers_resolved_confidence(store):
    """domains carries a raw basis string; final_statements carries MEDIUM/P3."""
    base, row = store.rule_index()["CLIN-INTERVAL-PR-01"]
    assert row["confidence"] == "MEDIUM"
    assert row["priority"] == "P3"
    assert base.startswith("/clinical_interpretation/final_statements/")
    assert store.resolve(f"{base}/confidence").value == "MEDIUM"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def test_every_tool_citation_resolves(registry, store):
    from ecgagent.evidence.ledger import visible_citations

    calls = [
        ("get_chart_briefing", {}),
        ("list_findings", {}),
        ("get_rule_detail", {"rule_id": "CLIN-INTERVAL-PR-01"}),
        ("get_measurement", {"pointer": "/global_features/pr_ms"}),
        ("get_global_table", {"fields": ["heart_rate_bpm", "PR", "QTc"]}),
        ("get_lead_table", {"fields": ["q_duration_ms", "q_amp_mv", "r_amp_mv"]}),
        ("get_beat_table", {"fields": ["rr_prev_ms", "rr_next_ms"]}),
        ("get_beat_table", {"fields": ["qrs_ms"], "lead": "I"}),
        ("search_measurements", {"query": "qt"}),
    ]
    for name, args in calls:
        result = registry.call(name, args)
        assert result.ok, f"{name} failed: {result.text}"
        for citation in result.citations:
            assert store.try_resolve(citation) is not None, f"{name} cited unresolvable {citation}"
        shown = visible_citations(result.render(), result.citations)
        registry.authorize_model_visible(
            tool=name,
            arguments=args,
            citations=shown,
            source=f"test:{name}",
        )
        assert set(shown) <= set(result.citations)
    assert registry.whitelist


def test_lead_table_marks_unreliable_cells(registry):
    result = registry.call("get_lead_table", {"fields": ["t_amp_mv"]})
    assert result.ok
    lines = {line.split("|")[1].strip(): line for line in result.text.splitlines() if "|" in line}
    assert "*" in lines["V2"]
    assert "*" not in lines["I"]


def test_measurement_tables_print_exact_citation_tokens(registry):
    """The provider sees rendered text, not ToolResult.citations."""
    lead = registry.call(
        "get_lead_table",
        {"fields": ["qrs_ms"], "leads": ["I"]},
    )
    beat = registry.call(
        "get_beat_table",
        {"fields": ["qrs_ms"], "lead": "I", "beat_ids": [0]},
    )

    assert "[ev:/representative_leads/I/params/qrs_ms]" in lead.text
    assert "[ev:/beat_features/0/qrs_ms]" in beat.text
    assert "ev:/get_lead_table/" not in lead.text
    assert "ev:/get_beat_table/" not in beat.text


def test_global_table_batches_measurements_and_carries_caveats(registry):
    result = registry.call(
        "get_global_table",
        {"fields": ["heart_rate_bpm", "PR", "QTc"]},
    )
    assert result.ok
    assert "/global_features/heart_rate_bpm" in result.text
    assert "/global_features/pr_ms" in result.text
    assert "caution" in result.text
    assert "qt_reportable=false" in result.text
    assert "/global_features/qt_reportable" in result.citations


def test_lead_table_rejects_unknown_field_with_a_hint(registry):
    result = registry.call("get_lead_table", {"fields": ["qt_duration_nonsense"]})
    assert not result.ok
    assert "unknown per-lead field" in result.text


def test_list_findings_default_hides_uninformative_negatives(registry):
    result = registry.call("list_findings", {})
    assert "CLIN-INTERVAL-PR-01" in result.text
    assert "CLIN-RHYTHM-AFL-01" in result.text
    assert "needs rhythm.multilead_F_wave_morphology" in result.text


def test_rule_detail_keeps_criteria_from_being_starved(registry):
    """lead_facts must not crowd out the thresholds the rule actually applied."""
    result = registry.call("get_rule_detail", {"rule_id": "CLIN-INTERVAL-PR-01"})
    assert result.ok
    assert "criteria:" in result.text
    assert "pr_upper_ms = 200" in result.text


def test_unknown_rule_suggests_near_matches(registry):
    result = registry.call("get_rule_detail", {"rule_id": "CLIN-INTERVAL"})
    assert not result.ok
    assert "CLIN-INTERVAL-PR-01" in result.text


def test_unknown_tool_and_bad_arguments_are_reported_not_raised(registry):
    assert not registry.call("no_such_tool", {}).ok
    assert not registry.call("get_measurement", {}).ok            # missing required
    assert not registry.call("get_measurement", {"ptr": "x"}).ok  # unknown kwarg
    audit = registry.audit()
    assert audit["n_calls"] == 3
    assert all(not call["ok"] for call in audit["calls"])
    assert all(call["result_sha256"] for call in audit["calls"])


def test_handler_exceptions_become_tool_errors(store):
    from ecgagent.tools.registry import ToolRegistry, ToolSpec

    def _boom(_store):
        raise RuntimeError("kaboom")

    registry = ToolRegistry(store=store)
    registry.register(ToolSpec(name="boom", description="", parameters={}, handler=_boom))
    result = registry.call("boom", {})
    assert not result.ok
    assert "kaboom" in result.text


def test_budget_stops_further_calls(store):
    registry = build_default_registry(store, budget=2)
    registry.begin_phase("test", 2)
    assert registry.call("list_findings", {}).ok
    assert registry.call("list_findings", {}).ok
    exhausted = registry.call("list_findings", {})
    assert not exhausted.ok
    assert "budget" in exhausted.text
    assert registry.remaining == 0


def test_diagnostic_mode_suppresses_exact_duplicate_without_spending_budget(store):
    registry = build_default_registry(store, budget=None)
    registry.deduplicate_within_phase = True
    registry.begin_phase("investigate", 3)

    first = registry.call(
        "get_measurement",
        {"pointer": "/global_features/pr_ms"},
    )
    duplicate = registry.call(
        "get_measurement",
        {"pointer": "/global_features/pr_ms"},
    )

    assert first.ok
    assert not duplicate.ok
    assert duplicate.note == "duplicate"
    assert registry.phase_successes == 1
    assert registry.phase_distinct_successes == 1
    assert registry.remaining == 2
    assert registry.audit()["duplicate_calls_suppressed"] == 1


def test_rejected_calls_do_not_consume_the_budget(store):
    """The budget bounds evidence gathering; it should not charge for a typo.

    Charging for failures pushed the model into synthesising from a half-read
    chart after guessing a few field names wrong.
    """
    registry = build_default_registry(store, budget=None)
    registry.begin_phase("test", 2)
    for _ in range(4):
        assert not registry.call("get_lead_table", {"fields": ["no_such_field"]}).ok
    assert registry.remaining == 2, "failed calls must not spend the budget"

    assert registry.call("list_findings", {}).ok
    assert registry.remaining == 1


def test_thrashing_on_rejected_calls_is_still_capped(store):
    registry = build_default_registry(store, budget=None)
    registry.begin_phase("test", 2)
    results = [
        registry.call("get_lead_table", {"fields": ["no_such_field"]}) for _ in range(8)
    ]
    assert any("stop guessing" in result.text for result in results)


def test_begin_phase_scopes_the_budget_to_the_phase(store):
    registry = build_default_registry(store, budget=None)
    registry.begin_phase("test", 1)
    assert registry.call("list_findings", {}).ok
    assert not registry.call("list_findings", {}).ok      # test budget spent

    registry.begin_phase("adjudicate", 1)
    assert registry.remaining == 1, "a new phase gets its own allowance"
    assert registry.call("list_findings", {}).ok
    # The audit stays one continuous list across phases.
    assert [call["phase"] for call in registry.audit()["calls"]] == [
        "test",
        "test",
        "adjudicate",
    ]


def test_call_json_drives_the_guided_decoding_path(registry):
    result = registry.call_json('{"tool": "get_measurement", "args": {"pointer": "PR"}}')
    assert result.ok and "214" in result.text
    assert not registry.call_json("not json").ok
    assert not registry.call_json('{"args": {}}').ok


def test_audit_records_every_call(registry):
    registry.call("list_findings", {})
    registry.call("get_measurement", {"pointer": "PR"})
    audit = registry.audit()
    assert audit["n_calls"] == 2
    assert [call["tool"] for call in audit["calls"]] == ["list_findings", "get_measurement"]
    assert all("elapsed_ms" in call for call in audit["calls"])
    assert all("citations" not in call for call in audit["calls"])
    assert all("citation_manifest" in call for call in audit["calls"])
    assert "citations" in registry.calls[0].to_trace_dict()


# ---------------------------------------------------------------------------
# Schemas and briefing
# ---------------------------------------------------------------------------
def test_tool_schemas_are_well_formed(registry):
    for tool in registry.anthropic_tools():
        assert tool["name"] and tool["description"]
        assert tool["input_schema"]["type"] == "object"
        for name in tool["input_schema"]["required"]:
            assert name in tool["input_schema"]["properties"]
    json.dumps(registry.openai_tools())


def test_guided_schema_constrains_arguments_per_tool(registry):
    """An open `args` object would only guarantee valid JSON, not a callable request."""
    schema = registry.guided_json_schema()
    branches = {branch["properties"]["tool"]["const"]: branch for branch in schema["anyOf"]}
    assert set(branches) == set(registry.specs)
    assert branches["get_measurement"]["properties"]["args"]["required"] == ["pointer"]


def test_briefing_is_compact_and_fully_citable(store):
    briefing = build_chart_briefing(store)
    assert len(briefing.text) < 6000
    for citation in briefing.citations:
        assert store.try_resolve(citation) is not None
    assert "gate=partial" in briefing.text
    assert "CLIN-INTERVAL-PR-01" in briefing.text
    assert "MEDIUM" in briefing.text          # resolved confidence, not the raw basis string
    assert "CLIN-RHYTHM-AFL-01" in briefing.text


def test_briefing_explains_every_distrust_marker(store):
    briefing = build_chart_briefing(store)
    assert "!" in briefing.text
    assert "[distrusted]" in briefing.text
    assert "qt_reportable=false" in briefing.text


def test_diagnostic_briefing_is_neutral_and_citable(store):
    briefing = build_diagnostic_briefing(store)
    for citation in briefing.citations:
        assert store.try_resolve(citation) is not None
    assert "CLIN-" not in briefing.text
    assert "first_degree_av_delay" not in briefing.text
    assert "[matched]" not in briefing.text
    assert "lower-weight, not discarded" in briefing.text
    assert "null/not-produced values are unavailable" in briefing.text
    assert "No ecgfeat rule conclusion" in briefing.text


def test_dissent_ignores_a_per_lead_map_with_no_positive_lead():
    """`pathological_q_leads` is a dict; twelve False values are still truthy."""
    payload = _payload()
    payload["interpretation"]["pathological_q_leads"] = {
        lead: False for lead in ("I", "II", "III", "aVF", "V1", "V2")
    }
    store = EvidenceStore.from_dict(payload, record_id="NOQ")
    assert "pathological Q" not in build_chart_briefing(store).text


def test_dissent_names_the_leads_the_reference_layer_flagged():
    payload = _payload()
    payload["interpretation"]["pathological_q_leads"] = {
        "I": False, "II": True, "III": True, "aVF": True, "V1": False,
    }
    store = EvidenceStore.from_dict(payload, record_id="INFQ")
    text = build_chart_briefing(store).text
    assert "pathological Q" in text
    assert "II, III, aVF" in text
    assert "{" not in text.split("[dissent]")[-1], "should not dump the raw dict"


def test_dissent_accepts_either_engines_name_for_the_same_concept():
    """The reference layer says IVCD where clinical_rules emits nonspecific_ivcd."""
    payload = _payload()
    payload["interpretation"]["bundle_branch_block"] = "IVCD"
    payload["clinical_interpretation"]["final_statements"][0]["statement_code"] = (
        "nonspecific_ivcd"
    )
    store = EvidenceStore.from_dict(payload, record_id="IVCD")
    assert "intraventricular conduction" not in build_chart_briefing(store).text


def test_briefing_separates_the_two_abstention_shapes():
    """Rule abstentions carry rule_id + missing_inputs; finding-layer ones carry finding_id."""
    payload = _payload()
    payload["clinical_interpretation"]["abstentions"].append(
        {"finding_id": "BRUGADA_TYPE1", "domain": "finding_layer", "status": "UNKNOWN",
         "reason": "finding_not_projected_by_measurement_layer", "evidence": []}
    )
    store = EvidenceStore.from_dict(payload, record_id="MIXED")
    briefing = build_chart_briefing(store)
    assert "[abstained]" in briefing.text and "CLIN-RHYTHM-AFL-01" in briefing.text
    assert "[unknown]" in briefing.text and "BRUGADA_TYPE1" in briefing.text
    for citation in briefing.citations:
        assert store.try_resolve(citation) is not None, citation


def test_briefing_survives_a_measurement_only_payload():
    """A payload without clinical_interpretation must still brief, not crash."""
    payload = _payload()
    payload.pop("clinical_interpretation")
    payload.pop("interpretation")
    store = EvidenceStore.from_dict(payload, record_id="BARE")
    briefing = build_chart_briefing(store)
    assert "[intervals]" in briefing.text
    registry = build_default_registry(store)
    assert not registry.call("list_findings", {}).ok
    assert registry.call("get_lead_table", {"fields": ["qrs_ms"]}).ok
