"""Behavioral regression tests for bounded, evidence-triggered adjudication."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.agent.diagnostic_pathways import build_diagnostic_pathway
from ecgagent.agent.protocol import DEFAULT_DIAGNOSTIC_PROTOCOL
from ecgagent.backends.qwen_local import QwenLocalBackend
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_compact_workflow import _empty_backend, _response
from tests.test_ecgagent_tools import _payload


def agent_with(payload=None):
    payload = copy.deepcopy(payload or _payload())
    payload.pop("clinical_interpretation", None)
    return ECGDiagnosticAgent(store=EvidenceStore.from_dict(payload), backend=_empty_backend())


def candidate(code, candidate_id="h1"):
    return {"id": candidate_id, "code": code, "checks": [], "sources": ["independent_plan"],
            "diagnostic_pathway": build_diagnostic_pathway(code)}


class SchemaCompletion:
    """Respond unknown to every requested model task using its exact schema."""
    def __init__(self):
        self.schemas = []
        self.messages = []

    def create(self, **kwargs):
        schema = kwargs["response_format"]["json_schema"]["schema"]
        self.schemas.append(schema)
        self.messages.append(kwargs["messages"])
        rows = []
        for shape in schema["properties"]["decisions"].get("prefixItems", []):
            props = shape["properties"]
            rows.append({"id": props["id"]["enum"][0], "code": props["code"]["enum"][0],
                         "urgency": "ROUTINE", "pathway_steps": [{"id": step["properties"]["id"]["enum"][0],
                                            "status": "unknown", "evidence": [], "note": "Unresolved measurement"}
                                           for step in props["pathway_steps"]["prefixItems"]]})
        return _response(content=json.dumps({"overall_status": "insufficient_evidence",
                                            "decisions": rows, "interval_contexts": []}))


def test_complete_paths_are_retained_across_actual_batches(monkeypatch):
    agent = agent_with()
    monkeypatch.setattr(agent, "_required_component_intervals", lambda: set())
    rows = [candidate(code, f"h{i}") for i, code in enumerate([
        "first_degree_av_block", "lvh_voltage_criteria", "rbbb_pattern",
        "prior_infarct_q_wave_pattern", "st_elevation", "premature_ventricular_complexes"])]
    retained, deferred = agent._schedule_compact_candidates(rows)
    assert len(retained) == len(rows) and not deferred
    assert 1 < len(agent._compact_batches) <= DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_adjudication_batches
    completion = SchemaCompletion()
    agent.backend.client = SimpleNamespace(chat=SimpleNamespace(completions=completion))
    agent._compact_plan = {"candidates": retained, "review_tools": [], "quality_limitations": []}
    messages = [{"role": "user", "content": "PROGRAM-VALIDATED COMPACT PLAN. full plan"}]
    record = agent._run_phase(agent.phases[1], messages, [])
    assert not any("full plan" in str(message.get("content")) for batch in completion.messages for message in batch)
    assert messages[0]["content"].endswith("full plan")
    assert record.phase_guard_passed, record.phase_guard_problems
    assert agent._compact_executed_candidates >= {row["id"] for row in retained}
    assert len(agent._compact_batch_audit) > 1
    assert all(row["tool_budget"] <= 15 for row in agent._compact_batch_audit)
    assert all(row["adjudication_batch"] >= 1 for row in record.response_history)
    assert len({row["id"] for row in json.loads(record.text)["decisions"]}) == len(retained)


def test_unexecuted_program_candidate_cannot_borrow_previous_batch_measurements():
    agent = agent_with()
    agent.store.document["global_features"]["qrs_axis_deg"] = -45
    agent._compact_plan = {"candidates": [candidate("left_axis_deviation")], "quality_limitations": []}
    agent.registry.begin_phase("adjudicate", 1)
    agent.registry.call("get_global_table", {"fields": ["qrs_axis_deg"]})
    agent._compact_executed_candidates = set()
    verdict = agent._expand_compact_verdict({"decisions": []})
    assert not verdict["diagnoses"]
    assert agent._compact_decision_audit["decisions"][0]["final_placement"] == "unresolved"


def test_update_requires_visible_reliable_measurement_and_deduplicates_family():
    agent = agent_with()
    pointer = "/global_features/qrs_axis_deg"
    agent.backend._alias_pointers["Q1"] = pointer
    agent.registry.seed_citations([pointer], source="test", model_text="ev:" + pointer)
    proposals = [{"code": "left_axis_deviation", "support": ["Q1"]}]
    plan = {"candidates": []}
    assert agent._evidence_update_candidates(proposals, plan, set()) == []
    updates = agent._evidence_update_candidates(proposals, plan, {pointer})
    assert len(updates) == 1 and updates[0]["diagnostic_pathway"]["steps"]
    assert agent._evidence_update_candidates(proposals, {"candidates": updates}, {pointer}) == []
    agent.store.document["global_features"]["qrs_axis_deg"] = None
    assert agent._evidence_update_candidates(proposals, plan, {pointer}) == []


def test_same_view_planned_falsification_is_preserved():
    agent = agent_with()
    row = candidate("left_axis_deviation")
    row["checks"] = [{"tool": "get_global_table", "purpose": "falsify", "question": "Check axis applicability"}]
    steps = agent._merge_compact_rule_candidates([row])[0]["diagnostic_pathway"]["steps"]
    assert any(step["id"] == "planned_falsification" and step["owner"] == "model" for step in steps)


def test_rate_phenotype_survives_unresolved_sinus_identity():
    payload = _payload()
    payload["global_features"]["heart_rate_bpm"] = 48
    agent = agent_with(payload)
    rows = agent._merge_compact_rule_candidates([candidate("sinus_bradycardia")])
    agent._compact_plan = {"candidates": rows, "quality_limitations": []}
    agent.registry.begin_phase("adjudicate", 1)
    agent.registry.call("get_global_table", {"fields": ["heart_rate_bpm"]})
    verdict = agent._expand_compact_verdict({"decisions": []})
    assert [row["code"] for row in verdict["diagnoses"]] == ["bradycardia"]
    assert verdict["diagnoses"][0]["confidence"] == "MEDIUM"


@pytest.mark.parametrize("sex,s_value,quality,expected", [
    ("female", -1.5, True, "pass"), (None, -1.5, True, "unknown"),
    ("male", 3.0, True, "unknown"), ("female", -1.5, False, "unknown"),
])
def test_lvh_requires_signed_reliable_components_and_sex_reference(sex, s_value, quality, expected):
    payload = _payload()
    payload["metadata"]["patient_meta"]["sex"] = sex
    payload["representative_leads"] = {
        "aVL": {"params": {"r_amp_mv": .8, "reliable_for_qrs": quality}},
        "V3": {"params": {"s_amp_mv": s_value, "reliable_for_qrs": quality}},
    }
    agent = agent_with(payload)
    pointers = {f"/representative_leads/{lead}/params/{field}": {"get_lead_table"}
                for lead, row in payload["representative_leads"].items() for field in row["params"]}
    result = agent._compact_deterministic_pathway_step(code="lvh_voltage_criteria",
        step_id="cross_lead_voltage_criterion", expected_tool="get_lead_table", pointer_tools=pointers)
    assert result[0] == expected
    assert "/metadata/patient_meta/sex" in result[4]


def test_newly_read_opposite_st_polarity_receives_one_complete_followup(monkeypatch):
    payload = _payload()
    for lead in payload["representative_leads"]:
        payload["representative_leads"][lead]["params"].update(
            st_hybrid_j_mv=-.2, st_hybrid_reliable=True, st_j_reliable=True,
            st_j_unreliable_reason=None, reliable_for_t=True)
    agent = agent_with(payload)
    monkeypatch.setattr(agent, "_required_component_intervals", lambda: set())
    completion = SchemaCompletion()
    agent.backend.client = SimpleNamespace(chat=SimpleNamespace(completions=completion))
    row = candidate("st_elevation")
    agent._compact_plan = {"candidates": [row], "review_tools": [], "quality_limitations": []}
    record = agent._run_phase(agent.phases[1], [], [])
    assert record.phase_guard_passed, record.phase_guard_problems
    assert len(agent._compact_batch_audit) == 2
    assert any(row["code"] == "st_depression" and row["status"] == "raised"
               for row in agent._compact_update_audit)
    assert len(agent._compact_plan["candidates"]) == 2
    assert completion.schemas[-1]["properties"]["candidate_updates"]["maxItems"] == 0
    verdict = agent._expand_compact_verdict(json.loads(record.text))
    assert not verdict["diagnoses"]  # Opening a new path never passes its nodes.


def test_raw_conflict_adds_veto_but_status_only_cannot_resolve_it(monkeypatch):
    from tests.test_ecgagent_v44_waveform import _synthetic, _build
    raw, features = _synthetic()
    features["beat_features"][0]["s_amp_mv"] = 2.0
    payload = _payload()
    payload["waveform_review"] = _build(raw, features)
    assert payload["waveform_review"]["profiles"]["qrs"]["status"] == "conflict"
    agent = agent_with(payload)
    rows = agent._merge_compact_rule_candidates([candidate("prior_infarct_q_wave_pattern")])
    path = rows[0]["diagnostic_pathway"]
    node = next(row for row in path["steps"] if row["id"] == "raw_measurement_consistency")
    assert node["unknown_blocks_confirmation"] and node["failure_means_not_applicable"]
    agent._compact_plan = {"candidates": rows, "quality_limitations": []}
    agent.registry.begin_phase("adjudicate", 1)
    result = agent.registry.call("get_waveform_review", {"profile": "qrs"})
    pointer = "/waveform_review/profiles/qrs/status"
    agent.registry.authorize_model_visible(tool="get_waveform_review", arguments={"profile": "qrs"},
        citations=[pointer], source="test")
    agent.backend._alias_pointers["Q1"] = pointer
    agent._expand_compact_verdict({"decisions": [{"id": "h1", "code": rows[0]["code"],
        "pathway_steps": [{"id": "raw_measurement_consistency", "status": "pass",
        "evidence": [{"citation": "Q1", "claim": "A profile status exists"}]}]}]})
    audit = agent._compact_decision_audit["decisions"][0]
    step = next(row for row in audit["pathway_steps"] if row["id"] == node["id"])
    assert step["effective_status"] == "unknown"
    assert step["authorization"] == "raw_status_without_measurement"
    assert audit["final_placement"] == "unresolved"


def test_waveform_minimum_preserves_complete_observed_row_and_profile_identity():
    from tests.test_ecgagent_v44_waveform import _synthetic, _build
    from ecgagent.tools.waveform_review import get_waveform_review
    from ecgagent.evidence.model_view import build_model_evidence_view
    from ecgagent.evidence.packing import evidence_citation_label
    raw, features = _synthetic()
    features["waveform_review"] = _build(raw, features)
    store = EvidenceStore.from_dict(features).diagnostic_view()
    result = get_waveform_review(store, "qrs")
    view = build_model_evidence_view(store, tool="get_waveform_review", arguments={"profile": "qrs"},
        rendered_text=result.text, citations=result.citations, required_by=("h1:raw_measurement_consistency",))
    required = view.required_evidence["h1:raw_measurement_consistency"]
    root = "/waveform_review/profiles/qrs/observations/0/"
    assert {root + key for key in features["waveform_review"]["profiles"]["qrs"]["observations"][0]} <= set(required)
    assert evidence_citation_label(root + "raw_max_mv") == "waveform.qrs[0].raw_max_mv"


def test_earlier_batch_program_evidence_remains_verifiable_without_model_exposure(monkeypatch):
    agent = agent_with()
    monkeypatch.setattr(agent, "_required_component_intervals", lambda: set())
    agent.store.document["global_features"]["heart_rate_bpm"] = 48
    agent.store.document["global_features"]["qrs_axis_deg"] = -45
    agent._compact_plan = {"candidates": [candidate("bradycardia", "rate"),
        candidate("left_axis_deviation", "axis")], "review_tools": [], "quality_limitations": []}
    agent._compact_batches = [["rate"], ["axis"]]
    record = agent._run_phase(agent.phases[1], [], [])
    verdict = agent._expand_compact_verdict(json.loads(record.text))
    pointer = "/global_features/heart_rate_bpm"
    assert pointer in agent.registry.whitelist
    assert pointer not in agent.registry.model_visible_whitelist
    assert {row["code"] for row in verdict["diagnoses"]} == {"bradycardia", "left_axis_deviation"}
    assert agent._verify(verdict).passed


def test_later_batch_failure_preserves_earlier_responses_and_request_totals(monkeypatch):
    agent = agent_with()
    monkeypatch.setattr(agent, "_required_component_intervals", lambda: set())
    class FailAfterFirst(SchemaCompletion):
        calls = 0
        def create(self, **kwargs):
            self.calls += 1
            return super().create(**kwargs) if self.calls == 1 else _response(content="{}")
    completion = FailAfterFirst()
    agent.backend.client = SimpleNamespace(chat=SimpleNamespace(completions=completion))
    rows = [candidate("left_axis_deviation", name) for name in ("a", "b")]
    for row in rows:
        row["diagnostic_pathway"]["steps"].append({"id": "planned_falsification", "owner": "model",
            "gate": "required", "tool": "get_global_table", "arguments": {"fields": ["qrs_axis_deg"]},
            "question": "Is the measured axis interpretable?"})
    agent._compact_plan = {"candidates": rows, "review_tools": []}
    agent._compact_batches = [["a"], ["b"]]
    record = agent._run_phase(agent.phases[1], [], [])
    assert record.phase_guard_passed is False
    assert len(record.response_history) == completion.calls >= 2
    assert record.turns == completion.calls
    assert record.response_history[0]["candidate_ids"] == ["a"]
    assert record.response_history[-1]["candidate_ids"] == ["b"]
    assert agent._compact_executed_candidates == {"a"}


def test_missing_standard_global_field_does_not_discard_valid_neighbor():
    from ecgagent.tools.query import get_global_table
    agent = agent_with()
    assert agent.store.try_resolve("/global_features/rr_cv") is None
    result = get_global_table(agent.store, ["heart_rate_bpm", "rr_cv"])
    assert result.ok and "not exported" in result.text
    assert "/global_features/heart_rate_bpm" in result.citations
    assert "/global_features/rr_cv" not in result.citations
