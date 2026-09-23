from __future__ import annotations

import json
from types import SimpleNamespace

from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.evidence.store import EvidenceStore
from ecgagent.trace import render_agent_trace
from ecgagent.backends.qwen_local import QwenLocalBackend
from tests.test_ecgagent_compact_workflow import _empty_backend, _response
from tests.test_ecgagent_performance import Capture
from tests.test_ecgagent_tools import _payload


def make_agent():
    payload = _payload()
    payload.pop("clinical_interpretation", None)
    payload["global_features"].update(qrs_ms=103, qrs_wide_ms=114, qrs_axis_deg=-30.74)
    return ECGDiagnosticAgent(store=EvidenceStore.from_dict(payload), backend=_empty_backend())


def axis_candidate(tool="get_lead_table"):
    return {"id": "h1", "code": "left_axis_deviation", "domains": ["axis"],
            "checks": [{"tool": tool, "purpose": "falsify",
                        "question": "Do the limb leads corroborate the measured axis?"}]}


def test_cited_qrs_limitation_survives_report_expansion(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(agent, "_compact_pointer", lambda token:
                        "/global_features/qrs_ms" if token == "Q7" else None)
    agent._compact_plan["quality_limitations"] = ["Q7"]
    verdict = agent._expand_compact_verdict({"decisions": [], "interval_contexts": []})
    limitations = verdict["quality_assessment"]["limitations"]
    assert any("QRS duration" in value and "limited reliability" in value for value in limitations)
    assert agent._compact_limitation_sources[0]["citation"] == "ev:/global_features/qrs_ms"
    assert agent._compact_limitation_sources[0]["caveats"]


def test_unknown_quality_citation_is_explicitly_unresolved():
    assert "could not be resolved" in make_agent()._compact_report_limitation("Q999")


def test_uncovered_falsification_is_required_and_budgeted(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(agent, "_required_component_intervals", lambda: set())
    candidates = agent._merge_compact_rule_candidates([axis_candidate()])
    agent._compact_plan = {"candidates": candidates, "review_tools": []}
    spec = agent._compact_runtime_phase_spec(agent.phases[1])
    steps = candidates[0]["diagnostic_pathway"]["steps"]
    assert any(s["id"] == "planned_falsification" and s["gate"] == "required" for s in steps)
    assert any(name == "get_lead_table" for name, _ in spec.prefetch_tool_calls)
    assert any(row[0] == "h1:planned_falsification" for row in spec.model_evidence_requirements)
    assert agent._program_phase_response(spec) is None
    # Omitting the independent check cannot produce a confirmed diagnosis.
    agent.registry.begin_phase("adjudicate", 15)
    for name, args in spec.prefetch_tool_calls:
        agent.registry.call(name, dict(args))
    verdict = agent._expand_compact_verdict({"decisions": []})
    assert not verdict["diagnoses"]


def test_program_only_phase_skips_model_but_runs_measurement_tools(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(agent, "_required_component_intervals", lambda: set())
    candidate = axis_candidate("get_global_table")
    candidate["checks"] = []
    agent._compact_plan = {"candidates": agent._merge_compact_rule_candidates(
        [candidate]), "review_tools": []}
    spec = agent._compact_runtime_phase_spec(agent.phases[1])
    record = agent._run_phase(spec, [], [])
    assert record.turns == 0
    assert record.tool_calls == 1
    assert record.stop_reason == "program_completed"
    assert json.loads(record.text)["decisions"] == []
    assert agent.registry.calls[0].program_only
    trace = render_agent_trace({"phases": [record.to_dict()]},
                              tool_calls=[agent.registry.calls[0].to_trace_dict()])
    assert "Program-only result; not sent to model" in trace
    assert "Total batch runtime: N/A" in trace


def test_review_task_prevents_program_shortcut(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(agent, "_required_component_intervals", lambda: set())
    agent._compact_plan = {"candidates": agent._merge_compact_rule_candidates(
        [axis_candidate("get_global_table")]), "review_tools": ["get_morphology_groups"]}
    spec = agent._compact_runtime_phase_spec(agent.phases[1])
    assert agent._program_phase_response(spec) is None


def test_qwen_structured_response_disallows_whitespace_padding():
    capture = Capture([_response(content="{}")])
    backend = QwenLocalBackend(client=SimpleNamespace(chat=SimpleNamespace(completions=capture)))
    backend.complete(system="S", messages=[], tools=[], phase="plan", response_schema={"type": "object"})
    assert capture.calls[0]["extra_body"]["structured_outputs"]["disable_any_whitespace"] is True
    assert capture.calls[0]["extra_body"]["structured_outputs"]["json"] == {"type": "object"}


def test_required_packet_grows_only_for_actual_mandatory_content():
    from ecgagent.evidence.packing import json_size, required_packet_size
    from tests.test_ecgagent_performance import evidence_row
    backend = _empty_backend()
    backend.max_retained_evidence_chars = 600
    backend.max_required_evidence_chars = 6000
    rows = [evidence_row(str(i), mandatory=3, optional=10) for i in range(3)]
    packed = backend._bounded_evidence_rows(rows)
    assert required_packet_size(rows) > 600
    assert sum(bool(a.get("required_by")) for row in packed for a in row["result"]["evidence"]) == 9
    assert json_size(packed) <= backend._effective_evidence_chars <= 6000
    backend.max_required_evidence_chars = None
    assert json_size(backend._bounded_evidence_rows(rows)) <= 600


def test_first_view_preserves_required_bundle_before_shared_packing():
    from ecgagent.evidence.model_view import build_model_evidence_view
    payload = _payload()
    fields = ["q_duration_ms", "q_amp_mv", "r_amp_mv"]
    for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        payload["representative_leads"][lead] = {"params": {field: .1 for field in fields}}
    store = EvidenceStore.from_dict(payload)
    pointers = [f"/representative_leads/{lead}/params/{field}" for lead in store.leads for field in fields]
    view = build_model_evidence_view(store, tool="get_lead_table", arguments={"fields": fields},
        citations=pointers, rendered_text=" ".join("ev:" + p for p in pointers), required_by=("candidate:node",))
    assert len(view.required_evidence["candidate:node"]) == 36
    assert set(view.required_evidence["candidate:node"]) <= set(view.citations)


def test_reliable_authorized_axis_conflict_is_removed_before_budgeting():
    agent = make_agent()
    agent.registry.begin_phase("plan", 1)
    result = agent.registry.call("get_global_table", {"fields": ["qrs_axis_deg"]})
    agent.registry.authorize_model_visible(tool="get_global_table", arguments={"fields": ["qrs_axis_deg"]},
        citations=result.citations, source="test")
    assert agent._compact_candidate_semantic_conflict("right_axis_deviation")
    assert agent._compact_candidate_semantic_conflict("left_axis_deviation") is None


def test_positive_s_amplitude_cannot_confirm_early_rotation():
    from ecgagent.agent.deterministic_pathways import resolve_additional_deterministic_step
    payload = _payload()
    leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
    for lead in leads:
        payload["representative_leads"][lead] = {"params": {"r_amp_mv": .2, "s_amp_mv": -1.0}}
    payload["representative_leads"]["V1"]["params"]["s_amp_mv"] = .2
    store = EvidenceStore.from_dict(payload)
    available = {f"/representative_leads/{lead}/params/{field}" for lead in leads for field in ("r_amp_mv", "s_amp_mv")}
    result = resolve_additional_deterministic_step(store, code="counterclockwise_rotation",
        step_id="precordial_progression", available_pointers=available)
    assert result.status == "unknown"
    assert result.reason_code == "invalid_signed_r_s_amplitudes"


def test_null_only_model_refutation_becomes_unknown(monkeypatch):
    agent = make_agent()
    candidate = {"id": "h1", "code": "prior_infarct_q_wave_pattern", "domains": ["q_st_t_u"], "checks": []}
    candidates = agent._merge_compact_rule_candidates([candidate])
    agent._compact_plan = {"candidates": candidates, "quality_limitations": []}
    agent.registry.begin_phase("adjudicate", 15)
    args = {"fields": ["q_duration_ms", "q_amp_mv", "r_amp_mv", "qrs_ms"]}
    result = agent.registry.call("get_lead_table", args)
    pointer = "/representative_leads/I/params/q_duration_ms"
    agent.registry.authorize_model_visible(tool="get_lead_table", arguments=args,
        citations=[pointer], source="test")
    monkeypatch.setattr(agent, "_compact_pointer", lambda token: pointer if token == "Q1" else None)
    agent._expand_compact_verdict({"decisions": [{"id": "h1", "code": candidate["code"],
        "pathway_steps": [{"id": "q_wave_morphology", "status": "fail",
        "evidence": [{"citation": "Q1", "claim": "Q duration is unavailable"}]}]}]})
    decision = agent._compact_decision_audit["decisions"][0]
    assert decision["final_placement"] != "rejected"
    assert decision["pathway_steps"][0]["effective_status"] == "unknown"
    assert decision["pathway_steps"][0]["authorization"] == "unavailable_measurements_only"
