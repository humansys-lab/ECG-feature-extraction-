from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest

from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.agent.loop import PhaseRecord
from ecgagent.agent.protocol import DEFAULT_DIAGNOSTIC_PROTOCOL
from ecgagent.backends.deepseek import DeepSeekBackend
from ecgagent.backends.qwen_local import QwenLocalBackend
from ecgagent.backends.request_budget import openai_completion
from ecgagent.evidence.model_view import build_model_evidence_view
from ecgagent.evidence.packing import json_size, pack_evidence_rows
from ecgagent.evidence.store import EvidenceStore
from ecgagent.performance import label_metrics, summarize_performance
from tests.test_ecgagent_compact_workflow import _response, _empty_backend
from tests.test_ecgagent_tools import _payload


class Capture:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def empty_plan():
    return {"candidates": [], "review_tools": [], "quality_limitations": []}


def empty_decision():
    return {"overall_status": "insufficient_evidence", "decisions": [], "interval_contexts": []}


@pytest.mark.parametrize("backend_type", [QwenLocalBackend, DeepSeekBackend])
def test_explicit_phase_keeps_adjudication_thinking_with_plan_in_history(backend_type):
    capture = Capture([_response(content="{}"), _response(content="{}")])
    backend = backend_type(client=SimpleNamespace(chat=SimpleNamespace(completions=capture)), thinking=True)
    messages = [{"role": "user", "content": "PROGRAM-VALIDATED COMPACT PLAN"}]
    for phase in ("plan", "adjudicate"):
        backend.complete(system="S", messages=messages, tools=[], phase=phase)
    if backend_type is QwenLocalBackend:
        assert [row["extra_body"]["chat_template_kwargs"]["enable_thinking"] for row in capture.calls] == [False, True]
    else:
        assert [row["extra_body"]["thinking"]["type"] for row in capture.calls] == ["disabled", "enabled"]


@pytest.mark.parametrize("broken_phase", ["plan", "adjudicate"])
def test_compact_repairs_only_broken_phase_without_repeating_tools(broken_phase):
    responses = [_response(content=json.dumps(empty_plan())), _response(content=json.dumps(empty_decision()))]
    broken = _response(content='{"unfinished":')
    broken.choices[0].finish_reason = "length"
    responses.insert(0 if broken_phase == "plan" else 1, broken)
    capture = Capture(responses)
    backend = QwenLocalBackend(client=SimpleNamespace(chat=SimpleNamespace(completions=capture)), thinking=True)
    payload = _payload()
    payload.pop("clinical_interpretation")
    agent = ECGDiagnosticAgent(store=EvidenceStore.from_dict(payload), backend=backend)
    assert agent.workflow == "compact"
    result = agent.run()
    assert result.ok and result.verified, result.summary()
    assert [phase.turns for phase in result.phases] == ([2, 1] if broken_phase == "plan" else [1, 2])
    assert sum(call.tool == "get_diagnostic_overview" for call in agent.registry.calls) == 1
    assert len(capture.calls) == 3
    history = result.phases[0 if broken_phase == "plan" else 1].response_history
    assert history[0]["text"] == '{"unfinished":'
    assert history[0]["guard_problems"]
    assert "HARD PHASE STATE GUARD" in history[0]["repair_feedback"]
    assert len(history) == 2
    assert all(not call.get("tools") for call in capture.calls)
    assert capture.calls[1 if broken_phase == "plan" else 2]["max_tokens"] > capture.calls[0 if broken_phase == "plan" else 1]["max_tokens"]


def test_failed_compact_structure_stops_after_one_local_retry():
    capture = Capture([_response(content="[]"), _response(content="[]")])
    backend = QwenLocalBackend(client=SimpleNamespace(chat=SimpleNamespace(completions=capture)))
    result = ECGDiagnosticAgent(store=EvidenceStore.from_dict(_payload()), backend=backend).run()
    assert not result.ok
    assert len(capture.calls) == 2
    assert len(result.phases) == 1


def test_review_view_survives_nonempty_candidate_plan():
    agent = ECGDiagnosticAgent(store=EvidenceStore.from_dict(_payload()), backend=_empty_backend())
    candidates = agent._merge_compact_rule_candidates([{"id": "h1", "code": "first_degree_av_block", "domains": ["intervals"], "checks": []}])
    agent._compact_plan = {"candidates": candidates, "review_tools": ["get_pacing_profile"]}
    spec = agent._compact_runtime_phase_spec(agent.phases[1])
    assert any(name == "get_pacing_profile" for name, _ in spec.prefetch_tool_calls)
    assert len(spec.prefetch_tool_calls) <= DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_tool_ceiling


def evidence_row(tool, mandatory=3, optional=10):
    atoms = [{"citation": f"Q{tool}{index}", "value": index, "unit": "ms", "reliability": "reliable", "caveats": [],
              **({"required_by": [f"{tool}:node"]} if index < mandatory else {})}
             for index in range(mandatory + optional)]
    return {"tool": tool, "arguments": {}, "result": {"contract": "ecgagent.model-evidence.v5", "evidence": atoms, "omitted_atom_count": 0}}


@pytest.mark.parametrize("limit", [1, 180, 600, 1200, 2400, 8000])
def test_packer_preserves_mandatory_bundles_and_exact_hard_limit(limit):
    rows = [evidence_row(str(i)) for i in range(5)]
    original = copy.deepcopy(rows)
    result = pack_evidence_rows(rows, limit)
    assert rows == original
    # An empty JSON array takes two characters even when no view fits.
    assert json_size(result) <= max(2, limit)
    for row in result:
        atoms = row["result"]["evidence"]
        assert sum(bool(atom.get("required_by")) for atom in atoms) in {0, 3}
        assert len(atoms) + row["result"]["omitted_atom_count"] == 13


def test_model_view_preserves_cross_lead_minima_and_reports_omitted_pointers():
    store = EvidenceStore.from_dict(_payload()).diagnostic_view()
    pointers = [f"/representative_leads/{lead}/params/{field}"
                for lead in store.leads for field in ("r_amp_mv", "s_amp_mv", "t_amp_mv")
                if store.try_resolve(f"/representative_leads/{lead}/params/{field}")]
    view = build_model_evidence_view(store, tool="get_lead_table", arguments={"fields": ["r_amp_mv", "s_amp_mv", "t_amp_mv"]},
                                    citations=pointers, rendered_text=" ".join(f"ev:{p}" for p in pointers),
                                    required_by=("h1:morphology",), char_limit=8000)
    assert view.required_evidence["h1:morphology"]
    assert set(view.required_evidence["h1:morphology"]) <= set(view.citations)
    assert all(atom["required_by"] == ["h1:morphology"] for atom in json.loads(view.text)["evidence"])


def test_expired_deadline_never_calls_http_client():
    capture = Capture([])
    with pytest.raises(TimeoutError):
        openai_completion(SimpleNamespace(chat=SimpleNamespace(completions=capture)), {},
                          deadline=time.perf_counter() - 1, timeout=600, max_retries=2)
    assert capture.calls == []


def test_request_deadline_disables_sdk_retries_and_bounds_timeout():
    capture = Capture([object()])
    options = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=capture))
    client.with_options = lambda **kw: options.append(kw) or client
    openai_completion(client, {}, deadline=time.perf_counter() + 5, timeout=600, max_retries=2)
    assert options[0]["max_retries"] == 0
    assert 0 < options[0]["timeout"] <= 5


def test_full_cohort_counts_abstentions_as_false_negatives():
    metrics = label_metrics([{"reference_categories": {"af"}, "agent_categories": {"af"}},
                             {"reference_categories": {"af"}, "agent_categories": set()}])
    assert metrics["agent"]["recall"] == .5
    assert metrics["agent"]["f1"] == pytest.approx(2 / 3)
    assert metrics["record_count"] == 2


def test_performance_includes_failed_attempt_requests():
    payload = {"batch": {"attempts": [
        {"model_requests": [{"phase": "plan", "elapsed_seconds": 2, "prompt_tokens": 10, "completion_tokens": 2}]},
        {"model_requests": [{"phase": "plan", "elapsed_seconds": 3, "prompt_tokens": 20, "completion_tokens": 4}]},
    ]}}
    result = summarize_performance([payload])
    assert result["model_calls_per_record"]["mean"] == 2
    assert result["prompt_tokens_per_record"]["mean"] == 30
    assert result["complete_request_history_records"] == 1


def test_budget_selection_prefers_urgent_candidate_over_earlier_expensive_path():
    agent = ECGDiagnosticAgent(store=EvidenceStore.from_dict(_payload()), backend=_empty_backend())
    def candidate(identifier, code, count):
        return {"id": identifier, "code": code, "domains": ["rhythm_rate"],
                "diagnostic_pathway": {"steps": [
                    {"id": f"s{i}", "tool": "get_global_table", "arguments": {"fields": [f"test_field_{identifier}_{i}"]}}
                    for i in range(count)]}}
    expensive = candidate("h1", "sinus_rhythm", 14)
    urgent = candidate("h2", "wide_complex_tachycardia", 2)
    kept, dropped = agent._compact_bound_candidates_by_view_budget([expensive, urgent])
    assert [row["id"] for row in kept] == ["h2"]
    assert dropped[0]["id"] == "h1"
    reversed_kept, _ = agent._compact_bound_candidates_by_view_budget([urgent, expensive])
    assert [row["id"] for row in reversed_kept] == ["h2"]


def test_model_node_coverage_reports_hidden_required_evidence():
    from dataclasses import replace
    agent = ECGDiagnosticAgent(store=EvidenceStore.from_dict(_payload()), backend=_empty_backend())
    spec = replace(agent.phases[1], model_evidence_requirements=(("h1:shape", "get_lead_table", ()),))
    agent._record_evidence_coverage(spec, {"h1:shape": ("/not/visible",)})
    assert agent._compact_evidence_coverage["h1:shape"]["status"] == "omitted"
    assert agent._compact_evidence_coverage["h1:shape"]["visible_atom_count"] == 0


def test_batch_full_cohort_keeps_missing_and_failed_records(tmp_path):
    from ecgagent.batch import analyze_results
    from ecgagent.agent.diagnostic import DIAGNOSTIC_AGENT_PROTOCOL_VERSION, DIAGNOSTIC_PROMPT_FINGERPRINT
    records = [{"record": name, "reference_available": True, "reference_codes_active": ["AFIB"]}
               for name in ("ok", "failed", "missing")]
    (tmp_path / "diagnoses").mkdir()
    for name, verified in (("ok", True), ("failed", False)):
        payload = {"record_id": name, "verified": verified, "error": None if verified else "timeout",
                   "audit": {"agent_protocol": DIAGNOSTIC_AGENT_PROTOCOL_VERSION, "prompt_fingerprint": DIAGNOSTIC_PROMPT_FINGERPRINT},
                   "verdict": {"diagnoses": [{"code": "atrial_fibrillation_pattern", "status": "added"}]},
                   "source": {"baseline_codes": []}}
        (tmp_path / "diagnoses" / f"{name}.json").write_text(json.dumps(payload))
    (tmp_path / "diagnosis_manifest.json").write_text(json.dumps({"workers": 2, "records_per_second": .5}))
    report = analyze_results(records, tmp_path)
    full = report["full_cohort_direct_label_agreement"]
    assert full["record_count"] == 3
    assert full["agent"]["tp"] == 1
    assert full["agent"]["fn"] == 2
    assert full["agent"]["recall"] == pytest.approx(1 / 3)
    assert report["label_evaluated_records"] == 1
    manifest = json.loads((tmp_path / "diagnosis_manifest.json").read_text())
    assert manifest["workers"] == 2 and manifest["records_per_second"] == .5


def test_medgemma_atomic_excerpt_remains_json_with_complete_atoms():
    from ecgagent.backends.medgemma_local import _compact_tool_excerpt
    row = evidence_row("lead", mandatory=3, optional=60)
    text, _ = _compact_tool_excerpt(json.dumps(row["result"]), 1000)
    parsed = json.loads(text)
    assert len(text) <= 1000
    assert sum(bool(atom.get("required_by")) for atom in parsed["evidence"]) in {0, 3}


def test_missing_measurements_complete_acquisition_but_bad_arguments_do_not():
    from dataclasses import replace
    agent = ECGDiagnosticAgent(store=EvidenceStore.from_dict(_payload()), backend=_empty_backend())
    args = (("include_beats", False),)
    spec = replace(agent.phases[1], required_tool_calls=(("get_morphology_groups", args, 1),))
    agent.registry.begin_phase("adjudicate", 2)
    result = agent.registry.call("get_morphology_groups", dict(args))
    assert not result.ok and result.note == "measurement_unavailable"
    assert agent._unmet_runtime_coverage(spec) == []
    assert agent.registry.calls[-1].citations == ()
    bad_args = (("invalid_argument", True),)
    spec = replace(spec, required_tool_calls=(("get_morphology_groups", bad_args, 1),))
    agent.registry.call("get_morphology_groups", dict(bad_args))
    assert agent._unmet_runtime_coverage(spec)


def test_thinking_budget_reserves_output_room_and_survives_session_clone():
    capture = Capture([_response(content="{}")])
    backend = QwenLocalBackend(client=SimpleNamespace(chat=SimpleNamespace(completions=capture)), thinking_token_budget=768)
    session = backend.new_session()
    session.complete(system="S", messages=[], tools=[], phase="adjudicate", max_tokens=1200)
    assert capture.calls[0]["extra_body"]["thinking_token_budget"] == 400
    assert session.audit_config()["thinking_token_budget"] == 768


def test_http_retries_recalculate_remaining_record_budget(monkeypatch):
    from ecgagent.backends import request_budget
    clock = [100.0]
    monkeypatch.setattr(request_budget.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(request_budget.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    seen = []
    class Retryable(Exception):
        status_code = 429
    class Completions:
        attempts = 0
        def create(self, **kwargs):
            self.attempts += 1
            clock[0] += 1
            if self.attempts == 1:
                raise Retryable()
            return "done"
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.with_options = lambda **kwargs: seen.append(kwargs) or client
    assert openai_completion(client, {}, deadline=105, timeout=600, max_retries=2) == "done"
    assert [row["timeout"] for row in seen] == [5, 3.75]
    assert all(row["max_retries"] == 0 for row in seen)
