from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ecgagent.backends.base import ToolOutcome
from ecgagent.backends.deepseek import DeepSeekBackend
from ecgagent.backends.qwen_local import QwenLocalBackend
from ecgagent.evidence.ledger import EvidenceLedger
from ecgagent.evidence.packing import (
    DEFAULT_MAX_REQUIRED_EVIDENCE_CHARS,
    DEFAULT_MAX_RETAINED_EVIDENCE_CHARS,
    DEFAULT_REQUIRED_EVIDENCE_HEADROOM_CHARS,
    EVIDENCE_PACKING_POLICY_VERSION,
    EvidencePackingPolicy,
    json_size,
    pack_evidence_rows,
    required_packet_size,
)
from ecgagent.performance import summarize_performance


BACKENDS = (QwenLocalBackend, DeepSeekBackend)


def _row(view: int, *, mandatory: int = 3, optional: int = 5, width: int = 10):
    prefixes = ("/representative_leads/V1/params", "/quality/signals", "/rhythm_inputs/rr")
    atoms = [
        {
            "citation": f"ev:{prefixes[view % len(prefixes)]}/measurement_{view}_{index}",
            "value": '数值"\\\n' + "x" * width,
            "unit": "ms",
            "reliability": "limited",
            "caveats": ["Independent confirmation unavailable"],
            **({"required_by": [f"h{view}:shape"]} if index < mandatory else {}),
        }
        for index in range(mandatory + optional)
    ]
    return {
        "tool": "get_lead_table",
        "arguments": {"view": view},
        "result": {
            "contract": "ecgagent.model-evidence.v5",
            "tool": "get_lead_table",
            "arguments": {"view": view},
            "evidence": atoms,
            "omitted_atom_count": 2,
        },
    }


def _outcomes(rows):
    outcomes = []
    for index, row in enumerate(rows):
        pointers = tuple(atom["citation"].removeprefix("ev:") for atom in row["result"]["evidence"])
        outcomes.append(ToolOutcome(
            call_id=f"prefetch-adjudicate-{index}",
            name=row["tool"],
            arguments=row["arguments"],
            text="Unabridged audit table",
            citations=(*pointers, f"/hidden/{index}"),
            model_text=json.dumps(row["result"], ensure_ascii=False),
            model_citations=pointers,
            model_candidate_citations=pointers,
            model_omitted_count=2,
        ))
    return outcomes


def _packet(backend, outcomes):
    messages = backend.tool_result_turn(outcomes)
    if isinstance(backend, QwenLocalBackend):
        messages = backend.compact_phase_context("adjudicate", messages)
        rows = json.loads(messages[-1]["content"].split(":\n", 1)[1])["evidence_views"] if messages else []
    else:
        rows = json.loads(messages[-1]["content"].split(":\n", 1)[1])
    return messages, rows


def test_shared_defaults_and_policy_audit():
    assert DEFAULT_MAX_RETAINED_EVIDENCE_CHARS == 8000
    assert DEFAULT_MAX_REQUIRED_EVIDENCE_CHARS == 24000
    assert DEFAULT_REQUIRED_EVIDENCE_HEADROOM_CHARS == 1024
    policy = EvidencePackingPolicy()
    for backend_type in BACKENDS:
        backend = backend_type(client=object())
        assert backend.evidence_packing_policy() == policy
        audit = backend.audit_config()["context_compaction"]
        assert {key: audit[key] for key in policy.audit_config()} == policy.audit_config()
        assert audit["packing_policy_version"] == EVIDENCE_PACKING_POLICY_VERSION


@pytest.mark.parametrize("target,ceiling,width", [
    (8000, 24000, 280), (8000, 24000, 1200), (8000, None, 280),
    (1200, 5000, 280), (1, None, 280), (8000, 4000, 280),
])
def test_backend_packets_and_actual_authorization_match(target, ceiling, width):
    source = [_row(index, mandatory=8, optional=12, width=width) for index in range(3)]
    outcomes = _outcomes(source)
    original = copy.deepcopy(outcomes)
    packets, authorizations, audits = [], [], []
    for backend_type in BACKENDS:
        backend = backend_type(client=object(), max_retained_evidence_chars=target,
                               max_required_evidence_chars=ceiling)
        messages, packed = _packet(backend, outcomes)
        visible = backend.model_visible_citations(outcomes, messages)
        aliases = backend.citation_aliases()
        rendered_pointers = {aliases[atom["citation"]] for row in packed for atom in row["result"]["evidence"]}
        ledger = EvidenceLedger()
        ledger.record_touched(pointer for outcome in outcomes for pointer in outcome.citations)
        for outcome in outcomes:
            expected = tuple(pointer for pointer in outcome.model_citations if pointer in rendered_pointers)
            assert visible[outcome.call_id] == expected
            ledger.authorize(visible[outcome.call_id], source=outcome.call_id)
        assert ledger.visible == rendered_pointers
        assert all(f"/hidden/{index}" in ledger.hidden for index in range(3))
        assert len(ledger.visible) < len(ledger.touched)
        trace = backend.tool_context_trace()
        assert {pointer for row in trace for pointer in row["visible_citations"]} == rendered_pointers
        assert all(row["omitted_atom_count"] + len(row["visible_citations"]) == 22 for row in trace)
        for row in packed:
            atoms = row["result"]["evidence"]
            assert sum(bool(atom.get("required_by")) for atom in atoms) in {0, 8}
            assert len(atoms) + row["result"]["omitted_atom_count"] == 22
            assert all({"value", "unit", "reliability", "caveats"} <= atom.keys() for atom in atoms)
        audit = backend.audit_config()["context_compaction"]
        assert json_size(packed) <= max(2, audit["effective_evidence_chars"])
        packets.append(packed)
        authorizations.append(visible)
        audits.append({key: audit[key] for key in backend.evidence_packing_policy().audit_config()})
    assert packets[0] == packets[1]
    assert authorizations[0] == authorizations[1]
    assert audits[0] == audits[1]
    if ceiling == 24000:
        assert audits[0]["effective_evidence_chars"] > target
        if width == 280:
            assert sum(bool(atom.get("required_by")) for row in packets[0] for atom in row["result"]["evidence"]) == 24
        else:
            assert audits[0]["effective_evidence_chars"] == 24000
            assert 0 < sum(len(row["result"]["evidence"]) for row in packets[0]) < 24
    assert outcomes == original


def test_optional_shells_do_not_starve_required_bundle_or_trigger_growth():
    optional = [_row(index, mandatory=0, optional=1) for index in range(20)]
    required = _row(21, mandatory=3, optional=1)
    raw = {"tool": "raw", "result": {"evidence": [{"required_by": ["h:raw"], "value": "x" * 5000}]}}
    rows = [*optional, raw, required]
    size = required_packet_size(rows)
    assert required_packet_size([*optional, raw]) == 2
    assert size == required_packet_size([required])
    original = copy.deepcopy(rows)
    packed = pack_evidence_rows(rows, size)
    assert json_size(packed) == size
    assert len(packed) == 1
    assert packed[0]["arguments"] == required["arguments"]
    assert len(packed[0]["result"]["evidence"]) == 3
    assert rows == original
    assert EvidencePackingPolicy(size).effective_limit(rows) == size
    assert EvidencePackingPolicy(1).effective_limit([*optional, raw]) == 1


def test_growth_ceiling_and_scheduler_estimate_are_exact():
    rows = [_row(index, mandatory=3, optional=5) for index in range(4)]
    size = required_packet_size(rows)
    policy = EvidencePackingPolicy(600, 6000, 37)
    assert policy.effective_limit(rows) == size + 37
    packed = pack_evidence_rows(rows, size)
    assert json_size(packed) == size
    assert sum(len(row["result"]["evidence"]) for row in packed) == 12
    assert EvidencePackingPolicy(600, size - 1, 37).effective_limit(rows) == size - 1
    assert EvidencePackingPolicy(600, None).effective_limit(rows) == 600
    assert EvidencePackingPolicy(size, 600).effective_limit(rows) == size
    assert EvidencePackingPolicy(600, 6000, 0).effective_limit(rows) == size


def test_oversized_bundle_does_not_starve_smaller_required_view():
    oversized = _row(0, mandatory=3, optional=1, width=9000)
    smaller = _row(1, mandatory=3, optional=0)
    packed = pack_evidence_rows([oversized, smaller], 5000)
    assert json_size(packed) <= 5000
    assert packed[0]["result"]["evidence"] == []
    assert packed[0]["result"]["omitted_atom_count"] == 6
    assert packed[1]["result"]["evidence"] == smaller["result"]["evidence"]


@pytest.mark.parametrize("limit", [1, 2, 120, 350, 700, 1100, 2500, 8000])
def test_atomic_bundles_and_exact_unicode_json_limit(limit):
    rows = [_row(index, mandatory=3, optional=9) for index in range(5)]
    original = copy.deepcopy(rows)
    packed = pack_evidence_rows(rows, limit)
    assert json_size(packed) <= max(2, limit)
    assert rows == original
    for row in packed:
        atoms = row["result"]["evidence"]
        assert sum(bool(atom.get("required_by")) for atom in atoms) in {0, 3}
        assert len(atoms) + row["result"]["omitted_atom_count"] == 14


@pytest.mark.parametrize("field,value", [
    ("max_retained_evidence_chars", 0), ("max_retained_evidence_chars", True),
    ("max_required_evidence_chars", -1), ("max_required_evidence_chars", "24000"),
    ("max_required_evidence_chars", False), ("required_evidence_headroom_chars", -1),
    ("required_evidence_headroom_chars", 1.5),
])
def test_shared_policy_validation_before_client_creation(field, value):
    for constructor in (EvidencePackingPolicy, *BACKENDS):
        with pytest.raises(ValueError, match=field):
            constructor(**{field: value})


@pytest.mark.parametrize("backend_type", BACKENDS)
def test_session_clone_preserves_mutable_settings_and_isolates_audit(backend_type):
    backend = backend_type(client=object(), name="custom", max_retained_evidence_chars=8000)
    backend.max_retained_evidence_chars = 700
    backend.max_required_evidence_chars = 6000
    backend.required_evidence_headroom_chars = 41
    _packet(backend, _outcomes([_row(0, mandatory=8)]))
    backend.turns.append({"usage": {"prompt_tokens": 12}})
    backend.reasoning.append("prior reasoning")
    assert backend.audit_config()["context_compaction"]["effective_evidence_chars"] > 700
    clone = backend.new_session()
    assert clone.client is backend.client
    assert clone.name == "custom"
    assert clone.evidence_packing_policy() == backend.evidence_packing_policy()
    assert clone.audit_config()["context_compaction"]["effective_evidence_chars"] == 700
    assert clone.turns == clone.reasoning == clone.tool_context_trace() == []
    assert clone.citation_aliases() == {}
    clone.reset_session()
    assert backend.citation_aliases() and backend.tool_context_trace()
    backend.reset_session()
    assert backend.turns == backend.reasoning == backend.tool_context_trace() == []
    assert backend.citation_aliases() == {}
    assert backend.audit_config()["context_compaction"]["effective_evidence_chars"] == 700


@pytest.mark.parametrize("backend_type", BACKENDS)
def test_trace_visibility_without_optional_candidate_manifest(backend_type):
    backend = backend_type(client=object())
    outcome = replace(_outcomes([_row(0)])[0], model_candidate_citations=())
    messages, _ = _packet(backend, [outcome])
    visible = backend.model_visible_citations([outcome], messages)[outcome.call_id]
    assert visible == outcome.model_citations
    trace = backend.tool_context_trace()[0]
    assert trace["visible_citations"] == list(visible)
    assert trace["omitted_atom_count"] == 2
    assert trace["truncated_for_model"] is True


@pytest.mark.parametrize("backend_type", BACKENDS)
def test_repeated_phase_batches_preserve_aliases_and_historical_trace(backend_type):
    backend = backend_type(client=object(), max_required_evidence_chars=None)
    backend.begin_evidence_batch("adjudicate")
    first = _outcomes([_row(0)])
    _packet(backend, first)
    aliases = backend.citation_aliases()
    history = copy.deepcopy(backend.tool_context_trace())
    backend.turns.append({"usage": {"prompt_tokens": 12}})
    if isinstance(backend, QwenLocalBackend):
        backend._inflight_evidence["plan"] = {"sentinel": {"result": "plan"}}
        backend._retained_evidence["sentinel"] = {"result": "retained"}
        backend._current_phase_state = {"state": "keep"}
    backend.begin_evidence_batch("adjudicate")
    assert backend.citation_aliases() == aliases
    assert backend.tool_context_trace() == history
    assert backend.turns == [{"usage": {"prompt_tokens": 12}}]
    if isinstance(backend, QwenLocalBackend):
        assert not backend._inflight_evidence.get("adjudicate")
        assert backend._inflight_evidence["plan"] == {"sentinel": {"result": "plan"}}
        assert backend._retained_evidence == {"sentinel": {"result": "retained"}}
        assert backend._current_phase_state == {"state": "keep"}
    # Reuse both tool signature and call id, with different evidence values.
    second_row = _row(1)
    second_row["arguments"] = second_row["result"]["arguments"] = {"view": 0}
    second = _outcomes([second_row])
    messages, _ = _packet(backend, second)
    assert backend.model_visible_citations(first, messages)[first[0].call_id] == ()
    assert backend.model_visible_citations(second, messages)[second[0].call_id]
    assert backend.tool_context_trace()[:len(history)] == history
    assert all(backend.citation_aliases()[key] == value for key, value in aliases.items())
    # An entirely omitted third batch must not leave the previous packet visible.
    backend.begin_evidence_batch("adjudicate")
    backend.max_retained_evidence_chars = 1
    history = copy.deepcopy(backend.tool_context_trace())
    messages, packed = _packet(backend, second)
    assert packed == []
    assert backend.model_visible_citations(second, messages)[second[0].call_id] == ()
    assert backend.tool_context_trace()[:len(history)] == history
    assert backend.tool_context_trace()[-1]["visible_citations"] == []


def _runtime(requests, elapsed=7):
    return {"audit": {"runtime_controls": {"elapsed_seconds": elapsed, "model_requests": requests}}}


@pytest.mark.parametrize("batch_duration,expected", [(None, 7), (0, 0), (12, 12), (float("nan"), 7), (True, 7), (-1, 7)])
def test_latency_uses_cli_runtime_fallback_and_valid_batch_precedence(batch_duration, expected):
    payload = _runtime([], elapsed=7)
    payload["batch"] = {"runtime_seconds": batch_duration}
    result = summarize_performance([payload])
    assert result["record_latency_seconds"] == {"count": 1, "p50": expected, "p95": expected, "mean": expected}
    assert result["prompt_tokens_per_record"]["mean"] == 0  # Explicitly no calls.


def test_missing_and_partial_token_telemetry_are_not_zero_samples():
    result = summarize_performance([
        _runtime([{"prompt_tokens": 10, "completion_tokens": 2},
                  {"prompt_tokens": None, "completion_tokens": 3, "stop_reason": "error"}]),
        _runtime([{"prompt_tokens": 0, "completion_tokens": 0}]),
        _runtime([]),
        {},
    ])
    assert result["record_count"] == 4
    assert result["model_calls_per_record"]["count"] == 3
    assert result["prompt_tokens_per_record"] == {"count": 2, "p50": 0, "p95": 0, "mean": 0}
    assert result["completion_tokens_per_record"]["count"] == 3
    assert result["completion_tokens_per_record"]["mean"] == pytest.approx(5 / 3)
    assert result["token_telemetry"]["prompt_tokens"] == {
        "observed_requests": 2, "missing_requests": 1, "observed_tokens": 10,
        "complete_records": 2, "incomplete_records": 2,
    }
    absent = summarize_performance([_runtime([{}]), {}])
    assert absent["prompt_tokens_per_record"] == {"count": 0, "p50": None, "p95": None, "mean": None}
    assert absent["completion_tokens_per_record"]["mean"] is None


def test_failed_attempts_keep_latencies_and_partial_token_subtotals():
    payload = _runtime([{"prompt_tokens": 999, "completion_tokens": 999}])
    payload["batch"] = {"attempts": [
        {"model_requests": [{"phase": "plan", "elapsed_seconds": 2,
                             "prompt_tokens": None, "completion_tokens": None}]},
        {"model_requests": [{"phase": "plan", "elapsed_seconds": 4,
                             "prompt_tokens": 20, "completion_tokens": 5}]},
    ]}
    result = summarize_performance([payload])
    assert result["model_calls_per_record"]["mean"] == 2
    assert result["model_request_latency_by_phase"]["plan"]["mean"] == 3
    assert result["complete_request_history_records"] == 1
    assert result["prompt_tokens_per_record"]["mean"] is None
    assert result["token_telemetry"]["prompt_tokens"]["observed_tokens"] == 20


def test_legacy_partial_request_history_is_not_reported_as_record_token_total():
    payload = _runtime([{"prompt_tokens": 20, "completion_tokens": 5}])
    payload["batch"] = {"attempts": [{"error": "timeout"}, {"model_requests": None}]}
    result = summarize_performance([payload])
    assert result["complete_request_history_records"] == 0
    assert result["prompt_tokens_per_record"]["mean"] is None
    assert result["token_telemetry"]["prompt_tokens"]["observed_tokens"] == 20


@pytest.mark.parametrize("invalid", [None, True, -1, 1.5, "4", float("nan"), float("inf")])
def test_invalid_token_telemetry_stays_unknown(invalid):
    result = summarize_performance([_runtime([{"prompt_tokens": invalid, "completion_tokens": invalid}])])
    assert result["prompt_tokens_per_record"]["mean"] is None
    assert result["completion_tokens_per_record"]["mean"] is None


def test_deepseek_schema_minification_preserves_the_complete_contract():
    calls = []
    response = SimpleNamespace(
        model="test", usage=None,
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content="{}", reasoning_content=None, tool_calls=[]))],
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or response)))
    backend = DeepSeekBackend(client=client)
    schema = {"type": "object", "properties": {"candidate_updates": {"type": "array", "maxItems": 2}},
              "required": ["candidate_updates"], "additionalProperties": False}
    original = copy.deepcopy(schema)
    backend.complete(system="S", messages=[], tools=[], phase="adjudicate", response_schema=schema)
    text = calls[0]["messages"][-1]["content"]
    assert json.loads(text[text.index("{"):]) == original == schema
    assert calls[0]["max_tokens"] == backend.max_tokens
    assert len(text[text.index("{"):]) < len(json.dumps(schema, indent=2))
