"""Contract tests for the agent loop and the Anthropic backend.

The backend tests assert on the *request* the backend builds, not on a live
call. Several of those fields are the difference between a working request and
a 400 on this model family (`temperature` is rejected; `budget_tokens` is
rejected), and one of them — the cache breakpoint — is invisible in a passing
response but re-bills the whole tool+system prefix on every turn when absent.
A fake client is the only way to hold them still without an API key.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ecgagent.agent import prompts
from ecgagent.agent.loop import DEFAULT_PHASES, ECGAgent, PhaseSpec, adaptive_phases
from ecgagent.backends.anthropic_api import AnthropicBackend, parse_json_text
from ecgagent.backends.base import LLMResponse, ToolCall, ToolOutcome
from ecgagent.backends.deepseek import DeepSeekBackend
from ecgagent.backends.mock import ScriptedBackend, text_response, tool_response
from ecgagent.evidence.briefing import build_chart_briefing
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_tools import _payload


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore.from_dict(_payload(), record_id="TEST001")


def _verdict(pr_value: float = 214.0, claim: str = "PR interval") -> dict:
    return {
        "summary": "Confirmed first-degree AV delay.",
        "diagnoses": [
            {
                "code": "first_degree_av_delay",
                "statement": "First-degree AV delay",
                "status": "unchanged",
                "confidence": "MEDIUM",
                "rule_ids": ["CLIN-INTERVAL-PR-01"],
                "evidence": [
                    {
                        "claim": claim,
                        "value": pr_value,
                        "unit": "ms",
                        "citations": ["ev:/global_features/pr_ms"],
                    }
                ],
                "counterevidence": [],
                "adjudication": "",
            }
        ],
        "abstentions": [],
        "human_review": {
            "required": True,
            "reasons": ["baseline clinical rules require physician review"],
        },
    }


def _happy_script(verdict: dict | None = None) -> list[LLMResponse]:
    return [
        text_response("H1 [confirm_rule] PR 214 ms supports first-degree AV delay."),
        tool_response(("get_measurement", {"pointer": "/global_features/pr_ms"})),
        text_response("H1 supported."),
        text_response("No conflicts."),
        text_response(json.dumps(verdict or _verdict())),
    ]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def test_loop_runs_every_phase_and_produces_a_verified_verdict(store):
    backend = ScriptedBackend(script=_happy_script())
    result = ECGAgent(store=store, backend=backend).run()

    assert result.ok and result.verified, result.summary()
    assert [phase.key for phase in result.phases] == [spec.key for spec in DEFAULT_PHASES]
    assert result.revisions == 0
    assert result.verdict["diagnoses"][0]["code"] == "first_degree_av_delay"


def test_default_evidence_budgets_adapt_to_chart_complexity(store):
    simple_budgets = {
        phase.key: phase.tool_budget for phase in adaptive_phases(store)
    }
    assert simple_budgets["test"] == 10
    assert simple_budgets["adjudicate"] == 3

    payload = _payload()
    final = payload["clinical_interpretation"]["final_statements"][0]
    payload["clinical_interpretation"]["final_statements"] = [
        {**final, "rule_id": f"CLIN-X-{index}", "statement_code": f"code_{index}"}
        for index in range(8)
    ]
    payload["clinical_interpretation"]["suppressed_statements"] = [
        {
            **final,
            "rule_id": f"CLIN-S-{index}",
            "statement_code": f"suppressed_{index}",
        }
        for index in range(4)
    ]
    complex_store = EvidenceStore.from_dict(payload, record_id="COMPLEX")
    complex_budgets = {
        phase.key: phase.tool_budget
        for phase in adaptive_phases(complex_store)
    }
    assert complex_budgets["test"] == 18
    assert complex_budgets["adjudicate"] == 5


def test_orient_and_synthesize_get_no_tool_schemas(store):
    """An unusable tool in context is an invitation to try it and waste a turn."""
    backend = ScriptedBackend(script=_happy_script())
    ECGAgent(store=store, backend=backend).run()
    tool_counts = [call["n_tools"] for call in backend.calls]
    assert tool_counts[0] == 0        # orient
    assert tool_counts[1] > 0         # test
    assert tool_counts[-1] == 0       # synthesize


def test_synthesis_turn_requests_structured_output(store):
    backend = ScriptedBackend(script=_happy_script())
    ECGAgent(store=store, backend=backend).run()
    assert backend.calls[-1]["structured"] is True
    assert backend.calls[0]["structured"] is False


def test_phase_budget_is_enforced_and_audited(store):
    """A phase that keeps calling tools is cut off at its budget, not the model's discretion."""
    greedy = [tool_response(("get_measurement", {"pointer": "/global_features/pr_ms"}))] * 8
    script = [
        text_response("hypotheses"),
        *greedy,
        text_response("no conflicts"),
        text_response(json.dumps(_verdict())),
    ]
    phases = (
        PhaseSpec("orient", prompts.ORIENT_INSTRUCTION, tool_budget=0),
        PhaseSpec("test", prompts.TEST_INSTRUCTION, tool_budget=3),
        PhaseSpec("adjudicate", prompts.ADJUDICATE_INSTRUCTION, tool_budget=0),
        PhaseSpec("synthesize", prompts.SYNTHESIZE_INSTRUCTION, tool_budget=0, structured=True),
    )
    events: list[str] = []
    agent = ECGAgent(
        store=store,
        backend=ScriptedBackend(script=script),
        phases=phases,
        on_event=events.append,
    )
    result = agent.run()

    test_phase = next(phase for phase in result.phases if phase.key == "test")
    assert test_phase.tool_calls == 3
    assert agent.registry.audit()["n_calls"] == 8
    assert agent.registry.remaining == 0
    # The model kept asking after the budget ran out. Those attempts did not
    # read more evidence, but they remain visible as rejected audit records.
    assert test_phase.rejected_calls == 4
    assert test_phase.turns > 3
    assert any("ok=False" in event for event in events)


def test_budgets_are_cumulative_so_the_audit_stays_one_list(store):
    script = [
        text_response("hypotheses"),
        tool_response(("get_measurement", {"pointer": "/global_features/pr_ms"})),
        text_response("done"),
        tool_response(("get_measurement", {"pointer": "/global_features/qrs_ms"})),
        text_response("no conflicts"),
        text_response(json.dumps(_verdict())),
    ]
    agent = ECGAgent(store=store, backend=ScriptedBackend(script=script))
    agent.run()
    audit = agent.registry.audit()
    assert audit["n_calls"] == 2
    assert [call["phase"] for call in audit["calls"]] == ["test", "adjudicate"]


def test_parallel_tool_calls_return_in_one_message(store):
    """Splitting results across messages trains the model out of parallel calls."""
    script = [
        text_response("hypotheses"),
        tool_response(
            ("get_measurement", {"pointer": "/global_features/pr_ms"}),
            ("get_measurement", {"pointer": "/global_features/qrs_ms"}),
        ),
        text_response("done"),
        text_response("no conflicts"),
        text_response(json.dumps(_verdict())),
    ]
    backend = ScriptedBackend(script=script)
    ECGAgent(store=store, backend=backend).run()
    messages = backend.tool_result_turn(
        [ToolOutcome("a", "one"), ToolOutcome("b", "two", is_error=True)]
    )
    # Anthropic shape: every result batched into one user message.
    assert len(messages) == 1
    assert messages[0]["role"] == "user" and len(messages[0]["content"]) == 2
    assert messages[0]["content"][1]["is_error"] is True


def test_tool_calls_are_answered_even_when_provider_reports_max_tokens():
    response = LLMResponse(
        stop_reason="max_tokens",
        tool_calls=(
            ToolCall(
                id="call_1",
                name="get_measurement",
                arguments={"pointer": "PR"},
            ),
        ),
    )

    assert response.wants_tools


def test_normalizer_qualifies_the_exact_caveated_evidence_item(store):
    """A paragraph-level warning cannot qualify a different structured claim."""
    candidate = _verdict()
    evidence = candidate["diagnoses"][0]["evidence"][0]
    evidence.update(
        {
            "claim": "QTcB reads 407 ms",
            "value": 407,
            "unit": "ms",
            "citations": ["ev:/global_features/qtc_bazett_ms"],
        }
    )
    agent = ECGAgent(store=store, backend=ScriptedBackend(script=[]))

    normalized = agent._normalize_verdict(candidate)

    claim = normalized["diagnoses"][0]["evidence"][0]["claim"]
    assert "limited evidence" in claim
    assert agent._verdict_normalizations[-1]["kind"] == (
        "caveated_evidence_qualification"
    )


def test_phase_feedback_with_literal_braces_is_not_formatted_as_a_field(store):
    backend = ScriptedBackend(script=[text_response("revision complete")])
    agent = ECGAgent(store=store, backend=backend)
    phase = PhaseSpec(
        "revise",
        "Caveat: qt_excluded_leads={aVF=qt_outside_path_bounds}; budget={budget}",
        tool_budget=0,
    )

    result = agent._run_phase(phase, [], [])

    assert result.text == "revision complete"


def test_failed_verification_triggers_a_bounded_revision(store):
    bad = _verdict(pr_value=168.0)           # contradicts /global_features/pr_ms = 214
    # Keep the item deliberately ambiguous: a single direct numeric citation
    # is now deterministically materialized from the store before verification.
    bad["diagnoses"][0]["evidence"][0]["citations"].append(
        "ev:/global_features/qrs_ms"
    )
    script = [*_happy_script(bad), text_response(json.dumps(_verdict()))]
    result = ECGAgent(store=store, backend=ScriptedBackend(script=script)).run()

    assert result.revisions == 1
    assert result.verified
    assert result.phases[-1].key == "revise"


def test_unparseable_initial_synthesis_uses_revision_allowance(store):
    script = [
        text_response("hypotheses"),
        tool_response(
            ("get_measurement", {"pointer": "/global_features/pr_ms"})
        ),
        text_response("supported"),
        text_response("no conflicts"),
        text_response("not JSON"),
        text_response(json.dumps(_verdict())),
    ]

    result = ECGAgent(
        store=store,
        backend=ScriptedBackend(script=script),
        max_revisions=1,
    ).run()

    assert result.revisions == 1
    assert result.verified
    assert result.error is None


def test_contract_feedback_names_citable_baseline_repair_route(store):
    verdict = _verdict()
    verdict["diagnoses"][0]["evidence"][0]["citations"] = []
    agent = ECGAgent(store=store, backend=ScriptedBackend())
    briefing = build_chart_briefing(store)
    agent.registry.seed_citations(
        briefing.citations,
        source="chart_briefing",
        model_text=briefing.text,
    )

    feedback = agent._contract_feedback(
        verdict,
        ["diagnoses[0].evidence[0] has no evidence citation"],
    )

    assert "get_rule_detail" in feedback
    assert "copy the exact ev:/ pointers returned" in feedback


def test_revision_gives_up_after_max_revisions(store):
    bad_verdict = _verdict(pr_value=168.0)
    bad_verdict["diagnoses"][0]["evidence"][0]["citations"].append(
        "ev:/global_features/qrs_ms"
    )
    bad = json.dumps(bad_verdict)
    script = [*_happy_script(bad_verdict), text_response(bad), text_response(bad)]
    result = ECGAgent(
        store=store, backend=ScriptedBackend(script=script), max_revisions=2
    ).run()

    assert result.revisions == 2
    assert not result.verified
    assert result.verification.counts()["value_mismatch"] >= 1


def test_unparseable_synthesis_is_reported_not_raised(store):
    script = [*_happy_script()[:-1], text_response("I could not produce JSON.")]
    result = ECGAgent(
        store=store,
        backend=ScriptedBackend(script=script),
        max_revisions=0,
    ).run()
    assert not result.ok
    assert "parseable JSON" in result.error


def test_a_refusal_stops_the_run_and_is_reported(store):
    script = [
        LLMResponse(stop_reason="refusal", refusal="declined (category=cyber)", model="mock")
    ]
    result = ECGAgent(store=store, backend=ScriptedBackend(script=script)).run()
    assert result.refused and "cyber" in result.refused
    assert not result.ok


def test_backend_exceptions_land_in_the_result(store):
    class Exploding(ScriptedBackend):
        def complete(self, **kwargs):
            raise RuntimeError("connection reset")

    result = ECGAgent(store=store, backend=Exploding()).run()
    assert not result.ok and "connection reset" in result.error


def test_global_token_budget_stops_before_another_model_request(store):
    first = text_response("orientation complete")
    first.usage = {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
    }
    backend = ScriptedBackend(
        script=[first, text_response(json.dumps(_verdict()))]
    )
    result = ECGAgent(
        store=store,
        backend=backend,
        phases=(
            PhaseSpec("orient", prompts.ORIENT_INSTRUCTION, tool_budget=0),
            PhaseSpec(
                "synthesize",
                prompts.SYNTHESIZE_INSTRUCTION,
                tool_budget=0,
                structured=True,
            ),
        ),
        max_total_tokens=10,
    ).run()

    assert not result.ok
    assert "cumulative token budget exhausted" in result.error
    assert len(backend.calls) == 1
    assert result.audit["runtime_controls"]["total_tokens"] == 10


def test_checkpoint_callback_reports_phase_and_final_runtime(store):
    checkpoints: list[dict] = []
    result = ECGAgent(
        store=store,
        backend=ScriptedBackend(script=_happy_script()),
        on_checkpoint=checkpoints.append,
    ).run()

    assert result.verified
    assert any(
        row["status"].startswith("phase_completed:") for row in checkpoints
    )
    assert checkpoints[-1]["status"] == "finished"
    assert checkpoints[-1]["runtime"]["model_turns"] == (
        result.audit["runtime_controls"]["model_turns"]
    )


def test_briefing_is_the_first_user_message(store):
    captured: list[list[dict]] = []

    def script(messages):
        captured.append(messages)
        return text_response(json.dumps(_verdict()))

    ECGAgent(
        store=store,
        backend=ScriptedBackend(script=script),
        phases=(PhaseSpec("synthesize", prompts.SYNTHESIZE_INSTRUCTION, 0, structured=True),),
    ).run()
    first = captured[0][0]
    assert first["role"] == "user"
    assert "[intervals]" in first["content"][0]["text"]


def test_result_serializes_for_batch_output(store):
    result = ECGAgent(store=store, backend=ScriptedBackend(script=_happy_script())).run()
    payload = json.loads(json.dumps(result.to_dict(), default=str))
    assert payload["verified"] is True
    assert payload["audit"]["tools"]["n_calls"] == 1
    assert len(payload["phases"]) == len(DEFAULT_PHASES)
    assert payload["audit"]["input_fingerprint"]
    assert payload["audit"]["prompt_fingerprint"]
    seeded = payload["audit"]["tools"]["seeded_provenance"]["chart_briefing"]
    assert seeded["count"] > 0
    assert seeded["evidence_set_id"].startswith("sha256:")
    briefing = build_chart_briefing(store)
    assert seeded["count"] == len(set(briefing.citations))

    tools = payload["audit"]["tools"]
    call = tools["calls"][0]
    assert "citations" not in call
    assert "visible_citations" not in call
    assert call["citation_manifest"]["visible_count"] == 1

    selected = tools["effective_evidence"]["items"]
    assert len(selected) == 1
    assert selected[0]["pointer"] == "/global_features/pr_ms"
    assert selected[0]["value"] == 214.0
    assert selected[0]["unit"] == "ms"
    assert selected[0]["uses"][0]["role"] == "supporting"


def test_single_direct_evidence_value_is_materialized_from_store(store):
    result = ECGAgent(
        store=store,
        backend=ScriptedBackend(script=_happy_script(_verdict(pr_value=168.0))),
    ).run()

    item = result.verdict["diagnoses"][0]["evidence"][0]
    assert item["value"] == 214.0
    assert item["unit"] == "ms"
    assert result.verified
    assert any(
        row["kind"] == "deterministic_evidence_value_materialization"
        for row in result.audit["verdict_normalizations"]
    )


def test_contract_rejects_vacuous_added_diagnosis(store):
    verdict = {
        "summary": "Unsupported diagnosis.",
        "diagnoses": [
            {
                "code": "invented_diagnosis",
                "statement": "Invented diagnosis",
                "status": "added",
                "confidence": "HIGH",
                "rule_ids": [],
                "evidence": [],
                "counterevidence": [],
                "adjudication": "No evidence.",
            }
        ],
        "abstentions": [],
        "human_review": {"required": False, "reasons": []},
    }

    problems = prompts.validate_verdict(
        verdict,
        evidence_document=store.document,
    )
    assert any("no supporting `evidence`" in problem for problem in problems)
    assert any("human_review.required" in problem for problem in problems)
    assert any("not a registered statement_code" in problem for problem in problems)


def test_contract_requires_counterevidence_for_withdrawal(store):
    verdict = _verdict()
    diagnosis = verdict["diagnoses"][0]
    diagnosis["status"] = "withdrawn"
    diagnosis["counterevidence"] = []
    diagnosis["adjudication"] = "Withdrawn."

    problems = prompts.validate_verdict(
        verdict,
        evidence_document=store.document,
    )
    assert any("no `counterevidence`" in problem for problem in problems)


def test_contract_requires_every_baseline_diagnosis(store):
    verdict = _verdict()
    verdict["diagnoses"] = []

    problems = prompts.validate_verdict(
        verdict,
        evidence_document=store.document,
    )

    assert any("baseline diagnosis `first_degree_av_delay` is missing" in p for p in problems)


def test_contract_keeps_quantities_in_citable_evidence_only(store):
    verdict = _verdict()
    verdict["summary"] = "PR remains 214 ms."
    verdict["diagnoses"][0]["adjudication"] = "Confirmed at 214 ms."

    problems = prompts.validate_verdict(
        verdict,
        evidence_document=store.document,
    )

    assert any("`summary` contains patient-specific" in p for p in problems)
    assert any("diagnoses[0].adjudication contains patient-specific" in p for p in problems)


def test_contract_keeps_evidence_pointers_out_of_uncitable_fields(store):
    verdict = _verdict()
    verdict["abstentions"] = [
        {
            "topic": "PR association",
            "reason": "See ev:/global_features/pr_ms",
            "what_would_resolve_it": "Manual review",
        }
    ]

    problems = prompts.validate_verdict(
        verdict,
        evidence_document=store.document,
    )

    assert any("abstentions[0].reason contains an evidence pointer" in p for p in problems)


def test_contract_forbids_adding_diagnosis_when_rule_was_indeterminate(store):
    payload = _payload()
    row = payload["clinical_interpretation"]["domains"]["rhythm"][0]
    row["evidence"] = {"evaluates_code": "atrial_flutter_pattern"}
    verdict = _verdict()
    diagnosis = verdict["diagnoses"][0]
    diagnosis.update(
        {
            "code": "atrial_flutter_pattern",
            "statement": "Atrial flutter pattern",
            "status": "added",
            "confidence": "HIGH",
            "rule_ids": ["CLIN-RHYTHM-AFL-01"],
            "adjudication": "Resolved from per-beat evidence.",
        }
    )
    verdict["human_review"]["required"] = True
    custom_store = EvidenceStore.from_dict(payload, record_id="INDETERMINATE")

    problems = prompts.validate_verdict(
        verdict,
        evidence_document=custom_store.document,
    )

    assert any(
        "may only restore a code whose rule status was suppressed or borderline"
        in problem
        for problem in problems
    )


# ---------------------------------------------------------------------------
# Anthropic backend request shape
# ---------------------------------------------------------------------------
class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, message):
        self._message = message
        self.kwargs: dict = {}

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(self._message)


class _FakeClient:
    def __init__(self, message):
        self.beta = SimpleNamespace(messages=_FakeMessages(message))


def _message(content, stop_reason="end_turn", **extra):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model="claude-opus-5",
        usage=SimpleNamespace(
            model_dump=lambda: {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 0,
            }
        ),
        **extra,
    )


@pytest.fixture
def backend_and_client():
    message = _message([SimpleNamespace(type="text", text="hello")])
    client = _FakeClient(message)
    return AnthropicBackend(client=client), client


def test_request_defaults_to_opus_5_with_adaptive_thinking(backend_and_client):
    backend, client = backend_and_client
    backend.complete(system="S", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=None)
    kwargs = client.beta.messages.kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "high"


def test_request_omits_sampling_parameters(backend_and_client):
    """temperature / top_p / top_k are rejected on this model family."""
    backend, client = backend_and_client
    backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    kwargs = client.beta.messages.kwargs
    for banned in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert banned not in kwargs


def test_cache_breakpoint_sits_on_the_system_block(backend_and_client):
    """Render order is tools -> system -> messages, so this caches both."""
    backend, client = backend_and_client
    backend.complete(system="S", messages=[], tools=[{"name": "t"}], max_tokens=None)
    system = client.beta.messages.kwargs["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    backend.enable_caching = False
    backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    assert "cache_control" not in client.beta.messages.kwargs["system"][0]


def test_refusal_fallback_is_enabled_by_default(backend_and_client):
    backend, client = backend_and_client
    backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    kwargs = client.beta.messages.kwargs
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["extra_body"] == {"fallbacks": "default"}

    backend.enable_fallbacks = False
    backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    assert "betas" not in client.beta.messages.kwargs


def test_structured_output_goes_inside_output_config(backend_and_client):
    backend, client = backend_and_client
    backend.complete(
        system="S", messages=[], tools=[], max_tokens=None, response_schema=prompts.OUTPUT_SCHEMA
    )
    output_config = client.beta.messages.kwargs["output_config"]
    assert output_config["format"]["type"] == "json_schema"
    assert output_config["format"]["schema"] is prompts.OUTPUT_SCHEMA
    assert "output_format" not in client.beta.messages.kwargs   # deprecated spelling


def test_anthropic_required_tool_turn_hides_terminal_schema(backend_and_client):
    backend, client = backend_and_client
    backend.complete(
        system="S",
        messages=[],
        tools=[{"name": "t"}],
        max_tokens=None,
        response_schema=prompts.OUTPUT_SCHEMA,
        require_tool_call=True,
    )
    kwargs = client.beta.messages.kwargs
    assert kwargs["tool_choice"] == {"type": "any"}
    assert "format" not in kwargs["output_config"]


def test_tools_are_omitted_rather_than_sent_empty(backend_and_client):
    backend, client = backend_and_client
    backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    assert "tools" not in client.beta.messages.kwargs


def test_tool_use_response_is_parsed(backend_and_client):
    backend, client = backend_and_client
    client.beta.messages._message = _message(
        [
            SimpleNamespace(type="text", text="checking"),
            SimpleNamespace(type="tool_use", id="toolu_1", name="get_measurement",
                            input={"pointer": "PR"}),
        ],
        stop_reason="tool_use",
    )
    response = backend.complete(system="S", messages=[], tools=[{"name": "t"}], max_tokens=None)
    assert response.wants_tools
    assert response.tool_calls[0].name == "get_measurement"
    assert response.tool_calls[0].arguments == {"pointer": "PR"}


def test_assistant_turn_echoes_provider_blocks_verbatim(backend_and_client):
    """Thinking blocks must round-trip unmodified, so never rebuild from text."""
    backend, client = backend_and_client
    blocks = [SimpleNamespace(type="text", text="hi")]
    client.beta.messages._message = _message(blocks)
    response = backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    assert backend.assistant_turn(response)["content"] is blocks


def test_refusal_is_surfaced_with_its_category(backend_and_client):
    backend, client = backend_and_client
    client.beta.messages._message = _message(
        [], stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation=None),
    )
    response = backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    assert response.refused and "cyber" in response.refusal


def test_cache_report_reads_usage(backend_and_client):
    backend, client = backend_and_client
    backend.complete(system="S", messages=[], tools=[], max_tokens=None)
    assert "900 tokens read" in backend.cache_report()


class _FakeDeepSeekCompletions:
    def __init__(self, completion):
        self.completion = completion
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.completion


def _deepseek_completion(*, with_tool: bool = True):
    tool_calls = []
    finish_reason = "stop"
    if with_tool:
        finish_reason = "tool_calls"
        tool_calls = [
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="get_measurement",
                    arguments='{"pointer":"PR"}',
                ),
            )
        ]
    message = SimpleNamespace(
        content="checking" if with_tool else "done",
        reasoning_content="provider reasoning",
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        model="deepseek-v4-pro",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=message,
            )
        ],
        usage=None,
    )


def test_deepseek_v4_tool_turn_round_trips_reasoning_content():
    completions = _FakeDeepSeekCompletions(_deepseek_completion())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    backend = DeepSeekBackend(client=client)

    response = backend.complete(
        system="S",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "get_measurement", "input_schema": {"type": "object"}}],
        max_tokens=None,
    )
    assistant = backend.assistant_turn(response)

    assert assistant["reasoning_content"] == "provider reasoning"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert completions.kwargs["reasoning_effort"] == "high"
    assert completions.kwargs["extra_body"]["thinking"] == {"type": "enabled"}
    assert completions.kwargs["extra_body"]["user_id"] == "ecgagent"


def test_deepseek_required_tool_turn_hides_json_terminal_branch():
    completions = _FakeDeepSeekCompletions(_deepseek_completion())
    backend = DeepSeekBackend(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    backend.complete(
        system="S",
        messages=[],
        tools=[{"name": "get_measurement", "input_schema": {"type": "object"}}],
        max_tokens=None,
        response_schema=prompts.OUTPUT_SCHEMA,
        require_tool_call=True,
    )

    assert completions.kwargs["tool_choice"] == "required"
    assert completions.kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in completions.kwargs
    assert "response_format" not in completions.kwargs
    assert all(
        "JSON schema" not in str(message.get("content") or "")
        for message in completions.kwargs["messages"]
    )


def test_deepseek_compact_prefetch_uses_atomic_q_citations():
    backend = DeepSeekBackend(
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=_FakeDeepSeekCompletions(
                    _deepseek_completion(with_tool=False)
                )
            )
        )
    )
    pointer = "/representative_leads/V1/params/qrs_ms"
    outcome = ToolOutcome(
        call_id="prefetch-adjudicate-1",
        name="get_morphology_map",
        arguments={"profile": "qrs", "leads": ["V1"]},
        text="wide human table that must not reach the model",
        citations=(pointer,),
        model_text=(
            '{"contract":"ecgagent.model-evidence.v2","evidence":['
            '{"citation":"ev:/representative_leads/V1/params/qrs_ms",'
            '"value":138,"unit":"ms","reliability":"reliable",'
            '"caveats":[]}]}'
        ),
        model_citations=(pointer,),
    )

    messages = backend.tool_result_turn([outcome])

    assert backend.capabilities.orchestrated_prefetch is True
    assert backend.capabilities.citation_aliases is True
    assert messages[0]["role"] == "user"
    assert "wide human table" not in messages[0]["content"]
    assert '"citation":"Q1"' in messages[0]["content"]
    assert '"label":"V1.qrs_ms"' in messages[0]["content"]
    assert "ev:/representative_leads/V1/params/qrs_ms" not in messages[0]["content"]
    assert backend.citation_aliases() == {"Q1": pointer}
    assert backend.model_visible_citations([outcome], messages) == {
        outcome.call_id: (pointer,)
    }
    assert backend.normalize_verdict({"citations": ["Q1"]}) == {
        "citations": [f"ev:{pointer}"]
    }


def test_deepseek_disables_thinking_for_compact_plan_only():
    completions = _FakeDeepSeekCompletions(
        _deepseek_completion(with_tool=False)
    )
    backend = DeepSeekBackend(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )

    backend.complete(
        system="S",
        messages=[{"role": "user", "content": "COMPACT PLAN — route evidence"}],
        tools=[],
        max_tokens=1200,
        response_schema=prompts.OUTPUT_SCHEMA,
    )

    assert completions.kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in completions.kwargs


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_output_schema_meets_structured_output_constraints():
    """Every object needs additionalProperties:false and a required list."""

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(prompts.OUTPUT_SCHEMA)
    json.dumps(prompts.OUTPUT_SCHEMA)


def test_system_prompt_is_frozen_across_phases(store):
    """Editing it per phase would invalidate the tools+system cache every time."""

    def script(messages):
        return text_response(json.dumps(_verdict()))

    backend = ScriptedBackend(script=script)
    original = backend.complete
    seen: list[str] = []

    def spy(**kwargs):
        seen.append(kwargs["system"])
        return original(**kwargs)

    backend.complete = spy  # type: ignore[method-assign]
    ECGAgent(store=store, backend=backend).run()
    assert len(seen) >= len(DEFAULT_PHASES)
    assert len(set(seen)) == 1, "system prompt changed between phases"


def test_citing_a_pointer_no_tool_returned_fails_verification(store):
    """A verdict cannot cite a value absent from both briefing and tool results."""
    verdict = _verdict(pr_value=96.0, claim="Lead I QRS duration")
    verdict["diagnoses"][0]["evidence"][0]["citations"] = [
        "ev:/representative_leads/I/params/qrs_ms"
    ]

    def script(messages):
        return text_response(json.dumps(verdict))

    result = ECGAgent(store=store, backend=ScriptedBackend(script=script)).run()
    assert not result.verified
    assert result.verification.counts()["pointer_not_from_tool"] >= 1


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'here it is:\n{"a": 1}\ndone'],
)
def test_verdict_json_survives_common_wrappers(text):
    assert parse_json_text(text) == {"a": 1}


def test_verdict_json_accepts_literal_newline_inside_string():
    text = '{"claim":"first line\nsecond line","citations":["Q1"]}'
    assert parse_json_text(text) == {
        "claim": "first line\nsecond line",
        "citations": ["Q1"],
    }


def test_unparseable_json_returns_none():
    assert parse_json_text("no json here") is None
    assert parse_json_text("[1,2,3]") is None    # must be an object
