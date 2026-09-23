from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ecgagent.agent import diagnostic_prompts
from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.agent.loop import PhaseSpec
from ecgagent.backends import build_backend
from ecgagent.backends.base import ToolOutcome
from ecgagent.backends.medgemma_local import (
    DEFAULT_MODEL_MAX_LEN,
    MedGemmaLocalBackend,
)
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_diagnostic import _diagnostic_verdict
from tests.test_ecgagent_tools import _payload
from tests.diagnostic_script_helpers import (
    HEART_RATE,
    T_AMP,
    legacy_phase_patches,
)


class _FakeLLM:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages, *, sampling_params, use_tqdm):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        text = self.texts.pop(0)
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    SimpleNamespace(
                        text=text,
                        token_ids=[4, 5],
                        finish_reason="stop",
                    )
                ],
            )
        ]


class _BatchFakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, *, sampling_params, use_tqdm):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    SimpleNamespace(
                        text=json.dumps({"summary": f"batch-{index}"}),
                        token_ids=[4, 5],
                        finish_reason="stop",
                    )
                ],
            )
            for index in range(len(messages))
        ]


def _backend(fake: _FakeLLM) -> MedGemmaLocalBackend:
    return MedGemmaLocalBackend(
        model="/not/loaded/in/unit/tests",
        llm=fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        structured_outputs_factory=lambda **kwargs: kwargs,
        enforce_phase_coverage=False,
        hard_phase_guards=False,
    )


def test_local_backend_uses_schema_constrained_tool_envelope():
    fake = _FakeLLM(
        [
            json.dumps(
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "get_measurement",
                            "arguments": {"pointer": "QTc"},
                        }
                    ],
                    "text": "",
                }
            )
        ]
    )
    backend = _backend(fake)

    response = backend.complete(
        system="system",
        messages=[
            {"role": "user", "content": "briefing"},
            {"role": "user", "content": "survey"},
        ],
        tools=[
            {
                "name": "get_measurement",
                "description": "read one measurement",
                "input_schema": {
                    "type": "object",
                    "properties": {"pointer": {"type": "string"}},
                    "required": ["pointer"],
                },
            }
        ],
        max_tokens=512,
    )

    assert response.wants_tools
    assert response.tool_calls[0].name == "get_measurement"
    assert response.tool_calls[0].arguments == {"pointer": "QTc"}
    assert response.tool_calls[0].id == "medgemma_tool_1"
    envelope_schema = fake.calls[0]["sampling_params"]["structured_outputs"]["json"]
    assert [branch["properties"]["action"]["const"] for branch in envelope_schema["oneOf"]] == [
        "tool_calls",
        "finish",
    ]
    tool_branch = envelope_schema["oneOf"][0]["properties"]["tool_calls"][
        "items"
    ]["oneOf"][0]
    assert tool_branch["properties"]["name"]["const"] == "get_measurement"
    assert tool_branch["properties"]["arguments"]["required"] == ["pointer"]
    finish_memory = envelope_schema["oneOf"][1]["properties"]["text"]
    assert finish_memory["properties"]["findings"]["maxItems"] == 24
    assert set(finish_memory["properties"]["domains"]["required"]) == {
        "quality",
        "rhythm_rate",
        "p_av",
        "intervals",
        "axis",
        "conduction_preexcitation",
        "ectopy_pauses",
        "voltage_chamber_r_progression",
        "q_st_t_u",
        "pacing_high_risk",
    }
    # The bundled Gemma template requires strict alternation; consecutive
    # briefing/phase user messages are merged by the backend.
    assert [row["role"] for row in fake.calls[0]["messages"]] == [
        "system",
        "user",
    ]
    assert "briefing" in fake.calls[0]["messages"][1]["content"]
    assert "survey" in fake.calls[0]["messages"][1]["content"]


def test_local_backend_hides_finish_schema_until_required_coverage_is_met():
    fake = _FakeLLM(
        [
            json.dumps(
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "get_measurement",
                            "arguments": {"pointer": "QTc"},
                        }
                    ],
                    "text": "",
                }
            )
        ]
    )
    backend = _backend(fake)

    response = backend.complete(
        system="system",
        messages=[{"role": "user", "content": "survey"}],
        tools=[
            {
                "name": "get_measurement",
                "description": "read one measurement",
                "input_schema": {
                    "type": "object",
                    "properties": {"pointer": {"type": "string"}},
                    "required": ["pointer"],
                },
            }
        ],
        max_tokens=512,
        response_schema={
            "type": "object",
            "properties": {"phase_summary": {"type": "string"}},
            "required": ["phase_summary"],
        },
        require_tool_call=True,
    )

    assert response.wants_tools
    schema = fake.calls[0]["sampling_params"]["structured_outputs"]["json"]
    assert schema["properties"]["action"]["const"] == "tool_calls"
    assert "oneOf" not in schema
    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "cannot finish the phase" in system_prompt
    assert "FINAL JSON SCHEMA" not in system_prompt


def test_local_unchanged_intermediate_state_does_not_spend_repair_turn():
    unchanged = json.dumps(
        {"hypotheses": [], "state_changes": []},
        ensure_ascii=False,
    )
    fake = _FakeLLM([unchanged])
    backend = MedGemmaLocalBackend(
        model="/not/loaded/in/unit/tests",
        llm=fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        structured_outputs_factory=lambda **kwargs: kwargs,
        enforce_phase_coverage=False,
        hard_phase_guards=True,
    )
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="HARD-GUARD"),
        backend=backend,
        max_revisions=0,
        knowledge_guidance=False,
    )
    agent._last_phase_state_text = unchanged
    record = agent._run_phase(
        PhaseSpec(
            "investigate",
            "investigate",
            tool_budget=0,
            response_schema={"type": "object"},
        ),
        [],
        [],
    )

    assert record.phase_guard_passed is True
    assert record.phase_guard_attempts == 0
    assert record.stop_reason == "end_turn"
    assert len(fake.calls) == 1


def test_local_backend_applies_final_json_schema():
    verdict = _diagnostic_verdict()
    fake = _FakeLLM([json.dumps(verdict, ensure_ascii=False)])
    backend = _backend(fake)
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    response = backend.complete(
        system="system",
        messages=[{"role": "user", "content": "synthesize"}],
        tools=[],
        max_tokens=1024,
        response_schema=schema,
    )

    assert json.loads(response.text)["summary"] == verdict["summary"]
    assert (
        fake.calls[0]["sampling_params"]["structured_outputs"]["json"]
        == schema
    )


def test_diagnostic_output_schema_avoids_vllm_unsupported_unique_items():
    def paths_with_key(value, key, path="$"):
        found = []
        if isinstance(value, dict):
            if key in value:
                found.append(path)
            for child_key, child in value.items():
                found.extend(paths_with_key(child, key, f"{path}.{child_key}"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(paths_with_key(child, key, f"{path}[{index}]"))
        return found

    assert paths_with_key(diagnostic_prompts.OUTPUT_SCHEMA, "uniqueItems") == []


def test_local_backend_preserves_verdict_when_revision_tools_are_available():
    verdict = _diagnostic_verdict()
    fake = _FakeLLM([json.dumps(verdict, ensure_ascii=False)])
    backend = _backend(fake)
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "diagnoses": {"type": "array"},
        },
        "required": ["summary", "diagnoses"],
        "additionalProperties": True,
    }
    tools = [
        {
            "name": "get_measurement",
            "description": "read one measurement",
            "input_schema": {
                "type": "object",
                "properties": {"pointer": {"type": "string"}},
                "required": ["pointer"],
            },
        }
    ]

    response = backend.complete(
        system="system",
        messages=[{"role": "user", "content": "revise the verdict"}],
        tools=tools,
        max_tokens=1024,
        response_schema=schema,
    )

    assert json.loads(response.text)["summary"] == verdict["summary"]
    assert not response.wants_tools
    guided = fake.calls[0]["sampling_params"]["structured_outputs"]["json"]
    assert guided["oneOf"][0]["properties"]["action"]["const"] == "tool_calls"
    assert guided["oneOf"][1] == schema
    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "return the final diagnosis object directly" in system_prompt


def test_phase_memory_keeps_only_current_state_and_retains_atrial_tools():
    backend = _backend(_FakeLLM([]))
    transient = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="p1",
                name="get_p_assessment_table",
                text="accepted=true; ta_ambiguous=true",
                arguments={"limit": 8},
            ),
            ToolOutcome(
                call_id="a1",
                name="get_atrial_event_table",
                text="candidate atrial events",
                arguments={"limit": 8},
            ),
        ]
    )
    backend.phase_memory_turn(
        "survey",
        json.dumps({"phase_summary": "survey-state"}),
        transient,
    )
    turns = backend.phase_memory_turn(
        "investigate",
        json.dumps({"phase_summary": "investigate-state"}),
        [],
    )

    payload = json.loads(turns[0]["content"].split("\n", 1)[1])
    assert payload["phase_history"] == ["survey", "investigate"]
    assert payload["current_phase"] == "investigate"
    assert payload["current_phase_state"] == {
        "phase_summary": "investigate-state"
    }
    retained = {row["tool"] for row in payload["retained_key_tool_evidence"]}
    assert retained == {"get_p_assessment_table", "get_atrial_event_table"}
    assert backend.audit_config()["phase_memory"]["completed_phases"] == [
        "survey",
        "investigate",
    ]


def test_local_revision_context_keeps_only_latest_state_and_candidate():
    backend = _backend(_FakeLLM([]))
    pointer = "/global_features/heart_rate_bpm"
    backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="rate",
                name="get_global_table",
                text=f"rate E [ev:{pointer}]",
                citations=(pointer,),
            )
        ]
    )
    memory = {
        "role": "user",
        "content": "CUMULATIVE EVIDENCE MEMORY.\n{}",
    }
    compacted = backend.compact_revision_context(
        [
            {"role": "user", "content": "briefing"},
            memory,
            {"role": "assistant", "content": "old synthesis " + "x" * 10000},
            {"role": "assistant", "content": "old revision " + "y" * 10000},
        ],
        {
            "summary": "current",
            "diagnoses": [{"citations": [f"ev:{pointer}"]}],
        },
    )

    rendered = json.dumps(compacted, ensure_ascii=False)
    assert len(compacted) == 3
    assert "old synthesis" not in rendered
    assert "old revision" not in rendered
    assert "CURRENT DIAGNOSIS CANDIDATE" in rendered
    assert "E1" in rendered
    assert pointer not in rendered


def test_local_backend_structured_revision_can_still_request_a_tool():
    fake = _FakeLLM(
        [
            json.dumps(
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "get_measurement",
                            "arguments": {"pointer": "QTc"},
                        }
                    ],
                    "text": "",
                }
            )
        ]
    )
    backend = _backend(fake)

    response = backend.complete(
        system="system",
        messages=[{"role": "user", "content": "revise the verdict"}],
        tools=[
            {
                "name": "get_measurement",
                "input_schema": {
                    "type": "object",
                    "properties": {"pointer": {"type": "string"}},
                    "required": ["pointer"],
                },
            }
        ],
        response_schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    )

    assert response.wants_tools
    assert response.tool_calls[0].name == "get_measurement"


def test_local_backend_compacts_and_restores_citation_aliases():
    backend = _backend(_FakeLLM([]))
    pointer = "/rhythm_inputs/native_beat_profiles/rows/0/qrs_ms"

    turns = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="call-1",
                name="get_native_beat_profile",
                text=f"QRS 96 ms [ev:{pointer}]",
                citations=(pointer,),
            )
        ]
    )

    assert "E1" in turns[0]["content"]
    assert pointer not in turns[0]["content"]
    normalized = backend.normalize_verdict(
        {
            "diagnoses": [
                {
                    "evidence": [
                        {
                            "citations": ["E1"],
                        }
                    ]
                }
            ]
        }
    )
    assert normalized["diagnoses"][0]["evidence"][0]["citations"] == [
        f"ev:{pointer}"
    ]
    audit = backend.audit_config()
    assert audit["citation_aliases"] == {"E1": pointer}
    assert audit["tool_context_compression"]["saved_chars"] > 0


def test_local_phase_memory_merges_later_domain_deltas():
    backend = _backend(_FakeLLM([]))
    survey = {
        "phase_summary": "survey",
        "domains": {
            "quality": {
                "status": "assessed",
                "observation": "quality adequate",
                "citations": ["E1"],
                "limitations": [],
            },
            "rhythm_rate": {
                "status": "assessed",
                "observation": "regular rhythm",
                "citations": ["E2"],
                "limitations": [],
            },
        },
        "hypotheses": [],
        "state_changes": [],
        "detector_conflicts": [],
        "unresolved": [],
    }
    backend.phase_memory_turn("survey", json.dumps(survey))
    hypothesis_delta = {
        "phase_summary": "hypothesis",
        "hypotheses": [],
        "state_changes": [],
        "detector_conflicts": [],
        "unresolved": [],
        "targeted_checks": [],
    }

    turns = backend.phase_memory_turn(
        "hypothesize",
        json.dumps(hypothesis_delta),
    )
    payload = json.loads(turns[0]["content"].rsplit("\n", 1)[-1])
    domains = payload["current_phase_state"]["domains"]

    assert domains["quality"]["observation"] == "quality adequate"
    assert domains["rhythm_rate"]["status"] == "assessed"


def test_local_backend_caps_large_transient_result_and_builds_inflight_ledger():
    backend = _backend(_FakeLLM([]))
    rows = ["native repeated-beat profile", "| beat | lead | qrs |", "|---|---|---|"]
    rows.extend(
        f"| {index} | V{(index % 6) + 1} | {80 + index} E{index} |"
        for index in range(1200)
    )
    turns = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="huge-1",
                name="get_native_beat_profile",
                arguments={"profile": "qrs_infarct", "max_beats": 5},
                text="\n".join(rows),
            )
        ]
    )

    assert len(turns[0]["content"]) < 14000
    compacted = backend.compact_phase_context(
        "challenge",
        [{"role": "user", "content": "challenge instruction"}, *turns],
    )
    rendered = json.dumps(compacted, ensure_ascii=False)
    assert "IN-PHASE EVIDENCE LEDGER" in rendered
    assert "get_native_beat_profile" in rendered
    assert len(rendered) < 16000
    compression = backend.audit_config()["tool_context_compression"]
    assert compression["truncated_outcomes"] == 1
    assert compression["omitted_chars"] > 0


def test_local_backend_preflight_blocks_before_vllm_context_overflow():
    fake = _FakeLLM([])
    backend = MedGemmaLocalBackend(
        model="/not/loaded/in/unit/tests",
        model_max_len=4096,
        llm=fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        structured_outputs_factory=lambda **kwargs: kwargs,
    )

    with pytest.raises(RuntimeError, match="preflight context budget exceeded"):
        backend.complete(
            system="system",
            messages=[{"role": "user", "content": "x" * 30000}],
            tools=[],
            max_tokens=1024,
        )

    assert fake.calls == []
    check = backend.audit_config()["context_preflight"]["checks"][-1]
    assert check["status"] == "overflow"


def test_local_backend_defaults_to_128k_context():
    assert DEFAULT_MODEL_MAX_LEN == 131072


def test_local_agent_fuses_duplicate_only_tool_batch():
    fake = _FakeLLM(
        [
            json.dumps(
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "get_global_table",
                            "arguments": {"fields": ["heart_rate_bpm"]},
                        }
                    ],
                    "text": "",
                }
            ),
            json.dumps(
                {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "get_global_table",
                            "arguments": {"fields": ["heart_rate_bpm"]},
                        }
                    ],
                    "text": "",
                }
            ),
            "duplicate loop resolved",
        ]
    )
    backend = _backend(fake)
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="DUP-FUSE"),
        backend=backend,
        max_revisions=0,
        knowledge_guidance=False,
    )
    messages = [backend.user_turn("briefing")]
    spec = PhaseSpec(
        "investigate",
        "targeted investigation",
        tool_budget=4,
        allowed_tools=("get_global_table",),
        minimum_tool_calls=4,
    )

    record = agent._run_phase(spec, messages, agent.registry.anthropic_tools())

    assert record.tool_calls == 1
    assert record.duplicate_batches == 1
    assert record.tool_loop_fused
    assert record.turns == 3
    assert record.text == "duplicate loop resolved"
    assert agent.registry.audit()["duplicate_calls_suppressed"] == 1
    # The final turn has no tool schema, forcing a phase conclusion.
    assert "structured_outputs" not in fake.calls[-1]["sampling_params"]


def test_local_backend_microbatches_independent_sessions():
    fake = _BatchFakeLLM()
    root = MedGemmaLocalBackend(
        model="/not/loaded/in/unit/tests",
        llm=fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        structured_outputs_factory=lambda **kwargs: kwargs,
        max_batch_size=2,
        batch_wait_ms=200,
    )
    sessions = [root.new_session(), root.new_session()]
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    def generate(index):
        return sessions[index].complete(
            system="system",
            messages=[{"role": "user", "content": f"record-{index}"}],
            tools=[],
            max_tokens=64,
            response_schema=schema,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(generate, range(2)))
    finally:
        root.close()

    assert len(fake.calls) == 1
    assert len(fake.calls[0]["messages"]) == 2
    assert sorted(
        json.loads(response.text)["summary"] for response in responses
    ) == [
        "batch-0",
        "batch-1",
    ]
    assert sessions[0].turns is not sessions[1].turns
    scheduler = sessions[0].audit_config()["scheduler"]
    assert scheduler["batches"] == 1
    assert scheduler["requests"] == 2
    assert scheduler["max_observed_batch_size"] == 2


def test_build_backend_accepts_medgemma_alias_without_loading_when_injected():
    fake = _FakeLLM([])

    backend = build_backend(
        "medgemma",
        model="/injected",
        llm=fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        structured_outputs_factory=lambda **kwargs: kwargs,
    )

    assert isinstance(backend, MedGemmaLocalBackend)
    assert backend.name == "medgemma-local"


def test_local_backend_runs_the_existing_ecg_diagnostic_agent_protocol():
    verdict = _diagnostic_verdict()
    survey, hypothesize, investigate, challenge = legacy_phase_patches()

    def call(name, arguments):
        return json.dumps(
            {
                "action": "tool_calls",
                "tool_calls": [{"name": name, "arguments": arguments}],
                "text": "",
            },
            ensure_ascii=False,
        )

    fake = _FakeLLM(
        [
            call("get_diagnostic_overview", {}),
            json.dumps(survey, ensure_ascii=False),
            json.dumps(hypothesize, ensure_ascii=False),
            call("get_measurement", {"pointer": HEART_RATE}),
            call("get_interval_waveform_context", {"interval": "qt"}),
            json.dumps(investigate, ensure_ascii=False),
            call("get_measurement", {"pointer": T_AMP}),
            call(
                "get_beat_table",
                {"fields": ["rr_prev_ms", "rr_next_ms"]},
            ),
            json.dumps(challenge, ensure_ascii=False),
            json.dumps(verdict, ensure_ascii=False),
        ]
    )
    backend = _backend(fake)
    store = EvidenceStore.from_dict(_payload(), record_id="LOCAL")

    result = ECGDiagnosticAgent(
        workflow="legacy",
        store=store,
        backend=backend,
        max_revisions=0,
    ).run()

    assert result.ok and result.verified, result.summary()
    assert result.verdict["diagnoses"][0]["code"] == "sinus_rhythm"
    assert result.audit["model"]["backend"] == "medgemma-local"
    assert result.audit["model"]["usage"]["turns"] == 10
    assert result.audit["tools"]["n_calls"] == 5
    # The governed cards are present only for provisional planning. Transient
    # call envelopes and the cards are gone by targeted investigation, while
    # cumulative phase memory retains selected compressed patient evidence.
    hypothesize_prompt = json.dumps(
        fake.calls[2]["messages"],
        ensure_ascii=False,
    )
    assert "INTERIM MEDICAL KNOWLEDGE NAVIGATION" in hypothesize_prompt
    investigate_prompt = json.dumps(fake.calls[3]["messages"], ensure_ascii=False)
    assert "CUMULATIVE EVIDENCE MEMORY" in investigate_prompt
    assert "INTERIM MEDICAL KNOWLEDGE NAVIGATION" not in investigate_prompt
    assert "ECGFEAT TOOL RESULTS" not in investigate_prompt
    assert "retained_key_tool_evidence" in investigate_prompt
    assert "get_global_table" in investigate_prompt
    compactions = result.audit["context_compactions"]
    assert [
        row["phase"]
        for row in compactions
        if row.get("kind") == "phase_boundary"
    ] == [
        "survey",
        "hypothesize",
        "investigate",
        "challenge",
    ]
    assert any(
        row.get("kind") == "incremental_tool_batch"
        and row["saved_chars"] > 0
        for row in compactions
    )
    trace = result.trace_report
    assert "Result excerpt actually seen by the model" in trace
    assert "Complete raw result (audit only" in trace


def test_local_agent_requires_minimum_evidence_coverage_before_finishing():
    backend = _backend(_FakeLLM([]))
    backend.capabilities = replace(
        backend.capabilities,
        enforce_phase_coverage=True,
    )
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="COVERAGE"),
        backend=backend,
        max_revisions=0,
    )

    feedback = agent._phase_completion_feedback(agent.phases[0], 0)

    assert feedback is not None
    assert "get_diagnostic_overview x1" in feedback
    assert "coverage requirement, not a diagnostic rule" in feedback


def test_local_survey_prefetches_fixed_packet_and_uses_one_model_turn():
    domains = {
        key: {
            "status": "limited",
                "observation": "Available basic measurements were read",
            "citations": [],
                "limitations": ["Unavailable views remain explicit limitations"],
        }
        for key in diagnostic_prompts.SURVEY_STATE_SCHEMA["properties"][
            "domains"
        ]["properties"]
    }
    state = {
        "phase_summary": "Initial review of the basic measurement packet is complete",
        "domains": domains,
        "hypotheses": [],
        "state_changes": [],
        "detector_conflicts": [],
        "unresolved": [],
    }
    fake = _FakeLLM([json.dumps(state, ensure_ascii=False)])
    backend = _backend(fake)
    backend.capabilities = replace(
        backend.capabilities,
        enforce_phase_coverage=True,
        structured_output_level="grammar",
    )
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="PREFETCH"),
        backend=backend,
        max_revisions=0,
        knowledge_guidance=False,
    )
    messages = [backend.user_turn("briefing")]

    record = agent._run_phase(
        agent.phases[0],
        messages,
        agent.registry.anthropic_tools(),
    )

    assert record.turns == 1
    assert record.prefetched_tool_calls == 1
    assert record.tool_calls == record.prefetched_tool_calls
    assert len(fake.calls) == 1
    assert "BASELINE SURVEY PREFETCH COMPLETE" in json.dumps(
        fake.calls[0]["messages"], ensure_ascii=False
    )
    assert "tool_choice" not in fake.calls[0]["sampling_params"]
    assert any(
        row.get("kind") == "orchestrated_prefetch"
        for row in agent._context_compactions
    )


def test_survey_coverage_requires_only_the_compact_overview():
    payload = _payload()
    backend = _backend(_FakeLLM([]))
    backend.capabilities = replace(
        backend.capabilities,
        enforce_phase_coverage=True,
    )
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(payload, record_id="BOTH-LIMITED"),
        backend=backend,
        max_revisions=0,
    )
    survey = agent.phases[0]
    agent.registry.begin_phase("survey", survey.tool_budget)
    overview = agent.registry.call("get_diagnostic_overview", {})

    feedback = agent._phase_completion_feedback(survey, 0)

    assert overview.ok
    assert feedback is None
    assert survey.tool_budget == 1
