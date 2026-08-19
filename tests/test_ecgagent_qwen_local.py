from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ecgagent.agent import diagnostic_prompts
from ecgagent.backends import build_backend
from ecgagent.backends.base import ToolOutcome
from ecgagent.backends.medgemma_local import MedGemmaLocalBackend
from ecgagent.backends.qwen_local import QwenLocalBackend, _compact_schema


class _FakeCompletions:
    def __init__(self, completion):
        self.completion = completion
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.completion


def _completion(*, tool: bool = True, content: str | None = None):
    calls = []
    finish_reason = "stop"
    if tool:
        finish_reason = "tool_calls"
        calls = [
            SimpleNamespace(
                id="call_qwen_1",
                function=SimpleNamespace(
                    name="get_measurement",
                    arguments='{"pointer":"PR"}',
                ),
            )
        ]
    message = SimpleNamespace(
        content=("checking" if tool else '{"status":"ok"}')
        if content is None
        else content,
        reasoning_content="qwen reasoning",
        tool_calls=calls,
        refusal=None,
    )
    return SimpleNamespace(
        model="qwen3.8-27b",
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
            prompt_tokens_details=SimpleNamespace(cached_tokens=8),
        ),
    )


def _backend(completion=None):
    completions = _FakeCompletions(completion or _completion())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return QwenLocalBackend(client=client), completions


def test_qwen_uses_native_openai_tool_calls_and_preserves_tool_reasoning():
    backend, completions = _backend()

    response = backend.complete(
        system="S",
        messages=[{"role": "user", "content": "inspect"}],
        tools=[
            {
                "name": "get_measurement",
                "description": "read one measurement",
                "input_schema": {
                    "type": "object",
                    "properties": {"pointer": {"type": "string"}},
                },
            }
        ],
        max_tokens=512,
        require_tool_call=True,
    )

    assert response.wants_tools
    assert response.tool_calls[0].name == "get_measurement"
    assert response.tool_calls[0].arguments == {"pointer": "PR"}
    assert completions.kwargs["tool_choice"] == "required"
    assert completions.kwargs["tools"][0]["type"] == "function"
    assert completions.kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }

    assistant = backend.assistant_turn(response)
    assert assistant["tool_calls"][0]["id"] == "call_qwen_1"
    assert assistant["reasoning_content"] == "qwen reasoning"


def test_qwen_tool_results_use_standard_tool_call_id_messages():
    backend, _ = _backend()

    messages = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="call_qwen_1",
                name="get_measurement",
                text="PR=173 ms",
            )
        ]
    )

    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_qwen_1",
            "content": "PR=173 ms",
        }
    ]


def test_qwen_prefers_program_owned_atomic_evidence_view():
    backend, _ = _backend()

    messages = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="call_qwen_atomic",
                name="get_measurement",
                text="wide human table with stale copied value",
                citations=("/global_features/qrs_ms",),
                model_text=(
                    '{"contract":"ecgagent.model-evidence.v2","evidence":['
                    '{"citation":"ev:/global_features/qrs_ms","value":132,'
                    '"unit":"ms","reliability":"limited","caveats":[]}]}'
                ),
                model_citations=("/global_features/qrs_ms",),
            )
        ]
    )

    assert "stale copied value" not in messages[0]["content"]
    assert '"value":132' in messages[0]["content"]
    assert '"citation":"Q1"' in messages[0]["content"]
    assert '"label":"global.qrs_ms"' in messages[0]["content"]
    assert "ev:/global_features/qrs_ms" not in messages[0]["content"]
    assert backend.citation_aliases() == {"Q1": "/global_features/qrs_ms"}
    assert backend.model_visible_citations(
        [
            ToolOutcome(
                call_id="call_qwen_atomic",
                name="get_measurement",
                text="unused",
                model_text=messages[0]["content"],
                model_citations=("/global_features/qrs_ms",),
            )
        ],
        messages,
    ) == {"call_qwen_atomic": ("/global_features/qrs_ms",)}


def test_qwen_compacts_program_prefetch_packets_into_one_bounded_ledger():
    backend, _ = _backend()
    backend.max_retained_evidence_chars = 1_600
    outcomes = []
    for call_index, tool in enumerate(
        ("get_rhythm_profile", "get_morphology_groups", "get_beat_table"),
        start=1,
    ):
        pointers = tuple(
            f"/synthetic/{call_index}/field_{index}" for index in range(8)
        )
        payload = {
            "contract": "ecgagent.model-evidence.v2",
            "tool": tool,
            "arguments": {"view": call_index},
            "evidence": [
                {
                    "citation": f"ev:{pointer}",
                    "value": index,
                    "unit": None,
                    "reliability": "reliable",
                    "caveats": [],
                }
                for index, pointer in enumerate(pointers)
            ],
            "omitted_atom_count": 0,
        }
        outcomes.append(
            ToolOutcome(
                call_id=f"prefetch-adjudicate-{call_index}",
                name=tool,
                text="full audit result",
                model_text=json.dumps(payload),
                model_citations=pointers,
                arguments={"view": call_index},
            )
        )

    raw_messages = backend.tool_result_turn(outcomes)
    compacted = backend.compact_phase_context("adjudicate", raw_messages)

    assert all(message["role"] == "user" for message in raw_messages)
    assert len(compacted) == 1
    assert compacted[0]["content"].startswith("QWEN IN-PHASE EVIDENCE LEDGER.")
    assert "PROGRAM-PREFETCHED PATIENT EVIDENCE" not in compacted[0]["content"]
    assert len(compacted[0]["content"]) < 2_200
    visible = backend.model_visible_citations(outcomes, compacted)
    assert set(visible) == {outcome.call_id for outcome in outcomes}
    assert all(visible[outcome.call_id] for outcome in outcomes)


def test_qwen_alias_keeps_compact_lead_and_field_identity():
    backend, _ = _backend()
    messages = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="call_lead_atom",
                name="get_morphology_map",
                text="unused",
                model_text=(
                    '{"contract":"ecgagent.model-evidence.v2","evidence":['
                    '{"citation":"ev:/representative_leads/V2/params/t_amp_mv",'
                    '"value":0.087,"unit":"mV","reliability":"reliable",'
                    '"caveats":[]}]}'
                ),
                model_citations=(
                    "/representative_leads/V2/params/t_amp_mv",
                ),
            )
        ]
    )

    assert '"citation":"Q1"' in messages[0]["content"]
    assert '"label":"V2.t_amp_mv"' in messages[0]["content"]
    assert "/representative_leads/V2/params/t_amp_mv" not in messages[0]["content"]


def test_qwen_malformed_arguments_are_returned_to_the_tool_as_an_error():
    completion = _completion()
    completion.choices[0].message.tool_calls[0].function.arguments = "{bad-json"
    backend, _ = _backend(completion)

    response = backend.complete(
        system="S",
        messages=[],
        tools=[{"name": "get_measurement", "input_schema": {"type": "object"}}],
        require_tool_call=True,
    )

    arguments = response.tool_calls[0].arguments
    assert "__tool_argument_parse_error__" in arguments
    assert arguments["__raw_arguments__"] == "{bad-json"


def test_qwen_uses_strict_vllm_schema_only_after_tools_are_closed():
    schema = {
        "type": "object",
        "description": "large top-level explanation",
        "properties": {
            "status": {
                "type": "string",
                "description": "large field explanation",
            }
        },
        "required": ["status"],
        "additionalProperties": False,
    }
    backend, completions = _backend(_completion(tool=False))

    backend.complete(
        system="S",
        messages=[],
        tools=[],
        max_tokens=128,
        response_schema=schema,
    )

    assert completions.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "ecgagent_response",
            "schema": schema,
            "strict": True,
        },
    }
    assert len(completions.kwargs["messages"]) == 1

    backend.complete(
        system="S",
        messages=[],
        tools=[{"name": "get_measurement", "input_schema": {"type": "object"}}],
        max_tokens=128,
        response_schema=schema,
    )

    assert "response_format" not in completions.kwargs
    schema_note = completions.kwargs["messages"][-1]["content"]
    assert "Return one JSON object" in schema_note
    assert "large top-level explanation" not in schema_note
    assert "large field explanation" not in schema_note
    assert '"required":["status"]' in schema_note


def test_qwen_prompt_only_phase_schema_is_materially_smaller():
    schema = diagnostic_prompts.SURVEY_STATE_SCHEMA
    original = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    outline = json.dumps(
        _compact_schema(schema),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    assert len(outline) < len(original) * 0.35
    assert '"maxItems":0' in outline
    assert "acute_occlusion_pattern" not in outline
    assert '"base_version"' in outline


def test_qwen_backend_registration_and_audit_are_native_and_secret_free():
    backend, _ = _backend()
    built = build_backend(
        "qwen",
        model="qwen-test",
        api_key="secret",
        client=backend.client,
    )

    assert isinstance(built, QwenLocalBackend)
    assert built.capabilities.native_tool_calls is True
    assert built.capabilities.hard_phase_guards is True
    assert built.capabilities.context_compaction is True
    assert built.capabilities.phase_memory is True
    assert built.capabilities.citation_aliases is True
    assert built.name == "qwen-local"
    audit = built.audit_config()
    assert audit["model"] == "qwen-test"
    assert "secret" not in str(audit)
    assert "--tool-call-parser qwen3_xml" in audit["required_vllm_flags"]


def test_qwen_sessions_share_client_but_not_audit_state():
    backend, _ = _backend()
    backend.turns.append({"model": "old"})
    backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="call_alias",
                name="get_measurement",
                text="rate [ev:/global_features/heart_rate_bpm]",
                citations=("/global_features/heart_rate_bpm",),
            )
        ]
    )

    session = backend.new_session()

    assert session.client is backend.client
    assert session.turns == []
    assert session.reasoning == []
    assert session.citation_aliases() == {}


def test_qwen_compacts_resolved_native_calls_and_revision_history():
    backend, _ = _backend()
    rate_result = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="call_qwen_1",
                name="get_measurement",
                text=(
                    "heart rate 50 bpm "
                    "[ev:/global_features/heart_rate_bpm]"
                ),
                citations=("/global_features/heart_rate_bpm",),
                arguments={"pointer": "heart_rate_bpm"},
            )
        ]
    )[0]
    transient = [
        {"role": "user", "content": "Investigate the rate."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_qwen_1",
                    "type": "function",
                    "function": {
                        "name": "get_measurement",
                        "arguments": '{"pointer":"heart_rate_bpm"}',
                    },
                }
            ],
        },
        rate_result,
    ]

    compacted = backend.compact_phase_context("investigate", transient)

    assert all(message["role"] not in {"assistant", "tool"} for message in compacted)
    assert "Q1" in compacted[-1]["content"]
    assert "ev:/global_features/heart_rate_bpm" not in compacted[-1]["content"]
    memory = backend.phase_memory_turn(
        "investigate",
        '{"phase_summary":"regular bradycardia"}',
        compacted,
    )
    assert backend.is_phase_memory_turn(memory[0])
    assert "regular bradycardia" in memory[0]["content"]
    assert "heart_rate_bpm" not in memory[0]["content"]
    assert "uncommitted_evidence_views" not in memory[0]["content"]

    candidate = {"summary": "latest", "diagnoses": []}
    revision = backend.compact_revision_context(
        [
            {"role": "user", "content": "initial briefing"},
            *memory,
            {"role": "assistant", "content": '{"summary":"old"}'},
        ],
        candidate,
    )
    assert len(revision) == 3
    assert '"summary":"latest"' in revision[-1]["content"]
    assert '"summary":"old"' not in str(revision)

    qt_result = backend.tool_result_turn(
        [
            ToolOutcome(
                call_id="call_qwen_2",
                name="get_measurement",
                text="QT 438 ms [ev:/global_features/qt_ms]",
                citations=("/global_features/qt_ms",),
                arguments={"pointer": "qt_ms"},
            )
        ]
    )[0]
    reread = [
        {"role": "user", "content": "Re-read QT."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_qwen_2",
                    "type": "function",
                    "function": {
                        "name": "get_measurement",
                        "arguments": '{"pointer":"qt_ms"}',
                    },
                }
            ],
        },
        qt_result,
    ]
    backend.compact_phase_context("revise", reread)
    next_revision = backend.compact_revision_context(revision, candidate)
    assert "Q2" in str(next_revision)
    assert "ev:/global_features/qt_ms" not in str(next_revision)


def test_qwen_compacts_canonical_ledger_and_normalizes_short_citations():
    backend, _ = _backend()
    canonical = {
        "ledger_schema": "diagnostic-ledger.v3",
        "version": 3,
        "current_phase": "investigate",
        "frozen": False,
        "input_fingerprint": "sha256:large-record-hash",
        "domains": {
            "rate_rhythm": {
                "status": "completed",
                "claims": [
                    {
                        "text": "regular bradycardia",
                        "citations": ["Ecanonical"],
                    }
                ],
            }
        },
        "hypotheses": [
            {
                "id": "H1",
                "diagnosis_code": "sinus_bradycardia",
                "phenotype": "slow sinus rhythm",
                "status": "supported",
                "support": ["Ecanonical"],
                "audit_only_field": "drop me",
            }
        ],
        "evidence": {
            "Ecanonical": {
                "id": "Ecanonical",
                "pointer": "/global_features/heart_rate_bpm",
                "raw_value": 50.1234,
                "display_value": 50,
                "unit": "bpm",
                "reliability": "usable",
                "caveats": [],
                "source": "ecgfeat",
                "evidence_family": "global_features",
                "input_fingerprint": "sha256:large-record-hash",
                "first_seen_phase": "survey",
            }
        },
        "targeted_checks": [],
        "detector_conflicts": [],
        "unresolved": [],
        "allowed_final_placement": {},
    }

    memory = backend.phase_memory_turn(
        "investigate",
        json.dumps(canonical),
    )[0]["content"]

    assert '"Q1":{"value":50,"unit":"bpm"' in memory
    assert '"support":["Q1"]' in memory
    assert "Ecanonical" not in memory
    assert "/global_features/heart_rate_bpm" not in memory
    assert "large-record-hash" not in memory
    assert "audit_only_field" not in memory
    assert backend.normalize_verdict(
        {"evidence": [{"citations": ["Q1", "ev:Q1"]}]}
    ) == {
        "evidence": [
            {
                "citations": [
                    "ev:/global_features/heart_rate_bpm",
                    "ev:/global_features/heart_rate_bpm",
                ]
            }
        ]
    }


def test_medgemma_backend_rejects_qwen_checkpoint_with_migration_help(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_5"}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="--backend qwen-local"):
        MedGemmaLocalBackend(model=str(tmp_path))
