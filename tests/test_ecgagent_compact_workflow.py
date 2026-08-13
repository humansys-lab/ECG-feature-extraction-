from __future__ import annotations

import json
from types import SimpleNamespace

from ecgagent.agent import compact_diagnostic_prompts
from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.agent.loop import PhaseRecord
from ecgagent.agent.protocol import DEFAULT_DIAGNOSTIC_PROTOCOL
from ecgagent.agent.rule_second_opinion import build_rule_second_opinion
from ecgagent.backends.qwen_local import QwenLocalBackend
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_tools import _payload


def _response(*, content: str | None = None, tool: str | None = None, args=None):
    message = SimpleNamespace(
        content=content,
        reasoning_content="",
        tool_calls=[],
        refusal=None,
    )
    finish_reason = "stop"
    if tool is not None:
        finish_reason = "tool_calls"
        message.tool_calls = [
            SimpleNamespace(
                id="compact_call_1",
                function=SimpleNamespace(
                    name=tool,
                    arguments=json.dumps(args or {}),
                ),
            )
        ]
    return SimpleNamespace(
        model="qwen-test",
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


class _SequenceCompletions:
    def __init__(self, responses):
        self.responses = list(responses)

    def create(self, **_kwargs):
        return self.responses.pop(0)


def test_compact_qwen_workflow_stops_after_named_interval_coverage():
    plan = {
        "candidates": [],
        "review_tools": [],
        "quality_limitations": ["QT measurement is limited"],
    }
    decision = {
        "overall_status": "insufficient_evidence",
        "decisions": [],
        "interval_contexts": [],
        "limitations": ["QT measurement is limited"],
        "human_review_reasons": ["Review the original waveforms"],
    }
    completions = _SequenceCompletions(
        [
            _response(content=json.dumps(plan, ensure_ascii=False)),
            _response(
                tool="get_interval_waveform_context",
                args={"interval": "qt"},
            ),
            _response(content=json.dumps(decision, ensure_ascii=False)),
        ]
    )
    backend = QwenLocalBackend(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
        thinking=False,
    )
    payload = _payload()
    # This test isolates deterministic limited-QT coverage. Rule merging has
    # separate tests below and would intentionally add a PR candidate/view.
    payload.pop("clinical_interpretation")
    result = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="COMPACT-E2E"),
        backend=backend,
        workflow="compact",
        max_revisions=0,
    ).run()

    assert result.ok and result.verified, result.summary()
    assert [phase.key for phase in result.phases] == ["plan", "adjudicate"]
    assert result.phases[1].tool_budget == 1
    assert result.phases[1].tool_calls == 1
    assert result.verdict["diagnoses"] == []
    assert result.verdict["interval_measurement_contexts"][0]["interval"] == "QT_QTc"
    assert result.audit["diagnostic_workflow"] == "compact"


def _empty_backend():
    return QwenLocalBackend(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_SequenceCompletions([]))
        ),
        thinking=False,
    )


def test_compact_plan_canonicalizes_non_english_non_routing_prose():
    backend = _empty_backend()
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="PLAN-LANGUAGE"),
        backend=backend,
        workflow="compact",
    )
    backend._alias_pointers["Q1"] = "/global_features/heart_rate_bpm"
    agent.registry.seed_citations(
        ["/global_features/heart_rate_bpm"],
        source="test_plan_overview",
        model_text="ev:/global_features/heart_rate_bpm",
    )
    record = PhaseRecord(
        "plan",
        text=json.dumps(
            {
                "candidates": [
                    {
                        "id": "h1",
                        "code": "first_degree_av_block",
                        "domains": ["intervals"],
                        "support": ["Q1"],
                        "counter": [],
                        "checks": [
                            {
                                "tool": "get_interval_waveform_context",
                                "purpose": "falsify",
                                "question": "复核心律机制",
                            }
                        ],
                        "uncertainty": "需要复核",
                    }
                ],
                "review_tools": [],
                "quality_limitations": ["信号质量需要复核"],
            },
            ensure_ascii=False,
        ),
    )

    agent._capture_compact_plan(record, [])

    assert record.phase_guard_passed is True
    candidate = agent._compact_plan["candidates"][0]
    assert candidate["checks"][0]["tool"] == "get_interval_waveform_context"
    assert candidate["checks"][0]["purpose"] == "falsify"
    assert candidate["checks"][0]["question"].startswith("Review the selected")
    assert candidate["uncertainty"] == "Targeted measurement review is required."
    assert agent._compact_plan["quality_limitations"] == [
        "The overview indicates a quality limitation requiring human review."
    ]
    assert len(record.phase_sanitizations) == 3


def test_compact_system_prompt_defers_to_adaptive_candidate_limit():
    assert "at most three clinically meaningful candidate" not in (
        compact_diagnostic_prompts.SYSTEM_PROMPT
    )
    assert "record-specific limit" in compact_diagnostic_prompts.SYSTEM_PROMPT


def test_rule_second_opinion_is_bounded_and_contains_no_patient_measurements():
    payload = _payload()
    packet = build_rule_second_opinion(
        EvidenceStore.from_dict(payload, record_id="RULE-PACKET")
    )

    assert [row["code"] for row in packet["candidate_rows"]] == [
        "first_degree_av_delay"
    ]
    row = packet["candidate_rows"][0]
    assert row["publication"] == "final"
    assert row["confidence"] == "MEDIUM"
    assert "statement" not in row
    assert "evidence" not in row
    assert "thresholds" not in row
    assert "not_matched" not in json.dumps(packet)

    payload["clinical_interpretation"]["final_statements"][0]["status"] = (
        "not_matched"
    )
    negative_packet = build_rule_second_opinion(
        EvidenceStore.from_dict(payload, record_id="RULE-NON-MATCH")
    )
    assert negative_packet["candidate_rows"] == []


def test_compact_rule_candidate_is_added_only_after_blind_plan():
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="RULE-MERGE"),
        backend=_empty_backend(),
        workflow="compact",
    )

    # The active tool store remains physically scrubbed; only the private,
    # non-citable second-opinion packet retains a rule candidate.
    assert "clinical_interpretation" not in agent.store.document
    merged = agent._merge_compact_rule_candidates([])

    assert len(merged) == 1
    candidate = merged[0]
    assert candidate["code"] == "first_degree_av_delay"
    assert candidate["support"] == []
    assert candidate["sources"] == ["ecgfeat_rule_second_opinion"]
    assert [check["tool"] for check in candidate["checks"]] == [
        "get_interval_waveform_context",
        "get_atrial_event_table",
    ]
    assert [step["id"] for step in candidate["diagnostic_pathway"]["steps"]] == [
        "pr_criterion",
        "one_to_one_av",
    ]
    assert agent._compact_prefetch_arguments(
        "get_interval_waveform_context", merged
    )["interval"] == "pr"


def test_blind_candidate_is_preserved_and_rule_only_annotates_duplicate():
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="RULE-DEDUP"),
        backend=_empty_backend(),
        workflow="compact",
    )
    independent = {
        "id": "h1",
        "code": "first_degree_av_delay",
        "domains": ["intervals", "p_av"],
        "support": ["Q1"],
        "counter": [],
        "checks": [
            {
                "tool": "get_interval_waveform_context",
                "purpose": "falsify",
                "question": "复核PR构成波",
            }
        ],
        "uncertainty": "PR可靠性待复核",
    }

    merged = agent._merge_compact_rule_candidates([independent])

    assert len(merged) == 1
    assert merged[0]["support"] == ["Q1"]
    assert merged[0]["sources"] == [
        "independent_measurement_plan",
        "ecgfeat_rule_second_opinion",
    ]
    assert merged[0]["rule_second_opinion"]["confidence"] == "MEDIUM"


def test_semantic_family_duplicate_annotates_instead_of_consuming_a_slot():
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="RULE-FAMILY-DEDUP"),
        backend=_empty_backend(),
        workflow="compact",
    )
    independent = {
        "id": "h1",
        "code": "first_degree_av_block",
        "domains": ["intervals", "p_av"],
        "support": ["Q1"],
        "counter": [],
        "checks": [],
        "uncertainty": "PR关联待复核",
    }

    merged = agent._merge_compact_rule_candidates([independent])

    assert [row["code"] for row in merged] == ["first_degree_av_block"]
    assert merged[0]["sources"] == [
        "independent_measurement_plan",
        "ecgfeat_rule_second_opinion",
    ]
    assert merged[0]["rule_second_opinions"][0]["code"] == (
        "first_degree_av_delay"
    )
    assert agent._compact_rule_merge_audit["rule_annotated_family_codes"] == [
        "first_degree_av_delay"
    ]


def test_rule_priority_and_mechanism_precede_generic_high_confidence_observation():
    payload = _payload()
    base = dict(payload["clinical_interpretation"]["final_statements"][0])
    preexcitation = {
        **base,
        "statement_code": "ventricular_preexcitation_pattern",
        "statement": "Ventricular pre-excitation pattern",
        "priority": "P2",
        "confidence": "MEDIUM",
        "severity": "abnormal",
        "rule_id": "PREEXCITATION",
    }
    bradycardia = {
        **base,
        "statement_code": "bradycardia",
        "statement": "Bradycardia",
        "priority": "P5",
        "confidence": "HIGH",
        "severity": "observation",
        "rule_id": "RATE",
    }
    payload["clinical_interpretation"]["final_statements"] = [
        bradycardia,
        preexcitation,
    ]

    packet = build_rule_second_opinion(
        EvidenceStore.from_dict(payload, record_id="RULE-RANK")
    )

    assert [row["code"] for row in packet["candidate_rows"][:2]] == [
        "ventricular_preexcitation_pattern",
        "bradycardia",
    ]


def test_specific_ectopy_mechanism_precedes_nonurgent_generic_morphology():
    payload = _payload()
    base = dict(payload["clinical_interpretation"]["final_statements"][0])
    poor_progression = {
        **base,
        "statement_code": "poor_r_wave_progression",
        "priority": "P2",
        "confidence": "HIGH",
        "rule_id": "R_PROGRESSION",
    }
    ventricular_ectopy = {
        **base,
        "statement_code": "premature_ventricular_complexes",
        "priority": "P5",
        "confidence": "MEDIUM",
        "rule_id": "PVC",
    }
    payload["clinical_interpretation"]["final_statements"] = [
        poor_progression,
        ventricular_ectopy,
    ]

    packet = build_rule_second_opinion(
        EvidenceStore.from_dict(payload, record_id="RULE-MECHANISM-RANK")
    )

    assert [row["code"] for row in packet["candidate_rows"][:2]] == [
        "premature_ventricular_complexes",
        "poor_r_wave_progression",
    ]


def test_candidate_specific_morphology_views_are_not_collapsed_by_tool_name():
    payload = _payload()
    payload.pop("clinical_interpretation")
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="EXACT-VIEWS"),
        backend=_empty_backend(),
        workflow="compact",
    )
    agent._compact_plan = {
        "candidates": agent._merge_compact_rule_candidates(
            [
                {
                    "id": "h1",
                    "code": "left_bundle_branch_block",
                    "domains": ["conduction_preexcitation"],
                    "checks": [],
                },
                {
                    "id": "h2",
                    "code": "secondary_t_wave_abnormality",
                    "domains": ["q_st_t_u"],
                    "checks": [],
                },
            ]
        ),
        "review_tools": [],
        "quality_limitations": [],
    }
    from ecgagent.agent.diagnostic import compact_diagnostic_phases

    runtime = agent._compact_runtime_phase_spec(
        compact_diagnostic_phases(agent.store)[1]
    )
    calls = [(name, dict(arguments)) for name, arguments in runtime.prefetch_tool_calls]

    assert (
        "get_qrs_measurement_bundle",
        {"leads": ["I", "II", "III", "aVL", "aVF", "V1", "V2", "V5", "V6"]},
    ) in calls
    assert ("get_morphology_map", {"profile": "t_u"}) in calls
    assert agent._tool_selection_audit[-1]["missing_required_pathway_views"] == []

    decision_schema = runtime.response_schema["properties"]["decisions"]
    assert decision_schema["minItems"] == 2
    assert decision_schema["maxItems"] == 2
    decision_options = decision_schema["items"]["oneOf"]
    assert {
        option["properties"]["id"]["enum"][0]
        for option in decision_options
    } == {"h1", "h2"}
    assert all(
        option["properties"]["pathway_steps"]["minItems"] >= 1
        for option in decision_options
    )


def test_program_binds_model_claim_lead_to_selected_evidence_pointer():
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="LEAD-BIND"),
        backend=_empty_backend(),
        workflow="compact",
    )

    pointer = "/representative_leads/V2/params/r_amp_mv"
    alias = agent.backend._citation_alias(pointer)
    agent.registry.seed_citations(
        [pointer],
        source="lead-binding-test",
        model_text=f"ev:{pointer}",
    )
    items = agent._compact_evidence_items(
        [
            {
                "citation": alias,
                "claim": "V3导联R波支持该形态",
            }
        ]
    )

    assert items[0]["claim"].startswith("Lead V2: ")
    assert "V3" not in items[0]["claim"]


def test_rule_status_cannot_confirm_a_diagnosis_without_measurement_evidence():
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="RULE-NO-BYPASS"),
        backend=_empty_backend(),
        workflow="compact",
    )
    agent._compact_plan = {
        "candidates": agent._merge_compact_rule_candidates([]),
        "review_tools": [],
        "quality_limitations": [],
    }
    candidate = agent._compact_plan["candidates"][0]
    verdict = agent._expand_compact_verdict(
        {
            "overall_status": "confirmed_diagnosis",
            "decisions": [
                {
                    "id": candidate["id"],
                    "code": candidate["code"],
                    "urgency": "NONE",
                    "pathway_steps": [
                        {
                            "id": step["id"],
                            "status": "pass",
                            "evidence": [],
                            "note": "规则状态提示",
                        }
                        for step in candidate["diagnostic_pathway"]["steps"]
                    ],
                }
            ],
            "interval_contexts": [],
            "limitations": [],
            "human_review_reasons": ["复核原始波形"],
        }
    )

    assert verdict["diagnoses"] == []
    assert verdict["differential_diagnoses"] == []
    assert verdict["abstentions"][0]["topic"] == "First-degree AV delay"
    assert agent._compact_decision_audit[
        "rule_candidates_without_measurement_adjudication"
    ] == ["first_degree_av_delay"]


def test_unique_code_repairs_small_model_step_id_used_as_candidate_id():
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="DECISION-ID-REPAIR"),
        backend=_empty_backend(),
        workflow="compact",
    )
    candidate = agent._merge_compact_rule_candidates(
        [
            {
                "id": "h1",
                "code": "rvh_pattern",
                "domains": ["voltage_chamber_r_progression"],
                "checks": [],
                "support": [],
            }
        ]
    )[0]
    agent._compact_plan = {
        "candidates": [candidate],
        "review_tools": [],
        "quality_limitations": [],
    }

    agent._expand_compact_verdict(
        {
            "overall_status": "insufficient_evidence",
            "decisions": [
                {
                    "id": "right_precordial_voltage",
                    "code": "rvh_pattern",
                    "urgency": "ROUTINE",
                    "pathway_steps": [],
                }
            ],
            "interval_contexts": [],
        }
    )

    assert agent._compact_decision_audit["rejected_model_rows"] == []
    assert agent._compact_decision_audit["repaired_model_rows"] == [
        {
            "index": 0,
            "code": "rvh_pattern",
            "submitted_id": "right_precordial_voltage",
            "repaired_id": "h1",
            "reason": "unique_code_to_validated_candidate_id",
        }
    ]
    assert agent._compact_decision_audit["decisions"][0][
        "model_decision_present"
    ] is True


def test_definition_level_wide_qrs_path_nodes_are_program_owned():
    payload = _payload()
    payload["groups"] = {
        "1": {
            "member_pct": 92.0,
            "mean_qrs_ms": 140.0,
            "flags": {"dominant_group": True},
        },
        "2": {
            "member_pct": 8.0,
            "mean_qrs_ms": 90.0,
            "flags": {"dominant_group": False},
        },
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="PATHWAY-QRS"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/groups/1/mean_qrs_ms": {"get_morphology_groups"},
        "/groups/1/member_pct": {"get_morphology_groups"},
    }

    wide = agent._compact_deterministic_pathway_step(
        code="nonspecific_ivcd",
        step_id="dominant_qrs_wide",
        expected_tool="get_morphology_groups",
        pointer_tools=pointer_tools,
    )
    representative = agent._compact_deterministic_pathway_step(
        code="nonspecific_ivcd",
        step_id="wide_qrs_representative",
        expected_tool="get_morphology_groups",
        pointer_tools=pointer_tools,
    )

    assert wide is not None and wide[0] == "pass"
    assert representative is not None and representative[0] == "pass"
    assert len(representative[1]) == 2


def test_candidate_only_delta_detector_cannot_refute_ivcd():
    payload = _payload()
    payload.setdefault("rhythm_inputs", {})["preexcitation"] = {
        "short_pr_interval": False,
        "delta_lead_count": 2,
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="PATHWAY-DELTA"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/rhythm_inputs/preexcitation/short_pr_interval": {"get_rhythm_profile"},
        "/rhythm_inputs/preexcitation/delta_lead_count": {"get_rhythm_profile"},
    }

    result = agent._compact_deterministic_pathway_step(
        code="nonspecific_ivcd",
        step_id="preexcitation_excluded",
        expected_tool="get_rhythm_profile",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "unknown"
    assert len(result[1]) == 2


def test_rate_and_sinus_mechanism_nodes_are_program_owned():
    payload = _payload()
    payload["global_features"]["heart_rate_bpm"] = 52.0
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="PATHWAY-RATE"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/global_features/heart_rate_bpm": {"get_global_table"},
    }

    result = agent._compact_deterministic_pathway_step(
        code="sinus_bradycardia",
        step_id="rate_threshold",
        expected_tool="get_global_table",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "pass"
    path = agent._merge_compact_rule_candidates(
        [
            {
                "id": "h1",
                "code": "sinus_bradycardia",
                "domains": ["rhythm_rate"],
                "checks": [],
            }
        ]
    )[0]["diagnostic_pathway"]
    assert [step["id"] for step in path["steps"]] == [
        "rate_threshold",
        "sinus_mechanism_support",
        "sinus_p_support",
        "sinus_candidate_stream_reconciled",
    ]
    assert all(step["owner"] == "program" for step in path["steps"])


def test_sinus_mechanism_uses_direct_rates_and_p_axis_not_candidate_event_counts():
    payload = _payload()
    payload["global_features"].update(
        {
            "heart_rate_bpm": 106.6,
            "atrial_rate_bpm": 106.0,
            "p_axis_deg": 79.9,
        }
    )
    payload["rhythm_inputs"] = {
        "av_block": {"evidence": {"atrial_events_per_rr": [3, 1, 2, 0, 2]}}
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-SINUS-FACT"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        f"/global_features/{field}": {"get_global_table"}
        for field in ("heart_rate_bpm", "atrial_rate_bpm", "p_axis_deg")
    }

    result = agent._compact_deterministic_pathway_step(
        code="sinus_tachycardia",
        step_id="sinus_mechanism_support",
        expected_tool="get_global_table",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "pass"
    assert result[2] == "compatible_p_axis_and_atrial_ventricular_rates"


def test_sinus_candidate_stream_stays_unknown_when_excess_needs_validation():
    payload = _payload()
    payload["rhythm_inputs"] = {
        "av_block": {
            "evidence": {
                "localized_atrial_event_excess": True,
                "constant_multiple_atrial_events_requires_validation": True,
            }
        }
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-SINUS-STREAM"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/rhythm_inputs/av_block/evidence/localized_atrial_event_excess": {
            "get_rhythm_profile"
        },
        (
            "/rhythm_inputs/av_block/evidence/"
            "constant_multiple_atrial_events_requires_validation"
        ): {"get_rhythm_profile"},
    }

    result = agent._compact_deterministic_pathway_step(
        code="sinus_tachycardia",
        step_id="sinus_candidate_stream_reconciled",
        expected_tool="get_rhythm_profile",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "unknown"
    assert result[2] == "unresolved_atrial_candidate_stream_excess"


def test_preexcitation_components_require_short_pr_and_multilead_delta():
    payload = _payload()
    payload.setdefault("rhythm_inputs", {})["preexcitation"] = {
        "short_pr_interval": True,
        "short_pr_segment": False,
        "delta_lead_count": 3,
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="PATHWAY-PREEXCITATION"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/rhythm_inputs/preexcitation/short_pr_interval": {"get_rhythm_profile"},
        "/rhythm_inputs/preexcitation/short_pr_segment": {"get_rhythm_profile"},
        "/rhythm_inputs/preexcitation/delta_lead_count": {"get_rhythm_profile"},
    }

    result = agent._compact_deterministic_pathway_step(
        code="ventricular_preexcitation_pattern",
        step_id="preexcitation_components",
        expected_tool="get_rhythm_profile",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "pass"
    assert len(result[1]) == 2
    path = agent._merge_compact_rule_candidates(
        [
            {
                "id": "h1",
                "code": "ventricular_preexcitation_pattern",
                "domains": ["conduction_preexcitation"],
                "checks": [],
            }
        ]
    )[0]["diagnostic_pathway"]
    assert {
        step["id"]: step["gate"] for step in path["steps"]
    }["pr_component"] == "supporting"


def test_wide_complex_tachycardia_is_dropped_when_record_max_rate_is_not_tachycardic():
    payload = _payload()
    payload["global_features"]["heart_rate_max_bpm"] = 55.0
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="WCT-RATE-CONFLICT"),
        backend=_empty_backend(),
        workflow="compact",
    )

    conflict = agent._compact_candidate_semantic_conflict(
        "wide_complex_tachycardia"
    )

    assert conflict is not None
    assert "wide QRS without a tachycardic sequence" in conflict


def _adjudicate_lead_table(agent):
    """Fetch one real per-lead view inside the adjudication phase."""

    arguments = {
        "fields": ["q_duration_ms", "q_amp_mv", "r_amp_mv", "qrs_ms"],
    }
    agent.registry.begin_phase("adjudicate", budget=8)
    result = agent.registry.call("get_lead_table", arguments)
    assert result.ok
    agent.registry.authorize_model_visible(
        tool="get_lead_table",
        arguments=arguments,
        citations=list(result.citations),
        source="namespace-test",
    )
    return sorted(agent._compact_new_evidence_pointers())


def test_sibling_view_of_same_namespace_authorizes_a_pathway_step():
    """A step keeps its status when a sibling view supplied the measurement.

    `get_native_beat_profile` and `get_lead_table` both project
    `/representative_leads`.  Requiring the exact named view collapsed such a
    step to `unknown` whenever its tool lost a prefetch slot, which is what
    stranded every `prior_infarct_q_wave_pattern` path in the diverse batch.
    """

    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="NS-SIBLING"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointers = _adjudicate_lead_table(agent)
    pointer = next(p for p in pointers if p.endswith("q_duration_ms"))

    assert compact_diagnostic_prompts.namespace_authorizes_step(
        pointer,
        "get_native_beat_profile",
    )
    # The exact view still authorizes, and an unrelated namespace never does.
    assert compact_diagnostic_prompts.namespace_authorizes_step(
        pointer,
        "get_lead_table",
    )
    assert not compact_diagnostic_prompts.namespace_authorizes_step(
        pointer,
        "get_p_assessment_table",
    )
    assert not compact_diagnostic_prompts.namespace_authorizes_step(
        "/p_wave_assessments/0/status",
        "get_lead_table",
    )


def test_namespace_relaxation_still_rejects_evidence_not_fetched_this_phase():
    """Freshness is the anti-fabrication guard and must survive the relaxation.

    A pointer in the right namespace that no adjudication call returned is
    still unusable, so a remembered or invented measurement cannot complete a
    pathway node.
    """

    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="NS-STALE"),
        backend=_empty_backend(),
        workflow="compact",
    )
    agent.registry.begin_phase("adjudicate", budget=8)
    fresh = agent._compact_new_evidence_pointers()

    assert fresh == set()
    # The namespace map would accept this pointer, but nothing fetched it.
    assert compact_diagnostic_prompts.namespace_authorizes_step(
        "/representative_leads/I/params/q_duration_ms",
        "get_native_beat_profile",
    )


def test_tool_ceiling_covers_the_widest_observed_pathway_demand():
    """The ceiling must fit a full path plus both required interval views.

    It bounds prefetching and candidate admission at once, so a value below
    measured demand silently drops candidates before adjudication.
    """

    ceiling = DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_tool_ceiling
    widest_pathway_signatures = 10
    required_interval_views = 2

    assert ceiling >= widest_pathway_signatures + required_interval_views


def test_v38_pathways_mark_program_nodes_and_add_qtc_definition_gate():
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_payload(), record_id="V38-OWNERS"),
        backend=_empty_backend(),
        workflow="compact",
    )
    path = agent._merge_compact_rule_candidates(
        [
            {
                "id": "h1",
                "code": "prolonged_qt",
                "domains": ["intervals"],
                "checks": [],
            }
        ]
    )[0]["diagnostic_pathway"]

    assert [step["id"] for step in path["steps"]] == [
        "interval_reportable",
        "qt_threshold",
        "component_endpoint_support",
    ]
    assert {step["owner"] for step in path["steps"]} == {"program"}
    assert "Omit every `owner=program`" in compact_diagnostic_prompts.SYSTEM_PROMPT


def test_reportable_marked_qtc_threshold_is_program_owned():
    payload = _payload()
    payload["global_features"].update(
        {
            "qt_reportable": True,
            "qt_reliability": "reliable",
            "qtc_fridericia_ms": 512.0,
        }
    )
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-QTC"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/global_features/qt_reportable": {"get_interval_waveform_context"},
        "/global_features/qt_reliability": {"get_interval_waveform_context"},
        "/global_features/qtc_fridericia_ms": {"get_interval_waveform_context"},
    }

    result = agent._compact_deterministic_pathway_step(
        code="markedly_prolonged_qt",
        step_id="qt_threshold",
        expected_tool="get_interval_waveform_context",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "pass"
    assert result[2] == "qtc_meets_marked_prolongation_threshold"
    assert result[3]["qtc_ms"] == 512.0


def test_candidate_atrial_excess_does_not_become_validated_av_ratio():
    payload = _payload()
    payload["rhythm_inputs"] = {
        "av_block": {
            "evidence": {
                "atrial_events_per_rr": [2, 1, 2, 1, 2, 1],
                "pr_series_ms": [180.0, 182.0, 179.0, 181.0, 180.0],
                "dropped_p_evidence": False,
                "dropped_p_interval_indices": [],
                "constant_multiple_atrial_events_requires_validation": False,
            }
        }
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-AV-RATIO"),
        backend=_empty_backend(),
        workflow="compact",
    )
    root = "/rhythm_inputs/av_block/evidence"
    pointer_tools = {
        f"{root}/{field}": {"get_rhythm_profile"}
        for field in (
            "atrial_events_per_rr",
            "pr_series_ms",
            "dropped_p_evidence",
            "dropped_p_interval_indices",
            "constant_multiple_atrial_events_requires_validation",
        )
    }

    result = agent._compact_deterministic_pathway_step(
        code="second_degree_av_block",
        step_id="sequential_av_pattern",
        expected_tool="get_rhythm_profile",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "unknown"
    assert result[2] == "candidate_atrial_excess_without_validated_conduction_ratio"
    assert result[3]["atrial_events_per_rr_histogram"] == {1: 3, 2: 3}


def test_low_voltage_uses_program_calculated_peak_to_peak_amplitude():
    payload = _payload()
    payload["representative_leads"] = {}
    pointer_tools = {}
    for lead in ("I", "II", "III", "aVR", "aVL", "aVF"):
        payload["representative_leads"][lead] = {
            "params": {"r_amp_mv": 0.18, "s_amp_mv": -0.16, "q_amp_mv": 0.0}
        }
        for field in ("r_amp_mv", "s_amp_mv", "q_amp_mv"):
            pointer_tools[
                f"/representative_leads/{lead}/params/{field}"
            ] = {"get_lead_table"}
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-LOW-VOLTAGE"),
        backend=_empty_backend(),
        workflow="compact",
    )

    result = agent._compact_deterministic_pathway_step(
        code="low_voltage_limb_leads",
        step_id="territorial_qrs_voltage",
        expected_tool="get_lead_table",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "pass"
    assert result[2] == "all_territorial_leads_below_low_voltage_threshold"
    assert result[3]["peak_to_peak_mv"]["I"] == 0.34


def test_pacing_capture_and_failure_have_opposite_program_logic():
    payload = _payload()
    payload["rhythm_inputs"] = {
        "pacing": {
            "state": "on",
            "spike_count": 8,
            "qrs_associated_spike_fraction": 0.92,
            "capture_alignment_fraction": 0.91,
            "evidence_conflicted": False,
            "supports_measurement_routing": True,
            "intermittent_pacing": False,
            "capture_failure_suspected": False,
            "sensing_failure_suspected": {"available": False, "value": None},
        }
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-PACING"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        f"/rhythm_inputs/pacing/{field}": {"get_pacing_profile"}
        for field in (
            "state",
            "spike_count",
            "qrs_associated_spike_fraction",
            "capture_alignment_fraction",
            "evidence_conflicted",
            "supports_measurement_routing",
            "intermittent_pacing",
            "capture_failure_suspected",
        )
    }

    captured = agent._compact_deterministic_pathway_step(
        code="ventricular_paced_rhythm",
        step_id="capture_relation_support",
        expected_tool="get_pacing_profile",
        pointer_tools=pointer_tools,
    )
    failure = agent._compact_deterministic_pathway_step(
        code="pacing_failure_to_capture_suspected",
        step_id="capture_relation_support",
        expected_tool="get_pacing_profile",
        pointer_tools=pointer_tools,
    )

    assert captured is not None and captured[0] == "pass"
    assert failure is not None and failure[0] == "fail"


def test_all_program_qt_path_can_confirm_with_empty_model_step_array():
    payload = _payload()
    payload["global_features"].update(
        {
            "qt_reportable": True,
            "qt_reliability": "reliable",
            "qtc_fridericia_ms": 512.0,
        }
    )
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-PROGRAM-ONLY"),
        backend=_empty_backend(),
        workflow="compact",
    )
    candidate = {
        "id": "h1",
        "code": "markedly_prolonged_qt",
        "domains": ["intervals"],
        "checks": [],
        "support": [],
        "diagnostic_pathway": agent._merge_compact_rule_candidates(
            [
                {
                    "id": "h1",
                    "code": "markedly_prolonged_qt",
                    "domains": ["intervals"],
                    "checks": [],
                }
            ]
        )[0]["diagnostic_pathway"],
    }
    agent._compact_plan = {"candidates": [candidate], "quality_limitations": []}
    agent.registry.begin_phase("adjudicate", budget=2)
    result = agent.registry.call(
        "get_interval_waveform_context",
        {"interval": "qt", "include_beat_flags": False},
    )
    assert result.ok

    verdict = agent._expand_compact_verdict(
        {
            "overall_status": "confirmed_diagnosis",
            "decisions": [
                {
                    "id": "h1",
                    "code": "markedly_prolonged_qt",
                    "urgency": "ROUTINE",
                    "pathway_steps": [],
                }
            ],
            "interval_contexts": [],
        }
    )

    assert [row["code"] for row in verdict["diagnoses"]] == [
        "markedly_prolonged_qt"
    ]
    assert "/global_features/qtc_fridericia_ms" in agent.registry.whitelist
    assert agent.registry.model_visible_whitelist == frozenset()
    steps = agent._compact_decision_audit["decisions"][0]["pathway_steps"]
    assert all(row["resolution_owner"] == "deterministic_measurement_gate" for row in steps)


def test_legacy_program_gate_authorizes_hidden_rate_input():
    payload = _payload()
    payload["global_features"]["heart_rate_bpm"] = 112.0
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="V38-HIDDEN-RATE"),
        backend=_empty_backend(),
        workflow="compact",
    )
    candidate = {
        "id": "h1",
        "code": "tachycardia",
        "domains": ["rhythm_rate"],
        "checks": [],
    }
    candidate["diagnostic_pathway"] = agent._merge_compact_rule_candidates(
        [candidate]
    )[0]["diagnostic_pathway"]
    agent._compact_plan = {"candidates": [candidate], "quality_limitations": []}
    agent.registry.begin_phase("adjudicate", budget=1)
    result = agent.registry.call(
        "get_global_table",
        {"fields": ["heart_rate_bpm"]},
    )
    assert result.ok

    verdict = agent._expand_compact_verdict(
        {
            "overall_status": "confirmed_diagnosis",
            "decisions": [],
            "interval_contexts": [],
        }
    )

    assert [row["code"] for row in verdict["diagnoses"]] == ["tachycardia"]
    pointer = "/global_features/heart_rate_bpm"
    assert pointer in agent.registry.whitelist
    assert pointer not in agent.registry.model_visible_whitelist
    assert agent.registry.evidence_provenance(pointer)["program_authorized"] is True


def _lvh_payload(*, mean_qrs_ms: float, pacing_state: str = "off"):
    payload = _payload()
    payload["groups"] = {
        "1": {
            "member_pct": 100.0,
            "mean_qrs_ms": mean_qrs_ms,
            "flags": {"dominant_group": True},
        }
    }
    payload.setdefault("rhythm_inputs", {})["pacing"] = {
        "state": pacing_state,
        "ventricular_pacing_present": pacing_state == "on",
    }
    return payload


def _voltage_precondition(payload, step_id, tool, record_id):
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id=record_id),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/groups/1/mean_qrs_ms": {"get_morphology_groups"},
        "/groups/1/flags/dominant_group": {"get_morphology_groups"},
        "/rhythm_inputs/pacing/state": {"get_pacing_profile"},
        "/rhythm_inputs/pacing/ventricular_pacing_present": {"get_pacing_profile"},
    }
    return agent._compact_deterministic_pathway_step(
        code="lvh_voltage_criteria",
        step_id=step_id,
        expected_tool=tool,
        pointer_tools=pointer_tools,
    )


def test_voltage_criteria_are_invalidated_by_a_wide_dominant_qrs():
    result = _voltage_precondition(
        _lvh_payload(mean_qrs_ms=146.0),
        "voltage_criteria_qrs_valid",
        "get_morphology_groups",
        "LVH-WIDE",
    )

    assert result is not None
    assert result[0] == "fail"
    assert result[2] == "voltage_criteria_invalid_under_wide_qrs"


def test_voltage_criteria_survive_a_narrow_dominant_qrs():
    result = _voltage_precondition(
        _lvh_payload(mean_qrs_ms=84.0),
        "voltage_criteria_qrs_valid",
        "get_morphology_groups",
        "LVH-NARROW",
    )

    assert result is not None and result[0] == "pass"


def test_voltage_criteria_are_invalidated_by_ventricular_pacing():
    result = _voltage_precondition(
        _lvh_payload(mean_qrs_ms=110.0, pacing_state="on"),
        "voltage_criteria_pacing_valid",
        "get_pacing_profile",
        "LVH-PACED",
    )

    assert result is not None
    assert result[0] == "fail"
    assert result[2] == "voltage_criteria_invalid_under_ventricular_pacing"


def test_unmeasurable_voltage_precondition_stays_unknown_and_does_not_reject():
    payload = _lvh_payload(mean_qrs_ms=84.0)
    payload["groups"] = {}
    result = _voltage_precondition(
        payload,
        "voltage_criteria_qrs_valid",
        "get_morphology_groups",
        "LVH-NOGROUP",
    )

    assert result is not None and result[0] == "unknown"


def test_invalidator_gate_rejects_while_unknown_invalidator_does_not_stall():
    from ecgagent.agent.diagnostic_pathways import PATHWAY_GATES, build_diagnostic_pathway

    assert "invalidator" in PATHWAY_GATES
    steps = build_diagnostic_pathway("lvh_voltage_criteria")["steps"]
    gates = {step["id"]: step["gate"] for step in steps}
    assert gates["voltage_criteria_qrs_valid"] == "invalidator"
    assert gates["voltage_criteria_pacing_valid"] == "invalidator"
    # The voltage criterion itself stays `required`: an invalidator may only
    # veto, never substitute for the positive criterion.
    assert gates["cross_lead_voltage_criterion"] == "required"


def test_unknown_pathway_gate_is_rejected_at_construction():
    import pytest

    from ecgagent.agent.diagnostic_pathways import _step

    with pytest.raises(ValueError):
        _step("bogus", "问题", "get_global_table", gate="advisory")


def _lvh_verdict(mean_qrs_ms: float, record_id: str):
    """Adjudicate an LVH candidate whose voltage criterion the model passes."""

    payload = _lvh_payload(mean_qrs_ms=mean_qrs_ms)
    payload["representative_leads"]["aVL"] = {
        "lead": "aVL",
        "params": {"r_amp_mv": 1.60, "s_amp_mv": -0.10, "reliable_for_qrs": True},
    }
    payload["representative_leads"]["V3"] = {
        "lead": "V3",
        "params": {"r_amp_mv": 0.40, "s_amp_mv": -1.60, "reliable_for_qrs": True},
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id=record_id),
        backend=_empty_backend(),
        workflow="compact",
    )
    candidate = agent._merge_compact_rule_candidates(
        [
            {
                "id": "h1",
                "code": "lvh_voltage_criteria",
                "domains": ["voltage_chamber_r_progression"],
                "checks": [],
                "support": [],
            }
        ]
    )[0]
    agent._compact_plan = {
        "candidates": [candidate],
        "review_tools": [],
        "quality_limitations": [],
    }
    agent.registry.begin_phase("adjudicate", budget=6)
    for tool, arguments in (
        ("get_morphology_groups", {"include_beats": False}),
        ("get_pacing_profile", {"include_beats": False}),
        (
            "get_lead_table",
            {"fields": ["r_amp_mv", "s_amp_mv"], "leads": ["aVL", "V2", "V3", "V4"]},
        ),
    ):
        agent.registry.call(tool, arguments)
    verdict = agent._expand_compact_verdict(
        {
            "overall_status": "confirmed_diagnosis",
            "decisions": [
                {
                    "id": candidate["id"],
                    "code": candidate["code"],
                    "urgency": "ROUTINE",
                    "pathway_steps": [
                        {"id": step["id"], "status": "pass", "evidence": []}
                        for step in candidate["diagnostic_pathway"]["steps"]
                    ],
                }
            ],
            "interval_contexts": [],
            "limitations": [],
            "human_review_reasons": [],
        }
    )
    placement = agent._compact_decision_audit["decisions"][0]["final_placement"]
    return verdict, placement


def test_wide_qrs_invalidator_rejects_an_otherwise_passing_voltage_criterion():
    verdict, placement = _lvh_verdict(146.0, "LVH-REJECT")

    assert placement == "rejected"
    assert [row["code"] for row in verdict["diagnoses"]] == []


def test_narrow_qrs_leaves_the_voltage_criterion_free_to_confirm():
    _, placement = _lvh_verdict(84.0, "LVH-KEEP")

    assert placement != "rejected"


def test_shadow_mode_returns_the_nodes_to_the_model_and_still_records_the_program_answer(
    monkeypatch,
):
    """Shadow mode must flip both halves or it silently stalls every path.

    The program answer is what routes a node's view away from the model, so a
    shadow switch that only changed placement would leave the model unable to
    cite anything and turn every program node into `unknown`.
    """

    from ecgagent.agent import deterministic_pathways

    monkeypatch.setenv(deterministic_pathways.DETERMINISTIC_NODE_SHADOW_ENV, "1")
    assert deterministic_pathways.deterministic_nodes_shadowed() is True
    assert (
        deterministic_pathways.program_owns_pathway_step(
            "prolonged_qt", "qt_threshold"
        )
        is False
    )

    shadowed = build_diagnostic_pathway_owners("prolonged_qt")
    monkeypatch.delenv(deterministic_pathways.DETERMINISTIC_NODE_SHADOW_ENV)
    enforced = build_diagnostic_pathway_owners("prolonged_qt")

    assert set(shadowed.values()) == {"model"}
    assert set(enforced.values()) == {"program"}


def build_diagnostic_pathway_owners(code: str) -> dict[str, str]:
    from ecgagent.agent.diagnostic_pathways import build_diagnostic_pathway

    return {
        step["id"]: step["owner"]
        for step in build_diagnostic_pathway(code)["steps"]
    }


def test_step_audit_carries_both_model_and_program_status():
    payload = _lvh_payload(mean_qrs_ms=146.0)
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="SHADOW-AUDIT"),
        backend=_empty_backend(),
        workflow="compact",
    )
    candidate = agent._merge_compact_rule_candidates(
        [
            {
                "id": "h1",
                "code": "lvh_voltage_criteria",
                "domains": ["voltage_chamber_r_progression"],
                "checks": [],
                "support": [],
            }
        ]
    )[0]
    agent._compact_plan = {
        "candidates": [candidate],
        "review_tools": [],
        "quality_limitations": [],
    }
    agent.registry.begin_phase("adjudicate", budget=6)
    agent.registry.call("get_morphology_groups", {"include_beats": False})
    agent._expand_compact_verdict(
        {
            "overall_status": "confirmed_diagnosis",
            "decisions": [
                {
                    "id": candidate["id"],
                    "code": candidate["code"],
                    "urgency": "ROUTINE",
                    "pathway_steps": [
                        {
                            "id": "voltage_criteria_qrs_valid",
                            "status": "pass",
                            "evidence": [],
                        }
                    ],
                }
            ],
            "interval_contexts": [],
            "limitations": [],
            "human_review_reasons": [],
        }
    )
    steps = agent._compact_decision_audit["decisions"][0]["pathway_steps"]
    row = next(step for step in steps if step["id"] == "voltage_criteria_qrs_valid")

    # The model said `pass`; the program measured a wide QRS and said `fail`.
    # Both survive in the audit, which is what makes agreement measurable.
    assert row["model_status"] == "unknown"
    assert row["program_status"] == "fail"
    assert row["effective_status"] == "fail"


def test_preexcitation_exclusion_reads_the_pr_segment_not_only_the_interval():
    """A short PR *segment* with delta waves must still block exclusion.

    When the global PR interval is unmeasurable, `short_pr_interval` is False
    and reading it alone scored a nine-lead delta record as `unknown`, so the
    node excluded nothing. `preexcitation_components` already considers both
    constituents; this node has to agree with it.
    """

    payload = _payload()
    payload.setdefault("rhythm_inputs", {})["preexcitation"] = {
        "short_pr_interval": False,
        "short_pr_segment": True,
        "delta_lead_count": 9,
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="PREEXC-SEGMENT"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        "/rhythm_inputs/preexcitation/short_pr_interval": {"get_rhythm_profile"},
        "/rhythm_inputs/preexcitation/short_pr_segment": {"get_rhythm_profile"},
        "/rhythm_inputs/preexcitation/delta_lead_count": {"get_rhythm_profile"},
    }

    result = agent._compact_deterministic_pathway_step(
        code="rbbb_pattern",
        step_id="preexcitation_excluded",
        expected_tool="get_rhythm_profile",
        pointer_tools=pointer_tools,
    )

    assert result is not None and result[0] == "fail"


def test_bundle_branch_pathway_carries_a_preexcitation_invalidator():
    """Pre-excitation widens QRS, so a BBB read needs it excluded.

    It has to be an `invalidator`, not `required`: a genuine RBBB record leaves
    the pre-excitation detector at `unknown`, and a required gate would stall
    the true positive instead of filtering the false one.
    """

    from ecgagent.agent.diagnostic_pathways import build_diagnostic_pathway

    steps = {
        step["id"]: step["gate"]
        for step in build_diagnostic_pathway("rbbb_pattern")["steps"]
    }
    assert steps["preexcitation_excluded"] == "invalidator"


def _organized_p_absence_fact(rows, record_id):
    from ecgagent.agent.deterministic_pathways import (
        resolve_additional_deterministic_step,
    )

    payload = _payload()
    payload["p_wave_assessments"] = rows
    store = EvidenceStore.from_dict(payload, record_id=record_id).diagnostic_view()
    available = {
        f"/p_wave_assessments/{index}/{field}"
        for index in range(len(rows))
        for field in (
            "accepted",
            "ta_ambiguous",
            "onset_confidence",
            "offset_confidence",
            "valid_leads",
            "reject_reasons",
        )
    }
    return resolve_additional_deterministic_step(
        store,
        code="atrial_fibrillation",
        step_id="organized_p_absent",
        available_pointers=available,
    )


def test_af_gate_rejects_circular_rejection_when_strong_p_boundaries_remain():
    rows = [
        {
            "accepted": False,
            "ta_ambiguous": False,
            "onset_confidence": 0.80,
            "offset_confidence": 0.75,
            "valid_leads": ["II", "V1"],
            "reject_reasons": ["RHYTHM_AF_LIKE"],
        }
        for _ in range(6)
    ]

    fact = _organized_p_absence_fact(rows, "AF-CIRCULAR-P")

    assert fact.status == "fail"
    assert fact.reason_code == "repeated_independent_p_boundaries_present"
    assert fact.metrics["independent_p_boundary_count"] == 6


def test_af_gate_can_pass_repeated_analyzable_rows_without_p_boundaries():
    rows = [
        {
            "accepted": False,
            "ta_ambiguous": False,
            "onset_confidence": 0.15,
            "offset_confidence": 0.20,
            "valid_leads": ["II", "V1"],
            "reject_reasons": ["RHYTHM_AF_LIKE"],
        }
        for _ in range(6)
    ]

    fact = _organized_p_absence_fact(rows, "AF-NO-P")

    assert fact.status == "pass"
    assert fact.reason_code == "no_independent_p_boundaries_in_analyzable_rows"


def test_af_gate_keeps_technical_p_measurement_failure_unknown():
    rows = [
        {
            "accepted": False,
            "ta_ambiguous": False,
            "onset_confidence": 0.10,
            "offset_confidence": 0.10,
            "valid_leads": ["II", "V1"],
            "reject_reasons": ["BASELINE_UNOBSERVABLE"],
        }
        for _ in range(6)
    ]

    fact = _organized_p_absence_fact(rows, "AF-P-LIMITED")

    assert fact.status == "unknown"
    assert fact.reason_code == "organized_p_absence_not_established"


def test_open_ended_alternative_cause_node_cannot_be_a_required_gate():
    """`required` on an unprovable negative can only ever stall a candidate."""

    from ecgagent.agent.diagnostic_pathways import build_diagnostic_pathway

    for code in ("poor_r_wave_progression", "clockwise_rotation"):
        steps = {
            step["id"]: step["gate"]
            for step in build_diagnostic_pathway(code)["steps"]
        }
        assert steps["alternative_qrs_cause_excluded"] == "invalidator"


def _pvc_morphology(groups, record_id):
    from ecgagent.agent.deterministic_pathways import (
        resolve_additional_deterministic_step,
    )

    payload = _payload()
    payload["groups"] = groups
    store = EvidenceStore.from_dict(payload, record_id=record_id)
    available = {
        f"/groups/{group_id}/{field}"
        for group_id in groups
        for field in ("mean_qrs_ms", "member_count", "member_pct")
    }
    return resolve_additional_deterministic_step(
        store,
        code="premature_ventricular_complexes",
        step_id="ectopic_morphology",
        available_pointers=available,
    )


def test_a_lone_markedly_wide_complex_is_a_ventricular_morphology():
    """One 218 ms complex is a PVC; requiring repetition made it undiagnosable."""

    fact = _pvc_morphology(
        {
            "1": {
                "member_count": 13,
                "member_pct": 81.25,
                "mean_qrs_ms": 92.0,
                "flags": {"dominant_group": True},
            },
            "3": {
                "member_count": 1,
                "member_pct": 6.25,
                "mean_qrs_ms": 218.0,
                "flags": {"dominant_group": False},
            },
        },
        "PVC-LONE",
    )

    assert fact.status == "pass"
    assert fact.reason_code == "lone_markedly_wide_ectopic_morphology"


def test_a_lone_borderline_wide_complex_stays_unknown():
    """The borderline band survives: only a markedly wide lone complex passes."""

    fact = _pvc_morphology(
        {
            "1": {
                "member_count": 13,
                "member_pct": 81.25,
                "mean_qrs_ms": 92.0,
                "flags": {"dominant_group": True},
            },
            "3": {
                "member_count": 1,
                "member_pct": 6.25,
                "mean_qrs_ms": 126.0,
                "flags": {"dominant_group": False},
            },
        },
        "PVC-BORDERLINE",
    )

    assert fact.status == "unknown"


def _preexcitation_exclusion(*, short_interval, short_segment, delta_leads, pr_available):
    payload = _payload()
    payload.setdefault("rhythm_inputs", {})["preexcitation"] = {
        "short_pr_interval": short_interval,
        "short_pr_segment": short_segment,
        "delta_lead_count": delta_leads,
    }
    payload["rhythm_inputs"].setdefault("record", {})["availability"] = {
        "pr_available": pr_available
    }
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(payload, record_id="PREEXC-GATE"),
        backend=_empty_backend(),
        workflow="compact",
    )
    pointer_tools = {
        pointer: {"get_rhythm_profile"}
        for pointer in (
            "/rhythm_inputs/preexcitation/short_pr_interval",
            "/rhythm_inputs/preexcitation/short_pr_segment",
            "/rhythm_inputs/preexcitation/delta_lead_count",
            "/rhythm_inputs/record/availability/pr_available",
        )
    }
    return agent._compact_deterministic_pathway_step(
        code="rbbb_pattern",
        step_id="preexcitation_excluded",
        expected_tool="get_rhythm_profile",
        pointer_tools=pointer_tools,
    )


def test_short_pr_segment_substitutes_only_when_the_interval_is_unmeasurable():
    """The segment is the interval minus P duration, not a second opinion on it.

    A wide P wave shortens the segment while the interval stays normal, which is
    not pre-excitation. Reading the segment unconditionally vetoed a true LAFB
    whose PR measured 141 ms.
    """

    unmeasurable = _preexcitation_exclusion(
        short_interval=False, short_segment=True, delta_leads=9, pr_available=False
    )
    assert unmeasurable is not None and unmeasurable[0] == "fail"

    measured_and_normal = _preexcitation_exclusion(
        short_interval=False, short_segment=True, delta_leads=3, pr_available=True
    )
    assert measured_and_normal is not None
    assert measured_and_normal[0] != "fail"


def test_measured_short_pr_still_blocks_exclusion():
    result = _preexcitation_exclusion(
        short_interval=True, short_segment=True, delta_leads=4, pr_available=True
    )

    assert result is not None and result[0] == "fail"
