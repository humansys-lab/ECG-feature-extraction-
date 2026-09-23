from __future__ import annotations

import json

from ecgagent.agent import diagnostic_prompts
from ecgagent.agent.diagnostic import (
    DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
    ECGDiagnosticAgent,
    SURVEY_TOOLS,
    compact_diagnostic_phases,
    diagnostic_phases,
)
from ecgagent.agent import compact_diagnostic_prompts
from ecgagent.agent.diagnostic_pathways import build_diagnostic_pathway
from ecgagent.agent.loop import AgentResult
from ecgagent.agent.protocol import DEFAULT_DIAGNOSTIC_PROTOCOL
from ecgagent.agent.urgent_review import assess_urgent_review
from ecgagent.backends.mock import ScriptedBackend, text_response, tool_response
from ecgagent.evidence.store import EvidenceStore
from ecgagent.verify import VerificationPolicy, verify_structured
from tests.test_ecgagent_tools import _payload
from tests.diagnostic_script_helpers import (
    HEART_RATE,
    T_AMP,
    legacy_phase_patches,
)


def _diagnostic_verdict() -> dict:
    return {
        "summary": "Sinus rhythm is the leading independent interpretation.",
        "ranked_complete_interpretations": [
            {
                "rank": 1,
                "interpretation_type": "PRIMARY",
                "complete_diagnosis": "Sinus rhythm with limited QT/QTc measurement requiring T/U-wave review.",
                "confidence": "MEDIUM",
                "basis_codes": ["sinus_rhythm"],
                "key_uncertainty": "The original 12-lead waveforms were not reviewed directly.",
            }
        ],
        "diagnoses": [
            {
                "code": "sinus_rhythm",
                "statement": "Sinus rhythm",
                "category": "rhythm",
                "confidence": "MEDIUM",
                "urgency": "NONE",
                "evidence": [
                    {
                        "claim": "The measured heart rate is 62 bpm.",
                        "value": 62,
                        "unit": "bpm",
                        "citations": ["ev:/global_features/heart_rate_bpm"],
                    }
                ],
                "counterevidence": [],
                "reasoning": (
                    "The available rate and rhythm observations favor an organized "
                    "sinus mechanism, while direct wave review remains limited."
                ),
            }
        ],
        "differential_diagnoses": [],
        "interval_measurement_contexts": [
            {
                "interval": "QT_QTc",
                "status": "limited",
                "interval_conclusion": "QT/QTc does not meet reliable reporting conditions.",
                "component_waveform_assessment": (
                    "A limited interval does not make the T wave uninformative; available T-wave morphology still requires independent assessment."
                ),
                "residual_evidence": [
                    {
                        "claim": "QT is marked non-reportable while per-lead T-wave measurements remain available.",
                        "value": None,
                        "unit": None,
                        "citations": [
                            "ev:/global_features/qt_reportable",
                            "ev:/representative_leads/I/params/t_amp_mv",
                        ],
                    }
                ],
                "interpretive_impact": (
                    "QT values cannot establish an interval abnormality, but residual repolarization morphology cannot be ignored."
                ),
                "what_would_resolve_it": "Review T waves and T-wave endpoints across multiple leads.",
            }
        ],
        "abstentions": [],
        "quality_assessment": {
            "interpretability": "limited",
            "limitations": ["Automated feature measurements cannot replace waveform review"],
        },
        "human_review": {
            "required": True,
            "reasons": ["Research interpretation requires clinician confirmation"],
        },
    }


def test_diagnostic_prompt_makes_model_the_reasoner():
    prompt = diagnostic_prompts.SYSTEM_PROMPT
    assert "primary diagnostic reasoner" in prompt
    assert "measurement is evidence, not a diagnosis" in prompt
    assert "lowers evidence weight; it does not erase" in prompt
    assert "null/not-produced value is unavailable" in prompt
    assert "in clear English" in prompt
    assert "logic understandable to a non-specialist reader" in prompt
    assert "candidate-only" in prompt
    assert "accepted-but-T/A-ambiguous" in prompt
    assert "Review T waves as a lead distribution" in prompt
    assert "authoritative diagnostic statements" not in prompt
    assert "added" not in diagnostic_prompts.OUTPUT_SCHEMA["properties"]["diagnoses"]["items"][
        "properties"
    ]


def test_quality_stop_bypasses_the_model_with_non_diagnostic_output():
    payload = _payload()
    payload["metadata"]["diagnostic_gate"] = {
        "state": "stop",
        "stop_reasons": ["fewer_than_3_detected_beats"],
        "partial_reasons": [],
        "allowed_domains": [],
        "suppressed_domains": [],
    }
    result = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(payload, record_id="QUALITY-STOP"),
        backend=ScriptedBackend(script=[]),
        knowledge_guidance=False,
    ).run()

    assert result.ok and result.verified
    assert result.phases[0].key == "quality_gate"
    assert result.phases[0].turns == 0
    assert result.verdict["diagnoses"] == []
    assert result.verdict["differential_diagnoses"] == []
    assert result.verdict["quality_assessment"]["interpretability"] == "non_diagnostic"


def test_missing_or_incomplete_quality_gate_fails_closed_without_model_calls():
    for invalid_gate in (None, "pass", {"state": "pass"}):
        payload = _payload()
        if invalid_gate is None:
            payload["metadata"].pop("diagnostic_gate", None)
        else:
            payload["metadata"]["diagnostic_gate"] = invalid_gate
        backend = ScriptedBackend(script=[])

        result = ECGDiagnosticAgent(
            workflow="legacy",
            store=EvidenceStore.from_dict(payload, record_id="INVALID-GATE"),
            backend=backend,
            knowledge_guidance=False,
        ).run()

        assert result.ok and result.verified
        assert result.phases[0].turns == 0
        assert result.verdict["diagnoses"] == []
        assert result.audit["diagnostic_gate_validation"]["source"] == (
            "fail_closed_validation"
        )


def test_urgent_review_is_a_versioned_routing_flag_not_a_diagnosis():
    payload = _payload()
    payload["global_features"]["heart_rate_bpm"] = 35.0
    assessment = assess_urgent_review(
        EvidenceStore.from_dict(payload, record_id="URGENT-ROUTE")
    )

    assert assessment["urgent_review_required"] is True
    assert assessment["do_not_delay_human_review"] is True
    assert assessment["triggers"][0]["code"] == "extreme_bradycardia_measurement"
    assert assessment["unassessed_categories"]


def test_pediatric_age_days_disables_adult_urgent_rate_thresholds():
    payload = _payload()
    payload["metadata"]["patient_meta"].update({"age": 40, "age_days": 30})
    payload["global_features"]["heart_rate_bpm"] = 160.0

    assessment = assess_urgent_review(
        EvidenceStore.from_dict(payload, record_id="INFANT-ROUTE").diagnostic_view()
    )

    assert assessment["adult_thresholds_applicable"] is False
    assert not any(
        row["code"] == "extreme_tachycardia_measurement"
        for row in assessment["triggers"]
    )
    assert "pediatric_or_unknown_age_rate_qrs_qtc_thresholds" in assessment[
        "unassessed_categories"
    ]


def test_nonadult_pathways_leave_adjudication_steps_model_owned():
    path = build_diagnostic_pathway(
        "right_ventricular_hypertrophy",
        program_owned=False,
    )

    assert path["steps"]
    assert {step["owner"] for step in path["steps"]} == {"model"}


def test_phase_contracts_are_compact_and_phase_specific():
    survey_statuses = diagnostic_prompts.SURVEY_STATE_SCHEMA["properties"][
        "create_hypotheses"
    ]["items"]["properties"]["status"]["enum"]
    hypothesis = diagnostic_prompts.HYPOTHESIS_STATE_SCHEMA

    assert survey_statuses == ["raised"]
    assert "targeted_checks" in hypothesis["properties"]
    assert diagnostic_prompts.CHALLENGE_STATE_SCHEMA["properties"][
        "targeted_checks"
    ]["maxItems"] == 0
    assert "phase_summary" not in hypothesis["properties"]
    assert hypothesis["properties"]["create_hypotheses"]["maxItems"] == 2
    assert "diagnosis_code" in hypothesis["properties"]["create_hypotheses"]["items"][
        "required"
    ]
    evidence_item = hypothesis["properties"]["create_hypotheses"]["items"][
        "properties"
    ]["supporting_observations"]["items"]
    assert evidence_item["required"] == ["citations"]
    assert evidence_item["properties"]["citations"]["maxItems"] == 1
    assert "claim" not in evidence_item["properties"]
    assert "observation" not in diagnostic_prompts.SURVEY_STATE_SCHEMA[
        "properties"
    ]["domains"]["properties"]["intervals"]["properties"]
    assert hypothesis["properties"]["targeted_checks"]["maxItems"] == 4
    assert set(
        diagnostic_prompts.SURVEY_STATE_SCHEMA["properties"]["domains"]["required"]
    )
    assert "domains" not in hypothesis["properties"]
    assert "domains" not in diagnostic_prompts.INVESTIGATION_STATE_SCHEMA["properties"]


def test_phase_sanitizer_removes_copy_forward_transition_without_new_evidence():
    state = {
        "base_version": 1,
        "create_hypotheses": [],
        "evidence_updates": [],
        "domain_updates": [],
        "transitions": [
            {
                "hypothesis_id": "atrial_question",
                "from_status": "provisional",
                "to_status": "supported",
                "citations": ["ev:/old/evidence"],
            }
        ],
        "plan_updates": [],
        "refuting_tests": [],
        "targeted_checks": [],
        "detector_conflicts": [],
        "unresolved": [],
    }

    sanitized, notes = diagnostic_prompts.sanitize_phase_state(
        "investigate",
        state,
        phase_citations=("ev:/new/evidence",),
    )

    assert sanitized["transitions"] == []
    assert len(notes) == 1


def test_survey_exposes_only_compact_diagnostic_overview():
    store = EvidenceStore.from_dict(_payload(), record_id="SELECTIVE-SURVEY")
    survey = diagnostic_phases(store.diagnostic_view())[0]

    assert survey.allowed_tools == SURVEY_TOOLS
    assert survey.allowed_tools == ("get_diagnostic_overview",)
    assert "search_measurements" not in survey.allowed_tools
    assert "get_atrial_event_table" not in survey.allowed_tools
    assert "get_pacing_profile" not in survey.allowed_tools


def test_compact_workflow_has_two_model_decisions_and_program_prefetch():
    store = EvidenceStore.from_dict(_payload(), record_id="COMPACT-PREFETCH")
    phases = compact_diagnostic_phases(store.diagnostic_view())

    assert [phase.key for phase in phases] == ["plan", "adjudicate"]
    assert phases[0].prefetch_tool_calls == (("get_diagnostic_overview", ()),)
    assert (
        phases[1].tool_budget
        == DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_tool_ceiling
    )
    # Assert the schemas stay wired to the candidate-cap constants rather than
    # restating their values: a literal here only duplicates the constant and
    # fails whenever the cap is deliberately retuned, which is exactly what it
    # did when the merged cap was raised from 5 to 8 to stop deferring correct
    # rule-engine candidates past adjudication.
    assert (
        compact_diagnostic_prompts.PLAN_SCHEMA["properties"]["candidates"]["maxItems"]
        == compact_diagnostic_prompts.INDEPENDENT_PLAN_MAX_CANDIDATES
    )
    assert (
        compact_diagnostic_prompts.COMPACT_VERDICT_SCHEMA["properties"]["decisions"][
            "maxItems"
        ]
        == compact_diagnostic_prompts.MERGED_CANDIDATE_MAX
    )
    assert (
        compact_diagnostic_prompts.INDEPENDENT_PLAN_MAX_CANDIDATES
        <= compact_diagnostic_prompts.MERGED_CANDIDATE_MAX
    )
    assert compact_diagnostic_prompts.COMPACT_VERDICT_SCHEMA["properties"][
        "decisions"
    ]["items"]["properties"]["pathway_steps"]["minItems"] == 0
    check_tools = compact_diagnostic_prompts.PLAN_SCHEMA["properties"][
        "candidates"
    ]["items"]["properties"]["checks"]["items"]["properties"]["tool"][
        "enum"
    ]
    assert "search_measurements" not in check_tools
    assert "get_measurement" not in check_tools

    agent = ECGDiagnosticAgent(
        store=store,
        backend=ScriptedBackend(script=[]),
        knowledge_guidance=False,
        workflow="compact",
    )
    agent._compact_plan = {
        "candidates": [
            {
                "id": "C1",
                "code": "sinus_tachycardia",
                "domains": ["rhythm_rate", "ectopy_pauses"],
                "checks": [
                    {"tool": "get_rhythm_profile", "purpose": "support"},
                    {"tool": "get_morphology_groups", "purpose": "falsify"},
                ],
            },
            {
                "id": "C2",
                "code": "premature_ventricular_complexes",
                "domains": ["ectopy_pauses"],
                "checks": [
                    {"tool": "get_beat_table", "purpose": "support"},
                    {"tool": "get_lead_table", "purpose": "falsify"},
                ],
            },
        ],
        "review_tools": ["get_morphology_map", "get_pacing_profile"],
        "quality_limitations": [],
    }
    for candidate in agent._compact_plan["candidates"]:
        candidate["diagnostic_pathway"] = build_diagnostic_pathway(
            candidate["code"], fallback_checks=candidate["checks"]
        )
    runtime = agent._compact_runtime_phase_spec(phases[1])

    # Every distinct pathway view is budgeted, including model P-morphology.
    assert runtime.tool_budget == len(runtime.prefetch_tool_calls)
    assert runtime.tool_budget <= DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_tool_ceiling
    program_calls = set((name, json.dumps(dict(args), sort_keys=True))
                        for name, args in runtime.program_only_prefetch_tool_calls)
    assert program_calls
    assert len(program_calls) < runtime.tool_budget
    assert any(row[0].endswith(":sinus_p_morphology") for row in runtime.model_evidence_requirements)
    assert runtime.coverage_requirements == ()
    assert runtime.allowed_tools == tuple(
        dict.fromkeys(name for name, _arguments in runtime.prefetch_tool_calls)
    )
    first_name, first_arguments = runtime.prefetch_tool_calls[0]
    assert first_name == "get_interval_waveform_context"
    assert dict(first_arguments) == {"interval": "qt", "include_beat_flags": False}


def test_compact_human_text_drops_working_memory_q_ids():
    assert ECGDiagnosticAgent._compact_clean_text(
        "RR不规则（Q69），Q81提示房性活动。"
    ) == "RR不规则，提示房性活动。"
    assert ECGDiagnosticAgent._compact_clean_text(
        "QRS超过110-120 ms，但仍需复核Q56。"
    ) == "QRS超过，但仍需复核。"
    assert ECGDiagnosticAgent._compact_humanize_limitation("pr unavailable") == (
        "PR interval unavailable"
    )


def test_compact_conduction_and_voltage_views_are_focused_for_small_model():
    store = EvidenceStore.from_dict(_payload(), record_id="FOCUSED-PATHWAY")
    agent = ECGDiagnosticAgent(
        store=store,
        backend=ScriptedBackend(script=[]),
        knowledge_guidance=False,
        workflow="compact",
    )
    candidates = [
        {"code": "nonspecific_ivcd", "domains": ["conduction_preexcitation"]},
        {"code": "lvh_voltage_criteria", "domains": ["voltage_chamber_r_progression"]},
    ]

    assert agent._compact_prefetch_arguments(
        "get_morphology_map", candidates
    ) == {"profile": "qrs", "leads": ["I", "V1", "V5", "V6"]}
    assert agent._compact_prefetch_arguments("get_lead_table", candidates) == {
        "fields": ["r_amp_mv", "s_amp_mv"],
        "leads": ["aVL", "V2", "V3", "V4"],
    }


def test_investigation_tools_follow_targeted_hypotheses():
    store = EvidenceStore.from_dict(_payload(), record_id="SELECTIVE-INVESTIGATE")
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=store,
        backend=ScriptedBackend(script=[]),
        knowledge_guidance=False,
    )
    agent._latest_phase_state = {
        "targeted_checks": [
            {
                "hypothesis_id": "atrial_question",
                "domain": "p_av",
                "tool": "get_p_assessment_table",
                "question": "确认P波边界和房室关联",
            },
            {
                "hypothesis_id": "t_wave_question",
                "domain": "q_st_t_u",
                "tool": "get_morphology_map",
                "question": "确认T波跨导联分布",
            },
        ],
        "hypotheses": [
            {
                "id": "atrial_question",
                "phenotype": "P波形态问题",
                "domains": ["p_av"],
                "status": "provisional",
                "required_evidence": ["多搏P波稳定性"],
                "next_tools": ["get_p_assessment_table"],
            },
            {
                "id": "t_wave_question",
                "phenotype": "T波异常",
                "domains": ["q_st_t_u"],
                "status": "weak",
                "required_evidence": ["T波导联分布"],
                "next_tools": ["get_morphology_map"],
            },
        ],
    }
    investigate = next(
        phase for phase in agent.phases if phase.key == "investigate"
    )

    runtime = agent._runtime_phase_spec(investigate, AgentResult("SELECTIVE"))

    assert "get_p_assessment_table" in runtime.allowed_tools
    assert "get_morphology_map" in runtime.allowed_tools
    assert "get_pacing_profile" not in runtime.allowed_tools
    assert "search_measurements" in runtime.allowed_tools
    assert 3 <= runtime.tool_budget <= 5
    assert runtime.minimum_tool_calls == 0
    assert {
        coverage_id for coverage_id, _tools, _count in runtime.coverage_requirements
    } >= {"hypothesis:atrial_question", "hypothesis:t_wave_question"}


def test_challenge_tools_are_available_and_mandatory():
    store = EvidenceStore.from_dict(_payload(), record_id="SELECTIVE-CHALLENGE")
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=store,
        backend=ScriptedBackend(script=[]),
        knowledge_guidance=False,
    )
    agent._latest_phase_state = {
        "hypotheses": [
            {
                "id": "t_wave_question",
                "diagnosis_code": "t_wave_abnormality",
                "phenotype": "T波异常",
                "domains": ["q_st_t_u"],
                "status": "supported",
                "required_evidence": [],
                "next_tools": ["get_morphology_map"],
            }
        ]
    }
    challenge = next(phase for phase in agent.phases if phase.key == "challenge")

    runtime = agent._runtime_phase_spec(challenge, AgentResult("CHALLENGE"))

    assert 3 <= runtime.tool_budget <= 4
    assert "get_morphology_map" in runtime.allowed_tools
    assert runtime.minimum_tool_calls == 0
    assert runtime.require_novel_tool_views is True
    assert len(runtime.coverage_requirements) == 2


def test_phase_guard_requires_new_evidence_for_investigation_transition():
    state = {
        "base_version": 1,
        "transitions": [
            {
                "hypothesis_id": "atrial_question",
                "from_status": "provisional",
                "to_status": "supported",
                "citations": ["ev:/global_features/rr_cv"],
            }
        ],
    }

    sanitized, notes = diagnostic_prompts.sanitize_phase_state(
        "investigate",
        state,
        phase_citations=("/global_features/heart_rate_bpm",),
    )

    assert sanitized["transitions"] == []
    assert any("newly read" in note for note in notes)


def test_challenge_gate_does_not_treat_nondiscriminative_evidence_as_support():
    pointer = "/rhythm_inputs/af_afl/f_wave_confidence"
    state = {
        "base_version": 3,
        "refuting_tests": [
            {
                "hypothesis_id": "atrial_fibrillation",
                "would_refute": "独立形态视图不支持该机制",
                "citations": [f"ev:{pointer}"],
                "test_outcome": "not_refuted",
                "evidence_direction": "not_discriminative",
            }
        ],
        "transitions": [],
    }

    problems = diagnostic_prompts.validate_phase_state(
        "challenge",
        state,
        phase_citations=(pointer,),
        ledger_state={
            "version": 3,
            "hypotheses": [
                {"id": "atrial_fibrillation", "status": "supported"}
            ],
        },
    )

    assert any("no discriminative" in problem for problem in problems)


def test_investigation_guard_materializes_visible_component_wave_evidence():
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="COMPONENT-GUARD"),
        backend=ScriptedBackend(),
        knowledge_guidance=False,
    )
    t_pointer = "/representative_leads/I/params/t_amp_mv"
    qt_pointer = "/global_features/qt_ms"
    state = {
        "domain_updates": [
            {
                "domain": "q_st_t_u",
                "citations": [f"ev:{qt_pointer}"],
            }
        ]
    }

    problems = agent._component_wave_coverage_problems(
        state,
        new_phase_citations=(qt_pointer, t_pointer),
    )

    assert any("T/U component-wave evidence" in problem for problem in problems)
    state["domain_updates"][0]["citations"].append(f"ev:{t_pointer}")
    assert agent._component_wave_coverage_problems(
        state,
        new_phase_citations=(qt_pointer, t_pointer),
    ) == []


def test_investigation_runtime_requires_missing_interval_component_view():
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="INTERVAL-PLAN"),
        backend=ScriptedBackend(),
        knowledge_guidance=False,
    )
    investigate = next(phase for phase in agent.phases if phase.key == "investigate")

    runtime = agent._runtime_phase_spec(investigate, AgentResult("INTERVAL-PLAN"))

    assert "get_interval_waveform_context" in runtime.allowed_tools
    assert (
        "get_interval_waveform_context",
        (("interval", "qt"),),
        1,
    ) in runtime.required_tool_calls
    assert 'get_interval_waveform_context(interval="qt")' in runtime.instruction


def test_new_hypothesis_guard_rejects_ivcd_label_for_p_av_observation():
    payload = _payload()
    payload["global_features"]["qrs_ms"] = 80.0
    payload["global_features"]["qrs_wide_ms"] = 80.0
    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(payload, record_id="CODE-GUARD"),
        backend=ScriptedBackend(),
        knowledge_guidance=False,
    )
    state = {
        "create_hypotheses": [
            {
                "diagnosis_code": "nonspecific_ivcd",
                "phenotype": "non-conducted P waves",
                "domains": ["p_av", "rhythm_rate"],
            }
        ]
    }

    problems = agent._new_hypothesis_semantic_problems(state)

    assert any("intraventricular QRS-conduction identity" in item for item in problems)
    assert any("definitionally contradicted" in item for item in problems)

    state["create_hypotheses"][0].update(
        {
            "diagnosis_code": "second_degree_av_block",
            "domains": ["p_av", "rhythm_rate"],
        }
    )
    assert agent._new_hypothesis_semantic_problems(state) == []


def test_unresolved_challenge_hypothesis_cannot_enter_positive_diagnoses():
    verdict = _diagnostic_verdict()
    state = {
        "hypotheses": [
            {
                "id": "sinus_origin",
                "diagnosis_code": "sinus_rhythm",
                "status": "unresolved",
            }
        ]
    }

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
        hypothesis_state=state,
    )

    assert any("bypasses the final hypothesis gate" in problem for problem in problems)

    state["hypotheses"][0]["status"] = "supported"
    assert diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
        hypothesis_state=state,
    ) == []


def test_diagnostic_contract_is_independent_of_baseline_rule_status():
    payload = _payload()
    # The fixture's only baseline diagnosis is first_degree_av_delay. A
    # diagnosis-first model can nevertheless conclude sinus rhythm.
    problems = diagnostic_prompts.validate_verdict(
        _diagnostic_verdict(),
        evidence_document=payload,
    )
    assert problems == []


def test_diagnostic_contract_rejects_unknown_code_and_requires_review():
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["code"] = "model_invented_code"
    verdict["human_review"]["required"] = False
    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )
    assert any("not in the diagnostic vocabulary" in problem for problem in problems)
    assert any("must be true" in problem for problem in problems)


def test_diagnostic_contract_requires_component_wave_review_for_limited_pr():
    payload = _payload()
    payload["global_features"]["pr_ms"] = None
    payload.setdefault("rhythm_inputs", {}).setdefault("record", {}).setdefault(
        "availability", {}
    )["pr_available"] = False
    verdict = _diagnostic_verdict()

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=payload,
    )

    assert any(
        "missing interval_measurement_contexts entry for limited `PR`" in problem
        for problem in problems
    )


def test_diagnostic_contract_accepts_residual_p_evidence_when_pr_is_unavailable():
    payload = _payload()
    payload["global_features"]["pr_ms"] = None
    payload.setdefault("rhythm_inputs", {}).setdefault("record", {}).setdefault(
        "availability", {}
    )["pr_available"] = False
    verdict = _diagnostic_verdict()
    verdict["interval_measurement_contexts"].append(
        {
            "interval": "PR",
            "status": "unavailable",
                "interval_conclusion": "The PR interval cannot be reported reliably.",
            "component_waveform_assessment": (
                    "P waves remain visible, but boundary stability and P-QRS association require separate assessment."
            ),
            "residual_evidence": [
                {
                        "claim": "PR is unavailable, but lead-level P-wave amplitude measurements remain available.",
                    "value": None,
                    "unit": None,
                    "citations": [
                        "ev:/global_features/pr_ms",
                        "ev:/representative_leads/I/params/p_amp_mv",
                    ],
                }
            ],
            "interpretive_impact": (
                    "Missing PR cannot be interpreted as absent P waves; atrial activity still requires analysis."
            ),
                "what_would_resolve_it": "Review P-wave boundaries and beat-level QRS association.",
        }
    )

    assert diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=payload,
    ) == []


def test_interval_context_rejects_interval_status_without_component_wave_evidence():
    verdict = _diagnostic_verdict()
    verdict["interval_measurement_contexts"][0]["residual_evidence"] = [
        {
            "claim": "QT不可报告。",
            "value": None,
            "unit": None,
            "citations": ["ev:/global_features/qt_reportable"],
        }
    ]

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any(
        "does not cite any T/U component-wave observation" in problem
        for problem in problems
    )


def test_diagnostic_normalizer_preserves_grounded_and_masks_only_ungrounded_numbers():
    verdict = _diagnostic_verdict()
    verdict["summary"] = (
        "Sinus rhythm; see evidence for exact measurements. QTc is 408 ms. Human review remains required."
    )
    verdict["diagnoses"][0]["reasoning"] = (
        "A heart rate of 62 bpm supports the diagnosis. Rhythm organization also supports a sinus mechanism."
    )
    original_evidence = json.loads(
        json.dumps(verdict["diagnoses"][0]["evidence"])
    )

    normalized, paths = diagnostic_prompts.normalize_verdict(verdict)

    assert normalized["diagnoses"][0]["evidence"] == original_evidence
    assert "QTc is corresponding measured value" in normalized["summary"]
    assert "62 bpm" in normalized["diagnoses"][0]["reasoning"]
    assert paths == ["summary"]
    problems = diagnostic_prompts.validate_verdict(
        normalized,
        evidence_document=_payload(),
    )
    assert not any("`summary` contains ungrounded" in problem for problem in problems)
    assert not any("diagnoses[0].reasoning contains ungrounded" in problem for problem in problems)


def test_diagnostic_normalizer_removes_padded_duplicate_complete_alternative():
    verdict = _diagnostic_verdict()
    verdict["ranked_complete_interpretations"].append(
        {
            **verdict["ranked_complete_interpretations"][0],
            "rank": 2,
            "interpretation_type": "ALTERNATIVE",
        }
    )

    normalized, paths = diagnostic_prompts.normalize_verdict(verdict)

    assert len(normalized["ranked_complete_interpretations"]) == 1
    assert "ranked_complete_interpretations[1]" in paths


def test_diagnostic_contract_allows_grounded_numbers_in_summary_and_reasoning():
    verdict = _diagnostic_verdict()
    verdict["summary"] = "Sinus rhythm with a heart rate of 62 bpm."
    verdict["diagnoses"][0]["statement"] = "Sinus rhythm at 62 bpm."
    verdict["diagnoses"][0]["reasoning"] = "A regular rhythm and rate of 62 bpm support a sinus mechanism."

    assert diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    ) == []


def test_diagnostic_contract_allows_reference_range_in_reasoning():
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["reasoning"] = (
        "The measured heart rate is within the common reference range of 60-100 bpm, and the rhythm is regular."
    )

    assert diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    ) == []


def test_diagnostic_contract_allows_single_guideline_cutoff_in_reasoning():
    verdict = _diagnostic_verdict()
    diagnosis = verdict["diagnoses"][0]
    diagnosis["reasoning"] = (
        "The patient's heart rate is 62 bpm, below the common tachycardia threshold of 100 bpm, "
        "and the regular rhythm supports a sinus mechanism."
    )

    assert diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    ) == []


def test_diagnostic_contract_rejects_code_category_and_qt_direction_conflicts():
    verdict = _diagnostic_verdict()
    diagnosis = verdict["diagnoses"][0]
    diagnosis.update(
        {
            "code": "borderline_short_qt",
            "category": "quality",
            "statement": "QTc 间期轻度延长。",
            "reasoning": "现有证据支持 QTc 延长。",
        }
    )

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("expected `interval`" in problem for problem in problems)
    assert any("conflicts with narrative describing QT/QTc prolongation" in problem for problem in problems)


def test_low_confidence_hypothesis_must_be_a_differential_not_positive_diagnosis():
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["confidence"] = "LOW"

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("LOW-confidence hypotheses belong" in problem for problem in problems)


def _candidate_evidence_payload() -> dict:
    payload = _payload()
    payload["rhythm_inputs"] = {
        "background": {
            "background_atrial_rate_bpm": 83.0,
            "background_ventricular_rate_bpm": 83.0,
        },
        "af_afl": {
            "F_wave_confidence": 0.64,
            "F_wave_multilead_consensus": True,
            "qrst_subtraction_quality": {
                "organized_spectral_power_ratio": 0.30,
            },
        },
        "preexcitation": {
            "short_pr_interval": False,
            "short_pr_segment": False,
            "delta_lead_count": 6,
        },
    }
    payload["p_wave_assessments"] = [
        {
            "beat_id": index,
            "accepted": True,
            "ta_ambiguous": True,
            "morphology_cluster_id": index,
        }
        for index in range(4)
    ]
    return payload


def _verdict_for(code: str, category: str) -> dict:
    verdict = _diagnostic_verdict()
    diagnosis = verdict["diagnoses"][0]
    diagnosis.update(
        {
            "code": code,
            "statement": "候选诊断。",
            "category": category,
            "reasoning": "根据已读取证据形成候选诊断。",
        }
    )
    verdict["ranked_complete_interpretations"][0]["basis_codes"] = [code]
    return verdict


def test_contract_blocks_delta_candidate_promotion_without_pr_corroboration():
    problems = diagnostic_prompts.validate_verdict(
        _verdict_for("ventricular_preexcitation_pattern", "conduction"),
        evidence_document=_candidate_evidence_payload(),
    )

    assert any("candidate-only evidence family" in problem for problem in problems)


def test_contract_blocks_flutter_candidate_when_rates_match_and_p_is_ambiguous():
    problems = diagnostic_prompts.validate_verdict(
        _verdict_for("atrial_flutter_pattern", "rhythm"),
        evidence_document=_candidate_evidence_payload(),
    )

    assert any("same-chain flutter candidate" in problem for problem in problems)


def test_compact_promotion_gate_keeps_weak_flutter_candidate_unconfirmed():
    payload = _candidate_evidence_payload()
    payload["rhythm_inputs"]["af_afl"]["qrst_subtraction_quality"][
        "organized_spectral_power_ratio"
    ] = 0.10
    diagnosis = _verdict_for("atrial_flutter_pattern", "rhythm")["diagnoses"][0]

    problems = diagnostic_prompts.candidate_evidence_hierarchy_problems(
        [diagnosis],
        payload,
    )

    assert any("residual signal lacks corroborating organization" in item for item in problems)


def test_contract_excludes_ambiguous_p_clusters_from_multifocal_support():
    payload = _candidate_evidence_payload()
    problems = diagnostic_prompts.validate_verdict(
        _verdict_for("multifocal_atrial_rhythm", "rhythm"),
        evidence_document=payload,
    )
    assert any("fewer than three morphology clusters" in problem for problem in problems)

    for row in payload["p_wave_assessments"][:3]:
        row["ta_ambiguous"] = False
    problems = diagnostic_prompts.validate_verdict(
        _verdict_for("multifocal_atrial_rhythm", "rhythm"),
        evidence_document=payload,
    )
    assert not any("morphology clusters" in problem for problem in problems)


def test_contract_requires_clean_p_support_for_ectopic_atrial_primary_diagnosis():
    problems = diagnostic_prompts.validate_verdict(
        _verdict_for("ectopic_atrial_rhythm_pattern", "rhythm"),
        evidence_document=_candidate_evidence_payload(),
    )

    assert any("accepted non-T/A-ambiguous" in problem for problem in problems)


def test_contract_blocks_atrial_abnormality_when_p_boundaries_are_ambiguous():
    problems = diagnostic_prompts.validate_verdict(
        _verdict_for("left_atrial_abnormality", "chamber"),
        evidence_document=_candidate_evidence_payload(),
    )

    assert any(
        "P-morphology atrial-abnormality" in problem for problem in problems
    )


def test_contract_never_treats_ta_ambiguity_as_p_wave_abnormality_or_clear_p():
    payload = _candidate_evidence_payload()
    verdict = _verdict_for("p_wave_abnormality", "rhythm")
    verdict["diagnoses"][0]["evidence"][0] = {
        "claim": "该心搏显示清晰且异常的 P 波。",
        "value": None,
        "unit": None,
        "citations": ["ev:/p_wave_assessments/0/ta_ambiguous"],
    }

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=payload,
    )

    assert any("describes a T/A-ambiguous P assessment as clear" in item for item in problems)
    assert any("uses T/A ambiguity as positive P-wave morphology" in item for item in problems)


def test_contract_rejects_derived_or_second_quantity_inside_one_evidence_item():
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["evidence"][0]["claim"] = (
        "测得心率为 62 bpm，同时 QRS 为 92 ms。"
    )

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any(
        "evidence[0].claim contains ungrounded" in problem and "92 ms" in problem
        for problem in problems
    )


def test_contract_rejects_compound_numeric_evidence_even_when_cited():
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["evidence"][0] = {
        "claim": "T轴为 -125 deg，QRS轴为 -15 deg。",
        "value": -125,
        "unit": "deg",
        "citations": [
            "ev:/global_features/t_axis_deg",
            "ev:/global_features/qrs_axis_deg",
        ],
    }

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("multiple patient measurements" in problem for problem in problems)
    assert any("must cite exactly one pointer" in problem for problem in problems)


def test_contract_rejects_qtc_formula_label_swap():
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["evidence"][0] = {
        "claim": "Fridericia QTc 为 407 ms。",
        "value": 407,
        "unit": "ms",
        "citations": ["ev:/global_features/qtc_bazett_ms"],
    }

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any(
        "labels a Bazett pointer as Fridericia" in problem for problem in problems
    )


def test_contract_rejects_lead_label_pointer_swap():
    verdict = _diagnostic_verdict()
    verdict["interval_measurement_contexts"][0]["residual_evidence"][0] = {
        "claim": "II导联T波振幅为0.3 mV。",
        "value": 0.3,
        "unit": "mV",
        "citations": ["ev:/representative_leads/V2/params/t_amp_mv"],
    }

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("pointer belongs to lead(s) ['V2']" in problem for problem in problems)


def test_contract_blocks_positive_summary_when_diagnoses_is_empty():
    verdict = _diagnostic_verdict()
    verdict["diagnoses"] = []

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("summary` must explicitly state" in problem for problem in problems)
    assert any("unconfirmed/indeterminate" in problem for problem in problems)
    assert any("unconfirmed sinus_rhythm" in problem for problem in problems)

    verdict["summary"] = "本次心电图未确认明确的阳性诊断。"
    verdict["ranked_complete_interpretations"][0].update(
        {
            "complete_diagnosis": "未确认明确诊断；仅保留受限观察",
            "basis_codes": [],
        }
    )
    repaired = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )
    assert not any("summary` must explicitly state" in item for item in repaired)
    assert not any("unconfirmed/indeterminate" in item for item in repaired)


def test_contract_rejects_normal_ecg_with_positive_abnormal_diagnosis():
    verdict = _verdict_for("normal_ecg", "other")
    abnormal = json.loads(json.dumps(verdict["diagnoses"][0]))
    abnormal.update(
        {
            "code": "t_wave_abnormality",
            "category": "ischemia_repolarization",
            "statement": "非特异性 ST-T 异常。",
            "reasoning": "现有测量提示非特异性复极异常。",
        }
    )
    verdict["diagnoses"].append(abnormal)
    verdict["ranked_complete_interpretations"][0]["basis_codes"].append(
        "t_wave_abnormality"
    )

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("combines `normal_ecg`" in problem for problem in problems)


def test_complete_interpretation_must_integrate_all_positive_codes():
    verdict = _diagnostic_verdict()
    verdict["ranked_complete_interpretations"][0]["basis_codes"] = []

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any(
        "must integrate exactly every positive diagnosis code" in problem
        for problem in problems
    )


def test_complete_interpretation_duplicate_basis_codes_are_still_blocked():
    verdict = _diagnostic_verdict()
    verdict["ranked_complete_interpretations"][0]["basis_codes"] = [
        "sinus_rhythm",
        "sinus_rhythm",
    ]

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("basis_codes contains duplicates" in problem for problem in problems)


def test_complete_alternative_requires_a_registered_differential_mechanism():
    verdict = _diagnostic_verdict()
    verdict["ranked_complete_interpretations"].append(
        {
            "rank": 2,
            "interpretation_type": "ALTERNATIVE",
            "complete_diagnosis": "另一套完整但没有鉴别依据的解释。",
            "confidence": "LOW",
            "basis_codes": ["sinus_rhythm"],
            "key_uncertainty": "没有独立鉴别机制支持。",
        }
    )

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=_payload(),
    )

    assert any("is padded despite there being no differential" in problem for problem in problems)


def test_differential_evidence_is_deterministically_verified():
    store = EvidenceStore.from_dict(_payload(), record_id="DIFF")
    verdict = _diagnostic_verdict()
    verdict["differential_diagnoses"] = [
        {
            "code": "sinus_tachycardia",
            "statement": "Sinus tachycardia",
            "confidence": "LOW",
            "supporting_evidence": [
                {
                    "claim": "The measured heart rate is 999 bpm.",
                    "value": 999,
                    "unit": "bpm",
                    "citations": ["ev:/global_features/heart_rate_bpm"],
                }
            ],
            "counterevidence": [],
            "what_would_resolve_it": "Confirm the rhythm against the source waveform",
        }
    ]
    report = verify_structured(
        verdict,
        store,
        whitelist=frozenset({"/global_features/heart_rate_bpm"}),
        policy=VerificationPolicy.for_structured(),
    )
    assert not report.passed
    assert any(finding.code == "value_mismatch" for finding in report.findings)


def test_diagnostic_agent_uses_measurement_tools_and_outputs_independent_diagnosis():
    store = EvidenceStore.from_dict(_payload(), record_id="DX001")
    verdict = _diagnostic_verdict()
    survey, hypothesize, investigate, challenge = legacy_phase_patches()
    backend = ScriptedBackend(
        script=[
            tool_response(
                ("get_diagnostic_overview", {}),
            ),
                text_response(json.dumps(survey, ensure_ascii=False)),
                text_response(json.dumps(hypothesize, ensure_ascii=False)),
                tool_response(
                    (
                        "get_measurement",
                        {"pointer": HEART_RATE},
                    )
                ),
                tool_response(
                    ("get_interval_waveform_context", {"interval": "qt"}),
                ),
                text_response(json.dumps(investigate, ensure_ascii=False)),
                tool_response(
                    (
                        "get_measurement",
                        {"pointer": T_AMP},
                    )
                ),
                tool_response(
                    ("get_beat_table", {"fields": ["rr_prev_ms", "rr_next_ms"]}),
                ),
                text_response(json.dumps(challenge, ensure_ascii=False)),
            text_response(json.dumps(verdict)),
        ]
    )

    result = ECGDiagnosticAgent(workflow="legacy", store=store, backend=backend).run()

    assert result.ok and result.verified, result.summary()
    assert result.verdict["diagnoses"][0]["code"] == "sinus_rhythm"
    assert result.audit["agent_protocol"] == DIAGNOSTIC_AGENT_PROTOCOL_VERSION
    assert result.audit["mode"] == "diagnose"
    assert [phase.key for phase in result.phases] == [
        "survey",
        "hypothesize",
        "investigate",
        "challenge",
        "synthesize",
    ]
    assert result.knowledge_navigation["patient_evidence"] is False
    assert all(
        "list_findings" not in call["tool_names"]
        and "get_rule_detail" not in call["tool_names"]
        for call in backend.calls
    )
    briefing = backend.calls[0]
    assert briefing["n_tools"] > 0
    trace = result.trace_report
    assert "# ECG Agent Execution Trace" in trace
    assert "## Phase Timeline" in trace
    assert "## Model-Visible Initial Input and Interphase Context" in trace
    assert "Neutral ECG intake" in trace
    assert "INTERIM MEDICAL KNOWLEDGE NAVIGATION" in trace
    assert "`hypothesize`" in trace
    assert "## Knowledge Navigation After Survey" in trace
    assert "## ecgfeat Tool-Call Trace" in trace
    assert "Complete model-visible result" in trace
    assert "heart_rate_bpm" in trace
    assert "## Deterministic Validation and Revision" in trace
    # Full returns belong only to the Markdown trace; diagnosis JSON remains compact.
    assert "result_text" not in result.to_dict()["audit"]["tools"]["calls"][0]
