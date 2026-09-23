"""Disease-pathway contracts, including placement under incomplete evidence.

The placement tests supply controlled model judgements over real tool renders.
They test gating/visibility, not model accuracy or clinical sensitivity.
"""
from __future__ import annotations

import copy
import inspect
import json

import jsonschema
import pytest

from ecgagent.agent import diagnostic_pathways as pathways
from ecgagent.agent.compact_diagnostic_prompts import COMPACT_DIRECT_TOOLS
from ecgagent.agent.diagnosis_catalog import DIAGNOSIS_CATALOG
from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.backends.mock import ScriptedBackend
from ecgagent.evidence.store import EvidenceStore
from ecgagent.tools.modalities import _MORPHOLOGY_PROFILES
from ecgagent.tools.registry import build_default_registry
from tests.test_ecgagent_tools import _payload


def _path(code):
    return pathways.build_diagnostic_pathway(code)


def _rich_payload():
    """Synthetic visible atoms for contract tests, not a clinical test record."""
    payload = _payload()
    payload.pop("clinical_interpretation", None)
    payload["metadata"]["diagnostic_gate"].update(state="pass", partial_reasons=[])
    payload["metadata"]["input_contract"] = {
        "amplitude_calibration": {"per_lead_normalized": False, "units": "mV"}
    }
    fields = {field for names in _MORPHOLOGY_PROFILES.values() for field in names}
    for code in DIAGNOSIS_CATALOG:
        for node in _path(code)["steps"]:
            if node["tool"] == "get_lead_table":
                fields.update(node["arguments"]["fields"])
    for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"):
        params = dict.fromkeys(fields, 0.1)
        params.update(qrs_ms=96.0, q_duration_ms=40.0, q_amp_mv=-0.1,
                      r_amp_mv=1.0, s_amp_mv=-0.3, reliable_for_qrs=True,
                      st_hybrid_reliable=True, st_hybrid_unreliable_reason="",
                      st_hybrid_baseline_confidence=0.95, p_amp_mv=0.1)
        payload["representative_leads"][lead] = {"lead": lead, "params": params}
        payload["quality"][lead] = {"grade": "Q0", "reliable_for_qrs": True,
                                     "reliable_for_p": True, "reliable_for_t": True,
                                     "reliable_for_qt": True}
    payload["groups"] = {"1": {
        "member_count": 2, "member_pct": 100.0, "longest_run": 2,
        "mean_qrs_ms": 96.0, "mean_rr_ms": 800.0,
        "flags": {"dominant_group": True, "wide_qrs": False},
    }}
    payload["p_wave_assessments"] = [{
        "beat_id": i, "accepted": True, "ta_ambiguous": False,
        "onset_confidence": 0.95, "offset_confidence": 0.95,
        "valid_leads": ["II", "V1"], "reject_reasons": [],
        "morphology_cluster_id": 1,
    } for i in range(2)]
    payload["rhythm_inputs"] = {
        "record": {"availability": {"pr_available": True, "atrial_rhythm_available": True}},
        "background": {"background_rr_regular": True, "background_ventricular_rate_bpm": 75.0},
        "af_afl": {"rr_cv": 0.01, "f_wave_confidence": 0.0},
        "av_block": {"evidence": {"localized_atrial_event_excess": False}},
        "pacing": {"state": "off", "ventricular_pacing_present": False},
        "p_events": [{"p_event_id": i, "time_ms": 100.0 + i * 800,
                      "association_type": "conducted", "confidence": 0.95,
                      "source_leads": ["II", "V1"], "associated_qrs_beat_id": i,
                      "pr_ms": 214.0} for i in range(2)],
    }
    return payload


def _placement(monkeypatch, code, statuses=None, *, omit=(), review_conflict=None):
    """Use the production compiler, tool ledger and placement implementation."""
    agent = ECGDiagnosticAgent(
        store=EvidenceStore.from_dict(_rich_payload()), backend=ScriptedBackend(script=[])
    )
    path = _path(code)
    if review_conflict is not None:
        path = pathways.attach_waveform_review(path, "st", conflict=review_conflict)
    candidate = {"id": "h1", "code": code, "domains": [], "checks": [],
                 "diagnostic_pathway": path}
    agent._compact_plan = {"candidates": [candidate], "quality_limitations": []}
    # The numerical resolvers have their own adversarial tests. Here we isolate
    # whether required model interpretation and applicability affect placement.
    monkeypatch.setattr(agent, "_compact_deterministic_pathway_step", lambda **kwargs: None)
    agent.registry.begin_phase("adjudicate", 20)
    evidence_by_step = {}
    results = {}
    aliases = {}
    monkeypatch.setattr(agent.backend, "citation_aliases", lambda: aliases, raising=False)
    for tool, args, step_id in agent._compact_candidate_pathway_calls(candidate):
        signature = (tool, json.dumps(args, sort_keys=True))
        if signature not in results:
            results[signature] = agent.registry.call(tool, args)
        result = results[signature]
        if tool == "get_waveform_review" and not result.ok:
            assert result.note == "measurement_unavailable"
            evidence_by_step[step_id] = None
            continue
        assert result.ok, (code, step_id, result.text)
        agent.registry.authorize_model_visible(tool=tool, arguments=args,
                                               citations=result.citations, source="pathway_contract")
        pointer = next(p for p in result.citations
                       if agent.store.try_resolve(p) is not None
                       and not agent.store.try_resolve(p).is_null)
        alias = f"Q{len(aliases) + 1}"
        aliases[alias] = pointer
        evidence_by_step[step_id] = alias
    rows = [{"id": node["id"], "status": (statuses or {}).get(node["id"], "pass"),
             "evidence": ([{"citation": evidence_by_step[node["id"]],
                            "claim": "Controlled visible observation for placement contract"}]
                          if evidence_by_step[node["id"]] else [])}
            for node in path["steps"] if node["id"] not in omit]
    verdict = agent._expand_compact_verdict({"decisions": [{"id": "h1", "code": code,
                                                           "pathway_steps": rows}],
                                              "interval_contexts": []})
    return verdict, agent._compact_decision_audit["decisions"][0]


def test_all_catalog_codes_have_explicit_bounded_executable_paths():
    registry = build_default_registry(EvidenceStore.from_dict(_payload()))
    for code in DIAGNOSIS_CATALOG:
        path = _path(code)
        nodes = path["steps"]
        assert 1 <= len(nodes) <= 6, code
        assert len({node["id"] for node in nodes}) == len(nodes), code
        assert not {"direct_measurement_support", "discriminative_countercheck"} & {
            node["id"] for node in nodes
        }, code
        assert any(node["gate"] == "required" for node in nodes), code
        for node in nodes:
            assert node["tool"] in COMPACT_DIRECT_TOOLS, (code, node["tool"])
            spec = registry.specs[node["tool"]]
            jsonschema.validate(node["arguments"], spec.input_schema())
            inspect.signature(spec.handler).bind(registry.store, **node["arguments"])


@pytest.mark.parametrize("code", ["first_degree_av_block", "first_degree_av_delay",
                                 "possible_first_degree_av_delay", "complete_av_block",
                                 "second_degree_av_block"])
def test_av_identity_is_visible_model_work_and_required(monkeypatch, code):
    node = next(n for n in _path(code)["steps"] if n["id"] == "atrial_wave_identity")
    assert node["owner"] == "model" and node["gate"] == "required"
    verdict, decision = _placement(monkeypatch, code, omit={node["id"]})
    assert decision["final_placement"] == "unresolved"
    assert not verdict["diagnoses"]


def test_av_event_view_preserves_identity_quality_and_timing():
    for code in ("first_degree_av_delay", "second_degree_av_block", "complete_av_block"):
        event_nodes = [n for n in _path(code)["steps"] if n["tool"] == "get_atrial_event_table"]
        assert event_nodes
        for node in event_nodes:
            assert {"p_event_id", "time_ms", "association_type", "confidence", "source_leads",
                    "associated_qrs_beat_id", "pr_ms"} <= set(node["arguments"]["fields"])


@pytest.mark.parametrize("status", ["unknown", "fail"])
def test_lvh_ineligible_or_unknown_activation_does_not_reject_anatomic_lvh(monkeypatch, status):
    verdict, decision = _placement(monkeypatch, "lvh_voltage_criteria",
                                   {"voltage_criteria_qrs_valid": status})
    assert decision["final_placement"] == "unresolved"
    assert not verdict["diagnoses"]
    assert verdict["quality_assessment"]["limitations"]


def test_applicable_lvh_path_can_complete(monkeypatch):
    verdict, decision = _placement(monkeypatch, "lvh_voltage_criteria")
    assert decision["final_placement"] == "confirmed"
    assert [row["code"] for row in verdict["diagnoses"]] == ["lvh_voltage_criteria"]


@pytest.mark.parametrize("code, expected", [("wide_complex_tachycardia", "confirmed"),
                                           ("ventricular_rhythm", "unresolved")])
def test_unknown_origin_preserves_wide_complex_phenotype(monkeypatch, code, expected):
    _, decision = _placement(monkeypatch, code,
                             {"ventricular_morphology": "unknown", "atrial_relation": "unknown"})
    assert decision["final_placement"] == expected


@pytest.mark.parametrize("code, expected", [("st_elevation", "confirmed"),
                                           ("st_depression", "confirmed"),
                                           ("acute_occlusion_pattern", "unresolved")])
def test_secondary_activation_does_not_erase_st_phenotype(monkeypatch, code, expected):
    _, decision = _placement(monkeypatch, code, {"st_pattern_applicability": "fail"})
    assert decision["final_placement"] == expected


@pytest.mark.parametrize("status", ["pass", "fail"])
def test_null_q_duration_cannot_confirm_or_refute_a_duration_criterion(monkeypatch, status):
    payload = _payload()
    payload.pop("clinical_interpretation", None)
    agent = ECGDiagnosticAgent(store=EvidenceStore.from_dict(payload), backend=ScriptedBackend(script=[]))
    code = "pathological_q_waves"
    path = _path(code)
    agent._compact_plan = {"candidates": [{"id": "h1", "code": code,
                                           "diagnostic_pathway": path}], "quality_limitations": []}
    node = next(n for n in path["steps"] if n["id"] == "q_wave_morphology")
    pointer = "/representative_leads/I/params/q_duration_ms"
    monkeypatch.setattr(agent.backend, "citation_aliases", lambda: {"Q1": pointer}, raising=False)
    agent.registry.begin_phase("adjudicate", 2)
    agent.registry.call(node["tool"], node["arguments"])
    agent.registry.authorize_model_visible(tool=node["tool"], arguments=node["arguments"],
                                           citations=[pointer], source="missing_duration")
    agent._expand_compact_verdict({"decisions": [{"id": "h1", "code": code, "pathway_steps": [
        {"id": node["id"], "status": status,
         "evidence": [{"citation": "Q1", "claim": "Q duration unavailable"}]}]}]})
    decision = agent._compact_decision_audit["decisions"][0]
    audit = next(n for n in decision["pathway_steps"] if n["id"] == node["id"])
    assert audit["effective_status"] == "unknown"
    assert decision["final_placement"] == "unresolved"


def test_af_regular_response_is_supporting_not_a_mechanism_veto(monkeypatch):
    _, decision = _placement(monkeypatch, "atrial_fibrillation", {"irregular_ventricular_response": "unknown"})
    assert decision["final_placement"] == "confirmed"
    _, decision = _placement(monkeypatch, "atrial_fibrillation", {"af_atrial_mechanism": "unknown"})
    assert decision["final_placement"] == "unresolved"
    assert pathways.semantic_candidate_family("atrial_fibrillation_flutter_indeterminate") != (
        pathways.semantic_candidate_family("atrial_fibrillation")
    )


@pytest.mark.parametrize("code", ["premature_atrial_complexes", "premature_ventricular_complexes",
                                 "probable_premature_ventricular_complexes"])
def test_prematurity_without_origin_is_unresolved(monkeypatch, code):
    _, decision = _placement(monkeypatch, code, {"ectopic_origin": "unknown"})
    assert decision["final_placement"] == "unresolved"
    assert "representative_event" not in {node["id"] for node in _path(code)["steps"]}


@pytest.mark.parametrize("code", ["pacing_failure_to_capture_suspected", "pacing_sensing_failure_suspected"])
def test_pacing_failure_requires_its_own_event_definition(monkeypatch, code):
    path = _path(code)
    assert not any(n["id"] == "capture_relation_support" and n["gate"] == "required"
                   for n in path["steps"])
    node = next(n for n in path["steps"] if n["id"] == "pacing_pattern_definition")
    assert node["owner"] == "model"
    _, decision = _placement(monkeypatch, code, {node["id"]: "unknown"})
    assert decision["final_placement"] == "unresolved"


def test_voltage_and_progression_views_include_qrs_reliability():
    for code in ("low_voltage_limb_leads", "low_voltage_precordial_leads", "rvh_pattern",
                 "poor_r_wave_progression", "lvh_voltage_criteria"):
        tables = [n for n in _path(code)["steps"] if n["tool"] == "get_lead_table"]
        assert tables
        assert all("reliable_for_qrs" in n["arguments"]["fields"] for n in tables)
    low = next(n for n in _path("low_voltage_limb_leads")["steps"] if n["tool"] == "get_lead_table")
    assert "q_amp_mv" in low["arguments"]["fields"]


def test_record_extension_preserves_exact_views_and_does_not_mutate_plan():
    checks = [{"tool": "get_lead_table", "arguments": {"fields": ["t_amp_mv"], "leads": ["V2"]}},
              {"tool": "get_native_beat_profile", "arguments": {"profile": "repolarization", "max_beats": 2}}]
    original = copy.deepcopy(checks)
    path = pathways.build_diagnostic_pathway("record_local_repolarization", fallback_checks=checks)
    assert [n["arguments"] for n in path["steps"]] == [n["arguments"] for n in checks]
    path["steps"][0]["arguments"]["fields"].append("r_amp_mv")
    assert checks == original
    assert all(n["owner"] == "model" and n["gate"] == "required" for n in path["steps"])


def test_nonadult_paths_cannot_inherit_adult_program_thresholds():
    for code in DIAGNOSIS_CATALOG:
        assert all(n["owner"] == "model" for n in pathways.build_diagnostic_pathway(
            code, program_owned=False)["steps"]), code


def test_catalog_growth_is_visible_in_the_audit():
    from pathlib import Path
    audit = (Path(__file__).resolve().parents[1] / "docs/ecgagent_v44_pathway_audit.md").read_text()
    for code in DIAGNOSIS_CATALOG:
        assert f"`{code}`" in audit, code


def test_waveform_attachment_is_copying_idempotent_and_never_drops_definition_nodes():
    base = _path("normal_ecg")
    original = copy.deepcopy(base)
    attached = pathways.attach_waveform_review(base, "p_av")
    assert base == original
    assert attached["steps"][:-1] == original["steps"]
    assert len(attached["steps"]) == 7
    assert pathways.attach_waveform_review(attached, "p_av") == attached
    node = attached["steps"][-1]
    assert node["id"] == "raw_measurement_consistency"
    assert node["owner"] == "model" and node["gate"] == "invalidator"
    assert node["unknown_blocks_confirmation"] and node["failure_means_not_applicable"]
    attached["steps"][0]["arguments"]["changed_in_test"] = True
    assert base == original


def test_waveform_attachment_cannot_downgrade_a_known_conflict():
    base = _path("pathological_q_waves")
    observation = pathways.attach_waveform_review(base, "qrs", conflict=False)
    node = observation["steps"][-1]
    assert node["gate"] == "supporting"
    assert not node.get("unknown_blocks_confirmation")
    assert not node.get("failure_means_not_applicable")
    strict = pathways.attach_waveform_review(observation, "qrs")
    merged = pathways.attach_waveform_review(strict, "st", conflict=False)
    assert len(merged["steps"]) == len(base["steps"]) + 1
    assert merged["steps"][-1]["arguments"] == {"profile": "all"}
    assert merged["steps"][-1]["unknown_blocks_confirmation"]


@pytest.mark.parametrize("conflict, expected", [(False, "confirmed"), (True, "unresolved")])
def test_missing_raw_review_blocks_only_an_already_triggered_conflict(monkeypatch, conflict, expected):
    _, decision = _placement(monkeypatch, "st_elevation", review_conflict=conflict)
    assert decision["final_placement"] == expected
    node = next(n for n in decision["pathway_steps"] if n["id"] == "raw_measurement_consistency")
    assert node["effective_status"] == "unknown"


def test_supporting_raw_review_cannot_replace_missing_definition_evidence(monkeypatch):
    _, decision = _placement(monkeypatch, "st_elevation", {"st_pattern_definition": "unknown"},
                             review_conflict=False)
    assert decision["final_placement"] == "unresolved"


def test_waveform_profile_validation_does_not_modify_the_input():
    base = _path("first_degree_av_delay")
    original = copy.deepcopy(base)
    with pytest.raises(ValueError, match="profile"):
        pathways.attach_waveform_review(base, "invented_profile")
    with pytest.raises(TypeError, match="boolean"):
        pathways.attach_waveform_review(base, "p_av", conflict="false")
    assert base == original
