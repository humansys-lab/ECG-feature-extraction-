from __future__ import annotations

import json

from ecgagent.agent.tool_planner import build_tool_plan
from ecgagent.acceptance import AcceptanceThresholds, _clinical_gate
from ecgagent.evaluation_protocol import evaluate_arms
from ecgagent.evidence.diagnostic_contract import (
    DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION,
    build_diagnostic_document,
    unclassified_source_fields,
)
from ecgagent.evidence.ledger import visible_citations
from ecgagent.evidence.store import EvidenceStore
from ecgagent.knowledge.safety import sanitize_runtime_excerpt
from ecgagent.tools.registry import build_default_registry


def test_only_rendered_value_citations_enter_the_session_whitelist():
    store = EvidenceStore.from_dict(
        {"global_features": {"heart_rate_bpm": 72.0}},
        record_id="visible-ledger",
    )
    registry = build_default_registry(
        store,
        budget=None,
        include=("get_measurement",),
    )
    result = registry.call(
        "get_measurement",
        {"pointer": "/global_features/heart_rate_bpm"},
    )
    assert not registry.whitelist
    shown = visible_citations(result.render(), result.citations)
    registry.authorize_model_visible(
        tool="get_measurement",
        arguments={"pointer": "/global_features/heart_rate_bpm"},
        citations=shown,
        source="test:model-turn",
    )
    assert registry.whitelist == frozenset({"/global_features/heart_rate_bpm"})


def test_deterministic_program_evidence_is_authorized_without_model_visibility():
    store = EvidenceStore.from_dict(
        {"global_features": {"heart_rate_bpm": 72.0}},
        record_id="program-ledger",
    )
    registry = build_default_registry(
        store,
        budget=None,
        include=("get_measurement",),
    )
    registry.begin_phase("adjudicate", budget=None)
    result = registry.call(
        "get_measurement",
        {"pointer": "/global_features/heart_rate_bpm"},
    )

    accepted = registry.authorize_program_evidence(
        tool="get_measurement",
        citations=result.citations,
        source="deterministic_pathway:test",
    )

    assert accepted == ("/global_features/heart_rate_bpm",)
    assert registry.whitelist == frozenset({"/global_features/heart_rate_bpm"})
    assert registry.model_visible_whitelist == frozenset()
    provenance = registry.evidence_provenance(
        "/global_features/heart_rate_bpm"
    )
    assert provenance["program_authorized"] is True
    assert provenance["model_visible"] is False


def test_model_evidence_view_materializes_one_atomic_value_source():
    store = EvidenceStore.from_dict(
        {"global_features": {"heart_rate_bpm": 72.0}},
        record_id="atomic-view",
    )
    registry = build_default_registry(
        store,
        budget=None,
        include=("get_measurement",),
    )
    result = registry.call(
        "get_measurement",
        {"pointer": "/global_features/heart_rate_bpm"},
    )

    view = registry.model_evidence_view(
        result,
        tool="get_measurement",
        arguments={"pointer": "/global_features/heart_rate_bpm"},
    )

    assert view is not None
    assert view.citations == ("/global_features/heart_rate_bpm",)
    assert '"value":72' in view.text
    assert '"unit":"bpm"' in view.text
    assert "ev:/global_features/heart_rate_bpm" in view.text


def test_citation_visibility_is_atomic_and_arguments_do_not_authorize_aliases():
    assert not visible_citations(
        "[ev:/atrial_rate]",
        ("/a", "/atrial"),
    )
    assert not visible_citations(
        {"arguments": {"pointer": "ev:/global_features/heart_rate_bpm"}},
        ("/global_features/heart_rate_bpm",),
        aliases={"E1": "/global_features/heart_rate_bpm"},
        allow_exact=False,
    )


def test_diagnosis_contract_drops_unknown_and_interpretive_families():
    source = {
        "global_features": {"heart_rate_bpm": 72.0},
        "clinical_interpretation": {"diagnosis": "hidden"},
        "future_vendor_diagnosis": {"label": "must be classified first"},
        "metadata": {
            "input_fs": 500,
            "clinical_interpretation": {"diagnosis": "hidden"},
            "patient_meta": {"age": 60, "clinical_classifications": ["AF"]},
        },
        "rhythm_inputs": {
            "record": {
                "availability": {
                    "atrial_rhythm_available": False,
                    "reasons": ["probable_flutter"],
                }
            },
            "background": {"background_ventricular_rate_bpm": 72.0},
            "statement_evidence": {"AF": True},
        },
    }
    document, audit = build_diagnostic_document(source)
    assert audit.version == DIAGNOSTIC_EVIDENCE_CONTRACT_VERSION
    assert unclassified_source_fields(source) == ("/future_vendor_diagnosis",)
    assert "future_vendor_diagnosis" not in document
    assert "clinical_interpretation" not in document
    assert "clinical_interpretation" not in document["metadata"]
    assert "clinical_classifications" not in document["metadata"]["patient_meta"]
    assert "statement_evidence" not in document["rhythm_inputs"]
    assert "reasons" not in document["rhythm_inputs"]["record"]["availability"]
    assert "/clinical_interpretation" in audit.dropped_sensitive_fields
    assert (
        "/metadata/patient_meta/clinical_classifications"
        in audit.dropped_sensitive_fields
    )
    assert (
        "/rhythm_inputs/record/availability/reasons"
        in audit.dropped_sensitive_fields
    )


def test_tool_plan_is_enum_driven_and_repairs_unassessed_domains():
    plan = build_tool_plan(
        {
            "hypotheses": [
                {
                    "id": "h1",
                    "status": "provisional",
                    "domains": ["rhythm_rate"],
                    "next_tools": ["get_rhythm_profile", "invented_tool"],
                }
            ],
            "targeted_checks": [
                {
                    "hypothesis_id": "h1",
                    "domain": "p_av",
                    "tool": "get_p_assessment_table",
                    "question": "organized atrial activity?",
                }
            ],
        },
        survey_state={
            "domains": {
                "q_st_t_u": {"status": "not_assessed"},
            }
        },
        include_targeted_checks=True,
    )
    assert "invented_tool" not in plan.tools
    assert "invented_tool" in plan.rejected_tools
    assert "get_measurement" in plan.tools
    assert "search_measurements" in plan.tools
    assert "q_st_t_u" in plan.repaired_domains
    assert "get_morphology_map" in plan.tools


def test_runtime_knowledge_excerpt_removes_case_specific_numbers():
    sanitized = sanitize_runtime_excerpt(
        "General PR threshold is 200 ms.\n"
        "Patients with PR above 200 ms need contextual interpretation.\n"
        "record 11 improved from 219 ms to 149.5 ms.\n"
        "JS00956 had PR 214 ms."
    )
    assert "200 ms" in sanitized.text
    assert "Patients with PR above 200 ms" in sanitized.text
    assert "219 ms" not in sanitized.text
    assert "JS00956" not in sanitized.text
    assert sanitized.removed_lines == 2


def test_blind_ablation_hides_roles_until_release_export(tmp_path):
    truth = tmp_path / "truth.jsonl"
    truth.write_text(
        json.dumps(
            {"record_id": "r1", "patient_id": "p1", "labels": ["serious"]}
        )
        + "\n",
        encoding="utf-8",
    )
    arms = {}
    for name in ("rules", "single_turn", "agent"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            json.dumps({"record_id": "r1", "labels": ["serious"]}) + "\n",
            encoding="utf-8",
        )
        arms[name] = path

    blinding_key = "test-only-blinding-key"
    blinded = evaluate_arms(
        truth,
        arms,
        serious_labels={"serious"},
        blinding_key=blinding_key,
    )
    assert blinded["blinded"] is True
    assert "arm_roles" not in blinded
    assert set(blinded["arms"]) != set(arms)

    released = evaluate_arms(
        truth,
        arms,
        serious_labels={"serious"},
        unblind=True,
        blinding_key=blinding_key,
        reviewed_report=blinded,
    )
    assert set(released["arms"]) == set(blinded["arms"])
    assert released["reviewed_report_sha256"].startswith("sha256:")
    report_path = tmp_path / "released.json"
    report_path.write_text(json.dumps(released), encoding="utf-8")
    clinical, failures = _clinical_gate(
        report_path,
        AcceptanceThresholds(min_records=1),
        required=True,
    )
    assert failures == []
    assert clinical["serious_miss_rate"] == 0.0
