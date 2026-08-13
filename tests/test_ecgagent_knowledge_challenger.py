from __future__ import annotations

import json

from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.backends.mock import ScriptedBackend, text_response, tool_response
from ecgagent.evidence.store import EvidenceStore
from ecgagent.knowledge.challenger import (
    EXCLUDED_RUNTIME_CATEGORIES,
    KnowledgeChallenger,
    RUNTIME_CATEGORIES,
)
from ecgagent.knowledge.index import KnowledgeBase, KnowledgeChunk
from ecgagent.knowledge.navigator import KnowledgeNavigator
from tests.test_ecgagent_diagnostic import _diagnostic_verdict
from tests.test_ecgagent_tools import _payload
from tests.diagnostic_script_helpers import legacy_script_prefix


def _knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(
        [
            KnowledgeChunk(
                chunk_id="chunk-t-wave",
                category="diagnostic_reference",
                source="docs/reference.md",
                title="Primary and secondary T-wave changes",
                line_start=10,
                line_end=20,
                text=(
                    "Before interpreting a T-wave abnormality, review secondary factors such as QRS conduction, hypertrophy, pre-excitation and pacing. "
                    "When QT is unavailable, repolarization and T-wave morphology still require assessment; an isolated T-wave observation cannot establish etiology."
                ),
                source_sha256="sha256:test",
            ),
            KnowledgeChunk(
                chunk_id="chunk-rules",
                category="clinical_rules_reference",
                source="clinical_rules/t.py",
                title="rule implementation",
                line_start=1,
                line_end=10,
                text="This implementation-only rule must stay outside patient review.",
                source_sha256="sha256:rules",
            ),
        ],
        project_root="/workspace/ecg_gemma",
    )


def _review_response() -> dict:
    return {
        "summary": "Secondary factors still need to be excluded before interpreting the T wave.",
        "issues": [
            {
                "type": "confounder",
                "priority": "HIGH",
                "topic": "Primary versus secondary origin of the T-wave abnormality",
                "why_current_reasoning_is_incomplete": (
                    "The current interpretation does not state whether conduction or pacing sufficiently explains the repolarization change."
                ),
                "evidence_review_question": (
                    "Recheck whether QRS conduction, hypertrophy and pacing measurements explain the T-wave change; if they do not discriminate, state that the mechanism is unresolved."
                ),
                "knowledge_refs": ["K1"],
            }
        ],
    }


def test_knowledge_challenger_uses_only_governed_categories_and_neutral_feedback():
    backend = ScriptedBackend(script=[text_response(json.dumps(_review_response()))])
    review = KnowledgeChallenger(_knowledge_base(), max_chunks=4).review(
        _diagnostic_verdict(),
        backend=backend,
    )

    assert review["status"] == "issues_found"
    assert {row["category"] for row in review["references"]} <= set(
        RUNTIME_CATEGORIES
    )
    assert not ({row["category"] for row in review["references"]} & set(
        EXCLUDED_RUNTIME_CATEGORIES
    ))
    feedback = KnowledgeChallenger.neutral_revision_feedback(review)
    assert feedback is not None
    assert "Recheck whether QRS conduction" in feedback
    assert "K1" not in feedback
    assert "does not state whether conduction" not in feedback
    assert "reference.md" not in feedback


def test_knowledge_navigator_builds_provisional_plan_context_without_patient_evidence():
    prompt, audit = KnowledgeNavigator(_knowledge_base(), max_chunks=4).navigate(
        "QT cannot be measured reliably, but multiple leads retain T-wave abnormalities requiring primary-versus-secondary differentiation."
    )

    assert audit["status"] == "matched"
    assert audit["patient_evidence"] is False
    assert audit["reference_count"] == 1
    assert {row["category"] for row in audit["references"]} <= set(
        RUNTIME_CATEGORIES
    )
    assert not (
        {row["category"] for row in audit["references"]}
        & set(EXCLUDED_RUNTIME_CATEGORIES)
    )
    assert "INTERIM MEDICAL KNOWLEDGE NAVIGATION" in prompt
    assert audit["references"][0]["source"] == "docs/reference.md"
    assert "docs/reference.md" not in prompt
    assert "clinical_rules/t.py" not in prompt
    assert "not patient evidence" in prompt


def test_diagnostic_agent_challenges_only_after_verified_first_pass_and_rereads_ecgfeat():
    store = EvidenceStore.from_dict(_payload(), record_id="KB001")
    first = _diagnostic_verdict()
    revised = _diagnostic_verdict()
    revised["summary"] = "Patient measurements were reread after the knowledge challenge, and the independent diagnosis remains supported."
    revised_patch = {
        "base_candidate_hash": ECGDiagnosticAgent._candidate_hash(first),
        "changed_sections": ["summary"],
        "replacement_sections": {"summary": revised["summary"]},
    }
    backend = ScriptedBackend(
        script=[
            *legacy_script_prefix(),
            text_response(json.dumps(first)),
            # This tool-free call is the isolated knowledge challenger.
            text_response(json.dumps(_review_response())),
            # The diagnostic conversation receives only a neutral question and
            # must re-read patient evidence before its revision is accepted.
            tool_response(
                (
                    "get_measurement",
                    {"pointer": "/global_features/qrs_ms"},
                )
            ),
            text_response(json.dumps(revised_patch, ensure_ascii=False)),
        ]
    )

    result = ECGDiagnosticAgent(
        store=store,
        backend=backend,
        knowledge_challenge=True,
        knowledge_base=_knowledge_base(),
        max_revisions=2,
    ).run()

    assert result.ok and result.verified, result.summary()
    assert result.pre_knowledge_verdict == first
    assert result.verdict["summary"] == revised["summary"]
    assert result.knowledge_review["revision"]["applied"] is True
    assert result.knowledge_review["revision"]["successful_measurement_calls"] == 1
    assert result.knowledge_revisions == 1
    assert [phase.key for phase in result.phases][-1] == "knowledge_revise"
    assert result.audit["knowledge_navigation"]["enabled"] is True
    assert result.audit["knowledge_navigation"]["patient_evidence"] is False
    assert result.audit["knowledge_challenge"]["first_pass_isolated"] is False
    assert result.audit["knowledge_challenge"]["posthoc_excerpts_isolated"] is True


def test_knowledge_revision_without_ecgfeat_reread_is_rejected_safely():
    store = EvidenceStore.from_dict(_payload(), record_id="KB002")
    first = _diagnostic_verdict()
    ungrounded_revision = _diagnostic_verdict()
    ungrounded_revision["summary"] = "This must not replace the first verdict."
    ungrounded_patch = {
        "base_candidate_hash": ECGDiagnosticAgent._candidate_hash(first),
        "changed_sections": ["summary"],
        "replacement_sections": {"summary": ungrounded_revision["summary"]},
    }
    backend = ScriptedBackend(
        script=[
            *legacy_script_prefix(),
            text_response(json.dumps(first)),
            text_response(json.dumps(_review_response())),
            # Mock backends do not enforce minimum phase coverage, so the
            # acceptance guard itself must still reject this no-tool revision.
            text_response(json.dumps(ungrounded_patch)),
        ]
    )

    result = ECGDiagnosticAgent(
        store=store,
        backend=backend,
        knowledge_challenge=True,
        knowledge_base=_knowledge_base(),
        max_revisions=2,
    ).run()

    assert result.ok and result.verified
    assert result.verdict == first
    assert result.knowledge_review["revision"]["applied"] is False
    assert "no successful ecgfeat call" in result.knowledge_review["revision"][
        "rejection_reason"
    ]
