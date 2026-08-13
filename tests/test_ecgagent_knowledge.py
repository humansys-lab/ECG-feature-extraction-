from __future__ import annotations

from ecgagent.knowledge import KnowledgeBase
from ecgagent.tools.registry import build_default_registry
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_tools import _payload


def test_detached_knowledge_base_indexes_documents_and_reference_engines():
    base = KnowledgeBase.from_project()
    manifest = base.manifest()

    assert manifest["agent_runtime_registered"] is False
    assert manifest["chunk_count"] > 100
    assert {
        "morphology_reference",
        "diagnostic_reference",
        "failure_modes",
        "clinical_rules_reference",
        "philips_dxl_reference",
        "glasgow_reference",
        "capability_blueprint",
        "measurement_reliability_reference",
    } <= set(manifest["categories"])


def test_detached_knowledge_search_finds_ecg_topics_and_source_lines():
    base = KnowledgeBase.from_project()
    st_results = base.search("ST elevation reciprocal depression contiguous leads", limit=5)
    assert st_results
    assert any(
        result.chunk.source
        == "ecgagent/knowledge/english_diagnostic_reference.md"
        for result in st_results
    )
    assert all(result.chunk.line_start > 0 for result in st_results)

    qtc_results = base.search(
        "QTc Bazett Hodges divergence",
        categories=["clinical_rules_reference"],
        limit=8,
    )
    assert qtc_results
    assert any("sources.py" in result.chunk.source for result in qtc_results)


def test_knowledge_index_round_trips_without_joining_agent_registry(tmp_path):
    base = KnowledgeBase.from_project()
    path = base.save(tmp_path / "knowledge.json")
    loaded = KnowledgeBase.load(path)
    assert loaded.manifest()["chunk_count"] == base.manifest()["chunk_count"]
    assert loaded.search("atrial flutter F wave")

    store = EvidenceStore.from_dict(_payload(), record_id="NO-KB")
    registry = build_default_registry(store, budget=None)
    assert all(not name.startswith("lookup_") for name in registry.specs)
    assert "knowledge" not in " ".join(registry.specs)
