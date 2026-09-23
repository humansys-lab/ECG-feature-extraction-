"""Validation tiers are mechanically tied to evidence (document 06)."""

import copy
import json
from pathlib import Path

import pytest

from tools.check_validation_registry import problems, resource_mismatches

REGISTRY = json.loads(Path("schemas/ecg-record/1.0/validation-evidence.json").read_text())


def test_shipped_registry_passes_and_copies_are_identical():
    assert resource_mismatches() == []
    assert problems(REGISTRY) == []


def test_every_published_field_is_unvalidated_in_schema_1_0():
    assert {entry["validation_tier"] for entry in REGISTRY["fields"]} == {"unvalidated"}


def test_promotion_without_evidence_fails(tmp_path):
    registry = copy.deepcopy(REGISTRY)
    registry["fields"][0]["validation_tier"] = "benchmark_validated"
    assert any("without evidence" in item for item in problems(registry, tmp_path))
    registry["fields"][0]["evidence_ids"] = ["ludb-p-onset"]
    assert any("missing" in item for item in problems(registry, tmp_path))


def test_complete_passing_manifest_allows_promotion(tmp_path):
    registry = copy.deepcopy(REGISTRY)
    entry = registry["fields"][2]
    entry.update(validation_tier="benchmark_validated", evidence_ids=["demo"], validated_for="LUDB QRS onset, 12-lead")
    manifest = {"evidence_id": "demo", "tool": {"name": "evaluate_ludb.py"}, "dataset": {"name": "LUDB"},
                "baseline_id": "b", "metric": {"name": "Se"}, "acceptance_threshold": 0.95,
                "result": {"value": 0.99, "passed": True}, "validated_for": "LUDB", "schema_major": 1,
                "code_revision": "abc", "approved_by": "Validation Owner"}
    (tmp_path / "demo.json").write_text(json.dumps(manifest))
    assert problems(registry, tmp_path) == []
    manifest["result"]["passed"] = False
    (tmp_path / "demo.json").write_text(json.dumps(manifest))
    assert problems(registry, tmp_path)


@pytest.mark.parametrize("ceiling", ["unvalidated", "indirectly_validated", "benchmark_validated"])
def test_st_morphology_ceiling_is_exactly_known_problem(ceiling):
    registry = copy.deepcopy(REGISTRY)
    cap = next(e for e in registry["debug_caps"] if e["pointer"] == "/debug/st_morphology")
    assert (cap["validation_tier"], cap["validation_ceiling"]) == ("known_problem", "known_problem")
    cap["validation_ceiling"] = ceiling
    assert any("st_morphology" in item for item in problems(registry))


def test_debug_capped_quantities_cannot_be_published():
    registry = copy.deepcopy(REGISTRY)
    registry["fields"].append(dict(registry["fields"][0], pointer="/measurements/amplitudes/st_hybrid_level_uv"))
    assert any("debug-capped" in item for item in problems(registry))


def test_record_reader_rejects_a_claim_above_the_registry():
    from feature_extraction.ecgfeat import ecg_emit
    from feature_extraction.ecgfeat.errors import RecordValidationError
    from feature_extraction.ecgfeat.record import dumps_record, loads_record
    from tests.test_library_design_review import measured_fixture

    document = json.loads(dumps_record(ecg_emit(measured_fixture())))
    document["measurements"]["intervals"]["qrs_duration_ms"]["validation"] = {"status": "benchmark_validated", "evidence": ["x"]}
    with pytest.raises(RecordValidationError, match="exceeds the registry"):
        loads_record(json.dumps(document))
    document["schema_version"] = "1.1.0"  # a later minor may promote; it carries its own registry
    document["provenance"]["schema_version"] = "1.1.0"
    loads_record(json.dumps(document))


def test_emitted_records_take_validation_only_from_the_registry():
    from feature_extraction.ecgfeat import ecg_emit
    from tests.test_library_design_review import measured_fixture

    document = ecg_emit(measured_fixture(), profile="all").as_dict()
    by_pointer = {entry["pointer"]: entry for entry in REGISTRY["fields"]}
    groups = {"/delineation/fiducials": document["delineation"]["fiducials"],
              **{f"/measurements/{k}": v for k, v in document["measurements"].items()}}
    for root, fields in groups.items():
        for name, field in fields.items():
            entry = by_pointer[f"{root}/{name}"]
            assert field["validation"] == {"status": entry["validation_tier"], "evidence": entry["evidence_ids"]}
            assert document["profile"] in entry["profiles"]
