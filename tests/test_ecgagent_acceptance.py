from __future__ import annotations

import json

import pytest

from ecgagent.acceptance import AcceptanceThresholds, evaluate_acceptance
from ecgagent.agent.diagnostic import DIAGNOSTIC_PROMPT_FINGERPRINT
from ecgagent.agent.protocol import DIAGNOSTIC_AGENT_PROTOCOL_VERSION
from ecgagent.evaluation_protocol import evaluate_arms


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _diagnosis(record_id: str, *, citation: bool = True) -> dict:
    pointer = "/global_features/heart_rate_bpm"
    return {
        "record_id": record_id,
        "ok": True,
        "verified": True,
        "error": None,
        "revisions": 0,
        "phases": [{"phase_guard_passed": True}],
        "verdict": {
            "summary": f"supported by ev:{pointer}" if citation else "no citation"
        },
        "verification": {"passed": True},
        "audit": {
            "agent_protocol": DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
            "prompt_fingerprint": DIAGNOSTIC_PROMPT_FINGERPRINT,
            "runtime_controls": {"elapsed_seconds": 1.0},
            "model": {
                "backend": "medgemma-local",
                "config": {
                    "model_path": "/models/test",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 7,
                },
            },
            "tools": {
                "effective_evidence": {
                    "items": [{"pointer": pointer, "model_visible": True}]
                }
            },
        },
        "batch": {
            "backend": "medgemma-local",
            "model": "/models/test",
            "thinking": False,
            "diagnostic_workflow": "compact",
            "generation_request": {
                "model_max_len": None,
                "gpu_memory_utilization": None,
            },
            "runtime_fingerprint": "sha256:" + "1" * 64,
            "agent_code_fingerprint": "sha256:" + "2" * 64,
            "model_artifact_fingerprint": "sha256:" + "3" * 64,
        },
    }


def _manifest(root, record_ids):
    payload = {
        "record_count": len(record_ids),
        "records": [{"record": record_id} for record_id in record_ids],
    }
    _write_json(root / "record_manifest.json", payload)
    return payload


def test_acceptance_fails_closed_for_duplicate_and_bad_artifacts(tmp_path):
    _manifest(tmp_path, ["r1"])
    _write_json(tmp_path / "diagnoses" / "a.json", _diagnosis("r1"))
    _write_json(tmp_path / "diagnoses" / "b.json", _diagnosis("r1"))
    (tmp_path / "diagnoses" / "broken.json").write_text("{", encoding="utf-8")

    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        require_clinical=False,
    )

    assert report["status"] == "failed"
    assert any("duplicate record_id" in failure for failure in report["failures"])
    assert any("unreadable diagnosis artifact" in failure for failure in report["failures"])
    assert any("do not exactly match" in failure for failure in report["failures"])


def test_acceptance_does_not_treat_zero_citations_as_full_provenance(tmp_path):
    _manifest(tmp_path, ["r1"])
    _write_json(
        tmp_path / "diagnoses" / "r1.json",
        _diagnosis("r1", citation=False),
    )

    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        require_clinical=False,
    )

    assert report["metrics"]["visible_final_citation_rate"] == 0.0
    assert report["status"] == "failed"


def test_acceptance_binds_clinical_export_to_manifest_and_reviewed_report(tmp_path):
    manifest = _manifest(tmp_path, ["r1"])
    _write_json(tmp_path / "diagnoses" / "r1.json", _diagnosis("r1"))
    truth = tmp_path / "truth.jsonl"
    truth.write_text(
        json.dumps({"record_id": "r1", "patient_id": "p1", "labels": ["serious"]})
        + "\n",
        encoding="utf-8",
    )
    arms = {}
    for role in ("rules", "single_turn", "agent"):
        path = tmp_path / f"{role}.jsonl"
        path.write_text(
            json.dumps({"record_id": "r1", "labels": ["serious"]}) + "\n",
            encoding="utf-8",
        )
        arms[role] = path
    key = "test-blinding-key"
    blinded = evaluate_arms(
        truth,
        arms,
        serious_labels={"serious"},
        blinding_key=key,
        record_manifest=manifest,
    )
    released = evaluate_arms(
        truth,
        arms,
        serious_labels={"serious"},
        blinding_key=key,
        unblind=True,
        reviewed_report=blinded,
        record_manifest=manifest,
    )
    reviewed_path = tmp_path / "reviewed.json"
    released_path = tmp_path / "released.json"
    _write_json(reviewed_path, blinded)
    _write_json(released_path, released)

    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        clinical_report=released_path,
        reviewed_report=reviewed_path,
        trusted_truth=truth,
    )
    assert report["status"] == "passed"

    changed_manifest = {**manifest, "dataset_dir": "/different/source"}
    _write_json(tmp_path / "record_manifest.json", changed_manifest)
    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        clinical_report=released_path,
        reviewed_report=reviewed_path,
        trusted_truth=truth,
    )
    assert report["status"] == "failed"
    assert any("exact locked record manifest" in failure for failure in report["failures"])

    _write_json(tmp_path / "record_manifest.json", manifest)

    released["record_ids_sha256"] = "sha256:wrong"
    _write_json(released_path, released)
    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        clinical_report=released_path,
        reviewed_report=reviewed_path,
        trusted_truth=truth,
    )
    assert report["status"] == "failed"
    assert any("record identifiers" in failure for failure in report["failures"])


def test_acceptance_requires_complete_runtime_and_model_fingerprints(tmp_path):
    _manifest(tmp_path, ["r1"])
    diagnosis = _diagnosis("r1")
    diagnosis["batch"].pop("runtime_fingerprint")
    diagnosis["batch"].pop("agent_code_fingerprint")
    diagnosis["batch"].pop("model_artifact_fingerprint")
    _write_json(tmp_path / "diagnoses" / "r1.json", diagnosis)

    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        require_clinical=False,
    )

    assert report["status"] == "failed"
    assert any(
        "missing required fingerprint" in failure
        for failure in report["failures"]
    )


def test_acceptance_requires_explicit_generation_request(tmp_path):
    _manifest(tmp_path, ["r1"])
    diagnosis = _diagnosis("r1")
    diagnosis["batch"].pop("generation_request")
    _write_json(tmp_path / "diagnoses" / "r1.json", diagnosis)

    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        require_clinical=False,
    )

    assert report["status"] == "failed"
    assert "cohort execution signature is incomplete" in report["failures"]


def test_acceptance_binds_clinical_report_to_explicit_trusted_truth(tmp_path):
    manifest = _manifest(tmp_path, ["r1"])
    _write_json(tmp_path / "diagnoses" / "r1.json", _diagnosis("r1"))
    truth = tmp_path / "truth.jsonl"
    truth.write_text(
        json.dumps({"record_id": "r1", "patient_id": "p1", "labels": ["serious"]})
        + "\n",
        encoding="utf-8",
    )
    arms = {}
    for role in ("rules", "single_turn", "agent"):
        path = tmp_path / f"{role}.jsonl"
        path.write_text(
            json.dumps({"record_id": "r1", "labels": ["serious"]}) + "\n",
            encoding="utf-8",
        )
        arms[role] = path
    blinded = evaluate_arms(
        truth,
        arms,
        serious_labels={"serious"},
        blinding_key="trusted-truth-test-key",
        record_manifest=manifest,
    )
    released = evaluate_arms(
        truth,
        arms,
        serious_labels={"serious"},
        unblind=True,
        blinding_key="trusted-truth-test-key",
        reviewed_report=blinded,
        record_manifest=manifest,
    )
    reviewed_path = tmp_path / "reviewed.json"
    released_path = tmp_path / "released.json"
    _write_json(reviewed_path, blinded)
    _write_json(released_path, released)
    wrong_truth = tmp_path / "wrong_truth.jsonl"
    wrong_truth.write_text(
        json.dumps({"record_id": "r1", "patient_id": "p1", "labels": ["easy"]})
        + "\n",
        encoding="utf-8",
    )

    missing = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        clinical_report=released_path,
        reviewed_report=reviewed_path,
    )
    mismatched = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        clinical_report=released_path,
        reviewed_report=reviewed_path,
        trusted_truth=wrong_truth,
    )

    assert any("trusted truth source is required" in item for item in missing["failures"])
    assert any("trusted truth source" in item for item in mismatched["failures"])


@pytest.mark.parametrize(
    "inconsistency",
    ("not_ok", "error", "missing_verdict", "missing_verification", "failed_verification"),
)
def test_acceptance_rejects_inconsistent_verified_artifacts(
    tmp_path,
    inconsistency,
):
    _manifest(tmp_path, ["r1"])
    diagnosis = _diagnosis("r1")
    if inconsistency == "not_ok":
        diagnosis["ok"] = False
    elif inconsistency == "error":
        diagnosis["error"] = "execution failed"
    elif inconsistency == "missing_verdict":
        diagnosis["verdict"] = None
    elif inconsistency == "missing_verification":
        diagnosis["verification"] = None
    else:
        diagnosis["verification"]["passed"] = False
    _write_json(tmp_path / "diagnoses" / "r1.json", diagnosis)

    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        require_clinical=False,
    )

    assert report["status"] == "failed"
    assert any(
        "declare verified=true without ok=true" in failure
        for failure in report["failures"]
    )
    assert report["metrics"]["verified_execution_success_rate"] == 0.0


@pytest.mark.parametrize("phase_state", ("missing", "empty", "missing_guard"))
def test_acceptance_requires_auditable_phase_guards(tmp_path, phase_state):
    _manifest(tmp_path, ["r1"])
    diagnosis = _diagnosis("r1")
    if phase_state == "missing":
        diagnosis.pop("phases")
    elif phase_state == "empty":
        diagnosis["phases"] = []
    else:
        diagnosis["phases"] = [{}]
    _write_json(tmp_path / "diagnoses" / "r1.json", diagnosis)

    report = evaluate_acceptance(
        tmp_path,
        thresholds=AcceptanceThresholds(min_records=1),
        require_clinical=False,
    )

    assert report["status"] == "failed"
    if phase_state == "missing_guard":
        assert any("phase_guard_passed" in failure for failure in report["failures"])
        assert report["metrics"]["phase_guard_failure_rate"] == 1.0
    else:
        assert "cohort contains zero auditable phases" in report["failures"]
        assert report["metrics"]["phase_guard_failure_rate"] is None


def test_acceptance_requires_success_and_verification_on_the_same_records(tmp_path):
    record_ids = [f"r{index:02d}" for index in range(20)]
    _manifest(tmp_path, record_ids)
    for index, record_id in enumerate(record_ids):
        diagnosis = _diagnosis(record_id)
        if index == 18:
            diagnosis["verified"] = False
            diagnosis["verification"]["passed"] = False
        elif index == 19:
            diagnosis["ok"] = False
            diagnosis["error"] = "execution failed"
        _write_json(tmp_path / "diagnoses" / f"{record_id}.json", diagnosis)

    report = evaluate_acceptance(tmp_path, require_clinical=False)

    assert report["metrics"]["execution_success_rate"] == pytest.approx(0.95)
    assert report["metrics"]["verified_rate"] == pytest.approx(0.95)
    assert report["metrics"]["verified_execution_success_rate"] == pytest.approx(0.90)
    assert report["status"] == "failed"
    assert any("joint threshold" in failure for failure in report["failures"])
