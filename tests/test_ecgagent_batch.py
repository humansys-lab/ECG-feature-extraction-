from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from ecgagent.agent.diagnostic import (
    DIAGNOSTIC_AGENT_PROTOCOL_VERSION as AGENT_PROTOCOL_VERSION,
    DIAGNOSTIC_PROMPT_FINGERPRINT as PROMPT_FINGERPRINT,
)
from ecgagent import batch
from ecgagent.backends.mock import ScriptedBackend, text_response, tool_response
from ecgagent.batch import (
    _audit_whitelist,
    _diagnosis_codes,
    _extraction_provenance,
    _fatal_external_error,
    _reusable_feature,
    _reusable_diagnosis,
)
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_tools import _payload
from tests.test_ecgagent_diagnostic import _diagnostic_verdict


def test_saved_whitelist_uses_only_model_visible_provenance():
    payload = {
        "audit": {
            "tools": {
                "provenance": {
                    "visible_by_source": {"survey": ["/shown"]},
                },
                "calls": [
                    {
                        "citations": ["/hidden", "/call-shown"],
                        "visible_citations": ["/call-shown"],
                    }
                ],
            }
        }
    }
    assert _audit_whitelist(payload) == frozenset({"/shown", "/call-shown"})


def test_saved_whitelist_prefers_materialized_final_evidence():
    payload = {
        "audit": {
            "tools": {
                "effective_evidence": {
                    "items": [
                        {
                            "pointer": "/used-and-visible",
                            "model_visible": True,
                            "value": 12,
                        },
                        {
                            "pointer": "/used-but-hidden",
                            "model_visible": False,
                            "value": 13,
                        },
                    ]
                },
                "provenance": {
                    "visible_by_source": {
                        "survey": {
                            "count": 120,
                            "evidence_set_id": "sha256:compact-only",
                        }
                    }
                },
                "calls": [],
            }
        }
    }

    assert _audit_whitelist(payload) == frozenset({"/used-and-visible"})


def test_reuse_requires_matching_protocol_prompt_model_and_input(tmp_path):
    feature_path = tmp_path / "record_features.json"
    feature_path.write_text(
        json.dumps(
            {
                "clinical_interpretation": {
                    "artifact_fingerprint": "sha256:feature",
                }
            }
        ),
        encoding="utf-8",
    )
    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    diagnostic_fingerprint = (
        EvidenceStore.from_dict(feature).diagnostic_view().fingerprint()
    )
    policy = batch.resolve_backend_privacy("deepseek")
    model_provenance = batch._model_provenance_fingerprint(
        backend="deepseek",
        model="deepseek-v4-flash",
        endpoint_fingerprint=policy.endpoint_fingerprint,
        thinking=False,
        reasoning_effort="high",
        model_max_len=None,
        gpu_memory_utilization=None,
    )
    payload = {
        "verified": True,
        "audit": {
            "agent_protocol": AGENT_PROTOCOL_VERSION,
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "input_fingerprint": diagnostic_fingerprint,
            "knowledge_navigation": {
                "enabled": True,
                "max_chunks": 4,
                "patient_evidence": False,
            },
        },
        "batch": {
            "backend": "deepseek",
            "model": "deepseek-v4-flash",
            "thinking": False,
            "reasoning_effort": None,
            "runtime_fingerprint": batch._runtime_fingerprint(),
            "agent_code_fingerprint": batch._agent_code_fingerprint(),
            "model_artifact_fingerprint": model_provenance,
            "model_provenance_fingerprint": model_provenance,
            "external_egress": policy.audit(allowed=True),
        },
    }

    assert _reusable_diagnosis(
        payload,
        feature_path,
        model="deepseek-v4-flash",
        thinking=False,
        reasoning_effort="high",
    )

    stale = json.loads(json.dumps(payload))
    stale["audit"]["prompt_fingerprint"] = "old"
    assert not _reusable_diagnosis(
        stale,
        feature_path,
        model="deepseek-v4-flash",
        thinking=False,
        reasoning_effort="high",
    )
    assert not _reusable_diagnosis(
        payload,
        feature_path,
        model="deepseek-v4-pro",
        thinking=False,
        reasoning_effort="high",
    )
    assert not _reusable_diagnosis(
        payload,
        feature_path,
        model="deepseek-v4-flash",
        thinking=False,
        reasoning_effort="high",
        backend="medgemma-local",
    )
    legacy = json.loads(json.dumps(payload))
    legacy["audit"]["agent_protocol"] = "ecgagent.v7"
    assert not _reusable_diagnosis(
        legacy,
        feature_path,
        model="deepseek-v4-flash",
        thinking=False,
        reasoning_effort="high",
    )


def test_feature_reuse_requires_source_config_schema_and_code_fingerprint(tmp_path):
    record_base = tmp_path / "r1"
    record_base.with_suffix(".hea").write_text("header", encoding="utf-8")
    record_base.with_suffix(".dat").write_bytes(b"signal-v1")
    record = {
        "record": "r1",
        "record_path": str(record_base),
        "age": 42,
        "sex": "female",
    }
    provenance = _extraction_provenance(record, 500)
    assert provenance is not None
    feature_path = tmp_path / "r1_features.json"
    feature_path.write_text(
        json.dumps(
            {
                "clinical_interpretation": {},
                "_ecgagent_extraction": provenance,
            }
        ),
        encoding="utf-8",
    )

    assert _reusable_feature(feature_path, record, fs_internal=500)
    assert not _reusable_feature(feature_path, record, fs_internal=250)

    record["age_days"] = 42 * 365.25 + 30
    assert not _reusable_feature(feature_path, record, fs_internal=500)

    record.pop("age_days")
    record_base.with_suffix(".dat").write_bytes(b"signal-v2")
    assert not _reusable_feature(feature_path, record, fs_internal=500)

    feature_path.write_text(
        json.dumps({"clinical_interpretation": {}}),
        encoding="utf-8",
    )
    assert not _reusable_feature(feature_path, record, fs_internal=500)


def _write_provenanced_feature(tmp_path, record_id="r1", *, fs_internal=500):
    record_base = tmp_path / "source" / record_id
    record_base.parent.mkdir(parents=True, exist_ok=True)
    record_base.with_suffix(".hea").write_text("header", encoding="utf-8")
    record_base.with_suffix(".dat").write_bytes(b"signal-v1")
    record = {
        "record": record_id,
        "record_path": str(record_base),
        "age": 50,
        "sex": "female",
        "reference_available": False,
    }
    provenance = _extraction_provenance(record, fs_internal)
    assert provenance is not None
    feature_path = tmp_path / "features" / f"{record_id}_features.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text(
        json.dumps(
            {
                "clinical_interpretation": {},
                "_ecgagent_extraction": provenance,
            }
        ),
        encoding="utf-8",
    )
    return record, feature_path, record_base


def test_diagnosis_rejects_wrong_feature_provenance_before_model_call(
    tmp_path,
    monkeypatch,
):
    record, _, record_base = _write_provenanced_feature(tmp_path)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("stale feature reached the model")

    monkeypatch.setattr(batch, "_diagnose_one", unexpected_call)
    wrong_fs = batch.run_diagnosis(
        [record],
        tmp_path,
        workers=1,
        model="served-qwen",
        max_revisions=0,
        record_retries=0,
        thinking=False,
        reasoning_effort="high",
        verbose=False,
        reuse_existing=True,
        retry_failed=True,
        backend="qwen-local",
        fs_internal=250,
    )
    assert wrong_fs[0]["status"] == "missing_feature"

    record_base.with_suffix(".dat").write_bytes(b"signal-v2")
    changed_source = batch.run_diagnosis(
        [record],
        tmp_path,
        workers=1,
        model="served-qwen",
        max_revisions=0,
        record_retries=0,
        thinking=False,
        reasoning_effort="high",
        verbose=False,
        reuse_existing=True,
        retry_failed=True,
        backend="qwen-local",
        fs_internal=500,
    )
    assert changed_source[0]["status"] == "missing_feature"


def test_no_retry_failed_does_not_preserve_stale_verified_diagnosis(
    tmp_path,
    monkeypatch,
):
    record, _, _ = _write_provenanced_feature(tmp_path)
    diagnosis_path = tmp_path / "diagnoses" / "r1.json"
    diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
    diagnosis_path.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "ok": True,
                "verified": True,
                "error": None,
                "audit": {"prompt_fingerprint": "stale"},
                "batch": {},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_diagnose(task, **kwargs):
        calls.append(task[0]["record"])
        return {
            "record": task[0]["record"],
            "status": "verified",
            "runtime_seconds": 0.01,
            "attempts": 1,
            "error": None,
            "diagnosis_path": str(diagnosis_path),
        }

    import ecgagent.backends

    monkeypatch.setattr(ecgagent.backends, "build_backend", lambda *a, **k: object())
    monkeypatch.setattr(batch, "_diagnose_one", fake_diagnose)

    rows = batch.run_diagnosis(
        [record],
        tmp_path,
        workers=1,
        model="served-qwen",
        max_revisions=0,
        record_retries=0,
        thinking=False,
        reasoning_effort="high",
        verbose=False,
        reuse_existing=True,
        retry_failed=False,
        backend="qwen-local",
        fs_internal=500,
    )

    assert calls == ["r1"]
    assert rows[0]["status"] == "verified"


def test_batch_cli_returns_nonzero_for_each_operational_failure_stage(
    tmp_path,
    monkeypatch,
):
    record = {"record": "r1", "reference_available": False}
    monkeypatch.setattr(batch, "discover_records", lambda *a, **k: [record])
    monkeypatch.setattr(batch, "_feature_code_fingerprint", lambda: "sha256:test")

    monkeypatch.setattr(
        batch,
        "run_extraction",
        lambda *a, **k: [{"record": "r1", "status": "failed"}],
    )
    assert batch.main(
        ["extract", "--output-dir", str(tmp_path / "extract")]
    ) == 2

    monkeypatch.setattr(
        batch,
        "run_diagnosis",
        lambda *a, **k: [{"record": "r1", "status": "unverified"}],
    )
    assert batch.main(
        ["diagnose", "--output-dir", str(tmp_path / "diagnose")]
    ) == 2

    monkeypatch.setattr(
        batch,
        "run_extraction",
        lambda *a, **k: [{"record": "r1", "status": "ok"}],
    )
    monkeypatch.setattr(
        batch,
        "analyze_results",
        lambda *a, **k: {"missing_or_invalid_results": []},
    )
    assert batch.main(
        ["all", "--output-dir", str(tmp_path / "all")]
    ) == 2

    monkeypatch.setattr(
        batch,
        "reverify_results",
        lambda *a, **k: {
            "counts": {"still_unverified": 1},
            "records": [{"record": "r1", "status": "still_unverified"}],
        },
    )
    assert batch.main(
        ["reverify", "--output-dir", str(tmp_path / "reverify")]
    ) == 2


def test_reverify_rejects_verified_artifact_with_stale_input(tmp_path):
    record_id = "stale-verified"
    feature_path = tmp_path / "features" / f"{record_id}_features.json"
    diagnosis_path = tmp_path / "diagnoses" / f"{record_id}.json"
    feature_path.parent.mkdir(parents=True)
    diagnosis_path.parent.mkdir(parents=True)
    feature_path.write_text(json.dumps(_payload()), encoding="utf-8")
    diagnosis_path.write_text(
        json.dumps(
            {
                "record_id": record_id,
                "ok": True,
                "verified": True,
                "error": None,
                "verdict": {"diagnoses": []},
                "verification": {"passed": True},
                "audit": {
                    "agent_protocol": AGENT_PROTOCOL_VERSION,
                    "prompt_fingerprint": PROMPT_FINGERPRINT,
                    "input_fingerprint": "sha256:stale",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = batch.reverify_results(
        [{"record": record_id, "reference_available": False}],
        tmp_path,
    )

    assert summary["counts"] == {"stale": 1}
    assert summary["records"][0]["verified"] is False


def test_pseudonym_key_is_atomic_private_and_stable(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = list(
            pool.map(
                lambda _: batch._load_or_create_pseudonym_key(tmp_path),
                range(24),
            )
        )

    assert all(key == keys[0] for key in keys)
    assert len(keys[0]) == 32
    key_path = tmp_path / batch.PSEUDONYM_KEY_FILENAME
    assert os.stat(key_path).st_mode & 0o777 == 0o600
    identity = batch._record_identity_audit("real-record", keys[0])
    assert identity == batch._record_identity_audit("real-record", keys[0])
    assert identity != batch._record_identity_audit("another-record", keys[0])
    assert "real-record" not in json.dumps(identity)
    assert keys[0].hex() not in json.dumps(identity)


def test_remote_qwen_from_environment_requires_authorization_and_is_audited(
    tmp_path,
    monkeypatch,
):
    endpoint = "https://user:secret@qwen.example:9443/v1?token=hidden"
    monkeypatch.setenv("QWEN_BASE_URL", endpoint)

    with pytest.raises(RuntimeError, match="allow-external-egress"):
        batch.run_diagnosis(
            [],
            tmp_path,
            workers=1,
            model="served-qwen",
            max_revisions=0,
            record_retries=0,
            thinking=True,
            reasoning_effort="high",
            verbose=False,
            reuse_existing=False,
            retry_failed=True,
            backend="qwen-local",
        )

    rows = batch.run_diagnosis(
        [],
        tmp_path,
        workers=1,
        model="served-qwen",
        max_revisions=0,
        record_retries=0,
        thinking=True,
        reasoning_effort="high",
        verbose=False,
        reuse_existing=False,
        retry_failed=True,
        backend="qwen-local",
        allow_external_egress=True,
    )
    assert rows == []
    manifest_text = (tmp_path / "diagnosis_manifest.json").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(manifest_text)
    assert manifest["external_egress"]["external"] is True
    assert manifest["external_egress"]["authorized"] is True
    assert manifest["external_egress"]["endpoint"]["display"] == (
        "https://qwen.example:9443"
    )
    assert "secret" not in manifest_text
    assert "token" not in manifest_text


def test_cache_signature_rejects_endpoint_generation_and_malformed_values(
    tmp_path,
):
    feature_path = tmp_path / "record_features.json"
    feature_path.write_text(
        json.dumps({"clinical_interpretation": {}}),
        encoding="utf-8",
    )
    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    input_fingerprint = EvidenceStore.from_dict(feature).diagnostic_view().fingerprint()
    policy = batch.resolve_backend_privacy(
        "qwen-local",
        qwen_base_url="http://127.0.0.1:8000/v1",
    )
    egress = policy.audit(allowed=False)
    model_provenance = batch._model_provenance_fingerprint(
        backend="qwen-local",
        model="served-qwen",
        endpoint_fingerprint=policy.endpoint_fingerprint,
        thinking=True,
        reasoning_effort="high",
        model_max_len=None,
        gpu_memory_utilization=None,
    )
    identity = {
        "model_visible_record_id": "ecg_deadbeefdeadbeefdeadbeef",
        "strategy": batch.PSEUDONYM_STRATEGY,
        "key_scope": "output_directory",
    }
    payload = {
        "verified": True,
        "audit": {
            "agent_protocol": AGENT_PROTOCOL_VERSION,
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "input_fingerprint": input_fingerprint,
            "diagnostic_workflow": "compact",
            "knowledge_navigation": {"enabled": False},
            "knowledge_challenge": {"enabled": False},
            "record_identity": identity,
        },
        "batch": {
            "backend": "qwen-local",
            "model": "served-qwen",
            "thinking": True,
            "reasoning_effort": "high",
            "max_revisions": 1,
            "record_retries": 0,
            "runtime_fingerprint": batch._runtime_fingerprint(),
            "agent_code_fingerprint": batch._agent_code_fingerprint(),
            "model_provenance_fingerprint": model_provenance,
            "generation_request": {
                "model_max_len": None,
                "gpu_memory_utilization": None,
            },
            "external_egress": egress,
        },
    }
    signature = {
        "model": "served-qwen",
        "thinking": True,
        "reasoning_effort": "high",
        "backend": "qwen-local",
        "diagnostic_workflow": "compact",
        "external_egress": egress,
        "record_identity": identity,
        "max_revisions": 1,
        "record_retries": 0,
        "model_provenance_fingerprint": model_provenance,
        "generation_request": {
            "model_max_len": None,
            "gpu_memory_utilization": None,
        },
    }
    assert batch._diagnosis_signature_matches(payload, feature_path, **signature)

    changed = json.loads(json.dumps(payload))
    changed["batch"]["generation_request"]["model_max_len"] = 4096
    assert not batch._diagnosis_signature_matches(
        changed, feature_path, **signature
    )
    changed = json.loads(json.dumps(payload))
    changed["batch"]["external_egress"]["endpoint"]["fingerprint"] = (
        "sha256:" + "0" * 64
    )
    assert not batch._diagnosis_signature_matches(
        changed, feature_path, **signature
    )
    changed = json.loads(json.dumps(payload))
    changed["batch"]["max_revisions"] = "malformed"
    assert not batch._diagnosis_signature_matches(
        changed, feature_path, **signature
    )


def test_local_model_provenance_is_fresh_for_each_batch_run(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "weights.bin"
    weights.write_bytes(b"v1")
    fingerprint_args = {
        "backend": "medgemma-local",
        "model": str(model_dir),
        "endpoint_fingerprint": "local",
        "thinking": False,
        "reasoning_effort": "high",
        "model_max_len": None,
        "gpu_memory_utilization": None,
    }

    first = batch._fresh_model_provenance_fingerprint(**fingerprint_args)
    weights.write_bytes(b"v2")
    second = batch._fresh_model_provenance_fingerprint(**fingerprint_args)

    assert first != second
    for fingerprint in (first, second):
        prefix, digest = fingerprint.split(":", 1)
        assert prefix == "sha256"
        assert len(digest) == 64
        int(digest, 16)


def test_batch_defaults_local_and_requires_explicit_cloud_egress(tmp_path):
    args = batch.build_parser().parse_args(["diagnose"])
    assert args.backend == "medgemma-local"
    assert args.allow_external_egress is False

    assert batch.main(["diagnose", "--backend", "deepseek"]) == 2
    with pytest.raises(RuntimeError, match="allow-external-egress"):
        batch.run_diagnosis(
            [],
            tmp_path,
            workers=1,
            model="cloud-model",
            max_revisions=0,
            record_retries=0,
            thinking=False,
            reasoning_effort="high",
            verbose=False,
            reuse_existing=False,
            retry_failed=True,
            backend="deepseek",
        )


def test_diagnosis_codes_excludes_withdrawn_findings():
    codes, statuses = _diagnosis_codes(
        {
            "verdict": {
                "diagnoses": [
                    {"code": "sinus_rhythm", "status": "unchanged"},
                    {"code": "old_false_positive", "status": "withdrawn"},
                    {"code": "first_degree_av_delay", "status": "added"},
                ]
            }
        }
    )

    assert codes == {"sinus_rhythm", "first_degree_av_delay"}
    assert statuses == {
        "unchanged": 1,
        "withdrawn": 1,
        "added": 1,
    }


@pytest.mark.parametrize(
    "error",
    [
        "APIStatusError: Error code: 402 - Insufficient Balance",
        "AuthenticationError: invalid API key",
        "PermissionDeniedError: Error code: 403",
        "insufficient_quota",
    ],
)
def test_fatal_external_error_detection(error):
    assert _fatal_external_error(error)


def test_rate_limit_is_not_treated_as_a_fatal_account_error():
    assert not _fatal_external_error("RateLimitError: Error code: 429")


def test_diagnose_one_saves_standalone_full_markdown_trace(tmp_path):
    record_id = "TRACE001_hr"
    feature_path = tmp_path / "features" / f"{record_id}_features.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature = _payload()
    feature["metadata"]["diagnostic_gate"] = {
        "state": "pass",
        "stop_reasons": [],
        "partial_reasons": [],
        "allowed_domains": ["all"],
        "suppressed_domains": [],
    }
    feature_path.write_text(json.dumps(feature), encoding="utf-8")
    backend = ScriptedBackend(
        script=[
            tool_response(
                ("get_diagnostic_overview", {}),
            ),
            text_response("Survey complete with ten-domain observations."),
            text_response("Provisional diagnosis and targeted plan complete."),
            tool_response(
                (
                    "get_measurement",
                    {"pointer": "/global_features/heart_rate_bpm"},
                )
            ),
            text_response("Targeted investigation complete."),
            text_response("Falsification complete."),
            text_response(json.dumps(_diagnostic_verdict())),
        ]
    )
    task = (
        {"record": record_id, "reference_available": False},
        str(tmp_path),
        "mock-model",
        0,
        0,
        False,
        "high",
        False,
        False,
        8,
    )

    row = batch._diagnose_one(
        task,
        backend_name="mock",
        backend_override=backend,
    )

    trace_path = tmp_path / "agent_traces" / f"{record_id}_agent_trace.md"
    assert row["trace_path"] == str(trace_path)
    assert trace_path.exists()
    trace = trace_path.read_text(encoding="utf-8")
    assert "# ECG Agent Execution Trace" in trace
    assert "Complete model-visible result" in trace
    assert "get_diagnostic_overview" in trace
    assert "INTERIM MEDICAL KNOWLEDGE NAVIGATION" in trace
    diagnosis = json.loads(
        (tmp_path / "diagnoses" / f"{record_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnosis["trajectory"]["path"] == str(trace_path)
    assert diagnosis["trajectory"]["full_model_visible_tool_results"] is True
    brief_path = tmp_path / "brief_reports" / f"{record_id}_brief.md"
    assert row["brief_report_path"] == str(brief_path)
    assert diagnosis["brief_report"] == brief_path.read_text(
        encoding="utf-8"
    ).rstrip()
    assert "# Brief ECG Diagnostic Report" in diagnosis["brief_report"]
    assert "## Recommendation" in diagnosis["brief_report"]
    identity = diagnosis["audit"]["record_identity"]
    assert identity["model_visible_record_id"].startswith("ecg_")
    assert identity["model_visible_record_id"] != record_id
    assert identity["strategy"] == batch.PSEUDONYM_STRATEGY


def test_local_batch_microbatches_one_backend_without_deepseek_credentials(
    tmp_path,
    monkeypatch,
):
    records = []
    for index in range(2):
        record = f"{index:05d}_hr"
        records.append({"record": record, "reference_available": False})
        feature_path = tmp_path / "features" / f"{record}_features.json"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text(
            json.dumps({"clinical_interpretation": {}}),
            encoding="utf-8",
        )

    shared_backend = object()
    builds = []
    calls = []

    def fake_build_backend(name, **kwargs):
        builds.append((name, kwargs))
        return shared_backend

    def fake_diagnose(
        task,
        *,
        backend_name="deepseek",
        backend_override=None,
    ):
        record = task[0]["record"]
        calls.append((record, backend_name, backend_override))
        return {
            "record": record,
            "status": "verified",
            "runtime_seconds": 0.01,
            "attempts": 1,
            "error": None,
            "diagnosis_path": str(tmp_path / "diagnoses" / f"{record}.json"),
        }

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    import ecgagent.backends

    monkeypatch.setattr(ecgagent.backends, "build_backend", fake_build_backend)
    monkeypatch.setattr(batch, "_diagnose_one", fake_diagnose)
    monkeypatch.setattr(batch, "_reusable_feature", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        batch,
        "_fresh_model_provenance_fingerprint",
        lambda **kwargs: "sha256:" + "a" * 64,
    )

    result = batch.run_diagnosis(
        records,
        tmp_path,
        workers=4,
        model="/workspace/ecg_gemma/medgemma-27b",
        max_revisions=0,
        record_retries=0,
        thinking=False,
        reasoning_effort="high",
        verbose=False,
        reuse_existing=False,
        retry_failed=True,
        backend="medgemma-local",
        model_max_len=16384,
        gpu_memory_utilization=0.8,
    )

    assert len(result) == 2
    assert builds == [
        (
            "medgemma-local",
            {
                "model": "/workspace/ecg_gemma/medgemma-27b",
                "model_max_len": 16384,
                "gpu_memory_utilization": 0.8,
                "max_batch_size": 4,
            },
        )
    ]
    assert sorted(row[0] for row in calls) == ["00000_hr", "00001_hr"]
    assert all(row[1] == "medgemma-local" for row in calls)
    assert all(row[2] is shared_backend for row in calls)


def test_diagnosis_circuit_breaker_bounds_inflight_work(
    tmp_path,
    monkeypatch,
):
    records = []
    for index in range(10):
        record = f"{index:05d}_hr"
        records.append({"record": record, "reference_available": False})
        feature_path = tmp_path / "features" / f"{record}_features.json"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text(
            json.dumps({"clinical_interpretation": {}}),
            encoding="utf-8",
        )

    calls = []

    def fail_with_balance(task):
        record = task[0]["record"]
        calls.append(record)
        return {
            "record": record,
            "status": "failed",
            "runtime_seconds": 0.01,
            "attempts": 1,
            "error": "APIStatusError: Error code: 402 - Insufficient Balance",
            "diagnosis_path": str(tmp_path / "diagnoses" / f"{record}.json"),
        }

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    monkeypatch.setattr(batch, "_diagnose_one", fail_with_balance)
    monkeypatch.setattr(batch, "_reusable_feature", lambda *args, **kwargs: True)

    with pytest.raises(RuntimeError, match="circuit breaker"):
        batch.run_diagnosis(
            records,
            tmp_path,
            workers=2,
            model="test-model",
            max_revisions=0,
            record_retries=0,
            thinking=False,
            reasoning_effort="high",
            verbose=False,
            reuse_existing=False,
            retry_failed=True,
            backend="deepseek",
            allow_external_egress=True,
        )

    assert len(calls) <= 2
    manifest = json.loads(
        (tmp_path / "diagnosis_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["circuit_breaker"]["tripped"] is True
    assert manifest["circuit_breaker"]["deferred_records"] >= 8


def test_analysis_excludes_unverified_candidate_diagnoses(tmp_path):
    records = [
        {"record": "00001_hr", "reference_available": False},
        {"record": "00002_hr", "reference_available": False},
    ]
    diagnoses = tmp_path / "diagnoses"
    diagnoses.mkdir()
    common = {
        "reference": {"available": False, "codes_active": []},
        "verification": None,
        "revisions": 0,
        "audit": {
            "agent_protocol": AGENT_PROTOCOL_VERSION,
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "tools": {"n_calls": 0},
            "model": {"usage": {}},
        },
        "batch": {"runtime_seconds": 1.0},
    }
    (diagnoses / "00001_hr.json").write_text(
        json.dumps(
            {
                **common,
                "record_id": "00001_hr",
                "verified": True,
                "error": None,
                "source": {
                    "baseline_codes": ["sinus_rhythm"],
                    "gate_state": "pass",
                    "record_grade": "Q0",
                },
                "verdict": {
                    "diagnoses": [
                        {
                            "code": "sinus_rhythm",
                            "status": "unchanged",
                        }
                    ],
                    "human_review": {"required": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (diagnoses / "00002_hr.json").write_text(
        json.dumps(
            {
                **common,
                "record_id": "00002_hr",
                "verified": False,
                "error": "APIStatusError: Error code: 402 - Insufficient Balance",
                "source": {
                    "baseline_codes": ["lvh_voltage_criteria"],
                    "gate_state": "pass",
                    "record_grade": "Q0",
                },
                "verdict": {
                    "diagnoses": [
                        {
                            "code": "invented_candidate",
                            "status": "added",
                        }
                    ],
                    "human_review": {"required": True},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = batch.analyze_results(records, tmp_path)

    assert summary["execution"] == {"verified": 1, "error": 1}
    assert summary["agent_code_counts"] == {"sinus_rhythm": 1}
    assert "invented_candidate" not in summary["agent_code_counts"]
    assert summary["baseline_code_counts"] == {"sinus_rhythm": 1}
    assert summary["all_record_baseline_code_counts"] == {
        "sinus_rhythm": 1,
        "lvh_voltage_criteria": 1,
    }
    assert summary["error_category_counts"] == {
        "deepseek_insufficient_balance": 1
    }
    manifest = json.loads(
        (tmp_path / "diagnosis_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["manifest_origin"] == "analysis_reconstruction"
    assert manifest["record_count"] == 2
    assert manifest["verified_or_skipped"] == 1


def test_resume_probes_prior_fatal_error_before_unverified_candidate(
    tmp_path,
    monkeypatch,
):
    records = [
        {"record": "00001_hr", "reference_available": False},
        {"record": "00002_hr", "reference_available": False},
    ]
    for record in records:
        feature_path = (
            tmp_path / "features" / f"{record['record']}_features.json"
        )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text(
            json.dumps({"clinical_interpretation": {}}),
            encoding="utf-8",
        )
    diagnoses = tmp_path / "diagnoses"
    diagnoses.mkdir()
    (diagnoses / "00001_hr.json").write_text(
        json.dumps({"verified": False, "error": None}),
        encoding="utf-8",
    )
    fatal = "APIStatusError: Error code: 402 - Insufficient Balance"
    (diagnoses / "00002_hr.json").write_text(
        json.dumps({"verified": False, "error": fatal}),
        encoding="utf-8",
    )
    calls = []

    def fail(task):
        record = task[0]["record"]
        calls.append(record)
        return {
            "record": record,
            "status": "failed",
            "runtime_seconds": 0.01,
            "attempts": 1,
            "error": fatal,
            "diagnosis_path": str(diagnoses / f"{record}.json"),
        }

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    monkeypatch.setattr(batch, "_diagnose_one", fail)
    monkeypatch.setattr(batch, "_reusable_feature", lambda *args, **kwargs: True)

    with pytest.raises(RuntimeError, match="circuit breaker"):
        batch.run_diagnosis(
            records,
            tmp_path,
            workers=1,
            model="test-model",
            max_revisions=0,
            record_retries=0,
            thinking=False,
            reasoning_effort="high",
            verbose=False,
            reuse_existing=True,
            retry_failed=True,
            backend="deepseek",
            allow_external_egress=True,
        )

    assert calls == ["00002_hr"]


def test_local_reverification_recovers_safe_pointer_alias(tmp_path):
    record_id = "00001_hr"
    feature = _payload()
    feature["clinical_interpretation"]["artifact_fingerprint"] = "sha256:test"
    feature_path = tmp_path / "features" / f"{record_id}_features.json"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_text(json.dumps(feature), encoding="utf-8")
    diagnostic_fingerprint = (
        EvidenceStore.from_dict(feature).diagnostic_view().fingerprint()
    )
    diagnosis_path = tmp_path / "diagnoses" / f"{record_id}.json"
    diagnosis_path.parent.mkdir()
    diagnosis_path.write_text(
        json.dumps(
            {
                "record_id": record_id,
                "ok": True,
                "verified": False,
                "error": None,
                    "verdict": {
                        "summary": "Independent ECG interpretation.",
                        "ranked_complete_interpretations": [
                            {
                                "rank": 1,
                                "interpretation_type": "PRIMARY",
                                "complete_diagnosis": "First-degree AV delay with limited QT/QTc measurement.",
                                "confidence": "MEDIUM",
                                "basis_codes": ["first_degree_av_delay"],
                                    "key_uncertainty": "Human review against the original waveforms remains required.",
                            }
                        ],
                        "diagnoses": [
                        {
                            "code": "first_degree_av_delay",
                            "statement": "First-degree AV delay",
                            "category": "conduction",
                            "confidence": "MEDIUM",
                            "urgency": "ROUTINE",
                            "evidence": [
                                {
                                    "claim": "Ventricular rate is 62 bpm.",
                                    "value": 62,
                                    "unit": "bpm",
                                    "citations": [
                                        "ev:/global_features/ventricular_rate_bpm"
                                    ],
                                }
                            ],
                            "counterevidence": [],
                            "reasoning": "The measured rate supports the interpretation.",
                        }
                    ],
                    "differential_diagnoses": [],
                    "interval_measurement_contexts": [
                        {
                            "interval": "QT_QTc",
                            "status": "limited",
                            "interval_conclusion": "QT/QTc measurement is limited.",
                            "component_waveform_assessment": (
                                    "T-wave measurements remain available and must be interpreted separately from the QT limitation."
                            ),
                            "residual_evidence": [
                                {
                                        "claim": "QT is not reportable, but T-wave measurements remain available.",
                                    "value": None,
                                    "unit": None,
                                    "citations": [
                                        "ev:/global_features/qt_reportable",
                                        "ev:/representative_leads/I/params/t_amp_mv",
                                    ],
                                }
                            ],
                            "interpretive_impact": (
                                    "QT values cannot be used, but residual repolarization information cannot be ignored."
                            ),
                                "what_would_resolve_it": "Review T-wave endpoints and cross-lead morphology.",
                        }
                    ],
                    "abstentions": [],
                    "quality_assessment": {
                        "interpretability": "limited",
                        "limitations": ["Automated measurements require review"],
                    },
                    "human_review": {
                        "required": True,
                        "reasons": ["Research output requires clinician review"],
                    },
                },
                "verification": {
                    "counts": {"unresolvable_citation": 1}
                },
                "audit": {
                    "agent_protocol": AGENT_PROTOCOL_VERSION,
                    "prompt_fingerprint": PROMPT_FINGERPRINT,
                    "input_fingerprint": diagnostic_fingerprint,
                    "tools": {
                        "provenance": {
                            "visible_by_source": {
                                "chart_briefing": [
                                    "/global_features/heart_rate_bpm",
                                    "/global_features/qt_reportable",
                                    "/representative_leads/I/params/t_amp_mv",
                                ]
                            }
                        },
                        "calls": [],
                    },
                },
                "batch": {},
            }
        ),
        encoding="utf-8",
    )

    summary = batch.reverify_results(
        [{"record": record_id, "reference_available": False}],
        tmp_path,
    )

    assert summary["counts"]["recovered"] == 1
    updated = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    assert updated["verified"] is True
    assert updated["audit"]["local_reverification"]["verdict_edited"] is False
