from __future__ import annotations

import json

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
    _fatal_external_error,
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
            "model": "deepseek-v4-flash",
            "thinking": False,
            "reasoning_effort": None,
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
    feature_path.write_text(json.dumps(_payload()), encoding="utf-8")
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
