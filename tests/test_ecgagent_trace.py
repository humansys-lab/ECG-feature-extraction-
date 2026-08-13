from __future__ import annotations

from ecgagent.trace import TRACE_FORMAT_VERSION, render_agent_trace, trace_filename


def test_trace_renderer_preserves_full_tool_result_and_runtime_audit() -> None:
    payload = {
        "record_id": "TRACE/001",
        "ok": True,
        "verified": True,
        "revisions": 1,
        "knowledge_revisions": 0,
        "phases": [
            {
                "phase": "survey",
                "turns": 2,
                "tool_calls": 1,
                "prefetched_tool_calls": 1,
                "rejected_calls": 0,
                "elapsed_s": 1.25,
                "stop_reason": "end_turn",
                "text": "已完成十个诊断域盘点。",
            },
            {
                "phase": "hypothesize",
                "turns": 1,
                "tool_calls": 0,
                "rejected_calls": 0,
                "elapsed_s": 0.5,
                "stop_reason": "end_turn",
                "phase_guard_passed": True,
                "phase_sanitizations": [
                    "hypothesis h1 transition removed: no new phase evidence"
                ],
                "text": "暂定诊断，并计划针对性检查 T/U 图。",
            },
        ],
        "knowledge_navigation": {
            "enabled": True,
            "status": "matched",
            "reference_count": 1,
            "patient_evidence": False,
            "references": [
                {
                    "ref": "N1",
                    "category": "diagnostic_reference",
                    "source": "docs/reference.md",
                    "line_start": 10,
                    "line_end": 20,
                    "title": "T波鉴别",
                    "score": 3.2,
                }
            ],
        },
        "verification": {
            "passed": True,
            "n_claims": 2,
            "n_citations": 3,
            "n_quantities": 1,
            "n_supported_quantities": 1,
            "counts": {},
            "findings": [],
        },
        "audit": {
            "agent_protocol": "ecgagent.diagnostic.test",
            "prompt_fingerprint": "sha256:prompt",
            "input_fingerprint": "sha256:input",
            "tools": {"whitelist_size": 2, "calls": []},
            "model": {
                "backend": "mock",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        },
        "batch": {"model": "test-model", "runtime_seconds": 2.0},
        "verdict": {
            "summary": "ECG结论：测试诊断。",
            "diagnoses": [{"code": "sinus_rhythm", "statement": "窦性心律"}],
            "differential_diagnoses": [],
        },
    }
    trace = render_agent_trace(
        payload,
        briefing_text="Neutral briefing with quality and available bundles.",
        intermediate_contexts=[
            {
                "stage": "knowledge_navigation_after_survey",
                "text": "INTERIM KNOWLEDGE CARD N1",
            }
        ],
        tool_calls=[
            {
                "tool": "get_morphology_map",
                "args": {"profile": "t_u"},
                "ok": True,
                "citations": ["/representative_leads/II/params/t_amp_mv"],
                "visible_citations": [
                    "/representative_leads/II/params/t_amp_mv"
                ],
                "elapsed_ms": 4.5,
                "phase": "survey",
                "result_summary": "short",
                "result_text": "FULL MODEL-VISIBLE T/U RESULT\nlead II: t_amp_mv=0.12",
            }
        ],
    )

    assert TRACE_FORMAT_VERSION in trace
    assert "Provisional diagnoses and targeted-check plan" in trace
    assert "Neutral briefing with quality" in trace
    assert "INTERIM KNOWLEDGE CARD N1" in trace
    assert "FULL MODEL-VISIBLE T/U RESULT" in trace
    assert "ev:/representative_leads/II/params/t_amp_mv" in trace
    assert "input_tokens" in trace
    assert "1 (prefetched 1)" in trace
    assert "pass; sanitized=1" in trace
    assert "vendor-hidden internal reasoning fields" in trace


def test_trace_filename_sanitizes_record_id() -> None:
    assert trace_filename("folder/05490 hr") == "folder_05490_hr_agent_trace.md"
