from __future__ import annotations

from ecgagent.agent.loop import AgentResult
from ecgagent.report import render_brief_report, render_human_report
from tests.test_ecgagent_diagnostic import _diagnostic_verdict


def _explainable_verdict() -> dict:
    verdict = _diagnostic_verdict()
    verdict["summary"] = "Current evidence most strongly supports sinus rhythm, but a clinician must still confirm it on the original waveforms."
    verdict["diagnoses"][0].update(
        {
            "statement": "Sinus rhythm",
            "reasoning": (
                "The heart rate and rhythm organization are concordant and support regular sinus activation, "
                "but uncertainty remains because the original waveforms were not reviewed directly."
            ),
        }
    )
    verdict["diagnoses"][0]["evidence"][0]["claim"] = (
        "The measured heart rate of 62 bpm supports the current rhythm interpretation."
    )
    verdict["diagnoses"][0]["counterevidence"] = [
        {
            "claim": "Limited P-wave evidence quality reduces confidence in the rhythm interpretation.",
            "value": None,
            "unit": None,
            "citations": ["ev:/global_features/p_axis_deg"],
        }
    ]
    verdict["differential_diagnoses"] = [
        {
            "code": "ectopic_atrial_rhythm_pattern",
            "statement": "Ectopic atrial rhythm pattern",
            "confidence": "LOW",
            "supporting_evidence": [
                {
                    "claim": "Incomplete P-wave axis measurement does not fully exclude an ectopic origin.",
                    "value": None,
                    "unit": None,
                    "citations": ["ev:/global_features/p_axis_deg"],
                }
            ],
            "counterevidence": [],
            "what_would_resolve_it": "Review P-wave morphology and axis on the original 12-lead ECG",
        }
    ]
    verdict["abstentions"] = [
        {
            "topic": "Exact P-wave origin",
            "reason": "Available P-wave measurements have insufficient reliability",
            "what_would_resolve_it": "Clinician review of the original waveforms",
        }
    ]
    verdict["quality_assessment"] = {
        "interpretability": "limited",
        "limitations": ["Some P-wave measurements can be used only as low-weight evidence"],
    }
    verdict["human_review"] = {
        "required": True,
        "reasons": ["Confirm the rhythm origin against the original waveforms"],
    }
    return verdict


def test_human_report_explains_diagnosis_evidence_uncertainty_and_next_steps():
    report = render_human_report(
        _explainable_verdict(),
        record_id="DX-HUMAN",
        verified=True,
    )

    assert "# ECG Agent Diagnostic Interpretation Report" in report
    assert "passed measurement-citation, reliability-qualification, numeric-logic" in report
    assert "## Top 3 Most Likely Complete ECG Interpretations" in report
    assert report.index("## Top 3 Most Likely Complete ECG Interpretations") < report.index("## Overall Conclusion")
    assert "1. **Sinus rhythm with limited QT/QTc measurement requiring T/U-wave review.**" in report
    assert "Primary complete interpretation; overall confidence: Medium" in report
    assert "## Reporting Levels" in report
    assert "## Major ECG Findings" in report
    assert "Rationale" in report
    assert "Supporting evidence" in report
    assert "Counterevidence or inconsistencies" in report
    assert "ev:/global_features/heart_rate_bpm" in report
    assert "## Differential Diagnoses" in report
    assert "How to resolve" in report
    assert "## Waveform Information Retained When Intervals Are Limited" in report
    assert "Component waveform still shows" in report
    assert "QT is marked non-reportable" in report
    assert "## Data Quality and Interpretation Limitations" in report
    assert report.index("## Data Quality and Interpretation Limitations") < report.index("## Major ECG Findings")
    assert "## Human Review Recommendations" in report
    assert "does not replace clinician waveform review" in report


def test_brief_report_keeps_conclusion_evidence_limits_and_review_actions():
    report = render_brief_report(
        _explainable_verdict(),
        record_id="DX-BRIEF",
        verified=True,
    )

    assert report.startswith("# Brief ECG Diagnostic Report")
    assert "Record: DX-BRIEF" in report
    assert "Result status: Passed deterministic evidence validation" in report
    assert "## Conclusion" in report
    assert "## Confirmed Diagnoses" in report
    assert "**Sinus rhythm**" in report
    assert "Key evidence: The measured heart rate of 62 bpm" in report
    assert "62 bpm" in report
    assert "Main limitation: Limited P-wave evidence quality" in report
    assert "## Important but Unconfirmed" in report
    assert "Ectopic atrial rhythm pattern" in report
    assert "## Main Limitations" in report
    assert "## Human Review Recommendations" in report
    assert "Confirm the rhythm origin against the original waveforms" in report
    assert "ev:/" not in report
    assert len(report) < 2200


def test_brief_report_marks_unverified_and_failed_results_prominently():
    unverified = render_brief_report(
        _explainable_verdict(),
        record_id="DX-UNVERIFIED-BRIEF",
        verified=False,
    )
    failed = render_brief_report(
        None,
        record_id="DX-FAILED-BRIEF",
        error="backend timeout",
    )

    assert "not a confirmed diagnosis" in unverified
    assert "## Unverified Candidates" in unverified
    assert "did not produce a usable structured diagnosis" in failed
    assert "backend timeout" in failed


def test_unverified_human_report_has_a_prominent_warning():
    report = render_human_report(
        _explainable_verdict(),
        record_id="DX-UNVERIFIED",
        verified=False,
    )
    assert "failed; the following content is not a confirmed conclusion" in report
    assert "## Top 3 ECG Interpretation Candidates (Unconfirmed)" in report
    assert "not a confirmed diagnosis" in report
    assert "## Unverified Candidate Interpretations" in report
    assert "## Major ECG Findings" not in report
    assert "deterministic validation confirms" not in report
    assert "cannot be confirmed as consistent with ecgfeat" in report


def test_empty_failure_report_exposes_actual_agent_error():
    report = render_human_report(
        None,
        record_id="DX-FAILED",
        error="phase `investigate` failed the hard state/evidence guard",
    )

    assert "stopped before completing the final structured diagnosis" in report
    assert "phase `investigate` failed" in report
    assert "Agent trace" in report


def test_top_three_renders_integrated_interpretations_not_isolated_findings():
    verdict = _explainable_verdict()
    verdict["diagnoses"] = [
        {
            "code": "medium_first",
            "statement": "中置信度第一项",
            "category": "other",
            "confidence": "MEDIUM",
            "urgency": "NONE",
        },
        {
            "code": "high_first",
            "statement": "高置信度第一项",
            "category": "rhythm",
            "confidence": "HIGH",
            "urgency": "NONE",
        },
        {
            "code": "high_second",
            "statement": "高置信度第二项",
            "category": "conduction",
            "confidence": "HIGH",
            "urgency": "URGENT",
        },
        {
            "code": "medium_second",
            "statement": "中置信度第二项",
            "category": "axis",
            "confidence": "MEDIUM",
            "urgency": "NONE",
        },
    ]
    verdict["differential_diagnoses"][0]["statement"] = "不可进入Top3的鉴别诊断"
    verdict["ranked_complete_interpretations"] = [
        {
            "rank": 1,
            "interpretation_type": "PRIMARY",
            "complete_diagnosis": "窦性心律；心房传导异常；伴非特异性复极改变。",
            "confidence": "MEDIUM",
            "basis_codes": ["high_first", "high_second", "medium_first"],
            "key_uncertainty": "P 波边界稳定性不足。",
        },
        {
            "rank": 2,
            "interpretation_type": "ALTERNATIVE",
            "complete_diagnosis": "窦性心律；P 波异常更可能来自边界伪差；无明确复极异常。",
            "confidence": "LOW",
            "basis_codes": ["high_first"],
            "key_uncertainty": "需要人工复核原始 P 波。",
        },
        {
            "rank": 3,
            "interpretation_type": "ALTERNATIVE",
            "complete_diagnosis": "异位房性心律可能；其余间期与传导未见明确异常。",
            "confidence": "LOW",
            "basis_codes": ["ectopic_atrial_rhythm_pattern"],
            "key_uncertainty": "心房激动来源尚未确认。",
        },
    ]

    report = render_human_report(verdict, record_id="TOP3", verified=True)
    top_section = report.split("## Overall Conclusion", 1)[0]

    assert "窦性心律；心房传导异常；伴非特异性复极改变" in top_section
    assert "P 波异常更可能来自边界伪差" in top_section
    assert "异位房性心律可能；其余间期与传导未见明确异常" in top_section
    assert "高置信度第一项" not in top_section
    assert "高置信度第二项" not in top_section
    assert "中置信度第一项" not in top_section
    assert "不可进入Top3的鉴别诊断" not in top_section
    assert "Key uncertainty" in top_section


def test_human_report_separates_posthoc_knowledge_from_patient_evidence():
    report = render_human_report(
        _explainable_verdict(),
        record_id="KB-REPORT",
        verified=True,
        knowledge_navigation={
            "enabled": True,
            "status": "matched",
            "reference_count": 3,
            "patient_evidence": False,
        },
        knowledge_review={
            "status": "issues_found",
            "summary": "需要重新检查一个混杂因素。",
            "issues": [
                {
                    "type": "confounder",
                    "priority": "HIGH",
                    "topic": "继发性复极改变",
                    "evidence_review_question": "请重新检查QRS与T波方向关系。",
                    "knowledge_refs": ["K1"],
                }
            ],
            "references": [
                {
                    "ref": "K1",
                    "source": "docs/reference.md",
                    "line_start": 10,
                    "line_end": 20,
                    "title": "原发与继发改变",
                }
            ],
            "revision": {"applied": True},
        },
    )

    assert "Medical Knowledge Navigation Before Provisional Diagnosis" in report
    assert "form provisional diagnoses, differential diagnoses and targeted ecgfeat checks" in report
    assert "not patient evidence" in report
    assert "Medical Knowledge Challenge Review" in report
    assert "请重新检查QRS与T波方向关系" in report
    assert "general review references, not patient evidence" in report
    assert "ecgfeat tools were called for review" in report


def test_agent_result_json_carries_the_same_human_readable_report():
    result = AgentResult(
        record_id="DX-PAYLOAD",
        verdict=_explainable_verdict(),
    )
    payload = result.to_dict()

    assert payload["human_report"] == result.human_report
    assert payload["brief_report"] == result.brief_report
    assert "Brief ECG Diagnostic Report" in payload["brief_report"]
    assert "Overall Conclusion" in payload["human_report"]
    assert "Structured measurement: 62 bpm" in payload["human_report"]
    assert "failed; the following content is not a confirmed conclusion" in payload["human_report"]
