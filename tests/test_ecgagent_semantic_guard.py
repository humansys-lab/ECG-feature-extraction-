from __future__ import annotations

import copy
import json

from ecgagent.agent import diagnostic_prompts
from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.agent.semantic_guard import (
    validate_diagnosis_measurement_semantics,
)
from ecgagent.backends.mock import ScriptedBackend, text_response, tool_response
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_diagnostic import _diagnostic_verdict
from tests.test_ecgagent_tools import _payload
from tests.diagnostic_script_helpers import legacy_script_prefix


def _positive_diagnosis(
    code: str,
    *,
    category: str,
    value: float,
    unit: str,
    citation: str,
    reasoning: str = "测量支持该结论。",
) -> dict:
    return {
        "code": code,
        "statement": code,
        "category": category,
        "confidence": "MEDIUM",
        "urgency": "NONE",
        "evidence": [
            {
                "claim": f"测量值为{value:g} {unit}。",
                "value": value,
                "unit": unit,
                "citations": [citation],
            }
        ],
        "counterevidence": [],
        "reasoning": reasoning,
    }


def _guard(code: str, diagnosis: dict, **globals_: object) -> list[str]:
    payload = _payload()
    payload["global_features"].update(globals_)
    verdict = _diagnostic_verdict()
    verdict["diagnoses"] = [diagnosis]
    return validate_diagnosis_measurement_semantics(verdict, payload)


def test_false_chinese_numeric_comparisons_are_blocked() -> None:
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["reasoning"] = (
        "心率63 bpm低于正常范围下限（通常为60 bpm）；"
        "PR间期152 ms大于正常上限（通常为200 ms）。"
    )

    problems = validate_diagnosis_measurement_semantics(verdict, None)

    assert len([item for item in problems if "false numeric comparison" in item]) == 2
    assert any("63" in item and "60" in item for item in problems)
    assert any("152" in item and "200" in item for item in problems)


def test_true_numeric_comparisons_are_not_blocked() -> None:
    verdict = _diagnostic_verdict()
    verdict["diagnoses"][0]["reasoning"] = (
        "心率59 bpm低于60 bpm；PR间期201 ms大于200 ms。"
    )

    assert validate_diagnosis_measurement_semantics(verdict, None) == []


def test_adult_bradycardia_at_or_above_60_is_blocked() -> None:
    diagnosis = _positive_diagnosis(
        "bradycardia",
        category="rate",
        value=63,
        unit="bpm",
        citation="ev:/global_features/heart_rate_bpm",
    )

    problems = _guard("bradycardia", diagnosis, heart_rate_bpm=63.0)
    assert any("bradycardia requires a rate below 60 bpm" in item for item in problems)

    assert not _guard("bradycardia", diagnosis, heart_rate_bpm=59.0)


def test_pediatric_rate_is_not_forced_through_adult_cutoff() -> None:
    payload = _payload()
    payload["metadata"]["patient_meta"]["age"] = 12
    payload["global_features"]["heart_rate_bpm"] = 63.0
    verdict = _diagnostic_verdict()
    verdict["diagnoses"] = [
        _positive_diagnosis(
            "bradycardia",
            category="rate",
            value=63,
            unit="bpm",
            citation="ev:/global_features/heart_rate_bpm",
        )
    ]

    assert validate_diagnosis_measurement_semantics(verdict, payload) == []


def test_first_degree_av_delay_requires_pr_above_200_ms() -> None:
    diagnosis = _positive_diagnosis(
        "first_degree_av_delay",
        category="conduction",
        value=152,
        unit="ms",
        citation="ev:/global_features/pr_ms",
    )

    problems = _guard("first_degree_av_delay", diagnosis, pr_ms=152.0)
    assert any("requires PR above 200 ms" in item for item in problems)

    assert not _guard("first_degree_av_delay", diagnosis, pr_ms=201.0)


def test_definite_first_degree_av_delay_is_blocked_when_pr_is_missing() -> None:
    diagnosis = _positive_diagnosis(
        "first_degree_av_delay",
        category="conduction",
        value=152,
        unit="ms",
        citation="ev:/global_features/pr_ms",
    )

    problems = _guard("first_degree_av_delay", diagnosis, pr_ms=None)
    assert any("did not produce a PR interval" in item for item in problems)


def test_short_pr_boundary_is_enforced() -> None:
    diagnosis = _positive_diagnosis(
        "short_pr_interval",
        category="interval",
        value=120,
        unit="ms",
        citation="ev:/global_features/pr_ms",
    )

    assert any(
        "requires PR below 120 ms" in item
        for item in _guard("short_pr_interval", diagnosis, pr_ms=120.0)
    )
    assert not _guard("short_pr_interval", diagnosis, pr_ms=119.0)


def test_bundle_guard_uses_widest_global_qrs_estimate() -> None:
    diagnosis = _positive_diagnosis(
        "right_bundle_branch_block",
        category="conduction",
        value=108,
        unit="ms",
        citation="ev:/global_features/qrs_ms",
    )

    assert not _guard(
        "right_bundle_branch_block",
        diagnosis,
        qrs_ms=108.0,
        qrs_wide_ms=124.0,
    )
    assert any(
        "too narrow" in item
        for item in _guard(
            "right_bundle_branch_block",
            diagnosis,
            qrs_ms=98.0,
            qrs_wide_ms=102.0,
        )
    )


def test_atrial_tachycardia_cannot_use_residual_cycle_as_atrial_rate() -> None:
    payload = _payload()
    payload["global_features"]["atrial_rate_bpm"] = 64.0
    payload["rhythm_inputs"] = {
        "background": {"background_atrial_rate_bpm": 64.0},
        "af_afl": {
            "qrst_subtraction_quality": {"dominant_cycle_ms": 360.0}
        },
    }
    diagnosis = _positive_diagnosis(
        "atrial_tachycardia",
        category="rhythm",
        value=360,
        unit="ms",
        citation=(
            "ev:/rhythm_inputs/af_afl/qrst_subtraction_quality/"
            "dominant_cycle_ms"
        ),
        reasoning="残差信号主导周期被解释为房性心动过速。",
    )
    verdict = _diagnostic_verdict()
    verdict["diagnoses"] = [diagnosis]

    problems = validate_diagnosis_measurement_semantics(verdict, payload)

    assert any("direct measured atrial rate" in item for item in problems)
    assert any("residual dominant cycle" in item.lower() for item in problems)


def test_atrial_tachycardia_requires_a_cited_direct_atrial_rate() -> None:
    diagnosis = _positive_diagnosis(
        "atrial_tachycardia",
        category="rhythm",
        value=145,
        unit="bpm",
        citation="ev:/global_features/atrial_rate_bpm",
    )

    assert not _guard(
        "atrial_tachycardia",
        diagnosis,
        atrial_rate_bpm=145.0,
    )

    diagnosis["evidence"][0]["citations"] = [
        "ev:/rhythm_inputs/af_afl/qrst_subtraction_quality/dominant_cycle_ms"
    ]
    assert any(
        "lacks a cited direct atrial-rate measurement" in item
        for item in _guard(
            "atrial_tachycardia",
            diagnosis,
            atrial_rate_bpm=145.0,
        )
    )


def test_variable_av_conduction_requires_validated_dropped_atrial_activity() -> None:
    payload = _payload()
    payload["rhythm_inputs"] = {
        "av_block": {
            "evidence": {
                "atrial_events_per_rr": [3, 3, 2, 2],
                "localized_atrial_event_excess": False,
                "constant_multiple_atrial_events_requires_validation": True,
                "dropped_p_interval_indices": [],
                "dropped_p_evidence": False,
            }
        }
    }
    diagnosis = _positive_diagnosis(
        "second_degree_av_block_pattern",
        category="conduction",
        value=62,
        unit="bpm",
        citation="ev:/global_features/heart_rate_bpm",
        reasoning="存在2:1至3:2可变性房室传导。",
    )
    verdict = _diagnostic_verdict()
    verdict["diagnoses"] = [diagnosis]

    problems = validate_diagnosis_measurement_semantics(verdict, payload)

    assert any("lacks validated non-conducted atrial activity" in item for item in problems)
    assert any("positively asserts a conduction ratio" in item for item in problems)


def test_retrograde_candidate_stream_is_not_independent_confirmation() -> None:
    payload = _payload()
    for row in payload["beat_features"]:
        row["p_localization_retrograde"] = False
    payload["rhythm_inputs"] = {
        "p_events": [
            {
                "association_type": "retrograde",
                "confidence": 1.0,
                "source": "independent_atrial_event_stream",
            }
        ]
    }
    diagnosis = _positive_diagnosis(
        "sinus_rhythm",
        category="rhythm",
        value=62,
        unit="bpm",
        citation="ev:/global_features/heart_rate_bpm",
        reasoning="同时存在逆传心房激动。",
    )
    verdict = _diagnostic_verdict()
    verdict["diagnoses"] = [diagnosis]

    problems = validate_diagnosis_measurement_semantics(verdict, payload)

    assert any("positively asserts retrograde atrial activation" in item for item in problems)


def test_nonspecific_ivcd_requires_qrs_prolongation_not_notching_alone() -> None:
    diagnosis = _positive_diagnosis(
        "nonspecific_ivcd",
        category="conduction",
        value=110,
        unit="ms",
        citation="ev:/global_features/qrs_wide_ms",
        reasoning="全导联QRS顿挫，因此诊断非特异性室内传导延迟。",
    )

    problems = _guard(
        "nonspecific_ivcd",
        diagnosis,
        qrs_ms=100.0,
        qrs_wide_ms=110.0,
    )
    assert any("notching or fragmentation alone" in item for item in problems)

    assert not _guard(
        "nonspecific_ivcd",
        diagnosis,
        qrs_ms=112.0,
        qrs_wide_ms=116.0,
    )


def test_summary_cannot_bypass_positive_diagnosis_evidence_gates() -> None:
    payload = _payload()
    payload["global_features"].update(
        {"atrial_rate_bpm": 64.0, "qrs_ms": 100.0, "qrs_wide_ms": 110.0}
    )
    payload["rhythm_inputs"] = {
        "av_block": {
            "evidence": {
                "localized_atrial_event_excess": False,
                "constant_multiple_atrial_events_requires_validation": True,
                "dropped_p_interval_indices": [],
                "dropped_p_evidence": False,
            }
        }
    }
    verdict = _diagnostic_verdict()
    verdict["summary"] = (
        "房性心动过速伴2:1可变性房室传导及逆传心房激动；"
        "非特异性室内传导延迟。"
    )

    problems = validate_diagnosis_measurement_semantics(verdict, payload)

    assert any("summary` asserts atrial tachycardia" in item for item in problems)
    assert any("summary` positively asserts an AV" in item for item in problems)
    assert any("summary` positively asserts retrograde" in item for item in problems)
    assert any("summary` asserts nonspecific IVCD" in item for item in problems)


def test_summary_allows_explicitly_unconfirmed_atrial_tachycardia_differential() -> None:
    payload = _payload()
    verdict = {
        "summary": (
            "本次心电图未确认明确的阳性诊断；可能为房性心动过速，"
            "但尚未证实，需作为鉴别诊断。"
        ),
        "diagnoses": [],
    }

    problems = validate_diagnosis_measurement_semantics(verdict, payload)

    assert not any("summary` asserts atrial tachycardia" in item for item in problems)


def test_summary_allows_explicitly_unconfirmed_ivcd_differential() -> None:
    payload = _payload()
    verdict = {
        "summary": (
            "ECG conclusion: Left axis deviation. Unconfirmed or limited: "
            "Probable nonspecific intraventricular conduction delay (unconfirmed)."
        ),
        "diagnoses": [],
    }

    problems = validate_diagnosis_measurement_semantics(verdict, payload)

    assert not any("summary` asserts nonspecific IVCD" in item for item in problems)


def test_qt_guard_only_uses_reliable_reportable_qtc() -> None:
    diagnosis = _positive_diagnosis(
        "short_qt",
        category="interval",
        value=407,
        unit="ms",
        citation="ev:/global_features/qtc_bazett_ms",
    )
    reliable = _guard(
        "short_qt",
        diagnosis,
        qt_reportable=True,
        qt_reliability="reliable",
        qtc_bazett_ms=407.0,
        qtc_fridericia_ms=402.0,
    )
    assert any("cannot support a positive short-QT" in item for item in reliable)

    unavailable = _guard(
        "short_qt",
        diagnosis,
        qt_reportable=False,
        qt_reliability="unreliable",
        qtc_bazett_ms=407.0,
        qtc_fridericia_ms=402.0,
    )
    assert unavailable == []


def test_diagnostic_contract_includes_semantic_guard() -> None:
    payload = _payload()
    payload["global_features"]["pr_ms"] = 152.0
    verdict = _diagnostic_verdict()
    verdict["diagnoses"] = [
        _positive_diagnosis(
            "first_degree_av_delay",
            category="conduction",
            value=152,
            unit="ms",
            citation="ev:/global_features/pr_ms",
        )
    ]

    problems = diagnostic_prompts.validate_verdict(
        verdict,
        evidence_document=payload,
    )
    assert any("diagnosis_measurement_conflict" in item for item in problems)


def test_semantic_failure_forces_ecgfeat_reread_before_revision() -> None:
    payload = _payload()
    payload["global_features"]["heart_rate_bpm"] = 63.0
    store = EvidenceStore.from_dict(payload, record_id="SEMANTIC")
    invalid = _diagnostic_verdict()
    invalid["diagnoses"] = [
        _positive_diagnosis(
            "bradycardia",
            category="rate",
            value=63,
            unit="bpm",
            citation="ev:/global_features/heart_rate_bpm",
            reasoning="心率63 bpm低于60 bpm，因此为心动过缓。",
        )
    ]
    repaired = copy.deepcopy(_diagnostic_verdict())
    repaired["diagnoses"][0]["evidence"][0].update(
        {"claim": "The measured heart rate is 63 bpm.", "value": 63}
    )
    repaired_patch = {
        "base_candidate_hash": ECGDiagnosticAgent._candidate_hash(invalid),
        "changed_sections": ["diagnoses"],
        "replacement_sections": {"diagnoses": repaired["diagnoses"]},
    }

    backend = ScriptedBackend(
        script=[
            *legacy_script_prefix(),
            text_response(json.dumps(invalid)),
            tool_response(
                (
                    "get_measurement",
                    {"pointer": "/global_features/heart_rate_bpm"},
                )
            ),
            text_response(json.dumps(repaired_patch)),
        ]
    )

    result = ECGDiagnosticAgent(
        store=store,
        backend=backend,
        max_revisions=2,
        knowledge_challenge=False,
    ).run()

    assert result.ok and result.verified, result.summary()
    assert result.revisions == 1
    assert result.phases[-1].key == "revise"
    assert result.phases[-1].tool_calls == 1
    assert result.verdict["diagnoses"][0]["code"] == "sinus_rhythm"
