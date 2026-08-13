"""Contract tests for the LUDB-133 failure chain.

One record produced five independent defects, and each layer here is one of
them. They are grouped because they only make sense together: a calibration
artefact became an electrode-swap verdict, the swap survived a challenge phase
that never tested it, the claim arguing against it lost its numbers to a
presentation rule, and verification then failed the run for a digit string the
pipeline had injected itself.
"""
from __future__ import annotations

import numpy as np

from ecgagent.agent import diagnostic_prompts
from ecgagent.agent.semantic_guard import validate_diagnosis_measurement_semantics
from ecgagent.evidence.store import EvidenceStore
from ecgagent.verify import VerificationPolicy, verify_structured
from feature_extraction.ecgfeat.quality import (
    build_diagnostic_gate,
    detect_limb_lead_reversal,
)
from feature_extraction.ecgfeat.validation import _assess_amplitude_calibration
from tests.test_ecgagent_tools import _payload


def _limb_leads_violating_einthoven() -> np.ndarray:
    """Twelve leads whose limb equation fails without any electrode swap.

    This is LUDB's normalisation reproduced in miniature: lead I is scaled up
    relative to II and III, so II != I + III in millivolts while every lead
    keeps its own polarity and shape.
    """
    time = np.linspace(0.0, 10.0, 5000)
    beat = np.sin(2.0 * np.pi * 0.83 * time) ** 21
    leads = np.zeros((12, time.size), dtype=float)
    leads[0] = beat * 5.0
    leads[1] = beat * 1.0
    leads[2] = beat * 0.4
    for index in range(3, 12):
        leads[index] = beat * (0.5 + 0.1 * index)
    return leads


def test_einthoven_residual_alone_is_not_an_electrode_swap():
    result = detect_limb_lead_reversal(_limb_leads_violating_einthoven())

    assert result["einthoven_residual_ratio"] > 1.0
    assert result["probable_extremity_reversal"] is False
    assert not any(
        result[key]
        for key in result
        if key.startswith("probable_") and key != "probable_extremity_reversal"
    )


def test_per_lead_normalized_amplitudes_are_detected_and_scoped():
    leads = _limb_leads_violating_einthoven()
    normalized = np.stack([row / np.ptp(row) for row in leads])

    calibration = _assess_amplitude_calibration(normalized)
    assert calibration["per_lead_normalized"] is True
    assert calibration["amplitudes_diagnostic"] is False

    gate = build_diagnostic_gate(
        record_quality={"diagnostic_gate": "pass"},
        duration_sec=10.0,
        n_beats=8,
        beat_groups={1: [0, 1, 2, 3, 4, 5, 6, 7]},
        amplitude_calibration=calibration,
    )
    # Rescaled amplitudes invalidate voltage reasoning, not the recording.
    assert gate["state"] == "partial"
    assert "voltage_chamber_r_progression" in gate["suppressed_domains"]

    assert _assess_amplitude_calibration(leads)["per_lead_normalized"] is False


def test_gate_publishes_criteria_for_a_refutable_flag():
    gate = build_diagnostic_gate(
        record_quality={"diagnostic_gate": "pass"},
        duration_sec=10.0,
        n_beats=8,
        beat_groups={1: [0, 1]},
        limb_reversal={"probable_ra_la": True, "probable_extremity_reversal": True},
    )
    flag = next(
        item
        for item in gate["refutable_flags"]
        if item["reason"] == "probable_limb_lead_reversal"
    )

    assert flag["refutable"] is True
    assert flag["refuted_when_any_fails"] is True
    criteria = {item["id"] for item in flag["criteria"]}
    assert "lead_I_dominantly_negative" in criteria
    assert "aVR_dominantly_positive" in criteria


def _document_with_upright_lead_i() -> dict:
    """Record 133's limb leads: lead I upright, aVR inverted. No swap."""
    return {
        "representative_leads": {
            "I": {"params": {"r_amp_mv": 0.614, "s_amp_mv": -0.110}},
            "aVR": {"params": {"r_amp_mv": 0.071, "s_amp_mv": -0.318}},
        }
    }


def test_limb_reversal_claim_is_rejected_when_its_criteria_fail():
    verdict = {
        "diagnoses": [
            {
                "code": "limb_lead_reversal_suspected",
                "statement": "肢体导联反接（疑似左右臂互换）",
                "reasoning": "I导联主波向下，aVR直立",
                "evidence": [],
                "counterevidence": [],
            }
        ]
    }

    problems = validate_diagnosis_measurement_semantics(
        verdict, _document_with_upright_lead_i()
    )

    assert any("diagnosis_criteria_conflict" in problem for problem in problems)
    assert any("lead I is dominantly positive" in problem for problem in problems)


def test_limb_reversal_claim_stands_when_its_criteria_hold():
    verdict = {
        "diagnoses": [
            {
                "code": "limb_lead_reversal_suspected",
                "statement": "肢体导联反接",
                "evidence": [],
                "counterevidence": [],
            }
        ]
    }
    document = {
        "representative_leads": {
            "I": {"params": {"r_amp_mv": 0.05, "s_amp_mv": -0.62}},
            "aVR": {"params": {"r_amp_mv": 0.41, "s_amp_mv": -0.03}},
        }
    }

    problems = validate_diagnosis_measurement_semantics(verdict, document)

    assert not any("diagnosis_criteria_conflict" in problem for problem in problems)


def test_comparative_claim_must_use_two_atomic_evidence_items():
    """Both sides survive, but no compound item can hide one from validation."""
    verdict = {
        "diagnoses": [
            {
                "code": "limb_lead_reversal_suspected",
                "statement": "疑似肢体导联反接",
                "evidence": [
                    {
                        "claim": "I导联S波 -0.11 mV，R波 0.614 mV",
                        "value": -0.11,
                        "unit": "mV",
                        "citations": ["ev:/representative_leads/I/params/s_amp_mv"],
                    }
                ],
                "counterevidence": [],
            }
        ]
    }

    problems = diagnostic_prompts.validate_verdict(verdict)
    assert any("multiple patient measurements" in problem for problem in problems)

    verdict["diagnoses"][0]["evidence"] = [
        {
            "claim": "I导联S波为 -0.11 mV",
            "value": -0.11,
            "unit": "mV",
            "citations": ["ev:/representative_leads/I/params/s_amp_mv"],
        },
        {
            "claim": "I导联R波为 0.614 mV",
            "value": 0.614,
            "unit": "mV",
            "citations": ["ev:/representative_leads/I/params/r_amp_mv"],
        },
    ]
    normalized, _ = diagnostic_prompts.normalize_verdict(verdict)
    claims = [item["claim"] for item in normalized["diagnoses"][0]["evidence"]]
    assert any("-0.11" in claim for claim in claims)
    assert any("0.614" in claim for claim in claims)
    assert all("相应测量值" not in claim for claim in claims)


def test_challenge_sanitizer_removes_stale_support_transition_and_test():
    state = {
        "base_version": 3,
        "create_hypotheses": [],
        "evidence_updates": [],
        "domain_updates": [],
        "transitions": [
            {
                "hypothesis_id": "limb_lead_reversal",
                "from_status": "provisional",
                "to_status": "supported",
                "citations": ["ev:/old/evidence"],
            }
        ],
        "plan_updates": [],
        "refuting_tests": [
            {
                "hypothesis_id": "limb_lead_reversal",
                "would_refute": "limb-lead polarity remains physiologic",
                "citations": ["ev:/old/evidence"],
                "test_outcome": "not_refuted",
                "evidence_direction": "supports",
            }
        ],
        "targeted_checks": [],
        "detector_conflicts": [],
        "unresolved": [],
    }

    sanitized, notes = diagnostic_prompts.sanitize_phase_state(
        "challenge",
        state,
        phase_citations=("ev:/representative_leads/I/params/r_amp_mv",),
    )

    assert sanitized["transitions"] == []
    assert sanitized["refuting_tests"] == []
    assert len(notes) == 2


def test_challenge_sanitizer_keeps_current_evidence_transition_and_test():
    state = {
        "base_version": 3,
        "create_hypotheses": [],
        "evidence_updates": [],
        "domain_updates": [],
        "transitions": [
            {
                "hypothesis_id": "sinus_bradycardia",
                "from_status": "provisional",
                "to_status": "supported",
                "citations": ["ev:/global_features/heart_rate_bpm"],
            }
        ],
        "plan_updates": [],
        "refuting_tests": [
            {
                "hypothesis_id": "sinus_bradycardia",
                "would_refute": "rate is not below the general bradycardia cutoff",
                "citations": ["ev:/global_features/heart_rate_bpm"],
                "test_outcome": "not_refuted",
                "evidence_direction": "supports",
            }
        ],
        "targeted_checks": [],
        "detector_conflicts": [],
        "unresolved": [],
    }

    sanitized, notes = diagnostic_prompts.sanitize_phase_state(
        "challenge",
        state,
        phase_citations=("ev:/global_features/heart_rate_bpm",),
    )

    assert len(sanitized["transitions"]) == 1
    assert len(sanitized["refuting_tests"]) == 1
    assert notes == []


def _store_with_unrounded_rate() -> EvidenceStore:
    """A store whose heart rate is not a round number, as real ones are not."""
    payload = _payload()
    payload["global_features"]["heart_rate_bpm"] = 50.08347245409015
    return EvidenceStore.from_dict(payload, record_id="TEST-DISPLAY")


def _open_policy() -> VerificationPolicy:
    return VerificationPolicy.for_structured(require_provenance=False)


def test_value_written_as_the_tool_displayed_it_verifies():
    """50 bpm is the display of 50.083... bpm, not an unsupported number."""
    verdict = {
        "diagnoses": [
            {
                "evidence": [
                    {
                        "claim": "背景心室率为50 bpm",
                        "value": 50,
                        "unit": "bpm",
                        "citations": ["ev:/global_features/heart_rate_bpm"],
                    }
                ]
            }
        ]
    }

    report = verify_structured(
        verdict, _store_with_unrounded_rate(), None, _open_policy()
    )

    assert report.passed, report.summary()
    assert "unsupported_number" not in report.counts()


def test_supporting_value_is_verified_against_its_own_pointer():
    supported = {
        "diagnoses": [
            {
                "evidence": [
                    {
                        "claim": "I导联 R 波 0.74 mV 大于 V2 的 0.13 mV",
                        "value": 0.74,
                        "unit": "mV",
                        "citations": ["ev:/representative_leads/I/params/r_amp_mv"],
                        "supporting_values": [
                            {
                                "value": 0.13,
                                "unit": "mV",
                                "citation": "ev:/representative_leads/V2/params/r_amp_mv",
                            }
                        ],
                    }
                ]
            }
        ]
    }
    fabricated = {
        "diagnoses": [
            {
                "evidence": [
                    {
                        "claim": "I导联 R 波 0.74 mV 大于 V2 的 0.42 mV",
                        "value": 0.74,
                        "unit": "mV",
                        "citations": ["ev:/representative_leads/I/params/r_amp_mv"],
                        "supporting_values": [
                            {
                                "value": 0.42,
                                "unit": "mV",
                                "citation": "ev:/representative_leads/V2/params/r_amp_mv",
                            }
                        ],
                    }
                ]
            }
        ]
    }

    store = _store_with_unrounded_rate()
    assert verify_structured(supported, store, None, _open_policy()).passed
    wrong = verify_structured(fabricated, store, None, _open_policy())
    assert not wrong.passed
    assert any(item.code == "value_mismatch" for item in wrong.blocking)
