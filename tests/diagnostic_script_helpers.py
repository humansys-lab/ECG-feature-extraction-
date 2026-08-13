from __future__ import annotations

import json

from ecgagent.agent.protocol import DIAGNOSTIC_DOMAINS
from ecgagent.backends.mock import text_response, tool_response


HEART_RATE = "/global_features/heart_rate_bpm"
T_AMP = "/representative_leads/I/params/t_amp_mv"
T_RELIABILITY = "/quality/I/reliable_for_t"


def _empty_patch(version: int) -> dict:
    return {
        "base_version": version,
        "create_hypotheses": [],
        "evidence_updates": [],
        "domain_updates": [],
        "transitions": [],
        "plan_updates": [],
        "refuting_tests": [],
        "targeted_checks": [],
        "detector_conflicts": [],
        "unresolved": [],
    }


def legacy_phase_patches() -> tuple[dict, dict, dict, dict]:
    """Valid deterministic fixtures for the versioned five-stage protocol."""

    survey = _empty_patch(0)
    survey["domains"] = {
        domain: {
            "status": "assessed",
            "finding": "normal",
            "citations": [f"ev:{HEART_RATE}"],
            "limitations": [],
        }
        for domain in DIAGNOSTIC_DOMAINS
    }
    survey["create_hypotheses"] = [
        {
            "hypothesis_id": "h_sinus",
            "diagnosis_code": "sinus_rhythm",
            "phenotype": "Sinus-rhythm candidate",
            "kind": "diagnosis",
            "domains": ["rhythm_rate", "p_av"],
            "status": "raised",
            "supporting_observations": [
                {"citations": [f"ev:{HEART_RATE}"]}
            ],
            "counterevidence": [],
            "required_evidence": ["Review rhythm and atrial activity"],
            "next_tools": ["get_measurement"],
            "alternative_explanations": ["Other regular rhythms"],
            "alternative_group": "rhythm_origin",
        }
    ]

    hypothesize = _empty_patch(1)
    hypothesize["transitions"] = [
        {
            "hypothesis_id": "h_sinus",
            "from_status": "raised",
            "to_status": "provisional",
            "citations": [f"ev:{HEART_RATE}"],
        }
    ]
    hypothesize["plan_updates"] = [
        {
            "hypothesis_id": "h_sinus",
            "required_evidence": ["Review rhythm"],
            "next_tools": ["get_measurement"],
            "alternative_explanations": ["Other regular rhythms"],
            "alternative_group": "rhythm_origin",
        }
    ]
    hypothesize["targeted_checks"] = [
        {
            "hypothesis_id": "h_sinus",
            "domain": "rhythm_rate",
            "tool": "get_measurement",
            "question": "Review rate evidence",
        }
    ]

    investigate = _empty_patch(2)
    investigate["evidence_updates"] = [
        {
            "hypothesis_id": "h_sinus",
            "evidence_type": "support",
            "citations": [f"ev:{HEART_RATE}"],
        }
    ]
    investigate["domain_updates"] = [
        {
            "domain": "q_st_t_u",
            "status": "limited",
            "finding": "indeterminate",
            "citations": [f"ev:{T_RELIABILITY}"],
            "limitations": ["T-wave review required"],
        }
    ]
    investigate["transitions"] = [
        {
            "hypothesis_id": "h_sinus",
            "from_status": "provisional",
            "to_status": "supported",
            "citations": [f"ev:{HEART_RATE}"],
        }
    ]

    challenge = _empty_patch(3)
    challenge["refuting_tests"] = [
        {
            "hypothesis_id": "h_sinus",
            "would_refute": "An independent view does not support an organized rhythm",
            "citations": [f"ev:{T_AMP}"],
            "test_outcome": "not_refuted",
            "evidence_direction": "supports",
        }
    ]
    return survey, hypothesize, investigate, challenge


def legacy_script_prefix():
    survey, hypothesize, investigate, challenge = legacy_phase_patches()
    return [
        tool_response(("get_diagnostic_overview", {})),
        text_response(json.dumps(survey, ensure_ascii=False)),
        text_response(json.dumps(hypothesize, ensure_ascii=False)),
        tool_response(("get_measurement", {"pointer": HEART_RATE})),
        tool_response(("get_interval_waveform_context", {"interval": "qt"})),
        text_response(json.dumps(investigate, ensure_ascii=False)),
        tool_response(("get_measurement", {"pointer": T_AMP})),
        tool_response(
            ("get_beat_table", {"fields": ["rr_prev_ms", "rr_next_ms"]})
        ),
        text_response(json.dumps(challenge, ensure_ascii=False)),
    ]
