from __future__ import annotations

from typing import Any, Dict, List, Optional

from .statement_engine import CandidateStatement, resolve_statement_candidates


def _statement_dicts(statements: Any) -> List[Dict[str, Any]]:
    if not isinstance(statements, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in statements:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        candidate.setdefault("category", "rhythm")
        candidate.setdefault("severity", 1)
        candidate.setdefault("source", "rhythm")
        candidate.setdefault("evidence", {})
        out.append(candidate)
    return out


def _af_afl_candidates(
    af_afl_summary: Optional[Dict[str, object]],
    atrial_residual: Optional[Dict[str, object]],
) -> List[CandidateStatement]:
    summary = af_afl_summary if isinstance(af_afl_summary, dict) else {}
    residual = atrial_residual if isinstance(atrial_residual, dict) else {}
    validated = bool(residual.get("validated_qrst_subtraction", False))
    unavailable_inputs = [] if validated else ["qrst_subtraction_not_validated"]
    candidates: List[CandidateStatement] = []
    if bool(summary.get("probable_af", False)):
        candidates.append(CandidateStatement(
            code="probable_af",
            text="probable atrial fibrillation",
            category="af_afl",
            severity=3,
            confidence=summary.get("confidence") if isinstance(summary.get("confidence"), float) else None,
            evidence={
                "rr_cv": summary.get("rr_cv"),
                "organized_p_ratio": summary.get("organized_p_ratio"),
                "qrst_subtraction": {
                    "validated": validated,
                    "reason": residual.get("reason"),
                    "template_correlation": residual.get("template_correlation"),
                    "template_residual_rms_ratio": residual.get("template_residual_rms_ratio"),
                },
            },
            source="rhythm",
            required_inputs=["qrst_subtraction"],
            unavailable_inputs=list(unavailable_inputs),
        ))
    flutter_confidence = summary.get("F_wave_confidence", summary.get("flutter_wave_confidence"))
    try:
        flutter_conf = float(flutter_confidence)
    except (TypeError, ValueError):
        flutter_conf = 0.0
    F_wave_validated = bool(
        residual.get("F_wave_morphology_validated")
        or (
            validated
            and summary.get("F_wave_multilead_consensus")
        )
    )
    if bool(summary.get("probable_flutter", False)):
        candidates.append(CandidateStatement(
            code="atrial_flutter",
            text="atrial flutter pattern with organized multilead F waves",
            category="af_afl",
            severity=3,
            confidence=flutter_conf,
            evidence={
                "flutter_wave_confidence": flutter_conf,
                "F_wave_multilead_consensus": summary.get("F_wave_multilead_consensus"),
                "F_wave_rate_bpm": residual.get("F_wave_rate_bpm"),
                "dominant_cycle_ms": residual.get("dominant_cycle_ms"),
                "repetitiveness": residual.get("repetitiveness"),
                "stability": residual.get("stability"),
                "qrst_subtraction": {
                    "validated": validated,
                    "reason": residual.get("reason"),
                    "template_correlation": residual.get("template_correlation"),
                    "template_residual_rms_ratio": residual.get("template_residual_rms_ratio"),
                },
            },
            source="rhythm",
            required_inputs=["qrst_subtraction", "multilead_F_wave_morphology"],
            unavailable_inputs=(
                list(unavailable_inputs)
                + ([] if F_wave_validated else ["multilead_F_wave_morphology_not_validated"])
            ),
        ))
    if bool(summary.get("af_afl_indeterminate", False)):
        candidates.append(CandidateStatement(
            code="af_afl_indeterminate",
            text="atrial tachyarrhythmia; AF versus atrial flutter indeterminate",
            category="af_afl",
            severity=3,
            confidence=(
                float(summary.get("diagnostic_confidence"))
                if isinstance(summary.get("diagnostic_confidence"), (int, float))
                else None
            ),
            evidence={
                "rr_cv": summary.get("rr_cv"),
                "organized_p_ratio": summary.get("organized_p_ratio"),
                "f_wave_confidence": summary.get("f_wave_confidence"),
                "F_wave_confidence": summary.get("F_wave_confidence"),
            },
            source="rhythm",
        ))
    return candidates


def build_rhythm_statement_candidates(
    *,
    rhythm_summary: Dict[str, object],
    preexcitation: Dict[str, object],
    pacing_context: Dict[str, object],
    availability: Dict[str, object],
    af_afl_summary: Optional[Dict[str, object]] = None,
    atrial_residual: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    availability = availability if isinstance(availability, dict) else {}
    stop = bool(pacing_context.get("suppress_further_rhythm_interpretation"))
    candidates: List[Any] = []
    if preexcitation.get("wpw_pattern") and not stop:
        candidates.append(CandidateStatement(
            code="wpw_pattern",
            text="preexcitation / WPW pattern",
            category="preexcitation",
            severity=3,
            confidence=1.0,
            evidence=preexcitation,
            source="rhythm",
            bypasses=["mi", "hypertrophy", "st_t", "pediatric_morphology"],
        ))
    elif pacing_context.get("continuous_pacing"):
        candidates.append(CandidateStatement(
            code="paced_rhythm",
            text="paced rhythm",
            category="pacing",
            severity=3,
            confidence=1.0,
            evidence=pacing_context,
            source="rhythm",
        ))
    else:
        candidates.extend(_statement_dicts(rhythm_summary.get("statements", [])))
        primary = rhythm_summary.get("primary_statement")
        if primary and not any(candidate.get("code") == primary for candidate in candidates if isinstance(candidate, dict)):
            candidates.append({
                "code": primary,
                "category": "rhythm",
                "severity": 1,
                "confidence": 1.0,
                "evidence": {},
                "source": "rhythm",
            })

    candidates.extend(_af_afl_candidates(af_afl_summary, atrial_residual))
    resolution = resolve_statement_candidates(
        candidates,
        context={
            "preexcitation": preexcitation,
            "pacing_context": pacing_context,
            "availability": availability,
            "af_afl": af_afl_summary if isinstance(af_afl_summary, dict) else {},
        },
    ).to_dict()
    final_codes = [str(item.get("code")) for item in resolution["final_statements"]]
    primary_statement = final_codes[0] if final_codes else rhythm_summary.get("primary_statement")
    if preexcitation.get("wpw_pattern") and not stop:
        primary_statement = "wpw_pattern"
    elif pacing_context.get("continuous_pacing"):
        primary_statement = "paced_rhythm"

    bypass_remaining = bool(preexcitation.get("wpw_pattern") and not stop) or bool(
        pacing_context.get("continuous_pacing")
    )
    additional = [
        code for code in final_codes
        if code != primary_statement
    ]
    if not additional:
        additional = list(rhythm_summary.get("additional_statements", []) or [])

    result = {
        "available": True,
        "availability": availability,
        "primary_statement": primary_statement,
        "additional_statements": additional,
        "statements": resolution["candidates"],
        "stop_further_interpretation": stop,
        "bypass_remaining_algorithm": bypass_remaining,
    }
    result.update({
        "candidates": resolution["candidates"],
        "final": resolution["final"],
        "final_statements": resolution["final_statements"],
        "suppressed": resolution["suppressed"],
        "bypassed": resolution["bypassed"],
        "unavailable": resolution["unavailable"],
    })
    return result
