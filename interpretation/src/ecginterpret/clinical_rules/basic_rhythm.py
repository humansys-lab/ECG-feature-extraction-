from __future__ import annotations

from .config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import RuleEvaluation


_INTERVALS = DEFAULT_DIAGNOSTIC_CONFIG.intervals
_RHYTHM = DEFAULT_DIAGNOSTIC_CONFIG.rhythm


def evaluate_basic_rhythm(context) -> list[RuleEvaluation]:
    metadata = context.features.metadata
    rhythm = metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    atrial = rhythm.get("af_afl_summary", {})
    atrial = atrial if isinstance(atrial, dict) else {}
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    hr = context.global_value("heart_rate_bpm")
    qrs = context.global_value("qrs_ms")
    p_axis = context.global_value("p_axis_deg")
    rr_cv = atrial.get("rr_cv")
    organized_p = atrial.get("organized_p_ratio")
    missing = [
        name
        for name, value in (
            ("global.heart_rate_bpm", hr),
            ("rhythm.rr_cv", rr_cv),
            ("rhythm.organized_p_ratio", organized_p),
        )
        if value is None
    ]
    confounded = bool(
        atrial.get("probable_af")
        or atrial.get("probable_flutter")
        or pacing.get("suppress_further_rhythm_interpretation")
    )
    sinus_source = bool(
        not missing
        and not confounded
        and float(organized_p) >= 0.70
        and p_axis is not None
        and _INTERVALS.sinus_p_axis_min_deg
        <= p_axis
        <= _INTERVALS.sinus_p_axis_max_deg
    )
    if sinus_source and hr is not None:
        if hr < _INTERVALS.adult_bradycardia_bpm:
            code, label = "sinus_bradycardia", "Sinus bradycardia"
        elif hr > _INTERVALS.adult_tachycardia_bpm:
            code, label = "sinus_tachycardia", "Sinus tachycardia"
        else:
            code, label = "sinus_rhythm", "Sinus rhythm"
    else:
        code = label = None
    sinus = RuleEvaluation(
        rule_id="CLIN-RHYTHM-SINUS-01",
        domain="rhythm",
        status=(
            "matched"
            if code
            else "unavailable" if missing or p_axis is None else "not_matched"
        ),
        statement_code=code,
        statement=label,
        severity="normal" if code == "sinus_rhythm" else (
            "observation" if code else "normal"
        ),
        confidence="high" if code else None,
        required_inputs=[
            "global.heart_rate_bpm",
            "global.p_axis_deg",
            "rhythm.rr_cv",
            "rhythm.organized_p_ratio",
        ],
        missing_inputs=missing + ([] if p_axis is not None else ["global.p_axis_deg"]),
        evidence={
            "heart_rate_bpm": hr,
            "qrs_ms": qrs,
            "p_axis_deg": p_axis,
            "rr_cv": rr_cv,
            "organized_p_ratio": organized_p,
            "confounded": confounded,
            "evaluates_code": "sinus_mechanism",
        },
        thresholds={
            "organized_p_ratio_gte": 0.70,
            "p_axis_deg": [
                _INTERVALS.sinus_p_axis_min_deg,
                _INTERVALS.sinus_p_axis_max_deg,
            ],
            "bradycardia_lt_bpm": _INTERVALS.adult_bradycardia_bpm,
            "tachycardia_gt_bpm": _INTERVALS.adult_tachycardia_bpm,
        },
    )

    flutter_review = bool(
        hr is not None
        and qrs is not None
        and rr_cv is not None
        and _RHYTHM.flutter_review_rate_min_bpm
        <= hr
        <= _RHYTHM.flutter_review_rate_max_bpm
        and qrs < _INTERVALS.qrs_wide_ms
        and float(rr_cv) <= 0.08
        and not atrial.get("probable_flutter")
    )
    rate_150 = RuleEvaluation(
        rule_id="CLIN-RHYTHM-AFL150-01",
        domain="rhythm",
        status="matched" if flutter_review else (
            "unavailable"
            if hr is None or qrs is None or rr_cv is None
            else "not_matched"
        ),
        statement_code="exclude_2_to_1_atrial_flutter" if flutter_review else None,
        statement=(
            "Regular narrow-complex rate near 150 bpm; exclude atrial flutter with 2:1 conduction"
            if flutter_review
            else None
        ),
        severity="observation" if flutter_review else "normal",
        confidence="medium" if flutter_review else None,
        priority="P1" if flutter_review else None,
        human_review_required=flutter_review,
        normality_role="supporting",
        missing_inputs=[
            name
            for name, value in (
                ("global.heart_rate_bpm", hr),
                ("global.qrs_ms", qrs),
                ("rhythm.rr_cv", rr_cv),
            )
            if value is None
        ],
        evidence={
            "heart_rate_bpm": hr,
            "qrs_ms": qrs,
            "rr_cv": rr_cv,
            "evaluates_code": "exclude_2_to_1_atrial_flutter",
        },
    )
    return [sinus, rate_150]
