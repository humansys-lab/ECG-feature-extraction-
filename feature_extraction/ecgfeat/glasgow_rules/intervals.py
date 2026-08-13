from __future__ import annotations

import math
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple

from .context import GlasgowContext
from .models import RuleEvaluation, RuleSpec


_PR_SOURCE = "Physician's Guide section 6.1, PDF page 16"
_QT_SOURCE = "Physician's Guide section 6.2, PDF pages 17-18"


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def qtc_values(qt_ms: float, hr_bpm: float) -> Dict[str, float]:
    qt = float(qt_ms)
    heart_rate = float(hr_bpm)
    if not isfinite(qt) or not isfinite(heart_rate) or qt <= 0.0 or heart_rate <= 0.0:
        raise ValueError("QT and heart rate must be finite positive values")
    rr_s = 60.0 / heart_rate
    return {
        "bazett_ms": qt / math.sqrt(rr_s),
        "fridericia_ms": qt / math.pow(rr_s, 1.0 / 3.0),
        "hodges_ms": qt + 1.75 * (heart_rate - 60.0),
        "framingham_ms": qt + 154.0 * (1.0 - rr_s),
    }


def _pr_omitted_by(context: GlasgowContext) -> List[str]:
    reasons: List[str] = []
    if not bool(context.rhythm.get("p_wave_flag")):
        reasons.append("p_wave_flag_not_set")
    if not bool(context.rhythm.get("sinus_rhythm")):
        reasons.append("non_sinus_rhythm")
    if bool(context.rhythm.get("wpw_pattern")):
        reasons.append("wpw_pattern")
    return reasons


def _age_fidelity(context: GlasgowContext, *, uses_days: bool) -> tuple[str, Optional[Dict[str, Any]]]:
    if context.patient.age_missing:
        return (
            "existing_dxl_approximation",
            {
                "source": "existing_dxl",
                "reason": "age_missing_adult_fallback",
                "original_rule_reproducible": False,
            },
        )
    if uses_days and context.patient.age_days_source == "derived_from_age_years":
        return (
            "existing_dxl_approximation",
            {
                "source": "existing_dxl",
                "reason": "age_days_derived_from_age_years",
                "original_rule_reproducible": False,
            },
        )
    return "glasgow_explicit", None


def _pr_limits(context: GlasgowContext) -> Dict[str, float]:
    # NOTE ON CROSS-ENGINE DIVERGENCE: ../interpret.py's 1st-degree AVB rule
    # (_PR_AVB1_TABLE) uses a different methodology -- an age x heart-rate 2D
    # lookup table from the DXL threshold reference -- rather than this
    # continuous age-only formula from GAN Physician's Guide section 6.1.
    # Both are faithful to their own cited source; see the matching note next
    # to _PR_AVB1_TABLE in interpret.py.
    age_days = context.patient.age_days
    if age_days is None:
        return {"short": 110.0, "first_degree": 220.0, "borderline": 200.0}
    age = float(age_days)
    short = 75.0 + 0.006 * age if age < 16.0 * 365.25 else 110.0
    if age <= 18.0 * 365.25:
        first_degree = 163.0 + 0.0087 * age
        borderline = 143.0 + 0.0087 * age
    else:
        first_degree = 220.0
        borderline = 200.0
    return {"short": short, "first_degree": first_degree, "borderline": borderline}


def _pr_result(context: GlasgowContext, rule_id: str, statement: str) -> RuleEvaluation:
    pr_ms = _finite(context.global_value("pr_ms"))
    omitted_by = _pr_omitted_by(context)
    limits = _pr_limits(context)
    uses_days = bool(context.patient.age_days is not None and context.patient.age_days < 18.0 * 365.25)
    fidelity, approximation = _age_fidelity(context, uses_days=uses_days)
    threshold_map = {
        "GAN-06.01-01": {"lower_limit_ms": limits["short"]},
        "GAN-06.01-02": {"first_degree_limit_ms": limits["first_degree"]},
        "GAN-06.01-03": {"borderline_limit_ms": limits["borderline"]},
    }
    common = {
        "rule_id": rule_id,
        "statement": statement,
        "required_inputs": ["global.pr_ms", "rhythm.p_wave_flag", "rhythm.sinus_rhythm", "rhythm.wpw_pattern"],
        "thresholds": threshold_map[rule_id],
        "fidelity": fidelity,
        "source": {"kind": "glasgow_guide", "reference": _PR_SOURCE},
        "approximation": approximation,
        "summary_code": 5 if rule_id == "GAN-06.01-02" else 4,
    }
    evidence = {
        "pr_ms": pr_ms,
        "age_days": context.patient.age_days,
        "age_days_source": context.patient.age_days_source,
        "omitted_by": omitted_by,
    }
    if omitted_by:
        return RuleEvaluation(
            **common,
            evaluation_status="not_applicable",
            evidence=evidence,
        )
    if pr_ms is None:
        return RuleEvaluation(
            **common,
            evaluation_status="unavailable",
            evidence=evidence,
            missing_inputs=["global.pr_ms"],
        )
    if rule_id == "GAN-06.01-01":
        matched = pr_ms < limits["short"]
    elif rule_id == "GAN-06.01-02":
        matched = pr_ms >= limits["first_degree"]
    else:
        matched = pr_ms < limits["first_degree"] and pr_ms >= limits["borderline"]
        evidence["first_degree_rule_matched"] = pr_ms >= limits["first_degree"]
    return RuleEvaluation(
        **common,
        evaluation_status="matched" if matched else "not_matched",
        resolution_status="final" if matched else "no_statement",
        evidence=evidence,
    )


def _qtc_thresholds(context: GlasgowContext) -> tuple[float, float, str, Optional[Dict[str, Any]]]:
    patient = context.patient
    if patient.age_missing or patient.sex_missing:
        return (
            465.0,
            485.0,
            "existing_dxl_approximation",
            {
                "source": "existing_dxl",
                "reason": "missing_demographic_sex_neutral_thresholds",
                "original_rule_reproducible": False,
            },
        )
    age_days = float(patient.age_days or 0.0)
    if age_days < 0.5 * 365.25:
        return 500.0, 520.0, "glasgow_explicit", None
    if patient.sex_normalized == "female" and age_days >= 50.0 * 365.25:
        return 470.0, 490.0, "glasgow_explicit", None
    return 460.0, 480.0, "glasgow_explicit", None


def _qtc_common(context: GlasgowContext) -> tuple[Optional[Dict[str, float]], List[str]]:
    qt_ms = _finite(context.global_value("qt_ms"))
    hr_bpm = _finite(context.global_value("heart_rate_bpm"))
    qrs_ms = _finite(context.global_value("qrs_ms"))
    values = qtc_values(qt_ms, hr_bpm) if qt_ms is not None and hr_bpm is not None and qt_ms > 0 and hr_bpm > 0 else None
    omitted_by: List[str] = []
    if qrs_ms is not None and qrs_ms >= 120.0:
        omitted_by.append("qrs_duration_ge_120")
    if hr_bpm is not None and hr_bpm > 125.0:
        omitted_by.append("heart_rate_gt_125")
    return values, omitted_by


def _qtc_result(context: GlasgowContext, rule_id: str, statement: str) -> RuleEvaluation:
    values, omitted_by = _qtc_common(context)
    borderline, prolonged, fidelity, approximation = _qtc_thresholds(context)
    thresholds = (
        {"short_max_ms": 350.0}
        if rule_id == "GAN-06.02-03"
        else {"borderline_min_ms": borderline, "prolonged_min_ms": prolonged}
    )
    common = {
        "rule_id": rule_id,
        "statement": statement,
        "required_inputs": ["global.qt_ms", "global.heart_rate_bpm", "global.qrs_ms"],
        "thresholds": thresholds,
        "fidelity": fidelity,
        "source": {"kind": "glasgow_guide", "reference": _QT_SOURCE},
        "approximation": approximation,
        "summary_code": 4 if rule_id == "GAN-06.02-01" else 5,
    }
    evidence = {
        "qt_ms": _finite(context.global_value("qt_ms")),
        "heart_rate_bpm": _finite(context.global_value("heart_rate_bpm")),
        "qrs_duration_ms": _finite(context.global_value("qrs_ms")),
        "qtc_formula": context.config.qtc_formula,
        "qtc_values_ms": values,
        "omitted_by": omitted_by,
        "demographic_route": context.patient.qtc_demographic_route,
    }
    if values is None:
        missing = [
            name
            for name, value in (
                ("global.qt_ms", evidence["qt_ms"]),
                ("global.heart_rate_bpm", evidence["heart_rate_bpm"]),
            )
            if value is None
        ]
        return RuleEvaluation(
            **common,
            evaluation_status="unavailable",
            evidence=evidence,
            missing_inputs=missing,
        )
    if omitted_by:
        return RuleEvaluation(
            **common,
            evaluation_status="not_applicable",
            evidence=evidence,
        )
    selected = values[f"{context.config.qtc_formula}_ms"]
    evidence["selected_qtc_ms"] = selected
    if rule_id == "GAN-06.02-01":
        matched = borderline <= selected < prolonged
    elif rule_id == "GAN-06.02-02":
        matched = selected >= prolonged
    else:
        matched = selected <= 350.0
    return RuleEvaluation(
        **common,
        evaluation_status="matched" if matched else "not_matched",
        resolution_status="final" if matched else "no_statement",
        evidence=evidence,
    )


def interval_rules() -> List[RuleSpec]:
    definitions: List[Tuple[str, str, str, int, int, Any]] = [
        ("GAN-06.01-01", "6.1", "Short PR Interval", 60101, 4, lambda context: _pr_result(context, "GAN-06.01-01", "Short PR Interval")),
        ("GAN-06.01-02", "6.1", "with 1st degree A-V block", 60102, 5, lambda context: _pr_result(context, "GAN-06.01-02", "with 1st degree A-V block")),
        ("GAN-06.01-03", "6.1", "with borderline 1st degree A-V block", 60103, 4, lambda context: _pr_result(context, "GAN-06.01-03", "with borderline 1st degree A-V block")),
        ("GAN-06.02-01", "6.2", "Borderline prolonged QT interval", 60201, 4, lambda context: _qtc_result(context, "GAN-06.02-01", "Borderline prolonged QT interval")),
        ("GAN-06.02-02", "6.2", "Prolonged QT – consider ischemia, electrolyte imbalance, drug effects", 60202, 5, lambda context: _qtc_result(context, "GAN-06.02-02", "Prolonged QT – consider ischemia, electrolyte imbalance, drug effects")),
        ("GAN-06.02-03", "6.2", "Short QT interval", 60203, 5, lambda context: _qtc_result(context, "GAN-06.02-03", "Short QT interval")),
    ]
    rules: List[RuleSpec] = []
    for rule_id, chapter, statement, order, summary_code, evaluator in definitions:
        rules.append(
            RuleSpec(
                rule_id=rule_id,
                chapter=chapter,
                pdf_page=16 if chapter == "6.1" else 17,
                category="pr" if chapter == "6.1" else "qtc",
                order=order,
                statement=statement,
                required_inputs=("global.pr_ms",) if chapter == "6.1" else ("global.qt_ms", "global.heart_rate_bpm", "global.qrs_ms"),
                fidelity="glasgow_explicit",
                source_kind="glasgow_guide",
                source_reference=_PR_SOURCE if chapter == "6.1" else _QT_SOURCE,
                evaluator=evaluator,
                summary_code=summary_code,
            )
        )
    return rules
