from __future__ import annotations

import math
from typing import Dict, Optional

from .config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import RuleEvaluation
from .rhythm import (
    av_dissociation_corroborated,
    disorganized_atrial_activity,
    has_borderline_af_evidence,
)
from .sources import SOURCE_AHA_QT_2009


_INTERVALS = DEFAULT_DIAGNOSTIC_CONFIG.intervals

ADULT_QT_PROLONGED_FEMALE_MS = _INTERVALS.qtc_prolonged_female_ms
ADULT_QT_PROLONGED_MALE_MS = _INTERVALS.qtc_prolonged_male_ms
ADULT_QT_MARKEDLY_PROLONGED_MS = _INTERVALS.qtc_markedly_prolonged_ms
ADULT_QT_SEVERE_MS = _INTERVALS.qtc_severe_ms
ADULT_QT_MARKEDLY_SHORT_MS = _INTERVALS.qtc_short_ms
ADULT_QT_POSSIBLY_SHORT_MS = _INTERVALS.qtc_possibly_short_ms
ADULT_QT_BORDERLINE_SHORT_MS = _INTERVALS.qtc_borderline_short_ms


def qtc_values(qt_ms: float, hr_bpm: float) -> Dict[str, float]:
    qt = float(qt_ms)
    heart_rate = float(hr_bpm)
    if not math.isfinite(qt) or not math.isfinite(heart_rate) or qt <= 0 or heart_rate <= 0:
        raise ValueError("QT and heart rate must be finite positive values")
    rr_s = 60.0 / heart_rate
    return {
        "bazett_ms": qt / math.sqrt(rr_s),
        "fridericia_ms": qt / math.pow(rr_s, 1.0 / 3.0),
        "hodges_ms": qt + 1.75 * (heart_rate - 60.0),
        "framingham_ms": qt + 154.0 * (1.0 - rr_s),
    }


def classify_adult_qt(
    values: Dict[str, float],
    *,
    sex: str,
    reliable: bool,
) -> RuleEvaluation:
    normalized_sex = str(sex or "unknown").strip().lower()
    missing = []
    if not reliable:
        missing.append("global.qt_reliability")
    if normalized_sex not in {"female", "f", "male", "m"}:
        missing.append("patient.sex")
    threshold = (
        ADULT_QT_PROLONGED_FEMALE_MS
        if normalized_sex in {"female", "f"}
        else ADULT_QT_PROLONGED_MALE_MS
    )
    selected = float(values["hodges_ms"])
    statement_code = None
    statement = None
    severity = "normal"
    confidence = None
    status = "not_matched"
    # Two independent routes into the markedly-prolonged tier: the historical
    # Bazett >500 ms critical alert, and the >=480 ms definite-prolongation
    # step on the primary (Hodges) correction. Without the second one there
    # was no grade between "prolonged" at 450/460 ms and the 500 ms alert.
    bazett_critical = float(values["bazett_ms"]) > ADULT_QT_SEVERE_MS
    primary_markedly_prolonged = (
        float(values["hodges_ms"]) >= ADULT_QT_MARKEDLY_PROLONGED_MS
    )
    severe = bazett_critical or primary_markedly_prolonged
    if not reliable:
        status = "unavailable"
    elif normalized_sex not in {"female", "f", "male", "m"} and not severe:
        status = "unavailable"
    elif severe:
        status = "matched"
        statement_code = "markedly_prolonged_qt"
        statement = (
            "Markedly prolonged QTc (>500 ms by Bazett); urgent medication and electrolyte review"
            if bazett_critical
            else (
                f"Markedly prolonged QTc (>={ADULT_QT_MARKEDLY_PROLONGED_MS:.0f} ms "
                "by Hodges correction); medication and electrolyte review"
            )
        )
        severity = "high"
        confidence = "high"
    elif selected <= ADULT_QT_MARKEDLY_SHORT_MS:
        status = "matched"
        statement_code = "markedly_short_qt"
        statement = "Markedly short QT interval; manual and clinical confirmation required"
        severity = "abnormal"
        confidence = "moderate"
    elif selected <= ADULT_QT_POSSIBLY_SHORT_MS:
        status = "matched"
        statement_code = "possible_short_qt_pattern"
        statement = "Possible short-QT pattern; clinical and family-history correlation required"
        severity = "borderline"
        confidence = "low"
    elif selected <= ADULT_QT_BORDERLINE_SHORT_MS:
        status = "matched"
        statement_code = "borderline_short_qt"
        statement = "Borderline short QT interval"
        severity = "borderline"
        confidence = "low"
    elif selected >= threshold:
        status = "matched"
        statement_code = "prolonged_qt"
        statement = "Prolonged QT interval by Hodges correction; manual review required"
        severity = "abnormal"
        confidence = "moderate"
    return RuleEvaluation(
        rule_id="CLIN-INTERVAL-QT-01",
        domain="intervals",
        status=status,
        statement_code=statement_code,
        statement=statement,
        severity=severity,
        confidence=confidence,
        required_inputs=["global.qt_ms", "global.heart_rate_bpm", "global.qt_reliability", "patient.sex"],
        missing_inputs=missing,
        evidence={
            "primary_formula": "hodges",
            "selected_qtc_ms": selected,
            "formula_values_ms": dict(values),
            "manual_review_required": status == "matched",
        },
        thresholds={
            "prolonged_ms": threshold,
            "markedly_prolonged_gte_ms": ADULT_QT_MARKEDLY_PROLONGED_MS,
            "bazett_critical_gt_ms": ADULT_QT_SEVERE_MS,
            "markedly_short_lte_ms": ADULT_QT_MARKEDLY_SHORT_MS,
            "possibly_short_lte_ms": ADULT_QT_POSSIBLY_SHORT_MS,
            "borderline_short_lte_ms": ADULT_QT_BORDERLINE_SHORT_MS,
        },
        source=dict(SOURCE_AHA_QT_2009),
        priority="P1" if bazett_critical else ("P2" if severe else None),
        human_review_required=status == "matched",
    )


def evaluate_qt_values(
    qt_ms: Optional[float],
    hr_bpm: Optional[float],
    *,
    sex: str,
    reliable: bool,
    qrs_ms: Optional[float] = None,
    jt_ms: Optional[float] = None,
    rr_cv: Optional[float] = None,
) -> RuleEvaluation:
    if qrs_ms is not None and qrs_ms >= 120.0:
        wide_missing = []
        if hr_bpm is None:
            wide_missing.append("global.heart_rate_bpm")
        if not reliable:
            wide_missing.append("global.qt_reliability")
        if rr_cv is not None and rr_cv > 0.15:
            wide_missing.append("rhythm.stable_rr_for_jt_correction")
        jt_value = jt_ms
        if jt_value is None and qt_ms is not None:
            jt_value = float(qt_ms) - float(qrs_ms)
        if jt_value is None or jt_value <= 0:
            wide_missing.append("global.jt_ms")
        if wide_missing:
            return RuleEvaluation(
                rule_id="CLIN-INTERVAL-QT-01",
                domain="intervals",
                status="unavailable",
                required_inputs=["global.jt_ms", "global.heart_rate_bpm"],
                missing_inputs=wide_missing,
                evidence={
                    "qt_ms": qt_ms,
                    "jt_ms": jt_value,
                    "heart_rate_bpm": hr_bpm,
                    "qrs_ms": qrs_ms,
                    "rr_cv": rr_cv,
                },
                source=dict(SOURCE_AHA_QT_2009),
            )
        jtc_hodges = float(jt_value) + 1.75 * (float(hr_bpm) - 60.0)
        return RuleEvaluation(
            rule_id="CLIN-INTERVAL-QT-01",
            domain="intervals",
            status="matched",
            statement_code="wide_qrs_repolarization_review",
            statement="Wide-QRS repolarization interval review using JT/JTc",
            severity="observation",
            confidence="low",
            required_inputs=["global.jt_ms", "global.heart_rate_bpm", "global.qrs_ms"],
            evidence={
                "qt_ms": qt_ms,
                "jt_ms": float(jt_value),
                "jtc_hodges_ms": jtc_hodges,
                "heart_rate_bpm": float(hr_bpm),
                "qrs_ms": float(qrs_ms),
                "manual_review_required": True,
            },
            source=dict(SOURCE_AHA_QT_2009),
        )
    missing = []
    if qt_ms is None:
        missing.append("global.qt_ms")
    if hr_bpm is None:
        missing.append("global.heart_rate_bpm")
    if not reliable:
        missing.append("global.qt_reliability")
    if rr_cv is not None and rr_cv > 0.15:
        missing.append("rhythm.stable_rr_for_qt_correction")
    if missing:
        return RuleEvaluation(
            rule_id="CLIN-INTERVAL-QT-01",
            domain="intervals",
            status="unavailable",
            required_inputs=["global.qt_ms", "global.heart_rate_bpm", "global.qt_reliability"],
            missing_inputs=missing,
            evidence={"qt_ms": qt_ms, "heart_rate_bpm": hr_bpm, "qrs_ms": qrs_ms, "rr_cv": rr_cv},
            source=dict(SOURCE_AHA_QT_2009),
        )
    return classify_adult_qt(qtc_values(float(qt_ms), float(hr_bpm)), sex=sex, reliable=True)


def classify_adult_pr(
    pr_ms: Optional[float],
    *,
    heart_rate_bpm: Optional[float] = None,
    atrial_association_unreliable: bool = False,
    technically_limited: bool = False,
) -> RuleEvaluation:
    if pr_ms is None:
        return RuleEvaluation(
            rule_id="CLIN-INTERVAL-PR-01",
            domain="intervals",
            status="unavailable",
            required_inputs=["global.pr_ms"],
            missing_inputs=["global.pr_ms"],
        )
    value = float(pr_ms)
    heart_rate = (
        float(heart_rate_bpm)
        if heart_rate_bpm is not None
        and math.isfinite(float(heart_rate_bpm))
        and float(heart_rate_bpm) > 0.0
        else None
    )
    rr_ms = 60000.0 / heart_rate if heart_rate is not None else None
    pr_rr_ratio = value / rr_ms if rr_ms is not None else None
    # At faster rates, a measured PR occupying more than half of the RR
    # interval is commonly a cross-cycle P-to-QRS association.  Do not turn
    # that geometrically suspicious measurement into a definite AV-delay
    # statement without beat-level same-cycle confirmation.
    association_gate_reasons: list[str] = []
    if (
        value > 200.0
        and heart_rate is not None
        and heart_rate > 100.0
        and pr_rr_ratio is not None
        and pr_rr_ratio > 0.50
    ):
        association_gate_reasons.append("tachycardia_pr_exceeds_half_rr")
    if value > 200.0 and atrial_association_unreliable:
        association_gate_reasons.append(
            "atrial_rhythm_evidence_invalidates_definite_pr_association"
        )
    if (
        value > 200.0
        and technically_limited
        and heart_rate is not None
        and heart_rate > 100.0
        and pr_rr_ratio is not None
        and pr_rr_ratio > 0.45
    ):
        association_gate_reasons.append(
            "technical_limitation_with_tachycardic_high_pr_rr_ratio"
        )
    if association_gate_reasons:
        return RuleEvaluation(
            rule_id="CLIN-INTERVAL-PR-01",
            domain="intervals",
            status="matched",
            statement_code="possible_first_degree_av_delay",
            statement=(
                "Possible first-degree atrioventricular delay; "
                "same-cycle P-to-QRS confirmation required"
            ),
            severity="borderline",
            confidence="low",
            coverage="partial",
            required_inputs=[
                "global.pr_ms",
                "global.heart_rate_bpm",
                "beat_level.same_cycle_p_qrs_association",
            ],
            missing_inputs=["beat_level.same_cycle_p_qrs_association"],
            evidence={
                "pr_ms": value,
                "heart_rate_bpm": heart_rate,
                "rr_ms": rr_ms,
                "pr_rr_ratio": pr_rr_ratio,
                "association_gate": association_gate_reasons[0],
                "association_gate_reasons": association_gate_reasons,
                "atrial_association_unreliable": atrial_association_unreliable,
                "technically_limited": technically_limited,
                "evaluates_code": "first_degree_av_delay",
                "manual_confirmation_required": True,
            },
            thresholds={
                "short_lt_ms": 120.0,
                "first_degree_gt_ms": 200.0,
                "tachycardia_gt_bpm": 100.0,
                "same_cycle_pr_rr_ratio_max": 0.50,
                "technically_limited_pr_rr_ratio_max": 0.45,
            },
            source={
                "authority": "AHA/ACC/HRS",
                "document": "Bradycardia and Conduction Delay Guideline",
                "section": "Atrioventricular conduction",
                "version": "2018",
            },
        )
    if value < 120.0:
        code = "short_pr"
        statement = "Short PR interval"
    elif value > 200.0:
        code = "first_degree_av_delay"
        statement = "First-degree atrioventricular delay"
    else:
        code = None
        statement = None
    return RuleEvaluation(
        rule_id="CLIN-INTERVAL-PR-01",
        domain="intervals",
        status="matched" if code else "not_matched",
        statement_code=code,
        statement=statement,
        severity="abnormal" if code else "normal",
        required_inputs=["global.pr_ms"],
        evidence={
            "pr_ms": value,
            "heart_rate_bpm": heart_rate,
            "rr_ms": rr_ms,
            "pr_rr_ratio": pr_rr_ratio,
            "atrial_association_unreliable": atrial_association_unreliable,
            "technically_limited": technically_limited,
        },
        thresholds={
            "short_lt_ms": 120.0,
            "first_degree_gt_ms": 200.0,
            "tachycardia_gt_bpm": 100.0,
            "same_cycle_pr_rr_ratio_max": 0.50,
            "technically_limited_pr_rr_ratio_max": 0.45,
        },
        source={
            "authority": "AHA/ACC/HRS",
            "document": "Bradycardia and Conduction Delay Guideline",
            "section": "Atrioventricular conduction",
            "version": "2018",
        },
    )


def classify_pediatric_pr(pr_ms: Optional[float], *, age_years: Optional[float]) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id="CLIN-INTERVAL-PR-PEDS-01",
        domain="intervals",
        status="unavailable",
        required_inputs=["pediatric_pr_public_reference_table"],
        missing_inputs=["pediatric_pr_public_reference_table"],
        evidence={"pr_ms": pr_ms, "age_years": age_years},
    )


def evaluate_intervals(context) -> list[RuleEvaluation]:
    global_features = context.features.global_features
    interpretation = context.features.interpretation
    rr_cv = getattr(interpretation, "rr_cv", None) if interpretation is not None else None
    qt_reliability = getattr(global_features, "qt_reliability", "unavailable")
    qt_reliable = qt_reliability in {"reliable", "rescued"}
    metadata = getattr(context.features, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    rhythm = metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    af_afl = rhythm.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    rhythm_availability = rhythm.get("availability", {})
    rhythm_availability = (
        rhythm_availability if isinstance(rhythm_availability, dict) else {}
    )
    rule_summary = rhythm.get("rule_summary", {})
    rule_summary = rule_summary if isinstance(rule_summary, dict) else {}
    # Disorganized atrial activity makes the P-to-QRS association unreliable
    # even when no AF/flutter call was reached, which is the commoner case:
    # the flutter rule frequently sits at `unavailable`, and an F wave landing
    # in front of the QRS then measures as a long PR.
    disorganized_p = disorganized_atrial_activity(
        af_afl,
        threshold=DEFAULT_DIAGNOSTIC_CONFIG.rhythm.organized_p_ratio_min_for_morphology,
    )
    atrial_association_unreliable = bool(
        af_afl.get("probable_af")
        or af_afl.get("probable_flutter")
        or af_afl.get("af_afl_indeterminate")
        or has_borderline_af_evidence(af_afl)
        or disorganized_p is not None
    )
    pr_unavailable_reasons = list(rhythm_availability.get("reasons") or [])
    if bool(rule_summary.get("complete_av_block")):
        pr_unavailable_reasons.append("complete_av_block")
    # `av_dissociation` and `af_afl_indeterminate` are kept out of the blocking
    # set unless corroborated, matching `build_measurement_availability`.  This
    # layer used to re-add `av_dissociation` from `rule_summary` unconditionally,
    # which would have left CLIN-INTERVAL-PR-01 unavailable on exactly the
    # records whose PR class the rhythm layer now reports.
    if bool(rule_summary.get("av_dissociation")) and av_dissociation_corroborated(
        rule_summary
    ):
        pr_unavailable_reasons.append("av_dissociation")
    pr_blocking_reasons = {
        "complete_av_block",
        "continuous_pacing",
        "probable_af",
        "probable_flutter",
        "wide_qrs_pacing_like_context",
        "atrial_measurements_unavailable",
    }
    if bool(rule_summary.get("av_dissociation")) and av_dissociation_corroborated(
        rule_summary
    ):
        pr_blocking_reasons.add("av_dissociation")
    if disorganized_p is not None:
        pr_blocking_reasons.add("af_afl_indeterminate")
    pr_measurement_available = bool(
        rhythm_availability.get("pr_available", True)
        and not pr_blocking_reasons.intersection(pr_unavailable_reasons)
    )
    record_quality = metadata.get("record_quality", {})
    record_quality = record_quality if isinstance(record_quality, dict) else {}
    record_grade = record_quality.get("record_grade")
    technically_limited = record_grade not in {None, "Q0"}
    qt = evaluate_qt_values(
        context.global_value("qt_ms"),
        context.global_value("heart_rate_bpm"),
        sex=context.sex,
        reliable=qt_reliable,
        qrs_ms=context.global_value("qrs_ms"),
        jt_ms=context.global_value("jt_ms"),
        rr_cv=rr_cv,
    )
    if context.age_years is not None and context.age_years < 18.0:
        pr = classify_pediatric_pr(
            context.global_value("pr_ms"), age_years=context.age_years
        )
    elif not pr_measurement_available:
        pr = RuleEvaluation(
            rule_id="CLIN-INTERVAL-PR-01",
            domain="intervals",
            status="unavailable",
            required_inputs=[
                "global.pr_ms",
                "rhythm.stable_same_cycle_p_qrs_association",
            ],
            missing_inputs=["rhythm.stable_same_cycle_p_qrs_association"],
            evidence={
                "measured_pr_ms": context.global_value("pr_ms"),
                "pr_available": False,
                "unavailable_reasons": sorted(set(pr_unavailable_reasons)),
                "evaluates_code": "first_degree_av_delay",
            },
        )
    else:
        pr = classify_adult_pr(
            context.global_value("pr_ms"),
            heart_rate_bpm=context.global_value("heart_rate_bpm"),
            atrial_association_unreliable=atrial_association_unreliable,
            technically_limited=technically_limited,
        )
    return [pr, qt]
