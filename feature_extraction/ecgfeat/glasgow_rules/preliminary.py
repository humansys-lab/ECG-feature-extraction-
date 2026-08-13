from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .context import GlasgowContext
from .models import RuleEvaluation, RuleSpec


_SOURCE_41 = "Physician's Guide section 4.1, PDF pages 10-13"
_SOURCE_42 = "Physician's Guide section 4.2, PDF page 13"
_SOURCE_43 = "Physician's Guide section 4.3, PDF pages 13-14"
_SOURCE_44 = "Physician's Guide section 4.4, PDF page 14"


def _source(reference: str) -> Dict[str, str]:
    return {"kind": "glasgow_guide", "reference": reference}


def _simple_result(
    *,
    rule_id: str,
    statement: str,
    matched: bool,
    evidence: Optional[Dict[str, Any]] = None,
    summary_code: Optional[int] = None,
    reference: str,
    not_applicable: bool = False,
) -> RuleEvaluation:
    status = "not_applicable" if not_applicable else ("matched" if matched else "not_matched")
    return RuleEvaluation(
        rule_id=rule_id,
        evaluation_status=status,
        resolution_status="advisory" if matched and not not_applicable else "no_statement",
        statement=statement,
        evidence=dict(evidence or {}),
        fidelity="glasgow_explicit",
        source=_source(reference),
        summary_code=summary_code,
    )


def evaluate_unsuitable_leads(context: GlasgowContext) -> RuleEvaluation:
    leads = sorted(
        lead
        for lead, quality in context.features.quality.items()
        if bool(getattr(quality, "missing", False))
    )
    statement = "Lead(s) unsuitable for analysis"
    if leads:
        statement += f": {', '.join(leads)}"
    return _simple_result(
        rule_id="GAN-04.01-07",
        statement=statement,
        matched=bool(leads),
        evidence={"leads": leads},
        reference=_SOURCE_41,
    )


def evaluate_measurement_error(context: GlasgowContext) -> RuleEvaluation:
    best_lead: Optional[str] = None
    best_value: Optional[float] = None
    for lead, fact in context.leads.items():
        values: List[float] = []
        for key in ("p_positive_amp_mv", "p_negative_amp_mv"):
            raw = fact.measurements.get(key)
            if isinstance(raw, (int, float)):
                values.append(abs(float(raw)))
        if not values:
            raw = fact.measurements.get("p_amp_mv")
            if isinstance(raw, (int, float)):
                values.append(abs(float(raw)))
        if values:
            value = max(values)
            if best_value is None or value > best_value:
                best_lead, best_value = lead, value
    return _simple_result(
        rule_id="GAN-04.01-08",
        statement="--- Possible measurement error ---",
        matched=bool(best_value is not None and best_value > 1.0),
        evidence={
            "max_abs_p_component_mv": best_value,
            "lead": best_lead,
        },
        reference=_SOURCE_41,
    )


def evaluate_rhythm_warning(context: GlasgowContext) -> RuleEvaluation:
    matched = bool(context.rhythm.get("abnormal_ventricular_conduction"))
    return _simple_result(
        rule_id="GAN-04.02-01",
        statement="If rhythm is confirmed, the following report may not be valid.",
        matched=matched,
        evidence={
            "abnormal_ventricular_conduction": matched,
            "bundle_branch_block": context.rhythm.get("bundle_branch_block"),
            "qrs_duration_ms": context.global_value("qrs_ms"),
        },
        reference=_SOURCE_42,
    )


def evaluate_invalid_clinical_data(context: GlasgowContext) -> RuleEvaluation:
    values = [value.strip().lower() for value in context.patient.clinical_classifications]
    matched = len(values) > 1 and ("normal" in values or "unknown" in values)
    return _simple_result(
        rule_id="GAN-04.03-01",
        statement="--- Invalid clinical data entry ---",
        matched=matched,
        evidence={"clinical_classifications": list(context.patient.clinical_classifications)},
        reference=_SOURCE_43,
    )


def evaluate_invalid_medication(context: GlasgowContext) -> RuleEvaluation:
    values = [value.strip().lower() for value in context.patient.meds]
    matched = len(values) > 1 and "unknown" in values
    return _simple_result(
        rule_id="GAN-04.03-02",
        statement="--- Invalid medication entry ---",
        matched=matched,
        evidence={"medications": list(context.patient.meds)},
        reference=_SOURCE_43,
    )


def evaluate_missing_gender(context: GlasgowContext) -> RuleEvaluation:
    both_missing = context.patient.age_missing and context.patient.sex_missing
    return _simple_result(
        rule_id="GAN-04.03-03",
        statement="--- Interpretation made without knowing patient’s gender ---",
        matched=context.patient.sex_missing and not context.patient.age_missing,
        evidence={"age_missing": context.patient.age_missing, "sex_missing": context.patient.sex_missing},
        reference=_SOURCE_43,
        not_applicable=both_missing,
    )


def evaluate_missing_age(context: GlasgowContext) -> RuleEvaluation:
    both_missing = context.patient.age_missing and context.patient.sex_missing
    return _simple_result(
        rule_id="GAN-04.03-04",
        statement="--- Interpretation made without knowing patient’s age ---",
        matched=context.patient.age_missing and not context.patient.sex_missing,
        evidence={"age_missing": context.patient.age_missing, "sex_missing": context.patient.sex_missing},
        reference=_SOURCE_43,
        not_applicable=both_missing,
    )


def evaluate_missing_gender_age(context: GlasgowContext) -> RuleEvaluation:
    return _simple_result(
        rule_id="GAN-04.03-05",
        statement="--- Interpretation made without knowing patient’s gender/age ---",
        matched=context.patient.age_missing and context.patient.sex_missing,
        evidence={"age_missing": context.patient.age_missing, "sex_missing": context.patient.sex_missing},
        reference=_SOURCE_43,
    )


def evaluate_pediatric_route(context: GlasgowContext) -> RuleEvaluation:
    return _simple_result(
        rule_id="GAN-04.03-06",
        statement="--- Interpretation based on pediatric criteria ---",
        matched=context.patient.pediatric,
        evidence={
            "age_years": context.patient.age_years,
            "age_days": context.patient.age_days,
            "algorithm_age_group": context.patient.algorithm_age_group,
        },
        reference=_SOURCE_43,
    )


def evaluate_pacemaker_restriction(context: GlasgowContext) -> RuleEvaluation:
    matched = bool(context.rhythm.get("paced_rhythm"))
    return _simple_result(
        rule_id="GAN-04.04-01",
        statement="Pacemaker rhythm – no further analysis",
        matched=matched,
        evidence={"paced_rhythm": matched},
        reference=_SOURCE_44,
    )


def evaluate_no_dominant_qrs(context: GlasgowContext) -> RuleEvaluation:
    available = bool(context.rhythm.get("dominant_qrs_available"))
    return _simple_result(
        rule_id="GAN-04.04-02",
        statement="--- No further analysis due to lack of dominant QRS ---",
        matched=not available,
        evidence={"dominant_qrs_available": available},
        reference=_SOURCE_44,
    )


def evaluate_similar_precordial_qrs(context: GlasgowContext) -> RuleEvaluation:
    if "similar_precordial_qrs" not in context.features.metadata:
        return RuleEvaluation(
            rule_id="GAN-04.04-03",
            evaluation_status="unavailable",
            statement="--- Similar QRS in V leads ---",
            required_inputs=["metadata.similar_precordial_qrs"],
            missing_inputs=["metadata.similar_precordial_qrs"],
            fidelity="not_reproducible_from_guide",
            source={"kind": "none", "reference": _SOURCE_44},
        )
    matched = bool(context.features.metadata.get("similar_precordial_qrs"))
    result = _simple_result(
        rule_id="GAN-04.04-03",
        statement="--- Similar QRS in V leads ---",
        matched=matched,
        evidence={"similar_precordial_qrs": matched},
        reference=_SOURCE_44,
    )
    result.fidelity = "existing_dxl_approximation"
    result.source = {"kind": "existing_dxl", "reference": "metadata.similar_precordial_qrs"}
    result.approximation = {
        "source": "existing_dxl",
        "reason": "guide_does_not_disclose_similarity_algorithm",
        "original_rule_reproducible": False,
    }
    return result


def evaluate_technical_quality(context: GlasgowContext) -> RuleEvaluation:
    grade = str(context.record_quality.get("record_grade") or "")
    return _simple_result(
        rule_id="GAN-04.04-04",
        statement="--- Technically unsatisfactory tracing ---",
        matched=grade == "Q3",
        evidence={
            "record_grade": grade or None,
            "reason_codes": list(context.record_quality.get("reason_codes") or []),
        },
        summary_code=6,
        reference=_SOURCE_44,
    )


def restricted_analysis(evaluations: Iterable[RuleEvaluation]) -> Dict[str, Any]:
    restricting = {
        "GAN-04.04-01",
        "GAN-04.04-02",
        "GAN-04.04-03",
        "GAN-04.04-04",
    }
    stopped_by = sorted(
        item.rule_id
        for item in evaluations
        if item.rule_id in restricting and item.evaluation_status == "matched"
    )
    return {
        "active": bool(stopped_by),
        "stop_categories": ["qrs_t_morphology"] if stopped_by else [],
        "stopped_by": stopped_by,
    }


def preliminary_base_rules() -> List[RuleSpec]:
    definitions: List[
        Tuple[str, str, int, str, int, str, Callable[[GlasgowContext], RuleEvaluation], str, Optional[int]]
    ] = [
        ("GAN-04.01-07", "4.1", 11, "preliminary_lead", 40107, "Lead(s) unsuitable for analysis", evaluate_unsuitable_leads, _SOURCE_41, None),
        ("GAN-04.01-08", "4.1", 11, "preliminary_lead", 40108, "--- Possible measurement error ---", evaluate_measurement_error, _SOURCE_41, None),
        ("GAN-04.02-01", "4.2", 13, "preliminary_rhythm", 40201, "If rhythm is confirmed, the following report may not be valid.", evaluate_rhythm_warning, _SOURCE_42, None),
        ("GAN-04.03-01", "4.3", 13, "preliminary_demographic", 40301, "--- Invalid clinical data entry ---", evaluate_invalid_clinical_data, _SOURCE_43, None),
        ("GAN-04.03-02", "4.3", 13, "preliminary_demographic", 40302, "--- Invalid medication entry ---", evaluate_invalid_medication, _SOURCE_43, None),
        ("GAN-04.03-03", "4.3", 14, "preliminary_demographic", 40303, "--- Interpretation made without knowing patient’s gender ---", evaluate_missing_gender, _SOURCE_43, None),
        ("GAN-04.03-04", "4.3", 14, "preliminary_demographic", 40304, "--- Interpretation made without knowing patient’s age ---", evaluate_missing_age, _SOURCE_43, None),
        ("GAN-04.03-05", "4.3", 14, "preliminary_demographic", 40305, "--- Interpretation made without knowing patient’s gender/age ---", evaluate_missing_gender_age, _SOURCE_43, None),
        ("GAN-04.03-06", "4.3", 14, "preliminary_demographic", 40306, "--- Interpretation based on pediatric criteria ---", evaluate_pediatric_route, _SOURCE_43, None),
        ("GAN-04.04-01", "4.4", 14, "restricted", 40401, "Pacemaker rhythm – no further analysis", evaluate_pacemaker_restriction, _SOURCE_44, None),
        ("GAN-04.04-02", "4.4", 14, "restricted", 40402, "--- No further analysis due to lack of dominant QRS ---", evaluate_no_dominant_qrs, _SOURCE_44, None),
        ("GAN-04.04-03", "4.4", 14, "restricted", 40403, "--- Similar QRS in V leads ---", evaluate_similar_precordial_qrs, _SOURCE_44, None),
        ("GAN-04.04-04", "4.4", 14, "restricted", 40404, "--- Technically unsatisfactory tracing ---", evaluate_technical_quality, _SOURCE_44, 6),
    ]
    rules: List[RuleSpec] = []
    for rule_id, chapter, page, category, order, statement, evaluator, reference, summary_code in definitions:
        fidelity = "not_reproducible_from_guide" if rule_id == "GAN-04.04-03" else "glasgow_explicit"
        source_kind = "none" if rule_id == "GAN-04.04-03" else "glasgow_guide"
        rules.append(
            RuleSpec(
                rule_id=rule_id,
                chapter=chapter,
                pdf_page=page,
                category=category,
                order=order,
                statement=statement,
                required_inputs=(),
                fidelity=fidelity,
                source_kind=source_kind,
                source_reference=reference,
                evaluator=evaluator,
                summary_code=summary_code,
                stops=("qrs_t_morphology",) if category == "restricted" else (),
            )
        )
    return rules
