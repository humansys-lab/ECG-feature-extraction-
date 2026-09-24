from __future__ import annotations

from math import isfinite
from typing import Any, Dict, List, Optional

from .context import GlasgowContext
from .models import GlasgowConfig, RuleEvaluation, RuleSpec


_SOURCE = "Physician's Guide chapter 5, PDF page 15"
_ADULT_AGE_DAYS = 18.0 * 365.25


def _linear(
    age: float,
    start_age: float,
    end_age: float,
    start_value: float,
    end_value: float,
) -> float:
    fraction = (age - start_age) / (end_age - start_age)
    return float(start_value + fraction * (end_value - start_value))


def tachycardia_limit(age_days: Optional[float], config: GlasgowConfig) -> float:
    if age_days is None:
        return float(config.adult_tachycardia_bpm)
    age = max(0.0, float(age_days))
    if age <= 28.0:
        return _linear(age, 0.0, 28.0, 163.0, 180.0)
    if age <= 180.0:
        return 180.0
    if age < _ADULT_AGE_DAYS:
        return _linear(
            age,
            181.0,
            _ADULT_AGE_DAYS,
            180.0,
            config.adult_tachycardia_bpm,
        )
    return float(config.adult_tachycardia_bpm)


def bradycardia_limit(age_days: Optional[float], config: GlasgowConfig) -> float:
    if age_days is None:
        return float(config.adult_bradycardia_bpm)
    age = max(0.0, float(age_days))
    if age <= 28.0:
        return _linear(age, 0.0, 28.0, 88.0, 105.0)
    if age <= 365.0:
        return 105.0
    if age <= 2191.0:
        return _linear(age, 366.0, 2191.0, 105.0, 60.0)
    if age <= 4600.0:
        return _linear(age, 2192.0, 4600.0, 60.0, 50.0)
    return float(config.adult_bradycardia_bpm)


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _demographic_provenance(context: GlasgowContext) -> tuple[str, Optional[Dict[str, Any]]]:
    patient = context.patient
    threshold_uses_continuous_age = bool(
        patient.age_days is not None and patient.age_days < _ADULT_AGE_DAYS
    )
    if patient.age_missing:
        return (
            "existing_dxl_approximation",
            {
                "source": "existing_dxl",
                "reason": "age_missing_adult_fallback",
                "original_rule_reproducible": False,
            },
        )
    if threshold_uses_continuous_age and patient.age_days_source == "derived_from_age_years":
        return (
            "existing_dxl_approximation",
            {
                "source": "existing_dxl",
                "reason": "age_days_derived_from_age_years",
                "original_rule_reproducible": False,
            },
        )
    return "glasgow_explicit", None


def _evaluate_rate(
    context: GlasgowContext,
    *,
    rule_id: str,
    statement: str,
    threshold_name: str,
    threshold: float,
    comparator: str,
    summary_code: int,
    require_sinus: bool = False,
) -> RuleEvaluation:
    heart_rate = _finite(context.global_value("heart_rate_bpm"))
    fidelity, approximation = _demographic_provenance(context)
    common = {
        "rule_id": rule_id,
        "statement": statement,
        "required_inputs": ["global.heart_rate_bpm"],
        "fidelity": fidelity,
        "source": {"kind": "glasgow_guide", "reference": _SOURCE},
        "approximation": approximation,
        "summary_code": summary_code,
        "thresholds": {threshold_name: float(threshold)},
    }
    if heart_rate is None:
        return RuleEvaluation(
            **common,
            evaluation_status="unavailable",
            missing_inputs=["global.heart_rate_bpm"],
        )
    evidence = {
        "heart_rate_bpm": heart_rate,
        "age_days": context.patient.age_days,
        "age_days_source": context.patient.age_days_source,
        "comparison": comparator,
    }
    if require_sinus and not bool(context.rhythm.get("sinus_rhythm")):
        evidence["reason"] = "non_sinus_rhythm"
        return RuleEvaluation(
            **common,
            evaluation_status="not_applicable",
            evidence=evidence,
        )
    matched = heart_rate > threshold if comparator == "gt" else heart_rate < threshold
    return RuleEvaluation(
        **common,
        evaluation_status="matched" if matched else "not_matched",
        resolution_status="final" if matched else "no_statement",
        evidence=evidence,
    )


def evaluate_tachycardia(context: GlasgowContext) -> RuleEvaluation:
    threshold = tachycardia_limit(context.patient.age_days, context.config)
    return _evaluate_rate(
        context,
        rule_id="GAN-05.01-01",
        statement="Tachycardia",
        threshold_name="tachycardia_limit_bpm",
        threshold=threshold,
        comparator="gt",
        summary_code=2,
    )


def evaluate_bradycardia(context: GlasgowContext) -> RuleEvaluation:
    threshold = bradycardia_limit(context.patient.age_days, context.config)
    return _evaluate_rate(
        context,
        rule_id="GAN-05.02-01",
        statement="Bradycardia",
        threshold_name="bradycardia_limit_bpm",
        threshold=threshold,
        comparator="lt",
        summary_code=2,
    )


def evaluate_marked_sinus_bradycardia(context: GlasgowContext) -> RuleEvaluation:
    return _evaluate_rate(
        context,
        rule_id="GAN-05.03-01",
        statement="Marked sinus bradycardia",
        threshold_name="marked_sinus_bradycardia_bpm",
        threshold=40.0,
        comparator="lt",
        summary_code=5,
        require_sinus=True,
    )


def rate_rules() -> List[RuleSpec]:
    return [
        RuleSpec(
            rule_id="GAN-05.01-01",
            chapter="5.1",
            pdf_page=15,
            category="rate",
            order=50101,
            statement="Tachycardia",
            required_inputs=("global.heart_rate_bpm", "patient.age_days"),
            fidelity="glasgow_explicit",
            source_kind="glasgow_guide",
            source_reference=_SOURCE,
            evaluator=evaluate_tachycardia,
            summary_code=2,
        ),
        RuleSpec(
            rule_id="GAN-05.02-01",
            chapter="5.2",
            pdf_page=15,
            category="rate",
            order=50201,
            statement="Bradycardia",
            required_inputs=("global.heart_rate_bpm", "patient.age_days"),
            fidelity="glasgow_explicit",
            source_kind="glasgow_guide",
            source_reference=_SOURCE,
            evaluator=evaluate_bradycardia,
            summary_code=2,
        ),
        RuleSpec(
            rule_id="GAN-05.03-01",
            chapter="5.3",
            pdf_page=15,
            category="rate",
            order=50301,
            statement="Marked sinus bradycardia",
            required_inputs=("global.heart_rate_bpm", "rhythm.sinus_rhythm"),
            fidelity="glasgow_explicit",
            source_kind="glasgow_guide",
            source_reference=_SOURCE,
            evaluator=evaluate_marked_sinus_bradycardia,
            summary_code=5,
        ),
    ]
