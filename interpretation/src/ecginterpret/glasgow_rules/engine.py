from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from .context import GlasgowContext, build_context
from .intervals import interval_rules
from .measurement_matrix import build_measurement_matrix, measurement_rule_specs
from .models import GlasgowAnalysis, GlasgowConfig, RuleEvaluation, RuleSpec
from .preliminary import preliminary_base_rules, restricted_analysis
from .preliminary_leads import lead_exclusions, preliminary_lead_rules
from .rate import rate_rules
from .registry import RuleRegistry
from .summary import resolve_summary_code, summary_rule


def build_foundation_registry() -> RuleRegistry:
    return RuleRegistry(
        [
            *preliminary_lead_rules(),
            *preliminary_base_rules(),
            *rate_rules(),
            *interval_rules(),
            summary_rule(),
            *measurement_rule_specs(),
        ]
    )


def _failed_evaluation(spec: RuleSpec, error: Exception) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=spec.rule_id,
        evaluation_status="unavailable",
        resolution_status="no_statement",
        statement=spec.statement,
        required_inputs=list(spec.required_inputs),
        missing_inputs=[],
        fidelity=spec.fidelity,
        source={"kind": spec.source_kind, "reference": spec.source_reference},
        evaluation_error=type(error).__name__,
        evidence={"error_message": str(error)},
        summary_code=spec.summary_code,
    )


def _evaluate(spec: RuleSpec, context: GlasgowContext) -> RuleEvaluation:
    try:
        result = spec.evaluator(context)
    except Exception as error:
        if context.config.strict:
            raise
        return _failed_evaluation(spec, error)
    if result.rule_id != spec.rule_id:
        error = ValueError(
            f"evaluator returned {result.rule_id!r} for registered rule {spec.rule_id!r}"
        )
        if context.config.strict:
            raise error
        return _failed_evaluation(spec, error)
    if not result.required_inputs:
        result.required_inputs = list(spec.required_inputs)
    if not result.source:
        result.source = {"kind": spec.source_kind, "reference": spec.source_reference}
    if result.summary_code is None:
        result.summary_code = spec.summary_code
    return result


def _apply_restrictions(
    evaluations: Iterable[RuleEvaluation],
    specs: Dict[str, RuleSpec],
    restrictions: Dict[str, Any],
) -> None:
    stopped_categories = set(restrictions.get("stop_categories") or [])
    stopped_by = list(restrictions.get("stopped_by") or [])
    if not stopped_categories:
        return
    category_map = {
        "qtc": "qrs_t_morphology",
        "qrs_t_morphology": "qrs_t_morphology",
    }
    for evaluation in evaluations:
        spec = specs[evaluation.rule_id]
        if (
            evaluation.evaluation_status == "matched"
            and category_map.get(spec.category) in stopped_categories
        ):
            evaluation.resolution_status = "suppressed"
            evaluation.suppressed_by = stopped_by


def _coverage(
    registry: RuleRegistry,
    evaluations: List[RuleEvaluation],
) -> Dict[str, Any]:
    by_id = {item.rule_id: item for item in evaluations}
    chapter_specs: Dict[str, List[RuleSpec]] = defaultdict(list)
    for rule in registry.rules:
        chapter_specs[rule.chapter.split(".")[0]].append(rule)

    chapters: Dict[str, Any] = {
        "3": {
            "status": "complete_via_linked_measurements",
            "registered_rule_total": 0,
            "linked_measurement_definitions": 34,
            "linked_rule_ids": [rule.rule_id for rule in chapter_specs["19"]],
        }
    }
    for chapter in ("4", "5", "6", "18", "19"):
        specs = chapter_specs[chapter]
        results = [by_id[spec.rule_id] for spec in specs]
        chapters[chapter] = {
            "status": "complete",
            "registered_rule_total": len(specs),
            "fidelity_counts": dict(Counter(spec.fidelity for spec in specs)),
            "evaluation_counts": dict(Counter(item.evaluation_status for item in results)),
            "resolution_counts": dict(Counter(item.resolution_status for item in results)),
            "unavailable_rule_ids": [
                item.rule_id for item in results if item.evaluation_status == "unavailable"
            ],
        }
    for chapter in range(7, 18):
        chapters[str(chapter)] = {
            "status": "pending_phase",
            "registered_rule_total": 0,
            "pending_rule_inventory": True,
        }
    return {
        "foundation": {
            "status": "complete",
            "registered_rule_total": len(registry.rules),
            "evaluated_rule_total": len(evaluations),
            "scope": "chapters_3_4_5_6_18_19",
            "not_certified_equivalence": True,
        },
        "chapters": chapters,
    }


def analyze_glasgow(
    features: Any,
    config: Optional[GlasgowConfig] = None,
) -> GlasgowAnalysis:
    selected_config = config or GlasgowConfig()
    registry = build_foundation_registry()
    specs = {rule.rule_id: rule for rule in registry.rules}
    context = build_context(features, selected_config)
    results: Dict[str, RuleEvaluation] = {}

    lead_specs = [rule for rule in registry.rules if rule.category == "preliminary_lead"]
    for spec in lead_specs:
        results[spec.rule_id] = _evaluate(spec, context)
    context = context.with_exclusions(lead_exclusions(results.values()))

    for spec in registry.rules:
        if spec.rule_id not in results and spec.category != "summary":
            results[spec.rule_id] = _evaluate(spec, context)

    restrictions = restricted_analysis(results.values())
    _apply_restrictions(results.values(), specs, restrictions)
    ordered_without_summary = [
        results[spec.rule_id]
        for spec in registry.rules
        if spec.category != "summary"
    ]
    final = [
        item
        for item in ordered_without_summary
        if item.evaluation_status == "matched" and item.resolution_status == "final"
    ]
    preliminary = [
        item
        for item in ordered_without_summary
        if item.evaluation_status == "matched" and item.resolution_status == "advisory"
    ]
    core_error = any(item.evaluation_error for item in ordered_without_summary)
    summary_code = resolve_summary_code(
        final,
        core_error=core_error,
        lead_limited=bool(context.excluded_leads),
    )
    summary_evaluation = RuleEvaluation(
        rule_id="GAN-18-01",
        evaluation_status="matched",
        resolution_status="no_statement",
        statement="Summary code resolution",
        evidence={
            "summary_code": summary_code,
            "final_rule_ids": [item.rule_id for item in final],
            "lead_limited": bool(context.excluded_leads),
            "core_error": core_error,
        },
        fidelity="glasgow_explicit",
        source={
            "kind": "glasgow_guide",
            "reference": "Physician's Guide chapter 18, PDF page 72",
        },
    )
    results[summary_evaluation.rule_id] = summary_evaluation
    ordered = [results[spec.rule_id] for spec in registry.rules]

    diagnostic_unavailable = [
        item
        for item in ordered
        if item.evaluation_status == "unavailable"
        and specs[item.rule_id].category not in {"measurement", "summary"}
    ]
    resolution = {
        "summary_code": summary_code,
        "preliminary": [item.to_dict() for item in preliminary],
        "final": [item.to_dict() for item in final],
        "suppressed": [
            item.to_dict() for item in ordered if item.resolution_status == "suppressed"
        ],
        "unavailable": [item.to_dict() for item in diagnostic_unavailable],
        "restricted_analysis": restrictions,
        "lead_exclusions": {
            lead: list(reasons) for lead, reasons in context.excluded_leads.items()
        },
    }
    return GlasgowAnalysis(
        schema_version="glasgow_rules.v2",
        source="20210525_glasgow_GAN_v1.0_ENG_Druck.pdf",
        config=asdict(selected_config),
        patient_route=context.patient.to_dict(),
        measurement_matrix=build_measurement_matrix(context),
        rule_evaluations=ordered,
        statement_resolution=resolution,
        coverage=_coverage(registry, ordered),
        compatibility={},
    )

