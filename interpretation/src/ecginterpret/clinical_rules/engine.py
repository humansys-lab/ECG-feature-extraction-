from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping

from .atrial_rhythm import evaluate_atrial_rhythm
from .av_block import evaluate_advanced_av_block
from .basic_rhythm import evaluate_basic_rhythm
from .conduction import evaluate_conduction
from .context import build_context
from .ectopy import evaluate_ectopy
from .findings import build_findings, serialize_findings
from .hypertrophy import evaluate_hypertrophy
from .intervals import evaluate_intervals
from .ischemia import evaluate_ischemia
from .models import ClinicalAnalysis, RuleEvaluation
from .other_patterns import evaluate_other_patterns
from .pacing import evaluate_pacing
from .preexcitation import evaluate_preexcitation
from .quality import evaluate_quality
from .r_progression import evaluate_r_progression
from .repolarization import evaluate_t_wave_abnormalities
from .resolver import ClinicalStatementResolver
from .rhythm import project_rhythm_evidence
from .t_morphology import evaluate_t_morphology
from .u_wave import evaluate_u_wave
from .voltage import evaluate_low_voltage
from .wide_tachycardia import evaluate_wide_complex_tachycardia
from .serial import evaluate_serial_comparison


REQUIRED_DOMAINS = {
    "quality",
    "rhythm",
    "conduction",
    "intervals",
    "hypertrophy",
    "atrial_abnormality",
    "voltage",
    "ischemia_infarction",
    "high_risk_patterns",
}


def _suppress_dependent_matches(
    evaluations: Iterable[RuleEvaluation], *, suppressor: str
) -> None:
    """Suppress already-resolved morphology statements confounded by a mechanism."""
    for item in evaluations:
        if item.status != "matched":
            continue
        item.status = "suppressed"
        item.coverage = "partial"
        item.confidence = "low"
        if suppressor not in item.suppressed_by:
            item.suppressed_by.append(suppressor)


def _suppress_sinus_for_advanced_av_block(
    basic_rhythm: Iterable[RuleEvaluation],
    advanced_av_block: Iterable[RuleEvaluation],
) -> None:
    """Avoid reporting a conducted sinus rhythm across unresolved AV dissociation."""
    complete = next(
        (
            item
            for item in advanced_av_block
            if item.evidence.get("evaluates_code") == "complete_av_block_pattern"
            and item.status in {"matched", "indeterminate"}
        ),
        None,
    )
    if complete is None:
        return
    suppressor = (
        "complete_av_block_pattern"
        if complete.status == "matched"
        else "possible_complete_av_block_pattern"
    )
    for item in basic_rhythm:
        if (
            item.status == "matched"
            and item.evidence.get("evaluates_code") == "sinus_mechanism"
        ):
            item.status = "suppressed"
            item.coverage = "partial"
            item.confidence = "low"
            if suppressor not in item.suppressed_by:
                item.suppressed_by.append(suppressor)


def _serializable(value: Any) -> Any:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return vars(value).copy()
    except TypeError:
        return {"value": str(value)}


def group_by_domain(
    evaluations: Iterable[RuleEvaluation],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evaluations:
        grouped.setdefault(item.domain, []).append(item.to_dict())
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["rule_id"]))
    return dict(sorted(grouped.items()))


def _summary_label(reference: Any) -> str:
    if not isinstance(reference, Mapping):
        return ""
    analysis = reference.get("analysis", reference)
    if not isinstance(analysis, Mapping):
        return ""
    resolution = analysis.get("statement_resolution")
    if isinstance(resolution, Mapping):
        summary = resolution.get("summary_code")
        if isinstance(summary, Mapping):
            return str(summary.get("label") or "")
    summary = analysis.get("summary_code")
    if isinstance(summary, Mapping):
        return str(summary.get("label") or "")
    for key in ("overall_status", "summary", "summary_label"):
        value = analysis.get(key)
        if isinstance(value, str):
            return value
    return ""


def _label_status(label: str) -> str:
    lowered = label.strip().lower()
    if not lowered:
        return ""
    if "technical" in lowered:
        return "technically_limited"
    if "abnormal" in lowered:
        return "abnormal"
    if "borderline" in lowered:
        return "borderline"
    if "normal" in lowered:
        return "normal"
    return ""


def detect_reference_conflicts(
    analysis: ClinicalAnalysis,
    references: Mapping[str, Any],
) -> list[dict[str, str]]:
    conflicts = []
    for name, reference in sorted(references.items()):
        label = _summary_label(reference)
        reference_status = _label_status(label)
        if not reference_status:
            continue
        authoritative = analysis.overall_status
        conflict = (
            reference_status == "normal" and authoritative != "normal"
        ) or (
            reference_status in {"abnormal", "borderline"}
            and authoritative == "normal"
        )
        if conflict:
            conflicts.append(
                {
                    "reference": name,
                    "reference_summary": label,
                    "authoritative_status": authoritative,
                    "reason": "reference summary differs from authoritative public-guideline resolution",
                }
            )
    final_codes = {
        str(item.get("statement_code") or "")
        for item in analysis.final_statements
        if isinstance(item, Mapping)
    }
    dxl_reference = references.get("dxl", {})
    dxl_analysis = (
        dxl_reference.get("analysis", dxl_reference)
        if isinstance(dxl_reference, Mapping)
        else {}
    )
    dxl_probable_af = bool(
        dxl_analysis.get("probable_af")
        if isinstance(dxl_analysis, Mapping)
        else False
    )
    if "atrial_flutter_pattern" in final_codes and dxl_probable_af:
        conflicts.append(
            {
                "reference": "dxl",
                "reference_statement": "probable_af",
                "authoritative_statement": "atrial_flutter_pattern",
                "reason": (
                    "legacy DXL-inspired probable AF is superseded by the "
                    "authoritative validated multilead F-wave atrial flutter result"
                ),
                "resolution": "use_authoritative_unified_statement",
            }
        )
    return conflicts


def _available_domains(evaluations: list[RuleEvaluation]) -> set[str]:
    available = set()
    for domain in REQUIRED_DOMAINS:
        relevant = [
            item
            for item in evaluations
            if item.domain == domain
            and getattr(
                item,
                "normality_role",
                "core" if item.normality_required else "optional_screen",
            ) != "supporting"
        ]
        # Supporting observations do not block normality. Optional diagnostic
        # screens do: UNKNOWN is not evidence of absence.
        if not relevant or all(
            item.status not in {"unavailable", "suppressed", "indeterminate"}
            and getattr(item, "coverage", "full") == "full"
            for item in relevant
        ):
            available.add(domain)
    return available


def analyze_clinical(
    features: Any,
    strict: bool = False,
    prior_features: Any = None,
) -> ClinicalAnalysis:
    """Resolve the authoritative public-guideline interpretation.

    ``strict`` is retained as an extension point. The current v2 rules are
    safety-conservative in both modes: missing evidence never becomes a match
    or evidence of normality.
    """
    del strict
    context = build_context(features)
    findings = build_findings(context)
    serialized_findings = serialize_findings(findings)
    evaluations = evaluate_quality(context)
    gate = features.metadata.get("diagnostic_gate", {})
    gate = gate if isinstance(gate, Mapping) else {}
    gate_state = str(gate.get("state") or "")
    morphology_complex = bool(gate.get("morphology_complex"))

    if gate_state == "stop":
        references = {
            "dxl": {
                "reference_only": True,
                "analysis": _serializable(features.interpretation),
            },
        }
        result = ClinicalStatementResolver().resolve(
            evaluations=evaluations,
            required_domains=set(REQUIRED_DOMAINS),
            available_domains=_available_domains(evaluations),
            reference_interpretations=references,
            domains=group_by_domain(evaluations),
            findings=serialized_findings,
        )
        result.conflicts = detect_reference_conflicts(result, references)
        return result

    # Rhythm mechanisms must be resolved before morphology. Pre-excitation and
    # advanced AV/pacing contexts can invalidate downstream QRS/ST-T premises.
    rhythm = project_rhythm_evidence(context)
    basic_rhythm = evaluate_basic_rhythm(context)
    pacing = evaluate_pacing(context)
    preexcitation = evaluate_preexcitation(context)
    advanced_av_block = evaluate_advanced_av_block(context)
    _suppress_sinus_for_advanced_av_block(
        basic_rhythm,
        advanced_av_block,
    )
    ectopy = evaluate_ectopy(context)
    atrial_rhythm = evaluate_atrial_rhythm(context)
    # A confirmed sinus mechanism outranks both atrial-origin screens: it is
    # decided on the multilead P axis plus the organised-P ratio, where these
    # rest on two or three lead amplitudes. If they disagree, the screens lose.
    if any(
        item.status == "matched"
        and item.evidence.get("evaluates_code") == "sinus_mechanism"
        for item in basic_rhythm
    ):
        _suppress_dependent_matches(atrial_rhythm, suppressor="sinus_mechanism")
    evaluations.extend(rhythm)
    evaluations.extend(basic_rhythm)
    evaluations.extend(atrial_rhythm)
    evaluations.extend(pacing)
    evaluations.append(preexcitation)
    evaluations.extend(advanced_av_block)
    evaluations.extend(ectopy)
    evaluations.append(evaluate_wide_complex_tachycardia(context))

    if morphology_complex:
        references = {
            "dxl": {
                "reference_only": True,
                "analysis": _serializable(features.interpretation),
            },
        }
        result = ClinicalStatementResolver().resolve(
            evaluations=evaluations,
            required_domains=set(REQUIRED_DOMAINS),
            available_domains=_available_domains(evaluations),
            reference_interpretations=references,
            domains=group_by_domain(evaluations),
            findings=serialized_findings,
        )
        result.conflicts = detect_reference_conflicts(result, references)
        return result

    evaluations.extend(evaluate_intervals(context))
    conduction = evaluate_conduction(context)
    evaluations.extend(conduction)
    hypertrophy = evaluate_hypertrophy(context, conduction)
    evaluations.extend(hypertrophy)
    evaluations.extend(evaluate_low_voltage(context))
    ischemia = evaluate_ischemia(context, conduction)
    evaluations.extend(ischemia)
    # Pre-excitation also rewrites precordial R waves, so it joins the
    # bundle-branch evaluations as a suppressor for progression statements.
    evaluations.extend(
        evaluate_r_progression(context, [*conduction, preexcitation])
    )
    evaluations.append(
        evaluate_t_wave_abnormalities(
            context,
            conduction=conduction,
            hypertrophy=hypertrophy,
            preexcitation=preexcitation,
        )
    )
    evaluations.extend(evaluate_u_wave(context, conduction=conduction))
    t_morphology = evaluate_t_morphology(
        context, conduction=conduction, hypertrophy=hypertrophy
    )
    evaluations.extend(t_morphology)
    other_patterns = evaluate_other_patterns(context)
    evaluations.extend(other_patterns)
    acute_occlusion = next(
        (
            item for item in ischemia
            if item.status == "matched"
            and item.evidence.get("evaluates_code") == "acute_occlusion_pattern"
        ),
        None,
    )
    if acute_occlusion is not None:
        for item in other_patterns:
            if (
                item.status == "matched"
                and item.evidence.get("evaluates_code")
                == "acute_pericarditis_pattern"
            ):
                item.status = "suppressed"
                item.coverage = "partial"
                item.confidence = "low"
                item.suppressed_by.append("acute_occlusion_pattern")
        # A hyperacute T wave is clinically interesting because it precedes ST
        # elevation. Once ST elevation is on the record it adds nothing and
        # would read as a second, separate finding.
        _suppress_dependent_matches(
            [
                item
                for item in t_morphology
                if item.evidence.get("evaluates_code") == "hyperacute_t_wave_pattern"
            ],
            suppressor="acute_occlusion_pattern",
        )
    hyperkalemia = next(
        (
            item
            for item in other_patterns
            if item.status == "matched"
            and item.evidence.get("evaluates_code") == "hyperkalemia_pattern"
        ),
        None,
    )
    if hyperkalemia is not None:
        # Tall symmetric T waves already explained by the hyperkalemia screen
        # must not also be reported as impending infarction.
        _suppress_dependent_matches(
            [
                item
                for item in t_morphology
                if item.evidence.get("evaluates_code") == "hyperacute_t_wave_pattern"
            ],
            suppressor="hyperkalemia_pattern",
        )
    evaluations.extend(evaluate_serial_comparison(features, prior_features))

    if preexcitation.status == "matched":
        _suppress_dependent_matches(
            [*conduction, *hypertrophy, *ischemia],
            suppressor="ventricular_preexcitation_pattern",
        )

    references = {
        "dxl": {
            "reference_only": True,
            "analysis": _serializable(features.interpretation),
        },
    }
    result = ClinicalStatementResolver().resolve(
        evaluations=evaluations,
        required_domains=set(REQUIRED_DOMAINS),
        available_domains=_available_domains(evaluations),
        reference_interpretations=references,
        domains=group_by_domain(evaluations),
        findings=serialized_findings,
    )
    result.conflicts = detect_reference_conflicts(result, references)
    return result
