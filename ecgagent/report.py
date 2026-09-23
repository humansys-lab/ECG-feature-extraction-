"""Human-readable rendering for structured ECGAgent verdicts.

The model emits a constrained object so patient-specific measurements can be
verified deterministically.  This module renders that same object as both a
brief and a detailed clinician-readable Markdown report; it never asks a
second model to paraphrase the result, so the machine-readable verdict and the
displayed explanation cannot silently diverge.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping


_CONFIDENCE = {
    "HIGH": "High",
    "MEDIUM": "Medium",
    "LOW": "Low",
}
_URGENCY = {
    "EMERGENT": "Emergent",
    "URGENT": "Urgent",
    "ROUTINE": "Routine",
    "NONE": "No special urgency",
}
_INTERPRETABILITY = {
    "adequate": "Adequate",
    "limited": "Limited",
    "non_diagnostic": "Non-diagnostic",
}
_CATEGORY = {
    "quality": "Quality",
    "rhythm": "Rhythm",
    "rate": "Rate",
    "ectopy": "Ectopy",
    "conduction": "Conduction",
    "interval": "Intervals",
    "axis": "Axis",
    "chamber": "Chamber/voltage",
    "ischemia_repolarization": "Ischemia and repolarization",
    "pacing": "Pacing",
    "other": "Other",
}
_CATEGORY_ORDER = {
    "rhythm": 0,
    "rate": 1,
    "conduction": 2,
    "ectopy": 3,
    "interval": 4,
    "axis": 5,
    "chamber": 6,
    "ischemia_repolarization": 7,
    "pacing": 8,
    "quality": 9,
    "other": 10,
}
_STATUS = {
    "unchanged": "Retained",
    "added": "Added",
    "withdrawn": "Withdrawn",
    "downgraded": "Downgraded",
}
_INTERVAL_STATUS = {
    "limited": "Measurement limited",
    "unavailable": "Cannot be measured reliably",
}
_INTERVAL_NAME = {
    "PR": "PR interval and P-QRS association",
    "QT_QTc": "QT/QTc interval and T/U waves",
}
_WHITESPACE_RE = re.compile(r"\s+")


def _text(value: Any, fallback: str = "") -> str:
    rendered = _WHITESPACE_RE.sub(" ", str(value or "")).strip()
    return rendered or fallback


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _text(item))]


def _standard_diagnosis_order(
    items: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Render conclusions in the usual ECG-reading order, stably within a domain."""

    indexed = list(enumerate(items))
    indexed.sort(
        key=lambda row: (
            _CATEGORY_ORDER.get(_text(row[1].get("category")), 99),
            row[0],
        )
    )
    return [item for _, item in indexed]


def _complete_interpretations(
    verdict: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return model-authored integrated interpretations with a legacy fallback."""

    provided = _items(verdict.get("ranked_complete_interpretations"))
    if provided:
        return sorted(
            provided[:3],
            key=lambda item: (
                item.get("rank")
                if isinstance(item.get("rank"), int)
                and not isinstance(item.get("rank"), bool)
                else 99
            ),
        )

    # Older saved verdicts predate the complete-interpretation contract. Their
    # overall summary is safer than mechanically concatenating isolated finding
    # statements into a new diagnosis that the model never authored.
    diagnoses = _items(verdict.get("diagnoses"))
    confidences = {_text(item.get("confidence")) for item in diagnoses}
    confidence = (
        "LOW"
        if "LOW" in confidences or not diagnoses
        else "MEDIUM"
        if "MEDIUM" in confidences
        else "HIGH"
    )
    return [
        {
            "rank": 1,
            "interpretation_type": "PRIMARY",
            "complete_diagnosis": _text(
                verdict.get("summary"),
                "No usable complete ECG diagnosis was produced.",
            ),
            "confidence": confidence,
            "key_uncertainty": "This legacy result did not provide separate ranked complete interpretations.",
        }
    ]


def _number(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def _evidence_lines(items: Iterable[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        claim = _text(item.get("claim"), "No narrative description provided")
        details: list[str] = []
        measured = _number(item.get("value"))
        if measured is not None:
            unit = _text(item.get("unit"))
            details.append(f"Structured measurement: {measured}{(' ' + unit) if unit else ''}")
        for extra in item.get("supporting_values") or []:
            if not isinstance(extra, Mapping):
                continue
            extra_measured = _number(extra.get("value"))
            if extra_measured is None:
                continue
            extra_unit = _text(extra.get("unit"))
            extra_citation = _text(extra.get("citation"))
            details.append(
                f"Additional value: {extra_measured}"
                f"{(' ' + extra_unit) if extra_unit else ''}"
                f"{(' (' + extra_citation + ')') if extra_citation else ''}"
            )
        citations = _strings(item.get("citations"))
        if citations:
            details.append("Evidence location: " + ", ".join(citations))
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {claim}{suffix}")
    return lines


def _append_evidence_section(
    lines: list[str],
    label: str,
    items: Any,
    *,
    empty_text: str,
) -> None:
    evidence = _items(items)
    lines.append(f"  {label}:")
    if evidence:
        lines.extend(f"  {line}" for line in _evidence_lines(evidence))
    else:
        lines.append(f"  - {empty_text}")


def _brief_evidence(items: Any, *, limit: int = 2) -> list[str]:
    """Render a few evidence claims without exposing audit-only pointer noise."""

    rendered: list[str] = []
    for item in _items(items)[:limit]:
        claim = _text(item.get("claim"), "No narrative description provided")
        measurements: list[str] = []
        measured = _number(item.get("value"))
        if measured is not None:
            unit = _text(item.get("unit"))
            measurements.append(f"{measured}{(' ' + unit) if unit else ''}")
        for extra in _items(item.get("supporting_values"))[:1]:
            extra_measured = _number(extra.get("value"))
            if extra_measured is None:
                continue
            extra_unit = _text(extra.get("unit"))
            measurements.append(
                f"{extra_measured}{(' ' + extra_unit) if extra_unit else ''}"
            )
        suffix = f" ({'; '.join(measurements)})" if measurements else ""
        rendered.append(f"{claim}{suffix}")
    return rendered


def _unique_texts(values: Iterable[Any], *, limit: int) -> tuple[list[str], int]:
    """Return stable, non-empty, de-duplicated text plus its omitted count."""

    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique[:limit], max(0, len(unique) - limit)


def render_brief_report(
    verdict: Mapping[str, Any] | None,
    *,
    record_id: str = "unknown",
    verified: bool | None = None,
    error: str | None = None,
) -> str:
    """Render a short English Markdown report from the verified verdict.

    The brief form keeps the diagnosis, a bounded set of exact-value evidence,
    major limitations and review actions.  Full citations and the exhaustive
    audit explanation remain in :func:`render_human_report` and the JSON result.
    """

    status = (
        "Passed deterministic evidence validation"
        if verified is True
        else "Failed deterministic evidence validation"
        if verified is False
        else "Deterministic evidence validation was not run"
    )
    lines = [
        "# Brief ECG Diagnostic Report",
        "",
        f"- Record: {_text(record_id, 'unknown')}",
        f"- Result status: {status}",
    ]

    if not isinstance(verdict, Mapping):
        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                "The Agent did not produce a usable structured diagnosis, so no ECG conclusion can be confirmed.",
            ]
        )
        failure = _text(error)
        if failure:
            lines.append(f"- Failure reason: `{failure}`")
        lines.extend(
            [
                "",
                "## Recommendation",
                "",
                "- Review the Agent trace, resolve the runtime error, and repeat the diagnosis.",
                "",
                "> This report is for research assistance only and does not replace clinician waveform review or a formal medical diagnosis.",
            ]
        )
        return "\n".join(lines)

    if verified is not True:
        lines.extend(
            [
                "",
                "> **Caution: validation failed or was not run. The following content is only for troubleshooting and human review, not a confirmed diagnosis.**",
            ]
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            _text(verdict.get("summary"), "No overall conclusion provided."),
            "Evidence-strength labels are qualitative and uncalibrated; they are not disease probabilities.",
            "",
            (
                "## Confirmed Diagnoses"
                if verified is True
                else "## Unverified Candidates (Not Confirmed Diagnoses)"
            ),
            "",
        ]
    )

    diagnoses = _standard_diagnosis_order(_items(verdict.get("diagnoses")))
    if not diagnoses:
        lines.append("- No specific diagnosis reached the positive-evidence threshold.")
    for diagnosis in diagnoses:
        statement = _text(
            diagnosis.get("statement"),
            _text(diagnosis.get("code"), "Unnamed diagnosis"),
        )
        attributes: list[str] = []
        confidence = _text(diagnosis.get("confidence"))
        if confidence:
            attributes.append(f"Evidence strength: {_CONFIDENCE.get(confidence, confidence)}")
        urgency = _text(diagnosis.get("urgency"))
        if urgency:
            attributes.append(f"Urgency: {_URGENCY.get(urgency, urgency)}")
        suffix = f" ({'; '.join(attributes)})" if attributes else ""
        lines.append(f"- **{statement}**{suffix}")
        evidence = _brief_evidence(diagnosis.get("evidence"))
        if evidence:
            lines.append("  - Key evidence: " + "; ".join(evidence))
        counterevidence = _brief_evidence(
            diagnosis.get("counterevidence"),
            limit=1,
        )
        if counterevidence:
            lines.append("  - Main limitation: " + counterevidence[0])

    differentials = _items(verdict.get("differential_diagnoses"))
    lines.extend(["", "## Important but Unconfirmed", ""])
    if not differentials:
        lines.append("- No differential diagnosis requires separate reporting.")
    for differential in differentials[:3]:
        statement = _text(
            differential.get("statement"),
            _text(differential.get("code"), "Unnamed differential diagnosis"),
        )
        confidence = _text(differential.get("confidence"))
        confidence_text = _CONFIDENCE.get(confidence, confidence)
        suffix = f" (Evidence strength: {confidence_text})" if confidence_text else ""
        lines.append(f"- **{statement}**{suffix}")
        evidence = _brief_evidence(differential.get("supporting_evidence"))
        if evidence:
            lines.append("  - Supporting evidence: " + "; ".join(evidence))
        counterevidence = _brief_evidence(
            differential.get("counterevidence"),
            limit=1,
        )
        if counterevidence:
            lines.append("  - Why it remains unconfirmed: " + counterevidence[0])
        resolution = _text(differential.get("what_would_resolve_it"))
        if resolution:
            lines.append("  - How to resolve: " + resolution)
    if len(differentials) > 3:
        lines.append(f"- See the full report for {len(differentials) - 3} additional differential diagnoses.")

    quality = verdict.get("quality_assessment")
    quality = quality if isinstance(quality, Mapping) else {}
    limitation_candidates: list[str] = []
    interpretability = _text(quality.get("interpretability"))
    if interpretability:
        limitation_candidates.append(
            "Overall interpretability: "
            + _INTERPRETABILITY.get(interpretability, interpretability)
        )
    limitation_candidates.extend(_strings(quality.get("limitations")))
    for context in _items(verdict.get("interval_measurement_contexts")):
        interval = _text(context.get("interval"), "Unnamed interval")
        conclusion = _text(context.get("interval_conclusion"))
        if conclusion:
            limitation_candidates.append(
                f"{_INTERVAL_NAME.get(interval, interval)}: {conclusion}"
            )
    for abstention in _items(verdict.get("abstentions")):
        limitation_candidates.append(
            f"{_text(abstention.get('topic'), 'Unnamed item')}: "
            f"{_text(abstention.get('reason'), 'Available evidence is insufficient')}"
        )
    limitations, omitted_limitations = _unique_texts(
        limitation_candidates,
        limit=5,
    )
    lines.extend(["", "## Main Limitations", ""])
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- No major interpretation limitation was recorded.")
    if omitted_limitations:
        lines.append(f"- See the full report for {omitted_limitations} additional limitations.")

    review = verdict.get("human_review")
    review = review if isinstance(review, Mapping) else {}
    recommendation_candidates: list[str] = []
    recommendation_candidates.extend(_strings(review.get("reasons")))
    recommendation_candidates.extend(
        _text(item.get("what_would_resolve_it")) for item in differentials
    )
    recommendation_candidates.extend(
        _text(item.get("what_would_resolve_it"))
        for item in _items(verdict.get("interval_measurement_contexts"))
    )
    recommendation_candidates.extend(
        _text(item.get("what_would_resolve_it"))
        for item in _items(verdict.get("abstentions"))
    )
    recommendations, omitted_recommendations = _unique_texts(
        recommendation_candidates,
        limit=5,
    )
    lines.extend(["", "## Human Review Recommendations", ""])
    if review.get("required"):
        lines.append("- This result requires human review.")
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    elif not review.get("required"):
        lines.append("- The structured result does not mark human review as mandatory.")
    if omitted_recommendations:
        lines.append(f"- See the full report for {omitted_recommendations} additional recommendations.")

    lines.extend(
        [
            "",
            "> Key values come from structured ecgfeat measurements. See the full report and JSON for complete evidence paths and audit details. "
            "This report is for research assistance only and does not replace clinician waveform review, clinical information or a formal medical diagnosis.",
        ]
    )
    return "\n".join(lines).rstrip()


def render_human_report(
    verdict: Mapping[str, Any] | None,
    *,
    record_id: str = "unknown",
    verified: bool | None = None,
    knowledge_navigation: Mapping[str, Any] | None = None,
    knowledge_review: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> str:
    """Render one structured verdict as an auditable English Markdown report."""
    lines = [
        "# ECG Agent Diagnostic Interpretation Report",
        "",
        f"- Record: {_text(record_id, 'unknown')}",
    ]
    if verified is True:
        lines.append(
            "- Evidence and semantic validation: passed measurement-citation, reliability-qualification, numeric-logic, "
            "definition-level diagnosis-to-measurement consistency, and candidate-promotion checks"
        )
    elif verified is False:
        lines.append("- Evidence and semantic validation: failed; the following content is not a confirmed conclusion")
    else:
        lines.append("- Evidence and semantic validation: not run or status unknown")

    if not isinstance(verdict, Mapping):
        failure = _text(error)
        lines.extend(
            [
                "",
                "## Overall Conclusion",
                "",
                "The Agent stopped before completing the final structured diagnosis; this record has no confirmed diagnosis.",
                *(
                    ["", f"- Failure reason: `{failure}`"]
                    if failure
                    else []
                ),
                "- Completed measurement acquisition and phase outputs remain available in the Agent trace for troubleshooting.",
                "",
                "> This report is a research aid and does not replace clinician waveform review, clinical information or a formal medical diagnosis.",
            ]
        )
        return "\n".join(lines)

    diagnoses_raw = _items(verdict.get("diagnoses"))
    complete_interpretations = _complete_interpretations(verdict)
    top_heading = (
        "## Ranked ECG Findings and Interpretations"
        if verified is True
        else "## Top 3 ECG Interpretation Candidates (Unconfirmed)"
    )
    lines.extend(["", top_heading, "",
                  "Evidence-strength labels are qualitative and uncalibrated; they are not disease probabilities.", ""])
    if verified is not True:
        lines.append(
            "> Caution: evidence and semantic validation failed or has unknown status. This ranking is for troubleshooting only, not a confirmed diagnosis."
        )
        if _text(error):
            lines.append(f"> Stop reason: `{_text(error)}`")
        lines.append("")
    for index, interpretation in enumerate(complete_interpretations, start=1):
        diagnosis = _text(
            interpretation.get("complete_diagnosis"),
            "No usable complete ECG diagnosis was produced.",
        )
        confidence = _text(interpretation.get("confidence"))
        interpretation_type = _text(interpretation.get("interpretation_type"))
        type_text = "Primary interpretation" if interpretation_type == "PRIMARY" else "Alternative interpretation"
        confidence_text = _CONFIDENCE.get(confidence, confidence)
        suffix = f" ({type_text}; overall evidence strength: {confidence_text})"
        lines.append(f"{index}. **{diagnosis}**{suffix}")
        uncertainty = _text(interpretation.get("key_uncertainty"))
        if uncertainty:
            lines.append(f"   - Key uncertainty: {uncertainty}")

    lines.extend(
        [
            "",
            "## Overall Conclusion",
            "",
            _text(verdict.get("summary"), "No overall conclusion provided."),
            "",
            "## Reporting Levels",
            "",
            (
                "- Major ECG findings include only findings that reached the positive-interpretation evidence threshold."
                if verified is True
                else "- Validation did not pass. All diagnosis entries are unverified candidates, and no conclusion reached the positive threshold."
            ),
            "- Candidate-detector results and insufficiently supported mechanisms are listed under Differential Diagnoses and are not confirmed diagnoses.",
            "- Items that cannot be assessed reliably are listed as limited or abstained; missing values are never used to invent a diagnosis.",
        ]
    )

    if isinstance(knowledge_navigation, Mapping) and knowledge_navigation.get(
        "enabled"
    ):
        status = _text(knowledge_navigation.get("status"), "unknown")
        count = knowledge_navigation.get("reference_count")
        count = int(count) if isinstance(count, (int, float)) else 0
        status_text = {
            "matched": f"Retrieved {count} applicable general-knowledge excerpts",
            "no_knowledge_match": "No applicable general-knowledge excerpt was retrieved",
            "error": "Knowledge navigation did not complete; subsequent checks were planned only from the measurement survey",
            "pending": "Knowledge-navigation status has not yet been finalized",
        }.get(status, status)
        lines.extend(
            [
                "",
                "## Medical Knowledge Navigation Before Provisional Diagnosis",
                "",
                f"- Navigation status: {status_text}.",
                "- Purpose: form provisional diagnoses, differential diagnoses and targeted ecgfeat checks after the global measurement survey.",
                "- Evidence boundary: this is general medical knowledge, not patient evidence. Final conclusions accept only patient measurements with `ev:/...` citations.",
            ]
        )

    lines.extend(
        [
            "",
            "## Data Quality and Interpretation Limitations",
            "",
        ]
    )

    quality = verdict.get("quality_assessment")
    quality = quality if isinstance(quality, Mapping) else {}
    interpretability = _text(quality.get("interpretability"))
    if interpretability:
        lines.append(
            f"- Overall interpretability: {_INTERPRETABILITY.get(interpretability, interpretability)}"
        )
    limitations = _strings(quality.get("limitations"))
    if limitations:
        lines.extend(f"- {limitation}" for limitation in limitations)
    elif not interpretability:
        lines.append("- No separate data-quality assessment was provided.")

    diagnosis_heading = (
        "## Major ECG Findings (Positive-Interpretation Threshold Met)"
        if verified is True
        else "## Unverified Candidate Interpretations (Not Positive Conclusions)"
    )
    lines.extend(["", diagnosis_heading, ""])

    diagnoses = _standard_diagnosis_order(diagnoses_raw)
    if not diagnoses:
        lines.append(
            (
                "No diagnosis reached the positive-conclusion threshold. Review the differential diagnoses, measurement limitations and human-review recommendations."
                if verified is True
                else "No diagnostic candidate was generated for troubleshooting. Review measurement limitations and runtime errors."
            )
        )
    for index, diagnosis in enumerate(diagnoses, start=1):
        statement = _text(
            diagnosis.get("statement"),
            _text(diagnosis.get("code"), "Unnamed diagnosis"),
        )
        code = _text(diagnosis.get("code"))
        lines.append(f"### {index}. {statement}")
        attributes: list[str] = []
        if code:
            attributes.append(f"Code: {code}")
        category = _text(diagnosis.get("category"))
        if category:
            attributes.append(f"Category: {_CATEGORY.get(category, category)}")
        confidence = _text(diagnosis.get("confidence"))
        if confidence:
            attributes.append(
                f"Evidence strength: {_CONFIDENCE.get(confidence, confidence)}"
            )
        urgency = _text(diagnosis.get("urgency"))
        if urgency:
            attributes.append(f"Urgency: {_URGENCY.get(urgency, urgency)}")
        status = _text(diagnosis.get("status"))
        if status:
            attributes.append(f"Rule-comparison status: {_STATUS.get(status, status)}")
        if attributes:
            lines.append("- " + "; ".join(attributes))

        explanation = _text(
            diagnosis.get("reasoning") or diagnosis.get("adjudication"),
            "No additional reasoning provided.",
        )
        lines.append(f"- Rationale: {explanation}")
        _append_evidence_section(
            lines,
            "Supporting evidence",
            diagnosis.get("evidence"),
            empty_text="No supporting evidence listed",
        )
        _append_evidence_section(
            lines,
            "Counterevidence or inconsistencies",
            diagnosis.get("counterevidence"),
            empty_text="No explicit counterevidence was recorded among the evidence reviewed in this run",
        )
        lines.append("")

    lines.extend(["## Differential Diagnoses (Unconfirmed Candidates)", ""])
    differentials = _items(verdict.get("differential_diagnoses"))
    if not differentials:
        lines.append("No differential diagnosis requires separate reporting.")
        lines.append("")
    for index, differential in enumerate(differentials, start=1):
        statement = _text(
            differential.get("statement"),
            _text(differential.get("code"), "Unnamed differential diagnosis"),
        )
        confidence = _text(differential.get("confidence"))
        confidence_text = _CONFIDENCE.get(confidence, confidence)
        lines.append(f"### {index}. {statement}")
        if confidence_text:
            lines.append(f"- Current evidence strength: {confidence_text}")
        _append_evidence_section(
            lines,
            "Evidence supporting this possibility",
            differential.get("supporting_evidence"),
            empty_text="No supporting evidence listed",
        )
        _append_evidence_section(
            lines,
            "Evidence against this possibility",
            differential.get("counterevidence"),
            empty_text="No explicit counterevidence was recorded among the evidence reviewed in this run",
        )
        lines.append(
            "- How to resolve: "
            + _text(
                differential.get("what_would_resolve_it"),
                "A clinician must review the original waveforms and clinical information.",
            )
        )
        lines.append("")

    lines.extend(["## Waveform Information Retained When Intervals Are Limited", ""])
    interval_contexts = _items(verdict.get("interval_measurement_contexts"))
    if not interval_contexts:
        lines.append("No limited PR or QT/QTc measurement requires separate reporting.")
        lines.append("")
    for index, context in enumerate(interval_contexts, start=1):
        interval = _text(context.get("interval"), "Unnamed interval")
        title = _INTERVAL_NAME.get(interval, interval)
        status = _text(context.get("status"))
        lines.append(f"### {index}. {title}")
        if status:
            lines.append(f"- Measurement status: {_INTERVAL_STATUS.get(status, status)}")
        lines.append(
            "- Interval: "
            + _text(context.get("interval_conclusion"), "The reason for interval limitation was not provided.")
        )
        lines.append(
            "- Component waveform still shows: "
            + _text(
                context.get("component_waveform_assessment"),
                "No remaining usable waveform information was described.",
            )
        )
        _append_evidence_section(
            lines,
            "Retained waveform evidence",
            context.get("residual_evidence"),
            empty_text="No residual waveform evidence listed",
        )
        lines.append(
            "- Diagnostic impact: "
            + _text(
                context.get("interpretive_impact"),
                "Available evidence is insufficient to define the diagnostic impact of this limitation.",
            )
        )
        lines.append(
            "- How to resolve: "
            + _text(
                context.get("what_would_resolve_it"),
                "A clinician must review the original waveforms.",
            )
        )
        lines.append("")

    lines.extend(["## Indeterminate Items", ""])
    abstentions = _items(verdict.get("abstentions"))
    if not abstentions:
        lines.append("No indeterminate item requires separate reporting.")
    for abstention in abstentions:
        topic = _text(abstention.get("topic"), "Unnamed item")
        reason = _text(abstention.get("reason"), "Available evidence is insufficient")
        resolution = _text(
            abstention.get("what_would_resolve_it"),
            "Additional original waveforms or clinical information is required",
        )
        lines.append(f"- {topic}: {reason}. Required: {resolution}.")

    if isinstance(knowledge_review, Mapping):
        lines.extend(["", "## Medical Knowledge Challenge Review", ""])
        status = _text(knowledge_review.get("status"), "unknown")
        status_text = {
            "passed": "No material issue requiring new evidence was found",
            "issues_found": "Issues requiring new evidence were found",
            "no_knowledge_match": "No applicable knowledge excerpt was retrieved",
            "error": "The knowledge challenge did not complete",
        }.get(status, status)
        lines.append(f"- Challenge status: {status_text}")
        summary = _text(knowledge_review.get("summary"))
        if summary:
            lines.append(f"- Challenge summary: {summary}")
        revision = knowledge_review.get("revision")
        revision = revision if isinstance(revision, Mapping) else {}
        if revision:
            lines.append(
                "- Evidence reacquisition: "
                + (
                    "ecgfeat tools were called for review, and the revised result passed evidence validation"
                    if revision.get("applied")
                    else "attempted, but a revision that failed structural or evidence validation was not applied"
                )
            )
        issues = _items(knowledge_review.get("issues"))
        if not issues:
            lines.append("- No specific challenge issue was raised.")
        for index, issue in enumerate(issues, start=1):
            issue_type = {
                "omission": "Possible omission",
                "confounder": "Confounder",
                "causal_jump": "Causal leap",
                "measurement_limitation": "Measurement limitation",
            }.get(_text(issue.get("type")), _text(issue.get("type"), "Issue"))
            lines.append(
                f"- {index}. [{issue_type}/{_text(issue.get('priority'), 'Unrated')}] "
                f"{_text(issue.get('topic'), 'Unnamed topic')}: "
                f"{_text(issue.get('evidence_review_question'), 'Patient measurement evidence requires re-examination.')}"
            )
        used_refs = {
            str(ref)
            for issue in issues
            for ref in (issue.get("knowledge_refs") or [])
        }
        references = [
            row
            for row in _items(knowledge_review.get("references"))
            if str(row.get("ref") or "") in used_refs
        ]
        for row in references:
            lines.append(
                "  - Knowledge reference: "
                f"{_text(row.get('ref'))} · {_text(row.get('source'))}:"
                f"{_text(row.get('line_start'))}-{_text(row.get('line_end'))} · "
                f"{_text(row.get('title'))}"
            )
        lines.append(
            "- Boundary: the knowledge citations above are general review references, not patient evidence. Patient conclusions still accept only "
            "ecgfeat measurements with `ev:/...` citations."
        )

    review = verdict.get("human_review")
    review = review if isinstance(review, Mapping) else {}
    lines.extend(["", "## Human Review Recommendations", ""])
    if review.get("required"):
        lines.append("- Human review is required.")
    else:
        lines.append("- The structured result does not mark human review as mandatory.")
    reasons = _strings(review.get("reasons"))
    lines.extend(f"- {reason}" for reason in reasons)

    if verified is True:
        footer = (
            "> Note: deterministic validation confirms that measurements, citations and reliability qualifications in the report match ecgfeat output, "
            "and applies numeric-logic, definition-level diagnosis and candidate-promotion gates. It does not cover all clinical diagnostic knowledge "
            "and does not mean that the clinical diagnosis has been validated. This report is a research aid and does not replace clinician waveform review, "
            "history, symptoms, prior testing or a formal medical diagnosis."
        )
    else:
        footer = (
            "> Note: deterministic validation failed or was not run, so measurements, citations, reliability qualifications and diagnostic promotion "
            "in candidate content cannot be confirmed as consistent with ecgfeat. The content above is only for troubleshooting and human review, "
            "not a confirmed conclusion or formal medical diagnosis."
        )
    lines.extend(["", footer])
    return "\n".join(lines).rstrip()
