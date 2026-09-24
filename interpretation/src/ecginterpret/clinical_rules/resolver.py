from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Set

from .models import ClinicalAnalysis, RuleEvaluation
from .sources import RESOLVER_POLICY_VERSION, RULESET_VERSION, SCHEMA_VERSION
from .standard_codes import codes_for


_SEVERITY_PRIORITY = {
    "critical": "P0",
    "high": "P1",
    "abnormal": "P2",
    "borderline": "P4",
    "technical": "P3",
    "observation": "P5",
    "normal": "P6",
}


def _decorate(item: RuleEvaluation) -> dict:
    row = item.to_dict()
    row["priority"] = row.get("priority") or _SEVERITY_PRIORITY.get(
        str(row.get("severity")), "P5"
    )
    raw_confidence = str(row.get("confidence") or "").strip().lower()
    if raw_confidence == "unavailable":
        standardized_confidence = "UNAVAILABLE"
    elif any(
        token in raw_confidence
        for token in ("low", "borderline", "single", "probable", "partial")
    ):
        standardized_confidence = "LOW"
    elif any(
        token in raw_confidence
        for token in ("high", "validated", "multiple", "definite")
    ):
        standardized_confidence = "HIGH"
    elif raw_confidence:
        standardized_confidence = "MEDIUM"
    else:
        standardized_confidence = (
            "HIGH" if row.get("coverage") == "full" else "LOW"
        )
    row["confidence_basis"] = row.get("confidence")
    row["confidence"] = standardized_confidence
    row["standard_codes"] = {
        **codes_for(row.get("statement_code")),
        **dict(row.get("standard_codes") or {}),
    }
    row["human_review_required"] = bool(
        row.get("human_review_required")
        or row["priority"] in {"P0", "P1", "P2", "P3", "P4"}
    )
    statement = row.get("statement")
    if statement:
        if standardized_confidence == "MEDIUM":
            row["report_statement"] = f"Consider {statement[0].lower() + statement[1:]}"
        elif standardized_confidence == "LOW":
            row["report_statement"] = (
                f"Possible {statement[0].lower() + statement[1:]}; "
                "correlate clinically"
            )
        else:
            row["report_statement"] = statement
    return row


class ClinicalStatementResolver:
    def resolve(
        self,
        evaluations: Iterable[RuleEvaluation],
        required_domains: Set[str],
        available_domains: Set[str],
        reference_interpretations: Optional[Mapping[str, Any]] = None,
        domains: Optional[Mapping[str, Any]] = None,
        findings: Optional[Mapping[str, Any]] = None,
    ) -> ClinicalAnalysis:
        rows = list(evaluations)
        unavailable_domains = sorted(set(required_domains) - set(available_domains))
        final = [_decorate(item) for item in rows if item.status == "matched"]
        borderline = [item for item in final if item["severity"] == "borderline"]
        abnormal = [
            item
            for item in final
            if item["severity"]
            not in {"normal", "borderline", "technical", "observation"}
        ]
        technical = [item for item in final if item["severity"] == "technical"]
        observations = [
            item for item in final if item["severity"] == "observation"
        ]
        serialized_domains = dict(domains or {})
        capability_coverage = {}
        for domain, evaluations_in_domain in sorted(serialized_domains.items()):
            rows_in_domain = (
                evaluations_in_domain if isinstance(evaluations_in_domain, list) else []
            )
            coverages = {
                str(item.get("coverage") or "unavailable")
                for item in rows_in_domain
                if isinstance(item, Mapping)
            }
            if not coverages:
                capability_coverage[domain] = "unavailable"
            elif coverages == {"full"}:
                capability_coverage[domain] = "full"
            elif coverages == {"unavailable"}:
                capability_coverage[domain] = "unavailable"
            else:
                capability_coverage[domain] = "partial"
        partial_evaluation = bool(unavailable_domains)
        if technical and abnormal:
            overall = "technically_limited_with_findings"
        elif technical:
            overall = "technically_limited"
        elif abnormal:
            # A confirmed abnormal finding must surface even when unrelated
            # domains are unavailable; `incomplete` only governs whether the
            # record can be called `normal`, not whether a positive finding
            # can be reported. `partial_evaluation` on the summary flags that
            # other domains were not fully evaluated and may hide additional
            # findings.
            overall = (
                "abnormal_with_limited_coverage"
                if unavailable_domains
                else "abnormal"
            )
        elif borderline:
            overall = "borderline"
        elif unavailable_domains:
            overall = "incomplete"
        else:
            overall = "normal_with_core_coverage"
        suppressed = [_decorate(item) for item in rows if item.status == "suppressed"]
        abstentions = [
            {
                "rule_id": item.rule_id,
                "domain": item.domain,
                "status": item.status,
                "missing_inputs": list(item.missing_inputs),
                "reason": (
                    "missing_or_indeterminate_required_evidence"
                    if item.status in {"unavailable", "indeterminate"}
                    else "suppressed_by_confounder"
                ),
                "suppressed_by": list(item.suppressed_by),
            }
            for item in rows
            if item.status in {"unavailable", "indeterminate", "suppressed"}
        ]
        for finding_id, finding in (findings or {}).items():
            if (
                isinstance(finding, Mapping)
                and str(finding.get("value") or "") == "UNKNOWN"
            ):
                abstentions.append(
                    {
                        "finding_id": str(finding_id),
                        "domain": "finding_layer",
                        "status": "UNKNOWN",
                        "reason": finding.get("unavailable_reason")
                        or "required_evidence_unavailable",
                        "evidence": finding.get("evidence") or [],
                    }
                )
        review_reasons = sorted({
            str(item.get("statement_code") or item.get("rule_id"))
            for item in final
            if item.get("human_review_required")
        } | {
            f"incomplete_domain:{domain}" for domain in unavailable_domains
        })
        ordering = lambda item: (
            str(item.get("priority") or "P5"),
            str(item["domain"]),
            str(item["rule_id"]),
        )
        return ClinicalAnalysis(
            schema_version=SCHEMA_VERSION,
            ruleset_version=RULESET_VERSION,
            overall_status=overall,
            summary={
                "status": overall,
                "authoritative": True,
                "resolver_policy": RESOLVER_POLICY_VERSION,
                "partial_evaluation": partial_evaluation,
                "findings": len(abnormal),
                "observations": len(observations),
                "limitations": len(technical) + len(unavailable_domains),
                "domain_coverage": {
                    domain: (
                        "full" if domain in available_domains else "unavailable"
                    )
                    for domain in sorted(required_domains)
                },
                # Includes optional P3 screens. Unlike `domain_coverage`, this
                # never silently treats a missing/partial optional capability
                # as proof of normality.
                "capability_coverage": capability_coverage,
            },
            final_statements=sorted(final, key=ordering),
            borderline_statements=sorted(borderline, key=ordering),
            suppressed_statements=sorted(suppressed, key=ordering),
            unavailable_domains=unavailable_domains,
            conflicts=[],
            domains=serialized_domains,
            reference_interpretations=dict(reference_interpretations or {}),
            generated_at=datetime.now(timezone.utc).isoformat(),
            findings=dict(findings or {}),
            abstentions=abstentions,
            review_required=bool(review_reasons),
            review_reasons=review_reasons,
        )
