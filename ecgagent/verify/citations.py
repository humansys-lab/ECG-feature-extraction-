"""The citation verifier: does every number in the answer trace back to evidence?

Five checks, all deterministic:

1. resolvable   - each `ev:/...` pointer parses and resolves in the store
2. provenance   - each pointer was actually returned by a tool this session
3. consistency  - each written quantity matches the value its claim cites
4. support      - no unit-bearing quantity appears without a citation backing it
5. qualification- a claim citing a measurement ecgfeat distrusts must say so

Check 5 is the one that changes behaviour most: quoting a measurement the
extractor marked unreportable stops being a subtle clinical error a reviewer
has to catch, and becomes a failed build with a specific line number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..evidence.pointer import POINTER_PREFIX, PointerError
from ..evidence.store import EvidenceStore, EvidenceValue
from .numbers import (
    Quantity,
    comparable,
    extract_quantities,
    is_threshold_reference,
    normalize_unit,
    quantities_match,
    tolerance_for,
)

CITATION_RE = re.compile(re.escape(POINTER_PREFIX) + r"(/[^\s,;)\]}]+)")

# Language-generous on purpose: a legitimately hedged claim must not be
# reported, or the check gets switched off. The repo runs en/ja prompts and
# documents in zh.
_HEDGE_TERMS: tuple[str, ...] = (
    # English
    "unreliable", "not reliable", "not reportable", "unreportable", "cannot",
    "can not", "can't", "uncertain", "indeterminate", "unavailable", "limited",
    "caution", "insufficient", "not confirmed", "unconfirmed", "may ", "might ",
    "possible", "possibly", "probable", "suspected", "questionable",
    "low confidence", "not interpretable", "do not rely", "should not",
    "unstable", "degraded", "suppressed", "excluded", "flagged", "not valid",
    "provisional", "apparent", "cannot be", "unverified",
    "not produced", "was not produced", "not measured", "missing", "null",
    # Disagreement between two measurements of the same thing is one of the
    # most natural ways to qualify a value, and the caveat this layer raises
    # most often (consensus vs per-lead) is exactly that shape.
    "disagree", "differ", "discrepan", "discordant", "conflict",
    "inconsisten", "mismatch", "ambiguous", "borderline", "equivocal",
    "unresolved", "not reconciled",
    # Chinese
    "\u4e0d\u53ef\u9760", "\u4e0d\u53ef\u62a5\u544a", "\u4e0d\u786e\u5b9a", "\u65e0\u6cd5", "\u53d7\u9650", "\u8c28\u614e", "\u4e0d\u8db3", "\u672a\u786e\u8ba4",
    "\u53ef\u80fd", "\u7591\u4f3c", "\u4f4e\u7f6e\u4fe1", "\u5b58\u7591", "\u4e0d\u5b9c", "\u5df2\u6291\u5236", "\u5df2\u6392\u9664", "\u4e0d\u8db3\u4ee5",
    "\u5206\u6b67", "\u4e0d\u4e00\u81f4", "\u77db\u76fe", "\u6709\u51fa\u5165", "\u4e34\u754c",
    # Japanese
    "\u4fe1\u983c\u3067\u304d\u306a\u3044", "\u4e0d\u78ba\u5b9f", "\u5224\u5b9a\u4e0d\u80fd", "\u9650\u5b9a\u7684", "\u6ce8\u610f", "\u4e0d\u5341\u5206",
    "\u53ef\u80fd\u6027", "\u7591\u3044", "\u4f4e\u4fe1\u983c", "\u3067\u304d\u306a\u3044", "\u4e0d\u4e00\u81f4", "\u3070\u3089\u3064\u304d",
)

BLOCKING = "blocking"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    line_no: int
    excerpt: str
    detail: str
    citation: str | None = None

    def render(self) -> str:
        location = f"line {self.line_no}" if self.line_no else "output"
        return f"[{self.severity}] {self.code} @ {location}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "line_no": self.line_no,
            "excerpt": self.excerpt,
            "detail": self.detail,
            "citation": self.citation,
        }


@dataclass(frozen=True)
class VerificationPolicy:
    """Which checks run, and how hard each failure is.

    `require_provenance` is off when there is no tool session to check against,
    for instance when auditing output from the existing push-based pipeline.
    """

    require_provenance: bool = True
    require_support: bool = True
    require_qualification: bool = True
    unsupported_number_severity: str = WARNING
    missing_qualifier_severity: str = BLOCKING

    @classmethod
    def audit_only(cls) -> "VerificationPolicy":
        """Relaxed policy for measuring an existing pipeline rather than gating one."""
        return cls(
            require_provenance=False,
            unsupported_number_severity=WARNING,
            missing_qualifier_severity=WARNING,
        )

    @classmethod
    def for_structured(cls, **overrides: Any) -> "VerificationPolicy":
        """Policy for output whose evidence items carry an explicit citations array.

        An uncited number is only a *warning* in prose, where legitimate numbers
        appear without one (a guideline cutoff, a lead count). In the structured
        contract there is no such ambiguity: an evidence item with an empty
        `citations` list is an unsupported claim.

        Leaving it a warning is worse than merely lenient — it inverts the
        incentive. Dropping a citation downgrades a *blocking* `value_mismatch`
        to a *warning* `unsupported_number`, so a model can clear verification
        by citing less. Observed live: a verdict went from 15 blocking findings
        to zero by deleting its citations.
        """
        settings: dict[str, Any] = {"unsupported_number_severity": BLOCKING}
        settings.update(overrides)
        return cls(**settings)


@dataclass
class VerificationReport:
    findings: list[Finding] = field(default_factory=list)
    n_claims: int = 0
    n_citations: int = 0
    n_quantities: int = 0
    n_supported_quantities: int = 0
    n_thresholds: int = 0

    @property
    def blocking(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == BLOCKING]

    @property
    def passed(self) -> bool:
        return not self.blocking

    @property
    def support_rate(self) -> float | None:
        if not self.n_quantities:
            return None
        return self.n_supported_quantities / self.n_quantities

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for item in self.findings:
            tally[item.code] = tally.get(item.code, 0) + 1
        return tally

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        rate = "n/a" if self.support_rate is None else f"{self.support_rate:.0%}"
        lines = [
            f"{verdict}: {self.n_claims} claims, {self.n_citations} citations, "
            f"{self.n_quantities} asserted quantities ({rate} supported)"
            + (f", {self.n_thresholds} threshold reference(s) ignored" if self.n_thresholds else ""),
        ]
        for code, count in sorted(self.counts().items(), key=lambda item: -item[1]):
            lines.append(f"  {code}: {count}")
        return "\n".join(lines)

    def feedback(self, limit: int = 12) -> str:
        """Revision instructions for the model.

        Generic feedback produces generic edits, so each line names the claim,
        the pointer and the correction required.
        """
        if self.passed:
            return ""
        lines = ["The following claims failed verification. Fix each one specifically:"]
        for item in self.blocking[:limit]:
            lines.append(f"- line {item.line_no}: {item.detail}")
            if item.excerpt:
                lines.append(f'    claim: "{item.excerpt.strip()[:160]}"')
        remaining = len(self.blocking) - limit
        if remaining > 0:
            lines.append(f"- ... and {remaining} more of the same kind")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_claims": self.n_claims,
            "n_citations": self.n_citations,
            "n_quantities": self.n_quantities,
            "n_supported_quantities": self.n_supported_quantities,
            "n_thresholds": self.n_thresholds,
            "support_rate": self.support_rate,
            "counts": self.counts(),
            "findings": [item.to_dict() for item in self.findings],
        }


def _supporting_values(item: Any) -> list[dict[str, Any]]:
    """Extra cited numbers an evidence item reports besides its primary value.

    One `value` per item forces every claim into a single measurement, but the
    findings that matter are often relations between two: R against S, V1
    against V6, this beat against the last. The list is optional, so items
    written before it exist are unaffected.
    """
    if not isinstance(item, dict):
        return []
    extras = item.get("supporting_values")
    if not isinstance(extras, list):
        return []
    return [row for row in extras if isinstance(row, dict)]


def claim_is_qualified(claim: str) -> bool:
    """Return whether a claim explicitly limits a caveated measurement.

    This is public so deterministic structured-output normalization can apply
    the exact same language policy as the verifier.  Keeping two independent
    qualifier vocabularies caused safe Chinese caveats to be added in one
    layer but rejected in another.
    """
    lowered = claim.lower()
    return any(term in lowered for term in _HEDGE_TERMS)


def _mentions_caveat(claim: str, caveats: Sequence[str]) -> bool:
    """True when the claim already repeats the caveat text itself."""
    lowered = claim.lower()
    for caveat in caveats:
        head = caveat.split(";")[0].split("=")[0].strip().lower()
        if head and len(head) > 4 and head in lowered:
            return True
    return False


def _claims(text: str) -> list[tuple[int, str]]:
    """Split output into claims. One line is one claim.

    Line granularity fits how these models are prompted to answer (bulleted
    findings), and it keeps a caveat on line 9 from silently excusing an
    unqualified assertion on line 4.
    """
    return [
        (index, line)
        for index, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]


def _resolve_citations(
    store: EvidenceStore,
    claim: str,
    line_no: int,
    whitelist: frozenset[str] | None,
    policy: VerificationPolicy,
    findings: list[Finding],
) -> list[EvidenceValue]:
    resolved: list[EvidenceValue] = []
    for match in CITATION_RE.finditer(claim):
        pointer = match.group(1)
        try:
            evidence = store.resolve(pointer)
        except (PointerError, TypeError) as exc:
            findings.append(
                Finding(
                    code="unresolvable_citation",
                    severity=BLOCKING,
                    line_no=line_no,
                    excerpt=claim,
                    detail=f"citation {POINTER_PREFIX}{pointer} does not resolve: {exc}",
                    citation=pointer,
                )
            )
            continue

        if policy.require_provenance and whitelist is not None and evidence.pointer not in whitelist:
            findings.append(
                Finding(
                    code="pointer_not_from_tool",
                    severity=BLOCKING,
                    line_no=line_no,
                    excerpt=claim,
                    detail=(
                        f"citation {evidence.citation} resolves, but its value-bearing "
                        "citation token was not visible in this model session; read "
                        "it in a rendered tool result before citing it"
                    ),
                    citation=evidence.pointer,
                )
            )
            continue
        resolved.append(evidence)
    return resolved


def _check_quantities(
    claim: str,
    line_no: int,
    quantities: Sequence[Quantity],
    evidences: Sequence[EvidenceValue],
    policy: VerificationPolicy,
    findings: list[Finding],
) -> int:
    """Pair each cited value with the number that reports it, then judge the rest.

    Pairing first matters because clinical prose routinely names a threshold
    beside a measurement - "PR 214 ms, above the 200 ms limit".  Judging each
    number against the citation independently would call that 200 a
    contradiction.  A citation is only contradicted when *no* number in the
    claim reports it, which is exactly the hallucination worth blocking.
    """
    numeric = [
        evidence
        for evidence in evidences
        if isinstance(evidence.value, (int, float)) and not isinstance(evidence.value, bool)
    ]

    paired_quantities: set[int] = set()
    unpaired_evidence: list[EvidenceValue] = []
    for evidence in numeric:
        match_index = next(
            (
                index
                for index, quantity in enumerate(quantities)
                if index not in paired_quantities
                and quantities_match(quantity, float(evidence.value), evidence.unit)
            ),
            None,
        )
        if match_index is None:
            unpaired_evidence.append(evidence)
        else:
            paired_quantities.add(match_index)

    for index, quantity in enumerate(quantities):
        if index in paired_quantities:
            continue
        contradicted = [
            evidence for evidence in unpaired_evidence if comparable(quantity.unit, evidence.unit)
        ]
        if contradicted:
            expected = ", ".join(
                f"{evidence.citation} = {evidence.format_value()} {evidence.unit}"
                for evidence in contradicted[:3]
            )
            findings.append(
                Finding(
                    code="value_mismatch",
                    severity=BLOCKING,
                    line_no=line_no,
                    excerpt=claim,
                    detail=f"wrote {quantity.render()} but the cited evidence says {expected}",
                    citation=contradicted[0].pointer,
                )
            )
        elif policy.require_support:
            findings.append(
                Finding(
                    code="unsupported_number",
                    severity=policy.unsupported_number_severity,
                    line_no=line_no,
                    excerpt=claim,
                    detail=(
                        f"{quantity.render()} has no citation backing it; "
                        "read the value with a tool and cite its pointer"
                    ),
                )
            )
    return len(paired_quantities)


def _check_qualification(
    claim: str,
    line_no: int,
    evidences: Sequence[EvidenceValue],
    policy: VerificationPolicy,
    findings: list[Finding],
) -> None:
    if not policy.require_qualification:
        return
    distrusted = [evidence for evidence in evidences if evidence.caveats]
    if not distrusted or claim_is_qualified(claim):
        return
    for evidence in distrusted:
        if _mentions_caveat(claim, evidence.caveats):
            continue
        findings.append(
            Finding(
                code="missing_qualifier",
                severity=policy.missing_qualifier_severity,
                line_no=line_no,
                excerpt=claim,
                detail=(
                    f"states {evidence.citation} without qualification, but ecgfeat "
                    f"flagged it: {evidence.caveats[0]}. Either qualify the claim or "
                    "report the measurement as indeterminate."
                ),
                citation=evidence.pointer,
            )
        )


def verify_output(
    text: str,
    store: EvidenceStore,
    whitelist: frozenset[str] | None = None,
    policy: VerificationPolicy | None = None,
) -> VerificationReport:
    """Verify model output against the evidence store it was supposed to use."""
    policy = policy or VerificationPolicy()
    report = VerificationReport()

    for line_no, claim in _claims(text):
        report.n_claims += 1
        evidences = _resolve_citations(store, claim, line_no, whitelist, policy, report.findings)
        report.n_citations += len(evidences)

        # Guideline cutoffs are reference values, not assertions about this
        # record, so they are excluded before judging and before scoring.
        asserted: list[Quantity] = []
        for quantity in extract_quantities(claim):
            if is_threshold_reference(claim, quantity):
                report.n_thresholds += 1
            else:
                asserted.append(quantity)
        report.n_quantities += len(asserted)
        report.n_supported_quantities += _check_quantities(
            claim, line_no, asserted, evidences, policy, report.findings
        )
        _check_qualification(claim, line_no, evidences, policy, report.findings)

    return report


def verify_structured(
    output: dict[str, Any],
    store: EvidenceStore,
    whitelist: frozenset[str] | None = None,
    policy: VerificationPolicy | None = None,
) -> VerificationReport:
    """Verify an output following the `diagnoses[].evidence[]` contract.

    Each evidence item is flattened into one claim line so the same five checks
    apply, with citations attached rather than embedded in prose. The structural
    contract separately forbids patient-specific quantities in fields that have
    no citation slot (summary, statement, adjudication and abstentions). Defaults
    to `VerificationPolicy.for_structured()`, which blocks on uncited numbers.
    """
    policy = policy or VerificationPolicy.for_structured()
    lines: list[str] = []
    structured_items: list[tuple[int, dict[str, Any]]] = []

    def reports_structured_value(claim: str, value: Any, unit: Any) -> bool:
        """Whether claim already renders this value at an equivalent precision.

        String containment is incorrect for clinical measurements: ``50 bpm``
        is the canonical display of a raw ``50.083... bpm`` value. Use the same
        unit conversion and tolerance rules as citation verification, while
        excluding guideline thresholds from patient-value matching.
        """

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        return any(
            quantities_match(quantity, float(value), str(unit))
            for quantity in extract_quantities(claim)
            if not is_threshold_reference(claim, quantity)
        )

    def flatten(item: Any) -> None:
        """Render one evidence item as a claim line with all its citations.

        `supporting_values` carry their own pointer, so their tokens join the
        line and the numbers reporting them in the claim resolve like any other
        citation. Without that, a comparative claim - which is how QRS
        dominance, R-wave progression and inter-lead ratios are argued - could
        only be written by dropping every number but one.
        """
        if not isinstance(item, dict):
            return
        claim = str(item.get("claim") or "")
        value = item.get("value")
        unit = item.get("unit")
        if value is not None and unit and not reports_structured_value(
            claim, value, unit
        ):
            claim = f"{claim} ({value} {unit})"
        tokens = [str(token) for token in (item.get("citations") or [])]
        for extra in _supporting_values(item):
            token = str(extra.get("citation") or "").strip()
            if token:
                tokens.append(token)
        lines.append(f"{claim} {' '.join(tokens)}".strip())
        structured_items.append((len(lines), item))

    for diagnosis in output.get("diagnoses") or []:
        if not isinstance(diagnosis, dict):
            continue
        for key in ("evidence", "counterevidence"):
            for item in diagnosis.get(key) or []:
                flatten(item)
    for differential in output.get("differential_diagnoses") or []:
        if not isinstance(differential, dict):
            continue
        for key in ("supporting_evidence", "counterevidence"):
            for item in differential.get(key) or []:
                flatten(item)
    for context in output.get("interval_measurement_contexts") or []:
        if not isinstance(context, dict):
            continue
        for item in context.get("residual_evidence") or []:
            flatten(item)

    report = verify_output("\n".join(lines), store, whitelist, policy)
    # Each supporting value names exactly one pointer, so it is checked against
    # that pointer alone. This is stricter than the prose path: there is no
    # pairing ambiguity to give the benefit of the doubt to.
    for line_no, item in structured_items:
        for extra in _supporting_values(item):
            value = extra.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            token = str(extra.get("citation") or "").strip()
            match = CITATION_RE.fullmatch(token)
            evidence = None
            if match is not None:
                try:
                    evidence = store.resolve(match.group(1))
                except (PointerError, TypeError):
                    evidence = None
            if evidence is None or (
                policy.require_provenance
                and whitelist is not None
                and evidence.pointer not in whitelist
            ):
                report.findings.append(
                    Finding(
                        code="unsupported_supporting_value",
                        severity=BLOCKING,
                        line_no=line_no,
                        excerpt=str(item.get("claim") or ""),
                        detail=(
                            f"supporting value {value} {extra.get('unit') or ''}"
                            f" cites {token or '(nothing)'}, which is not a "
                            "pointer read in this session"
                        ),
                        citation=token or None,
                    )
                )
                continue
            if not isinstance(evidence.value, (int, float)) or isinstance(
                evidence.value, bool
            ):
                continue
            written = Quantity(
                value=float(value),
                unit=normalize_unit(extra.get("unit")) or (evidence.unit or ""),
                text=str(value),
                start=0,
                end=0,
            )
            if quantities_match(written, float(evidence.display_value), evidence.unit):
                continue
            report.findings.append(
                Finding(
                    code="value_mismatch",
                    severity=BLOCKING,
                    line_no=line_no,
                    excerpt=str(item.get("claim") or ""),
                    detail=(
                        f"supporting value {value} {extra.get('unit') or ''} does "
                        f"not match {evidence.citation} = "
                        f"{evidence.format_value()} {evidence.unit or ''}"
                    ),
                    citation=evidence.pointer,
                )
            )
    # Dimensionless values such as ratios are not mined from free text because
    # bare numbers are too ambiguous there. In structured evidence the numeric
    # `value` is explicit, so require a directly cited dimensionless source.
    # This blocks a model from silently computing and reporting a Q/R ratio
    # from two amplitude pointers despite the "only numbers read from tools"
    # contract.
    for line_no, item in structured_items:
        value = item.get("value")
        unit = item.get("unit")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or normalize_unit(unit) is not None
        ):
            continue
        numeric_sources: list[EvidenceValue] = []
        for token in item.get("citations") or []:
            match = CITATION_RE.fullmatch(str(token).strip())
            if match is None:
                continue
            try:
                evidence = store.resolve(match.group(1))
            except (PointerError, TypeError):
                continue
            if (
                policy.require_provenance
                and whitelist is not None
                and evidence.pointer not in whitelist
            ):
                continue
            if (
                isinstance(evidence.value, (int, float))
                and not isinstance(evidence.value, bool)
                and evidence.unit is None
            ):
                numeric_sources.append(evidence)
        if numeric_sources and any(
            abs(float(evidence.value) - float(value)) <= tolerance_for("")
            for evidence in numeric_sources
        ):
            continue
        report.findings.append(
            Finding(
                code="derived_or_unsupported_value",
                severity=BLOCKING,
                line_no=line_no,
                excerpt=str(item.get("claim") or ""),
                detail=(
                    f"structured value {value} {unit or '(dimensionless)'} is not "
                    "the value of any cited dimensionless evidence pointer; do not "
                    "report a number computed from other measurements"
                ),
            )
        )
    if output.get("diagnoses") and not any(
        (diagnosis.get("evidence") or diagnosis.get("counterevidence"))
        for diagnosis in output.get("diagnoses") or []
        if isinstance(diagnosis, dict)
    ):
        report.findings.append(
            Finding(
                code="empty_diagnostic_evidence",
                severity=BLOCKING,
                line_no=0,
                excerpt="",
                detail=(
                    "the verdict contains diagnoses but no evidence or counterevidence; "
                    "a diagnosis cannot pass verification vacuously"
                ),
            )
        )
    if output.get("differential_diagnoses") and not any(
        (
            differential.get("supporting_evidence")
            or differential.get("counterevidence")
        )
        for differential in output.get("differential_diagnoses") or []
        if isinstance(differential, dict)
    ):
        report.findings.append(
            Finding(
                code="empty_differential_evidence",
                severity=BLOCKING,
                line_no=0,
                excerpt="",
                detail=(
                    "the verdict contains differential diagnoses but no supporting "
                    "or counterevidence"
                ),
            )
        )
    return report
