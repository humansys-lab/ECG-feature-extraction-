"""Traceability audit for output that carries no citations.

The existing layered pipeline was never designed to cite its sources, so
running the citation verifier over it would only restate that by construction:
zero citations, zero supported quantities.  The question worth answering about
that pipeline is different and answerable without citations: *does every number
it wrote actually exist in the features payload it was given?*

A number that matches nothing anywhere in the payload cannot have come from the
data.  A number that does match may still be coincidence, since a 2-4 MB
payload holds tens of thousands of values and a plausible `126 ms` will collide
with something.  So the untraceable count is a **lower bound** on invented
numbers, and the traceable count is an **upper bound** on sourced ones.  Read
it as a floor, not a score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..evidence.pointer import infer_unit, walk_leaves
from ..evidence.store import EvidenceStore
from .context import field_matches_term, lead_matches, parse_claim
from .numbers import Quantity, comparable, extract_quantities, quantities_match

# Sections worth indexing. `beats`/`beat_features` are included because rhythm
# claims quote per-beat intervals; the reference-only interpretation layers are
# excluded so a match cannot be credited to a label the model was never shown.
_INDEXED_SECTIONS = (
    "global_features",
    "representative_leads",
    "beats",
    "beat_features",
    "quality",
    "metadata",
    "clinical_interpretation",
)


@dataclass(frozen=True)
class NumericLeaf:
    pointer: str
    value: float
    unit: str | None
    lead: str | None = None


@dataclass
class TraceResult:
    quantity: Quantity
    line_no: int
    claim: str
    term: str | None = None
    matches: tuple[str, ...] = ()        # unit + term + lead all agree
    loose_matches: tuple[str, ...] = ()  # magnitude and unit agree, subject does not

    @property
    def status(self) -> str:
        if self.matches:
            return "traceable"
        if self.loose_matches:
            return "coincidental"
        return "untraceable"

    @property
    def traceable(self) -> bool:
        return bool(self.matches)


@dataclass
class AuditReport:
    record_id: str
    results: list[TraceResult] = field(default_factory=list)
    n_claims: int = 0
    index_size: int = 0

    @property
    def n_quantities(self) -> int:
        return len(self.results)

    @property
    def traceable(self) -> list[TraceResult]:
        return [item for item in self.results if item.status == "traceable"]

    @property
    def coincidental(self) -> list[TraceResult]:
        return [item for item in self.results if item.status == "coincidental"]

    @property
    def untraceable(self) -> list[TraceResult]:
        return [item for item in self.results if item.status == "untraceable"]

    @property
    def unsourced(self) -> list[TraceResult]:
        """Quantities with no subject-consistent source: the number of interest."""
        return self.coincidental + self.untraceable

    @property
    def traceable_rate(self) -> float | None:
        if not self.results:
            return None
        return len(self.traceable) / len(self.results)

    def summary(self) -> str:
        if not self.results:
            return f"{self.record_id}: no unit-bearing quantities found in the output"
        rate = self.traceable_rate or 0.0
        return (
            f"{self.record_id}: {self.n_quantities} quantities in {self.n_claims} claims | "
            f"traceable {len(self.traceable)} ({rate:.0%}), "
            f"coincidental {len(self.coincidental)}, "
            f"untraceable {len(self.untraceable)} "
            f"| index={self.index_size} numeric leaves"
        )

    def detail(self, limit: int = 15) -> str:
        lines = []
        for item in self.unsourced[:limit]:
            if item.status == "untraceable":
                verdict = "matches no value in the payload"
            else:
                sample = ", ".join(item.loose_matches[:2])
                verdict = (
                    f"matches only unrelated fields ({sample}); "
                    f"nothing under subject '{item.term or 'unknown'}'"
                )
            lines.append(f"  line {item.line_no}: {item.quantity.render()} {verdict}")
            lines.append(f'    "{item.claim.strip()[:140]}"')
        remaining = len(self.unsourced) - limit
        if remaining > 0:
            lines.append(f"  ... {remaining} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "n_claims": self.n_claims,
            "n_quantities": self.n_quantities,
            "n_traceable": len(self.traceable),
            "n_coincidental": len(self.coincidental),
            "n_untraceable": len(self.untraceable),
            "traceable_rate": self.traceable_rate,
            "index_size": self.index_size,
            "unsourced": [
                {
                    "line_no": item.line_no,
                    "quantity": item.quantity.render(),
                    "status": item.status,
                    "term": item.term,
                    "loose_matches": list(item.loose_matches[:3]),
                    "claim": item.claim.strip()[:200],
                }
                for item in self.unsourced
            ],
        }


def build_numeric_index(store: EvidenceStore, max_depth: int = 6) -> list[NumericLeaf]:
    """Every finite, *dimensioned* numeric leaf in the indexed sections.

    Only leaves whose field name yields a unit are indexed.  Confidence scores,
    support counts, sample indices and ratios are numbers too, and letting them
    into the index means a written `0.25 mV` can be "traced" to a confidence of
    0.25 - a match that is worse than no match, because it looks like evidence.
    """
    # `beat_features` rows carry their lead in a field, not in the path, so the
    # lead a value belongs to has to be recovered here or lead scoping silently
    # stops constraining the largest part of the index.
    beat_leads = {
        f"/beat_features/{index}": row.get("lead")
        for index, row in enumerate(store.document.get("beat_features") or [])
        if isinstance(row, dict)
    }

    leaves: list[NumericLeaf] = []
    for section in _INDEXED_SECTIONS:
        node = store.document.get(section)
        if node is None:
            continue
        for pointer, value in walk_leaves(node, f"/{section}", max_depth=max_depth):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                continue
            unit = infer_unit(pointer.rsplit("/", 1)[-1])
            if unit is None:
                continue
            leaves.append(
                NumericLeaf(
                    pointer=pointer,
                    value=number,
                    unit=unit,
                    lead=_lead_of(pointer, beat_leads),
                )
            )
    return leaves


def _lead_of(pointer: str, beat_leads: dict[str, Any]) -> str | None:
    parts = pointer.split("/")
    if len(parts) > 2 and parts[1] in {"representative_leads", "quality"}:
        return parts[2]
    if len(parts) > 2 and parts[1] == "beat_features":
        return beat_leads.get(f"/beat_features/{parts[2]}")
    return None


def audit_untraceable_numbers(
    text: str,
    store: EvidenceStore,
    index: list[NumericLeaf] | None = None,
    max_matches: int = 3,
) -> AuditReport:
    """Check each unit-bearing number in `text` against the whole payload."""
    leaves = build_numeric_index(store) if index is None else index
    report = AuditReport(record_id=store.record_id, index_size=len(leaves))

    for line_no, claim in enumerate(text.splitlines(), start=1):
        if not claim.strip():
            continue
        report.n_claims += 1
        context = parse_claim(claim)
        for quantity in extract_quantities(claim):
            term = context.term_for(quantity.start)
            strict: list[str] = []
            loose: list[str] = []
            for leaf in leaves:
                if not comparable(quantity.unit, leaf.unit):
                    continue
                if not quantities_match(quantity, leaf.value, leaf.unit):
                    continue
                if field_matches_term(leaf.pointer, term) and lead_matches(leaf.lead, context.leads):
                    strict.append(leaf.pointer)
                    if len(strict) >= max_matches:
                        break
                elif len(loose) < max_matches:
                    loose.append(leaf.pointer)
            report.results.append(
                TraceResult(
                    quantity=quantity,
                    line_no=line_no,
                    claim=claim,
                    term=term,
                    matches=tuple(strict),
                    loose_matches=tuple(loose),
                )
            )
    return report
