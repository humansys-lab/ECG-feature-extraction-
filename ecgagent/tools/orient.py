"""Orientation tools: what the rule engines concluded, and what they refused to.

`ecgfeat.clinical_rules` is the authoritative diagnostic layer, and it already
distinguishes "no" from "cannot tell".  Surfacing both is the point: the
abstentions are where an agent has something to add, and the matched rules are
what it must verify rather than restate.
"""
from __future__ import annotations

from typing import Any

from ..evidence.store import EvidenceStore
from ._render import bullet_list, markdown_table, truncate_lines
from .registry import ToolResult, ToolSpec

MAX_FINDING_ROWS = 40

# `not_matched` and `not_applicable` are the uninformative negatives; a rule
# engine emits dozens of them per record and they crowd out the signal.
DEFAULT_STATUSES = ("matched", "indeterminate", "unavailable", "suppressed")
ALL_STATUSES = (
    "matched",
    "not_matched",
    "indeterminate",
    "unavailable",
    "not_applicable",
    "suppressed",
)
_PRIORITY_ORDER = {f"P{n}": n for n in range(7)}


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        head = ", ".join(_scalar(item) for item in value[:8])
        return f"[{head}]" if len(value) <= 8 else f"[{head}, ... {len(value)} total]"
    return str(value)


def _is_flat(node: Any) -> bool:
    return isinstance(node, dict) and all(
        not isinstance(value, dict) for value in node.values()
    )


def _weight(value: Any) -> int:
    """Rough number of output lines a value would occupy."""
    if isinstance(value, dict):
        return sum(_weight(child) for child in value.values()) or 1
    return 1


def _render_evidence(
    node: dict[str, Any],
    base: str,
    citations: list[str],
    budget: int = 44,
) -> list[str]:
    """Render a rule's evidence tree, cheapest and most decisive parts first.

    Nested evidence is the whole reason to open a rule: `criteria` holds the
    thresholds actually applied and `territory_results` holds which leads
    carried a territory.  Those are small; `lead_facts` is a 12x3 bulk dump
    that duplicates what get_lead_table renders better.  Depth-first rendering
    with one shared budget lets the bulk dump starve the decisive parts, so
    each top-level key gets its own share and small keys are emitted first.
    """
    if not node:
        return []
    entries = sorted(node.items(), key=lambda item: _weight(item[1]))
    per_key = max(4, budget // len(entries))
    lines: list[str] = []

    for index, (key, value) in enumerate(entries):
        remaining = budget - len(lines)
        if remaining <= 1:
            lines.append(f"  ... {len(entries) - index} more evidence key(s) under {base}")
            break
        # Small keys usually finish under budget; hand what they leave to the rest.
        keys_left = len(entries) - index
        allowance = max(per_key, remaining - per_key * (keys_left - 1))
        lines.extend(_render_entry(key, value, f"{base}/{key}", citations, min(allowance, remaining)))
    return lines


def _render_entry(
    key: str,
    value: Any,
    pointer: str,
    citations: list[str],
    allowance: int,
) -> list[str]:
    if not isinstance(value, dict) or not value:
        citations.append(pointer)
        return [f"  {key} = {_scalar(value)}"]

    if _is_flat(value):
        # e.g. criteria: one line per threshold, all citable.
        lines = [f"  {key}:"]
        for child_key, child_value in list(value.items())[: allowance - 1]:
            citations.append(f"{pointer}/{child_key}")
            lines.append(f"    {child_key} = {_scalar(child_value)}")
        if len(value) > allowance - 1:
            lines.append(f"    ... {len(value) - (allowance - 1)} more under {pointer}")
        return lines

    if all(_is_flat(child) for child in value.values()):
        # e.g. lead_facts / territory_results: one compact line per child.
        lines = [f"  {key}:"]
        for child_key, child in list(value.items())[: allowance - 1]:
            body = ", ".join(f"{k}={_scalar(v)}" for k, v in child.items())
            for grandchild in child:
                citations.append(f"{pointer}/{child_key}/{grandchild}")
            lines.append(f"    {child_key}: {body}"[:160])
        if len(value) > allowance - 1:
            lines.append(f"    ... {len(value) - (allowance - 1)} more under {pointer}")
        return lines

    return [f"  {key}: <{len(value)} keys; read with get_measurement {pointer}/...>"]


def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    status_rank = {"matched": 0, "suppressed": 1, "indeterminate": 2, "unavailable": 3}
    return (
        status_rank.get(str(row.get("status")), 9),
        _PRIORITY_ORDER.get(str(row.get("priority")), 9),
        str(row.get("rule_id") or ""),
    )


# ---------------------------------------------------------------------------
# list_findings
# ---------------------------------------------------------------------------
def list_findings(
    store: EvidenceStore,
    status: list[str] | str | None = None,
    domain: str | None = None,
    limit: int = MAX_FINDING_ROWS,
) -> ToolResult:
    clinical = store.clinical()
    if not clinical:
        return ToolResult.error(
            "this payload carries no clinical_interpretation block; "
            "measurements are still available through get_measurement / get_lead_table"
        )

    if status is None:
        wanted = set(DEFAULT_STATUSES)
    elif isinstance(status, str):
        wanted = {status} if status != "all" else set(ALL_STATUSES)
    else:
        wanted = set(status)
    invalid = wanted - set(ALL_STATUSES)
    if invalid:
        return ToolResult.error(
            f"unknown status {', '.join(sorted(invalid))}. Valid: {', '.join(ALL_STATUSES)}, or 'all'"
        )

    domains = clinical.get("domains") or {}
    if domain is not None and domain not in domains:
        return ToolResult.error(
            f"unknown domain {domain!r}. Available: {', '.join(sorted(domains))}"
        )

    selected: list[tuple[str, dict[str, Any]]] = []
    for base, row in store.rule_index().values():
        if domain is not None and row.get("domain") != domain:
            continue
        if str(row.get("status")) in wanted:
            selected.append((base, row))
    selected.sort(key=lambda item: _sort_key(item[1]))

    if not selected:
        return ToolResult(
            ok=True,
            text=f"no rule evaluations with status in {sorted(wanted)}"
                 + (f" for domain {domain}" if domain else ""),
        )

    limit = max(1, min(int(limit), MAX_FINDING_ROWS))
    total = len(selected)
    truncated_rows = total > limit
    selected = selected[:limit]

    citations: list[str] = []
    rows_out: list[list[str]] = []
    blocked: list[str] = []
    for base, row in selected:
        citations.extend([f"{base}/status", f"{base}/statement", f"{base}/rule_id"])
        if row.get("statement_code"):
            citations.append(f"{base}/statement_code")
        statement = row.get("statement") or row.get("statement_code") or ""
        if row.get("confidence"):
            citations.append(f"{base}/confidence")
        if row.get("priority"):
            citations.append(f"{base}/priority")
        rows_out.append(
            [
                str(row.get("rule_id") or ""),
                str(row.get("statement_code") or ""),
                str(row.get("domain") or ""),
                str(row.get("status") or ""),
                str(statement)[:52],
                str(row.get("confidence") or ""),
                str(row.get("priority") or ""),
            ]
        )
        missing = row.get("missing_inputs") or []
        suppressed_by = row.get("suppressed_by") or []
        if missing:
            citations.append(f"{base}/missing_inputs")
            blocked.append(f"{row.get('rule_id')}: needs {', '.join(str(m) for m in missing)}")
        if suppressed_by:
            citations.append(f"{base}/suppressed_by")
            blocked.append(
                f"{row.get('rule_id')}: suppressed by {', '.join(str(s) for s in suppressed_by)}"
            )

    parts = [markdown_table(
        [
            "rule_id",
            "statement_code",
            "domain",
            "status",
            "statement",
            "confidence",
            "priority",
        ],
        rows_out,
    )]
    if blocked:
        parts.append("blocked / missing inputs:\n" + bullet_list(blocked))
    if truncated_rows:
        parts.append(f"(showing {limit} of {total} rows; filter by domain or status to narrow)")

    text, truncated = truncate_lines("\n\n".join(parts), 80)
    return ToolResult(
        ok=True,
        text=text,
        citations=tuple(dict.fromkeys(citations)),
        truncated=truncated or truncated_rows,
    )


# ---------------------------------------------------------------------------
# get_rule_detail
# ---------------------------------------------------------------------------
def get_rule_detail(store: EvidenceStore, rule_id: str) -> ToolResult:
    located = store.rule_index().get(rule_id)
    if located is None:
        known = sorted(
            str(row.get("rule_id")) for row in store.rule_evaluations() if row.get("rule_id")
        )
        near = [name for name in known if rule_id.upper() in name.upper()][:8]
        hint = ("\nClosest ids:\n" + bullet_list(near)) if near else (
            "\nUse list_findings to see the available rule ids."
        )
        return ToolResult.error(f"unknown rule id {rule_id!r}.{hint}")

    base, row = located
    citations = [
        f"{base}/{key}"
        for key in ("status", "statement", "confidence", "priority")
    ]
    if row.get("statement_code"):
        citations.append(f"{base}/statement_code")

    lines = [
        f"{row.get('rule_id')}  [{row.get('domain') or 'unknown domain'}]",
        f"status      : {row.get('status')}   coverage={row.get('coverage')}  confidence={row.get('confidence')}",
        f"statement_code: {row.get('statement_code') or '(none)'}",
        f"statement   : {row.get('statement') or row.get('statement_code') or '(none)'}",
        f"severity    : {row.get('severity')}   priority={row.get('priority')}"
        f"   human_review_required={row.get('human_review_required')}",
    ]

    evidence = row.get("evidence") or {}
    if evidence:
        lines.append("evidence:")
        lines.extend(_render_evidence(evidence, f"{base}/evidence", citations))

    thresholds = row.get("thresholds") or {}
    if thresholds:
        lines.append("thresholds applied:")
        lines.extend(f"  - {key} = {value}" for key, value in thresholds.items())
        citations.extend(f"{base}/thresholds/{key}" for key in thresholds)

    for label, key in (("missing inputs", "missing_inputs"),
                       ("required inputs", "required_inputs"),
                       ("suppressed by", "suppressed_by")):
        values = row.get(key) or []
        if values:
            lines.append(f"{label}: {', '.join(str(v) for v in values)}")
            citations.append(f"{base}/{key}")

    source = row.get("source") or {}
    if source:
        lines.append(
            "source: "
            + " / ".join(str(source.get(k)) for k in ("authority", "document", "section", "version") if source.get(k))
        )

    text, truncated = truncate_lines("\n".join(lines), 70)
    return ToolResult(
        ok=True,
        text=text,
        citations=tuple(dict.fromkeys(citations)),
        truncated=truncated,
        payload=row,
    )


# ---------------------------------------------------------------------------
# get_chart_briefing
# ---------------------------------------------------------------------------
def get_chart_briefing(store: EvidenceStore) -> ToolResult:
    from ..evidence.briefing import build_chart_briefing

    briefing = build_chart_briefing(store)
    return ToolResult(
        ok=True,
        text=briefing.text,
        citations=tuple(briefing.citations),
        note="re-orientation snapshot; the same content was provided at the start of the session",
    )


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_findings",
        description=(
            "List what ecgfeat's clinical rule engine concluded. Defaults to the informative "
            "statuses: matched (asserted), indeterminate and unavailable (the engine could not "
            "decide, and says what input it lacked), and suppressed (blocked by another finding). "
            "Start here: matched rules are claims to verify, abstentions are gaps worth probing."
        ),
        parameters={
            "status": {
                "type": "array",
                "items": {"type": "string", "enum": list(ALL_STATUSES)},
                "description": "Statuses to include. Omit for the informative default, or pass 'all'.",
            },
            "domain": {
                "type": "string",
                "description": "Restrict to one domain, e.g. rhythm, conduction, ischemia_infarction, hypertrophy.",
            },
            "limit": {"type": "integer", "description": f"Max rows, 1-{MAX_FINDING_ROWS}."},
        },
        required=(),
        handler=list_findings,
    ),
    ToolSpec(
        name="get_rule_detail",
        description=(
            "Show exactly why one rule reached its status: the evidence values it used, the "
            "thresholds it applied, the inputs it was missing, what suppressed it, and the "
            "guideline it cites. Use this before agreeing or disagreeing with a rule."
        ),
        parameters={
            "rule_id": {
                "type": "string",
                "description": "Rule id from list_findings, e.g. CLIN-INTERVAL-PR-01.",
            }
        },
        required=("rule_id",),
        handler=get_rule_detail,
    ),
    ToolSpec(
        name="get_chart_briefing",
        description=(
            "Re-read the one-page record summary: gate state, quality, global intervals and axes, "
            "matched findings, abstentions and cross-engine disagreements."
        ),
        parameters={},
        required=(),
        handler=get_chart_briefing,
    ),
)
