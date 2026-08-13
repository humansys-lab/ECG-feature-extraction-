"""Schema-driven diagnostic tool planning.

Clinical prose is intentionally not parsed for routing.  The model selects
enum-constrained domains and tool names; this module validates that plan,
fills deterministic domain coverage, and always preserves exact lookup and
discovery fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .protocol import DIAGNOSTIC_DOMAINS, DIAGNOSTIC_TOOLS, DOMAIN_TOOL_MAP


FALLBACK_TOOLS: tuple[str, ...] = ("get_measurement", "search_measurements")


@dataclass(frozen=True)
class ToolPlan:
    tools: tuple[str, ...]
    domains: tuple[str, ...]
    planned_views: int
    rejected_tools: tuple[str, ...] = ()
    repaired_domains: tuple[str, ...] = ()


def _append_unique(target: list[str], values: Sequence[str], *, limit: int) -> None:
    for value in values:
        name = str(value)
        if name in DIAGNOSTIC_TOOLS and name not in target:
            target.append(name)
        if len(target) >= limit:
            return


def build_tool_plan(
    state: Mapping[str, Any] | None,
    *,
    survey_state: Mapping[str, Any] | None = None,
    include_targeted_checks: bool,
    repair_survey_domains: bool = True,
    max_tools: int = 8,
) -> ToolPlan:
    # Exact lookup and discovery are never removed by a provisional plan.
    selected: list[str] = list(FALLBACK_TOOLS)
    domains: list[str] = []
    rejected: list[str] = []
    views = 0
    state = state if isinstance(state, Mapping) else {}

    checks = state.get("targeted_checks") if include_targeted_checks else []
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, Mapping):
            continue
        domain = str(check.get("domain") or "")
        if domain in DIAGNOSTIC_DOMAINS and domain not in domains:
            domains.append(domain)
        tool = str(check.get("tool") or "")
        if tool in DIAGNOSTIC_TOOLS:
            _append_unique(selected, (tool,), limit=max_tools)
            views += 1
        elif tool:
            rejected.append(tool)

    hypotheses = state.get("hypotheses")
    for row in hypotheses if isinstance(hypotheses, list) else []:
        if not isinstance(row, Mapping) or str(row.get("status") or "") in {
            "rejected",
            "abstain",
        }:
            continue
        views += 1
        for domain in row.get("domains") or []:
            domain = str(domain)
            if domain in DIAGNOSTIC_DOMAINS and domain not in domains:
                domains.append(domain)
        for tool in row.get("next_tools") or []:
            name = str(tool)
            if name in DIAGNOSTIC_TOOLS:
                _append_unique(selected, (name,), limit=max_tools)
            elif name:
                rejected.append(name)

    repaired: list[str] = []
    survey_domains = (
        survey_state.get("domains")
        if isinstance(survey_state, Mapping)
        else None
    )
    if repair_survey_domains and isinstance(survey_domains, Mapping):
        for domain, row in survey_domains.items():
            if domain not in DIAGNOSTIC_DOMAINS or not isinstance(row, Mapping):
                continue
            if str(row.get("status") or "") not in {"limited", "not_assessed"}:
                continue
            if domain not in domains:
                domains.append(str(domain))
                repaired.append(str(domain))

    for domain in domains:
        _append_unique(selected, DOMAIN_TOOL_MAP.get(domain, ()), limit=max_tools)
    return ToolPlan(
        tools=tuple(selected[:max_tools]),
        domains=tuple(domains),
        planned_views=max(views, len(domains)),
        rejected_tools=tuple(dict.fromkeys(rejected)),
        repaired_domains=tuple(repaired),
    )


__all__ = ["FALLBACK_TOOLS", "ToolPlan", "build_tool_plan"]
