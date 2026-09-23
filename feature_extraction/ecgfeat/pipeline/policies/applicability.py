"""Input-contract applicability policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...record.availability import Availability, Measured, NotApplicable, Unavailable
from ..context import PipelineContext, PolicyDecision, PolicyEvent


@dataclass(frozen=True, slots=True)
class ApplicabilityDecision:
    states: Mapping[str, Availability[Any]]
    events: tuple[PolicyEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class ApplicabilityPolicy:
    name: str = "input_applicability_policy"

    def decide(self, field_states: Mapping[str, Availability[Any]], *, context: PipelineContext) -> PolicyDecision:
        states = dict(field_states)
        events: list[PolicyEvent] = []
        if getattr(context.config, "input_mode", None) == "limited":
            for field_name in ("formal_qt", "axis", "diagnosis"):
                states[field_name] = NotApplicable("input_mode_limited")
                events.append(PolicyEvent(self.name, "not_applicable", "input_mode_limited", (), {"field": field_name}))
        event = PolicyEvent(self.name, "no_change", "applicability_evaluated", details={"limited": getattr(context.config, "input_mode", None) == "limited"})
        events.append(event)
        return PolicyDecision(ApplicabilityDecision(states, tuple(events)), event)


__all__ = ["ApplicabilityDecision", "ApplicabilityPolicy"]
