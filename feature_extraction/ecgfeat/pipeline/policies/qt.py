"""QT reportability policy carriers and conservative default decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...record.availability import Measured, NotApplicable, Unavailable
from ..context import PipelineContext, PolicyDecision, PolicyEvent


@dataclass(frozen=True, slots=True)
class QTEvidence:
    path: Any = None
    candidates_ms: Mapping[str, float] = field(default_factory=dict)
    lead_candidates_ms: Mapping[str, float] = field(default_factory=dict)
    selected_group_id: int | None = None
    native_group_id: int | None = None
    pacing: Any = None
    tail_settling_candidate_ms: float | None = None
    reliability_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QTDecision:
    value: Any
    source: str | None
    used_leads: tuple[str, ...]
    excluded_leads: tuple[str, ...]
    events: tuple[PolicyEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class QTPolicy:
    name: str = "default_qt_rescue_policy"
    version: str = "1"

    def decide(self, evidence: QTEvidence, *, context: PipelineContext) -> PolicyDecision:
        raise NotImplementedError("The qt policy is still embedded in the legacy engine; independent decisions require the migration equivalence gate")


__all__ = ["QTEvidence", "QTDecision", "QTPolicy"]
