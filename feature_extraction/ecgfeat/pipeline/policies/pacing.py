"""Deterministic pacing-context policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import PipelineContext, PolicyDecision, PolicyEvent


@dataclass(frozen=True, slots=True)
class PacingEvidence:
    spike_indices: tuple[int, ...] = ()
    validated_spike_indices: tuple[int, ...] = ()
    pacing_by_beat: Mapping[int, bool] = field(default_factory=dict)
    qrs_widths_ms: Mapping[int, float] = field(default_factory=dict)
    capture_alignment: Mapping[int, bool] = field(default_factory=dict)
    rr_ms: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PacingDecision:
    context: str
    paced_beat_ids: tuple[int, ...]
    qrs_override_ms: float | None
    selected_qrs_source: str
    events: tuple[PolicyEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class PacingPolicy:
    name: str = "default_pacing_policy"
    version: str = "1"

    def decide(self, evidence: PacingEvidence, *, context: PipelineContext) -> PolicyDecision:
        raise NotImplementedError("The pacing policy is still embedded in the legacy engine; independent decisions require the migration equivalence gate")


__all__ = ["PacingEvidence", "PacingDecision", "PacingPolicy"]
