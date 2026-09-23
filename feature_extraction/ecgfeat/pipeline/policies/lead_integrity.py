"""Lead-integrity policy that records evidence without relabelling leads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..context import PipelineContext, PolicyDecision, PolicyEvent


@dataclass(frozen=True, slots=True)
class LeadIntegrityEvidence:
    limb_reversal: Any = None
    precordial_reversal: Any = None
    channel_delay: Any = None


@dataclass(frozen=True, slots=True)
class LeadIntegrityDecision:
    excluded_leads: tuple[str, ...] = ()
    cleared_flags: tuple[str, ...] = ()
    events: tuple[PolicyEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class LeadIntegrityPolicy:
    name: str = "lead_integrity_policy"

    def decide(self, evidence: LeadIntegrityEvidence, *, context: PipelineContext) -> PolicyDecision:
        raise NotImplementedError("The lead_integrity policy is still embedded in the legacy engine; independent decisions require the migration equivalence gate")


__all__ = ["LeadIntegrityEvidence", "LeadIntegrityDecision", "LeadIntegrityPolicy"]
