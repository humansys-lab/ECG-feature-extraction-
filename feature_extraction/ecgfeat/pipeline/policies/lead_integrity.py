"""Lead-integrity policy that records evidence without relabelling leads.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..context import PipelineContext, PolicyDecision, PolicyEvent


def _clear_precordial_reversal_flags(representative_leads: Dict[str, object]) -> None:
    for lead in [f"V{i}" for i in range(1, 7)]:
        rep = representative_leads.get(lead)
        if rep is None:
            continue
        rep.params.pop("precordial_reversal_detail", None)
        rep.params.pop("probable_precordial_reversal", None)



def _probable_limb_lead_reversal(lead_reversal: Dict[str, Any]) -> bool:
    if not isinstance(lead_reversal, dict):
        return False
    return bool(
        lead_reversal.get("probable_extremity_reversal")
        or lead_reversal.get("probable_ra_la")
        or lead_reversal.get("probable_ra_ll")
        or lead_reversal.get("probable_la_ll")
    )



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
