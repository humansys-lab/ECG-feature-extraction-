"""Lead-integrity policy that records evidence without relabelling leads.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..context import PipelineContext, PolicyDecision, PolicyEvent
from ._events import event


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
        raise NotImplementedError(
            "a target-architecture lead-integrity decision from abstract evidence is not implemented; "
            "the legacy-parity decisions are PrecordialReversalFlagPolicy and LimbLeadReversalPolicy"
        )


# --------------------------------------------------------------------------- #
# Decision-cluster policy objects (see ``pipeline.context`` for ownership rules).
# --------------------------------------------------------------------------- #

_POLICY = "lead_integrity"
_PRECORDIAL_FLAGS = ("precordial_reversal_detail", "probable_precordial_reversal")


@dataclass(frozen=True, slots=True)
class PrecordialReversalFlagPolicy:
    """Clear precordial-reversal flags when lead-reversal detection is disabled."""

    name: str = f"{_POLICY}.precordial_reversal_flags"

    def decide(self, representative_leads: Dict[str, object]) -> PolicyDecision:
        flagged = tuple(
            lead for lead in [f"V{i}" for i in range(1, 7)]
            if isinstance(getattr(representative_leads.get(lead), "params", None), dict)
            and any(flag in representative_leads[lead].params for flag in _PRECORDIAL_FLAGS)
        )
        result = _clear_precordial_reversal_flags(representative_leads)
        outcome = event(
            self.name,
            "reject" if flagged else "no_change",
            "lead_reversal_detection_disabled",
            tuple(f"lead:{lead}" for lead in flagged),
        )
        return PolicyDecision(result, outcome)


@dataclass(frozen=True, slots=True)
class LimbLeadReversalPolicy:
    """Is a limb-electrode reversal probable (axis is then withheld in pacing-like context)?"""

    name: str = f"{_POLICY}.limb_lead_reversal"

    def decide(self, lead_reversal: Dict[str, Any]) -> PolicyDecision:
        probable = _probable_limb_lead_reversal(lead_reversal)
        return PolicyDecision(probable, event(
            self.name,
            "accept" if probable else "no_change",
            "probable_limb_lead_reversal" if probable else "no_probable_limb_lead_reversal",
            ("metadata.lead_reversal.limb",),
        ))


PRECORDIAL_REVERSAL_FLAGS = PrecordialReversalFlagPolicy()
LIMB_LEAD_REVERSAL = LimbLeadReversalPolicy()


__all__ = [
    "LeadIntegrityEvidence", "LeadIntegrityDecision", "LeadIntegrityPolicy",
    "PrecordialReversalFlagPolicy", "LimbLeadReversalPolicy",
    "PRECORDIAL_REVERSAL_FLAGS", "LIMB_LEAD_REVERSAL",
]
