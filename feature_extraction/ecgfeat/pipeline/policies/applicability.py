"""Measurement applicability policy.

Moved verbatim from ``ecgfeat/api.py`` in Phase 3 of the library migration
(docs/library_design/01_architecture.md, "Destination of all 49 private
``api.py`` helpers").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ...record.availability import Availability, NotApplicable
from ..context import PipelineContext, PolicyDecision, PolicyEvent
from ._events import event


def _apply_measurement_availability_to_representatives(
    representative_leads: Dict[str, object],
    global_features: object,
    availability: Dict[str, Any],
    af_afl_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Remove native AV-conduction measurements when rhythm context invalidates them."""
    if not isinstance(availability, dict):
        return
    if bool(availability.get("pr_available", True)):
        return
    reasons = set(availability.get("reasons") or [])
    # Every reason here is a positive finding that AV conduction cannot be
    # measured: the atria are fibrillating, fluttering, paced, or dissociated,
    # so a PR interval would be an artifact of pairing unrelated waves.
    #
    # `af_afl_indeterminate` is deliberately NOT in this set. It is the
    # detector explicitly declining to decide, and deleting an independently
    # measured PR on the strength of a non-decision converts uncertainty into
    # destruction of evidence. Measured on the diverse50 cohort: it fired on
    # four sinus-labelled records (09786, 09741, 09249, 09466), erasing a
    # physiologic PR of 145-162 ms from the global feature AND from every
    # representative lead. Downstream that removed the interval the agent needs
    # for first-degree AV block and pre-excitation, and the loss propagated
    # into QT conclusions through shared interval-reportability gating.
    # Consumers still see `pr_available=False` and can weigh the value
    # accordingly; they can no longer be silently denied it.
    hard_pr_unavailable = {
        "probable_af",
        "probable_flutter",
        "continuous_pacing",
        "wide_qrs_pacing_like_context",
        "complete_av_block",
    }
    rr_cv_raw = (af_afl_summary or {}).get("rr_cv") if isinstance(af_afl_summary, dict) else None
    try:
        rr_cv = float(rr_cv_raw)
    except (TypeError, ValueError):
        rr_cv = None
    irregular_atrial_unavailable = (
        "atrial_measurements_unavailable" in reasons
        and rr_cv is not None
        and np.isfinite(rr_cv)
        and rr_cv >= 0.15
    )
    if not reasons.intersection(hard_pr_unavailable) and not irregular_atrial_unavailable:
        return
    if hasattr(global_features, "pr_ms"):
        global_features.pr_ms = None
    for rep in representative_leads.values():
        params = getattr(rep, "params", None)
        if not isinstance(params, dict):
            continue
        params["pr_ms"] = None
        params["pr_consensus_ms"] = None



def _atrial_measurements_invalid_for_availability(
    global_features: object,
    af_afl_summary: Dict[str, Any],
) -> bool:
    rr_cv_raw = af_afl_summary.get("rr_cv") if isinstance(af_afl_summary, dict) else None
    try:
        rr_cv = float(rr_cv_raw)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(rr_cv) or rr_cv < 0.15:
        return False
    return (
        getattr(global_features, "pr_ms", None) is None
        and getattr(global_features, "p_axis_deg", None) is None
    )



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


# --------------------------------------------------------------------------- #
# Decision-cluster policy objects (see ``pipeline.context`` for ownership rules).
# --------------------------------------------------------------------------- #

_POLICY = "applicability"


@dataclass(frozen=True, slots=True)
class AtrialMeasurementValidityPolicy:
    """Are atrial measurements invalid for availability (irregular RR, no PR and no P axis)?"""

    name: str = f"{_POLICY}.atrial_measurements_invalid"

    def decide(self, global_features: object, af_afl_summary: Dict[str, Any]) -> PolicyDecision:
        invalid = _atrial_measurements_invalid_for_availability(global_features, af_afl_summary)
        return PolicyDecision(invalid, event(
            self.name,
            "reject" if invalid else "no_change",
            "irregular_rr_without_atrial_measurements" if invalid else "atrial_measurements_usable",
            ("global_features.pr_ms", "global_features.p_axis_deg"),
        ))


def _pr_state(representative_leads: Dict[str, object], global_features: object) -> tuple:
    rows = []
    items = representative_leads.items() if isinstance(representative_leads, dict) else ()
    for lead, rep in items:
        params = getattr(rep, "params", None)
        if isinstance(params, dict):
            rows.append((lead, "pr_ms" in params, params.get("pr_ms"),
                         "pr_consensus_ms" in params, params.get("pr_consensus_ms")))
    return getattr(global_features, "pr_ms", None), tuple(rows)


@dataclass(frozen=True, slots=True)
class PRAvailabilityPolicy:
    """Withdraw PR where rhythm context makes AV conduction unmeasurable."""

    name: str = f"{_POLICY}.pr_availability"

    def decide(
        self,
        representative_leads: Dict[str, object],
        global_features: object,
        availability: Dict[str, Any],
        af_afl_summary: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        before = _pr_state(representative_leads, global_features)
        result = _apply_measurement_availability_to_representatives(
            representative_leads,
            global_features,
            availability,
            af_afl_summary,
        )
        withdrawn = _pr_state(representative_leads, global_features) != before
        reasons = sorted(availability.get("reasons") or []) if isinstance(availability, dict) else []
        outcome = event(
            self.name,
            "reject" if withdrawn else "no_change",
            "pr_withdrawn_by_rhythm_context" if withdrawn else "pr_retained",
            ("global_features.pr_ms", "representative_leads.pr_ms", "representative_leads.pr_consensus_ms"),
            pr_available=availability.get("pr_available") if isinstance(availability, dict) else None,
            availability_reasons=reasons,
            withdrawn_pr_ms=before[0] if withdrawn else None,
        )
        return PolicyDecision(result, outcome)


ATRIAL_MEASUREMENT_VALIDITY = AtrialMeasurementValidityPolicy()
PR_AVAILABILITY = PRAvailabilityPolicy()


__all__ = [
    "ApplicabilityDecision", "ApplicabilityPolicy",
    "AtrialMeasurementValidityPolicy", "PRAvailabilityPolicy",
    "ATRIAL_MEASUREMENT_VALIDITY", "PR_AVAILABILITY",
]
