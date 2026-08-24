"""Single fail-closed patient-age resolver shared by agent safety layers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


DAYS_PER_YEAR = 365.25
ADULT_AGE_DAYS = 18.0 * DAYS_PER_YEAR


def _finite_nonnegative(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


@dataclass(frozen=True)
class ResolvedPatientAge:
    age_years: float | None
    age_days: float | None
    source: str

    @property
    def known(self) -> bool:
        return self.age_days is not None

    @property
    def adult(self) -> bool | None:
        if self.age_days is None:
            return None
        return self.age_days >= ADULT_AGE_DAYS


def resolve_patient_age(patient: Mapping[str, Any] | None) -> ResolvedPatientAge:
    """Resolve age once, treating a supplied ``age_days`` as authoritative.

    An invalid, explicitly supplied day-age does not fall back to a possibly
    stale year-age.  This is important for neonatal routing when two source
    metadata fields disagree.
    """

    patient = patient if isinstance(patient, Mapping) else {}
    raw_days = patient.get("age_days")
    if raw_days is not None:
        days = _finite_nonnegative(raw_days)
        if days is None:
            return ResolvedPatientAge(None, None, "invalid_age_days")
        return ResolvedPatientAge(days / DAYS_PER_YEAR, days, "age_days")

    raw_years = patient.get("age")
    if raw_years is not None:
        years = _finite_nonnegative(raw_years)
        if years is None:
            return ResolvedPatientAge(None, None, "invalid_age_years")
        return ResolvedPatientAge(years, years * DAYS_PER_YEAR, "age_years")

    return ResolvedPatientAge(None, None, "missing")


def canonicalize_patient_age(patient: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy patient metadata with one internally consistent age pair."""

    normalized = dict(patient) if isinstance(patient, Mapping) else {}
    resolved = resolve_patient_age(patient)
    normalized["age"] = resolved.age_years
    normalized["age_days"] = resolved.age_days
    return normalized


__all__ = [
    "ADULT_AGE_DAYS",
    "DAYS_PER_YEAR",
    "ResolvedPatientAge",
    "canonicalize_patient_age",
    "resolve_patient_age",
]
