"""Validation metadata for published ECG Record fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .availability import Availability, Measured

ValidationTier = Literal["published_measurement", "provenance", "internal_debug"]
# Publication category is not evidence of numerical accuracy.
PublicationTier = ValidationTier
ValidationStatusName = Literal["benchmark_validated", "indirectly_validated", "unvalidated"]


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    ref: str
    kind: str
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.ref.strip() or not self.kind.strip():
            raise ValueError("evidence references require ref and kind")


@dataclass(frozen=True, slots=True)
class FieldValidation:
    tier: ValidationTier
    evidence: tuple[EvidenceReference, ...] = ()
    notes: tuple[str, ...] = ()
    status: ValidationStatusName = "unvalidated"

    def __post_init__(self) -> None:
        if self.tier not in {"published_measurement", "provenance", "internal_debug"}:
            raise ValueError(f"unknown field validation tier: {self.tier!r}")
        if self.status not in {"benchmark_validated", "indirectly_validated", "unvalidated"}:
            raise ValueError("unknown validation status")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.status != "unvalidated" and not self.evidence:
            raise ValueError("validation claims require field-specific evidence")


def validation_for(
    *,
    tier: ValidationTier,
    evidence: Sequence[EvidenceReference] = (),
    notes: Sequence[str] = (),
    status: ValidationStatusName = "unvalidated",
) -> FieldValidation:
    return FieldValidation(tier=tier, evidence=tuple(evidence), notes=tuple(notes), status=status)


def validate_publishable_field(
    pointer: str,
    value: Availability[object],
    validation: FieldValidation,
) -> None:
    if not pointer.startswith("/"):
        raise ValueError("published field pointers must be RFC 6901 paths")
    if validation.tier == "internal_debug" and isinstance(value, Measured):
        raise ValueError("internal-debug values cannot be published as measurements")
    if isinstance(value, Measured) and value.value is None:
        raise ValueError("measured fields cannot contain None")


__all__ = ["ValidationTier", "EvidenceReference", "FieldValidation", "validation_for", "validate_publishable_field"]
