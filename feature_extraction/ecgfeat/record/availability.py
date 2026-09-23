"""Explicit availability states used by published record fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeGuard, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Measured(Generic[T]):
    value: T
    state: str = "measured"

    def __post_init__(self) -> None:
        if self.value is None:
            raise ValueError("Measured(None) is not a valid published value")
        if self.state != "measured":
            raise ValueError("Measured.state must be 'measured'")


@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: str
    detail: str | None = None
    state: str = "unavailable"

    def __post_init__(self) -> None:
        if not str(self.reason).strip():
            raise ValueError("unavailable reason must be non-empty")
        if self.state != "unavailable":
            raise ValueError("Unavailable.state must be 'unavailable'")


@dataclass(frozen=True, slots=True)
class NotApplicable:
    reason: str
    detail: str | None = None
    state: str = "not_applicable"

    def __post_init__(self) -> None:
        if not str(self.reason).strip():
            raise ValueError("not_applicable reason must be non-empty")
        if self.state != "not_applicable":
            raise ValueError("NotApplicable.state must be 'not_applicable'")


Availability = Measured[T] | Unavailable | NotApplicable


def measured(value: T) -> Measured[T]:
    return Measured(value)


def unavailable(reason: str, *, detail: str | None = None) -> Unavailable:
    return Unavailable(reason, detail)


def not_applicable(reason: str, *, detail: str | None = None) -> NotApplicable:
    return NotApplicable(reason, detail)


def is_measured(value: Availability[T]) -> TypeGuard[Measured[T]]:
    return isinstance(value, Measured)


__all__ = [
    "Availability", "Measured", "Unavailable", "NotApplicable", "measured",
    "unavailable", "not_applicable", "is_measured",
]
