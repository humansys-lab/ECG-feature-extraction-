"""The field validation-evidence registry shipped with the schema.

``validation-evidence.json`` is the only source of a published field's
validation status: the record builder copies status and evidence from it, and
strict reading rejects any record whose field claims more than the registry's
ceiling for that field.  There is no call-site override.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

from ..errors import RecordValidationError

REGISTRY_RESOURCE = ("schemas", "ecg-record", "1.0", "validation-evidence.json")
STATUS_ORDER = ("known_problem", "unvalidated", "indirectly_validated", "benchmark_validated")
# Document 06 spells the same tiers differently; readers accept both.
STATUS_ALIASES = {"validated": "benchmark_validated", "partially_validated": "indirectly_validated"}


@lru_cache(maxsize=1)
def load_registry() -> Mapping[str, Any]:
    package = resources.files("ecgfeat")
    resource = package.joinpath(*REGISTRY_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _fields() -> Mapping[str, Mapping[str, Any]]:
    return {entry["pointer"]: entry for entry in load_registry()["fields"]}


def field_entry(pointer: str) -> Mapping[str, Any]:
    try:
        return _fields()[pointer]
    except KeyError:
        raise RecordValidationError(f"{pointer}: field is not in the validation-evidence registry",
                                    code="unregistered_field") from None


def field_validation(pointer: str) -> dict[str, Any]:
    """Wire ``validation`` object for a published field, from the registry only."""

    entry = field_entry(pointer)
    return {"status": entry["validation_tier"], "evidence": list(entry["evidence_ids"])}


def rank(status: str) -> int:
    return STATUS_ORDER.index(STATUS_ALIASES.get(status, status))


def check_ceiling(pointer: str, status: str, schema_version: str) -> None:
    """Reject a claim above the registry ceiling (serialization never upgrades).

    Only records of this registry's schema family are checked: a later schema
    minor may legitimately promote a field and carries its own registry.
    """

    family = load_registry()["schema_family"]
    if ".".join(schema_version.split(".")[:2]) != family:
        return
    entry = _fields().get(pointer)
    if entry is None:
        return  # an additive field from a later schema minor: tolerated
    if rank(status) > rank(entry["validation_ceiling"]) or rank(status) > rank(entry["validation_tier"]):
        raise RecordValidationError(
            f"{pointer}: validation status {status!r} exceeds the registry "
            f"({entry['validation_tier']!r}, ceiling {entry['validation_ceiling']!r})",
            code="validation_above_registry",
        )


__all__ = ["load_registry", "field_entry", "field_validation", "check_ceiling", "rank", "STATUS_ORDER"]
