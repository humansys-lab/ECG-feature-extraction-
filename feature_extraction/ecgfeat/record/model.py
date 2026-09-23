"""Immutable in-memory representation of an ECG Record document."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..errors import RecordValidationError
from .availability import Availability, Measured, NotApplicable, Unavailable
from .provenance import Provenance
from .validation import FieldValidation


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise RecordValidationError("JSON object keys must be strings")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise RecordValidationError("ECG Record cannot contain NaN or infinity", code="nonfinite_record_value")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise RecordValidationError(f"unsupported JSON value: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _availability_json(value: Availability[Any]) -> Any:
    if isinstance(value, Measured):
        return value.value
    if isinstance(value, Unavailable):
        return {"state": "unavailable", "reason": value.reason, **({"detail": value.detail} if value.detail else {})}
    if isinstance(value, NotApplicable):
        return {"state": "not_applicable", "reason": value.reason, **({"detail": value.detail} if value.detail else {})}
    raise TypeError(f"unsupported availability value: {type(value)!r}")


def _pointer_tokens(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise RecordValidationError("JSON Pointer must be empty or start with '/'", code="invalid_json_pointer")
    tokens = []
    for token in pointer[1:].split("/"):
        tokens.append(token.replace("~1", "/").replace("~0", "~"))
    return tokens


@dataclass(frozen=True, slots=True)
class RecordSidecar:
    media_type: str
    href: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class ECGRecord:
    record_id: str
    schema_version: str
    profile: str = "all"
    acquisition: Mapping[str, Any] = field(default_factory=dict)
    axes: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] | Provenance = field(default_factory=dict)
    delineation: Mapping[str, Any] = field(default_factory=dict)
    measurements: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, FieldValidation] = field(default_factory=dict)
    sidecars: tuple[RecordSidecar, ...] = ()
    extensions: Mapping[str, Any] = field(default_factory=dict)
    debug: Mapping[str, Any] = field(default_factory=dict)
    extra_members: Mapping[str, Any] = field(default_factory=dict, repr=False)
    present_optional: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip() or any(c in self.record_id for c in "@#\n\r"):
            raise RecordValidationError("record_id must be non-empty", code="missing_record_id")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise RecordValidationError("schema_version must be non-empty", code="missing_schema_version")
        if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", self.schema_version):
            raise RecordValidationError("schema_version must use SemVer", code="invalid_schema_version")
        if not isinstance(self.profile, str) or self.profile not in {"summary", "all", "measurement", "debug"}:
            raise RecordValidationError(f"unsupported record profile: {self.profile!r}", code="invalid_profile")
        for field_name in ("acquisition", "axes", "quality", "delineation", "measurements", "artifacts", "metadata", "extensions", "debug", "extra_members"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise RecordValidationError(f"{field_name} must be an object")
            object.__setattr__(self, field_name, _freeze(getattr(self, field_name)))
        if not isinstance(self.provenance, (Mapping, Provenance)):
            raise RecordValidationError("provenance must be an object")
        object.__setattr__(self, "provenance", _freeze(_provenance_dict(self.provenance)))
        object.__setattr__(self, "validation", MappingProxyType(dict(self.validation)))
        object.__setattr__(self, "sidecars", tuple(self.sidecars))
        if self.sidecars:
            raise RecordValidationError("model sidecars are not yet supported; use explicit artifact metadata")

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "record_id": self.record_id,
            "acquisition": _thaw(self.acquisition),
            "axes": _thaw(self.axes),
            "quality": _thaw(self.quality),
            "provenance": _provenance_dict(self.provenance),
            "delineation": _thaw(self.delineation),
            "measurements": _thaw(self.measurements),
            "artifacts": _thaw(self.artifacts),
        }
        if self.metadata or "metadata" in self.present_optional:
            data["metadata"] = _thaw(self.metadata)
        if self.extensions or "extensions" in self.present_optional:
            data["extensions"] = _thaw(self.extensions)
        if self.debug or "debug" in self.present_optional:
            data["debug"] = _thaw(self.debug)
        for key, value in self.extra_members.items():
            if key in data:
                raise RecordValidationError(f"extra member shadows canonical member: {key}")
            data[key] = _thaw(value)
        return data

    def field(self, pointer: str) -> Any:
        from .query import resolve_pointer
        value = resolve_pointer(self, pointer)
        tokens = _pointer_tokens(pointer)
        # The compact model-level API returns availability objects for direct
        # ``/measurements/<name>`` fields.  Rich schema paths intentionally
        # return their complete field entry so callers can address values and
        # sparse absence maps separately.
        if len(tokens) == 2 and tokens[0] == "measurements" and isinstance(value, Mapping) and "values" in value:
            if value.get("values") is not None:
                return Measured(value["values"])
            absence = value.get("absence")
            if isinstance(absence, Mapping):
                kind = absence.get("kind")
                reason = str(absence.get("reason", "unspecified"))
                if kind == "not_applicable":
                    return NotApplicable(reason)
                if kind in {"unavailable", "unmeasurable"}:
                    return Unavailable(reason)
        return value


def _provenance_dict(value: Mapping[str, Any] | Provenance) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _thaw(value)
    return {
        "library_version": value.library_version,
        "schema_version": value.schema_version,
        "config": _thaw(value.resolved_config),
        "input": {
            "sha256": value.input.sha256,
            "n_channels": value.input.n_channels,
            "n_samples": value.input.n_samples,
            "fs_hz": value.input.fs_hz,
            "channels": list(value.input.channels),
        },
        "algorithms": [
            {"name": item.name, "version": item.version, **({"parameters_sha256": item.parameters_sha256} if item.parameters_sha256 else {})}
            for item in value.algorithms
        ],
        "policy_decisions": [
            {
                "policy": item.policy,
                "decision": item.decision,
                "reason_codes": list(item.reason_codes),
                "evidence_refs": list(item.evidence_refs),
                "affected_fields": list(item.affected_fields),
            }
            for item in value.policy_decisions
        ],
        "created_by": value.created_by,
    }


def _field_entry_from_availability(value: Availability[Any]) -> dict[str, Any]:
    if isinstance(value, Measured):
        return {"type": _json_type(value.value), "unit": None, "axes": [], "values": value.value, "nullable": False}
    if isinstance(value, (Unavailable, NotApplicable)):
        return {
            "type": "string", "unit": None, "axes": [], "values": None, "nullable": True,
            # The low-level model calls the applicable-but-missing state
            # ``unavailable``; the JSON contract names the same state
            # ``unmeasurable``.  Normalize only at the wire boundary.
            "absence": {"kind": "unmeasurable" if isinstance(value, Unavailable) else "not_applicable", "reason": value.reason},
        }
    raise TypeError(type(value))


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def build_record(
    *,
    record_id: str,
    schema_version: str,
    measurements: Mapping[str, Availability[Any] | Mapping[str, Any]],
    provenance: Provenance | Mapping[str, Any],
    validation: Mapping[str, FieldValidation],
    acquisition: Mapping[str, Any],
    axes: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    sidecars: Sequence[RecordSidecar] = (),
) -> ECGRecord:
    """Build scalar global fields using the same wire layout as extraction.

    Acquisition and axes are explicit; no sampling rate or lead is invented.
    """
    fields: dict[str, Any] = {}
    for name, value in measurements.items():
        fields[str(name).lstrip("/")] = (
            _field_entry_from_availability(value)
            if isinstance(value, (Measured, Unavailable, NotApplicable))
            else dict(value)
        )
    provenance_document = _provenance_dict(provenance)
    provenance_document.setdefault("fields", {}).setdefault("imported", {"method": "caller_supplied"})
    for name, value in fields.items():
        evidence = validation.get(name)
        if evidence is not None and evidence.tier != "published_measurement":
            raise RecordValidationError("build_record only accepts published measurements")
        value.setdefault("validation", {"status": evidence.status if evidence else "unvalidated",
                                       "evidence": [item.ref for item in evidence.evidence] if evidence else []})
        value.setdefault("provenance_ref", "/provenance/fields/imported")
    if sidecars:
        raise NotImplementedError("build_record cannot link dense sidecars; use inline field values")
    return ECGRecord(
        record_id=record_id,
        schema_version=schema_version,
        profile="all",
        acquisition=acquisition, axes=axes, quality={}, provenance=provenance_document,
        delineation={"fiducials": {}}, measurements={"global": fields}, artifacts={}, metadata=metadata or {},
        validation=validation, sidecars=tuple(sidecars),
    )


def record_from_document(document: Mapping[str, Any]) -> ECGRecord:
    if not isinstance(document, Mapping):
        raise RecordValidationError("ECG Record must be a JSON object", code="record_not_object")
    required = {"schema_version", "profile", "record_id", "acquisition", "axes", "quality", "provenance", "delineation", "measurements", "artifacts"}
    missing = sorted(required - set(document))
    if missing:
        raise RecordValidationError(f"ECG Record is missing required members: {', '.join(missing)}", code="missing_record_members")
    return ECGRecord(
        record_id=document["record_id"],
        schema_version=document["schema_version"],
        profile=document["profile"],
        acquisition=document["acquisition"], axes=document["axes"], quality=document["quality"],
        provenance=document["provenance"], delineation=document["delineation"],
        measurements=document["measurements"], artifacts=document["artifacts"],
        metadata=document.get("metadata", {}), validation={},
        extensions=document.get("extensions", {}), debug=document.get("debug", {}),
        extra_members={key: value for key, value in document.items()
                       if key not in required | {"metadata", "extensions", "debug"}},
        present_optional=tuple(key for key in ("metadata", "extensions", "debug") if key in document),
    )


__all__ = ["RecordSidecar", "ECGRecord", "build_record", "record_from_document"]
