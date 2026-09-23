"""Stable RFC 6901 addressing and measurement queries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from ..errors import (
    AddressNotFoundError,
    AddressSyntaxError,
    MeasurementNotFoundError,
    MeasurementSelectorError,
    RecordIdentityError,
)
from .model import ECGRecord


@dataclass(frozen=True, slots=True)
class RecordAddress:
    record_id: str
    schema_version: str
    pointer: str

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id or any(c in self.record_id for c in "@#\n\r"):
            raise AddressSyntaxError("invalid record identity")
        if not isinstance(self.schema_version, str) or not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", self.schema_version):
            raise AddressSyntaxError("invalid schema version")
        if not isinstance(self.pointer, str) or (self.pointer and not self.pointer.startswith("/")):
            raise AddressSyntaxError("invalid JSON Pointer")
        for token in self.pointer[1:].split("/") if self.pointer else []:
            _unescape(token)

    def __str__(self) -> str:
        return f"ecg-record:{self.record_id}@{self.schema_version}#{self.pointer}"


def _escape(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _unescape(token: str) -> str:
    # RFC 6901 only recognizes these two escapes.  Reject malformed '~'
    # sequences instead of silently resolving a different field.
    pieces: list[str] = []
    index = 0
    while index < len(token):
        char = token[index]
        if char != "~":
            pieces.append(char)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise AddressSyntaxError("malformed RFC 6901 escape", code="invalid_json_pointer")
        pieces.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(pieces)


def parse_address(value: str) -> RecordAddress:
    if not isinstance(value, str) or not value.startswith("ecg-record:"):
        raise AddressSyntaxError("address must start with 'ecg-record:'", code="invalid_record_address")
    body = value[len("ecg-record:"):]
    if "#" not in body or "@" not in body.split("#", 1)[0]:
        raise AddressSyntaxError("address must be ecg-record:<id>@<schema>#<pointer>", code="invalid_record_address")
    identity, pointer = body.split("#", 1)
    record_id, schema = identity.rsplit("@", 1)
    if not record_id or not schema or (pointer and not pointer.startswith("/")):
        raise AddressSyntaxError("invalid record identity or JSON Pointer", code="invalid_record_address")
    # Validate token escapes now, before a caller can accidentally store a bad citation.
    for token in pointer[1:].split("/") if pointer else []:
        _unescape(token)
    return RecordAddress(record_id, schema, pointer)


parse_record_address = parse_address


def make_record_address(record: ECGRecord, pointer: str) -> RecordAddress:
    if pointer and not pointer.startswith("/"):
        raise AddressSyntaxError("JSON Pointer must start with '/'", code="invalid_json_pointer")
    return RecordAddress(record.record_id, record.schema_version, pointer)


@dataclass(frozen=True, slots=True)
class AddressResolution:
    address: RecordAddress
    value: Any
    node_kind: Literal["record", "measurement", "metadata", "provenance", "debug"]


def resolve_pointer(record: ECGRecord, pointer: str) -> Any:
    if pointer == "":
        return record.as_dict()
    if not pointer.startswith("/"):
        raise AddressSyntaxError("JSON Pointer must start with '/'", code="invalid_json_pointer")
    value: Any = record.as_dict()
    for raw_token in pointer[1:].split("/"):
        token = _unescape(raw_token)
        if isinstance(value, Mapping) and token in value:
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", token) and int(token) < len(value):
            value = value[int(token)]
        else:
            raise AddressNotFoundError(f"JSON Pointer does not exist: {pointer}", code="address_not_found")
    return value


def has_pointer(record: ECGRecord, pointer: str) -> bool:
    try:
        resolve_pointer(record, pointer)
    except (AddressNotFoundError, AddressSyntaxError):
        return False
    return True


def resolve_address(record: ECGRecord, address: str | RecordAddress) -> AddressResolution:
    parsed = parse_address(address) if isinstance(address, str) else address
    if parsed.record_id != record.record_id or parsed.schema_version != record.schema_version:
        raise RecordIdentityError("address does not match the supplied record", code="record_identity_mismatch")
    value = resolve_pointer(record, parsed.pointer)
    root = parsed.pointer.split("/", 2)[1] if parsed.pointer.startswith("/") else ""
    kind: Literal["record", "measurement", "metadata", "provenance", "debug"]
    if root == "measurements" or root == "delineation":
        kind = "measurement"
    elif root == "provenance":
        kind = "provenance"
    elif root == "metadata":
        kind = "metadata"
    elif root == "debug":
        kind = "debug"
    else:
        kind = "record"
    return AddressResolution(parsed, value, kind)


@dataclass(frozen=True, slots=True)
class MeasurementQuery:
    name: str
    lead: str | None = None
    beat: int | None = None


@dataclass(frozen=True, slots=True)
class NullAbsence:
    kind: Literal["null"] = "null"


@dataclass(frozen=True, slots=True)
class UnmeasurableAbsence:
    reason: str
    kind: Literal["unmeasurable"] = "unmeasurable"


@dataclass(frozen=True, slots=True)
class NotApplicableAbsence:
    reason: str
    kind: Literal["not_applicable"] = "not_applicable"


@dataclass(frozen=True, slots=True)
class MeasurementProvenance:
    method_requested: str
    method_resolved: str
    config_hash: str
    fallback_path: tuple[str, ...]
    field_ref: str | None = None
    field_method: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationStatus:
    tier: Literal["validated", "partially_validated", "unvalidated"]
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    name: str
    address: RecordAddress
    value: int | float | str | bool | None
    unit: str | None
    lead: str | None
    beat: int | None
    absence: NullAbsence | UnmeasurableAbsence | NotApplicableAbsence | None
    provenance: MeasurementProvenance
    validation: ValidationStatus


def _field_location(record: ECGRecord, name: str) -> tuple[str, Mapping[str, Any]]:
    document = record.as_dict()
    for group_path, group in (
        ("/delineation/fiducials", document.get("delineation", {}).get("fiducials", {})),
        ("/measurements/intervals", document.get("measurements", {}).get("intervals", {})),
        ("/measurements/amplitudes", document.get("measurements", {}).get("amplitudes", {})),
        ("/measurements/areas", document.get("measurements", {}).get("areas", {})),
        ("/measurements/flags", document.get("measurements", {}).get("flags", {})),
        ("/measurements/global", document.get("measurements", {}).get("global", {})),
    ):
        if isinstance(group, Mapping) and name in group:
            field = group[name]
            return f"{group_path}/{_escape(name)}", field
    raise MeasurementNotFoundError(f"measurement is not published: {name}", code="measurement_not_found")


def _axis_index(record: ECGRecord, lead: str | None, beat: int | None, axes: Sequence[str]) -> tuple[int | None, int | None]:
    document = record.as_dict()
    leads = list(document.get("axes", {}).get("leads", []))
    beat_items = list(document.get("axes", {}).get("beats", []))
    lead_index = None
    beat_index = None
    if "lead" in axes:
        if lead is None or lead not in leads:
            raise MeasurementSelectorError(f"unknown or missing lead selector: {lead!r}", code="invalid_lead_selector")
        lead_index = leads.index(lead)
    elif lead is not None:
        raise MeasurementSelectorError("lead selector is not valid for a global field", code="invalid_lead_selector")
    if "beat" in axes:
        if type(beat) is not int or beat < 0 or beat >= len(beat_items):
            raise MeasurementSelectorError(f"invalid beat selector: {beat!r}", code="invalid_beat_selector")
        beat_index = beat
    elif beat is not None:
        raise MeasurementSelectorError("beat selector is not valid for a global field", code="invalid_beat_selector")
    return lead_index, beat_index


def _sparse_absence(field: Mapping[str, Any], coordinate: str) -> UnmeasurableAbsence | NotApplicableAbsence | None:
    absence = field.get("absence")
    if not isinstance(absence, Mapping):
        return None
    if absence.get("kind") in {"unmeasurable", "not_applicable"}:
        kind = str(absence["kind"])
        reason = str(absence.get("reason", "unspecified"))
        return UnmeasurableAbsence(reason) if kind == "unmeasurable" else NotApplicableAbsence(reason)
    for kind, cls in (("unmeasurable", UnmeasurableAbsence), ("not_applicable", NotApplicableAbsence)):
        states = absence.get(kind)
        if isinstance(states, Mapping) and coordinate in states:
            return cls(str(states[coordinate]))
    default = absence.get("default")
    if isinstance(default, Mapping) and default.get("kind") in {"unmeasurable", "not_applicable"}:
        cls = UnmeasurableAbsence if default["kind"] == "unmeasurable" else NotApplicableAbsence
        return cls(str(default.get("reason", "unspecified")))
    return None


def query_measurement(record: ECGRecord, name: str, *, lead: str | None = None, beat: int | None = None) -> MeasurementResult:
    pointer, field = _field_location(record, name)
    axes = tuple(str(axis) for axis in field.get("axes", []))
    lead_index, beat_index = _axis_index(record, lead, beat, axes)
    values = field.get("values")
    if "sidecar" in field:
        from .sidecar import SidecarError

        if record.sidecar_source is None:
            raise SidecarError(f"{pointer} is stored in the NPZ sidecar; load the record from its file or pass sidecar=",
                               code="sidecar_missing")
        values = record.sidecar_source.values(record.as_dict())[pointer]
    value: Any = values
    if axes:
        if not isinstance(values, list):
            raise MeasurementSelectorError(f"field {name!r} has invalid values for axes {axes!r}", code="invalid_field_axes")
        if "beat" in axes and "lead" in axes:
            value = values[beat_index][lead_index]  # type: ignore[index]
            coordinate = f"{record.as_dict()['axes']['beats'][beat_index].get('id', f'b{beat_index + 1:04d}')}|{lead}"
            absence = _sparse_absence(field, coordinate)
        elif "beat" in axes:
            value = values[beat_index]  # type: ignore[index]
            coordinate = str(record.as_dict()["axes"]["beats"][beat_index].get("id", f"b{beat_index + 1:04d}"))
            absence = _sparse_absence(field, coordinate)
        else:
            value = values[lead_index]  # type: ignore[index]
            coordinate = str(lead)
            absence = _sparse_absence(field, coordinate)
    else:
        absence = _sparse_absence(field, "")
    if value is not None:
        absence = None  # a field default applies to null cells only
    elif absence is None:
        absence = NullAbsence()
    provenance = record.as_dict().get("provenance", {})
    config = provenance.get("config", {}) if isinstance(provenance, Mapping) else {}
    if not isinstance(config, Mapping):
        config = {}
    method_requested = str(provenance.get("method_requested", "default")) if isinstance(provenance, Mapping) else "default"
    method_resolved = str(provenance.get("method_resolved", provenance.get("algorithm_paths", {}).get("delineation", method_requested))) if isinstance(provenance, Mapping) else method_requested
    fallback = provenance.get("fallback_path", []) if isinstance(provenance, Mapping) else []
    if not isinstance(fallback, list):
        fallback = []
    evidence = field.get("validation", {}).get("evidence", []) if isinstance(field.get("validation"), Mapping) else []
    status = str(field.get("validation", {}).get("status", "unvalidated")) if isinstance(field.get("validation"), Mapping) else "unvalidated"
    tier: Literal["validated", "partially_validated", "unvalidated"] = {
        "benchmark_validated": "validated", "indirectly_validated": "partially_validated",
        "validated": "validated", "partially_validated": "partially_validated",
    }.get(status, "unvalidated")  # type: ignore[assignment]
    evidence_text = ",".join(str(item) for item in evidence) if evidence else None
    field_ref = field.get("provenance_ref")
    field_provenance = resolve_pointer(record, field_ref) if field_ref else {}
    field_method = field_provenance.get("method") if isinstance(field_provenance, Mapping) else None
    return MeasurementResult(
        name=name,
        address=make_record_address(record, f"{pointer}/values" + (f"/{beat_index}/{lead_index}" if "beat" in axes and "lead" in axes else f"/{beat_index}" if "beat" in axes else f"/{lead_index}" if "lead" in axes else "")),
        value=value,
        unit=field.get("unit"),
        lead=lead,
        beat=beat,
        absence=absence,
        provenance=MeasurementProvenance(method_requested, method_resolved, str(config.get("sha256", provenance.get("config_hash", ""))), tuple(str(item) for item in fallback), field_ref, field_method),
        validation=ValidationStatus(tier, evidence_text),
    )


def query_many(record: ECGRecord, queries: Sequence[MeasurementQuery]) -> tuple[MeasurementResult, ...]:
    return tuple(query_measurement(record, item.name, lead=item.lead, beat=item.beat) for item in queries)


@dataclass(frozen=True, slots=True)
class MeasurementSelection:
    record_id: str
    schema_version: str
    results: tuple[MeasurementResult, ...]


def select_measurements(record: ECGRecord, queries: Sequence[MeasurementQuery]) -> MeasurementSelection:
    return MeasurementSelection(record.record_id, record.schema_version, query_many(record, queries))


@dataclass(frozen=True, slots=True)
class LeadView:
    name: str
    index: int
    record: ECGRecord


@dataclass(frozen=True, slots=True)
class BeatView:
    index: int
    sample: int | None
    time_s: float | None
    lead: str | None
    record: ECGRecord


def iter_leads(record: ECGRecord) -> Iterator[LeadView]:
    for index, name in enumerate(record.as_dict().get("axes", {}).get("leads", [])):
        yield LeadView(str(name), index, record)


def iter_beats(record: ECGRecord, *, lead: str | None = None) -> Iterator[BeatView]:
    document = record.as_dict()
    leads = list(document.get("axes", {}).get("leads", []))
    if lead is not None and lead not in leads:
        raise MeasurementSelectorError(f"unknown lead selector: {lead!r}", code="invalid_lead_selector")
    fs = document.get("acquisition", {}).get("sample_rate_hz")
    for index, beat in enumerate(document.get("axes", {}).get("beats", [])):
        sample = beat.get("r_sample") if isinstance(beat, Mapping) else None
        time_s = float(sample) / float(fs) if sample is not None and fs else None
        yield BeatView(index, int(sample) if isinstance(sample, int) else None, time_s, lead, record)


def dump_measurements(selection: MeasurementSelection, destination: str | Path | Any, *, format: Literal["json", "jsonl"] = "json") -> None:
    if format not in {"json", "jsonl"}:
        raise ValueError("format must be 'json' or 'jsonl'")
    objects = [
        {
            "name": result.name,
            "address": str(result.address),
            "value": result.value,
            "unit": result.unit,
            "lead": result.lead,
            "beat": result.beat,
            "absence": None if result.absence is None else result.absence.__dict__ if hasattr(result.absence, "__dict__") else {"kind": result.absence.kind, **({"reason": result.absence.reason} if hasattr(result.absence, "reason") else {})},
            "provenance": {"method_requested": result.provenance.method_requested, "method_resolved": result.provenance.method_resolved, "config_hash": result.provenance.config_hash, "fallback_path": list(result.provenance.fallback_path), "field_ref": result.provenance.field_ref, "field_method": result.provenance.field_method},
            "validation": {"tier": result.validation.tier, "evidence": result.validation.evidence},
        }
        for result in selection.results
    ]
    payload = objects if format == "jsonl" else {"record_id": selection.record_id, "schema_version": selection.schema_version, "results": objects}
    text = "\n".join(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for obj in objects) + ("\n" if format == "jsonl" else "") if format == "jsonl" else json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    if hasattr(destination, "write"):
        destination.write(text)
    else:
        Path(destination).write_text(text, encoding="utf-8")


__all__ = [
    "RecordAddress", "AddressResolution", "parse_address", "parse_record_address",
    "make_record_address", "resolve_pointer", "resolve_address", "has_pointer",
    "MeasurementQuery", "NullAbsence", "UnmeasurableAbsence", "NotApplicableAbsence",
    "MeasurementProvenance", "ValidationStatus", "MeasurementResult", "query_measurement",
    "query_many", "MeasurementSelection", "select_measurements", "LeadView", "BeatView",
    "iter_leads", "iter_beats", "dump_measurements",
]
