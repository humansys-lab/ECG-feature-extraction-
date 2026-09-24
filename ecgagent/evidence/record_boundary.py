"""ECG Record addresses at the ecgagent evidence boundary (document 05, section E).

ecgagent's diagnostic evidence still comes from the legacy measurement payload
(ECG Record schema 1.0 publishes a compact measurement subset).  This module is
the schema-aware entry for record evidence: it parses canonical addresses
``ecg-record:<record_id>@<schema_version>#<RFC 6901 pointer>``, rejects
unsupported schema versions before anything reaches prompts or tools, and
resolves values through ``ecgfeat.record`` with their availability state,
unit, provenance and validation status preserved (never flattened to null).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_RECORD_SCHEMA_MAJORS = frozenset({"1"})
_CELL = re.compile(r"^/(?:delineation/fiducials|measurements/(?:intervals|amplitudes|areas|flags|global))/"
                   r"(?P<name>[^/]+)/values(?:/(?P<beat>\d+)(?:/(?P<lead>\d+))?)?$")


class RecordAddressError(ValueError):
    """An address is malformed, targets another record, or uses an unsupported schema."""


@dataclass(frozen=True)
class RecordEvidence:
    address: str
    record_id: str
    schema_version: str
    pointer: str
    value: Any
    state: str  # measured | null | unmeasurable | not_applicable | structure
    reason: str | None = None
    unit: str | None = None
    lead: str | None = None
    beat: int | None = None
    validation: str | None = None
    method: str | None = None
    config_hash: str | None = None


def check_schema(schema_version: str) -> None:
    if str(schema_version).split(".")[0] not in SUPPORTED_RECORD_SCHEMA_MAJORS:
        raise RecordAddressError(f"unsupported ECG Record schema version {schema_version!r}")


def parse_record_address(address: str):
    from ecgfeat.record import parse_address

    try:
        parsed = parse_address(address)
    except Exception as exc:
        raise RecordAddressError(f"malformed ECG Record address: {address!r}") from exc
    check_schema(parsed.schema_version)
    return parsed


def resolve_record_evidence(record: Any, address: str) -> RecordEvidence:
    """Resolve one canonical address against a loaded ``ecgfeat.record.ECGRecord``."""

    from ecgfeat.record import query_measurement, resolve_address

    parsed = parse_record_address(address)
    check_schema(record.schema_version)
    try:
        resolution = resolve_address(record, address)
    except Exception as exc:
        raise RecordAddressError(str(exc)) from exc
    match = _CELL.match(parsed.pointer)
    if not match:
        return RecordEvidence(address, parsed.record_id, parsed.schema_version, parsed.pointer, resolution.value,
                              state="structure")
    leads = list(record.axes["leads"])
    beat = int(match["beat"]) if match["beat"] is not None else None
    lead = leads[int(match["lead"])] if match["lead"] is not None else None
    result = query_measurement(record, match["name"], lead=lead, beat=beat)
    absence = result.absence
    state = "measured" if absence is None else absence.kind
    return RecordEvidence(
        address=address, record_id=parsed.record_id, schema_version=parsed.schema_version, pointer=parsed.pointer,
        value=result.value, state=state, reason=getattr(absence, "reason", None), unit=result.unit,
        lead=lead, beat=beat, validation=result.validation.tier,
        method=result.provenance.field_method, config_hash=result.provenance.config_hash,
    )


def record_path_for(output_dir: str | Path, record: str) -> Path:
    """Where ``ecgagent batch extract`` writes the ECG Record for ``record``."""
    return Path(output_dir) / "records" / f"{record}.record.json"


def load_record(path: str | Path):
    from ecgfeat.record import load_record as _load

    record = _load(Path(path))
    check_schema(record.schema_version)
    return record


__all__ = ["SUPPORTED_RECORD_SCHEMA_MAJORS", "RecordAddressError", "RecordEvidence", "check_schema",
           "parse_record_address", "resolve_record_evidence", "record_path_for", "load_record"]
