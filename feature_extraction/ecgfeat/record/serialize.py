"""Canonical JSON and optional dense-array serialization for ECG Records."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from ..errors import RecordValidationError
from .model import ECGRecord, record_from_document

RecordProfile = Literal["summary", "all", "measurement", "debug"]

_SUMMARY_FIELDS = {
    "delineation": {"p_onset", "p_offset", "qrs_onset", "r_peak", "qrs_offset", "j_point", "t_offset"},
    "intervals": {"p_duration_ms", "pr_interval_ms", "qrs_duration_ms", "qt_interval_ms", "tpeak_tend_ms"},
    "amplitudes": {"p_amplitude_uv", "r_amplitude_uv", "s_amplitude_uv", "st_80ms_uv", "t_amplitude_uv"},
    "areas": {"qrs_signed_area_uv_ms"},
}


@dataclass(frozen=True, slots=True)
class SerializedRecord:
    json_bytes: bytes
    sidecars: Mapping[str, bytes]


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _finite(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecordValidationError("ECG Record cannot contain NaN or infinity", code="nonfinite_record_value")
    elif isinstance(value, Mapping):
        for child in value.values():
            _finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite(child)


def _profile_document(record: ECGRecord, profile: RecordProfile | None) -> dict[str, Any]:
    # Reading and writing a profile is not a request to reapply an older
    # writer's publication list; doing so would drop future optional fields.
    if profile is None or profile == record.profile:
        return _thaw(record.as_dict())
    profile = record.profile if profile is None else profile
    if profile not in {"summary", "all", "measurement", "debug"}:
        raise ValueError("profile must be summary, all, measurement, or debug")
    ranks = {"summary": 0, "all": 1, "measurement": 1, "debug": 2}
    if ranks[profile] > ranks[record.profile]:
        raise RecordValidationError("cannot promote a record profile without re-emitting its measurements", code="profile_data_unavailable")
    document = _thaw(record.as_dict())
    document["profile"] = profile
    if profile == "summary":
        delineation = document.get("delineation", {}).get("fiducials", {})
        if isinstance(delineation, dict):
            document["delineation"]["fiducials"] = {
                name: value for name, value in delineation.items() if name in _SUMMARY_FIELDS["delineation"]
            }
        measurements = document.get("measurements", {})
        if isinstance(measurements, dict):
            compact_fields = {
                key: value for key, value in measurements.items()
                if key not in {"intervals", "amplitudes", "areas", "global", "flags"}
            }
            for group in ("intervals", "amplitudes", "areas"):
                values = measurements.get(group)
                if isinstance(values, dict):
                    measurements[group] = {
                        name: value for name, value in values.items() if name in _SUMMARY_FIELDS[group]
                    }
            global_values = measurements.get("global")
            if isinstance(global_values, dict):
                measurements["global"] = {
                    name: value for name, value in global_values.items()
                    if name in {"heart_rate_bpm", "frontal_qrs_axis_deg"}
                }
            # Flags and debug extensions have no place in the compact core.
            measurements["flags"] = {}
            # ``build_record`` from the low-level model guide uses a compact
            # direct field map.  Preserve it; pipeline records use the five
            # schema groups above and therefore have no compact fields.
            if compact_fields:
                measurements.update(compact_fields)
        document.pop("debug", None)
    elif profile in {"all", "measurement"}:
        document.pop("debug", None)
    return document


def to_json_obj(record: ECGRecord, *, profile: RecordProfile | None = None) -> dict[str, Any]:
    validate_record(record)
    result = _profile_document(record, profile)
    _finite(result)
    return result


def from_json_obj(value: Mapping[str, Any]) -> ECGRecord:
    """Decode a JSON-compatible mapping after structural validation."""
    return validate_record(record_from_document(value))


def record_to_dict(record: ECGRecord) -> dict[str, Any]:
    """Return a detached mapping without re-projecting the record profile."""
    result = _thaw(record.as_dict())
    _finite(result)
    return result


def validate_record(record: ECGRecord | Mapping[str, Any], *, level: Literal["strict", "schema"] = "strict") -> ECGRecord:
    if level not in {"strict", "schema"}:
        raise ValueError("validation level must be 'strict' or 'schema'")
    value = record if isinstance(record, ECGRecord) else record_from_document(record)
    from .contract import validate_document
    validate_document(value, strict=level == "strict")
    return value


def load_record(
    source: str | Any | Mapping[str, Any],
    *,
    validate: Literal["strict", "schema", "none"] = "strict",
    sidecar: str | Any | bytes | None = None,
) -> ECGRecord:
    """Load a record; a dense NPZ sidecar is resolved next to a record file.

    ``sidecar`` may be a path or the sidecar bytes.  Strict loading verifies a
    sidecar-backed record's NPZ immediately (digest, identity, shape, dtype and
    state mask); ``schema`` and ``none`` verify lazily on first access.
    """
    from pathlib import Path

    if validate not in {"strict", "schema", "none"}:
        raise ValueError("validation level must be strict, schema, or none")
    record_path = None
    if isinstance(source, Mapping):
        value = record_from_document(source)
    elif hasattr(source, "read"):
        raw = source.read()
        value = loads_record(raw, validate="none")
    else:
        record_path = Path(source)
        value = loads_record(record_path.read_bytes(), validate="none")
    value = _attach_sidecar(value, record_path, sidecar)
    if validate == "none":
        return value
    value = validate_record(value, level=validate)
    if validate == "strict" and value.sidecar_source is not None:
        value.sidecar_source.values(value.as_dict())
    elif validate == "strict" and (value.artifacts.get("dense_measurements") is not None):
        from .sidecar import SidecarError

        raise SidecarError("strict loading of a sidecar-backed record requires its sidecar file or bytes",
                           code="sidecar_missing")
    return value


def _attach_sidecar(record: ECGRecord, record_path: Any, sidecar: Any) -> ECGRecord:
    from dataclasses import replace
    from pathlib import Path

    from .sidecar import SidecarSource, resolve_uri

    descriptor = record.artifacts.get("dense_measurements")
    if descriptor is None:
        return record
    if isinstance(sidecar, (bytes, bytearray)):
        source = SidecarSource(data=bytes(sidecar))
    elif sidecar is not None:
        source = SidecarSource(path=Path(sidecar))
    elif record_path is not None:
        source = SidecarSource(path=resolve_uri(record_path, str(descriptor.get("uri", ""))))
    else:
        return record
    return replace(record, sidecar_source=source)


def materialize_record(record: ECGRecord) -> ECGRecord:
    """Return an inline record with every sidecar-backed matrix restored."""
    from .sidecar import SidecarError, materialize

    if record.artifacts.get("dense_measurements") is None:
        return record
    if record.sidecar_source is None:
        raise SidecarError("record has a sidecar but no sidecar source was attached", code="sidecar_missing")
    document = record.as_dict()
    record.sidecar_source.values(document)
    return validate_record(record_from_document(materialize(document, record.sidecar_source._data)))


def dumps_record(record: ECGRecord, *, profile: RecordProfile | None = None, indent: int | None = None) -> bytes:
    obj = to_json_obj(record, profile=profile)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise RecordValidationError(f"duplicate JSON member: {key}", code="duplicate_json_member")
        value[key] = child
    return value


def loads_record(data: bytes | str, *, validate: Literal["strict", "schema", "none"] = "strict") -> ECGRecord:
    try:
        value = json.loads(data.decode("utf-8") if isinstance(data, bytes) else data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordValidationError("invalid ECG Record JSON", code="invalid_record_json") from exc
    record = record_from_document(value)
    if validate == "none":
        return record
    return validate_record(record, level=validate)


def encode_npz_sidecar(arrays: Mapping[str, np.ndarray], *, compressed: bool = True) -> bytes:
    import numpy as np

    buffer = io.BytesIO()
    normalized = {str(key): np.ascontiguousarray(value) for key, value in arrays.items()}
    if any(array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)) for array in normalized.values()):
        raise RecordValidationError("NPZ sidecars require finite numeric arrays without objects")
    writer = np.savez_compressed if compressed else np.savez
    writer(buffer, **normalized)
    return buffer.getvalue()


def serialize_record(
    record: ECGRecord,
    *,
    profile: RecordProfile | None = None,
    sidecar_uri: str | None = None,
) -> SerializedRecord:
    """Canonical JSON, optionally moving dense beat/lead matrices to an NPZ sidecar.

    ``sidecar_uri`` is a plain file name; the sidecar must be stored in the
    same directory as the JSON document.  Sidecar bytes are deterministic.
    """
    if sidecar_uri is None:
        return SerializedRecord(json_bytes=dumps_record(record, profile=profile), sidecars={})
    from .sidecar import split_dense

    document, data = split_dense(to_json_obj(record, profile=profile), sidecar_uri)
    validate_record(record_from_document(document))
    json_bytes = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                            allow_nan=False).encode("utf-8")
    return SerializedRecord(json_bytes=json_bytes, sidecars={sidecar_uri: data})


__all__ = [
    "RecordProfile", "SerializedRecord", "to_json_obj", "from_json_obj", "record_to_dict",
    "validate_record", "load_record", "dumps_record", "loads_record", "serialize_record",
    "encode_npz_sidecar", "materialize_record",
]
