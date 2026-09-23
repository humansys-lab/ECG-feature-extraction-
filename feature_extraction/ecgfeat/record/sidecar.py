"""Optional NPZ sidecar for dense ``["beat", "lead"]`` measurement matrices.

Protocol (ECG Record schema 1.0):

* ``/artifacts/dense_measurements`` describes the sidecar: a *relative*
  same-directory ``uri`` (no scheme, no path separators), ``media_type``
  ``application/x-npz``, the SHA-256 of the exact sidecar bytes, the owning
  ``record_id`` and ``schema_version``, the matrix ``shape`` and the list of
  sidecar-backed field pointers.
* A sidecar-backed field keeps its JSON entry (type, unit, axes, validation,
  provenance and absence reasons) with ``values: null`` and a ``sidecar``
  member naming its ``array_key`` (``int32`` values) and ``state_key``
  (``uint8``: 0 measured, 1 plain null, 2 unmeasurable, 3 not applicable).
* JSON stays authoritative for identity, units, axes and absence reasons; the
  state mask must agree with the JSON absence maps cell by cell.
* The NPZ also carries ``__identity__`` (UTF-8 JSON of record id and schema
  version) so a sidecar cannot be attached to another record.
* Bytes are deterministic: zip entries use a fixed timestamp and order.

Consumers cite a sidecar value by the field pointer plus beat/lead axis
identities, never by ZIP member or offset.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

from ..errors import RecordValidationError

MEDIA_TYPE = "application/x-npz"
STATE_CODES = {"measured": 0, "null": 1, "unmeasurable": 2, "not_applicable": 3}
DENSE_GROUPS = (("delineation", "fiducials"), ("measurements", "intervals"),
                ("measurements", "amplitudes"), ("measurements", "areas"))
_URI = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.npz")
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


class SidecarError(RecordValidationError):
    """A sidecar is missing, does not match its descriptor, or is inconsistent."""


def _require(condition: bool, message: str, code: str = "invalid_sidecar") -> None:
    if not condition:
        raise SidecarError(message, code=code)


def array_key(pointer: str) -> str:
    return pointer.strip("/").replace("/", ".")


def dense_fields(document: Mapping[str, Any]):
    """Yield ``(pointer, field)`` for every dense beat/lead field in the document."""

    for top, group in DENSE_GROUPS:
        fields = (document.get(top) or {}).get(group) or {}
        for name, field in fields.items():
            if isinstance(field, Mapping) and list(field.get("axes", [])) == ["beat", "lead"]:
                yield f"/{top}/{group}/{name}", field


def _coordinates(document: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    return [beat["id"] for beat in document["axes"]["beats"]], list(document["axes"]["leads"])


def _states(field: Mapping[str, Any], beats: list[str], leads: list[str], values: list[list[Any]]) -> list[list[int]]:
    absence = field.get("absence") or {}
    whole = absence.get("kind") if "kind" in absence else None
    default = (absence.get("default") or {}).get("kind")
    states = []
    for i, beat in enumerate(beats):
        row = []
        for j, lead in enumerate(leads):
            coordinate = f"{beat}|{lead}"
            if values[i][j] is not None:
                row.append(STATE_CODES["measured"])
            elif whole:
                row.append(STATE_CODES[whole])
            elif coordinate in (absence.get("unmeasurable") or {}):
                row.append(STATE_CODES["unmeasurable"])
            elif coordinate in (absence.get("not_applicable") or {}):
                row.append(STATE_CODES["not_applicable"])
            elif default:
                row.append(STATE_CODES[default])
            else:
                row.append(STATE_CODES["null"])
        states.append(row)
    return states


def _write_npz(arrays: Mapping[str, Any]) -> bytes:
    import numpy as np

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def split_dense(document: dict[str, Any], uri: str) -> tuple[dict[str, Any], bytes]:
    """Move every dense beat/lead matrix of ``document`` into an NPZ sidecar."""

    import numpy as np

    _require(bool(_URI.fullmatch(uri)), "sidecar uri must be a plain relative *.npz file name", "invalid_sidecar_uri")
    beats, leads = _coordinates(document)
    arrays: dict[str, Any] = {}
    pointers = []
    for pointer, field in dense_fields(document):
        _require(field.get("type") == "integer" and "sidecar" not in field, f"{pointer}: only inline integer matrices can move to a sidecar")
        values = field["values"]
        key = array_key(pointer)
        arrays[key] = np.array([[0 if value is None else value for value in row] for row in values],
                               dtype=np.int32).reshape(len(beats), len(leads))
        arrays[key + ".state"] = np.array(_states(field, beats, leads, values), dtype=np.uint8).reshape(len(beats), len(leads))
        field["values"] = None
        field["sidecar"] = {"array_key": key, "state_key": key + ".state", "dtype": "int32"}
        pointers.append(pointer)
    identity = {"record_id": document["record_id"], "schema_version": document["schema_version"]}
    arrays["__identity__"] = np.frombuffer(json.dumps(identity, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    data = _write_npz(arrays)
    document["artifacts"]["dense_measurements"] = {
        "uri": uri,
        "media_type": MEDIA_TYPE,
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "record_id": document["record_id"],
        "schema_version": document["schema_version"],
        "axes": ["beat", "lead"],
        "shape": [len(beats), len(leads)],
        "state_codes": dict(STATE_CODES),
        "fields": pointers,
    }
    return document, data


def validate_descriptor(document: Mapping[str, Any]) -> None:
    """Structural checks that need no sidecar bytes (run by strict validation)."""

    descriptor = (document.get("artifacts") or {}).get("dense_measurements")
    backed = {pointer for pointer, field in dense_fields(document) if "sidecar" in field}
    if descriptor is None:
        _require(not backed, "sidecar-backed fields require /artifacts/dense_measurements")
        return
    _require(descriptor.get("media_type") == MEDIA_TYPE, "dense_measurements must be application/x-npz")
    _require(bool(_URI.fullmatch(str(descriptor.get("uri", "")))), "sidecar uri must be a plain relative *.npz file name", "invalid_sidecar_uri")
    _require(descriptor.get("record_id") == document.get("record_id"), "sidecar descriptor names another record", "sidecar_identity")
    _require(descriptor.get("schema_version") == document.get("schema_version"), "sidecar schema version disagrees", "sidecar_identity")
    beats, leads = _coordinates(document)
    _require(descriptor.get("shape") == [len(beats), len(leads)] and descriptor.get("axes") == ["beat", "lead"], "sidecar shape disagrees with record axes")
    _require(set(descriptor.get("fields", [])) == backed, "sidecar field list disagrees with sidecar-backed fields")
    for pointer, field in dense_fields(document):
        if "sidecar" in field:
            ref = field["sidecar"]
            _require(field.get("values") is None, f"{pointer}: sidecar-backed values must be null in JSON")
            _require(ref == {"array_key": array_key(pointer), "state_key": array_key(pointer) + ".state", "dtype": "int32"},
                     f"{pointer}: invalid sidecar reference")


def read_sidecar(document: Mapping[str, Any], data: bytes) -> dict[str, list[list[Any]]]:
    """Verify ``data`` against the descriptor and return inline value matrices."""

    import numpy as np

    validate_descriptor(document)
    descriptor = document["artifacts"]["dense_measurements"]
    _require(descriptor is not None, "record has no sidecar")
    _require("sha256:" + hashlib.sha256(data).hexdigest() == descriptor["sha256"], "sidecar bytes do not match the record digest", "sidecar_digest")
    try:
        archive = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:  # a corrupt container is a sidecar error, not a crash
        raise SidecarError("sidecar is not a readable NPZ file", code="invalid_sidecar") from exc
    with archive:
        keys = set(archive.files)
        identity = json.loads(bytes(archive["__identity__"]).decode("utf-8")) if "__identity__" in keys else None
        _require(identity == {"record_id": document["record_id"], "schema_version": document["schema_version"]},
                 "sidecar identity does not match the record", "sidecar_identity")
        beats, leads = _coordinates(document)
        expected_keys = {"__identity__"}
        inline: dict[str, list[list[Any]]] = {}
        for pointer, field in dense_fields(document):
            if "sidecar" not in field:
                continue
            ref = field["sidecar"]
            expected_keys |= {ref["array_key"], ref["state_key"]}
            _require(ref["array_key"] in keys and ref["state_key"] in keys, f"{pointer}: array missing from sidecar")
            values, states = archive[ref["array_key"]], archive[ref["state_key"]]
            _require(values.dtype == np.int32 and states.dtype == np.uint8, f"{pointer}: unexpected sidecar dtype")
            _require(values.shape == (len(beats), len(leads)) == states.shape, f"{pointer}: sidecar shape mismatch")
            restored = [[int(values[i, j]) if states[i, j] == STATE_CODES["measured"] else None
                         for j in range(len(leads))] for i in range(len(beats))]
            _require(_states(field, beats, leads, restored) == states.tolist(),
                     f"{pointer}: sidecar state mask disagrees with JSON absence", "sidecar_state")
            inline[pointer] = restored
        _require(keys == expected_keys, "sidecar contains undeclared arrays")
    return inline


def materialize(document: Mapping[str, Any], data: bytes) -> dict[str, Any]:
    """Return a detached inline document with sidecar values restored."""

    inline = read_sidecar(document, data)
    restored = json.loads(json.dumps(document))
    for pointer, values in inline.items():
        _, top, group, name = pointer.split("/")
        field = restored[top][group][name]
        field["values"] = values
        del field["sidecar"]
    restored["artifacts"]["dense_measurements"] = None
    return restored


class SidecarSource:
    """Lazily loaded, verified sidecar bytes attached to a loaded record."""

    def __init__(self, *, path: Path | None = None, data: bytes | None = None) -> None:
        self._path, self._data, self._inline = path, data, None

    def values(self, document: Mapping[str, Any]) -> dict[str, list[list[Any]]]:
        if self._inline is None:
            if self._data is None:
                _require(self._path is not None and self._path.is_file(), f"sidecar not found: {self._path}", "sidecar_missing")
                self._data = self._path.read_bytes()
            self._inline = read_sidecar(document, self._data)
        return self._inline


def resolve_uri(record_path: Path, uri: str) -> Path:
    _require(bool(_URI.fullmatch(uri)), "sidecar uri must be a plain relative *.npz file name", "invalid_sidecar_uri")
    return record_path.parent / uri


__all__ = ["MEDIA_TYPE", "STATE_CODES", "SidecarError", "SidecarSource", "split_dense", "read_sidecar",
           "materialize", "validate_descriptor", "resolve_uri", "array_key"]
