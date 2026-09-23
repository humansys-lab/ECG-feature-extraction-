"""Dependency-free wire validation, including cross-field axis invariants.

The packaged JSON Schema describes structural constraints. This validator also
checks constraints JSON Schema cannot express (shapes, pointers, coordinates).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..errors import RecordValidationError, AddressNotFoundError, AddressSyntaxError
from .registry import check_ceiling
from .sidecar import validate_descriptor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordValidationError(message, code="invalid_record_contract")


def _check_fiducial_order(fiducials: Mapping[str, Any]) -> None:
    """Published landmarks of one wave must be ordered: onset <= peak <= offset."""

    for chain in (("p_onset", "p_offset"), ("qrs_onset", "r_peak", "qrs_offset")):
        present = [fiducials[name]["values"] for name in chain
                   if name in fiducials and fiducials[name]["axes"] == ["beat", "lead"]
                   and fiducials[name]["values"] is not None]
        if len(present) != len(chain):
            continue
        for i, row in enumerate(present[0]):
            for j in range(len(row)):
                values = [matrix[i][j] for matrix in present]
                ordered = [value for value in values if value is not None]
                _require(ordered == sorted(ordered), f"{'/'.join(chain)}: fiducial order violated at beat {i}, lead {j}")


def validate_document(record: Any, *, strict: bool = True) -> None:
    from .query import resolve_pointer

    doc = record.as_dict()
    _require(record.schema_version.split(".")[0] == "1", "unsupported schema major")
    acq, axes = doc["acquisition"], doc["axes"]
    _require({"sample_rate_hz", "sample_count", "input_mode", "leads", "amplitude_unit"} <= acq.keys(), "incomplete acquisition")
    _require(type(acq["sample_rate_hz"]) in (int, float) and acq["sample_rate_hz"] > 0, "invalid sample_rate_hz")
    _require(type(acq["sample_count"]) is int and acq["sample_count"] > 0, "invalid sample_count")
    _require(acq["amplitude_unit"] in ("mV", "uV"), "invalid acquisition amplitude unit")
    _require(acq["input_mode"] in ("standard_12", "limited"), "invalid acquisition input mode")
    leads, beats = axes.get("leads"), axes.get("beats")
    _require(isinstance(leads, list) and bool(leads), "axes.leads must be nonempty")
    _require(all(isinstance(x, str) and x and "|" not in x for x in leads), "invalid lead name")
    _require(len(leads) == len(set(leads)), "duplicate lead axis")
    _require(isinstance(beats, list), "axes.beats must be an array")
    beat_ids = []
    for beat in beats:
        _require(isinstance(beat, dict) and isinstance(beat.get("id"), str) and bool(beat["id"]) and "|" not in beat["id"], "invalid beat identity")
        beat_ids.append(beat["id"])
        sample = beat.get("r_sample")
        _require(sample is None or type(sample) is int and 0 <= sample < acq["sample_count"], "beat sample outside acquisition")
    _require(len(beat_ids) == len(set(beat_ids)), "duplicate beat identity")
    if strict:
        _require(acq["leads"] == leads, "acquisition and lead axis disagree")
        if acq["input_mode"] == "standard_12":
            canonical = {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}
            _require(set(leads) == canonical, "standard_12 requires all canonical leads")
        else:
            _require(1 <= len(leads) <= 8, "limited requires 1–8 leads")
        _require(doc["provenance"].get("schema_version", record.schema_version) == record.schema_version, "provenance schema version disagrees")
    groups = [("fiducials", doc["delineation"].get("fiducials", {}))]
    for group, values in doc["measurements"].items():
        _require(group in {"intervals", "amplitudes", "areas", "flags", "global"}, "unknown measurement group; use extensions")
        groups.append((group, values))
    for group, fields in groups:
        _require(isinstance(fields, dict), f"{group} must be an object")
        for name, field in fields.items():
            _require(bool(name), "measurement names must be nonempty")
            _require(isinstance(field, dict), f"{name}: field must be an object")
            _require({"type", "unit", "axes", "values", "nullable", "validation", "provenance_ref"} <= field.keys(), f"{name}: incomplete field wrapper")
            kind, unit, field_axes = field["type"], field["unit"], field["axes"]
            _require(kind in ("integer", "number", "boolean", "string"), f"{name}: invalid type")
            _require(unit is None or isinstance(unit, str), f"{name}: invalid unit")
            expected = {"fiducials": "sample", "intervals": "ms", "amplitudes": "uV", "areas": "uV_ms"}.get(group)
            if expected:
                _require(unit == expected and kind == "integer", f"{name}: expected integer {expected}")
            global_units = {"heart_rate_bpm": "1/min", "frontal_qrs_axis_deg": "deg", "qtc_bazett_ms": "ms", "qrs_wide_ms": "ms"}
            if group == "global" and name in global_units:
                _require(unit == global_units[name] and kind == "integer", f"{name}: incorrect global unit/type")
            _require(field_axes in ([], ["beat"], ["lead"], ["beat", "lead"]), f"{name}: invalid axes")
            if group == "global":
                _require(field_axes == [], f"{name}: global fields cannot have beat/lead axes")
            if group == "flags":
                _require(kind == "boolean" and unit is None, f"{name}: flags require boolean type with no unit")
            _require(type(field["nullable"]) is bool, f"{name}: nullable must be boolean")
            validation = field["validation"]
            _require(isinstance(validation, dict) and validation.get("status") in ("benchmark_validated", "indirectly_validated", "unvalidated", "validated", "partially_validated"), f"{name}: invalid validation status")
            evidence = validation.get("evidence", [])
            _require(isinstance(evidence, list), f"{name}: evidence must be an array")
            if validation["status"] != "unvalidated":
                _require(bool(evidence), f"{name}: validation claim requires evidence")
            if strict:
                root = "/delineation/fiducials" if group == "fiducials" else f"/measurements/{group}"
                check_ceiling(f"{root}/{name}", validation["status"], record.schema_version)
            ref = field["provenance_ref"]
            _require(isinstance(ref, str) and ref.startswith("/provenance/"), f"{name}: invalid provenance pointer")
            if strict:
                try:
                    resolve_pointer(record, ref)
                except (AddressNotFoundError, AddressSyntaxError) as exc:
                    raise RecordValidationError(f"{name}: unresolved provenance pointer") from exc
            values = field["values"]
            sidecar_backed = "sidecar" in field
            if sidecar_backed:
                _require(field_axes == ["beat", "lead"] and values is None, f"{name}: only null dense values may be sidecar-backed")
                values = [[None] * len(leads) for _ in beats]
            if field_axes == ["beat", "lead"]:
                _require(isinstance(values, list) and len(values) == len(beats), f"{name}: beat dimension mismatch")
                _require(all(isinstance(row, list) and len(row) == len(leads) for row in values), f"{name}: lead dimension mismatch")
                cells = {f"{b}|{lead}": values[i][j] for i, b in enumerate(beat_ids) for j, lead in enumerate(leads)}
            elif field_axes:
                labels = beat_ids if field_axes == ["beat"] else leads
                _require(isinstance(values, list) and len(values) == len(labels), f"{name}: dimension mismatch")
                cells = dict(zip(labels, values))
            else:
                cells = {"": values}
            absence = field.get("absence", {})
            _require(isinstance(absence, dict), f"{name}: invalid absence")
            if strict and acq["input_mode"] == "limited" and group == "global" and name in {"frontal_qrs_axis_deg", "qtc_bazett_ms"}:
                _require(absence.get("kind") == "not_applicable" and all(value is None for value in cells.values()), f"{name}: formal report is not applicable to limited input")
            reasons = set()
            if "kind" in absence:
                _require(absence["kind"] in ("unmeasurable", "not_applicable") and isinstance(absence.get("reason"), str) and bool(absence["reason"]), f"{name}: invalid field absence")
                _require(all(value is None for value in cells.values()), f"{name}: absence contradicts measured values")
                reasons.update(cells)
            else:
                _require(set(absence) <= {"unmeasurable", "not_applicable", "default"}, f"{name}: unknown absence kind")
                default = absence.get("default")
                if default is not None:
                    _require(isinstance(default, dict) and set(default) == {"kind", "reason"}
                             and default["kind"] in ("unmeasurable", "not_applicable")
                             and isinstance(default["reason"], str) and bool(default["reason"]),
                             f"{name}: invalid default absence")
                    _require(field_axes != [], f"{name}: default absence needs axes")
                    reasons.update(coordinate for coordinate, value in cells.items() if value is None)
                for state_kind, mapping in absence.items():
                    if state_kind == "default":
                        continue
                    _require(isinstance(mapping, dict), f"{name}: absence map must be an object")
                    for coordinate, reason in mapping.items():
                        # Sidecar cells are checked against the state mask when the sidecar is read.
                        _require(coordinate in cells and cells[coordinate] is None, f"{name}: absence coordinate has no null cell")
                        _require((default is not None or coordinate not in reasons) and isinstance(reason, str) and bool(reason),
                                 f"{name}: duplicate absence or invalid reason")
                        _require(not (state_kind == "unmeasurable" and coordinate in (absence.get("not_applicable") or {})),
                                 f"{name}: cell has two absence states")
                        reasons.add(coordinate)
            for coordinate, value in cells.items():
                if value is None:
                    _require(sidecar_backed or field["nullable"] or coordinate in reasons, f"{name}: plain null forbidden")
                    continue
                types = {"integer": (int,), "number": (int, float), "boolean": (bool,), "string": (str,)}
                _require(type(value) in types[kind], f"{name}: cell type mismatch")
                if strict and unit == "sample":
                    _require(0 <= value < acq["sample_count"], f"{name}: sample outside acquisition")
                if strict and group == "intervals":
                    _require(value >= 0, f"{name}: negative interval")
    validate_descriptor(doc)
    if strict:
        _check_fiducial_order(doc["delineation"].get("fiducials", {}))
    for artifact in doc["artifacts"].values():
        if artifact is None:
            continue
        _require(isinstance(artifact, dict), "artifact must be an object or null")
        _require(isinstance(artifact.get("uri"), str) and isinstance(artifact.get("media_type"), str), "artifact requires uri and media_type")
        _require(isinstance(artifact.get("sha256"), str) and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["sha256"])), "artifact requires SHA-256 digest")
