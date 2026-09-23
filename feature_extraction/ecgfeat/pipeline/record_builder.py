"""Assemble the ECG Record document from a completed measurement state.

This is the only place that decides how engine measurements become published
record cells.  Three rules matter for consumers:

* validation status comes from the shipped validation-evidence registry only;
* a value the engine produced but that violates a published invariant is not
  published: inverted P/QRS fiducials or an R peak outside its QRS window
  become ``unmeasurable(fiducial_order_violation)`` and negative derived
  durations ``unmeasurable(negative_interval)``;
* missing values keep a stable reason code rather than collapsing to null.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..errors import ComputationInvariantError
from ..record.provenance import fingerprint_input
from ..record.registry import field_validation

SCHEMA_VERSION = "1.0.0"
LIBRARY_VERSION = "0.1.0"

# Stable absence reason codes emitted by this builder (open enum, schema 1.x).
REASON_UNAVAILABLE = "measurement_unavailable"
REASON_OUTSIDE = "sample_outside_acquisition"
REASON_ORDER = "fiducial_order_violation"
REASON_NEGATIVE = "negative_interval"
REASON_LIMITED = "input_mode_limited"


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _round_even(value: float | None) -> int | None:
    number = _as_float(value)
    return None if number is None else int(round(number))


def _uv(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else _round_even(number * 1000)


def _field_entry(pointer: str, values: Any, *, type_name: str, unit: str | None, axes: list[str],
                 provenance_ref: str) -> dict[str, Any]:
    return {
        "type": type_name,
        "unit": unit,
        "axes": axes,
        "values": values,
        "nullable": True,
        "validation": field_validation(pointer),
        "provenance_ref": provenance_ref,
    }


def _matrix(beat_features: Sequence[Any], beats: Sequence[Any], leads: Sequence[str], getter: Callable[[Any], Any],
            *, transform: Callable[[Any], Any] = lambda value: value) -> list[list[Any]]:
    by_cell = {(int(getattr(item, "beat_id", -1)), str(getattr(item, "lead", ""))): item for item in beat_features}
    output: list[list[Any]] = []
    for beat in beats:
        beat_id = int(getattr(beat, "beat_id", len(output) + 1))
        row: list[Any] = []
        for lead in leads:
            item = by_cell.get((beat_id, lead))
            try:
                row.append(transform(getter(item)) if item is not None else None)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ComputationInvariantError(
                    "malformed legacy measurement cannot be converted to an absence",
                    beat_id=beat_id, lead=lead,
                ) from exc
        output.append(row)
    return output


def _safe_patient(patient: Any) -> dict[str, Any]:
    if patient is None:
        return {}
    return {
        key: value for key in ("age", "age_days", "sex", "patient_id")
        if (value := getattr(patient, key, None)) is not None
    }


def _json_safe(value: Any) -> Any:
    """Keep debug provenance JSON-compatible without exposing live objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return str(value)


class _CellReasons:
    """Per-field, per-cell absence reasons collected while building matrices."""

    def __init__(self) -> None:
        self._reasons: dict[str, dict[tuple[int, int], str]] = {}

    def set(self, field: str, cell: tuple[int, int], reason: str) -> None:
        # The first specific reason wins; a later generic reason never masks it.
        self._reasons.setdefault(field, {}).setdefault(cell, reason)

    def get(self, field: str, cell: tuple[int, int]) -> str:
        return self._reasons.get(field, {}).get(cell, REASON_UNAVAILABLE)


def _enforce_published_invariants(fiducials: dict[str, dict[str, Any]], intervals: dict[str, dict[str, Any]],
                                  reasons: _CellReasons) -> None:
    """Withhold cells that would publish an impossible ordering or duration."""

    def cells(name: str) -> list[list[Any]]:
        return fiducials[name]["values"]

    def withhold(group: dict[str, dict[str, Any]], name: str, i: int, j: int, reason: str) -> None:
        if name in group and group[name]["values"][i][j] is not None:
            group[name]["values"][i][j] = None
            reasons.set(name, (i, j), reason)

    p_on, p_off = cells("p_onset"), cells("p_offset")
    q_on, r_pk, q_off = cells("qrs_onset"), cells("r_peak"), cells("qrs_offset")
    for i, row in enumerate(q_on):
        for j in range(len(row)):
            if p_on[i][j] is not None and p_off[i][j] is not None and p_on[i][j] > p_off[i][j]:
                for name in ("p_onset", "p_offset"):
                    withhold(fiducials, name, i, j, REASON_ORDER)
                withhold(intervals, "p_duration_ms", i, j, REASON_ORDER)
            onset, peak, offset = q_on[i][j], r_pk[i][j], q_off[i][j]
            inverted = onset is not None and offset is not None and onset > offset
            peak_outside = (not inverted and None not in (onset, peak, offset) and not onset <= peak <= offset)
            if inverted or peak_outside:
                for name in ("qrs_onset", "qrs_offset") + (("r_peak",) if peak_outside else ()):
                    withhold(fiducials, name, i, j, REASON_ORDER)
                withhold(intervals, "qrs_duration_ms", i, j, REASON_ORDER)
    for name, field in intervals.items():
        if field["axes"] != ["beat", "lead"]:
            continue
        for i, row in enumerate(field["values"]):
            for j, value in enumerate(row):
                if value is not None and value < 0:
                    withhold(intervals, name, i, j, REASON_NEGATIVE)


def _encode_absence(missing: dict[str, str], cell_count: int) -> dict[str, Any]:
    """Compact, lossless absence encoding for a dense field.

    Every missing cell here is unmeasurable.  One reason shared by all cells
    uses the whole-field form; otherwise the most frequent reason (ties broken
    by name) becomes the field ``default`` and only the other cells are listed.
    """
    counts: dict[str, int] = {}
    for reason in missing.values():
        counts[reason] = counts.get(reason, 0) + 1
    default = min(counts, key=lambda reason: (-counts[reason], reason))
    if len(missing) == cell_count and len(counts) == 1:
        return {"kind": "unmeasurable", "reason": default}
    encoded: dict[str, Any] = {"default": {"kind": "unmeasurable", "reason": default}}
    exceptions = {coordinate: reason for coordinate, reason in missing.items() if reason != default}
    if exceptions:
        encoded["unmeasurable"] = exceptions
    return encoded


def build_record_document(measurements: Any, profile: str) -> dict[str, Any]:
    """Build the wire document for ``profile`` from an ``ECGMeasurements`` state."""

    prepared = measurements.prepared
    features = measurements.legacy_features
    leads = list(prepared.lead_names)
    metadata = getattr(features, "metadata", {})
    slot_map = metadata.get("input_contract", {}).get("lead_slot_map") or metadata.get("limited_lead_capabilities", {}).get("channel_to_slot", {})
    source_leads = [slot_map.get(lead, lead) for lead in leads]
    internal_fs = float(features.fs)
    if not np.isfinite(internal_fs) or internal_fs <= 0:
        raise ComputationInvariantError("engine sampling rate must be finite and positive")
    reasons = _CellReasons()

    def to_native(value: Any) -> int | None:
        number = _as_float(value)
        if number is None:
            return None
        return int(round(number * prepared.sampling_rate / internal_fs))

    def in_range(sample: int | None) -> int | None:
        return sample if sample is not None and 0 <= sample < prepared.signal.shape[1] else None

    beats = list(getattr(features, "beats", []) or [])
    beat_axis = [
        {"id": f"b{index + 1:04d}", "r_sample": in_range(to_native(getattr(beat, "r_index", None)))}
        for index, beat in enumerate(beats)
    ]
    fp = fingerprint_input(prepared.signal, fs_hz=prepared.sampling_rate, channels=prepared.lead_names, amplitude_unit=prepared.amplitude_unit)
    provenance = dict(measurements.provenance)
    config_payload = provenance.pop("config")
    config_hash = provenance.pop("config_hash")
    provenance.update({
        "library_version": LIBRARY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "input_fingerprint": "sha256:" + fp.sha256,
        "internal_sample_rate_hz": internal_fs,
        "fiducial_sample_domain": "native",
        "coordinate_conversion": "round_half_even(internal_sample * input_fs / internal_fs)",
        "mains_frequency_hz": metadata.get("mains_frequency_hz", measurements.config.mains_frequency_hz),
        "input": {
            "sha256": "sha256:" + fp.sha256,
            "n_channels": fp.n_channels,
            "n_samples": fp.n_samples,
            "fs_hz": fp.fs_hz,
            "channels": list(fp.channels),
        },
        "config": {"id": config_hash, "sha256": config_hash, "resolved": config_payload},
        "config_hash": config_hash,
        "method_requested": measurements.provenance["method_requested"],
        "method_resolved": measurements.provenance["method_resolved"],
        "fallback_path": list(measurements.provenance.get("fallback_path", [])),
        "policy_execution": "embedded_legacy_engine; modular_policies_not_yet_extracted",
        "algorithm_paths": {
            "delineation": "legacy_delineation",
            "st_80ms_published": "legacy_native_st_80ms_mv",
            "st_hybrid_candidate_source": measurements.config.st_amplitude_source,
            "qt": "legacy_qt_policy",
        },
        "fields": {
            "delineation_default": {"method": "selected_legacy_fiducials"},
            "measurement_default": {"method": "selected_legacy_measurements"},
            "st_native": {"method": "legacy_native_st_80ms_mv", "signal_domain": "analysis", "hybrid_candidates_published": False},
        },
    })
    field_ref_d = "/provenance/fields/delineation_default"
    field_ref_m = "/provenance/fields/measurement_default"
    get = lambda name: lambda item: getattr(item, name)
    wave_get = lambda wave, bound: lambda item: getattr(getattr(item, wave), bound)
    cells = features.beat_features

    def dense(group: str, name: str, getter: Callable[[Any], Any], *, unit: str, ref: str,
              transform: Callable[[Any], Any] = lambda value: value) -> dict[str, Any]:
        root = "/delineation/fiducials" if group == "fiducials" else f"/measurements/{group}"
        return _field_entry(f"{root}/{name}", _matrix(cells, beats, source_leads, getter, transform=transform),
                            type_name="integer", unit=unit, axes=["beat", "lead"], provenance_ref=ref)

    fiducial_sources = {
        "p_onset": wave_get("p", "onset"), "p_offset": wave_get("p", "offset"),
        "qrs_onset": wave_get("qrs", "onset"), "r_peak": get("r_peak_index"),
        "qrs_offset": wave_get("qrs", "offset"), "j_point": get("j_index"), "t_offset": wave_get("t", "offset"),
    }
    fiducials = {name: dense("fiducials", name, source, unit="sample", ref=field_ref_d, transform=to_native)
                 for name, source in fiducial_sources.items()}
    for name, field in fiducials.items():
        for i, row in enumerate(field["values"]):
            for j, value in enumerate(row):
                if value is not None and in_range(value) is None:
                    row[j] = None
                    reasons.set(name, (i, j), REASON_OUTSIDE)
    intervals = {
        "p_duration_ms": dense("intervals", "p_duration_ms", get("p_dur_ms"), unit="ms", ref=field_ref_m, transform=_round_even),
        "pr_interval_ms": dense("intervals", "pr_interval_ms", get("pr_ms"), unit="ms", ref=field_ref_m, transform=_round_even),
        "qrs_duration_ms": dense("intervals", "qrs_duration_ms", get("qrs_ms"), unit="ms", ref=field_ref_m, transform=_round_even),
        "qt_interval_ms": dense("intervals", "qt_interval_ms", get("qt_ms"), unit="ms", ref=field_ref_m, transform=_round_even),
        "tpeak_tend_ms": dense("intervals", "tpeak_tend_ms", get("tpe_ms"), unit="ms", ref=field_ref_m, transform=_round_even),
    }
    amplitudes = {
        "p_amplitude_uv": dense("amplitudes", "p_amplitude_uv", get("p_amp_mv"), unit="uV", ref=field_ref_m, transform=_uv),
        "r_amplitude_uv": dense("amplitudes", "r_amplitude_uv", get("r_amp_mv"), unit="uV", ref=field_ref_m, transform=_uv),
        "s_amplitude_uv": dense("amplitudes", "s_amplitude_uv", get("s_amp_mv"), unit="uV", ref=field_ref_m, transform=_uv),
        "st_80ms_uv": dense("amplitudes", "st_80ms_uv", get("st_80ms_mv"), unit="uV", ref="/provenance/fields/st_native", transform=_uv),
        "t_amplitude_uv": dense("amplitudes", "t_amplitude_uv", get("t_amp_mv"), unit="uV", ref=field_ref_m, transform=_uv),
    }
    areas = {
        # Legacy trapezoid() uses dx=1: mV * sample -> uV * ms.
        "qrs_signed_area_uv_ms": dense("areas", "qrs_signed_area_uv_ms", get("qrs_signed_area"), unit="uV_ms", ref=field_ref_m,
                                       transform=lambda value: _round_even(None if _as_float(value) is None else float(value) * 1_000_000 / internal_fs)),
    }
    if profile in {"all", "debug"}:
        intervals["jt_interval_ms"] = dense("intervals", "jt_interval_ms", get("jt_ms"), unit="ms", ref=field_ref_m, transform=_round_even)
        amplitudes["u_amplitude_uv"] = dense("amplitudes", "u_amplitude_uv", get("u_amp_signed_mv"), unit="uV", ref=field_ref_m, transform=_uv)
    _enforce_published_invariants(fiducials, intervals, reasons)

    global_features = getattr(features, "global_features", None)
    limited = prepared.input_mode == "limited"

    def scalar(name: str, attribute: str, unit: str) -> dict[str, Any]:
        value = _round_even(_as_float(getattr(global_features, attribute, None)))
        return _field_entry(f"/measurements/global/{name}", value, type_name="integer", unit=unit, axes=[], provenance_ref=field_ref_m)

    global_measurements = {
        "heart_rate_bpm": scalar("heart_rate_bpm", "heart_rate_bpm", "1/min"),
        "frontal_qrs_axis_deg": scalar("frontal_qrs_axis_deg", "qrs_axis_deg", "deg"),
    }
    if profile in {"all", "debug"}:
        global_measurements["qtc_bazett_ms"] = scalar("qtc_bazett_ms", "qtc_bazett_ms", "ms")
        global_measurements["qrs_wide_ms"] = scalar("qrs_wide_ms", "qrs_wide_ms", "ms")
    if limited:
        for name in ("frontal_qrs_axis_deg", "qtc_bazett_ms"):
            if name in global_measurements:
                global_measurements[name].update(values=None, absence={"kind": "not_applicable", "reason": REASON_LIMITED})

    record_quality = metadata.get("record_quality", {})
    quality_leads: dict[str, Any] = {}
    legacy_quality = getattr(features, "quality", {}) or {}
    for lead in leads:
        item = legacy_quality.get(slot_map.get(lead, lead))
        reliable = bool(getattr(item, "reliable", False)) if item is not None else False
        quality_leads[lead] = {"status": "usable" if reliable else "limited", **({"reason": "quality_gate"} if not reliable else {})}
    # Profile does not enter run identity; signal calibration, configuration
    # and patient metadata do because they can change the measured result.
    patient_payload = asdict(prepared.patient) if is_dataclass(prepared.patient) else prepared.patient
    identity = {"input": fp.sha256, "config": config_hash, "patient": patient_payload,
                "method": provenance["method_resolved"], "library": LIBRARY_VERSION, "schema": SCHEMA_VERSION}
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    raw_hash = hashlib.sha256(np.asarray(prepared.signal, dtype="<f8", order="C").tobytes()).hexdigest()
    for group in (fiducials, intervals, amplitudes, areas, global_measurements):
        for name, field in group.items():
            if "absence" in field:
                continue
            if field["axes"] == ["beat", "lead"]:
                missing = {f"{beat_axis[i]['id']}|{leads[j]}": reasons.get(name, (i, j))
                           for i, row in enumerate(field["values"]) for j, value in enumerate(row) if value is None}
                if missing:
                    field["absence"] = _encode_absence(missing, len(beats) * len(leads))
            elif field["values"] is None:
                field["absence"] = {"kind": "unmeasurable", "reason": REASON_UNAVAILABLE}
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "record_id": "rec-" + run_hash,
        "intended_use": "research_and_engineering_only",
        "acquisition": {
            "sample_rate_hz": prepared.sampling_rate,
            "sample_count": int(prepared.signal.shape[1]),
            "input_mode": prepared.input_mode,
            "leads": leads,
            "amplitude_unit": prepared.amplitude_unit,
            "capabilities": {
                "diagnosis": {"status": "not_applicable", "reason": "measurement_package_boundary"},
                "axis": {"status": "not_applicable", "reason": REASON_LIMITED} if limited else {"status": "available"},
                "formal_qt": {"status": "not_applicable", "reason": REASON_LIMITED} if limited else {"status": "available"},
            },
        },
        "axes": {"beats": beat_axis, "leads": leads},
        "quality": {"record": {"status": "usable" if record_quality.get("record_grade") in {"Q0", "Q1"} else "limited", "grade": record_quality.get("record_grade")}, "leads": quality_leads},
        "provenance": provenance,
        "delineation": {"fiducials": fiducials},
        "measurements": {"intervals": intervals, "amplitudes": amplitudes, "areas": areas, "flags": {}, "global": global_measurements},
        "artifacts": {
            "raw_signal": {"uri": "urn:sha256:" + raw_hash, "media_type": "application/octet-stream", "sha256": "sha256:" + raw_hash,
                           "encoding": "little_endian_float64_c_order", "shape": list(prepared.signal.shape), "amplitude_unit": prepared.amplitude_unit},
            "dense_measurements": None,
        },
    }
    if _safe_patient(prepared.patient):
        document["metadata"] = {"patient": _safe_patient(prepared.patient)}
    if profile == "debug":
        # Measurement records do not export diagnostic interpretations, even
        # in debug. Expose only measurement-engine metadata selected by name.
        allowed = {"input_contract", "limited_lead_capabilities", "record_quality", "mains_frequency_hz", "refinement_config"}
        document["debug"] = {"legacy_metadata": _json_safe({key: value for key, value in metadata.items() if key in allowed})}
    return document


__all__ = ["build_record_document", "SCHEMA_VERSION", "LIBRARY_VERSION"]
