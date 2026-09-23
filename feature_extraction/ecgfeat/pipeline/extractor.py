"""Compatibility-backed ECG Record extraction pipeline.

The existing numerical implementation remains the measurement engine during
migration.  This module owns the public input contract and converts its typed
legacy result into the versioned, consumer-safe ECG Record document.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import ECGConfig, STANDARD_12_LEADS, config_provenance
from ..errors import (
    AmplitudeUnitError,
    ConfigurationError,
    ComputationInvariantError,
    LeadNameError,
    SamplingRateError,
    SignalShapeError,
)
from ..models import PatientMeta as LegacyPatientMeta
from ..record.model import ECGRecord, record_from_document
from ..record.provenance import fingerprint_input
from ..record.serialize import dumps_record, validate_record

SCHEMA_VERSION = "1.0.0"
LIBRARY_VERSION = "0.1.0"


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(round(number))


def _round_even(value: float | None) -> int | None:
    number = _as_float(value)
    return None if number is None else int(round(number))


def _patient_meta(patient: Any) -> Any:
    if patient is None or isinstance(patient, LegacyPatientMeta):
        return patient
    if isinstance(patient, Mapping):
        unknown = set(patient) - LegacyPatientMeta.__dataclass_fields__.keys()
        if unknown:
            raise ConfigurationError(f"unknown patient metadata members: {sorted(unknown)}")
        allowed = {key: patient[key] for key in LegacyPatientMeta.__dataclass_fields__ if key in patient}
        return LegacyPatientMeta(**allowed)
    if is_dataclass(patient):
        return _patient_meta(asdict(patient))
    raise ConfigurationError("patient must be PatientMeta or a metadata mapping")


@dataclass(frozen=True, slots=True)
class ECGInput:
    signal: np.ndarray
    sampling_rate: float
    lead_names: tuple[str, ...]
    amplitude_unit: str
    patient: Any = None
    input_mode: str = "standard_12"

    def __post_init__(self) -> None:
        # Immutable bytes-backed copy: callers keep a writable input and
        # cannot mutate prepared samples through an alias or setflags().
        source, fs, names, unit = _validate_prepare(self.signal, self.sampling_rate, self.lead_names, self.amplitude_unit, self.input_mode)
        array = np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)
        object.__setattr__(self, "signal", array)
        object.__setattr__(self, "sampling_rate", fs)
        object.__setattr__(self, "lead_names", names)
        object.__setattr__(self, "amplitude_unit", unit)
        object.__setattr__(self, "patient", deepcopy(_patient_meta(self.patient)))


@dataclass(frozen=True, slots=True)
class ECGMeasurements:
    prepared: ECGInput
    legacy_features: Any
    config: ECGConfig
    provenance: Mapping[str, Any]


def _validate_prepare(
    ecg: Any,
    sampling_rate: Any,
    lead_names: Sequence[str],
    amplitude_unit: str,
    input_mode: str,
) -> tuple[np.ndarray, float, tuple[str, ...], str]:
    try:
        fs = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise SamplingRateError("sampling_rate must be finite and positive", code="invalid_sampling_rate") from exc
    if isinstance(sampling_rate, bool) or not np.isfinite(fs) or fs < 100:
        raise SamplingRateError("sampling_rate must be finite and at least 100 Hz", code="invalid_sampling_rate")
    try:
        array = np.asarray(ecg, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SignalShapeError("ecg must be a numeric 2D array", code="invalid_signal_type") from exc
    if array.ndim != 2:
        raise SignalShapeError("ecg must have shape (n_leads, n_samples)", code="invalid_signal_shape")
    if isinstance(lead_names, (str, bytes)) or not isinstance(lead_names, Sequence):
        raise LeadNameError("lead_names must be a sequence of names")
    if any(not isinstance(item, str) or "|" in item for item in lead_names):
        raise LeadNameError("lead names must be strings without '|'")
    names = tuple(item.strip() for item in lead_names)
    if any(not item for item in names) or len(set(names)) != len(names):
        raise LeadNameError("lead_names must be unique and non-empty", code="invalid_lead_names")
    if input_mode not in {"standard_12", "limited"}:
        raise ConfigurationError("input_mode must be 'standard_12' or 'limited'", code="invalid_input_mode")
    if input_mode == "standard_12" and (len(names) != 12 or set(names) != set(STANDARD_12_LEADS)):
        raise LeadNameError("standard_12 input requires each canonical lead exactly once", code="standard_lead_contract")
    if input_mode == "limited" and not 1 <= len(names) <= 8:
        raise LeadNameError("limited input requires between 1 and 8 explicit channels", code="limited_lead_count")
    if array.shape[0] != len(names):
        raise SignalShapeError(
            f"signal must be channel-major with {len(names)} rows",
            code="signal_lead_count_mismatch",
        )
    if array.shape[1] == 0:
        raise SignalShapeError("ecg must contain at least one sample", code="empty_signal")
    if array.shape[1] / fs < 1.0:
        raise SignalShapeError("ECG duration must be at least one second", code="record_too_short")
    if not np.all(np.isfinite(array)):
        raise SignalShapeError("ecg contains NaN or infinity", code="nonfinite_signal")
    unit = str(amplitude_unit).strip().replace("μ", "u").replace("µ", "u").lower()
    if unit not in {"mv", "uv"}:
        raise AmplitudeUnitError("amplitude_unit must be 'mV' or 'uV'", code="unsupported_amplitude_unit")
    return np.ascontiguousarray(array), fs, names, "mV" if unit == "mv" else "uV"


def ecg_prepare(
    ecg: Any,
    *,
    sampling_rate: float,
    lead_names: Sequence[str],
    amplitude_unit: str = "mV",
    patient: Any = None,
    input_mode: str = "standard_12",
) -> ECGInput:
    array, fs, names, unit = _validate_prepare(ecg, sampling_rate, lead_names, amplitude_unit, input_mode)
    return ECGInput(array, fs, names, unit, _patient_meta(patient), input_mode)


def _legacy_refinement(config: ECGConfig) -> Any:
    return config.refinement


def _legacy_extractor(config: ECGConfig) -> Any:
    from ..api import ECGFeatureExtractor

    return ECGFeatureExtractor(
        fs_internal=None if config.fs_internal is None else int(config.fs_internal),
        mains_freq=config.mains_frequency_hz,
        st_amplitude_source=config.st_amplitude_source,
        refinement=_legacy_refinement(config),
        input_mode="limited" if config.input_mode == "limited" else "standard",
    )


def ecg_measure(prepared: ECGInput, *, method: str = "default", config: ECGConfig | None = None) -> ECGMeasurements:
    if not isinstance(prepared, ECGInput):
        raise ConfigurationError("prepared must be an ECGInput from ecg_prepare")
    if method not in {"default", "legacy_dxl_inspired"}:
        raise ConfigurationError(f"unsupported algorithm method: {method}", code="unknown_method")
    if config is not None and hasattr(config, "input") and getattr(config.input, "mode", None) == "limited":
        if config.input.channels != prepared.lead_names:
            raise ConfigurationError("configured channels do not match prepared input")
    if config is not None and not isinstance(config, ECGConfig) and hasattr(config, "to_ecg_config"):
        config = config.to_ecg_config()  # type: ignore[assignment]
    resolved = config or ECGConfig(input_mode=prepared.input_mode)
    if not isinstance(resolved, ECGConfig):
        raise ConfigurationError("config must be ECGConfig or ExtractionConfig")
    if resolved.input_mode != prepared.input_mode:
        raise ConfigurationError("config input_mode does not match prepared input", code="input_mode_mismatch")
    extractor = _legacy_extractor(resolved)
    legacy = extractor.extract(
        prepared.signal,
        prepared.sampling_rate,
        meta=deepcopy(prepared.patient),
        lead_names=list(prepared.lead_names),
        amplitude_unit=prepared.amplitude_unit,
    )
    return ECGMeasurements(prepared, legacy, resolved, config_provenance(resolved, method=method))


def _field_entry(
    values: Any,
    *,
    type_name: str,
    unit: str | None,
    axes: list[str],
    provenance_ref: str,
    status: str = "unvalidated",
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": type_name,
        "unit": unit,
        "axes": axes,
        "values": values,
        "nullable": True,
        "validation": {"status": status, "evidence": evidence or []},
        "provenance_ref": provenance_ref,
    }


def _matrix(
    beat_features: Sequence[Any],
    beats: Sequence[Any],
    leads: Sequence[str],
    getter: Any,
    *,
    transform: Any = lambda value: value,
) -> list[list[Any]]:
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
        from dataclasses import asdict
        return _json_safe(asdict(value))
    return str(value)


def _record_document(measurements: ECGMeasurements, profile: str) -> dict[str, Any]:
    prepared = measurements.prepared
    features = measurements.legacy_features
    leads = list(prepared.lead_names)
    metadata = getattr(features, "metadata", {})
    slot_map = metadata.get("input_contract", {}).get("lead_slot_map") or metadata.get("limited_lead_capabilities", {}).get("channel_to_slot", {})
    source_leads = [slot_map.get(lead, lead) for lead in leads]
    internal_fs = float(features.fs)
    if not np.isfinite(internal_fs) or internal_fs <= 0:
        raise ComputationInvariantError("engine sampling rate must be finite and positive")

    def native_sample(value: Any) -> int | None:
        number = _as_float(value)
        if number is None:
            return None
        sample = int(round(number * prepared.sampling_rate / internal_fs))
        return sample if 0 <= sample < prepared.signal.shape[1] else None
    beats = list(getattr(features, "beats", []) or [])
    if not beats:
        # An empty beat axis is valid for a quality-only record; use no dense cells.
        beats = []
    beat_axis = [
        {"id": f"b{index + 1:04d}", "r_sample": native_sample(getattr(beat, "r_index", None))}
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
    fiducials = {
        "p_onset": _field_entry(_matrix(features.beat_features, beats, source_leads, wave_get("p", "onset")), type_name="integer", unit="sample", axes=["beat", "lead"], provenance_ref=field_ref_d),
        "p_offset": _field_entry(_matrix(features.beat_features, beats, source_leads, wave_get("p", "offset")), type_name="integer", unit="sample", axes=["beat", "lead"], provenance_ref=field_ref_d),
        "qrs_onset": _field_entry(_matrix(features.beat_features, beats, source_leads, wave_get("qrs", "onset")), type_name="integer", unit="sample", axes=["beat", "lead"], provenance_ref=field_ref_d),
        "r_peak": _field_entry(_matrix(features.beat_features, beats, source_leads, get("r_peak_index")), type_name="integer", unit="sample", axes=["beat", "lead"], provenance_ref=field_ref_d),
        "qrs_offset": _field_entry(_matrix(features.beat_features, beats, source_leads, wave_get("qrs", "offset")), type_name="integer", unit="sample", axes=["beat", "lead"], provenance_ref=field_ref_d),
        "j_point": _field_entry(_matrix(features.beat_features, beats, source_leads, get("j_index")), type_name="integer", unit="sample", axes=["beat", "lead"], provenance_ref=field_ref_d),
        "t_offset": _field_entry(_matrix(features.beat_features, beats, source_leads, wave_get("t", "offset")), type_name="integer", unit="sample", axes=["beat", "lead"], provenance_ref=field_ref_d),
    }
    intervals = {
        "p_duration_ms": _field_entry(_matrix(features.beat_features, beats, source_leads, get("p_dur_ms"), transform=_round_even), type_name="integer", unit="ms", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "pr_interval_ms": _field_entry(_matrix(features.beat_features, beats, source_leads, get("pr_ms"), transform=_round_even), type_name="integer", unit="ms", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "qrs_duration_ms": _field_entry(_matrix(features.beat_features, beats, source_leads, get("qrs_ms"), transform=_round_even), type_name="integer", unit="ms", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "qt_interval_ms": _field_entry(_matrix(features.beat_features, beats, source_leads, get("qt_ms"), transform=_round_even), type_name="integer", unit="ms", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "tpeak_tend_ms": _field_entry(_matrix(features.beat_features, beats, source_leads, get("tpe_ms"), transform=_round_even), type_name="integer", unit="ms", axes=["beat", "lead"], provenance_ref=field_ref_m),
    }
    amplitudes = {
        "p_amplitude_uv": _field_entry(_matrix(features.beat_features, beats, source_leads, get("p_amp_mv"), transform=lambda value: _round_even(None if value is None else float(value) * 1000)), type_name="integer", unit="uV", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "r_amplitude_uv": _field_entry(_matrix(features.beat_features, beats, source_leads, get("r_amp_mv"), transform=lambda value: _round_even(None if value is None else float(value) * 1000)), type_name="integer", unit="uV", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "s_amplitude_uv": _field_entry(_matrix(features.beat_features, beats, source_leads, get("s_amp_mv"), transform=lambda value: _round_even(None if value is None else float(value) * 1000)), type_name="integer", unit="uV", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "st_80ms_uv": _field_entry(_matrix(features.beat_features, beats, source_leads, get("st_80ms_mv"), transform=lambda value: _round_even(None if value is None else float(value) * 1000)), type_name="integer", unit="uV", axes=["beat", "lead"], provenance_ref=field_ref_m),
        "t_amplitude_uv": _field_entry(_matrix(features.beat_features, beats, source_leads, get("t_amp_mv"), transform=lambda value: _round_even(None if value is None else float(value) * 1000)), type_name="integer", unit="uV", axes=["beat", "lead"], provenance_ref=field_ref_m),
    }
    areas = {
        # Legacy trapezoid() uses dx=1: mV * sample -> uV * ms.
        "qrs_signed_area_uv_ms": _field_entry(_matrix(features.beat_features, beats, source_leads, get("qrs_signed_area"), transform=lambda value: _round_even(None if value is None else float(value) * 1_000_000 / internal_fs)), type_name="integer", unit="uV_ms", axes=["beat", "lead"], provenance_ref=field_ref_m),
    }
    amplitudes["st_80ms_uv"]["provenance_ref"] = "/provenance/fields/st_native"
    global_features = getattr(features, "global_features", None)
    global_values = {
        "heart_rate_bpm": _round_even(_as_float(getattr(global_features, "heart_rate_bpm", None))),
        "frontal_qrs_axis_deg": _round_even(_as_float(getattr(global_features, "qrs_axis_deg", None))) if prepared.input_mode != "limited" else None,
    }
    global_measurements = {
        "heart_rate_bpm": _field_entry(global_values["heart_rate_bpm"], type_name="integer", unit="1/min", axes=[], provenance_ref=field_ref_m),
        "frontal_qrs_axis_deg": _field_entry(global_values["frontal_qrs_axis_deg"], type_name="integer", unit="deg", axes=[], provenance_ref=field_ref_m),
    }
    if profile in {"all", "debug"}:
        intervals["jt_interval_ms"] = _field_entry(
            _matrix(features.beat_features, beats, source_leads, get("jt_ms"), transform=_round_even),
            type_name="integer", unit="ms", axes=["beat", "lead"], provenance_ref=field_ref_m,
        )
        amplitudes["u_amplitude_uv"] = _field_entry(
            _matrix(features.beat_features, beats, source_leads, get("u_amp_signed_mv"), transform=lambda value: _round_even(None if value is None else float(value) * 1000)),
            type_name="integer", unit="uV", axes=["beat", "lead"], provenance_ref=field_ref_m,
        )
        global_measurements_extra = {
            "qtc_bazett_ms": _round_even(_as_float(getattr(global_features, "qtc_bazett_ms", None))),
            "qrs_wide_ms": _round_even(_as_float(getattr(global_features, "qrs_wide_ms", None))),
        }
    else:
        global_measurements_extra = {}
    global_measurements.update({
        name: _field_entry(value, type_name="integer", unit="ms", axes=[], provenance_ref=field_ref_m)
        for name, value in global_measurements_extra.items()
    })
    if prepared.input_mode == "limited":
        global_measurements["frontal_qrs_axis_deg"]["absence"] = {"kind": "not_applicable", "reason": "input_mode_limited"}
        if "qtc_bazett_ms" in global_measurements:
            global_measurements["qtc_bazett_ms"].update(values=None, absence={"kind": "not_applicable", "reason": "input_mode_limited"})
    record_quality = getattr(features, "metadata", {}).get("record_quality", {})
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
    for field in fiducials.values():
        field["values"] = [[native_sample(value) for value in row] for row in field["values"]]
    for group in (fiducials, intervals, amplitudes, areas, global_measurements):
        for field in group.values():
            if "absence" in field:
                continue
            if field["axes"] == ["beat", "lead"]:
                missing = {f"{beat_axis[i]['id']}|{leads[j]}": "measurement_unavailable"
                           for i, row in enumerate(field["values"]) for j, value in enumerate(row) if value is None}
                if missing:
                    # A field-wide state avoids repeating identical reasons.
                    if len(missing) == len(beats) * len(leads):
                        field["absence"] = {"kind": "unmeasurable", "reason": "measurement_unavailable"}
                    else:
                        field["absence"] = {"unmeasurable": missing}
            elif field["values"] is None:
                field["absence"] = {"kind": "unmeasurable", "reason": "measurement_unavailable"}
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
                "axis": {"status": "not_applicable", "reason": "input_mode_limited"} if prepared.input_mode == "limited" else {"status": "available"},
                "formal_qt": {"status": "not_applicable", "reason": "input_mode_limited"} if prepared.input_mode == "limited" else {"status": "available"},
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


def ecg_emit(measurements: ECGMeasurements, *, profile: str = "summary") -> ECGRecord:
    if profile == "measurement":
        profile = "all"
    if profile not in {"summary", "all", "debug"}:
        raise ConfigurationError("profile must be 'summary', 'all', or 'debug'", code="invalid_profile")
    document = _record_document(measurements, profile)
    try:
        return validate_record(record_from_document(document))
    except Exception as exc:
        if isinstance(exc, (ConfigurationError, ComputationInvariantError)):
            raise
        raise ComputationInvariantError("failed to build a valid ECG Record", code="record_build_failed", cause=str(exc)) from exc


def ecg_record(
    ecg: Any,
    *,
    sampling_rate: float,
    lead_names: Sequence[str],
    amplitude_unit: str = "mV",
    patient: Any = None,
    method: str = "default",
    profile: str = "summary",
    config: ECGConfig | None = None,
    input_mode: str | None = None,
    record_id: str | None = None,
) -> ECGRecord:
    if config is None:
        configured_mode = "standard_12"
    elif hasattr(config, "input_mode"):
        configured_mode = str(config.input_mode)
    elif hasattr(config, "to_ecg_config"):
        configured_mode = str(config.to_ecg_config().input_mode)
    else:
        configured_mode = "standard_12"
    selected_mode = input_mode or configured_mode
    prepared = ecg_prepare(
        ecg,
        sampling_rate=sampling_rate,
        lead_names=lead_names,
        amplitude_unit=amplitude_unit,
        patient=patient,
        input_mode=selected_mode,
    )
    measured = ecg_measure(prepared, method=method, config=config)
    record = ecg_emit(measured, profile=profile)
    if record_id is not None:
        document = record.as_dict()
        document["record_id"] = record_id
        record = record_from_document(document)
    return record


extract_record = ecg_record


def extract_ecg_record(signal: Any, fs_hz: float, lead_names: Sequence[str], *, config: Any = None, patient: Any = None, record_id: str | None = None) -> ECGRecord:
    """Target-architecture spelling used by the staged pipeline guide."""
    return ecg_record(signal, sampling_rate=fs_hz, lead_names=lead_names, patient=patient, config=config, record_id=record_id, input_mode=(config.input_mode if hasattr(config, "input_mode") else None))


class ECGRecordExtractor:
    """Reusable, stateless extractor around the public record boundary."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def extract(self, signal: Any, fs_hz: float, lead_names: Sequence[str], *, patient: Any = None, record_id: str | None = None) -> ECGRecord:
        return extract_ecg_record(signal, fs_hz, lead_names, config=self.config, patient=patient, record_id=record_id)


def ecg_dump(record: ECGRecord, destination: Any, *, indent: int | None = None) -> None:
    payload = record.as_dict()
    data = json.dumps(payload, sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent, ensure_ascii=False, allow_nan=False)
    if hasattr(destination, "write"):
        destination.write(data)
    else:
        from pathlib import Path
        Path(destination).write_text(data, encoding="utf-8")


__all__ = [
    "ECGInput", "ECGMeasurements", "ecg_prepare", "ecg_measure", "ecg_emit",
    "ecg_record", "extract_record", "extract_ecg_record", "ECGRecordExtractor", "ecg_dump", "SCHEMA_VERSION", "LIBRARY_VERSION",
]
