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
from .record_builder import LIBRARY_VERSION, SCHEMA_VERSION, build_record_document



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


def ecg_emit(measurements: ECGMeasurements, *, profile: str = "summary") -> ECGRecord:
    if profile == "measurement":
        profile = "all"
    if profile not in {"summary", "all", "debug"}:
        raise ConfigurationError("profile must be 'summary', 'all', or 'debug'", code="invalid_profile")
    document = build_record_document(measurements, profile)
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
