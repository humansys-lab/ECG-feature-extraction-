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
    """Validate and normalize signal, channel, unit and patient metadata into an immutable input."""
    array, fs, names, unit = _validate_prepare(ecg, sampling_rate, lead_names, amplitude_unit, input_mode)
    return ECGInput(array, fs, names, unit, _patient_meta(patient), input_mode)


def _settings_from_config(config: ECGConfig) -> Any:
    """Resolved engine settings for ``config`` (the legacy extractor's defaults).

    Equal to ``ExtractorSettings.from_extractor(ECGFeatureExtractor(...))`` for
    the same options (tested), without importing the legacy entry point.
    """
    from .context import ExtractorSettings

    return ExtractorSettings(
        fs_internal=None if config.fs_internal is None else int(config.fs_internal),
        mains_freq=config.mains_frequency_hz,
        lp_hz=40.0,
        enable_pacing=True,
        enable_lead_reversal=True,
        compute_grouping=True,
        enable_hybrid_r_localization=True,
        enable_hybrid_wave_localization=True,
        enable_hybrid_st_measurement=True,
        enable_t_wave_refinement=True,
        st_amplitude_source=config.st_amplitude_source,
        refinement=config.refinement,
        input_mode="limited" if config.input_mode == "limited" else "standard",
    )


# --------------------------------------------------------------------------- #
# Staged legacy-parity pipeline (library migration Phase 3).
#
# The pre-decomposition ``ECGFeatureExtractor.extract`` body runs as the fixed
# stage sequence below; each entry is (label, stage module, stage function).
# The beats stage refines its measurement group after primary delineation, and
# delineation settles the QRS tail and refines T after that refinement, exactly
# where the legacy body did.  Stage modules are imported lazily so that
# record-only users of ``ecgfeat.pipeline`` do not load the numerical engine.
# --------------------------------------------------------------------------- #

LEGACY_ORCHESTRATION_ENV = "ECGFEAT_LEGACY_ORCHESTRATION"

LEGACY_STAGE_SEQUENCE: tuple[tuple[str, str, str], ...] = (
    ("input", "input", "run"),
    ("quality", "quality", "run"),
    ("ventricular", "ventricular", "run"),
    ("beats", "beats", "run"),
    ("delineation", "delineation", "run"),
    ("beats.refine_measurement_group", "beats", "refine_measurement_group"),
    ("delineation.settle_qrs_tail_and_refine_t", "delineation", "settle_qrs_tail_and_refine_t"),
    ("atrial", "atrial", "run"),
    ("measurement", "measurement", "run"),
    ("finalize", "finalize", "run"),
)


def legacy_orchestration_requested() -> bool:
    """True when ``ECGFEAT_LEGACY_ORCHESTRATION`` routes extraction to the rollback path.

    The rollback path is the preserved pre-decomposition orchestration in
    ``ecgfeat.compat._api_v0_legacy_orchestration`` (migration plan, Phase 3 / F).
    """
    import os

    return os.environ.get(LEGACY_ORCHESTRATION_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class LegacyPipelineRun:
    """Final pipeline context of one staged legacy-parity extraction."""

    context: Any

    @property
    def features(self) -> Any:
        return self.context.legacy_features

    @property
    def policy_events(self) -> tuple[Any, ...]:
        return self.context.policy_events


def run_legacy_pipeline(
    extractor: Any,
    ecg_12lead: Any,
    fs: Any,
    meta: Any = None,
    *,
    lead_names: Any = None,
    amplitude_unit: Any = None,
    gain_uv_per_lsb: Any = None,
    prior_features: Any = None,
    hooks: Any = None,
    stop_after: str | None = None,
) -> LegacyPipelineRun:
    """Run the staged pipeline that reproduces ``ECGFeatureExtractor.extract``.

    ``extractor`` supplies the resolved legacy configuration (any object with the
    ``ECGFeatureExtractor`` attributes).  ``hooks`` are the interpretation hooks the
    legacy entry point injects (``ecgfeat.compat.interpretation_hooks``); without
    them no interpretation output is produced.  ``stop_after`` names a stage label
    after which to stop (diagnosis and ablation only).
    """
    from dataclasses import replace
    from importlib import import_module

    from .context import ExtractionRequest, ExtractorSettings, PipelineContext

    labels = [label for label, _module, _function in LEGACY_STAGE_SEQUENCE]
    if stop_after is not None and stop_after not in labels:
        raise ValueError(f"unknown stage label {stop_after!r}; expected one of {labels}")
    context = PipelineContext(
        config=ExtractorSettings.from_extractor(extractor),
        patient=meta,
        request=ExtractionRequest(ecg_12lead, fs, meta, lead_names, amplitude_unit, gain_uv_per_lsb, prior_features),
        hooks=hooks,
    )
    for label, module, function in LEGACY_STAGE_SEQUENCE:
        stage = getattr(import_module(f".stages.{module}", __package__), function)
        result = stage(context)
        context = result.context
        if result.issues:
            context = replace(context, issues=context.issues + tuple(result.issues))
        if label == stop_after:
            break
    return LegacyPipelineRun(context)


def ecg_measure(prepared: ECGInput, *, method: str = "default", config: ECGConfig | None = None) -> ECGMeasurements:
    """Run the measurement pipeline and return measurements plus provenance before profile projection."""
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
    legacy = _staged_legacy_features(_settings_from_config(resolved), prepared)
    return ECGMeasurements(prepared, legacy, resolved, config_provenance(resolved, method=method))


def _staged_legacy_features(settings: Any, prepared: ECGInput) -> Any:
    """Completed engine measurement state for the record path, via the staged pipeline.

    No interpretation hooks are injected: the ECG Record publishes measurements
    only, so core-only installations extract records without ``ecginterpret``
    (the record-crosswalk gate proves no published value depends on them).  The
    ``ECGFEAT_LEGACY_ORCHESTRATION`` rollback switch applies to the legacy entry
    point (``ecgfeat.compat.api_v0``) only.
    """
    return run_legacy_pipeline(
        settings, prepared.signal, prepared.sampling_rate, hooks=None, meta=deepcopy(prepared.patient),
        lead_names=list(prepared.lead_names), amplitude_unit=prepared.amplitude_unit,
    ).features


def ecg_emit(measurements: ECGMeasurements, *, profile: str = "summary") -> ECGRecord:
    """Assemble and validate a versioned ECG Record for ``profile`` from measured quantities."""
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
    """Measure an ECG and return one validated, versioned ECG Record.

    ``ecg`` is channel-major ``(n_leads, n_samples)`` with one explicit name per row;
    ``input_mode="limited"`` accepts 1-8 named channels. ``profile`` selects
    ``summary`` (default), ``all`` or ``debug``."""
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
    """Serialize an ECG Record as UTF-8 JSON without changing its content (compact unless ``indent``)."""
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
