"""Immutable configuration for the public ECG Record extraction API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence, TypeAlias

from .errors import ConfigurationError, LeadNameError, SamplingRateError

InputModeName: TypeAlias = Literal["standard", "limited"]
STAmplitudeSource: TypeAlias = Literal["analysis", "calibrated_pr", "adaptive_pr_tp"]

STANDARD_12_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
)


@dataclass(frozen=True, slots=True)
class StandardInput:
    mode: Literal["standard"] = "standard"

    def __post_init__(self) -> None:
        if self.mode != "standard":
            raise ConfigurationError("StandardInput mode must be standard")


@dataclass(frozen=True, slots=True)
class LimitedInput:
    channels: tuple[str, ...]
    mode: Literal["limited"] = "limited"

    def __post_init__(self) -> None:
        if self.mode != "limited" or isinstance(self.channels, (str, bytes)) or any(not isinstance(item, str) or "|" in item for item in self.channels):
            raise LeadNameError("limited channels must be explicit string names without '|'")
        channels = tuple(item.strip() for item in self.channels)
        if not 1 <= len(channels) <= 8:
            raise LeadNameError("limited input requires between 1 and 8 named channels", code="limited_lead_count")
        if any(not item for item in channels) or len(set(channels)) != len(channels):
            raise LeadNameError("limited input channels must be unique and non-empty", code="invalid_limited_leads")
        object.__setattr__(self, "channels", channels)


InputMode: TypeAlias = StandardInput | LimitedInput


@dataclass(frozen=True, slots=True)
class AnalysisST:
    source: Literal["analysis"] = "analysis"


@dataclass(frozen=True, slots=True)
class CalibratedPRST:
    source: Literal["calibrated_pr"] = "calibrated_pr"


@dataclass(frozen=True, slots=True)
class AdaptivePRTPST:
    source: Literal["adaptive_pr_tp"] = "adaptive_pr_tp"


STMeasurementConfig: TypeAlias = AnalysisST | CalibratedPRST | AdaptivePRTPST

# Use one value type across the public facade and legacy engine.
from .refinement import RefinementConfig


@dataclass(frozen=True, slots=True)
class PWaveConfig:
    enabled: bool = True
    validate_waveform: bool = True
    allow_residual_analysis: bool = True
    min_support_beats: int = 1

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.validate_waveform) is not bool or type(self.allow_residual_analysis) is not bool:
            raise ConfigurationError("P-wave boolean options must be boolean", code="invalid_p_wave_config")
        if type(self.min_support_beats) is not int or self.min_support_beats < 1:
            raise ConfigurationError("min_support_beats must be a positive integer", code="invalid_p_wave_config")


@dataclass(frozen=True, slots=True)
class PatientMeta:
    age: float | None = None
    age_days: float | None = None
    sex: str | None = None
    patient_id: str | None = None


@dataclass(frozen=True, slots=True)
class ECGConfig:
    input_mode: Literal["standard_12", "limited"] = "standard_12"
    fs_internal: float | None = None
    mains_frequency_hz: Literal[50, 60] | None = 50
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    p_wave: PWaveConfig = field(default_factory=PWaveConfig)
    st_amplitude_source: STAmplitudeSource = "analysis"

    def __post_init__(self) -> None:
        if self.input_mode not in {"standard_12", "limited", "standard"}:
            raise ConfigurationError("input_mode must be 'standard_12' or 'limited'", code="invalid_input_mode")
        if self.input_mode == "standard":
            object.__setattr__(self, "input_mode", "standard_12")
        if self.fs_internal is not None:
            try:
                value = float(self.fs_internal)
            except (TypeError, ValueError) as exc:
                raise SamplingRateError("fs_internal must be finite and positive", code="invalid_internal_sampling_rate") from exc
            if isinstance(self.fs_internal, bool) or value < 100 or not value.is_integer():
                raise SamplingRateError("fs_internal must be a positive integer sampling rate", code="invalid_internal_sampling_rate")
            object.__setattr__(self, "fs_internal", value)
        if self.mains_frequency_hz not in {None, 50, 60}:
            raise ConfigurationError("mains_frequency_hz must be 50, 60, or None", code="invalid_mains_frequency")
        if not isinstance(self.refinement, RefinementConfig):
            raise ConfigurationError("refinement must be a RefinementConfig")
        if not isinstance(self.p_wave, PWaveConfig):
            raise ConfigurationError("p_wave must be a record PWaveConfig")
        if self.p_wave != PWaveConfig():
            raise ConfigurationError("Non-default record P-wave controls are not wired to the legacy engine yet")
        if self.st_amplitude_source not in {"analysis", "calibrated_pr", "adaptive_pr_tp"}:
            raise ConfigurationError("unsupported ST amplitude source", code="invalid_st_amplitude_source")

    @property
    def input(self) -> InputMode | None:
        # ``ECGConfig`` intentionally keeps the actual channel names on the
        # call to ``ecg_prepare``.  A limited config therefore has no
        # standalone channel object until it is paired with an input signal.
        return StandardInput() if self.input_mode == "standard_12" else None

    def to_provenance_dict(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "fs_internal": self.fs_internal,
            "mains_frequency_hz": self.mains_frequency_hz,
            "st_amplitude_source": self.st_amplitude_source,
            "refinement": asdict(self.refinement),
            "p_wave": asdict(self.p_wave),
        }


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    fs_internal: float | None = None
    mains_freq: Literal[50, 60] | None = 50
    input: InputMode = field(default_factory=StandardInput)
    st: STMeasurementConfig = field(default_factory=AnalysisST)
    refinements: RefinementConfig = field(default_factory=RefinementConfig)
    p_wave: PWaveConfig = field(default_factory=PWaveConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.input, (StandardInput, LimitedInput)):
            raise ConfigurationError("input must be StandardInput or LimitedInput")
        if not isinstance(self.st, (AnalysisST, CalibratedPRST, AdaptivePRTPST)):
            raise ConfigurationError("st must be a supported ST configuration")
        expected = {AnalysisST: "analysis", CalibratedPRST: "calibrated_pr", AdaptivePRTPST: "adaptive_pr_tp"}[type(self.st)]
        if self.st.source != expected:
            raise ConfigurationError("ST configuration source does not match its type")
        self.to_ecg_config()

    @classmethod
    def standard(cls, **kwargs: Any) -> "ExtractionConfig":
        return cls(input=StandardInput(), **kwargs)

    @classmethod
    def limited(cls, channels: Sequence[str], **kwargs: Any) -> "ExtractionConfig":
        return cls(input=LimitedInput(tuple(channels)), **kwargs)

    def to_ecg_config(self) -> ECGConfig:
        return ECGConfig(
            input_mode="standard_12" if isinstance(self.input, StandardInput) else "limited",
            fs_internal=self.fs_internal,
            mains_frequency_hz=self.mains_freq,
            refinement=self.refinements,
            p_wave=self.p_wave,
            st_amplitude_source=self.st.source,
        )

    def to_provenance_dict(self) -> dict[str, Any]:
        return self.to_ecg_config().to_provenance_dict()


def config_provenance(config: ECGConfig, *, method: str = "default") -> dict[str, Any]:
    resolved = config.to_provenance_dict()
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {
        "config": resolved,
        "config_hash": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "method_requested": method,
        "method_resolved": "legacy_dxl_inspired" if method == "default" else method,
        "experimental": sorted(config.refinement.experimental_flags),
        "fallback_path": [],
    }


__all__ = [
    "InputModeName", "STAmplitudeSource", "StandardInput", "LimitedInput", "InputMode",
    "AnalysisST", "CalibratedPRST", "AdaptivePRTPST", "STMeasurementConfig",
    "RefinementConfig", "PWaveConfig", "PatientMeta", "ECGConfig", "ExtractionConfig",
    "STANDARD_12_LEADS", "config_provenance",
]
