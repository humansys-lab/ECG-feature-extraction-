"""Measurement-engine thresholds (signal quality, rhythm minimums, morphology).

These values historically lived in the diagnostic rule configuration
(``clinical_rules.config.DEFAULT_DIAGNOSTIC_CONFIG``) and were read by the
engine.  The engine owns its copy now that interpretation ships separately;
a repository integration test asserts the shared defaults stay equal.  They
select measurement quality and availability, never a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SignalQualityThresholds:
    flatline_peak_to_peak_mv: float = 0.05
    flatline_fraction_max: float = 0.10
    saturation_fraction_max: float = 0.005
    baseline_drift_ratio_max: float = 0.30
    powerline_ratio_max: float = 0.15
    emg_ratio_max: float = 0.40
    psqi_min: float = 0.40
    psqi_max: float = 0.90
    ksqi_min: float = 5.0
    bsqi_min: float = 0.85
    bsqi_record_stop: float = 0.70
    minimum_usable_leads: int = 6
    full_coverage_leads: int = 10


@dataclass(frozen=True)
class RhythmMinimums:
    minimum_rhythm_duration_seconds: float = 8.0
    minimum_diagnostic_duration_seconds: float = 10.0
    minimum_detected_beats: int = 3
    organized_p_ratio_min_for_morphology: float = 0.35


@dataclass(frozen=True)
class MorphologyLimits:
    dominant_group_fraction_min: float = 0.30
    maximum_template_classes: int = 5


@dataclass(frozen=True)
class EngineThresholds:
    quality: SignalQualityThresholds = field(default_factory=SignalQualityThresholds)
    rhythm: RhythmMinimums = field(default_factory=RhythmMinimums)
    morphology: MorphologyLimits = field(default_factory=MorphologyLimits)


ENGINE_THRESHOLDS = EngineThresholds()

__all__ = ["ENGINE_THRESHOLDS", "EngineThresholds", "SignalQualityThresholds", "RhythmMinimums", "MorphologyLimits"]
