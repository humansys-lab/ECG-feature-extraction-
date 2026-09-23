"""Immutable carriers shared by the staged extraction implementation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np

from ..config import ECGConfig, ExtractionConfig
from ..record.availability import Availability

SignalDomain = Literal["native", "analysis", "detection", "pacing_cleaned"]
StageName = Literal["input", "quality", "ventricular", "beats", "delineation", "atrial", "measurement", "finalize"]
IssueSeverity = Literal["info", "warning", "recoverable", "fatal"]
JsonValue: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class StageIssue:
    stage: StageName
    code: str
    severity: IssueSeverity
    message: str
    field_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyEvent:
    policy: str
    decision: str
    reason_code: str
    source_ids: tuple[str, ...] = ()
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    value: Any
    event: PolicyEvent


@dataclass(frozen=True, slots=True)
class SignalView:
    samples: np.ndarray
    fs_hz: float
    lead_names: tuple[str, ...]
    domain: SignalDomain

    def __post_init__(self) -> None:
        source = np.asarray(self.samples, dtype=np.float64)
        array = np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)
        object.__setattr__(self, "samples", array)
        object.__setattr__(self, "lead_names", tuple(self.lead_names))


@dataclass(frozen=True, slots=True)
class PipelineContext:
    record_id: str
    config: ECGConfig | ExtractionConfig
    patient: Any
    native: SignalView
    signals: Mapping[SignalDomain, SignalView] = field(default_factory=dict)
    quality: Any = None
    ventricular: Any = None
    beats: Any = None
    delineation: Any = None
    atrial: Any = None
    measurements: Any = None
    field_states: Mapping[str, Availability[Any]] = field(default_factory=dict)
    policy_events: tuple[PolicyEvent, ...] = ()
    issues: tuple[StageIssue, ...] = ()

    def __post_init__(self) -> None:
        signals = dict(self.signals)
        signals.setdefault(self.native.domain, self.native)
        object.__setattr__(self, "signals", MappingProxyType(signals))
        object.__setattr__(self, "field_states", MappingProxyType(dict(self.field_states)))
        object.__setattr__(self, "policy_events", tuple(self.policy_events))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class StageResult:
    context: PipelineContext
    issues: tuple[StageIssue, ...] = ()


def with_policy_event(context: PipelineContext, event: PolicyEvent) -> PipelineContext:
    return replace(context, policy_events=context.policy_events + (event,))


def with_issue(context: PipelineContext, issue: StageIssue) -> PipelineContext:
    return replace(context, issues=context.issues + (issue,))


__all__ = [
    "SignalDomain", "StageName", "IssueSeverity", "StageIssue", "StageResult",
    "PolicyEvent", "PolicyDecision", "SignalView", "PipelineContext",
    "with_policy_event", "with_issue",
]
