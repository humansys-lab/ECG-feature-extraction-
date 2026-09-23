"""Immutable carriers shared by the staged extraction implementation.

Ownership-transfer rule (Phase 3 parity contract)
-------------------------------------------------
The staged pipeline must reproduce the pre-decomposition
``ECGFeatureExtractor.extract`` byte for byte
(docs/library_design/05_migration_plan.md, Phase 3).  The legacy engine works by
mutating its objects in place: later steps rewrite fields of per-beat
``LeadBeatFeatures``, of ``GlobalFeatures``, of representative-lead ``params`` and
of dictionaries created by earlier steps.  The pipeline keeps exactly those
semantics:

* :class:`PipelineContext` and every stage bundle are frozen dataclasses.  A stage
  never assigns to a field of a context or bundle it received; it returns a new
  context built with :func:`dataclasses.replace`.
* The engine objects *referenced* by a bundle are not copied.  When a stage hands
  its bundle to the next stage, ownership of the referenced engine objects
  transfers with it: the successor may mutate them in place exactly as the legacy
  code did, and the producer never touches them again.  A bundle from an earlier
  stage is therefore a view of engine state *as later stages leave it*, not a
  snapshot.
* Engine objects are never deep-copied to "purify" a stage.  Copies change
  identity-dependent behavior (several legacy steps rely on two names aliasing one
  list or dict, and on metadata holding the very object a later step mutates) and
  cost time; either would break parity.
* When legacy code *rebinds* a name that an earlier stage produced (for example the
  flutter retraction replacing ``af_afl_summary`` with an edited copy), the new
  binding travels in the later stage's bundle; the earlier bundle keeps the binding
  it produced, and consumers read the latest one.

Policy events and stage issues accumulate on the context in stage order.  During
Phase 3 they are neither written into the legacy ``ECGFeatures``/metadata (that
would change the frozen legacy bytes) nor into the ECG Record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, TypeAlias

import numpy as np

from ..record.availability import Availability

SignalDomain = Literal["native", "analysis", "detection", "pacing_cleaned"]
StageName = Literal["input", "quality", "ventricular", "beats", "delineation", "atrial", "measurement", "finalize"]
IssueSeverity = Literal["info", "warning", "recoverable", "fatal"]
DecisionKind = Literal["accept", "reject", "override", "rescue", "not_applicable", "no_change"]
JsonValue: TypeAlias = Any

STAGE_ORDER: tuple[StageName, ...] = (
    "input", "quality", "ventricular", "beats", "delineation", "atrial", "measurement", "finalize",
)


class StageUnavailableError(NotImplementedError):
    """A stage was invoked without the upstream state it consumes.

    Phase 3 stages run inside :func:`ecgfeat.pipeline.extractor.run_legacy_pipeline`,
    which threads each stage's bundle to its successors.  Running a stage on its own,
    from a bare target-architecture context, is not implemented; this error says so
    instead of returning fabricated success.
    """


@dataclass(frozen=True, slots=True)
class StageIssue:
    stage: StageName
    code: str
    severity: IssueSeverity
    message: str
    field_paths: tuple[str, ...] = ()


class FrozenDetails(dict):
    """Read-only JSON-like mapping for policy-event details.

    A ``dict`` subclass (not ``MappingProxyType``) so events stay picklable, deep-copyable
    and ``dataclasses.asdict``-able for the later record-provenance change.
    """

    __slots__ = ()

    def _read_only(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("policy event details are read-only")

    __setitem__ = __delitem__ = __ior__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only

    def __reduce__(self) -> Any:
        return (type(self), (dict(self),))

    def __copy__(self) -> "FrozenDetails":
        return self

    def __deepcopy__(self, memo: Any) -> "FrozenDetails":
        return self


@dataclass(frozen=True, slots=True)
class PolicyEvent:
    """Descriptive record of one policy decision (policy, kind, stable reason code)."""

    policy: str
    decision: str
    reason_code: str
    source_ids: tuple[str, ...] = ()
    details: Mapping[str, JsonValue] = field(default_factory=FrozenDetails)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The value a policy decided (exactly the legacy helper's result) plus its event."""

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
class ExtractionRequest:
    """One legacy ``extract`` call's arguments, untouched until the input stage runs.

    The raw signal is deliberately not converted here: input validation (and its
    ``ECGInputError`` codes) belongs to the input stage.
    """

    signal: Any
    fs: Any
    meta: Any = None
    lead_names: Any = None
    amplitude_unit: Any = None
    gain_uv_per_lsb: Any = None
    prior_features: Any = None


@dataclass(frozen=True, slots=True)
class ExtractorSettings:
    """Resolved legacy extractor configuration captured once per extraction."""

    fs_internal: Any
    mains_freq: Any
    lp_hz: float
    enable_pacing: bool
    enable_lead_reversal: bool
    compute_grouping: bool
    enable_hybrid_r_localization: bool
    enable_hybrid_wave_localization: bool
    enable_hybrid_st_measurement: bool
    enable_t_wave_refinement: bool
    st_amplitude_source: str
    refinement: Any
    input_mode: str

    @classmethod
    def from_extractor(cls, extractor: Any) -> "ExtractorSettings":
        if isinstance(extractor, cls):
            return extractor
        return cls(**{item.name: getattr(extractor, item.name) for item in fields(cls)})


class InterpretationHooks(Protocol):
    """Interpretation calls the legacy entry point makes at fixed extraction points.

    Stages never import interpretation code.  The compatibility layer injects an
    implementation (``ecgfeat.compat.interpretation_hooks``); when the context carries
    no hooks the corresponding legacy outputs are simply not produced.
    """

    def rhythm_statement_evidence(self, **evidence: Any) -> Any: ...

    def interpret(self, features: Any) -> Any: ...

    def clinical_interpretation(self, features: Any, *, prior_features: Any = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class PipelineContext:
    record_id: str | None = None
    config: Any = None
    patient: Any = None
    native: SignalView | None = None
    signals: Mapping[SignalDomain, SignalView] = field(default_factory=dict)
    request: ExtractionRequest | None = None
    hooks: InterpretationHooks | None = None
    input: Any = None
    quality: Any = None
    ventricular: Any = None
    beats: Any = None
    delineation: Any = None
    atrial: Any = None
    measurements: Any = None
    legacy_features: Any = None
    field_states: Mapping[str, Availability[Any]] = field(default_factory=dict)
    policy_events: tuple[PolicyEvent, ...] = ()
    issues: tuple[StageIssue, ...] = ()

    def __post_init__(self) -> None:
        signals = dict(self.signals)
        if self.native is not None:
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


def with_policy_events(context: PipelineContext, events: Any) -> PipelineContext:
    events = tuple(events)
    return replace(context, policy_events=context.policy_events + events) if events else context


def with_issue(context: PipelineContext, issue: StageIssue) -> PipelineContext:
    return replace(context, issues=context.issues + (issue,))


def record_decision(events: list[PolicyEvent], decision: PolicyDecision) -> Any:
    """Append ``decision.event`` to a stage's event list and return the decided value.

    Stages use this in place of the former direct helper call, so the value flows
    into exactly the expression the legacy code used (including short-circuiting).
    """
    events.append(decision.event)
    return decision.value


def require(context: Any, stage: str, *names: str) -> PipelineContext:
    """Return ``context`` if it carries every named upstream field, else raise."""
    if not isinstance(context, PipelineContext):
        raise StageUnavailableError(
            f"the {stage} stage needs a PipelineContext produced by the preceding stages "
            f"(got {type(context).__name__}); stand-alone stage execution is not implemented"
        )
    missing = [name for name in names if getattr(context, name) is None]
    if missing:
        raise StageUnavailableError(
            f"the {stage} stage needs upstream state {missing}; run it through "
            "ecgfeat.pipeline.extractor.run_legacy_pipeline"
        )
    return context


__all__ = [
    "SignalDomain", "StageName", "IssueSeverity", "DecisionKind", "STAGE_ORDER", "StageUnavailableError",
    "StageIssue", "StageResult", "FrozenDetails", "PolicyEvent", "PolicyDecision", "SignalView", "ExtractionRequest",
    "ExtractorSettings", "InterpretationHooks", "PipelineContext", "with_policy_event",
    "with_policy_events", "with_issue", "record_decision", "require",
]
