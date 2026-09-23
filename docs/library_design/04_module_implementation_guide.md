# Module Implementation Guide

> Implementation status: this document includes target contracts, not completed
> release claims. [Document 08](08_implementation_status.md) records the implemented
> subset, reconciled decisions and outstanding gates. Document 02 is authoritative
> for the JSON layout; Python carrier sketches must not create a second wire format.

This document is the implementation contract for decomposing the current `ecgfeat`
package into a versioned ECG Record library, a staged measurement pipeline, private
measurement engines, compatibility adapters, and separately distributed interpretation
and visualization packages.

The published consumer boundary is the pair **(raw signal, ECG Record)**. The ECG Record
is a schema-versioned JSON document. Interpretation is a separate, separately versioned
document produced by the `ecginterpret` namespace. Core runtime dependencies remain
NumPy and SciPy; Numba is optional. Plotting and interpretation dependencies must not be
pulled into the core distribution.

This guide uses the prepared module inventory and callable digest as its source of truth.
The digest lists current callable names but not their parameter lists. Accordingly, every
signature below is a **target signature for the new module contract**, not a claim that
the old callable already has that exact signature. Migration code must adapt old call
shapes to these contracts explicitly; it must not preserve accidental signatures merely
because they exist today.

## Conventions used in signatures

The following names are shared notation for the signatures in this guide. They do not
require a new shared runtime module unless implementation proves that worthwhile.

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int64]
JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
LeadName: TypeAlias = str
SampleIndex: TypeAlias = int
Milliseconds: TypeAlias = float
Millivolts: TypeAlias = float
SchemaVersion: TypeAlias = str
RecordId: TypeAlias = str
```

Unless a signature says otherwise, arrays are channel-major `(n_channels, n_samples)`,
sample indices are zero-based indices into the stage's stated signal, lead ordering is
stable and follows the validated input channel order, and all serialized numbers must be
finite JSON numbers. NaN and infinities are internal computation values only; they become
an explicit absence state or are rejected before record construction.

Times in the published record are milliseconds unless the field name explicitly declares
another unit. Voltage measurements are millivolts unless explicitly named otherwise.
Public serializers and queries are deterministic for the same record value.

## Layer and dependency rules

The dependency direction is intentionally one-way:

```text
ecgfeat facade
    -> record/*, config, io, pipeline public entry point
pipeline/*
    -> record/*, config, _engine/*
record/*
    -> stdlib, NumPy only where sidecar arrays require it
_engine/*
    -> _engine/foundation/* and lower/sibling engine primitives
compat/*
    -> public core modules plus narrowly required private adapters
ecgfeat_interpret/*
    -> ECG Record public API; no core private engine imports
ecgfeat_viz/*
    -> ECG Record public API + raw signal; no core private engine imports
```

`record/*` must never import `pipeline/*`, `_engine/*`, compatibility modules,
interpretation modules, or visualization modules. Private engine code must not construct
serialized record dictionaries. Pipeline stages may construct internal context values;
only the finalize stage creates the published record.

A module may import standard-library modules freely. Core modules may import NumPy and
SciPy. Only `_engine/foundation/kalman.py` may opportunistically import Numba. No core
module may import matplotlib, a rule-engine package, or interpretation code.

# Phase 1 — record model, configuration, I/O, and public facades

## `ecgfeat/config.py`

### Responsibility

Own the stable, validated extraction configuration exposed to consumers. It owns input
mode, sampling/preprocessing choices, measurement-source choices, explicit refinements,
and P-wave configuration; it does **not** own stage execution, measurement results, or
diagnostic interpretation.

The new API must make invalid ST-source combinations unconstructible. In particular,
`st_amplitude_source="analysis"` is the default. The non-default
`"calibrated_pr"` and `"adaptive_pr_tp"` sources imply the hybrid ST measurement path;
the public configuration therefore does not expose a contradictory pair of
`st_amplitude_source` and `enable_hybrid_st_measurement=False`. A legacy contradictory
pair is rejected by the compatibility adapter with an error message that names the
actual condition, rather than reproducing the current message/condition mismatch.

### Public surface

```python
InputModeName: TypeAlias = Literal["standard", "limited"]
STAmplitudeSource: TypeAlias = Literal[
    "analysis", "calibrated_pr", "adaptive_pr_tp"
]

@dataclass(frozen=True, slots=True)
class StandardInput:
    mode: Literal["standard"] = "standard"

@dataclass(frozen=True, slots=True)
class LimitedInput:
    channels: tuple[LeadName, ...]
    mode: Literal["limited"] = "limited"

    def __post_init__(self) -> None: ...

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

from ecgfeat.refinement import RefinementConfig
# Preserve the real 18-field configuration, with all flags False by default.
# Examples: p_boundary_correction, t_boundary_projection, qrs_adaptive_consensus.
# enabled is an aggregate bool property; experimental_flags lists active names.
# Do not alias invented names to unrelated numerical paths.

@dataclass(frozen=True, slots=True)
class PWaveConfig:
    enabled: bool = True
    validate_waveform: bool = True
    allow_residual_analysis: bool = True
    min_support_beats: int = 1

@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    fs_internal: float | None = None
    mains_freq: Literal[50, 60] | None = 50
    input: InputMode = StandardInput()
    st: STMeasurementConfig = AnalysisST()
    refinements: RefinementConfig = RefinementConfig()
    p_wave: PWaveConfig = PWaveConfig()

    def __post_init__(self) -> None: ...

    @classmethod
    def standard(
        cls,
        *,
        fs_internal: float | None = None,
        mains_freq: Literal[50, 60] | None = 50,
        st: STMeasurementConfig = AnalysisST(),
        refinements: RefinementConfig = RefinementConfig(),
        p_wave: PWaveConfig = PWaveConfig(),
    ) -> "ExtractionConfig": ...

    @classmethod
    def limited(
        cls,
        channels: Sequence[LeadName],
        *,
        fs_internal: float | None = None,
        mains_freq: Literal[50, 60] | None = 50,
        st: STMeasurementConfig = AnalysisST(),
        refinements: RefinementConfig = RefinementConfig(),
        p_wave: PWaveConfig = PWaveConfig(),
    ) -> "ExtractionConfig": ...

    def to_provenance_dict(self) -> dict[str, JsonValue]: ...
```

### What it absorbs

- `refinement.py:RefinementConfig`, including
  `refinement.py:RefinementConfig.experimental` and
  `refinement.py:RefinementConfig.enabled`.
- `p_wave_engine.py:PWaveConfig`; the P-wave algorithms themselves move to
  `_engine/atrial/p_wave.py`.
- The stable extraction options currently constructed and validated around
  `api.py:ECGFeatureExtractor.extract`, specifically the verified
  `fs_internal`, `mains_freq`, `input_mode`, `st_amplitude_source`, and refinement
  choices. `api.py:_resolve_mains_frequency` does **not** move here; it belongs to the
  input stage because it depends on input metadata.

### Invariants it must preserve

- `fs_internal=None` means use the native input sampling frequency; no implicit fixed
  resampling rate is introduced.
- `mains_freq` defaults to 50 Hz. `None` means defer resolution to input metadata or
  input-stage policy; it must not silently become 60 Hz.
- Standard input remains the default mode.
- Limited input contains 1–8 **explicitly named**, synchronously sampled channels. It is
  invalid to construct a limited configuration with zero channels, more than eight
  channels, duplicate names, or unnamed placeholders.
- Limited mode must never imply pre-padding to twelve rows.
- Limited mode disables diagnosis, formal axis reporting, and formal QT reporting through
  applicability policy; configuration itself records the mode but does not fabricate
  unavailable measurements.
- The default ST amplitude source is `analysis`. Both non-default sources activate the
  hybrid ST measurement route by construction.
- Refinements remain opt-in. No migration is allowed to turn an experimental boundary
  refinement on merely because its code moved.
- Configuration values included in provenance are canonical and deterministic.

### Dependencies

May import only standard-library typing/dataclass support. It must not import pipeline,
engine, record model, interpretation, or compatibility code. Keeping configuration at
the bottom of the dependency graph prevents stage/config cycles.

---

## `ecgfeat/io.py`

### Responsibility

Provide convenience adapters for WFDB-style headers and MAT signal files. It returns raw
signal and input metadata suitable for the public extractor; it does **not** perform
clinical validation, signal processing, feature extraction, or record serialization.

### Public surface

```python
@dataclass(frozen=True, slots=True)
class WFDBHeader:
    fs_hz: float
    channel_names: tuple[LeadName, ...]
    units: tuple[str, ...]
    gains: tuple[float | None, ...]
    baseline: tuple[float | None, ...]
    metadata: Mapping[str, JsonValue]

def parse_wfdb_header(
    source: str | Path,
    *,
    text: bool = False,
) -> WFDBHeader: ...

def load_wfdb_mat(
    path: str | Path,
    *,
    key: str = "val",
    dtype: np.dtype[Any] | type[np.float64] = np.float64,
) -> FloatArray: ...
```

The current digest does not expose the old parameter lists. These signatures deliberately
normalize the new adapter around an explicit header value object and a channel-major
floating array.

### What it absorbs

- `io.py:parse_wfdb_header`.
- `io.py:load_wfdb_mat`.

No other current module belongs here.

### Invariants it must preserve

- MAT loading preserves channel order and sample order.
- Header parsing does not invent channel names, gains, units, or sampling frequency.
- Raw adapter output is not pre-padded or interpreted as a twelve-lead matrix.
- The adapter does not resample.
- File parsing failures are I/O/parsing failures, while physiological input-contract
  failures are raised later by `pipeline/stages/input.py`.

### Dependencies

May import standard library, NumPy, and SciPy I/O functionality. It may import
`config.py` only for shared lead-name typing if that becomes a runtime type; preferably
it remains independent. It must not import pipeline stages or private engines.

---

## `ecgfeat/record/availability.py`

### Responsibility

Represent the three semantically distinct field states: measured,
unavailable-with-reason, and not-applicable-with-reason. It does **not** decide which
state a measurement should receive; pipeline applicability/finalization does that.

### Public surface

```python
T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class Measured(Generic[T]):
    value: T
    state: Literal["measured"] = "measured"

@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: str
    detail: str | None = None
    state: Literal["unavailable"] = "unavailable"

@dataclass(frozen=True, slots=True)
class NotApplicable:
    reason: str
    detail: str | None = None
    state: Literal["not_applicable"] = "not_applicable"

Availability: TypeAlias = Measured[T] | Unavailable | NotApplicable

def measured(value: T) -> Measured[T]: ...

def unavailable(reason: str, *, detail: str | None = None) -> Unavailable: ...

def not_applicable(reason: str, *, detail: str | None = None) -> NotApplicable: ...

def is_measured(value: Availability[T]) -> TypeGuard[Measured[T]]: ...
```

### What it absorbs

This is a new published value model. It replaces the ambiguous use of missing keys,
`None`, NaN, and ad-hoc availability maps spread across the current object/export path.
Relevant legacy behavior comes from
`rhythm_rules.py:build_measurement_availability`,
`rhythm_rules.py:detect_av_block_availability_flags`, and the current exported payloads,
but those functions themselves move to pipeline policy/finalization rather than here.

### Invariants it must preserve

- The three states remain distinguishable after JSON round-trip.
- `Measured(None)` is not permitted for a measurement field; semantic absence uses one of
  the two explicit absence variants.
- Reasons are stable machine-readable codes; human detail is optional and non-normative.
- Availability is a property of a published field, not a diagnostic conclusion.
- Equality and serialization are deterministic.

### Dependencies

Standard library only. This module must not import any stage, policy, engine, or
interpretation code.

---

## `ecgfeat/record/provenance.py`

### Responsibility

Own reproducibility metadata: library/schema version, resolved configuration, input
fingerprint, algorithm fingerprint, and recorded policy decisions. It does **not**
serialize full internal stage state or claim that a fingerprint is clinical validation.

### Public surface

```python
@dataclass(frozen=True, slots=True)
class InputFingerprint:
    sha256: str
    n_channels: int
    n_samples: int
    fs_hz: float
    channels: tuple[LeadName, ...]

@dataclass(frozen=True, slots=True)
class AlgorithmFingerprint:
    name: str
    version: str
    parameters_sha256: str | None = None

@dataclass(frozen=True, slots=True)
class PolicyDecisionRecord:
    policy: str
    decision: str
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    affected_fields: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class Provenance:
    library_version: str
    schema_version: SchemaVersion
    resolved_config: Mapping[str, JsonValue]
    input: InputFingerprint
    algorithms: tuple[AlgorithmFingerprint, ...]
    policy_decisions: tuple[PolicyDecisionRecord, ...]
    created_by: str = "ecgfeat"

def fingerprint_input(
    signal: FloatArray,
    *,
    fs_hz: float,
    channels: Sequence[LeadName],
) -> InputFingerprint: ...

def clinical_fingerprint(
    values: Mapping[str, JsonValue],
    *,
    algorithm: str,
    version: str,
) -> str: ...

def build_provenance(
    *,
    library_version: str,
    schema_version: SchemaVersion,
    resolved_config: Mapping[str, JsonValue],
    input_fingerprint: InputFingerprint,
    algorithms: Sequence[AlgorithmFingerprint],
    policy_decisions: Sequence[PolicyDecisionRecord],
) -> Provenance: ...
```

### What it absorbs

- `export.py:clinical_fingerprint`.
- Fingerprint/provenance behavior currently embedded in `export.py` private helpers, where
  identifiable by implementation during migration.
- Resolved refinement/config flags currently scattered through the extraction/export path
  are copied in through `config.py:ExtractionConfig.to_provenance_dict`; the provenance
  module does not resolve them itself.
- Named pacing/QT and other policy decisions arrive from `pipeline/policies/*`; they are
  recorded here but not decided here.

### Invariants it must preserve

- Fingerprints are deterministic for identical normalized inputs.
- Input hashing includes the actual channel order, sample count, sampling frequency, and
  signal bytes in a canonical dtype/byte-order convention chosen once and versioned.
- Provenance contains enough information to distinguish native-fs from resampled
  extraction and to identify the selected ST amplitude source.
- Policy records contain reason codes and affected record pointers/fields rather than
  serializing arbitrary Python objects.
- A calibration score from `_engine/measurement/calibration.py` is never represented as
  an externally validated clinical probability.

### Dependencies

May import `record/availability.py` only if needed for canonical normalization, though a
standalone canonicalizer is preferable. May import NumPy for signal fingerprinting.
Must not import pipeline or engine modules.

---

## `ecgfeat/record/validation.py`

### Responsibility

Represent **publication category, independent validation status, and evidence**.
`published_measurement` is a placement category, not a claim of benchmark accuracy.
Wire validation status uses document 02's vocabulary; new fields remain
`unvalidated` until endpoint-specific evidence is available.
It does **not** validate raw ECG inputs; raw input validation belongs to
`pipeline/stages/input.py`.

### Public surface

```python
ValidationTier: TypeAlias = Literal[
    "published_measurement",
    "provenance",
    "internal_debug",
]

@dataclass(frozen=True, slots=True)
class EvidenceReference:
    ref: str
    kind: str
    description: str | None = None

@dataclass(frozen=True, slots=True)
class FieldValidation:
    tier: ValidationTier
    evidence: tuple[EvidenceReference, ...] = ()
    notes: tuple[str, ...] = ()
    status: Literal["benchmark_validated", "indirectly_validated", "unvalidated"] = "unvalidated"

def validation_for(
    *,
    tier: ValidationTier,
    evidence: Sequence[EvidenceReference] = (),
    notes: Sequence[str] = (),
) -> FieldValidation: ...

def validate_publishable_field(
    pointer: str,
    value: Availability[JsonValue],
    validation: FieldValidation,
) -> None: ...
```

### What it absorbs

No current `validation.py` callable moves here. The name is intentionally about record
field validation, while current `validation.py:ECGInputError` and
`validation.py:validate_ecg_input` move to the pipeline input stage.

The field-tier decisions replace implicit export-time assumptions. In particular, the
22 `st_hybrid_*` keys and 16 `twelve_sl_*` keys are internal debug/compatibility
extension data, never default published measurements.

### Invariants it must preserve

- Field tier is assigned by meaning and stability, not by the current Python object that
  happens to contain the value.
- A field marked `internal_debug` cannot silently appear in the default `summary`
  measurement profile.
- Evidence references are stable identifiers/pointers, not live object references.
- Validation metadata round-trips without changing the field's availability state.

### Dependencies

May import `record/availability.py`. It must not import pipeline, engine, compatibility,
or interpretation code.

---

## `ecgfeat/record/model.py`

### Responsibility

Define the immutable schema-versioned ECG Record value model. It owns the structural
separation of measurements, provenance, validation metadata, subject/input metadata, and
optional sidecar references; it does **not** run extraction, perform interpretation, or
preserve the legacy `ECGFeatures` object graph.

### Public surface

```python
@dataclass(frozen=True, slots=True)
class RecordSidecar:
    media_type: str
    href: str
    sha256: str
    role: str

@dataclass(frozen=True, slots=True)
class ECGRecord:
    record_id: RecordId
    schema_version: SchemaVersion
    measurements: Mapping[str, Availability[JsonValue]]
    provenance: Provenance
    validation: Mapping[str, FieldValidation]
    metadata: Mapping[str, JsonValue]
    sidecars: tuple[RecordSidecar, ...] = ()

    def field(self, pointer: str) -> Availability[JsonValue]: ...

def build_record(
    *,
    record_id: RecordId,
    schema_version: SchemaVersion,
    measurements: Mapping[str, Availability[JsonValue]],
    provenance: Provenance,
    validation: Mapping[str, FieldValidation],
    metadata: Mapping[str, JsonValue] | None = None,
    sidecars: Sequence[RecordSidecar] = (),
) -> ECGRecord: ...
```

### What it absorbs

No legacy dataclass moves wholesale. The record model is deliberately independent of
`models.py:ECGFeatures`, `models.py:GlobalFeatures`,
`models.py:RepresentativeLeadFeatures`, and the other internal measurement carriers.
Selected stable values produced by those carriers are copied into record fields during
finalization.

Legacy `models.py:ECGInterpretation` moves out with interpretation. Legacy aliases and
adapters remain in `compat/models_v0.py`.

### Invariants it must preserve

- `record_id` and `schema_version` are mandatory.
- Raw signal is not embedded in the JSON record. The published consumer boundary is raw
  signal **plus** the record.
- Every published measurement has an explicit availability state.
- Internal debug fields cannot masquerade as default measurements.
- Mapping iteration order must not determine semantic equality or fingerprints.
- The model is immutable after construction.
- Construction rejects non-finite JSON numbers, invalid JSON pointers in validation keys,
  and provenance/schema-version mismatches.
- Sidecar references are optional. Dense measurement matrices belong in an optional NPZ
  sidecar, whose bytes count separately from the JSON size envelope.

### Dependencies

May import `record/availability.py`, `record/provenance.py`, and
`record/validation.py`. It must not import serialization to avoid a model/serializer
cycle.

---

## `ecgfeat/record/serialize.py`

### Responsibility

Encode and decode ECG Records deterministically under bounded publication profiles. It
owns summary/all-measurements projection and optional dense NPZ sidecar encoding; it does
**not** decide measurement applicability, perform extraction, or recreate legacy export
payloads.

### Public surface

```python
RecordProfile: TypeAlias = Literal["summary", "all", "debug"]

@dataclass(frozen=True, slots=True)
class SerializedRecord:
    json_bytes: bytes
    sidecars: Mapping[str, bytes]

def to_json_obj(
    record: ECGRecord,
    *,
    profile: RecordProfile | None = None,
) -> dict[str, JsonValue]: ...

def from_json_obj(value: Mapping[str, JsonValue]) -> ECGRecord: ...

def dumps_record(
    record: ECGRecord,
    *,
    profile: RecordProfile | None = None,
    indent: int | None = None,
) -> bytes: ...

def loads_record(data: bytes | str) -> ECGRecord: ...

def serialize_record(
    record: ECGRecord,
    *,
    profile: RecordProfile | None = None,
    dense_arrays: Mapping[str, FloatArray] | None = None,
) -> SerializedRecord: ...

def encode_npz_sidecar(
    arrays: Mapping[str, FloatArray],
    *,
    compressed: bool = True,
) -> bytes: ...
```

### What it absorbs

- The new bounded serializer replaces the consumer-facing role of `export.py:to_dict`,
  `export.py:prepare_json_export`, and `export.py:build_structured_payload` for the new
  ECG Record.
- The exact legacy formats of those functions remain in `compat/export_v0.py`.
- Rule payload builders `export.py:build_morphology_inputs`,
  `export.py:build_rhythm_inputs`, and
  `export.py:build_statement_engine_payload` do **not** move here; they move with the
  interpretation compatibility/adapter path.

### Invariants it must preserve

- Default profile is `summary`.
- The default summary profile targets 24,000 bytes for a 10 s 12-lead record; the settled
  envelope has 22,295 bytes used. The implementation must keep a regression budget around
  this envelope rather than treating 24,000 as permission to grow without review.
- The all-measurements JSON reference size is about 63,030 bytes.
- Dense matrices are excluded from JSON by default and use an optional NPZ sidecar
  (reference raw size about 24,240 bytes), counted separately.
- `st_hybrid_*` and `twelve_sl_*` internal/debug fields never enter `summary`.
- Encoding is deterministic: canonical key ordering, stable float normalization, and no
  dependence on hash iteration order.
- Decoding preserves all three absence states.
- Unknown future fields are either retained in an explicit extension area or rejected
  according to schema-version rules; they must not be silently reinterpreted.
- Serialization never converts NaN/Inf to non-standard JSON tokens.

### Dependencies

May import all `record/*` model modules and NumPy for sidecars. Must not import pipeline,
private engines, compatibility exporters, or interpretation code.

---

## `ecgfeat/record/query.py`

### Responsibility

Provide stable addressing and lookup over decoded records using the published
`ecg-record:<record_id>@<schema_version>#<RFC 6901 JSON pointer>` form. It does **not**
evaluate clinical rules or reach into private engine objects.

### Public surface

```python
@dataclass(frozen=True, slots=True)
class RecordAddress:
    record_id: RecordId
    schema_version: SchemaVersion
    pointer: str

    def __str__(self) -> str: ...

def parse_record_address(value: str) -> RecordAddress: ...

def make_record_address(
    record: ECGRecord,
    pointer: str,
) -> RecordAddress: ...

def resolve_pointer(
    record: ECGRecord,
    pointer: str,
) -> JsonValue | Availability[JsonValue]: ...

def resolve_address(
    record: ECGRecord,
    address: str | RecordAddress,
) -> JsonValue | Availability[JsonValue]: ...

def has_pointer(record: ECGRecord, pointer: str) -> bool: ...
```

### What it absorbs

This is a new stable consumer API motivated by current pointer-based access in consumers.
No current module moves wholesale. It replaces ad-hoc dictionary walking over values
formerly emitted by `export.py:to_dict`.

### Invariants it must preserve

- Address syntax is exactly
  `ecg-record:<record_id>@<schema_version>#<RFC 6901 JSON pointer>`.
- Pointer escaping follows RFC 6901, including `~0` and `~1`.
- `resolve_address` verifies both record ID and schema version before dereferencing.
- Array indices are zero-based JSON Pointer indices.
- Missing pointer, mismatched record ID, and mismatched schema version are distinct errors.
- Querying does not mutate the record and is deterministic.
- The API exposes explicit availability objects rather than collapsing absence to `None`.

### Dependencies

May import `record/model.py` and `record/availability.py`. It must not import
serialization except for shared pure JSON-Pointer helpers if those are factored into a
lower-level utility.

---

## `ecgfeat/record/__init__.py`

### Responsibility

Expose the supported ECG Record namespace. It is a lean facade only; it does **not**
contain implementation logic.

### Public surface

```python
from .availability import (
    Availability,
    Measured,
    NotApplicable,
    Unavailable,
    measured,
    not_applicable,
    unavailable,
)
from .model import ECGRecord, RecordSidecar, build_record
from .provenance import (
    AlgorithmFingerprint,
    InputFingerprint,
    PolicyDecisionRecord,
    Provenance,
)
from .query import RecordAddress, make_record_address, parse_record_address, resolve_address
from .serialize import (
    RecordProfile,
    SerializedRecord,
    dumps_record,
    loads_record,
    serialize_record,
)
from .validation import EvidenceReference, FieldValidation, ValidationTier

__all__: tuple[str, ...]
```

### What it absorbs

No algorithmic code. It replaces the record-related portion of the broad current root
`__init__.py` re-export behavior.

### Invariants it must preserve

- Importing `ecgfeat.record` must not import SciPy-heavy engines, Numba, plotting, or
  interpretation.
- `__all__` contains only supported consumer names.
- Private implementation types are not re-exported accidentally.

### Dependencies

May import only sibling `record/*` modules.

---

## `ecgfeat/pipeline/__init__.py`

### Responsibility

Expose the supported extraction entry point while keeping stage/context/policy machinery
private to the core implementation. It does **not** re-export individual stages.

### Public surface

```python
from .extractor import extract_record

__all__ = ("extract_record",)
```

### What it absorbs

The supported extraction entry point replaces the consumer-facing role of
`api.py:ECGFeatureExtractor.extract`. The class-shaped legacy API is preserved only by
`compat/api_v0.py`.

### Invariants it must preserve

- Importing `ecgfeat.pipeline` has no plotting or interpretation side effects.
- Consumers can extract an ECG Record without importing compatibility modules.
- The entry point returns `ECGRecord`, never a mixed legacy `ECGFeatures` object.

### Dependencies

May import only `pipeline/extractor.py`.

---

## `ecgfeat/__init__.py`

### Responsibility

Be the lean public facade for record extraction, configuration, record values/querying,
and convenience I/O. During migration it may expose **deprecated forwarding names** for
legacy imports, but it does **not** eagerly import interpretation, plotting, or the
private engine.

### Public surface

```python
from .config import (
    AdaptivePRTPST,
    AnalysisST,
    CalibratedPRST,
    ExtractionConfig,
    LimitedInput,
    PWaveConfig,
    RefinementConfig,
    StandardInput,
)
from .io import WFDBHeader, load_wfdb_mat, parse_wfdb_header
from .pipeline import extract_record
from .record import (
    ECGRecord,
    Measured,
    NotApplicable,
    RecordAddress,
    Unavailable,
    dumps_record,
    loads_record,
    make_record_address,
    parse_record_address,
    resolve_address,
)

__all__: tuple[str, ...]

def __getattr__(name: str) -> object: ...
```

`__getattr__` is the migration hook for deprecated forwarding exports such as
`ECGFeatureExtractor`, old model names, and old export helpers. It must import the
relevant `ecgfeat.compat` module lazily and emit a deprecation warning.

### What it absorbs

- Current `__init__.py` facade responsibilities, drastically reduced.
- It retains record/extraction/I/O/config exports.
- Existing interpretation, export, plotting, and model re-exports become lazy deprecated
  forwarding exports to their new owners; they are not eager dependencies.

### Invariants it must preserve

- `import ecgfeat` remains cheap and works without plotting or interpretation packages.
- The import name remains `ecgfeat` during migration; recommended core distribution name
  is `ecg-records`.
- Deprecated compatibility access is lazy, emits a stable warning, and does not change the
  new default extraction return type.
- No import-time signal processing, model fitting, or environment-dependent discovery.

### Dependencies

May import public `config.py`, `io.py`, `record/__init__.py`, and
`pipeline/__init__.py`. Compatibility modules are accessed only from `__getattr__`.
It must never import `_engine/*` directly.

---

## Phase 1 implementation order

Implement this phase in the following order because each step keeps dependencies acyclic:

1. `record/availability.py` and `record/validation.py`.
2. `record/provenance.py`.
3. `record/model.py`.
4. `record/query.py` and `record/serialize.py`.
5. `record/__init__.py`.
6. `config.py`.
7. `io.py`.
8. `pipeline/__init__.py` only after Phase 2 provides `extractor.py`.
9. Root `ecgfeat/__init__.py` after the new public imports exist.

The main circular-dependency risk in Phase 1 is model/serialization. Keep serialization
one-way (`serialize -> model`) and never add `ECGRecord.to_json()` methods that import
the serializer. A second risk is configuration importing pipeline types; avoid it by
keeping config as pure validated values.


# Phase 2 — pipeline orchestration, stages, and policies

Phase 2 replaces the orchestration half of `api.py` with a small pipeline whose state
transitions are explicit. The pipeline owns sequencing and policy decisions; numerical
signal-processing implementations remain in `_engine/*`. The only stage allowed to
construct an `ECGRecord` is `finalize`.

## Shared pipeline stage contract

All eight stage modules implement the same internal contract. The contract is private to
the extraction implementation and is not a supported consumer import.

```python
from dataclasses import dataclass, replace
from typing import Generic, Literal, Protocol, TypeVar

T = TypeVar("T")

StageName = Literal[
    "input",
    "quality",
    "ventricular",
    "beats",
    "delineation",
    "atrial",
    "measurement",
    "finalize",
]
IssueSeverity = Literal["info", "warning", "recoverable", "fatal"]


@dataclass(frozen=True, slots=True)
class StageIssue:
    stage: StageName
    code: str
    severity: IssueSeverity
    message: str
    field_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageResult:
    context: PipelineContext
    issues: tuple[StageIssue, ...] = ()


class Stage(Protocol):
    def __call__(self, context: PipelineContext) -> StageResult: ...


def run(context: PipelineContext) -> StageResult: ...
```

`PipelineContext` is an immutable carrier. A stage treats every array and nested carrier
received from an earlier stage as read-only and returns a new context using replacement,
never by mutating an earlier result. Array identity may be reused when the value is
unchanged. Sample indices are always zero-based indices into the signal explicitly named
by the carrier; a stage that changes sampling rate must create a new signal/index domain
rather than silently reinterpreting old indices.

Expected inability to produce a measurement is data, not an exception. Stages carry the
same three semantic states that the record exposes:

```python
@dataclass(frozen=True, slots=True)
class Measured(Generic[T]):
    value: T


@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: str


@dataclass(frozen=True, slots=True)
class NotApplicable:
    reason: str


EvidenceValue = Measured[T] | Unavailable | NotApplicable
```

A stage propagates an existing `Unavailable` or `NotApplicable` value unless it owns a
documented rescue path for that quantity. A rescue must return `Measured` with a policy
decision/provenance event naming the source it replaced. A stage may introduce
`NotApplicable` only when applicability is already determined by the validated input
contract or by an applicability policy; algorithmic failure is `Unavailable`, never
`NotApplicable`.

Input-contract violations raise `ECGInputError` before analysis begins. A known local
algorithmic insufficiency is represented as `Unavailable(reason)` plus a `StageIssue`.
Unexpected programming errors are not converted to null measurements and are not caught
by a blanket exception handler. This distinction prevents bugs from becoming apparently
valid sparse records.

A stage may read only the context produced by prior stages, resolved configuration, and
the private engine functions needed for its own work. It may not import the interpretation
distribution, serialize JSON, emit legacy dictionaries, alter configuration, infer a
diagnosis, or call a later stage. Stages are deterministic for the same validated signal,
metadata, resolved configuration, and library version. Any tie-breaking rule must be
stable in validated input lead order and then sample order.

The intended stage order is:

```text
input -> quality -> ventricular -> beats -> delineation
      -> atrial -> measurement -> finalize
```

No stage imports another stage implementation. `extractor.py` imports and sequences
them, which keeps the stage dependency graph acyclic.

## Shared policy object contract

Policies are pure decision objects over already-computed evidence. They separate
"measurement candidate exists" from "candidate is the one the record should use."

```python
DecisionKind = Literal[
    "accept",
    "reject",
    "override",
    "rescue",
    "not_applicable",
    "no_change",
]


@dataclass(frozen=True, slots=True)
class PolicyEvent:
    policy: str
    decision: DecisionKind
    reason_code: str
    source_ids: tuple[str, ...] = ()
    details: Mapping[str, JsonValue] = {}


@dataclass(frozen=True, slots=True)
class PolicyDecision(Generic[T]):
    value: T
    event: PolicyEvent


class Policy(Protocol[T]):
    def decide(self, evidence: T, *, context: PipelineContext) -> PolicyDecision[T]: ...
```

A policy may read resolved configuration, validated input mode/lead identities, quality
summaries, beat/group identities, candidate measurements, and prior policy events. It may
choose among existing candidates, reject a candidate, mark a field not applicable,
authorize a documented rescue, or select an override value that is explicitly computed
from supplied evidence. It may not filter the raw signal, detect new fiducials, mutate
engine objects, construct record JSON, or create diagnostic statements.

Every non-`no_change` policy result becomes a structured provenance event. At minimum
the event records the policy name, stable reason code, selected source/group/beat IDs,
and the relevant resolved configuration value. Numeric evidence included in provenance
uses the same units as the associated measurement. Provenance is descriptive: a reason
such as `paced_native_qt_rescue` states how the value was selected; it is not a clinical
interpretation.

### Worked example: pacing/QT rescue

The pacing/QT cluster is the reference case for the policy split because it accounts for
most of the former `api.py` private helpers.

1. The quality stage detects pacing spikes and acquisition defects and produces evidence;
   it does not decide the final QRS or QT value.
2. The ventricular stage runs QRS detection on the approved signal variants and supplies
   raw, pacing-cleaned, and agreement evidence to `PacingPolicy`.
3. `PacingPolicy` determines pacing context, capture alignment, paced-beat membership,
   and whether a QRS-width rescue/override is justified. A paced-wide override retains
   both the originally measured width and the replacement source in its `PolicyEvent`.
4. The beats and delineation stages build native/paced group candidates and boundaries.
   Tail-settling rescue remains a delineation primitive; whether its result is reportable
   QT is a policy decision.
5. The measurement stage computes candidate QT values without silently copying values
   between groups. `QTPolicy` classifies the measurement path, rejects paths that fail
   evidence gates, and may rescue intermittent-paced QT from a native group or from a
   post-tail-settling candidate.
6. The finalize stage translates the selected value to `measured`,
   `unavailable(reason)`, or `not_applicable(reason)` and copies the policy events into
   record provenance. The record therefore explains that QT came from native beats or
   that QRS duration came from a paced-wide override without serializing the complete
   intermediate graph.

A rescue is idempotent: applying the same policy twice to the same evidence yields the
same selected source and value and does not stack a second numerical correction. Policy
ordering is explicit: QRS/pacing decisions precede beat selection and QT policy; QT policy
never feeds back into ventricular detection.

## `ecgfeat/pipeline/extractor.py`

**Responsibility.** Provide the supported extraction entry point and execute the fixed
stage sequence. It owns orchestration, cancellation of later work after a fatal input
error, and assembly of stage diagnostics for provenance. It does not implement signal
processing, policy thresholds, record serialization, or interpretation.

**Public surface.**

```python
def extract_ecg_record(
    signal: FloatArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
    *,
    config: ExtractionConfig | None = None,
    patient: PatientMeta | None = None,
    record_id: RecordId | None = None,
) -> ECGRecord: ...


@dataclass(slots=True)
class ECGRecordExtractor:
    config: ExtractionConfig

    def extract(
        self,
        signal: FloatArray,
        fs_hz: float,
        lead_names: Sequence[LeadName],
        *,
        patient: PatientMeta | None = None,
        record_id: RecordId | None = None,
    ) -> ECGRecord: ...
```

`ECGRecordExtractor` may cache only immutable resolved configuration or reusable
configuration-derived constants. It must not retain patient signal, context, or prior
result state between calls.

**What it absorbs.** The orchestration body of `api.py:ECGFeatureExtractor.extract`.
The legacy class name and legacy `ECGFeatures` return type do not move here; those are
handled by `compat/api_v0.py`. Calls currently made directly from the god-object become
calls to the eight stages and four policies described below.

**Invariants it must preserve.** Extraction order is fixed and deterministic. Input lead
order is preserved. The public result is an `ECGRecord`, never an interpretation or a
legacy export dictionary. One extractor instance is safe to call repeatedly without
cross-record state. `input_mode="limited"` is passed through without synthesizing
missing 12-lead rows. The resolved config, library version, schema version, and signal
fingerprint used for a result are the ones captured for that same invocation.

**Dependencies.** May import `config`, `record/model`, `record/provenance`,
`pipeline/context`, and each `pipeline/stages/*` module. It may not import private
engine modules directly; engine calls are stage-owned. It may not import compatibility,
interpretation, visualization, or serialization modules.

## `ecgfeat/pipeline/context.py`

**Responsibility.** Define immutable carriers shared by stages and policy events. It owns
index-domain labels and stage-to-stage state shape. It does not perform algorithms,
validation, policy, record construction, or serialization.

**Public surface (private pipeline contract).**

```python
SignalDomain = Literal["native", "analysis", "detection", "pacing_cleaned"]


@dataclass(frozen=True, slots=True)
class SignalView:
    samples: FloatArray
    fs_hz: float
    lead_names: tuple[LeadName, ...]
    domain: SignalDomain


@dataclass(frozen=True, slots=True)
class PipelineContext:
    record_id: RecordId
    config: ResolvedExtractionConfig
    patient: PatientMeta | None
    native: SignalView
    signals: Mapping[SignalDomain, SignalView]
    quality: QualityBundle | None = None
    ventricular: VentricularBundle | None = None
    beats: BeatBundle | None = None
    delineation: DelineationBundle | None = None
    atrial: AtrialBundle | None = None
    measurements: MeasurementBundle | None = None
    field_states: Mapping[str, EvidenceValue[JsonValue]] = {}
    policy_events: tuple[PolicyEvent, ...] = ()
    issues: tuple[StageIssue, ...] = ()


def with_policy_event(
    context: PipelineContext,
    event: PolicyEvent,
) -> PipelineContext: ...


def with_issue(
    context: PipelineContext,
    issue: StageIssue,
) -> PipelineContext: ...
```

**What it absorbs.** No current top-level function moves verbatim. It replaces the large
set of transient locals and in-place cross-step state currently held inside
`api.py:ECGFeatureExtractor.extract`, while reusing internal carrier types moved from
`models.py` into `_engine/foundation/models.py`.

**Invariants it must preserve.** Carriers are immutable after construction; mappings
exposed by the context are read-only snapshots. Each `SignalView` states its own
sampling rate and domain, so sample indices cannot be confused across resampling.
`record_id` and resolved configuration never change during an extraction. Policy events
and issues retain append order, which is stage order. Context contains no serialized JSON
representation of the record.

**Dependencies.** May import `config`, record availability/provenance value types, and
type carriers from `_engine/foundation/models.py`. It imports no stage implementation
and no numerical engine implementation.

## `ecgfeat/pipeline/stages/input.py`

**Responsibility.** Enforce the raw-input contract, resolve channel identities and mains
frequency, create native/analysis/detection signal views, and perform configured
resampling. It does not detect beats, judge clinical suitability, or invent absent leads.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


def resolve_mains_frequency(
    signal: FloatArray,
    fs_hz: float,
    configured_hz: Literal[50, 60] | None,
) -> Literal[50, 60]: ...
```

The second function is internal to the stage module even though it has a typed signature.

**What it absorbs.** `validation.py:validate_ecg_input` and the public
`validation.py:ECGInputError` implementation move to this boundary; the error continues
to be exported from the package facade. It orchestrates
`preprocess.py:resample_ecg`, `:analysis_signal`, `:detection_signal`,
`:lowpass_filter`, `:highpass_filter`, `:bandpass_filter`, `:notch_filter`, and
`:remove_baseline_median` through `_engine/preprocess.py`. It absorbs
`api.py:_resolve_mains_frequency`.

**Invariants it must preserve.** Input is explicitly channel-major and finite where the
input contract requires finite samples. `input_mode="standard"` and
`input_mode="limited"` remain distinct. Limited mode accepts only 1–8 explicitly named,
synchronously sampled channels and never pads to 12 channels. `fs_internal=None` means
native sampling; any explicit internal sampling rate creates a distinct index domain.
`mains_freq` defaults to 50 Hz and any automatic/derived resolution is recorded.
Lead order after validation equals caller order. Resampling is deterministic.

**Dependencies.** `pipeline/context`, `config`, `_engine/preprocess.py`, and the
minimal input-validation helpers colocated in this module. No quality/detection stage or
policy imports.

## `ecgfeat/pipeline/stages/quality.py`

**Responsibility.** Compute acquisition-chain, signal-quality, lead-integrity, detector
agreement precursors, and pacing pre-analysis evidence. It produces evidence consumed by
later policies; it does not decide diagnoses or final reportability.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class QualityBundle:
    lead_quality: tuple[LeadQuality, ...]
    record_summary: Mapping[str, JsonValue]
    acquisition: AcquisitionAssessment
    pacing: PacingEvidence
    limb_reversal: LimbLeadReversalEvidence
    precordial_reversal: PrecordialReversalEvidence
```

**What it absorbs.** Stage-level orchestration around
`quality.py:compute_quality`, `:summarize_record_quality`,
`:prepare_pacing_detection_cache`, `:detect_pacing_spikes`,
`:validate_pacing_spikes_against_qrs`, `:remove_pacing_spikes`,
`:detect_limb_lead_reversal`, `:compute_adjacent_precordial_correlations`, and
`:detect_precordial_reversal`; plus
`acquisition_qc.py:apply_channel_delay_compensation` and
`:assess_acquisition_chain`. The algorithms themselves move to
`_engine/quality/*`. Pacing-context decisions formerly exposed by
`rhythm_rules.py:assess_pacing_evidence_quality`,
`:classify_pacing_context`, and `:detect_pacing_failures` are consumed here as evidence
but their selection semantics belong to `PacingPolicy`.

**Invariants it must preserve.** Quality results retain one entry per measured input
channel in stable lead order. Any channel-delay compensation creates a named compensated
signal view and never changes the caller's raw samples. Pacing-spike locations are
zero-based in the signal domain that produced them. Removal of spikes is used only as an
analysis candidate; the original signal remains available. Reversal detection does not
rename channels. A suspected reversal is evidence, not a silent lead permutation.

**Dependencies.** `pipeline/context`, `_engine/quality/acquisition.py`,
`_engine/quality/signal.py`, and `pipeline/policies/lead_integrity.py`. It may use
`PacingPolicy` only for evidence classification that must precede QRS detection; final
QRS rescue decisions remain in the ventricular stage.

## `ecgfeat/pipeline/stages/ventricular.py`

**Responsibility.** Detect ventricular events and resolve pacing-aware QRS candidates.
It owns the transition from quality evidence to the QRS event set used downstream. It
does not group beats, delineate wave boundaries, or select QT.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class VentricularBundle:
    primary: QRSDetectorResult
    adaptive: QRSDetectorResult | None
    selected_r_peaks: IntArray
    pacing_decision: PacingDecision
    detector_agreement: Mapping[str, JsonValue]
```

**What it absorbs.** Orchestration around
`qrs.py:detect_qrs_multilead_with_meta`,
`adaptive_qrs.py:detect_adaptive_qrs`, and
`quality.py:compute_qrs_detector_agreement`. Pacing rescue/override helpers assigned to
`pipeline/policies/pacing.py` are invoked here rather than remaining private methods of
`api.py`.

**Invariants it must preserve.** R peaks are strictly increasing, unique, and zero-based
in the stated detection/analysis domain. Detector output is deterministic and refractory
ordering is stable. A pacing-cleaned rescue does not erase the raw-detector result.
Overrides preserve the original measured QRS width as evidence. No result is shifted
between sample domains without an explicit deterministic conversion.

**Dependencies.** `pipeline/context`, `_engine/detection/qrs.py`,
`_engine/detection/adaptive_qrs.py`, `_engine/quality/signal.py`, and
`pipeline/policies/pacing.py`.

## `ecgfeat/pipeline/stages/beats.py`

**Responsibility.** Build beat annotations, clusters, morphology families,
representatives, and the deterministic measurement-group selection used by later stages.
It does not delineate boundaries or compute published measurements.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class BeatBundle:
    annotations: tuple[BeatAnnotation, ...]
    clusters: tuple[BeatCluster, ...]
    families: tuple[BeatFamilyAssignment, ...]
    representatives: Mapping[int, FloatArray]
    representative_meta: Mapping[int, RepBeatMeta]
    measurement_group_id: int | None
    measurement_beat_ids: tuple[int, ...]
```

**What it absorbs.** Stage orchestration around `grouping.py:cluster_beats`,
`:build_beat_annotations`, `family_representative.py:assign_morphology_families`,
`:select_family_medoid`, and
`representative.py:build_representative_beats_with_meta`.
From `rhythm_rules.py`, measurement-only
`:classify_post_pause_or_interpolated_beats` and `:select_measurement_beat_ids` move
into this stage; diagnostic statement evidence does not. The assigned `api.py` helpers
are `_select_measurement_group`, `_beat_qrs_profiles`, `_group_qrs_profile`,
`_refine_measurement_group_after_delineation`, and `_build_rhythm_beat_rows`.
The refinement helper is callable again after delineation but remains owned here; the
extractor invokes a stage-owned refinement function rather than moving ownership to the
delineation module.

**Invariants it must preserve.** Beat IDs are stable integers derived from chronological
ventricular-event order and do not change when groups are refined. Group/family IDs have
stable deterministic tie breaks. Representative windows retain a mapping to contributing
beat IDs. Measurement-beat selection is deterministic and never turns a diagnostic
classification into a measurement fact. An empty eligible set is explicit
`Unavailable`, not group 0 or a fabricated representative.

**Dependencies.** `pipeline/context`, `_engine/beats/grouping.py`,
`_engine/beats/representative.py`, `_engine/beats/families.py`, and
`pipeline/policies/pacing.py` for paced-beat membership only.

## `ecgfeat/pipeline/stages/delineation.py`

**Responsibility.** Delineate beat and representative wave boundaries and run explicitly
enabled boundary refinements. It owns fiducial candidates, not publication/reportability
of derived intervals.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class DelineationBundle:
    beat_bounds: Mapping[int, Mapping[LeadName, WaveBounds]]
    representative_bounds: Mapping[int, Mapping[LeadName, WaveBounds]]
    refinements: tuple[BoundaryRefinementEvent, ...]
```

**What it absorbs.** Orchestration around `delineate.py:delineate_beats` and
`:apply_systematic_qrs_tail_settling_rescue`;
`r_localization.py:apply_hybrid_r_localization`;
`wave_localization.py:apply_hybrid_wave_localization`;
`boundary_refinement.py:correct_p_boundaries`;
`t_wave_refinement.py:refine_t_wave_boundaries`; and the corresponding candidate,
repolarization, and localization engine modules. No `api.py` helper is assigned directly
to this stage in the supplied destination table; the former
`_rescue_qt_after_qrs_tail_settling` is correctly owned by `QTPolicy` because the
delineation module supplies only the rescued boundary candidate.

**Invariants it must preserve.** All boundaries are zero-based in an explicitly named
signal domain; onset <= peak <= offset whenever all three exist. Missing boundaries stay
missing rather than being replaced by zero. Opt-in refinements remain independently
ablatable through resolved configuration and do not become implicit defaults. Running a
refinement twice with identical inputs is idempotent. Original and replacement boundary
evidence remain distinguishable for provenance.

**Dependencies.** `pipeline/context` and the private
`_engine/delineation/{core,candidates,refinement,wave_localization,r_localization,repolarization,t_refinement}.py`
modules. It may read pacing decisions but does not import `QTPolicy`.

## `ecgfeat/pipeline/stages/atrial.py`

**Responsibility.** Build P-wave/atrial measurements and measurement-side AV evidence.
It owns atrial signal evidence required by measurement availability; it does not emit AF,
flutter, AV-block, or pre-excitation diagnoses.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class AtrialBundle:
    p_assessments: tuple[PWaveBeatAssessment, ...]
    atrial_events: tuple[AtrialEvent, ...]
    organized_p_ratio: float | None
    pr_dispersion_ms: Milliseconds | None
    av_measurement_evidence: Mapping[str, JsonValue]
```

**What it absorbs.** Orchestration around
`p_wave_engine.py:build_p_wave_assessments`,
`:finalize_p_wave_states`, `:backfill_missing_p_from_robust_engine`,
`:summarize_p_wave_assessments`;
`atrial.py:build_qrst_subtracted_residual`, `:flutter_analysis_signal`,
`:select_atrial_leads`, `:contextual_atrial_analysis`, `:extract_atrial_events`,
`:compute_organized_p_ratio`, and `:compute_pr_dispersion_ms`; and
`atrial_validation.py:validate_atrial_events`.
Assigned `api.py` helpers are `_av_block_evidence`, `_delta_evidence`,
`_pr_series_ms`, and `_atrial_events_per_rr`.
`rhythm_rules.py:detect_av_block_availability_flags` contributes only its
measurement-availability evidence here/applicability policy. Diagnostic
`detect_preexcitation`, pause/AV-block statement evidence, and AF/AFL classification
move to the interpretation distribution.

**Invariants it must preserve.** PR and P-duration values are milliseconds; atrial event
positions are zero-based in their named signal domain. P absence caused by inadequate
signal is `unavailable`, while a field disabled by limited mode can be
`not_applicable`. The stage must not infer absent P solely from RR irregularity. Atrial
event ordering is chronological and deterministic. Backfill operates only on missing
eligible states and never overwrites a measured P result.

**Dependencies.** `pipeline/context`,
`_engine/atrial/{core,p_wave,p_morphology,validation}.py`, and measurement-neutral beat
and delineation carriers. It may read `PacingPolicy` events but does not import
interpretation rules.

## `ecgfeat/pipeline/stages/measurement.py`

**Responsibility.** Compute representative-, group-, and global numerical measurement
candidates and opt-in measurement profiles, then apply measurement-only backfills such as
T axis. It produces candidate values and source metadata; final field applicability and
record construction remain later responsibilities.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class MeasurementBundle:
    representative: Mapping[int, RepresentativeLeadFeatures]
    groups: Mapping[int, GroupFeatures]
    global_features: GlobalFeatures
    qt_decision: QTDecision
    auxiliary: Mapping[str, JsonValue]
```

**What it absorbs.** Orchestration around
`features.py:build_representative_lead_features`, `:compute_group_features`,
`:compute_global_features`, and `:estimate_pr_segment_ms`;
`dispersion.py:summarize_qt_dispersion`;
`measurement_paths.py:build_qt_path_decision`;
`st_baseline.py:calibrated_st_signal` / `:adaptive_st_signal`;
`st_localization.py:apply_hybrid_st_measurement`;
`u_wave.py:measure_u_wave`;
`vector_axis.py:compute_t_axis_from_cluster`;
`twelve_sl.py:apply_twelve_sl_measurement_profile`; and
`glasgow_measurements.py:measure_glasgow_profile`.
Assigned `api.py` helper: `_backfill_t_axis_after_measurement_profile`.
QT selection/rejection/rescue helpers remain in `pipeline/policies/qt.py`.

**Invariants it must preserve.** Published interval candidates use milliseconds and
voltage candidates use millivolts. Selected representatives/groups retain source IDs.
The configured ST amplitude source defaults to `"analysis"`; accepted values are
`"analysis"`, `"calibrated_pr"`, and `"adaptive_pr_tp"`. Every non-default ST source
requires `enable_hybrid_st_measurement=True`; the validation error must name the actual
requested non-default source rather than mentioning only `calibrated_pr`.
`st_hybrid_*` and `twelve_sl_*` fields remain internal-debug/compatibility extension
data and are never shaped as default published measurements. The four-class ST morphology
output remains explicitly unvalidated and is not converted to ischemia evidence here.
Formal QT and axis candidates may still be computed for debugging in limited mode, but
publication state is `not_applicable` and default record payloads omit them as
measurements.

**Dependencies.** `pipeline/context`,
`_engine/measurement/{features,dispersion,paths,st_baseline,st_localization,u_wave,vector_axis}.py`,
measurement profiles, and `pipeline/policies/qt.py`. Calibration is opt-in tooling and
must not be silently invoked during normal extraction.

## `ecgfeat/pipeline/stages/finalize.py`

**Responsibility.** Convert pipeline evidence into the schema-versioned `ECGRecord`,
including field availability, validation tiers, provenance references, and algorithm/
configuration fingerprints. It does not recompute signal measurements, run interpretation,
or serialize the record to JSON text.

**Public surface.**

```python
def run(context: PipelineContext) -> StageResult: ...


def build_record(context: PipelineContext) -> ECGRecord: ...
```

**What it absorbs.** The record-building tail of `api.py:ECGFeatureExtractor.extract`;
measurement-availability semantics from
`rhythm_rules.py:build_measurement_availability` and
`:detect_av_block_availability_flags`; and the transition formerly performed by
constructing `models.py:ECGFeatures`. Fingerprint generation moves through
`record/provenance.py`, not through legacy `export.py`. The old calls to
`interpret`, `to_dict`, and statement builders are removed from extraction.

**Invariants it must preserve.** Every published field has exactly one of
`measured`, `unavailable(reason)`, or `not_applicable(reason)`. No NaN/Inf reaches
the record model. Published measurements, provenance, and internal-debug extensions are
kept in their assigned tiers. Field-address pointers are stable RFC 6901 paths under the
record schema. Limited mode marks diagnosis-dependent/formal QT/axis fields not applicable
rather than filling synthetic 12-lead values. Policy events preserve decision order and
source IDs. Record construction is deterministic; serialization is still one-way through
`record/serialize.py` and is not called here.

**Dependencies.** `pipeline/context`,
`pipeline/policies/{applicability,lead_integrity,qt,pacing}.py`,
`record/{availability,model,provenance,validation}.py`, and `config.py`.
No `_engine/*` import is allowed in this stage.

## `ecgfeat/pipeline/policies/pacing.py`

**Responsibility.** Centralize pacing-context, capture, QRS-rescue, and paced-measurement
selection decisions. It owns decision thresholds and reason codes, not spike detection or
QRS delineation algorithms.

**Public surface.**

```python
@dataclass(frozen=True, slots=True)
class PacingEvidence:
    spike_indices: tuple[SampleIndex, ...]
    validated_spike_indices: tuple[SampleIndex, ...]
    pacing_by_beat: Mapping[int, bool]
    qrs_widths_ms: Mapping[int, Milliseconds]
    capture_alignment: Mapping[int, bool]
    rr_ms: tuple[Milliseconds, ...]


@dataclass(frozen=True, slots=True)
class PacingDecision:
    context: Literal["none", "possible", "intermittent", "dominant"]
    paced_beat_ids: tuple[int, ...]
    qrs_override_ms: Milliseconds | None
    selected_qrs_source: str
    events: tuple[PolicyEvent, ...]


@dataclass(frozen=True, slots=True)
class PacingPolicy:
    def decide(
        self,
        evidence: PacingEvidence,
        *,
        context: PipelineContext,
    ) -> PolicyDecision[PacingDecision]: ...
```

**What it absorbs.** From `rhythm_rules.py`: measurement-side behavior of
`assess_pacing_evidence_quality`, `classify_pacing_context`, and
`detect_pacing_failures`; statement construction remains outside core.
The supplied `api.py` assignments landing here are:
`_pacing_like_av_evidence`, `_dominant_group_qrs_ms`,
`_wide_qrs_pacing_like_context`, `_overwide_qrs_pacing_like_context`,
`_overwide_pacing_qrs_override_ms`, `_raw_pacing_qrs_underestimate_override_ms`,
`_near_wide_pacing_qrs_override_ms`, `_dominant_paced_wide_qrs_override_ms`,
`_intermittent_paced_wide_qrs_override_ms`, `_qrs_wide_values`,
`_intermittent_pacing_wide_measurement_context`,
`_borderline_paced_qrs_wide_offset_override_ms`,
`_secondary_paced_wide_group_qrs_override_ms`,
`_near_wide_paced_qrs_offset_override_ms`,
`_overwide_paced_qrs_raw_consensus_override_ms`, `_paced_beat_fraction`,
`_selected_group_paced_majority`, `_pacing_capture_alignment_confirmed`,
`_confirmed_attenuated_pacing_rescue`, `_pacing_capture_beat_ids`,
`_robust_rr_profile`, `_pacing_qrs_rescue_evidence`,
`_pacing_qrs_cleaned_single_lead_rescue`, `_try_attenuated_pacing_rescue`,
`_pacing_measurement_effect`, `_pacing_segmentation_effect`,
`_measurement_pacing_state`, and `_build_pacing_evidence_by_beat`.

This large list is intentionally cohesive: every helper either derives pacing evidence or
selects the measurement consequence of that evidence. It is the main reason these
functions should become a policy object rather than remain beside orchestration.

**Invariants it must preserve.** Decisions are deterministic and pure. Widths are
milliseconds; beat IDs refer to the stable beat set. An override never overwrites its
source evidence in place. Rescue precedence is fixed and documented so two eligible
rescue rules cannot depend on Python dictionary order. "Possible" pacing alone does not
authorize a wide-QRS override without its required evidence. Re-running the policy is
idempotent.

**Dependencies.** `pipeline/context`, type carriers from
`_engine/foundation/models.py`, and record provenance event types. It consumes quality/
QRS evidence but does not import `_engine/quality/*` or detection algorithms directly.

## `ecgfeat/pipeline/policies/qt.py`

**Responsibility.** Decide which QT candidate, if any, is reportable after path
classification, quality rejection, pacing context, and rescue evidence. It owns QT
selection semantics; it does not detect T waves or compute raw QT intervals.

**Public surface.**

```python
@dataclass(frozen=True, slots=True)
class QTEvidence:
    path: QTPathDecision
    candidates_ms: Mapping[str, Milliseconds]
    lead_candidates_ms: Mapping[LeadName, Milliseconds]
    selected_group_id: int | None
    native_group_id: int | None
    pacing: PacingDecision
    tail_settling_candidate_ms: Milliseconds | None
    reliability_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QTDecision:
    value: EvidenceValue[Milliseconds]
    source: str | None
    used_leads: tuple[LeadName, ...]
    excluded_leads: tuple[LeadName, ...]
    events: tuple[PolicyEvent, ...]


@dataclass(frozen=True, slots=True)
class QTPolicy:
    def decide(
        self,
        evidence: QTEvidence,
        *,
        context: PipelineContext,
    ) -> PolicyDecision[QTDecision]: ...
```

**What it absorbs.** `measurement_paths.py:qt_allowed_for_path`,
`:classify_qt_path`, and `:build_qt_path_decision` supply path evidence through the
engine module. The supplied `api.py` assignments are `_copy_qt_measurements`,
`_apply_qt_reject_gate`, `_rescue_intermittent_paced_qt_from_native`, and
`_rescue_qt_after_qrs_tail_settling`. Any current reliable-lead selection embedded in
the extractor moves here as a private policy helper because it governs reportability
rather than signal detection.

**Invariants it must preserve.** QT values are milliseconds and never copied without a
source identifier. Rejected QT becomes `Unavailable(reason)`; limited-input formal QT is
`NotApplicable(reason)`. Intermittent-paced rescue can select a native-beat candidate
only when native beat/group identity and evidence are explicit. Tail-settling rescue may
replace a rejected candidate only under its documented gate and records both old and new
sources. Lead ordering in `used_leads` and `excluded_leads` follows validated input
order. No QTc is produced from an unavailable QT.

**Dependencies.** `pipeline/context`, `pipeline/policies/pacing.py`,
`_engine/measurement/paths.py` carrier types, and provenance/availability types. It may
not import delineation algorithms.

## `ecgfeat/pipeline/policies/applicability.py`

**Responsibility.** Convert input mode and measurement-side evidence into per-field
applicability/availability decisions. It owns whether a field makes sense for this input;
it does not decide numerical values.

**Public surface.**

```python
@dataclass(frozen=True, slots=True)
class ApplicabilityDecision:
    states: Mapping[str, Measured[bool] | Unavailable | NotApplicable]
    events: tuple[PolicyEvent, ...]


@dataclass(frozen=True, slots=True)
class ApplicabilityPolicy:
    def decide(
        self,
        field_states: Mapping[str, EvidenceValue[JsonValue]],
        *,
        context: PipelineContext,
    ) -> PolicyDecision[ApplicabilityDecision]: ...
```

**What it absorbs.** The supplied `api.py` assignments are
`_apply_measurement_availability_to_representatives` and
`_atrial_measurements_invalid_for_availability`. It also absorbs measurement-only
semantics from `rhythm_rules.py:build_measurement_availability` and
`:detect_av_block_availability_flags`. Diagnostic rule outputs from that module do not
move here.

**Invariants it must preserve.** Applicability is determined before serialization and is
stable for the same validated input/configuration. Limited mode disables diagnosis, axis,
and formal QT reporting with `not_applicable` reasons; it does not manufacture absent
leads. A failed algorithm on an otherwise applicable field is `unavailable`, not
`not_applicable`. Applying availability to representative fields cannot overwrite a
measured value with a weaker state unless a documented invalidation predicate fires.
Reason codes are stable schema-facing strings.

**Dependencies.** `pipeline/context`, `record/availability.py`, and config/input-mode
types. It may read Pacing/QT policy events but imports no engine module.

## `ecgfeat/pipeline/policies/lead_integrity.py`

**Responsibility.** Decide measurement consequences of suspected limb/precordial reversal
and channel-delay findings. It owns exclusions/flag clearing where evidence is refuted; it
does not reorder, relabel, or repair leads silently.

**Public surface.**

```python
@dataclass(frozen=True, slots=True)
class LeadIntegrityEvidence:
    limb_reversal: LimbLeadReversalEvidence
    precordial_reversal: PrecordialReversalEvidence
    channel_delay: AcquisitionAssessment


@dataclass(frozen=True, slots=True)
class LeadIntegrityDecision:
    excluded_leads: tuple[LeadName, ...]
    cleared_flags: tuple[str, ...]
    events: tuple[PolicyEvent, ...]


@dataclass(frozen=True, slots=True)
class LeadIntegrityPolicy:
    def decide(
        self,
        evidence: LeadIntegrityEvidence,
        *,
        context: PipelineContext,
    ) -> PolicyDecision[LeadIntegrityDecision]: ...
```

**What it absorbs.** The supplied `api.py` assignments are
`_clear_precordial_reversal_flags` and `_probable_limb_lead_reversal`. Detection
algorithms remain in `_engine/quality/signal.py`; channel-delay assessment remains in
`_engine/quality/acquisition.py`.

**Invariants it must preserve.** Lead names and caller order never change. A probable
reversal is recorded as evidence and may exclude a lead from a measurement, but the
policy does not swap samples or claim the anatomical correction is known. Clearing a
precordial flag requires explicit refuting evidence and creates a provenance event.
Excluded-lead ordering follows validated input order. Reapplying the policy is
idempotent.

**Dependencies.** `pipeline/context`, quality evidence carrier types, and provenance
types. No direct imports from delineation, measurement, compatibility, or interpretation.

## Complete `api.py` private-helper decomposition

The destination table supplied for this design assigns every one of the 49 private
top-level helpers. Grouped by destination, the assignments are:

| Destination | Assigned helpers |
|---|---|
| `pipeline/stages/input.py` | `_resolve_mains_frequency` |
| `pipeline/stages/beats.py` | `_select_measurement_group`, `_beat_qrs_profiles`, `_group_qrs_profile`, `_refine_measurement_group_after_delineation`, `_build_rhythm_beat_rows` |
| `pipeline/stages/atrial.py` | `_av_block_evidence`, `_delta_evidence`, `_pr_series_ms`, `_atrial_events_per_rr` |
| `pipeline/stages/measurement.py` | `_backfill_t_axis_after_measurement_profile` |
| `pipeline/policies/lead_integrity.py` | `_clear_precordial_reversal_flags`, `_probable_limb_lead_reversal` |
| `pipeline/policies/applicability.py` | `_apply_measurement_availability_to_representatives`, `_atrial_measurements_invalid_for_availability` |
| `pipeline/policies/qt.py` | `_copy_qt_measurements`, `_apply_qt_reject_gate`, `_rescue_intermittent_paced_qt_from_native`, `_rescue_qt_after_qrs_tail_settling` |
| `pipeline/policies/pacing.py` | `_pacing_like_av_evidence`, `_dominant_group_qrs_ms`, `_wide_qrs_pacing_like_context`, `_overwide_qrs_pacing_like_context`, `_overwide_pacing_qrs_override_ms`, `_raw_pacing_qrs_underestimate_override_ms`, `_near_wide_pacing_qrs_override_ms`, `_dominant_paced_wide_qrs_override_ms`, `_intermittent_paced_wide_qrs_override_ms`, `_qrs_wide_values`, `_intermittent_pacing_wide_measurement_context`, `_borderline_paced_qrs_wide_offset_override_ms`, `_secondary_paced_wide_group_qrs_override_ms`, `_near_wide_paced_qrs_offset_override_ms`, `_overwide_paced_qrs_raw_consensus_override_ms`, `_paced_beat_fraction`, `_selected_group_paced_majority`, `_pacing_capture_alignment_confirmed`, `_confirmed_attenuated_pacing_rescue`, `_pacing_capture_beat_ids`, `_robust_rr_profile`, `_pacing_qrs_rescue_evidence`, `_pacing_qrs_cleaned_single_lead_rescue`, `_try_attenuated_pacing_rescue`, `_pacing_measurement_effect`, `_pacing_segmentation_effect`, `_measurement_pacing_state`, `_build_pacing_evidence_by_beat` |
| `_engine/foundation/numeric.py` | `_finite_float`, `_median_or_none` |

That accounts for 49 helpers exactly. I do not recommend a silent reassignment. Three
placements are intentionally close to a boundary and should be kept as assigned:
`_av_block_evidence` stays in the atrial stage only because it computes measurement
evidence, not an AV-block diagnosis; `_build_rhythm_beat_rows` stays in the beats stage
because its rows support deterministic beat selection rather than statement generation;
and `_rescue_qt_after_qrs_tail_settling` stays in `QTPolicy` because delineation owns
the candidate boundary while policy owns whether that candidate may replace published QT.
If any of those helpers currently emits diagnostic labels as a side effect, that side
effect must be split out during the move rather than carrying it into core.


# Phase 3 — private engine and compatibility layer

The `_engine` namespace is an implementation boundary, not a supported consumer API.
The signatures below are the internal contracts that pipeline callers may rely on. The
move should preserve current numerical behavior first; changing an algorithm and moving
it between modules in the same commit would make regressions unnecessarily hard to
attribute.

Because the allowed digest lists class/function names but not every current parameter or
dataclass field, the signatures below define the target typed contract rather than
claiming to reproduce undocumented current positional signatures. During migration,
existing dataclass field sets that are moved mechanically must be preserved until their
callers have been converted; any constructor change is a separate, reviewable change.

## `ecgfeat/_engine/foundation/numeric.py`

**Responsibility.** Hold tiny numerical helpers that are algorithm-neutral and have no
knowledge of ECG stages. It does not own filtering, signal semantics, or measurement
policy.

**Internal surface (unsupported consumer import).**

```python
def trapezoid(y: FloatArray, x: FloatArray | None = None) -> float: ...


def finite_float(value: float | np.floating[Any] | None) -> float | None: ...


def median_or_none(values: Sequence[float | None]) -> float | None: ...
```

**What it absorbs.** `numeric.py:trapezoid`, plus the assigned
`api.py:_finite_float` and `api.py:_median_or_none`. These two private helpers are the
only `api.py` helpers whose destination is below the pipeline layer.

**Invariants it must preserve.** `finite_float` maps None/non-finite values to None and
never emits NaN/Inf. `median_or_none` ignores only explicitly absent/non-finite inputs
according to its documented rule and returns None for no finite values. No helper depends
on lead order or mutable global state. Results are deterministic for identical arrays.

**Dependencies.** Standard library and NumPy only. It imports no other `ecgfeat`
module.

## `ecgfeat/_engine/foundation/models.py`

**Responsibility.** Own internal measurement carriers shared by multiple engine
subpackages. These types model computation state, not the published ECG Record schema.
It does not define availability JSON, schema versions, serialization, or interpretation.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class PatientMeta:
    age_years: float | None = None
    sex: str | None = None
    extras: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class ResolvedPatientAge:
    years: float | None
    source: str | None

    @property
    def known(self) -> bool: ...


def resolve_patient_age(meta: PatientMeta | None) -> ResolvedPatientAge: ...


@dataclass(frozen=True, slots=True)
class LeadQuality:
    lead: LeadName
    score: float | None
    usable: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BeatAnnotation:
    beat_id: int
    r_index: SampleIndex
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveBounds:
    p_on: SampleIndex | None = None
    p_off: SampleIndex | None = None
    qrs_on: SampleIndex | None = None
    qrs_off: SampleIndex | None = None
    t_on: SampleIndex | None = None
    t_off: SampleIndex | None = None


@dataclass(frozen=True, slots=True)
class QRSDetectorResult:
    r_indices: IntArray
    scores: FloatArray | None
    source: str
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class RepresentativeLeadFeatures:
    lead: LeadName
    values: Mapping[str, float | int | str | None]


@dataclass(frozen=True, slots=True)
class GroupFeatures:
    group_id: int
    values: Mapping[str, float | int | str | None]


@dataclass(frozen=True, slots=True)
class GlobalFeatures:
    values: Mapping[str, float | int | str | bool | None]
```

The remaining moved carrier names — `BoundaryEstimate`, `QRSCandidateWindow`,
`PWaveLeadBoundary`, `PWaveBeatAssessment`, and `LeadBeatFeatures` — remain typed
frozen dataclasses with their current field meaning during the mechanical move. Their
exact constructor fields should be copied from the current model definitions when code is
migrated; they must not be guessed from this guide.

**What it absorbs.** The measurement-side classes from `models.py`:
`BoundaryEstimate`, `PatientMeta`, `ResolvedPatientAge`,
`resolve_patient_age`, `LeadQuality`, `BeatAnnotation`, `WaveBounds`,
`QRSCandidateWindow`, `QRSDetectorResult`, `PWaveLeadBoundary`,
`PWaveBeatAssessment`, `LeadBeatFeatures`, `RepresentativeLeadFeatures`,
`GroupFeatures`, and `GlobalFeatures`. `models.py:ECGFeatures` goes to the
compatibility layer, and `models.py:ECGInterpretation` belongs to the separate
interpretation distribution.

**Invariants it must preserve.** Internal sample indices are zero-based and always
interpreted with the caller's declared signal domain. Durations/intervals use
milliseconds and voltages use millivolts when encoded in feature carriers. Carriers do
not serialize themselves. They may contain internal NaN during computation only where
the current algorithm requires it; pipeline finalization must convert such values before
record construction. Equality/order behavior used by existing algorithms must not change
during the move.

**Dependencies.** Standard library and NumPy. It may import tiny type aliases from a
foundation typing location if one is later introduced, but it must not import pipeline,
record, compatibility, or interpretation modules.

## `ecgfeat/_engine/foundation/kalman.py`

**Responsibility.** Provide the optional compiled scalar Kalman recursion used by engine
algorithms. It owns only the recursion implementation and optional JIT dispatch.

**Internal surface (unsupported consumer import).**

```python
def kalman_recursion(
    observations: FloatArray,
    process_variance: float,
    measurement_variance: float,
    initial_state: float,
    initial_variance: float,
) -> FloatArray: ...


def run_kalman(
    observations: FloatArray,
    *,
    process_variance: float,
    measurement_variance: float,
    initial_state: float,
    initial_variance: float,
) -> FloatArray: ...
```

**What it absorbs.** `_kalman.py:kalman_recursion` and
`_kalman.py:run_kalman`.

**Invariants it must preserve.** Python and JIT paths are numerically equivalent within
the existing tolerance and use identical update ordering. No fast-math relaxation is
introduced. `ECGFEAT_DISABLE_JIT=1` continues to force the Python path. No patient
signal/result is cached; only compiled machine code may be cached by Numba.

**Dependencies.** NumPy; Numba is optional and may be imported only here. No other core
module may require Numba.

## `ecgfeat/_engine/preprocess.py`

**Responsibility.** Supply deterministic filtering, baseline, normalization, and
resampling primitives. It does not choose which preprocessing path a record uses; the
input stage owns that choice.

**Internal surface (unsupported consumer import).**

```python
def resample_ecg(
    signal: FloatArray,
    fs_hz: float,
    target_fs_hz: float,
) -> FloatArray: ...


def bandpass_filter(
    signal: FloatArray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
) -> FloatArray: ...


def highpass_filter(signal: FloatArray, fs_hz: float, cutoff_hz: float) -> FloatArray: ...


def lowpass_filter(signal: FloatArray, fs_hz: float, cutoff_hz: float) -> FloatArray: ...


def notch_filter(
    signal: FloatArray,
    fs_hz: float,
    mains_hz: Literal[50, 60],
) -> FloatArray: ...


def zscore(signal: FloatArray, *, axis: int = -1) -> FloatArray: ...


def remove_baseline_median(
    signal: FloatArray,
    fs_hz: float,
    *,
    window_s: float,
) -> FloatArray: ...


def analysis_signal(
    signal: FloatArray,
    fs_hz: float,
    *,
    mains_hz: Literal[50, 60],
) -> FloatArray: ...


def detection_signal(
    signal: FloatArray,
    fs_hz: float,
    *,
    mains_hz: Literal[50, 60],
) -> FloatArray: ...
```

**What it absorbs.** All public functions in current `preprocess.py`:
`resample_ecg`, `bandpass_filter`, `highpass_filter`, `lowpass_filter`,
`notch_filter`, `zscore`, `remove_baseline_median`, `analysis_signal`, and
`detection_signal`.

**Invariants it must preserve.** Inputs/outputs remain channel-major and keep channel
count/order. A no-op resample to identical sampling frequency returns numerically
equivalent data and never shifts sample alignment. Filter behavior at edges and phase
characteristics remain unchanged during the move. Functions are deterministic and do not
modify caller-owned arrays in place.

**Dependencies.** NumPy and SciPy, plus `_engine/foundation/numeric.py` only if needed.
No pipeline/config import.

## `ecgfeat/_engine/quality/acquisition.py`

**Responsibility.** Detect and, where explicitly configured, compensate acquisition-chain
artifacts such as inter-channel delay. It does not infer lead anatomy or clinical
diagnosis.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class AcquisitionAssessment:
    channel_delay_samples: Mapping[LeadName, int]
    compensated: bool
    flags: tuple[str, ...]
    metrics: Mapping[str, float | int | None]


def apply_channel_delay_compensation(
    signal: FloatArray,
    delays: Mapping[LeadName, int],
    lead_names: Sequence[LeadName],
) -> FloatArray: ...


def assess_acquisition_chain(
    signal: FloatArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> AcquisitionAssessment: ...
```

**What it absorbs.** `acquisition_qc.py:apply_channel_delay_compensation` and
`acquisition_qc.py:assess_acquisition_chain`, including their private supporting
helpers.

**Invariants it must preserve.** Delay units are integer samples in the supplied signal
domain. Compensation preserves channel count/order and length according to the current
edge-padding rule. The original signal is never mutated. Assessment is deterministic and
does not relabel channels.

**Dependencies.** NumPy/SciPy and foundation utilities. It must not import pipeline
policies; the quality stage decides whether/where the compensated view is used.

## `ecgfeat/_engine/quality/signal.py`

**Responsibility.** Provide signal-quality, pacing-spike, lead-reversal, and detector-
agreement primitives. It produces observations, not final suitability, measurement
availability, or diagnosis.

**Internal surface (unsupported consumer import).**

```python
def compute_qrs_detector_agreement(
    primary_r: IntArray,
    secondary_r: IntArray,
    fs_hz: float,
    *,
    tolerance_ms: Milliseconds,
) -> Mapping[str, float | int | None]: ...


def summarize_record_quality(
    lead_quality: Sequence[LeadQuality],
) -> Mapping[str, JsonValue]: ...


def build_refutable_gate_flags(
    lead_quality: Sequence[LeadQuality],
) -> Mapping[str, bool]: ...


def compute_quality(
    signal: FloatArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> tuple[LeadQuality, ...]: ...


def prepare_pacing_detection_cache(
    signal: FloatArray,
    fs_hz: float,
) -> Mapping[str, FloatArray]: ...


def detect_pacing_spikes(
    signal: FloatArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
    *,
    cache: Mapping[str, FloatArray] | None = None,
) -> IntArray: ...


def validate_pacing_spikes_against_qrs(
    spike_indices: IntArray,
    r_indices: IntArray,
    fs_hz: float,
) -> IntArray: ...


def remove_pacing_spikes(
    signal: FloatArray,
    spike_indices: IntArray,
    fs_hz: float,
) -> FloatArray: ...


def detect_limb_lead_reversal(
    signal: FloatArray,
    lead_names: Sequence[LeadName],
) -> Mapping[str, JsonValue]: ...


def compute_adjacent_precordial_correlations(
    signal: FloatArray,
    lead_names: Sequence[LeadName],
) -> Mapping[str, float | None]: ...


def detect_precordial_reversal(
    signal: FloatArray,
    lead_names: Sequence[LeadName],
) -> Mapping[str, JsonValue]: ...
```

`quality.py:build_diagnostic_gate` is deliberately not part of this engine surface. Its
measurement-suitability pieces become pipeline availability/policy decisions; any
diagnostic-only gate semantics move to the interpreter.

**What it absorbs.** The current `quality.py` functions listed above and their private
algorithm helpers. `build_diagnostic_gate` is split rather than copied wholesale.

**Invariants it must preserve.** Lead-quality outputs align one-for-one with validated
input lead order. Pacing spike and QRS indices are zero-based in a documented signal
domain. Spike removal returns a new array. Reversal detection reports evidence without
permuting samples. Quality metrics are deterministic and finite when exposed beyond the
engine.

**Dependencies.** NumPy/SciPy, `_engine/foundation/*`, and internal model carriers. No
pipeline policy import.

## `ecgfeat/_engine/detection/qrs.py`

**Responsibility.** Perform multilead ventricular QRS detection and return detector
metadata. It does not choose pacing rescue policy or measurement groups.

**Internal surface (unsupported consumer import).**

```python
def make_vector_magnitude(signal: FloatArray) -> FloatArray: ...


def detect_qrs_multilead_with_meta(
    signal: FloatArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> QRSDetectorResult: ...


def detect_qrs_multilead(
    signal: FloatArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> IntArray: ...
```

**What it absorbs.** `qrs.py:make_vector_magnitude`,
`qrs.py:detect_qrs_multilead_with_meta`, `qrs.py:detect_qrs_multilead`, and their
private helpers.

**Invariants it must preserve.** Detected indices are unique, strictly increasing,
zero-based, and in the supplied signal domain. Refractory and tie-breaking behavior
remain unchanged. Lead input order is stable. The convenience function returns the same
indices as the metadata function's selected result.

**Dependencies.** NumPy/SciPy, `_engine/foundation/models.py`, numeric helpers, and
preprocessing primitives only if currently required. No pacing policy import.

## `ecgfeat/_engine/detection/adaptive_qrs.py`

**Responsibility.** Produce deterministic per-channel adaptive QRS candidates and
corroborated multichannel candidates. It is a detector/evidence source, not the authority
for final QRS selection.

**Internal surface (unsupported consumer import).**

```python
def guard_weak_additions(
    primary: IntArray,
    additions: IntArray,
    *,
    fs_hz: float,
    min_separation_ms: Milliseconds,
) -> IntArray: ...


def channel_candidates(
    signal_1d: FloatArray,
    fs_hz: float,
) -> IntArray: ...


def detect_adaptive_qrs(
    signal: FloatArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> QRSDetectorResult: ...
```

**What it absorbs.** `adaptive_qrs.py:guard_weak_additions`,
`:channel_candidates`, and `:detect_adaptive_qrs`.

**Invariants it must preserve.** Candidate indices are sorted, unique, and zero-based.
Per-channel candidate generation is deterministic. Weak additions never violate the
configured refractory separation. Corroboration preserves stable lead-order tie breaks.

**Dependencies.** NumPy/SciPy and foundation model/numeric helpers. It does not import
the primary QRS detector or pipeline policy unless a shared leaf helper is extracted
first.

## `ecgfeat/_engine/beats/grouping.py`

**Responsibility.** Build beat feature vectors, clusters, and stable beat annotations.
It does not build representative waveforms or decide clinical rhythm labels.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class BeatCluster:
    group_id: int
    beat_ids: tuple[int, ...]
    medoid_beat_id: int | None


def beat_template_vector(
    signal: FloatArray,
    r_index: SampleIndex,
    fs_hz: float,
    *,
    window_ms: tuple[Milliseconds, Milliseconds],
) -> FloatArray: ...


def cluster_beats(
    signal: FloatArray,
    r_indices: IntArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> tuple[BeatCluster, ...]: ...


def build_beat_annotations(
    r_indices: IntArray,
    clusters: Sequence[BeatCluster],
) -> tuple[BeatAnnotation, ...]: ...
```

**What it absorbs.** `grouping.py:BeatCluster`, `:beat_template_vector`,
`:cluster_beats`, and `:build_beat_annotations`.

**Invariants it must preserve.** Beat IDs follow chronological R-peak order and remain
stable after clustering. Every beat belongs to at most one primary cluster. Group IDs and
medoids use deterministic tie-breaking. Template extraction states its sample domain and
does not silently pad missing leads.

**Dependencies.** NumPy/SciPy and foundation models/numeric. No representative,
delineation, or policy import.

## `ecgfeat/_engine/beats/representative.py`

**Responsibility.** Construct group representative beats and metadata linking each
representative to its source beats. It does not choose which group is the published
measurement group.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class RepBeatMeta:
    group_id: int
    contributing_beat_ids: tuple[int, ...]
    center_r_index: SampleIndex
    source_domain: str


def build_representative_beats(
    signal: FloatArray,
    annotations: Sequence[BeatAnnotation],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> Mapping[int, FloatArray]: ...


def build_representative_beats_with_meta(
    signal: FloatArray,
    annotations: Sequence[BeatAnnotation],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> tuple[Mapping[int, FloatArray], Mapping[int, RepBeatMeta]]: ...
```

**What it absorbs.** `representative.py:RepBeatMeta`,
`:build_representative_beats`, and `:build_representative_beats_with_meta`.

**Invariants it must preserve.** Representatives remain channel-major in validated lead
order. Each representative has a stable center index and explicit source beat IDs.
Aggregation and alignment are deterministic. The non-metadata function is exactly the
waveform projection of the metadata-returning function.

**Dependencies.** NumPy/SciPy, foundation models, and beat grouping types. No pipeline
context or policies.

## `ecgfeat/_engine/beats/families.py`

**Responsibility.** Assign morphology families and select medoids within grouped beats.
It does not select the overall measurement group.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class BeatFamilyAssignment:
    beat_id: int
    family_id: int
    distance: float


def assign_morphology_families(
    templates: Mapping[int, FloatArray],
) -> tuple[BeatFamilyAssignment, ...]: ...


def select_family_medoid(
    family_id: int,
    assignments: Sequence[BeatFamilyAssignment],
    templates: Mapping[int, FloatArray],
) -> int: ...
```

**What it absorbs.** `family_representative.py:BeatFamilyAssignment`,
`:assign_morphology_families`, and `:select_family_medoid`.

**Invariants it must preserve.** Each eligible beat receives one family assignment.
Family IDs and medoid selection are deterministic under ties. Distance values are finite
and use one consistent template normalization. No family label is interpreted
clinically.

**Dependencies.** NumPy/SciPy and foundation numeric/model helpers; it may consume
grouping outputs but does not import pipeline stages.

## `ecgfeat/_engine/delineation/core.py`

**Responsibility.** Own the main beat delineation algorithm and systematic QRS-tail
settling candidate repair. It does not decide whether repaired QT becomes reportable.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class BeatWindow:
    beat_id: int
    start: SampleIndex
    r_index: SampleIndex
    stop: SampleIndex


@dataclass(frozen=True, slots=True)
class QRSOffsetRepairTarget:
    beat_id: int
    lead: LeadName
    candidate_index: SampleIndex


@dataclass(frozen=True, slots=True)
class TEndRepairTarget:
    beat_id: int
    lead: LeadName
    candidate_index: SampleIndex


def build_group_priors(
    representative_beats: Mapping[int, FloatArray],
    fs_hz: float,
) -> Mapping[int, Mapping[str, float | int | None]]: ...


def apply_systematic_qrs_tail_settling_rescue(
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    signal: FloatArray,
    fs_hz: float,
) -> Mapping[int, Mapping[LeadName, WaveBounds]]: ...


def delineate_beats(
    signal: FloatArray,
    r_indices: IntArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
    *,
    group_priors: Mapping[int, Mapping[str, float | int | None]] | None = None,
) -> Mapping[int, Mapping[LeadName, WaveBounds]]: ...
```

The additional current classes `PCandidateAlternative`, `PCandidateMeasurement`,
`STJRepairTarget`, `TDualEndpointEvidence`, and `TDualRescueTarget` remain private
typed carriers in this module unless a lower-level delineation module becomes their sole
owner.

**What it absorbs.** `delineate.py:BeatWindow`,
`:PCandidateAlternative`, `:PCandidateMeasurement`, `:QRSOffsetRepairTarget`,
`:STJRepairTarget`, `:TEndRepairTarget`, `:TDualEndpointEvidence`,
`:TDualRescueTarget`, `:build_group_priors`,
`:apply_systematic_qrs_tail_settling_rescue`, and `:delineate_beats`, plus private
helpers that remain central rather than moving to one of the focused delineation modules.

**Invariants it must preserve.** All fiducials are zero-based in the declared signal
domain. Boundary ordering is monotone when boundaries are present. Existing missing-value
semantics are preserved. Tail-settling rescue keeps the original and replacement evidence
distinguishable and is deterministic/idempotent. No record availability or QT
reportability decision is made here.

**Dependencies.** NumPy/SciPy,
`_engine/foundation/{models,numeric,kalman}.py`,
`_engine/delineation/{candidates,repolarization,wave_localization,r_localization}.py`,
and optionally refinement helpers. It must not import pipeline or measurement policy.

## `ecgfeat/_engine/delineation/candidates.py`

**Responsibility.** Generate classical deterministic boundary candidates and select a
coherent candidate sequence. It does not own global beat delineation or policy.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class BoundaryState:
    name: str
    index: SampleIndex
    score: float


def phasor_p_candidates(
    signal_1d: FloatArray,
    fs_hz: float,
    qrs_on: SampleIndex,
) -> tuple[SampleIndex, ...]: ...


def boundary_projection(
    signal_1d: FloatArray,
    fs_hz: float,
    candidate_indices: IntArray,
) -> FloatArray: ...


def select_boundary_sequence(
    states: Sequence[BoundaryState],
) -> tuple[BoundaryState, ...]: ...
```

**What it absorbs.** `classical_candidates.py:phasor_p_candidates`,
`:boundary_projection`, `:BoundaryState`, and `:select_boundary_sequence`.

**Invariants it must preserve.** Candidates are zero-based and sorted by sample index.
Sequence selection is deterministic under equal scores and enforces the same physiological
ordering constraints as the current implementation. It remains model-free and does not
add fitted-runtime dependencies.

**Dependencies.** NumPy/SciPy and foundation numeric helpers only.

## `ecgfeat/_engine/delineation/refinement.py`

**Responsibility.** Provide opt-in boundary refinement candidates. It does not enable
itself or decide whether a candidate should become a published measurement.

**Internal surface (unsupported consumer import).**

```python
def sustained_qrs_offset(
    signal_1d: FloatArray,
    fs_hz: float,
    *,
    r_index: SampleIndex,
) -> SampleIndex | None: ...


def t_onset_change_point(
    signal_1d: FloatArray,
    fs_hz: float,
    *,
    t_peak: SampleIndex,
) -> SampleIndex | None: ...


def correct_p_boundaries(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> Mapping[int, Mapping[LeadName, WaveBounds]]: ...
```

**What it absorbs.** `boundary_refinement.py:sustained_qrs_offset`,
`:t_onset_change_point`, and `:correct_p_boundaries`.

**Invariants it must preserve.** Every candidate is explicit and independently ablatable.
Disabled configuration produces no changes. A replacement remains within its valid beat
window and preserves onset/offset ordering. Re-running a chosen refinement is idempotent.

**Dependencies.** NumPy/SciPy and foundation/delineation carrier types. It never imports
public configuration; the pipeline passes only the decision to invoke it.

## `ecgfeat/_engine/delineation/wave_localization.py`

**Responsibility.** Localize P, S, and T morphology/prominence around existing beat
landmarks and apply the hybrid localization pass. It does not own global delineation
ordering or record fields.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class WaveLocalization:
    index: SampleIndex | None
    amplitude_mv: Millivolts | None
    score: float | None


@dataclass(frozen=True, slots=True)
class SLocalization:
    index: SampleIndex | None
    amplitude_mv: Millivolts | None
    morphology: str | None


def localize_t_prominence(
    signal_1d: FloatArray,
    fs_hz: float,
    search_window: tuple[SampleIndex, SampleIndex],
) -> WaveLocalization: ...


def localize_p_prominence(
    signal_1d: FloatArray,
    fs_hz: float,
    search_window: tuple[SampleIndex, SampleIndex],
) -> WaveLocalization: ...


def localize_s_morphology(
    signal_1d: FloatArray,
    fs_hz: float,
    search_window: tuple[SampleIndex, SampleIndex],
) -> SLocalization: ...


def apply_hybrid_wave_localization(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> Mapping[int, Mapping[LeadName, WaveBounds]]: ...
```

**What it absorbs.** `wave_localization.py:WaveLocalization`,
`:SLocalization`, `:localize_t_prominence`, `:localize_p_prominence`,
`:localize_s_morphology`, and `:apply_hybrid_wave_localization`. The current private
`_Candidate` stays module-private.

**Invariants it must preserve.** Indices are zero-based in the supplied signal domain.
Amplitude is millivolts. Search windows are inclusive/exclusive exactly as established by
the current algorithm and must be documented in code during migration. Hybrid localization
does not alter unrelated lead/beat bounds.

**Dependencies.** NumPy/SciPy and foundation/delineation carriers. No atrial or
measurement policy imports.

## `ecgfeat/_engine/delineation/r_localization.py`

**Responsibility.** Localize R prominence and apply hybrid R localization to existing QRS
bounds. It does not detect ventricular beats.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class RLocalization:
    index: SampleIndex | None
    amplitude_mv: Millivolts | None
    score: float | None


def localize_r_prominence(
    signal_1d: FloatArray,
    qrs_window: tuple[SampleIndex, SampleIndex],
) -> RLocalization: ...


def apply_hybrid_r_localization(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    lead_names: Sequence[LeadName],
) -> Mapping[int, Mapping[LeadName, WaveBounds]]: ...
```

**What it absorbs.** `r_localization.py:RLocalization`,
`:localize_r_prominence`, and `:apply_hybrid_r_localization`.

**Invariants it must preserve.** The chosen R index remains inside the supplied QRS
window, is zero-based, and has deterministic tie-breaking. It never creates/removes
beats; it only refines localization within an existing ventricular event.

**Dependencies.** NumPy and foundation/delineation carriers only.

## `ecgfeat/_engine/delineation/repolarization.py`

**Responsibility.** Generate and score T-wave candidates/components and infer local
polarity. It does not decide QT reportability.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class TWaveComponent:
    peak_index: SampleIndex
    amplitude_mv: Millivolts
    polarity: Literal[-1, 1]


@dataclass(frozen=True, slots=True)
class TWaveCandidate:
    onset: SampleIndex | None
    peak: SampleIndex
    offset: SampleIndex | None
    score: float
    components: tuple[TWaveComponent, ...]


@dataclass(frozen=True, slots=True)
class TWaveMeasurement:
    selected: TWaveCandidate | None
    candidates: tuple[TWaveCandidate, ...]


def infer_cluster_polarity(values: FloatArray) -> Literal[-1, 1] | None: ...


def candidates_from_triplets(
    signal_1d: FloatArray,
    triplets: Sequence[tuple[SampleIndex, SampleIndex, SampleIndex]],
) -> tuple[TWaveCandidate, ...]: ...


def detect_t_wave(
    signal_1d: FloatArray,
    fs_hz: float,
    qrs_offset: SampleIndex,
    next_qrs_onset: SampleIndex | None,
) -> TWaveMeasurement: ...
```

**What it absorbs.** `repolarization.py:TWaveCandidate`,
`:TWaveMeasurement`, `:TWaveComponent`, `:infer_cluster_polarity`,
`:candidates_from_triplets`, and `:detect_t_wave`.

**Invariants it must preserve.** Candidate ordering is deterministic. Onset <= peak <=
offset when endpoints exist. Amplitudes are millivolts and indices are zero-based.
Biphasic/multicomponent evidence remains explicit rather than flattened into a diagnostic
label.

**Dependencies.** NumPy/SciPy, foundation helpers, and candidate/localization primitives.
No QT policy import.

## `ecgfeat/_engine/delineation/t_refinement.py`

**Responsibility.** Fuse T-offset evidence and provide optional T-boundary refinements.
It owns candidate fusion and morphology evidence, not QT publication.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class TFusionResult:
    t_offset: SampleIndex | None
    support: int
    mad_samples: float | None
    reliable: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TFusionMorphologyAssessment:
    accepted: bool
    reasons: tuple[str, ...]


def robust_t_offset_fusion(
    candidates: Mapping[LeadName, Sequence[SampleIndex]],
) -> TFusionResult: ...


def mallat_t_boundaries(
    signal_1d: FloatArray,
    fs_hz: float,
    search_window: tuple[SampleIndex, SampleIndex],
) -> tuple[SampleIndex | None, SampleIndex | None]: ...


def trapezium_t_offset(
    signal_1d: FloatArray,
    fs_hz: float,
    search_window: tuple[SampleIndex, SampleIndex],
) -> SampleIndex | None: ...


def refine_t_wave_boundaries(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> tuple[Mapping[int, Mapping[LeadName, WaveBounds]], TFusionResult]: ...
```

The current private `_LeadTCandidates` remains private to this module.

**What it absorbs.** `t_wave_refinement.py:TFusionResult`,
`:TFusionMorphologyAssessment`, `:robust_t_offset_fusion`,
`:mallat_t_boundaries`, `:trapezium_t_offset`, and
`:refine_t_wave_boundaries`, plus its private helpers.

**Invariants it must preserve.** All indices share one declared signal domain. Fusion is
robust to missing lead candidates and uses stable validated lead ordering under ties.
Support counts are integer counts of actual evidence. Refinement is idempotent and keeps
reliability reasons. A fused offset never becomes a reportable QT solely because this
module produced it.

**Dependencies.** NumPy/SciPy, foundation helpers, repolarization/candidate carriers, and
possibly `_engine/foundation/kalman.py` if currently used. No pipeline imports.

## `ecgfeat/_engine/atrial/core.py`

**Responsibility.** Extract atrial residual/event evidence and aggregate atrial
measurements that are independent of clinical interpretation. It does not classify AF,
flutter, AV block, or other diagnoses.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class AtrialEvent:
    index: SampleIndex
    lead: LeadName
    amplitude_mv: Millivolts | None
    score: float | None


def build_qrst_subtracted_residual(
    signal: FloatArray,
    r_indices: IntArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> FloatArray: ...


def flutter_analysis_signal(
    residual: FloatArray,
    fs_hz: float,
) -> FloatArray: ...


def select_atrial_leads(
    lead_quality: Sequence[LeadQuality],
    lead_names: Sequence[LeadName],
) -> tuple[LeadName, ...]: ...


def contextual_atrial_analysis(
    residual: FloatArray,
    r_indices: IntArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> Mapping[str, JsonValue]: ...


def extract_atrial_events(
    residual: FloatArray,
    r_indices: IntArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> tuple[AtrialEvent, ...]: ...


def compute_organized_p_ratio(
    events: Sequence[AtrialEvent],
    r_indices: IntArray,
    fs_hz: float,
) -> float | None: ...


def compute_pr_dispersion_ms(
    p_assessments: Sequence[PWaveBeatAssessment],
) -> Milliseconds | None: ...
```

**What it absorbs.** `atrial.py:build_qrst_subtracted_residual`,
`:flutter_analysis_signal`, `:select_atrial_leads`,
`:contextual_atrial_analysis`, `:extract_atrial_events`,
`:compute_organized_p_ratio`, and `:compute_pr_dispersion_ms`, plus their private
measurement helpers. Diagnostic classification currently adjacent to those calculations
must not move into this module.

**Invariants it must preserve.** Residual arrays preserve channel-major shape and
validated lead order. Event indices are zero-based in the residual's declared signal
domain. Event ordering is chronological with stable lead-order tie breaking. PR
dispersion is milliseconds. Empty/insufficient evidence returns None or an empty tuple as
specified and is converted to an explicit absence state by the pipeline, never to a
diagnosis.

**Dependencies.** NumPy/SciPy, foundation models/numeric, and delineation carrier types.
No pipeline, record, or interpretation imports.

## `ecgfeat/_engine/atrial/p_wave.py`

**Responsibility.** Assess P-wave evidence, finalize per-beat/lead P states, and perform
the existing robust missing-P backfill. Configuration values are supplied by callers;
the public `PWaveConfig` type itself lives in `config.py`.

**Internal surface (unsupported consumer import).**

```python
def build_p_wave_assessments(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    r_indices: IntArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
    *,
    config: PWaveConfig,
) -> tuple[PWaveBeatAssessment, ...]: ...


def finalize_p_wave_states(
    assessments: Sequence[PWaveBeatAssessment],
    *,
    config: PWaveConfig,
) -> tuple[PWaveBeatAssessment, ...]: ...


def backfill_missing_p_from_robust_engine(
    assessments: Sequence[PWaveBeatAssessment],
    atrial_events: Sequence[AtrialEvent],
    fs_hz: float,
    *,
    config: PWaveConfig,
) -> tuple[PWaveBeatAssessment, ...]: ...


def summarize_p_wave_assessments(
    assessments: Sequence[PWaveBeatAssessment],
) -> Mapping[str, JsonValue]: ...
```

**What it absorbs.** The algorithmic parts of
`p_wave_engine.py:build_p_wave_assessments`,
`:finalize_p_wave_states`, `:backfill_missing_p_from_robust_engine`, and
`:summarize_p_wave_assessments`. `p_wave_engine.py:PWaveConfig` moves to
`config.py` as established in Phase 1.

**Invariants it must preserve.** Assessment ordering is deterministic by beat then
validated lead order. Backfill only fills eligible missing values and never overwrites
already measured P boundaries. All durations are milliseconds and indices remain
zero-based in their declared signal domain. Configuration is immutable input and is not
modified or inferred from the record.

**Dependencies.** NumPy/SciPy, `config.py` for `PWaveConfig`, foundation models,
atrial core carrier types, and delineation candidates. It does not import pipeline
context or interpretation.

## `ecgfeat/_engine/atrial/p_morphology.py`

**Responsibility.** Measure P-wave component morphology from an already localized P
window. It does not decide P presence or diagnostic atrial enlargement.

**Internal surface (unsupported consumer import).**

```python
def measure_p_components(
    signal_1d: FloatArray,
    fs_hz: float,
    p_on: SampleIndex,
    p_off: SampleIndex,
) -> Mapping[str, float | int | str | None]: ...
```

**What it absorbs.** `p_morphology.py:measure_p_components` and its private component
measurement helpers.

**Invariants it must preserve.** P-component amplitudes use millivolts and durations use
milliseconds. The supplied boundary interval is not widened silently. Results are
deterministic, and absent components remain absent rather than becoming zero-amplitude
components.

**Dependencies.** NumPy/SciPy and foundation numeric helpers only.

## `ecgfeat/_engine/atrial/validation.py`

**Responsibility.** Validate atrial/P candidates against measured waveform evidence and
ventricular timing. It does not infer absent P from rhythm irregularity or create
diagnostic labels.

**Internal surface (unsupported consumer import).**

```python
def p_waveform_evidence(
    signal_1d: FloatArray,
    fs_hz: float,
    window: tuple[SampleIndex, SampleIndex],
) -> Mapping[str, float | bool | None]: ...


def ventricular_windows(
    r_indices: IntArray,
    fs_hz: float,
) -> tuple[tuple[SampleIndex, SampleIndex], ...]: ...


def validate_atrial_events(
    signal: FloatArray,
    events: Sequence[AtrialEvent],
    r_indices: IntArray,
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> tuple[AtrialEvent, ...]: ...
```

**What it absorbs.** `atrial_validation.py:p_waveform_evidence`,
`:ventricular_windows`, and `:validate_atrial_events`.

**Invariants it must preserve.** Validation uses only measured channels and detected
ventricular timing. Event order and indices remain stable; rejection removes/marks an
event according to the existing rule but never fabricates a replacement. Results are
deterministic and diagnosis-free.

**Dependencies.** NumPy/SciPy, atrial core carriers, and foundation numeric/model types.

## `ecgfeat/_engine/measurement/features.py`

**Responsibility.** Construct numerical representative-, group-, and global feature
candidates from delineated measurements. It is the main numerical aggregation module,
not a schema, reportability, or interpretation layer.

**Internal surface (unsupported consumer import).**

```python
def estimate_pr_segment_ms(
    assessments: Sequence[PWaveBeatAssessment],
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
) -> Milliseconds | None: ...


def estimate_initial_qrs_axis_deg(
    representative: Mapping[LeadName, FloatArray],
    fs_hz: float,
) -> float | None: ...


def build_representative_lead_features(
    representative: FloatArray,
    bounds: Mapping[LeadName, WaveBounds],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> tuple[RepresentativeLeadFeatures, ...]: ...


def compute_group_features(
    group_id: int,
    representatives: Sequence[RepresentativeLeadFeatures],
    *,
    beat_ids: Sequence[int],
) -> GroupFeatures: ...


def compute_global_features(
    groups: Sequence[GroupFeatures],
    *,
    measurement_group_id: int | None,
    rr_ms: Sequence[Milliseconds],
) -> GlobalFeatures: ...
```

**What it absorbs.** `features.py:estimate_pr_segment_ms`,
`:estimate_initial_qrs_axis_deg`, `:build_representative_lead_features`,
`:compute_group_features`, and `:compute_global_features`, plus the private numerical
helpers that construct the current 73 global feature fields. Publication-tier selection
is not copied here; only numerical candidates remain engine concerns.

**Invariants it must preserve.** Time features are milliseconds, voltage features are
millivolts, axes are degrees, and ratios are dimensionless. Representative/group IDs and
lead ordering are preserved in outputs. Aggregation is deterministic under missing leads
and ties. Internal NaN may exist only where required by current arithmetic and is never
treated as a published measurement. Fields known to be misleading remain internal even
if this engine computes them.

**Dependencies.** NumPy/SciPy, foundation models/numeric, atrial/delineation carriers,
and measurement leaf modules such as dispersion/vector-axis where the dependency remains
one-way. No record or policy import.

## `ecgfeat/_engine/measurement/dispersion.py`

**Responsibility.** Compute explicit QT dispersion estimands over quality-approved,
independent leads. It does not select which leads are approved or decide QT
reportability.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class QTDispersion:
    max_minus_min_ms: Milliseconds | None
    p90_minus_p10_ms: Milliseconds | None
    used_leads: tuple[LeadName, ...]


def summarize_qt_dispersion(
    qt_by_lead_ms: Mapping[LeadName, Milliseconds | None],
    *,
    approved_leads: Sequence[LeadName],
) -> QTDispersion: ...
```

**What it absorbs.** `dispersion.py:QTDispersion` and
`dispersion.py:summarize_qt_dispersion`.

**Invariants it must preserve.** Inputs/outputs are milliseconds. Only explicitly
approved leads enter an estimand. Used-lead order follows validated input order. Fewer
than the required independent leads yields None rather than zero dispersion. Computation
is deterministic.

**Dependencies.** NumPy and foundation numeric helpers only.

## `ecgfeat/_engine/measurement/paths.py`

**Responsibility.** Classify the measurement path used to obtain QT and expose the
path-level allowability facts consumed by `QTPolicy`. It does not make the final record
reportability decision.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class QTPathDecision:
    path: str
    allowed: bool
    reasons: tuple[str, ...]
    source_group_id: int | None


def qt_allowed_for_path(path: str) -> bool: ...


def classify_qt_path(
    *,
    pacing_context: str,
    source_group_id: int | None,
    tail_refined: bool,
) -> str: ...


def build_qt_path_decision(
    *,
    pacing_context: str,
    source_group_id: int | None,
    tail_refined: bool,
) -> QTPathDecision: ...
```

**What it absorbs.** `measurement_paths.py:QTPathDecision`,
`:qt_allowed_for_path`, `:classify_qt_path`, and
`:build_qt_path_decision`.

**Invariants it must preserve.** Path labels/reason codes are stable strings. The module
returns the same path for identical evidence and has no dependence on mutable global
state. It never copies a QT value or converts a path decision into a record absence
state.

**Dependencies.** Standard library/foundation types only. It must not import
`pipeline/policies/qt.py`; dependency points from policy to this module.

## `ecgfeat/_engine/measurement/st_baseline.py`

**Responsibility.** Produce calibrated/adaptive ST analysis signals using measured PR/TP
baseline anchors. It does not choose `st_amplitude_source` or replace the default
analysis signal globally.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class STBaselineResult:
    signal: FloatArray
    source: Literal["calibrated_pr", "adaptive_pr_tp"]
    anchor_count: int
    metadata: Mapping[str, JsonValue]


def calibrated_st_signal(
    raw_signal: FloatArray,
    fs_hz: float,
    pr_anchors: Mapping[LeadName, Sequence[SampleIndex]],
    lead_names: Sequence[LeadName],
) -> STBaselineResult: ...


def adaptive_st_signal(
    raw_signal: FloatArray,
    fs_hz: float,
    pr_anchors: Mapping[LeadName, Sequence[SampleIndex]],
    tp_anchors: Mapping[LeadName, Sequence[SampleIndex]],
    lead_names: Sequence[LeadName],
) -> STBaselineResult: ...
```

**What it absorbs.** `st_baseline.py:STBaselineResult`,
`:calibrated_st_signal`, and `:adaptive_st_signal`.

**Invariants it must preserve.** Output remains channel-major in validated lead order and
same sample domain/length as input. Baseline anchors are zero-based. This path works from
calibrated samples before median-baseline subtraction, as the current module documents.
It never changes detection/delineation signals. Insufficient anchors produce explicit
metadata/failure to the caller rather than silently falling back to another configured
source.

**Dependencies.** NumPy/SciPy and foundation numeric helpers. It may consume delineation
indices but no pipeline config/policy module.

## `ecgfeat/_engine/measurement/st_localization.py`

**Responsibility.** Localize ST measurement points, compute ST quantities, and classify
the existing four-class ST morphology observation. It does not turn morphology into
ischemia evidence.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class STLocalization:
    j_index: SampleIndex | None
    measure_index: SampleIndex | None
    amplitude_mv: Millivolts | None
    slope_mv_per_s: float | None
    morphology: str | None


def classify_st_pattern(
    signal_1d: FloatArray,
    fs_hz: float,
    j_index: SampleIndex,
    measure_index: SampleIndex,
) -> str | None: ...


def localize_st_robust(
    signal_1d: FloatArray,
    fs_hz: float,
    qrs_offset: SampleIndex,
) -> STLocalization: ...


def apply_hybrid_st_measurement(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> Mapping[int, Mapping[LeadName, STLocalization]]: ...
```

**What it absorbs.** `st_localization.py:STLocalization`,
`:classify_st_pattern`, `:localize_st_robust`, and
`:apply_hybrid_st_measurement`, plus private localization helpers.

**Invariants it must preserve.** ST amplitudes are millivolts, slope units are explicit,
and indices are zero-based. `st_hybrid_*` outputs remain internal-debug/compatibility
extension data. The four-class morphology is explicitly unvalidated and cannot be
consumed by the core as diagnostic ischemia evidence.

**Dependencies.** NumPy/SciPy, foundation/delineation carriers, and
`st_baseline.py` output types where needed. No interpretation imports.

## `ecgfeat/_engine/measurement/u_wave.py`

**Responsibility.** Localize and measure U-wave evidence after T-wave analysis. It does
not reinterpret or extend T-offset policy implicitly.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class UWaveMeasurement:
    peak_index: SampleIndex | None
    amplitude_mv: Millivolts | None
    polarity: Literal[-1, 1] | None
    confidence: float | None


def measure_u_wave(
    signal_1d: FloatArray,
    fs_hz: float,
    *,
    t_offset: SampleIndex,
    next_qrs_onset: SampleIndex | None,
) -> UWaveMeasurement: ...
```

**What it absorbs.** `u_wave.py:UWaveMeasurement` and
`u_wave.py:measure_u_wave`, plus its private helpers. Knowledge currently used by
T-offset fusion to identify a late same-polarity hump becomes a typed measurement result
rather than an incidental side effect.

**Invariants it must preserve.** Index is zero-based and constrained to the post-T,
pre-next-QRS search window. Amplitude is signed millivolts. No detected U wave moves T
offset by itself; any T/U interaction remains an explicit delineation/refinement decision.

**Dependencies.** NumPy/SciPy and foundation/delineation carriers.

## `ecgfeat/_engine/measurement/vector_axis.py`

**Responsibility.** Compute axis measurements from eligible lead clusters. It does not
decide whether formal axis reporting applies to the input mode.

**Internal surface (unsupported consumer import).**

```python
@dataclass(frozen=True, slots=True)
class AxisResult:
    degrees: float | None
    used_leads: tuple[LeadName, ...]
    reliable: bool
    reasons: tuple[str, ...]


def compute_t_axis_from_cluster(
    amplitudes_mv: Mapping[LeadName, Millivolts | None],
) -> AxisResult: ...
```

**What it absorbs.** `vector_axis.py:AxisResult` and
`vector_axis.py:compute_t_axis_from_cluster`, plus private helpers. QRS-axis numerical
helpers currently in `features.py` remain there unless a later refactor proves they
share the same leaf contract.

**Invariants it must preserve.** Axis is degrees in the established wrap convention and
is finite when present. Used-lead order is stable. Missing required orthogonal/limb
evidence produces an unreliable/absent result, not a guessed axis. Limited-mode
not-applicability is applied by pipeline policy, not here.

**Dependencies.** NumPy and foundation numeric helpers only.

## `ecgfeat/_engine/measurement/calibration.py`

**Responsibility.** Provide the empirical QT error-risk calibration research utility.
It is optional tooling and is not invoked by default extraction or represented as a
clinical probability.

**Internal surface (unsupported consumer import).**

```python
def confidence_bin(probability: float) -> str: ...


@dataclass(slots=True)
class QTErrorCalibration:
    def fit(
        self,
        predicted_error_features: FloatArray,
        observed_error_ms: FloatArray,
    ) -> QTErrorCalibration: ...

    def predict(self, features: FloatArray) -> FloatArray: ...

    def to_dict(self) -> dict[str, JsonValue]: ...

    @classmethod
    def from_dict(cls, payload: Mapping[str, JsonValue]) -> QTErrorCalibration: ...
```

**What it absorbs.** `calibration.py:confidence_bin` and
`calibration.py:QTErrorCalibration` with its `fit`, `predict`, `to_dict`, and
`from_dict` methods.

**Invariants it must preserve.** Fit/predict behavior is deterministic for fixed inputs
and parameters. Serialization contains model parameters only, never patient data. Output
continues to describe the reference cohort/pipeline calibration and must not be labeled
an externally validated clinical probability. Default extraction never loads/fits this
model implicitly.

**Dependencies.** NumPy/SciPy and standard library only. No pipeline or record import.

## `ecgfeat/_engine/measurement/profiles/twelve_sl.py`

**Responsibility.** Compute the measurement-only 12SL-inspired profile used by current
extraction. It does not implement diagnostic statements or promote profile-specific
extension keys to default published measurements.

**Internal surface (unsupported consumer import).**

```python
def special_t_amplitude_mv(
    signal_1d: FloatArray,
    bounds: WaveBounds,
    fs_hz: float,
) -> Millivolts | None: ...


def twelve_sl_wave_measurements_from_signal(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> Mapping[str, JsonValue]: ...


def apply_twelve_sl_measurement_profile(
    features: GlobalFeatures,
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
) -> GlobalFeatures: ...
```

**What it absorbs.** `twelve_sl.py:special_t_amplitude_mv`,
`:twelve_sl_wave_measurements_from_signal`, and
`:apply_twelve_sl_measurement_profile`, plus private profile helpers.

**Invariants it must preserve.** Profile measurements use standard core units. The 16
known `twelve_sl_*` keys remain internal debug/compatibility extension fields and are
never shaped as default measurements. Applying the profile is deterministic and does not
erase base measurements that it does not own.

**Dependencies.** NumPy/SciPy, foundation models, and delineation/measurement primitives.
No interpretation package.

## `ecgfeat/_engine/measurement/profiles/glasgow.py`

**Responsibility.** Compute Glasgow-style numerical measurement quantities needed by
measurement consumers. It does not own Glasgow statement guards, catalogs, payloads, or
diagnostic interpretation.

**Internal surface (unsupported consumer import).**

```python
def measure_glasgow_profile(
    signal: FloatArray,
    bounds: Mapping[int, Mapping[LeadName, WaveBounds]],
    fs_hz: float,
    lead_names: Sequence[LeadName],
    *,
    patient: PatientMeta | None = None,
) -> Mapping[str, JsonValue]: ...
```

**What it absorbs.** `glasgow_measurements.py:measure_glasgow_profile` and its private
measurement helpers. Current `glasgow.py:build_qtc_statement_guard`,
`:build_measurement_matrix`, `:build_statement_catalog`, and
`:build_glasgow_payload` move to `ecgfeat_interpret/glasgow.py` and are not imported
back into core.

**Invariants it must preserve.** Numerical units stay explicit and identical to the core
measurement units. Missing required leads remain missing; the profile does not synthesize
a full 12-lead input. Results are deterministic. No statement code, diagnostic category,
or rule-engine object enters this module.

**Dependencies.** NumPy/SciPy, foundation models, and measurement/delineation primitives
only.

## `ecgfeat/compat/__init__.py`

**Responsibility.** Expose the deprecated compatibility namespace and nothing else. It
provides a clear migration location for legacy names without making compatibility
internals part of the new primary API.

**Public surface.**

```python
from .api_v0 import ECGFeatureExtractor
from .export_v0 import build_structured_payload, prepare_json_export, to_dict
from .models_v0 import ECGFeatures
```

**What it absorbs.** Deprecated forwarding exports currently reachable from the package
root `__init__.py`. Interpretation-specific compatibility names are kept lazy and live
with the separate interpretation adapter rather than being eagerly imported here.

**Invariants it must preserve.** Importing `ecgfeat.compat` does not import plotting or
the interpretation rule engine. Deprecation warnings are emitted on use (or once at the
documented import boundary), not repeatedly per internal helper call. No compatibility
module changes numerical extraction.

**Dependencies.** Only the three compatibility modules and public core APIs they use.

## `ecgfeat/compat/api_v0.py`

**Responsibility.** Preserve the legacy `ECGFeatureExtractor` object-return workflow
while routing computation through the new record extraction pipeline. It is an adapter,
not a second extraction implementation.

**Public surface.**

```python
@dataclass(slots=True)
class ECGFeatureExtractor:
    config: ExtractionConfig

    def extract(
        self,
        signal: FloatArray,
        fs_hz: float,
        lead_names: Sequence[LeadName],
        *,
        patient: PatientMeta | None = None,
    ) -> ECGFeatures: ...
```

If the current legacy constructor accepts individual keyword options rather than a single
config object, the adapter keeps those accepted keywords during the deprecation window
and resolves them into `ExtractionConfig` before calling the new extractor.

**What it absorbs.** The public class identity and call shape of
`api.py:ECGFeatureExtractor`. Its 49 private helpers do not move here; their destinations
are listed in Phase 2. The adapter calls `pipeline/extractor.py:extract_ecg_record` and
then converts the resulting record through `compat/models_v0.py`.

**Invariants it must preserve.** One input record is analyzed once. The adapter must not
run old and new extraction paths in parallel. Legacy return field meanings remain stable
for the deprecation window, including expected None/default behavior where representable.
Any information impossible to reconstruct from the bounded ECG Record is a documented
compatibility gap rather than a hidden private-engine import. No patient/result state is
cached between calls.

**Dependencies.** Public `config.py`, `pipeline/extractor.py`, record query/model
APIs, and `compat/models_v0.py`. Narrow private adapters are allowed only when a
documented legacy field cannot be reconstructed otherwise; they must not become a new
long-lived dependency surface.

## `ecgfeat/compat/models_v0.py`

**Responsibility.** Preserve legacy measurement dataclass names/shapes long enough for
callers to migrate, and convert from `ECGRecord` into those objects. It does not become
the new internal engine model namespace.

**Public surface.**

```python
@dataclass(frozen=True, slots=True)
class ECGFeatures:
    # Preserve the current legacy field set during migration.
    ...


def features_from_record(record: ECGRecord) -> ECGFeatures: ...
```

The exact `ECGFeatures` constructor field list must be copied mechanically from the
current `models.py:ECGFeatures`; the allowed digest does not enumerate those fields, so
this guide does not invent them. Legacy measurement carrier aliases needed by external
tests may be re-exported temporarily, but new engine code must import their replacements
from `_engine/foundation/models.py`.

**What it absorbs.** `models.py:ECGFeatures` and any legacy aliases/adapters required
for its measurement dataclasses. `models.py:ECGInterpretation` belongs to the separate
interpretation distribution/legacy bridge and must be lazy if re-exported for
compatibility.

**Invariants it must preserve.** Legacy field names, units, and None semantics are
preserved where the new record carries equivalent information. Conversion is
deterministic and one-way from the new record into a compatibility object. The adapter
does not add diagnosis or reconstruct missing raw intermediate arrays. New schema-only
availability states are mapped to the documented legacy representation without losing
the original record.

**Dependencies.** Public `record/model.py`, `record/query.py`, and standard library.
No direct private-engine imports for ordinary conversion.

## `ecgfeat/compat/export_v0.py`

**Responsibility.** Preserve legacy dictionary/export profiles for callers that have not
migrated to the bounded ECG Record serializer. It does not define the new record JSON
schema, fingerprints, or interpretation payload contract.

**Public surface.**

```python
def to_dict(
    value: ECGFeatures | ECGRecord,
    *,
    profile: str | None = None,
) -> dict[str, JsonValue]: ...


def prepare_json_export(
    value: ECGFeatures | ECGRecord,
    *,
    profile: str | None = None,
) -> dict[str, JsonValue]: ...


def build_structured_payload(
    value: ECGFeatures | ECGRecord,
    *,
    profile: str | None = None,
) -> dict[str, JsonValue]: ...
```

**What it absorbs.** Legacy behavior from `export.py:to_dict`,
`:prepare_json_export`, and `:build_structured_payload`. As already assigned in Phase
1, `export.py:clinical_fingerprint` moves to `record/provenance.py`.
`export.py:build_morphology_inputs`, `:build_rhythm_inputs`, and
`:build_statement_engine_payload` are interpretation-adapter concerns and do not remain
in the measurement compatibility exporter.

**Invariants it must preserve.** Legacy profiles remain deterministic for equivalent
legacy objects. New callers are directed to `record/serialize.py`; this module never
changes the bounded record schema. Non-finite numbers are handled exactly as the legacy
profile documented, while the new record path remains stricter. Compatibility export
does not trigger interpretation or plotting.

**Dependencies.** `compat/models_v0.py`, public record model/query/serializer APIs, and
standard library. No private engine import.

# Implementation ordering across Phases 2 and 3

Implementation should follow dependency direction, not the order in which the sections
appear in this document.

1. **Freeze Phase 1 contracts first.** Land `config.py`, record availability/model/
   provenance/validation/query/serialization, and the lean package facades before moving
   extraction. In particular, keep serialization one-way (`serialize -> model`) and
   keep config free of pipeline imports.
2. **Move leaf engine utilities without behavior changes.** Move
   `foundation/{numeric,models,kalman}.py` and `preprocess.py`, then
   `quality/{acquisition,signal}.py` and `detection/{qrs,adaptive_qrs}.py`. Preserve
   existing tests through temporary import forwarding if necessary.
3. **Move beat and delineation engines.** Land
   `beats/{grouping,representative,families}.py`, followed by
   `delineation/{candidates,r_localization,wave_localization,repolarization,refinement,t_refinement,core}.py`.
   Keep `core.py` last within delineation because it composes the leaf algorithms.
4. **Move atrial and measurement engines.** Land the four `atrial/*` modules, then
   measurement leaf modules (`dispersion`, `paths`, `st_baseline`,
   `st_localization`, `u_wave`, `vector_axis`, `calibration`, profiles), and only
   then `measurement/features.py`.
5. **Introduce `pipeline/context.py` and policy event carriers.** Put `PolicyEvent`
   and other shared immutable policy-result carriers in `pipeline/context.py` (or an
   equally low pipeline contract module if the tree is later extended). Context must not
   import any policy implementation.
6. **Move the four policies.** Implement lead integrity and applicability first, then
   pacing, then QT. Pacing depends only on evidence carriers; QT may depend on
   `measurement/paths.py` and pacing decision carriers. Engine modules must never import
   these policies.
7. **Implement stages in dataflow order.** Input, quality, ventricular, beats,
   delineation, atrial, measurement, finalize. At each step replace one slice of
   `ECGFeatureExtractor.extract` with a stage call and keep the numerical output
   comparison local to that slice.
8. **Land `pipeline/extractor.py` after all stages exist.** Make
   `extract_ecg_record` the sole new orchestration path and remove the old god-object
   orchestration only after parity checks pass.
9. **Add compatibility last.** `compat/models_v0.py` and `compat/export_v0.py` adapt
   the stable record contract; `compat/api_v0.py` wraps the new extractor. Root-level
   deprecated forwarding exports can then be switched to lazy imports.

The migration should avoid these circular-dependency traps:

- `pipeline/context.py` must define or own the shared `PolicyEvent`/decision carrier
  types rather than importing `pipeline/policies/*`; policies import the context
  contract, never the reverse.
- `_engine/*` must never import `pipeline/*`. Engine functions accept primitive values
  and foundation carriers. This is the main guard against stage/policy cycles.
- `record/*` must never import pipeline or engine types. Finalize translates engine/
  pipeline carriers into record values at the boundary.
- `PacingPolicy` may consume quality/detection carriers, but quality/detection modules
  must not import the policy. Likewise, `QTPolicy` may consume
  `measurement/paths.py:QTPathDecision`, while `paths.py` cannot import the policy.
- Delineation may produce QT-relevant candidate boundaries, but it cannot import
  measurement/QT policy. Measurement consumes delineation results; QT policy selects
  among measurement candidates. This preserves
  `delineation -> measurement -> policy/finalize` data flow without a feedback import.
- `_engine/foundation/models.py` must remain a leaf carrier module. Do not let it import
  specialized atrial, delineation, or measurement modules merely to annotate fields; use
  primitive/foundation types or forward protocols instead.
- Compatibility code may import public core APIs. Core modules must not import
  compatibility code. Root facade compatibility exports must be lazy so importing
  `ecgfeat` does not pull legacy models, interpretation, or plotting into the runtime.
- The interpretation distribution consumes only the published ECG Record API (plus raw
  signal only for an explicitly documented interpretation need). It must never reach
  back into `_engine` to recover evidence omitted from the record; if evidence is truly
  required, promote a stable measurement/provenance field through the record design.

# Open questions

- **Exact moved dataclass constructors.** The allowed module digest names the internal
  model classes but does not enumerate every current field. Before code movement, copy
  the exact field sets for `BoundaryEstimate`, `QRSCandidateWindow`,
  `PWaveLeadBoundary`, `PWaveBeatAssessment`, `LeadBeatFeatures`, and legacy
  `ECGFeatures` mechanically. Decide field redesign only after import migration, so a
  module move is not conflated with a data-model rewrite.
- **Stable reason-code registry.** Policy/absence reason strings are effectively part of
  the published record contract once consumers address fields by JSON pointer. Decide
  whether reason codes are versioned in the record schema itself or in a separately
  documented registry; either way, free-form messages must not be the machine contract.
- **Internal extension namespace.** The field-tier decision is settled
  (`st_hybrid_*` and `twelve_sl_*` are internal debug/compatibility only), but the
  bounded JSON location for opt-in compatibility extensions still needs one explicit
  schema choice. Do not place them beside default published measurements.
- **Compatibility reconstruction limit.** Determine which legacy `ECGFeatures` fields
  cannot be reconstructed from the published record plus sanctioned compatibility
  extensions. Those fields need an explicit deprecation gap or a narrowly scoped
  compatibility extension; they should not justify exposing the complete private engine
  graph.
- **`build_rhythm_beat_rows` long-term home.** Its assigned Phase-2 destination
  (`pipeline/stages/beats.py`) is coherent while those rows drive measurement-beat
  selection. If, after interpretation extraction, the rows are used only for diagnostic
  statements, move the diagnostic projection to `ecgfeat_interpret` and keep only the
  measurement-selection attributes in core.
- **QT calibration packaging.** `_engine/measurement/calibration.py` can remain in the
  core wheel because it has no extra runtime dependency, but it is research tooling
  rather than extraction runtime. Decide whether a later release should move it to an
  optional tooling package; this does not block the pipeline split.
