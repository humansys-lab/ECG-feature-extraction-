# Public API Design

> Implementation status: this document includes target contracts, not completed
> release claims. [Document 08](08_implementation_status.md) records the implemented
> subset, reconciled decisions and outstanding gates. Document 02 is authoritative
> for the JSON layout; Python carrier sketches must not create a second wire format.

This document defines the public Python and command-line surface for the measurement package. The stable consumer boundary is the raw signal plus a versioned ECG Record JSON document. Interpretation is outside this package and has its own versioned document and distribution.

## A. Entry points

The public namespace uses flat <code>ecg_*</code> functions. The one-call path is the default for application code; lower-level functions expose the same pipeline without requiring callers to depend on internal model classes or intermediate algorithm objects.

~~~python
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, TextIO, TypeAlias

from numpy.typing import ArrayLike

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
InputMode: TypeAlias = Literal["standard_12", "limited"]
AmplitudeUnit: TypeAlias = Literal["mV", "uV"]
RecordProfile: TypeAlias = Literal["summary", "all", "debug"]
AlgorithmName: TypeAlias = str
~~~

~~~python
def ecg_record(
    ecg: ArrayLike,
    *,
    sampling_rate: float,
    lead_names: Sequence[str],
    amplitude_unit: AmplitudeUnit = "mV",
    patient: PatientMeta | Mapping[str, JsonValue] | None = None,
    method: AlgorithmName = "default",
    profile: RecordProfile = "summary",
    config: ECGConfig | None = None,
) -> ECGRecord:
    """Measure an ECG and return one validated, versioned ECG Record."""
~~~

<code>ecg_record()</code> is the recommended one-call API. For <code>input_mode="standard_12"</code>, <code>lead_names</code> must identify the canonical 12 leads exactly once each, although row order may differ. For <code>input_mode="limited"</code>, one through eight synchronously sampled rows are accepted; every row must have an explicit unique name, and callers must not pre-pad missing leads to twelve rows. Limited mode retains raw measurements but records axis and formal QT outputs as not applicable where the input contract cannot support them. Diagnosis remains outside the measurement package.

<code>method=</code> is a stable algorithm identifier rather than a generic keyword passthrough. <code>"default"</code> means the package's versioned default policy; the resolved algorithms and any fallback path are written to provenance. <code>profile=</code> controls record emission, not measurement semantics: <code>"summary"</code> is the default JSON profile, <code>"all"</code> includes all published measurements, and <code>"debug"</code> additionally includes unstable internal diagnostics. Experimental behavior is selected only through explicit configuration fields described in section C.

~~~python
def ecg_prepare(
    ecg: ArrayLike,
    *,
    sampling_rate: float,
    lead_names: Sequence[str],
    amplitude_unit: AmplitudeUnit = "mV",
    patient: PatientMeta | Mapping[str, JsonValue] | None = None,
    input_mode: InputMode = "standard_12",
) -> ECGInput:
    """Validate and normalize signal, channel, unit, and patient metadata into an immutable ECG input."""
~~~

~~~python
def ecg_measure(
    prepared: ECGInput,
    *,
    method: AlgorithmName = "default",
    config: ECGConfig | None = None,
) -> ECGMeasurements:
    """Run measurement algorithms and return typed measurements plus provenance before profile filtering."""
~~~

~~~python
def ecg_emit(
    measurements: ECGMeasurements,
    *,
    profile: RecordProfile = "summary",
) -> ECGRecord:
    """Assemble and validate a versioned ECG Record from measured quantities and provenance."""
~~~

~~~python
def ecg_dump(
    record: ECGRecord,
    destination: str | Path | TextIO | BinaryIO,
    *,
    indent: int | None = None,
) -> None:
    """Serialize an ECG Record as UTF-8 JSON without changing its measurement content."""
~~~

The split is deliberate. <code>ecg_prepare()</code> owns the input contract, <code>ecg_measure()</code> owns algorithms, and <code>ecg_emit()</code> owns schema/profile projection. A caller that only needs the stable record uses <code>ecg_record()</code> and never sees intermediate structures.

The design adopts NeuroKit2's flat verb-noun style, the <code>method=</code> string convention, and a one-call function layered over composable primitives. It diverges in six places:

1. Multi-lead orientation and explicit lead names are part of the signature because the package's contract is 12-lead ECG plus a defined limited-input mode, rather than an implicit single-lead vector.
2. No public function accepts algorithm <code>**kwargs</code>; discoverable immutable config objects carry every supported option.
3. Mutable DataFrames are not a boundary type; public results are typed immutable views or schema-backed records.
4. No unstructured <code>info</code> dictionary is returned; units, provenance, validation status, and absence semantics are represented in named types and in the ECG Record schema.
5. Plotting is not part of measurement entry points and never draws into global matplotlib state.
6. <code>numpy</code> and <code>scipy</code> remain the mandatory runtime dependencies. Pandas, matplotlib, and scikit-learn are not required by the measurement package.

Unknown <code>method</code> identifiers raise <code>ConfigurationError</code> before measurement starts. Package releases may add identifiers, but an existing identifier may not silently change to a materially different algorithm without a documented compatibility change.

## B. Record read and query API

Record reading validates schema identity, schema version, field types, units, axes, and absence encoding before exposing typed query objects. A parsed record is immutable from the public API's perspective.

~~~python
RecordValidationLevel: TypeAlias = Literal["strict", "schema", "none"]

def load_record(
    source: str | Path | TextIO | BinaryIO | Mapping[str, JsonValue],
    *,
    validate: RecordValidationLevel = "strict",
) -> ECGRecord:
    """Load ECG Record JSON or a mapping into an immutable typed record."""
~~~

~~~python
def validate_record(
    record: ECGRecord | Mapping[str, JsonValue],
    *,
    level: Literal["strict", "schema"] = "strict",
) -> ECGRecord:
    """Validate a record and return the corresponding immutable typed record."""
~~~

<code>validate="strict"</code> is the default and is required for records crossing a trust boundary. <code>"schema"</code> performs structural/schema validation without optional cross-field consistency checks. <code>"none"</code> is accepted only by <code>load_record()</code> for already-trusted in-process data and does not suppress JSON parsing errors.

Canonical addresses use the settled form <code>ecg-record:&lt;record_id&gt;@&lt;schema_version&gt;#&lt;json-pointer&gt;</code>, with the fragment encoded as an RFC 6901 JSON Pointer.

~~~python
@dataclass(frozen=True, slots=True)
class RecordAddress:
    record_id: str
    schema_version: str
    pointer: str

def parse_address(address: str) -> RecordAddress:
    """Parse and validate one canonical ECG Record address."""
~~~

~~~python
@dataclass(frozen=True, slots=True)
class AddressResolution:
    address: RecordAddress
    value: JsonValue
    node_kind: Literal["record", "measurement", "metadata", "provenance", "debug"]

def resolve_address(
    record: ECGRecord,
    address: str | RecordAddress,
) -> AddressResolution:
    """Resolve an ECG Record address against a loaded record after identity and version checks."""
~~~

<code>resolve_address()</code> never performs network or filesystem lookup. The supplied record must match both <code>record_id</code> and <code>schema_version</code>; a mismatch raises <code>RecordIdentityError</code>. A syntactically valid pointer that is not present raises <code>AddressNotFoundError</code>.

Measurement queries use stable measurement names, with optional lead and beat selectors. They do not expose the storage layout of dense arrays.

~~~python
@dataclass(frozen=True, slots=True)
class MeasurementQuery:
    name: str
    lead: str | None = None
    beat: int | None = None

@dataclass(frozen=True, slots=True)
class NullAbsence:
    kind: Literal["null"] = "null"

@dataclass(frozen=True, slots=True)
class UnmeasurableAbsence:
    reason: str
    kind: Literal["unmeasurable"] = "unmeasurable"

@dataclass(frozen=True, slots=True)
class NotApplicableAbsence:
    reason: str
    kind: Literal["not_applicable"] = "not_applicable"

Absence: TypeAlias = NullAbsence | UnmeasurableAbsence | NotApplicableAbsence
MeasurementScalar: TypeAlias = int | float | str | bool

@dataclass(frozen=True, slots=True)
class MeasurementProvenance:
    method_requested: str
    method_resolved: str
    config_hash: str
    fallback_path: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ValidationStatus:
    tier: Literal["validated", "partially_validated", "unvalidated"]
    evidence: str | None = None

@dataclass(frozen=True, slots=True)
class MeasurementResult:
    name: str
    address: RecordAddress
    value: MeasurementScalar | None
    unit: str | None
    lead: str | None
    beat: int | None
    absence: Absence | None
    provenance: MeasurementProvenance
    validation: ValidationStatus
~~~

~~~python
def query_measurement(
    record: ECGRecord,
    name: str,
    *,
    lead: str | None = None,
    beat: int | None = None,
) -> MeasurementResult:
    """Return one measurement cell with unit, absence, provenance, and validation status."""
~~~

~~~python
def query_many(
    record: ECGRecord,
    queries: Sequence[MeasurementQuery],
) -> tuple[MeasurementResult, ...]:
    """Evaluate measurement queries in input order without converting the record to a table."""
~~~

~~~python
@dataclass(frozen=True, slots=True)
class MeasurementSelection:
    record_id: str
    schema_version: str
    results: tuple[MeasurementResult, ...]

def select_measurements(
    record: ECGRecord,
    queries: Sequence[MeasurementQuery],
) -> MeasurementSelection:
    """Create an immutable subset view suitable for emitting selected feature quantities."""
~~~

~~~python
def dump_measurements(
    selection: MeasurementSelection,
    destination: str | Path | TextIO | BinaryIO,
    *,
    format: Literal["json", "jsonl"] = "json",
) -> None:
    """Serialize selected measurement results while preserving units, absence, and provenance."""
~~~

The three absence states are returned as data, not exceptions:

| Record state | <code>MeasurementResult.value</code> | <code>MeasurementResult.absence</code> | Meaning |
|---|---:|---|---|
| Present value | scalar | <code>None</code> | Measurement exists and has a value. |
| JSON <code>null</code> | <code>None</code> | <code>NullAbsence()</code> | Schema permits an explicit null but no stronger reason is asserted. |
| Unmeasurable | <code>None</code> | <code>UnmeasurableAbsence(reason=...)</code> | Input contract supports the quantity, but this record cannot produce a trustworthy value. |
| Not applicable | <code>None</code> | <code>NotApplicableAbsence(reason=...)</code> | The quantity does not apply to this input contract, such as formal QT in limited mode. |

A missing measurement name is different from an absent measurement value and raises <code>MeasurementNotFoundError</code>. An invalid lead name or beat selector raises <code>MeasurementSelectorError</code>. This makes typo/schema failures impossible to confuse with physiological non-measurability.

Lead and beat traversal is explicit and preserves record order.

~~~python
@dataclass(frozen=True, slots=True)
class LeadView:
    name: str
    index: int
    record: ECGRecord

@dataclass(frozen=True, slots=True)
class BeatView:
    index: int
    sample: int | None
    time_s: float | None
    lead: str | None
    record: ECGRecord

def iter_leads(record: ECGRecord) -> Iterator[LeadView]:
    """Iterate named leads in the record's declared lead order."""

def iter_beats(
    record: ECGRecord,
    *,
    lead: str | None = None,
) -> Iterator[BeatView]:
    """Iterate beats globally or within one lead using stable beat indices."""
~~~

<code>MeasurementResult.validation</code> is the field's explicit validation status, not a quality score invented at query time. <code>MeasurementResult.provenance</code> contains the algorithm/config/fallback information relevant to the returned field. Debug-only values are queryable only when they actually exist in a record emitted with <code>profile="debug"</code>; their addresses carry no compatibility guarantee.

The dense all-measurements matrix may live in the optional NPZ sidecar. <code>load_record()</code> represents sidecar-backed values through an immutable sidecar reference and validates sidecar identity before access. JSON remains authoritative for record identity, schema, units, axes, provenance, and absence semantics.

## C. Configuration objects

Configuration is immutable, explicit, serializable, and hashable by canonical content. Input-specific sampling rate and lead names remain function arguments; algorithm policy lives in frozen configuration objects.

~~~python
STAmplitudeSource: TypeAlias = Literal["analysis", "calibrated_pr", "adaptive_pr_tp"]

from ecgfeat.refinement import RefinementConfig
# Existing 18 named boolean flags remain the single source of truth; all default
# to False. enabled and experimental_flags are computed properties, not fields.
# experimental(*actual_flag_names, enabled=True) rejects unknown names.
# experimental() without names retains the legacy all-candidates research preset.

@dataclass(frozen=True, slots=True)
class PWaveConfig:
    enabled: bool = True
    validate_waveform: bool = True
    allow_residual_analysis: bool = True
    min_support_beats: int = 1

@dataclass(frozen=True, slots=True)
class ECGConfig:
    input_mode: InputMode = "standard_12"
    fs_internal: float | None = None
    mains_frequency_hz: Literal[50, 60] | None = 50
    refinement: RefinementConfig = RefinementConfig()
    p_wave: PWaveConfig = PWaveConfig()
    st_amplitude_source: STAmplitudeSource = "analysis"
~~~

The record-layer P-wave controls above are a reserved target surface; non-default
values currently raise `ConfigurationError` because they are not yet wired to
the numerical engine. The top-level `ecgfeat.PWaveConfig` retains the legacy
engine's constructor, while `ecgfeat.config.PWaveConfig` is this record-layer
type. Unifying them requires an explicit mapping and parity tests, not ignored
configuration. Existing `RefinementConfig` named flags remain effective and
are shared by both extraction entry points.

<code>sampling_rate</code> describes each input signal, not configuration.
<code>fs_internal=None</code> preserves the native rate, as required by the frozen
baseline in document 05. Explicit resampling still publishes fiducials in native
sample coordinates using round-half-to-even. <code>mains_frequency_hz=None</code>
retains the existing engine's 50/60 Hz spectral inference (50 Hz for a tie or
sampling rates at or below 122 Hz); it does **not** disable notch filtering.
The resolved frequency is recorded in provenance. An explicit no-notch mode
would require a separate behavioral change and validation.

<code>st_amplitude_source="analysis"</code> is the package's stable production path, verified against <code>api.py:1771</code> where the parameter defaults to <code>"analysis"</code>. The validated set is <code>{"analysis", "calibrated_pr", "adaptive_pr_tp"}</code> (<code>api.py:1778</code>); any non-default source requires <code>enable_hybrid_st_measurement</code>, which itself already defaults to <code>True</code> (<code>api.py:1769</code>, guard at <code>api.py:1783</code>), so in practice the opt-in is a single field. The guard covers all non-default sources but its error message names only <code>calibrated_pr</code> (<code>api.py:1784</code>); the new API should fold the flag into the source selection so no invalid combination is constructible and no misleading message is possible. <code>"calibrated_pr"</code> and <code>"adaptive_pr_tp"</code> are explicit opt-in research paths and carry coverage costs.

Experimental selection stays explicit. <code>RefinementConfig.experimental("flag_name")</code> is convenience syntax for named experimental flags; unknown flags raise <code>ConfigurationError</code>. Production defaults never enable experimental flags. If an experimental combination degrades or triggers fallback behavior, the resolved path is recorded rather than hidden.

Configuration provenance is canonical JSON data rather than representation text.

~~~python
@dataclass(frozen=True, slots=True)
class ConfigProvenance:
    config: Mapping[str, JsonValue]
    config_hash: str
    method_requested: str
    method_resolved: str
    experimental: tuple[str, ...]
    fallback_path: tuple[str, ...]

def config_provenance(
    config: ECGConfig,
    *,
    method: AlgorithmName = "default",
) -> ConfigProvenance:
    """Return the canonical serialized configuration and its reproducibility metadata."""
~~~

Canonicalization uses schema-defined field names, UTF-8, sorted object keys, stable numeric encoding, and a documented hash algorithm. The ECG Record stores the canonical configuration payload or a schema-defined normalized subset plus <code>config_hash</code>, the requested and resolved algorithm identifiers, every enabled experimental flag, and any fallback path. A defaulted option is serialized with its resolved value so two runs that differ because package defaults changed cannot accidentally share a provenance identity.

## D. Error and warning taxonomy

<code>ECGInputError</code> remains the root of the public exception hierarchy so existing callers that already catch it continue to catch user-correctable failures.

~~~python
class ECGInputError(ValueError):
    """Base class for all public ECG measurement, record, and query errors."""

class SignalShapeError(ECGInputError):
    """Signal rank, orientation, length, or channel count violates the input contract."""

class LeadNameError(ECGInputError):
    """Lead names are missing, duplicated, unknown where prohibited, or inconsistent with rows."""

class SamplingRateError(ECGInputError):
    """Sampling rate or internal resampling configuration is invalid."""

class AmplitudeUnitError(ECGInputError):
    """Amplitude unit is unsupported or inconsistent with the declared input contract."""

class ConfigurationError(ECGInputError):
    """A method or configuration value is unsupported or internally inconsistent."""

class RecordValidationError(ECGInputError):
    """ECG Record JSON violates the supported schema or cross-field invariants."""

class RecordIdentityError(ECGInputError):
    """A canonical address targets a different record identity or schema version."""

class AddressSyntaxError(ECGInputError):
    """An ECG Record address or JSON Pointer is malformed."""

class AddressNotFoundError(ECGInputError):
    """A valid pointer does not exist in the supplied record."""

class MeasurementNotFoundError(ECGInputError):
    """A requested measurement name is not published in this record/schema."""

class MeasurementSelectorError(ECGInputError):
    """A requested lead or beat selector is invalid for the measurement axes."""

class ComputationInvariantError(ECGInputError):
    """An internal invariant failed such that a trustworthy record cannot be emitted."""
~~~

The raise-versus-record rule is:

> **Raise when the package cannot establish or honor the contract for the call or cannot trust the record as a whole. Record an absence when the input contract is valid and the record remains trustworthy, but one published quantity has no valid value for this record.**

Consequences are concrete:

- Invalid shape, missing or duplicate lead names, illegal limited-mode padding, unsupported amplitude units, impossible sampling rates, unknown methods, invalid experimental flags, malformed record JSON, bad addresses, and impossible selectors raise.
- Failure to detect a trustworthy P wave, T offset, interval, morphology value, or other quantity on an otherwise valid ECG records <code>unmeasurable</code> with a stable reason code.
- Quantities excluded by the valid input contract, including formal QT or axis outputs in limited mode where specified, record <code>not_applicable</code> with a stable reason code.
- A schema-defined nullable value with no stronger semantic claim records JSON <code>null</code>.
- If an algorithm fallback still produces a trustworthy measurement, the value is returned and the fallback path is provenance. If fallback cannot produce a trustworthy value, the field is <code>unmeasurable</code>.
- A violated internal invariant that calls the integrity of the record into question raises <code>ComputationInvariantError</code>; it must not be converted into many misleading nulls.

Absence reasons are stable machine-readable codes with optional human-readable detail. The stable code is part of the published contract; free-form detail is not.

Warnings are intentionally narrow.

~~~python
class ECGWarning(UserWarning):
    """Base class for actionable non-fatal public warnings."""

class ECGCompatibilityWarning(ECGWarning):
    """A compatibility shim accepted legacy input whose behavior needs caller attention."""

class ECGDeprecationWarning(FutureWarning, ECGWarning):
    """A public API is scheduled for removal or semantic replacement."""
~~~

Expected physiological absence, limited-mode not-applicable fields, and normal algorithm fallback do not emit Python warnings; the record already carries those facts in structured form. Experimental selection does not warn because it is explicit. Deprecation warnings name the replacement symbol and planned removal release. Compatibility warnings are reserved for legacy behavior that cannot be represented perfectly and may surprise a caller.

## E. Visualization API

Visualization is a consumer of a raw signal and an already-produced ECG Record. It cannot trigger measurement, mutate the record, or import the interpretation/rule engine. Every plotting call takes <code>(signal, record)</code> as its first two arguments so the distinction is visible in the API.

Matplotlib belongs to the optional <code>viz</code> extra. Importing <code>ecgfeat</code>, measuring records, loading records, or querying them must not import matplotlib. The <code>ecgfeat.viz</code> module loads plotting dependencies only when that module is imported. With the recommended distribution name, installation is <code>ecg-records[viz]</code>.

~~~python
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
~~~

~~~python
def plot_record(
    signal: ArrayLike,
    record: ECGRecord,
    *,
    leads: Sequence[str] | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    annotations: Literal["none", "beats", "waves", "all"] = "waves",
    figure: Figure | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot selected ECG leads with record-backed annotations and return the figure and axes."""
~~~

~~~python
def plot_beat(
    signal: ArrayLike,
    record: ECGRecord,
    *,
    beat: int,
    lead: str,
    figure: Figure | None = None,
) -> tuple[Figure, Axes]:
    """Plot one beat on one lead using boundaries and measurements from the record."""
~~~

~~~python
def plot_beat_all_leads(
    signal: ArrayLike,
    record: ECGRecord,
    *,
    beat: int,
    leads: Sequence[str] | None = None,
    figure: Figure | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot one beat across selected leads and return the figure and axes."""
~~~

~~~python
def plot_representative_beat(
    signal: ArrayLike,
    record: ECGRecord,
    *,
    lead: str,
    figure: Figure | None = None,
) -> tuple[Figure, Axes]:
    """Plot the record-selected representative beat for one lead."""
~~~

~~~python
def plot_quality_summary(
    signal: ArrayLike,
    record: ECGRecord,
    *,
    leads: Sequence[str] | None = None,
    figure: Figure | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot signal-quality measurements already present in the record."""
~~~

When <code>figure=None</code>, the function constructs a new <code>Figure</code> using the object-oriented matplotlib API. When a figure is supplied, axes are created on that figure. Functions never call <code>pyplot.gcf()</code>, <code>pyplot.show()</code>, or otherwise depend on global current-figure state. Displaying, saving, and closing the returned figure are caller responsibilities.

The signal must describe the same acquisition represented by the record: lead count/names, sample count where recorded, sampling rate, and input fingerprint must agree. A mismatch raises <code>VisualizationInputError</code>; an absent annotation is simply omitted and may be explained in the returned record query data.

~~~python
class VisualizationInputError(ECGInputError):
    """Signal data do not match the acquisition identity declared by the ECG Record."""
~~~

This intentionally rejects NeuroKit2's plotting convention of returning <code>None</code> after drawing into global matplotlib state. Returning figure and axes objects makes plotting composable in notebooks, services, tests, and noninteractive renderers while keeping plotting dependencies optional.

## F. CLI surface

The executable is <code>ecg-record</code>. It is a thin interface over the public Python API, and it never invokes interpretation.

~~~text
ecg-record measure SIGNAL
    --sampling-rate HZ
    --lead-names LEAD [LEAD ...]
    [--amplitude-unit {mV,uV}]
    [--input-mode {standard_12,limited}]
    [--method METHOD]
    [--profile {summary,all,debug}]
    [--config CONFIG.json]
    [--sidecar OUTPUT.npz]
    [-o OUTPUT.json]

ecg-record validate RECORD.json
    [--level {strict,schema}]

ecg-record query RECORD.json NAME
    [--lead LEAD]
    [--beat INDEX]

ecg-record resolve RECORD.json ADDRESS

ecg-record select RECORD.json
    --queries QUERIES.json
    [--format {json,jsonl}]
    [-o OUTPUT]

ecg-record batch MANIFEST.jsonl
    --output-dir DIR
    [--jobs N]
    [--fail-fast]
~~~

The command mapping is direct:

~~~python
def cli_measure(
    signal: str | Path,
    *,
    sampling_rate: float,
    lead_names: Sequence[str],
    amplitude_unit: AmplitudeUnit = "mV",
    input_mode: InputMode = "standard_12",
    method: AlgorithmName = "default",
    profile: RecordProfile = "summary",
    config: str | Path | None = None,
    sidecar: str | Path | None = None,
    output: str | Path = "-",
) -> int:
    """Measure one signal file and write one ECG Record."""

def cli_validate(
    record: str | Path,
    *,
    level: Literal["strict", "schema"] = "strict",
) -> int:
    """Validate one ECG Record and emit a machine-readable validation result."""

def cli_query(
    record: str | Path,
    name: str,
    *,
    lead: str | None = None,
    beat: int | None = None,
) -> int:
    """Query one measurement and emit one MeasurementResult JSON object."""

def cli_resolve(
    record: str | Path,
    address: str,
) -> int:
    """Resolve one canonical address and emit one AddressResolution JSON object."""

def cli_select(
    record: str | Path,
    *,
    queries: str | Path,
    format: Literal["json", "jsonl"] = "json",
    output: str | Path = "-",
) -> int:
    """Emit a selected set of measurement results."""

def cli_batch(
    manifest: str | Path,
    *,
    output_dir: str | Path,
    jobs: int = 1,
    fail_fast: bool = False,
) -> int:
    """Measure manifest entries and emit one JSONL status object per input."""
~~~

These <code>cli_*</code> signatures document command semantics; they are not re-exported from the Python package and are not supported as a programmatic subprocess replacement for the Python API.

### Signal and stdin/stdout contract

<code>measure</code> accepts <code>.npy</code> and <code>.npz</code> signal containers directly. A two-dimensional array is channel-major: shape <code>(n_leads, n_samples)</code>. The NPZ form uses a required <code>signal</code> array; sampling rate and lead names still come from flags or an explicit config/manifest entry so the CLI never silently trusts unrelated archive fields.

<code>SIGNAL=-</code> reads one binary NPY stream from stdin. In that mode, <code>--sampling-rate</code> and <code>--lead-names</code> are mandatory. Standard output is reserved for the ECG Record JSON when <code>-o -</code>, and all diagnostics go to stderr.

Record-consuming commands accept <code>RECORD.json=-</code> to read one UTF-8 ECG Record JSON document from stdin. <code>validate</code>, <code>query</code>, and <code>resolve</code> each write exactly one UTF-8 JSON object plus a trailing newline to stdout. <code>select --format=json</code> writes one JSON object; <code>--format=jsonl</code> writes one result object per line. No progress bars, warnings, or log prefixes are written to stdout.

<code>--config</code> is a UTF-8 JSON encoding of <code>ECGConfig</code>. CLI flags override the corresponding config-file field only for the explicitly supplied flag. The effective configuration, including defaults after resolution, is still serialized into record provenance.

### Batch contract

Each nonblank manifest line is one JSON object:

~~~json
{
  "id": "rec-123",
  "signal": "signals/rec-123.npy",
  "sampling_rate": 500.0,
  "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
  "amplitude_unit": "mV",
  "input_mode": "standard_12",
  "method": "default",
  "profile": "summary",
  "output": "rec-123.json"
}
~~~

Relative signal paths are resolved against the manifest's directory; for <code>MANIFEST=-</code>, they are resolved against the current working directory. Output paths are resolved under <code>--output-dir</code> and may not escape it. An item's record is written atomically after successful validation. Failed items leave no new final record file.

Batch processing continues after per-record failures by default and preserves manifest order in its stdout JSONL status stream even when <code>--jobs</code> is greater than one. Each status object contains the manifest <code>id</code>, output path if successful, and otherwise the stable error class and message. <code>--fail-fast</code> stops scheduling new items after the first observed failure; already-running items may finish.

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | Requested operation succeeded; for batch, every item succeeded. |
| 2 | CLI syntax or configuration error. |
| 3 | Invalid signal, record, address, measurement name, or selector. |
| 4 | Batch completed with one or more failed items. |
| 5 | Filesystem or stream I/O failure. |
| 70 | Unexpected internal failure or <code>ComputationInvariantError</code>. |

A physiological <code>unmeasurable</code>, <code>not_applicable</code>, or JSON <code>null</code> query result is successful and exits 0 because it is record data, not command failure.

## G. Typing and public-surface policy

The package ships <code>py.typed</code>. Public signatures use concrete types and standard typing primitives; runtime dependencies do not include a DataFrame library merely to provide annotations.

The top-level <code>ecgfeat.__all__</code> is the compatibility contract for the measurement distribution. New code should be able to discover the supported surface from that list and the documented public submodules.

~~~python
__all__ = [
    "ecg_record",
    "ecg_prepare",
    "ecg_measure",
    "ecg_emit",
    "ecg_dump",
    "load_record",
    "validate_record",
    "record_to_dict",
    "parse_address",
    "resolve_address",
    "query_measurement",
    "query_many",
    "select_measurements",
    "dump_measurements",
    "iter_leads",
    "iter_beats",
    "ECGRecord",
    "ECGInput",
    "ECGMeasurements",
    "RecordAddress",
    "AddressResolution",
    "MeasurementQuery",
    "MeasurementResult",
    "MeasurementSelection",
    "NullAbsence",
    "UnmeasurableAbsence",
    "NotApplicableAbsence",
    "MeasurementProvenance",
    "ValidationStatus",
    "ECGConfig",
    "RefinementConfig",
    "PWaveConfig",
    "PatientMeta",
    "STANDARD_12_LEADS",
    "ECGInputError",
    "SignalShapeError",
    "LeadNameError",
    "SamplingRateError",
    "AmplitudeUnitError",
    "ConfigurationError",
    "RecordValidationError",
    "RecordIdentityError",
    "AddressSyntaxError",
    "AddressNotFoundError",
    "MeasurementNotFoundError",
    "MeasurementSelectorError",
    "ComputationInvariantError",
    "ECGWarning",
    "ECGCompatibilityWarning",
    "ECGDeprecationWarning",
]
~~~

Serialization to an ordinary mapping is explicit and does not re-run profile selection.

~~~python
def record_to_dict(record: ECGRecord) -> dict[str, JsonValue]:
    """Return a detached JSON-compatible mapping for an already-emitted ECG Record."""
~~~

The following public submodules have their own explicit <code>__all__</code>: <code>ecgfeat.record</code>, <code>ecgfeat.query</code>, <code>ecgfeat.config</code>, <code>ecgfeat.io</code>, and <code>ecgfeat.errors</code>. <code>ecgfeat.viz</code> is public only when the visualization extra is installed and is intentionally not re-exported at top level.

WFDB decoding remains available as typed I/O rather than as top-level measurement behavior.

~~~python
from numpy.typing import NDArray

@dataclass(frozen=True, slots=True)
class SignalData:
    values: NDArray[np.float64]
    sampling_rate: float
    lead_names: tuple[str, ...]
    amplitude_unit: AmplitudeUnit

@dataclass(frozen=True, slots=True)
class WFDBHeader:
    sampling_rate: float
    sample_count: int
    lead_names: tuple[str, ...]
    amplitude_units: tuple[str, ...]
    metadata: Mapping[str, str]

def read_wfdb(
    signal_path: str | Path,
    *,
    header_path: str | Path | None = None,
) -> SignalData:
    """Read and calibrate a supported WFDB MAT/header pair into typed signal data."""

def read_wfdb_header(
    source: str | Path | TextIO,
) -> WFDBHeader:
    """Parse supported WFDB header fields into an immutable typed header."""
~~~

<code>ecgfeat.io.__all__</code> contains <code>SignalData</code>, <code>WFDBHeader</code>, <code>read_wfdb</code>, and <code>read_wfdb_header</code>. They are not top-level re-exports because format-specific I/O should not dominate completion for the measurement API.

### What is not public

A symbol is not public merely because it is importable. Internal algorithm classes, detector candidates, repair deltas, clustering details, experimental bypass objects, vendor-compatibility duplicates, schema implementation helpers, private dataclasses, and intermediate score types have no compatibility guarantee unless they are explicitly listed in a documented module's <code>__all__</code>.

The <code>debug</code> profile exposes internal debug data in a record for troubleshooting, but it does not make corresponding Python implementation types public and does not grant stable-address guarantees to those debug fields.

No object from the separate interpretation/rule-engine distribution is re-exported by the new measurement surface. The deprecated top-level <code>interpret</code> shim is the sole temporary exception, and it performs a lazy import only when called.

### Re-export rules

A new public symbol has one defining public submodule and at most one convenience re-export from <code>ecgfeat</code>. Re-exporting the same concept under multiple names is prohibited. Optional-dependency APIs are never imported by <code>ecgfeat.__init__</code>. The canonical documentation names the defining module even when the top-level re-export is convenient.

The package uses postponed annotations or typing-only imports where needed so annotations cannot make optional dependencies mandatory at runtime.

### Deprecation mechanism

The current surface has real consumers, including 40 test files importing <code>models</code> and 39 importing <code>export</code>, so migration uses compatibility modules rather than deleting names in one release.

The policy is concrete:

1. The first migration release adds the new API and keeps <code>ecgfeat.models</code>, <code>ecgfeat.export</code>, and every currently exported top-level name importable.
2. A legacy function or constructor emits <code>ECGDeprecationWarning</code> on use, not merely on module import. The warning names the replacement and the earliest removal release.
3. Plain import of <code>ecgfeat.models</code> or <code>ecgfeat.export</code> is silent so test collection and type checking do not produce warning floods.
4. Legacy aliases remain for at least two consecutive minor releases and at least six months. Removal occurs no earlier than the next major release after both conditions are met.
5. Compatibility shims may adapt old objects to the new record boundary, but they must not introduce pandas, matplotlib, or rule-engine dependencies into ordinary measurement imports.
6. Type stubs mark legacy names with <code>typing_extensions.deprecated</code> when available, matching the runtime warning message.
7. During the compatibility window, legacy names remain in top-level <code>__all__</code> only when they are currently top-level exports. They are grouped and documented as deprecated. At removal, they leave <code>__all__</code> and the compatibility modules stop providing them.

This duration is intentionally longer than one minor release because the import counts show broad in-repository coupling. It is still bounded: compatibility aliases are not a permanent second API.

## H. Compatibility shims

The table below maps every current top-level export to the replacement contract. A retained symbol means its role still belongs in the measurement package; a shim means the old call remains temporarily but forwards to the stated replacement.

| Current export | Replacement | Compatibility behavior |
|---|---|---|
| <code>ECGFeatureExtractor</code> | <code>ecg_record()</code>; advanced callers use <code>ecg_prepare()</code> + <code>ecg_measure()</code> + <code>ecg_emit()</code> | Deprecated class remains as a thin adapter that stores <code>ECGConfig</code> and forwards <code>extract()</code> to the new pipeline. |
| <code>RefinementConfig</code> | <code>ecgfeat.RefinementConfig</code> in <code>ECGConfig.refinement</code> | Retained as a frozen public configuration type; mutable legacy construction is normalized with a deprecation warning if necessary. |
| <code>PWaveConfig</code> | <code>ecgfeat.PWaveConfig</code> in <code>ECGConfig.p_wave</code> | Retained as a frozen public configuration type. |
| <code>ECGInputError</code> | <code>ecgfeat.ECGInputError</code> | Retained as the public exception root. |
| <code>interpret</code> | <code>ecginterpret.interpret_record(record: ECGRecord) -&gt; InterpretationDocument</code> | Deprecated shim lazily imports the separate interpretation distribution only when called; absence of that distribution raises an installation-focused <code>ImportError</code>. |
| <code>to_dict</code> | <code>record_to_dict(record: ECGRecord) -&gt; dict[str, JsonValue]</code> | Deprecated shim accepts legacy feature objects during the compatibility window; new code serializes an <code>ECGRecord</code>. |
| <code>load_wfdb_mat</code> | <code>ecgfeat.io.read_wfdb()</code> | Deprecated top-level shim forwards to typed WFDB I/O. |
| <code>parse_wfdb_header</code> | <code>ecgfeat.io.read_wfdb_header()</code> | Deprecated top-level shim forwards to typed WFDB I/O. |
| <code>plot_beat</code> | <code>ecgfeat.viz.plot_beat(signal, record, ...)</code> | Deprecated top-level shim lazily imports the visualization extra; new function returns figure and axes. |
| <code>plot_beat_all_leads</code> | <code>ecgfeat.viz.plot_beat_all_leads(signal, record, ...)</code> | Deprecated top-level shim lazily imports the visualization extra. |
| <code>plot_rep_beat</code> | <code>ecgfeat.viz.plot_representative_beat(signal, record, ...)</code> | Deprecated rename shim; returns figure and axes. |
| <code>plot_quality_summary</code> | <code>ecgfeat.viz.plot_quality_summary(signal, record, ...)</code> | Deprecated top-level shim lazily imports the visualization extra. |
| <code>ECGFeatures</code> | <code>ECGRecord</code> | Legacy object remains readable by export shims; new measurement calls return the schema-backed record boundary. |
| <code>GlobalFeatures</code> | <code>query_measurement()</code> / <code>query_many()</code> over global-axis measurements | No new aggregate model class; legacy wrapper reads corresponding record measurements. |
| <code>LeadBeatFeatures</code> | <code>BeatView</code> plus <code>query_measurement(..., lead=..., beat=...)</code> | Legacy wrapper becomes a read-only adapter over record queries. |
| <code>RepresentativeLeadFeatures</code> | <code>query_many()</code> for the representative-beat measurement names | No dedicated replacement class; the stable contract is the record fields. |
| <code>GroupFeatures</code> | <code>MeasurementSelection</code> | Legacy grouping adapter produces a selected record view; group implementation types are not preserved. |
| <code>LeadQuality</code> | <code>MeasurementResult</code> for published lead-quality quantities | No dedicated quality dataclass; unit/status/provenance come from the query result. |
| <code>PWaveBeatAssessment</code> | P-wave measurement queries plus <code>MeasurementProvenance</code> | No dedicated public assessment model; unpublished scoring details move to <code>debug</code>. |
| <code>PWaveLeadBoundary</code> | Published P-wave onset/offset measurement queries | No dedicated public boundary class. |
| <code>PatientMeta</code> | <code>ecgfeat.PatientMeta</code> | Retained as a frozen input metadata type accepted by <code>ecg_record()</code> and <code>ecg_prepare()</code>. |
| <code>QRSDetectorResult</code> | Published R-peak/QRS measurements plus <code>MeasurementProvenance</code> | Detector implementation result is no longer public; legacy wrapper exposes only published fields. |
| <code>QRSCandidateWindow</code> | <code>debug</code>-profile data only | No stable Python replacement and no stable record address; compatibility type is deprecated and removed after the migration window. |
| <code>WaveBounds</code> | Onset/peak/offset measurements queried from <code>ECGRecord</code> | No dedicated public class; absence semantics are represented per bound. |
| <code>ECGInterpretation</code> | <code>ecginterpret.InterpretationDocument</code> | Deprecated alias is provided only through the lazy interpretation shim and leaves the measurement package at removal. |
| <code>STANDARD_12_LEADS</code> | <code>ecgfeat.STANDARD_12_LEADS</code> | Retained as an immutable <code>tuple[str, ...]</code>. |

The separate interpretation package's replacement entry point is fixed as:

~~~python
def interpret_record(
    record: ECGRecord,
) -> InterpretationDocument:
    """Produce a separately versioned interpretation document from an ECG Record."""
~~~

It is intentionally record-only. The rule engine consumes published measurements and their absence/provenance contract rather than reaching into measurement internals. If it needs raw signal for a future rule, that is a new explicit interpretation-package API rather than an implicit back-reference.

<code>STANDARD_12_LEADS</code> has the stable type:

~~~python
STANDARD_12_LEADS: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
~~~

### Open questions

1. **Interpretation distribution publication name.** The import namespace in this design is <code>ecginterpret</code>, but the separately distributed package's PyPI/project name is not settled by the supplied decisions. Pick that publication name before compatibility warnings contain install instructions.
2. **~~Historical ST source~~ — RESOLVED.** The stable source is <code>"analysis"</code> (<code>api.py:1771</code>); the full set is <code>{"analysis", "calibrated_pr", "adaptive_pr_tp"}</code> (<code>api.py:1778</code>). Remaining sub-question: whether <code>enable_hybrid_st_measurement</code> (<code>api.py:1783</code>), which both non-default sources require, stays a separate public flag or is folded into the source selection so an invalid combination cannot be constructed.
3. **~~NPZ sidecar locator and integrity fields~~ — RESOLVED.** Relative same-directory file name only (no scheme, no path separators), SHA-256 of the exact bytes, record id and schema version bound on both sides, and a `uint8` state mask checked against the JSON absence encoding; see document 02 D.5. `load_record(path)` resolves the sidecar next to the record (or takes `sidecar=` bytes/path); the query API is unchanged.
4. **Validation evidence vocabulary.** This design fixes the query shape and the three tiers <code>validated</code>, <code>partially_validated</code>, and <code>unvalidated</code>. The final controlled vocabulary for the optional <code>ValidationStatus.evidence</code> field still needs to be aligned with the validation program so callers do not parse free-form prose.
