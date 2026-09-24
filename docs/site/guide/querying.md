# Reading and querying records

Records can be read without the signal and without the numerical engine.
Loading and querying a record imports neither NumPy-heavy measurement code nor
plotting code, so it is cheap in services and notebooks.

## Loading

```python
from ecgfeat import load_record

record = load_record("ecg.record.json")                  # strict validation (default)
record = load_record("ecg.record.json", validate="schema")
```

`load_record(source, *, validate="strict", sidecar=None)` accepts a path, a
text or binary stream, or an already parsed mapping.

| `validate` | Checks |
|---|---|
| `"strict"` (default) | JSON Schema structure **and** cross-field invariants: axis shapes, beat and lead coordinates, fiducial order, units, validation claims no higher than the registry allows, and sidecar integrity |
| `"schema"` | structure only; sidecars are verified lazily on first access |
| `"none"` | parse only; for trusted, already validated data |

A record that fails validation raises `RecordValidationError`, with a `code`
such as `invalid_record_contract` or `validation_above_registry`. JSON with
duplicate members is rejected (`duplicate_json_member`).

`validate_record(record_or_mapping, level="strict")` validates an in-memory
record or mapping and returns an `ECGRecord`.

## One value: `query_measurement`

```python
from ecgfeat import load_record, query_measurement

record = load_record("ecg.record.json")
result = query_measurement(record, "qt_interval_ms", lead="V5", beat=2)

result.value        # e.g. 418 (int), or None when missing
result.unit         # 'ms'
result.lead         # 'V5'
result.beat         # 2
result.absence      # None when measured, else NullAbsence / UnmeasurableAbsence / NotApplicableAbsence
result.validation   # ValidationStatus(tier='unvalidated', evidence=None)
result.provenance   # MeasurementProvenance(method_requested='default', ...)
str(result.address) # 'ecg-record:rec-…@1.0.0#/measurements/intervals/qt_interval_ms/values/2/10'
```

- `name` is the field name without its group: `qrs_duration_ms`, `r_peak`,
  `heart_rate_bpm`, … (all names are in the
  [published fields](record-format.md#published-fields) table).
- **Per-beat, per-lead fields need both `lead` and `beat`.** `beat` is the
  zero-based position in `record.axes["beats"]`, and `lead` is a name from
  `record.axes["leads"]`.
- **Record-level fields** (`heart_rate_bpm`, `frontal_qrs_axis_deg`,
  `qtc_bazett_ms`, `qrs_wide_ms`) take neither.
- A field absent from the record's profile (for example `jt_interval_ms` in a
  `summary` record) raises `MeasurementNotFoundError`; so does an unknown name.
- A missing or unknown selector raises `MeasurementSelectorError`
  (`invalid_lead_selector` or `invalid_beat_selector`).

### Missing values

A missing value gives `value=None` plus its resolved absence state, whatever
compact [encoding](record-format.md#absence-states) the record uses:

```python
from ecgfeat import load_record, query_measurement, UnmeasurableAbsence, NotApplicableAbsence

record = load_record("ecg.record.json")
result = query_measurement(record, "p_duration_ms", lead="aVR", beat=0)

if result.absence is None:
    print("measured:", result.value, result.unit)
elif isinstance(result.absence, UnmeasurableAbsence):
    print("no trustworthy value:", result.absence.reason)      # e.g. measurement_unavailable
elif isinstance(result.absence, NotApplicableAbsence):
    print("does not apply:", result.absence.reason)            # e.g. input_mode_limited
else:
    print("missing, no reason given")                          # NullAbsence (plain null)
```

Every absence object has `kind` (`"null"`, `"unmeasurable"` or
`"not_applicable"`); the last two also have `reason`.

## Many values: `query_many` and selections

```python
from ecgfeat import load_record, MeasurementQuery, query_many

record = load_record("ecg.record.json")
queries = [MeasurementQuery("heart_rate_bpm")] + [
    MeasurementQuery("qrs_duration_ms", lead=lead, beat=0) for lead in record.axes["leads"]
]
for result in query_many(record, queries):
    print(result.name, result.lead, result.value, result.unit)
```

`query_many` returns results in input order. To export a set of values with
their units, absence states and provenance, build a selection and dump it:

```python
from ecgfeat import load_record, MeasurementQuery, select_measurements, dump_measurements

record = load_record("ecg.record.json")
selection = select_measurements(record, [
    MeasurementQuery("heart_rate_bpm"),
    MeasurementQuery("qrs_duration_ms", lead="V1", beat=0),
])
dump_measurements(selection, "selection.json")                     # one JSON object
dump_measurements(selection, "selection.jsonl", format="jsonl")    # one JSON line per result
```

The JSON form is `{"record_id": ..., "results": [...]}`. Each result carries
`name`, `lead`, `beat`, `value`, `unit`, `absence`, `address`, `provenance` and
`validation`, so an exported number never loses its context.

## A table of values

A per-beat, per-lead field as a NumPy array, with `NaN` for missing cells:

```python
import numpy as np
from ecgfeat import load_record, query_measurement

record = load_record("ecg.record.json")
leads = record.axes["leads"]
beats = range(len(record.axes["beats"]))
qrs = np.array([[query_measurement(record, "qrs_duration_ms", lead=lead, beat=b).value
                 for lead in leads] for b in beats], dtype=float)   # None -> nan
print(qrs.shape)                                                    # (n_beats, n_leads)
print(np.nanmedian(qrs, axis=0))                                    # median per lead
```

The same data, without per-cell absence resolution, is directly in the record:
`record.field("/measurements/intervals/qrs_duration_ms")["values"]` is the
`[beat][lead]` matrix (`None` for missing cells). For sidecar-backed records,
use the query API or `materialize_record` first ([sidecars](serialization.md#dense-sidecars)).

## Iterating leads and beats

```python
from ecgfeat import load_record, iter_leads, iter_beats

record = load_record("ecg.record.json")
for lead in iter_leads(record):
    print(lead.index, lead.name)                    # 0 I, 1 II, …
for beat in iter_beats(record):
    print(beat.index, beat.sample, beat.time_s)     # 0 250 0.5, …
```

`iter_beats(record, lead="II")` yields the same beats with `beat.lead` set.
Beat `sample` is the beat's reference R sample in raw-signal coordinates, and
`time_s` is that sample divided by the sampling rate.

## Addresses

Every value has a canonical address:

```text
ecg-record:<record_id>@<schema_version>#<RFC 6901 JSON pointer>
```

For example,
`ecg-record:rec-5f0b…@1.0.0#/measurements/intervals/qrs_duration_ms/values/3/6`
is beat index 3, lead index 6 (`V1` in canonical order) of the QRS duration.
Addresses are stable: the same record always gives the same addresses. That
makes them suitable for citing values in reports, annotations or model
outputs.

```python
from ecgfeat import load_record, parse_address, resolve_address

record = load_record("ecg.record.json")
address = f"ecg-record:{record.record_id}@{record.schema_version}#/quality/record"

parsed = parse_address(address)       # RecordAddress(record_id=..., schema_version='1.0.0', pointer='/quality/record')
resolved = resolve_address(record, address)
print(resolved.node_kind, dict(resolved.value))   # record {'grade': 'Q0', 'status': 'usable'}
```

`resolve_address` first checks that the address names **this** record and a
compatible schema version, then resolves the pointer. Any pointer into the
document works. `node_kind` says what was reached: `measurement`, `record`,
`metadata`, `provenance` or `debug`.

| Error | When |
|---|---|
| `AddressSyntaxError` | the address or pointer is malformed |
| `RecordIdentityError` | the address names another record or an incompatible schema version |
| `AddressNotFoundError` | the pointer is valid but does not exist in this record |

## Direct access

`ECGRecord` exposes its members as read-only attributes, and
`record.field(pointer)` returns the node at a JSON pointer:

```python
from ecgfeat import load_record

record = load_record("ecg.record.json")
print(record.acquisition["sample_rate_hz"], record.quality["record"]["grade"])
print(record.field("/measurements/global/heart_rate_bpm")["values"])
document = record.as_dict()          # detached, mutable plain-Python copy
```

Prefer `query_measurement` for values: it resolves absence encodings and
sidecars, and returns units and validation status with the number.
