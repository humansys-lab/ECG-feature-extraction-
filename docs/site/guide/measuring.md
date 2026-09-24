# Measuring ECGs

## `ecg_record`: signal in, record out

```text
ecg_record(ecg, *, sampling_rate, lead_names, amplitude_unit="mV", patient=None,
           method="default", profile="summary", config=None, input_mode=None,
           record_id=None) -> ECGRecord
```

| Argument | Meaning |
|---|---|
| `ecg` | channel-major array `(n_leads, n_samples)`; see [Preparing input signals](input.md) |
| `sampling_rate` | samples per second of `ecg` (≥ 100) |
| `lead_names` | one explicit name per row |
| `amplitude_unit` | `"mV"` or `"uV"` |
| `patient` | optional mapping or `PatientMeta` ([details](input.md#patient-metadata)) |
| `method` | `"default"` or `"legacy_dxl_inspired"`; both select the same algorithm (recorded as `method_resolved = "legacy_dxl_inspired"`) |
| `profile` | `"summary"` (default), `"all"` or `"debug"` ([profiles](#profiles)) |
| `config` | optional [`ECGConfig`](configuration.md) |
| `input_mode` | `"standard_12"` or `"limited"`; defaults to `config.input_mode`, or `"standard_12"` without a config |
| `record_id` | optional override of the generated id (see [Record identity](#record-identity)) |

The call runs the whole pipeline and returns an immutable, validated
[`ECGRecord`](record-format.md). It raises an [`ECGInputError`](errors.md)
subclass when the input or configuration is invalid, and
`ComputationInvariantError` in the rare case that no valid record can be built.

### What the pipeline does

The pipeline runs eight stages in a fixed order:

| Stage | Work |
|---|---|
| input | validate, normalize units, resample to the internal rate, record the input contract |
| quality | per-lead and record signal quality, acquisition checks, lead-reversal screening |
| ventricular | multilead QRS detection and pacing-spike handling |
| beats | beat grouping (dominant rhythm, ectopy), representative beats |
| delineation | P, QRS and T boundaries and peaks per beat and lead; QRS-tail and T refinement |
| atrial | atrial activity and P-wave assessment |
| measurement | intervals, amplitudes, areas, ST level, global quantities |
| finalize | consensus values, invariant checks, record assembly |

Four policies make the pipeline's contextual decisions explicit and appear in
the provenance: pacing, QT rescue, measurement applicability and lead
integrity. See the [architecture page](../development/architecture.md) for the
design.

### Determinism

The pipeline has no randomness. In a given environment, the same signal,
settings, patient metadata and library version give byte-identical records.
Published values are integers (samples, milliseconds, microvolts), so the
last-digit floating-point differences that can occur between NumPy versions or
platforms normally never reach them. On the 156-record golden sentinel set,
NumPy 2.2 and 2.5 produced identical `summary` and `all` records; only one
`debug`-profile diagnostic differed, in its 17th significant digit. The optional
Numba kernel is tested to match the pure-Python path exactly.

## Profiles

| Profile | Contents | Size |
|---|---|---|
| `summary` (default) | acquisition, axes, quality, provenance; 7 fiducials, 5 intervals, 5 amplitudes, 1 area, heart rate and frontal QRS axis | ≤ 24,000 bytes for a 10 s, 12-lead record with 10 beats (contract-tested) |
| `all` | everything published: adds the JT interval, U amplitude, QTc (Bazett) and wide-QRS duration | no budget |
| `debug` | `all` plus selected engine metadata under `/debug`; **no compatibility guarantee** | no budget |

`summary ⊂ all ⊂ debug`: a value present in two profiles is identical in both.
The profile does not change the `record_id`. Pick `summary` for storage and
exchange, `all` when you need every published quantity, and `debug` only for
investigating the pipeline itself. `/debug` content may change in any release.

```python
import numpy as np
from ecgfeat import ecg_record, dumps_record

signal = np.load("ecg.npy")
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
summary = ecg_record(signal, sampling_rate=500, lead_names=leads)
full = ecg_record(signal, sampling_rate=500, lead_names=leads, profile="all")

print(len(dumps_record(summary)), len(dumps_record(full)))     # e.g. 18061 20052
print(summary.record_id == full.record_id)                     # True
print(sorted(full.measurements["global"]))                     # includes 'qtc_bazett_ms'
```

## Record identity

Unless you pass `record_id`, the id is `rec-` followed by a SHA-256 over:
the input fingerprint (samples, rate, names, unit), the resolved configuration
hash, the patient metadata, the resolved method, the library version and the
schema version. Consequences:

- re-running the same input gives the same id, so ids work as cache keys;
- a new library version gives a new id, because it may measure differently;
- the profile is not part of the id.

Pass `record_id="my-study/0001"` to use your own identifier. It is used in
every [address](querying.md#addresses) of that record.

## Staged API

`ecg_record` is three public stages chained together. Call them separately to
validate input once, measure once and emit several profiles:

```python
import numpy as np
from ecgfeat import ecg_prepare, ecg_measure, ecg_emit

signal = np.load("ecg.npy")
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

prepared = ecg_prepare(signal, sampling_rate=500, lead_names=leads)   # validation only
measured = ecg_measure(prepared)                                      # the expensive part
summary = ecg_emit(measured, profile="summary")
debug = ecg_emit(measured, profile="debug")
print(summary.record_id == debug.record_id)                           # True
```

| Function | Returns | Notes |
|---|---|---|
| `ecg_prepare(ecg, *, sampling_rate, lead_names, amplitude_unit="mV", patient=None, input_mode="standard_12")` | `ECGInput` | immutable; holds a private read-only copy of the samples |
| `ecg_measure(prepared, *, method="default", config=None)` | `ECGMeasurements` | `config.input_mode` must match `prepared.input_mode` |
| `ecg_emit(measurements, *, profile="summary")` | `ECGRecord` | cheap; validates the record |
| `ecg_dump(record, destination, *, indent=None)` | `None` | writes UTF-8 JSON to a path or text stream |

`ECGMeasurements` is an intermediate object: keep it in memory, do not persist
it. Its attributes are not part of the stable API.

## Reusing a configuration: `ECGRecordExtractor`

For many recordings with one configuration, `ECGRecordExtractor` holds the
config and exposes a positional `extract` method:

```python
import numpy as np
from ecgfeat import ECGConfig, ECGRecordExtractor

extractor = ECGRecordExtractor(ECGConfig(mains_frequency_hz=60))
signal = np.load("ecg.npy")
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
record = extractor.extract(signal, 500, leads, record_id="site-a/0001")
print(record.record_id)                                   # site-a/0001
```

The extractor keeps no state between calls and can be shared between threads.

## Performance

One 10 s, 12-lead record at 500 Hz took about 2.3 s in a warm process on the
reference machine (a 16-core Linux x86-64 host under full load). The first call
in a new process took about 3.9 s, because modules load, and peak memory was
about 240 MB. The repository gates these numbers with a performance budget
(`tools/check_performance.py`, `benchmarks/performance/budget.json`).

For many records:

- use the [`ecg-record batch`](cli.md#batch) command, which processes a
  manifest with a bounded worker pool; or
- run `ecg_record` in a `concurrent.futures.ProcessPoolExecutor`. The work is
  CPU-bound NumPy/SciPy code, so processes scale better than threads;
- install `ecg-records[performance]` for Numba-accelerated kernels.
