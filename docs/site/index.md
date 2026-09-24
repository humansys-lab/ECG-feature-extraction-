# ecg-records

**ecg-records** measures 12-lead (or 1–8-channel) electrocardiograms and emits a
small, versioned JSON document, the **ECG Record**. It contains fiducial points,
intervals, amplitudes, areas, signal quality and full provenance, all in
raw-sample coordinates. Every value has an explicit unit, every missing value
carries an explicit reason, every value has a stable address, and every field
carries a validation status.

The Python import name is `ecgfeat`.

> This software is research and engineering software. It is not a medical
> device and is not intended to diagnose, treat, cure, or prevent disease.
> Outputs require independent validation for the intended use and must not be
> used as a substitute for professional medical judgment.
> See [Limitations and intended use](limitations.md).

```bash
pip install ecg-records            # NumPy + SciPy only
pip install "ecg-records[viz]"     # adds Matplotlib for ecgfeat.viz
```

```python
import numpy as np
from ecgfeat import ecg_record, query_measurement

leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
signal = np.load("ecg.npy")                       # shape (12, n_samples), millivolts
record = ecg_record(signal, sampling_rate=500, lead_names=leads)

qrs = query_measurement(record, "qrs_duration_ms", lead="V1", beat=0)
print(qrs.value, qrs.unit, qrs.validation.tier)   # e.g. 154 ms unvalidated
```

## What you get

- **One call from signal to record.** [`ecg_record`](guide/measuring.md) runs
  signal-quality assessment, multilead QRS detection, beat grouping,
  representative beats, delineation, atrial analysis and measurement. The
  result is a validated, immutable record.
- **A published, versioned format.** The [ECG Record](guide/record-format.md)
  (schema `1.0.0`) ships with a JSON Schema. Records are canonical JSON: the
  same input always gives the same bytes. The default `summary` profile of a
  10 s, 12-lead record stays under 24,000 bytes.
- **Honest missing values.** A value is either measured, plain `null`,
  `unmeasurable(reason)` or `not_applicable(reason)`. See
  [Absence states](guide/record-format.md#absence-states).
- **Stable addresses.** Every value has an address such as
  `ecg-record:<id>@1.0.0#/measurements/intervals/qrs_duration_ms/values/3/6`
  that you can cite and resolve later ([Querying](guide/querying.md)).
- **Provenance and validation status on every field**, so no value looks more
  trustworthy than it is ([Validation](validation.md)).
- **Plotting** of `(signal, record)` pairs with [`ecgfeat.viz`](guide/plotting.md).
  Matplotlib stays optional: `import ecgfeat` never loads it.
- **A command line** for measuring, validating, querying and batch processing
  ([CLI](guide/cli.md)).

## Where to go next

| If you want to… | Read |
|---|---|
| install the package | [Installation](getting-started/installation.md) |
| measure your first ECG in five minutes | [Quickstart](getting-started/quickstart.md) |
| know what the signal must look like | [Preparing input signals](guide/input.md) |
| understand every member of a record | [The ECG Record](guide/record-format.md) |
| read values, iterate beats and leads, resolve addresses | [Reading and querying records](guide/querying.md) |
| store records, use NPZ sidecars | [Saving and loading](guide/serialization.md) |
| draw waveforms with annotations | [Plotting](guide/plotting.md) |
| process many files from a shell | [Command line](guide/cli.md) |
| tune the pipeline | [Configuration](guide/configuration.md) |
| handle errors programmatically | [Errors and warnings](guide/errors.md) |
| move off the legacy `ECGFeatureExtractor` API | [Migration guide](migration.md) |
| look up a signature | [API reference](reference/api-ecgfeat.md) |
| work on the library itself | [Developer guide](development/architecture.md) |

## Package facts

| | |
|---|---|
| Distribution | `ecg-records` on PyPI |
| Import name | `ecgfeat` |
| Current release | 0.1.0 |
| ECG Record schema | 1.0.0 (reads every `1.*`) |
| Python | 3.10 – 3.13 |
| Runtime dependencies | NumPy ≥ 1.26, SciPy ≥ 1.11 |
| Optional extras | `viz` (Matplotlib), `performance` (Numba), `wfdb` (WFDB reader), `interpret` (the separate `ecginterpret` distribution, not yet released) |
| License | Apache-2.0 |
