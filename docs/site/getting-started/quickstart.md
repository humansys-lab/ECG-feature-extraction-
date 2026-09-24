# Quickstart

This page takes one 10-second, 12-lead recording from a NumPy file to a stored,
queried and plotted ECG Record. It assumes `pip install "ecg-records[viz]"`.

## 1. Load a signal

`ecg-records` takes a **channel-major** array, one row per lead, with an
explicit name for every row. Amplitudes default to millivolts.

```python
import numpy as np

signal = np.load("ecg.npy")     # shape (12, 5000): 12 leads, 10 s at 500 Hz
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
print(signal.shape, signal.dtype)
```

If your data is sample-major, shaped `(n_samples, n_leads)` (common in CSV
files and in `wfdb`), transpose it with `signal.T`.
[Preparing input signals](../guide/input.md) covers units, lead order, WFDB
files and fewer than 12 leads.

## 2. Measure

```python
from ecgfeat import ecg_record

record = ecg_record(signal, sampling_rate=500, lead_names=leads)

print(record.record_id)          # rec-5f0b3c...: derived from the signal and settings
print(record.schema_version)     # 1.0.0
print(record.profile)            # summary
print(len(record.axes["beats"]), "beats")
```

`ecg_record` returns an immutable, already validated
[`ECGRecord`](../guide/record-format.md). With the same input and settings, the
record, and its `record_id`, are identical on every run.

## 3. Read values

```python
from ecgfeat import query_measurement

hr = query_measurement(record, "heart_rate_bpm")
print(hr.value, hr.unit)                     # 60 1/min

qrs = query_measurement(record, "qrs_duration_ms", lead="V1", beat=3)
print(qrs.value, qrs.unit)                   # e.g. 160 ms
print(qrs.absence)                           # None: the value was measured
print(qrs.validation.tier)                   # unvalidated
print(qrs.address)                           # ecg-record:rec-…@1.0.0#/measurements/intervals/qrs_duration_ms/values/3/6
```

Per-beat, per-lead measurements need a `lead` and a `beat` (a zero-based index
into `record.axes["beats"]`). Record-level measurements such as the heart rate
take neither. When a value is missing, `value` is `None` and `absence` says
why; see [Reading and querying records](../guide/querying.md).

## 4. Save and load

```python
from ecgfeat import dumps_record, load_record

with open("ecg.record.json", "wb") as handle:
    handle.write(dumps_record(record))       # canonical UTF-8 JSON bytes

restored = load_record("ecg.record.json")    # strict validation by default
assert dumps_record(restored) == dumps_record(record)
```

Reading a record needs neither the signal nor the numerical engine. Anyone
with the JSON file can query it.

## 5. Plot

```python
import matplotlib
matplotlib.use("Agg")                        # headless; omit in a notebook

from ecgfeat.viz import plot_record

fig, axes = plot_record(signal, record, leads=["II", "V1", "V5"],
                        start_s=1.0, end_s=4.0, annotations="all")
fig.savefig("record.png", dpi=150)
```

The plot shows the raw signal with the record's P, QRS and T landmarks. The
signal must be the exact array that produced the record (its SHA-256 is
checked). See [Plotting](../guide/plotting.md).

## 6. The same from a shell

```bash
ecg-record measure ecg.npy --sampling-rate 500 \
    --lead-names I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 -o ecg.record.json
ecg-record validate ecg.record.json
ecg-record query ecg.record.json qrs_duration_ms --lead V1 --beat 3
```

See [Command line](../guide/cli.md) for every subcommand, including parallel
batch processing.

## Next steps

- [The ECG Record](../guide/record-format.md): what each member means
- [Measuring ECGs](../guide/measuring.md): profiles, the staged API, reuse
- [Limitations and intended use](../limitations.md): read this before relying on
  any value
