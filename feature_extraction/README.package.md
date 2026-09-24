# ecg-records

`ecg-records` measures 12-lead (or 1–8-channel) ECG signals and emits a small,
versioned JSON document, the **ECG Record**: fiducials, intervals,
amplitudes, areas, quality and provenance in raw-sample coordinates, with
explicit units, an explicit reason for every missing value, a stable address
for every value, and a per-field validation status. The Python import name is
`ecgfeat`.

> This software is research and engineering software. It is not a medical
> device and is not intended to diagnose, treat, cure, or prevent disease.
> Outputs require independent validation for the intended use and must not be
> used as a substitute for professional medical judgment.

The measurement pipeline (signal quality, multilead QRS detection, beat
grouping, representative beats, delineation) is inspired by publicly
documented 12-lead measurement practice. It is not a reproduction of any
proprietary product.

## Install

```bash
pip install ecg-records                  # measurements and ECG Records (NumPy, SciPy)
pip install "ecg-records[performance]"   # optional Numba acceleration
pip install "ecg-records[wfdb]"          # WFDB readers
pip install "ecg-records[viz]"           # plotting (ecg-records-viz, Matplotlib)
pip install "ecg-records[interpret]"     # rule-based interpretation (ecginterpret)
```

Python 3.10–3.13. The core package depends only on NumPy and SciPy.

## Measure

```python
import numpy as np
from ecgfeat import ecg_record, dumps_record

leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
signal = np.load("ecg.npy")        # shape (12, n_samples), channel-major, millivolts
record = ecg_record(signal, sampling_rate=500, lead_names=leads)   # profile="summary"
with open("ecg.record.json", "wb") as handle:
    handle.write(dumps_record(record))
```

Rows must be named explicitly; for fewer than 12 channels pass
`input_mode="limited"` (axis and formal QT are then `not_applicable`, raw
measurements stay available). Profiles: `summary` (default; under 24,000 bytes
for a 10 s, 12-lead, 10-beat record), `all` (every published measurement) and
`debug`. Dense matrices can move to a verified NPZ sidecar with
`serialize_record(record, sidecar_uri="ecg.dense.npz")`.

## Read and query

Reading records needs no NumPy, SciPy or plotting code:

```python
from ecgfeat import load_record, query_measurement, resolve_address

record = load_record("ecg.record.json")              # strict validation by default
qrs = query_measurement(record, "qrs_duration_ms", lead="V1", beat=3)
qrs.value, qrs.unit, qrs.absence, qrs.validation.tier, str(qrs.address)
# absence: None (measured), NullAbsence, UnmeasurableAbsence(reason) or NotApplicableAbsence(reason)

address = f"ecg-record:{record.record_id}@{record.schema_version}#/measurements/global/heart_rate_bpm/values"
resolve_address(record, address).value
```

Addresses follow `ecg-record:<record_id>@<schema_version>#<RFC 6901 pointer>`.
The JSON Schema and the validation-evidence registry ship in the package
(`ecgfeat/schemas/ecg-record/1.0/`).

## Command line

```bash
ecg-record measure signal.npy --sampling-rate 500 --lead-names I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 -o rec.json
ecg-record validate rec.json
ecg-record query rec.json qrs_duration_ms --lead V1 --beat 3
ecg-record batch manifest.jsonl --output-dir out --jobs 4
```

## Validation status

Every published field carries a validation status taken from the shipped
validation-evidence registry. In schema 1.0.0 **every field is
`unvalidated`**: benchmark history exists for several endpoints, but no field
has an approved evidence manifest yet, and a status can only be raised by
citing one. The four-class ST morphology is deliberately not published (it was
never validated and is anti-correlated with ischemia).

## Versions and compatibility

- The package version and the ECG Record schema version are independent. This
  release writes schema `1.0.0` and reads schema major `1`, tolerating
  additive fields from later minors.
- Legacy APIs (`ECGFeatureExtractor`, `to_dict`, `interpret`, `plot_*`, the
  legacy result dataclasses) remain for a compatibility window. Legacy
  functions emit `ECGDeprecationWarning` on use; `ecgfeat.compat` provides the
  exact legacy contract without warnings. Removal is no earlier than 0.3.0.
