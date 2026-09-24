# ecg-records

Versioned ECG measurement records for research and engineering.

> This software is research and engineering software. It is not a medical
> device and is not intended to diagnose, treat, cure, or prevent disease.
> Outputs require independent validation for the intended use and must not be
> used as a substitute for professional medical judgment.
> See [Limitations and intended use](limitations.md).

| Distribution | Import | Version | Contents |
|---|---|---|---|
| `ecg-records` | `ecgfeat` | 0.1.0 | Measurement pipeline, ECG Record schema 1.0.0, read/query API, CLI |
| `ecginterpret` | `ecginterpret` | 0.1.0 | Rule-based interpretation; Interpretation document 1.0.0 |
| `ecg-records-viz` | `ecgrecords_viz` | 0.1.0 | Matplotlib plots of `(signal, record)` |

```bash
pip install ecg-records            # core: NumPy + SciPy only
pip install "ecg-records[viz,interpret,performance,wfdb]"
```

```python
from ecgfeat import ecg_record, query_measurement
record = ecg_record(signal, sampling_rate=500, lead_names=leads)
query_measurement(record, "qrs_duration_ms", lead="V1", beat=0)
```

- [Concepts](concepts.md): profiles, absence states, addressing, validation tiers, sidecars
- [Migration guide](migration.md) from `ecgfeat-dxl-inspired` / legacy `ECGFeatures`
- [Validation and benchmarks](validation.md)
- [Compatibility policy](compatibility.md)
- Reference: [ecgfeat API](reference/api-ecgfeat.md) ·
  [ECG Record schema 1.0](reference/ecg-record-schema-1.0.md) ·
  [legacy crosswalk](reference/legacy-crosswalk.md) ·
  [ecginterpret API](reference/api-ecginterpret.md) ·
  [Interpretation document 1.0](reference/interpretation-schema-1.0.md) ·
  [ecg-records-viz API](reference/api-ecgrecords-viz.md)
