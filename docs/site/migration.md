# Migration guide

From the unpublished `ecgfeat-dxl-inspired` working tree and the legacy
`ECGFeatures` object to `ecg-records` 0.1.0.

## In one minute

Before (legacy object; warns, and needs `ecginterpret`):

<!-- doc-check: skip (legacy API; needs the separate ecginterpret distribution) -->
```python
from ecgfeat import ECGFeatureExtractor
features = ECGFeatureExtractor().extract(signal, 500)
```

After:

```python
import numpy as np
from ecgfeat import ecg_record, query_measurement

signal = np.load("ecg.npy")
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
record = ecg_record(signal, sampling_rate=500, lead_names=leads)
qrs = query_measurement(record, "qrs_duration_ms", lead="V1", beat=0)
```

The new API requires explicit lead names (the legacy extractor defaulted to the
canonical order when `lead_names` was omitted) and returns a versioned,
validated record instead of a mutable object graph.

## Imports

| Old | New | Old path supported until |
|---|---|---|
| `from ecgfeat import ECGFeatureExtractor` / `ecgfeat.api` | `ecgfeat.ecg_record(...)` for records; `ecgfeat.compat.api_v0.ECGFeatureExtractor` for the exact legacy object | warns now; removal no earlier than 0.3.0 |
| `ecgfeat.to_dict`, `ecgfeat.export.to_dict` / `prepare_json_export` | `ecgfeat.dumps_record(record)` / `record_to_dict`; legacy payload: `ecgfeat.compat.export_v0` | warns now; ≥ 0.3.0 |
| `ecgfeat.models.*` dataclasses | `ecgfeat.compat.models_v0.*` (same classes) | silent alias; ≥ 0.3.0 |
| `ecgfeat.interpret`, `ecgfeat.clinical_rules`, `ecgfeat.glasgow_rules`, … | the separate `ecginterpret` distribution (not yet released); it will provide `ecginterpret.interpret_record(record, signal=...)` | aliases warn; ≥ 0.3.0 |
| `ecgfeat.plot_beat`, `plot_rep_beat`, … / `ecgfeat.visualize` | `ecgfeat.viz.plot_beat(signal, record, ...)` (`pip install "ecg-records[viz]"`) | warn; ≥ 0.3.0 |
| `ecgfeat.load_wfdb_mat`, `parse_wfdb_header` | `ecgfeat.io.read_wfdb`, `ecgfeat.io.read_wfdb_header` | warn; ≥ 0.3.0 |
| `ecgfeat.delineate`, `ecgfeat.quality`, `ecgfeat.features`, … (private engine) | no public replacement; use record fields | alias warns; ≥ 0.3.0 |

`ecgfeat.compat` is the explicit legacy contract: it does not warn, and the
whole namespace is removed together no earlier than 0.3.0.

## Values

Every published record value maps exactly to a legacy leaf through the
reviewed [crosswalk](reference/legacy-crosswalk.md) (sample indices converted
to native sampling rate, milliseconds and microvolts rounded half-to-even).
Values that violate a published invariant are published as `unmeasurable`
instead. The legacy payload itself is unchanged: `ecgfeat.compat.export_v0`
reproduces the pre-migration bytes on 1,063 golden records.

## Records vs. legacy object

The legacy object carries many intermediate quantities that ECG Record 1.0
does not publish (representative-lead morphology, rhythm inputs, P-wave
assessments). Consumers that need them keep using `ecgfeat.compat` during the
compatibility window; new quantities become record fields only with a unit,
absence semantics, provenance, a validation entry and size accounting.

## Interpretation is no longer part of this package

The legacy extractor and the legacy JSON export include interpretation
sections, and interpretation now lives in the separate `ecginterpret`
distribution. Without it, `ECGFeatureExtractor().extract(...)`, `to_dict(...)`
and `ecgfeat.interpret` raise an `ImportError` that names the missing
distribution. The record API never needs interpretation.

## Turning warnings into errors

To find every remaining legacy call, run your tests with deprecation warnings
as errors:

```bash
python -W error::FutureWarning -m pytest
```

`ECGDeprecationWarning` is a `FutureWarning`, so this catches exactly the
legacy calls. See [Errors and warnings](guide/errors.md#warnings).
