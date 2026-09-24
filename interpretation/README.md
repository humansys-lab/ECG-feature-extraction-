# ecginterpret

Rule-based ECG interpretation statements (rhythm, conduction, hypertrophy,
ischemia, pediatric and Glasgow-style rule sets) computed from the
measurements produced by [`ecg-records`](../feature_extraction/README.md).
The output is a separately versioned **Interpretation document**; it is not
part of the ECG Record measurement contract.

> This software is research and engineering software. It is not a medical
> device and is not intended to diagnose, treat, cure, or prevent disease.
> Outputs require independent validation for the intended use and must not be
> used as a substitute for professional medical judgment.

## Install

```bash
pip install ecginterpret          # or: pip install "ecg-records[interpret]"
```

## Use

```python
import numpy as np
from ecgfeat import ecg_record
from ecginterpret import interpret_record

record = ecg_record(signal, sampling_rate=500, lead_names=leads, profile="summary")
document = interpret_record(record, signal=signal)   # versioned Interpretation document
result = document.as_dict()
result["interpretation"]["heart_rate_class"]        # e.g. "normal"
result["interpretation"]["bundle_branch_block"]     # e.g. "LBBB", or None
[s["statement"] for s in result["clinical"]["final_statements"]]
```

`interpret_record` needs the raw signal: ECG Record schema 1.0 does not yet
carry every measurement the rule engines read (representative-lead
morphology, rhythm and P-wave evidence), so the interpreter re-derives them
from the signal with the record's own resolved configuration after checking
the signal against the record's raw-signal SHA-256.  `interpret_features` takes
the legacy measurement object from `ecgfeat.compat.api_v0.ECGFeatureExtractor`
directly (compatibility input, removed with `ecgfeat.compat`).

## Versioning

The Interpretation document carries `schema_version` (currently `1.0.0`),
independent of both the package version and the ECG Record schema version.
The JSON Schema ships at `ecginterpret/schemas/interpretation/1.0/schema.json`.
Supported input: ECG Record schema major `1`.
