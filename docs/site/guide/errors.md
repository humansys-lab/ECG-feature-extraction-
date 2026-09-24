# Errors and warnings

## Exception hierarchy

All public errors derive from `ecgfeat.ECGInputError`, which derives from
`ValueError`. Code that already catches `ValueError` keeps working; catch
`ECGInputError` to handle only this package's errors.

```text
ValueError
└── ECGInputError                     base class of every public error
    ├── SignalShapeError              array rank, orientation, length, non-finite values
    ├── LeadNameError                 missing, duplicate or non-canonical lead names
    ├── SamplingRateError             invalid sampling rate or internal rate
    ├── AmplitudeUnitError            unsupported amplitude unit
    ├── ConfigurationError            unsupported method or configuration value
    ├── RecordValidationError         a record violates the schema or its invariants
    │   └── SidecarError              a sidecar is missing, altered or inconsistent
    ├── RecordIdentityError           an address targets another record or schema version
    ├── AddressSyntaxError            malformed address or JSON pointer
    ├── AddressNotFoundError          the pointer does not exist in the record
    ├── MeasurementNotFoundError      the measurement is not published in this record
    ├── MeasurementSelectorError      invalid lead or beat selector
    ├── ComputationInvariantError     no trustworthy record could be built
    └── ecgfeat.viz.VisualizationInputError   signal/record mismatch or bad selector in a plot
```

All of these are importable from `ecgfeat` (except `VisualizationInputError`,
which lives in `ecgfeat.viz`).

## Error codes

Every error has a stable, machine-readable `code`, a `message` and optional
`details`. Branch on `code`, not on the message text:

```python
import numpy as np
from ecgfeat import ecg_record, ECGInputError, LeadNameError

signal = np.load("ecg.npy")
try:
    ecg_record(signal, sampling_rate=500, lead_names=["I", "II", "III", "AVR", "aVL", "aVF",
                                                      "V1", "V2", "V3", "V4", "V5", "V6"])
except LeadNameError as exc:
    print(exc.code)          # standard_lead_contract ("AVR" is not canonical)
    print(exc.to_dict())     # {'code': 'standard_lead_contract', 'message': '...', 'details': {}}
except ECGInputError as exc:
    print("other input problem:", exc.code)
```

Common codes:

| Code | Raised by | Meaning |
|---|---|---|
| `invalid_signal_shape`, `invalid_signal_type` | input | not a numeric 2-D array |
| `signal_lead_count_mismatch` | input | row count differs from the number of lead names (often a transposed array) |
| `record_too_short`, `empty_signal` | input | less than 1 second, or no samples |
| `nonfinite_signal` | input | NaN or infinity in the signal |
| `invalid_sampling_rate` | input | below 100 Hz or not finite |
| `invalid_lead_names` | input | empty or duplicate names |
| `standard_lead_contract` | input | standard mode without exactly the twelve canonical leads |
| `limited_lead_count` | input | limited mode outside 1–8 channels |
| `unsupported_amplitude_unit` | input | unit other than mV or µV |
| `invalid_input_mode`, `input_mode_mismatch` | configuration | unknown mode, or config and call disagree |
| `unknown_method` | configuration | unsupported `method` |
| `invalid_profile` | emit, serialization | profile other than summary, all, debug |
| `record_build_failed` | measurement | internal failure (`ComputationInvariantError`) |
| `invalid_record_json`, `duplicate_json_member` | reading | not valid JSON, or a member appears twice |
| `invalid_record_contract` | reading | a structural or cross-field invariant is violated |
| `validation_above_registry` | reading | a field claims a better validation tier than the registry allows |
| `profile_data_unavailable` | serialization | asked to serialize a larger profile than the record holds |
| `sidecar_missing`, `invalid_sidecar` | reading | sidecar absent, altered or inconsistent |
| `measurement_not_found` | query | name unknown or not in this profile |
| `invalid_lead_selector`, `invalid_beat_selector` | query | lead or beat missing or unknown |
| `invalid_record_address`, `invalid_json_pointer` | addresses | malformed address or pointer |
| `record_identity_mismatch` | addresses | the address names another record |
| `address_not_found` | addresses | the pointer is not in the record |

## Warnings

| Warning | Base classes | Meaning |
|---|---|---|
| `ECGWarning` | `UserWarning` | base class of actionable, non-fatal warnings from this package |
| `ECGCompatibilityWarning` | `ECGWarning` | a legacy compatibility behaviour was used |
| `ECGDeprecationWarning` | `FutureWarning`, `ECGWarning` | a legacy API is scheduled for removal (no earlier than 0.3.0) |

`ECGDeprecationWarning` subclasses `FutureWarning`, so Python shows it by
default: legacy calls stay visible in scripts, not only in test runs. To turn
legacy usage into errors while migrating:

```python
import warnings
from ecgfeat import ECGDeprecationWarning

warnings.simplefilter("error", ECGDeprecationWarning)
```

Or run your test suite with `python -W error::FutureWarning`. See the
[migration guide](../migration.md) for replacements.
