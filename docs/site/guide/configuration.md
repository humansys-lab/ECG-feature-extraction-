# Configuration

The defaults are the reviewed, benchmarked configuration, and most users never
change them. When you do, every setting is recorded in the record's
provenance (`/provenance/config/resolved`, with its SHA-256 in
`/provenance/config_hash`) and enters the `record_id`.

## `ECGConfig`

```python
from ecgfeat import ECGConfig

config = ECGConfig(
    input_mode="standard_12",       # or "limited"
    fs_internal=None,               # internal analysis rate; None = native rate
    mains_frequency_hz=50,          # 50, 60 or None
    st_amplitude_source="analysis", # see below
)
```

`ECGConfig` is an immutable (frozen) dataclass. Invalid values raise
`ConfigurationError` or `SamplingRateError` when you construct it.

| Field | Default | Meaning |
|---|---|---|
| `input_mode` | `"standard_12"` | `"standard_12"` or `"limited"`; must agree with the `input_mode` passed to `ecg_record` / `ecg_prepare` (`input_mode_mismatch` otherwise) |
| `fs_internal` | `None` | internal analysis sampling rate in Hz (an integer ≥ 100). `None` analyses at the native rate rounded to an integer. Published fiducials are always converted back to native sample indices. |
| `mains_frequency_hz` | `50` | power-line frequency to suppress: `50` (Europe, Asia), `60` (the Americas, parts of Japan) or `None` for no mains filtering. Set it to match where the recording was made. |
| `refinement` | all off | optional [experimental refinements](#experimental-refinements) |
| `p_wave` | default | reserved; must be left at its default in 0.1.0 (see [below](#p-wave-settings)) |
| `st_amplitude_source` | `"analysis"` | baseline for the *hybrid* ST candidates: `"analysis"`, `"calibrated_pr"` or `"adaptive_pr_tp"`. Hybrid candidates are not published in schema 1.0, so this does not change any `summary` or `all` value; the published `st_80ms_uv` always uses the native method (`/provenance/fields/st_native`). |

Pass the config to `ecg_record`, `ecg_measure`, `ECGRecordExtractor` or, as a
file, to the command line:

```python
import numpy as np
from ecgfeat import ECGConfig, ecg_record

signal = np.load("ecg.npy")
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
record = ecg_record(signal, sampling_rate=500, lead_names=leads,
                    config=ECGConfig(mains_frequency_hz=60))
print(record.provenance["config"]["resolved"]["mains_frequency_hz"])   # 60
print(record.provenance["mains_frequency_hz"])                         # 60
```

For limited input, set the mode in both places or only in the config:

```python
import numpy as np
from ecgfeat import ECGConfig, ecg_record

two = np.load("ecg.npy")[[1, 6]]
record = ecg_record(two, sampling_rate=500, lead_names=["II", "V1"],
                    config=ECGConfig(input_mode="limited"))      # input_mode taken from config
print(record.acquisition["input_mode"])                          # limited
```

## Experimental refinements

`RefinementConfig` holds independently switchable accuracy candidates, all off
by default. Default extraction keeps its established measurements. Enable a
candidate only after reviewing its results on your own data.

```python
from ecgfeat import ECGConfig, RefinementConfig

config = ECGConfig(refinement=RefinementConfig(t_bidirectional=True))
print(sorted(config.refinement.experimental_flags))    # ['t_bidirectional']
```

Available switches: `p_model_arbitration`, `p_multiple_candidates`,
`p_boundary_correction`, `p_phasor_candidates`, `p_pathology_candidates`,
`t_onset_change_point`, `t_bidirectional`, `t_correlated_fusion`,
`t_boundary_projection`, `t_sequence_selection`, `t_projection_offset_only`,
`t_sequence_offset_only`, `qrs_terminal_multiscale`, `qrs_quality_reference`,
`qrs_adaptive_consensus`, `representative_robust`, `grouping_outliers` and
`atrial_event_validation`.

Every enabled switch is listed in `/provenance/experimental`. A record with a
non-empty list was not produced by the default, benchmarked configuration.
Treat it accordingly.

## P-wave settings

In 0.1.0 the P-wave engine runs with fixed settings. `ECGConfig.p_wave` exists
for a future release and accepts only the default value of
`ecgfeat.config.PWaveConfig`.

!!! warning "Known issue in 0.1.0"
    The top-level name `ecgfeat.PWaveConfig` refers to the legacy engine's
    P-wave settings class, not to the class `ECGConfig.p_wave` expects.
    `ECGConfig(p_wave=ecgfeat.PWaveConfig())` raises
    `ConfigurationError: p_wave must be a record PWaveConfig`. Leave `p_wave`
    unset.

## Configuration files

The command line (`--config`) and batch manifests read configuration from a
JSON object with the same field names. Nested `refinement` and `p_wave`
objects are converted automatically:

```json
{
  "mains_frequency_hz": 60,
  "fs_internal": 500,
  "refinement": {"t_bidirectional": true}
}
```

Unknown keys or invalid values fail with exit code 2 (`ConfigurationError`).

## Algorithm method

`method="default"` (the default) and `method="legacy_dxl_inspired"` select the
same algorithm; the record stores both the requested and the resolved name.
Any other value raises `ConfigurationError` (`unknown_method`). The parameter
exists so that future algorithm versions can be selected explicitly without
changing the meaning of existing records.
