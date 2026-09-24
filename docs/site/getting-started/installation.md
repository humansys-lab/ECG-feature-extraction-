# Installation

## Requirements

- Python 3.10, 3.11, 3.12 or 3.13
- NumPy ≥ 1.26 (NumPy 1.26 and 2.x are both supported) and SciPy ≥ 1.11,
  installed automatically

The package is pure Python (a `py3-none-any` wheel) and runs on Linux, macOS
and Windows.

## Install from PyPI

```bash
pip install ecg-records
```

This installs the measurement pipeline, the ECG Record read/write/query API
and the `ecg-record` command line. It depends only on NumPy and SciPy.

## Optional extras

| Extra | Command | Adds | Needed for |
|---|---|---|---|
| `viz` | `pip install "ecg-records[viz]"` | Matplotlib ≥ 3.7 | [`ecgfeat.viz`](../guide/plotting.md) plots |
| `performance` | `pip install "ecg-records[performance]"` | Numba | faster numeric kernels; results are identical without it |
| `wfdb` | `pip install "ecg-records[wfdb]"` | the `wfdb` package | reading PhysioNet `.dat` records (see [Preparing input signals](../guide/input.md#physionet-wfdb-records)) |
| `interpret` | `pip install "ecg-records[interpret]"` | the separate `ecginterpret` distribution | rule-based interpretation and the legacy `ECGFeatureExtractor` / `to_dict` path |

Extras combine: `pip install "ecg-records[viz,performance]"`.

!!! note "About `interpret`"
    `ecginterpret` is a separate distribution that has **not been released yet**.
    Until it is, `pip install "ecg-records[interpret]"` cannot be resolved. The
    record path (`ecg_record`, `load_record`, the queries, `ecgfeat.viz`, the
    CLI) never needs it.

## Verify the installation

```python
import ecgfeat

print(ecgfeat.__version__)               # 0.1.0
print(ecgfeat.STANDARD_12_LEADS)         # ('I', 'II', 'III', 'aVR', ..., 'V6')
```

```bash
ecg-record --help
```

`import ecgfeat` is cheap: imports are lazy, so reading and querying records
does not load the numerical engine, and nothing in the package imports
Matplotlib until you use `ecgfeat.viz`.

## Install from source

```bash
git clone https://github.com/humansys-lab/ECG-feature-extraction-.git
cd ECG-feature-extraction-
python -m pip install -e "./feature_extraction[viz]"
```

The package lives in `feature_extraction/`. See the
[contributor guide](../development/contributing.md) for the development tools.
