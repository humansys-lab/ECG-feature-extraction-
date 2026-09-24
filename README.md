# ecg-records

**ecg-records** measures 12-lead (or 1–8-channel) ECG signals and emits a
small, versioned JSON document, the **ECG Record**. It contains fiducial
points, intervals, amplitudes, areas, signal quality and provenance in
raw-sample coordinates, with explicit units, an explicit reason for every
missing value, a stable address for every value and a per-field validation
status. The Python import name is `ecgfeat`.

> This software is research and engineering software. It is not a medical
> device and is not intended to diagnose, treat, cure, or prevent disease.
> Outputs require independent validation for the intended use and must not be
> used as a substitute for professional medical judgment.

```bash
pip install ecg-records            # NumPy + SciPy only
pip install "ecg-records[viz]"     # adds Matplotlib for ecgfeat.viz
```

```python
import numpy as np
from ecgfeat import ecg_record, query_measurement, dumps_record

leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
signal = np.load("ecg.npy")                        # shape (12, n_samples), millivolts
record = ecg_record(signal, sampling_rate=500, lead_names=leads)

print(query_measurement(record, "qrs_duration_ms", lead="V1", beat=0))
open("ecg.record.json", "wb").write(dumps_record(record))
```

```bash
ecg-record measure ecg.npy --sampling-rate 500 \
    --lead-names I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 -o ecg.record.json
```

## Documentation

The documentation lives in [`docs/site/`](docs/site/index.md). Build it with
`mkdocs build` (output in `build/site/`):

- [Installation](docs/site/getting-started/installation.md) and
  [Quickstart](docs/site/getting-started/quickstart.md)
- User guide: [input signals](docs/site/guide/input.md),
  [measuring](docs/site/guide/measuring.md),
  [the ECG Record format](docs/site/guide/record-format.md),
  [querying](docs/site/guide/querying.md),
  [saving and sidecars](docs/site/guide/serialization.md),
  [plotting](docs/site/guide/plotting.md),
  [command line](docs/site/guide/cli.md),
  [configuration](docs/site/guide/configuration.md),
  [errors](docs/site/guide/errors.md),
  [migration from the legacy API](docs/site/migration.md)
- [Validation](docs/site/validation.md),
  [limitations](docs/site/limitations.md),
  [compatibility](docs/site/compatibility.md), [FAQ](docs/site/faq.md)
- [API reference](docs/site/reference/api-ecgfeat.md),
  [schema reference](docs/site/reference/ecg-record-schema-1.0.md)
- Developer guide: [architecture](docs/site/development/architecture.md),
  [contributing and testing](docs/site/development/contributing.md),
  [golden corpus](docs/site/development/golden-corpus.md),
  [validation policy](docs/site/development/validation-policy.md),
  [releasing](docs/site/development/releasing.md)

## Repository layout

| Path | Content |
|---|---|
| `feature_extraction/` | the `ecg-records` distribution: `pyproject.toml`, the `ecgfeat` package, `CHANGELOG.md` |
| `schemas/ecg-record/1.0/` | JSON Schema, validation-evidence registry, legacy crosswalk |
| `tests/` | test suite and the pinned reference fixture |
| `benchmarks/golden/` | golden regression corpus: manifests, frozen baselines and harness (`snapshot_regression.py`); datasets are not included |
| `tools/` | release, documentation, registry and performance checks |
| `docs/site/`, `mkdocs.yml` | documentation |
| `.github/workflows/` | CI (`test.yml`), dataset parity (`parity.yml`), releases (`release.yml`) |

## Development

```bash
python -m pip install -r requirements-ci.txt
python -m pip install -e "./feature_extraction[dev]"
python -m pytest -q tests
lint-imports --config feature_extraction/.importlinter
```

See [contributing and testing](docs/site/development/contributing.md) for the
full list of checks.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
