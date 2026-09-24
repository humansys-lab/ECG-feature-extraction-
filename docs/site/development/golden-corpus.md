# Golden regression corpus

The golden corpus protects the library against unintended changes in what it
measures. It freezes the output of one code version on 1,063 real recordings
from seven public datasets (LUDB, QTDB, EDB, PTB-XL, BUT-PDB, NSTDB, GUDB) and
reports every difference later code produces. A difference is not
automatically wrong, but it must be understood and reviewed.

It is a regression gate, not a validation: it says nothing about whether a
value is correct (see [Validation](../validation.md)).

## Tiers and baselines

| Tier | Cases | Typical runtime (14 workers) |
|---|---|---|
| `sentinel` | 156: 96 standard cases plus limited-mode, ST-source and one case pair per refinement flag | about 1 minute |
| `full` | 1,063: every eligible record of the seven datasets (PTB-XL capped at 500) | about 7 minutes |

Cases are selected by a hash of the record id (rule `ecg-records-golden-v1`),
never by looking at outputs. Each case is a 10 s window; the manifests store
the exact input identity (source-file and signal SHA-256) and the fully
resolved configuration.

| Baseline | Meaning |
|---|---|
| `ecg-records-0.1.0` | output of the released 0.1.0 code: the gate for new changes |
| `phase0-52a339c` | output of the pre-migration code, kept for history |

## Running it

The datasets are never committed. Mount them and point `--data-root` at their
root:

```bash
python snapshot_regression.py --data-root /data --workers 8 \
    check --tier sentinel --baseline-id ecg-records-0.1.0 --mode record-bytes
```

The check needs only `ecg-records`. Use the NumPy/SciPy versions recorded in
the baseline (NumPy 2.5.3, SciPy 1.18.1), because `debug`-profile diagnostics
can differ in their last floating-point digit across versions.

In CI, the `sentinel-golden` job (`test.yml`) and the weekly `parity`
workflow run this on a self-hosted runner labelled `ecg-datasets`, enabled
with the repository variable `ECG_DATASET_RUNNER=enabled`.

## When outputs change on purpose

1. Run the full tier and read the report: which cases, profiles and fields
   changed, and why.
2. Review the change like any behavioural change, with before/after evidence
   (plots, benchmark metrics).
3. Freeze a new baseline, for example `ecg-records-0.2.0`
   (`snapshot_regression.py freeze --tier sentinel|full --baseline-id ...`).
   Freezing runs the legacy surface too, so it needs `ecginterpret`. It refuses
   to overwrite an existing baseline.
4. Point the CI jobs at the new baseline. Changes to baselines and manifests
   need the Validation Owner's approval (`.github/CODEOWNERS`).

The in-repository reference, `benchmarks/GOLDEN.md`, documents the dataset
adapters, configurations, normalization rules and report format in detail.
