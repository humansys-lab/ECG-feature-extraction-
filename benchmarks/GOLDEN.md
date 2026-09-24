# Golden regression corpus

The golden corpus freezes what a given version of the code produced on a fixed
set of real recordings, so that any later change to a published value shows up
as a difference that has to be reviewed. It is **not** evidence that any value
is correct.

Two baselines are frozen:

| Baseline | Produced by | Use |
|---|---|---|
| `ecg-records-0.1.0` | the released `ecg-records` 0.1.0 code | **the regression gate**: `check --mode record-bytes` must pass 156/156 (sentinel) and 1,063/1,063 (full). Needs only `ecg-records`. |
| `phase0-52a339c` | the pre-migration code (revision `52a339c`) | historical. The migration kept the legacy export byte-identical to it (`legacy-bytes`, which needs the separate `ecginterpret` distribution) and every record value equal to its legacy value (`record-crosswalk`, which needs the local payload cache written by `freeze`). Its *record* hashes no longer match: the record layer was deliberately changed afterwards (per-field default absence states, withheld invariant violations). |

## What is frozen

| File | Content | Mutability |
|---|---|---|
| `golden/manifest.sentinel.json` | 156 cases: selection rule, pinned configurations (declared and fully resolved, with SHA-256), and each case's input identity (adapter, window, source-file SHA-256, canonical signal SHA-256, metadata SHA-256) | Immutable. `freeze` refuses to overwrite it with a different selection. |
| `golden/manifest.full.json` | Every eligible record of the seven datasets under `standard-default-v0` (PTB-XL capped at 500 ranked records) | Immutable |
| `golden/baselines/phase0-52a339c/expected.<tier>.json` | Per case and profile: SHA-256, byte length and key-order-independent leaf digest of the legacy export (`summary`, `audit`, `debug`, `full`) and of the ECG Record (`summary`, `all`, `debug`), plus the environment and git revision | Replaced only by a separately reviewed behavioral baseline |
| `_payloads/` (git-ignored) | gzip legacy `debug` payloads and record documents, for pointer-level diff reports | Local cache |

Signals are never committed; the corpus is regenerated from the mounted
datasets and verified against the frozen hashes before any output is compared.

### Selection rule `ecg-records-golden-v1`

For each dataset, list eligible records, rank by
`sha256("ecg-records-golden-v1:<dataset>:<record>")`, and take the first N whose
adapter loads. The rule never looks at extraction output.

Sentinel counts: 24 LUDB, 20 QTDB, 16 EDB, 16 PTB-XL, 8 BUT-PDB, 6 NSTDB,
6 GUDB (96 cases under `standard-default-v0`), plus 60 targeted cases:
`limited-default-v0` (12), `st-calibrated-pr-v0` (6), `st-adaptive-pr-tp-v0` (6)
and one case pair per refinement flag, each flag alone (36). There is no
all-flags configuration.

### Adapters (`golden/datasets.py`)

Every case is a 10 s window in mV, channel-major float64:

| Dataset | Standard-mode input | Limited-mode input | Window start |
|---|---|---|---|
| LUDB, PTB-XL (`*_hr`) | 12 named leads, canonical order | LUDB I, II, V2 | 0 s |
| QTDB, EDB, BUT-PDB, NSTDB | channel 0 → II, channel 1 → V2, other slots zero (the convention of the project's original external-dataset evaluation) | both channels, header names | QTDB: first `q1c` annotation − 2 s; EDB/NSTDB: 300 s; BUT: 0 s |
| GUDB (cables) | columns 1, 2 → II, V2; V → mV | `cables_1`, `cables_2` | 30 s |

### Configurations

`standard-default-v0` is the pinned default configuration: `fs_internal=None`,
`mains_freq=50`, `input_mode="standard"`, `st_amplitude_source="analysis"`,
every refinement flag off. The manifest stores the fully resolved extractor
configuration; `check` fails on any resolved-configuration drift.

### Normalization

`legacy-bytes` compares exporter bytes verbatim (exporter key order, `repr`
floats, `round_ndigits=None`). The only normalization is the wall-clock
`generated_at` timestamp at `/clinical_interpretation/generated_at` and
`/metadata/clinical_interpretation/generated_at`; the Phase 0 double run showed
these were the only nondeterministic leaves. Failures are frozen too: a case
whose surface raised in the baseline must raise the identical error.

## Commands

The datasets are never committed. Mount them read-only and pass their root
with `--data-root` (default `<repo>/data`). The expected directory layout is
the one `golden/datasets.py` reads (PhysioNet release directory names).

```bash
# the regression gate for this branch (no ecginterpret needed)
python snapshot_regression.py --data-root /data --workers 8 \
    check --tier sentinel --baseline-id ecg-records-0.1.0 --mode record-bytes
python snapshot_regression.py --data-root /data --workers 12 \
    check --tier full --baseline-id ecg-records-0.1.0 --mode record-bytes

# freeze a new baseline after a reviewed behavioural change
# (runs the legacy surface too, so ecginterpret must be installed; refuses to overwrite)
python snapshot_regression.py --data-root /data freeze --tier sentinel --baseline-id ecg-records-<version>
python snapshot_regression.py --data-root /data freeze --tier full --baseline-id ecg-records-<version>
```

Modes: `record-bytes` (record SHA-256 per profile), `legacy-bytes` (legacy
export; needs `ecginterpret`), `record-crosswalk` and `canonical-document`
(need the baseline's local payload cache under `benchmarks/_payloads/`).

Exit codes: 0 pass, 1 output difference, 2 integrity failure (changed input,
changed resolved configuration, or refusal to overwrite). Reports name every
differing case and profile.

### Environment

A baseline records the environment it was frozen in (`environment` in
`expected.<tier>.json`). `ecg-records-0.1.0` was frozen with Python 3.12,
NumPy 2.5.3 and SciPy 1.18.1. Published `summary`/`all` values are integers and
were identical under NumPy 2.2 and 2.5, but a `debug`-profile diagnostic can
differ in its last floating-point digit between NumPy versions, so run the gate
with the recorded NumPy/SciPy versions (the CI jobs pin them).

## History

The corpus was frozen before the library migration (baseline `phase0-52a339c`)
and gated every migration phase. Two independent runs of the unchanged code
agreed on 156/156 sentinel cases; the only nondeterministic leaves were the
interpretation timestamps. At the end of the migration, the full tier
reproduced the legacy export on 1,063/1,063 records, and every record value
equalled its legacy value through the crosswalk.

The pre-migration record path failed on 6 of 156 sentinel cases (BUT-PDB 10;
NSTDB 118e_6, 118e12, 119e00), because a single negative per-beat QRS interval
failed validation of the whole record. The released code publishes such cells
as `unmeasurable(negative_interval)` instead; `ecg-records-0.1.0` has no
failures.
