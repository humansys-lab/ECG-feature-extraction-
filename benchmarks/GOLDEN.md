# Golden parity corpus

The golden corpus is the Phase 0 oracle from
[`docs/library_design/05_migration_plan.md`](../docs/library_design/05_migration_plan.md)
section B. It freezes what the unchanged pre-migration code produced, so every
structural phase can prove that it changed structure and not measurements.
It is **not** evidence that any value is correct.

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
| QTDB, EDB, BUT-PDB, NSTDB | channel 0 → II, channel 1 → V2, other slots zero (the `scripts/evaluate_ecgfeat_external.py` convention) | both channels, header names | QTDB: first `q1c` annotation − 2 s; EDB/NSTDB: 300 s; BUT: 0 s |
| GUDB (cables) | columns 1, 2 → II, V2; V → mV | `cables_1`, `cables_2` | 30 s |

### Configurations

`standard-default-v0` is document 05's pinned default: `fs_internal=None`,
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

```bash
# freeze (only from an unchanged tree; refuses to overwrite)
python snapshot_regression.py --code-root /path/to/baseline/feature_extraction \
    freeze --tier sentinel --baseline-id phase0-52a339c

# check a working tree
python snapshot_regression.py check --tier sentinel --baseline-id phase0-52a339c \
    --mode legacy-bytes --with-record
python snapshot_regression.py check --tier full --baseline-id phase0-52a339c --mode legacy-bytes
python snapshot_regression.py check --tier sentinel --baseline-id phase0-52a339c --mode record-crosswalk
```

Exit codes: 0 pass, 1 output difference, 2 integrity failure (changed input,
changed resolved configuration, or refusal to overwrite). Reports go to
`_reports/` and name every changed JSON pointer.

## Phase gates

| Phase | Required golden gate |
|---|---|
| 0 | Two independent baseline runs, zero diffs (done: 156/156 sentinel) |
| 1 `_engine` moves | sentinel `legacy-bytes --with-record`, full `legacy-bytes` |
| 2 record shadow | `record-crosswalk` exact; `legacy-bytes` through `compat.export_v0` |
| 3 `api.py` decomposition | full `legacy-bytes` through the compatibility entry point; `record-crosswalk` |
| 4 record primary | `record-crosswalk`; `legacy-bytes` via compat |
| 5 interpretation split | `legacy-bytes` with `ecginterpret` installed; `canonical-document` for interpretation |

## Phase 0 findings

- The legacy export is deterministic apart from the two `generated_at` timestamps.
- The pre-migration record path raises `ComputationInvariantError` on 6 of 156
  sentinel cases (BUT-PDB 10; NSTDB 118e_6, 118e12, 119e00) because a negative
  per-beat QRS interval fails record validation for the whole record. The
  baseline freezes this failure; fixing it is a reviewed record-layer change.
