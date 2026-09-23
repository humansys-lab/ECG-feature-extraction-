# Benchmarks

<!-- BEGIN evaluator-harness (owned by benchmarks/harness; edit only inside this block) -->
## Evaluator harness

`benchmarks/harness/` wraps the ten root-level evaluation tools in one benchmark protocol.
It follows `docs/library_design/06_testing_and_validation.md` ("Benchmark harness as a
release gate") and `05_migration_plan.md` §B ("Assign every existing regression tool a gate
role"). The harness never edits a tool. It:

- runs each tool with pinned inputs and flags (`benchmarks/harness/spec.json`);
- normalizes the tool's own outputs into one JSON result per tool;
- freezes the doc-06 tolerances into every metric before any migrated result exists;
- compares a candidate result set against that frozen baseline.

Frozen baseline: `benchmarks/baselines/phase0-52a339c/`. It was produced from the unchanged
revision `52a339c` (git worktree `/workspace/ecg_gemma_baseline`) and contains one normalized
JSON per tool plus the `baseline.json` index (environment, git revision, spec hash, and
determinism summary). Raw tool outputs, logs and harness-built corpora go to
`benchmarks/_raw/` (git-ignored). The first-written copy of the baseline results is kept in
`benchmarks/_raw/phase0-52a339c/results_as_first_written/`.

### How to run

Run these from the repository root. The tree under test must contain `data/`, a directory or
symlink holding the datasets. `evaluate_ludb.py`, `evaluate_ludb_cse.py` and
`compare_annotations.py` hard-code `<tool dir>/data/lobachevsky-...-1.0.1`, so this path
cannot be redirected.

```bash
# freeze a baseline from a pinned, unchanged tree (2 runs per tool = determinism check)
python -m benchmarks.harness run --repo-root /workspace/ecg_gemma_baseline \
    --baseline-id phase0-52a339c --repeat 2 --max-workers 8

# candidate (e.g. the migrated tree); one run is enough once the baseline proved determinism
python -m benchmarks.harness run --repo-root . --baseline-id candidate \
    --out-dir benchmarks/_candidate --raw-dir benchmarks/_raw/candidate --repeat 1 --max-workers 8

# gate: exit 0 = pass, 1 = any regression / new failure / missing metric /
# manifest or config mismatch / blocked candidate, 2 = harness error
python -m benchmarks.harness compare --baseline benchmarks/baselines/phase0-52a339c \
    --candidate benchmarks/_candidate --report evaluator-report.json

python -m benchmarks.harness list                    # pinned tools
python -m benchmarks.harness renormalize ...         # re-derive results from existing raw runs
python -m pytest benchmarks/harness/test_harness.py  # compare-logic tests (synthetic, no data)
```

Options:

- `--tools <id ...>` restricts `run` or `compare` to a subset of tools.
- `run` refuses to overwrite an existing per-tool result file unless `--force` is given.
- `renormalize` only re-reads raw outputs. It refuses if the pinned invocation changed and never
  re-runs a tool. Use it for harness-side normalizer fixes during baseline freezing only,
  never after a candidate has been seen.

Execution rules:

- Every tool process runs with `PYTHONPATH=<tree>/feature_extraction:<tree>`.
  `OPENBLAS/OMP/MKL/NUMEXPR/NUMBA` threads are set to 1, and `PYTHONHASHSEED=0`.
- A harness-only `sitecustomize` (`benchmarks/harness/_importlog/`) logs where each tool process
  resolved `ecgfeat`/`ecgagent` from. A run whose imports resolve outside the tree under test is
  marked `contaminated_import`, which catches the editable install of another checkout.
- The scheduler keeps the total declared worker processes at or below `--max-workers`
  (capped at 8).

### Result format and tolerances

Each `<tool>.json` records the following:

- the tool name and sha256;
- a `config_digest` of the pinned invocation;
- per dataset: name, release, subset rule, the resolved record list, and a manifest (sha256
  over sorted `(relative path, file sha256)` of the input files actually read);
- the combined `dataset_manifest_sha256`;
- the extraction configuration;
- the environment: Python, NumPy, SciPy, Numba, wfdb, NeuroKit2, torch, BLAS, OS, CUDA state,
  and the git revision of the tree;
- both runs: exit codes, wall time, and import origins;
- the determinism block;
- the metrics.

Every metric carries:

- `value`, `unit` and `direction`;
- `family`: `bounded_accuracy | error | coverage | failure_count | exact`;
- `level`: `aggregate | subgroup`;
- the reporting `quantum`;
- `compare_on`: `abs` for signed biases and extremes;
- the frozen `tolerance`.

Tolerances (doc 06, frozen per metric):

| family | rule |
| --- | --- |
| bounded_accuracy (Se, PPV, F1, specificity, recall, Pearson r, direction accuracy, within-limit share; fractions) | aggregate ≤ 0.5 pp (0.005), subgroup/lead/category ≤ 1.0 pp (0.010) |
| error (ms / mV / deg / count) | ≤ max(2 % of baseline, one reporting quantum); signed bias and min/max compared on \|value\| |
| coverage | fraction ≤ 0.5 pp; yield count ≥ 99.5 % of baseline count |
| failure_count | 0 new failures |
| exact (corpus-fixed counts, annotation-derived reference values/digest, legacy-export key set) | 0 differences |
| ungated (`gated: false`) | recorded only: raw TP/FP/FN, detection counts, runtimes, descriptive value distributions, tool pass/fail booleans |

The following interpretations are fixed here before any candidate exists:

- **Levels.** A tool's headline per-wave or per-dataset rows are `aggregate` (the stricter
  0.5 pp). Per-lead and per-diagnostic-category rows are `subgroup`.
- **Reporting quanta.** These follow each tool's own printed precision: 0.1 ms (LUDB, QTDB,
  detectors, annotations), 0.01 ms (LUDB-CSE), 0.001 mV (ST), 0.1 deg (axes).
- **Metric definitions.** A metric whose family, unit, direction or `compare_on` changes fails
  as `definition_changed`.
- **Output digests.** These are sha256 hashes of per-record outputs, with absolute paths and the
  report `Generated :` wall-clock line removed. `compare` lists them as changed or unchanged, but
  they do not gate. Byte-level parity is owned by the golden snapshot harness.

`compare` checks each tool in this order:

1. The baseline status. Blocked or delegated baseline tools are reported as `ungated`, not as
   passing.
2. Whether the candidate exists and is not blocked.
3. The dataset manifest hash. A changed corpus fails before any metric is scored.
4. The configuration digest.
5. Every baseline metric under its frozen tolerance. A missing metric fails.

### Pinned tools (baseline `phase0-52a339c`)

Wall time is per run under an 8-worker budget, with another job sharing the machine.

| tool | status | pinned input / subset | runtime per run | metrics (gated) | double run |
| --- | --- | --- | --- | --- | --- |
| `snapshot_regression.py` | **delegated** | owned by the golden snapshot harness (`benchmarks/golden/`); not run here | — | — | — |
| `evaluate_ludb.py` | pinned | LUDB 1.0.1, all 200 records × 12 leads (tool has only `--n`) | 591 / 598 s (serial) | 157 (156) | identical |
| `evaluate_qtdb.py` | pinned | QTDB, all 105 records with `.q1c` (channel 0), ecgfeat + NeuroKit2, annotated-span restriction on, 4 workers | 271 / 369 s | 190 (153) | identical |
| `evaluate_ludb_cse.py` | pinned | LUDB, all 200 records, outlier count 8, 4 workers | 136 / 137 s | 170 (115) | identical |
| `compare_annotations.py` | pinned | `--batch-reports --no-visuals --measurement-profile native`, all 200 LUDB records, 4 workers | 142 / 142 s | 89 (81) | identical |
| `compare_ludb_detectors.py` | pinned | LUDB 200 × 12 leads, ecgfeat / NeuroKit2 / BioSPPy, `--restrict-to-annotated-span`, 4 workers | 180 / 200 s | 2656 (2283) | identical |
| `analyze_physionet_st.py` | pinned | EDB (90) + LTSTDB (86, `.stb`), 3 events + 1 control per record, 12 s windows (serial) | 1196 / 1224 s | 93 (77) | identical |
| `batch_extract_ecgfeat.py` | pinned | stand-in `.pt` corpus built from PTB-XL: 5 superclasses × 20 records at 500 Hz and at 100 Hz, `--export-profile summary --no-pt`, 5 workers | 103 / 107 s | 28 (22) | identical |
| `score_measurement_verifiable.py` | pinned (small corpus only) | the 2 frozen agent verdicts on this machine (01000_hr, 01006_hr); `features/` regenerated by the tree under test | 5 / 5 s | 12 (11) | identical |
| `evaluate_target_ecgfeat_diagnosis.py` | pinned | PTB-XL shard `09000` (all 1000 records) + `data/010` (100 Chapman/Ningbo records), 6 workers | 508 / 459 s | 432 (218) | identical |

The full release run of all nine tools, twice, took about 43 minutes of wall time with 8 workers
(08:23 to 09:06 UTC). A single candidate run (`--repeat 1`) is estimated at about 25 minutes,
bounded by the 20-minute serial ST tool; this estimate was not measured end to end.

Notes on specific inputs:

- **`batch_extract_ecgfeat.py` corpus.** The tool's production corpus (`all_diseases_pt100/`,
  generated and real tensors) is not on this machine. The harness builds a deterministic
  stand-in at `benchmarks/_raw/corpora/batch_pt/`. For each of NORM, MI, STTC, CD and HYP it
  takes the first 20 records (ascending ecg_id) of PTB-XL shard 12000 whose active diagnostic
  statements map to exactly one superclass. Records are stored as float32 `real.pt` at 500 Hz and
  at 100 Hz (`resample_poly(1,5)`, the tool's production rate). The recipe and selection are in
  `corpus_build.json`. "Schema validation" is represented by the JSON parse rate plus an exact
  legacy top-level key set, because the legacy export has no JSON schema at this revision.
- **`score_measurement_verifiable.py` corpus.** The tool scores LLM agent output. The frozen
  verdicts are copied from `ecgagent_sample_review_20260908/` and
  `ecgagent_complex_review_20260908/`, which are untracked workspace directories.
  `features/` is regenerated by `python -m ecgagent.batch extract` from the tree under test, so
  the gate checks that ecgfeat's measurements still verify the same frozen calls.
  - The corpus holds 3 confirmed calls: 2 are measurement-verifiable and correct, and 1 is a
    morphology call.
  - The "full corpus per release" tier is **blocked**: it needs a live LLM agent run.
  - On a machine without those two directories (or a prebuilt `_raw/corpora/smv`), this tool is
    reported as `blocked`.
- **`analyze_physionet_st.py` variants.** The tool has no `st_amplitude_source` flag, so doc 05's
  pinned `st_amplitude_source` variants cannot be exercised through this CLI. The tool default
  is pinned.

### Non-zero exit codes (documented verdicts, not crashes)

The harness classifies runs by each tool's own `ok_exit_codes`. For every non-zero exit below,
the outputs are complete and gated.

| tool | exit code | cause | failures |
| --- | --- | --- | --- |
| `evaluate_qtdb.py` | 1 | `release_gate.numeric_pass=false`: the ecgfeat boundary-error SDs exceed the CSE 2σ limits on all five boundaries | 0 failed records |
| `evaluate_ludb_cse.py` | 1 | `proxy_verdict=does_not_meet_numeric_limits`: P duration fails, and PR is paired for 94.3 % of reference-available records | 0 failed records |
| `compare_ludb_detectors.py` | 2 | 19 **NeuroKit2 comparator** lead failures (`ValueError: cannot convert float NaN to integer`, records 24/33/44/45/64/74/82/90/116/121) | ecgfeat 0/200 failed calls, BioSPPy 0/2400 |

The 19 NeuroKit2 failures are frozen as `failure_count` baselines, so 0 new failures are allowed.

### Determinism of the double run

Both baseline runs of every pinned tool ran on unchanged code, partly concurrently. For all
nine tools:

- gated metrics are **bit-identical**;
- output digests are **bit-identical**;
- every logged tool process imported `ecgfeat` and `ecgagent` from `/workspace/ecg_gemma_baseline`.

The only differences are the following ungated values:

- wall-clock and runtime values: QTDB `wall_clock_seconds`, detector runtimes, and
  target-diagnosis `runtime_seconds_total`;
- the `Generated : <timestamp>` line in the human-readable `*_report.txt` files (removed before
  digesting);
- `clinical_interpretation.generated_at` in the feature JSON (dropped as volatile).

No tolerance was widened. A third, independent `--repeat 1` run of `evaluate_ludb_cse`,
`batch_extract_ecgfeat` and `score_measurement_verifiable` passed `compare` with identical
digests.

### Adding or changing a pinned tool

`spec.json` edits change a tool's `config_digest`, so candidates will fail `config_mismatch`
until the Validation Owner re-freezes that tool's baseline. That is deliberate: the tolerance and
inputs of a benchmark cannot be changed after seeing a result.
<!-- END evaluator-harness -->
