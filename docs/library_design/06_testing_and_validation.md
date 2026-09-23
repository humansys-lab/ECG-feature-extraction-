# 06. Testing and validation

> Implementation status: this document includes target contracts, not completed
> release claims. [Document 08](08_implementation_status.md) records the implemented
> subset, reconciled decisions and outstanding gates. Document 02 is authoritative
> for the JSON layout; Python carrier sketches must not create a second wire format.

## Purpose

The redesign changes the public boundary of `ecgfeat`: the existing `models` and `export`
surfaces are replaced by the ECG Record layer, the extraction flow is split into pipeline
stages, and interpretation moves to a separate distribution. The current test suite is large
enough to be useful but is concentrated on the code being replaced: 40 test files import
`feature_extraction.ecgfeat.models`, 39 import `feature_extraction.ecgfeat.export`, 19 import
`feature_extraction.ecgfeat.interpret`, and 22 tests consume the exported JSON contract.

The testing strategy therefore treats compatibility with the published ECG Record as the
primary release gate. Internal unit tests remain important, but they cannot substitute for
contract tests or benchmark evidence.

All release gates must be reproducible from a pinned code revision, dependency environment,
configuration, input hash, dataset version, and baseline identifier. A test or benchmark that
cannot state a falsifiable pass/fail condition is diagnostic tooling, not a release gate.

## Test taxonomy

The target test portfolio is deliberately layered. Counts below are planning ranges, not a
requirement to manufacture tests to hit a number.

| Layer | What it guards | What it must not be used for | Target size |
| --- | --- | --- | ---: |
| Schema and contract tests | ECG Record syntax and semantics, profiles, addressing, serialization, absence states, schema compatibility, and the 24,000-byte summary envelope | Algorithmic accuracy or clinical validity | 50–70 focused tests |
| Engine unit tests | Small deterministic transforms inside `ecgfeat/_engine/*`: filtering, fiducial calculations, interval arithmetic, lead-local measurements, and error handling | Public API compatibility; a passing engine test does not prove a published field is stable or validated | 250–400 tests, favoring parameterization |
| Pipeline stage tests | The eight `ecgfeat/pipeline/` stages, stage inputs/outputs, policy-object behavior, stage-local provenance, and failure/absence propagation | End-to-end clinical validation or direct testing of private helper implementation details | 80–120 tests |
| Golden-fixture regression | End-to-end stability of representative records for pinned signals and configurations | Proving correctness merely because old output is reproduced | 15–25 golden cases |
| Property-based tests | Cross-cutting invariants across generated signals, sampling rates, record fragments, and profile combinations | Replacing carefully chosen pathological examples or dataset benchmarks | 30–50 properties |
| Dataset benchmarks | Accuracy and robustness against external annotations and reference datasets: LUDB, QTDB, EDB, PTB-XL, BUT, NSTDB, and GUDB | Unit-level debugging or silently redefining a baseline after an algorithm change | One pinned benchmark specification per supported dataset/task |

The existing 124 files under `tests/` should be migrated into these layers rather than kept
as a single undifferentiated suite. Tests coupled to `ecgagent` remain consumer integration
tests; they are useful evidence that the record boundary works, but they do not belong to the
core package's unit-test definition.

## Schema and contract tests

### Canonical schema

Publish each ECG Record schema version as a repository resource under
`schemas/ecg-record/<major>.<minor>/schema.json`, using JSON Schema Draft 2020-12. The Python
record layer must expose the same schema bytes as package data so the installed wheel and the
documentation site cannot describe different contracts.

CI must validate every checked-in golden record and every record emitted by contract tests
against the schema. Validation failure is always a hard failure; warnings are not accepted for
unknown required fields, invalid units, malformed absence states, or invalid validation-tier
metadata.

### Required contract assertions

The contract suite must include all of the following pass/fail checks.

1. **JSON Schema validation.** Every `summary`, `all`, and `debug` record produced by a
   pinned contract fixture validates with zero schema errors.
2. **Round-trip encode/decode.** Decoding a valid record into the record model and re-encoding
   it produces a semantically identical JSON document. For the canonical encoder, the UTF-8
   bytes must also be identical.
3. **Absence-state exclusivity.** Every published measurement has exactly one state:
   `measured`, `unavailable(reason)`, or `not_applicable(reason)`. A measured value cannot
   carry an absence reason; either absent state must carry a non-empty reason from the
   documented reason vocabulary.
4. **Pointer resolution.** Every published address of the form
   `ecg-record:<record_id>@<schema_version>#<RFC 6901 JSON pointer>` resolves to exactly the
   intended field. Escaped pointer tokens such as `~0` and `~1`, missing paths, array
   indexes, and wrong record/schema identifiers receive explicit negative tests.
5. **Profile nesting.** For the same signal and configuration, compare the sets of published
   JSON pointers after excluding profile metadata itself. CI asserts
   `summary ⊂ all ⊂ debug`; both inclusions are strict. Values shared by profiles must be
   identical. A lower profile may omit a field but may not change its value, unit, provenance,
   absence state, or validation tier.
6. **Unknown-field behavior.** A reader for schema major `N` accepts documented additive
   fields from later compatible schema minors and rejects a record requiring an unsupported
   schema major.
7. **NPZ sidecar linkage.** When a dense-matrix sidecar is present, its declared hash, shape,
   dtype, and record identifier must match the sidecar bytes. The sidecar is not included in
   the JSON size budget.

### 24,000-byte summary regression

The 24,000-byte limit is a contract test, not a documentation target.

Create one pinned 10-second, 12-lead reference case at
`tests/fixtures/golden/reference_10s_12lead/`. Produce its `summary` profile using the
default production configuration and serialize it with the canonical compact JSON encoder:
UTF-8, sorted keys, no insignificant whitespace, and no ASCII escaping of ordinary Unicode.
The test computes `len(encoded_bytes)` and fails when the result is greater than 24,000.

The current design envelope is 22,295 bytes, leaving 1,705 bytes of headroom. The CI assertion
is nevertheless `encoded_size <= 24000`; it must not be relaxed merely because a new field
is useful. A schema change that cannot fit must remove or relocate data, or explicitly revise
the envelope through the schema/versioning process. The optional dense NPZ sidecar, expected
to be roughly 24,240 raw bytes for the dense matrix, is measured and reported separately.

CI should also print the pinned record's actual byte count and delta from 24,000 so gradual
growth is visible before it reaches the hard limit.

## Golden fixtures

### What is frozen

Each golden case freezes four things:

- the exact raw-signal bytes, or a content-addressed signal object, with a SHA-256 hash;
- input metadata relevant to extraction, including sampling rate, lead ordering, units, and
  duration;
- the complete pipeline configuration and policy-object settings;
- the expected ECG Record output for the profiles exercised by that case.

A golden result is therefore identified by signal hash + metadata + configuration + schema
version. Updating an expected JSON file without changing or re-verifying that identity is not
an acceptable fixture update.

### Layout and size limits

Use:

```text
tests/fixtures/golden/
  <case-id>/
    manifest.json
    signal.npz
    record.summary.json
    record.all.json
    record.debug.json        # only when the case specifically exercises debug output
```

`manifest.json` records SHA-256 hashes, sampling information, configuration, code baseline,
schema version, fixture purpose, and source/licensing notes. Do not copy a source ECG into the
repository when its dataset license forbids redistribution; in that case keep only a manifest
with the external dataset record identifier and hash, and run that case in the controlled
dataset benchmark environment.

For redistributable checked-in fixtures, cap each case at 1.5 MiB compressed and the whole
golden corpus at 25 MiB. A fixture exceeding either limit must move its signal payload to
content-addressed CI storage and retain only its manifest and expected compact records in the
repository. This keeps ordinary clones and per-PR tests small while still allowing realistic
10-second, 12-lead cases.

### Deliberate update procedure

A golden change is accepted only when the review contains:

1. the reason for the behavioral change and the field paths affected;
2. the old/new canonical JSON diff, summarized by changed JSON pointers rather than a raw
   multi-thousand-line blob;
3. the old and new summary byte counts;
4. the relevant unit/stage tests showing why the new value is intended;
5. benchmark evidence for every changed field that is labelled `validated` or
   `partially_validated`;
6. confirmation that unrelated golden cases did not change.

The fixture updater must regenerate one named case at a time. Bulk acceptance of regenerated
goldens is prohibited as a review shortcut. If an algorithm intentionally changes many cases,
the review may contain a single rationale, but it must still report per-field and per-case
diff statistics and pass the same benchmark gates.

Golden fixtures detect unintended behavioral drift. They are never evidence that a field is
clinically correct simply because its previous value was reproduced.

## Property-based tests

Use Hypothesis for generated tests. Generated signals should include multiple supported
sampling rates, durations, lead subsets, constant/near-constant signals, finite low-amplitude
noise, boundary-valued metadata, and record fragments used to exercise the serializer. Keep
resource limits strict enough that these tests remain per-PR tests.

At minimum, encode the following invariants:

- **Units are sampling-rate invariant.** Changing the sampling rate changes sample indexes and
  resampling behavior, but a field declared in milliseconds, millivolts, degrees, hertz, or
  another published unit never silently changes unit.
- **Fiducial ordering holds.** Whenever all three are measured, onset ≤ peak ≤ offset for a
  wave or complex. Violations produce an explicit unavailable state or error; they are never
  emitted as valid measurements.
- **Intervals are non-negative and derived consistently.** A duration derived from two
  published fiducials equals their difference within the documented rounding tolerance.
- **Absence always has a reason.** `unavailable` and `not_applicable` can never be encoded
  without a non-empty reason; `measured` can never encode one.
- **Profiles are monotone.** For identical input/configuration, `summary` is a strict subset
  of `all`, and `all` is a strict subset of `debug`; common paths carry identical data.
- **Round-trip stability.** Decode → encode → decode preserves the complete semantic record,
  including provenance, validation tiers, and absence reasons.
- **Canonical encoding is idempotent.** Encoding a decoded canonical record twice produces
  byte-identical UTF-8.
- **RFC 6901 addressing is total over published fields.** Every generated published leaf has a
  valid pointer that resolves to that same leaf.
- **Lead permutation cannot relabel measurements.** Reordering input leads while preserving
  lead names may change processing order but cannot swap lead identities in output.
- **Finite-input/finite-output rule.** For finite, in-range generated signals, a measured
  numeric output must be finite. NaN and infinity are represented as unavailable with a
  reason, never as JSON numeric values.
- **Serialization never upgrades validation.** Decoding, profile conversion, or re-encoding
  cannot change `unvalidated` to `partially_validated` or `validated`.

Property tests should shrink failures to reproducible examples, and every production bug found
this way should gain a small explicit regression example when the shrunk case is stable enough
to preserve.

## Benchmark harness as a release gate

### Baseline model

Turn the ten existing root-level tools into one benchmark protocol without assuming their
current command-line flags. A thin harness should invoke each tool with explicit pinned input
locations and normalize its output into a machine-readable result containing:

- tool name and tool version/hash;
- dataset name, dataset release, record subset, and dataset-content manifest hash;
- extraction configuration and ECG Record schema version;
- metric name, value, direction (`higher_is_better` or `lower_is_better`), and unit;
- baseline value and allowed regression;
- environment details, including Python, NumPy, SciPy, OS, and optional accelerator state.

Store accepted baseline metadata under `benchmarks/baselines/<baseline-id>/`; large raw tool
outputs belong in CI artifact storage, not in the source tree.

The **Validation Owner** is the baseline owner. Before the first public release, this must be
an explicitly assigned maintainer role in repository ownership metadata. The Validation Owner
approves benchmark-baseline changes; the Release Manager verifies that the referenced
baseline is the one used for a release. An implementation author cannot silently replace a
baseline in the same change that regresses against it.

### Common regression tolerances

Use these default tolerances unless a benchmark specification gives a stricter one:

- bounded accuracy-like metrics reported in percentage points (sensitivity, PPV, F1, recall,
  precision, specificity, AUROC, or equivalent): **no regression beyond 0.5 percentage
  points against the pinned baseline** for an aggregate metric, and no regression beyond
  1.0 percentage point for any predeclared subgroup/lead/wave metric;
- timing or amplitude error metrics where lower is better: **no regression beyond 2% relative
  or one reporting quantum, whichever is larger, against the pinned baseline**;
- coverage/yield metrics: **no regression beyond 0.5 percentage points against the pinned
  baseline**;
- crash/error counts on a fixed corpus: **no regression beyond 0 new failures against the
  pinned baseline**;
- exact snapshot or schema-contract results: **0 unexpected differences**.

The benchmark manifest must state the metric family and reporting quantum before a candidate
release is run. A release cannot choose a more favorable tolerance after seeing its result.

These values are engineering release tolerances, not claims of clinical equivalence. They
bound regressions from an accepted baseline; they do not by themselves establish that the
baseline is good enough for clinical use.

### Cadence for the existing tools

The digest establishes the tool names and dataset references, but not every tool's exact CLI
or emitted metric vocabulary. The harness must discover and pin those details during
implementation; this design does not invent flags that were not inspected.

| Existing tool | Cadence | Release-gate condition |
| --- | --- | --- |
| `snapshot_regression.py` | Every PR | Run on checked-in golden cases. Pass only with 0 unexpected snapshot differences and all changed snapshots explicitly approved through the golden-update procedure. |
| `score_measurement_verifiable.py` | Every PR on the checked-in/pinned small corpus; full corpus per release | Every normalized accuracy/coverage metric stays within the common tolerance, and there are 0 new extraction failures. |
| `evaluate_ludb.py` | Every release; optional on PRs touching delineation | On the pinned LUDB subset/version, aggregate accuracy-like metrics lose ≤0.5 percentage points, predeclared wave/lead metrics lose ≤1.0 point, error metrics worsen ≤2%, and failures do not increase. |
| `evaluate_qtdb.py` | Every release; optional on PRs touching delineation/intervals | Same normalized thresholds against the pinned QTDB baseline; 0 new failed records. |
| `evaluate_ludb_cse.py` | Every release | Same normalized thresholds against its pinned LUDB/CSE baseline; 0 new failed records. |
| `compare_annotations.py` | Every release when annotation or fiducial code changed; otherwise manual quarterly audit | All normalized matching/accuracy metrics stay within the common thresholds; 0 unexplained annotation-mapping changes. |
| `compare_ludb_detectors.py` | Manual before any detector-selection/policy change; required for a release containing such a change | The production detector may change only with a recorded comparison. The selected detector must not regress any declared aggregate metric by >0.5 points or an error metric by >2% against the prior production detector baseline unless the baseline itself is deliberately replaced with documented evidence. |
| `analyze_physionet_st.py` | Every release that changes ST measurement; otherwise manual before a schema release exposing new ST fields | On the pinned EDB/PhysioNet ST corpus used by the tool, declared accuracy metrics lose ≤0.5 points, error metrics worsen ≤2%, and no field with an `unvalidated` ceiling is promoted. |
| `evaluate_target_ecgfeat_diagnosis.py` | Every `ecginterpret` release and every core release that changes fields consumed by interpretation | Against its pinned target corpus, aggregate bounded metrics lose ≤0.5 points, subgroup metrics lose ≤1.0 point, and record-processing failures increase by 0. Core releases need this compatibility gate even though interpretation ships separately. |
| `batch_extract_ecgfeat.py` | Release candidate on the pinned batch corpus; manual for very large full-dataset runs | 0 new record failures, 100% of produced records pass schema validation, coverage drops by ≤0.5 points, and the pinned summary reference still satisfies the 24,000-byte hard limit. |

A release is blocked if a required dataset job cannot run. “Dataset unavailable” is not a
passing result. The Release Manager may reschedule the candidate, but may not waive the gate
without publishing a release note that the affected capability is excluded from that release's
validation claims.

## Keeping validation tiers honest

Validation status is per field and must be mechanically tied to evidence. It must never be a
free-form label chosen by the serializer.

### Field registry

Define one authoritative registry, proposed as
`schemas/ecg-record/validation-evidence.toml`, keyed by stable ECG Record JSON pointer. Each
published measurement entry contains:

- `validation_tier`: `validated`, `partially_validated`, or `unvalidated`;
- `validation_ceiling`: the best tier this field is currently allowed to claim;
- `evidence_ids`: zero or more references into pinned benchmark-result manifests;
- `validated_for`: the population/task/measurement claim actually supported by the evidence;
- `known_limitations`: mandatory for `partially_validated` and any field with material
  scope limits;
- `publication_tier`: published measurement, provenance, or internal debug.

The schema generator and record serializer consume this registry; they do not accept a
call-site override that can improve a tier.

### Mechanical promotion rule

CI fails if a field is labelled `validated` unless all of these are true:

1. its `validation_ceiling` is `validated`;
2. it has at least one non-stale `evidence_id`;
3. every referenced evidence manifest identifies a benchmark tool, pinned dataset
   version/hash, baseline, metric, acceptance threshold, and passing result;
4. the evidence's declared `validated_for` scope includes the field's published claim;
5. the evidence was produced by the current schema major and by code no older than the last
   algorithm change affecting that field.

`partially_validated` has the same traceability requirement but may cite evidence with a
restricted population, lead set, rhythm class, or benchmark scope. `unvalidated` requires no
positive benchmark claim and cannot be represented as validated in docs, schema annotations,
or runtime provenance.

The documentation build must generate a field-by-field validation table directly from the
registry and evidence manifests. Hand-written validation labels in API prose are prohibited.

### Hard ceilings for misleading or compatibility fields

`st_morphology` is the canonical negative-control field: the four-class measurement was
never validated and is anti-correlated with ischemia. Its registry entry must set
`validation_ceiling = "unvalidated"`. CI contains an exact assertion for that ceiling and
fails if any generated schema, record, or documentation output reports a better tier.

The 22 `st_hybrid_*` keys and 16 `twelve_sl_*` keys are internal debug/compatibility
extensions. Their `publication_tier` is capped at internal debug, so they cannot become
published measurements merely because a profile or serializer starts exposing them.

Changing either a validation ceiling or publication tier is a reviewed evidence change. A
code refactor, rename, schema regeneration, or fixture update cannot promote it implicitly.

## CI design

There is currently no `.github/workflows` directory, so none of the 124 tests is an automated
gate. The first packaging-ready release must add CI before publication.

### Required matrix

Support Python 3.10–3.13 for the first redesigned release. Exercise both NumPy families without
pretending every cross-product is installable:

| Runner | Python | NumPy | Purpose |
| --- | --- | --- | --- |
| Ubuntu | 3.10 | 1.26.x | Oldest Python + last NumPy 1.x family |
| Ubuntu | 3.11 | 1.26.x | NumPy 1.x compatibility |
| Ubuntu | 3.12 | 1.26.x | NumPy 1.x on the current development Python generation |
| Ubuntu | 3.12 | 2.2.x | Reproduce the measured development family (`numpy 2.2.6`) |
| Ubuntu | 3.13 | latest allowed 2.x | Newest supported Python and dependency family |
| macOS | 3.12 | latest allowed 2.x | OS smoke and wheel/import behavior |
| Windows | 3.12 | latest allowed 2.x | OS smoke, paths, and wheel/import behavior |

The dependency resolver must also have one Linux job using the oldest supported SciPy and one
using the newest allowed SciPy. Exact dependency bounds belong to the packaging design; CI
must test both ends of every declared range.

### What runs where

Every PR runs, on Ubuntu, schema/contract tests, engine units, pipeline stage tests, golden
fixtures, property-based tests, `snapshot_regression.py`, and the small-corpus
`score_measurement_verifiable.py` gate. The full Python/NumPy Linux matrix runs the
non-dataset suite. macOS and Windows run contract tests, import/package smoke tests, and a
representative pipeline/golden subset.

PR wall-clock target: **≤12 minutes** with jobs parallelized. No individual non-dataset shard
may exceed 8 minutes. A job that exceeds its budget on three consecutive runs is split or
optimized; the timeout is not simply increased.

Release candidates run the complete non-dataset matrix plus every required dataset benchmark
listed above. Dataset jobs may take hours and therefore run in a separately triggered
`release-validation` workflow whose successful result is required before a release tag is
published.

### Dataset-dependent jobs

Large or non-redistributable datasets do not belong in the repository or ordinary hosted-CI
caches. Run them on an authorized self-hosted runner or controlled benchmark environment with
read-only dataset mounts. CI receives only dataset identifiers/manifest hashes and publishes
metrics, logs, and normalized result manifests as artifacts; it never uploads restricted ECG
data.

Each dataset job first verifies the dataset manifest hash. A missing record, changed annotation
set, or mismatched dataset version fails before scoring so a changed corpus cannot masquerade
as an algorithm regression or improvement.

For public datasets whose terms allow automated retrieval, a future cache may download them
from the canonical source, but only after licensing and reproducibility have been documented.
Until then, the controlled runner is the release source of truth.

## Failure triage

A failing gate is classified before any baseline is touched:

- **contract failure:** fix the record/schema implementation or intentionally version the
  contract;
- **golden drift:** explain the changed fields and run the required benchmark evidence;
- **benchmark regression:** fix the algorithm or deliberately replace the baseline with
  Validation Owner approval and documented evidence;
- **environment-only failure:** reproduce on the declared dependency boundary and either fix
  compatibility or narrow the declared support range before release;
- **dataset-integrity failure:** repair the dataset mount/manifest; do not record a benchmark
  result from an unverified corpus.

Flaky tests are failures. A test may be quarantined only with an issue, an owner, and an expiry
date; release-critical schema and benchmark gates cannot be quarantined.

## Open questions

1. Which exact releases and legal terms apply to the LUDB, QTDB, EDB, PTB-XL, BUT, NSTDB, and
   GUDB copies that will define the first public baseline? This must be recorded before the
   release-validation runner is provisioned.
2. The prepared inventory identifies the ten benchmark tools but does not establish each
   tool's current CLI or complete metric vocabulary. The first harness implementation must pin
   those outputs before baseline files are accepted; no undocumented metric may be silently
   dropped from the normalized result.
3. Which maintainer will hold the named Validation Owner role? Publication must not proceed
   until that ownership is explicit.
4. Which small, redistributable ECGs can legally be checked in as golden signals? Cases without
   clear redistribution rights must remain manifest-only and run in the controlled dataset
   environment.
