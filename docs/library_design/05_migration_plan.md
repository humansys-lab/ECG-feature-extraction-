# Migration plan

> Implementation status: this document includes target contracts, not completed
> release claims. [Document 08](08_implementation_status.md) records the implemented
> subset, reconciled decisions and outstanding gates. Document 02 is authoritative
> for the JSON layout; Python carrier sketches must not create a second wire format.

This migration changes package structure and the published data shape while preserving the measurements produced by the current extractor. Structural work and behavioral work therefore stay separate: every structural phase must prove parity against a frozen pre-migration baseline, and any future algorithm change must be reviewed and gated independently.

## A. Migration principles

### Preserve measurement behavior while structure changes

The migration may change module locations, package ownership, import paths, orchestration boundaries, and the JSON document shape exposed by the ECG Record API. It must not silently change measurement values, default configuration, signal interpretation, applicability decisions, or the behavior of currently opt-in refinements.

The baseline configuration is frozen before any code moves. Its resolved defaults are `fs_internal=None` (native sampling frequency), `mains_freq=50`, `input_mode="standard"`, and `st_amplitude_source="analysis"`. Experimental refinement flags remain off unless a golden case explicitly exercises one flag. The migration must not enable refinements by default or combine them merely to increase coverage; some combinations are known to degrade results.

A structural phase is not a chance to fix a measurement bug. If parity exposes a suspected bug in the current implementation, record it and defer the behavioral change to a separate change with independent before/after evidence. That keeps the migration baseline trustworthy.

The new supported consumer boundary is the versioned ECG Record plus the raw signal. Interpretation is separately versioned and separately distributed. The internal `_engine` tree is not a compatibility surface; once direct tests have been redirected through public APIs or explicit internal test seams, `_engine` modules may be reorganized without deprecation guarantees.

### Compatibility is finite and explicit

Legacy imports that are in active use do not disappear when implementations move. A compatibility module forwards them to the replacement and emits `DeprecationWarning` with `stacklevel=2`. The warning names the replacement API and the earliest release in which the legacy import may be removed. Tests assert both the warning class and message so that a silent compatibility layer cannot become permanent.

A deprecated import remains functional for two consecutive minor releases after the release that first warns. If release N first emits the warning, releases N and N+1 still import and behave compatibly; N+2 is the earliest compatibility-breaking release that may remove the import. Removal never happens in a patch release.

At removal, the old module path becomes a one-release tombstone that raises `ImportError` with a short migration message naming the replacement import. After that tombstone release the module may disappear, at which point Python's normal `ModuleNotFoundError` is acceptable. This policy applies in particular to legacy `api`, `models`, `export`, and interpretation-facing imports. Old object-return behavior is implemented by `ecgfeat.compat.api_v0`; legacy model adapters live in `ecgfeat.compat.models_v0`; legacy JSON export lives in `ecgfeat.compat.export_v0`.

The repository-local test imports under `feature_extraction.ecgfeat.*` are migration debt, not a new public namespace. They are kept working long enough to migrate the suite, while the supported installed import name remains `ecgfeat`.

### Version new contracts before making them defaults

The ECG Record schema receives an explicit schema version before any consumer is
switched to it. Field addresses use
`ecg-record:<record_id>@<schema_version>#<RFC 6901 JSON pointer>`.
A measured value is not an absence. The three wire absence states are plain
null, `unmeasurable(reason)`, and `not_applicable(reason)` (document 02).
The low-level `Unavailable` carrier maps to wire `unmeasurable`; imported plain
null remains distinct and is allowed only for nullable fields.

The interpretation document has its own version. Core extraction does not import the interpretation distribution eagerly. During the compatibility window, the old interpretation entry point forwards lazily to `ecginterpret` and raises a targeted installation/migration error if that optional distribution is absent.

No new distribution should be published from this repository until the absence of a repository `LICENSE` file is resolved. The migration may be implemented and tested before then, but a release artifact must not guess at licensing terms.

### A phase is done only when its claim is falsifiable and proven

Every phase must satisfy all of these conditions:

1. Its phase-specific parity gate passes against the frozen baseline.
2. The relevant `pytest` slice passes, and the full 124-file suite passes before the phase joins the migration branch.
3. Legacy imports covered by the phase either still behave identically or emit the specified deprecation warning with an asserted replacement path.
4. New record or interpretation surfaces have contract tests for schema version, serialization, addressing, and absence-state semantics where applicable.
5. The rollback path has been exercised at least once in a test or staging run.
6. No unexplained golden diff is waived. Expected shape diffs must be accounted for by a reviewed crosswalk, never by blanket snapshot regeneration.

Because the repository has no CI configuration today, Phase 0 creates the gate rather than assuming one exists. Until that gate is running, no code-moving phase is considered started.

## B. Parity-proof mechanism

### Freeze the baseline before moving code

Before the first module move, create an immutable golden corpus from LUDB, QTDB, EDB, PTB-XL, BUT, NSTDB, and GUDB. Use two tiers:

- A 96-record sentinel set on every migration pull request: 24 LUDB, 20 QTDB, 16 EDB, 16 PTB-XL, 8 BUT, 6 NSTDB, and 6 GUDB records. Select record IDs once with a deterministic documented rule and freeze them. If a dataset has fewer eligible records, use all eligible records and record that exception in the manifest.
- A full evaluation set equal to the records exercised by the existing dataset-specific evaluation jobs. This slower tier is required before completing high-risk phases and before a release, but not for every small commit.

The prepared inventory does not provide exact installed dataset record IDs, so this plan does not invent them. Phase 0 must write the chosen IDs into the golden manifest before any structural change lands.

For each golden record, freeze:

- dataset name, record ID, source file identities, sampling frequency, units, and lead order;
- SHA-256 of each source signal file where a stable source file exists;
- SHA-256 of the canonical loaded signal array, defined as C-contiguous little-endian float64 bytes, plus a separate metadata hash for shape, sampling frequency, units, and lead names;
- the fully resolved extraction configuration and a SHA-256 of its canonical JSON form;
- the exact legacy JSON bytes from the current exporter, plus a parsed leaf-value digest independent of object key order;
- the current library version/commit identifier and the Python/dependency environment used to generate the baseline;
- per-tool baseline metrics for the existing evaluation scripts.

The primary pinned configuration is `standard-default-v0`: `fs_internal=None`, `mains_freq=50`, `input_mode="standard"`, `st_amplitude_source="analysis"`, with all experimental refinements disabled. Freeze a smaller targeted subset for `input_mode="limited"`, for `st_amplitude_source="calibrated_pr"`, and for `st_amplitude_source="adaptive_pr_tp"`. Pin experimental refinement cases one flag at a time only where current tests or regressions already exercise that flag; do not create an all-flags-enabled golden configuration.

Do not commit raw dataset signals merely to make CI convenient. Until dataset redistribution rights and the repository license are settled, commit manifests, hashes, and expected outputs only. The authoritative full-data parity job runs on a controlled or self-hosted runner with datasets mounted read-only.

### Make `snapshot_regression.py` the standing parity mechanism

`snapshot_regression.py` should become the reusable parity entry point rather than remain a one-off 90-line utility. Keep it small: it loads a manifest, invokes a selected extraction surface, applies only the normalization allowed for that phase, compares against the frozen result, and emits a machine-readable diff report. If the comparison logic grows, move that logic into a regression helper module and keep `snapshot_regression.py` as the stable CLI front end used by CI and release checks.

The harness needs three explicit comparison modes:

1. `legacy-bytes`: exact byte-for-byte equality of legacy JSON. Key order, float rendering, null placement, and list order must match. Use this for mechanical moves and the `api.py` decomposition because those phases claim no externally observable change.
2. `record-crosswalk`: compare parsed legacy leaves to the new ECG Record through a reviewed path crosswalk. Object key order is ignored, but mapped measurement scalars, enums, booleans, semantically ordered lists, and reason codes compare exactly. Use this for the record redesign because the document shape intentionally changes.
3. `canonical-document`: compare canonicalized JSON for separately versioned documents where serialization key order is not a contract. Use this for interpretation packaging once old and new interpretation fields have an explicit semantic crosswalk.

No migration phase may hide a measurement change behind numeric tolerances. Measurement leaves are exact in `legacy-bytes` and `record-crosswalk`. Tolerance-bounded comparisons are reserved for aggregate external-reference benchmark metrics from LUDB/QTDB/EDB/PTB-XL tooling. Freeze those tolerances in Phase 0 from repeated baseline runs on supported environments, before any migrated result is observed; they may not be widened after a failure.

### Gate the shape-changing record redesign without weakening the oracle

The ECG Record cannot be compared byte-for-byte with the old export, so the redesign gets two independent gates.

First, maintain a checked path crosswalk from every legacy measurement/export field consumed by tests or known consumers to its new RFC 6901 pointer. The crosswalk also states how each legacy missing/null state maps to `measured`, `unavailable(reason)`, or `not_applicable(reason)`. `snapshot_regression.py --mode record-crosswalk` compares the new record directly with the frozen old payload using this table. The crosswalk is reviewable data, not code hidden inside the compatibility serializer.

Second, `ecgfeat.compat.export_v0` projects the new record back to the old payload and must reproduce the frozen legacy JSON bytes exactly for the golden corpus. This checks the compatibility behavior that old consumers actually use.

These gates must not share an oracle implementation. In particular, the crosswalk comparator must not call `compat.export_v0`; otherwise a bug in the adapter could make both sides agree. New fields with no legacy equivalent are validated by record schema/contract tests and provenance assertions rather than by inventing an old counterpart.

### Assign every existing regression tool a gate role

| Tool | Required phase gate | Purpose |
|---|---|---|
| `snapshot_regression.py` | Phase 0 onward; every phase | Primary old-vs-new parity harness over the golden manifest. |
| `evaluate_ludb.py` | Phases 1 and 3; full set before Phase 3 completion | Detect delineation onset/offset regressions after internal moves and pipeline decomposition. |
| `evaluate_qtdb.py` | Phases 1 and 3 | Guard QT/QRS/T measurements and boundary behavior. |
| `evaluate_ludb_cse.py` | Phases 1 and 3 | Guard CSE-style delineation/measurement behavior while modules move. |
| `compare_annotations.py` | Phases 1 and 3 | Compare beat/fiducial annotations between legacy and migrated paths. |
| `compare_ludb_detectors.py` | Phases 1 and 3 | Guard QRS detector equivalence while detection code moves under `_engine`. |
| `analyze_physionet_st.py` | Phases 1, 3, and 4 | Guard ST localization/amplitude paths, including pinned `st_amplitude_source` variants. |
| `batch_extract_ecgfeat.py` | Phases 2 and 4 | End-to-end batch serialization/compatibility gate for legacy export and the new record path. |
| `score_measurement_verifiable.py` | Phases 2 and 4 | Verify that measurement values remain traceable after provenance and record-layer changes. |
| `evaluate_target_ecgfeat_diagnosis.py` | Phase 5 | Guard the interpretation-distribution split and its compatibility adapter. |

These tools are planned gates, not commands to run during this documentation task. Phase 0 captures their baseline outputs, and later runs compare with those frozen baselines.

### Create CI before depending on CI

Phase 0 creates two GitHub Actions workflows because `.github/workflows` does not exist today:

- `.github/workflows/test.yml` runs the 124-file `pytest` suite, deprecation-warning checks, schema/contract tests, and the sentinel snapshot job on data legally available to that runner.
- `.github/workflows/parity.yml` runs the full golden manifest and required dataset tools on a controlled self-hosted runner with LUDB/QTDB/EDB/PTB-XL/BUT/NSTDB/GUDB mounted read-only. It uploads the parity report and fails on any unexplained diff.

If the project later moves away from GitHub Actions, the required jobs remain the same. During this migration, the absence of a functioning parity runner blocks Phases 1 through 5; it is not a reason to waive the gate.

## C. Phase plan

The sequence deliberately puts the highest-risk change, decomposing `api.py`, after the baseline corpus, reusable snapshot harness, CI, low-risk relocation work, and record crosswalk are already proven.

### Phase 0 — freeze behavior and install the gates

**What moves:** No production code. Create the golden manifest and frozen outputs, promote `snapshot_regression.py` into the reusable harness, record benchmark baselines, and create the two CI workflows. Add only the minimum regression support needed to run those gates.

**Compatibility shim:** None; current behavior remains the source of truth.

**Gate:** Generate the same snapshots twice from unchanged code and require zero `legacy-bytes` diffs. Run all 124 tests. Run every dataset evaluator needed by later phases and freeze its baseline metrics/tolerances before seeing migrated output.

**Expected duration/risk:** 3–5 engineer-days; medium risk because a bad baseline invalidates every later proof.

**Rollback:** No runtime rollback is needed because production behavior has not moved. If the corpus or runner is unstable, discard the untrusted baseline artifacts, fix determinism/data mounting, and regenerate from the untouched implementation before Phase 1.

**Done when:** sentinel and full manifests are immutable and reviewable; two baseline runs agree; both CI workflows execute; the 124-file suite is green; and every later phase has an assigned gate.

### Phase 1 — move low-risk internals under `_engine`

**What moves:** Mechanically relocate cohesive implementation modules that do not define the new record or interpretation boundary: numeric/foundation helpers, preprocessing, acquisition/signal quality, QRS detection, beat grouping/representatives/families, delineation helpers, atrial helpers, and measurement helpers according to the 46-module mapping. Do not decompose `api.py`, redesign `models.py`, replace `export.py`, or split interpretation yet. Preserve code paths and call order.

**Compatibility shim:** Existing importable module paths forward to `_engine` destinations for the compatibility window. Tests that intentionally exercise internals may temporarily use those shims; new package code imports the destination modules directly.

**Gate:** `snapshot_regression.py --mode legacy-bytes` is byte-identical for the complete sentinel set. The full phase gate also runs `evaluate_ludb.py`, `evaluate_qtdb.py`, `evaluate_ludb_cse.py`, `compare_annotations.py`, `compare_ludb_detectors.py`, and `analyze_physionet_st.py` against frozen baselines. The full 124-file suite passes.

**Expected duration/risk:** 5–8 engineer-days; medium risk. The moves are mechanical, but `delineate.py`, `features.py`, pacing/QT helpers, and cross-module imports are highly connected.

**Rollback:** Restore imports to original module implementations and leave new `_engine` modules unused. Because no public result type changes, rollback is a routing change rather than a data migration.

**Done when:** every moved module has one implementation owner, legacy imports still resolve with the specified warning behavior, legacy JSON is byte-identical on the sentinel corpus, and all assigned full-data evaluator gates remain within frozen thresholds.

### Phase 2 — introduce the ECG Record in shadow mode

**What moves:** Add `ecgfeat.record` (`model`, `availability`, `provenance`, `validation`, `serialize`, `query`) and split record responsibilities out of `models.py` and `export.py`. Keep the existing extractor result and legacy export authoritative. Build a new record from the same completed legacy measurement state in shadow mode, but do not switch consumers yet. Establish the legacy-to-record pointer crosswalk and explicit mappings from legacy missing/null states to the three absence states.

**Compatibility shim:** `ecgfeat.compat.models_v0` preserves legacy dataclass names/adapters. `ecgfeat.compat.export_v0` preserves existing `to_dict` profiles and exact legacy payloads. Existing `ecgfeat.models` and `ecgfeat.export` imports forward to those compatibility surfaces and warn.

**Gate:** `snapshot_regression.py --mode record-crosswalk` shows exact mapped leaf equality, and the `compat.export_v0` projection also passes `legacy-bytes`. `batch_extract_ecgfeat.py` completes through legacy and shadow-record serialization paths with no legacy-output diff. `score_measurement_verifiable.py` meets its frozen baseline. Record contract tests verify schema versioning, bounded serialization, RFC 6901 addressing, provenance, and all three absence states.

**Expected duration/risk:** 5–7 engineer-days; medium-high risk because `models` and `export` are the two most-imported modules in the suite, at 40 and 39 test files.

**Rollback:** Disable shadow-record construction and route all serialization through the untouched legacy result/export path. Keep record code present but unreachable from default extraction until the crosswalk is corrected.

**Done when:** every legacy field used by the 22 identified export-contract consumers has a reviewed crosswalk rule or an explicit documented reason for removal; mapped values are exact; `compat.export_v0` is byte-identical on the golden corpus; and no existing consumer has been forced onto the new shape.

### Phase 3 — decompose `api.py` into pipeline stages and policies

**What moves:** Replace the 3,155-line orchestration body with `ecgfeat.pipeline.extractor`, the eight named stages, and the four policy objects. Move all 49 private helpers to their assigned destinations from the package-tree mapping. Pacing and QT rescue decisions become named policy results with reason codes and provenance, while their actual decisions and measurement values remain unchanged. The legacy object-return entry point becomes `ecgfeat.compat.api_v0`.

**Compatibility shim:** Existing `ECGFeatureExtractor` construction and old `api` imports route through `compat.api_v0`, which calls the staged pipeline and adapts the result back to the legacy object contract. Legacy callers do not instantiate stage objects.

**Gate:** This is the strongest parity phase. Require full-corpus `legacy-bytes` equality through the compatibility entry point, exact shadow-record crosswalk equality, the entire 124-file suite, and all Phase 1 dataset tools. `analyze_physionet_st.py` additionally runs against each pinned ST source configuration. No unexplained change in a pacing/QT rescue reason is accepted even when the final scalar matches.

**Expected duration/risk:** 8–12 engineer-days; high risk, the highest in the migration.

**Rollback:** Keep the pre-decomposition orchestration implementation available behind `compat.api_v0` until Phase 3 and the following release are proven. If any parity or evaluator gate fails, route the public extractor back to that implementation while retaining the staged pipeline behind an internal opt-in path for diagnosis. Do not delete the rollback implementation in the same release that first makes the staged pipeline authoritative.

**Done when:** all 49 helpers have their assigned owner; `api.py` contains no production orchestration; compatibility extraction is byte-identical over the full golden corpus; the shadow record remains exact under the crosswalk; all dataset gates pass; and rollback routing has been exercised.

### Phase 4 — make the ECG Record the primary extraction result

**What moves:** Promote the staged pipeline's finalized ECG Record to the supported extraction result and make `ecgfeat.record.query` the supported field-address access path. The old object graph and old export remain available only through `compat.api_v0`, `compat.models_v0`, and `compat.export_v0`. The public facade shrinks toward `ecgfeat/__init__.py`, `config.py`, `io.py`, `record/*`, and `pipeline/*`.

**Compatibility shim:** Legacy callers continue to receive the old object/export through the deprecated old entry point. Existing `ecgfeat.export` and `ecgfeat.models` imports continue forwarding during the compatibility window.

**Gate:** Primary parity is `record-crosswalk`, not byte equality of the new record, because the shape intentionally changed. Every mapped measurement leaf is still exact; every legacy null/missing case maps to its reviewed absence state; and `compat.export_v0` remains byte-identical. Run `batch_extract_ecgfeat.py`, `score_measurement_verifiable.py`, and `analyze_physionet_st.py`, plus all record contract tests and the full 124-file suite.

**Expected duration/risk:** 3–5 engineer-days after Phase 3; high consumer risk despite lower implementation complexity.

**Rollback:** Switch the facade/default extraction path back to the legacy object adapter while leaving explicit record APIs available. Never reuse a released schema version for changed semantics; if a released schema is defective, issue a new schema version.

**Done when:** the record API is the documented default; all known legacy consumers still pass through shims; the old payload is exactly reproducible from the new record for the golden corpus; and addressing/absence-state behavior is covered by packaged tests.

### Phase 5 — split interpretation into `ecginterpret`

**What moves:** Move `interpret.py`, `clinical_rules/`, `glasgow_rules/`, `statement_engine.py`, diagnostic rhythm/MI/pediatric/Glasgow statement logic, and interpretation-only models into the separate interpretation distribution. Keep measurement-only helpers such as the Glasgow measurement profile in the core engine. Interpretation consumes the versioned ECG Record and produces its own versioned Interpretation document.

**Compatibility shim:** The old core interpretation import is a lazy forwarding shim. It imports `ecginterpret` only when interpretation is requested. If the optional distribution is absent, it raises a targeted `ImportError` naming the install requirement and migration path. The legacy-object-to-record bridge remains available only during the compatibility window.

**Gate:** Core extraction remains unchanged under `legacy-bytes` and `record-crosswalk`. Interpretation parity uses canonical semantic comparison between the frozen legacy interpretation payload and the new separately versioned document through an explicit interpretation crosswalk. `evaluate_target_ecgfeat_diagnosis.py` meets its frozen baseline, and all tests that currently import interpretation pass against the separately installed distribution plus the forwarding shim.

**Expected duration/risk:** 5–8 engineer-days; high risk because 19 test files import `interpret`, additional tests import `clinical_rules`/`glasgow_rules`, and 29 tests touch `ecgagent`.

**Rollback:** Release or select a core build that re-enables the in-core forwarding implementation and pins the last known-compatible interpretation package. If the separate package is already published, do not rewrite the artifact; fix forward with a patch and restore the compatibility route in core.

**Done when:** core can be installed and extract records without the interpretation distribution; installing it restores interpretation through both new API and legacy shim; canonical interpretation parity and diagnostic evaluation gates pass; and no core module eagerly imports interpretation.

### Phase 6 — retire shims after consumers and tests have moved

**What moves:** Remove expired legacy forwarders, reduce the top-level facade to the supported surface, and stop shipping compatibility-only fixtures after their final supported release. This phase occurs only after the test-suite and consumer migrations in Sections D and E are complete.

**Compatibility shim:** During the first breaking release, removed module paths are tombstones raising the targeted `ImportError`. They disappear in the following release.

**Gate:** Run the complete core, interpretation, record-contract, and repo-level integration suites with deprecations treated as errors. Golden record and interpretation gates pass through supported APIs. A clean-environment install proves that no supported test or repository consumer imports `compat` accidentally.

**Expected duration/risk:** 2–4 engineer-days after the compatibility window; medium risk concentrated in overlooked external consumers.

**Rollback:** Restore tombstone/forwarding modules in a patch release if removal breaks a supported consumer that was missed. A published breaking release is not rewritten; recovery is additive.

**Done when:** no supported test or repository consumer uses a deprecated path; tombstone-message tests pass for the breaking release; parity and integration gates pass from clean installs; and the documented compatibility window has elapsed.

### Parallelism and hard ordering

Phase 0 is a strict predecessor of every code-moving phase. Phases 1 and 2 may proceed in parallel after Phase 0 if ownership stays disjoint: Phase 1 must not touch `api.py`, `models.py`, `export.py`, or interpretation modules, while Phase 2 owns the record/model/export split. Both must integrate cleanly before Phase 3 begins.

Phase 3 must be isolated from work that changes extraction orchestration, pacing/QT policy, or measurement call order. The purpose of the sequence is to make an `api.py` parity diff attributable; overlapping behavioral work would destroy that property.

Phase 4 is strictly after Phase 3 because the record becomes the default only after the staged pipeline is parity-proven. Phase 5 package scaffolding and rule-file moves may be prepared in parallel once the Phase 2 record schema is frozen, but interpretation cutover cannot complete until Phase 4 stabilizes the published record contract it consumes. Phase 6 is strictly last and cannot start until both the release-count compatibility window and consumer/test migrations are complete.

## D. Test-suite migration

The current 124-file `tests/` tree should not be copied wholesale into either distribution. Split tests by the contract they protect, while preserving a smaller repository-level integration suite that installs the distributions together and exercises real consumers.

### Core distribution tests

Move tests whose assertions concern extraction, record construction, configuration, I/O, measurement behavior, availability, provenance, serialization, querying, or public core errors into the core distribution's test suite. The core package should own contract tests for:

- the public `ecgfeat` facade, `config.py`, `io.py`, `pipeline`, and `record` APIs;
- schema versioning and bounded record serialization;
- `ecg-record:<record_id>@<schema_version>#<pointer>` resolution;
- `measured`, `unavailable(reason)`, and `not_applicable(reason)` semantics;
- compatibility behavior of `api_v0`, `models_v0`, and `export_v0` for as long as those shims are supported;
- deterministic stage/policy behavior that can be tested without importing `_engine` modules as a public API.

Direct tests of numerical internals may remain distribution-local, but they should import `_engine` only from tests clearly marked as internal implementation tests. No downstream integration test should depend on an `_engine` import.

### Interpretation distribution tests

The 19 files that import `interpret` cannot remain core-package tests after the split. Classify them by what they assert, then move interpretation-owned cases into the interpretation distribution alongside tests for `clinical_rules`, `glasgow_rules`, `statement_engine`, rhythm statements, MI logic, and pediatric rules.

Those tests must consume a serialized or constructed ECG Record fixture rather than an `ECGFeatures` object wherever practical. During the compatibility window, keep a focused adapter-test subset that feeds the legacy object through `ecgfeat_interpret.legacy_adapter` and proves semantic equality with the direct-record path. Delete adapter-only tests when the adapter deprecation window closes.

Interpretation tests should not require the core extractor to run for every case. Freeze small versioned ECG Record fixtures for rule tests so failures distinguish record-contract changes from rule-engine changes. End-to-end extraction-plus-interpretation coverage remains in the repository integration suite.

### Repository-level integration tests

Keep tests at repository root only when they verify interaction among separately owned components or executable consumers. This suite should cover:

- installation of core alone and core plus interpretation;
- `ecgagent` reading the new record boundary;
- legacy forwarding imports across distribution boundaries during the compatibility window;
- `batch_extract_ecgfeat.py` and the supported root-level visualization/reporting tools;
- representative end-to-end extraction, record serialization, interpretation, and agent-query flows;
- packaging failures such as interpretation being absent when a legacy interpretation shim is invoked.

The current `pytest.ini` can continue to point to `tests` while the repository remains a monorepo, but distribution-local test commands must also exist and be run independently in CI. A green root suite alone is insufficient because it can hide accidental cross-package imports that only work from the source checkout.

### Migrate `models` and `export` importers incrementally

The 40 test files importing `models` and 39 importing `export` are too many to rewrite atomically. Use compatibility imports as a controlled bridge:

1. In Phase 2, leave all existing tests green through `ecgfeat.models` and `ecgfeat.export`, now forwarding to `compat.models_v0` and `compat.export_v0` with asserted `DeprecationWarning` behavior.
2. Add new record contract tests first. These establish the replacement API before old tests are touched.
3. Migrate tests in coherent groups: serialization/export contract tests first, then pure model/value tests, then pipeline tests, then consumer/integration tests. Each group changes assertions from legacy object shape to record paths only where the new contract is the subject of the test.
4. Keep a deliberately small compatibility suite exercising every supported legacy profile and public legacy type. Do not retain duplicate legacy-shape assertions in dozens of migrated tests.
5. Before Phase 6, run tests with deprecations treated as errors and require zero unapproved legacy imports outside the compatibility suite.

This approach preserves diagnostic value: while both APIs coexist, a failure in a migrated test can be compared with the still-running compatibility path rather than being obscured by a mass rewrite.

### Test migration exit condition

Test migration is complete when core and interpretation tests each pass from clean installed environments, the repository integration suite passes with both distributions installed, the core-only suite passes without interpretation installed, and no test outside the designated compatibility suite imports deprecated `models`, `export`, `api`, or interpretation paths.

## E. Consumer migration

### `ecgagent/`

`ecgagent/` is the highest-priority in-repository consumer because 29 tests touch it and its evidence layer already treats exported JSON pointers as a contract. Migrate it before making the ECG Record the default for general callers.

The migration sequence is:

1. Add a schema-aware address parser in the agent evidence boundary that accepts `ecg-record:<record_id>@<schema_version>#<RFC 6901 pointer>` and rejects unsupported schema versions before evidence reaches prompts or tools.
2. Replace the current field whitelist in `ecgagent/evidence/diagnostic_contract.py` with a whitelist keyed by schema version and canonical ECG Record pointers. The whitelist remains explicit; schema versioning is not permission to expose every record field.
3. Update `ecgagent/evidence/model_view.py` so the model-facing view resolves only approved addresses and preserves absence-state meaning instead of flattening `unavailable` and `not_applicable` into an undifferentiated missing value.
4. Update `ecgagent/tools/query.py` to use `ecgfeat.record.query` rather than manually navigating the legacy export. Query results should carry the canonical address used, schema version, availability state, and provenance needed by the existing evidence model.
5. During the compatibility window, accept legacy pointers at the agent boundary only through a checked legacy-pointer-to-record-pointer table. Emit one deprecation warning per request/session boundary rather than one warning per field lookup.
6. Run the 22 identified export-contract consumer tests plus all 29 `ecgagent`-touching tests before Phase 4 makes the record primary.

Do not make `ecgagent` depend on `_engine`, legacy model dataclasses, or interpretation internals. Its durable dependency is the record/query contract plus the separately versioned interpretation document when diagnostic statements are required.

### Root-level batch, visualization, reporting, analysis, and plot scripts

Migrate scripts by role rather than filename order.

`batch_extract_ecgfeat.py` moves first because it is both a consumer and a parity gate. Add an explicit output mode during the compatibility window: legacy export remains available, while the new default after Phase 4 is the versioned ECG Record. The script must stamp schema/config provenance and must never silently emit a different shape under the same mode name.

`visualize_ecg.py` and `render_ecgfeat_record_explanations.py` should consume `(raw signal, ECG Record)` and resolve measurements through record pointers/query helpers. Plotting must not reach into pipeline or `_engine` state. This makes later movement into `ecg-records-viz` mechanical.

The `analyze_*` and `plot_*` scripts should be migrated when they are next used as gates or maintained tools. Replace direct legacy-object access with record queries and make expected schema versions explicit near their input boundary. Scripts retained only for historical experiments may stay pinned to `compat` for the two-release window, but they must be marked deprecated and cannot block shim removal indefinitely.

`evaluate_ludb.py`, `evaluate_qtdb.py`, `evaluate_ludb_cse.py`, `evaluate_target_ecgfeat_diagnosis.py`, `compare_annotations.py`, `compare_ludb_detectors.py`, `analyze_physionet_st.py`, `batch_extract_ecgfeat.py`, `snapshot_regression.py`, and `score_measurement_verifiable.py` are migration infrastructure during this work. Their inputs may be adapted, but their baseline semantics and frozen comparison outputs must not be casually rewritten to make a phase pass.

### Compatibility window and discovery

Consumers get the same two-minor-release warning window defined in Section A. Discovery is active, not documentation-only:

- legacy import shims emit `DeprecationWarning` naming the replacement;
- legacy export entry points include a deprecation warning and, where a metadata envelope already permits it without changing the frozen payload, documentation points to the schema-aware replacement;
- CI runs one job with deprecations visible and a later Phase 6 job with them treated as errors;
- the migration guide includes old import, new import, old pointer, new address, and the last release supporting the old path;
- in-repository consumers are searched and migrated before Phase 4 completion, so warnings in maintained repository code are treated as defects rather than left for users to discover.

Consumer migration is done when `ecgagent`, maintained root scripts, and the repository integration tests run against supported record/interpretation APIs with no deprecated imports or legacy pointer lookups except the dedicated compatibility tests.

## F. Rollback and risk register

### Per-phase rollback triggers

| Phase | Failure signal | Rollback action | Unrecoverable consequence |
|---|---|---|---|
| 0 — baseline/gates | Repeated unchanged-code snapshot diffs; unstable evaluator metrics; runner cannot reproduce hashes | Stop all migration work, fix determinism/data mounting, discard and regenerate untrusted baselines | None if caught before structural work; a baseline already used for accepted changes would require re-auditing those changes |
| 1 — `_engine` moves | Any `legacy-bytes` diff, detector/delineation metric outside frozen bounds, or broken legacy import | Route imports back to original modules; leave new modules unused until corrected | None if no release removed originals |
| 2 — record shadowing | Crosswalk mismatch, legacy projection byte diff, wrong absence state, provenance mismatch | Disable shadow record construction and restore legacy export as sole runtime path | A published schema version with wrong semantics cannot be redefined; it must be superseded |
| 3 — `api.py` decomposition | Any full-corpus legacy diff, changed pacing/QT decision/reason, evaluator regression | Route public/compat extraction back to preserved pre-decomposition orchestration | None while old orchestration is retained; deleting it in the first cutover release would remove the fast rollback path |
| 4 — record primary | Consumer failures, pointer/address mismatch, compat projection diff | Restore legacy object adapter as facade/default; keep explicit record API available | Records already emitted with a released schema cannot have their meaning rewritten |
| 5 — interpretation split | Diagnostic evaluator regression, canonical interpretation diff, circular/eager dependency, missing-package failure in core extraction | Re-enable in-core compatibility implementation and pin last known-compatible interpretation distribution | A published bad package version remains published; recovery is a new patch/version |
| 6 — shim retirement | Supported consumer still imports removed path; clean-install integration failure | Restore forwarding/tombstone module in a patch release | The breaking release itself cannot be unpublished or made invisible to users who consumed it |

No rollback regenerates golden output from the failing implementation. A golden baseline changes only for an intentional, separately reviewed behavioral release with independent evidence.

### Top risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Golden corpus is incomplete or nondeterministic | Medium | High | Freeze raw-signal and config hashes, run unchanged baseline twice, use sentinel plus full tiers, and forbid post-failure tolerance widening |
| `api.py` decomposition changes call order or rescue decisions | High | High | Decompose only after Phases 0–2, map all 49 helpers in advance, require full `legacy-bytes` parity plus evaluator gates, retain old orchestration for rollback |
| Record redesign loses legacy semantics behind null/missing translation | High | High | Maintain a reviewed field/pointer/absence-state crosswalk and independently require exact `compat.export_v0` reconstruction |
| `models`/`export` test churn hides real regressions | High | Medium | Migrate test groups incrementally while compatibility path remains green; keep a small explicit legacy-contract suite |
| Interpretation split creates hidden core dependency or version skew | Medium | High | Core-only installation test, lazy forwarding import, explicit supported record-schema range in interpretation, canonical interpretation parity gate |
| `ecgagent` whitelist accidentally broadens or drops evidence | Medium | High | Versioned explicit pointer whitelist, reject unsupported schema versions, preserve availability/provenance, run all 29 agent-coupled tests |
| Experimental refinement defaults drift during refactor | Medium | High | Freeze resolved config in every golden case, keep refinements opt-in, exercise targeted single-flag fixtures, reject config hash changes |
| CI gives false confidence because public datasets are unavailable on hosted runners | High | Medium | Separate fast hosted/sentinel jobs from authoritative self-hosted full-data parity; require full-data gate for high-risk phase completion |
| Missing repository license blocks distribution publication | High until resolved | High for release | Treat licensing as a release blocker; do implementation/testing without inventing or assuming license terms |
| External consumers are missed before shim removal | Medium | Medium | Two-minor-release warning window, repository deprecation-as-error job, published migration table, tombstone release before disappearance |

### What is intentionally not recoverable

Three events cannot be rolled back as though they never happened: a published schema version, a published distribution version, and a compatibility-breaking release consumed by users. For those, rollback means restoring compatibility in a new release and issuing a new version where semantics changed. The plan therefore delays publication and deletion until parity and consumer gates are already proven.

## Open questions

1. **Distribution/import naming must be reconciled before Phase 5 packaging.** The settled architecture calls the separate distribution `ecginterpret`, while the prepared package tree names the import package `ecgfeat_interpret` and the distribution `ecg-records-interpretation`. Pick one import name and one distribution name before scaffolding; do not ship aliases for both unless there is a concrete external compatibility requirement. My recommendation is import package `ecginterpret` because it matches the settled decision and clearly separates interpretation from the `ecgfeat` core namespace.
2. **Golden record IDs are not available in the prepared inventory.** Phase 0 must select and commit the exact LUDB/QTDB/EDB/PTB-XL/BUT/NSTDB/GUDB IDs deterministically before code moves. This plan intentionally does not invent them.
3. **Aggregate benchmark tolerances need empirical baseline runs.** Exact measurement parity is already decided; only evaluator-level floating aggregate tolerances remain to be measured from repeated unchanged-code runs on supported environments.
4. **Dataset redistribution and CI access rights need an explicit owner.** Until licensing and dataset terms are known, the plan assumes hashes/manifests in the repository and full signals on a controlled runner. The team must decide who operates that runner and which datasets may legally be cached there.
5. **Repository licensing must be resolved before publishing `ecg-records`, the interpretation distribution, or visualization distribution.** There is no `LICENSE` file today, so the release phase cannot infer permission or terms.
6. **The exact supported schema-version range between core and interpretation must be declared.** Before Phase 5 completes, the interpretation package should state which ECG Record schema major/minor versions it accepts and fail clearly on unsupported versions rather than guessing from document shape.
7. **Legacy pointer syntax needs a frozen inventory.** The known `ecgagent` files establish that pointer whitelisting exists, but the prepared digest does not enumerate every legacy pointer string. Before migrating `ecgagent`, extract that table once and use it as the auditable old-to-new address crosswalk.
