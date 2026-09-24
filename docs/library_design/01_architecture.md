# 01 — Architecture

Status (2026-09-24): this architecture is implemented (staged pipeline, policies,
private engine, separate interpretation and visualization distributions); layer
rules are enforced by import-linter. See [document 08](08_implementation_status.md).
Design date: 2026-09-22. This document inherits the decisions in
`00_overview.md`: the published consumer boundary is a versioned **ECG Record**
JSON document plus the raw signal, the standard-record budget is 24,000
uncompressed UTF-8 bytes for the reference 10 s 12-lead record, interpretation
has its own versioned document, and the import name remains `ecgfeat` during
migration.

## Architectural decision

The measurement distribution should have one durable outward product:
`(raw signal, ECG Record)`. The extraction engine may remain internally complex,
but visualization, feature queries, reports, diagnostic rules, and `ecgagent`
must not import detector, delineator, feature, or legacy `ECGFeatures` internals.

The target architecture therefore has four boundaries:

1. **Consumer-safe record contract.** `ecgfeat.record` contains the versioned
   record model, availability states, provenance, validation metadata,
   serialization, and read/query helpers. It imports no extraction modules.
2. **Measurement engine.** Signal processing, detection, delineation, quality,
   and measurement remain in the core distribution but under private/internal
   namespaces. They produce internal measurement state, not public JSON-shaped
   dicts.
3. **Pipeline and policies.** `ECGFeatureExtractor.extract()` is decomposed into
   pipeline stages plus explicit policy objects for pacing, QT/reportability,
   applicability, and lead-integrity decisions. This is where experimental paths
   are selected and recorded.
4. **Consumers.** Plotting and clinical interpretation consume `(signal, record)`.
   Diagnostic statement generation moves to a separate distribution. The
   existing top-level API remains as a compatibility facade during migration.

This mirrors the useful part of mature scientific-library design: convenient
entry points and explicit algorithm choices, while keeping the serialized
interchange contract independent from in-memory processing objects. The record
contract, rather than a DataFrame or `ECGFeatures`, is the architectural seam.

## Layer diagram and enforced dependency rules

```text
                           downstream consumers
             +-------------------------------------------+
             | feature query | reports | ecgagent        |
             | ecg-records-viz | ecg-records-interpret  |
             +--------------------+----------------------+
                                  |
                         (raw signal, ECG Record)
                                  |
                    +-------------v--------------+
                    | L4  ecgfeat.record         |
                    | schema / JSON / query      |
                    | availability / provenance  |
                    +-------------^--------------+
                                  |
                       record assembly only
                                  |
                    +-------------+--------------+
                    | L3  pipeline + policies    |
                    | ordered stages / decisions |
                    +-------------+--------------+
                                  |
                    +-------------v--------------+
                    | L2  measurement engine     |
                    | QC / QRS / delineation /   |
                    | atrial / waves / features  |
                    +-------------+--------------+
                                  |
                    +-------------v--------------+
                    | L1  foundation             |
                    | config / internal models / |
                    | numeric / NumPy / SciPy    |
                    +----------------------------+
```

| Layer | Responsibility | Allowed dependencies | Concrete enforcement |
|---|---|---|---|
| L1 foundation | Numeric helpers, immutable configuration, internal-only data models, optional Numba kernel | stdlib, NumPy, SciPy; optional Numba behind current performance extra | `import-linter` contract: `ecgfeat._engine.foundation` cannot import pipeline, record, compatibility, visualization, or interpretation namespaces |
| L2 measurement engine | Preprocess, QC, QRS, grouping, delineation, atrial analysis, wave localization, measurements | L1 and sibling L2 modules | `import-linter` layered contract plus a CI grep/AST check forbidding imports from `ecgfeat.record`, `ecgfeat.compat`, `ecgfeat_viz`, and `ecgfeat_interpret` |
| L3 pipeline + policies | Order stages, select algorithm paths, apply pacing/QT/applicability policies, assemble reproducibility metadata | L1, L2, public record types | only `ecgfeat.pipeline` may import both `ecgfeat._engine` and `ecgfeat.record`; CI asserts no consumer package imports `ecgfeat._engine` |
| L4 record contract | Versioned ECG Record model, serialization, availability states, provenance/validation metadata, field queries | stdlib only for schema/JSON; raw signal remains out-of-band | `import-linter` independence contract: `ecgfeat.record` may not import `ecgfeat.pipeline`, `ecgfeat._engine`, legacy models, plotting, or interpretation |
| Consumer packages | Visualization, reports, rules, agents | `ecgfeat.record`; NumPy only where raw-signal operations are needed | packaging split plus CI import test that builds each consumer with only its declared dependencies and rejects `ecgfeat._engine` imports |
| Compatibility facade | Preserve current imports and shapes long enough to migrate callers | may call L3 and adapters; must not become a dependency of L1–L4 | `import-linter` forbidden-dependency contract from all new modules to `ecgfeat.compat`; deprecation tests exercise old imports separately |

The boundary check should be executable, not advisory. Add `import-linter` as a
development dependency and make its contracts a required CI job. A second small
AST check should reject `ecgfeat._engine` imports from `ecgagent/`, the
interpretation distribution, and the visualization distribution. This catches
dynamic package layout mistakes that a prose rule will not.

## Proposed package and distribution tree

The tree is a target layout, not a request to rename files immediately. Each line
below is one Python module. Private `_engine` modules are deliberately not part
of the supported import surface.

```text
# Core distribution: recommended distribution name "ecg-records"; import package "ecgfeat"
ecgfeat/__init__.py                         # lean public facade + compatibility exports
ecgfeat/config.py                           # RefinementConfig, PWaveConfig, stable extraction configuration
ecgfeat/io.py                               # existing MAT/WFDB-header convenience adapters
ecgfeat/record/__init__.py                  # public record namespace
ecgfeat/record/model.py                     # schema-versioned ECG Record value model
ecgfeat/record/availability.py              # measured / unavailable(reason) / not_applicable(reason) states
ecgfeat/record/provenance.py                # library/schema/config/input/algorithm fingerprints
ecgfeat/record/validation.py                # per-field validation tier and evidence references
ecgfeat/record/serialize.py                 # bounded JSON encode/decode and profile handling
ecgfeat/record/query.py                     # stable field-address queries over decoded records
ecgfeat/pipeline/__init__.py                # public extraction entry point
ecgfeat/pipeline/extractor.py               # stage orchestration replacing api.py god-object flow
ecgfeat/pipeline/context.py                 # internal immutable stage context/result carriers
ecgfeat/pipeline/stages/input.py            # validation, channel contract, resampling, mains resolution
ecgfeat/pipeline/stages/quality.py          # acquisition, lead-integrity, pacing pre-analysis
ecgfeat/pipeline/stages/ventricular.py      # QRS detection and pacing-aware QRS decisions
ecgfeat/pipeline/stages/beats.py            # annotations, grouping, representative selection
ecgfeat/pipeline/stages/delineation.py      # fiducial delineation and opt-in boundary refinements
ecgfeat/pipeline/stages/atrial.py            # P/atrial measurements and validation
ecgfeat/pipeline/stages/measurement.py      # representative/group/global measurement assembly
ecgfeat/pipeline/stages/finalize.py         # availability, reportability, validation, provenance, record build
ecgfeat/pipeline/policies/pacing.py          # PacingPolicy and pacing-rescue decisions
ecgfeat/pipeline/policies/qt.py              # QTPolicy, lead selection, reject/rescue/reportability
ecgfeat/pipeline/policies/applicability.py   # standard vs limited-input field applicability
ecgfeat/pipeline/policies/lead_integrity.py  # lead-reversal/channel-delay consequences
ecgfeat/_engine/foundation/numeric.py        # generic numeric helpers
ecgfeat/_engine/foundation/models.py         # internal beat/group/feature models; not record schema
ecgfeat/_engine/foundation/kalman.py         # optional compiled scalar recursion
ecgfeat/_engine/preprocess.py                # filtering, baseline handling, resampling primitives
ecgfeat/_engine/quality/acquisition.py       # acquisition-chain QC
ecgfeat/_engine/quality/signal.py            # signal, pacing, reversal, detector-agreement QC
ecgfeat/_engine/detection/qrs.py             # multilead QRS detector
ecgfeat/_engine/detection/adaptive_qrs.py    # deterministic per-channel QRS candidates
ecgfeat/_engine/beats/grouping.py            # beat clustering/annotations
ecgfeat/_engine/beats/representative.py      # representative-beat construction
ecgfeat/_engine/beats/families.py            # morphology-family assignment/medoids
ecgfeat/_engine/delineation/core.py          # current delineate_beats core
ecgfeat/_engine/delineation/candidates.py    # classical boundary candidates/sequences
ecgfeat/_engine/delineation/refinement.py    # optional boundary refinements
ecgfeat/_engine/delineation/wave_localization.py # P/S/T morphology localization
ecgfeat/_engine/delineation/r_localization.py # R localization
ecgfeat/_engine/delineation/repolarization.py # T-wave candidates/components
ecgfeat/_engine/delineation/t_refinement.py  # T-offset fusion and optional T refinements
ecgfeat/_engine/atrial/core.py               # residual, atrial-event and contextual analysis
ecgfeat/_engine/atrial/p_wave.py             # P-wave assessments/finalization
ecgfeat/_engine/atrial/p_morphology.py       # P-component measurements
ecgfeat/_engine/atrial/validation.py         # waveform validation of atrial candidates
ecgfeat/_engine/measurement/features.py      # lead/group/global numerical measurements
ecgfeat/_engine/measurement/dispersion.py    # QT dispersion estimands
ecgfeat/_engine/measurement/paths.py         # path classification inputs for QTPolicy
ecgfeat/_engine/measurement/st_baseline.py   # opt-in calibrated/adaptive ST signal
ecgfeat/_engine/measurement/st_localization.py # ST localization/morphology observation
ecgfeat/_engine/measurement/u_wave.py        # U-wave measurement
ecgfeat/_engine/measurement/vector_axis.py   # axis measurement
ecgfeat/_engine/measurement/calibration.py   # empirical QT error calibration utility
ecgfeat/_engine/measurement/profiles/twelve_sl.py # measurement-only 12SL-inspired profile
ecgfeat/_engine/measurement/profiles/glasgow.py   # measurement-only Glasgow-profile quantities
ecgfeat/compat/__init__.py                   # deprecated compatibility namespace
ecgfeat/compat/api_v0.py                     # ECGFeatureExtractor legacy-object adapter
ecgfeat/compat/models_v0.py                  # legacy dataclass aliases/adapters
ecgfeat/compat/export_v0.py                  # existing to_dict profiles and legacy payloads

# Separate distribution: "ecg-records-interpretation"
ecgfeat_interpret/__init__.py                # interpretation entry point
ecgfeat_interpret/engine.py                  # consumes ECG Record (+ signal only if explicitly required)
ecgfeat_interpret/statement_engine.py        # candidate resolution/final statements
ecgfeat_interpret/rhythm.py                  # diagnostic rhythm evidence/rules after measurement helpers move out
ecgfeat_interpret/rhythm_statements.py       # rhythm statement candidates
ecgfeat_interpret/mi.py                      # MI evidence and statement candidates
ecgfeat_interpret/pediatric.py               # pediatric diagnostic thresholds/rules
ecgfeat_interpret/glasgow.py                 # Glasgow-style statement payload/catalog logic
ecgfeat_interpret/legacy_adapter.py          # temporary ECGFeatures-to-record compatibility bridge

# Separate optional visualization distribution: "ecg-records-viz"
ecgfeat_viz/__init__.py                      # plotting public surface
ecgfeat_viz/plots.py                         # overlays and quality plots from (signal, record)
```

The existing `clinical_rules/` and `glasgow_rules/` subpackages move wholesale
under `ecgfeat_interpret`; their internal file-by-file redesign is outside the
46-module inventory supplied for this pass.

## Full mapping of the 46 existing modules

"Core internal" means the code ships in the measurement distribution but is not a
supported consumer import. "Moves out" means the target ownership is another
distribution; a temporary compatibility forwarding import may remain under
`ecgfeat` while existing callers migrate.

| Existing module | Lines | Current responsibility / evidence | Target destination |
|---|---:|---|---|
| `__init__.py` | 53 | Re-exports extractor, configs, errors, interpretation, export, I/O, plots, and model dataclasses | Public facade; shrink to record/extraction/I/O/config plus deprecated forwarding exports |
| `_kalman.py` | 91 | Optional scalar recursion/JIT via `_kalman.py:run_kalman` | Core internal → `_engine/foundation/kalman.py` |
| `acquisition_qc.py` | 341 | Channel-delay correction and acquisition-chain checks via `acquisition_qc.py:apply_channel_delay_compensation`, `:assess_acquisition_chain` | Core internal → `_engine/quality/acquisition.py` |
| `adaptive_qrs.py` | 118 | Per-channel candidates and adaptive QRS via `adaptive_qrs.py:detect_adaptive_qrs` | Core internal → `_engine/detection/adaptive_qrs.py` |
| `api.py` | 3,155 | Orchestration in `api.py:ECGFeatureExtractor.extract` plus 49 private helpers | Module does not survive; split across pipeline stages/policies, with `compat/api_v0.py` preserving old object-return behavior |
| `atrial.py` | 2,430 | QRST-subtracted residual, atrial events, organized-P ratio, PR dispersion via `atrial.py:extract_atrial_events` et al. | Core internal → `_engine/atrial/core.py` |
| `atrial_validation.py` | 114 | Waveform verification via `atrial_validation.py:validate_atrial_events` | Core internal → `_engine/atrial/validation.py` |
| `boundary_refinement.py` | 126 | Opt-in boundary candidates via `boundary_refinement.py:correct_p_boundaries` and related functions | Core internal → `_engine/delineation/refinement.py`; activation remains explicit in resolved config |
| `calibration.py` | 85 | Empirical QT error-risk model via `calibration.py:QTErrorCalibration` | Core internal research utility → `_engine/measurement/calibration.py`; never represented as external clinical probability |
| `classical_candidates.py` | 153 | Deterministic boundary candidates/sequence selection via `classical_candidates.py:select_boundary_sequence` | Core internal → `_engine/delineation/candidates.py` |
| `delineate.py` | 8,219 | Main beat delineation and rescue machinery via `delineate.py:delineate_beats`, `:apply_systematic_qrs_tail_settling_rescue` | Core internal → `_engine/delineation/core.py`; later internal splits are desirable but do not change the record contract |
| `dispersion.py` | 50 | Independent-lead QT dispersion via `dispersion.py:summarize_qt_dispersion` | Core internal → `_engine/measurement/dispersion.py` |
| `export.py` | 2,647 | Legacy serialization plus fingerprints and rule payload builders via `export.py:to_dict`, `:clinical_fingerprint`, `:build_statement_engine_payload` | Split: provenance → `record/provenance.py`; new bounded serializer → `record/serialize.py`; old export → `compat/export_v0.py`; rule payload builders move to interpretation adapter |
| `family_representative.py` | 138 | Morphology families/medoids via `family_representative.py:assign_morphology_families` | Core internal → `_engine/beats/families.py` |
| `features.py` | 5,258 | Representative, group and global numerical feature construction via `features.py:compute_global_features` | Core internal → `_engine/measurement/features.py`; only selected record fields become public |
| `glasgow.py` | 109 | Statement guards/catalog/payload via `glasgow.py:build_statement_catalog` | Moves out of core → `ecgfeat_interpret/glasgow.py` |
| `glasgow_measurements.py` | 382 | Measurement profile via `glasgow_measurements.py:measure_glasgow_profile` | Core internal → `_engine/measurement/profiles/glasgow.py`; retained as measurement support, not statement logic |
| `grouping.py` | 213 | Beat clustering and annotations via `grouping.py:cluster_beats`, `:build_beat_annotations` | Core internal → `_engine/beats/grouping.py` |
| `interpret.py` | 3,181 | Clinical interpretation entry point via `interpret.py:interpret` | Moves out of core → `ecgfeat_interpret/engine.py`; output is separately versioned Interpretation JSON |
| `io.py` | 107 | WFDB-header parsing and MAT loading via `io.py:parse_wfdb_header`, `:load_wfdb_mat` | Stays public convenience adapter in core; output is raw signal/metadata, not engine state |
| `measurement_paths.py` | 133 | QT path classification via `measurement_paths.py:build_qt_path_decision` | Core internal → `_engine/measurement/paths.py`; `QTPolicy` consumes its decision |
| `mi.py` | 257 | MI evidence/statement candidates via `mi.py:build_mi_statement_candidates` | Moves out of core → `ecgfeat_interpret/mi.py` |
| `models.py` | 834 | Internal measurement dataclasses plus `ECGInterpretation` and `ECGFeatures` | Split: measurement carriers → `_engine/foundation/models.py`; interpretation model moves out; legacy names remain in `compat/models_v0.py`; new record model is independent |
| `numeric.py` | 13 | Generic trapezoid helper via `numeric.py:trapezoid` | Core internal → `_engine/foundation/numeric.py` |
| `p_morphology.py` | 176 | P-component measurement via `p_morphology.py:measure_p_components` | Core internal → `_engine/atrial/p_morphology.py` |
| `p_wave_engine.py` | 2,443 | P-wave assessment/finalization via `p_wave_engine.py:build_p_wave_assessments`, `:finalize_p_wave_states` | Core internal → `_engine/atrial/p_wave.py`; `PWaveConfig` moves to public `config.py` |
| `pediatric_rules.py` | 388 | Age bins and pediatric diagnostic evidence/candidates via `pediatric_rules.py:build_pediatric_repolarization_candidates` | Moves out of core → `ecgfeat_interpret/pediatric.py` |
| `preprocess.py` | 140 | Filtering, resampling, baseline and analysis/detection signals | Core internal → `_engine/preprocess.py`; pipeline input stage owns path selection |
| `qrs.py` | 354 | Multilead QRS detection via `qrs.py:detect_qrs_multilead_with_meta` | Core internal → `_engine/detection/qrs.py` |
| `quality.py` | 1,669 | Signal quality, pacing detection/removal, lead-reversal checks via `quality.py:compute_quality`, `:detect_pacing_spikes` | Core internal → `_engine/quality/signal.py`; diagnostic-only gate semantics are expressed as record suitability/applicability or move to interpreter |
| `r_localization.py` | 156 | Hybrid R localization via `r_localization.py:apply_hybrid_r_localization` | Core internal → `_engine/delineation/r_localization.py` |
| `refinement.py` | 41 | Explicit opt-in refinement flags via `refinement.py:RefinementConfig.experimental` | Public configuration → `config.py`; resolved flags are copied into record provenance |
| `repolarization.py` | 625 | T-wave candidates and morphology via `repolarization.py:detect_t_wave` | Core internal → `_engine/delineation/repolarization.py` |
| `representative.py` | 267 | Representative-beat construction via `representative.py:build_representative_beats_with_meta` | Core internal → `_engine/beats/representative.py` |
| `rhythm_rules.py` | 768 | Mixed measurement support and rule evidence: pacing context/failures, beat selection, AV-block/preexcitation evidence | Module does not survive: measurement/pacing/availability helpers move into core pipeline policies/stages; diagnostic statement evidence moves to `ecgfeat_interpret/rhythm.py` |
| `rhythm_statements.py` | 206 | Rhythm statement candidates via `rhythm_statements.py:build_rhythm_statement_candidates` | Moves out of core → `ecgfeat_interpret/rhythm_statements.py` |
| `st_baseline.py` | 171 | Opt-in calibrated/adaptive ST signal via `st_baseline.py:adaptive_st_signal` | Core internal → `_engine/measurement/st_baseline.py`; chosen `st_amplitude_source` recorded in provenance |
| `st_localization.py` | 797 | ST localization/measurement and morphology class via `st_localization.py:apply_hybrid_st_measurement`, `:classify_st_pattern` | Core internal → `_engine/measurement/st_localization.py`; four-class morphology remains explicitly unvalidated and ineligible as ischemia evidence |
| `statement_engine.py` | 420 | Candidate resolution/final statements via `statement_engine.py:resolve_statement_candidates`, `:finalize_statements` | Moves out of core → `ecgfeat_interpret/statement_engine.py` |
| `t_wave_refinement.py` | 1,521 | T-boundary fusion/refinement via `t_wave_refinement.py:refine_t_wave_boundaries` | Core internal → `_engine/delineation/t_refinement.py`; opt-in paths remain configuration-controlled |
| `twelve_sl.py` | 510 | Signal-derived wave measurements/profile via `twelve_sl.py:apply_twelve_sl_measurement_profile` | Core internal → `_engine/measurement/profiles/twelve_sl.py`; kept with measurement because current extraction invokes it before record finalization |
| `u_wave.py` | 262 | U-wave localization/measurement via `u_wave.py:measure_u_wave` | Core internal → `_engine/measurement/u_wave.py` |
| `validation.py` | 360 | Input-contract validation and structured errors via `validation.py:validate_ecg_input`, `:ECGInputError` | Pipeline input boundary; public error stays exported, implementation moves to `pipeline/stages/input.py` |
| `vector_axis.py` | 159 | T-axis measurement via `vector_axis.py:compute_t_axis_from_cluster` | Core internal → `_engine/measurement/vector_axis.py`; marked not-applicable in limited mode when formal axis reporting is disabled |
| `visualize.py` | 284 | Beat/representative/quality plots via `visualize.py:plot_beat`, `:plot_quality_summary` | Moves out of core → `ecg-records-viz`; plotting reads only `(signal, record)` |
| `wave_localization.py` | 599 | Hybrid P/S/T localization via `wave_localization.py:apply_hybrid_wave_localization` | Core internal → `_engine/delineation/wave_localization.py` |

All 46 modules have a target. The modules marked "does not survive" are exactly
where present file boundaries mix orchestration, policy, compatibility, and
measurement concerns.

## Rule-engine decision: separate distribution

**Decision: publish diagnostic interpretation as a separate distribution,
`ecg-records-interpretation`, rather than as core code or a core optional extra.**

The reason is architectural, not dependency weight. Interpretation has a different
versioning and validation cadence from measurement, and `00_overview.md` already
settles that its output is a separately versioned document. Keeping statement
rules in the measurement distribution would make a change to MI, pediatric, or
rhythm wording look like a change to the measurement contract. An optional extra
would avoid installing some dependencies but would not create an ownership or
import boundary.

The split is based on responsibility rather than today's filenames:

- `interpret.py:interpret`, `statement_engine.py:finalize_statements`,
  `rhythm_statements.py:build_rhythm_statement_candidates`,
  `mi.py:build_mi_statement_candidates`, the diagnostic parts of
  `rhythm_rules.py`, `pediatric_rules.py`, `glasgow.py`,
  `clinical_rules/`, and `glasgow_rules/` move to the interpretation
  distribution.
- Measurement support currently mixed into that cluster remains in core.
  `twelve_sl.py:apply_twelve_sl_measurement_profile` is called during extraction,
  `glasgow_measurements.py:measure_glasgow_profile` computes measurements, and
  pacing/measurement-selection helpers in `rhythm_rules.py` affect what the
  measurement engine reports. Those pieces must be extracted into L2/L3 before
  the old rule modules move.
- The interpretation distribution accepts a schema-versioned ECG Record and,
  only for algorithms that genuinely require waveform samples, the associated
  raw signal. It must not accept `ECGFeatures`, `GlobalFeatures`, beat-window
  objects, or private detector state as its normal API.

The consequence is precise: **the published measurement contract ends at ECG
Record JSON.** Interpretation reads that contract and emits Interpretation JSON.
No diagnostic-rule change can require a new private field from the extractor.
If a rule needs new measured evidence, that evidence must first become an explicit
record field with units, availability semantics, provenance, validation status,
and size-budget accounting.

Backward compatibility does not require keeping the architectural coupling.
During migration, top-level `ecgfeat.interpret` can be a deprecated forwarding
adapter that converts the legacy extraction result to the current record shape
and invokes the interpretation package when installed. Existing behavior can be
held stable for current callers while the new `extract_record` path stays free
of statement-engine imports.

## Decomposing `api.py`

`api.py:ECGFeatureExtractor.extract` currently calls at least 115 distinct
callees and owns 49 private top-level helpers. It should become a thin compatibility
wrapper around an explicit staged pipeline.

### Pipeline stages and seams

| Stage | Input → output | Existing evidence | Boundary |
|---|---|---|---|
| 1. Input contract | samples, fs, lead names, config → normalized input context | `validation.py:validate_ecg_input`, `preprocess.py:resample_ecg`, `preprocess.py:analysis_signal` | resolves standard vs limited mode; records original sampling/lead contract |
| 2. Quality/acquisition | normalized context → quality context | `quality.py:compute_quality`, `quality.py:summarize_record_quality`, `acquisition_qc.py:assess_acquisition_chain` | quality observations are data; policy consequences occur later |
| 3. Ventricular detection | analysis signal + quality → QRS detections | `qrs.py:detect_qrs_multilead_with_meta` | pacing rescue is a `PacingPolicy` decision with explicit evidence/reason |
| 4. Beat selection | detections → annotations, groups, representatives | `grouping.py:build_beat_annotations`, `representative.py:build_representative_beats_with_meta` | selected measurement group is explicit stage output |
| 5. Delineation | representatives/groups → fiducials | `delineate.py:delineate_beats` | optional refinements are named algorithm paths, never implicit |
| 6. Atrial analysis | fiducials + signal → P/atrial evidence | `p_wave_engine.py:build_p_wave_assessments`, `atrial.py:extract_atrial_events`, `atrial_validation.py:validate_atrial_events` | atrial evidence remains measurement, not rhythm diagnosis |
| 7. Measurement | delineated beats → lead/group/global measurements | `features.py:build_representative_lead_features`, `:compute_group_features`, `:compute_global_features` | no JSON formatting and no diagnostic statements |
| 8. Reporting policy | measurements + quality/config/input mode → reportability/applicability decisions | `measurement_paths.py:build_qt_path_decision`, current pacing/QT helpers in `api.py` | pure decision objects carry chosen value/source/reasons |
| 9. Record finalization | measured state + decisions → ECG Record | new record builder; provenance begins from `export.py:clinical_fingerprint` | only this stage constructs public record fields |
| 10. Compatibility | ECG Record + internal legacy state → old `ECGFeatures`/`to_dict` shapes | `models.py:ECGFeatures`, `export.py:to_dict` | one-way adapter; new pipeline never consumes compatibility objects |

Each stage should receive an immutable context/value object and return a new stage
result. A stage may call multiple L2 algorithms, but it must not reach forward
into a later stage. This makes ablation and regression testing possible without
turning every low-level function into public API.

### Policy objects

Three policy objects should absorb the decisions currently embedded in
orchestration:

- **`PacingPolicy`**: QRS-width override selection, capture alignment, attenuated
  pacing rescue, cleaned-signal rescue, paced-beat state, segmentation and
  measurement effects. Its output is a decision object containing selected value,
  evidence ids, reason code, and algorithm path.
- **`QTPolicy`**: measurement path, reliable-lead selection, reject gate,
  paced/native rescue, final reportability and reason codes. It consumes measured
  QT candidates; it does not itself delineate waves.
- **`ApplicabilityPolicy`**: distinguishes `measured`,
  `unavailable(reason=...)`, and `not_applicable(reason=...)`. In
  `input_mode="limited"`, diagnosis, axis, and formal QT reporting become
  `not_applicable`; a failed standard-mode QT measurement is
  `unavailable`. These states must never collapse to JSON null.

`LeadIntegrityPolicy` is a smaller fourth policy for reversal/channel-delay
consequences so that QC observations remain separate from downstream overrides.

### Destination of all 49 private `api.py` helpers

| # | Existing helper | Destination | Kind |
|---:|---|---|---|
| 1 | `_backfill_t_axis_after_measurement_profile` | `pipeline/stages/measurement.py` | stage finalization |
| 2 | `_clear_precordial_reversal_flags` | `pipeline/policies/lead_integrity.py` | policy |
| 3 | `_apply_measurement_availability_to_representatives` | `pipeline/policies/applicability.py` | policy |
| 4 | `_atrial_measurements_invalid_for_availability` | `pipeline/policies/applicability.py` | policy predicate |
| 5 | `_resolve_mains_frequency` | `pipeline/stages/input.py` | incidental input normalization |
| 6 | `_av_block_evidence` | `pipeline/stages/atrial.py` | measurement evidence; diagnostic use stays downstream |
| 7 | `_pacing_like_av_evidence` | `pipeline/policies/pacing.py` | policy evidence |
| 8 | `_dominant_group_qrs_ms` | `pipeline/policies/pacing.py` | policy helper |
| 9 | `_wide_qrs_pacing_like_context` | `pipeline/policies/pacing.py` | policy predicate |
| 10 | `_overwide_qrs_pacing_like_context` | `pipeline/policies/pacing.py` | policy predicate |
| 11 | `_overwide_pacing_qrs_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 12 | `_raw_pacing_qrs_underestimate_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 13 | `_near_wide_pacing_qrs_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 14 | `_dominant_paced_wide_qrs_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 15 | `_intermittent_paced_wide_qrs_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 16 | `_qrs_wide_values` | `pipeline/policies/pacing.py` | policy helper |
| 17 | `_intermittent_pacing_wide_measurement_context` | `pipeline/policies/pacing.py` | policy evidence |
| 18 | `_borderline_paced_qrs_wide_offset_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 19 | `_secondary_paced_wide_group_qrs_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 20 | `_near_wide_paced_qrs_offset_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 21 | `_overwide_paced_qrs_raw_consensus_override_ms` | `pipeline/policies/pacing.py` | policy decision |
| 22 | `_copy_qt_measurements` | `pipeline/policies/qt.py` | policy helper |
| 23 | `_apply_qt_reject_gate` | `pipeline/policies/qt.py` | policy decision |
| 24 | `_rescue_intermittent_paced_qt_from_native` | `pipeline/policies/qt.py` | QT/pacing policy decision |
| 25 | `_probable_limb_lead_reversal` | `pipeline/policies/lead_integrity.py` | policy predicate |
| 26 | `_select_measurement_group` | `pipeline/stages/beats.py` | pipeline stage |
| 27 | `_beat_qrs_profiles` | `pipeline/stages/beats.py` | stage helper |
| 28 | `_group_qrs_profile` | `pipeline/stages/beats.py` | stage helper |
| 29 | `_refine_measurement_group_after_delineation` | `pipeline/stages/beats.py` | stage decision after delineation |
| 30 | `_paced_beat_fraction` | `pipeline/policies/pacing.py` | policy helper |
| 31 | `_rescue_qt_after_qrs_tail_settling` | `pipeline/policies/qt.py` | policy decision |
| 32 | `_selected_group_paced_majority` | `pipeline/policies/pacing.py` | policy predicate |
| 33 | `_pacing_capture_alignment_confirmed` | `pipeline/policies/pacing.py` | policy evidence |
| 34 | `_confirmed_attenuated_pacing_rescue` | `pipeline/policies/pacing.py` | policy decision |
| 35 | `_pacing_capture_beat_ids` | `pipeline/policies/pacing.py` | policy evidence helper |
| 36 | `_robust_rr_profile` | `pipeline/stages/measurement.py` | rhythm measurement helper |
| 37 | `_pacing_qrs_rescue_evidence` | `pipeline/policies/pacing.py` | policy evidence |
| 38 | `_pacing_qrs_cleaned_single_lead_rescue` | `pipeline/policies/pacing.py` | policy decision |
| 39 | `_try_attenuated_pacing_rescue` | `pipeline/policies/pacing.py` | policy decision |
| 40 | `_pacing_measurement_effect` | `pipeline/policies/pacing.py` | policy decision |
| 41 | `_pacing_segmentation_effect` | `pipeline/policies/pacing.py` | policy decision |
| 42 | `_measurement_pacing_state` | `pipeline/policies/pacing.py` | policy result |
| 43 | `_build_pacing_evidence_by_beat` | `pipeline/policies/pacing.py` | policy evidence builder |
| 44 | `_build_rhythm_beat_rows` | `pipeline/stages/measurement.py` | rhythm measurement stage |
| 45 | `_finite_float` | `_engine/foundation/numeric.py` | genuinely incidental numeric helper |
| 46 | `_median_or_none` | `_engine/foundation/numeric.py` | genuinely incidental numeric helper |
| 47 | `_delta_evidence` | `pipeline/stages/measurement.py` | rhythm measurement evidence |
| 48 | `_pr_series_ms` | `pipeline/stages/measurement.py` | rhythm measurement helper |
| 49 | `_atrial_events_per_rr` | `pipeline/stages/measurement.py` | rhythm measurement helper |

The important change is that pacing/QT rescues stop being anonymous mutations of
a long function. They become named decisions with reason codes and provenance.
That makes it possible for the record to state, for example, that QRS duration
came from a paced-wide override or QT was rescued from native beats, without
serializing the whole intermediate object graph.

The two tiny numeric helpers are genuinely incidental. Mains-frequency resolution
is input normalization. The rest are either stage logic or policy and should not
remain top-level private functions beside the extractor.

## End-to-end data flow

```text
caller
  |
  | raw samples + fs + named leads + config
  v
validate input contract
  |-- standard 12-lead
  '-- limited 1..8 named channels
  |
  v
preprocess / acquisition QC / signal quality
  |
  v
QRS detection -----> PacingPolicy evidence/rescue
  |
  v
beat grouping + representatives
  |
  v
delineation + explicitly enabled refinements
  |
  +----------> atrial/P analysis
  |
  v
lead/group/global measurements
  |
  +----------> QTPolicy
  +----------> ApplicabilityPolicy
  +----------> LeadIntegrityPolicy
  |
  v
record finalizer
  |  schema_version
  |  library_version
  |  resolved config
  |  input fingerprint
  |  algorithm path / policy reasons
  |  per-field units, availability, validation status
  v
ECG Record JSON  <= 24,000 bytes for standard reference profile
  |
  +--> ecgfeat.record.query ----------> selected feature quantities
  +--> ecg-records-viz + raw signal --> fiducial/quality visualization
  +--> report renderer --------------> measurement report
  +--> ecg-records-interpretation ---> separately versioned Interpretation JSON
  '--> ecgagent ---------------------> pointer-addressed evidence/agent workflow
```

The raw signal is never copied into the ECG Record. Sample coordinates in the
record address that separately retained signal. A consumer that needs only
measurements can operate on JSON alone; a waveform consumer receives the same
record plus the signal identified by the record's input fingerprint.

### Availability and limited-input flow

Availability is a state machine, not a nullable scalar:

```text
field applicable?
  no  -> not_applicable(reason="limited_input_contract" | ...)
  yes -> measurement attempted?
           no  -> unavailable(reason="quality_gate" | "no_valid_lead" | ...)
           yes -> measured(value, units, source/path, validation)
```

For `input_mode="limited"`, diagnosis, axis, and formal QT reporting are
`not_applicable` by contract while raw measurements that can be computed remain
eligible for `measured` or `unavailable`. This distinction should be made by
`ApplicabilityPolicy` before serialization; `record.serialize` merely preserves
it.

### Reproducibility flow

The finalizer records at least:

- library/distribution version;
- ECG Record schema version;
- fully resolved extraction configuration, including defaults;
- explicit experimental flags and `st_amplitude_source`;
- input fingerprint covering signal identity, sampling rate, lead names/order,
  and input mode;
- selected algorithm path and policy reason codes for fields affected by rescue,
  override, or fallback.

`export.py:clinical_fingerprint` is the existing starting point, but the new
record provenance must cover configuration and path choices rather than only
clinical content identity.

### Validation flow

Validation status belongs to each published field or homogeneous field group. It
is not inferred from confidence. Benchmark-backed fields should name the relevant
evidence family (LUDB, QTDB, EDB, BUT, NSTDB, GUDB, PTB-XL as applicable);
unvalidated outputs remain explicitly unvalidated.

In particular, `st_localization.py:classify_st_pattern` may continue to expose a
four-class morphology observation for research, but the record must mark it
unvalidated and the interpretation package must reject it as ischemia evidence.
A successful delineation benchmark must not be inherited by unrelated derived
fields.

## Compatibility architecture

A clean break is unavailable because `ecgagent/`, `batch_extract_ecgfeat.py`,
128 repository tests, and analysis scripts consume current structures. Migration
should therefore be additive:

```text
new caller --------------------> pipeline ----------------> ECG Record
                                      |
legacy ECGFeatureExtractor ----------+
                                      |
                                      +--> compat/api_v0 --> ECGFeatures
                                                       |
                                                       '--> compat/export_v0 --> legacy to_dict
```

The new record path is primary. Compatibility conversion is downstream of the
pipeline and cannot feed state back into it. This prevents legacy shape
requirements from dictating the compact record schema.

Top-level imports may continue forwarding the existing names for a deprecation
window, including plotting and `interpret`, but those forwarding imports should
be lazy so the measurement distribution does not acquire plotting or rule-engine
runtime dependencies. Existing defaults remain stable: experimental refinements
stay opt-in, and the exact resolved path is recorded.

## What stays public

The target public surface for this architecture is intentionally narrow:

- extraction/configuration entry points;
- `ECGInputError`;
- `ecgfeat.record` model, load/dump/validate/query operations;
- existing I/O convenience readers;
- deprecated compatibility exports while migration is active.

The 14 current model dataclasses are not automatically part of the durable API
simply because `__init__.py` exports them today. Their compatibility names may
remain, but downstream new code should target record fields and stable JSON
addresses.

## Open questions

1. The digest establishes responsibilities and call names but not the exact
   import graph. Before implementation, run an automated import-graph check to
   identify cycles that affect the proposed physical move order; this does not
   change the target boundaries.
2. The supplied validation history names benchmark datasets but does not map
   every individual published field to a dataset/result. The record schema must
   not assign benchmark tiers until that mapping is audited.
3. The exact set of measurement support currently required by
   `clinical_rules/` and `glasgow_rules/` was not re-read in this pass. Any
   required quantity absent from the ECG Record must be either promoted as an
   explicit measured field or recomputed by the interpretation package from the
   raw signal; importing private engine state is not an acceptable shortcut.
4. The compatibility lifetime for `ECGFeatures`, `to_dict()`, plotting
   forwards, and `ecgfeat.interpret` needs a release policy. Architecture only
   requires that the migration be additive and that new consumers do not depend
   on those legacy shapes.
