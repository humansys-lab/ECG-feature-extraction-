# Changelog — ecg-records

All notable changes to the `ecg-records` distribution (import name `ecgfeat`).
The ECG Record *schema* is versioned separately (see `schema_version` in every
record); this file tracks the package.

## [Unreleased]

## [0.1.0] - 2026-09-24

First release of the redesigned library (previously the unpublished
`ecgfeat-dxl-inspired` 0.1.0 working tree).

### Added
- **ECG Record** (schema `1.0.0`): `ecg_record`, `ecg_prepare` / `ecg_measure` /
  `ecg_emit`, `load_record`, `validate_record`, `dumps_record`, `query_measurement`,
  `query_many`, `select_measurements`, `resolve_address`, `iter_leads`, `iter_beats`,
  profiles `summary` / `all` / `debug`, canonical addresses
  `ecg-record:<record_id>@<schema_version>#<RFC 6901 pointer>`, and three explicit
  absence states (plain null, `unmeasurable(reason)`, `not_applicable(reason)`,
  with a lossless per-field default state).
- Packaged JSON Schema and validation-evidence registry
  (`ecgfeat/schemas/ecg-record/1.0/`); strict reading rejects validation claims
  above the registry. Every published field is `unvalidated` in this release.
- Verified NPZ sidecar for dense matrices (`serialize_record(..., sidecar_uri=...)`,
  CLI `measure --sidecar`), deterministic bytes, SHA-256 and identity bound.
- `ecg-record` command line: `measure`, `validate`, `query`, `resolve`, `select`, `batch`.
- Extras: `viz` (installs `ecg-records-viz`), `interpret` (installs `ecginterpret`),
  `performance` (Numba), `wfdb`.

### Changed
- Extraction runs as an explicit staged pipeline (input, quality, ventricular,
  beats, delineation, atrial, measurement, finalize) with pacing, QT,
  applicability and lead-integrity policy objects. Measurements are unchanged:
  legacy output is byte-identical to the pre-migration baseline on 1,063 golden
  records from seven public datasets.
- Engine modules are private (`ecgfeat._engine`); interpretation moved to the
  `ecginterpret` distribution and plotting to `ecg-records-viz`.
- Published records never contain an impossible fiducial order or a negative
  interval: such cells are `unmeasurable(fiducial_order_violation)` /
  `unmeasurable(negative_interval)` instead of failing the whole record.

### Deprecated
- `ecgfeat.ECGFeatureExtractor` / `ecgfeat.api`, `ecgfeat.to_dict` and the
  `ecgfeat.export` entry points, top-level `interpret`, `plot_*`,
  `load_wfdb_mat`, `parse_wfdb_header`, and the old private module paths.
  They warn on use and are removed no earlier than 0.3.0; `ecgfeat.compat`
  is the explicit legacy contract meanwhile.

### Known limitations
- No published field has an approved evidence manifest yet (all `unvalidated`).
- P-wave delineation is the weakest endpoint in the repository benchmarks.
- ECG Record 1.0 publishes a compact subset of the engine's measurements;
  `ecginterpret.interpret_record` therefore needs the raw signal.
