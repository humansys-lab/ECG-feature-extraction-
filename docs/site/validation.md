# Validation and benchmarks

## What is (and is not) claimed

Every published field of schema 1.0.0 is `unvalidated`. The repository has
benchmark history for several endpoints (LUDB and QTDB delineation, EDB ST
amplitude, PTB-XL, BUT-PDB, NSTDB and GUDB detection), but a field can claim a
better tier only through an approved evidence manifest
(`benchmarks/evidence/`), checked mechanically by
`tools/check_validation_registry.py`.

## Release gates

- **Golden parity** (`snapshot_regression.py`): 156 sentinel and 1,063 full-tier
  records from seven public datasets, frozen before the migration. Legacy
  output is byte-identical; every published record value equals its legacy
  leaf through the crosswalk; the interpretation document equals the legacy
  interpretation sections.
- **Evaluator baselines** (`python -m benchmarks.harness`): nine existing
  evaluation tools with pinned inputs and document-06 tolerances (0.5 pp
  aggregate, 1.0 pp subgroup, 2 % error metrics, zero new failures).
- **Contract, property and golden-fixture tests**, the 24,000-byte summary
  contract, and a performance budget (`tools/check_performance.py`).

These gates bound regressions from an accepted baseline; they are not claims
of clinical accuracy.
