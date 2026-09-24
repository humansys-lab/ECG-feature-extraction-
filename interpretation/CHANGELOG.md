# Changelog — ecginterpret

## [Unreleased]

## [0.1.0] - 2026-09-24

### Added
- Rule-based interpretation extracted from `ecgfeat` into its own distribution:
  interpreter, clinical rule engine, Glasgow-style rules, statement engine, MI
  and pediatric rules.
- Versioned Interpretation document (schema `1.0.0`, JSON Schema shipped):
  `interpret_features(features)` for the legacy measurement object and
  `interpret_record(record, signal=...)`, which re-derives measurements with the
  record's resolved configuration after checking the raw-signal SHA-256.
  Output is identical to the interpretation sections of the legacy export on
  1,063 golden records.

### Known limitations
- Accepts ECG Record schema major 1 only; record 1.0 does not carry every rule
  input, so the raw signal is required.
