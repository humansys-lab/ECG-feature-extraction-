# Compatibility policy

- **Code versions** follow SemVer per distribution (`ecg-records`,
  `ecginterpret`, `ecg-records-viz` are versioned independently).
- **Schema versions** are independent of package versions. A schema minor
  (additive fields, reason codes, validation evidence) forces at least a code
  minor; a schema major forces a code major.
- `ecg-records 0.1.x` writes ECG Record schema `1.0.0` and reads every schema
  `1.*`: readers ignore fields they do not understand, and a consumer that uses
  only fields from `1.m` can read later `1.*` records without any change of
  meaning, unit or absence semantics.
- Deprecated APIs remain for at least two minor releases after the first
  warning release (0.1.0) and are removed no earlier than 0.3.0; removal is
  preceded by a one-release tombstone that raises a targeted `ImportError`.
- Released artifacts are immutable; defects are fixed forward in a new release.
