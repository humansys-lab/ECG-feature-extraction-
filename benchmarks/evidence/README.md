# Validation evidence manifests

A field in `schemas/ecg-record/1.0/validation-evidence.json` may only claim
`benchmark_validated` or `indirectly_validated` by citing a manifest here.
`tools/check_validation_registry.py` enforces the promotion rule.

Each manifest is `<evidence_id>.json` with: `evidence_id`, `tool` (name and
SHA-256), `dataset` (name, release, content-manifest SHA-256), `baseline_id`,
`metric` (name, direction, unit), `acceptance_threshold`, `result` (value,
`passed: true`), `validated_for` (population/task/endpoint), `schema_major`,
`code_revision`, and `approved_by` (the Validation Owner).

There are no manifests yet: every published field of schema 1.0.0 is
`unvalidated`. Benchmark results reported elsewhere are not manifests and do
not promote a field. See docs/site/development/validation-policy.md.
