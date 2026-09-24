# Validation policy

A record must never make a value look more trustworthy than the evidence
supports. This page explains how validation status is assigned and how a field
could earn a better one.

## One source of truth

Validation status comes **only** from the validation-evidence registry,
`ecgfeat/schemas/ecg-record/1.0/validation-evidence.json` (mirrored at
`schemas/ecg-record/1.0/`). No code path can override it. For every published
field, the registry records:

| Key | Meaning |
|---|---|
| `pointer` | the field, for example `/measurements/intervals/qt_interval_ms` |
| `profiles` | profiles that publish it |
| `publication_tier` | `published_measurement` for every schema 1.0 field |
| `validation_tier` | the current status (`unvalidated` for every schema 1.0 field) |
| `validation_ceiling` | the best status the field may ever claim |
| `evidence_ids` | evidence manifests supporting the status (none yet) |
| `validated_for` | the population, task and endpoint the evidence covers |
| `known_limitations` | caveats shown in the [schema reference](../reference/ecg-record-schema-1.0.md) |
| `legacy_source` | where the value came from in the legacy payload |

Debug-only quantities are capped too: the four-class ST morphology at
`known_problem` (it was never validated and is anti-correlated with ischemia),
and the ST pattern class, ST/T confusion, hybrid ST candidates (`st_hybrid_*`)
and 12SL-style quantities (`twelve_sl_*`) at `unvalidated`.

The record writer copies the registry status into each field. The strict
reader rejects any record whose fields claim more than the registry allows
(`validation_above_registry`), including records produced elsewhere.

## Status vocabulary

In order from worst to best: `known_problem` < `unvalidated` <
`indirectly_validated` < `benchmark_validated`. The query API reports the
last two as `partially_validated` and `validated`
([table](../validation.md#what-is-and-is-not-claimed)).

## Promotion rule

A field may carry `benchmark_validated` or `indirectly_validated` only if:

1. its `validation_ceiling` allows it;
2. it cites at least one evidence manifest in `benchmarks/evidence/`;
3. every cited manifest names a tool (with SHA-256), a pinned dataset version
   and content hash, a baseline, a metric with direction and unit, an
   acceptance threshold and a **passing** result;
4. the manifest's `validated_for` scope covers the field;
5. the manifest was produced under the current schema major;
6. the Validation Owner approved it (`approved_by`).

`tools/check_validation_registry.py` checks rules 1–5 mechanically in CI.
Rule 6 is a people decision, enforced through `.github/CODEOWNERS`, which
requires the Validation Owner's review for changes to the registry, evidence
manifests, golden manifests, baselines and the reference fixture.

## What does not count as evidence

- benchmark history, papers, plots or notebooks that are not an approved manifest;
- golden-corpus agreement, which proves stability, not correctness;
- agreement with the legacy implementation.

## Why everything is `unvalidated` in 1.0.0

The project has benchmark results for several endpoints (LUDB and QTDB
delineation, EDB ST amplitude, detection on five datasets). None has yet been
written up as an approved manifest with a pinned dataset hash, a
pre-declared threshold and a stated scope. Publishing `unvalidated` until then
is the honest choice: consumers can rely on the status meaning what it says.
