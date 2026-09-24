# Architecture

This page is for contributors. It explains how the package is organized, and
which rules keep the parts separate.

## Repository layout

```text
feature_extraction/            the ecg-records distribution (pyproject.toml, README.package.md, CHANGELOG.md)
  ecgfeat/                     the import package
    __init__.py                lazy public facade (__all__ = PUBLIC_API + LEGACY_API)
    config.py, errors.py       public configuration and error types
    refinement.py              RefinementConfig (experimental switches)
    record/                    L4: the ECG Record model, validation, query, serialization, sidecars
    pipeline/                  L3: extractor, stages, policies, record builder
    _engine/                   L1/L2: private numerical engine
    viz/                       plotting of (signal, record); Matplotlib only
    compat/                    the explicit legacy contract (removed no earlier than 0.3.0)
    io.py                      WFDB MAT/header reading
    cli.py                     the ecg-record command
    schemas/ecg-record/1.0/    JSON Schema and validation-evidence registry (package data)
    <old module names>.py      deprecated aliases of moved modules
schemas/ecg-record/1.0/        repository copy of the schema, registry and the legacy crosswalk
tests/                         unit, contract, property and fixture tests
benchmarks/golden/             golden regression corpus: manifests, frozen baselines, harness
tools/                         release, documentation, registry and performance checks
docs/site/                     this documentation (MkDocs)
```

## Layers

```text
            ecgfeat (facade)           cli            viz (Matplotlib)
                  │                     │               │
                  ▼                     ▼               ▼
   L4  record ◄──────────────────────── pipeline ──► record builder
        ▲  (model, contract, query,      │  L3: extractor, stages, policies
        │   serialize, sidecar)          ▼
        │                              _engine  L2: preprocess, quality, detection,
        │                                │         beats, delineation, atrial, measurement
        │                                ▼
        │                              _engine.foundation  L1: numeric helpers, internal models
        │
   compat ──► legacy API over the same pipeline (lazy access to ecginterpret)
```

- **L1 `_engine.foundation`**: numeric helpers, internal data models and the
  optional Numba kernel. Imports no other engine package.
- **L2 `_engine`**: the measurement algorithms. Never imports the record,
  pipeline, compatibility, plotting or interpretation code.
- **L3 `pipeline`**: orchestration. `extractor.py` holds the public `ecg_*`
  functions, `stages/` the eight stages, `policies/` the four policy objects,
  and `record_builder.py` projects measurements into a record.
- **L4 `record`**: everything about the published format, with no dependency
  on the engine. Reading and querying a record never loads NumPy-heavy code.
- **`viz`** reads records (and, for the legacy helpers, legacy models) and
  draws them. It never measures.
- **`compat`** keeps the legacy API working on top of the same pipeline. It
  is the only module that may reach the separate `ecginterpret` distribution,
  and it does so lazily.

## Enforced rules

`feature_extraction/.importlinter` states seven contracts, checked in CI with
`lint-imports`:

1. `ecgfeat.record` imports no engine, pipeline, legacy, plotting or
   interpretation code.
2. `ecgfeat._engine` never imports record, pipeline, compatibility, plotting
   or interpretation code.
3. `ecgfeat._engine.foundation` imports no measurement-engine siblings.
4. The pipeline never imports interpretation, plotting or the legacy exporter.
5. Only `ecgfeat.compat` reaches the interpretation distribution.
6. `ecgfeat.viz` never reaches the pipeline or interpretation.
7. `ecgfeat.viz` never imports the private engine directly.

Tests also check that `import ecgfeat` loads neither Matplotlib nor the
engine, and that `import ecgfeat.viz` does not load Matplotlib.

## The pipeline

`ecg_record` = `ecg_prepare` → `ecg_measure` → `ecg_emit`.

`ecg_measure` runs the stage sequence in `pipeline/extractor.py`
(`LEGACY_STAGE_SEQUENCE`) over an immutable `PipelineContext`. Each stage
returns a new context, never mutating the previous one:

| Order | Stage (`pipeline/stages/…`) |
|---|---|
| 1 | `input.run` |
| 2 | `quality.run` |
| 3 | `ventricular.run` |
| 4 | `beats.run` |
| 5 | `delineation.run` |
| 6 | `beats.refine_measurement_group` |
| 7 | `delineation.settle_qrs_tail_and_refine_t` |
| 8 | `atrial.run` |
| 9 | `measurement.run` |
| 10 | `finalize.run` |

Contextual decisions live in four **policy objects** (`pipeline/policies/`):
`pacing`, `qt` (QT rescue), `applicability` and `lead_integrity`. Their
decisions accumulate as events in the context and are summarized in the
record's provenance (`policy_execution`).

`ecg_emit` calls `record_builder.build_record_document`. It maps engine
quantities to record fields through the reviewed
[legacy crosswalk](../reference/legacy-crosswalk.md), applies the
[invariants](../guide/record-format.md#published-fields) (withholding violating
cells as `unmeasurable`), encodes absence states, attaches validation status
from the registry, and validates the result.

## Adding or changing a published field

A new record field is a schema change. It needs all of the following:

1. a name, unit, axes and absence semantics, documented in
   [The ECG Record](../guide/record-format.md);
2. an entry in the validation-evidence registry
   (`validation-evidence.json`, both copies) with profiles, publication tier,
   validation status and ceiling, legacy source and known limitations;
3. a crosswalk entry when it comes from an existing engine quantity;
4. size accounting: the `summary` profile must stay within 24,000 bytes;
5. a schema **minor** version (additive) and at least a package minor release;
6. regenerated reference pages (`python tools/build_docs.py`).

Changing a measured value (an algorithm change) is a behavioural change. It
must show up as golden-corpus differences, be reviewed, and ship with a new
frozen baseline; see [Golden corpus](golden-corpus.md).
