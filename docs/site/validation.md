# Validation and benchmarks

## What is (and is not) claimed

Every published field of schema 1.0.0 has validation status **`unvalidated`**.
This is deliberate and conservative. The project has benchmark history for
several endpoints: LUDB and QTDB delineation, EDB ST amplitude, and detection on
PTB-XL, BUT-PDB, NSTDB and GUDB. But a field may claim a better status only by
citing an approved **evidence manifest**, and none has been approved yet.

Each field's status is in the record (`/<field>/validation/status`) and in
every query result (`result.validation.tier`). It comes from the
validation-evidence registry shipped with the package
(`ecgfeat/schemas/ecg-record/1.0/validation-evidence.json`). The strict reader
rejects any record that claims more than the registry allows
(`validation_above_registry`).

| Record status | Query tier (`result.validation.tier`) | Meaning |
|---|---|---|
| `unvalidated` | `unvalidated` | no approved evidence; every schema 1.0.0 field |
| `indirectly_validated` | `partially_validated` | supported by evidence on a related endpoint (none yet) |
| `benchmark_validated` | `validated` | an approved manifest shows the field meets a stated threshold on a named dataset (none yet) |
| `known_problem` | `unvalidated` | known to be unreliable; the debug-only four-class ST morphology is capped here |

The registry also records a **validation ceiling** per field: the best status
the field could ever reach with evidence. See the
[schema reference](reference/ecg-record-schema-1.0.md) and the
[validation policy](development/validation-policy.md).

## Regression gates

The repository guards against unintended changes in what the library
measures. These gates bound regressions from an accepted baseline. They are
**not** claims of clinical accuracy.

- **Golden corpus** (`snapshot_regression.py`, [details](development/golden-corpus.md)):
  156 sentinel and 1,063 full-tier records from seven public datasets (LUDB,
  QTDB, EDB, PTB-XL, BUT-PDB, NSTDB, GUDB). The frozen `ecg-records-0.1.0`
  baseline holds the SHA-256 of every record profile that release produces.
  Any change to a published value shows up as a difference. The older
  `phase0-52a339c` baseline preserves the pre-migration outputs: the legacy
  export reproduces them byte for byte, and every record value equals its
  legacy value through the [crosswalk](reference/legacy-crosswalk.md).
- **Contract, property and fixture tests**: schema and invariant checks,
  Hypothesis property tests of the record builder, the pinned 10 s reference
  record and the 24,000-byte summary contract.
- **Performance budget** (`tools/check_performance.py`): import time, first and
  warm extraction time, and peak memory on the reference record.
- **Layer contracts** (`import-linter`): the record API, the engine, the
  pipeline and the plotting package stay separated.

## Known weaknesses

- P-wave delineation is the weakest endpoint in the benchmarks, especially at
  short RR intervals or when the P wave overlaps the preceding T wave.
- LUDB distributes amplitude-normalized signals, so amplitude accuracy cannot
  be assessed on LUDB.
- The four-class ST morphology was never validated and is anti-correlated with
  ischemia. It is not published (debug-only, capped at `known_problem`).

See also [Limitations and intended use](limitations.md).
