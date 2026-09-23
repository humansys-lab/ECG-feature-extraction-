# ECG Record JSON schema

> Implementation status: this document includes target contracts, not completed
> release claims. [Document 08](08_implementation_status.md) records the implemented
> subset, reconciled decisions and outstanding gates. Document 02 is authoritative
> for the JSON layout; Python carrier sketches must not create a second wire format.

This document defines the published, versioned **ECG Record** boundary for measurement output. It deliberately excludes clinical interpretation: interpretation is a separate document with its own schema and distribution. The JSON record is designed for research and engineering use, not as a medical-device interchange format.

## A. Design goals and size budget

The reference case used for every size calculation in this section is **10 s, 12 leads, 500 Hz, 10 beats**. The current default `to_dict()` export is 2,603,976 uncompressed JSON bytes.

The 24,000-byte target is achievable for the default `summary` ECG Record, but it is **not achievable if every one of the 88 currently measurement-like beat fields is serialized for all 120 beat/lead cells as ordinary JSON numbers**. That distinction is important: the schema keeps all published measurements addressable, while the default JSON profile carries a compact core and moves the full dense matrix to an optional binary sidecar.

### A.1 Where the 2,603,976 bytes go

The current section sizes sum to 2,603,976 bytes. Three interpretation sections are outside the ECG Record by design:

- `clinical_interpretation`: 128,003 bytes
- `interpretation`: 6,420 bytes
- `statement_engine`: 1,912 bytes

Arithmetic: 128,003 + 6,420 + 1,912 = **136,335 bytes removed**, leaving 2,603,976 - 136,335 = **2,467,641 bytes**.

Four more sections are algorithm inputs or intermediate assessments rather than published record data:

- `p_wave_assessments`: 246,779 bytes
- `rhythm_inputs`: 126,374 bytes
- `representative_leads`: 126,360 bytes
- `morphology_inputs`: 94,360 bytes

Arithmetic: 246,779 + 126,374 + 126,360 + 94,360 = **593,873 bytes removed**, leaving 2,467,641 - 593,873 = **1,873,768 bytes**.

The remaining dominant item is `beat_features`: 1,699,188 bytes. It is 120 entries (10 beats x 12 leads), each reported as about 14,195 bytes and 311 keys. The family measurements supplied for one entry add to 14,195 bytes; multiplied by 120 this is 1,703,400 bytes, 4,212 bytes (0.25%) above the measured section size because the per-family figures are rounded. For whole-section accounting, **1,699,188 is authoritative**.

The per-entry family arithmetic is:

| family | measured bytes/entry | x 120 entries | schema disposition |
|---|---:|---:|---|
| repair/consensus bookkeeping | 3,162 | 379,440 | debug, except one compact resolved-path summary |
| provenance / method / reason | 1,975 | 237,000 | compact provenance, detailed form outside `summary` |
| experimental `st_hybrid_*` | 945 | 113,400 | opt-in debug only |
| localization internals | 784 | 94,080 | debug only |
| `twelve_sl_*` vendor-style block | 692 | 83,040 | debug/compatibility extension only |
| duration/interval measurements | 843 | 101,160 | published measurement |
| amplitude measurements | 648 | 77,760 | published measurement |
| fiducial indices | 639 | 76,680 | published measurement |
| area measurements | 200 | 24,000 | published measurement |
| flags/booleans | 193 | 23,160 | split by semantics; see C |
| other, including nested objects | 4,114 | 493,680 | split by contents; see C |
| **total** | **14,195** | **1,703,400** | rounded family total |

The five clearly non-measurement implementation families consume 3,162 + 1,975 + 945 + 784 + 692 = **7,558 bytes/entry**, or 7,558 x 120 = **906,960 bytes** before considering the debug content inside `other`. The five direct measurement/flag families consume 843 + 648 + 639 + 200 + 193 = **2,523 bytes/entry**, or **302,760 bytes** in the current nested-dict representation.

Columnar encoding is therefore necessary but not sufficient. There are 88 direct measurement/fiducial/flag keys. If all 88 are dense over 120 cells, there are 88 x 120 = **10,560 scalar slots**. Even the impossible best case of one one-digit number plus one delimiter per slot is roughly 10,560 x 2 = **21,120 bytes** before field names, metadata, quality, provenance, validation status, or JSON structure. A realistic compact-width model is much larger:

- 32 interval fields: 32 x 501 bytes per 10x12 matrix = **16,032 bytes**
- 19 amplitude fields: 19 x 621 bytes = **11,799 bytes**
- 23 fiducial fields: 23 x 621 bytes = **14,283 bytes**
- 6 area fields: 6 x 741 bytes = **4,446 bytes**
- 8 boolean/flag fields: 8 x 261 bytes = **2,088 bytes**

Each matrix figure counts 120 value digits at the stated width, 110 commas inside ten 12-value rows, 9 commas between rows, 20 row brackets, and 2 outer brackets. For example, an interval matrix is 120 x 3 + 110 + 9 + 20 + 2 = **501 bytes**. Value bytes alone are therefore approximately 16,032 + 11,799 + 14,283 + 4,446 + 2,088 = **48,648 bytes**. Adding one compact field wrapper budget of 85 bytes x 88 = 7,480, a 5,250-byte acquisition/provenance/quality base, 2,000 bytes of global measurements, and 1,500 bytes for sparse absence/reason metadata gives **64,878 bytes**. That is the nearest defensible planning number for an all-measurement-family JSON profile under these width assumptions, not 24,000.

The 24,000-byte requirement is therefore defined for the default `summary` profile. The full published measurement matrix is available either in a larger JSON `all` profile or, preferably, in the optional dense sidecar described in D.

### A.2 24,000-byte summary envelope

The default `summary` profile carries a curated core of 18 beat/lead matrices: 7 fiducials, 5 intervals, 5 amplitudes, and 1 area. The exact promotion list is schema-versioned. For the 10 s reference case, the byte envelope is:

| component | arithmetic | budget |
|---|---|---:|
| envelope/schema/profile/record id | fixed allowance | 450 |
| acquisition + raw-signal reference | fixed allowance | 1,200 |
| reproducibility provenance | fixed allowance | 1,700 |
| beat/lead axes | fixed allowance | 400 |
| signal quality | fixed allowance | 1,500 |
| 7 fiducial matrices | 7 x 630 values-bytes + 7 x 90 wrapper bytes | 5,040 |
| 5 interval matrices | 5 x 510 + 5 x 85 | 2,975 |
| 5 amplitude matrices | 5 x 750 + 5 x 85 | 4,175 |
| 1 area matrix | 870 + 85 | 955 |
| global measurements | fixed allowance | 2,000 |
| absence/reason dictionaries | fixed allowance | 1,000 |
| JSON delimiters and growth headroom | fixed allowance | 900 |
| **total** | sum above | **22,295** |

The matrix numbers use reproducible worst-width assumptions for this reference window and include the 10x12 row structure. A matrix contributes 110 commas inside rows + 9 commas between rows + 20 row brackets + 2 outer brackets = **141 structural bytes**. A fiducial sample is in 0..4999, so its matrix is 120 x 4 + 141 = **621 bytes**; the budget rounds that to **630 bytes**. An interval uses at most three decimal digits, so 120 x 3 + 141 = **501 bytes**, rounded to **510**. A signed amplitude budget of five characters gives 120 x 5 + 141 = **741 bytes**, rounded to **750**. An area budget of six characters gives 120 x 6 + 141 = **861 bytes**, rounded to **870**.

The resulting headroom is 24,000 - 22,295 = **1,705 bytes**, or 7.1% of the target. This is a design envelope, not a measurement of a rewritten serializer; the implementation must add a byte-size regression fixture for the exact reference record before declaring the budget met.

## B. Document split

The present 15-section export mixes acquisition facts, measurement results, implementation state, and interpretation. The replacement boundary is five logical documents/regions with only the first four in the ECG Record JSON.

1. **Record/acquisition metadata.** Contains schema/profile identifiers, record id, sample rate, signal length, ordered lead names, `input_mode`, raw-signal reference and checksum, source/input fingerprint, library version, config identity/hash, and the resolved high-level algorithm path. The oversized current `metadata` and `reference_metadata` sections are normalized into this compact form. The raw signal remains a separate published artifact referenced from the record rather than embedded as JSON samples.

2. **Signal quality.** Contains record-level and per-lead quality results plus reasons required to decide whether a measurement can be trusted. It replaces only the consumer-facing subset of the current `quality` section; diagnostic traces stay in debug output.

3. **Fiducials/delineation.** Dense, array-oriented fields indexed by the record's canonical beat and lead axes. It contains published sample indices such as P onset/offset, QRS onset/offset, R peak, J point, and T offset. Detailed candidate clusters, repairs, reselection scores, tangent alternatives, and consensus internals are not delineation outputs; they are debug data.

4. **Derived measurements.** Contains intervals, durations, amplitudes, areas, stable measurement flags, and compact global measurements. Published per-beat/per-lead data are columnar. The current `global_features`, the genuine measurement subset of `beat_features`, and stable measurement outputs from nested `p`/`qrs`/`t` objects map here. Dense full-detail matrices may be referenced through the optional sidecar.

5. **Interpretation document, separate schema and distribution.** `clinical_interpretation`, `interpretation`, `statement_engine`, and rule-engine outputs do not appear in the ECG Record. The interpretation document may cite ECG Record pointers but has its own schema version. This follows the already-set distribution split for `interpret.py`, `clinical_rules/`, `glasgow_rules/`, and `statement_engine.py`.

The current `rhythm_inputs`, `representative_leads`, `morphology_inputs`, and `p_wave_assessments` are implementation inputs/intermediates. Stable consumer-facing results extracted from them may be promoted under the field-tiering rule below; the sections themselves do not survive as public top-level sections.

## C. Field tiering

Every output is assigned by **meaning and stability**, not by where the current implementation happens to store it.

- **Published measurement.** A value is a published measurement when it describes the ECG or acquisition in a consumer-meaningful unit or stable categorical state, has a defined type/unit/axes/absence contract, and can be assigned an explicit validation status. Publication does not require that a field be benchmark-validated; it requires that an unvalidated field be clearly labelled and that its semantics are suitable for consumers. Fields known to be misleading do not qualify.
- **Provenance.** A value is provenance when it explains reproducibility: library/schema version, canonical config or config hash, input fingerprint, selected algorithm/method, high-level fallback/rescue path, or a stable reason for a published absence. Provenance describes how a result was produced; it is not itself a measurement.
- **Internal debug.** Candidate values, repair deltas, cluster memberships, intermediate scores, implementation-specific control flow, experimental bypass outputs, duplicated vendor-compatibility blocks, and unstable classifications belong here. Debug fields may be emitted by the `debug` profile but have no stable-address or compatibility guarantee across schema minor releases unless separately promoted.

Applied to every family in the measured `beat_features` table:

| family | tier verdict | concrete rule application |
|---|---|---|
| repair/consensus bookkeeping (85 keys) | **internal debug** | Names such as `t_offset_repair_consensus_index`, `p_onset_cluster_precordial_support`, and `qrs_offset_repair_delta_ms` describe selection machinery, not the ECG. One normalized final path/reason may be emitted separately as provenance. |
| provenance / method / reason (51) | **provenance** | Stable selected-method and absence-reason facts are retained once per field or record. Candidate-method lists and verbose traces remain debug. |
| `st_hybrid_*` (22) | **internal debug, experimental opt-in** | This is an experimental bypass. It is never shaped like a default measurement. If enabled, provenance records the resolved path, e.g. `"st":"st_hybrid_v1"`; raw `st_hybrid_*` internals require `debug`. |
| localization internals (22) | **internal debug** | Search windows, local extrema candidates, support scores, and similar localization machinery are implementation details. Final fiducial sample indices are published separately. |
| `twelve_sl_*` (16) | **internal debug / compatibility extension** | The vendor-style duplicate block is excluded from the core schema. A future namespaced compatibility extension may expose it, but it does not become core merely because it is currently serialized. |
| duration/interval measurements (32) | **split: published measurement plus provenance/debug exceptions** | Direct ECG durations/intervals are published with explicit units, axes, absence semantics, provenance, and validation status. Candidate spread, disagreement, rescue, and dimensionally misclassified area/slope members are reassigned by meaning; see E.5. |
| amplitude measurements (19) | **split: mostly published measurement** | Direct amplitudes are consumer data and use integer microvolts where calibration permits. Local-noise and window-drift members move to quality/provenance; see E.6. |
| fiducial indices (23) | **split: final landmarks published, method candidates debug** | Selected physiologic landmarks are consumer data and use zero-based sample indices. Method-specific alternative T-offset candidates and window bookkeeping remain debug/provenance; see E.4. |
| area measurements (6) | **published measurement** | Direct integrated measurements are public when their unit and window definition are explicit. |
| flags/booleans (8) | **split** | Stable measurement-validity or morphology-presence flags are published measurements. Flags that expose algorithm branches, rescue choices, or candidate acceptance are debug/provenance according to meaning. |
| other / nested objects (27) | **split by member** | `beat_id` and `lead` are published identity/axes. Stable measurements/fiducials inside `p`, `qrs`, and `t` are published. Method and reason members are provenance. `glasgow_measurements` is not copied wholesale; only independently specified measurements may be promoted. `st_pattern_class` and `st_t_confusion` remain debug until separately specified and validated. |

`st_morphology` receives an explicit **internal-debug** verdict. The supplied evidence says its four-class output was never validated and is anti-correlated with ischemia. Publishing that categorical result alongside validated measurements would make an unsafe semantic promise even if a `validation="unvalidated"` tag were attached. It may be inspected in `debug` or an explicitly experimental extension, but it is absent from `summary` and `measurement`.

This tiering also resolves the distinction between an experimental path and provenance about that path: the experimental values stay debug, while the fact that the path was selected is stable provenance.

## D. Encoding recommendation and byte arithmetic

**Recommendation: use canonical columnar JSON for the default record, with integer sample/measurement units, field-level sparse omission, explicit sparse absence maps, and an optional NPZ sidecar for the full dense published matrix.** Do not serialize 120 copies of 311-key dictionaries.

### D.1 Columnar shape

Per-beat/per-lead fields use a common two-dimensional axis: `[beat_index][lead_index]`. The lead order is declared once under acquisition/axes, and beat identities are declared once. A field therefore looks like one metadata wrapper plus one numeric matrix rather than 120 repeated key/value dictionaries.

For one fiducial field in the 10 s reference case, a dense 10x12 sample-index matrix is budgeted at **630 value bytes** as shown in A. The repeated-dict layout paid key and object overhead 120 times. Columnar JSON pays the field name, type, unit, validation metadata, and provenance once.

### D.2 Sample indices versus milliseconds

Store **fiducials as zero-based sample indices**, not floating-point milliseconds. At 500 Hz, 10 s contains samples 0..4999, so every index fits in four decimal digits. A value such as sample 1234 is four bytes; the equivalent JSON millisecond value may require `2468`, `2468.0`, or a rounded fractional representation depending on sampling rate. Sample indices also preserve the exact link to the raw signal and avoid irreversible rounding.

Intervals and durations are published in integer milliseconds when their source precision does not justify fractions. Their derivation must use the declared sample rate and the rounding rule below. Consumers needing exact boundaries can recompute durations from fiducial samples.

Amplitudes are stored as integer microvolts when the calibrated signal supports that resolution. Areas use integer microvolt-milliseconds when that unit is sufficient for the algorithm's actual precision. A field that truly requires fractional precision may use a JSON number with at most three decimal places.

### D.3 Float precision and rounding

Canonical numeric serialization uses:

- integers whenever the quantity has an exact integer representation in its published unit;
- otherwise at most three decimal places;
- round-half-to-even for conversion to the published unit;
- no trailing fractional zeros;
- no NaN or Infinity in JSON.

The serializer must record the original sample rate and calibration needed to reproduce conversions. A schema major version is required to change a field's unit or rounding rule.

### D.4 Sparsity and absence

The current record has only 229 non-null values out of 311 keys per beat/lead entry, so the design treats missingness as normal. The compact rule is:

1. Omit an optional field object entirely when it has no produced values and its absence is explained at a higher contract level.
2. For a published dense field with some values present, keep the fixed `[beat][lead]` matrix so JSON Pointer addresses remain stable; use `null` only for the plain-null state.
3. Encode `unmeasurable` and `not_applicable` cells in sparse maps keyed by beat id and lead, with compact reason codes defined once. Do not replace those states with bare null.
4. Debug-only sparse fields may use coordinate/value arrays because they do not carry the public pointer stability guarantee.

Cell-level null omission is not the default for published matrices: removing array entries would shift indices and destroy stable pointers. The useful sparsity win comes from dropping whole unused fields and from sparse reason maps, not from making public arrays ragged.

### D.5 Dense sidecar

The `all` profile may put its dense published per-beat/per-lead arrays in a referenced **NPZ sidecar**. NPZ is recommended over Parquet/Arrow here because the data are naturally N-dimensional numeric matrices and the Python measurement package can preserve exact integer dtypes without flattening them into rows.

Using all 88 measurement-family keys as a conservative upper-bound raw binary planning model:

- intervals: 32 x 120 x 2-byte `int16` = **7,680 bytes**
- amplitudes: 19 x 120 x 2-byte `int16` = **4,560 bytes**
- fiducials: 23 x 120 x 2-byte `int16` = **5,520 bytes**
- areas: 6 x 120 x 4-byte `int32` = **2,880 bytes**
- flags: 8 x 120 x 1-byte `uint8` = **960 bytes**

Raw values total 7,680 + 4,560 + 5,520 + 2,880 + 960 = **21,600 bytes** before NPZ container metadata, compression, and absence masks. A two-bit state mask over all 10,560 slots would add 10,560 x 2 / 8 = **2,640 bytes**, giving **24,240 raw bytes** before reason dictionaries. This sidecar size is separate from the 24,000-byte JSON target.

The JSON sidecar reference records media type, relative URI, SHA-256 digest, axes, and schema version. The sidecar is optional: `summary` is self-contained for its promoted core fields, and consumers that need every published dense measurement opt into `measurement`.

**Implemented sidecar protocol (schema 1.0.0).** Any profile may move its dense
integer `["beat","lead"]` fields to one NPZ file (`serialize_record(record,
sidecar_uri=...)`, CLI `measure --sidecar`):

- `/artifacts/dense_measurements` = `{uri, media_type: "application/x-npz",
  sha256, record_id, schema_version, axes: ["beat","lead"], shape, state_codes,
  fields}`. `uri` is a plain file name in the record's own directory (no
  scheme, no path separator); `sha256` covers the exact sidecar bytes;
  `fields` lists every sidecar-backed field pointer.
- A sidecar-backed field keeps its complete JSON entry (type, unit, axes,
  validation, provenance, absence) with `values: null` and
  `sidecar: {array_key, state_key, dtype: "int32"}`. The array is `int32`; the
  state mask is `uint8` (0 measured, 1 plain null, 2 unmeasurable, 3 not
  applicable) and must agree cell by cell with the JSON absence encoding, which
  remains authoritative for reasons.
- The NPZ carries `__identity__` (record id and schema version) so it cannot be
  attached to another record. Zip members use a fixed timestamp and sorted
  order, so identical records give identical sidecar bytes.
- Strict loading verifies digest, identity, dtype, shape and state mask before
  returning; `validate="schema"` verifies on first access. Queries read
  sidecar cells transparently, `materialize_record()` restores the inline
  record exactly, and the sidecar is never counted in the 24,000-byte budget.

## E. Field specification

### E.1 Root contract

An ECG Record has these top-level members:

| member | type | required | meaning |
|---|---|---:|---|
| `schema_version` | SemVer string | yes | ECG Record schema version, independent of package version |
| `profile` | `summary` / `all` / `debug` | yes | materialization profile |
| `record_id` | string | yes | stable id within the producing system |
| `acquisition` | object | yes | sample rate, sample count, lead order, input mode, raw-signal artifact, capabilities |
| `axes` | object | yes | canonical beat and lead axes used by all dense fields |
| `quality` | object | yes | record and per-lead quality summaries |
| `provenance` | object | yes | library/config/input/policy identities and resolved algorithm paths |
| `delineation` | object | yes | published final fiducials |
| `measurements` | object | yes | intervals, amplitudes, areas, flags, and global values |
| `artifacts` | object | yes | raw signal and optional dense-measurement sidecar descriptors |

`input_mode` is an acquisition contract, not a profile. In particular, `input_mode="limited"` may contain 1--8 named channels and disables diagnosis, axis reporting, and formal QT reporting while retaining raw measurable quantities. Those disabled outputs are `not_applicable`, not `unmeasurable`.

The `provenance` object must include:

- `library_version`
- `schema_version`
- canonical config identity and SHA-256 hash
- input fingerprint
- resolved algorithm path per subsystem
- explicit pacing policy identity/version
- explicit QT-rescue policy identity/version

The existing `export.py:clinical_fingerprint` is a partial precedent for an input/config fingerprint. The new record-level fingerprint is broader: it must identify the actual signal bytes (or a canonical signal digest), lead order, sample rate, and configuration sufficient to reproduce the measurement run.

### E.2 One field entry

A field name is the key in its containing object and is therefore not repeated inside the field payload. For example, the canonical field name in `/measurements/intervals/qrs_duration_ms` is `qrs_duration_ms`.

Every published field entry has the following logical schema:

| member | type | required | contract |
|---|---|---:|---|
| `type` | string enum | yes | `integer`, `number`, `boolean`, or `string` |
| `unit` | string or null | yes | UCUM-like stable unit token; null only for dimensionless/categorical fields |
| `axes` | array of strings | yes | `[]`, `["beat"]`, `["lead"]`, or `["beat","lead"]` |
| `values` | scalar, array, matrix, or null | yes | value container matching `axes`; null is allowed by the absence rules below |
| `nullable` | boolean | yes | whether plain null is permitted |
| `validation` | object | yes | `status` plus evidence identifiers |
| `provenance_ref` | JSON Pointer string | yes | pointer into `/provenance/fields` or a higher-level path object |
| `absence` | object | no | sparse or field-wide explicit absence state |
| `deprecated_since` | SemVer string | no | present only during schema deprecation |

The validation object is:

```json
{
  "status": "benchmark_validated",
  "evidence": ["LUDB"]
}
```

Allowed `status` values are:

- `benchmark_validated`: directly evaluated against a named benchmark/reference set for the field's defined endpoint.
- `indirectly_validated`: derived from validated primitives but not independently benchmarked for this exact endpoint.
- `unvalidated`: no suitable benchmark result is recorded.
- `known_problem`: evidence contradicts the intended consumer interpretation. Such a field cannot appear in `summary` or `all`; it is debug/experimental only.

The known four-class `st_morphology` result is `known_problem`: the supplied evidence says it was never validated and is anti-correlated with ischemia. This status is why it remains out of the published measurement tiers rather than merely carrying a warning next to a normal-looking value.

Validation evidence must be field-specific. The repository-level list of benchmark families -- LUDB, QTDB, EDB, BUT, NSTDB, GUDB, and PTB-XL -- is not permission to attach all dataset names to every field. Where the exact evidence mapping is not established, the field is `unvalidated` until that mapping is documented.

### E.3 Three absence states

There are exactly three absence states for an applicable field:

1. **Plain null.** `values` contains `null`, and there is no matching entry in `absence`. This means no value was supplied and no stronger reason is asserted. It is appropriate for legacy/imported data or an unknown missing value.
2. **Unmeasurable.** The field is applicable and the algorithm attempted or considered it, but a reliable value could not be produced. A reason code is mandatory. This preserves the existing semantic distinction represented by `export.py:_unavailable(reason=...)`.
3. **Not applicable.** The field is excluded by the input contract or profile capability. A reason code is mandatory. For example, formal axis and formal QT reporting under `input_mode="limited"` use `not_applicable`, never `unmeasurable`.

For a scalar or whole field, absence is encoded as:

```json
{
  "values": null,
  "absence": {
    "kind": "not_applicable",
    "reason": "input_mode_limited"
  }
}
```

For a dense `["beat","lead"]` field, `values` retains its fixed shape and explicit states are sparse maps:

```json
{
  "values": [[468, null]],
  "absence": {
    "unmeasurable": {
      "b0001|II": "low_snr"
    }
  }
}
```

A `null` at `b0001|II` plus the matching sparse map entry means unmeasurable. A null with no map entry is plain null. Keys in sparse state maps use the canonical `<beat_id>|<lead_name>` coordinate; `|` is forbidden in beat ids and lead names.

A dense field may also carry one **field default state** so a reason shared by
many cells is written once:

```json
{
  "values": [[468, null], [null, null]],
  "absence": {
    "default": {"kind": "unmeasurable", "reason": "measurement_unavailable"},
    "unmeasurable": {"b0002|II": "fiducial_order_violation"}
  }
}
```

A null cell listed in a sparse map takes that state; any other null cell takes
the `default` state. A field with a `default` therefore has no plain-null
cells, and a field without one keeps the plain-null meaning above. The
producer uses the whole-field form when every cell shares one reason and
otherwise makes the most frequent reason the default. This is lossless and
was added before 1.0.0 was published: on real 10-beat records with frequent
P-wave absence, per-cell repetition alone pushed `summary` to 32,851 bytes.

The producer never publishes a value that violates a published invariant. An
inverted P or QRS onset/offset pair, or an R peak before a present QRS onset
or after a present QRS offset, becomes `unmeasurable(fiducial_order_violation)`
for the affected landmarks and duration; any other negative per-beat interval
becomes `unmeasurable(negative_interval)`; a converted sample outside the
acquisition becomes `unmeasurable(sample_outside_acquisition)`. Strict reading
rejects records that break these invariants.

Reason codes are open enums within a schema major version. Consumers must preserve unknown codes and may display them as opaque strings. Adding a new reason code is therefore non-breaking.

### E.4 Fiducial and delineation group

Canonical published delineation fields live under `/delineation/fiducials/<name>`. They use:

- `type="integer"`
- `unit="sample"`
- `axes=["beat","lead"]`
- zero-based indices into the referenced raw signal
- allowed value range `0 <= value < acquisition.sample_count`
- fixed matrix shape `len(axes.beats) x len(axes.leads)`
- plain null / unmeasurable / not-applicable semantics from E.3
- a required validation object and provenance reference

The `summary` profile publishes these seven final landmarks:

| schema name | meaning | source disposition |
|---|---|---|
| `p_onset` | selected P-wave onset sample | selected final value from P delineation |
| `p_offset` | selected P-wave offset sample | selected final value from P delineation |
| `qrs_onset` | selected QRS onset sample | normalized final Q/QRS onset |
| `r_peak` | selected R peak sample | current `r_peak_index`/selected QRS result |
| `qrs_offset` | selected QRS offset sample | selected final QRS offset |
| `j_point` | selected J-point sample | current `j_index`/selected J result |
| `t_offset` | selected T-wave offset sample | selected final T delineation result |

The `all` profile may additionally publish stable final landmarks such as Q offset, QS nadir, S/S-prime peaks, T onset/peak, and U peak when their field-level validation metadata is defined.

The 23 current top-level fiducial-index keys are not copied mechanically. Their migration disposition is:

- final/physiologic locations such as `j_index`, `q_onset`, `q_offset`, `qs_nadir_index`, `r_peak_index`, `s_amplitude_index`, `s_peak_index`, `s_prime_peak_index`, and `u_peak_index` are candidates for published normalized landmarks;
- `beat_window_start_index` is axis/window provenance rather than a physiologic fiducial;
- method-specific alternatives `t_offset_dual_chord_index`, `t_offset_dual_rescued_index`, `t_offset_dual_slope_index`, `t_offset_dual_tangent_index`, `t_offset_latest_p85_index`, `t_offset_robust_center_index`, `t_pc1_offset_index`, `t_pc1_onset_index`, `t_rms_offset_index`, `t_rms_onset_index`, `t_trapezium_offset_index`, `t_wavelet_offset_index`, and `t_wavelet_onset_index` are algorithm-specific candidates and belong in debug unless one is promoted as a separately specified research measurement.

This prevents alternative localization methods from masquerading as multiple independent final landmarks while preserving the selected result and the method used as provenance.

### E.5 Interval and duration group

Canonical interval/duration fields live under `/measurements/intervals/<name>`. They use `type="integer"`, `unit="ms"`, and either `axes=["beat","lead"]` or a documented lower-dimensional axis for a true global value. Conversion from samples uses the sample rate and round-half-to-even rule in D.

The `summary` core is:

- `p_duration_ms`
- `pr_interval_ms`
- `qrs_duration_ms`
- `qt_interval_ms` -- a raw measured interval, distinct from formal QT reporting
- `tpeak_tend_ms`

The current duration/interval-classified source names map by semantics, not by the auto-generated family label. Direct quantities eligible for the published measurement tier include `jt_ms`, `p_dur_ms`, `p_initial_duration_ms`, `p_notch_interval_ms`, `p_terminal_duration_ms`, `p_tp_gap_ms`, `pr_ms`, `q_duration_ms`, `qrs_ms`, `qrs_wide_ms`, `qt_ms`, `r_duration_ms`, `r_prime_duration_ms`, `s_duration_ms`, `s_prime_duration_ms`, `t_dur_ms`, `t_global_tpte_ms`, `tpe_ms`, `u_dur_ms`, `u_isoelectric_gap_ms`, and `vat_ms`.

Fields whose names describe candidate spread, disagreement, or method-specific rescue -- `qt_latest_p85_ms`, `t_boundary_stability_ms`, `t_candidate_spread_ms`, `t_derived_spread_ms`, `t_offset_derived_disagreement_ms`, and `t_offset_dual_delta_ms` -- are provenance/quality or debug unless explicitly promoted with independent consumer semantics. Despite being caught by the same name-based family, they are not automatically public intervals.

The remaining auto-classified names are dimensionally different and move to their correct groups: `initial_qrs_area_mv_ms`, `p_terminal_area_mv_ms`, `ptf_v1_mv_ms`, and `q_area_mv_ms` are area/integral quantities; `st_slope_mv_per_ms` is a slope measurement.

### E.6 Amplitude and area groups

Canonical amplitudes live under `/measurements/amplitudes/<name>`. Published amplitudes use:

- `type="integer"`
- `unit="uV"`
- normally `axes=["beat","lead"]`
- conversion from current `*_mv` values by multiplying by 1,000 and applying round-half-to-even
- the same validation, provenance, shape, and absence requirements as delineation fields

The `summary` core is:

- `p_amplitude_uv`
- `r_amplitude_uv`
- `s_amplitude_uv`
- `st_80ms_uv`
- `t_amplitude_uv`

The 19 current amplitude-family keys are handled as follows. Direct signal quantities `initial_qrs_net_mv`, `p_amp_mv`, `p_initial_amp_mv`, `p_terminal_amp_mv`, `pr_segment_level_mv`, `q_amp_mv`, `r_amp_mv`, `r_prime_amp_mv`, `s_amp_mv`, `s_prime_amp_mv`, `st_80ms_mv`, `st_mid_mv`, `st_on_mv`, `t_amp_mv`, `u_amp_mv`, `u_amp_signed_mv`, and `u_prominence_mv` are eligible published measurements after unit normalization. `p_local_noise_rms_mv` and `p_window_drift_mv` are signal-quality/provenance quantities and move out of the amplitude measurement group.

Canonical integrated areas live under `/measurements/areas/<name>`, normally as
integer `uV_ms` on `["beat","lead"]`. The six legacy area candidates require
individual unit/window audits. For the admitted `qrs_signed_area_uv_ms`, the
current delineator integrates the baseline-subtracted analysis signal over
`[qrs_onset, qrs_offset]` inclusive with trapezoid `dx=1`. Its legacy unit is
therefore **mV × internal sample**, and conversion is
`round_half_even(qrs_signed_area * 1_000_000 / internal_sample_rate_hz)`.
Multiplying by 1,000 alone is incorrect. This dimensional check does not confer
benchmark validation or admit the other area candidates.

### E.7 Flags and morphology

Stable ECG-presence flags may be published as booleans with `unit=null`; algorithm-control flags are not. From the current eight-key flag family, `p_biphasic`, `p_notched`, `qrs_slur_flag`, and `u_wave_flag` are plausible published morphology-presence fields if their definitions are stabilized. `p_informative` and `t_sqi_pass` are quality fields. `delta_present` requires a stable signal-level definition before promotion. The opaque aggregate `flags` object is not published as-is.

`st_morphology`, `st_pattern_class`, and `st_t_confusion` remain debug/experimental until each has a stable definition and appropriate validation. They do not share a namespace with validated measurements.

## F. Stable addressing

Consumers, including agents, need to cite a single value without depending on Python object layout. The canonical address is an ECG Record identity plus an RFC 6901 JSON Pointer:

```text
ecg-record:<record_id>@<schema_version>#<json-pointer>
```

Examples:

```text
ecg-record:rec-123@1.0.0#/delineation/fiducials/qrs_onset/values/3/1
ecg-record:rec-123@1.0.0#/measurements/intervals/qrs_duration_ms/values/3/1
ecg-record:rec-123@1.0.0#/measurements/global/heart_rate_bpm/values
```

For a `["beat","lead"]` field, the first array index is the position in `/axes/beats` and the second is the position in `/axes/leads`. Thus the value at `.../values/3/1` is identified by the immutable pair:

- `/axes/beats/3/id`
- `/axes/leads/1`

The axis arrays are part of the schema contract. Reordering them without changing the document identity is forbidden; changing axis-order semantics is a schema-major change. A producer may create a new record revision, but existing record/pointer pairs must remain immutable.

RFC 6901 escaping applies: `~` becomes `~0` and `/` becomes `~1`. Public field names therefore use lower snake case and avoid `/` to keep pointers readable.

A pointer to an explicit absence reason addresses the sparse state map, for example:

```text
ecg-record:rec-123@1.0.0#/delineation/fiducials/p_onset/absence/unmeasurable/b0004|II
```

The optional NPZ sidecar does not weaken the JSON contract. Its field entry remains addressable at a stable JSON Pointer and adds an `array_key` plus axes. A citation to a sidecar value consists of the field-entry pointer plus the beat and lead axis identities; consumers must not cite ZIP member offsets or NPZ implementation details.

A schema migration must publish a pointer rewrite table when a major version moves or renames public fields. Internal-debug pointers are explicitly excluded from this guarantee.

## G. Versioning policy and profiles

### G.1 Schema SemVer

`schema_version` is required and follows SemVer independently of the Python package version.

A **major** schema version is required for any change that can alter the meaning or address of an existing public value, including:

- removing, renaming, or moving a published field;
- changing type, unit, axes, rounding, null/absence semantics, or numeric interpretation;
- changing the lead/beat axis-order contract;
- changing the JSON Pointer grammar or identity rules;
- changing a required member to optional, or vice versa, in a way that invalidates previously valid consumers;
- changing a field from one physiologic meaning to another even if the JSON shape is identical.

A **minor** version may add optional published fields, add a new profile, add validation evidence, promote an experimental field into a newly specified public field, add a reason code, or add optional provenance. Existing pointers and meanings must remain valid.

A **patch** version is for clarifications and corrections that do not change serialized meaning or validation status. A correction to a field's validation status is at least a minor version because consumers may use validation metadata when filtering outputs.

Library releases may support multiple schema majors concurrently. The library version belongs in `provenance.library_version`; it must never be inferred from `schema_version`.

### G.2 Deprecation and migration

A public field is first marked with `deprecated_since` in a minor release. It remains readable for the rest of that schema major. Removal or a unit/path/meaning change waits for the next major.

Every schema-major release must publish a machine-readable migration map with, for each old public pointer, one of:

- `renamed_to`
- `moved_to`
- `unit_conversion` with an explicit deterministic transform
- `removed` with a reason and replacement if one exists
- `no_equivalent`

A migration must preserve the distinction among plain null, unmeasurable, and not applicable. It must not turn an unavailable reason into an ordinary null.

### G.3 Profiles

Profiles control materialization, not semantic tier. A field's tier and validation status are intrinsic to the schema.

Canonical names are `summary`, `all`, and `debug`. `measurement` is accepted as
a transitional alias for `all`; the previously suggested `all_measurements`
spelling is not a public profile. Extraction defaults to `summary`, but
serialization without an explicit profile preserves the record's existing
profile. Projection may discard fields; it cannot recreate absent all/debug
data. Re-emit retained measurements to request a richer profile.

The initial shadow implementation's `all` profile includes only the currently
admitted fields listed in document 08, not all 88 legacy candidates. Dense
sidecar-backed querying is a target capability, not yet a release claim.

**`summary` is the default published JSON profile.** It contains acquisition, axes, compact quality, reproducibility provenance, the 18 core beat/lead fields budgeted in A.2, compact global measurements, and raw-signal artifact metadata. Its reference-size requirement is <=24,000 uncompressed bytes for 10 s, 12 leads, 500 Hz, 10 beats.

**`all` is the complete published-measurement profile.** It contains every field admitted to the published-measurement tier plus the same acquisition/quality/provenance contract. It has no 24,000-byte JSON SLA. Dense matrices should use the NPZ sidecar when the consumer requests compact output; a fully inline JSON materialization remains valid when direct JSON Pointer access to every numeric cell is required.

**`debug` is a diagnostic superset.** It may include detailed provenance, repair/consensus bookkeeping, localization internals, `st_hybrid_*` when explicitly enabled, `twelve_sl_*` compatibility data, candidate fiducials, and experimental morphology outputs. Debug members are not stable across minor schema versions unless separately documented as public. No size budget applies.

Experimental algorithms never become active merely because `debug` is requested. They require an explicit configuration/policy opt-in, and the resolved path is recorded in `provenance.algorithm_paths`. Thus a record can say that an experimental ST path was selected without presenting its 22 implementation fields as ordinary default measurements.

`input_mode="limited"` is orthogonal to all three profiles. In every profile it still disables diagnosis, axis, and formal QT reporting by marking those capabilities `not_applicable`, while raw measurements that can be computed from the available channels remain eligible.

## H. Worked example

The following is a complete, valid JSON `summary` payload for a deliberately short synthetic record with one beat and two leads. The numeric values are illustrative. The validation entries demonstrate the schema shape; they are not assertions about the repository's current field-by-field benchmark mapping.

```json
{
  "schema_version": "1.0.0",
  "profile": "summary",
  "record_id": "example-short-001",
  "acquisition": {
    "sample_rate_hz": 500,
    "sample_count": 1000,
    "input_mode": "limited",
    "amplitude_unit": "mV",
    "leads": ["I", "II"],
    "capabilities": {
      "diagnosis": {
        "status": "not_applicable",
        "reason": "input_mode_limited"
      },
      "axis": {
        "status": "not_applicable",
        "reason": "input_mode_limited"
      },
      "formal_qt": {
        "status": "not_applicable",
        "reason": "input_mode_limited"
      }
    }
  },
  "axes": {
    "beats": [
      {
        "id": "b0001",
        "r_sample": 500
      }
    ],
    "leads": ["I", "II"]
  },
  "quality": {
    "record": {
      "status": "usable"
    },
    "leads": {
      "I": {
        "status": "usable"
      },
      "II": {
        "status": "limited",
        "reason": "low_snr"
      }
    }
  },
  "provenance": {
    "library_version": "0.0.0-example",
    "schema_version": "1.0.0",
    "input_fingerprint": "sha256:example-input",
    "config": {
      "id": "default-example",
      "sha256": "sha256:example-config"
    },
    "policies": {
      "pacing": {
        "name": "default_pacing_policy",
        "version": "1"
      },
      "qt_rescue": {
        "name": "default_qt_rescue_policy",
        "version": "1"
      }
    },
    "algorithm_paths": {
      "delineation": "default",
      "st": "default",
      "qt": "default"
    },
    "fields": {
      "delineation_default": {
        "method": "selected_consensus"
      },
      "measurement_default": {
        "method": "published_core"
      }
    }
  },
  "delineation": {
    "fiducials": {
      "p_onset": {
        "type": "integer",
        "unit": "sample",
        "axes": ["beat", "lead"],
        "values": [[430, null]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/delineation_default",
        "absence": {
          "unmeasurable": {
            "b0001|II": "low_snr"
          }
        }
      },
      "p_offset": {
        "type": "integer",
        "unit": "sample",
        "axes": ["beat", "lead"],
        "values": [[470, null]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/delineation_default",
        "absence": {
          "unmeasurable": {
            "b0001|II": "low_snr"
          }
        }
      },
      "qrs_onset": {
        "type": "integer",
        "unit": "sample",
        "axes": ["beat", "lead"],
        "values": [[482, 483]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/delineation_default"
      },
      "r_peak": {
        "type": "integer",
        "unit": "sample",
        "axes": ["beat", "lead"],
        "values": [[500, 500]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/delineation_default"
      },
      "qrs_offset": {
        "type": "integer",
        "unit": "sample",
        "axes": ["beat", "lead"],
        "values": [[526, 527]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/delineation_default"
      },
      "j_point": {
        "type": "integer",
        "unit": "sample",
        "axes": ["beat", "lead"],
        "values": [[526, 527]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/delineation_default"
      },
      "t_offset": {
        "type": "integer",
        "unit": "sample",
        "axes": ["beat", "lead"],
        "values": [[690, 694]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/delineation_default"
      }
    }
  },
  "measurements": {
    "intervals": {
      "p_duration_ms": {
        "type": "integer",
        "unit": "ms",
        "axes": ["beat", "lead"],
        "values": [[80, null]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default",
        "absence": {
          "unmeasurable": {
            "b0001|II": "low_snr"
          }
        }
      },
      "pr_interval_ms": {
        "type": "integer",
        "unit": "ms",
        "axes": ["beat", "lead"],
        "values": [[104, null]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default",
        "absence": {
          "unmeasurable": {
            "b0001|II": "low_snr"
          }
        }
      },
      "qrs_duration_ms": {
        "type": "integer",
        "unit": "ms",
        "axes": ["beat", "lead"],
        "values": [[88, 88]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      },
      "qt_interval_ms": {
        "type": "integer",
        "unit": "ms",
        "axes": ["beat", "lead"],
        "values": [[416, 422]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      },
      "tpeak_tend_ms": {
        "type": "integer",
        "unit": "ms",
        "axes": ["beat", "lead"],
        "values": [[78, 82]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      }
    },
    "amplitudes": {
      "p_amplitude_uv": {
        "type": "integer",
        "unit": "uV",
        "axes": ["beat", "lead"],
        "values": [[112, null]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default",
        "absence": {
          "unmeasurable": {
            "b0001|II": "low_snr"
          }
        }
      },
      "r_amplitude_uv": {
        "type": "integer",
        "unit": "uV",
        "axes": ["beat", "lead"],
        "values": [[1040, 1265]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      },
      "s_amplitude_uv": {
        "type": "integer",
        "unit": "uV",
        "axes": ["beat", "lead"],
        "values": [[-310, -420]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      },
      "st_80ms_uv": {
        "type": "integer",
        "unit": "uV",
        "axes": ["beat", "lead"],
        "values": [[24, null]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      },
      "t_amplitude_uv": {
        "type": "integer",
        "unit": "uV",
        "axes": ["beat", "lead"],
        "values": [[345, 318]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      }
    },
    "areas": {
      "qrs_signed_area_uv_ms": {
        "type": "integer",
        "unit": "uV_ms",
        "axes": ["beat", "lead"],
        "values": [[18200, 19750]],
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      }
    },
    "flags": {},
    "global": {
      "heart_rate_bpm": {
        "type": "integer",
        "unit": "1/min",
        "axes": [],
        "values": 60,
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default"
      },
      "frontal_qrs_axis_deg": {
        "type": "integer",
        "unit": "deg",
        "axes": [],
        "values": null,
        "nullable": true,
        "validation": {
          "status": "unvalidated",
          "evidence": []
        },
        "provenance_ref": "/provenance/fields/measurement_default",
        "absence": {
          "kind": "not_applicable",
          "reason": "input_mode_limited"
        }
      }
    }
  },
  "artifacts": {
    "raw_signal": {
      "uri": "example-short-001.npy",
      "media_type": "application/x-npy",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "dense_measurements": null
  }
}
```

The `st_80ms_uv` value for lead II demonstrates the third state: it is plain null because there is no matching explicit absence entry. By contrast, the P-wave values on lead II are explicitly unmeasurable with `low_snr`, and `frontal_qrs_axis_deg` is explicitly not applicable because of the limited input contract. The raw `qt_interval_ms` remains measurable even though the acquisition capability says formal QT reporting is not applicable.

The all-zero artifact digest above is a syntactically valid illustrative
placeholder, not the checksum of a supplied waveform asset. A real writer must
hash the actual artifact bytes, including its container header if applicable.

## Open questions

1. **Field-by-field validation inventory.** The evidence supplied here establishes that some outputs are benchmark-validated and identifies the benchmark families, but it does not give the exact field-to-dataset mapping. Before schema 1.0, every published field needs that mapping; unknown entries remain `unvalidated`. No validation status should be inferred from a neighboring field.

2. **Exact normalized mapping for nested `p`, `qrs`, and `t` objects.** The design fixes the public concepts and the seven-field summary delineation core, but the precise current source member for each selected final onset/offset should be captured in the migration map. This is mapping work, not a reason to expose all candidate indices.

3. **Area units and windows.** The current `p_area`, `qrs_area`, and `t_area` names do not encode units in the supplied evidence. They must not be promoted as `uV_ms` until the implementation's numerical unit and integration window are confirmed. If those semantics differ by method, each method needs either a distinct field definition or a major-version normalization.

4. **NPZ interoperability.** NPZ is the recommended first dense sidecar because the data are N-dimensional numeric arrays and the measurement package is Python-first. If a concrete non-Python consumer requires Arrow IPC later, add it as another media type without changing field semantics or JSON pointers; do not replace NPZ merely for speculative interoperability.

5. **Summary promotion list regression.** The 18-field list is the recommended schema-1.0 core because it fits the 22,295-byte envelope. Before release, measure the exact serializer output on the reference record. If it exceeds 24,000 bytes, reduce wrapper/provenance repetition or move non-core globals to `measurement`; do not weaken absence or validation metadata to save bytes.
