# Frequently asked questions

### Can I use the measurements clinically?

No. This is research and engineering software, not a medical device. Every
published field is `unvalidated` in schema 1.0.0. See
[Limitations and intended use](limitations.md).

### Why does `ecg_record` insist on lead names?

A mislabelled or transposed signal produces plausible but wrong numbers, and
nothing downstream can detect that. Explicit names make the input contract
checkable. In standard mode the rows may come in any order; see
[Preparing input signals](guide/input.md).

### My data is `(n_samples, 12)`. What do I do?

Transpose it: `ecg_record(signal.T, ...)`. The error
`signal_lead_count_mismatch` usually means the array is transposed.

### Why is a value `None`?

Look at `result.absence`: `unmeasurable(reason)` means the pipeline could not
produce a trustworthy value, `not_applicable(reason)` means the input excludes
the quantity (for example the axis in limited mode), and `NullAbsence` means
no reason was recorded. See [Absence states](guide/record-format.md#absence-states).

### Why does the same signal get a different `record_id` after upgrading?

The id includes the library version, because a new version may measure
differently. Pass your own `record_id=` if you need ids that do not change.

### How do I get a per-beat value for every lead?

Loop over `record.axes["leads"]` and call `query_measurement` for each lead, or
read the `[beat][lead]` matrix at
`record.field("/measurements/intervals/<name>")["values"]`. See
[A table of values](guide/querying.md#a-table-of-values).

### Which beat index should I use?

Beat indices are positions in `record.axes["beats"]` (zero-based, in time
order). Each beat has an `id` and a reference `r_sample`. There is no single
"representative" beat in the record; aggregate per-beat values yourself, for
example with a median over beats.

### Does `ecg-records` diagnose rhythms or conditions?

No. Interpretation is a separate distribution, `ecginterpret`, which has not
been released yet. `pip install "ecg-records[interpret]"` cannot be resolved
until it is.

### `ECGFeatureExtractor().extract()` raises `ImportError` about `ecginterpret`. Why?

The legacy extractor always included the interpretation. Use the record API
(`ecg_record`), which never needs interpretation. See the
[migration guide](migration.md).

### `ECGConfig(p_wave=ecgfeat.PWaveConfig())` fails. Is that a bug?

It is a known issue in 0.1.0: the top-level `ecgfeat.PWaveConfig` name refers
to the legacy engine's class. P-wave settings are fixed in 0.1.0 anyway, so
leave `p_wave` unset. See [Configuration](guide/configuration.md#p-wave-settings).

### Can I read records without installing SciPy or the engine?

Reading, validating and querying records does not load the numerical engine.
The package still depends on NumPy and SciPy at install time. Records are
plain JSON with a published JSON Schema, so other languages can read them
directly.

### How big is a record?

About 18 kB for a 10 s, 12-lead, 10-beat `summary` record, with a contracted
maximum of 24,000 bytes. Longer recordings grow with the number of beats. For
very long recordings, move the dense matrices into an
[NPZ sidecar](guide/serialization.md#dense-sidecars).

### How long can a recording be?

At least 1 second, with no fixed maximum. Processing time and record size grow
with length; the pipeline is tuned and tested on 10-second, diagnostic-style
recordings.

### Where do I report problems?

Open an issue at
<https://github.com/humansys-lab/ECG-feature-extraction-/issues>, ideally with
the record (JSON) and, if you can share it, the signal.
