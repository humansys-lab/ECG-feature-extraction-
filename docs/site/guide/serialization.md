# Saving and loading

## Canonical JSON

Records serialize to **canonical JSON**: UTF-8, keys sorted, no insignificant
whitespace, no `NaN` or `Infinity`. The same record always gives the same
bytes, so you can hash, deduplicate and diff records.

```python
import numpy as np
from ecgfeat import ecg_record, dumps_record, loads_record

signal = np.load("ecg.npy")
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
record = ecg_record(signal, sampling_rate=500, lead_names=leads)

data = dumps_record(record)                 # bytes
assert dumps_record(loads_record(data)) == data
pretty = dumps_record(record, indent=2)     # human-readable, same content
```

| Function | Purpose |
|---|---|
| `dumps_record(record, *, profile=None, indent=None) -> bytes` | canonical bytes; `profile` must be the record's own profile or a smaller one |
| `loads_record(data, *, validate="strict") -> ECGRecord` | parse bytes or `str` |
| `ecg_dump(record, destination, *, indent=None)` | write JSON to a path or text stream |
| `load_record(source, *, validate="strict", sidecar=None)` | read from a path, stream or mapping; resolves sidecars |
| `record_to_dict(record) -> dict` | detached plain-Python mapping (same as `record.as_dict()`) |

A `summary` record of a 10 s, 12-lead recording with 10 beats stays within
24,000 bytes. The test suite enforces this contract.

## Dense sidecars

Per-beat, per-lead matrices dominate record size. `serialize_record` can move
them into a companion **NPZ sidecar** next to the JSON. The JSON keeps every
other member, plus a descriptor that binds the sidecar to it:

```python
import numpy as np
from ecgfeat import ecg_record, serialize_record, load_record, query_measurement

signal = np.load("ecg.npy")
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
record = ecg_record(signal, sampling_rate=500, lead_names=leads, profile="all")

encoded = serialize_record(record, sidecar_uri="ecg.dense.npz")
with open("ecg.record.json", "wb") as handle:
    handle.write(encoded.json_bytes)
for name, data in encoded.sidecars.items():        # {"ecg.dense.npz": b"PK..."}
    with open(name, "wb") as handle:
        handle.write(data)

restored = load_record("ecg.record.json")          # finds ecg.dense.npz next to it
print(query_measurement(restored, "qrs_duration_ms", lead="V1", beat=0).value)
```

Properties of the sidecar:

- **Same directory.** `sidecar_uri` is a plain file name, resolved next to the
  JSON file. Pass `sidecar=` (a path or the NPZ bytes) to `load_record` when
  the JSON does not come from a file.
- **Integrity-bound in both directions.** The JSON records the sidecar's
  SHA-256, and the sidecar records the record id and schema version. Each
  matrix carries a `uint8` state mask that must agree with the JSON absence
  encoding. Any mismatch raises `SidecarError` (a `RecordValidationError`).
- **Deterministic.** The same record gives the same NPZ bytes.
- **Transparent.** Queries read sidecar-backed values exactly like inline
  ones. `strict` loading verifies the sidecar immediately; `schema` and `none`
  verify it on first access.

To get an ordinary, self-contained record back:

```python
from ecgfeat import load_record, materialize_record, dumps_record

inline = materialize_record(load_record("ecg.record.json"))
with open("ecg.inline.json", "wb") as handle:
    handle.write(dumps_record(inline))
```

Write the sidecar before (or together with) the JSON, so that a record file
never points at a sidecar that does not exist yet. The command line does this
for you (`ecg-record measure --sidecar`).

## Storing the raw signal

The record identifies its signal but does not contain it. Keep both, and check
that they still belong together:

```python
import hashlib
import numpy as np
from ecgfeat import load_record

record = load_record("ecg.record.json")
signal = np.load("ecg.npy")
digest = "sha256:" + hashlib.sha256(np.ascontiguousarray(signal, dtype="<f8").tobytes()).hexdigest()
assert digest == record.artifacts["raw_signal"]["sha256"]
```

`ecgfeat.viz` performs this check automatically before plotting.

## Reading records in other languages

A record is plain JSON validated by a standard JSON Schema (draft 2020-12)
shipped at `ecgfeat/schemas/ecg-record/1.0/schema.json`. Any JSON library can
read it. To resolve an absence state without Python, apply the rules in
[Absence states](record-format.md#absence-states). The strict reader's
cross-field invariants are documented in
[The ECG Record](record-format.md#published-fields).
