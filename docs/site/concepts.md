# Concepts

## The record and the raw signal

The durable product is `(raw signal, ECG Record)`. The record never embeds
samples; `/artifacts/raw_signal` carries the SHA-256 of the little-endian
float64 C-order signal, and every fiducial is a zero-based sample index into
that signal at its native sampling rate.

## Profiles

| Profile | Contents | Size |
|---|---|---|
| `summary` (default) | acquisition, axes, quality, provenance, 7 fiducials, 5 intervals, 5 amplitudes, 1 area, heart rate, frontal QRS axis | ≤ 24,000 bytes for a 10 s, 12-lead, 10-beat record (contract-tested) |
| `all` | every published measurement (adds JT interval, U amplitude, QTc Bazett, wide-QRS duration) | no budget |
| `debug` | `all` plus selected engine metadata; no compatibility guarantee | no budget |

`summary ⊂ all ⊂ debug`, and shared values are identical across profiles.

## Absence states

A missing value is never a bare "no number":

- **plain null**: no stronger claim (imported data);
- **`unmeasurable(reason)`**: applicable, but no trustworthy value (for example
  `measurement_unavailable`, `fiducial_order_violation`, `negative_interval`);
- **`not_applicable(reason)`**: excluded by the input contract (for example formal
  QT and axis under `input_mode="limited"`).

Dense fields encode reasons as a whole-field state, a per-field `default`
state, or sparse `<beat_id>|<lead>` entries that override the default.

## Addressing

`ecg-record:<record_id>@<schema_version>#<RFC 6901 JSON pointer>`, for example
`ecg-record:rec-…@1.0.0#/measurements/intervals/qrs_duration_ms/values/3/1`
(beat index 3, lead index 1 in `/axes`). `resolve_address` checks record
identity and schema version before resolving.

## Validation tiers

Each field's `validation.status` comes from the shipped registry
(`validation-evidence.json`) and can only exceed `unvalidated` by citing an
evidence manifest. All fields in schema 1.0.0 are `unvalidated`.

## Dense sidecar

`serialize_record(record, sidecar_uri="x.npz")` moves dense matrices to a
deterministic NPZ next to the JSON (SHA-256, record id and schema version bound
on both sides; `uint8` state mask cross-checked against the JSON absence
encoding). `load_record` verifies it; queries read it transparently.

## Input modes

`standard_12` requires the twelve canonical leads, each named once, in any row
order. `limited` accepts 1–8 explicitly named channels and never synthesizes
missing leads.
