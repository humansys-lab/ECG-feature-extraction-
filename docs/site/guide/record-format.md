# The ECG Record

An ECG Record is one JSON object: the published, versioned result of measuring
one recording. This page describes schema **1.0.0** member by member. The
machine-readable definition is the JSON Schema shipped in the package
(`ecgfeat/schemas/ecg-record/1.0/schema.json`), and the
[schema reference](../reference/ecg-record-schema-1.0.md) lists every published
field with its profiles and validation ceiling.

In Python a record is an immutable `ECGRecord`. Its top-level members are
attributes (`record.acquisition`, `record.measurements`, …; nested mappings are
read-only), and `record.as_dict()` returns a detached, mutable copy.

## The record and the raw signal

The durable product is the pair **(raw signal, ECG Record)**. The record never
embeds samples. `/artifacts/raw_signal` carries the SHA-256 of the signal
exactly as measured (little-endian float64, C order, shape
`[n_leads, n_samples]`), so a record can always be checked against the signal
it claims to describe. Every fiducial is a zero-based sample index into that
signal at its native sampling rate.

## Top-level members

| Member | Required | Content |
|---|---|---|
| `schema_version` | yes | `"1.0.0"`: the version of this format, independent of the package version |
| `profile` | yes | `"summary"`, `"all"` or `"debug"` ([profiles](measuring.md#profiles)) |
| `record_id` | yes | identifier used in every [address](querying.md#addresses); any non-empty string without `@`, `#` or line breaks |
| `intended_use` | no | always `"research_and_engineering_only"` in records this package writes |
| `acquisition` | yes | how the signal was recorded (below) |
| `axes` | yes | the beat and lead axes that index every matrix |
| `quality` | yes | record and per-lead signal quality |
| `provenance` | yes | library, configuration, methods and policies that produced the values |
| `delineation` | yes | fiducial points (`/delineation/fiducials/*`) |
| `measurements` | yes | intervals, amplitudes, areas, record-level quantities, flags |
| `artifacts` | yes | identity of the raw signal; optional dense-matrix sidecar |
| `metadata` | no | published patient metadata (`/metadata/patient`) when supplied |
| `debug` | no | engine diagnostics in the `debug` profile only; no compatibility guarantee |
| `extensions` | no | reserved for namespaced third-party additions |

Readers must ignore members they do not understand: later `1.x` schema versions
only add optional members (see [Compatibility](../compatibility.md)).

### `acquisition`

| Key | Example | Meaning |
|---|---|---|
| `sample_rate_hz` | `500.0` | native sampling rate of the measured signal |
| `sample_count` | `5000` | samples per lead |
| `input_mode` | `"standard_12"` | `standard_12` or `limited` |
| `amplitude_unit` | `"mV"` | unit of the input samples |
| `leads` | `["I", "II", …]` | lead names in row order |
| `capabilities` | see below | what the input supports as a whole |

`capabilities` states, for `axis`, `formal_qt` and `diagnosis`, whether the
record supports them (`{"status": "available"}`) or why not
(`{"status": "not_applicable", "reason": "input_mode_limited"}`). `diagnosis`
is always `not_applicable` with reason `measurement_package_boundary`: this
package measures and never diagnoses.

### `axes`

- `axes.leads`: the lead names; the second index of every per-lead matrix.
- `axes.beats`: one object per detected beat, `{"id": "b0001", "r_sample": 250}`;
  the first index of every per-beat matrix. `r_sample` is the beat's reference
  R sample. Beat ids are stable within a record and are used in
  [absence overrides](#absence-states).

### `quality`

```json
{"record": {"status": "usable", "grade": "Q0"},
 "leads": {"II": {"status": "usable"}, "V1": {"status": "limited", "reason": "quality_gate"}}}
```

- `quality.record.grade` is the record signal-quality grade from `Q0` (best) to
  `Q3`. `status` is `usable` for `Q0`/`Q1` and `limited` otherwise.
- `quality.leads.<lead>.status` is `usable` or `limited`. A `limited` lead
  failed the per-lead quality gate; its values are still published, but treat
  them with care.

### `provenance`

| Key | Meaning |
|---|---|
| `library_version` | package version that produced the record |
| `schema_version` | schema version written |
| `input_fingerprint` | SHA-256 over the samples, rate, lead names and unit |
| `config`, `config_hash` | `config.resolved` is the fully resolved [configuration](configuration.md); `config.id` and `config_hash` are its SHA-256 |
| `method_requested`, `method_resolved` | the requested method (`default`) and the algorithm that ran (`legacy_dxl_inspired`) |
| `fallback_path` | fallbacks taken, in order (empty when none) |
| `experimental` | names of enabled experimental refinements (empty by default) |
| `internal_sample_rate_hz`, `mains_frequency_hz` | internal analysis rate and mains-interference setting |
| `fiducial_sample_domain`, `coordinate_conversion` | how internal positions were mapped to native sample indices |
| `algorithm_paths`, `policy_execution` | which algorithm paths and policies ran |
| `fields` | named provenance entries that fields point to with `provenance_ref` |

## Fields

Every value in `delineation` and `measurements` lives in a **field object**.
This is `/measurements/intervals/qrs_duration_ms` of the reference record,
showing the first two of its ten beats:

```json
{
  "type": "integer",
  "unit": "ms",
  "axes": ["beat", "lead"],
  "values": [[78, 78, 70, 94, 88, 70, 154, 168, 204, 116, 80, 76],
             [76, 78, 70, 94, 108, 70, 158, 148, 234, 116, 78, 76]],
  "nullable": true,
  "validation": {"status": "unvalidated", "evidence": []},
  "provenance_ref": "/provenance/fields/measurement_default"
}
```

| Key | Meaning |
|---|---|
| `type` | JSON type of each value (`integer` for all published fields) |
| `unit` | `sample`, `ms`, `uV`, `uV_ms`, `1/min` or `deg` |
| `axes` | `["beat", "lead"]` for a matrix indexed like `values[beat][lead]`, or `[]` for a single record-level value |
| `values` | the numbers; `null` where there is no value |
| `nullable` | whether `null` may appear |
| `validation` | the field's validation status and evidence ([Validation](../validation.md)) |
| `provenance_ref` | JSON pointer to the provenance entry for this field |
| `absence` | why values are missing ([below](#absence-states)); omitted when nothing needs explaining |

### Units and rounding

| Unit | Used for | Convention |
|---|---|---|
| `sample` | fiducials | zero-based index into the raw signal at `sample_rate_hz` |
| `ms` | intervals | rounded half-to-even to whole milliseconds |
| `uV` | amplitudes | microvolts, rounded half-to-even; sign preserved (negative S and ST values) |
| `uV_ms` | QRS signed area | microvolt-milliseconds |
| `1/min` | heart rate | beats per minute |
| `deg` | frontal QRS axis | degrees |

### Published fields

| Group | Field | Axes | Profiles |
|---|---|---|---|
| `delineation/fiducials` | `p_onset`, `p_offset`, `qrs_onset`, `r_peak`, `qrs_offset`, `j_point`, `t_offset` | beat × lead | all |
| `measurements/intervals` | `p_duration_ms`, `pr_interval_ms`, `qrs_duration_ms`, `qt_interval_ms`, `tpeak_tend_ms` | beat × lead | all |
| `measurements/intervals` | `jt_interval_ms` | beat × lead | `all`, `debug` |
| `measurements/amplitudes` | `p_amplitude_uv`, `r_amplitude_uv`, `s_amplitude_uv`, `st_80ms_uv`, `t_amplitude_uv` | beat × lead | all |
| `measurements/amplitudes` | `u_amplitude_uv` | beat × lead | `all`, `debug` |
| `measurements/areas` | `qrs_signed_area_uv_ms` | beat × lead | all |
| `measurements/global` | `heart_rate_bpm`, `frontal_qrs_axis_deg` | record | all |
| `measurements/global` | `qtc_bazett_ms`, `qrs_wide_ms` | record | `all`, `debug` |

"all" in the Profiles column means `summary`, `all` and `debug`.
`measurements/flags` is reserved and empty in schema 1.0.

Invariants that every record written by this package satisfies:

- within each wave, landmarks are ordered: `p_onset ≤ p_offset` and
  `qrs_onset ≤ r_peak ≤ qrs_offset` for every beat and lead where both are
  present (the strict reader checks this for any record);
- every fiducial lies inside the signal (`0 ≤ sample < sample_count`);
- no published interval is negative.

A cell that would violate one of these is published as
`unmeasurable(fiducial_order_violation)`, `unmeasurable(sample_outside_acquisition)`
or `unmeasurable(negative_interval)`, and the dependent duration is withheld
with it. The rest of the record is still published.

## Absence states

A missing value is never a bare "no number". Each `null` cell has one of three
states:

| State | Meaning | Example reasons |
|---|---|---|
| plain `null` | no stronger claim is made | a value absent in imported data |
| `unmeasurable(reason)` | the quantity applies to this beat and lead, but no trustworthy value exists | `measurement_unavailable`, `fiducial_order_violation`, `negative_interval`, `sample_outside_acquisition` |
| `not_applicable(reason)` | the input contract excludes the quantity | `input_mode_limited` |

Reasons are an open vocabulary: any non-empty string is valid. The reasons
listed are the ones this package writes; records from other producers (like
the example at the end of this page, with `low_snr`) may carry others.

The `absence` member of a field encodes the reasons compactly, in one of these
forms:

```json
{"kind": "not_applicable", "reason": "input_mode_limited"}
```

**Whole field**: every `null` in the field has this state (typical for
record-level values).

```json
{"default": {"kind": "unmeasurable", "reason": "measurement_unavailable"},
 "unmeasurable": {"b0003|V1": "fiducial_order_violation"}}
```

**Default plus overrides.** Each `null` cell whose coordinate is listed under
`unmeasurable` or `not_applicable` takes that reason. Every other `null` cell
takes the `default`. Cells that hold a value are unaffected.

```json
{"unmeasurable": {"b0001|II": "low_snr"}}
```

**Overrides only.** The listed cells have a reason. An unlisted `null` cell is
a plain null.

Coordinates are `<beat id>|<lead name>`, which is why lead names may not
contain `|`. You rarely need to decode this yourself: `query_measurement`
returns the resolved state of any cell as `result.absence`
([Querying](querying.md)).

## Validation status

Each field carries `validation.status`, taken from the validation-evidence
registry shipped with the package, and never higher than the registry allows.
**In schema 1.0.0 every published field is `unvalidated`.** See
[Validation and benchmarks](../validation.md).

## Complete example

A complete, valid `summary` record for a short two-lead recording in limited
mode. It shows all three absence forms: `low_snr` overrides on lead II, a
plain `null` in `st_80ms_uv`, and a whole-field `not_applicable` frontal axis.
The repository's test suite validates this exact example against the JSON
Schema and the strict reader.

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
