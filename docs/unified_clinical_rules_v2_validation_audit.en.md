<!-- i18n-nav -->
[中文](unified_clinical_rules_v2_validation_audit.md) | [English](unified_clinical_rules_v2_validation_audit.en.md) | [日本語](unified_clinical_rules_v2_validation_audit.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# Unified Clinical Rules v2 Validation Audit

Date: 2026-07-12

Ruleset: `clinical_rules.v2` / `2026.07.3`

Status: engineering validation only. This audit does not establish clinical
validation or medical-device performance.

Follow-up: P3 engineering coverage was added in ruleset `2026.07.4`; see
`docs/unified_clinical_rules_p3_implementation.md`. The distributions below are
the historical P0-P2 / `2026.07.3` audit and were not rewritten as P3 results.

## Scope

This audit covers the P0–P2 diagnosis-usability redesign:

- canonical representative measurement contracts;
- advisory versus confirmed lead-reversal handling;
- independent rule coverage and normality roles;
- compositional overall statuses;
- measurement-specific ST quality;
- territory-aware Q-wave availability;
- posterior ischemia evidence tiers;
- short-QT tiers and wide-QRS JT/JTc review;
- organized conducted-P/PR clustering and classic AF precedence;
- P-morphology non-applicability during AF/AFL.

The 100 records under `data/010` were extracted in memory with the current
source. Existing root-level `JS*_features.json` and `JS*_report.txt` artifacts
were not overwritten and remain v1 evidence until explicitly regenerated.

## Root-cause repairs

### Measurement contracts

- Representative parameters now export `r_prime_amp_mv` and
  `s_prime_amp_mv` from dominant-group beat measurements.
- Representative parameters expose the native Twelve-SL
  `qrs_signed_area_uv_ms`; Sgarbossa can fall back to native signed-area
  polarity without inventing a unit conversion.
- RAE consumes `p_dur_consensus_ms`, the field produced by the representative
  layer, rather than the nonexistent `p_duration_ms` key.

Observed engineering results:

- P-duration available for RAE assessment: 92/100.
- Beat-level R-prime present but representative R-prime absent: 0 records.
- RBBB output: one definite and three probable patterns. The four reference
  RBBB labels require statement-level review before treating probable and
  definite tiers as equivalent.

### Precordial reversal

The monotonic R-progression heuristic now returns only `possible`, never
`confirmed`. Possible reversal remains an observation and excludes no lead.
Only independently confirmed reversal may exclude its explicitly implicated
lead pair.

Batch states:

- `possible`: 64
- `not_suspected`: 36
- all-V1–V6 exclusions caused by the heuristic: 0

This preserves the advisory signal while removing the previous catastrophic
loss of all chest-lead evidence.

### Availability and resolver

Rule evaluations now carry `coverage` and `normality_role` independently from
their conclusion. Optional screens do not block core normality. The resolver
supports abnormal findings with limited coverage and technical limitations with
reliable findings. Rate abnormalities are observations rather than standalone
global disease classifications.

Schema/ruleset consistency during the in-memory batch: 100/100.

## Clinical rule changes

### Q waves

Q-wave duration is decisive only when Q amplitude or Q/R ratio could meet a
pathological criterion. A territory becomes indeterminate only when unresolved
candidate leads could still form the required two-lead pattern.

Effect on unavailable ischemia domains:

- P0/P1 intermediate: 92/100 unavailable.
- P2 final: 35/100 unavailable.

Four prior-infarct Q-wave patterns remain final findings and require manual
waveform/confounder review.

### Posterior ischemia

Two negative ST-J measurements are no longer sufficient. A final screen
requires:

- depression in at least two V1–V3 leads;
- persistent depression at ST-mid/ST80 or horizontal/downsloping morphology;
- positive terminal T or dominant R support;
- no unhandled BBB/IVCD confounder.

Intermediate P0/P1 result: 38 posterior-pattern positives.

First P2 support-only result: 28 positives. Review showed 63/66 involved leads
were upsloping and commonly returned above baseline by ST-mid/ST80.

Final P2 persistent-depression result: 0 positives.

Zero does not prove perfect specificity; it shows that the earlier calls were
not supported by persistent posterior ST depression in this engineering set.

### QT and JT

Short QT output is tiered:

- `<=340 ms`: markedly short, abnormal manual-review finding;
- `341–360 ms`: possible short-QT pattern, borderline;
- `361–390 ms`: borderline short QT;
- `>390 ms`: not short by this rule.

The 13 former `short_qt` abnormal statements became 13
`borderline_short_qt` statements. Wide-QRS records can produce a
`wide_qrs_repolarization_review` JT/JTc observation rather than becoming
unavailable solely because QRS is at least 120 ms.

### AF

The old event-count ratio could exceed 1 and was dominated by retrograde and
random candidate events. The new ratio counts distinct beats in the dominant
conducted-P PR cluster within ±30 ms and is bounded to `[0,1]`.

Classic AF evidence is:

- RR CV at least 0.15;
- consistent conducted-P ratio below 0.60;
- non-repetitive residual evidence;
- no requirement for validated QRST subtraction when the classic surface-ECG
  route is complete.

Reference-only spectral flutter evidence no longer vetoes a complete classic AF
pattern. It remains a flutter candidate only when classic AF is absent.

Against weak header labels:

- TP: 20
- FN: 2
- FP: 1
- TN: 77
- organized-P ratio above 1: 0

These values are engineering regression evidence, not clinical performance.
The single apparent false positive and two apparent false negatives require
waveform adjudication.

## Final 100-record distribution

### Overall status

| Status | Count |
|---|---:|
| `normal_with_core_coverage` | 10 |
| `borderline` | 8 |
| `abnormal` | 6 |
| `abnormal_with_limited_coverage` | 43 |
| `technically_limited_with_findings` | 7 |
| `technically_limited` | 9 |
| `incomplete` | 17 |

Records with partial evaluation: 75/100. The old v1 artifacts had 84
`incomplete`, 16 `technically_limited`, and no normal or abnormal overall
resolution.

### Unavailable domains

| Domain | Count |
|---|---:|
| intervals | 44 |
| ischemia/infarction | 35 |
| atrial abnormality | 12 |
| hypertrophy | 5 |
| conduction | 2 |

Optional high-risk screening and possible precordial reversal no longer create
unavailable core domains.

### Final statement counts

| Code | Count |
|---|---:|
| possible precordial lead reversal observation | 64 |
| bradycardia observation | 51 |
| atrial fibrillation pattern | 21 |
| tachycardia observation | 19 |
| LVH voltage criteria | 18 |
| borderline short QT | 13 |
| acute occlusion screening pattern | 10 |
| technically limited | 16 |
| first-degree AV delay | 5 |
| nonspecific IVCD | 5 |
| prolonged QT | 5 |
| prior-infarct Q-wave pattern | 4 |
| probable RBBB | 3 |
| left atrial abnormality | 3 |
| LPFB | 3 |
| definite RBBB | 1 |
| LAFB | 1 |
| wide-QRS repolarization review | 1 |
| posterior ischemia screen | 0 |

## Automated verification

Focused clinical, AF/AFL, atrial-event, Q-wave and reversal suite:

```text
127 passed, 3 deprecation warnings
```

Broader export, report and feature-pipeline suite:

```text
267 passed, 1 deselected, 985 deprecation warnings
```

The deselected test imports a missing external `snapshot_regression` module.
The unfiltered baseline additionally contains 13 LUDB regression failures due
to a missing historical virtual environment and missing LUDB waveform files.
These environment failures existed before the diagnosis changes and are not
used as evidence of success.

## Remaining limitations

- Intervals remain unavailable in 44 records, mainly due to QT measurement
  confidence and irregular-RR correction constraints. Beat-paired AF QT/RR
  aggregation remains future work.
- Ischemia remains unavailable in 35 records where potentially decisive Q-wave
  evidence is incomplete.
- Acute ST findings still require waveform review, symptoms, serial ECGs and
  biomarkers; ten screen positives must not be treated as adjudicated coronary
  occlusions.
- A public surface-morphology atrial-flutter final rule is not implemented;
  spectral evidence remains reference-only.
- Unified primary/secondary T-wave abnormalities, PAC/PVC burden, low voltage,
  pre-excitation and higher-grade AV block coverage remain P3 work.
- Header diagnosis codes are weak labels and cannot replace blinded clinician
  adjudication.
