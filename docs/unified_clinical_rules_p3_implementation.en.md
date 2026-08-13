<!-- i18n-nav -->
[中文](unified_clinical_rules_p3_implementation.md) | [English](unified_clinical_rules_p3_implementation.en.md) | [日本語](unified_clinical_rules_p3_implementation.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# Unified Clinical Rules P3 Engineering Implementation

Date: 2026-07-12

Ruleset: `clinical_rules.v2` / `2026.07.4`

Status: engineering implementation and regression validation only. This is not
clinical validation and does not establish medical-device performance.

## Implemented unified capabilities

- Primary versus secondary T-wave abnormality screening based on reliable,
  contiguous-lead T-wave inversion or tall positive T-wave patterns. BBB/IVCD,
  ventricular hypertrophy, pacing and pre-excitation are retained as secondary
  repolarization causes rather than being reported as primary abnormalities.
- Structured PAC/PVC candidates with beat IDs, analyzable-beat denominators,
  burden percentages, morphology context and compensatory-pause support. PAC is
  not applicable during confirmed AF. A reference-only flutter candidate lowers
  confidence but does not suppress a unified PAC finding.
- Separate limb-lead and precordial low-QRS-voltage rules. One reliable lead at
  or above the regional threshold disproves the corresponding all-leads rule;
  every lead in the region is required for a positive result.
- Ventricular pre-excitation projection from short-PR/PR-segment, multilead
  delta-wave and QRS evidence. A positive pattern suppresses dependent
  conduction, hypertrophy and ischemia statements.
- Second-degree and complete AV-block projections. Second-degree output requires
  both a classified pattern and dropped-P evidence. Dropped-P evidence without
  stable P-QRS classification is `indeterminate`, not negative. Complete block
  requires the upstream complete-block flag plus independent AV-relation
  evidence.

All high-risk P3 families except descriptive low voltage remain
`optional_screen` for normality. Their positive findings can surface, but their
absence does not expand the meaning of `normal_with_core_coverage` before
clinical validation.

## Architecture and coverage changes

Rhythm mechanisms now run before conduction, hypertrophy, ischemia and T-wave
resolution. This permits pre-excitation and rhythm context to suppress or
qualify dependent morphology statements.

The clinical summary now includes `capability_coverage` for core and optional
families. Optional screens that are unavailable or partial remain visible rather
than silently appearing as completed normal evaluations. The existing
`domain_coverage` field retains its core-normality meaning.

## Engineering verification

Focused P3, clinical, rhythm, statement and export contract suite:

```text
150 passed, 1 deselected
```

The deselected test imports the unavailable external `snapshot_regression`
module.

Broader clinical, feature-pipeline, atrial, report, batch, MedGemma, MI and
repolarization selection:

```text
462 passed, 1 deselected, 2 environment failures
```

The two failures import `app.py` and are caused by the environment not having
the optional `gradio` dependency. They are unrelated to the clinical-rule
changes.

Real-record smoke checks with current extraction included:

- `JS00019`: secondary T-wave abnormality surfaced with IVCD context.
- `JS00027`: one structured PAC surfaced; a reference-only flutter candidate no
  longer hides it.
- `JS00030`: PVC surfaced while PAC and higher-grade AV-block logic were not
  applied through confirmed AF.
- `JS02212`: the third-degree AV-block weak label produced a unified complete
  AV-block pattern.
- `JS01355`: dropped-P evidence without a stable second-degree classification is
  exported as `indeterminate`; the PAC candidate is retained at partial/low
  confidence because of contradictory reference-only flutter evidence.

## Remaining validation boundaries

- The local ventricular-pre-excitation weak-label slice is extremely small.
  `JS01944` has two delta-positive leads but measured PR 163 ms and PR segment
  75 ms, so it correctly does not meet the implemented short-PR pattern rather
  than causing a label-driven threshold relaxation.
- The single local second-degree AV-block weak label lacks stable dropped-P/PR
  classification. It remains indeterminate pending waveform adjudication.
- A low-voltage weak label (`JS01156`) disagrees with the extracted QRS
  peak-to-peak amplitudes in multiple leads. The implementation preserves the
  measured evidence and does not tune thresholds to force agreement.
- Flat and biphasic T-wave morphology still needs a stronger structured
  producer contract before it can extend the current contiguous inversion/tall-T
  rule.
- Dataset headers remain weak labels. Promotion of optional screens into core
  normality requires blinded waveform adjudication and larger positive slices.
