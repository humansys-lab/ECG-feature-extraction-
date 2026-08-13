# Unified Clinical ECG Rules Design

Date: 2026-07-12

## 1. Objective

Introduce one authoritative clinical interpretation layer for the ECG feature
extractor. Public clinical standards define diagnostic behavior. Existing
Glasgow and Philips/DXL-inspired outputs remain available as traceable reference
interpretations, but they no longer decide the final summary.

The new layer must eliminate these current failure modes:

- a partial Glasgow implementation reporting a normal ECG while diagnostic
  chapters are pending;
- Glasgow and DXL-inspired logic producing incompatible interval or morphology
  conclusions without arbitration;
- missing measurements satisfying a positive rule;
- axis-only fascicular-block diagnoses;
- non-contiguous or demographically incorrect ST-elevation decisions;
- old generated artifacts appearing current after rule changes.

The implementation remains a research system. Completing the engineering rule
set does not claim medical-device certification, proprietary Glasgow/Philips
equivalence, or clinical validation.

## 2. Confirmed Decisions

1. Public AHA/ACC/HRS, ESC, and Universal Definition of Myocardial Infarction
   standards are the authority for final clinical statements.
2. Existing output fields remain available for compatibility.
3. Glasgow and DXL-inspired conclusions are reference-only.
4. A missing required input produces `unavailable` or `indeterminate`; it can
   produce neither a positive diagnosis nor evidence of normality.
5. The implementation uses a new clinical rules package alongside the current
   feature extractor rather than rewriting signal processing.
6. The final resolution component is named `ClinicalStatementResolver`.

## 3. Architecture

The data flow is:

```text
ECGFeatureExtractor / ECGFeatures
            |
            v
ClinicalContext adapter
            |
            v
quality -> rhythm -> conduction/intervals -> hypertrophy -> ST/T/MI
            |
            v
ClinicalStatementResolver
            |
            v
clinical_interpretation.v1
            |
            +-> report
            +-> app
            +-> MedGemma
            +-> batch exports
```

The new package is `feature_extraction/ecgfeat/clinical_rules/`:

- `models.py`: rule specifications, evaluations, domain results, conflicts,
  summary and complete analysis models;
- `context.py`: normalized patient, measurement, reliability, lead-orientation,
  pacing and reference-result facts;
- `quality.py`: technical limitations and domain availability;
- `rhythm.py`: unified projections of the existing rhythm evidence;
- `conduction.py`: QRS width, BBB and fascicular-block rules;
- `intervals.py`: PR and QT calculations and classifications;
- `hypertrophy.py`: ECG voltage-criteria statements;
- `ischemia.py`: ST deviation, Q-wave and infarction-pattern rules;
- `resolver.py`: suppression, conflicts, domain completeness and final summary;
- `engine.py`: deterministic orchestration;
- `compatibility.py`: non-destructive legacy annotations and projections;
- `sources.py`: stable rule-source identifiers and rule-set version.

No rule evaluator is allowed to read presentation text or another evaluator's
rendered statement. Rules consume only normalized facts and structured prior
domain results.

## 4. Rule Evaluation Contract

Every rule evaluation contains:

```json
{
  "rule_id": "CLIN-CONDUCTION-LAFB-01",
  "domain": "conduction",
  "status": "matched",
  "statement_code": "lafb_pattern",
  "statement": "ECG pattern consistent with left anterior fascicular block",
  "severity": "abnormal",
  "confidence": "criteria_met",
  "required_inputs": [],
  "missing_inputs": [],
  "evidence": {},
  "thresholds": {},
  "suppressed_by": [],
  "source": {
    "authority": "AHA/ACC/HRS",
    "document": "Recommendations for Standardization and Interpretation of the ECG, Part III",
    "section": "Fascicular blocks",
    "version": "2009"
  }
}
```

Allowed evaluation statuses are:

- `matched`
- `not_matched`
- `unavailable`
- `not_applicable`
- `suppressed`

`unavailable` is mandatory when any required measurement is missing, not finite,
unreliable, excluded because of lead orientation, or unsupported for the routed
patient population.

## 5. Domain Behavior

### 5.1 Quality and availability

Quality is evaluated first. Limitations apply to the smallest affected domain:

- limb-lead reversal limits frontal axis and limb-lead morphology;
- precordial reversal limits precordial ST, Q-wave, BBB, RVH and R-progression
  rules;
- P-wave unreliability limits atrial rhythm, PR and atrial enlargement;
- QRS unreliability limits conduction, Q-wave and hypertrophy rules;
- T/QT unreliability limits repolarization and QT rules.

A technically limited domain is not a negative domain.

### 5.2 Rhythm

The clinical layer reuses structured QRST-subtracted atrial residuals, organized
P-wave evidence, RR variability, pacing context, premature-complex evidence and
AV relationship evidence already produced by the extractor.

It does not infer AF solely from RR coefficient of variation. AF/AFL statements
must expose recording duration, atrial-signal availability, organized-P ratio,
RR evidence, validation status and confidence. Pacing and AV-block restrictions
are resolved before dependent rhythm statements.

### 5.3 Conduction

- Nonspecific IVCD requires the guideline QRS-duration criterion and absence of
  complete RBBB/LBBB morphology.
- RBBB and LBBB require both duration and morphology evidence.
- Incomplete BBB labels require their specific morphology and duration range.
- LAFB requires the defined left-axis range plus qR in aVL and rS/inferior-lead
  morphology. Axis alone is insufficient.
- LPFB requires its right-axis range plus rS in I/aVL and qR in III/aVF, with
  competing causes of right-axis deviation represented as confounders.
- If component duration or required morphology is not measured, the relevant
  diagnosis is unavailable rather than inferred from amplitude alone.

### 5.4 QT and PR intervals

The payload exposes Bazett, Fridericia, Hodges and Framingham calculations.
Adult final QT classification uses Hodges as the primary correction because it
is a linear correction and is already the Glasgow configuration default.
Formula-specific values and thresholds remain visible.

Bazett is reference-only at materially non-60 bpm rates and must not independently
produce the final prolonged-QT statement. Automated QT prolongation includes a
manual-review advisory. QT classification is unavailable when the T end is
unreliable, RR variability invalidates correction, or the configured wide-QRS
guard applies.

Adult PR classification uses `<120 ms` for short PR and `>200 ms` for
first-degree AV delay. Pediatric PR classification requires a supported
age-specific public reference table; until that table is encoded, the pediatric
PR classification is unavailable. Vendor-specific age/heart-rate thresholds
remain visible only in their reference interpretations.

### 5.5 Hypertrophy

The clinical layer reports named ECG voltage criteria, not anatomical chamber
hypertrophy as a confirmed diagnosis. Cornell voltage/product and Sokolow-Lyon
criteria retain sex, age, conduction and lead-quality context. Results use text
such as `meets ECG voltage criteria for LVH`.

RVH and atrial-abnormality statements require the morphology and duration facts
specified by their rule. Missing component duration does not count as passing.
Pediatric percentile rules that are not fully encoded are unavailable and do not
fall back to adult positive rules.

### 5.6 ST, Q waves and infarction patterns

- ST elevation uses J-point measurements in anatomically contiguous leads.
- V2-V3 thresholds are selected by sex and male age; other leads use their
  guideline threshold.
- Every contributing lead must be reliable and not excluded.
- ST depression territory statements use contiguous leads and distinguish
  nonspecific repolarization abnormality from an ischemic pattern.
- LBBB and ventricular pacing enter a dedicated Sgarbossa evaluation path.
- Posterior ischemia on V1-V3 is reported as suggestive and recommends V7-V9.
- Q-wave rules require measurable duration and the applicable amplitude or
  Q/R criterion in the required lead grouping.
- Lead-placement error, pre-excitation, BBB, LVH/RVH and other known confounders
  are retained in evidence and can downgrade or suppress a statement.
- ECG output says `ECG pattern suggestive of acute ischemia/occlusion` rather
  than diagnosing acute myocardial infarction without clinical and biomarker
  evidence.

The first implementation includes a standard-V1/V2 Brugada type-1 screening
rule using coved J/ST elevation, descending ST morphology and negative T-wave
evidence. It is reported as a screening pattern, not a Brugada syndrome
diagnosis, because high-intercostal recordings and clinical criteria are not
available. Other high-risk patterns that cannot be evaluated appear as explicit
unavailable rule families; they cannot be silently absent from a normal summary.

### 5.7 Pediatric routing

Pediatric routing is explicit and independent from the legacy Glasgow and DXL
age definitions. Each pediatric rule states its supported age range. Missing
percentile tables or age precision causes an unavailable result for dependent
rules. No unsupported pediatric rule falls back to an adult positive diagnosis.

## 6. ClinicalStatementResolver

The resolver does not define medical thresholds. It performs only these control
functions:

1. propagate domain unavailability;
2. apply technical and diagnostic suppression relationships;
3. deduplicate equivalent statements;
4. retain contradictory reference conclusions as conflicts;
5. rank final and borderline statements;
6. determine whether all normality-required domains are complete;
7. produce the single final summary.

Allowed overall summaries are:

- `normal`
- `borderline`
- `abnormal`
- `technically_limited`
- `incomplete`

`normal` is allowed only when all normality-required domains were evaluated,
quality is sufficient, no matched abnormal/borderline statement exists, and no
critical required input is missing. Pending or unimplemented rule families force
`incomplete`.

## 7. Output Contract

The full export gains:

```json
{
  "clinical_interpretation": {
    "schema_version": "clinical_rules.v1",
    "ruleset_version": "2026.07.1",
    "artifact_fingerprint": "sha256:4c4518b2-example",
    "generated_at": "2026-07-12T12:00:00Z",
    "overall_status": "incomplete",
    "summary": {},
    "final_statements": [],
    "borderline_statements": [],
    "suppressed_statements": [],
    "unavailable_domains": [],
    "conflicts": [],
    "domains": {},
    "reference_interpretations": {
      "glasgow": {},
      "dxl": {}
    }
  }
}
```

Existing `interpretation`, `glasgow` and `statement_engine` structures remain.
Their top-level export metadata gains:

```json
{
  "reference_only": true,
  "superseded_by": "clinical_interpretation"
}
```

The compatibility addition is non-destructive: existing keys and value types are
not removed or renamed.

## 8. Presentation and Consumers

- The text and image reports show the unified summary first.
- Glasgow and DXL-inspired results move to a reference-algorithm appendix.
- App and MedGemma context use `clinical_interpretation` as the authoritative
  result and describe reference conflicts separately.
- Batch exports include the unified fields and ruleset version.
- Report rendering compares the embedded fingerprint with the current payload.
  A mismatch displays `STALE ARTIFACT` and prevents the artifact from being
  presented as current.

The fingerprint is deterministic over the ruleset version, normalized clinical
analysis and source feature identity. `generated_at` is excluded from the hash.

## 9. Error Handling

- A single rule exception becomes an unavailable evaluation with an error code;
  it does not disappear.
- Failure in a normality-required rule forces an incomplete summary.
- Non-finite measurements are treated as missing.
- Unknown sex, insufficient age precision, unsupported leads and absent clinical
  context are represented explicitly.
- Strict mode raises evaluator errors for development and tests; normal runtime
  records them in the analysis.

## 10. Testing and Acceptance

### 10.1 Rule boundaries

Each threshold has below/equal/above tests. Tests cover contiguous-lead logic,
age and sex routes, quality gates, confounders and missing inputs.

### 10.2 Resolver invariants

- unavailable required evidence can never create a matched rule;
- any missing normality-required domain forbids `normal`;
- reference algorithms cannot alter the unified final summary;
- a precordial reversal suppresses dependent precordial diagnoses;
- LBBB/pacing uses the dedicated ischemia path;
- conflicts remain visible.

### 10.3 End-to-end regression

JS00059 is reprocessed from the raw record. Its new JSON and reports must not
contain the stale contradictions between QT classification, precordial reversal,
RVH and posterior-MI output. Legacy fields remain readable.

Additional fixtures cover AF, AFL, pacing, AV block, RBBB, LBBB, fascicular
blocks, WPW, ST-elevation patterns, posterior patterns, Q waves, LVH voltage,
low voltage and pediatric boundaries.

### 10.4 Clinical validation interface

The repository gains evaluation utilities for sensitivity, specificity, PPV,
NPV and F1 by diagnosis, age, sex and signal-quality stratum. Unit tests are not
presented as clinical validation. Research-use labeling remains until an
independent clinician-adjudicated validation meets separately approved targets.

### 10.5 Engineering acceptance

- all new rule tests pass;
- no new failures appear in the currently runnable legacy suite;
- App, MedGemma and batch consumer contract tests pass;
- all regenerated JS00059 artifacts share one ruleset version and fingerprint;
- report and JSON outputs clearly identify reference-only algorithms;
- incomplete or unimplemented domains never produce a normal summary.

## 11. Delivery Sequence

1. Models, sources, context and resolver invariants.
2. Quality and domain-completeness behavior.
3. QT and conduction corrections.
4. ST/Q-wave/ischemia rules and Sgarbossa path.
5. Hypertrophy and pediatric availability policy.
6. Export compatibility and consumer migration.
7. Artifact fingerprinting and report regeneration.
8. Validation utilities and full verification.

## 12. Non-Claims

This work does not claim:

- exact proprietary Glasgow or Philips algorithm reproduction;
- replacement for cardiologist review;
- diagnosis of acute myocardial infarction from ECG alone;
- medical-device certification;
- clinical accuracy solely because unit tests pass.
