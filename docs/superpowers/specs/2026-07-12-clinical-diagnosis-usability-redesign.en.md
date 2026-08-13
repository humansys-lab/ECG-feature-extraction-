<!-- i18n-nav -->
[中文](2026-07-12-clinical-diagnosis-usability-redesign.md) | [English](2026-07-12-clinical-diagnosis-usability-redesign.en.md) | [日本語](2026-07-12-clinical-diagnosis-usability-redesign.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# ECG Clinical Diagnosis Usability Redesign

**Date:** 2026-07-12  
**Status:** Design approved in conversation; written-spec review pending  
**Scope:** Public-guideline unified clinical rules, their measurement contracts,
availability semantics, report resolution, and validation. The Glasgow and
DXL-inspired layers remain reference-only.

## 1. Objective

Make the unified ECG interpretation clinically useful without converting
missing or unreliable evidence into diagnoses. The redesign must correct the
known producer/consumer contract defects, stop false lead-reversal suppression,
separate diagnostic polarity from coverage, improve the AF/AFL, ischemia and QT
paths, and make every exported result reproducible and measurable.

The target is not to maximize the number of abnormal statements. It is to
produce the strongest supported conclusion with explicit confidence and
coverage, while preserving enough detail for manual review.

## 2. Non-goals

- No claim of medical-device certification or proprietary Glasgow/Philips/DXL
  equivalence.
- No automatic diagnosis of clinical syndromes from ECG evidence alone where
  symptoms, serial ECGs, biomarkers, imaging, family history or genetics are
  required.
- No threshold relaxation merely to increase apparent sensitivity.
- No use of header diagnosis codes as runtime diagnostic inputs.
- No broad rewrite of delineation algorithms unrelated to a demonstrated rule
  input defect.

## 3. Delivery strategy

The work is delivered as independently testable phases in one implementation
series:

1. **P0 — Contract and suppression repairs:** correct impossible field reads,
   make lead-reversal evidence advisory unless confirmed, and repair artifact
   versioning.
2. **P1 — Availability and quality model:** separate rule conclusion, coverage
   and confidence; introduce core versus optional normality requirements; make
   the overall resolver preserve positive findings and limitations together.
3. **P2 — Clinical rule corrections:** rebuild AF/AFL projection, ischemia and
   posterior screening, QT/JT handling, P-dependent suppression and contextual
   rate reporting.
4. **P3 — Coverage and validation:** add missing high-value rule families and
   batch-level performance/coverage gates.

Every behavior change follows a failing-test-first RED/GREEN cycle. A phase is
not accepted until its focused tests, the clinical regression suite and a
100-record batch audit pass their stated gates.

## 4. Architecture

### 4.1 Measurement contract

`RepresentativeLeadFeatures.params` is the only public input contract for
lead-level unified rules. A contract registry defines canonical names, units,
producer functions and allowed backward-compatible aliases.

Canonical repairs:

- P duration: `p_dur_consensus_ms`, with a documented pooled beat-level
  fallback when consensus is unavailable.
- R-prime amplitude: `r_prime_amp_mv`, aggregated from reliable dominant-group
  beat measurements and exported in representative parameters.
- Signed QRS area: one canonical representative key and unit. Existing
  `qrs_signed_area` and Twelve-SL values are normalized at the producer
  boundary, not guessed independently by clinical rules.

Contract tests must fail when a required rule input is never produced or when
its unit differs between producer and consumer.

### 4.2 Lead-reversal state

Lead reversal becomes a confidence-bearing observation:

- `not_suspected`
- `possible`
- `confirmed`

The current monotonic R-progression improvement heuristic may generate only
`possible`; it cannot exclude any chest lead. A normal V5-to-V6 amplitude
decrease is not evidence of reversal by itself. `confirmed` requires an
independently validated detector with morphology and cross-beat support.

For `possible`, rules run on the recorded lead identities and attach a quality
advisory. For `confirmed`, only the implicated lead pair is remapped or excluded;
the system never discards all V1–V6 because one adjacent pair is suspicious.
Limb and precordial reversal states remain separate.

### 4.3 Rule evaluation model

Each rule exports three independent dimensions:

- `status`: `matched`, `not_matched`, `indeterminate`, or `not_applicable`.
- `coverage`: `full`, `partial`, or `unavailable`.
- `confidence`: `high`, `moderate`, `low`, or `unavailable`.

`missing_inputs` is explanatory data and does not automatically overwrite a
logically proven positive or negative status. A rule decides whether a missing
input can change its result. Legacy `suppressed` and `unavailable` values remain
accepted during schema migration and are deterministically projected into the
new dimensions.

Rules also declare `normality_role`:

- `core`: required before the applicable domain can support a normal result.
- `supporting`: improves confidence but does not block core normality.
- `optional_screen`: absence or unavailability never blocks a general normal
  ECG conclusion.

### 4.4 Domain and overall resolution

A domain summary contains:

- strongest supported positive conclusion;
- assessed core-rule count and applicable core-rule count;
- coverage and confidence;
- explicit unassessed reasons.

Overall statuses are compositional:

- `abnormal`
- `abnormal_with_limited_coverage`
- `technically_limited_with_findings`
- `technically_limited`
- `borderline`
- `normal_with_core_coverage`
- `incomplete`

An abnormal finding is never hidden by an unrelated missing domain. A technical
limitation is never allowed to hide reliable positive findings. A normal result
requires full coverage of applicable core rules, not every rare optional
screen. Rate-only findings are contextual observations and do not by themselves
make the ECG globally abnormal.

### 4.5 Quality model

Quality is split into:

- `signal_quality`: noise, clipping, flatline, lead-map completeness and
  reversal evidence;
- `measurement_quality`: per measurement family and lead, including beat
  support, variability, provenance and confidence;
- `domain_quality`: the measurements needed by a specific clinical rule.

ST rules consume ST-J reliability rather than generic QRS reliability. QT
rules consume QT-specific support. P-dependent rules consume P association and
P morphology validity. Record grade remains an overview, not the sole clinical
gate.

## 5. Clinical rule behavior

### 5.1 AF and atrial flutter

AF supports two evidence routes:

1. Classic surface-ECG route: irregular RR activation plus absent consistent
   P-to-QRS association across reliable leads.
2. Validated residual route: QRST-subtracted residual evidence for ambiguous,
   regular or low-P-confidence recordings.

QRST subtraction increases confidence but is not mandatory for a classic AF
pattern. `organized_p_ratio` is replaced by the fraction of eligible beats with
one consistent associated organized P event, constrained to `[0, 1]`. Multiple
candidate events in one beat cannot increase the numerator.

Atrial flutter gains a public morphology route using organized atrial activity,
atrial rate/cycle range, inferior-lead/V1 morphology support and plausible AV
conduction ratio. Spectral and residual evidence may support this route but
cannot independently produce a high-confidence diagnosis. Ambiguous AF/AFL
evidence returns `indeterminate`, never mutually contradictory final diagnoses.

### 5.2 Conduction

RBBB uses representative R-prime amplitude and duration when available, plus
alternative accepted V1/V2 terminal morphologies and lateral terminal S-wave
evidence. No single missing morphology field makes RBBB impossible when an
equivalent morphology path is complete.

Fascicular and nonspecific IVCD rules use territory-level short-circuit logic:
known disqualifying evidence can produce `not_matched`; missing evidence only
causes `indeterminate` when it could change the result.

### 5.3 Hypertrophy and atrial abnormality

LVH exports each voltage criterion separately and reports “no criterion among
assessed criteria” for partial negative coverage. It continues to describe ECG
voltage criteria, not anatomical LVH.

RAE consumes the canonical P-duration field. LAE/RAE are `not_applicable` when
AF/AFL or invalid P association makes P morphology uninterpretable. RVH supports
multiple evidence paths and preserves explicit suppression by confirmed RBBB or
LPFB where appropriate.

### 5.4 Ischemia and infarction

Acute ST elevation is a high-priority screening result, not a standalone acute
coronary occlusion diagnosis. It requires contiguous leads, ST-specific
measurement confidence, age/sex thresholds where applicable, and explicit
confounder handling. Its statement requires urgent clinical correlation.

Posterior ischemia screening requires V1–V3 ST depression plus supporting
terminal T or posterior morphology evidence. V7–V9 elevation, when available,
raises confidence. BBB/IVCD and poor ST confidence yield an indeterminate
high-risk screen rather than an unqualified positive or negative.

Pathological Q-wave evaluation is territory based. Tiny physiologic Q
amplitudes do not make missing duration clinically decisive. Duration becomes a
required fact only for a Q wave whose amplitude or Q/R ratio could satisfy a
pathological criterion. Each territory reports its own coverage.

Sgarbossa consumes normalized QRS polarity and ST evidence. Original and
modified criteria are reported separately and are not combined into one
undocumented score.

### 5.5 Intervals and rate

QT exports Bazett, Fridericia, Hodges and Framingham values with the selected
formula and selection reason. Wide QRS records use JT/JTc or a documented
modified-QT path instead of becoming automatically unavailable. Irregular
rhythms use robust beat-paired QT/RR aggregation where sufficient data exist.

Short QT is stratified:

- QTc `<= 340 ms`: markedly short, high-priority manual confirmation.
- QTc `341–360 ms`: possible short-QT pattern requiring clinical/family context.
- QTc `361–390 ms`: borderline interval only, not an abnormal syndrome-level
  statement.

PR and atrial-interval rules become not applicable when P-to-QRS association is
invalid. Bradycardia and tachycardia remain reported facts, but their severity
is contextual and rate alone does not force a globally abnormal ECG.

### 5.6 Missing high-value coverage

After P0–P2 stabilize, P3 adds unified rule families for:

- primary and secondary ST-T abnormalities;
- PAC/PVC and ectopy burden;
- low voltage;
- pre-excitation;
- second- and third-degree AV block evidence;
- paced-rhythm handling when pacing detection is enabled.

Each family requires a separate clinical definition, input contract and labeled
validation slice before becoming a core normality rule.

## 6. Compatibility and reproducibility

The schema version advances because rule status semantics change. The ruleset
version advances on every diagnostic behavior change. The exported artifact
fingerprint includes schema version, ruleset version, rule definitions and
resolver policy.

Reports generated under an older resolver remain readable but are explicitly
marked stale when loaded by a newer ruleset. Existing top-level
`clinical_interpretation` remains the authoritative output location. Reference
algorithms never overwrite unified final statements.

The current 100 JSON/report artifacts must be regenerated after implementation;
re-resolving stale evaluations is useful for debugging but is not accepted as a
final validation run.

## 7. Error handling

- Unknown contract fields fail focused contract tests and produce an explicit
  runtime `indeterminate/unavailable` reason; they never silently become zero.
- Non-finite measurements are normalized to missing at the producer boundary.
- Unit conversions occur once at the producer boundary and carry provenance.
- Evaluator exceptions are isolated to that rule and exported as a technical
  evaluation error without converting other domains to normal.
- Contradictory evidence is preserved in the audit payload and resolved to
  `indeterminate` unless a documented priority rule applies.

## 8. Testing and validation

### 8.1 Unit and contract tests

- Producer/consumer field-name and unit tests for every unified-rule input.
- Lead reversal tests covering normal V5-to-V6 decline, low voltage, poor R
  progression, true synthetic adjacent swaps and partial lead maps.
- Rule short-circuit tests proving irrelevant missing inputs do not erase known
  positives or negatives.
- Resolver tests for every compositional overall status.
- AF/AFL, RBBB, hypertrophy, ischemia, Q-wave, QT/JT and rate boundary tests.
- Serialization and backward-compatibility tests for the new schema.

### 8.2 Batch validation

The 100-record set is an engineering regression set, not sufficient clinical
validation. It must nevertheless meet these gates:

- zero rule-required fields that the producer can never emit;
- zero records made partial solely by an optional screen;
- zero exclusion of all V1–V6 from the monotonic-R heuristic;
- RAE measurement coverage at least 90% when P morphology is applicable;
- any record with beat-level R-prime evidence exposes the representative
  R-prime contract;
- 100% of artifacts report the current schema/ruleset and a reproducible
  fingerprint;
- every change in final statement counts is included in the audit report.

For diagnostic metrics, unavailable/indeterminate cases are reported both as
excluded conditional metrics and as failures in effective sensitivity. Reports
include coverage, sensitivity, specificity, PPV, NPV, F1 and Wilson confidence
intervals stratified by quality, sex, age and rhythm.

### 8.3 Clinical validation boundary

Dataset header labels are weak reference labels and cannot adjudicate acute
ischemia, posterior infarction or syndrome-level QT diagnoses alone. High-risk
rule disagreements require blinded clinician review of the waveform and
relevant clinical context before thresholds are promoted to production use.

## 9. Acceptance criteria

The redesign is complete when:

1. All P0–P2 contract and clinical regression tests pass from a clean command.
2. The full existing test suite has no new failures.
3. The 100-record extraction is regenerated with the new ruleset.
4. No record is globally incomplete because only an optional screen is absent.
5. Batch audit clearly distinguishes abnormality, coverage, confidence and
   technical limitations.
6. AF, RBBB and RAE are no longer structurally impossible because of internal
   fields or validation gates.
7. Posterior ischemia and short-QT outputs use the redesigned evidence tiers.
8. The remaining limitations and unimplemented P3 families are explicitly
   listed rather than represented as completed normal evaluations.

