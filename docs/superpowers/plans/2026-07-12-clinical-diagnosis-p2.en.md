<!-- i18n-nav -->
[中文](2026-07-12-clinical-diagnosis-p2.md) | [English](2026-07-12-clinical-diagnosis-p2.en.md) | [日本語](2026-07-12-clinical-diagnosis-p2.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# Clinical Diagnosis P2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the clinical rule paths that remain unusable or overcall disease after P0/P1: Q-wave territory coverage, posterior ischemia, short/wide-QRS QT, AF evidence normalization and P-dependent applicability.

**Architecture:** Each rule uses decisive-evidence short-circuiting and exports evidence tiers rather than weakening thresholds. High-risk ECG patterns remain screening findings with explicit confirmation requirements. The existing v2 status/coverage/normality-role model is used without another schema change; the ruleset advances once after all P2 behavior is complete.

**Tech Stack:** Python 3.12, NumPy, pytest, existing `feature_extraction.ecgfeat` measurement and clinical-rule layers.

## Global Constraints

- Do not treat missing data as zero or normal evidence.
- Do not use dataset diagnosis labels at runtime.
- Keep acute ischemia and syndrome-level QT statements explicitly advisory/manual-review.
- Preserve raw formula values and per-lead facts in exported evidence.
- Follow RED/GREEN/REFACTOR for each task.
- Current workspace has no Git repository; record verification checkpoints instead of commits.

---

### Task 1: Territory-aware pathological Q-wave logic

**Files:**
- Modify: `tests/test_clinical_ischemia.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/ischemia.py:99-143`

**Interfaces:**
- Consumes: per-lead Q amplitude, Q duration and R amplitude.
- Produces: per-territory status/coverage and a global match only from at least two qualifying contiguous/grouped leads.

- [ ] Add failing tests proving a tiny Q with missing duration does not make the rule unavailable, a potentially pathological Q with missing duration remains indeterminate only in its territory, and a complete normal territory can return not matched.
- [ ] Run `PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_ischemia.py -k q_wave` and verify RED.
- [ ] Prefilter Q waves with `abs(q_amp) < 0.10` and absent pathological Q/R ratio before duration can become decisive.
- [ ] Track `territory_results` with assessed, candidate-missing and qualifying leads. Return matched if any territory has two qualifying leads; return unavailable only when unresolved candidate leads could still form a qualifying pair; otherwise return not matched with partial coverage evidence.
- [ ] Run the focused test plus `tests/test_mi_qwave.py`; expect zero failures.

### Task 2: Posterior ischemia evidence tiers

**Files:**
- Modify: `tests/test_clinical_ischemia.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/ischemia.py:146-171`

**Interfaces:**
- Consumes: reliable V1–V3 ST-J, T amplitude and R/S morphology.
- Produces: `posterior_ischemia_screen` only when ST depression has supporting posterior morphology; isolated ST depression becomes borderline/indeterminate evidence.

- [ ] Add failing tests: two depressed leads without positive terminal T or dominant R do not match; depression plus positive T in two leads matches moderate-confidence screening; confirmed RBBB/IVCD confounding returns indeterminate.
- [ ] Run the posterior-focused tests and verify RED.
- [ ] Require at least two V1–V3 leads with ST `<= -0.05 mV` plus supporting positive T (`> 0.02 mV`) or dominant R (`R > abs(S)`) in the depressed leads. Store which support path matched.
- [ ] Rename the final code to `posterior_ischemia_screen`, keep V7–V9 recommendation, set confidence moderate, and require manual confirmation.
- [ ] Run `tests/test_clinical_ischemia.py` and expect zero failures.

### Task 3: QT evidence tiers and wide-QRS handling

**Files:**
- Modify: `tests/test_clinical_intervals.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/intervals.py`

**Interfaces:**
- Consumes: QT, HR, sex, QRS duration, RR stability and optional JT.
- Produces: markedly short (`<=340`), possible short (`341–360`), borderline short (`361–390`), prolonged QT, or wide-QRS repolarization review.

- [ ] Add failing boundary tests at 340, 350, 380 and 391 ms selected QTc; assert 380 is borderline rather than abnormal.
- [ ] Add a failing test proving QRS `>=120 ms` yields a `wide_qrs_repolarization_review` observation with JT/JTc evidence when JT is available instead of unavailable QT.
- [ ] Run interval tests and verify RED.
- [ ] Replace the single 390-ms short threshold with tiered statements and severities. Keep formula values and manual-review flags.
- [ ] Pass `global.jt_ms` from context into the QT evaluator and compute a documented JTc observation for wide QRS; do not diagnose long-QT syndrome from that fallback.
- [ ] Run `tests/test_clinical_intervals.py tests/test_clinical_export_consumers.py`; expect zero failures.

### Task 4: AF evidence normalization and P-dependent applicability

**Files:**
- Modify: `tests/test_af_afl_features.py`
- Modify: `tests/test_clinical_models_resolver.py`
- Modify: `tests/test_clinical_hypertrophy.py`
- Modify: `feature_extraction/ecgfeat/api.py:1492-1507`
- Modify: `feature_extraction/ecgfeat/clinical_rules/rhythm.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py`

**Interfaces:**
- Consumes: atrial events grouped per eligible beat, RR CV, P association and residual validation.
- Produces: organized-P fraction constrained to `[0,1]`, classic AF evidence independent of QRST validation, and not-applicable atrial morphology under AF/AFL.

- [ ] Add a failing test with two candidate P events on one beat and assert organized-P fraction counts that beat once and never exceeds 1.0.
- [ ] Add failing clinical tests proving classic RR/P AF can match without residual validation and RAE/LAE become not applicable when AF or AFL is matched.
- [ ] Run focused AF and hypertrophy tests and verify RED.
- [ ] Compute organized beats as a set of associated beat IDs divided by eligible R-peak beat count.
- [ ] Expose a helper for active atrial arrhythmia and return not-applicable atrial morphology rules with the suppressor evidence.
- [ ] Keep flutter unavailable unless public morphology inputs exist; do not promote spectral evidence alone.
- [ ] Run the focused tests and expect zero failures.

### Task 5: P2 version, focused regression and 100-record audit

**Files:**
- Modify: `feature_extraction/ecgfeat/clinical_rules/sources.py`
- Modify: affected version assertions.
- Create: `docs/unified_clinical_rules_v2_validation_audit.md`

**Interfaces:**
- Consumes: final P2 code and fresh 100-record extraction.
- Produces: ruleset `2026.07.3` with before/P0-P1/P2 distributions and known limitations.

- [ ] Write a failing ruleset assertion for `2026.07.3`.
- [ ] Advance only `RULESET_VERSION`; keep `clinical_rules.v2` schema.
- [ ] Run all `tests/test_clinical_*.py`, AF/AFL, Q-wave, lead reversal and export consumer tests.
- [ ] Run the broader suite excluding the 14 documented environment-dependent baseline failures.
- [ ] Recompute all 100 records without overwriting v1 artifacts and report overall status, partial coverage, unavailable domains, final-code counts, AF/RBBB/RAE coverage, posterior-screen count and QT tiers.
- [ ] Accept P2 only if Q-wave-driven ischemia unavailability materially decreases, posterior screening requires support evidence, no organized-P ratio exceeds 1, and all artifacts use v2/2026.07.3.

