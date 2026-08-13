# Clinical Diagnosis P0/P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the unified ECG rule input contracts and false lead-reversal suppression, then replace all-or-nothing availability with explicit rule coverage, confidence, normality roles and compositional overall statuses.

**Architecture:** Representative lead parameters remain the canonical clinical-rule input boundary. Rule evaluations gain independent status, coverage and confidence dimensions with a legacy migration adapter; the resolver consumes normality roles and produces an overall result that preserves both findings and limitations. Lead-reversal heuristics remain advisory unless a detector explicitly marks a pair confirmed.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest, existing `feature_extraction.ecgfeat` package.

## Global Constraints

- Preserve Glasgow and DXL-inspired analyses as reference-only outputs.
- Do not use dataset header diagnosis codes as runtime inputs.
- Do not convert missing or unreliable evidence into a positive diagnosis or evidence of normality.
- Keep backward reading support for `matched`, `not_matched`, `unavailable`, `not_applicable`, and `suppressed` rule rows.
- Advance schema and ruleset versions for the changed semantics.
- Follow RED/GREEN/REFACTOR for every production change.
- `/workspace/ecg_gemma` is not currently a Git repository; replace commit steps with explicit verification checkpoints and a changed-file inventory.

---

## File map

- `feature_extraction/ecgfeat/features.py`: produce canonical representative measurement fields.
- `feature_extraction/ecgfeat/quality.py`: classify precordial-reversal evidence as advisory or confirmed.
- `feature_extraction/ecgfeat/api.py`: preserve structured reversal metadata.
- `feature_extraction/ecgfeat/clinical_rules/context.py`: consume measurement aliases and exclude only confirmed leads.
- `feature_extraction/ecgfeat/clinical_rules/models.py`: define new rule status, coverage, confidence and normality roles.
- `feature_extraction/ecgfeat/clinical_rules/engine.py`: summarize domain coverage from core roles.
- `feature_extraction/ecgfeat/clinical_rules/resolver.py`: compose findings, coverage and technical limitations.
- `feature_extraction/ecgfeat/clinical_rules/quality.py`: project signal/measurement/domain quality evidence.
- `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py`: consume canonical P duration.
- `feature_extraction/ecgfeat/clinical_rules/conduction.py`: consume representative R-prime evidence.
- `feature_extraction/ecgfeat/clinical_rules/ischemia.py`: consume normalized signed QRS area.
- `feature_extraction/ecgfeat/clinical_rules/sources.py`: advance schema/ruleset versions.
- `feature_extraction/ecgfeat/export.py`: fingerprint versioned resolver semantics.
- `tests/test_clinical_measurement_contract.py`: new producer/consumer contract tests.
- Existing focused test modules: add behavior regressions without duplicating fixtures.

---

### Task 1: Canonical representative measurement contract

**Files:**
- Create: `tests/test_clinical_measurement_contract.py`
- Modify: `feature_extraction/ecgfeat/features.py:612-652`
- Modify: `feature_extraction/ecgfeat/clinical_rules/context.py:35-70`
- Modify: `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py:169-199`
- Modify: `feature_extraction/ecgfeat/clinical_rules/ischemia.py:174-203`

**Interfaces:**
- Consumes: reliable dominant-group `LeadBeatFeatures` values.
- Produces: canonical representative keys `p_dur_consensus_ms`, `r_prime_amp_mv`, and `qrs_signed_area_uv_ms`; `ClinicalContext.lead_value()` resolves documented legacy aliases only.

- [ ] **Step 1: Write failing contract tests**

Add a real `LeadBeatFeatures` fixture derived from `tests/test_lead_reversal.py::_make_beat_feature` and assert:

```python
def test_representative_contract_exports_r_prime_amplitude():
    beat = make_beat("V1", r_prime_amp_mv=0.42, r_prime_duration_ms=46.0)
    reps = build_representative_lead_features([beat], quality_map())
    assert reps["V1"].params["r_prime_amp_mv"] == pytest.approx(0.42)


def test_representative_contract_exports_signed_qrs_area_in_uv_ms():
    beat = make_beat("I", qrs_signed_area=0.125)
    reps = build_representative_lead_features([beat], quality_map())
    assert reps["I"].params["qrs_signed_area_uv_ms"] == pytest.approx(125.0)


def test_rae_consumes_consensus_p_duration():
    context = ContractContext({"II": {"p_amp_mv": 0.30, "p_dur_consensus_ms": 105.0}})
    rae = by_code(evaluate_hypertrophy(context, []), "right_atrial_abnormality")
    assert rae.status == "matched"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_measurement_contract.py
```

Expected: failures for missing representative R-prime and signed-area keys, and RAE unavailable because it reads `p_duration_ms`.

- [ ] **Step 3: Add canonical producer fields**

Add these entries to the representative `params` dictionary:

```python
"r_prime_amp_mv": _pick_representative_value(rep_feature, items, "r_prime_amp_mv"),
"s_prime_amp_mv": _pick_representative_value(rep_feature, items, "s_prime_amp_mv"),
"qrs_signed_area_uv_ms": (
    None
    if (signed := _pick_representative_value(rep_feature, items, "qrs_signed_area")) is None
    else float(signed) * 1000.0
),
```

If the stored signed area is already in microvolt-milliseconds, prove that from its producer test and omit conversion. The contract test must establish the unit with a known waveform value.

- [ ] **Step 4: Add a single documented alias boundary**

In `ClinicalContext`, add:

```python
LEAD_PARAM_ALIASES = {
    "p_duration_ms": ("p_dur_consensus_ms",),
    "qrs_signed_area_uv_ms": ("qrs_signed_area_uv_ms", "twelve_sl_qrs_signed_area_uv_ms"),
}
```

`lead_raw_value()` must try the requested canonical key first and then aliases; it must never substitute zero for missing values.

- [ ] **Step 5: Update rule consumers**

RAE requests `p_dur_consensus_ms`. Sgarbossa requests only canonical `qrs_signed_area_uv_ms`. Keep the alias boundary in context for old artifacts rather than adding fallback logic inside each rule.

- [ ] **Step 6: Verify GREEN and focused regressions**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_measurement_contract.py \
  tests/test_clinical_hypertrophy.py \
  tests/test_clinical_conduction.py \
  tests/test_clinical_ischemia.py \
  tests/test_ecgfeat_pipeline.py -k 'representative_lead_features'
```

Expected: all selected tests pass.

- [ ] **Step 7: Record checkpoint**

Record changed files and focused test output in the working log. Do not invoke Git commands until the workspace becomes a repository.

---

### Task 2: Advisory precordial-reversal state

**Files:**
- Modify: `tests/test_lead_reversal.py`
- Create: `tests/test_clinical_context.py`
- Modify: `feature_extraction/ecgfeat/quality.py:741-775`
- Modify: `feature_extraction/ecgfeat/features.py:741-748`
- Modify: `feature_extraction/ecgfeat/api.py:1546-1560`
- Modify: `feature_extraction/ecgfeat/clinical_rules/context.py:73-94`
- Modify: `feature_extraction/ecgfeat/interpret.py:2360-2382`

**Interfaces:**
- Consumes: structured `metadata.lead_reversal.precordial`.
- Produces: `state` (`not_suspected`, `possible`, `confirmed`), `confidence`, `implicated_leads`, and legacy `suspected` for compatibility.

- [ ] **Step 1: Write failing reversal-state tests**

```python
def test_normal_v5_to_v6_decline_is_advisory_only():
    result = detect_precordial_reversal(
        {"V1": 0.12, "V2": 0.70, "V3": 0.72, "V4": 1.28, "V5": 1.40, "V6": 1.30}
    )
    assert result["state"] in {"not_suspected", "possible"}
    assert result["state"] != "confirmed"


def test_possible_reversal_does_not_exclude_precordial_leads():
    features = make_context_features(
        precordial={"state": "possible", "suspected": True, "implicated_leads": ["V5", "V6"]}
    )
    context = build_context(features)
    assert not ({"V1", "V2", "V3", "V4", "V5", "V6"} & context.excluded_leads)


def test_confirmed_reversal_excludes_only_implicated_pair():
    features = make_context_features(
        precordial={"state": "confirmed", "suspected": True, "implicated_leads": ["V2", "V3"]}
    )
    assert build_context(features).excluded_leads == frozenset({"V2", "V3"})
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_lead_reversal.py tests/test_clinical_context.py
```

Expected: missing `state` and current context exclusion of all V1–V6.

- [ ] **Step 3: Extend detector output without claiming confirmation**

Return:

```python
{
    "suspected": possible,
    "state": "possible" if possible else "not_suspected",
    "confidence": "low" if possible else "high",
    "implicated_leads": list(best_swap or ()),
    "progression_score": progression_score,
    "best_adjacent_swap": best_swap,
    "best_swap_score": best_swap_score,
}
```

The monotonic heuristic never returns `confirmed`. Preserve existing low-voltage and incomplete-map reasons.

- [ ] **Step 4: Change context exclusion policy**

Read structured metadata first. Only when `state == "confirmed"` add `implicated_leads` to `excluded_leads`. A legacy boolean without structured confirmation becomes `possible` and excludes nothing.

- [ ] **Step 5: Preserve advisory reporting**

Keep `precordial_reversal_suspected=True` for existing report compatibility when state is possible, but add structured state to metadata and clinical quality evidence. Change clinical ischemia/Brugada tests so possible reversal adds an advisory instead of suppressing the rule.

- [ ] **Step 6: Verify GREEN and JS00006 regression**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_lead_reversal.py \
  tests/test_clinical_context.py \
  tests/test_clinical_ischemia.py \
  tests/test_js00059_clinical_regression.py
```

Then run a read-only fixture assertion using JS00006 reversal metadata and confirm its reconstructed context excludes zero precordial leads.

- [ ] **Step 7: Record checkpoint**

Record the test output and the before/after excluded-lead set for JS00006.

---

### Task 3: Rule status, coverage and confidence schema

**Files:**
- Modify: `tests/test_clinical_models_resolver.py`
- Create: `tests/test_clinical_schema_migration.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/models.py:7-42`

**Interfaces:**
- Consumes: new and legacy rule rows.
- Produces: `RuleEvaluation(status, coverage, confidence, normality_role)` with deterministic legacy projection.

- [ ] **Step 1: Replace the old missing-input test with failing semantic tests**

```python
def test_missing_optional_input_does_not_erase_proven_match():
    result = RuleEvaluation(
        rule_id="X", domain="rhythm", status="matched",
        missing_inputs=["optional.residual"], coverage="partial",
    )
    assert result.status == "matched"
    assert result.coverage == "partial"


def test_indeterminate_rule_can_explain_missing_decisive_input():
    result = RuleEvaluation(
        rule_id="X", domain="intervals", status="indeterminate",
        missing_inputs=["global.qt_ms"], coverage="unavailable",
    )
    assert result.status == "indeterminate"


def test_legacy_unavailable_row_projects_to_new_dimensions():
    result = RuleEvaluation.from_dict({
        "rule_id": "X", "domain": "intervals", "status": "unavailable"
    })
    assert result.status == "indeterminate"
    assert result.coverage == "unavailable"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_schema_migration.py
```

Expected: `indeterminate` invalid, coverage/normality role missing, and old `__post_init__` overwrites a proven match.

- [ ] **Step 3: Implement new enums and validation**

Define:

```python
VALID_STATUSES = {"matched", "not_matched", "indeterminate", "not_applicable"}
VALID_COVERAGE = {"full", "partial", "unavailable"}
VALID_CONFIDENCE = {"high", "moderate", "low", "unavailable"}
VALID_NORMALITY_ROLES = {"core", "supporting", "optional_screen"}
```

Add fields with compatible defaults:

```python
coverage: str = "full"
confidence: str = "moderate"
normality_role: str = "core"
```

Remove the automatic `missing_inputs -> unavailable` and `suppressed_by -> suppressed` mutations. Evaluators must state their logical outcome explicitly.

- [ ] **Step 4: Implement legacy migration at deserialization**

Add `RuleEvaluation.from_dict()` that maps:

```python
"unavailable" -> status="indeterminate", coverage="unavailable", confidence="unavailable"
"suppressed" -> status="indeterminate", coverage="partial", confidence="low"
```

Map `normality_required=False` to `normality_role="optional_screen"`; retain the old field in serialized output during the migration window.

- [ ] **Step 5: Update rule helper constructors**

Update `_evaluation()` and `_result()` helpers in conduction, hypertrophy and ischemia so every legacy unavailable or suppressed branch explicitly returns the new status/coverage/confidence dimensions.

- [ ] **Step 6: Verify GREEN and serialization compatibility**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_schema_migration.py \
  tests/test_clinical_conduction.py \
  tests/test_clinical_hypertrophy.py \
  tests/test_clinical_ischemia.py \
  tests/test_clinical_intervals.py
```

Expected: all selected tests pass and no serialized rule lacks coverage, confidence or normality role.

- [ ] **Step 7: Record checkpoint**

Record the legacy-to-new mapping table and test output.

---

### Task 4: Domain coverage and compositional resolver

**Files:**
- Modify: `tests/test_clinical_models_resolver.py`
- Modify: `tests/test_clinical_export_consumers.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/engine.py:17-25,117-166`
- Modify: `feature_extraction/ecgfeat/clinical_rules/resolver.py:10-67`
- Modify: `feature_extraction/ecgfeat/clinical_rules/models.py:45-62`

**Interfaces:**
- Consumes: rule status, coverage, confidence and normality role.
- Produces: domain summaries and compositional overall statuses.

- [ ] **Step 1: Write failing resolver tests**

```python
def test_abnormal_plus_missing_core_is_abnormal_with_limited_coverage():
    analysis = resolve([abnormal_rule()], required={"rhythm", "intervals"})
    assert analysis.overall_status == "abnormal_with_limited_coverage"


def test_technical_limitation_preserves_reliable_positive_finding():
    analysis = resolve([technical_rule(), abnormal_rule()])
    assert analysis.overall_status == "technically_limited_with_findings"


def test_optional_screen_does_not_block_normal_core_result():
    analysis = resolve([normal_core_rule(), unavailable_optional_screen()])
    assert analysis.overall_status == "normal_with_core_coverage"


def test_rate_fact_alone_does_not_make_overall_abnormal():
    analysis = resolve([matched_rate_observation(), complete_normal_core_rules()])
    assert analysis.overall_status == "normal_with_core_coverage"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_models_resolver.py
```

Expected: existing resolver returns `abnormal`, `technically_limited`, `incomplete`, or treats rate as abnormal.

- [ ] **Step 3: Replace binary domain availability with summaries**

Implement `_domain_summaries(evaluations)` returning, per domain:

```python
{
    "coverage": "full" | "partial" | "unavailable",
    "confidence": "high" | "moderate" | "low" | "unavailable",
    "core_assessed": int,
    "core_applicable": int,
    "unassessed_reasons": list[str],
}
```

Only applicable `normality_role == "core"` rules determine core normality coverage.

- [ ] **Step 4: Implement resolver precedence**

Use this exact precedence:

```text
technical + abnormal findings -> technically_limited_with_findings
technical without abnormal findings -> technically_limited
abnormal + incomplete core coverage -> abnormal_with_limited_coverage
abnormal + complete core coverage -> abnormal
borderline -> borderline
complete applicable core coverage -> normal_with_core_coverage
otherwise -> incomplete
```

Statements of severity `observation` are final report facts but not members of the abnormal list.

- [ ] **Step 5: Export domain summaries**

Keep detailed rule rows in `domains`; add `summary["domain_coverage"]`, `summary["partial_evaluation"]`, and separate `findings`, `observations`, and `limitations` counts.

- [ ] **Step 6: Verify GREEN and consumer compatibility**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_export_consumers.py \
  tests/test_medgemma_ecg_core.py \
  tests/test_ecg_report_sheet.py
```

Expected: all selected tests pass and consumers render new statuses without falling back to legacy summaries.

- [ ] **Step 7: Record checkpoint**

Record the resolver truth table and focused test output.

---

### Task 5: Quality projection and normality roles

**Files:**
- Create: `tests/test_clinical_quality_domains.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/quality.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/ischemia.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/intervals.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/rhythm.py`

**Interfaces:**
- Consumes: record quality, representative measurement confidence and reversal state.
- Produces: separate signal-quality rule, measurement-quality evidence and rule normality roles.

- [ ] **Step 1: Write failing quality-routing tests**

```python
def test_q0_does_not_override_low_st_measurement_confidence():
    context = make_context(record_grade="Q0", st_j_reliable=False)
    st = by_code(evaluate_ischemia(context, []), "acute_occlusion_pattern")
    assert st.status == "indeterminate"
    assert st.coverage == "unavailable"


def test_optional_brugada_screen_does_not_block_core_normality():
    result = by_code(evaluate_ischemia(no_v1_v2_context(), []), "brugada_type1_screening")
    assert result.normality_role == "optional_screen"


def test_possible_reversal_is_quality_advisory_not_technical_failure():
    quality = evaluate_quality(possible_reversal_context())
    assert any(row.severity == "observation" for row in quality)
    assert not any(row.severity == "technical" and row.status == "matched" for row in quality)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_quality_domains.py
```

Expected: ST uses generic QRS reliability, Brugada defaults to core normality, and reversal advisory is absent.

- [ ] **Step 3: Project structured quality evidence**

The quality evaluator returns a record-quality core rule plus observation rows for possible reversal and measurement limitations. Evidence includes `record_grade`, rejected functions, reliable lead counts, reversal state and excluded leads.

- [ ] **Step 4: Route measurement-specific gates**

Use representative `st_j_reliable` for ST rules, QT reliability/support for QT, and P reliability/association for atrial morphology. Generic `reliable_for_qrs` remains appropriate only for QRS morphology.

- [ ] **Step 5: Assign normality roles**

Set Brugada screening and other rare high-risk screens to `optional_screen`; core rhythm, conduction, interval and acute repolarization assessments remain core when applicable. Supporting voltage criteria use `supporting` when they cannot establish a complete domain-negative conclusion alone.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_quality_domains.py \
  tests/test_clinical_ischemia.py \
  tests/test_clinical_intervals.py \
  tests/test_clinical_models_resolver.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Record checkpoint**

Record the normality-role inventory for every P0/P1 rule.

---

### Task 6: Schema, ruleset, fingerprint and stale artifacts

**Files:**
- Modify: `tests/test_clinical_export_consumers.py`
- Modify: `tests/test_js00059_clinical_regression.py`
- Modify: `feature_extraction/ecgfeat/clinical_rules/sources.py`
- Modify: `feature_extraction/ecgfeat/export.py:48-86`
- Modify: `demo_feature_extraction.py`
- Modify: `medgemma_ecg_core.py`

**Interfaces:**
- Consumes: new clinical schema and resolver policy.
- Produces: `clinical_rules.v2`, a new ruleset version, deterministic fingerprint, and visible stale-artifact warnings.

- [ ] **Step 1: Write failing version/fingerprint tests**

```python
def test_new_semantics_advance_schema_and_ruleset():
    analysis = analyze_clinical(_features())
    assert analysis.schema_version == "clinical_rules.v2"
    assert analysis.ruleset_version == "2026.07.2"


def test_fingerprint_changes_when_resolver_policy_changes():
    first = clinical_fingerprint({"schema_version": "clinical_rules.v2", "resolver_policy": "p1"})
    second = clinical_fingerprint({"schema_version": "clinical_rules.v2", "resolver_policy": "p2"})
    assert first != second
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_export_consumers.py \
  tests/test_js00059_clinical_regression.py
```

Expected: current schema/ruleset assertions still report v1/2026.07.1.

- [ ] **Step 3: Advance versions and fingerprint inputs**

Set:

```python
RULESET_VERSION = "2026.07.2"
SCHEMA_VERSION = "clinical_rules.v2"
RESOLVER_POLICY_VERSION = "coverage-confidence-v1"
```

Include resolver policy in clinical analysis summary so it is covered by the existing deterministic fingerprint.

- [ ] **Step 4: Update consumers and stale warnings**

Consumers accept v1 and v2 for reading, render v2 statuses, and mark stored v1 artifacts stale when the current runtime ruleset is v2. They must never silently relabel a stored v1 result as current.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_export_consumers.py \
  tests/test_js00059_clinical_regression.py \
  tests/test_export_contract.py \
  tests/test_medgemma_ecg_core.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Record checkpoint**

Record schema migration notes, consumer compatibility and focused test output.

---

### Task 7: P0/P1 regression and batch audit

**Files:**
- Modify: `docs/unified_clinical_rules_v1_validation_audit.md` only after generating fresh v2 evidence, or create `docs/unified_clinical_rules_v2_validation_audit.md` if retaining the old audit unchanged.
- Create or modify a deterministic audit script under `scripts/` only if no existing batch audit entry point can produce the required counts.

**Interfaces:**
- Consumes: freshly extracted 100-record v2 artifacts.
- Produces: reproducible coverage, status, field-contract and statement-count audit.

- [ ] **Step 1: Run all focused P0/P1 tests**

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_measurement_contract.py \
  tests/test_clinical_context.py \
  tests/test_clinical_schema_migration.py \
  tests/test_clinical_quality_domains.py \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_export_consumers.py \
  tests/test_clinical_conduction.py \
  tests/test_clinical_hypertrophy.py \
  tests/test_clinical_ischemia.py \
  tests/test_clinical_intervals.py \
  tests/test_lead_reversal.py
```

Expected: zero failures.

- [ ] **Step 2: Run the full existing test suite**

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q
```

Expected: zero failures. If unrelated pre-existing failures exist, capture their exact node IDs and demonstrate they also fail without the P0/P1 patch before classifying them as pre-existing.

- [ ] **Step 3: Regenerate the 100-record artifacts**

Use the existing batch extractor with the same input directory and options used for the v1 audit, writing to a new v2 output directory first. Do not overwrite the v1 evidence until counts and fingerprints are checked.

- [ ] **Step 4: Enforce batch engineering gates**

Assert in the audit script:

```text
required producer fields with zero availability = 0
records excluding all V1-V6 due monotonic heuristic = 0
records partial solely due optional screen = 0
applicable RAE P-duration coverage >= 90%
beat-level R-prime present but representative R-prime absent = 0
schema version clinical_rules.v2 = 100/100
ruleset version 2026.07.2 = 100/100
unique deterministic artifact fingerprint present = 100/100
```

- [ ] **Step 5: Compare v1 and v2 statement distributions**

Report old versus new counts for overall statuses, partial coverage, unavailable domains, reversal states and every final statement code. Do not interpret a higher statement count as improvement without reviewing false positives.

- [ ] **Step 6: Write the v2 audit**

Include the commands, input population, exact code version fields, engineering gates, known limitations, and the boundary that header labels are weak references rather than clinical adjudication.

- [ ] **Step 7: Review P0/P1 acceptance checklist**

Confirm every P0/P1 requirement in the design specification has evidence in a passing test or the v2 audit. List deferred P2 items explicitly: AF/AFL morphology, posterior/acute ischemia evidence tiers, Q-wave territory logic, QT/JT correction and contextual rate severity refinements not already necessary for resolver compatibility.

---

## Plan self-review

- **Spec coverage:** Tasks 1–2 cover P0 input and reversal defects; Tasks 3–5 cover P1 status, resolver and quality architecture; Task 6 covers reproducibility; Task 7 covers regression and audit gates.
- **Scope boundary:** AF/AFL, ischemia, QT/JT and missing diagnostic families remain P2/P3 except for compatibility changes required by the P0/P1 schema. They will receive a separate detailed TDD plan after P0/P1 batch evidence is available.
- **Type consistency:** Canonical rule fields are `status`, `coverage`, `confidence`, and `normality_role`; canonical representative fields are `p_dur_consensus_ms`, `r_prime_amp_mv`, and `qrs_signed_area_uv_ms`.
- **No placeholder behavior:** Each task identifies exact files, expected RED condition, minimal production behavior and verification command.

