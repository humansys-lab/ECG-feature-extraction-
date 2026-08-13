# Unified Clinical ECG Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one authoritative, safety-conservative clinical ECG interpretation layer while preserving Glasgow, DXL-inspired, App, MedGemma, batch, and report compatibility.

**Architecture:** Keep the existing ECG measurement pipeline intact and adapt `ECGFeatures` into a normalized `ClinicalContext`. Independent public-guideline rule modules emit structured evaluations, and `ClinicalStatementResolver` alone produces the final summary; legacy interpretations remain reference-only.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest/unittest, existing `ecgfeat` package and report/export code.

## Global Constraints

- Public AHA/ACC/HRS, ESC, and Universal Definition of Myocardial Infarction standards are authoritative for final statements.
- Missing required evidence returns `unavailable`; it never satisfies a positive or normal rule.
- Existing `interpretation`, `glasgow`, and `statement_engine` keys remain backward compatible.
- Glasgow and DXL-inspired results are reference-only and cannot determine the final summary.
- `normal` requires every normality-required domain to be complete and technically evaluable.
- ECG output uses “pattern suggestive of acute ischemia/occlusion,” not a standalone acute-MI diagnosis.
- Research-use labeling remains.
- The current `.git` directory has no repository metadata. Do not initialize Git; use test checkpoints instead of commits.

---

## File Structure

Create:

- `feature_extraction/ecgfeat/clinical_rules/__init__.py` — public API.
- `feature_extraction/ecgfeat/clinical_rules/models.py` — immutable result contracts.
- `feature_extraction/ecgfeat/clinical_rules/sources.py` — ruleset/source identifiers.
- `feature_extraction/ecgfeat/clinical_rules/context.py` — normalized facts and quality gates.
- `feature_extraction/ecgfeat/clinical_rules/quality.py` — technical status projection.
- `feature_extraction/ecgfeat/clinical_rules/rhythm.py` — structured rhythm evidence projection.
- `feature_extraction/ecgfeat/clinical_rules/resolver.py` — single final summary resolver.
- `feature_extraction/ecgfeat/clinical_rules/intervals.py` — PR/QT calculations.
- `feature_extraction/ecgfeat/clinical_rules/conduction.py` — QRS/BBB/fascicular rules.
- `feature_extraction/ecgfeat/clinical_rules/ischemia.py` — ST/Q-wave/Sgarbossa/Brugada screening.
- `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py` — named voltage criteria.
- `feature_extraction/ecgfeat/clinical_rules/engine.py` — orchestration and conflicts.
- `feature_extraction/ecgfeat/clinical_rules/validation.py` — diagnostic metric utility.
- `tests/test_clinical_models_resolver.py`
- `tests/test_clinical_intervals.py`
- `tests/test_clinical_conduction.py`
- `tests/test_clinical_ischemia.py`
- `tests/test_clinical_hypertrophy.py`
- `tests/test_clinical_export_consumers.py`
- `tests/test_clinical_validation.py`
- `tests/test_js00059_clinical_regression.py`

Modify:

- `feature_extraction/ecgfeat/api.py` — run unified engine after legacy analysis.
- `feature_extraction/ecgfeat/export.py` — export authoritative result and reference markers.
- `feature_extraction/ecgfeat/models.py` — document metadata contract only; do not break dataclass constructors.
- `demo_feature_extraction.py` — authoritative report section, reference appendices, fingerprint.
- `medgemma_ecg_core.py` — summarize unified result first.
- `app.py` — reuse shared unified summary.
- `batch_extract_ecgfeat.py` — preserve unified payload and version.
- `requirements.txt` — add no new runtime dependency.

---

### Task 1: Core Models, Context, and Resolver Invariants

**Files:**
- Create: `feature_extraction/ecgfeat/clinical_rules/__init__.py`
- Create: `feature_extraction/ecgfeat/clinical_rules/models.py`
- Create: `feature_extraction/ecgfeat/clinical_rules/sources.py`
- Create: `feature_extraction/ecgfeat/clinical_rules/context.py`
- Create: `feature_extraction/ecgfeat/clinical_rules/quality.py`
- Create: `feature_extraction/ecgfeat/clinical_rules/rhythm.py`
- Create: `feature_extraction/ecgfeat/clinical_rules/resolver.py`
- Test: `tests/test_clinical_models_resolver.py`

**Interfaces:**
- Consumes: `ECGFeatures`, `LeadQuality`, `RepresentativeLeadFeatures`.
- Produces: `RuleEvaluation.to_dict()`, `ClinicalContext`, `ClinicalAnalysis.to_dict()`, and `ClinicalStatementResolver.resolve(evaluations, required_domains, available_domains, reference_interpretations, domains)`.

- [ ] **Step 1: Write failing model and resolver tests**

```python
from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation
from feature_extraction.ecgfeat.clinical_rules.resolver import ClinicalStatementResolver


def test_missing_required_input_cannot_be_matched():
    result = RuleEvaluation(
        rule_id="CLIN-TEST-01",
        domain="intervals",
        status="matched",
        required_inputs=["global.qt_ms"],
        missing_inputs=["global.qt_ms"],
    )
    assert result.status == "unavailable"


def test_incomplete_required_domain_forbids_normal():
    analysis = ClinicalStatementResolver().resolve(
        evaluations=[],
        required_domains={"quality", "rhythm", "conduction", "intervals"},
        available_domains={"quality", "rhythm", "conduction"},
    )
    assert analysis.overall_status == "incomplete"
    assert "intervals" in analysis.unavailable_domains


def test_reference_conflict_never_changes_final_status():
    abnormal = RuleEvaluation(
        rule_id="CLIN-TEST-02", domain="intervals", status="matched",
        statement_code="prolonged_qt", severity="abnormal",
    )
    analysis = ClinicalStatementResolver().resolve(
        evaluations=[abnormal],
        required_domains={"intervals"}, available_domains={"intervals"},
        reference_interpretations={"glasgow": {"summary": "normal"}},
    )
    assert analysis.overall_status == "abnormal"
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_models_resolver.py`

Expected: FAIL with `ModuleNotFoundError: feature_extraction.ecgfeat.clinical_rules`.

- [ ] **Step 3: Implement model normalization and source constants**

```python
# models.py
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

VALID_STATUSES = {"matched", "not_matched", "unavailable", "not_applicable", "suppressed"}

@dataclass
class RuleEvaluation:
    rule_id: str
    domain: str
    status: str
    statement_code: Optional[str] = None
    statement: Optional[str] = None
    severity: str = "normal"
    confidence: Optional[str] = None
    required_inputs: List[str] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    suppressed_by: List[str] = field(default_factory=list)
    source: Dict[str, str] = field(default_factory=dict)
    normality_required: bool = True

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid rule status: {self.status}")
        if self.missing_inputs:
            self.status = "unavailable"
        if self.suppressed_by and self.status == "matched":
            self.status = "suppressed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ClinicalAnalysis:
    schema_version: str
    ruleset_version: str
    overall_status: str
    summary: Dict[str, Any]
    final_statements: List[Dict[str, Any]]
    borderline_statements: List[Dict[str, Any]]
    suppressed_statements: List[Dict[str, Any]]
    unavailable_domains: List[str]
    conflicts: List[Dict[str, Any]]
    domains: Dict[str, Any]
    reference_interpretations: Dict[str, Any]
    generated_at: str
    artifact_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

```python
# sources.py
RULESET_VERSION = "2026.07.1"
SCHEMA_VERSION = "clinical_rules.v1"
SOURCE_AHA_CONDUCTION_2009 = {
    "authority": "AHA/ACCF/HRS", "document": "ECG Standardization Part III",
    "section": "Intraventricular conduction disturbances", "version": "2009",
}
SOURCE_AHA_QT_2009 = {
    "authority": "AHA/ACCF/HRS", "document": "ECG Standardization Part IV",
    "section": "QT interval", "version": "2009",
}
SOURCE_UDMI_2018 = {
    "authority": "ESC/ACC/AHA/WHF", "document": "Fourth Universal Definition of MI",
    "section": "ECG manifestations", "version": "2018",
}
```

- [ ] **Step 4: Implement normalized context helpers**

```python
# context.py
from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional

from ..models import ECGFeatures

LIMB_LEADS = frozenset({"I", "II", "III", "aVR", "aVL", "aVF"})
PRECORDIAL_LEADS = frozenset({"V1", "V2", "V3", "V4", "V5", "V6"})

def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None

@dataclass(frozen=True)
class ClinicalContext:
    features: Any
    age_years: Optional[float]
    sex: str
    excluded_leads: frozenset[str]
    precordial_reversal: bool
    limb_reversal: bool

    def global_value(self, name: str) -> Optional[float]:
        return _finite(getattr(self.features.global_features, name, None))

    def lead_available(self, lead: str, reliability: str) -> bool:
        quality = self.features.quality.get(lead)
        return bool(
            lead not in self.excluded_leads
            and quality is not None
            and getattr(quality, reliability, False)
        )

    def lead_value(self, lead: str, name: str, reliability: str) -> Optional[float]:
        if not self.lead_available(lead, reliability):
            return None
        representative = self.features.representative_leads.get(lead)
        params = getattr(representative, "params", {}) if representative is not None else {}
        return _finite(params.get(name))

def build_context(features: ECGFeatures) -> ClinicalContext:
    meta = features.metadata.get("patient_meta")
    age = _finite(getattr(meta, "age", None))
    sex = str(getattr(meta, "sex", "unknown") or "unknown").strip().lower()
    interpretation = features.interpretation
    precordial = bool(getattr(interpretation, "precordial_reversal_suspected", False))
    limb = bool(getattr(interpretation, "limb_reversal_suspected", None))
    excluded = set()
    if precordial:
        excluded.update(PRECORDIAL_LEADS)
    if limb:
        excluded.update(LIMB_LEADS)
    return ClinicalContext(
        features=features,
        age_years=age,
        sex=sex,
        excluded_leads=frozenset(excluded),
        precordial_reversal=precordial,
        limb_reversal=limb,
    )
```

Implement the methods using finite-float conversion, `features.quality`,
`features.representative_leads[lead].params`, and existing lead-reversal metadata.
Precordial reversal adds V1-V6 to `excluded_leads`; limb reversal adds I, II,
III, aVR, aVL, aVF.

- [ ] **Step 5: Implement quality and rhythm projections**

```python
# quality.py
from .models import RuleEvaluation

def evaluate_quality(context):
    record_quality = context.features.metadata.get("record_quality", {})
    grade = record_quality.get("record_grade") if isinstance(record_quality, dict) else None
    missing = [] if grade is not None else ["metadata.record_quality.record_grade"]
    status = "matched" if grade not in {None, "Q0"} else "not_matched"
    return [RuleEvaluation(
        rule_id="CLIN-QUALITY-01",
        domain="quality",
        status=status,
        statement_code="technically_limited" if status == "matched" else None,
        statement="Technically limited ECG" if status == "matched" else None,
        severity="technical" if status == "matched" else "normal",
        required_inputs=["metadata.record_quality.record_grade"],
        missing_inputs=missing,
        evidence={"record_grade": grade, "excluded_leads": sorted(context.excluded_leads)},
    )]
```

```python
# rhythm.py
from .models import RuleEvaluation

def project_rhythm_evidence(context):
    analysis = context.features.metadata.get("rhythm_analysis", {})
    af = analysis.get("af_afl_summary", {}) if isinstance(analysis, dict) else {}
    validated = bool(
        isinstance(analysis, dict)
        and isinstance(analysis.get("atrial_residual"), dict)
        and analysis["atrial_residual"].get("validated_qrst_subtraction")
    )
    evaluations = []
    if af.get("probable_af"):
        evaluations.append(RuleEvaluation(
            rule_id="CLIN-RHYTHM-AF-01", domain="rhythm", status="matched",
            statement_code="atrial_fibrillation_pattern",
            statement="ECG pattern consistent with atrial fibrillation",
            severity="abnormal",
            missing_inputs=[] if validated else ["rhythm.validated_qrst_subtraction"],
            evidence=dict(af),
        ))
    else:
        evaluations.append(RuleEvaluation(
            rule_id="CLIN-RHYTHM-AF-01", domain="rhythm",
            status="not_matched" if validated else "unavailable",
            missing_inputs=[] if validated else ["rhythm.validated_qrst_subtraction"],
            evidence=dict(af),
        ))
    return evaluations
```

Add tests proving missing QRST validation makes AF evaluation unavailable and a
technical quality match causes `technically_limited`, not `abnormal` or
`normal`.

- [ ] **Step 6: Implement resolver status rules**

```python
# resolver.py
from datetime import datetime, timezone

from .models import ClinicalAnalysis
from .sources import RULESET_VERSION, SCHEMA_VERSION

class ClinicalStatementResolver:
    def resolve(self, evaluations, required_domains, available_domains,
                reference_interpretations=None, domains=None):
        unavailable_domains = sorted(set(required_domains) - set(available_domains))
        final = [e.to_dict() for e in evaluations if e.status == "matched"]
        borderline = [x for x in final if x["severity"] == "borderline"]
        abnormal = [x for x in final if x["severity"] not in {"normal", "borderline"}]
        technical = [x for x in final if x["severity"] == "technical"]
        if technical:
            overall = "technically_limited"
        elif unavailable_domains:
            overall = "incomplete"
        elif abnormal:
            overall = "abnormal"
        elif borderline:
            overall = "borderline"
        else:
            overall = "normal"
        suppressed = [e.to_dict() for e in evaluations if e.status == "suppressed"]
        return ClinicalAnalysis(
            schema_version=SCHEMA_VERSION,
            ruleset_version=RULESET_VERSION,
            overall_status=overall,
            summary={"status": overall, "authoritative": True},
            final_statements=sorted(final, key=lambda x: (x["domain"], x["rule_id"])),
            borderline_statements=sorted(borderline, key=lambda x: (x["domain"], x["rule_id"])),
            suppressed_statements=sorted(suppressed, key=lambda x: (x["domain"], x["rule_id"])),
            unavailable_domains=unavailable_domains,
            conflicts=[],
            domains=dict(domains or {}),
            reference_interpretations=dict(reference_interpretations or {}),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
```

Use UTC ISO-8601 timestamps and deterministic statement ordering by
`(domain, rule_id)`.

- [ ] **Step 7: Run core tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_models_resolver.py`

Expected: PASS.

- [ ] **Step 8: Verification checkpoint**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_glasgow_engine.py tests/test_statement_engine.py tests/test_clinical_models_resolver.py`

Expected: all tests pass; record the output in the implementation log.

---

### Task 2: QT/PR Rules with Formula-Specific Evidence

**Files:**
- Create: `feature_extraction/ecgfeat/clinical_rules/intervals.py`
- Test: `tests/test_clinical_intervals.py`

**Interfaces:**
- Consumes: `ClinicalContext`.
- Produces: `evaluate_intervals(context) -> list[RuleEvaluation]` and `qtc_values(qt_ms, hr_bpm) -> dict[str, float]`.

- [ ] **Step 1: Write failing QT and PR tests**

```python
def test_js00059_rate_does_not_use_bazett_as_final():
    values = qtc_values(qt_ms=376.5, hr_bpm=102.040816)
    result = classify_adult_qt(values, sex="female", reliable=True)
    assert values["bazett_ms"] > 480
    assert values["hodges_ms"] < 460
    assert result.statement_code is None
    assert result.evidence["primary_formula"] == "hodges"

def test_missing_qt_is_unavailable():
    result = evaluate_qt_values(None, 70.0, sex="male", reliable=True)
    assert result.status == "unavailable"
    assert "global.qt_ms" in result.missing_inputs

def test_adult_first_degree_av_delay_boundary():
    assert classify_adult_pr(200.0).status == "not_matched"
    assert classify_adult_pr(201.0).statement_code == "first_degree_av_delay"

def test_pediatric_pr_without_supported_table_is_unavailable():
    assert classify_pediatric_pr(180.0, age_years=8).status == "unavailable"
```

- [ ] **Step 2: Run tests and verify missing symbols**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_intervals.py`

Expected: FAIL on imports.

- [ ] **Step 3: Implement QT calculations and adult classification**

Use exact formulas:

```python
rr_s = 60.0 / heart_rate
bazett = qt_ms / sqrt(rr_s)
fridericia = qt_ms / (rr_s ** (1 / 3))
hodges = qt_ms + 1.75 * (heart_rate - 60.0)
framingham = qt_ms + 154.0 * (1.0 - rr_s)
```

Use Hodges as primary. Adult prolonged limits are `>=460 ms` for women and
`>450 ms` for men; short QT is `<=390 ms`. Add `manual_review_required=True`
for any matched abnormal QT rule. If QT reliability is not `reliable` or
`rescued`, return unavailable. If QRS is `>=120 ms` or RR is markedly variable,
return unavailable with the guard reason.

- [ ] **Step 4: Implement PR rules**

Adult: short `<120 ms`, first-degree AV delay `>200 ms`, otherwise normal.
Pediatric: unavailable with missing capability
`pediatric_pr_public_reference_table` until the table exists.

- [ ] **Step 5: Run interval tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_intervals.py tests/test_glasgow_rate_intervals.py`

Expected: PASS.

---

### Task 3: Strict Conduction and Fascicular Rules

**Files:**
- Create: `feature_extraction/ecgfeat/clinical_rules/conduction.py`
- Test: `tests/test_clinical_conduction.py`

**Interfaces:**
- Consumes: `ClinicalContext` and global QRS axis/duration.
- Produces: `evaluate_conduction(context) -> list[RuleEvaluation]`.

- [ ] **Step 1: Write failing morphology tests**

```python
def test_axis_alone_never_diagnoses_lafb(context_factory):
    context = context_factory(axis=-60, qrs_ms=100, leads={})
    result = by_code(evaluate_conduction(context), "lafb_pattern")
    assert result.status == "unavailable"

def test_lafb_requires_qr_avl_and_rs_inferior(context_factory):
    context = context_factory(
        axis=-60, qrs_ms=100,
        leads={"aVL": {"q_amp_mv": -0.05, "r_amp_mv": 0.7},
               "II": {"r_amp_mv": 0.1, "s_amp_mv": -0.5},
               "III": {"r_amp_mv": 0.1, "s_amp_mv": -0.6},
               "aVF": {"r_amp_mv": 0.1, "s_amp_mv": -0.5}},
    )
    assert by_code(evaluate_conduction(context), "lafb_pattern").status == "matched"

def test_qrs_108_without_bbb_morphology_is_not_ivcd(context_factory):
    context = context_factory(axis=0, qrs_ms=108, leads=normal_qrs_leads())
    assert not matched_code(evaluate_conduction(context), "nonspecific_ivcd")

def test_rbbb_requires_terminal_duration(context_factory):
    context = context_factory(qrs_ms=130, leads=rbbb_amplitudes_without_durations())
    assert by_code(evaluate_conduction(context), "rbbb_pattern").status == "unavailable"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_conduction.py`

Expected: FAIL on imports.

- [ ] **Step 3: Implement conduction rules**

Implement:

- nonspecific IVCD: adult QRS `>110 ms` with neither RBBB nor LBBB morphology;
- complete BBB: QRS `>=120 ms` plus required terminal morphology and duration;
- incomplete RBBB: QRS `110-119 ms` plus RBBB morphology;
- LAFB: axis `-45..-90`, aVL qR, and dominant S in II plus inferior rS support;
- LPFB: axis `90..180`, rS in I/aVL, qR in III/aVF, and confounder evidence;
- missing required component/duration: unavailable.

Each rule includes the AHA conduction source and actual operands.

- [ ] **Step 4: Run conduction tests and legacy rule tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_conduction.py tests/test_interpret_rule_fixes.py`

Expected: PASS; legacy behavior remains reference-only.

---

### Task 4: ST, Q-Wave, Sgarbossa, Posterior, and Brugada Screening

**Files:**
- Create: `feature_extraction/ecgfeat/clinical_rules/ischemia.py`
- Test: `tests/test_clinical_ischemia.py`

**Interfaces:**
- Consumes: `ClinicalContext`, conduction results, pacing context.
- Produces: `evaluate_ischemia(context, conduction) -> list[RuleEvaluation]`.

- [ ] **Step 1: Write failing ST and Q-wave tests**

```python
def test_noncontiguous_st_elevation_does_not_match(context_factory):
    context = context_factory(sex="female", age=39,
        st={"V2": 0.16, "V5": 0.11})
    assert not matched_code(evaluate_ischemia(context, []), "acute_occlusion_pattern")

def test_female_v2_v3_threshold_is_015_mv(context_factory):
    context = context_factory(sex="female", age=39,
        st={"V2": 0.15, "V3": 0.15})
    assert matched_code(evaluate_ischemia(context, []), "acute_occlusion_pattern")

def test_missing_q_duration_cannot_match_old_mi(context_factory):
    context = context_factory(q={"II": (-0.10, None, 0.30),
                                 "III": (-0.10, None, 0.25),
                                 "aVF": (-0.10, None, 0.20)})
    q_result = by_code(evaluate_ischemia(context, []), "prior_infarct_q_wave_pattern")
    assert q_result.status == "unavailable"

def test_precordial_reversal_suppresses_posterior_pattern(context_factory):
    context = context_factory(precordial_reversal=True,
        st={"V1": -0.12, "V2": -0.12, "V3": -0.12})
    assert by_code(evaluate_ischemia(context, []), "posterior_ischemia_pattern").status == "suppressed"

def test_lbbb_uses_sgarbossa_instead_of_blanket_suppression(context_factory):
    context = context_factory(qrs_ms=150, st={"I": 0.12}, qrs_polarity={"I": 1})
    result = by_code(evaluate_sgarbossa(context, paced=False), "sgarbossa_positive")
    assert result.status == "matched"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_ischemia.py`

Expected: FAIL on imports.

- [ ] **Step 3: Implement contiguous ST thresholds**

Use contiguous groups and inclusive thresholds:

- all leads except V2-V3: `>=0.10 mV` in two contiguous leads;
- V2-V3 women: `>=0.15 mV`;
- V2-V3 men age `>=40`: `>=0.20 mV`;
- V2-V3 men age `<40`: `>=0.25 mV`.

Do not use V2 and V5 as a contiguous pair. Record reciprocal changes as support,
not a requirement.

- [ ] **Step 4: Implement strict Q-wave and posterior rules**

Require measurable Q duration and two qualifying leads in an anatomical group.
Return unavailable when amplitude is present but duration is absent. Posterior
V1-V3 ST depression is `posterior_ischemia_pattern`, severity `abnormal`, with
`recommended_leads=["V7", "V8", "V9"]`, never an acute-MI diagnosis.

- [ ] **Step 5: Implement Sgarbossa and Brugada screening**

Original Sgarbossa scoring:

- concordant ST elevation `>=0.1 mV`: 5 points;
- V1-V3 ST depression `>=0.1 mV`: 3 points;
- excessively discordant ST elevation `>=0.5 mV`: 2 points;
- score `>=3`: positive screening pattern.

Brugada type-1 screening requires V1 or V2 J/ST elevation `>=0.2 mV`, coved or
descending ST morphology, and a negative T wave. Missing ST morphology returns
unavailable. The statement text includes `screening pattern; syndrome requires clinical confirmation`.

- [ ] **Step 6: Run ischemia tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_ischemia.py tests/test_mi_qwave.py tests/test_st_j_raw_remeasurement.py`

Expected: PASS.

---

### Task 5: Hypertrophy Labels and Pediatric Availability

**Files:**
- Create: `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py`
- Test: `tests/test_clinical_hypertrophy.py`

**Interfaces:**
- Consumes: `ClinicalContext`, conduction results.
- Produces: `evaluate_hypertrophy(context, conduction) -> list[RuleEvaluation]`.

- [ ] **Step 1: Write failing terminology and availability tests**

```python
def test_cornell_reports_voltage_criteria_not_anatomic_lvh(context_factory):
    context = context_factory(sex="female", age=50,
        leads={"aVL": {"r_amp_mv": 1.2}, "V3": {"s_amp_mv": -1.2}})
    result = by_code(evaluate_hypertrophy(context, []), "lvh_voltage_criteria")
    assert result.status == "matched"
    assert "ECG voltage criteria" in result.statement

def test_pediatric_lvh_without_percentile_table_is_unavailable(context_factory):
    context = context_factory(age=8, sex="male")
    result = by_code(evaluate_hypertrophy(context, []), "pediatric_lvh_voltage")
    assert result.status == "unavailable"
    assert "pediatric_lvh_percentile_table" in result.missing_inputs

def test_missing_rprime_duration_cannot_match_rvh(context_factory):
    context = context_factory(leads={"V1": {"r_prime_amp_mv": 0.5,
                                             "r_prime_duration_ms": None}})
    assert by_code(evaluate_hypertrophy(context, []), "rvh_pattern").status == "unavailable"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_hypertrophy.py`

Expected: FAIL on imports.

- [ ] **Step 3: Implement named adult voltage criteria**

Implement Cornell voltage, Cornell product, Sokolow-Lyon, and R-aVL evidence
using existing mV and QRS-ms fields. Statement text must be `Meets ECG voltage
criteria for LVH`; the output must not say confirmed anatomical LVH.

- [ ] **Step 4: Implement strict RVH/atrial availability and pediatric policy**

RVH and atrial morphology require their component durations. Pediatric LVH/RVH
percentile-dependent evaluations return unavailable with explicit capability
keys. Supported pediatric interval/axis rules remain independent.

- [ ] **Step 5: Run hypertrophy tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_hypertrophy.py tests/test_pediatric_rules.py`

Expected: PASS.

---

### Task 6: Engine Orchestration, Conflicts, and API Integration

**Files:**
- Create: `feature_extraction/ecgfeat/clinical_rules/engine.py`
- Modify: `feature_extraction/ecgfeat/api.py:1963-1975`
- Test: `tests/test_clinical_export_consumers.py`

**Interfaces:**
- Consumes: all clinical evaluators and legacy metadata.
- Produces: `analyze_clinical(features, strict=False) -> ClinicalAnalysis` stored in `features.metadata["clinical_interpretation"]`.

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_api_populates_authoritative_clinical_analysis(extracted_features):
    payload = extracted_features.metadata["clinical_interpretation"]
    assert payload["schema_version"] == "clinical_rules.v1"
    assert payload["reference_interpretations"]["glasgow"]["reference_only"] is True
    assert payload["reference_interpretations"]["dxl"]["reference_only"] is True

def test_glasgow_normal_conflict_is_recorded_not_adopted(feature_factory):
    features = feature_factory(qt_ms=500, hr_bpm=60)
    features.metadata["glasgow_analysis"] = {
        "statement_resolution": {"summary_code": {"code": 1, "label": "Normal ECG"}}}
    result = analyze_clinical(features)
    assert result.overall_status == "abnormal"
    assert any(c["reference"] == "glasgow" for c in result.conflicts)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py`

Expected: FAIL because engine/API integration is absent.

- [ ] **Step 3: Implement engine orchestration**

```python
def analyze_clinical(features, strict=False):
    context = build_context(features)
    evaluations = evaluate_quality(context)
    evaluations += evaluate_intervals(context)
    conduction = evaluate_conduction(context)
    evaluations += conduction
    evaluations += evaluate_hypertrophy(context, conduction)
    evaluations += evaluate_ischemia(context, conduction)
    evaluations += project_rhythm_evidence(context)
    required = {
        "quality", "rhythm", "conduction", "intervals", "hypertrophy",
        "ischemia_infarction", "high_risk_patterns",
    }
    available = set()
    for domain in required:
        required_evaluations = [
            evaluation for evaluation in evaluations
            if evaluation.domain == domain and evaluation.normality_required
        ]
        if required_evaluations and all(
            evaluation.status != "unavailable"
            for evaluation in required_evaluations
        ):
            available.add(domain)
    references = {
        "glasgow": {
            "reference_only": True,
            "analysis": features.metadata.get("glasgow_analysis", {}),
        },
        "dxl": {
            "reference_only": True,
            "analysis": asdict(features.interpretation) if features.interpretation else {},
        },
    }
    result = ClinicalStatementResolver().resolve(
        evaluations=evaluations,
        required_domains=required,
        available_domains=available,
        reference_interpretations=references,
        domains=group_by_domain(evaluations),
    )
    result.conflicts = detect_reference_conflicts(result, references)
    return result
```

Implement `group_by_domain` as a dictionary from domain name to serialized,
rule-id-sorted evaluations. Implement `detect_reference_conflicts` to compare
only explicit reference summary labels with the authoritative `overall_status`;
for example, a reference label containing `normal` conflicts with authoritative
`abnormal`, `borderline`, `technically_limited`, or `incomplete`. Each conflict
has `reference`, `reference_summary`, `authoritative_status`, and `reason`.

Required domains are quality, rhythm, conduction, intervals, hypertrophy,
repolarization/ischemia, and high-risk-pattern screening. Domain availability
comes from rule statuses, not merely function return.

- [ ] **Step 4: Add API call after legacy analyses**

```python
ecg_features.interpretation = interpret(ecg_features)
ecg_features.metadata["glasgow_analysis"] = analyze_glasgow(ecg_features).to_dict()
ecg_features.metadata["clinical_interpretation"] = analyze_clinical(ecg_features).to_dict()
```

- [ ] **Step 5: Run API and engine tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_ecgfeat_pipeline.py`

Expected: PASS.

---

### Task 7: Export Compatibility, Fingerprint, and Consumer Migration

**Files:**
- Modify: `feature_extraction/ecgfeat/export.py:1722-1804`
- Modify: `medgemma_ecg_core.py:241-360`
- Modify: `app.py:417-494`
- Modify: `batch_extract_ecgfeat.py:200-260`
- Test: `tests/test_clinical_export_consumers.py`

**Interfaces:**
- Consumes: metadata clinical analysis.
- Produces: top-level `clinical_interpretation`, legacy reference markers, unified prompt summary, deterministic artifact fingerprint.

- [ ] **Step 1: Extend failing export/consumer tests**

```python
def test_full_export_preserves_legacy_and_adds_clinical(features):
    payload = to_dict(features)
    assert "interpretation" in payload
    assert "glasgow" in payload
    assert payload["clinical_interpretation"]["schema_version"] == "clinical_rules.v1"
    assert payload["reference_metadata"]["interpretation"]["reference_only"] is True

def test_medgemma_summary_leads_with_unified_result(payload):
    text = summarize_current_features(payload)
    assert "Unified clinical summary" in text
    assert text.index("Unified clinical summary") < text.index("Glasgow reference")

def test_fingerprint_ignores_generated_at(analysis):
    a = clinical_fingerprint({**analysis, "generated_at": "2026-01-01T00:00:00Z"})
    b = clinical_fingerprint({**analysis, "generated_at": "2026-02-01T00:00:00Z"})
    assert a == b
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py`

Expected: FAIL on missing top-level fields/helpers.

- [ ] **Step 3: Implement deterministic export fingerprint**

Serialize the clinical analysis after removing `generated_at` and
`artifact_fingerprint` using sorted compact JSON, then SHA-256 it. Prefix output
with `sha256:`. Add `reference_metadata` without changing legacy field types.

- [ ] **Step 4: Migrate MedGemma and App summaries**

Add shared formatting in `medgemma_ecg_core.py`:

```python
def summarize_clinical_interpretation(clinical):
    statements = [
        item.get("statement") or item.get("statement_code")
        for item in clinical.get("final_statements", [])
    ]
    conflicts = [
        f"{item.get('reference')}: {item.get('reason')}"
        for item in clinical.get("conflicts", [])
    ]
    return [
        f"- Unified clinical summary: {clinical.get('overall_status', 'unavailable')}",
        f"- Unified final statements: {_fmt_list(statements)}",
        f"- Unified unavailable domains: {_fmt_list(clinical.get('unavailable_domains'))}",
        f"- Reference conflicts: {_fmt_list(conflicts)}",
    ]
```

Use it before Glasgow and legacy DXL lines in both MedGemma and App.

- [ ] **Step 5: Preserve batch payload**

Ensure batch serialization uses `to_dict(result)` and does not filter the new
top-level fields. Add ruleset version to its summary rows when a summary is
written.

- [ ] **Step 6: Run consumer tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_export_contract.py tests/test_medgemma_ecg_core.py tests/test_batch_extract_ecgfeat.py`

Expected: all runnable tests pass; environment-only missing optional modules are
reported separately, not counted as rule failures.

---

### Task 8: Text/Image Report Authority and Stale-Artifact Detection

**Files:**
- Modify: `demo_feature_extraction.py:560-713`
- Modify: `demo_feature_extraction.py:950-1352`
- Test: `tests/test_clinical_export_consumers.py`
- Test: `tests/test_ecg_report_sheet.py`

**Interfaces:**
- Consumes: `metadata["clinical_interpretation"]` and fingerprint.
- Produces: unified report section, reference appendix, `STALE ARTIFACT` warning.

- [ ] **Step 1: Write failing report tests**

```python
def test_report_shows_unified_summary_before_reference_algorithms(report_model):
    lines = report_model["interpretation_lines"]
    assert lines[0].startswith("Unified clinical summary")
    assert next(i for i, x in enumerate(lines) if "Glasgow reference" in x) > 0

def test_stale_fingerprint_is_visible(report_text):
    assert "STALE ARTIFACT" in render_with_mismatched_fingerprint(report_text)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_ecg_report_sheet.py`

Expected: FAIL on ordering/warning assertions.

- [ ] **Step 3: Render unified section and reference appendix**

Report order:

1. patient and measurements;
2. `UNIFIED CLINICAL INTERPRETATION`;
3. signal quality;
4. reference algorithms (`GLASGOW REFERENCE`, `DXL-INSPIRED REFERENCE`);
5. rule audit and limitations.

Remove the standalone Glasgow “Normal ECG” headline from the authoritative
section. Keep it verbatim only inside the labelled reference appendix.

- [ ] **Step 4: Add fingerprint check**

Compare the clinical fingerprint stored in the payload with the newly computed
fingerprint. On mismatch, prepend `*** STALE ARTIFACT — REGENERATE REPORT ***`.

- [ ] **Step 5: Run report tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_ecg_report_sheet.py tests/test_glasgow_report_rendering.py`

Expected: PASS.

---

### Task 9: Clinical Validation Metric Utility

**Files:**
- Create: `feature_extraction/ecgfeat/clinical_rules/validation.py`
- Test: `tests/test_clinical_validation.py`

**Interfaces:**
- Consumes: iterable of `{truth: bool, predicted: bool|None, age, sex, quality}` rows.
- Produces: `binary_metrics(rows)`, `stratified_metrics(rows, keys)`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_binary_metrics_exclude_unavailable_predictions():
    rows = [
        {"truth": True, "predicted": True},
        {"truth": True, "predicted": False},
        {"truth": False, "predicted": False},
        {"truth": False, "predicted": None},
    ]
    result = binary_metrics(rows)
    assert result["tp"] == 1 and result["fn"] == 1 and result["tn"] == 1
    assert result["unavailable"] == 1
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 1.0

def test_stratification_keeps_sex_and_quality_groups():
    result = stratified_metrics(sample_rows(), keys=("sex", "quality"))
    assert ("female", "reliable") in result
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_validation.py`

Expected: FAIL on imports.

- [ ] **Step 3: Implement zero-safe metrics**

Return counts, unavailable rate, sensitivity, specificity, PPV, NPV, F1 and
coverage. A denominator of zero returns `None`, never zero. Stratification uses
exact tuple keys and calls the same binary function.

- [ ] **Step 4: Run validation tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_validation.py`

Expected: PASS.

---

### Task 10: JS00059 End-to-End Regression and Artifact Regeneration

**Files:**
- Create: `tests/test_js00059_clinical_regression.py`
- Regenerate: `JS00059_features.json`
- Regenerate: `JS00059_report.txt`
- Regenerate: `JS00059_ecg.png`
- Regenerate: `JS00059_ecg_annotated.png`
- Regenerate: `JS00059_ecg_report.png`

**Interfaces:**
- Consumes: `dataset/JS00059.mat`, `dataset/JS00059.hea`, full pipeline.
- Produces: current unified artifacts with one ruleset version/fingerprint.

- [ ] **Step 1: Write failing regression test**

```python
def test_js00059_unified_result_has_no_stale_contradictions(js00059_features):
    clinical = js00059_features.metadata["clinical_interpretation"]
    qt = clinical["domains"]["intervals"]["qt"]
    assert qt["evidence"]["primary_formula"] == "hodges"
    assert qt["statement_code"] is None
    assert "rvh_pattern" not in final_codes(clinical)
    assert "posterior_ischemia_pattern" not in final_codes(clinical)
    assert "prior_infarct_q_wave_pattern" not in final_codes(clinical)
    assert clinical["overall_status"] != "normal"

def test_js00059_artifacts_share_fingerprint():
    payload = json.loads(Path("JS00059_features.json").read_text())
    fingerprint = payload["clinical_interpretation"]["artifact_fingerprint"]
    report = Path("JS00059_report.txt").read_text()
    assert fingerprint in report
    assert "STALE ARTIFACT" not in report
```

- [ ] **Step 2: Run regression test and verify failure on old artifacts**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_js00059_clinical_regression.py`

Expected: FAIL because old artifacts lack `clinical_interpretation` and a
fingerprint.

- [ ] **Step 3: Run the current demo generator**

Run: `PYTHONPATH=. .venv/bin/python demo_feature_extraction.py JS00059`

Expected: exit 0 and all five artifacts receive current timestamps.

- [ ] **Step 4: Run regression and focused clinical suite**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_intervals.py \
  tests/test_clinical_conduction.py \
  tests/test_clinical_ischemia.py \
  tests/test_clinical_hypertrophy.py \
  tests/test_clinical_export_consumers.py \
  tests/test_clinical_validation.py \
  tests/test_js00059_clinical_regression.py
```

Expected: PASS.

---

### Task 11: Full Verification and Documentation Sync

**Files:**
- Modify: `docs/philips_adult_morphology_feature_schema_checklist.md`
- Modify: `docs/philips_child_morphology_feature_schema_checklist.md`
- Modify: `docs/philips_rhythm_coverage_matrix.md`

**Interfaces:**
- Consumes: completed implementation and test output.
- Produces: accurate documentation distinguishing unified public rules from reference vendor approximations.

- [ ] **Step 1: Update status documentation**

Document the authoritative layer, legacy reference status, deliberately
unavailable pediatric rules, new Sgarbossa/Brugada screening, and the exact test
commands. Remove stale claims that the statement engine or pacing-first flow is
missing when current code implements it.

- [ ] **Step 2: Run focused verification**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_*.py tests/test_js00059_clinical_regression.py`

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `PYTHONPATH=. .venv/bin/pytest -q`

Expected: no new clinical or legacy assertion failures. Record pre-existing
environment failures separately: hard-coded absent `.venv_report_regen_20260625`,
missing optional `wfdb`, `gradio`, or `snapshot_regression` must not be
misreported as clinical-rule failures.

- [ ] **Step 4: Inspect generated outputs**

Run:

```bash
rg -n "UNIFIED CLINICAL|reference-only|ruleset|fingerprint|STALE ARTIFACT|Normal ECG" \
  JS00059_report.txt JS00059_features.json
```

Expected: unified summary and matching version/fingerprint are present; `Normal
ECG except for rate` appears only in the labelled Glasgow reference appendix;
`STALE ARTIFACT` is absent.

- [ ] **Step 5: Final verification checkpoint**

Summarize changed files, focused/full test counts, pre-existing environment
failures, JS00059 unified results, and remaining explicitly unavailable clinical
domains. Do not claim clinical validation or proprietary algorithm equivalence.
