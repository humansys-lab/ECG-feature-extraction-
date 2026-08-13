<!-- i18n-nav -->
[中文](2026-07-12-unified-clinical-ecg-rules.md) | [English](2026-07-12-unified-clinical-ecg-rules.en.md) | [日本語](2026-07-12-unified-clinical-ecg-rules.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="unified-clinical-ecg-rules-implementation-plan"></a>
# 統一臨床 ECG ルール実装計画

> **エージェント ワーカーの場合:** 必須サブスキル: superpowers:subagent-driven-development (推奨) または superpowers:executing-plan を使用して、この計画をタスクごとに実装します。ステップでは、追跡にチェックボックス (`- [ ]`) 構文を使用します。

**目標:** Glasgow、DXL にインスピレーションを受けたアプリ、MedGemma、バッチ、およびレポートの互換性を維持しながら、権威的で安全性を重視した臨床 ECG 解釈レイヤーを 1 つ追加します。

**アーキテクチャ:** 既存の ECG 測定パイプラインをそのまま維持し、`ECGFeatures` を正規化された `ClinicalContext` に適応させます。独立した公開ガイドライン ルール モジュールが構造化された評価を生成し、`ClinicalStatementResolver` のみが最終的な概要を生成します。従来の解釈は参照専用のままです。

**技術スタック:** Python 3.12、データクラス、NumPy、pytest/unittest、既存の `ecgfeat` パッケージ、およびレポート/エクスポート コード。

<a id="global-constraints"></a>
## グローバル制約

- 公的な AHA/ACC/HRS、ESC、および心筋梗塞の世界的定義基準が最終声明として権威を持っています。
- 必要な証拠が不足している場合は、`unavailable` が返されます。肯定的なルールや通常のルールを決して満たしません。
- 既存の `interpretation`、`glasgow`、および `statement_engine` キーは下位互換性を維持します。
- グラスゴーおよび DXL に影響を受けた結果は参考のみであり、最終的な概要を決定することはできません。
- `normal` では、正規性が必要なすべてのドメインが完全で技術的に評価可能であることが必要です。
- ECG 出力は、独立した急性 MI 診断ではなく、「急性虚血/閉塞を示唆するパターン」を使用します。
- 研究用途のラベルは残ります。
- 現在の `.git` ディレクトリにはリポジトリ メタデータがありません。 Git を初期化しないでください。コミットの代わりにテスト チェックポイントを使用します。

---

<a id="file-structure"></a>
## ファイル構造

作成:

- `feature_extraction/ecgfeat/clinical_rules/__init__.py` — パブリック API。
- `feature_extraction/ecgfeat/clinical_rules/models.py` — 不変の結果コントラクト。
- `feature_extraction/ecgfeat/clinical_rules/sources.py` — ルールセット/ソース識別子。
- `feature_extraction/ecgfeat/clinical_rules/context.py` — 正規化されたファクトと品質ゲート。
- `feature_extraction/ecgfeat/clinical_rules/quality.py` — 技術的ステータスの予測。
- `feature_extraction/ecgfeat/clinical_rules/rhythm.py` — 構造化リズム証拠投影。
- `feature_extraction/ecgfeat/clinical_rules/resolver.py` — 単一の最終サマリー リゾルバー。
- `feature_extraction/ecgfeat/clinical_rules/intervals.py` — PR/QT の計算。
- `feature_extraction/ecgfeat/clinical_rules/conduction.py` — QRS/BBB/筋束ルール。
- `feature_extraction/ecgfeat/clinical_rules/ischemia.py` — ST/Q-wave/Sgarbossa/Brugada の上映。
- `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py` — 名前付き電圧基準。
- `feature_extraction/ecgfeat/clinical_rules/engine.py` — オーケストレーションと競合。
- `feature_extraction/ecgfeat/clinical_rules/validation.py` — 診断メトリック ユーティリティ。
- `tests/test_clinical_models_resolver.py`
- `tests/test_clinical_intervals.py`
- `tests/test_clinical_conduction.py`
- `tests/test_clinical_ischemia.py`
- `tests/test_clinical_hypertrophy.py`
- `tests/test_clinical_export_consumers.py`
- `tests/test_clinical_validation.py`
- `tests/test_js00059_clinical_regression.py`

変更:

- `feature_extraction/ecgfeat/api.py` — 従来の分析後に統合エンジンを実行します。
- `feature_extraction/ecgfeat/export.py` — 信頼できる結果マーカーと参照マーカーをエクスポートします。
- `feature_extraction/ecgfeat/models.py` — ドキュメントのメタデータ コントラクトのみ。データクラスのコンストラクターを壊さないでください。
- `demo_feature_extraction.py` — 権威あるレポートセクション、参考付録、フィンガープリント。
- `medgemma_ecg_core.py` — 最初に統合結果を要約します。
- `app.py` — 共有された統合サマリーを再利用します。
- `batch_extract_ecgfeat.py` — 統合されたペイロードとバージョンを保持します。
- `requirements.txt` — 新しいランタイム依存関係を追加しません。

---

<a id="task-1-core-models-context-and-resolver-invariants"></a>
### タスク 1: コア モデル、コンテキスト、およびリゾルバーの不変条件

**ファイル:**
- 作成: `feature_extraction/ecgfeat/clinical_rules/__init__.py`
- 作成: `feature_extraction/ecgfeat/clinical_rules/models.py`
- 作成: `feature_extraction/ecgfeat/clinical_rules/sources.py`
- 作成: `feature_extraction/ecgfeat/clinical_rules/context.py`
- 作成: `feature_extraction/ecgfeat/clinical_rules/quality.py`
- 作成: `feature_extraction/ecgfeat/clinical_rules/rhythm.py`
- 作成: `feature_extraction/ecgfeat/clinical_rules/resolver.py`
- テスト: `tests/test_clinical_models_resolver.py`

**インターフェース:**
- 消費: `ECGFeatures`、`LeadQuality`、`RepresentativeLeadFeatures`。
- 生成: `RuleEvaluation.to_dict()`、`ClinicalContext`、`ClinicalAnalysis.to_dict()`、および `ClinicalStatementResolver.resolve(evaluations, required_domains, available_domains, reference_interpretations, domains)`。

- [ ] **ステップ 1: 失敗したモデルとリゾルバーのテストを作成する**

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

- [ ] **ステップ 2: テストを実行し、インポートの失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_models_resolver.py`

予期: `ModuleNotFoundError: feature_extraction.ecgfeat.clinical_rules` で失敗します。

- [ ] **ステップ 3: モデルの正規化とソース定数を実装する**

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

- [ ] **ステップ 4: 正規化されたコンテキスト ヘルパーを実装する**

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

有限浮動小数点変換 `features.quality` を使用してメソッドを実装します。
`features.representative_leads[lead].params`、および既存のリード逆転メタデータ。
前胸部反転により、V1 ～ V6 が `excluded_leads` に追加されます。四肢反転は I、II、を追加します。
III、aVR、aVL、aVF。

- [ ] **ステップ 5: 品質とリズムの投影を実装する**

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

QRST 検証が欠落していることを証明するテストを追加すると、AF 評価が利用できなくなり、
技術的な品質が一致すると、`abnormal` ではなく、`technically_limited` が発生します。
`normal`。

- [ ] **ステップ 6: リゾルバー ステータス ルールを実装する**

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

UTC ISO-8601 タイムスタンプと決定的なステートメントの順序付けを使用します。
`(domain, rule_id)`。

- [ ] **ステップ 7: コア テストの実行**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_models_resolver.py`

期待値: 合格。

- [ ] **ステップ 8: 検証チェックポイント**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_glasgow_engine.py tests/test_statement_engine.py tests/test_clinical_models_resolver.py`

期待値: すべてのテストに合格します。出力を実装ログに記録します。

---

<a id="task-2-qtpr-rules-with-formula-specific-evidence"></a>
### タスク 2: フォーミュラ固有の証拠を備えた QT/PR ルール

**ファイル:**
- 作成: `feature_extraction/ecgfeat/clinical_rules/intervals.py`
- テスト: `tests/test_clinical_intervals.py`

**インターフェース:**
- 消費: `ClinicalContext`。
- 生成: `evaluate_intervals(context) -> list[RuleEvaluation]` および `qtc_values(qt_ms, hr_bpm) -> dict[str, float]`。

- [ ] **ステップ 1: 失敗した QT および PR テストを作成する**

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

- [ ] **ステップ 2: テストを実行し、欠落しているシンボルを確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_intervals.py`

予期: インポートで失敗します。

- [ ] **ステップ 3: QT 計算と成人分類の実装**

正確な数式を使用してください。

```python
rr_s = 60.0 / heart_rate
bazett = qt_ms / sqrt(rr_s)
fridericia = qt_ms / (rr_s ** (1 / 3))
hodges = qt_ms + 1.75 * (heart_rate - 60.0)
framingham = qt_ms + 154.0 * (1.0 - rr_s)
```

ホッジスをプライマリとして使用します。成人の長期制限は女性の場合は `>=460 ms` です。
`>450 ms` 男性用。短い QT は `<=390 ms` です。 `manual_review_required=True`を追加
一致した異常な QT ルールの場合。 QT 信頼性が `reliable` でない場合、または
`rescued`、返品不可。 QRS が `>=120 ms` であるか、RR が著しく変動している場合、
ガードの理由により返品できません。

- [ ] **ステップ 4: PR ルールを実装する**

成人：低身長 `<120 ms`、第 1 度房室遅延 `>200 ms`、それ以外は正常。
小児: 機能が不足しているため使用不可
テーブルが存在するまで `pediatric_pr_public_reference_table`。

- [ ] **ステップ 5: 間隔テストの実行**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_intervals.py tests/test_glasgow_rate_intervals.py`

期待値: 合格。

---

<a id="task-3-strict-conduction-and-fascicular-rules"></a>
### タスク 3: 厳密な伝導と線維束のルール

**ファイル:**
- 作成: `feature_extraction/ecgfeat/clinical_rules/conduction.py`
- テスト: `tests/test_clinical_conduction.py`

**インターフェース:**
- 消費: `ClinicalContext` およびグローバル QRS 軸/期間。
- プロデュース: `evaluate_conduction(context) -> list[RuleEvaluation]`。

- [ ] **ステップ 1: 失敗する形態学テストを作成する**

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

- [ ] **ステップ 2: テストを実行して失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_conduction.py`

予期: インポートで失敗します。

- [ ] **ステップ 3: 伝導ルールの実装**

実装:

- 非特異的 IVCD: RBBB も LBBB 形態も持たない成人 QRS `>110 ms`。
- 完全な BBB: QRS `>=120 ms` に必要な末端形態と期間を加えたもの。
- 不完全な RBBB: QRS `110-119 ms` プラス RBBB 形態。
- LAFB: 軸 `-45..-90`、aVL qR、および II のドミナント S と劣った rS サポート。
- LPFB: 軸 `90..180`、I/aVL の rS、III/aVF の qR、および交絡因子の証拠。
- 必要なコンポーネント/期間がありません: 利用できません。

各ルールには、AHA 伝導ソースと実際のオペランドが含まれます。

- [ ] **ステップ 4: 伝導テストとレガシー ルール テストを実行する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_conduction.py tests/test_interpret_rule_fixes.py`

期待値: 合格。従来の動作は参照のみのままです。

---

<a id="task-4-st-q-wave-sgarbossa-posterior-and-brugada-screening"></a>
### タスク 4: ST、Q-Wave、Sgarbossa、Posterior、および Brugada のスクリーニング

**ファイル:**
- 作成: `feature_extraction/ecgfeat/clinical_rules/ischemia.py`
- テスト: `tests/test_clinical_ischemia.py`

**インターフェース:**
- 消費: `ClinicalContext`、伝導結果、ペーシング コンテキスト。
- プロデュース: `evaluate_ischemia(context, conduction) -> list[RuleEvaluation]`。

- [ ] **ステップ 1: 失敗した ST および Q 波テストを作成する**

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

- [ ] **ステップ 2: テストを実行して失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_ischemia.py`

予期: インポートで失敗します。

- [ ] **ステップ 3: 連続した ST しきい値を実装する**

連続したグループと包括的なしきい値を使用します。

- V2-V3 を除くすべてのリード: 2 つの連続するリードの `>=0.10 mV`。
- V2～V3 女性: `>=0.15 mV`;
- V2～V3 男性年齢 `>=40`: `>=0.20 mV`;
- V2-V3 男性年齢 `<40`: `>=0.25 mV`。

V2 と V5 を連続したペアとして使用しないでください。相互の変化をサポートとして記録し、
要件ではありません。

- [ ] **ステップ 4: 厳格な Q 波と事後ルールを実装する**

解剖学的グループ内に測定可能な Q 期間と 2 つの適格なリードが必要です。
振幅は存在するが持続時間が存在しない場合、リターンは利用できません。後部
V1-V3 ST 低下は `posterior_ischemia_pattern`、重症度は `abnormal`、
`recommended_leads=["V7", "V8", "V9"]`、急性心筋梗塞の診断ではありません。

- [ ] **ステップ 5: Sgarbossa と Brugada のスクリーニングを実施する**

Sgarbossa のオリジナルスコア:

- 一致する ST 上昇 `>=0.1 mV`: 5 ポイント;
- V1-V3 ST 落ち込み `>=0.1 mV`: 3 ポイント。
- 過度に不一致な ST 上昇 `>=0.5 mV`: 2 ポイント。
- スコア `>=3`: 陽性スクリーニング パターン。

ブルガダ 1 型スクリーニングには、V1 または V2 J/ST 昇格 `>=0.2 mV`、カバードまたは
下降性 ST 形態、および陰性 T 波。欠落した ST 形態が返される
利用できません。ステートメントのテキストには `screening pattern; syndrome requires clinical confirmation` が含まれています。

- [ ] **ステップ 6: 虚血検査を実行します**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_ischemia.py tests/test_mi_qwave.py tests/test_st_j_raw_remeasurement.py`

期待値: 合格。

---

<a id="task-5-hypertrophy-labels-and-pediatric-availability"></a>
### タスク 5: 肥大のラベルと小児の利用可能性

**ファイル:**
- 作成: `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py`
- テスト: `tests/test_clinical_hypertrophy.py`

**インターフェース:**
- 消費: `ClinicalContext`、伝導結果。
- プロデュース: `evaluate_hypertrophy(context, conduction) -> list[RuleEvaluation]`。

- [ ] **ステップ 1: 失敗する用語と可用性テストを作成する**

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

- [ ] **ステップ 2: テストを実行して失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_hypertrophy.py`

予期: インポートで失敗します。

- [ ] **ステップ 3: 名前付き成人電圧基準を実装します**

Cornell 電圧、Cornell 積、Sokolow-Lyon、および R-aVL の証拠を実装する
既存の mV および QRS-ms フィールドを使用します。ステートメントのテキストは「ECG 電圧に適合」である必要があります
LVH`の基準;出力には、解剖学的 LVH が確認されたと表示されてはなりません。

- [ ] **ステップ 4: 厳格な RVH/心房の利用可能性と小児ポリシーを導入する**

RVH および心房形態には、コンポーネントの持続時間が必要です。小児LVH/RVH
パーセンタイル依存の評価は、明示的な機能では使用できないことを返します
キー。サポートされている小児間隔/軸ルールは独立したままです。

- [ ] **ステップ 5: 肥大テストを実行します**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_hypertrophy.py tests/test_pediatric_rules.py`

期待値: 合格。

---

<a id="task-6-engine-orchestration-conflicts-and-api-integration"></a>
### タスク 6: エンジン オーケストレーション、競合、および API の統合

**ファイル:**
- 作成: `feature_extraction/ecgfeat/clinical_rules/engine.py`
- 変更: `feature_extraction/ecgfeat/api.py:1963-1975`
- テスト: `tests/test_clinical_export_consumers.py`

**インターフェース:**
- 消費: すべての臨床評価者と従来のメタデータ。
- 生成: `analyze_clinical(features, strict=False) -> ClinicalAnalysis` が `features.metadata["clinical_interpretation"]` に保存されます。

- [ ] **ステップ 1: 失敗するオーケストレーション テストを作成する**

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

- [ ] **ステップ 2: テストを実行して失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py`

予期: エンジン/API 統合が存在しないため、失敗します。

- [ ] **ステップ 3: エンジン オーケストレーションの実装**

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

`group_by_domain` をドメイン名からシリアル化された辞書として実装します。
ルール ID でソートされた評価。 `detect_reference_conflicts`を実装して比較します
権威のある `overall_status` を持つ明示的な参照概要ラベルのみ。
たとえば、`normal` を含む参照ラベルは、権威のあるラベルと競合します。
`abnormal`、`borderline`、`technically_limited`、または `incomplete`。それぞれの葛藤
`reference`、`reference_summary`、`authoritative_status`、および `reason` があります。

必要な領域は、質、リズム、伝導、間隔、肥大、
再分極/虚血、および高リスクパターンのスクリーニング。ドメインの可用性
関数の戻り値だけではなく、ルールのステータスからもたらされます。

- [ ] **ステップ 4: 従来の分析後に API 呼び出しを追加します**

```python
ecg_features.interpretation = interpret(ecg_features)
ecg_features.metadata["glasgow_analysis"] = analyze_glasgow(ecg_features).to_dict()
ecg_features.metadata["clinical_interpretation"] = analyze_clinical(ecg_features).to_dict()
```

- [ ] **ステップ 5: API とエンジン テストを実行します**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_ecgfeat_pipeline.py`

期待値: 合格。

---

<a id="task-7-export-compatibility-fingerprint-and-consumer-migration"></a>
### タスク 7: エクスポートの互換性、フィンガープリント、および消費者の移行

**ファイル:**
- 変更: `feature_extraction/ecgfeat/export.py:1722-1804`
- 変更: `medgemma_ecg_core.py:241-360`
- 変更: `app.py:417-494`
- 変更: `batch_extract_ecgfeat.py:200-260`
- テスト: `tests/test_clinical_export_consumers.py`

**インターフェース:**
- 消費: メタデータの臨床分析。
- 生成: トップレベルの `clinical_interpretation`、レガシー参照マーカー、統合されたプロンプト サマリー、決定論的なアーティファクト フィンガープリント。

- [ ] **ステップ 1: 失敗したエクスポート/コンシューマ テストを延長する**

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

- [ ] **ステップ 2: テストを実行して失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py`

予期: 最上位フィールド/ヘルパーが欠落している場合は失敗します。

- [ ] **ステップ 3: 決定論的なエクスポート フィンガープリントを実装する**

`generated_at` を削除した後、臨床分析をシリアル化し、
`artifact_fingerprint` はソートされたコンパクトな JSON を使用し、それを SHA-256 にします。プレフィックス出力
`sha256:`と。従来のフィールド タイプを変更せずに、`reference_metadata` を追加します。

- [ ] **ステップ 4: MedGemma とアプリの概要を移行する**

`medgemma_ecg_core.py` に共有書式設定を追加します。

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

MedGemma とアプリの両方で、Glasgow 回線と従来の DXL 回線の前に使用します。

- [ ] **ステップ 5: バッチ ペイロードを保存する**

バッチ シリアル化で `to_dict(result)` が使用され、新しいデータがフィルターされないことを確認します。
最上位のフィールド。サマリーが次の場合に、そのサマリー行にルールセットのバージョンを追加します。
書かれた。

- [ ] **ステップ 6: 消費者テストを実行する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_export_contract.py tests/test_medgemma_ecg_core.py tests/test_batch_extract_ecgfeat.py`

期待値: 実行可能なテストはすべて合格します。環境のみで欠落しているオプションのモジュールは次のとおりです。
個別に報告され、ルールの失敗としてカウントされません。

---

<a id="task-8-textimage-report-authority-and-stale-artifact-detection"></a>
### タスク 8: テキスト/画像レポートの権限と古いアーティファクトの検出

**ファイル:**
- 変更: `demo_feature_extraction.py:560-713`
- 変更: `demo_feature_extraction.py:950-1352`
- テスト: `tests/test_clinical_export_consumers.py`
- テスト: `tests/test_ecg_report_sheet.py`

**インターフェース:**
- 消費: `metadata["clinical_interpretation"]` と指紋。
- 生成: 統合レポート セクション、参照付録、`STALE ARTIFACT` 警告。

- [ ] **ステップ 1: 失敗したレポートのテストを作成する**

```python
def test_report_shows_unified_summary_before_reference_algorithms(report_model):
    lines = report_model["interpretation_lines"]
    assert lines[0].startswith("Unified clinical summary")
    assert next(i for i, x in enumerate(lines) if "Glasgow reference" in x) > 0

def test_stale_fingerprint_is_visible(report_text):
    assert "STALE ARTIFACT" in render_with_mismatched_fingerprint(report_text)
```

- [ ] **ステップ 2: テストを実行して失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_ecg_report_sheet.py`

予期: 順序付け/警告アサーションで失敗します。

- [ ] **ステップ 3: 統合されたセクションと参照付録をレンダリングします**

レポートの順序:

1. 患者と測定値。
2. `UNIFIED CLINICAL INTERPRETATION`;
3. 信号品質。
4. 参照アルゴリズム (`GLASGOW REFERENCE`、`DXL-INSPIRED REFERENCE`)。
5. ルールの監査と制限。

独立したグラスゴーの「通常の ECG」という見出しを権威ある記事から削除します。
セクション。ラベルの付いた参照付録内にのみそのまま記載してください。

- [ ] **ステップ 4: 指紋チェックを追加します**

ペイロードに保存されている臨床フィンガープリントと新しく計算された臨床フィンガープリントを比較します。
指紋。不一致の場合は、先頭に `*** STALE ARTIFACT — REGENERATE REPORT ***` を追加します。

- [ ] **ステップ 5: レポート テストの実行**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_export_consumers.py tests/test_ecg_report_sheet.py tests/test_glasgow_report_rendering.py`

期待値: 合格。

---

<a id="task-9-clinical-validation-metric-utility"></a>
### タスク 9: 臨床検証指標ユーティリティ

**ファイル:**
- 作成: `feature_extraction/ecgfeat/clinical_rules/validation.py`
- テスト: `tests/test_clinical_validation.py`

**インターフェース:**
- 消費: `{truth: bool, predicted: bool|None, age, sex, quality}` 行の反復可能。
- プロデュース: `binary_metrics(rows)`、`stratified_metrics(rows, keys)`。

- [ ] **ステップ 1: 失敗したメトリック テストを作成する**

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

- [ ] **ステップ 2: テストを実行して失敗を確認する**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_validation.py`

予期: インポートで失敗します。

- [ ] **ステップ 3: ゼロセーフ メトリクスを実装する**

戻り数、利用不可率、感度、特異度、PPV、NPV、F1、および
カバレッジ。分母がゼロの場合は、`None` が返されます。決してゼロではありません。層別化の用途
正確なタプルキーを取得し、同じバイナリ関数を呼び出します。

- [ ] **ステップ 4: 検証テストの実行**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_validation.py`

期待値: 合格。

---

<a id="task-10-js00059-end-to-end-regression-and-artifact-regeneration"></a>
### タスク 10: JS00059 エンドツーエンドの回帰とアーティファクトの再生成

**ファイル:**
- 作成: `tests/test_js00059_clinical_regression.py`
- 再生成: `JS00059_features.json`
- 再生成: `JS00059_report.txt`
- 再生成: `JS00059_ecg.png`
- 再生成: `JS00059_ecg_annotated.png`
- 再生成: `JS00059_ecg_report.png`

**インターフェース:**
- 消費: `dataset/JS00059.mat`、`dataset/JS00059.hea`、完全なパイプライン。
- 生成: 1 つのルールセット バージョン/フィンガープリントを持つ現在の統合アーティファクト。

- [ ] **ステップ 1: 失敗した回帰テストを作成する**

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

- [ ] **ステップ 2: 回帰テストを実行し、古いアーティファクトでの失敗を確認します**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_js00059_clinical_regression.py`

予期: 古いアーティファクトに `clinical_interpretation` と
指紋。

- [ ] **ステップ 3: 現在のデモ ジェネレーターを実行します**

実行: `PYTHONPATH=. .venv/bin/python demo_feature_extraction.py JS00059`

予期: 0 を終了し、5 つのアーティファクトすべてが現在のタイムスタンプを受け取ります。

- [ ] **ステップ 4: 回帰と集中的な臨床スイートの実行**

実行:

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

期待値: 合格。

---

<a id="task-11-full-verification-and-documentation-sync"></a>
### タスク 11: 完全な検証とドキュメントの同期

**ファイル:**
- 変更: `docs/philips_adult_morphology_feature_schema_checklist.md`
- 変更: `docs/philips_child_morphology_feature_schema_checklist.md`
- 変更: `docs/philips_rhythm_coverage_matrix.md`

**インターフェース:**
- 消費: 完了した実装とテスト出力。
- 生成: 統一された公開ルールと参照ベンダーの近似を区別する正確な文書。

- [ ] **ステップ 1: ステータス ドキュメントを更新する**

権限のあるレイヤー、レガシー参照ステータスを意図的に文書化します。
利用できない小児規則、新しいスガルボッサ/ブルガダスクリーニング、正確な検査
コマンド。ステートメント エンジンまたはペーシングファースト フローが無効であるという古い主張を削除します。
現在のコードがそれを実装する場合には欠落します。

- [ ] **ステップ 2: 重点的な検証を実行します**

実行: `PYTHONPATH=. .venv/bin/pytest -q tests/test_clinical_*.py tests/test_js00059_clinical_regression.py`

期待値: 合格。

- [ ] **ステップ 3: スイート全体を実行する**

実行: `PYTHONPATH=. .venv/bin/pytest -q`

予期: 新しい臨床アサーションまたは従来のアサーションの失敗はありません。既存のレコード
個別の環境障害: ハードコーディングされた不在の `.venv_report_regen_20260625`、
オプションの `wfdb`、`gradio`、または `snapshot_regression` が欠落していてはなりません
臨床ルールの失敗として誤って報告された。

- [ ] **ステップ 4: 生成された出力を検査する**

実行:

```bash
rg -n "UNIFIED CLINICAL|reference-only|ruleset|fingerprint|STALE ARTIFACT|Normal ECG" \
  JS00059_report.txt JS00059_features.json
```

予期: 統合された概要と一致するバージョン/フィンガープリントが存在します。 「正常」
レートを除く ECG は、ラベル付きのグラスゴー参照付録にのみ表示されます。
`STALE ARTIFACT` がありません。

- [ ] **ステップ 5: 最終検証チェックポイント**

変更されたファイル、重点的/完全なテスト数、既存の環境の概要
失敗、JS00059 の統合結果、および臨床的に明らかに利用できない残りのデータ
ドメイン。臨床的検証や独自のアルゴリズムの同等性を主張しないでください。
