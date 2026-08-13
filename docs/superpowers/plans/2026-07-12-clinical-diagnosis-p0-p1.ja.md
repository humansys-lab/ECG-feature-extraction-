<!-- i18n-nav -->
[中文](2026-07-12-clinical-diagnosis-p0-p1.md) | [English](2026-07-12-clinical-diagnosis-p0-p1.en.md) | [日本語](2026-07-12-clinical-diagnosis-p0-p1.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="clinical-diagnosis-p0p1-implementation-plan"></a>
# 臨床診断 P0/P1 実施計画

> **エージェント ワーカーの場合:** 必須サブスキル: superpowers:subagent-driven-development (推奨) または superpowers:executing-plan を使用して、この計画をタスクごとに実装します。ステップでは、追跡にチェックボックス (`- [ ]`) 構文を使用します。

**目標:** 統合された ECG ルール入力コントラクトと誤ったリード逆転抑制を修復し、全か無かの可用性を明示的なルール カバレッジ、信頼性、正常性の役割、および構成全体のステータスに置き換えます。

**アーキテクチャ:** 代表的なリードパラメータは、標準的な臨床ルールの入力境界のままです。ルール評価では、レガシー移行アダプターを使用して、独立したステータス、カバレッジ、および信頼度のディメンションを取得します。リゾルバーは正規性の役割を消費し、結果と制限の両方を保持する全体的な結果を生成します。リード反転ヒューリスティックは、検出器がペアを確認済みとして明示的にマークしない限り、アドバイスのままです。

**技術スタック:** Python 3.12、データクラス、NumPy、pytest、既存の `feature_extraction.ecgfeat` パッケージ。

<a id="global-constraints"></a>
## グローバル制約

- グラスゴーおよび DXL からインスピレーションを得た分析を参照専用の出力として保存します。
- データセット ヘッダー診断コードをランタイム入力として使用しないでください。
- 欠けている証拠や信頼性の低い証拠を、肯定的な診断や正常性の証拠に変換しないでください。
- `matched`、`not_matched`、`unavailable`、`not_applicable`、および `suppressed` ルール行に対する逆方向読み取りのサポートを維持します。
- 変更されたセマンティクスに合わせてスキーマとルールセットのバージョンを進めます。
- すべての生産変更については、RED/GREEN/REFACTOR に従ってください。
- `/workspace/ecg_gemma` は現在 Git リポジトリではありません。コミットステップを明示的な検証チェックポイントと変更されたファイルのインベントリに置き換えます。

---

<a id="file-map"></a>
## ファイルマップ

- `feature_extraction/ecgfeat/features.py`: 正規の代表的な測定フィールドを生成します。
- `feature_extraction/ecgfeat/quality.py`: 前胸部逆転証拠を勧告または確認済みとして分類します。
- `feature_extraction/ecgfeat/api.py`: 構造化された反転メタデータを保存します。
- `feature_extraction/ecgfeat/clinical_rules/context.py`: 測定エイリアスを消費し、確認されたリードのみを除外します。
- `feature_extraction/ecgfeat/clinical_rules/models.py`: 新しいルールのステータス、カバレッジ、信頼性、および正規性の役割を定義します。
- `feature_extraction/ecgfeat/clinical_rules/engine.py`: コア役割からのドメイン範囲を要約します。
- `feature_extraction/ecgfeat/clinical_rules/resolver.py`: 調査結果、適用範囲、および技術的制限を構成します。
- `feature_extraction/ecgfeat/clinical_rules/quality.py`: プロジェクト信号/測定/ドメイン品質の証拠。
- `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py`: 正規の P 持続時間を消費します。
- `feature_extraction/ecgfeat/clinical_rules/conduction.py`: 代表的な R-プライム証拠を消費します。
- `feature_extraction/ecgfeat/clinical_rules/ischemia.py`: 正規化された署名付き QRS 領域を消費します。
- `feature_extraction/ecgfeat/clinical_rules/sources.py`: 高度なスキーマ/ルールセット バージョン。
- `feature_extraction/ecgfeat/export.py`: 指紋バージョン対応リゾルバー セマンティクス。
- `tests/test_clinical_measurement_contract.py`: 新しいプロデューサー/コンシューマー契約テスト。
- 既存の集中テスト モジュール: フィクスチャを複製せずに動作回帰を追加します。

---

<a id="task-1-canonical-representative-measurement-contract"></a>
### タスク 1: 正規代表測定契約

**ファイル:**
- 作成: `tests/test_clinical_measurement_contract.py`
- 変更: `feature_extraction/ecgfeat/features.py:612-652`
- 変更: `feature_extraction/ecgfeat/clinical_rules/context.py:35-70`
- 変更: `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py:169-199`
- 変更: `feature_extraction/ecgfeat/clinical_rules/ischemia.py:174-203`

**インターフェース:**
- 消費: 信頼できる支配的なグループの `LeadBeatFeatures` 値。
- 生成: 正規代表キー `p_dur_consensus_ms`、`r_prime_amp_mv`、および `qrs_signed_area_uv_ms`。 `ClinicalContext.lead_value()` は、文書化されたレガシー エイリアスのみを解決します。

- [ ] **ステップ 1: 失敗するコントラクト テストを作成する**

`tests/test_lead_reversal.py::_make_beat_feature` から派生した実際の `LeadBeatFeatures` フィクスチャを追加し、次のようにアサートします。

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

- [ ] **ステップ 2: テストを実行して RED を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_measurement_contract.py
```

予期: 代表的な R プライム キーと署名付きエリア キーが欠落しているためエラーが発生し、`p_duration_ms` を読み取るため RAE が使用できなくなります。

- [ ] **ステップ 3: 正規プロデューサー フィールドを追加する**

これらのエントリを代表的な `params` 辞書に追加します。

```python
"r_prime_amp_mv": _pick_representative_value(rep_feature, items, "r_prime_amp_mv"),
"s_prime_amp_mv": _pick_representative_value(rep_feature, items, "s_prime_amp_mv"),
"qrs_signed_area_uv_ms": (
    None
    if (signed := _pick_representative_value(rep_feature, items, "qrs_signed_area")) is None
    else float(signed) * 1000.0
),
```

格納された署名領域がすでにマイクロボルトミリ秒単位である場合は、そのプロデューサー テストでそれを証明し、変換を省略します。契約テストでは、既知の波形値でユニットを確立する必要があります。

- [ ] **ステップ 4: 文書化された単一のエイリアス境界を追加します**

`ClinicalContext` に次を追加します。

```python
LEAD_PARAM_ALIASES = {
    "p_duration_ms": ("p_dur_consensus_ms",),
    "qrs_signed_area_uv_ms": ("qrs_signed_area_uv_ms", "twelve_sl_qrs_signed_area_uv_ms"),
}
```

`lead_raw_value()` は、最初に要求された正規キーを試行し、次にエイリアスを試行する必要があります。欠損値をゼロに置き換えてはなりません。

- [ ] **ステップ 5: ルール コンシューマを更新する**

RAE は `p_dur_consensus_ms` をリクエストします。 Sgarbossa は正規の `qrs_signed_area_uv_ms` のみを要求します。各ルール内にフォールバック ロジックを追加するのではなく、古いアーティファクトのコンテキスト内でエイリアスの境界を維持します。

- [ ] **ステップ 6: GREEN および焦点を絞った回帰を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_measurement_contract.py \
  tests/test_clinical_hypertrophy.py \
  tests/test_clinical_conduction.py \
  tests/test_clinical_ischemia.py \
  tests/test_ecgfeat_pipeline.py -k 'representative_lead_features'
```

期待値: 選択したすべてのテストに合格します。

- [ ] **ステップ 7: チェックポイントを記録**

変更されたファイルとフォーカスされたテスト出力を作業ログに記録します。ワークスペースがリポジトリになるまでは、Git コマンドを呼び出さないでください。

---

<a id="task-2-advisory-precordial-reversal-state"></a>
### タスク 2: 前胸部反転状態の勧告

**ファイル:**
- 変更: `tests/test_lead_reversal.py`
- 作成: `tests/test_clinical_context.py`
- 変更: `feature_extraction/ecgfeat/quality.py:741-775`
- 変更: `feature_extraction/ecgfeat/features.py:741-748`
- 変更: `feature_extraction/ecgfeat/api.py:1546-1560`
- 変更: `feature_extraction/ecgfeat/clinical_rules/context.py:73-94`
- 変更: `feature_extraction/ecgfeat/interpret.py:2360-2382`

**インターフェース:**
- 消費: 構造化された `metadata.lead_reversal.precordial`。
- 互換性のために、`state` (`not_suspected`、`possible`、`confirmed`)、`confidence`、`implicated_leads`、およびレガシー `suspected` を生成します。

- [ ] **ステップ 1: 失敗した反転状態テストを作成する**

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

- [ ] **ステップ 2: テストを実行して RED を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_lead_reversal.py tests/test_clinical_context.py
```

予期: `state` が欠落しており、すべての V1 ～ V6 の現在のコンテキストが除外されています。

- [ ] **ステップ 3: 確認を要求せずに検出器の出力を拡張します**

戻り値:

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

単調ヒューリスティックは `confirmed` を返しません。既存の低電圧および不完全なマップの理由を維持します。

- [ ] **ステップ 4: コンテキスト除外ポリシーを変更する**

最初に構造化メタデータを読み取ります。 `state == "confirmed"` が `implicated_leads` を `excluded_leads` に追加する場合のみ。構造化確認のない従来のブール値は `possible` となり、何も除外されません。

- [ ] **ステップ 5: 勧告レポートの保存**

状態が可能な場合は、既存のレポートの互換性のために `precordial_reversal_suspected=True` を保持しますが、構造化された状態をメタデータと臨床品質証拠に追加します。臨床虚血/ブルガダ検査を変更して、逆転の可能性がルールを抑制するのではなく勧告を追加するようにします。

- [ ] **ステップ 6: GREEN および JS00006 回帰を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_lead_reversal.py \
  tests/test_clinical_context.py \
  tests/test_clinical_ischemia.py \
  tests/test_js00059_clinical_regression.py
```

次に、JS00006 反転メタデータを使用して読み取り専用フィクスチャ アサーションを実行し、再構築されたコンテキストで前胸部誘導がゼロであることを確認します。

- [ ] **ステップ 7: チェックポイントを記録**

テスト出力と、JS00006 の除外リード セットの前後を記録します。

---

<a id="task-3-rule-status-coverage-and-confidence-schema"></a>
### タスク 3: ルールのステータス、適用範囲、および信頼スキーマ

**ファイル:**
- 変更: `tests/test_clinical_models_resolver.py`
- 作成: `tests/test_clinical_schema_migration.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/models.py:7-42`

**インターフェース:**
- 消費: 新しいルール行と従来のルール行。
- 生成: 決定論的なレガシー投影を使用した `RuleEvaluation(status, coverage, confidence, normality_role)`。

- [ ] **ステップ 1: 古い入力欠落テストを失敗したセマンティック テストに置き換えます**

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

- [ ] **ステップ 2: テストを実行して RED を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_schema_migration.py
```

予期: `indeterminate` が無効、カバレッジ/正規性の役割が欠落しており、古い `__post_init__` が証明された一致を上書きします。

- [ ] **ステップ 3: 新しい列挙型と検証を実装する**

定義:

```python
VALID_STATUSES = {"matched", "not_matched", "indeterminate", "not_applicable"}
VALID_COVERAGE = {"full", "partial", "unavailable"}
VALID_CONFIDENCE = {"high", "moderate", "low", "unavailable"}
VALID_NORMALITY_ROLES = {"core", "supporting", "optional_screen"}
```

互換性のあるデフォルトを持つフィールドを追加します。

```python
coverage: str = "full"
confidence: str = "moderate"
normality_role: str = "core"
```

自動 `missing_inputs -> unavailable` および `suppressed_by -> suppressed` 変異を削除します。評価者は論理的な結果を明示的に述べなければなりません。

- [ ] **ステップ 4: 逆シリアル化時にレガシー移行を実装する**

マッピングする `RuleEvaluation.from_dict()` を追加します。

```python
"unavailable" -> status="indeterminate", coverage="unavailable", confidence="unavailable"
"suppressed" -> status="indeterminate", coverage="partial", confidence="low"
```

`normality_required=False` を `normality_role="optional_screen"` にマップします。移行期間中にシリアル化された出力に古いフィールドを保持します。

- [ ] **ステップ 5: ルール ヘルパー コンストラクターを更新する**

伝導、肥大、虚血の `_evaluation()` ヘルパーと `_result()` ヘルパーを更新して、従来の利用不可能または抑制されたすべてのブランチが新しいステータス/カバレッジ/信頼度のディメンションを明示的に返すようにしました。

- [ ] **ステップ 6: GREEN とシリアル化の互換性を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_schema_migration.py \
  tests/test_clinical_conduction.py \
  tests/test_clinical_hypertrophy.py \
  tests/test_clinical_ischemia.py \
  tests/test_clinical_intervals.py
```

期待値: 選択されたすべてのテストに合格し、カバレッジ、信頼性、または正規性の役割が不足しているシリアル化されたルールはありません。

- [ ] **ステップ 7: チェックポイントを記録**

従来から新しいマッピング テーブルとテスト出力を記録します。

---

<a id="task-4-domain-coverage-and-compositional-resolver"></a>
### タスク 4: ドメイン カバレッジと構成リゾルバー

**ファイル:**
- 変更: `tests/test_clinical_models_resolver.py`
- 変更: `tests/test_clinical_export_consumers.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/engine.py:17-25,117-166`
- 変更: `feature_extraction/ecgfeat/clinical_rules/resolver.py:10-67`
- 変更: `feature_extraction/ecgfeat/clinical_rules/models.py:45-62`

**インターフェース:**
- 消費: ルールのステータス、カバレッジ、信頼性、および正規性の役割。
- 生成: ドメインの概要と構成全体のステータス。

- [ ] **ステップ 1: 失敗するリゾルバー テストを作成する**

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

- [ ] **ステップ 2: テストを実行して RED を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_models_resolver.py
```

予期: 既存のリゾルバーは `abnormal`、`technically_limited`、`incomplete` を返すか、レートを異常として扱います。

- [ ] **ステップ 3: バイナリ ドメインの可用性を概要に置き換えます**

ドメインごとに、`_domain_summaries(evaluations)` を返すように実装します。

```python
{
    "coverage": "full" | "partial" | "unavailable",
    "confidence": "high" | "moderate" | "low" | "unavailable",
    "core_assessed": int,
    "core_applicable": int,
    "unassessed_reasons": list[str],
}
```

該当する `normality_role == "core"` ルールのみがコア正規性の適用範囲を決定します。

- [ ] **ステップ 4: リゾルバーの優先順位を実装する**

この正確な優先順位を使用してください。

```text
technical + abnormal findings -> technically_limited_with_findings
technical without abnormal findings -> technically_limited
abnormal + incomplete core coverage -> abnormal_with_limited_coverage
abnormal + complete core coverage -> abnormal
borderline -> borderline
complete applicable core coverage -> normal_with_core_coverage
otherwise -> incomplete
```

重大度 `observation` の記述は最終レポートの事実ですが、異常リストのメンバーではありません。

- [ ] **ステップ 5: ドメインの概要をエクスポートする**

詳細なルール行を `domains` に保持します。 `summary["domain_coverage"]`、`summary["partial_evaluation"]`、および個別の `findings`、`observations`、および `limitations` カウントを追加します。

- [ ] **ステップ 6: GREEN と消費者向けの互換性を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_models_resolver.py \
  tests/test_clinical_export_consumers.py \
  tests/test_medgemma_ecg_core.py \
  tests/test_ecg_report_sheet.py
```

予想通り: 選択されたすべてのテストに合格し、コンシューマーは従来の概要にフォールバックすることなく新しいステータスを表示します。

- [ ] **ステップ 7: チェックポイントを記録**

リゾルバーの真理値表と焦点を当てたテスト出力を記録します。

---

<a id="task-5-quality-projection-and-normality-roles"></a>
### タスク 5: 品質の予測と正常性の役割

**ファイル:**
- 作成: `tests/test_clinical_quality_domains.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/quality.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/ischemia.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/intervals.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/rhythm.py`

**インターフェース:**
- 消費: 記録品質、代表的な測定の信頼性、および反転状態。
- 生成: 個別の信号品質ルール、測定品質証拠、およびルール正規性の役割。

- [ ] **ステップ 1: 失敗する品質ルーティング テストを作成する**

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

- [ ] **ステップ 2: テストを実行して RED を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_quality_domains.py
```

予想: ST は一般的な QRS 信頼性を使用し、Brugada はデフォルトでコア正常性を設定し、反転勧告は存在しません。

- [ ] **ステップ 3: プロジェクトの構造化された品質証拠**

品質エバリュエーターは、記録品質のコア ルールに加えて、逆転の可能性と測定制限に関する観測行を返します。証拠には、`record_grade`、拒否された関数、信頼できるリード数、反転状態、および除外されたリードが含まれます。

- [ ] **ステップ 4: ルート測定固有のゲート**

ST ルール、QT の信頼性/サポート、心房形態の P 信頼性/関連付けには代表的な `st_j_reliable` を使用します。汎用 `reliable_for_qrs` は、依然として QRS 形態にのみ適切です。

- [ ] **ステップ 5: 正規性の役割を割り当てる**

ブルガダ スクリーニングおよびその他の稀な高リスク スクリーニングを `optional_screen` に設定します。コアリズム、伝導、インターバル、および急性再分極の評価は、該当する場合には引き続きコアとなります。サポート電圧基準だけでは完全なドメインネガティブの結論を確立できない場合、`supporting` を使用します。

- [ ] **ステップ 6: グリーンを確認**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_quality_domains.py \
  tests/test_clinical_ischemia.py \
  tests/test_clinical_intervals.py \
  tests/test_clinical_models_resolver.py
```

期待値: 選択したすべてのテストに合格します。

- [ ] **ステップ 7: チェックポイントを記録**

すべての P0/P1 ルールの正常性ロール インベントリを記録します。

---

<a id="task-6-schema-ruleset-fingerprint-and-stale-artifacts"></a>
### タスク 6: スキーマ、ルールセット、フィンガープリント、および古いアーティファクト

**ファイル:**
- 変更: `tests/test_clinical_export_consumers.py`
- 変更: `tests/test_js00059_clinical_regression.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/sources.py`
- 変更: `feature_extraction/ecgfeat/export.py:48-86`
- 変更: `demo_feature_extraction.py`
- 変更: `medgemma_ecg_core.py`

**インターフェース:**
- 消費: 新しい臨床スキーマとリゾルバー ポリシー。
- 生成: `clinical_rules.v2`、新しいルールセット バージョン、確定的なフィンガープリント、および目に見える古いアーティファクト警告。

- [ ] **ステップ 1: 失敗したバージョン/フィンガープリント テストを作成する**

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

- [ ] **ステップ 2: テストを実行して RED を確認する**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_export_consumers.py \
  tests/test_js00059_clinical_regression.py
```

予想: 現在のスキーマ/ルールセット アサーションは引き続き v1/2026.07.1 を報告します。

- [ ] **ステップ 3: アドバンス バージョンと指紋入力**

設定:

```python
RULESET_VERSION = "2026.07.2"
SCHEMA_VERSION = "clinical_rules.v2"
RESOLVER_POLICY_VERSION = "coverage-confidence-v1"
```

既存の決定論的フィンガープリントでカバーされるように、臨床分析の概要にリゾルバー ポリシーを含めます。

- [ ] **ステップ 4: コンシューマーと古い警告を更新する**

現在のランタイム ルールセットが v2 の場合、コンシューマは読み取り用に v1 と v2 を受け入れ、v2 ステータスをレンダリングし、保存されている v1 アーティファクトを古いものとしてマークします。保存されている v1 の結果を黙って現在のものとして再ラベル付けしてはなりません。

- [ ] **ステップ 5: グリーンを確認**

実行:

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q \
  tests/test_clinical_export_consumers.py \
  tests/test_js00059_clinical_regression.py \
  tests/test_export_contract.py \
  tests/test_medgemma_ecg_core.py
```

期待値: 選択したすべてのテストに合格します。

- [ ] **ステップ 6: チェックポイントを記録**

スキーマ移行メモ、コンシューマ互換性、および焦点を絞ったテスト出力を記録します。

---

<a id="task-7-p0p1-regression-and-batch-audit"></a>
### タスク 7: P0/P1 回帰とバッチ監査

**ファイル:**
- 変更: 新しい v2 証拠を生成した後にのみ `docs/unified_clinical_rules_v1_validation_audit.md` を作成するか、古い監査を変更せずに保持する場合は `docs/unified_clinical_rules_v2_validation_audit.md` を作成します。
- 既存のバッチ監査エントリ ポイントが必要なカウントを生成できない場合にのみ、`scripts/` で決定論的監査スクリプトを作成または変更します。

**インターフェース:**
- 消費: 新たに抽出された 100 レコードの v2 アーティファクト。
- 生成: 再現可能なカバレッジ、ステータス、フィールド契約およびステートメント数の監査。

- [ ] **ステップ 1: すべての集中した P0/P1 テストを実行します**

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

期待値: 失敗はゼロ。

- [ ] **ステップ 2: 既存のテスト スイート全体を実行する**

```bash
PYTHONPATH=/workspace/ecg_gemma pytest -q
```

期待値: 失敗はゼロ。無関係な既存の障害が存在する場合は、それらを既存として分類する前に、正確なノード ID を取得し、P0/P1 パッチなしでも障害が発生することを実証します。

- [ ] **ステップ 3: 100 レコードのアーティファクトを再生成**

v1 監査に使用したものと同じ入力ディレクトリとオプションを備えた既存のバッチ抽出ツールを使用し、最初に新しい v2 出力ディレクトリに書き込みます。カウントとフィンガープリントがチェックされるまで、v1 証拠を上書きしないでください。

- [ ] **ステップ 4: バッチ エンジニアリング ゲートを強制する**

監査スクリプトで次のようにアサートします。

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

- [ ] **ステップ 5: v1 と v2 ステートメントの分布を比較**

全体的なステータス、部分的なカバレッジ、利用できないドメイン、取り消し状態、およびすべての最終ステートメント コードについて、古いカウントと新しいカウントをレポートします。誤検知を確認せずに、ステートメント数の増加を改善と解釈しないでください。

- [ ] **ステップ 6: v2 監査を作成する**

コマンド、入力母集団、正確なコード バージョン フィールド、エンジニアリング ゲート、既知の制限、およびヘッダー ラベルが臨床的判断ではなく弱参照であるという境界を含めます。

- [ ] **ステップ 7: P0/P1 受け入れチェックリストを確認する**

設計仕様内のすべての P0/P1 要件に、合格したテストまたは v2 監査の証拠があることを確認します。延期された P2 項目を明示的にリストします: AF/AFL 形態、後部/急性虚血証拠層、Q 波テリトリー ロジック、QT/JT 補正、およびリゾルバー互換性にはまだ必要ではないコンテキスト レート重症度の調整。

---

<a id="plan-self-review"></a>
## 自己レビューを計画する

- **仕様範囲:** タスク 1 ～ 2 は、P0 入力および反転の欠陥をカバーします。タスク 3 ～ 5 では、P1 ステータス、リゾルバー、および品質アーキテクチャについて説明します。タスク 6 では再現性について説明します。タスク 7 では、回帰ゲートと監査ゲートについて説明します。
- **範囲境界:** AF/AFL、虚血、QT/JT、および欠落している診断ファミリーは、P0/P1 スキーマで必要な互換性の変更を除き、P2/P3 のままです。 P0/P1 バッチ証拠が入手可能になった後、詳細な TDD 計画を別途受け取ります。
- **型の一貫性:** 正規ルール フィールドは、`status`、`coverage`、`confidence`、および `normality_role` です。正規の代表フィールドは、`p_dur_consensus_ms`、`r_prime_amp_mv`、および `qrs_signed_area_uv_ms` です。
- **プレースホルダー動作なし:** 各タスクは、正確なファイル、予想される RED 状態、最小限の運用動作、および検証コマンドを識別します。

