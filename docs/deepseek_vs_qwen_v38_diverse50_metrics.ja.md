<!-- i18n-nav -->
[中文](deepseek_vs_qwen_v38_diverse50_metrics.md) | [English](deepseek_vs_qwen_v38_diverse50_metrics.en.md) | [日本語](deepseek_vs_qwen_v38_diverse50_metrics.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="deepseek-vs-qwenv38-diverse50指标计算方法与结果全记录"></a>
# DeepSeek vs. Qwen (v38 Divers50) 指数計算方法と結果の全記録

2026 年 8 月 3 日の分析で使用された各指標の**正確な定義、計算式、および完全な結果**を記録し、再導出を避けるため、その後の再計算、レビュー、またはレポート資料を作成する際の参照に備えます。

<a id="0-数据来源"></a>
## 0. データソース

- `qwen_agent_output/deepseek_v38_diverse50/`: バックボーンとして実行されている DeepSeek の 50 レコード
- `qwen_agent_output/qwen_v38_diverse50/`: バックボーンとして実行されている Qwen の 50 件のレコード
- 2 つの `record_manifest.json` の `dataset_dir` はまったく同じです (両方とも `qwen_v36_diverse50/wfdb_headers` を指します)。これらは 50 PTB-XL レコードの同じバッチであり、直接比較できることが確認されています。
- 関連するスクリプト: ウェアハウスに付属する `evaluate_target_ecgfeat_diagnosis.py` (`CATEGORIES` 定義) および `ecgagent/batch.py` (`_category_set`/`_metric_rows`) は、`category_metrics.csv`/`record_results.csv` 公式ロジックを生成します)、`score_measurement_verifiable.py` (測定レビュー);このドキュメントで「一時スクリプト」とマークされている部分は、この分析中に作成されたものであり、ウェアハウスにはありません。再現のために式を以下に示します。

---

<a id="1-第一层逐记录-逐诊断家族的四格表"></a>
## 1. 第 1 レベル: レコードごとのレコード × 診断ファミリーごとの 4 つのグリッド テーブル

<a id="11-诊断家族定义"></a>
### 1.1 診断ファミリーの定義

`CATEGORIES: tuple[CategorySpec, ...]` は `evaluate_target_ecgfeat_diagnosis.py` で事前定義されており、各ファミリーに 1 つの `CategorySpec` があります。

```python
@dataclass
class CategorySpec:
    key: str                          # 家族 id，如 "atrial_fibrillation"
    display_name_zh: str              # 中文名
    ptbxl_refs: frozenset[str]        # 参考标签集合，如 {"AFIB"}
    local_refs: frozenset[str]
    prediction_codes: frozenset[str]  # 模型/规则引擎输出中，命中该家族的代码集合
    evaluates_codes: frozenset[str] = frozenset()
    rule_ids: frozenset[str] = frozenset()
    semantic_scope: str = "direct"    # "direct" | "broad" | "screening"
```

この分析で使用された 50 レコードの中には、`semantic_scope="direct"` の **17** ファミリがあります (完全なリストについては、付録 A を参照してください)。ヘッドライン インジケーターは、`direct` ファミリのみを要約しています。 `broad` (T 波異常、心房異常、QRS 低電圧など) および `screening` (虚血/ST スクリーニング、心筋梗塞/Q 波スクリーニング) ファミリーの定義は曖昧であり、基準は単一の数値比較ではないため、個別の議論のために除外されています。

<a id="12-四格表判定"></a>
### 1.2 四マス表の判定

各レコード `i`、各ファミリー `spec`:

```
reference_positive = (记录 i 的 PTB-XL 标签集合) ∩ spec.ptbxl_refs ≠ ∅
predicted_positive  = (模型/规则引擎输出的代码集合) ∩ spec.prediction_codes ≠ ∅
```

| |参照_ポジティブ | ←参照_正 |
|---------------------|--------------------------|---------------------|
| **予測陽性** | TP | FP |
| **| 予測陽性**| FN | TN（F1不参加） |

レコードごと、ファミリーごとにレコードを蓄積し、`category_metrics.csv` (`ecgagent/batch.py::_metric_rows`) の各ファミリーの TP/FP/FN を取得します。

---

<a id="2-第二层precision-recall-f1及五种平均口径"></a>
## 2. 第 2 レベル: 精度/再現率/F1、および 5 つの平均口径

標準配合:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

同じ 4 つのグリッド テーブルのデータであっても、「平均する方法」については多くの方法が認められており、結論も異なります。今回はそれらをすべて計算してみました。

<a id="21-micro-f1汇总后算一次"></a>
### 2.1 Micro-F1 (集計後に 1 回カウント)

まず、すべての `direct` ファミリの TP/FP/FN と 50 レコードすべてを合計テーブルに追加し、P/R/F1 を計算します。 **このレポートのデフォルトの内容**も、最初に指定される番号です。

<a id="22-macro-f1label-centric逐家族先各算-f1-再平均"></a>
### 2.2 マクロ F1 (ラベル中心、最初に各ファミリーの F1 を計算し、次に平均を計算します)

```
Macro-F1 = (1/|F|) · Σ_{f∈F} F1_f
```
`F` は 17 の直接ファミリーのセットであり、各ファミリーは、50 レコード内に 1 回出現するか 6 回出現するかに関係なく、同じ重みを持ちます。これは [PTB-XL 公式ベンチマーク ペーパー](https://arxiv.org/abs/2004.13701) (Strodthoff et al. 2020) §II.C (元のテキストは AUC に使用されます。今回の出力は確率スコアなしの 2 値判定であるため、同じ「クラスごとの等しい重み平均」のアイデアが F1 に適用されます): 論文の元の言葉 - 「クラスの不均衡が予想され、スコアが少数の大きなクラスによって支配されることを望まないため、マクロ平均化が推奨されます。」

<a id="23-sample-centric-f1record-centric逐记录先各算准召再平均"></a>
### 2.3 サンプル中心の F1 (レコード中心、レコードごとの計算と平均)

また、[PTB-XL 論文](https://arxiv.org/abs/2004.13701) §II.C、CAFA タンパク質機能予測チャレンジから導出された式:

```
Precision = (1/N_P) · Σ_{i: |P_i|>0}  TP_i / |P_i|      # 只在"该记录至少预测了1个家族"的记录上取平均
Recall    = (1/N_T) · Σ_{i: |T_i|>0}  TP_i / |T_i|      # 只在"该记录真实至少有1个家族"的记录上取平均
F1        = 2 · Precision · Recall / (Precision + Recall)
```

ここで、`P_i`/`T_i` は、レコード `i` の予測/実際の直接ファミリー セットであり、`N_P`/`N_T` は、分母がゼロ以外のレコードの数です。元の論文における予測側の分母処理は「予測のあるサンプルの平均のみ」です。今回は、リコール側も「実際のラベルを持つサンプルの平均」に対称的に制限されます（このデータセット内の多数のレコードには直接ファミリーの空の実ラベルが含まれるため、各タンパク質に機能ラベルが必要な CAFA シナリオとは異なります。この対称的な処理は、ゼロ除算を防ぐために必要です）。 **元の論文の Fmax も、予測閾値 τ scan の最大値をとります**。このモデルの出力はハードバイナリ判定 (スキャンする確率スコアがない) であり、実際の Fmax ではなく、しきい値スキャン曲線上の唯一の点の F1 のみをカウントすることに相当します。

<a id="24-subset-accuracy子集精确匹配率"></a>
### 2.4 サブセット精度 (サブセット完全一致率)

```
Subset Accuracy = (1/N) · Σ_i 𝟙[P_i = T_i]
```
「予測されたファミリー セットが実際のラベル セットと完全に一致しているかどうか」をレコードごとに、はい/いいえバイナリで判断します。これは、マルチラベル分類文献における古典的な指標の 1 つです。

<a id="25-平均-jaccard-相似度"></a>
### Jaccard 類似度の平均 2.5

```
Jaccard_i = |P_i ∩ T_i| / |P_i ∪ T_i|     （P_i、T_i 都为空时记为 1）
Mean Jaccard = (1/N) · Σ_i Jaccard_i
```

---

<a id="3-第三层-a测量复核scoremeasurementverifiablepy"></a>
## 3. 第 3 レベル A: 測定レビュー (`score_measurement_verifiable.py`)

1 層目と 2 層目は、値が正しいかどうかに関係なく、「モデル コード文字列とラベル コード文字列」が一致するかどうかだけを調べます。定義自体がしきい値比較 (心拍数、PR/QT 間隔、QRS 電圧、スピンドル) であるファミリーの場合、スクリプトはラベルをバイパスし、`*_features.json` レコード自体を使用して直接再計算します。

```python
CHECKS = {
    "sinus_bradycardia": lambda r: r.g("heart_rate_bpm") < 60.0,
    "first_degree_av_block": lambda r: r.g("pr_ms") > 200.0,
    "prolonged_qt": lambda r: r.qtc() > (460.0 if r.female else 450.0),
    "left_axis_deviation": lambda r: -90.0 < r.g("qrs_axis_deg") < -30.0,
    "low_qrs_voltage_limb_leads": lambda r: all(amp < 0.50 for amp in 肢导振幅),
    # ... 完整列表见 score_measurement_verifiable.py
}
```

各モデルによる診断では、コードが `CHECKS` にある場合、真の値が再計算され、4 つのバケットに分割されます。

|バケツ |意味 |
|---|---|
| `measurement-confirmed and labelled` |それは true であると計算され、ラベルも | と書かれています。
| `unlabelled-but-correct` |計算は正しいですが、ラベルにはそれが記載されていませんでした (PTB-XL は、医師がそれについて言及していないと報告したため、モデルのエラーとはみなされません)。
| `measurement-contradicted` |計算上は誤りです——**本当の間違い** |
| `not-checkable` |対応する `CHECKS` 関数 (LVH/RVH/WPW などの形態学的ファミリー) が存在しないか、レコードに特性データがありません。

使用法: `python3 score_measurement_verifiable.py qwen_agent_output/deepseek_v38_diverse50 qwen_agent_output/qwen_v38_diverse50`

---

<a id="4-第三层-b乐观口径两步修正临时脚本本次现算未入库"></a>
## 4. 3 番目のレイヤー B: 楽観的な 2 段階の修正 (今回計算された一時的なスクリプト、データベースには含まれていません)

どちらのステップも、第 1 レベルの 4 グリッド テーブルに基づいており、「表面ラベル F1」から 2 種類の既知の会計ノイズを除去し、「実際の誤判定率」の楽観的な推定値を取得することを目的としています。

<a id="41-第一步验证过的同义代码合并"></a>
### 4.1 ステップ 1: 検証済みの同義コードをマージする

裸の単語のバリアントの 3 つのグループが `prediction_codes` コレクションに手動で追加され、意味上の同等性が 1 つずつ検証されました。

```python
ALIAS_ADD = {
    "atrial_fibrillation":     {"atrial_fibrillation"},
    "first_degree_av_block":   {"first_degree_av_block", "possible_first_degree_av_delay"},
    "complete_av_block":       {"complete_av_block"},
}
```

**マージするかどうかを決定する方法**: 候補エイリアスを個別に追加し、ルール ベースライン (非 LLM、安定した動作、A/B 比較に適した) の 4 セル テーブルを再計算し、TP/FP の変更を確認します。

|候補別名 |ベースライン ΔTP |ベースラインΔFP |結論 |
|---|---|---|---|
| `bradycardia` → `sinus_bradycardia` | +1 | **+5** |拒否 - 同じ所見の別の言葉ではなく、より広義の「心拍数の低下」を指します。
| `tachycardia` → `sinus_tachycardia` | +3 | **+4** |拒否されました、上記と同じ |
| `possible_lvh_voltage` → `lvh_voltage_criteria` | +0 | **+3** |拒否されました。意味的に同等ではありません |
| `atrial_fibrillation` → `atrial_fibrillation_pattern` ファミリー | 0 | 0 | **採用** |
| `first_degree_av_block`/`possible_first_degree_av_delay` → `first_degree_av_delay` ファミリー | 0 | +1 | **採用** (FP 増分は無視できる程度であり、測定検証の 2 番目のステップでさらにクリーンアップできます) |
| `complete_av_block` → `complete_av_block_pattern` | 0 | 0 | **採用** |

原則: **候補のエイリアスをマージする前に、ルール ベースラインで A/B テストを実行します**。 FP 増分が TP 増分よりも大幅に大きい場合、2 つのコードが同じものを参照しておらず、マージできないことを意味します。

<a id="42-第二步测量复核剔除假-fp"></a>
### 4.2 ステップ 2: 誤った FP を排除するための測定とレビュー

マージの最初のステップの後に新しく生成された各 FP について、それをトリガーしたコードを 3 番目のレイヤー A の `CHECKS` 関数にスローし、それを使用して再検証のために独自の特徴を記録します。検証が真の場合 (測定によって確認されたが、ラベルが書き込まれていない)、FP は統計から直接削除されます (TP または FP としてカウントされません)。検証が偽である場合、またはコードが検証できない場合、そのコードは FP として保持されます。

疑似コード:

```python
for record in records:
    for spec in direct_scope_categories:
        ref_pos = bool(record.ref_codes & spec.ptbxl_refs)
        pred_codes = spec.prediction_codes | ALIAS_ADD.get(spec.key, set())
        matched = record.agent_codes & pred_codes
        pred_pos = bool(matched)
        if ref_pos and pred_pos:
            TP += 1
        elif pred_pos:
            if any(CHECKS.get(c) and CHECKS[c](record.features) is True for c in matched):
                pass  # 剔除，不计入 FP
            else:
                FP += 1
        elif ref_pos:
            FN += 1
```

---

<a id="5-结果总表"></a>
## 5. 結果の要約表

<a id="51-micro-f1原始-vs-乐观修正"></a>
### 5.1 Micro-F1 (オリジナル vs 楽観的に修正)

|出典 |キャリバー | TP | FP | FN |精度 |思い出す | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
|ルールのベースライン |オリジナル | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
|通常のベースライン |楽観主義修飾子 | 23 | 28 | 23 | 0.451 | 0.500 | **0.474** |
| DeepSeek |オリジナル | 12 | 17 | 34 | 0.414 | 0.261 | 0.320 |
| DeepSeek |楽観主義の修正 | 12 | 9 | 34 | 0.571 | 0.261 | **0.358** |
|クウェン |オリジナル | 9 | 19 | 37 | 0.321 | 0.196 | 0.243 |
|クウェン |楽観主義の修正 | 11 | 9 | 35 | 0.550 | 0.239 | **0.333** |

> クウェンの楽観的修正後、TP は 9→11 に変更されました (FP の減少だけではありません): `first_degree_av_block`/`atrial_fibrillation` 裸の単語がマージされた後、2 つのレコードが FN から実際の TP に変更されました (元のモデルは正しかったが、コード文字列が認識されませんでした)。

<a id="52-五种平均口径对照direct-家族"></a>
### 5.2 5 つの平均キャリバーの比較 (直系)

|キャリバー |ルールのベースライン | DeepSeek |クウェン |ルールのベースライン (楽観的) | DeepSeek (楽観的) |クウェン (楽観的) |
|---|---:|---:|---:|---:|---:|---:|
|マイクロF1 | 0.426 | 0.320 | 0.243 | 0.474 | 0.358 | 0.333 |
|マクロ-F1 (平等な権利を持つ 17 家族) | 0.452 | 0.188 | 0.154 | 0.471 | 0.211 | 0.197 |
|マクロ-F1 (真の積極的なサポートがあるのは 16 家族のみ) | 0.480 | 0.200 | 0.164 | — | — | — |
|サンプル中心の F1 (PTB-XL/CAFA 式) | 0.425 | 0.347 | 0.261 | 0.469 | 0.389 | 0.352 |
|サブセットの精度 | 0.280 | **0.320** | 0.300 | — | — | — |
|平均的なジャカード | 0.412 | 0.387 | 0.370 | — | — | — |
|アドホック サンプル マクロ F1 (0/0→0、保守的) | 0.277 | 0.190 | 0.133 | — | — | — |
|アドホック サンプル マクロ F1 (0/0→1、ルーズ) | 0.457 | 0.410 | 0.393 | — | — | — |

補足統計 (元の基準、上記の表の分母を理解するために使用): 50 レコードのうち **21** には、直接ファミリーに空の実タグがあります (参照タグにヒットする直接ファミリーはありません)。このうち、「予測も空、実数も空」(些細な一致) のレコード数: ルール ベースライン 9 件、DeepSeek 11 件、Qwen 13 件。

**重要なポイント**: DeepSeek は、すべての口径において Qwen よりも常に優れています。 2 つのモデルに対するベースラインの先行振幅は、口径に非常に敏感です。Macro-F1 のベースラインは 2 つのモデルの約 2.2 ～ 3 倍であり (どちらのモデルも 6 つの伝導ブロック ファミリの再現率が 0% で、均等に増幅されます)、マイクロ/サンプル中心では 1.2 ～ 1.6 倍に圧縮されます (これらのファミリーはサンプルが少なく、平均して希釈されています)。サブセット精度 DeepSeek では、逆に、ルール ベースラインの方が「話しやすい」(特に LVH ファミリには多数の誤検知がある) ため、ベースライン (0.320 対 0.280) を超えています。 「ファミリー セットが完全にヒットしているかどうか」をレコードごとに確認すると、あと 1 ～ 2 単語のせいでフルスコアを見逃してしまいがちです。

<a id="53-测量复核结果"></a>
### 5.3 測定レビュー結果

|モデル |確定診断の総数 |チェック不可 |ラベルなしですが正しい |確認済み、ラベル付き |検証可能な診断の数 |検証されたエラーの数 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 59 | 34 (57.6%) | 23 (39.0%) | 2 (3.4%) | 25 | **0 (0.0%)** |
|クウェン | 74 | 44 (59.5%) | 28 (37.8%) | 2 (2.7%) | 30 | **0 (0.0%)** |

<a id="54-分诊断家族召回率13-个有信号的-direct-家族原始口径"></a>
### 5.4 ポイントの診断家族想起 (13 のシグナル伝達された直系家族、オリジナルの基準)

|家族 | n (真陽性の数) |ルールのベースライン | DeepSeek |クウェン |
|---|---:|---:|---:|---:|
|洞性徐脈 | 2 | 50% | 50% | 0% |
|洞性頻脈 | 4 | 25% | 25% | 25% |
|心房細動 | 4 | 75% | 50% | 0%* |
|心房性期外拍 | 2 | 100% | 0% | 0% |
|心室性期外収縮 | 5 | 60% | 60% | 60% |
|右脚ブロック RBBB | 6 | 33% | 0% | 0% |
|左脚ブロック LBBB | 2 | 50% | 0% | 0% |
|左前枝ブロック LAFB | 2 | 100% | 0% | 0% |
|左後筋膜ブロック LPFB | 1 | 100% | 0% | 0% |
|非特異的な心室内伝導遅延 | 4 | 25% | 0% | 0% |
|心室前興奮 WPW | 1 | 100% | 100%* | 100% |
|左心室肥大電圧基準 | 4 | 50% | 100% | 100% |
|右心室肥大 RVH | 3 | 100% | 0% | 0% |

`*` 心房細動 Qwen の 0% はノイズとして考慮されます (§6.2 を参照)。 WPW DeepSeek の適合率は、再現率 100% 未満でわずか 0.5 です (§6.1 のケースを参照)。

**構造的結論**: RBBB/LBBB/LAFB/LPFB/非特異的 IVCD/RVH/心房性期外収縮、合計 7 家族、合計 20/46 (約 43%) の真陽性、両モデルの再現率は 0% - これは誤った判断ではなく、これらの診断コードが現在のエージェントの出力にまったく表示されないためです。アカウンティング/ラベリングの修正によってこの部分を変更することはできません。パイプライン自体を修正する必要があります (§7 既知のバグ リストを参照)。

---

<a id="6-案例分析逐条核实"></a>
## 6. 事例分析（項目ごとに検証）

<a id="61-09275hr-deepseek-独有-wpw-假阳性"></a>
### 6.1 09275_hr — DeepSeek 排他的な WPW 誤検知

- 参照ラベル: `CRBBB|LAFB|RVH|SR` (完全右脚ブロック + 左前束ブロック + 右心室肥大、二束ブロック)、WPW なし。
- ルールのベースライン: `lafb_pattern|rvh_pattern|secondary_t_wave_abnormality`、これも WPW なし。
- DeepSeek 固有の診断: `ventricular_preexcitation_pattern`、信頼度が高く、参照 `ev:/rhythm_inputs/preexcitation/short_pr_interval` および `delta_lead_count=3`、グローバル `pr_ms=141`。
- 自身を記録します `metadata.rhythm_analysis.rule_summary.preexcitation` (`ptbxl_09000_ecgfeat/features/09275_hr_features.json`): `wpw_pattern: false`、`short_pr_interval: false` (141ms は通常の範囲 120 ～ 200ms であり、短い PR ではありません)、`delta_lead_count: 3` < `delta_threshold: 8`。 `feature_extraction/ecgfeat/clinical_rules/preexcitation.py::evaluate_preexcitation()` を置き換えて、`status="not_matched"` を取得します。
- **プログラム独自の確実性基準は偽として計算され、フィールドはモデルによって参照される証拠パス内にあります。このモデルでは、依然として反対の結論と高い信頼性が得られます。 **
- コントロール: Qwen は同じレコードで `left_axis_deviation` を与えましたが、誤解はありませんでした。両方のモデルは、実際の WPW ケース `09699_hr` (精度率 1.0/1.0) で正しく確認されました。

<a id="62-09114hr-09592hr-09913hr-09570hr-09136hr-09550hr-诊断代码同义词分裂"></a>
### 6.2 09114_hr / 09592_hr / 09913_hr / 09570_hr / 09136_hr / 09550_hr - 診断コードの同義語の分割

`evaluate_target_ecgfeat_diagnosis.py` の一部のファミリーの `prediction_codes` のバリアントは 1 つだけ登録されています。モデルが未登録の同義バリアントを使用している場合、検出漏れとして記録されます。

|記録 |モデル |実際の出力コード |スコアリングによって識別されるコードのセット |結果 |
|---|---|---|---|---|
| 09114_hr |クウェン | `atrial_fibrillation` | `atrial_fibrillation_pattern`/`probable_*` |心房細動は正しく判定されましたが、一致数が 0 としてカウントされました。
| 09592_hr |クウェン | `atrial_fibrillation` |同上 |上記と同様に、AFIB ファミリのリコールを直接引き下げます。
| 09913_hr |クウェン | `first_degree_av_block` | `first_degree_av_delay` |第一級部屋遅延の判定は正しかったが、得点されなかった |
| 09570_hr | DeepSeek | `first_degree_av_block` | `first_degree_av_delay` |同上 |
| 09136_hr | DeepSeek | `t_wave_abnormality` | `primary_/secondary_t_wave_abnormality` | T 波異常はスコア化されていない (広範なグループ、ヘッドライン F1 には影響しない) |
| 09550_hr |クウェン | `possible_lvh_voltage` | `lvh_voltage_criteria` | LVH プロンプトがスコア化されません |

このサンプル: Qwen は 4 回ヒットし、DeepSeek は 2 回ヒットしました。両方の側が影響を受けましたが、今回は Qwen がさらにヒットし、そのうち 2 回は AFIB ファミリーが持つべきリコール率を直接遮断しました。

<a id="63-09114hr-两模型都成功剪除规则引擎历史噪声"></a>
### 6.3 09114_hr — どちらのモデルもルール エンジンの履歴ノイズを正常にトリミングしました

ルールのベースラインでは 9 つのコードが与えられ、そのうち `prior_infarct_q_wave_pattern` + `rvh_pattern` + `lvh_voltage_criteria` が同時に出現します (以前に記録された「病的な Q 波の誤判定を引き起こす高振幅 QRS」の組み合わせ)。 DeepSeek と Qwen はどちらも、9 個のコードを 2 個 (心房細動 + LVH) に削減しており、参照ラベル `AFIB|INVT|NDT|PVC|STD_` と非常に一致しています。サンプルサイズは限られており (n=1)、欠陥の体系的な修復を表すものではありません。

<a id="64-09570hr-deepseek-识别陈旧心梗-q-波qwen-漏检"></a>
### 6.4 09570_hr — DeepSeek 陳旧性心筋梗塞の Q 波を特定、Qwen の検出漏れ

参照ラベルには、`ASMI|ILMI` (前中隔 + 下側壁陳旧性心筋梗塞) が含まれています。 DeepSeek は、ラベルと直接一致する `prior_infarct_q_wave_pattern` を出力します。 Qwen はこのコードを与えません。

---

<a id="7-已知缺陷复查状态"></a>
## 7. 既知の欠陥のレビューステータス

|既知の問題 |ステータス |このサンプル証拠 |
|---|---|---|
|伝導ブロックファミリー (RBBB/LBBB/LAFB/LPFB/IVCD/RVH) の合計検出ミス |まだ存在します |通常のベースラインの 25% ～ 100% とは対照的に、両方のモデルの 6 つのファミリーすべての再現率は 0% であり、その性質は以前のポジショニングと一致しています。
|虚血・心筋梗塞スクリーニングの再現率は極めて低い |まだ存在します | DeepSeek 虚血 0/18、心筋梗塞 1/16。クウェン虚血 1/18、心筋梗塞 0/16。通常のベースライン 7/18、4/16 |
|診断コードの同義語の分割 |まだ存在します |今回新たに確認された 6 つの特定のインスタンス (§6.2) |
| LVH ゲート カバレッジは古いルール エンジンより狭い |部分的な改善 | F1 は 0.222 から 0.533 に増加しました。2 つのモデルは完全に一貫しており (プログラムされており)、7 つの FP は 1 つずつ検証されていません。
| Q 波偽陽性/高振幅 QRS は病的な Q 波の誤判断を誘発します。今回は新しい現象は見つかりませんでした。 09114_hr どちらのモデルもこのノイズの組み合わせをアクティブにカットし、サンプルは制限されています (n=1)。
| WPW/事前励起パスにモデルがない - プログラムの一貫性チェック | **新しい発見** | 09275_hr: DeepSeek は、プログラム基準が明らかに false の場合でも高い信頼度を与えます。
| sinus_rhythm ゲート停滞/AF モデルオーバーライド |このサンプルはトリガーされません |関連する証拠は観察されず、サンプルサイズは小さいため、修復されたことを意味しません。

---

<a id="8-运行效率对比"></a>
## 8. 作業効率の比較

|指標 | DeepSeek |クウェン |
|---|---:|---:|
|中央時間 / P90 | 22.058秒 / 28.316秒 | 87.734秒 / 109.671秒 |
|合計ツール呼び出し数 / レコード平均 | 495 / 9.9 | 507 / 10.14 |
|トークン: プロンプト / 完了 / キャッシュ | 617,993 / 60,756 / 286,336 | 427,957 / 71,790 / 0 |
|検証合格率 / 改訂ラウンド | 50/50、0 | 50/50、0 |
|記録_成績（Q0/Q1） | 46/4 | 46/4 |
|ベースラインと比較したレコードごと: 改善/不変/悪化 | 6月23日/21日 | 5月22日/23日 |

DeepSeek は約 4 倍高速で、多数のプロンプト キャッシュにヒットします (Qwen 側ではキャッシュ ヒットはありません)。

---

<a id="9-跨论文数字对比的坑重要避免误用"></a>
## 9. 論文間で数値を比較する際の落とし穴 (重要、誤用を避ける)

PTB-XL F1 は、公開された ECG-LLM/エージェントの論文 (例: CARE-ECG、ECG-Chat) で報告されており、**NORM (正常) が含まれる可能性が最も高く、SR (洞調律) などの「正常」ラベルがスコアリング可能なカテゴリとして使用されます**。また、そのようなラベルの再現率は当然のことながらこれは、集約インデックスを大幅に増加させます。

- ECG-Chat 論文の元の言葉 (図 4 の説明): 「『ノーマル (NORM)』、『サイナス リズム (SR)』などのいくつかの一般的なラベルは、高い F1 スコアを持っています...多くのラベルの F1 スコアは 0 です。」
- ECG-Chat は論文で CE 疾患 (NORM を含む) F1 = 22.33% を報告しました。 CARE-ECG は、同じ「ECG-Chat」モデルをベースラインとして使用し、論文では同じ PTB-XL タスクを使用しましたが、F1 = 0.70 を報告しました - 同じモデル、同じデータセット、異なるスコアリングプロトコルにより、結果として 3 倍以上異なる数値が得られました。
- 当社独自の `evaluate_target_ecgfeat_diagnosis.CATEGORIES` **NORM/正常カテゴリー** がなければ、「すべてが正常」というモデルは TP を獲得することはなく、実際の異常を見逃した場合にのみ減点されます。

**結論**: この文書の §5 のすべての数値は、「当社独自の 3 つのソース (ベースライン/DeepSeek/Qwen) 間」でのみ水平的に比較できます。相手のスコアリングカテゴリセットに NORM/SR が含まれているかどうか、およびマクロ/ミクロ/サンプル中心のどの平均口径が使用されているかを最初に確認しない限り、それらを外部論文の F1/精度数値と直接比較しないでください。

---

<a id="附录-a本次涉及的-17-个-direct-scope-家族完整列表"></a>
## 付録 A: この記事に関係する 17 の直接スコープ ファミリの完全なリスト

```
sinus_bradycardia, sinus_tachycardia, atrial_fibrillation, atrial_flutter,
premature_atrial_complexes, premature_ventricular_complexes,
first_degree_av_block, complete_av_block,
right_bundle_branch_block, left_bundle_branch_block,
left_anterior_fascicular_block, left_posterior_fascicular_block,
nonspecific_ivcd, ventricular_preexcitation,
left_ventricular_hypertrophy, right_ventricular_hypertrophy,
prolonged_qt
```

(`unspecified_ectopy_pattern`、`ambiguous_left_conduction_family`、`atrial_abnormality`、`low_qrs_voltage`、`t_wave_abnormality` は `broad` スコープです。`ischemia_or_st_abnormality`、`infarction_or_q_wave` は `screening` です。範囲; この文書の §5 の見出し指標には含まれません)。

<a id="附录-b复现命令"></a>
## 付録 B: 再生コマンド

```bash
# 测量复核（§3）
python3 score_measurement_verifiable.py \
  qwen_agent_output/deepseek_v38_diverse50 \
  qwen_agent_output/qwen_v38_diverse50

# 第一/二层四格表与各平均口径、第三层B乐观修正：
# 均为本次分析中现写的一次性脚本，未入库；核心逻辑见本文档 §1、§2、§4 的伪代码，
# 依赖 evaluate_target_ecgfeat_diagnosis.CATEGORIES 与 score_measurement_verifiable.CHECKS，
# 输入取自各 run 目录下的 record_results.csv 与 features/*.json。
```

<a id="附录-c相关项目记忆"></a>
## 付録 C: 関連するプロジェクト メモリ

- `project_v38_diverse50_deepseek_vs_qwen.md`
- `project_deepseek_wpw_gate_override.md`
- `project_diagnosis_catalog_synonym_split.md`
- `project_measurement_verifiable_scoring.md`
- `project_ecg_llm_literature_f1_not_comparable.md`
- `project_conduction_block_total_miss.md`
