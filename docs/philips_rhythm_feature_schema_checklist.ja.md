<!-- i18n-nav -->
[中文](philips_rhythm_feature_schema_checklist.md) | [English](philips_rhythm_feature_schema_checklist.en.md) | [日本語](philips_rhythm_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="philips-rhythm-feature-schema-checklist"></a>
# Philips Rhythm 機能スキーマ チェックリスト

<a id="结论先行"></a>
## 結論を先に

現在のフル機能 JSON (例: `JS00024_features.json`) には、リズム分析に必要な多くの測定機能がすでに含まれています。

- R/QRS 位置と RR 間隔
- ビートレベル P/QRS/T ボーダー
- PR/QRS/QT/JT/QTc
- P/QRS/T/ST軸
- P 波の信頼性、P 振幅、P 面積
- Q/R/S振幅、QRSエリア、QRSノッチ/スラー、デルタフラグ
- リードごとの品質と波形固有の信頼性
- ビートグループ化、ドミナントグループ、グループ平均 RR/PR/QRS/QT

ただし、`要点.pdf` のフィリップス スタイルのリズム分析の入力要件をすべて完全に満たすことはできません。主なギャップは以下に集中しています。

- QRST subtraction と AF/AFL の心房残留信号機能
- ペーシングの連続/断続/チャンバー/アーティファクト/キャプチャ/センスの細分化機能
- P 波 > QRS 群、ドロップビート、第 2 度 AV ブロックの独立した P 波イベント ストリーム
- ビートレベル P 形態、QRS 極性サイン、代償性一時停止、および早漏分類のその他の証拠
- 前興奮のデルタ誘導数、short-PR 閾値の減少、副経路側
- ルールエンジンのステートメント証拠スキーマ

<a id="数据层级判断"></a>
## データレベル判定

|データレベル | `要点.pdf` リズム分析をサポートできるかどうか |説明 |
| --- | --- | --- |
| `.txt report` |十分ではありません |人間による読み取りに適しており、HR/RR/PR/QRS/QT/軸/品質/リードごとの中央値のみが表示され、完全なビートレベル機能はありません。 |
|再生成された `JS00024_features.json` などのフル機能 JSON | MVP をサポートでき、多数のフィールドが利用可能です。 `beats`、`beat_features`、`representative_leads`、`groups`、`global_features`、`metadata`、`interpretation`、および新しい `rhythm_inputs` が含まれます。 |
| `build_structured_payload()` は現在エクスポートされています |十分ではありません |現在、`signal / quality / beats / groups / global / provenance` のみがエクスポートされ、`beat_features`、`representative_leads`、および `interpretation` はエクスポートされません。 `feature_extraction/ecgfeat/export.py:19-42`を参照してください。 |

<a id="状态定义"></a>
## 状態の定義

- `OK`: 現在、JSON の全機能が提供されており、ルール分析に直接使用できます。
- `Partial`: 関連フィールドはありますが、`要点.pdf` の完全なルールをサポートできるほど安定していません。
- `Missing`: 直接使用できるフィールドまたは派生機能は現在見つかりません。

<a id="feature-checklist"></a>
## 機能チェックリスト

|モジュール | `要点.pdf` 必要な機能 |現在利用可能なフィールド |ステータス |ギャップ/リスク |
| --- | --- | --- | --- | --- |
|基本入力 |サンプリングレート、リードシーケンス、レコード長 | `fs`、`metadata.input_fs`、`metadata.internal_fs`、`metadata.lead_order` | OK |フル JSON はい;薄型ペイロードも利用可能です。 |
|患者情報 |年齢、性別 | `metadata.patient_meta.age`、`metadata.patient_meta.sex` | OK |小児のしきい値には、依然として外部の満年齢テーブルが必要です。 |
|信号品質 |リードごとの品質、P/QRS/T/QT の可用性 | `quality.*.reliable_for_p/qrs/t/qt`、`flags`、`baseline_wander_score`、`muscle_noise_score`、`powerline_score` | OK | `JS00024_features.json` `record_quality` には新しいバージョンはありません。現在のコードは `metadata.record_quality` でサポートされています。 |
| R/QRS 検出 | R ピーク / QRS 位置 | `beats[*].r_index`、`metadata.qrs_detector.r_locs` | OK | RR、心室心拍数、一時停止、初期心拍をサポートします。 |
| RR特性 | `rr_prev_ms`、`rr_next_ms`、平均 RR、RR 変動 | `beats[*].rr_prev_ms`、`beats[*].rr_next_ms`、`groups[*].mean_rr_ms`、`interpretation.rr_cv` | OK |バックグラウンド RR は現在中央値/平均値を使用しており、「異音/一時停止/ペーシングを除くクリーンなバックグラウンド RR」を明確に出力していません。
|心室心拍数 |心室心拍数 | `global_features.heart_rate_bpm`、`groups[*].mean_ventr_rate_bpm` | OK |タキー/ブレイディ/完全な AVB レート条件をサポートできます。 |
|心房レート |心房レート | `global_features.atrial_rate_bpm` |部分的 | P 検定から推定される単一の値です。一連の独立した心房イベントと自信が欠けています。 |
| P波検出 |ビート/リードごとのオンセット/ピーク/オフセット | `beat_features[*].p.onset`、`p.peak`、`p.offset` | OK | QRS 中心の P 検出は利用可能ですが、ドロップされた P または独立した P イベント ストリームを検出するには十分ではありません。 |
| P波の存在 | P 波動 / P 信頼度がある | `beat_features[*].p_confidence`、`representative_leads.*.params.p_confidence_mean` | OK |現在のルールでは、しきい値を使用して「P があるかどうか」を判断できます。 |
| P 波の形態 |正常 / 非定型 / 逆行性 / 異所性 P | `interpretation.p_morphology_class`、P 振幅/面積/持続時間 |部分的 |現在の`p_morphology_class`は部分的な心房拡大であり、APC/JPCが要求する拍動レベルの非定型P形態ではありません。 |
| P軸 |前方 P 軸 | `global_features.p_axis_deg`、`interpretation.p_axis_normal` | OK |洞由来判定をサポートできます。 |
| PR 間隔 | 拍/導联/全局の PR | `beat_features[*].pr_ms`, `representative_leads.*.params.pr_ms`, `global_features.pr_ms` | OK | AVB1、短 PR、Wenckebach の初期判定をサポート可能。 |
| PR 区間 | PR 区間の持続時間 | 直接フィールドなし | Missing | WPW ルールにおいて `PR segment < 55 ms` は直接判定不能；`baseline_source="pr_segment"` だけでは持続時間ではない。 |
| PR 傾向 | 一時停止前の PR 系列 | `beat_features[*].pr_ms` + 拍順序から派生可能 | Partial | 現在派生可能だが、拍ごとのグローバル PR 系列/出所が明示的に出力されない。 |
| QRS 持続時間 | 拍/導联/全局の QRSd | `beat_features[*].qrs_ms`, `representative_leads.*.params.qrs_ms`, `global_features.qrs_ms` | OK | 正常/広幅 QRS、VPC/JPC/APC の粗分類をサポート可能。 |
| QRS 波形 | Q/R/S 振幅、R prime、S prime、ノッチ/スラー、VAT | `q_amp_mv`, `r_amp_mv`, `s_amp_mv`, `r_prime_amp_mv`, `s_prime_amp_mv`, `qrs_notch_count`, `qrs_slur_flag`, `vat_ms` | OK | 波形情報が豊富で、極性/特徴のさらなる分析に十分。 |
| QRS 極性特徴 | 導联ごとの極性ベクトル / 波形ベクトル | Q/R/S 振幅と符号付き面積から派生可能 | Partial | `qrs_polarity_signature` または拍テンプレート埋め込みが直接出力されない。 |
| QRS 符号付き面積 | 符号付き QRS 面積 | 内部モデルに `qrs_signed_area` あり；現在の `JS00024_features.json` 最初の beat_features キーリストには表示されない | Partial | コードモデルにはフィールドがあるが、現在の JSON は古いエクスポートから生成されている可能性があり、新しいエクスポートに含まれるか確認が必要。 |
| デルタ波 | 拍/導联ごとのデルタフラグ | `beat_features[*].delta_present` | OK | WPW の初期スクリーニングをサポート。 |
| デルタ導联数 | デルタ波がある導联の数 | `beat_features[*].delta_present` から集計可能 | Partial | 導联数、信頼度、短 PR 閾値低下に必要な設定証拠が直接出力されない。 |
| 初期 QRS 軸 | 経路側のための初期 QRS 軸 | 直接フィールドなし | Missing | グローバル QRS 軸のみで、左/右付加経路の判定には不十分。 |
| QT/QTc | QT、QTcB、QTcF | `global_features.qt_ms`, `qtc_bazett_ms`, `qtc_fridericia_ms`、導联ごとの QT/QTc から派生可能 | OK | リズムのメインラインは核心的なギャップではない。 |
| 拍グループ化 | グループ ID、支配的グループ、グループ指標 | `beats[*].group_id`, `groups`, `metadata.representative_group_id` | OK | 支配的リズム / 異所性グループの粗分析をサポート。 |
| ペースドフラグ | 拍レベルのペースド | `beats[*].paced`, `metadata.paced_beat_ids` | Partial | `enable_pacing=True` の場合のみ信頼性あり；通常呼び出しでは無効化されている可能性がある。 |
| ペーシングスパイク | スパイクタイムスタンプ | `global_features.pacing_spikes`, `metadata.pacing_state` | Partial | `JS00024_features.json` は `null/off`；デフォルトパイプラインでは通常ペーシングが有効化されていない。 |
| ペーシングタイプ | 心房 / 心室 / デュアル / AV 順次 | 直接フィールドなし | Missing | `PACE-001/002/006/007` の細分化制御を満たせない。 |
| 連続/間欠ペーシング | continuous_pacing、intermittent_pacing | 直接フィールドなし | Missing | 現在拍タグ付けのみで、非ペースドリズムパターン分析の停止には不十分。 |
| デマンド動作 | パルス抑制の証拠 | 直接フィールドなし | Missing | デマンドペーシング動作の特徴なし。 |
| ペースメーカーアーティファクト | アーティファクト信頼度 / ノイジーなスパイクの証拠 | 独立したフィールドなし | Missing | 現在の `pacemaker_like_artifact` は独立した測定証拠ではない。 |
| センス/キャプチャ失敗 | スパイク-QRS/P の脱結合、固定レート非同期スパイク | 直接フィールドなし | Missing | マグネット/センス失敗/キャプチャ分析の入力なし。 |
| AF RR 特徴 | RR CV、不規則性、RMSSD/ローレンツ/エントロピー | `interpretation.rr_cv`, `rr_irregularity_class`；RR 系列から派生可能 | Partial | CV はある；RMSSD、エントロピー、ローレンツ系特徴は出力されない。 |
| AF P 波の証拠 | 組織化された P 波の欠如 | `p_confidence`, `probable_af` | Partial | 粗判定は可能だが、心房残留信号のサポートがない。 |
| QRST subtraction | 心房残留信号 | 直接フィールドなし | Missing | これは AF/AFL の安定した区別の核心的なギャップ。 |
| 心房波形の安定性 | 残留安定性指標 | 直接フィールドなし | Missing | `要点.pdf` に基づいて AF/AFL を区別できない。 |
|心房反復性 |残留反復/自己相関/周波数特徴 |直接フィールドが見つかりません |欠落しています | AFL は基本的な欠落入力を検出します。 |
|フラッターの証拠 |粗動波の信頼性 / 心房周期長 |直接フィールドが見つかりません |欠落しています |現在、明示的な心房粗動機能はありません。 |
| APC/JPC/VPC 初期ビートの証拠 | RR短縮とバックグラウンドRR | `beats[*].rr_prev_ms`;バックグラウンド RR はグループから派生できます。 OK | 15% 早期ビートスクリーニングには十分です。 |
| APC の証拠 |早期 + 正常 QRS + 非定型 P 形態 | RR + QRS + P の信頼性が利用可能 |部分的 |拍動レベルの非定型 P 形態が欠落しています。 |
| JPC 証拠 |初期+通常 QRS+Pなし | RR + QRS + P の信頼性が利用可能 | OK | 「no P」タイプの JPC の初期スクリーニングに十分です。 |
| VPC の証拠 |アーリー + ワイド QRS + 極性違い + 補償ポーズ | RR + QRS が利用可能です。 QRS 形態を導出可能 |部分的 |明確な極性の違い、代償的な一時停止、通常のテンプレートの比較の欠如。 |
| VPC ペア/NSVT |連続Vビートラン |ビートクラスから派生できます。現在の解釈は `non_sustained_vt` |部分的 |現在のビート クラスはエクスポートされず、最終フラグのみがエクスポートされます。証拠の連鎖は不完全です。 |
|二連/三連 | N/V または N/A シーケンス パターン |現在の解釈には、`bigeminy`、`trigeminy` フィールドがあります。 `JS00024_features.json` の古いバージョンにはこれらのフィールドがありません。部分的 |新しいモデルにはフィールドがありますが、現時点では JSON には 37 の解釈フィールドのみが含まれており、高度なリズム フィールドは含まれていません。 |
|一時停止 | RR > 140% バックグラウンド RR | `beats[*].rr_next_ms`、グループ平均 RR;新しい解釈には一時停止フィールドがあります。部分的 |現在の `JS00024_features.json` と古いバージョンでは、`pauses_detected/pause_longest_ms` はエクスポートされません。 |
|エスケープビートオリジン | P 存在 + 一時停止後の QRSd | P信頼度＋QRSdは事前審査に利用可能 |部分的 |特別な一時停止後のエスケープ ビート オブジェクトとオリジン ラベルの欠如。 |
|第二級 AVB | P 波 > QRS 複合体 |独立した P イベントが見つかりません |欠落しています | QRS 中心の P 検出では、ブロックされた P を確実にカウントするには不十分です。
|モビッツI |ドロップビート前のプログレッシブ PR 延長 | PR シーケンスを導出することができます。新しい解釈は `second_degree_avb` |部分的 |ドロップされたビート/P のみのイベントの証拠が欠落しています。 |
| AV解離 |心室心拍数は正常 + 見かけの PR 変動 / AV 非同期 | PR シーケンス、心房レート、心拍数が利用可能 |部分的 |明示的な AV 非同期スコアと独立した心房/心室イベント ストリームが欠落しています。 |
|完全な AVB |心室心拍数 <45 + AV 非同期 |人材採用可能 |部分的 | AV 非同期スコアがありません。 |
|補間されたビート |前/次の RR は約 0.5 背景 RR | `rr_prev_ms`、`rr_next_ms` 利用可能 |部分的 |派生可能ですが、直接のフィールド/フラグはありません。 |
|異常なコンプレックス |わずかなRR短縮+ワイド QRS | RR + QRSd が利用可能 |部分的 | 「わずかな短縮」の特定の閾値と形態の区別の証拠が欠けています。 |
|証拠の出力 |各リズムステートメントの証拠 | `rhythm_inputs.statement_evidence` および `clinical_interpretation.domains` |部分的 |現在公開されているルールのセットをカバーしました。フィリップスの完全なリズム ステートメント インベントリはまだ実装されていません。 |
|ルールの優先順位フラグ |停止/バイパス/抑制 | `statement_engine` および `clinical_interpretation.suppressed_statements` |部分的 |決定的な停止/抑制が可能です。完全なリズム エンジンの優先事項はまだフォローアップの作業です。 |

<a id="js00024-当前-json-快照"></a>
## JS00024 現在の JSON スナップショット

旧バージョン `JS00024_features.json` には、現在次のものが含まれていることが確認されています。

- トップレベル: `fs`、`quality`、`beats`、`beat_features`、`representative_leads`、`groups`、`global_features`、`metadata`、 `interpretation`
- ビート: 19、フィールドは `beat_id`、`r_index`、`paced`、`group_id`、`rr_prev_ms`、`rr_next_ms`
- Beat_features: 228 項目、P/QRS/T 境界、PR/QRS/QT、P/QRS/T 振幅/面積、デルタ、信頼性、ノッチ/スラー、VAT、TPE、ST 形態などをカバーするフィールド。
-代表リード: 12 個のリード、それぞれ `params` および `variance`
- global_features: HR、心房レート、PR/QRS/QT/QTc、軸、QT 分散、ペーシング、PTF-V1
- メタデータ: 入力/内部 fs、リード順序、リード反転、n_beats、qrs_detector、representative_beat_meta、paced_beat_ids、patient_meta
- 解釈: 古いバージョンには 37 のフィールドがあり、`premature_complexes`、`bigeminy`、`trigeminy`、`pauses_detected`、`complete_av_block`、`second_degree_avb` などの新しいバージョンの高度なリズム フィールドがありません。

フル機能の JSON を再生成した後、ダウンストリームのリズム ルール エンジン用にトップレベルの `rhythm_inputs` が追加で含まれます。

<a id="已新增的最小目标-schema"></a>
## 最小ターゲット スキーマを追加しました

`rhythm_inputs` の全機能 JSON は次のブロックに出力されています。現在、既存の測定値から安定して導出できるフィールドには、特定の値が与えられます。新しい信号アルゴリズムを必要とするフィールドには、明示的に `available=false` および `reason` が与えられます。

### `rhythm_inputs.record`

- `fs`
- `duration_sec`
- `age_years`
- `sex`
- `lead_order`
- `record_quality`
- `rejected_functions`

### `rhythm_inputs.beats[]`

- `beat_id`
- `r_index`
- `r_time_ms`
- `rr_prev_ms`
- `rr_next_ms`
- `group_id`
- `paced`
- `pacer_spike_index`
- `qrs_duration_ms`
- `pr_interval_ms`
- `p_confidence`
- `p_axis_deg`
- `p_morphology`
- `qrs_polarity_signature`
- `delta_leads`
- `beat_quality`

### `rhythm_inputs.p_events[]`

- `p_event_id`
- `time_ms`
- `confidence`
- `associated_qrs_beat_id`
- `association_type`: `conducted`、`blocked`、`retrograde`、`unknown`
- `pr_ms`
- `axis_deg`
- `morphology`

### `rhythm_inputs.background`

- `clean_rr_ms`
- `background_rr_regular`
- `background_ventricular_rate_bpm`
- `background_atrial_rate_bpm`
- `dominant_group_id`
- `excluded_beat_ids`
- `exclusion_reasons`

### `rhythm_inputs.pacing`

- `enabled`
- `spike_times`
- `spike_count`
- `paced_beat_ids`
- `continuous_pacing`
- `intermittent_pacing`
- `ventricular_pacing_present`
- `atrial_pacing_present`
- `dual_chamber_pacing_present`
- `artifact_confidence`
- `capture_failure_suspected`
- `sensing_failure_suspected`

### `rhythm_inputs.af_afl`

- `rr_cv`
- `rr_rmssd`
- `rr_entropy`
- `qrst_subtraction_quality`
- `atrial_residual_signal_summary`
- `atrial_signal_stability`
- `atrial_signal_repetitiveness`
- `dominant_atrial_cycle_ms`
- `flutter_wave_confidence`

### `rhythm_inputs.preexcitation`

- `short_pr_interval`
- `short_pr_segment`
- `delta_lead_count`
- `delta_leads`
- `delta_confidence_by_lead`
- `mean_qrs_duration_ms`
- `initial_qrs_axis_deg`
- `accessory_pathway_side`

### `rhythm_inputs.statement_evidence`

- `primary_statement`
- `additional_statements`
- `statement.code`
- `statement.category`
- `statement.priority`
- `statement.evidence`
- `statement.confidence`
- `statement.suppresses`
- `stop_further_interpretation`
- `bypass_remaining_algorithm`

<a id="实用判断"></a>
## 実際の判断

現在のフル機能の JSON は **MVP リズム ルール エンジン**として適しています。

- 副鼻腔炎/頻脈/徐脈
- 定期/不定期
- 第 1 度房室ブロック / PR 正常-短-長
- ワイド/ナロー QRS
- シンプルWPW
- 単純な APC/JPC/VPC
-二叉神経/三叉神経
- 簡単な一時停止

Glasgow 解釈レイヤーは、現在のランタイム中に実行またはエクスポートされなくなりました。監査可能な出力は次のようになります。
`statement_engine`、`clinical_interpretation.domains`と各ルール
証拠/閾値/ソースフィールドが提供されます。

現在のところ、完全なフィリップス スタイルのリズム エンジンの次の部分には適していません。

- 堅牢な AF と AFL
- ペースのあるリズムステートメントの制御フロー
- P波が遮断された第2度房室ブロック
-エスケープビートの原点
- AFのみの心房診断による心室ペーシング
- マグネット/キャプチャ/センシングの失敗
-副経路の局在化
