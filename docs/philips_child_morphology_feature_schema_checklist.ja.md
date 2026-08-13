<!-- i18n-nav -->
[中文](philips_child_morphology_feature_schema_checklist.md) | [English](philips_child_morphology_feature_schema_checklist.en.md) | [日本語](philips_child_morphology_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="philips-pediatric-morphology-feature-schema-checklist"></a>
# Philips 小児形態学機能スキーマ チェックリスト

<a id="结论先行"></a>
## 結論を先に

> 2026-07-12 ステータス更新: 統合権限層は大人/子供のルーティングをサポートしていますが、バージョン管理がありません。
> 公的に追跡可能な小児 PR および心室肥大パーセンタイル表、`pediatric_pr_public_reference_table`、
> `pediatric_lvh_percentile_table` と `pediatric_rvh_percentile_table` が明確に出力されます
>`unavailable`。既存の DXL に基づいた子しきい値は参照専用ペイロードに残ります。
> 権威ある最終結論には入らず、それに応じて「正常」も出力されません。

`philips_child_morphology.pdf` は、**誕生から 16 歳未満** の年齢向けのフィリップス DXL 第 4 章小児形態解析、ECG です。成人の形態との最大の違いは、子供の ECG の正常範囲が年齢に大きく依存しており、一部の規則は性別にも依存することです。

現在の測定結果とコードは、**小児形態学 MVP** の最初のバージョンをサポートできます。

- 患者の年齢/性別
- P/QRS/T/ST軸
- リードごとの P/QRS/T/ST 振幅と境界
- P持続時間、P振幅、PTF-V1
- Q/R/S/R'/S' 振幅
- QRS デュレーション、QRS ピークツーピーク、VAT、ノッチ/スラー/fQRS
- ST J 点/中間/80ms、T 振幅/極性
- QT/QTc
- 解釈レイヤーには、`is_pediatric`、年齢調整された QRS 軸、年齢調整された QRS 幅、小児 QTc、LSH、BVH、心膜炎、早期再分極などのフィールドがすでにあります。

この一連の変更の後、再生成された完全な機能 JSON では、`morphology_inputs` の成人/小児の形態コンテキストが明確に区別されます。

- `record.algorithm_age_group` は `adult` または `pediatric` を出力します。新生児から16歳未満までの小児。
- 年齢が欠落しているか無効な場合、フィリップスのドキュメントに従ってデフォルトで成人となり、`record.age_defaulted_to_adult_algorithm` でマークされます。
- `derived_facts.adult` 成人向けのしきい値コンテキストを出力します。
- `derived_facts.pediatric` 出力年齢バケット、小児軸 / QRS / QTc しきい値、小児右心筋症 / RAE / BBB / 肥大 / ST-T / QT ルールにはしきい値コンテキストが必要です。

しかし、フィリップス式の小児形態学エンジンを完全に満足させることはできません。

- Davignon の付録 A の年齢層別正常値の完全な表、特に RVH/LVH 電圧の 98 パーセンタイルしきい値が欠落しています。
- RBBB/IRBBB/LBBB には、R'/ターミナル 40ms/QRS コンポーネント期間が必要ですが、現時点では部分的にのみ近似されています。
- RVH には、軽度の RVH と IRBBB を区別するのに役立つ合成水平面ベクトル/終端角度が必要ですが、現在は利用できません。
- P 波の二相性/ノッチ/始端成分は、成人と同様にまだ不完全です。
- 出力層にはすでに統合されたステートメント/ステータス/証拠/抑制契約があります。子のパーセンタイル依存関係ルールは、パブリック テーブルがバージョン管理されておらず、完全なフィリップスのコード/ダウングレード マトリックスがまだ実装されていないため、意図的に決定不可能になっています。
- 先天性心疾患は組み合わせのアイデアを与えるだけであり、現在、体系化された先天性心疾患のルールはありません。

したがって、判断は次のようになります。 **現在の機能 JSON は、大人/子供向けの迂回と部分的な子供の測定をサポートできますが、公的標準の権限層は欠落テーブル ルールを保守的に拒否します。完全なフィリップス第 4 章では、法的に追跡可能な参照テーブル、ターミナル形態、および独自のステートメント マトリックスが依然として必要とされており、既存の参照ヒューリスティックによって偽装することはできません。 **

<a id="文档范围"></a>
## ドキュメントのスコープ

このリストはローカル PDF と比較されます。

- `/home/chtmedgemma/projects/ecg_gemma/philips_child_morphology.pdf`

ドキュメントの対象範囲:

- 小児のルーティングと年齢の処理
- 右心筋症
-レイ/レイ/ベイ
- 年齢調整済み QRS 軸
- 年齢調整済み VCD / RBBB / IRBBB / LBBB / LAFB
-RVH/LSH/LVH/BVH
- 低電圧 / COPD パターン
- Q波異常/MI
- ST低下/T波異常/再分極異常
- ST上昇 / 心膜炎 / 早期再分極
- 背の高いT
- 小児 QTc / 電解質障害
- 先天性心臓欠陥

<a id="当前代码与-json-状态"></a>
## 現在のコードと JSON ステータス

現在のコード層には多くの小児用拡張機能があります。

- `PEDS_MAX_AGE_YEARS = 16.0`
- 小児右心筋閾値
- 小児用RAE P振幅閾値 `0.20 mV`
- 年齢に応じた QRS 期間の通常制限
- 年齢に応じた QRS 軸テーブル
- 小児用 RBBB R' 振幅閾値
- 小児 QTc の年齢/性別しきい値
- LSH および BVH ヘルパー
- 心膜炎の年齢ゲート 5 ～ 15 歳
- 早期再分極年齢ゲート 13 ～ 15 歳
- `ECGInterpretation.is_pediatric`
- `ECGInterpretation.lsh_suspected`
- `ECGInterpretation.bvh_suspected`
- `ECGInterpretation.pericarditis_suspected`
- `ECGInterpretation.early_repolarization_suspected`

ただし、現在存在するサンプル JSON (`JS00029_features.json`、`JS00024_features.json`、`JS00007_features.json` など) はまだ古いエクスポートである可能性があり、患者のほとんどは成人であるため、次のようになります。

- 古い JSON には最新の `morphology_inputs.derived_facts.pediatric` が含まれていない可能性があります
- 古い JSON には最新の `record.algorithm_age_group` が含まれていない可能性があります
- 成人サンプルは小児ルーティングの検証には適していません
- フル機能の JSON を再生成して最新のスキーマを含めます

<a id="状态定义"></a>
## 状態の定義

- `OK`: 現在、JSON / モデルの全機能が提供されており、直接または安定して派生できます。
- `Partial`: 関連するフィールドまたは最初のバージョンのルールはありますが、経過時間のしきい値、詳細な測定値、ルールの証拠、または新しいエクスポートが欠落しています。
- `Missing`: 直接使用できる測定フィールドまたは構造が現在見つかりません。

<a id="feature-checklist"></a>
## 機能チェックリスト

|モジュール |フィリップスの小児形態学に必要な機能 |現在利用可能なフィールド/コード |ステータス |ギャップ/リスク |
| --- | --- | --- | --- | --- |
|小児科のルーティング |年齢 誕生から16歳未満まで 小児アルゴリズム | `metadata.patient_meta.age`; `is_pediatric` | OK |年齢が欠落しているか無効である場合、文書では成人のデフォルトを押して仮説ステートメントを印刷する必要があります。現在、声明の証拠はありません。 |
|患者のセックス | 13歳以上 QTc 閾値は男性/女性で分かれています | `metadata.patient_meta.sex` | OK |セックスが欠落している場合は、不明なフォールバックが必要です。 |
|年齢ビン | 0 ～ 23 時間、1 ～ 3 日、4 ～ 6 日、7 ～ 29 日、月/年セグメンテーション |コードは QRS 軸/QRS 期間ルックアップ テーブル |部分的 | RVH/LVH 電圧にはまだ完全な付録 A の正常値テーブルがありません。 |
|デキストロ心臓P 軸 90-180 度 | `global_features.p_axis_deg`; `_peds_dextrocardia()` | OK |利用可能。 |
|デキストロ心臓リード I または V6 マイナス P | `p_amp_mv` | OK | P 振幅シンボルのみを見て、より洗練された P 形態を見逃してください。 |
|デキストロ心臓I および V6 大きい S > 0.6 mV | `s_amp_mv`; `PEDS_DEXTRO_S_MV` | OK |利用可能。 |
|デキストロ心臓P 振幅 III > II | `p_amp_mv` | OK |利用可能。 |
|デキストロ心臓バイパス剰余 | `dextrocardia_suspected` |部分的 |ルールは存在します。 `statement_evidence.stop_further_interpretation` の小児科の理由がありません。 |
|レイ | P 持続時間 >= 60 ミリ秒 | `p_dur_ms` または P 境界集約 | OK |利用可能。 |
|レイ | P 振幅 >= 0.20 mV | `p_amp_mv`; `PEDS_RAE_P_AMP_MV` | OK |現在の一般的な P 形態では依然として成人の 0.24 mV が主に使用されており、小児のパスが 0.20 mV を完全に適用するかどうかを確認する必要があります。 |
|レイ | V1 二相性 P は、RAE の可能性をサポートします。 `p_biphasic_flag` が見つかりません |欠落しています |大人と同じギャップ。 |
|ラエ |リム P 持続時間 > 110 ms + 振幅 > 0.10 mV | `p_dur_ms`、`p_amp_mv` | OK |利用可能。 |
|ラエ |ノッチ付き P が重要性を追加 | `p_notch_flag` が見つかりません |欠落しています | LAE スコアを完全に回復することはできません。 |
|ラエ | V1 マイナス P 端子持続時間/振幅/面積 | `ptf_v1_mv_ms` |部分的 | PTF-V1 はい。ターミナルの持続時間/振幅/エリアは分割されません。 |
|ペ |重度のRAE + LAE -> 両房肥大 | `p_morphology_class`、`rae_leads`、`lae_*` |部分的 |ステートメントの重大度および重大度の高い候補者の証拠が欠落しています。 |
| QRS軸 |年齢調整済み LAD/RAD | `qrs_axis_deg`; `_peds_classify_qrs_axis()` | OK |コードにはすでに表 4-1 から 4-4 の検索が含まれています。 |
| QRS軸 | 15 度ゾーン内の境界線 LAD/RAD | `_peds_classify_qrs_axis()` | OK | `borderline_LAD` / `borderline_RAD` が出力可能です。 |
| VCD |年齢調整済み QRS 通常の制限 | `qrs_ms`; `_PEDS_QRS_NORMAL_MS` | OK |コードにはすでに表 4-5 が含まれています。 |
| VCD | >110% 正常 -> 境界線 IVCD | `_peds_classify_qrs_width()` | OK |利用可能。 |
| VCD | >120% 正常 -> 非特異的 IVCD / BBB 範囲 | `_peds_classify_qrs_width()` | OK |利用可能。 |
| ＲＢＢＢ |年齢用の VCD + RSR' または V1 の純粋な R | `r_prime_amp_mv`、`r_amp_mv`、Q/S 振幅 |部分的 | RSR'/純粋なRは大まかに判断できます。 R'期間が欠落しており、負の成分がないという事実。 |
| ＲＢＢＢ | R' 持続時間 >= 20 ms および振幅 >= 0.15 mV | `r_prime_amp_mv`;期間がありません |部分的 |振幅は存在します。 R' 持続時間は測定されません。 |
|いりRBBB のような形態だが、QRS <120% 正常 | `_peds_classify_qrs_width()` |部分的 | RVH からの合成ベクターの区別がありません。 |
| ＬＢＢＢ |年齢のせいで長引くQRS | `qrs_ms` | OK |利用可能。 |
| ＬＢＢＢ |端子 40ms QRS 軸 -90 ～ 90 時計回り |ターミナル 40ms 軸が見つかりません |欠落しています |基礎となる重要な測定値が欠落しています。 |
| ＬＢＢＢ | I/aVL/V5/V6 の S が短い/欠如、V1 ～ V3 の R が小さい/欠落 | Q/R/S振幅は大まかに判断できます |部分的 | S 持続時間がありません。小さい/なしのしきい値を設定する必要があります。 |
| LAFB | LBBB なし + 平均 QRS 軸 -60 ～ -90 | `qrs_axis_deg`、`bundle_branch_block` | OK |コードにはすでに小児 LAFB 範囲が含まれています。 |
| RVH | RBBB | の場合はバイパスします。 `bundle_branch_block` | OK |利用可能。 |
| RVH電圧 |年齢依存の RVH 電圧基準、6 つの年齢グループ、24 の条件 |部分的な Q/R/S/R' 振幅は |部分的 |完全な付録 A の 98 パーセンタイルしきい値がありません。 |
| RVH | V1/V2 の R/R'、V6 の S、R/S 比、QR パターン V1 | `r_amp_mv`、`r_prime_amp_mv`、`s_amp_mv`、`q_amp_mv` |部分的 |振幅は利用可能です。 R'持続時間、QRパターンオブジェクト、年齢別閾値テーブルが欠落しています。 |
| RVH | 48 歳以上 9 歳未満の V1 では直立 T、V5/V6 では逆 T が存在しない。 `t_amp_mv`、`t_polarity`、年齢 | OK |を導き出すことができます。 |
| RVH | RAD / ボーダーライン RAD のサポート |小児用 `qrs_axis_class` | OK |利用可能。 |
| RVH vs IRBBB |合成水平面終端角 |直接フィールドが見つかりません |欠落しています |軽度の RVH と IRBBB を明確に文書化します。 |
| LSH | V1 の顕著な R + V5/V6 の Q | `r_amp_mv`、`q_amp_mv`; `_peds_lsh()` |部分的 |現在、完全な 98 パーセンタイル テーブルではなく、固定のプロキシしきい値が使用されています。 |
| LVH | RBBB または LBBB の場合はバイパス | `bundle_branch_block` | OK |コードにはすでに小児用バイパスが含まれています。 |
| LVH電圧 | I/II/aVL/aVF/V5/V6 の R。 V1/V2 では S。 RV6+SV1;著名なQ | Q/R/S振幅 |部分的 |データが利用可能。不完全な年齢依存の 98 パーセンタイル閾値。 |
| LVHサポート | LAD および LAE は LVH をサポートします | `qrs_axis_class`、LAE フィールド | OK |利用可能。 |
| LVH株 | I/aVL/V4/V5/V6 ST/T 再分極パターン | `st_mid_mv`、`st_on_mv`、`t_amp_mv`、`t_polarity` |部分的 |データが利用可能。ルールには小児特有のパターンの証拠が必要です。 |
| BVH | RVH + LVH 高重大度 | `rvh_class`、`lvh_class`、`bvh_suspected` |部分的 |コードには要約フラグがあります。候補者の重大度の証拠が欠けている。 |
| BVH | R V1 >1.0 mV (LVH あり) | `r_amp_mv` | OK |利用可能。 |
| BVH | RVH + Q V6 >10ms および >0.07mV + R V6 >1.0mV | Q持続時間の概算、Q/R振幅 |部分的 | Q 持続時間は概算であり、独立した Q オンセット/オフセットがありません。 |
| BVH | V2/V3/V4 の >=2 で R+S >6.0mV | Q/R/S振幅 | OK |派生可能。 |
| BVH抑制 | BVH は個々の RVH/LVH を抑制します | `bvh_suspected` |部分的 |コードは何らかの抑制を行います。ステートメントレベルの抑制された_byがありません。 |
|低電圧 |前頭/前胸部 QRS ピークツーピーク | Q/R/S から導き出すことができます。 `low_voltage_class` | OK |大人と同じです。 |
| COPD パターン |低電圧 + 右方向 P/QRS + RAE |低電圧、軸、RAE |部分的 |導き出すことができます。小児声明の証拠が欠けている。 |
| Q波異常 |グループの 2 つのリードに大きな Q 波が発生 | `q_amp_mv`、`pathological_q_leads` |部分的 |グループ数を導き出すことができます。小児科特有の記述が欠落しています。 |
| Q波梗塞 | Q > 1/5 R 振幅は梗塞を示唆します。 `q_amp_mv`、`r_amp_mv` | OK |利用可能。 |
| STうつ病 |前方/側方/下方グループ | `st_depression_leads`、地域 | OK |利用可能。 |
| STうつ病 | 1 つのグループで >0.20 mV -> 非特異的 ST 低下 | `st_on_mv`など |部分的 |データが利用可能。現在の一般的な ST 閾値は低いため、小児特有の閾値出力が必要です。 |
| STうつ病 |頻脈 -> 心拍数関連のステートメント | `heart_rate_bpm`、年齢 |部分的 |この文書には小児用の頻脈表は記載されていません。今の大人っぽい`190-age`では物足りない。 |
| ST抑制うつ病 |肥大または VCD が抑制する | RVH/LSH/LVH/BVH/VCD フィールド |部分的 |ステートメントレベルの抑制出力がありません。 |
| T異常 |前方/側方/前側方/下方グループの逆T字 | `t_amp_mv`、`t_polarity` | OK |導出できる。 |
| T異常 |グループ内の 2 リード以上で逆 T 振幅 >1.0mV -> トール T 異常 | `t_amp_mv`、リードグループ | OK |導出できる。 |
| RVH/LVHに続発するT異常 | RVH/LVH フラグ + 逆 T グループ |利用可能なフィールド |部分的 |最終陳述の証拠が欠けている。 |
|再分極異常 | ST押し+逆Tの組み合わせ | ST/T の事実 |部分的 |データが利用可能。結合された重大度/候補エンジンがありません。 |
| ST上昇 |すべてのリード線がテストされ、>0.15mV は正常な変動の可能性を示唆しています。 `st_elevation_leads`、`st_on_mv` |部分的 |データが利用可能。小児の閾値と声明が必要です。 |
| ST上昇抑制 |肥大と VCD 抑制 | RVH/LSH/LVH/BVH/VCD フィールド |部分的 |ステートメントレベルの抑制出力がありません。 |
|心膜炎 |前方/側方/下方グループの ST 上昇、5 ～ 15 歳 | `pericarditis_suspected` | OK |コードにはすでに概要フラグが含まれています。 |
|早期再分極 |非特異的 ST 上昇、T 反転なし、13 ～ 15 歳 | `early_repolarization_suspected` | OK |コードにはすでに概要フラグが含まれています。 |
|背の高いT | T >1.20mV または >0.50mV かつ >0.5 QRS ピークツーピーク | `t_amp_mv`、QRS ピークツーピーク | OK |利用可能。 |
| QTショート | QTc <340ms ボーダーラインショート | `qtc_bazett_ms`、`qtc_class` | OK |コードにはすでに小児分類子が含まれています。 |
| QT境界線が延長 | >450 <5y; >454 5～12 歳。 13 歳以上の男子は 458 人以上。 13 歳以上の女の子が 465 人以上 |年齢/性別/QTc | OK |コードは利用可能です。 |
| QT延長 |境界しきい値 +20ms |年齢/性別/QTc | OK |コードはすでに存在します。 |
| QT抑制 | RVH/LSH/LVH/BVH/VCD -> ワイド QRS に続発する QT 延長 |利用可能なフィールド |部分的 |現時点では、クラスを正規化/抑制し、印刷された二次的なステートメントの証拠が欠落している可能性があります。 |
|電解質 | QTc <310 hypercalcemia; >520 ST 低下 + T 陽性を伴う低カルシウム血症/低カリウム血症 | QTc、ST、T | OK |コードにはすでにヒントが含まれています。 |
|先天性欠陥 |心房/心室肥大、VCD、軸、QRS 形態の組み合わせ |入手可能な概要事実 |部分的 |この文書には完全なルールはありません。現在、CHD 候補エンジンはありません。 |

<a id="和当前-morphologyinputs-的关系"></a>
## 現在の `morphology_inputs` との関係

最上位の `morphology_inputs` が最後のラウンドで追加されました。そのうちの `record/global/leads/derived_facts/statement_evidence` は子供向けのルールにも役立ちます。

- `record.age_years` および `record.sex` は小児ルーティングに使用できます。
- `global.qrs_duration_ms`、`global.qrs_axis_frontal_deg`、`global.qtc_bazett_ms` は、年齢調整された QRS/QTc をサポートします。
- `leads.*.p/qrs/st/t` は、右心筋症、RAE/LAE、VCD/BBB、肥大、ST/T/QT をサポートします。
- `derived_facts` には成人向けの axis/VCD/RVH/LVH/ST/T/QT ブロックがすでにあります。

`derived_facts.pediatric` サブブロックは、子供のしきい値と大人の事実の混合を避けるためにこのラウンドに追加されました。再生成された構造は次のようになります。

```json
"pediatric": {
  "is_pediatric": true,
  "age_bucket": "5-7_years",
  "route_selected": true,
  "axis_limits": {
    "lad_threshold_deg": -10,
    "borderline_lad_upper_deg": 1,
    "rad_threshold_360_deg": 201,
    "borderline_rad_threshold_360_deg": 160
  },
  "qrs_duration_limits": {
    "normal_limit_ms": 88,
    "borderline_ivcd_ms": 96.8,
    "nonspecific_ivcd_ms": 105.6
  },
  "qtc_limits": {
    "short_ms": 340,
    "borderline_prolonged_ms": 454,
    "prolonged_ms": 474,
    "severe_ms": 520
  },
  "hypertrophy_thresholds": {
    "rvh_lvh_age_percentile_tables": {
      "available": false,
      "reason": "appendix_a_percentile_tables_not_digitized"
    },
    "lsh_r_v1_prominent_mV": 1.0,
    "bvh_rs_sum_v2_v3_v4_mV": 6.0
  },
  "repolarization_thresholds": {
    "st_depression_threshold_mV": 0.20,
    "st_elevation_normal_variation_threshold_mV": 0.15,
    "pericarditis_age_gate": "5-15",
    "early_repolarization_age_gate": "13-15"
  }
}
```

<a id="推荐优先级"></a>
## 推奨される優先順位

- `P0`: 完了しました。 `adult` / `pediatric` サブブロックを `morphology_inputs.derived_facts` の下に追加し、アルゴリズムの年齢ルート、年齢バケット、小児軸/QRS/QTc しきい値、および既存の小児サマリー フラグを公開します。
- `P1`: 小児の RVH/LVH 電圧閾値表を完成させます。これは、文書に引用されている付録 A の通常の測定値です。
- `P2`: R' 期間、S 期間、終端 40ms QRS 軸、水平終端ベクトルを追加して、RBBB/IRBBB/LBBB/RVH の区別能力を向上させます。
- `P3`: 公的に追跡可能でバージョン管理された小児参照テーブルを取得した後、既存の統合ステートメント/抑制エンジンを拡張します。それまでは `unavailable` のままです。
- `P4`: 先天性心疾患ルール レイヤーを作成するには、最初に特定の先天性心疾患の組み合わせルールを整理する必要があります。そうしないと、「ルールが実装されていません」プレースホルダーのみを出力できます。
