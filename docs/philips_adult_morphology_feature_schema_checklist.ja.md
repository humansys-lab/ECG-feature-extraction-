<!-- i18n-nav -->
[中文](philips_adult_morphology_feature_schema_checklist.md) | [English](philips_adult_morphology_feature_schema_checklist.en.md) | [日本語](philips_adult_morphology_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="philips-adult-morphology-feature-schema-checklist"></a>
# Philips 成人形態学機能スキーマ チェックリスト

<a id="结论先行"></a>
## 結論を先に

> 2026-07-12 ステータス更新: トップレベルが現在のエクスポートに追加されました
> `clinical_interpretation` (`clinical_rules.v1`/ルールセット `2026.07.1`)。これは公開です
> AHA/ACCF/HRS、ESC、MI の第 4 普遍定義に基づく権威ルール層。
> `interpretation` および `statement_engine` は予約されており、`reference_only` として明確にマークされています。
> グラスゴー解釈レイヤーは、現在のランタイムおよびエクスポート契約から削除されました。このプロジェクトは以下に該当するものではなく、またそれを主張するものでもありません。
> Philips/DXL 独自のアルゴリズムを再現。
> 各統合ルールはステータス、必須/欠落入力、証拠、しきい値、ソースを出力します
> 抑制と同じ。必要な証拠が不足している場合は、`unavailable` が返され、陽性または「正常」の根拠として使用されません。

現在の全機能 JSON (`JS00029_features.json` など) は、**成人形態学 MVP** のバージョン、特に次のモジュールをすでにサポートできます。

- QRS/P/T/ST 前頭軸と QRS-T 角度
- PR/QRS/QT/QTc/JT およびその他のグローバル間隔
- リードごとの P/QRS/T 境界、P/QRS/T 振幅、ST J ポイント/中間/80ms
- Q/R/S/R'/S' 振幅、QRS エリア、ノッチ/スラー、VAT、fQRS スコア
- T振幅、T極性、TPE、U波フラグ
- PTF-V1、P持続時間、Pエリア
- LVH電圧、低電圧、ST低下/上昇、病的Q、R進行、高T、その他の事前説明フィールド

フル機能の JSON を再生成した後、現在の測定値を形態学ルール エンジンの入力スキーマに編成するために、トップレベルの `morphology_inputs` が追加で含まれます。しかし、`要点adult_morphology.pdf` でコンパイルされたフィリップス スタイルの成人形態学エンジンを完全に満足させることはできません。主なギャップは「まったく測定できない」ことではなく、次のとおりです。

- 構造化された候補者の声明と統一ルール監査が利用可能ですが、現在実装されている一連の公開ルールのみをカバーしています。完全な Philips コード マトリックスとダウングレード/交換セマンティクスはまだカバーされていません。
- P 波には、微細な形態に明らかなギャップがあります。`is_notched`、`is_biphasic`、初期/終了期間/振幅/面積は部分的にのみ近似しています。
- QRS コンポーネントの持続時間が不完全: Q 持続時間は現時点では概算であり、R'/S'/Q/S 持続時間は直接測定されていません。
・Dextrocardia、BBB、RVH、MI等で要求される終端/水平/初期QRS方向は明示的な事実として出力されません。
- ST の形状と傾斜の単位が正確に一致しません 元のテキスト: 現在、`st_slope_mv_per_ms` と `upsloping/horizontal/downsloping` があり、元のテキストでは角度と直線/上凹/下凹が必要です。
- ST マップ、カブレラ座標、V4R/V7-V9 拡張リード、原因となる動脈の出力がまだ見つかりません。
- 医薬品/臨床入力が不十分: `meds` には場所がありますが、現在のサンプルはほとんどが `null` であり、独立した `clinical_dx_codes` / `rx_codes` ルール コンテキストがありません。

したがって、評決は次のとおりです。 **公開ガイドラインの統合 MVP は、最終ルールとして実行および出力する準備ができています。 Philips/DXL スタイルの結果は参考用のみです。フィリップスの成人形態の完全なレプリカには、独自のコード マトリックス、いくつかの基礎となる形態学的事実、および実証済みの独自の調停ロジックがまだ欠けています。 **

<a id="文档范围"></a>
## ドキュメントのスコープ

このリストでは、2 つのローカル PDF を比較しています。

- `/home/chtmedgemma/projects/ecg_gemma/philips_adult_morphology.pdf`
- `/home/chtmedgemma/projects/ecg_gemma/要点adult_morphology.pdf`

`要点adult_morphology.pdf` は、フィリップスの成人形態解析から編集された仕様であり、右心、心房異常、QRS 軸、VCD/BBB、RVH、LVH、低電圧/COPD、MI、ST/T/QT、電解質/薬物効果、抑制エンジンをカバーしています。

<a id="数据层级判断"></a>
## データレベル判定

|データレベル |成人の形態解析をサポートできますか |説明 |
| --- | --- | --- |
| `.txt report` |不十分 |人間による読み取りに適しており、リードごと/拍動ごとの形態学的事実、候補ステートメント、抑制証拠が完全に欠けています。 |
|現在の古いフル機能 JSON (例: `JS00029_features.json` | MVP をサポートでき、多数のフィールドが利用可能 | `beats`、`beat_features`、`representative_leads`、`global_features`、`metadata`、`interpretation`が含まれます。ただし、`morphology_inputs`ではありません。 |
|再生されたフル機能 JSON |ダウンストリーム ルール エンジンにより適しています。現在の `to_dict(ECGFeatures)` は、`rhythm_inputs` と `morphology_inputs` を追加し、元の測定値を保持します。 |
| `build_structured_payload()` 薄いペイロード |十分ではありません |完全な `beat_features`、`representative_leads`、候補証拠はエクスポートされず、形態学ルール エンジンへの入力には適していません。 |

<a id="js00029-当前-json-快照"></a>
## JS00029 現在の JSON スナップショット

現在、`JS00029_features.json` は古いエクスポートであり、以下が含まれていることが確認されています。

- トップレベル: `fs`、`quality`、`beats`、`beat_features`、`representative_leads`、`groups`、`global_features`、`metadata`、 `interpretation`
- ビート: 14
- Beat_features: 168、12 リード x 14 ビート
- ビートレベルフィールド: P/QRS/T 境界、`qt_ms`、`pr_ms`、`qrs_ms`、P/Q/R/S/T 振幅、ST J/mid/80ms、`j_index`、`delta_present`、 `jt_ms`、信頼度、R'/S'、QRS ピーク数、QRS ノッチ数、VAT、P/T 期間/エリア、T 極性、U 波、QRS スラー、ST スロープ、TPE、ST モルフォロジー、fQRS、PTF-V1、コンセンサスQT/PR
-代表リード: 12 個のリード、それぞれ `params` および `variance`
- global_features: 心拍数、心房率、PR/QRS/QT/QTc、P/QRS/T/ST 轴、QT 离散度、起搏、PTF-V1
- metadata: 输入/内部采样率、导联顺序、心搏数、qrs_检测器、代表性组ID、患者元数据
- interpretation: 旧版形态学标志，如 `pathological_q_leads`, `q_wave_territories`, `r_progression_class`, `st_elevation_leads`, `st_depression_leads`, `lvh_voltage_criteria`, `lvh_class`, `low_voltage_class`, `rvh_suspected`, `tall_t_leads`

当前旧版 JSON 不包含：

- 顶层 `morphology_inputs`
- 顶层 `rhythm_inputs`
- 结构化的候选解读
- 陈述 `code/category/severity/reason/suppressed_by`
- `dextrocardia_suspected`, `rvh_class`, `copd_pattern`, `qtc_electrolyte_hint`, `posterior_mi_suspected`, `st_rate_related` 等较新的解读扩展字段

重新生成完整特征 JSON 后，将额外包含顶层 `morphology_inputs`，包括 `record`、`global`、`leads`、`derived_facts`、`statement_evidence` 五个部分。

## 状态定义

- `OK`: 当前完整特征 JSON 已提供，可直接使用或稳定派生。
- `Partial`: 存在相关字段，但需要近似、聚合、重新导出，或缺少证据链/精细分量。
- `Missing`: 当前未发现可直接使用的测量字段或结构。

<a id="feature-checklist"></a>
## 特征清单

| 模块 | `要点adult_morphology.pdf` 所需的特征 | 当前可用字段 | 状态 | 缺口 / 风险 |
| --- | --- | --- | --- | --- |
| 基础输入 | 采样率、导联顺序、记录长度 | `fs`, `metadata.input_fs`, `metadata.internal_fs`, `metadata.lead_order` | OK | 记录长度并不总是显式输出为 `duration_sec`。 |
| 患者信息 | 年龄, 性别 | `metadata.patient_meta.age`, `metadata.patient_meta.sex` | OK | 年龄/性别已有；完整的年龄/性别校正表仍需配置。 |
| 临床输入 | 临床诊断代码 | 未发现稳定字段 | Missing | 无法可靠使用二尖瓣疾病等抑制条件。 |
| 药物输入 | rx_codes / 洋地黄 | `metadata.patient_meta.meds` | Partial | 字段位置存在，但样本多为 `null`；没有结构化的 `rx_codes`。 |
| 质量控制 | 导联可靠性、波形特定可靠性 | `quality.*.reliable_for_p/qrs/t/qt`, `beat_measurement_reliable` | OK | 可以对 P/QRS/T/ST/QT 测量进行门控。 |
| 输出模型 | 代码/类别/严重程度/原因/抑制 | `clinical_interpretation.final_statements/domains/suppressed_statements/conflicts`；另有仅参考用的 `statement_engine` | Partial | 公开规则已有完整审计；不覆盖完整的 Philips 代码/类别/降级/替换矩阵。 |
| 右位心 | P 轴右偏 | `global_features.p_axis_deg` | OK | 可判断。 |
| 右位心 | QRS 额面轴右偏 | `global_features.qrs_axis_deg` | OK | 可判断。 |
| 右位心 | 水平 QRS 方向右偏 | 可由 V5/V6 R/S 粗略派生 | Partial | 没有明确的 `qrs_axis_horizontal_direction`。 |
| 右位心 | V5/V6 小 QRS | 可由 Q/R/S 峰峰值派生 | Partial | 原文未给出阈值；当前需要配置。 |
| 右位心 | 提前停止 / 绕过剩余形态 | 新模型有 `dextrocardia_suspected`，旧 JSON 无 | Partial | 缺少结构化的控制流证据。 |
| 右房扩大 (RAE) | 肢体导联 P 波持续时间 >= 60 ms | `beat_features[*].p_dur_ms`; 当前代码可聚合 | OK | 代表性参数未直接列出 P 波持续时间，需从心搏特征中聚合。 |
| 右房扩大 (RAE) | 肢体导联 P 波振幅 >= 0.24 mV | `p_amp_mv` | OK | 可用。 |
| 右房扩大 (RAE) | V1 双相 P 波 | 未发现 `p_biphasic_flag` | Missing | 仅有 PTF-V1/振幅近似，不能完整判断双相形态。 |
| 左房扩大 (LAE) | 肢体导联 P 波持续时间 > 110 ms + 振幅 > 0.10 mV | `p_dur_ms`, `p_amp_mv` | OK | 可聚合。 |
| 左房扩大 (LAE) | 切迹 P 波 | 未发现 `p_notch_flag` | Missing | 原文指出切迹 P 波提高权重，当前不能稳定判断。 |
| 左房扩大 (LAE) | V1 终末负向 P 波持续时间/振幅/面积 | `ptf_v1_mv_ms` | Partial | 有 PTF-V1，但初始/终末持续时间/振幅/面积未完整拆分。 |
| 双房扩大 (BAE) | RAE + LAE 高严重程度组合 | `interpretation.p_morphology_class`, `rae_leads`, `lae_*` | Partial | 当前有汇总结果，缺少候选语句严重程度与组合证据。 |
| QRS 轴 | LAD/RAD 基础范围 | `global_features.qrs_axis_deg`, `interpretation.qrs_axis_class` | OK | 可支撑基础判断。 |
| QRS軸 |年齢/性別調整後の正常範囲 |年齢・性別あり |部分的 |元のテキストには完全な表が示されていないため、構成する必要があります。 |
| VCD/BBB | QRS 持続時間 100/110/120 ミリ秒 レイヤード | `global_features.qrs_ms`、リードごと `qrs_ms` | OK |利用可能。 |
|束状ブロック | LAFB/LPFB 軸範囲 | `global_features.qrs_axis_deg`、`interpretation.qrs_axis_class` |部分的 |主に軸で判断できますが、完全な反時計回り/時計回りの方向を正規化する必要があります。 |
| ＲＢＢＢ |端子 QRS 右向き。 I/aVL/V6 端子マイナス、V1 端子プラス | R/S/R'/S' 振幅、QRS 符号付きエリア 新しいエクスポートが利用可能 |部分的 |端子部分の方向は直接測定されていません。古い JSON に `qrs_signed_area` がありません。 |
| ＬＢＢＢ |端子 QRS 左向き。 I/aVL/V6 プラス、V1 マイナス | R/S/R'/S' 振幅、QRS 符号付きエリア 新しいエクスポートが利用可能 |部分的 |明確な最終強制事実が欠落しています。 |
| R'/S' の形態 | R' / S' 振幅 | `r_prime_amp_mv`、`s_prime_amp_mv` | OK |振幅が利用可能。 |
| R'/S' 持続時間 | R' / S' 持続時間 | `r_prime_duration_ms`、`s_duration_ms`、`s_prime_duration_ms`、`r_duration_ms` | OK |統合 BBB/RVH ルールは、欠落している期間を `unavailable` としてマークします。 |
| RVH | V1 顕著な R/R' | `r_amp_mv`、`r_prime_amp_mv` |部分的 | R/R' 振幅が利用可能。 R' 持続時間がありません。 |
| RVH | V1 正成分 > 負成分 | Q/R/S | から近似できます。部分的 |直接の `positive_component_mV` / `negative_component_mV` はありません。 |
| RVH | I/V6 顕著な Q/S/S の継続時間 + アンプ | `q_amp_mv`、`s_amp_mv`、`s_prime_amp_mv` |部分的 |振幅はい。 Q/S/S の期間がありません。 |
| RVH | RV ひずみパターン | `st_on_mv`、`t_amp_mv`、`t_polarity`、`st_depression_leads` | OK | II/aVF/V1-V3 ST 低下 + 逆 T を確認可能。
| RVH |段階的に考慮/可能性が高い/最終的な |新しいモデルには `rvh_class` があります。古い JSON はほとんどが `rvh_suspected` |部分的 |現在の古い JSON にはグレーディングがありません。完全な証拠が欠けています。 |
| LVH電圧 | R aVL、R I + S III、R V5/V6、S V1/V2 + R V5/V6 | `r_amp_mv`、`s_amp_mv`、`lvh_voltage_criteria` | OK |表 3-1 の電圧項目をサポートできます。 |
| LVHコーネル | R aVL + S V3、コーネル製品 | `r_amp_mv`、`s_amp_mv`、`global_features.qrs_ms` | OK |利用可能。 |
| LVH 年齢/軸抑制 |年齢 < 35、右軸 |患者の年齢、`qrs_axis_class` | OK |データが利用可能です。 |
| LVH アドオン フラグ | LAD、LAE、LV 系統、QRS/VAT ワイド |軸、LAE フラグ、ST/T、VAT |部分的 |決定可能;ただし、構造化された flags_set はありません。 |
| LVH 表 3-2/3-3 出力 | LVHV/LVHCNV/LVHCNP など コード + 重大度 + 理由 | `lvh_voltage_criteria`、`lvh_class` |部分的 |コード出力とその理由が欠落しています。 |
|低電圧 |前頭/前胸部 QRS ピークツーピーク | Q/R/S 振幅から導き出すことができます。 `low_voltage_class` | OK | `qrs_peak_to_peak_mV` を直接出力することをお勧めします。 |
| COPD パターン |低電圧 + 右向き P/QRS + RAE |低電圧、軸、RAE が利用可能。新しいモデルには `copd_pattern` があります |部分的 |古い JSON に `copd_pattern` がありません。最終陳述の証拠が欠けている。 |
| MI Q ウェーブ | Q振幅、Q/R比 | `q_amp_mv`、`r_amp_mv`、`pathological_q_leads` | OK |利用可能。 |
| MI Q 持続時間 |下/側方/前方 Q 持続時間閾値 | `q_onset`、`q_offset`、`q_duration_ms` | OK |信頼できる Q 時間制限を取得できない場合、統一ルールは `unavailable` を返し、Q 振幅は時間制限の置き換えに使用されません。 |
| MI Q波領域 |前Q波領域 | `q_area_mv_ms`、`initial_qrs_area_mv_ms` |部分的 |ダイレクト Q エリアはすでに存在します。地域/しきい値の適用性は、特定のガイドラインに従って確認する必要があります。 |
| MI 初期 QRS 軸 |初期 QRS 軸左方向 |直接フィールドが見つかりません |欠落しています |元のテキストは、劣性 MI の可能性を高めるために使用されます。 |
| MI R の進行 |移行 / 進行不良 | `r_progression_class`、`r_s_transition_lead` | OK |利用可能。 |
| MI ST/T 年齢推定 | T 反転深さ、ST 偏差の大きさ | `t_amp_mv`、`st_*`、地域 |部分的 |データが利用可能。梗塞年齢の重症度/ランキングルールの出力がありません。 |
|ミシガン州 |下/外側/前/前外側/後 | `q_wave_territories`、ST 地域、R 進行 |部分的 |既存の領土は荒れています。後方/右側のリードが不完全。 |
|犯人の動脈 | RCA/LCx/LAD/LM-MVD 推論 | ST II/III/V2/V3/aVR が利用可能 |部分的 |導出可能なルール。 V4R/V7-V9 と構造化された原因となる出力が欠落しています。 |
|リードの拡大 | V4R、V7、V8、V9 | `STANDARD_12_LEADS` のみ |欠落しています |後壁/右心室梗塞に対する感度が不十分。 |
| STマップ |カブレラ角、ST偏差mm、陰影マップ | ST偏差あり |部分的 | Cabrera マッピング、極性調整値、描画構造が欠落しています。 |
| STうつ病 | J点、中点、J+80ms | `st_on_mv`、`st_mid_mv`、`st_80ms_mv` | OK |重要なポイントは3つあります。 |
| STうつ病 | ST終了/T開始 | `t.onset` は T で始まるとして利用可能 |部分的 |明示的な `st_end_mv` はなく、生の信号または T オンセットから派生する必要があります。 |
| STうつ病 |傾斜度 | `st_slope_mv_per_ms` |部分的 |単位は度ではありません。 |
| STうつ病 |形状 ストレート/凹型 上/下 | `st_morphology` = 上り/水平/下り |部分的 |分類体系は元の形状と一致しません。 |
|レート関連の ST 低下 | HR > 190 - 年齢 | `heart_rate_bpm`、年齢、`st_depression_leads`;新しいモデルには `st_rate_related` があります |部分的 |古い JSON には `st_rate_related` がありません。ステートメントがありません。 |
| T波異常 | T 振幅 | `t_amp_mv` | OK |利用可能。 |
| T波異常 |相対 T/QRS 振幅 | `t_amp_mv` + QRS ピークツーピークを導出可能 | OK |比率を直接出力することをお勧めします。 |
| T波異常 |極性プラス/マイナス/フラット/二相 | `t_polarity` |部分的 |ポジティブとネガティブ。 flat/biphasic は明確に構造化されていません。 |
| T軸 / QRS-T角度 |前方 T 軸、QRS-T 角度 | `global_features.t_axis_deg`、`interpretation.qrs_t_angle_deg` | OK |利用可能。 |
| ST-T 再分極コンボ | ST低下+T異常の複合重症度 | ST/T フィールド はい |部分的 |候補カテゴリと重大度の組み合わせがありません。 |
| ST上昇 | J ポイントと J+80 の正の値 | `st_on_mv`、`st_80ms_mv` | OK |利用可能。 |
| ST上昇 |怪我/心膜炎/早期再発 | ST 地域 はい。新しいモデルには心膜炎/早期再ポールフラグがあります |部分的 |古い JSON 新しいフラグがありません。完全な抑制/交絡因子リストがありません。 |
|背の高いT |正の T > 1.2 mV、または >0.5 mV かつ >0.5 QRS PP | `t_amp_mv`、`t_polarity`、QRS ピークツーピーク、`tall_t_leads` | OK |十分なデータ。ルールは正の T を制限する必要があります。
| QT ショート/ロング | QTc しきい値 310/340/465/485/520 | `qtc_bazett_ms`、`qtc_fridericia_ms`、`qtc_class` | OK |利用可能。 |
|電解質のヒント |高カルシウム血症/低カルシウム血症/低カリウム血症 | QTc + ST 落ち込み + T 陽性 |部分的 |データが利用可能。古い JSON に `qtc_electrolyte_hint` がありません。 |
|ジギタリス効果 | Rx ジギタリス + ST/T/QT パターン | `patient_meta.meds` |部分的 |薬剤入力が不安定で、`rx_codes` がありません。 |
|抑制ルール | RVH/LVH/BBB/梗塞/薬剤/電解質抑制 ST/T/QTc | `clinical_interpretation.suppressed_statements`、ルールごと `suppressed_by` |部分的 |リード反転、BBB/IVCD、および選択された肥大/虚血ルールをカバー。フィリップス抑制マトリックスを完全にはカバーしていません。 |
|ルール監査 |すべてのルールの理由の出力 | `clinical_interpretation.domains.*` にはステータス/証拠/しきい値/ソース/missing_inputs が含まれています | OK (現在公開ルールセット) |これは実装ルールが監査可能であることを意味するだけであり、臨床検証や独自のアルゴリズムの同等性検証が行われたことを意味するものではありません。 |

<a id="实用判断"></a>
## 実際の判断

現在の全機能 JSON は、**MVP 成人形態学ルール エンジン**にすでに適しています。

- 右心筋の大まかな判断
- LAD/RAD/QRS軸
- QRS 期間階層化、基本的な IVCD/RBBB/LBBB
- 低電圧
- LVH 電圧 / コーネル / ソコロフ リヨン
- RVH粗スクリーン
- 病理学的Q波領域の粗いスクリーニング
- ST上昇/低下領域の粗スクリーニング
- T波異常粗画面
- 背の高いT
- 基本的な電解質のヒントを含む QTc ショート/ロング

現在のところ、完全な複製には適していません。フィリップスの成人形態学:

- RAE/LAE/BAE の P ノッチ/二相性/端子コンポーネントの詳細なグレーディング
- BBB/RVHの終端力と成分持続時間を細かく判断
- MI の年齢と原因となる動脈の完全な証拠チェーン
- ST マップのカブレラ空間表示
- V4R/V7-V9拡張リードロジック
- 完全な LVH 表 3-2/3-3 ステートメント コード マトリックス
- グローバル抑制/ダウングレード/置換/ランキング ルール
- 各出力ステートメントの理由、重大度、suppressed_by、confidence_rank

<a id="已新增的最小目标-schema"></a>
## 最小ターゲット スキーマを追加しました

フル機能 JSON 新しいトップレベルの `morphology_inputs` がリズムのように追加されました。これは、元の測定値を置き換えるのではなく、既存のフィールドをルール エンジンに適したファクトに編成します。

### `morphology_inputs.record`

- `fs`
- `duration_sec`
- `age_years`
- `age_valid`
- `sex`
- `algorithm_age_group`
- `algorithm_age_reason`
- `age_defaulted_to_adult_algorithm`
- `clinical_dx_codes`
- `rx_codes`
- `lead_order`
- `available_extended_leads`
- `record_quality`
- `rejected_functions`

### `morphology_inputs.global`

- `heart_rate_bpm`
- `qrs_duration_ms`
- `qt_ms`
- `qtc_bazett_ms`
- `qtc_fridericia_ms`
- `p_axis_frontal_deg`
- `qrs_axis_frontal_deg`
- `qrs_axis_horizontal_direction`
- `t_axis_frontal_deg`
- `qrs_t_angle_deg`

### `morphology_inputs.leads`

リードごとの出力:

- `p.duration_ms`
- `p.amplitude_mV`
- `p.area`
- `p.is_notched`
- `p.is_biphasic`
- `p.initial_duration_ms`
- `p.initial_amplitude_mV`
- `p.terminal_duration_ms`
- `p.terminal_amplitude_mV`
- `p.terminal_area_ashman`
- `qrs.q_duration_ms`
- `qrs.q_amplitude_mV`
- `qrs.r_amplitude_mV`
- `qrs.r_prime_duration_ms`
- `qrs.r_prime_amplitude_mV`
- `qrs.s_duration_ms`
- `qrs.s_amplitude_mV`
- `qrs.s_prime_duration_ms`
- `qrs.s_prime_amplitude_mV`
- `qrs.peak_to_peak_mV`
- `qrs.positive_component_mV`
- `qrs.negative_component_mV`
- `qrs.area_sign`
- `qrs.terminal_direction`
- `st.j_point_mV`
- `st.midpoint_mV`
- `st.j80_mV`
- `st.end_mV`
- `st.slope_deg`
- `st.shape`
- `t.amplitude_mV`
- `t.polarity`
- `t.relative_to_qrs`
- `quality`

### `morphology_inputs.derived_facts`

- `adult`
- `pediatric`
- `dextrocardia_criteria`
- `rae_criteria`
- `lae_criteria`
- `bae_criteria`
- `axis_flags`
- `vcd_bbb_criteria`
- `rvh_criteria`
- `lvh_voltage_criteria`
- `lvh_additional_flags`
- `low_voltage_criteria`
- `copd_pattern_criteria`
- `mi_territory_criteria`
- `culprit_artery_criteria`
- `st_depression_criteria`
- `t_wave_criteria`
- `st_elevation_criteria`
- `tall_t_criteria`
- `qt_electrolyte_criteria`

### `morphology_inputs.statement_evidence`

- `candidates[]`
- `final_statements[]`
- `statement.code`
- `statement.category`
- `statement.severity`
- `statement.reason`
- `statement.evidence`
- `statement.flags_set`
- `statement.suppression_tags`
- `statement.suppressed_by`
- `statement.confidence_rank`
- `stop_further_interpretation`

<a id="优先级建议"></a>
## 優先的な提案

- `P0`: 完了しました。 `morphology_inputs` スキーマを追加し、成人/小児の年齢ルートとしきい値コンテキストを補足し、現時点で安定して導出できる事実を出力し、複雑な信号アルゴリズムを追加しません。
- `P1`: 補足 P ノッチ/二相性、P 端子コンポーネント、Q/R'/S'/S 期間、QRS 端子方向。
- `P2`: 既存のステートメント/抑制エンジンをまだカバーされていない公開ルールに拡張し、`unavailable` を参照アルゴリズムから分離します。
- `P3`: ST マップ、V4R/V7-V9、および MI 原因動脈の完全な強化ロジック。
