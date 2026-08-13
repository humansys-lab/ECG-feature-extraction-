<!-- i18n-nav -->
[中文](dxl_ecg_dev_task_checklist.md) | [English](dxl_ecg_dev_task_checklist.en.md) | [日本語](dxl_ecg_dev_task_checklist.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="dxl-inspired-ecg-边界定位开发任务清单"></a>
# DXL にインスパイアされた ECG 境界線位置特定開発タスクリスト

> 目標：現在のライブラリ内のビートごとの導联別 P/QRS/T 境界線位置特定モジュールを、「ヒューリスティックな第1版」から **多導联 + グループ化 + representative beat + 幾何学的 T-end + 信頼性スコアリング** を備えた完全な測定コアへとアップグレードする。  
> DXL マニュアルで公開されている方法論の主要流れは、まず波形品質評価を行い、次に波形認識を行い、その後 representative beat を形成し、包括的な測定値を生成し、最後に解釈層に進むというものである。その中でグループ1では外れ値の除外、整列、平均化が行われ、QT の難点は T 波終点（T-end）であり、グローバル QT は信頼性の高い導联の中央値を採用する。

---

<a id="0-文档使用说明"></a>
## 0. ドキュメントの使用説明

<a id="范围"></a>
### 範囲
本リストは以下のモジュールをカバーする：

- `quality/`
- `qrs/`
- `grouping/`
- `representative/`
- `delineate/`
- `features/qt.py`
- `pacing/`
- `validate/`

<a id="不在本轮范围"></a>
### 本ラウンドの範囲外
本ラウンドでは以下は実施しない：

- 完全な解釈ステートメント
- 臨床ルール分類の全セット
- DXL のプライベート閾値の 1:1 再現

<a id="本轮交付定义"></a>
### 本ラウンドの納品定義
完了時には以下を備えること：

1. 多導联 QRS アンカー検出
2. ビートグループ化
3. representative beat 駆動による境界線の微調整
4. P/QRS/T の単一ビート局所修正
5. 幾何学的 T-end
6. 信頼性導联 / 信頼度体系
7. グローバル QT の計算
8. 基本的な可視化と検証基準

---

<a id="1-目录与代码结构重构"></a>
## 1. ディレクトリとコード構造のリファクタリング

<a id="任务-t001重构-delineation-目录"></a>
### タスク T001：delineation ディレクトリのリファクタリング

**目標**  
既存の `delineate.py` を階層化されたモジュールに分割し、反復開発とテストを容易にする。

**入力**  
既存：

- `ecgfeat/delineate.py`

**出力**  
新規構造：

```text
ecgfeat/delineate/
├── __init__.py
├── rough.py
├── representative_refine.py
├── beat_local_refine.py
├── p_wave.py
├── qrs_bounds.py
├── t_wave.py
├── t_end_geometric.py
├── confidence.py
└── fusion.py
```

**コアタスク**
- 既存の関数を責務ごとに分割する
- 互換性のあるエントリポイントを保持する
- モジュールレベルの docstring を追加する

**受け入れ基準**
- `from ecgfeat.delineate import delineate_beats` が引き続き動作すること
- ユニットテストが新旧の API をカバーすること
- 旧呼び出しチェーンでエラーが発生しないこと

**優先度**  
P0

**依存関係**  
なし

---

<a id="任务-t002建立统一边界数据结构"></a>
### タスク T002：統一された境界データ構造の確立

**目的**  
rough / refined / fused の境界に統一されたスキーマを提供し、後続の関数間で生身の dict を渡すことを避ける。

**入力**  
現在の `models.py` 内の `WaveBounds`

**出力**
新しい dataclass を追加：

```python
@dataclass
class BoundaryEstimate:
    onset: Optional[int]
    peak1: Optional[int]
    peak2: Optional[int]
    offset: Optional[int]
    confidence: float
    method: str
    flags: list[str]
```

**コアタスク**
- `BoundaryEstimate` を追加
- `LeadBoundaryResult` を追加
- `BeatBoundaryResult` を追加
- raw candidate / refined / fused の3段階結果をサポート

**受け入れ基準**
- すべての delineation サブモジュールが統一された構造を返す
- `None` の境界ケースでクラッシュしない
- JSON のシリアライズをサポート

**優先度**  
P0

**依存関係**  
T001

---

<a id="2-质量门控与基础可靠性"></a>
## 2. 品質ゲートと基本信頼性

<a id="任务-t003实现-lead-level-质量评分重构"></a>
### タスク T003：リードレベルの品質スコアリングの再構築を実装

**目的**  
境界検出がすべてのリードを等しく扱うのではなく、まずリード品質ゲートを実装する。

**入力**
- 12リードの ECG
- 元のサンプリングレート
- 現在の `quality.py`

**出力**
各リードの出力：

- `baseline_wander_score`
- `muscle_noise_score`
- `powerline_score`
- `flatline_score`
- `clipping_score`
- `lead_reliable_for_p`
- `lead_reliable_for_qrs`
- `lead_reliable_for_t`
- `lead_reliable_for_qt`

**コアアルゴリズム**
- ベースラインウォンダー：低周波エネルギー比 + ビートベースラインシフト
- 筋電ノイズ：QRS 外の高周波エネルギー
- パワーライン：50/60 Hz の狭帯域ピーク
- フラットライン：微分値がゼロに近い割合
- クリッピング：飽和/繰り返しサンプリングの検出

**実装のポイント**
- `quality/metrics.py` を追加
- `quality/gating.py` を追加
- 波形タイプごとに異なる信頼性閾値を定義可能にする

**受け入れ基準**
- ノイズの多い ECG の信頼性の低いリードが識別される
- きれいな ECG の大部分のリードのスコアが安定している
- 可視化結果と一致する

**優先度**  
P0

**依存関係**  
T002

---

<a id="任务-t004实现-beat-level-质量评分"></a>
### タスク T004：ビートレベルの品質スコアリングを実装

**目的**  
「全体のリード品質」を「各ビートの局所品質」にドリルダウンする。

**入力**
- 各ビートの切り出し信号
- リードレベルの品質

**出力**
各ビートの出力：

- `beat_noise_score`
- `beat_baseline_shift`
- `beat_template_corr`
- `beat_measurement_reliable`

**コアアルゴリズム**
- ビート内の局所 SNR
- ビートとグループテンプレートの相関性
- PR 区の安定性
- T 領域のノイズレベル

**受け入れ基準**
- 期外収縮 / ノイズの多いビートの信頼度が、きれいな主要ビートより低い
- グループ内の異常ビートは除外可能

**優先度**  
P0

**依存関係**  
T003、T008

---

<a id="3-多导联-qrs-检测与锚点生成"></a>
## 3. 多リード QRS 検出とアンカー生成

<a id="任务-t005升级多导联-qrs-detector"></a>
### タスク T005：多リード QRS デテクタのアップグレード

**目的**  
現在の QRS デテクタを明確な多リード融合ソリューションとして固定し、delineation 全体のアンカー層とする。

**入力**
- 12リードの ECG
- `fs`

**出力**
- `r_locs`
- `qrs_candidate_windows`
- `qrs_detector_confidence`

**コアアルゴリズム**
1. リード選択：`I, II, V1-V6`
2. z-score 標準化
3. 多リードベクトルエネルギーの構築
4. `5–30 Hz` バンドパス
5. 微分 + 二乗 + MWI
6. 適応閾値 + 不応期
7. 参照リードの局所的な R 波微調整

**実装のポイント**
- 現在の実装思路を保持
- パラメータ設定クラスを追加
- デテクタのデバッグ出力を追加

**受け入れ基準**
- 通常の洞調律の検出が安定している
- PVC / BBB / 低電圧下でも再現率が許容範囲内
- 単一リード II に依存しない

**優先度**  
P0

**依存関係**  
なし

---

<a id="任务-t006qrs-onsetoffset-两阶段精修"></a>
### タスク T006：QRS onset/offset の2段階微調整

**目的**  
QRS はエネルギー閾値による外側拡張だけでなく、representative-aware な微調整を追加する。

**入力**
- `r_locs`
- グループ representative beats
- リード品質

**出力**
- `qrs_on`
- `qrs_off`
- `qrs_on_confidence`
- `qrs_off_confidence`

**コアアルゴリズム**
- 粗段階：局所エネルギーの上昇/回落
- 精段階：
  - 1階微分の顕著な上昇点
  - 2階微分のゼロクロス
  - terminal force の末端転換
  - representative beat との整合性修正

**実装のポイント**
- `delineate/qrs_bounds.py` を新規作成
- まず representative beat で高信頼度の位置特定を行う
- 単一ビートは小窓内でのみ修正を許可

**受け入れ基準**
- 広い QRS の境界が現在版より安定
- 多峰性の QRS でも `qrs_off` が早期に切断されない
- QRS の持続時間のジッターが低減

**優先度**  
P0

**依存関係**  
T005、T010

---

<a id="4-beat-grouping-与-representative-beat"></a>
## 4. ビートグループ化と representative beat

<a id="任务-t007重写-beat-grouping-描述子"></a>
### タスク T007：ビートグループ化記述子の書き直し

**目的**  
グループ化がリズムだけでなく、テンプレート化された測定にも直接役立つようにする。

**入力**
- `r_locs`
- 各ビートの局所波形
- リード品質

**出力**
各ビートの記述子：

- `rr_prev`
- `rr_next`
- `qrs_width_rough`
- `vector_area`
- `paced_flag`
- `template_embedding`
- `lead_corr_signature`

**コアアルゴリズム**
- ベクトル振幅スニペット
- PCA 埋め込み
- 形態コサイン距離
- RR 特徴量

**受け入れ基準**
- 主要ビートファミリーが自動的に Group 1 を形成する
- PVC と正常ビートは分離可能
- ペースドビートは非ペースドビートと混在しない

**優先度**  
P0

**依存関係**  
T005

---

<a id="任务-t008实现两阶段分组器"></a>
### タスク T008：2段階グループ化器を実装

**目的**  
まず粗く分類し、その後細かく分類することで、純粋なクラスタリングの不安定性を減らす。

**入力**
- ビート記述子

**出力**
- 各ビートの `group_id`
- `group_summary`

**コアアルゴリズム**
- 第1段階：ルールによる粗分類
  - ペースド / 非ペースド
  - 狭い / 広い
  - 早期 / 非早期
- 第2段階：グループ内の形態クラスタリング
  - 階層型クラスタリングまたは DBSCAN
  - 距離尺度：テンプレート相互相関

**受け入れ基準**
- 最大5グループ
- Group 1 はメンバー数が最も多いグループ
- 同じ ECG に対する再実行でグループ化結果が安定している

**優先度**  
P0

**依存関係**  
T007

---

<a id="任务-t009实现-representative-beat-生成增强版"></a>
### タスク T009：representative beat 生成の強化版を実装

**目的**  
representative beat を後続の境界微調整の主要入力とする。

**入力**
- グループメンバーのビートスニペット
- ビートレベルの品質

**出力**
各グループ、各リードごとに出力：

- representative beat
- `member_count`
- `mean_template_corr`
- `outlier_count`
- `beat_to_beat_onset_std`
- `beat_to_beat_offset_std`

**コアアルゴリズム**
- R 波の整列
- QRS 内での局所相互相関による微調整
- 外れ値の除外
- 平均ビート（mean beat）の形成
- グループ内の変異の記録

**受入基準**
- representative beat は単一ビートよりも滑らかであること
- グループ内の筋電ノイズが顕著に低下すること
- 変異度のメタデータを保持すること

**優先度**  
P0

**依存関係**  
T004、T008

---

<a id="5-rough-delineation先粗定位区域"></a>
## 5. 粗い区画設定：まず領域を粗く位置特定

<a id="任务-t010实现-approximate-waveform-regions"></a>
### タスク T010：近似波形領域の実装

**目標**  
現在の「ピークを直接探す」方式を、「まず領域を粗く定め、その後境界を細かく設定する」方式に変更する。

**入力**
- `r_locs`
- representative beats
- RR 情報

**出力**
各ビート、各リードの粗い領域：

- `P_search_region`
- `QRS_search_region`
- `T_search_region`

**コアアルゴリズム**
- P 領域：`qrs_on - 260ms` から `qrs_on - 20ms` まで
- QRS 領域：`R ± 120ms`
- T 領域：`qrs_off + 20ms` から `min(qrs_off+500ms, next_expected_p-margin)` まで

**受入基準**
- P/T の検索が誤った領域にまたがらないこと
- 高心拍数時でも T 領域が次の拍の P に重ならないこと
- RR が非常に長い場合でも検索領域が過大にならないこと

**優先度**  
P0

**依存関係**  
T006、T009

---

<a id="6-p-波检测改进"></a>
## 6. P 波検出の改善

<a id="任务-t011实现-multilead-p-candidate-detector"></a>
### タスク T011：マルチリード P 候補検出器の実装

**目標**  
P 波は単一リードの絶対ピーク（abs-peak）に依存せず、マルチリードによる候補生成を行う必要がある。

**入力**
- `P_search_region`
- リード品質
- representative beat

**出力**
各リードの P 候補：

- `peak_candidates`
- `candidate_scores`

**コアアルゴリズム**
- 極値点
- 微分値の転換点
- 局所面積ピーク
- ウェーブレットスケールピーク

**実装のポイント**
- 各リードで 0〜N 個の候補を許可
- 振幅、面積、SNR、グループ内の一貫性に基づいてスコアリング

**受入基準**
- 低振幅の P 波がすべて見逃されないこと
- 負の P 波でも候補が出力されること
- 心房細動サンプルでは、誤検出ではなく信頼度が低下すること

**優先度**  
P1

**依存関係**  
T010

---

<a id="任务-t012实现-p-多导联融合与双分量模型"></a>
### タスク T012：P 波のマルチリード融合と二成分モデルの実装

**目標**  
単峰性 P 波、切欠き P 波、双相性 P 波をサポートする。

**入力**
- 各リードの P 候補
- リードの信頼性
- 代表事前確率（representative priors）

**出力**
- `p_on`
- `p_peak1`
- `p_peak2`
- `p_off`
- `p_biphasic_flag`
- `p_notch_flag`
- `p_confidence`

**コアアルゴリズム**
- 候補リードの可視性スコアリング
- Top-K リードの融合
- 二成分モデル：
  - `P1`、`P2`、切欠き深さ
- 境界は以下の3種類の証拠で共同決定：
  - 振幅のベースライン復帰
  - 累積面積比
  - 微分値/曲率の減衰

**受入基準**
- 通常の洞性 P 波の結果が安定すること
- 双相性 P 波で2つのピークを出力できること
- `PR` のジッターが現在の実装より小さいこと

**優先度**  
P1

**依存関係**  
T011

---

<a id="7-qrs-多峰形态与细节特征"></a>
## 7. QRS の多峰形態と詳細特徴

<a id="任务-t013实现-qrrss-分量提取"></a>
### タスク T013：Q/R/R'/S/S' 成分の抽出の実装

**目標**  
QRS が総時間幅だけでなく、形態解析（morphology）に使用可能な成分も出力するようにする。

**入力**
- 精製された QRS 境界
- representative beat

**出力**
- `Q_amp`
- `R_amp`
- `R_prime_amp`
- `S_amp`
- `S_prime_amp`
- `qrs_num_peaks`
- `qrs_notch_count`
- `qrs_slur_flag`

**コアアルゴリズム**
- QRS 領域の局所極値系列
- ピークとバレーの交互制約
- 小ピークのノイズ除去閾値
- 末端の slur/notch の識別

**受入基準**
- RBBB/LBBB/断片化された QRS の成分特徴がより合理的であること
- 通常の狭い QRS が過剰に多峰として適合されないこと
- 切欠き（notch）の計数が安定すること

**優先度**  
P1

**依存関係**  
T006

---

<a id="任务-t014实现-vat-j-point-精修"></a>
### タスク T014：VAT / J 点の精製の実装

**目標**  
より完全な形態解析リード測定値をサポートする。

**入力**
- 精製された QRS
- representative beat

**出力**
- `vat_ms`
- `j_point_idx`
- `j_point_confidence`

**コアアルゴリズム**
- VAT：開始から主 R 波ピークまでの時間
- J 点：QRS 末端の局所的な安定した曲がり点

**受入基準**
- QRS が広い場合でも VAT のロジックが安定すること
- J 点が ST ノイズによって明らかに引きずられないこと

**優先度**  
P2

**依存関係**  
T013

---

<a id="8-t-波与几何-t-end"></a>
## 8. T 波と幾何学的 T 終点

<a id="任务-t015实现-multilead-t-candidate-detector"></a>
### タスク T015：マルチリード T 候補検出器の実装

**目標**  
T 波は正位、逆位、双相性、切欠き形態を許可し、絶対ピーク（abs-peak）のみを取得しないようにする。

**入力**
- `T_search_region`
- リード品質
- representative beat

**出力**
- `T1_peak`
- `T2_peak`
- `t_main_polarity`
- `t_notch_flag`
- `t_candidate_score`

**コアアルゴリズム**
- 局所極値 + 波形面積 + 曲率
- 主ピークと副ピークのスコアリング
- グループテンプレートとの一貫性制約

**受入基準**
- 逆位 T 波でも安定して出力されること
- 切欠き T 波で主ピークと副ピークを区別できること
- ノイズの多いリードの候補スコアが顕著に低下すること

**優先度**  
P1

**依存関係**  
T010

---

<a id="任务-t016实现-dxl-like-几何-t-end-算法"></a>
### タスク T016：DXL ライクな幾何学的 T 終点アルゴリズムの実装

**目標**  
現在の「閾値ベースライン復帰法」を幾何学的曲がり点法に置き換える。

**入力**
- T 候補ピーク
- 代表 T 波形
- ベースライントレンド

**出力**
- `t_end_candidate`
- `t_end_confidence`
- `t_flat_tail_score`
- `u_wave_present`
- `p_on_t_flag`

**コアアルゴリズム**
1. `T_peak` を取得
2. 右側の検索領域を定義
3. `peak -> search_start` または `peak -> search_end` 参照線を作成
4. 垂直距離を計算
5. 最大垂直距離の点を曲がり点候補とする
6. 傾きに上下限の制約を適用
7. U 波が存在する場合、T-U 底点を優先的に位置特定

**実装のポイント**
- 単一リードで実装
- その後、リード間融合を行う
- 平坦な尾部（flat tail）では信頼度を低下させる

**受入基準**
- 緩やかな T 尾部で、現在の閾値法よりも安定すること
- 大きな U 波サンプルで QT が系統的に長くならないこと
- `T_end` のグループ内分散が低下すること

**優先度**  
P0

**依存関係**  
T015、T018

---

<a id="任务-t017实现-t-波多导联融合"></a>
### タスク T017：T 波のマルチリード融合の実装

**目標**  
単一の不良リードが最終的な T 終点を決定しないようにする。

**入力**
- 各リード毎 `t_end_candidate`
- リードの信頼性
- ビート間変動

**出力**
- `lead_t_end_used`
- `t_end_fused`
- `qt_reliable_leads`

**コアアルゴリズム**
- リードスコアリング：
  - `lead_quality`
  - `t_amp`
  - `t_end_var`
  - `rep_confidence`
- 融合戦略：
  - Top-K リード
  - ロバスト中央値 / 加重中央値

**受入基準**
- ノイズの多いリードは自動的に除外される
- 各リード毎のQT外れ値はグローバルQTに影響しない
- 融合結果は単一リードよりも安定している

**優先度**  
P0

**依存関係**  
T016、T019

---

<a id="9-单搏局部修正与代表搏回投"></a>
## 9. 単一ビート局所修正と代表ビートへのフィードバック

<a id="任务-t018实现-representative-boundary-priors"></a>
### タスク T018：representative-boundary priorsの実装

**目標**  
まず representative beat で高信頼度の境界を決定し、その後グループ内の単一ビートにフィードバックする。

**入力**
- representative beats
- 粗い検索領域

**出力**
- `rep_p_prior`
- `rep_qrs_prior`
- `rep_t_prior`

**コアアルゴリズム**
- まず representative beat で高精度の delineation を完了
- 各境界に対して prior window を出力

**受入基準**
- 代表ビートの境界は再現可能
- グループ内のビートは同一の prior セットを共有

**優先度**  
P0

**依存関係**  
T009、T010、T012、T016

---

<a id="任务-t019实现-beat-wise-local-correction"></a>
### タスク T019：ビート毎の局所修正の実装

**目標**  
各ビートの最終境界を、代表ビートの予測値＋小窓局所修正とする。

**入力**
- 代表ビートの priors
- 単一ビート信号
- 局所品質

**出力**
- 単一ビートの最終 `p_on/off`, `qrs_on/off`, `t_on/off`

**コアアルゴリズム**
- まずテンプレートにアラインメント
- 次に `±20~30 ms` の小窓内のみで：
  - 局所微分修正
  - 局所相互相関修正
  - ベースライン回帰修正

**受入基準**
- グループ内の境界のジッターは現在のバージョンより小さい
- ノイズの多いビートは大きく逸脱しない
- ビート間変動は後続の信頼できるリード判定に使用可能

**優先度**  
P0

**依存関係**  
T018

---

<a id="10-置信度与-reliable-lead-体系"></a>
## 10. 信頼度と信頼できるリード体系

<a id="任务-t020实现-boundary-confidence-引擎"></a>
### タスク T020：境界信頼度エンジンを実装

**目標**  
すべての境界に信頼度を持たせる。そうしないと、後続のQTおよびルール層が調整不能になる。

**入力**
- 局所ノイズ
- 微分の整合性
- テンプレート相関
- リード品質
- ビート変動

**出力**
- `p_confidence`
- `qrs_confidence`
- `t_confidence`
- `t_end_confidence`

**コアアルゴリズム**
複合スコアリング：

```text
conf = f(local_snr, template_corr, baseline_stability, boundary_sharpness, group_consistency)
```

**验收标准**
- 干净的 beat 的置信度显著高
- flat T / noisy P 的置信度明显降低
- confidence 与人工目测趋势一致

**优先级**  
P0

**依赖**  
T004、T019

---

<a id="任务-t021实现-reliable-lead-判定器"></a>
### 任务 T021：实现可靠的导联判定器

**目标**  
复制 DXL 的方法学精神：global QT 只使用可靠的导联，中位数优先于极值。

**输入**
- 各导联 QT
- 各导联 onset/offset 方差
- 导联质量
- 置信度分数

**输出**
- `reliable_for_qt`
- `reliable_for_axis`
- `reliable_for_global_measure`

**核心算法**
导联可靠性条件示例：

```python
reliable = (
    lead_quality < th1 and
    t_end_var < th2 and
    qrs_on_var < th3 and
    rep_confidence > th4 and
    t_end_confidence > th5
)
```

**受け入れ基準**
- 低振幅・高ノイズの誘導は除外される
- グローバルQTは単一の不良誘導によって引きずられることはない
- 信頼できる誘導（reliable lead）の数が適切である

**優先度**  
P0

**依存関係**  
T020

---

<a id="11-global-qt-与-qtc"></a>
## 11. グローバルQTとQTc

<a id="任务-t022实现-per-lead-qt-jt-输出"></a>
### タスク T022：誘導ごとのQT / JT出力の実装

**目標**  
まず各誘導のQTを確実に行い、その後グローバルQTを算出する。

**入力**
- `qrs_on`
- `t_end`
- 信頼度（confidence）
- 信頼性（reliability）

**出力**
- `qt_ms`
- `jt_ms`
- `qt_confidence`

**コアアルゴリズム**
- `QT = T_end - QRS_on`
- `JT = T_end - QRS_off`

**受け入れ基準**
- 各誘導に独立したQT値がある
- 信頼性の低い誘導は数値を保持するが、グローバル計算には使用不可としてマークされる

**優先度**  
P0

**依存関係**  
T017、T021

---

<a id="任务-t023实现-global-qt-qtc"></a>
### タスク T023：グローバルQT / QTcの実装

**目標**  
DXLスタイルのグローバルQT選択を実装する。

**入力**
- 誘導ごとのQT
- 信頼できる誘導のマスク
- RR間隔

**出力**
- `global_qt_ms`
- `qtc_bazett_ms`
- `qtc_fridericia_ms`
- `qt_dispersion_ms`

**コアアルゴリズム**
- `global_qt = median(qt_candidates_from_reliable_leads)`
- `qt_dispersion = max - min`
- QTc はBazettとFridericiaの両方を出力

**受け入れ基準**
- グローバルQTは外れ値となる誘導に対して感度が低い
- 現在の単純な閾値法と比較して、再現性がより良い
- RR間隔が変化した場合、QTcのロジックが正しい

**優先度**  
P0

**依存関係**  
T022

---

<a id="12-pacing-专用分支"></a>
## 12. ペーシング専用ブランチ

<a id="任务-t024实现-pacing-spike-detector"></a>
### タスク T024：ペーシングスパイク検出器の実装

**目標**  
ペーシングスパイクを個別に識別し、ペースドビート（paced beat）の境界決定の前処理を行う。

**入力**
- 生の高忠実度ECG
- `fs`

**出力**
- `spike_events`
- `spike_confidence`
- `paced_status`

**コアアルゴリズム**
- 高周波チャネルの`>100 Hz`
- 極狭パルス幅の検出
- 差分ピーク＋面積制約
- 複数誘導の時間クラスタリング

**受け入れ基準**
- ペーシングスパイクは通常の狭いQRSと混同されない
- 明らかなスパイクが検出される
- 偽スパイクの数が制御可能である

**優先度**  
P1

**依存関係**  
T005

---

<a id="任务-t025paced-beat-delineation-分支"></a>
### タスク T025：ペースドビートの境界決定ブランチ

**目標**  
ペースドビートには通常のQRSモデルを直接適用できない。

**入力**
- ペースドビート
- スパイクイベント
- 代表的なペースドテンプレート

**出力**
- ペースドビートの`qrs_on/off`
- `stim_to_qrs_ms`
- `ventricular_paced_flag`

**コアアルゴリズム**
- スパイク後の広いQRSパターン
- QRSのオンセットはより早期に設定可能
- 終端オフセットはより保守的に設定

**受け入れ基準**
- ペースドQRSの時間幅は現在のバージョンよりも合理的である
- スパイクとQRSの関係は説明可能である

**優先度**  
P2

**依存関係**  
T024

---

<a id="13-morphology-字段补全"></a>
## 13. 形態学フィールドの補完

<a id="任务-t026补全-lead-level-morphology-measurements"></a>
### タスク T026：誘導レベルの形態学測定値の補完

**目標**  
ライブラリの出力をExtended Measurementsレポートの粒度に近づける。

**入力**
- 最終的な境界結果
- 振幅 / 面積 / 傾き計算機

**出力**
各誘導・各ビート、または代表的な誘導に対して少なくとも以下を補完：

- P波：`on/peak1/peak2/off/amp/area/dur`
- QRS：`Q/R/R'/S/S'/dur/area/notch/slur/VAT`
- ST間隔：`J_point/ST_on/ST_mid/ST_80/ST_end/ST_slope`
- T波：`on/peak1/peak2/off/amp/area/dur/U_wave_flag`

**受け入れ基準**
- フィールド名が統一されている
- JSON / データフレームのエクスポートが完全である
- 欠損フィールドには明確な`None`のセマンティクスがある

**優先度**  
P1

**依存関係**  
T012、T013、T014、T016

---

<a id="14-可视化与调试工具"></a>
## 14. 可視化およびデバッグツール

<a id="任务-t027实现边界叠加可视化"></a>
### タスク T027：境界の重ね合わせ可視化の実装

**目標**  
可視化がないと、波形境界アルゴリズムの調整が困難である。

**入力**
- ECG
- 境界結果
- 品質/信頼度

**出力**
デバッグ用図：

- 単一誘導のビート図
- representative beat図
- P/QRS/T境界の重ね合わせ
- 信頼度の注釈

**コア機能**
- matplotlibの描画
- 不良ビートのハイライト表示
- 信頼できる誘導のマーク

**受け入れ基準**
- 1行のコードで特定の誘導・特定のビートの境界図を描画できる
- 可視化はエラー調査に直接使用できる

**優先度**  
P0

**依存関係**  
T019、T020

---

<a id="任务-t028实现回归快照工具"></a>
### タスク T028：回帰スナップショットツールの実装

**目標**  
アルゴリズムの変更を追跡可能にし、「ここは直ったが、あそこが壊れた」状況を避ける。

**入力**
- 固定されたECGサンプルセット
- 現在のバージョンの出力

**出力**
- JSONスナップショット
- PNGオーバーレイ
- 差分レポート

**受け入れ基準**
- 変更のたびに主要な境界の偏移を自動的に比較できる
- 回帰テスト失敗時にサンプルリストを提供する

**優先度**  
P1

**依存関係**  
T027

---

<a id="15-验证与基准"></a>
## 15. 検証およびベンチマーク

<a id="任务-t029建立-synthetic-signal-benchmark"></a>
### タスク T029：合成信号ベンチマークの確立

**目標**  
まず制御可能な信号でオンセット/オフセットを検証し、その後実データに適用する。

**入力**
- 合成ECGジェネレーター
- 制御可能なノイズインジェクター

**出力**
評価指標：
- オンセット誤差
- オフセット誤差
- QT誤差
- ST誤差
- ノイズロバスト性曲線

**受け入れ基準**
- ベースラインウォンダー / 筋電ノイズ / パワーラインノイズを個別に注入できる
- 誤差統計が自動的に生成される

**優先度**  
P0

**依存関係**  
T023

---

<a id="任务-t030建立专家标注数据评测"></a>
### タスク T030：専門家アノテーションデータによる評価の確立

**目標**  
実際のECGにおいて、境界の再現性と誤差分布を検証する。

**入力**
- アノテーション付きデータセット
- 現在のアルゴリズムの出力

**出力**
- `P_on/off error`
- `QRS_on/off error`
- `T_end error`
- `PR/QRS/QT error`

**受け入れ基準**
- 評価コードは繰り返し実行可能である
- 平均 / 標準偏差 / パーセンタイルを出力する
- シナリオ別統計をサポート：洞調律、広いQRS、低振幅T波、ノイズECG

**優先度**  
P1

**依存関係**  
T029

---

<a id="16-里程碑计划"></a>
## 16. マイルストーン計画

<a id="milestone-m1可用的第二版测量内核"></a>
## マイルストーン M1：使用可能な第2版測定コア
**含まれるタスク**
- T001 ~ T010
- T016
- T018 ~ T023
- T027
- T029

**完了後の機能**
- QRSおよびT終端の精度が現在のバージョンより明らかに向上
- representative beatが実際に境界位置決定に参加
- グローバルQTはreliable-lead medianを採用

---

<a id="milestone-m2形态特征增强"></a>
## マイルストーンM2：波形特徴の強化
**含まれるタスク**
- T011 ～ T015
- T026
- T028
- T030

**完了後の機能**
- P/T 双成分
- QRS 多峰
- 波形導出測定値がより完全になる

---

<a id="milestone-m3paced-高复杂度场景"></a>
## マイルストーンM3：ペースメーカ／高複雑度シナリオ
**含まれるタスク**
- T024
- T025
- T014

**完了後の機能**
- ペースドビート専用ブランチ
- VAT／J点の測定がより完全になる

---

<a id="17-优先级总表"></a>
## 17. 優先度一覧表

### P0
- T001
- T002
- T003
- T004
- T005
- T006
- T007
- T008
- T009
- T010
- T016
- T017
- T018
- T019
- T020
- T021
- T022
- T023
- T027
- T029

### P1
- T011
- T012
- T013
- T015
- T024
- T026
- T028
- T030

### P2
- T014
- T025

---

<a id="18-definition-of-done"></a>
## 18. 完了定義（Definition of Done）

タスクは以下の5項目をすべて満たした場合のみ、完了とみなされます：

- コードがメインブランチにマージ済み
- ユニットテストが合格
- 少なくとも1つの可視化例が人的チェックを通過
- 回帰スナップショットに異常なドリフトなし
- ドキュメントの補完：入力、出力、失敗モード、パラメータ説明

---

<a id="19-建议的开发顺序"></a>
## 19. 推奨開発順序

コストパフォーマンスが最も高い順に実施：

1. `T001-T010`
2. `T016-T023`
3. `T027-T029`
4. `T011-T015, T026`
5. `T024-T025, T030`

この順序の理由は単純です：  
まず **QRS アンカー、representative beat、幾何学的T終点、reliable leads、グローバルQT** というメインチェーンを打通することで、最大の効果を得られ、かつDXLマニュアルで公開されている方法論の重点と最も一致します。

---

<a id="20-建议你立刻开工的首批-todo"></a>
## 20. 直ちに着手すべき最初のTODOリスト

```text
[ ] T001 重构 delineate 目录
[ ] T002 建立统一边界数据结构
[ ] T003 lead-level 质量评分重构
[ ] T005 升级多导联 QRS detector
[ ] T007 重写 beat grouping 描述子
[ ] T008 实现两阶段分组器
[ ] T009 representative beat 增强版
[ ] T010 approximate waveform regions
[ ] T016 几何 T-end
[ ] T018 representative-boundary priors
[ ] T019 beat-wise local correction
[ ] T020 boundary confidence 引擎
[ ] T021 reliable lead 判定器
[ ] T022 per-lead QT / JT
[ ] T023 global QT / QTc
[ ] T027 边界叠加可视化
[ ] T029 synthetic benchmark
```
