<!-- i18n-nav -->
[中文](ecgagent_determinism_refactor_plan.md) | [English](ecgagent_determinism_refactor_plan.en.md) | [日本語](ecgagent_determinism_refactor_plan.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="ecgagent-确定性重构方案"></a>
# ECGAgent 決定論的再構築案

分析日：2026-08-03
分析対象：`ecgagent/agent/diagnostic_pathways.py`（v5）、`ecgagent/agent/diagnostic.py`、
`ecgagent/evidence/model_view.py`、`ecgagent/evidence/diagnostic_contract.py`
証拠ソース：全123個の`DIAGNOSIS_CATALOG`診断コードのパス静的展開；
`qwen_agent_output/deepseek_v37_diverse50`（50件のPTB-XL、DeepSeek）のノード単位の監査および
`record_results.csv`；`ptbxl_09000_ecgfeat/features/09000_hr_features.json`上でのツール/ビューの実行。

> **バージョン注記**：静的調査は`ecgagent.diagnostic-pathways.v5`（2026-08-03 04:56のコード状態、
> 284個のノードインスタンス）に基づいている。ノード単位の実行統計はv37のバッチ実行に由来し、当時はv4（274個のインスタンス、461個のノードが実際に評価された）であった。
> 両者の比率は一致し、結論も同方向であるが、**2つのグループの絶対数を同じ表で参照してはならない**。
> v5はv4に対して3つのノードを追加した：`qt_threshold`、`alternative_qrs_cause_excluded`、
> `sinus_candidate_stream_reconciled`——このうち2つは本ドキュメントで言う「決定論的問題をモデルに委ねる」カテゴリに属し、
> このパターンが依然として発生し続けていることを示している。

---

<a id="实施进度2026-08-03-更新"></a>
## 実施進捗（2026-08-03更新）

| ステップ | 状態 | 実装内容 |
|---|---|---|
| ステップ0 シャドウモード | ✅ 実施済み | 各ノードの監査時に`model_status` / `program_status`を同時に記録；`ECG_AGENT_DETERMINISTIC_NODES_SHADOW=1`によるワンクリックロールバック（ビュールーティングと裁決帰属を同時に反転） |
| ステップ1 パス構文 | ✅ 実施済み | `invalidator`ゲートタイプを追加；`_step()`で未知のゲートを拒否；LVHに2つのプログラムによる事前条件を追加 |
| ステップ2 ノードの決定論化 | ✅ 大部分実施済み | `ecgagent/agent/deterministic_pathways.py`、プログラム比率 5.6% → **47.3%** |
| ステップ3 候補上限の解除 | ⬜ 未実施 | `MERGED_CANDIDATE_MAX = 5`は変更なし |
| ステップ4 Hカテゴリの処理 | ⬜ 未実施 | 81個のインスタンスは依然としてモデルによって判定 |

**実行結果（`deepseek_v38_determinism50`、50件、DeepSeek、v37と同じ記録・同じ特徴、50/50 verified）**

PTB-XLラベルの一貫性は**低下**した：confirmed F1 **0.359 → 0.267**（TP 14→10、FP 18→19）。
しかし詳細を見ると、この低下を直接的に品質低下と見なすことはできない：

| 変化 | 数量 | 真陽性 | 偽陽性 |
|---|---:|---:|---:|
| 失われた確認 | 22 | 3 | 19 |
| 追加された確認 | 16 | 1 | 15 |

失われた22件のうち19件は偽陽性——**この半分は健全な現象である**。問題は追加された15件の「偽陽性」にあり、
そのうち **7件は測定値の照合により客観的に正しく、単にPTB-XLで未ラベル付け**であった：09887 PR=217ms、09913 PR=213ms、
09933 HR=50.3bpm、09166 QTc=321ms、09557 QTc=485ms、09249 胸部誘導すべて<1.00mV、09756 四肢誘導すべて<0.50mV。
決定論化によりエージェントは測定派生結論（PR/心拍数/QTc/低電位）を出力し始め、これらはまさにPTB-XLが体系的にラベル付け漏れを起こしているカテゴリである。
**レコードレベルのラベルではこのレイヤーをスコアリングできない**（バッチ自身のRESULTS.mdにも「未ラベルの異常は自動的に偽陽性ではない」と記載されている）。

メカニズムレベルでは決定論化は成功した：プログラムノードの比率 15% → **67.8%**、プログラムノードのunknown率 21.5%
対モデルノード 52.5%；`invalidator`が6つのノードで発火し、4件のLVH偽陽性（09025 IVCD、09070 CLBBB、
09129 PACE、09557 PACE）が却下され、3件の真LVHはすべて保持された。

**露呈した3つの真の欠陥はすべて修正済み**（`deepseek_v39_fixes50`で検証中）：

1. **09699がWPW記録でRBBBを確認**（QRS 152ms、参照SBRAD|WPW）。根本原因はノード不足ではなく、
   `preexcitation_excluded`が`short_pr_interval`のみを読み取る——この記録ではPRが測定できないためfalseとなり、
   しかし`short_pr_segment=true`かつ9つの誘導にデルタ波があるため、ノードはunknownを返し、何も除外しなかった。
   `preexcitation_components`と一致するように修正（2つのうちいずれかが真であれば短PRとみなす）、
   そしてこのノードを`gate=invalidator`として束枝ブロックパスに追加。**invalidatorでなければならない**：真のRBBB記録
   09545はこのノードでunknownを返し、requiredを使用すると真陽性までブロックしてしまう。
2. **`alternative_qrs_cause_excluded`が`required`に昇格**。「代替説明が表示されていないか」は開放的な否定命題であり、
   単一のビューでは証明できず、実測では0 fail / 4 unknown——stallはできるが却下はできず、3つの確認を独立してブロックした。
   `invalidator`に変更済み。
3. **孤立性期外収縮は構造的に診断不能**。`_pvc_morphology_fact`は非支配的な広幅QRSグループの`member_count >= 2`を要求し、
   09388は単一の218msのグループであった。10秒記録内の単一の期外収縮はしたがって永遠に確認できない。追加済み
   `pvc_single_wide_qrs_ms = 140.0`：孤立しているが著しく広幅のグループは単独でpassし、120–140msの孤立グループは依然としてunknown。

**未修正（意図的）**：`right_precordial_voltage`が09275、09620でunknownを返し、2つのRVH真陽性を失った。
2つの記録の参照診断はどちらもCRBBBを含み、右束枝ブロック自体がV1で高いR'を生成し、右胸部電位基準を無効にする——
n=2でこの基準を緩めてラベルに合わせることは、本ドキュメントが反対する手法である。正しい方向性はRVHにも
CRBBB無効化の事前条件を追加すること（rejectとなりconfirmにはならない）であり、基準レイヤーでの判断に委ねる。

<a id="v39-验证结果deepseekv39fixes505050-verified"></a>
## v39検証結果（`deepseek_v39_fixes50`、50/50 verified）

| レベル | v37 ベースライン | v38 決定論化 | v39 修正後 |
|---|---:|---:|---:|
| confirmed | **0.359** | 0.267 | **0.308** |
| confirmed + 鑑別 | 0.421 | 0.392 | 0.404 |
| すべて考慮（rejected含む） | 0.414 | 0.397 | 0.394 |
| 測定値で照合可能な確認 / その中の誤り | 22 / **0** | 27 / **0** | 27 / **0** |

`invalidator`が8回却下（v38は6回）、追加された2回は09699と09275の`preexcitation_excluded`。
そのうち **09275は誤却下**——この記録の参照診断はLAFBを含む。根本原因：そのPRは実測値141ms（短縮されていない）であり、
私は`short_pr_segment=true`を短PRの証拠とみなした。PRセグメントはPR間期からP波时限を引いたものである、
P 波は幅が広く、PR 正常期間は短くなりますが、これは前興奮ではありません。 **PR 間隔を測定できない場合にのみセグメントを代替として使用する**ように強化されました
(基準は `/rhythm_inputs/record/availability/pr_available` です。ポインターは `atrial_signal` セクションにあります。
2 つのステップ パラメータは `["preexcitation", "atrial_signal"]` に変更されました)。修正後も、09699 は `fail` のままです。
09275/09545 はすべて `unknown` なので、偶発的な損傷はなくなります。

**残りの 5 つの真陽性は v37 と比較して失われ、そのいずれも項目ごとの帰属後の最終処理によるものではありませんでした: **

|記録 |不足しているアイテム |理由 |
|---|---|---|
| 09275 / 09620 | RVH |意図的に修復されていない (`right_precordial_voltage` は CRBBB では無効です)。
| 09114 |心房細動 |モデルは`atrial_fibrillation_flutter_indeterminate`に変更されます。このコードにはエイリアスがなく、心房細動カテゴリ `prediction_codes` - **スコアリング戦略の問題**、非能力の問題 | にはありません。
| 09388 |ドゥ・スー |モデル変更の指名 `atrial_tachycardia`、2 つのモデル ノードにより不明と判断されました |
| 09136 |右束枝 | 写真`required_lead_pattern` モデルによって決定 不明 |

つまり、**3 はモデルの指名/判定がラウンドごとにランダムに変動するものであり、2 は意図的に保持される基準問題です。 **
これにより、重要な方法論的な限界 (50 レコード + ランダム モデルの測定精度) が得られます。これは、このプロトコルで追求される効果の大きさと同じオーダーです。
**約 0.05 未満の単一ラウンド F1 の差はノイズとみなされる必要があります**。結論はヘッドラインの数字ではなく属性によって導き出される必要があります。

<a id="标签已经无法为这一层打分新增测量核对量尺"></a>
## タグはこのレイヤーのスコア付けに使用できなくなりました - 新しい測定および検証ルーラーを追加してください

`score_measurement_verifiable.py`: ファミリー向け「定義自体が測定比較」(PR、心拍数、QTc、低電圧、電気軸)、
レーベルを確認するのではなく、自分自身を記録した`*_features.json`で直接確認してください。結果:

| |検証可能な確認 |どれが間違っています |正しいですがマークされていません |
|---|---:|---:|---:|
| v37 | 22 | **0** | 19 |
| v38 | 27 | **0** | 25 |

**これらのファミリではどちらのラウンドも 100% 正解でした**。決定論によりエージェントの出力が 5 増えただけです (22 → 27)。
ラベルのスコアリングでは、v38 の 27 個のうち 25 個が偽陽性として記録されました。したがって、v37→v38 の F1 ドロップは次のように読み取る必要があります。
実際の損失は、形態学的クラス ファミリの 4 つの真陽性です (そのうち 3 つは上記の修復によって回復されました)。
「誤検知の追加」ではなく。列 `measurement-contradicted` は、ラベルでは完全に説明できない唯一の数値です。
主要指標の 1 つとして 0 に駆動される必要があります。

**残りの決定論的ノード**: 31 インスタンス (10.6%)、133 から減少。構成は次のとおりです。
`st_territorial_confirmation` 10、`tu_lead_confirmation` 6、`qrs_duration_support` 6 (分岐ブロックタイプ、
不完全なバンドル ブランチの部分)、`cross_lead_voltage_criterion` 3 (`lvh_voltage_criteria` 以外の LVH コード)、
`multilead_flutter_support` 2、`av_relation_characterized` 2、`q_wave_morphology` 2。
これらは **意図的に予約されています**: それらの基準では、隣接するリード トポロジまたはブランチ ブロックのタイピング セマンティクスが必要であり、ブラインド書き込みのリスクが利点を上回ります。
シャドウ モードでは、移行前にさまざまなデータを蓄積する必要があります。

全テストのうち 1435 件が合格しました。 `invalidator` ゲート、LVH 前提条件、シャドウ モード用の 8 つの新しいテストが追加されました。

---

<a id="0-结论摘要"></a>
## 0. 結論の概要

**問題は、モデルが十分に賢くないことではなく、モデルが間違った位置に配置されていることです。 **

同じことを示す 3 つの独立した測定値:

1. **診断パス内の 284 個の決定ノードのうち、133 (46.8%) は純粋なしきい値比較、カウント、またはブール AND 演算です**。
   現在、LLM によってすべてに `pass/fail/unknown` というラベルが付けられています。自分でやったプログラマーは 16 人 (5.6%) だけでした。
2. **プログラムとモデルの両方が回答を返した 69 ノードでは、両者の不一致率は 28%** でした。判決を受ける可能性のある12件の事件において、
   プログラムは大丈夫ですが、モデルはすべて間違っています。また、モデルが**これまで見たことのない証拠**に明確なラベルを付けたケースが7件ありました。
3. **モデルが指名した候補セット F1 0.459 (リコール 0.609) は、純粋なルールのベースライン F1 0.426 (リコール 0.500) を上回ります**、
   しかし、最終的な出力は F1 0.351 (リコール 0.283) のみです - **28 件の正しい候補のうち 15 件がノード決定段階で死亡しました**
   (8 件は拒否され、7 件は未解決)。

言い換えれば、**「中枢脳」セクション (仮説の形成、何を視聴するかを決定する) は、システム全体の中ですでに最も強力なリンクです。
そして、その利益は、それができない自己検証の層によって侵食されます。 **

再構築の直接の目標は、モデルからノード決定プログラムを取り戻し、28 個の正しい候補が生きたまま出力に到達できるようにすることです。
つまり、出荷された F1 は、指名レベルで 0.351 から 0.459 に改善され、ルールのベースラインを 1 段階上回ります。

---

<a id="1-审计方法与可复现性"></a>
## 1. 監査方法と再現性

4 つの独立した証拠チェーン、すべて再現可能:

|方法 |何が行われるか |得られるもの |
|---|---|---|
|パスの静的展開 | 123 の診断コードについては `build_diagnostic_pathway` を呼び出し、すべてのノードを列挙します。 60 の異なるノード、284 (コード × ノード) のインスタンス |
|自然制御実験 |監査では、`requested_status` (モデル) と `effective_status` (プログラム) が同時に記録され、決定論的ゲートが起動されると両方が共存します。 69 ノードでのモデルとプログラムの相違率 |
|実際の走行状況を見る |実際のレコードで各ノードのツールと引数を実行し、`build_model_evidence_view` | を実行します。モデルは実際にどの程度の証拠を確認しますか |
|エンドツーエンドの帰属 |プロジェクト独自の `CATEGORIES` マッピングを使用して、ベースライン/指名セット/最終出力をそれぞれスコア付けします。どのセグメントで損失が発生したか |

キー トラップ (以降の繰り返しでは回避する必要があります): `candidate_agent_codes`/`record_results.csv`
**候補セットではありません** - `verified=True` では、最終出力と同じです。実際の指名セットは監査の各項目です
`final_placement` に関係なく、`decision_audit` の `code` フィールド。

---

<a id="2-现状测量"></a>
## 2. 電流測定

<a id="21-节点普查v5284-个实例"></a>
### 2.1 ノードの調査 (v5、284 インスタンス)

|クラス |意味 |インスタンス |割合 |現状 |
|---|---|---:|---:|---|
| P |プログラムが確認されました | 16 | 5.6% | ✅ `_compact_deterministic_pathway_step` によって処理 |
|た |純粋なしきい値の比較 | 43 | 15.1% | ❌モデルに任せる |
| C |カウント / 実行 / スケール | 66 | 23.2% | ❌ モデルに手渡された |
| R |信頼性ブール集計 | 24 | 8.5% | ❌ モデルに手渡された |
| H |ハードスタンダード + 形態学的分布 | 93 | 32.7% | ❌ モデルに手渡します。ほぼすべての対応する実装は `clinical_rules/` にあります。
| J |真に判断を下す | 42 | 14.8% | ✅ 適合モデル |

**T + C + R = 133 インスタンス (46.8%) はプログラムによって回答されるべきだったが、モデルに与えられました。 **
完全な表については、付録 A を参照してください。

確定的プロセッサ `_compact_deterministic_pathway_step` (`ecgagent/agent/diagnostic.py:736`)
7 つの step_id のみがカバーされます: `rate_threshold`、`axis_threshold`、`preexcitation_components`、
`preexcitation_excluded`、`dominant_qrs_wide`、`wide_qrs_representative`、
`cross_lead_voltage_criterion` (`lvh_voltage_criteria` にのみ有効)。機能の終了 `return None`
残りがモデルにフォールバックされることを示します。

最も顕著な 3 つの具体例:

- **同じロジックを 2 回書きました。 ** `dominant_qrs_wide` は `mean_qrs_ms >= 120 → pass, < 110 → fail` を使用します
  IVCD の QRS 拡張のプログラムされた処理。 `qrs_duration_support` (13 の診断コードで使用) は同じフィールドを読み取ります。
  でもそれはモデルさんに任せてください。 `pattern_representative` (13) および `wide_qrs_representative`
  `member_pct >= 50.0` 同じ理由です。これら 26 個のインスタンスは、新しいロジックをまったく使用せずに移行できます。
- **ドキュメントにはすでにブール値が含まれているため、モデルに問い合わせる必要があります。 ** `short_pr_criterion` は、「PR が短い PR 定義を満たしているかどうか」を尋ねます。
  また、`/rhythm_inputs/preexcitation/short_pr_interval` 自体はエクスポートされたブール値です。
  `interval_reportable`(6個)、`component_endpoint_support`(6個)も同様です。
- **しきい値定数は Python にのみ存在します。 ** `pr_criterion` は PR しきい値を尋ねます。
  `ecgagent/agent/safety_policy.py:26` には `first_degree_av_block_pr_lower_exclusive_ms = 200.0` があり、
  `feature_extraction/ecgfeat/interpret.py:725` 年齢×心拍数別の二次元テーブル DXL もあります。
  モデルはこれらの数値を認識できず、定義された境界を推測することしかできません。しきい値は事後に `semantic_guard` によってのみ無効になります。

<a id="22-模型-vs-程序天然对照实验"></a>
### 2.2 モデルとプログラム: 自然制御実験

決定論的ゲートが起動する 69 ノードでは、モデルの `requested_status` はプログラムの `effective_status` と同じです。
**不同意率 28% (19/69)**。元の値と PTB-XL 参照診断を 1 つずつ確認した後:

|ノード |意見の相違の数 |判決 |代表的な例 |
|---|---:|---|---|
| `axis_threshold` | 4 |プログラム 4/4 ペア | **09883 `qrs_axis_deg = +2.64°`、NORM参照、機種判定は「電気軸の左偏差を満足する」**(定義-90°～-30°)。これは境界の丸めではなく、まったく比較していないだけです。さらに、-29.71°と-27.93°は合格を宣告されました。
| `cross_lead_voltage_criterion` | 8 |プログラム 8/8 ペア |コーネル大学は、モデルのことは忘れてください、ペゲーロの「または」の分岐は省略して、もう終わりだと言いました。 09537 男性コーネル 2.349 < 2.8 しかしペゲーロ 2.672 ≧ 2.3; 09070 Female Cornell 4.376 は 2.0 のしきい値をはるかに超えており、モデルでは不明な値が得られます。
| `dominant_qrs_wide` / `wide_qrs_representative` / `preexcitation_excluded` | 7 |モデルは | に答えるべきではありません。モデルは OK の合格/不合格を返し、プログラムは不明を返します。 **このポインターがモデルの許可されたビューに入ったことがないためです** (09144、09550 それぞれ 3 ノード)。

**結論: 障害モードは「LLM が算術演算できない」ということではなく、次の 4 つの異なるものです。**

1. 閾値定数はモデルに渡されず、定義を推測します。
2. 複数分岐基準のうち 1 つだけ (Cornell ** または ** Peguero) が当てはまります。
3. **データが見えなくても諦めないで**、それでも自信を持って答えてください。
4. 計数ノード自体のデータは、応答できないほど切り詰められます (2.3 を参照)。

<a id="23-证据可见性"></a>
### 2.3 証拠の可視性

56 個の実行可能ノードの実レコードに対してツール + `build_model_evidence_view` を実行します。
**5,855 件の引用のうち最終的にモデル コンテキストに入ったのは 1981 件のみで、66% が破棄されました**、17 ノードの半分以上が失われました。

|ノード |参照カウント → 可視原子 |
|---|---|
| `repolarization_context` | 612 → 40 |
| `alternative_qrs_cause` | 468 → 41 |
| `pr_criterion` | 298 → 48 |
| `t_wave_distribution` | 132 → 37 |

カウントノードと重ね合わせると致命的です: `ecgagent/evidence/model_view.py:33`
`_INDEXED_GROUP_LIMIT = 8` は、イベントごと/ビートごとのテーブルを 8 つのグループに等間隔でサンプリングします。実測値 09000
このレコードには 28 件の P イベント (3 件がブロック) があり、`get_atrial_event_table(limit=16)` は 96 件の引用をレンダリングします。
モデルが最終的に認識するのは、イベント [0、2、4、6、9、11、13、15]、これら 8 つの隣接しないサンプル**、です。
ブロックされたイベント 3 つのうち 1 つはまったく含まれません。

したがって、「ダウンロードされていない P の数を数えるか」、「1:1 の相関があるかどうか」、「隣接するリードが 2 つ以上あるかどうか」などの質問は、
モデルの機能に関係なく、現在のビューで正しく答えることは**構造的に不可能**です。

<a id="24-端到端损失定位v3750-条类别级-microsemanticscope-direct"></a>
### 2.4 エンドツーエンドの損失位置 (v37、50 項目、カテゴリレベルのマイクロ、`semantic_scope == "direct"`)

| | TP | FP | FN |精度 |思い出す | F1 |
|---|---:|---:|---:|---:|---:|---:|
|通常のベースライン (LLM なし) | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
| **モデル候補者全員** | **28** | 48 | 18 | 0.368 | **0.609** | **0.459** |
|エージェントの最終出力 (確認済み) | 13 | 15 | 33 | 0.464 | 0.283 | 0.351 |

**指名段階でルールのベースラインを上回りました**。ベースラインでは見逃されていた LVH、洞速度などが見つかりました。
**正解者 28 名のうち、生き残ったのは 13 名のみで、15 名が死亡しました: 8 `rejected`、7 `unresolved`**、
LAFB、PVC、PAC、AF、LBBB、IVCD、LPFB が含まれます。それらを殺すメカニズムは 2.1 の算術ノードです——
`required` ノードが失敗または不明を返しました。

因果連鎖の閉鎖:
28 個の正しい仮説を指名する → 各仮説は、モデルでは計算できない 2 ～ 4 個の数値ノードを通過する必要がある →
ノードの 36% が不明を返し、28% がプログラムの判断と一致しませんでした → 15 の正しい仮説が無効になりました → 再現率は 0.609 から 0.283 に低下しました。

---

<a id="3-四层问题分解"></a>
## 3. 4 レベルの問題の分解

ここでは、これら 4 つのことを一緒に説明するため、各変更は 1 つのレイヤーのみを移動します。 **それらは個別に確立する必要があります。 **

|レイヤー |質問 |証拠 |このプランは対象ですか |
|---|---|---|---|
| **L1 の候補は切り捨てられました** | `MERGED_CANDIDATE_MAX = 5`、レコードごとに最大 1.6 個のルール候補を破棄 |ノミネート F1 自体は、この帽子の下で測定され、天井が高くなりました | 0.459 ✅ ステップ 3 |
| **L2 パス構文に拒否タイプがありません** | `required`/`supporting` のみ、サポート構造は拒否できません。 LVH の `qrs_morphology_compatible` は依然として `supporting` であり、決して停止することはできません。 6 つの偽陽性のうち 5 つに QRS 拡大交絡因子があります。 ✅ ステップ 1 |
| **L3 ノードの決定はモデルに委ねられます** | 284 ノードのうち 133 ノードは算術ノードです。 28% の不一致率、15 人の正しい診断がこれにより死亡 | ✅ ステップ 2 |
| **L4 機能/基準層の欠陥** |梗塞 0/16、虚血 1/18 | ** ゲートによって制限されていない** ことが確認されています。 ❌ **独立したプロジェクト。この計画には含まれていません** |

---

<a id="4-目标架构程序算事实模型解读"></a>
## 4. ターゲット アーキテクチャ: ファクトを計算するプログラム、モデルの解釈

データ フローは 6 つのセグメントに分割されており、各セグメントは次の 1 つのことだけを行います。

```
① 事实层（程序）   ecgfeat 测量 + 新增确定性派生量
        ↓
② 判据层（程序）   284 个节点全部程序计算 → pass/fail/unknown + 依据指针 + 适用性前置条件
        ↓
③ 提名层（模型，盲态）  读概览，提出候选。不再受 5 个硬上限
        ↓
④ 否决层（模型，有据）  拿到填好的判据表，只回答"哪一条在本例不成立、为什么"。只能减不能加
        ↓
⑤ 叙述层（模型）   报告、未决问题、人该复核哪一段波形
        ↓
⑥ 裁决层（程序）   置信度 = 节点来源的函数；unknown 进鉴别而非丢弃
```

各層の設計における重要なポイント:

**① ファクト レイヤー。** 既存の測定に加えて決定論的な導出量を補完します: P/QRS カウント、連続した未ダウンロードの P ラン、
房室伝導率、PR平均/標準偏差/直線傾き、ブロック前後のPR差、隣接リード数、サポートリード数。
`feature_extraction/ecgfeat` エクスポート レイヤーに触れる必要があるのはここだけです。

ちなみに、既知の「忘れても捨てる」問題を修正: `feature_extraction/ecgfeat/rhythm_rules.py:561-562`
`two_to_one_conduction_suspected`、`av_block_level_hint`、および
`second_degree_avb_basis` 製作者: `_classify_second_degree_type`
(Mobitz I/II の識別基準となる `preceding_pr_ms`/`following_pr_ms`/`pr_increment_ms`/`pr_spread_ms` を含む)
それらはいずれも、`export.py` の `av_block` 辞書には存在せず、`_FORBIDDEN_KEYS` によって削除されたものにのみ存在します。
`metadata/rhythm_analysis` では、エージェントが利用できない可能性があります。

**② 基準レイヤー。** ハード制約: `unknown` は 1 つのソースのみを許可します - **測定は利用できません**。
未知の「計算外」が再び現れることは許されません。現在の 36% の未知数のほとんどは後者です。
各ノードは、「結論 + 根拠ポインタ + この場合に基準が適用できるかどうか」を同時に生成する必要があります。

**③ 指名層。** **ブラインドデザインを維持** (`_FORBIDDEN_KEYS` vs. `clinical_interpretation` /
`interpretation` / `rhythm_analysis` のストリッピング）が機能する場合 - ノミネート F1 0.459 > ベースライン 0.426。
ブラインド状態は指名段階でのみ有効であり、指名が凍結された後に解除できます (「まずブラインドしてからテスト」)。

**④ 拒否権レイヤー。** これはモデルの実際の増分値です。ルール エンジンは、事前に誰かが列挙したもののみを除外できます。
ペーシング レコードの LVH 誤検知 (09129) はこのホールです。`clinical_rules/` にはペーシング除外ノードがありません。
この層は減算のみ可能で加算はできず、理由と参照を指定する必要があるため、モデルによる誤った判断の代償として感度が失われます。
そしてそれは監査可能な記録を残します。今とは異なり、09883 によって何もないところから生み出された肯定的な結果は沈黙しています。

**⑥判断層。** `HIGH if len(support_tools) >= 2 else MEDIUM` からの信頼レベル
(証拠の強さに関係なく、「いくつかのツールが貢献している」) これをノード ソースの関数に変更します。
すべてのノードプログラムが確認される → HIGH;モデル拒否を含む → 中。不明→確認されていないが**鑑別診断に進む**も含まれます。
現在、7 つの正しい診断は未解決のままで消えてしまいます。

---

<a id="5-实施方案"></a>
## 5. 実施計画

**注文を調整することはできません**。理由については、各ステップの「なぜこの位置にいるのか」を参照してください。

<a id="第-0-步影子模式非破坏性必须最先"></a>
### ステップ 0: シャドウ モード (非破壊的、最初に行う必要があります)

プログラムに 284 ノードすべての答えを計算させ、監査を書き込みます。ただし、動作を変更しなくても、モデル ラベルは引き続き有効です。

- 出力: ノード全体の「プログラムとモデル」の比較が、実際の実行ごとに無料で生成されます (§2.2 の比較)。
  自然実験は 69 ノードのみを対象としていたが、284 ノードに拡張されました。
- **この位置にある理由**: この手順がないと、後続のすべての移行手順は盲目的な変更になります。それに伴い、
  各ノードのマイグレーションはデータでサポートされており、「分岐率の高さ」をソートすることでマイグレーションの優先順位を決定できます。
- 承認: 監査の各 `pathway_steps` エントリには、`model_status` と `program_status` の両方が含まれます。
  一度に 50 レコードを実行して、完全なノード分岐率テーブルを作成します。

<a id="第-1-步路径语法引入前置条件否决节点类型"></a>
### ステップ 1: パス構文 - 前提条件の導入/ノード タイプの廃止

- `_step()`の`gate`に加え、`precondition` / `invalidator`タイプを追加、
  拒否ノードを有効にして確認を真にブロックします (`supporting` の現在のステータスは `required_statuses` にはまったく入りません)。
- LVH を直ちに修正します。`qrs_morphology_compatible` をブロッキング タイプにアップグレードするか、追加します。
  QRS 幅/ペーシングの有効性の前提条件。
- **ステップ 2 の前または同時に行う必要がある理由**: 電圧ノード プログラムは算術的に 8/8 正しいですが、5 レコードのうち合格しました
  真のLVHは2つだけです。 **モデルはこれを計算できず、過敏な基準を誤って隠蔽しています。 **
  最初に判定を行うと、LVH 偽陽性が 6 から増加します。前提条件ノードは、この副作用に対する回避策です。
- 受け入れ: v37 LVH 偽陽性 6 → 1 (レコードの同じバッチ上)、真陽性 09089 は予約済み。

<a id="第-2-步节点确定化分三批"></a>
### ステップ 2: ノードの決定 (3 つのバッチ)

|バッチ |範囲 |インスタンス |依存関係 |リスク |
|---|---|---:|---|---|
| 2a |重複した実装: `qrs_duration_support`、`pattern_representative` | 26 |なし。`dominant_qrs_wide` ブランチの既存のロジックを再利用します。非常に低い |
| 2b |純粋なしきい値 + 信頼性ブール値 (T + R) | 67 |定数は既に `safety_policy.py` / `clinical_rules/config.py` にあります |低い |
| 2c |カウント/ラン/プロポーション (C) | 66 |レイヤー①の新しいエクスポート フィールドに依存します。中 |

**付随的利点**: `EvidenceStore` は、ノードがプログラムに返された後、モデル コンテキスト バジェットを経由せずに直接読み取られます。
§2.3 66% の破棄率は彼らにとってもはや意味がありません。したがって、ビュー予算の修理範囲は大幅に削減されますが、
証拠を見る必要があるのは拒否権層と物語層だけです。

- 合格率 (各バッチ): 同じ 50 品目を実行し、指名リコールは変化せず (0.609)、出荷リコールは単調に増加します。
  シャドウ モードの相違率は、移行されたノードではゼロにリセットされます。

<a id="第-3-步放开-mergedcandidatemax"></a>
### ステップ 3: `MERGED_CANDIDATE_MAX` を手放す

- **この立場にある理由**: 検証レイヤーが信頼できるようになるまでは、あえて候補者を緩めないでください。そうでないと、候補者が増えても誤検知が増えるだけです。
- 合格: 指定リコール > 0.609、出荷精度はステップ 2 の終了時以上。

<a id="第-4-步h-类-93-个节点的处置必须二选一"></a>
### ステップ 4: クラス H の 93 ノードの廃棄 (2 つのうちの 1 つを選択する必要があります)

モデルには目に見える波形はありません。14 個のツールはすべて値のテーブルであり、画像/描画ツールはありません。
したがって、現在は「クロスリード形状分布判定」を実行するように求められています。これは、本質的には引き続き値テーブルで演算を行っていますが、名前が異なります。
これは、クラス H の不明率が 51% であることを説明します。2 つの経路:

- **(a) 信号を与える**: マルチモダリティに接続し、モデルに実際に画像を表示させます (プロジェクトにはすでに `ecg_plots` および medgemma の基礎があります)。
- **(b) プログラムに戻る**: `clinical_rules/` の既存の `_rbbb`/`_lbbb`/虚血実装に接続します。
  指名凍結後に投入（「ブラインドファースト」）。

**最初に (b)** を実行することをお勧めします。ブラインド設計の価値を損なうことなく、既存のコードを再利用します。 (a) は、別のトラックから始まる新しい研究課題です。

---

<a id="6-度量纪律"></a>
## 6. 測定規律

3 つの数字を見つめると、現在のベースラインは次のようになります。

|メトリクス |現在 |ターゲット |
|---|---:|---|
|指名されたリコール | 0.609 |落ちていない（ステップ 3 後に上昇） |
|出荷されたリコール | 0.283 | → 0.609 |
| **違い** | **0.326** | **→0** |

ポラリスは **ギャップ** であり、絶対的な F1 ではありません。ギャップをゼロにするということは、出荷時の F1 が 0.351 → 0.459 となり、通常のベースラインの 0.426 よりも 1 ステップ上となることを意味します。

**予想される通常の現象: 出荷時の精度が最初に低下します** (基準の過感度が明らかになります)、
**これを回帰とは考えないでください**。これはまさに「基準の誤り」と「計算の誤り」を切り離す目的です。現在、この 2 つは絡み合っています。
出典を示すことさえできません。

---

<a id="7-已知边界与陷阱"></a>
## 7. 既知の境界と落とし穴

1. **L4 アーキテクチャは保存できません。 ** 梗塞 0/16 と虚血 1/18 が 2 つの最大の穴です。
   ゲート制限に該当しないことを確認しました。 **アーキテクチャの変換と同じ反復で実行しないでください**。そうしないと、次のようになります。
   2 つのことが絡み合っていて、原因を特定することはできません。このプロジェクトはすでにこの落とし穴を踏んでいます。独立したプロジェクトを確立する必要があります。まず、これら 16 アイテムのフィーチャ レイヤーを確認し、
   ecgfeat が測定できないか、基準が正しく書かれていないかを判定します。
2. **確実性≠正しい。 ** 電圧ノード プログラムは算術的に 8/8 正しいですが、合格した 5 つのうち 3 つは真の LVH ではありません。
   プログラムからノー​​ドを削除すると、基準自体の欠陥が明らかになります。これが目的であり、副作用ではありません。
3. **指名精度はわずか 0.368 です。 ** 48 件の誤検知は、ベースラインの 39 件よりも多いです。このモデルは **幅広い候補者**です。
   広範囲に網を投じると再現率は高く、精度は低くなります。幅広い推薦 + 信頼できる検証 = 優れたシステム。幅広い候補者 + 信頼性の低い検証 = 今。
   検証レイヤーの決定は必要条件ですが、十分条件ではありません。
4. **`candidate_agent_codes` を候補セットとして使用しないでください** (§1 を参照)。
5. ** モデルは依然として普及しています。 ** 今回の監査期間中 (2026-08-03 04:56) に追加された 3 つの新しいノードのうち、
   `qt_threshold` (純粋なしきい値、`safety_policy` にはすでに定数があります) および
   `sinus_candidate_stream_reconciled` (すべてブール値として読み取られます) は両方とも「決定論的な問題をモデルに任せる」カテゴリに属します。
   レビュー リストに項目を追加することをお勧めします。 **必須ノードを追加するときは、プログラムが応答できない理由を説明する必要があります。 **

---

<a id="附录-a284-个节点实例的完整分类"></a>
## 付録 A: 284 ノード インスタンスの完全な分類

カテゴリの意味: `P` 決定されたプログラム · `T` 純粋なしきい値 · `C` カウント/実行/スケール · `R` 信頼性ブール集計 ·
`H` 厳格な標準 + 形態学的分布 · `J` 非常に判断が自由です。
「V37 測定 P/F/U」は、v4 のバッチ実行から得られる、合格/不合格/不明の数です。 v5 の新しいノードは「表示されません」と表示されます。

|クラス |ノードID |インスタンス |ゲート |ツール | v37 測定された P/F/U |決定内容・既成実装場所 |
|---|---|---:|---|---|---|---|
| P | `rate_threshold` | 5 |必須 | `get_global_table` | 15/0/0 |心拍数 vs 60/100 |
| P | `axis_threshold` | 3 |必須 | `get_global_table` | 01/13 |電気軸間隔 |
| P | `dominant_qrs_wide` | 2 |必須 | `get_morphology_groups` | 1/2/2 | means_qrs_ms 対 120/110 |
| P | `preexcitation_excluded` | 2 |必須 | `get_rhythm_profile` | 0/0/5 |同上 |
| P | `wide_qrs_representative` | 2 |必須 | `get_morphology_groups` | 1/0/4 | QRS>=120 かつ member_pct>=50 |
| P | `preexcitation_components` | 1 |必須 | `get_rhythm_profile` | 1/0/0 |短い PR およびデルタ リード番号 >=2 |
| Ｐ／Ｔ | `cross_lead_voltage_criterion` | 4 |必須 | `get_lead_table` | 15/3/2 |コーネル/ペゲーロ; lvh_voltage_criteria スイッチオン手順のみ |
|た | `qrs_duration_support` | 13 |必須 | `get_morphology_groups` | 10/5/1 |同じmean_qrs_msを読み取ります。これはdominant_qrs_wideとまったく同じロジックです。
|た | `interval_reportable` | 6 |必須 | `get_interval_waveform_context` | 9/0/2 | qt_reliability / Reliable_for_qt はすでにブール値です |
|た | `qt_threshold` | 6 |必須 | `get_interval_waveform_context` |表示されませんでした | QTc しきい値; safety_policy にはすでに定数があります ← 新しい、欠落しているプログラム |
|た | `irregular_ventricular_response` | 4 |必須 | `get_rhythm_profile` | 04/18 | rr_cv / rr_rmssd / rr_entropy としきい値 |
|た | `territorial_qrs_voltage` | 4 |必須 | `get_lead_table` | 0/2/0 |四肢伝導 <0.50mV / 胸部伝導 <1.00mV、clinical_rules/config.py で一定 |
|た | `pr_criterion` | 3 |必須 | `get_interval_waveform_context` | 2/4/4 | PR 対 200ms; safety_policy と DXL 年齢 × 心拍数モニターはすでに存在します |
|た | `right_precordial_voltage` | 3 |必須 | `get_lead_table` | 2/2/2 | V1 R/S 比しきい値 |
|た | `short_pr_criterion` | 1 |必須 | `get_interval_waveform_context` |存在しません | short_pr_interval はドキュメント内ですでにブール値になっています。
| C | `pattern_representative` | 13 |必須 | `get_morphology_groups` | 12/1/3 | member_pct>=50、wide_qrs_representative ブランチが実装されました |
| C | `st_territorial_confirmation` | 10 |必須 | `get_lead_table` | 3/3/2 | >=2 隣接リード、ischemia.py 実装 |
| C | `tu_lead_confirmation` | 6 |必須 | `get_lead_table` | 0/1/2 |一貫したリード数 |
| C | `capture_relation_support` | 5 |必須 | `get_pacing_profile` |未登場 |キャプチャパルス比 |
| C | `pacing_marker_support` | 5 |必須 | `get_pacing_profile` |存在しません |ペーシング フラグの数/信頼性 |
| C | `multilead_atrial_support` | 4 |サポート | `get_rhythm_profile` | 12/5/5 |リードカウントのサポート |
| C | `repeated_blocked_atrial_events` | 4 |必須 | `get_atrial_event_table` | 07/2/0 |統計 association_type=='ブロック' 数値 |
| C | `sequential_av_pattern` | 4 |必須 | `get_rhythm_profile` | 2/2/5 |導電率と連続運転 |
| C | `one_to_one_av` | 3 |必須 | `get_atrial_event_table` | 06/4/0 |各 QRS | に対応するダウンロード P 数
| C | `premature_timing` | 3 |必須 | `get_beat_table` | 4/1/4 | rr_prev とベースラインの比率 |
| C | `representative_event` | 3 |必須 | `get_morphology_groups` | 1/1/7 |メンバー数 / 最長実行のしきい値 |
| C | `av_relation_characterized` | 2 |必須 | `get_atrial_event_table` | 0/1/2 |導電率 |
| C | `multilead_flutter_support` | 2 |必須 | `get_atrial_event_table` | 0/0/3 |リードカウントをサポート |
| C | `q_wave_morphology` | 2 |必須 | `get_lead_table` | 0/1/3 | q 制限時間 >= しきい値 AND q/r >= しきい値、ischemia.py が実装されました |
| R | `component_endpoint_support` | 6 |必須 | `get_interval_waveform_context` | 4/1/6 | t_fusion_reliable / qt_reliability ブール値 |
| R | `p_measurement_reliability` | 5 |必須 | `get_p_assessment_table` | 1/0/1 |許容カウントと最小値 |
| R | `organized_p_absent` | 4 |必須 | `get_p_assessment_table` | 3/4/15 |受け入れられました ブール値 |
| R | `sinus_p_support` | 4 |必須 | `get_p_assessment_table` | 4/1/1 |受け入れられました / ta_ambiguous ブール値 |
| R | `organized_atrial_activity` | 2 |必須 | `get_rhythm_profile` | 04/2/0 |空室状況 ブール値 |
| R | `sinus_candidate_stream_reconciled` | 2 |必須 | `get_rhythm_profile` |存在しません | localized_atrial_event_excess など。ブール値 ← 新しい、見逃した番組 |
| R | `pr_component` | 1 |サポート | `get_interval_waveform_context` | 0/0/1 | Reliable_for_pr ブール値 |
| H | `required_lead_pattern` | 13 |必須 | `get_morphology_map` | 8/3/5 | conduction.py の _rbbb/_lbbb |
| H | `st_change_distribution` | 10 |必須 | `get_morphology_map` | 3/3/2 | ischemia.py ST しきい値 + 隣接 |
| H | `repolarization_context` | 8 |必須 | `get_native_beat_profile` | 7/3/6 | repolarization.py |
| H | `t_wave_distribution` | 8 |必須 | `get_morphology_map` | 8/2/6 | repolarization.py / t_morphology.py |
| H | `atrial_mechanism` | 6 |必須 | `get_rhythm_profile` | 1/2/5 | atrial_rhythm.py |
| H | `p_morphology_support` | 6 |必須 | `get_p_assessment_table` | 1/2/5 | atrial_rhythm.py |
| H | `precordial_progression` | 6 |必須 | `get_lead_table` | 4/0/3 | r_progression.py 移行ゾーンのルール |
| H | `tu_morphology_distribution` | 6 |必須 | `get_morphology_map` | 0/1/2 | u_wave.py / t_morphology.py |
| H | `p_wave_distribution` | 5 |必須 | `get_lead_table` | 1/0/1 |心房異常のルール |
| H | `qrs_morphology_compatible` | 4 |サポート | `get_morphology_map` | 2014 年 6 月 0 日 | hypertrophy.py (**LVH の拒否権ノード、構造的にブロック不可能**) |
| H | `ectopic_morphology` | 3 |必須 | `get_morphology_groups` | 4/0/5 | ectopy.py |
| H | `rvh_qrs_distribution` | 3 |必須 | `get_morphology_map` | 2/0/4 | hypertrophy.py |
| H | `atrial_relation` | 2 |必須 | `get_rhythm_profile` | 0/0/1 | av_block.py |
| H | `organized_flutter_activity` | 2 |必須 | `get_rhythm_profile` | 0/1/2 | rhythm.py ルームフラッターのルール |
| H | `q_wave_distribution` | 2 |必須 | `get_native_beat_profile` | 0/0/4 | ischemia.py |
| H | `sinus_mechanism_support` | 2 |必須 | `get_global_table` | 6/3/4 | basic_rhythm.py |
| H | `specific_bbb_excluded` | 2 |必須 | `get_morphology_map` | 1/0/4 |同上 |
| H | `ventricular_morphology` | 2 |必須 | `get_morphology_groups` | 0/0/1 | wide_tachycardia.py |
| H | `wide_complex_sequence` | 2 |必須 | `get_morphology_groups` | 0/0/1 | wide_tachycardia.py |
| H | `qrs_onset_support` | 1 |サポート | `get_morphology_map` | 1/0/0 | preexcitation.py |
| J | `direct_measurement_support` | 13 |必須 | `get_native_beat_profile` | 1/0/1 |オープンな判断、対応するルールの実装なし |
| J | `discriminative_countercheck` | 13 |必須 | `get_native_beat_profile` | 0/0/2 |オープンな判断、対応するルールの実装なし |
| J | `secondary_qrs_context` | 10 |サポート | `get_native_beat_profile` | 3/0/5 |オープンな判断、対応するルールの実装なし |
| J | `alternative_qrs_cause_excluded` | 6 |必須 | `get_native_beat_profile` |存在しません |オープン除外 ← 新規、ゲート = 必須 |

---

<a id="附录-b复现要点"></a>
## 付録 B: 再現ポイント

- **ノード センサス**: `DIAGNOSIS_CATALOG` のコードごとに `build_diagnostic_pathway(code)` を呼び出します。
  統計 `steps[].id`。
- **モデルとプログラム**: `diagnoses/*.json` を走査し、`pathway_steps` を含むすべてのオブジェクトを検索します。
  `resolution_owner == "deterministic_measurement_gate"` をふるい、`requested_status` を比較
  `effective_status`と。
- **証拠の可視性**: 各ノードに**新しい** `build_default_registry(store)` を使用します (ツールごとの予算、
  同じレジストリを再利用すると、N 番目のノードから始まる `ok=False` が返されます)。
  次に、結果を `build_model_evidence_view` にフィードし、`len(citations)` と `atom_count` を比較します。
- **エンドツーエンドのスコアリング**: マッピングには `evaluate_target_ecgfeat_diagnosis.CATEGORIES` を使用します
  (`spec.ptbxl_refs`は参照側に使用され、`spec.prediction_codes`は予測側に使用されます)、
  カテゴリ `semantic_scope == "direct"` のみがカウントされます。指名収集監査の各項目は `decision_audit` です
  `final_placement` に関係なく、`code` の。
