<!-- i18n-nav -->
[中文](philips_rhythm_coverage_matrix.md) | [English](philips_rhythm_coverage_matrix.en.md) | [日本語](philips_rhythm_coverage_matrix.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="philips-rhythm-spec-coverage-matrix"></a>
# Philips Rhythm 仕様対象範囲マトリックス

<a id="purpose"></a>
## 目的

このメモは、`要点.pdf` に要約されたリズム分析要件をマップします。
現在のコードベースに戻り、各項目のステータスが表示されます。

- `Implemented`: 仕様の意図と実質的に一致する具体的なコード パスがあります。
- `Partial`: 関連するコードはありますが、簡素化されているか、不完全であるか、必要な制御フローが欠落しています。
- `Missing`: 現在のリポジトリに対応する実装が見つかりませんでした。

<a id="short-answer"></a>
## 短い答え

**2026-07-12 更新:** 以下の歴史的なギャップに関する記述は、もはやすべてではありません
現在。エクストラクターはペーシング検出/制御、QRST-template を実行するようになりました。
減算、AF/AFL 残差分析、一時停止/エスケープ/補間候補
ロジック、事前励起可用性ゲーティング、および構造化された候補リゾルバー。
公開ガイドライン `clinical_interpretation` レイヤーは、検証されたもののみを消費します
リズムの証拠。よりリッチな DXL からインスピレーションを得た `rhythm_inputs` と
`statement_engine` は参照専用のままです。これはまだフィリップスの主張ではありません
または DXL と同等ですが、臨床的に検証されていません。

現在のリポジトリは完全なリズム分析フローを**実装していません**
`philips_rhythm.pdf` / `要点.pdf` に記載されています。

現在では、次のように説明するのが適切です。

- `DXL-inspired` 測定および解釈パイプライン
- ペーシング優先測定ポリシー、構造化リズム候補、QRST
  減算および AF/AFL 信号解析ブランチが実装されました
- ただし、完全なフィリップス独自の第 2 章コード マトリクスと
  臨床的に検証された同等性

その枠組みの証拠:

- プロジェクトの README には、「`DXL` スタイルであるが、プライベート DXL 実装ではない」と明示的に記載されています。
  `feature_extraction/README.md:3-20`
- 解釈層は自らを研究足場と呼びます。
  `feature_extraction/ecgfeat/interpret.py:1-16`
- フェーズ 1 のギャップ解消計画では、目標は同等性を主張するのではなく、Philips/DXL の要件セットに「大幅に近づく」ことであると述べています。
  `docs/superpowers/plans/2026-06-26-ecg-spec-gap-closure-phase1.md:5-20`
- README には、次のステップとして「リズム/モルフォロジー ルール エンジンのオーバーレイ」がまだ記載されています。
  `feature_extraction/README.md:69-75`

<a id="about-要点pdf"></a>
## `要点.pdf` について

`要点.pdf` は、第 2 章の合理的なエンジニアリングの要約です。
`philips_rhythm.pdf`。構造、疑似コード、構成ガイダンスを追加します。
また、Philips PDF にはそのままでは存在しない**テスト提案も含まれますが、
大まかなルールの内訳はソース文書と一致しています。

<a id="coverage-matrix"></a>
## カバレッジ マトリックス

|スペックエリア | `要点.pdf` からの要件 |現在のコードの証拠 |ステータス |メモ |
| --- | --- | --- | --- | --- |
|トップレベルのオーケストレーション |基本リズム前のペーシングファーストの流れ | `ECGFeatureExtractor` は、最終測定の前にスパイクの検出/検証、状態のキャプチャ、測定ビートの選択、およびペーシング ポリシーを実行します。実装済み |公開された最終ステートメントでは、依然として保守的な可用性ゲートが使用されています。 |
|トップレベルのオーケストレーション | `primary_statement` + `additional_statements` 出力オブジェクト | `rhythm_statements.py`、`statement_engine.py`;エクスポートされた `statement_engine` 候補/最終/抑制リスト |実装済み (参照レイヤー) |統合されたパブリック最終出力は、`clinical_interpretation` で個別に解決されます。 |
|トップレベルのオーケストレーション | `stop_further_interpretation` / `bypass_remaining_algorithm` フラグ |一致するランタイム出力オブジェクトが見つかりません。計画/仕様テキストのみがそれらについて言及しています。欠落しています | `要点.pdf` で説明されているコントロール プレーンは、具体的な結果タイプとしては存在しません。 |
| トップレベルのオーケストレーション | 優先度/抑制を備えた証拠豊富なステートメントオブジェクト | `CandidateStatement`、`resolve_statement_candidates`、および統合された `RuleEvaluation` | 実装済み | 独自同等性は主張されていません。参照出力と権威ある出力は明確に分離されています。 |
| ペースドリズム | マルチリードペーススパイク検出 | `detect_pacing_spikes` はハイパスフィルタとマルチリード投票クラスタリングを実装：`feature_extraction/ecgfeat/quality.py:153-227` | 実装済み | これは、すでに導入されている最も明確なペースドリズム関連機能です。 |
| ペースドリズム | 通常の抽出器実行におけるデフォルトのペース解析 | ほとんどのランタイムエントリポイントは、`enable_pacing=True` なしで `ECGFeatureExtractor(...)` をインスタンス化：`batch_extract_ecgfeat.py:366`、`evaluate_ludb.py:581`、`compare_annotations.py:784` | 未実装 | 実際には、呼び出し元が明示的に有効にしない限り、ペース機能は通常オフになっています。 |
| ペースドリズム | 連続ペースと間欠ペースを区別し、非ペースのリズムパターン解析を停止 | `api.py` は `paced_beat_ids` だけをタグ付けし、それらをグループ化/区画化に渡す：`feature_extraction/ecgfeat/api.py:114-159` | 部分的 | ビートタグ付けは存在しますが、Philips 方式のリズム制御停止ロジックは見つかりませんでした。 |
| ペースドリズム | `PACED RHYTHM` と `PACED COMPLEXES` のステートメント | ペースステートメントジェネレーターは見つかりませんでした | 未実装 | 現在のコードは、`paced_rhythm` をリズムステートメントではなくブール値として保存しています。 |
| ペースドリズム | デマンドペース動作の検出 | 該当するコードは見つかりませんでした | 未実装 | デマンド動作ロジックは見つかりませんでした。 |
| ペースドリズム | ペースメーカー類似アーティファクトを別個の出力として扱う | `pacemaker_like_artifact` は現在、`gf.paced_rhythm` から派生：`feature_extraction/ecgfeat/interpret.py:2092`、`feature_extraction/ecgfeat/interpret.py:2347` | 部分的 | これは、アーティファクトと真のペースドリズムを区別するという仕様の意図と一致しません。 |
| ペースドリズム | マグネット／感知失敗／キャプチャ失敗ロジック | 該当するコードは見つかりませんでした | 未実装 | 固定レート非同期スパイク解析は見つかりませんでした。 |
| ペースドリズム | 心室ペース下では AF のみを許可し、他の心房リズムラベルを抑制 | 該当するコードは見つかりませんでした | 未実装 | 心室ペース固有の AF 分岐は見つかりませんでした。 |
| ベーシックリズム | 成人の心拍数閾値：頻脈 >= 100、徐脈 < 50、完全房室ブロックのランドマーク < 45 | `feature_extraction/ecgfeat/interpret.py:39-46`、`feature_extraction/ecgfeat/interpret.py:464-473` | 実装済み | 第2章の成人デフォルト値と一致します。 |
| ベーシックリズム | P軸の洞性起源範囲（-30度から120度） | `feature_extraction/ecgfeat/interpret.py:77-78`、`feature_extraction/ecgfeat/interpret.py:2100-2104` | 実装済み | P軸の洞性範囲ロジックが存在します。 |
| ベーシックリズム | 洞性／心房性／房室結節性／心室性リズムに対するベーシックリズムステートメント | `heart_rate_class` や `p_axis_normal` のようなスカラーフィールドのみが計算される：`feature_extraction/ecgfeat/models.py:216-235`、`feature_extraction/ecgfeat/interpret.py:2287-2304` | 部分的 | 測定値は存在しますが、主要なリズムステートメント層は見つかりませんでした。 |
| ベーシックリズム | 低心室率と房室非同期に基づく完全房室ブロック | `complete_avb` は実質的に `hr < 45 bpm`：`feature_extraction/ecgfeat/interpret.py:1574-1607` | 部分的 | 現在のコードは、仕様に記載されている完全な非同期ロジックを必要としません。 |
|基本リズム | AV解離 | PR 範囲 / PR SD ヒューリスティックが存在します: `feature_extraction/ecgfeat/interpret.py:1597-1602` |部分的 |完全なメカニズム指向のルール セットではなく、単純化されたヒューリスティックとして存在します。 |
|基本リズム | QRST subtraction 後の RR 変動と心房信号分析を使用した心房細動 | `atrial.py:build_qrst_subtracted_residual`、`_classify_af_afl`;検証フラグは統一ルールで必要です |実装済み |未検証の減算により、パブリック AF ルール `unavailable` が作成されます。 |
|基本リズム |心房粗動の検出 | `atrial.py` の時間領域およびスペクトル残差解析。 `rhythm_statements.py` の候補者 |実装済み |研究用途のままであり、外部の臨床検証が必要です。 |
|心室前興奮 |デルタ波ベースの予励起サポート |ビートレベルデルタフラグが存在します: `feature_extraction/ecgfeat/models.py:122`、`feature_extraction/ecgfeat/delineate.py:1068-1103`、`feature_extraction/ecgfeat/delineate.py:1297`、`feature_extraction/ecgfeat/delineate.py:1690` |実装済み |ビートレベルでは実際のデルタ波がサポートされています。 |
|心室前興奮 |ショート PR + ワイド QRS + デルタ -> WPW/予励磁判定 | `_detect_wpw`: `feature_extraction/ecgfeat/interpret.py:708-719` |部分的 |存在しますが、`要点.pdf` で説明されているマルチリードしきい値処理に比べて簡素化されています。 |
|心室前興奮 | PR が短い場合の異なるデルタリードカウントしきい値 |一致する構成可能なリードカウント ロジックが見つかりません |欠落しています |現在のコードでは、リード数のしきい値ではなく、`delta_any` を使用します。 |
|心室前興奮 |副経路側の分類 |一致するコードが見つかりません |欠落しています |左右の副経路の分類が見つかりません。 |
|心室前興奮 |予励磁が検出されると、残りのアルゴリズムをバイパスします。バイパス制御オブジェクトが見つかりません |欠落しています | WPW はブール解釈フィールドであり、ハード パイプライン ブランチではありません。 |
|早発コンプレックス |期外拍閾値: RR が 15 パーセント短縮 | `PVC_RR_SHORTENING_PCT = 15.0`: `feature_extraction/ecgfeat/interpret.py:47-49`; `_detect_premature_complexes` で使用: `feature_extraction/ecgfeat/interpret.py:1396-1492` |実装済み |これは仕様概要とよく一致しています。 |
|早発コンプレックス | APC / JPC / VPC 分類 | `_detect_premature_complexes`: `feature_extraction/ecgfeat/interpret.py:1396-1492` |部分的 |存在しますが、完全な形態/極性/補償一時停止ロジックではなく、RR + QRS 幅 + P 信頼度に簡略化されています。 |
|早発コンプレックス | VPC ペア / トリプレット / NSVT |連続した V ランが検出されました: `feature_extraction/ecgfeat/interpret.py:1449-1474`、`feature_extraction/ecgfeat/interpret.py:2340-2341` |部分的 | NSVT サポートは存在しますが、依然として簡略化された分類器内にあります。 |
|早発コンプレックス |心室/上室二連 |パターン検出器が存在します: `feature_extraction/ecgfeat/interpret.py:1495-1538`、`feature_extraction/ecgfeat/interpret.py:2339` |実装済み |シーケンスパターンロジックが存在します。 |
|早発コンプレックス |心室三叉神経 | 心室三叉神経パターン検出器が存在します: `feature_extraction/ecgfeat/interpret.py:1495-1538`、`feature_extraction/ecgfeat/interpret.py:2340` |実装済み |この部分はすでにコーディングされています。 |
|早発コンプレックス |補間された VPC | `rhythm_rules.classify_post_pause_or_interpolated_beats` |部分的 |決定的な独自のラベルではなく、保守的な候補者を送り出します。 |
|早発コンプレックス |異常な上室性期外収縮 |専用の異常分類子が見つかりません。欠落しています |そのモジュールに一致するルールが見つかりませんでした。 |
|一時停止 / エスケープ / AV ブロック | RR > バックグラウンド RR の 140 パーセントの場合、長い一時停止 | `RR_PAUSE_FACTOR = 1.40`: `feature_extraction/ecgfeat/interpret.py:47-49`; `_detect_pauses` で使用: `feature_extraction/ecgfeat/interpret.py:1541-1557` |実装済み |このしきい値は存在します。 |
|一時停止 / エスケープ / AV ブロック |エスケープビートの起点分類 (心房/接合部/心室) | `rhythm_rules.detect_pauses_and_av_block` エクスポート `escape_origin` |部分的 |現在の出力は上室/心室であり、すべての独自の原点ラベルを完全に再現するわけではありません。 |
|一時停止 / エスケープ / AV ブロック | P カウント > QRS カウントからの第 2 度 AV ブロック | `detect_pauses_and_av_block` の心房イベント数と一時証拠部分的 |保守的な証拠の道筋が存在します。フィリップスとの完全な同等性は確立されていません。 |
|一時停止 / エスケープ / AV ブロック |モビッツ I / ヴェンケバッハ | `_detect_wenckebach` ヒューリスティックが存在します: `feature_extraction/ecgfeat/interpret.py:1560-1571`、`feature_extraction/ecgfeat/interpret.py:1603-1605` に接続 |部分的 | PR トレンド ヒューリスティックとして実装されていますが、仕様には完全な一時停止/P カウント フレーミングはありません。 |
| PR・AV実施 |年齢 x 心拍数に応じた AVB1 閾値テーブル | `_PR_AVB1_TABLE`、`_pr_avb1_threshold`、`_classify_pr`: `feature_extraction/ecgfeat/interpret.py:50-62`、`feature_extraction/ecgfeat/interpret.py:560-589` |実装済み |これは最も強力なスペックマッチの 1 つです。 |
| PR・AV実施 |ボーダーラインの長期PRカテゴリー |現在の `pr_class` は `short / normal / avb1 / indeterminate` のみです: `feature_extraction/ecgfeat/models.py:229`、`feature_extraction/ecgfeat/interpret.py:575-589` |欠落しています | `要点.pdf` は、別の境界線の PR 状態を予期します。 |
|レポート / ペイロード |リズムステートメントの構造化された証拠オブジェクト |トップレベルの `rhythm_inputs`、`statement_engine`、および権限のある `clinical_interpretation` |実装済み |従来のステートメント エンジンは参照専用です。統一された AF 投影には検証済みの証拠が必要です。 |
|レポート / ペイロード |デフォルトのアプリ/レポート フローで高度なリズム結果を公開 | `app.py` 概要には、`probable_af`、`heart_rate_class`、`pr_class`、`qrs_width_class`、`BBB`、`WPW`: `app.py:442-450` が含まれます。デモ ステップ 2 では、レート / RR 不規則性 / AF の可能性のみを出力します: `demo_feature_extraction.py:1150-1155` |部分的 |第 2 章の高度なフィールドの多くは内部に存在しますが、完全には表面化されていません。 |
|テスト |第 2 章のリズム ルールを広範囲にカバーするユニット |軽いペーシング テストといくつかの解釈フィールド チェックのみが見つかりました: `tests/test_pacing_grouping.py:13-107`、`tests/test_ecg_report_sheet.py:55-61`、`tests/test_medgemma_ecg_core.py:50-62` |部分的 |カバレッジは、`要点.pdf` で提案されているテスト マトリックスに近くありません。 |

<a id="what-is-already-strong"></a>
## すでに強いもの

これらの領域はすでに仕様と十分に一致しているため、固体のように見えます。
作業を怠るのではなく、基礎を構築します。

- 大人料金のしきい値
- 正弦波範囲の P 軸ロジック
- 年齢 x HR テーブルからの動的な PR AVB1 しきい値
- 15% RR 期外拍閾値
- 140 パーセントの RR 一時停止しきい値
- 二連/三連パターン検出
- ビートレベルのデルタ波のサポート

<a id="biggest-gaps-relative-to-要点pdf"></a>
## `要点.pdf` と比較した最大のギャップ

実装を要約したフィリップスのように動作させることが目標の場合
第 2 章の流れで、最も大きな影響を与える欠落部分は次のとおりです。

1. 残りの独自のリズム コード マトリックスを完成および検証し、
   制御フローのセマンティクスを、パブリックの最終層と混同することなく実現します。

2. より完全な早発/一時停止/AV ブロック ロジック
   - 補間された VPC
   - 異常な上室性期外拍動
   - エスケープビートオリジン
   - P/QRS カウントによる第 2 度房室ブロック

3. 予励磁制御フロー
   - マルチリードデルタしきい値処理
   - 副経路側の分類
   - ダウンストリームのリズム解釈のハードバイパス

<a id="practical-bottom-line"></a>
## 実際的な結論

誰かが「現在のリズム分析は次のとおりに実装されていますか?」と尋ねたら、
`philips_rhythm.pdf` / `要点.pdf`?」の場合、最も正確な答えは次のとおりです。

> リポジトリは、以下を含む意味のあるリズム分析パイプラインを実装します。
> ペーシング ポリシー、構造化ステートメント、QRST-残差 AF/AFL 分析、
> 保守的な公共ルールの予測。それは完全な、または臨床的なものではありません
> フィリップス独自のリズム仕様の実装を検証。
> Philips/DXL からインスピレーションを得た結果は参照のみです。
