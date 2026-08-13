<!-- i18n-nav -->
[中文](ludb_detector_comparison.md) | [English](ludb_detector_comparison.en.md) | [日本語](ludb_detector_comparison.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="ludb-三种-ecg-检测方法对比"></a>
# LUDB 3 つの ECG 検出方法の比較

このツールは、LUDB 1.0.1 エキスパート アノテーションに関する次のメソッドを比較します。

1. プロジェクトアルゴリズム `ecgfeat`;
2. NeuroKit2;
3. BioSPPy。

エントリ スクリプトは、プロジェクトのルート ディレクトリにある `compare_ludb_detectors.py` です。

<a id="1-安装"></a>
## 1. インストール

```bash
cd /workspace/ecg_gemma

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ludb-compare.txt
```

デフォルトのデータ ディレクトリは次のとおりです。

```text
/workspace/ecg_gemma/data/lobachevsky-university-electrocardiography-database-1.0.1
```

このスクリプトは、上記の LUDB ルート ディレクトリとその内部の `data/` ディレクトリの両方を受け入れます。

<a id="2-运行命令"></a>
## 2. コマンドを実行します

まず、レコード 1、リード II で簡単な検証を実行します。

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate

python compare_ludb_detectors.py \
  --records 1 \
  --leads II \
  --out-dir ludb_three_way_smoke
```

最初の 10 レコード、12 個のリードすべてを実行します。

```bash
python compare_ludb_detectors.py \
  --limit 10 \
  --workers 2 \
  --out-dir ludb_three_way_first10
```

完全な LUDB を実行します。

```bash
python compare_ludb_detectors.py \
  --dataset-dir /workspace/ecg_gemma/data/lobachevsky-university-electrocardiography-database-1.0.1 \
  --workers 4 \
  --out-dir /workspace/ecg_gemma/ludb_three_way_comparison
```

各リードのローカル調整後の ecgfeat QRS 基準を使用して照合します。

```bash
python compare_ludb_detectors.py \
  --limit 10 \
  --workers 4 \
  --ecgfeat-r-source lead-fiducial \
  --out-dir ludb_three_way_first10_per_lead_r
```

デフォルトの評価スクリプトは依然として `--ecgfeat-r-source global` であり、これは履歴結果を再現可能に保つために使用されます。
リードバイリードの結果では、明示的に `lead-fiducial` を選択する必要があります。 2 つの `summary.json` を実行すると、
実際のピークソースを記録します。

最も安定したタイミング結果を得るには、`--workers 1` を使用してください。並列実行は検出指標をより速く取得するのに適していますが、プロセスの競合により時間の消費に影響します。

一部のメソッドまたはリードのみを比較します。

```bash
python compare_ludb_detectors.py \
  --methods ecgfeat,neurokit2 \
  --leads II,V1,V5 \
  --limit 10
```

すべてのパラメータを表示します。

```bash
python compare_ludb_detectors.py --help
```

<a id="3-对比范围"></a>
## 3. 比較範囲

LUDB の P 波、QRS 複合体、および T 波の注釈はリードごとに与えられるため、次のようになります。

- NeuroKit2 は、選択された各リードで R ピーク検出と波形定義を独立して実行します。
- BioSPPy は、選択された各リードに対して独立して動作します。
- `ecgfeat` はネイティブ設計を保持します。レコードの 12 リード上のグローバル R ピークを共同で検出し、各リードの P/QRS/T 境界を読み取ります。

3 つのメソッドのパブリック機能はまったく同じではありません。

|方法 | Rピーク | QRS 開始点と終了点 | P ピーク/開始点と終了点 | T ピーク/開始点と終了点 |
|---|---:|---:|---:|---:|
| `ecgfeat` |はい |はい |はい |はい |
| NeuroKit2 |はい |はい |はい |はい |
| BioSPPy パブリック `ecg()` プロセス |はい |いいえ |いいえ |いいえ |

したがって、BioSPPy は、QRS/R ピークの検出と位置比較にのみ参加します。このスクリプトは、BioSPPy によって提供されていない P/T スイープ境界を推測したり改ざんしたりしません。

<a id="4-匹配和指标定义"></a>
## 4. マッチングとインジケーターの定義

各リードは、LUDB の P、QRS、および T トリプルをそれぞれ読み取ります。

```text
( onset, peak symbol, ) offset
```

デフォルトの一致許容値:

- QRS の R ピーク: 75 ミリ秒。
- P/T ピーク: 150 ミリ秒。

次のパラメータを使用して変更できます。

```bash
--r-tolerance-ms 75
--wave-tolerance-ms 150
```

イベントは時系列順に 1 対 1 で照合されます。動的プログラミングでは、最初に TP の数を最大化し、次に同じ TP ソリューション内のピーク絶対誤差を最小化して、検出結果の繰り返しの一致を回避し、単純な貪欲による隣接イベント間の一致の欠落を回避します。

検出指数は、すべての「記録×先行」波イベントに基づいてマイクロ平均されます。

- `sensitivity = TP / (TP + FN)`;
- `precision` (PPV) `= TP / (TP + FP)`;
- `F1 = 2TP / (2TP + FP + FN)`。

ピークのマッチングが成功した後は、波のオンセット、ピーク、オフセットのみが比較されます。位置決め誤差は次のように定義されます。

```text
error_ms = (detected_sample - ground_truth_sample) / fs × 1000
```

境界ごとの出力:

- `bias_ms`: 符号付き平均誤差。
- `sd_ms`: エラー標準偏差。
- `mae_ms`: 平均絶対誤差。
- `median_ae_ms`: 絶対誤差の中央値。
- `p95_ae_ms`: 絶対誤差の 95 パーセンタイル。
- `n`: この境界統計で利用可能な一致の数。

LUDB 専門家によってマークされていない検出イベントが両端に存在する可能性があります。これらのイベントは、標準の定義に従って FP としてカウントされます。記録エッジを無視する必要がある場合は、ルールを事前定義して記録した後で評価戦略を変更する必要があります。結果を確認した後でルールを手動で削除することはできません。

<a id="41---restrict-to-annotated-span可选默认关闭"></a>
### 4.1 `--restrict-to-annotated-span` (オプション、デフォルトでは閉じられています)

LUDB は、各 10 秒レコードの中間セクションのみをマークします (たとえば、レコード 1 のマーク間隔は 1.32 秒～7.94 秒、
レコード 2 は 1.14 秒～8.44 秒です)。範囲外の心拍数は **実際の心拍数ですが、参照ラベルがありません**。
デフォルトの絞りですべてを FP にカウントします。これは検出器の精度ではなく、アノテーションのカバレッジを測定します。
そして**正しいことを行うあらゆる方法を罰してください**。実際の測定では、3 つの方法の PPV はすべて 0.63 ～ 0.84 に減少しました。
QRS の GT は 21843 ですが、各メソッドは 25400 ～ 25920 を検出します。違いは基本的にラベルのないマージナルビートです。

このスイッチを有効にすると、スコアリングは各リードのマークされた間隔内でのみ実行されます。基準は機械的なものであり、主観的な選択は必要ありません。

> 注釈間隔からの距離が **マッチング許容値**を超える検出イベントを破棄します。このようなイベントは定義上、どのようなイベントにも関連付けることはできません。
>真のイベントペアリングなので、注釈がないという理由だけでFPと判断されます。許容範囲内の検出は **すべて維持されます**。

このルールは 3 つの方法に同様に適用され、感度と境界誤差は変わりません。
(真理イベントのセットは変化せず、一致するペアも変化しません) - 精度と F1 のみが解釈可能になります。
`detection_rows.csv` の `det_outside_annotated_span` 列には、毎回ドロップされたイベントの数が記録されます。

**整合性ステートメントを記録する**: このセクションの上記の原則では、評価ルールが「事前定義」されていることが必要です。初めてスイッチが入ります
PPVが異常に低く、厳格な事前登録順を満たしていないことを見て、フルボリューム（200件）で実施された。
したがって、デフォルトではオフのままであり、両方のキャリバーセットの結果が、オンになった後の数値だけでなく、一緒に報告される必要があります。
この基準自体には方法論的な自由度が含まれていないため、依然として有用ですが、読者はその起源を認識する必要があります。

<a id="5-输出文件"></a>
## 5. 出力ファイル

各実行では、`--out-dir` が生成されます。

|ドキュメント |コンテンツ |
|---|---|
| `summary.csv` |手法と波形別にまとめたマイクロアベレージインジケーター |
| `summary_by_lead.csv` |方式・波形・リード別まとめ |
| `record_summary.csv` |各レコードの概要 |
| `detection_rows.csv` |各レコード、リード、メソッド、波形の TP/FP/FN |
| `matched_events.csv` |成功した各マッチング イベントのサンプル位置と境界エラー |
| `runtime.csv` |各アルゴリズム呼び出しに時間がかかる |
| `runtime_summary.csv` |時間のかかる要約 |
| `failures.csv` |アルゴリズム例外とトレースバック |
| `summary.json` |構成、バージョン、メトリック、および失敗の数 |
| `comparison.png` | F1とポジショニングMAEの比較表 |

`failures.csv` は空で、通話が失敗していないことを示します。デフォルト モードでは、アルゴリズム呼び出しが失敗すると、その呼び出しでサポートされているすべての実際のイベントが FN によるサマリーに含まれ、失敗したサンプルをサイレントに破棄することで誤って高い指標が取得されるのを防ぎます。同時に、スクリプトは最終的にゼロ以外の終了コードを返します。最初の失敗時に直接停止するには、`--fail-fast` を使用します。

<a id="6-运行时间的正确理解"></a>
## 6. 実行時間の正しい理解

`ecgfeat` は、完全な 12 リード レコードを一度に処理し、時間単位は `record_12lead` です。 NeuroKit2 および BioSPPy は単一リードで動作し、時間単位は `single_lead` です。 `runtime_summary.csv` はこのユニットを明示的に書き込みます。

したがって、3 つのうちの `mean_seconds` を同じワークロードとして直接比較することはできません。外部ライブラリが 12 リードのレコードを処理するのにかかるシリアル時間を見積もるには、同じレコードに対する 12 リードの呼び出し時間を合計します。厳密なパフォーマンス ベンチマークを実行するには、単一プロセスを使用し、ウォームアップし、ハードウェアを修正し、繰り返し実行する必要があります。

<a id="7-在同一条-ecg-上画-ecgfeat-与-neurokit2-标注"></a>
## 7. ecgfeat と NeuroKit2 注釈を同じ ECG に描画します

`plot_ecgfeat_neurokit_comparison.py` により、両方の検出器が同じ LUDB を読み取ります
同じ元の ECG サンプルとタイムラインに結果を記録して表示します。

完全な記録を描画します。

```bash
python plot_ecgfeat_neurokit_comparison.py 1 \
  --lead II \
  --out ludb_ecgfeat_neurokit_plots/record_1_II_full.png
```

LUDB の N 番目の QRS を押して、ズームインしたシングル ショットをマークします。

```bash
python plot_ecgfeat_neurokit_comparison.py 8 \
  --lead V2 \
  --beat 3 \
  --out ludb_ecgfeat_neurokit_plots/record_8_V2_beat3.png
```

2 ～ 5 番目の QRS に対応する 4 つの連続するハートビートを表示します。

```bash
python plot_ecgfeat_neurokit_comparison.py 8 \
  --lead V2 \
  --beat-range 2 5 \
  --out ludb_ecgfeat_neurokit_plots/record_8_V2_beats2-5.png
```

`--before-sec` と `--after-sec` は、それぞれ前の QRS と最後の QRS を制御します
その後に保持するコンテキストは、デフォルトで 0.45 秒と 0.70 秒になります。

固定期間を選択します:

```bash
python plot_ecgfeat_neurokit_comparison.py 1 \
  --lead II \
  --start-sec 1.5 \
  --end-sec 4.5
```

LUDB エキスパートリファレンスを表示しない:

```bash
python plot_ecgfeat_neurokit_comparison.py 1 --lead II --no-ludb
```

出力図の各行の意味は次のとおりです。

1. 最初の行は、同じ ECG 上の 2 つのアルゴリズムの P/R/T ピークを重ね合わせます。しっかりとしたマークは
   `ecgfeat`、白抜きのマークは NeuroKit2、十字は LUDB。
2. 2 行目は、P/QRS/T 間隔と `ecgfeat` の境界を示します。
3. 3 行目は、P/QRS/T 間隔と NeuroKit2 の境界を示します。
4. デフォルトでは、4 行目に LUDB エキスパート境界が表示されますが、`--no-ludb` を使用する場合は省略されます。

境界パネルでは、点線がオンセット、点線がオフセット、半透明領域が
オンセットとオフセットの間隔。すべてのパネルは同じオリジナルの ECG をプロットします。 2つの検出器はまだ残っています
それぞれの内部前処理を使用します。端末は、現在表示されている時間枠内の相対的な LUDB も出力します。
TP、FP、FN、F1、およびピーク MAE。

<a id="完整-12-导联10-秒对比"></a>
### 完全な 12 リード、10 秒間の比較

`plot_ludb_12lead_10s_comparison.py` は 12 リードのレコード全体を一度に実行し、
各リードの完全な 0 ～ 10 秒の信号が 12 行に配置されます。プロジェクト アルゴリズムはレコードに対して 1 回だけ実行されます。
心拍の存在と数は、マルチリード検出によって判断されます。図の `ecgfeat` QRS はデフォルトでマークされています
個別に保存されたリードごとの位置決め `hybrid-prominence` を使用します。 NeuroKit2 その後、リードごとにリード
独立したテスト。

```bash
# 单条记录
python plot_ludb_12lead_10s_comparison.py 1

# LUDB 前 10 条记录
python plot_ludb_12lead_10s_comparison.py {1..10} \
  --out-dir ludb_12lead_10s_comparison
```

`--ecgfeat-r-source` は 4 つのタグから選択できます。

- `hybrid-prominence`: オリジナルのリードバイリード基準でのマルチリード心拍数に基づく
  35 ミリ秒以内で最大のプロミネンスを持つ正のローカル ピークを選択し、デフォルト値を描画します。
- `lead-fiducial`: 各リードのオリジナル QRS 基準。
- `positive-r`: 各リードの QRS 間隔内の最大正 R 波。
- `global`: すべてのリードで共有されるマルチリードのグローバル R 基準、検出評価スクリプトも
  履歴メトリックの一貫性を保つために、デフォルト値が引き続き使用されます。

`hybrid-prominence` は独立した `r_localized_index` と他のフィールドを上書きせずに書き込みます
`r_locs`、`qrs.peak`、`r_peak_index` または波形境界なので、心拍数は変わりません。
間隔、代表的なリードの特性、および解釈層の結果。

各レコードはデフォルトで生成されます。

1. スリーパーティ P/R/T ピーク オーバーレイ ページ;
2. `ecgfeat` の P/QRS/T 境界ページ。
3. NeuroKit2 の P/QRS/T 境界ページ。
4. LUDB エキスパート P/QRS/T 境界ページ。
5. 上記の 4 ページの PDF を組み立てます。

LUDB エキスパート ページまたは PDF が必要ない場合は、それぞれ `--no-ludb` および `--no-pdf` を使用できます。

<a id="8-当前实现的边界"></a>
## 8. 現在の実装の境界

- 比較は検出と波形定義に関するものであり、最終的な臨床診断ルールの精度ではありません。
- `ecgfeat` の R ピークはマルチリードのコンセンサス結果ですが、外部ライブラリはシングルリードの結果であり、ネイティブの使用法を反映していますが、まったく同じ入力タスクではありません。
- P/T マッチングでは波頭がアンカー ポイントとして使用され、境界インジケーターは一致した波のみをカウントします。
- デフォルトでは、NeuroKit2 は `neurokit` R ピーク メソッドと `dwt` 境界メソッドを使用します。これらはコマンド ラインから切り替えることができます。
- 異なる公差、リードサブセット、または NeuroKit2 メソッドによって得られた結果については、対応する `summary.json` を保持する必要があり、混合比較は許可されません。
