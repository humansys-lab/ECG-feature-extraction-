<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="ecg-gemma12-导联-ecg-特征提取与临床证据工程"></a>
# ECG Gemma: 12 リード ECG 特徴抽出と臨床証拠エンジニアリング

このプロジェクトは、解釈可能な 12 リード ECG 特徴抽出パイプラインを提供し、ルール エンジン、レポーター、および MedGemma で使用できる構造化された証拠に測定値を編成します。

単なる R ピーク検出器ではありません。現在のプロセスには、信号の前処理、品質管理、ペーシング スパイク検出、マルチリード QRS 検出、拍動のグループ化、代表的な拍動の構築、P/QRS/T 波境界点の位置決め、間隔および心軸の測定、心房調律分析、および伝導、肥大、ST-T などの臨床ルールが含まれます。

> **重要な注意事項**
>
> このプロジェクトは研究およびエンジニアリングを目的としたソフトウェアであり、医療機器ではありません。医師の解釈に代わるものではなく、臨床診断や治療の意思決定に直接使用すべきではありません。

<a id="1-项目结构"></a>
## 1. プロジェクトの構造

```text
ecg_gemma/
├── feature_extraction/
│   ├── ecgfeat/                  # ECG 特征提取核心包
│   ├── README.md                 # 核心包的补充说明
│   └── pyproject.toml
├── demo_feature_extraction.py    # 单条 WFDB .mat/.hea 记录演示
├── batch_extract_ecgfeat.py      # ECG 特征批处理
├── medgemma_ecg_core.py          # MedGemma 上下文与诊断流程
├── medgemma_runtime.py           # 共享 GPU/vLLM 运行时初始化
├── app.py                        # Gradio 界面入口
├── evaluate_*.py / validate_*.py # 数据集评估与验证入口
├── dataset -> data/010           # demo 默认读取的数据目录
├── tests/                        # 算法、导出与临床规则测试
├── data/                         # 数据、特征、报告与图像
├── requirements.txt              # 完整项目依赖
└── medgemma-27b/                 # MedGemma 相关代码
```

エントリの分類、共有モジュールの境界、および保守規約の詳細については、「」を参照してください。
[コードベースのナビゲーションとメンテナンスの規則](docs/codebase_guide.ja.md)。

ECG Diagnostic Agent の現在のデュアルチャネル アーキテクチャ、証拠契約、コンパクト/レガシー ワークフロー、
ツールと検証の設計については、[ECGAgent 完全な設計ドキュメント](docs/ecg_agent_complete_design.ja.md) を参照してください。

コアモジュール:

|モジュール |機能 |
|---|---|
| `ecgfeat/api.py` |完全なパイプライン入口 `ECGFeatureExtractor.extract()` |
| `ecgfeat/preprocess.py` |リサンプリング、ベースライン解除、​​ノッチおよび検出信号の構築 |
| `ecgfeat/quality.py` |リードごとおよび測定機能ごとの高品質なゲート |
| `ecgfeat/pacing.py` |ペーシング スパイクの検出と検証 |
| `ecgfeat/qrs.py` |多リード QRS/R ピーク検出 |
| `ecgfeat/grouping.py` |ハートビート形状のグループ化 |
| `ecgfeat/representative.py` |ハートビートの調整と融合を表します |
| `ecgfeat/delineate.py` | P/QRS/T 波のピークと境界の位置 |
| `ecgfeat/st_localization.py` |堅牢な PR ベースライン、J ポイント コンセンサス、および ST マルチポイント バイパス測定 |
| `ecgfeat/features.py` |単一リード、グループ化、およびグローバルな特徴の集約 |
| `ecgfeat/atrial.py` |性的事件、QRST subtraction、AF/AFL の証拠 |
| `ecgfeat/clinical_rules/` |リズム、伝導、間隔、肥大、虚血およびその他の規則 |
| `ecgfeat/export.py` | JSON 変換と構造化エクスポート |
| `medgemma_ecg_core.py` | MedGemma のコンテキスト、プロンプト、および階層化された診断プロセス |

<a id="2-运行环境"></a>
## 2. 動作環境

Python 3.10 以降を推奨します。

<a id="仅运行-ecg-feature-extraction"></a>
### ECG 特徴抽出のみを実行します

```bash
cd /workspace/ecg_gemma

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ./feature_extraction
python -m pip install matplotlib pytest
```

`feature_extraction` パッケージは、コアの依存関係 NumPy および SciPy をインストールします。描画にはMatplotlibが必要です。

<a id="完整项目依赖"></a>
### プロジェクトの依存関係を完了する

```bash
python -m pip install -r requirements.txt
```

完全な依存関係には、`vllm`、`transformers`、Gradio などのより大きなモデルとインターフェイス コンポーネントが含まれます。 ECG 機能抽出のみを実行する場合は、すべての依存関係をインストールする必要はありません。

`.pt` ファイルを処理する場合は、マシンの CPU/CUDA 環境に応じて、別途 PyTorch をインストールする必要があります。

<a id="ludb-上比较-ecgfeatneurokit2-和-biosppy"></a>
### ecgfeat、NeuroKit2、および BioSPPy を LUDB で比較します

比較ツールは、LUDB のリードごとのエキスパート アノテーションを読み取り、R ピーク、P/QRS/T 波検出率、一致するイベントの境界誤差をカウントします。 BioSPPy のパブリック ECG プロセスは R ピークのみを提供するため、QRS/R ピークの比較のみが含まれます。

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pip install -r requirements-ludb-compare.txt

# 快速验证：记录 1、II 导联
python compare_ludb_detectors.py --records 1 --leads II \
  --out-dir ludb_three_way_smoke

# 完整 LUDB、全部 12 导联
python compare_ludb_detectors.py --workers 4 \
  --out-dir ludb_three_way_comparison

# 前10条记录，使用每个导联自己的 ecgfeat QRS fiducial
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --ecgfeat-r-source lead-fiducial \
  --out-dir ludb_three_way_first10_per_lead_r

# 前10条记录，使用混合 prominence 定位
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --methods ecgfeat \
  --ecgfeat-r-source hybrid-prominence \
  --out-dir ludb_ecgfeat_first10_hybrid_prominence

# 前10条记录：混合 R 定位 + 双极性 T 峰细化
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --methods ecgfeat \
  --ecgfeat-r-source hybrid-prominence \
  --ecgfeat-wave-source hybrid-peaks \
  --out-dir ludb_ecgfeat_first10_hybrid_waves
```

完全なメソッド定義、パラメーター、インジケーター、出力ファイルの説明については、[LUDB 3 つの検出メソッドの比較](docs/ludb_detector_comparison.ja.md) を参照してください。

NeuroKit2のローカル測位の利点を吸収し、ecgfeatのR/P/T/S波検出を改善し、
堅牢な ST/J 測定ソリューション、
[ecgfeat × NeuroKit2 ハイブリッド改良設計](docs/ecgfeat_neurokit_hybrid_improvement.ja.md)をご覧ください。

T 波強調はデフォルトでオンです。マラー 2 次スプラインは、専用の 22.5 Hz ブランチを使用して並列計算されます。
T-SQI、RMS/PC1 を組み合わせたウェーブレットおよび低振幅 T 台形領域の候補
加重中央値/MAD/Huber マルチリード フュージョン、堅牢な中心と
P85は最も最近再二極化している。 QT コンセンサスとローカル T オフセットは、融合品質管理に合格した場合にのみ更新されます。
厳密なトランスリードレイトテールレスキューの下でのみ修正されます。履歴回帰が必要な場合は、次のように渡すことができます。
`enable_t_wave_refinement=False`。

ロバスト ST 拡張出力はデフォルトで有効になっており、PR セグメント ベースライン、P85 マルチリード コンセンサス J ポイント、
J/J+20/J+40/J+60/J+80 および RR/Ton 適応ポイントのウィンドウ中央値、およびロバスト性
傾斜、二次曲率、トレンド、バンプ パターン。これはネイティブの ST 分野をカバーするものではなく、臨床解釈を変更するものでもありません。厳密な履歴回帰を行う必要がある場合は、次のようにすることができます。
エクストラクターを構築するときに、`enable_hybrid_st_measurement=False` を渡します。

ローカル `dataset` の ST タグ付きレコードを確認します。

```bash
# 扫描全部标签并对100条记录做标签级比较
PYTHONPATH=feature_extraction .venv/bin/python validate_dataset_st.py \
  --dataset-dir dataset \
  --out-dir dataset_st_validation \
  --workers 4

# 仅提取明确 ST 标签和 ST-T/心梗探索记录
PYTHONPATH=feature_extraction .venv/bin/python validate_dataset_st.py \
  --dataset-dir dataset \
  --out-dir dataset_st_candidates \
  --candidates-only
```

このスクリプトは、候補リスト、レコードごと/リードごと/ビートごとの CSV、ラベルの一貫性メトリック、およびクリアを出力します。
STタグ記録の12リードラベル図。このデータにはレコードレベルの診断タグのみがあるため、
リードバイリードの J ポイントまたは ST 振幅の真の値の出力は、臨床精度と見なすことはできません。

プロジェクト アルゴリズムと NeuroKit2 のピーク値を同じ ECG 上に重ね合わせ、2 つの境界を別々の行に表示します。

```bash
# 完整记录
python plot_ecgfeat_neurokit_comparison.py 1 --lead II

# 显示第 2–5 个 LUDB QRS 对应的连续心搏
python plot_ecgfeat_neurokit_comparison.py 8 --lead V2 --beat-range 2 5
```

完全に文書化された 12 リード、10 秒間の比較を描画します。

```bash
# 记录 1；生成 4 张 PNG 和一个四页 PDF
python plot_ludb_12lead_10s_comparison.py 1

# 批量绘制记录 1–10
python plot_ludb_12lead_10s_comparison.py {1..10}

# 如需复现旧图中的全局多导联 R 标记
python plot_ludb_12lead_10s_comparison.py 1 \
  --ecgfeat-r-source global
```

プロットでは、デフォルトで独立したリードごとのプロミネンス配置が使用されます (`hybrid-prominence`)。
ビート数は、依然として ecgfeat のマルチリード検出によって決定されます。このバイパス フィールドはグローバル フィールドに置き換わるものではありません。
`r_locs`、従来の `qrs.peak` または QRS 境界であるため、最終的な測定と解釈には影響しません。

<a id="3-输入约定"></a>
## 3. 規約に入る

特徴抽出機能は次のことを期待します。

- 標準 12 リード ECG;
- 配列形状は `[12, n_samples]`、`[n_samples, 12]` も受け入れられます。
- 推奨されるリードシーケンスは `I, II, III, aVR, aVL, aVF, V1–V6` です。
- 入力信号の振幅単位は mV です。
- 元のサンプルレートを明示的に提供します。
- 患者の年齢と性別はオプションのメタデータですが、一部の臨床規則ではそれらが必要です。

デモ スクリプトはデフォルトで次のようになります。

```text
dataset/
├── JS00001.hea
├── JS00001.mat
├── JS00002.hea
└── JS00002.mat
```

`.mat` には、形状 `[12, n_samples]` の `val` 配列が含まれている必要があります。現在のデモは、デフォルトで `1000 ADC units/mV` に従って変換されます。新しいデータ セットにアクセスする前に、各 `.hea` の実際のゲイン、単位、およびリード シーケンスを確認する必要があります。

<a id="4-快速运行"></a>
## 4. 素早く実行する

<a id="随机抽取一条记录"></a>
### レコードをランダムに選択します

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python demo_feature_extraction.py
```

<a id="抽取指定记录"></a>
### 指定したレコードを抽出する

```bash
python demo_feature_extraction.py JS00010
```

デフォルトでは、プロジェクトのルート ディレクトリに生成されます。

```text
JS00010_features.json
JS00010_report.txt
JS00010_ecg.png
JS00010_ecg_annotated.png
JS00010_ecg_report.png
```

<a id="保留逐拍逐导联完整特征"></a>
### ビートごとおよびリードごとの完全な特性を維持する

```bash
python demo_feature_extraction.py JS00010 --full-beats
```

デフォルトでは、JSON は最大の `beat_features` を省略します。 `--full-beats` は、波の境界位置のデバッグ、シングルショット結果の監査、またはトレーニングでビートごとのデータが必要な場合にのみ推奨されます。

<a id="5-抽取-dataset-的前-10-条"></a>
## 5. データセットから最初の 10 項目を抽出します

次のコマンドは、レコード番号で自然に並べ替え、上位 10 レコードを抽出し、生成されたファイルを別のディレクトリに移動します。

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate

output_dir="data/dataset_first10_ecgfeat_out"
mkdir -p "$output_dir"

find -L dataset -maxdepth 1 -type f -name '*.hea' -printf '%f\n' \
  | sed 's/\.hea$//' \
  | sort -V \
  | head -n 10 \
  | while read -r record_id; do
      python demo_feature_extraction.py "$record_id"

      for suffix in \
        features.json \
        report.txt \
        ecg.png \
        ecg_annotated.png \
        ecg_report.png; do
        artifact="${record_id}_${suffix}"
        if [ -f "$artifact" ]; then
          mv "$artifact" "$output_dir/"
        fi
      done
    done
```

現在生成されている上位 10 件の結果は次のとおりです。

```text
data/dataset_first10_ecgfeat_out/
```

<a id="6-pt-批量抽取"></a>
## 6. `.pt` バッチ抽出

`batch_extract_ecgfeat.py` は WFDB `.mat/.hea` を処理しませんが、代わりに次のディレクトリを処理します。

```text
all_diseases_pt100/
├── norm/
│   ├── generated.pt
│   ├── real.pt
│   └── metadata.pt
├── ami/
│   ├── generated.pt
│   └── real.pt
└── ...
```

各テンソルは次のようになります。

- `[n_samples, 12, n_points]`
- `[n_samples, n_points, 12]`
- シングル `[12, n_points]`
- シングル `[n_points, 12]`

まず、少数のサンプルを使用して煙テストを実行します。

```bash
python batch_extract_ecgfeat.py \
  --input-dir all_diseases_pt100 \
  --output-dir all_diseases_ecgfeat \
  --sampling-rate 100 \
  --sources generated,real \
  --limit 2 \
  --annotated-plots
```

正式なバッチ処理:

```bash
python batch_extract_ecgfeat.py \
  --input-dir all_diseases_pt100 \
  --output-dir all_diseases_ecgfeat \
  --sampling-rate 100 \
  --sources generated,real \
  --skip-existing
```

JSON でビートごとの詳細を保持するには、次を追加します。

```bash
--include-beat-features
```

バッチ処理された `.pt` フィーチャー ファイルには、常に完全なビートごとのデータが保持されます。このパラメータは、JSON が `beat_features` を保持するかどうかのみを制御します。

## 7. Python API

```python
import numpy as np

from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.export import prepare_json_export, to_dict
from ecgfeat.models import PatientMeta

# 形状：[12, n_samples]；单位：mV
ecg = np.random.randn(12, 5000) * 0.02
fs = 500

extractor = ECGFeatureExtractor(
    fs_internal=500,
    mains_freq=50,
)

features = extractor.extract(
    ecg,
    fs=fs,
    meta=PatientMeta(age=45, sex="Male"),
)

print(features.global_features)

payload = prepare_json_export(
    to_dict(features),
    include_beat_features=False,
)
```

データが ADC 整数値に由来する場合、各レコードは最初に独自のゲインを使用して mV に変換される必要があります。

```python
ecg_mv = ecg_adc.astype(np.float64) / gain_units_per_mv
```

<a id="8-算法流程"></a>
## 8. アルゴリズム処理

```text
原始 12 导联
    ↓
输入方向检查与重采样
    ↓
去基线和工频陷波
    ├── analysis signal：保留 ST/T 幅度
    └── detection signal：额外低通，增强 QRS
            ↓
质量评估、导联反接和起搏检测
            ↓
多导联 QRS 检测
            ↓
心拍形态聚类和代表心拍
            ↓
P/QRS/T 波定位及多导联共识修复
            ↓
PR、QRS、QT、JT、QTc、ST、振幅、面积和心电轴
            ↓
房性事件、AF/AFL 和临床规则
            ↓
JSON、文本报告、ECG 图像和 MedGemma 证据
```

<a id="81-双信号预处理"></a>
### 8.1 デュアル信号前処理

- `analysis_signal` は、2 段階のメディアン フィルタリングを使用してベースライン ドリフトを除去し、電力周波数のノッチを実行します。
- `detection_signal` は、QRS 検出のために追加のローパスを実行します。
- ST セグメント、T 波、および振幅の測定では、可能な限り忠実度の高い分析信号を使用します。

<a id="82-质量控制"></a>
### 8.2 品質管理

ベースラインドリフト、EMGノイズ、電源周波数干渉、クリッピング、低振幅などの指標は、リードごとに個別に計算されます。 P 波、QRS、T 波、および QT は、異なる信頼性しきい値を使用します。

合格した品質グレードは、測定が完全に正しいことを意味するものではありません。呼び出し元は、値、信頼度、ソース、および可用性を同時に読み取る必要があります。

<a id="83-qrs-与代表心拍"></a>
### 8.3 QRS 代表者とのハートビート

QRS 検出は、複数の四肢リードと前胸部リードを融合し、バンドパス、差分、二乗、移動積分、および適応しきい値処理を使用して候補 R ピークを見つけます。検出された心拍は形態によってクラスター化され、相互相関アラインメントと中央値/中央値法によって代表的な心拍が構築されます。

<a id="84-pqrst-delineation"></a>
### 8.4 P/QRS/T の説明

システムは各心拍と各リードを推定します。

- P オンセット、ピーク、オフセット。
- QRS オンセット、Q/R/S ピーク、QRS オフセット。
- T ピークと T エンド;
- ST-J、ST中間点、ST終了点。

位置特定プロセスには、代表的なビート事前分布、局所的な傾き、接線/幾何学的手法、生理的範囲、およびマルチリードのサポートが組み込まれています。広い QRS、ペーシング、低振幅、および長い PR 用の追加の回復パスがあります。

<a id="85-全局特征"></a>
### 8.5 グローバル機能

主な出力には次のものが含まれます。

- HR と RR;
-PR;
- QRS 期間;
- QT と JT;
- QTc ルール層のバゼット、フリデリシア、ホッジス、フレーミングハム。
- P/QRS/T/ST 軸;
- QT分散;
- P、Q、R、S、T の振幅と面積。
- ST シフトと形態学的証拠。

ワイド QRS またはペーシング ケースでは、通常の QT の重みが軽減され、JT およびワイド QRS 専用パスがより有効に活用されます。

<a id="86-房性节律"></a>
### 8.6 心房リズム

AF/AFL 分析では、RR の不規則性だけでなく、以下も使用されます。

- P ウェーブ組織。
- 部屋の関連付け。
- QRST テンプレートの減算。
- 残留信号の再現性。
- マルチリード F 波周波数の一貫性。

QRST subtraction が品質検証に失敗した場合、システムは足場の証拠を保持しますが、それを信頼できる結果として扱いません。

<a id="9-输出说明"></a>
## 9. 出力の説明

デモとバッチは現在デフォルトで使用されています:

```python
prepare_json_export(to_dict(features))
```

デフォルトの JSON の主要なトップレベル フィールドは次のとおりです。

|フィールド |コンテンツ |
|---|---|
| `quality` |リードごとの品質とレコードレベルのゲーティング |
| `beats` | R ピーク、RR、ビートのグループ化 |
| `groups` |形態学的グループおよびグループレベルの測定 |
| `representative_leads` |各リードの代表的な特徴 |
| `global_features` | HR、PR、QRS、QT、QTc および心軸 |
| `rhythm_inputs` |ペーシング、P イベント、AF/AFL、および可用性 |
| `morphology_inputs` | P/QRS/ST/T フォームと派生事実 |
| `interpretation` |初期の DXL を参考にしたリファレンスの説明 |
| `clinical_interpretation` |現在の信頼できる臨床規則の結果 |
| `statement_engine` |診断ステートメント、抑制関係および証拠 |
| `metadata` |検出器、コンセンサス、修正および来歴情報 |

<a id="解释层的使用原则"></a>
### 説明レイヤーの使用原則

このプロジェクトは現在、信頼できる臨床解釈と初期の参考解釈を保持しています。ダウンストリーム プログラムでは、以下を優先的に使用する必要があります。

```text
clinical_interpretation
```

`interpretation` は、DXL からインスピレーションを得た初期の参照証拠であり、信頼できる結果と無条件に統合することはできません。

<a id="structured-v3"></a>
### 構造化 v3

`ecgfeat.export.build_structured_payload(features)` は、`signal / quality / beats / groups / global / provenance` などの安定したパーティションを含む構造化ペイロードを生成します。

現在のスキーマのバージョンは `ecgfeat_structured_payload.v3` です。 v3 では、古いグラスゴー解釈ブロックが削除されました。

これは、デモのデフォルト タイル JSON と同じスキーマではありません。バリデーターが `signal`、`global`、または `provenance` が欠落していると報告した場合は、まず呼び出し元がどのエクスポート コントラクトを期待しているかを確認する必要があります。

<a id="10-验证"></a>
## 10. 検証

テストを実行します。

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pytest -q tests
```

2026 年 7 月 26 日の時点で、QRS、波境界位置決め、QT、品質、心房調律、ペーシング、エクスポート、臨床ルールなどのコアテストがローカルに選択され、実行されています。結果は次のとおりです。

```text
385 passed
```

既存の 200 個の履歴比較スナップショットのおおよその平均絶対誤差 LUDB:

|指標 |前 |
|---|---:|
|人事 | 0.66bpm |
|広報 | 9.26ミリ秒 |
| QRS | 9.88ミリ秒 |
| QT | 9.99ミリ秒 |
|バゼット | QTc 11.36ミリ秒 |
| QTc 10.78ミリ秒 |
| P軸 | 14.68° |
| QRS軸 | 10.09° |
| T軸 | 10.85° |
| QT分散 | 28.37ミリ秒 |

このスナップショットは、最新のコード変更の一部よりも古いものであり、歴史的なエンジニアリングの参考としてのみ使用できます。現在のバージョンの再検証に代わるものではなく、臨床診断性能を証明するものでもありません。

<a id="11-已知限制"></a>
## 11. 既知の制限事項

1. 現在のシステムは研究グレードの実装であり、規制当局によって検証された医療機器ではありません。
2. AF/AFL、複雑なペーシング、低振幅の P 波、および激しいノイズには、依然として大規模な外部検証が必要です。
3. `acute_occlusion_pattern` は、隣接するリードでの ST 上昇スクリーニングに近いため、急性冠状動脈閉塞を独立して診断することはできません。
4. QT 分散の履歴誤差は、純粋な QT の履歴誤差よりも大幅に大きくなります。
5. 従来の解釈や権威ある臨床規則は、異なる結論につながる可能性があります。
6. `<16 岁` と `<18 岁` の両方に適用される境界は、異なる小児科規則に存在し、16 ～ 18 歳の結果には特別なレビューが必要です。
7. デモには、ゲインと標準的なリード次数に関するデフォルトの仮定があります。
8. 波の境界位置には複数のレスキュー/フォールバック層が含まれており、上流のエラーが後方に伝播する可能性があります。
9. 一部のコードは依然として非推奨の `numpy.trapz` を使用しており、テスト中に警告が生成されます。

<a id="12-常见问题"></a>
## 12. よくある質問

### `ModuleNotFoundError`

まず、正しい仮想環境を使用していることを確認してください。

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pip install -e ./feature_extraction
python -m pip install matplotlib
```

特定のタスクに従って不足しているコンポーネントをインストールします。

```bash
python -m pip install wfdb pandas gradio
```

<a id="处理-pt-时提示缺少-torch"></a>
### `.pt` の処理時にトーチが見つからないことを示すプロンプトを表示

PyTorch は、コア機能抽出の依存関係には含まれていません。マシンのCPUまたはCUDAの環境に応じて、対応するバージョンをインストールしてください。

<a id="没有-gpu"></a>
### GPU なし

ECG 特徴抽出自体は NumPy/SciPy CPU アルゴリズムであり、GPU は必要ありません。 GPU は主に、後続の MedGemma 推論に使用されます。

<a id="图片生成失败"></a>
### 画像生成に失敗しました

```bash
python -m pip install matplotlib
export MPLBACKEND=Agg
```

<a id="json-文件太大"></a>
### JSON ファイルが大きすぎます

`--full-beats` または `--include-beat-features` は使用しないでください。デフォルトの概要出力は、リード、全体的な測定値、および臨床的解釈を表すもののままです。

<a id="schema-校验失败"></a>
### スキーマ検証に失敗しました

確認バリデータは次のことを期待します。

- デモ/バッチのデフォルト `to_dict()` タイル構造。まだ
- `build_structured_payload()` によって生成された構造化 v2。

2 つの出力で同じトップレベル フィールド要件のセットを使用することはできません。

<a id="13-建议的后续工作"></a>
## 13. 推奨されるフォローアップ作業

1. デフォルトの JSON スキーマを統一し、バージョンと権限のある説明フィールドを明確にします。
2. リード名、単位、ゲイン、NaN、サンプル レート、およびレコード長について厳密な入力契約を確立します。
3. 患者レベルの独立したテストセットを使用して、間隔エラー、リズム分類、および臨床ルールを検証します。
4. AF/AFL、P 軸、ペーシング、および ST 上昇の混合シナリオを強化します。
5. 小児の年齢ルーティングを統一します。
6. 特大の `api.py`、`delineate.py`、および `features.py` を明確でテスト可能なステージに分割します。
7. アルゴリズムのしきい値を一元管理し、しきい値の構成にバージョン番号を追加します。
8. デモ、バッチ、レガシー JSON、構造化 v2 のそれぞれにコントラクト テストを追加します。

<a id="license"></a>
## ライセンス

このプロジェクト コードまたは出力を外部配布、研究出版、または商用目的で使用する前に、プロジェクト ライセンス、データセット ライセンス、およびモデル ライセンスを補足および確認してください。
