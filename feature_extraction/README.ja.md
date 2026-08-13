<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

# ecgfeat_dxl_inspired

**DXL スタイルだが、DXL の独自実装ではない** 12誘導 ECG 特徴抽出ライブラリのスケルトン。公開マニュアルのメインフローに従って構成されています：

1. 入力標準化と分析信号の構築
2. 波形品質評価
3. 四肢誘導逆接続のヒューリスティック検出
4. 多誘導 QRS 検出
5. ビートグループ化
6. representative beat の構築
7. ビートごと・誘導ごとの波境界の位置特定
8. 誘導/グループ/グローバルな特徴量の計算
9. グローバル QT / QTc / 軸の出力

<a id="重要说明"></a>
## 重要な注意事項

- これは **研究/エンジニアリング用の足場** であり、医療機器ソフトウェアではありません。
- コードは **DXL にインスパイアされた** 手法を実装したものであり、Philips の独自アルゴリズムの 1:1 再現ではありません。
- 特に複雑な室上性/室性リズム、重度のノイズ条件下での境界位置特定、およびペースメータービート用の専用ブランチについては、さらなる反復と検証が必要です。

<a id="安装"></a>
## インストール

```bash
pip install -e .
```

<a id="最小示例"></a>
## 最小例

```python
import numpy as np
from ecgfeat import ECGFeatureExtractor, PatientMeta, to_dict

# ecg shape: [12, n_samples], unit: mV
fs = 500
n = 5000
ecg = np.random.randn(12, n) * 0.02

extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
features = extractor.extract(ecg, fs=fs, meta=PatientMeta(age=45, sex="M"))

print(features.global_features)
print(to_dict(features)["metadata"])
```

<a id="当前输出"></a>
## 現在の出力

- `quality`: 各誘導の品質スコアとフラグ
- `structured payload`: `signal / quality / beats / groups / global / provenance` 6つの安定したセグメント
- `beats`: 各ビートのR波位置とグループ化
- `beat_features`: 各ビート/各誘導のP-QRS-T境界と基本波形パラメータ
- `representative_leads`: 各誘導の代表値パラメータと変動性
- `groups`: リズムグループのサマリー
- `global_features`: HR / PR / QRS / QT / QTc / P/QRS/T/ST軸 / QT分散
- `metadata`: QRS デテクタのデバッグ情報、記録品質、ペーシング状態、representative beat メタ情報、QT信頼性のある誘導リスト
- `ecgfeat.pediatric_rules`: 小児用付録AのRVH/LVH/LSH/BVH波形証拠のための電圧閾値ルックアップ; 利用できないDXL列は`None`として保持され、`tests.test_pediatric_rules`は年齢ビンと必須キーのカバレッジを検証する
- フェーズ2A測定ノート: P/Tの初期ピーク検索は、現在マルチ誘導候補と代表事前情報に基づいてビートレベルの融合を一度実行し、その後既存の単一誘導の接線/幾何学的境界位置決定に進む; 公開dataclass/エクスポートフィールド名と`signal / quality / beats / groups / global / provenance`構造は変更なし

<a id="rhythm-gap-closure-additions"></a>
## リズムギャップクローズ追加

- `atrial.py`: 独立した心房イベント抽出、ビート間ブロックPスキャン、およびAF/AFLルール入力用のQRSTウィンドウ心房残差スケルトン特徴
- `rhythm_rules.py`: ペーシングリズム制御フロー、前興奮ルール、期外収縮/停止/AVブロック証拠、測定利用可能性、およびステートメント抑制
- `metadata["rhythm_analysis"]`: `export.py`および`interpret.py`で使用される中間心房、AF/AFL、ペーシング、ルール、および利用可能性結果
- `export.py`: AF/AFLサマリー、ペーシングコンテキスト、ルール証拠、および`rhythm_inputs.record.availability`を含むリズム入力を公開

<a id="availability-semantics"></a>
## 利用可能性セマンティクス

- 生間隔/軸測定値は`global_features`に残る
- `rhythm_inputs.record.availability`は、心房リズム由来の値が下流のリズムロジックで信頼できるかどうかを示す
- 連続心室/デュアルペーシング、AF/AFL、完全AVブロック、およびAV解離は、PR/P軸に基づくリズム決定を抑制する
- ペーシングポリシーは、ペーシングがこれらの測定値を信頼できないものにする場合、信頼できないAVブロック、AV解離、PR、およびP軸の決定を抑制しながら、ペーシング証拠を保持する
- `rhythm_inputs.af_afl.qrst_subtraction.available`は、検証済みの各誘導QRSTテンプレート減算を意味する; 検証が満たされない場合、QRSTウィンドウ残差は`scaffold_available`、`method`、`qrst_subtraction_quality`、および`atrial_residual_signal_summary`を通じてスケルトンとして公開される

<a id="dxl-gap-closure-status"></a>
## DXL ギャップクローズステータス

- P0測定信頼性は、現在QRSコンセンサスガード、広域QRS QTリカバリ、ST-J出所、および長PR P開始リカバリを含む。
- P波波形エクスポートは、ノッチ、双相、初期、および終末成分の事実を出力する。
- 小児波形は年齢ビン電圧閾値を使用し、バイパス理由とともにRVH/LVH/BVH証拠をエクスポートする。
- リズム入力は、ステートメント証拠、ペーシング失敗事実、AF/AFL QRST subtraction検証状態、およびAVブロック証拠を公開する。
- 残りの制限事項は`docs/philips_*_feature_schema_checklist.md`に記載されている。

<a id="验证命令"></a>
## 検証コマンド

```bash
.venv_report_regen_20260625/bin/python -m unittest tests.test_multilead_pt_fusion tests.test_ecgfeat_pipeline -v
.venv_report_regen_20260625/bin/python compare_annotations.py --batch-reports --records 1 4 5 8 --out-dir tmp_phase2_multilead_pt --no-visuals
```

出力説明：

- `tmp_phase2_multilead_pt/summary.csv` は、このグループの LUDB スポットチェック記録におけるレポートの差異をまとめます
- レコードごとのレポートは、このサンプルグループで P / T 欠損カウントが顕著に増加しているかどうかを素早く確認するために使用できます

<a id="下一步建议"></a>
## 次のステップの提案

1. CSE / LUDB / QTDB に基づくシステム検証を行う
2. ペースドビート専用の区画化ブランチを追加する
3. より強力なマルチリード P/T 融合および U 波分離を導入する
4. Extended Measurements スタイルの完全なフィールドセットを拡張する
5. リズム / 形態ルールエンジンを重ね合わせる
