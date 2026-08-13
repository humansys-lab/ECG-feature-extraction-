<!-- i18n-nav -->
[中文](codebase_guide.md) | [English](codebase_guide.en.md) | [日本語](codebase_guide.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="代码库导航与维护约定"></a>
# コードベースのナビゲーションと保守に関する規約

本リポジトリには、アルゴリズムライブラリ、コマンドラインエントリポイント、モデル推論、データセット、および実験成果物が含まれています。既存のコマンドとの互換性を維持するため、エントリポイントスクリプトは当面ルートディレクトリに保持されます。新たに追加される再利用可能なロジックは、複数のスクリプトにコピーするのではなく、優先的に共有モジュールに配置してください。

<a id="目录与职责"></a>
## ディレクトリと役割

| 場所 | 役割 |
|---|---|
| `feature_extraction/ecgfeat/` | ECG 測定、品質管理、解釈ルール、構造化エクスポートの中核ライブラリ |
| `feature_extraction/ecgfeat/clinical_rules/` | 統一された臨床ルール。各ファイルは診断分野ごとに分割されています |
| `feature_extraction/ecgfeat/glasgow_rules/` | グラスゴースタイルの測定と初期ルール |
| `ecgagent/` | `ecgfeat` 成果物用の LLM エージェント：診断証拠のホワイトリスト、ポインタアドレス指定、信頼性に関する注意書き、ツール登録、監査、および矛盾する陽性結論のみをブロックするバージョン管理された最小限の安全戦略。測定や診断は生成しません |
| `tests/` | ユニットテスト、契約テスト、データ付きリグレッションテスト |
| `docs/` | アルゴリズムの説明、検証記録、設計ドキュメント |
| `data/`、`all_diseases_*` | データセット、バッチ処理入力、実験成果物。再利用可能なコードを保持すべきではありません |
| `document/` | 外部参照資料 |
| `scripts/` | アルゴリズムとは無関係な運用・ダウンロードスクリプト |

<a id="根目录-python-入口"></a>
## ルートディレクトリの Python エントリポイント

ルートディレクトリのファイルは用途に応じて以下のグループに分類できます：

- デモとインターフェース：`demo_feature_extraction.py`、`visualize_ecg.py`、`app.py`
- 特徴量バッチ処理：`batch_extract_ecgfeat.py`
- MedGemma 推論：`batch_medgemma_diagnostics.py`、`run_layered_single.py`、
  `run_layered_batch_flat.py`
- エージェント：`python -m ecgagent.cli --features X.json --agent`（有界な5段階の診断サイクル：
  Survey の後、制御された知識に基づいて仮の診断を形成し、その後対象的な ecgfeat 検査を行います。続いて
  決定論的な検証フェーズに入り、デフォルトでは `claude-opus-5` を使用します。`ANTHROPIC_API_KEY` が必要です）。同じエントリポイントでは
  `--briefing`、`--tool`、`--schema`、`--verify`、`--audit` も提供されます。
  設計は `docs/ecg_agent_architecture.md`、使用方法は `ecgagent/README.md` を参照してください。
- LUDB 比較：`evaluate_ludb.py`、`compare_ludb_detectors.py`、
  `compare_annotations.py`、`plot_ludb_*.py`
- ST と目標診断の検証：`validate_*.py`、`analyze_physionet_st.py`、
  `evaluate_target_ecgfeat_diagnosis.py`
- 結果の分析とレンダリング：`analyze_ludb_*.py`、`ludb_*analysis.py`、
  `render_ecgfeat_record_explanations.py`

ルートディレクトリのエントリポイントはパラメータと出力ファイルを組み合わせることができますが、共有アルゴリズム、入力解析、またはランタイム初期化はエントリポイント内に引き続き配置しないでください。現在、共有の境界は以下の通りです：

- `ecgfeat.io`：PhysioNet/WFDB スタイルの `.hea` メタデータと `.mat` 信号の読み取り
- `medgemma_runtime.py`：CUDA、量子化、テンソル並列処理、および vLLM テキスト生成の初期化
- `medgemma_ecg_core.py`：MedGemma コンテキスト、プロンプト、解析、および診断フロー

<a id="修改建议"></a>
## 変更提案

1. 中核的な測定またはルールの変更は `ecgfeat` に配置し、同時に `tests/` に最小限の契約テストを追加してください。
2. 新しいデータセットは、アダプター内でリード順序、ゲイン、メタデータのマッピングのみを行い、データセットの判定ロジックを中核アルゴリズムに書き込まないでください。
3. 新しい CLI は共有モジュールを再利用してください。パラメータ解析、タスク列挙、ファイルの保存のみをエントリポイントスクリプトに保持してください。
4. 大容量データ、画像、ログ、デモ出力はソースコードとしてコミットしないでください。対応するパターンは
   `.gitignore` に記述されています。
5. コミット前に実行：

   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/python -m compileall -q feature_extraction/ecgfeat *.py
   ```

`tests/test_js00059_clinical_regression.py` は製品回帰テストです。ルートディレクトリのみが存在する
`JS00059_features.json` および `JS00059_report.txt` の場合にのみ実行されます。渡せる
`python demo_feature_extraction.py JS00059` は、これら 2 つのファイルと関連イメージを生成します。
