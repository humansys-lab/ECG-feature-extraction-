<!-- i18n-nav -->
[中文](project_organization_and_cleanup.md) | [English](project_organization_and_cleanup.en.md) | [日本語](project_organization_and_cleanup.ja.md)
<!-- /i18n-nav -->

# プロジェクトのコードとファイル整理レポート

整理日：2026-08-19

## 1. 整理結果

今回の整理では、ECG アルゴリズムと Agent の業務ロジックを変更せず、古いテスト用スクラッチ、キャッシュ、ログ、実行時マニフェスト、再生成可能な実験出力を削除しました。ワークスペースの使用量は約 137 GB から約 125 GB へ減り、約 12 GB を解放しました。

正式なテスト、元データセット、ローカルモデル、外部参考資料、現在の Python 仮想環境は保持しました。完全テストの結果は 1515 件成功、設計どおり 2 件スキップ、さらに 150 件のサブテスト成功です。

デモのエントリポイントが必要としていた欠落ローカルリンク `dataset -> data/010` を復元し、サンプルの `.hea` と `.mat` ファイルへアクセスできることを確認しました。

## 2. 削除した内容

| 分類 | 削除内容 | 判断理由 | 復元方法 |
|---|---|---|---|
| 古いテスト用スクラッチ | `tests/_multilead_pt_scratch.py` | pytest の収集規則に合わず、正式なカバレッジが `tests/test_multilead_pt_fusion.py` などに存在するため | Git 履歴から復元可能 |
| Python キャッシュ | プロジェクト内の `__pycache__/`、`.pytest_cache/`、`*.pyc`、`*.egg-info/` | Python、pytest、インストール処理が自動生成するため | 実行または再インストールで再生成 |
| 実行時中間ファイル | `.doc_translation_cache.json`、2 個の manifest、Web UI アップロード、ログ | `.gitignore` の対象であり、ソースコードではないため | 対応するコマンドで再生成 |
| 検証・描画成果物 | `ecgfeat_validation_20260811/`、`ludb_delineation_plots/`、`p_wave_failure_plots/` | 検証および描画スクリプトから再現可能なため | 対応するスクリプトを再実行 |
| 特徴・推論出力 | `ptbxl_09000_ecgfeat/`、`qwen_agent_output/`、`deepseek_v4pro_05490_v12/` | 入力データやソースではなく、派生特徴、診断、実験出力であるため | 保持したデータ、モデル、スクリプトから再構築 |

無視対象の生成物には Git 版がないため、削除後に Git から直接復元することはできません。外部バックアップがある場合は、そこからも復元できます。

## 3. 保持内容とディレクトリの責務

```text
ecg_gemma/
├── feature_extraction/ecgfeat/  # reusable ECG feature extraction library
├── ecgagent/                    # diagnostic agent and evidence pipeline
├── webchat/                     # web UI and model server
├── scripts/                     # maintenance, localization, and cohort utilities
├── tools/generation_conditions/ # focused audit and probe tools
├── tests/                       # unit, contract, and regression tests
├── docs/                        # design, algorithm, and validation documents
├── dataset -> data/010          # default demo dataset link (ignored)
├── data/                        # local raw/reference datasets (ignored)
├── medgemma-27b/                # local model (ignored)
├── qwen3.8-27b/                 # local model (ignored)
├── document/                    # external reference material (ignored)
├── *.py                         # compatible command-line entry points
└── requirements*.txt            # dependency definitions
```

| 領域 | 保持理由と保守境界 |
|---|---|
| `feature_extraction/ecgfeat/` | ECG の測定、品質管理、ルール、エクスポートの中核実装であり、再利用可能なアルゴリズムを配置する領域 |
| `ecgagent/` | LLM バックエンド、診断フロー、証拠、ツール、検証、レポートを担当し、低レベル ECG 測定は担当しない領域 |
| `tests/` | 1517 件の正式テストのソースであり、回帰保護なので一時ファイルとして削除しない領域 |
| `docs/` | アルゴリズム設計、検証結論、保守記録を保持し、Markdown 拡張子を統一した領域 |
| `data/`、モデルディレクトリ、`document/` | 大容量でも必要な入力または参考資料であり、ローカルに保持して `.gitignore` で分離する領域 |
| ルートの Python ファイル | 既存の CLI、デモ、評価、描画エントリポイントであり、直接 import が複数あるためコマンドとテストの互換性を保つ領域 |

ルートのエントリポイントは用途別に、UI とデモ（`app.py`、`demo_feature_extraction.py`、`visualize_ecg.py`）、抽出とランタイム（`batch_extract_ecgfeat.py`、`medgemma_ecg_core.py`、`medgemma_runtime.py`）、Agent/モデルのバッチ（`run_*.py`、`batch_medgemma_diagnostics.py`）、データセット評価と検証（`evaluate_*.py`、`validate_*.py`、`compare_*.py`）、結果分析と可視化（`analyze_*.py`、`ludb_*analysis.py`、`plot_*.py`、`render_*.py`）に分類されます。

## 4. 文書ファイル名の標準化

拡張子がなかった 6 件の正式文書を、`P波文档.md`、`QRS.md`、`Twave related.md`、`morphological.md`、`p wave doc.md`、`流程.md` という明示的な Markdown 名に変更しました。中国語、英語、日本語のナビゲーションと本文参照も同時に更新しました。

ローカライズスクリプトも修正しました。存在しない古いインデックスパスを無視し、新しい非無視文書を検査するため、Git でステージする前に改名や追加を検証できます。

## 5. 今後の保守ルール

- プロジェクト内の Python/pytest キャッシュ、ログ、アップロード、明確な派生出力は随時削除できますが、リポジトリ全体を対象にした無制限のクリーンアップは実行しません。
- `data/`、`medgemma-27b/`、`qwen3.8-27b/`、`document/`、`.venv/` は中間生成物として扱わず、削除前に個別確認が必要です。
- 正式テストは `tests/test_*.py` の形式を使用し、一時的なプローブは `tools/` に置き、収集されないアンダースコア付きファイルを `tests/` に残しません。
- 新しい再利用可能ロジックは `ecgfeat` または `ecgagent` に置き、ルートの入口には引数解析、タスク編成、出力保存だけを残します。
- 新しい大規模結果ディレクトリは `.gitignore` に追加し、実行コマンドでは明確な出力ディレクトリ名を使用します。

## 6. 検証結果

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q feature_extraction/ecgfeat ecgagent *.py scripts webchat
.venv/bin/python scripts/check_doc_localizations.py
```

- pytest：1515 passed、2 skipped、150 subtests passed。
- Python コンパイル検査：成功。
- 文書ローカライズの構造、コードブロック、ナビゲーション検査：成功。

スキップされた 2 件は `tests/test_js00059_clinical_regression.py` にあり、任意の回帰成果物を `python demo_feature_extraction.py JS00059` で先に生成する必要があります。これはテスト設計どおりです。

## 7. 実施しなかった高リスクの再構成

今回はルートにある 40 個以上の CLI スクリプトを多層ディレクトリへ強制移動していません。スクリプトとテストには直接モジュール import があるため、移動すると公開コマンドと Python import パスが変わります。将来ルートを縮小する場合は、互換ラッパー、文書更新、完全回帰テストを含む別のエントリポイント移行として実施します。
