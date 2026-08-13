<!-- i18n-nav -->
[中文](2026-07-12-clinical-diagnosis-p2.md) | [English](2026-07-12-clinical-diagnosis-p2.en.md) | [日本語](2026-07-12-clinical-diagnosis-p2.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="clinical-diagnosis-p2-implementation-plan"></a>
# 臨床診断P2実施計画

> **エージェント ワーカーの場合:** 必須サブスキル: superpowers:subagent-driven-development (推奨) または superpowers:executing-plan を使用して、この計画をタスクごとに実装します。ステップでは、追跡にチェックボックス (`- [ ]`) 構文を使用します。

**目標:** P0/P1 後に使用できないままになっている臨床ルール パス、または疾患をオーバーコールする臨床ルール パスを修正します: Q 波領域の範囲、後部虚血、ショート/ワイド QRS QT、AF 証拠の正規化、および P 依存の適用可能性。

**アーキテクチャ:** 各ルールは決定的な証拠のショートサーキットを使用し、しきい値を弱めるのではなく証拠層をエクスポートします。高リスクの ECG パターンは、明示的な確認要件を伴うスクリーニング所見のままです。既存の v2 ステータス/カバレッジ/正常性ロール モデルは、別のスキーマ変更なしで使用されます。すべての P2 動作が完了すると、ルールセットが 1 回進みます。

**技術スタック:** Python 3.12、NumPy、pytest、既存の `feature_extraction.ecgfeat` 測定および臨床ルール レイヤー。

<a id="global-constraints"></a>
## グローバル制約

- 欠損データをゼロまたは通常の証拠として扱わないでください。
- 実行時にデータセット診断ラベルを使用しないでください。
- 急性虚血および症候群レベルの QT ステートメントは、明示的に勧告/手動レビューとして保持してください。
- エクスポートされた証拠に生のフォーミュラ値とリードごとの事実を保存します。
- 各タスクの RED/GREEN/REFACTOR に従います。
- 現在のワークスペースには Git リポジトリがありません。コミットの代わりに検証チェックポイントを記録します。

---

<a id="task-1-territory-aware-pathological-q-wave-logic"></a>
### タスク 1: 領域を意識した病理学的 Q 波ロジック

**ファイル:**
- 変更: `tests/test_clinical_ischemia.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/ischemia.py:99-143`

**インターフェース:**
- リードごとの Q 振幅、Q 持続時間、および R 振幅を消費します。
- 生成: テリトリーごとのステータス/カバレッジ、および少なくとも 2 つの条件を満たす連続/グループ化されたリードからのみのグローバル マッチを生成します。

- [ ] 継続時間が欠落している小さな Q によってルールが使用不可にならないこと、継続時間が欠落している潜在的に病的な Q はその領域内でのみ不定のままであり、完全な正常領域は一致しないことを返す可能性があることを証明する失敗するテストを追加します。
- [ ] `PYTHONPATH=/workspace/ecg_gemma pytest -q tests/test_clinical_ischemia.py -k q_wave` を実行し、RED を確認します。
- [ ] 持続時間が決定的になる前に、`abs(q_amp) < 0.10` と病理学的 Q/R 比が存在しない Q 波をプレフィルターします。
- [ ] 評価済み、候補者不足、適格なリードで `territory_results` を追跡します。いずれかのテリトリーに 2 つの資格のあるリードがある場合に一致したものを返します。未解決のリード候補がまだ適格なペアを形成できる場合にのみ、リターンは利用できません。それ以外の場合は、部分的な報道証拠と一致しないものを返します。
- [ ] 重点的なテストと `tests/test_mi_qwave.py` を実行します。失敗がゼロであることを期待します。

<a id="task-2-posterior-ischemia-evidence-tiers"></a>
### タスク 2: 後部虚血の証拠の層

**ファイル:**
- 変更: `tests/test_clinical_ischemia.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/ischemia.py:146-171`

**インターフェース:**
- 消費: 信頼性の高い V1 ～ V3 ST-J、T 振幅、および R/S 形態。
- 生成: `posterior_ischemia_screen` は、ST 低下に支持される後方形態がある場合にのみ。孤立した ST 低下は境界線/不確定な証拠となります。

- [ ] 失敗したテストを追加します。正の端子 T またはドミナント R を持たない 2 つの押下リードは一致しません。うつ病と 2 つの誘導における陽性 T は、中程度の信頼度のスクリーニングと一致します。 RBBB/IVCD交絡リターンが不定であることが確認されました。
- [ ] 事後焦点テストを実行し、RED を確認します。
- [ ] ST `<= -0.05 mV` を備えた少なくとも 2 つの V1 ～ V3 リードと、ディプレスト リードのサポート ポジティブ T (`> 0.02 mV`) またはドミナント R (`R > abs(S)`) が必要です。どのサポート パスが一致したかを保存します。
- [ ] 最終コードの名前を `posterior_ischemia_screen` に変更し、V7 ～ V9 の推奨事項を維持し、信頼度を中程度に設定し、手動による確認を必要とします。
- [ ] `tests/test_clinical_ischemia.py` を実行すると、失敗はゼロになります。

<a id="task-3-qt-evidence-tiers-and-wide-qrs-handling"></a>
### タスク 3: QT 証拠層とワイド QRS の処理

**ファイル:**
- 変更: `tests/test_clinical_intervals.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/intervals.py`

**インターフェース:**
- 消費: QT、HR、性別、QRS 持続時間、RR 安定性、およびオプションの JT。
- 生成: 著しく短い (`<=340`)、短い可能性がある (`341–360`)、境界線の短い (`361–390`)、QT の延長、または幅広い QRS 再分極のレビュー。

- [ ] 選択された 340、350、380、および 391 ミリ秒での失敗境界テストを追加 QTc。アサート 380 は異常というより境界線です。
- [ ] QRS を証明する失敗したテストを追加します。 `>=120 ms` は、利用できない QT の代わりに JT が利用可能な場合に、JT/JTc 証拠を含む `wide_qrs_repolarization_review` 観測を生成します。
- [ ] 間隔テストを実行し、RED を確認します。
- [ ] 単一の 390 ミリ秒の短いしきい値を段階的なステートメントと重大度に置き換えます。式の値と手動レビューフラグを保持します。
- [ ] `global.jt_ms` をコンテキストから QT エバリュエーターに渡し、ワイド QRS について文書化された JTc 観測を計算します。そのフォールバックから QT 延長症候群を診断しないでください。
- [ ] `tests/test_clinical_intervals.py tests/test_clinical_export_consumers.py` を実行します。失敗がゼロであることを期待します。

<a id="task-4-af-evidence-normalization-and-p-dependent-applicability"></a>
### タスク 4: AF 証拠の正規化と P 依存の適用可能性

**ファイル:**
- 変更: `tests/test_af_afl_features.py`
- 変更: `tests/test_clinical_models_resolver.py`
- 変更: `tests/test_clinical_hypertrophy.py`
- 変更: `feature_extraction/ecgfeat/api.py:1492-1507`
- 変更: `feature_extraction/ecgfeat/clinical_rules/rhythm.py`
- 変更: `feature_extraction/ecgfeat/clinical_rules/hypertrophy.py`

**インターフェース:**
- 消費: 適格な拍動、RR CV、P 関連、および残存検証ごとにグループ化された心房イベント。
- 生成: `[0,1]` に限定された組織化 P 画分、QRST 検証とは独立した古典的な AF 証拠、および AF/AFL の下では適用されない心房形態。

- [ ] 1 拍に 2 つの候補 P イベントを含む不合格テストを追加し、1 拍で 1.0 を超えない組織化された P 分数カウントをアサートします。
- [ ] 古典的な RR/P AF が残留検証なしで一致し、AF または AFL が一致した場合には RAE/LAE が適用されなくなることを証明する不合格の臨床検査を追加します。
- [ ] 焦点を絞った AF および肥大テストを実行し、RED を確認します。
- [ ] 組織化された拍動を、関連する拍動 ID のセットとして適格な R ピーク拍動数で割って計算します。
- [ ] 活動性心房性不整脈のヘルパーを公開し、サプレッサーの証拠を含む適用されない心房形態規則を返します。
- [ ] パブリック形態学の入力が存在しない限り、フラッターを利用できないようにします。スペクトル証拠のみを宣伝しないでください。
- [ ] 焦点を絞ったテストを実行し、失敗がゼロであることを期待します。

<a id="task-5-p2-version-focused-regression-and-100-record-audit"></a>
### タスク 5: P2 バージョン、焦点を絞った回帰および 100 レコードの監査

**ファイル:**
- 変更: `feature_extraction/ecgfeat/clinical_rules/sources.py`
- 変更: 影響を受けるバージョン アサーション。
- 作成: `docs/unified_clinical_rules_v2_validation_audit.md`

**インターフェース:**
- 消費: 最終的な P2 コードと新しい 100 レコードの抽出。
- 生成: before/P0-P1/P2 配布と既知の制限を含むルールセット `2026.07.3`。

- [ ] `2026.07.3` の失敗したルールセット アサーションを書き込みます。
- [ ] 前払いのみ `RULESET_VERSION`; `clinical_rules.v2` スキーマを保持します。
- [ ] すべての `tests/test_clinical_*.py`、AF/AFL、Q-wave、リード反転、およびエクスポート コンシューマー テストを実行します。
- [ ] 文書化された 14 個の環境依存のベースライン障害を除いて、より広範なスイートを実行します。
- [ ] v1 アーティファクトを上書きせずに 100 レコードすべてを再計算し、全体的なステータス、部分的なカバレッジ、使用できないドメイン、最終コード数、AF/RBBB/RAE カバレッジ、事後スクリーン数、および QT 層をレポートします。
- [ ] Q 波による虚血の利用不能性が大幅に減少し、事後スクリーニングに裏付け証拠が必要で、組織化 P 比が 1 を超えず、すべてのアーティファクトが v2/2026.07.3 を使用する場合にのみ P2 を受け入れます。

