<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="ecg-general-knowledge-layer"></a>
# ECG 一般知識レイヤー

このローカルな医療知識レイヤーは、ecgfeat 患者測定ツールから厳密に分離されています。`build_default_registry` に登録されておらず、`ev:/...` 引用を使用できず、患者証拠になることは決してありません。

デフォルトの `ECGDiagnosticAgent` は、これらの境界内でこれを使用します：

1. Survey は患者測定のみを読み取り、10の診断ドメインの中立的な概要を作成します。
2. Survey の後、知識ナビゲーションは、英語の診断、測定信頼性、および故障モードの参照の小さな管理されたセットを取得します。
3. Hypothesize は患者観察と一般知識を組み合わせて、仮説、鑑別診断、およびターゲットされた ecgfeat チェックを作成します。参照基準は、患者にすでに存在する所見として表現されることはありません。
4. 取得された抜粋は、Hypothesize の後に削除されます。Investigate は、ecgfeat 患者測定を使用して、すべての仮説を確認、弱体化、または拒否する必要があります。
5. 最終診断は、現在のセッションで読み取られた `ev:/...` 患者測定のみを引用でき、決定論的検証に合格する必要があります。

抜粋が患者計画またはチャレンジプロンプトに入る前に、ランタイムフィルターはケース固有の行と記録識別子を削除します。また、言語境界の保護策として、英語以外の行もすべて削除します。サニタイズカウントは監査証跡に記録されます。

`--knowledge-challenge` が有効になっている場合、別のツールなしセッションは、すでに検証された診断をレビューします。中立的な再取得質問のみを生成します。承認された修正は、ecgfeat を再度呼び出し、患者証拠検証を再度通過する必要があります。

<a id="runtime-references"></a>
## ランタイム参照

- `english_diagnostic_reference.md`
- `english_measurement_reliability.md`
- `english_failure_modes.md`

臨床ルール定義、Philips DXL、グラスゴー、および機能ブループリントなどの実装比較ソースは、ランタイム診断プロンプトから除外されます。

<a id="usage"></a>
## 使用法

```bash
# List indexed sources
python -m ecgagent.knowledge sources

# Build an in-memory index and search it
python -m ecgagent.knowledge search "atrial flutter F wave P-QRS association"

# Search only failure modes
python -m ecgagent.knowledge search "single representative beat" \
  --category failure_modes

# Persist an index explicitly
python -m ecgagent.knowledge build \
  --output /tmp/ecg_knowledge_index.json
```

ナレッジ検索は参照資料を返すものであり、患者証拠ではありません。その内容は、患者証拠の引用ホワイトリストに決して含まれません。
