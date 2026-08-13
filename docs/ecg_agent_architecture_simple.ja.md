<!-- i18n-nav -->
[中文](ecg_agent_architecture_simple.md) | [English](ecg_agent_architecture_simple.en.md) | [日本語](ecg_agent_architecture_simple.ja.md)
<!-- /i18n-nav -->

> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、原文を優先します。

<a id="ecgagent-simplified-architecture"></a>
# ECGAgent 簡略化されたアーキテクチャ

![ECGAgent 簡略化されたアーキテクチャ](ecg_agent_architecture_simple.svg)

このアーキテクチャは、ECG 入力 → 特徴抽出 → 証拠とツール → デュアルチャネル エージェントの決定 → 検証済みレポートという 1 つの明確なパスに従います。

ダウンロード: [SVG ベクター](ecg_agent_architecture_simple.svg) · [PNG イメージ](ecg_agent_architecture_simple.png)

```mermaid
flowchart LR
    ECG[12-Lead ECG] --> FEAT[ecgfeat<br/>Feature Extraction]
    FEAT --> EVIDENCE[Evidence Store<br/>+ Measurement Tools]

    subgraph AGENT[ECGAgent Decision Engine]
        MODEL[Model Channel<br/>Clinical Reasoning]
        PROGRAM[Program Channel<br/>Deterministic Safety Checks]
        MODEL --> DECISION[Final Decision]
        PROGRAM --> DECISION
    end

    EVIDENCE --> MODEL
    EVIDENCE --> PROGRAM
    DECISION --> OUTPUT[Verified Report<br/>+ Audit Trail]
```

すべての出力には人間によるレビューが必要です。
