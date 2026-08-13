<!-- i18n-nav -->
[中文](ecg_agent_architecture_diagram.md) | [English](ecg_agent_architecture_diagram.en.md) | [日本語](ecg_agent_architecture_diagram.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# ECGAgent v38 Architecture Diagram

![ECGAgent v38 dual-channel architecture](ecg_agent_architecture_diagram.svg)

Blue denotes patient measurement and evidence flow, purple model reasoning, green deterministic program decisions, and gray dashed lines non-patient evidence or optional/bypass flows.

Downloads: [SVG vector](ecg_agent_architecture_diagram.svg) · [PNG image](ecg_agent_architecture_diagram.png)

Editable Mermaid source:

```mermaid
flowchart LR
    subgraph INPUT[① Input & Measurement]
        ECG[Standard 12-Lead ECG] --> FEAT[ecgfeat Deterministic Feature Extraction]
        FEAT --> FULL[Full features.json]
        FULL -.After blind planning.-> RULE[Rule Second-Opinion Snapshot<br/>Non-citable]
    end

    subgraph EVIDENCE[② Evidence & Tools]
        DTO[diagnosis-evidence.v2<br/>Diagnosis Evidence Allowlist DTO]
        STORE[Immutable EvidenceStore<br/>Pointers · units · caveats · fingerprints]
        TOOLS[Tool Registry<br/>14 Measurement Tools]
        VIEW[model-evidence.v3<br/>Atomic Qn Evidence View]
        DTO --> STORE --> TOOLS --> VIEW
    end

    FULL --> DTO

    subgraph ORCH[③ Agent Orchestration]
        QUALITY[Quality Gate + Urgent Review<br/>pass / partial / stop]
        PLAN[Compact Plan<br/>Up to 3 Blind Candidates]
        MERGE[Program-Validated Candidate Merge<br/>Rule Second Opinion · Up to 5 Total]
        PATH[Fixed Diagnostic Pathway Compiler<br/>Up to 13 Views / 6 Nodes per Path]
        QUALITY -->|pass / partial| PLAN --> MERGE --> PATH
    end

    STORE --> QUALITY
    VIEW --> PLAN
    RULE -.Non-patient evidence routing.-> MERGE

    subgraph DUAL[④ Dual-Channel Decision]
        MODEL[Model Channel<br/>Candidates and owner=model Open Nodes<br/>pass / fail / unknown]
        PROGRAM[Program Channel<br/>Thresholds · Counts · Ratios · Sequences<br/>owner=program Three-State Nodes]
        PLACE[Program Owns Final Placement<br/>confirmed / unresolved / rejected]
        MODEL --> PLACE
        PROGRAM --> PLACE
    end

    PATH --> MODEL
    PATH --> PROGRAM

    subgraph OUTPUT[⑤ Verification & Output]
        VERDICT[Expanded Structured Verdict<br/>Program-Materialized Values and Caveats]
        VERIFY[Deterministic Verification<br/>Schema · Provenance · Numbers · Qualifiers · Semantics]
        KNOW[Optional Isolated Knowledge Check<br/>Neutral Evidence-Review Questions Only]
        ARTIFACTS[JSON · Clinical Report · Markdown Trace<br/>Checkpoints · Batch · Blind Review / Release Gate]
        VERDICT --> VERIFY --> ARTIFACTS
        VERIFY -.After verification.-> KNOW
    end

    PLACE --> VERDICT
    QUALITY -.stop: program emits non-diagnostic verdict.-> VERDICT
    KNOW -.Reread patient evidence and reverify.-> TOOLS
```

See the [complete ECGAgent design document](ecg_agent_complete_design.en.md) (Chinese).
