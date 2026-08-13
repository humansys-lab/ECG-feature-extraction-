# ECGAgent Simplified Architecture

![ECGAgent simplified architecture](ecg_agent_architecture_simple.svg)

The architecture follows one clear path: ECG input → feature extraction → evidence and tools → dual-channel agent decision → verified report.

Downloads: [SVG vector](ecg_agent_architecture_simple.svg) · [PNG image](ecg_agent_architecture_simple.png)

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

All outputs require human review.
