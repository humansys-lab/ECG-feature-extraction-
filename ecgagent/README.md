<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

# ECGAgent

ECGAgent is a measurement-grounded diagnostic Agent for standard 12-lead ECG feature artifacts produced by ecgfeat. It is a research system, not a medical device.

The runtime, model prompts, diagnostic pathways, reports, traces and batch-analysis artifacts are English-only. Historical result files are not rewritten.

## Core design

The language model is the diagnostic reasoner. ecgfeat is a read-only measurement instrument that provides global, lead-level and beat-level values, reliability flags and auditable JSON pointers. A measurement is evidence, not a diagnosis.

The default compact workflow is designed for local 27B-class models:

1. **Plan** — the model reads one compact, diagnosis-neutral overview and proposes at most three candidates with support and falsification views.
2. **Programmatic routing** — the orchestrator adds diagnosis-specific pathway steps, merges overlapping views and prefetches bounded tool results.
3. **Adjudicate** — deterministic nodes are computed in code; the model answers only morphology, mechanism and differential questions that require interpretation.
4. **Render and verify** — code merges both node types, determines final placement, materializes true values from citations, renders reports and performs structural, semantic, numeric and evidence validation.

The older five-phase ledger workflow remains available with `--diagnostic-workflow legacy`. The CLI and batch runner default to `compact`.

## Evidence rules

- A patient-specific number must come from a tool result and carry an exact `ev:/...` citation.
- Values are materialized together with unit, reliability and caveats.
- Tool access and model visibility are separate ledgers. Only pointers actually visible to the model become authorized citations.
- Candidate-detector fields from one algorithmic chain are not independent corroboration.
- A representative beat is not automatically representative of the entire recording.
- Missing PR does not mean absent P waves; missing QT does not mean uninformative T waves.
- Rate and rhythm mechanism are assessed separately.
- Positive diagnoses require definition-level evidence, independent corroboration when morphology or mechanism is involved, and active counterevidence review.

## Diagnosis-specific pathways

Each active candidate receives a short pathway with explicit required, supporting and invalidator nodes. Nodes have one of two owners:

- `owner=program`: thresholds, counts, ratios already returned by tools, minimum sample gates, sequence summaries and other deterministic calculations.
- `owner=model`: cross-lead morphology, rhythm mechanism, competing explanations and clinical interpretation of conflicting observations.

Program-owned coverage includes PR and QT reportability and thresholds, P-QRS association counts, repeated nonconducted atrial events, AV sequence summaries, directly measured atrial and ventricular rates, P-wave axis, RR irregularity, QRS duration, morphology-group prevalence, premature timing, repeated wide-QRS sequences, low voltage, pacing markers/capture/sensing, right-precordial voltage relationships and precordial R/S progression.

Missing fields, insufficient reliability or inadequate sample counts remain `unknown`. Candidate-event counts are never promoted into AV conduction ratios.

## Rule-based second opinion

The compact workflow uses a dual-channel candidate process. The model first proposes independent candidates from a neutral measurement overview. Only after that plan is frozen can the program add a small number of sanitized ecgfeat rule-based second-opinion candidates.

Rule status is never support or counterevidence. A rule-added candidate must complete the same independent measurement pathway and falsification process as a model candidate. Unadjudicated rule candidates are abstained or sent for human review, never confirmed automatically.

## Available tools

| Tool | Purpose |
|---|---|
| `get_diagnostic_overview()` | Compact quality, rate, global intervals, axes and modality availability for initial planning |
| `get_global_table(fields)` | Read selected global measurements |
| `get_measurement(pointer)` | Read one measurement by JSON pointer or clinical alias |
| `get_lead_table(fields, leads?)` | Read selected fields across leads |
| `get_beat_table(fields, lead?, beat_ids?)` | Read beat-level rhythm or lead-specific fields |
| `search_measurements(query, limit?)` | Discover measurement names when the pointer is unknown |
| `get_rhythm_profile(sections?)` | Organize background rhythm, atrial signal, AV association, pre-excitation and aberrancy observations |
| `get_atrial_event_table(...)` | Read the atrial candidate-event stream and P-QRS associations |
| `get_p_assessment_table(...)` | Review P-wave boundary acceptance, ambiguity and confidence |
| `get_morphology_groups(...)` | Review QRS morphology-family prevalence, runs and beat membership |
| `get_morphology_map(profile, leads?)` | Review bounded cross-lead QRS, ST or T/U morphology |
| `get_interval_waveform_context(interval)` | Review PR or QT/QTc status together with retained component-wave information |
| `get_pacing_profile(...)` | Review pacing markers, capture, sensing and evidence conflicts |
| `get_native_beat_profile(profile, leads?, max_beats?)` | Review repeated native/non-paced beats for Q waves, R-wave progression or repolarization |
| `list_findings` / `get_rule_detail` | Legacy adjudication mode only; unavailable to diagnostic phases |

## English general-knowledge layer

The optional knowledge layer is not an ecgfeat tool and never becomes patient evidence. Runtime navigation uses concise English diagnostic, measurement-reliability and failure-mode references in `ecgagent/knowledge/`.

Knowledge excerpts can suggest relationships, confounders and targeted checks. Final conclusions must return to patient measurements with `ev:/...` citations. Clinical-rule source, Philips DXL, Glasgow and capability blueprints remain excluded from patient-level runtime prompts.

## Reports and traces

Every successful run can produce:

- `diagnoses/<record>.json` — structured verdict, validation and audit data
- `diagnoses/<record>.md` — detailed English interpretation report
- `brief_reports/<record>_brief.md` — concise English conclusion, evidence, limitations and recommendations
- `agent_traces/<record>_agent_trace.md` — phase output, model-visible tool context, complete raw tool audit, validation and runtime usage

Reports are rendered deterministically from the structured verdict. No second model paraphrases the result.

## Run one record with a local Qwen server

Start vLLM separately, then run:

```bash
cd /workspace/ecg_gemma

.venv/bin/python -m ecgagent.cli \
  --features /workspace/ecg_gemma/ptbxl_09000_ecgfeat/features/09025_hr_features.json \
  --record-id 09025_hr \
  --agent \
  --backend qwen-local \
  --model qwen3.6-27b \
  --qwen-base-url http://127.0.0.1:8000/v1 \
  --diagnostic-workflow compact
```

The Qwen backend uses OpenAI-compatible `tools` and `tool_calls`. In compact mode, tool views are prefetched by the orchestrator, so Qwen does not need to generate tool calls. Strict JSON-schema decoding constrains both the plan and final adjudication.

## Batch diagnosis

```bash
cd /workspace/ecg_gemma

.venv/bin/python -m ecgagent.batch diagnose \
  --output-dir /workspace/ecg_gemma/qwen_agent_output/qwen_english_run \
  --backend qwen-local \
  --model qwen3.6-27b \
  --qwen-base-url http://127.0.0.1:8000/v1 \
  --diagnostic-workflow compact \
  --agent-workers 2 \
  --verbose
```

Reference labels are joined only after Agent diagnosis completes and are never included in model context.

## Context controls for local models

Long `ev:/...` pointers are projected to stable short references such as `Q1` inside Qwen context while retaining semantic labels such as `V2.t_amp_mv` or `global.qrs_ms`. Program code restores original pointers before final validation.

Tool results are packed as complete evidence atoms containing citation, value, unit, reliability and caveats. Packing truncates only at atom boundaries and rotates fairly across tools, leads, beats and event sections. The full raw results remain available in the trace.

Identical tool-and-argument calls within a phase are deduplicated. Whole-record limits bound model turns, token usage, wall time and consecutive `max_tokens` truncations.

## Validation boundary

Deterministic validation checks output structure, citation authorization, exact values and units, reliability qualification, numeric comparisons, diagnosis-code placement and selected definition-level contradictions. It never creates a diagnosis and does not read dataset labels or clinical-rule conclusions.

All positive findings, downgraded findings and non-pass quality-gate results require qualified human review. ECGAgent output does not replace the original waveform, clinical history, symptoms, prior tests or a formal medical diagnosis.
