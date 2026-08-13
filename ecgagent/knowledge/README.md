<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

# ECG General-Knowledge Layer

This local medical-knowledge layer is strictly separated from ecgfeat patient-measurement tools. It is not registered in `build_default_registry`, cannot use `ev:/...` citations, and can never become patient evidence.

The default `ECGDiagnosticAgent` uses it within these boundaries:

1. Survey reads patient measurements only and creates a neutral overview of the ten diagnostic domains.
2. After Survey, knowledge navigation retrieves a small governed set of English diagnostic, measurement-reliability and failure-mode references.
3. Hypothesize combines patient observations with general knowledge to create provisional hypotheses, differentials and targeted ecgfeat checks. Reference criteria are never represented as findings already present in the patient.
4. Retrieved excerpts are removed after Hypothesize. Investigate must confirm, weaken or reject every hypothesis using ecgfeat patient measurements.
5. Final diagnoses may cite only `ev:/...` patient measurements read in the current session and must pass deterministic validation.

Before an excerpt enters a patient-planning or challenge prompt, runtime filters remove case-specific lines and record identifiers. They also remove any non-English line as a language-boundary safeguard. Sanitization counts are written to the audit trail.

When `--knowledge-challenge` is enabled, a separate tool-free session reviews an already verified diagnosis. It produces neutral reacquisition questions only. Any accepted revision must call ecgfeat again and pass patient-evidence validation again.

## Runtime references

- `english_diagnostic_reference.md`
- `english_measurement_reliability.md`
- `english_failure_modes.md`

Implementation-comparison sources such as clinical-rule definitions, Philips DXL, Glasgow and capability blueprints remain excluded from runtime diagnostic prompts.

## Usage

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

Knowledge search returns reference material, not patient evidence. Its content never enters the patient-evidence citation whitelist.
