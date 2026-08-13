<!-- i18n-nav -->
[中文](codebase_guide.md) | [English](codebase_guide.en.md) | [日本語](codebase_guide.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="代码库导航与维护约定"></a>
# Code base navigation and maintenance conventions

This warehouse also includes algorithm libraries, command line entries, model inference, data sets and experimental products. to maintain
Existing commands are compatible, and the entry script is temporarily retained in the root directory; newly added reusable logic should be given priority in entering the share
Modules are no longer copied to multiple scripts.

<a id="目录与职责"></a>
## Directory and Responsibilities

| Position | Responsibilities |
|---|---|
| `feature_extraction/ecgfeat/` | ECG Measurement, quality control, interpretation rules and structured export core library |
| `feature_extraction/ecgfeat/clinical_rules/` | Unified clinical rules, each document split by diagnostic field |
| `feature_extraction/ecgfeat/glasgow_rules/` | Measurements and Preliminary Rules for Glasgow Style |
| `ecgagent/` | LLM Agent for `ecgfeat` product: diagnostic evidence whitelisting, pointer addressing, reliability caveats, tool registration, auditing, and a versioned minimal security policy that only prevents contradictory positive conclusions; no measurements or diagnostics are generated |
| `tests/` | Unit, contract and regression testing with data |
| `docs/` | Algorithm description, verification records and design documents |
| `data/`, `all_diseases_*` | Data sets, batch inputs and experimental products should not carry reusable code |
| `document/` | External References |
| `scripts/` | Algorithm-independent operation and maintenance, download scripts |

<a id="根目录-python-入口"></a>
## Root directory Python entry

Root directory files can be divided into the following groups according to their uses:

- Demo and interface: `demo_feature_extraction.py`, `visualize_ecg.py`, `app.py`
- Feature batch processing: `batch_extract_ecgfeat.py`
- MedGemma Reasoning: `batch_medgemma_diagnostics.py`, `run_layered_single.py`,
  `run_layered_batch_flat.py`
- Agent: `python -m ecgagent.cli --features X.json --agent` (bounded five-stage diagnostic cycle:
  After Survey, form a tentative diagnosis with controlled knowledge, and then conduct targeted ecgfeat inspection; then enter
  Deterministic verification, default `claude-opus-5`; `ANTHROPIC_API_KEY` required). The same entrance also offers
  `--briefing`, `--tool`, `--schema`, `--verify`, `--audit`.
  See `docs/ecg_agent_architecture.md` for design, and `ecgagent/README.md` for usage.
- LUDB Comparison: `evaluate_ludb.py`, `compare_ludb_detectors.py`,
  `compare_annotations.py`, `plot_ludb_*.py`
- ST and target diagnostic verification: `validate_*.py`, `analyze_physionet_st.py`,
  `evaluate_target_ecgfeat_diagnosis.py`
- Result analysis and rendering: `analyze_ludb_*.py`, `ludb_*analysis.py`,
  `render_ecgfeat_record_explanations.py`

Root entries can combine parameters and output files, but share algorithms, input parsing, or runtime initialization
It should not remain inside the entrance. Currently shared boundaries include:

- `ecgfeat.io`: PhysioNet/WFDB style `.hea` metadata and `.mat` signal reading
- `medgemma_runtime.py`: CUDA, quantization, tensor parallelism and vLLM text generation initialization
- `medgemma_ecg_core.py`: MedGemma context, prompts, analysis and diagnostic process

<a id="修改建议"></a>
## Modification suggestions

1. Core measurements or rule changes are placed in `ecgfeat`, and minimum contract tests are added in `tests/`.
2. The new data set only completes the lead sequence, gain and metadata mapping in the adapter, and does not judge the data set.
   Written into the core algorithm.
3. The new CLI reuses shared modules; only parameter parsing, task enumeration and file placement remain in the entry script.
4. Do not submit large data, images, logs and demo output as source code; the corresponding schema has been written
   `.gitignore`.
5. Run before submitting:

   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/python -m compileall -q feature_extraction/ecgfeat *.py
   ```

`tests/test_js00059_clinical_regression.py` is a product regression test. Only the root directory exists
Only executed when `JS00059_features.json` and `JS00059_report.txt`; can be passed
`python demo_feature_extraction.py JS00059` generates these two files and related images.
