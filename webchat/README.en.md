<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="本地模型对话前端"></a>
# Local Model Chat Frontend

A web interface that directly interacts with local weights, allowing the same codebase to run multiple models. All inference is performed on the local GPU, and chat content never leaves this machine.

```
浏览器  ──►  webchat/server.py (FastAPI)  ──►  vLLM OpenAI API  ──►  本地权重
```

Splitting into two processes is intentional: the model is loaded once and remains in VRAM, allowing the frontend to be restarted or modified at any time without needing to reload tens of gigabytes of weights.

<a id="已配置的模型"></a>
## Configured Models

| Profile | Model | UI Port | Model Port | GPU | Features |
| --- | --- | --- | --- | --- | --- |
| `medgemma` | MedGemma-27B | 7860 | 8000 | 0 | Text-only, specialized for medicine/ECG |
| `qwen3.6` | Qwen3.6-27B | 7861 | 8001 | 3 | Multimodal (can read images) + thinking mode |

The ports for the two profiles do not conflict, so they can be run simultaneously. All parameters are in [models.json](models.json), and adding a new model simply involves adding a new entry there.

<a id="启动"></a>
## Startup

```bash
./webchat/start_all.sh              # MedGemma（默认档案）→ http://localhost:7860
./webchat/start_all.sh qwen3.6      # Qwen3.6            → http://localhost:7861
```

The script first starts the model service, waits for the weights to finish loading, and then launches the interface; pressing Ctrl+C stops both simultaneously. The initial loading of weights, which are typically around 50 GB, usually takes a few minutes.

For routine frontend modifications, it is recommended to start them separately so that restarting the interface does not require reloading the model:

```bash
./webchat/start_model.sh qwen3.6    # 终端 1：模型服务，一直开着
./webchat/start_ui.sh    qwen3.6    # 终端 2：网页界面，随便重启
```

<a id="远程访问"></a>
## Remote Access

The most convenient approach is within VS Code Dev Containers: open the "Ports / PORTS" panel, forward ports 7860 / 7861,
and then open `http://localhost:7860` in your local browser.

Alternatively, create an SSH tunnel from your local machine to the host:

```bash
ssh -L 7860:<容器IP>:7860 <user>@<host>
```

<a id="常用环境变量"></a>
## Common Environment Variables

Any `CHAT_*` variable will override the settings in the models.json profile:

| Variable | Description |
| --- | --- |
| `CHAT_PROFILE` | Which profile to use (can also be passed directly as the first argument to the script) |
| `CHAT_GPUS` | Which GPUs to use, equivalent to `CUDA_VISIBLE_DEVICES` |
| `CHAT_TP` | Tensor parallelism; when using multiple GPUs, this must match the number of GPUs in `CHAT_GPUS` |
| `CHAT_MAX_LEN` | Maximum context length (MedGemma weights support up to 128k, Qwen3.6 up to 256k) |
| `CHAT_GPU_UTIL` | Proportion of VRAM used per GPU |
| `CHAT_PORT` / `CHAT_UI_PORT` | Model service / Web UI port |
| `CHAT_BASE_URL` | Model service address for the UI to connect to |
| `CHAT_UI_HOST` | UI listening address, default is `0.0.0.0` |

Dual-GPU example: `CHAT_GPUS=0,3 CHAT_TP=2 CHAT_MAX_LEN=131072 ./webchat/start_all.sh qwen3.6`

<a id="界面功能"></a>
## UI Features

- **Streaming output**, with tokens/s, time-to-first-token latency, and input/output token statistics
- **Multiple sessions**, stored in browser localStorage (separated by profile), persist across refreshes, and can be exported as Markdown
- **Upload ECG materials** (📎 / drag-and-drop / paste):
  - `*_features.json` will go through `medgemma_ecg_core.build_context_summary_from_paths`,
    meaning it uses the same summary format fed to the model in batch pipelines, rather than dumping raw JSON;
  - `*_report.txt` defaults to calling `sanitize_report_text` to remove `Dx` diagnostic tags
    (consistent with the pipeline to avoid feeding answers directly to the model); this can be disabled in "Generation Parameters".
- **Image input** (multimodal profiles only): drag-and-drop or paste; images larger than 2048px are resized before being sent to the model to prevent context overflow, and PNGs retain PNG encoding (JPEG compression blurs fine ECG traces).
  Thumbnails are stored in history to avoid bloating localStorage.
- **Thinking process** (profiles with reasoning parser only): expands in real-time during generation, then automatically collapses into
  "Thinking Process · N characters" after completion; click to expand again; can be globally disabled in "Generation Parameters".
- Stop generation, regenerate, edit and resend, copy, light/dark theme

<a id="说明"></a>
## Notes

- The model service only listens on `127.0.0.1`, but the web UI defaults to listening on `0.0.0.0`, allowing access from other machines.
  If this machine is on a shared network, change it to `CHAT_UI_HOST=127.0.0.1` and use port forwarding, as the UI
  has no authentication.
- The checkpoint for MedGemma is `Gemma3ForCausalLM` (text-only) and cannot read images; the ECG
  provided to it must be feature JSON or text reports. Qwen3.6 is `Qwen3_5ForConditionalGeneration` and can read images.
- Gemma's chat template requires strict alternation between user/assistant; `server.py` automatically merges adjacent messages from the same role, so the frontend won't error out if two messages are sent consecutively.
- Attachment content is stored in the service process's memory; old attachments become invalid after restarting the UI process (though historical text records remain).
- This is a general chat entry point using the raw model; the diagnostic workflow with tool calls and evidence verification remains in
  `ecgagent/`.
