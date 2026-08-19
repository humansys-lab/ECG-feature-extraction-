<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

# 本地模型对话前端

一个直接与本地权重对话的网页界面，同一套代码可以跑多个模型。全部推理在本机 GPU
上完成，对话内容不出这台机器。

```
浏览器  ──►  webchat/server.py (FastAPI)  ──►  vLLM OpenAI API  ──►  本地权重
```

拆成两个进程是有意为之：模型加载一次常驻显存，前端可以随时重启、改代码，
不需要重新加载几十 GB 的权重。

## 已配置的模型

| 档案 | 模型 | 界面端口 | 模型端口 | GPU | 特点 |
| --- | --- | --- | --- | --- | --- |
| `medgemma` | MedGemma-27B | 7860 | 8000 | 0 | 纯文本，医学/ECG 专用 |
| `qwen3.8` | Qwen3.8-27B | 7861 | 8001 | 3 | 多模态（能读图）+ 思考模式 |

两个档案端口不冲突，可以同时开着。全部参数在 [models.json](models.json) 里，
加一个新模型就是往里面加一段。

## 启动

```bash
./webchat/start_all.sh              # MedGemma（默认档案）→ http://localhost:7860
./webchat/start_all.sh qwen3.8      # Qwen3.8            → http://localhost:7861
```

脚本会先起模型服务、等权重加载完，再拉起界面；Ctrl+C 同时停掉两个。首次加载
50 GB 上下的权重通常要几分钟。

日常改前端建议分开起，这样重启界面不用重载模型：

```bash
./webchat/start_model.sh qwen3.8    # 终端 1：模型服务，一直开着
./webchat/start_ui.sh    qwen3.8    # 终端 2：网页界面，随便重启
```

## 远程访问

VS Code Dev Containers 里最省事：打开「端口 / PORTS」面板，转发 7860 / 7861，
然后在本机浏览器开 `http://localhost:7860`。

或者从本机 SSH 隧道到宿主机：

```bash
ssh -L 7860:<容器IP>:7860 <user>@<host>
```

## 常用环境变量

任何 `CHAT_*` 变量都会覆盖 models.json 里的档案设置：

| 变量 | 说明 |
| --- | --- |
| `CHAT_PROFILE` | 用哪个档案（也可以直接作为脚本的第一个参数） |
| `CHAT_GPUS` | 用哪几张卡，等价于 `CUDA_VISIBLE_DEVICES` |
| `CHAT_TP` | 张量并行度，多卡时要和 `CHAT_GPUS` 的卡数一致 |
| `CHAT_MAX_LEN` | 上下文长度上限（MedGemma 权重支持到 128k，Qwen3.8 到 256k） |
| `CHAT_GPU_UTIL` | 单卡显存占用比例 |
| `CHAT_PORT` / `CHAT_UI_PORT` | 模型服务 / 网页界面端口 |
| `CHAT_BASE_URL` | 界面连接的模型服务地址 |
| `CHAT_UI_HOST` | 界面监听地址，默认 `0.0.0.0` |

双卡示例：`CHAT_GPUS=0,3 CHAT_TP=2 CHAT_MAX_LEN=131072 ./webchat/start_all.sh qwen3.8`

## 界面功能

- **流式输出**，带 tokens/s、首字延迟、输入/输出 token 统计
- **多会话**，存在浏览器 localStorage 里（按档案分开存），刷新不丢，可导出 Markdown
- **上传 ECG 资料**（📎 / 拖拽 / 粘贴）：
  - `*_features.json` 会走 `medgemma_ecg_core.build_context_summary_from_paths`，
    也就是批量流水线喂给模型的同一套摘要格式，而不是把原始 JSON 倒进去；
  - `*_report.txt` 默认调 `sanitize_report_text` 去掉 `Dx` 诊断标签
    （和流水线一致，避免把答案直接喂给模型），可在「生成参数」里关掉。
- **图片输入**（仅多模态档案）：拖进来或粘贴即可；超过 2048px 的图会先缩放再送入
  模型以免撑爆上下文，PNG 保持 PNG 编码（JPEG 压缩会糊掉细的心电走线）。
  历史记录里存的是缩略图，避免撑爆 localStorage。
- **思考过程**（仅带 reasoning parser 的档案）：生成时实时展开，结束后自动折叠成
  「思考过程 · N 字」，点击可再展开；可在「生成参数」里整体关闭。
- 停止生成、重新生成、编辑并重发、复制、明暗主题

## 说明

- 模型服务只监听 `127.0.0.1`，但网页界面默认监听 `0.0.0.0`，方便从别的机器访问。
  如果这台机器在共享网络里，改成 `CHAT_UI_HOST=127.0.0.1` 并用端口转发，因为界面
  本身没有任何鉴权。
- MedGemma 的 checkpoint 是 `Gemma3ForCausalLM`（纯文本），不能读图；给它的 ECG
  只能是特征 JSON 或文本报告。Qwen3.8 是 `Qwen3_5ForConditionalGeneration`，可以读图。
- Gemma 的对话模板要求 user/assistant 严格交替，`server.py` 会自动合并同角色的
  相邻消息，所以前端不会因为连发两条而报错。
- 附件正文存在服务进程内存里，重启界面进程后旧附件失效（历史文字记录仍在）。
- 这是通用对话入口，走的是原始模型；带工具调用和证据校验的诊断流程仍然在
  `ecgagent/` 里。
