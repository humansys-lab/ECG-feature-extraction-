# 代码库导航与维护约定

本仓库同时包含算法库、命令行入口、模型推理、数据集和实验产物。为保持
现有命令兼容，入口脚本暂时保留在根目录；新增的可复用逻辑应优先进入共享
模块，不再复制到多个脚本。

## 目录与职责

| 位置 | 职责 |
|---|---|
| `feature_extraction/ecgfeat/` | ECG 测量、质量控制、解释规则和结构化导出核心库 |
| `feature_extraction/ecgfeat/clinical_rules/` | 统一临床规则，各文件按诊断领域拆分 |
| `feature_extraction/ecgfeat/glasgow_rules/` | Glasgow 风格的测量与初步规则 |
| `ecgagent/` | `ecgfeat` 产物的 LLM Agent：诊断证据白名单、指针寻址、可靠性 caveat、工具注册、审计，以及只会阻止矛盾阳性结论的版本化最小安全策略；不产生测量或诊断 |
| `tests/` | 单元、契约和带数据的回归测试 |
| `docs/` | 算法说明、验证记录和设计文档 |
| `data/`、`all_diseases_*` | 数据集、批处理输入及实验产物，不应承载可复用代码 |
| `document/` | 外部参考资料 |
| `scripts/` | 与算法无关的运维、下载脚本 |

## 根目录 Python 入口

根目录文件按用途可分为以下几组：

- 演示与界面：`demo_feature_extraction.py`、`visualize_ecg.py`、`app.py`
- 特征批处理：`batch_extract_ecgfeat.py`
- MedGemma 推理：`batch_medgemma_diagnostics.py`、`run_layered_single.py`、
  `run_layered_batch_flat.py`
- 智能体：`python -m ecgagent.cli --features X.json --agent`（有界五阶段诊断循环：
  Survey 后以受控知识形成暂定诊断，再做针对性 ecgfeat 检查；随后进入
  确定性校验，默认 `claude-opus-5`；需 `ANTHROPIC_API_KEY`）。同一入口还提供
  `--briefing`、`--tool`、`--schema`、`--verify`、`--audit`。
  设计见 `docs/ecg_agent_architecture.md`，用法见 `ecgagent/README.md`
- LUDB 对比：`evaluate_ludb.py`、`compare_ludb_detectors.py`、
  `compare_annotations.py`、`plot_ludb_*.py`
- ST 与目标诊断验证：`validate_*.py`、`analyze_physionet_st.py`、
  `evaluate_target_ecgfeat_diagnosis.py`
- 结果分析与渲染：`analyze_ludb_*.py`、`ludb_*analysis.py`、
  `render_ecgfeat_record_explanations.py`

根目录入口可以组合参数和输出文件，但共享算法、输入解析或运行时初始化
不应继续放在入口内。目前共享的边界包括：

- `ecgfeat.io`：PhysioNet/WFDB 风格的 `.hea` 元数据和 `.mat` 信号读取
- `medgemma_runtime.py`：CUDA、量化、张量并行和 vLLM 文本生成初始化
- `medgemma_ecg_core.py`：MedGemma 上下文、提示、解析与诊断流程

## 修改建议

1. 核心测量或规则变更放到 `ecgfeat`，同时在 `tests/` 增加最小契约测试。
2. 新数据集只在适配器中完成导联顺序、增益和元数据映射，不把数据集判断
   写进核心算法。
3. 新 CLI 复用共享模块；仅参数解析、任务枚举和文件落盘保留在入口脚本。
4. 大型数据、图像、日志和 demo 输出不要提交为源码；对应模式已写入
   `.gitignore`。
5. 提交前运行：

   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/python -m compileall -q feature_extraction/ecgfeat *.py
   ```

`tests/test_js00059_clinical_regression.py` 属于生成物回归测试。只有根目录存在
`JS00059_features.json` 和 `JS00059_report.txt` 时才执行；可通过
`python demo_feature_extraction.py JS00059` 生成这两个文件及相关图像。
