<!-- i18n-nav -->
[中文](project_organization_and_cleanup.md) | [English](project_organization_and_cleanup.en.md) | [日本語](project_organization_and_cleanup.ja.md)
<!-- /i18n-nav -->

# 项目代码与文件整理报告

整理日期：2026-08-19

## 1. 整理结果

本次整理在不修改 ECG 算法和 Agent 业务逻辑的前提下，清除了历史测试草稿、缓存、日志、运行清单和可重新生成的实验输出。工作区占用从约 137 GB 降至约 125 GB，释放约 12 GB。

正式测试、原始数据集、本地模型、外部参考资料和当前 Python 虚拟环境均已保留。完整测试结果为 1515 项通过、2 项按设计跳过，另有 150 个子测试通过。

演示入口依赖但原先缺失的本地链接 `dataset -> data/010` 已恢复，并已验证示例 `.hea` 与 `.mat` 文件可访问。

## 2. 已删除内容

| 类别 | 删除内容 | 判断依据 | 恢复方式 |
|---|---|---|---|
| 历史测试草稿 | `tests/_multilead_pt_scratch.py` | 文件名不符合 pytest 收集规则，正式覆盖已存在于 `tests/test_multilead_pt_fusion.py` 等测试中 | 可从 Git 历史恢复 |
| Python 缓存 | 项目内的 `__pycache__/`、`.pytest_cache/`、`*.pyc` 和 `*.egg-info/` | 均由解释器、pytest 或安装流程自动生成 | 重新运行或安装即可生成 |
| 运行时中间文件 | `.doc_translation_cache.json`、两个 manifest、Web UI 上传目录及日志 | 已被 `.gitignore` 标记，不属于源码 | 重新运行对应命令生成 |
| 验证与绘图产物 | `ecgfeat_validation_20260811/`、`ludb_delineation_plots/`、`p_wave_failure_plots/` | 可由验证和绘图脚本重建 | 重新运行对应脚本生成 |
| 特征与推理输出 | `ptbxl_09000_ecgfeat/`、`qwen_agent_output/`、`deepseek_v4pro_05490_v12/` | 属于派生特征、诊断结果或实验输出，不是输入数据和源码 | 从保留的数据、模型和脚本重建 |

这些被忽略的生成产物没有 Git 版本，删除后不能通过 Git 直接还原；如有外部备份，也可从备份恢复。

## 3. 保留内容与目录职责

```text
ecg_gemma/
├── feature_extraction/ecgfeat/  # reusable ECG feature extraction library
├── ecgagent/                    # diagnostic agent and evidence pipeline
├── webchat/                     # web UI and model server
├── scripts/                     # maintenance, localization, and cohort utilities
├── tools/generation_conditions/ # focused audit and probe tools
├── tests/                       # unit, contract, and regression tests
├── docs/                        # design, algorithm, and validation documents
├── dataset -> data/010          # default demo dataset link (ignored)
├── data/                        # local raw/reference datasets (ignored)
├── medgemma-27b/                # local model (ignored)
├── qwen3.8-27b/                 # local model (ignored)
├── document/                    # external reference material (ignored)
├── *.py                         # compatible command-line entry points
└── requirements*.txt            # dependency definitions
```

| 区域 | 保留理由与维护边界 |
|---|---|
| `feature_extraction/ecgfeat/` | 核心 ECG 测量、质控、规则和导出实现；可复用算法应放在这里 |
| `ecgagent/` | LLM 后端、诊断流程、证据、工具、验证和报告；不承担底层 ECG 测量 |
| `tests/` | 1517 个正式测试用例的来源；测试是回归保护，不作为临时文件删除 |
| `docs/` | 算法设计、验证结论和维护记录；已统一 Markdown 扩展名 |
| `data/`、模型目录、`document/` | 体积大但属于必要输入或参考材料，保持本地且由 `.gitignore` 隔离 |
| 根目录 Python 文件 | 现有 CLI、演示、评估和绘图入口；多处存在直接导入，为保持命令和测试兼容暂不搬迁 |

根目录入口按用途分为：界面与演示（`app.py`、`demo_feature_extraction.py`、`visualize_ecg.py`）；提取与运行时（`batch_extract_ecgfeat.py`、`medgemma_ecg_core.py`、`medgemma_runtime.py`）；Agent/模型批处理（`run_*.py`、`batch_medgemma_diagnostics.py`）；数据集评估与验证（`evaluate_*.py`、`validate_*.py`、`compare_*.py`）；结果分析与可视化（`analyze_*.py`、`ludb_*analysis.py`、`plot_*.py`、`render_*.py`）。

## 4. 文档文件规范化

六份没有扩展名的正式文档已改为明确的 Markdown 文件：`P波文档.md`、`QRS.md`、`Twave related.md`、`morphological.md`、`p wave doc.md` 和 `流程.md`。对应的中、英、日导航和正文引用已同步更新。

本地化脚本也已修正：它现在会忽略索引中已不存在的旧路径，并检查未忽略的新文档，因此文件重命名或新增文档可在 Git 暂存前完成校验。

## 5. 后续维护规则

- 可随时清理项目源码目录内的 Python/pytest 缓存、日志、上传文件和明确的派生输出；不要对整个仓库运行无范围限制的清理命令。
- `data/`、`medgemma-27b/`、`qwen3.8-27b/`、`document/` 和 `.venv/` 不按中间产物处理，删除前必须单独确认。
- 正式测试使用 `tests/test_*.py` 命名；一次性探针放入 `tools/`，不要以不会被收集的下划线文件长期留在 `tests/`。
- 新的可复用逻辑进入 `ecgfeat` 或 `ecgagent`；根目录入口只保留参数解析、任务编排和输出落盘。
- 新生成的大型结果目录应加入 `.gitignore`，并在运行命令中使用清晰的输出目录名。

## 6. 验证结果

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q feature_extraction/ecgfeat ecgagent *.py scripts webchat
.venv/bin/python scripts/check_doc_localizations.py
```

- pytest：1515 passed，2 skipped，150 subtests passed。
- Python 编译检查：通过。
- 文档本地化结构、代码块和导航检查：通过。

两个跳过项均来自 `tests/test_js00059_clinical_regression.py`，需要先运行 `python demo_feature_extraction.py JS00059` 生成可选回归产物，符合该测试的设计。

## 7. 未执行的高风险重构

本次没有强制把根目录的 40 余个 CLI 脚本移动到多层子目录。它们之间以及测试代码中存在直接模块导入，直接搬迁会改变公开命令和 Python 导入路径。若以后需要进一步缩减根目录，应单独进行带兼容包装器、文档迁移和完整回归测试的入口迁移。
