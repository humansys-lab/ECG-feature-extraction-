<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

# ECG Gemma：12 导联 ECG 特征提取与临床证据工程

本项目提供一套可解释的 12 导联 ECG 特征提取流水线，并将测量结果组织成可供规则引擎、报告程序和 MedGemma 使用的结构化证据。

它不只是 R 峰检测器。当前流程包含信号预处理、质量控制、起搏尖峰检测、多导联 QRS 检测、心拍分组、代表心拍构建、P/QRS/T 波界点定位、间期与心电轴测量、房性节律分析，以及传导、肥厚和 ST-T 等临床规则。

> **重要说明**
>
> 本项目是研究和工程用途的软件，不是医疗器械，不能代替医生判读，也不应直接用于临床诊断或治疗决策。

## 1. 项目结构

```text
ecg_gemma/
├── feature_extraction/
│   ├── ecgfeat/                  # ECG 特征提取核心包
│   ├── README.md                 # 核心包的补充说明
│   └── pyproject.toml
├── ecgagent/                     # 诊断 Agent、证据与校验
├── webchat/                      # Web UI 与模型服务
├── scripts/                      # 维护、本地化与队列工具
├── tools/generation_conditions/  # 专项审计与探针
├── tests/                        # 单元、契约与回归测试
├── docs/                         # 设计、算法与验证文档
├── demo_feature_extraction.py    # 单条 WFDB .mat/.hea 记录演示
├── batch_extract_ecgfeat.py      # ECG 特征批处理
├── medgemma_ecg_core.py          # MedGemma 上下文与诊断流程
├── medgemma_runtime.py           # 共享 GPU/vLLM 运行时初始化
├── app.py                        # Gradio 界面入口
├── evaluate_*.py / validate_*.py # 数据集评估与验证入口
├── dataset -> data/010           # demo 默认读取的数据目录
├── data/                         # 本地原始/参考数据（Git 忽略）
├── medgemma-27b/                 # 本地 MedGemma 模型（Git 忽略）
├── qwen3.8-27b/                  # 本地 Qwen 模型（Git 忽略）
├── document/                     # 外部参考资料（Git 忽略）
├── requirements.txt              # 完整项目依赖
└── .gitignore                    # 本地大文件与生成物隔离规则
```

更完整的入口分类、共享模块边界和维护约定见
[代码库导航与维护约定](docs/codebase_guide.md)。

本次文件清理、保留范围和后续规则见
[项目代码与文件整理报告](docs/project_organization_and_cleanup.md)。

ECG 诊断 Agent 的当前双通道架构、证据合同、Compact/Legacy 工作流、
工具与校验设计见 [ECGAgent 完整设计文档](docs/ecg_agent_complete_design.md)。

核心模块：

| 模块 | 作用 |
|---|---|
| `ecgfeat/api.py` | 完整流水线入口 `ECGFeatureExtractor.extract()` |
| `ecgfeat/preprocess.py` | 重采样、去基线、陷波和检测信号构建 |
| `ecgfeat/quality.py` | 每导联及每种测量功能的质量门控 |
| `ecgfeat/pacing.py` | 起搏尖峰检测与验证 |
| `ecgfeat/qrs.py` | 多导联 QRS/R 峰检测 |
| `ecgfeat/grouping.py` | 心拍形态分组 |
| `ecgfeat/representative.py` | 代表心拍对齐与融合 |
| `ecgfeat/delineate.py` | P/QRS/T 波峰值和边界定位 |
| `ecgfeat/st_localization.py` | 稳健 PR 基线、J 点共识和 ST 多点旁路测量 |
| `ecgfeat/features.py` | 单导联、分组和全局特征聚合 |
| `ecgfeat/atrial.py` | 房性事件、QRST subtraction、AF/AFL 证据 |
| `ecgfeat/clinical_rules/` | 节律、传导、间期、肥厚、缺血等规则 |
| `ecgfeat/export.py` | JSON 转换和结构化导出 |
| `medgemma_ecg_core.py` | 面向 MedGemma 的上下文、提示与分层诊断流程 |

## 2. 运行环境

建议使用 Python 3.10 或更高版本。

### 仅运行 ECG feature extraction

```bash
cd /workspace/ecg_gemma

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ./feature_extraction
python -m pip install matplotlib pytest
```

`feature_extraction` 包会安装核心依赖 NumPy 和 SciPy。绘图需要 Matplotlib。

### 完整项目依赖

```bash
python -m pip install -r requirements.txt
```

完整依赖包含 `vllm`、`transformers`、Gradio 等较大的模型和界面组件。如果只进行 ECG 特征提取，不需要安装全部依赖。

处理 `.pt` 文件时还需要根据机器的 CPU/CUDA 环境单独安装 PyTorch。

### LUDB 上比较 ecgfeat、NeuroKit2 和 BioSPPy

对比工具会读取 LUDB 的逐导联专家标注，统计 R 峰、P/QRS/T 波检测率和匹配事件的边界误差。BioSPPy 的公共 ECG 流程只提供 R 峰，因此只参加 QRS/R 峰比较。

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pip install -r requirements-ludb-compare.txt

# 快速验证：记录 1、II 导联
python compare_ludb_detectors.py --records 1 --leads II \
  --out-dir ludb_three_way_smoke

# 完整 LUDB、全部 12 导联
python compare_ludb_detectors.py --workers 4 \
  --out-dir ludb_three_way_comparison

# 前10条记录，使用每个导联自己的 ecgfeat QRS fiducial
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --ecgfeat-r-source lead-fiducial \
  --out-dir ludb_three_way_first10_per_lead_r

# 前10条记录，使用混合 prominence 定位
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --methods ecgfeat \
  --ecgfeat-r-source hybrid-prominence \
  --out-dir ludb_ecgfeat_first10_hybrid_prominence

# 前10条记录：混合 R 定位 + 双极性 T 峰细化
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --methods ecgfeat \
  --ecgfeat-r-source hybrid-prominence \
  --ecgfeat-wave-source hybrid-peaks \
  --out-dir ludb_ecgfeat_first10_hybrid_waves
```

完整的方法定义、参数、指标和输出文件说明见 [LUDB 三种检测方法对比](docs/ludb_detector_comparison.md)。

关于吸收 NeuroKit2 局部定位优点、改进 ecgfeat 的 R/P/T/S 波检测及
稳健 ST/J 测量方案，
见 [ecgfeat × NeuroKit2 混合改进设计](docs/ecgfeat_neurokit_hybrid_improvement.md)。

T 波增强默认开启：使用 22.5 Hz 专用支路并行计算 Mallat 二次样条
小波和低幅 T 梯形面积候选，结合 T-SQI、RMS/PC1 与
weighted-median/MAD/Huber 多导联融合，同时分别输出稳健中心和
P85 最晚复极。只有融合质控通过时才更新 QT 共识，本地 T offset
仅在严格的跨导联 late-tail rescue 下修改。如需历史回归，可传入
`enable_t_wave_refinement=False`。

稳健 ST 增强输出默认开启，输出 PR 段基线、P85 多导联共识 J 点、
J/J+20/J+40/J+60/J+80 和 RR/Ton 自适应点的窗口中位数，以及稳健
斜率、二次曲率、趋势和凹凸形态。它不覆盖原生 ST 字段，也不改变临床解释；如需做严格历史回归，可在
构造提取器时传入 `enable_hybrid_st_measurement=False`。

验证本地 `dataset` 中带 ST 标签的记录：

```bash
# 扫描全部标签并对100条记录做标签级比较
PYTHONPATH=feature_extraction .venv/bin/python validate_dataset_st.py \
  --dataset-dir dataset \
  --out-dir dataset_st_validation \
  --workers 4

# 仅提取明确 ST 标签和 ST-T/心梗探索记录
PYTHONPATH=feature_extraction .venv/bin/python validate_dataset_st.py \
  --dataset-dir dataset \
  --out-dir dataset_st_candidates \
  --candidates-only
```

脚本会输出候选清单、逐记录/逐导联/逐心搏 CSV、标签一致性指标及明确
ST 标签记录的 12 导联标记图。由于该数据只有记录级诊断标签，没有
逐导联 J 点或 ST 幅值真值，输出不能当作临床准确率。

在同一条 ECG 上叠加项目算法与 NeuroKit2 的峰值，并分行显示两者边界：

```bash
# 完整记录
python plot_ecgfeat_neurokit_comparison.py 1 --lead II

# 显示第 2–5 个 LUDB QRS 对应的连续心搏
python plot_ecgfeat_neurokit_comparison.py 8 --lead V2 --beat-range 2 5
```

绘制一条记录完整的 12 导联、10 秒对比：

```bash
# 记录 1；生成 4 张 PNG 和一个四页 PDF
python plot_ludb_12lead_10s_comparison.py 1

# 批量绘制记录 1–10
python plot_ludb_12lead_10s_comparison.py {1..10}

# 如需复现旧图中的全局多导联 R 标记
python plot_ludb_12lead_10s_comparison.py 1 \
  --ecgfeat-r-source global
```

绘图默认使用独立的逐导联 prominence 定位（`hybrid-prominence`）；
心搏数量仍由 ecgfeat 的多导联检测决定。该旁路字段不替换全局
`r_locs`、原有 `qrs.peak` 或 QRS 边界，因此不影响最终测量和解释。

## 3. 输入约定

特征提取器期望：

- 标准 12 导联 ECG；
- 数组形状为 `[12, n_samples]`，也接受 `[n_samples, 12]`；
- 推荐导联顺序为 `I, II, III, aVR, aVL, aVF, V1–V6`；
- 传入信号的幅度单位为 mV；
- 显式提供原始采样率；
- 患者年龄和性别是可选元数据，但部分临床规则需要它们。

demo 脚本默认读取：

```text
dataset/
├── JS00001.hea
├── JS00001.mat
├── JS00002.hea
└── JS00002.mat
```

`.mat` 中应包含形状为 `[12, n_samples]` 的 `val` 数组。当前 demo 默认按 `1000 ADC units/mV` 转换；接入新数据集前，应确认每条 `.hea` 中的实际增益、单位和导联顺序。

## 4. 快速运行

### 随机抽取一条记录

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python demo_feature_extraction.py
```

### 抽取指定记录

```bash
python demo_feature_extraction.py JS00010
```

默认会在项目根目录生成：

```text
JS00010_features.json
JS00010_report.txt
JS00010_ecg.png
JS00010_ecg_annotated.png
JS00010_ecg_report.png
```

### 保留逐拍、逐导联完整特征

```bash
python demo_feature_extraction.py JS00010 --full-beats
```

默认 JSON 会省略体积最大的 `beat_features`。只有调试波界定位、审计单拍结果或训练需要逐拍数据时，才建议使用 `--full-beats`。

## 5. 抽取 dataset 的前 10 条

下面的命令按记录编号自然排序，抽取前 10 条，并把生成文件移动到独立目录：

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate

output_dir="data/dataset_first10_ecgfeat_out"
mkdir -p "$output_dir"

find -L dataset -maxdepth 1 -type f -name '*.hea' -printf '%f\n' \
  | sed 's/\.hea$//' \
  | sort -V \
  | head -n 10 \
  | while read -r record_id; do
      python demo_feature_extraction.py "$record_id"

      for suffix in \
        features.json \
        report.txt \
        ecg.png \
        ecg_annotated.png \
        ecg_report.png; do
        artifact="${record_id}_${suffix}"
        if [ -f "$artifact" ]; then
          mv "$artifact" "$output_dir/"
        fi
      done
    done
```

当前已生成的前 10 条结果位于：

```text
data/dataset_first10_ecgfeat_out/
```

## 6. `.pt` 批量抽取

`batch_extract_ecgfeat.py` 不处理 WFDB `.mat/.hea`，而是处理下面这种目录：

```text
all_diseases_pt100/
├── norm/
│   ├── generated.pt
│   ├── real.pt
│   └── metadata.pt
├── ami/
│   ├── generated.pt
│   └── real.pt
└── ...
```

每个张量可以是：

- `[n_samples, 12, n_points]`
- `[n_samples, n_points, 12]`
- 单条 `[12, n_points]`
- 单条 `[n_points, 12]`

先用少量样本进行 smoke test：

```bash
python batch_extract_ecgfeat.py \
  --input-dir all_diseases_pt100 \
  --output-dir all_diseases_ecgfeat \
  --sampling-rate 100 \
  --sources generated,real \
  --limit 2 \
  --annotated-plots
```

正式批处理：

```bash
python batch_extract_ecgfeat.py \
  --input-dir all_diseases_pt100 \
  --output-dir all_diseases_ecgfeat \
  --sampling-rate 100 \
  --sources generated,real \
  --skip-existing
```

如需在 JSON 中保留逐拍明细，增加：

```bash
--include-beat-features
```

批处理的 `.pt` 特征文件始终保留完整逐拍数据；该参数只控制 JSON 是否保留 `beat_features`。

## 7. Python API

```python
import numpy as np

from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.export import prepare_json_export, to_dict
from ecgfeat.models import PatientMeta

# 形状：[12, n_samples]；单位：mV
ecg = np.random.randn(12, 5000) * 0.02
fs = 500

extractor = ECGFeatureExtractor(
    fs_internal=500,
    mains_freq=50,
)

features = extractor.extract(
    ecg,
    fs=fs,
    meta=PatientMeta(age=45, sex="Male"),
)

print(features.global_features)

payload = prepare_json_export(
    to_dict(features),
    include_beat_features=False,
)
```

如果数据来自 ADC 整数值，应先使用每条记录自己的 gain 转换为 mV：

```python
ecg_mv = ecg_adc.astype(np.float64) / gain_units_per_mv
```

## 8. 算法流程

```text
原始 12 导联
    ↓
输入方向检查与重采样
    ↓
去基线和工频陷波
    ├── analysis signal：保留 ST/T 幅度
    └── detection signal：额外低通，增强 QRS
            ↓
质量评估、导联反接和起搏检测
            ↓
多导联 QRS 检测
            ↓
心拍形态聚类和代表心拍
            ↓
P/QRS/T 波定位及多导联共识修复
            ↓
PR、QRS、QT、JT、QTc、ST、振幅、面积和心电轴
            ↓
房性事件、AF/AFL 和临床规则
            ↓
JSON、文本报告、ECG 图像和 MedGemma 证据
```

### 8.1 双信号预处理

- `analysis_signal` 使用两级中值滤波去除基线漂移，并进行工频陷波；
- `detection_signal` 额外进行低通，供 QRS 检测使用；
- ST 段、T 波和振幅测量尽量使用保真度更高的分析信号。

### 8.2 质量控制

每个导联分别计算基线漂移、肌电噪声、工频干扰、削波和低振幅等指标。P 波、QRS、T 波和 QT 使用不同的可靠性门槛。

质量等级通过不等于测量绝对正确。调用方应同时读取数值、置信度、来源和 availability。

### 8.3 QRS 与代表心拍

QRS 检测融合多个肢体和胸前导联，使用带通、差分、平方、移动积分和自适应阈值寻找候选 R 峰。检测出的心拍按形态聚类，然后通过互相关对齐和中位数/medoid 方法构造代表心拍。

### 8.4 P/QRS/T delineation

系统为每个心拍、每个导联估计：

- P onset、peak、offset；
- QRS onset、Q/R/S 峰及 QRS offset；
- T peak 和 T end；
- ST-J、ST midpoint 和 ST end。

定位过程中会结合代表心拍先验、局部斜率、切线/几何方法、生理范围和多导联支持。宽 QRS、起搏、低振幅和长 PR 有额外补救路径。

### 8.5 全局特征

主要输出包括：

- HR 和 RR；
- PR；
- QRS duration；
- QT 和 JT；
- QTc Bazett、Fridericia，以及规则层中的 Hodges、Framingham；
- P/QRS/T/ST axis；
- QT dispersion；
- P、Q、R、S、T 振幅与面积；
- ST 偏移和形态证据。

宽 QRS或起搏情况下会降低普通 QT 的权重，并更多使用 JT 和宽 QRS 专用路径。

### 8.6 房性节律

AF/AFL 分析不仅使用 RR 不规则性，还包括：

- P 波组织性；
- 房室关联；
- QRST 模板减除；
- 残差信号重复性；
- 多导联 F 波频率一致性。

如果 QRST subtraction 没有通过质量验证，系统会保留 scaffold 证据，但不会将其当成可靠结果。

## 9. 输出说明

demo 和 batch 当前默认使用：

```python
prepare_json_export(to_dict(features))
```

默认 JSON 主要顶层字段为：

| 字段 | 内容 |
|---|---|
| `quality` | 每导联质量与记录级门控 |
| `beats` | R 峰、RR 和心拍分组 |
| `groups` | 形态组和组级测量 |
| `representative_leads` | 每导联代表性特征 |
| `global_features` | HR、PR、QRS、QT、QTc 和心电轴 |
| `rhythm_inputs` | 起搏、P 事件、AF/AFL 和可用性 |
| `morphology_inputs` | P/QRS/ST/T 形态和派生事实 |
| `interpretation` | 早期 DXL-inspired 参考解释 |
| `clinical_interpretation` | 当前权威临床规则结果 |
| `statement_engine` | 诊断语句、抑制关系和证据 |
| `metadata` | 检测器、共识、修复和来源信息 |

### 解释层的使用原则

项目当前保留权威临床解释和早期参考解释。下游程序应优先使用：

```text
clinical_interpretation
```

`interpretation` 是早期 DXL-inspired 参考证据，不能与权威结果无条件合并。

### Structured v3

`ecgfeat.export.build_structured_payload(features)` 会生成包含 `signal / quality / beats / groups / global / provenance` 等稳定分区的 structured payload。

当前 schema 版本为 `ecgfeat_structured_payload.v3`。v3 已移除旧的 Glasgow 解释区块。

它与 demo 默认的平铺 JSON 不是同一个 schema。若校验器报告缺少 `signal`、`global` 或 `provenance`，应先确认调用方期待的是哪一种导出契约。

## 10. 验证

运行测试：

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pytest -q tests
```

截至 2026-07-26，本地选取 QRS、波界定位、QT、质量、房性节律、起搏、导出和临床规则等核心测试运行，结果为：

```text
385 passed
```

现有 LUDB 200 条历史对比快照的大致平均绝对误差：

| 指标 | MAE |
|---|---:|
| HR | 0.66 bpm |
| PR | 9.26 ms |
| QRS | 9.88 ms |
| QT | 9.99 ms |
| QTc Bazett | 11.36 ms |
| QTc Fridericia | 10.78 ms |
| P axis | 14.68° |
| QRS axis | 10.09° |
| T axis | 10.85° |
| QT dispersion | 28.37 ms |

该快照早于部分最新代码修改，只能作为历史工程参考，不能代替当前版本的重新验证，也不能证明临床诊断性能。

## 11. 已知限制

1. 当前系统是研究级实现，不是经过监管验证的医疗器械。
2. AF/AFL、复杂起搏、低振幅 P 波和严重噪声仍需要更大规模外部验证。
3. `acute_occlusion_pattern` 更接近相邻导联 ST 抬高筛查，不能独立诊断急性冠脉闭塞。
4. QT dispersion 的历史误差明显高于单纯 QT。
5. 传统解释和权威临床规则可能产生不同结论。
6. 不同儿科规则中同时存在 `<16 岁` 和 `<18 岁` 的适用边界，16–18 岁结果需要特别审查。
7. demo 对增益和标准导联顺序存在默认假设。
8. 波界定位包含多层 rescue/fallback，可能出现上游错误向后传播。
9. 部分代码仍使用已弃用的 `numpy.trapz`，测试时会产生 warning。

## 12. 常见问题

### `ModuleNotFoundError`

先确认使用了正确的虚拟环境：

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pip install -e ./feature_extraction
python -m pip install matplotlib
```

根据具体任务再安装缺失组件：

```bash
python -m pip install wfdb pandas gradio
```

### 处理 `.pt` 时提示缺少 torch

PyTorch 没有包含在核心 feature extraction 依赖中。请按照机器的 CPU 或 CUDA 环境安装对应版本。

### 没有 GPU

ECG feature extraction 本身是 NumPy/SciPy CPU 算法，不要求 GPU。GPU 主要用于后续 MedGemma 推理。

### 图片生成失败

```bash
python -m pip install matplotlib
export MPLBACKEND=Agg
```

### JSON 文件太大

不要使用 `--full-beats` 或 `--include-beat-features`。默认汇总输出仍保留代表导联、全局测量和临床解释。

### Schema 校验失败

确认校验器期望的是：

- demo/batch 默认的 `to_dict()` 平铺结构；还是
- `build_structured_payload()` 生成的 structured v2。

这两种输出不能使用同一套顶层字段要求。

## 13. 建议的后续工作

1. 统一默认 JSON schema，并明确版本和权威解释字段。
2. 为导联名称、单位、增益、NaN、采样率和记录长度建立严格输入契约。
3. 使用患者级独立测试集验证间期误差、节律分类和临床规则。
4. 加强 AF/AFL、P 轴、起搏及 ST 抬高混杂场景。
5. 统一儿科年龄路由。
6. 将超大的 `api.py`、`delineate.py` 和 `features.py` 拆分成清晰、可测试的阶段。
7. 集中管理算法阈值并为阈值配置增加版本号。
8. 为 demo、batch、legacy JSON 和 structured v2 分别增加契约测试。

## License

在将本项目代码或输出用于外部分发、研究发表或商业用途前，请补充并确认项目许可证、数据集许可证和模型许可证。
