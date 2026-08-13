<!-- i18n-nav -->
[中文](ludb_detector_comparison.md) | [English](ludb_detector_comparison.en.md) | [日本語](ludb_detector_comparison.ja.md)
<!-- /i18n-nav -->

# LUDB 三种 ECG 检测方法对比

本工具在 LUDB 1.0.1 专家标注上比较以下方法：

1. 项目算法 `ecgfeat`；
2. NeuroKit2；
3. BioSPPy。

入口脚本是项目根目录下的 `compare_ludb_detectors.py`。

## 1. 安装

```bash
cd /workspace/ecg_gemma

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ludb-compare.txt
```

默认数据目录为：

```text
/workspace/ecg_gemma/data/lobachevsky-university-electrocardiography-database-1.0.1
```

脚本既接受上述 LUDB 根目录，也接受其内部的 `data/` 目录。

## 2. 运行命令

先在第 1 条记录、II 导联上做快速验证：

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate

python compare_ludb_detectors.py \
  --records 1 \
  --leads II \
  --out-dir ludb_three_way_smoke
```

运行前 10 条记录、全部 12 导联：

```bash
python compare_ludb_detectors.py \
  --limit 10 \
  --workers 2 \
  --out-dir ludb_three_way_first10
```

运行完整 LUDB：

```bash
python compare_ludb_detectors.py \
  --dataset-dir /workspace/ecg_gemma/data/lobachevsky-university-electrocardiography-database-1.0.1 \
  --workers 4 \
  --out-dir /workspace/ecg_gemma/ludb_three_way_comparison
```

使用每个导联局部细化后的 ecgfeat QRS fiducial 进行匹配：

```bash
python compare_ludb_detectors.py \
  --limit 10 \
  --workers 4 \
  --ecgfeat-r-source lead-fiducial \
  --out-dir ludb_three_way_first10_per_lead_r
```

评价脚本默认仍为 `--ecgfeat-r-source global`，用于保持历史结果可复现；
逐导联结果必须显式选择 `lead-fiducial`。两种运行的 `summary.json` 会
记录实际峰值来源。

如需最稳定的计时结果，使用 `--workers 1`。并行运行适合更快地得到检测指标，但进程竞争会影响耗时。

只比较部分方法或导联：

```bash
python compare_ludb_detectors.py \
  --methods ecgfeat,neurokit2 \
  --leads II,V1,V5 \
  --limit 10
```

查看所有参数：

```bash
python compare_ludb_detectors.py --help
```

## 3. 对比范围

LUDB 的 P 波、QRS 波群和 T 波标注是逐导联给出的，因此：

- NeuroKit2 在每个选定导联上独立运行 R 峰检测和波形界定；
- BioSPPy 在每个选定导联上独立运行；
- `ecgfeat` 保留原生设计：对一条记录的 12 导联联合检测全局 R 峰，再读取各导联的 P/QRS/T 边界。

三种方法的公共能力不是完全相同：

| 方法 | R 峰 | QRS 起止点 | P 波峰/起止点 | T 波峰/起止点 |
|---|---:|---:|---:|---:|
| `ecgfeat` | 是 | 是 | 是 | 是 |
| NeuroKit2 | 是 | 是 | 是 | 是 |
| BioSPPy 公共 `ecg()` 流程 | 是 | 否 | 否 | 否 |

因此，BioSPPy 只参与 QRS/R 峰的检测与定位比较。脚本不会推测或伪造 BioSPPy 不提供的 P/T 波及边界。

## 4. 匹配和指标定义

每个导联分别读取 LUDB 的 P、QRS、T 三元组：

```text
( onset, peak symbol, ) offset
```

默认匹配容差：

- QRS 的 R 峰：75 ms；
- P/T 波峰：150 ms。

可使用以下参数修改：

```bash
--r-tolerance-ms 75
--wave-tolerance-ms 150
```

事件按时间顺序进行一对一匹配。动态规划先最大化 TP 数，再在 TP 相同的方案中最小化峰值绝对误差，避免一个检测结果被重复匹配，也避免简单贪心在相邻事件间漏配。

检测指标按所有“记录 × 导联”的波事件进行微平均：

- `sensitivity = TP / (TP + FN)`；
- `precision`（PPV）`= TP / (TP + FP)`；
- `F1 = 2TP / (2TP + FP + FN)`。

只有波峰匹配成功后才比较该波的 onset、peak、offset。定位误差定义为：

```text
error_ms = (detected_sample - ground_truth_sample) / fs × 1000
```

每个边界输出：

- `bias_ms`：有符号平均误差；
- `sd_ms`：误差标准差；
- `mae_ms`：平均绝对误差；
- `median_ae_ms`：绝对误差中位数；
- `p95_ae_ms`：绝对误差第 95 百分位数；
- `n`：可用于该边界统计的匹配数。

LUDB 两端可能存在没有专家标注的检测事件。这些事件仍按标准定义计为 FP；如需忽略记录边缘，应该在预先定义并记录规则后修改评估策略，不能在看到结果后人工删除。

### 4.1 `--restrict-to-annotated-span`（可选，默认关闭）

LUDB 只标注每条 10 秒记录中间的一段（例如记录 1 的标注区间是 1.32 s–7.94 s，
记录 2 是 1.14 s–8.44 s）。区间以外的心搏是**真实心搏但没有参考标注**，
按默认口径全部计入 FP —— 这测量的是标注覆盖率，不是检测器精度,
并且**惩罚所有方法做对的事**。实测中三种方法的 PPV 都被压到 0.63–0.84,
QRS 的 GT 为 21843 而各方法检出 25400–25920，差额基本就是未标注的边缘心搏。

启用该开关后，评分只在每个导联的标注区间内进行。判据是机械的、无需主观取舍的：

> 丢弃距离标注区间超过**匹配容差**的检测事件。这类事件在定义上不可能与任何
> 真值事件配对，因此它被判为 FP 只是因为那里没有标注；容差以内的检测**全部保留**。

该规则对三种方法完全一致地施加，且**不改变** sensitivity 与边界误差
（真值事件集合没有变，匹配对也没有变）—— 只有 precision 与 F1 变得可解释。
`detection_rows.csv` 中的 `det_outside_annotated_span` 列记录每次丢弃的事件数。

**记录一个诚实性说明**：本节上方的原则要求评估规则「预先定义」。该开关是在第一次
全量运行（200 条记录）看到 PPV 异常低之后才补充实现的，不满足严格的预注册顺序。
因此它保持默认关闭，两套口径的结果都应一并报告，不能只报告开启后的数字。
判据本身不含任何按方法调节的自由度，这是它仍然可用的理由，但读者应知道它的来历。

## 5. 输出文件

每次运行会在 `--out-dir` 中生成：

| 文件 | 内容 |
|---|---|
| `summary.csv` | 按方法、波形汇总的微平均指标 |
| `summary_by_lead.csv` | 按方法、波形、导联汇总 |
| `record_summary.csv` | 每条记录的汇总 |
| `detection_rows.csv` | 每个记录、导联、方法、波形的 TP/FP/FN |
| `matched_events.csv` | 每个成功匹配事件的样本位置和边界误差 |
| `runtime.csv` | 每次算法调用耗时 |
| `runtime_summary.csv` | 耗时汇总 |
| `failures.csv` | 算法异常及 traceback |
| `summary.json` | 配置、版本、指标和失败数 |
| `comparison.png` | F1 和定位 MAE 对比图 |

`failures.csv` 为空表示没有调用失败。默认模式下，某个算法调用失败时，该调用支持的所有真实事件按 FN 计入汇总，避免通过静默丢弃失败样本得到虚高指标；同时脚本最终返回非零退出码。使用 `--fail-fast` 可以在第一次失败时直接停止。

## 6. 运行时间的正确理解

`ecgfeat` 一次处理完整 12 导联记录，时间单位为 `record_12lead`；NeuroKit2 和 BioSPPy 按单导联运行，时间单位为 `single_lead`。`runtime_summary.csv` 会显式写出此单位。

所以不能直接把三者的 `mean_seconds` 当成同一工作量比较。若要估计外部库处理一条 12 导联记录的串行耗时，可按同一记录的 12 个导联调用时间求和；若要做严格性能基准，应使用单进程、预热、固定硬件并重复运行。

## 7. 在同一条 ECG 上画 ecgfeat 与 NeuroKit2 标注

`plot_ecgfeat_neurokit_comparison.py` 会让两个检测器读取同一条 LUDB
记录，并在相同的原始 ECG 样本和时间轴上显示结果。

绘制完整记录：

```bash
python plot_ecgfeat_neurokit_comparison.py 1 \
  --lead II \
  --out ludb_ecgfeat_neurokit_plots/record_1_II_full.png
```

按 LUDB 的第 N 个 QRS 标注放大单拍：

```bash
python plot_ecgfeat_neurokit_comparison.py 8 \
  --lead V2 \
  --beat 3 \
  --out ludb_ecgfeat_neurokit_plots/record_8_V2_beat3.png
```

显示第 2–5 个 QRS 对应的 4 个连续心搏：

```bash
python plot_ecgfeat_neurokit_comparison.py 8 \
  --lead V2 \
  --beat-range 2 5 \
  --out ludb_ecgfeat_neurokit_plots/record_8_V2_beats2-5.png
```

`--before-sec` 和 `--after-sec` 分别控制第一个 QRS 之前和最后一个 QRS
之后保留的上下文，默认是 0.45 秒和 0.70 秒。

选择固定时间段：

```bash
python plot_ecgfeat_neurokit_comparison.py 1 \
  --lead II \
  --start-sec 1.5 \
  --end-sec 4.5
```

不显示 LUDB 专家参照：

```bash
python plot_ecgfeat_neurokit_comparison.py 1 --lead II --no-ludb
```

输出图各行含义：

1. 第一行在同一条 ECG 上叠加两种算法的 P/R/T 峰；实心标记是
   `ecgfeat`，空心标记是 NeuroKit2，叉号是 LUDB；
2. 第二行显示 `ecgfeat` 的 P/QRS/T 区间和边界；
3. 第三行显示 NeuroKit2 的 P/QRS/T 区间和边界；
4. 默认第四行显示 LUDB 专家边界，使用 `--no-ludb` 时省略。

边界面板中，点线是 onset，虚线是 offset，半透明区域是
onset–offset 区间。所有面板绘制的是相同原始 ECG；两个检测器仍分别
使用各自的内部预处理。终端还会打印当前可视时间窗内相对 LUDB 的
TP、FP、FN、F1 和峰值 MAE。

### 完整 12 导联、10 秒对比

`plot_ludb_12lead_10s_comparison.py` 会一次运行整条 12 导联记录，并把
每个导联完整的 0–10 秒信号排成 12 行。项目算法只对记录运行一次，
心搏的存在和数量由多导联检测决定；图中的 `ecgfeat` QRS 标记默认
使用独立保存的逐导联 `hybrid-prominence` 定位。NeuroKit2 则逐导联
独立检测。

```bash
# 单条记录
python plot_ludb_12lead_10s_comparison.py 1

# LUDB 前 10 条记录
python plot_ludb_12lead_10s_comparison.py {1..10} \
  --out-dir ludb_12lead_10s_comparison
```

`--ecgfeat-r-source` 可以选择四种标记：

- `hybrid-prominence`：以多导联心搏为基础，在原逐导联 fiducial
  周围 35 ms 内选择 prominence 最大的正向局部峰，绘图默认值；
- `lead-fiducial`：每个导联原有的 QRS fiducial；
- `positive-r`：每个导联 QRS 区间内的最大正向 R 波；
- `global`：所有导联共享的多导联全局 R fiducial，也是检测评价脚本
  为保持历史指标一致而继续使用的默认值。

`hybrid-prominence` 写入独立的 `r_localized_index` 等字段，不覆盖
`r_locs`、`qrs.peak`、`r_peak_index` 或波形边界，因此不会改变心率、
间期、代表导联特征和解释层结果。

每条记录默认生成：

1. 三方 P/R/T 峰值叠加页；
2. `ecgfeat` 的 P/QRS/T 边界页；
3. NeuroKit2 的 P/QRS/T 边界页；
4. LUDB 专家 P/QRS/T 边界页；
5. 汇集以上四页的 PDF。

如不需要 LUDB 专家页或 PDF，可分别使用 `--no-ludb`、`--no-pdf`。

## 8. 当前实现的边界

- 比较的是检测和波形界定，不是最终临床诊断规则准确率；
- `ecgfeat` 的 R 峰是多导联共识结果，而外部库是单导联结果，这反映各自原生使用方式，但并非完全相同的输入任务；
- P/T 匹配以波峰为锚点，边界指标只统计已经匹配的波；
- 默认 NeuroKit2 使用 `neurokit` R 峰方法和 `dwt` 界定方法，可通过命令行切换；
- 对不同容差、导联子集或 NeuroKit2 方法得到的结果，必须保留对应的 `summary.json`，不要混合比较。
