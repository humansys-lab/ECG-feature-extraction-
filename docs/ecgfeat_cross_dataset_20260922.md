# ECG feature extraction：新增数据集冻结评估（2026-09-22）

本轮只新增评估工具，没有调整生产算法、阈值或默认开关，也没有使用深度学习。生产代码与实际评估副本逐文件 SHA-256 一致；所有测试对象、窗口与容差在看结果之前固定。

主要结论：QRS 增强在 INCART 上只提高约 0.03 个 F1 百分点，在 SVDB 上下降约 0.09 个百分点，按来源分组的统计区间均跨过零，尚不能证明普遍收益。PWAVE 的 P_native F1 小幅提高，但 ±50 ms 峰位匹配 F1 仍只有 69.12%，门控后 P 波召回率仅 47.94%。下一轮应优先验证 P 峰定位、QRS 质量筛选后的峰位稳定性，以及 P 波接受门控的召回。

## 数据与覆盖

| 数据 | 记录 / 人群 | 原始总时长 | 运行范围 | 没有已知旧来源重叠的记录 |
|---|---|---:|---|---:|
| INCART | 75 条 / 32 位患者 | 37.500 h | 全时长 QRS 检测器；每条首/中/尾各 30 s 完整提取 | 75 |
| SVDB | 78 条，按来源记录统计 | 39.000 h | 全时长 QRS 检测器；每条首/中/尾各 30 s 完整提取 | 61 |
| PWAVE | 12 条，按来源记录统计 | 6.019 h | 全时长完整提取，评分 P 和 QRS | 1 |

- [INCART 1.0.0](https://physionet.org/content/incartdb/1.0.0/)：12 导联，257 Hz；参考点通常是 QRS 中部，位置未逐一人工校正。峰值 MAE 受这一标注定义影响。
- [SVDB 1.0.0](https://physionet.org/content/svdb/1.0.0/)：78 条半小时记录，实际文件为 2 通道、128 Hz。与 BUT、QTDB 有来源重叠，按整条来源记录排除后另报。
- [PWAVE 1.0.0](https://physionet.org/content/pwave/1.0.0/)：12 条 MIT-BIH 记录，360 Hz；两位专家制作/复核 P 波峰标注。官方说明标注不保证穷尽，未匹配预测不能一律认定为生理上不存在的 P 波。QRS 参考来自原 [MIT-BIH 1.0.0](https://physionet.org/content/mitdb/1.0.0/)。
- PWAVE 只有记录 122 未出现在此前 BUT、QTDB 或 NSTDB 的来源记录中。新增的 P 波标注测试与新患者泛化是不同证据。
- PWAVE 原始 22,108 个 P 标记包含 78 个同位置同符号重复项：119 中 77 个、214 中 1 个；去重后真值为 22,030。初次 4 项任务被严格标注检查拦截，修正评分器后补跑；其他 20 项复用原预测。对最终 24 项用修正后的评分器逐项重算，216 行评分及 336,147 个匹配与保存结果完全一致。没有改动提取算法。审计见 `pwave_duplicate_annotation_audit.json` 与 `pwave/composition_manifest.json`。
- 三组数据均无本轮可用的人工 P/T 起止点真值，不能据此宣称 P 波宽度、T 终点或 QT 测量精度改善。

## 配置和评分

`default`：当前代码，全部 refinement 开关关闭。`enhanced`：开启 `qrs_adaptive_consensus`、`p_pathology_candidates`、`atrial_event_validation`、`t_sequence_offset_only`。单独 QRS 检测器只应用对应的 QRS 开关。

INCART 使用标准 12 导联输入；SVDB/PWAVE 使用 limited 模式与原始通道名称。WFDB 增益/基线换算后的 mV 直接输入；只规范 AVR/AVL/AVF 名称大小写。内部统一 500 Hz，INCART 50 Hz 陷波，其余 60 Hz。每 30 s 核心区前后各带 2 s 上下文；固定抽样窗口分别为首 30 s、中间 30 s、末 30 s，每条只覆盖 90 s，不能写成全时长完整提取。

检测点换回原始采样轴，以核心区归属去重；全记录进行一对一匹配，优先最大匹配数、再最小时间误差。不利用标注选导联、选窗口、移动时间轴或调参。QRS 主容差 75 ms，补充 50/150 ms；P 主容差 150 ms，补充 50 ms。容差按原始采样率取整。异常窗口不产生预测，真值仍保留为漏检。

Se=TP/(TP+FN)，PPV=TP/(TP+FP)，F1=2TP/(2TP+FP+FN)。MAE 只针对已匹配事件，必须结合漏检率看。

## 主结果

| 数据 / 范围 | 输出 / 容差 | 配置 | TP / FP / FN | Se % | PPV % | F1 % | 峰值 MAE ms |
|---|---|---|---:|---:|---:|---:|---:|
| INCART / 全时长 QRS 检测器 | QRS / 75ms | default | 171179 / 2250 / 4721 | 97.32 | 98.70 | 98.00 | 9.06 |
| INCART / 全时长 QRS 检测器 | QRS / 75ms | enhanced | 171236 / 2194 / 4664 | 97.35 | 98.73 | 98.04 | 9.51 |
| INCART / 固定片段完整提取 | QRS / 75ms | default | 8347 / 131 / 255 | 97.04 | 98.45 | 97.74 | 8.86 |
| INCART / 固定片段完整提取 | QRS / 75ms | enhanced | 8357 / 133 / 245 | 97.15 | 98.43 | 97.79 | 9.38 |
| SVDB / 全时长 QRS 检测器 | QRS / 75ms | default | 180024 / 1915 / 4558 | 97.53 | 98.95 | 98.23 | 20.07 |
| SVDB / 全时长 QRS 检测器 | QRS / 75ms | enhanced | 179893 / 2128 / 4689 | 97.46 | 98.83 | 98.14 | 20.15 |
| SVDB / 固定片段完整提取 | QRS / 75ms | default | 8949 / 95 / 286 | 96.90 | 98.95 | 97.92 | 19.64 |
| SVDB / 固定片段完整提取 | QRS / 75ms | enhanced | 8957 / 96 / 278 | 96.99 | 98.94 | 97.95 | 19.81 |
| PWAVE / 全时长完整提取 | P_native / 150ms | default | 20247 / 3192 / 1783 | 91.91 | 86.38 | 89.06 | 28.61 |
| PWAVE / 全时长完整提取 | QRS / 75ms | default | 24706 / 371 / 322 | 98.71 | 98.52 | 98.62 | 3.50 |
| PWAVE / 全时长完整提取 | P_native / 150ms | enhanced | 20384 / 3185 / 1646 | 92.53 | 86.49 | 89.41 | 28.10 |
| PWAVE / 全时长完整提取 | QRS / 75ms | enhanced | 24728 / 274 / 300 | 98.80 | 98.90 | 98.85 | 3.53 |

![结果对比](../ecgfeat_cross_dataset_20260922/validation_results.png)

QRS 检测器全时长结果不含后续起搏处理、补检与描记；完整提取结果取最终输出的心搏，二者分开报告。

## 更严格定位与宽容差复核

| 数据 / 范围 | 输出 / 容差 | 配置 | TP / FP / FN | Se % | PPV % | F1 % | 峰值 MAE ms |
|---|---|---|---:|---:|---:|---:|---:|
| INCART / 全时长 QRS 检测器 | QRS / 150ms | default | 173138 / 291 / 2762 | 98.43 | 99.83 | 99.13 | 9.92 |
| INCART / 全时长 QRS 检测器 | QRS / 50ms | default | 166840 / 6589 / 9060 | 94.85 | 96.20 | 95.52 | 7.70 |
| INCART / 全时长 QRS 检测器 | QRS / 150ms | enhanced | 173292 / 138 / 2608 | 98.52 | 99.92 | 99.21 | 10.41 |
| INCART / 全时长 QRS 检测器 | QRS / 50ms | enhanced | 167017 / 6413 / 8883 | 94.95 | 96.30 | 95.62 | 8.18 |
| SVDB / 全时长 QRS 检测器 | QRS / 150ms | default | 181763 / 176 / 2819 | 98.47 | 99.90 | 99.18 | 20.77 |
| SVDB / 全时长 QRS 检测器 | QRS / 50ms | default | 172055 / 9884 / 12527 | 93.21 | 94.57 | 93.89 | 18.09 |
| SVDB / 全时长 QRS 检测器 | QRS / 150ms | enhanced | 181877 / 144 / 2705 | 98.53 | 99.92 | 99.22 | 20.96 |
| SVDB / 全时长 QRS 检测器 | QRS / 50ms | enhanced | 172007 / 10014 / 12575 | 93.19 | 94.50 | 93.84 | 18.20 |
| PWAVE / 全时长完整提取 | P_native / 50ms | default | 15550 / 7889 / 6480 | 70.59 | 66.34 | 68.40 | 9.95 |
| PWAVE / 全时长完整提取 | P_native / 50ms | enhanced | 15758 / 7811 / 6272 | 71.53 | 66.86 | 69.12 | 9.79 |

## 排除既往来源记录

| 数据 / 范围 | 输出 / 容差 | 配置 | TP / FP / FN | Se % | PPV % | F1 % | 峰值 MAE ms |
|---|---|---|---:|---:|---:|---:|---:|
| SVDB / 全时长 QRS 检测器 | QRS / 75ms | default | 144902 / 1751 / 3620 | 97.56 | 98.81 | 98.18 | 20.40 |
| SVDB / 全时长 QRS 检测器 | QRS / 75ms | enhanced | 144686 / 1951 / 3836 | 97.42 | 98.67 | 98.04 | 20.51 |
| SVDB / 固定片段完整提取 | QRS / 75ms | default | 7214 / 84 / 218 | 97.07 | 98.85 | 97.95 | 19.87 |
| SVDB / 固定片段完整提取 | QRS / 75ms | enhanced | 7222 / 85 / 210 | 97.17 | 98.84 | 98.00 | 20.08 |
| PWAVE / 全时长完整提取 | P_native / 150ms | default | 2249 / 105 / 226 | 90.87 | 95.54 | 93.15 | 22.29 |
| PWAVE / 全时长完整提取 | QRS / 75ms | default | 2476 / 0 / 0 | 100.00 | 100.00 | 100.00 | 2.63 |
| PWAVE / 全时长完整提取 | P_native / 150ms | enhanced | 2270 / 104 / 205 | 91.72 | 95.62 | 93.63 | 21.79 |
| PWAVE / 全时长完整提取 | QRS / 75ms | enhanced | 2476 / 0 / 0 | 100.00 | 100.00 | 100.00 | 2.63 |

完整排除名单保存在 `analysis.json → cohorts`。这是已知来源去重，无法排除数据发布方未提供的跨库身份关系。

## 参考心搏类型的检出率

按 ±75 ms 与参考点匹配统计 QRS 检出：N=正常心搏，A/S=参考标记的房性/室上性早搏，V=室早。这里没有评估检测器能否正确分类心搏。其余类型保存在 JSON 和 CSV。

| 数据 / 参考类型 | 心搏数 | 默认 Se % | 增强 Se % |
|---|---:|---:|---:|
| INCART / N | 150406 | 98.51 | 98.60 |
| INCART / A | 1944 | 99.64 | 99.49 |
| INCART / S | 16 | 100.00 | 100.00 |
| INCART / V | 20011 | 89.96 | 89.57 |
| SVDB / N | 162338 | 98.21 | 98.13 |
| SVDB / S | 12188 | 96.99 | 97.14 |
| SVDB / V | 9943 | 87.36 | 87.07 |
| PWAVE / N | 17646 | 99.31 | 99.43 |
| PWAVE / A | 427 | 98.36 | 98.59 |
| PWAVE / V | 1801 | 90.51 | 90.28 |

室早的增强匹配召回率在 ±75/150 ms 下分别为：INCART 89.57%/96.85%、SVDB 87.07%/92.87%、PWAVE 90.28%/92.56%。因此这些错误同时包含峰位偏差和未匹配心搏；尤其 INCART 的参考是 QRS 中部，不能将全部匹配 FN 等同于没有发现 QRS。

## P 波候选、接受结果与房性事件

`P_native` 是首个实测通道的描记 P 波峰；`P_accepted` 还要求当前 P 波质量评估接受；`P_atrial` 是房性事件候选。三者目标不同，不能用候选召回率代替最终接受率。

| 数据 / 范围 | 输出 / 容差 | 配置 | TP / FP / FN | Se % | PPV % | F1 % | 峰值 MAE ms |
|---|---|---|---:|---:|---:|---:|---:|
| PWAVE / 全时长完整提取 | P_accepted / 150ms | default | 10541 / 845 / 11489 | 47.85 | 92.58 | 63.09 | 26.38 |
| PWAVE / 全时长完整提取 | P_atrial / 150ms | default | 21212 / 13879 / 818 | 96.29 | 60.45 | 74.27 | 27.09 |
| PWAVE / 全时长完整提取 | P_accepted / 150ms | enhanced | 10561 / 834 / 11469 | 47.94 | 92.68 | 63.19 | 26.26 |
| PWAVE / 全时长完整提取 | P_atrial / 150ms | enhanced | 21245 / 13170 / 785 | 96.44 | 61.73 | 75.28 | 26.61 |

PWAVE 103 的全时长增强 P_native F1 在 ±150 ms 时为 98.94%，±50 ms 时为 53.86%。原始 MLII 波形的事后核查显示，开始几拍的预测峰落在参考 P 波峰后约 75–108 ms 的位置，这不是统一移动参考轴得到的结果。本轮保持算法不变，该例用于定位下一轮改进问题。

![PWAVE 103 峰位复核](../ecgfeat_cross_dataset_20260922/pwave_103_peak_audit.png)

## 改善是否稳定

对相同来源成对比较 F1，固定种子 20260922、5000 次簇 bootstrap。INCART 按 32 位患者重采样；SVDB/PWAVE 按来源记录重采样。区间只描述本数据集内的抽样不确定性，不能消除旧数据来源重叠。

| 数据 / 范围 / 输出 | ΔF1 百分点 | 95% 区间 | 簇数 |
|---|---:|---:|---:|
| INCART / 全时长 QRS 检测器 / QRS | +0.032 | [-0.225, +0.328] | 32 |
| INCART / 固定片段完整提取 / QRS | +0.048 | [-0.313, +0.410] | 32 |
| PWAVE / 全时长完整提取 / P_native | +0.347 | [+0.038, +0.655] | 12 |
| PWAVE / 全时长完整提取 / QRS | +0.236 | [+0.012, +0.523] | 12 |
| SVDB / 全时长 QRS 检测器 / QRS | -0.093 | [-0.388, +0.084] | 78 |
| SVDB / 固定片段完整提取 / QRS | +0.039 | [-0.023, +0.150] | 78 |

## 最明显的退化记录

| 数据 / 范围 / 记录 | 输出 | 默认 F1 % | 增强 F1 % | Δ 百分点 |
|---|---|---:|---:|---:|
| SVDB / 全时长 QRS 检测器 / 862 | QRS | 99.18 | 88.42 | -10.755 |
| INCART / 固定片段完整提取 / I18 | QRS | 99.66 | 94.24 | -5.424 |
| INCART / 固定片段完整提取 / I31 | QRS | 96.97 | 92.25 | -4.716 |
| INCART / 全时长 QRS 检测器 / I18 | QRS | 99.23 | 97.24 | -1.995 |
| INCART / 固定片段完整提取 / I42 | QRS | 98.64 | 96.89 | -1.754 |
| INCART / 全时长 QRS 检测器 / I29 | QRS | 98.90 | 97.36 | -1.547 |
| INCART / 全时长 QRS 检测器 / I31 | QRS | 96.13 | 94.74 | -1.391 |
| INCART / 全时长 QRS 检测器 / I65 | QRS | 99.42 | 98.21 | -1.212 |
| INCART / 全时长 QRS 检测器 / I30 | QRS | 99.22 | 98.05 | -1.171 |
| INCART / 全时长 QRS 检测器 / I08 | QRS | 98.66 | 97.62 | -1.032 |

SVDB 862 的退化主要表现为定位偏移：增强配置在 ±75 ms 下为 TP/FP/FN=1928/245/260，放宽至 ±150 ms 后为 2173/0/15；默认分别为 2164/12/24 和 2176/0/12。其中 245 个增强点在宽容差下仍能匹配。对 60–90 s 核心窗口的复现与原保存预测逐点一致：ECG1/ECG2 的 bSQI 分别为 1.0/0.738，增强路径只使用 ECG1，默认使用两通道。图示和检测窗审计指向质量筛选后能量峰/局部搜索的落点稳定性问题，而总体匹配分数本身不能直接说明真实漏拍数量。审计保存于 `svdb/862_timing_audit.json`；本轮未据此修改参数。

![SVDB 862 定位退化复核](../ecgfeat_cross_dataset_20260922/svdb_862_timing_audit.png)

逐记录全部结果与参考心搏类型分层均保留。类型分层统计的是已知类型心搏的检出率，不是心律分类准确率。

## 运行速度

完整队列的并行作业耗时保存在各目录 `execution.json`；进程竞争下的时延不能作为纯算法加速比。下面另用单进程、数值库单线程：每库固定首/末记录前 32 s，预热一次、计时三次、交替配置顺序。排除文件读取与评分，包含预处理、质量评估、检测、描记及其余完整提取步骤。重复运行的 QRS/P 输出哈希一致。

| 数据 / 记录 | 配置 | 32 s 信号耗时中位数 | 耗时 / 信号时长 |
|---|---|---:|---:|
| INCART / I01 | default | 10.063 s | 0.3145 |
| INCART / I01 | enhanced | 10.847 s | 0.3390 |
| INCART / I75 | default | 7.538 s | 0.2355 |
| INCART / I75 | enhanced | 8.006 s | 0.2502 |
| SVDB / 800 | default | 2.670 s | 0.0834 |
| SVDB / 800 | enhanced | 2.730 s | 0.0853 |
| SVDB / 894 | default | 3.234 s | 0.1011 |
| SVDB / 894 | enhanced | 3.361 s | 0.1050 |
| PWAVE / 100 | default | 3.534 s | 0.1104 |
| PWAVE / 100 | enhanced | 3.633 s | 0.1135 |
| PWAVE / 231 | default | 3.042 s | 0.0951 |
| PWAVE / 231 | enhanced | 3.128 s | 0.0978 |

另在每库首记录对增强配置做 cProfile，以下列出累计耗时较高的内部函数。插桩会增加耗时；累计时间存在嵌套，不能相加，也不能替代上面的独立计时。

| 数据 | 函数 | 调用数 | 累计耗时 s |
|---|---|---:|---:|
| INCART | `delineate.py:delineate_beats` | 2 | 6.356 |
| INCART | `p_wave_engine.py:build_p_wave_assessments` | 1 | 5.734 |
| INCART | `p_wave_engine.py:assess_window` | 50 | 5.668 |
| INCART | `p_wave_engine.py:_lead_boundary` | 600 | 4.089 |
| INCART | `p_wave_engine.py:_local_boundary_estimate` | 600 | 3.983 |
| SVDB | `p_wave_engine.py:build_p_wave_assessments` | 1 | 1.244 |
| SVDB | `p_wave_engine.py:assess_window` | 29 | 1.200 |
| SVDB | `delineate.py:delineate_beats` | 2 | 1.165 |
| SVDB | `p_wave_engine.py:_constrained_t_reconstruction` | 29 | 0.725 |
| SVDB | `p_wave_engine.py:_robust_affine_errors_multilead` | 420 | 0.534 |
| PWAVE | `delineate.py:delineate_beats` | 2 | 1.716 |
| PWAVE | `p_wave_engine.py:build_p_wave_assessments` | 1 | 1.631 |
| PWAVE | `p_wave_engine.py:assess_window` | 39 | 1.581 |
| PWAVE | `p_wave_engine.py:_constrained_t_reconstruction` | 39 | 1.065 |
| PWAVE | `p_wave_engine.py:_robust_affine_errors_multilead` | 570 | 0.710 |

这是固定窗口的离线吞吐测试，带双向滤波和上下文，不能直接解释为在线设备的实时延迟承诺。

## 复现与完整性

共 636 个记录/范围/配置任务、20742 次窗口处理，失败任务 0，失败窗口 0。生产文件未变化：True；运行副本匹配冻结哈希：True。

INCART 初始进程在 298 项完成后收到 SIGTERM；最后 2 项使用归档的原始评估脚本断点续跑。续跑前核对了代码、原始数据、标注来源及现有检查点哈希，保持原 fingerprint，未跳过未完成记录。

```bash
.venv/bin/python scripts/download_ecgfeat_cross_dataset.py --workers 8
# source 可指向当前仓库，严格复现本轮请使用冻结副本；使用新的 out，避免覆盖已有结果。
.venv/bin/python scripts/evaluate_ecgfeat_cross_dataset.py \
  --datasets incartdb svdb pwave \
  --source ecgfeat_cross_dataset_20260922/frozen_source \
  --out ecgfeat_cross_dataset_reproduction --workers 12
.venv/bin/python -m pytest tests/test_ecgfeat_cross_dataset_protocol.py -q
```

本轮目录：`ecgfeat_cross_dataset_20260922/`。`protocol_predeclared.json` 保存预先固定的范围；`production_source_sha256.json` 保存算法冻结哈希；每库 `manifest.json` 记录代码、数据及协议哈希；`checkpoints/` 保存逐记录预测、匹配、排除标注、窗口计时；CSV 可供复核。`analysis.json` 汇总结果与统计区间，`validation.json` 保存完整性检查。
