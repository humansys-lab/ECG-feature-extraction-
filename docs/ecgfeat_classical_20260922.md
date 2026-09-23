# ECG 传统算法续进与等价加速（2026-09-22）

本轮只使用传统信号处理、确定性动态规划和可选的数值循环编译，没有新增深度学习模型、训练、权重或推理依赖。默认路径新增 Kalman 编译加速；准确性候选独立开关、默认关闭。

## 实现范围

| 配置 | 实现及边界 |
|---|---|
| `p_phasor_candidates` | 在排除前一 T 波及当前 QRS 的搜索窗内做相量变换，以显著性筛选最多三个候选；去重后接入既有 P 上下文重选。保留导联质量、起搏和逆传 P 保护。当前仅复核已有 P，不直接以相量峰补出缺失 P，也没有复刻论文的完整病理分类器。 |
| `t_boundary_projection` | 使用 I、II、V1–V6 中至少两个有效物理导联，依据残差噪声加权，在每个波界附近的导数窗口估计投影方向。起点、终点分别拟合，要求局部方法支持才替换逐导联波界；投影不被计为额外独立导联。 |
| `t_sequence_selection` | 对已有 T 起止点候选对做动态规划，使用有上限的跨搏变化惩罚；缺失波、低信噪比、起搏、QRS 宽度和 RR 明显变化有保护。保留原始候选，不凭序列先验创建缺失波。没有训练完整 HSMM，也未联合重测 P/QRS。 |
| `t_projection_offset_only` | 上述局部投影的仅终点消融，保留原 T 起点。 |
| `t_sequence_offset_only` | 上述动态规划的仅终点消融，保留原 T 起点。 |

所有候选沿用现有测量更新函数，修改 T 终点时刷新 QT、JT、Tpeak–Tend、T 时限和面积；随后执行既有融合、ST 和心房处理。原始 mV 信号不因投影而改写。审计信息写入 `metadata.refinement.native_t_fusion` 和逐搏标志。

方法来源：[病理 P 波相量方法](https://pubmed.ncbi.nlm.nih.gov/35449228/)、[多导联小波空间投影](https://diec.unizar.es/~laguna/personal/publicaciones/MultileadDelineationECG.pdf)、[双向半马尔可夫分割](https://pubmed.ncbi.nlm.nih.gov/36130422/)。本实现是面向当前流程的适配，不应将原论文成绩当成本项目复现结果。

## 开发消融与选择过程

先冻结源码，在既有固定 49 条 LUDB 开发记录上对默认、相量 P、起止点局部投影、起止点序列选择、三者组合进行消融，无执行失败。开发数据见 [development](../ecgfeat_classical_20260922/development/manifest.json)。

起止点投影使 T onset MAE 从 38.703 升至 38.921 ms；起止点序列选择升至 38.817 ms。因此继续拆出仅终点分支，冻结后执行完整 LUDB 和 QTDB 验证，没有依据 QTDB 结果继续调阈值。

LUDB 已长期用于开发，完整 200 条结果包含开发集，不是独立留出验证；QTDB 也在此前轮次被检查过，应称跨数据集回归，不应宣称全新盲测。所有评估保存固定记录列表、参数、源码哈希、失败记录、逐事件误差和覆盖计数。

## 完整 LUDB：200 条，五种配置各完成一次

误差均为 ms；每列 MAE 以该配置有有效输出的事件为分母，配对统计单独保存。

| 配置 | P onset MAE / 有效数 | P offset MAE / 有效数 | T onset MAE | T offset MAE | QT MAE |
|---|---:|---:|---:|---:|---:|
| 默认 | 25.900 / 15,849 | 24.692 / 15,978 | 34.761 | 25.229 | 28.477 |
| 相量 P 候选 | 25.886 / 15,850 | 24.677 / 15,979 | 34.761 | 25.229 | 28.477 |
| 仅终点投影 | 25.857 / 15,841 | 24.658 / 15,970 | 34.761 | 24.818 | 28.552 |
| 仅终点序列选择 | 25.899 / 15,844 | 24.687 / 15,973 | 34.761 | 25.007 | 28.309 |
| 两项仅终点组合 | 25.847 / 15,835 | 24.655 / 15,964 | 34.761 | 24.673 | 28.429 |

五种配置均匹配 21,632 个逐导联 QRS，T offset/QT 各有 19,225 个有效误差，T onset 有 19,269 个。P 起止点的标注分母各为 16,621。所有候选的 T offset P95 仍为 100 ms、QT P95 仍为 104 ms，未解决大误差长尾。

关键判断：

- **仅终点序列选择**：T offset MAE 降低 0.223 ms，QT 降低 0.168 ms。记录平均 QT 误差差值为 −0.156 ms，按记录 bootstrap 的 95% 区间为 [−0.206, −0.108] ms；82 条记录改善、30 条变差。P 起止点各少 5 个有效评分，不能描述为全指标无损提升。
- **仅终点投影**：T offset 降低 0.411 ms，但 QT 增加 0.076 ms。QRS 起点与 T 终点误差共同决定 QT，单个波界改善并不保证整个间期改善。
- **组合**：T offset 降低 0.556 ms，但 QT 改善仅 0.048 ms，记录 bootstrap 区间跨零；P 起止点各少 14 个有效评分。不能因 T 终点 MAE 最低就默认启用组合。
- **相量 P**：覆盖各增加 1 个评分；共同事件上的 P onset/offset MAE 改善约 0.025/0.017 ms。整体收益很小，不能称为明显提高 P 检测能力。

T 终点变化会影响后续 P 波静默区和心房评估，所以 T 候选也可能改变 P 覆盖率。上表中 P 条件 MAE 的下降不能代替配对误差和覆盖率检查。置信区间未进行多候选比较校正。

净覆盖变化也会掩盖事件替换：仅终点序列选择实际丢失 9 个原有 P onset 输出、补回 4 个；相量 P 丢失 2 个、补回 3 个。逐事件清单和改善/退化记录见 [case_review.json](../ecgfeat_classical_20260922/case_review.json)。

按人工 QRS 时限分层：仅终点序列选择使宽 QRS 的 QT MAE 从 38.136 降至 38.081 ms，窄 QRS 从 27.225 降至 27.043 ms。宽 QRS 仍是明显短板。本轮没有修改 QRS 检测算法。

原始结果：[LUDB manifest](../ecgfeat_classical_20260922/ludb_full/manifest.json)、[分层结果](../ecgfeat_classical_20260922/stratified.json)。

## 跨数据集验证

QTDB 沿用 q1c、两条真实通道映射到 II/V2、其余导联置零的既定协议，固定匹配容差和标注簇上下文。此处是稀疏导联验证，不能等同于完整 12 导联验证。

五种配置各完成全部 105 条记录，无执行失败。

| 配置 | P F1 | P onset MAE/ms | P offset MAE/ms | T offset MAE/ms |
|---|---:|---:|---:|---:|
| 默认 | 92.618% | 49.189 | 41.057 | 31.240 |
| 相量 P 候选 | 92.650% | 48.664 | 40.808 | 31.240 |
| 仅终点投影 | 92.618% | 49.189 | 41.057 | 31.005 |
| 仅终点序列选择 | 92.618% | 49.189 | 41.057 | 31.206 |
| 两项仅终点组合 | 92.618% | 49.189 | 41.057 | 30.986 |

相量 P 的 TP/FP/FN 为 2,937/209/257，默认是 2,936/210/258；P 检出率从 91.922% 变为 91.954%，精确率从 93.325% 变为 93.357%。起点有效误差数由 2,896 增至 2,897，终点由 2,936 增至 2,937。共同事件上的 P onset MAE 从 49.189 降至 48.677 ms；有有效起点误差的记录中，13 条改善、0 条变差、83 条不变。记录平均误差差值 −0.518 ms，bootstrap 95% 区间 [−0.857, −0.234] ms；P offset 区间为 [−0.466, −0.061] ms。这是本轮跨库方向较一致的 P 改进，但绝对收益仍小，F1 只提高约 0.032 个百分点。

所有配置的 QRS F1 为 96.648%、T F1 为 91.651%；T 起点 MAE 均为 46.774 ms，T 终点有效评分数均为 3,314。仅终点投影、序列选择及组合的 T offset 记录平均误差差值区间均跨零，分别约 [−0.520, 0.029]、[−0.153, 0.073]、[−0.605, 0.046] ms。因此不能把小幅条件 MAE 改善表述成已证实稳定泛化。

三种 T 候选的终点 P95 还从默认的 117.4 ms 上升至 120.0 ms；平均误差下降没有改善尾部。这也是保留可选开关、不默认启用的原因。

结果见 [QTDB manifest](../ecgfeat_classical_20260922/qtdb/manifest.json) 和 [配对比较](../ecgfeat_classical_20260922/comparisons.json)。本轮未增加 BUT PDB/ISP，也未重做 EDB ST 验证或候选专属风险校准；这些限制不因跨库误差下降而消失。

## 默认等价加速

将 P 波状态空间基线中的原标量 Kalman 递推抽取到 `_kalman.py`，有 Numba 时编译执行，使用 `fastmath=False`，保持原运算顺序，并将最后的平方根仍交由 NumPy 完成。没有 Numba、禁用 JIT 或编译不可用时回退原 Python 运算；不缓存患者信号。

同时处理了安装导入 `ecgfeat._kalman` 与仓库测试导入 `feature_extraction.ecgfeat._kalman` 的磁盘缓存兼容性：只让安装名称写入可持久化编译缓存，避免安装环境加载仅在开发目录中可导入的模块。首次探测因该问题回退到 Python 的计时保留为 `benchmark_fallback_probe.json`，不作为加速结果。

最终 5 条记录、各预热一次后重复 3 次、交替 A/B 顺序：各记录提取中位耗时总和 **9.541 → 8.991 秒，减少 5.76%（1.061×）**。这次基线已包含上轮默认加速，不能直接把两个百分比相加。Numba 版本为 0.65.0；计时固定在 CPU 0/1，其他评估任务限制在其余 CPU。

- 最终默认路径通过完整 LUDB 200 条等价门：21,632 个匹配事件的既有边界误差、覆盖计数和全局字段一致（数值容差 1e-9）。
- A/B 的 5 条记录中，未舍入 summary/debug JSON 均完全相同，只排除两个生成时间戳。
- 单独的编译核测试覆盖 250/500/1000 Hz、正向/负步长、缺失观测、异常值和不同噪声方差，逐元素精确相同。
- 首次编译/加载有额外开销；5.76% 是预热后提取收益，不是单次冷启动收益。

记录 1 的单次首调用探测：旧路径 2.026 秒，编译缓存已有时 2.002 秒，空缓存需要编译时 2.388 秒；对应第二次调用为 1.796、1.689、1.672 秒。这是每种方式各一次的开销示例，不是稳健的冷启动性能估计。

数据：[等价门](../ecgfeat_classical_20260922/equivalence_release/regression_gate.json)、[顺序 A/B](../ecgfeat_classical_20260922/benchmark.json)、[首调用探测](../ecgfeat_classical_20260922/cold_start.json)。

## 测试与复现

- 提取及相关完整测试：1,209 passed、150 subtests passed；2 个缺少预生成诊断文件的既有测试跳过，2 个既有 `trapz` 弃用警告。
- 后续增加仅终点更新检查、修复导入缓存后，相关定向测试：286 passed，包含安装布局子进程检查。
- 显式禁用 JIT 的参考路径：264 passed。
- 三种采样率、两种 P 极性、强弱波共存、平线、非有限输入、QRS 排除窗、双导联反极性投影、缺失导联和质量拒绝均有检查；动态规划与穷举最优解对照。

```bash
# 传统算法完整消融，输出目录需尚未包含实验。
.venv/bin/python scripts/evaluate_ecgfeat_candidates.py \
  --variants default p_phasor_candidates t_projection_offset_only \
  t_sequence_offset_only classical_offsets --workers 8 --out /tmp/classical_ludb

.venv/bin/python scripts/evaluate_ecgfeat_candidates.py --dataset qtdb \
  --variants default p_phasor_candidates t_projection_offset_only \
  t_sequence_offset_only classical_offsets --workers 8 --out /tmp/classical_qtdb

.venv/bin/python scripts/validate_ecgfeat_regression.py \
  --baseline ecgfeat_improvements_20260922/equivalence_release \
  --out /tmp/classical_equivalence --workers 8

.venv/bin/python scripts/summarize_ecgfeat_classical.py \
  --root ecgfeat_classical_20260922

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_classical_candidates.py tests/test_kalman_acceleration.py
```

实验用源码快照仅在本地保存并被该实验目录的 `.gitignore` 排除，manifest 保存源码哈希。准确性消融冻结后只修复编译缓存导入；最终默认回归使用修复后的源码。所有准确性候选仍属于未校准研究分支；原 QT 风险校准表不能直接移植到这些分支。
