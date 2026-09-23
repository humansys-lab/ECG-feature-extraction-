# Feature extraction：实现与验证结果（2026-09-22）

本轮已将等价加速接入默认路径，并实现十项可独立消融的波界/分组候选、PR/TP 自适应 ST 幅值路径、QT 误差风险校准及跨库评估工具。验收结果支持默认启用等价加速；不支持将全部准确性候选一起默认启用。

## 1. 默认路径：已落地并通过等价验证

- `quality.py`：同一导联的七个频段复用一次 Welch PSD；原始信号与分析信号各自计算，168 次变为 24 次。
- `preprocess.py`：有界缓存 Butterworth/notch 系数；只缓存参数对应的系数，不跨记录缓存患者信号。
- `p_wave_engine.py`：同一次提取的静默区活动阈值只计算一次。
- `t_wave_refinement.py`：同一导联的小波细节序列在逐搏搜索间复用。
- `export.py`：`to_dict(features, profile="summary")` 在递归复制前跳过将被丢弃的逐搏数据；继续由 `prepare_json_export` 完成舍入和 JSON 清理。
- `batch_extract_ecgfeat.py`：JSON-only 且 serializer 支持 profile 时使用直接导出；保存 PT 时继续生成完整 payload，兼容原有单参数注入 serializer。

最终 [LUDB 等价门](../ecgfeat_improvements_20260922/equivalence_release/regression_gate.json)：200/200 条、21,632 个匹配导联事件，既有全局字段、逐事件误差和覆盖计数均在 1e-9 容差内一致，变化的波界误差数为 **0**。

另外对下表五条记录比较未经舍入的完整 debug 与 summary 导出，所有字段一致。比较只排除了 `clinical_interpretation.generated_at` 及其 metadata 副本两个运行时间戳；没有排除质量、门控、来源或诊断字段。首次哈希比较发现差异后，逐字段检查确认只来自这两个时间戳，随后重新完成了全部计时/比较。

## 2. 同机性能结果

单 BLAS/OMP 线程，500 Hz、12×5000、10 秒输入，每条预热一次、提取三次取中位数。旧/新实现使用隔离进程顺序执行，记录间交替先后顺序。计时不包括读文件、导入、报告、绘图和 JSON 编码，也不代表全库延迟分位数或冷启动耗时。

| LUDB 记录 | 改前提取/s | 改后提取/s | 耗时下降 | 完整转 summary/ms | 直接 summary/ms |
|---|---:|---:|---:|---:|---:|
| 1 | 1.943 | 1.773 | 8.8% | 22.75 | 13.98 |
| 74 | 1.930 | 1.770 | 8.3% | 28.51 | 16.62 |
| 125 | 1.824 | 1.674 | 8.2% | 25.65 | 16.06 |
| 127 | 1.586 | 1.436 | 9.5% | 22.29 | 14.97 |
| 57 | 2.956 | 2.752 | 6.9% | 32.31 | 16.51 |

五条中位数耗时合计下降 **8.15%**，约 **1.089×** 加速；直接 summary 导出合计下降 **40.58%**，其绝对收益约为每条 7–16 ms。导出内容与体积保持一致。这些数字来自选定样本，不把多进程评估总时间当作单条提取速度。

优化后同一组样本的独立阶段计时如下；只汇总 API 最外层调用，避免把父子耗时重复相加：

| 阶段 | 平均秒/条 | 占比 |
|---|---:|---:|
| `build_p_wave_assessments` | 0.612 | 32.2% |
| `delineate_beats` | 0.470 | 24.8% |
| `analysis_signal` | 0.214 | 11.3% |
| `apply_hybrid_st_measurement` | 0.120 | 6.3% |
| `refine_t_wave_boundaries` | 0.088 | 4.6% |
| `build_qrst_subtracted_residual` | 0.073 | 3.8% |
| `prepare_pacing_detection_cache` | 0.068 | 3.6% |
| `extract_atrial_events` | 0.057 | 3.0% |
| `compute_quality` | 0.056 | 3.0% |

**剩余计算瓶颈仍是 P 波评估和逐搏 delineation，两者约占 57%。** 大窗口中值基线也约占 11%。质量频谱计算已降到约 3%。本轮没有引入编译后端、替换中值基线、实现 ST/心房残差的跨阶段缓存或改写批处理调度；不能将这些后续空间算作已实现的加速。相关依赖和输出一致性尚需另行验证。

原始数据：[benchmark.json](../ecgfeat_improvements_20260922/benchmark.json)、[stage_profile.json](../ecgfeat_improvements_20260922/stage_profile.json)。

## 3. 准确性候选的实现

`RefinementConfig` 的十个布尔开关全部默认关闭。启用后的配置、P 修正审计和 T 融合支持信息写入 `metadata["refinement"]`。

| 开关 | 实现内容 | 本轮结论 |
|---|---|---|
| `p_model_arbitration` | 一致模型按证据加权；明显分歧时选强模型或拒绝，不直接平均不同波形假设；区间包含模型分歧 | 误差下降伴随覆盖损失，保持实验 |
| `p_multiple_candidates` | 弱/拒绝候选时增加最多两个活动窗，每窗独立估计 SVD 维度 | 未显示稳定总体收益 |
| `p_boundary_correction` | 最终心房状态接受后才替换弱 P 边界；更新 PR、时限、面积、形态、PTF、心房证据和依赖 P offset 的 hybrid ST | 未显示稳定总体收益；AF/AFL 门控测试通过 |
| `t_onset_change_point` | 常数 ST 段到 T 斜坡的变化点，并要求另一局部方法支持 | 开发集 T onset 退化，保持实验 |
| `t_bidirectional` | 局部双方法、其他独立导联及残余活动共同支持时，允许 T 终点向前或向后修正 | 全 LUDB 小幅收益，部分记录变差；无 QTDB 收益 |
| `t_correlated_fusion` | 去重复来源、限制单导联/肢体/派生信号权重，使用有效支持数估计区间 | 修正证据计数语义，未改善局部终点误差；区间仍未校准 |
| `qrs_terminal_multiscale` | 对宽 QRS 寻找多尺度一致的持续静默终点，限制最大延伸 | 未显示稳定总体收益 |
| `qrs_quality_reference` | 在可靠、非平线导联内选择 QRS 参考，重检测保持相同选项 | 开发集 P/QRS onset/QT 退化 |
| `representative_robust` | 限制相关搜索范围，以完整窗的 medoid 初始化，扩展慢心率分析窗 | 有改善也有退化，不能默认替换 |
| `grouping_outliers` | 超出常规组数且明显不相似时保留离群组 | 未显示总体准确性收益 |

`RefinementConfig.experimental()` 是消融入口，**不是验证通过的推荐 preset**。这里的“已实现”与“已证明提高准确性”是不同结论。代表搏候选也尚未实现完整的缺失样本掩码相关和跨全部采集场景验证。

## 4. LUDB：开发消融及完整回归

按记录名的固定 SHA256 划分得到 49 条开发记录，另有 151 条用于内部校准验证。LUDB 历史上已经参与算法开发，因此后者不是独立外部留出集。逐项开发消融基于 `development/manifest.json` 的源码；最终组合验证基于 `ludb_all_final/manifest.json`。每个实验均保留源码哈希、记录清单、失败记录、逐事件 CSV 和覆盖计数。

49 条开发消融的 MAE（ms；各列自身有效输出集合，配对数据另存）：

| 候选 | P onset | P offset | QRS offset | T onset | T offset | QT |
|---|---:|---:|---:|---:|---:|---:|
| all | 28.310 | 28.211 | 13.210 | 37.297 | 28.418 | 32.579 |
| default | 25.933 | 26.132 | 14.174 | 38.703 | 28.283 | 31.607 |
| grouping_outliers | 25.933 | 26.132 | 14.186 | 38.703 | 28.285 | 31.674 |
| p_boundary_correction | 25.953 | 26.171 | 14.174 | 38.703 | 28.283 | 31.607 |
| p_model_arbitration | 25.563 | 25.795 | 14.174 | 38.703 | 28.283 | 31.607 |
| p_multiple_candidates | 25.925 | 26.195 | 14.174 | 38.703 | 28.283 | 31.607 |
| qrs_quality_reference | 28.545 | 28.290 | 13.433 | 37.098 | 28.041 | 32.085 |
| qrs_terminal_multiscale | 25.924 | 26.137 | 14.148 | 38.550 | 28.390 | 31.708 |
| representative_robust | 26.047 | 26.196 | 13.910 | 38.504 | 28.848 | 32.120 |
| t_bidirectional | 25.911 | 26.103 | 14.174 | 38.703 | 28.235 | 31.558 |
| t_correlated_fusion | 25.933 | 26.132 | 14.174 | 38.709 | 28.283 | 31.607 |
| t_onset_change_point | 25.983 | 26.176 | 14.174 | 38.889 | 28.283 | 31.607 |

例如 P 仲裁的 P onset MAE 从 25.933 降至 25.563 ms，但有效评分从 4,089 降至 4,055，不能只报平均误差。全部候选同时开启的开发结果更差，后续完整 LUDB 也确认了这一点。

200 条完整数据的 MAE（ms）：

| 配置 | P onset | P offset | QRS onset | QRS offset | T onset | T offset | QT |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 25.900 | 24.692 | 11.634 | 13.479 | 34.761 | 25.229 | 28.477 |
| all | 26.468 | 25.375 | 12.415 | 13.381 | 34.249 | 25.331 | 29.263 |
| t_bidirectional | 25.895 | 24.683 | 11.634 | 13.479 | 34.761 | 25.177 | 28.416 |
| t_correlated_fusion | 25.900 | 24.692 | 11.634 | 13.479 | 34.763 | 25.229 | 28.477 |

- T 双向修正：T offset 25.229→25.177 ms，QT 28.477→28.416 ms；对应 P95 仍为 100/104 ms，T/QT 有效事件仍为 19,225。配对记录平均 T offset 误差改善 0.040 ms，记录 bootstrap 95% 区间 −0.076 至 −0.012 ms；20 条记录改善、8 条变差。P onset/offset 各减少 3 个有效评分。因此收益很小，不能描述为解决了 T 波长尾问题。
- 全部候选：QT MAE 28.477→29.263 ms，P95 104→106 ms；QRS 匹配数 21,632→21,604，QT 有效事件 19,225→19,196。配对记录平均 QT 误差增加约 0.804 ms，bootstrap 区间 0.044 至 1.591 ms。保持关闭。
- 相关性修正没有改变局部 T offset/QT MAE。权重整体缩放、重复相同来源不会再凭空缩窄其区间，但有效支持数与区间宽度仍是启发式统计，不是已校准的 95% 临床区间。

按专家标注的逐导联 QRS 宽度分层：宽 QRS（≥120 ms）默认 QRS onset/offset MAE 为 **22.74/33.66 ms**，窄 QRS 为 **10.13/10.74 ms**。全部候选使宽 QRS offset 降至 29.07 ms，但 onset 上升至 26.15 ms、QT 上升至 40.69 ms，说明修正一个终点并未改善整条测量链。T 双向修正在宽 QRS 中把 QT MAE 从 38.14 降至 37.96 ms，幅度仍很小。

配对统计按共同标注事件计算，再按记录求均值、以记录为单位进行 5,000 次 bootstrap；区间未作多重候选比较校正。误差与未输出分别保留，不把拒绝样本删除后的 MAE 当作全面提升。

数据：[comparisons.json](../ecgfeat_improvements_20260922/comparisons.json)、[stratified.json](../ecgfeat_improvements_20260922/stratified.json)。

## 5. QTDB：补齐当前版本的 105 条验证

采用现有 q1c 标注、真实两通道映射至 II/V2、其余通道置零、限定标注簇并加 2 秒上下文的固定适配协议。它是稀疏导联外部测量验证，不等同于 12 导联临床验证。

| 事件 | 召回率 | 精确率 | onset MAE/ms | offset MAE/ms |
|---|---:|---:|---:|---:|
| P | 91.92% | 93.32% | 49.19 | 41.06 |
| QRS | 98.68% | 94.70% | 15.17 | 25.03 |
| T | 93.59% | 89.79% | 46.77 | 31.24 |

默认、T 双向修正、T 相关性修正各跑完 105 条，无执行失败，事件汇总及误差结果相同。T 双向修正要求至少三个其他独立导联支持，两通道适配不具备该条件；这不能证明该候选在真实 12 导联外部数据上有效。QTDB 的 P 波边界和部分困难 QRS 仍是明显短板。

数据：[QTDB summary](../ecgfeat_improvements_20260922/qtdb/default/summary.json)。

## 6. ST：90 条 EDB 的独立对照

三条幅值路径各完成 90 条记录；每条最多取三个事件和固定对照协议，共 215 个事件、180 个通道对照。原始电压校准路径只更改 hybrid ST 旁路结果，不声称所有 native ST 字段或诊断规则已随之改写。

新增 `adaptive_pr_tp`：从已接受波界寻找静默 PR/TP 锚点，剔除污染锚点，估计缓慢基线和启发式 mV 不确定度；锚点间距、外推距离或不确定度不足时拒绝输出。API 对 J 到 J+80 ms 的采样窗检查支持，代表搏要求捐赠搏支持；稀疏导联评估适配器也保留此拒绝条件。

| hybrid 路径 | 有效事件/215 | 条件 MAE/mV | 同一有效事件上旧路径 MAE/mV | 全部215事件中阈值检出/215 |
|---|---:|---:|---:|---:|
| analysis | 209（97.21%） | 0.09285 | — | 133/215（61.86%） |
| calibrated_pr | 176（81.86%） | 0.06131 | 0.09076 | 134/215（62.33%） |
| adaptive_pr_tp | 145（67.44%） | 0.06263 | 0.08499 | 108/215（50.23%） |

PR 路径在共同的 176 个事件上 MAE 下降约 32.4%；按记录平均的误差下降约 0.0294 mV，bootstrap 区间为 −0.0432 至 −0.0160 mV。但它的有效输出减少，不能只用有输出事件上的敏感度 76.14% 替代全部事件的检出比例。PR+TP 候选覆盖更低、未优于 PR，保持实验状态；未自动切换默认 ST 通路。

合成测试覆盖 ±0.18 mV ST、平坦/线性/二次漂移、无锚点和远距离不支持区。误差与覆盖率见 [EDB 比较](../ecgfeat_improvements_20260922/comparisons.json)。

## 7. QT 置信度校准

新增 `QTErrorCalibration`，学习“局部 QT 绝对误差超过 40 ms”的参考队列风险。每条记录的总权重相同，按原始 confidence 的五个区间分组并向训练集风险收缩，不强制原始分数越大误差越小。该模型与原始 `qt_confidence` 和报告门控分开。

49 条拟合、151 条验证；14,417 个具有有限 QT 误差的验证事件。记录加权 Brier：常数训练风险基线 **0.16401**，分箱模型 **0.15029**。验证集中 0.6–0.8 分数的 >40 ms 风险约为 9.62%，0.8–1.0 分数反而约为 32.81%，印证原 confidence 不能直接解释为准确率。

模型保留训练记录 ID、数据 SHA256、适用管线和状态标签，支持 JSON 保存/加载。缺失 QT 不参与风险拟合，另外报告有限结果数。这里只完成默认管线的 LUDB 内部 QT 校准；**P 波置信度、独立外部校准、误差区间覆盖率校准尚未完成**。不应将此风险表直接移植到另一配置或人群。

模型：[model.json](../ecgfeat_improvements_20260922/calibration/model.json)；验证：[validation.json](../ecgfeat_improvements_20260922/calibration/validation.json)。

## 8. 测试、复现和使用

- 特征提取相关完整测试：**1,157 passed，150 subtests passed**；两个既有 NumPy `trapz` 弃用警告。
- 最低依赖环境（Python 3.10 / NumPy 1.24 / SciPy 1.10）：**75 passed**。
- 最后一处 P→hybrid ST 依赖刷新后，针对 API、候选、导出、批处理再测：**267 passed**；最终默认 LUDB 等价门通过，全部候选重新跑完 200 条，无执行失败。
- 新增测试覆盖候选开关、心房状态拒绝、P 下游量更新、非零 ST 上的 T 变化点、宽 QRS 候选保护、离群组、融合缩放/重复不变性、自适应 ST 支持、稀疏适配器拒绝、校准记录权重及直接导出等价。

复现示例（仓库根目录；实验输出必须选择新目录）：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate_ecgfeat_regression.py \
  --baseline ecgfeat_validation_fixes_20260908/final --out /tmp/ecgfeat_equivalence --workers 4

.venv/bin/python scripts/evaluate_ecgfeat_candidates.py \
  --partition development --variants default t_bidirectional t_correlated_fusion all \
  --out /tmp/ecgfeat_development --workers 8

.venv/bin/python scripts/evaluate_ecgfeat_candidates.py --dataset qtdb \
  --variants default t_bidirectional --out /tmp/ecgfeat_qtdb --workers 8

.venv/bin/python scripts/evaluate_ecgfeat_st_candidates.py \
  --out /tmp/ecgfeat_edb --workers 8

.venv/bin/python scripts/calibrate_ecgfeat_qt.py \
  --errors ecgfeat_validation_fixes_20260908/final/ludb_errors.csv \
  --manifest ecgfeat_improvements_20260922/development/manifest.json \
  --out /tmp/ecgfeat_calibration

.venv/bin/python scripts/benchmark_ecgfeat_equivalence.py \
  --baseline-source ecgfeat_improvements_20260922/baseline_source \
  --out /tmp/ecgfeat_benchmark.json
```

计时所用旧源码快照保留在本地实验目录并通过该目录 `.gitignore` 排除版本跟踪；重新取得仓库后需另行提供旧源码根目录。各实验文件夹的 manifest 记录了当时的源码哈希，开发、最终组合和外部验证的快照版本可据此区分。根目录其他既有修改未回退。

代码入口：[RefinementConfig](../feature_extraction/ecgfeat/refinement.py)、[边界候选](../feature_extraction/ecgfeat/boundary_refinement.py)、[ST 基线](../feature_extraction/ecgfeat/st_baseline.py)、[风险校准](../feature_extraction/ecgfeat/calibration.py)。API 用法见 [feature_extraction README](../feature_extraction/README.md)。
