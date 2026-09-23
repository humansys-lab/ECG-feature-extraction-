ECG 特征提取修正与回归验收（2026-09-08）

本次采用保守的兼容策略：修复明确的代码错误，为会改变算法结果的方案提供独立输出或显式开关，并在冻结的 LUDB 基线上逐项比较。最终默认通路通过严格回归门，未通过的候选没有替换默认结果。

**默认通路的验收结果**

- LUDB 1.0.1 全部 200 条记录，内部采样率 500 Hz、工频参数 50 Hz、默认配置、不传患者 metadata。
- 相对 2026-09-07 保存的基线，200 条记录既有全局字段和心搏计数全部一致。
- 21,632 个匹配导联事件的 QRS/P/T 边界及 QT 误差全部一致，数值容差为 `1e-9`。
- 2,400 个记录×导联的标注覆盖计数一致；其中无标注的导联仍记入覆盖表。
- QTDB 的 sel100、sel102、sel103：检测事件明细、匹配明细和按波型统计与之前结果一致。该结论仅限现有双导联适配器和这 3 条记录。
- 当前环境：72 个相关测试文件，1,079 项通过、150 项子测试通过；不是包含其他应用模块的全仓库测试。
- 实际最低依赖环境：Python 3.10.12、NumPy 1.24.4、SciPy 1.10.1，284 项相关测试通过。

直接人工波界误差（修正前后相同）：

| 指标 | MAE（ms） | P95 绝对误差（ms） |
|---|---:|---:|
| QRS onset | 11.6335 | 32 |
| QRS offset | 13.4795 | 42 |
| P onset | 25.9004 | 86 |
| P offset | 24.6917 | 80 |
| T onset | 34.7614 | 114 |
| T offset | 25.2292 | 100 |
| 单导联逐搏 QT | 28.4766 | 104 |

“没有退化”在这里指上述冻结配置和数据上的逐项一致，不是对所有未来记录、设备、噪声和患者群体的数学保证。新增统计字段和可选 ST 通路不在既有数值一致性约束内，分别验证。

**同机耗时比较**

使用相同 Python/NumPy/SciPy 环境、单 BLAS 线程，每条记录先预热一次，再测 5 次；按记录交错基线和候选的运行顺序。计时仅覆盖 extract，不含 JSON 导出。

| LUDB 记录 | 基线中位秒数 | 修正后中位秒数 | 变化 |
|---|---:|---:|---:|
| 1 | 1.9175 | 1.9330 | +0.81% |
| 74 | 1.9762 | 1.9641 | -0.61% |
| 125 | 1.8397 | 1.8536 | +0.76% |

观测变化为 −0.61% 到 +0.81%，耗时基本持平。不能据此承诺任意输入耗时严格不增加，也没有宣称稳定加速。原始重复测量见 runtime_comparison.json / paired_benchmark.json。

**修正内容与行为**

1. QRS 参考导联遵守调用者选择。默认精修只使用参与检测且非平线的导联；改变未选择的 II 导联不会移动结果。对于起搏捕获救援，增加显式 `fiducial_lead` 参数，允许不同能量导联共用 II 时间参考。这样保留该专门分支的定位契约，同时消除普通调用中的隐式依赖。
2. QRS 能量轨迹和记录级置信度归一化统计只计算一次，主阈值与 fallback 复用结果。阈值和置信度公式不变。
3. ST 面积积分使用已有 NumPy 1.x/2.x 兼容封装。真实最低版本测试又发现 `repolarization._trapz` 将同名导入覆盖成 `None`，已一并改为调用兼容封装。现代 NumPy 下计算方法不变。
4. LUDB 评估把参考波界转换到检测器时间基准，保留小数样本，避免重采样后混用索引。增加标注覆盖计数、未匹配 QRS 计数和运行失败信息；即使提取失败，也尽可能保留已可读取标注的分母。空标注导联仍保留零计数行。误差与覆盖率同时保存到 `summary.json`。
   时间基准实测：LUDB 记录 1、内部采样率 250 Hz 时，修复前只匹配 28 项且 QT MAE 490 ms；修复后匹配 72 项、QT MAE 17.07 ms。默认 500 Hz 的 72 项匹配和 QT MAE 17.50 ms 不变。
5. QT dispersion 的统计定义拆开表达。`qt_dispersion_ms` 保留既有兼容估计；新增 `qt_dispersion_independent_ms`（合格导联完整极差）、`qt_dispersion_p90_p10_ms`、来源、参与导联和旧聚类剔除明细。相关逻辑独立到 `dispersion.py`。
6. 增加 `st_amplitude_source="calibrated_pr"` 可选通路，在原始标定信号上用安静 PR 段估计基线，更新 hybrid ST 字段。默认仍为 `"analysis"`；缺少足够 PR 锚点时显式标记不可用。保真算法独立到 `st_baseline.py`。
7. `metadata.confidence_semantics` 明确局部 QT 和 QRS 分数尚未校准，不代表准确率概率；最终 QT 报告门控仍为 `global_features.qt_reportable`。本次没有用同一 LUDB 测试集重拟合置信度。

**没有直接启用的算法替换**

取消 QT dispersion 的紧簇截断后，195 条成对记录的 MAE 从 27.8462 降至 27.4308 ms，P95 从 80.9 降至 78.6 ms，但有 **9 条记录误差变大**。例如记录 8 的误差由 24 增至 76 ms。该候选未满足用户要求的保守回归条件，所以仅作为新增独立极差字段提供。

QRS 修正最初使起搏记录 74 的 37 项局部误差发生变化。将起搏分支的共同时间参考显式化后，该记录的既有全局值与逐搏值恢复一致；随后重新运行整个 200 条记录验收。

ST 通路在已知 ±0.18 mV 波形、有/无线性漂移下通过 0.003 mV 绝对误差测试；在 LUDB 记录 1 上，启用该通路后 native P/QRS/T 波界和 HR/PR/QRS/QT/QTc 保持一致。它确实改变 hybrid ST 电压，且可能影响读取 hybrid 质量/形态的解释规则。由于尚无独立 ST 全库结果证明不退化，它没有成为默认通路；不能将合成保真测试等同临床 ST 准确性验证。

**复现与持续回归**

在仓库根目录执行，需要已保存的基线及本地 LUDB 数据：

```bash
python scripts/validate_ecgfeat_regression.py \
  --baseline ecgfeat_validation_review_20260907 \
  --out ecgfeat_validation_fixes_20260908/recheck --workers 4
```

检查程序在任何既有全局数值、匹配误差、事件集合或覆盖计数变化时返回非零退出码。这个门比“平均性能不下降”更严格，算法改进也可能触发失败，需要针对独立评价证据作单独审查，不能为获得通过而静默放宽条件。测试还覆盖了“均值相同但个别事件变差”和“丢失困难样本使 MAE 变好”的反例。

本次证据保存在 `ecgfeat_validation_fixes_20260908/`：

- [最终回归门结果](/workspace/ecg_gemma/ecgfeat_validation_fixes_20260908/final/regression_gate.json)
- [最终指标、版本、源码 SHA256](/workspace/ecg_gemma/ecgfeat_validation_fixes_20260908/final/summary.json)
- [完整测试日志](/workspace/ecg_gemma/ecgfeat_validation_fixes_20260908/final_feature_tests.log)
- [最低依赖测试日志](/workspace/ecg_gemma/ecgfeat_validation_fixes_20260908/minimum_tests_final.log)
- [未采用的 dispersion 候选结果](/workspace/ecg_gemma/ecgfeat_validation_fixes_20260908/rejected_dispersion_candidate.json)
- [ST 可选通路实测检查](/workspace/ecg_gemma/ecgfeat_validation_fixes_20260908/st_optin_check.json)
- [同机耗时原始结果](/workspace/ecg_gemma/ecgfeat_validation_fixes_20260908/paired_benchmark.json)

源码变更集中在 `feature_extraction`、LUDB 评估、回归检查脚本及相关测试。工作区同时存在的 ECG Agent 等其他修改未由本次任务改动。
