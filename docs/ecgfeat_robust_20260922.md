# ECG 传统提取算法改进与完整验证（2026-09-22）

本轮推进了病理 P 波、抗噪 QRS、少导联输入与输出契约、T 终点候选修复，并完成等价缓存加速。
全部使用传统信号处理、规则和动态规划；没有增加深度学习模型或训练权重。
默认十二导联测量保持原值；有改变测量结果的算法采用独立开关，适用配置见下文。

![验证结果](../ecgfeat_robust_20260922/validation_results.png)

## 实现与算法依据

**抗噪 QRS。** 每个实际通道先独立带通，再进行能量检测与通道选择，避免在带通前的非线性多导联融合中混入运动伪迹。
短/长移动平均采用 111/667 ms，阈值包含平均能量的 2%，思路来自
[Elgendi 等的双移动平均方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC3774726/)。本项目增加的质量选择、跨通道支持和回退规则不属于论文原算法的复现。
高检测器一致性时使用原检测器；混合噪声窗口按 4 秒区段、两侧各 2 秒上下文判定局部质量。
使用 200 ms 不应期和 50 ms 通道支持，不强制 RR 等间隔。

开发中发现 BUT 13 的非传导 P 可触发窄带 QRS 包络。最终增加相对振幅约束：至少六个原激活、两检测器保留原激活的比例不少于 90%、
至少一个通道的 QRS 振幅稳定时，仅剔除在所有实际通道均低于参考 QRS 振幅 35%、且在一个稳定通道低于 15% 的新增弱激活。
只在存在至少三个新增激活的情形启用，不删除原检测器的激活，不通过规则化 RR 排除异位搏动。
最近峰搜索使用二分查找，避免长记录建立二次方大小的距离矩阵。低振幅异位搏动的全面验证仍是后续工作。

**P 波。** 在原生候选缺失时也允许 phasor 搜索，将搜索上限扩展到 450 ms，同时受前一 RR 的 65% 和前一 T 终点约束。
候选需通过局部基线突出度、噪声、半高宽及至少两个实际独立通道的证据检查，再进入原有上下文重选流程。
这吸收了[病理 ECG 的 phasor P 检测研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9023481/)的候选搜索思路；
没有把论文报告的准确率当成本项目结果。

独立心房事件采用紧邻 QRS 的跨通道冲突核验（25 ms）和已测得 T 峰附近的冲突核验。
第一版“必须有重复模板”的过滤器会漏掉真实非传导 P，已撤回；孤立、无重复模板的非传导 P 不因此被删除。
残余事件仍属于候选，不自动成为被接受的 P 波或节律诊断。

**少导联。** `input_mode="limited"` 接受 1～8 个明确命名的实际通道，缺失导联不再按平线坏导联计数。
本轮定量性能验证主要覆盖单/双通道，3～8 通道仅属于接口支持范围。
不生成未采集的肢体导联；P 质量支持数按实际通道数调整，但保留噪声、重复通道、同步性、边界一致性和节律门控。
未知通道的 II/V2 等名称仅是计算槽位，映射在 `metadata.input_contract.lead_slot_map` 中。
输出删除空槽位，关闭解剖电轴、十二导联诊断、形态规则导出、诊断语句引擎和正式 QT 报告资格，保留可审核的原始区间。
这也修复了此前“双导联补十个零通道后，所有 P 测量被十二导联门槛拒绝”的契约问题。

**T 波。** 先完成原有波界与晚尾修复，再施加可选终点候选，随后重新计算融合；避免候选改变融合后再次触发不同的原生 Ton 修复。
被接受的局部候选通过同一更新函数同步 QT、JT、Tpeak–Tend、T 时长和面积。
ISP train/324 与 train/343 已加入真实数据回归测试。475 条 ISP 记录的三种终点配置均未改变 II 导联原生 Ton/Tpeak。
后续 P 回填依赖 T 窗口，仍可能改变 P 输出；终点选项不是全流水线等价变换。
原有个别 `Ton == Tpeak` 的退化测量没有被这次候选修复全面解决，也不声称所有旧波界均正确。

**等价加速。** P 候选支持通道预先筛选，同一通道/候选位置的波形证据复用。
五条 LUDB 固定记录、每条预热一次后测三次，中位耗时之和从 9.429 s 到 9.290 s，降低约 1.48%；
完整 debug 和 summary 导出内容哈希一致，仅排除两处生成时间戳。该对照仅打开 `p_pathology_candidates`，比较 v2 与 v5 保存的源码。
此前已完成的频谱、滤波器、P/T 细节缓存与可选 Numba 加速，见[前一轮报告](ecgfeat_classical_20260922.md)。

## 数据与评价口径

| 数据 | 本轮范围 | 用途与约束 |
| --- | --- | --- |
| LUDB | 200 条 | 十二导联回归和消融；按导联标注区间评分，边界覆盖率不是 P/T 事件灵敏度 |
| BUT PDB | 50 条、5,429 个有效 P 标注、7,572 个有效 QRS 标注 | 病理 P 与 QRS；P 容差 50/150 ms，QRS 75 ms |
| NSTDB | 118/119 两个来源、各六档噪声 | 分开统计噪声段和干净段；预生成数据仅包含电极运动噪声 |
| ISP v2 | train 403、test 72，共 475 条 | 综合波段标注；IoU≥0.5 一对一匹配，主输出为有效导联边界中位数 |
| GUDB 1.1.0 | 25 人、125 次记录；229 个有标注设备流 | 双导联 cable 和单通道 chest strap 分别评分，QRS 容差 50/75 ms |

BUT/NSTDB 使用 30 秒核心窗和两侧各 2 秒上下文，按核心区间保留预测，每个标注计一次；失败窗口仍计入漏检分母。
GUDB 使用相同窗口，评估生产预处理和 QRS 检测器，不包含后续整条流水线的起搏救援；这是与 BUT/NSTDB 端到端评价的区别。
所有提取均不读取标注来决定窗口、通道或候选位置。

BUT 开发记录固定为 01、08、10、13、22、30、42、48。其余 42 条曾在前一轮评估中被查看，因此只是回归余集，不能称为新的盲测。
LUDB 与 ISP 也已在此前使用。GUDB 首次以冻结算法评估；之后的弱激活约束仅依据原开发记录 BUT 13 修订，
GUDB 用于再次验证，最终结果不能重新标成“首次盲测”。BUT 来自部分相同来源患者，NSTDB 只有两个原始来源，不能把所有心搏当作独立样本。

GUDB 来源和设备说明见[官方仓库](https://github.com/berndporr/ECG-GUDB)，版本固定为
[Zenodo 10925903](https://zenodo.org/records/10925903)，压缩包 MD5 为 `9f50e13470715d58d2aa9675e15a8f54`。
依[官方绘图示例](https://github.com/berndporr/ECG-GUDB/blob/1.1.0/usage_example.py)将伏特换算成 mV，仅读取 ECG 列。
有 21 个设备流缺少标注，明确列为不可评分；其中 19 个 cable、2 个 chest strap。

最初将两个设备流直接融合的试验无效：subject_00/sitting 的带通包络互相关时差约 −632 ms，分别提供的 R 标注也显示不同时间轴。
未使用真值平移数据；最终完全分流评估。错误协议结果保存在 `gudb_holdout`，不纳入最终指标。
[时间轴审计](../ecgfeat_robust_20260922/probes/gudb_clock_audit.json)保留了原始证据。

NSTDB 的段落时间和噪声来源遵循[PhysioNet 官方说明](https://physionet.org/content/nstdb/1.0.0/)。
ISP 中两个越界 T 终点仅裁剪可见区间并对终点误差删失；一个零长 T 标注排除。
BUT 的负数 sentinel 和非心搏 flutter 标记独立审计。完整单位、标签审计沿用[外部数据报告](ecgfeat_external_20260922.md)。

## BUT：P 与 QRS 的全量消融

`robust_p` = P phasor 扩展 + 心房冲突核验；`limited_p` 在此基础上使用少导联契约；
`limited_qrs` = 少导联 + 抗噪 QRS；`limited_robust` = 少导联 + P 改进 + QRS 改进 + T 终点序列选择。

| 配置 | P F1 / 50 ms (%) | P F1 / 150 ms (%) | P TP / FP / FN (150 ms) | QRS F1 / 75 ms (%) |
| --- | --- | --- | --- | --- |
| default | 60.344 | 80.678 | 4547 / 1296 / 882 | 95.314 |
| robust_p | 61.592 | 81.768 | 4689 / 1351 / 740 | 95.314 |
| limited | 59.990 | 80.248 | 4591 / 1422 / 838 | 95.314 |
| limited_p | 61.512 | 81.676 | 4727 / 1419 / 702 | 95.314 |
| limited_qrs | 60.296 | 80.923 | 4649 / 1412 / 780 | 96.211 |
| limited_robust | 61.892 | 82.356 | 4789 / 1412 / 640 | 96.211 |

主表是原生 P 候选的检测表现。最终严格门控 P 的 TP/FP/FN 为
1891/285/3538，precision=86.903%，recall=34.831%。
原始心房事件流的 TP/FP/FN 为 5099/5311/330，precision=48.982%；
这个候选流不能直接作为正式 P 波清单。

两条无 P 记录仍是重要限制：

| 记录 | 旧 native P FP | 改进 native P FP | 改进 accepted P FP |
| --- | --- | --- | --- |
| 08 | 102 | 114 | 0 |
| 48 | 33 | 33 | 0 |

原生候选误检并未全部解决。严格门控保持零接受，不等于底层候选检测已经实现零误检。
不同病例间仍有取舍；完整逐记录结果和开发/余集拆分保存在 `analysis.json` 与各实验 CSV 中。

## NSTDB：噪声段结果

| SNR (dB) | 原 QRS F1 (%) | 改进 QRS F1 (%) | 改进 TP / FP / FN |
| --- | --- | --- | --- |
| 24 | 99.973 | 99.973 | 1856 / 1 / 0 |
| 18 | 99.919 | 99.919 | 1855 / 2 / 1 |
| 12 | 94.659 | 94.706 | 1789 / 133 / 67 |
| 6 | 80.681 | 89.716 | 1723 / 262 / 133 |
| 0 | 56.847 | 76.450 | 1529 / 615 / 327 |
| -6 | 34.877 | 42.230 | 837 / 1271 / 1019 |

每档噪声段共有 1,856 个真值 QRS。干净区段另行检查，不能以大量干净心搏稀释重噪声错误。
最终干净段逐峰一致性见 [external_regression.json](../ecgfeat_robust_20260922/external_regression.json)。
−6 dB 的结果仍远不足以称为稳定检测。该数据库只有两个原始来源，不据此给出跨患者置信区间。

## GUDB：外部受试者验证

| 设备流 | 原 F1 / 75 ms (%) | 改进 F1 / 75 ms (%) | 原 → 改进 F1 / 50 ms (%) | 改进 TP / FP / FN |
| --- | --- | --- | --- | --- |
| cables | 95.918 | 97.211 | 95.713 → 97.087 | 18040 / 137 / 898 |
| chest_strap | 89.620 | 91.792 | 89.473 → 91.641 | 21381 / 1983 / 1841 |

双导联按活动拆分：

| 活动 | 双导联原 F1 (%) | 双导联改进 F1 (%) |
| --- | --- | --- |
| sitting | 99.603 | 99.603 |
| maths | 99.507 | 99.507 |
| walking | 94.963 | 95.260 |
| hand_bike | 96.891 | 97.285 |
| jogging | 81.976 | 92.171 |

按受试者聚类进行 5,000 次配对 bootstrap，保留同一人的多种活动在同一次抽样中：

| 任务 | F1 提升(百分点) | 95% 按受试者 bootstrap 区间(百分点) |
| --- | --- | --- |
| cables / 50 ms | 1.375 | [0.509, 2.407] |
| cables / 75 ms | 1.293 | [0.454, 2.298] |
| chest_strap / 50 ms | 2.168 | [0.849, 3.683] |
| chest_strap / 75 ms | 2.171 | [0.875, 3.668] |

这些区间描述此数据队列上的重抽样不确定性，不包含开发选型和协议修订带来的全部不确定性。
F1 提高不代表所有匹配峰的平均时间误差都会下降；新增难例会改变匹配样本组成，时间误差另见原始 summary。

## LUDB 与 ISP：边界精度、覆盖和回归

LUDB 的单位均为 ms，有效数为按导联可评分的边界数：

| 配置 | P-on MAE / 有效数 | P-off MAE / 有效数 | T-off MAE / 有效数 | QRS 匹配数 |
| --- | --- | --- | --- | --- |
| default | 25.900 / 15849 | 24.692 / 15978 | 25.229 / 19225 | 21632 |
| robust_p | 25.777 / 15919 | 24.540 / 16048 | 25.229 / 19225 | 21632 |
| t_sequence_offset_only | 25.899 / 15844 | 24.687 / 15973 | 25.004 / 19225 | 21632 |
| qrs_adaptive_consensus | 25.870 / 15853 | 24.703 / 15982 | 25.226 / 19228 | 21630 |

原默认 LUDB 全部 200 条、21,632 个匹配行的既有全局字段、波界误差、标注覆盖在 1e-9 容差内一致。
P 扩展在 LUDB 增加了可用 P 边界；T 终点序列选择小幅降低 T-off 误差，但通过后续门控改变了少数 P 回填，
不能只看 T 平均误差就宣布全项改善。十二导联 QRS 的变化也单独保留，不自动替换原默认。

ISP 官方 test 72 条的 T 综合区间结果：

| T 配置 | 匹配数 | F1 (%) | Ton MAE (ms) | Toff MAE (ms) |
| --- | --- | --- | --- | --- |
| classical_offsets | 808 | 92.927 | 19.538 | 15.610 |
| default | 810 | 93.157 | 19.596 | 15.907 |
| t_projection_offset_only | 808 | 92.927 | 19.538 | 15.623 |
| t_sequence_offset_only | 810 | 93.157 | 19.596 | 15.806 |

T 终点序列选择在 test 的共同 810 个匹配上，平均绝对误差减少约 0.101 ms；
按记录平均差的 bootstrap 区间约 [−0.186, −0.009] ms。
这小于单个 500 Hz 采样点的 2 ms 间隔，只是队列平均的微小改善。
投影及投影组合虽然条件 MAE 更低，却少匹配两个 T；不作为默认推荐。
train 部分的差值、失去/新增匹配及置信区间见 `analysis.json`，不隐藏不利结果。

## 运行成本

下表为单进程单计算线程、预热一次、重复三次的中位耗时；运行时没有并发评估作业。
每行是固定样本，不是全数据集速度承诺。检测到更多真实搏动会增加后续分界与特征计算量。

| 数据 / 记录 | 配置 | 信号时长(s) | 提取中位耗时(s) | 检测搏数 |
| --- | --- | --- | --- | --- |
| but-pdb / 01 | limited | 32.0 | 2.242 | 24 |
| but-pdb / 01 | limited_p | 32.0 | 2.298 | 24 |
| but-pdb / 01 | limited_qrs | 32.0 | 2.250 | 24 |
| but-pdb / 01 | limited_robust | 32.0 | 2.322 | 24 |
| but-pdb / 13 | limited | 32.0 | 1.921 | 19 |
| but-pdb / 13 | limited_robust | 32.0 | 1.978 | 19 |
| nstdb / 118e00 | default | 32.0 | 2.880 | 37 |
| nstdb / 118e00 | qrs_adaptive_consensus | 32.0 | 3.462 | 44 |
| isp / test_data/1 | default | 9.999 | 2.553 | 11 |
| isp / test_data/1 | t_sequence_offset_only | 9.999 | 2.567 | 11 |
| isp / test_data/1 | classical_offsets | 9.999 | 2.585 | 11 |

等价 P 缓存加速与准确性算法的新增成本应分开看；前者保持完整输出相同，后者有意改变候选和测量。
速度数据、逐次计时及内容哈希见 [缓存 A/B](../ecgfeat_robust_20260922/p_cache_equivalence.json) 和
[配置耗时](../ecgfeat_robust_20260922/profile_runtime.json)。

## 使用方式与交付范围

标准十二导联保持 `ECGFeatureExtractor()` 原默认。需要双导联病理/噪声提取时，可显式启用本轮已评估组合：

```python
from ecgfeat import ECGFeatureExtractor, RefinementConfig

extractor = ECGFeatureExtractor(
    fs_internal=500, mains_freq=50, input_mode="limited",
    refinement=RefinementConfig(
        p_pathology_candidates=True, atrial_event_validation=True,
        qrs_adaptive_consensus=True, t_sequence_offset_only=True,
    ),
)
features = extractor.extract(
    ecg_channels, fs=360, amplitude_unit="mV",
    lead_names=["channel_0", "channel_1"],
)
accepted_beats = {a.beat_id for a in features.p_wave_assessments if a.accepted}
```

通道必须同步，不能先补十行零再传入 limited 模式。已知导联名称可直接使用 `II`、`V1` 等。
如果只需要 P 改进，关闭 `qrs_adaptive_consensus`；如果只需要抗噪 R 检测，关闭两个 P 开关。
T 序列选择是小收益可选项；`RefinementConfig.experimental()` 仍不推荐用作整套默认配置。

可复现入口（在仓库根目录运行，输出使用新目录）：

```bash
.venv/bin/python scripts/download_ecgfeat_external.py --datasets gudb
.venv/bin/python scripts/evaluate_ecgfeat_external.py --dataset but-pdb --variants default robust_p limited limited_p limited_qrs limited_robust --workers 6 --out /tmp/ecgfeat_but_recheck
.venv/bin/python scripts/evaluate_ecgfeat_external.py --dataset nstdb --variants default qrs_adaptive_consensus --workers 6 --out /tmp/ecgfeat_nstdb_recheck
.venv/bin/python scripts/evaluate_ecgfeat_gudb.py --workers 4 --out /tmp/ecgfeat_gudb_recheck
.venv/bin/python scripts/evaluate_ecgfeat_external.py --dataset isp --variants default t_projection_offset_only t_sequence_offset_only classical_offsets --workers 8 --out /tmp/ecgfeat_isp_recheck
.venv/bin/python scripts/evaluate_ecgfeat_candidates.py --dataset ludb --variants robust_p qrs_adaptive_consensus t_sequence_offset_only --workers 4 --out /tmp/ecgfeat_ludb_recheck
.venv/bin/python -m pytest -q tests/test_limited_leads.py tests/test_robust_detectors.py tests/test_gudb_protocol.py tests/test_t_refinement_isp_regression.py
```

验证总账见 [validation.json](../ecgfeat_robust_20260922/validation.json)。默认 LUDB 回归、默认 ISP/BUT 输出等价、
GUDB 单位/通道/缺失标注协议、孤立非传导 P、增益缩放、极性、强异位激活保留、T 已知真实回归及少导联导出均有检查。
与当前默认保持等价的路径和有意改变结果的路径分别记录。报告使用的最终目录为
`but_release`、`nstdb_release`、`gudb_release`、`ludb_release`；未变化的 P/T 消融来自 `but_final`、`ludb_ablation`、`isp_t_final`。

源码、清单和失败结果保留在工作区；外部数据评估包含源码/数据 SHA256，LUDB 消融保存源码 SHA256，版本不匹配时需要新输出目录。
[本轮相对起点的代码差异](../ecgfeat_robust_20260922/extraction_changes.patch)排除了原工作区中无关的 ecgagent 修改。
尚未解决的重点是严重噪声的可信拒绝、无 P 波候选假阳性、低振幅异位 QRS、非传导 P 的高精度覆盖，以及低幅/融合 T 的波界。
