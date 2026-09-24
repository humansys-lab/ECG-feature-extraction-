<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

# ecgfeat_dxl_inspired

一个 **DXL 风格但不是 DXL 私有实现** 的 12-lead ECG 特征提取库骨架。它按公开手册中的主流程组织：

1. 输入标准化与分析信号构建
2. 波形质量评估
3. 肢体导联反接启发式检测
4. 多导联 QRS 检测
5. beat grouping
6. representative beat 构建
7. 逐搏逐导联波界定位
8. lead/group/global 特征量计算
9. global QT / QTc / axis 输出

## 重要说明

- 这是 **研究/工程脚手架**，不是医疗器械软件。
- 代码实现的是 **DXL-inspired** 方案，不是 1:1 复刻 Philips 私有算法。
- 特别是复杂室上性/室性节律、严重噪声条件下的边界定位，以及 paced beat 专用分支，还需要进一步迭代和验证。

## 安装

```bash
pip install -e .
```

## 发行包与 ECG Record（ecg-records 0.1.0）

本目录是发行包 **`ecg-records`**（导入名 `ecgfeat`）。推荐接口是版本化的 ECG Record：

```python
from ecgfeat import ecg_record, dumps_record, loads_record, query_measurement

# signal: (n_leads, n_samples)，每行有明确名称；默认要求标准 12 导联。
record = ecg_record(signal, sampling_rate=500, lead_names=lead_names)
payload = dumps_record(record)            # 规范 JSON；summary ≤ 24,000 字节（10 s/12 导联/10 搏）
restored = loads_record(payload)          # 默认严格校验
heart_rate = query_measurement(restored, "heart_rate_bpm")
```

- 1–8 个通道需显式 `input_mode="limited"`；读写/查询 Record 不加载数值、绘图或解释代码。
- 解释规则在独立发行包 `ecginterpret`（`pip install "ecg-records[interpret]"`）；
  绘图在本包的 `ecgfeat.viz`，Matplotlib 为可选依赖（`pip install "ecg-records[viz]"`）。
- 下文的 `ECGFeatureExtractor` / `to_dict` 为旧接口：使用时发出 `ECGDeprecationWarning`，
  最早 0.3.0 移除；需要旧对象时请从 `ecgfeat.compat` 导入（不告警）。
- 输出属于研究/工程测量；schema 1.0.0 中所有已发表字段均为 `unvalidated`，不能当作临床结论。
- 迁移状态与门禁结果见 [实现状态](../docs/library_design/08_implementation_status.md)，
  英文包说明见 [README.package.md](README.package.md)。

## 最小示例（旧对象接口）

```python
import numpy as np
from ecgfeat import PatientMeta
from ecgfeat.compat.api_v0 import ECGFeatureExtractor  # legacy object contract (no warning)
from ecgfeat.compat.export_v0 import to_dict

# ecg shape: [12, n_samples], unit: mV
fs = 500
n = 5000
ecg = np.random.randn(12, n) * 0.02

extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
features = extractor.extract(ecg, fs=fs, meta=PatientMeta(age=45, sex="M"))

print(features.global_features)
print(to_dict(features)["metadata"])
```

## 2026-09-22 性能与准确性候选

默认路径已加入质量频谱复用、滤波器系数缓存、P 活动阈值复用、T 小波细节复用。
默认测量通过完整 LUDB 等价回归；详细数据和限制见
[实现与验证报告](../docs/ecgfeat_improvements_20260922.md)。

准确性候选通过独立开关启用，全部关闭是默认配置：

```python
from ecgfeat import ECGFeatureExtractor, RefinementConfig
from ecgfeat.export import prepare_json_export, to_dict

extractor = ECGFeatureExtractor(
    fs_internal=500,
    refinement=RefinementConfig(t_bidirectional=True),  # 研究候选，收益很小
    st_amplitude_source="calibrated_pr",  # 可选 hybrid ST 幅值路径，有覆盖率代价
)
features = extractor.extract(ecg, fs=fs)
# 直接省去最终不需要的逐搏对象复制；结果与先完整导出再裁剪相同。
payload = prepare_json_export(to_dict(features, profile="summary"), profile="summary")
```

`RefinementConfig.experimental()` 启用全部候选，用于消融研究；此前十项候选的完整 LUDB
测试发现组合退化，**不推荐作为生产配置**。`adaptive_pr_tp` 提供 PR/TP
锚点和支持不足拒绝机制，EDB 覆盖率低于 `calibrated_pr`，同样保持实验状态。
可选 ST 路径更新 `st_hybrid_*` 等旁路字段，不等同于替换所有 native ST 或诊断规则。

`ecgfeat.calibration.QTErrorCalibration` 提供按记录加权的 QT 误差风险模型；
本轮拟合模型仅通过 LUDB 内部分组验证，不改变 `qt_confidence` 或报告门控。
JSON-only 批处理在 `--export-profile summary --no-pt` 下自动使用直接导出。

### 传统算法续进与可选编译加速

本轮新增算法只使用传统信号处理和动态规划，不引入深度学习模型、训练或权重。
结果见[传统算法续进报告](../docs/ecgfeat_classical_20260922.md)。

最新的病理 P、抗噪 QRS、少导联契约及 T 终点修复，见
[完整实现与外部验证](../docs/ecgfeat_robust_20260922.md)。双导联可使用以下显式配置：

```python
from ecgfeat import ECGFeatureExtractor, RefinementConfig

extractor = ECGFeatureExtractor(
    fs_internal=500,
    mains_freq=50,  # 按采集环境设定；BUT/NSTDB 评估使用 60 Hz
    input_mode="limited",
    refinement=RefinementConfig(
        qrs_adaptive_consensus=True,
        p_pathology_candidates=True,
        atrial_event_validation=True,
        t_sequence_offset_only=True,
    ),
)
# ecg_channels 为实际提供的通道，不能预先补成 12 行；必须提供非空通道名。
features = extractor.extract(
    ecg_channels, fs=360, amplitude_unit="mV",
    lead_names=["channel_0", "channel_1"],
)
accepted_p_beats = {p.beat_id for p in features.p_wave_assessments if p.accepted}
```

`limited` 支持 1～8 个明确命名、同步采样的通道。已知解剖名称可直接传入，如 `II`、`V1`；
本轮定量性能验证主要覆盖单/双通道，3～8 通道仅属于接口支持范围。
未知名称的内部槽位映射在 `metadata.input_contract.lead_slot_map` 中，槽位名不代表解剖位置。
缺失通道不参与质量计数，也不重建肢体导联；重复、平线、噪声和同步性门控仍生效。
此模式仅输出测量及证据，关闭十二导联诊断、形态规则输出、电轴和正式 QT 报告资格，保留原始区间。
`beat_features.p` 和 `metadata.rhythm_analysis.atrial_events` 仍可能包含假阳性，不能代替 `p_wave_assessments.accepted`。

BUT 全量 P F1 从 80.68% 到 82.36%（150 ms），QRS F1 从 95.31% 到 96.21%（75 ms）；
NSTDB 0 dB 噪声段 QRS F1 从 56.85% 到 76.45%，
GUDB 独立双导联 QRS F1 从 95.92% 到 97.21%（75 ms）。这些是不同数据集、不同任务的结果，不应合并成一个准确率。
标准十二导联仍使用原默认配置；新增算法保持独立开关，尤其不应直接启用所有实验项。

新增 ISP、BUT PDB、NSTDB 的外部评估入口为
`scripts/download_ecgfeat_external.py` 和 `scripts/evaluate_ecgfeat_external.py`。
它们分别验证整合波段定位、病理 P 波峰和电极运动噪声下的 QRS 检测；
标注协议、固定队列、候选比较及限制见[外部数据集报告](../docs/ecgfeat_external_20260922.md)。

新增 INCART、SVDB、PWAVE 的冻结评估见[跨数据集报告](../docs/ecgfeat_cross_dataset_20260922.md)，
入口为 `scripts/download_ecgfeat_cross_dataset.py` 和 `scripts/evaluate_ecgfeat_cross_dataset.py`。
报告分别列出全时长 QRS 检测、固定片段完整提取、全时长 P 波结果、来源重叠排除及运行耗时；
这些数据的峰值标注不能验证 P/T 波界或 QT 精度，增强开关也不保证在所有数据集上改善。

- `p_phasor_candidates`：相量变换补充弱 P 候选，交给既有上下文重选；不单凭相量峰创建缺失 P。
- `t_boundary_projection`：分别面向起点、终点的噪声加权局部投影，需要至少两个有效独立物理导联。
- `t_sequence_selection`：对已有 T 波界候选做跨搏动态规划，缺失波、起搏和节律变化有保护。
- `t_projection_offset_only` / `t_sequence_offset_only`：上述 T 候选的仅终点版本，保留原 T 起点。

这些准确性开关均默认关闭。完整 LUDB 上仅终点序列选择改善幅度较小；
多项候选组合不代表更优配置，且 T 波界变化可能影响后续 P 波评估。
完整 QTDB 上相量 P 候选使 P 起点 MAE 从 49.19 降至 48.66 ms；
T 候选的平均误差虽略降，P95 却从 117.4 升至 120 ms，详细覆盖率与配对结果见报告。

```python
# 研究消融示例；不是新的默认配置。
extractor = ECGFeatureExtractor(
    refinement=RefinementConfig(t_sequence_offset_only=True),
)
```

P 波状态空间基线的 Kalman 递推可通过 Numba 编译原标量运算加速，关闭 fast-math。
安装了 Numba 时自动使用，无此依赖时保持 Python 路径。可选安装：

```bash
pip install -e './feature_extraction[performance]'  # 在仓库根目录运行
```

首次调用存在编译/缓存加载开销，收益应按预热后的批量提取评价。
`ECGFEAT_DISABLE_JIT=1` 可强制使用 Python 参考路径。缓存只保存编译代码，
不会保存患者信号。安装导入与仓库测试导入使用不同的磁盘缓存策略，避免互相污染。

## 当前输出

- `quality`: 每导联质量评分和 flags
- `structured payload`: `signal / quality / beats / groups / global / provenance` 六个稳定分区
- `beats`: 每个 beat 的 R 位置和分组
- `beat_features`: 每个 beat / 每个导联的 P-QRS-T 边界与基础形态量
- `representative_leads`: 每导联 representative 参数和变异度
- `groups`: 节律组摘要
- `global_features`: HR / PR / QRS / QT / QTc / P/QRS/T/ST axis / QT dispersion
- `metadata`: QRS detector 调试信息、record quality、pacing state、representative beat 元信息、QT 可靠导联列表
- `ecgfeat.pediatric_rules`: pediatric Appendix A voltage threshold lookups for RVH/LVH/LSH/BVH morphology evidence; unavailable DXL columns are kept as `None`, and `tests.test_pediatric_rules` validates age-bin and required-key coverage
- Phase 2A measurement note: P / T 的初始峰值搜索现在会先基于多导联候选和 representative prior 做一次 beat-level 融合，再进入现有的单导联 tangent / geometric 边界定位；公开 dataclass / 导出字段名和 `signal / quality / beats / groups / global / provenance` 结构保持不变

## Rhythm Gap Closure Additions

- `atrial.py`: independent atrial-event extraction, inter-beat blocked-P scan, and QRST-window atrial residual scaffold features for AF/AFL rule inputs
- `rhythm_rules.py`: paced-rhythm control flow, preexcitation rules, premature/pause/AV-block evidence, measurement availability, and statement suppression
- `metadata["rhythm_analysis"]`: intermediate atrial, AF/AFL, pacing, rule, and availability results used by `export.py` and `interpret.py`
- `export.py`: exposes rhythm inputs including AF/AFL summaries, pacing context, rule evidence, and `rhythm_inputs.record.availability`

## Availability Semantics

- Raw interval/axis measurements remain in `global_features`
- `rhythm_inputs.record.availability` states whether atrial-rhythm-derived values should be trusted for downstream rhythm logic
- Continuous ventricular/dual pacing, AF/AFL, complete AV block, and AV dissociation suppress PR / P-axis based rhythm decisions
- The pacing policy preserves pacing evidence while suppressing unreliable AV-block, AV-dissociation, PR, and P-axis decisions when pacing makes those measurements unreliable
- `rhythm_inputs.af_afl.qrst_subtraction.available` means validated per-lead QRST template subtraction; when validation is not met, the QRST-window residual is exposed as a scaffold through `scaffold_available`, `method`, `qrst_subtraction_quality`, and `atrial_residual_signal_summary`

## DXL Gap Closure Status

- P0 measurement reliability now includes QRS consensus guarding, wide-QRS QT rescue, ST-J provenance, and long-PR P-onset rescue.
- P-wave morphology exports notched, biphasic, initial, and terminal component facts.
- Pediatric morphology uses age-binned voltage thresholds and exports RVH/LVH/BVH evidence with bypass reasons.
- Rhythm inputs expose statement evidence, pacing failure facts, AF/AFL QRST subtraction validation state, and AV-block evidence.
- Remaining limitations are listed in `docs/philips_*_feature_schema_checklist.md`.

## 验证命令

```bash
python -m pytest tests/test_multilead_pt_fusion.py tests/test_ecgfeat_pipeline.py
python compare_annotations.py --batch-reports --records 1 4 5 8 --out-dir tmp_phase2_multilead_pt --no-visuals
```

输出说明：

- `tmp_phase2_multilead_pt/summary.csv` 会汇总这组 LUDB spot-check 记录的报告差异
- per-record report 可用于快速检查这组样本上是否出现明显增加的 P / T missing 计数

在仓库根目录运行上述命令。完整 LUDB 回归可使用保存的基线结果：

```bash
python scripts/validate_ecgfeat_regression.py \
  --baseline ecgfeat_validation_review_20260907 \
  --out ecgfeat_validation_fixes_20260908/recheck --workers 4
```

该检查要求既有全局字段、匹配波界误差和标注覆盖计数在 `1e-9` 数值容差内一致；新增字段单独验证。失败返回非零退出码，不以平均误差改善抵消某些记录变差。需要本地 LUDB 数据和基线的 `records.jsonl`、`ludb_errors.csv`、`coverage.csv`。

`evaluate_ludb.py` 现在将标注时间转换到提取器的内部采样率，并在 `summary.json` 中同时保存误差和标注覆盖率。覆盖率包含未匹配及未输出的测量；边界覆盖率不等同于 P/T 波事件的灵敏度。

QT dispersion 输出包含不同统计定义：

- `qt_dispersion_ms`：保留既有估计，避免未经验证地改变现有消费者的结果。
- `qt_dispersion_independent_ms`：合格独立导联 QT 的完整极差，不经过聚类截断。
- `qt_dispersion_p90_p10_ms`：同组导联的 P90−P10。
- `qt_dispersion_source`、`qt_dispersion_used_leads`、`qt_dispersion_legacy_excluded_leads`：明确来源、参与导联和旧聚类估计剔除的导联。

ST 振幅提供可选的 PR 锚点通路：

```python
extractor = ECGFeatureExtractor(st_amplitude_source="calibrated_pr")
```

该通路从标定后的原始信号构造基线，使用已测得的安静 PR 段插值，避免两级中值滤波消除持续 ST 偏移。它更新 hybrid ST 字段，保留 native 波界和振幅通路。缺少至少两个合格 PR 锚点的导联被标记为不可用。此方案已通过合成保真测试，仍需独立 ST 数据验证，因此默认继续使用 `st_amplitude_source="analysis"`。该可选模式可能影响使用 hybrid ST 质量与形态的解释规则。

局部 `qt_confidence` 是尚未校准的证据分数，不应当作准确率概率。全局 QT 是否可以报告应查看 `global_features.qt_reportable`；相关语义在 `metadata.confidence_semantics` 中给出。

## 下一步建议

1. 做基于 CSE / LUDB / QTDB 的系统验证
2. 增加 paced beat 专用 delineation 分支
3. 引入更强的多导联 P/T 融合与 U-wave 分离
4. 扩展 Extended Measurements 风格的完整字段集
5. 叠加 rhythm / morphology rule engine
