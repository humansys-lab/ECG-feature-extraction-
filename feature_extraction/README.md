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

## 最小示例

```python
import numpy as np
from ecgfeat import ECGFeatureExtractor, PatientMeta, to_dict

# ecg shape: [12, n_samples], unit: mV
fs = 500
n = 5000
ecg = np.random.randn(12, n) * 0.02

extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
features = extractor.extract(ecg, fs=fs, meta=PatientMeta(age=45, sex="M"))

print(features.global_features)
print(to_dict(features)["metadata"])
```

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
.venv_report_regen_20260625/bin/python -m unittest tests.test_multilead_pt_fusion tests.test_ecgfeat_pipeline -v
.venv_report_regen_20260625/bin/python compare_annotations.py --batch-reports --records 1 4 5 8 --out-dir tmp_phase2_multilead_pt --no-visuals
```

输出说明：

- `tmp_phase2_multilead_pt/summary.csv` 会汇总这组 LUDB spot-check 记录的报告差异
- per-record report 可用于快速检查这组样本上是否出现明显增加的 P / T missing 计数

## 下一步建议

1. 做基于 CSE / LUDB / QTDB 的系统验证
2. 增加 paced beat 专用 delineation 分支
3. 引入更强的多导联 P/T 融合与 U-wave 分离
4. 扩展 Extended Measurements 风格的完整字段集
5. 叠加 rhythm / morphology rule engine
