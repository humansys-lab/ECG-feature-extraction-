<!-- i18n-nav -->
[中文](p_wave_snr_stability_fusion_implementation.md) | [English](p_wave_snr_stability_fusion_implementation.en.md) | [日本語](p_wave_snr_stability_fusion_implementation.ja.md)
<!-- /i18n-nav -->

# P 波局部 SNR、边界稳定度与稳健融合改造

生成时间：2026-07-28

## 改造范围

本次改造保持现有 P candidate、raw/corrected/consensus 三层边界和下游输出
兼容，新增以下确定性 DSP 证据：

1. 对最终 P candidate 生成零相位 `35 Hz` 低通检测视图；幅度、面积和形态
   继续使用原始 measurement signal。
2. 从逐拍经过上一拍 T offset 验证的 TP quiet 窗估计局部噪声。PR/PQ
   含 Ta，不作为噪声窗；首拍的 pre-P fallback 明确标为前一 T 波上下文未知。
   输出：
   - `p_local_noise_rms_mv`
   - `p_local_noise_source`
   - `p_local_snr`
   - `p_informative`
   - `p_tp_gap_ms`
   - `p_quiet_window_available`
   - `p_on_t_overlap_risk`
   - `p_ta_overlap_risk`
   - `p_baseline_method`
   - `p_baseline_confidence`
3. 在低通 P 视图上比较 tangent 与 `4%、6%、8%、10%、12%` 阈值边界，
   用 MAD 构造：
   - `p_onset_sigma_ms`
   - `p_offset_sigma_ms`
   - `p_boundary_stability_method`
4. 当至少 4 个具有局部 SNR/稳定度证据的导联形成有效 cluster 时：
   - onset 使用第二早边界；
   - offset 使用第二晚边界。
5. 若受保护的早 onset 令所有 offset 违反 P-duration 几何约束，则回退到
   cluster median，避免降低完整边界覆盖率。
6. 低于 `2 sigma` 的候选标记 `p_isoelectric_uninformative`。它不参与高置信
   次序统计，但在高质量导联不足时仍可进入旧的低置信 cluster fallback。
7. 高心率下 TP 不足时，不计算伪造的局部 SNR。代码只在 P 候选两侧拟合
   连续的鲁棒二次趋势作为诊断视图，将边界置信度封顶为 `0.45`，并禁止
   second-earliest/second-latest 融合；该趋势不会覆盖原始 P 边界。
8. Ta 风险单独作用于 offset 置信度。P 存在置信度不会因为 Ta/无 TP 被
   直接清零，从而区分“看到了 P”与“无法可靠确定等电位边界”。

## LUDB 全量结果

对 200 条 LUDB 记录运行 `analyze_ludb_p_boundaries.py`：

| 指标 | 改造前 | 改造后 |
|---|---:|---:|
| 完整 P 边界覆盖率 | 91.10% | 91.17% |
| Consensus onset bias | +9.31 ms | +4.95 ms |
| Consensus onset SD | 35.45 ms | 34.33 ms |
| Consensus onset MAE | 25.64 ms | 22.03 ms |
| Consensus offset bias | -9.75 ms | -8.65 ms |
| Consensus offset SD | 26.13 ms | 26.84 ms |
| Consensus offset MAE | 17.32 ms | 17.37 ms |
| Consensus P-duration bias | -19.06 ms | -13.60 ms |
| Consensus P-duration SD | 23.07 ms | 22.35 ms |
| Consensus P-duration MAE | 25.40 ms | 21.31 ms |
| Record-level P-duration bias | -20.48 ms | -15.04 ms |
| Record-level P-duration MAE | 23.53 ms | 18.59 ms |

结果目录：

- `ludb_p_boundary_analysis_snr_fusion/`
- `ludb_cse_evaluation_snr_fusion/`

## CSE 风格代理评估

| 指标 | 改造前 | 改造后 | 说明 |
|---|---:|---:|---|
| P duration 原始 bias | -20.48 ms | -15.04 ms | 改善 5.44 ms |
| P duration 原始 MAE | 23.53 ms | 18.59 ms | 改善 4.94 ms |
| P duration trimmed bias | -22.11 ms | -16.06 ms | 仍未达到 ±10 ms |
| P duration trimmed SD | 13.12 ms | 13.15 ms | 保持在 15 ms 限值内 |
| PR 原始 bias | -5.51 ms | -4.57 ms | 小幅改善 |
| PR 原始 MAE | 10.59 ms | 10.20 ms | 小幅改善 |
| PR 覆盖率 | 92.6% | 92.6% | 未下降，仍是后续缺口 |
| QRS/QT | 不变 | 不变 | 未产生连带回归 |

本次改造显著减轻了多导联融合造成的 P-duration 收缩，但 P offset 的 SD/MAE
没有同步改善，说明下一阶段应优先增加独立的 offset area/quiet-window 方法，
而不是继续仅调整全局次序统计量。

## 验证

- 全测试集：`927 passed, 2 skipped, 145 subtests passed`
- LUDB P 边界：200/200 成功
- LUDB CSE 风格代理：200/200 成功

## Ta / 高心率无 TP 防护追加改造

针对 Ta 和 P-on-T 的生理不可辨识性，追加了以下保护：

1. PR/PQ 段完全退出 P-local noise 候选集。后续拍只有上一拍 T offset 到
   本拍 P onset 之间至少留出 guard 后的 `16 ms` 数据，才标记
   `p_quiet_window_available=True`。
2. TP 不足时 `p_local_snr=None`，不伪造低噪声证据；用 P 两侧 sideband
   拟合连续鲁棒二次趋势，仅作稳定度视图，baseline confidence 为 `0.25`。
3. `p_tp_gap_ms <30 ms` 标记 `p_on_t_overlap_risk`，onset/offset confidence
   上限 `0.45`，并禁止 second-earliest/second-latest 极值融合。
4. 只有具备已验证 TP 基线时，才在 P 后 PR 区检查与 P 极性相反、幅度至少
   `max(10 uV, 2 sigma_noise)` 的 Ta-like excursion；命中时单独压低 offset
   置信度，不清零 P-presence confidence。
5. 极值融合被禁用后若不同 lead 子集形成无效全局几何，只允许使用同一批
   有效 lead 的 onset/offset **成对中位数**恢复覆盖，不恢复极值统计。
6. QRST residual 的 ST-T 相减增加逐拍时移、鲁棒幅值/偏置适配、余弦接缝
   taper 和二阶差分高频能量门控；中位 RR `<520 ms` 时该 residual 路径
   直接不可用，避免模板侵入下一次 P 区域。

相对上一版 SNR fusion 的 LUDB 200 条全量结果：

| 指标 | SNR fusion | Ta/TP guard final |
|---|---:|---:|
| 完整 P 边界覆盖率 | 91.17% | 93.30% |
| Consensus onset bias / SD | +4.95 / 34.33 ms | +4.64 / 35.80 ms |
| Consensus offset bias / SD | -8.65 / 26.84 ms | -8.69 / 27.58 ms |
| Consensus P-duration bias / SD | -13.60 / 22.35 ms | -13.33 / 23.00 ms |
| Record P-duration bias / MAE | -15.04 / 18.59 ms | -15.03 / 18.66 ms |
| P-duration trimmed SD | 13.15 ms | 12.56 ms |
| PR 覆盖率 / MAE | 92.6% / 10.20 ms | 93.1% / 9.85 ms |

安全门控没有解决现有 P-duration 约 `-15 ms` 的系统性收缩；它的收益主要是
避免把 Ta/T-tail 当作“安静基线”，并提高完整边界覆盖。beat-level onset、
offset 和 duration 的 SD/MAE 有轻微退化，说明新增的成对中位数 fallback
扩展了较难样本覆盖；这些难样本必须依靠独立高心率/Ta 标注集继续验证。

最终结果目录：

- `ludb_p_boundary_analysis_ta_tp_guard_final/`
- `ludb_cse_evaluation_ta_tp_guard_final/`
