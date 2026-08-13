<!-- i18n-nav -->
[中文](af_afl_f_F_wave_implementation.md) | [English](af_afl_f_F_wave_implementation.en.md) | [日本語](af_afl_f_F_wave_implementation.ja.md)
<!-- /i18n-nav -->

# AF/AFL f 波与 F 波识别实现说明

Ruleset：`clinical_rules.v2 / 2026.07.9`

## 目标

将 AF/AFL 判断从单纯的 RR 不规则和 P 波缺失，扩展为可审计的多证据判断：

- 房颤：不规则 RR、缺乏一致的传导 P 波，并结合宽带、低组织性的房颤 `f` 波证据。
- 房扑：QRST 相减后，在多个独立导联检出频率一致的窄带、有组织房扑 `F` 波。
- 证据不足或冲突：输出 `atrial_fibrillation_flutter_indeterminate`，不强制二选一。

## 信号处理

1. 对可靠的 I、II、III、V1、V2 导联分别进行连续节律 QRST 模板相减。
   QRS 保持 R 对齐；ST-T 段按逐拍导数相关性平移，并用鲁棒仿射拟合适配
   当前拍幅值/偏置。QRS→ST-T 接口和模板窗两端使用余弦渐变，避免硬接缝。
2. 验证模板相关系数、心搏数、相减后残差能量、高频（二阶差分）能量比和
   可用导联数。残差高频能量相对原波形放大超过 `1.25` 时拒绝验证。
3. 当中位 RR `<520 ms` 时，固定 QRST 模板窗可能侵入下一次 P 区域；
   连续 residual 路径直接标记不可用，不用相减结果寻找 P-on-T。
4. 对每个导联的连续房性残差执行 Welch PSD，而不是拼接不同导联/心搏后做单次 FFT。
5. 在 220–400 bpm 搜索 F 波基频，同时计算：
   - 峰值与房性频带背景比；
   - 基频及谐波功率集中度；
   - 频谱熵；
   - 时域重复性与稳定性。
6. 至少两个导联且至少 40% 可用导联在 ±30 bpm 内一致，才形成 F 波多导联候选。
7. `f` 波使用高频谱熵、低组织性和低时域重复性形成支持分数；至少 60% 可用导联支持才形成多导联一致性。

## 判定策略

| 输出 | 必要证据 |
|---|---|
| `atrial_fibrillation` | RR-CV ≥ 0.15、无一致 P 波或已验证强 f 波否决伪 P 序列、无强 F 波 |
| `atrial_flutter` | 多导联 F 波一致、F 波置信度 ≥ 0.60；最终临床输出还要求 QRST 相减验证通过 |
| `af_afl_indeterminate` | f/F 证据冲突、RR-CV 处于 0.12–0.15 边界、未验证强 f 波与 P 波结果矛盾，或 F 波置信度处于 0.50–0.60 |
| `none` | 不满足上述条件 |

房扑伴可变房室传导时 RR 可以不规则，因此已验证的强多导联 F 波优先于 RR 不规则，不再由 AF 无条件覆盖。

## 输出证据

`rhythm_inputs.af_afl` 和 `metadata.rhythm_analysis` 现在输出：

- `atrial_rhythm_classification`
- `diagnostic_confidence`
- `probable_af` / `probable_flutter`
- `af_afl_indeterminate` / `indeterminate_reasons`
- `f_wave_confidence` / `f_wave_multilead_consensus`
- `F_wave_confidence` / `F_wave_multilead_consensus`
- `F_wave_rate_bpm`
- `flutter_supporting_leads`
- `fibrillatory_supporting_leads`
- 每导联频谱和组织性证据

文本报告的 `RHYTHM SUMMARY` 同步显示分类、RR-CV、P 波比例、f/F 置信度、F 波频率、支持导联和 QRST 验证状态。

## 验证边界

这些修改提高算法可解释性和保守性，但不能仅凭当前弱标签数据宣称达到临床诊断准确率。正式使用前仍需要：

- 专家逐条复核的 AF、典型房扑、非典型房扑和噪声对照集；
- 患者级独立训练/验证划分；
- 灵敏度、特异度、阳性预测值及 AF/AFL 混淆矩阵；
- 对房早、室早、房性心动过速、起搏和肌电噪声的专项压力测试。

此外，逐拍幅值/时移适配只能降低呼吸和 RR 变化造成的模板失配，不能解决
“每一拍 P 都以相同相位埋在 T 内”的不可辨识问题。该场景必须输出 residual
不可用/低置信，不能把模板相减当成已恢复的真实 P 波。
