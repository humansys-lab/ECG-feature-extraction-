<!-- i18n-nav -->
[中文](ecgfeat_neurokit_hybrid_improvement.md) | [English](ecgfeat_neurokit_hybrid_improvement.en.md) | [日本語](ecgfeat_neurokit_hybrid_improvement.ja.md)
<!-- /i18n-nav -->

# ecgfeat 吸收 NeuroKit2 优点的波形检测改进方案

> 状态：R/P/T/S 定位与稳健 ST/J 测量旁路已实现，原测量与解释层保持不变  
> 数据集：LUDB 1.0.1，500 Hz，12 导联，10 秒  
> 适用范围：R、P、T、S 波定位，以及 ST/J 点、幅值、趋势和凹凸形态测量  
> 核心约束：新增定位结果先作为旁路字段，不直接覆盖现有特征计算和解释层输入

## 1. 目标

ecgfeat 的优势是多导联心搏共识、完整 QRS/P/T 边界、代表波形和临床
特征层。NeuroKit2 的部分算法在单导联局部峰值定位上更准确，尤其是
R 峰和基于 prominence 的 T 峰定位。

本方案不是用 NeuroKit2 替换 ecgfeat，而是组合两者的优势：

1. ecgfeat 继续决定心搏是否存在、心搏分组、QRS 大致范围和最终特征层级；
2. 在每个导联的受限生理窗口内，使用正负双极性 prominence 细化峰值；
3. 分开保存“形态学峰位置”和“用于幅值测量的位置”；
4. 使用多导联时间共识过滤单导联噪声；
5. 在充分验证前，不改变间期、幅值、代表导联和解释层结果。

## 2. 当前对比结论

### 2.1 R 峰

LUDB 前 10 条记录、全部 12 导联：

| 方法或标记来源 | QRS F1 | R 峰 MAE | 中位绝对误差 | P95 绝对误差 |
|---|---:|---:|---:|---:|
| ecgfeat 原多导联全局 R | 0.8852 | 13.42 ms | 6 ms | 44 ms |
| ecgfeat 原逐导联 fiducial | 0.8917 | 11.83 ms | 4 ms | 46 ms |
| ecgfeat 混合 prominence | **0.8917** | **9.71 ms** | **2 ms** | 50 ms |
| NeuroKit2 | 约 0.819 | **4.57 ms** | 2 ms | 14 ms |
| BioSPPy | 约 0.873 | 3.9 ms | — | — |

结论：

- ecgfeat 多导联策略具有较高灵敏度，但共享全局位置不适合直接评价每个导联的 R 峰；
- NeuroKit2 的逐导联局部校正具有更好的 R 峰定位精度；
- 已实现的混合方案在不改变心搏数量的条件下，将 ecgfeat R 峰 MAE
  从 13.42 ms 降至 9.71 ms；
- 少数负向或复杂 QRS 仍存在长尾误差，不能只看平均 MAE。

### 2.2 T 波

| 方法 | T 波 F1 | T 峰 MAE | T onset MAE | T offset MAE |
|---|---:|---:|---:|---:|
| ecgfeat | **0.8324** | 8.68 ms | 30.81 ms | 16.60 ms |
| ecgfeat 混合 T prominence | **0.8333** | **7.25 ms** | 30.91 ms | 16.79 ms |
| NeuroKit2 DWT | 0.6925 | 26.19 ms | 53.97 ms | 32.12 ms |
| NeuroKit2 prominence | 0.8076 | **8.13 ms** | 25.49 ms | 20.92 ms |

结论：

- 原对比中的 NeuroKit2 T 波使用 DWT，低估了 NeuroKit2 对异常 T 波的能力；
- prominence 同时搜索局部极大值和局部极小值，能够处理负向 T 波；
- ecgfeat 的 T 波检出 F1 更高，NeuroKit2 prominence 的 T 峰定位略准；
- 已实现的 polarity-aware prominence 旁路将 ecgfeat T 峰 MAE 从
  8.68 ms 降至 7.25 ms，P95 从 78.0 ms 降至 58.6 ms；
- T onset/offset 仍使用原生边界，尚未由旁路覆盖。

### 2.3 P 波

| NeuroKit2 描记方法 | P 波 F1 | P 峰 MAE | 备注 |
|---|---:|---:|---|
| DWT | **0.7110** | 30.93 ms | 当前工作方法中 F1 最好 |
| prominence | 0.707 | **17.4 ms** | 1/120 导联运行失败 |
| peak | 0.677 | 18.0 ms | 13/120 导联运行失败 |

ecgfeat 在相同前 10 条记录上的 P 波 F1 为 0.7469。

结论：

- NeuroKit2 的 T 波 prominence 分支同时使用正峰和负峰；
- 但其 P 波 prominence 分支只搜索局部极大值；
- `peak`、DWT、CWT 也存在正向形态偏好；
- 因此更换 NeuroKit2 内置描记方法不能完整解决倒置、双相、逆行、
  低幅、缺失或多个 P 波的问题；
- 原先 NeuroKit2 P 波约 0.711 的结果没有像 T 波一样被明显低估。

### 2.4 S 波

LUDB 只提供 P、整个 QRS 和 T 波的起点、代表峰与终点，不提供独立
Q 峰或 S 峰专家标注。因此不能使用 LUDB 直接计算 S 峰 F1/MAE，
也不能把 QRS offset 当作 S 峰真值。

当前算法语义不同：

| 内容 | ecgfeat | NeuroKit2 |
|---|---|---|
| 搜索范围 | 本导联 QRS onset–offset | R 后心搏区间；prominence 约束在 Q 后 180 ms |
| 主要选择规则 | 正向 R 后最深偏转 | R 后第一个负向局部极小值 |
| S 幅值 | `s_amp_mv` | 默认不输出 |
| S 峰位置 | 未独立保存 | `ECG_S_Peaks` |
| R′/S′/切迹 | 有部分测量 | 无显式分类 |
| 多导联约束 | QRS 范围受多导联结果影响 | 单导联独立 |

这两个定义分别更适合不同目标：

- “第一个有效负向偏转”更接近形态学 S 峰；
- “QRS 内最深负向偏转”更适合临床电压测量；
- 在 RSR′、QS、qR、宽 QRS 和起搏 QRS 中，二者不能混为一个位置。

## 3. 总体架构

建议把增强定位放在主描记完成后、全局特征和解释层计算前：

```text
原始 12 导联 ECG
        │
        ▼
ecgfeat 预处理与多导联 QRS 检测
        │
        ▼
现有逐导联 P/QRS/T 描记与特征计算
        │
        ├── 原字段：继续供间期、幅值、代表波形、解释层使用
        │
        ▼
混合局部定位旁路
        ├── R：正向 prominence，必要时负向回退
        ├── T：正负双极性 prominence
        ├── P：正负双极性、多候选、允许缺失
        ├── S：形态学 S 与幅值 S 分离
        └── ST：PR 基线、J 点局部稳定、多导联共识和窗口中位数
        │
        ▼
独立定位、极性、形态和置信度字段
        │
        ▼
可视化与独立评价
```

只有当旁路结果在独立验证集和测试集上均通过验收，才考虑按具体字段
逐项进入测量层；不能一次性替换全部原始结果。

## 4. R 峰混合定位

### 4.1 已实现

实现位置：

- [`r_localization.py`](../feature_extraction/ecgfeat/r_localization.py)
- [`models.py`](../feature_extraction/ecgfeat/models.py)
- [`api.py`](../feature_extraction/ecgfeat/api.py)

当前规则：

1. 多导联 `global_r` 只负责确定心搏是否存在；
2. 每个导联在 `global_r ± 90 ms` 内搜索；
3. 再与原逐导联 fiducial `±35 ms` 相交；
4. 再限制到现有 QRS onset/offset 附近；
5. 选择 prominence 最大的正向局部峰；
6. 没有正向局部峰时才使用负向峰；
7. 无有效候选时回退到原 `qrs.peak`。

独立字段：

```text
r_localized_index
r_localization_method
r_localization_confidence
r_localization_prominence_mv
r_localization_polarity
```

### 4.2 后续改进

当前 P95 误差仍为 50 ms，主要需要处理：

- aVR、V1 和部分 III 导联的负向主导 QRS；
- QS 型和 rS 型复合波；
- R′ 比 R 更高的束支传导阻滞；
- 边界拍和低质量导联；
- LUDB 代表峰定义与临床正向 R 定义不一致的情况。

建议新增 QRS 形态先验：

```text
positive-dominant
negative-dominant
QS
rS
RS
RSR-prime
paced
uncertain
```

不同形态使用不同的候选评分，避免所有导联固定优先正峰。

## 5. T 波混合定位与异常形态

### 5.1 借鉴 NeuroKit2 的部分

NeuroKit2 prominence 的有效设计包括：

1. 同时搜索局部极大值和局部极小值；
2. 对正负峰分别计算 prominence；
3. 如果 T 峰是负向谷值，先将信号反相再计算 onset/offset；
4. 使用 S 波和 RR 区间限制 T 波候选范围。

### 5.2 已实现的旁路

新增独立结果：

```text
t_localized_index
t_localized_onset
t_localized_offset
t_localization_confidence
t_localization_prominence_mv
t_localization_polarity
t_localization_morphology
t_localization_secondary_index
```

其中 `t_localization_morphology` 支持：

```text
positive
negative
biphasic-positive-negative
biphasic-negative-positive
uncertain
```

候选窗口应由以下条件共同确定：

- 不早于可靠 QRS offset/ST 段之后；
- 不超过下一次 R 峰前的安全边界；
- 根据当前 RR 动态调整，而不是只使用固定时间；
- 优先落在现有 ecgfeat T 波范围或其有限扩展范围内。

当前候选评分包含 prominence、噪声和动态范围；后续仍建议加入：

```text
score =
    prominence_score
  + timing_score
  + width_score
  + beat_template_score
  + multilead_time_consensus
  - noise_penalty
  - u_wave_penalty
```

双相 T 波不能只保留最大的一瓣。需要保存主峰、次峰、两瓣顺序及两者
相对幅度，避免把 T2 或 U 波当成单一 T 峰。

## 6. P 波双极性和异常节律处理

### 6.1 不能直接复制 NeuroKit2 P 波实现

NeuroKit2 prominence 的 P 波只使用正向局部极大值，不能可靠处理：

- aVR 等导联的负向 P 波；
- 异位房性节律导致的 P 波倒置；
- 双相 P 波；
- 低幅 P 波；
- 房颤中的 P 波缺失；
- 房扑或房室传导阻滞中的多个房性波；
- QRS 后出现的逆行 P 波；
- 超出固定 PR 搜索范围的长 PR。

### 6.2 已实现的安全旁路

新增字段：

```text
p_localized_index
p_localized_onset
p_localized_offset
p_localization_confidence
p_localization_prominence_mv
p_localization_polarity
p_localization_morphology
p_localization_candidate_count
p_localization_absent_probability
p_localization_retrograde
```

当前检测逻辑：

1. 在当前 QRS onset 前建立正常 P 波窗口；
2. 对起搏心搏或疑似逆行传导，在 QRS 后建立第二搜索窗口；
3. 同时搜索正向和负向 prominence；
4. 保存显著候选数量，允许明确的无候选结果；
5. 已有原生 P 峰时，只分析其附近 40 ms 的候选并保持原生峰位置；
6. 对双相候选保留两瓣，而不是只选最高正峰。

尚待实现的第二阶段包括 PR/PP 一致性、模板相关性、多导联时间支持，
以及房颤/房扑状态下的一对一 P-QRS 假设调整。

LUDB 试验表明，直接用最大 prominence 覆盖原 P 峰会显著增大峰值
MAE；因此当前实现保留已有原生 P 峰，只用双极性候选标注极性和形态。
原生 P 缺失时的候选仍保存在旁路字段，但尚不进入正式评价或解释层。

导联极性不能单独决定是否异常。例如 aVR 中负向 P 常为正常表现，
需要结合导联方向、多导联 P 轴和节律上下文判断。

## 7. S 波位置、幅值和形态分离

### 7.1 当前 ecgfeat 需要明确的语义

当前 `s_amp_mv` 是本导联 QRS 内真实正向 R 之后的最低值，适合电压
测量，但存在以下问题：

- 没有保存对应的样本位置；
- RSR′ 中最深谷值不一定是第一个形态学 S；
- R 后没有负向偏转时可能得到正的 `s_amp_mv`；
- QS 型复合波不应被强制拆分为 R 和 S；
- S 幅值路径和 S 持续时间路径使用的 R 分割点不完全一致。

### 7.2 已实现的旁路字段

```text
s_peak_index
s_amplitude_index
s_localization_confidence
s_localization_prominence_mv
s_localization_morphology
s_prime_peak_index
qs_nadir_index
```

字段语义：

- `s_peak_index`：R 后第一个满足幅度、prominence 和宽度要求的负向偏转；
- `s_amplitude_index`：QRS 内用于 S 电压测量的最深负向偏转；
- `s_prime_peak_index`：R′ 后的后续负向偏转；
- `qs_nadir_index`：无真实正向 R 时的 QS 最低点；
- 原有 `s_amp_mv` 当前不被旁路修改；以后若接入测量层，没有负向 S
  时应改为 `0` 或 `None`，不能为正值。

### 7.3 当前检测流程

1. 使用现有逐导联 QRS onset/offset；
2. 使用真实 `r_peak_index`，而不是全局 R 或负向主导 fiducial；
3. 找出 R 后全部局部极小值并计算 prominence；
4. 排除小于噪声门限的切迹；
5. 第一个有效负峰作为形态学 S；
6. 最深有效负峰用于 S 幅值；
7. 如果出现 R′，对 R′ 后负峰单独标记 S′；
8. 没有跨越基线时标记 `no-S`；
9. 没有有效正向 R 时进入 `QS` 分支。

## 8. ST/J 点稳健测量

### 8.1 原算法仍保留的路径

原生 ST 路径继续以逐导联 `qrs.offset` 为 J 点，计算 `st_on_mv`、
`st_mid_mv`、`st_80ms_mv` 和 `st_slope_mv_per_ms`。当 J 点与
ST40/ST80 出现明显 QRS 尾部跳变时，原有 raw remeasurement pass
会使用同一心搏的多导联 QRS offset 中位数重新采样。

这条路径仍是当前解释层和兼容性导出的输入。

### 8.2 新增的稳健旁路

实现位置：

- [`st_localization.py`](../feature_extraction/ecgfeat/st_localization.py)
- [`models.py`](../feature_extraction/ecgfeat/models.py)
- [`features.py`](../feature_extraction/ecgfeat/features.py)
- [`export.py`](../feature_extraction/ecgfeat/export.py)
- [`api.py`](../feature_extraction/ecgfeat/api.py)

新流程：

1. 基线优先取 `P offset + 8 ms` 到 `QRS onset - 8 ms` 的真正 PR 段；
2. PR 段不足时，依次回退到 QRS 前安静窗和旧式 R 前窗口；
3. 每个心搏使用可靠 QRS 导联建立 J 点中位数共识；
4. 每个导联仍保留本导联 QRS offset，在本地 ±30 ms 范围寻找
   “QRS 高斜率向 ST 低斜率过渡”的稳定点；
5. J、J+40 ms、J+80 ms 不再使用单个样本，而使用约 ±6 ms
   窗口中位数；
6. 使用 J+12 ms 到 J+80 ms、且不越过 T onset 的稳健线性拟合
   计算 ST slope；
7. 趋势分为 `upsloping`、`horizontal`、`downsloping`；
8. 二阶幅值变化单独分为 `straight`、`concave-up`、`concave-down`；
9. 起搏、QRS 不可靠、T 波过早重叠、基线或总体置信度不足时保留数值，
   但把旁路标为不可靠。

主要字段：

```text
st_hybrid_j_index
st_hybrid_j_mv
st_hybrid_40ms_mv
st_hybrid_80ms_mv
st_hybrid_mean_mv
st_hybrid_area_mv_ms
st_hybrid_slope_mv_per_ms
st_hybrid_trend
st_hybrid_shape
st_hybrid_baseline_mv
st_hybrid_baseline_source
st_hybrid_baseline_confidence
st_hybrid_j_method
st_hybrid_j_confidence
st_hybrid_consensus_index
st_hybrid_consensus_support
st_hybrid_noise_mv
st_hybrid_reliable
st_hybrid_unreliable_reason
```

代表导联的稳健 ST 优先汇总同组可靠心搏的中位数，不优先依赖一个
medoid 心搏。结构化导出位于：

```text
morphology_inputs.leads.<lead>.st.hybrid_robust
```

由于 LUDB 没有 ST elevation/depression 或 J 点专家真值，本阶段不能
用 LUDB 给出 ST 灵敏度、特异度或 MAE。新旁路默认开启，但不会覆盖
原生 ST 字段，也不会进入缺血、早复极或心包炎解释规则。

### 8.3 LUDB 前 10 条内部验证

全部 12 导联共得到 1,188 个 lead × beat：

| 检查项 | 结果 |
|---|---:|
| 通过稳健 ST 可靠性门控 | 1,044 / 1,188（87.9%） |
| T 波过早与 J+80 窗口重叠 | 126 |
| 本导联 J 与多导联共识差异过大 | 18 |
| 使用 P 后真正 PR 段基线 | 759 |
| 使用 QRS 前回退基线 | 429 |
| J 点相对原有效 anchor 的绝对偏移中位数 | 0 ms |
| J 点绝对偏移 P95 | 10 ms |
| J 点最大绝对偏移 | 26 ms |

开启和关闭 `enable_hybrid_st_measurement` 对 LUDB 记录 1 的
`global_features`、最终 `interpretation`、全部原生 beat 字段及全部
原生 representative 字段逐项一致。新字段本身以及对应 metadata
当然只在开关开启时存在。

这里的偏移分布仅用于检查算法是否发生不受控的 J 点跳变，不能解释为
相对专家真值的 J 点误差。

## 9. 安全接入策略

第一阶段只允许旁路字段用于：

- 可视化；
- LUDB 或其他数据集的独立评价；
- 调试信息；
- 置信度和错误案例分析。

元数据应明确记录：

```json
{
  "wave_localization": {
    "r": {"enabled": true, "affects_measurements": false},
    "p": {"enabled": false, "affects_measurements": false},
    "t": {"enabled": false, "affects_measurements": false},
    "s": {"enabled": false, "affects_measurements": false},
    "affects_interpretation": false
  },
  "st_measurement": {
    "enabled": true,
    "method": "robust_pr_baseline_local_settling_multilead_consensus",
    "affects_native_measurements": false,
    "affects_interpretation": false
  }
}
```

在这个阶段必须保证以下内容与关闭旁路时完全一致：

- `global_features`
- `representative_leads`
- `beats` 和 `groups`
- 原有 `qrs.peak` 与 P/QRS/T 边界
- 心率、PR、QRS、QT/QTc
- 电轴、电压和 ST-T 特征
- 最终解释层输出

## 10. 公平评价要求

### 10.1 必须注明完整配置

不能只写“NeuroKit2”，至少应注明：

```text
NeuroKit2 (R detector: neurokit, delineation: dwt)
NeuroKit2 (R detector: neurokit, delineation: prominence)
ecgfeat (QRS marker: global)
ecgfeat (QRS marker: lead-fiducial)
ecgfeat (QRS marker: hybrid-prominence)
```

### 10.2 防止测试集挑选算法

不同方法应先在验证集选择，随后固定参数，在独立测试集一次性报告。
不能看到测试集结果后，再为每个波形选择最好的方法作为唯一成绩。

### 10.3 指标

每个波形至少报告：

- sensitivity、precision、F1；
- 峰值 bias、MAE、中位绝对误差、P95；
- onset/offset MAE；
- 失败导联数；
- 正常、倒置、双相、低幅、宽 QRS、起搏等分层结果；
- 逐导联和逐记录结果。

S 波必须使用具有独立 S 峰定义的专家标注。若没有这样的真值，只能
报告内部一致性、形态学案例和人工复核结果，不能报告伪造的 S 峰 F1。

## 11. 测试要求

### 11.1 合成波形单元测试

至少覆盖：

- 正常 qRS；
- rS、RS、qR；
- QS；
- RSR′、rSr′；
- 宽 QRS 和起搏 QRS；
- 正向、倒置和双相 P/T；
- 无 P、无 S；
- 低幅波、基线漂移和高频噪声；
- 记录开头和结尾的不完整心搏。

### 11.2 回归测试

每次旁路改动都必须验证：

1. 关闭旁路时结果与历史版本一致；
2. 开启旁路时，除新字段外原有序列化结果一致；
3. 解释层结果逐字段一致；
4. 可视化正确使用新位置，但心搏数量仍来自多导联检测；
5. 异常或缺失候选时能够安全回退。

## 12. 推荐实施顺序

1. **R 峰：已完成。** 继续处理负向主导和长尾案例。
2. **T 波：旁路已完成。** LUDB 前10条的峰值 MAE 已改善，下一步处理
   U 波混淆和双相边界。
3. **P 波：双极性形态旁路已完成。** 原生峰保持不动，下一步加入
   多导联共识后再验证缺失 P 救援。
4. **S 波：形态旁路已完成。** 已分离第一个 S、最深幅值点、S′、
   QS 和 no-S；仍需独立 S 峰人工真值。
5. **ST：稳健测量旁路已完成。** 已加入 PR 基线、局部 J settling、
   多导联共识、多点中位数、趋势/凹凸形态和可靠性门控；下一步需要
   带 J 点与 ST 诊断真值的数据集。
6. **解释层接入：最后实施。** 只有独立测试集和回归测试均通过后，
   才允许按具体字段逐项替换原测量值。

## 13. 当前建议

短期内最有价值且风险最低的组合是：

```text
心搏存在与分组       → ecgfeat 多导联共识
R 峰显示与评价       → 已实现的逐导联 hybrid-prominence
T 峰位置与极性       → 已实现正负双极性 prominence 旁路
P 波                 → 原生峰不动，已增加双极性形态旁路
S 波电压             → 保留 ecgfeat 的 QRS 内最深负向测量
S 波形态位置         → 已增加第一个有效负峰和 QS/no-S 分类
ST/J 稳健测量        → 使用新旁路做研究导出，原生 ST 继续供解释层使用
最终临床解释         → 暂时完全保持现状
```

这种设计吸收了 NeuroKit2 在局部 prominence 定位上的优势，同时保留
ecgfeat 在多导联一致性、复杂形态、临床测量和解释层方面的结构优势。
