<!-- i18n-nav -->
[中文](ludb_full_compare_result_analysis.md) | [English](ludb_full_compare_result_analysis.en.md) | [日本語](ludb_full_compare_result_analysis.ja.md)
<!-- /i18n-nav -->

# LUDB Full Compare 结果分析与改进建议

分析对象：`/home/chtmedgemma/projects/ecg_gemma/ludb_full_compare`

分析时间：2026-07-05

## 1. 数据概况

- `summary.csv` 共 200 条记录。
- 每条记录包含 12 导联代表测量对比、全局 interval/axis 对比、beat/缺失/质量计数。
- per-lead comparison 文本共解析到 16,800 行指标，包括 `PR/QRS/QT/QTcB/R/T/ST-J`。

整体上，当前算法的 R peak/HR 基础能力比较稳，主要误差集中在：

1. PR/P 波边界与 PR 可用性。
2. QRS onset/offset，尤其宽 QRS、起搏、terminal slur。
3. T end/QT rescue 或 low-support 路径。
4. QT dispersion 被明显压低。
5. axis 和 per-lead/global comparison 存在评估口径问题。

## 2. 全局指标表现

| 指标 | 有效记录数 | Bias | MAE | Median abs | P75 abs | Max abs | 主要结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| HR | 200 | -0.21 bpm | 0.66 bpm | 0.22 bpm | 0.59 bpm | 29.84 bpm | 大多数很好，少数漏检/半频问题严重 |
| PR | 151 | -11.77 ms | 16.85 ms | 12 ms | 22.75 ms | 88 ms | 系统性偏短，且 48 条算法 PR 缺失 |
| QRS | 200 | +5.73 ms | 9.88 ms | 8.75 ms | 14 ms | 46 ms | 整体偏宽，但 outlier 有偏宽也有偏窄 |
| QT | 200 | -1.09 ms | 13.58 ms | 8 ms | 15 ms | 91.5 ms | 中位表现尚可，rescue/low-support outlier 明显 |
| QTcB | 200 | -1.88 ms | 14.99 ms | 8.78 ms | 15.97 ms | 117.74 ms | 受 QT 和 HR outlier 共同影响 |
| QTcF | 200 | -1.64 ms | 14.41 ms | 8.35 ms | 15.48 ms | 100.83 ms | 同上 |
| P axis | 172 | -10.12 deg | 24.25 deg | 8.88 deg | 21.48 deg | 228.83 deg | 部分是真错误，部分可能是角度环绕比较问题 |
| QRS axis | 199 | +0.78 deg | 12.90 deg | 8.21 deg | 16.48 deg | 88.37 deg | 总体可接受，少数导联/低幅异常 |
| T axis | 150 | -1.13 deg | 16.60 deg | 6.73 deg | 20.50 deg | 352.21 deg | 50 条算法 T axis 缺失；max 多半是 360 度环绕问题 |
| QT dispersion | 190 | -25.35 ms | 27.83 ms | 20.5 ms | 42 ms | 112 ms | 明显低估，是当前最稳定的系统性问题之一 |

超过阈值的记录数：

| 指标 | 阈值 | 超阈值 |
|---|---:|---:|
| HR | > 2 bpm | 9/200 |
| PR | > 20 ms | 44/151 |
| QRS | > 20 ms | 17/200 |
| QT | > 30 ms | 25/200 |
| QTcB | > 30 ms | 28/200 |
| QTcF | > 30 ms | 26/200 |
| P axis | > 30 deg | 36/172 |
| QRS axis | > 30 deg | 21/199 |
| T axis | > 30 deg | 15/150 |
| QT dispersion | > 40 ms | 49/190 |

## 3. Beat 检测与质量计数

算法 beat 数通常比 GT 多：

- 平均 `algorithm_beats - ground_truth_beats = +1.645`。
- 中位数为 `+2`。
- 138/200 条记录正好比 GT 多 2 个 beat。
- 这大概率来自首尾 beat inclusion 规则不同：算法把 10 秒记录边缘的 beat 也纳入，GT 报告/summary 可能只统计完整可测 beat。

这点对 HR 的影响通常不大，因为 HR 多数由 RR 中位数/内部 beat 决定；但会污染：

- `algorithm_total` vs `ground_truth_total`
- missing 计数
- unreliable 计数
- per-record quality 百分比

建议先把评估脚本增加 beat matching/window alignment：

- 全局 HR 可以继续独立评估。
- beat-level missing/unreliable 应只比较匹配到 GT 的 beat。
- 首尾不完整 beat 要么双方都排除，要么单独列为 edge beats。

## 4. P 波与 PR：需要优先改

现象：

- 全局 PR bias = `-11.77 ms`，MAE = `16.85 ms`。
- 44/151 条 PR 误差超过 20 ms。
- per-lead PR 更明显：bias = `-21.72 ms`，MAE = `28.31 ms`。
- PR 误差最大的导联集中在 V3/V4/V5、III、aVR 等，说明低幅/形态复杂 P 波的 onset 更容易被算法放晚。
- 算法全局 PR 缺失 48 条，GT 缺失 25 条。也就是说，算法虽然 beat-level `P missing` 比 GT 少，但 PR 可用性更差，主要是 P 被标记为 unreliable 或 atrial measurement 被抑制。

典型 outlier：

- Record 39：PR `-88 ms`
- Record 57：PR `+86 ms`
- Record 130：PR `-84.5 ms`
- Record 120：PR `-61 ms`

根因判断：

1. P onset 多数偏晚，导致 PR 系统性偏短。
2. 长 PR / 低幅 P / P 与 T 或 flutter-like residual 混淆时，当前可靠性规则容易让全局 PR 变成 N/A。
3. 多导联 P anchor 对 PR 汇总的保护还不够，特别是某些导联 P 波清楚但其他导联不可靠时，全局 PR 会被过度压制。

建议改进：

1. 增强 P onset 的早期 rescue：对已经有稳定 P peak 的 beat，用多导联 atrial anchor 向前寻找一致 onset，而不是只依赖单导联 tangent。
2. PR 汇总不要要求过多导联同时可靠；可采用 limb-lead 优先、V1 辅助的稳健 PR。
3. 对 long PR 场景放宽 P lookback/early-onset 限制，并把 long-PR rescue 的触发条件从单导联改成多导联一致性。
4. 评估时区分 `P detected`、`P reliable`、`PR globally reportable` 三个层级。

## 5. QRS onset/offset：宽 QRS、起搏和 terminal slur 是主要风险

全局 QRS：

- Bias = `+5.73 ms`
- MAE = `9.88 ms`
- 17/200 条误差超过 20 ms。

典型 outlier：

- Record 108：QRS `-46 ms`
- Record 90：QRS `+44.5 ms`
- Record 99：QRS `+35.5 ms`
- Record 5：QRS `+32 ms`
- Record 8：QRS `+28.9 ms`

视觉确认：

- Record 108：算法 QRS offset 在部分导联明显早于 GT，把 terminal QRS/slur 截掉，导致 ST-J 偏差，例如 V1 ST-J diff 达 `+0.59 mV`。
- Record 8：QRS detector confidence 极低，中位约 `0.049`，还有疑似 pacing spike metadata；全局路由进入 `wide_qrs_jt`，QRS/QT 都被牵动。
- Record 74：起搏/低幅 QRS 场景，算法只检测到 5 个 beat，GT 为 9 个，HR 从约 60 bpm 变成约 30 bpm。

根因判断：

1. QRS terminal boundary 对宽 QRS、低幅 terminal slur、ST/T 交界敏感。
2. 起搏和低幅 QRS 下，R detector confidence 很低时仍进入后续测量，容易产生半频或错分组。
3. QRS offset 错误会直接污染 ST-J、JT、QT path、wide_qrs_jt 路径。

建议改进：

1. 对低 QRS detector confidence 的记录增加多导联 fallback：只要多个导联在预期 RR 附近有同步 deflection，就不应漏掉 beat。
2. 起搏场景增加 spike-anchored capture search：从 pacing spike 后固定生理窗口找捕获 QRS，而不是只靠普通 detector。
3. QRS offset consensus 需要引入 terminal slur/ST-J guard：如果 offset 后 `ST40/ST80` 与 J 点差异很大，应该回溯判断是否 offset 太早或太晚。
4. 宽 QRS route 不应只影响 QT/JT，也应反向约束 QRS offset 可信度。

## 6. T end/QT：主路径可用，rescue 和 low-support 路径需要收紧

总体 QT：

- Bias = `-1.09 ms`
- MAE = `13.58 ms`
- 25/200 条误差超过 30 ms。

按 QT source 分组：

| QT source | 记录数 | Bias | MAE | P75 abs | Max abs |
|---|---:|---:|---:|---:|---:|
| reliable_lead_median | 125 | +5.1 ms | 9.6 ms | 11.5 ms | 63 ms |
| low_qt_support_fallback | 36 | -5.9 ms | 20.9 ms | 28.4 ms | 81 ms |
| reliable_raw_qt_cluster_rescue | 25 | -13.8 ms | 16.1 ms | 10.5 ms | 91.5 ms |
| raw_qt_core_rescue | 8 | -27.3 ms | 29.3 ms | 47 ms | 58 ms |
| wide_qrs_low_support_qt_fallback | 4 | -20.6 ms | 28.6 ms | 30.6 ms | 73 ms |

结论很明确：

- `reliable_lead_median` 主路径表现最好。
- `low_qt_support` 和各种 rescue 路径明显更差，并且整体偏短。
- 大 QT outlier 大多带有 `st_t_confusion`、`t_end_fallback`、`beat_unreliable` 或 rescue source。

典型 outlier：

- Record 57：QT `-91.5 ms`。图上可见 T offset 被算法提前截断。
- Record 73：QT `-81 ms`。
- Record 125：QT `-73 ms`。
- Record 81：QT `-72.5 ms`。
- Record 133：QT `-68.5 ms`。
- Record 24：QT `+63 ms`。

建议改进：

1. 在 rescue/low-support QT 路径中增加“不要过早结束”的约束，例如 T peak 后最短 tail、T area tail、跨导联 late-tail support。
2. `st_t_confusion` 出现时，不要直接把早期 trough 当成 T end；优先寻找后续同极性或双相 T 波尾。
3. 对 `t_end_fallback` 的 beat 降低参与全局 QT 的权重，而不是只要通过基本范围就进入全局中位数。
4. wide QRS/JT 路径中，QT 不能只依赖 JT 合理性，还要检查 QRS offset 是否可靠。

## 7. QT dispersion：当前明显低估

QT dispersion：

- Bias = `-25.35 ms`
- MAE = `27.83 ms`
- 49/190 条超过 40 ms。

这是一个很稳定的系统性问题。原因不是单个 T end outlier，而是当前代表导联/consensus 策略倾向把多个导联的 QT 拉到相似值，压缩了导联间差异。

典型 outlier：

- Record 132：`-112 ms`
- Record 29：`-101 ms`
- Record 21：`-99 ms`
- Record 84：`-97 ms`
- Record 52：`-94 ms`

建议改进：

1. QT dispersion 应该用独立 per-lead QT，而不是全局 `qt_consensus_ms` 填到每个导联后再计算。
2. 对每个导联保留 independent QT、consensus QT、used-for-global-QT 三套值。
3. dispersion 计算时只过滤明显无效导联，不要把所有导联强制对齐到全局 T end。
4. 报告中明确写出 dispersion source：`independent_lead_qt` 或 `consensus_qt`。

## 8. Axis：算法问题和评估问题混在一起

Axis 指标需要谨慎解读：

- P axis MAE = `24.25 deg`，最大 diff = `228.83 deg`。
- T axis MAE = `16.60 deg`，最大 diff = `352.21 deg`。
- 这些最大值很可能包含角度环绕问题，例如 179 deg 与 -179 deg 直接相减会得到 358 deg，但临床角度差应是 2 deg。

同时，算法 T axis 缺失 50 条，而 GT T axis 没有缺失。这是算法可用性问题。

建议改进：

1. comparison 脚本计算 axis diff 时必须使用 circular angular difference：`min(abs(a-b), 360-abs(a-b))`。
2. T axis 汇总不应过度依赖所有导联 T 都可靠；只要主要 limb leads 有可用 T amplitude，就应能给出低置信但可报告的 T axis。
3. P axis 大 outlier 需要在修正 circular diff 后重新排序，再判断是真 P 波方向错误还是比较口径错误。

## 9. ST-J/T/R amplitude

per-lead 振幅总体表现较好：

| 指标 | Bias | MAE | P75 abs | Max abs |
|---|---:|---:|---:|---:|
| R amplitude | -0.005 mV | 0.023 mV | 0.03 mV | 0.51 mV |
| T amplitude | +0.003 mV | 0.035 mV | 0.03 mV | 0.81 mV |
| ST-J | -0.004 mV | 0.029 mV | 0.03 mV | 0.68 mV |

这些指标的中位表现可以，但 outlier 明显：

- Record 116：多个 limb leads 出现 T/ST-J 极性或基线级别大偏差。
- Record 108：ST-J 大偏差来自 QRS offset 截短。
- Record 35/116：T amplitude 大 outlier 多半是 T polarity/选择错误。

建议改进：

1. ST-J outlier 优先从 QRS offset 和 baseline 修，而不是单独调 ST-J 采样。
2. T amplitude outlier 应和 T polarity/T candidate 选择一起修。
3. 对 ST-J 超过生理范围且 QRS terminal confidence 低的 beat，自动降权或标记 measurement unreliable。

## 10. 报告/比较脚本本身需要改

当前 comparison 文本存在几个口径混用问题，会影响人工判断：

1. per-lead 表里很多 QRS/QT 值在 12 导联完全相同，说明它比较的可能是 consensus/global 值，而不是真正 independent per-lead 值。
2. 有些记录 global measurement 和 per-lead comparison 不一致：
   - Record 8：global QRS 为 `144.4 ms`，per-lead 表中 QRS 为 `204 ms`。
   - Record 109：global QT 为 `305 ms`，per-lead 表中 QT 为 `492 ms`。
3. axis diff 没有做 circular normalization，导致部分 outlier 被夸大。

建议先修 comparison/report harness：

1. per-lead 表明确分列：
   - `raw_independent`
   - `lead_representative`
   - `consensus`
   - `global_used`
2. summary 中 global 指标只和同一 source 的 GT global 指标比较。
3. per-lead 指标只和 per-lead GT 比较，不要把 global consensus 复制到每个 lead 后当 per-lead measurement。
4. axis diff 使用 circular distance。

## 11. 改进优先级

### P0：先修评估口径

这些不修，后续算法调参会被误导：

1. axis diff 使用 circular difference。
2. beat-level comparison 做 beat matching，单独处理首尾 edge beat。
3. per-lead/global source 分离，避免 consensus 值重复比较到每个 lead。
4. QT dispersion 改用 independent per-lead QT 重新评估。

### P1：修起搏/低幅 QRS detector

目标病例：

- 74
- 8
- 83
- 99
- 110

目标：

- 消除 HR 半频/漏检。
- QRS detector confidence 极低时进入多导联 fallback。
- 起搏 spike 后建立 capture QRS 搜索窗口。

### P2：修 QRS terminal boundary

目标病例：

- 108
- 90
- 5
- 23
- 170

目标：

- 降低 QRS >20 ms outlier。
- 减少 ST-J 因 QRS offset 错误产生的大偏差。
- 让 wide_qrs_jt 路径依赖可靠的 QRS offset。

### P3：修 T end rescue/low-support QT

目标病例：

- 57
- 73
- 125
- 81
- 133
- 24

目标：

- 把 low-support/rescue QT MAE 从约 20-29 ms 降到接近主路径的 10-12 ms。
- 减少 QT >30 ms outlier。

### P4：修 P onset/PR 可用性

目标病例：

- 39
- 57
- 130
- 120
- 64

目标：

- PR bias 从约 `-12 ms` 拉回接近 0。
- 减少算法 PR 缺失记录。
- 提升 long PR / low amplitude P 的 onset 稳定性。

## 12. 建议的回归评估清单

每次改算法后，建议固定追踪：

1. `summary.csv` 全局 MAE/Bias。
2. PR >20 ms 记录数。
3. QRS >20 ms 记录数。
4. QT >30 ms 记录数。
5. QT dispersion bias。
6. HR >2 bpm 记录数，特别是 record 74。
7. low-support/rescue QT source 的 MAE。
8. `algorithm_pr`、`algorithm_t_axis` 缺失数量。
9. `st_t_confusion`、`t_end_fallback`、`qrs_terminal_classification_only` flag 计数。
10. Record 57/74/108/8/109/116 的对比图人工复核。

## 13. 当前结论

当前算法的主干已经能稳定完成多数普通 LUDB 记录的 HR/QRS/QT 测量。最值得先投入的是评估口径和边界 outlier，而不是整体重写。

短期最划算的路线是：

1. 先修 comparison/report harness。
2. 再修起搏/低幅 QRS 漏检。
3. 然后修 QRS terminal boundary。
4. 接着修 T end rescue/low-support。
5. 最后再系统调 P onset/PR 和 axis 可用性。

