<!-- i18n-nav -->
[中文](p_wave_rerun_result_analysis.md) | [English](p_wave_rerun_result_analysis.en.md) | [日本語](p_wave_rerun_result_analysis.ja.md)
<!-- /i18n-nav -->

# P 波相关重跑结果分析

分析对象：`ludb_full_compare` 当前重跑结果。

结果时间线：

- `ludb_full_compare/summary.csv` 更新时间：2026-07-05 16:55。
- 200 条 `*_features.json` 均已生成。
- 新增字段已进入结果，例如 `p_offset_consensus_index`、`p_dur_consensus_ms`、`pr_segment_consensus_ms`、`p_boundary_consensus_source`。

## 1. 总体结论

这次 P 波相关修改已经生效，尤其是对一批 **raw limb PR 被靠近 QRS 的假 P 拉短** 的病例有明显帮助。但全库 PR 总体指标没有显著变好，说明第一阶段的保守 consensus 只解决了部分场景，剩余 outlier 仍然集中在 P onset cluster 选择、PR core rescue 过度、P axis/P morphology 不稳定等问题上。

最重要的观察：

1. **PR consensus gating 是有价值的。**
   当前 actual global PR 的 MAE 为 `17.02 ms`；如果用当前 features 反算 raw limb median baseline，MAE 约为 `26.82 ms`。也就是说，全局聚合/救援路径整体确实在救一批病例。

2. **但当前全库 PR 指标仍有轻微负偏。**
   当前 PR bias 为 `-12.02 ms`，median diff 为 `-10 ms`，说明 P onset 仍然整体偏晚。

3. **P offset consensus 覆盖率不错，但还不应直接替代 raw morphology。**
   beat-level `p_offset_consensus_index` 覆盖约 `59.91%` 的 lead-beat rows；representative-level P duration consensus median 约 `80 ms`。但仍有短/长 outlier，例如 record 57 的 consensus P duration 可到 `152 ms`。

4. **P axis 仍是明显短板。**
   P axis MAE 约 `24.25°`，有 14 条记录绝对误差超过 `90°`。这部分主要不是 PR consensus 能解决的，而是 P polarity、P signed area、低幅 P 和 lead 支持度的问题。

5. **高 P missing 记录仍然很多。**
   例如 record 104、45、111、90、34、74 的 algorithm P missing rate 均很高。这些更像是 P 存在性/低幅 P/节律上下文问题，而不是单纯 onset/offset 边界问题。

## 2. 全局 PR 统计

当前 `summary.csv` 中 PR 指标：

| 指标 | 数值 |
|---|---:|
| 可比较记录数 | 151 |
| bias | `-12.02 ms` |
| MAE | `17.02 ms` |
| median diff | `-10.0 ms` |
| abs p75 | `24.25 ms` |
| abs p90 | `39.0 ms` |
| max abs | `86.0 ms` |
| `abs(diff_pr) > 20 ms` | 47 |
| `abs(diff_pr) > 40 ms` | 13 |
| `abs(diff_pr) > 60 ms` | 4 |
| algorithm PR missing | 48 |
| GT PR missing | 25 |

与第一阶段修改前文档中的旧统计相比：

- 旧 PR MAE 约 `16.85 ms`，当前 `17.02 ms`，整体持平略差。
- 旧 `abs(diff_pr) > 20 ms` 为 44 条，当前 47 条。
- 旧 max abs 为 `88 ms`，当前 `86 ms`。

所以这次修改不是全局一刀切改善，而是把某些严重负偏 outlier 拉回来了，同时仍有其他 P 相关问题没有解决。

## 3. PR consensus 的实际效果

### 3.1 反算 raw limb baseline

用当前 `representative_leads` 反算：

| 路径 | n | bias | MAE | median | abs p75 | max abs |
|---|---:|---:|---:|---:|---:|---:|
| 当前 actual global PR | 151 | `-12.02` | `17.02` | `-10.0` | `24.25` | `86.0` |
| raw limb median baseline | 151 | `-22.34` | `26.82` | `-18.0` | `32.5` | `131.0` |
| naive consensus median | 151 | `-19.15` | `21.37` | `-18.0` | `27.0` | `123.0` |

结论：

- 直接 raw limb median 明显更差。
- 无脑使用 consensus median 也更差。
- 当前保守 gating 是必要的。

### 3.2 明确使用 PR consensus 的记录

当前能明确看出 global PR 采用 `pr_consensus_ms` 的记录有 12 条：

| record | raw limb median | actual PR | GT PR | raw diff | new diff |
|---:|---:|---:|---:|---:|---:|
| 17 | 115.5 | 199.0 | 207.0 | -91.5 | -8.0 |
| 23 | 106.5 | 145.0 | 210.0 | -103.5 | -65.0 |
| 39 | 136.0 | 201.0 | 224.0 | -88.0 | -23.0 |
| 40 | 129.0 | 164.0 | 191.5 | -62.5 | -27.5 |
| 49 | 98.5 | 136.0 | 174.0 | -75.5 | -38.0 |
| 50 | 92.0 | 122.0 | 139.0 | -47.0 | -17.0 |
| 61 | 82.0 | 146.0 | 186.0 | -104.0 | -40.0 |
| 73 | 80.5 | 126.0 | 170.0 | -89.5 | -44.0 |
| 75 | 76.0 | 136.0 | 156.0 | -80.0 | -20.0 |
| 102 | 98.0 | 162.0 | 201.0 | -103.0 | -39.0 |
| 120 | 159.0 | 186.0 | 220.0 | -61.0 | -34.0 |

另有 record 52 采用 consensus，但 GT PR 为空，无法评价。

这说明 PR consensus 对“raw PR 被短假 P 拉低”的病例很有效，但对 record 23/61/73/102/120 仍救得不够，说明 onset consensus 仍偏晚。

## 4. 当前 PR outlier

当前 PR 误差最大的记录：

| record | diff_pr | algorithm PR | GT PR | 特征 |
|---:|---:|---:|---:|---|
| 57 | +86.0 | 225.0 | 139.0 | raw limb PR 偏长；短 consensus=102 被保护机制拒绝 |
| 130 | -84.5 | 122.5 | 207.0 | raw 和 consensus 都锁在靠近 QRS 的短 PR cluster |
| 182 | +68.12 | 202.12 | 134.0 | raw/consensus 都短，但后续 PR core rescue 过度拉长 |
| 23 | -65.0 | 145.0 | 210.0 | consensus 已救，但仍偏短 |
| 64 | -48.0 | 127.0 | 175.0 | consensus=136，仍偏短，未触发足够修正 |
| 22 | -46.5 | 125.0 | 171.5 | onset spread 大，raw limb median 偏短 |
| 180 | -46.0 | 124.0 | 170.0 | raw 和 consensus 都偏短 |
| 73 | -44.0 | 126.0 | 170.0 | consensus 已救，但仍偏短 |
| 139 | +44.0 | 206.0 | 162.0 | raw limb PR 偏长，consensus=124 又偏短 |
| 27 | -41.5 | 122.0 | 163.5 | consensus=142 但未接管 |

## 5. 失败模式分类

### 5.1 长 PR 没救够

代表记录：23、39、40、49、61、73、75、102、120、130。

表现：

- raw limb median 明显短于 GT。
- `pr_consensus_ms` 能把 PR 往长方向拉，但不少记录仍比 GT 短 30-60ms。
- `p_onset_consensus_spread_ms` 往往很大，说明多 lead onset 候选分裂严重。

根因判断：

当前 P onset consensus 仍然是比较粗的全体 onset 聚合，缺少真正的 onset cluster 选择。它能避免最晚的假 P，但还不能在存在多个 P onset cluster 时稳定选择“更早且真实”的长 PR cluster。

下一步应做：

- 给 P onset 加 cluster-level consensus，而不是只记录全体 onset spread。
- 对每个 onset cluster 记录 support、lead territory、cluster center、与 representative prior 的距离。
- 对长 PR 场景允许选择更早 cluster，但必须有多 lead 和代表波形支持。

### 5.2 短 PR 假 P 仍主导

代表记录：130、64、22、180、27、9、29。

表现：

- `pr_consensus_ms` 也偏短，说明 fused P peak/onset 已经锁到靠近 QRS 的假 deflection。
- P offset consensus 可能看起来很稳定，但这是建立在错误 P 波上的稳定。

根因判断：

这是“错误共识化”：多个 lead 都在靠近 QRS 的小波动处找到候选，时间一致，看起来像共识，但它不是实际 P onset。

下一步应做：

- P peak fusion 不应只输出一个 peak index，需要输出 candidate clusters。
- 当短 PR cluster 和长 PR cluster 竞争时，不能只按幅度/时间一致性选。
- 应加入 PR segment、P duration、lead territory、representative template prior 共同打分。

### 5.3 PR 被救得过长

代表记录：57、182、139。

record 57：

- raw limb median = 225ms，GT = 139ms。
- `pr_consensus_ms = 102ms`，被当前 gating 拒绝，这是正确保护。
- 但 raw limb PR 仍过长，说明需要识别 false early P / T-tail contamination，不能只防短 consensus。

record 182：

- raw limb median 约 113.5ms。
- consensus median 约 109ms。
- 最终 algorithm PR = 202.12ms。
- 这提示后续 `_physiologic_pr_core_from_beats(...)` 或 API 层 PR rescue 把 PR 过度拉长。

record 139：

- raw limb median 约 206ms，GT = 162ms。
- consensus median 约 124ms。
- raw 太长、consensus 太短，真实值在中间。

下一步应做：

- PR core rescue 不能只因为 current PR <120 或 >240 就覆盖，需要检查 raw lead median、consensus median、P onset cluster 是否支持。
- 对正向 outlier 增加保护：如果 raw PR 很长但 consensus/offset/P duration 提示另一簇，不能直接信 raw limb median。
- 需要“中间 cluster”策略，而不是 raw 与 consensus 二选一。

### 5.4 P offset consensus 已有覆盖，但仍需谨慎

当前覆盖：

- beat-level rows 总数：25860。
- `p_onset_consensus_index` 覆盖：23940 rows，约 `92.58%`。
- `p_offset_consensus_index` 覆盖：15492 rows，约 `59.91%`。
- representative-level 有 P offset consensus 的记录：183/200。

P duration consensus：

- median：`80 ms`
- p75：`90 ms`
- p90：`106 ms`
- max：`152 ms`

P offset support：

- median support：7 leads
- p90 support：11 leads
- offset spread median：23ms
- offset spread p90：40ms
- max spread：68ms

说明：

- P offset consensus 覆盖面不错，且多数支持 lead 数足够。
- 但 offset spread p90 已到 40ms，说明困难病例仍然分散。
- record 57 的 P duration consensus 为 152ms，偏长；record 24/171 有偏短 P duration。不能马上把它无条件用于 P morphology/PTF。

下一步应做：

- 给 P offset consensus 增加 confidence 等级。
- `p_dur_consensus_ms <60` 或 `>140` 时标记 low confidence。
- morphology/PTF 使用 consensus boundary 前，要检查 P duration、PR segment、lead territory 支持。

### 5.5 P axis 仍是明显问题

当前 P axis 统计：

| 指标 | 数值 |
|---|---:|
| n | 172 |
| bias | `-10.12°` |
| MAE | `24.25°` |
| median diff | `-1.97°` |
| abs p90 | `58.10°` |
| max abs | `228.83°` |
| abs > 30° | 36 |
| abs > 60° | 17 |
| abs > 90° | 14 |

最大 outlier：

- record 92：diff `-228.8°`
- record 106：diff `-193.0°`
- record 10：diff `-181.2°`
- record 133：diff `+160.6°`
- record 31：diff `-144.7°`
- record 17：diff `-142.7°`
- record 127：diff `+140.4°`

根因判断：

P axis 依赖 P signed area / P amplitude / lead polarity。当前第一阶段主要改善 PR onset/offset，不会自动修复 P axis。P axis outlier 往往和低幅 P、错误 P polarity、错误 P area 积分窗口、lead 支持不足有关。

下一步应做：

- P axis 不应只用 raw P area；应优先使用 high-confidence P boundary 或 representative/template P area。
- 对 limb lead P polarity 做 cross-lead consistency 检查。
- 对明显异常轴，例如接近 180° 或翻转，检查 lead reversal、P wave polarity cluster、低幅 P 支持度。

## 6. 高 P missing 记录

P missing rate 高的记录：

| record | P missing rate | algorithm P missing | total | PR |
|---:|---:|---:|---:|---|
| 104 | 95.5% | 126 | 132 | None |
| 45 | 86.7% | 104 | 120 | None |
| 111 | 84.3% | 91 | 108 | None |
| 90 | 84.2% | 101 | 120 | None |
| 34 | 80.8% | 97 | 120 | None |
| 74 | 80.0% | 48 | 60 | None |
| 44 | 65.2% | 86 | 132 | None |
| 99 | 63.5% | 99 | 156 | None |
| 51 | 56.7% | 68 | 120 | None |
| 95 | 53.3% | 64 | 120 | None |

这类记录不应优先靠 onset/offset tuning 解决，而应先分类：

- 是否真实无 P 或 AF/flutter-like。
- 是否 P 波太低幅导致被 `low_p_support` 抑制。
- 是否 pacing/unknown context 把 P/PR 抑制掉。
- 是否 representative group 选择不适合 P 波测量。

## 7. 建议下一轮修改优先级

### P0：给 P onset 做真正 cluster consensus

当前已有 P offset cluster，但 P onset 仍然缺少 selected cluster provenance。下一步应新增：

- `p_onset_cluster_center_index`
- `p_onset_cluster_support`
- `p_onset_cluster_spread_ms`
- `p_onset_cluster_leads`
- `p_onset_cluster_rank`
- `p_onset_consensus_reason`

目标是区分：

- 靠近 QRS 的短 PR cluster。
- 更早的真实 P onset cluster。
- T-tail / baseline drift cluster。

### P1：限制 PR core rescue 过度覆盖

record 182 显示后续 PR rescue 可能把本来短的 raw/consensus PR 拉到 202ms。建议：

- 如果 raw limb median 和 `pr_consensus_ms` 都在 100-130ms，PR core rescue 不应直接覆盖到 >180ms，除非有明确多 lead 长 PR onset cluster。
- PR core rescue 应记录 provenance，例如 `pr_source=physiologic_pr_core_rescue`。
- 对 rescue 前后差异 >50ms 的记录输出 debug 字段。

### P2：对正向 PR outlier 做 false early-P/long-PR 识别

record 57/139 说明 raw limb PR 可被早期假 P 拉长。建议：

- 若 raw PR 很长但 consensus P duration/PR segment 支持较短 cluster，先降权 raw long PR。
- 不要直接采用短 consensus，而是用 cluster scoring 选中间可信 cluster。

### P3：P offset consensus 加 confidence，不直接进 morphology

建议新增：

- `p_offset_consensus_confidence`
- `p_duration_consensus_reliable`
- `p_boundary_consensus_reject_reason`

规则：

- support < 4：低置信度。
- offset spread > 35-40ms：低置信度。
- P duration <60ms 或 >140ms：低置信度，除非有特殊形态支持。

### P4：P axis 单独修

P axis 与 PR 不同，应单独一轮：

- 使用 representative/template P signed area。
- 加 limb-lead P polarity consistency。
- 对低幅 lead 降权。
- 对疑似 lead reversal 的 P axis 输出保守值或 unavailable。

## 8. 推荐回归记录

下一轮建议固定这些记录做小批次：

- PR consensus 改善：17、39、40、75、120。
- 长 PR 仍不足：23、61、73、102、130。
- 正向 outlier：57、139、182。
- P offset 边界：5、24、57、171。
- P axis outlier：10、17、31、92、106、127、133。
- 高 P missing：34、45、74、90、104、111。

推荐命令：

```bash
.venv_report_regen_20260625/bin/python compare_annotations.py \
  --batch-reports \
  --records 5 17 23 39 40 57 61 73 75 102 120 130 139 171 182 \
  --out-dir tmp_p_wave_next_regression \
  --no-visuals
```

## 9. 小结

这次重跑证明第一阶段方向是对的：P boundary consensus 和 PR consensus gating 能救一批明显的 P onset 偏晚/短 PR outlier，record 39 是最典型例子。

但当前 P 波模块还没有真正完成“每个 lead 每个 beat -> 代表波形 -> 多 lead cluster consensus”的闭环。下一步最值得做的是 **P onset cluster consensus** 和 **PR rescue provenance/gating**，而不是继续调单个阈值。P offset consensus 已经有基础，但要加 confidence 后再进入 morphology/PTF/P axis 等更敏感特征。
