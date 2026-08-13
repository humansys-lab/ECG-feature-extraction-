<!-- i18n-nav -->
[中文](p_wave_onset_offset_consensus_analysis.md) | [English](p_wave_onset_offset_consensus_analysis.en.md) | [日本語](p_wave_onset_offset_consensus_analysis.ja.md)
<!-- /i18n-nav -->

# P 波 onset/offset 检测问题与共识增强分析

本文聚焦当前 P 波相关检测，尤其是 `P onset` 和 `P offset` 偏差较大的原因，以及如何用“每个 lead 每个 beat -> 代表波形 -> 多 lead 共识”的方式增强稳定性。

## 1. 结论概览

当前算法已经有一部分 P 波共识机制，但它主要用于 P 峰候选的锚定，以及派生一个 `pr_consensus_ms`。真正的 P 波边界，尤其是 `P offset`，仍然主要由每个 lead 的局部切线/阈值逻辑独立决定。

### 1.1 已实现的第一阶段修改

本轮已完成一版保守实现：

1. 在 beat-level `LeadBeatFeatures` 中新增 P 边界共识旁路字段，包括：
   - `p_onset_consensus_index`
   - `p_offset_consensus_index`
   - `p_dur_consensus_ms`
   - `pr_segment_consensus_ms`
   - `p_onset_consensus_support`
   - `p_offset_consensus_support`
   - `p_onset_consensus_spread_ms`
   - `p_offset_consensus_spread_ms`
   - `p_boundary_consensus_source`

2. `_apply_multilead_consensus(...)` 现在会对同一 beat 的可靠 P offset 做 cluster 共识：
   - 至少需要 3 个可靠 lead 支持同一 offset cluster。
   - 不覆盖原始 `bf.p.offset`。
   - 只把 consensus offset、P duration、PR segment 作为旁路字段记录下来。

3. `build_representative_lead_features(...)` 会把这些 P boundary consensus 字段聚合到 `representative_leads.*.params`。

4. `compute_global_features(...)` 的全局 PR 增加保守 consensus gating：
   - raw limb-lead PR 必须明显分裂。
   - supported `pr_consensus_ms` 必须至少有 3 个 P onset consensus 支持。
   - consensus PR 必须在 `120-300 ms` 范围内。
   - consensus PR 必须比 raw limb median 至少长 25ms，主要用于修正 P onset 偏晚导致的 PR 偏短。
   - 短 consensus 不会接管全局 PR，避免 record 57 这类 102ms consensus 被误用。

LUDB spot check：

- record 39：global PR 从 `136 ms` 调整为 `201 ms`，相对 GT `224 ms`，误差从 `-88 ms` 改善到 `-23 ms`。
- record 57：`pr_consensus_ms=102 ms` 未接管，全局 PR 仍保持 raw 路径，避免短 consensus 误用。
- record 130：`pr_consensus_ms=124 ms` 与 raw median 差异不足，未接管。
- record 5：生成了 P offset consensus / P duration consensus 旁路字段，但全局 PR 保持原 raw 路径。

### 1.2 已实现的第二阶段修改：P onset cluster consensus

后续又完成了 P onset 的 cluster-level 共识增强，核心目标是解决“raw per-lead PR 被靠近 QRS 的假 P 拉短”的问题，同时避免把短 PR 假共识无条件用于全局 PR。

实现要点：

1. `_apply_multilead_consensus(...)` 不再只用所有 P onset 的简单 P10/median；会先在同一 beat 的可靠 lead 中按 onset 时间聚类。
2. P onset cluster 需要满足通用约束：
   - 至少 3 个可靠 lead 支持。
   - cluster spread 不超过约 `45 ms`。
   - cluster PR 位于 `120-260 ms` 的生理范围。
   - 记录 limb/precordial 支持数和参与 lead 列表。
3. 若存在合格的生理性 cluster，则 `pr_consensus_ms` 使用该 cluster center；否则回退原来的 percentile 路径。
4. `build_representative_lead_features(...)` 会把 `p_onset_cluster_*` 元数据聚合到代表 lead。
5. `compute_global_features(...)` 的 PR consensus gating 新增一条较窄的 rescue 路径：只有在 raw PR 分布存在明显短 PR outlier，并且 P onset cluster 有足够 support、较小 spread、足够 limb lead 支持时，才允许把全局 PR 从 raw median 上调到 cluster consensus。

LUDB outlier spot check：

- record 2：global PR 从 `170.5 ms` 调整为 `178.0 ms`，相对 GT `194.5 ms`，误差从 `-24.0 ms` 改善到 `-16.5 ms`。
- record 64：global PR 从 `127.0 ms` 调整为 `143.0 ms`，相对 GT `175.0 ms`，误差从 `-48.0 ms` 改善到 `-32.0 ms`。
- record 139：global PR 从 `206.0 ms` 调整为 `199.0 ms`，相对 GT `162.0 ms`，误差从 `+44.0 ms` 改善到 `+37.0 ms`。

### 1.3 已实现的第三阶段修改：PR core rescue 按 beat 投票

进一步分析发现，`_physiologic_pr_core_from_beats(...)` 曾经把所有 lead-beat PR 候选直接放在一起取 upper percentile。这样会让某一个 beat 上多个 lead 的同一类假长 PR 被重复计票，从而把全局 PR 过度拉长。

当前已改为：

1. 仍然先按 PR 生理范围、P confidence 和 `p_unreliable` flag 过滤候选。
2. 先按 `beat_id` 分组，每个 beat 内取一个 PR 中位数。
3. 再对 beat-level PR core 做原来的 median / upper-percentile 聚合。

这保持了“短 PR 假候选占多数时用 upper core rescue”的能力，但避免单个异常 beat 在多个 lead 上重复投票。

LUDB outlier spot check：

- record 182：global PR 从 `202.12 ms` 调整为 `136.58 ms`，相对 GT `134.0 ms`，误差从 `+68.12 ms` 改善到 `+2.58 ms`。
- record 139：global PR 从第二阶段的 `199.0 ms` 进一步调整为 `173.68 ms`，相对 GT `162.0 ms`，误差从 `+37.0 ms` 改善到 `+11.68 ms`。
- 在 12 条 PR outlier 小批量记录上，PR MAE 从第一阶段结果约 `50.47 ms`、第二阶段约 `47.93 ms`，进一步降到约 `40.40 ms`。

以上修改均不使用 record ID、LUDB 标注、文件名、诊断标签或 patient metadata 做条件判断；record 只作为回归验证样例。

### 1.4 已实现的第四阶段修改：强共识回写原始 P WaveBounds

参考项目根目录 `P wave` 技术文档后，进一步明确：目标不能只是让最终 report 的 PR 更接近参考值，而是要让原始 P 波边界检测本身更可靠。文档强调 median/representative P detection 应输出真实的 P onset/offset，report 只应消费这些测量。

因此当前 `_apply_multilead_consensus(...)` 已从“只写 consensus 旁路字段”扩展为“在强共识时保守修正 raw `WaveBounds`”：

1. 对明显离群的 raw `p.onset`：
   - 必须存在合格的 physiologic P onset cluster。
   - raw onset 与 cluster center 差异至少约 `20 ms`。
   - 修正后必须满足 `p_onset < p_peak < p_offset`。
   - P duration、onset-to-peak、PR segment 均需在生理范围内。
   - 修正后同步更新 raw `pr_ms` 和 `p_dur_ms`，并加 `p_onset_corrected_by_consensus` flag。

2. 对明显离群的 raw `p.offset`：
   - 必须存在支持度足够、spread 合理的 P offset cluster。
   - raw offset 与 consensus offset 差异至少约 `20 ms`。
   - 修正后必须满足 `p_onset < p_peak < p_offset`。
   - P duration 和 PR segment 必须合理。
   - 修正后同步更新 raw `p_dur_ms`，并加 `p_offset_corrected_by_consensus` flag。

这一步的重点是提高原始 beat/lead 级边界，而不是只修全局 PR。LUDB spot check：

- record 2：aVL、V1 等部分 lead 的 raw PR 被拉向多导联共识；仍有 aVR/V3/V5/V6 这类 peak 本身疑似选错的短 PR 假阳性没有硬修。
- record 5：12 个 lead-beat row 的 raw P offset 被 consensus 修正，减少 offset 分散。
- 12 条 PR outlier 小批量上，PR MAE 与第三阶段基本持平略好：约 `40.40 ms -> 40.07 ms`。

这也暴露出下一层问题：如果某个 lead 的 P peak 本身选到了靠近 QRS 的假 deflection，仅移动 onset/offset 不够安全。下一步应做的是 P peak/candidate 级重选或抑制，最好结合 `P wave` 文档里的 raw rhythm residual、I/II + V1-V6 lead selection、baseline-activity threshold、template similarity 和 PR/PP consistency。

### 1.5 已实现的第五阶段修改：短 PR 假 P candidate 抑制

第四阶段只在“同一个 P peak 仍然可信”的前提下平移 onset/offset。对于某些 lead，P peak 本身已经选到了 QRS 前的局部假波，此时硬把 onset/offset 移到全局共识位置会产生不合理几何。因此当前新增了保守的 candidate suppression：

1. 必须存在强 physiologic P onset cluster。
2. 当前 lead 的 raw PR 明显过短，例如小于约 `100 ms`。
3. cluster PR 比 raw PR 至少长约 `50 ms`。
4. 当前 lead 的 P peak 距离 cluster onset 过远，超过单个 P 波 onset-to-peak 可接受范围。
5. 满足上述条件时，清空该 lead/beat 的 raw P `WaveBounds`、`pr_ms`、P amplitude/area、P morphology 派生字段，并加：
   - `p_candidate_suppressed_by_consensus`
   - `p_unreliable`

这一步的目标不是“编造一个更好看的 P 波”，而是在 peak 本身不可信时停止把错误 P candidate 传播到 representative、P axis、P morphology 和 report。

LUDB spot check：

- record 2：aVR/V3/V5/V6 等靠近 QRS 的短 PR 假候选明显减少；全局 PR 从第四阶段的 `175.5 ms` 调整为 `181.0 ms`，相对 GT `196.5 ms`，误差从 `-21.0 ms` 改善到 `-15.5 ms`。per-lead PR 中，aVR 从极短 `94 ms` 提升到 `164 ms`，V6 变为 N/A，避免继续报告假 P。
- record 22：global PR 从 `125.0 ms` 调整为 `148.0 ms`，相对 GT `170.0 ms`，误差从 `-45.0 ms` 改善到 `-22.0 ms`。
- 12 条 PR outlier 小批量上，PR MAE 从第四阶段约 `40.07 ms` 进一步降到约 `37.69 ms`。

这仍然是 candidate 抑制，不是完整的 candidate 重选。下一步更大的实现应基于 `P wave` 文档：在 raw rhythm residual 或代表波形候选集中重新选择 P peak，并使用 template similarity、PR/PP consistency、dual-lead support 和 baseline activity threshold 评分。

### 1.6 已实现的第六阶段修改：P peak 上游融合锚点重选

第五阶段可以阻止明显错误的短 PR 假 P candidate 继续传播，但它本质上仍是“抑制”。为了让原始波形检测本身更准，当前又把改进前移到每个 lead/beat 的 P peak 选择阶段。

实现要点：

1. 在 `delineate_beats(...)` 的每 lead P 搜索中，`_fused_p_peak` 不再只用于缩小搜索窗口或在幅度完全相等时打破平局。
2. 新增 `_select_p_peak_with_fused_anchor(...)`：
   - fused P peak 必须位于当前 lead 的 P 搜索窗内。
   - fused P peak 幅度必须达到该 lead 的 P 最小幅度阈值。
   - 当前 abs-mode peak 若明显靠近 QRS，而 fused peak 到 QRS 的距离处于合理 P peak 范围，则允许回到 fused peak。
   - 如果两者幅度接近，且 fused peak 更符合时序，也允许回到 fused peak。
3. 该逻辑同时用于初始 P peak 搜索，以及 P peak 被判为超出生理窗后的 constrained re-search。
4. 条件只依赖局部波形幅度、QRS 参考点、采样率和已有 multi-lead fused anchor，不使用任何 record ID 或 LUDB 特异规则。

这一步和第五阶段的区别是：第五阶段在 peak 已经不可信时清空 P；第六阶段在边界测量之前尝试选回更可信的 P peak。因此它能减少后续 `p_unreliable` 和 P 缺失，而不是简单把错误 candidate 隐藏掉。

LUDB outlier spot check：

- record 2：global PR 从第五阶段的 `181.0 ms` 调整为 `183.0 ms`，相对 GT `196.5 ms`，误差从 `-15.5 ms` 改善到 `-13.5 ms`；P 缺失从 `37` 降到 `34`。
- record 22：global PR 从 `148.0 ms` 调整为 `154.0 ms`，相对 GT `170.0 ms`，误差从 `-22.0 ms` 改善到 `-16.0 ms`；P 缺失从 `29` 降到 `28`。
- record 23：global PR 不变，但 P 缺失从 `12` 降到 `11`。
- record 182：global PR 保持 `136.58 ms`，相对 GT `134.0 ms` 仍很接近；P 缺失从 `3` 降到 `2`。
- 12 条 PR outlier 小批量上，PR MAE 从第五阶段约 `37.69 ms` 小幅降到约 `37.07 ms`。

这仍不是完整的 raw rhythm residual/template similarity 重检测，但已经把“共识”从最终 PR report 前移到原始 P peak 选择，对原始 `WaveBounds` 的质量更直接。

### 1.7 已实现的第七阶段修改：P peak cluster 支持度融合

第六阶段解决的是“单个 lead 在搜索窗内选错 peak”问题，但 `_fused_p_peak` 本身仍可能被少数高幅导联，或 QRS onset 附近的共同小波动拉偏。当前进一步增强了 beat-level P peak pre-pass 融合。

实现要点：

1. `_fuse_peak_anchor(...)` 新增可选的 per-lead normalization：
   - 每个 lead 的候选先在本 lead 内归一化，避免某个高幅 lead 用绝对幅度压过多导联低幅 P 共识。
   - cluster 评分使用 unique lead support；同一 lead 的多个候选不会重复投票。
2. P peak pre-pass 调用启用该模式；T peak 融合仍保持原幅度/面积策略，避免扰动 T 波检测。
3. P peak cluster 评分加入 P-to-QRS 生理约束：
   - qrs reference 改为优先使用代表 prior 的 QRS onset。
   - 没有 QRS onset prior 时，用 `R - 20 ms` 作为近似 QRS onset，而不是直接用 R peak。
   - 这样 QRS onset 附近的短间期假 cluster 不会因为距离 R peak 还有几十毫秒而被误判为合理 P peak。
4. 对短 QRS-adjacent cluster 加强惩罚；如果没有更合理替代，它仍可被选中，但只要存在时序合理的 P cluster，就优先选择后者。
5. 全局 PR consensus 增加 raw-support guard：
   - 长 PR cluster 要接管全局 PR，必须有至少 2 个 raw PR 值靠近该 cluster，避免单个/少数异常 lead 把正常 PR 拉成长 PR。
   - 当 raw median 被长 PR outliers 拉高，而强 onset cluster 有足够 raw 支持时，允许 cluster 向下救回。
   - raw representative PR 很稀疏时，优先使用强 onset cluster，而不是让单导联 raw PR 决定全局 PR。

这一步比第六阶段更上游：不只是每个 lead 在已给定 fused anchor 后纠偏，而是在 fused anchor 生成阶段就利用“每 lead 一票 + 多 lead 支持 + QRS onset 时序”防止 anchor 本身被带偏。

LUDB full-batch 对比：

- `ludb_full_compare/summary.csv` 参考：PR MAE 约 `17.02 ms`，PR bias 约 `-12.02 ms`，`|PR diff| > 20 ms` 为 `47/151`，`|PR diff| > 40 ms` 为 `13/151`。
- 当前 `tmp_p_anchor_final_full/summary.csv`：PR MAE 约 `9.78 ms`，PR bias 约 `-6.02 ms`，`|PR diff| > 20 ms` 为 `18/152`，`|PR diff| > 40 ms` 为 `0/152`。
- beat/lead 级 P 任一边界缺失从 `6049/25860` 降到 `3005/25860`。
- `p_unreliable` 从 `3104` 降到 `2669`。

典型改善：

- record 130：global PR 从 `122.5 ms` 调整到 `216.0 ms`，相对 GT `207.0 ms`，从 `-84.5 ms` 改善为 `+9.0 ms`。
- record 57：global PR 从 `225.0 ms` 调整到 `124.0 ms`，相对 GT `139.0 ms`，从 `+86.0 ms` 改善为 `-15.0 ms`。
- record 23：global PR 从 `145.0 ms` 调整到 `197.0 ms`，相对 GT `210.0 ms`，从 `-65.0 ms` 改善为 `-13.0 ms`。
- record 17/39/102 曾在中间版本因错误参考 R peak 出现回退；改为 QRS onset reference 后分别恢复到接近 GT。
- record 11/56/133 曾在中间版本因长 PR cluster 过接管回退；增加 raw-support guard 后，record 11 从 `219 ms` 回到 `149.5 ms`，record 56 从 `207 ms` 回到 `130 ms`，record 133 从 `221 ms` 回到 `182 ms`。

### 1.8 已实现的第八阶段修改：PR core 双峰保护

进一步分析剩余 outlier 时发现，某些记录的代表 lead raw PR 已经明显异常，`api.py` 会启用 beat-level `_physiologic_pr_core_from_beats(...)` 作为保护。但旧逻辑在 beat-level PR core 数量不少时固定取 `78th percentile`，这是为了避免短 PR 假候选占多数时把 PR 拉短；当 beat-level PR 本身出现两个相隔明显的 cluster 时，这个上分位策略会误把上方 cluster 当作真实 PR。

当前新增了一个保守的双峰保护：

1. 仍然先按 beat 聚合，每个 beat 只贡献一个 PR 中位数，避免同一 beat 的多个 lead 重复投票。
2. 当 beat-level PR core 至少 5 个，并且相邻排序值中存在显著大 gap 时，先判断是否形成下方 cluster 和上方 cluster。
3. 只有在下方 cluster 至少 3 个 beat、上方 cluster 至少 2 个 beat，并且最大 gap 同时满足绝对阈值和相对常规 gap 的倍数阈值时，才认为这是双峰分裂。
4. 对这种双峰分布，返回下方 cluster 的上四分位，而不是全体 beat core 的 78 分位。
5. 如果只是单个长 PR outlier，或 PR 分布连续渐变，则保持原来的 78 分位策略。

这一步不是 record 特异性规则，也不改动原始 P `WaveBounds`。它的作用是避免 report 层在 raw P onset 已经分裂时再次把错误的长 PR cluster 放大。

LUDB full-batch 对比第七阶段：

- `tmp_p_anchor_final_full/summary.csv`：PR MAE 约 `9.78 ms`，PR bias 约 `-6.02 ms`，`|PR diff| > 20 ms` 为 `18/152`，`|PR diff| > 40 ms` 为 `0/152`。
- 当前 `tmp_pr_core_split_only_full/summary.csv`：PR MAE 约 `9.54 ms`，PR bias 约 `-6.30 ms`，`|PR diff| > 20 ms` 为 `17/152`，`|PR diff| > 40 ms` 为 `0/152`。
- beat/lead 级 P 任一边界缺失保持 `3005/25860` 不变。
- `p_unreliable` 保持 `2669` 不变。

典型变化：

- record 139：global PR 从 `201.8 ms` 调整到 `159.5 ms`，相对 GT `162.0 ms`，误差从 `+39.8 ms` 改善为 `-2.5 ms`。
- 其余 full-batch PR 结果保持不变，说明该保护只在真正满足双峰条件时触发。

### 1.9 已实现的第九阶段修改：P candidate template/PP context guard

当前新增了原始 beat/lead 级 P candidate context scoring。它直接接在每个 lead/beat 的 P 检测之后、multi-lead consensus 之前，因此能让后续 P onset/offset 共识看到候选本身的上下文质量，而不是只在最终 global PR report 上修补。

新增字段包括：

1. `p_template_corr`：同 group、同 lead 的 P 片段与 median template 的归一化相关，取正向/反向相关的较大值，避免单纯因为 P 极性不同而误罚。
2. `p_multilead_support` 和 `p_multilead_support_score`：同一 beat 相近 P peak 是否被多个 reliable P lead 支持。
3. `p_pr_consistency_score`：只在 PR 分布稳定且证据足够时启用；PR 不稳定时保持 neutral，避免 AV block/AV dissociation 场景被误罚。
4. `p_pp_consistency_score`：只在 PP 节律稳定且证据足够时启用；证据不足或 PP 不稳定时保持 neutral。
5. `p_candidate_context_score` 和 `p_context_reason`：汇总上述证据，便于在 `*_features.json` 中解释每个 P candidate 的质量。

当前 suppression gate 非常保守：只有 template 弱、multi-lead support 弱、timing 弱，并且总 context score 也低时才会清空候选。实现时特意避免在循环中即时 suppress，以免前一个 lead 被清空后影响后续 lead 的 support 计数。

LUDB full-batch 对比第八阶段：

- `tmp_pr_core_split_only_full/summary.csv`：PR MAE 约 `9.54 ms`，PR bias 约 `-6.30 ms`，`|PR diff| > 20 ms` 为 `17/152`，`|PR diff| > 40 ms` 为 `0/152`。
- 当前 `tmp_p_context_guard_full/summary.csv`：PR MAE 约 `9.54 ms`，PR bias 约 `-6.30 ms`，`|PR diff| > 20 ms` 为 `17/152`，`|PR diff| > 40 ms` 为 `0/152`。
- beat/lead 级 P 任一边界缺失保持 `3005/25860` 不变。
- `p_unreliable` 保持 `2669` 不变。
- `p_candidate_context_score` 覆盖 `22861/25860` 个 beat/lead row；最终 full batch 中 `p_candidate_suppressed_by_context` 为 `0`，说明当前阶段主要提供可观测评分，未牺牲 P 检测完整性。

这一步为后续 candidate re-selection 和 raw rhythm residual 二轮搜索打基础：现在可以先观察哪些候选低分，再决定是否在同一 P 搜索窗内换用更合理的 alternative peak。

### 1.10 已实现的第十阶段修改：P candidate context re-selection

当前在原始 P detection 后新增了同一 lead/beat P 搜索窗内的 alternative peak re-selection。该阶段不新增 missing P，也不做 QRST residual 搜索；它只在当前 P candidate 明显低分、alternative full geometry 合法且 context score 明显更高时替换原始 `WaveBounds`。

实现要点：

1. 在每个 lead/beat 的最终 P 搜索窗中保存 top-k turning-point alternatives。
2. alternative 先从 smoothed signal 的 turning point 回到 raw ECG 附近局部峰值，避免边界测量落在平滑后的肩部 sample。
3. 对 alternative 重新测量 P onset/peak/offset、PR、P duration、P area、P morphology 和 PTF-V1。
4. 用同一套 template similarity、multi-lead support、PR consistency、PP consistency 重新评分。
5. re-selection gate 只允许 `old_score < 0.50` 的明显弱候选进入替换；borderline plausible candidate 即使 alternative 更高也不替换，避免对 report-level PR 造成不必要扰动。

新增诊断字段包括 `p_candidate_alternative_count`、`p_reselected_from_peak_index`、`p_reselected_to_peak_index`、`p_reselected_old_context_score`、`p_reselected_new_context_score` 和 `p_reselection_reason`。替换成功时添加 `p_candidate_reselected_by_context` flag。

LUDB full-batch 对比第九阶段：

- `tmp_p_context_guard_full/summary.csv`：PR MAE `9.5372 ms`，PR bias `-6.3022 ms`，`|PR diff| > 20 ms` 为 `17/152`，`|PR diff| > 40 ms` 为 `0/152`。
- 当前 `tmp_p_reselection_full/summary.csv`：PR MAE `9.5372 ms`，PR bias `-6.3022 ms`，`|PR diff| > 20 ms` 为 `17/152`，`|PR diff| > 40 ms` 为 `0/152`。
- beat/lead 级 P 任一边界缺失保持 `3005/25860` 不变。
- `p_unreliable` 保持 `2669` 不变。
- `p_candidate_reselected_by_context` 在 full batch 中触发 `30` 个 beat/lead row，分布于 `24` 条记录；report-level PR 没有任何 record 发生变化，说明当前 gate 对最终报告是中性的，但已经开始改进底层 raw `WaveBounds`。

剩余问题：

- 少数记录仍会出现中等程度偏差，例如 record 133 的 PR 仍偏长。下一步不应继续只靠 global PR 后处理，而应在更高置信度前提下扩展 raw rhythm residual / second-pass P search，或把 P onset/P offset boundary confidence 独立建模后再进入 consensus。

因此原始 P 边界/PR 聚合问题可以概括为：

1. **P peak 已开始使用 fused anchor 重选、cluster 支持度融合和 context scoring，但尚未充分利用这些分数进行 alternative candidate 重选。**
2. **全局 PR 目前主要使用 raw per-lead PR 的 limb lead median，很多时候没有使用已经算出的 `pr_consensus_ms`。**
3. **P onset/offset 的边界置信度没有被单独建模；`p_confidence` 更多反映 P 波幅度和噪声，不足以说明边界是准确的。**
4. **代表波形目前更多作为 prior 使用，还没有充分把代表波形上的 P 边界反哺到每个 beat/lead 的最终 P onset/offset。**
5. **P offset 对 P duration、P area、P terminal force、notched/biphasic 等形态学特征影响很大，当前偏差会向后传播。**

可以、也应该用“共识”增强 P 波边界检测，但不能简单套用目前 `P10 onset` 这种规则。P 波幅度小、形态多变、容易被 T 波尾部/基线漂移/PR segment 低幅波动干扰，建议使用“候选聚类 + 代表波形 prior + 多 lead 支持度 + 生理约束”的分层共识。

## 2. 当前 P 波检测流程

### 2.1 代表波形 prior

代码中会先构建每个 group/lead 的代表 beat prior：

- `build_group_priors(...)` 会从 `representative_beats` 里提取每个 group、每个 lead 的代表波形。
- `_delineate_rep_prior(...)` 会在代表波形上寻找 P/QRS/T 的先验位置。
- P 波 prior 目前使用 `_find_peak(...)` 找 P peak，再用 `_find_wave_bounds(..., frac=0.08)` 估计 P onset/offset。

这个 prior 的价值是：它能给后续 beat-level 检测一个相对稳定的时间窗口，减少每个 beat 从头搜索的自由度。

但这里有两个限制：

1. 代表波形 prior 的 P onset/offset 仍是阈值边界，不是多 lead 共识边界。
2. 当前代表波形往往是 medoid/single representative beat，不一定是 median/average template；它对形态保真较好，但对低幅 P 波边界降噪不够。

### 2.2 每个 beat 的多 lead P peak 预融合

在每个 beat 的 P 搜索窗口里，算法会先从所有 lead 搜 P 候选：

- `_candidate_triplets(...)` 产生每个 lead 的 P peak 候选。
- `_fuse_peak_anchor(...)` 根据幅度、面积、prior proximity、可靠 lead 权重，把多 lead 候选融合成一个 `_fused_p_peak`。

这一步是当前 P 波检测里最明确的“共识”使用点。它的作用是给所有 lead 一个共同的 P peak 锚点，避免每个 lead 各自选到完全不同的局部波动。

不过它也有风险：

- 如果 fused peak 被某个靠近 QRS 的假 P、T 尾部残留或基线波动吸引，后续所有 lead 的 P 搜索窗口都会被拉偏。
- 当前融合主要返回一个 peak index，缺少足够的 provenance，例如支持 lead 数、cluster spread、候选 cluster 之间的竞争关系。
- P peak 共识不能直接保证 P onset/offset 正确。P peak 稳定时，边界仍可能因为低幅起止点和基线不稳而明显偏移。

### 2.3 每个 lead 的 P onset/offset

每个 lead 上，P 检测的主流程大致是：

1. 根据 `_fused_p_peak` 或代表波形 prior 缩小 P 搜索窗口。
2. 在窗口中用 `_find_peak(..., mode="abs")` 找 P peak。
3. 用 `_select_p_peak_with_fused_anchor(...)` 检查 abs-mode peak 是否被靠近 QRS 的假峰吸走；如果 fused P peak 幅度达标且时序更合理，则把 P peak 选回 fused anchor。
4. 检查 P peak 到 R/QRS 的距离是否在生理范围内；若需要 constrained re-search，也再次应用 fused anchor 选择。
5. 用切线法估计 P onset：
   - 当前调用的是 `_t_onset_tangent(...)`。
6. 用 `_p_offset_tangent(...)` 估计 P offset。
7. 必要时用 `_rescue_long_pr_p_onset(...)` 把长 PR 的 P onset 往前救回。
8. 如果几何关系异常，`_p_wave_qrs_geometry_mode(...)` 可能清掉 P 或清掉 P onset。

这说明现在每个 lead 的 P onset/offset 主要是局部几何结果。它们会受到以下因素影响：

- P 波幅度低，导数和切线点容易被噪声支配。
- baseline 估计偏一点，onset/offset 就会明显移动。
- P 波双相、切迹、低平时，单一切线点不稳定。
- offset 搜索终点可以到 `qrs_on - 1`，如果 PR segment 低幅波动或 QRS 起点不稳，offset 容易被拖到离 QRS 过近。
- fallback 的 `_find_wave_bounds(..., frac=0.08)` 对幅度和局部噪声敏感，不同 lead 会给出不同长度的 P duration。

### 2.4 当前 multi-lead consensus

`_apply_multilead_consensus(...)` 会对同一个 beat 的多 lead 结果做一些共识处理。

对 P 波相关字段，目前最关键的是：

- 收集可靠 lead 的 `bf.p.onset`。
- 用一个较早分位数计算 `global_p_onset`。
- 派生每个 lead 的 `pr_consensus_ms`。

但这里有几个重要缺口：

1. **没有计算 P offset consensus。**
2. **没有计算 P duration consensus。**
3. **没有把 consensus P onset/offset 回写成最终 `WaveBounds`。**
4. **`global_p_onset` 使用简单分位数，缺少候选聚类和 cluster support 判断。**
5. **后续全局 PR 主要没有使用 `pr_consensus_ms` 作为主路径。**

所以当前“共识”对 P 边界的加强是局部、间接的，而不是完整闭环。

## 3. LUDB 结果中看到的问题

### 3.1 总体 PR 表现

从 `ludb_full_compare/summary.csv` 看，PR 指标显示 P onset 相关误差并不小：

- PR 可比较记录数：151
- 平均偏差：约 `-11.77 ms`
- MAE：约 `16.85 ms`
- median diff：约 `-10 ms`
- 绝对误差 p75：约 `22.75 ms`
- 最大绝对误差：`88 ms`
- PR 误差超过 20ms 的记录：44/151

这说明当前 P onset 有系统性偏短倾向，也就是算法 P onset 往往被检测得偏晚，导致 PR 比标注短。

另外，全库里 `p_unreliable` 标记很多，说明 P/QRS 几何关系和低支持度场景并不少见。典型高比例记录包括：

- 104：`126/132` beats 被标记 P 不可靠。
- 45：`104/120` beats 被标记 P 不可靠。
- 111：`91/108` beats 被标记 P 不可靠。
- 90：`101/120` beats 被标记 P 不可靠。

这些记录提示：当前 per-lead P 检测在困难样本上经常进入“不可靠但仍产生部分数值”的状态。

### 3.2 记录 5：P peak 尚可，P offset 明显分散

记录 5 的全局 PR 差异不算大，算法约 `130 ms`，GT 约 `138 ms`。但看各 lead 的 P 边界，问题很明显：

- lead I：P onset 约在 R 前 `157 ms`，P offset 约在 R 前 `50 ms`。
- lead II：P onset 约在 R 前 `162 ms`，P offset 约在 R 前 `100 ms`。
- lead aVF：P onset 约在 R 前 `158 ms`，P offset 约在 R 前 `104 ms`。
- lead V1：P onset 约在 R 前 `146 ms`，P offset 约在 R 前 `40 ms`。
- lead V5：P onset 约在 R 前 `152 ms`，P offset 约在 R 前 `42 ms`。

这里 P onset 大致集中在 R 前 `150-160 ms`，说明 P peak/前缘锚点并非完全失控；但 P offset 从 R 前 `40 ms` 到 `104 ms` 都有，导致 P duration 和 PR segment 长度严重不一致。

这正是当前缺少 P offset consensus 的典型表现：每个 lead 自己决定 P offset，最终没有一个跨 lead 的生理合理边界来校正。

### 3.3 记录 39：已有 PR consensus 更接近 GT，但全局 PR 没用它

记录 39 是非常关键的例子：

- 算法全局 PR：约 `136 ms`
- GT PR：约 `224 ms`
- diff：约 `-88 ms`

但从 per-lead representative 结果看，部分 lead 已经有接近 GT 的长 PR：

- aVF：约 `210 ms`
- V3：约 `221 ms`
- V4：约 `222 ms`

而现有 `pr_consensus_ms` 在多个 lead 上约为 `201 ms`，明显比全局 PR `136 ms` 更接近 GT `224 ms`。

这说明：

1. 当前算法内部已经捕捉到了一部分“长 PR 共识”信息。
2. 但最终 `compute_global_features(...)` 的全局 PR 主要仍走 raw limb lead median 路径。
3. raw limb median 被 I/III/aVR 等短 PR 假阳性拉短，导致全局 PR 明显偏小。

这类样本最能说明：P onset consensus 是有价值的，但目前没有被充分用于最终特征。

### 3.4 记录 57/130：简单 consensus 也可能错

需要注意的是，不能把现有 `pr_consensus_ms` 直接无条件替换成全局 PR。

记录 57：

- 算法全局 PR：约 `225 ms`
- GT PR：约 `139 ms`
- 当前 `pr_consensus_ms` 在多个 lead 上约 `102 ms`

这里 raw PR 和 consensus PR 都可能偏离 GT，只是方向不同。说明当前简单 onset 分位数/聚合并不足够可靠。

记录 130：

- 算法全局 PR：约 `122.5 ms`
- GT PR：约 `207 ms`
- 多个 lead raw PR 和 `pr_consensus_ms` 都偏短。

这说明如果 fused P peak 或 onset 候选一开始就锁到了靠近 QRS 的假 deflection，后面的简单共识会把错误一起“共识化”。

所以 P 波共识增强必须做候选聚类、支持度评估和生理约束，不能只是把所有 onset 求一个分位数。

## 4. 为什么 P onset 和 P offset 容易错

### 4.1 P 波信号本身弱

P 波通常幅度小、斜率低，起止点不是清晰尖峰。QRS onset 通常有更明显的斜率变化，T offset 可以借助更长的单峰/双峰形态，而 P onset/offset 更容易被以下因素影响：

- baseline wander
- 肌电噪声
- P 波低平
- P 波双相或切迹
- PR segment 上的小波动
- QRS onset 检测的轻微偏差

因此单 lead 单 beat 的局部切线法天然不够稳定。

### 4.2 P onset 被 fused peak 窗口牵引

当前 P peak 预融合后，每个 lead 的 P 搜索窗口会围绕 `_fused_p_peak` 或代表 prior 缩窄。如果 fused peak 是正确的，这会显著提高稳定性；但如果 fused peak 被错误候选吸引，所有 lead 都会在错误时间附近找 P 波。

这会造成一种“错误同步”：

- 多个 lead 都能找到一个局部小波。
- 这些小波时间接近，看起来像有共识。
- 但它们其实是 QRS 前的低幅噪声、T/P 混淆或 PR segment 波动。

所以 P peak consensus 需要输出 cluster support 和竞争 cluster 信息，不能只输出一个 peak index。

### 4.3 P offset 没有跨 lead 校验

P offset 目前主要用 `_p_offset_tangent(...)`，搜索到 `qrs_on - 1`。在 PR segment 波动或 QRS onset 稍偏的情况下，offset 可能被拖到非常靠近 QRS。

一旦 offset 偏晚，会导致：

- P duration 偏长。
- PR segment 被压缩。
- P area 偏大。
- P terminal force 可能被错误扩大。
- P notch/biphasic 判断可能被 PR segment 或 QRS 前波动污染。

一旦 offset 偏早，又会导致：

- P duration 偏短。
- P terminal component 被截断。
- 双相 P 波后半段丢失。
- P area 和 PTF 被低估。

记录 5 里 offset 分散就是这种问题。

### 4.4 P confidence 不是边界 confidence

当前 `p_confidence` 主要与 P amplitude、noise floor、检测支持有关。一个 lead 可以有很高的 P amplitude 和 `p_confidence`，但 onset/offset 仍然错。

因此需要拆分：

- `p_presence_confidence`：是否存在 P 波。
- `p_peak_confidence`：P peak 是否可信。
- `p_onset_confidence`：P onset 是否可信。
- `p_offset_confidence`：P offset 是否可信。
- `p_boundary_consensus_support`：边界是否被多 lead/代表波形支持。

否则后续 consensus 会把“P 存在很可信”误当成“P 边界很可信”。

## 5. 是否可以用“共识”加强？

可以，而且建议把 P 波共识拆成 4 层。

### 5.1 第一层：P peak 候选聚类共识

当前 `_fuse_peak_anchor(...)` 已经开始按 per-lead normalization、unique lead support 和 P-to-QRS onset 时序增强 P peak 融合。但它仍只返回一个 selected peak，后续还建议继续扩展为候选 cluster 输出：

- 每个 cluster 的中心时间。
- 支持 lead 数。
- 支持 lead territory，例如 limb、inferior、precordial。
- cluster 内时间离散度。
- cluster 内幅度/面积加权分数。
- 是否接近代表波形 prior。
- 是否满足合理 PR 区间。

不要只返回一个 fused peak，而是返回：

- `selected_p_peak`
- `selected_cluster_support`
- `selected_cluster_spread_ms`
- `competing_clusters`
- `selection_reason`

这样后续 onset/offset 可以知道当前 peak 锚点是否稳。

### 5.2 第二层：P onset 边界共识

对每个 beat，在每个 reliable lead 得到 raw P onset 后，不应直接用 P10。建议：

1. 收集可靠 lead 的 raw P onset。
2. 排除明显不合理候选：
   - P onset 晚于 P peak。
   - PR 太短或太长，除非已有 paced/retrograde/AV block 等上下文。
   - P duration 超出宽松范围。
   - onset 与当前 selected P peak 不属于同一 P 波形。
3. 对 onset 时间做聚类。
4. 选择与 P peak cluster、代表波形 prior、lead 支持度共同一致的 onset cluster。
5. 在 cluster 内用 robust statistic 得到 consensus onset，例如 median 或轻微偏早 percentile。

这里的关键是：对于 P onset，偏早和偏晚的代价不一样。

- 如果要测 PR，onset 偏晚会系统性低估 PR。
- 但 onset 过早又会把 T 波尾部/噪声算进 P。

因此建议使用 cluster 内的 robust early boundary，而不是全体 onset 的 P10：

- cluster 支持强、spread 小：可用 cluster P25 或 median。
- cluster 支持一般：用 median，并保留较低 confidence。
- 存在长 PR cluster 与短 PR cluster 竞争：优先看代表波形 prior、lead territory 支持和 P peak 一致性，不要只按幅度最大选。

### 5.3 第三层：P offset 边界共识

这是当前最缺的一层。

建议对每个 beat 计算：

- `p_offset_consensus_idx`
- `p_duration_consensus_ms`
- `pr_segment_consensus_ms`
- `p_offset_support_leads`
- `p_offset_spread_ms`

候选规则：

1. 每个 reliable lead 给出 raw P offset。
2. 删除明显异常 offset：
   - offset 早于 peak。
   - offset 晚到离 QRS onset 太近，除非明确短 PR/pre-excitation。
   - P duration 超出宽松上限，例如超过 160-180ms 且没有强双峰/异常形态证据。
   - P duration 过短，例如小于 40-50ms。
3. 对 offset 候选做聚类。
4. 优先选择与 onset consensus 和 P peak consensus 共同形成合理 P duration 的 cluster。
5. 加入 PR segment guard：
   - 正常情况下 `p_offset` 应早于 `qrs_onset` 一段距离。
   - 如果 offset 贴近 QRS，必须有多 lead 强支持，否则视为 QRS 前噪声/PR segment 波动。

对记录 5 这种场景，offset consensus 可以避免某些 lead 的 offset 被拖到 R 前 40ms，也可以避免另一些 lead 过早截断。最终 P duration 应落在多 lead 共同支持的范围。

### 5.4 第四层：代表波形模板共识

建议保留当前 medoid representative beat，但额外构建 median/trimmed-mean template 专门用于低幅边界：

- medoid：保留真实形态，适合展示和局部形态测量。
- median template：降低随机噪声，适合 P onset/offset 边界。

流程可以是：

1. 对同 group 的 beats 按 R peak 对齐。
2. 对每个 lead 构建 median template。
3. 在 template 上检测 P onset/offset。
4. 将 template P bounds 作为 per-beat/per-lead 的 prior 或 soft constraint。
5. 如果 per-beat raw boundary 与 template boundary 差太大，但没有多 lead 支持，就降权或替换为 consensus。

这会特别有利于 P onset/offset，因为 P 波边界低幅且 beat-to-beat 噪声影响大。

## 6. 建议的改进设计

### 6.1 数据结构层面

建议在 beat-level 特征中新增或保留以下 provenance 字段：

- `p_onset_raw_ms`
- `p_offset_raw_ms`
- `p_onset_consensus_ms`
- `p_offset_consensus_ms`
- `p_duration_consensus_ms`
- `p_peak_consensus_support`
- `p_onset_consensus_support`
- `p_offset_consensus_support`
- `p_boundary_spread_ms`
- `p_boundary_confidence`
- `p_boundary_source`

`p_boundary_source` 可以取：

- `raw_local`
- `lead_consensus`
- `template_prior`
- `template_and_lead_consensus`
- `suppressed`

这样后续调试能知道每个 P 边界到底来自哪里。

### 6.2 算法层面

建议按下面顺序修改：

1. **增强 `_fuse_peak_anchor(...)`**
   - 从单一 peak 输出改为 cluster-level 输出。
   - 保留支持 lead、cluster spread、候选竞争信息。

2. **新增 `_cluster_p_boundaries(...)`**
   - 输入同一 beat 多 lead raw P onset/offset。
   - 输出 consensus onset、offset、duration、support、spread。

3. **扩展 `_apply_multilead_consensus(...)`**
   - 当前只派生 `pr_consensus_ms`。
   - 增加 P onset consensus、P offset consensus、P duration consensus。
   - 对不可靠 raw boundary 进行降权或标记。

4. **修改全局 PR 聚合**
   - 当前 `_global_pr_median(...)` 主要使用 raw limb PR。
   - 建议当 consensus 支持强、raw limb spread 大时，优先使用 supported consensus PR。
   - 但不能无条件使用 `pr_consensus_ms`，需要 cluster support 和生理约束。

5. **代表波形新增 median template**
   - medoid 继续保留。
   - median template 用于 P 边界 prior。

6. **P 形态学特征使用边界 confidence**
   - P area、P terminal force、notched/biphasic 应记录使用 raw boundary 还是 consensus boundary。
   - 如果 boundary confidence 低，形态学结论也应降级。

### 6.3 选择 consensus 的建议规则

可以先用一个保守版本：

1. 如果 reliable P leads 少于 3 个，不强行覆盖 raw boundary，只记录 consensus 低置信度。
2. 如果 onset cluster support >= 3 且 spread <= 20ms，则可生成 high-confidence onset consensus。
3. 如果 offset cluster support >= 3 且 spread <= 25ms，并且 P duration 在合理范围内，则生成 high-confidence offset consensus。
4. 如果 raw PR median 与 consensus PR 差异 > 40ms，检查：
   - 哪个有更多 lead 支持。
   - 哪个更接近 representative template prior。
   - 哪个产生更合理 P duration 和 PR segment。
5. 如果存在两个 P onset cluster，一个短 PR、一个长 PR：
   - 不按时间早晚直接选。
   - 看 P peak cluster 是否同源。
   - 看 limb/inferior lead 是否支持。
   - 看 template prior 是否支持。
   - 看该 cluster 的 P duration/offset 是否生理合理。

## 7. 优先修复点

### P0：不要只把 `pr_consensus_ms` 当旁路字段

记录 39 显示，当前 `pr_consensus_ms` 有时明显优于全局 PR，但全局 PR 没有使用它。建议先实现一个保守 gating：

- 当 raw limb PR spread 很大；
- 且 consensus PR support 足够；
- 且 consensus PR 与代表波形 prior 一致；
- 且 consensus PR 在合理生理范围内；

则全局 PR 使用 consensus PR 或 consensus/raw 混合估计。

这一步能先改善一批 P onset 偏晚导致 PR 偏短的病例。

### P1：增加 P offset consensus

P offset 当前是最大缺口。建议先不急着覆盖所有 lead 的 `WaveBounds`，可以先新增字段：

- `p_offset_consensus_ms`
- `p_duration_consensus_ms`
- `p_offset_consensus_support`

然后在 representative/global features 中优先用高支持度的 consensus P duration。

### P2：把 P boundary confidence 从 P presence confidence 中拆出来

如果不拆，后续会继续出现“P 存在很确定，但边界很错”的情况。建议新增 boundary-level confidence，并在 P area/PTF/notched/biphasic 中使用它。

### P3：代表波形从 medoid 扩展到 median template

P 波边界是低幅问题，median template 对 onset/offset 会比单 beat medoid 更稳。建议保留 medoid 作为代表展示，新增 median template 作为 boundary prior。

## 8. 推荐验证集

建议围绕以下记录做回归验证：

- **记录 5**：重点验证 P offset consensus 是否能减少 lead 间 offset 分散。
- **记录 39**：重点验证 supported consensus PR 是否能纠正全局 PR 偏短。
- **记录 57**：防止 naive consensus 把 PR 拉到错误的短 PR。
- **记录 120**：验证存在长 PR 支持 lead 时，算法是否能避免被短 PR 假候选主导。
- **记录 130**：验证 fused peak 错误时，cluster/provenance 能否识别低可信度，而不是错误共识化。

建议的自动化检查：

1. PR MAE 是否下降。
2. PR bias 是否从当前约 `-11.8 ms` 向 0 收敛。
3. PR > 20ms outlier 数是否下降。
4. P duration 的 lead 间 IQR 是否下降。
5. P offset 到 QRS onset 的异常贴近比例是否下降。
6. P area/PTF 的异常大值是否减少。

## 9. 小结

当前 P 波检测的问题不只是某个阈值偏松或偏紧，而是 P 边界还没有形成完整的“多 lead、多 beat、代表波形”闭环。

现在已经有了很好的基础：

- per lead/per beat 结果完整；
- 代表波形 prior 已存在；
- P peak 已经有跨 lead 融合；
- `pr_consensus_ms` 已经能在部分病例中捕捉到更好的信息。

下一步应把共识从 P peak 推进到 P onset/P offset/P duration，并让全局 PR 和 P 形态学特征真正使用有支持度的 consensus boundary。这样才能系统性改善 P onset 偏晚、P offset 分散、P duration 不稳，以及由此导致的 PR/PTF/P morphology 误差。

## 10. QRST residual raw P event 反哺更新

参考 `/home/chtmedgemma/projects/ecg_gemma/P wave` 中的 raw rhythm P/atrial activity 路线，当前实现补上了一个更完整但仍保守的闭环：

1. `feature_extraction/ecgfeat/atrial.py` 中的 composite QRST-residual detector 不再只输出 P event 的 peak/sample，而是同时输出 `onset_sample`、`offset_sample`、`duration_ms`、`amplitude_mv`、`area_mv_ms`、`signed_area_mv_ms`、`template_similarity`、`pp_ms` 等测量字段。
2. `feature_extraction/ecgfeat/delineate.py` 会在 per-lead/per-beat P context 打分后调用 composite raw P detector，把高置信 raw atrial event 转成对应 beat/lead 的 P alternative。
3. raw event 不直接覆盖原始 P 检测；它只作为候选进入已有 template similarity、PR consistency、PP consistency、multi-lead support 的 reselection gate。
4. 普通 alternative 仍只审查 `old_score < 0.50` 的候选；raw residual event 可以审查 `old_score < 0.65` 的 borderline 候选，但最终仍必须满足明显 context improvement。
5. export 层保留 raw P event 的新增测量字段，便于后续节律分析、AV block/blocked P 检查，以及调试 P 检测质量。

LUDB full-batch 验证：

- `tmp_p_raw_atrial_context_full/summary.csv` 相对当前 `ludb_full_compare/summary.csv`，PR/QRS/QT/QTc/T axis/QT dispersion/missing counts 均无变化。
- P axis MAE 极小改善：`17.3203977896` -> `17.3201184719`，变化来自 record 100 的两个 raw-event context reselection。
- 全量 200 条中生成 `2203` 个 composite raw atrial events，全部带 `duration_ms`；实际触发 raw-event P reselection 的 lead-beat 为 `7` 个。
- 生产代码扫描未发现 record ID、LUDB 文件名或特定记录条件分支；触发条件只依赖 raw residual event confidence、PR window、template/PP/PR/support context。

结论：这一步主要补齐 raw rhythm P event 的测量和反哺通道，属于基础设施型改进。它没有大幅改变现有 report 指标，但让 P 波原始检测具备了文档要求的 QRST residual + contextual second opinion，为后续更积极的 P onset/offset raw boundary 修正提供了安全入口。
