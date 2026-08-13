# Per-Lead/Per-Beat、代表波形与多导联共识利用审计

分析对象：

- 当前代码：`feature_extraction/ecgfeat`
- 重点样例：`ludb_full_compare/5/5_features.json`
- 目标问题：各波形、各 segment、各特征是否正确遵循“先每个 lead 每个 beat，再合成代表波形，再利用多导联共识”的测量策略。

## 1. 总体结论

当前架构已经具备三层信息：

1. `beat_features`：每个 `lead × beat` 的 P/QRS/T/ST/interval/morphology 独立测量。
2. `representative_leads`：按导联汇总后的代表导联参数。
3. `*_consensus_ms` 与 `global_features`：跨导联共识后的全局 interval。

但是利用方式不完全一致：

- QRS/QT 全局 interval 比较充分地用了多导联共识。
- P/PR 部分用了 per-lead/per-beat 和 limb-lead 汇总，但代表波形和跨导联 P onset 共识利用不足。
- ST segment 已新增 raw remeasurement pass：QRS offset raw repair 之后会用 beat-level J consensus 重测有风险的 ST-J/ST40/ST80；但 representative-template ST 测量仍有提升空间。
- QT dispersion 虽然用 independent per-lead QT，但会选择 core cluster，容易压低导联间离散度。
- 代表波形目前常常是 medoid 单拍，不一定是多拍平均/中位模板，因此对降噪和边界精修的贡献有限。
- report/comparison 存在 raw、representative、consensus 混用，容易误判算法本身。

换句话说：**框架方向是对的，但“代表波形”和“多导联共识”还没有被所有波形/segment 以一致、可追踪的方式充分利用。**

## 2. 当前流水线是否符合目标结构

### 2.1 已经做到了的部分

主流程在 `feature_extraction/ecgfeat/api.py`：

1. 先做多导联 QRS/R peak 检测。
2. 之后按 R peak 建 beat。
3. 对 beat 进行 morphology grouping。
4. 为每个 group 构建 representative beat。
5. `delineate_beats()` 对每个 beat × lead 独立提取边界和特征。
6. `delineate_beats()` 内部再执行 `_apply_multilead_consensus()`，写入 `qrs_consensus_ms`、`pr_consensus_ms`、`qt_consensus_ms`。
7. 再构建 measurement representative beat，并对这个代表 beat 再做一次 delineation。
8. `build_representative_lead_features()` 把 selected beats 和 representative beat feature 汇总成每导联代表参数。
9. `compute_global_features()` 从 representative leads 和 beat features 汇总全局 PR/QRS/QT/axis/dispersion。

对应代码位置：

- `api.py:1303-1420`：QRS 检测、分组、代表波形、beat×lead delineation。
- `api.py:1474-1506`：measurement representative beat 重新构建和 representative lead features。
- `delineate.py:1964`：每个 beat × lead 的 P/QRS/T/ST 提取。
- `delineate.py:1596`：多导联 consensus。
- `features.py:348`：构建 representative lead features。
- `features.py:2001`：计算 global features。

### 2.2 关键限制

当前 representative beat 并不总是“多拍平均模板”。`build_representative_beats_with_meta()` 会做对齐、family clustering、outlier 排除，然后优先选 family medoid。若 medoid 存在，`used_count=1`。

record 5 中：

- `member_count = 9`
- `outlier_count = 1`
- `used_count = 1`

这说明代表波形实际上是“代表性单拍”，不是 8 个干净 beat 的 median/average 模板。这个设计能保留真实形态，但降噪能力不如 median template。

建议后续同时保留两种 representative：

- `medoid_template`：用于形态分类、Q/R/S/notch/slur 等。
- `median_template`：用于 P onset、QRS onset/offset、T end、ST segment 等边界和低噪测量。

## 3. 各波形/segment 的利用情况

## 3.1 P 波与 PR segment/PR interval

当前做法：

- 每个 lead × beat 独立检测 P onset/peak/offset。
- P morphology，如 notched、biphasic、terminal negative、PTF-V1，在每个 beat×lead 上测量。
- representative lead 汇总时，P amplitude/area/morphology 优先取 representative beat feature；没有时取 pooled beats median。
- PR interval 在 representative lead 中优先取 pooled median，而不是代表波形。
- global PR 用 limb leads 优先的 median，并有 physiologic PR core rescue。

优点：

- P morphology 已经利用了代表波形，有助于稳定 P 波形态。
- global PR 没有盲目用所有 precordial leads，避免 V leads 假 P 拉短 PR。
- record 5 中 global PR 为 130 ms，GT 为 138 ms，误差只有 -8 ms，说明 core rescue 对该例有效。

不足：

- PR/P onset 仍有系统性偏短风险。此前 summary 中 PR bias 约 -11.77 ms。
- 代表波形没有作为 PR interval 的主路径，interval 先用 pooled median。
- 多导联 P onset consensus 只写 `pr_consensus_ms`，但 global PR 主要仍走 `_global_pr_median()` 的 per-lead PR，不优先使用 `pr_consensus_ms`。
- 低幅 P 或长 PR 时，代表波形的 SNR 优势没有完全释放。

建议：

1. 为 P 波保留 `p_on_independent`、`p_on_template`、`p_on_consensus` 三套 landmark。
2. global PR 改成优先：
   - 多导联 P onset consensus。
   - limb-lead raw PR core。
   - representative-template PR。
3. 对长 PR/低幅 P，用 median representative template 重新测 P onset。
4. PR segment/baseline 不要只依赖单拍局部 PR 段，可用同导联代表 beat 的 PR segment 中位数做 baseline prior。

## 3.2 QRS 波群

当前做法：

- 每个 lead × beat 独立找 QRS onset/offset、Q/R/S、R'/S'、notch、slur、VAT、QRS area。
- representative beat 先用于构建 group-level boundary prior，帮助单拍局部修正。
- `_apply_multilead_consensus()` 用多导联 QRS onset/offset 得到 `qrs_consensus_ms` 和 `qrs_wide_ms`。
- global QRS 优先使用 consensus，并有 raw core fallback。

优点：

- QRS 是当前最充分利用多导联共识的部分。
- per-lead WaveBounds 不被 consensus 覆盖，保留了导联形态差异。
- late terminal QRS 有 classification-only / measurement 分流，避免把 ST/T tail 全部当 QRS。

record 5 暴露的问题：

- Global QRS = 112 ms，GT = 80 ms，误差 +32 ms。
- independent per-lead QRS 差异很大：
  - I: 70 ms
  - II: 80 ms
  - III: 191 ms
  - aVF: 128 ms
  - V1: 66 ms
  - V5: 82 ms
- `qrs_consensus_ms = 112 ms`，`qrs_wide_ms = 128 ms`。

这说明当前 consensus 确实把 III 的 191 ms 极端值压下去了，但仍被若干 late offset 拉宽，导致 global QRS 偏宽。

不足：

- QRS consensus 对晚 offset 的鲁棒性还不够。
- representative template 没有作为“标准 QRS 终点”强约束 consensus。
- QRS offset 错误会传递到 ST-J、JT、QT path。

建议：

1. QRS global measurement 应加入 `representative_median_template_qrs` 作为第三方证据。
2. 若 per-lead raw QRS core 在 70-90 ms，而 consensus >110 ms，应要求 late offset 至少有多导联形态证据。
3. QRS offset consensus 应输出：
   - measurement offset
   - classification/wide offset
   - raw core offset
   - template offset
4. ST-J/ST40/ST80 应基于 measurement offset；BBB/IVCD 分类可以基于 classification offset。

### QRS offset raw repair update

当前新增了 QRS offset raw repair pass。它在每个 beat 内用 reliable QRS leads 的 measurement terminal consensus 识别明显 late/early 且低置信的 raw `qrs.offset` outlier，只修被判定为 measurement 错误的 lead×beat，不把所有 lead 强行拉到同一个 offset。

修正后会回写 `LeadBeatFeatures.qrs.offset`，并重算 `qrs_ms`、`qrs_area`、`qrs_signed_area`、`j_index`、`st_on_mv`、`st_mid_mv`、`st_80ms_mv`、`jt_ms`、`qt_ms` 以及可安全重算的 terminal QRS morphology 字段。宽 QRS classification 仍保留 `qrs_wide_ms` 路径，不与 measurement offset 混用。

为避免把合理 QRS 过度缩短，repair gate 还会保护原始宽度合理、且没有 ST/T confusion 或 beat-unreliable 证据的 lead×beat；只有明显 long-tail、低置信或不可靠 context 才允许被共识反哺修正。

LUDB full-batch 结果：

- QRS MAE: 9.88 -> 9.88
- QRS bias: 5.727 -> 5.727
- `|QRS diff| > 20 ms` / `> 40 ms`: 17/2 -> 17/2
- QT MAE: 13.4625 -> 13.4625
- QRS/P/T missing: 0/2669/258 -> 0/2669/258
- raw QRS any-boundary missing: 0 / 25860
- `qrs_offset_repaired_by_consensus`: 7 rows across 6 records

## 3.3 ST segment / J point

当前做法：

- ST-J、ST40、ST80 在每个 lead × beat 上测量。
- representative lead 的 `st_on_mv` 通过 `_representative_st_j_params()` 汇总。
- 如果发现 QRS tail guard，会标记 ST-J unreliable。
- 12SL profile 另有 STJ/STM/STE。
- QRS offset raw repair 之后，新增 ST/J raw remeasurement pass：对有 `st_j_unreliable`、tail jump、J offset outlier 或 beat-unreliable 证据的 lead×beat，用同一 beat 内 reliable QRS leads 的 J-point consensus 在原始导联波形上重采样 ST-J/ST40/ST80。

优点：

- ST 是 per-lead 特征，没有被全局 consensus 简单覆盖。
- 有 QRS tail guard，可避免部分 QRS 末端污染 ST。
- ST/J raw remeasurement 不改变 `qrs.onset`、`qrs.offset` 或兼容性的 `j_index`，只把实际 ST 采样 anchor 写入 `st_j_remeasured_index` 并重算 ST 值，因此可以保留真实导联间 QRS 差异。

不足：

- 未触发 raw remeasurement 的 ST-J 仍主要基于每导联局部 QRS offset，而不是 representative-template J point。
- `_apply_multilead_consensus()` 仍只在 `qrs_wide_ms` 与 ST-J/ST40/ST80 冲突时把 ST-J 置为 unreliable；真正的重测发生在更早的 ST/J raw remeasurement pass。
- representative beat 对 ST segment 的使用不够稳定：ST-J 有时取 rep_feature，有时取 pooled median，但没有明确使用 consensus J point 重采样。

### ST/J raw remeasurement update

当前新增了 ST/J raw remeasurement pass。它在 QRS offset raw repair 之后，用同一 beat 内 reliable QRS leads 的 J-point consensus 作为采样 anchor，只重测带 `st_j_unreliable`、tail jump、offset outlier 或 beat-unreliable 证据的 lead×beat ST segment。

该 pass 不改变 `qrs.onset`、`qrs.offset` 或兼容性的 `j_index`，而是把 ST 采样 anchor 记录到 `st_j_remeasured_index`。修正后会重算 `st_on_mv`、`st_mid_mv`、`st_80ms_mv`、`st_slope_mv_per_ms` 和 `st_morphology`。若 J/ST40/ST80 已经一致，真实 ST elevation/depression 不会被抹平。

LUDB full-batch 结果：

- PR MAE: 9.5372 -> 9.5372; QRS MAE: 9.8800 -> 9.8800; QT MAE: 12.8575 -> 12.8575
- PR/QRS/QT `|diff| > 20 ms`: 17/17/33 -> 17/17/33
- QRS/P/T missing: 0/2669/258 -> 0/2669/258
- `st_j_unreliable`: 305
- `st_j_remeasured_by_consensus`: 1724 rows across 164 records
- reason split: `offset_outlier=1092`, `tail_guard=417`, `beat_unreliable=215`

record 5 中：

- I lead representative ST-J = -0.104 mV，但 pooled median ST-J 约 -0.069 mV。
- aVL representative ST-J = +0.042 mV，但 pooled median ST-J 约 -0.061 mV。

这说明代表单拍可能与 pooled beats 的 ST level 不一致。如果代表波形是 medoid 单拍，这种 ST-J 差异可能来自单拍基线/offset，而不是稳定 ST segment。

建议：

1. 对 ST segment 建立独立的 representative-template measurement：
   - `st_j_raw_local`
   - `st_j_consensus_j`
   - `st_j_template`
   - `st_j_12sl`
2. 诊断/报告优先用 `st_j_template_consensus_j` 或 reliable pooled median，而不是单个 medoid ST-J。
3. 若 representative ST-J 与 pooled median 差异 >0.05-0.08 mV，应标记不稳定或回退 pooled median。
4. ST segment 应绑定 QRS measurement offset，而不是 wide/classification offset。

## 3.4 T 波与 QT

当前做法：

- 每个 lead × beat 独立检测 T peak/T end。
- T candidate 选择考虑极性、面积、宽度、ST-T confusion。
- 多导联 consensus 用 reliable QT leads 的 T-end median 得到 `qt_consensus_ms`。
- representative lead 中保留 independent `qt_ms` 和 consensus `qt_consensus_ms`。
- global QT 优先走 reliable leads median，并记录 `consensus_vs_independent_per_lead`。

优点：

- QT 是当前除 QRS 外最充分利用多导联 consensus 的特征。
- record 5 中 III independent QT = 504 ms，V1 = 328 ms；global QT 使用共识后为 402 ms，没有被单导联极端值直接拖走。
- `global_qt` metadata 保留了每个导联 independent vs consensus QT，便于审计。

不足：

- T 波形态、T amplitude、T area 仍可能来自 representative beat 或 pooled raw，不一定使用 consensus T-end 重新积分。
- `qt_consensus_ms` 在 global QT 中优先级很高，可能牺牲真实导联差异。
- low-support/rescue 路径误差比主路径明显更大。

record 5 中：

- Global QT = 402 ms，GT = 382.5 ms，误差 +19.5 ms。
- independent QT：
  - V1: 328 ms
  - V2/V3: 368 ms
  - I/II/V5: 384 ms
  - aVF: 416 ms
  - III: 504 ms
- consensus QT = 402 ms。

这说明共识抑制了 outlier，但整体 T end 或 QRS onset 仍略偏晚/偏长。

建议：

1. T morphology 应区分：
   - raw local T bounds
   - consensus T end
   - representative-template T bounds
2. T area/T duration/Tpe 若用于诊断，应在 representative template 上按 consensus T-end 重新计算。
3. `st_t_confusion` 的 lead 不应直接参与 T morphology 主表，除非代表波形和邻近导联一致支持。
4. global QT 可继续优先 consensus，但 report 必须同时显示 independent QT 和 consensus QT。

### T-end raw repair update

当前新增了 T-end raw repair pass。它在每个 beat 内用 reliable QT leads 的 T-end measurement consensus 识别明显 early/late 且低置信或带风险证据的 raw `t.offset` outlier，只修被判定为 measurement 错误的 lead×beat，不把所有 lead 强行拉到同一个 T-end。

修正后会回写 `LeadBeatFeatures.t.offset`，并重算 `qt_ms`、`jt_ms`、`t_dur_ms`、`t_area`、`t_signed_area` 和 `tpe_ms`。高置信且形态合理的真实长 T 不会被强行拉回 consensus；若修复候选会导致 QT `<280 ms`，则拒绝修复，避免把 QRS tail/ST-T confusion 误当成 T-end。

在 consensus raw repair 之前，当前还新增了一个更保守的 T-end dual-method raw rescue pass。它只处理“明显早截断、低置信或 fallback/risky”的 raw `t.offset`，先用可靠 QT leads 建立 beat-level T-end consensus，再在同一导联原始波形上同时计算 chord、slope-return、tangent 三种 T offset 证据。只有三种端点 spread `<=30 ms`、候选不会跨过下一 QRS、不会导致 QT `<280 ms`、且延长不超过 `160 ms` 时，才把 raw `t.offset` 写回。

该 pass 的目标不是提高最终 report 聚合，而是让原始 `lead × beat` T offset 本身更可信。它单独记录 `t_offset_dual_original_index`、`t_offset_dual_rescued_index`、`t_offset_dual_delta_ms`、`t_offset_dual_consensus_index`、`t_offset_dual_support`、`t_offset_dual_used_leads`、`t_offset_dual_chord_index`、`t_offset_dual_slope_index`、`t_offset_dual_tangent_index` 和 `t_offset_dual_method_spread_ms`，避免与后续 consensus repair provenance 混淆。

LUDB full-batch 结果：

- QT MAE: 13.4625 -> 12.8575
- QT bias: -1.1875 -> -1.2125
- `|QT diff| > 20 ms` / `> 40 ms`: 34/17 -> 33/14
- QRS MAE: 9.88 -> 9.88
- PR MAE: 9.537236842105264 -> 9.537236842105264
- QRS/P/T missing: 0/2669/258 -> 0/2669/258
- raw T peak/offset missing: 378 / 25860
- `t_offset_dual_method_rescued`: 1 row across 1 record
- dual rescue case: record 54, lead V1, beat 5, `3522 -> 3534` samples (`+24 ms`), support 8, method spread `28 ms`
- `t_offset_repaired_by_consensus`: 1201 rows across 167 records
- consensus repair reason split: 1001 `late_outlier`, 200 `early_truncation`

## 3.5 U 波

当前做法：

- U wave 主要作为 T-end 后的 secondary peak flag。
- 明显 U 波会影响 T-U nadir 和 T-end clamp。
- 没有独立的 U onset/peak/offset 代表波形测量。

不足：

- 没有完整 per-lead/per-beat U wave feature pipeline。
- 没有 representative U morphology。
- prominent U 对 QT/T-end 的影响可以标记，但不够可解释。

建议：

1. 若要系统利用 U wave，应新增 `u` bounds 和 `u_amp_mv/u_area/u_confidence`。
2. QT 与 QU 应分开输出：`qt_ms`、`qu_ms`。
3. T-end 被 U wave 修正时，在 metadata 中保留原始 T-end 和 T-U nadir。

## 3.6 QT dispersion

当前做法：

- `_raw_qt_dispersion()` 使用 representative leads 的 independent `qt_ms`，不是 consensus QT。
- 但它会过滤低 confidence、高 SD、不可见 T、path 不合理的 leads。
- 如果 raw dispersion 太大，会选 core cluster，返回核心导联簇的 max-min。

record 5 中：

- Algorithm QT dispersion = 22 ms。
- GT QT dispersion = 64 ms。
- independent QT 其实有明显差异：V1 328 ms，III 504 ms，aVF 416 ms。
- 当前逻辑把这些外侧值过滤/聚类后，只保留核心簇，所以 dispersion 被压到 22 ms。

结论：

- 当前不是完全错误地用 consensus QT 算 dispersion，但“core cluster”策略会系统性低估临床意义上的导联离散度。

建议：

1. 同时输出：
   - `qt_dispersion_raw_all_reliable`
   - `qt_dispersion_core_cluster`
   - `qt_dispersion_reported`
2. 与 LUDB GT 比较时，优先用更接近 annotation 定义的 raw independent lead dispersion。
3. 在 report 中不要只显示 core dispersion，否则会隐藏导联间差异。

## 3.7 Axis

当前做法：

- P/QRS/T axis 从 representative leads 的 signed area 或 net amplitude 计算。
- 使用 confidence 作为权重/过滤。

优点：

- Axis 本质需要跨导联组合，当前没有只依赖单导联。
- 使用 signed area 比单点 peak 更稳。

不足：

- 如果代表导联的 P/T area 来自 medoid 单拍，噪声和 ST baseline 会影响 axis。
- T axis 缺失较多，说明 T amplitude/confidence gating 偏保守。
- comparison 中 axis diff 没做 circular distance，会放大部分误差。

建议：

1. Axis 使用 median representative template 的 signed area，而不是 medoid 单拍。
2. 对 T axis 输出 reliability 和 used leads。
3. comparison 使用 circular difference。

## 4. Report / comparison 的混用问题

当前 `compare_annotations.py` 中 per-lead QRS/QT 比较优先使用 consensus：

- `qrs_ms -> qrs_consensus_ms`
- `qt_ms -> qt_consensus_ms`

这会导致 `*_comparison.txt` 的 per-lead QRS/QT 看起来 12 导联完全相同。

record 5 中：

- algorithm report 的 per-lead 表显示 raw representative：
  - I QRS 70, II QRS 80, III QRS 191, aVF QRS 128
  - I QT 384, III QT 504, V1 QT 328
- comparison 表却显示所有导联 QRS = 112、QT = 402。

所以 comparison 的 per-lead QRS/QT 不是纯 per-lead representative measurement，而是 consensus measurement。这个口径会掩盖：

- 哪些导联独立测得很差。
- representative waveform 是否改善了该导联。
- consensus 是否过度压平导联差异。

建议修改 comparison/report：

1. per-lead comparison 中分列：
   - `raw_independent`
   - `representative_template`
   - `pooled_median`
   - `consensus`
2. summary 中 global QRS/QT 用 consensus。
3. per-lead 表不要默认用 consensus 替代 raw。
4. QT dispersion 比较要注明使用 raw 还是 core cluster。

## 5. 特征级审计表

| 特征 | per lead × beat | representative waveform | multi-lead consensus | 当前评价 |
|---|---|---|---|---|
| P onset/offset | 有 | 有，但 interval 不优先用 | 有 `pr_consensus_ms`，但 global PR 不优先用 | 部分充分 |
| P morphology | 有 | 优先 representative/pooled | 不需要强 consensus | 基本合理 |
| PR interval | 有 | 不是主路径 | 有但未充分用于 global PR | 需加强 |
| Baseline/PR segment | 有局部 baseline | 未充分用代表模板 baseline | 无明确 consensus | 需加强 |
| QRS onset/offset | 有 | 用作 prior | 强 consensus | 基本充分，但 terminal offset 需改 |
| Q/R/S/R'/S' | 有 | 优先 representative/pooled | 不应全局覆盖 | 合理 |
| ST-J/ST40/ST80 | 有 | 有但 medoid 可能不稳 | 只做 guard，不重测 | 需加强 |
| ST morphology/slope | 有 | 有 | 弱 | 需加强 |
| T peak/T end | 有 | 有 | T-end consensus 强 | 基本充分 |
| T amplitude/area | 有 | 优先 representative/pooled | 未按 consensus T-end 重积分 | 需加强 |
| QT interval | 有 | pooled/representative 均保留 | global 优先 consensus | 基本充分，但 outlier/rescue 需改 |
| JT interval | 有 | 有 | 间接受 QRS/QT consensus 影响 | 部分充分 |
| Tpe | 有 | pooled interval 优先 | 弱 | 需加强 |
| U wave | flag 级别 | 无完整代表测量 | 无 | 不充分 |
| QT dispersion | 用 independent QT | 用 representative leads | 避免直接用 consensus，但 core cluster 会压低 | 定义需拆分 |
| Axis | representative leads | 有 | 通过导联组合实现 | 基本合理 |

## 6. 推荐的目标架构

建议把每个特征都显式分成四层，而不是混在一个字段里：

1. `raw_lead_beat`：单导联单拍原始测量。
2. `template_lead`：同导联代表波形测量。
3. `beat_multilead_consensus`：同一 beat 的跨导联 fiducial consensus。
4. `global_reported`：用于最终报告的稳定全局值。

字段示例：

```text
qrs_ms_raw_lead
qrs_ms_template_lead
qrs_ms_consensus_beat
qrs_ms_global_reported

qt_ms_raw_lead
qt_ms_template_lead
qt_ms_consensus_beat
qt_ms_global_reported

st_j_raw_local
st_j_template_consensus_j
st_j_reported
```

## 7. 优先修改建议

### P0：先改报告/比较口径

先把 raw、representative、consensus 分开显示。否则很难判断算法是否真的利用了代表波形。

重点文件：

- `compare_annotations.py`
- report/export 相关表格生成

### P1：代表波形同时保留 medoid 与 median template

medoid 适合形态，median template 适合边界和 segment。

建议：

- `representative_medoid`
- `representative_median`
- `representative_mean_or_trimmed_mean`

### P2：ST segment 重新绑定 consensus J point

对每个 lead 的 representative median template，用 consensus QRS offset 重新测：

- ST-J
- ST40
- ST80
- ST slope

### P3：QRS terminal consensus 加 template guard

当 global QRS 比 raw/template core 明显更宽时，要求 late offset 有多导联 terminal morphology 支持。

### P4：QT dispersion 拆成 raw/core/reported

不要只输出 core cluster dispersion。LUDB 对比应使用更接近 independent lead annotation 的版本。

### P5：P onset/PR 使用 representative + consensus 主路径

尤其针对长 PR、低幅 P、LAE/RAE、噪声场景。

## 8. 对 record 5 的具体判断

record 5 说明当前流程有用，但还不够：

- HR 与 PR 尚可：HR 误差 -0.2 bpm，PR 误差 -8 ms。
- QRS 偏宽：global QRS 112 ms vs GT 80 ms，说明 QRS consensus 被 late offsets 拉宽。
- QT 偏长：global QT 402 ms vs GT 382.5 ms，部分来自 QRS/global fiducial 偏差。
- QT dispersion 低估：22 ms vs GT 64 ms，来自 core cluster 过滤。
- ST-J 在部分导联 representative 与 pooled median 差异明显，说明 medoid 单拍不一定适合 ST segment。

因此，record 5 的改进重点不是“是否有 representative/consensus”，而是：

1. QRS terminal consensus 要更受 raw/template core 约束。
2. ST segment 要用 representative median template + consensus J point 重测。
3. QT dispersion 要用 independent lead values 重新定义。
4. comparison 表要停止把 per-lead QRS/QT 默认替换成 consensus。
