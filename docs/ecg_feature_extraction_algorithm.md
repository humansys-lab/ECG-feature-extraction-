<!-- i18n-nav -->
[中文](ecg_feature_extraction_algorithm.md) | [English](ecg_feature_extraction_algorithm.en.md) | [日本語](ecg_feature_extraction_algorithm.ja.md)
<!-- /i18n-nav -->

# ECG 特征提取算法逻辑与流程说明

本文档基于当前仓库代码整理，日期为 2026-07-06。核心实现是
`feature_extraction/ecgfeat` 下的 `ECGFeatureExtractor.extract()`，整体是
DXL-inspired 的 12 导联 ECG 特征提取流程，不是 Philips DXL 私有算法的
逐行复刻，也不是经过验证的医疗器械软件。

## 1. 入口与整体数据流

### 1.1 主要入口

| 场景 | 文件/函数 | 作用 |
| --- | --- | --- |
| 单条记录演示 | `demo_feature_extraction.py` | 从本地 `dataset/*.mat/.hea` 读取 ECG 和患者信息，调用提取器，生成 `*_features.json`、报告和可视化。 |
| 批量 `.pt` 数据 | `batch_extract_ecgfeat.py` | 规范化 batch 输入形状，逐样本调用提取器，保存 JSON/PT/report/manifest。 |
| 算法主入口 | `feature_extraction/ecgfeat/api.py::ECGFeatureExtractor.extract` | 完整特征提取、起搏/节律策略、解释层和 metadata 组装。 |
| 导出 | `feature_extraction/ecgfeat/export.py::to_dict` | 将 dataclass 结果展开为 JSON 友好的结构，并补充 `rhythm_inputs`、`morphology_inputs`、`statement_engine`。 |

### 1.2 输入约定

核心输入是：

- `ecg_12lead`: `numpy.ndarray`，形状期望为 `[12, n_samples]`，单位为 mV。
- `fs`: 原始采样率。
- `meta`: 可选 `PatientMeta(age, sex, meds)`，用于成人/儿童规则路由、性别相关阈值和报告上下文。

如果 ECG 输入是 `[n_samples, 12]`，提取器会自动转置。如果既不是 12 行也不是 12 列，会抛出 `ValueError`。

标准导联顺序固定为：

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

### 1.3 主流程总览

```text
输入 ECG
  -> 采样率统一和双路预处理
  -> 导联质量评估、导联反接检测、起搏 spike 检测
  -> 多导联 QRS/R 峰检测
  -> 起搏 spike 与 QRS 校验，必要时去 spike 后重跑 QRS
  -> beat grouping 和 measurement group 选择
  -> representative beat 构建
  -> 逐搏逐导联 P/QRS/T 边界定位和基础测量
  -> 根据定位结果重新评估 measurement group
  -> atrial event、AF/AFL、起搏/AV block/preexcitation 规则输入
  -> measurement representative 和 representative lead 特征
  -> group/global 特征聚合，QT/PR/QRS rescue，axis 计算
  -> 12SL-style 并行 measurement profile
  -> rhythm availability 和 statement evidence
  -> clinical interpretation
  -> JSON 导出
```

## 2. 预处理

实现位置：`preprocess.py`

### 2.1 采样率

`fs_run` 由构造参数控制：

- `fs_internal is None`: 使用输入 `fs` 四舍五入后的整数。
- `fs_internal` 指定时：重采样到该内部采样率。

重采样使用 `scipy.signal.resample_poly`，按输入/输出采样率的最大公约数化简上下采样比。

### 2.2 两路信号

提取器保留两路不同用途的信号：

| 信号 | 构建方式 | 用途 |
| --- | --- | --- |
| `ecg_an` / measurement signal | 两级中值滤波去基线漂移，默认 200 ms 和 600 ms 窗口；再做工频 notch。无低通。 | 所有振幅、边界、ST/T/P/QRS 测量，尽量保留峰值。 |
| `ecg_det` / detection signal | `analysis_signal` 后再做默认 40 Hz 四阶低通。 | 只用于 QRS/R 峰检测，降低肌电噪声。 |

## 3. 质量评估与反接检测

实现位置：`quality.py`

### 3.1 每导联质量

`compute_quality()` 对每个标准导联计算：

- baseline wander: `0.5 Hz` 以下趋势相对整体标准差。
- muscle noise: `35-100 Hz` 频带功率占比。
- powerline noise: 工频 `mains_freq +/- 1 Hz` 功率占比。
- clipping: 相邻点差分为 0 的比例。
- flatline/missing: 标准差 `< 0.005 mV`。

输出 `LeadQuality`：

- 总体 `reliable`
- 波形专用可靠性：`reliable_for_p`、`reliable_for_qrs`、`reliable_for_t`、`reliable_for_qt`
- `grade`: `Q0/Q1/Q2/Q3`
- `flags` 和 `reason_codes`

主要门限：

- P 波比 QRS 更严格，baseline wander `< 0.30`、muscle `< 0.20`、powerline `< 0.10`。
- QRS 容忍更高肌电噪声，muscle `< 0.40`、powerline `< 0.20`。
- QT 需要 T 和 QRS 都可靠。

`summarize_record_quality()` 汇总全记录可靠导联数量，并给出：

- `record_grade`
- `rejected_functions`: 例如 `p_measurement`、`qt_measurement`、`record`
- 全局 reason codes

### 3.2 导联反接

肢体导联反接使用启发式：

- Einthoven 关系误差：`II - I - III`
- 导联相关性，例如 `I` 与 `-II`、`aVR` 与 `aVL/aVF`
- 输出 `probable_ra_la`、`probable_ra_ll`、`probable_la_ll`、`probable_extremity_reversal`

胸前导联反接在 representative lead 阶段根据 V1-V6 R 波进展检测：

- 若 R 波进展分数低，且某一对相邻胸导联交换后明显改善，则标记 `probable_precordial_reversal`。
- 低电压胸导联会跳过该判断。

## 4. 起搏 spike 检测与去 spike

实现位置：`quality.py`、`api.py`

起搏逻辑是主流程中的重要旁路，因为它会改变测量信号、beat 标记、QRS 宽度和 PR 可用性。

### 4.1 spike 候选

默认检测器：

- 对每导联做差分增强。
- 用滚动 MAD 估计局部噪声。
- 候选需要越过 `4 * sigma` 动态阈值。
- spike 宽度要求约 `<= 6 ms`。
- 默认绝对 prominence 门限为 `250 uV`，高幅 spike 参考门限为 `1000 uV`。

另有 legacy high-pass RMS 检测作为强证据 fallback。

### 4.2 多导联共识

候选按约 `5 ms` 窗口合并成事件。事件需满足：

- 默认至少 4 个导联投票，或弱投票加权分数足够高。
- 若有 3 个以上 spike，会去掉间期小于中位间期一半的近邻伪影。
- spike 间期 robust CV `< 0.15` 时认为节律 paced。

### 4.3 QRS 校验与去除伪影

初次 QRS 检测后，`validate_pacing_spikes_against_qrs()` 会排除 QRS 边缘被误识别为 spike 的情况：

- 计算 spike 与最近 R 峰的 offset。
- 比较 spike 高频峰值与 QRS 高频峰值。
- 宽度窄、相对 QRS 高频能量占优时保留。
- 若大量 spike 与 QRS 固定同步且像 QRS edge artifact，则清空 spike 并标记 `qrs_edge_artifact`。
- 对可能的 atrial/non-ventricular pacing 做保守处理，不直接影响 ventricular paced 测量。

若 spike 被确认，`remove_pacing_spikes()` 会在每个 spike 附近默认 `+-4 ms` 窗口线性插值，随后用去 spike 后的信号重建检测信号并重跑 QRS。

### 4.4 attenuated pacing rescue

如果初始起搏证据不足，`api.py` 会尝试降低 prominence 到 `150/100/25 uV` 的 rescue 路径。该路径只有在重新去 spike、重跑 QRS、并通过 QRS capture 校验后才会影响下游。

### 4.5 beat 级起搏状态

若 spike 落在 `R - 80 ms` 到 `R + 20 ms` 范围内，且 capture alignment 成立，则该 beat 被标记为 paced。后续会生成：

- `paced_beat_ids`
- `pacing_spike_beat_ids`
- `pacing_spike_offsets_ms`
- `pacing_capture_confirmed`
- `pacing_detection_state`
- `measurement_pacing_state`
- `pacing_measurement_effect`
- `pacing_segmentation_effect`

## 5. 多导联 QRS/R 峰检测

实现位置：`qrs.py`

### 5.1 检测信号

默认使用导联 `I, II, V2, V3, V4, V5, V6` 构建 vector magnitude：

```text
VM = sqrt(mean(zscore(lead)^2))
```

之后：

- bandpass `5-25 Hz`
- 一阶差分平方得到能量
- `120 ms` 移动平均

### 5.2 阈值和 R 峰精修

主阈值：

```text
max(P95(energy) * 0.30, mean + 0.5 * std)
```

峰间最小距离约 `220 ms`。每个能量峰在 `+-50 ms` 内用 lead II 局部波形精修 R 位置；若是负向 QS 主导波形，会把 fiducial 放在陡峭负向起始处，而不是晚期 S/QS 谷底。

边缘保护：距离记录边缘 `<160 ms` 的候选会被丢弃，避免无法完整测量。

### 5.3 robust fallback

若主检测过稀，会尝试基于 median/MAD 的 robust threshold。只有 robust 结果至少 3 个 beat、检测率合理且比主检测明显补足时才替换主结果。

## 6. Beat grouping 与 measurement group

实现位置：`grouping.py`、`api.py`

### 6.1 beat template

每个 beat 取 `R - 80 ms` 到 `R + 120 ms` 的 12 导联片段，每导联插值到 32 点，去均值并归一化后拼接为 template vector。

### 6.2 两阶段聚类

`cluster_beats()`：

1. 根据 vector magnitude 粗略估计 QRS 宽度，按 `<120 ms` 和 `>=120 ms` 分窄/宽两类。
2. 各类内部在线 morphology clustering，cosine similarity `>=0.88` 归入已有组。
3. 最多 5 个组，宽 QRS 分支最多 2 个组。
4. 按组大小排序，`group 1` 是 dominant group。
5. paced beats 会被单独重组，避免污染 native measurement family。

### 6.3 measurement group 选择

`api.py::_select_measurement_group()` 根据 paced beat 占比选择可测组：

- 若全局 paced majority，则优先 paced beat。
- 否则优先 native/non-paced beat。
- 如果后续 delineation 发现 paced group 不稳定且有稳定 native family，会重新选择 measurement group。

最终结果写入：

- `representative_group_id`
- `measurement_beat_ids`
- `measurement_group_reselected`
- `measurement_group_reselect_reason`

## 7. Representative beat 构建

实现位置：`representative.py`

每个 group 构建代表搏动：

- 窗口：`R - 300 ms` 到 `R + 500 ms`
- 用 vector magnitude 做 cross-correlation 对齐，最大 lag `+-30 ms`
- 对不规则节律，如果组内 beat 数 `>=6` 且 RR CV `>0.15`，会优先保留 RR 在组中位数 `+-20%` 内的 beat
- 使用 morphology family assignment 选择 dominant family
- 用 medoid 代表 dominant family，必要时用相关系数阈值 `0.70` 排除离群 beat
- fallback 为 per-sample median

代表搏动主要用于：

- 提供 P/QRS/T 边界先验
- 构建 measurement representative beat
- 稳定 representative lead 参数

## 8. 逐搏逐导联 P/QRS/T 边界定位

实现位置：`delineate.py`、`repolarization.py`、`p_morphology.py`

### 8.1 beat 窗口和搜索范围

每个 beat 默认分析窗口：

```text
beat_start = R - 350 ms
beat_end   = R + 550 ms
```

T-end 不能越过下一次 QRS 前的保护区：

```text
t_cap = next_R - max(150 ms, 28% * RR)
```

P 波 lookback 自适应：

- 首 beat: `400 ms`
- 其他 beat: `min(450 ms, 65% * previous_RR)`

### 8.2 representative prior 和多导联 P/T anchor

若已有 representative beats，会先对代表搏动做一次粗 delineation，得到每个 group/lead 的边界 offset 先验。

逐搏测量时还会做 beat-level P/T anchor fusion：

- 从所有导联收集 P/T peak candidate triplets。
- 结合质量标记和 representative prior。
- P anchor cluster 半径约 `12 ms`。
- T anchor cluster 半径约 `20 ms`。
- 融合后的 anchor 会缩窄每导联局部搜索窗，但不会硬覆盖所有边界。

### 8.3 QRS 边界

`_qrs_bounds()` 的核心：

- 对单导联信号 bandpass `5-30 Hz`。
- 用一阶导数平方能量，`12 ms` 平滑。
- 阈值为 `max(4% * QRS peak energy, 1.5 * noise_floor)`。
- 若 QRS detector 使用低斜率 fallback，会启用 low-slope fiducial guard。
- 用二阶导数 zero-crossing 对 onset/offset 做细化。
- 对持续低斜率 QRS foot 做额外 rescue。
- 返回 onset/offset、notch count、onset/off confidence。

起搏 beat 特殊处理：

- 若 spike 在 beat 附近，QRS onset 不允许早于 spike。
- 若 capture 确认且 QRS 过窄，会对 paced beat 应用最小 QRS 宽度 `120 ms` floor。
- 如果 intrinsic QRS 明显窄，floor 会被抑制，避免 QRS edge artifact 被误当 paced QRS。

QRS offset raw repair second pass：

- 在每个 lead × beat 完成初始 QRS 测量后，先用 reliable QRS leads 建立 beat-level measurement terminal consensus。
- 对明显偏离 consensus、offset 低置信、且被多导联共识判为 late/early outlier 的 raw `qrs.offset`，允许局部回查并修正原始 `WaveBounds`。
- 修正后立即重算该 lead×beat 的 `qrs_ms`、QRS area/signed area、`j_index`、ST-J/ST40/ST80、JT/QT 和可安全重算的 terminal morphology。
- 该 pass 不改 QRS onset，不把所有导联强行拉到同一 offset；宽 QRS/BBB/起搏分类仍保留 `qrs_wide_ms` 路径。
- 若原始 QRS 宽度本身合理、且没有 ST/T confusion 或 beat-unreliable 证据，过度缩短的修复会被拒绝。

ST/J raw remeasurement second pass：

- 在 QRS offset raw repair 之后，用 reliable QRS leads 建立 beat-level J-point measurement consensus。
- 对 `st_j_unreliable`、J 点与 ST40/ST80 跳变过大、局部 J 点明显偏离 consensus 或 beat 不可靠的 lead×beat，在同导联原始波形上重新采样 ST-J/ST40/ST80。
- 修正后重算 `st_on_mv`、`st_mid_mv`、`st_80ms_mv`、`st_slope_mv_per_ms` 和 `st_morphology`。
- 该 pass 不改变 `qrs.onset`、`qrs.offset` 或兼容性的 `j_index`；实际 ST 采样 anchor 写入 `st_j_remeasured_index`，并记录原始 index、delta、原因、支持导联和排除导联。
- 若 J/ST40/ST80 内部一致，说明可能是真实 ST elevation/depression，不做重测。
- LUDB full-batch 中 report-level PR/QRS/QT 和 missing 指标保持不变，同时重测了 1724 个 raw ST/J lead×beat 特征，覆盖 164 条记录。

### 8.4 P 波

native P 波路径：

- 搜索窗在 QRS 前，受 fused P anchor 或 prior 限制。
- 候选 peak 必须落在大致生理 PR 范围内，当前代码使用约 `60-350 ms` 前范围校验。
- 振幅需要至少 `max(8 uV, 1.5% * R amplitude)`。
- onset 使用上升支 tangent，offset 使用下降支 tangent。
- 在最终 candidate re-selection 后，额外生成零相位 `35 Hz` 低通的 P
  检测视图；原始 measurement signal 仍用于幅度、面积和形态测量。
- 后续拍只从上一拍 T offset 与本拍 P onset 之间经过验证的 TP quiet 窗估计
  局部噪声；PR/PQ 段含 Ta，不再作为噪声参考。首拍只能使用明确标记为
  `previous_t_out_of_context` 的未验证 pre-P fallback。输出
  `p_local_noise_rms_mv`、`p_local_snr` 和 `p_informative`。低于 `2 sigma`
  的候选保留作低置信覆盖 fallback，但不参与高置信边界次序统计。
- 输出 `p_tp_gap_ms`、`p_quiet_window_available`、`p_on_t_overlap_risk`、
  `p_ta_overlap_risk`、`p_baseline_method` 与 `p_baseline_confidence`。
  TP 消失时仅在 P 候选两侧拟合连续鲁棒二次趋势，用于边界稳定度诊断，
  不覆盖原始边界；边界置信度封顶并强制多导联融合回退到 cluster median。
- 用低通 P 视图上的 tangent 与 `4%-12%` 阈值扰动测量
  `p_onset_sigma_ms` / `p_offset_sigma_ms`，将 P 波存在置信度与边界稳定度分开。
- long PR 时可 rescue 更早 P onset。
- 若 P/QRS 几何关系不合理，丢弃 P 或只保留不可用于 PR 的信息。
- P candidate context 会为每个原始候选计算 template similarity、multilead support、PR consistency 和 PP consistency。短 PR 候选若 template 与 PP timing 同时较弱，且只由胸导/非 limb lead 支持，即使有多个 lead 同时误检，也会被作为 noise-like P candidate 抑制，避免最终 PR report 掩盖 raw P onset/offset 错误。
- representative P axis 使用 `p_area` 结合 `p_amp_mv` 符号得到 signed P area，并结合 P candidate context 做 limb-lead polarity consensus。若 I/II/aVF 等高上下文 P 波形成稳定 atrial vector，低 confidence/低 context 且投影方向冲突的 aVR/aVL/III contributor 会被排除，避免 QRS 前 Na/Q 谷或 baseline 偏移把 P axis 翻到反向。
- 若整体 P vector 呈明显反转，但 PR 支持为稳定 sinus-like 范围，且 I/II/aVF/aVR 给出足够反转投票，会将该组 P-axis contributor 整体翻回，修复“真实 P 被 QRS 前负向谷压过”的极性问题。真正由多个高上下文 lead 支持、但没有 sinus-like PR 证据的负向 P vector 不会被强行改回。
- P axis 还要求最小 frontal support：少于 2 个肢体导联不报轴；只有 2 个导联时必须有稳定 PR 支持，否则报 N/A。这样避免用单导联或无 PR 支持的双导联噪声硬算 180 度等假轴。

paced beat P 波路径：

- 在 QRS offset 后 `40-250 ms` 搜索 retrograde P。
- 振幅需约 `>=0.03 mV`。
- 宽度需 `<=120 ms`。
- 若与 QRS/T 重叠，则丢弃。

### 8.5 T 波和 T-end

T 波搜索从 QRS 后开始：

- paced beat: 至少 QRS offset 后 `40 ms`。
- 宽 QRS: QRS offset 后 `40 ms`。
- 窄 QRS: QRS offset 后 `20 ms`。
- 同时有 R+40/R+80 ms 等保护，避免落在 QRS tail/J wave 上。

`detect_t_wave()` 会结合：

- 导联预期极性，例如 I、II、aVF、V5、V6 期望正向。
- 多导联 cluster polarity。
- ST-T confusion 标记。
- T 搜索窗内按 baseline crossing 切分的显著 component；component 面积达到 `160 uV*ms` 才作为 T/T' 形态证据。
- 面积和宽度支持的显著 component 会反哺 T peak 选择，抑制窄的早期 ST spike 抢占真正 T peak。
- 若 T peak 后存在反向、baseline-crossing、面积显著的 T' component，初始 T offset 可延伸到最后一个显著 component 末端，并把 `t_prime_peak` 传入 12SL-style profile。

T-end 优先使用 chord/geometric 方法：

- 从 T peak 到搜索终点画 chord。
- 取极性方向上的最大 chord distance 位置作为 T-end。
- 对低振幅或短 arm 使用 threshold fallback。

额外保护：

- 双相/切迹 T 会尝试延伸到第二成分。
- 若有显著 U 波，T-end 可夹到 T-U nadir。
- representative prior 只允许在合理情况下延长 T-end，避免错误缩短 QT。
- 若 QT `<200 ms`，T-end 被丢弃。
- 在 global QT 聚合阶段，如果 reliable raw QT 的多导联集群稳定且明显晚于 consensus QT，说明 consensus T-end 可能过早，算法会使用 late raw QT cluster rescue。这不是按 record 特判，而是要求正常 QRS、规律 RR、足够 lead 支持、T 可见且 T-end confidence 足够。

T-end raw repair second pass：

- 在每个 lead × beat 完成初始 T peak/T-end 测量后，先用 reliable QT leads 建立 beat-level T-end measurement consensus。
- 在 general consensus repair 之前，先运行更保守的 dual-method raw rescue：只处理明显早截断、低置信或 fallback/risky 的 raw `t.offset`，并要求 chord、slope-return、tangent 三种 T-end 端点 spread `<=30 ms`。
- dual rescue 会记录独立 provenance：`t_offset_dual_original_index`、`t_offset_dual_rescued_index`、`t_offset_dual_consensus_index`、`t_offset_dual_support`、`t_offset_dual_chord_index`、`t_offset_dual_slope_index`、`t_offset_dual_tangent_index` 和 `t_offset_dual_method_spread_ms`。
- 对明显早于或晚于 consensus、且低置信或带 `t_end_fallback` / `st_t_confusion` / `beat_unreliable` 证据的 raw `t.offset`，允许在同导联信号上重新寻找 T-end。
- 修正后立即重算 `qt_ms`、`jt_ms`、`t_dur_ms`、`t_area`、`t_signed_area` 和 `tpe_ms`。
- 高置信且形态合理的真实长 T 不会被强行拉回 consensus；若修复候选会导致 QT `<280 ms`、跨过下一 QRS guard、或 dual-method 分歧过大，则拒绝修复。
- 该 pass 不改变 QRS，也不改变 QT dispersion/report comparison 口径。
- T-end dual-method raw rescue 和 T-end consensus raw repair 在 QRS offset raw repair 与 ST/J raw remeasurement 之后、最终 `_apply_multilead_consensus()` 之前运行。

### 8.6 每搏每导联输出

`LeadBeatFeatures` 会保存：

- `p/qrs/t` 三类 `WaveBounds(onset, peak, offset)`
- `pr_ms/qrs_ms/qt_ms/jt_ms`
- `p_amp_mv/q_amp_mv/r_amp_mv/s_amp_mv/t_amp_mv`
- `qrs_area/qrs_signed_area`
- `st_on_mv/st_mid_mv/st_80ms_mv`
- Q 波 duration/area/ratio，initial QRS area/net
- R'/S' 振幅和 duration，QRS notch/peak count
- VAT、P/T duration、P/T area、P terminal components、PTF-V1
- Tpe、ST morphology、fragmented QRS score、U-wave flag
- 12SL-style per-beat ST/QRS/T profile 字段
- beat-level noise/baseline/template correlation/reliability
- P/QRS/QT confidence 和 flags

### 8.7 多导联 consensus

`_apply_multilead_consensus()` 以 beat 为单位融合可靠导联边界：

| 全局边界 | 来源 |
| --- | --- |
| QRS onset | reliable_for_qrs 导联的 P10；长 RR 且 onset 分散时可用更居中的 onset。 |
| QRS offset | reliable_for_qrs 导联的终末 offset，measurement 通常用 P75；paced 或低支持时可用 median。 |
| QRS wide offset | classification 用 P90，paced 时用 P75。 |
| P onset | 生理 onset cluster；局部 SNR/边界稳定度证据充分时取受保护的第二早 onset，否则使用 cluster median/P10 fallback。 |
| P offset | 可靠导联支持的 offset cluster；诊断证据充分时取受保护的第二晚 offset，并保留 paired-duration guard。 |
| T end | reliable_for_qt 导联的 median。 |

P onset/offset 使用联合几何回退：如果第二早 onset 会令全部支持的 offset
违反宽松 P-duration 约束，则该拍回退到 onset cluster median，而不是丢失
完整 P 边界。

晚期 QRS terminal offset 有保护：

- 若某些导联晚 offset 像 ST/T tail，且没有足够 terminal morphology 支持，会标为 `classification_only`，不进入 measurement QRS。
- 若支持足够，晚 offset 可用于 measurement 或 wide classification。
- QRS offset raw repair 会先于最终 `_apply_multilead_consensus()` 运行，使明显错误的 lead×beat raw offset 在进入 consensus/report 前被修正。

Consensus 写回每个 beat/lead：

- `qrs_consensus_ms`
- `qrs_wide_ms`
- `pr_consensus_ms`
- `qt_consensus_ms`

QT 还有 per-lead 和 global sanity：若 implied QTcB `>590 ms`，相关 T-end 会被过滤或全局 T-end 置空。

## 9. Representative lead 特征

实现位置：`features.py::build_representative_lead_features`

该阶段把 selected measurement beats 汇总成每导联代表参数：

- 优先使用 measurement representative beat 的测量。
- fallback 使用 selected beat pool 的中位数/均值/众数。
- 汇总 beat-to-beat variance，例如 `pr_ms_sd/qrs_ms_sd/qt_ms_sd/jt_ms_sd/st_on_mv_sd/t_amp_mv_sd`。
- 写入每导联 reliability、confidence、measurement source。
- 若 P 波支持不足，会把 `pr_ms/pr_consensus_ms` 置空并标记 `p_measurement_suppressed=low_p_support`。
- 根据 V1-V6 R 波进展做 precordial reversal 检测。

## 10. Group 和 global 特征

实现位置：`features.py`

### 10.1 group features

`compute_group_features()` 按 beat group 统计：

- member count 和百分比
- longest run
- mean RR/PR/QRS/QT
- mean ventricular rate
- `dominant_group`
- `wide_qrs`

### 10.2 global HR 和 PR

HR 使用 R-R 中位数：

```text
HR = 60000 / median(RR_ms)
```

PR 初始来自 representative leads 的全局 PR 中位数；若 atrial measurement invalid，例如 AF/AFL 或 P 支持不足，可置空。

`api.py` 还有 physiologic PR core rescue：

- 从逐搏 PR 中提取核心值。
- 若节律允许、QRS 窄、RR 规则且 PR core 在 `120-240 ms`，会替换明显异常或缺失的 global PR。

### 10.3 global QRS

global QRS 优先来自 consensus measurement QRS：

- measurement 用 `qrs_consensus_ms`
- wide classification 可参考 `qrs_wide_ms`
- 若起搏或 pacing-like context 下 raw/consensus 不一致，`api.py` 会应用多条 QRS override 路径，防止 paced 或宽 QRS 被低估或被 ST/T tail 过度拉长。

### 10.4 global QT 和 QTc

QT 是当前实现中最复杂的聚合路径。

可靠 QT 导联需满足：

- `reliable_for_qt`
- T 波可见，默认 `|T_amp| >= 0.05 mV`
- QT beat-to-beat SD `<25 ms`
- T-end confidence `>0.20`
- QT 在生理范围内

若严格路径没有导联，rescue pass 可选 1 个最佳导联：

- QT SD `<30 ms`
- confidence `>0.10`

宽 QRS/起搏时 QT path 使用 JT 合理性：

```text
JT = QT - QRS
允许范围约 160-550 ms
```

QT 聚合优先级大致为：

1. reliable leads 的 consensus QT median。
2. 宽 QRS raw QT rescue。
3. raw QT core rescue。
4. reliable raw QT cluster rescue。
5. late raw QT cluster rescue：正常 QRS/规律 RR 下，若稳定 raw QT 集群比 consensus QT 晚至少约 `20 ms`，用 raw 集群防止短 QT outlier。
6. split raw QT rescue。
7. reliable leads 的 amplitude-weighted median。
8. global reliable lead fallback。
9. irregular beat QT core。
10. raw QT late core rescue。
11. preferred lead fallback: `II, V5, V4, V6, III, aVF`。
12. low-support late raw QT cluster rescue：当 `low_qt_support` 路径只能给出偏短 QT，但窄 QRS、RR 规则、多个 reliable/global lead 的 raw QT 形成紧密晚期集群时，用该 raw 集群修正 global QT；该路径不回写每导联 raw QT，因此不会把 QT dispersion 强行压平。

QT path decision 会标记：

- `normal_qrs`
- `wide_qrs_jt`
- `paced_jt`
- `low_qt_support`

最终输出：

- `qt_ms`
- `qtc_bazett_ms = QT / sqrt(RR_sec)`
- `qtc_fridericia_ms = QT / cbrt(RR_sec)`
- `qt_dispersion_ms`
- `qt_source`
- `qt_used_leads`
- `qt_reliability`
- `qt_path`
- per-lead consensus vs independent 对照

若 `qt_reliability == low_confidence` 或没有可靠 QT 导联，解释层会抑制 QTc prolongation 判读。

### 10.5 Axis

frontal axis 使用六个肢体导联的 signed area 或振幅估计：

- P axis: P signed area/P amplitude，要求 P 可靠。
- QRS axis: QRS net area/amplitude，必要时可用 peak-net 或 initial-core QRS axis 修正。
- T axis: T signed area/amp，经 ST-corrected、cluster、suppression 和 fallback 多层处理。
- ST axis: ST mid value。

T axis 额外保护：

- 宽 QRS、起搏、T 振幅过低、极性冲突、ST-T confusion、RR 明显不规则时可能置空。
- 若 I/II/aVF 形成强而同向的 limb-seed T 共识，aVR/aVL/III 中与 seed 投影明显冲突且缺少足够置信支持的值会被排除，避免单个 limb lead 极性误检拉偏 T axis。
- ST-corrected T-axis 只使用 ST confidence 足够的 ST samples；低 QRS offset confidence、QRS offset repair 或 ST 样本不稳定时，STJ/STM/STE 仍输出，但不主导 T-axis。
- 若 QT support 低导致常规 T axis 被抑制，但 QRS 窄、RR 规则、多个肢体导联有稳定 T amplitude/signed area 且轴不极端，则可走 low-support T-axis coverage fallback。12SL-style profile 写回 STJ/STM/STE、special T 和 ST confidence 后，API 层还会做一次窄范围 T-axis backfill，只填补原本 N/A 的稳定低支持 T 轴。这个 fallback/backfill 只恢复“形态证据稳定但 QT confidence gate 过保守”的记录，不用于宽 QRS/ST plateau 或低振幅不稳定 T。
- `t_axis_reliable` 要求至少 2 个肢体导联 `|T_amp| > 0.15 mV`。

LUDB comparison 的 axis 汇总同时保留 signed raw diff 和 circular absolute diff：

- `diff_p_axis/diff_qrs_axis/diff_t_axis` 仍是 algorithm - ground truth 的直接差值，用于查看方向。
- `abs_p_axis_circular_diff/abs_qrs_axis_circular_diff/abs_t_axis_circular_diff` 使用圆周角差，避免 `176°` vs `-175°` 被误计为 `351°+` 的大错。

## 11. Atrial、AF/AFL 和 rhythm availability

实现位置：`atrial.py`、`rhythm_rules.py`、`api.py`

### 11.1 atrial event extraction

`extract_atrial_events()` 来源包括：

- 每导联 P peak，要求 `p_confidence >=0.30` 且导联 `reliable_for_p`。
- inter-beat atrial scan。
- 如果原始 ECG 可用，优先 composite raw P detection；否则 fallback 到粗略 per-lead scan。

候选会按时间聚类、去重，并关联最近 QRS：

- `conducted`
- `blocked`
- `retrograde`
- `unknown`

### 11.2 QRST residual 与 AF/AFL

`build_qrst_subtracted_residual()` 构建 QRST window residual summary：

- 优先尝试 template subtraction，但它只服务 AF/AFL residual，不被视为
  P-on-T 的通用解。QRS 保持 R 对齐；ST-T 按逐拍导数相关性平移，并以
  Huber 式迭代重加权拟合幅值/偏置；QRS→ST-T 和模板窗两端使用余弦渐变。
- 只有 template correlation `>=0.85`、beat 数足够、残差 RMS ratio 合格，
  且二阶差分高频能量比 `<=1.25` 时，才标记 validated QRST subtraction。
- 中位 RR `<520 ms` 时，模板支持窗可能侵入下一次 P 区域，连续 QRST
  residual 路径直接不可用。
- 否则仍暴露 scaffold summary，供 rule input 使用。

`_classify_af_afl()`：

- RR CV `>=0.15` 且 organized P ratio `<0.25`、residual repetitiveness 不强时，倾向 probable AF。
- flutter 使用 residual dominant cycle `120-400 ms` 和 spectral evidence。
- flutter confidence `>=0.60` 时标记 probable flutter，并避免同时判 AF。

### 11.3 rhythm rules

`rhythm_rules.py` 提供：

- pacing context: continuous/intermittent/ventricular/atrial/dual chamber pacing。
- pacing failure: spike 后 `160 ms` 内没有 QRS 时疑似 capture failure。
- preexcitation/WPW: short PR、short PR segment、delta leads、QRS `>=100 ms`、initial QRS axis。
- pauses 和二度 AV block: RR pause、atrial events per RR、PR series。
- post-pause escape/interpolated beat candidates。
- measurement availability。

### 11.4 measurement availability

以下场景会使 atrial rhythm、PR、P-axis 等测量不再适合下游 rhythm 判读：

- continuous pacing
- wide QRS pacing-like context
- probable AF
- probable flutter
- complete AV block
- AV dissociation
- atrial measurements unavailable

`api.py::_apply_measurement_availability_to_representatives()` 会在强不可用情境下把 global PR 和 representative PR 字段置空。

## 12. 12SL-style 并行 measurement profile

实现位置：`twelve_sl.py`

该模块生成并行 profile，不直接覆盖 native DXL-style 主测量。

每 beat/lead 计算：

- STJ、STM、STE
- STM offset = RR / 16
- STE offset = RR / 8
- ST/T amplitude reference = QRS onset 电压
- QRS area、signed area、balance、deflection
- minimum ST
- T/T'：T' 由 T peak 后显著 baseline-crossing component 推断，显著面积阈值 `160 uV*ms`
- ST confidence：低 QRS offset confidence、QRS offset repair、STJ/STM/STE 不稳定或 ST sample 缺失会降低 `twelve_sl_st_confidence`，并记录 `twelve_sl_st_confidence_reason`。
- special T amplitude：基础规则为 `min(T, T - STE)`；但当 ST confidence 较低时，STE 不参与 special T 分支，避免 ST tail/QRS offset 误差污染 T morphology。显著负向 T' 会覆盖 special T；小负向 T' 在正向 T 至少为其 4 倍且 `|T'| < 70 uV` 时被忽略；负向 T 且无 T' 时还比较 T offset 电压分支。
- QRS significance，阈值 `160 uV*ms`

record-level profile 包括：

- first-last QRS heart rate
- multilead global fiducials: P/QRS/T offset 相对 R 的中位摘要
- constants 和 profile version

导出层提供三套 measurement profile：

- `native`
- `12sl`
- `hybrid`

hybrid 策略是 ST/QRS/T profile 值优先使用 12SL 可用值，而 global axis 和 QT dispersion 保留 native。

## 13. Clinical interpretation 层

实现位置：`interpret.py`、`mi.py`、`pediatric_rules.py`、`statement_engine.py`

解释层不重新处理原始波形，而是消费：

- `global_features`
- `representative_leads`
- `beats`
- `beat_features`
- `metadata["rhythm_analysis"]`

### 13.1 判读流程

代码按类似临床读图的层次组织：

1. 技术质量和导联反接。
2. Rhythm 和 rate。
3. Axis 与 intervals。
4. P/QRS/T/ST morphology。
5. 解剖定位、MI、hypertrophy、reciprocal changes。

### 13.2 rhythm/rate

- HR `<45 bpm`: `extreme_bradycardia`
- HR `<50 bpm`: `bradycardia`
- HR `<100 bpm`: `normal`
- HR `>=100 bpm`: `tachycardia`
- RR CV `>0.15`: `irregular`
- RR CV `>0.08`: `mildly_irregular`
- irregular 且 P confidence 低时 probable AF。

### 13.3 axis/conduction/QTc

- QRS axis: normal `[-30, 90]`，LAD/LAFB/RAD/LPFB/ERAD 分层。
- T axis: normal `[-10, 100]`。
- PR: short `<120 ms`；一度 AVB 阈值使用年龄和 HR 动态表。
- QRS:
  - `<100 ms`: normal
  - `100-110 ms`: borderline IVCD
  - `110-120 ms`: nonspecific IVCD
  - `>=120 ms`: BBB/IVCD，并根据 V1 R'、lateral S、V1/V6/I 振幅判 RBBB/LBBB。
- QTc:
  - `<340 ms`: short
  - `>465 ms`: borderline prolonged
  - `>485 ms`: prolonged
  - `>520 ms`: significantly prolonged
  - BBB、RVH/LVH 等可抑制 QTc prolongation statement。

### 13.4 morphology 和解剖定位

解释层覆盖：

- WPW/preexcitation
- RAE/LAE/BAE，包含 P duration、P amplitude、V1 terminal component 和 PTF-V1
- pathological Q waves 和 territory: inferior/anterior/lateral
- R-wave progression 和 R/S transition
- ST elevation/depression territories、STEMI-style codes、reciprocal changes
- LVH voltage/score，含 Cornell/Sokolow/aVL 等
- low voltage
- RVH graded score
- tall T
- dextrocardia
- COPD pattern
- posterior MI 和 culprit artery evidence
- statement candidates 和 suppression/bypass

LBBB 会抑制继发性复极相关判断：

- ST elevation/depression
- T-wave abnormality
- Q-wave infarct signs
- reciprocal changes

### 13.5 儿童规则

年龄 `0 <= age < 16` 时走 pediatric morphology route：

- 儿童 QRS axis 和 QRS duration 使用年龄分箱阈值。
- pediatric QTc 阈值按年龄和性别变化。
- RVH/LVH/LSH/BVH 使用 `pediatric_rules.py` 中的年龄分箱电压阈值。
- RBBB/LBBB 可 bypass 部分 hypertrophy 判读。
- 特定年龄范围内处理 pericarditis 和 early repolarization。

## 14. 输出结构

当前批处理和 demo 保存的 `*_features.json` 使用 `export.to_dict()`，顶层包含：

```text
fs
quality
beats
beat_features
representative_leads
groups
global_features
metadata
interpretation
rhythm_inputs
morphology_inputs
statement_engine
clinical_interpretation
reference_metadata
```

### 14.1 主要分区

| 分区 | 内容 |
| --- | --- |
| `quality` | 每导联质量、flags、Q0-Q3、各波形可靠性。 |
| `beats` | 每个 beat 的 R index、paced、group id、前后 RR。 |
| `beat_features` | 每 beat x lead 的 P/QRS/T bounds、intervals、amplitudes、ST、morphology、confidence、flags。 |
| `representative_leads` | 每导联 representative 参数和方差。 |
| `groups` | beat group 摘要。 |
| `global_features` | HR、PR、QRS、QT/QTc、axis、QT provenance、pacing status 等。 |
| `metadata` | 输入/内部采样率、QRS detector、record quality、lead reversal、pacing、measurement group、rhythm analysis。 |
| `interpretation` | 临床派生类别和 flags。 |
| `rhythm_inputs` | rhythm rule layer 的规范化输入。 |
| `morphology_inputs` | morphology rule layer 的规范化输入，含 native/12SL/hybrid profile。 |
| `statement_engine` | statement candidate、final/suppressed/bypassed 结果。 |
| `clinical_interpretation` | 当前权威临床规则结果、coverage、suppression 和审计证据。 |
| `reference_metadata` | 仍保留的旧解释层及其 reference-only/supersession 状态。 |

另有 `export.build_structured_payload(features)` 可生成稳定分区式 payload：

```text
record / signal / quality / beats / groups / global / pacing /
rhythm_inputs / morphology_inputs / statement_engine / provenance /
clinical_interpretation / reference_metadata /
schema_version
```

当前 structured schema 为 `ecgfeat_structured_payload.v3`。v3 不再包含 Glasgow
解释区块。

注意：当前 `to_dict()` 导出的样例 JSON 不会自动包含名为 `structured_payload` 的顶层字段，除非调用方显式调用 `build_structured_payload()`。

## 15. 批处理流程

实现位置：`batch_extract_ecgfeat.py`

批处理输入支持：

- 单条 `[12, n_points]`
- 单条 `[n_points, 12]`
- batch `[n_samples, 12, n_points]`
- batch `[n_samples, n_points, 12]`

统一规范化到：

```text
[n_samples, 12, n_points]
```

每个样本输出：

- `{sample_id}_features.json`
- `{sample_id}_features.pt`
- `{sample_id}_report.txt`
- 可选 `{sample_id}_ecg_annotated.png`
- `manifest.json`

## 16. 关键设计约束和已知限制

1. 本项目是 DXL-inspired research scaffold，不是经监管验证的诊断软件。
2. 质量、起搏、AF/AFL、宽 QRS、LBBB 等场景会触发多处 suppression/rescue，不能只看单个 raw interval。
3. `beat_features` 是逐搏逐导联原始测量视角，`representative_leads` 是 measurement group 摘要，`global_features` 是用于下游解释的全局聚合视角。
4. 12SL profile 是并行输出面，不等价于替换 native 测量。
5. QRST subtraction 只有满足 template validation 时才算 validated；否则只是 scaffold residual summary。
6. paced beat 专用 delineation 仍是工程实现和保护逻辑，复杂起搏类型仍需要更多验证。
7. 儿童阈值依赖 age/sex；缺失或非法 age 会默认走 adult route。
8. 导联反接检测是启发式，会影响 axis、precordial progression 和 interpretation，但不是最终医学确认。

## 16.1 已退役的 Glasgow 解释层

Glasgow 规则框架已从当前运行时解释链中移除：

- `ECGFeatureExtractor.extract()` 不再执行或缓存 Glasgow 分析；
- `to_dict()` 和 structured v3 不再导出顶层 `glasgow`；
- 临床冲突解析、文本/图形报告和 MedGemma 上下文不再消费 Glasgow 结论；
- 导出器会清理旧对象中的 `metadata.glasgow_analysis`；
- 旧报告进入 MedGemma 前会过滤历史 Glasgow appendix。

历史 `glasgow_rules/` 文件暂时保留，供旧实验和结果追溯使用，但不属于当前
解释层。delineation 中沿用的 `glasgow_measurements` 是底层数值测量 profile，
不是诊断解释，因此本次没有删除，避免改变 P/QRS/T 基础测量。

## 17. 推荐阅读顺序

如果需要从代码继续深入，建议按以下顺序阅读：

1. `feature_extraction/ecgfeat/api.py::ECGFeatureExtractor.extract`
2. `preprocess.py`、`quality.py`、`qrs.py`
3. `grouping.py`、`representative.py`
4. `delineate.py`
5. `features.py`
6. `atrial.py`、`rhythm_rules.py`
7. `twelve_sl.py`
8. `interpret.py`
9. `export.py`
