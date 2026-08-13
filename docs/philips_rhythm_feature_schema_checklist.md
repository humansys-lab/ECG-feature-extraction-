# Philips Rhythm Feature Schema Checklist

## 结论先行

当前 full features JSON（例如 `JS00024_features.json`）已经包含很多 rhythm 分析需要的测量特征：

- R/QRS 位置和 RR 间期
- beat-level P/QRS/T 边界
- PR/QRS/QT/JT/QTc
- P/QRS/T/ST axis
- P 波置信度、P 振幅、P 面积
- Q/R/S 振幅、QRS 面积、QRS notch/slur、delta flag
- per-lead quality 和 wave-specific reliability
- beat grouping、dominant group、group mean RR/PR/QRS/QT

但它还不能完整满足 `要点.pdf` 中 Philips-style rhythm analysis 的所有输入需求。主要缺口集中在：

- AF/AFL 的 QRST subtraction 和 atrial residual signal 特征
- pacing 的 continuous/intermittent/chamber/artifact/capture/sense 细分特征
- 独立 P-wave event stream，用于 P waves > QRS complexes、dropped beat、二度 AV block
- beat-level P morphology、QRS polarity signature、compensatory pause 等早搏精分类证据
- preexcitation 的 delta lead count、short-PR 降阈、accessory pathway side
- 面向 rule engine 的 statement evidence schema

## 数据层级判断

| 数据层级 | 能否支撑 `要点.pdf` rhythm 分析 | 说明 |
| --- | --- | --- |
| `.txt report` | 不够 | 适合人工阅读，只显示 HR/RR/PR/QRS/QT/axis/quality/per-lead median，缺少完整 beat-level 特征。 |
| full features JSON，如重新生成后的 `JS00024_features.json` | 可支撑 MVP，大量字段可用 | 包含 `beats`、`beat_features`、`representative_leads`、`groups`、`global_features`、`metadata`、`interpretation`，并新增 `rhythm_inputs`。 |
| `build_structured_payload()` 当前导出 | 不够 | 当前只导出 `signal / quality / beats / groups / global / provenance`，没有导出 `beat_features`、`representative_leads`、`interpretation`。见 `feature_extraction/ecgfeat/export.py:19-42`。 |

## 状态定义

- `OK`: 当前 full features JSON 已提供，可直接用于规则分析。
- `Partial`: 有相关字段，但不足以稳定支持 `要点.pdf` 中完整规则。
- `Missing`: 当前未发现可直接使用的字段或派生特征。

## Feature Checklist

| 模块 | `要点.pdf` 需要的特征 | 当前可用字段 | 状态 | 缺口 / 风险 |
| --- | --- | --- | --- | --- |
| 基础输入 | 采样率、导联顺序、记录长度 | `fs`, `metadata.input_fs`, `metadata.internal_fs`, `metadata.lead_order` | OK | full JSON 有；瘦 payload 也有。 |
| 患者信息 | age, sex | `metadata.patient_meta.age`, `metadata.patient_meta.sex` | OK | 儿童阈值仍需要外部完整年龄表。 |
| 信号质量 | per-lead quality、P/QRS/T/QT 可用性 | `quality.*.reliable_for_p/qrs/t/qt`, `flags`, `baseline_wander_score`, `muscle_noise_score`, `powerline_score` | OK | `JS00024_features.json` 里没有新版 `record_quality`；当前代码已在 `metadata.record_quality` 支持。 |
| R/QRS 检测 | R peak / QRS 位置 | `beats[*].r_index`, `metadata.qrs_detector.r_locs` | OK | 可支撑 RR、ventricular rate、pause、early beat。 |
| RR 特征 | `rr_prev_ms`, `rr_next_ms`, mean RR, RR variability | `beats[*].rr_prev_ms`, `beats[*].rr_next_ms`, `groups[*].mean_rr_ms`, `interpretation.rr_cv` | OK | 背景 RR 目前多用 median/mean，尚未明确输出“clean background RR excluding ectopy/pause/pacing”。 |
| 心室率 | ventricular rate | `global_features.heart_rate_bpm`, `groups[*].mean_ventr_rate_bpm` | OK | 可支撑 tachy/brady/complete AVB 的 rate 条件。 |
| 心房率 | atrial rate | `global_features.atrial_rate_bpm` | Partial | 是从 P 检测估计的单值；缺少独立 atrial event series 和置信度。 |
| P 波检测 | P onset/peak/offset per beat/lead | `beat_features[*].p.onset`, `p.peak`, `p.offset` | OK | QRS-centered P 检测可用，但不足以发现 dropped P 或独立 P event stream。 |
| P 波存在性 | has P wave / P confidence | `beat_features[*].p_confidence`, `representative_leads.*.params.p_confidence_mean` | OK | 当前规则可用阈值判断“有无 P”。 |
| P 波形态 | normal / atypical / retrograde / ectopic P | `interpretation.p_morphology_class`, P amplitude/area/duration | Partial | 当前 `p_morphology_class` 偏 atrial enlargement，不是 APC/JPC 所需的 beat-level atypical P morphology。 |
| P axis | frontal P axis | `global_features.p_axis_deg`, `interpretation.p_axis_normal` | OK | 可支撑 sinus origin 判断。 |
| PR interval | PR per beat/lead/global | `beat_features[*].pr_ms`, `representative_leads.*.params.pr_ms`, `global_features.pr_ms` | OK | 可支撑 AVB1、short PR、Wenckebach 的初步判断。 |
| PR segment | PR segment duration | 未发现直接字段 | Missing | WPW 规则中 `PR segment < 55 ms` 无法直接判断；只有 `baseline_source="pr_segment"` 不是 duration。 |
| PR trend | PR sequence before pause | 可从 `beat_features[*].pr_ms` + beat order 派生 | Partial | 当前可派生，但没有明确输出 per-beat global PR sequence/provenance。 |
| QRS duration | QRSd per beat/lead/global | `beat_features[*].qrs_ms`, `representative_leads.*.params.qrs_ms`, `global_features.qrs_ms` | OK | 可支撑 normal/wide QRS、VPC/JPC/APC 粗分类。 |
| QRS morphology | Q/R/S amplitude、R prime、S prime、notch/slur、VAT | `q_amp_mv`, `r_amp_mv`, `s_amp_mv`, `r_prime_amp_mv`, `s_prime_amp_mv`, `qrs_notch_count`, `qrs_slur_flag`, `vat_ms` | OK | 形态信息不少，足够进一步做 polarity/signature。 |
| QRS polarity signature | per-lead polarity vector / morphology vector | 可从 Q/R/S amplitudes 和 signed area 派生 | Partial | 没有直接输出 `qrs_polarity_signature` 或 beat-template embedding。 |
| QRS signed area | signed QRS area | 内部 model 有 `qrs_signed_area`；当前 `JS00024_features.json` 首个 beat_features key 列表未显示该字段 | Partial | 代码 model 有字段，但当前 JSON 可能由旧导出生成，需确认新导出是否包含。 |
| Delta wave | per-beat/lead delta flag | `beat_features[*].delta_present` | OK | 支撑 WPW 初筛。 |
| Delta lead count | number of leads with delta wave | 可从 `beat_features[*].delta_present` 聚合 | Partial | 没有直接输出 lead count、confidence、short-PR 降阈所需的配置证据。 |
| Initial QRS axis | initial QRS axis for pathway side | 未发现直接字段 | Missing | 只有 global QRS axis，不足以判断 left/right accessory pathway。 |
| QT/QTc | QT, QTcB, QTcF | `global_features.qt_ms`, `qtc_bazett_ms`, `qtc_fridericia_ms`, per-lead QT/QTc 可派生 | OK | rhythm 主线不是核心缺口。 |
| Beat grouping | group id, dominant group, group metrics | `beats[*].group_id`, `groups`, `metadata.representative_group_id` | OK | 支撑 dominant rhythm / ectopic group 粗分析。 |
| Paced flag | beat-level paced | `beats[*].paced`, `metadata.paced_beat_ids` | Partial | 只有在 `enable_pacing=True` 时可靠；常规调用可能关闭。 |
| Pacing spikes | spike timestamps | `global_features.pacing_spikes`, `metadata.pacing_state` | Partial | `JS00024_features.json` 为 `null/off`；默认 pipeline 通常未启用 pacing。 |
| Pacing type | atrial / ventricular / dual / AV sequential | 未发现直接字段 | Missing | 无法满足 `PACE-001/002/006/007` 的细分控制。 |
| Continuous/intermittent pacing | continuous_pacing, intermittent_pacing | 未发现直接字段 | Missing | 当前只有 beat tagging，不足以 stop non-paced rhythm-pattern analysis。 |
| Demand behavior | pulse inhibition evidence | 未发现直接字段 | Missing | 无 demand pacing behavior 特征。 |
| Pacemaker artifact | artifact confidence / noisy spike evidence | 未发现独立字段 | Missing | 当前 `pacemaker_like_artifact` 不是独立测量证据。 |
| Failure to sense/capture | spike-QRS/P decoupling, fixed-rate async spikes | 未发现直接字段 | Missing | 无 magnet/failure-to-sense/capture 分析输入。 |
| AF RR features | RR CV, irregularity, RMSSD/Lorenz/entropy | `interpretation.rr_cv`, `rr_irregularity_class`; RR sequence 可派生 | Partial | CV 有；RMSSD、entropy、Lorenz 类特征未输出。 |
| AF P-wave evidence | absent organized P waves | `p_confidence`, `probable_af` | Partial | 可粗判，但没有 atrial residual signal 支撑。 |
| QRST subtraction | atrial residual signal | 未发现直接字段 | Missing | 这是 AF/AFL 稳定区分的核心缺口。 |
| Atrial waveform stability | residual stability metric | 未发现直接字段 | Missing | 无法按 `要点.pdf` 区分 AF/AFL。 |
| Atrial repetitiveness | residual repetitive/autocorrelation/frequency feature | 未发现直接字段 | Missing | AFL 检测基本缺输入。 |
| Flutter evidence | flutter wave confidence / atrial cycle length | 未发现直接字段 | Missing | 当前没有 explicit atrial flutter feature。 |
| APC/JPC/VPC early beat evidence | RR shortening vs background RR | `beats[*].rr_prev_ms`; background RR 可从 groups 派生 | OK | 足够做 15% 早搏初筛。 |
| APC evidence | early + normal QRS + atypical P morphology | RR + QRS + P confidence 可用 | Partial | 缺 beat-level atypical P morphology。 |
| JPC evidence | early + normal QRS + no P | RR + QRS + P confidence 可用 | OK | 对“无 P”型 JPC 初筛够用。 |
| VPC evidence | early + wide QRS + polarity different + compensatory pause | RR + QRS 可用；QRS morphology 可派生 | Partial | 缺明确 polarity difference、compensatory pause、normal template comparison。 |
| VPC pair/NSVT | consecutive V beat run | 可从 beat classes 派生；当前 interpretation 有 `non_sustained_vt` | Partial | 当前 beat class 不导出，只导出最终 flag；证据链不完整。 |
| Bigeminy/trigeminy | N/V or N/A sequence pattern | 当前 interpretation 有 `bigeminy`, `trigeminy` 字段；`JS00024_features.json` 旧版缺这些字段 | Partial | 新 model 有字段，但当前 JSON 只含 37 个 interpretation 字段，未包含 advanced rhythm 字段。 |
| Pause | RR > 140% background RR | `beats[*].rr_next_ms`, group mean RR；新 interpretation 有 pause 字段 | Partial | 当前 `JS00024_features.json` 旧版未导出 `pauses_detected/pause_longest_ms`。 |
| Escape beat origin | P presence + QRSd after pause | P confidence + QRSd 可用于初筛 | Partial | 缺专门的 post-pause escape beat object 和 origin label。 |
| Second-degree AVB | P waves > QRS complexes | 未发现独立 P event count | Missing | QRS-centered P 检测不足以可靠统计 blocked P。 |
| Mobitz I | progressive PR lengthening before dropped beat | PR sequence 可派生；新 interpretation 有 `second_degree_avb` | Partial | 缺 dropped beat/P-only event 证据。 |
| AV dissociation | ventricular rate normal + apparent PR variation / AV asynchrony | PR sequence、atrial_rate、heart_rate 可用 | Partial | 缺 explicit AV asynchrony score 和 independent atrial/ventricular event streams。 |
| Complete AVB | ventricular rate <45 + AV asynchrony | HR 可用 | Partial | 缺 AV asynchrony score。 |
| Interpolated beats | prev/next RR around 0.5 background RR | `rr_prev_ms`, `rr_next_ms` 可用 | Partial | 可派生，但没有直接字段/flag。 |
| Aberrant complexes | slight RR shortening + wide QRS | RR + QRSd 可用 | Partial | 缺“slight shortening”专用阈值证据和 morphology differentiation。 |
| Evidence output | 每条 rhythm statement 的 evidence | `rhythm_inputs.statement_evidence` 与 `clinical_interpretation.domains` | Partial | 已覆盖当前公开规则集合；完整 Philips rhythm statement inventory 仍未实现。 |
| Rule priority flags | stop/bypass/suppression | `statement_engine` 与 `clinical_interpretation.suppressed_statements` | Partial | 已有确定性 stop/suppression；完整 rhythm engine 优先级仍是后续工作。 |

## JS00024 当前 JSON 快照

旧版 `JS00024_features.json` 当前确认包含：

- top-level: `fs`, `quality`, `beats`, `beat_features`, `representative_leads`, `groups`, `global_features`, `metadata`, `interpretation`
- beats: 19 条，字段为 `beat_id`, `r_index`, `paced`, `group_id`, `rr_prev_ms`, `rr_next_ms`
- beat_features: 228 条，字段覆盖 P/QRS/T bounds、PR/QRS/QT、P/QRS/T amplitude/area、delta、confidence、notch/slur、VAT、TPE、ST morphology 等
- representative_leads: 12 导联，每个有 `params` 和 `variance`
- global_features: HR、atrial rate、PR/QRS/QT/QTc、axis、QT dispersion、pacing、PTF-V1
- metadata: input/internal fs、lead order、lead reversal、n_beats、qrs_detector、representative_beat_meta、paced_beat_ids、patient_meta
- interpretation: 旧版 37 个字段，缺少新版 advanced rhythm 字段如 `premature_complexes`, `bigeminy`, `trigeminy`, `pauses_detected`, `complete_av_block`, `second_degree_avb`

重新生成 full features JSON 后，会额外包含 top-level `rhythm_inputs`，用于下游 rhythm rule engine。

## 已新增的最小目标 Schema

full features JSON 的 `rhythm_inputs` 已按下面这些块输出。当前能从已有测量稳定派生的字段会给出具体值；需要新增信号算法的字段会显式给出 `available=false` 和 `reason`。

### `rhythm_inputs.record`

- `fs`
- `duration_sec`
- `age_years`
- `sex`
- `lead_order`
- `record_quality`
- `rejected_functions`

### `rhythm_inputs.beats[]`

- `beat_id`
- `r_index`
- `r_time_ms`
- `rr_prev_ms`
- `rr_next_ms`
- `group_id`
- `paced`
- `pacer_spike_index`
- `qrs_duration_ms`
- `pr_interval_ms`
- `p_confidence`
- `p_axis_deg`
- `p_morphology`
- `qrs_polarity_signature`
- `delta_leads`
- `beat_quality`

### `rhythm_inputs.p_events[]`

- `p_event_id`
- `time_ms`
- `confidence`
- `associated_qrs_beat_id`
- `association_type`: `conducted`, `blocked`, `retrograde`, `unknown`
- `pr_ms`
- `axis_deg`
- `morphology`

### `rhythm_inputs.background`

- `clean_rr_ms`
- `background_rr_regular`
- `background_ventricular_rate_bpm`
- `background_atrial_rate_bpm`
- `dominant_group_id`
- `excluded_beat_ids`
- `exclusion_reasons`

### `rhythm_inputs.pacing`

- `enabled`
- `spike_times`
- `spike_count`
- `paced_beat_ids`
- `continuous_pacing`
- `intermittent_pacing`
- `ventricular_pacing_present`
- `atrial_pacing_present`
- `dual_chamber_pacing_present`
- `artifact_confidence`
- `capture_failure_suspected`
- `sensing_failure_suspected`

### `rhythm_inputs.af_afl`

- `rr_cv`
- `rr_rmssd`
- `rr_entropy`
- `qrst_subtraction_quality`
- `atrial_residual_signal_summary`
- `atrial_signal_stability`
- `atrial_signal_repetitiveness`
- `dominant_atrial_cycle_ms`
- `flutter_wave_confidence`

### `rhythm_inputs.preexcitation`

- `short_pr_interval`
- `short_pr_segment`
- `delta_lead_count`
- `delta_leads`
- `delta_confidence_by_lead`
- `mean_qrs_duration_ms`
- `initial_qrs_axis_deg`
- `accessory_pathway_side`

### `rhythm_inputs.statement_evidence`

- `primary_statement`
- `additional_statements`
- `statement.code`
- `statement.category`
- `statement.priority`
- `statement.evidence`
- `statement.confidence`
- `statement.suppresses`
- `stop_further_interpretation`
- `bypass_remaining_algorithm`

## 实用判断

当前 full features JSON 已经适合做一个 **MVP rhythm rule engine**：

- sinus/tachy/brady
- regular/irregular
- first-degree AV block / PR normal-short-long
- wide/narrow QRS
- simple WPW
- simple APC/JPC/VPC
- bigeminy/trigeminy
- simple pause

当前运行时不再执行或导出 Glasgow 解释层。可审计输出由
`statement_engine`、`clinical_interpretation.domains` 和每条规则的
evidence/threshold/source 字段提供。

它暂时不适合做完整 Philips-style rhythm engine 中这些部分：

- robust AF vs AFL
- paced rhythm statement control flow
- second-degree AV block with blocked P waves
- escape-beat origin
- ventricular pacing with AF-only atrial diagnosis
- magnet / capture / sensing failure
- accessory pathway localization
