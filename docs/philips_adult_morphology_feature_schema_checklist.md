<!-- i18n-nav -->
[中文](philips_adult_morphology_feature_schema_checklist.md) | [English](philips_adult_morphology_feature_schema_checklist.en.md) | [日本語](philips_adult_morphology_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

# Philips Adult Morphology Feature Schema Checklist

## 结论先行

> 2026-07-12 状态更新：当前导出已新增 top-level
> `clinical_interpretation`（`clinical_rules.v1` / ruleset `2026.07.1`）。这是以公开
> AHA/ACCF/HRS、ESC 与 Fourth Universal Definition of MI 为依据的权威规则层；
> `interpretation` 和 `statement_engine` 保留并明确标为 `reference_only`；
> Glasgow 解释层已从当前运行时和导出契约中移除。项目不等同于、也不宣称
> 复刻 Philips/DXL 私有算法。
> 每条统一规则均输出 status、required/missing inputs、evidence、thresholds、source
> 与 suppression；缺少必要证据会返回 `unavailable`，不会作为阳性或“正常”的依据。

当前 full features JSON（例如 `JS00029_features.json`）已经能支撑一版 **成人 morphology MVP**，尤其是这些模块：

- QRS/P/T/ST frontal axis 与 QRS-T angle
- PR/QRS/QT/QTc/JT 等全局 interval
- per-lead P/QRS/T 边界、P/QRS/T 振幅、ST J-point/mid/80ms
- Q/R/S/R'/S' 幅度、QRS area、notch/slur、VAT、fQRS score
- T amplitude、T polarity、TPE、U wave flag
- PTF-V1、P duration、P area
- LVH voltage、low voltage、ST depression/elevation、pathological Q、R progression、tall T 等初步解释字段

重新生成 full features JSON 后，会额外包含 top-level `morphology_inputs`，把当前测量整理成面向 morphology rule engine 的输入 schema。但它还不能完整满足 `要点adult_morphology.pdf` 里整理的 Philips-style adult morphology engine。主要缺口不是“完全没测量”，而是：

- 已有结构化候选语句和统一规则审计，但只覆盖当前已实现的公开规则集合；尚未覆盖完整 Philips code matrix 与 downgrade/replacement 语义。
- P-wave 精细形态缺口明显：`is_notched`、`is_biphasic`、initial/terminal duration/amplitude/area 只有部分近似。
- QRS component duration 不完整：Q duration 当前是近似，R'/S'/Q/S duration 未直接测量。
- Dextrocardia、BBB、RVH、MI 等需要的 terminal / horizontal / initial QRS direction 没有作为明确 fact 输出。
- ST shape 与 slope 单位不完全匹配原文：当前有 `st_slope_mv_per_ms` 和 `upsloping/horizontal/downsloping`，原文需要 degrees 与 straight/concave-up/concave-down。
- ST Map、Cabrera 坐标、V4R/V7-V9 扩展导联、culprit artery 输出仍缺。
- 药物/临床输入不足：`meds` 有位置但当前样本多为 `null`，没有独立 `clinical_dx_codes` / `rx_codes` 规则上下文。

所以判断是：**公开准则的统一 MVP 已可运行并作为最终规则输出；Philips/DXL 风格结果仅作参考。要完整复刻 Philips adult morphology 仍缺专有 code matrix、部分底层形态事实和经验证的专有仲裁逻辑。**

## 文档范围

本清单对照两个本地 PDF：

- `/home/chtmedgemma/projects/ecg_gemma/philips_adult_morphology.pdf`
- `/home/chtmedgemma/projects/ecg_gemma/要点adult_morphology.pdf`

`要点adult_morphology.pdf` 是从 Philips Adult Morphology Analysis 整理出的工程化规格，覆盖 dextrocardia、心房异常、QRS axis、VCD/BBB、RVH、LVH、低电压/COPD、MI、ST/T/QT、电解质/药物效应与 suppression engine。

## 数据层级判断

| 数据层级 | 能否支撑 adult morphology 分析 | 说明 |
| --- | --- | --- |
| `.txt report` | 不够 | 适合人工阅读，缺少完整 per-lead/per-beat morphology facts、候选语句和 suppression 证据。 |
| 当前旧 full features JSON，如 `JS00029_features.json` | 可支撑 MVP，大量字段可用 | 包含 `beats`、`beat_features`、`representative_leads`、`global_features`、`metadata`、`interpretation`；但没有 `morphology_inputs`。 |
| 重新生成后的 full features JSON | 更适合下游规则引擎 | 当前 `to_dict(ECGFeatures)` 会新增 `rhythm_inputs` 和 `morphology_inputs`，并保留原始测量。 |
| `build_structured_payload()` 瘦 payload | 不够 | 不导出完整 `beat_features`、`representative_leads`、candidate evidence，不适合作为 morphology rule engine 输入。 |

## JS00029 当前 JSON 快照

当前 `JS00029_features.json` 是旧导出，确认包含：

- top-level: `fs`, `quality`, `beats`, `beat_features`, `representative_leads`, `groups`, `global_features`, `metadata`, `interpretation`
- beats: 14 条
- beat_features: 168 条，12 leads x 14 beats
- beat-level fields: P/QRS/T bounds, `qt_ms`, `pr_ms`, `qrs_ms`, P/Q/R/S/T amplitude, ST J/mid/80ms, `j_index`, `delta_present`, `jt_ms`, confidence, R'/S', QRS peak count, QRS notch count, VAT, P/T duration/area, T polarity, U wave, QRS slur, ST slope, TPE, ST morphology, fQRS, PTF-V1, consensus QT/PR
- representative_leads: 12 导联，每个有 `params` 和 `variance`
- global_features: HR、atrial rate、PR/QRS/QT/QTc、P/QRS/T/ST axis、QT dispersion、pacing、PTF-V1
- metadata: input/internal fs、lead_order、n_beats、qrs_detector、representative_group_id、patient_meta
- interpretation: 旧版 morphology flags，如 `pathological_q_leads`, `q_wave_territories`, `r_progression_class`, `st_elevation_leads`, `st_depression_leads`, `lvh_voltage_criteria`, `lvh_class`, `low_voltage_class`, `rvh_suspected`, `tall_t_leads`

当前旧 JSON 不包含：

- top-level `morphology_inputs`
- top-level `rhythm_inputs`
- structured candidate interpretations
- statement `code/category/severity/reason/suppressed_by`
- `dextrocardia_suspected`, `rvh_class`, `copd_pattern`, `qtc_electrolyte_hint`, `posterior_mi_suspected`, `st_rate_related` 等较新的 interpretation extension 字段

重新生成 full features JSON 后，会额外包含 top-level `morphology_inputs`，包括 `record`、`global`、`leads`、`derived_facts`、`statement_evidence` 五块。

## 状态定义

- `OK`: 当前 full features JSON 已提供，可直接或稳定派生。
- `Partial`: 有相关字段，但需要近似、聚合、重新导出、或缺少证据链/精细分量。
- `Missing`: 当前未发现可直接使用的测量字段或结构。

## Feature Checklist

| 模块 | `要点adult_morphology.pdf` 需要的特征 | 当前可用字段 | 状态 | 缺口 / 风险 |
| --- | --- | --- | --- | --- |
| 基础输入 | 采样率、导联顺序、记录长度 | `fs`, `metadata.input_fs`, `metadata.internal_fs`, `metadata.lead_order` | OK | 记录长度不总是显式输出为 `duration_sec`。 |
| 患者信息 | age, sex | `metadata.patient_meta.age`, `metadata.patient_meta.sex` | OK | 年龄/性别有；完整年龄/性别修正表仍需配置。 |
| 临床输入 | clinical diagnosis codes | 未发现稳定字段 | Missing | Mitral Valvular Disease 等 suppression 条件无法可靠使用。 |
| 药物输入 | rx_codes / digitalis | `metadata.patient_meta.meds` | Partial | 字段位置存在，但样本多为 `null`；没有结构化 `rx_codes`。 |
| 质量控制 | lead reliability, wave-specific reliability | `quality.*.reliable_for_p/qrs/t/qt`, `beat_measurement_reliable` | OK | 可以 gate P/QRS/T/ST/QT 测量。 |
| 输出模型 | code/category/severity/reason/suppression | `clinical_interpretation.final_statements/domains/suppressed_statements/conflicts`；另有 reference-only `statement_engine` | Partial | 公开规则已有完整审计；不覆盖完整 Philips code/category/downgrade/replace matrix。 |
| Dextrocardia | P axis rightward | `global_features.p_axis_deg` | OK | 可判断。 |
| Dextrocardia | QRS frontal axis rightward | `global_features.qrs_axis_deg` | OK | 可判断。 |
| Dextrocardia | horizontal QRS direction rightward | 可由 V5/V6 R/S 粗派生 | Partial | 没有明确 `qrs_axis_horizontal_direction`。 |
| Dextrocardia | V5/V6 small QRS | 可由 Q/R/S peak-to-peak 派生 | Partial | 原文未给阈值；当前需要配置。 |
| Dextrocardia | early stop / bypass remaining morphology | 新 model 有 `dextrocardia_suspected`，旧 JSON 无 | Partial | 缺结构化 control-flow evidence。 |
| RAE | limb P duration >= 60 ms | `beat_features[*].p_dur_ms`; current code 可聚合 | OK | representative params 未直接列出 P duration，需从 beat_features 聚合。 |
| RAE | limb P amplitude >= 0.24 mV | `p_amp_mv` | OK | 可用。 |
| RAE | V1 biphasic P | 未发现 `p_biphasic_flag` | Missing | 只有 PTF-V1/幅度近似，不能完整判断 biphasic morphology。 |
| LAE | limb P duration > 110 ms + amp > 0.10 mV | `p_dur_ms`, `p_amp_mv` | OK | 可聚合。 |
| LAE | notched P wave | 未发现 `p_notch_flag` | Missing | 原文指出 notched P 提高权重，当前不能稳定判断。 |
| LAE | V1 terminal negative P duration/amplitude/area | `ptf_v1_mv_ms` | Partial | PTF-V1 有，但 initial/terminal duration/amplitude/area 未完整拆分。 |
| BAE | RAE + LAE 高严重度组合 | `interpretation.p_morphology_class`, `rae_leads`, `lae_*` | Partial | 当前有汇总结果，缺候选语句严重度与组合证据。 |
| QRS axis | LAD/RAD 基础范围 | `global_features.qrs_axis_deg`, `interpretation.qrs_axis_class` | OK | 可支撑基础判断。 |
| QRS axis | age/sex-adjusted normal range | age/sex 可用 | Partial | 原文未给完整表，需要配置。 |
| VCD/BBB | QRS duration 100/110/120 ms 分层 | `global_features.qrs_ms`, per-lead `qrs_ms` | OK | 可用。 |
| Fascicular block | LAFB/LPFB axis ranges | `global_features.qrs_axis_deg`, `interpretation.qrs_axis_class` | Partial | 主要可由 axis 判断，完整 counterclockwise/clockwise 方向需规则化。 |
| RBBB | terminal QRS rightward; I/aVL/V6 terminal negative, V1 positive | R/S/R'/S' amplitude, QRS signed area 新导出可用 | Partial | terminal portion direction 未直接测量；旧 JSON 缺 `qrs_signed_area`。 |
| LBBB | terminal QRS leftward; I/aVL/V6 positive, V1 negative | R/S/R'/S' amplitude, QRS signed area 新导出可用 | Partial | 缺明确 terminal force fact。 |
| R'/S' morphology | R' / S' amplitude | `r_prime_amp_mv`, `s_prime_amp_mv` | OK | 幅度可用。 |
| R'/S' duration | R' / S' duration | `r_prime_duration_ms`, `s_duration_ms`, `s_prime_duration_ms`, `r_duration_ms` | OK | 统一 BBB/RVH 规则会把缺失时限标为 `unavailable`。 |
| RVH | V1 prominent R/R' | `r_amp_mv`, `r_prime_amp_mv` | Partial | R/R' 幅度可用；R' duration 缺。 |
| RVH | V1 positive component > negative component | 可从 Q/R/S 近似 | Partial | 没有直接 `positive_component_mV` / `negative_component_mV`。 |
| RVH | I/V6 prominent Q/S/S' duration + amp | `q_amp_mv`, `s_amp_mv`, `s_prime_amp_mv` | Partial | 幅度有；Q/S/S' duration 缺。 |
| RVH | RV strain pattern | `st_on_mv`, `t_amp_mv`, `t_polarity`, `st_depression_leads` | OK | 可检查 II/aVF/V1-V3 ST depression + inverted T。 |
| RVH | graded consider/probable/definitive | 新 model 有 `rvh_class`；旧 JSON 多为 `rvh_suspected` | Partial | 当前旧 JSON 缺分级；缺完整 evidence。 |
| LVH voltage | R aVL, R I + S III, R V5/V6, S V1/V2 + R V5/V6 | `r_amp_mv`, `s_amp_mv`, `lvh_voltage_criteria` | OK | 可支撑 Table 3-1 电压项。 |
| LVH Cornell | R aVL + S V3, Cornell product | `r_amp_mv`, `s_amp_mv`, `global_features.qrs_ms` | OK | 可用。 |
| LVH age/axis suppression | age < 35, right axis | patient age, `qrs_axis_class` | OK | 数据可用。 |
| LVH add-on flags | LAD, LAE, LV strain, QRS/VAT wide | axis, LAE flags, ST/T, VAT | Partial | 可判断；但没有结构化 flags_set。 |
| LVH Table 3-2/3-3 outputs | LVHV/LVHCNV/LVHCNP 等 code + severity + reason | `lvh_voltage_criteria`, `lvh_class` | Partial | 缺逐 code 输出和 reason。 |
| Low voltage | frontal/precordial QRS peak-to-peak | 可由 Q/R/S 幅度派生；`low_voltage_class` | OK | 建议直接输出 `qrs_peak_to_peak_mV`。 |
| COPD pattern | low voltage + rightward P/QRS + RAE | low voltage、axis、RAE 可用；新 model 有 `copd_pattern` | Partial | 旧 JSON 缺 `copd_pattern`；缺 final statement evidence。 |
| MI Q wave | Q amplitude, Q/R ratio | `q_amp_mv`, `r_amp_mv`, `pathological_q_leads` | OK | 可用。 |
| MI Q duration | inferior/lateral/anterior Q duration 阈值 | `q_onset`, `q_offset`, `q_duration_ms` | OK | 若无法得到可信 Q 时限，统一规则返回 `unavailable`，不会用 Q 幅度代替时限。 |
| MI Q wave area | anterior Q wave area | `q_area_mv_ms`, `initial_qrs_area_mv_ms` | Partial | 已有直接 Q area；仍需按具体准则确认 territory/阈值适用性。 |
| MI initial QRS axis | initial QRS axis leftward | 未发现 direct field | Missing | 原文用于增强 inferior MI likelihood。 |
| MI R progression | transition / poor progression | `r_progression_class`, `r_s_transition_lead` | OK | 可用。 |
| MI ST/T age estimation | T inversion depth, ST deviation magnitude | `t_amp_mv`, `st_*`, territories | Partial | 数据有；缺 infarct age severity/ranking 规则输出。 |
| MI territories | inferior/lateral/anterior/anterolateral/posterior | `q_wave_territories`, ST territories, R progression | Partial | 现有 territory 粗略；posterior/right-sided leads 不完整。 |
| Culprit artery | RCA/LCx/LAD/LM-MVD inference | ST II/III/V2/V3/aVR 可用 | Partial | 规则可派生；缺 V4R/V7-V9 和 structured culprit output。 |
| Extended leads | V4R, V7, V8, V9 | `STANDARD_12_LEADS` only | Missing | 后壁/右室梗死敏感性不足。 |
| ST Map | Cabrera angle, ST deviation mm, shaded map | ST deviation 有 | Partial | 缺 Cabrera mapping、polarity-adjusted value、绘图结构。 |
| ST depression | J point, midpoint, J+80 ms | `st_on_mv`, `st_mid_mv`, `st_80ms_mv` | OK | 三个关键点可用。 |
| ST depression | ST end / T beginning | `t.onset` 可作为 T beginning | Partial | 没有显式 `st_end_mv`，需从 raw signal 或 T onset 派生。 |
| ST depression | slope degrees | `st_slope_mv_per_ms` | Partial | 单位不是 degrees。 |
| ST depression | shape straight/concave up/down | `st_morphology` = upsloping/horizontal/downsloping | Partial | 分类体系不匹配原文 shape。 |
| Rate-related ST depression | HR > 190 - age | `heart_rate_bpm`, age, `st_depression_leads`; new model 有 `st_rate_related` | Partial | 旧 JSON 缺 `st_rate_related`，缺 statement。 |
| T wave abnormality | T amplitude | `t_amp_mv` | OK | 可用。 |
| T wave abnormality | relative T/QRS amplitude | `t_amp_mv` + QRS peak-to-peak 可派生 | OK | 建议直接输出 ratio。 |
| T wave abnormality | polarity positive/negative/flat/biphasic | `t_polarity` | Partial | 正负有；flat/biphasic 未明确结构化。 |
| T axis / QRS-T angle | frontal T axis, QRS-T angle | `global_features.t_axis_deg`, `interpretation.qrs_t_angle_deg` | OK | 可用。 |
| ST-T repolarization combo | ST depression + T abnormality combined severity | ST/T fields 有 | Partial | 缺 candidate category 与 severity combination。 |
| ST elevation | J point and J+80 positive values | `st_on_mv`, `st_80ms_mv` | OK | 可用。 |
| ST elevation | injury/pericarditis/early repol | ST territories 有；new model 有 pericarditis/early repol flags | Partial | 旧 JSON 缺新 flags；缺完整 suppression/confounder list。 |
| Tall T | positive T > 1.2 mV, or >0.5 mV and >0.5 QRS PP | `t_amp_mv`, `t_polarity`, QRS peak-to-peak, `tall_t_leads` | OK | 数据足够；规则必须限定 positive T。 |
| QT short/long | QTc thresholds 310/340/465/485/520 | `qtc_bazett_ms`, `qtc_fridericia_ms`, `qtc_class` | OK | 可用。 |
| Electrolyte hints | hypercalcemia/hypocalcemia/hypokalemia | QTc + ST depression + positive T | Partial | 数据有；旧 JSON 缺 `qtc_electrolyte_hint`。 |
| Digitalis effect | Rx digitalis + ST/T/QT pattern | `patient_meta.meds` | Partial | 药物输入不稳定，缺 `rx_codes`。 |
| Suppression rules | RVH/LVH/BBB/infarct/drug/electrolyte suppress ST/T/QTc | `clinical_interpretation.suppressed_statements`, per-rule `suppressed_by` | Partial | 已覆盖导联反接、BBB/IVCD 与选定肥大/缺血规则；未覆盖完整 Philips suppression matrix。 |
| Rule audit | all rules reason output | `clinical_interpretation.domains.*` 包含 status/evidence/thresholds/source/missing_inputs | OK（当前公开规则集） | 仅表示实现规则可审计，不表示已做临床验证或专有算法等价验证。 |

## 实用判断

当前 full features JSON 已经适合做 **MVP adult morphology rule engine**：

- dextrocardia 粗判断
- LAD/RAD/QRS axis
- QRS duration 分层、基础 IVCD/RBBB/LBBB
- low voltage
- LVH voltage / Cornell / Sokolow-Lyon
- RVH 粗筛
- pathological Q wave territory 粗筛
- ST elevation/depression territory 粗筛
- T wave abnormality 粗筛
- tall T
- QTc short/long 与基础 electrolyte hint

它暂时不适合完整复刻 Philips Adult Morphology：

- RAE/LAE/BAE 的 P notched/biphasic/terminal component 精细分级
- BBB/RVH 的 terminal force 和 component duration 精细判定
- MI 新旧程度与 culprit artery 的完整证据链
- ST Map 的 Cabrera spatial display
- V4R/V7-V9 扩展导联逻辑
- 完整 LVH Table 3-2/3-3 statement code matrix
- 全局 suppression/downgrade/replace/ranking 规则
- 每条输出语句的 reason、severity、suppressed_by、confidence_rank

## 已新增的最小目标 Schema

full features JSON 已像 rhythm 一样新增 top-level `morphology_inputs`，不会替换原始测量，而是把现有字段整理成 rule-engine-friendly facts。

### `morphology_inputs.record`

- `fs`
- `duration_sec`
- `age_years`
- `age_valid`
- `sex`
- `algorithm_age_group`
- `algorithm_age_reason`
- `age_defaulted_to_adult_algorithm`
- `clinical_dx_codes`
- `rx_codes`
- `lead_order`
- `available_extended_leads`
- `record_quality`
- `rejected_functions`

### `morphology_inputs.global`

- `heart_rate_bpm`
- `qrs_duration_ms`
- `qt_ms`
- `qtc_bazett_ms`
- `qtc_fridericia_ms`
- `p_axis_frontal_deg`
- `qrs_axis_frontal_deg`
- `qrs_axis_horizontal_direction`
- `t_axis_frontal_deg`
- `qrs_t_angle_deg`

### `morphology_inputs.leads`

每个 lead 输出：

- `p.duration_ms`
- `p.amplitude_mV`
- `p.area`
- `p.is_notched`
- `p.is_biphasic`
- `p.initial_duration_ms`
- `p.initial_amplitude_mV`
- `p.terminal_duration_ms`
- `p.terminal_amplitude_mV`
- `p.terminal_area_ashman`
- `qrs.q_duration_ms`
- `qrs.q_amplitude_mV`
- `qrs.r_amplitude_mV`
- `qrs.r_prime_duration_ms`
- `qrs.r_prime_amplitude_mV`
- `qrs.s_duration_ms`
- `qrs.s_amplitude_mV`
- `qrs.s_prime_duration_ms`
- `qrs.s_prime_amplitude_mV`
- `qrs.peak_to_peak_mV`
- `qrs.positive_component_mV`
- `qrs.negative_component_mV`
- `qrs.area_sign`
- `qrs.terminal_direction`
- `st.j_point_mV`
- `st.midpoint_mV`
- `st.j80_mV`
- `st.end_mV`
- `st.slope_deg`
- `st.shape`
- `t.amplitude_mV`
- `t.polarity`
- `t.relative_to_qrs`
- `quality`

### `morphology_inputs.derived_facts`

- `adult`
- `pediatric`
- `dextrocardia_criteria`
- `rae_criteria`
- `lae_criteria`
- `bae_criteria`
- `axis_flags`
- `vcd_bbb_criteria`
- `rvh_criteria`
- `lvh_voltage_criteria`
- `lvh_additional_flags`
- `low_voltage_criteria`
- `copd_pattern_criteria`
- `mi_territory_criteria`
- `culprit_artery_criteria`
- `st_depression_criteria`
- `t_wave_criteria`
- `st_elevation_criteria`
- `tall_t_criteria`
- `qt_electrolyte_criteria`

### `morphology_inputs.statement_evidence`

- `candidates[]`
- `final_statements[]`
- `statement.code`
- `statement.category`
- `statement.severity`
- `statement.reason`
- `statement.evidence`
- `statement.flags_set`
- `statement.suppression_tags`
- `statement.suppressed_by`
- `statement.confidence_rank`
- `stop_further_interpretation`

## 优先级建议

- `P0`: 已完成。新增 `morphology_inputs` schema，并补 adult / pediatric age route 与阈值上下文，把当前已能稳定派生的 facts 输出出来，不新增复杂信号算法。
- `P1`: 补 P notched/biphasic、P terminal component、Q/R'/S'/S duration、QRS terminal direction。
- `P2`: 扩展现有 statement/suppression engine 到尚未覆盖的公开规则，并保持 `unavailable` 与参考算法隔离。
- `P3`: 做 ST Map、V4R/V7-V9、MI culprit artery 的完整增强逻辑。
