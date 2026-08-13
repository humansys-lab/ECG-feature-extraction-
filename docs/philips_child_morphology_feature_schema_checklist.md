<!-- i18n-nav -->
[中文](philips_child_morphology_feature_schema_checklist.md) | [English](philips_child_morphology_feature_schema_checklist.en.md) | [日本語](philips_child_morphology_feature_schema_checklist.ja.md)
<!-- /i18n-nav -->

# Philips Pediatric Morphology Feature Schema Checklist

## 结论先行

> 2026-07-12 状态更新：统一权威层已支持成人/儿童路由，但在缺少经过版本化、
> 可公开追溯的儿童 PR 与心室肥大百分位表时，`pediatric_pr_public_reference_table`、
> `pediatric_lvh_percentile_table` 和 `pediatric_rvh_percentile_table` 会明确输出
> `unavailable`。现有 DXL-inspired 儿童阈值仍保留在 reference-only payload 中，
> 不会进入权威最终结论，也不会据此输出“正常”。

`philips_child_morphology.pdf` 是 Philips DXL Chapter 4 Pediatric Morphology Analysis，适用于 **出生到未满 16 岁** 的 ECG。和成人 morphology 最大的区别是：儿童 ECG 的正常范围高度依赖年龄，部分规则还依赖性别。

当前测量结果和代码已经能支撑一版 **pediatric morphology MVP**：

- patient age / sex
- P/QRS/T/ST axis
- per-lead P/QRS/T/ST 振幅和边界
- P duration、P amplitude、PTF-V1
- Q/R/S/R'/S' amplitude
- QRS duration、QRS peak-to-peak、VAT、notch/slur/fQRS
- ST J-point/mid/80ms、T amplitude/polarity
- QT/QTc
- interpretation 层已有 `is_pediatric`、age-adjusted QRS axis、age-adjusted QRS width、pediatric QTc、LSH、BVH、pericarditis、early repolarization 等字段

本轮修改后，重新生成 full features JSON 会在 `morphology_inputs` 中明确区分 adult / pediatric morphology context：

- `record.algorithm_age_group` 会输出 `adult` 或 `pediatric`；出生到未满 16 岁走 pediatric。
- 年龄缺失或无效时按 Philips 文档默认走 adult，并通过 `record.age_defaulted_to_adult_algorithm` 标记。
- `derived_facts.adult` 输出成人阈值上下文。
- `derived_facts.pediatric` 输出 age bucket、pediatric axis / QRS / QTc thresholds、儿童 dextrocardia / RAE / BBB / hypertrophy / ST-T / QT 规则所需阈值上下文。

但它还不能完整满足 Philips-style pediatric morphology engine：

- 缺少 Davignon Appendix A 的完整年龄分层正常值表，尤其是 RVH/LVH voltage 98th percentile thresholds。
- RBBB/IRBBB/LBBB 需要 R'/terminal 40ms/QRS component duration，当前只部分近似。
- RVH 需要 synthesized horizontal-plane vector / terminal angle 辅助区分 mild RVH vs IRBBB，当前没有。
- P-wave biphasic/notched/initial-terminal components 与成人一样仍不完整。
- 输出层已有统一 statement/status/evidence/suppression contract；儿童百分位依赖规则因公开表未版本化而故意不可判定，完整 Philips code/downgrade matrix 仍未实现。
- congenital heart defects 只给出组合思路，当前没有结构化先天性心脏病规则。

所以判断是：**当前 feature JSON 可支撑成人/儿童分流和部分儿童测量，但公开准则的权威层会保守拒绝缺表规则；完整 Philips Chapter 4 仍需合法可追溯的参考表、terminal morphology 与专有 statement matrix，不能由现有参考启发式冒充。**

## 文档范围

本清单对照本地 PDF：

- `/home/chtmedgemma/projects/ecg_gemma/philips_child_morphology.pdf`

文档覆盖：

- pediatric routing and age handling
- dextrocardia
- RAE / LAE / BAE
- age-adjusted QRS axis
- age-adjusted VCD / RBBB / IRBBB / LBBB / LAFB
- RVH / LSH / LVH / BVH
- low voltage / COPD pattern
- Q wave abnormality / MI
- ST depression / T wave abnormality / repolarization abnormality
- ST elevation / pericarditis / early repolarization
- tall T
- pediatric QTc / electrolyte disturbance
- congenital heart defects

## 当前代码与 JSON 状态

当前代码层已有不少 pediatric extension：

- `PEDS_MAX_AGE_YEARS = 16.0`
- pediatric dextrocardia thresholds
- pediatric RAE P amplitude threshold `0.20 mV`
- age-dependent QRS duration normal limits
- age-dependent QRS axis tables
- pediatric RBBB R' amplitude threshold
- pediatric QTc age/sex thresholds
- LSH and BVH helpers
- pericarditis age gate 5-15 years
- early repolarization age gate 13-15 years
- `ECGInterpretation.is_pediatric`
- `ECGInterpretation.lsh_suspected`
- `ECGInterpretation.bvh_suspected`
- `ECGInterpretation.pericarditis_suspected`
- `ECGInterpretation.early_repolarization_suspected`

但当前已存在的样本 JSON（如 `JS00029_features.json`、`JS00024_features.json`、`JS00007_features.json`）可能仍是旧导出，且患者年龄多为成人，所以：

- 旧 JSON 可能不包含最新 `morphology_inputs.derived_facts.pediatric`
- 旧 JSON 可能不包含最新 `record.algorithm_age_group`
- 成人样本不适合验证 pediatric routing
- 重新生成 full features JSON 后才会包含最新 schema

## 状态定义

- `OK`: 当前 full features JSON / model 已提供，可直接或稳定派生。
- `Partial`: 有相关字段或初版规则，但缺年龄阈值、精细测量、规则证据或新导出。
- `Missing`: 当前未发现可直接使用的测量字段或结构。

## Feature Checklist

| 模块 | Philips child morphology 需要的特征 | 当前可用字段 / 代码 | 状态 | 缺口 / 风险 |
| --- | --- | --- | --- | --- |
| Pediatric routing | age 从出生到未满 16 岁走 pediatric algorithm | `metadata.patient_meta.age`; `is_pediatric` | OK | 若 age 缺失或无效，文档要求按 adult default 并打印假设声明；当前缺 statement evidence。 |
| Patient sex | 13 岁以上 QTc 阈值按男/女区分 | `metadata.patient_meta.sex` | OK | sex 缺失时需要 unknown fallback。 |
| Age bins | 0-23h, 1-3d, 4-6d, 7-29d, months/years 分段 | 代码有 QRS axis/QRS duration lookup tables | Partial | RVH/LVH voltage 仍缺完整 Appendix A 正常值表。 |
| Dextrocardia | P axis 90-180 deg | `global_features.p_axis_deg`; `_peds_dextrocardia()` | OK | 可用。 |
| Dextrocardia | Lead I 或 V6 negative P | `p_amp_mv` | OK | 只看 P amplitude 符号，缺更精细 P morphology。 |
| Dextrocardia | I 和 V6 large S > 0.6 mV | `s_amp_mv`; `PEDS_DEXTRO_S_MV` | OK | 可用。 |
| Dextrocardia | P amplitude III > II | `p_amp_mv` | OK | 可用。 |
| Dextrocardia | bypass remainder | `dextrocardia_suspected` | Partial | 规则有；缺 `statement_evidence.stop_further_interpretation` 的 pediatric reason。 |
| RAE | P duration >= 60 ms | `p_dur_ms` 或 P bounds 聚合 | OK | 可用。 |
| RAE | P amplitude >= 0.20 mV | `p_amp_mv`; `PEDS_RAE_P_AMP_MV` | OK | 当前通用 P morphology 仍主要用 adult 0.24 mV，需确认 pediatric path 是否完全应用 0.20 mV。 |
| RAE | V1 biphasic P supports probable RAE | 缺 `p_biphasic_flag` | Missing | 和成人同一缺口。 |
| LAE | limb P duration > 110 ms + amplitude > 0.10 mV | `p_dur_ms`, `p_amp_mv` | OK | 可用。 |
| LAE | notched P adds significance | 缺 `p_notch_flag` | Missing | 不能完整还原 LAE score。 |
| LAE | V1 negative P terminal duration/amplitude/area | `ptf_v1_mv_ms` | Partial | PTF-V1 有；terminal duration/amplitude/area 未拆分。 |
| BAE | RAE + LAE high severity -> biatrial hypertrophy | `p_morphology_class`, `rae_leads`, `lae_*` | Partial | 缺 statement severity 和 high-severity candidate evidence。 |
| QRS axis | age-adjusted LAD/RAD | `qrs_axis_deg`; `_peds_classify_qrs_axis()` | OK | 代码已有 Table 4-1 到 4-4 lookup。 |
| QRS axis | borderline LAD/RAD within 15 deg zone | `_peds_classify_qrs_axis()` | OK | 可输出 `borderline_LAD` / `borderline_RAD`。 |
| VCD | age-adjusted QRS normal limit | `qrs_ms`; `_PEDS_QRS_NORMAL_MS` | OK | 代码已有 Table 4-5。 |
| VCD | >110% normal -> borderline IVCD | `_peds_classify_qrs_width()` | OK | 可用。 |
| VCD | >120% normal -> nonspecific IVCD / BBB range | `_peds_classify_qrs_width()` | OK | 可用。 |
| RBBB | VCD for age + RSR' or pure R in V1 | `r_prime_amp_mv`, `r_amp_mv`, Q/S amplitude | Partial | RSR'/pure R 可以粗判；缺 R' duration 和 no-negative-component fact。 |
| RBBB | R' duration >= 20 ms and amplitude >= 0.15 mV | `r_prime_amp_mv`; missing duration | Partial | 幅度有；R' duration 未测量。 |
| IRBBB | RBBB-like morphology but QRS <120% normal | `_peds_classify_qrs_width()` | Partial | 缺 synthesized vector differentiation from RVH。 |
| LBBB | prolonged QRS for age | `qrs_ms` | OK | 可用。 |
| LBBB | terminal 40ms QRS axis -90 to 90 clockwise | 未发现 terminal 40ms axis | Missing | 关键底层测量缺。 |
| LBBB | short/absent S in I/aVL/V5/V6 and small/absent R in V1-V3 | Q/R/S amplitude 可粗判 | Partial | S duration 缺；small/absent 阈值需配置。 |
| LAFB | absent LBBB + mean QRS axis -60 to -90 | `qrs_axis_deg`, `bundle_branch_block` | OK | 代码已有 pediatric LAFB range。 |
| RVH | bypass if RBBB | `bundle_branch_block` | OK | 可用。 |
| RVH voltage | age-dependent RVH voltage criteria, 6 age groups, 24 conditions | 部分 Q/R/S/R' 幅度有 | Partial | 缺完整 Appendix A 98th percentile thresholds。 |
| RVH | R/R' in V1/V2, S in V6, R/S ratios, QR pattern V1 | `r_amp_mv`, `r_prime_amp_mv`, `s_amp_mv`, `q_amp_mv` | Partial | 幅度有；R' duration、QR pattern object、age-specific threshold table 缺。 |
| RVH | upright T in V1 for age >48h and <9y, absent inverted T in V5/V6 | `t_amp_mv`, `t_polarity`, age | OK | 可派生。 |
| RVH | RAD / borderline RAD support | pediatric `qrs_axis_class` | OK | 可用。 |
| RVH vs IRBBB | synthesized horizontal plane terminal angle | 未发现直接字段 | Missing | 文档明确用于 mild RVH vs IRBBB。 |
| LSH | prominent R in V1 + Q in V5/V6 | `r_amp_mv`, `q_amp_mv`; `_peds_lsh()` | Partial | 当前用固定 proxy 阈值，不是完整 98th percentile table。 |
| LVH | bypass if RBBB or LBBB | `bundle_branch_block` | OK | 代码已有 pediatric bypass。 |
| LVH voltage | R in I/II/aVL/aVF/V5/V6; S in V1/V2; RV6+SV1; prominent Q | Q/R/S amplitude | Partial | 数据有；缺完整 age-dependent 98th percentile thresholds。 |
| LVH support | LAD and LAE support LVH | `qrs_axis_class`, LAE fields | OK | 可用。 |
| LVH strain | I/aVL/V4/V5/V6 ST/T repolarization pattern | `st_mid_mv`, `st_on_mv`, `t_amp_mv`, `t_polarity` | Partial | 数据有；规则需要 pediatric-specific pattern evidence。 |
| BVH | RVH + LVH high severity | `rvh_class`, `lvh_class`, `bvh_suspected` | Partial | code has summary flag;缺 candidate severity evidence。 |
| BVH | R V1 >1.0 mV with LVH | `r_amp_mv` | OK | 可用。 |
| BVH | RVH + Q V6 >10ms and >0.07mV + R V6 >1.0mV | Q duration approx, Q/R amplitude | Partial | Q duration 是近似，缺独立 Q onset/offset。 |
| BVH | R+S >6.0mV in >=2 of V2/V3/V4 | Q/R/S amplitude | OK | 可派生。 |
| BVH suppression | BVH suppresses individual RVH/LVH | `bvh_suspected` | Partial | 代码做了部分 suppress；缺 statement-level suppressed_by。 |
| Low voltage | frontal/precordial QRS peak-to-peak | 可由 Q/R/S 派生；`low_voltage_class` | OK | 和成人相同。 |
| COPD pattern | low voltage + rightward P/QRS + RAE | low voltage、axis、RAE | Partial | 可派生；缺 pediatric statement evidence。 |
| Q wave abnormality | large Q waves in 2 leads of a group | `q_amp_mv`, `pathological_q_leads` | Partial | group count 可派生；缺 pediatric-specific statement。 |
| Q wave infarct | Q > 1/5 R amplitude suggests infarct | `q_amp_mv`, `r_amp_mv` | OK | 可用。 |
| ST depression | anterior/lateral/inferior groups | `st_depression_leads`, territories | OK | 可用。 |
| ST depression | >0.20 mV in one group -> nonspecific ST depression | `st_on_mv` 等 | Partial | 数据有；当前 general ST threshold 更低，需 pediatric-specific threshold 输出。 |
| ST depression | tachycardia -> rate-related statement | `heart_rate_bpm`, age | Partial | 文档未给 pediatric tachycardia table；当前 adult-like `190-age` 不足。 |
| ST depression suppression | hypertrophy or VCD suppresses | RVH/LSH/LVH/BVH/VCD fields | Partial | 缺 statement-level suppression output。 |
| T abnormality | inverted T in anterior/lateral/anterolateral/inferior groups | `t_amp_mv`, `t_polarity` | OK | 可派生。 |
| T abnormality | inverted T amplitude >1.0mV in >=2 leads in group -> tall T abnormality | `t_amp_mv`, lead groups | OK | 可派生。 |
| T abnormality secondary to RVH/LVH | RVH/LVH flags + inverted T groups | fields available | Partial | 缺 final statement evidence。 |
| Repolarization abnormality | ST depression + inverted T combination | ST/T facts | Partial | 数据有；缺 combined severity/candidate engine。 |
| ST elevation | all leads tested, >0.15mV suggests probable normal variation | `st_elevation_leads`, `st_on_mv` | Partial | 数据有；需 pediatric threshold and statement. |
| ST elevation suppression | hypertrophy and VCD suppress | RVH/LSH/LVH/BVH/VCD fields | Partial | 缺 statement-level suppression output。 |
| Pericarditis | ST elevation in anterior/lateral/inferior groups, age 5-15 | `pericarditis_suspected` | OK | 代码已有 summary flag。 |
| Early repolarization | nonspecific ST elevation, no T inversion, age 13-15 | `early_repolarization_suspected` | OK | 代码已有 summary flag。 |
| Tall T | T >1.20mV or >0.50mV and >0.5 QRS peak-to-peak | `t_amp_mv`, QRS peak-to-peak | OK | 可用。 |
| QT short | QTc <340ms borderline short | `qtc_bazett_ms`, `qtc_class` | OK | 代码已有 pediatric classifier。 |
| QT borderline prolonged | >450 <5y; >454 age 5-12; >458 boys >=13; >465 girls >=13 | age/sex/QTc | OK | 代码已有。 |
| QT prolonged | borderline threshold +20ms | age/sex/QTc | OK | 代码已有。 |
| QT suppression | RVH/LSH/LVH/BVH/VCD -> prolonged QT secondary to wide QRS | fields available | Partial | 当前可能 normalizes/suppresses class，缺 printed secondary statement evidence。 |
| Electrolyte | QTc <310 hypercalcemia; >520 hypocalcemia/hypokalemia with ST depression + positive T | QTc, ST, T | OK | 代码已有 hint。 |
| Congenital defects | combinations of atrial/ventricular hypertrophy, VCD, axis, QRS morphology | summary facts available | Partial | 文档没有 full rules；当前也没有 CHD candidate engine。 |

## 和当前 `morphology_inputs` 的关系

上一轮已新增 top-level `morphology_inputs`，其中 `record/global/leads/derived_facts/statement_evidence` 对儿童规则也有帮助：

- `record.age_years` 和 `record.sex` 可做 pediatric routing。
- `global.qrs_duration_ms`, `global.qrs_axis_frontal_deg`, `global.qtc_bazett_ms` 可支撑 age-adjusted QRS/QTc。
- `leads.*.p/qrs/st/t` 可支撑 dextrocardia、RAE/LAE、VCD/BBB、hypertrophy、ST/T/QT。
- `derived_facts` 已有 adult-style axis/VCD/RVH/LVH/ST/T/QT blocks。

本轮已补 `derived_facts.pediatric` 子块，避免把儿童阈值混在 adult facts 中。重新生成后的结构类似：

```json
"pediatric": {
  "is_pediatric": true,
  "age_bucket": "5-7_years",
  "route_selected": true,
  "axis_limits": {
    "lad_threshold_deg": -10,
    "borderline_lad_upper_deg": 1,
    "rad_threshold_360_deg": 201,
    "borderline_rad_threshold_360_deg": 160
  },
  "qrs_duration_limits": {
    "normal_limit_ms": 88,
    "borderline_ivcd_ms": 96.8,
    "nonspecific_ivcd_ms": 105.6
  },
  "qtc_limits": {
    "short_ms": 340,
    "borderline_prolonged_ms": 454,
    "prolonged_ms": 474,
    "severe_ms": 520
  },
  "hypertrophy_thresholds": {
    "rvh_lvh_age_percentile_tables": {
      "available": false,
      "reason": "appendix_a_percentile_tables_not_digitized"
    },
    "lsh_r_v1_prominent_mV": 1.0,
    "bvh_rs_sum_v2_v3_v4_mV": 6.0
  },
  "repolarization_thresholds": {
    "st_depression_threshold_mV": 0.20,
    "st_elevation_normal_variation_threshold_mV": 0.15,
    "pericarditis_age_gate": "5-15",
    "early_repolarization_age_gate": "13-15"
  }
}
```

## 推荐优先级

- `P0`: 已完成。在 `morphology_inputs.derived_facts` 下新增 `adult` / `pediatric` 子块，暴露 algorithm age route、age bucket、pediatric axis/QRS/QTc thresholds 和已有 pediatric summary flags。
- `P1`: 补完整 pediatric RVH/LVH voltage threshold table，也就是文档引用的 Appendix A normal measurement values。
- `P2`: 补 R' duration、S duration、terminal 40ms QRS axis、horizontal terminal vector，提升 RBBB/IRBBB/LBBB/RVH 区分能力。
- `P3`: 在取得可公开追溯且版本化的儿科参考表后，扩展现有统一 statement/suppression engine；在此之前保持 `unavailable`。
- `P4`: 做 congenital heart defects rule layer，需要先整理具体先心病组合规则，否则只能输出“规则未实现”占位。
