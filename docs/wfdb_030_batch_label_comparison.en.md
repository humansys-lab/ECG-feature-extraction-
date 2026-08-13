<!-- i18n-nav -->
[中文](wfdb_030_batch_label_comparison.md) | [English](wfdb_030_batch_label_comparison.en.md) | [日本語](wfdb_030_batch_label_comparison.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="wfdbrecords03030-批次-逐条诊断对照最新一轮重新生成"></a>
# WFDBRecords/03/030 batch - item-by-item diagnostic control (the latest round of regeneration)

- Data: `data/WFDBRecords_03_030_ecgfeat_out/`, regenerated on 2026-07-12 17:4x (manifest: processed=100, failed=0)
- The SNOMED code table has been completed: added Left/Right axis deviation, Low QRS voltages,
  Counterclockwise cardiac rotation, Nonspecific IVCD, Ventricular escape beat
- Comparison method: Dx tag (SNOMED) of each record → corresponding unified layer rule → match/miss/unavailable/N/A
- verdict meaning: `match`=The rule hit and the label is supported; `miss`=The rule ran but the label was not supported;
  `unavailable`=The rule is waived due to lack of measurement; `N/A(规则恒不可用)`=The design has not yet been implemented (such as AFL);
  `N/A(无对应规则)`=This label has no corresponding rules (such as normal labels such as "sinus rhythm");
  `N/A(未映射)`=Code that appears in the data set but has not been included in the local SNOMED code table

<a id="1-逐条记录"></a>
## 1. Record one by one

| Record | Comprehensive conclusion overall_status | Tag comparison |
|---|---|---|
| JS02082 | abnormal_with_limited_coverage | Atrial flutter=match |
| JS02083 | abnormal_with_limited_coverage | Sinus bradycardia=match |
| JS02084 | abnormal_with_limited_coverage | Sinus bradycardia=match; Left axis deviation=N/A (no corresponding rule) |
| JS02085 | incomplete | Sinus rhythm=N/A (no corresponding rule) |
| JS02086 | abnormal_with_limited_coverage | Sinus bradycardia=match; Left ventricular hypertrophy=match |
| JS02087 | abnormal_with_limited_coverage | Sinus bradycardia=match; First-degree AV block=match |
| JS02088 | borderline | Sinus bradycardia=match |
| JS02089 | technically_limited | Sinus rhythm=N/A (no corresponding rule) |
| JS02090 | abnormal_with_limited_coverage | Sinus bradycardia=match; Left ventricular hypertrophy=match; Nonspecific ST-T abnormality=match |
| JS02091 | abnormal_with_limited_coverage | Atrial fibrillation=match; Q-wave abnormality=match; Nonspecific intraventricular conduction delay=miss; Left bundle branch block=miss; T-wave abnormality=match |
| JS02092 | incomplete | Sinus bradycardia=match |
| JS02093 | incomplete | Sinus bradycardia=match |
| JS02094 | incomplete | Sinus bradycardia=match |
| JS02095 | abnormal_with_limited_coverage | Sinus tachycardia=match; Counterclockwise cardiac rotation=N/A (no corresponding rule) |
| JS02096 | normal_with_core_coverage | Sinus bradycardia=match; Nonspecific ST-T abnormality=miss |
| JS02097 | borderline | Sinus bradycardia=match |
| JS02098 | abnormal_with_limited_coverage | Atrial fibrillation=match; Nonspecific intraventricular conduction delay=unavailable; Nonspecific ST-T abnormality=miss |
| JS02099 | abnormal | Sinus rhythm=N/A (no corresponding rule) |
| JS02100 | incomplete | Sinus rhythm=N/A (no corresponding rule) |
| JS02101 | normal_with_core_coverage | Sinus rhythm=N/A (no corresponding rule); Left axis deviation=N/A (no corresponding rule) |
| JS02102 | incomplete | Sinus rhythm=N/A (no corresponding rule); Premature atrial contraction=match |
| JS02103 | abnormal_with_limited_coverage | Atrial fibrillation=unavailable; ST depression=N/A (no corresponding rule); Nonspecific ST-T abnormality=match; T-wave abnormality=match |
| JS02104 | abnormal_with_limited_coverage | Atrial fibrillation=match; Right axis deviation=N/A (no corresponding rule); Nonspecific ST-T abnormality=match |
| JS02105 | abnormal_with_limited_coverage | Sinus bradycardia=match; Premature atrial contraction=match; Left ventricular hypertrophy=match; T-wave abnormality=match |
| JS02106 | abnormal_with_limited_coverage | Atrial fibrillation=match |
| JS02107 | abnormal_with_limited_coverage | Atrial fibrillation=unavailable; Right axis deviation=N/A (no corresponding rule); Ventricular premature beats=match |
| JS02108 | technically_limited | Sinus rhythm=N/A (no corresponding rule) |
| JS02109 | abnormal_with_limited_coverage | Atrial flutter=N/A (the rule is always unavailable); Atrioventricular block=miss |
| JS02110 | incomplete | Sinus bradycardia=match |
| JS02111 | technically_limited | Sinus rhythm=N/A (no corresponding rule); Nonspecific ST-T abnormality=miss |
| JS02112 | abnormal_with_limited_coverage | Sinus tachycardia=match; Right axis deviation=N/A(无对应规则) |
| JS02113 | technically_limited_with_findings | Sinus tachycardia=unavailable |
| JS02114 | incomplete | Sinus bradycardia=match; T-wave abnormality=miss |
| JS02115 | abnormal_with_limited_coverage | Sinus bradycardia=match; Left bundle branch block=miss |
| JS02116 | technically_limited | Sinus bradycardia=match |
| JS02117 | borderline | Sinus rhythm=N/A(无对应规则) |
| JS02118 | abnormal_with_limited_coverage | Atrial fibrillation=match; ST depression=N/A(无对应规则); Nonspecific ST-T abnormality=match; T-wave abnormality=match; Ventricular escape beat=N/A(无对应规则) |
| JS02119 | technically_limited_with_findings | Sinus tachycardia=match; T-wave abnormality=miss |
| JS02120 | incomplete | Sinus tachycardia=unavailable |
| JS02121 | abnormal_with_limited_coverage | Sinus tachycardia=match; Premature atrial contraction=match |
| JS02122 | incomplete | Sinus tachycardia=match |
| JS02124 | technically_limited_with_findings | Sinus bradycardia=match; Left ventricular hypertrophy=match |
| JS02125 | incomplete | Sinus bradycardia=match; T-wave abnormality=miss |
| JS02126 | abnormal_with_limited_coverage | Sinus rhythm=N/A(无对应规则); Nonspecific ST-T abnormality=match |
| JS02127 | abnormal | Sinus bradycardia=match |
| JS02128 | technically_limited_with_findings | Sinus rhythm=N/A(无对应规则) |
| JS02129 | borderline | Sinus bradycardia=match |
| JS02130 | abnormal_with_limited_coverage | Sinus tachycardia=match; Nonspecific ST-T abnormality=match |
| JS02131 | incomplete | Sinus rhythm=N/A(无对应规则) |
| JS02132 | abnormal_with_limited_coverage | Atrial fibrillation=match; Left ventricular hypertrophy=match; Nonspecific ST-T abnormality=miss |
| JS02133 | abnormal_with_limited_coverage | Sinus bradycardia=match; T-wave abnormality=miss |
| JS02134 | abnormal_with_limited_coverage | Sinus tachycardia=match; Left axis deviation=N/A(无对应规则); Left ventricular hypertrophy=miss |
| JS02135 | normal_with_core_coverage | Sinus rhythm=N/A(无对应规则) |
| JS02136 | technically_limited_with_findings | Atrial fibrillation=unavailable; T-wave abnormality=miss |
| JS02137 | abnormal | Atrial fibrillation=miss; Left axis deviation=N/A(无对应规则); T-wave abnormality=miss |
| JS02138 | abnormal_with_limited_coverage | Atrial flutter=N/A(规则恒不可用); Left bundle branch block=miss; Nonspecific ST-T abnormality=match |
| JS02139 | abnormal | Sinus bradycardia=match |
| JS02140 | abnormal_with_limited_coverage | Sinus bradycardia=match |
| JS02141 | incomplete | Sinus bradycardia=match |
| JS02143 | borderline | Sinus bradycardia=match; Q-wave abnormality=unavailable; T-wave inversion=miss |
| JS02144 | abnormal_with_limited_coverage | Sinus tachycardia=match |
| JS02145 | incomplete | Sinus rhythm=N/A(无对应规则) |
| JS02146 | technically_limited | Sinus rhythm=N/A(无对应规则) |
| JS02147 | abnormal | Sinus rhythm=N/A(无对应规则); Low QRS voltages=miss |
| JS02148 | abnormal_with_limited_coverage | Sinus tachycardia=match |
| JS02149 | abnormal_with_limited_coverage | Atrial fibrillation=match; Left ventricular hypertrophy=match; Nonspecific ST-T abnormality=match |
| JS02150 | borderline | Sinus bradycardia=match |
| JS02151 | technically_limited_with_findings | Sinus bradycardia=match; Nonspecific ST-T abnormality=miss |
| JS02152 | incomplete | Sinus rhythm=N/A(无对应规则) |
| JS02153 | abnormal_with_limited_coverage | Sinus tachycardia=match |
| JS02154 | normal_with_core_coverage | Sinus rhythm=N/A(无对应规则) |
| JS02155 | normal_with_core_coverage | Sinus bradycardia=match |
| JS02156 | technically_limited | Sinus bradycardia=match |
| JS02157 | abnormal | Sinus tachycardia=match |
| JS02158 | incomplete | Sinus tachycardia=match |
| JS02159 | technically_limited_with_findings | Sinus bradycardia=match; Left ventricular hypertrophy=miss; T-wave abnormality=match |
| JS02160 | abnormal_with_limited_coverage | Atrial fibrillation=match |
| JS02161 | abnormal_with_limited_coverage | Atrial fibrillation=match |
| JS02163 | abnormal_with_limited_coverage | Sinus tachycardia=match |
| JS02164 | abnormal_with_limited_coverage | Atrial fibrillation=match; T-wave abnormality=miss |
| JS02165 | incomplete | Sinus rhythm=N/A(无对应规则) |
| JS02166 | abnormal_with_limited_coverage | Sinus tachycardia=match |
| JS02167 | normal_with_core_coverage | Sinus bradycardia=match |
| JS02168 | technically_limited | Sinus rhythm=N/A(无对应规则) |
| JS02169 | abnormal_with_limited_coverage | Atrial fibrillation=match; Right axis deviation=N/A(无对应规则); Nonspecific intraventricular conduction delay=miss |
| JS02170 | abnormal_with_limited_coverage | Atrial fibrillation=unavailable; ST depression=N/A(无对应规则); Nonspecific ST-T abnormality=match; T-wave abnormality=match |
| JS02171 | abnormal_with_limited_coverage | Sinus bradycardia=match; Counterclockwise cardiac rotation=N/A(无对应规则); Left ventricular hypertrophy=match |
| JS02172 | abnormal_with_limited_coverage | Sinus bradycardia=match |
| JS02173 | abnormal_with_limited_coverage | Sinus bradycardia=match; Low QRS voltages=match; T-wave abnormality=match |
| JS02174 | incomplete | Sinus bradycardia=match; Counterclockwise cardiac rotation=N/A(无对应规则) |
| JS02176 | abnormal_with_limited_coverage | Sinus tachycardia=match |
| JS02177 | incomplete | Sinus bradycardia=match; Q-wave abnormality=unavailable; T-wave abnormality=miss |
| JS02178 | normal_with_core_coverage | Sinus bradycardia=match |
| JS02179 | technically_limited | Sinus bradycardia=match |
| JS02180 | incomplete | Sinus rhythm=N/A(无对应规则); Left ventricular hypertrophy=miss |
| JS02181 | abnormal | Sinus bradycardia=match; Q-wave abnormality=match; Left ventricular hypertrophy=miss; T-wave abnormality=miss |
| JS02182 | abnormal_with_limited_coverage | Sinus rhythm=N/A(无对应规则) |
| JS02183 | abnormal_with_limited_coverage | Sinus bradycardia=match; Right bundle branch block=miss |
| JS02184 | abnormal_with_limited_coverage | Atrial fibrillation=match |
| JS02185 | incomplete | Sinus tachycardia=match |

## 2. 按标签类型汇总

| 标签 | match | miss | unavailable | N/A(恒不可用) | N/A(无对应规则) | N/A(未映射) |
|---|---|---|---|---|---|---|
| Atrial fibrillation | 12 | 1 | 4 | 0 | 0 | 0 |
| Atrial flutter | 1 | 0 | 0 | 2 | 0 | 0 |
| Atrioventricular block | 0 | 1 | 0 | 0 | 0 | 0 |
| Counterclockwise cardiac rotation | 0 | 0 | 0 | 0 | 3 | 0 |
| First-degree AV block | 1 | 0 | 0 | 0 | 0 | 0 |
| Left axis deviation | 0 | 0 | 0 | 0 | 4 | 0 |
| Left bundle branch block | 0 | 3 | 0 | 0 | 0 | 0 |
| Left ventricular hypertrophy | 7 | 4 | 0 | 0 | 0 | 0 |
| Low QRS voltages | 1 | 1 | 0 | 0 | 0 | 0 |
| Nonspecific ST-T abnormality | 9 | 5 | 0 | 0 | 0 | 0 |
| Nonspecific intraventricular conduction delay | 0 | 2 | 1 | 0 | 0 | 0 |
| Premature atrial contraction | 3 | 0 | 0 | 0 | 0 | 0 |
| Q-wave abnormality | 2 | 0 | 2 | 0 | 0 | 0 |
| Right axis deviation | 0 | 0 | 0 | 0 | 4 | 0 |
| Right bundle branch block | 0 | 1 | 0 | 0 | 0 | 0 |
| ST depression | 0 | 0 | 0 | 0 | 3 | 0 |
| Sinus bradycardia | 40 | 0 | 0 | 0 | 0 | 0 |
| Sinus rhythm | 0 | 0 | 0 | 0 | 22 | 0 |
| Sinus tachycardia | 16 | 0 | 2 | 0 | 0 | 0 |
| T-wave abnormality | 7 | 9 | 0 | 0 | 0 | 0 |
| T-wave inversion | 0 | 1 | 0 | 0 | 0 | 0 |
| Ventricular escape beat | 0 | 0 | 0 | 0 | 1 | 0 |
| Ventricular premature beats | 1 | 0 | 0 | 0 | 0 | 0 |