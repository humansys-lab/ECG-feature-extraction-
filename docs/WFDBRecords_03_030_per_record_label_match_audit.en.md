<!-- i18n-nav -->
[中文](WFDBRecords_03_030_per_record_label_match_audit.md) | [English](WFDBRecords_03_030_per_record_label_match_audit.en.md) | [日本語](WFDBRecords_03_030_per_record_label_match_audit.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="wfdbrecords03030-逐记录标签检测匹配审计"></a>
# WFDBRecords/03/030 Record-by-record tag—detection matching audit

- Data range: 100 records (JS02082–JS02185, listed according to actual existing files)
- Feature results: data/WFDBRecords_03_030_ecgfeat_out
- Unified Clinical Rules: 2026.07.4
- Audit caliber: SNOMED semantic exact match; STTC is not equivalent to TWC/TWO; generalized AVB is only marked "family compatible" and does not count as hierarchical exact hits.
- Note: The database header diagnosis is a reference/weak label, not the absolute gold standard after waveform-by-wave review; "additional testing" does not mean confirmed false positives.

<a id="结果解释表"></a>
## Result explanation table

| Detection items | Number of true tags | Number of detections | Correct hits | Missed detections | Additional detections | Current conclusions |
|---|---:|---:|---:|---:|---:|---|
| Unified T-wave changes (TWC) | 16 | 18 | 7 | 9 | 11 | The hit rate is low, and there are many detections without synonymous tags. It is necessary to continue to adjust the threshold and secondary T wave attribution |
| T wave inversion (TWO) | 1 | 13 | 0 | 1 | 13 | The unique label case is missed, and there are many candidates for continuous lead inversion, and there is an obvious risk of over-detection |
| Premature atrial contractions (PAC/APB) | 3 | 7 | 3 | 0 | 4 | All 3 labeled cases were hit, but the additional 4 still require manual review |
| Premature ventricular contractions (PVC/VPB) | 1 | 1 | 1 | 0 | 0 | All current samples match, but there is only 1 positive sample, insufficient evidence |
| Low voltage in all leads (LVQRSAL) | 2 | 0 | 0 | 2 | 0 | Neither of the two labeled cases was completely hit; JS02173 only detected low voltage in the chest leads, which is a partial match |
| Limb lead low voltage (LVQRSLL) | 0 | 0 | 0 | 0 | 0 | This batch does not have a positive label and the recall ability cannot be evaluated |
| Chest lead low voltage (LVQRSCL) | 0 | 1 | 0 | 0 | 1 | JS02173 detected, but the meter header is low voltage for all leads, which cannot be counted as an accurate hit of the chest lead label |
| Pre-excitation (VPE/WPW) | 0 | 1 | 0 | 0 | 1 | JS02159 is suspected pre-excitation; there is no positive label to verify, manual confirmation should be maintained |
| Second AV block | 0 | 6 | 0 | 0 | 6 | There are no positive labels but 6 candidates are generated. It is necessary to focus on troubleshooting misjudgments caused by atrial flutter/atrial fibrillation and P wave detection |
| Third-degree AV block | 0 | 1 | 0 | 0 | 1 | JS02105 is a suspected third-degree AV block; there is no positive label to verify, and it cannot be used as a definite diagnosis |

Note: The "correct hit" here is a semantic match with the database header label, which is not equivalent to the clinical accuracy rate after review by an electrocardiogram expert.

<a id="p3-精确匹配汇总"></a>
## P3 exact match summary

| Project | Number of tags | Number of detections | Exact hits | Missed detections | Detections without synonymous tags |
|---|---:|---:|---:|---:|---:|
| T wave change (TWC) | 16 | 18 | 7 | 9 | 11 |
| T-wave inversion (TWO) | 1 | 13 | 0 | 1 | 13 |
| PAC/APB | 3 | 7 | 3 | 0 | 4 |
| PVC/VPB | 1 | 1 | 1 | 0 | 0 |
| Full lead low voltage | 2 | 0 | 0 | 2 | 0 |
| Limb lead low voltage | 0 | 0 | 0 | 0 | 0 |
| Low voltage in chest leads | 0 | 1 | 0 | 0 | 1 |
| Pre-excitation (VPE/WPW) | 0 | 1 | 0 | 0 | 1 |
| Second AV block | 0 | 6 | 0 | 0 | 6 |
| Three-dimensional AV block | 0 | 1 | 0 | 0 | 1 |

<a id="逐条结果"></a>
## Results one by one

Abbreviation: AFIB = atrial fibrillation label, AF = atrial flutter label; AF in the detection column = unified rule atrial fibrillation pattern; LVH-V = meets LVH voltage standard; ? = screening/requires manual confirmation.

| Record | Header label | Overall status | Unified rule final result | P3 precise hit | P3 missed detection | P3 additional detection | Remarks |
|---|---|---|---|---|---|---|---|
| JS02082 | AF | abnormal_with_limited_coverage | LPFB, LVH-V, LeadRev?, T-secondary, AF, Tachy | — | — | TWC candidate | — |
| JS02083 | SB | abnormal_with_limited_coverage | LVH-V, ShortQT-borderline, LeadRev?, Brady | — | — | — | — |
| JS02084 | SB, ALS | abnormal_with_limited_coverage | LAFB, LeadRev?, Brady | — | — | — | — |
| JS02085 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02086 | SB, LVH | abnormal_with_limited_coverage | LVH-V, LongQT, LeadRev?, Brady | — | — | — | — |
| JS02087 | SB, 1AVB | abnormal_with_limited_coverage | AVB1, LeadRev?, Brady | — | — | — | — |
| JS02088 | SB | borderline | ShortQT?, LeadRev?, Brady | — | — | — | — |
| JS02089 | SR | technically_limited | TechLimited | — | — | — | — |
| JS02090 | SB, LVH, STTC | abnormal_with_limited_coverage | LVH-V, LeadRev?, T-secondary, Brady | — | — | TWC candidate, TWO candidate | STTC does not make an exact synonymous match with TWC/TWO |
| JS02091 | AFIB, AQW, IVB, LFBBB, TWC | abnormal_with_limited_coverage | AVB2?, PriorMI-Q, T-primary, Tachy | TWC | — | TWO candidate, AVB2 | — |
| JS02092 | SB | incomplete | LeadRev?, Brady | — | — | — | — |
| JS02093 | SB | incomplete | Brady | — | — | — | — |
| JS02094 | SB | incomplete | LeadRev?, Brady | — | — | — | — |
| JS02095 | ST, CCR | abnormal_with_limited_coverage | PostIschemia?, AF, Tachy | — | — | — | — |
| JS02096 | SB, STTC | normal_with_core_coverage | Brady | — | — | — | STTC does not make an exact synonymous match with TWC/TWO |
| JS02097 | SB | borderline | ShortQT-borderline, LeadRev?, Brady | — | — | — | — |
| JS02098 | AFIB, IVB, STTC | abnormal_with_limited_coverage | LVH-V | — | — | — | STTC does not make an exact synonymous match with TWC/TWO |
| JS02099 | SR | abnormal | PostIschemia?, Occlusion? | — | — | — | — |
| JS02100 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02101 | SR, ALS | normal_with_core_coverage | — | — | — | — | — |
| JS02102 | SR, APB | incomplete | PAC, LeadRev? | APB/PAC | — | — | — |
| JS02103 | AFIB, STDD, STTC, TWC | abnormal_with_limited_coverage | LVH-V, LeadRev?, T-secondary | TWC | — | TWO candidate | STTC does not make an exact synonymous match with TWC/TWO |
| JS02104 | AFIB, ARS, STTC | abnormal_with_limited_coverage | AVB2?, T-primary, Tachy | — | — | TWC candidate, AVB2 | STTC does not make an exact synonymous match with TWC/TWO |
| JS02105 | SB, APB, LVH, TWC | abnormal_with_limited_coverage | AVB3?, PAC, LVH-V, Occlusion?, LeadRev?, T-secondary, Brady | TWC, APB/PAC | — | TWO candidate, AVB3 | — |
| JS02106 | AFIB | abnormal_with_limited_coverage | AVB2?, PAC, LeadRev? | — | — | PAC, AVB2 | — |
| JS02107 | AFIB, ARS, VPB | abnormal_with_limited_coverage | LPFB, PVC, LeadRev? | VPB/PVC | — | — | — |
| JS02108 | SR | technically_limited | TechLimited | — | — | — | — |
| JS02109 | AF, AVB | abnormal_with_limited_coverage | LAA, LAFB, AVB1, LongQT, LeadRev?, Tachy | — | — | — | AVB pan tag only family compatible: AVB1 |
| JS02110 | SB | incomplete | Brady | — | — | — | — |
| JS02111 | SR, STTC | technically_limited | TechLimited | — | — | — | STTC does not make an exact synonymous match with TWC/TWO |
| JS02112 | ST, ARS | abnormal_with_limited_coverage | LPFB, LVH-V, LongQT, Tachy | — | — | — | — |
| JS02113 | ST | technically_limited_with_findings | AVB2?, TechLimited | — | — | AVB2 | — |
| JS02114 | SB, TWC | incomplete | LeadRev?, Brady | — | TWC | — | — |
| JS02115 | SB, LFBBB | abnormal_with_limited_coverage | NSIVCD, WideQRS-repol?, Brady | — | — | — | — |
| JS02116 | SB | technically_limited | TechLimited, Brady | — | — | — | — |
| JS02117 | SR | borderline | ShortQT-borderline, LeadRev? | — | — | — | — |
| JS02118 | AFIB, STDD, STTC, TWC, VEB | abnormal_with_limited_coverage | AVB2?, PAC, LVH-V, LeadRev?, T-secondary, Tachy | TWC | — | TWO candidate, PAC, AVB2 | STTC does not make an exact synonymous match with TWC/TWO |
| JS02119 | ST, TWC | technically_limited_with_findings | AVB1, TechLimited, Tachy | — | TWC | — | — |
| JS02120 | ST | incomplete | LeadRev? | — | — | — | — |
| JS02121 | ST, APB | abnormal_with_limited_coverage | PAC, LongQT, PostIschemia?, T-primary, Tachy | APB/PAC | — | TWC candidate, TWO candidate | — |
| JS02122 | ST | incomplete | LeadRev?, Tachy | — | — | — | — |
| JS02124 | SB, LVH | technically_limited_with_findings | LVH-V, TechLimited, LeadRev?, Brady | — | — | — | — |
| JS02125 | SB, TWC | incomplete | LeadRev?, Brady | — | TWC | — | — |
| JS02126 | SR, STTC | abnormal_with_limited_coverage | LongQT, T-primary | — | — | TWC candidate, TWO candidate | STTC does not make an exact synonymous match with TWC/TWO |
| JS02127 | SB | abnormal | LAA, LeadRev?, Brady | — | — | — | — |
| JS02128 | SR | technically_limited_with_findings | LAA, TechLimited, LeadRev? | — | — | — | — |
| JS02129 | SB | borderline | ShortQT-borderline, Brady | — | — | — | — |
| JS02130 | ST, STTC | abnormal_with_limited_coverage | LAA, LeadRev?, T-primary, Tachy | — | — | TWC candidate, TWO candidate | STTC does not make an exact synonymous match with TWC/TWO |
| JS02131 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02132 | AFIB, LVH, STTC | abnormal_with_limited_coverage | LVH-V, PriorMI-Q, LeadRev?, AF, Tachy | — | — | — | STTC does not make an exact synonymous match with TWC/TWO |
| JS02133 | SB, TWC | abnormal_with_limited_coverage | PriorMI-Q, LeadRev?, Brady | — | TWC | — | — |
| JS02134 | ST, ALS, LVH | abnormal_with_limited_coverage | LAFB, AF, Tachy | — | — | — | — |
| JS02135 | SR | normal_with_core_coverage | — | — | — | — | — |
| JS02136 | AFIB, TWC | technically_limited_with_findings | PAC, ShortQT-borderline, TechLimited | — | TWC | PAC | — |
| JS02137 | AFIB, ALS, TWC | abnormal_with_limited_coverage | PAC, LVH-V, LongQT, PriorMI-Q | — | TWC | PAC | — |
| JS02138 | AF, LFBBB, STTC | abnormal_with_limited_coverage | RBBB?, LVH-V, T-secondary, AF, Tachy | — | — | TWC候选, TWO候选 | STTC不与TWC/TWO作精确同义匹配 |
| JS02139 | SB | abnormal | LPFB, LVH-V, ShortQT-borderline, Brady | — | — | — | — |
| JS02140 | SB | abnormal_with_limited_coverage | LAFB, LVH-V, LeadRev?, Brady | — | — | — | — |
| JS02141 | SB | incomplete | Brady | — | — | — | — |
| JS02143 | SB, AQW, TWO | borderline | ShortQT-borderline, LeadRev?, Brady | — | TWO | — | — |
| JS02144 | ST | incomplete | LeadRev?, Tachy | — | — | — | — |
| JS02145 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02146 | SR | technically_limited | TechLimited, LeadRev? | — | — | — | — |
| JS02147 | SR, LVQRSAL | abnormal | AVB1, LongQT, LeadRev? | — | LVQRSAL | — | — |
| JS02148 | ST | abnormal_with_limited_coverage | LAA, LeadRev?, Tachy | — | — | — | — |
| JS02149 | AFIB, LVH, STTC | abnormal_with_limited_coverage | LVH-V, T-secondary, AF, Tachy | — | — | TWC候选, TWO候选 | STTC不与TWC/TWO作精确同义匹配 |
| JS02150 | SB | borderline | ShortQT-borderline, Brady | — | — | — | — |
| JS02151 | SB, STTC | technically_limited_with_findings | NSIVCD, Occlusion?, TechLimited, LeadRev?, Brady | — | — | — | STTC不与TWC/TWO作精确同义匹配 |
| JS02152 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02153 | ST | abnormal_with_limited_coverage | LVH-V, LeadRev?, Tachy | — | — | — | — |
| JS02154 | SR | normal_with_core_coverage | LeadRev? | — | — | — | — |
| JS02155 | SB | normal_with_core_coverage | LeadRev?, Brady | — | — | — | — |
| JS02156 | SB | technically_limited | TechLimited, LeadRev?, Brady | — | — | — | — |
| JS02157 | ST | abnormal | AVB1, Tachy | — | — | — | — |
| JS02158 | ST | incomplete | LeadRev?, Tachy | — | — | — | — |
| JS02159 | SB, LVH, TWC | technically_limited_with_findings | Preexc?, TechLimited, LeadRev?, T-secondary, Brady | TWC | — | TWO候选, Preexc | — |
| JS02160 | AFIB | abnormal_with_limited_coverage | AF | — | — | — | — |
| JS02161 | AFIB | abnormal_with_limited_coverage | LVH-V, LeadRev?, AF | — | — | — | — |
| JS02163 | ST | incomplete | Tachy | — | — | — | — |
| JS02164 | AFIB, TWC | abnormal_with_limited_coverage | LeadRev?, AF | — | TWC | — | — |
| JS02165 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02166 | ST | abnormal_with_limited_coverage | LVH-V, Tachy | — | — | — | — |
| JS02167 | SB | normal_with_core_coverage | Brady | — | — | — | — |
| JS02168 | SR | technically_limited | TechLimited | — | — | — | — |
| JS02169 | AFIB, ARS, IVB | abnormal_with_limited_coverage | LPFB, LVH-V, LeadRev?, T-secondary, AF, Tachy | — | — | TWC候选 | — |
| JS02170 | AFIB, STDD, STTC, TWC | abnormal_with_limited_coverage | AVB2?, LVH-V, Occlusion?, LeadRev?, T-secondary, Tachy | TWC | — | TWO候选, AVB2 | STTC不与TWC/TWO作精确同义匹配 |
| JS02171 | SB, CCR, LVH | abnormal_with_limited_coverage | LVH-V, LeadRev?, Brady | — | — | — | — |
| JS02172 | SB | abnormal_with_limited_coverage | ShortQT?, T-primary, Brady | — | — | TWC候选 | — |
| JS02173 | SB, LVQRSAL, TWC | abnormal_with_limited_coverage | LeadRev?, T-primary, Brady, LowV-chest | TWC | LVQRSAL | TWO候选, LowV-chest | 全导联低电压仅胸导联部分匹配 |
| JS02174 | SB, CCR | incomplete | LeadRev?, Brady | — | — | — | — |
| JS02176 | ST | abnormal_with_limited_coverage | LVH-V, LeadRev?, Tachy | — | — | — | — |
| JS02177 | SB, AQW, TWC | incomplete | Brady | — | TWC | — | — |
| JS02178 | SB | normal_with_core_coverage | LeadRev?, Brady | — | — | — | — |
| JS02179 | SB | technically_limited | TechLimited, Brady | — | — | — | — |
| JS02180 | SR, LVH | incomplete | LeadRev? | — | — | — | — |
| JS02181 | SB, AQW, LVH, TWC | abnormal | PriorMI-Q, LeadRev?, Brady | — | TWC | — | — |
| JS02182 | SR | abnormal_with_limited_coverage | AVB1 | — | — | — | — |
| JS02183 | SB, RBBB | abnormal_with_limited_coverage | Occlusion?, Brady | — | — | — | — |
| JS02184 | AFIB | abnormal_with_limited_coverage | LVH-V, LeadRev?, T-secondary, AF, Brady | — | — | TWC candidate | — |
| JS02185 | ST | incomplete | Tachy | — | — | — | — |

<a id="判读重点"></a>
## Interpret key points

- P3 precise hit: label and rule output are at the same diagnostic semantic level.
- P3 missed detection: The corresponding header label exists, but the final rule does not produce synonymous results; unavailable/uncertain will not be counted as a hit.
- P3 additional detection: The rule generates results, but there is no synonymous label in the table header; the waveform and evidence must be reviewed, and it cannot be directly called a false positive.
- For LVQRSAL, only the low voltage in limb leads and chest leads is established at the same time to be considered an accurate hit in all leads.
