# WFDBRecords/03/030 逐记录标签—检测匹配审计

- 数据范围：100 条记录（JS02082–JS02185，按实际存在文件列出）
- 特征结果：data/WFDBRecords_03_030_ecgfeat_out
- 统一临床规则：2026.07.4
- 审计口径：SNOMED 语义精确匹配；STTC 不等同于 TWC/TWO；泛化 AVB 只标记“家族兼容”，不算分级精确命中。
- 注意：数据库表头诊断是参考/弱标签，不是逐波形复核后的绝对金标准；“额外检测”不等于已证实假阳性。

## 结果解释表

| 检测项目 | 真实标签数 | 检测数 | 正确命中 | 漏检 | 额外检测 | 当前结论 |
|---|---:|---:|---:|---:|---:|---|
| 统一 T 波改变（TWC） | 16 | 18 | 7 | 9 | 11 | 命中率偏低，同时存在较多无同义标签检出，需要继续调整阈值和继发性 T 波归因 |
| T 波倒置（TWO） | 1 | 13 | 0 | 1 | 13 | 唯一标签病例漏检，且连续导联倒置候选较多，存在明显过检风险 |
| 房性早搏（PAC/APB） | 3 | 7 | 3 | 0 | 4 | 3 个标签病例全部命中，但额外 4 条仍需人工复核 |
| 室性早搏（PVC/VPB） | 1 | 1 | 1 | 0 | 0 | 当前样本全部匹配，但只有 1 个阳性样本，证据量不足 |
| 全导联低电压（LVQRSAL） | 2 | 0 | 0 | 2 | 0 | 两个标签病例均未完整命中；JS02173 只检出胸导联低电压，属于部分匹配 |
| 肢体导联低电压（LVQRSLL） | 0 | 0 | 0 | 0 | 0 | 本批没有阳性标签，无法评价召回能力 |
| 胸导联低电压（LVQRSCL） | 0 | 1 | 0 | 0 | 1 | JS02173 检出，但表头为全导联低电压，不能算胸导联标签的精确命中 |
| 预激（VPE/WPW） | 0 | 1 | 0 | 0 | 1 | JS02159 为疑似预激；没有阳性标签可验证，应保持人工确认 |
| 二度 AV block | 0 | 6 | 0 | 0 | 6 | 没有阳性标签却产生 6 个候选，需重点排查房扑/房颤及 P 波检测造成的误判 |
| 三度 AV block | 0 | 1 | 0 | 0 | 1 | JS02105 为疑似三度 AV block；无阳性标签可验证，不能作为确定诊断 |

说明：这里的“正确命中”是与数据库表头标签的语义匹配，不等同于经过心电专家复核后的临床正确率。

## P3 精确匹配汇总

| 项目 | 标签数 | 检出数 | 精确命中 | 漏检 | 无同义标签的检出 |
|---|---:|---:|---:|---:|---:|
| T波改变(TWC) | 16 | 18 | 7 | 9 | 11 |
| T波倒置(TWO) | 1 | 13 | 0 | 1 | 13 |
| PAC/APB | 3 | 7 | 3 | 0 | 4 |
| PVC/VPB | 1 | 1 | 1 | 0 | 0 |
| 全导联低电压 | 2 | 0 | 0 | 2 | 0 |
| 肢体导联低电压 | 0 | 0 | 0 | 0 | 0 |
| 胸导联低电压 | 0 | 1 | 0 | 0 | 1 |
| 预激(VPE/WPW) | 0 | 1 | 0 | 0 | 1 |
| 二度AV block | 0 | 6 | 0 | 0 | 6 |
| 三度AV block | 0 | 1 | 0 | 0 | 1 |

## 逐条结果

缩写：AFIB=房颤标签，AF=房扑标签；检测列中的 AF=统一规则房颤模式；LVH-V=满足LVH电压标准；?=筛查性/需人工确认。

| 记录 | 表头标签 | 总体状态 | 统一规则最终结果 | P3精确命中 | P3漏检 | P3额外检测 | 备注 |
|---|---|---|---|---|---|---|---|
| JS02082 | AF | abnormal_with_limited_coverage | LPFB, LVH-V, LeadRev?, T-secondary, AF, Tachy | — | — | TWC候选 | — |
| JS02083 | SB | abnormal_with_limited_coverage | LVH-V, ShortQT-borderline, LeadRev?, Brady | — | — | — | — |
| JS02084 | SB, ALS | abnormal_with_limited_coverage | LAFB, LeadRev?, Brady | — | — | — | — |
| JS02085 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02086 | SB, LVH | abnormal_with_limited_coverage | LVH-V, LongQT, LeadRev?, Brady | — | — | — | — |
| JS02087 | SB, 1AVB | abnormal_with_limited_coverage | AVB1, LeadRev?, Brady | — | — | — | — |
| JS02088 | SB | borderline | ShortQT?, LeadRev?, Brady | — | — | — | — |
| JS02089 | SR | technically_limited | TechLimited | — | — | — | — |
| JS02090 | SB, LVH, STTC | abnormal_with_limited_coverage | LVH-V, LeadRev?, T-secondary, Brady | — | — | TWC候选, TWO候选 | STTC不与TWC/TWO作精确同义匹配 |
| JS02091 | AFIB, AQW, IVB, LFBBB, TWC | abnormal_with_limited_coverage | AVB2?, PriorMI-Q, T-primary, Tachy | TWC | — | TWO候选, AVB2 | — |
| JS02092 | SB | incomplete | LeadRev?, Brady | — | — | — | — |
| JS02093 | SB | incomplete | Brady | — | — | — | — |
| JS02094 | SB | incomplete | LeadRev?, Brady | — | — | — | — |
| JS02095 | ST, CCR | abnormal_with_limited_coverage | PostIschemia?, AF, Tachy | — | — | — | — |
| JS02096 | SB, STTC | normal_with_core_coverage | Brady | — | — | — | STTC不与TWC/TWO作精确同义匹配 |
| JS02097 | SB | borderline | ShortQT-borderline, LeadRev?, Brady | — | — | — | — |
| JS02098 | AFIB, IVB, STTC | abnormal_with_limited_coverage | LVH-V | — | — | — | STTC不与TWC/TWO作精确同义匹配 |
| JS02099 | SR | abnormal | PostIschemia?, Occlusion? | — | — | — | — |
| JS02100 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02101 | SR, ALS | normal_with_core_coverage | — | — | — | — | — |
| JS02102 | SR, APB | incomplete | PAC, LeadRev? | APB/PAC | — | — | — |
| JS02103 | AFIB, STDD, STTC, TWC | abnormal_with_limited_coverage | LVH-V, LeadRev?, T-secondary | TWC | — | TWO候选 | STTC不与TWC/TWO作精确同义匹配 |
| JS02104 | AFIB, ARS, STTC | abnormal_with_limited_coverage | AVB2?, T-primary, Tachy | — | — | TWC候选, AVB2 | STTC不与TWC/TWO作精确同义匹配 |
| JS02105 | SB, APB, LVH, TWC | abnormal_with_limited_coverage | AVB3?, PAC, LVH-V, Occlusion?, LeadRev?, T-secondary, Brady | TWC, APB/PAC | — | TWO候选, AVB3 | — |
| JS02106 | AFIB | abnormal_with_limited_coverage | AVB2?, PAC, LeadRev? | — | — | PAC, AVB2 | — |
| JS02107 | AFIB, ARS, VPB | abnormal_with_limited_coverage | LPFB, PVC, LeadRev? | VPB/PVC | — | — | — |
| JS02108 | SR | technically_limited | TechLimited | — | — | — | — |
| JS02109 | AF, AVB | abnormal_with_limited_coverage | LAA, LAFB, AVB1, LongQT, LeadRev?, Tachy | — | — | — | AVB泛标签仅家族兼容：AVB1 |
| JS02110 | SB | incomplete | Brady | — | — | — | — |
| JS02111 | SR, STTC | technically_limited | TechLimited | — | — | — | STTC不与TWC/TWO作精确同义匹配 |
| JS02112 | ST, ARS | abnormal_with_limited_coverage | LPFB, LVH-V, LongQT, Tachy | — | — | — | — |
| JS02113 | ST | technically_limited_with_findings | AVB2?, TechLimited | — | — | AVB2 | — |
| JS02114 | SB, TWC | incomplete | LeadRev?, Brady | — | TWC | — | — |
| JS02115 | SB, LFBBB | abnormal_with_limited_coverage | NSIVCD, WideQRS-repol?, Brady | — | — | — | — |
| JS02116 | SB | technically_limited | TechLimited, Brady | — | — | — | — |
| JS02117 | SR | borderline | ShortQT-borderline, LeadRev? | — | — | — | — |
| JS02118 | AFIB, STDD, STTC, TWC, VEB | abnormal_with_limited_coverage | AVB2?, PAC, LVH-V, LeadRev?, T-secondary, Tachy | TWC | — | TWO候选, PAC, AVB2 | STTC不与TWC/TWO作精确同义匹配 |
| JS02119 | ST, TWC | technically_limited_with_findings | AVB1, TechLimited, Tachy | — | TWC | — | — |
| JS02120 | ST | incomplete | LeadRev? | — | — | — | — |
| JS02121 | ST, APB | abnormal_with_limited_coverage | PAC, LongQT, PostIschemia?, T-primary, Tachy | APB/PAC | — | TWC候选, TWO候选 | — |
| JS02122 | ST | incomplete | LeadRev?, Tachy | — | — | — | — |
| JS02124 | SB, LVH | technically_limited_with_findings | LVH-V, TechLimited, LeadRev?, Brady | — | — | — | — |
| JS02125 | SB, TWC | incomplete | LeadRev?, Brady | — | TWC | — | — |
| JS02126 | SR, STTC | abnormal_with_limited_coverage | LongQT, T-primary | — | — | TWC候选, TWO候选 | STTC不与TWC/TWO作精确同义匹配 |
| JS02127 | SB | abnormal | LAA, LeadRev?, Brady | — | — | — | — |
| JS02128 | SR | technically_limited_with_findings | LAA, TechLimited, LeadRev? | — | — | — | — |
| JS02129 | SB | borderline | ShortQT-borderline, Brady | — | — | — | — |
| JS02130 | ST, STTC | abnormal_with_limited_coverage | LAA, LeadRev?, T-primary, Tachy | — | — | TWC候选, TWO候选 | STTC不与TWC/TWO作精确同义匹配 |
| JS02131 | SR | incomplete | LeadRev? | — | — | — | — |
| JS02132 | AFIB, LVH, STTC | abnormal_with_limited_coverage | LVH-V, PriorMI-Q, LeadRev?, AF, Tachy | — | — | — | STTC不与TWC/TWO作精确同义匹配 |
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
| JS02184 | AFIB | abnormal_with_limited_coverage | LVH-V, LeadRev?, T-secondary, AF, Brady | — | — | TWC候选 | — |
| JS02185 | ST | incomplete | Tachy | — | — | — | — |

## 判读重点

- P3精确命中：标签与规则输出在同一诊断语义层级。
- P3漏检：存在对应表头标签，但最终规则未产生同义结果；不可用/不确定不计命中。
- P3额外检测：规则产生结果，但表头没有同义标签；必须回看波形和证据，不能直接称为假阳性。
- 对 LVQRSAL，只有肢体导联和胸导联低电压同时成立才算全导联精确命中。
