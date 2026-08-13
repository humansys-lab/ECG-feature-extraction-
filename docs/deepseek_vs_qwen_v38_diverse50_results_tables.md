<!-- i18n-nav -->
[中文](deepseek_vs_qwen_v38_diverse50_results_tables.md) | [English](deepseek_vs_qwen_v38_diverse50_results_tables.en.md) | [日本語](deepseek_vs_qwen_v38_diverse50_results_tables.ja.md)
<!-- /i18n-nav -->

# DeepSeek vs Qwen（v38 diverse50）结果表格

纯表格版，无正文说明；公式定义与逐条案例分析见 [`deepseek_vs_qwen_v38_diverse50_metrics.md`](deepseek_vs_qwen_v38_diverse50_metrics.md)；同一份数据的 Excel 多工作表版见 [`deepseek_vs_qwen_v38_diverse50_results.xlsx`](deepseek_vs_qwen_v38_diverse50_results.xlsx)。

## 1. 核心指标总览

| 指标 | 规则基线 | DeepSeek | Qwen | 备注 |
|---|---|---|---|---|
| 表面 Label-F1（direct家族, micro） | 0.426 | 0.320 | 0.243 | 均低于基线，含大量标签遗漏噪音 |
| 乐观口径 F1（同义词合并+测量剔除假FP） | 0.474 | 0.358 | 0.333 | 两步修正后的保守估计 |
| 测量可核验诊断数 / 错误数 | — | 25 / 0 | 30 / 0 | 两模型均 0 算术错误 |
| 测量可核验诊断中零错误占比 | — | 100% | 100% | |
| 假阳性中"标签遗漏但实为正确"占比 | — | 92% (23/25) | 93% (28/30) | |
| 中位单条耗时 | — | 22.1s | 87.7s | DeepSeek 约快4倍 |
| P90 单条耗时 | — | 28.3s | 109.7s | |
| 工具调用总数 / 记录均值 | — | 495 / 9.9 | 507 / 10.1 | 路径长度相近 |
| Prompt / Completion / Cached tokens | — | 617,993 / 60,756 / 286,336 | 427,957 / 71,790 / 0 | DeepSeek命中大量缓存 |
| 验证通过率 | — | 50/50 | 50/50 | 0 执行错误，0 修订轮次 |
| record_grade 分布 (Q0/Q1) | 46/4 | 46/4 | 46/4 | 数据质量分级完全一致 |
| 逐记录相对基线：改善/不变/恶化 | — | 23/21/6 | 22/23/5 | |

## 2. 五种平均口径对照（direct 家族）

| 口径 | 规则基线（原始） | DeepSeek（原始） | Qwen（原始） | 规则基线（乐观） | DeepSeek（乐观） | Qwen（乐观） | 定义来源 |
|---|---|---|---|---|---|---|---|
| Micro-F1（汇总四格表后算一次） | 0.426 | 0.320 | 0.243 | 0.474 | 0.358 | 0.333 | 本报告默认口径 |
| Macro-F1（17家族全量平权） | 0.452 | 0.188 | 0.154 | 0.471 | 0.211 | 0.197 | PTB-XL论文 term-centric 思路 |
| Macro-F1（16个有真实阳性支持的家族） | 0.480 | 0.200 | 0.164 | — | — | — | 同上，排除0支持家族 |
| Sample-centric F1（PTB-XL/CAFA公式） | 0.425 | 0.347 | 0.261 | 0.469 | 0.389 | 0.352 | PTB-XL论文 §II.C，源自CAFA挑战赛 |
| Subset Accuracy（家族集合完全匹配） | 0.280 | 0.320 | 0.300 | — | — | — | 多标签分类经典指标 |
| 平均 Jaccard 相似度 | 0.412 | 0.387 | 0.370 | — | — | — | 多标签分类经典指标 |
| 样本宏F1（0/0→0，保守） | 0.277 | 0.190 | 0.133 | — | — | — | 本次ad hoc，sklearn式零除处理 |
| 样本宏F1（0/0→1，宽松） | 0.457 | 0.410 | 0.393 | — | — | — | 本次ad hoc，trivial-match惯例 |

## 3. Micro-F1 底层混淆矩阵明细

| 来源 | 口径 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 规则基线 | 原始 | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
| 规则基线 | 乐观修正 | 23 | 28 | 23 | 0.451 | 0.500 | 0.474 |
| DeepSeek | 原始 | 12 | 17 | 34 | 0.414 | 0.261 | 0.320 |
| DeepSeek | 乐观修正 | 12 | 9 | 34 | 0.571 | 0.261 | 0.358 |
| Qwen | 原始 | 9 | 19 | 37 | 0.321 | 0.196 | 0.243 |
| Qwen | 乐观修正 | 11 | 9 | 35 | 0.550 | 0.239 | 0.333 |

## 4. 测量复核结果

| 模型 | 确认诊断总数 | not-checkable（形态学，无法核查） | unlabelled-but-correct（标签遗漏但正确） | confirmed-and-labelled（测量确认且有标签） | 可核验诊断数 | 核验错误数 | 错误率 |
|---|---:|---|---|---|---:|---:|---:|
| DeepSeek | 59 | 34 (57.6%) | 23 (39.0%) | 2 (3.4%) | 25 | 0 | 0.0% |
| Qwen | 74 | 44 (59.5%) | 28 (37.8%) | 2 (2.7%) | 30 | 0 | 0.0% |

## 5. 分诊断家族召回率（13个有信号的 direct 家族，原始口径）

| 诊断家族 | 真实阳性数 n | 规则基线 | DeepSeek | Qwen | 备注 |
|---|---:|---|---|---|---|
| 窦性心动过缓 | 2 | 50% | 50% | 0% | |
| 窦性心动过速 | 4 | 25% | 25% | 25% | |
| 心房颤动 | 4 | 75% | 50% | 0% | Qwen为记账噪音，见 §7 |
| 房性早搏 | 2 | 100% | 0% | 0% | 两模型总漏检 |
| 室性早搏 | 5 | 60% | 60% | 60% | |
| 右束支阻滞 RBBB | 6 | 33% | 0% | 0% | 两模型总漏检 |
| 左束支阻滞 LBBB | 2 | 50% | 0% | 0% | 两模型总漏检 |
| 左前分支阻滞 LAFB | 2 | 100% | 0% | 0% | 两模型总漏检 |
| 左后分支阻滞 LPFB | 1 | 100% | 0% | 0% | 两模型总漏检 |
| 非特异性室内传导延迟 | 4 | 25% | 0% | 0% | 两模型总漏检 |
| 心室预激 WPW | 1 | 100% | 100% | 100% | DeepSeek精确率仅0.5，见 §6 |
| 左室肥厚电压标准 | 4 | 50% | 100% | 100% | 两模型完全一致（已程序化） |
| 右室肥厚 RVH | 3 | 100% | 0% | 0% | 两模型总漏检 |

## 6. 关键分歧记录案例分析

| 记录ID | 案例类型 | 参考标签 | DeepSeek输出 | Qwen输出 | 结论 |
|---|---|---|---|---|---|
| 09275_hr | DeepSeek独有WPW假阳性（新发现） | CRBBB\|LAFB\|RVH\|SR | ventricular_preexcitation_pattern (HIGH) | left_axis_deviation | 程序自身wpw_pattern判据=false，且在模型引用证据路径中，模型仍给出相反结论 |
| 09699_hr | 对照：真实WPW病例，两模型均正确 | 含WPW | ventricular_preexcitation_pattern ✓ | ventricular_preexcitation_pattern ✓ | 证明DeepSeek不是系统性偏向误报WPW |
| 09114_hr | 两模型均剪除规则引擎历史噪声 | AFIB\|INVT\|NDT\|PVC\|STD_ | AF+LVH（从基线9代码剪至2） | AF+LVH（同样剪至2） | 均未沿用基线的Q波/RVH误报组合 |
| 09570_hr | DeepSeek优势案例 | ASMI\|ILMI\|IRBBB\|SVARR\|ABQRS | 含prior_infarct_q_wave_pattern ✓ | 未给出该代码 ✗ | DeepSeek正确识别陈旧心梗Q波，Qwen漏检 |

## 7. 诊断代码同义词分裂 — 确认实例

| 记录ID | 模型 | 实际输出代码 | 被计分识别的代码集合 | 后果 |
|---|---|---|---|---|
| 09114_hr | Qwen | atrial_fibrillation | atrial_fibrillation_pattern / probable_* | 房颤判断正确但计为0匹配 |
| 09592_hr | Qwen | atrial_fibrillation | atrial_fibrillation_pattern / probable_* | 同上，直接拉低AFIB家族召回 |
| 09913_hr | Qwen | first_degree_av_block | first_degree_av_delay | 一度房室延迟判断正确但未计分 |
| 09570_hr | DeepSeek | first_degree_av_block | first_degree_av_delay | 同上 |
| 09136_hr | DeepSeek | t_wave_abnormality | primary_/secondary_t_wave_abnormality | T波异常未计分（broad家族，不影响headline F1） |
| 09550_hr | Qwen | possible_lvh_voltage | lvh_voltage_criteria | LVH提示未计分 |

本次样本：Qwen 命中 4 次、DeepSeek 命中 2 次——两侧都受影响，Qwen 这次踩中更多，其中 2 次直接砍掉 AFIB 家族本该有的召回率。

## 8. 已知缺陷复查状态

| 已知问题 | 状态 | 本次样本证据 |
|---|---|---|
| 传导阻滞家族总漏检（RBBB/LBBB/LAFB/LPFB/IVCD/RVH） | 🔴 仍然存在 | 两模型全部6家族0%召回，与规则基线25%~100%形成对比，性质与既往定位一致 |
| 缺血/心梗筛查召回率极低 | 🔴 仍然存在 | DeepSeek缺血0/18、心梗1/16；Qwen缺血1/18、心梗0/16；规则基线7/18、4/16 |
| 诊断代码同义词分裂 | 🔴 仍然存在 | 本次新确认6处具体实例（§7） |
| LVH gate覆盖面窄于旧规则引擎 | 🟡 部分改善 | F1由0.222提升到0.533，两模型完全一致（已程序化），7个FP仍未逐条核实 |
| Q波假阳性/高振幅QRS诱发病理性Q波误判 | 🟢 本次未见新发 | 09114_hr两模型均主动剪除该组合噪声，样本有限（n=1） |
| WPW/预激路径缺乏模型-程序一致性校验 | 🔴 新发现 | 09275_hr：DeepSeek在程序判据明确为false时仍给出HIGH置信度WPW |
| sinus_rhythm gate停滞/AF model-override | ⚪ 本次样本未触发 | 未观察到相关证据，样本量小，不代表已修复 |

## 9. 运行效率对比

| 指标 | DeepSeek | Qwen |
|---|---|---|
| 中位耗时 / P90 | 22.058s / 28.316s | 87.734s / 109.671s |
| 工具调用总数 / 记录均值 | 495 / 9.9 | 507 / 10.14 |
| Token：prompt / completion / cached | 617,993 / 60,756 / 286,336 | 427,957 / 71,790 / 0 |
| 验证通过率 / 修订轮次 | 50/50，0 | 50/50，0 |
| record_grade（Q0/Q1） | 46/4 | 46/4 |
| 逐记录相对基线：改善/不变/恶化 | 23/21/6 | 22/23/5 |
