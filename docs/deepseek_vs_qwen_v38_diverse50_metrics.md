# DeepSeek vs. Qwen（v38 diverse50）指标计算方法与结果全记录

记录 2026-08-03 分析中用到的每一个指标的**精确定义、计算公式、以及在本次实验数据上的完整结果**，供后续复算、复核、或写入汇报材料时引用，避免重新推导。

## 0. 数据来源

- `qwen_agent_output/deepseek_v38_diverse50/`：DeepSeek 作为 backbone 跑的 50 条记录
- `qwen_agent_output/qwen_v38_diverse50/`：Qwen 作为 backbone 跑的 50 条记录
- 两者 `record_manifest.json` 里的 `dataset_dir` 完全一致（均指向 `qwen_v36_diverse50/wfdb_headers`），确认是**同一批 50 条 PTB-XL 记录**，可直接头对头比较。
- 涉及脚本：仓库自带的 `evaluate_target_ecgfeat_diagnosis.py`（`CATEGORIES` 定义）、`ecgagent/batch.py`（`_category_set`/`_metric_rows`，生成 `category_metrics.csv`/`record_results.csv` 的官方逻辑）、`score_measurement_verifiable.py`（测量复核）；本文档里标注"临时脚本"的部分是本次分析中现写的，不在仓库里，公式列在下面以便复现。

---

## 1. 第一层：逐记录 × 逐诊断家族的四格表

### 1.1 诊断家族定义

`evaluate_target_ecgfeat_diagnosis.py` 里预定义 `CATEGORIES: tuple[CategorySpec, ...]`，每个家族一个 `CategorySpec`：

```python
@dataclass
class CategorySpec:
    key: str                          # 家族 id，如 "atrial_fibrillation"
    display_name_zh: str              # 中文名
    ptbxl_refs: frozenset[str]        # 参考标签集合，如 {"AFIB"}
    local_refs: frozenset[str]
    prediction_codes: frozenset[str]  # 模型/规则引擎输出中，命中该家族的代码集合
    evaluates_codes: frozenset[str] = frozenset()
    rule_ids: frozenset[str] = frozenset()
    semantic_scope: str = "direct"    # "direct" | "broad" | "screening"
```

本次分析用到的 50 条记录里，`semantic_scope="direct"` 的家族共 **17 个**（见附录 A 完整列表）。headline 指标只汇总 `direct` 家族，`broad`（如 T波异常、心房异常、QRS低电压）和 `screening`（缺血/ST筛查、心肌梗死/Q波筛查）家族的定义本身模糊、判据不是单一数值比较，排除在外单独讨论。

### 1.2 四格表判定

对每条记录 `i`、每个家族 `spec`：

```
reference_positive = (记录 i 的 PTB-XL 标签集合) ∩ spec.ptbxl_refs ≠ ∅
predicted_positive  = (模型/规则引擎输出的代码集合) ∩ spec.prediction_codes ≠ ∅
```

|                        | reference_positive | ¬reference_positive |
|------------------------|---------------------|----------------------|
| **predicted_positive** | TP                  | FP                   |
| **¬predicted_positive**| FN                  | TN（不参与 F1）       |

逐记录、逐家族累加，得到 `category_metrics.csv` 里每个家族的 TP/FP/FN（`ecgagent/batch.py::_metric_rows`）。

---

## 2. 第二层：Precision / Recall / F1，及五种平均口径

标准公式：

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

同一份四格表数据，"怎么平均"有多种公认做法，结论会不一样。本次全部实算了一遍：

### 2.1 Micro-F1（汇总后算一次）

把所有 `direct` 家族、所有 50 条记录的 TP/FP/FN 先加总成一个总表，再算一次 P/R/F1。**本报告默认口径**，也是最先给出的数字。

### 2.2 Macro-F1（label-centric，逐家族先各算 F1 再平均）

```
Macro-F1 = (1/|F|) · Σ_{f∈F} F1_f
```
`F` 是 17 个 direct 家族的集合，每个家族权重相等，不管它在 50 条记录里出现了 1 次还是 6 次。这是 [PTB-XL 官方基准论文](https://arxiv.org/abs/2004.13701)（Strodthoff et al. 2020）§II.C 的核心口径选择（原文用于 AUC，本次我们的输出是二元判断，没有概率分数，故用同一"逐类别平权平均"的思路套在 F1 上）：论文原话——"macro-averaging is preferred, since we expect class imbalance and do not want the score to be dominated by a few large classes"。

### 2.3 Sample-centric F1（record-centric，逐记录先各算准召再平均）

同样来自 [PTB-XL 论文](https://arxiv.org/abs/2004.13701) §II.C，公式源自 CAFA 蛋白质功能预测挑战赛：

```
Precision = (1/N_P) · Σ_{i: |P_i|>0}  TP_i / |P_i|      # 只在"该记录至少预测了1个家族"的记录上取平均
Recall    = (1/N_T) · Σ_{i: |T_i|>0}  TP_i / |T_i|      # 只在"该记录真实至少有1个家族"的记录上取平均
F1        = 2 · Precision · Recall / (Precision + Recall)
```

其中 `P_i`/`T_i` 是记录 `i` 的预测/真实 direct 家族集合，`N_P`/`N_T` 是分母非零的记录数。原论文对预测端的分母处理是"只在有预测的样本上平均"，本次把召回端也对称地限制在"有真实标签的样本上平均"（因为本数据集里大量记录 direct 家族真实标签为空，跟 CAFA 场景——每个蛋白质必有功能标注——不同，需要这个对称处理才不会除以零）。**原论文的 Fmax 还会对预测阈值 τ 扫描取最大值**；本次模型输出是硬性二元判断（没有概率分数可扫描），相当于只算了阈值扫描曲线上唯一那一个点的 F1，不是真正的 Fmax。

### 2.4 Subset Accuracy（子集精确匹配率）

```
Subset Accuracy = (1/N) · Σ_i 𝟙[P_i = T_i]
```
逐记录判断"预测的家族集合是否与真实标签集合完全一致"，是/否二元，多标签分类文献里的经典指标之一。

### 2.5 平均 Jaccard 相似度

```
Jaccard_i = |P_i ∩ T_i| / |P_i ∪ T_i|     （P_i、T_i 都为空时记为 1）
Mean Jaccard = (1/N) · Σ_i Jaccard_i
```

---

## 3. 第三层 A：测量复核（`score_measurement_verifiable.py`）

第一、二层只看"模型代码字符串 vs 标签代码字符串"是否匹配，不管数值对不对。对定义本身就是一次阈值比较的家族（心率、PR/QT 区间、QRS 电压、心轴），脚本绕开标签，直接用记录自身的 `*_features.json` 重新计算：

```python
CHECKS = {
    "sinus_bradycardia": lambda r: r.g("heart_rate_bpm") < 60.0,
    "first_degree_av_block": lambda r: r.g("pr_ms") > 200.0,
    "prolonged_qt": lambda r: r.qtc() > (460.0 if r.female else 450.0),
    "left_axis_deviation": lambda r: -90.0 < r.g("qrs_axis_deg") < -30.0,
    "low_qrs_voltage_limb_leads": lambda r: all(amp < 0.50 for amp in 肢导振幅),
    # ... 完整列表见 score_measurement_verifiable.py
}
```

每条模型给出的诊断，若代码在 `CHECKS` 里，重算一次真值，分四桶：

| 桶 | 含义 |
|---|---|
| `measurement-confirmed and labelled` | 算出来是真的，标签也写了 |
| `unlabelled-but-correct` | 算出来是真的，标签没写（PTB-XL 报告医生没提，不算模型错） |
| `measurement-contradicted` | 算出来是假的——**真错误** |
| `not-checkable` | 没有对应 `CHECKS` 函数（LVH/RVH/WPW 等形态学家族），或该记录缺特征数据 |

用法：`python3 score_measurement_verifiable.py qwen_agent_output/deepseek_v38_diverse50 qwen_agent_output/qwen_v38_diverse50`

---

## 4. 第三层 B：乐观口径两步修正（临时脚本，本次现算，未入库）

两步都建立在第一层四格表之上，目的是从"表面 label-F1"里剥离两类已知记账噪音，得到"真实误判率"的乐观估计。

### 4.1 第一步：验证过的同义代码合并

`prediction_codes` 集合里手工加入三组、且逐条验证过语义等价的裸词变体：

```python
ALIAS_ADD = {
    "atrial_fibrillation":     {"atrial_fibrillation"},
    "first_degree_av_block":   {"first_degree_av_block", "possible_first_degree_av_delay"},
    "complete_av_block":       {"complete_av_block"},
}
```

**决定是否合并的方法**：把候选别名单独加进去，对规则基线（非 LLM，行为稳定，适合做 A/B 对照）重算一次四格表，看 TP/FP 变化：

| 候选别名 | 基线 ΔTP | 基线 ΔFP | 结论 |
|---|---|---|---|
| `bradycardia` → `sinus_bradycardia` | +1 | **+5** | 拒绝——指代更宽泛的"心率慢"，不是同一发现的换词 |
| `tachycardia` → `sinus_tachycardia` | +3 | **+4** | 拒绝，同上 |
| `possible_lvh_voltage` → `lvh_voltage_criteria` | +0 | **+3** | 拒绝，语义不等价 |
| `atrial_fibrillation` → `atrial_fibrillation_pattern` 家族 | 0 | 0 | **采纳** |
| `first_degree_av_block`/`possible_first_degree_av_delay` → `first_degree_av_delay` 家族 | 0 | +1 | **采纳**（FP 增量可忽略且能被第二步测量核查进一步清理） |
| `complete_av_block` → `complete_av_block_pattern` | 0 | 0 | **采纳** |

原则：**任何候选别名合并前，先对规则基线做一次 A/B 测试**，FP 增量明显大于 TP 增量就说明两个代码指代的不是同一件事，不能合并。

### 4.2 第二步：测量复核剔除假 FP

第一步合并后新产生的每一条 FP，把触发它的代码丢回第三层 A 的 `CHECKS` 函数，用该记录自己的 features 重新核查：核查为真（测量证实、只是标签没写）就把这条 FP 直接从统计里删除（既不算 TP 也不算 FP）；核查为假或该代码不可核查，则保留为 FP。

伪代码：

```python
for record in records:
    for spec in direct_scope_categories:
        ref_pos = bool(record.ref_codes & spec.ptbxl_refs)
        pred_codes = spec.prediction_codes | ALIAS_ADD.get(spec.key, set())
        matched = record.agent_codes & pred_codes
        pred_pos = bool(matched)
        if ref_pos and pred_pos:
            TP += 1
        elif pred_pos:
            if any(CHECKS.get(c) and CHECKS[c](record.features) is True for c in matched):
                pass  # 剔除，不计入 FP
            else:
                FP += 1
        elif ref_pos:
            FN += 1
```

---

## 5. 结果总表

### 5.1 Micro-F1（原始 vs 乐观修正）

| 来源 | 口径 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 规则基线 | 原始 | 23 | 39 | 23 | 0.371 | 0.500 | 0.426 |
| 规则基线 | 乐观修正 | 23 | 28 | 23 | 0.451 | 0.500 | **0.474** |
| DeepSeek | 原始 | 12 | 17 | 34 | 0.414 | 0.261 | 0.320 |
| DeepSeek | 乐观修正 | 12 | 9 | 34 | 0.571 | 0.261 | **0.358** |
| Qwen | 原始 | 9 | 19 | 37 | 0.321 | 0.196 | 0.243 |
| Qwen | 乐观修正 | 11 | 9 | 35 | 0.550 | 0.239 | **0.333** |

> Qwen 乐观修正后 TP 从 9→11（不只是 FP 减少）：`first_degree_av_block`/`atrial_fibrillation` 裸词合并后，有 2 条记录从 FN 变成了真正的 TP（原本模型说对了，只是代码字符串没被识别）。

### 5.2 五种平均口径对照（direct 家族）

| 口径 | 规则基线 | DeepSeek | Qwen | 规则基线（乐观） | DeepSeek（乐观） | Qwen（乐观） |
|---|---:|---:|---:|---:|---:|---:|
| Micro-F1 | 0.426 | 0.320 | 0.243 | 0.474 | 0.358 | 0.333 |
| Macro-F1（17家族全量平权） | 0.452 | 0.188 | 0.154 | 0.471 | 0.211 | 0.197 |
| Macro-F1（仅 16 个有真实阳性支持的家族） | 0.480 | 0.200 | 0.164 | — | — | — |
| Sample-centric F1（PTB-XL/CAFA 公式） | 0.425 | 0.347 | 0.261 | 0.469 | 0.389 | 0.352 |
| Subset Accuracy | 0.280 | **0.320** | 0.300 | — | — | — |
| 平均 Jaccard | 0.412 | 0.387 | 0.370 | — | — | — |
| 我方 ad hoc 样本宏 F1（0/0→0，保守） | 0.277 | 0.190 | 0.133 | — | — | — |
| 我方 ad hoc 样本宏 F1（0/0→1，宽松） | 0.457 | 0.410 | 0.393 | — | — | — |

补充统计（原始口径，用于理解上表分母）：50 条记录里 **21 条**在 direct 家族上真实标签为空（无任何 direct 家族命中参考标签）；其中"预测也为空、真实也为空"（trivial match）的记录数：规则基线 9 条，DeepSeek 11 条，Qwen 13 条。

**要点**：DeepSeek 在全部口径下都稳定优于 Qwen。基线相对两模型的领先幅度对口径非常敏感——Macro-F1 下基线约是两模型的 2.2~3 倍（6 个传导阻滞家族两模型都是 0% 召回，被平权放大），Micro/Sample-centric 下压缩到 1.2~1.6 倍（这些家族样本少，被平均稀释）；Subset Accuracy 下 DeepSeek 反而反超基线（0.320 vs 0.280），因为规则基线更"话多"（尤其 LVH 家族大量误报），逐记录看"家族集合是否完全命中"时反而容易因为多说了一两句而错过满分。

### 5.3 测量复核结果

| 模型 | 确认诊断总数 | not-checkable | unlabelled-but-correct | confirmed-and-labelled | 可核验诊断数 | 核验错误数 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 59 | 34 (57.6%) | 23 (39.0%) | 2 (3.4%) | 25 | **0 (0.0%)** |
| Qwen | 74 | 44 (59.5%) | 28 (37.8%) | 2 (2.7%) | 30 | **0 (0.0%)** |

### 5.4 分诊断家族召回率（13 个有信号的 direct 家族，原始口径）

| 家族 | n（真实阳性数） | 规则基线 | DeepSeek | Qwen |
|---|---:|---:|---:|---:|
| 窦性心动过缓 | 2 | 50% | 50% | 0% |
| 窦性心动过速 | 4 | 25% | 25% | 25% |
| 心房颤动 | 4 | 75% | 50% | 0%* |
| 房性早搏 | 2 | 100% | 0% | 0% |
| 室性早搏 | 5 | 60% | 60% | 60% |
| 右束支阻滞 RBBB | 6 | 33% | 0% | 0% |
| 左束支阻滞 LBBB | 2 | 50% | 0% | 0% |
| 左前分支阻滞 LAFB | 2 | 100% | 0% | 0% |
| 左后分支阻滞 LPFB | 1 | 100% | 0% | 0% |
| 非特异性室内传导延迟 | 4 | 25% | 0% | 0% |
| 心室预激 WPW | 1 | 100% | 100%* | 100% |
| 左室肥厚电压标准 | 4 | 50% | 100% | 100% |
| 右室肥厚 RVH | 3 | 100% | 0% | 0% |

`*` 心房颤动 Qwen 的 0% 是记账噪音（见 §6.2）；WPW DeepSeek 的 100% 召回下精确率只有 0.5（见 §6.1 案例）。

**结构性结论**：RBBB/LBBB/LAFB/LPFB/非特异性IVCD/RVH/房性早搏 共 7 个家族、合计 20/46（约 43%）的真实阳性，两模型召回率都是 0%——不是判断错，是这些诊断代码在当前 agent 输出里完全不出现。任何记账/标签修正都动不了这部分，需要修 pipeline 本身（见 §7 已知缺陷表）。

---

## 6. 案例分析（逐条核实）

### 6.1 09275_hr — DeepSeek 独有 WPW 假阳性

- 参考标签：`CRBBB|LAFB|RVH|SR`（完全性右束支阻滞+左前分支阻滞+右室肥厚，双分支阻滞），不含 WPW。
- 规则基线：`lafb_pattern|rvh_pattern|secondary_t_wave_abnormality`，同样不含 WPW。
- DeepSeek 唯一诊断：`ventricular_preexcitation_pattern`，HIGH 置信度，引用 `ev:/rhythm_inputs/preexcitation/short_pr_interval` 和 `delta_lead_count=3`、全局 `pr_ms=141`。
- 记录自身 `metadata.rhythm_analysis.rule_summary.preexcitation`（`ptbxl_09000_ecgfeat/features/09275_hr_features.json`）：`wpw_pattern: false`、`short_pr_interval: false`（141ms 属正常范围 120–200ms，不是短 PR）、`delta_lead_count: 3` < `delta_threshold: 8`。代入 `feature_extraction/ecgfeat/clinical_rules/preexcitation.py::evaluate_preexcitation()` 得到 `status="not_matched"`。
- **程序自己的确定性判据已经算出 false，且该字段就在模型引用的证据路径里，模型仍给出相反结论、HIGH 置信度。**
- 对照：Qwen 在同一条记录上给出 `left_axis_deviation`，未被误导；两模型在真实 WPW 病例 `09699_hr` 上都正确确认（精确率 1.0/1.0）。

### 6.2 09114_hr / 09592_hr / 09913_hr / 09570_hr / 09136_hr / 09550_hr — 诊断代码同义词分裂

`evaluate_target_ecgfeat_diagnosis.py` 里部分家族的 `prediction_codes` 只登记了一个变体，模型用了未登记的同义变体就被记为漏检：

| 记录 | 模型 | 实际输出代码 | 被计分识别的代码集合 | 后果 |
|---|---|---|---|---|
| 09114_hr | Qwen | `atrial_fibrillation` | `atrial_fibrillation_pattern`/`probable_*` | 房颤判断正确但计为 0 匹配 |
| 09592_hr | Qwen | `atrial_fibrillation` | 同上 | 同上，直接拉低 AFIB 家族召回 |
| 09913_hr | Qwen | `first_degree_av_block` | `first_degree_av_delay` | 一度房室延迟判断正确但未计分 |
| 09570_hr | DeepSeek | `first_degree_av_block` | `first_degree_av_delay` | 同上 |
| 09136_hr | DeepSeek | `t_wave_abnormality` | `primary_/secondary_t_wave_abnormality` | T波异常未计分（broad 家族，不影响 headline F1） |
| 09550_hr | Qwen | `possible_lvh_voltage` | `lvh_voltage_criteria` | LVH 提示未计分 |

本次样本：Qwen 命中 4 次、DeepSeek 命中 2 次——两侧都受影响，Qwen 这次踩中更多，其中 2 次直接砍掉 AFIB 家族本该有的召回率。

### 6.3 09114_hr — 两模型都成功剪除规则引擎历史噪声

规则基线给出 9 个代码，其中 `prior_infarct_q_wave_pattern` + `rvh_pattern` + `lvh_voltage_criteria` 同时出现（既往记录过的"高振幅QRS诱发病理性Q波误判"组合）。DeepSeek 与 Qwen 都把 9 个代码剪到只剩 2 个（房颤+LVH），与参考标签 `AFIB|INVT|NDT|PVC|STD_` 高度吻合。样本量有限（n=1），不代表缺陷系统性修复。

### 6.4 09570_hr — DeepSeek 识别陈旧心梗 Q 波，Qwen 漏检

参考标签含 `ASMI|ILMI`（前间壁+下侧壁陈旧心梗）。DeepSeek 输出 `prior_infarct_q_wave_pattern`，与标签直接吻合；Qwen 未给出该代码。

---

## 7. 已知缺陷复查状态

| 已知问题 | 状态 | 本次样本证据 |
|---|---|---|
| 传导阻滞家族总漏检（RBBB/LBBB/LAFB/LPFB/IVCD/RVH） | 仍然存在 | 两模型全部 6 家族 0% 召回，与规则基线 25%~100% 形成对比，性质与既往定位一致 |
| 缺血/心梗筛查召回率极低 | 仍然存在 | DeepSeek 缺血 0/18、心梗 1/16；Qwen 缺血 1/18、心梗 0/16；规则基线 7/18、4/16 |
| 诊断代码同义词分裂 | 仍然存在 | 本次新确认 6 处具体实例（§6.2） |
| LVH gate 覆盖面窄于旧规则引擎 | 部分改善 | F1 由 0.222 提升到 0.533，两模型完全一致（已程序化），7 个 FP 仍未逐条核实 |
| Q波假阳性/高振幅QRS诱发病理性Q波误判 | 本次未见新发 | 09114_hr 两模型均主动剪除该组合噪声，样本有限（n=1） |
| WPW/预激路径缺乏模型-程序一致性校验 | **新发现** | 09275_hr：DeepSeek 在程序判据明确为 false 时仍给出 HIGH 置信度 WPW |
| sinus_rhythm gate 停滞/AF model-override | 本次样本未触发 | 未观察到相关证据，样本量小，不代表已修复 |

---

## 8. 运行效率对比

| 指标 | DeepSeek | Qwen |
|---|---:|---:|
| 中位耗时 / P90 | 22.058s / 28.316s | 87.734s / 109.671s |
| 工具调用总数 / 记录均值 | 495 / 9.9 | 507 / 10.14 |
| Token：prompt / completion / cached | 617,993 / 60,756 / 286,336 | 427,957 / 71,790 / 0 |
| 验证通过率 / 修订轮次 | 50/50，0 | 50/50，0 |
| record_grade（Q0/Q1） | 46/4 | 46/4 |
| 逐记录相对基线：改善/不变/恶化 | 23/21/6 | 22/23/5 |

DeepSeek 约快 4 倍，且命中大量 prompt 缓存（Qwen 侧无缓存命中）。

---

## 9. 跨论文数字对比的坑（重要，避免误用）

已发表的 ECG-LLM/agent 论文（如 CARE-ECG、ECG-Chat）报告的 PTB-XL F1，**很可能包含 NORM（正常）/ SR（窦性心律）这类"一切正常"标签作为可得分类别**，且这类标签召回率天然很高，会显著拉高聚合指标：

- ECG-Chat 论文原话（Fig.4 说明）："Some common labels, such as 'Normal (NORM)', 'Sinus rhythm (SR)', have high F1 scores... Many labels have an F1 score of 0."
- ECG-Chat 自己论文报的 CE-Disease（含 NORM）F1 = 22.33%；CARE-ECG 论文里用同一个 "ECG-Chat" 模型做基线、同一个 PTB-XL 任务，却报出 F1 = 0.70——同一模型、同一数据集，评分协议不同导致数字差 3 倍以上。
- 我们自己的 `evaluate_target_ecgfeat_diagnosis.CATEGORIES` **没有 NORM/正常类别**，模型说"一切正常"永远拿不到 TP，只能在漏诊真实异常时被扣分。

**结论**：本文档 §5 的所有数字都只在"我们自己的三个来源（基线/DeepSeek/Qwen）之间"横向可比；不要直接拿去跟外部论文的 F1/Accuracy 数字比较，除非先确认对方的评分类别集合是否包含 NORM/SR，以及用的是 macro/micro/sample-centric 哪种平均口径。

---

## 附录 A：本次涉及的 17 个 direct-scope 家族完整列表

```
sinus_bradycardia, sinus_tachycardia, atrial_fibrillation, atrial_flutter,
premature_atrial_complexes, premature_ventricular_complexes,
first_degree_av_block, complete_av_block,
right_bundle_branch_block, left_bundle_branch_block,
left_anterior_fascicular_block, left_posterior_fascicular_block,
nonspecific_ivcd, ventricular_preexcitation,
left_ventricular_hypertrophy, right_ventricular_hypertrophy,
prolonged_qt
```

（`unspecified_ectopy_pattern`、`ambiguous_left_conduction_family`、`atrial_abnormality`、`low_qrs_voltage`、`t_wave_abnormality` 为 `broad` scope；`ischemia_or_st_abnormality`、`infarction_or_q_wave` 为 `screening` scope；均不计入本文档 §5 的 headline 指标。）

## 附录 B：复现命令

```bash
# 测量复核（§3）
python3 score_measurement_verifiable.py \
  qwen_agent_output/deepseek_v38_diverse50 \
  qwen_agent_output/qwen_v38_diverse50

# 第一/二层四格表与各平均口径、第三层B乐观修正：
# 均为本次分析中现写的一次性脚本，未入库；核心逻辑见本文档 §1、§2、§4 的伪代码，
# 依赖 evaluate_target_ecgfeat_diagnosis.CATEGORIES 与 score_measurement_verifiable.CHECKS，
# 输入取自各 run 目录下的 record_results.csv 与 features/*.json。
```

## 附录 C：相关项目记忆

- `project_v38_diverse50_deepseek_vs_qwen.md`
- `project_deepseek_wpw_gate_override.md`
- `project_diagnosis_catalog_synonym_split.md`
- `project_measurement_verifiable_scoring.md`
- `project_ecg_llm_literature_f1_not_comparable.md`
- `project_conduction_block_total_miss.md`
