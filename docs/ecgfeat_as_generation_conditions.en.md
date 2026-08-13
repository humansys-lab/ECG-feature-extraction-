<!-- i18n-nav -->
[中文](ecgfeat_as_generation_conditions.md) | [English](ecgfeat_as_generation_conditions.en.md) | [日本語](ecgfeat_as_generation_conditions.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecgfeat-作为条件生成的特征源-可用性说明"></a>
# ecgfeat Feature source generated as a condition - usability instructions

Targeted use: Use ecgfeat as a feature extractor to provide conditional quantities for **conditional ECG generation**,
Control ST, rhythm, morphology, infarct area. This is different from the requirements for diagnostic use, and this article is re-evaluated against the generated criteria.

Package: [EDB ST Amplitude Verification](edb_st_cross_dataset_validation.en.md),
[QTDB tracing verification](qtdb_cross_dataset_validation.en.md), [Diagnostic feature gap registration](%E8%AF%8A%E6%96%AD%E7%89%B9%E5%BE%81%E7%BC%BA%E5%8F%A3%E7%99%BB%E8%AE%B0.en.md).

---

<a id="0-一句话结论"></a>
## 0. One sentence conclusion

**Lead-by-lead continuous measurements can be used directly; do not use ecgfeat for category labels, use the data set’s own annotation. **

The value of ecgfeat lies in providing fine-grained continuous quantities that cannot be expressed by dataset labels.
("V3 lead ST depression 0.15 mV at 80 ms", "QRS axis +45°", "P wave duration 110 ms"),
rather than reproducing the label itself - which would just introduce its own inaccuracies.

---

<a id="1-为什么不能照搬诊断结论"></a>
## 1. Why can’t we just copy the diagnostic conclusion?

The diagnostic evaluation asks "whether the report is accurate", while the generation condition asks three other things:

| Metrics | Meaning | Why it’s critical to generate |
|---|---|---|
| **Coverage** | Non-empty proportion | How many training samples are left when the feature is required to be non-empty |
| **Missing deviation** | Abnormal group coverage − Normal group coverage | **Non-zero means alarm**, see below |
| **Degradation** | Proportion of the most common values / coefficient of variation | Approximately constant characteristics, the generator cannot "adjust" |

**Missing bias is a pitfall unique to generation. ** If a feature cannot be calculated, it is safe to abstain in diagnosis;
But if it abstains more often on anomalous records, then "requiring the feature to be non-null" is equivalent to systematically filtering out anomalous samples,
The abnormal distribution learned by the generator is shaved off. Interpolation is equally harmful – it’s creating structures out of thin air.

---

<a id="2-推荐的条件集"></a>
## 2. Recommended condition set

Measurement basis: `cond_feature_audit.py`, PTB-XL 250 pieces
(NORM/MI/ISCHEMIC/AFIB/Conduction Abnormalities 50 each), 2026-08-09.

<a id="21-逐导联-12-维向量-覆盖-100缺失偏差-000"></a>
### 2.1 Lead-by-lead 12-dimensional vector - coverage 1.00, missing bias 0.00

```
ST     st_hybrid_j_mv, st_hybrid_60ms_mv, st_hybrid_80ms_mv,
       st_hybrid_slope_mv_per_ms, st_hybrid_baseline_confidence
形态   p_amp_mv, p_dur_ms, t_amp_mv, t_polarity, qrs_ms,
       r_amp_mv, s_amp_mv, q_amp_mv, t_symmetry
```

Among them, ST amplitude is the only group with cross-dataset accuracy verification: on EDB
MAE **0.099 mV**, bias −0.021 mV, r **0.878** (215 expert-labeled event peaks).

<a id="22-记录级标量"></a>
### 2.2 Record-level scalars

| Features | Coverage | Deviation | Degradation | Range |
|---|---|---|---|---|
| `rr_cv` | 1.00 | 0.00 | **0.00** | 0.002 – 0.464 |
| `qrs_axis_deg` | 1.00 | 0.00 | **0.00** | −114° – 123° |
| `heart_rate_bpm` | 1.00 | 0.00 | 0.72 | 37 – 172 |
| `qrs_ms` | 1.00 | 0.00 | 0.76 | 80 – 206 |
| `qt_ms` | 0.97 | −0.04 | 0.88 | 247 – 497 |

`rr_cv` was the best single condition for rhythm control (degradation 0.00 and AUC 0.913 for atrial fibrillation vs sinus).

<a id="23-类别条件"></a>
### 2.3 Category conditions

**Use the data set’s own annotation** (SCP code of PTB-XL). See Section 4 for reasons.

---

<a id="3-禁用清单"></a>
## 3. Disabled list

<a id="31-缺失系统性有偏-最危险的一类"></a>
### 3.1 Lack of systematic bias - the most dangerous type

| Features | Coverage | Missing Bias |
|---|---|---|
| `pr_ms` | 0.59 | **−0.34** |
| `t_axis_deg` | 0.60 | **−0.48** |
| `qrs_t_angle_deg` | 0.49 | **−0.46** |

Coverage on abnormal records is 34–48 percentage points lower than normal. The cause is physiological
(There is no PR in atrial fibrillation, and abnormal repolarization cannot be calculated as the T axis), **This is why it is dangerous** - the lack itself carries label information.

If you must use: **Add mask channel to explicitly model missing**, do not discard or interpolate.
"AF has no PR" is meaningful fact, not noise.

`q_duration_ms` (coverage 0.39, bias **+0.16**) is the opposite: bias is positive,
Because pathological Q waves only appear on abnormalities, its null means "no Q waves" rather than "undetectable".
The two types of null have different semantics and cannot be combined.

<a id="32-分布退化-与临床含义反相关"></a>
### 3.2 Distribution degradation + anti-correlation with clinical implications

`st_pattern_class` (four categories of forms), `st_morphology` (three categories of slope).

Actual measurement (PTB-XL 135, 1620 leads, 2026-08-09):

- `horizontal` represents **49%** of leads with a median ST80 of **+0.001 mV** ——
  It actually means "baseline flat" rather than "horizontal depression" as the class name implies.
- `downsloping` only 3%, `j_point_elevation` only 0.4% of leads, almost never trigger
- As an indicator of ischemia **AUC 0.38–0.47 (below random)**, while pure "proportion of depressed leads" is **0.857**
- **The combination of pattern and amplitude (0.72) is not as good as using amplitude alone (0.86)** ——The pattern is a negative contribution

Reason: These classes are determined by slope/curvature only, not magnitude. Most ST segments in resting electrocardiograms are flat.
The clinically meaningful entity is "horizontal **depression**", not "horizontal".

**Corollary**: The morphological classification of this warehouse is mostly a coarse-grained continuous quantity.
**Always take the continuous amount before binning** —— `st_hybrid_slope_mv_per_ms` is better than `st_morphology`,
`qrs_axis_deg` is better than `qrs_axis_class`.

<a id="33-梗死区域-首次评估不通过"></a>
### 3.3 Infarcted area - first assessment, failed

Measurement basis: `mi_territory_audit.py`, PTB-XL 50 + 50 in each area is normal.
The MI code of PTB-XL has its own area (IMI=inferior wall, ASMI=anterior septal wall, ALMI=anterior lateral wall), which can be directly compared.

| Output | Non-empty rate | Hit true value region | Hit rate in non-empty samples |
|---|---|---|---|
| `q_wave_territories` | 0.85 | 0.55 | 0.64 |
| `mi_statement_candidates.territory` | 0.75 | 0.54 | 0.72 |
| `st_territories_elevated` | **0.05** | 0.02 | — |

Confusion (a record may be assigned to multiple areas):

| True value | The area given by ecgfeat |
|---|---|
| inferior wall (n=50) | inferior **32** / anterior 17 / anterolateral 12 / lateral 6 |
| Anterior wall (n=50) | anterior **24** / anterolateral 16 / inferior 15 |
| **anterolateral** (n=50) | anterolateral **25** / anterior 24 / inferior 24 / lateral 18 |

**Anterior and lateral walls are basically non-specific** (correct 25, misjudged to the other three areas are 24/24/18 respectively, close to random).
And **34% of normal records were given MI statement**, `st_territories_depressed`
The trigger in the normal group was 0.94, **higher than** the 0.64–0.74 in the infarct group.

---

<a id="4-为什么类别条件要用数据集标签"></a>
## 4. Why do we need to use data set labels for category conditions?

When training on PTB-XL, the region/rhythm/diagnostic annotations are themselves ground truth.
Letting ecgfeat re-infer will only inject its error into the condition:

| Conditions | Error of ecgfeat |
|---|---|
| Infarct area | Hit rate 54% |
| Atrial fibrillation | Detected 67.3% |
| Atrial flutter | Detection 79.4% |

**The two types of conditions are sourced separately, each using its own strengths:**

- Category/Diagnostic/Region → **Dataset Label** (zero error)
- Continuous morphological quantity → **ecgfeat** (the part that cannot be expressed by labels)

---

<a id="5-q-波与-s-波可用作几何量不可用作病理量"></a>
## 5. Q wave and S wave: can be used as geometric quantities, but cannot be used as pathological quantities

Measurement basis: `qs_wave_probe.py`, PTB-XL NORM / MI / LVH each 50, 1800 lead rows.

<a id="51-无真值可对照"></a>
### 5.1 No true value to compare with

LUDB and QTDB **Only mark the starting point/peak/end point of QRS, not Q, R, S** separately.
Therefore, the amplitude and time limit** do not have any reference marks to check**; `test_mi_qwave.py` are all synthetic unit tests.

<a id="52-定义是极值而非负向波"></a>
### 5.2 The definition is "extreme value" rather than "negative wave"

```python
r_amp = max(0, max(seg))        # QRS 内最大正向偏转
q_amp = min(seg[:r_pos + 1])    # R 峰之前的最小值
s_amp = min(seg[r_pos:])        # R 峰之后的最小值
```

A QRS that starts with an r wave and has no Q wave at all will still get `q_amp_mv`, and it can be a positive number:

| | Measured ratio |
|---|---|
| `q_amp` is positive (no Q wave in this lead) | **24.4%** (median −0.044 mV, p90 **+0.037**) |
| `s_amp` is positive (no S wave) | **15.7%** |
| `q_duration_ms` has value | 32.1% |

<a id="53-病理判别力等同随机"></a>
### 5.3 Pathological discrimination is equal to random

| Features | MI vs Normal | LVH vs Normal |
|---|---|---|
| `q_amp` | **0.512** | 0.457 |
| `q_r_ratio` | 0.589 | 0.467 |
| `q_dur` | 0.598 | 0.538 |
| `s_amp` | **0.489** | 0.498 |

`significant_q` triggered 10.7% in **normal leads** and 18.7% in the MI group (only 1.75-fold enrichment).

<a id="54-但对生成而言仍然可用"></a>
### 5.4 but still available for builds

The generator does not need the feature to be diagnostic, only that it faithfully and reproducibly describes the waveform.
"minimum before R peak" is a deterministic, 100% covered geometric quantity that is fully available as a conditional dimension.

**Risk is purely interpretive**:

- Changed the name to `min_before_R` / `min_after_R` in documents and experimental descriptions, do not call it Q wave / S wave amplitude
- To control "the presence or absence of pathological Q waves" → use the data set MI tag, do not use `significant_q`
- Want faithful Q-wave dimensions → use `q_duration_ms` non-empty **indicator bit**,
  The amplitude is split into two dimensions according to the presence/absence of Q waves. Do not stuff the two physical quantities into the same axis.

---

<a id="6-使用须知"></a>
## 6. Instructions for use

1. **Missing bias is dataset dependent. **Numbers in Section 3.1 are from PTB-XL.
   Changing the corpus must be retested using `cond_feature_audit.py` - this is exactly the property that will quietly distort the learned distribution.

2. **AUC caliber. ** The AUC compared to PTB-XL record-level annotation in this article contains label noise,
   And the shape is lead by lead and the annotation is recording level (ischaemia often affects only some leads, and aggregation will dilute the signal).
   **Should read relative comparisons** (e.g. form vs pure magnitude), don't think of absolute values ​​as classifier performance.
   r = 0.878 on the EDB is a cleaner measure of ST amplitude.

3. **Verify blank list. ** The following have never been verified with real annotations in this repository:
   ST four types of morphology, infarct area (first evaluation in this article, failed), Q/S wave amplitude and time limit,
   12-lead adjacent zone rule (EDB has only 2–3 physical channels and is structurally unverifiable).

---

<a id="附复现用脚本"></a>
## Attached: Script for reproduction

It has been archived to `tools/generation_conditions/` and can be reused by changing the data path:

| Script | Function |
|---|---|
| `cond_feature_audit.py` | Coverage / Missing Bias / Degradation (Four Domains) |
| `mi_territory_audit.py` | Infarct area control PTB-XL area annotation |
| `qs_wave_probe.py` | Q/S semantic validity and discriminability |
| `st_morph_probe.py`, `st_joint_probe.py` | Test of clinical meaning of ST morphology class |
