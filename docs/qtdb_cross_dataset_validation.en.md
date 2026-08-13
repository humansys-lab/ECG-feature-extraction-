<!-- i18n-nav -->
[中文](qtdb_cross_dataset_validation.md) | [English](qtdb_cross_dataset_validation.en.md) | [日本語](qtdb_cross_dataset_validation.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecgfeat-跨数据集性能验证qt-databaseludb-之外"></a>
# ecgfeat Cross-dataset performance verification: QT Database (outside LUDB)

<a id="1-目的"></a>
## 1. Purpose

All previous boundary tracing verifications for ecgfeat were only done on LUDB. A single database cannot answer one key question:
**Are these indicators the true capabilities of ecgfeat, or are they overfitting of LUDB’s collection conditions and annotation habits? **

This time, independent verification is done on **QT Database (QTDB)**. QTDB is the field standard test set of waveform boundary algorithm.
And it is different from LUDB in every dimension that affects generalization judgment:

| Dimensions | LUDB | QTDB |
|---|---|---|
| Population source | Single center conventional 12-lead | MIT-BIH Arrhythmia/ST, ESC ST-T, Holter ECG and other multiple sources |
| Sampling rate | 500 Hz | 250 Hz (time resolution 4 ms) |
| Leads | Standard 12 leads | 2 ambulatory ECG channels (MLII, ECG1/2, D3/D4, CM5, etc.) |
| Annotation | Each record is annotated lead by lead | Each 15-minute record has only about 30 beats, and is annotated by another group of cardiologists according to another set of procedures |

<a id="2-数据集盘点"></a>
## 2. Data set inventory

Available data sets under local `data/` and their ability to verify:

| Data set | True value type | Whether to use this time |
|---|---|---|
| LUDB | Lead-by-lead P/QRS/T Boundary | Already have a baseline for comparison |
| **QTDB** | **Artificial P/QRS/T boundary (`.q1c`/`.q2c`)** | **✅ This selection** |
| EDB / LTSTDB | ST events and ST levels | ✅ For another ability axis, see [EDB ST Verification](edb_st_cross_dataset_validation.en.md) |
| PTB-XL | Diagnostic tag only | Local metadata does not contain commercial algorithm measurements for PTB-XL+, interval cannot be verified |
| Chapman/Shaoxing | Diagnostic tags | Unbounded truth values |

**QTDB is the only local non-LUDB dataset** that has ground-truth beat-by-beat boundaries, and is therefore the only choice to verify tracing performance.

<a id="3-方法"></a>
## 3. Method

The matching, error statistics and aggregation functions used in scoring are imported directly from `compare_ludb_detectors.py`**,
There is no reimplementation, so QTDB is directly comparable to the existing LUDB numbers. The only new ones are the truth parser and lead adapter.

<a id="31-稀疏导联适配"></a>
### 3.1 Sparse lead adaptation

`ECGFeatureExtractor` only accepts a 12-lead matrix, QTDB only has 2 channels. Use this repository
`analyze_physionet_st.py` Established convention for EDB/LTSTDB: put the real channel into QRS of ecgfeat
Detect slots **II and V2**, and the remaining 10 slots are zero. The slots are just adaptation positions and do not represent anatomical leads;
No 12-lead axis or diagnostic conclusions are generated. The manual annotations of QTDB are all written in channel 0, so only slot II is scored.

<a id="32-真值解析一处必须偏离-ludb-做法"></a>
### 3.2 Truth value analysis (one must deviate from the LUDB approach)

The annotation grammar of QTDB is `( 峰 )`, but the starting point and end point are optional** - QTDB often marks the T wave as `t)`,
There is no starting point. The parser for LUDB is anchored by `(` because each wave of LUDB is a complete pair.
If copied to QTDB, all T waves without a starting point will be silently discarded.

Actual measurement: Only **1412 (40%) of the 3542 T waves in the entire database are marked with a starting point**, and they are only distributed in 49/105 records.
Using `(` as an anchor shrinks the T-wave reference set to 40% and therefore falsely high sensitivity.
This parser is changed to use the peak symbol as the anchor, `(` and `)` are both processed as optional, and 3542 T waves are completely recalled.

<a id="33-按簇限制评分区间一处必须扩展-ludb-做法"></a>
### 3.3 Limit the scoring interval by cluster (one point must be extended LUDB method)

Only about 30 beats in each 15-minute record are marked. If the scoring interval is not restricted, the remaining real beats in the record will be recorded.
False positives - that measures annotation coverage, not detection accuracy. LUDB solves this problem with a single interval `[min, max]`.

**But this approach is not valid on QTDB**: QTDB has 43 records of annotations that are cut into multiple discontinuous clusters
(The 85 beats of `sel102` are scattered in 28 clusters, with a maximum interval of 62 s between clusters). A single `[min, max]` will
All normal heartbeats are scanned into the scoring area and counted as false positives.

Measured impact of this defect (first 6 records):

| | ecgfeat P | ecgfeat QRS | ecgfeat T |
|---|---|---|---|
| Single `[min,max]` interval | PPV 18.3% | PPV 20.0% | PPV 16.3% |
| Limit by cluster | **PPV 93.8%** | **PPV 85.3%** | **PPV 69.8%** |

NeuroKit2 also went from ~19% to ~85% - **Both methods collapse together, indicating that this is a scoring aperture defect rather than a detector defect**.
This assessment is clustered by reference beats (clusters separated by >3.0 s intervals) and is scored only within clusters. **When there is only one cluster,
This rule is completely equivalent to the single-interval approach of LUDB**, so the existing results of LUDB are not affected.

<a id="34-容差"></a>
### 3.4 Tolerance

QRS has a match tolerance of 75 ms and a P/T match tolerance of 150 ms, consistent with the LUDB benchmark. The error symbol is `检测 − 参考`.

<a id="4-结果105-条记录标注者-q1c3623-个参考心拍"></a>
## 4. Results (105 records, annotated by q1c, 3623 reference heartbeats)

Operating environment: wfdb 4.3.1, NeuroKit2 0.2.13, NumPy 2.2.6, SciPy 1.18.0;
The 12 process took 385 s; **105 records were recorded, and both methods failed in 0 cases** (no exceptions, no skipped records).

<a id="41-全库汇总"></a>
### 4.1 Full database summary

| Method | Wave | Reference | Se % | PPV % | Start bias±SD (ms) | Peak bias±SD (ms) | End bias±SD (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| ecgfeat | P | 3194 | 91.5 | 91.5 | 19.0±49.4 | −2.6±47.5 | −2.0±56.7 |
| ecgfeat | QRS | 3623 | 98.7 | 94.7 | −2.2±24.5 | −10.5±15.7 | 6.4±37.2 |
| ecgfeat | T | 3542 | 93.4 | 89.6 | −5.4±61.7 | −6.7±40.9 | −4.6±48.6 |
| neurokit2 | P | 3194 | 96.7 | 93.7 | 7.9±33.1 | −1.7±27.0 | −7.4±34.2 |
| neurokit2 | QRS | 3623 | 98.4 | 92.5 | −63.2±55.1 | −11.5±15.9 | 6.2±41.5 |
| neurokit2 | T | 3542 | 71.9 | 83.5 | 40.2±61.7 | 2.8±41.3 | −21.1±49.0 |

<a id="42-剔除起搏记录后"></a>
### 4.2 After excluding pacing records

QTDB inherits the MIT-BIH pacing record. **Confirmed by scanning `.hea` comment** (not hypothetical):
In the entire library, only two records, `sel102` and `sel104`, are marked as pacing - and they happen to be the two records with abnormal endpoint error of QRS.

| Method | Wave | Reference | Se % | PPV % | Endpoint bias±SD (ms) |
|---|---|---:|---:|---:|---:|
| ecgfeat | P | 3177 | 91.5 | 91.5 | −1.9±56.8 |
| ecgfeat | QRS | 3461 | **99.9** | 95.6 | 9.6±30.1 |
| ecgfeat | T | 3380 | **95.7** | 91.5 | −4.1±48.2 |
| neurokit2 | P | 3177 | 96.7 | 93.6 | −7.4±34.2 |
| neurokit2 | QRS | 3461 | 99.2 | 93.0 | 9.1±36.6 |
| neurokit2 | T | 3380 | 73.3 | 83.9 | −20.3±46.8 |

**After excluding 2 pacing records, ecgfeat only missed 4 out of 3461 reference heart beats (Se 99.9%). **

View pacing records alone (only 2 records, 162 beats):

| Method | QRS Se % | QRS End point bias±SD | T Se % |
|---|---:|---:|---:|
| ecgfeat | 72.8 | −99.1±76.6 | 45.7 |
| neurokit2 | 81.5 | −104.8±60.5 | 44.4 |

Both methods collapse together on the pacing recording (endpoint deviation is about −100 ms), indicating that this is **pacing pulse interference boundary positioning**
common problems; however, the sensitivity of QRS (72.8%) of ecgfeat is significantly lower than that of NeuroKit2 (81.5%), which can be improved.

<a id="43-逐记录分布每条记录只计一次避免个别坏记录主导结论"></a>
### 4.3 Record-by-record distribution (each record is only counted once to avoid individual bad records dominating the conclusion)

| Methods | Waves | Number of records | Endpoint MAE Median (ms) | p90 (ms) | Se Median % | Number of records with Se<90% |
|---|---|---:|---:|---:|---:|---:|
| ecgfeat | QRS | 105 | 14.6 | 52.4 | 100.0 | **1** |
| ecgfeat | P | 98 | 27.5 | 67.4 | 96.7 | **23** |
| ecgfeat | T | 103 | 21.3 | 84.7 | 100.0 | 12 |
| neurokit2 | QRS | 105 | 12.3 | 46.4 | 100.0 | 3 |
| neurokit2 | P | 98 | 12.4 | 49.3 | 100.0 | 6 |
| neurokit2 | T | 103 | 22.4 | 134.5 | 100.0 | **38** |

<a id="44-与-ludb-对照泛化结论"></a>
### 4.4 Comparison with LUDB - Generalization Conclusion

| Method | Wave | LUDB Se% | QTDB Se% | LUDB End point SD | QTDB End point SD |
|---|---|---:|---:|---:|---:|
| ecgfeat | P | 87.2 | 91.5 | 34.9 | 56.8 |
| ecgfeat | QRS | 99.0 | **99.9** | 18.7 | 30.1 |
| ecgfeat | T | 94.4 | **95.7** | 38.9 | 48.2 |
| neurokit2 | P | 95.8 | 96.7 | 47.0 | 34.2 |
| neurokit2 | QRS | 97.0 | 99.2 | 35.3 | 36.6 |
| neurokit2 | T | 76.4 | 73.3 | 51.2 | 46.8 |

**Detection sensitivity is fully maintained across libraries, with no signs of overfitting. ** Boundary SD generally becomes larger on QTDB,
Consistent with three factors of 250 Hz (4 ms quantization), ambulatory ECG lead noise, and 2 channels only.

<a id="45-cse-2σ-容差研究性对照非正式符合性结论"></a>
### 4.5 CSE 2σ Tolerance (Research Control, Informal Conformity Conclusion)

| Boundary | n | bias (ms) | SD (ms) | CSE 2σ limit | On target |
|---|---:|---:|---:|---:|:--:|
| P starting point | 2822 | 19.1 | 49.3 | 10.2 | ❌ |
| P end | 2908 | −1.9 | 56.8 | 12.7 | ❌ |
| QRS Starting point | 3208 | −2.1 | 24.7 | 6.5 | ❌ |
| QRS End | 3208 | 9.6 | 30.1 | 11.6 | ❌ |
| T end | 3231 | −4.1 | 48.2 | 30.6 | ❌ |

Neither ecgfeat nor NeuroKit2 **reached any of the CSE 2σ limits**. Note: CSE limit is for 12 leads
Global measurement design, using 2-channel dynamic ECG lead-by-lead results to compare with it; but even so,
**The biases are very small (≤19 ms), indicating that the systematic deviation is not large and the problem lies in the dispersion. **

<a id="46-运行时间"></a>
### 4.6 Running time

| Method | Median time consumption of each line | Real-time magnification |
|---|---:|---:|
| ecgfeat | 16.28 s | 2.3× |
| neurokit2 | 0.22 s | 171.6× |

ecgfeat is approximately **75 times** slower than NeuroKit2. It does not hinder offline analysis, but excludes real-time scenarios with low computing power.

<a id="5-关键结论"></a>
## 5. Key conclusions

1. **QRS detects excellent generalization across libraries. ** After excluding 2 pacing records, Se 99.9% (3461 beats missed, 4 beats),
   Only 1 out of 105 records has Se<90%. This is the most reliable capability of the ecgfeat.
2. **P waves are a stable recurring weakness. ** 87.2% on LUDB vs 95.8% on NeuroKit2, 91.5% vs 96.7% on QTDB;
   Looking record by record, there are **23/98 records with P sensitivity <90%**, while NeuroKit2 only has 6 records.
   Two independent databases gave the same conclusion, **This is not the LUDB annotation noise, but the real algorithm gap**.
3. **T wave is the advantage of stable recurrence. ** Se 95.7% on QTDB vs 73.3% on NeuroKit2 (94.4% vs 76.4% on LUDB).
   NeuroKit2 has 38/103 records with T sensitivity <90%, and ecgfeat has only 12 records.
4. **Paced rhythm is the clear failure mode**, both methods crash, but ecgfeat is worse (Se 72.8% vs 81.5%).
5. **Accuracy (dispersion) is weaker than detection. ** The SD of all boundaries exceeds the CSE 2σ limit, but the deviations are small.

<a id="5b-漏检成因分类是没输出还是放错位置"></a>
## 5b. Classification of causes of missed detection: "no output" or "misplaced"

The sensitivity numbers cannot differentiate between the two misses, which fix in opposite directions:

- **Abstention**: Nothing is output near the reference wave ecgfeat → The repair direction is **Relax gating**
- **Mislocalised**: ecgfeat does output the wave, but the position is out of tolerance → The repair direction is **Correction of search window/peak picking rule**

The scoring CSV cannot answer this question (only matching pairs are stored, unmatched detections are not visible),
Therefore, use `analyze_qtdb_failures.py` to run again and measure the distance to the nearest similar detection for each unmatched reference wave.
(>400 ms is judged as no output, within 400 ms is judged as misalignment; 400 ms is wider than any P/T wave and shorter than the typical RR,
Will not mistakenly borrow the waves of adjacent heartbeats).

<a id="5b1-三种波的漏检构成"></a>
### 5b.1 Missing detection composition of three types of waves

| Wave | Total number of missed detections | Misalignment | Misalignment proportion | Not output | Not output proportion | Misalignment median distance |
|---|---:|---:|---:|---:|---:|---:|
| QRS | 48 | 25 | 52.1% | 23 | 47.9% | 104 ms |
| P | 273 | 132 | 48.4% | 141 | 51.6% | 180 ms |
| **T** | 235 | **201** | **85.5%** | 34 | 14.5% | 216 ms |

**85.5% of missed T wave detections are due to misalignment, not lack of output. **That is, the 6.6% gap for T-wave sensitivity of 93.4%,
The main problem is not "the detector gave up", but "the output was put in the wrong place" - this is a peak/boundary rule problem,
Relaxing detection gating will not help. The P wave is half dislocated and half not output, and both paths must be repaired.

<a id="5b2-错位方向几乎全部偏早"></a>
### 5b.2 Dislocation direction: almost all "early"

| Wave | Early proportion | Late proportion | Signed median |
|---|---:|---:|---:|
| P | 76.5% | 23.5% | **−176 ms** |
| T | 88.1% | 11.9% | **−212 ms** |
| QRS | **100.0%** | 0.0% | −104 ms |

The three wave heights are consistently early, indicating that it is not random jitter, but that the starting point of the search window or the peak criterion is systematically early.

<a id="5b3-p-波错位落在哪里排除了-t-波和-u-波"></a>
### 5b.3 Where does the P-wave dislocation fall (T-wave and U-wave excluded)

PR intervals for 132 misplaced P waves: reference median **156 ms**, detection median **318 ms**. The distribution is bimodal:

| Detection PR | Number of cases | Proportion | Meaning |
|---|---:|---:|---|
| < 120 ms | 30 | 22.7% | Falling into the QRS frontier (`sel42` mode) |
| 120–250 ms | 0 | 0.0% | Normal PR area (if it falls here, it will be matched, so it is 0) |
| 250–400 ms | **102** | **77.3%** | Obviously early |

Attribution was made to the early batch (5 records with the most severe P wave failure, 71 cases):

| Placement | Number of examples | Proportion |
|---|---:|---:|
| T wave peak ±60 ms | 0 | 0.0% |
| T wave end ±60 ms | 0 | 0.0% |
| **Between T end point and true P (TP segment baseline)** | **42** | **59.2%** |
| Others (mainly after true P and close to QRS) | 29 | 40.8% |

**The hypothesis of false locking of T wave is denied (0%), and the hypothesis of false locking of U wave is basically denied**
(`sel42`, `sele0303` have no U-wave annotation at all, and `sele0704` has only 21%).
The real landing point is the **flat baseline area of the TP segment**——ecgfeat selected the peak of the baseline noise in the P-wave search window,
rather than a true P deflection. On the dynamic ECG leads of QTDB (ECG1, CM5, D3, V5), the P wave amplitude is already very small.
With only 2 channels, it cannot be corrected by multi-lead consensus. The baseline noise is comparable to the P wave amplitude, so it is easy to choose the wrong peak.

<a id="5b4-两个纯粹的对照案例"></a>
### 5b.4 Two purely comparative cases

| Record | Channel | P missed detection | Dislocation | Not output | Properties |
|---|---|---:|---:|---:|---|
| `sel42` | ECG1 | 30 | **29** | 1 | Almost pure dislocation |
| `sele0112` | D3 | 22 | **0** | **22** | Pure not output (PPV 100%, all pairs detected) |

These two records are also "P sensitivity is extremely low", but the causes are completely opposite: `sel42` is a misaligned peak.
`sele0112` is the detection gate that rejects all true Ps (all outputs are correct).
**It is impossible to modify these two records with the same change**, which is why looking at sensitivity alone can mislead the optimization direction.

Records with the highest concentration of missed detections (top 6):

| Record | Channel | Wave | Missing detection | Dislocation / Not output |
|---|---|---|---:|---|
| `sel102` (pacing) | V5 | T | 51 | 36 / 15 |
| `sel102` (pacing) | V5 | QRS | 39 | 21 / 18 |
| `sel104` (pacing) | V5 | T | 37 | 32 / 5 |
| `sel42` | ECG1 | P | 30 | 29 / 1 |
| `sele0106` | D3 | T | 29 | 28 / 1 |
| `sele0112` | D3 | P | 22 | 0 / 22 |

Recurrence: `python analyze_qtdb_failures.py --workers 12 --out qtdb_failure_analysis`,
Details are at `qtdb_failure_analysis/missed_events.csv`.

<a id="p-波失效的具体机制sel42-实例"></a>
### Specific mechanism of P wave failure (`sel42` example)

30 reference P waves for `sel42` ecgfeat **None matched (Se 0%)**, but it is not "no P detected"——
It detected 29 Ps. After alignment, look at the interval from P peak to QRS:

- Reference: **220 ms** (normal PR interval)
- ecgfeat: **68–72 ms**

That is, ecgfeat is locked to a certain deflection immediately before QRS, rather than a real P wave, with a deviation of about 200 ms.
Exceeding the 150 ms matching tolerance, all are judged as missed detections. This is a reproducible pattern of P-wave localization failure,
Consistent with the statistical weaknesses of Conclusion 2.

<a id="5c-基于上述定位做的改进锚点旁证门控"></a>
## 5c. Improvements based on the above positioning: anchor point circumstantial evidence gating

<a id="5c1-因果链"></a>
### 5c.1 Causal chain

The P anchor point derived from multi-lead fusion will be used to clamp the lead-by-lead search window to `±P_PRIOR_MARGIN` (40 ms).
The scorebook has used `_P_FUSED_ANCHOR_SINGLE_LEAD_PENALTY = 0.15` to suppress the single-lead cluster.
But the suppressed cluster may still be the winner, and once the winner falls on the wrong deflection,
The real P wave is completely removed from the searchable range, and the fine engine with SNR criterion has no chance of recovery.

<a id="5c2-改动"></a>
### 5c.2 Changes

`_fuse_peak_anchor` returns `(anchor, lead_support)` instead; when the anchor lacks circumstantial evidence,
Remove all three `_fused_p_peak` usage points uniformly (rather than blocking just one call point).

The rule is **relative**, which is key: **require collateral evidence only if ≥2 qualified leads would already provide it**.
Originally written as an absolute "support number ≥ 2", it was blocked by existing tests - those tests put `STANDARD_12_LEADS`
Piling into `["II"]`, the maximum number of supports in the single-lead world can only be 1, and the absolute rule will completely abolish this ability.
**The goal is "one lead is inconsistent with multiple other leads", not "one lead exists in isolation". **

<a id="5c3-双库验证结果判定标准在跑之前已定死"></a>
### 5c.3 Dual database verification results (the judgment criteria have been determined before running)

QTDB 105 items, ecgfeat:

| Wave | TP | FP | Sensitivity % | PPV % | Endpoint SD (ms) |
|---|---|---|---|---|---|
| **P** | 2921 → **2944** | 272 → **244** | 91.45 → **92.17** | 91.48 → **92.35** | 56.74 → **55.16** |
| QRS | 3575 → 3575 | 200 → 200 | 98.68 → 98.68 | 94.70 → 94.70 | 37.17 → 37.17 |
| T | 3307 → 3307 | 383 → 383 | 93.37 → 93.37 | 89.62 → 89.62 | 48.58 → 48.58 |

**Four indicators of P wave improved in the same direction** (23 more true Ps were detected, 28 fewer false positives were detected, and the boundary dispersion was also reduced by 1.58 ms),
It’s not about exchanging one item for another. QRS and T remain unchanged bit by bit, which is consistent with the expectation that the change only affects the P channel.

LUDB 200 items: **All indicators remain unchanged bit by bit**. When the qualified leads on the clean 12 leads are ≥2, the anchor point can always obtain multi-lead collateral evidence.
The gate never fires - this is exactly what the relative rule is designed to do, the risk of zero regression is an actual measurement and not an inference.

Judgment: **PASS** (the default standard is "QTDB P's Se or PPV improves, and LUDB does not deteriorate beyond
0.3pp / 0.5 ms", the actual Se and PPV are both improved, LUDB has no change).

<a id="5c4-被证伪后回退的尝试保留记录避免重走"></a>
### 5c.4 Attempt to roll back after being falsified (keep records to avoid repeating)

The first thing to do is **noise relative gating** (change the fixed 8 µV lower limit to `max(8µV, 2σ)`),
The dual database verification is unchanged bit by bit, it is a complete no-op and has been fully rolled back. Reasons worth remembering:
The threshold is indeed raised (2σ median 28 µV on failure record, trigger rate 90–100%),
But the bad candidate's baseline deviation is as high as **135 µV**, well beyond any threshold of this magnitude -
**The problem lies in candidate ranking, not candidate gating**.

This also excludes two other types of practices:
- **local bump**: bad candidate bump 110 µV, **larger** than true P's 95 µV, cannot be separated;
- **PR Prior**: Bad candidate PRs are concentrated at 250–400 ms, while long PRs for true first-degree AV blocks
  Falling in the same range, adding a penalty is equivalent to exchanging the accuracy of AV blocking for the P sensitivity of QTDB.

Only when all three types of discriminants fail do they point to the "number of supported leads", a dimension that has nothing to do with amplitude.

<a id="5c5-t-侧尝试合并-stt-分量的峰重定位-fail已回退"></a>
### 5c.5 T-side attempt: merge peak relocation of ST+T components - FAIL, rolled back

**Change**: When the peaks of consecutive segments with the same number fall in the ST segment (distance from the end of QRS < `_T_ST_SEGMENT_GUARD_MS` = 120 ms),
When the component extends beyond the boundary, the peak is relocated to the extreme value in the later period (only the peak position is changed, and the component is not re-segmented).
And the condition "there must be an internal inflection point in the later stage" is added to prevent pure ST monotonic decay from being mistakenly improved.
(The unit verifies that all four situations are in line with the design, including that the guard is valid).

**Dual database results**:

| | QTDB T | LUDB T |
|---|---|---|
| Sensitivity | 93.37 → **93.62** | 94.42 → **94.82** |
| PPV | 89.62 → **89.82** | 95.64 → **95.99** |
| Peak SD (ms) | — | 33.24 → **33.89** ❌ |
| Start/end SD (ms) | — | +0.31 / +0.11 (within tolerance) |

P and QRS remain unchanged bit by bit on the two libraries, confirming that the change only affects the T channel.

**Judgment FAIL**: LUDB T peak SD degradation +0.65 ms, exceeding the preset 0.5 ms tolerance.
**T detection improved for both libraries**, this is a real trade-off rather than a simple failure; but the default criteria is
"If any item exceeds it, it will fall back." After seeing the data, relaxing the standards to let it pass is a hindsight indicator, so it will be rolled back according to the rules.

**Root cause judgment**: `_T_ST_SEGMENT_GUARD_MS` is a **fixed offset and does not expand or contract with the heart rate**.
At a fast heart rate (short QT), the true T peak may originally fall at < 120 ms from the end point of QRS.
The rules will misjudge these **early T peaks that are originally correct as ST segments and move them back**——
This just explains the combination of "detection getting better and peak accuracy getting worse": catching more Ts that are overwhelmed by STs,
At the same time, a batch of legal early T-peaks were destroyed.

**Reserved part**: 5 scattered `120.0` literals have been raised as shared constants (behavior neutral,
Verified by P/QRS bit by bit invariant), the annotation states that it does not expand and contract with heart rate,
Therefore ** is only used to exclude candidates, never to move candidates **, and this lesson is fixed in the code.

<a id="5c6-t-侧第二次尝试率自适应边界-同样-fail诊断被推翻"></a>
### 5c.6 T-side second attempt: rate adaptive bounds - same FAIL, diagnosis overturned

It was speculated in the previous section that "the degradation of peak SD comes from the mis-shifting of the legitimate early T peak at fast heart rates", so the boundary was changed to
`√(RR/1000ms)` scaling (same rate relationship as Bazett QTc, not a new parameter; the scaling factor is sandwiched between
[0.5, 1.25] is only used as a safety guardrail, and RR takes the previous RR interval).

The unit verification verification mechanism works as designed: at fast heart rates (RR=400 ms, boundary 75.9 ms),
The legitimate early T peak ** no longer moved ** at 96 ms from the QRS end point, but would be mistakenly moved at the fixed 120 ms.

**But the dual database results overturned this diagnosis**:

| Indicators | Fixed 120 ms | Rate adaptive |
|---|---|---|
| LUDB T Sensitivity | +0.40pp | **+0.56pp** |
| LUDB T PPV | +0.35pp | **+0.52pp** |
| **LUDB T Peak SD** | **+0.65 ms** ❌ | **+0.81 ms** ❌ (worse) |
| QTDB T Sensitivity | +0.25pp | **+0.39pp** |
| QTDB T PPV | +0.20pp | +0.31pp |

**The detection gain is greater, and so is the loss in peak accuracy—the trade-off is steepened, not resolved. **
If the diagnosis is established, rate adaptation should maintain the gain and eliminate the loss; in fact, the two amplify in the same direction,
This indicates that the peak SD degradation is **not** coming from fast heart rate misshifts.

**More possible explanation**: The T peak label of LUDB itself may fall on the main peak of the merged deflection.
Therefore, "moving the peak out of the ST segment" is itself a deviation from the reference standard, not a correction.
Both attempts failed at the same indicator, and the more moves, the greater the failure, which is consistent with this explanation.

**Conclusion: The peak relocation path was falsified under the reference of LUDB, and has been rolled back, and the third variant will not be tried again. **
Continuing to adjust parameters without new evidence is fishing on the data. Rollback integrity checked:
All 15 items of the record × wave comparison are consistent with the verified status of the P side, with 0 differences.

If restarted in the future, what is needed is **new evidence** rather than new parameters - for example, first verify the respective LUDB/QTDB
Where exactly does the T peak mark fall on the ST elevation heart beat, and then determine whether the "peak position of the combined component" is really calculated incorrectly.

<a id="6-已知局限影响结论解读需明确"></a>
## 6. Known limitations (affect interpretation of conclusions, need to be clear)

1. **ecgfeat was entered under the condition of missing 10 leads. ** The P wave positioning of ecgfeat relies on multi-lead consensus,
   QTDB only provides 2 channels; NeuroKit2 is a single-lead algorithm and is not affected by this.
   Therefore the ecgfeat-vs-NeuroKit2 comparison on QTDB is therefore unfavorable on ecgfeat and the P-wave gap is amplified**.
   Cross-database recurrence (the gap also exists in LUDB 12 leads) is the real basis for "P wave is indeed a weakness".
2. **Only 40% of the T waves at the starting point have true values**, and they are concentrated in 49/105 records. This statistic has selection bias.
3. **PPV depends on clustering threshold** (see §3.3). The threshold of 3.0 s is a judgmental choice and is not naturally separable in the data:
   The intra-cluster interval is up to 3.00 s and the inter-cluster interval is as low as 3.02 s, with no clean bimodal demarcation.

   Two thresholds were measured on the affected 43 multi-cluster records (ecgfeat):

   | Threshold | P Se / PPV | QRS Se / PPV | T Se / PPV | Number of matches TP |
   |---|---|---|---|---|
   | 1.5 s | 92.8 / **96.3** | 97.3 / **97.1** | 91.1 / **92.7** | 1269 / 1714 / 1531 |
   | 3.0 s (used in this report) | 92.8 / **89.1** | 97.3 / **89.6** | 91.1 / **83.7** | 1269 / 1714 / 1531 |

   **The TP numbers of the three waves are exactly the same (1269/1714/1531), and the sensitivity is not different by one digit** - This is confirmed by actual measurement
   The clustering threshold only removes detections and does not change the reference set, so the sensitivity and all boundary error statistics are immune to this threshold.
   Can be quoted unconditionally. Only PPV is affected, and the 3.0 s used in this report is on the more conservative end
   (PPV is low), so the PPV in §4 should be understood as a lower bound.
4. This evaluation only performs border tracing and does not involve the 12-lead axis, ST positioning, or diagnostic layers.
5. U-waves are annotated in QTDB (821), but the beat-by-beat feature of ecgfeat does not export U-wave boundaries and is not within the scoring range.

<a id="7-复现"></a>
## 7. Recurrence

```bash
# 全库评估（105 条记录，双方法，12 进程约 6.5 分钟）
python evaluate_qtdb.py --workers 12 --out qtdb_evaluation

# 分层分析（起搏剔除、逐记录分布、与 LUDB 对照）
python analyze_qtdb_results.py --qtdb-dir qtdb_evaluation

# 诊断：不做按簇限制时 PPV 如何崩塌
python evaluate_qtdb.py --n 6 --no-restrict-to-annotated-span --out /tmp/qtdb_nospan
```

Output: `qtdb_evaluation/` under `summary_by_method_wave.csv`, `summary_by_record.csv`,
`detection_by_record.csv`, `matched_events.csv`, `cse_conformance.csv`,
`metrics.json`, `report.md`.
