<!-- i18n-nav -->
[中文](QRS.md) | [English](QRS.en.md) | [日本語](QRS.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="多导联心电图波形检测ecg-delineation技术调研"></a>
# Multi-lead ECG waveform detection (ECG Delineation) technology research

**Topic**: QRS High-precision extraction of onset/offset, R wave position and morphology, Q wave and QT related features; multi-lead reliability screening + robust fusion; processing of notch/abnormal morphology and misalignment between leads.

**Version**: 2026-07 | **Positioning**: Engineering Implementation Guide + Literature Review

---

<a id="0-执行摘要"></a>
## 0. Executive Summary

<a id="01-核心结论"></a>
### 0.1 Core Conclusion

1. **"Global point" and "Single lead point" must be defined and stored separately. ** Appendix FF.2 of IEC 60601-2-25 clearly stipulates: The global time course of P/QRS/T is defined from "earliest onset in any lead" to "latest offset in any lead", because the projection time of the exciting wave front on different leads is inherently different. CSE database record #001 is a prime example - the global QRS time course reference is 127 ms, while visual inspection on lead I is only about 100 ms, the difference coming from the significantly earlier onset in lead III. **So "lead-to-lead misalignment" is largely a physiological fact, not an error to be eliminated. **

2. **But the naive min/max global rule is extremely unstable**: a noisy lead can pull the global onset 30 ms earlier. In engineering, "**reliable lead screening + robust order statistics (not extreme values) + multi-lead mutual confirmation**" must be used instead of directly taking min/max.

3. **The accuracy ceiling is determined by three things**, but the algorithm itself is not the most important:
   - Pre-processing (high-pass cutoff frequency, zero phase or not, equipotential line estimation)
   - Temporal resolution (500 Hz sampling = 2 ms/sample, whereas pathological Q-wave threshold is at 30/40 ms and must be upsampled)
   - Build quality of representative/median beat

4. **The current best practice is hybrid**: deep learning segmentation provides robust coarse positioning and waveform semantics (which segment is P/QRS/T, and whether it exists), traditional signal processing (wavelet mode maximum/derivative threshold/area method) performs **sub-sampling level boundary refinement** within the coarse positioning window, multi-lead robust statistics make the final decision, and the rule engine does rationality verification and confidence output. The boundary variance of pure DL is usually larger than that of the refined hybrid method, and the pure rule method has insufficient recall under arrhythmia and noise.

5. **Among the two routes of multi-lead fusion, "projection/dimensionality reduction and then delineate" is better than "each lead is delineated separately and then the rules are selected"**. Almeida et al. (IEEE TBME 2009) concluded that the boundary error dispersion of the optimal derived lead based on the spatial projection of the wavelet transform loop (WT loop) is lower than that of any single lead, and it is also better than making a selection rule based on the single lead annotation result. The selection rule method can only reduce the error dispersion by about 15%, and the multi-lead projection method can reduce it by about 40%.

<a id="02-推荐流水线一览"></a>
### 0.2 Recommended pipeline (at a glance)

```
原始 12 导联
   │
   ├─[1] 前处理：去工频 → 基线（零相位高通 0.05 Hz 或三次样条/中值）→ 上采样至 1000–2000 Hz
   │
   ├─[2] 逐导联 SQI：kSQI / basSQI / pSQI / 饱和 / 平坦 / 电极脱落 → 导联可靠度 w_l ∈ [0,1]
   │        └─ 导联反接 / 电极错位检测（否则所有形态特征失真）
   │
   ├─[3] 心搏检测：各导联单导联检测 + 空间幅值(VCG magnitude)/PCA 通道检测 → 决策级融合投票
   │        └─ 跨导联事件配对（±50 ms 窗）→ beat table
   │
   ├─[4] 心搏聚类（形态模板）→ 选主导型 → 构建"时间相干"代表搏（各导联共用同一对齐基准）
   │
   ├─[5] 粗分割：DL 分割网络（逐样本 P/QRS/T/背景 + peak 类）在每个导联上运行
   │        └─ 单调序约束解码（DP/Viterbi），保证 Pon<Poff<QRSon<QRSoff<Ton<Toff
   │
   ├─[6] 边界精修（每导联，在粗窗口内）：
   │        QRS on/off → 小波模极大阈值法 + 导数/曲率复核
   │        T end      → 切线法 / 梯形面积法 / 小波，三法交叉验证
   │
   ├─[7] QRS 内部成分标注：Q/R/S/R'/S' 命名、notch/slur 检测、幅度（相对等电位线）
   │
   ├─[8] 多导联稳健融合：
   │        全局 onset  = 加权稳健分位数（不是 min）+ ≥N 导联互证
   │        全局 offset = 同上（不是 max）
   │        单导联量（Q 幅度/时程、R 幅度）保持逐导联，不融合
   │
   └─[9] QC：生理约束校验、导联间离散度 → 置信度分级 → 不可靠则降级/拒答
```

---

<a id="1-问题定义与术语"></a>
## 1. Problem definition and terminology

<a id="11-要测什么特征清单"></a>
### 1.1 What to measure (feature list)

| Category | Characteristics | Properties | Remarks |
|---|---|---|---|
| Time Base | Global QRS onset / offset | Global | IEC: earliest onset / latest offset |
| | Lead by lead QRS onset / offset | Lead by lead | For QRS time course dispersion, local morphology |
| | global T end | global | end of QT |
| R wave | R peak position, R amplitude, R duration | Lead by lead | Amplitude versus equipotential lines |
| | R' presence and position, R/S ratio | Lead by lead | rSR' morphological discrimination |
| | notch / slur position, depth, time | lead by lead | fQRS, QRS terminal notch |
| Q wave | Q peak position, Q amplitude (negative values), Q duration | Lead by lead | Q onset ≡ this lead QRS onset |
| | Q/R amplitude ratio, Q area | Lead by lead | Pathological Q criterion core |
| Segment/Interval | QRS Time course (global/lead-by-lead), QT, QTc, PR, ST Level and Slope | Mixed | QT with global definition |
| Repolarization morphology | T amplitude, T peak-T end, T symmetry, T area, TMD, TCRT, PCA ratio | Mixed | See §10.4 |
| Space | QRS electric axis, QRS-T included angle, VCG ring parameters | Global | Requires orthogonal lead transformation |

<a id="12-全局点-vs-单导联点最容易踩的坑"></a>
### 1.2 Global point vs single lead point (the easiest pitfall)

- **Single lead onset/offset**: The moment when the waveform leaves/returns to the isoelectric line on this lead signal.
- **Global onset/offset**: earliest onset/latest offset among all leads. ** is the official definition benchmark of QRS schedule and QT. **
- **Key corollary**: The global QRS duration ≥ any single lead QRS duration, and may be significantly longer (a difference of 20–30 ms is not uncommon).
- **IEC's I wave/K wave rules**: After the global QRS onset, or before the global QRS offset, the isoelectric segment (called I wave and K wave respectively) that appears on a certain lead is **counted into** the time course measurement of adjacent waves. The boundary cannot be closed in because "it is flat here".
- **What is measured is reported**: If the device only provides lead-by-lead intervals, strictly speaking, the IEC interval accuracy limit does not apply; conversely, a model trained with global definitions cannot directly use lead-by-lead annotations for evaluation (LUDB is a lead-by-lead annotation, and QTDB is an integrated annotation. Mixed use will introduce system bias).

<a id="13-精度目标"></a>
### 1.3 Accuracy Target

**Error standard deviation tolerance derived by the CSE Working Group based on inter-expert variation** (de facto standard in the industry, commonly cited by Martinez 2004 and others):

| datum point | σ tolerance (ms) |
|---|---|
| P onset | 10.2 |
| P offset | 12.7 |
| QRS onset | 6.5 |
| QRS offset | 11.6 |
| T offset | 30.6 |

**IEC 60601-2-25:2011 Table 201.105 (Global measurement, for CSE 100 records)**:

| Global measurement quantity | Acceptable mean difference (ms) | Acceptable standard deviation (ms) |
|---|---|---|
| P schedule | 10 | 15 |
| PQ interval | 10 | 10 |
| QRS schedule | 10 | 10 |
| QT interval | 25 | 30 |

**Interpretation**: QRS onset has the tightest tolerance (σ ≤ 6.5 ms), with only about 3 samples at 500 Hz; T offset has the loosest tolerance (σ ≤ 30.6 ms), but it is also the most difficult - this is the most difficult point recognized by the industry. Reference magnitude: ECGdeli on QTDB P/QRS mark median error < 4 samples (16 ms @250 Hz), T offset median error 7 samples; better wavelet method can achieve QRS onset SD ≈ 2.8 ms, QRS offset SD ≈ 4.3 ms, T offset SD ≈ 12.9 ms (on CSE).

---

<a id="2-数据与评测"></a>
## 2. Data and evaluation

<a id="21-可用数据库"></a>
### 2.1 Available databases

| Database | Scale | Leads | Annotation Type | Sampling Rate | Usage |
|---|---|---|---|---|---|
| **CSE Multilead (DS-3)** | 125 records (IEC designated 100-record subset, MO1_ series) | 12/15 | Global, referee consensus | 500 Hz | **Regulatory-level acceptance baseline**, available for a fee |
| **QT Database (QTDB)** | 105 records | 2 | Integrated, each record ≥30 beats manually annotated | 250 Hz | Academic standard benchmark; 11 records with double annotation can be calculated as inter-observer variation |
| **LUDB** | 200 subjects × 10 s | 12 | **Lead by Lead** P/QRS/T Boundary | 500 Hz | DL Training/Evaluation Main |
| **PTB-XL** | 21k+ records | 12 | No artificial boundary annotation (commercial algorithms such as Schiller ETM can be used to generate weak labels) | 500/100 Hz | Large-scale weakly supervised training |
| **INCART (St. Petersburg)** | 75 × 30 min | 12 | Beat level | 257 Hz | Detector robustness |
| **New libraries such as ISP / ICDIRS** | ICDIRS contains 156,000 QRS onset annotations | 12 | Lead by lead | — | Large-scale DL |
| **SemiSegECG (2025)** | Multi-database unification | Hybrid | Unified evaluation protocol | — | Semi-supervised/cross-domain generalization benchmark |

**Practical Suggestions**: Use LUDB + PTB-XL weak label + own label for training; use QTDB for parameter adjustment/selection; final acceptance (if it is a medical device) must use CSE and report according to IEC Table 201.105. Cross-library evaluation is necessary - one of the conclusions of SemiSegECG is that there is a significant gap between the model under in-domain and cross-domain.

<a id="22-评测指标"></a>
### 2.2 Evaluation indicators

**Detection layer (whether this wave is found)**
- Se (sensitivity), P+ (positive predictive value), F1; tolerance windows need to be clearly stated. The tolerance window in the literature varies from 10 ms to 320 ms (!). Common P waves are 80 ms and T waves are 160 ms. QRS detection uses 150 ms according to ANSI/AAMI EC57. **The window width must be stated when reporting, otherwise the figures will not be comparable. **

**Positioning layer (can’t find it accurately)**
- Mean m (bias) and standard deviation σ (dispersion) of signed errors – directly compared to CSE/IEC tolerances.
- Median absolute error versus IQR - more robust to outliers, recommended to be reported simultaneously.
- The segmentation task additionally reports IoU/Dice, but **don’t just report IoU**: IoU is insensitive to the offset of a few samples at the boundary, and it is these samples that are of clinical concern.

**Interval Tier (Final Delivery Volume)**
- Bland-Altman analysis of QRS time course, QT, PR (bias + 95% agreement limits).
- Repeatability: the dispersion of measurements between adjacent beats/adjacent recordings from the same subject (especially important for QT studies).

**Tierized Assessment (Highly Recommended)**
- Stratified by rhythm: sinus/atrial fibrillation/atrial flutter/ventricular tachycardia/premature ventricular/pacing. The DL model has an F1>99% on the standard benchmark, but may drop by 15 percentage points on rhythms that are underrepresented in the benchmark (such as various types of tachycardia).
- Stratified by QRS width: narrow QRS / BBB / pacing.
- Stratified by signal quality: binning with SQI.

---

<a id="3-前处理精度的真正瓶颈"></a>
## 3. Preprocessing: the real bottleneck of accuracy

> Rule of thumb: **If you choose the wrong filter, all subsequent algorithm efforts will be in vain. ** The σ tolerance of QRS onset is 6.5 ms, and an improper 0.5 Hz single-pole high-pass can introduce much larger deformation in the ST segment.

<a id="31-采样率与插值"></a>
### 3.1 Sampling rate and interpolation

- Diagnostic grade ECG: AHA/ACC/HRS recommends low pass at least 150 Hz (pediatric 250 Hz), sampling rate ≥500 Hz.
- **1 sample = 2 ms at 500 Hz**, while the pathological Q wave criterion is at 20/30/40 ms, QRS time course 110 vs 120 ms determines whether to report BBB. Quantification errors directly cross clinical thresholds.
- **Method**: Before boundary refinement, use band-limited interpolation (sinc/cubic spline) to upsample the window to be measured (~200 ms before and after QRS) to 1000–2000 Hz. Note that this does not increase the amount of information, but eliminates the quantization step and makes the estimation of the threshold crossing point continuous, which can significantly reduce σ in actual measurements.
- The R peak position can be estimated by **parabolic three-point fitting** for subsampling: `δ = 0.5(y[-1]-y[+1]) / (y[-1]-2y[0]+y[+1])`.

<a id="32-基线漂移"></a>
### 3.2 Baseline drift

| Method | Advantages | Risks |
|---|---|---|
| Single pole high pass 0.05 Hz | Comply with AHA 1975/1990 traditional recommendations, ST/QT deformation is negligible | Weak drift suppression capability |
| Linear zero-phase digital highpass, cutoff can be relaxed to ≤0.67 Hz | AHA 1990 / ANSI-AAMI allowed; zero phase no group delay | Must **true zero phase** (filtfilt), otherwise ST is depressed |
| Cubic spline fitting (with PQ segment as node) | Good adaptability to drift | Rely on PQ segment positioning, chicken and egg problem (needs to be processed twice) |
| Median filter cascade (200 ms + 600 ms window) | No phase problem, good protection for QRS | Window length needs to be adjusted with heart rate |
| Wavelet multi-scale reconstruction (removing the coarsest scale) | Shared transformation with subsequent wavelet delineation | Boundary effects |

**STRONGLY RECOMMENDED**: Process twice. The first pass of coarse filtering → detect QRS → locate the PQ/TP segment → the second pass uses spline/median fine to remove the baseline. ECGdeli uses an explicit "isoline correction" step (histogram mode estimation of equipotential levels).

**Absolutely prohibited**: Use different filters or different phase characteristics for different leads - this will artificially create time shifts between leads, directly contaminating the global onset/offset.

<a id="33-工频与肌电"></a>
### 3.3 Power frequency and electromyography

- Power frequency: Give priority to adaptive notch (LMS) or spectral line interpolation to avoid ringing tail of fixed IIR notch from contaminating the starting point of QRS.
- EMG/Broadband Noise: **Don't pull the low pass to 40 Hz just to look good**. The 40 Hz low-pass will cut off the high-frequency components of QRS, directly affecting the visibility of the notch and the Q-wave amplitude measurement. Research shows that the 40 Hz vs 150 Hz cutoff has a substantial impact on Q-wave measurement and notch identification. If noise reduction is necessary, use edge-preserving methods such as wavelet thresholding or non-local mean.
- Spikes/pacing pulses: detect separately and do interpolation replacement, do not let it enter the derivative/wavelet channel (will produce huge pseudo-mode maxima).

<a id="34-等电位线isoelectric-level估计"></a>
### 3.4 Estimation of isoelectric levels

All amplitude characteristics (R amplitude, Q amplitude, ST level) are **relative quantities**, and if you choose the wrong reference level, you are wrong all around.

Common solutions:
1. **PQ segment method**: Take the mean/median of the 10–30 ms window before the global QRS onset (commercial algorithms commonly use "the average level 16 ms before the earliest QRS onset"). Most commonly used.
2. **TP segment method**: between the end of T and the beginning of the next P. Not available when heart rate is high.
3. **Histogram mode method**: Take the amplitude histogram peak value for the entire signal. Residual sensitivity to baseline drift but independent of other reference points.
4. **Lead-by-lead vs global**: The amplitude must be **lead-by-lead** at their respective equipotential levels; the time base can be global.

---

<a id="4-导联可靠性评估与筛选"></a>
## 4. Lead reliability assessment and screening

<a id="41-单导联-sqi-家族"></a>
### 4.1 Single-lead SQI family

| Metrics | Definition | Detected Issues |
|---|---|---|
| **kSQI** | Signal kurtosis | Gaussian-type noise pollution (clean ECG has high kurtosis, noise reduces it); more robust to noise than skewness |
| **basSQI** | Baseline frequency band (<1 Hz) power proportion | Baseline drift/breathing |
| **pSQI / sSQI** | QRS band (approximately 5–15 Hz) power / full-band power | EMG, broadband noise |
| **bSQI / qSQI** | Consistency rate of heartbeat detection by two QRS detectors with different principles (such as Pan-Tompkins vs wqrs) | Comprehensive usability, one of the strongest single indicators |
| Saturation/clipping rate | Proportion of samples reaching ADC full scale | Amplifier saturation |
| Flatness | Continuous constant sample length | Electrode fall off, lead disconnection |
| Step/transient count | Number of large jumps | Poor contact |

**Warning**: Statistical SQI with fixed thresholds has poor stability across data sets (the performance of the same threshold fluctuates greatly on different libraries), and is often biased towards "clean signals". Therefore:
- Use **multi-index fusion** (fuzzy comprehensive evaluation/SVM/small CNN) instead of single-index hard threshold;
- The threshold is **adaptive** (relatively sorted based on the distribution of each lead in this record), not the absolute value;
- Outputs **continuous reliability w_l ∈ [0,1]** instead of binary good/bad.

<a id="42-跨导联一致性-sqi多导联特有价值最高"></a>
### 4.2 Cross-lead consistency SQI (unique to multi-leads, highest value)

Problems that cannot be seen with single-lead SQI can be seen with cross-lead SQI:

1. **Heartbeat consistency**: Using the fused heartbeat sequence as a reference, the R peak matching rate of lead l.
2. **Time dispersion**: The deviation of the R peak moment of lead l relative to the multi-lead median (residual MAD after removing the constant offset).
3. **Boundary Outlier**: The standardized residual z_l of the QRS onset relative to the robust center for lead l.
4. **Physical consistency check (12-lead unique strong constraints)**:
   - Einthoven's law: `II = I + III` (should be approximately true sample by sample)
   - Enhanced leads: `aVR = -(I+II)/2`, `aVL = (I-III)/2`, `aVF = (II+III)/2`
   - **Abnormal residual energy → Reverse/wrong connection of limb lead electrodes**. This is a must-check: LA/RA reverse connection will invert lead I, completely distort the Q and R wave morphology, and make any downstream characteristics untrustworthy.
   - Chest leads have no linear constraints, but adjacent lead correlations (V1–V6 adjacent correlations should change smoothly) can be used to identify V electrode misplacements/interchanges.
5. **VCG reconstruction residual**: Use Kors/inverse-Dower transformation to find XYZ, and then back-project back to 8 independent leads. Leads with large residuals are suspicious (because 12 leads essentially only have 8 independent channels, and redundancy can be used for self-test).

<a id="43-从选一条到加权"></a>
### 4.3 From "select one" to "weighted"

There are two classic routes for multi-lead QRS detection: (1) First select the best lead according to quality and then use the single-lead algorithm; (2) Each lead is detected separately and then fused. Route (1) is sensitive to noise distribution and the strategy needs to be rewritten when the number of leads changes; route (2) the fusion logic also needs to be changed when the number of leads changes.

**Recommended practices (avoiding the shortcomings of both)**:
- Instead of making a hard choice, use w_l as the **weight** throughout the entire process;
- Set a lower limit for exclusion of w_l (for example, w_l < 0.2 is directly eliminated to prevent extremely poor leads from being affected by weight residuals);
- The fusion logic is written in a form (quantile, weighted median) that is insensitive to the number of leads L, instead of hard-coding "7 out of 12".

---

<a id="5-心搏检测与跨导联配准"></a>
## 5. Heartbeat detection and cross-lead registration

<a id="51-单导联-qrs-检测方法"></a>
### 5.1 Single-lead QRS detection method

| Method Family | Representative | Characteristics |
|---|---|---|
| Bandpass + Differential + Square + Integral | Pan-Tompkins (1985), Hamilton-Tompkins | De facto standard, easy to implement, requires good adaptive threshold and refractory period logic |
| Length/slope transformation | wqrs (Zong), RS-slope | Robust to baseline drift; poor performance to paced rhythms |
| Wavelet mode maximum | Li 1995, Martinez 2004 (wavedet) | Multi-scale noise immunity, natural connection boundary positioning |
| Filter banks | Afonso 1999 | Multi-subband parallel decision making |
| Envelope/Energy | Hilbert envelope, Shannon energy, phasor transformation | Insensitive to morphological changes |
| Empirical Modes/VMD | EMD/VMD + Threshold | Adaptive but computationally heavy, mode aliasing |
| Deep Learning | 1D CNN/U-Net/CRNN | Generalizes well, but attention should be paid to latency and computing power |

**Actual test experience**: On high-quality signals, the F1 of mainstream algorithms is >99%. The difference appears in **low-quality signals** (F1 may fall below 80%) and **pacing heart rhythm** (most algorithms can maintain >94%, but some slope methods will collapse to 79%). So the selection depends on your target scenario rather than the average score.

<a id="52-多导联融合的三个层次"></a>
### 5.2 Three levels of multi-lead fusion

**(a) Signal level fusion (synthesize first, then detect)**
- **Spatial Magnitude/VCG Magnitude**: `M(n) = sqrt(x²+y²+z²)`, where XYZ is obtained from 12 leads by Kors/inverse-Dower transform. The physical meaning is clear and has nothing to do with the lead direction.
- **RMS Channel**: `R(n) = sqrt(Σ_l w_l · s_l(n)²)`. Multi-lead RMS wavelet delineation was demonstrated on QTDB to the best single-lead level (with an error within one sample of the CSE tolerance).
- **PCA first principal component**: data-driven, optimal SNR, but the direction of the principal component drifts with the heartbeat, and the polarity/scale is unstable, so a fixed sign convention is required.
- **Optimization of weighted quadratic projection**: Limiting the projection functional to a time-invariant quadratic type and solving the weight based on the "minimum beat-to-beat variation" criterion can further reduce variation (compared to uniform weighting, variation is further reduced in most cases).
- ⚠️ **The cost of signal-level fusion**: the synthesized signal loses the information of "which lead is the earliest/latest" and **cannot be directly used for global onset/offset**. It is suitable for heartbeat detection and coarse positioning, but not suitable for replacing global boundary determination.

**(b) Decision-level fusion (voting after detection of each lead)**
- Simple majority voting: ≥L/2 leads are detected.
- Chair-Varshney optimal fusion: weighted by the prior detection rate/false alarm rate of each lead, theoretically optimal.
- Reliability weighted voting: weighted by w_l of §4.
- OR/AND rule: OR improves sensitivity, AND improves accuracy. In practice, "OR detection + subsequent consistency pruning" is commonly used.

**(c) Feature-level fusion**: See §11.

**Recommended combination**: Use the spatial amplitude channel of (a) + the weighted voting of (b) for **dual-channel mutual verification** - the spatial amplitude channel is guaranteed to be leak-free (especially when QRS on some leads is close to equipotential, such as the electrical axis is perpendicular to the lead axis), and the voting channel is guaranteed to not cause false alarms.

<a id="53-跨导联事件配对"></a>
### 5.3 Cross-lead event pairing

The number of R peaks detected in each lead is different (missed detection/false alarm), and the corresponding relationship of "same heartbeat" needs to be established:

```
输入：各导联 R 峰时刻列表 {R_l(i)}
1. 取参考序列 R_ref（空间幅值通道，或票数最多的簇中心）
2. 对每个参考时刻 t_k，在 [t_k - Δ, t_k + Δ] 内收集各导联的候选（Δ ≈ 40–60 ms）
3. 一对多冲突 → 取最近；一对零 → 记为该导联在此搏缺失
4. 未被任何 t_k 吸收的检出 → 候选新心搏，需票数 ≥ N_min 才立案
5. 输出 beat table：[beat_id, lead, R_time, matched?]
```

The choice of Δ: to cover the physiological differences between leads (the R peak moment in QRS can differ by 10–40 ms on different leads, and the width QRS is larger), but it cannot be so large as to incorporate adjacent heart beats. It is recommended that Δ = min(60 ms, 0.3·RR).

<a id="54-导联间不对齐来源辨析与处理关键"></a>
### 5.4 "Misalignment" between leads: source identification and processing (key)

This is one of the core issues of this survey. You must first identify the source and then decide how to deal with it:

| Source | Characteristics | Processing |
|---|---|---|
| **Physiological** (the projection direction of the exciting wave front is different) | Changes with the heart beat shape, related to the electrical axis; larger when the width is QRS/BBB | **Absolutely cannot be "corrected"** - it is the physical basis of the global QRS time course |
| **ADC channel time division multiplexing skew** | Each channel has a fixed tiny delay (μs~ms level), which is related to the acquisition hardware design | Correctable: constant delay, compensated with known hardware parameters or calibration signals |
| **Filtering of each lead is inconsistent** | Fixed group delay difference | Should be avoided from the source; if it occurs, redo it with the same zero-phase filter |
| **Asynchronous acquisition** (such as channel splicing from different devices/different time periods, Holter blocking) | Large and possibly drifting time shifts | Must be corrected, aligned with cross-correlation |
| **Exported leads/reconstructed leads** | Introduced by linear transformation, no additional delay but the shape is smoothed | Label the source and reduce its weight in the fusion |

**How to identify and estimate**:

```python
# 伪代码：分离常量器械时移与生理性差异
for lead l:
    for beat k:
        # 在带通(5–30Hz)信号上，以空间幅值通道为参照做互相关
        tau[l,k] = argmax_tau  xcorr(s_l[beat k window], M[beat k window], tau)
    tau_const[l] = median_k(tau[l,k])       # 常量分量 → 疑似器械/异步
    tau_var[l]   = MAD_k(tau[l,k])          # 波动分量 → 生理 + 噪声

# 判定规则
if |tau_const[l]| > 1 sample and tau_var[l] < small:
    → 常量偏斜，可补偿
if tau_var[l] large and correlates with morphology/axis:
    → 生理性，保留
```

**Important reminder**: Even if there is real device deflection, the global onset/offset must be calculated after **deskewing**; and physiological differences must be preserved. Confusing the two to do "full-lead alignment" will systematically underestimate the QRS time course (this is the hidden reason why many self-developed algorithms fail to meet the standards on CSE).

---

<a id="6-代表搏representative-median-beat构建"></a>
## 6. Representative / Median Beat construction

<a id="61-为什么需要"></a>
### 6.1 Why is it necessary

Commercial 12-lead diagnostic algorithms (GE 12SL, Glasgow, Schiller ETM, etc.) almost all make global measurements on representative beats for the following reasons:
- The SNR is increased by about √N times, and the positioning variance of low slope points such as T end is significantly reduced;
- Eliminates boundary jumps caused by single-beat noise and significantly improves measurement repeatability;
- The "earliest/latest" rule for global onset/offset only makes sense on low-noise waveforms.

<a id="62-构建步骤"></a>
### 6.2 Build steps

```
1. 心搏聚类：以 QRS 模板相关系数 / DTW 距离聚类 → 分出主导型（dominant）、室性早搏、伪差搏
2. 排除：形态离群搏、RR 显著偏离搏、SQI 差的搏、噪声段所在搏
3. 对齐：关键——所有导联使用【同一个】对齐时刻
   - 用空间幅值通道或多导联互相关最大化求各搏的公共对齐点
   - 亚采样精度对齐（互相关抛物线插值），否则平均会产生低通效应，
     人为削平 notch、拉长 QRS
4. 合成：逐样本【中位数】优于均值——中位数天然剔除离群样本，噪声更低
   （已有大规模研究明确采用逐样本中位数，并报告其优于逐样本平均）
5. 输出：12 导联时间相干的代表搏（同一时间轴），并叠加到同一等电位基准
```

**"Temporal coherence" is the keyword**: Each lead must not be aligned independently - that will destroy the physiological time difference between the leads, and the global QRS time course will be systematically shortened.

<a id="63-逐搏-vs-代表搏"></a>
### 6.3 Zhubo vs representative fight

| | Representative fight | Zhu fight |
|---|---|---|
| Accuracy/Repeatability | High | Low |
| Applicable | Resting 12-lead diagnostic measurements, QT studies, regulatory acceptance | QT variability (QTV), T-wave alternans, beat-to-beat dynamics, arrhythmia analysis |
| Risk | Masking beat-to-beat variability; choosing the wrong dominant type can lead to a complete mistake | Noise dominance |

**It is recommended to output two sets at the same time** and clearly mark which set each feature is based on.

---

<a id="7-qrs-边界onset-offset检测方法"></a>
## 7. QRS boundary (onset/offset) detection method

<a id="71-方法总览"></a>
### 7.1 Method Overview

| Method | Principle | Advantages | Disadvantages |
|---|---|---|---|
| **Derivative threshold method** | Search \|dx/dt\| before/after the QRS peak and drop to a certain proportion of the peak | Simple and fast | The threshold needs to be adaptive; noise sensitive; slow start QRS late |
| **Wavelet Modulus Maximum Method** | Quadratic spline wavelet 2^1–2^5; boundary = outside the first/last significant mode maximum, \|WT\| drops to ξ·\|WT_peak\| or local minimum appears | Anti-noise, multi-scale natural processing wide/narrow QRS; most fully verified | ξ needs to be adjusted according to scale and wave type; wide QRS Required cutting dimensions |
| **Area/Trapezoid Method** | Maximize the geometric quantity (trapezoid area) within the candidate window | No empirical threshold, anti-broadband noise | Originally designed for T end, used for QRS and needs to be modified |
| **Curvature/Polygon Approximation** | Douglas-Peucker selects vertices and uses curvature extremes to determine boundaries | Candidate points are greatly reduced, interpretable, and sensitive to morphological changes | Vertex selection parameters are sensitive |
| **Template/Correlation Method** | Aligned to known template | Stable | Pathological morphology mismatch |
| **HSMM / Bayesian / Kalman** | Explicitly model band duration distribution and state transition | Naturally give posterior probability and order constraints | Training/inference is complex |
| **DL segmentation** | Sample-by-sample classification P/QRS/T/Background | Good generalization, no need to adjust threshold, can handle abnormal shapes | Large boundary variance, need post-processing, black box |

<a id="72-小波模极大法wavedet-系实现要点"></a>
### 7.2 Key points of implementation of wavelet modulus maximization method (wavedet system)

The most implemented solutions at present are worth elaborating on:

1. **Transformation**: quadratic spline wavelet (approximately "smoothed first derivative"), using the à trous algorithm to maintain the time resolution of each scale (no extraction). Scale 2^1…2^5. The energy of QRS is mainly at 2^1–2^4, and there is basically no QRS above 2^4; P/T is significant at 2^4–2^5.
2. **QRS Detection**: Find the pairwise mode maximum (positive and negative adjacent pairs, corresponding to steep rise/fall) in the 2^2 scale, and the zero-crossing point between the pairs is the wave peak.
3. **Component identification**: Multiple modular maximum pairs in QRS → multiple zero-crossing points → corresponding to Q, R, S, R'... (see §8.1 naming rules).
4. **onset**: Search forward from the **first** significant mode maximum `n_first` in QRS, and take the first of the following two:
   - `|WT(n)| < ξ_on · |WT(n_first)|` (threshold crossing)
   - Local minimum of `|WT(n)|` (slope turning)
5. **offset**: backward from the **last** significant module maximum `n_last`, the rules are symmetrical; ξ_off is usually greater than ξ_on (the end of QRS is often accompanied by J-point elevation/start of ST).
6. **Scale Adaptation**: Noisy or QRS is wide → use a coarser scale (2^3/2^4); signal is clean and QRS is narrow → use 2^1/2^2 for better time resolution. There are embedded implementations that are directly fixed at 2^4 (equivalent to a 3 dB passband of about 4–14 Hz), which has good real-time performance but slightly reduced accuracy.

**ξ Tuning**: Different implementations vary greatly (the QRS boundary is often in the order of 0.05–0.15). It must be calibrated according to the CSE tolerance on the target data, and the onset/offset and scale must be distinguished. **Do not copy the paper constants directly. **

<a id="73-多导联投影法全局边界的最强单项方案"></a>
### 7.3 Multi-lead projection method (the strongest single solution for global boundaries)

Multi-lead (ML) strategy by Almeida et al. (IEEE TBME 2009):

```
1. 从 3 个正交导联出发（12 导联经 Kors / inverse-Dower / PCA / Gram-Schmidt 得到 XYZ）
2. 在待定界的时间邻域内，计算 WT 域的空间"环"(loop)：WT_x, WT_y, WT_z 构成 3D 轨迹
3. 用 SVD/PCA 求该邻域内环的主方向 u
4. 把 WT 投影到 u 上，得到一条【为该边界点量身定制的】导出导联
5. 在这条导出导联上跑单导联 delineation 规则
```

**Why it works**: The electromagnetic vector in the neighborhood of each reference point has a dominant direction. Projection along this direction can maximize the signal-to-noise ratio and slope at the boundary, minimizing the positioning variance of the threshold crossing point.

**Actual test conclusion**:
- The boundary error dispersion is lower than that of any single lead, and is better than "each lead is delineated separately and then the selection rule is used";
- For T end jitter caused by breathing: the selection rule method reduces the error dispersion by about 15%, and the projection method reduces it by about 40%.

**Prices and Notes**:
- Orthogonal leads are required; when there are only 2 leads, vector projection/Gram-Schmidt/PCA can be used for approximate orthogonalization (special work has been done to study the generalization under 2–3 leads).
- **It gives the "boundary on the best derived lead", not the "boundary on the earliest lead"**. If the target is the IEC global definition, the projection method should be used as a **robust reference center** and combined with the relative earlyness and lateness of each lead to make a global decision (see §11.1c).

<a id="74-深度学习分割-后处理"></a>
### 7.4 Deep learning segmentation + post-processing

**Mainstream Architecture**
- **U-Net Series** (most commonly used): Sample-by-sample 4 categories (P/QRS/T/Background). The F1 of the working report P/T/QRS reaches 97.8% / 99.5% / 99.9% respectively. The number of parameters is small, and it generalizes well to different sampling rates and equipment. Each lead is processed independently and then integrated to improve the overall quality.
- **Peak Attention U-Net (2026)**: Expand the categories to 7 categories, **explicitly add three peak categories: P peak / R peak / T peak**. On LUDB, R peak F1 reaches 99.72, T peak 97.51, and P peak 89.26, which is significantly improved compared to the suboptimal model. The technique of "treating peaks as separate categories" is very valuable for this task.
- **CED-Net / CED-Res-Net / CED-LSTM-Net / CED-U-Net (2024)**: Directly output the three intervals P / QRS / QT on the **representative stroke** for global delineation. On CSE CED-Net achieved QRS duration −2.4 ± 5.4 ms, QT −0.7 ± 10.3 ms, P duration 2.6 ± 11.0 ms, PQ 0.9 ± 5.8 ms, **meeting all limits of IEC Table 201.105** – this is important evidence that the DL route has passed regulatory level acceptance. The same study also found that the pure U-Net architecture is less immune to noise (high frequency/low frequency/power frequency) than the other three.
- **Transformer / Semi-supervised**: SemiSegECG (2025) benchmark shows that Transformer outperforms convolutional networks in a semi-supervised setting; the benchmark provides both in-domain and cross-domain evaluation settings.
- **Variational Encoding-Decoding (ECGVEDNET)**: Specifically targeted at the problem of morphological variation, supporting the large-scale annotation library ICDIRS.

**Post-processing is required, not optional**
1. **Morphological Cleaning**: Delete too short segments (QRS segment < 40 ms 或 > 250 ms to determine falsehood) and fill small holes.
2. **Sequential constraint decoding**: Use DP/Viterbi on the category probability sequence to find the optimal path that satisfies the legal transfer (background→P→background→QRS→background/T→…), which naturally guarantees the correct order. Much more robust than after-the-fact rule patching.
3. **Boundary Refinement**: The argmax boundary of DL is jittered at the sample level. Take the **sub-sampling zero-crossing point** of the probability curve (the linear interpolation point where P(QRS) crosses 0.5), or treat the DL output** as a search window**, and use §7.2 wavelet legal final boundary within the window. In practice this step can reduce σ by 20–40%.
4. **Heart rhythm sensing gating**: Suppressing P wave output (using a lightweight rhythm classification head for gating) during atrial fibrillation/atrial flutter can significantly reduce P wave false alarms - there have been studies that specifically use "classification guidance strategies" to solve this problem.

**Hybrid Division of Labor Suggestions**

| Links | Using DL | Using signal processing |
|---|---|---|
| Is there such a wave (existence) | ✓ | |
| Approximately where (±20 ms) | ✓ | |
| Exact boundaries (±3 ms) | | ✓ |
| Abnormal morphological semantics (notch / fusion wave) | ✓ | ✓ (mutual verification) |
| Sequence and Physiological Rationality | ✓ (DP Decoding) | ✓ (Rules) |

---

<a id="8-r-波与-qrs-内部成分"></a>
## 8. R wave and internal components of QRS

<a id="81-成分命名规则必须先标准化否则r-波位置无定义"></a>
### 8.1 Component naming rules (must be standardized first, otherwise "R wave position" is not defined)

Standard ECG nomenclature:
- **Q**: The first negative wave in QRS, and **there is no positive wave in front**
- **R**: the first positive wave; **R'**: the second positive wave (with a negative wave in front)
- **S**: negative wave after R; **S'**: negative wave after R'
- **QS**: The entire QRS has only one negative wave (without any positive component)
- Capitalize according to amplitude: use uppercase letters for larger amplitudes (qRs / rSR', etc.). The threshold needs to be determined in engineering (commonly used is a certain ratio of QRS peak-to-peak value, or absolute 0.5 mV)

**Project Realization**
```
1. 在 [QRS_on, QRS_off] 内找所有相对等电位线的极值（一阶导过零 + 显著性筛选）
2. 显著性阈值：幅度 > max(α · QRS_pp, A_min)，典型 α = 0.05–0.1，A_min = 20–50 μV
   —— 太低会把噪声当成分，太高会漏掉浅 Q 波（而临床恰恰关心浅 Q）
3. 按时间顺序 + 极性 应用上述命名规则
4. 输出形态字符串（"qRs"、"rSR'"、"QS"）与各成分的 (时刻, 幅度, 时程)
```

**Ambiguity of "R-wave position" (must be distinguished at the interface layer)**
- `R_peak`: R-wave peak moment in the sense of nomenclature (**not necessarily** the absolute maximum point, such as S is larger in rS form)
- `QRS_dominant_peak`: The absolute maximum point in QRS (this is usually what the heartbeat detector outputs)
- `QRS_fiducial`: datum point used for alignment (often taken as energy center of gravity or maximum slope point)

The three overlap in normal beats, but are completely different in rS/QS/rSR' morphology. **A lot of bugs come from mixing the three. **

<a id="82-r-峰的亚采样定位"></a>
### 8.2 Subsampling positioning of R peak

- Parabolic three-point fitting (§3.1);
- or template cross-correlation + subsampling interpolation (more robust to beat-to-beat QT/HRV);
- When used with RR intervals, **consistency is more important than absolute accuracy** (systematic offsets cancel out in the difference).

<a id="83-notch-slur-fqrs-检测重点"></a>
### 8.3 notch / slur / fQRS detection (key points)

**clinical definition layer**
- fQRS of Das et al. (2006): narrow QRS (<120 ms）中出现额外 R 波（R'）、R 波顶部或 S 波底部的切迹、或 >1 R', and appear in **≥2 consecutive leads** corresponding to the same coronary blood supply area; typical bundle branch block and incomplete right bundle branch block are excluded. Wide QRS (>120 ms) has other criteria (such as >2 R's or >2 notches on S wave); pacing QRS (f-pQRS) is also otherwise defined.
- This definition has **weak reproducibility** (high intra/inter-interpretation variability), and subsequent work attempts to refine it: stratification by number of fragmented leads (Torigoe 2012), automatic classification of different fQRS patterns (Maheshwari 2013), differentiation of benign/malignant variants (Haukilahti), and proposals for a clearer classification based on morphology (Frontiers) 2016, pointed out that the original criterion did not set conditions for the notch amplitude and position, and proposed to quantify the amplitude difference at three points where the notch point/peak/peak is located).
- **Engineering Implication**: Don't just output the boolean `has_fQRS`. The **quantifiable notch descriptor** (number, time, depth, component, and lead group) should be output, leaving the clinical criteria to the upper-level configurable rules.

**Signal processing layer**

| Type | Characteristics | Detection method |
|---|---|---|
| **notch (cusp notch)** | The first-order derivative** has ** sign change (there is a real local extreme value pair) | First-order derivative zero-crossing detection + deep significance screening |
| **slur (stutter/slope)** | The first-order derivative** has no** sign change, only the slope decreases sharply | The local minimum of the first-order derivative (not zero); **The second-order derivative extreme value**; the curvature κ = \|x''\|/(1+x'²)^{3/2} extreme value; the maximum deviation from the straight line fitting (Douglas-Peucker residual) |

**Multi-scale method**: CWT has a strong response to notches at fine scales and a weak response to main waves, and can be used as an independent channel; the fine-scale energy peaks in the QRS interval are used as notch candidates, and then morphological quantification is performed.

**Automatic QRS terminal notch/slur rules for the Glasgow program** (published and verified, can be used for reference directly)
- Limited to QRS The last component is the R wave;
- R wave duration > 40 ms;
- The notch/contusion is located on the **descending limb** of the R wave;
- The distance between the starting point of the notch/contusion and the end point of QRS is > 10 ms;
- Amplitude criterion: notch peak/stutter starting point amplitude ≥ 0.1 mV;
- Need to appear in ≥2 consecutive leads (lead combination rules are clear, such as I and aVL and (V4 and V5 or V5 and V6)).
- Reported performance: training set Se 92.1% / Sp 96.6%, test set Se 90.5% / Sp 96.5%, overall accuracy 95.7%, precision 84.9%.

**Machine learning route**: The fQRS automatic quantification method has been verified by multiple centers (Sci Rep 2022). The process is also "first use **multi-lead information** to do robust QRS segmentation and automatically eliminate abnormal beats, and then do notch quantification" - once again confirming the value of multi-lead mutual confirmation.

**Specific usage of multi-lead mutual confirmation on notch**
1. True notches are **spatially continuous**: adjacent leads (V4/V5/V6, or II/III/aVF) should present notches at similar times, because they are projections of the same electrical activity vector in similar directions.
2. **VCG verification**: Map the notch moment to the 3D ring. The true notch corresponds to the direction mutation/stop point on the ring, and there will be no noise.
3. **Removal**: If the notch only appears in one lead and the SQI of this lead is low → it is judged as noise; if it only appears in one high SQI lead and this lead is the best projection of the vector direction → it may be true ("isolated notch", marked as low confidence rather than discarded directly).

**Related but different concepts (worth exporting together)**
- **RAZ (reduced amplitude zone)/HF-QRS**: QRS The amplitude notch in the envelope of the high-frequency component (about 150–250 Hz) is more sensitive than visual notch, but requires a high sampling rate and high bandwidth (this is one of the reasons not to use a 40 Hz low pass).
- **QRS Continuous quantity of fragmentation**: QRS internal first-order derivative zero-crossing times, QRS spectral entropy, wavelet fine-scale energy proportion - more suitable for statistical modeling than binary fQRS.

<a id="84-幅度与形态特征清单逐导联"></a>
### 8.4 List of amplitude and morphological characteristics (lead by lead)

```
R_amplitude   = s(R_peak) − isoelectric_level        # 相对等电位线，不是绝对值
R_duration    = 该 R 成分的起止（相邻过零 / 回到等电位线）
Q_amplitude   = isoelectric_level − s(Q_peak)        # 报为正的"深度"或保留负号，接口需固定约定
QRS_pp        = max − min（QRS 区间内）
R/S、R/Q、Q/R 幅度比
QRS_area, Q_area = 对 (s − isoelectric) 在对应区间积分   # 比峰值更抗噪
VAT / intrinsicoid deflection = QRS_onset → R_peak    # 左室激动时间
notch_count, notch_times, notch_depths
QRS_slope_upstroke / downstroke                      # 最大上升/下降斜率，缺血标志物
```

> **Recommendation**: Prioritize **area-based** features over peak-based features in statistical modeling—area is significantly more robust to noise and single-point outliers. Stanford's computerized athlete criteria use percentiles of **Q wave area** rather than amplitude thresholds.

---

<a id="9-q-波专项"></a>
## 9. Q Wave Specifics

<a id="91-定义与边界"></a>
### 9.1 Definition and Boundaries

- **Q onset ≡ QRS onset of that lead** (Q is the first component of QRS, coinciding by definition). Therefore, **the accuracy of Q duration depends entirely on the accuracy of QRS onset**—this is precisely why CSE has the tightest tolerance for QRS onset (σ ≤ 6.5 ms).
- **Q offset = the moment the Q wave returns to the isoelectric line and begins to rise**; in implementation, this is taken as the interpolated time when the signal **first crosses upward** `isoelectric + ε` after the Q peak (ε is set to 2–3 times the noise RMS).
- **QS morphology**: If there is no R wave, then Q offset = QRS offset, and Q duration = QRS duration; in criteria, QS and deep Q are usually treated equally.
- **Note**: Residual baseline drift causes systematic bias in the "return to isoelectric line" criterion, which is the largest source of error in Q duration measurement; §3.4 must be performed first.

<a id="92-数值实现细节"></a>
### 9.2 Numerical Implementation Details

1. **Upsampling is mandatory**: At 500 Hz, Q duration can only be an integer multiple of 2 ms, while clinical thresholds are 20 / 30 / 40 ms. Near the 20 ms threshold, quantization error is sufficient to flip the classification. Upsample to ≥1000 Hz before boundary refinement.
2. **Amplitude reference**: Q amplitude must be relative to the **isoelectric level of that lead itself**; global or other leads' levels cannot be used.
3. **Existence determination of shallow Q waves**: Set a minimum amplitude threshold (e.g., 30–50 μV or 5% of QRS_pp); below this threshold, record as "no Q wave" rather than "Q = 0". This threshold directly determines the detection rate of pathological Q waves and must be calibrated on a validation set with sensitivity analysis.
4. **Directional bias of noise**: Noise causes "first upward crossing" to trigger early → Q duration is systematically shortened. Using representative beats (§6) can significantly mitigate this.
5. **Interaction with notches**: If there is a notch within the Q wave, "first upward crossing" may prematurely terminate the Q wave. The significance threshold from §8.1 must be used to distinguish "true R wave onset" from "notch within Q".

<a id="93-临床判据做成可配置项不要硬编码"></a>
### 9.3 Clinical Criteria (Make configurable, do not hardcode)

| Criterion | Content |
|---|---|
| Classic criterion | Q duration > 40 ms **and** Q amplitude > 25% of the corresponding R wave |
| Universal Myocardial Infarction Definition (UDMI) | Any Q wave or QS ≥20 ms in V2–V3; or Q wave or QS ≥30 ms and ≥0.1 mV deep in I, II, aVL, aVF, V4–V6, appearing in **any two leads of the same continuous lead group**; Posterior wall: R ≥40 ms and R/S ≥1 in V1–V2 with upright T waves (in the absence of conduction block) |
| Continuous lead groups | (I, aVL, V6), (V4–V6), (II, III, aVF) |
| Athlete screening | Early Seattle: >3 mm deep or >40 ms, ≥2 leads (excluding III, aVR); Seattle-2 replaced absolute amplitude with **Q/R amplitude ratio**, resulting in significantly fewer positive cases |
| Normal variants (to be excluded) | Q in lead III <30 ms and <25% of R when axis is 30°–0°; Q in aVL when axis is 60°–90°; septal q waves in I, aVL, aVF, V4–V6 (<30 ms and <25% of R) |

**Engineering Recommendation**: The algorithm should only output **measurement-level** results `(lead, Q_present, Q_duration_ms, Q_amplitude_uV, Q_R_ratio, Q_area, QS_pattern, confidence)`; the criteria layer should be an independent, switchable rule module. Studies show that different criteria identify significantly different populations; hardcoding criteria into the algorithm prevents the system from adapting to different clinical scenarios.

<a id="94-多导联在-q-波上的作用"></a>
### 9.4 Role of Multiple Leads in Q Waves

- **The "continuous leads" requirement itself is multi-lead corroboration**: Isolated single-lead Q waves are often artifacts or normal variants.
- **aVR and III are usually excluded** (physiological Q waves are common).
- **Precordial electrode position**: "Pseudo-septal infarction" Q waves in V1–V2 are often due to electrodes placed too high. This can be combined with P wave morphology (abnormal terminal negative P in V1) or compared with previous ECG to assist in discrimination—pure algorithms struggle to completely resolve this, but it is worth outputting a warning flag.
- **Lead reversal must be excluded first** (§4.2): LA/RA reversal creates non-existent Q waves in I and aVL.

---

<a id="10-qt-与复极相关特征"></a>
## 10. QT and Repolarization-Related Features

<a id="101-t-波终点t-end全流程最难点"></a>
### 10.1 T Wave End (T end) — The Most Difficult Part of the Entire Process

| Method | Principle | Performance |
|---|---|---|
| **Threshold method** | Point where T returns to isoelectric line, or first derivative drops to a certain proportion of the peak | Most common; sensitive to baseline drift and noise; threshold is highly empirical |
| **Tangent method (Lepeschkin)** | Intersection of the tangent at the steepest point of the T descending limb with the baseline | Low reader variability, easy to teach (teaching research shows students using the tangent method have higher accuracy than some experts); but **systematically earlier** than the true end, and has large bias for asymmetric T waves |
| **Trapezium area method** | Search for the point on the descending limb that maximizes the trapezoidal area | **No empirical threshold**; superior accuracy and repeatability compared to first-derivative threshold method under broadband noise |
| **Wavelet modulus maxima method** | Same framework as QRS, taking the point after the last modulus maximum where \|WT\| < ξ | One of the SOTA on QTDB, with particularly outstanding performance for T end |
| **Model fitting** | Extrapolation after Gaussian / Hermite / polynomial fitting | Smooth and noise-resistant; large bias when model mismatch occurs |
| **DL segmentation** | Directly output T segment | Robust but high boundary variance (multiple studies report T offset σ > 30 ms on QTDB) |

**Reference baseline for reader variability**: Central ECG laboratory studies show that the mean absolute intra-reader variability for the tangent and threshold methods is approximately 3.4–6.9 ms (tangent) / 3.5–5.2 ms (threshold), with intra-reader SD on the order of 7 ms. **An algorithm achieving σ ≈ 10–15 ms is already excellent**; results claiming σ < 5 ms require scrutiny of their evaluation protocol.

**Engineering Recommendation**: **Run 3 methods simultaneously (tangent + trapezium + wavelet), take the robust median, and use the dispersion of the three as a confidence indicator.** Consistency among the three methods → high confidence; divergence > 20 ms → mark as unreliable (likely T-U fusion or bifid T wave).

<a id="102-全局-qt"></a>
### 10.2 Global QT

```
QT_global = latest_T_offset(over reliable leads) − earliest_QRS_onset(over reliable leads)
```
Likewise, naive max/min cannot be used (see §11.1). ICH E14 studies also typically designate a primary link (II or V5) for human review.

<a id="103-qtc-与-rr-修正"></a>
### 10.3 QTc and RR correction

| Formula | Expression | Remarks |
|---|---|---|
| Bazett | QT/√RR | Most commonly used; overcorrection at fast heart rates |
| Fridericia | QT/∛RR | One of the top choices for drug research |
| Framingham | QT + 0.154(1−RR) | Linear |
| Hodges | QT + 1.75(HR−60) | |
| Individualization | Fitting individual QT–RR relationships | For TQT research, a large amount of data is required |

**Hysteresis effect (QT hysteresis)**: QT's adaptation to changes in RR has a minute-level time constant. **You cannot just use the current RR**, you should use the exponentially weighted historical RR (time constant ~30–120 s). This is the main artificial source of QTc fluctuations in dynamic records.

<a id="104-t-波形态特征"></a>
### 10.4 T wave morphological characteristics

| Characteristics | Definition | Description |
|---|---|---|
| T amplitude, T area | Relative equipotential lines | Lead by lead |
| **Tpeak–Tend (Tp-e)** | T peak to T end | Transwall repolarization dispersion proxy |
| T symmetry | Ascending limb/descending limb slope ratio or area ratio | Ischemia sensitivity |
| **TMD** | The mean value of the angles between the T wave morphological vectors of each lead (SVD basis) | Multiple leads required |
| **TCRT** | Cosine of the angle between QRS principal vector and T principal vector | VCG/orthogonal lead required |
| **PCA ratio** | T-wave matrix SVD second/first singular value ratio | Complex polar non-two-dimensionality measure |
| **QRS-T angle** | Space/frontal plane | Strong prognostic value |
| **TWA** | Beat-to-beat alternating amplitude (spectral method/MMA) | Extremely high alignment accuracy requirements |
| **QT Dispersion (QTD)** | The range/SD of QT in each lead | ⚠️ The academic community has doubts about its value, which largely reflects the measurement error and T-ring projection effect, and is not recommended as the main indicator |

<a id="105-u-波与-t-u-融合"></a>
### 10.5 U wave and T-U fusion

- The U wave will cause the threshold method to push T end back, and the tangent method may ignore it → The difference between the two methods itself is a signal for the existence of the U wave.
- Detection: Secondary forward deflection approximately 200 ms after T end; significant in hypokalemia, long QT, bradycardia.
- **Processing**: Explicitly output `TU_fusion_flag`; reduce the weight of relevant leads in the T end decision during fusion, or give priority to using leads with inconspicuous U waves (U waves are most obvious in V2–V4, and limb leads are relatively clean).

---

<a id="11-多导联稳健融合的方法学"></a>
## 11. Methodology for robust multi-lead fusion

<a id="111-为什么不能直接-minmax核心问题"></a>
### 11.1 Why can’t min/max be used directly (core issue)

The IEC definition is min/max, but the definition is for "correct labeling". Once the onset of a lead is triggered early by noise, min is contaminated. Stabilization plan:

**(a) Weighted Robust Quantile**
```
1. 剔除 w_l < w_min 的导联
2. 稳健中心 c = weighted_median(onset_l)，尺度 s = 1.4826 · MAD
3. 剔除 |onset_l − c| > k·s 的离群导联（k ≈ 3）
4. 在剩余导联中取 min（或第 5–10 百分位）作为全局 onset
```
It respects the "earliest" semantics of the IEC without being dominated by single-point outliers.

**(b) Corroboration**
```
全局 onset := 满足"至少有 N_min 条可靠导联的 onset 落在 [t, t+δ] 内"的最小 t
典型 N_min = 2，δ = 6–10 ms
```
Physical rationale: True early excitement is projected onto multiple leads simultaneously (even if the amplitude is small), while noise is not synchronized across leads.

**(c) Projection method as reference center (recommended combination)**
```
1. 用 §7.3 的 WT-loop 投影法得到高稳健性参考边界 t_ML（低方差，但不是"最早"）
2. 只在 [t_ML − 40 ms, t_ML + 40 ms] 内接受各导联的 onset 候选
3. 在此约束下应用 (a) + (b) 得到全局 onset
```
Combine the advantages of "robust but possibly late" with "definitely but fragile".

**(d) M-estimation / RANSAC**: Treat each lead boundary as an outlier observation, and use Huber or Tukey loss to make M-estimation; or RANSAC sampling consistent set. It is suitable for scenarios with a large number of leads and a high proportion of outliers.

<a id="112-融合层次对比"></a>
### 11.2 Fusion level comparison

| Level | Practice | Applicable | Not applicable |
|---|---|---|---|
| Signal level | RMS / spatial amplitude / PCA / WT-loop projection | Beat detection, coarse localization, T end robust estimation, beat-to-beat variation | "Earliest/latest" semantics for global onset/offset |
| Feature level | Robust statistics for individual lead boundaries | Global boundary adjudication, QRS time course | When signal-to-noise ratio is extremely low for all leads |
| Decision-making level | Voting / Chair-Varshney optimal fusion / D-S evidence theory | Existence judgment (with or without Q wave, presence or absence of notch, presence or absence of P wave) | Continuous quantity estimation |
| Learning level | 12-channel joint input/lead attention/graph network | End-to-end; lead missingness robustness | Interpretability, regulatory justification |

**Reemphasize Almeida's conclusion**: On the specific task of boundary location, **Signal level projection > Feature level selection rules > Single lead**. However, due to the particularity of global definition, the final system needs to use both.

<a id="113-顺序与生理约束"></a>
### 11.3 Sequence and physiological constraints

Made into hard constraints (infeasible transitions in DP decoding) or soft penalties:

```
P_on < P_off ≤ QRS_on < QRS_off ≤ T_on < T_off < 下一个 P_on
40 ms  ≤ QRS 时程 ≤ 200 ms（起搏/BBB 放宽到 250 ms）
80 ms  ≤ PR ≤ 400 ms（一度 AVB 可更长，需可配置）
200 ms ≤ QT ≤ 700 ms
Q 时程 ≤ QRS 时程
QRS_on(global) ≤ min_l QRS_on(l)     # 自洽性检查
QRS_off(global) ≥ max_l QRS_off(l)   # 自洽性检查
```
**Do not fix silently** when a constraint is violated, log `constraint_violation` and lower the confidence level.

<a id="114-置信度与不确定度"></a>
### 11.4 Confidence and Uncertainty

Must be output (required by both downstream models and clinical users):

1. **Inter-Lead Dispersion** `σ_inter = MAD(onset_l over reliable leads)` - The most direct and cheapest uncertainty proxy.
2. **Inter-method dispersion**: The dispersion of results of different algorithms (wavelet/derivative/tangent/DL) on the same lead.
3. **Beat-to-beat dispersion**: The dispersion across beats in the same lead (in beat-to-beat mode).
4. **Model uncertainty**: DL uses deep integration or MC dropout; existing work has made the uncertainty of MC dropout into a "reliability heat map" for interpretability.
5. **Comprehensive Confidence Rating**
   ```
   A 级：三类离散度均小 + SQI 高 + 无约束违反 → 可直接用于诊断测量
   B 级：中等 → 可用于趋势/统计，需人工抽检
   C 级：任一项超限 → 拒绝输出数值，只报"不可测"
   ```
   **It is better to refuse to answer than to give the wrong number** - This is the most important difference between the medical measurement system and the general ML system.


---

<a id="12-困难情形处理清单"></a>
## 12. Difficult Situation Handling Checklist

| Situation | Phenomenon | Countermeasures |
|---|---|---|
| **Wide QRS / BBB** | QRS 200 ms+, multi-notch, QRS is difficult to distinguish from T | Wavelet scale is cut to 2^3/2^4; time course constraints are relaxed; fQRS criterion is cut to wide QRS Version; T end mainly uses the tangent method (the threshold method is easily interfered by QRS terminal tailing) |
| **Pace Rhythm** | The pacing nail amplitude is large and the rising edge is extremely steep | Perform pacing pulse detection first (requires high sampling rate, ideally ≥2 kHz) and perform interpolation removal; QRS onset should be set at **ventricular depolarization onset** rather than the nail; some detectors (slope type) collapse under pacing, and need to be replaced by energy/envelope type |
| **PVC / fusion stroke** | The morphology is completely different from the dominant type | Process separately after clustering, **do not** mix in representative strokes; output morphological characteristics separately for PVC |
| **Atrial fibrillation/atrial flutter** | P wave does not exist / f wave is misjudged as P | Rhythm classification head gates P wave output; DL model has significant P wave false alarm on AF/AFL (some studies report that F1 dropped 15 percentage points) |
| **Low amplitude QRS (<0.5 mV)** | Low SNR, unreliable boundaries | Increase SQI threshold; prioritize spatial amplitude channels; automatically degrade confidence level |
| **V1's rSr' (normal variation)** | Easily misjudged as fQRS | Add r' amplitude and time course conditions; require ≥2 continuous leads; combined with QRS time course to determine whether it is incomplete RBBB |
| **T-U fusion, bimodal T** | T end The three methods are very different | See §10.5; output flag and downgrade |
| **T wave inversion/bidirectional** | Wrong peak polarity judgment | DL segmentation + peak category (Peak Attention U-Net idea) is more stable than the rule method; the wavelet method itself is not sensitive to polarity (use \|WT\|) |
| **Noisy Lead/Electrode Off** | Single lead completely unusable | SQI culling; ensures fusion logic is insensitive to L changes |
| **limb lead reverse connection** | I inversion, false Q wave | Einthoven/aVR consistency check (§4.2), after detection, refuse to output morphological characteristics and alarm |
| **Chest lead exchange/misalignment** | R-wave incremental abnormality | Adjacent lead correlation smoothness test; VCG reconstruction residual |
| **Extremely fast heart rate (>150 bpm)** | P is buried in T, T end is truncated by the next QRS | Clearly output "unmeasured"; do not force extrapolation |
| **Export/Reconstruct Leads** | Obtained from linear transformation, the shape is smoothed | Mark the source, reduce the weight, and not be used as the only basis for notch detection |

---

<a id="13-推荐工程实现方案"></a>
## 13. Recommended project implementation plan

<a id="131-分阶段落地路线"></a>
### 13.1 Phased implementation route

**Phase 1 (2–4 weeks): Baseline available**
- Pre-processing (zero-phase filtering + median baseline removal + upsampling)
- Pan-Tompkins + spatial amplitude channel dual detection, voting fusion
- Wavelet module maximum method to do QRS boundary + tangent method to do T end
- Basic SQI (kSQI + basSQI + flatness)
- Evaluated on QTDB, aligned to CSE tolerance

**Phase 2 (4–8 weeks): Robustization**
- Beat clustering + time-coherent representation of beats
- Weighted robust quantile fusion + mutual confirmation constraints
- Lead reverse connection/misalignment detection
- Q/R/S naming and notch/slur detection
- Complete confidence system

**Phase 3 (8–16 weeks): DL Enhancement**
- U-Net (including peak category) is trained on LUDB + own annotation
- DP/Viterbi sequential constraint decoding
- DL coarse positioning + signal processing refined hybrid pipeline
- Cross-database generalization evaluation (LUDB → QTDB → own data)

**Phase 4: Multi-Lead Projection Method**
- Kors/PCA orthogonalization → WT-loop projection → derived lead delineation
- Access as a Robust Reference Center §11.1(c)

<a id="132-关键参数默认值起点需在自有数据上标定"></a>
### 13.2 Default values of key parameters (starting point, needs to be calibrated on own data)

```yaml
preprocessing:
  powerline_notch: adaptive           # 50/60 Hz
  highpass: {type: zero_phase_butter, fc: 0.5, order: 2}  # 或 median cascade
  baseline: {method: median_cascade, windows_ms: [200, 600]}
  lowpass: {fc: 150}                  # 不要用 40！
  upsample_to_hz: 1000                # 边界精修前

sqi:
  window_s: 2.0
  weights: {kSQI: 0.25, basSQI: 0.25, pSQI: 0.25, bSQI: 0.25}
  lead_reject_below: 0.20

beat:
  match_window_ms: 50                 # min(60, 0.3*RR)
  min_votes: 3                        # 12 导联下
  cluster_corr_threshold: 0.90

delineation:
  wavelet: quadratic_spline
  scales: [1, 2, 3, 4, 5]
  qrs_scale_default: 2                # 噪声大时切到 3
  xi_qrs_onset: 0.06                  # 必须标定
  xi_qrs_offset: 0.12                 # 必须标定
  component_min_amp_uv: 30
  component_min_ratio: 0.05

fusion:
  outlier_k_mad: 3.0
  corroboration_n: 2
  corroboration_delta_ms: 8
  global_percentile: 0.05             # 用 5% 分位替代 min

qc:
  qrs_duration_ms: [40, 250]
  pr_ms: [80, 400]
  qt_ms: [200, 700]
  inter_lead_mad_warn_ms: 8
  inter_lead_mad_reject_ms: 15
```

<a id="133-输出数据结构建议"></a>
### 13.3 Suggestions for output data structure

```json
{
  "record_id": "...",
  "quality": {
    "lead_sqi": {"I": 0.92, "II": 0.95, "...": 0.0},
    "lead_reversal_suspected": false,
    "excluded_leads": ["V3"]
  },
  "beats": [{"id": 0, "r_time_ms": 412.0, "cluster": "dominant"}],
  "representative_beat": {
    "n_beats_used": 8,
    "global": {
      "qrs_onset_ms": 120.4, "qrs_offset_ms": 218.6,
      "t_offset_ms": 512.0,
      "qrs_duration_ms": 98.2, "qt_ms": 391.6, "qtcf_ms": 402.1,
      "confidence": "A",
      "inter_lead_mad_onset_ms": 3.2
    },
    "per_lead": {
      "V2": {
        "qrs_onset_ms": 124.0, "qrs_offset_ms": 216.0,
        "morphology": "qRs",
        "components": [
          {"type": "q", "peak_ms": 130.0, "amp_uv": -85, "duration_ms": 24.0},
          {"type": "R", "peak_ms": 148.0, "amp_uv": 1420, "duration_ms": 48.0},
          {"type": "s", "peak_ms": 200.0, "amp_uv": -310, "duration_ms": 30.0}
        ],
        "notches": [{"time_ms": 156.0, "depth_uv": 120, "type": "slur", "on": "R_downstroke"}],
        "isoelectric_uv": 12,
        "t_offset_ms": {"tangent": 508.0, "trapezium": 514.0, "wavelet": 512.0,
                        "consensus": 512.0, "spread_ms": 6.0}
      }
    }
  },
  "flags": ["TU_fusion_V3"],
  "measurement_derived": {
    "pathological_q_leads": ["V2", "V3"],
    "fqrs_leads": [],
    "criteria_version": "UDMI-3rd"
  }
}
```

**Design points**: The measurement layer and the criterion layer are separated; the global quantity and the lead-by-lead quantity are separated; each value has an uncertainty or consistency index.

---

<a id="14-验证与质控"></a>
## 14. Verification and Quality Control

<a id="141-验证矩阵"></a>
### 14.1 Verification Matrix

| Dimensions | Hierarchy |
|---|---|
| Database | QTDB / LUDB / CSE / Owned |
| Rhythm | Sinus / AF / AFL / Tachycardia / Pacing / PVC |
| QRS width | <100 / 100–120 / >120 ms |
| Signal quality | SQI tertiles |
| Sampling rate | 250 / 500 / 1000 Hz (downsampling robustness test) |
| Noise injection | High frequency / low frequency / power frequency / motion artifact (using NSTDB noise library) |

**Noise injection testing is important**: The CED series of studies found that the immunity of different architectures to noise is significantly different (most architectures have an average timing error of < 2.5 ms under noise, while pure U-Net is significantly worse). This is something you can’t see from F1 on paper.

<a id="142-必做的健全性检查"></a>
### 14.2 Required sanity checks

```
□ 全局 QRS 时程 ≥ 所有单导联 QRS 时程（自洽）
□ 同一记录内相邻心搏的测量离散度合理（不应有跳变）
□ 上采样前后结果一致（差异应 < 1 原始样本）
□ 导联顺序打乱后结果不变（融合逻辑无导联顺序依赖）
□ 随机剔除 1–3 条导联后全局量的变化幅度可接受（鲁棒性）
□ 反转信号极性后成分命名正确翻转（Q↔R 逻辑正确）
□ 时间反转的合成信号上算法不产生"合理"输出（防过拟合到统计先验）
```

<a id="143-标注策略若要自建数据集"></a>
### 14.3 Labeling strategy (if you want to build your own data set)

- **Double independent annotation + third person arbitration**, and retain dual annotation to quantify inter-observer variation (which is the ceiling of algorithm performance).
- Annotation tools should display **all leads overlay** and allow annotators to switch between global and single-lead views - single-lead annotation systematically underestimates global time courses.
- When labeling T end, **unify the method** (tangent or threshold) and write it down in the document; mixed use will introduce a systematic deviation of 10–20 ms.
- Prioritize marking ** to represent strokes** instead of random single strokes, which is more cost-effective.

---

<a id="15-开源工具与代码资源"></a>
## 15. Open source tools and code resources

| Tools | Language | Content | Notes |
|---|---|---|---|
| **ECGdeli** (KIT-IBT) | MATLAB, GPLv3 | Full 12-lead filtering + delineation with explicit equipotential line correction, multi-lead simultaneous annotation (`Annotate_ECG_Multi.m`) | QTDB on P/QRS Median error <4 Sample; **The most complete open source solution with multi-lead support**, can be used as baseline |
| **ecg-kit** (Demski & Llamedo) | MATLAB | Integrate **wavedet** (the original delineator of the Laguna/Martinez/Almeida group) | The first choice for reproducing the wavelet method |
| **ecgpuwave** | C (PhysioNet) | Classic delineator | Old baseline, the detection rate is lower than the previous two |
| **NeuroKit2** | Python | `ecg_delineate()` supports three methods of dwt/cwt/peak | fastest to get started; average accuracy, suitable for exploration |
| **BioSPPy / py-ecg-detectors / wfdb-python** | Python | QRS detector collection, WFDB read and write | detection layer |
| **Moskalenko et al. U-Net** | Python | DL segmentation reference implementation on LUDB | DL baseline |
| **SemiSegECG** | Python | Multi-library unification + semi-supervised evaluation framework (2025) | Cross-domain generalization evaluation |
| **LUDB / QTDB / PTB-XL / INCART** | — | PhysioNet public data | CSE required |

**Note**: The CSE database has extended annotations and new version software testing recommendations (Smíšek et al., MBEC 2017), which are worth checking before regulatory acceptance.

---

<a id="16-常见踩坑清单速查"></a>
## 16. List of common pitfalls (quick check)

1. ❌ Use 40 Hz low pass to "beautify" → the notch disappears and the Q amplitude is underestimated
2. ❌ Use 0.5 Hz high pass with non-zero phase → ST depression, T end offset
3. ❌ Use different filtering parameters for different leads → artificially create time shift between leads
4. ❌ Direct 500 Hz signal Q duration → 2 ms quantization across 20/30/40 ms threshold
5. ❌ Global onset directly takes min → one noise lead destroys everything
6. ❌ Each lead is independently aligned before synthesizing the representative beat → Global QRS time course is systematically short
7. ❌ Treat "the absolute maximum point within QRS" as the R peak → all wrong in rS/QS form
8. ❌ Use absolute values instead of relative equipotential lines for amplitude → All baseline drifts are taken into account
9. ❌ Only outputs Boolean `has_fQRS` → cannot adapt to different criterion versions and cannot do statistical modeling.
10. ❌ Hard-code clinical criterion thresholds into the algorithm → rewrite the criteria once the criteria are changed
11. ❌ Only use IoU/Dice to evaluate segmentation → Masks the boundary error of several samples of clinical concern
12. ❌ The tolerance window width is not declared during evaluation → the numbers are not comparable (window widths range from 10 ms to 320 ms in the literature)
13. ❌ Evaluate directly with CSE (global annotation) after training on LUDB (lead-by-lead annotation) → Systematic bias
14. ❌ QTc only uses the current RR → ignores QT lag, QTc fluctuates violently during dynamic recording
15. ❌ No "rejection" path → Outputs seemingly reasonable false values on unmeasured signals

---

<a id="17-主要参考文献"></a>
## 17. Main references

**Basic Method**
1. Li C, Zheng C, Tai C. Detection of ECG characteristic points using wavelet transforms. *IEEE TBME* 1995;42(1):21–28.
2. Martínez JP, Almeida R, Olmos S, Rocha AP, Laguna P. A wavelet-based ECG delineator: evaluation on standard databases. *IEEE TBME* 2004;51(4):570–581. (wavedet original text)
3. Almeida R, Martínez JP, Rocha AP, Laguna P. Multilead ECG delineation using spatially projected leads from wavelet transform loops. *IEEE TBME* 2009;56(8):1996–2005. (Multilead projection method)
4. Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE TBME* 1985;32(3):230–236.
5. Rincón F, Recas J, Khaled N, Atienza D. Development and evaluation of multilead wavelet-based ECG delineation algorithms for embedded wireless sensor nodes. *IEEE TITB* 2011;15(6):854–863.
6. Vázquez-Seisdedos CR, et al. New approach for T-wave end detection on electrocardiogram: performance in noisy conditions. *BioMedical Engineering OnLine* 2011;10:77. (Trapezoidal area method)
7. Bosnjak A, Ledezma F, et al. Automated T wave end detection methods: comparison of four different methods. *BIOSIGNALS* 2017. (tangent/trapezoid/area/template comparison)
8. Panicker GK, et al. Intra- and interreader variability in QT interval measurement by tangent and threshold methods. *J Electrocardiol* 2009. (Artificial readout variability baseline)

**Deep Learning**
9. Moskalenko V, Zolotykh N, Osipov G. Deep learning for ECG segmentation. 2019. (LUDB U-Net)
10. Kalyakulina AI, et al. LUDB: a new open-access validation tool for ECG delineation algorithms. *IEEE Access* 2020;8:186181.
11. Jimenez-Perez G, et al. Delineation of the electrocardiogram with a mixed-quality-annotations dataset using CNNs.
12. Aziz S, et al. / Chen et al. Deep learning based ECG segmentation for delineation of diverse arrhythmias. *PLOS ONE* 2024;19(6):e0303178. (Performance degradation and countermeasures under arrhythmias)
13. Jekova I, Krasteva V, et al. Delineation of 12-lead ECG representative beats using convolutional encoder–decoders with residual and recurrent connections. *Sensors* 2024;24(14):4645. (represents stroke + IEC compliance)
14. Peak Attention U-Net: enhancing ECG delineation with attention. *Biomed Signal Process Control* 2026. (Explicit peak category)
15. Park M, Yu T, et al. SemiSegECG: a multi-dataset benchmark for semi-supervised semantic segmentation in ECG delineation. arXiv:2507.18323, 2025.
16. ECGVEDNET/ICDIRS: Large-scale 12-lead delineation dataset with variational encoding-decoding models, 2025.

**Multi-Lead Fusion and Quality**
17. Chauhan C, Agrawal M, Sabherwal P. A multi-lead fusion method for the accurate delineation of QRS complex location in 12-lead ECG. arXiv:2107.05469, 2021; and extended version of *Measurement* 2023.
18. Mondelo V, et al. Combining 12-lead ECG information for a beat detection algorithm. 2017.
19. Li Q, Mark RG, Clifford GD. Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter. *Physiol Meas* 2008. (bSQI/sSQI/kSQI)
20. Zhao Z, Zhang Y. SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation. *Front Physiol* 2018;9:727.
21. Robustness of electrocardiogram signal quality indices. *J R Soc Interface* / PMC9006023, 2022. (Limitations of fixed threshold SQI)

**notch/fQRS**
22. Das MK, et al. Significance of a fragmented QRS complex versus a Q wave in patients with coronary artery disease. *Circulation* 2006;113:2495–2501.
23. Macfarlane PW, et al. Automatic detection of end QRS notching or slurring. *J Electrocardiol* 2013;46(6):782–786. (Glasgow Procedural Rules and Performance)
24. Haukilahti MA, et al. QRS fragmentation patterns representing myocardial scar need to be separated from benign normal variants. *Front Physiol* 2016;7:653.
25. A machine learning algorithm for electrocardiographic fQRS quantification validated on multi-center data. *Sci Rep* 2022;12:6863.

**Standards and clinical criteria**
26. IEC 60601-2-25:2011 — Medical electrical equipment, Part 2-25: Particular requirements for electrocardiographs. (Table 201.105 Global measurement limits; Appendix FF.2 Global time course definition)
27. Kligfield P, et al. Recommendations for the standardization and interpretation of the electrocardiogram, Part I. *Circulation* 2007;115:1306–1324. (AHA/ACC/HRS bandwidth and sampling rate)
28. Willems JL, et al. CSE working party — assessment of the performance of ECG computer programs. (CSE tolerance source)
29. Smíšek R, et al. CSE database: extended annotations and new recommendations for ECG software testing. *Med Biol Eng Comput* 2017;55(8):1473–1482.
30. Thygesen K, et al. Universal definition of myocardial infarction (3/4th edition). (Pathological Q-wave criterion)
31. Drezner JA, et al. International criteria for ECG interpretation in athletes (Seattle / Seattle-2).
32. ANSI/AAMI EC57 — Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.

**Open Source Tools**
33. Pilia N, Nagel C, Lenis G, Becker S, Dössel O, Loewe A. ECGdeli — an open source ECG delineation toolbox for MATLAB. *SoftwareX* 2021;13:100639. (https://github.com/KIT-IBT/ECGdeli）
34. Demski A, Llamedo Soria M. ecg-kit: a MATLAB toolbox for cardiovascular signal processing. *J Open Res Softw* 2016;4:e8. (including wavedet)
35. Makowski D, et al. NeuroKit2: a Python toolbox for neurophysiological signal processing. *Behav Res Methods* 2021.

---

<a id="附录-a全局-vs-单导联的数值示例"></a>
## Appendix A: Numerical Examples of Global vs Single Lead

CSE Record #001 is the best textbook to understand this problem:
- Referenced **Global** QRS Timing: **127 ms**
- Visual inspection of QRS duration on lead I: approximately **100 ms**
- Cause: QRS onset for lead III is significantly earlier than for lead I

**If your algorithm measures 100 ms on lead I as the final output, even if the measurement itself is completely correct, it will be wrong by IEC** (an error of 27 ms, well outside the 10 ms mean limit). This is why "multi-lead cross-referencing" is not an optional optimization, but a defining requirement for global measurements.

<a id="附录-b快速自检问题"></a>
## Appendix B: Quick Self-Test Questions

Answering these 10 questions can identify the shortcomings of most systems:

1. Is your QRS onset global or specific to a certain lead? Is it clearly written in the document?
2. Is your high-pass filter zero-phase? Are the leads consistent?
3. Q What is the time resolution of the time course in ms? Is it enough to resolve the 20/30/40 ms threshold?
4. When a lead is contaminated by noise, how much will the global QRS time course change?
5. Can you differentiate between "physiological lead-to-lead time lag" and "device time skew"?
6. What does `R_peak` return on the rS-shaped leads?
7. What benchmark is the amplitude calculated against? How is this benchmark estimated?
8. When detecting notch, did you use adjacent leads for mutual verification?
9. Does the system have a "refuse to answer" channel? What are the triggering conditions?
10. What are your σ on QTDB / LUDB / CSE respectively? How much is lost across libraries?

---

*This document is based on a review of public literature and engineering practice. The implementation of medical devices must comply with applicable standards such as IEC 60601-2-25 and complete corresponding verification. *
