<!-- i18n-nav -->
[中文](P%E6%B3%A2%E6%96%87%E6%A1%A3.md) | [English](P%E6%B3%A2%E6%96%87%E6%A1%A3.en.md) | [日本語](P%E6%B3%A2%E6%96%87%E6%A1%A3.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="多导联-ecg-p-波-onsetoffset-精确定界"></a>
# Multi-lead ECG P wave onset/offset precise definition

**Technical solution: reliable lead screening + robust fusion (pure signal processing route, no machine learning used)**

---

<a id="0-方法学边界"></a>
## 0. Methodological boundaries

This solution does not use any model that requires annotated data training. The following methods **are not machine learning** and can be used - they have no training set and no learned parameters, and are all deterministic transformations calculated online on a beat-by-beat basis:

| Available (deterministic DSP / numerical methods) | Excluded (requires training) |
|---|---|
| Wavelet transform, modular maximum analysis | U-Net / DENS-ECG and other segmentation networks |
| PCA / SVD spatial projection (calculated in each shooting window) | Trained classifiers and regressors |
| Kalman filtering, median filtering | Any parameters that depend on fitting the labeled data set |
| Nonlinear least squares curve fitting (LM) | |
| Template related (the template is obtained by superimposing this record itself) | |
| Kors / Dower transformation matrix (exposed fixed constants) | |

**The main loss after giving up the network** is not delimitation accuracy, but lead-by-lead soft evidence (per-sample probability). Section 4 is dedicated to addressing this issue.

---

<a id="1-问题特性与精度上限"></a>
## 1. Problem characteristics and upper limit of accuracy

P wave amplitude is typically **50–250 µV**, with main energy below **15 Hz**, overlapping with baseline drift, respiratory modulation, and electromyographic frequency bands. The essence of onset/offset is the determination of the **low slope turning point**, that is, the moment when "the signal slope drops below the noise level".

Therefore, the final upper limit of accuracy is determined by **SNR and baseline stability**, and the algorithm can only approach this upper limit but cannot exceed it.

**Evaluation Scale (CSE Inter-Expert Agreement, 2SD)**

| Bounds | Tolerance |
|---|---|
| P onset | 10.2 ms |
| P offset | 12.7 ms |
| QRS onset | 6.5 ms |
| QRS offset | 11.6 ms |
| T offset | 30.6 ms |

---

<a id="2-预处理影响大于算法选择"></a>
## 2. Preprocessing (influence greater than algorithm selection)

<a id="21-基线去除"></a>
### 2.1 Baseline Removal
- **Preferred**: Only use the TP that has been confirmed to be between the previous beat T offset and the current beat P onset
  Isoelectric points serve as cubic spline nodes. **PQ/PR segment contains Ta (atrial repolarization) component and must not be defaulted
  as equipotential lines. **
- When a high heart rate causes the TP segment to disappear, it is no longer claimed that there is a "measured baseline": only P candidates are allowed on both sides
  The low-order continuous trend is partially reconstructed and marked as low confidence, and it is prohibited to trigger boundary coverage alone.
- Second choice: zero-phase high-pass (filtfilt), cutoff ≤ 0.5 Hz.
- ⚠️ **Avoid one-way IIR Qualcomm**. Qualcomm above 0.5 Hz will systematically push back the P onset and lower the P amplitude, and the offset of each lead will be consistent. Robust fusion cannot detect this **common mode error**.

<a id="22-低通-3040-hz"></a>
### 2.2 Low Pass 30–40 Hz
The P wave has almost no components above 30 Hz. Cutting it off can significantly improve the SNR, and there is almost no shift in the onset position under the premise of zero phase.

<a id="23-上采样至-1000-hz"></a>
### 2.3 Upsampling to 1000 Hz
The pure quantization error is 4 ms at original 250 Hz. Redelimiting after interpolation can reclaim a few milliseconds.

<a id="24-选择性叠加平均"></a>
### 2.4 Selective superposition averaging
If the task allows (not beat-by-beat analysis), perform R-wave or P-peak alignment averaging on similar P-waves after clustering according to morphology, and the SNR will be improved by √N. This is what SAECG does, and it is the most cost-effective method for low-amplitude P waves.

> Ectopic P and sinus P must be clustered first and **never averaged across morphologies**.

---

<a id="3-单导联确定性定界方法"></a>
## 3. Single-lead deterministic delimitation method

| Methods | Applicable points | Precautions |
|---|---|---|
| **Wavelet mode maximum** (Martínez 2004) | Main method, scale 2⁴/2⁵, good anti-noise | Biphasic P (especially V1) must explicitly distinguish the three templates `+−` / `−+` / `+−+`, and the "first maximum" rule cannot be used |
| **Phasor Transform** (Phasor Transform) | Designed specifically for low-amplitude P waves, mapping small amplitude changes into large phase angle changes | P peak positioning is excellent, but the boundaries still need to be determined with thresholds |
| **Tangent method** | Simple and interpretable | The single-point derivative is too jittery; instead, do **least squares fitting straight line** in the maximum slope section and then intersect with the baseline, and the stability will be improved by about an order of magnitude |
| **Area / Trapezoidal Method** | P offset (low slope end point) is better than threshold method | Rely on reliable baseline reference |
| **Curve length transformation/sliding window absolute derivative integration** | Generate a smooth "activity" envelope, suitable for global detection functions | The window length determines the time resolution, and the P wave takes 20–30 ms |
| **Gaussian / Skewed Gaussian LM fitting** | Sub-sampling point accuracy, residuals can naturally be used as confidence | The initial value needs to be good, it is recommended to use wavelet results as the initial value |

<a id="31-推荐组合"></a>
### 3.1 Recommended combination

```
小波模极大值  →  定位 P 峰 + 判定极性模式
      ↓
窄窗内精定界：最小二乘切线法（onset）+ 面积法（offset）
      ↓
高斯拟合交叉校验  →  三者分歧大即判为低置信
```

The failure modes of the three methods are not related to each other, which is why the degree of disagreement can be used as a confidence level.

<a id="32-等电位窗检验自适应关键步骤"></a>
### 3.2 Equipotential window inspection (key adaptive step)

Search forward from the P peak to find the **last** window of length W (15–20 ms), so that the absolute value and variance of the slope within the window are both below the threshold.

**The threshold is not a constant** but is proportional to the measured noise level in the TP segment verified by T offset from that shot
Export (e.g. 3σ_noise). The PR/PQ segment contains Ta and cannot be used as a noise window; if there is not enough length
In the TP segment, the local SNR must be reported as unavailable, and the boundary can only be set to a lower confidence fallback.

---

<a id="4-无-ml-的置信度构造替代-softmax-的核心"></a>
## 4. Confidence construction without ML (replacing the core of softmax)

Each lead and each beat need to output a boundary uncertainty σᵢ for subsequent weighted fusion use. The following methods are sorted by cost-effectiveness:

1. **Threshold perturbation method (preferred, almost zero cost)**
   Sweeping the wavelet threshold ξ over 5–7 values in the range 3%–10% yields a set of onset estimates.
   - Point estimate = median of the set of values
   - **Uncertainty σᵢ = the dispersion of this set of values**

2. **Multi-scale consistency**
   Delimited at three scales: 2³, 2⁴, and 2⁵. Large deviation between scales → P wave in this lead is dominated by noise.

3. **Multi-method consistency**
   The degree of divergence between tangent method vs area method vs fitting method.

4. **Noise Injection Monte Carlo**
   Generate the same color noise superposition 10 times according to the measured noise level of the TP segment verified by T offset, and redefine
   Take the variance. When there is no TP segment, the PR segment noise cannot be used to pretend to be.

5. **Fitting Residuals**
   Normalized residuals of Gaussian fit.

6. **Beat-to-beat jitter**
   MAD of the most recent N-beat onset markers in the same lead (reliability can be assessed without the need for true values).

With σᵢ, the subsequent weighted fusion is exactly the same as when "with a network": the weight **wᵢ ∝ 1/σᵢ²**, multiplied by SNR gating.

---

<a id="5-多导联全局检测函数"></a>
## 5. Multi-lead global detection function

This has been standard practice for commercial ECG programs (Marquette/Glasgow kind) for decades and does not rely on ML at all.

<a id="51-空间速度函数spatial-velocity-推荐主力"></a>
### 5.1 Spatial Velocity function - recommended main force

$$SV(t)=\sqrt{\sum_{i\in\mathcal{L}} w_i\left(\frac{dx_i}{dt}\right)^2}$$

Find the derivative vector modulo 8 independent leads (or X/Y/Z for VCG).

**Principle**: At the moment when atrial activation begins, **all leads** experience potential changes at the same time, so SV has a clear rise at P onset; while the independent noise of each lead is incoherently superimposed after the square summation, and the SNR improves by √L.

**Judgment**: P onset = the moment when SV first and lasts longer than k·σ_SV (verified TP segment).
If TP does not exist, this adaptive threshold evidence is not available.

**Practical Points**
- Use **Savitzky-Golay** (2nd order, 25–35 ms window) for the derivative, without simple difference, otherwise the noise will be amplified.
- **Only include leads that pass the quality threshold**, bad leads entering the sum of squares will only raise the background.
- SV gives the "global earliest", which is consistent with the CSE gold standard definition - this is more consistent than SVD projection.

<a id="52-svd-空间投影-保留作互补"></a>
### 5.2 SVD space projection - reserved for complementation

For each shot, SVD is performed on the multi-lead data matrix in the P window and projected to the first singular vector (the main direction of the P ring). Pure linear algebra operations.

**Key Details**
- The projection direction must be estimated only within the P window (containing T or QRS will contaminate the main direction).
- Must **iterate**: Coarse P-window → Project → Delimit → Narrow Window → Re-project, usually 2 rounds to converge.
- Advantages: Automatically avoid the fatal problem of "the P axis is perpendicular to a certain lead, causing the lead P to be near isoelectric potential".
- Pay attention to the caliber: the onset obtained by projection is closer to the "real three-dimensional starting point of excitement", and there is a systematic deviation from the CSE definition of "the earliest in each lead".

<a id="53-两者互补"></a>
### 5.3 The two complement each other

- **SV** emphasizes "whether there have been changes"
- **Projection** emphasizes the "maximum P energy direction"

The discrepancy between the two results is itself a valuable warning sign.

Other equivalent global functions: weighted sum of absolute derivatives of each lead, RMS resultant lead, VCG vector magnitude. Usually SV is the sharpest.

---

<a id="6-可靠导联判据全部纯-dsp"></a>
## 6. Reliable lead criteria (all pure DSP)

| Criterion | Description |
|---|---|
| **P-wave local SNR** | P-peak amplitude / verified TP segment noise RMS - **Weight principal**; N/A when TP disappears |
| Template correlation coefficient | Correlation coefficient between the current P wave and the same lead online superimposed template, < 0.8 weight reduction |
| Baseline stationarity | PQ / TP segment isoelectric level difference, residual slope |
| Morphological threshold | P time limit 60–140 ms, amplitude 0.03–0.35 mV, P–QRS interval is reasonable; out of bounds are directly eliminated |
| Beat-to-beat stability | MAD of the most recent N beat onset markers |
| Consistency Metrics | Section 4 Threshold Perturbation/Multi-Scale/Multi-Method Dispersion |
| Lead loss / saturation / flat line | Hard exclusion |

<a id="最重要的一条等电位导联判定"></a>
### ⚠️ The most important one: Determination of equipotential leads

If the P amplitude of a certain lead is < 2σ_noise, it will be marked as "no information", the weight will be reset to 0, and the boundary will never be output.

The root cause of most rollover cases in multi-lead fusion is that the noise on the equipotential leads is used as a P wave.

---

<a id="7-稳健融合"></a>
## 7. Robust integration

<a id="71-不要直接取-min-max"></a>
### 7.1 Don’t take min / max directly
The "earliest onset, latest offset" definition of CSE is extremely sensitive to a single bad lead.

**Robust alternative**: Take the order statistic within the set of leads that meet the SQI standard——
- **2** earliest onset (not earliest)
- **2** latest offset (not latest)
- Or take 15 / 85 percentile

<a id="72-加权稳健估计"></a>
### 7.2 Weighted Robust Estimation
Weighted median or **Huber M estimate** with weights wᵢ ∝ SQIᵢ / σᵢ².

<a id="73-迭代重加权剔除"></a>
### 7.3 Iterative weighted elimination
```
计算稳健中心
  → 偏离 > 2.5 × MAD 的导联权重置 0
  → 重算
  → 通常 2 轮收敛
```

<a id="74-跨导联一致性硬约束"></a>
### 7.4 Cross-lead consistency hard constraints
For the same atrial activation, the true dispersion of onset in each lead should be **< 20–30 ms**.

If the dispersion still exceeds the standard after eliminating outliers → **Output "untrustworthy", do not give a hard number**. This usually means there is a systemic problem: atrial flutter, large area artifacts, misconnected leads.

---

<a id="8-拍间时间一致性"></a>
## 8. Time consistency between beats

- Perform median filtering or scalar Kalman smoothing on **PR interval** and **P time** series.
- Tags whose mutations exceed the threshold are considered abnormal.
- **Cluster by template correlation first** (sinus P / ectopic P), **never smooth across clusters**.

---

<a id="9-推荐落地流水线"></a>
## 9. Recommended floor-to-ceiling assembly line

```
① 预处理
   样条去基线（仅用已验证 TP 等电位点）
   TP 消失 → 连续低阶局部趋势、低置信标记
   + 40 Hz 零相位低通
   + 上采样至 1000 Hz
        ↓
② 参考点检测
   QRS 检测 → T offset 检测
   → 依 RR 自适应给出 P 搜索窗
        ↓
③ 多导联粗定界
   SV 函数 → 全局 P 区间
   + 「P 是否存在」门控（房颤 / 房扑在此拦截）
        ↓
④ 单导联精定界（窄窗下发）
   小波模极大值 + 最小二乘切线法 / 面积法
        ↓
⑤ 置信度评估
   阈值扰动法 → σᵢ
   质量指标   → SQIᵢ
        ↓
⑥ 稳健融合
   加权中位数 + 迭代重加权
   → 全局 P onset / offset
        ↓
⑦ 后处理
   拍间中位数 / Kalman 平滑
   + 跨导联一致性校验
   → 输出边界及置信度（或「不可信」标记）
```

Narrow window delivery (steps ③→④) is the core design: it not only obtains the SNR dividend of the global function, but also retains the time resolution of a single lead, while significantly reducing false detections.

---

<a id="10-参数标定无训练集情况下"></a>
## 10. Parameter calibration (without training set)

Calibrate using simulated signals with known true values:

1. Use ECGSYN to generate, or make a morphological template for a real low-noise P wave.
2. Superimpose **real noise** by controllable SNR - `bw` (baseline drift) / `ma` (myoelectricity) / `em` (electrode motion) recordings from the MIT-BIH Noise Stress Test Database.
3. Scan the parameters and observe the mean and SD of the error.
4. Select the parameter with the smallest **SD** and deduct the **mean as a fixed deviation**.
5. Independent verification on LUDB/CSE.

> This is **calibration** rather than training - the parameters are a few constants with clear physical meanings (threshold ratio, window length, derivative order), and there is no risk of overfitting.

---

<a id="11-常见坑"></a>
## 11. Common pitfalls

| Problems | Countermeasures |
|---|---|
| **Ta wave contaminates PR/PQ baseline** | PR/PQ is not used as P noise window or equipotential node; Ta is a low-frequency physiological component and must not be mislabeled as "low noise" after detrend |
| **High heart rate, TP disappears or P-on-T** | First determine whether there is still observable TP based on the T offset of the previous beat. Without TP, only continuous low-order trends are used to generate low-confidence detection views, retaining the original boundaries and disabling aggressive earliest/latest fusion. T template subtraction is off by default |
| **Really required T/QRST cancellation** | Can only be used as second opinion when homomorphic clusters, beat-by-beat ST-T time shift and amplitude adaptation, low-pass detection view, seam taper, and residual high-frequency artifact gating are all passed; fixed R alignment template point-by-point hard reduction cannot be used. When the heart rate is high throughout and P and T are overlapping in phase lock, they are indistinguishable and should be rejected instead of hard separated |
| **Atrial fibrillation/atrial flutter** | There must be a pre-set "P wave presence discrimination" gate control. Otherwise the bounder will consistently output error bounds on f-waves with high confidence |
| **Filter Phase Offset** | Any non-zero phase filtering will systematically shift onset, and the offset will be consistent across leads → not detected by robust fusion. Use simulation signals to calibrate the inherent deviation of the entire filter chain before going online |
| **Biphasic P(V1)** | Modular maximum mode is different from single-phase, and three templates must be processed explicitly |
| **Equipotential Leads** | See Section 6, weight reset to 0 |
| **System deviation vs dispersion** | Deviation can be deducted by calibration, **SD is the real skill**, and must be reported separately during evaluation |

---

<a id="12-评估基准"></a>
## 12. Evaluation Baseline

| Database | Features | Usage |
|---|---|---|
| **LUDB** | 200 lines, 12 leads, 10 s, full waveform annotation | The first choice for multi-lead solutions |
| **QT Database** | 105 entries, 2 leads, many beats | Single-lead method verification |
| **CSE Multilead Measurement DB** | Multilead delimitation official scale | Final compliance verification |
| **MIT-BIH Noise Stress Test DB** | Real noise record | Parameter calibration, robustness test |

**Reported**: Mean ± SD is given, and SD is compared to the CSE tolerance (P onset 10.2 ms / P offset 12.7 ms).

---

<a id="13-预期性能与边界"></a>
## 13. Expected performance and boundaries

On the 12-lead resting ECG with acceptable signal-to-noise ratio, the delimitation accuracy of the pure DSP solution is basically the same as that of the deep model.

The differences mainly appear in:
- Strong myoelectric interference
- Pathological morphology (fragmentation of intraatrial block P, deep biphasic inversion P)

This part relies on the government tightening gate control and preferring to refuse judgment rather than impose a hard sentence. Giving a clear "untrustworthy" mark is often more valuable than giving a false but high-confidence bound.
