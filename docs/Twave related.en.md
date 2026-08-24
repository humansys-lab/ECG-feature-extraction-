<!-- i18n-nav -->
[中文](Twave%20related.md) | [English](Twave%20related.en.md) | [日本語](Twave%20related.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="多导联心电-t-波定界与-st-段特征提取技术规范"></a>
# Technical specifications for multi-lead ECG T wave delimitation and ST segment feature extraction

**Method scope: Purely deterministic signal processing (wavelet, differential, area, geometry, curve fitting, robust statistics). Does not contain any machine learning/deep learning/HMM methods. **

---

<a id="0-约定与符号"></a>
## 0. Conventions and symbols

| Item | Agreement |
|---|---|
| Sampling rate | The full text is based on `fs = 500 Hz` (2 ms/sample point), and the 1000 Hz conversion is given in parentheses |
| Amplitude unit | µV |
| Time base | Global R-peak position `R_k` (single moment after multi-lead fusion) |
| Reference point | `QRSon` (QRS starting point), `J` (QRS end point/J point), `Ton`, `Tpk`, `Toff`, `B` (equipotential baseline level) |
| Lead set | 12 leads; 8 independent leads: I, II, V1–V6 (the rest are linear combinations) |

> **Threshold Description**: All thresholds below are marked in three categories -
> `【文献】` Common values from public algorithms/standards;
> `【工程】` Engineering practice experience value needs to be calibrated on your own data;
> `【临床】` clinical interpretation standard is only used as a reference for feature values and does not constitute diagnostic output**.

---

<a id="1-前处理"></a>
## 1. Preprocessing

The systematic error of preprocessing is usually larger than the algorithm difference itself. The total error budget of the T end point is about ±30 ms, of which more than 15 ms can be consumed alone if preprocessing is not done properly.

<a id="11-采样率"></a>
### 1.1 Sampling rate

| Purpose | Minimum | Recommended |
|---|---|---|
| Conventional T bounding | 500 Hz | 500 Hz |
| QT Precision Measurement / TpTe | 500 Hz | 1000 Hz |
| ST level | 250 Hz | 500 Hz |

It is recommended that raw ≤250 Hz signals be first interpolated to 1000 Hz using cubic splines or polyphase FIR. This does not increase the amount of information, but it eliminates the ±2 ms quantization error caused by the sample point grid; it has a significant impact on small relative errors such as TpTe (typically 80–100 ms).

<a id="12-基线漂移去除对-st-影响最大的一步"></a>
### 1.2 Baseline drift removal (the step that has the greatest impact on ST)

**Recommended: Cascade Median Filter** (PhysioNet convention)

```
y1 = medfilt(x,  W1)     # W1 = 200 ms → 101 样点 @500Hz（201 @1000Hz）
y2 = medfilt(y1, W2)     # W2 = 600 ms → 301 样点 @500Hz（601 @1000Hz）
out = x - y2
```

The window length must be an odd number of samples. W1 needs to be larger than the widest QRS (covering ~120 ms), and W2 needs to be larger than the widest QRS+ST+T.

**Alternative A: Isoelectric point cubic spline**
- Node position: `QRSon − 30 ms` (midpoint of PQ segment) for each beat, take the median of the ±10 ms window of this point as the node ordinate
- PR segment is too short when heart rate >100 bpm, use ±6 ms window at `QRSon − 20 ms` instead
- Degenerates to median filtering when nodes are missing (indiscernible P wave, atrial fibrillation)

**Alternative B: Zero Phase Qualcomm**
- Butterworth 4th order, `fc = 0.05 Hz`, `filtfilt` forward and backward filtering

**❌ BANNED**: Unidirectional IIR Qualcomm for `fc = 0.5 Hz`. Actual measurements can introduce 50–100 µV of false elevation/depression in the ST segment, the direction of which is related to the QRS polarity, which is the most insidious source of systematic errors in ST characteristics.

<a id="13-工频干扰"></a>
### 1.3 Power frequency interference

- Priority: adaptive notch, center frequency 50/60 Hz, `Q ≈ 35` (−3 dB bandwidth ≈ 1.4 Hz), zero phase implementation
- If only T/ST analysis is performed: direct low-pass 40 Hz is fully covered, no need for separate notch

<a id="14-分支滤波关键设计"></a>
### 1.4 Branch filtering (key design)

**Different tasks use different filter branches, and it is prohibited to share the same filter chain for the entire process. **

| Branch | Filter | Purpose |
|---|---|---|
| `x_qrs` | Bandpass 5–15 Hz (Pan-Tompkins) or 8–20 Hz | QRS Detection, beat alignment |
| `x_fid` | Zero-phase low-pass 40 Hz, FIR equal ripple, order ≈ 0.1·fs | QRSon / J point positioning (QRS end high frequency needs to be retained) |
| `x_t` | Zero-phase low-pass **20–25 Hz**, FIR | T-wave delimitation and morphology (T-energy <10 Hz) |
| `x_amp` | Zero-phase low-pass 40 Hz | ST level, T amplitude and other amplitude measurements |

`x_t` If 15 Hz low pass is used, the anti-noise will be further improved, but the T–U fusion will be aggravated, and it needs to cooperate with the U wave discrimination in §4.7.

<a id="15-伪迹与无效段标记"></a>
### 1.5 Artifacts and invalid segment markers

| Phenomenon | Criteria | Disposal |
|---|---|---|
| Pacing pulse | Narrow spike with width <2 ms 且幅度 >2 mV | Post-detection linear interpolation padding |
| ADC saturation | ≥8 consecutive samples at full scale ±1 LSB | This segment flag is invalid |
| Electrode off | Peak-to-peak within 50 ms sliding window <10 µV | Whole lead eliminated |
| Jump artifacts | A single sample jumps >1 mV and adjacent samples do not continue | Median replacement |

<a id="16-中位拍representative-beat构建"></a>
### 1.6 Median beat (representative beat) construction

**T delimitation should be performed on the median beat**, beat-to-beat delimitation is only used when beat-to-beat variability (TWA, QT variability) is required, and the median beat result must be used as the search prior.

```
1) QRS 对齐：在 x_qrs 上以模板互相关，搜索范围 ±30 ms，
   峰值处做三点抛物线插值 → 亚样点对齐精度
2) 形态聚类：
   - 相关系数 ρ ≥ 0.90 且 QRS 宽度差 ≤ 25 ms  → 归入同簇   【工程】
   - ρ < 0.85 或宽度差 > 25 ms                 → 新建簇（PVC/差传）
3) 主簇选取：拍数最多且平均 RR 变异最小的簇
4) 逐样点取【中位数】（不是均值，抗离群拍）
5) 输出窗：R − 250 ms  到  R + min(500 ms, 0.70·RR_median)
6) 置信门限：簇内拍数 ≥ 8 才输出高置信中位拍；3–7 拍降级；<3 拍不输出
```

The SNR improvement is about √N. The 8-beat median reduces the random error in the T endpoint from ±20 ms to the order of ±7 ms.

---

<a id="2-基准点与等电位基线"></a>
## 2. Reference point and equipotential baseline

<a id="21-qrson-与-j-点"></a>
### 2.1 QRSon and J point

**Wavelet method (recommended)**: Positioning on quadratic spline wavelet scale 2¹, 2². QRS has the most energy concentration at these two scales.

- `QRSon`: Before the first mode maximum before the R peak, `|w₂|` drops to `0.05 × max|w₂|` for the first time [Literature]
- `J`: After the last modular maximum value after R/S, `|w₂|` drops to `0.125 × 该模极大值` [Literature]

**Slope refinement (J point)**: Search backward from the R peak, and take the moment when `|dy/dt|` is lower than the maximum slope of the QRS segment by 5% for the first time for **continuous 8 ms** [Engineering].

**Unreliable mark**: QRS When the width is >120 ms, delta waves exist, or bundle branch block morphology is present, the J-point positioning error can reach more than 20 ms, a low confidence mark should be set, and the ST measurement points should be uniformly moved backward.

<a id="22-等电位基线-b"></a>
### 2.2 Equipotential baseline `B`

| Priority | Window | Additional Conditions |
|---|---|---|
| 1 | PR segment: `[QRSon − 40 ms, QRSon − 10 ms]` | Peak-to-peak value in window < 30 µV [Engineering] |
| 2 | The flattest sub-window in the PR segment: slide the 20 ms sub-window within the above window, and take the one with the smallest variance | Heart rate <100 bpm |
| 3 | Short PR: `[QRSon − 26 ms, QRSon − 6 ms]` | RR < 600 ms |
| 4 | TP segment: `[Toff + 40 ms, Pon − 20 ms]` | Toff needs to be known (second iteration) |

`B` takes the **median** within the window instead of the mean.

**`B`** is calculated independently lead by lead and cannot be shared across leads.

---

<a id="3-t-波搜索窗"></a>
## 3. T wave search window

The definition of the search window directly determines the missed detection rate and false detection rate.

```
Twin_start = J + max(60 ms, 0.04·RR)
Twin_end   = J + min(0.62·RR, 500 ms)          【工程】
```

According to the actual value of heart rate (RR is the RR of the previous beat):

| HR (bpm) | RR (ms) | Twin_start (J+) | Twin_end (J+) | Remarks |
|---|---|---|---|---|
| 50 | 1200 | 60 ms | 500 ms | The upper limit is in effect |
| 60 | 1000 | 60 ms | 500 ms | The upper limit is in effect |
| 75 | 800 | 60 ms | 496 ms | |
| 100 | 600 | 60 ms | 372 ms | |
| 120 | 500 | 60 ms | 310 ms | P-on-T required |
| 150 | 400 | 60 ms | 248 ms | T and P highly overlap, confidence degraded |

**RR Smoothing**: Use the median of the RR of the first 5 beats to avoid a single premature beat from collapsing the window.

**P-on-T protection**: When `Twin_end` is less than 120 ms from the next beat `QRSon`, truncate `Twin_end` to `nextQRSon − 120 ms` and mark "T-P overlap".

---

<a id="4-单导联-t-定界方法"></a>
## 4. Single-lead T delimitation method

<a id="41-小波模极大值法主力方法"></a>
### 4.1 Wavelet Modulus Maximum Method (Main Method)

**Wavelet selection**: Mallat quadratic spline wavelet (first derivative type), filters `h = [1,3,3,1]/8`, `g = [2,−2]`, using à trous algorithm (no decimation, keeping time alignment).

**Scale selection**:

| fs | T-wave usage scale | Equivalent passband |
|---|---|---|
| 250 Hz | 2³ | ~3–11 Hz |
| 500 Hz | 2⁴ | ~3–10 Hz |
| 1000 Hz | 2⁵ | ~3–10 Hz |

**Steps**:

```
1) 在搜索窗内计算 w4，噪声门限
     ε_T = max( 0.25 · RMS(w4 | Twin) ,  0.125 · max|w4 | Twin| )   【文献】

2) 提取所有 |w4| > ε_T 的模极大值，按时间排序

3) 形态判定与 T 峰定位：
   - 2 个异号模极大值        → 单相 T，两者之间的过零点 = Tpk
                               （模极大值对的符号顺序决定 T 的极性）
   - 3 个交替符号模极大值     → 双相 T，2 个过零点 = 两个峰
   - 4 个及以上              → 切迹/多相 T，取幅度最大的两个相邻异号极大值定主峰
   - 0 或 1 个               → T 波不可辨，返回 NaN 并标记

4) 边界定位：
   Ton  = 第一个模极大值【之前】，|w4| 首次降到 ξ_on · |第一模极大值| 的时刻
   Toff = 最后一个模极大值【之后】，|w4| 首次降到 ξ_off · |末模极大值| 的时刻

   参考阈值                                          【文献】
     ξ_on  = 0.25
     ξ_off = 0.40
   低幅 T（|Tpk − B| < 200 µV）建议 ξ_off = 0.50     【工程】

5) 兜底：若在最大搜索距离内未降到阈值，取 |w4| 的第一个局部最小点
     最大搜索距离：Toff 距末模极大值 ≤ 160 ms，Ton 距首模极大值 ≤ 120 ms  【工程】
```

**Performance Reference**: This method has a T endpoint error of approximately **0.8 ± 18 ms** on the CSE library (CSE allows a 2SD tolerance of 30.6 ms) [Literature].

**Advantages**: Treats forward, inverted, and two-phase T equally; multi-scale natural noise immunity; does not rely on equal wires.
**Weaknesses**: The mode maximum is swallowed up by the noise threshold at very low amplitude T (<80 µV); the final mode maximum may fall on the U wave during T–U fusion.

<a id="42-梯形面积法低幅-t-的主力"></a>
### 4.2 Trapezoidal area method (the main force of low-amplitude T)

The trapezium area method proposed by Zhang et al. does not rely on isoelectric lines and is therefore immune to residual baseline drift.

```
输入：x_t（低通 20–25 Hz），T 峰位置 Tpk

1) 在 [Tpk, Tpk + 200 ms] 内求一阶导数绝对值最大点 → (x_m, y_m)
   （即降支最大斜率点；一阶导数用 Savitzky-Golay，2 阶多项式、9 点窗）

2) 设参考点横坐标  x_r = x_m + 160 ms        【工程，范围 120–200 ms】

3) 对每个候选 x_i ∈ [x_m + 10 ms, x_r − 10 ms]，计算梯形面积：
       A(x_i) = 0.5 · |y_m − y(x_i)| · (2·x_r − x_i − x_m)

4) Toff = argmax_i A(x_i)
```

The area function `A` takes the maximum at the end point of the real T: `|y_m − y_i|` is still growing before the end point, and `(2x_r − x_i − x_m)` decays rapidly after the end point.

**Parameter sensitivity**: The value of `x_r` affects the result by about ±5 ms. It is recommended to fix it to 160 ms and calibrate it once on your own data.

**Advantages**: It is significantly better than the tangent method and the threshold method under low amplitude T; it resists baseline drift; it is simple to implement and requires no iteration.
**Weaknesses**: Correct `Tpk` prior is required; for biphasic T, it is necessary to first determine which phase of the descending branch to use.

<a id="43-切线法临床对齐用"></a>
### 4.3 Tangent method (for clinical alignment)

It is most consistent with manual interpretation habits and must be implemented when doing clinical benchmarking.

```
1) 降支最大斜率点 (x_s, y_s)：一阶导数极值
   （Savitzky-Golay 2 阶 / 9 点，或 5 点中心差分 + 3 点平滑）
2) 斜率 k = dy/dt |_{x_s}
3) Toff = x_s + (B − y_s) / k
```

**Prerequisites for use (must be checked)** [Project]:
- `|Tpk − B| ≥ 100 µV`, otherwise the error can be >40 ms, disabled
- The obtained `Toff ≤ x_s + 200 ms`, otherwise it will be judged as failure.
- `|k| ≥ 0.8 µV/ms`, extrapolation is unstable when the slope is too slow

**Systematic Bias Tip**: The tangent method gives the T endpoint systematically** earlier than the **threshold method/wavelet method, with a typical difference of **10–20 ms**, and changes with the symmetry of the T wave. When conducting longitudinal follow-up or drug trials, the same method and parameter set must be locked throughout.

<a id="44-阈值法快速备用"></a>
### 4.4 Threshold method (quick backup)

```
Toff = Tpk 之后，|y − B| 首次 < max( 0.10 · |Tpk − B| , 15 µV ) 的时刻   【工程】
去抖：需连续 3 个样点（6 ms @500Hz）满足才确认
```

The fastest but most sensitive to the baseline, it is only used as a third method of cross-validation or as a cover for extreme cases.

<a id="45-累积面积法"></a>
### 4.5 Cumulative area method

```
S_total = Σ |y(i) − B|,  i ∈ [Ton, Twin_end]
Toff = 累积面积首次达到 0.98 · S_total 的位置      【工程，0.97–0.985】
```

It has a natural smoothing effect on noise, but is sensitive to the right endpoint of the search window.

<a id="46-曲线拟合法同时给出形态参数"></a>
### 4.6 Curve fitting method (morphological parameters are also given)

Use nonlinear least squares (Levenberg–Marquardt) to fit the parametric model, and obtain the boundary and morphological quantities at the same time in one fitting. **This is deterministic curve fitting and is not machine learning. **

**Double Gaussian model** (adapted to single-phase and two-phase T):

```
T(t) = Σ_{j=1..J} a_j · exp( −(t − µ_j)² / (2σ_j²) )
J = 1（单相）或 2（双相/切迹）
```

- Initial value: `a₁ = Tpk − B`, `µ₁ = Tpk`, `σ₁ = (Toff_wavelet − Ton_wavelet)/5`
- Boundary analysis definition: `Ton = µ₁ − 2.5σ₁`, `Toff = µ_J + 2.5σ_J` [Engineering, k = 2.0–3.0 requires calibration]
- Fitting goodness threshold: `R² ≥ 0.95` will be accepted; otherwise, it will fall back to the wavelet result [Project]

**skewed Gaussian** (better suited to the inherent asymmetry of the T wave):

```
T(t) = a · exp(−(t−µ)²/(2σ²)) · [1 + erf( α(t−µ)/(σ√2) )]
```

The parameter `α` is directly the characteristic of asymmetry. The `α` of normal sinus T is typically **−1.5 ~ −0.5** (the descending limb is steeper than the ascending limb) [Engineering].

<a id="47-u-波判别与-tu-融合处理"></a>
### 4.7 U wave discrimination and T–U fusion processing

**T The largest single source of error in the end point. **

**Independent U-wave criterion** [Engineering]:
- Within the range `[Toff + 60 ms, Toff + 200 ms]`
- Presence of bumps with amplitude **20–150 µV**
- On the low-pass 15 Hz signal, its first derivative appears in a complete "positive → zero crossing → negative" sequence
- Satisfies the above → judged as independent U wave, does not affect Toff

**T–U Fusion Criteria** [Project]:
- At `Toff` obtained by wavelet method, `|dy/dt| > 5 µV/ms` (i.e. >10 µV/sample @500 Hz)
- or `w4` does not return below the noise threshold within 40 ms after `Toff`
- Meet the above → mark "T-U fusion" and reduce the confidence level

**Disposal during fusion**:
Calculate the second derivative `d²y/dt²` (Savitzky-Golay 2nd order / 11 points) within `[x_s, x_s + 200 ms]`, and take the **curvature maximum point** (the first significant positive maximum of `d²y/dt²`) as the upper bound of `Toff`. This point corresponds to the "flattening" position of the descending T branch, which is usually earlier than the onset of the U wave.

**High-risk scenario**: Hypokalemia, long QT syndrome, bradycardia (HR <50) with significant U wave, fusion detection should be mandatory.

<a id="48-方法选择决策树"></a>
### 4.8 Method selection decision tree

```
读取 A = |Tpk − B|，形态类别 M，U 波标记 U

if T 波不可辨（小波模极大值 < 2 个）:
        → 返回 NaN，导联剔除

elif M == 双相 或 M == 切迹:
        → 小波法（唯一可靠）
        → 若拟合 R² ≥ 0.95，用双高斯结果交叉校验

elif U == 融合:
        → 曲率上界法 (§4.7) 与 梯形法 取【较早】者
        → 置信度降一级

elif A ≥ 200 µV:
        → 主：小波法
        → 校验：切线法。若两者差 > 25 ms → 置信降级
        → 输出：小波法结果

elif 100 µV ≤ A < 200 µV:
        → 小波法 + 梯形法，取两者【中位】（此时即中点）
        → 若两者差 > 30 ms → 置信降级

else  (A < 100 µV):
        → 梯形面积法（唯一可用）
        → 禁用切线法与阈值法
        → 置信度设为最低档；若 A < 50 µV 直接剔除该导联
```

---

<a id="5-多导联可靠导联筛选与稳健融合"></a>
## 5. Multi-lead: reliable lead screening and robust fusion

<a id="51-派生全局导联"></a>
### 5.1 Derive "Global Lead"

In addition to the original leads, three additional derived leads are constructed for independent delimitation as fusion cross-checks.

**(a) RMS lead**

```
对 8 个独立导联 (I, II, V1..V6)，各自先减去自身基线 B_l：
    RMS(k) = sqrt( (1/8) · Σ_l [x_l(k) − B_l]² )
```

The T wave in the RMS lead is always positive, has a single shape, and is the most stable in delimitation. Processed by §4.1 wavelet method, `ξ_off = 0.40`.

**(b) PCA first principal component**

```
1) 取 T 窗矩阵 X ∈ R^{8×N}，N = Twin 内样点数
2) 逐导联去均值（减自身在该窗内的均值）
3) SVD: X = UΣVᵀ
4) PC1 = U[:,0]ᵀ · X   → 1×N 时间序列
5) 在 PC1 上定界
```

PC1 typically carries 85–95% of the energy within the T window, with significantly better SNR than either original lead.
The by-product `λ₂/λ₁ = (σ₂/σ₁)²` is the T-wave complexity characteristic (§6).

**(c) VCG Space Velocity**

First reconstruct XYZ from 12 leads. **Inverse Dower transformation matrix**:

```
X = −0.172·V1 − 0.074·V2 + 0.122·V3 + 0.231·V4 + 0.239·V5 + 0.194·V6 + 0.156·I − 0.010·II
Y =  0.057·V1 − 0.019·V2 − 0.106·V3 − 0.022·V4 + 0.041·V5 + 0.048·V6 − 0.227·I + 0.887·II
Z = −0.229·V1 − 0.310·V2 − 0.246·V3 − 0.063·V4 + 0.055·V5 + 0.108·V6 + 0.022·I + 0.102·II
```

> Kors regression transformation is better than Inverse Dower in most studies (especially the QRS-T included angle feature). It is recommended to check the coefficients directly from the original Kors text when implementing it, and do not copy from second-hand sources.

Space speed:

```
SV(k) = sqrt( (dX/dt)² + (dY/dt)² + (dZ/dt)² )
导数用 Savitzky-Golay 2 阶 / 9 点

Toff_SV = SV 在 T 峰之后降到 0.15 · max(SV | T窗) 的时刻      【工程，0.10–0.20】
或取 T 波之后 SV 的第一个显著局部极小
```

The SV method best fits the physiological definition of "end of global repolarization".

<a id="52-导联质量指标sqi与阈值"></a>
### 5.2 Lead Quality Index (SQI) and Threshold

**Generic SQI** (calculated lead by lead, per 10 s segment):

| Metrics | Definition | Qualification Threshold | Description |
|---|---|---|---|
| kSQI | Signal kurtosis | **> 5** Good; < 4 Judge noise [Documentation] | Normal ECG High kurtosis due to QRS peak |
| sSQI | Absolute value of skewness \|skew\| | Reference interval 0.5–2.5 [Engineering] | |
| pSQI | P(5–15 Hz) / P(5–40 Hz) | **0.5–0.8** Qualified [Literature] | QRS Energy proportion |
| basSQI | 1 − P(0–1 Hz)/P(0–40 Hz) | **> 0.95** Passed [Documentation] | Baseline drift ratio |
| bSQI | R peak agreement rate of two different QRS detectors (match within ±150 ms) | **> 0.90** [Engineering] | Strong indicator |
| ρ_tmpl | Correlation coefficient with the median beat template of this lead | **> 0.90** [Engineering] | |
| Saturation rate | Full-scale sample point ratio | **< 0.1%** | |

**T wave-specific reliability index** (this is the core of screening reliable leads for T demarcation):

| Metrics | Definition | Thresholds |
|---|---|---|
| `A_T` | \|Tpk − B\| | ≥ 200 µV excellent; 100–200 µV medium; 50–100 µV poor; **< 50 µV reject** [Engineering] |
| `SNR_T` | 20·log₁₀( A_T / RMS(TP segment residual) ) | **> 10 dB** Excellent; 6–10 dB Medium; **< 6 dB Eliminate** [Engineering] |
| `σ_beat` | The standard deviation of the lead's Toff (relative R peak) across beats | **< 8 ms** 优；8–20 ms 中；**> 20 ms elimination** [Engineering] |
| `spread_alg` | The range of 2–3 algorithm results on this lead | **< 10 ms** 优；> 30 ms elimination [Engineering] |
| `U_flag` | T–U fusion flag | If set, the weight ×0.5 |

<a id="53-导联权重"></a>
### 5.3 Lead weight

```
w_l = w_prior(l) · w_SQI(l) · w_amp(l) · w_consist(l)

w_amp     = min(1, A_T / 200 µV)
w_consist = min(1, 10 ms / max(spread_alg, 10 ms))
w_SQI     = 各 SQI 归一化后的几何平均（任一项不达标 → w_SQI = 0）
```

**Lead prior weight `w_prior`** [Project, for starting point use, should be recalibrated on own data]:

| Lead | Weight | Reason |
|---|---|---|
| V2, V3 | 1.00 | T has the largest amplitude and optimal SNR |
| V4, V5 | 0.95 | Large amplitude and stable shape; V5 is the preferred lead for clinical QT |
| II | 0.90 | Preferred lead for clinical QT, typical T morphology |
| V6 | 0.85 | |
| I | 0.80 | Medium range |
| aVF | 0.75 | |
| aVR | 0.70 | The polarity needs to be flipped first (multiplied by −1) |
| V1 | 0.60 | Often biphasic, U wave is easy to mix in |
| III, aVL | 0.55 | The amplitude is often too small and the direction is unstable |
| **RMS(Derived)** | 1.00 | |
| **PC1(Derived)** | 1.00 | |
| **SV (derived)** | 0.90 | Highly correlated with the rest to avoid overweighting |

**Hard elimination rules** (do not enter fusion):
- `A_T < 50 µV`
- `SNR_T < 6 dB`
- Any general SQI is not up to standard
- The T wave in this lead is indistinguishable (wavelet mode maxima <2)

<a id="54-稳健融合算法"></a>
### 5.4 Robust fusion algorithm

**Prerequisite**: All leads must first be aligned according to the **same global R peak** so that `Toff_l` of each lead falls on the same time axis. Fusion is the offset from the R peak `d_l = Toff_l − R`.

```
输入：{ d_l }, { w_l },  l ∈ 保留导联集 L

Step 1  初始估计
        m₀ = weighted_median(d, w)

Step 2  稳健尺度
        MAD = 1.4826 · median( |d_l − m₀| )
        σ̂  = max(MAD, 3 ms)                      # 下限防退化

Step 3  内点判定
        Inliers = { l : |d_l − m₀| ≤ max(3·σ̂, 15 ms) }      【工程】

Step 4  Huber IRLS（仅用内点），c = 1.345·σ̂
        repeat (最多 5 次，或位移 < 0.5 ms 时停):
            r_l  = d_l − m
            u_l  = w_l · (1 if |r_l| ≤ c else c/|r_l|)
            m    = Σ u_l·d_l / Σ u_l

Step 5  输出
        Toff_global = R + m
        标准误 SE   = σ̂ / sqrt( Σ_{Inliers} w_l )
        95% CI      = ± 1.96 · SE
```

**Two definitions of "global T endpoint"** - must be explicitly selected and fixed in the document:

| Definition | Calculation | Applicable |
|---|---|---|
| **Robust Center** | The above IRLS results `m` | Alignment with human interpretation (II/V5), algorithm evaluation, conventional feature extraction |
| **Latest repolarization** | **P85 quantile** of `d_l` in the interior point set (**Do not use max**) | In line with the physiological/standard definition of "global QT = earliest depolarization → latest repolarization" |

Taking `max` directly is extremely sensitive to a single noisy lead and is not available in engineering; P85 is a practical compromise between accuracy and physiological definition.

In the same way, the global `QRSon` takes the **P15 quantile** (not min) of the interior point, and the global `J` takes the **P85 quantile** of the interior point.

<a id="55-融合结果质控门限-工程"></a>
### 5.5 Fusion result quality control threshold [Project]

| Check items | Threshold | Disposal |
|---|---|---|
| Number of points | < 4 | This shot/record is invalid |
| interior point MAD | > 20 ms | lower confidence |
| Deviation between derived lead (RMS/PC1/SV) and fusion value | > 25 ms | Alarm, manual review |
| The three derived leads are extremely different from each other | > 30 ms | Alarm |
| Global QT | < 250 ms 或 > 700 ms | Physiologically unreliable, invalid |
| QTc (Fridericia) | < 300 ms 或 > 600 ms | Review |

---

<a id="6-t-波形态特征"></a>
## 6. T wave morphological characteristics

All features are calculated on the **median beat**; when variability is required, they are calculated separately on a beat-by-beat basis.

<a id="61-单导联时域特征"></a>
### 6.1 Single-lead time domain characteristics

| Characteristics | Definition | Units | Reference Range |
|---|---|---|---|
| `T_amp` | `y(Tpk) − B` | µV | V2–V5 Normal 200–900; limb conduction 100–400 [Clinical] |
| `T_area` | `Σ (y(i) − B) / fs`, i ∈ [Ton, Toff] | µV·s | Polarity sensitive |
| `T_absarea` | `Σ \|y(i) − B\| / fs` | µV·s | |
| `T_dur` | `Toff − Ton` | ms | Normal **150–250 ms** [Clinical] |
| `TpTe` | `Toff − Tpk` | ms | Normal **< 100 ms**（V5）；> 110 ms Abnormal [Clinical] |
| `TpTe/QT` | Ratio | — | Normal **0.15–0.25**; > 0.28 Abnormal [Clinical] |
| `T_asc` / `T_desc` | `Tpk − Ton` / `Toff − Tpk` | ms | |
| `SymR_time` | `T_asc / T_desc` | — | Normal T asymmetry, typical **0.5–0.7**; approaching 1.0 indicates ischemia/hyperkalemia [Engineering] |
| `SymR_slope` | \|Maximum slope of ascending branch\| / \|Maximum slope of descending branch\| | — | Normal **0.5–0.8** [Engineering] |
| `T_flat` | `T_amp / T_dur` | µV/ms | Flatness |

<a id="62-形状矩在-ton-toff-内以-yb-归一化为概率密度"></a>
### 6.2 Shape moment (within `[Ton, Toff]`, normalized to probability density by `\|y−B\|`)

```
p(i) = |y(i) − B| / Σ|y − B|
µ₁ = Σ i·p(i)                       # 重心时刻
m_k = Σ (i − µ₁)^k · p(i)
Skew = m₃ / m₂^1.5                  # 正常窦性 T 典型 0.2 ~ 0.6   【工程】
Kurt = m₄ / m₂²                     # 正常典型 2.0 ~ 3.0          【工程】
```

<a id="63-切迹与多相判定"></a>
### 6.3 Notch and polyphase determination

On `x_t` (low pass **15 Hz**, must be sufficiently low pass, otherwise noise creates false extremes):

```
Notch_count = [Ton, Toff] 内一阶导数过零点数 − 1
判定为切迹 T：存在次级极值，其幅度 ≥ 0.10 · T_amp 且距主峰 ≥ 30 ms   【工程】
判定为双相 T：存在跨越基线的极性反转，反向峰幅度 ≥ 50 µV           【工程】
```

<a id="64-多导联空间特征仅多导联可得"></a>
### 6.4 Multi-lead/spatial features (only available for multi-leads)

| Characteristics | Definition | Reference range |
|---|---|---|
| **PCA ratio** | `λ₂/λ₁`, T window 8-lead SVD singular value square ratio | Normal **< 0.20**；> 0.30 prompts repolarization abnormality [Engineering] |
| **TWR** (T-wave Residuum) | `sqrt(Σ_{i=4..8} λ_i)`, residual energy beyond the first 3 principal components | Normal low value; relative dimension, self-calibration required |
| **TMD** (T-wave Morphology Dispersion) | The mean value of the angle between the reconstructed vectors of the T wave in each lead on the first 2 principal component planes | Normal **< 30°**；> 50° Abnormal [Literature] |
| **TCRT** (Total Cosine R-to-T) | Cosine of the angle between the QRS principal vector and the T principal vector (in the VCG principal component space) | Normal **> 0.5**; < 0 significant abnormality [Literature] |
| **Space QRS-T angle** | The three-dimensional angle between the QRS mean vector and the T mean vector in VCG | Normal **< 75°**；临界 75–105°；异常 **> 105°** [Clinical] |
| **T ring planarity** | T ring residual of the best-fitting plane / ring length | The smaller, the more planar |
| **T ring area/eccentricity** | The area of the ring on the main plane and the ratio of the major and minor axes | |

**Space QRS-T included angle calculation**:

```
QRS 平均向量 = (1/N₁)·Σ_{k∈[QRSon,J]} [X(k), Y(k), Z(k)]
T   平均向量 = (1/N₂)·Σ_{k∈[Ton,Toff]} [X(k), Y(k), Z(k)]
angle = arccos( (v_QRS · v_T) / (|v_QRS|·|v_T|) ) · 180/π
```

<a id="65-逐拍动态特征需逐拍定界"></a>
### 6.5 Beat-by-beat dynamic characteristics (beat-by-beat delimitation required)

**T-Wave Alternating (TWA) - Spectral Method**:

```
1) 取连续 128 拍（必须窦性、RR 变异 < 10%）
2) 对 T 窗内每个时间样点，构造跨拍序列（长度 128）
3) 加 Hanning 窗，FFT
4) TWA 幅度 = 0.5 Hz 处（Nyquist，即每两拍交替）的谱峰
5) 噪声估计 = 0.43–0.47 Hz 频段的平均功率
6) K-score = (峰值 − 噪声均值) / 噪声标准差
   判阳性：K ≥ 3.0 且 TWA 电压 ≥ 1.9 µV        【文献】
```

**MMA (modified moving average) method**: Do an exponential moving average of odd and even beats respectively, update coefficient **1/8** (that is, the new beat weight is 12.5%) [Literature], TWA = the maximum absolute difference of the odd and even average beats in the ST-T segment. The anti-noise is better than the spectral method and does not require strict sinus law.

**QT variability**: `QTVI = log10[ (QTv/QTm²) / (RRv/RRm²) ]`, normal is about **−1.9 ~ −0.9** [Literature].

---

<a id="7-st-段特征"></a>
## 7. ST segment characteristics

<a id="71-测量点定义"></a>
### 7.1 Definition of measuring points

All ST levels are relative to the isoelectric baseline `B` of the same beat (§2.2).

```
STj   = y(J)        − B
ST20  = y(J + 20ms) − B
ST40  = y(J + 40ms) − B
ST60  = y(J + 60ms) − B
ST80  = y(J + 80ms) − B
```

**Heart rate adaptive measurement point** (exercise test practice) [Literature]:

| HR (bpm) | Main measuring point |
|---|---|
| < 100 | J + 80 ms |
| 100–110 | J + 72 ms |
| 110–120 | J + 64 ms |
| > 120 | J + 60 ms |

Or use the continuous formula: `t_meas = J + min(80 ms, 0.10·RR)`.

**Safety margin**: `t_meas` must be earlier than `Ton − 20 ms`, otherwise the ascending branch of the T wave will be measured instead of the ST segment. When the heart rate is too fast, the time is shortened to J+40 ms and marked first.

<a id="72-斜率与形状"></a>
### 7.2 Slope and shape

**ST slope**: Perform least squares linear regression on `[J + 20 ms, J + 80 ms]` and take the slope `k_ST` (µV/ms, equivalent to mV/s).

Classification threshold [Project]:

| Form | `k_ST` |
|---|---|
| Rising slope type | > +0.5 µV/ms |
| Horizontal type | −0.5 ~ +0.5 µV/ms |
| Downslope type | < −0.5 µV/ms |

**ST curvature (concave/convex)**: Quadratic fit `y = a·t² + b·t + c` on `[J, J + 100 ms]`

- `a > 0`: concave up - typical form of early repolarization and pericarditis
- `a < 0`: convex up - typical morphology of acute myocardial injury
- Judgment threshold: `|a| > 0.02 µV/ms²` is used to determine the shape, otherwise it is judged as linear [Engineering]

**ST Area**: `ST_area = Σ (y(i) − B) / fs`, `i ∈ [J, Ton]`, unit: µV·s.

**ST-T combined quantity**:
- `ST_T_area`: `[J, Toff]` points
- `ST_T_ratio = STj / T_amp`
- `ST_integral_norm = ST_area / (Ton − J)` (average ST level)

<a id="73-临床参考阈值仅作特征标定参考不作诊断输出"></a>
### 7.3 Clinical reference threshold (only for feature calibration reference, not for diagnostic output)

**ST elevation** (at J point, ≥2 anatomically adjacent leads required) [Clinical, Fourth Edition Universal Definition of Myocardial Infarction]:

| Lead | Population | Threshold |
|---|---|---|
| V2–V3 | Male ≥40 years old | ≥ 200 µV |
| V2–V3 | Male <40 years old | ≥ 250 µV |
| V2–V3 | Females (all ages) | ≥ 150 µV |
| All other leads | All | ≥ 100 µV |
| V3R–V4R | All | ≥ 50 µV (Male <30 years old ≥100 µV) |
| V7–V9 | All | ≥ 50 µV |

**ST depression**: new horizontal or downsloping type **≥50 µV**, ≥2 adjacent leads [Clinical]. Commonly used thresholds for exercise testing are **≥ 100 µV** horizontal/downslope type.

<a id="74-st-向量vcg-域"></a>
### 7.4 ST vector (VCG domain)

```
ST 向量 = [X(t_meas), Y(t_meas), Z(t_meas)] − [X_B, Y_B, Z_B]

幅值   = |v_ST|
方位角 azimuth   = atan2(Z, X) · 180/π      # 水平面
仰角   elevation = atan2(Y, sqrt(X²+Z²)) · 180/π
```

The information content of the ST vector direction for locating the ischemic area is much higher than the single-lead threshold judgment.

<a id="75-多导联-j-点融合"></a>
### 7.5 Multi-lead J-point fusion

J-point error is **linearly propagated** to all ST features and is the primary source of error in ST analysis.

- Each lead independently determines `J_l` (§2.1), unified to the global R peak time axis
- Use the same robust process as in §5.4, but **output the P85 quantile** of the interior point (the end time of different leads QRS is different, and the global J should correspond to the latest one)
- Quality control: Internal point range > 15 ms → All ST features are downgraded

---

<a id="8-精度预算与验证"></a>
## 8. Accuracy budget and verification

<a id="81-cse-库容差标准2sd-上限文献"></a>
### 8.1 CSE library tolerance standard (2SD upper limit) [Literature]

| datum point | tolerance |
|---|---|
| P onset | 10.2 ms |
| P offset | 12.7 ms |
| QRS onset | 6.5 ms |
| QRS offset | 11.6 ms |
| **T offset** | **30.6 ms** |

The T end point has the widest tolerance, which shows that it is the most difficult point in the entire process.

<a id="82-验证数据集"></a>
### 8.2 Validation data set

| Dataset | Content | Purpose |
|---|---|---|
| **QTDB** (PhysioNet QT Database) | 105 2-lead records, about 3600 beats, annotated by 2 cardiologists | T delimitation master datum |
| **CSE Multilead** | Multi-lead simultaneous annotation | Multi-lead fusion verification, tolerance benchmarking |
| **LUDB** | 200 12-lead, full-lead P/QRS/T wave-by-wave delimitation annotation | **Multi-lead method preferred** |
| **PTB-XL** | Large-scale 12-channel, including diagnostic labels | Morphological feature distribution statistics, robust stress testing |
| **European ST-T Database** | 90 items, ST and T changes are annotated beat by beat | ST feature verification |

<a id="83-报告规范"></a>
### 8.3 Reporting specifications

- `mean ± SD` reporting **signed error** (exposed system bias), cannot only report MAE
- Stratified reports: stratified by T amplitude (<100 / 100–200 / >200 µV), stratified by morphology (monophasic/biphasic/notch/inversion), stratified by heart rate
- Also reports **detection rate** (Se/PPV): missed beats cannot be silently eliminated from error statistics
- Report each lead separately, and then report the fusion result - fusion must bring about a visible decrease in SD, otherwise there will be problems with the fusion logic

<a id="84-目标性能参考"></a>
### 8.4 Target Performance Reference

| Item | Single lead (good lead) | Multi-lead fusion target |
|---|---|---|
| T offset error SD | 15–20 ms | **< 12 ms** [Engineering] |
| T peak error SD | 6–10 ms | < 6 ms |
| J point error SD | 8–12 ms | < 8 ms |
| T detection rate | > 98% | > 99.5% |

---

<a id="9-常见误差来源清单"></a>
## 9. List of common error sources

| # | Error Sources | Typical Magnitudes | Avoidance |
|---|---|---|---|
| 1 | **U wave mixing/T–U fusion** | 20–60 ms | §4.7 Judgment; forced to be enabled in cases of low potassium, slow pulse, and long QT |
| 2 | **ST distortion due to 0.5 Hz high pass** | 50–100 µV | With median filtering or 0.05 Hz zero phase |
| 3 | **Systematic bias between methods** (tangent vs threshold) | 10–20 ms | Lock a single method and parameters throughout the process |
| 4 | **Equal Wire Selection** | 20–50 µV | PR Segment Median + Flatness Check |
| 5 | **Use tangent method for low amplitude T** | > 40 ms | Disable tangent method for `A_T < 100 µV` |
| 6 | **J point error propagated to ST** | Proportional to slope | Multi-lead P85 fusion; degraded at wide QRS |
| 7 | **PVC/fusion wave mixed into the median beat** | Undefined, can be extremely large | Morphological clustering must be before the median beat |
| 8 | **High Heart Rate P-on-T** | 30–80 ms | Search window truncated to `nextQRSon − 120 ms` |
| 9 | **The right endpoint of the search window is too short** | Systematically early | Use median RR instead of instantaneous RR |
| 10 | **Low sample rate raster quantization** | ±2–4 ms | Interpolated to 1000 Hz |
| 11 | **Atrial fibrillation RR violent variation** | Window collapse | RR uses the 5-beat median; in atrial fibrillation, the window is recalculated beat by beat |
| 12 | **Lead polarity is not normalized (aVR)** | Shape determination error | aVR is uniformly multiplied by −1 before entering fusion |

---

<a id="10-参数速查表"></a>
## 10. Parameter quick lookup table

| Parameters | Symbols | Recommended values | Category |
|---|---|---|---|
| Sampling rate | fs | 500 Hz (QT precision 1000 Hz) | Engineering |
| Baseline median filter window | W1 / W2 | 200 ms / 600 ms | Literature |
| Zero-phase high-pass cutoff | fc | 0.05 Hz, Butterworth order 4 | Literature |
| T branch low pass | — | 20–25 Hz (15 Hz for U wave discrimination) | Engineering |
| Amplitude Branch Low Pass | — | 40 Hz | Engineering |
| Beat alignment search range | — | ±30 ms | Engineering |
| Cluster correlation threshold | ρ | 0.90 | Engineering |
| Minimum number of median beats | — | 8 | Engineering |
| Wavelet scale (T) | — | 2⁴ @500Hz; 2⁵ @1000Hz | Literature |
| Wavelet noise threshold | ε_T | max(0.25·RMS(w4), 0.125·max\|w4\|) | Literature |
| T onset threshold | ξ_on | 0.25 | Literature |
| T offset threshold | ξ_off | 0.40 (0.50 at low amplitude) | Literature |
| Boundary maximum search distance | — | Toff ≤ 160 ms; Ton ≤ 120 ms | Engineering |
| Trapezoidal method reference point | x_r | x_m + 160 ms | Engineering |
| Minimum T amplitude for tangent method | — | 100 µV | Engineering |
| Threshold method scaling | — | max(0.10·A_T, 15 µV) | Engineering |
| Cumulative area ratio | — | 0.98 | Engineering |
| Gaussian fitting boundary coefficient | k | 2.5σ | Engineering |
| Goodness of Fit Threshold | R² | 0.95 | Engineering |
| T search window start | — | J + max(60 ms, 0.04·RR) | Engineering |
| T search window stop | — | J + min(0.62·RR, 500 ms) | Engineering |
| Lead Hard Cull Amplitude | A_T | < 50 µV | Engineering |
| Lead Hard Culling SNR | SNR_T | < 6 dB | Engineering |
| Upper limit of cross-beat stability | σ_beat | 20 ms | Engineering |
| Interior point determination | — | max(3·MAD, 15 ms) | Engineering |
| MAD scale lower limit | — | 3 ms | Engineering |
| Huber's constant | c | 1.345·σ̂ | Literature |
| Minimum number of internal points | — | 4 | Engineering |
| "Latest repolarization" quantile | — | P85 (**without max**) | Engineering |
| SV method threshold | — | 0.15·max(SV) | Engineering |
| ST main measuring point | — | J + min (80 ms, 0.10·RR) | Literature |
| ST slope window | — | [J+20 ms, J+80 ms] | Engineering |
| ST morphological boundary slope | k_ST | ±0.5 µV/ms | Engineering |
| ST curvature judgment threshold | \|a\| | 0.02 µV/ms² | Engineering |
| kSQI Passed | — | > 5 | Literature |
| pSQI Pass | — | 0.5–0.8 | Literature |
| basSQI pass | — | > 0.95 | Literature |
| bSQI Passed | — | > 0.90 | Engineering |

---

<a id="11-完整流程伪代码"></a>
## 11. Complete process pseudo code

```python
def delineate_12lead(sig12, fs):
    # ---------- 1. 前处理 ----------
    if fs < 500: sig12 = resample(sig12, 1000); fs = 1000
    x = remove_baseline_median(sig12, fs, w1=0.200, w2=0.600)
    x = remove_artifacts(x, fs)                  # 起搏尖峰/饱和/平线
    x_qrs = bandpass(x, 5, 15, fs)
    x_fid = lowpass_zerophase(x, 40, fs)
    x_t   = lowpass_zerophase(x, 22, fs)
    x[aVR] *= -1                                 # 极性归一

    # ---------- 2. 全局 R 峰 ----------
    R = fuse_qrs_detections(x_qrs, fs)           # 多导联投票
    RR = median_filter(diff(R), 5)

    # ---------- 3. 中位拍 ----------
    clusters = cluster_beats(x_qrs, R, rho=0.90, dwidth=0.025)
    med = {l: median_beat(x_t[l], clusters.main) for l in LEADS}
    if len(clusters.main) < 8: flag('LOW_BEAT_COUNT')

    # ---------- 4. 基准点与基线 ----------
    QRSon = {l: wavelet_qrs_onset(x_fid[l]) for l in LEADS}
    J     = {l: wavelet_j_point(x_fid[l])   for l in LEADS}
    B     = {l: isoelectric(x[l], QRSon[l], RR) for l in LEADS}
    QRSon_g = robust_quantile(QRSon, weights, q=0.15)
    J_g     = robust_quantile(J,     weights, q=0.85)

    # ---------- 5. 派生导联 ----------
    rms  = rms_lead(x_t, B, INDEP8)
    pc1  = pca_first_component(x_t, INDEP8, T_window)
    svel = spatial_velocity(inverse_dower(x_t), fs)

    # ---------- 6. 逐导联定界 ----------
    res = {}
    for l in LEADS + ['RMS', 'PC1', 'SV']:
        win  = t_search_window(J[l], RR)
        wav  = wavelet_delineate_T(med[l], win, fs,
                                   xi_on=0.25, xi_off=0.40)
        if wav is None: res[l] = None; continue
        A_T  = abs(wav.Tpk_amp - B[l])
        U    = detect_u_wave(med[l], wav.Toff, fs)
        res[l] = select_method(wav, med[l], B[l], A_T, U, fs)   # §4.8
        res[l].sqi = compute_sqi(x[l], med[l], A_T, fs)

    # ---------- 7. 筛选 + 稳健融合 ----------
    keep = [l for l in res
            if res[l] and res[l].sqi.pass_all
            and res[l].A_T >= 50 and res[l].snr_T >= 6]
    if len(keep) < 4: return FAIL('INSUFFICIENT_LEADS')

    w = {l: w_prior[l] * res[l].sqi.score
            * min(1, res[l].A_T/200)
            * consistency_weight(res[l]) for l in keep}

    Toff_g, ci, inliers = robust_fuse([res[l].Toff for l in keep], w)   # §5.4
    Ton_g,  _,  _       = robust_fuse([res[l].Ton  for l in keep], w)

    # ---------- 8. 质控 ----------
    qc = quality_check(inliers, Toff_g, res['RMS'], res['PC1'], res['SV'])

    # ---------- 9. 特征提取 ----------
    feats = {}
    feats.update(t_morphology(med, B, Ton_g, Toff_g, res))       # §6
    feats.update(spatial_features(inverse_dower(x_t), QRSon_g,
                                  J_g, Ton_g, Toff_g))          # §6.4
    feats.update(st_features(med, B, J, RR, Ton_g))              # §7

    return Result(QRSon_g, J_g, Ton_g, Toff_g, feats, ci, qc)
```

---

<a id="12-实施建议"></a>
## 12. Implementation recommendations

1. **Make the baseline version first**: median beat + wavelet delimitation + SQI weighted median fusion. Quantify error distribution on LUDB and QTDB (stratified reporting). This step can achieve T offset SD ≈ 15 ms.
2. **Add trapezoidal method and decision tree**: Low amplitude T is the main source of remaining large errors. After adding §4.2 and §4.8, SD can usually be reduced to less than 12 ms.
3. **Finally add U-wave processing and derived lead cross-check**: The main contribution is to suppress the long tail (large error >50 ms), which has limited improvement in SD but significant improvement in the worst case.
4. **All `【工程】` thresholds must be recalibrated on own data**, especially lead prior weights, SQI thresholds, `ξ_off`, and trapezoidal methods `x_r`. Different collection equipment, electrode positions, and crowd composition will significantly shift these optimal values.
5. **Parameters cannot be changed midway once determined**. If changes are necessary, all historical data must be recalculated, otherwise the longitudinal comparison will be invalid.
