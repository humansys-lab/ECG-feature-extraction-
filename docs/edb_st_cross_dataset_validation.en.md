<!-- i18n-nav -->
[中文](edb_st_cross_dataset_validation.md) | [English](edb_st_cross_dataset_validation.en.md) | [日本語](edb_st_cross_dataset_validation.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecgfeat-在-edb-上的-st-测量验证ludb-之外第二个数据集"></a>
# ST measurement verification of ecgfeat on EDB (second data set besides LUDB)

Supporting document: [QTDB Tracing Verification](qtdb_cross_dataset_validation.en.md). The two verify different ability axes——
QTDB validates **Waveform Boundary Tracing**, and this article validates **ST Amplitude Measurement**.

<a id="1-运行配置"></a>
## 1. Run configuration

Use the existing `analyze_physionet_st.py` in the warehouse (no source code of `feature_extraction/ecgfeat` has been changed):

```bash
python analyze_physionet_st.py --dataset edb --max-events-per-record 3 \
    --no-plots --out-dir edb_st_evaluation
```

- Data set: European ST-T Database (EDB), **90 records were all run, and 0 cases failed**
- Sample: 215 expert annotated event peaks + 180 comparison windows
- Lead adaptation: same sparse lead adapter as QTDB (real channel goes into slot II/V2, rest is set to zero)
- Analysis window 12 s; single thread takes about 35 minutes

<a id="2-结果"></a>
## 2. Results

| indicators | native | hybrid |
|---|---:|---:|
| Event Peak ST Offset MAE | **0.099 mV** | 0.099 mV |
| bias bias (signed) | −0.021 mV | −0.015 mV |
| Correlation coefficient r (signed) | 0.878 | 0.872 |
| 0.10 mV Threshold Sensitivity | **0.642** | 0.641 |
| 0.10 mV Threshold Specificity | **0.994** | 0.989 |
| F1 | 0.780 | 0.777 |

There is essentially no difference between native and hybrid (hybrid has 6 less measurable events). Only native will be discussed below.

<a id="3-结论高特异度低灵敏度瓶颈是离散度而非偏倚"></a>
## 3. Conclusion: High specificity, low sensitivity, the bottleneck is dispersion rather than bias

**Very clean on control window. **The median |ST offset| for 180 control windows is only **0.004 mV**,
Only **0.6%** exceed 0.10 mV - ecgfeat almost does not falsely report ST changes, and the specificity is 99.4%.

**Systematic underestimation on the event window. **Expert Notes |Offset| Median 0.200 mV, ecgfeat only measured 0.149 mV.
The best-fit slope through the origin is **0.842**, which is approximately 16% amplitude compression.

**But compression is not the main reason. ** After correcting by this single global coefficient, the sensitivity only rises from **64.2% to 72.6%**,
Far from 100%; the corrected residual SD is 0.144 mV, which is larger than the 0.10 mV decision threshold itself.

Stratified by expert offset amplitude, the detection rate increases monotonically with the amplitude:

| Expert \|Offset\| | Number of events | Detection rate | ecgfeat Measured median |
|---|---:|---:|---:|
| 0.10–0.15 mV | 49 | 40.8% | 0.085 mV |
| 0.15–0.20 mV | 52 | 50.0% | 0.100 mV |
| 0.20–0.30 mV | 60 | 73.3% | 0.196 mV |
| ≥ 0.30 mV | 54 | **88.9%** | 0.273 mV |

**That is: ST events with clear amplitude (≥0.30 mV) can be reliably detected (88.9%),
Sensitivity loss is almost entirely concentrated in the critical interval of 0.10–0.20 mV**, which is precisely the beat-to-beat measurement noise
(residual SD 0.144 mV) can overwhelm the judgment.

<a id="4-必须附带的解读前提"></a>
## 4. Required prerequisites for interpretation

1. **Truth definition introduces independent error. **The true value of EDB is "the offset relative to the subject's own reference waveform",
   The evaluation script uses the first 30 seconds of recording to estimate the reference baseline on the ecgfeat side and then subtracts it. If the first 30 seconds itself already contains
   As ST changes, the reference is pulled away, and the calculated offset will shrink. Therefore the 0.842 compression of §3 with the 0.144 mV residual,
   **How much comes from the ST measurement of ecgfeat and how much comes from this reference estimation method cannot be separated in this experiment**.
   This is a limitation stated in the script's own documentation and is not a new issue this time.
2. **Only single channel ST amplitude verified. **EDB only has 2 channels, using sparse lead adapter,
   **Unable to verify location and diagnostic rules for 12-lead contiguous territory**.
3. The 0.10 mV threshold is a research comparison threshold, not a clinical diagnostic conclusion.
4. The control window is taken from outside the labeled events, but it does not guarantee absolute normality, so the 0.6% false positive rate is an upper bound estimate.

<a id="5-漏检案例分析77-个漏检事件到底怎么回事"></a>
## 5. Missed detection case analysis: What happened to the 77 missed detection incidents?

**77 out of 215 events (35.8%)** were missed. After breaking it down event by event, the conclusion is **different** from the rough impression of "overall compression of 16%".

<a id="51-不是均匀压缩是双峰"></a>
### 5.1 It’s not uniform compression, it’s double peaks

| | Expert true value median | ecgfeat measured median |
|---|---:|---:|
| **138 detected** | 0.200 mV | **0.202 mV** (almost identical) |
| **77 missed** | 0.150 mV | **0.060 mV** (only 40% of the true value was measured) |

**The measurement of ecgfeat on the detection event is basically unbiased. ** The so-called "16% global compression" is an illusion created by mixing these two groups:
The real situation is that most events are measured accurately, and about one-third of events are seriously underestimated. This determines the direction of repair -
Rather than multiplying all measurements by a factor, one should look for common characteristics among those thirds.

<a id="52-ecgfeat-自己的置信度信号识别不出这些失败关键"></a>
### 5.2 ecgfeat own confidence signal cannot identify these failures (critical)

| Indicators | Missed events | Detected events |
|---|---:|---:|
| Point J confidence level | **0.669** | 0.631 |
| Baseline Confidence | **0.862** | 0.821 |
| Consensus support number | 2.0 | 2.0 |
| IQR of beat-by-beat ST | **0.037 mV** | 0.058 mV |

**The confidence of missed events is higher than that of detected events, and the beat-to-beat dispersion is smaller. **
In other words, ecgfeat measured a low value** stably and "confidently" on these events.
Rather than measuring erratically. The consequences are very practical: **Downstream cannot rely on the existing confidence field to filter out this batch of results**,
Existing quality gating is completely insensitive to this failure mode. This is the most actionable discovery this time.

Other accompanying features: shorter missed events (median 132 s vs 267 s), lower heart rate (77.7 vs 92.0 bpm),
Fewer shots available (14 vs 18).

<a id="53-根因定位问题出在减参考基线这一步"></a>
### 5.3 Root cause location: The problem lies in the step of "reducing the reference baseline"

Prediction offset = event window ST − reference window ST. Split the two components (signed median):

| | Expert true value | Event window ST | Reference window ST | Prediction offset |
|---|---:|---:|---:|---:|
| Missed detection | −0.100 | **−0.088** | −0.042 | **−0.041** |
| Detection | −0.150 | −0.202 | −0.036 | −0.167 |

**The raw ST measurement of the missed event (−0.088) is actually very close to the expert true value (−0.100)**,
It is after subtracting the reference baseline (−0.042) that it drops to −0.041, falling below the 0.10 mV threshold.
**55% of missed events have the reference window itself |ST| ≥ 0.05 mV** - the reference window is not "clean",
It eats up about half of the signal.

Direction judgment is good: the predicted deviation is consistent with the expert direction **96.3%** (only looking at missed events, it is also 89.6%).
**The error is the amplitude, not the direction. **

<a id="54-但去掉减参考这一步并不能解决问题"></a>
### 5.4 But "removing the step of subtracting the reference" does not solve the problem

Since the reference baseline eats up the signal, what will happen if the original absolute ST level is used directly? Measured counterfactual:

| Scheme | MAE | bias | r | Sensitivity | Control false positive rate | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Current (Event − Reference) | **0.099** | −0.021 | **0.878** | 64.2% | **0.6%** | **0.780** |
| Original absolute ST (unreduced) | 0.114 | −0.058 | 0.860 | **76.3%** | **23.9%** | 0.777 |

**Sensitivity does go up to 76.3%, but the control false positive rate explodes from 0.6% to 23.9%, and F1 is almost unchanged (0.780 → 0.777). **

So the step of subtracting the reference is not a bug. It is exchanging sensitivity for huge specificity gains. Removing it just reduces the work.
Panning, the overall discrimination does not improve. The real bottleneck is the number in §3:
**Event-by-event residual SD 0.144 mV is larger than the 0.10 mV decision threshold**. Under this dispersion,
No single fixed threshold can achieve high sensitivity and high specificity at the same time.

**Conclusion: The missed detection cannot be simply attributed to the "reference estimation method of the evaluation script", nor can it be simply attributed to the "inaccurate measurement of ecgfeat". **
The direction that can be improved is to reduce the event-by-event dispersion (especially the selection of the reference window to avoid the period when ST has changed),
Rather than adjusting the threshold or removing the subtraction step.

<a id="6-与-qtdb-结论的呼应"></a>
## 6. Echo with the conclusion of QTDB

The two data sets point to the same personality: **ecgfeat is conservative - high accuracy/specificity, but sensitivity is a shortcoming. **

| | ecgfeat Strengths | ecgfeat Weaknesses |
|---|---|---|
| QTDB (tracing) | QRS detects Se 99.9%, T wave Se 95.7% | P wave Se 91.5% (NeuroKit2 96.7%) |
| EDB (ST measurement) | Specificity 99.4%, control median 0.004 mV | Sensitivity 64.2%, critical interval 0.10–0.20 mV, more than half missed |

The two shortcomings are "the report was not reported", not "the error was reported".
