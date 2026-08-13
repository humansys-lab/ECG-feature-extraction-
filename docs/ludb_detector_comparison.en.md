<!-- i18n-nav -->
[中文](ludb_detector_comparison.md) | [English](ludb_detector_comparison.en.md) | [日本語](ludb_detector_comparison.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ludb-三种-ecg-检测方法对比"></a>
# LUDB Comparison of three ECG detection methods

This tool compares the following methods on LUDB 1.0.1 expert annotations:

1. Project algorithm `ecgfeat`;
2. NeuroKit2;
3. BioSPPy.

The entry script is `compare_ludb_detectors.py` in the project root directory.

<a id="1-安装"></a>
## 1. Installation

```bash
cd /workspace/ecg_gemma

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ludb-compare.txt
```

The default data directory is:

```text
/workspace/ecg_gemma/data/lobachevsky-university-electrocardiography-database-1.0.1
```

The script accepts both the LUDB root directory above and the `data/` directory inside it.

<a id="2-运行命令"></a>
## 2. Run command

First do a quick verification on record 1, lead II:

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate

python compare_ludb_detectors.py \
  --records 1 \
  --leads II \
  --out-dir ludb_three_way_smoke
```

Run first 10 records, all 12 leads:

```bash
python compare_ludb_detectors.py \
  --limit 10 \
  --workers 2 \
  --out-dir ludb_three_way_first10
```

Run complete LUDB:

```bash
python compare_ludb_detectors.py \
  --dataset-dir /workspace/ecg_gemma/data/lobachevsky-university-electrocardiography-database-1.0.1 \
  --workers 4 \
  --out-dir /workspace/ecg_gemma/ludb_three_way_comparison
```

Match using ecgfeat QRS fiducial after local refinement of each lead:

```bash
python compare_ludb_detectors.py \
  --limit 10 \
  --workers 4 \
  --ecgfeat-r-source lead-fiducial \
  --out-dir ludb_three_way_first10_per_lead_r
```

The default evaluation script is still `--ecgfeat-r-source global`, which is used to keep historical results reproducible;
Lead-by-lead results must explicitly select `lead-fiducial`. Two running `summary.json` will
Record the actual peak source.

For the most stable timing results, use `--workers 1`. Parallel running is suitable for obtaining detection indicators faster, but process competition will affect the time consumption.

Compare only some methods or leads:

```bash
python compare_ludb_detectors.py \
  --methods ecgfeat,neurokit2 \
  --leads II,V1,V5 \
  --limit 10
```

View all parameters:

```bash
python compare_ludb_detectors.py --help
```

<a id="3-对比范围"></a>
## 3. Comparison range

The P wave, QRS complex, and T wave annotations for LUDB are given on a lead-by-lead basis, so:

- NeuroKit2 independently runs R-peak detection and waveform definition on each selected lead;
- BioSPPy operates independently on each selected lead;
- `ecgfeat` retains the native design: jointly detect the global R peak on 12 leads of a record, and then read the P/QRS/T boundary of each lead.

The public capabilities of the three methods are not exactly the same:

| Method | R peak | QRS start and end points | P peak/start and end points | T peak/start and end points |
|---|---:|---:|---:|---:|
| `ecgfeat` | Yes | Yes | Yes | Yes |
| NeuroKit2 | Yes | Yes | Yes | Yes |
| BioSPPy Public `ecg()` Process | Yes | No | No | No |

Therefore, BioSPPy only participates in the detection and positioning comparison of QRS/R peaks. The script does not speculate or falsify P/T sweep boundaries that are not provided by BioSPPy.

<a id="4-匹配和指标定义"></a>
## 4. Matching and indicator definitions

Each lead reads the P, QRS, and T triples of LUDB respectively:

```text
( onset, peak symbol, ) offset
```

Default match tolerance:

- R peak of QRS: 75 ms;
- P/T peak: 150 ms.

Can be modified using the following parameters:

```bash
--r-tolerance-ms 75
--wave-tolerance-ms 150
```

Events are matched one-to-one in chronological order. Dynamic programming first maximizes the number of TPs, and then minimizes the peak absolute error in the same TP solution to avoid repeated matching of a detection result and avoid missing matches between adjacent events due to simple greed.

The detection index is micro-averaged based on all "record × lead" wave events:

- `sensitivity = TP / (TP + FN)`;
- `precision` (PPV) `= TP / (TP + FP)`;
- `F1 = 2TP / (2TP + FP + FN)`.

Only the onset, peak, and offset of the wave are compared after successful peak matching. The positioning error is defined as:

```text
error_ms = (detected_sample - ground_truth_sample) / fs × 1000
```

Output per boundary:

- `bias_ms`: signed mean error;
- `sd_ms`: error standard deviation;
- `mae_ms`: mean absolute error;
- `median_ae_ms`: median absolute error;
- `p95_ae_ms`: 95th percentile of absolute error;
- `n`: Number of matches available for this boundary statistic.

LUDB There may be detection events at both ends that are not marked by experts. These events are still counted as FP according to the standard definition; if you need to ignore recording edges, you should modify the evaluation strategy after pre-defining and recording the rules, and cannot manually delete them after seeing the results.

<a id="41---restrict-to-annotated-span可选默认关闭"></a>
### 4.1 `--restrict-to-annotated-span` (optional, closed by default)

LUDB only marks the middle section of each 10-second record (for example, the mark interval of record 1 is 1.32 s–7.94 s,
Record 2 is 1.14 s–8.44 s). Heartbeats outside the range are **real heartbeats but have no reference labels**,
Count everything into FP at the default aperture - this measures annotation coverage, not detector accuracy,
And **punish all methods of doing the right thing**. In actual measurements, the PPVs of the three methods were all reduced to 0.63–0.84.
The GT of QRS is 21843, while each method detects 25400–25920. The difference is basically unlabeled marginal beats.

When this switch is enabled, scoring is only performed within the marked interval of each lead. The criteria are mechanical and do not require subjective choices:

> Discard detection events whose distance from the annotation interval exceeds the **matching tolerance**. Such events cannot by definition be associated with any
>True event pairing, so it's judged as FP just because there's no annotation there; detections within tolerance are **all kept**.

This rule applies identically to the three methods and does not change sensitivity and bounding error
(The set of truth events does not change, nor does the matching pair change) - only precision and F1 become interpretable.
The `det_outside_annotated_span` column in `detection_rows.csv` records the number of events dropped each time.

**Record an Integrity Statement**: The principles above this section require that the evaluation rules be "pre-defined". The switch is on for the first time
It was implemented in full volume (200 records) after seeing that the PPV was abnormally low, and did not meet the strict pre-registration order.
Therefore it remains off by default, and the results for both sets of calibers should be reported together, not just the numbers after they are turned on.
The criterion itself does not contain any methodological degrees of freedom, which is why it is still useful, but the reader should be aware of its origins.

<a id="5-输出文件"></a>
## 5. Output file

Each run will generate in `--out-dir`:

| Documentation | Content |
|---|---|
| `summary.csv` | Micro average indicators summarized by method and waveform |
| `summary_by_lead.csv` | Summary by method, waveform, lead |
| `record_summary.csv` | Summary of each record |
| `detection_rows.csv` | TP/FP/FN for each record, lead, method, waveform |
| `matched_events.csv` | Sample position and boundary error for each successful matching event |
| `runtime.csv` | Time consuming for each algorithm call |
| `runtime_summary.csv` | Time-consuming summary |
| `failures.csv` | Algorithm exception and traceback |
| `summary.json` | Configuration, version, metrics and number of failures |
| `comparison.png` | Comparison chart between F1 and positioning MAE |

`failures.csv` is empty indicating no call failure. In the default mode, when an algorithm call fails, all real events supported by the call are included in the summary by FN to avoid obtaining falsely high indicators by silently discarding failed samples; at the same time, the script eventually returns a non-zero exit code. Use `--fail-fast` to stop directly on first failure.

<a id="6-运行时间的正确理解"></a>
## 6. Correct understanding of running time

`ecgfeat` processes a complete 12-lead record at one time, and the time unit is `record_12lead`; NeuroKit2 and BioSPPy operate on a single lead, and the time unit is `single_lead`. `runtime_summary.csv` will write this unit out explicitly.

Therefore, `mean_seconds` of the three cannot be directly compared as the same workload. To estimate the serial time it takes for an external library to process a 12-lead record, you can sum the call times of 12 leads for the same record; to do a strict performance benchmark, you should use a single process, warm up, fix the hardware, and run repeatedly.

<a id="7-在同一条-ecg-上画-ecgfeat-与-neurokit2-标注"></a>
## 7. Draw ecgfeat and NeuroKit2 annotations on the same ECG

`plot_ecgfeat_neurokit_comparison.py` will cause both detectors to read the same LUDB
Record and display the results on the same original ECG sample and timeline.

Draw the complete record:

```bash
python plot_ecgfeat_neurokit_comparison.py 1 \
  --lead II \
  --out ludb_ecgfeat_neurokit_plots/record_1_II_full.png
```

Press the Nth QRS of LUDB to mark the zoomed-in single shot:

```bash
python plot_ecgfeat_neurokit_comparison.py 8 \
  --lead V2 \
  --beat 3 \
  --out ludb_ecgfeat_neurokit_plots/record_8_V2_beat3.png
```

Display the 4 consecutive heartbeats corresponding to the 2nd–5th QRS:

```bash
python plot_ecgfeat_neurokit_comparison.py 8 \
  --lead V2 \
  --beat-range 2 5 \
  --out ludb_ecgfeat_neurokit_plots/record_8_V2_beats2-5.png
```

`--before-sec` and `--after-sec` control the first QRS before and the last QRS respectively
The context to keep after that, defaults to 0.45 seconds and 0.70 seconds.

Select a fixed time period:

```bash
python plot_ecgfeat_neurokit_comparison.py 1 \
  --lead II \
  --start-sec 1.5 \
  --end-sec 4.5
```

Do not display LUDB Expert reference:

```bash
python plot_ecgfeat_neurokit_comparison.py 1 --lead II --no-ludb
```

The meaning of each line of the output diagram:

1. The first row superimposes the P/R/T peaks of the two algorithms on the same ECG; the solid marks are
   `ecgfeat`, the hollow mark is NeuroKit2, and the cross is LUDB;
2. The second line shows the P/QRS/T interval and boundary of `ecgfeat`;
3. The third line shows the P/QRS/T interval and boundary of NeuroKit2;
4. By default, the fourth line displays the LUDB expert boundary, which is omitted when using `--no-ludb`.

In the boundary panel, the dotted line is onset, the dotted line is offset, and the semi-transparent area is
onset–offset interval. All panels plot the same original ECG; the two detectors are still
Use respective internal preprocessing. The terminal will also print the relative LUDB within the current visible time window.
TP, FP, FN, F1 and peak MAE.

<a id="完整-12-导联10-秒对比"></a>
### Full 12-lead, 10-second comparison

`plot_ludb_12lead_10s_comparison.py` will run the entire 12-lead record at once and
The complete 0–10 sec signal for each lead is arranged in 12 rows. The project algorithm is run only once on a record,
The presence and number of heart beats are determined by multi-lead detection; `ecgfeat` QRS in the figure are marked by default
Use independently saved lead-by-lead positioning `hybrid-prominence`. NeuroKit2 then lead by lead
Independent testing.

```bash
# 单条记录
python plot_ludb_12lead_10s_comparison.py 1

# LUDB 前 10 条记录
python plot_ludb_12lead_10s_comparison.py {1..10} \
  --out-dir ludb_12lead_10s_comparison
```

`--ecgfeat-r-source` can choose from four tags:

- `hybrid-prominence`: based on multi-lead heartbeat, in the original lead-by-lead fiducial
  Select the positive local peak with the largest prominence within 35 ms, and draw the default value;
- `lead-fiducial`: Original QRS fiducial for each lead;
- `positive-r`: Maximum positive R wave in the interval QRS for each lead;
- `global`: Multi-lead global R fiducial shared by all leads, also a detection evaluation script
  The default value continues to be used to keep historical metrics consistent.

`hybrid-prominence` writes independent `r_localized_index` and other fields without overwriting
`r_locs`, `qrs.peak`, `r_peak_index` or waveform boundaries, so it does not change the heart rate,
intervals, representative lead characteristics, and interpretation layer results.

Each record is generated by default:

1. Three-party P/R/T peak overlay page;
2. P/QRS/T boundary page of `ecgfeat`;
3. P/QRS/T boundary page of NeuroKit2;
4. LUDB Expert P/QRS/T boundary page;
5. Assemble the above four-page PDF.

If you do not need the LUDB expert page or PDF, you can use `--no-ludb` and `--no-pdf` respectively.

<a id="8-当前实现的边界"></a>
## 8. Boundaries of current implementation

- The comparison is about detection and waveform definition, not the accuracy of the final clinical diagnosis rule;
- The R peak of `ecgfeat` is a multi-lead consensus result, while the external library is a single-lead result, which reflects their native usage, but is not exactly the same input task;
- P/T matching uses the wave crest as the anchor point, and the boundary indicator only counts the matched waves;
- By default NeuroKit2 uses the `neurokit` R peak method and `dwt` bounding method, which can be switched through the command line;
- For results obtained by different tolerances, lead subsets or NeuroKit2 methods, the corresponding `summary.json` must be retained and no mixed comparisons are allowed.
