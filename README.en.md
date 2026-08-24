<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="ecg-gemma12-导联-ecg-特征提取与临床证据工程"></a>
# ECG Gemma: 12-lead ECG Feature Extraction and Clinical Evidence Engineering

This project provides an interpretable 12-lead ECG feature extraction pipeline and organizes measurements into structured evidence that can be used by rules engines, reporters, and MedGemma.

It's not just an R-peak detector. The current process includes signal preprocessing, quality control, pacing spike detection, multi-lead QRS detection, beat grouping, representative beat construction, P/QRS/T wave boundary point positioning, interval and cardiac axis measurement, atrial rhythm analysis, and clinical rules such as conduction, hypertrophy and ST-T.

> **Important Note**
>
> This project is software for research and engineering purposes, not a medical device. It cannot replace a doctor's interpretation, nor should it be directly used for clinical diagnosis or treatment decision-making.

<a id="1-项目结构"></a>
## 1. Project structure

```text
ecg_gemma/
├── feature_extraction/
│   ├── ecgfeat/                  # ECG 特征提取核心包
│   ├── README.md                 # 核心包的补充说明
│   └── pyproject.toml
├── ecgagent/                     # 诊断 Agent、证据与校验
├── webchat/                      # Web UI 与模型服务
├── scripts/                      # 维护、本地化与队列工具
├── tools/generation_conditions/  # 专项审计与探针
├── tests/                        # 单元、契约与回归测试
├── docs/                         # 设计、算法与验证文档
├── demo_feature_extraction.py    # 单条 WFDB .mat/.hea 记录演示
├── batch_extract_ecgfeat.py      # ECG 特征批处理
├── medgemma_ecg_core.py          # MedGemma 上下文与诊断流程
├── medgemma_runtime.py           # 共享 GPU/vLLM 运行时初始化
├── app.py                        # Gradio 界面入口
├── evaluate_*.py / validate_*.py # 数据集评估与验证入口
├── dataset -> data/010           # demo 默认读取的数据目录
├── data/                         # 本地原始/参考数据（Git 忽略）
├── medgemma-27b/                 # 本地 MedGemma 模型（Git 忽略）
├── qwen3.8-27b/                  # 本地 Qwen 模型（Git 忽略）
├── document/                     # 外部参考资料（Git 忽略）
├── requirements.txt              # 完整项目依赖
└── .gitignore                    # 本地大文件与生成物隔离规则
```

For a more complete description of entry classification, shared module boundaries, and maintenance conventions see
[Code Base Navigation and Maintenance Conventions](docs/codebase_guide.en.md).

For the cleanup scope, retained files, and ongoing rules, see the
[Project Code and File Organization Report](docs/project_organization_and_cleanup.en.md).

ECG Diagnostic Agent’s current dual-channel architecture, evidence contract, Compact/Legacy workflow,
For tool and verification design, see [ECGAgent complete design document](docs/ecg_agent_complete_design.en.md).

Core modules:

| Module | Function |
|---|---|
| `ecgfeat/api.py` | Complete pipeline entrance `ECGFeatureExtractor.extract()` |
| `ecgfeat/preprocess.py` | Resampling, de-baselining, notching and detection signal construction |
| `ecgfeat/quality.py` | Quality gating per lead and per measurement function |
| `ecgfeat/pacing.py` | Pacing spike detection and verification |
| `ecgfeat/qrs.py` | Multi-lead QRS/R peak detection |
| `ecgfeat/grouping.py` | Heartbeat shape grouping |
| `ecgfeat/representative.py` | Represents heartbeat alignment and fusion |
| `ecgfeat/delineate.py` | P/QRS/T Wave peak and boundary positioning |
| `ecgfeat/st_localization.py` | Robust PR baseline, J-point consensus and ST multi-point bypass measurements |
| `ecgfeat/features.py` | Single lead, grouping and global feature aggregation |
| `ecgfeat/atrial.py` | Sexual incident, QRST subtraction, AF/AFL evidence |
| `ecgfeat/clinical_rules/` | Rhythm, conduction, interval, hypertrophy, ischemia and other rules |
| `ecgfeat/export.py` | JSON Conversion and Structured Export |
| `medgemma_ecg_core.py` | Context, prompts and layered diagnostic process for MedGemma |

<a id="2-运行环境"></a>
## 2. Operating environment

Python 3.10 or higher is recommended.

<a id="仅运行-ecg-feature-extraction"></a>
### Only run ECG feature extraction

```bash
cd /workspace/ecg_gemma

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ./feature_extraction
python -m pip install matplotlib pytest
```

The `feature_extraction` package will install core dependencies NumPy and SciPy. Drawing requires Matplotlib.

<a id="完整项目依赖"></a>
### Complete project dependencies

```bash
python -m pip install -r requirements.txt
```

The complete dependencies include larger models and interface components such as `vllm`, `transformers`, Gradio. If you only perform ECG feature extraction, you do not need to install all dependencies.

When processing `.pt` files, you also need to install PyTorch separately according to the CPU/CUDA environment of the machine.

<a id="ludb-上比较-ecgfeatneurokit2-和-biosppy"></a>
### Compare ecgfeat, NeuroKit2 and BioSPPy on LUDB

The comparison tool will read the lead-by-lead expert annotation of LUDB and count the R peak, P/QRS/T wave detection rate and boundary error of matching events. The public ECG process for BioSPPy only provides R peaks, so only the QRS/R peak comparison is included.

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pip install -r requirements-ludb-compare.txt

# 快速验证：记录 1、II 导联
python compare_ludb_detectors.py --records 1 --leads II \
  --out-dir ludb_three_way_smoke

# 完整 LUDB、全部 12 导联
python compare_ludb_detectors.py --workers 4 \
  --out-dir ludb_three_way_comparison

# 前10条记录，使用每个导联自己的 ecgfeat QRS fiducial
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --ecgfeat-r-source lead-fiducial \
  --out-dir ludb_three_way_first10_per_lead_r

# 前10条记录，使用混合 prominence 定位
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --methods ecgfeat \
  --ecgfeat-r-source hybrid-prominence \
  --out-dir ludb_ecgfeat_first10_hybrid_prominence

# 前10条记录：混合 R 定位 + 双极性 T 峰细化
python compare_ludb_detectors.py --limit 10 --workers 4 \
  --methods ecgfeat \
  --ecgfeat-r-source hybrid-prominence \
  --ecgfeat-wave-source hybrid-peaks \
  --out-dir ludb_ecgfeat_first10_hybrid_waves
```

For complete method definitions, parameters, indicators and output file descriptions, see [LUDB Comparison of three detection methods](docs/ludb_detector_comparison.en.md).

Regarding absorbing the advantages of local positioning of NeuroKit2, improving the R/P/T/S wave detection of ecgfeat and
Robust ST/J measurement solution,
See [ecgfeat × NeuroKit2 hybrid improved design](docs/ecgfeat_neurokit_hybrid_improvement.en.md).

T-wave enhancement is on by default: Mallat quadratic splines are calculated in parallel using a dedicated 22.5 Hz branch
Wavelet and low-amplitude T-trapezoid area candidates, combining T-SQI, RMS/PC1 with
weighted-median/MAD/Huber multi-lead fusion, while outputting robust center and
P85 is the latest to repolarize. The QT consensus and local T offset are updated only when the fusion quality control passes.
Only modified under strict translead late-tail rescue. If you need historical regression, you can pass in
`enable_t_wave_refinement=False`.

Robust ST enhanced output is enabled by default and outputs PR segment baseline, P85 multi-lead consensus J point,
Window median of J/J+20/J+40/J+60/J+80 and RR/Ton adaptive points, and robust
Slope, quadratic curvature, trend and bump patterns. It does not cover the native ST field, nor does it change the clinical interpretation; if you need to do strict historical regression, you can
Pass in `enable_hybrid_st_measurement=False` when constructing the extractor.

Verify the ST-tagged records in local `dataset`:

```bash
# 扫描全部标签并对100条记录做标签级比较
PYTHONPATH=feature_extraction .venv/bin/python validate_dataset_st.py \
  --dataset-dir dataset \
  --out-dir dataset_st_validation \
  --workers 4

# 仅提取明确 ST 标签和 ST-T/心梗探索记录
PYTHONPATH=feature_extraction .venv/bin/python validate_dataset_st.py \
  --dataset-dir dataset \
  --out-dir dataset_st_candidates \
  --candidates-only
```

The script outputs candidate lists, record-by-record/lead-by-beat/beat-by-beat CSV, label consistency metrics, and clear
12-Lead Label Diagram of ST Tag Recording. Since this data only has record-level diagnostic tags, there is no
Lead-by-lead J-point or ST amplitude true value, the output cannot be regarded as clinical accuracy.

Superimpose the peak values ​​of the project algorithm and NeuroKit2 on the same ECG, and display the boundaries of the two in separate lines:

```bash
# 完整记录
python plot_ecgfeat_neurokit_comparison.py 1 --lead II

# 显示第 2–5 个 LUDB QRS 对应的连续心搏
python plot_ecgfeat_neurokit_comparison.py 8 --lead V2 --beat-range 2 5
```

Draw a fully documented 12-lead, 10-second comparison:

```bash
# 记录 1；生成 4 张 PNG 和一个四页 PDF
python plot_ludb_12lead_10s_comparison.py 1

# 批量绘制记录 1–10
python plot_ludb_12lead_10s_comparison.py {1..10}

# 如需复现旧图中的全局多导联 R 标记
python plot_ludb_12lead_10s_comparison.py 1 \
  --ecgfeat-r-source global
```

Plotting uses independent lead-by-lead prominence positioning by default (`hybrid-prominence`);
The number of beats is still determined by the multi-lead detection of ecgfeat. This bypass field does not replace the global
`r_locs`, legacy `qrs.peak` or QRS boundaries and therefore do not affect final measurements and interpretations.

<a id="3-输入约定"></a>
## 3. Enter the convention

Feature extractor expects:

- Standard 12-lead ECG;
- The array shape is `[12, n_samples]`, `[n_samples, 12]` is also accepted;
- The recommended lead sequence is `I, II, III, aVR, aVL, aVF, V1–V6`;
- The amplitude unit of the incoming signal is mV;
- Explicitly provide the original sample rate;
- Patient age and gender are optional metadata, but some clinical rules require them.

The demo script reads by default:

```text
dataset/
├── JS00001.hea
├── JS00001.mat
├── JS00002.hea
└── JS00002.mat
```

`.mat` should contain the `val` array of shape `[12, n_samples]`. The current demo converts according to `1000 ADC units/mV` by default; before accessing the new data set, you should confirm the actual gain, unit and lead sequence in each `.hea`.

<a id="4-快速运行"></a>
## 4. Run quickly

<a id="随机抽取一条记录"></a>
### Randomly select a record

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python demo_feature_extraction.py
```

<a id="抽取指定记录"></a>
### Extract specified records

```bash
python demo_feature_extraction.py JS00010
```

By default, it will be generated in the project root directory:

```text
JS00010_features.json
JS00010_report.txt
JS00010_ecg.png
JS00010_ecg_annotated.png
JS00010_ecg_report.png
```

<a id="保留逐拍逐导联完整特征"></a>
### Keep the complete characteristics of beat-by-beat and lead-by-lead

```bash
python demo_feature_extraction.py JS00010 --full-beats
```

By default, JSON will omit the largest `beat_features`. `--full-beats` is recommended only when debugging wave boundary positioning, auditing single-shot results, or training requires beat-by-beat data.

<a id="5-抽取-dataset-的前-10-条"></a>
## 5. Extract the first 10 items from the dataset

The following command sorts naturally by record number, extracts the top 10 records, and moves the generated files to a separate directory:

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate

output_dir="data/dataset_first10_ecgfeat_out"
mkdir -p "$output_dir"

find -L dataset -maxdepth 1 -type f -name '*.hea' -printf '%f\n' \
  | sed 's/\.hea$//' \
  | sort -V \
  | head -n 10 \
  | while read -r record_id; do
      python demo_feature_extraction.py "$record_id"

      for suffix in \
        features.json \
        report.txt \
        ecg.png \
        ecg_annotated.png \
        ecg_report.png; do
        artifact="${record_id}_${suffix}"
        if [ -f "$artifact" ]; then
          mv "$artifact" "$output_dir/"
        fi
      done
    done
```

The top 10 results currently generated are at:

```text
data/dataset_first10_ecgfeat_out/
```

<a id="6-pt-批量抽取"></a>
## 6. `.pt` batch extraction

`batch_extract_ecgfeat.py` does not process WFDB `.mat/.hea`, but instead processes the following directory:

```text
all_diseases_pt100/
├── norm/
│   ├── generated.pt
│   ├── real.pt
│   └── metadata.pt
├── ami/
│   ├── generated.pt
│   └── real.pt
└── ...
```

Each tensor can be:

- `[n_samples, 12, n_points]`
- `[n_samples, n_points, 12]`
- Single `[12, n_points]`
- Single `[n_points, 12]`

First use a small number of samples to perform smoke test:

```bash
python batch_extract_ecgfeat.py \
  --input-dir all_diseases_pt100 \
  --output-dir all_diseases_ecgfeat \
  --sampling-rate 100 \
  --sources generated,real \
  --limit 2 \
  --annotated-plots
```

Formal batch processing:

```bash
python batch_extract_ecgfeat.py \
  --input-dir all_diseases_pt100 \
  --output-dir all_diseases_ecgfeat \
  --sampling-rate 100 \
  --sources generated,real \
  --skip-existing
```

To retain beat-by-beat details in JSON, add:

```bash
--include-beat-features
```

The batch-processed `.pt` feature file always retains complete beat-by-beat data; this parameter only controls whether JSON retains `beat_features`.

## 7. Python API

```python
import numpy as np

from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.export import prepare_json_export, to_dict
from ecgfeat.models import PatientMeta

# 形状：[12, n_samples]；单位：mV
ecg = np.random.randn(12, 5000) * 0.02
fs = 500

extractor = ECGFeatureExtractor(
    fs_internal=500,
    mains_freq=50,
)

features = extractor.extract(
    ecg,
    fs=fs,
    meta=PatientMeta(age=45, sex="Male"),
)

print(features.global_features)

payload = prepare_json_export(
    to_dict(features),
    include_beat_features=False,
)
```

If the data comes from ADC integer values, each record should first be converted to mV using its own gain:

```python
ecg_mv = ecg_adc.astype(np.float64) / gain_units_per_mv
```

<a id="8-算法流程"></a>
## 8. Algorithm process

```text
原始 12 导联
    ↓
输入方向检查与重采样
    ↓
去基线和工频陷波
    ├── analysis signal：保留 ST/T 幅度
    └── detection signal：额外低通，增强 QRS
            ↓
质量评估、导联反接和起搏检测
            ↓
多导联 QRS 检测
            ↓
心拍形态聚类和代表心拍
            ↓
P/QRS/T 波定位及多导联共识修复
            ↓
PR、QRS、QT、JT、QTc、ST、振幅、面积和心电轴
            ↓
房性事件、AF/AFL 和临床规则
            ↓
JSON、文本报告、ECG 图像和 MedGemma 证据
```

<a id="81-双信号预处理"></a>
### 8.1 Dual signal preprocessing

- `analysis_signal` uses two-stage median filtering to remove baseline drift and perform power frequency notching;
- `detection_signal` performs additional low pass for QRS detection;
- ST segment, T wave and amplitude measurements use higher fidelity analysis signals whenever possible.

<a id="82-质量控制"></a>
### 8.2 Quality Control

Indicators such as baseline drift, EMG noise, power frequency interference, clipping and low amplitude are calculated separately for each lead. P-wave, QRS, T-wave, and QT use different reliability thresholds.

A passing quality grade does not mean that the measurement is absolutely correct. The caller should read the value, confidence, source, and availability simultaneously.

<a id="83-qrs-与代表心拍"></a>
### 8.3 QRS Heartbeat with the representative

The QRS detection fuses multiple limb and precordial leads to find candidate R-peaks using bandpass, difference, square, moving integration, and adaptive thresholding. The detected heartbeats are clustered by morphology, and then representative heartbeats are constructed through cross-correlation alignment and median/medoid methods.

### 8.4 P/QRS/T delineation

The system estimates for each heartbeat and each lead:

- P onset, peak, offset;
- QRS onset, Q/R/S peaks and QRS offset;
- T peak and T end;
- ST-J, ST midpoint and ST end.

The localization process incorporates representative beat priors, local slopes, tangent/geometric methods, physiological ranges, and multi-lead support. There are additional recovery paths for wide QRS, pacing, low amplitude, and long PR.

<a id="85-全局特征"></a>
### 8.5 Global Features

Main output includes:

- HR and RR;
-PR;
- QRS duration;
- QT and JT;
- QTc Bazett, Fridericia, and Hodges, Framingham in the rules layer;
- P/QRS/T/ST axis;
- QT dispersion;
- P, Q, R, S, T amplitude and area;
- ST shift and morphological evidence.

Wide QRS or pacing cases will reduce the weight of the normal QT and make greater use of the JT and wide QRS dedicated paths.

<a id="86-房性节律"></a>
### 8.6 Atrial Rhythm

AF/AFL analysis uses not only RR irregularities but also:

- P wave organization;
- room association;
- QRST template subtraction;
- Residual signal repeatability;
- Multi-lead F-wave frequency consistency.

If QRST subtraction fails quality verification, the system will retain the scaffold evidence but will not treat it as a reliable result.

<a id="9-输出说明"></a>
## 9. Output description

demo and batch currently use by default:

```python
prepare_json_export(to_dict(features))
```

The default JSON main top-level fields are:

| Field | Content |
|---|---|
| `quality` | Per-lead quality and record-level gating |
| `beats` | R peak, RR and beat grouping |
| `groups` | Morphological group and group level measurements |
| `representative_leads` | Representative characteristics of each lead |
| `global_features` | HR, PR, QRS, QT, QTc and cardiac axis |
| `rhythm_inputs` | Pacing, P-Events, AF/AFL and Availability |
| `morphology_inputs` | P/QRS/ST/T Forms and Derivative Facts |
| `interpretation` | Early DXL-inspired reference explanation |
| `clinical_interpretation` | Current authoritative clinical rule results |
| `statement_engine` | Diagnostic statements, suppression relationships and evidence |
| `metadata` | Detectors, consensus, fixes and provenance information |

<a id="解释层的使用原则"></a>
### Principles for using the explanation layer

The project currently retains authoritative clinical interpretations and early reference interpretations. Downstream programs should preferentially use:

```text
clinical_interpretation
```

`interpretation` is an early DXL-inspired reference evidence and cannot be unconditionally merged with authoritative results.

### Structured v3

`ecgfeat.export.build_structured_payload(features)` will generate a structured payload containing stable partitions such as `signal / quality / beats / groups / global / provenance`.

The current schema version is `ecgfeat_structured_payload.v3`. v3 has removed the old Glasgow interpretation block.

It is not the same schema as the demo's default tile JSON. If the validator reports that `signal`, `global`, or `provenance` is missing, you should first confirm which export contract the caller is expecting.

<a id="10-验证"></a>
## 10. Verification

Run the test:

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pytest -q tests
```

As of 2026-07-26, core tests such as QRS, wave boundary positioning, QT, quality, atrial rhythm, pacing, export and clinical rules were locally selected and run. The results are:

```text
385 passed
```

The approximate average absolute error of the existing 200 historical comparison snapshots LUDB:

| Indicators | MAE |
|---|---:|
| HR | 0.66 bpm |
| PR | 9.26 ms |
| QRS | 9.88 ms |
| QT | 9.99 ms |
| QTc Bazett | 11.36 ms |
| QTc Fridericia | 10.78 ms |
| P axis | 14.68° |
| QRS axis | 10.09° |
| T axis | 10.85° |
| QT dispersion | 28.37 ms |

This snapshot predates some of the latest code modifications and can only be used as a historical engineering reference. It cannot replace revalidation of the current version, nor can it prove clinical diagnostic performance.

<a id="11-已知限制"></a>
## 11. Known limitations

1. The current system is a research-grade implementation and is not a regulatory-validated medical device.
2. AF/AFL, complex pacing, low-amplitude P waves and severe noise still require larger-scale external validation.
3. `acute_occlusion_pattern` is closer to ST elevation screening in adjacent leads and cannot independently diagnose acute coronary occlusion.
4. The historical error of QT dispersion is significantly higher than that of pure QT.
5. Traditional interpretations and authoritative clinical rules may lead to different conclusions.
6. Applicable boundaries for both `<16 岁` and `<18 岁` exist in different pediatric rules, and the 16–18 year old results require special review.
7. The demo has default assumptions about gain and standard lead order.
8. Wave boundary positioning includes multiple layers of rescue/fallback, and upstream errors may propagate backward.
9. Some codes still use the deprecated `numpy.trapz`, and a warning will be generated during testing.

<a id="12-常见问题"></a>
## 12. FAQ

### `ModuleNotFoundError`

First make sure you are using the correct virtual environment:

```bash
cd /workspace/ecg_gemma
source .venv/bin/activate
python -m pip install -e ./feature_extraction
python -m pip install matplotlib
```

Install missing components according to specific tasks:

```bash
python -m pip install wfdb pandas gradio
```

<a id="处理-pt-时提示缺少-torch"></a>
### Prompt that torch is missing when processing `.pt`

PyTorch is not included in the core feature extraction dependencies. Please install the corresponding version according to the machine's CPU or CUDA environment.

<a id="没有-gpu"></a>
### No GPU

ECG feature extraction itself is NumPy/SciPy CPU algorithm and does not require GPU. The GPU is mainly used for subsequent MedGemma inference.

<a id="图片生成失败"></a>
### Image generation failed

```bash
python -m pip install matplotlib
export MPLBACKEND=Agg
```

<a id="json-文件太大"></a>
### JSON The file is too large

Do not use `--full-beats` or `--include-beat-features`. The default summary output remains representative of leads, global measurements, and clinical interpretations.

<a id="schema-校验失败"></a>
### Schema verification failed

The confirmation validator expects:

- demo/batch default `to_dict()` tile structure; still
- structured v2 generated by `build_structured_payload()`.

The two outputs cannot use the same set of top-level field requirements.

<a id="13-建议的后续工作"></a>
## 13. Recommended follow-up work

1. Unify the default JSON schema, and clarify the version and authoritative explanation fields.
2. Establish strict input contracts for lead name, units, gain, NaN, sample rate, and record length.
3. Validate interval errors, rhythm classification, and clinical rules using patient-level independent test sets.
4. Enhance AF/AFL, P-axis, pacing and ST elevation mixed scenarios.
5. Unify pediatric age routing.
6. Split the oversized `api.py`, `delineate.py`, and `features.py` into clear, testable stages.
7. Centrally manage algorithm thresholds and add version numbers to threshold configurations.
8. Add contract tests for demo, batch, legacy JSON and structured v2 respectively.

## License

Please supplement and confirm the Project License, Dataset License, and Model License before using this project code or output for external distribution, research publication, or commercial use.
