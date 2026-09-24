<!-- i18n-nav -->
[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

# ecgfeat_dxl_inspired

A 12-lead ECG feature extraction library skeleton in the style of **DXL but not a DXL proprietary implementation**. It is organized according to the main workflow in the public manual:

1. Input standardization and analysis signal construction
2. Waveform quality assessment
3. Limb lead reversal heuristic detection
4. Multi-lead QRS detection
5. Beat grouping
6. representative beat construction
7. Beat-by-beat, lead-by-lead wave boundary localization
8. Lead/group/global feature quantity calculation
9. Global QT / QTc / axis output

<a id="重要说明"></a>
## Important Notes

- This is **research/engineering scaffolding**, not medical device software.
- The code implements a **DXL-inspired** approach, not a 1:1 replica of Philips' proprietary algorithm.
- In particular, boundary localization under complex supraventricular/ventricular rhythms and severe noise conditions, as well as dedicated branches for paced beats, require further iteration and validation.

<a id="安装"></a>
## Installation

```bash
pip install -e .
```

<a id="最小示例"></a>
## Minimal Example

```python
import numpy as np
from ecgfeat import PatientMeta
from ecgfeat.compat.api_v0 import ECGFeatureExtractor  # legacy object contract (no warning)
from ecgfeat.compat.export_v0 import to_dict

# ecg shape: [12, n_samples], unit: mV
fs = 500
n = 5000
ecg = np.random.randn(12, n) * 0.02

extractor = ECGFeatureExtractor(fs_internal=500, mains_freq=50)
features = extractor.extract(ecg, fs=fs, meta=PatientMeta(age=45, sex="M"))

print(features.global_features)
print(to_dict(features)["metadata"])
```

<a id="当前输出"></a>
## Current Output

- `quality`: Per-lead quality scores and flags
- `structured payload`: `signal / quality / beats / groups / global / provenance` six stable segments
- `beats`: R positions and grouping for each beat
- `beat_features`: P-QRS-T boundaries and baseline morphology metrics for each beat / each lead
- `representative_leads`: Per-lead representative parameters and variability
- `groups`: Rhythm group summary
- `global_features`: HR / PR / QRS / QT / QTc / P/QRS/T/ST axis / QT dispersion
- `metadata`: QRS detector debug info, record quality, pacing state, representative beat metadata, QT reliable lead list
- `ecgfeat.pediatric_rules`: pediatric Appendix A voltage threshold lookups for RVH/LVH/LSH/BVH morphology evidence; unavailable DXL columns are kept as `None`, and `tests.test_pediatric_rules` validates age-bin and required-key coverage
- Phase 2A measurement note: The initial peak search for P / T now performs beat-level fusion based on multi-lead candidates and representative priors before entering the existing single-lead tangent / geometric boundary localization; public dataclass / export field names and `signal / quality / beats / groups / global / provenance` structure remain unchanged

## Rhythm Gap Closure Additions

- `atrial.py`: independent atrial-event extraction, inter-beat blocked-P scan, and QRST-window atrial residual scaffold features for AF/AFL rule inputs
- `rhythm_rules.py`: paced-rhythm control flow, preexcitation rules, premature/pause/AV-block evidence, measurement availability, and statement suppression
- `metadata["rhythm_analysis"]`: intermediate atrial, AF/AFL, pacing, rule, and availability results used by `export.py` and `interpret.py`
- `export.py`: exposes rhythm inputs including AF/AFL summaries, pacing context, rule evidence, and `rhythm_inputs.record.availability`

## Availability Semantics

- Raw interval/axis measurements remain in `global_features`
- `rhythm_inputs.record.availability` states whether atrial-rhythm-derived values should be trusted for downstream rhythm logic
- Continuous ventricular/dual pacing, AF/AFL, complete AV block, and AV dissociation suppress PR / P-axis based rhythm decisions
- The pacing policy preserves pacing evidence while suppressing unreliable AV-block, AV-dissociation, PR, and P-axis decisions when pacing makes those measurements unreliable
- `rhythm_inputs.af_afl.qrst_subtraction.available` means validated per-lead QRST template subtraction; when validation is not met, the QRST-window residual is exposed as a scaffold through `scaffold_available`, `method`, `qrst_subtraction_quality`, and `atrial_residual_signal_summary`

## DXL Gap Closure Status

- P0 measurement reliability now includes QRS consensus guarding, wide-QRS QT rescue, ST-J provenance, and long-PR P-onset rescue.
- P-wave morphology exports notched, biphasic, initial, and terminal component facts.
- Pediatric morphology uses age-binned voltage thresholds and exports RVH/LVH/BVH evidence with bypass reasons.
- Rhythm inputs expose statement evidence, pacing failure facts, AF/AFL QRST subtraction validation state, and AV-block evidence.
- Remaining limitations are listed in `docs/philips_*_feature_schema_checklist.md`.

<a id="验证命令"></a>
## Verification Commands

```bash
.venv_report_regen_20260625/bin/python -m unittest tests.test_multilead_pt_fusion tests.test_ecgfeat_pipeline -v
.venv_report_regen_20260625/bin/python compare_annotations.py --batch-reports --records 1 4 5 8 --out-dir tmp_phase2_multilead_pt --no-visuals
```

Output Description:

- `tmp_phase2_multilead_pt/summary.csv` will summarize the report discrepancies for this set of LUDB spot-check records
- The per-record report can be used to quickly check for any significant increase in P/T missing counts across this set of samples

<a id="下一步建议"></a>
## Next Steps Recommendations

1. Perform system validation based on CSE / LUDB / QTDB
2. Add a dedicated delineation branch for paced beats
3. Introduce stronger multi-lead P/T fusion and U-wave separation
4. Expand to a complete field set in the style of Extended Measurements
5. Overlay a rhythm / morphology rule engine
