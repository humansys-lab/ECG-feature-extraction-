<!-- i18n-nav -->
[中文](ecgfeat_diagnostic_contract_v10.md) | [English](ecgfeat_diagnostic_contract_v10.en.md) | [日本語](ecgfeat_diagnostic_contract_v10.ja.md)
<!-- /i18n-nav -->

# ecgfeat 诊断契约 v2026.07.10

本版本将 `docs/流程` 中的“测量”和“诊断”分开实现：即使诊断门控失败，
仍返回可获得的测量值、质量证据和失败原因，但不再把不可靠测量解释为阴性或
阳性诊断。本项目仍是研究与辅助判读工具，不是经过监管验证的医疗器械。

## 项目特定例外

本实现**不要求原始采样率必须达到 500 Hz**：

- 支持的最低输入采样率仍为 100 Hz；
- 可使用原始采样率，也可通过 `fs_internal` 重采样；
- 报告记录原始/内部采样率、时间量化分辨率及是否低于 500 Hz；
- 低于 500 Hz 本身不触发诊断停止；
- 但报告会标记起搏钉灵敏度可能受限，不能把“未检出起搏钉”解释为确定无起搏。

## 已实现的输入和质量契约

- 接受完整标准 12 导联，或带 `lead_names` 的 8 个独立导联
  `I, II, V1-V6`；后者按 Einthoven/Goldberger 关系派生其余肢体导联。
- 支持 `mV`、`uV`、`V` 和带 `gain_uv_per_lsb` 的数字量，内部统一为 mV。
- 检查 NaN/Inf、时长、平线、饱和、基线漂移、工频、肌电、
  pSQI、kSQI、bSQI、肢体导联方程残差和导联放置异常。
- 主 QRS 检测与独立多导联幅度检测进行一致性复核；严重不一致时停止诊断。
- 诊断门控状态为 `pass`、`partial` 或 `stop`，并给出
  `stop_reasons`、`partial_reasons` 和允许继续的诊断域。
- 少于 3 个心搏、短于 8 秒、严重质量失败、严重导联方程不一致或疑似肢体导联
  反接时停止诊断；复杂多形态记录仅保留质量、节律和异位搏分析。

## 新增测量与 Finding

- HR 最小/最大值，RR mean/SD/CV、RMSSD、pNN50、Poincaré SD1/SD2 和 RR entropy。
- Bazett、Fridericia、Framingham、Hodges QTc，宽 QRS 时保留 JT/JTc 限制。
- QRS-T 夹角和 R/S 转换区。
- U 波幅度、T 波对称性、PR 段偏移、QRS 切迹/顿挫、VAT 和 ST 斜率。
- 独立三态 Finding：`TRUE`、`FALSE`、`UNKNOWN`。缺少必要输入时必须为
  `UNKNOWN` 并进入 `abstentions`，不得静默转成 `FALSE`。

## 诊断与报告

- 新增或强化窦性节律、约 150 次/分的 2:1 房扑复核、起搏节律和设备异常、
  宽 QRS 心动过速、严重 QTc、LVH/RVH 扩展标准。
- 新增或强化 ST 压低、de Winter、Wellens、左主干型、原始/改良
  Sgarbossa、Brugada 1 型、高/低钾提示、电交替、早复极、心包炎、
  右位心、S1Q3T3/右心负荷和洋地黄效应。
- 支持传入 `prior_features` 做序列比较：节律、PR、QRS、QT、ST、心电轴及
  T 波极性变化。
- 最终陈述含固定置信等级 `HIGH/MEDIUM/LOW/UNAVAILABLE`、优先级 `P0-P6`、
  `report_statement`、`human_review_required`、证据、阈值、抑制原因和可用的
  SNOMED CT 编码。
- 所有报告均带辅助判读声明；高风险、技术限制、边界结果和显著序列变化强制
  人工复核。

## API 示例

```python
features = ECGFeatureExtractor(
    fs_internal=None,       # 使用原始采样率；也可指定内部采样率
    mains_freq="auto",      # 也可固定为 50 或 60
).extract(
    signal,
    fs,
    meta=PatientMeta(
        age=55,
        sex="male",
        amplitude_unit="uV",
        device="device-name",
        acquisition_time="2026-07-28T12:00:00Z",
    ),
    lead_names=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    prior_features=previous_features,
)

gate = features.metadata["diagnostic_gate"]
analysis = features.metadata["clinical_interpretation"]
```

生产或临床验证前仍需完成文档要求的 CSE 测量验证、逐诊断类别外部验证、罕见高危
形态专项集、跨设备/滤波/人群鲁棒性、校准与前瞻性医师复核研究。
