# P 波稳健定界引擎实施与验证审计

生成时间：2026-07-28

## 结论

`docs/p wave doc` 中要求的首轮确定性 DSP 改造已经接入 `ecgfeat` 主流程。
新实现不再把单个共识边界当作无条件有效结果，而是同时导出逐导联证据、
strict/robust 边界、经验 CI、P 状态、采集链质控和结构化拒判原因。

在 200 条 LUDB 记录的冻结版 v3 代理评估中，完整 P 边界覆盖率保持为
93.3%。相对上一版 Ta/TP guard，beat 级 onset、offset 和 P 时限的 bias、
SD、MAE、P95 绝对误差均改善。记录级 P 时限均差从 -15.03 ms 改善到
-8.52 ms。

这仍不是正式 IEC/CSE 合规结论。LUDB 同时参与了开发和参数冻结，经验 CI
尚未在外部数据库校准；剔除 8 个偏差后的记录级均差为 -10.05 ms，严格说
仍比 ±10 ms 代理限多 0.05 ms。

## 实现范围

### 采集链和同步

新增 `feature_extraction/ecgfeat/acquisition_qc.py`：

- 用 QRS 导数包络估计逐导联固定延迟，支持亚采样抛物线细化。
- 只有延迟 8–20 ms、拍间 MAD 不超过 2.5 ms、校正后相关系数不低于
  0.75 且相关提升至少 0.08 时才自动补偿。
- 补偿后重新检测 QRS；beat 数改变或 QRS P95 位移超过 12 ms 时完整回滚。
- 检查重复通道、组内异常增益、滤波不一致和严重肢体导联配置错误。
- 无法安全补偿的明显不同步以 `CHANNEL_DESYNCHRONIZED` 阻断权威融合。

### 基线、检测和空间处理

新增 `feature_extraction/ecgfeat/p_wave_engine.py`：

- 从已排除 P/QRS/T 且低活动的样本建立 quiet observation mask。
- 以共同漂移加逐导联残差的确定性 Kalman 状态空间模型估计基线。
- offline 使用前向/后向平滑，online 只使用前向状态，避免未来信息泄漏。
- 精度分支保留较宽带宽；检测分支使用 35 Hz 低通。
- 从 quiet 样本估计噪声协方差并白化多导联导数活动。
- 在粗 P 窗内按奇异值比例选择 1 或 2 维空间表示。
- 无可验证 quiet 样本时不伪造局部 SNR，并显式降低基线置信度。

### P-on-T、T 重建和 Ta

- T 重建只从 RR 相近的其他拍学习，并掩膜源拍和目标拍的全部 P 区。
- 模板拟合只允许 ±8 ms 位移、0.97/1.00/1.03 拉伸以及受限鲁棒仿射变换。
- 相减边缘使用余弦 taper；残差能量和二阶差分高频比必须通过验证。
- 并行保留原始信号、验证后的 T 重建残差和 T 空间子空间抑制分支。
- T 重建失败、跨分支不一致或 Ta offset 不可辨识时降低置信度或拒判，
  不以“残差更干净”作为接受依据。

### 逐导联边界和不确定度

每个候选同时运行：

- 4%、6%、8%、10%、12% amplitude hysteresis
- 多尺度导数/DoG 活动边界
- 最小二乘切线
- 局部能量变化点
- 非 Ta 场景下的条件面积边界
- 正负基线不确定度扰动

输出 `PWaveLeadBoundary`，包含 onset/peak/offset、两端 CI、sigma、独立置信度、
local SNR、质量评分、基线模式、T 重建状态、Ta/P-on-T 标记、分支差异、
使用方法和 flags。等电位、高噪声或硬质控失败的导联不进入权威融合。

### 多导联融合和拒判

- 候选 cluster 使用固定 anchor，禁止单链式 chaining。
- onset 选择最早的合格 cluster，offset 选择最晚的合格 cluster。
- 至少需要 3 个有效导联和 2 个独立组：limb、right precordial、
  left precordial。
- 每个导联权重封顶，每个相关导联组的总权重再次封顶，避免重复计票。
- strict 边界保留硬质控后的最早 onset/最晚 offset；robust 边界使用组限权
  分位融合。
- 新稳健模型与既有共识模型作为两个独立模型互证。既有模型缺少完整边界
  时，新候选保留用于审计但以 `MODEL_DISAGREEMENT` 拒判。
- onset 对两模型等权；offset 采用 late-preserving 融合，较晚模型权重
  87.5%，用于抑制已证实的 P 时限系统性收缩。

每拍输出 `PWaveBeatAssessment`，状态为：

- `P_PRESENT`
- `AF_LIKE`
- `ORGANIZED_ATRIAL_ACTIVITY`
- `OVERLAP_UNCERTAIN`

支持的拒判原因包括：

- `BASELINE_UNOBSERVABLE`
- `P_ON_T_UNRESOLVED`
- `P_OFFSET_TA_AMBIGUOUS`
- `INSUFFICIENT_INFORMATIVE_LEADS`
- `CROSS_LEAD_DISAGREEMENT`
- `CHANNEL_DESYNCHRONIZED`
- `LEAD_CONFIGURATION_INVALID`
- `MODEL_DISAGREEMENT`

### 拍间一致性与接口

- 按相对 R 点的 onset/offset 形态向量建立确定性拍间 cluster。
- 只用 cluster 内抖动扩展 CI，不对边界做可能掩盖异位搏动的时间平滑。
- `ECGFeatures.p_wave_assessments` 保存完整新契约。
- `metadata.acquisition_qc` 和 `metadata.p_wave_contract` 保存记录级审计摘要。
- 结构化导出的 `provenance` 包含 acquisition QC、P 契约和逐拍 assessment。
- 只有互证通过的 robust 边界写回原有
  `p_onset_consensus_index`/`p_offset_consensus_index`，保持下游兼容。

## LUDB 200 条冻结版结果

基线为 `ludb_p_boundary_analysis_ta_tp_guard_final/`，最终结果为
`ludb_p_boundary_analysis_robust_engine_v3/`。

| 指标 | 旧版 | 稳健引擎 v3 |
|---|---:|---:|
| 完整边界配对 | 1295 | 1289 |
| 完整边界覆盖率 | 93.30% | 93.27% |
| Onset bias | +4.64 ms | +3.00 ms |
| Onset SD | 35.80 ms | 32.15 ms |
| Onset MAE | 22.89 ms | 18.98 ms |
| Onset P95 绝对误差 | 94.60 ms | 67.20 ms |
| Offset bias | -8.69 ms | -3.78 ms |
| Offset SD | 27.58 ms | 25.07 ms |
| Offset MAE | 17.89 ms | 15.04 ms |
| Offset P95 绝对误差 | 70.60 ms | 52.00 ms |
| P-duration bias | -13.33 ms | -6.78 ms |
| P-duration SD | 23.00 ms | 21.45 ms |
| P-duration MAE | 21.78 ms | 17.42 ms |
| P-duration P95 绝对误差 | 50.00 ms | 44.00 ms |
| 记录级 P-duration bias | -15.03 ms | -8.52 ms |
| 记录级 P-duration SD | 16.68 ms | 14.52 ms |
| 记录级 P-duration MAE | 18.66 ms | 13.49 ms |
| 剔除 8 条后的 bias | -16.52 ms | -10.05 ms |
| 剔除 8 条后的 SD | 12.56 ms | 11.71 ms |

全部 200 条记录处理成功。结果显示系统性时限收缩约减少 6.55 ms，主要
P 端点误差和长尾均同步下降；覆盖率只下降 6 个 beat，四舍五入后保持
93.3%。

## 自动化验证

- 全测试集：`959 passed, 2 skipped, 145 subtests passed`
- 新增 P 引擎专项测试：独立导联组、strict/robust、基线不可观测、
  P-on-T、AF overrule、并行模型互证、亚采样延迟补偿及回滚证据
- 真实 LUDB 评估：200/200 成功

4 条运行时 warning 来自既有 LUDB 104/111 回归中的空切片均值；测试通过，
未产生新失败。

## 尚未关闭的验证缺口

1. 经验 CI 仅来自阈值、方法、基线和拍间扰动，尚未完成 nominal coverage
   校准；输出明确标记为 `empirical_perturbation_uncalibrated`。
2. LUDB 已用于选参，必须在冻结参数后使用第二个带逐导联 P 标注的数据集
   做独立验证。
3. 仍需单独报告 AF/房扑、无 P、P-on-T、高心率无 TP、Ta 强干扰和低幅 P
   分层的接受率、假阳性率和 CI coverage。
4. 记录级 trimmed bias 为 -10.05 ms，虽然仅超代理限 0.05 ms，但不能写成
   已通过 ±10 ms。
5. 新引擎增加了确定性多分支计算量；还需在目标部署硬件上测量在线延迟和
   最坏运行时间。
