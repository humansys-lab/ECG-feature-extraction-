<!-- i18n-nav -->
[中文](dxl_ecg_dev_task_checklist.md) | [English](dxl_ecg_dev_task_checklist.en.md) | [日本語](dxl_ecg_dev_task_checklist.ja.md)
<!-- /i18n-nav -->

# DXL-inspired ECG 边界定位开发任务清单

> 目标：把当前库中的逐搏逐导联 P/QRS/T 边界定位模块，从“启发式第一版”升级为 **多导联 + 分组 + representative beat + 几何 T-end + 可靠性评分** 的完整测量内核。  
> DXL 手册公开的方法学主线是：先做 waveform quality，再做 waveform recognition，随后形成 representative beat，生成 comprehensive measurements，最后才进入解释层；其中 Group 1 会做 outlier 排除、对齐、平均；QT 的难点是 T-wave end，global QT 采用 reliable leads 的中位数。

---

## 0. 文档使用说明

### 范围
本清单覆盖以下模块：

- `quality/`
- `qrs/`
- `grouping/`
- `representative/`
- `delineate/`
- `features/qt.py`
- `pacing/`
- `validate/`

### 不在本轮范围
本轮先不做：

- 完整 interpretive statements
- 全套临床规则分类
- 1:1 复刻 DXL 私有阈值

### 本轮交付定义
完成后应具备：

1. 多导联 QRS 锚点检测
2. beat grouping
3. representative beat 驱动的边界精修
4. P/QRS/T 单搏局部修正
5. 几何法 T-end
6. reliable lead / confidence 体系
7. global QT 计算
8. 基本可视化与验证基准

---

## 1. 目录与代码结构重构

### 任务 T001：重构 delineation 目录

**目标**  
将现有 `delineate.py` 拆分成分层模块，便于迭代和测试。

**输入**  
现有：

- `ecgfeat/delineate.py`

**输出**  
新增结构：

```text
ecgfeat/delineate/
├── __init__.py
├── rough.py
├── representative_refine.py
├── beat_local_refine.py
├── p_wave.py
├── qrs_bounds.py
├── t_wave.py
├── t_end_geometric.py
├── confidence.py
└── fusion.py
```

**核心工作**
- 把现有函数按职责拆分
- 保留兼容入口
- 增加模块级 docstring

**验收标准**
- `from ecgfeat.delineate import delineate_beats` 仍可工作
- 单元测试能覆盖新旧 API
- 旧调用链不报错

**优先级**  
P0

**依赖**  
无

---

### 任务 T002：建立统一边界数据结构

**目标**  
给 rough / refined / fused 边界统一 schema，避免后续函数之间互相传裸 dict。

**输入**  
当前 `models.py` 中的 `WaveBounds`

**输出**
新增 dataclass：

```python
@dataclass
class BoundaryEstimate:
    onset: Optional[int]
    peak1: Optional[int]
    peak2: Optional[int]
    offset: Optional[int]
    confidence: float
    method: str
    flags: list[str]
```

**核心工作**
- 增加 `BoundaryEstimate`
- 增加 `LeadBoundaryResult`
- 增加 `BeatBoundaryResult`
- 支持 raw candidate / refined / fused 三阶段结果

**验收标准**
- 所有 delineation 子模块都返回统一结构
- `None` 边界场景不崩溃
- 支持 JSON 序列化

**优先级**  
P0

**依赖**  
T001

---

## 2. 质量门控与基础可靠性

### 任务 T003：实现 lead-level 质量评分重构

**目标**  
让边界检测不再默认所有导联等权，而是先有导联质量门控。

**输入**
- 12-lead ECG
- 原始采样率
- 当前 `quality.py`

**输出**
每导联输出：

- `baseline_wander_score`
- `muscle_noise_score`
- `powerline_score`
- `flatline_score`
- `clipping_score`
- `lead_reliable_for_p`
- `lead_reliable_for_qrs`
- `lead_reliable_for_t`
- `lead_reliable_for_qt`

**核心算法**
- baseline wander：低频能量比 + beat baseline shift
- muscle noise：QRS 外高频能量
- powerline：50/60 Hz 窄带峰值
- flatline：导数近零比例
- clipping：饱和/重复采样检测

**实现要点**
- 增加 `quality/metrics.py`
- 增加 `quality/gating.py`
- 支持按波形类型定义不同可靠性阈值

**验收标准**
- 噪声 ECG 的 unreliable lead 能被标出
- 干净 ECG 的多数导联评分稳定
- 与可视化结果一致

**优先级**  
P0

**依赖**  
T002

---

### 任务 T004：实现 beat-level 质量评分

**目标**  
把“整条导联质量”下钻到“每个 beat 的局部质量”。

**输入**
- 每个 beat 的切窗信号
- lead-level quality

**输出**
每 beat 输出：

- `beat_noise_score`
- `beat_baseline_shift`
- `beat_template_corr`
- `beat_measurement_reliable`

**核心算法**
- beat 内局部 SNR
- beat 与组模板的相关性
- PR 段稳定性
- T 区域噪声水平

**验收标准**
- ectopic / noisy beat 的置信度低于 clean dominant beat
- 组内异常 beat 可被剔除

**优先级**  
P0

**依赖**  
T003、T008

---

## 3. 多导联 QRS 检测与锚点生成

### 任务 T005：升级多导联 QRS detector

**目标**  
把当前 QRS detector 固化为明确的多导联融合方案，作为整个 delineation 的锚点层。

**输入**
- 12-lead ECG
- `fs`

**输出**
- `r_locs`
- `qrs_candidate_windows`
- `qrs_detector_confidence`

**核心算法**
1. 选导联：`I, II, V1-V6`
2. z-score 标准化
3. 构造多导联向量能量
4. `5–30 Hz` 带通
5. derivative + squaring + MWI
6. adaptive threshold + refractory
7. 参考导联局部精修 R

**实现要点**
- 保留当前实现思路
- 补充参数配置类
- 增加 detector debug 输出

**验收标准**
- 普通窦律检测稳定
- PVC / BBB / low voltage 下召回率可接受
- 不依赖单导联 II

**优先级**  
P0

**依赖**  
无

---

### 任务 T006：QRS onset/offset 两阶段精修

**目标**  
QRS 不能只用能量阈值外扩，要增加 representative-aware refinement。

**输入**
- `r_locs`
- group representative beats
- lead quality

**输出**
- `qrs_on`
- `qrs_off`
- `qrs_on_confidence`
- `qrs_off_confidence`

**核心算法**
- 粗阶段：局部能量上升/回落
- 精阶段：
  - 一阶导数显著上升点
  - 二阶导数零交叉
  - terminal force 末端转折
  - 与 representative beat 对齐修正

**实现要点**
- 新建 `delineate/qrs_bounds.py`
- 先在 representative beat 上做高置信定位
- 单 beat 只允许在小窗内修正

**验收标准**
- 宽 QRS 边界较当前版更稳
- 多峰 QRS 时 `qrs_off` 不提前截断
- QRS duration 抖动降低

**优先级**  
P0

**依赖**  
T005、T010

---

## 4. Beat grouping 与 representative beat

### 任务 T007：重写 beat grouping 描述子

**目标**  
让分组不只服务于 rhythm，还直接服务于模板化测量。

**输入**
- `r_locs`
- 每 beat 局部波形
- lead quality

**输出**
每 beat descriptor：

- `rr_prev`
- `rr_next`
- `qrs_width_rough`
- `vector_area`
- `paced_flag`
- `template_embedding`
- `lead_corr_signature`

**核心算法**
- vector magnitude snippet
- PCA embedding
- morphology cosine distance
- RR features

**验收标准**
- dominant beat family 能自动形成 Group 1
- PVC 与 normal beat 可分开
- paced beats 不与 non-paced 混组

**优先级**  
P0

**依赖**  
T005

---

### 任务 T008：实现两阶段分组器

**目标**  
先粗分，再细分，减少纯聚类不稳定性。

**输入**
- beat descriptors

**输出**
- `group_id` for each beat
- `group_summary`

**核心算法**
- 第一阶段：规则粗分
  - paced / non-paced
  - narrow / wide
  - early / non-early
- 第二阶段：组内形态聚类
  - hierarchical clustering 或 DBSCAN
  - 距离度量：template cross-correlation

**验收标准**
- 最多 5 组
- Group 1 为成员最多组
- 分组结果对同一条 ECG 复跑稳定

**优先级**  
P0

**依赖**  
T007

---

### 任务 T009：实现 representative beat 生成增强版

**目标**  
使 representative beat 成为后续边界精修主输入。

**输入**
- group 成员 beat snippets
- beat-level quality

**输出**
每 group、每 lead 输出：

- representative beat
- `member_count`
- `mean_template_corr`
- `outlier_count`
- `beat_to_beat_onset_std`
- `beat_to_beat_offset_std`

**核心算法**
- R 对齐
- QRS 内局部互相关微调
- outlier exclusion
- mean beat formation
- 记录组内变异

**验收标准**
- representative beat 比单 beat 更平滑
- 组内肌电噪声明显下降
- 保留变异度元数据

**优先级**  
P0

**依赖**  
T004、T008

---

## 5. Rough delineation：先粗定位区域

### 任务 T010：实现 approximate waveform regions

**目标**  
把当前“直接找峰”改为“先粗区域，再细边界”。

**输入**
- `r_locs`
- representative beats
- RR 信息

**输出**
每 beat 每 lead 粗区域：

- `P_search_region`
- `QRS_search_region`
- `T_search_region`

**核心算法**
- P 区：`qrs_on - 260ms` 到 `qrs_on - 20ms`
- QRS 区：`R ± 120ms`
- T 区：`qrs_off + 20ms` 到 `min(qrs_off+500ms, next_expected_p-margin)`

**验收标准**
- P/T 搜索不跨错区
- 快心率下 T 区不会压到下一搏 P
- RR 很长时搜索区不过宽

**优先级**  
P0

**依赖**  
T006、T009

---

## 6. P 波检测改进

### 任务 T011：实现 multilead P candidate detector

**目标**  
P 波不能只靠单导联 abs-peak，需要多导联候选生成。

**输入**
- `P_search_region`
- lead quality
- representative beat

**输出**
每导联 P 候选：

- `peak_candidates`
- `candidate_scores`

**核心算法**
- 极值点
- 导数换向点
- 局部面积峰
- 小波尺度峰

**实现要点**
- 每导联允许 0~N 个候选
- 按振幅、面积、SNR、组内一致性打分

**验收标准**
- 低振幅 P 不会全部漏检
- 负向 P 仍能出候选
- 房扑样本中置信度下降而不是乱报

**优先级**  
P1

**依赖**  
T010

---

### 任务 T012：实现 P 多导联融合与双分量模型

**目标**  
支持单峰 P、切迹 P、双向 P。

**输入**
- 各导联 P candidates
- lead reliability
- representative priors

**输出**
- `p_on`
- `p_peak1`
- `p_peak2`
- `p_off`
- `p_biphasic_flag`
- `p_notch_flag`
- `p_confidence`

**核心算法**
- 候选导联可见性评分
- Top-K lead 融合
- 双分量模型：
  - `P1`, `P2`, notch depth
- 边界由三类证据联合决定：
  - 振幅回基线
  - 累积面积比例
  - 导数/曲率衰减

**验收标准**
- 常规窦性 P 结果稳定
- biphasic P 可输出两个峰
- `PR` 抖动小于当前实现

**优先级**  
P1

**依赖**  
T011

---

## 7. QRS 多峰形态与细节特征

### 任务 T013：实现 Q/R/R'/S/S' 分量提取

**目标**  
让 QRS 不只输出总时限，还输出可用于 morphology 的分量。

**输入**
- refined QRS bounds
- representative beat

**输出**
- `Q_amp`
- `R_amp`
- `R_prime_amp`
- `S_amp`
- `S_prime_amp`
- `qrs_num_peaks`
- `qrs_notch_count`
- `qrs_slur_flag`

**核心算法**
- QRS 区局部极值序列
- 峰谷交替约束
- 小峰去噪阈值
- 末端 slur/notch 识别

**验收标准**
- RBBB/LBBB/fragmented QRS 的分量特征更合理
- 普通窄 QRS 不会被过拟合成多峰
- notch 计数稳定

**优先级**  
P1

**依赖**  
T006

---

### 任务 T014：实现 VAT / J-point 精修

**目标**  
支持更完整的 morphology lead measurements。

**输入**
- refined QRS
- representative beat

**输出**
- `vat_ms`
- `j_point_idx`
- `j_point_confidence`

**核心算法**
- VAT：起始到主 R 峰时间
- J-point：QRS 末端局部稳定拐点

**验收标准**
- QRS 宽大时 VAT 逻辑稳定
- J-point 不被 ST 噪声明显拖偏

**优先级**  
P2

**依赖**  
T013

---

## 8. T 波与几何 T-end

### 任务 T015：实现 multilead T candidate detector

**目标**  
T 波允许正向、倒置、双向、切迹形态，不再只取 abs-peak。

**输入**
- `T_search_region`
- lead quality
- representative beat

**输出**
- `T1_peak`
- `T2_peak`
- `t_main_polarity`
- `t_notch_flag`
- `t_candidate_score`

**核心算法**
- 局部极值 + 波形面积 + 曲率
- 主峰与次峰评分
- 与组模板一致性约束

**验收标准**
- 倒置 T 仍能稳定输出
- notched T 可区分主次峰
- 噪声导联候选分数明显降低

**优先级**  
P1

**依赖**  
T010

---

### 任务 T016：实现 DXL-like 几何 T-end 算法

**目标**  
用几何拐点法替换当前“阈值回基线法”。

**输入**
- T candidate peaks
- representative T waveform
- baseline trend

**输出**
- `t_end_candidate`
- `t_end_confidence`
- `t_flat_tail_score`
- `u_wave_present`
- `p_on_t_flag`

**核心算法**
1. 取 `T_peak`
2. 定义右侧搜索区
3. 构造 `peak -> search_start` 或 `peak -> search_end` 参考线
4. 计算垂距
5. 取最大垂距点为 inflection candidate
6. 对斜率做上下界约束
7. 如存在 U 波，优先定位 T-U nadir

**实现要点**
- 单导联实现
- 再做跨导联融合
- 支持 flat tail 降置信度

**验收标准**
- 平缓 T 尾较当前阈值法更稳
- 大 U 波样本不再系统性拉长 QT
- `T_end` 组内方差下降

**优先级**  
P0

**依赖**  
T015、T018

---

### 任务 T017：实现 T 波多导联融合

**目标**  
不让单个坏导联决定最终 T-end。

**输入**
- per-lead `t_end_candidate`
- lead reliability
- beat-to-beat variance

**输出**
- `lead_t_end_used`
- `t_end_fused`
- `qt_reliable_leads`

**核心算法**
- 导联打分：
  - `lead_quality`
  - `t_amp`
  - `t_end_var`
  - `rep_confidence`
- 融合策略：
  - Top-K leads
  - robust median / weighted median

**验收标准**
- noisy lead 被自动剔除
- per-lead QT 离群值不影响 global QT
- 融合结果比任一单导联稳定

**优先级**  
P0

**依赖**  
T016、T019

---

## 9. 单搏局部修正与代表搏回投

### 任务 T018：实现 representative-boundary priors

**目标**  
先在 representative beat 上做高置信边界，再回投到组内单 beat。

**输入**
- representative beats
- rough search regions

**输出**
- `rep_p_prior`
- `rep_qrs_prior`
- `rep_t_prior`

**核心算法**
- 先在 representative beat 完成高精度 delineation
- 对每个边界输出 prior window

**验收标准**
- representative 边界可重复
- 组内 beat 共享同一套 prior

**优先级**  
P0

**依赖**  
T009、T010、T012、T016

---

### 任务 T019：实现 beat-wise local correction

**目标**  
让每个 beat 的最终边界 = representative 预测 + 小窗局部修正。

**输入**
- representative priors
- 单 beat 信号
- local quality

**输出**
- 单 beat 最终 `p_on/off`, `qrs_on/off`, `t_on/off`

**核心算法**
- 先模板对齐
- 再仅在 `±20~30 ms` 小窗内：
  - 局部导数修正
  - 局部互相关修正
  - 基线回归修正

**验收标准**
- 组内边界抖动小于当前版本
- noisy beat 不会跑飞太远
- beat-to-beat variance 可用于后续 reliable lead 判定

**优先级**  
P0

**依赖**  
T018

---

## 10. 置信度与 reliable lead 体系

### 任务 T020：实现 boundary confidence 引擎

**目标**  
所有边界都要有置信度，否则后续 QT 和规则层不可调。

**输入**
- local noise
- derivative consistency
- template corr
- lead quality
- beat variance

**输出**
- `p_confidence`
- `qrs_confidence`
- `t_confidence`
- `t_end_confidence`

**核心算法**
组合评分：

```text
conf = f(local_snr, template_corr, baseline_stability, boundary_sharpness, group_consistency)
```

**验收标准**
- 干净 beat 置信度显著高
- flat T / noisy P 置信度明显降低
- confidence 与人工目测趋势一致

**优先级**  
P0

**依赖**  
T004、T019

---

### 任务 T021：实现 reliable lead 判定器

**目标**  
复制 DXL 的方法学精神：global QT 只用 reliable leads，中位数优先于极值。

**输入**
- per-lead QT
- per-lead onset/offset variance
- lead quality
- confidence scores

**输出**
- `reliable_for_qt`
- `reliable_for_axis`
- `reliable_for_global_measure`

**核心算法**
导联可靠性条件示例：

```python
reliable = (
    lead_quality < th1 and
    t_end_var < th2 and
    qrs_on_var < th3 and
    rep_confidence > th4 and
    t_end_confidence > th5
)
```

**验收标准**
- 低振幅高噪声导联被排除
- global QT 不再被单个坏导联拖偏
- reliable lead 数量合理

**优先级**  
P0

**依赖**  
T020

---

## 11. Global QT 与 QTc

### 任务 T022：实现 per-lead QT / JT 输出

**目标**  
先把每导联 QT 做扎实，再求 global QT。

**输入**
- `qrs_on`
- `t_end`
- confidence
- reliability

**输出**
- `qt_ms`
- `jt_ms`
- `qt_confidence`

**核心算法**
- `QT = T_end - QRS_on`
- `JT = T_end - QRS_off`

**验收标准**
- 每导联都有独立 QT
- unreliable lead 可保留数值但标记不可用于 global

**优先级**  
P0

**依赖**  
T017、T021

---

### 任务 T023：实现 global QT / QTc

**目标**  
实现 DXL 风格 global QT 选择。

**输入**
- per-lead QT
- reliable lead mask
- RR

**输出**
- `global_qt_ms`
- `qtc_bazett_ms`
- `qtc_fridericia_ms`
- `qt_dispersion_ms`

**核心算法**
- `global_qt = median(qt_candidates_from_reliable_leads)`
- `qt_dispersion = max - min`
- QTc 同时输出 Bazett 与 Fridericia

**验收标准**
- global QT 对离群导联不敏感
- 与当前简单阈值法相比，重复性更好
- RR 变化时 QTc 逻辑正确

**优先级**  
P0

**依赖**  
T022

---

## 12. Pacing 专用分支

### 任务 T024：实现 pacing spike detector

**目标**  
单独识别 pacing spike，为 paced beat delineation 做前置。

**输入**
- 原始高保真 ECG
- `fs`

**输出**
- `spike_events`
- `spike_confidence`
- `paced_status`

**核心算法**
- 高频通道 `>100 Hz`
- 极窄脉宽检测
- 差分峰 + 面积约束
- 多导联时间聚类

**验收标准**
- pacing spike 不与普通窄 QRS 混淆
- 明显 spike 能被检出
- 假 spike 数量可控

**优先级**  
P1

**依赖**  
T005

---

### 任务 T025：paced beat delineation 分支

**目标**  
paced beats 不能直接套普通 QRS 模型。

**输入**
- paced beats
- spike events
- representative paced template

**输出**
- paced beat 的 `qrs_on/off`
- `stim_to_qrs_ms`
- `ventricular_paced_flag`

**核心算法**
- spike 后宽 QRS 模式
- QRS onset 可更早
- terminal offset 更保守

**验收标准**
- paced QRS 时限较当前版本更合理
- spike 与 QRS 关系可解释

**优先级**  
P2

**依赖**  
T024

---

## 13. Morphology 字段补全

### 任务 T026：补全 lead-level morphology measurements

**目标**  
让库输出接近 Extended Measurements report 粒度。

**输入**
- 最终边界结果
- amplitude / area / slope 计算器

**输出**
每 lead 每 beat 或 representative lead 至少补齐：

- P：`on/peak1/peak2/off/amp/area/dur`
- QRS：`Q/R/R'/S/S'/dur/area/notch/slur/VAT`
- ST：`J_point/ST_on/ST_mid/ST_80/ST_end/ST_slope`
- T：`on/peak1/peak2/off/amp/area/dur/U_wave_flag`

**验收标准**
- 字段命名统一
- JSON / dataframe 导出完整
- 缺失字段有明确 `None` 语义

**优先级**  
P1

**依赖**  
T012、T013、T014、T016

---

## 14. 可视化与调试工具

### 任务 T027：实现边界叠加可视化

**目标**  
没有可视化，很难调波界算法。

**输入**
- ECG
- boundary results
- quality/confidence

**输出**
调试图：

- 单导联 beat 图
- representative beat 图
- P/QRS/T 边界叠加
- confidence 标注

**核心功能**
- matplotlib 绘图
- bad beat 高亮
- reliable lead 标记

**验收标准**
- 一行代码能画某 lead 某 beat 的边界图
- 可视化能直接用于误差排查

**优先级**  
P0

**依赖**  
T019、T020

---

### 任务 T028：实现回归快照工具

**目标**  
让算法改动可追踪，避免“改好了这里，坏了那里”。

**输入**
- 固定 ECG 样本集
- 当前版本输出

**输出**
- JSON snapshots
- PNG overlays
- diff report

**验收标准**
- 每次改动后能自动比较关键边界偏移
- 回归失败时给出样本列表

**优先级**  
P1

**依赖**  
T027

---

## 15. 验证与基准

### 任务 T029：建立 synthetic signal benchmark

**目标**  
先在可控信号上验证 onsets/offsets，再上真实数据。

**输入**
- 合成 ECG 生成器
- 可控噪声注入器

**输出**
评测指标：
- onset error
- offset error
- QT error
- ST error
- 噪声鲁棒性曲线

**验收标准**
- 能分别注入 baseline wander / muscle noise / powerline
- 误差统计自动生成

**优先级**  
P0

**依赖**  
T023

---

### 任务 T030：建立专家标注数据评测

**目标**  
在真实 ECG 上验证边界重复性和误差分布。

**输入**
- 带注释数据集
- 当前算法输出

**输出**
- `P_on/off error`
- `QRS_on/off error`
- `T_end error`
- `PR/QRS/QT error`

**验收标准**
- 评估代码可重复运行
- 输出 mean / SD / percentiles
- 支持分场景统计：窦律、宽 QRS、低振幅 T、噪声 ECG

**优先级**  
P1

**依赖**  
T029

---

## 16. 里程碑计划

## Milestone M1：可用的第二版测量内核
**包含**
- T001 ~ T010
- T016
- T018 ~ T023
- T027
- T029

**完成后能力**
- QRS 与 T-end 明显强于当前版
- representative beat 真正参与边界定位
- global QT 采用 reliable-lead median

---

## Milestone M2：形态特征增强
**包含**
- T011 ~ T015
- T026
- T028
- T030

**完成后能力**
- P/T 双分量
- QRS 多峰
- morphology lead measurements 更完整

---

## Milestone M3：paced / 高复杂度场景
**包含**
- T024
- T025
- T014

**完成后能力**
- paced beat 专用分支
- VAT / J-point 更完整

---

## 17. 优先级总表

### P0
- T001
- T002
- T003
- T004
- T005
- T006
- T007
- T008
- T009
- T010
- T016
- T017
- T018
- T019
- T020
- T021
- T022
- T023
- T027
- T029

### P1
- T011
- T012
- T013
- T015
- T024
- T026
- T028
- T030

### P2
- T014
- T025

---

## 18. Definition of Done

一个任务只有同时满足下面 5 条，才算完成：

- 代码已合入主分支
- 单元测试通过
- 至少 1 个可视化示例通过人工检查
- 回归快照无异常漂移
- 文档补齐：输入、输出、失败模式、参数说明

---

## 19. 建议的开发顺序

按性价比最高的顺序做：

1. `T001-T010`
2. `T016-T023`
3. `T027-T029`
4. `T011-T015, T026`
5. `T024-T025, T030`

这个顺序的原因很简单：  
先把 **QRS 锚点、representative beat、几何 T-end、reliable leads、global QT** 这条主链打通，收益最大，也最符合 DXL 手册公开的方法学重点。

---

## 20. 建议你立刻开工的首批 TODO

```text
[ ] T001 重构 delineate 目录
[ ] T002 建立统一边界数据结构
[ ] T003 lead-level 质量评分重构
[ ] T005 升级多导联 QRS detector
[ ] T007 重写 beat grouping 描述子
[ ] T008 实现两阶段分组器
[ ] T009 representative beat 增强版
[ ] T010 approximate waveform regions
[ ] T016 几何 T-end
[ ] T018 representative-boundary priors
[ ] T019 beat-wise local correction
[ ] T020 boundary confidence 引擎
[ ] T021 reliable lead 判定器
[ ] T022 per-lead QT / JT
[ ] T023 global QT / QTc
[ ] T027 边界叠加可视化
[ ] T029 synthetic benchmark
```
