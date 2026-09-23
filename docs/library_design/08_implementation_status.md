# 08 — 实现审查、已落地范围与剩余门禁

审查日期：2026-09-22。本文件区分**目标设计**与**已运行的实现**，不构成
临床有效性、全量迁移完成或可发布的声明。00–07 中的阶段迁移、消费者拆分和
发布门禁仍是目标；不能用旧算法测试通过来代替这些门禁。

## 结论

当前可用的是 **兼容既有数值引擎的 Record 影子接口**：
`ecg_prepare → ecg_measure → ecg_emit`，以及单次入口 `ecg_record`。
它提供经过校验的列式 JSON、原始采样坐标、单位、显式缺失原因、来源和查询。
既有 `ECGFeatureExtractor` 仍返回 `ECGFeatures`；既有消费者没有被强制切换。

之前文件结构中的 `_engine` 转发模块、八阶段载体和策略类不等于架构迁移完成。
独立 stage 的 `run`、尚未迁移的 QT/pacing/lead-integrity 决策现在显式抛出
`NotImplementedError`，不再无操作返回成功或用候选插入顺序伪造选择。
主提取入口不调用这些未迁移入口，继续执行旧引擎原有逻辑。

## 已解决的文档冲突

| 项目 | 本轮统一的约定 |
|---|---|
| 默认采样率 | `fs_internal=None`，保留原生采样率；删除文档 03 中默认 500 Hz 的冲突 |
| 工频 `None` | 保留旧引擎的 50/60 Hz 频谱推断；低采样率和功率相同时取 50，不表示关闭陷波 |
| Schema 版本 | `MAJOR.MINOR.PATCH`，当前写入 `1.0.0`；`schemas/ecg-record/1.0/` 是版本族目录 |
| Profile | `summary/all/debug`；`measurement` 是过渡别名，`all_measurements` 不支持 |
| 默认序列化 | 保留已有 profile；显式降级可以丢字段，禁止从缺失数据“升级”到 all/debug |
| Record 形状 | 文档 02 的分组列式布局为准；低层 builder 也使用 `/measurements/global`，不再输出第二种直接字段布局 |
| Refinement | 使用真实的 18 个布尔开关，默认全关；不将设计草图中的名字映射到无关算法 |
| 发表类别与验证 | `published_measurement` 是类别，不是准确性等级；验证状态及逐字段证据独立保存 |
| 面积 | 已确认 `qrs_signed_area` 的单位是 mV × internal sample，换算乘 `1_000_000/internal_fs` |

代码中的 `RefinementConfig.enabled` 和 `experimental_flags` 为计算属性。
无参数 `experimental()` 保留既有“全部候选”的研究预设；这不是推荐默认。
Record 层 P-wave 控制尚未接入，非默认设置会报配置错误。顶层
`PWaveConfig` 保留旧引擎类型，不再被同名的新类型覆盖。

## 实现中的数据正确性修复

- 输入强制 channel-major，显式导联名、单位和频率；不自动猜测转置、不补成 12 导联。
  准备阶段复制到只读样本缓冲区，不冻结或修改调用方数组。
- 重采样后的逐搏位置及七个 fiducial 均按频率比映射回原始样本域，使用
  round-half-to-even；合法的 R 样本 0 不再被 `or` 表达式替换。
- limited 输入按旧引擎的 channel-to-slot 映射获取测量，保留用户原始通道名。
  正式电轴和 QTc 为 `not_applicable`，原始逐导联 QT 间期仍可查询。
- 修正 Q0/Q1 与较差质量等级的反向映射；缺失测量附带状态和原因。
  不把结构损坏或 getter 异常静默变成生理缺失。
- 默认运行标识包含信号及其校准、配置、患者元数据、方法、库和 schema 版本，
  不包含 profile。原始信号 artifact 明确为 little-endian float64 C-order
  字节的摘要和 URN，不把元数据指纹冒充 NPY 文件校验和，也不假称已写出文件。
- 已发布 `st_80ms_uv` 仍来源于 legacy native `st_80ms_mv`。可选 ST source
  影响 hybrid 候选；候选来源与已发布字段来源分开记录，不冒称已替换主 ST 测量。
- Record 中保留 `intended_use` 和未知顶层扩展；模型递归脱离输入容器并拒绝
  非 JSON 对象、非字符串键和非有限数值。debug 仅保留白名单测量元数据，
  不将旧解释结果整包塞回 JSON。

## 可运行的契约与工具

Record 读取默认检查 acquisition、轴、字段类型/单位、矩阵形状、样本范围、
缺失状态/坐标、来源引用和有限数值；重复 JSON 成员会报错。验证声明没有证据时
报错，limited 正式输出的适用性也会检查。`validate="none"` 仅跳过契约校验，
不允许 NaN、无效 JSON 或损坏的基础模型。

JSON Schema 与标准库跨字段校验职责不同：Schema 描述结构，严格校验还检查
矩阵长度、来源指针、坐标存在性等跨字段关系。仓库版与包内版 schema 的字节一致性
由测试保证。文档 02 的完整 JSON 示例也纳入机器校验。

`record`、查询接口及 CLI 的导入不加载 NumPy、SciPy、Matplotlib 或提取/解释引擎；
只有指纹计算、NPZ 编码或真正的提取才加载相应数值依赖。分发包仍声明 NumPy/SciPy
为运行依赖，这不等于已推出独立 record-only distribution。

CLI 实现 `measure/validate/query/resolve/select/batch`，支持 Record stdin、
结构化错误和文档 03 的退出码。批处理有界并发、按 manifest 顺序输出状态，
单项失败默认继续、拒绝输出路径逃逸及重复目标。文件在验证成功后通过临时文件和
原子替换写入。未知配置不再静默忽略。

严格 WFDB `.hea/.mat` 适配器要求明确的采样率、命名通道和校准，使用每通道
`(digital-baseline)/gain` 后按单位换算。未显式给 baseline 时使用 ADC zero；
拒绝缺失 header、多段、skew、多采样频率以及文件/维度不一致等未支持布局。
旧 `parse_wfdb_header(path)` / `load_wfdb_mat` 调用保留其兼容行为。
格式依据：[WFDB header specification](https://physionet.org/physiotools/wag/header-5.htm)。
JSON Pointer 的转义和数组索引依据 [RFC 6901](https://www.rfc-editor.org/rfc/rfc6901)。

## 当前字段覆盖

`summary`：文档 02 的 18 个逐搏/导联字段（7 fiducials、5 intervals、
5 amplitudes、1 area）以及心率、额面 QRS 电轴。
`all` 额外包含 `jt_interval_ms`、`u_amplitude_uv`、`qtc_bazett_ms` 和
`qrs_wide_ms`。这并不意味着全部 88 个 legacy 候选已经完成发表审查。
所有当前自动生成的字段保持 `unvalidated`，没有凭借仓库已有 benchmark 名称
自动提高验证状态。

`serialize_record(dense_arrays=...)` 可生成补充 NPZ artifact，但 JSON 数值仍内联。
**尚未实现**将字段迁到 NPZ 后的透明查询、外部文件摘要验证和状态掩码协议。
CLI `--sidecar` 因此明确报配置错误。不能把附件编码器称为完整 sidecar 后端。

## 验证与证据边界

可重跑的命令：

```bash
.venv/bin/pytest -q tests/test_library_design_review.py tests/test_ecg_record_contract.py \
  tests/test_ecg_io.py tests/test_accuracy_refinements.py
.venv/bin/pytest -q
```

定向回归覆盖坐标换算、面积、零样本、limited 映射、身份、配置、WFDB 校准、
读写保真、错误记录拒绝、来源查询、CLI 批处理/原子写入及无重依赖导入。
本轮最终结果：定向测试 **85 passed**；当前工作区全量测试
**2232 passed、150 subtests passed、2 skipped、2 warnings**。
两个跳过项都需要预先生成 JS00059 回归产物；两个 warning 来自旧 NumPy
`trapz` 兼容性测试。它们没有被计作已经通过的验证。
测试还构造 10 s / 500 Hz / 12 导联 / 10 搏的完整矩阵，检查 summary ≤24,000
字节；这是**序列化契约测试**，不是文档 06 所要求的冻结波形及全流程 golden 门禁。

本轮额外对本地 LUDB/1 和 pwave/100 做了真实提取及三种 profile 的无损回读：

| 输入 | 配置 | 搏/导联 | summary / all / debug 字节数 |
|---|---|---|---|
| LUDB/1 | 原始 500 Hz，内部 500 Hz | 7 / 12 | 14,654 / 16,308 / 17,946 |
| pwave/100 前 10 s | limited，原始 360 Hz，内部 500 Hz | 12 / 2 | 10,317 / 12,084 / 13,764 |

这是 smoke test，不是全 LUDB 等价验证，更不是跨数据库准确性验证。
隔离构建的 wheel 已确认包含 `py.typed`、完整 schema，且脱离仓库源码可导入
Record/CLI。Python 3.10 语法解析通过；实际运行测试环境为 Python 3.12，
不冒称已跑完 3.10–3.13 / 各 NumPy 版本的平台矩阵。

## 未完成项及下一步顺序

| 迁移阶段 | 当前状态 / 必需证据 |
|---|---|
| Phase 0：冻结基线 | 未完成：冻结输入 hash、解析配置、数值结果、revision 和 CI golden；已有旧测试不能代替它 |
| Phase 1：引擎迁移 | 未完成：`_engine` 仍主要转发旧模块，不符合最终单向依赖边界 |
| Phase 2：Record 影子接口 | 已有可用子集及回归；其余字段清单、完整 sidecar、golden/性能门禁待完成 |
| Phase 3：八阶段与策略 | 未完成：主流程仍在 `api.py`，不可独立运行占位 stages/policies |
| Phase 4：主要结果切换 | 未完成：旧 API/消费者保持旧结果；不强行改变旧导出形状 |
| Phase 5：解释拆分 | 未完成：旧引擎仍依赖/执行解释逻辑；仅 Record 读取边界与输出已隔离 |
| Phase 6：退休兼容层 | 未开始：需要消费者迁移、弃用版本表和两次 minor 兼容窗口 |
| 发布 | 未完成：许可证/分发名可用性、平台矩阵、真实性能预算和逐字段验证清单仍需独立门禁 |

优先冻结基线，再逐阶段搬移并比较原始未舍入数值；不要将目录树齐全或测试总数
作为阶段完成判据。消费者迁移、解释分包和发布不在本轮中偷偷推进。
