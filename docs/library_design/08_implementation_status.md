# 08 — 实现状态、已通过门禁与发布就绪状态

更新日期：2026-09-24（上一版：2026-09-22 审查）。本文件记录**已运行并验证**的实现，
与 00–07 的目标设计区分。它不是临床有效性声明：所有已发表字段在 schema 1.0.0 中
均为 `unvalidated`。

## 结论

迁移 Phase 0–5 已完成并通过各自门禁；Phase 6（退役兼容层）按定义只能在两个
minor 版本的兼容窗口之后进行，不属于首次发布。维护者已于 2026-09-24 决定：
**Apache-2.0** 许可证、作者/主页为 **humansys-lab**（`github.com/humansys-lab/ECG-feature-extraction-`）、
**绘图并入核心包**（`ecgfeat.viz`，Matplotlib 为 `viz` 可选依赖；原 `ecg-records-viz` 不再单独发布）、
**`ecg-records` 与 `ecginterpret` 两个发行包一起发布**、**由维护者上传**。据此仓库已处于
**可发布状态**：最终产物由干净检出的提交 `1cb2ccc` 构建，全部产物检查（含 LICENSE）与
`twine check --strict` 通过。上传步骤见仓库根目录 `RELEASING.md`；尚未上传任何包
（本机没有 PyPI 凭据）。

| 发行包 | 导入名 | 版本 | 内容 |
|---|---|---|---|
| `ecg-records` | `ecgfeat` | 0.1.0 | 八阶段流水线、四个策略对象、ECG Record schema 1.0.0、查询/CLI、NPZ sidecar；`ecgfeat.viz`：`(signal, record)` 绘图 API + 旧 `ECGFeatures` 绘图（`ecgfeat.viz.legacy`，已弃用），需 `[viz]`（Matplotlib） |
| `ecginterpret` | `ecginterpret` | 0.1.0 | 规则引擎（解释/临床/Glasgow/语句引擎/MI/儿科）+ Interpretation 文档 1.0.0 |

## 各阶段结果（均以冻结的 Phase 0 基线为准）

基线修订为 `52a339c`（迁移前工作树快照）。黄金语料：sentinel 156 例（96 标准 + 60
个 limited/ST 源/逐一 refinement 开关）与 full 1,063 例，覆盖 LUDB、QTDB、EDB、
PTB-XL、BUT-PDB、NSTDB、GUDB；选择规则 `ecg-records-golden-v1` 只看记录 ID 的哈希。
两次独立运行未改代码时 156/156 字节一致，唯一非确定性字段是解释层的
`generated_at` 时间戳（唯一允许的规范化）。

| 阶段 | 内容 | 门禁结果 |
|---|---|---|
| 0 冻结基线 | `benchmarks/golden`（manifest、期望输出、`snapshot_regression.py` 四种模式）；`benchmarks/harness` 包装 9 个既有评估工具并按文档 06 容差冻结 | 双跑 156/156 一致；9 个评估工具双跑门控指标逐位一致 |
| 1 `_engine` 迁移 | 31 个引擎模块移入 `ecgfeat/_engine`，旧路径为同一模块对象的别名并在导入时告警 | sentinel legacy-bytes+record-bytes 156/156；full 1,063/1,063 |
| 2 Record 影子 | models→`_engine/foundation`、export→`compat/export_v0`（静默别名）；独立 record builder；crosswalk 数据 + 独立比较器；验证证据注册表；NPZ sidecar；字段默认缺失状态；黄金参考夹具；性质测试；性能预算；batch 输出模式 | record-crosswalk full 1,063/1,063（含迁移前记录构建失败的 44 例）；24 kB 参考摘要 18,061 字节 |
| 3 拆分 `api.py` | 3,155 行编排拆为 8 个 stage + 4 个 policy；49 个私有 helper 归位；旧编排原样保留于 `compat/_api_v0_legacy_orchestration.py`（`ECGFEAT_LEGACY_ORCHESTRATION=1` 回滚） | full legacy-bytes 1,063/1,063；回滚路径 sentinel 156/156；full record-crosswalk 1,063/1,063；分阶段与回滚速度相同 |
| 4 Record 为主接口 | 门面 `__all__` = 文档 03 列表 + 分组的旧名称；旧拼写使用时 `ECGDeprecationWarning`，`ecgfeat.compat` 为显式静默旧契约；仓库内消费者改用 compat；ecgagent 写出 record 并在证据边界解析带 schema 检查的地址 | sentinel legacy/crosswalk/interpretation 各 156/156；全量测试 2,586 通过 |
| 5 解释分包 | 规则代码移入 `interpretation/src/ecginterpret`；引擎拥有自己的阈值与两个可用性谓词（与规则引擎等值测试）；record 路径不再依赖解释；核心仅经 `compat` 惰性访问 `ecginterpret` | full legacy-bytes 1,063/1,063（经 ecginterpret）；full record-crosswalk 1,063/1,063；full interpretation 1,063/1,063；屏蔽 ecginterpret 时参考记录字节一致 |
| 评估工具 | 9 个工具对冻结基线 compare（Phase 5 树） | 0 回归、0 新失败、0 精确差异、0 缺失指标 |

## 本轮发现并修复的真实缺陷

- 迁移前 record 路径在 44/1,063 条真实记录（4.1%）上整条失败：单个倒置 QRS 或负区间
  使整份记录校验失败。现改为该单元 `unmeasurable(fiducial_order_violation|negative_interval)`。
- 真实 10 搏 LUDB 记录 summary 达 32,851 字节（逐单元重复缺失原因）。新增无损的字段
  `default` 缺失状态后为 18,501；所有真实 12 导联 sentinel 记录（8–15 搏）≤23,347。
- 性质测试发现 builder 与校验器对不完整 QRS 链的顺序判定不一致，已统一。
- `feature_extraction.ecgfeat` 与 `ecgfeat` 曾加载两份模块；现为同一对象（别名 finder，
  并恢复被 importlib 覆盖的 `__spec__`，否则 `importlib.resources` 失效）。
- 性能探针在 Linux 上继承父进程 `ru_maxrss`；改用 `VmHWM`。
- 干净安装矩阵暴露测试对 NumPy 2 / Matplotlib 3.8 API 的依赖，已改为可移植写法。
- `ecginterpret` README 示例读取不存在的 `rhythm_class` 键（该 README 即 PyPI 项目页，上传后无法修改）；
  按真实文档改为 `heart_rate_class` / `bundle_branch_block` / `clinical.final_statements`，并经端到端运行确认。
- CI 工作流从未在 GitHub 上运行，按工作流原样命令在本地复现后发现三处必然失败：suite、sentinel-golden 与
  `parity.yml` 两个作业未安装 `ecginterpret`（39 个测试模块与 legacy-bytes 对比依赖它）；numpy/scipy 矩阵值
  缺少运算符（展开为 `numpy1.26.*`，pip 拒绝，5 个矩阵项中 4 个无法安装）；无数据集的托管 runner 上
  `test_dxl_regression` 的 14 个测试失败而非跳过。均已修复并按工作流命令复跑通过。

## 已实现的契约与工具

- **Record**：三种缺失状态 + 字段默认状态；发表的 fiducial 保证顺序、区间非负；验证状态
  只来自注册表，严格读取拒绝高于上限的声明；`st_morphology` 上限 `known_problem`。
- **Sidecar**：同目录相对 URI、字节 SHA-256、record id/schema 双向绑定、`uint8` 状态掩码
  与 JSON 缺失编码逐单元一致、确定性 zip；严格加载即校验。
- **层次约束**：import-linter 7 个契约（record 独立、引擎无上行、foundation 最底层、
  pipeline 不触及旧导出/解释/绘图、仅 compat 可达 ecginterpret、`ecgfeat.viz` 不触及流水线/解释、
  `ecgfeat.viz` 不直接导入私有引擎）；消费者 AST 检查 0 违规。`import ecgfeat` / `import ecgfeat.viz`
  不加载 Matplotlib（隔离测试在新解释器中验证）。
- **门禁脚本**：`snapshot_regression.py`、`python -m benchmarks.harness`、
  `tools/check_validation_registry.py`、`check_performance.py`、`check_artifacts.py`、
  `check_changelog.py`、`build_docs.py --check`、`release_smoke.py`。
- **CI**：`.github/workflows/test.yml`（矩阵、边界、注册表、性能、文档、自托管
  sentinel）、`parity.yml`（自托管全量黄金 + 评估工具）、`release.yml`（构建→检查→
  3.10/3.13 冒烟→TestPyPI 演练→人工批准→PyPI 可信发布→发布后验证→打 tag）。
  suite（py3.10/NumPy 1.26 与 py3.12/NumPy 2.2 两项）、import-boundaries 与 gates 作业已按工作流原样的
  安装命令在全新环境中本地运行通过；工作流本身尚未在 GitHub 上实际运行。
- **文档站点**：`docs/site`（免责声明、概念、迁移指南、验证方法、局限、兼容策略）+
  由已安装包与随包资源生成的参考页；`mkdocs build --strict` 通过。

## 发布前验证（本地，2026-09-24，最终产物）

| 检查 | 结果 |
|---|---|
| 构建 2 个 sdist + 2 个 wheel（干净检出，提交 `1cb2ccc`） | 成功；`twine check --strict` 通过；与合并后首次构建逐文件内容一致 |
| `tools/check_artifacts.py` | 全部通过（含 LICENSE/NOTICE、py.typed、schema、验证注册表、`ecgfeat/viz`、免责声明） |
| wheel：Python 3.10 / NumPy 1.26.4 / SciPy 1.11.4 / Matplotlib 3.7.5，无 Numba，无数据集检出 | 2,545 通过，41 跳过；冒烟通过 |
| wheel：Python 3.13 / NumPy 2.5.3 / SciPy 1.18.1 / Matplotlib 3.11.2，无数据集检出 | 2,544 通过，42 跳过；冒烟通过 |
| 仅核心 wheel（无 Matplotlib、无 ecginterpret），Python 3.12 | 冒烟通过；绘图给出安装提示；Matplotlib 未加载；隔离/契约子集 75 通过 |
| 由 sdist 重建的 wheel，Python 3.11 | 冒烟通过（含 `ecgfeat.viz` 与 `ecginterpret`） |
| CI suite 作业原样命令：py3.10/NumPy 1.26、py3.12/NumPy 2.2 | 2,540 / 2,541 通过 |
| CI import-boundaries / gates 原样命令 | 7 契约保持；注册表 0 问题；changelog 通过；参考页 6/6 最新；`mkdocs build --strict` 通过 |
| 开发环境（含数据集）Python 3.12 / NumPy 2.2.6 | 2,584 通过，2 跳过 |
| 参考记录性能 | 首次 3.9 s、稳态中位 2.2 s、峰值 RSS 243 MB（预算 12 s / 6 s / 600 MB） |

跳过项：无数据集检出中依赖 LUDB/PTB-XL 的测试（托管 CI 同样跳过，数据集测试由自托管 runner 覆盖）、
Numba 专用测试（干净环境未装 Numba，证明核心无需 Numba）、依赖版本特有的模拟、需预先生成产物的
JS00059 测试。macOS/Windows 仅由 CI 覆盖，本地未运行。合并绘图前的三发行包候选（提交 `92f6bd2`）
另在 Python 3.11/3.12 × NumPy 1.26 上通过；合并只改动 `ecgfeat/viz*`、`ecgfeat/visualize.py` 与
`ecgfeat/_deprecated.py`，测量代码逐字节不变。

## 与设计文档的有意偏离（已记录理由）

- 验证证据注册表用 JSON（`validation-evidence.json`）而非 TOML：Python 3.10 无标准库 TOML。
- 核心发行包保持 `feature_extraction/ecgfeat` 扁平布局（解释与绘图包为 src 布局）：包位于
  子目录，仓库根目录运行测试不会意外导入检出副本；CI 用已安装 wheel 运行测试，满足文档
  07 的目的。
- `ecgfeat.models` 的数据类不在构造时告警（保持类身份与 pickle）；弃用信号放在旧入口函数。
- ecgagent 的诊断证据仍来自旧载荷：record 1.0 只发表紧凑子集，扩充需 schema minor；已提供
  record 输出与带版本检查的地址解析，未改变模型可见的工具与提示。
- `ecginterpret.interpret_record` 需要原始信号（record 1.0 不含全部规则输入），先校验信号
  SHA-256 再用 record 自身配置重算。
- 文档 07 设计为三个发行包；维护者于 2026-09-24 决定把绘图并入 `ecg-records`（`ecgfeat.viz`，
  Matplotlib 作为 `viz` 可选依赖）。设计中拆分的两个目的仍然满足：核心安装不带 Matplotlib，
  `import ecgfeat` 不加载绘图代码，import-linter 禁止 record/引擎/流水线层导入 `ecgfeat.viz`；
  代价是绘图不再独立定版本。`ecgrecords_viz` 导入名从未发布，直接删除。

## 发布阻碍的处理状态

1. **许可证**：已解决。Apache-2.0（维护者确认权利），LICENSE/NOTICE 位于仓库根目录与三个发行包，
   元数据为 PEP 639 `License-Expression: Apache-2.0`。
2. **元数据**：已解决。作者/维护者 `humansys-lab`，项目 URL 指向上游仓库（链接在代码合入该仓库
   `main` 后生效）。
3. **上传**：由维护者执行（`RELEASING.md` 方式 A：GitHub 可信发布；方式 B：本地 twine）。
   本机没有 PyPI/TestPyPI 凭据。2026-09-24 复查：`ecg-records`、`ecginterpret` 在 PyPI 与 TestPyPI
   均返回 404（不代表保留）。
4. **尚需确认（不阻塞上传）**：`.github/CODEOWNERS` 中的 Validation Owner 账号（暂为 `@adsyhub`）；
   文档站点托管；CI 工作流在 GitHub 上的首次实际运行；自托管数据集 runner。

## 发布后（Phase 6）

兼容窗口：0.1.0 首次告警，最早 0.3.0 移除；移除前一个版本改为抛出指向替代接口的
`ImportError` 墓碑。需在此前完成仓库内测试与消费者从旧路径迁出，并以弃用即错误运行一次。
