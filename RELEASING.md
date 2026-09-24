# 发布手册：ecg-records 0.1.0 / ecginterpret 0.1.0

本仓库已处于**可发布状态**，但尚未上传任何包（本机没有 PyPI 凭据；上传由维护者执行）。
上传到 PyPI/TestPyPI 后版本**不可撤回、不可覆盖**；发现问题只能发布新版本修复。

发行包（2026-09-24 维护者决定：绘图并入核心包，解释保持独立）：

| 发行包 | 导入名 | 内容 |
|---|---|---|
| `ecg-records` | `ecgfeat`（绘图：`ecgfeat.viz`） | 测量流水线、ECG Record schema 1.0.0、查询/CLI；绘图需可选依赖 `pip install "ecg-records[viz]"`（Matplotlib） |
| `ecginterpret` | `ecginterpret` | 规则引擎与 Interpretation 文档 1.0.0；也可经 `pip install "ecg-records[interpret]"` 安装 |

## 候选产物

- 构建来源：分支 `library-migration` 的提交 `1cb2ccc`，在干净检出中构建（文档 07 第 4 步）。
- 位置：`dist/`（已被 `.gitignore` 忽略，不入库），4 个文件 + `SHA256SUMS`：

| 文件 | SHA-256 |
|---|---|
| `ecg_records-0.1.0-py3-none-any.whl` | `98300d79c0e77415c44ff5de623fabed6594a5cad885b4690534e123ecb83c23` |
| `ecg_records-0.1.0.tar.gz` | `f1aadb1d67e511d47217c269d30f2e18b6447f14369d6e684b0f7db8eabbe2da` |
| `ecginterpret-0.1.0-py3-none-any.whl` | `4c226969bd8be629145c91bdb38bf3e12ffe6b0e029220aeae0c36cf7d491241` |
| `ecginterpret-0.1.0.tar.gz` | `6cf6056b31207b24d789e79945a623db3bef0e1776c5e246d764e893fe39a013` |

已完成的发布前检查（结果见 `docs/library_design/08_implementation_status.md`）：
`tools/check_artifacts.py dist`（含 LICENSE）通过；`twine check --strict` 通过；
Python 3.10/3.13 干净环境安装测试、仅核心（无 Matplotlib、无 ecginterpret）安装、由 sdist
重建的 wheel 冒烟、CI 工作流安装命令的本地等价运行均通过。

上传前在本机再确认一次（任何一步失败都不要上传）：

```bash
cd dist && sha256sum -c SHA256SUMS && cd ..
.venv/bin/python tools/check_artifacts.py dist
.venv/bin/twine check --strict dist/*.whl dist/*.tar.gz
```

## 方式 A（文档 07 推荐）：GitHub Actions 可信发布

1. 把 `library-migration` 合并/推送到 `github.com/humansys-lab/ECG-feature-extraction-`
   （pyproject 中的项目主页、文档与 changelog 链接指向该仓库的 `main` 分支）。
2. 在 GitHub 仓库设置中创建环境 `testpypi` 与 `pypi`；`pypi` 设为需人工批准（required reviewers）。
3. 在 TestPyPI 与 PyPI 各为 `ecg-records`、`ecginterpret` 添加 *pending trusted publisher*：
   owner `humansys-lab`，repository `ECG-feature-extraction-`，workflow `release.yml`，
   environment 分别为 `testpypi` / `pypi`。
4. Actions → `release` → Run workflow，`version = 0.1.0`。流水线依次：构建 → 产物检查 →
   3.10/3.13 冒烟 → 上传 TestPyPI → 从 TestPyPI 安装验证 → **等待你批准** → 上传 PyPI →
   从 PyPI 安装验证 → 打 tag `v0.1.0` 并创建 GitHub Release。
   注意：该流水线在 CI 中重新构建，产物哈希与本地 `dist/` 不同是正常的；批准时核对日志中打印的
   `SHA256SUMS`。

## 方式 B：本地用 twine 上传 `dist/` 中的这批字节

需要 TestPyPI 与 PyPI 的 API token（`~/.pypirc` 或 `TWINE_USERNAME=__token__` /
`TWINE_PASSWORD=<token>`）。两个包一起上传，因为 `ecginterpret` 依赖 `ecg-records`，
而 `ecg-records[interpret]` 指向 `ecginterpret`。

```bash
# 1) TestPyPI 演练
.venv/bin/twine upload --repository testpypi dist/*.whl dist/*.tar.gz

# 2) 在全新环境从 TestPyPI 安装（依赖从 PyPI 取）并冒烟
python3 -m venv /tmp/ecg-rc && /tmp/ecg-rc/bin/pip install \
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
  "ecg-records[viz,interpret]==0.1.0"
(cd /tmp && /tmp/ecg-rc/bin/python "$OLDPWD/tools/release_smoke.py" --version 0.1.0 --with-extras)

# 3) 确认无误后上传 PyPI（同一批字节，不要重新构建）
.venv/bin/twine upload dist/*.whl dist/*.tar.gz

# 4) 发布后验证
python3 -m venv /tmp/ecg-final && /tmp/ecg-final/bin/pip install "ecg-records[viz,interpret]==0.1.0"
(cd /tmp && /tmp/ecg-final/bin/python "$OLDPWD/tools/release_smoke.py" --version 0.1.0 --with-extras)

# 5) 上传成功后再打 tag（指向构建产物的提交）
git tag -a v0.1.0 1cb2ccc -m "ecg-records 0.1.0, ecginterpret 0.1.0"
git push origin v0.1.0
```

## 发布后

- 在两个 `CHANGELOG.md` 顶部保留 `## [Unreleased]` 开发小节（已存在），不改运行时版本号，
  直到选定下一个版本。
- 兼容窗口：0.1.0 为首次告警版本；旧接口最早在 0.3.0 移除，移除前一个版本改为抛出指向
  替代接口的 `ImportError`（文档 05 Phase 6）。
- 尚需维护者确认的非阻塞事项：`.github/CODEOWNERS` 中的 Validation Owner 账号；文档站点托管
  位置（`mkdocs build` 输出在 `build/site`）；自托管数据集 runner（`parity.yml`）。
