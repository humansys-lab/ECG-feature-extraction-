# 发布手册：ecg-records / ecginterpret / ecg-records-viz 0.1.0

本仓库已处于**可发布状态**，但尚未上传任何包（按维护者决定，由维护者本人上传）。
上传到 PyPI/TestPyPI 后版本**不可撤回、不可覆盖**；发现问题只能发布新版本修复。

## 候选产物

- 构建来源：分支 `library-migration` 的提交 `92f6bd2`，在干净检出中构建（文档 07 第 4 步）。
- 位置：`dist/`（已被 `.gitignore` 忽略，不入库），6 个文件 + `SHA256SUMS`：

| 文件 | SHA-256 |
|---|---|
| `ecg_records-0.1.0-py3-none-any.whl` | `34a813f05de901db0ec983cecd5f5ea32699c9a2551914eaaabd4450e851dc8b` |
| `ecg_records-0.1.0.tar.gz` | `27eb36715633976d9f4a5c795b2db91713961b2d68312fdbd9a952ea7499c85c` |
| `ecginterpret-0.1.0-py3-none-any.whl` | `6c5989d544c0fc969ef5603c494f4605ff2a66c9e86841a13463dcdc4515c03e` |
| `ecginterpret-0.1.0.tar.gz` | `553ebce05bbe997fb50f82f9327cd434c6a4bc86897fae05537e521b27c87117` |
| `ecg_records_viz-0.1.0-py3-none-any.whl` | `7b8e9d321c769e6354035178c90b0aeccac7349208c0693a6f8cc59cb0738ea6` |
| `ecg_records_viz-0.1.0.tar.gz` | `522e17078f8dffff92d23544b3237f0b0fb1dcb9770401986483f24bcf0b0f9c` |

已完成的发布前检查（结果见 `docs/library_design/08_implementation_status.md`）：
`tools/check_artifacts.py dist`（含 LICENSE）通过；`twine check --strict` 通过；
Python 3.10/3.11/3.12/3.13 × NumPy 1.26/2.x 干净环境安装测试通过；由 sdist 重建的
wheel 冒烟通过；黄金语料与 9 个评估工具门禁对冻结基线通过。

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
3. 在 TestPyPI 与 PyPI 各为 `ecg-records`、`ecginterpret`、`ecg-records-viz` 添加
   *pending trusted publisher*：owner `humansys-lab`，repository `ECG-feature-extraction-`，
   workflow `release.yml`，environment 分别为 `testpypi` / `pypi`。
4. Actions → `release` → Run workflow，`version = 0.1.0`。流水线依次：构建 → 产物检查 →
   3.10/3.13 冒烟 → 上传 TestPyPI → 从 TestPyPI 安装验证 → **等待你批准** → 上传 PyPI →
   从 PyPI 安装验证 → 打 tag `v0.1.0` 并创建 GitHub Release。
   注意：该流水线在 CI 中重新构建，产物哈希与本地 `dist/` 不同是正常的；批准时核对日志中打印的
   `SHA256SUMS`。

## 方式 B：本地用 twine 上传 `dist/` 中的这批字节

需要 TestPyPI 与 PyPI 的 API token（`~/.pypirc` 或 `TWINE_USERNAME=__token__` /
`TWINE_PASSWORD=<token>`）。三个包一起上传，因为 `ecginterpret` 与 `ecg-records-viz` 依赖
`ecg-records`。

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

# 5) 上传成功后再打 tag
git tag -a v0.1.0 92f6bd2 -m "ecg-records 0.1.0, ecginterpret 0.1.0, ecg-records-viz 0.1.0"
git push origin v0.1.0
```

## 发布后

- 在三个 `CHANGELOG.md` 顶部保留 `## [Unreleased]` 开发小节（已存在），不改运行时版本号，
  直到选定下一个版本。
- 兼容窗口：0.1.0 为首次告警版本；旧接口最早在 0.3.0 移除，移除前一个版本改为抛出指向
  替代接口的 `ImportError`（文档 05 Phase 6）。
- 尚需维护者确认的非阻塞事项：`.github/CODEOWNERS` 中的 Validation Owner 账号；文档站点托管
  位置（`mkdocs build` 输出在 `build/site`）；自托管数据集 runner（`parity.yml`）。
