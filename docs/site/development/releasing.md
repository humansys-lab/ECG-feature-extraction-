# Releasing

Releases go to PyPI as the `ecg-records` distribution. A released version can
never be replaced or re-uploaded: a defect is fixed forward in a new version.
Every step below exists to catch problems before that point.

## 1. Prepare the version

1. Choose the version with [Semantic Versioning](../compatibility.md). A schema
   minor needs at least a package minor.
2. Set the version in all three places, which must agree:
   `version` in `feature_extraction/pyproject.toml`, `__version__` in
   `feature_extraction/ecgfeat/__init__.py` and `LIBRARY_VERSION` in
   `feature_extraction/ecgfeat/pipeline/record_builder.py` (written into every
   record's provenance and `record_id`).
3. Move the `## [Unreleased]` entries in `feature_extraction/CHANGELOG.md` into a
   `## [X.Y.Z] - YYYY-MM-DD` section. `tools/check_changelog.py` requires it.
4. If published values changed on purpose, freeze a new golden baseline
   ([Golden corpus](golden-corpus.md)) and point CI at it.

## 2. Pass the gates

On a clean checkout of the release commit:

```bash
python -m pytest -q tests
lint-imports --config feature_extraction/.importlinter
python tools/check_validation_registry.py
python tools/check_changelog.py
python tools/check_performance.py reference --repeats 5
python tools/build_docs.py --check && python tools/check_doc_examples.py && mkdocs build --strict
python snapshot_regression.py --data-root /data check --tier full \
    --baseline-id ecg-records-<baseline> --mode record-bytes     # on the dataset runner
```

## 3. Build and inspect the artifacts

Build from a **clean checkout** (not a working tree with local files), so that
nothing untracked can leak into the sdist:

```bash
git worktree add /tmp/ecg-release <release-commit>
python -m build --outdir dist /tmp/ecg-release/feature_extraction
python tools/check_artifacts.py dist            # required files, no tests/data/secrets, LICENSE, disclaimer
twine check --strict dist/*
(cd dist && sha256sum *.whl *.tar.gz > SHA256SUMS)
```

Then smoke-test clean installs outside the repository, on the oldest and
newest supported Python:

```bash
python3.10 -m venv /tmp/rc310 && /tmp/rc310/bin/pip install "numpy==1.26.*" "scipy==1.11.*" "matplotlib==3.7.*" "$(ls dist/*.whl)[viz]"
(cd /tmp && /tmp/rc310/bin/python "$OLDPWD/tools/release_smoke.py" --version X.Y.Z --with-viz)
```

`release_smoke.py` checks the version, the packaged schema resources, a
canonical round trip and one extraction of a synthetic 12-lead signal. With
`--with-viz` it also draws a plot.

## 4. Publish

### Recommended: GitHub Actions trusted publishing

`.github/workflows/release.yml` builds once, inspects, smoke-tests on Python
3.10 and 3.13, uploads to **TestPyPI**, installs from TestPyPI to verify, then
**waits for a maintainer's approval** before uploading the same bytes to PyPI.
Finally it verifies the PyPI install, tags `ecg-records-vX.Y.Z` and creates a
GitHub release. It uses PyPI trusted publishing (OIDC), so no API tokens are
stored.

One-time setup:

1. In the GitHub repository settings, create the environments `testpypi` and
   `pypi`, and give `pypi` required reviewers.
2. On TestPyPI and PyPI, add a trusted publisher for the `ecg-records` project:
   owner `humansys-lab`, repository `ECG-feature-extraction-`, workflow
   `release.yml`, environment `testpypi` or `pypi` respectively.

Then run *Actions → release → Run workflow* with the version.

### Alternative: manual upload with twine

With API tokens in `~/.pypirc` (sections `pypi` and `testpypi`, username
`__token__`):

```bash
twine upload --repository testpypi dist/*.whl dist/*.tar.gz
python3 -m venv /tmp/rc && /tmp/rc/bin/pip install \
    --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
    "ecg-records[viz]==X.Y.Z"
(cd /tmp && /tmp/rc/bin/python "$OLDPWD/tools/release_smoke.py" --version X.Y.Z --with-viz)

twine upload dist/*.whl dist/*.tar.gz          # the same bytes; never rebuild between the two
```

Verify that PyPI serves the same hashes as `dist/SHA256SUMS`
(`https://pypi.org/pypi/ecg-records/X.Y.Z/json`), then tag the release commit:

```bash
git tag -a ecg-records-vX.Y.Z <release-commit> -m "ecg-records X.Y.Z"
git push origin ecg-records-vX.Y.Z
```

## 5. After the release

- Add a new `## [Unreleased]` section to the changelog.
- Keep the deprecation schedule: legacy names are removed no earlier than
  0.3.0, after a one-release tombstone ([Compatibility](../compatibility.md)).
- Revoke any API token that was scoped to the whole account, and use
  project-scoped tokens or trusted publishing from then on.

## Release history

| Version | Date | Notes |
|---|---|---|
| 0.1.0 | 2026-09-24 | first release; uploaded with twine after a TestPyPI rehearsal; the uploaded hashes match the artifacts built from commit `1cb2ccc`; tag `ecg-records-v0.1.0` |
