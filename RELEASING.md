# Releasing ecg-records

The full procedure is in the documentation:
[docs/site/development/releasing.md](docs/site/development/releasing.md).

Checklist:

1. Version in `feature_extraction/pyproject.toml`, `ecgfeat.__version__` and
   `ecgfeat/pipeline/record_builder.py` (`LIBRARY_VERSION`); changelog section
   `## [X.Y.Z] - YYYY-MM-DD`.
2. All gates pass: tests, `lint-imports`, registry, changelog, performance,
   docs, and the golden corpus on the dataset runner.
3. Build from a clean checkout; `tools/check_artifacts.py`, `twine check --strict`,
   `SHA256SUMS`; clean-install smoke tests on Python 3.10 and 3.13.
4. Publish with `.github/workflows/release.yml` (TestPyPI → approval → PyPI),
   or upload the same bytes with twine, TestPyPI first.
5. Verify the PyPI hashes, then tag `ecg-records-vX.Y.Z`.

PyPI versions are immutable: never re-upload, fix forward.
