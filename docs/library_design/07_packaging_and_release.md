# 07. Packaging and release

> Implementation status (2026-09-24): migration Phases 0–5 of this design are implemented and
> gated; [document 08](08_implementation_status.md) records the results, deliberate deviations and
> the maintainer decisions still needed before release. Document 02 is authoritative for the JSON
> layout; Python carrier sketches must not create a second wire format.

## Purpose

The redesigned library should ship as a small core record/extraction distribution with
independently versioned interpretation and visualization distributions. The import name
`ecgfeat` is retained during migration, but the core distribution name becomes
`ecg-records`.

The current package is `ecgfeat-dxl-inspired 0.1.0`, uses a flat setuptools layout, declares
only `numpy>=1.24` and `scipy>=1.10`, has no `LICENSE`, and has no CI configuration. The
measured development environment is Python 3.12.13 with NumPy 2.2.6, SciPy 1.18.0, Numba
0.65.0, and Matplotlib 3.11.0. Publication must not proceed until licensing and automated
release gates exist.

## Distribution topology

Publish three distributions.

| Distribution | Import surface | Contains | Does not contain |
| --- | --- | --- | --- |
| `ecg-records` | `ecgfeat` | ECG Record models/schema resources, canonical JSON encode/decode, raw-signal input boundary, eight-stage extraction pipeline, four policy objects, `ecgfeat/_engine/*`, provenance/validation metadata, NPZ-sidecar support | Interpretation rules, report rendering, plotting UI |
| `ecginterpret` | `ecginterpret` | The current ~8k-line interpretation/rule subsystem: `interpret.py`, `clinical_rules/`, `glasgow_rules/`, `statement_engine.py`, interpretation document schema, and adapters from ECG Record to interpretation input | Core signal extraction and visualization |
| `ecg-records-viz` | `ecgrecords_viz` | Plotting, waveform overlays, annotation display, record visualization, and Matplotlib-dependent helpers | Extraction algorithms and interpretation rules |

The core distribution exposes convenience extras named `interpret` and `viz`, but those
extras install the separate distributions; they do not move their code back into the core
wheel. For example, `ecg-records[interpret]` installs a compatible `ecginterpret`, and
`ecg-records[viz]` installs `ecg-records-viz`.

This split is preferable to one distribution containing all code behind extras. Python extras
only make dependencies optional; they do not create an independently releasable code boundary.
Keeping interpretation separate allows its rule changes and interpretation-document schema to
version independently from ECG Record extraction. Keeping visualization separate prevents
Matplotlib and rendering code from enlarging or destabilizing the core runtime. The core wheel
therefore stays suitable for services that only need deterministic record production.

The distributions may share a release train when convenient, but they do not share a version
number by rule.

## Core `pyproject.toml`

Use setuptools initially. Changing build backends while simultaneously restructuring the
package adds release risk without solving a current problem. The first public core package
should use the following near-complete design:

```toml
[build-system]
requires = ["setuptools>=75,<82", "wheel>=0.44,<1"]
build-backend = "setuptools.build_meta"

[project]
name = "ecg-records"
version = "0.1.0"
description = "Versioned ECG Record extraction and measurement library"
readme = "README.md"
requires-python = ">=3.10,<3.14"
license = { file = "LICENSE" }
authors = [
  { name = "ECG Records maintainers" },
]
keywords = ["ecg", "electrocardiography", "signal-processing", "physionet"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: Apache Software License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Typing :: Typed",
]
dependencies = [
  "numpy>=1.26,<3",
  "scipy>=1.11,<2",
]

[project.optional-dependencies]
performance = [
  "numba>=0.59,<0.67",
]
wfdb = [
  "wfdb>=4.1,<5",
]
viz = [
  "ecg-records-viz>=0.1,<1",
]
interpret = [
  "ecginterpret>=0.1,<1",
]
dev = [
  "build>=1.2,<2",
  "hypothesis>=6.100,<7",
  "jsonschema>=4.22,<5",
  "pytest>=8,<9",
  "pytest-cov>=5,<8",
  "ruff>=0.6,<1",
  "twine>=5,<7",
]

[project.urls]
Documentation = "https://<project-docs-host>/"
Repository = "https://<repository-host>/<project>"
Issues = "https://<repository-host>/<project>/issues"

[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
include = ["ecgfeat*"]

[tool.setuptools.package-data]
ecgfeat = [
  "py.typed",
  "schemas/**/*.json",
  "schemas/**/*.toml",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

The placeholder project URLs must be replaced with the real repository and documentation URLs
before release. Maintainer names should likewise be real project metadata rather than the
placeholder shown above.

### Dependency bounds

The NumPy declaration must change materially from the current `numpy>=1.24`. The project is
already being developed against NumPy 2.2.6, while NumPy 1.x remains part of the intended CI
matrix. Declare `numpy>=1.26,<3` and test both 1.26.x and supported 2.x releases. NumPy 1.26
is the recommended 1.x floor because it is the final NumPy 1.x family and gives one coherent
compatibility target across the supported Python range. The `<3` cap prevents an untested
future major version from being accepted automatically.

Declare `scipy>=1.11,<2`. The lower bound is a deliberate increase from the current
`>=1.10`; the release should support a bounded modern SciPy family rather than promise
compatibility with combinations that are not in the CI matrix. The `<2` cap applies the same
major-version rule. On each supported Python version, CI must install the oldest feasible SciPy
within this declared range and the newest allowed release. If Python 3.13 cannot satisfy the
oldest global SciPy floor with a supported wheel, that matrix cell uses the oldest SciPy
release supporting Python 3.13 and records it explicitly.

The Numba bound is optional because Numba is only a performance accelerator. `numba>=0.59,<0.67`
includes the measured 0.65.0 environment while limiting installation to a tested family.
Before release, CI must prove that `ecg-records` works without Numba and that the performance
extra produces equivalent published measurements.

The `wfdb`, visualization, and interpretation dependencies are extras because none is needed
to construct or decode a core ECG Record. Exact lower bounds for the two new sibling
distributions become their first published versions; the `<1` constraint prevents automatic
adoption of their first incompatible major release.

## Src-layout decision

Move the core distribution to a src layout:

```text
pyproject.toml
src/
  ecgfeat/
    __init__.py
    py.typed
    record/
    pipeline/
    _engine/
    schemas/
tests/
```

The current package discovery uses `where=["."]`, while all 124 test files live outside the
package. A src layout makes tests import the installed package rather than accidentally
importing the repository checkout. That matters during this migration because the import name
`ecgfeat` is retained even while the distribution name changes and modules are relocated.

CI must build a wheel first and run at least one full non-dataset test job against that wheel.
A source-tree-only test run is insufficient for release.

## Typing marker

Ship an empty `src/ecgfeat/py.typed` file in the wheel and include it through
`[tool.setuptools.package-data]`. CI must inspect the built wheel and fail unless
`ecgfeat/py.typed`, the ECG Record JSON Schema files, and the validation-evidence registry are
present.

The presence of `py.typed` is a promise that exported annotations are part of the package
interface. Public typing regressions should therefore be reviewed like API regressions.

## Versioning

### Code versions

Use Semantic Versioning for each distribution independently:

- patch: implementation fixes with no documented public API or schema change;
- minor: backward-compatible public Python API additions or behavior additions;
- major: incompatible Python API or runtime-behavior changes.

Pre-1.0 releases may still change quickly, but version increments must preserve the same
meaning; `0.y` is not permission to alter the ECG Record schema without versioning it.

### Schema versions

The ECG Record schema has its own `MAJOR.MINOR.PATCH` SemVer version embedded in every record and in its
address:

`ecg-record:<record_id>@<schema_version>#<RFC 6901 JSON pointer>`.

Schema versions are separate from the `ecg-records` package version.

- Schema **patches** correct schema descriptions or constraints without changing
  field semantics. Purely editorial prose changes do not require a schema bump.
  The initial emitted version is `1.0.0`, as specified by document 02; the
  `schemas/ecg-record/1.0/` directory identifies its major/minor family.
- A schema **minor** adds only backward-compatible material: optional fields, optional enum
  values where consumers are required to tolerate unknown values, or additional provenance
  that older readers can safely ignore.
- A schema **major** is required when a field is removed, renamed, retyped, changes unit or
  meaning, changes required/optional status incompatibly, or changes an absence-state
  interpretation.

A schema minor bump forces at least a **code minor** bump in `ecg-records`, because the core
package is gaining a new published record capability. A schema major bump forces a **code
major** bump. This rule applies even if the underlying extraction algorithm did not change.

A code major bump does not automatically require a schema major bump; Python APIs may change
while the serialized record contract remains compatible.

### Consumer compatibility statement

For every stable release line, publish this guarantee:

> `ecg-records X.y` can read every ECG Record schema version in the supported schema-major
> set declared in its metadata. Within a supported schema major, readers accept later
> backward-compatible schema minors by ignoring fields they do not understand. Writers emit
> exactly one documented schema version by default. A consumer that relies only on published
> fields from schema `N.m` may process records from later `N.*` versions without changing
> the meaning, unit, or absence semantics of those existing fields.

The package must expose its emitted schema version and supported read-version range as
constants or metadata, and contract tests must verify the statement.

The interpretation document has its own separately versioned schema owned by
`ecginterpret`. It must not reuse the ECG Record schema version number as its release
version.

## License and medical disclaimer

### License choice

Use **Apache License 2.0** for the new distributions, subject to confirmation that every
existing source file and incorporated dependency permits relicensing under those terms.

Apache-2.0 is a good fit for a research/engineering library because it is permissive,
commercially usable, includes an explicit patent grant and patent-termination provision, and
requires preservation of license/notice information without imposing copyleft on downstream
applications.

There is currently no `LICENSE` file anywhere in the repository. That is a publication
blocker. Do not upload any distribution to TestPyPI or PyPI as a release candidate until a
human maintainer has confirmed rights to license the existing code and committed the complete
Apache-2.0 `LICENSE` text. If provenance review finds code that cannot be distributed under
Apache-2.0, resolve that code first; do not publish under an assumed license.

### Disclaimer placement

Use one canonical disclaimer text, reviewed together with the license decision:

> This software is research and engineering software. It is not a medical device and is not
> intended to diagnose, treat, cure, or prevent disease. Outputs require independent
> validation for the intended use and must not be used as a substitute for professional
> medical judgment.

Place it in:

1. the top-level `README.md`, close to installation/usage material;
2. the documentation site's landing page and a dedicated safety/limitations page;
3. package metadata, through a concise equivalent in the long description loaded from the
   README;
4. the interpretation package README/docs as well as the core package README/docs.

Do **not** inject the disclaimer into every runtime call, log line, measurement, or JSON
field. Repeating legal prose in machine-readable output would bloat the record and make the
scientific provenance unstable.

Runtime provenance should instead contain a stable software identity and an intended-use
classification such as `intended_use = "research_and_engineering"`, plus package/code/schema
versions. The schema documentation defines that value and links it to the canonical
limitations text. Provenance must never say or imply that a record is medically cleared,
approved, or clinically validated as a whole.

The license and disclaimer solve different problems: the license grants rights to use the
software; the disclaimer states intended use and limitations. Both must be settled before
publication.

## Release process

No release is made from an uncommitted working tree or from a developer environment that has
not passed CI. The release checklist is:

1. **Select release versions — human decision.** Choose the new `ecg-records` code version
   and, when changed, the ECG Record schema version. Verify the code/schema bump rule above.
2. **Finalize changelog — human-authored, CI-checked.** Update `CHANGELOG.md` with user-visible
   Python API changes, schema changes, validation-tier changes, benchmark changes, dependency
   bounds, and known limitations. CI fails if the release version has no changelog section.
3. **Verify validation evidence — automated gate with human baseline ownership.** Run all
   gates from `06_testing_and_validation.md`. Every required benchmark must reference the
   approved pinned baseline; validation-tier changes require evidence.
4. **Build from a clean checkout — automated.** Run `python -m build`. The build must create
   exactly the expected sdist and wheel and complete without warnings classified as packaging
   errors.
5. **Inspect artifacts — automated.** Run `twine check dist/*`; inspect wheel contents; fail
   unless `LICENSE`, README metadata, `ecgfeat/py.typed`, schema resources, and the
   validation-evidence registry are included and no tests, datasets, caches, or secrets are
   packaged.
6. **Install artifacts in clean environments — automated.** Install the wheel into clean
   Python 3.10 and 3.13 environments and run package import, schema-resource, canonical
   encode/decode, and one representative extraction smoke test. Repeat one smoke install from
   the sdist-built wheel.
7. **Dependency-boundary test — automated.** Resolve the declared oldest supported dependency
   family and newest allowed family. At least one environment must use NumPy 1.26.x and one a
   supported NumPy 2.x. Both must pass the core contract suite.
8. **TestPyPI rehearsal — automated upload, human promotion decision.** Upload the exact
   candidate artifacts to TestPyPI using trusted publishing. Create a clean environment that
   installs from TestPyPI plus normal PyPI for dependencies, then run the installation smoke
   suite. Any artifact rebuild after this step invalidates the rehearsal.
9. **Publish to PyPI — human-triggered, automated upload.** A maintainer approves the exact
   artifact hashes rehearsed on TestPyPI. The release workflow publishes those bytes using
   PyPI trusted publishing. Local API tokens are not part of the release procedure.
10. **Tag/release notes — automated after successful publication.** Create the signed/versioned
    release tag and repository release notes from the approved version/changelog only after
    the package upload succeeds. If project policy requires the tag to trigger trusted
    publishing, create the protected tag after every pre-publication gate passes and configure
    the workflow so failed publication cannot advance any mutable release marker.
11. **Post-release verification — automated with human review of failures.** In a clean
    environment, install `ecg-records==<released-version>` from PyPI, import `ecgfeat`,
    load the bundled schema, encode/decode the pinned smoke record, and assert the reported
    package/schema versions. Verify the documentation site resolves the released API and
    schema reference.
12. **Open next-development section — automated or maintainer edit.** Start the next
    `CHANGELOG.md` development section without changing runtime version metadata until a
    subsequent release is selected.

Release artifacts are immutable. If post-release verification finds a defect, publish a new
patch release; never replace an existing wheel or sdist.

### Automation boundary

CI automates tests, benchmarks, builds, artifact inspection, clean-environment installs,
TestPyPI upload/rehearsal, final upload mechanics, and post-release smoke checks. Humans own
license approval, baseline approval, validation-tier promotion, version selection, release
notes, and the final decision to publish the already verified artifact hashes.

PyPI and TestPyPI publishing should use OpenID Connect trusted publishing from a protected CI
environment. The repository currently has no `.github/workflows`; creation and protection of
those workflows is therefore a prerequisite to the first release.

## Documentation site

Publish one versioned documentation site for the core package and separate versioned sites or
clearly separated version selectors for `ecginterpret` and `ecg-records-viz`.

The core site must contain:

- installation and dependency/extras guidance;
- a migration guide from `ecgfeat-dxl-inspired` and legacy `ecgfeat.models` /
  `ecgfeat.export` consumers;
- ECG Record concepts, three absence states, addressing, field tiers, profiles, and NPZ
  sidecars;
- generated Python API documentation for public `ecgfeat.record` and
  `ecgfeat.pipeline` interfaces;
- a generated ECG Record schema reference with field type, unit, publication tier,
  validation tier, provenance semantics, and JSON pointer;
- benchmark methodology and the released validation-evidence matrix;
- limitations, intended-use statement, and the medical disclaimer;
- changelog and compatibility policy.

Generate Python API pages from importable public objects and their docstrings using one
documentation toolchain; do not hand-maintain duplicate signatures. Private
`ecgfeat._engine` APIs may be documented for contributors but are explicitly excluded from
the compatibility promise.

The schema reference must be generated from the **same bundled schema files and
`validation-evidence.json` registry that the package ships**. CI builds the docs from the
wheel candidate, extracts those resources, generates the reference, and fails if the generated
schema pages differ from checked/generated documentation artifacts. This prevents the site
from describing a field, validation tier, or schema version that the released wheel does not
contain.

Each published docs version is immutable and maps to one `ecg-records` release. The site
shows both code version and emitted ECG Record schema version prominently.

## Release blockers for the first public package

The first public `ecg-records` release is blocked until all of these conditions are true:

- a human-approved `LICENSE` exists and code provenance is compatible with it;
- the medical disclaimer is present in README/package docs;
- CI exists and the Python/NumPy matrix passes;
- the 124-test migration has a passing core subset plus contract tests for the new record
  boundary;
- the 24,000-byte summary test passes on the pinned 10-second, 12-lead record;
- all release-required dataset benchmarks pass against an approved baseline;
- validation-tier evidence is mechanically checked;
- a wheel and sdist pass artifact inspection and clean-install tests;
- the exact candidate bytes pass a TestPyPI rehearsal.

Any false item blocks publication.

## Open questions

1. Can every existing source file be relicensed under Apache-2.0 by the current rights holders?
   This requires a human provenance review; the repository's lack of any license prevents an
   assumption.
2. What are the permanent project repository, documentation, and issue-tracker URLs? They must
   replace the placeholders in package metadata before the first build intended for release.
3. Should the first stable package support Python 3.13 immediately? This design recommends
   yes, but the declared support range must be reduced before release if SciPy, Numba, or the
   complete test matrix cannot satisfy the same gates on 3.13.
4. What first versions will be assigned to `ecginterpret` and `ecg-records-viz`? The
   proposed core extras assume `>=0.1,<1`; update the specifiers to match the versions
   actually published together.
5. Which documentation generator and hosting service will be used? The release requirement is
   tool-independent: API signatures come from installed public objects, and schema pages come
   from the exact schema/evidence resources in the release artifact.
