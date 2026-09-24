# 00 — Overview

Status (2026-09-24): Phases 0–5 of this design are implemented and gated; see
[implementation status and remaining release decisions](08_implementation_status.md).
Design date: 2026-09-22. Companion documents: `01_architecture.md` and
`02_record_json_schema.md`.

## Product decision

Publish a measurement library whose durable interface is a small, versioned ECG
record JSON document. The engine produces fiducials, quality observations and a
bounded selection of measurements. Visualization, feature queries, reports,
diagnostic rules and LLM agents consume that JSON together with the raw signal.
They must not depend on extraction objects or private algorithm functions.

Keep the Python import name `ecgfeat` during migration. Select a new distribution
name independently of imports. The proposed interchange format is called
**ECG Record**, with its own schema version and a separately versioned
interpretation document. The record API now exists in shadow mode; it has not
replaced legacy consumers or separated the interpretation distribution.

The first size acceptance criterion is **at most 24,000 uncompressed UTF-8 bytes**
for the standard record profile of the supplied 10 s, 500 Hz, 12-lead, ten-beat
case. This is over 100 times smaller than the supplied 2,603,976-byte legacy
export. Rich derived measurements, waveforms and interpretations have explicit
separate products; silently hiding them in a new `metadata` bag is prohibited.
Document 02 defines the retained fields, accounting and profile semantics.

## Intended users and use cases

| User | Workflow | Required product behavior |
|---|---|---|
| ECG algorithm researcher | Compare delineation and measurement algorithms against annotations | Stable sample coordinates, explicit paths/configuration, field-specific validation evidence |
| Dataset and batch engineer | Process many thousands of records and retain reproducible results | Compact JSON, deterministic identities, bounded profiles and independently stored raw signals |
| Visualization or reporting developer | Overlay boundaries, inspect quality and show selected measurements | Read `(signal, record)` without importing the measurement engine |
| Feature-analysis developer | Select per-lead, per-beat or global quantities | Units, missingness and stable addresses survive serialization |
| Research diagnostic-rule or agent developer | Cite measured evidence and abstain when evidence is unsuitable | Addressable values and availability/validation gates; independent interpretation output |

Standard 12-lead records are the primary measurement contract. An explicit limited
input mode accepts 1–8 named channels; it retains raw measurements while disabling
diagnosis, axis and formal QT reporting. A missing required lead is not a silent
invitation to synthesize a full 12-lead contract. These input constraints are
supplied project facts; architecture and schema documents specify their handling.

## Intended use and non-goals

**Research and engineering software; not a medical device.** Outputs are algorithmic
observations for research, software development and validation. They are not a
clinical diagnosis, a treatment recommendation, a monitoring alarm, or a substitute
for qualified clinical assessment. Include this statement in package metadata,
the documentation landing page and the machine-readable intended-use identifier.
Interpretation consumers must preserve the same scope in their reports.

Do not claim equivalence to proprietary Philips DXL or Glasgow implementations.
The existing README describes a DXL-inspired approach based on public material,
not a reproduction of a proprietary algorithm (`feature_extraction/README.md`,
“Important notes”). The new name should describe the capability, not a vendor.

Non-goals for the initial publication are streaming bedside operation, a clinical
workflow application, learned diagnostic models, dataset redistribution, and
rewriting already useful signal-file readers or general plotting frameworks.
The design itself does not validate new outputs. The implementation pass adds a
record boundary while retaining existing numerical defaults; see document 08
for the precise implemented subset and pending structural migration.

A successful delineation benchmark does not validate every derived field or
clinical statement. Carry a validation tier and evidence reference for each field
or homogeneous field group. In particular, the supplied validation history says
the four-class ST morphology field was never validated and is anti-correlated
with ischemia. Keep it explicitly unvalidated and ineligible as ischemia evidence;
do not promote it through packaging or a confidence score.

## Positioning and reuse

The comparison concerns intended contracts, not a claim of superior accuracy.
Current performance and inventory numbers below are supplied project measurements,
not measurements repeated for this design.

| Library | Existing emphasis | ECG Record's additional commitment | Adopt or reuse; avoid copying |
|---|---|---|---|
| NeuroKit2 0.2.13 | Broad physiological processing and a discoverable flat `nk.ecg_*` surface; ECG processing conventions are examined from installed source in document 01 | Multi-lead record identity, compact on-disk schema, per-field provenance/validation, explicit input applicability | Adopt convenient functions and explicit algorithm selection; do not make a DataFrame or all of NeuroKit's imported dependencies the core boundary |
| `wfdb` | Reading, writing and processing waveform records and annotations, with plotting and signal-processing utilities [W] | A measurement-result contract above the acquisition and annotation formats | Use an optional WFDB adapter and established sample-index conventions; retain current `.hea`/`.mat` compatibility |
| `py-ecg-detectors` | A collection of QRS/heartbeat detectors and HRV tools [D] | Multi-lead delineation, amplitudes, intervals, quality and field-level evidence in one interchange contract | Use as an optional research comparator; do not replace validated defaults merely to wrap another detector |
| `ecg-plot` | ECG chart layout, custom lead order and PNG/SVG output [P] | Measured fiducial overlays and availability-aware annotations tied to immutable evidence | Reuse layout conventions or an optional adapter; keep plotting dependencies outside engine imports |

The inspected NeuroKit2 top-level `__init__.py` re-exports the ECG namespace and
imports pandas, matplotlib and sklearn alongside NumPy/SciPy. That is evidence
for namespace convenience and dependency coupling, not evidence of a versioned
JSON schema. Document 01 examines `ecg_process`, `ecg_delineate` and `ecg_plot`
before making more specific comparisons.

External primary references, consulted 2026-09-22:

- [W] WFDB documentation: <https://wfdb.readthedocs.io/en/latest/>.
- [D] Project README and detector API: <https://github.com/berndporr/py-ecg-detectors>.
- [P] Project README and plot examples: <https://github.com/dy1901/ecg_plot>.

Reuse means interoperability first. Copying code or making a mandatory dependency
requires a separate licensing and maintenance decision; no such changes occur here.

## Existing investment and compatibility

The supplied inventory is 46 top-level modules and approximately 40,523 lines,
plus `clinical_rules` and `glasgow_rules`, under `feature_extraction/ecgfeat/`.
The current project metadata declares `ecgfeat-dxl-inspired==0.1.0`, Python >=3.10,
NumPy >=1.24 and SciPy >=1.10, with optional `performance`/Numba >=0.57
(`feature_extraction/pyproject.toml:[project]`). Preserve that small mandatory
runtime dependency set for the measurement distribution.

Today's top-level exports combine extraction, interpretation, JSON conversion,
file loading, plots and dataclasses (`ecgfeat/__init__.py:__all__`). The supplied
legacy default is 2,603,976 bytes with 15 sections; `summary` is still 645,273
bytes with 13 sections. A renamed `to_dict()` does not solve that problem.

Introduce an additive record export and public record reader first. Preserve
legacy extraction objects, imports and `to_dict()` until their consumers migrate.
The supplied 128 test files live in repository-root `tests/`, outside the package;
`ecgagent/`, batch extraction and analysis scripts also rely on existing data.
Do not require them to adopt a new JSON shape merely by upgrading a patch release.
The codebase guide identifies `ecgagent` as a pointer-addressed evidence consumer
(`docs/codebase_guide.en.md`, “Directory and Responsibilities”).

Defaults must retain the established measurement path. Experimental refinement
flags and ST source selection require explicit opt-in and resolved configuration
in the result. The README specifically warns that `RefinementConfig.experimental()`
can degrade combined results and that `calibrated_pr` updates side-channel fields
rather than all native measurements (`feature_extraction/README.md`, lines 55–78).

## Distribution naming

| Candidate | Reason | PyPI JSON API result on 2026-09-22 |
|---|---|---|
| **`ecg-records` — recommended** | Names the durable record boundary; accommodates limited leads and downstream measurement tooling | HTTP 404; no project returned |
| `ecg-measurements` | Clear functional name for researchers; less emphasis on interchange | HTTP 404; no project returned |
| `ecg-delineation` | Precise algorithm-facing name; understates quality and measurement scope | HTTP 404; no project returned |

These results came from live requests to `https://pypi.org/pypi/<name>/json`.
A 404 is not a reservation or proof that PyPI will accept the name. Recheck at
publication and confirm ownership, name-conflict and licensing questions then.
The current distribution name is syntactically usable but unsuitable product
positioning: it foregrounds a proprietary algorithm family and an exploratory
implementation rather than a stable research interface. No package was published.

## Acceptance criteria for this design

The architecture assigns every supplied module a home and prevents consumers from
importing engine internals. The record schema provides bounded serialized size,
explicit missingness and input applicability, reproducibility, field validation
and stable evidence paths. Interpretation evolves separately. Compatibility
adapters protect existing users while enabling a clean new consumer interface.

## Open questions

The final distribution name and release license need maintainer decisions; the
PyPI check is only a point-in-time observation. Dataset-specific evidence must be
linked to individual algorithms/fields before assigning benchmark tiers; this
pass does not independently audit the supplied validation history. Publication
metadata, wheel contents and the full migration schedule belong to later work.

## Following pass

Documents 03–07 are planned, not produced here: `03_public_api.md` will specify the
public API; `04_module_implementation_guide.md` will guide implementation seams;
`05_migration_plan.md` will sequence compatibility work; `06_testing.md` will
specify contract, regression and size checks; and `07_packaging.md` will cover
builds, dependencies, licensing and publication. Those documents should implement
the decisions in 00–02 without reopening the stable JSON boundary by default.
