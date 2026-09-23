"""Golden manifest definition: pinned configurations and deterministic selection.

Selection rule (``ecg-records-golden-v1``): for each dataset, list eligible
record ids with :func:`benchmarks.golden.datasets.discover`, rank them by
``sha256("ecg-records-golden-v1:<dataset>:<record>")`` (hex, ascending), and
take the first N ids whose adapter loads successfully.  Load failures are kept
in the manifest as explicit exclusions.  The rule never looks at extraction
output, so it cannot cherry-pick favourable records.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .datasets import discover, load_case

SELECTION_RULE = "ecg-records-golden-v1"

SENTINEL_COUNTS = {"ludb": 24, "qtdb": 20, "edb": 16, "ptbxl": 16, "but": 8, "nstdb": 6, "gudb": 6}
# Full tier: every eligible record, except PTB-XL where the local 10,000-record
# mirror is ranked and capped to keep the controlled-runner job bounded.
FULL_CAPS = {"ptbxl": 500}

REFINEMENT_FLAGS = (
    "p_model_arbitration", "p_multiple_candidates", "p_boundary_correction",
    "t_onset_change_point", "t_bidirectional", "t_correlated_fusion",
    "qrs_terminal_multiscale", "qrs_quality_reference", "representative_robust",
    "grouping_outliers", "p_phasor_candidates", "t_boundary_projection",
    "t_sequence_selection", "t_projection_offset_only", "t_sequence_offset_only",
    "qrs_adaptive_consensus", "p_pathology_candidates", "atrial_event_validation",
)


def _config(**overrides: Any) -> dict[str, Any]:
    spec = {"fs_internal": None, "mains_freq": 50, "input_mode": "standard",
            "st_amplitude_source": "analysis", "refinement": {}}
    spec.update(overrides)
    return spec


CONFIGS: dict[str, dict[str, Any]] = {
    "standard-default-v0": _config(),
    "limited-default-v0": _config(input_mode="limited"),
    "st-calibrated-pr-v0": _config(st_amplitude_source="calibrated_pr"),
    "st-adaptive-pr-tp-v0": _config(st_amplitude_source="adaptive_pr_tp"),
    **{f"refine-{flag}-v0": _config(refinement={flag: True}) for flag in REFINEMENT_FLAGS},
}


def rank_key(dataset: str, record: str) -> str:
    return hashlib.sha256(f"{SELECTION_RULE}:{dataset}:{record}".encode()).hexdigest()


def ranked(dataset: str, data_root: Path) -> list[str]:
    return sorted(discover(dataset, data_root), key=lambda record: rank_key(dataset, record))


def _pick(dataset: str, data_root: Path, count: int | None, mode: str = "standard"):
    chosen, excluded = [], []
    for record in ranked(dataset, data_root):
        if count is not None and len(chosen) >= count:
            break
        try:
            load_case(dataset, record, data_root, mode=mode)
        except Exception as exc:  # eligibility is decided by the adapter alone
            excluded.append({"dataset": dataset, "record": record, "mode": mode,
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        chosen.append(record)
    return chosen, excluded


def _case(config: str, dataset: str, record: str) -> dict[str, Any]:
    return {"case_id": f"{config}/{dataset}/{record}", "config": config,
            "dataset": dataset, "record": record}


def build_cases(tier: str, data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    if tier == "full":
        for dataset in SENTINEL_COUNTS:
            chosen, excluded = _pick(dataset, data_root, FULL_CAPS.get(dataset))
            exclusions += excluded
            cases += [_case("standard-default-v0", dataset, record) for record in chosen]
        return cases, exclusions
    if tier != "sentinel":
        raise ValueError(tier)

    picks: dict[str, list[str]] = {}
    for dataset, count in SENTINEL_COUNTS.items():
        picks[dataset], excluded = _pick(dataset, data_root, count)
        exclusions += excluded
        cases += [_case("standard-default-v0", dataset, record) for record in picks[dataset]]

    # limited: two ranked records from each two-channel dataset, plus LUDB I/II/V2.
    for dataset in ("qtdb", "edb", "but", "nstdb", "gudb", "ludb"):
        chosen, excluded = _pick(dataset, data_root, 2, mode="limited")
        exclusions += excluded
        cases += [_case("limited-default-v0", dataset, record) for record in chosen]

    # ST sources: ST-relevant EDB first, then 12-lead references.
    st_records = [("edb", r) for r in picks["edb"][:3]] + [("ludb", r) for r in picks["ludb"][:2]] \
        + [("ptbxl", picks["ptbxl"][0])]
    for config in ("st-calibrated-pr-v0", "st-adaptive-pr-tp-v0"):
        cases += [_case(config, dataset, record) for dataset, record in st_records]

    # One refinement flag at a time; never an all-flags configuration.
    for flag in REFINEMENT_FLAGS:
        config = f"refine-{flag}-v0"
        cases += [_case(config, "ludb", picks["ludb"][0]), _case(config, "qtdb", picks["qtdb"][0])]
    return cases, exclusions
