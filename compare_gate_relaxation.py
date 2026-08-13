#!/usr/bin/env python3
"""Compare two compact-workflow batches over the same records and labels.

The point of the comparison is to separate three things the single headline
F1 conflates: how many candidates the model formed, how many the adjudication
gate let through, and how many of those matched PTB-XL.  Every level is scored
with the same category map so the columns are directly comparable.

Diagnosis codes are alias-normalised first.  `DIAGNOSIS_CATALOG` carries
near-synonym pairs (`atrial_fibrillation` / `atrial_fibrillation_pattern`),
the model uses either, and `CATEGORIES` recognises only one of each -- without
folding them a correct call scores as a false positive.

Usage::

    python3 compare_gate_relaxation.py BASELINE_DIR CANDIDATE_DIR
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ecgagent.agent import diagnostic_prompts
from evaluate_target_ecgfeat_diagnosis import CATEGORIES

# Same clinical entity, two catalog codes. Fold to the variant `CATEGORIES`
# scores so a correct call is not counted as a miss and a false positive.
CODE_ALIASES = {
    "atrial_fibrillation": "atrial_fibrillation_pattern",
    "atrial_flutter": "atrial_flutter_pattern",
    "first_degree_av_block": "first_degree_av_delay",
    "left_bundle_branch_block": "lbbb_pattern",
    "complete_av_block": "complete_av_block_pattern",
    "tachycardia": "sinus_tachycardia",
    "bradycardia": "sinus_bradycardia",
    "probable_lafb_pattern": "lafb_pattern",
    "t_wave_abnormality": "primary_t_wave_abnormality",
}

_TOPIC_TO_CODE: dict[str, str] = {}
for _code, (_topic, _category) in diagnostic_prompts.DIAGNOSIS_CATALOG.items():
    _TOPIC_TO_CODE.setdefault(_topic, _code)


def _normalise(codes: Iterable[str]) -> set[str]:
    return {CODE_ALIASES.get(str(code), str(code)) for code in codes if code}


def _categories(codes: set[str], *, reference: bool) -> set[str]:
    return {
        spec.key
        for spec in CATEGORIES
        if codes & set(spec.ptbxl_refs if reference else spec.prediction_codes)
    }


def load_batch(directory: Path) -> dict[str, dict[str, Any]]:
    """Confirmed / differential / abstained code sets per record."""

    records: dict[str, dict[str, Any]] = {}
    for path in sorted((directory / "diagnoses").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        verdict = payload.get("verdict") or {}
        confirmed = _normalise(
            str(row.get("code") or "") for row in verdict.get("diagnoses") or []
        )
        differential = _normalise(
            str(row.get("code") or "")
            for row in verdict.get("differential_diagnoses") or []
        )
        abstained = _normalise(
            _TOPIC_TO_CODE.get(str(row.get("topic") or ""), "")
            for row in verdict.get("abstentions") or []
        )
        decisions = (
            (payload.get("audit") or {}).get("compact_decision") or {}
        ).get("decisions") or []
        steps = [
            step
            for decision in decisions
            for step in decision.get("pathway_steps") or []
        ]
        # Rejected candidates appear in no verdict list, so a union built only
        # from confirmed/differential/abstained silently shrinks as the gate
        # makes more decisions.  Carry them separately: "everything the model
        # weighed" is only comparable across runs when refutations are in it.
        rejected = _normalise(
            str(decision.get("code") or "")
            for decision in decisions
            if str(decision.get("final_placement") or "") == "rejected"
        )
        records[str(payload.get("record_id") or path.stem)] = {
            "ok": bool(payload.get("ok")),
            "verified": bool(payload.get("verified")),
            "confirmed": confirmed,
            "differential": differential,
            "abstained": abstained,
            "rejected": rejected,
            "steps": steps,
            "placements": collections.Counter(
                str(decision.get("final_placement") or "") for decision in decisions
            ),
        }
    return records


def load_reference(directory: Path) -> dict[str, set[str]]:
    manifest = json.loads(
        (directory / "record_manifest.json").read_text(encoding="utf-8")
    )
    return {
        str(row["record"]): set(row.get("reference_codes_active") or [])
        for row in manifest.get("records") or []
    }


def score(
    records: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, set[str]],
    *,
    level: str,
    direct_only: bool = True,
) -> dict[str, Any]:
    keys = {
        "confirmed": ("confirmed",),
        "confirmed+differential": ("confirmed", "differential"),
        "considered (no rejects)": ("confirmed", "differential", "abstained"),
        "considered + rejected": (
            "confirmed",
            "differential",
            "abstained",
            "rejected",
        ),
    }[level]
    tp = fp = fn = 0
    for spec in CATEGORIES:
        if direct_only and spec.semantic_scope != "direct":
            continue
        for record_id, row in records.items():
            predicted: set[str] = set()
            for key in keys:
                predicted |= set(row[key])
            reference_positive = spec.key in _categories(
                reference.get(record_id, set()), reference=True
            )
            predicted_positive = spec.key in _categories(predicted, reference=False)
            if reference_positive and predicted_positive:
                tp += 1
            elif predicted_positive:
                fp += 1
            elif reference_positive:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "p": precision, "r": recall, "f1": f1}


def _fmt(value: Any) -> str:
    return "  N/A" if value is None else f"{value:.3f}"


_LABEL_WIDTH = 34


def _row(label: str, stats: Mapping[str, Any]) -> str:
    return (
        f"{label:<{_LABEL_WIDTH}} {stats['tp']:3d} {stats['fp']:4d} "
        f"{stats['fn']:4d}  {_fmt(stats['p'])}  {_fmt(stats['r'])}  "
        f"{_fmt(stats['f1'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--all-scopes", action="store_true")
    args = parser.parse_args()

    reference = load_reference(args.baseline)
    batches = {
        "baseline": load_batch(args.baseline),
        "candidate": load_batch(args.candidate),
    }
    shared = sorted(set(batches["baseline"]) & set(batches["candidate"]))
    print(f"records compared: {len(shared)}")
    for name, records in batches.items():
        done = sum(1 for r in records.values() if r["verified"])
        print(f"  {name:<10} {len(records):3d} results, {done:3d} verified")

    for scope_direct in ([True] if not args.all_scopes else [True, False]):
        print()
        print(f"--- direct families ---" if scope_direct else "--- all families ---")
        print(
            f"{'level':<{_LABEL_WIDTH}} {'TP':>3} {'FP':>4} {'FN':>4}"
            f"  {'P':>5}  {'R':>5}  {'F1':>5}"
        )
        for level in (
            "confirmed",
            "confirmed+differential",
            "considered (no rejects)",
            "considered + rejected",
        ):
            for name in ("baseline", "candidate"):
                subset = {k: v for k, v in batches[name].items() if k in shared}
                stats = score(
                    subset, reference, level=level, direct_only=scope_direct
                )
                print(_row(f"{level} [{name}]", stats))

    print()
    print("--- adjudication placements ---")
    for name in ("baseline", "candidate"):
        total: collections.Counter[str] = collections.Counter()
        for record_id in shared:
            total.update(batches[name][record_id]["placements"])
        print(f"  {name:<10} {dict(total)}")

    print()
    print("--- pathway step outcomes ---")
    for name in ("baseline", "candidate"):
        status: collections.Counter[str] = collections.Counter()
        auth: collections.Counter[str] = collections.Counter()
        for record_id in shared:
            for step in batches[name][record_id]["steps"]:
                status[str(step.get("effective_status"))] += 1
                auth[str(step.get("authorization") or "n/a")] += 1
        print(f"  {name:<10} status={dict(status)}")
        print(f"  {'':<10} authorization={dict(auth)}")

    print()
    print("--- confirmed level, per family (only families either run touches) ---")
    print(
        f"{'family':<32} {'TP':>7} {'FP':>7} {'FN':>7}   (baseline -> candidate)"
    )
    for spec in CATEGORIES:
        cells = {}
        for name in ("baseline", "candidate"):
            tp = fp = fn = 0
            for record_id in shared:
                row = batches[name][record_id]
                reference_positive = spec.key in _categories(
                    reference.get(record_id, set()), reference=True
                )
                predicted_positive = spec.key in _categories(
                    set(row["confirmed"]), reference=False
                )
                if reference_positive and predicted_positive:
                    tp += 1
                elif predicted_positive:
                    fp += 1
                elif reference_positive:
                    fn += 1
            cells[name] = (tp, fp, fn)
        if cells["baseline"] == (0, 0, 0) and cells["candidate"] == (0, 0, 0):
            continue
        b, c = cells["baseline"], cells["candidate"]
        print(
            f"{spec.key:<32} {b[0]:>3}->{c[0]:<3} {b[1]:>3}->{c[1]:<3} "
            f"{b[2]:>3}->{c[2]:<3}"
        )

    print()
    print("--- per-record confirmed-diagnosis changes ---")
    for record_id in shared:
        before = batches["baseline"][record_id]["confirmed"]
        after = batches["candidate"][record_id]["confirmed"]
        if before == after:
            continue
        gained = sorted(after - before)
        lost = sorted(before - after)
        truth = sorted(reference.get(record_id, set()))
        print(
            f"  {record_id:<10} +{gained or '-'} -{lost or '-'}  ref={truth}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
