#!/usr/bin/env python3
"""Build a reproducible, disease-enriched PTB-XL Agent evaluation cohort.

The output directory is a flat WFDB dataset made from symlinks.  Reference
labels are used only for cohort construction and the selection manifest; the
batch runner continues to keep them out of the feature payload and model
context.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _excluded_records(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("records") or []:
            record = row.get("record") if isinstance(row, dict) else None
            if record:
                excluded.add(str(record))
    return excluded


def _link(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(
                f"existing symlink has a different target: {destination}"
            )
        return
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing file: {destination}")
    destination.symlink_to(source)


def _select(
    rows: list[dict[str, Any]],
    *,
    disease_count: int,
    normal_count: int,
    minimum_per_category: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    disease = [row for row in rows if row["categories"]]
    normal = [
        row
        for row in rows
        if not row["categories"]
        and set(row["active_codes"]) <= {"NORM", "SR"}
    ]
    if len(disease) < disease_count:
        raise ValueError(
            f"requested {disease_count} mapped-disease records, only {len(disease)} available"
        )
    if len(normal) < normal_count:
        raise ValueError(
            f"requested {normal_count} normal controls, only {len(normal)} available"
        )

    availability = Counter(
        category for row in disease for category in row["categories"]
    )
    targets = {
        category: min(minimum_per_category, count)
        for category, count in availability.items()
    }
    rng = random.Random(seed)
    tie_break = {row["record"]: rng.random() for row in rows}
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    counts: Counter[str] = Counter()

    # First cover rare diagnosis families.  The small complexity penalty keeps
    # the cohort from degenerating into only extreme multi-label cases.
    while len(chosen) < disease_count:
        best: dict[str, Any] | None = None
        best_key: tuple[float, float, float] | None = None
        for row in disease:
            if row["record"] in chosen_ids:
                continue
            gain = sum(
                (targets[category] - counts[category])
                / targets[category]
                / math.sqrt(availability[category])
                for category in row["categories"]
                if counts[category] < targets[category]
            )
            complexity_penalty = 0.02 * max(0, len(row["categories"]) - 3)
            key = (
                gain,
                -complexity_penalty,
                tie_break[row["record"]],
            )
            if best_key is None or key > best_key:
                best = row
                best_key = key
        if best is None or best_key is None or best_key[0] <= 0:
            break
        chosen.append(best)
        chosen_ids.add(best["record"])
        counts.update(best["categories"])

    # Fill the remainder with weighted random sampling.  Rare categories get a
    # modest boost, divided by label cardinality so highly multi-label records
    # do not dominate merely because they match more families.
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in disease:
        if row["record"] in chosen_ids:
            continue
        weight = sum(
            1.0 / math.sqrt(availability[category])
            for category in row["categories"]
        ) / math.sqrt(len(row["categories"]))
        priority = -math.log(max(tie_break[row["record"]], 1e-12)) / weight
        ranked.append((priority, row))
    for _, row in sorted(ranked, key=lambda item: item[0]):
        if len(chosen) >= disease_count:
            break
        chosen.append(row)
        chosen_ids.add(row["record"])
        counts.update(row["categories"])

    controls = sorted(normal, key=lambda row: tie_break[row["record"]])[
        :normal_count
    ]
    return sorted(chosen + controls, key=lambda row: row["record"]), counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "ptb-xl"
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ptb-xl-metadata",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--normal-controls", type=int, default=10)
    parser.add_argument("--minimum-per-category", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help="record_manifest.json whose records must not be sampled (repeatable)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.size < 1:
        raise SystemExit("--size must be positive")
    if not 0 <= args.normal_controls < args.size:
        raise SystemExit("--normal-controls must be in [0, size)")

    # Imported here so --help remains usable even outside the repository venv.
    from evaluate_target_ecgfeat_diagnosis import CATEGORIES

    raw_dir = args.raw_dir.resolve()
    metadata_dir = args.metadata_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {path.stem: path for path in raw_dir.rglob("*.hea")}
    statements = {
        row.get("", ""): row
        for row in _read_csv(metadata_dir / "scp_statements.csv")
        if row.get("")
    }
    category_refs = {
        spec.key: set(spec.ptbxl_refs) for spec in CATEGORIES if spec.ptbxl_refs
    }
    excluded = _excluded_records(args.exclude_manifest)

    rows: list[dict[str, Any]] = []
    for metadata in _read_csv(metadata_dir / "ptbxl_database.csv"):
        record = Path(metadata.get("filename_hr") or "").name
        header = headers.get(record)
        if header is None or record in excluded:
            continue
        try:
            score_map = ast.literal_eval(metadata.get("scp_codes") or "{}")
        except (SyntaxError, ValueError):
            score_map = {}
        active_codes: set[str] = set()
        for code, score in score_map.items():
            statement = statements.get(str(code), {})
            if statement.get("diagnostic") == "1.0" and float(score) < 50.0:
                continue
            active_codes.add(str(code))
        categories = {
            category
            for category, references in category_refs.items()
            if active_codes & references
        }
        rows.append(
            {
                "record": record,
                "header": header,
                "data": header.with_suffix(".dat"),
                "active_codes": sorted(active_codes),
                "categories": sorted(categories),
                "strat_fold": int(metadata["strat_fold"]),
            }
        )

    selected, category_counts = _select(
        rows,
        disease_count=args.size - args.normal_controls,
        normal_count=args.normal_controls,
        minimum_per_category=max(1, args.minimum_per_category),
        seed=args.seed,
    )
    for row in selected:
        if not row["data"].exists():
            raise FileNotFoundError(row["data"])
        _link(row["header"], output_dir / row["header"].name)
        _link(row["data"], output_dir / row["data"].name)

    cardinality = Counter(len(row["categories"]) for row in selected)
    fold_counts = Counter(row["strat_fold"] for row in selected)
    manifest = {
        "protocol": "ptbxl-agent-disease-enriched.v1",
        "seed": args.seed,
        "raw_dir": str(raw_dir),
        "metadata_dir": str(metadata_dir),
        "size": len(selected),
        "mapped_disease_records": sum(bool(row["categories"]) for row in selected),
        "normal_control_records": sum(not row["categories"] for row in selected),
        "minimum_per_category_requested": args.minimum_per_category,
        "excluded_records": len(excluded),
        "category_counts": dict(sorted(category_counts.items())),
        "category_cardinality": {
            str(key): value for key, value in sorted(cardinality.items())
        },
        "strat_fold_counts": {
            str(key): value for key, value in sorted(fold_counts.items())
        },
        "records": [
            {
                "record": row["record"],
                "categories": row["categories"],
                "active_codes": row["active_codes"],
                "strat_fold": row["strat_fold"],
            }
            for row in selected
        ],
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
