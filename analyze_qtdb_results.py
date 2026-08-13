#!/usr/bin/env python3
"""Stratified analysis of an ``evaluate_qtdb.py`` output directory.

The pooled QTDB summary is dominated by a small number of pathological
records, so reading it alone is misleading.  This script re-aggregates the
per-record CSVs into the views that actually support a conclusion:

* pooled over every record (what the raw benchmark reports),
* with the two paced records excluded, and paced-only for contrast,
* a per-record distribution, so one bad record cannot carry the headline,
* side by side with this project's existing LUDB numbers.

It reads only the CSVs written by ``evaluate_qtdb.py`` and never re-runs the
detectors, so it is cheap to iterate on.

Usage:
    python analyze_qtdb_results.py --qtdb-dir qtdb_evaluation
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_ludb_detectors import aggregate_metrics  # noqa: E402
from evaluate_qtdb import CSE_2SD_TOLERANCE_MS, WAVES  # noqa: E402

# QTDB inherits MIT-BIH's paced records.  Both are flagged as such in their
# own .hea comments, which is how this list is verified rather than assumed.
PACED_RECORDS = ("sel102", "sel104")

_INT_FIELDS = ("gt_count", "det_count", "tp", "fp", "fn")
_FLOAT_FIELDS = (
    "sensitivity",
    "precision",
    "f1",
    "onset_error_ms",
    "peak_error_ms",
    "offset_error_ms",
)


def _coerce(row: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for field in _INT_FIELDS:
        if field in out:
            out[field] = int(out[field]) if str(out[field]).strip() else 0
    for field in _FLOAT_FIELDS:
        if field in out:
            value = str(out[field]).strip()
            out[field] = float(value) if value else None
    return out


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [_coerce(row) for row in csv.DictReader(handle)]


def verify_paced_records(qtdb_root: Path) -> list[str]:
    """Confirm the paced list from the headers instead of trusting a constant."""

    found = []
    for header in sorted(qtdb_root.glob("*.hea")):
        text = header.read_text(encoding="utf-8", errors="ignore").lower()
        if "paced" in text or "pacemaker" in text or "pacing" in text:
            found.append(header.stem)
    return found


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "bias": None, "sd": None, "mae": None, "p50": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "bias": float(np.mean(array)),
        "sd": float(np.std(array)),
        "mae": float(np.mean(np.abs(array))),
        "p50": float(np.median(np.abs(array))),
    }


def stratum(
    detection_rows: Sequence[Mapping[str, Any]],
    match_rows: Sequence[Mapping[str, Any]],
    keep: Iterable[str] | None = None,
    drop: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    keep_set = set(keep) if keep is not None else None
    drop_set = set(drop) if drop is not None else set()

    def included(row: Mapping[str, Any]) -> bool:
        record = str(row["record"])
        if record in drop_set:
            return False
        return keep_set is None or record in keep_set

    return aggregate_metrics(
        [row for row in detection_rows if included(row)],
        [row for row in match_rows if included(row)],
        ("method", "wave"),
    )


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    print()
    print(title)
    print("-" * 108)
    print(
        f"{'method':<10s} {'wave':<4s} {'recs':>5s} {'GT':>6s} {'TP':>6s} "
        f"{'FP':>5s} {'FN':>5s} {'Se%':>6s} {'PPV%':>6s} "
        f"{'onset bias±SD':>17s} {'peak bias±SD':>17s} {'offset bias±SD':>17s}"
    )
    for row in sorted(rows, key=lambda item: (str(item["method"]), str(item["wave"]))):
        se = row.get("sensitivity")
        ppv = row.get("precision")
        print(
            f"{str(row['method']):<10s} {str(row['wave']):<4s} "
            f"{int(row['records']):>5d} {int(row['gt_count']):>6d} "
            f"{int(row['tp']):>6d} {int(row['fp']):>5d} {int(row['fn']):>5d} "
            f"{_fmt(None if se is None else se * 100):>6s} "
            f"{_fmt(None if ppv is None else ppv * 100):>6s} "
            f"{_fmt(row.get('onset_bias_ms')) + '±' + _fmt(row.get('onset_sd_ms')):>17s} "
            f"{_fmt(row.get('peak_bias_ms')) + '±' + _fmt(row.get('peak_sd_ms')):>17s} "
            f"{_fmt(row.get('offset_bias_ms')) + '±' + _fmt(row.get('offset_sd_ms')):>17s}"
        )


def per_record_distribution(
    detection_rows: Sequence[Mapping[str, Any]],
    match_rows: Sequence[Mapping[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    """Median-of-records view: each record contributes once, outliers cannot dominate."""

    by_record: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in match_rows:
        if row["method"] != method:
            continue
        value = row.get("offset_error_ms")
        if value is not None:
            by_record[(str(row["wave"]), str(row["record"]))].append(abs(value))

    sens: dict[tuple[str, str], tuple[int, int]] = {}
    for row in detection_rows:
        if row["method"] != method:
            continue
        key = (str(row["wave"]), str(row["record"]))
        tp, gt = sens.get(key, (0, 0))
        sens[key] = (tp + int(row["tp"]), gt + int(row["gt_count"]))

    out = []
    for wave in WAVES:
        record_maes = [
            float(np.mean(values))
            for (item_wave, _record), values in by_record.items()
            if item_wave == wave and values
        ]
        record_se = [
            tp / gt
            for (item_wave, _record), (tp, gt) in sens.items()
            if item_wave == wave and gt
        ]
        out.append(
            {
                "wave": wave,
                "records": len(record_se),
                "median_record_offset_mae_ms": (
                    float(np.median(record_maes)) if record_maes else None
                ),
                "p90_record_offset_mae_ms": (
                    float(np.percentile(record_maes, 90)) if record_maes else None
                ),
                "median_record_sensitivity": (
                    float(np.median(record_se)) if record_se else None
                ),
                "records_below_90pct_sensitivity": sum(
                    1 for value in record_se if value < 0.90
                ),
            }
        )
    return out


def cse_table(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    by_wave = {
        str(row["wave"]): row for row in rows if str(row["method"]) == method
    }
    out = []
    for (wave, boundary), limit in CSE_2SD_TOLERANCE_MS.items():
        row = by_wave.get(wave)
        if row is None:
            continue
        sd = row.get(f"{boundary}_sd_ms")
        out.append(
            {
                "boundary": f"{wave} {boundary}",
                "n": row.get(f"{boundary}_n"),
                "bias_ms": row.get(f"{boundary}_bias_ms"),
                "sd_ms": sd,
                "limit_ms": limit,
                "within": None if sd is None else float(sd) <= limit,
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=PROJECT_ROOT / "qtdb_evaluation")
    parser.add_argument("--qtdb-root", type=Path, default=PROJECT_ROOT / "data" / "qtdb")
    parser.add_argument(
        "--ludb-summary",
        type=Path,
        default=PROJECT_ROOT / "ludb_ecgfeat_vs_neurokit_span" / "summary.csv",
    )
    args = parser.parse_args(argv)

    detection_rows = read_rows(args.qtdb_dir / "detection_by_record.csv")
    match_rows = read_rows(args.qtdb_dir / "matched_events.csv")
    if not detection_rows:
        raise SystemExit(f"no detection rows under {args.qtdb_dir}")

    paced = verify_paced_records(args.qtdb_root)
    all_records = sorted({str(row["record"]) for row in detection_rows})
    print(f"records evaluated: {len(all_records)}")
    print(f"paced records found by header scan: {paced}")
    if set(paced) != set(PACED_RECORDS):
        print(f"  note: constant PACED_RECORDS={list(PACED_RECORDS)} differs from scan")

    pooled = stratum(detection_rows, match_rows)
    print_table("[1] Pooled over all records", pooled)

    excluded = stratum(detection_rows, match_rows, drop=paced)
    print_table(f"[2] Excluding paced records {paced}", excluded)

    paced_only = stratum(detection_rows, match_rows, keep=paced)
    print_table(f"[3] Paced records only {paced}", paced_only)

    for method in ("ecgfeat", "neurokit2"):
        print()
        print(f"[4] Per-record distribution — {method} (each record counts once)")
        print("-" * 108)
        print(
            f"{'wave':<5s} {'recs':>5s} {'median rec offset MAE':>23s} "
            f"{'p90':>8s} {'median rec Se':>15s} {'recs Se<90%':>12s}"
        )
        for row in per_record_distribution(detection_rows, match_rows, method):
            median_se = row["median_record_sensitivity"]
            print(
                f"{row['wave']:<5s} {row['records']:>5d} "
                f"{_fmt(row['median_record_offset_mae_ms']):>23s} "
                f"{_fmt(row['p90_record_offset_mae_ms']):>8s} "
                f"{_fmt(None if median_se is None else median_se * 100):>15s} "
                f"{row['records_below_90pct_sensitivity']:>12d}"
            )

    for label, rows in (("all records", pooled), (f"excluding {paced}", excluded)):
        print()
        print(f"[5] CSE 2-sigma conformance — ecgfeat, {label}")
        print("-" * 108)
        print(
            f"{'boundary':<14s} {'n':>6s} {'bias':>8s} {'SD':>8s} "
            f"{'limit':>8s}  within"
        )
        for row in cse_table(rows, "ecgfeat"):
            print(
                f"{row['boundary']:<14s} {_fmt(row['n'], 0):>6s} "
                f"{_fmt(row['bias_ms']):>8s} {_fmt(row['sd_ms']):>8s} "
                f"{_fmt(row['limit_ms']):>8s}  "
                f"{'yes' if row['within'] else 'no'}"
            )

    if args.ludb_summary.exists():
        ludb = read_rows(args.ludb_summary)
        qtdb_by_key = {
            (str(row["method"]), str(row["wave"])): row for row in excluded
        }
        print()
        print(f"[6] LUDB (12-lead, 500 Hz) vs QTDB (2-channel, 250 Hz, excluding paced)")
        print("-" * 108)
        print(
            f"{'method':<10s} {'wave':<4s} {'LUDB Se%':>9s} {'QTDB Se%':>9s} "
            f"{'LUDB PPV%':>10s} {'QTDB PPV%':>10s} "
            f"{'LUDB off SD':>12s} {'QTDB off SD':>12s}"
        )
        for row in sorted(ludb, key=lambda item: (str(item["method"]), str(item["wave"]))):
            key = (str(row["method"]), str(row["wave"]))
            other = qtdb_by_key.get(key)
            if other is None:
                continue

            def number(source: Mapping[str, Any], field: str) -> float | None:
                value = source.get(field)
                if value is None or str(value).strip() == "":
                    return None
                return float(value)

            print(
                f"{key[0]:<10s} {key[1]:<4s} "
                f"{_fmt((number(row, 'sensitivity') or 0) * 100):>9s} "
                f"{_fmt((number(other, 'sensitivity') or 0) * 100):>9s} "
                f"{_fmt((number(row, 'precision') or 0) * 100):>10s} "
                f"{_fmt((number(other, 'precision') or 0) * 100):>10s} "
                f"{_fmt(number(row, 'offset_sd_ms')):>12s} "
                f"{_fmt(number(other, 'offset_sd_ms')):>12s}"
            )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
