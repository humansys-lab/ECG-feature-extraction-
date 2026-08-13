#!/usr/bin/env python3
"""Classify *why* ecgfeat misses a reference wave on QTDB.

The benchmark reports that ecgfeat misses ~8.5% of P waves and ~4% of T waves,
but a sensitivity number cannot distinguish the two failures that matter,
because they have opposite fixes:

* **abstention** -- ecgfeat produced no wave anywhere near the reference. The
  detector declined to mark it. Fixing this means loosening a gate.
* **mislocalisation** -- ecgfeat *did* emit a wave nearby, but placed it far
  enough away that it could not match. Fixing this means correcting a search
  window or a peak rule, not loosening a gate.

The scoring CSVs cannot answer this: they store matched pairs and counts, so
an unmatched detection is invisible. This script therefore re-runs ecgfeat and,
for every unmatched reference event, measures the distance to the nearest
detection of the same wave.

For P waves it also records the reference and detected P-to-QRS intervals,
which is what exposes the "locked onto a pre-QRS deflection" mode found in
sel42 (reference PR 220 ms, detected 70 ms).

Usage:
    python analyze_qtdb_failures.py --workers 12 --out qtdb_failure_analysis
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_ludb_detectors import (  # noqa: E402
    WaveEvent,
    detect_ecgfeat,
    match_events,
    write_csv,
)
from evaluate_qtdb import (  # noqa: E402
    SCORED_SLOT,
    WAVES,
    _shift_events,
    discover_records,
    load_annotated_window,
    parse_qtdb_waves,
    restrict_to_annotated_clusters,
)

# A reference event with a detection this close is judged "mislocalised": the
# detector clearly responded to the same wave but placed it out of tolerance.
# Beyond this there is no plausible corresponding detection, so it is an
# abstention.  400 ms is wider than any P or T wave and still shorter than a
# typical RR interval, so it cannot silently borrow the neighbouring beat's wave.
NEARBY_MS = 400.0


def _nearest(peaks: Sequence[int], target: int) -> tuple[int | None, float | None]:
    if not peaks:
        return None, None
    array = np.asarray(peaks, dtype=float)
    index = int(np.argmin(np.abs(array - target)))
    return int(array[index]), float(array[index] - target)


def analyse_record(payload: tuple[str, str, float, float, float]) -> list[dict[str, Any]]:
    record_id, qtdb_root, wave_tol_ms, r_tol_ms, cluster_gap_sec = payload
    record_path = Path(qtdb_root) / record_id
    try:
        gt_absolute = parse_qtdb_waves(record_path, "q1c")
        ecg, fs, window_start, signal_names, _missing = load_annotated_window(
            record_path, gt_absolute, window_pad_sec=5.0
        )
    except Exception as exc:  # noqa: BLE001
        return [{"record": record_id, "wave": "-", "status": f"read_failed: {exc}"}]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detected, _elapsed = detect_ecgfeat(
                ecg, fs, (SCORED_SLOT,), 60.0
            )
    except Exception as exc:  # noqa: BLE001
        return [{"record": record_id, "wave": "-", "status": f"detect_failed: {exc}"}]

    gt_waves = {
        wave: _shift_events(events, window_start)
        for wave, events in gt_absolute.items()
    }
    per_wave = detected[SCORED_SLOT]
    gap_samples = int(round(cluster_gap_sec * fs))

    # QRS anchors let us express P failures as a PR interval, which is the
    # clinically meaningful way to see a P placed on the wrong deflection.
    gt_qrs_peaks = sorted(
        int(event.peak) for event in gt_waves["QRS"] if event.peak is not None
    )
    det_qrs_peaks = sorted(
        int(event.peak) for event in per_wave.get("QRS", []) if event.peak is not None
    )

    rows: list[dict[str, Any]] = []
    for wave in WAVES:
        tolerance_ms = r_tol_ms if wave == "QRS" else wave_tol_ms
        tolerance_samples = max(1, int(round(tolerance_ms * fs / 1000.0)))
        gt_events = gt_waves.get(wave, [])
        det_events, _dropped, _n = restrict_to_annotated_clusters(
            gt_events, per_wave.get(wave, []), tolerance_samples, gap_samples
        )
        pairs, _fn, _fp = match_events(gt_events, det_events, tolerance_samples)
        matched_gt = {id(gt_event) for gt_event, _det in pairs}
        det_peaks = [
            int(event.peak) for event in det_events if event.peak is not None
        ]

        for event in gt_events:
            if event.peak is None or id(event) in matched_gt:
                continue
            nearest_peak, delta = _nearest(det_peaks, int(event.peak))
            delta_ms = None if delta is None else delta * 1000.0 / fs
            if delta_ms is None or abs(delta_ms) > NEARBY_MS:
                mode = "abstention"
            else:
                mode = "mislocalised"

            row: dict[str, Any] = {
                "record": record_id,
                "source_channel": signal_names[0] if signal_names else "?",
                "wave": wave,
                "gt_peak_sample": int(event.peak),
                "nearest_det_sample": nearest_peak,
                "nearest_delta_ms": delta_ms,
                "mode": mode,
                "status": "ok",
            }
            if wave == "P":
                gt_next_qrs, _ = _nearest(
                    [peak for peak in gt_qrs_peaks if peak > int(event.peak)],
                    int(event.peak),
                )
                row["gt_pr_ms"] = (
                    None
                    if gt_next_qrs is None
                    else (gt_next_qrs - int(event.peak)) * 1000.0 / fs
                )
                if nearest_peak is not None:
                    det_next_qrs, _ = _nearest(
                        [peak for peak in det_qrs_peaks if peak > nearest_peak],
                        nearest_peak,
                    )
                    row["det_pr_ms"] = (
                        None
                        if det_next_qrs is None
                        else (det_next_qrs - nearest_peak) * 1000.0 / fs
                    )
            rows.append(row)
    return rows


def summarise(rows: Sequence[Mapping[str, Any]]) -> None:
    usable = [row for row in rows if row.get("status") == "ok"]
    print()
    print("=" * 92)
    print("  ecgfeat 漏检成因分类（QTDB，未匹配的参考波）")
    print("=" * 92)
    print()
    by_wave: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        by_wave[str(row["wave"])].append(row)

    print(f"  {'波':<5s} {'漏检总数':>8s} {'错位':>8s} {'错位%':>7s} "
          f"{'未输出':>8s} {'未输出%':>8s} {'错位中位距离':>14s}")
    print("  " + "-" * 88)
    for wave in WAVES:
        items = by_wave.get(wave, [])
        if not items:
            continue
        mis = [row for row in items if row["mode"] == "mislocalised"]
        abst = [row for row in items if row["mode"] == "abstention"]
        deltas = [
            abs(float(row["nearest_delta_ms"]))
            for row in mis
            if row.get("nearest_delta_ms") is not None
        ]
        print(
            f"  {wave:<5s} {len(items):>8d} {len(mis):>8d} "
            f"{len(mis) / len(items) * 100:>6.1f}% {len(abst):>8d} "
            f"{len(abst) / len(items) * 100:>7.1f}% "
            f"{(np.median(deltas) if deltas else float('nan')):>13.0f}ms"
        )

    # P-wave PR diagnosis: a detected P sitting far too close to the QRS is the
    # signature of locking onto a pre-QRS deflection rather than the true P.
    p_mis = [
        row
        for row in by_wave.get("P", [])
        if row["mode"] == "mislocalised"
        and row.get("gt_pr_ms") is not None
        and row.get("det_pr_ms") is not None
    ]
    if p_mis:
        gt_pr = np.array([float(row["gt_pr_ms"]) for row in p_mis])
        det_pr = np.array([float(row["det_pr_ms"]) for row in p_mis])
        print()
        print(f"  P 波错位案例的 PR 间期对照（n={len(p_mis)}）：")
        print(f"    参考 PR   中位 {np.median(gt_pr):6.0f} ms")
        print(f"    检测 PR   中位 {np.median(det_pr):6.0f} ms")
        print(f"    检测 PR < 120 ms（落进 QRS 前沿）占比 "
              f"{(det_pr < 120).mean() * 100:.1f}%")

    print()
    print("  漏检最集中的 15 条记录×波：")
    print("  " + "-" * 88)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    modes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in usable:
        key = (str(row["record"]), str(row["source_channel"]), str(row["wave"]))
        counts[key] += 1
        modes[key].append(str(row["mode"]))
    for key, count in sorted(counts.items(), key=lambda item: -item[1])[:15]:
        mode_list = modes[key]
        mis = sum(1 for value in mode_list if value == "mislocalised")
        print(
            f"    {key[0]:<10s} {key[1]:<7s} {key[2]:<4s} 漏检 {count:>3d}  "
            f"(错位 {mis:>3d} / 未输出 {count - mis:>3d})"
        )
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-root", type=Path, default=PROJECT_ROOT / "data" / "qtdb")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "qtdb_failure_analysis")
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--wave-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--r-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--annotation-cluster-gap-sec", type=float, default=3.0)
    args = parser.parse_args(argv)

    qtdb_root = args.qtdb_root.resolve()
    records = args.records or discover_records(qtdb_root, "q1c")
    if args.n is not None:
        records = records[: args.n]
    print(f"分析 {len(records)} 条记录，{args.workers} 进程")

    payloads = [
        (
            record,
            str(qtdb_root),
            float(args.wave_tolerance_ms),
            float(args.r_tolerance_ms),
            float(args.annotation_cluster_gap_sec),
        )
        for record in records
    ]
    rows: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            for index, result in enumerate(pool.map(analyse_record, payloads), start=1):
                rows.extend(result)
                if index % 10 == 0:
                    print(f"  [{index}/{len(records)}]", flush=True)
    else:
        for payload in payloads:
            rows.extend(analyse_record(payload))

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "missed_events.csv", rows)
    summarise(rows)
    print(f"  明细：{out_dir / 'missed_events.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
