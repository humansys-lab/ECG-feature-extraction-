#!/usr/bin/env python3
"""Fixed-cohort ablations; preserve missing outcomes alongside paired errors.

All variants use the same established evaluators and annotation denominators.
The deterministic LUDB development partition is for this experiment only: LUDB
was used by earlier development, so the remainder is not an external holdout.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"


def load_source(source):
    sys.path.insert(0, str(Path(source) / "feature_extraction"))
    sys.path.insert(1, str(ROOT))
    import ecgfeat
    return ecgfeat


def variant_options(name):
    from ecgfeat import RefinementConfig
    if name == "default":
        return {}
    if name == "limited":
        return {"input_mode": "limited"}
    if name in {"limited_qrs", "limited_p"}:
        return {"input_mode": "limited", "refinement": RefinementConfig(
            qrs_adaptive_consensus=name == "limited_qrs", p_pathology_candidates=name == "limited_p",
            atrial_event_validation=name == "limited_p")}
    if name in {"robust_p", "robust_combined", "limited_robust"}:
        options = {"refinement": RefinementConfig(p_pathology_candidates=True, atrial_event_validation=True,
                    qrs_adaptive_consensus=name != "robust_p", t_sequence_offset_only=name != "robust_p")}
        if name == "limited_robust":
            options["input_mode"] = "limited"
        return options
    if name in ("calibrated_pr", "adaptive_pr_tp"):
        return {"st_amplitude_source": name}
    if name == "all":
        return {"refinement": RefinementConfig.experimental()}
    if name == "classical_combined":
        return {"refinement": RefinementConfig(p_phasor_candidates=True,
                    t_boundary_projection=True, t_sequence_selection=True)}
    if name == "classical_offsets":
        return {"refinement": RefinementConfig(t_projection_offset_only=True, t_sequence_offset_only=True)}
    if name == "p_combined":
        return {"refinement": RefinementConfig(p_model_arbitration=True, p_multiple_candidates=True, p_boundary_correction=True)}
    if name == "t_combined":
        return {"refinement": RefinementConfig(t_onset_change_point=True, t_bidirectional=True, t_correlated_fusion=True)}
    return {"refinement": RefinementConfig(**{name: True})}


def evaluate_task(payload):
    dataset, record, variant, source = payload
    load_source(source)
    import ecgfeat.api as api
    options = variant_options(variant)
    start = time.perf_counter()
    if dataset == "ludb":
        import evaluate_ludb as ev
        ev.LUDB_DIR = ROOT / "data/lobachevsky-university-electrocardiography-database-1.0.1/data"
        class Capture:
            result = None
            def extract(self, ecg, fs):
                self.result = api.ECGFeatureExtractor(fs_internal=500, mains_freq=50, **options).extract(ecg, fs)
                return self.result
        capture = Capture()
        failures, coverage = [], []
        rows = ev.evaluate_record(record, capture, failure_details=failures, coverage_rows=coverage)
        result = {"record": record, "failures": failures, "matched_rows": len(rows),
                  "seconds": time.perf_counter() - start}
        if capture.result is not None:
            result.update(global_features=asdict(capture.result.global_features), beats=len(capture.result.beats),
                          record_quality=capture.result.metadata.get("record_quality"),
                          refinement=capture.result.metadata.get("refinement"))
        return variant, result, rows, coverage
    if dataset == "qtdb":
        import evaluate_qtdb as ev
        from unittest.mock import patch
        config = ev.QTDBConfig(str(ROOT / "data/qtdb"), "q1c", ("ecgfeat",), 75., 150.,
                              2., "global", "native", "neurokit", "dwt", 50., True, 3.)
        original = api.ECGFeatureExtractor
        with patch.object(api, "ECGFeatureExtractor", lambda **kwargs: original(**kwargs, **options)):
            result = ev.evaluate_record(record, config)
        return variant, {"record": record, "seconds": time.perf_counter() - start,
                         "failures": result.failure_rows}, result.detection_rows, result.match_rows
    raise ValueError(dataset)


def write_csv(path, rows):
    import csv
    with path.open("w", newline="") as handle:
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ludb", "qtdb"), default="ludb")
    parser.add_argument("--variants", nargs="+", default=["default", "p_combined", "t_combined", "all"])
    parser.add_argument("--partition", choices=("all", "development", "remainder"), default="all")
    parser.add_argument("--records", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    load_source(args.source_root)
    for variant in args.variants:
        variant_options(variant)
    if args.dataset == "ludb":
        records = [str(i) for i in range(1, 201)]
    else:
        import evaluate_qtdb as qt
        records = qt.discover_records(ROOT / "data/qtdb", "q1c")
    if args.records:
        unknown = set(args.records) - set(records)
        if unknown:
            parser.error(f"unknown records: {sorted(unknown)}")
        records = args.records
    if args.partition != "all":
        def development(record):
            return int(hashlib.sha256(("ecgfeat-20260922:" + record).encode()).hexdigest()[:8], 16) % 4 == 0
        records = [r for r in records if development(r) == (args.partition == "development")]
    if not records:
        parser.error("empty record cohort")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "manifest.json").exists():
        parser.error("output already contains an experiment; use a new directory")
    hashes = {str(p.relative_to(args.source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.source_root / "feature_extraction/ecgfeat").rglob("*.py")}
    manifest = {"dataset": args.dataset, "records": records, "variants": args.variants,
                "partition": args.partition, "source_sha256": hashes, "workers": args.workers,
                "caveat": "LUDB remainder is not an independent external validation set"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    all_records = {v: [] for v in args.variants}
    all_rows = {v: [] for v in args.variants}
    all_coverage = {v: [] for v in args.variants}
    tasks = [(args.dataset, r, v, str(args.source_root)) for r in records for v in args.variants]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (variant, record, rows, coverage) in enumerate(pool.map(evaluate_task, tasks), 1):
            all_records[variant].append(record)
            all_rows[variant].extend(rows)
            all_coverage[variant].extend(coverage)
            directory = args.out / variant
            directory.mkdir(exist_ok=True)
            with (directory / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"{n}/{len(tasks)} {variant} record={record['record']} failures={len(record['failures'])}", flush=True)
    for variant in args.variants:
        directory = args.out / variant
        if args.dataset == "ludb":
            import evaluate_ludb as ev
            write_csv(directory / "ludb_errors.csv", all_rows[variant])
            write_csv(directory / "coverage.csv", all_coverage[variant])
            summary = ev.compute_summary(all_rows[variant], coverage_rows=all_coverage[variant])
        else:
            write_csv(directory / "detection_by_record.csv", all_rows[variant])
            write_csv(directory / "matched_events.csv", all_coverage[variant])
            from compare_ludb_detectors import aggregate_metrics
            summary = {"metrics": aggregate_metrics(all_rows[variant], all_coverage[variant], ("method", "wave"))}
        summary.update(records=len(all_records[variant]), failed_records=[r["record"] for r in all_records[variant] if r["failures"]])
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "execution.json").write_text(json.dumps({"wall_seconds": time.perf_counter()-started}))
    return int(any(r["failures"] for records in all_records.values() for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
