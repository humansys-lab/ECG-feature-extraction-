#!/usr/bin/env python3
"""Repeatable wall-clock benchmark for the ecgfeat extraction pipeline."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "feature_extraction"


def _configure_threads(count: int) -> None:
    value = str(max(1, int(count)))
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = value


def _record_path(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.suffix in {".hea", ".mat", ".dat"}:
        path = path.with_suffix("")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "010" / "JS00001",
        help="WFDB record path, with or without .hea/.mat suffix.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--json-profile",
        choices=("summary", "audit", "debug"),
        default="summary",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.blas_threads < 1:
        parser.error("warmup must be >= 0; repeats and blas-threads must be >= 1")

    # Configure numerical libraries before importing NumPy/SciPy.
    _configure_threads(args.blas_threads)
    sys.path.insert(0, str(FEATURE_ROOT))

    import wfdb

    from ecgfeat.compat.api_v0 import ECGFeatureExtractor
    from ecgfeat.compat.export_v0 import prepare_json_export, to_dict

    record = _record_path(args.record)
    signal, fields = wfdb.rdsamp(str(record))
    ecg = signal.T
    fs = int(round(float(fields["fs"])))
    if ecg.shape[0] != 12:
        raise ValueError(f"expected 12 leads, got shape {tuple(ecg.shape)}")

    extractor = ECGFeatureExtractor(mains_freq=50)
    result = None
    for _ in range(args.warmup):
        result = extractor.extract(ecg, fs=fs)

    elapsed = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = extractor.extract(ecg, fs=fs)
        elapsed.append(time.perf_counter() - started)

    assert result is not None
    full_payload = to_dict(result)
    json_payload = prepare_json_export(full_payload, profile=args.json_profile)
    serialized_bytes = len(
        json.dumps(json_payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    compressed_bytes = len(
        gzip.compress(
            json.dumps(
                json_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    )
    report = {
        "record": str(record),
        "shape": list(ecg.shape),
        "fs": fs,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "blas_threads": args.blas_threads,
        "seconds": elapsed,
        "mean_seconds": statistics.fmean(elapsed),
        "median_seconds": statistics.median(elapsed),
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
        "json_profile": args.json_profile,
        "json_bytes": serialized_bytes,
        "json_gzip_bytes": compressed_bytes,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
