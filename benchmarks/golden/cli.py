"""Freeze and check the golden corpus (front end: ``snapshot_regression.py``)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "benchmarks" / "golden"
PAYLOAD_ROOT = REPO_ROOT / "benchmarks" / "_payloads"
REPORT_ROOT = REPO_ROOT / "benchmarks" / "_reports"

EXIT_PASS, EXIT_DIFF, EXIT_INTEGRITY = 0, 1, 2


def _activate_code_root(code_root: str | None) -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    if code_root:
        sys.path.insert(0, str(Path(code_root).resolve()))


def _environment(code_root: str | None) -> dict[str, Any]:
    import numpy
    import scipy
    import ecgfeat

    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    tree = Path(ecgfeat.__file__).resolve().parents[2]
    try:
        revision = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--", "feature_extraction"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
        "numba": numba_version, "platform": platform.platform(), "ecgfeat_file": ecgfeat.__file__,
        "revision": revision, "engine_tree_dirty": dirty,
    }


def _worker(task: tuple[dict[str, Any], dict[str, Any], str, tuple[str, ...], str | None, str | None]):
    case, spec, data_root, surfaces, payload_dir, code_root = task
    _activate_code_root(code_root)
    from .runner import execute_case

    started = time.perf_counter()
    try:
        result = execute_case(case, spec, Path(data_root), surfaces,
                              Path(payload_dir) if payload_dir else None)
        result["seconds"] = round(time.perf_counter() - started, 3)
        return result
    except Exception as exc:  # recorded, never swallowed silently
        return {"case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"}


def _run_all(cases, configs, data_root: Path, surfaces, payload_dir: Path | None, workers: int,
             code_root: str | None) -> list[dict[str, Any]]:
    tasks = [(case, configs[case["config"]], str(data_root), tuple(surfaces),
              str(payload_dir) if payload_dir else None, code_root) for case in cases]
    if workers <= 1:
        return [_worker(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, tasks, chunksize=1))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_freeze(args: argparse.Namespace) -> int:
    from .manifest import CONFIGS, SELECTION_RULE, build_cases
    from .runner import LEGACY_NORMALIZED_POINTERS, canonical_json_bytes, resolved_config, sha256_bytes

    data_root = Path(args.data_root).resolve()
    manifest_path = GOLDEN_DIR / f"manifest.{args.tier}.json"
    cases, exclusions = build_cases(args.tier, data_root)
    used = sorted({case["config"] for case in cases})
    configs = {name: CONFIGS[name] for name in used}
    resolved = {name: resolved_config(spec) for name, spec in configs.items()}
    manifest = {
        "manifest_version": 1, "tier": args.tier, "selection_rule": SELECTION_RULE,
        "window_seconds": 10.0, "legacy_normalized_pointers": list(LEGACY_NORMALIZED_POINTERS),
        "configs": {name: {"spec": configs[name], "resolved": resolved[name],
                           "resolved_sha256": sha256_bytes(canonical_json_bytes(resolved[name]))}
                    for name in used},
        "exclusions": exclusions,
        "cases": cases,
    }
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text())
        frozen_view = {k: v for k, v in frozen.items() if k != "inputs"}
        if frozen_view != manifest:
            print(f"refusing to overwrite immutable manifest {manifest_path}; selection/config changed",
                  file=sys.stderr)
            return EXIT_INTEGRITY
    baseline_dir = GOLDEN_DIR / "baselines" / args.baseline_id
    expected_path = baseline_dir / f"expected.{args.tier}.json"
    if expected_path.exists() and not args.overwrite_baseline:
        print(f"refusing to overwrite frozen baseline {expected_path}", file=sys.stderr)
        return EXIT_INTEGRITY
    payload_dir = PAYLOAD_ROOT / args.baseline_id / args.tier
    started = time.perf_counter()
    results = _run_all(cases, configs, data_root, ("legacy", "record"), payload_dir, args.workers, args.code_root)
    errors = [r for r in results if "error" in r]
    if errors:
        for item in errors:
            print(f"ERROR {item['case_id']}: {item['error']}", file=sys.stderr)
        return EXIT_INTEGRITY
    manifest["inputs"] = {r["case_id"]: r["input"] for r in results}
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
    elif json.loads(manifest_path.read_text())["inputs"] != manifest["inputs"]:
        print("input identities differ from frozen manifest (dataset integrity failure)", file=sys.stderr)
        return EXIT_INTEGRITY
    _write_json(expected_path, {
        "baseline_id": args.baseline_id, "tier": args.tier,
        "environment": _environment(args.code_root),
        "seconds": round(time.perf_counter() - started, 1),
        "outputs": {r["case_id"]: r["outputs"] for r in results},
        "case_seconds": {r["case_id"]: r["seconds"] for r in results},
    })
    print(f"froze {len(results)} cases ({len(exclusions)} exclusions) -> {expected_path}")
    return EXIT_PASS


def cmd_check(args: argparse.Namespace) -> int:
    from .compare import compare_case

    data_root = Path(args.data_root).resolve()
    manifest = json.loads((GOLDEN_DIR / f"manifest.{args.tier}.json").read_text())
    baseline_dir = GOLDEN_DIR / "baselines" / args.baseline_id
    expected = json.loads((baseline_dir / f"expected.{args.tier}.json").read_text())
    cases = [c for c in manifest["cases"] if not args.filter or any(f in c["case_id"] for f in args.filter)]
    configs = {name: entry["spec"] for name, entry in manifest["configs"].items()}

    # Resolved configuration drift is a failure in its own right.
    from .runner import canonical_json_bytes, resolved_config, sha256_bytes

    config_drift = {}
    for name, entry in manifest["configs"].items():
        now = resolved_config(entry["spec"])
        if sha256_bytes(canonical_json_bytes(now)) != entry["resolved_sha256"]:
            config_drift[name] = {"frozen": entry["resolved"], "current": now}

    label = args.label or time.strftime("check-%Y%m%dT%H%M%S")
    candidate_payloads = PAYLOAD_ROOT / "candidates" / label / args.tier
    surfaces = {"legacy-bytes": ("legacy",), "interpretation": ("interpretation",)}.get(args.mode, ("record",))
    if args.mode == "legacy-bytes" and args.with_record:
        surfaces = ("legacy", "record")
    started = time.perf_counter()
    results = _run_all(cases, configs, data_root, surfaces, candidate_payloads, args.workers, args.code_root)
    baseline_payloads = PAYLOAD_ROOT / args.baseline_id / args.tier
    reports = []
    for case, result in zip(cases, results):
        if "error" in result:
            reports.append({"case_id": case["case_id"], "status": "error", "error": result["error"]})
            continue
        want = {"case_id": case["case_id"], "input": manifest["inputs"][case["case_id"]],
                "outputs": expected["outputs"][case["case_id"]]}
        mode_reports = [compare_case(args.mode, want, result, baseline_payloads=baseline_payloads,
                                     candidate_payloads=candidate_payloads)]
        if args.mode == "legacy-bytes" and args.with_record:
            mode_reports.append(compare_case("record-bytes", want, result, baseline_payloads=baseline_payloads,
                                             candidate_payloads=candidate_payloads))
        merged = mode_reports[0]
        for extra in mode_reports[1:]:
            merged["differences"] += extra["differences"]
            if merged["status"] == "pass" and extra["status"] != "pass":
                merged["status"] = extra["status"]
        reports.append(merged)
    summary: dict[str, int] = {}
    for item in reports:
        summary[item["status"]] = summary.get(item["status"], 0) + 1
    report = {
        "mode": args.mode, "tier": args.tier, "baseline_id": args.baseline_id, "label": label,
        "environment": _environment(args.code_root), "seconds": round(time.perf_counter() - started, 1),
        "cases": len(cases), "summary": summary, "config_drift": config_drift,
        "surfaces": list(surfaces),
        "case_seconds": {r["case_id"]: r["seconds"] for r in results if "seconds" in r},
        "failures": [r for r in reports if r["status"] != "pass"],
    }
    out = Path(args.report) if args.report else REPORT_ROOT / f"{label}.{args.tier}.{args.mode}.json"
    _write_json(out, report)
    print(json.dumps({"summary": summary, "config_drift": sorted(config_drift), "report": str(out)}))
    if config_drift or any(r["status"] in {"integrity_error", "error", "missing_payload"} for r in reports):
        return EXIT_INTEGRITY if any(r["status"] == "integrity_error" for r in reports) else EXIT_DIFF
    return EXIT_DIFF if summary.get("diff") else EXIT_PASS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snapshot_regression.py", description=__doc__)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--code-root", default=None,
                        help="directory to prepend to sys.path so `ecgfeat` resolves there (e.g. a baseline worktree)")
    parser.add_argument("--workers", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="select cases, run the unchanged tree, and freeze expected outputs")
    freeze.add_argument("--tier", choices=("sentinel", "full"), default="sentinel")
    freeze.add_argument("--baseline-id", required=True)
    freeze.add_argument("--overwrite-baseline", action="store_true",
                        help="only for a separately reviewed behavioral baseline replacement")
    freeze.set_defaults(func=cmd_freeze)
    check = sub.add_parser("check", help="rerun the manifest and compare with a frozen baseline")
    check.add_argument("--tier", choices=("sentinel", "full"), default="sentinel")
    check.add_argument("--baseline-id", required=True)
    check.add_argument("--mode", choices=("legacy-bytes", "record-bytes", "record-crosswalk", "canonical-document",
                                          "interpretation"),
                       default="legacy-bytes")
    check.add_argument("--with-record", action="store_true",
                       help="in legacy-bytes mode also require exact record bytes")
    check.add_argument("--filter", action="append", default=[], help="substring filter on case ids")
    check.add_argument("--label", default=None)
    check.add_argument("--report", default=None)
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _activate_code_root(args.code_root)
    return args.func(args)
