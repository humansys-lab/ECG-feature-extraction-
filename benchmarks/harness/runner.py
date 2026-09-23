"""Execute pinned tool invocations and freeze normalized results."""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import corpora as _corpora
from . import normalizers as _normalizers
from .core import (
    BASELINE_INDEX_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    FileHashCache,
    HarnessError,
    PathNormalizer,
    assert_tree_import,
    canonical_json,
    capture_environment,
    dataset_manifest,
    dump_result,
    finalize_metrics,
    sha256_bytes,
)

HARNESS_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = HARNESS_DIR.parent
WORKSPACE_ROOT = BENCHMARKS_DIR.parent
DEFAULT_SPEC = HARNESS_DIR / "spec.json"
DEFAULT_RAW_ROOT = BENCHMARKS_DIR / "_raw"
DEFAULT_CORPUS_ROOT = DEFAULT_RAW_ROOT / "corpora"
DEFAULT_PYTHON = (WORKSPACE_ROOT / ".venv" / "bin" / "python"
                  if (WORKSPACE_ROOT / ".venv" / "bin" / "python").exists() else Path(sys.executable))
IMPORT_LOG_DIR = HARNESS_DIR / "_importlog"


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [tool["id"] for tool in spec["tools"]]
    if len(ids) != len(set(ids)):
        raise HarnessError("duplicate tool ids in spec")
    return spec


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Context:
    repo_root: Path
    python: Path
    spec: dict[str, Any]
    raw_dir: Path
    corpus_root: Path
    hash_cache: FileHashCache
    corpus_builds: dict[str, Any] = field(default_factory=dict)

    @property
    def data_root(self) -> Path:
        return self.repo_root / "data"

    def corpus_dir(self, corpus_id: str) -> Path:
        return self.corpus_root / corpus_id

    def expand(self, text: str, run_dir: Path | None = None) -> str:
        text = text.replace("{repo}", str(self.repo_root)).replace("{data}", str(self.data_root))
        if run_dir is not None:
            text = text.replace("{run_dir}", str(run_dir))
        for match in re.findall(r"\{corpus:([a-z0-9_]+)\}", text):
            text = text.replace("{corpus:%s}" % match, str(self.corpus_dir(match)))
        if "{" in text and re.search(r"\{(repo|data|run_dir|corpus:[^}]*)\}", text):
            raise HarnessError(f"unexpanded placeholder in {text!r}")
        return text

    def tool_env(self, import_log: Path | None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env.update(self.spec.get("thread_env", {}))
        paths = [str(self.repo_root / "feature_extraction"), str(self.repo_root)]
        if import_log is not None:
            paths.insert(0, str(IMPORT_LOG_DIR))
            env["ECGFEAT_HARNESS_IMPORT_LOG"] = str(import_log)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["NUMBA_CACHE_DIR"] = str(self.raw_dir / "_numba_cache")
        env["MPLCONFIGDIR"] = str(self.raw_dir / "_mplconfig")
        return env

    def path_normalizer(self, run_dir: Path) -> PathNormalizer:
        data_real = os.path.realpath(self.data_root)
        return PathNormalizer([
            (str(run_dir), "<RUN>"),
            (os.path.realpath(run_dir), "<RUN>"),
            (str(self.corpus_root), "<CORPUS>"),
            (os.path.realpath(self.corpus_root), "<CORPUS>"),
            (data_real, "<DATA>"),
            (str(self.data_root), "<DATA>"),
            (str(self.repo_root), "<REPO>"),
            (os.path.realpath(self.repo_root), "<REPO>"),
        ])


# ── Datasets ────────────────────────────────────────────────────────────────


def _sort_records(records: list[str], mode: str) -> list[str]:
    if mode == "numeric":
        return sorted(records, key=lambda r: (0, int(r)) if r.isdigit() else (1, r))
    return sorted(records)


def resolve_records(root: Path, rule: Mapping[str, Any]) -> list[str]:
    if "list" in rule:
        return [str(item) for item in rule["list"]]
    if "from_file" in rule:
        lines = [line.strip() for line in (root / rule["from_file"]).read_text(encoding="utf-8").splitlines()]
        records = [Path(line).name if rule.get("basename") else line for line in lines if line]
        suffix = rule.get("require_suffix")
        if suffix:
            records = [r for r in records if (root / f"{r}{suffix}").exists()]
        return records
    if "glob" in rule:
        pattern = rule["glob"]
        use_stem = rule.get("stem", pattern.endswith(".hea"))
        paths = [p for p in root.glob(pattern) if p.is_file()]
        records = [p.stem if use_stem else p.relative_to(root).as_posix() for p in paths]
        return _sort_records(records, rule.get("sort", "lexical"))
    raise HarnessError(f"unsupported record rule {rule}")


def resolve_datasets(ctx: Context, tool: Mapping[str, Any]) -> list[dict[str, Any]]:
    resolved = []
    for dataset in tool.get("datasets", []):
        root = Path(ctx.expand(dataset["root"]))
        records = resolve_records(root, dataset["records"])
        files: list[str] = []
        for pattern in dataset["files"]:
            if "{record}" in pattern:
                files.extend(pattern.replace("{record}", r) for r in records)
            else:
                files.append(pattern)
        for pattern in dataset.get("extra_glob", []):
            files.extend(p.relative_to(root).as_posix() for p in root.glob(pattern) if p.is_file())
        manifest = dataset_manifest(root, files, ctx.hash_cache)
        entry = {k: v for k, v in dataset.items() if k not in ("root", "files", "records", "extra_glob")}
        root_label = dataset["root"]
        entry.update({
            "root": root_label,
            "record_rule": dataset["records"],
            "record_count": len(records),
            "records": records,
            "records_sha256": sha256_bytes(canonical_json(records)),
            "manifest_file_patterns": dataset["files"] + dataset.get("extra_glob", []),
            "manifest": manifest,
        })
        resolved.append(entry)
    return resolved


def combined_manifest_sha(datasets: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json([[d["name"], d["manifest"]["sha256"]] for d in datasets]))


def prepare_corpora(ctx: Context, tools: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Build/verify every corpus the selected tools need.

    Returns ``{corpus_id: error}`` for corpora that could not be built; the
    tools depending on them are reported as ``blocked`` instead of aborting
    the whole run.
    """

    needed = set()
    for tool in tools:
        needed.update(_corpus_ids(tool))
    errors: dict[str, str] = {}
    for corpus_id in sorted(needed):
        spec = ctx.spec["corpora"][corpus_id]
        target = ctx.corpus_dir(corpus_id)
        try:
            if spec["builder"] == "batch_pt":
                build = _corpora.build_batch_pt_corpus(spec, ctx.data_root, target)
            elif spec["builder"] == "smv":
                build = _corpora.build_smv_corpus(spec, ctx.data_root, target, WORKSPACE_ROOT)
            else:
                raise HarnessError(f"unknown corpus builder {spec['builder']}")
        except (HarnessError, OSError, ImportError, ValueError) as exc:
            errors[corpus_id] = f"corpus {corpus_id!r} unavailable: {type(exc).__name__}: {exc}"
            continue
        ctx.corpus_builds[corpus_id] = build
    return errors


def _corpus_ids(tool: Mapping[str, Any]) -> set[str]:
    return set(re.findall(r"\{corpus:([a-z0-9_]+)\}", json.dumps(tool)))


# ── Execution ───────────────────────────────────────────────────────────────


def _config_digest(tool: Mapping[str, Any]) -> str:
    keys = ("script", "steps", "extraction_config", "normalizer")
    return sha256_bytes(canonical_json({k: tool.get(k) for k in keys} | {
        "datasets": [{k: d.get(k) for k in ("name", "root", "records", "files", "extra_glob")}
                     for d in tool.get("datasets", [])]}))


def _expand_argv(ctx: Context, argv: Sequence[str], run_dir: Path, records: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in argv:
        if item == "{records}":
            out.extend(records)
        else:
            out.append(ctx.expand(item, run_dir))
    return out


def _read_import_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def execute_run(ctx: Context, tool: Mapping[str, Any], run_index: int, run_dir: Path,
                records: Sequence[str], log: callable) -> dict[str, Any]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    import_log = run_dir / "import_origins.jsonl"
    env = ctx.tool_env(import_log)
    steps_out = []
    started = time.perf_counter()
    status = "ok"
    for step in tool["steps"]:
        step_started = time.perf_counter()
        kind = step["kind"]
        if kind == "copy":
            src = Path(ctx.expand(step["from"], run_dir))
            dst = Path(ctx.expand(step["to"], run_dir))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copyfile(src, dst)
            steps_out.append({"name": step["name"], "kind": kind, "exit_code": 0,
                              "wall_seconds": time.perf_counter() - step_started})
            continue
        if kind == "tool":
            script = ctx.repo_root / step.get("script", tool["script"])
            cmd = [str(ctx.python), str(script)]
        elif kind == "module":
            cmd = [str(ctx.python), "-m", step["module"]]
        else:
            raise HarnessError(f"unknown step kind {kind}")
        cmd += _expand_argv(ctx, step["argv"], run_dir, records)
        log_path = run_dir / f"{step['name']}.log"
        timeout = float(tool.get("timeout_minutes", max(30, 4 * tool.get("estimated_minutes", 10)))) * 60
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(cmd) + "\n")
            handle.flush()
            try:
                proc = subprocess.run(cmd, cwd=str(ctx.repo_root), env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, timeout=timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                exit_code = "timeout"
        ok_codes = step.get("ok_exit_codes", tool.get("ok_exit_codes", [0]))
        step_status = "ok" if exit_code in ok_codes else "crashed"
        steps_out.append({"name": step["name"], "kind": kind,
                          "argv": [a if not a.startswith(str(run_dir)) else a.replace(str(run_dir), "<RUN>") for a in cmd[1:]],
                          "exit_code": exit_code, "status": step_status,
                          "wall_seconds": round(time.perf_counter() - step_started, 2),
                          "log": str(log_path)})
        log(f"    {tool['id']} run{run_index} step {step['name']}: exit={exit_code} "
            f"({time.perf_counter() - step_started:.0f}s)")
        if step_status != "ok":
            status = "crashed"
            break
    origins = _read_import_log(import_log)
    root = os.path.realpath(ctx.repo_root) + os.sep
    foreign = [row for row in origins
               if not row.get("origin") or not os.path.realpath(row["origin"]).startswith(root)]
    if foreign:
        status = "contaminated_import"
    return {
        "run": run_index,
        "status": status,
        "raw_dir": str(run_dir),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "steps": steps_out,
        "import_origins": {
            "processes_logged": len(origins),
            "modules": sorted({(row["module"], row.get("origin")) for row in origins}),
            "all_inside_tree_under_test": not foreign,
            "foreign": foreign[:10],
        },
    }


def _tool_file_sha(ctx: Context, tool: Mapping[str, Any]) -> str | None:
    path = ctx.repo_root / tool["script"]
    return ctx.hash_cache.sha256(path) if path.exists() else None


def _metric_map(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {m["name"]: m.get("value") for m in metrics}


def determinism_report(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare repeated runs of the unchanged tree.

    ``identical`` means every *gated* metric value and every output digest
    (after removing wall-clock stamps and absolute paths) is bit-identical.
    Ungated values that are environment-dependent by construction (runtimes)
    are listed separately and never counted as nondeterminism of results.
    """

    ok_runs = [r for r in runs if r.get("normalized")]
    if len(ok_runs) < 2:
        return {"runs_compared": len(ok_runs), "identical": None,
                "note": "fewer than two successful runs"}
    first = ok_runs[0]
    diffs: list[dict[str, Any]] = []
    ungated: list[dict[str, Any]] = []
    digest_diffs: list[str] = []
    gated_names = {m["name"] for m in first["normalized"]["metrics"] if m.get("gated", True)}
    for other in ok_runs[1:]:
        a = _metric_map(first["normalized"]["metrics"])
        b = _metric_map(other["normalized"]["metrics"])
        for name in sorted(set(a) | set(b)):
            if a.get(name) != b.get(name):
                spread = None
                try:
                    spread = abs(float(a.get(name)) - float(b.get(name)))
                except (TypeError, ValueError):
                    pass
                row = {"metric": name, f"run{first['run']}": a.get(name),
                       f"run{other['run']}": b.get(name), "abs_spread": spread}
                (diffs if name in gated_names else ungated).append(row)
        da = first["normalized"].get("digests", {})
        db = other["normalized"].get("digests", {})
        for key in sorted(set(da) | set(db)):
            if da.get(key) != db.get(key):
                digest_diffs.append(key)
    return {
        "runs_compared": len(ok_runs),
        "identical_gated_metrics": not diffs,
        "identical_output_digests": not digest_diffs,
        "identical": not diffs and not digest_diffs,
        "metric_differences": diffs,
        "differing_output_digests": digest_diffs,
        "ungated_value_differences": ungated,
        "ungated_note": "runtime/wall-clock values; environment-dependent, not gated, not a result difference",
    }


def build_result(ctx: Context, tool: Mapping[str, Any], datasets: list[dict[str, Any]],
                 runs: list[dict[str, Any]], env: Mapping[str, Any], subset_run: bool) -> dict[str, Any]:
    good = [r for r in runs if r.get("normalized")]
    primary = good[0] if good else None
    status = "pinned" if primary and all(r["status"] == "ok" for r in runs) else "failed"
    metrics = finalize_metrics(primary["normalized"]["metrics"]) if primary else []
    determinism = determinism_report(runs)
    # Attach observed run-to-run spread next to the frozen tolerance, never widening it.
    spread_by_metric = {d["metric"]: d.get("abs_spread") for d in determinism.get("metric_differences", [])}
    flagged = []
    for item in metrics:
        if item["name"] in spread_by_metric:
            item["observed_run_to_run_spread"] = spread_by_metric[item["name"]]
            allowed = (item.get("tolerance") or {}).get("allowed_regression")
            spread = spread_by_metric[item["name"]]
            if item.get("gated") and (allowed is None or spread is None or spread > allowed):
                item["spread_exceeds_tolerance"] = True
                flagged.append(item["name"])
    if flagged:
        determinism["gated_metrics_with_spread_exceeding_tolerance"] = flagged
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "tool": {
            "id": tool["id"],
            "name": tool["script"],
            "sha256": _tool_file_sha(ctx, tool),
            "gate_role": tool.get("gate_role"),
            "cadence": tool.get("cadence"),
            "config_digest": _config_digest(tool),
        },
        "datasets": datasets,
        "dataset_manifest_sha256": combined_manifest_sha(datasets),
        "extraction_config": tool.get("extraction_config"),
        "record_schema_version": None,
        "record_schema_note": "legacy ecgfeat export (pre-ECG-Record); no record schema is exercised by these tools at this revision",
        "environment": env,
        "invocation": [
            {"name": s["name"], "kind": s["kind"], "argv": s.get("argv"), "module": s.get("module"),
             "from": s.get("from"), "to": s.get("to")}
            for s in tool["steps"]
        ],
        "runs": [{k: v for k, v in r.items() if k != "normalized"} for r in runs],
        "determinism": determinism,
        "metrics": metrics,
        "metric_count": len(metrics),
        "gated_metric_count": sum(1 for m in metrics if m.get("gated")),
        "digests": primary["normalized"].get("digests", {}) if primary else {},
        "notes": primary["normalized"].get("notes", {}) if primary else {},
        "subset_run": subset_run,
        "generated_at": _utcnow(),
    }
    if tool["id"] in ctx.corpus_builds or any(
        f"{{corpus:{cid}}}" in json.dumps(tool) for cid in ctx.corpus_builds
    ):
        result["corpus_builds"] = {
            cid: {k: v for k, v in build.items() if k != "source_file_sha256"}
            for cid, build in ctx.corpus_builds.items()
            if f"{{corpus:{cid}}}" in json.dumps(tool)
        }
    if status != "pinned":
        result["failure_reason"] = "; ".join(
            f"run{r['run']}: {r['status']}" + (f" ({r.get('normalize_error')})" if r.get("normalize_error") else "")
            for r in runs if r["status"] != "ok" or not r.get("normalized")
        )
    return result


def blocked_or_delegated_result(tool: Mapping[str, Any], ctx: Context | None) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": tool["status"],
        "tool": {"id": tool["id"], "name": tool["script"],
                 "sha256": _tool_file_sha(ctx, tool) if ctx else None,
                 "gate_role": tool.get("gate_role"), "cadence": tool.get("cadence")},
        "reason": tool.get("delegated_reason") or tool.get("blocked_reason"),
        "metrics": [],
        "metric_count": 0,
        "generated_at": _utcnow(),
    }


class Scheduler:
    """Run jobs concurrently while the declared worker total stays <= budget."""

    def __init__(self, budget: int, log: callable) -> None:
        self.budget = budget
        self.log = log

    def run(self, jobs: list[dict[str, Any]]) -> None:
        pending = sorted(jobs, key=lambda j: -j["estimate"])
        running: list[tuple[threading.Thread, dict[str, Any]]] = []
        used = 0
        while pending or running:
            launched = False
            for job in list(pending):
                if used + job["workers"] <= self.budget:
                    pending.remove(job)
                    used += job["workers"]
                    thread = threading.Thread(target=job["fn"], daemon=True)
                    thread.start()
                    running.append((thread, job))
                    self.log(f"  start {job['label']} (workers={job['workers']}, in use={used}/{self.budget})")
                    launched = True
            if not launched:
                time.sleep(2.0)
            for thread, job in list(running):
                if not thread.is_alive():
                    running.remove((thread, job))
                    used -= job["workers"]
            if pending and not running and not any(j["workers"] <= self.budget for j in pending):
                raise HarnessError("a job needs more workers than the budget")


def run_command(args) -> int:
    spec = load_spec(Path(args.spec))
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "feature_extraction" / "ecgfeat").is_dir():
        raise HarnessError(f"{repo_root} is not an ecgfeat tree")
    baseline_id = args.baseline_id
    out_dir = Path(args.out_dir) if args.out_dir else BENCHMARKS_DIR / "baselines" / baseline_id
    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_ROOT / baseline_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    budget = int(args.max_workers or spec.get("max_workers", 8))
    if budget > 8 and not args.allow_more_workers:
        raise HarnessError("max workers capped at 8 on this machine (pass --allow-more-workers to override)")
    ctx = Context(repo_root=repo_root, python=Path(args.python), spec=spec, raw_dir=raw_dir,
                  corpus_root=Path(args.corpus_root), hash_cache=FileHashCache(DEFAULT_RAW_ROOT / ".hash_cache.json"))
    lock = threading.Lock()

    def log(msg: str) -> None:
        with lock:
            print(msg, flush=True)

    selected = [t for t in spec["tools"] if not args.tools or t["id"] in args.tools]
    unknown = set(args.tools or []) - {t["id"] for t in spec["tools"]}
    if unknown:
        raise HarnessError(f"unknown tool ids: {sorted(unknown)}")

    for tool in selected:
        if tool["status"] != "pinned":
            path = out_dir / f"{tool['id']}.json"
            path.write_text(dump_result(blocked_or_delegated_result(tool, ctx)), encoding="utf-8")
            log(f"[{tool['id']}] {tool['status']}: {tool.get('delegated_reason') or tool.get('blocked_reason')}")
    pinned = [t for t in selected if t["status"] == "pinned"]
    for tool in pinned:
        path = out_dir / f"{tool['id']}.json"
        if path.exists() and not args.force:
            raise HarnessError(f"{path} exists; frozen results are immutable (use --force to replace deliberately)")

    log(f"tree under test: {repo_root}")
    env = capture_environment(ctx.python, repo_root, ctx.tool_env(None))
    assert_tree_import(env, repo_root)
    log(f"ecgfeat resolves to {env['ecgfeat_file']}; git {env['git'].get('commit')} clean={env['git'].get('clean')}")

    log("preparing corpora ...")
    corpus_errors = prepare_corpora(ctx, pinned)
    log("hashing dataset manifests ...")
    datasets_by_tool = {}
    runnable = []
    for tool in pinned:
        missing = [corpus_errors[c] for c in sorted(_corpus_ids(tool)) if c in corpus_errors]
        try:
            if missing:
                raise HarnessError("; ".join(missing))
            datasets_by_tool[tool["id"]] = resolve_datasets(ctx, tool)
        except (HarnessError, OSError, ValueError) as exc:
            reason = f"input unavailable ({type(exc).__name__}: {exc}); 'dataset unavailable' is not a pass"
            blocked = dict(tool, status="blocked", blocked_reason=reason)
            result = blocked_or_delegated_result(blocked, ctx)
            result["environment"] = env
            (out_dir / f"{tool['id']}.json").write_text(dump_result(result), encoding="utf-8")
            log(f"[{tool['id']}] BLOCKED: {reason}")
            continue
        finally:
            ctx.hash_cache.save()
        runnable.append(tool)
        log(f"  {tool['id']}: " + ", ".join(
            f"{d['name']} {d['record_count']} rec / {d['manifest']['file_count']} files "
            f"sha256={d['manifest']['sha256'][:12]}" for d in datasets_by_tool[tool["id"]]))
    pinned = runnable

    repeat = int(args.repeat)
    runs_by_tool: dict[str, list[dict[str, Any]]] = {t["id"]: [] for t in pinned}
    jobs = []
    for tool in pinned:
        records = datasets_by_tool[tool["id"]][0]["records"] if datasets_by_tool[tool["id"]] else []
        for k in range(1, repeat + 1):
            run_dir = raw_dir / tool["id"] / f"run{k}"

            def job(tool=tool, k=k, run_dir=run_dir, records=records):
                try:
                    run = execute_run(ctx, tool, k, run_dir, records, log)
                except Exception as exc:  # noqa: BLE001
                    run = {"run": k, "status": f"harness_error: {exc}", "raw_dir": str(run_dir)}
                if run["status"] == "ok":
                    try:
                        normalizer = getattr(_normalizers, tool["normalizer"])
                        run["normalized"] = normalizer(tool, run_dir, ctx.path_normalizer(run_dir),
                                                       datasets_by_tool[tool["id"]])
                    except Exception as exc:  # noqa: BLE001
                        import traceback
                        run["normalize_error"] = f"{type(exc).__name__}: {exc}"
                        run["normalize_traceback"] = traceback.format_exc()
                with lock:
                    runs_by_tool[tool["id"]].append(run)
                log(f"  done {tool['id']} run{k}: {run['status']}"
                    + (f" normalize_error={run.get('normalize_error')}" if run.get("normalize_error") else "")
                    + f" ({run.get('wall_seconds', 0):.0f}s)")

            jobs.append({"label": f"{tool['id']} run{k}", "workers": int(tool.get("workers", 1)),
                         "estimate": float(tool.get("estimated_minutes", 5)), "fn": job})
    Scheduler(budget, log).run(jobs)

    for tool in pinned:
        runs = sorted(runs_by_tool[tool["id"]], key=lambda r: r["run"])
        result = build_result(ctx, tool, datasets_by_tool[tool["id"]], runs, env, subset_run=False)
        (out_dir / f"{tool['id']}.json").write_text(dump_result(result), encoding="utf-8")
        det = result["determinism"]
        log(f"[{tool['id']}] status={result['status']} metrics={result['metric_count']} "
            f"(gated {result.get('gated_metric_count')}) identical_runs={det.get('identical')}")
    write_index(out_dir, baseline_id, spec, Path(args.spec))
    ctx.hash_cache.save()
    return 0


def write_index(out_dir: Path, baseline_id: str, spec: Mapping[str, Any], spec_path: Path) -> dict[str, Any]:
    tools = []
    envs = {}
    for tool in spec["tools"]:
        path = out_dir / f"{tool['id']}.json"
        if not path.exists():
            tools.append({"id": tool["id"], "script": tool["script"], "status": "not_run"})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {"id": tool["id"], "script": tool["script"], "status": result["status"],
               "file": path.name, "tool_sha256": result["tool"].get("sha256"),
               "metric_count": result.get("metric_count", 0),
               "gated_metric_count": result.get("gated_metric_count", 0)}
        if result.get("reason"):
            row["reason"] = result["reason"]
        if result.get("failure_reason"):
            row["failure_reason"] = result["failure_reason"]
        if "determinism" in result:
            det = result["determinism"]
            row["determinism"] = {"runs_compared": det.get("runs_compared"), "identical": det.get("identical"),
                                  "identical_gated_metrics": det.get("identical_gated_metrics"),
                                  "identical_output_digests": det.get("identical_output_digests"),
                                  "ungated_runtime_values_differing": len(det.get("ungated_value_differences", []))}
            row["dataset_manifest_sha256"] = result.get("dataset_manifest_sha256")
            row["wall_seconds_per_run"] = [r.get("wall_seconds") for r in result.get("runs", [])]
            envs[result["environment"].get("git", {}).get("commit")] = result["environment"]
        tools.append(row)
    env = next(iter(envs.values())) if len(envs) == 1 else {"multiple": envs}
    index = {
        "schema_version": BASELINE_INDEX_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "git_revision": env.get("git", {}).get("commit") if "git" in env else None,
        "environment": env,
        "spec_sha256": sha256_bytes(spec_path.read_bytes()),
        "tolerance_policy": {
            "source": "docs/library_design/06_testing_and_validation.md, 'Common regression tolerances'",
            "bounded_accuracy": "aggregate <= 0.5 pp; predeclared subgroup/lead <= 1.0 pp (values are fractions: 0.005 / 0.010)",
            "error": "<= max(2 % relative, one reporting quantum); signed biases compared on |value|",
            "coverage": "fraction: <= 0.5 pp; yield count: candidate >= 99.5 % of baseline count",
            "failure_count": "0 new failures",
            "exact": "0 differences",
            "frozen_before_any_migrated_result": True,
            "never_widened_for_observed_spread": True,
        },
        "tools": tools,
        "generated_at": _utcnow(),
    }
    (out_dir / "baseline.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index


def renormalize_command(args) -> int:
    """Rebuild normalized results from existing raw run directories.

    Harness-side only (normalizer/digest fixes): the tools are NOT re-run, the
    frozen run metadata (exit codes, wall time, import origins, dataset
    manifests, environment) is carried over from the existing result files.
    """

    spec = load_spec(Path(args.spec))
    out_dir = Path(args.out_dir) if args.out_dir else BENCHMARKS_DIR / "baselines" / args.baseline_id
    repo_root = Path(args.repo_root).resolve()
    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_ROOT / args.baseline_id
    ctx = Context(repo_root=repo_root, python=Path(args.python), spec=spec, raw_dir=raw_dir,
                  corpus_root=Path(args.corpus_root), hash_cache=FileHashCache(DEFAULT_RAW_ROOT / ".hash_cache.json"))
    tools = [t for t in spec["tools"] if t["status"] == "pinned" and (not args.tools or t["id"] in args.tools)]
    errors = prepare_corpora(ctx, tools)
    if errors:
        raise HarnessError(f"corpora unavailable for renormalize: {errors}")
    for tool in tools:
        path = out_dir / f"{tool['id']}.json"
        if not path.exists():
            print(f"[{tool['id']}] no existing result; skipped")
            continue
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("status") != "pinned":
            print(f"[{tool['id']}] status {old.get('status')}; skipped")
            continue
        if old["tool"].get("config_digest") != _config_digest(tool):
            raise HarnessError(f"{tool['id']}: spec changed since the raw runs; re-run instead")
        runs = []
        for run in old["runs"]:
            run = dict(run)
            run_dir = Path(run["raw_dir"])
            run["normalized"] = getattr(_normalizers, tool["normalizer"])(
                tool, run_dir, ctx.path_normalizer(run_dir), old["datasets"])
            runs.append(run)
        result = build_result(ctx, tool, old["datasets"], runs, old["environment"], subset_run=False)
        result["tool"]["sha256"] = old["tool"]["sha256"]
        result["renormalized_at"] = result["generated_at"]
        result["generated_at"] = old.get("generated_at")
        path.write_text(dump_result(result), encoding="utf-8")
        det = result["determinism"]
        print(f"[{tool['id']}] metrics={result['metric_count']} identical={det.get('identical')} "
              f"(gated metrics {det.get('identical_gated_metrics')}, digests {det.get('identical_output_digests')}, "
              f"ungated runtime diffs {len(det.get('ungated_value_differences', []))})")
    for tool in spec["tools"]:
        path = out_dir / f"{tool['id']}.json"
        if tool["status"] != "pinned" and path.exists():
            path.write_text(dump_result(json.loads(path.read_text(encoding="utf-8"))), encoding="utf-8")
    write_index(out_dir, args.baseline_id, spec, Path(args.spec))
    return 0
