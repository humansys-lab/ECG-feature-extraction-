"""Metric model, frozen tolerance policy, hashing and environment capture.

Everything in this module is independent of the tree under test: it never
imports ``ecgfeat``.  The tolerance policy is the one fixed by
``docs/library_design/06_testing_and_validation.md`` ("Common regression
tolerances") and is written into every metric of a frozen baseline *before*
any migrated result exists, so a candidate can never pick a friendlier rule
after seeing its own numbers.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RESULT_SCHEMA_VERSION = "ecgfeat-benchmark-result.v1"
BASELINE_INDEX_SCHEMA_VERSION = "ecgfeat-benchmark-baseline-index.v1"
COMPARE_SCHEMA_VERSION = "ecgfeat-benchmark-compare.v1"

FAMILIES = ("bounded_accuracy", "error", "coverage", "failure_count", "exact")
DIRECTIONS = ("higher_is_better", "lower_is_better")
LEVELS = ("aggregate", "subgroup")

# Doc 06 "Common regression tolerances".  Bounded-accuracy values in this
# harness are fractions (1.0 == 100 %), so 0.5 percentage points == 0.005.
BOUNDED_ACCURACY_AGGREGATE_PP = 0.5
BOUNDED_ACCURACY_SUBGROUP_PP = 1.0
ERROR_RELATIVE = 0.02
COVERAGE_PP = 0.5
FAILURE_COUNT_NEW = 0

# Numerical slack used only to stop binary floating point from turning an
# exactly-at-the-limit value into a failure.  It is not a tolerance.
_FLOAT_SLACK_REL = 1e-9
_FLOAT_SLACK_ABS = 1e-12


class HarnessError(RuntimeError):
    """Raised for harness misconfiguration (never for a benchmark regression)."""


# ── Metric construction ─────────────────────────────────────────────────────


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def tolerance_for(metric: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the frozen allowed regression for one baseline metric.

    ``allowed_regression`` is always expressed in the metric's own unit after
    the ``compare_on`` transform, and always as a non-negative amount by which
    the candidate may be *worse* than the baseline.
    """

    if not metric.get("gated", True):
        return None
    family = metric["family"]
    level = metric.get("level", "aggregate")
    value = metric.get("value")
    if family == "exact":
        return {"kind": "exact", "allowed_regression": 0,
                "rule": "exact result: 0 unexpected differences"}
    if family == "failure_count":
        return {"kind": "absolute", "allowed_regression": FAILURE_COUNT_NEW,
                "rule": "crash/error count on a fixed corpus: 0 new failures"}
    if family == "bounded_accuracy":
        pp = BOUNDED_ACCURACY_SUBGROUP_PP if level == "subgroup" else BOUNDED_ACCURACY_AGGREGATE_PP
        scale = 100.0 if metric.get("unit") == "percent" else 1.0
        return {
            "kind": "absolute",
            "allowed_regression": pp / 100.0 * scale,
            "rule": f"bounded accuracy ({level}): <= {pp} percentage points",
        }
    if family == "coverage":
        if metric.get("unit") == "count":
            base = _finite(value)
            allowed = None if base is None else abs(base) * COVERAGE_PP / 100.0
            return {
                "kind": "relative_count",
                "allowed_regression": allowed,
                "rule": (
                    "coverage/yield count: <= 0.5 percentage points of the "
                    "baseline count (candidate >= 99.5 % of baseline)"
                ),
            }
        scale = 100.0 if metric.get("unit") == "percent" else 1.0
        return {
            "kind": "absolute",
            "allowed_regression": COVERAGE_PP / 100.0 * scale,
            "rule": "coverage/yield fraction: <= 0.5 percentage points",
        }
    if family == "error":
        base = _finite(value)
        quantum = float(metric["quantum"])
        if base is None:
            return {
                "kind": "absolute",
                "allowed_regression": None,
                "rule": "error: baseline unavailable, candidate unconstrained",
            }
        if metric.get("compare_on") == "abs":
            base = abs(base)
        allowed = max(ERROR_RELATIVE * abs(base), quantum)
        return {
            "kind": "absolute",
            "allowed_regression": allowed,
            "rule": "error: <= max(2 % relative, one reporting quantum)",
        }
    raise HarnessError(f"unknown metric family {family!r}")


def metric(
    name: str,
    value: Any,
    *,
    family: str,
    unit: str,
    direction: str | None = None,
    quantum: float | None = None,
    level: str = "aggregate",
    compare_on: str = "value",
    gated: bool = True,
    note: str | None = None,
) -> dict[str, Any]:
    """Build one normalized metric record (tolerance attached later)."""

    if family not in FAMILIES:
        raise HarnessError(f"{name}: unknown family {family!r}")
    if level not in LEVELS:
        raise HarnessError(f"{name}: unknown level {level!r}")
    if direction is None:
        direction = {
            "bounded_accuracy": "higher_is_better",
            "coverage": "higher_is_better",
            "error": "lower_is_better",
            "failure_count": "lower_is_better",
            "exact": None,
        }[family]
    if direction is not None and direction not in DIRECTIONS:
        raise HarnessError(f"{name}: unknown direction {direction!r}")
    if quantum is None:
        quantum = {"failure_count": 1, "coverage": None, "exact": None}.get(family)
    if family == "error" and quantum is None:
        raise HarnessError(f"{name}: error metrics need a reporting quantum")
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    record: dict[str, Any] = {
        "name": name,
        "value": value,
        "family": family,
        "direction": direction,
        "unit": unit,
        "quantum": quantum,
        "level": level,
        "compare_on": compare_on,
        "gated": bool(gated),
    }
    if note:
        record["note"] = note
    return record


def info(name: str, value: Any, *, unit: str = "", note: str | None = None) -> dict[str, Any]:
    """An emitted value that is recorded but deliberately not gated."""

    return metric(name, value, family="exact", unit=unit, gated=False,
                  note=note or "recorded, not gated (no quality direction or environment-dependent)")


def finalize_metrics(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort, check name uniqueness and attach the frozen tolerance."""

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in metrics:
        record = dict(item)
        if record["name"] in seen:
            raise HarnessError(f"duplicate metric name {record['name']!r}")
        seen.add(record["name"])
        record["tolerance"] = tolerance_for(record)
        out.append(record)
    out.sort(key=lambda row: row["name"])
    return out


# ── Metric comparison ───────────────────────────────────────────────────────


def _worse_by(direction: str, baseline: float, candidate: float) -> float:
    return (baseline - candidate) if direction == "higher_is_better" else (candidate - baseline)


def compare_metric(base: Mapping[str, Any], cand: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply the *baseline's* frozen tolerance to one candidate metric.

    Returns a dict with ``status`` in {pass, improvement, informational,
    regression, new_failure, exact_diff, missing, became_unavailable,
    definition_changed}.
    """

    name = base["name"]
    row: dict[str, Any] = {"metric": name, "family": base["family"],
                           "baseline": base.get("value")}
    if cand is None:
        row.update(status="missing", candidate=None,
                   reason="metric absent from candidate result")
        return row
    row["candidate"] = cand.get("value")
    if not base.get("gated", True):
        row["status"] = "informational"
        row["changed"] = cand.get("value") != base.get("value")
        return row
    for key in ("family", "direction", "unit", "compare_on"):
        if cand.get(key) != base.get(key):
            row.update(status="definition_changed",
                       reason=f"{key}: baseline {base.get(key)!r} != candidate {cand.get(key)!r}")
            return row
    family = base["family"]
    if family == "exact":
        equal = cand.get("value") == base.get("value")
        row["status"] = "pass" if equal else "exact_diff"
        return row
    bval = _finite(base.get("value"))
    cval = _finite(cand.get("value"))
    tolerance = base.get("tolerance") or tolerance_for(base) or {}
    allowed = tolerance.get("allowed_regression")
    row["allowed_regression"] = allowed
    if bval is None:
        row["status"] = "pass"
        row["reason"] = "baseline value unavailable; candidate unconstrained"
        return row
    if cval is None:
        row["status"] = "became_unavailable"
        row["reason"] = "baseline had a value, candidate has none"
        return row
    if base.get("compare_on") == "abs":
        bval, cval = abs(bval), abs(cval)
    worse = _worse_by(base["direction"], bval, cval)
    row["worse_by"] = worse
    limit = float(allowed or 0.0)
    slack = _FLOAT_SLACK_ABS + _FLOAT_SLACK_REL * max(abs(bval), abs(cval), limit)
    if worse > limit + slack:
        row["status"] = "new_failure" if family == "failure_count" else "regression"
    elif worse < -slack:
        row["status"] = "improvement"
    else:
        row["status"] = "pass"
    return row


FAILING_STATUSES = frozenset(
    {"regression", "new_failure", "exact_diff", "missing", "became_unavailable",
     "definition_changed"}
)


# ── Hashing ─────────────────────────────────────────────────────────────────


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


class FileHashCache:
    """sha256 cache keyed on (realpath, size, mtime_ns); LTSTDB alone is ~10 GB."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path is not None and path.exists():
            try:
                self.entries = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.entries = {}
        self.dirty = False

    def sha256(self, file_path: Path) -> str:
        real = os.path.realpath(file_path)
        stat = os.stat(real)
        entry = self.entries.get(real)
        if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
            return entry["sha256"]
        digest = hashlib.sha256()
        with open(real, "rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self.entries[real] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": value}
        self.dirty = True
        return value

    def save(self) -> None:
        if self.path is None or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        self.dirty = False


def dataset_manifest(
    root: Path,
    relative_paths: Iterable[str],
    cache: FileHashCache,
) -> dict[str, Any]:
    """sha256 over sorted (relative path, file sha256) of the files used.

    Missing files raise: a corpus that is not complete must fail before scoring.
    """

    entries: list[tuple[str, str]] = []
    total = 0
    missing: list[str] = []
    for rel in sorted(set(relative_paths)):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        entries.append((rel, cache.sha256(path)))
        total += os.path.getsize(os.path.realpath(path))
    if missing:
        raise HarnessError(
            f"{len(missing)} manifest input file(s) missing under {root}: {missing[:5]}"
        )
    digest = hashlib.sha256()
    for rel, file_hash in entries:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256 over sorted '<relative path>\\0<file sha256>\\n'",
        "sha256": digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": total,
    }


# ── Output digests (determinism evidence; informational in compare) ─────────


class PathNormalizer:
    """Replace machine/tree specific absolute prefixes with placeholders."""

    def __init__(self, replacements: Sequence[tuple[str, str]]) -> None:
        pairs = {}
        for prefix, token in replacements:
            if prefix:
                pairs[str(prefix).rstrip("/")] = token
        self.pairs = sorted(pairs.items(), key=lambda item: len(item[0]), reverse=True)

    def text(self, value: str) -> str:
        for prefix, token in self.pairs:
            if prefix in value:
                value = value.replace(prefix, token)
        return value

    def obj(self, value: Any, drop_keys: frozenset[str] = frozenset()) -> Any:
        if isinstance(value, dict):
            return {k: self.obj(v, drop_keys) for k, v in value.items() if k not in drop_keys}
        if isinstance(value, list):
            return [self.obj(v, drop_keys) for v in value]
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value


def digest_csv(path: Path, normalizer: PathNormalizer, drop_columns: Iterable[str] = (),
               encoding: str = "utf-8-sig") -> str:
    drop = set(drop_columns)
    if not path.exists():
        return "absent"
    text = path.read_text(encoding=encoding)
    rows = []
    for row in csv.reader(io.StringIO(text)):
        rows.append(row)
    if not rows:
        return sha256_bytes(b"")
    header = rows[0]
    keep = [i for i, col in enumerate(header) if col not in drop]
    canon = [[normalizer.text(row[i]) if i < len(row) else "" for i in keep] for row in rows]
    return sha256_bytes(canonical_json(canon))


def digest_json(path: Path, normalizer: PathNormalizer, drop_keys: Iterable[str] = ()) -> str:
    if not path.exists():
        return "absent"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sha256_bytes(canonical_json(normalizer.obj(payload, frozenset(drop_keys))))


def digest_text(text: str, normalizer: PathNormalizer) -> str:
    return sha256_bytes(normalizer.text(text).encode("utf-8"))


# ── Environment ─────────────────────────────────────────────────────────────

_ENV_PROBE = r"""
import importlib.metadata as md, json, os, platform, sys
out = {
    "python": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "python_executable": sys.executable,
    "os": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
}
packages = {}
for name in ("numpy", "scipy", "numba", "llvmlite", "wfdb", "neurokit2", "biosppy",
             "peakutils", "pandas", "matplotlib", "torch", "PyWavelets"):
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
out["packages"] = packages
try:
    import numpy
    cfg = numpy.show_config(mode="dicts")
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    out["numpy_blas"] = {"name": blas.get("name"), "version": blas.get("version")}
except Exception as exc:
    out["numpy_blas"] = {"error": repr(exc)}
import ecgfeat
out["ecgfeat_file"] = os.path.realpath(ecgfeat.__file__)
try:
    import ecgagent
    out["ecgagent_file"] = os.path.realpath(ecgagent.__file__)
except Exception as exc:
    out["ecgagent_file"] = None
try:
    import torch
    out["accelerator"] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "used_by_tools": False,
    }
except Exception as exc:
    out["accelerator"] = {"error": repr(exc)}
out["thread_env"] = {k: os.environ.get(k) for k in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")}
print(json.dumps(out))
"""


def git_revision(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    try:
        head = run("rev-parse", "HEAD")
        dirty_lines = [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line]
        return {"commit": head, "tracked_changes": len(dirty_lines),
                "tracked_changes_sample": dirty_lines[:20], "clean": not dirty_lines}
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"commit": None, "error": str(exc)}


def capture_environment(python: Path, repo_root: Path, env: Mapping[str, str]) -> dict[str, Any]:
    proc = subprocess.run([str(python), "-c", _ENV_PROBE], cwd=str(repo_root), env=dict(env),
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise HarnessError(f"environment probe failed: {proc.stderr[-2000:]}")
    probe = json.loads(proc.stdout.strip().splitlines()[-1])
    probe["git"] = git_revision(repo_root)
    probe["repo_root"] = str(repo_root)
    probe["harness_python"] = sys.version.split()[0]
    return probe


def assert_tree_import(probe: Mapping[str, Any], repo_root: Path) -> None:
    root = os.path.realpath(repo_root) + os.sep
    for key in ("ecgfeat_file", "ecgagent_file"):
        origin = probe.get(key)
        if origin and not os.path.realpath(origin).startswith(root):
            raise HarnessError(
                f"{key} resolves to {origin}, outside the tree under test {repo_root}; "
                "refusing to run (contaminated import path)"
            )


def dump_result(result: Mapping[str, Any]) -> str:
    """Serialize a result with one metric per line (diff-friendly, compact)."""

    metrics = list(result.get("metrics", []))
    body = dict(result)
    body["metrics"] = "__HARNESS_METRICS__"
    text = json.dumps(body, indent=2, ensure_ascii=False)
    rendered = "[]" if not metrics else "[\n" + ",\n".join(
        "    " + json.dumps(item, ensure_ascii=False) for item in metrics) + "\n  ]"
    return text.replace('"__HARNESS_METRICS__"', rendered, 1) + "\n"
