"""Compare a candidate result set against a frozen baseline.

Order of checks per tool (a changed corpus fails *before* any metric is scored):

1. baseline status: tools blocked/delegated in the baseline are reported as
   ungated, never as passing;
2. candidate presence and status;
3. dataset-content manifest hash (and per-dataset manifest + record list);
4. pinned configuration digest (argv/subset/extraction config);
5. every baseline metric with the baseline's frozen tolerance; a baseline
   metric absent from the candidate is a failure.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import COMPARE_SCHEMA_VERSION, FAILING_STATUSES, compare_metric


def load_results(directory: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "baseline.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tool_id = (payload.get("tool") or {}).get("id")
        if tool_id:
            results[tool_id] = payload
    return results


def compare_tool(base: Mapping[str, Any], cand: Mapping[str, Any] | None) -> dict[str, Any]:
    tool_id = base["tool"]["id"]
    report: dict[str, Any] = {"tool": tool_id, "baseline_status": base.get("status")}
    if base.get("status") != "pinned":
        report.update(status="ungated", passed=None,
                      reason=f"baseline status is {base.get('status')!r}: {base.get('reason') or base.get('failure_reason')}")
        return report
    if cand is None:
        report.update(status="missing_candidate", passed=False, reason="no candidate result for a pinned tool")
        return report
    report["candidate_status"] = cand.get("status")
    if cand.get("status") in ("blocked", "delegated"):
        report.update(status="candidate_blocked", passed=False,
                      reason=f"candidate could not run a pinned tool: {cand.get('reason')}")
        return report
    report["tool_sha256"] = {"baseline": base["tool"].get("sha256"), "candidate": (cand.get("tool") or {}).get("sha256")}
    report["tool_sha256"]["changed"] = report["tool_sha256"]["baseline"] != report["tool_sha256"]["candidate"]
    b_manifest = base.get("dataset_manifest_sha256")
    c_manifest = cand.get("dataset_manifest_sha256")
    report["manifest"] = {"baseline": b_manifest, "candidate": c_manifest, "match": b_manifest == c_manifest}
    if b_manifest != c_manifest or not c_manifest:
        per_dataset = []
        c_sets = {d["name"]: d for d in cand.get("datasets", [])}
        for d in base.get("datasets", []):
            other = c_sets.get(d["name"])
            per_dataset.append({
                "dataset": d["name"],
                "baseline_sha256": d["manifest"]["sha256"],
                "candidate_sha256": other["manifest"]["sha256"] if other else None,
                "records_match": bool(other) and other.get("records_sha256") == d.get("records_sha256"),
            })
        report.update(status="manifest_mismatch", passed=False, datasets=per_dataset,
                      reason="dataset-content manifest hash differs; the corpus changed, metrics not scored")
        return report
    b_cfg = base["tool"].get("config_digest")
    c_cfg = (cand.get("tool") or {}).get("config_digest")
    if b_cfg != c_cfg:
        report.update(status="config_mismatch", passed=False,
                      reason="pinned invocation/extraction configuration digest differs")
        return report
    if cand.get("status") != "pinned":
        report.update(status="candidate_failed", passed=False,
                      reason=cand.get("failure_reason") or f"candidate status {cand.get('status')!r}")
        return report
    c_metrics = {m["name"]: m for m in cand.get("metrics", [])}
    rows = [compare_metric(m, c_metrics.get(m["name"])) for m in base.get("metrics", [])]
    extra = sorted(set(c_metrics) - {m["name"] for m in base.get("metrics", [])})
    counts = Counter(row["status"] for row in rows)
    failing = [row for row in rows if row["status"] in FAILING_STATUSES]
    report.update(
        status="fail" if failing else "pass",
        passed=not failing,
        counts=dict(sorted(counts.items())),
        failures=failing,
        improvements=[row for row in rows if row["status"] == "improvement"],
        informational_changes=[row for row in rows if row["status"] == "informational" and row.get("changed")],
        candidate_only_metrics=extra,
        output_digests_changed=sorted(
            key for key in set(base.get("digests", {})) | set(cand.get("digests", {}))
            if base.get("digests", {}).get(key) != cand.get("digests", {}).get(key)
        ),
    )
    return report


def compare_dirs(baseline_dir: Path, candidate_dir: Path, tools: Sequence[str] | None = None) -> dict[str, Any]:
    base = load_results(baseline_dir)
    cand = load_results(candidate_dir)
    selected = [t for t in base if not tools or t in tools]
    missing_selection = sorted(set(tools or []) - set(base))
    reports = {t: compare_tool(base[t], cand.get(t)) for t in selected}
    failing = [t for t, r in reports.items() if r.get("passed") is False]
    ungated = [t for t, r in reports.items() if r.get("passed") is None]
    index_path = Path(baseline_dir) / "baseline.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    summary = {
        "tools_compared": len(reports),
        "tools_passed": sorted(t for t, r in reports.items() if r.get("passed") is True),
        "tools_failed": sorted(failing),
        "tools_ungated": sorted(ungated),
        "unknown_tools_requested": missing_selection,
        "regressions": sum(r.get("counts", {}).get("regression", 0) for r in reports.values()),
        "new_failures": sum(r.get("counts", {}).get("new_failure", 0) for r in reports.values()),
        "exact_diffs": sum(r.get("counts", {}).get("exact_diff", 0) for r in reports.values()),
        "missing_metrics": sum(r.get("counts", {}).get("missing", 0) for r in reports.values()),
        "became_unavailable": sum(r.get("counts", {}).get("became_unavailable", 0) for r in reports.values()),
        "manifest_mismatches": sorted(t for t, r in reports.items() if r["status"] == "manifest_mismatch"),
    }
    return {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "baseline": {"dir": str(baseline_dir), "baseline_id": index.get("baseline_id"),
                     "git_revision": index.get("git_revision")},
        "candidate": {"dir": str(candidate_dir)},
        "passed": not failing and not missing_selection,
        "summary": summary,
        "tools": reports,
    }
