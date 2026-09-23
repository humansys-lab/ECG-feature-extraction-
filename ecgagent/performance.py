"""Backend-independent metrics for recorded runs; never supplies model input."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x))
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "mean": None}
    def percentile(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (pos - lower)
    return {"count": len(ordered), "p50": percentile(.5), "p95": percentile(.95),
            "mean": sum(ordered) / len(ordered)}


def summarize_performance(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize observed runs without treating unavailable telemetry as zero.

    Token distributions include only records with complete request history and
    valid telemetry for every request. Partial observations are exposed as
    subtotals under token_telemetry; a recorded empty request list is true zero.
    """
    phase_latency: dict[str, list[float]] = defaultdict(list)
    record_latency: list[float] = []
    calls: list[float] = []
    completion_tokens: list[float] = []
    prompt_tokens: list[float] = []
    token_samples = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    token_telemetry = {
        key: {"observed_requests": 0, "missing_requests": 0, "observed_tokens": 0}
        for key in token_samples
    }
    evidence: Counter[str] = Counter()
    selection: Counter[str] = Counter()
    local_repairs = local_repair_successes = 0
    request_history_records = 0
    for payload in payloads:
        audit = payload.get("audit") or {}
        batch = payload.get("batch") or {}
        runtime = audit.get("runtime_controls") or {}
        duration = batch.get("runtime_seconds")
        if not _valid_measurement(duration):
            duration = runtime.get("elapsed_seconds")
        if _valid_measurement(duration):
            record_latency.append(duration)
        attempts = batch.get("attempts") or []
        # New attempts persist logical requests from every attempt, including
        # unsuccessful ones. Older artifacts expose only the final attempt.
        complete_history = bool(attempts) and all(
            isinstance(row, Mapping) and isinstance(row.get("model_requests"), list)
            for row in attempts
        )
        if complete_history:
            requests = [request for row in attempts for request in row["model_requests"]]
        else:
            requests = runtime.get("model_requests")
        if isinstance(requests, list):
            history_known = complete_history or len(attempts) <= 1
            request_history_records += int(history_known)
            calls.append(len(requests))
            for key, samples in token_samples.items():
                values = [row.get(key) if isinstance(row, Mapping) else None for row in requests]
                observed = [value for value in values if _valid_token_count(value)]
                telemetry = token_telemetry[key]
                telemetry["observed_requests"] += len(observed)
                telemetry["missing_requests"] += len(values) - len(observed)
                telemetry["observed_tokens"] += sum(observed)
                if history_known and len(observed) == len(values):
                    samples.append(sum(observed))
            for row in requests:
                if not isinstance(row, Mapping):
                    continue
                elapsed = row.get("elapsed_seconds")
                if _valid_measurement(elapsed):
                    phase_latency[str(row.get("phase") or "unknown")].append(elapsed)
        for row in (audit.get("model_evidence_coverage") or {}).values():
            evidence[str(row.get("status") or "unknown")] += 1
        merge = (audit.get("rule_second_opinion") or {}).get("merge") or {}
        selection["proposed"] += len(merge.get("proposed_candidate_codes") or [])
        selection["retained"] += int(merge.get("merged_candidate_count") or 0)
        for row in merge.get("candidates_dropped_by_view_budget") or []:
            selection[str(row.get("reason") or "view_budget")] += 1
        for phase in payload.get("phases") or []:
            retries = int(phase.get("phase_guard_attempts") or 0)
            if retries and phase.get("phase") in {"plan", "adjudicate"}:
                local_repairs += 1
                local_repair_successes += int(phase.get("phase_guard_passed") is True)
    return {
        "record_count": len(payloads),
        "record_latency_seconds": distribution(record_latency),
        "model_calls_per_record": distribution(calls),
        "prompt_tokens_per_record": distribution(prompt_tokens),
        "completion_tokens_per_record": distribution(completion_tokens),
        "token_telemetry": {
            key: {**token_telemetry[key], "complete_records": len(samples),
                  "incomplete_records": len(payloads) - len(samples)}
            for key, samples in token_samples.items()
        },
        "model_request_latency_by_phase": {key: distribution(values) for key, values in sorted(phase_latency.items())},
        "complete_request_history_records": request_history_records,
        "model_node_evidence": dict(evidence),
        "model_node_minimum_visibility_rate": evidence["visible"] / sum(evidence.values()) if evidence else None,
        "candidate_selection": dict(selection),
        "local_repair_phases": local_repairs,
        "local_repair_successes": local_repair_successes,
    }


def _valid_measurement(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _valid_token_count(value: Any) -> bool:
    return _valid_measurement(value) and int(value) == value


def label_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score a locked cohort, representing failed/missing outputs as abstention."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    exact = 0
    for row in rows:
        expected = set(row["reference_categories"])
        predicted = set(row["agent_categories"])
        exact += int(expected == predicted)
        for label in expected | predicted:
            counts[label]["tp" if label in expected & predicted else "fn" if label in expected else "fp"] += 1
    def score(count: Mapping[str, int]) -> dict[str, Any]:
        tp, fp, fn = (count.get(key, 0) for key in ("tp", "fp", "fn"))
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    totals = Counter()
    for count in counts.values():
        totals.update(count)
    return {"record_count": len(rows), "agent": score(totals),
            "exact_match_rate": exact / len(rows) if rows else None,
            "per_category": {key: score(count) for key, count in sorted(counts.items())},
            "failure_policy": "missing, invalid, stale or unverified output is an abstention (empty prediction)"}


def main(argv: Sequence[str] | None = None) -> int:
    """Compare saved worker-count/model runs without making model requests."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = []
    for directory in args.runs:
        def read(name: str) -> dict[str, Any]:
            path = directory / name
            return json.loads(path.read_text()) if path.exists() else {}
        analysis = read("analysis.json")
        manifest = read("diagnosis_manifest.json")
        records = read("record_manifest.json").get("records") or []
        ids = sorted(str(row.get("record")) for row in records)
        report.append({
            "run": str(directory),
            "cohort_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest() if ids else None,
            "workers": manifest.get("workers"),
            "records_per_second": manifest.get("records_per_second"),
            "execution_wall_seconds": manifest.get("execution_wall_seconds"),
            "verified_coverage": analysis.get("verified_coverage"),
            "full_cohort_direct_label_agreement": analysis.get("full_cohort_direct_label_agreement"),
            "performance": analysis.get("performance"),
        })
    cohorts = {row["cohort_sha256"] for row in report}
    rendered = json.dumps({"same_locked_cohort": None not in cohorts and len(cohorts) == 1,
                           "runs": report}, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
