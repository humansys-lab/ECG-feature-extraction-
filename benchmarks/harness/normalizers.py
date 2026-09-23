"""Turn each tool's own output files into normalized metric records.

Normalizers only *read* what the tool wrote; they never recompute the tool's
metrics differently.  The one exception is ``compare_annotations.py``, which
emits only a per-record ``summary.csv`` and no aggregate at all: there the
harness aggregates the per-record differences the tool reports (documented
in the metric notes).

Metric naming: ``<group>.<subgroup...>.<statistic>``.  Family assignment:

* rate in [0, 1] of hits (Se, PPV, F1, specificity, recall, pearson r,
  direction accuracy, within-limit share) -> ``bounded_accuracy``;
* timing/amplitude error statistics -> ``error`` (signed biases/extremes are
  compared on their absolute value);
* number of scored items/records -> ``coverage`` (unit ``count``);
* failed records / failed calls / measurement-contradicted calls ->
  ``failure_count``;
* quantities fixed by the reference corpus alone (annotated event counts,
  record counts, annotation-derived reference values) -> ``exact``;
* raw confusion counts, detection counts, runtimes and descriptive value
  distributions -> recorded, not gated (``gated: false``).
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import (
    HarnessError,
    PathNormalizer,
    canonical_json,
    digest_csv,
    digest_json,
    digest_text,
    info,
    metric,
    sha256_bytes,
)

VOLATILE_KEYS = frozenset({
    "generated_at", "generated_at_utc", "wall_clock_seconds", "wall_runtime_seconds",
    "runtime_seconds", "runtime_seconds_total", "seconds", "total_seconds",
    "mean_seconds", "p95_seconds", "extractor_code_fingerprint", "traceback",
})

# Wall-clock stamps written into the human-readable reports by
# demo_feature_extraction.generate_report ("  Generated : 2026-09-23  09:02:51").
_VOLATILE_TEXT_LINE = re.compile(r"^\s*Generated\s*:.*$", re.MULTILINE)


def _stable_text(text: str, norm: PathNormalizer) -> str:
    return _VOLATILE_TEXT_LINE.sub("  Generated : <timestamp>", norm.text(text))


LUDB_STATS = {"n": "count", "mean": "signed", "sd": "error", "mae": "error", "p50": "error", "p95": "error"}
DET_STATS = {"n": "count", "bias_ms": "signed", "sd_ms": "error", "mae_ms": "error",
             "median_ae_ms": "error", "p95_ae_ms": "error"}
CSE_STATS = {"n": "count", "mean_ms": "signed", "sd_ms": "error", "mae_ms": "error",
             "median_abs_ms": "error", "p95_abs_ms": "error", "min_ms": "signed", "max_ms": "signed"}


def _slug(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text)).strip("_")


def _stats_metrics(prefix: str, stats: Mapping[str, Any] | None, keys: Mapping[str, str], *,
                   unit: str, quantum: float, level: str, key_prefix: str = "") -> list[dict[str, Any]]:
    stats = stats or {}
    out = []
    for key, kind in keys.items():
        value = stats.get(key_prefix + key)
        name = f"{prefix}.{key}"
        if kind == "count":
            out.append(metric(name, value, family="coverage", unit="count", level=level))
        elif kind == "signed":
            out.append(metric(name, value, family="error", unit=unit, quantum=quantum,
                              level=level, compare_on="abs"))
        elif kind == "error":
            out.append(metric(name, value, family="error", unit=unit, quantum=quantum, level=level))
        else:
            raise HarnessError(kind)
    return out


def _read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _log_text(run_dir: Path, step: str) -> str:
    text = (run_dir / f"{step}.log").read_text(encoding="utf-8", errors="replace")
    return text.split("\n", 1)[1] if text.startswith("$ ") else text


# ── evaluate_ludb.py ────────────────────────────────────────────────────────


def evaluate_ludb(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    out = run_dir / "out"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m: list[dict[str, Any]] = []
    for field in ("qrs_on_err", "qrs_off_err", "t_on_err", "t_off_err", "p_on_err", "p_off_err", "qt_err"):
        m += _stats_metrics(field, summary[field], LUDB_STATS, unit="ms", quantum=0.1, level="aggregate")
    for lead, stats in sorted(summary.get("qt_err_by_lead", {}).items()):
        m += _stats_metrics(f"qt_err.lead_{lead}", stats, LUDB_STATS, unit="ms", quantum=0.1, level="subgroup")
    for group in ("qt_err_reliable", "qt_err_unreliable"):
        m += _stats_metrics(group, summary[group], LUDB_STATS, unit="ms", quantum=0.1, level="subgroup")
    cov = summary["coverage"]
    m += [
        metric("coverage.gt_qrs", cov["gt_qrs"], family="exact", unit="count",
               note="annotated QRS complexes; fixed by the corpus"),
        metric("coverage.matched_qrs", cov["matched_qrs"], family="coverage", unit="count"),
        metric("coverage.unmatched_gt_qrs", cov["unmatched_gt_qrs"], family="error", unit="count", quantum=1),
        metric("coverage.unmatched_det_in_gt_span", cov["unmatched_det_in_gt_span"], family="error",
               unit="count", quantum=1),
    ]
    for key, row in sorted(cov["boundaries"].items()):
        m += [
            metric(f"coverage.{key}.annotated", row["annotated"], family="exact", unit="count"),
            metric(f"coverage.{key}.scored", row["scored"], family="coverage", unit="count"),
            metric(f"coverage.{key}.missing_or_unmatched", row["missing_or_unmatched"], family="error",
                   unit="count", quantum=1),
            metric(f"coverage.{key}.coverage", row["coverage"], family="coverage", unit="fraction"),
        ]
    m += [
        metric("requested_records", summary["requested_records"], family="exact", unit="count"),
        metric("failed_records", len(summary["failed_records"]), family="failure_count", unit="count"),
        info("operational_coverage_complete", summary["operational_coverage_complete"]),
    ]
    digests = {
        "ludb_errors.csv": digest_csv(out / "ludb_errors.csv", norm),
        "summary.json": digest_json(out / "summary.json", norm, VOLATILE_KEYS),
    }
    return {"metrics": m, "digests": digests,
            "notes": {"failed_records": summary["failed_records"],
                      "scored_beat_lead_pairs": summary["qrs_on_err"]["n"]}}


# ── shared detector-comparison rows (evaluate_qtdb / compare_ludb_detectors) ─


def _detector_row(prefix: str, row: Mapping[str, Any], level: str) -> list[dict[str, Any]]:
    m = [
        metric(f"{prefix}.records", row.get("records"), family="exact", unit="count"),
        metric(f"{prefix}.lead_records", row.get("lead_records"), family="exact", unit="count"),
        metric(f"{prefix}.failed_lead_records", row.get("failed_lead_records"), family="failure_count",
               unit="count"),
        metric(f"{prefix}.gt_count", row.get("gt_count"), family="exact", unit="count",
               note="reference events inside the scored span; fixed by the annotations"),
        info(f"{prefix}.det_count", row.get("det_count"), unit="count"),
        info(f"{prefix}.tp", row.get("tp"), unit="count"),
        info(f"{prefix}.fp", row.get("fp"), unit="count"),
        info(f"{prefix}.fn", row.get("fn"), unit="count"),
        metric(f"{prefix}.sensitivity", row.get("sensitivity"), family="bounded_accuracy",
               unit="fraction", level=level),
        metric(f"{prefix}.precision", row.get("precision"), family="bounded_accuracy",
               unit="fraction", level=level),
        metric(f"{prefix}.f1", row.get("f1"), family="bounded_accuracy", unit="fraction", level=level),
    ]
    for boundary in ("onset", "peak", "offset"):
        m += _stats_metrics(f"{prefix}.{boundary}", row, DET_STATS, unit="ms", quantum=0.1,
                            level=level, key_prefix=f"{boundary}_")
    return m


# ── evaluate_qtdb.py ────────────────────────────────────────────────────────


def evaluate_qtdb(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    out = run_dir / "out"
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    meta = payload["metadata"]
    m: list[dict[str, Any]] = []
    for row in payload["summary"]:
        m += _detector_row(f"{row['method']}.{row['wave']}", row, "aggregate")
    for row in payload["cse_conformance"]:
        m.append(info(f"cse.{row['method']}.{row['wave']}_{row['boundary']}.within_cse_2sd_limit",
                      row.get("within_cse_limit"),
                      note="boolean verdict of the tool; its bias/SD inputs are gated in the summary rows"))
    m += [
        metric("records_evaluated", meta["records_evaluated"], family="exact", unit="count"),
        metric("records_available", meta["records_available"], family="exact", unit="count"),
        metric("failures", meta["failures"], family="failure_count", unit="count"),
        info("release_gate.numeric_pass", meta["release_gate"]["numeric_pass"]),
        info("release_gate.operational_coverage_complete", meta["release_gate"]["operational_coverage_complete"]),
        info("wall_clock_seconds", meta.get("wall_clock_seconds"), unit="s"),
    ]
    digests = {name: digest_csv(out / name, norm, VOLATILE_KEYS)
               for name in ("detection_by_record.csv", "matched_events.csv", "summary_by_record.csv",
                            "summary_by_method_wave.csv", "cse_conformance.csv", "failures.csv")}
    digests["metrics.json"] = digest_json(out / "metrics.json", norm, VOLATILE_KEYS)
    return {"metrics": m, "digests": digests,
            "notes": {"per_record_rows_not_gated": "summary_by_record.csv (digest only)",
                      "package_versions": meta.get("package_versions")}}


# ── compare_ludb_detectors.py ───────────────────────────────────────────────


def compare_ludb_detectors(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    out = run_dir / "out"
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m: list[dict[str, Any]] = []
    for row in payload["summary"]:
        m += _detector_row(f"{row['method']}.{row['wave']}", row, "aggregate")
    for row in payload["summary_by_lead"]:
        m += _detector_row(f"{row['method']}.{row['wave']}.lead_{row['lead']}", row, "subgroup")
    for row in payload["runtime_summary"]:
        prefix = f"runtime.{row['method']}"
        m += [
            metric(f"{prefix}.calls", row["calls"], family="exact", unit="count"),
            metric(f"{prefix}.failed_calls", row["failed_calls"], family="failure_count", unit="count"),
            info(f"{prefix}.total_seconds", row["total_seconds"], unit="s"),
            info(f"{prefix}.mean_seconds", row["mean_seconds"], unit="s"),
            info(f"{prefix}.p95_seconds", row["p95_seconds"], unit="s"),
        ]
    m += [
        metric("failure_count", payload["failure_count"], family="failure_count", unit="count"),
        metric("dataset.record_count", payload["metadata"]["dataset"]["record_count"], family="exact",
               unit="count"),
    ]
    digests = {name: digest_csv(out / name, norm, VOLATILE_KEYS)
               for name in ("detection_rows.csv", "matched_events.csv", "record_summary.csv",
                            "summary.csv", "summary_by_lead.csv", "failures.csv")}
    return {"metrics": m, "digests": digests,
            "notes": {"versions": payload["metadata"].get("versions"),
                      "per_record_rows_not_gated": "record_summary.csv (digest only)"}}


# ── evaluate_ludb_cse.py ────────────────────────────────────────────────────


def evaluate_ludb_cse(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    out = run_dir / "out"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m: list[dict[str, Any]] = []
    for key, item in sorted(summary["metrics"].items()):
        m += [
            metric(f"{key}.reference_available", item["reference_available"], family="exact", unit="count"),
            metric(f"{key}.algorithm_available", item["algorithm_available"], family="coverage", unit="count"),
            metric(f"{key}.paired_records", item["paired_records"], family="coverage", unit="count"),
            metric(f"{key}.paired_coverage", item["paired_coverage"], family="coverage", unit="fraction"),
            info(f"{key}.mean_pass", item["mean_pass"]),
            info(f"{key}.sd_pass", item["sd_pass"]),
            info(f"{key}.numeric_pass", item["numeric_pass"]),
            info(f"{key}.complete_on_reference_available", item["complete_on_reference_available"]),
            info(f"{key}.outliers_removed", [row["record_id"] for row in item["outliers_removed"]]),
        ]
        m += _stats_metrics(f"{key}.raw", item["raw"], CSE_STATS, unit="ms", quantum=0.01, level="aggregate")
        m += _stats_metrics(f"{key}.trimmed", item["trimmed"], CSE_STATS, unit="ms", quantum=0.01,
                            level="aggregate")
        for stat, value in sorted(item["algorithm_values"].items()):
            m.append(info(f"{key}.algorithm_values.{stat}", value, unit="ms" if stat != "n" else "count",
                          note="distribution of algorithm measurements; no quality direction"))
        for stat, value in sorted(item["reference_values"].items()):
            m.append(metric(f"{key}.reference_values.{stat}", value, family="exact",
                            unit="ms" if stat != "n" else "count",
                            note="annotation-derived reference; must not move (0 unexplained annotation-mapping changes)"))
    m += [
        metric("requested_records", summary["requested_records"], family="exact", unit="count"),
        metric("successful_records", summary["successful_records"], family="coverage", unit="count"),
        metric("failed_records", len(summary["failed_records"]), family="failure_count", unit="count"),
        info("all_metrics_numeric_pass", summary["all_metrics_numeric_pass"]),
        info("measurement_coverage_complete", summary["measurement_coverage_complete"]),
        info("proxy_verdict", summary["proxy_verdict"]),
    ]
    digests = {
        "record_measurements.csv": digest_csv(out / "record_measurements.csv", norm, VOLATILE_KEYS),
        "summary.json": digest_json(out / "summary.json", norm, VOLATILE_KEYS),
    }
    return {"metrics": m, "digests": digests, "notes": {"failed_records": summary["failed_records"]}}


# ── compare_annotations.py (--batch-reports) ────────────────────────────────

_CA_INTERVALS = (("hr", "bpm"), ("pr", "ms"), ("qrs", "ms"), ("qt", "ms"), ("qtcb", "ms"),
                 ("qtcf", "ms"), ("qt_dispersion", "ms"))
_CA_AXES = ("p_axis", "qrs_axis", "t_axis")


def _diff_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "mae": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=float)
    absolute = np.abs(array)
    return {"n": int(array.size), "mean": float(np.mean(array)), "sd": float(np.std(array)),
            "mae": float(np.mean(absolute)), "p50": float(np.median(absolute)),
            "p95": float(np.percentile(absolute, 95))}


def compare_annotations(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    out = run_dir / "out"
    rows = _read_csv(out / "summary.csv", encoding="utf-8")
    requested = datasets[0]["records"]
    by_id = {row["record_id"]: row for row in rows}
    missing = [r for r in requested if r not in by_id]
    ordered = [by_id[r] for r in requested if r in by_id]
    agg_note = ("aggregated by the harness from the tool's per-record summary.csv "
                "(diff = algorithm - annotation-derived reference); the tool itself emits no aggregate")
    m: list[dict[str, Any]] = [
        metric("records_requested", len(requested), family="exact", unit="count"),
        metric("records_completed", len(ordered), family="coverage", unit="count"),
        metric("records_failed", len(missing), family="failure_count", unit="count",
               note="records printed as SKIP by the tool (absent from summary.csv)"),
    ]
    for key, unit in _CA_INTERVALS:
        diffs = [v for row in ordered if (v := _num(row.get(f"diff_{key}"))) is not None]
        stats = _diff_stats(diffs)
        for stat, kind in LUDB_STATS.items():
            name = f"{key}.diff.{stat}"
            value = stats[stat]
            if kind == "count":
                m.append(metric(name, value, family="coverage", unit="count", note=agg_note))
            else:
                m.append(metric(name, value, family="error", unit=unit, quantum=0.1,
                                compare_on="abs" if kind == "signed" else "value", note=agg_note))
        m.append(metric(f"{key}.algorithm_available",
                        sum(_num(row.get(f"algorithm_{key}")) is not None for row in ordered),
                        family="coverage", unit="count"))
        m.append(metric(f"{key}.reference_available",
                        sum(_num(row.get(f"ground_truth_{key}")) is not None for row in ordered),
                        family="exact", unit="count"))
    for axis in _CA_AXES:
        diffs = [v for row in ordered if (v := _num(row.get(f"abs_{axis}_circular_diff"))) is not None]
        stats = _diff_stats(diffs)
        m += [
            metric(f"{axis}.abs_circular_diff.n", stats["n"], family="coverage", unit="count", note=agg_note),
            metric(f"{axis}.abs_circular_diff.mean", stats["mean"], family="error", unit="deg",
                   quantum=0.1, note=agg_note),
            metric(f"{axis}.abs_circular_diff.p50", stats["p50"], family="error", unit="deg",
                   quantum=0.1, note=agg_note),
            metric(f"{axis}.abs_circular_diff.p95", stats["p95"], family="error", unit="deg",
                   quantum=0.1, note=agg_note),
            metric(f"{axis}.reference_available",
                   sum(_num(row.get(f"ground_truth_{axis}")) is not None for row in ordered),
                   family="exact", unit="count"),
        ]
    m.append(metric("ground_truth_beats.total", sum(int(row["ground_truth_beats"]) for row in ordered),
                    family="exact", unit="count"))
    m.append(info("algorithm_beats.total", sum(int(row["algorithm_beats"]) for row in ordered), unit="count"))
    for key in ("p_missing", "t_missing", "qrs_missing", "beat_unreliable", "total"):
        m.append(info(f"algorithm_{key}.total", sum(int(row[f"algorithm_{key}"]) for row in ordered),
                      unit="count"))
        m.append(metric(f"ground_truth_{key}.total", sum(int(row[f"ground_truth_{key}"]) for row in ordered),
                        family="exact", unit="count"))
    for column in ("record_grade", "pacing_state"):
        counts = Counter(row.get(column) or "" for row in ordered)
        m.append(info(f"{column}.distribution", dict(sorted(counts.items()))))
    gt_columns = sorted({c for row in ordered for c in row if c.startswith("ground_truth_")} | {"reference_gt_lead"})
    reference_table = [[row["record_id"]] + [row.get(c, "") for c in gt_columns] for row in ordered]
    m.append(metric("annotation_reference_digest", sha256_bytes(canonical_json([gt_columns, reference_table])),
                    family="exact", unit="sha256",
                    note="all ground_truth_* columns (annotation-derived reference); gate '0 unexplained annotation-mapping changes'"))
    comparison_texts = "".join(
        norm.text((out / r / f"{r}_comparison.txt").read_text(encoding="utf-8"))
        for r in requested if (out / r / f"{r}_comparison.txt").exists()
    )
    features = [digest_json(out / r / f"{r}_features.json", norm, VOLATILE_KEYS)
                for r in requested if (out / r / f"{r}_features.json").exists()]
    reports = "".join(
        _stable_text((out / r / f"{r}_{kind}_report.txt").read_text(encoding="utf-8"), norm)
        for r in requested for kind in ("algorithm", "ground_truth")
        if (out / r / f"{r}_{kind}_report.txt").exists()
    )
    digests = {
        "summary.csv": digest_csv(out / "summary.csv", norm, encoding="utf-8"),
        "per_record_comparison_txt": sha256_bytes(comparison_texts.encode("utf-8")),
        "per_record_features_json": sha256_bytes(canonical_json(features)),
        "per_record_reports_txt": sha256_bytes(reports.encode("utf-8")),
    }
    return {"metrics": m, "digests": digests, "notes": {"skipped_records": missing}}


# ── analyze_physionet_st.py ─────────────────────────────────────────────────


def analyze_physionet_st(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    out = run_dir / "out"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m: list[dict[str, Any]] = []
    for row in summary["metrics"]:
        prefix = ".".join(_slug(row[k]) for k in ("dataset", "cohort", "comparison", "method"))
        m.append(metric(f"{prefix}.n", row.get("n"), family="coverage", unit="count"))
        if "mae_mv" in row or ("tp" not in row and row.get("n") == 0):
            m += [
                metric(f"{prefix}.mae_mv", row.get("mae_mv"), family="error", unit="mV", quantum=0.001),
                metric(f"{prefix}.rmse_mv", row.get("rmse_mv"), family="error", unit="mV", quantum=0.001),
                metric(f"{prefix}.bias_mv", row.get("bias_mv"), family="error", unit="mV", quantum=0.001,
                       compare_on="abs"),
                metric(f"{prefix}.median_error_mv", row.get("median_error_mv"), family="error", unit="mV",
                       quantum=0.001, compare_on="abs"),
                metric(f"{prefix}.within_0p05_mv", row.get("within_0p05_mv"), family="bounded_accuracy",
                       unit="fraction"),
                metric(f"{prefix}.pearson_r", row.get("pearson_r"), family="bounded_accuracy",
                       unit="fraction", note="correlation coefficient treated as bounded accuracy-like"),
                metric(f"{prefix}.direction_accuracy", row.get("direction_accuracy"),
                       family="bounded_accuracy", unit="fraction"),
            ]
        else:
            for key in ("tp", "fp", "fn", "tn"):
                m.append(info(f"{prefix}.{key}", row.get(key), unit="count"))
            for key in ("sensitivity", "specificity", "precision", "f1"):
                m.append(metric(f"{prefix}.{key}", row.get(key), family="bounded_accuracy", unit="fraction"))
    for ds in ("edb", "ltstdb"):
        m += [
            metric(f"{ds}.records_selected", summary[f"{ds}_records_selected"], family="exact", unit="count"),
            metric(f"{ds}.records_processed", summary[f"{ds}_records_processed"], family="coverage",
                   unit="count"),
            metric(f"{ds}.event_rows", summary[f"{ds}_event_rows"], family="coverage", unit="count"),
            metric(f"{ds}.control_rows", summary[f"{ds}_control_rows"], family="coverage", unit="count"),
        ]
    m.append(metric("failures", len(summary["failures"]), family="failure_count", unit="count"))
    digests = {name: digest_csv(out / name, norm, VOLATILE_KEYS)
               for name in ("edb_event_measurements.csv", "edb_control_measurements.csv",
                            "ltstdb_event_measurements.csv", "ltstdb_control_measurements.csv",
                            "metrics.csv", "failures.csv")}
    return {"metrics": m, "digests": digests,
            "notes": {"failures": [{"dataset": f["dataset"], "record": f["record"], "error": f["error"]}
                                   for f in summary["failures"]]}}


# ── batch_extract_ecgfeat.py ────────────────────────────────────────────────


def batch_extract_ecgfeat(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    m: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    notes: dict[str, Any] = {}
    records = datasets[0]["records"]  # e.g. ptbxl500/cd/real.pt
    for step in tool["steps"]:
        name = step["name"]
        argv = step["argv"]
        variant_input = Path(argv[argv.index("--input-dir") + 1]).name
        out = run_dir / Path(argv[argv.index("--output-dir") + 1].replace("{run_dir}/", ""))
        jobs = [r for r in records if r.startswith(variant_input + "/")]
        manifests = sorted(out.glob("*/*/manifest.json"))
        processed = failed = skipped = expected = 0
        errors: list[str] = []
        versions = set()
        for path in manifests:
            man = json.loads(path.read_text(encoding="utf-8"))
            processed += int(man["processed"])
            failed += int(man["failed"])
            skipped += int(man["skipped"])
            expected += int(man["input_shape"][0])
            errors += [f"{path.parent.parent.name}: {e}" for e in man.get("errors", [])]
            if man.get("clinical_ruleset_version"):
                versions.add(man["clinical_ruleset_version"])
        feature_files = sorted(out.glob("*/*/*_features.json"))
        report_files = sorted(out.glob("*/*/*_report.txt"))
        parse_ok = 0
        sizes = []
        key_sets = []
        feature_digests = []
        for path in feature_files:
            raw = path.read_bytes()
            sizes.append(len(raw))
            try:
                payload = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            parse_ok += 1
            key_sets.append(tuple(sorted(payload)) if isinstance(payload, dict) else ("<non-object>",))
            feature_digests.append([path.relative_to(out).as_posix(),
                                    sha256_bytes(canonical_json(norm.obj(payload, VOLATILE_KEYS)))])
        top_keys = sorted(set().union(*key_sets)) if key_sets else []
        consistent = len(set(key_sets)) <= 1
        prefix = f"{name}"
        m += [
            metric(f"{prefix}.jobs", len(manifests), family="exact", unit="count",
                   note=f"label/source jobs found (corpus has {len(jobs)})"),
            metric(f"{prefix}.samples_expected", expected, family="exact", unit="count"),
            metric(f"{prefix}.processed", processed, family="coverage", unit="count"),
            metric(f"{prefix}.processed_fraction", processed / expected if expected else None,
                   family="coverage", unit="fraction"),
            metric(f"{prefix}.failed", failed, family="failure_count", unit="count"),
            metric(f"{prefix}.skipped", skipped, family="exact", unit="count"),
            metric(f"{prefix}.feature_json_written", len(feature_files), family="coverage", unit="count"),
            metric(f"{prefix}.feature_json_parse_ok_fraction",
                   parse_ok / len(feature_files) if feature_files else None, family="coverage",
                   unit="fraction",
                   note="stand-in for schema validation: the legacy export has no JSON schema at this revision"),
            metric(f"{prefix}.feature_json_top_level_keys", "|".join(top_keys), family="exact", unit="keys",
                   note="legacy export top-level contract"),
            metric(f"{prefix}.feature_json_top_level_keys_consistent", consistent, family="exact", unit="bool"),
            metric(f"{prefix}.report_txt_written", len(report_files), family="coverage", unit="count"),
            info(f"{prefix}.feature_json_bytes_max", max(sizes) if sizes else None, unit="bytes",
                 note="indent=2 legacy export; the 24,000-byte limit applies to the golden summary reference, not to this file"),
            info(f"{prefix}.feature_json_bytes_median", float(np.median(sizes)) if sizes else None, unit="bytes"),
            info(f"{prefix}.clinical_ruleset_version", sorted(versions)),
        ]
        reports_text = "".join(_stable_text(p.read_text(encoding="utf-8"), norm) for p in report_files)
        digests[f"{name}.feature_json"] = sha256_bytes(canonical_json(feature_digests))
        digests[f"{name}.report_txt"] = sha256_bytes(reports_text.encode("utf-8"))
        digests[f"{name}.manifests"] = sha256_bytes(canonical_json(
            [norm.obj(json.loads(p.read_text(encoding="utf-8")), VOLATILE_KEYS) for p in manifests]))
        notes[f"{name}.errors"] = errors[:50]
    return {"metrics": m, "digests": digests, "notes": notes}


# ── score_measurement_verifiable.py ─────────────────────────────────────────

_BUCKETS = (
    "measurement-confirmed and labelled",
    "unlabelled-but-correct",
    "measurement-contradicted",
    "not-checkable (no reliable measurement)",
    "not-checkable (morphology family)",
)


def score_measurement_verifiable(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    text = _log_text(run_dir, "score")
    total_match = re.search(r"--- confirmed diagnoses: (\d+) ---", text)
    if total_match is None:
        raise HarnessError("score output lacks the 'confirmed diagnoses' line")
    buckets = {name: 0 for name in _BUCKETS}
    for line in text.splitlines():
        hit = re.match(r"^  (.+?)\s+(\d+)\s+(?:[\d.]+%|--)\s*$", line)
        if hit and hit.group(1).strip() in buckets:
            buckets[hit.group(1).strip()] = int(hit.group(2))
    contradicted, unlabelled, section = [], [], None
    for line in text.splitlines():
        if "measurement-contradicted (real errors):" in line:
            section = contradicted
            continue
        if "correct but unlabelled" in line:
            section = unlabelled
            continue
        hit = re.match(r"^    (\S+)\s+(\S+)\s*$", line)
        if section is not None and hit:
            section.append([hit.group(1), hit.group(2)])
    extraction = json.loads((run_dir / "runs" / "smv_run" / "extraction_manifest.json").read_text(encoding="utf-8"))
    confirmed = buckets["measurement-confirmed and labelled"] + buckets["unlabelled-but-correct"]
    checkable = confirmed + buckets["measurement-contradicted"]
    m = [
        metric("extraction.records", extraction["record_count"], family="exact", unit="count"),
        metric("extraction.ok", extraction["ok_or_skipped"], family="coverage", unit="count"),
        metric("extraction.failed", extraction["failed"], family="failure_count", unit="count"),
        metric("confirmed_diagnoses_total", int(total_match.group(1)), family="exact", unit="count",
               note="fixed by the frozen verdicts (given features exist)"),
        metric("measurement_contradicted", buckets["measurement-contradicted"], family="failure_count",
               unit="count", note="a frozen call its own measurement now contradicts: 0 new allowed"),
        metric("measurement_confirmed", confirmed, family="coverage", unit="count"),
        metric("measurement_confirmed_and_labelled", buckets["measurement-confirmed and labelled"],
               family="coverage", unit="count"),
        info("unlabelled_but_correct", buckets["unlabelled-but-correct"], unit="count",
             note="split of measurement_confirmed by label presence; label set is fixed"),
        metric("not_checkable_no_reliable_measurement", buckets["not-checkable (no reliable measurement)"],
               family="error", unit="count", quantum=1),
        metric("not_checkable_morphology_family", buckets["not-checkable (morphology family)"],
               family="exact", unit="count", note="depends only on the frozen diagnosis codes"),
        metric("measurement_verifiable_calls", checkable, family="coverage", unit="count"),
        metric("measurement_verifiable_accuracy", (confirmed / checkable) if checkable else None,
               family="bounded_accuracy", unit="fraction",
               note="1 - wrong share printed by the tool"),
    ]
    digests = {"score_stdout": digest_text(text, norm),
               "call_assignment": sha256_bytes(canonical_json({"buckets": buckets,
                                                               "contradicted": contradicted,
                                                               "unlabelled": unlabelled}))}
    feature_dir = run_dir / "runs" / "smv_run" / "features"
    digests["features_json"] = sha256_bytes(canonical_json(
        [[p.name, digest_json(p, norm, VOLATILE_KEYS | {"_extraction_provenance", "extraction_provenance"})]
         for p in sorted(feature_dir.glob("*_features.json"))]))
    return {"metrics": m, "digests": digests,
            "notes": {"buckets": buckets, "contradicted": contradicted, "unlabelled_but_correct": unlabelled,
                      "corpus_caveat": "2 frozen agent verdicts only; full-corpus tier blocked (needs live LLM agent run)"}}


# ── evaluate_target_ecgfeat_diagnosis.py ────────────────────────────────────


def evaluate_target_ecgfeat_diagnosis(tool, run_dir: Path, norm: PathNormalizer, datasets) -> dict[str, Any]:
    out = run_dir / "out"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m: list[dict[str, Any]] = []
    for ds in summary["datasets"]:
        p = f"dataset_{ds['dataset']}"
        m += [
            metric(f"{p}.record_count", ds["record_count"], family="exact", unit="count"),
            metric(f"{p}.extraction_succeeded", ds["extraction_succeeded"], family="coverage", unit="count"),
            metric(f"{p}.extraction_failed", ds["extraction_failed"], family="failure_count", unit="count"),
            info(f"{p}.runtime_seconds_total", ds["runtime_seconds_total"], unit="s"),
            info(f"{p}.verdict_counts", dict(sorted(ds["verdict_counts"].items()))),
            info(f"{p}.root_cause_record_counts", dict(sorted(ds["root_cause_record_counts"].items()))),
            metric(f"{p}.supported_category_count", ds["supported_category_count"], family="exact", unit="count"),
            metric(f"{p}.direct_category_count", ds["direct_category_count"], family="exact", unit="count"),
        ]
        for prefix in ("micro", "direct_micro"):
            for key in ("tp", "fp", "fn"):
                m.append(info(f"{p}.{prefix}_{key}", ds[f"{prefix}_{key}"], unit="count"))
            for key in ("precision", "recall_sensitivity", "f1"):
                m.append(metric(f"{p}.{prefix}_{key}", ds[f"{prefix}_{key}"], family="bounded_accuracy",
                                unit="fraction", level="aggregate"))
    for row in _read_csv(out / "诊断家族指标.csv"):
        p = f"dataset_{row['dataset']}.category_{row['category']}"
        m += [
            metric(f"{p}.ground_truth_positive", int(row["ground_truth_positive"]), family="exact", unit="count"),
            info(f"{p}.predicted_positive", int(row["predicted_positive"]), unit="count"),
            info(f"{p}.tp", int(row["tp"]), unit="count"),
            info(f"{p}.fp", int(row["fp"]), unit="count"),
            info(f"{p}.fn", int(row["fn"]), unit="count"),
            info(f"{p}.tn", int(row["tn"]), unit="count"),
        ]
        for key in ("precision", "recall_sensitivity", "specificity", "f1"):
            m.append(metric(f"{p}.{key}", _num(row[key]), family="bounded_accuracy", unit="fraction",
                            level="subgroup",
                            note="None when the denominator is empty (N/A is not zero hits)"))
    gate = summary["release_gate"]
    m += [
        metric("release_gate.failed_records", gate["failed_records"], family="failure_count", unit="count"),
        info("release_gate.operational_coverage_complete", gate["operational_coverage_complete"]),
    ]
    raw_files = sorted((out / "raw").glob("*/*.json"))
    raw_digests = [[p.relative_to(out).as_posix(), digest_json(p, norm, VOLATILE_KEYS)] for p in raw_files]
    digests = {"raw_records_json": sha256_bytes(canonical_json(raw_digests)),
               "诊断家族指标.csv": digest_csv(out / "诊断家族指标.csv", norm, VOLATILE_KEYS)}
    for path in sorted(out.glob("逐条诊断结果_*.csv")):
        digests[path.name] = digest_csv(path, norm, VOLATILE_KEYS)
    fingerprints = sorted({json.loads(p.read_text(encoding="utf-8")).get("extraction_contract", {}).get(
        "extractor_code_fingerprint") for p in raw_files[:5]})
    return {"metrics": m, "digests": digests,
            "notes": {"extractor_code_fingerprint_sample": fingerprints}}
