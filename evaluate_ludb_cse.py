#!/usr/bin/env python3
"""Evaluate ecgfeat interval measurements on LUDB against CSE-style limits.

This is a proxy benchmark, not a formal IEC/CSE conformance test:

* LUDB is used instead of the prescribed CSE biological ECG records.
* LUDB supplies per-lead, per-beat wave boundaries rather than official CSE
  global reference measurements.
* Record-level expert references are reconstructed with the same annotation
  aggregation harness used by ``compare_annotations.py``.

For each interval, both untrimmed statistics and the IEC/CSE-style statistics
after removing the eight largest deviations from the initial mean are
reported. Differences are always algorithm minus LUDB expert reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from compare_annotations import (
    ECGFeatureExtractor,
    REPORT_INTERNAL_FS,
    REPORT_MAINS_FREQ,
    _build_patient_meta,
    build_annotation_derived_result,
    discover_record_ids,
    load_all_gt_annotations,
    load_record,
    parse_ludb_header,
)


@dataclass(frozen=True)
class CSELimit:
    label: str
    mean_limit_ms: float
    sd_limit_ms: float


CSE_LIMITS: Mapping[str, CSELimit] = {
    "p_duration": CSELimit("P duration", 10.0, 15.0),
    "pr_interval": CSELimit("PR (PQ) interval", 10.0, 10.0),
    "qrs_duration": CSELimit("QRS duration", 10.0, 10.0),
    "qt_interval": CSELimit("QT interval", 25.0, 30.0),
}

DEFAULT_OUTLIER_COUNT = 8
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ludb_cse_evaluation"


def _finite_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _median_or_none(values: Iterable[object]) -> Optional[float]:
    finite = [
        value
        for item in values
        if (value := _finite_or_none(item)) is not None
    ]
    return float(np.median(finite)) if finite else None


def record_p_duration_ms(result) -> tuple[Optional[float], str]:
    """Read the formal record-level P duration, with legacy fallbacks."""
    global_features = getattr(result, "global_features", None)
    global_value = _finite_or_none(
        getattr(global_features, "p_duration_ms", None)
        if global_features is not None
        else None
    )
    if global_value is not None:
        source = getattr(global_features, "p_duration_source", None)
        return global_value, str(source or "global_p_duration")

    consensus = _median_or_none(
        rep.params.get("p_dur_consensus_ms")
        for rep in result.representative_leads.values()
    )
    if consensus is not None:
        return consensus, "representative_multilead_consensus"

    raw = _median_or_none(
        beat.p_dur_ms
        for beat in result.beat_features
        if "p_unreliable" not in beat.flags
    )
    if raw is not None:
        return raw, "reliable_beat_median_fallback"
    return None, "unavailable"


def _record_metrics(result) -> tuple[Dict[str, Optional[float]], str]:
    p_duration, p_source = record_p_duration_ms(result)
    global_features = result.global_features
    return (
        {
            "p_duration": p_duration,
            "pr_interval": _finite_or_none(global_features.pr_ms),
            "qrs_duration": _finite_or_none(global_features.qrs_ms),
            "qt_interval": _finite_or_none(global_features.qt_ms),
        },
        p_source,
    )


def evaluate_record(record_id: str) -> Dict[str, object]:
    """Run ecgfeat and construct LUDB annotation references for one record."""
    try:
        ecg, fs = load_record(record_id)
        header = parse_ludb_header(record_id)
        patient_meta = _build_patient_meta(header)

        extractor = ECGFeatureExtractor(
            fs_internal=REPORT_INTERNAL_FS,
            mains_freq=REPORT_MAINS_FREQ,
        )
        algorithm_result = extractor.extract(ecg, float(fs), meta=patient_meta)
        gt_by_lead = load_all_gt_annotations(
            record_id,
            int(fs),
            REPORT_INTERNAL_FS,
        )
        gt_result = build_annotation_derived_result(
            ecg,
            int(fs),
            gt_by_lead,
            preferred_lead="II",
            fs_internal=REPORT_INTERNAL_FS,
            patient_meta=patient_meta,
        )

        algorithm_metrics, algorithm_p_source = _record_metrics(algorithm_result)
        reference_metrics, reference_p_source = _record_metrics(gt_result)
        record_quality = algorithm_result.metadata.get("record_quality", {})

        row: Dict[str, object] = {
            "record_id": record_id,
            "error": None,
            "record_grade": (
                record_quality.get("record_grade")
                if isinstance(record_quality, dict)
                else None
            ),
            "algorithm_beats": len(algorithm_result.beats),
            "reference_beats": len(gt_result.beats),
            "algorithm_p_duration_source": algorithm_p_source,
            "reference_p_duration_source": reference_p_source,
            "algorithm_qt_source": algorithm_result.global_features.qt_source,
        }
        for metric in CSE_LIMITS:
            algorithm_value = algorithm_metrics[metric]
            reference_value = reference_metrics[metric]
            difference = (
                algorithm_value - reference_value
                if algorithm_value is not None and reference_value is not None
                else None
            )
            row[f"algorithm_{metric}_ms"] = algorithm_value
            row[f"reference_{metric}_ms"] = reference_value
            row[f"diff_{metric}_ms"] = difference
        return row
    except Exception as exc:
        return {
            "record_id": record_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "n": 0,
            "mean_ms": None,
            "sd_ms": None,
            "mae_ms": None,
            "median_abs_ms": None,
            "p95_abs_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    absolute = np.abs(array)
    return {
        "n": int(array.size),
        "mean_ms": float(np.mean(array)),
        "sd_ms": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "mae_ms": float(np.mean(absolute)),
        "median_abs_ms": float(np.median(absolute)),
        "p95_abs_ms": float(np.percentile(absolute, 95)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def trim_largest_deviations(
    record_values: Sequence[tuple[str, float]],
    outlier_count: int = DEFAULT_OUTLIER_COUNT,
) -> tuple[List[tuple[str, float]], List[tuple[str, float]]]:
    """Remove values farthest from the initial arithmetic mean."""
    if not record_values or outlier_count <= 0:
        return list(record_values), []
    values = np.asarray([value for _, value in record_values], dtype=float)
    initial_mean = float(np.mean(values))
    ranked = sorted(
        range(len(record_values)),
        key=lambda index: (
            abs(record_values[index][1] - initial_mean),
            abs(record_values[index][1]),
            record_values[index][0],
        ),
        reverse=True,
    )
    remove_indices = set(ranked[: min(outlier_count, len(record_values))])
    kept = [
        item for index, item in enumerate(record_values)
        if index not in remove_indices
    ]
    removed = [
        record_values[index] for index in ranked
        if index in remove_indices
    ]
    return kept, removed


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    outlier_count: int = DEFAULT_OUTLIER_COUNT,
) -> Dict[str, object]:
    successful_rows = [row for row in rows if not row.get("error")]
    failed_rows = [row for row in rows if row.get("error")]
    metrics: Dict[str, object] = {}

    for metric, limit in CSE_LIMITS.items():
        algorithm_field = f"algorithm_{metric}_ms"
        reference_field = f"reference_{metric}_ms"
        difference_field = f"diff_{metric}_ms"

        reference_available = sum(
            _finite_or_none(row.get(reference_field)) is not None
            for row in successful_rows
        )
        algorithm_available = sum(
            _finite_or_none(row.get(algorithm_field)) is not None
            for row in successful_rows
        )
        paired = [
            (str(row["record_id"]), difference)
            for row in successful_rows
            if (difference := _finite_or_none(row.get(difference_field))) is not None
        ]
        kept, removed = trim_largest_deviations(paired, outlier_count)
        algorithm_values = [
            value
            for row in successful_rows
            if (value := _finite_or_none(row.get(algorithm_field))) is not None
        ]
        reference_values = [
            value
            for row in successful_rows
            if (value := _finite_or_none(row.get(reference_field))) is not None
        ]
        raw_stats = _stats([value for _, value in paired])
        trimmed_stats = _stats([value for _, value in kept])

        mean_value = trimmed_stats["mean_ms"]
        sd_value = trimmed_stats["sd_ms"]
        mean_pass = (
            mean_value is not None
            and abs(float(mean_value)) <= limit.mean_limit_ms
        )
        sd_pass = (
            sd_value is not None
            and float(sd_value) <= limit.sd_limit_ms
        )
        coverage = (
            float(len(paired) / reference_available)
            if reference_available
            else None
        )
        metrics[metric] = {
            "label": limit.label,
            "mean_limit_ms": limit.mean_limit_ms,
            "sd_limit_ms": limit.sd_limit_ms,
            "successful_records": len(successful_rows),
            "reference_available": reference_available,
            "algorithm_available": algorithm_available,
            "paired_records": len(paired),
            "paired_coverage": coverage,
            "algorithm_values": _stats(algorithm_values),
            "reference_values": _stats(reference_values),
            "raw": raw_stats,
            "trimmed": trimmed_stats,
            "outliers_removed": [
                {"record_id": record_id, "difference_ms": difference}
                for record_id, difference in removed
            ],
            "mean_pass": mean_pass,
            "sd_pass": sd_pass,
            "numeric_pass": bool(mean_pass and sd_pass),
            "complete_on_reference_available": len(paired) == reference_available,
        }

    numeric_pass = all(
        bool(metric_summary["numeric_pass"])
        for metric_summary in metrics.values()
    )
    complete = all(
        bool(metric_summary["complete_on_reference_available"])
        for metric_summary in metrics.values()
    )
    return {
        "schema_version": "ecgfeat_ludb_cse_proxy.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "dataset": "LUDB 1.0.1",
            "difference_definition": "algorithm minus LUDB annotation-derived reference",
            "record_aggregation": (
                "ecgfeat native global PR/QRS/QT; P duration is the median "
                "representative multilead-consensus duration with a reliable-beat "
                "median fallback"
            ),
            "standard_deviation": "sample SD (ddof=1)",
            "outlier_count": outlier_count,
            "outlier_rule": (
                f"remove the {outlier_count} differences farthest from each "
                "metric's initial mean"
            ),
            "formal_conformance": False,
            "formal_conformance_reason": (
                "LUDB is not the prescribed CSE biological ECG test set and its "
                "global references are reconstructed from per-lead annotations"
            ),
        },
        "requested_records": len(rows),
        "successful_records": len(successful_rows),
        "failed_records": [
            {
                "record_id": str(row.get("record_id", "")),
                "error": str(row.get("error", "")),
            }
            for row in failed_rows
        ],
        "metrics": metrics,
        "all_metrics_numeric_pass": numeric_pass,
        "all_metrics_complete": complete,
        "proxy_verdict": (
            "meets_numeric_limits_with_complete_coverage"
            if numeric_pass and complete
            else "meets_numeric_limits_with_incomplete_coverage"
            if numeric_pass
            else "does_not_meet_numeric_limits"
        ),
    }


def _format_number(value: object, digits: int = 2) -> str:
    number = _finite_or_none(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def render_markdown(summary: Mapping[str, object]) -> str:
    metrics = summary["metrics"]
    outlier_count = int(summary["method"]["outlier_count"])
    verdict = str(summary["proxy_verdict"])
    if verdict == "does_not_meet_numeric_limits":
        conclusion = "未达到全部 CSE 数值限值。"
    elif verdict == "meets_numeric_limits_with_incomplete_coverage":
        conclusion = "已达到数值限值，但测量覆盖不完整。"
    else:
        conclusion = "四项数值限值均达到，且对可用参考标注覆盖完整。"

    lines = [
        "# ecgfeat 在 LUDB 上的 CSE 间期限值代理评估",
        "",
        f"生成时间：{summary['generated_at_utc']}",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        (
            "这不是正式 CSE/IEC 符合性结论：测试数据是 LUDB，而不是标准指定的 "
            "CSE 生物心电图；LUDB 的全局参考值由逐导联专家边界重建。"
        ),
        "",
        "差值定义为 `ecgfeat − LUDB参考`。CSE 风格结果按每个指标初始均值，"
        f"剔除偏离最大的 {outlier_count} 条记录后计算；SD 使用样本标准差。",
        "",
        "## 汇总",
        "",
        "| 指标 | 配对数/参考可用 | 覆盖率 | 原始均差 | 原始SD | 剔除后均差/限值 | 剔除后SD/限值 | 结果 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in CSE_LIMITS:
        item = metrics[metric]
        raw = item["raw"]
        trimmed = item["trimmed"]
        coverage = item["paired_coverage"]
        coverage_text = (
            "N/A" if coverage is None else f"{100.0 * float(coverage):.1f}%"
        )
        failed_parts: List[str] = []
        if not item["mean_pass"]:
            failed_parts.append("均差")
        if not item["sd_pass"]:
            failed_parts.append("SD")
        if not item["complete_on_reference_available"]:
            failed_parts.append("覆盖")
        status = "通过" if not failed_parts else "未通过（" + "、".join(failed_parts) + "）"
        lines.append(
            f"| {item['label']} | {item['paired_records']}/{item['reference_available']} "
            f"| {coverage_text} | {_format_number(raw['mean_ms'])} ms "
            f"| {_format_number(raw['sd_ms'])} ms "
            f"| {_format_number(trimmed['mean_ms'])} / ±{item['mean_limit_ms']:.0f} ms "
            f"| {_format_number(trimmed['sd_ms'])} / {item['sd_limit_ms']:.0f} ms "
            f"| {status} |"
        )

    lines.extend(
        [
            "",
            "## 误差分布与剔除记录",
            "",
        ]
    )
    for metric in CSE_LIMITS:
        item = metrics[metric]
        raw = item["raw"]
        removed = ", ".join(
            f"{entry['record_id']} ({entry['difference_ms']:+.1f} ms)"
            for entry in item["outliers_removed"]
        ) or "无"
        lines.extend(
            [
                f"### {item['label']}",
                "",
                (
                    f"- 原始 MAE {_format_number(raw['mae_ms'])} ms；"
                    f"中位绝对误差 {_format_number(raw['median_abs_ms'])} ms；"
                    f"P95 绝对误差 {_format_number(raw['p95_abs_ms'])} ms。"
                ),
                f"- 剔除记录：{removed}。",
                "",
            ]
        )

    failed_records = summary["failed_records"]
    lines.extend(
        [
            "## 数据完整性",
            "",
            (
                f"- 请求 {summary['requested_records']} 条，成功 "
                f"{summary['successful_records']} 条，运行失败 {len(failed_records)} 条。"
            ),
        ]
    )
    if failed_records:
        lines.append(
            "- 失败记录："
            + "；".join(
                f"{entry['record_id']}: {entry['error']}"
                for entry in failed_records
            )
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            (
                "- CSE 的正式测试使用指定的 100 份生物心电图及官方全局参考值；"
                "本结果只能说明 ecgfeat 在 LUDB 上是否达到相同的均差/SD数值门槛。"
            ),
            (
                "- LUDB 的 P、QRS、T 边界是逐拍逐导联标注。PR、QRS、QT 的参考值"
                "由仓库现有 annotation-only 全局汇总链路产生；P 时限由多导联一致性"
                "时限的记录中位数产生。"
            ),
            (
                f"- 剔除 {outlier_count} 个离群值后的结果用于贴近标准统计流程；"
                "原始结果同时保留，"
                "避免离群值掩盖真实工程风险。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "record_measurements.csv"
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_summary_rows: List[Dict[str, object]] = []
    for metric, item in summary["metrics"].items():
        metric_summary_rows.append(
            {
                "metric": metric,
                "label": item["label"],
                "paired_records": item["paired_records"],
                "reference_available": item["reference_available"],
                "paired_coverage": item["paired_coverage"],
                "algorithm_mean_ms": item["algorithm_values"]["mean_ms"],
                "reference_mean_ms": item["reference_values"]["mean_ms"],
                "raw_mean_difference_ms": item["raw"]["mean_ms"],
                "raw_sd_ms": item["raw"]["sd_ms"],
                "raw_mae_ms": item["raw"]["mae_ms"],
                "trimmed_n": item["trimmed"]["n"],
                "trimmed_mean_difference_ms": item["trimmed"]["mean_ms"],
                "mean_limit_ms": item["mean_limit_ms"],
                "mean_pass": item["mean_pass"],
                "trimmed_sd_ms": item["trimmed"]["sd_ms"],
                "sd_limit_ms": item["sd_limit_ms"],
                "sd_pass": item["sd_pass"],
                "numeric_pass": item["numeric_pass"],
                "complete_on_reference_available": item[
                    "complete_on_reference_available"
                ],
            }
        )
    summary_csv_path = output_dir / "metrics_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(metric_summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(metric_summary_rows)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )


def _record_sort_key(record_id: str) -> tuple[int, object]:
    return (0, int(record_id)) if record_id.isdigit() else (1, record_id)


def run_evaluation(
    record_ids: Sequence[str],
    workers: int,
) -> List[Dict[str, object]]:
    worker_count = max(1, min(int(workers), len(record_ids)))
    if worker_count == 1:
        rows = []
        for index, record_id in enumerate(record_ids, 1):
            print(f"[{index:>3}/{len(record_ids)}] {record_id}", flush=True)
            rows.append(evaluate_record(record_id))
        return rows

    rows_by_id: Dict[str, Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(evaluate_record, record_id): record_id
            for record_id in record_ids
        }
        completed = 0
        for future in as_completed(futures):
            record_id = futures[future]
            completed += 1
            try:
                rows_by_id[record_id] = future.result()
            except Exception as exc:
                rows_by_id[record_id] = {
                    "record_id": record_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if completed == 1 or completed % 10 == 0 or completed == len(record_ids):
                print(
                    f"[{completed:>3}/{len(record_ids)}] completed",
                    flush=True,
                )
    return [
        rows_by_id[record_id]
        for record_id in sorted(rows_by_id, key=_record_sort_key)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ecgfeat on LUDB against CSE-style interval limits."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.name}).",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        default=None,
        help="Explicit LUDB record IDs. Default: all records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N selected records.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, int(os.cpu_count() or 1)),
        help="Parallel worker processes (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--outlier-count",
        type=int,
        default=DEFAULT_OUTLIER_COUNT,
        help="Number of largest deviations to remove per metric (default: 8).",
    )
    args = parser.parse_args()

    record_ids = list(args.records or discover_record_ids())
    if args.limit is not None:
        record_ids = record_ids[: max(0, args.limit)]
    if not record_ids:
        parser.error("No LUDB records selected or discovered.")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.outlier_count < 0:
        parser.error("--outlier-count must not be negative")

    print(
        f"Evaluating {len(record_ids)} LUDB records with "
        f"{min(args.workers, len(record_ids))} worker(s)."
    )
    rows = run_evaluation(record_ids, args.workers)
    summary = summarize_rows(rows, outlier_count=args.outlier_count)
    write_outputs(rows, summary, args.out_dir.resolve())
    print(render_markdown(summary))
    print(f"Outputs: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
