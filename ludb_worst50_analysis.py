from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


FAILURE_INTERPRETATIONS = {
    "qt_t_repolarization_failure": (
        "QT/T wave failure: few reliable QT leads, T-axis reversal, "
        "ischemia/scar/repolarization changes, or unstable T-end."
    ),
    "wide_qrs_conduction": (
        "Wide-QRS/conduction delay: QRS offset and T onset/end are hard to separate."
    ),
    "rhythm_morphology_mismatch": (
        "Mixed rhythm or morphology: representative beat selection is unstable."
    ),
    "p_pr_failure": (
        "P/PR failure: P wave missing, PR not clinically comparable, or PR error is large."
    ),
    "low_quality_or_missing_support": (
        "Algorithm support is weak: missing global metrics, many unreliable entries, "
        "BAD leads, or too few QT leads."
    ),
    "paced_av_block": (
        "Paced/III-degree AV-block rhythm: P/PR often unavailable, "
        "QRS/T boundaries are abnormal."
    ),
}


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_patient_field(report_text: str, label: str) -> str:
    lines = report_text.splitlines()
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(.*)$")
    next_field = re.compile(r"^\s*[A-Za-z][A-Za-z /-]{0,18}\s+:")
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        parts = [match.group(1).strip()]
        for nxt in lines[index + 1 :]:
            if next_field.match(nxt) or nxt.strip().startswith("───"):
                break
            if nxt.strip():
                parts.append(nxt.strip())
        return _clean_spaces(" ".join(parts))
    return ""


def _diagnosis_terms(diagnosis: str) -> list[str]:
    return [_clean_spaces(term) for term in diagnosis.split(";") if _clean_spaces(term)]


def _categorize(terms: Iterable[str]) -> list[str]:
    joined = " | ".join(terms).lower()
    categories: list[str] = []
    if "pacing" in joined:
        categories.append("pacing")
    if "av block" in joined or "av-block" in joined:
        categories.append("av_block")
    if "atrial fibrillation" in joined or "atrial flutter" in joined:
        categories.append("af_flutter")
    if "extrasystole" in joined or "pvc" in joined:
        categories.append("extrasystole_pvc")
    if any(
        token in joined
        for token in [
            "bundle branch",
            "hemiblock",
            "intraventricular",
            "intravintricular",
            "aberrant conduction",
            "ivcd",
        ]
    ):
        categories.append("bundle_branch_or_ivcd")
    if any(
        token in joined
        for token in [
            "ventricular hypertrophy",
            "ventricular overload",
            "right ventricular hypertrophy",
            "left ventricular hypertrophy",
            "left ventricular overload",
        ]
    ):
        categories.append("hypertrophy_or_overload")
    if "atrial hypertrophy" in joined or "atrial overload" in joined:
        categories.append("atrial_hypertrophy_or_overload")
    if any(token in joined for token in ["ischemia", "scar", "stemi", "nstemi"]):
        categories.append("ischemia_or_scar")
    if "repolarization" in joined:
        categories.append("repolarization_abnormality")
    if "bradycardia" in joined:
        categories.append("bradycardia")
    return categories


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "" or value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _int(row: dict[str, str], key: str) -> int:
    value = _float(row, key)
    return int(value) if value is not None else 0


def _parse_reliable_qt(algorithm_report: Path) -> tuple[str, int]:
    if not algorithm_report.exists():
        return "unknown", 0
    text = algorithm_report.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Reliable-QT leads\s*:\s*(.+)", text)
    if not match:
        return "unknown", 0
    leads = _clean_spaces(match.group(1))
    if not leads or leads.lower() == "none":
        return "none", 0
    parts = [part.strip() for part in leads.split(",") if part.strip()]
    return ", ".join(parts), len(parts)


def _count_bad_leads(algorithm_report: Path) -> int:
    if not algorithm_report.exists():
        return 0
    count = 0
    in_quality = False
    for line in algorithm_report.read_text(encoding="utf-8", errors="replace").splitlines():
        if "SIGNAL QUALITY" in line:
            in_quality = True
            continue
        if in_quality and "PER-LEAD MEASUREMENTS" in line:
            break
        if in_quality and "BAD" in line:
            count += 1
    return count


def _is_paced_or_high_av(row: dict[str, str]) -> bool:
    diagnosis = row.get("diagnosis", "").lower()
    categories = set(filter(None, row.get("diagnosis_categories", "").split(" | ")))
    return (
        "pacing" in categories
        or "iii degree av" in diagnosis
        or "iii-degree av" in diagnosis
        or "3 degree av" in diagnosis
    )


def _has_any_av_block(row: dict[str, str]) -> bool:
    diagnosis = row.get("diagnosis", "").lower()
    return "av block" in diagnosis or "av-block" in diagnosis


def _failure_buckets(row: dict[str, str]) -> list[str]:
    categories = set(filter(None, row.get("diagnosis_categories", "").split(" | ")))
    diagnosis = row.get("diagnosis", "").lower()
    buckets: set[str] = set()
    paced_high = _is_paced_or_high_av(row)
    rhythm_mixed = bool(categories & {"af_flutter", "extrasystole_pvc", "pacing"}) or any(
        token in diagnosis
        for token in ["wandering atrial pacemaker", "sinus arrhythmia", "aberrant conduction"]
    )
    conduction = "bundle_branch_or_ivcd" in categories or paced_high
    ischemic_repol = bool(categories & {"ischemia_or_scar", "repolarization_abnormality"})
    gt_qrs = _float(row, "gt_qrs")
    alg_qrs = _float(row, "algorithm_qrs")
    reliable_qt_count = _int(row, "algorithm_reliable_qt_lead_count")

    if paced_high:
        buckets.add("paced_av_block")
    if (
        conduction
        or (gt_qrs is not None and gt_qrs >= 120)
        or (alg_qrs is not None and alg_qrs >= 130)
        or (_float(row, "abs_qrs_diff") or 0) >= 25
    ):
        buckets.add("wide_qrs_conduction")
    if rhythm_mixed or abs(_int(row, "beat_diff")) >= 3:
        buckets.add("rhythm_morphology_mismatch")
    if (
        _has_any_av_block(row)
        or "af_flutter" in categories
        or (_float(row, "abs_pr_diff") or 0) >= 30
        or (row.get("algorithm_pr") and not row.get("gt_pr"))
        or _int(row, "algorithm_p_wave_undetected") >= 20
    ):
        buckets.add("p_pr_failure")
    if (
        ischemic_repol
        or paced_high
        or reliable_qt_count <= 2
        or (_float(row, "abs_qt_diff") or 0) >= 40
        or (_float(row, "abs_qtcb_diff") or 0) >= 40
        or (_float(row, "abs_qtcf_diff") or 0) >= 40
        or (_float(row, "abs_t_axis_circular_diff") or 0) >= 45
        or not row.get("algorithm_qt")
    ):
        buckets.add("qt_t_repolarization_failure")
    if (
        _int(row, "algorithm_metric_na_count") > 0
        or _int(row, "algorithm_unreliable_entries") >= 20
        or reliable_qt_count <= 1
        or _int(row, "algorithm_bad_leads") > 0
    ):
        buckets.add("low_quality_or_missing_support")

    order = [
        "paced_av_block",
        "wide_qrs_conduction",
        "rhythm_morphology_mismatch",
        "p_pr_failure",
        "qt_t_repolarization_failure",
        "low_quality_or_missing_support",
    ]
    return [bucket for bucket in order if bucket in buckets]


def _primary_reason(buckets: list[str]) -> str:
    if "paced_av_block" in buckets:
        return "paced/III-degree AV block rhythm: P-PR often unavailable, QRS/T boundaries are abnormal"
    if "wide_qrs_conduction" in buckets:
        return "wide QRS/conduction delay: QRS offset and T onset/end are hard to separate"
    if "rhythm_morphology_mismatch" in buckets:
        return "mixed rhythm or morphology: representative beat selection is unstable"
    if "qt_t_repolarization_failure" in buckets:
        return "QT/T repolarization failure: T polarity/end or reliable QT leads are weak"
    if "p_pr_failure" in buckets:
        return "P/PR failure: P wave or PR interval is not robust"
    if "low_quality_or_missing_support" in buckets:
        return "low algorithm support: missing metrics or unreliable entries"
    return "metric error outside tolerance"


def _display(row: dict[str, str], key: str) -> str:
    value = _float(row, key)
    return "NA" if value is None else f"{value:.1f}"


def _stat(rows: list[dict[str, str]], field: str) -> str:
    values = sorted(value for row in rows if (value := _float(row, field)) is not None)
    if not values:
        return "NA"
    median = (
        values[len(values) // 2]
        if len(values) % 2
        else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
    )
    return f"mean {sum(values) / len(values):.1f}, median {median:.1f}, max {max(values):.1f}"


def build_analysis(compare_dir: Path) -> tuple[Path, Path, Path]:
    ranked_path = compare_dir / "summary_ranked.csv"
    ranked_rows = list(csv.DictReader(ranked_path.open(newline="", encoding="utf-8")))
    worst_rows = [row for row in ranked_rows if int(row["rank"]) >= 151]

    analysis_rows: list[dict[str, str]] = []
    for row in worst_rows:
        rec = row["record_id"]
        gt_report = compare_dir / rec / f"{rec}_ground_truth_report.txt"
        alg_report = compare_dir / rec / f"{rec}_algorithm_report.txt"
        comparison = compare_dir / rec / f"{rec}_comparison.txt"
        diagnosis = (
            _parse_patient_field(
                gt_report.read_text(encoding="utf-8", errors="replace"), "Dx Detail"
            )
            if gt_report.exists()
            else ""
        )
        terms = _diagnosis_terms(diagnosis)
        reliable_leads, reliable_count = _parse_reliable_qt(alg_report)
        out = {
            "rank": row["rank"],
            "record_id": rec,
            "rank_score": row["rank_score"],
            "diagnosis": diagnosis,
            "diagnosis_terms": " | ".join(terms),
            "diagnosis_categories": " | ".join(_categorize(terms)),
            "algorithm_reliable_qt_leads": reliable_leads,
            "algorithm_reliable_qt_lead_count": str(reliable_count),
            "algorithm_bad_leads": str(_count_bad_leads(alg_report)),
            "comparison_path": str(comparison.resolve()),
            "ground_truth_report_path": str(gt_report.resolve()),
            "algorithm_report_path": str(alg_report.resolve()),
        }
        out.update(
            {
                "algorithm_beats": row.get("algorithm_beats", ""),
                "ground_truth_beats": row.get("ground_truth_beats", ""),
                "beat_diff": row.get("beat_diff", ""),
                "algorithm_metric_na_count": row.get("algorithm_metric_na_count", ""),
                "ground_truth_metric_na_count": row.get("ground_truth_metric_na_count", ""),
                "abs_pr_diff": row.get("abs_pr_diff", ""),
                "abs_qrs_diff": row.get("abs_qrs_diff", ""),
                "abs_qt_diff": row.get("abs_qt_diff", ""),
                "abs_qtcb_diff": row.get("abs_qtcb_diff", ""),
                "abs_qtcf_diff": row.get("abs_qtcf_diff", ""),
                "abs_p_axis_circular_diff": row.get("abs_p_axis_circular_diff", ""),
                "abs_qrs_axis_circular_diff": row.get("abs_qrs_axis_circular_diff", ""),
                "abs_t_axis_circular_diff": row.get("abs_t_axis_circular_diff", ""),
                "diff_pr": row.get("diff_pr", ""),
                "diff_qrs": row.get("diff_qrs", ""),
                "diff_qt": row.get("diff_qt", ""),
                "diff_qtcb": row.get("diff_qtcb", ""),
                "diff_t_axis": row.get("diff_t_axis", ""),
                "gt_hr": row.get("ground_truth_hr", ""),
                "gt_pr": row.get("ground_truth_pr", ""),
                "gt_qrs": row.get("ground_truth_qrs", ""),
                "gt_qt": row.get("ground_truth_qt", ""),
                "gt_qtcb": row.get("ground_truth_qtcb", ""),
                "gt_p_axis": row.get("ground_truth_p_axis", ""),
                "gt_qrs_axis": row.get("ground_truth_qrs_axis", ""),
                "gt_t_axis": row.get("ground_truth_t_axis", ""),
                "algorithm_hr": row.get("algorithm_hr", ""),
                "algorithm_pr": row.get("algorithm_pr", ""),
                "algorithm_qrs": row.get("algorithm_qrs", ""),
                "algorithm_qt": row.get("algorithm_qt", ""),
                "algorithm_qtcb": row.get("algorithm_qtcb", ""),
                "algorithm_p_axis": row.get("algorithm_p_axis", ""),
                "algorithm_qrs_axis": row.get("algorithm_qrs_axis", ""),
                "algorithm_t_axis": row.get("algorithm_t_axis", ""),
                "algorithm_p_wave_undetected": row.get("algorithm_p_missing", ""),
                "algorithm_t_wave_undetected": row.get("algorithm_t_missing", ""),
                "algorithm_qrs_bounds_failed": row.get("algorithm_qrs_missing", ""),
                "algorithm_unreliable_entries": row.get("algorithm_unreliable", ""),
            }
        )
        buckets = _failure_buckets(out)
        out["failure_buckets"] = " | ".join(buckets)
        out["primary_failure_reason"] = _primary_reason(buckets)
        analysis_rows.append(out)

    csv_path = compare_dir / "worst50_diagnosis_failure_analysis.csv"
    fieldnames = [
        "rank",
        "record_id",
        "rank_score",
        "diagnosis",
        "diagnosis_terms",
        "diagnosis_categories",
        "failure_buckets",
        "primary_failure_reason",
        "algorithm_beats",
        "ground_truth_beats",
        "beat_diff",
        "algorithm_metric_na_count",
        "ground_truth_metric_na_count",
        "abs_pr_diff",
        "abs_qrs_diff",
        "abs_qt_diff",
        "abs_qtcb_diff",
        "abs_qtcf_diff",
        "abs_p_axis_circular_diff",
        "abs_qrs_axis_circular_diff",
        "abs_t_axis_circular_diff",
        "diff_pr",
        "diff_qrs",
        "diff_qt",
        "diff_qtcb",
        "diff_t_axis",
        "gt_hr",
        "gt_pr",
        "gt_qrs",
        "gt_qt",
        "gt_qtcb",
        "gt_p_axis",
        "gt_qrs_axis",
        "gt_t_axis",
        "algorithm_hr",
        "algorithm_pr",
        "algorithm_qrs",
        "algorithm_qt",
        "algorithm_qtcb",
        "algorithm_p_axis",
        "algorithm_qrs_axis",
        "algorithm_t_axis",
        "algorithm_reliable_qt_leads",
        "algorithm_reliable_qt_lead_count",
        "algorithm_p_wave_undetected",
        "algorithm_t_wave_undetected",
        "algorithm_qrs_bounds_failed",
        "algorithm_unreliable_entries",
        "algorithm_bad_leads",
        "comparison_path",
        "ground_truth_report_path",
        "algorithm_report_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)

    summary_path = compare_dir / "worst50_failure_summary.md"
    focused_path = compare_dir / "worst_records_except_74_analysis.md"
    _write_summary(summary_path, analysis_rows, ranked_rows, compare_dir)
    _write_focused(focused_path, analysis_rows, ranked_rows, compare_dir)
    return csv_path, summary_path, focused_path


def _rank_tail_status(rank: str) -> str:
    try:
        return "inside" if int(rank) >= 151 else "outside"
    except (TypeError, ValueError):
        return "unknown relative to"


def _write_summary(
    path: Path,
    rows: list[dict[str, str]],
    ranked_rows: list[dict[str, str]],
    compare_dir: Path,
) -> None:
    category_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    bucket_records: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        category_counter.update(filter(None, row["diagnosis_categories"].split(" | ")))
        label_counter.update(filter(None, row["diagnosis_terms"].split(" | ")))
        buckets = list(filter(None, row["failure_buckets"].split(" | ")))
        bucket_counter.update(buckets)
        for bucket in buckets:
            bucket_records[bucket].append(row["record_id"])

    rank74 = next((row["rank"] for row in ranked_rows if row["record_id"] == "74"), "NA")
    rank13 = next((row["rank"] for row in ranked_rows if row["record_id"] == "13"), "NA")
    worst_row = max(ranked_rows, key=lambda row: int(row["rank"])) if ranked_rows else None
    worst_id = worst_row["record_id"] if worst_row else "NA"
    wide_gt_count = sum(1 for row in rows if (_float(row, "gt_qrs") or 0) >= 120)
    high_av_or_paced = sum(1 for row in rows if _is_paced_or_high_av(row))
    bad_any = sum(1 for row in rows if _int(row, "algorithm_bad_leads") > 0)

    lines = [
        "# LUDB Worst-50 Diagnosis And Failure Analysis",
        "",
        f"Source ranking: `{compare_dir.as_posix()}/summary_ranked.csv`. "
        "Records included: ranks 151-200 after the latest full rerun.",
        "",
        "## Main Takeaways",
        "",
        f"- The current worst record after the latest rerun is {worst_id} (rank 200 of 200).",
        f"- Record 74 ranks {rank74} of 200 ({_rank_tail_status(rank74)} the worst-50 tail); "
        f"record 13 ranks {rank13} of 200 ({_rank_tail_status(rank13)} the worst-50 tail).",
        f"- Wide QRS / conduction-related cases remain common: {wide_gt_count} of 50 have ground-truth QRS >= 120 ms.",
        f"- Pacing remains concentrated in the tail: {category_counter['pacing']} of 50 worst records have pacing labels.",
        f"- Pacing or III-degree AV block cases: {high_av_or_paced} of 50 worst records.",
        f"- Signal quality alone is still not the main driver: only {bad_any} of 50 have any BAD lead in the algorithm report.",
        "",
        "## Error Summary",
        "",
        "| Metric | Worst-50 statistic |",
        "| --- | --- |",
    ]
    for label, field in [
        ("abs PR error ms", "abs_pr_diff"),
        ("abs QRS error ms", "abs_qrs_diff"),
        ("abs QT error ms", "abs_qt_diff"),
        ("abs QTcB error ms", "abs_qtcb_diff"),
        ("abs T-axis circular error deg", "abs_t_axis_circular_diff"),
        ("algorithm unreliable entries", "algorithm_unreliable_entries"),
        ("algorithm P-wave undetected", "algorithm_p_wave_undetected"),
        ("algorithm reliable-QT lead count", "algorithm_reliable_qt_lead_count"),
    ]:
        lines.append(f"| {label} | {_stat(rows, field)} |")

    lines.extend(["", "## Diagnosis Categories", "", "| Category | Count |", "| --- | ---: |"])
    for category, count in category_counter.most_common():
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Exact Labels", "", "| Label | Count |", "| --- | ---: |"])
    for label, count in label_counter.most_common(25):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Failure Buckets",
            "",
            "| Bucket | Count | Records | Interpretation |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for bucket, count in bucket_counter.most_common():
        lines.append(
            f"| {bucket} | {count} | {', '.join(bucket_records[bucket])} | "
            f"{FAILURE_INTERPRETATIONS[bucket]} |"
        )
    lines.extend(
        [
            "",
            "## Worst 50 Records",
            "",
            "| Rank | Record | Buckets | Main reason | Key errors | Diagnosis |",
            "| ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        errors = "; ".join(
            [
                f"beat {row['beat_diff']}",
                f"PR {_display(row, 'abs_pr_diff')}",
                f"QRS {_display(row, 'abs_qrs_diff')}",
                f"QT {_display(row, 'abs_qt_diff')}",
                f"Taxis {_display(row, 'abs_t_axis_circular_diff')}",
                f"QTleads {row['algorithm_reliable_qt_lead_count']}",
            ]
        )
        lines.append(
            f"| {row['rank']} | {row['record_id']} | "
            f"{row['failure_buckets'].replace(' | ', ', ')} | "
            f"{row['primary_failure_reason']} | {errors} | {row['diagnosis']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_focused(
    path: Path,
    rows: list[dict[str, str]],
    ranked_rows: list[dict[str, str]],
    compare_dir: Path,
) -> None:
    rank74 = next((row["rank"] for row in ranked_rows if row["record_id"] == "74"), "NA")
    focus = [
        row
        for row in sorted(rows, key=lambda item: int(item["rank"]), reverse=True)
        if row["record_id"] != "74"
    ][:12]
    lines = [
        "# Worst Records Excluding Record 74: Latest Rerun Analysis",
        "",
        "Source files:",
        f"- Ranking: `{compare_dir.as_posix()}/summary_ranked.csv`",
        f"- Diagnosis/failure CSV: `{compare_dir.as_posix()}/worst50_diagnosis_failure_analysis.csv`",
        f"- Metric details: `{compare_dir.as_posix()}/<record>/<record>_comparison.txt`",
        "",
        "## Executive Summary",
        "",
        f"This view excludes record 74 (currently rank {rank74}/200). Across the rest of the tail, the worst failures are dominated by QT/T and conduction/pacing morphology.",
        "",
        "| Failure family | Representative records | What goes wrong |",
        "| --- | --- | --- |",
        "| Pacing / III-degree AV block | 104, 111, 90, 34, 45, 95 | PR/P axis should often be unavailable, QRS is paced/wide, and T-end or T-axis is unstable after paced depolarization. |",
        "| Wide QRS / bundle branch block / IVCD | 13, 116, 23, 65, 24, 83, 121 | QRS offset and early ST/T remain hard to separate, so QRS width, QT, and T polarity can shift together. |",
        "| AF/AFL/PVC or mixed morphology | 38, 35, 51, 83, 95, 109, 125, 127 | A single representative beat can mix different morphologies; PR/P axis can be clinically non-comparable. |",
        "| Ischemia / scar / repolarization abnormality | 13, 23, 57, 73, 81, 124, 133 | T polarity and T-end are abnormal or low-confidence; QT support is often sparse or inconsistent. |",
        "",
        "## Highest-Priority Records Besides 74",
        "",
        "| Rank | Record | True labels, condensed | Current major errors | Likely root cause |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    for row in focus:
        terms = [term.replace("Rhythm: ", "") for term in row["diagnosis_terms"].split(" | ") if term]
        condensed = "; ".join(terms[:5]) + ("; ..." if len(terms) > 5 else "")
        errors = ", ".join(
            [
                f"PR {_display(row, 'diff_pr')} ms",
                f"QRS {_display(row, 'diff_qrs')} ms",
                f"QT {_display(row, 'diff_qt')} ms",
                f"T-axis {_display(row, 'abs_t_axis_circular_diff')} deg",
                f"QTleads {row['algorithm_reliable_qt_lead_count']}",
            ]
        )
        lines.append(
            f"| {row['rank']} | {row['record_id']} | {condensed} | "
            f"{errors} | {row['primary_failure_reason']} |"
        )
    lines.extend(
        [
            "",
            "## What Changed Versus The Old Run",
            "",
            "- The tail is dominated by QT/T morphology, T-axis polarity, long-PR/P-onset, and paced/wide-QRS repolarization rather than pure beat detection.",
            "- Record 13 is a representative long-PR/wide-QRS case: HR is correct and QT is now available, but PR is unavailable (long-PR / AV block), QRS is too wide, and T-axis is not computed.",
            "",
            "## Suggested Next Improvement Order",
            "",
            "| Priority | Change | Records helped most |",
            "| ---: | --- | --- |",
            "| 1 | Add morphology-consistency gates for QT/T, rejecting leads where T polarity/end disagrees strongly with the lead cluster. | 13, 23, 57, 65, 73, 116 |",
            "| 2 | Compute T axis from signed T area with polarity clustering, analogous to the QRS signed-area fix. | 45, 23, 38, 57, 125 |",
            "| 3 | Add a wide-QRS / paced-QRS QT path using JT sanity checks and stricter minimum plausible QT/JT bounds. | 104, 111, 116, 23, 65, 83 |",
            "| 4 | Add rhythm-aware PR/P-axis suppression and long-PR P-onset support for sinus AVB1 records. | 13, 130, 104, 111, 51, 95 |",
            "| 5 | Use morphology-family medoids instead of a single averaged representative when PVC/pacing/AF/AFL morphology is mixed. | 35, 38, 45, 83, 95, 109, 125, 127 |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build LUDB worst-50 diagnosis and failure analysis from summary_ranked.csv."
    )
    parser.add_argument("compare_dir", nargs="?", default="ludb_full_compare")
    args = parser.parse_args()
    csv_path, summary_path, focused_path = build_analysis(Path(args.compare_dir))
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {focused_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
