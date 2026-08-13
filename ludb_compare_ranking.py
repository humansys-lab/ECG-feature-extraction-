from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_SPECS: Sequence[Tuple[str, str]] = (
    ("HR", "hr"),
    ("PR", "pr"),
    ("QRS", "qrs"),
    ("QT", "qt"),
    ("QTcB", "qtcb"),
    ("QTcF", "qtcf"),
    ("P Axis", "p_axis"),
    ("QRS Axis", "qrs_axis"),
    ("T Axis", "t_axis"),
    ("QT Dispersion", "qt_dispersion"),
)

ANGLE_FIELDS = {"p_axis", "qrs_axis", "t_axis"}

RANKING_TOLERANCES = {
    "abs_hr_diff": 2.0,
    "abs_pr_diff": 20.0,
    "abs_qrs_diff": 15.0,
    "abs_qt_diff": 20.0,
    "abs_qtcb_diff": 20.0,
    "abs_qtcf_diff": 20.0,
    "abs_p_axis_circular_diff": 15.0,
    "abs_qrs_axis_circular_diff": 10.0,
    "abs_t_axis_circular_diff": 20.0,
    "abs_qt_dispersion_diff": 10.0,
}

METRIC_LINE_RE = re.compile(
    r"^(HR|PR|QRS|QT|QTcB|QTcF|P Axis|QRS Axis|T Axis|QT Dispersion)"
    r"\s+([-0-9.]+|N/A)\s+([-0-9.]+|N/A)\s+([-0-9.]+|N/A)\s*$"
)
MISSING_SUMMARY_RE = re.compile(
    r"P=(?P<p>\d+)\s+T=(?P<t>\d+)\s+QRS=(?P<qrs>\d+)\s+Unreliable=(?P<unreliable>\d+)"
)


def _parse_float(token: str) -> Optional[float]:
    return None if token == "N/A" else float(token)


def circular_signed_diff(algorithm_value: float, ground_truth_value: float) -> float:
    raw_diff = algorithm_value - ground_truth_value
    return ((raw_diff + 180.0) % 360.0) - 180.0


def _format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return ""
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def parse_comparison_report(text: str, record_id: int) -> Dict[str, object]:
    row: Dict[str, object] = {"record_id": record_id}
    available_metric_count = 0
    algorithm_metric_na_count = 0
    ground_truth_metric_na_count = 0

    in_missing_summary = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = METRIC_LINE_RE.match(line)
        if match:
            metric_name, algorithm_token, gt_token, diff_token = match.groups()
            field_name = dict(METRIC_SPECS)[metric_name]
            algorithm_value = _parse_float(algorithm_token)
            ground_truth_value = _parse_float(gt_token)
            raw_diff = _parse_float(diff_token)

            row[f"algorithm_{field_name}"] = algorithm_value
            row[f"ground_truth_{field_name}"] = ground_truth_value
            row[f"diff_{field_name}"] = raw_diff

            if algorithm_value is None and ground_truth_value is not None:
                algorithm_metric_na_count += 1
            elif algorithm_value is not None and ground_truth_value is None:
                ground_truth_metric_na_count += 1
            elif algorithm_value is not None and ground_truth_value is not None:
                available_metric_count += 1

            if raw_diff is not None:
                row[f"abs_{field_name}_diff"] = abs(raw_diff)
            else:
                row[f"abs_{field_name}_diff"] = None

            if (
                field_name in ANGLE_FIELDS
                and algorithm_value is not None
                and ground_truth_value is not None
            ):
                circular_diff = circular_signed_diff(algorithm_value, ground_truth_value)
                row[f"raw_diff_{field_name}"] = raw_diff
                row[f"abs_{field_name}_raw_diff"] = abs(raw_diff) if raw_diff is not None else None
                row[f"circular_diff_{field_name}"] = circular_diff
                row[f"abs_{field_name}_circular_diff"] = abs(circular_diff)
            elif field_name in ANGLE_FIELDS:
                row[f"raw_diff_{field_name}"] = raw_diff
                row[f"abs_{field_name}_raw_diff"] = abs(raw_diff) if raw_diff is not None else None
                row[f"circular_diff_{field_name}"] = None
                row[f"abs_{field_name}_circular_diff"] = None

            continue

        if line == "Beat / Missing Summary":
            in_missing_summary = True
            continue

        if not in_missing_summary:
            continue

        if line.startswith("Algorithm beats"):
            row["algorithm_beats"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Ground-truth beats"):
            row["ground_truth_beats"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Algorithm missing"):
            missing_match = MISSING_SUMMARY_RE.search(line)
            if missing_match:
                row["algorithm_p_missing"] = int(missing_match.group("p"))
                row["algorithm_t_missing"] = int(missing_match.group("t"))
                row["algorithm_qrs_missing"] = int(missing_match.group("qrs"))
                row["algorithm_unreliable"] = int(missing_match.group("unreliable"))
        elif line.startswith("Ground-truth missing"):
            missing_match = MISSING_SUMMARY_RE.search(line)
            if missing_match:
                row["ground_truth_p_missing"] = int(missing_match.group("p"))
                row["ground_truth_t_missing"] = int(missing_match.group("t"))
                row["ground_truth_qrs_missing"] = int(missing_match.group("qrs"))
                row["ground_truth_unreliable"] = int(missing_match.group("unreliable"))

    algorithm_beats = int(row.get("algorithm_beats", 0))
    ground_truth_beats = int(row.get("ground_truth_beats", 0))
    row["beat_diff"] = algorithm_beats - ground_truth_beats
    row["abs_beat_diff"] = abs(row["beat_diff"])
    row["available_metric_count"] = available_metric_count
    row["algorithm_metric_na_count"] = algorithm_metric_na_count
    row["ground_truth_metric_na_count"] = ground_truth_metric_na_count
    return row


def _metric_component_values(row: Dict[str, object]) -> List[float]:
    values: List[float] = []
    for field_name, tolerance in RANKING_TOLERANCES.items():
        value = row.get(field_name)
        if value is None:
            continue
        values.append(float(value) / tolerance)
    return values


def build_ranked_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked_rows: List[Dict[str, object]] = []
    for base_row in rows:
        row = dict(base_row)
        components = _metric_component_values(row)
        metric_score_mean = sum(components) / len(components) if components else float("inf")
        beat_penalty = float(abs(int(row.get("beat_diff", 0))))
        algorithm_na_penalty = float(int(row.get("algorithm_metric_na_count", 0)) * 3)
        row["metric_score_mean"] = metric_score_mean
        row["beat_penalty"] = beat_penalty
        row["algorithm_na_penalty"] = algorithm_na_penalty
        row["rank_score"] = metric_score_mean + beat_penalty + algorithm_na_penalty
        ranked_rows.append(row)

    ranked_rows.sort(key=lambda row: (float(row["rank_score"]), int(row["record_id"])))
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
    return ranked_rows


def _comparison_file_record_id(path: Path) -> int:
    return int(path.stem.split("_", 1)[0])


def _discover_comparison_files(compare_dir: Path) -> List[Path]:
    return sorted(compare_dir.glob("*/*_comparison.txt"), key=_comparison_file_record_id)


def _csv_fieldnames(rows: Sequence[Dict[str, object]]) -> List[str]:
    preferred = [
        "rank",
        "record_id",
        "rank_score",
        "metric_score_mean",
        "beat_penalty",
        "algorithm_na_penalty",
        "available_metric_count",
        "algorithm_metric_na_count",
        "ground_truth_metric_na_count",
        "algorithm_beats",
        "ground_truth_beats",
        "beat_diff",
        "abs_beat_diff",
        "algorithm_p_missing",
        "algorithm_t_missing",
        "algorithm_qrs_missing",
        "algorithm_unreliable",
        "ground_truth_p_missing",
        "ground_truth_t_missing",
        "ground_truth_qrs_missing",
        "ground_truth_unreliable",
    ]
    for _, field_name in METRIC_SPECS:
        preferred.extend(
            [
                f"algorithm_{field_name}",
                f"ground_truth_{field_name}",
                f"diff_{field_name}",
                f"abs_{field_name}_diff",
            ]
        )
        if field_name in ANGLE_FIELDS:
            preferred.extend(
                [
                    f"circular_diff_{field_name}",
                    f"abs_{field_name}_circular_diff",
                ]
            )

    extra_fields = sorted({key for row in rows for key in row.keys()} - set(preferred))
    return [field for field in preferred if any(field in row for row in rows)] + extra_fields


def _write_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    fieldnames = _csv_fieldnames(rows)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_ranking_table(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    columns = [
        ("rank", "rank"),
        ("record_id", "record_id"),
        ("rank_score", "score"),
        ("metric_score_mean", "metric"),
        ("beat_diff", "beat_diff"),
        ("algorithm_metric_na_count", "alg_na"),
        ("abs_hr_diff", "HR"),
        ("abs_pr_diff", "PR"),
        ("abs_qrs_diff", "QRS"),
        ("abs_qt_diff", "QT"),
        ("abs_qtcb_diff", "QTcB"),
        ("abs_qtcf_diff", "QTcF"),
        ("abs_p_axis_circular_diff", "P_axis"),
        ("abs_qrs_axis_circular_diff", "QRS_axis"),
        ("abs_t_axis_circular_diff", "T_axis"),
    ]

    rendered_rows: List[Dict[str, str]] = []
    for row in rows:
        rendered: Dict[str, str] = {}
        for field_name, header in columns:
            value = row.get(field_name)
            if isinstance(value, float):
                rendered[header] = _format_float(value)
            else:
                rendered[header] = "" if value is None else str(value)
        rendered_rows.append(rendered)

    widths = {}
    for _, header in columns:
        widths[header] = max(
            len(header),
            max((len(rendered[header]) for rendered in rendered_rows), default=0),
        )

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "Ranking rule: score = mean(normalized comparable metric errors) "
            "+ abs(beat_diff) + 3 * algorithm_metric_na_count\n"
        )
        handle.write(
            "Angles use circular error, so wrap-around cases like -344.7 degrees "
            "are treated as their minimal equivalent.\n\n"
        )
        header_line = "  ".join(
            header.ljust(widths[header]) for _, header in columns
        )
        separator_line = "  ".join("-" * widths[header] for _, header in columns)
        handle.write(f"{header_line}\n")
        handle.write(f"{separator_line}\n")
        for rendered in rendered_rows:
            handle.write(
                "  ".join(
                    rendered[header].ljust(widths[header]) for _, header in columns
                )
                + "\n"
            )


def summarize_directory(
    compare_dir: Path,
    csv_name: str = "summary_ranked.csv",
    table_name: str = "ranking_table.txt",
) -> Tuple[Path, Path]:
    compare_dir = compare_dir.resolve()
    comparison_files = _discover_comparison_files(compare_dir)
    rows = [
        parse_comparison_report(
            comparison_path.read_text(encoding="utf-8"),
            record_id=_comparison_file_record_id(comparison_path),
        )
        for comparison_path in comparison_files
    ]
    ranked_rows = build_ranked_rows(rows)

    csv_path = compare_dir / csv_name
    table_path = compare_dir / table_name
    _write_csv(ranked_rows, csv_path)
    _write_ranking_table(ranked_rows, table_path)
    return csv_path, table_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize LUDB comparison outputs into a ranked CSV and table.",
    )
    parser.add_argument(
        "compare_dir",
        nargs="?",
        default="ludb_full_compare",
        help="Directory containing per-record comparison subdirectories.",
    )
    parser.add_argument(
        "--csv-name",
        default="summary_ranked.csv",
        help="Output CSV filename written inside compare_dir.",
    )
    parser.add_argument(
        "--table-name",
        default="ranking_table.txt",
        help="Output text table filename written inside compare_dir.",
    )
    args = parser.parse_args(argv)

    csv_path, table_path = summarize_directory(
        Path(args.compare_dir),
        csv_name=args.csv_name,
        table_name=args.table_name,
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
