#!/usr/bin/env python3
"""Render human-readable per-record ecgfeat explanations from compact results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluate_target_ecgfeat_diagnosis import PREDICTION_CODE_ZH


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / "ecgfeat_target_diagnosis_20260726"

STATUS_ZH = {
    "abnormal": "发现异常",
    "abnormal_with_limited_coverage": "发现异常，但部分诊断域覆盖受限",
    "normal_with_core_coverage": "核心规则范围内未发现异常",
    "incomplete": "未发现权威异常，但规则覆盖不完整",
    "borderline": "存在临界结果",
    "technically_limited": "技术质量受限",
    "technically_limited_with_findings": "技术质量受限，但仍有异常发现",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _natural_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", value)
    return (int(match.group(1)) if match else 10**12, value)


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value in (None, ""):
        return "不可用"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}{suffix}"


def _split_zh(value: str) -> list[str]:
    return [part.strip() for part in value.split("；") if part.strip()]


def _load_raw(result_dir: Path, dataset: str, record: str) -> dict[str, Any]:
    return json.loads(
        (result_dir / "raw" / dataset / f"{record}.json").read_text(
            encoding="utf-8"
        )
    )


def _reference_lines(raw: Mapping[str, Any]) -> list[str]:
    active = set(raw.get("reference_codes_active") or [])
    scores = raw.get("reference_scores") or {}
    names = raw.get("reference_names") or {}
    lines: list[str] = []
    for code in raw.get("reference_codes_raw") or []:
        name = names.get(code, code)
        if code in scores:
            score = float(scores[code])
            policy = (
                "本次作为参考阳性"
                if code in active
                else "诊断似然低于 50，本次未作为严格阳性"
            )
            lines.append(
                f"- `{code}` — {name}；原始分数 {score:g}；{policy}。"
            )
        else:
            lines.append(f"- `{code}` — {name}；无原始置信分数。")
    return lines or ["- 无记录级诊断标签。"]


def _prediction_lines(raw: Mapping[str, Any]) -> list[str]:
    final = (raw.get("clinical") or {}).get("final_statements") or []
    if not final:
        return ["- 无权威异常语句。"]
    lines = []
    for item in final:
        code = str(item.get("statement_code") or "unknown")
        label = PREDICTION_CODE_ZH.get(
            code, str(item.get("statement") or code)
        )
        details = [f"规则 `{item.get('rule_id')}`"]
        if item.get("confidence"):
            details.append(f"置信等级 `{item['confidence']}`")
        if item.get("coverage"):
            details.append(f"覆盖 `{item['coverage']}`")
        lines.append(f"- `{code}` — {label}；{'；'.join(details)}。")
    return lines


def _comparison_lines(row: Mapping[str, str]) -> list[str]:
    items = (
        ("命中", row.get("hit_categories", "")),
        ("漏诊", row.get("missed_categories", "")),
        ("相对标签的额外输出", row.get("extra_categories", "")),
        ("参考中不可比/未支持", row.get("unsupported_reference_codes", "")),
        ("算法输出但参考无法验证", row.get("unverifiable_prediction_codes", "")),
    )
    return [
        f"- {label}：{value or '无'}。"
        for label, value in items
    ]


def _reason_text(row: Mapping[str, str]) -> str:
    text = row.get("detailed_analysis_zh", "").strip()
    if "。 " in text:
        text = text.split("。 ", 1)[1]
    return text or "本条没有可比漏诊或相对标签的额外诊断家族。"


def _improvement_lines(row: Mapping[str, str]) -> list[str]:
    suggestions = _split_zh(row.get("algorithm_improvements_zh", ""))
    return [f"- {item}。" if not item.endswith("。") else f"- {item}" for item in suggestions] or [
        "- 当前不需要从本条单独推导算法修改；应继续结合相同诊断家族的其他病例验证。"
    ]


def _measurement_table(raw: Mapping[str, Any]) -> list[str]:
    values = raw.get("global") or {}
    quality = raw.get("quality") or {}
    return [
        "| HR | 房率 | PR | QRS | QT | QTcB | QRS 轴 | 心拍数 | QRS/P/T 可用导联 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            (
                _fmt(values.get("heart_rate_bpm"), 1, " bpm"),
                _fmt(values.get("atrial_rate_bpm"), 1, " bpm"),
                _fmt(values.get("pr_ms"), 0, " ms"),
                _fmt(values.get("qrs_ms"), 0, " ms"),
                _fmt(values.get("qt_ms"), 0, " ms"),
                _fmt(values.get("qtc_bazett_ms"), 0, " ms"),
                _fmt(values.get("qrs_axis_deg"), 1, "°"),
                str(raw.get("n_beats") if raw.get("n_beats") is not None else "不可用"),
                (
                    f"{len(quality.get('qrs_reliable_leads') or [])}/"
                    f"{len(quality.get('p_reliable_leads') or [])}/"
                    f"{len(quality.get('t_reliable_leads') or [])}"
                ),
            )
        )
        + " |",
    ]


def _render_record(
    result_dir: Path,
    row: Mapping[str, str],
    output_file: Path,
) -> list[str]:
    dataset = row["dataset"]
    record = row["record"]
    raw = _load_raw(result_dir, dataset, record)
    raw_path = (
        result_dir / "raw" / dataset / f"{record}.json"
    ).resolve()
    status = row.get("clinical_overall_status", "")
    status_text = STATUS_ZH.get(status, status or "未知")
    quality_flags = row.get("quality_flags") or "无"
    report = row.get("reference_report", "").strip()

    lines = [
        f"## `{record}` — {row.get('verdict', '未判定')}",
        "",
        f"- ecgfeat 总体状态：{status_text}（`{status}`）。",
        f"- 根因标签：{row.get('root_cause_tags') or '无'}。",
        f"- 信号质量标志：{quality_flags}。",
    ]
    if report:
        lines.append(f"- PTB-XL 原始报告：{report}")
    lines.extend(
        [
            "",
            "### 数据自带诊断",
            "",
            *_reference_lines(raw),
            "",
            "### ecgfeat 最终结果",
            "",
            *_prediction_lines(raw),
            "",
            "### 关键测量",
            "",
            *_measurement_table(raw),
            "",
            "### 对照判断",
            "",
            *_comparison_lines(row),
            "",
            "### 为什么会得到这个结果",
            "",
            _reason_text(row),
            "",
            "### 对算法改进的启发",
            "",
            *_improvement_lines(row),
            "",
            f"[查看本条完整规则证据]({raw_path})",
            "",
        ]
    )
    return lines


def _chunked(
    rows: Sequence[dict[str, str]], size: int
) -> Iterable[Sequence[dict[str, str]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _render_volume(
    *,
    result_dir: Path,
    output_dir: Path,
    dataset: str,
    rows: Sequence[dict[str, str]],
    part_number: int,
) -> tuple[Path, Counter[str]]:
    first = rows[0]["record"]
    last = rows[-1]["record"]
    filename = f"{dataset}_第{part_number:02d}卷_{first}-{last}.md"
    path = output_dir / filename
    verdicts = Counter(row["verdict"] for row in rows)
    first_raw = _load_raw(result_dir, dataset, first)
    ruleset_version = (
        (first_raw.get("clinical") or {}).get("ruleset_version") or "未知"
    )
    source_dir = Path(str(first_raw.get("record_path") or "")).parent
    lines = [
        f"# 数据集 `{dataset}` 逐条说明：第 {part_number} 卷",
        "",
        f"- 记录范围：`{first}` 至 `{last}`",
        f"- 本卷记录数：{len(rows)}",
        f"- 来源目录：`{source_dir}`",
        f"- 诊断结果来自 ecgfeat 规则集 `{ruleset_version}`。",
        "- “漏诊/额外输出”均是相对记录级标签而言，不等同最终临床判断。",
        "",
        "## 本卷判定分布",
        "",
    ]
    lines.extend(
        f"- {key}：{value}"
        for key, value in sorted(
            verdicts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    lines.extend(["", "---", ""])
    for row in rows:
        lines.extend(_render_record(result_dir, row, path))
        lines.extend(["---", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, verdicts


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _priority_observation_lines(
    result_dir: Path,
    datasets: set[str],
    dataset_summaries: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    metric_rows = [
        row
        for row in _read_csv(result_dir / "诊断家族指标.csv")
        if row.get("dataset") in datasets
        and (
            _as_int(row.get("ground_truth_positive")) > 0
            or _as_int(row.get("predicted_positive")) > 0
        )
    ]
    low_recall = sorted(
        (
            row
            for row in metric_rows
            if _as_int(row.get("ground_truth_positive")) > 0
            and _as_float(row.get("recall_sensitivity")) is not None
        ),
        key=lambda row: (
            _as_float(row.get("recall_sensitivity")) or 0.0,
            -_as_int(row.get("ground_truth_positive")),
        ),
    )[:6]
    high_fp = sorted(
        (row for row in metric_rows if _as_int(row.get("fp")) > 0),
        key=lambda row: -_as_int(row.get("fp")),
    )[:6]
    root_causes: Counter[str] = Counter()
    for dataset in datasets:
        root_causes.update(
            dataset_summaries.get(dataset, {}).get(
                "root_cause_record_counts", {}
            )
        )

    lines = ["## 本批优先改进线索", ""]
    if low_recall:
        recall_text = "；".join(
            (
                f"{row['category_zh']} "
                f"{row['tp']}/{row['ground_truth_positive']} "
                f"（召回率 {float(row['recall_sensitivity']):.3f}）"
            )
            for row in low_recall
        )
        lines.append(f"- 低召回诊断：{recall_text}。")
    if high_fp:
        fp_text = "；".join(
            f"{row['category_zh']} FP={row['fp']}" for row in high_fp
        )
        lines.append(f"- 额外输出较多：{fp_text}。")
    if root_causes:
        cause_text = "；".join(
            f"{name} {count}条" for name, count in root_causes.most_common(6)
        )
        lines.append(f"- 高频根因标签：{cause_text}。")
    lines.extend(
        [
            "- 房扑、房颤和房室传导阻滞应联合检查 QRST 减除、F/P 波跨导联一致性及 AF/AFL 控制流。",
            "- 束支阻滞和 IVCD 应使用采样率感知的 QRS 灰区，并改善 V1 R′、I/V6 终末 S 波提取。",
            "- PAC/PVC 应加强逐心拍提前量、代偿间歇、P 波关联及多导联形态投票。",
            "- T/ST、LVH 和 Q 波等宽泛形态标签需要专家复核，并与直接诊断指标分开评价。",
            "",
        ]
    )
    return lines


def _render_index(
    *,
    result_dir: Path,
    output_dir: Path,
    volumes: Sequence[tuple[str, Path, int, Counter[str]]],
) -> None:
    summary = json.loads(
        (result_dir / "summary.json").read_text(encoding="utf-8")
    )
    dataset_summaries = {
        item["dataset"]: item for item in summary["datasets"]
    }
    analysis_path = (result_dir / "ANALYSIS.md").resolve()
    metrics_path = (result_dir / "诊断家族指标.csv").resolve()
    datasets = {dataset for dataset, _, _, _ in volumes}
    total_records = sum(count for _, _, count, _ in volumes)
    lines = [
        "# ecgfeat 逐条结果说明索引",
        "",
        f"本目录把 {total_records} 条结果转换为可阅读的逐例解释。每条均包含数据标签、",
        "ecgfeat 最终语句、关键测量、对照判断、失败原因和病例级改进启发。",
        "",
        "## 阅读原则",
        "",
        "- 额外输出可能是算法误报，也可能是参考标签不完整，必须结合波形人工复核。",
        "- 本结果用于研究和工程分析，不能替代医生判读。",
    ]
    if "00000" in datasets:
        lines.append(
            "- `00000` 的 SCP 数字是 statement likelihood，不是校准概率；"
            " diagnostic 代码低于 50 时未作为严格阳性，form/rhythm 描述码按存在处理。"
        )
    if "010" in datasets:
        lines.append("- SNOMED-CT `#Dx` 只有诊断代码，没有原始置信度。")
    lines.extend(
        [
            "",
            "## 总体结果",
            "",
            "| 数据集 | 记录 | 执行成功 | 执行失败 | 微平均精确率 | 微平均召回率 | 微平均 F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in sorted(datasets, key=_natural_key):
        item = dataset_summaries[dataset]
        lines.append(
            f"| `{dataset}` | {item['record_count']} | "
            f"{item['extraction_succeeded']} | {item['extraction_failed']} | "
            f"{item['micro_precision']:.3f} | "
            f"{item['micro_recall_sensitivity']:.3f} | "
            f"{item['micro_f1']:.3f} |"
        )
    lines.extend([""])
    lines.extend(
        _priority_observation_lines(
            result_dir, datasets, dataset_summaries
        )
    )
    if analysis_path.exists():
        lines.extend(
            [
                f"[查看系统性根因与完整改进方案]({analysis_path})",
                "",
            ]
        )
    lines.extend(
        [
            f"[查看诊断家族指标 CSV]({metrics_path})",
            "",
            "## 分卷",
            "",
            "| 数据集 | 分卷 | 记录数 | 判定概览 |",
            "|---|---|---:|---|",
        ]
    )
    for dataset, path, count, verdicts in volumes:
        overview = "；".join(
            f"{key} {value}"
            for key, value in sorted(
                verdicts.items(), key=lambda item: (-item[1], item[0])
            )
        )
        lines.append(
            f"| `{dataset}` | [{path.stem}]({path.resolve()}) | "
            f"{count} | {overview} |"
        )
    lines.append("")
    (output_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, default=DEFAULT_RESULT_DIR
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ptbxl-volume-size",
        type=int,
        default=100,
        help="Number of PTB-XL records per Markdown volume.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_dir / "逐条说明"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    for csv_path in sorted(result_dir.glob("逐条诊断结果_*.csv")):
        dataset = csv_path.stem.removeprefix("逐条诊断结果_")
        rows = sorted(
            _read_csv(csv_path),
            key=lambda row: _natural_key(row["record"]),
        )
        if rows:
            rows_by_dataset[dataset] = rows
    if not rows_by_dataset:
        raise SystemExit(f"no non-empty per-record CSV found in {result_dir}")

    volumes: list[tuple[str, Path, int, Counter[str]]] = []
    for dataset, dataset_rows in rows_by_dataset.items():
        volume_size = (
            max(1, int(args.ptbxl_volume_size))
            if dataset == "00000"
            else len(dataset_rows)
        )
        for part, chunk in enumerate(
            _chunked(dataset_rows, volume_size), start=1
        ):
            path, verdicts = _render_volume(
                result_dir=result_dir,
                output_dir=output_dir,
                dataset=dataset,
                rows=chunk,
                part_number=part,
            )
            volumes.append((dataset, path, len(chunk), verdicts))
    _render_index(
        result_dir=result_dir,
        output_dir=output_dir,
        volumes=volumes,
    )
    print(
        f"rendered {sum(count for _, _, count, _ in volumes)} records "
        f"into {len(volumes)} volumes at {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
