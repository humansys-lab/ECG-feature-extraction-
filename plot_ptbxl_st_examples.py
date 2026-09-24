#!/usr/bin/env python3
"""Plot full 12-lead PTB-XL ECGs with native and hybrid ST annotations.

Examples are selected from the outputs of ``validate_ptbxl_st.py`` using the
record-level direct ``STD_`` label and strict NORM controls.  The default
selection contains six score-spaced examples from each of TP, FP, FN and TN.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from ecgfeat.compat.api_v0 import ECGFeatureExtractor
from ecgfeat.models import PatientMeta, STANDARD_12_LEADS
from ecgfeat.preprocess import analysis_signal, resample_ecg
from validate_dataset_st import _load_record


CATEGORY_ORDER = ("TP", "FP", "FN", "TN")
CATEGORY_COLORS = {
    "TP": "#2ca02c",
    "FP": "#d62728",
    "FN": "#ff7f0e",
    "TN": "#1f77b4",
}
TERRITORIES = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anterior": ("V1", "V2", "V3", "V4"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _category(row: dict[str, str], prediction_key: str) -> str | None:
    label = _bool(row.get("label_st_depression"))
    predicted = _bool(row.get(prediction_key))
    if label:
        return "TP" if predicted else "FN"
    if _bool(row.get("strict_normal_control")):
        return "FP" if predicted else "TN"
    return None


def _spaced_selection(
    rows: Iterable[dict[str, str]],
    *,
    count: int,
) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _finite(row.get("hybrid_st40_score_mv")) or 0.0,
            row["record"],
        ),
    )
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count)
    selected: list[dict[str, str]] = []
    seen: set[int] = set()
    for raw_index in indices:
        index = int(round(float(raw_index)))
        if index not in seen:
            selected.append(ordered[index])
            seen.add(index)
    return selected


def select_examples(
    record_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    *,
    prediction_key: str,
    per_category: int,
) -> list[dict[str, str]]:
    scores = {row["record"]: row for row in score_rows}
    candidates: dict[str, list[dict[str, str]]] = {
        category: [] for category in CATEGORY_ORDER
    }
    for record_row in record_rows:
        category = _category(record_row, prediction_key)
        score_row = scores.get(record_row["record"])
        if category is None or score_row is None:
            continue
        candidates[category].append(
            {
                **record_row,
                **{
                    key: value
                    for key, value in score_row.items()
                    if key not in record_row
                },
                "category": category,
            }
        )
    return [
        row
        for category in CATEGORY_ORDER
        for row in _spaced_selection(
            candidates[category],
            count=per_category,
        )
    ]


def _effective_native_j(feature: Any) -> int | None:
    for value in (
        getattr(feature, "st_j_remeasured_index", None),
        getattr(getattr(feature, "qrs", None), "offset", None),
    ):
        if value is not None:
            return int(value)
    return None


def _short_text(value: str, limit: int = 155) -> str:
    normalized = " ".join(str(value).split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _territory_prediction(
    representative_values: dict[str, float],
    *,
    threshold_mv: float,
) -> tuple[list[str], list[str]]:
    qualified = sorted(
        lead
        for lead, value in representative_values.items()
        if value <= -abs(threshold_mv)
    )
    territories = [
        name
        for name, leads in TERRITORIES.items()
        if sum(lead in qualified for lead in leads) >= 2
    ]
    return qualified, territories


def _render_one(task: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    row = task["row"]
    dataset_dir = Path(task["dataset_dir"])
    out_dir = Path(task["out_dir"])
    fs_internal = int(task["fs_internal"])
    threshold_mv = float(task["threshold_mv"])
    record_name = row["record"]
    category = row["category"]

    ecg, source_fs = _load_record(dataset_dir / record_name)
    patient_meta = PatientMeta(
        age=_finite(row.get("age")),
        sex=row.get("sex") or None,
    )
    features = ECGFeatureExtractor(
        fs_internal=fs_internal,
        enable_hybrid_st_measurement=True,
    ).extract(ecg, source_fs, meta=patient_meta)
    measurement_ecg = analysis_signal(
        resample_ecg(ecg, source_fs, fs_internal),
        fs_internal,
        mains_hz=50,
    )
    time_seconds = np.arange(measurement_ecg.shape[1], dtype=float) / fs_internal
    beat_by_lead = {
        lead: [
            item for item in features.beat_features if item.lead == lead
        ]
        for lead in STANDARD_12_LEADS
    }
    hybrid_values: dict[str, float] = {}
    native_values: dict[str, float] = {}
    hybrid_reliable: dict[str, bool] = {}
    for lead in STANDARD_12_LEADS:
        params = features.representative_leads[lead].params
        native = _finite(params.get("st_mid_mv"))
        hybrid = _finite(params.get("st_hybrid_40ms_mv"))
        if native is not None:
            native_values[lead] = native
        if hybrid is not None:
            hybrid_values[lead] = hybrid
        hybrid_reliable[lead] = bool(params.get("st_hybrid_reliable", False))
    reliable_hybrid_values = {
        lead: value
        for lead, value in hybrid_values.items()
        if hybrid_reliable.get(lead, False)
    }
    qualified_leads, territories = _territory_prediction(
        reliable_hybrid_values,
        threshold_mv=threshold_mv,
    )

    fig, axes = plt.subplots(
        12,
        1,
        figsize=(20, 20),
        sharex=True,
        constrained_layout=False,
    )
    for lead_index, (lead, axis) in enumerate(
        zip(STANDARD_12_LEADS, axes)
    ):
        signal = measurement_ecg[lead_index]
        if lead in qualified_leads:
            axis.set_facecolor("#fff0f0")
        axis.plot(
            time_seconds,
            signal,
            color="#202020",
            linewidth=0.72,
            zorder=1,
        )
        for feature in beat_by_lead[lead]:
            qrs_peak = getattr(getattr(feature, "qrs", None), "peak", None)
            native_j = _effective_native_j(feature)
            hybrid_j = getattr(feature, "st_hybrid_j_index", None)
            baseline = _finite(
                getattr(feature, "st_hybrid_baseline_mv", None)
            )
            hybrid_st40 = _finite(
                getattr(feature, "st_hybrid_40ms_mv", None)
            )
            reliable = bool(
                getattr(feature, "st_hybrid_reliable", False)
            )

            if qrs_peak is not None and 0 <= int(qrs_peak) < len(signal):
                index = int(qrs_peak)
                axis.scatter(
                    index / fs_internal,
                    signal[index],
                    marker="^",
                    color="#2ca02c",
                    s=15,
                    linewidths=0.5,
                    zorder=4,
                )
            if native_j is not None and 0 <= native_j < len(signal):
                axis.scatter(
                    native_j / fs_internal,
                    signal[native_j],
                    marker="x",
                    color="#6f6f6f",
                    s=17,
                    linewidths=0.9,
                    zorder=5,
                )
            if hybrid_j is None or not 0 <= int(hybrid_j) < len(signal):
                continue
            hybrid_j = int(hybrid_j)
            hybrid_color = "#d62728" if reliable else "#f3a2a2"
            axis.scatter(
                hybrid_j / fs_internal,
                signal[hybrid_j],
                marker="o",
                facecolors="none",
                edgecolors=hybrid_color,
                s=23,
                linewidths=1.0,
                zorder=5,
            )
            if baseline is not None:
                segment_start = max(0, hybrid_j - int(round(0.020 * fs_internal)))
                segment_end = min(
                    len(signal) - 1,
                    hybrid_j + int(round(0.100 * fs_internal)),
                )
                axis.plot(
                    [segment_start / fs_internal, segment_end / fs_internal],
                    [baseline, baseline],
                    color="#17becf",
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.75,
                    zorder=2,
                )
                axis.plot(
                    [hybrid_j / fs_internal, segment_end / fs_internal],
                    [baseline - threshold_mv, baseline - threshold_mv],
                    color="#9467bd",
                    linestyle=":",
                    linewidth=0.75,
                    alpha=0.80,
                    zorder=2,
                )
            if baseline is not None and hybrid_st40 is not None:
                st40_index = min(
                    len(signal) - 1,
                    hybrid_j + int(round(0.040 * fs_internal)),
                )
                axis.scatter(
                    st40_index / fs_internal,
                    baseline + hybrid_st40,
                    marker="D",
                    color="#1f77b4" if reliable else "#aec7e8",
                    s=15,
                    linewidths=0.4,
                    zorder=6,
                )

        native_text = native_values.get(lead)
        hybrid_text = hybrid_values.get(lead)
        status = "Q" if lead in qualified_leads else "-"
        annotation = (
            f"{lead}  N40={native_text:+.3f}"
            if native_text is not None
            else f"{lead}  N40=N/A"
        )
        annotation += (
            f"  H40={hybrid_text:+.3f}"
            if hybrid_text is not None
            else "  H40=N/A"
        )
        annotation += f"  rel={'Y' if hybrid_reliable[lead] else 'N'}  {status}"
        axis.text(
            0.004,
            0.88,
            annotation,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#a00000" if lead in qualified_leads else "#202020",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=8,
        )
        axis.grid(which="major", color="#f3a7a7", linewidth=0.42, alpha=0.7)
        axis.grid(which="minor", color="#f8d2d2", linewidth=0.25, alpha=0.65)
        axis.xaxis.set_major_locator(MultipleLocator(1.0))
        axis.xaxis.set_minor_locator(MultipleLocator(0.2))
        axis.set_ylabel("mV", fontsize=7)
        axis.tick_params(axis="both", labelsize=7)
        axis.margins(x=0)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    axes[-1].set_xlim(0.0, len(time_seconds) / fs_internal)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#2ca02c",
            markeredgecolor="#2ca02c",
            markersize=5,
            label="lead-local QRS peak",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#6f6f6f",
            linestyle="none",
            markersize=5,
            label="native J",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#d62728",
            markerfacecolor="none",
            linestyle="none",
            markersize=5,
            label="hybrid J (red=reliable)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#1f77b4",
            linestyle="none",
            markersize=4,
            label="hybrid ST40 value",
        ),
        Line2D(
            [0],
            [0],
            color="#17becf",
            linestyle="--",
            linewidth=1,
            label="hybrid PR baseline",
        ),
        Line2D(
            [0],
            [0],
            color="#9467bd",
            linestyle=":",
            linewidth=1,
            label=f"baseline - {threshold_mv:.3f} mV",
        ),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        ncol=3,
        fontsize=7.5,
        framealpha=0.88,
    )
    label = _bool(row.get("label_st_depression"))
    prediction = bool(territories)
    score = _finite(row.get("hybrid_st40_score_mv"))
    title = (
        f"{record_name} | {category} | STD_ label={label} | "
        f"hybrid prediction={prediction} | score={score:.3f} mV | "
        f"threshold={threshold_mv:.3f} mV | "
        f"territories={','.join(territories) or 'none'}"
        if score is not None
        else (
            f"{record_name} | {category} | STD_ label={label} | "
            f"hybrid prediction={prediction}"
        )
    )
    subtitle = (
        f"SCP codes: {row.get('dx_codes', '')} | "
        f"report: {_short_text(row.get('report', ''))}"
    )
    fig.suptitle(
        title + "\n" + subtitle,
        fontsize=11.5,
        color=CATEGORY_COLORS[category],
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.052,
        right=0.995,
        top=0.965,
        bottom=0.035,
        hspace=0.16,
    )
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    out_path = category_dir / f"{record_name}_{category}_12lead_10s_st.png"
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    return {
        "record": record_name,
        "category": category,
        "label_st_depression": label,
        "hybrid_prediction": prediction,
        "hybrid_st40_score_mv": score,
        "threshold_mv": threshold_mv,
        "qualified_leads": ",".join(qualified_leads),
        "territories": ",".join(territories),
        "dx_codes": row.get("dx_codes", ""),
        "report": row.get("report", ""),
        "plot_path": str(out_path),
        "status": "ok",
        "error": "",
    }


def _failed_result(task: dict[str, Any], error: Exception) -> dict[str, Any]:
    row = task["row"]
    return {
        "record": row["record"],
        "category": row["category"],
        "label_st_depression": row.get("label_st_depression", ""),
        "hybrid_prediction": "",
        "hybrid_st40_score_mv": row.get("hybrid_st40_score_mv", ""),
        "threshold_mv": task["threshold_mv"],
        "qualified_leads": "",
        "territories": "",
        "dx_codes": row.get("dx_codes", ""),
        "report": row.get("report", ""),
        "plot_path": "",
        "status": "failed",
        "error": str(error),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_contact_sheet(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    columns: int = 4,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        return
    thumb_width = 620
    thumb_height = 650
    label_height = 34
    rows_count = int(np.ceil(len(successful) / columns))
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows_count * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(successful):
        image = Image.open(row["plot_path"]).convert("RGB")
        image.thumbnail((thumb_width - 8, thumb_height - 8))
        col = index % columns
        line = index // columns
        x = col * thumb_width + (thumb_width - image.width) // 2
        y = line * (thumb_height + label_height) + 4
        sheet.paste(image, (x, y))
        label = (
            f"{row['category']} | {row['record']} | "
            f"score={float(row['hybrid_st40_score_mv']):.3f} mV | "
            f"{row['territories'] or 'no territory'}"
        )
        draw.text(
            (col * thumb_width + 8, y + thumb_height),
            label,
            fill=CATEGORY_COLORS[row["category"]],
            font=font,
        )
    sheet.save(out_path, quality=92)


def _write_readme(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    threshold_mv: float,
    per_category: int,
) -> None:
    lines = [
        "# PTB-XL ST 标注图",
        "",
        f"- 规则：修改后 ST40 ≤ -{threshold_mv:.3f} mV，"
        "同一解剖区域至少两个相邻导联。",
        f"- 例数：TP、FP、FN、TN 各 {per_category} 条，按连续得分等距抽样。",
        "- 每张图包含完整 12 导联、10 秒 ECG。",
        "- 浅红导联背景及 `Q` 表示该导联达到压低阈值；最终仍需相邻导联共识。",
        "- N40/H40 分别是代表心拍的原始/修改后 ST40 幅度。",
        "",
        "## 标记",
        "",
        "- 绿色三角：逐导联 QRS 峰。",
        "- 灰色叉号：原始 J 点。",
        "- 红色空心圆：修改后 J 点；浅红表示该拍不可靠。",
        "- 蓝色菱形：修改后 ST40 测量。",
        "- 青色虚线：局部 PR 基线。",
        "- 紫色点线：当前心拍的 PR 基线减去 ST 压低阈值。",
        "",
        "## 文件",
        "",
        "- `contact_sheet_24.png`：24 个例子的总览。",
        "- `example_manifest.csv`：分类、分数、诊断代码、报告和图片路径。",
        "- `TP/FP/FN/TN/`：各分类的原始大图。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "00000",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "ptbxl_00000_st_validation",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ptbxl_00000_st_validation"
        / "annotated_examples",
    )
    parser.add_argument("--threshold-mv", type=float, default=0.03)
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--fs-internal", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir.resolve()
    results_dir = args.results_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    record_rows = _read_csv(results_dir / "record_summary.csv")
    score_rows = _read_csv(results_dir / "record_st_scores.csv")
    prediction_key = (
        f"hybrid_depression_"
        f"{args.threshold_mv:.2f}".replace(".", "p")
    )
    if prediction_key not in record_rows[0]:
        raise SystemExit(
            f"prediction field {prediction_key!r} is absent from record_summary.csv"
        )
    selected = select_examples(
        record_rows,
        score_rows,
        prediction_key=prediction_key,
        per_category=max(1, int(args.per_category)),
    )
    tasks = [
        {
            "row": row,
            "dataset_dir": str(dataset_dir),
            "out_dir": str(out_dir),
            "fs_internal": int(args.fs_internal),
            "threshold_mv": float(args.threshold_mv),
        }
        for row in selected
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(_render_one, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _failed_result(task, exc)
            results.append(result)
            print(
                f"[{completed}/{len(tasks)}] {result['record']} "
                f"{result['category']} {result['status']}",
                flush=True,
            )
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    results.sort(
        key=lambda row: (
            order[row["category"]],
            float(row["hybrid_st40_score_mv"] or 0.0),
            row["record"],
        )
    )
    _write_csv(out_dir / "example_manifest.csv", results)
    _make_contact_sheet(
        results,
        out_path=out_dir / f"contact_sheet_{len(results)}.png",
    )
    _write_readme(
        out_dir,
        results,
        threshold_mv=float(args.threshold_mv),
        per_category=max(1, int(args.per_category)),
    )
    failures = sum(row["status"] != "ok" for row in results)
    print(
        f"Finished: plots={len(results) - failures} failures={failures} "
        f"out={out_dir}",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
