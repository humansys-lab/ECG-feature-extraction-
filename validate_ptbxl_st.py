#!/usr/bin/env python3
"""Evaluate ecgfeat ST measurements against PTB-XL record-level labels.

PTB-XL provides record-level SCP-ECG statements, but it does not provide
lead-specific J points or ST amplitudes.  This script therefore reports
record-level label agreement rather than fiducial localization accuracy.

The direct form statements ``STD_`` and ``STE_`` are used as the primary
ST-depression and ST-elevation references.  Other ST-T, ischemia, injury and
myocardial-infarction statements are kept as exploratory context, not silently
treated as negative controls.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.models import PatientMeta, STANDARD_12_LEADS
from validate_dataset_st import (
    _beat_rows,
    _finite,
    _lead_rows,
    _load_record,
    _qualifying_leads,
    _record_predictions,
    _write_csv,
)


DIRECT_ST_DEPRESSION_CODE = "STD_"
DIRECT_ST_ELEVATION_CODE = "STE_"

ST_T_FORM_CODES = {
    "STD_",
    "STE_",
    "LOWT",
    "NT_",
    "INVT",
    "TAB_",
}
ISCHEMIA_CODES = {
    "ISC_",
    "ISCAL",
    "ISCIN",
    "ISCIL",
    "ISCAS",
    "ISCLA",
    "ISCAN",
}
INJURY_CODES = {
    "INJAS",
    "INJAL",
    "INJIN",
    "INJLA",
    "INJIL",
}
T_WAVE_CODES = {"NDT", "LOWT", "NT_", "INVT", "TAB_"}

CONTIGUOUS_TERRITORIES = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anterior": ("V1", "V2", "V3", "V4"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _sex_from_ptbxl(value: Any) -> str | None:
    parsed = _finite(value)
    if parsed is None:
        return None
    if int(round(parsed)) == 1:
        return "male"
    if int(round(parsed)) == 0:
        return "female"
    return None


def _load_manifest(
    dataset_dir: Path,
    database_csv: Path,
    statements_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    statements = {
        row[""]: row
        for row in _read_csv(statements_csv)
        if row.get("")
    }
    database = _read_csv(database_csv)
    by_filename = {
        Path(row["filename_lr"]).name: row
        for row in database
        if row.get("filename_lr")
    }
    sttc_codes = {
        code
        for code, row in statements.items()
        if row.get("diagnostic_class") == "STTC"
    }
    mi_codes = {
        code
        for code, row in statements.items()
        if row.get("diagnostic_class") == "MI"
    }
    st_context_codes = (
        ST_T_FORM_CODES
        | sttc_codes
        | mi_codes
        | ISCHEMIA_CODES
        | INJURY_CODES
    )

    manifest: list[dict[str, Any]] = []
    missing: list[str] = []
    for header in sorted(dataset_dir.glob("*.hea")):
        record_name = header.stem
        source = by_filename.get(record_name)
        if source is None:
            missing.append(record_name)
            continue
        try:
            score_map = ast.literal_eval(source.get("scp_codes", "{}"))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"invalid scp_codes for {record_name}: {source.get('scp_codes')!r}"
            ) from exc
        codes = set(score_map)
        descriptions = [
            f"{code}: {statements.get(code, {}).get('description', '')}".rstrip(": ")
            for code in score_map
        ]
        diagnostic_classes = sorted(
            {
                statements.get(code, {}).get("diagnostic_class", "")
                for code in codes
                if statements.get(code, {}).get("diagnostic") == "1.0"
                and statements.get(code, {}).get("diagnostic_class")
            }
        )
        label_st_depression = DIRECT_ST_DEPRESSION_CODE in codes
        label_st_elevation = DIRECT_ST_ELEVATION_CODE in codes
        label_sttc = bool(codes & sttc_codes)
        label_mi = bool(codes & mi_codes)
        label_ischemia = bool(codes & ISCHEMIA_CODES)
        label_injury = bool(codes & INJURY_CODES)
        direct_st_label = label_st_depression or label_st_elevation
        exploratory_st_context = bool(
            label_sttc
            or label_mi
            or label_ischemia
            or label_injury
            or codes & T_WAVE_CODES
        )
        manifest.append(
            {
                "header_path": str(header),
                "record": record_name,
                "ecg_id": int(source["ecg_id"]),
                "patient_id": source.get("patient_id", ""),
                "age": _finite(source.get("age")),
                "sex": _sex_from_ptbxl(source.get("sex")),
                "report": source.get("report", ""),
                "dx_codes": ",".join(score_map),
                "dx_names": "; ".join(descriptions),
                "scp_codes": source.get("scp_codes", ""),
                "diagnostic_classes": ",".join(diagnostic_classes),
                "label_st_depression": label_st_depression,
                "label_st_elevation": label_st_elevation,
                "label_nonspecific_st_t": bool(
                    codes & {"NST_", "NDT", "NT_", "TAB_"}
                ),
                "label_myocardial_infarction": label_mi,
                "label_t_wave_abnormality": bool(codes & T_WAVE_CODES),
                "label_t_wave_inversion": "INVT" in codes,
                "label_sttc": label_sttc,
                "label_ischemia": label_ischemia,
                "label_injury": label_injury,
                "label_norm": "NORM" in codes,
                "direct_st_label": direct_st_label,
                "exploratory_st_context": exploratory_st_context,
                "st_related_candidate": direct_st_label
                or exploratory_st_context,
                "strict_normal_control": bool(
                    "NORM" in codes and not codes & st_context_codes
                ),
                "validated_by_human": _as_bool(
                    source.get("validated_by_human")
                ),
                "strat_fold": source.get("strat_fold", ""),
            }
        )
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} local records have no metadata match: {preview}"
        )
    return manifest, statements


def _add_threshold_predictions(
    lead_rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    for threshold in (0.03, 0.05, 0.10):
        suffix = f"{threshold:.2f}".replace(".", "p")
        for method, value_key, reliable_key, strict in (
            ("native", "native_st40_mv", "native_st_reliable", False),
            ("hybrid", "hybrid_st40_mv", "hybrid_st_reliable", False),
            (
                "hybrid_strict",
                "hybrid_st40_mv",
                "hybrid_st_reliable",
                True,
            ),
        ):
            for direction in ("depression", "elevation"):
                leads, territories = _qualifying_leads(
                    lead_rows,
                    value_key=value_key,
                    reliable_key=reliable_key,
                    threshold_mv=threshold,
                    direction=direction,
                    strict_support=strict,
                )
                prefix = f"{method}_{direction}_{suffix}"
                result[prefix] = bool(territories)
                result[f"{prefix}_leads"] = ",".join(leads)
                result[f"{prefix}_territories"] = ",".join(territories)


def _interpretation_value(interpretation: Any, field: str, default: Any) -> Any:
    if interpretation is None:
        return default
    if isinstance(interpretation, Mapping):
        return interpretation.get(field, default)
    return getattr(interpretation, field, default)


def _extract_record(task: tuple[dict[str, Any], int, bool]) -> dict[str, Any]:
    record_info, fs_internal, enable_pacing = task
    started = time.perf_counter()
    try:
        ecg, fs = _load_record(Path(record_info["header_path"]).with_suffix(""))
        meta = PatientMeta(age=record_info["age"], sex=record_info["sex"])
        features = ECGFeatureExtractor(
            fs_internal=fs_internal,
            enable_pacing=enable_pacing,
            enable_hybrid_st_measurement=True,
        ).extract(ecg, fs, meta=meta)
        lead_rows = _lead_rows(record_info, features)
        predictions = _record_predictions(lead_rows)
        _add_threshold_predictions(lead_rows, predictions)

        interpretation = features.interpretation
        production_dep_leads = _interpretation_value(
            interpretation, "st_depression_leads", {}
        ) or {}
        production_ele_leads = _interpretation_value(
            interpretation, "st_elevation_leads", {}
        ) or {}
        production_dep_territories = _interpretation_value(
            interpretation, "st_territories_depressed", []
        ) or []
        production_ele_territories = _interpretation_value(
            interpretation, "st_territories_elevated", []
        ) or []

        native_st40 = [
            value
            for row in lead_rows
            if (value := _finite(row.get("native_st40_mv"))) is not None
        ]
        hybrid_st40 = [
            value
            for row in lead_rows
            if (value := _finite(row.get("hybrid_st40_mv"))) is not None
        ]
        record_row = {
            **{
                key: value
                for key, value in record_info.items()
                if key != "header_path"
            },
            "source_fs_hz": fs,
            "internal_fs_hz": features.fs,
            "n_beats": len(features.beats),
            "heart_rate_bpm": _finite(features.global_features.heart_rate_bpm),
            "native_reliable_leads": sum(
                bool(row["native_st_reliable"]) for row in lead_rows
            ),
            "hybrid_reliable_leads": sum(
                bool(row["hybrid_st_reliable"]) for row in lead_rows
            ),
            "hybrid_strict_support_leads": sum(
                bool(row["hybrid_strict_support_pass"]) for row in lead_rows
            ),
            "native_min_st40_mv": min(native_st40) if native_st40 else None,
            "hybrid_min_st40_mv": min(hybrid_st40) if hybrid_st40 else None,
            "production_native_j_depression": bool(
                production_dep_territories
            ),
            "production_native_j_depression_leads": ",".join(
                production_dep_leads
            ),
            "production_native_j_depression_territories": ",".join(
                production_dep_territories
            ),
            "production_native_j_elevation": bool(
                production_ele_territories
            ),
            "production_native_j_elevation_leads": ",".join(
                production_ele_leads
            ),
            "production_native_j_elevation_territories": ",".join(
                production_ele_territories
            ),
            **predictions,
            "runtime_seconds": time.perf_counter() - started,
            "status": "ok",
            "error": "",
        }
        return {
            "record": record_info["record"],
            "record_row": record_row,
            "lead_rows": lead_rows,
            "beat_rows": _beat_rows(record_info, features),
            "error": None,
        }
    except Exception as exc:
        return {
            "record": record_info["record"],
            "record_row": {
                **{
                    key: value
                    for key, value in record_info.items()
                    if key != "header_path"
                },
                "runtime_seconds": time.perf_counter() - started,
                "status": "failed",
                "error": str(exc),
            },
            "lead_rows": [],
            "beat_rows": [],
            "error": {
                "record": record_info["record"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _metric(
    rows: Iterable[dict[str, Any]],
    *,
    label_key: str,
    prediction_key: str,
    cohort: str,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        label = bool(row.get(label_key))
        if cohort == "direct_positive_vs_strict_normal":
            if not label and not bool(row.get("strict_normal_control")):
                continue
        elif cohort != "all_records":
            raise ValueError(f"unsupported cohort: {cohort}")
        selected.append(row)
    tp = sum(bool(row[label_key]) and bool(row[prediction_key]) for row in selected)
    fp = sum(not bool(row[label_key]) and bool(row[prediction_key]) for row in selected)
    fn = sum(bool(row[label_key]) and not bool(row[prediction_key]) for row in selected)
    tn = sum(not bool(row[label_key]) and not bool(row[prediction_key]) for row in selected)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity
        else None
    )
    sens_low, sens_high = _wilson_interval(tp, tp + fn)
    spec_low, spec_high = _wilson_interval(tn, tn + fp)
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "cohort": cohort,
        "label": label_key,
        "prediction": prediction_key,
        "n": len(selected),
        "positive_labels": tp + fn,
        "negative_labels": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "sensitivity_ci95_low": sens_low,
        "sensitivity_ci95_high": sens_high,
        "specificity": specificity,
        "specificity_ci95_low": spec_low,
        "specificity_ci95_high": spec_high,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def _detection_rate(
    rows: Iterable[dict[str, Any]],
    *,
    cohort: str,
    selector: Any,
    prediction_key: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("status") == "ok" and selector(row)
    ]
    detected = sum(bool(row.get(prediction_key)) for row in selected)
    low, high = _wilson_interval(detected, len(selected))
    return {
        "cohort": cohort,
        "prediction": prediction_key,
        "n": len(selected),
        "detected": detected,
        "detection_rate": detected / len(selected) if selected else None,
        "ci95_low": low,
        "ci95_high": high,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    finite = _finite(value)
    return "N/A" if finite is None else f"{finite:.{digits}f}"


def _write_report(
    out_dir: Path,
    *,
    dataset_dir: Path,
    database_csv: Path,
    record_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    detection_rates: list[dict[str, Any]],
    fs_internal: int,
    enable_pacing: bool,
    elapsed: float,
) -> None:
    ok_rows = [row for row in record_rows if row.get("status") == "ok"]
    main_methods = [
        "production_native_j_depression",
        "native_depression_0p03",
        "hybrid_depression_0p03",
        "hybrid_strict_depression_0p03",
        "native_depression_0p05",
        "hybrid_depression_0p05",
        "hybrid_strict_depression_0p05",
        "native_depression_0p10",
        "hybrid_depression_0p10",
        "hybrid_strict_depression_0p10",
    ]
    metric_map = {
        (row["cohort"], row["label"], row["prediction"]): row
        for row in metrics
    }
    strict_rows = [
        metric_map[
            (
                "direct_positive_vs_strict_normal",
                "label_st_depression",
                method,
            )
        ]
        for method in main_methods
    ]
    best = max(
        strict_rows,
        key=lambda row: (
            -1.0
            if row["balanced_accuracy"] is None
            else row["balanced_accuracy"]
        ),
    )

    direct_positive = [
        row for row in ok_rows if row.get("label_st_depression")
    ]
    strict_normal = [
        row for row in ok_rows if row.get("strict_normal_control")
    ]
    best_key = str(best["prediction"])
    false_negatives = [
        str(row["record"])
        for row in direct_positive
        if not bool(row.get(best_key))
    ]
    false_positives = [
        str(row["record"])
        for row in strict_normal
        if bool(row.get(best_key))
    ]

    lines = [
        "# PTB-XL 00000：ecgfeat ST 修改验证",
        "",
        "## 结论摘要",
        "",
        f"- 成功处理：{len(ok_rows)}/{len(record_rows)} 条记录。",
        f"- 直接 ST 压低标签（`STD_`）：{len(direct_positive)} 条；"
        f"严格正常对照：{len(strict_normal)} 条。",
        "- 当前目录没有 `STE_` 标签，因此不能估计 ST 抬高灵敏度。",
        f"- 在直接阳性与严格正常对照上，平衡准确率最高的是 "
        f"`{best_key}`：灵敏度 {_fmt(best['sensitivity'])}，"
        f"特异度 {_fmt(best['specificity'])}，F1 {_fmt(best['f1'])}，"
        f"平衡准确率 {_fmt(best['balanced_accuracy'])}。",
        "- PTB-XL 只有记录级 SCP 标签，没有逐导联 J 点或 ST 幅度真值；"
        "这里是标签一致性评估，不是 J 点定位 MAE，也不是临床诊断验证。",
        "",
        "## 运行配置",
        "",
        f"- 波形目录：`{dataset_dir}`",
        f"- 元数据：`{database_csv}`",
        f"- ecgfeat 内部采样率：{fs_internal} Hz",
        f"- 起搏检测：{'开启' if enable_pacing else '关闭'}",
        f"- 总运行时间：{elapsed:.1f} 秒",
        "",
        "## ST 压低主结果",
        "",
        "评估队列为 75 条明确 `STD_` 阳性与不含 ST-T/MI 标签的 "
        "NORM 对照。预测规则要求同一解剖区域至少两个相邻导联越过阈值。",
        "",
        "| 方法 | 阈值/测量 | TP | FP | FN | TN | 灵敏度 | 特异度 | 精确率 | F1 | 平衡准确率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "production_native_j_depression": ("最终解释层", "原生 J 点 / 0.03 mV"),
        "native_depression_0p03": ("原始 ST40", "0.03 mV"),
        "hybrid_depression_0p03": ("修改后 ST40", "0.03 mV"),
        "hybrid_strict_depression_0p03": ("修改后 ST40 + 严格门控", "0.03 mV"),
        "native_depression_0p05": ("原始 ST40", "0.05 mV"),
        "hybrid_depression_0p05": ("修改后 ST40", "0.05 mV"),
        "hybrid_strict_depression_0p05": ("修改后 ST40 + 严格门控", "0.05 mV"),
        "native_depression_0p10": ("原始 ST40", "0.10 mV"),
        "hybrid_depression_0p10": ("修改后 ST40", "0.10 mV"),
        "hybrid_strict_depression_0p10": ("修改后 ST40 + 严格门控", "0.10 mV"),
    }
    for row in strict_rows:
        method, threshold = labels[str(row["prediction"])]
        lines.append(
            f"| {method} | {threshold} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {row['tn']} | {_fmt(row['sensitivity'])} | "
            f"{_fmt(row['specificity'])} | {_fmt(row['precision'])} | "
            f"{_fmt(row['f1'])} | {_fmt(row['balanced_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## 探索性队列检出率",
            "",
            "下表不是灵敏度：缺血、损伤和心肌梗死标签不保证当前 10 秒"
            "波形一定存在可见 ST 压低。",
            "",
            "| 队列 | N | 检出 | 检出率 | 预测规则 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in detection_rates:
        lines.append(
            f"| {row['cohort']} | {row['n']} | {row['detected']} | "
            f"{_fmt(row['detection_rate'])} | `{row['prediction']}` |"
        )

    lines.extend(
        [
            "",
            "## 最高平衡准确率规则的错误记录",
            "",
            f"- 假阴性（{len(false_negatives)}）："
            + (", ".join(false_negatives) if false_negatives else "无"),
            f"- 严格正常对照中的假阳性（{len(false_positives)}）："
            + (", ".join(false_positives) if false_positives else "无"),
            "",
            "逐记录预测、逐导联测量、全部指标及失败信息分别见同目录 CSV/JSON 文件。",
            "",
        ]
    )
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "00000",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ptb-xl-metadata-1.0.1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "ptbxl_00000_st_validation",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--fs-internal",
        type=int,
        default=500,
        help="ecgfeat internal sample rate (default: 500 Hz).",
    )
    parser.add_argument(
        "--disable-pacing",
        action="store_true",
        help="Disable pacing detection. Required if fs-internal is too low.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records for a smoke test.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir.resolve()
    metadata_dir = args.metadata_dir.resolve()
    out_dir = args.out_dir.resolve()
    database_csv = metadata_dir / "ptbxl_database.csv"
    statements_csv = metadata_dir / "scp_statements.csv"
    enable_pacing = not args.disable_pacing
    if enable_pacing and args.fs_internal <= 200:
        raise SystemExit(
            "pacing detection uses a 100 Hz high-pass; use --fs-internal 500 "
            "or add --disable-pacing"
        )
    for path in (dataset_dir, database_csv, statements_csv):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    manifest, _ = _load_manifest(
        dataset_dir,
        database_csv,
        statements_csv,
    )
    if args.limit is not None:
        manifest = manifest[: max(0, args.limit)]
    if not manifest:
        raise SystemExit("no records selected")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tasks = [
        (record_info, int(args.fs_internal), enable_pacing)
        for record_info in manifest
    ]
    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            results.append(_extract_record(task))
            print(
                f"[{index}/{len(tasks)}] {task[0]['record']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_record, task): task[0]["record"]
                for task in tasks
            }
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {futures[future]}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item["record"])
    record_rows = [item["record_row"] for item in results]
    lead_rows = [row for item in results for row in item["lead_rows"]]
    beat_rows = [row for item in results for row in item["beat_rows"]]
    errors = [
        item["error"] for item in results if item["error"] is not None
    ]

    _write_csv(out_dir / "record_summary.csv", record_rows)
    _write_csv(out_dir / "lead_features.csv", lead_rows)
    _write_csv(out_dir / "st_context_beat_features.csv", beat_rows)
    _write_csv(
        out_dir / "st_candidate_records.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key != "header_path"
            }
            for row in manifest
            if row["st_related_candidate"]
        ],
    )

    depression_predictions = [
        "production_native_j_depression",
        *[
            f"{method}_depression_{suffix}"
            for suffix in ("0p03", "0p05", "0p10")
            for method in ("native", "hybrid", "hybrid_strict")
        ],
    ]
    elevation_predictions = [
        "production_native_j_elevation",
        *[
            f"{method}_elevation_{suffix}"
            for suffix in ("0p05", "0p10")
            for method in ("native", "hybrid", "hybrid_strict")
        ],
    ]
    metrics = [
        _metric(
            record_rows,
            label_key=label_key,
            prediction_key=prediction,
            cohort=cohort,
        )
        for cohort in ("direct_positive_vs_strict_normal", "all_records")
        for label_key, predictions in (
            ("label_st_depression", depression_predictions),
            ("label_st_elevation", elevation_predictions),
        )
        for prediction in predictions
    ]
    _write_csv(out_dir / "label_agreement_metrics.csv", metrics)

    exploratory_prediction = "hybrid_strict_depression_0p05"
    cohort_selectors = [
        ("Direct STD_", lambda row: bool(row.get("label_st_depression"))),
        ("Strict NORM control", lambda row: bool(row.get("strict_normal_control"))),
        ("Diagnostic STTC", lambda row: bool(row.get("label_sttc"))),
        ("Ischemia", lambda row: bool(row.get("label_ischemia"))),
        ("Subendocardial injury", lambda row: bool(row.get("label_injury"))),
        (
            "Myocardial infarction",
            lambda row: bool(row.get("label_myocardial_infarction")),
        ),
    ]
    detection_rates = [
        _detection_rate(
            record_rows,
            cohort=name,
            selector=selector,
            prediction_key=exploratory_prediction,
        )
        for name, selector in cohort_selectors
    ]
    _write_csv(out_dir / "exploratory_detection_rates.csv", detection_rates)

    summary = {
        "dataset_dir": str(dataset_dir),
        "database_csv": str(database_csv),
        "statements_csv": str(statements_csv),
        "records_discovered": len(manifest),
        "records_processed": sum(
            row.get("status") == "ok" for row in record_rows
        ),
        "records_failed": len(errors),
        "direct_st_depression_records": sum(
            bool(row.get("label_st_depression")) for row in record_rows
        ),
        "direct_st_elevation_records": sum(
            bool(row.get("label_st_elevation")) for row in record_rows
        ),
        "strict_normal_controls": sum(
            bool(row.get("strict_normal_control")) for row in record_rows
        ),
        "fs_internal": int(args.fs_internal),
        "enable_pacing": enable_pacing,
        "workers": workers,
        "elapsed_seconds": elapsed,
        "errors": errors,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        out_dir,
        dataset_dir=dataset_dir,
        database_csv=database_csv,
        record_rows=record_rows,
        metrics=metrics,
        detection_rates=detection_rates,
        fs_internal=int(args.fs_internal),
        enable_pacing=enable_pacing,
        elapsed=elapsed,
    )
    print(
        f"Finished: processed={summary['records_processed']} "
        f"failed={summary['records_failed']} elapsed={elapsed:.1f}s "
        f"out={out_dir}",
        flush=True,
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
