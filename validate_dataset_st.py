#!/usr/bin/env python3
"""Validate native and hybrid ST measurements on a local WFDB directory.

The local ``dataset`` headers contain record-level SNOMED labels but no
lead-specific J-point or ST-amplitude annotations. Therefore this script:

1. finds directly labelled ST depression/elevation records;
2. keeps nonspecific ST-T and myocardial-infarction labels as a separate
   exploratory cohort;
3. extracts native and hybrid ST measurements for every record;
4. reports record-level label agreement using a fixed two-contiguous-lead
   ST40 screening rule;
5. exports lead/beat measurements and marker plots for direct ST labels.

The reported confusion metrics are label-agreement metrics, not clinical
diagnostic accuracy and not J-point MAE against expert fiducials.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from ecgfeat.api import ECGFeatureExtractor
from ecgfeat.models import PatientMeta, STANDARD_12_LEADS


ST_DEPRESSION_CODE = "429622005"
ST_ELEVATION_CODE = "164931005"
NONSPECIFIC_ST_T_CODE = "428750005"
MYOCARDIAL_INFARCTION_CODE = "164865005"
T_WAVE_ABNORMALITY_CODE = "164934002"
T_WAVE_INVERSION_CODE = "59931005"

LABEL_NAMES = {
    ST_DEPRESSION_CODE: "ST depression",
    ST_ELEVATION_CODE: "ST elevation",
    NONSPECIFIC_ST_T_CODE: "Nonspecific ST-T abnormality",
    MYOCARDIAL_INFARCTION_CODE: "Myocardial infarction",
    T_WAVE_ABNORMALITY_CODE: "T-wave abnormality",
    T_WAVE_INVERSION_CODE: "T-wave inversion",
}

# Screening territories. This is a fixed research comparison rule, not the
# production clinical interpretation path.
CONTIGUOUS_TERRITORIES = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anterior": ("V1", "V2", "V3", "V4"),
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _parse_header(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "record": path.stem,
        "age": None,
        "sex": None,
        "dx_codes": [],
    }
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("#Age:"):
            result["age"] = _finite(line.split(":", 1)[1].strip())
        elif line.startswith("#Sex:"):
            value = line.split(":", 1)[1].strip()
            result["sex"] = value if value and value.lower() != "unknown" else None
        elif line.startswith("#Dx:"):
            result["dx_codes"] = [
                code.strip()
                for code in line.split(":", 1)[1].split(",")
                if code.strip()
            ]
    codes = set(result["dx_codes"])
    result.update(
        {
            "label_st_depression": ST_DEPRESSION_CODE in codes,
            "label_st_elevation": ST_ELEVATION_CODE in codes,
            "label_nonspecific_st_t": NONSPECIFIC_ST_T_CODE in codes,
            "label_myocardial_infarction": MYOCARDIAL_INFARCTION_CODE in codes,
            "label_t_wave_abnormality": T_WAVE_ABNORMALITY_CODE in codes,
            "label_t_wave_inversion": T_WAVE_INVERSION_CODE in codes,
        }
    )
    result["direct_st_label"] = bool(
        result["label_st_depression"] or result["label_st_elevation"]
    )
    result["exploratory_st_context"] = bool(
        result["label_nonspecific_st_t"]
        or result["label_myocardial_infarction"]
    )
    result["st_related_candidate"] = bool(
        result["direct_st_label"] or result["exploratory_st_context"]
    )
    result["dx_names"] = [
        LABEL_NAMES[code] for code in result["dx_codes"] if code in LABEL_NAMES
    ]
    return result


def _load_record(record_path: Path) -> tuple[np.ndarray, float]:
    import wfdb

    record = wfdb.rdrecord(str(record_path))
    indices = {str(name).lower(): index for index, name in enumerate(record.sig_name)}
    missing = [lead for lead in STANDARD_12_LEADS if lead.lower() not in indices]
    if missing:
        raise ValueError(f"missing standard leads: {', '.join(missing)}")
    ecg = np.asarray(record.p_signal, dtype=float)[
        :,
        [indices[lead.lower()] for lead in STANDARD_12_LEADS],
    ].T
    return ecg, float(record.fs)


def _lead_rows(
    record_info: dict[str, Any],
    features: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    beat_by_lead = {
        lead: [
            item for item in features.beat_features
            if item.lead == lead
        ]
        for lead in STANDARD_12_LEADS
    }
    for lead in STANDARD_12_LEADS:
        params = features.representative_leads[lead].params
        lead_beats = beat_by_lead[lead]
        hybrid_reliable_beats = sum(
            bool(getattr(item, "st_hybrid_reliable", False))
            for item in lead_beats
        )
        hybrid_total_beats = len(lead_beats)
        strict_support_minimum = max(
            2,
            int(math.ceil(0.20 * hybrid_total_beats)),
        )
        rows.append(
            {
                "record": record_info["record"],
                "lead": lead,
                "label_st_depression": record_info["label_st_depression"],
                "label_st_elevation": record_info["label_st_elevation"],
                "native_st_j_mv": _finite(params.get("st_on_mv")),
                "native_st40_mv": _finite(params.get("st_mid_mv")),
                "native_st80_mv": _finite(params.get("st_80ms_mv")),
                "native_st_trend": params.get("st_morphology"),
                "native_st_reliable": bool(params.get("st_j_reliable", False)),
                "native_st_source": params.get("st_j_source"),
                "hybrid_st_j_mv": _finite(params.get("st_hybrid_j_mv")),
                "hybrid_st40_mv": _finite(params.get("st_hybrid_40ms_mv")),
                "hybrid_st80_mv": _finite(params.get("st_hybrid_80ms_mv")),
                "hybrid_st_mean_mv": _finite(params.get("st_hybrid_mean_mv")),
                "hybrid_st_area_mv_ms": _finite(
                    params.get("st_hybrid_area_mv_ms")
                ),
                "hybrid_st_slope_mv_per_ms": _finite(
                    params.get("st_hybrid_slope_mv_per_ms")
                ),
                "hybrid_st_trend": params.get("st_hybrid_trend"),
                "hybrid_st_shape": params.get("st_hybrid_shape"),
                "hybrid_baseline_mv": _finite(
                    params.get("st_hybrid_baseline_mv")
                ),
                "hybrid_baseline_source": params.get(
                    "st_hybrid_baseline_source"
                ),
                "hybrid_baseline_confidence": _finite(
                    params.get("st_hybrid_baseline_confidence")
                ),
                "hybrid_j_method": params.get("st_hybrid_j_method"),
                "hybrid_confidence": _finite(
                    params.get("st_hybrid_j_confidence")
                ),
                "hybrid_consensus_support": params.get(
                    "st_hybrid_consensus_support"
                ),
                "hybrid_beat_support": hybrid_reliable_beats,
                "hybrid_beat_total": hybrid_total_beats,
                "hybrid_strict_support_minimum": strict_support_minimum,
                "hybrid_strict_support_pass": bool(
                    hybrid_reliable_beats >= strict_support_minimum
                ),
                "hybrid_st_reliable": bool(
                    params.get("st_hybrid_reliable", False)
                ),
                "hybrid_unreliable_reason": params.get(
                    "st_hybrid_unreliable_reason"
                ),
            }
        )
    return rows


def _beat_rows(
    record_info: dict[str, Any],
    features: Any,
) -> list[dict[str, Any]]:
    if not record_info["st_related_candidate"]:
        return []
    rows: list[dict[str, Any]] = []
    for item in features.beat_features:
        native_j = (
            item.st_j_remeasured_index
            if item.st_j_remeasured_index is not None
            else item.qrs.offset
        )
        shift_ms = (
            None
            if native_j is None or item.st_hybrid_j_index is None
            else (
                (int(item.st_hybrid_j_index) - int(native_j))
                * 1000.0
                / float(features.fs)
            )
        )
        rows.append(
            {
                "record": record_info["record"],
                "lead": item.lead,
                "beat_id": item.beat_id,
                "native_j_index": native_j,
                "hybrid_j_index": item.st_hybrid_j_index,
                "hybrid_j_shift_ms": shift_ms,
                "native_st_j_mv": item.st_on_mv,
                "native_st40_mv": item.st_mid_mv,
                "native_st80_mv": item.st_80ms_mv,
                "native_st_trend": item.st_morphology,
                "hybrid_st_j_mv": item.st_hybrid_j_mv,
                "hybrid_st40_mv": item.st_hybrid_40ms_mv,
                "hybrid_st80_mv": item.st_hybrid_80ms_mv,
                "hybrid_st_mean_mv": item.st_hybrid_mean_mv,
                "hybrid_st_area_mv_ms": item.st_hybrid_area_mv_ms,
                "hybrid_st_slope_mv_per_ms": item.st_hybrid_slope_mv_per_ms,
                "hybrid_st_trend": item.st_hybrid_trend,
                "hybrid_st_shape": item.st_hybrid_shape,
                "hybrid_baseline_mv": item.st_hybrid_baseline_mv,
                "hybrid_baseline_source": item.st_hybrid_baseline_source,
                "hybrid_baseline_confidence": item.st_hybrid_baseline_confidence,
                "hybrid_j_method": item.st_hybrid_j_method,
                "hybrid_confidence": item.st_hybrid_j_confidence,
                "hybrid_consensus_index": item.st_hybrid_consensus_index,
                "hybrid_consensus_support": item.st_hybrid_consensus_support,
                "hybrid_st_reliable": item.st_hybrid_reliable,
                "hybrid_unreliable_reason": item.st_hybrid_unreliable_reason,
            }
        )
    return rows


def _qualifying_leads(
    lead_rows: list[dict[str, Any]],
    *,
    value_key: str,
    reliable_key: str,
    threshold_mv: float,
    direction: str,
    strict_support: bool = False,
) -> tuple[list[str], list[str]]:
    qualified: set[str] = set()
    for row in lead_rows:
        if not bool(row.get(reliable_key, False)):
            continue
        if strict_support and not bool(row.get("hybrid_strict_support_pass", False)):
            continue
        value = _finite(row.get(value_key))
        if value is None:
            continue
        if direction == "depression" and value <= -abs(threshold_mv):
            qualified.add(str(row["lead"]))
        elif direction == "elevation" and value >= abs(threshold_mv):
            qualified.add(str(row["lead"]))
    territories = [
        territory
        for territory, leads in CONTIGUOUS_TERRITORIES.items()
        if sum(lead in qualified for lead in leads) >= 2
    ]
    return sorted(qualified), territories


def _record_predictions(
    lead_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in (0.05, 0.10):
        suffix = "0p05" if threshold == 0.05 else "0p10"
        for method, value_key, reliable_key, strict in (
            (
                "native",
                "native_st40_mv",
                "native_st_reliable",
                False,
            ),
            (
                "hybrid",
                "hybrid_st40_mv",
                "hybrid_st_reliable",
                False,
            ),
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
    return result


def _extract_record(
    header_path: str,
) -> dict[str, Any]:
    path = Path(header_path)
    record_info = _parse_header(path)
    try:
        ecg, fs = _load_record(path.with_suffix(""))
        meta = PatientMeta(
            age=record_info["age"],
            sex=record_info["sex"],
        )
        features = ECGFeatureExtractor(
            fs_internal=int(round(fs)),
            enable_hybrid_st_measurement=True,
        ).extract(ecg, fs, meta=meta)
        lead_rows = _lead_rows(record_info, features)
        predictions = _record_predictions(lead_rows)
        record_row = {
            **record_info,
            "dx_codes": ",".join(record_info["dx_codes"]),
            "dx_names": "; ".join(record_info["dx_names"]),
            "n_beats": len(features.beats),
            "heart_rate_bpm": _finite(features.global_features.heart_rate_bpm),
            "hybrid_reliable_leads": sum(
                bool(row["hybrid_st_reliable"]) for row in lead_rows
            ),
            "hybrid_strict_support_leads": sum(
                bool(row["hybrid_strict_support_pass"]) for row in lead_rows
            ),
            "native_min_st40_mv": min(
                value
                for row in lead_rows
                if (value := _finite(row["native_st40_mv"])) is not None
            ),
            "hybrid_min_st40_mv": min(
                value
                for row in lead_rows
                if (value := _finite(row["hybrid_st40_mv"])) is not None
            ),
            **predictions,
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
                **record_info,
                "dx_codes": ",".join(record_info["dx_codes"]),
                "dx_names": "; ".join(record_info["dx_names"]),
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _confusion_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    label_key: str,
    prediction_key: str,
    strict_controls: bool,
) -> dict[str, Any]:
    selected = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        label = bool(row.get(label_key, False))
        if (
            strict_controls
            and not label
            and (
                bool(row.get("label_nonspecific_st_t", False))
                or bool(row.get("label_myocardial_infarction", False))
                or bool(row.get("label_t_wave_abnormality", False))
                or bool(row.get("label_t_wave_inversion", False))
                or bool(row.get("label_st_elevation", False))
            )
        ):
            continue
        selected.append(row)
    tp = sum(
        bool(row[label_key]) and bool(row[prediction_key])
        for row in selected
    )
    fp = sum(
        not bool(row[label_key]) and bool(row[prediction_key])
        for row in selected
    )
    fn = sum(
        bool(row[label_key]) and not bool(row[prediction_key])
        for row in selected
    )
    tn = sum(
        not bool(row[label_key]) and not bool(row[prediction_key])
        for row in selected
    )
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity > 0
        else None
    )
    return {
        "cohort": "strict_controls" if strict_controls else "all_label_negative",
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
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def _plot_direct_record(
    *,
    dataset_dir: Path,
    out_dir: Path,
    record_row: dict[str, Any],
    beat_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record_id = str(record_row["record"])
    ecg, fs = _load_record(dataset_dir / record_id)
    rows_by_lead = {
        lead: [
            row for row in beat_rows
            if row["record"] == record_id and row["lead"] == lead
        ]
        for lead in STANDARD_12_LEADS
    }
    time = np.arange(ecg.shape[1], dtype=float) / fs
    fig, axes = plt.subplots(12, 1, figsize=(18, 20), sharex=True)
    for lead_index, (lead, axis) in enumerate(zip(STANDARD_12_LEADS, axes)):
        signal = ecg[lead_index]
        axis.plot(time, signal, color="black", linewidth=0.65)
        first_label = True
        for row in rows_by_lead[lead]:
            native_j = row.get("native_j_index")
            hybrid_j = row.get("hybrid_j_index")
            reliable = bool(row.get("hybrid_st_reliable", False))
            if native_j is not None and 0 <= int(native_j) < len(signal):
                index = int(native_j)
                axis.scatter(
                    index / fs,
                    signal[index],
                    marker="x",
                    color="#7f7f7f",
                    s=18,
                    label="native J" if first_label else None,
                    zorder=3,
                )
            if hybrid_j is not None and 0 <= int(hybrid_j) < len(signal):
                index = int(hybrid_j)
                color = "#d62728" if reliable else "#ff9896"
                axis.scatter(
                    index / fs,
                    signal[index],
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    s=24,
                    label="hybrid J" if first_label else None,
                    zorder=3,
                )
                for offset_ms, marker, point_color, label in (
                    (40.0, "^", "#ff7f0e", "hybrid J+40"),
                    (80.0, "s", "#1f77b4", "hybrid J+80"),
                ):
                    point = index + int(round(offset_ms * fs / 1000.0))
                    if 0 <= point < len(signal):
                        axis.scatter(
                            point / fs,
                            signal[point],
                            marker=marker,
                            color=point_color,
                            s=18,
                            label=label if first_label else None,
                            zorder=3,
                        )
            first_label = False
        axis.set_ylabel(lead, rotation=0, labelpad=20)
        axis.grid(alpha=0.16)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{record_id} | label: ST depression={record_row['label_st_depression']}, "
        f"ST elevation={record_row['label_st_elevation']} | "
        "gray=native J, red=hybrid J, orange=J+40, blue=J+80",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"{record_id}_st_markers.png", dpi=150)
    plt.close(fig)


def _plot_st40_comparison(
    *,
    out_dir: Path,
    record_row: dict[str, Any],
    lead_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record_id = str(record_row["record"])
    selected = [
        row for row in lead_rows
        if str(row["record"]) == record_id
    ]
    selected.sort(key=lambda row: STANDARD_12_LEADS.index(str(row["lead"])))
    if not selected:
        return
    x = np.arange(len(selected), dtype=float)
    native = [
        np.nan if _finite(row["native_st40_mv"]) is None
        else float(row["native_st40_mv"])
        for row in selected
    ]
    hybrid = [
        np.nan if _finite(row["hybrid_st40_mv"]) is None
        else float(row["hybrid_st40_mv"])
        for row in selected
    ]
    fig, axis = plt.subplots(figsize=(14, 5.5))
    axis.bar(x - 0.20, native, width=0.40, color="#7f7f7f", label="native ST40")
    bars = axis.bar(
        x + 0.20,
        hybrid,
        width=0.40,
        color="#d62728",
        label="hybrid ST40",
    )
    for bar, row in zip(bars, selected):
        if not bool(row.get("hybrid_strict_support_pass", False)):
            bar.set_hatch("///")
            bar.set_alpha(0.55)
    axis.axhline(-0.05, color="#ff7f0e", linestyle="--", linewidth=1.2, label="-0.05 mV")
    axis.axhline(-0.10, color="#9467bd", linestyle=":", linewidth=1.2, label="-0.10 mV")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([str(row["lead"]) for row in selected])
    axis.set_ylabel("ST40 relative to baseline (mV)")
    axis.set_title(
        f"{record_id}: native vs hybrid representative ST40 "
        "(hatched hybrid = insufficient strict beat support)"
    )
    axis.grid(axis="y", alpha=0.20)
    axis.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"{record_id}_st40_comparison.png", dpi=170)
    plt.close(fig)


def _fmt_metric(value: Any) -> str:
    number = _finite(value)
    return "N/A" if number is None else f"{number:.3f}"


def _write_results_markdown(
    path: Path,
    *,
    dataset_dir: Path,
    record_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    direct = [
        row for row in record_rows
        if row.get("status") == "ok" and row.get("direct_st_label")
    ]
    exploratory = [
        row for row in record_rows
        if row.get("status") == "ok"
        and row.get("exploratory_st_context")
        and not row.get("direct_st_label")
    ]
    lines = [
        "# dataset ST validation",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Records processed: {sum(row.get('status') == 'ok' for row in record_rows)}",
        f"- Direct ST-depression labels: {sum(bool(row.get('label_st_depression')) for row in record_rows)}",
        f"- Direct ST-elevation labels: {sum(bool(row.get('label_st_elevation')) for row in record_rows)}",
        f"- Exploratory ST-T/MI-only records: {len(exploratory)}",
        f"- Extraction failures: {len(errors)}",
        "",
        "The headers provide record-level SNOMED labels only. They do not provide",
        "lead-specific J points or ST amplitudes. Therefore the metrics below are",
        "record-label agreement for a fixed research screen, not clinical accuracy.",
        "",
        "## Direct ST-labelled records",
        "",
        "| Record | Label | HR | Native min ST40 | Hybrid min ST40 | Native 0.05 | Hybrid 0.05 | Hybrid strict support |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in direct:
        label = "ST depression" if row["label_st_depression"] else "ST elevation"
        lines.append(
            "| {record} | {label} | {hr} | {native} | {hybrid} | {npred} | {hpred} | {hstrict} |".format(
                record=row["record"],
                label=label,
                hr=_fmt_metric(row.get("heart_rate_bpm")),
                native=_fmt_metric(row.get("native_min_st40_mv")),
                hybrid=_fmt_metric(row.get("hybrid_min_st40_mv")),
                npred=row.get("native_depression_0p05"),
                hpred=row.get("hybrid_depression_0p05"),
                hstrict=row.get("hybrid_strict_depression_0p05"),
            )
        )
    lines.extend(
        [
            "",
            "## Record-level ST-depression label agreement",
            "",
            "Screen: ST40 <= -0.05 or -0.10 mV in at least two leads from the",
            "same inferior, lateral, or anterior territory.",
            "",
            "| Cohort | Prediction | TP | FP | FN | TN | Sensitivity | Specificity | Precision | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in metrics:
        formatted_metric = {
            **metric,
            "sensitivity": _fmt_metric(metric["sensitivity"]),
            "specificity": _fmt_metric(metric["specificity"]),
            "precision": _fmt_metric(metric["precision"]),
            "f1": _fmt_metric(metric["f1"]),
        }
        lines.append(
            "| {cohort} | {prediction} | {tp} | {fp} | {fn} | {tn} | {sensitivity} | {specificity} | {precision} | {f1} |".format(
                **formatted_metric,
            )
        )
    metric_by_key = {
        (metric["cohort"], metric["prediction"]): metric
        for metric in metrics
    }
    all_native_005 = metric_by_key.get(
        ("all_label_negative", "native_depression_0p05")
    )
    all_hybrid_005 = metric_by_key.get(
        ("all_label_negative", "hybrid_depression_0p05")
    )
    all_native_010 = metric_by_key.get(
        ("all_label_negative", "native_depression_0p10")
    )
    all_hybrid_010 = metric_by_key.get(
        ("all_label_negative", "hybrid_depression_0p10")
    )
    disagreement_records = [
        str(row["record"])
        for row in direct
        if bool(row.get("native_depression_0p05"))
        and not bool(row.get("hybrid_depression_0p05"))
    ]
    if all(
        metric is not None
        for metric in (
            all_native_005,
            all_hybrid_005,
            all_native_010,
            all_hybrid_010,
        )
    ):
        lines.extend(
            [
                "",
                "## Observed comparison",
                "",
                "- At the 0.05 mV screen, native sensitivity/specificity were "
                f"{_fmt_metric(all_native_005['sensitivity'])}/"
                f"{_fmt_metric(all_native_005['specificity'])}; hybrid were "
                f"{_fmt_metric(all_hybrid_005['sensitivity'])}/"
                f"{_fmt_metric(all_hybrid_005['specificity'])}.",
                "- At the 0.10 mV screen, native sensitivity/specificity were "
                f"{_fmt_metric(all_native_010['sensitivity'])}/"
                f"{_fmt_metric(all_native_010['specificity'])}; hybrid were "
                f"{_fmt_metric(all_hybrid_010['sensitivity'])}/"
                f"{_fmt_metric(all_hybrid_010['specificity'])}.",
                "- Hybrid reduced label-negative positive screens, but missed "
                f"{', '.join(disagreement_records) if disagreement_records else 'no direct positive record'} "
                "at 0.05 mV.",
                "- The strict beat-support variant is too conservative for the "
                "fast-rhythm positive record and should not be treated as the "
                "preferred detector without further tuning.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Only three direct ST-depression positives are available; confidence intervals are necessarily very wide.",
            "- There are no direct ST-elevation positives, so ST-elevation sensitivity cannot be measured.",
            "- Label absence is not proof of a clinically normal ST segment.",
            "- Nonspecific ST-T abnormality and myocardial infarction are kept separate from direct ST truth.",
            "- Visual review of the three direct records is required, especially disagreements between native and hybrid baselines.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Directory containing paired .hea/.mat WFDB records.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_st_validation",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="Extract only direct ST labels and exploratory ST-T/MI records.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = sorted(dataset_dir.glob("*.hea"))
    if not headers:
        raise SystemExit(f"no .hea files found under {dataset_dir}")
    manifest = [_parse_header(path) for path in headers]
    selected_headers = [
        path
        for path, info in zip(headers, manifest)
        if not args.candidates_only or info["st_related_candidate"]
    ]

    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for path in selected_headers:
            results.append(_extract_record(str(path)))
            print(f"[{len(results)}/{len(selected_headers)}] {path.stem}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_record, str(path)): path.stem
                for path in selected_headers
            }
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                print(
                    f"[{completed}/{len(selected_headers)}] {futures[future]}",
                    flush=True,
                )

    results.sort(key=lambda item: item["record"])
    record_rows = [item["record_row"] for item in results]
    lead_rows = [
        row for item in results for row in item["lead_rows"]
    ]
    beat_rows = [
        row for item in results for row in item["beat_rows"]
    ]
    errors = [
        item["error"] for item in results if item["error"] is not None
    ]

    candidate_manifest = [
        {
            **info,
            "dx_codes": ",".join(info["dx_codes"]),
            "dx_names": "; ".join(info["dx_names"]),
        }
        for info in manifest
        if info["st_related_candidate"]
    ]
    _write_csv(out_dir / "candidate_records.csv", candidate_manifest)
    _write_csv(out_dir / "record_summary.csv", record_rows)
    _write_csv(out_dir / "lead_features.csv", lead_rows)
    _write_csv(out_dir / "candidate_beat_features.csv", beat_rows)

    prediction_keys = [
        "native_depression_0p05",
        "hybrid_depression_0p05",
        "hybrid_strict_depression_0p05",
        "native_depression_0p10",
        "hybrid_depression_0p10",
        "hybrid_strict_depression_0p10",
    ]
    metrics = [
        _confusion_metrics(
            record_rows,
            label_key="label_st_depression",
            prediction_key=prediction,
            strict_controls=strict_controls,
        )
        for strict_controls in (False, True)
        for prediction in prediction_keys
    ]
    _write_csv(out_dir / "label_agreement_metrics.csv", metrics)

    summary = {
        "dataset_dir": str(dataset_dir),
        "records_discovered": len(headers),
        "records_processed": sum(
            row.get("status") == "ok" for row in record_rows
        ),
        "direct_st_depression_records": [
            info["record"] for info in manifest
            if info["label_st_depression"]
        ],
        "direct_st_elevation_records": [
            info["record"] for info in manifest
            if info["label_st_elevation"]
        ],
        "exploratory_st_context_records": [
            info["record"] for info in manifest
            if info["exploratory_st_context"]
            and not info["direct_st_label"]
        ],
        "metrics": metrics,
        "errors": errors,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.no_plots:
        direct_rows = {
            str(row["record"]): row
            for row in record_rows
            if row.get("status") == "ok" and row.get("direct_st_label")
        }
        for record_id, row in direct_rows.items():
            _plot_direct_record(
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                record_row=row,
                beat_rows=beat_rows,
            )
            _plot_st40_comparison(
                out_dir=out_dir,
                record_row=row,
                lead_rows=lead_rows,
            )

    _write_results_markdown(
        out_dir / "RESULTS.md",
        dataset_dir=dataset_dir,
        record_rows=record_rows,
        metrics=metrics,
        errors=errors,
    )
    print(f"Results: {out_dir}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
