#!/usr/bin/env python3
"""
Demo: 对 dataset 中随机抽取一条记录进行 ECG 特征提取，展示全部测量结果。

用法:
    python demo_feature_extraction.py [RECORD_ID] [--full-beats]

示例:
    python demo_feature_extraction.py          # 随机取样
    python demo_feature_extraction.py JS00010  # 指定记录
    python demo_feature_extraction.py JS00010 --full-beats  # JSON 里保留逐拍逐导联明细(beat_features)

默认导出的 JSON 会省略 beat_features（逐拍逐导联明细，体积占比最大且大多数
消费者用不到），只保留 representative_leads/clinical_interpretation 等汇总
结果；加 --full-beats 可以完整保留用于流水线调试/审计。
"""

from __future__ import annotations

import json
import random
import sys
import textwrap
from pathlib import Path

import numpy as np

# 将 feature_extraction 包加入 Python 路径
PROJECT_ROOT = Path(__file__).parent
FEATURE_EXTRACTION_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTION_ROOT))

from ecgfeat.compat.api_v0 import ECGFeatureExtractor
from ecgfeat.compat.export_v0 import clinical_fingerprint, prepare_json_export, to_dict
from ecgfeat.io import load_wfdb_mat as load_ecg
from ecgfeat.io import parse_wfdb_header as parse_hea
from ecgfeat.models import PatientMeta, STANDARD_12_LEADS

DATASET_DIR = PROJECT_ROOT / "dataset"
CLINICAL_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II"

# Common SNOMED CT labels used by the local ECG dataset.
# Unknown codes fall back to the raw code string in the report.
DX_CODE_MAP = {
    "164889003": "Atrial fibrillation",
    "164890007": "Atrial flutter",
    "164909002": "Left bundle branch block",
    "164912004": "P-wave abnormality",
    "164917005": "Q-wave abnormality",
    "164934002": "T-wave abnormality",
    "17338001": "Ventricular premature beats",
    "233917008": "Atrioventricular block",
    "251173003": "Atrial bigeminy",
    "270492004": "First-degree atrioventricular block",
    "284470004": "Premature atrial contraction",
    "426177001": "Sinus bradycardia",
    "426783006": "Sinus rhythm",
    "427084000": "Sinus tachycardia",
    "428750005": "Nonspecific ST-T abnormality",
    "429622005": "ST depression",
    "55827005": "Left ventricular hypertrophy",
    "59118001": "Right bundle branch block",
    "59931005": "T-wave inversion",
    "39732003": "Left axis deviation",
    "47665007": "Right axis deviation",
    "251146004": "Low QRS voltages",
    "251199005": "Counterclockwise cardiac rotation",
    "698252002": "Nonspecific intraventricular conduction delay",
    "75532003": "Ventricular escape beat",
}


# ──────────────────────────────────────────────
# 格式化辅助
# ──────────────────────────────────────────────

def _fmt(v, unit: str = "", precision: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{precision}f}{unit}"


def _sep(title: str = "", width: int = 62) -> None:
    if title:
        pad = width - len(title) - 4
        print(f"\n── {title} {'─' * pad}")
    else:
        print("─" * width)


def _diagnosis_details(codes: list[str]) -> list[str]:
    details = []
    for code in codes:
        label = DX_CODE_MAP.get(code)
        details.append(f"{code} ({label})" if label else code)
    return details


def _dxl_af_reference_display(
    probable_af: bool,
    clinical_final_codes: set[str],
) -> tuple[str, str | None]:
    if probable_af and "atrial_flutter_pattern" in clinical_final_codes:
        return (
            "Yes  [REFERENCE CONFLICT — NOT FINAL]",
            "Superseded by authoritative atrial flutter",
        )
    return ("Yes" if probable_af else "No", None)


def _ecg_paper_dimensions(hdr: dict, ecg: np.ndarray, fig_width: float = 16.0) -> tuple[float, float]:
    fs = hdr["fs"]
    total_ms = ecg.shape[1] / fs * 1000.0
    segment_ms = total_ms / 4.0
    ms_per_mV = 400.0
    peak = float(np.max(np.abs(ecg))) if ecg.size else 1.0
    y_lim = max(1.5, np.ceil((peak + 0.2) / 0.5) * 0.5)
    left, right = 0.03, 0.98
    top, bottom = 0.90, 0.06
    hspace, wspace = 0.06, 0.03
    small_box_aspect = ((2.0 * y_lim) * ms_per_mV) / max(segment_ms, 1.0)
    fig_height = fig_width * ((right - left) / (top - bottom)) * small_box_aspect * (
        (4 + 3 * hspace) / (4 + 3 * wspace)
    )
    return fig_width, fig_height


def _draw_ecg_paper_layout(
    fig,
    gs,
    record_id: str,
    hdr: dict,
    ecg: np.ndarray,
    *,
    show_title: bool = True,
    lead_fontsize: float = 10.0,
    trace_linewidth: float = 1.0,
    bottom_labelsize: float = 7.0,
    preserve_paper_aspect: bool = True,
    panel_callback=None,
) -> None:
    import matplotlib.ticker as ticker

    fs = hdr["fs"]
    total_ms = ecg.shape[1] / fs * 1000.0
    segment_ms = total_ms / 4.0
    lead_to_idx = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    ms_per_mV = 400.0  # 25 mm/s and 10 mm/mV -> 0.1 mV equals 40 ms
    cal_width_ms = 200.0

    peak = float(np.max(np.abs(ecg))) if ecg.size else 1.0
    y_lim = max(1.5, np.ceil((peak + 0.2) / 0.5) * 0.5)
    y_lo, y_hi = -y_lim, y_lim

    def second_only_formatter(x: float, _pos: float) -> str:
        nearest_sec = round(x / 1000.0)
        if abs(x - nearest_sec * 1000.0) <= 1e-6:
            return f"{nearest_sec}s"
        return ""

    def style_ecg_axes(ax, x_span_ms: float, show_bottom: bool = False) -> None:
        ax.set_facecolor("#FFF8F8")
        ax.set_xlim(0.0, x_span_ms)
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(40))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        if preserve_paper_aspect and hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(((y_hi - y_lo) * ms_per_mV) / max(x_span_ms, 1.0))
        ax.grid(which="major", color="#FF7F7F", linewidth=0.75, alpha=0.95)
        ax.grid(which="minor", color="#FFB6B6", linewidth=0.35, alpha=0.95)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        if show_bottom:
            ax.tick_params(axis="x", which="major", length=0, labelsize=bottom_labelsize, colors="#777777")
            ax.tick_params(axis="x", which="minor", length=0, labelbottom=False)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(second_only_formatter))
        else:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_segment(ax, lead: str, start_ms: float, end_ms: float, show_bottom: bool = False, show_cal: bool = False) -> None:
        sig = ecg[lead_to_idx[lead]]
        i0 = max(0, int(round(start_ms * fs / 1000.0)))
        i1 = min(len(sig), int(round(end_ms * fs / 1000.0)))
        x_span_ms = max(end_ms - start_ms, 1.0)
        t_ms = (np.arange(i0, i1) - i0) * 1000.0 / fs
        style_ecg_axes(ax, x_span_ms, show_bottom=show_bottom)
        ax.plot(t_ms, sig[i0:i1], color="#222222", linewidth=trace_linewidth, zorder=5)
        ax.text(
            x_span_ms * 0.012,
            y_hi - (y_hi - y_lo) * 0.04,
            lead,
            fontsize=lead_fontsize,
            fontweight="bold",
            ha="left",
            va="top",
            color="#333333",
        )
        if show_cal:
            cal_x = x_span_ms * 0.015
            ax.plot(
                [cal_x, cal_x, cal_x + cal_width_ms, cal_x + cal_width_ms],
                [y_lo + 0.25, y_lo + 1.25, y_lo + 1.25, y_lo + 0.25],
                color="#555555",
                linewidth=max(1.2, trace_linewidth),
                zorder=6,
            )
        if panel_callback is not None:
            panel_callback(ax, lead, start_ms, end_ms, x_span_ms, y_lo, y_hi)

    for row_idx, row in enumerate(CLINICAL_LAYOUT):
        for col_idx, lead in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            start_ms = col_idx * segment_ms
            end_ms = min(total_ms, (col_idx + 1) * segment_ms)
            plot_segment(ax, lead, start_ms, end_ms)

    ax_rhythm = fig.add_subplot(gs[3, :])
    plot_segment(ax_rhythm, RHYTHM_LEAD, 0.0, total_ms, show_bottom=True, show_cal=True)

    if show_title:
        title = (
            f"12-Lead ECG  |  {record_id}  |  Age {hdr.get('age', '?')}  {hdr.get('sex', '?')}  "
            f"({hdr['fs']} Hz · 25 mm/s · 10 mm/mV)"
        )
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.965)


def generate_ecg_paper_plot(
    record_id: str,
    hdr: dict,
    ecg: np.ndarray,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Render a plain clinical-style 12-lead ECG sheet without diagnosis labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    left, right = 0.03, 0.98
    top, bottom = 0.90, 0.06
    hspace, wspace = 0.06, 0.03
    fig_width, fig_height = _ecg_paper_dimensions(hdr, ecg)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(
        4, 4,
        figure=fig,
        height_ratios=[1, 1, 1, 1],
        hspace=hspace,
        wspace=wspace,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    _draw_ecg_paper_layout(fig, gs, record_id, hdr, ecg, show_title=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _add_overlay(overlays: dict[str, list[dict[str, str]]], lead: str, label: str, color: str) -> None:
    entry = {"label": label, "color": color}
    existing = overlays.setdefault(lead, [])
    if entry not in existing:
        existing.append(entry)


def generate_ecg_annotated_plot(
    record_id: str,
    hdr: dict,
    ecg: np.ndarray,
    result,
    out_path: Path,
    dpi: int = 170,
) -> None:
    """Render a 12-lead ECG with wave-boundary markers and abnormal lead highlights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fs = int(hdr["fs"])
    lead_to_idx = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    beat_features = list(getattr(result, "beat_features", []) or [])
    beats_by_lead: dict[str, list] = {lead: [] for lead in STANDARD_12_LEADS}
    for bf in beat_features:
        lead = getattr(bf, "lead", None)
        if lead in beats_by_lead:
            beats_by_lead[lead].append(bf)

    interp = getattr(result, "interpretation", None)
    overlays: dict[str, list[dict[str, str]]] = {}
    st_elevation_leads = getattr(interp, "st_elevation_leads", {}) or {}
    st_depression_leads = getattr(interp, "st_depression_leads", {}) or {}
    pathological_q_leads = getattr(interp, "pathological_q_leads", {}) or {}
    tall_t_leads = set(getattr(interp, "tall_t_leads", []) or [])
    lvh_voltage_criteria = getattr(interp, "lvh_voltage_criteria", []) or []
    lvh_class = getattr(interp, "lvh_class", None)
    r_progression_class = getattr(interp, "r_progression_class", None)

    for lead in st_elevation_leads:
        _add_overlay(overlays, lead, "ST elevation", "#E45756")
    for lead in st_depression_leads:
        _add_overlay(overlays, lead, "ST depression", "#4C78A8")
    for lead, flagged in pathological_q_leads.items():
        if flagged:
            _add_overlay(overlays, lead, "Pathological Q", "#8E63CE")
    for lead in tall_t_leads:
        _add_overlay(overlays, lead, "Tall T", "#54A24B")
    if r_progression_class in {"poor", "reverse"}:
        for lead in ("V1", "V2", "V3", "V4", "V5", "V6"):
            _add_overlay(overlays, lead, f"R progression: {r_progression_class}", "#F2A541")
    if lvh_class and lvh_voltage_criteria:
        for lead in ("I", "aVL", "V1", "V2", "V3", "V5", "V6"):
            _add_overlay(overlays, lead, f"LVH: {lvh_class}", "#D6BC2F")

    def _idx_to_panel_ms(idx) -> float | None:
        if idx is None:
            return None
        return float(idx) * 1000.0 / fs

    def _draw_annotated_panel(ax, lead: str, start_ms: float, end_ms: float, x_span_ms: float, y_lo: float, y_hi: float) -> None:
        panel_overlays = overlays.get(lead, [])
        band_height = (y_hi - y_lo) * 0.105
        for band_idx, spec in enumerate(panel_overlays[:4]):
            band_top = y_hi - band_idx * band_height
            band_bottom = band_top - band_height * 0.82
            ax.axhspan(
                band_bottom,
                band_top,
                xmin=0.0,
                xmax=1.0,
                facecolor=spec["color"],
                alpha=0.18,
                zorder=1,
            )
            ax.text(
                x_span_ms * 0.985,
                (band_top + band_bottom) / 2.0,
                spec["label"],
                ha="right",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color=spec["color"],
                zorder=9,
                clip_on=True,
            )

        def draw_boundary(idx, label: str, color: str, linestyle: str, linewidth: float = 0.9) -> None:
            abs_ms = _idx_to_panel_ms(idx)
            if abs_ms is None or abs_ms < start_ms or abs_ms > end_ms:
                return
            x_ms = abs_ms - start_ms
            ax.axvline(x_ms, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.82, zorder=8)

        for bf in beats_by_lead.get(lead, []):
            draw_boundary(bf.p.onset, "P-on", "#4C72B0", "--")
            draw_boundary(bf.p.offset, "P-off", "#4C72B0", "--")
            draw_boundary(bf.qrs.onset, "QRS-on", "#DD8452", "-")
            draw_boundary(bf.qrs.offset, "QRS-off", "#DD8452", "-")
            draw_boundary(bf.t.onset, "T-on", "#55A868", "--")
            draw_boundary(bf.t.offset, "T-end", "#C44E52", "--")
            draw_boundary(getattr(bf, "j_index", None), "J/ST", "#E45756", ":", linewidth=0.8)

            # True positive R-wave peak (matches r_amp_mv); falls back to the
            # QRS beat-detection fiducial for features generated before this
            # field existed.
            r_idx = getattr(bf, "r_peak_index", None)
            if r_idx is None:
                r_idx = getattr(getattr(bf, "qrs", None), "peak", None)
            r_abs_ms = _idx_to_panel_ms(r_idx)
            if r_abs_ms is None or r_abs_ms < start_ms or r_abs_ms > end_ms:
                continue
            lead_idx = lead_to_idx[lead]
            if 0 <= int(r_idx) < ecg.shape[1]:
                ax.scatter(
                    [r_abs_ms - start_ms],
                    [ecg[lead_idx, int(r_idx)]],
                    s=13,
                    color="#111111",
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=10,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(22.0, 15.0), dpi=dpi, facecolor="white")
    gs = GridSpec(
        4,
        4,
        figure=fig,
        height_ratios=[1, 1, 1, 1],
        hspace=0.055,
        wspace=0.028,
        left=0.035,
        right=0.965,
        top=0.91,
        bottom=0.11,
    )
    _draw_ecg_paper_layout(
        fig,
        gs,
        record_id,
        hdr,
        ecg,
        show_title=False,
        lead_fontsize=12.0,
        trace_linewidth=1.05,
        bottom_labelsize=8.5,
        preserve_paper_aspect=False,
        panel_callback=_draw_annotated_panel,
    )

    fig.suptitle(
        f"Annotated 12-Lead ECG  |  {record_id}  |  wave detection + abnormality highlights",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.03,
        0.935,
        f"{hdr['fs']} Hz | 25 mm/s | 10 mm/mV | research visualization only",
        fontsize=9,
        fontweight="bold",
        color="#444444",
    )

    legend_handles = [
        Line2D([0], [0], color="#4C72B0", linestyle="--", linewidth=1.2, label="P onset/offset"),
        Line2D([0], [0], color="#DD8452", linestyle="-", linewidth=1.4, label="QRS onset/offset"),
        Line2D([0], [0], color="#55A868", linestyle="--", linewidth=1.2, label="T onset"),
        Line2D([0], [0], color="#C44E52", linestyle="--", linewidth=1.2, label="T end"),
        Line2D([0], [0], marker="o", color="#111111", linestyle="None", markersize=4, label="R peak"),
        Line2D([0], [0], color="#E45756", linestyle=":", linewidth=1.2, label="J/ST point"),
        Patch(facecolor="#E45756", alpha=0.22, label="ST elevation lead"),
        Patch(facecolor="#4C78A8", alpha=0.22, label="ST depression lead"),
        Patch(facecolor="#8E63CE", alpha=0.22, label="Pathological Q lead"),
        Patch(facecolor="#F2A541", alpha=0.22, label="R progression abnormal"),
        Patch(facecolor="#54A24B", alpha=0.22, label="Tall T lead"),
        Patch(facecolor="#D6BC2F", alpha=0.22, label="LVH-related lead"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────
# 可读报告生成
# ──────────────────────────────────────────────

def _flag(value, lo, hi, unit="", precision=0, invert=False):
    """Return (formatted_string, status_tag) for a measurement vs reference range."""
    if value is None:
        return "N/A", "?"
    fmt = f"{value:.{precision}f}{unit}"
    if invert:
        normal = not (lo <= value <= hi)
    else:
        normal = lo <= value <= hi
    if value < lo:
        tag = "LOW " if not invert else "OK  "
    elif value > hi:
        tag = "HIGH" if not invert else "OK  "
    else:
        tag = "OK  " if not invert else "ABN "
    return fmt, tag


def _tag_symbol(tag: str) -> str:
    return {"OK  ": "✓", "LOW ": "↓", "HIGH": "↑", "ABON": "!", "ABN ": "!", "?": "?"}.get(tag, tag)


def _display_value(value, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    if isinstance(value, str) and not value.strip():
        return fallback
    return str(value)


def _join_display(values, fallback: str = "N/A") -> str:
    if not values:
        return fallback
    return ", ".join(str(item) for item in values)


def _yes_no(value) -> str:
    return "Yes" if bool(value) else "No"


def _fmt_display(value, unit: str = "", precision: int = 0) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}{unit}"
    except (TypeError, ValueError):
        return str(value)


def _measurement_row(label: str, value, unit: str, precision: int, lo, hi, ref_range: str) -> dict:
    formatted, tag = _flag(value, lo, hi, unit, precision)
    return {
        "label": label,
        "value": formatted,
        "ref_range": ref_range,
        "status": _tag_symbol(tag),
    }


def _lead_value_summary(values: dict | None, fallback: str = "none") -> str:
    if not values:
        return fallback
    return ", ".join(f"{lead}:{value:+.3f}mV" for lead, value in values.items())


def _clinical_artifact_warning(clinical: object) -> str:
    if not isinstance(clinical, dict) or not clinical:
        return ""
    stored = str(clinical.get("artifact_fingerprint") or "")
    if stored and stored != clinical_fingerprint(clinical):
        return "*** STALE ARTIFACT — REGENERATE REPORT ***"
    return ""


def _clinical_report_lines(clinical: object) -> list[str]:
    if not isinstance(clinical, dict) or not clinical:
        return []
    final = [
        item.get("statement") or item.get("statement_code")
        for item in clinical.get("final_statements", [])
        if isinstance(item, dict)
    ]
    conflicts = [
        f"{item.get('reference')}: {item.get('reason')}"
        for item in clinical.get("conflicts", [])
        if isinstance(item, dict)
    ]
    return [
        f"Unified clinical summary: {clinical.get('overall_status', 'unavailable')} [AUTHORITATIVE]",
        f"Unified final statements: {_join_display(final, 'none')}",
        f"Unified unavailable domains: {_join_display(clinical.get('unavailable_domains', []), 'none')}",
        f"Reference conflicts: {_join_display(conflicts, 'none')}",
    ]


def build_report_summary(record_id: str, hdr: dict, ecg: np.ndarray, result) -> dict:
    """Build display-safe fields shared by text and composite image reports."""
    from datetime import datetime

    gf = result.global_features
    interp = getattr(result, "interpretation", None)
    metadata = getattr(result, "metadata", {}) or {}
    clinical = metadata.get("clinical_interpretation") if isinstance(metadata, dict) else None
    clinical_final_codes = {
        str(item.get("statement_code") or "")
        for item in (
            clinical.get("final_statements", [])
            if isinstance(clinical, dict)
            else []
        )
        if isinstance(item, dict)
    }
    artifact_warning = _clinical_artifact_warning(clinical)
    input_fs = hdr.get("fs")
    output_fs = getattr(result, "fs", input_fs)
    n_samples = hdr.get("n_samples")
    if n_samples is None and hasattr(ecg, "shape") and len(ecg.shape) >= 2:
        n_samples = ecg.shape[1]
    duration_s = (float(n_samples) / float(input_fs)) if n_samples is not None and input_fs else None

    dx_codes = [str(code) for code in hdr.get("dx", [])]
    dx_details = _diagnosis_details(dx_codes)
    patient_rows = {
        "Record ID": record_id,
        "Age": _display_value(hdr.get("age")),
        "Sex": _display_value(hdr.get("sex")),
        "Dx Codes": _join_display(dx_codes),
        "Dx Detail": "; ".join(dx_details) if dx_details else "N/A",
        "Rx": _display_value(hdr.get("rx", "N/A")),
        "Hx": _display_value(hdr.get("hx", "N/A")),
        "Sx": _display_value(hdr.get("sx", "N/A")),
        "Recording": (
            f"{duration_s:.1f} s @ {input_fs} Hz -> resampled to {output_fs} Hz"
            if duration_s is not None and input_fs
            else "N/A"
        ),
    }

    measurements = [
        _measurement_row("Heart Rate", gf.heart_rate_bpm, " bpm", 0, 60, 100, "60-100 bpm"),
        _measurement_row("PR Interval", gf.pr_ms, " ms", 0, 120, 200, "120-200 ms"),
        _measurement_row("QRS Duration", gf.qrs_ms, " ms", 0, 60, 120, "60-120 ms"),
        _measurement_row("QT Interval", gf.qt_ms, " ms", 0, 300, 470, "300-470 ms"),
        _measurement_row("QTc Bazett", gf.qtc_bazett_ms, " ms", 0, 360, 440, "360-440 ms"),
        _measurement_row("QTc Fridericia", gf.qtc_fridericia_ms, " ms", 0, 360, 440, "360-440 ms"),
        _measurement_row("QT Dispersion", gf.qt_dispersion_ms, " ms", 0, 0, 60, "<60 ms"),
        _measurement_row("P Axis", gf.p_axis_deg, "°", 0, 0, 75, "0-75°"),
        _measurement_row("QRS Axis", gf.qrs_axis_deg, "°", 0, -30, 90, "-30-90°"),
        _measurement_row("T Axis", gf.t_axis_deg, "°", 0, 0, 75, "0-75°"),
    ]

    rr_vals = [b.rr_prev_ms for b in getattr(result, "beats", []) if getattr(b, "rr_prev_ms", None) is not None]
    if rr_vals:
        mean_rr = float(np.mean(rr_vals))
        rhythm_detail = f"mean RR {mean_rr:.0f} ms, SD {float(np.std(rr_vals)):.0f} ms"
    else:
        rhythm_detail = "RR unavailable"

    if interp is None:
        interpretation_lines = ["Interpretation unavailable"]
    else:
        dxl_af_display, dxl_af_resolution = _dxl_af_reference_display(
            bool(getattr(interp, "probable_af", False)), clinical_final_codes
        )
        path_q_leads = [lead for lead, flagged in getattr(interp, "pathological_q_leads", {}).items() if flagged]
        interpretation_lines = [
            (
                f"Rhythm: {getattr(interp, 'heart_rate_class', 'N/A')}; "
                f"RR {getattr(interp, 'rr_irregularity_class', 'N/A')} ({rhythm_detail}); "
                f"DXL-reference AF probable {dxl_af_display}"
            ),
            (
                f"Conduction: PR {getattr(interp, 'pr_class', 'N/A')}; "
                f"QRS {getattr(interp, 'qrs_width_class', 'N/A')}; "
                f"BBB {_display_value(getattr(interp, 'bundle_branch_block', None), 'none')}; "
                f"WPW {_yes_no(getattr(interp, 'wpw_pattern', False))}"
            ),
            (
                f"P morphology: {_display_value(getattr(interp, 'p_morphology_class', None), 'N/A')}; "
                f"RAE leads {_join_display(getattr(interp, 'rae_leads', []), 'none')}; "
                f"LAE {_yes_no(getattr(interp, 'lae_suspected', False))}"
            ),
            (
                f"Q waves: {_join_display(getattr(interp, 'q_wave_territories', []), 'none')}; "
                f"leads {_join_display(path_q_leads, 'none')}"
            ),
            (
                f"R progression: {_display_value(getattr(interp, 'r_progression_class', None), 'N/A')}"
                + (
                f" (transition {getattr(interp, 'r_s_transition_lead')})"
                if getattr(interp, "r_s_transition_lead", None)
                else ""
                )
            ),
            (
                f"ST elevation: territories {_join_display(getattr(interp, 'st_territories_elevated', []), 'none')}; "
                f"leads {_lead_value_summary(getattr(interp, 'st_elevation_leads', {}))}"
            ),
            (
                f"ST depression: territories {_join_display(getattr(interp, 'st_territories_depressed', []), 'none')}; "
                f"leads {_lead_value_summary(getattr(interp, 'st_depression_leads', {}))}"
            ),
            (
                f"Reciprocal change: {_yes_no(getattr(interp, 'reciprocal_change_detected', False))}"
            ),
            (
                f"Hypertrophy: LVH {_display_value(getattr(interp, 'lvh_class', None), 'none')} "
                f"({_join_display(getattr(interp, 'lvh_voltage_criteria', []), 'no criteria')}); "
                f"RVH {_yes_no(getattr(interp, 'rvh_suspected', False))}"
            ),
        ]
        if dxl_af_resolution:
            interpretation_lines.insert(1, f"DXL reference resolution: {dxl_af_resolution}")

    clinical_lines = _clinical_report_lines(clinical)
    if clinical_lines:
        if interp is not None:
            interpretation_lines = [
                f"DXL-inspired reference — {line} [APPROXIMATION: EXISTING DXL]"
                for line in interpretation_lines
            ]
        interpretation_lines = clinical_lines + interpretation_lines
    if artifact_warning:
        interpretation_lines = [artifact_warning] + interpretation_lines

    return {
        "record_id": record_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_rows": patient_rows,
        "measurements": measurements,
        "interpretation_lines": interpretation_lines,
        "artifact_warning": artifact_warning,
        "paper_speed": "25 mm/s",
        "gain_label": "10 mm/mV",
        "review_disclaimer": (
            "Research algorithm output only. Not validated for clinical use. "
            "Do not use for medical decisions."
        ),
    }


def _panel_line(label: str, value: str) -> str:
    return f"{label}: {value}"


def _draw_report_panel(
    ax,
    title: str,
    lines: list[str],
    wrap_width: int = 48,
    *,
    title_fontsize: float = 13.0,
    body_fontsize: float = 10.6,
    body_fontweight: str = "bold",
    line_step: float = 0.076,
    paragraph_gap: float = 0.010,
) -> None:
    from matplotlib.patches import Rectangle

    ax.set_axis_off()
    ax.set_facecolor("#FFFFFF")
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor="#FFFFFF",
            edgecolor="#C8C8C8",
            linewidth=0.9,
        )
    )
    ax.text(
        0.035,
        0.94,
        title.upper(),
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        color="#222222",
        va="top",
    )
    y = 0.82
    for raw_line in lines:
        wrapped = textwrap.wrap(str(raw_line), width=wrap_width) or [""]
        for idx, line in enumerate(wrapped):
            ax.text(
                0.035,
                y,
                line if idx == 0 else f"  {line}",
                transform=ax.transAxes,
                fontsize=body_fontsize,
                fontweight=body_fontweight,
                color="#333333",
                va="top",
                clip_on=True,
            )
            y -= line_step
        y -= paragraph_gap
        if y < 0.045:
            break


def generate_ecg_report_sheet(
    record_id: str,
    hdr: dict,
    ecg: np.ndarray,
    result,
    out_path: Path,
) -> None:
    """Render a single-page ECG report sheet with waveform and key summary panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    summary = build_report_summary(record_id, hdr, ecg, result)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22.0, 16.4), dpi=220, facecolor="white")
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[0.28, 3.12, 1.95, 0.18],
        hspace=0.10,
        wspace=0.045,
        left=0.035,
        right=0.965,
        top=0.975,
        bottom=0.035,
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_axis_off()
    ax_header.text(
        0.0,
        0.82,
        "ECG MEASUREMENT REPORT",
        transform=ax_header.transAxes,
        fontsize=24,
        fontweight="bold",
        color="#1F1F1F",
        va="top",
    )
    ax_header.text(
        0.0,
        0.24,
        f"Record ID: {summary['record_id']}    Generated: {summary['generated_at']}",
        transform=ax_header.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#444444",
        va="top",
    )
    ax_header.text(
        1.0,
        0.24,
        f"{summary['paper_speed']} | {summary['gain_label']}",
        transform=ax_header.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#444444",
        ha="right",
        va="top",
    )

    waveform_gs = GridSpecFromSubplotSpec(
        4,
        4,
        subplot_spec=gs[1, :],
        height_ratios=[1, 1, 1, 1],
        hspace=0.055,
        wspace=0.028,
    )
    _draw_ecg_paper_layout(
        fig,
        waveform_gs,
        record_id,
        hdr,
        ecg,
        show_title=False,
        lead_fontsize=13.5,
        trace_linewidth=1.2,
        bottom_labelsize=10.5,
        preserve_paper_aspect=False,
    )

    patient_lines = [_panel_line(k, v) for k, v in summary["patient_rows"].items()]
    measurement_lines = [
        f"{row['label']:<15} {row['value']:<10} {row['status']}  ({row['ref_range']})"
        for row in summary["measurements"]
    ]
    interpretation_lines = list(summary["interpretation_lines"])

    _draw_report_panel(
        fig.add_subplot(gs[2, 0]),
        "Patient / Recording",
        patient_lines,
        wrap_width=56,
        line_step=0.058,
    )
    _draw_report_panel(
        fig.add_subplot(gs[2, 1]),
        "Key Measurements",
        measurement_lines,
        wrap_width=54,
        line_step=0.058,
    )
    _draw_report_panel(
        fig.add_subplot(gs[2, 2]),
        "Interpretation Summary",
        interpretation_lines,
        wrap_width=58,
        line_step=0.054,
    )

    ax_footer = fig.add_subplot(gs[3, :])
    ax_footer.set_axis_off()
    ax_footer.text(
        0.0,
        0.6,
        summary["review_disclaimer"],
        transform=ax_footer.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#555555",
        va="center",
    )
    ax_footer.text(
        1.0,
        0.6,
        "ECG Feature Extractor | Unified clinical rules",
        transform=ax_footer.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#555555",
        ha="right",
        va="center",
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_report(
    record_id: str,
    hdr: dict,
    ecg: np.ndarray,
    result,
    out_path: Path,
) -> None:
    """Write a human-readable clinical-style ECG report to out_path."""
    from datetime import datetime
    lines = []
    W = 66

    def h1(title):
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    def h2(title):
        lines.append("")
        lines.append(f"  {'─' * 3}  {title}  {'─' * max(1, W - len(title) - 8)}")

    def row(*cols, widths=None):
        if widths:
            parts = [str(c).ljust(w) for c, w in zip(cols, widths)]
        else:
            parts = [str(c) for c in cols]
        lines.append("  " + "  ".join(parts))

    def blank():
        lines.append("")

    def row_wrapped(label: str, value: str, label_width: int = 13, value_width: int | None = None) -> None:
        if value_width is None:
            value_width = W - 6 - 2 - 2 - 1 - label_width
        wrapped = textwrap.wrap(str(value), width=max(20, value_width)) or [""]
        prefix = f"{label:<{label_width}} : "
        indent = " " * (label_width + 3)
        lines.append("  " + prefix + wrapped[0])
        for extra in wrapped[1:]:
            lines.append("  " + indent + extra)

    gf = result.global_features
    metadata = getattr(result, "metadata", {}) or {}
    clinical = metadata.get("clinical_interpretation") if isinstance(metadata, dict) else None
    clinical_final_codes = {
        str(item.get("statement_code") or "")
        for item in (
            clinical.get("final_statements", [])
            if isinstance(clinical, dict)
            else []
        )
        if isinstance(item, dict)
    }
    artifact_warning = _clinical_artifact_warning(clinical)
    fs = result.fs
    beats = result.beats
    rr_vals = [b.rr_prev_ms for b in beats if b.rr_prev_ms is not None]
    dx_codes = hdr.get("dx", [])
    dx_details = _diagnosis_details(dx_codes)

    # ── Header ──────────────────────────────────────────────────────────────
    if artifact_warning:
        lines.append(artifact_warning)
    h1(f"ECG MEASUREMENT REPORT  —  {record_id}")
    row(f"Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    row("Algorithm : Public-guideline unified rules + reference algorithms  (research use only)")
    blank()

    # ── Patient ──────────────────────────────────────────────────────────────
    h2("PATIENT INFORMATION")
    row_wrapped("Record ID", record_id)
    row_wrapped("Age", str(hdr.get("age", "N/A")))
    row_wrapped("Sex", str(hdr.get("sex", "N/A")))
    row_wrapped("Dx Codes", ", ".join(dx_codes) or "N/A")
    row_wrapped("Dx Detail", "; ".join(dx_details) or "N/A")
    row_wrapped("Rx", str(hdr.get("rx", "N/A")))
    row_wrapped("Hx", str(hdr.get("hx", "N/A")))
    row_wrapped("Sx", str(hdr.get("sx", "N/A")))
    row_wrapped("Recording", f"{ecg.shape[1] / hdr['fs']:.1f} s  @  {hdr['fs']} Hz  →  resampled to {fs} Hz")

    # ── Global Measurements ──────────────────────────────────────────────────
    h2("GLOBAL MEASUREMENTS")
    blank()
    row("Measurement", "Value", "Ref Range", "Status", widths=[24, 12, 18, 6])
    row("─" * 24, "─" * 12, "─" * 18, "─" * 6, widths=[24, 12, 18, 6])

    hr = gf.heart_rate_bpm
    hr_str, hr_tag = _flag(hr, 60, 100, " bpm", 0)
    row("Heart Rate", hr_str, "60 – 100 bpm", _tag_symbol(hr_tag), widths=[24, 12, 18, 6])

    pr_str, pr_tag = _flag(gf.pr_ms, 120, 200, " ms", 0)
    row("PR Interval", pr_str, "120 – 200 ms", _tag_symbol(pr_tag), widths=[24, 12, 18, 6])

    qrs_str, qrs_tag = _flag(gf.qrs_ms, 60, 120, " ms", 0)
    row("QRS Duration", qrs_str, "60 – 120 ms", _tag_symbol(qrs_tag), widths=[24, 12, 18, 6])

    qt_str, qt_tag = _flag(gf.qt_ms, 300, 470, " ms", 0)
    row("QT Interval", qt_str, "300 – 470 ms", _tag_symbol(qt_tag), widths=[24, 12, 18, 6])

    qtcb_str, qtcb_tag = _flag(gf.qtc_bazett_ms, 360, 440, " ms", 0)
    row("QTc Bazett", qtcb_str, "360 – 440 ms", _tag_symbol(qtcb_tag), widths=[24, 12, 18, 6])

    qtcf_str, qtcf_tag = _flag(gf.qtc_fridericia_ms, 360, 440, " ms", 0)
    row("QTc Fridericia", qtcf_str, "360 – 440 ms", _tag_symbol(qtcf_tag), widths=[24, 12, 18, 6])

    qtd_str, qtd_tag = _flag(gf.qt_dispersion_ms, 0, 60, " ms", 0)
    row("QT Dispersion", qtd_str, "< 60 ms", _tag_symbol(qtd_tag), widths=[24, 12, 18, 6])

    qt_source = getattr(gf, "qt_source", None) or "unknown"
    qt_reliability = getattr(gf, "qt_reliability", None) or "unknown"
    qt_reportable = bool(getattr(gf, "qt_reportable", False))
    qt_unreliable_reasons = list(
        getattr(gf, "qt_unreliable_reasons", []) or []
    )
    qt_used_leads = getattr(gf, "qt_used_leads", None) or []
    qt_used_text = ", ".join(str(lead) for lead in qt_used_leads) if qt_used_leads else "beat-level/no lead list"
    row_wrapped(
        "Global QT source",
        f"{qt_source} | reliability={qt_reliability} | "
        f"reportable={'yes' if qt_reportable else 'no'} | "
        f"used leads={qt_used_text}",
        label_width=16,
    )
    if qt_unreliable_reasons:
        row_wrapped(
            "QT reliability QC",
            ", ".join(str(reason) for reason in qt_unreliable_reasons),
            label_width=16,
        )

    blank()
    row("Measurement", "Value", "Ref Range", "Status", widths=[24, 12, 18, 6])
    row("─" * 24, "─" * 12, "─" * 18, "─" * 6, widths=[24, 12, 18, 6])

    p_ax_str, p_ax_tag = _flag(gf.p_axis_deg, 0, 75, "°", 0)
    row("P Axis", p_ax_str, "0 – 75°", _tag_symbol(p_ax_tag), widths=[24, 12, 18, 6])

    q_ax_str, q_ax_tag = _flag(gf.qrs_axis_deg, -30, 90, "°", 0)
    row("QRS Axis", q_ax_str, "-30 – 90°", _tag_symbol(q_ax_tag), widths=[24, 12, 18, 6])

    t_ax_str, t_ax_tag = _flag(gf.t_axis_deg, 0, 75, "°", 0)
    row("T Axis", t_ax_str, "0 – 75°", _tag_symbol(t_ax_tag), widths=[24, 12, 18, 6])

    blank()
    row("  ✓ = within reference range   ↑ = above range   ↓ = below range")

    # ── Authoritative unified interpretation ────────────────────────────────
    if isinstance(clinical, dict) and clinical:
        h2("UNIFIED CLINICAL INTERPRETATION  [AUTHORITATIVE RULE OUTPUT]")
        row(f"Overall status : {clinical.get('overall_status', 'unavailable')}")
        row(f"Ruleset        : {clinical.get('ruleset_version', 'N/A')}")
        fingerprint = clinical.get("artifact_fingerprint") or clinical_fingerprint(clinical)
        # Keep the complete hash on one line so automated artifact checks can
        # compare the JSON and text report byte-for-byte.
        row(f"Fingerprint    : {fingerprint}")
        final_rows = [
            item for item in clinical.get("final_statements", [])
            if isinstance(item, dict)
        ]
        if final_rows:
            for item in final_rows:
                row_wrapped(
                    str(item.get("rule_id") or item.get("statement_code") or "rule"),
                    str(item.get("statement") or item.get("statement_code") or "matched"),
                    label_width=20,
                )
        else:
            row("Final statements: none")
        row_wrapped(
            "Unavailable",
            _join_display(clinical.get("unavailable_domains", []), "none"),
            label_width=13,
        )
        for conflict in clinical.get("conflicts", []):
            if isinstance(conflict, dict):
                row_wrapped(
                    "Ref conflict",
                    f"{conflict.get('reference')}: {conflict.get('reason')}",
                    label_width=13,
                )

    # ── RR / Rhythm summary ──────────────────────────────────────────────────
    h2("RHYTHM SUMMARY")
    n_beats = len(beats)
    row(f"Total beats detected : {n_beats}")
    if rr_vals:
        mean_rr = float(np.mean(rr_vals))
        sd_rr   = float(np.std(rr_vals))
        row(f"Mean RR interval     : {mean_rr:.0f} ms  ({60000/mean_rr:.0f} bpm)")
        row(f"RR std deviation     : {sd_rr:.0f} ms  ({'regular' if sd_rr < 80 else 'irregular'})")
        row(f"RR range             : {min(rr_vals):.0f} – {max(rr_vals):.0f} ms")

    rhythm_analysis = (
        result.metadata.get("rhythm_analysis", {})
        if isinstance(getattr(result, "metadata", None), dict)
        else {}
    )
    af_afl = rhythm_analysis.get("af_afl_summary", {})
    atrial_residual = rhythm_analysis.get("atrial_residual", {})
    if isinstance(af_afl, dict) and af_afl:
        classification = af_afl.get("atrial_rhythm_classification", "none")
        row(f"Atrial classification : {classification}")
        row(
            "AF/AFL evidence       : "
            f"RR-CV={_fmt(af_afl.get('rr_cv'), '', 3)}, "
            f"organized-P={_fmt(af_afl.get('organized_p_ratio'), '', 3)}, "
            f"f={_fmt(af_afl.get('f_wave_confidence'), '', 3)}, "
            f"F={_fmt(af_afl.get('F_wave_confidence'), '', 3)}"
        )
        if isinstance(atrial_residual, dict):
            row(
                "F-wave validation   : "
                f"QRST={'validated' if atrial_residual.get('validated_qrst_subtraction') else 'unvalidated'}, "
                f"rate={_fmt(atrial_residual.get('F_wave_rate_bpm'), ' bpm', 0)}, "
                f"leads={_join_display(atrial_residual.get('flutter_supporting_leads', []), 'none')}"
            )
        reasons = af_afl.get("indeterminate_reasons", [])
        if reasons:
            row_wrapped(
                "AF/AFL uncertain",
                _join_display(reasons, "unspecified"),
                label_width=17,
            )

    if result.groups:
        blank()
        row("Group", "Beats", "Share", "Mean RR", "QRS dur", "Dominant", widths=[6, 6, 7, 10, 9, 10])
        row("─"*6, "─"*6, "─"*7, "─"*10, "─"*9, "─"*10, widths=[6, 6, 7, 10, 9, 10])
        for gid, g in result.groups.items():
            dom = "Yes" if g.flags.get("dominant_group") else "No"
            row(
                str(gid),
                str(g.member_count),
                f"{g.member_pct:.0f}%",
                _fmt(g.mean_rr_ms, " ms", 0),
                _fmt(g.mean_qrs_ms, " ms", 0),
                dom,
                widths=[6, 6, 7, 10, 9, 10],
            )

    # ── Signal Quality ───────────────────────────────────────────────────────
    h2("SIGNAL QUALITY  (per lead)")
    blank()
    row("Lead", "Overall", "Baseline", "Muscle", "Powerline", "P-ok", "QRS-ok", "T-ok",
        widths=[5, 8, 9, 7, 10, 5, 7, 5])
    row("─"*5, "─"*8, "─"*9, "─"*7, "─"*10, "─"*5, "─"*7, "─"*5,
        widths=[5, 8, 9, 7, 10, 5, 7, 5])
    for lead in STANDARD_12_LEADS:
        lq = result.quality[lead]
        ok  = "✓ OK" if lq.reliable else "✗ BAD"
        p_ok  = "✓" if lq.reliable_for_p   else "✗"
        q_ok  = "✓" if lq.reliable_for_qrs else "✗"
        t_ok  = "✓" if lq.reliable_for_t   else "✗"
        row(
            lead, ok,
            f"{lq.baseline_wander_score:.3f}",
            f"{lq.muscle_noise_score:.3f}",
            f"{lq.powerline_score:.3f}",
            p_ok, q_ok, t_ok,
            widths=[5, 8, 9, 7, 10, 5, 7, 5],
        )

    # ── Per-lead measurements ─────────────────────────────────────────────────
    h2("PER-LEAD MEASUREMENTS  (raw representative; global QT uses robust aggregation)")
    blank()
    row("Lead", "PR ms", "QRS ms", "QT ms", "QTc-B", "R mV", "T mV", "ST-J mV", "QT-conf",
        widths=[5, 7, 7, 7, 7, 7, 7, 9, 8])
    row("─"*5, "─"*7, "─"*7, "─"*7, "─"*7, "─"*7, "─"*7, "─"*9, "─"*8,
        widths=[5, 7, 7, 7, 7, 7, 7, 9, 8])
    rr_sec = float(np.median(rr_vals) / 1000.0) if rr_vals else None
    reliable_qt = result.metadata.get("reliable_qt_leads", [])
    mask_per_lead_intervals = (
        not reliable_qt
        and (
            getattr(gf, "qt_path", None) == "low_qt_support"
            or getattr(gf, "qt_reliability", None) in {"low_confidence", "unavailable"}
        )
    )
    for lead in STANDARD_12_LEADS:
        p = result.representative_leads[lead].params
        qt = None if mask_per_lead_intervals else p.get("qt_ms")
        qtcb = (qt / np.sqrt(rr_sec)) if (qt and rr_sec) else None
        conf = p.get("qt_confidence_mean")
        row(
            lead,
            _fmt(None if mask_per_lead_intervals else p.get("pr_ms"), "", 0),
            _fmt(None if mask_per_lead_intervals else p.get("qrs_ms"), "", 0),
            _fmt(qt, "", 0),
            _fmt(qtcb, "", 0),
            _fmt(p.get("r_amp_mv"), "", 3),
            _fmt(p.get("t_amp_mv"), "", 3),
            _fmt(p.get("st_on_mv"), "", 3),
            "N/A" if mask_per_lead_intervals else (_fmt(conf, "", 2) if conf is not None else "N/A"),
            widths=[5, 7, 7, 7, 7, 7, 7, 9, 8],
        )

    # ── Beat-level quality summary ────────────────────────────────────────────
    h2("BEAT-LEVEL QUALITY SUMMARY  (T004)")
    total_bf = len(result.beat_features)
    n_unrel  = sum(1 for b in result.beat_features if not b.beat_measurement_reliable)
    n_pfail  = sum(1 for b in result.beat_features if "p_unreliable"  in b.flags)
    n_tfail  = sum(1 for b in result.beat_features if "t_unreliable"  in b.flags)
    n_qfail  = sum(1 for b in result.beat_features if "qrs_unreliable" in b.flags)
    blank()
    row(f"Total beat×lead entries  : {total_bf}")
    row(f"Unreliable entries       : {n_unrel}  ({100*n_unrel/max(1,total_bf):.1f}%)")
    row(f"P-wave undetected        : {n_pfail}  ({100*n_pfail/max(1,total_bf):.1f}%)")
    row(f"T-wave undetected        : {n_tfail}  ({100*n_tfail/max(1,total_bf):.1f}%)")
    row(f"QRS bounds failed        : {n_qfail}  ({100*n_qfail/max(1,total_bf):.1f}%)")
    blank()
    row(f"Reliable-QT leads        : {', '.join(reliable_qt) if reliable_qt else 'none'}")
    row(f"  (used for global QT / QTc computation)")

    # ── Key findings ─────────────────────────────────────────────────────────
    h2("DXL-INSPIRED REFERENCE FINDINGS  (non-validated — research use only)")
    blank()
    findings = []
    interp = result.interpretation

    if hr is not None:
        if hr > 100:  findings.append(f"• Tachycardia  (HR = {hr:.0f} bpm)")
        elif hr < 60: findings.append(f"• Bradycardia  (HR = {hr:.0f} bpm)")

    if gf.pr_ms is not None:
        if gf.pr_ms > 200: findings.append(f"• Prolonged PR interval  ({gf.pr_ms:.0f} ms > 200 ms)")
        elif gf.pr_ms < 120: findings.append(f"• Short PR interval  ({gf.pr_ms:.0f} ms < 120 ms)")

    if gf.qrs_ms is not None:
        if gf.qrs_ms >= 120: findings.append(f"• Wide QRS complex  ({gf.qrs_ms:.0f} ms ≥ 120 ms)")

    # Deriving prolongation straight off the raw QTcB value would contradict
    # Step 3 below, which suppresses the prolongation statement when a
    # confirmed VCD/RVH/LVH confounder is present (DXL rule). Defer to
    # interp.qtc_class so both sections agree.
    qtc_class = interp.qtc_class if interp is not None else None
    if gf.qtc_bazett_ms is not None and reliable_qt:
        if qtc_class == "significantly_prolonged":
            findings.append(f"• CRITICAL: Very prolonged QTcB  ({gf.qtc_bazett_ms:.0f} ms)")
        elif qtc_class == "prolonged":
            findings.append(f"• Prolonged QTcB  ({gf.qtc_bazett_ms:.0f} ms)")
        elif qtc_class is None and gf.qtc_bazett_ms > 500:
            findings.append(f"• CRITICAL: Very prolonged QTcB  ({gf.qtc_bazett_ms:.0f} ms)")
        elif qtc_class is None and gf.qtc_bazett_ms > 440:
            findings.append(f"• Prolonged QTcB  ({gf.qtc_bazett_ms:.0f} ms)")
        elif gf.qtc_bazett_ms < 340:
            findings.append(f"• Short QTcB  ({gf.qtc_bazett_ms:.0f} ms)")

    if gf.qrs_axis_deg is not None:
        if gf.qrs_axis_deg > 90:  findings.append(f"• Right axis deviation  (QRS axis {gf.qrs_axis_deg:.0f}°)")
        elif gf.qrs_axis_deg < -30: findings.append(f"• Left axis deviation  (QRS axis {gf.qrs_axis_deg:.0f}°)")

    if rr_vals and float(np.std(rr_vals)) > 150:
        findings.append(f"• Irregular rhythm  (RR SD = {float(np.std(rr_vals)):.0f} ms)")

    if len(result.groups) > 1:
        for gid, g in result.groups.items():
            if not g.flags.get("dominant_group") and g.member_pct >= 5:
                findings.append(f"• Ectopic beat group {gid}  ({g.member_count} beats, {g.member_pct:.0f}%)")

    if not reliable_qt:
        findings.append("• WARNING: No reliable-QT leads found — global QT may be inaccurate")

    if findings:
        for f_line in findings:
            row(f_line)
    else:
        row("• No significant automated findings.")

    # ── Clinical Interpretation ───────────────────────────────────────────────
    if interp is not None:
        h2("DXL-INSPIRED REFERENCE INTERPRETATION  (research use only)")
        blank()

        def _yn(b):
            return "Yes" if b else "No"

        def _lst(lst):
            return ", ".join(lst) if lst else "—"

        def _leads_with_val(d):
            if not d:
                return "—"
            return "  ".join(f"{k}:{v:+.3f}mV" for k, v in d.items())

        # ── Step 1: Technical quality ─────────────────────────────────────────
        row("  ┌─ Step 1: Technical Quality")
        if interp.limb_reversal_suspected:
            row(f"  │  Limb lead reversal  : {interp.limb_reversal_suspected}")
        else:
            row("  │  Limb lead reversal  : Not suspected")
        row(f"  └  Precordial reversal : {_yn(interp.precordial_reversal_suspected)}")
        blank()

        # ── Step 2: Rhythm & rate ─────────────────────────────────────────────
        row("  ┌─ Step 2: Rhythm & Rate")
        row(f"  │  Heart rate class    : {interp.heart_rate_class}  ({_fmt(gf.heart_rate_bpm, ' bpm', 0)})")
        row(f"  │  RR irregularity     : {interp.rr_irregularity_class}  (CV = {interp.rr_cv})")
        dxl_af_display, dxl_af_resolution = _dxl_af_reference_display(
            bool(interp.probable_af), clinical_final_codes
        )
        row(f"  └  Probable AF         : {dxl_af_display}")
        if dxl_af_resolution:
            row(f"     Resolution          : {dxl_af_resolution}")
        blank()

        # ── Step 3: Axis & intervals ──────────────────────────────────────────
        row("  ┌─ Step 3: Axis & Intervals")
        row(f"  │  P axis              : {_fmt(gf.p_axis_deg, '°', 0)}  →  sinus range: {_yn(interp.p_axis_normal)}")
        row(f"  │  QRS axis            : {_fmt(gf.qrs_axis_deg, '°', 0)}  →  {interp.qrs_axis_class}")
        row(f"  │  T axis              : {_fmt(gf.t_axis_deg, '°', 0)}  →  {interp.t_axis_class}  (QRS-T angle: {interp.qrs_t_angle_deg}°)")
        row(f"  │  PR interval         : {_fmt(gf.pr_ms, ' ms', 0)}  →  {interp.pr_class}"
            + (f"  [AVB grade {interp.avb_grade}]" if interp.avb_grade else ""))
        row(f"  │  QRS width           : {_fmt(gf.qrs_ms, ' ms', 0)}  →  {interp.qrs_width_class}"
            + (f"  [{interp.bundle_branch_block}]" if interp.bundle_branch_block else ""))
        row(f"  │  QTc (Bazett)        : {_fmt(gf.qtc_bazett_ms, ' ms', 0)}  →  {interp.qtc_class}")
        row(f"  └  WPW pattern         : {_yn(interp.wpw_pattern)}")
        blank()

        # ── Step 4: Wave morphology ───────────────────────────────────────────
        row("  ┌─ Step 4: Wave Morphology")
        row(f"  │  P morphology        : {interp.p_morphology_class or '—'}")
        row(f"  │  RAE leads (P≥0.24mV): {_lst(interp.rae_leads)}")
        row(f"  │  LAE suspected       : {_yn(interp.lae_suspected)}  |  Definite: {_yn(interp.lae_definite)}")
        row(f"  │  PTF-V1 class        : {interp.ptf_v1_class or '—'}  ({_fmt(gf.ptf_v1_mv_ms, ' mV·ms', 2)})")
        blank()
        path_q_leads = [l for l, v in interp.pathological_q_leads.items() if v]
        row(f"  │  Pathological Q leads: {_lst(path_q_leads)}")
        row(f"  │  Q wave territories  : {_lst(interp.q_wave_territories)}")
        blank()
        row(f"  └  R progression      : {interp.r_progression_class}"
            + (f"  (RS transition: {interp.r_s_transition_lead})" if interp.r_s_transition_lead else ""))
        blank()

        # ── Step 5: Anatomical localisation ──────────────────────────────────
        row("  ┌─ Step 5: Anatomical Localisation & ST")
        if interp.st_elevation_leads:
            row(f"  │  ST elevation leads  : {_leads_with_val(interp.st_elevation_leads)}")
            row(f"  │  ST elevated terr.   : {_lst(interp.st_territories_elevated)}")
        else:
            row("  │  ST elevation        : None significant")
        if interp.st_depression_leads:
            row(f"  │  ST depression leads : {_leads_with_val(interp.st_depression_leads)}")
            row(f"  │  ST depressed terr.  : {_lst(interp.st_territories_depressed)}")
        else:
            row("  │  ST depression       : None significant")
        if interp.stemi_suspected_codes:
            row(f"  │  *** STEMI codes     : {_lst(interp.stemi_suspected_codes)} ***")
        row(f"  │  Reciprocal change   : {_yn(interp.reciprocal_change_detected)}")
        if interp.reciprocal_pairs:
            for pair in interp.reciprocal_pairs:
                row(f"  │    {pair[0]}  ↔  {pair[1]}")
        blank()
        row(f"  │  Tall T leads        : {_lst(interp.tall_t_leads)}")
        blank()
        row(f"  │  LVH criteria        : {_lst(interp.lvh_voltage_criteria)}")
        row(f"  │  LVH class           : {interp.lvh_class or '—'}")
        row(f"  │  Low voltage         : {interp.low_voltage_class or '—'}")
        row(f"  └  RVH suspected       : {_yn(interp.rvh_suspected)}")

    blank()
    lines.append("─" * W)
    lines.append("  NOTE: This report is generated by a research algorithm and has NOT")
    lines.append("  been validated for clinical use. Do not use for medical decisions.")
    lines.append("─" * W)
    blank()

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main(record_id: str, include_beat_features: bool = False) -> None:
    hea_path = DATASET_DIR / f"{record_id}.hea"
    mat_path = DATASET_DIR / f"{record_id}.mat"

    if not hea_path.exists():
        sys.exit(f"[错误] 找不到文件: {hea_path}")

    print("=" * 62)
    print(f"   ECG 特征提取 Demo  —  {record_id}")
    print("=" * 62)

    # ── 1. 加载数据 ──────────────────────────────
    hdr = parse_hea(hea_path)
    fs  = hdr["fs"]
    ecg = load_ecg(mat_path)

    _sep("患者信息")
    print(f"  Record  : {record_id}")
    print(f"  年龄    : {hdr.get('age', 'N/A')} 岁")
    print(f"  性别    : {hdr.get('sex', 'N/A')}")
    print(f"  诊断码  : {', '.join(hdr.get('dx', []))}")
    print(f"  用药    : {hdr.get('rx', 'Unknown')}")

    _sep("原始信号")
    print(f"  Shape   : {ecg.shape}  (导联 × 采样点)")
    print(f"  采样率  : {fs} Hz")
    print(f"  时长    : {hdr['n_samples'] / fs:.1f} s")
    print(f"  幅值范围: [{ecg.min():.3f}, {ecg.max():.3f}] mV")

    # ── 2. 构建患者元信息 ─────────────────────────
    patient_meta = PatientMeta(
        age=hdr.get("age"),
        sex=hdr.get("sex"),
    )

    # ── 3. 运行特征提取器 ─────────────────────────
    _sep("运行特征提取流水线")
    extractor = ECGFeatureExtractor(mains_freq=50)
    result = extractor.extract(ecg, fs=fs, meta=patient_meta)
    print("  完成。")

    # ── 4. 全局测量值 ─────────────────────────────
    gf = result.global_features
    _sep("全局测量值 (Global Features)")
    print(f"  心率 (HR)         : {_fmt(gf.heart_rate_bpm, ' bpm')}")
    print(f"  心房率            : {_fmt(gf.atrial_rate_bpm, ' bpm')}")
    print(f"  PR 间期           : {_fmt(gf.pr_ms, ' ms', 0)}")
    print(f"  QRS 时限          : {_fmt(gf.qrs_ms, ' ms', 0)}")
    print(f"  QT 间期           : {_fmt(gf.qt_ms, ' ms', 0)}")
    print(f"  QTc (Bazett)      : {_fmt(gf.qtc_bazett_ms, ' ms', 0)}")
    print(f"  QTc (Fridericia)  : {_fmt(gf.qtc_fridericia_ms, ' ms', 0)}")
    print(f"  QT 离散度         : {_fmt(gf.qt_dispersion_ms, ' ms', 0)}")
    print(f"  P  轴             : {_fmt(gf.p_axis_deg, '°', 0)}")
    print(f"  QRS 轴            : {_fmt(gf.qrs_axis_deg, '°', 0)}")
    print(f"  T  轴             : {_fmt(gf.t_axis_deg, '°', 0)}")
    print(f"  ST 轴             : {_fmt(gf.st_axis_deg, '°', 0)}")

    # ── 5. 信号质量 ───────────────────────────────
    _sep("信号质量 (每导联)")
    hdr_line = f"  {'导联':>4s}  {'可用':^4s}  {'基线漂移':>8s}  {'肌电':>6s}  {'工频':>6s}  标记"
    print(hdr_line)
    for lead in STANDARD_12_LEADS:
        lq = result.quality[lead]
        ok = "✓" if lq.reliable else "✗"
        flags_str = " ".join(lq.flags) if lq.flags else "—"
        print(
            f"  {lead:>4s}  {ok:^4s}  "
            f"{lq.baseline_wander_score:>8.3f}  "
            f"{lq.muscle_noise_score:>6.3f}  "
            f"{lq.powerline_score:>6.3f}  "
            f"{flags_str}"
        )

    # ── 6. 心拍检测 ───────────────────────────────
    beats = result.beats
    _sep(f"心拍检测 (共 {len(beats)} 拍)")
    rr_vals = [b.rr_prev_ms for b in beats if b.rr_prev_ms is not None]
    if rr_vals:
        print(f"  平均 RR 间期 : {np.mean(rr_vals):.0f} ms")
        print(f"  RR 标准差   : {np.std(rr_vals):.0f} ms")
        print(f"  最小 RR     : {min(rr_vals):.0f} ms")
        print(f"  最大 RR     : {max(rr_vals):.0f} ms")

    if beats:
        print(f"\n  前 5 拍预览:")
        print(f"  {'Beat':>5s}  {'R索引':>7s}  {'GroupID':>7s}  {'RR_prev(ms)':>11s}  {'RR_next(ms)':>11s}")
        for b in beats[:5]:
            print(
                f"  {b.beat_id:>5d}  {b.r_index:>7d}  {b.group_id:>7d}  "
                f"{_fmt(b.rr_prev_ms, '', 0):>11s}  {_fmt(b.rr_next_ms, '', 0):>11s}"
            )

    # ── 7. 心拍分组 ───────────────────────────────
    if result.groups:
        _sep("心拍分组 (Beat Groups)")
        print(f"  {'Group':>5s}  {'拍数':>5s}  {'占比':>6s}  {'RR(ms)':>7s}  {'QRS(ms)':>7s}  {'QT(ms)':>7s}  标记")
        for gid, g in result.groups.items():
            flags_str = " ".join(
                k for k, v in g.flags.items() if v
            ) or "—"
            print(
                f"  {gid:>5d}  {g.member_count:>5d}  "
                f"{g.member_pct:>5.0f}%  "
                f"{_fmt(g.mean_rr_ms, '', 0):>7s}  "
                f"{_fmt(g.mean_qrs_ms, '', 0):>7s}  "
                f"{_fmt(g.mean_qt_ms, '', 0):>7s}  "
                f"{flags_str}"
            )

    # ── 8. 代表性导联特征 ─────────────────────────
    _sep("代表性导联特征 (Representative Lead Features)")
    print(f"  {'导联':>4s}  {'PR(ms)':>7s}  {'QRS(ms)':>7s}  {'QT(ms)':>7s}  "
          f"{'R幅(mV)':>8s}  {'T幅(mV)':>8s}  {'ST_J(mV)':>9s}  可用")
    for lead in STANDARD_12_LEADS:
        rf = result.representative_leads[lead]
        p  = rf.params
        ok = "✓" if p.get("reliable_for_global") else "✗"
        print(
            f"  {lead:>4s}  "
            f"{_fmt(p.get('pr_ms'), '', 0):>7s}  "
            f"{_fmt(p.get('qrs_ms'), '', 0):>7s}  "
            f"{_fmt(p.get('qt_ms'), '', 0):>7s}  "
            f"{_fmt(p.get('r_amp_mv'), '', 3):>8s}  "
            f"{_fmt(p.get('t_amp_mv'), '', 3):>8s}  "
            f"{_fmt(p.get('st_on_mv'), '', 3):>9s}  "
            f"{ok}"
        )

    # ── 9. 保存完整 JSON ──────────────────────────
    out_path = PROJECT_ROOT / f"{record_id}_features.json"
    export_payload = prepare_json_export(
        to_dict(result), include_beat_features=include_beat_features
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, default=str, ensure_ascii=False)

    # ── 10. 生成人类可读报告 ──────────────────────
    report_path = PROJECT_ROOT / f"{record_id}_report.txt"
    generate_report(record_id, hdr, ecg, result, report_path)

    # ── 11. 生成临床纸样式心电图 ───────────────────
    ecg_plot_path = PROJECT_ROOT / f"{record_id}_ecg.png"
    ecg_plot_saved = False
    try:
        generate_ecg_paper_plot(record_id, hdr, ecg, ecg_plot_path)
        ecg_plot_saved = True
    except Exception as exc:
        print(f"  [警告] 心电图图片生成失败: {exc}")

    # ── 12. 生成带检测标注和异常高亮的心电图 ─────────────
    ecg_annotated_plot_path = PROJECT_ROOT / f"{record_id}_ecg_annotated.png"
    ecg_annotated_plot_saved = False
    try:
        generate_ecg_annotated_plot(record_id, hdr, ecg, result, ecg_annotated_plot_path)
        ecg_annotated_plot_saved = True
    except Exception as exc:
        print(f"  [警告] 标注心电图图片生成失败: {exc}")

    # ── 13. 生成综合 ECG 报告图 ─────────────────────
    ecg_report_sheet_path = PROJECT_ROOT / f"{record_id}_ecg_report.png"
    ecg_report_sheet_saved = False
    try:
        generate_ecg_report_sheet(record_id, hdr, ecg, result, ecg_report_sheet_path)
        ecg_report_sheet_saved = True
    except Exception as exc:
        print(f"  [警告] 综合 ECG 报告图生成失败: {exc}")

    print(f"\n{'=' * 62}")
    print(f"  完整特征已保存至: {out_path.name}")
    print(f"  可读报告已保存至: {report_path.name}")
    if ecg_plot_saved:
        print(f"  心电图图片已保存至: {ecg_plot_path.name}")
    if ecg_annotated_plot_saved:
        print(f"  标注心电图图片已保存至: {ecg_annotated_plot_path.name}")
    if ecg_report_sheet_saved:
        print(f"  综合 ECG 报告图已保存至: {ecg_report_sheet_path.name}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full_beats = "--full-beats" in sys.argv[1:]
    if args:
        record_id = args[0]
    else:
        available = sorted(p.stem for p in DATASET_DIR.glob("*.hea"))
        record_id = random.choice(available)
        print(f"[随机取样] 从 {len(available)} 条记录中抽取: {record_id}")
    main(record_id, include_beat_features=full_beats)
