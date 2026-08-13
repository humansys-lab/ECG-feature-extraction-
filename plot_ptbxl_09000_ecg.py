#!/usr/bin/env python3
"""Render PTB-XL 09000 ECGs as clinical-format reports with ecgfeat diagnosis.

Each record becomes a self-contained HTML page:

  * a standard 12-lead clinical layout (3 rows x 4 columns of 2.5 s, plus a
    10 s lead-II rhythm strip) drawn as inline SVG at 25 mm/s and 10 mm/mV on
    the usual red ECG grid;
  * ecgfeat's delineation overlaid on the traces -- P/QRS onset-offset bands,
    T offset, R peak and J point, taken from the exported ``beat_features``;
  * ecgfeat's own rule-engine diagnosis (``clinical_interpretation``) rendered
    in the same section order as ``diagnoses/*.md``, alongside the PTB-XL
    reference labels for comparison.

HTML rather than PNG because matplotlib has no CJK font in this environment,
and the diagnosis text is Chinese.

Usage:
    python plot_ptbxl_09000_ecg.py                       # default sample
    python plot_ptbxl_09000_ecg.py --records 09000_hr 09003_hr
    python plot_ptbxl_09000_ecg.py --all                 # all 1000 records
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_DIR = PROJECT_ROOT / "ptbxl_09000_ecgfeat"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "ptb-xl-metadata"

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CLINICAL_LAYOUT = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)
RHYTHM_LEAD = "II"

# --- clinical paper geometry -------------------------------------------------
PX_PER_MM = 4.0
MM_PER_SEC = 25.0
MM_PER_MV = 10.0
ROW_HEIGHT_MM = 38.0
RHYTHM_HEIGHT_MM = 36.0
RIBBON_HEIGHT_MM = 3.5
LEFT_MARGIN_MM = 12.0
RIGHT_MARGIN_MM = 4.0
TOP_MARGIN_MM = 6.0
BOTTOM_MARGIN_MM = 6.0
COLUMN_SEC = 2.5

P_COLOR = "#0b6fb8"
QRS_COLOR = "#1c7c3c"
T_COLOR = "#d2691e"
R_COLOR = "#c62828"
J_COLOR = "#7b1fa2"

STATUS_ZH = {
    "abnormal": "发现异常",
    "abnormal_with_limited_coverage": "发现异常，但部分诊断域覆盖受限",
    "normal_with_core_coverage": "核心规则范围内未发现异常",
    "incomplete": "未发现权威异常，但规则覆盖不完整",
    "borderline": "存在临界结果",
    "technically_limited": "技术质量受限",
    "technically_limited_with_findings": "技术质量受限，但仍有异常发现",
}

SEVERITY_ZH = {
    "abnormal": "异常",
    "borderline": "临界",
    "observation": "观察",
    "technical": "技术",
    "normal": "正常",
}

CONFIDENCE_ZH = {"HIGH": "高", "MEDIUM": "中", "LOW": "低"}

DOMAIN_ZH = {
    "rhythm": "节律",
    "ectopy": "早搏/异位",
    "conduction": "传导",
    "intervals": "间期",
    "hypertrophy": "肥厚",
    "atrial_abnormality": "心房异常",
    "voltage": "电压",
    "ischemia_infarction": "缺血/梗死",
    "repolarization": "复极",
    "high_risk_patterns": "高危模式",
    "quality": "信号质量",
    "preexcitation": "预激",
    "advanced_av_block": "高度房室阻滞",
}

COVERAGE_ZH = {"full": "完整", "partial": "部分", "unavailable": "不可用", "none": "无"}

ABSTENTION_REASON_ZH = {
    "missing_or_indeterminate_required_evidence": "所需证据缺失或不确定",
    "suppressed_by_confounder": "被混杂因素抑制",
    "insufficient_coverage": "覆盖不足",
    "finding_not_projected_by_measurement_layer": "测量层未产出该发现所需的输入",
}

REVIEW_REASON_ZH = {
    "diagnostic_coverage_limited": "诊断覆盖受限",
    "poor_r_wave_progression": "胸导联 R 波递增不良",
    "premature_atrial_complexes": "房性早搏",
    "prolonged_qt": "QT 延长",
}


def _load_prediction_code_zh() -> dict[str, str]:
    """Reuse the statement-code Chinese map already maintained in the repo."""
    try:
        from evaluate_target_ecgfeat_diagnosis import PREDICTION_CODE_ZH

        return dict(PREDICTION_CODE_ZH)
    except Exception:  # pragma: no cover - fallback keeps rendering usable
        return {}


PREDICTION_CODE_ZH = _load_prediction_code_zh()

# Statement codes emitted by clinical_rules.v2 that the evaluation map above
# does not cover (it only carries the label-comparable subset).
EXTRA_CODE_ZH = {
    "diagnostic_coverage_limited": "诊断覆盖受限",
    "possible_precordial_lead_reversal": "可能的胸导联位置异常",
    "possible_limb_lead_reversal": "可能的肢体导联反接",
    "poor_r_wave_progression": "胸导联 R 波递增不良",
    "sinus_rhythm_pattern": "窦性心律模式",
    "regular_narrow_complex_rhythm": "规则窄 QRS 心律",
    "normal_ecg": "正常心电图",
    "short_pr_interval": "PR 间期缩短",
    "early_repolarization_pattern": "早期复极模式",
    "nonspecific_st_t_abnormality": "非特异性 ST-T 异常",
}


def code_zh(code: str | None) -> str:
    if not code:
        return "—"
    return PREDICTION_CODE_ZH.get(code) or EXTRA_CODE_ZH.get(code) or code


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fmt(value: Any, digits: int = 0, unit: str = "") -> str:
    if value is None or value == "":
        return "不可用"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not np.isfinite(number):
        return "不可用"
    return f"{number:.{digits}f}{unit}"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_manifest(batch_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((batch_dir / "record_manifest.json").read_text(encoding="utf-8"))
    return {row["record"]: row for row in manifest.get("records", [])}


def load_scp(metadata_dir: Path) -> dict[str, dict[str, str]]:
    path = metadata_dir / "scp_statements.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row[""]: row for row in csv.DictReader(handle) if row.get("")}


def load_signal(record_path: Path) -> tuple[np.ndarray, float]:
    record = wfdb.rdrecord(str(record_path))
    index = {str(name).lower(): position for position, name in enumerate(record.sig_name)}
    columns = [record.p_signal[:, index[lead.lower()]] for lead in LEADS]
    return np.column_stack(columns).astype(float), float(record.fs)


def index_beat_features(features: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    table: dict[tuple[str, int], dict[str, Any]] = {}
    for row in features.get("beat_features") or []:
        lead = row.get("lead")
        beat_id = row.get("beat_id")
        if lead is None or beat_id is None:
            continue
        table[(str(lead), int(beat_id))] = row
    return table


# ---------------------------------------------------------------------------
# SVG ECG rendering
# ---------------------------------------------------------------------------


class Panel:
    """One drawing strip: a lead, a time window, and its vertical placement."""

    def __init__(self, lead: str, t0: float, t1: float, x_mm: float, y_mm: float,
                 width_mm: float, height_mm: float, gain_mm_per_mv: float,
                 show_calibration: bool = False) -> None:
        self.lead = lead
        self.t0 = t0
        self.t1 = t1
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.gain = gain_mm_per_mv
        self.show_calibration = show_calibration

    @property
    def ribbon_y_mm(self) -> float:
        """Top of the thin delineation ribbon drawn under each strip."""
        return self.y_mm + self.height_mm - RIBBON_HEIGHT_MM - 1.0

    @property
    def trace_height_mm(self) -> float:
        """Vertical space left for the trace once the ribbon is reserved."""
        return self.height_mm - RIBBON_HEIGHT_MM - 1.0

    @property
    def baseline_mm(self) -> float:
        return self.y_mm + self.trace_height_mm / 2.0

    @property
    def clip_mv(self) -> float:
        return (self.trace_height_mm / 2.0 - 1.0) / self.gain

    def x_of(self, seconds: float) -> float:
        return (self.x_mm + (seconds - self.t0) * MM_PER_SEC) * PX_PER_MM

    def y_of(self, mv: float) -> float:
        return (self.baseline_mm - mv * self.gain) * PX_PER_MM

    def contains(self, seconds: float) -> bool:
        return self.t0 <= seconds < self.t1


def choose_gain(signal: np.ndarray) -> float:
    """Pick the paper gain a real cart would use: 20, 10 or 5 mm/mV.

    Standard is 10 mm/mV, but large-amplitude records (LVH, paced) flat-top
    against the neighbouring row at that gain, and low-voltage records become
    unreadable, so match the cart behaviour of halving or doubling.
    """
    room_mm = (ROW_HEIGHT_MM - RIBBON_HEIGHT_MM - 1.0) / 2.0 - 1.0
    finite = np.abs(signal[np.isfinite(signal)])
    if finite.size == 0:
        return MM_PER_MV
    # 99.5th, not the max: pacing spikes are a few samples wide and several mV
    # tall, and shrinking the whole record to fit them costs more than it buys.
    peak_mv = float(np.percentile(finite, 99.5))
    if peak_mv * MM_PER_MV <= room_mm:
        # low-voltage records are unreadable at standard gain; double it if it fits
        if peak_mv < 0.4 and peak_mv * 20.0 <= room_mm:
            return 20.0
        return MM_PER_MV
    return 5.0


def build_panels(duration: float, gain: float) -> tuple[list[Panel], float, float]:
    panels: list[Panel] = []
    column_mm = COLUMN_SEC * MM_PER_SEC
    for row_index, row in enumerate(CLINICAL_LAYOUT):
        for column_index, lead in enumerate(row):
            t0 = column_index * COLUMN_SEC
            panels.append(
                Panel(
                    lead=lead,
                    t0=t0,
                    t1=min(t0 + COLUMN_SEC, duration),
                    x_mm=LEFT_MARGIN_MM + column_index * column_mm,
                    y_mm=TOP_MARGIN_MM + row_index * ROW_HEIGHT_MM,
                    width_mm=column_mm,
                    height_mm=ROW_HEIGHT_MM,
                    gain_mm_per_mv=gain,
                    show_calibration=column_index == 0,
                )
            )
    rhythm_y = TOP_MARGIN_MM + len(CLINICAL_LAYOUT) * ROW_HEIGHT_MM
    panels.append(
        Panel(
            lead=RHYTHM_LEAD,
            t0=0.0,
            t1=duration,
            x_mm=LEFT_MARGIN_MM,
            y_mm=rhythm_y,
            width_mm=duration * MM_PER_SEC,
            height_mm=RHYTHM_HEIGHT_MM,
            gain_mm_per_mv=gain,
            show_calibration=True,
        )
    )
    width_mm = LEFT_MARGIN_MM + max(duration, 4 * COLUMN_SEC) * MM_PER_SEC + RIGHT_MARGIN_MM
    height_mm = rhythm_y + RHYTHM_HEIGHT_MM + BOTTOM_MARGIN_MM
    return panels, width_mm, height_mm


def grid_defs(width_mm: float, height_mm: float) -> str:
    small = PX_PER_MM
    large = PX_PER_MM * 5
    return f"""  <defs>
    <pattern id="mm" width="{small:.2f}" height="{small:.2f}" patternUnits="userSpaceOnUse">
      <path d="M {small:.2f} 0 L 0 0 0 {small:.2f}" fill="none" stroke="#f6c9c9" stroke-width="0.5"/>
    </pattern>
    <pattern id="mm5" width="{large:.2f}" height="{large:.2f}" patternUnits="userSpaceOnUse">
      <rect width="{large:.2f}" height="{large:.2f}" fill="url(#mm)"/>
      <path d="M {large:.2f} 0 L 0 0 0 {large:.2f}" fill="none" stroke="#e59a9a" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="{width_mm * PX_PER_MM:.1f}" height="{height_mm * PX_PER_MM:.1f}" fill="#fffafa"/>
  <rect width="{width_mm * PX_PER_MM:.1f}" height="{height_mm * PX_PER_MM:.1f}" fill="url(#mm5)"/>"""


def trace_polyline(panel: Panel, signal: np.ndarray, fs: float, decimate: int) -> tuple[str, bool]:
    """Return the trace element and whether it had to be clipped to fit."""
    lead_index = LEADS.index(panel.lead)
    start = int(round(panel.t0 * fs))
    stop = min(int(round(panel.t1 * fs)), signal.shape[0])
    if stop <= start:
        return "", False
    step = max(1, int(decimate))
    samples = np.arange(start, stop, step)
    values = signal[start:stop, lead_index][::step]
    values = np.where(np.isfinite(values), values, 0.0)
    # keep the trace inside its own strip so neighbouring rows stay readable
    limit_mv = panel.clip_mv
    clipped = bool(np.any(np.abs(values) > limit_mv))
    values = np.clip(values, -limit_mv, limit_mv)
    xs = (panel.x_mm + (samples / fs - panel.t0) * MM_PER_SEC) * PX_PER_MM
    ys = (panel.baseline_mm - values * panel.gain) * PX_PER_MM
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f'    <polyline points="{points}" fill="none" stroke="#111111" '
        f'stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round"/>',
        clipped,
    )


def calibration_pulse(panel: Panel) -> str:
    """The 1 mV / 200 ms step drawn once at the left edge of a row."""
    x0 = panel.x_mm * PX_PER_MM
    x1 = (panel.x_mm - 5.0) * PX_PER_MM
    top = (panel.baseline_mm - panel.gain) * PX_PER_MM
    base = panel.baseline_mm * PX_PER_MM
    return (
        f'    <path d="M {x1 - 12:.1f} {base:.1f} L {x1:.1f} {base:.1f} L {x1:.1f} {top:.1f} '
        f'L {x0:.1f} {top:.1f} L {x0:.1f} {base:.1f}" fill="none" stroke="#111111" stroke-width="1.1"/>'
    )


def column_separator(panel: Panel) -> str:
    """The short vertical tick real ECG carts print between columns."""
    if panel.t0 <= 0.0:
        return ""
    x = panel.x_mm * PX_PER_MM
    y0 = (panel.y_mm + 3.0) * PX_PER_MM
    y1 = (panel.ribbon_y_mm - 1.0) * PX_PER_MM
    return (
        f'    <line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" '
        f'stroke="#9aa0aa" stroke-width="0.8" opacity="0.8"/>'
    )


def _ribbon_band(panel: Panel, onset: float, offset: float, color: str) -> str:
    """A wave segment drawn in the thin ribbon below the trace.

    Full-height shading was unreadable once five beats per strip each carried a
    P, QRS and T band -- the strip turned into a barcode.
    """
    x0 = panel.x_of(max(onset, panel.t0))
    x1 = panel.x_of(min(offset, panel.t1))
    if x1 <= x0:
        return ""
    y = panel.ribbon_y_mm * PX_PER_MM
    height = RIBBON_HEIGHT_MM * PX_PER_MM
    return (
        f'    <rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 1.0):.1f}" height="{height:.1f}" '
        f'fill="{color}" opacity="0.75" rx="1"/>'
    )


def _boundary_tick(panel: Panel, seconds: float, color: str, dash: bool = False) -> str:
    """A short mark rising from the ribbon into the trace area."""
    if not panel.contains(seconds):
        return ""
    x = panel.x_of(seconds)
    # reach up to the isoelectric line so a ribbon block reads against its beat
    y0 = panel.baseline_mm * PX_PER_MM
    y1 = (panel.ribbon_y_mm + RIBBON_HEIGHT_MM) * PX_PER_MM
    style = ' stroke-dasharray="3,2"' if dash else ""
    return (
        f'    <line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" '
        f'stroke="{color}" stroke-width="0.6" opacity="0.4"{style}/>'
    )


def overlay_for_panel(
    panel: Panel,
    signal: np.ndarray,
    fs: float,
    beats: Sequence[Mapping[str, Any]],
    by_feature: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[str]:
    lead_index = LEADS.index(panel.lead)
    parts: list[str] = []
    for beat in beats:
        beat_id = int(beat.get("beat_id") or 0)
        row = by_feature.get((panel.lead, beat_id))
        if not row:
            continue
        p_wave = row.get("p") or {}
        qrs = row.get("qrs") or {}
        t_wave = row.get("t") or {}

        def seconds(value: Any) -> float | None:
            if not isinstance(value, (int, float)):
                return None
            return float(value) / fs

        p_on, p_off = seconds(p_wave.get("onset")), seconds(p_wave.get("offset"))
        if p_on is not None and p_off is not None and p_off > panel.t0 and p_on < panel.t1:
            parts.append(_ribbon_band(panel, p_on, p_off, P_COLOR))
        qrs_on, qrs_off = seconds(qrs.get("onset")), seconds(qrs.get("offset"))
        if qrs_on is not None and qrs_off is not None and qrs_off > panel.t0 and qrs_on < panel.t1:
            parts.append(_ribbon_band(panel, qrs_on, qrs_off, QRS_COLOR))
            parts.append(_boundary_tick(panel, qrs_on, QRS_COLOR))
        t_on, t_off = seconds(t_wave.get("onset")), seconds(t_wave.get("offset"))
        if t_on is not None and t_off is not None and t_off > panel.t0 and t_on < panel.t1:
            parts.append(_ribbon_band(panel, t_on, t_off, T_COLOR))
        if t_off is not None:
            parts.append(_boundary_tick(panel, t_off, T_COLOR, dash=True))

        # qrs.peak is this lead's own positive peak within its QRS onset-offset
        # window (verified against the raw signal, not the cross-lead
        # consensus r_index on `beat`) -- each lead gets its own marker.
        r_peak_index = qrs.get("peak")
        if isinstance(r_peak_index, (int, float)):
            r_seconds = float(r_peak_index) / fs
            if panel.contains(r_seconds) and 0 <= int(r_peak_index) < signal.shape[0]:
                value = float(np.clip(signal[int(r_peak_index), lead_index], -panel.clip_mv, panel.clip_mv))
                marker = "3.2" if not beat.get("paced") else "4.5"
                color = R_COLOR if not beat.get("paced") else "#000000"
                parts.append(
                    f'    <circle cx="{panel.x_of(r_seconds):.1f}" cy="{panel.y_of(value):.1f}" '
                    f'r="{marker}" fill="{color}" opacity="0.9"/>'
                )
        j_index = row.get("j_index") or row.get("st_hybrid_j_index")
        if isinstance(j_index, (int, float)):
            j_seconds = float(j_index) / fs
            if panel.contains(j_seconds) and 0 <= int(j_index) < signal.shape[0]:
                value = float(np.clip(signal[int(j_index), lead_index], -panel.clip_mv, panel.clip_mv))
                parts.append(
                    f'    <rect x="{panel.x_of(j_seconds) - 2.0:.1f}" y="{panel.y_of(value) - 2.0:.1f}" '
                    f'width="4" height="4" fill="{J_COLOR}" opacity="0.9"/>'
                )
    return [part for part in parts if part]


def render_svg(
    signal: np.ndarray,
    fs: float,
    features: Mapping[str, Any],
    decimate: int,
) -> str:
    duration = signal.shape[0] / fs
    gain = choose_gain(signal)
    panels, width_mm, height_mm = build_panels(duration, gain)
    beats = features.get("beats") or []
    by_feature = index_beat_features(features)

    body: list[str] = [grid_defs(width_mm, height_mm)]
    overlay: list[str] = []
    clipped_leads: list[str] = []
    for panel in panels:
        if panel.show_calibration:
            body.append(calibration_pulse(panel))
        body.append(column_separator(panel))
        trace, clipped = trace_polyline(panel, signal, fs, decimate)
        body.append(trace)
        if clipped and panel.lead not in clipped_leads:
            clipped_leads.append(panel.lead)
        label_x = (panel.x_mm + 1.5) * PX_PER_MM
        label_y = (panel.y_mm + 5.0) * PX_PER_MM
        suffix = "  (rhythm)" if panel.height_mm == RHYTHM_HEIGHT_MM else ""
        body.append(
            f'    <text x="{label_x:.1f}" y="{label_y:.1f}" font-size="13" font-weight="700" '
            f'fill="#111111" font-family="DejaVu Sans, Arial, sans-serif">{esc(panel.lead)}{suffix}</text>'
        )
        overlay.extend(overlay_for_panel(panel, signal, fs, beats, by_feature))

    # Kept ASCII: a standalone export of this SVG has no CJK font available.
    gain_note = f"{gain:.0f} mm/mV" + ("" if gain == MM_PER_MV else " (non-standard gain)")
    footer = (
        f"25 mm/s | {gain_note} | {fs:.0f} Hz | 2.5 s per column, "
        f"{RHYTHM_LEAD} rhythm strip = full {duration:.0f} s"
    )
    if clipped_leads:
        footer += (
            f" | peaks beyond +/-{panels[0].clip_mv:.2f} mV clipped in: {', '.join(clipped_leads)}"
        )
    footer_y = (height_mm - 1.5) * PX_PER_MM
    body.append(
        f'    <text x="{LEFT_MARGIN_MM * PX_PER_MM:.1f}" y="{footer_y:.1f}" font-size="11" '
        f'fill="#555555" font-family="DejaVu Sans, Arial, sans-serif">{esc(footer)}</text>'
    )

    return (
        f'<svg class="ecg" viewBox="0 0 {width_mm * PX_PER_MM:.0f} {height_mm * PX_PER_MM:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="12-lead ECG">\n'
        + "\n".join(part for part in body if part)
        + '\n  <g class="overlay">\n'
        + "\n".join(overlay)
        + "\n  </g>\n</svg>"
    )


# ---------------------------------------------------------------------------
# diagnosis panel (mirrors the section order of diagnoses/*.md)
# ---------------------------------------------------------------------------


def measurement_rows(features: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    g = features.get("global_features") or {}
    qt_note = f"{g.get('qt_reliability') or '未知'} / {g.get('qt_path') or '—'}"
    return [
        ("心室率", fmt(g.get("heart_rate_bpm"), 0, " bpm"), ""),
        ("心房率", fmt(g.get("atrial_rate_bpm"), 0, " bpm"), ""),
        ("PR 间期", fmt(g.get("pr_ms"), 0, " ms"), ""),
        ("QRS 时限", fmt(g.get("qrs_ms"), 0, " ms"), ""),
        ("P 波时限", fmt(g.get("p_duration_ms"), 0, " ms"), str(g.get("p_duration_reliability") or "")),
        ("QT", fmt(g.get("qt_ms"), 0, " ms"), qt_note),
        ("QTc (Bazett)", fmt(g.get("qtc_bazett_ms"), 0, " ms"), ""),
        ("QTc (Fridericia)", fmt(g.get("qtc_fridericia_ms"), 0, " ms"), ""),
        ("P 电轴", fmt(g.get("p_axis_deg"), 0, "°"), ""),
        ("QRS 电轴", fmt(g.get("qrs_axis_deg"), 0, "°"), ""),
        ("T 电轴", fmt(g.get("t_axis_deg"), 0, "°"), "" if g.get("t_axis_reliable") else "不可靠"),
        ("起搏心律", "是" if g.get("paced_rhythm") else "否", ""),
    ]


def evidence_lines(statement: Mapping[str, Any]) -> list[str]:
    evidence = statement.get("evidence")
    if not isinstance(evidence, Mapping):
        return []
    lines: list[str] = []
    for key, value in evidence.items():
        if key in {"evaluates_code", "not_applicable_by", "candidate_beats"}:
            continue
        if isinstance(value, (list, dict)):
            if not value:
                continue
            text = json.dumps(value, ensure_ascii=False)
            if len(text) > 220:
                text = text[:220] + "…"
            lines.append(f"{key} = {text}")
        elif value is not None:
            lines.append(f"{key} = {value}")
    beats = evidence.get("candidate_beats")
    if isinstance(beats, list) and beats:
        ids = ", ".join(str(b.get("beat_id")) for b in beats if isinstance(b, Mapping))
        lines.append(f"涉及心搏 = [{ids}]")
    return lines


def statement_block(statement: Mapping[str, Any]) -> str:
    code = statement.get("statement_code")
    severity = SEVERITY_ZH.get(str(statement.get("severity")), str(statement.get("severity") or "—"))
    confidence = CONFIDENCE_ZH.get(str(statement.get("confidence")), str(statement.get("confidence") or "—"))
    domain = DOMAIN_ZH.get(str(statement.get("domain")), str(statement.get("domain") or "—"))
    coverage = COVERAGE_ZH.get(str(statement.get("coverage")), str(statement.get("coverage") or "—"))
    review = "需人工复核" if statement.get("human_review_required") else "无需强制复核"
    lines = evidence_lines(statement)
    evidence_html = (
        "<ul class='ev'>" + "".join(f"<li>{esc(line)}</li>" for line in lines) + "</ul>"
        if lines
        else ""
    )
    return f"""<div class="stmt sev-{esc(statement.get('severity'))}">
  <div class="stmt-title">{esc(code_zh(code))} <span class="mono">{esc(code)}</span></div>
  <div class="stmt-meta">{esc(severity)} · 置信度 {esc(confidence)} · 域：{esc(domain)} · 覆盖：{esc(coverage)} · {esc(review)} · <span class="mono">{esc(statement.get('rule_id'))}</span></div>
  <div class="stmt-text">{esc(statement.get('statement'))}</div>
  {evidence_html}
</div>"""


def reference_block(record: str, meta: Mapping[str, Any] | None, scp: Mapping[str, Mapping[str, str]]) -> str:
    if not meta:
        return "<p class='muted'>该记录在 record_manifest.json 中没有参考标注。</p>"
    active = meta.get("reference_codes_active") or []
    raw = meta.get("reference_codes_raw") or []
    scores = meta.get("reference_scores") or {}
    filtered = [code for code in raw if code not in active]

    def chips(codes: Iterable[str], muted: bool = False) -> str:
        items = []
        for code in codes:
            description = (scp.get(code) or {}).get("description", "")
            score = scores.get(code)
            score_text = f" {score:.0f}" if isinstance(score, (int, float)) else ""
            cls = "chip muted-chip" if muted else "chip"
            items.append(
                f'<span class="{cls}" title="{esc(description)}">{esc(code)}{esc(score_text)}</span>'
            )
        return "".join(items) or "<span class='muted'>—</span>"

    report = meta.get("reference_report") or "—"
    return f"""<table class="kv">
  <tr><th>年龄 / 性别</th><td>{esc(fmt(meta.get('age'), 0, ' 岁'))} / {esc({'male': '男', 'female': '女'}.get(meta.get('sex'), '—'))}</td></tr>
  <tr><th>采纳标签 (active)</th><td>{chips(active)}</td></tr>
  <tr><th>低可信度被过滤</th><td>{chips(filtered, muted=True)}</td></tr>
  <tr><th>医师报告原文</th><td class="report">{esc(report)}</td></tr>
  <tr><th>strat_fold</th><td>{esc(meta.get('strat_fold') or '—')}</td></tr>
</table>"""


def quality_block(features: Mapping[str, Any]) -> str:
    metadata = features.get("metadata") or {}
    record_quality = metadata.get("record_quality") or {}
    gate = metadata.get("diagnostic_gate") or {}
    quality = features.get("quality") or {}
    rows = []
    for lead in LEADS:
        entry = quality.get(lead) or {}
        cautions = [
            name
            for name, key in (("P", "reliable_for_p"), ("QRS", "reliable_for_qrs"),
                              ("T", "reliable_for_t"), ("QT", "reliable_for_qt"))
            if entry.get(key) is False
        ]
        rows.append(
            f"<tr><td class='mono'>{esc(lead)}</td><td>{esc(entry.get('grade') or '—')}</td>"
            f"<td>{esc('/'.join(cautions) or '—')}</td>"
            f"<td>{esc(', '.join(entry.get('reason_codes') or []) or '—')}</td></tr>"
        )
    return f"""<table class="kv">
  <tr><th>记录质量等级</th><td>{esc(record_quality.get('record_grade') or '—')}</td></tr>
  <tr><th>诊断闸门状态</th><td>{esc(gate.get('state') or '—')}</td></tr>
  <tr><th>部分受限原因</th><td>{esc(', '.join(gate.get('partial_reasons') or []) or '—')}</td></tr>
  <tr><th>停止原因</th><td>{esc(', '.join(gate.get('stop_reasons') or []) or '—')}</td></tr>
</table>
<table class="grid">
  <thead><tr><th>导联</th><th>等级</th><th>不可靠波段</th><th>原因码</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def agent_diagnosis_summary(batch_dir: Path, record: str) -> str:
    path = batch_dir / "diagnoses" / f"{record}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## 总体结论\s*\n+(.+?)(?:\n## |\Z)", text, flags=re.S)
    conclusion = match.group(1).strip() if match else text[:600]
    return f"""<section>
  <h2>ECGAgent 智能体诊断（同目录 diagnoses/{esc(record)}.md）</h2>
  <div class="agent">{esc(conclusion)}</div>
  <p class="muted">完整报告见 <span class="mono">diagnoses/{esc(record)}.md</span>。</p>
</section>"""


PAGE_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 28px 48px; background: #f4f5f7; color: #16181d;
       font-family: "Helvetica Neue", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
       font-size: 14px; line-height: 1.6; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 2px solid #d9dce2; }
h3 { font-size: 15px; margin: 18px 0 8px; }
.wrap { max-width: 1240px; margin: 0 auto; }
.card { background: #ffffff; border: 1px solid #dfe2e8; border-radius: 8px; padding: 18px 20px; margin-bottom: 18px; }
.sub { color: #5a616e; margin: 0 0 14px; }
.mono { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; color: #5a616e; }
.muted { color: #8a909c; }
.ecg { width: 100%; height: auto; display: block; }
.ecg-frame { background: #fffafa; border: 1px solid #e3c4c4; border-radius: 6px; padding: 6px; overflow-x: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 0; font-size: 13px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
.toggle { margin: 12px 0 0; font-size: 13px; }
.hide-overlay .overlay { display: none; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; }
table.kv th { text-align: left; width: 180px; font-weight: 600; color: #454b57; vertical-align: top;
              padding: 5px 10px 5px 0; border-bottom: 1px solid #eceef2; }
table.kv td { padding: 5px 0; border-bottom: 1px solid #eceef2; vertical-align: top; }
table.grid th, table.grid td { border: 1px solid #e3e6ec; padding: 4px 8px; text-align: left; font-size: 13px; }
table.grid th { background: #f6f7f9; }
.meas { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 4px 20px; }
.meas div { border-bottom: 1px solid #eceef2; padding: 7px 0; }
.meas .name { display: block; color: #5a616e; font-size: 12.5px; }
.meas b { font-weight: 600; font-size: 15px; }
.meas .note { margin-left: 8px; font-size: 12px; }
.stmt { border-left: 4px solid #b9bec8; background: #fafbfc; padding: 10px 14px; margin: 10px 0; border-radius: 0 5px 5px 0; }
.stmt.sev-abnormal { border-left-color: #c62828; }
.stmt.sev-borderline { border-left-color: #ef6c00; }
.stmt.sev-observation { border-left-color: #1565c0; }
.stmt.sev-technical { border-left-color: #6a1b9a; }
.stmt.sev-normal { border-left-color: #2e7d32; }
.stmt-title { font-weight: 700; font-size: 15px; }
.stmt-meta { color: #5a616e; font-size: 12.5px; margin: 2px 0 6px; }
.ev { margin: 6px 0 0; padding-left: 18px; color: #454b57; font-size: 12.5px; }
.chip { display: inline-block; background: #e8f0fb; color: #14477d; border: 1px solid #c3d7f0;
        border-radius: 10px; padding: 1px 9px; margin: 2px 4px 2px 0; font-size: 12.5px; }
.muted-chip { background: #f1f2f4; color: #808894; border-color: #dfe2e8; }
.report { color: #333; }
.agent { white-space: pre-wrap; background: #fbfbfd; border: 1px dashed #ccd1da; padding: 10px 14px; border-radius: 5px; }
.disclaimer { margin-top: 22px; font-size: 12.5px; color: #7a808c; }
a { color: #14477d; }
"""


def render_page(
    record: str,
    features: Mapping[str, Any],
    signal: np.ndarray,
    fs: float,
    meta: Mapping[str, Any] | None,
    scp: Mapping[str, Mapping[str, str]],
    batch_dir: Path,
    decimate: int,
) -> str:
    interpretation = features.get("clinical_interpretation") or {}
    summary = interpretation.get("summary") or {}
    status = str(interpretation.get("overall_status") or "")
    status_zh = STATUS_ZH.get(status, status or "未知")
    finals = interpretation.get("final_statements") or []
    borderline = interpretation.get("borderline_statements") or []
    abstentions = interpretation.get("abstentions") or []

    svg = render_svg(signal, fs, features, decimate)
    gain = choose_gain(signal)
    gain_note = (
        f"25 mm/s · {gain:.0f} mm/mV"
        if gain == MM_PER_MV
        else f"25 mm/s · <b>{gain:.0f} mm/mV（非标准增益，本记录振幅过大/过小）</b>"
    )

    measurements = "".join(
        f"<div><span class='name'>{esc(name)}</span><b>{esc(value)}</b>"
        + (f"<span class='muted note'>{esc(note)}</span>" if note else "")
        + "</div>"
        for name, value, note in measurement_rows(features)
    )

    coverage = summary.get("domain_coverage") or {}
    coverage_rows = "".join(
        f"<tr><td>{esc(DOMAIN_ZH.get(domain, domain))}</td><td class='mono'>{esc(domain)}</td>"
        f"<td>{esc(COVERAGE_ZH.get(str(value), str(value)))}</td></tr>"
        for domain, value in sorted(coverage.items())
    )

    # The abstention list mixes two different things: rules that were evaluated
    # and withheld, and appendix findings the measurement layer never projected.
    rule_abstentions = [item for item in abstentions if item.get("rule_id")]
    finding_abstentions = [item for item in abstentions if not item.get("rule_id")]

    abstention_rows = "".join(
        f"<tr><td class='mono'>{esc(item.get('rule_id'))}</td>"
        f"<td>{esc(DOMAIN_ZH.get(str(item.get('domain')), str(item.get('domain'))))}</td>"
        f"<td>{esc(item.get('status'))}</td>"
        f"<td>{esc(ABSTENTION_REASON_ZH.get(str(item.get('reason')), str(item.get('reason') or '—')))}</td>"
        f"<td>{esc(', '.join(item.get('missing_inputs') or item.get('suppressed_by') or []) or '—')}</td></tr>"
        for item in rule_abstentions
    )
    finding_items = "".join(
        f"<li><span class='mono'>{esc(item.get('finding_id'))}</span> — "
        f"{esc(ABSTENTION_REASON_ZH.get(str(item.get('reason')), str(item.get('reason') or '—')))}</li>"
        for item in finding_abstentions
    )
    finding_html = (
        f"<details><summary>未评估的附录发现项（{len(finding_abstentions)} 条）</summary>"
        f"<ul class='ev'>{finding_items}</ul></details>"
        if finding_abstentions
        else ""
    )

    review_reasons = interpretation.get("review_reasons") or []
    review_html = "".join(
        f"<li>{esc(REVIEW_REASON_ZH.get(reason, reason))} <span class='mono'>{esc(reason)}</span></li>"
        for reason in review_reasons
    )

    finals_html = "".join(statement_block(item) for item in finals) or "<p class='muted'>无</p>"
    borderline_html = "".join(statement_block(item) for item in borderline) or "<p class='muted'>无临界结果。</p>"

    age_sex = ""
    if meta:
        sex = {"male": "男", "female": "女"}.get(meta.get("sex"), "—")
        age_sex = f"{fmt(meta.get('age'), 0, ' 岁')} / {sex} · "

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(record)} — 12 导联心电图与 ecgfeat 诊断</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>{esc(record)} — 12 导联心电图</h1>
    <p class="sub">{esc(age_sex)}{fmt(fs, 0, ' Hz')} · {fmt(signal.shape[0] / fs, 0, ' s')} · {gain_note} ·
       ecgfeat 规则集 <span class="mono">{esc(interpretation.get('ruleset_version') or '—')}</span> ·
       总体状态：<b>{esc(status_zh)}</b> <span class="mono">{esc(status)}</span></p>

    <div class="ecg-frame" id="frame">{svg}</div>
    <div class="legend">
      <span><i class="swatch" style="background:{P_COLOR};opacity:.35"></i>P 波（起点–终点）</span>
      <span><i class="swatch" style="background:{QRS_COLOR};opacity:.35"></i>QRS（起点–终点）</span>
      <span><i class="swatch" style="background:{T_COLOR};opacity:.30"></i>T 波，虚线为 T 终点</span>
      <span><i class="swatch" style="background:{R_COLOR}"></i>R 峰</span>
      <span><i class="swatch" style="background:{J_COLOR}"></i>J 点</span>
      <span class="muted">标注均来自该记录的 features JSON，非重新计算</span>
    </div>
    <div class="toggle">
      <label><input type="checkbox" id="overlay-toggle" checked> 显示 ecgfeat 波形标注</label>
    </div>
  </div>

  <div class="card">
    <h2>全局测量</h2>
    <div class="meas">{measurements}</div>
  </div>

  <div class="card">
    <h2>PTB-XL 参考标签（数据集自带真值）</h2>
    {reference_block(record, meta, scp)}
  </div>

  <div class="card">
    <h2>ecgfeat 规则引擎诊断</h2>
    <p class="sub">状态：<b>{esc(status_zh)}</b> ·
      发现 {esc(summary.get('findings'))} 条 · 观察 {esc(summary.get('observations'))} 条 ·
      限制 {esc(summary.get('limitations'))} 条 ·
      {esc('部分评估' if summary.get('partial_evaluation') else '完整评估')}</p>

    <h3>最终陈述</h3>
    {finals_html}

    <h3>临界结果</h3>
    {borderline_html}

    <h3>需人工复核的原因</h3>
    <ul>{review_html or "<li class='muted'>无</li>"}</ul>

    <h3>诊断域覆盖</h3>
    <table class="grid"><thead><tr><th>域</th><th>代码</th><th>覆盖</th></tr></thead>
      <tbody>{coverage_rows}</tbody></table>

    <h3>弃权与被抑制的规则</h3>
    <table class="grid"><thead><tr><th>规则</th><th>域</th><th>状态</th><th>原因</th><th>缺失输入 / 抑制来源</th></tr></thead>
      <tbody>{abstention_rows or "<tr><td colspan='5' class='muted'>无</td></tr>"}</tbody></table>
    {finding_html}
  </div>

  <div class="card">
    <h2>信号质量与可判读性</h2>
    {quality_block(features)}
  </div>

  {agent_diagnosis_summary(batch_dir, record)}

  <p class="disclaimer">{esc(interpretation.get('disclaimer') or '')}</p>
</div>
<script>
document.getElementById('overlay-toggle').addEventListener('change', function (event) {{
  document.getElementById('frame').classList.toggle('hide-overlay', !event.target.checked);
}});
</script>
</body>
</html>
"""


INDEX_STATE = "index_data.json"


def render_index(
    records: Sequence[tuple[str, Mapping[str, Any] | None, str]],
    output_dir: Path,
    manifest: Mapping[str, Mapping[str, Any]],
) -> None:
    """Rebuild the index over every page in the directory, not just this run.

    Rendering one record used to overwrite the index with a single row and
    silently orphan the other 999 pages, so the per-record status is cached
    alongside the pages and merged back in here.
    """
    state_path = output_dir / INDEX_STATE
    state: dict[str, str] = {}
    if state_path.exists():
        try:
            state = dict(json.loads(state_path.read_text(encoding="utf-8")))
        except (ValueError, TypeError):
            state = {}
    state.update({record: status for record, _, status in records})
    # Pick up pages written before the cache existed, or by an older run: the
    # overall status is already printed near the top of each page.
    for page in output_dir.glob("*.html"):
        if page.name == "index.html" or page.stem in state:
            continue
        head = page.read_text(encoding="utf-8")[:4000]
        found = re.search(r'总体状态：<b>[^<]*</b> <span class="mono">([a-z_]*)</span>', head)
        state[page.stem] = found.group(1) if found else ""
    state = {
        record: status
        for record, status in state.items()
        if (output_dir / f"{record}.html").exists()
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = "".join(
        f"<tr><td><a href='{esc(record)}.html'>{esc(record)}</a></td>"
        f"<td>{esc(fmt((meta or {}).get('age'), 0))}</td>"
        f"<td>{esc({'male': '男', 'female': '女'}.get((meta or {}).get('sex'), '—'))}</td>"
        f"<td>{esc(', '.join((meta or {}).get('reference_codes_active') or []) or '—')}</td>"
        f"<td>{esc(STATUS_ZH.get(status, status))}</td></tr>"
        for record, meta, status in (
            (record, manifest.get(record), status) for record, status in sorted(state.items())
        )
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PTB-XL 09000 心电图报告索引</title><style>{PAGE_CSS}</style></head>
<body><div class="wrap"><div class="card">
<h1>PTB-XL 09000 — 心电图与 ecgfeat 诊断报告</h1>
<p class="sub">共 {len(state)} 份报告。点击记录号查看 12 导联波形、ecgfeat 波形标注与规则引擎诊断。</p>
<table class="grid"><thead><tr><th>记录</th><th>年龄</th><th>性别</th><th>PTB-XL 标签</th><th>ecgfeat 总体状态</th></tr></thead>
<tbody>{rows}</tbody></table>
</div></div></body></html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def default_sample(manifest: Mapping[str, Mapping[str, Any]], batch_dir: Path, size: int) -> list[str]:
    """Pick a spread of records: the ones with agent diagnoses, then one per label."""
    chosen: list[str] = []
    seen: set[str] = set()

    def take(record: str) -> None:
        if record not in seen and (batch_dir / "features" / f"{record}_features.json").exists():
            seen.add(record)
            chosen.append(record)

    for path in sorted((batch_dir / "diagnoses").glob("*.md")):
        take(path.stem)

    by_code: dict[str, list[str]] = defaultdict(list)
    for record, meta in manifest.items():
        for code in meta.get("reference_codes_active") or []:
            by_code[code].append(record)
    for code in sorted(by_code, key=lambda c: (len(by_code[c]), c)):
        if len(chosen) >= size:
            break
        take(sorted(by_code[code])[0])
    for record in sorted(manifest):
        if len(chosen) >= size:
            break
        take(record)
    return chosen[:size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR / "ecg_plots")
    parser.add_argument("--records", nargs="*", default=None,
                        help="记录号，如 09000_hr（不带 _hr 也可）")
    parser.add_argument("--all", action="store_true", help="渲染批次内全部记录")
    parser.add_argument("--limit", type=int, default=24, help="默认抽样的记录数")
    parser.add_argument("--decimate", type=int, default=1,
                        help="波形抽取步长；>1 可显著减小 SVG 体积")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch_dir = args.batch_dir.resolve()
    manifest = load_manifest(batch_dir)
    scp = load_scp(args.metadata_dir)

    if args.records:
        records = [name if name.endswith("_hr") else f"{name}_hr" for name in args.records]
    elif args.all:
        records = sorted(
            path.name[: -len("_features.json")]
            for path in (batch_dir / "features").glob("*_features.json")
        )
    else:
        records = default_sample(manifest, batch_dir, args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Mapping[str, Any] | None, str]] = []
    for record in records:
        feature_path = batch_dir / "features" / f"{record}_features.json"
        if not feature_path.exists():
            print(f"[skip] {record}: 没有 features JSON")
            continue
        meta = manifest.get(record)
        record_path = Path((meta or {}).get("record_path") or "")
        if not record_path.name:
            record_path = PROJECT_ROOT / "data" / "ptb-xl" / "09000" / record
        if not record_path.with_suffix(".hea").exists():
            print(f"[skip] {record}: 找不到 WFDB 波形 {record_path}")
            continue

        features = json.loads(feature_path.read_text(encoding="utf-8"))
        signal, fs = load_signal(record_path)
        # Fiducial indices are in ecgfeat's internal sample rate; if extraction
        # resampled, overlaying them on the raw trace would silently misalign.
        internal_fs = (features.get("metadata") or {}).get("internal_fs")
        if internal_fs and abs(float(internal_fs) - fs) > 0.5:
            print(f"[skip] {record}: 波形 {fs:.0f} Hz 与 ecgfeat 内部 {internal_fs} Hz 不一致，标注会错位")
            continue
        page = render_page(record, features, signal, fs, meta, scp, batch_dir, args.decimate)
        output = args.output_dir / f"{record}.html"
        output.write_text(page, encoding="utf-8")
        status = str((features.get("clinical_interpretation") or {}).get("overall_status") or "")
        rendered.append((record, meta, status))
        print(f"[ok] {output}  ({output.stat().st_size / 1024:.0f} KB)")

    if rendered:
        render_index(rendered, args.output_dir, manifest)
        print(f"\n索引：{args.output_dir / 'index.html'}（本次 {len(rendered)} 份，索引覆盖目录内全部报告）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
