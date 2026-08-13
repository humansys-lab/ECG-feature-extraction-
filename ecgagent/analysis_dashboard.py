"""Render a batch analysis into a self-contained visual dashboard.

``analyze_results`` already writes ``analysis.json`` and the metric CSVs. This
module turns the same numbers into ``RESULTS.html``: one offline page of SVG
charts plus the underlying table, so per-family detection structure can be read
without scanning a 48-row CSV.

The page has no external assets, no fonts to install and no JavaScript
dependency for its static render.

Run it against an existing output directory with::

    python3 -m ecgagent.analysis_dashboard ptbxl_05000_ecgagent
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


# Chart colors come from the validated data-viz palette. Slots 1-3
# (blue/orange/aqua) clear the all-pairs CVD and normal-vision gates in both
# modes and carry the detection classes; blue/red against a muted neutral is
# the diverging pair used wherever a value means better/worse.
PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
        "hit": "rgba(11,11,11,0.05)",
        "tp": "#2a78d6",
        "fp": "#eb6834",
        "fn": "#1baf7a",
        "better": "#2a78d6",
        "worse": "#e34948",
        "same": "#898781",
        "accent": "#2a78d6",
        "context": "#898781",
        "delta_up": "#006300",
        "delta_down": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
        "hit": "rgba(255,255,255,0.07)",
        "tp": "#3987e5",
        "fp": "#d95926",
        "fn": "#199e70",
        "better": "#3987e5",
        "worse": "#e66767",
        "same": "#898781",
        "accent": "#3987e5",
        "context": "#898781",
        "delta_up": "#0ca30c",
        "delta_down": "#e66767",
    },
}

SCOPE_LABELS = {
    "direct": "Direct",
    "broad": "Broad family",
    "screening": "Screening family",
}
SOURCE_LABELS = {"baseline": "Rule baseline", "agent": "ECGAgent"}
EFFECT_LABELS = {"improved": "Improved", "unchanged": "Unchanged", "worsened": "Worsened"}
OUTCOME_LABELS = {
    "verified": "Verified",
    "unverified": "Unverified",
    "error": "Execution error",
    "stale": "Legacy protocol",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int(value: Any, default: int = 0) -> int:
    parsed = _number(value)
    return default if parsed is None else int(round(parsed))


def _fmt(value: Any, digits: int = 3) -> str:
    parsed = _number(value)
    return "—" if parsed is None else f"{parsed:.{digits}f}"


def _signed(value: Any, digits: int = 3) -> str:
    parsed = _number(value)
    if parsed is None:
        return "—"
    return f"{parsed:+.{digits}f}"


def _delta(
    value: float | None,
    reference: float | None,
) -> tuple[float | None, str, str]:
    """Change, its direction class and the glyph that carries the direction."""
    if value is None or reference is None:
        return None, "flat", "="
    change = value - reference
    if change > 5e-4:
        return change, "up", "▲"
    if change < -5e-4:
        return change, "down", "▼"
    return change, "flat", "="


def _text_width(text: str, size: float) -> float:
    """Approximate rendered width; CJK glyphs are full-width, latin is not."""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.55) for ch in text)


def _truncate(text: str, size: float, limit: float) -> str:
    if _text_width(text, size) <= limit:
        return text
    kept = ""
    for char in text:
        if _text_width(kept + char + "…", size) > limit:
            break
        kept += char
    return f"{kept}…" if kept else "…"


def _rrect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    tl: float = 0.0,
    tr: float = 0.0,
    br: float = 0.0,
    bl: float = 0.0,
) -> str:
    """SVG path for a rectangle with per-corner radii."""
    cap = min(abs(width), abs(height)) / 2
    tl, tr, br, bl = (max(0.0, min(r, cap)) for r in (tl, tr, br, bl))
    parts = [f"M{x + tl:.2f},{y:.2f}", f"H{x + width - tr:.2f}"]
    if tr:
        parts.append(f"A{tr:.2f},{tr:.2f} 0 0 1 {x + width:.2f},{y + tr:.2f}")
    parts.append(f"V{y + height - br:.2f}")
    if br:
        parts.append(
            f"A{br:.2f},{br:.2f} 0 0 1 {x + width - br:.2f},{y + height:.2f}"
        )
    parts.append(f"H{x + bl:.2f}")
    if bl:
        parts.append(f"A{bl:.2f},{bl:.2f} 0 0 1 {x:.2f},{y + height - bl:.2f}")
    parts.append(f"V{y + tl:.2f}")
    if tl:
        parts.append(f"A{tl:.2f},{tl:.2f} 0 0 1 {x + tl:.2f},{y:.2f}")
    parts.append("Z")
    return " ".join(parts)


TIP_SEPARATOR = "‖"


def _tip(*lines: str, named: bool = True) -> str:
    """Tooltip payload plus, for focusable marks, its accessible name.

    The reader splits ``data-tip`` on the separator and inserts each line with
    ``textContent``. The same text becomes an ``aria-label`` so the values are
    announced on focus without an ``aria-live`` region narrating every hover.
    Marks that are hover-only pass ``named=False``: their chart carries a
    summarising label instead, and the table view keeps the values reachable.
    """
    kept = [line for line in lines if line]
    payload = f'data-tip="{_esc(TIP_SEPARATOR.join(kept))}"'
    if not named:
        return payload
    return f'{payload} role="img" aria-label="{_esc(" · ".join(kept))}"'


def _svg_open(
    width: float,
    height: float,
    *,
    aria: str,
    title: str,
    interactive: bool = True,
    extra: str = "",
    klass: str = "chart",
) -> str:
    """Open an SVG element.

    A chart whose marks are focusable is a ``group`` — ``role="img"`` would make
    its own children presentational and hide the labels a keyboard user just
    moved onto.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="{klass}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
        f'height="{height:.0f}" {extra}role="{"group" if interactive else "img"}" '
        f'aria-label="{_esc(aria)}"><title>{_esc(title)}</title>'
    )


def _tick_step(span: float) -> int:
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if span / step <= 8:
            return step
    return 1000


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------


def _family_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fold the long baseline/agent metric rows into one row per family."""
    order: list[str] = []
    folded: dict[str, dict[str, Any]] = {}
    for row in metrics:
        key = str(row.get("category") or "")
        if not key:
            continue
        if key not in folded:
            order.append(key)
            folded[key] = {
                "key": key,
                "label": str(
                    row.get("category_label")
                    or key.replace("_", " ").title()
                ),
                "scope": str(row.get("semantic_scope") or "direct"),
            }
        source = str(row.get("source") or "")
        folded[key][source] = {
            "tp": _int(row.get("tp")),
            "fp": _int(row.get("fp")),
            "fn": _int(row.get("fn")),
            "tn": _int(row.get("tn")),
            "precision": _number(row.get("precision")),
            "recall": _number(row.get("recall")),
            "f1": _number(row.get("f1")),
        }

    blank = {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "precision": None, "recall": None, "f1": None,
    }
    rows: list[dict[str, Any]] = []
    for key in order:
        entry = folded[key]
        for source in SOURCE_LABELS:
            entry.setdefault(source, dict(blank))
        entry["support"] = max(
            entry[source]["tp"] + entry[source]["fn"] for source in SOURCE_LABELS
        )
        entry["activity"] = max(
            entry[source]["tp"] + entry[source]["fp"] + entry[source]["fn"]
            for source in SOURCE_LABELS
        )
        rows.append(entry)
    return rows


def _plot_f1(cell: Mapping[str, Any]) -> float | None:
    """F1 for plotting: zero hits with something to score is 0, not undefined.

    ``_metric_rows`` leaves F1 as ``None`` both when a family has no comparable
    records at all and when precision and recall are both 0. Only the first is
    genuinely undefined, so the charts separate them.
    """
    stored = _number(cell.get("f1"))
    if stored is not None:
        return stored
    if cell["tp"] + cell["fp"] + cell["fn"] == 0:
        return None
    return 0.0


# ---------------------------------------------------------------------------
# page pieces
# ---------------------------------------------------------------------------


def _stat_tile(
    label: str,
    value: str,
    *,
    delta: str = "",
    direction: str = "",
    note: str = "",
    hero: bool = False,
) -> str:
    classes = "tile hero" if hero else "tile"
    delta_html = (
        f'<span class="delta {direction}">{_esc(delta)}</span>' if delta else ""
    )
    note_html = f'<div class="tile-note">{_esc(note)}</div>' if note else ""
    return (
        f'<div class="{classes}">'
        f'<div class="tile-label">{_esc(label)}</div>'
        f'<div class="tile-value">{_esc(value)}{delta_html}</div>'
        f"{note_html}</div>"
    )


def _legend(items: Sequence[tuple[str, str, str]]) -> str:
    """items: (label, css color var, mark shape 'rect'|'dot'|'line')."""
    marks = []
    for label, color, shape in items:
        if shape == "dot":
            mark = f'<span class="key dot" style="background:{color}"></span>'
        elif shape == "line":
            mark = f'<span class="key line" style="background:{color}"></span>'
        else:
            mark = f'<span class="key rect" style="background:{color}"></span>'
        marks.append(f'<span class="legend-item">{mark}{_esc(label)}</span>')
    return f'<div class="legend">{"".join(marks)}</div>'


def _detection_chart(rows: Sequence[Mapping[str, Any]]) -> str:
    """Per-family diverging bars, faceted by source.

    Left of the centre line is what a source failed to report (FN); right of it
    is everything it did report, correct part first (TP) then wrong (FP). Source
    is a facet rather than a hue, so colour only ever means outcome.
    """
    active = [row for row in rows if row["activity"] > 0]
    if not active:
        return '<p class="empty">No comparable diagnostic family.</p>'
    active.sort(key=lambda row: (-row["support"], -row["activity"], row["key"]))

    pad, label_w, gap = 10.0, 268.0, 16.0
    facet_w, facet_gap, outer = 262.0, 34.0, 26.0
    row_h, bar_h = 26.0, 15.0
    top, bottom = 30.0, 44.0
    width = pad * 2 + label_w + gap + facet_w * 2 + facet_gap
    rows_h = row_h * len(active)
    height = top + rows_h + bottom

    left_max = max(row[source]["fn"] for row in active for source in SOURCE_LABELS)
    right_max = max(
        row[source]["tp"] + row[source]["fp"]
        for row in active
        for source in SOURCE_LABELS
    )
    span = max(1, left_max + right_max)
    unit = (facet_w - outer * 2) / span
    facet_x = [
        pad + label_w + gap,
        pad + label_w + gap + facet_w + facet_gap,
    ]
    centers = [x + outer + left_max * unit for x in facet_x]
    step = _tick_step(max(left_max, right_max) or 1)

    parts = [
        _svg_open(
            width,
            height,
            aria="Missed, true-positive and false-positive record counts by diagnostic family and source",
            title="Detection structure by diagnostic family",
            klass="chart rows-chart",
            extra=(
                f'data-chart="rows" data-row-h="{row_h:.0f}" '
                f'data-top="{top:.0f}" data-bottom="{bottom:.0f}" '
            ),
        )
    ]
    for center, title in zip(centers, SOURCE_LABELS.values()):
        parts.append(
            f'<text class="facet-title" x="{center:.1f}" y="16" '
            f'text-anchor="middle">{_esc(title)}</text>'
        )

    ticks: list[tuple[float, int]] = []
    parts.append(f'<g class="vrules" transform="translate(0,{top:.0f})">')
    for center in centers:
        value = -((left_max // step) * step)
        while value <= right_max:
            x = center + value * unit
            parts.append(
                f'<line class="{"zero" if value == 0 else "rule"}" '
                f'x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{rows_h:.0f}" '
                'data-rule="1"/>'
            )
            ticks.append((x, abs(value)))
            value += step
    parts.append("</g>")

    parts.append(f'<g class="rows" transform="translate(0,{top:.0f})">')
    for index, row in enumerate(active):
        bar_y = (row_h - bar_h) / 2
        middle = row_h / 2
        parts.append(
            f'<g class="row" data-scope="{_esc(row["scope"])}" '
            f'transform="translate(0,{index * row_h:.0f})">'
            f'<text class="row-label" x="{pad:.0f}" y="{middle + 4.4:.1f}">'
            f'{_esc(_truncate(str(row["label"]), 12.5, label_w - 46))}</text>'
            f'<text class="row-support" x="{pad + label_w:.0f}" '
            f'y="{middle + 4.2:.1f}" text-anchor="end">n={row["support"]}</text>'
        )
        for facet, source in enumerate(SOURCE_LABELS):
            cell = row[source]
            center = centers[facet]
            tip = _tip(
                f'{row["label"]} · {SOURCE_LABELS[source]}',
                f'True positive TP {cell["tp"]} | False positive FP {cell["fp"]}'
                f' | False negative FN {cell["fn"]}',
                f'Precision {_fmt(cell["precision"])} | '
                f'Recall {_fmt(cell["recall"])} | F1 {_fmt(_plot_f1(cell))}',
                f'Reference positive n={row["support"]} | Semantic scope {row["scope"]}',
            )
            parts.append(
                f'<g class="hit" tabindex="0" {tip}>'
                f'<rect class="hit-area" x="{facet_x[facet]:.1f}" y="0" '
                f'width="{facet_w:.0f}" height="{row_h:.0f}"/>'
            )
            if cell["fn"]:
                bar_w = cell["fn"] * unit
                parts.append(
                    '<path class="mk fn" d="'
                    f'{_rrect(center - bar_w, bar_y, bar_w, bar_h, tl=4, bl=4)}"/>'
                )
            drawn = 0.0
            for kind in ("tp", "fp"):
                value = cell[kind]
                if not value:
                    continue
                inset = 2.0 if drawn > 0 else 0.0
                bar_w = max(0.0, value * unit - inset)
                outermost = kind == "fp" or cell["fp"] == 0
                parts.append(
                    f'<path class="mk {kind}" d="'
                    + _rrect(
                        center + drawn + inset,
                        bar_y,
                        bar_w,
                        bar_h,
                        tr=4 if outermost else 0,
                        br=4 if outermost else 0,
                    )
                    + '"/>'
                )
                drawn += value * unit
            if cell["fn"] >= 4:
                parts.append(
                    f'<text class="mk-label" '
                    f'x="{center - cell["fn"] * unit - 5:.1f}" '
                    f'y="{middle + 4:.1f}" text-anchor="end">{cell["fn"]}</text>'
                )
            if cell["fp"] >= 4:
                end = center + (cell["tp"] + cell["fp"]) * unit
                parts.append(
                    f'<text class="mk-label" x="{end + 5:.1f}" '
                    f'y="{middle + 4:.1f}">{cell["fp"]}</text>'
                )
            parts.append("</g>")
        parts.append("</g>")
    parts.append("</g>")

    parts.append(f'<g class="xaxis" transform="translate(0,{top + rows_h:.0f})">')
    for center in centers:
        parts.append(
            f'<line class="axis-rule" x1="{center - left_max * unit - 6:.1f}" '
            f'y1="0" x2="{center + right_max * unit + 6:.1f}" y2="0"/>'
        )
    for x, value in sorted({(round(x, 1), value) for x, value in ticks}):
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="14" text-anchor="middle">'
            f"{value}</text>"
        )
    for center in centers:
        parts.append(
            f'<text class="axis-caption" x="{center - 8:.1f}" y="31" '
            'text-anchor="end">← Missed</text>'
            f'<text class="axis-caption" x="{center + 8:.1f}" y="31">Reported →</text>'
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _f1_chart(rows: Sequence[Mapping[str, Any]]) -> str:
    """Baseline → agent F1 per family, as a dumbbell coloured by direction."""
    prepared = []
    for row in rows:
        if row["activity"] <= 0 or row["support"] <= 0:
            continue
        base, agent = _plot_f1(row["baseline"]), _plot_f1(row["agent"])
        if base is None and agent is None:
            continue
        base, agent = base or 0.0, agent or 0.0
        prepared.append(
            {**row, "base_f1": base, "agent_f1": agent, "delta": agent - base}
        )
    if not prepared:
        return '<p class="empty">No diagnostic family has reference-positive samples.</p>'
    prepared.sort(key=lambda row: (-row["delta"], -row["support"]))

    pad, label_w, gap, plot_w, gutter = 10.0, 268.0, 16.0, 400.0, 118.0
    row_h, top, bottom = 24.0, 26.0, 34.0
    width = pad * 2 + label_w + gap + plot_w + gutter
    rows_h = row_h * len(prepared)
    height = top + rows_h + bottom
    plot_x = pad + label_w + gap

    ranked = sorted(prepared, key=lambda row: abs(row["delta"]), reverse=True)
    labelled = {row["key"] for row in ranked[:5] if abs(row["delta"]) > 1e-9}
    tick_values = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    parts = [
        _svg_open(
            width,
            height,
            aria="F1 change from rule baseline to ECGAgent for each diagnostic family",
            title="Diagnostic-family F1 change",
            klass="chart rows-chart",
            extra=(
                f'data-chart="rows" data-row-h="{row_h:.0f}" '
                f'data-top="{top:.0f}" data-bottom="{bottom:.0f}" '
            ),
        ),
        f'<g class="vrules" transform="translate(0,{top:.0f})">',
    ]
    for value in tick_values:
        x = plot_x + value * plot_w
        parts.append(
            f'<line class="{"zero" if value == 0 else "rule"}" x1="{x:.1f}" '
            f'y1="0" x2="{x:.1f}" y2="{rows_h:.0f}" data-rule="1"/>'
        )
    parts.append("</g>")

    parts.append(f'<g class="rows" transform="translate(0,{top:.0f})">')
    for index, row in enumerate(prepared):
        middle = row_h / 2
        direction = (
            "better" if row["delta"] > 1e-9
            else "worse" if row["delta"] < -1e-9 else "same"
        )
        x1 = plot_x + row["base_f1"] * plot_w
        x2 = plot_x + row["agent_f1"] * plot_w
        tip = _tip(
            f'{row["label"]} ({row["scope"]})',
            f'Rule baseline F1 {row["base_f1"]:.3f} → ECGAgent F1 '
            f'{row["agent_f1"]:.3f} ({_signed(row["delta"])})',
            f'Baseline TP {row["baseline"]["tp"]}/FP {row["baseline"]["fp"]}'
            f'/FN {row["baseline"]["fn"]} | Agent TP {row["agent"]["tp"]}'
            f'/FP {row["agent"]["fp"]}/FN {row["agent"]["fn"]}',
            f'Reference positive n={row["support"]}',
        )
        parts.append(
            f'<g class="row" data-scope="{_esc(row["scope"])}" '
            f'transform="translate(0,{index * row_h:.0f})">'
            f'<g class="hit" tabindex="0" {tip}>'
            f'<rect class="hit-area" x="{pad:.0f}" y="0" '
            f'width="{width - pad * 2:.0f}" height="{row_h:.0f}"/>'
            f'<text class="row-label" x="{pad:.0f}" y="{middle + 4.2:.1f}">'
            f'{_esc(_truncate(str(row["label"]), 12.5, label_w - 10))}</text>'
            f'<line class="link {direction}" x1="{x1:.1f}" y1="{middle:.1f}" '
            f'x2="{x2:.1f}" y2="{middle:.1f}"/>'
            f'<circle class="dot base" cx="{x1:.1f}" cy="{middle:.1f}" r="4.5"/>'
            f'<circle class="dot {direction}" cx="{x2:.1f}" cy="{middle:.1f}" '
            'r="5"/>'
        )
        if row["key"] in labelled:
            forward = x2 >= x1
            parts.append(
                f'<text class="mk-label" x="{x2 + (10 if forward else -10):.1f}" '
                f'y="{middle + 3.8:.1f}" '
                f'text-anchor="{"start" if forward else "end"}">'
                f'{row["agent_f1"]:.2f}</text>'
            )
        parts.append(
            f'<text class="row-support" x="{width - pad:.0f}" '
            f'y="{middle + 4.2:.1f}" text-anchor="end">n={row["support"]}'
            f' {_signed(row["delta"], 2)}</text></g></g>'
        )
    parts.append("</g>")

    parts.append(f'<g class="xaxis" transform="translate(0,{top + rows_h:.0f})">')
    parts.append(
        f'<line class="axis-rule" x1="{plot_x:.1f}" y1="0" '
        f'x2="{plot_x + plot_w:.1f}" y2="0"/>'
    )
    for value in tick_values:
        parts.append(
            f'<text class="tick" x="{plot_x + value * plot_w:.1f}" y="14" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    parts.append(
        f'<text class="axis-caption" x="{plot_x + plot_w / 2:.1f}" y="29" '
        'text-anchor="middle">F1 (agreement with PTB-XL labels)</text></g></svg>'
    )
    return "".join(parts)


def _waffle_chart(record_rows: Sequence[Mapping[str, Any]]) -> str:
    """One cell per record, coloured by whether its label match moved."""
    cells = [
        row for row in record_rows
        if str(row.get("label_effect") or "") in EFFECT_LABELS
    ]
    if not cells:
        return ""
    order = {"improved": 0, "unchanged": 1, "worsened": 2}
    cells.sort(
        key=lambda row: (order[str(row["label_effect"])],
                         str(row.get("record") or ""))
    )
    columns = min(30, max(10, int(math.ceil(math.sqrt(len(cells) * 2.4)))))
    rows_count = int(math.ceil(len(cells) / columns))
    size = 15.0 if len(cells) <= 200 else 10.0
    pitch = size + 3.0
    width = columns * pitch + 2
    height = rows_count * pitch + 2

    tally = Counter(str(row["label_effect"]) for row in cells)
    parts = [
        _svg_open(
            width,
            height,
            aria=(
                f"Label-agreement change for {len(cells)} records: "
                + ", ".join(
                    f"{EFFECT_LABELS[key]} {tally[key]}"
                    for key in ("improved", "unchanged", "worsened")
                    if tally[key]
                )
            ),
            title="Record-level label effect",
            interactive=False,
            klass="chart waffle",
        )
    ]
    for index, row in enumerate(cells):
        effect = str(row["label_effect"])
        x = 1 + (index % columns) * pitch
        y = 1 + (index // columns) * pitch
        tip = _tip(
            f'{row.get("record", "")} · {EFFECT_LABELS[effect]}',
            f'Reference categories: {str(row.get("reference_categories") or "None")}',
            f'Baseline categories: {str(row.get("baseline_categories") or "None")}',
            f'Agent categories: {str(row.get("agent_categories") or "None")}',
            named=False,
        )
        parts.append(
            f'<path class="cell {effect}" {tip} '
            f'd="{_rrect(x, y, size, size, tl=3, tr=3, br=3, bl=3)}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _columns_chart(
    pairs: Sequence[tuple[str, float]],
    *,
    aria: str,
    caption: str = "",
) -> str:
    """Single-series columns for the small run-health panels."""
    if not pairs:
        return '<p class="empty">No data.</p>'
    pad_l, pad_r, top, bottom = 34.0, 12.0, 22.0, 38.0
    plot_h = 118.0
    slot = max(30.0, min(62.0, 330.0 / len(pairs)))
    width = pad_l + slot * len(pairs) + pad_r
    height = top + plot_h + bottom
    peak = max((value for _, value in pairs), default=0.0) or 1.0
    bar_w = min(24.0, slot - 12.0)

    parts = [_svg_open(width, height, aria=aria, title=aria, klass="chart panel")]
    step, line = _tick_step(peak), 0.0
    while line <= peak + 1e-9:
        y = top + plot_h - (line / peak) * plot_h
        parts.append(
            f'<line class="rule" x1="{pad_l:.1f}" y1="{y:.1f}" '
            f'x2="{width - pad_r:.1f}" y2="{y:.1f}"/>'
            f'<text class="tick" x="{pad_l - 6:.1f}" y="{y + 3.6:.1f}" '
            f'text-anchor="end">{line:.0f}</text>'
        )
        line += step

    # Thin the category ticks until neighbouring labels stop touching, rather
    # than truncating them into unreadable stubs.
    stride = 1
    while stride < len(pairs):
        budget = slot * stride - 5
        if all(
            _text_width(label, 11) <= budget for label, _ in pairs[::stride]
        ):
            break
        stride += 1

    for index, (label, value) in enumerate(pairs):
        bar_h = (value / peak) * plot_h
        x = pad_l + index * slot + (slot - bar_w) / 2
        y = top + plot_h - bar_h
        if bar_h > 0.5:
            parts.append(
                f'<path class="mk accent" tabindex="0" '
                f'{_tip(label, f"{value:,.0f}")} '
                f'd="{_rrect(x, y, bar_w, bar_h, tl=4, tr=4)}"/>'
            )
        parts.append(
            f'<text class="mk-label" x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" '
            f'text-anchor="middle">{value:,.0f}</text>'
        )
        if index % stride == 0:
            parts.append(
                f'<text class="tick" x="{x + bar_w / 2:.1f}" '
                f'y="{top + plot_h + 16:.1f}" text-anchor="middle">'
                f"{_esc(_truncate(label, 11, slot * stride - 5))}</text>"
            )
    parts.append(
        f'<line class="axis-rule" x1="{pad_l:.1f}" y1="{top + plot_h:.1f}" '
        f'x2="{width - pad_r:.1f}" y2="{top + plot_h:.1f}"/>'
    )
    if caption:
        parts.append(
            f'<text class="axis-caption" x="{width / 2:.1f}" '
            f'y="{height - 8:.1f}" text-anchor="middle">{_esc(caption)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _hbars_chart(
    pairs: Sequence[tuple[str, float]],
    *,
    aria: str,
    klass: str = "accent",
) -> str:
    """Horizontal bars — the form for categories with long names."""
    if not pairs:
        return '<p class="empty">No data.</p>'
    pad, label_w, gap, plot_w = 6.0, 168.0, 10.0, 128.0
    row_h, bar_h = 26.0, 13.0
    width = pad * 2 + label_w + gap + plot_w + 34
    height = row_h * len(pairs) + 6
    peak = max((value for _, value in pairs), default=0.0) or 1.0
    plot_x = pad + label_w + gap

    parts = [_svg_open(width, height, aria=aria, title=aria, klass="chart panel")]
    for index, (label, value) in enumerate(pairs):
        y = 3 + index * row_h
        bar_w = (value / peak) * plot_w
        parts.append(
            f'<g class="hit" tabindex="0" {_tip(label, f"{value:,.0f}")}>'
            f'<rect class="hit-area" x="{pad:.0f}" y="{y:.1f}" '
            f'width="{width - pad * 2:.0f}" height="{row_h:.0f}"/>'
            f'<text class="row-label mono" x="{pad:.0f}" '
            f'y="{y + row_h / 2 + 4:.1f}">'
            f"{_esc(_truncate(label, 11, label_w))}</text>"
        )
        if bar_w > 0.5:
            parts.append(
                f'<path class="mk {klass}" d="'
                f'{_rrect(plot_x, y + (row_h - bar_h) / 2, bar_w, bar_h, tr=4, br=4)}"/>'
            )
        parts.append(
            f'<text class="mk-label" x="{plot_x + bar_w + 6:.1f}" '
            f'y="{y + row_h / 2 + 4:.1f}">{value:,.0f}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _share_bar(parts_in: Sequence[tuple[str, int, str]], *, aria: str) -> str:
    """Horizontal 100% stacked bar with a 2px surface gap between segments."""
    visible = [item for item in parts_in if item[1] > 0]
    total = sum(count for _, count, _ in visible)
    if total <= 0:
        return ""
    width, height = 640.0, 26.0
    pieces = [_svg_open(width, height, aria=aria, title=aria, klass="chart share")]
    x = 0.0
    for index, (label, count, klass) in enumerate(visible):
        segment = width * count / total
        inset = 2.0 if index else 0.0
        first, last = index == 0, index == len(visible) - 1
        pieces.append(
            f'<path class="mk {klass}" tabindex="0" '
            f'{_tip(label, f"{count} records · {count / total:.1%}")} d="'
            + _rrect(
                x + inset,
                0,
                max(0.0, segment - inset),
                height,
                tl=4 if first else 0,
                bl=4 if first else 0,
                tr=4 if last else 0,
                br=4 if last else 0,
            )
            + '"/>'
        )
        x += segment
    pieces.append("</svg>")
    return "".join(pieces)


def _code_chart(code_counts: Sequence[Mapping[str, Any]]) -> str:
    """Top diagnosis codes: the agent is the subject, the baseline is context."""
    rows = [
        {
            "code": str(row.get("code") or ""),
            "agent": _int(row.get("agent_count")),
            "baseline": _int(row.get("baseline_count")),
        }
        for row in code_counts
        if str(row.get("code") or "")
    ]
    if not rows:
        return ""
    rows.sort(key=lambda row: (-max(row["agent"], row["baseline"]), row["code"]))
    rows = rows[:14]

    pad, label_w, gap, plot_w = 10.0, 244.0, 12.0, 380.0
    row_h, bar_h = 32.0, 11.0
    top, bottom = 12.0, 34.0
    width = pad * 2 + label_w + gap + plot_w + 54
    rows_h = row_h * len(rows)
    height = top + rows_h + bottom
    peak = max(max(row["agent"], row["baseline"]) for row in rows) or 1
    plot_x = pad + label_w + gap
    step = _tick_step(peak)

    parts = [
        _svg_open(
            width,
            height,
            aria="Counts of the most frequent diagnosis codes in the rule baseline and ECGAgent",
            title="Diagnosis-code frequency",
        )
    ]
    value = 0
    while value <= peak:
        x = plot_x + (value / peak) * plot_w
        parts.append(
            f'<line class="{"zero" if value == 0 else "rule"}" x1="{x:.1f}" '
            f'y1="{top:.1f}" x2="{x:.1f}" y2="{top + rows_h:.1f}"/>'
            f'<text class="tick" x="{x:.1f}" y="{top + rows_h + 16:.1f}" '
            f'text-anchor="middle">{value}</text>'
        )
        value += step
    for index, row in enumerate(rows):
        y = top + index * row_h
        agent_y = y + 4.0 + bar_h
        parts.append(
            '<g class="hit" tabindex="0" '
            + _tip(
                row["code"],
                f'ECGAgent {row["agent"]} | Rule baseline {row["baseline"]}',
                f'Difference {row["agent"] - row["baseline"]:+d}',
            )
            + f'><rect class="hit-area" x="{pad:.0f}" y="{y:.1f}" '
            f'width="{width - pad * 2:.0f}" height="{row_h:.0f}"/>'
            f'<text class="row-label mono" x="{pad:.0f}" '
            f'y="{y + row_h / 2 + 4:.1f}">'
            f'{_esc(_truncate(row["code"], 11.5, label_w - 8))}</text>'
        )
        for bar_y, key, klass in (
            (y + 2.0, "baseline", "context"),
            (agent_y, "agent", "accent"),
        ):
            bar_w = (row[key] / peak) * plot_w
            if bar_w < 0.6:
                continue
            parts.append(
                f'<path class="mk {klass}" '
                f'd="{_rrect(plot_x, bar_y, bar_w, bar_h, tr=4, br=4)}"/>'
            )
        parts.append(
            f'<text class="mk-label" '
            f'x="{plot_x + (row["agent"] / peak) * plot_w + 6:.1f}" '
            f'y="{agent_y + bar_h - 1:.1f}">{row["agent"]}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _metrics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """The table twin of the family charts — every plotted value, readable."""
    columns = "".join(
        f'<th scope="col" class="num{" grp" if index == 0 else ""}">{name}</th>'
        for index, name in enumerate(("TP", "FP", "FN", "P", "R", "F1"))
    )
    body = []
    for row in sorted(rows, key=lambda item: (-item["support"], item["key"])):
        cells = [
            f'<th scope="row"><span class="cat">{_esc(row["label"])}</span>'
            f'<span class="cat-key">{_esc(row["key"])}</span></th>',
            f'<td>{_esc(SCOPE_LABELS.get(row["scope"], row["scope"]))}</td>',
            f'<td class="num">{row["support"]}</td>',
        ]
        for source in SOURCE_LABELS:
            cell = row[source]
            cells.extend(
                [
                    f'<td class="num grp">{cell["tp"]}</td>',
                    f'<td class="num">{cell["fp"]}</td>',
                    f'<td class="num">{cell["fn"]}</td>',
                    f'<td class="num">{_fmt(cell["precision"])}</td>',
                    f'<td class="num">{_fmt(cell["recall"])}</td>',
                    f'<td class="num">{_fmt(cell["f1"])}</td>',
                ]
            )
        body.append(f'<tr data-scope="{_esc(row["scope"])}">{"".join(cells)}</tr>')
    return (
        '<div class="table-wrap"><table class="metrics">'
        "<caption>Label-agreement details by diagnostic family; n is the number of reference-positive records. "
        "An em dash indicates that the family has no comparable sample.</caption>"
        '<thead><tr><td colspan="3"></td>'
        '<th colspan="6" scope="colgroup" class="grp">Rule baseline</th>'
        '<th colspan="6" scope="colgroup" class="grp">ECGAgent</th></tr>'
        '<tr><th scope="col">Diagnostic family</th><th scope="col">Semantic scope</th>'
        '<th scope="col" class="num">n</th>'
        f"{columns}{columns}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _css() -> str:
    light = PALETTES["light"]
    dark = PALETTES["dark"]

    def block(palette: Mapping[str, str]) -> str:
        return "".join(f"--{key}:{value};" for key, value in palette.items())

    return f"""
:root{{color-scheme:light;{block(light)}
--font:system-ui,-apple-system,"Segoe UI","PingFang SC","Hiragino Sans GB",
"Microsoft YaHei",sans-serif;}}
@media (prefers-color-scheme:dark){{
:root:where(:not([data-theme="light"])){{color-scheme:dark;{block(dark)}}}}}
:root[data-theme="dark"]{{color-scheme:dark;{block(dark)}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--plane);color:var(--ink);font-family:var(--font);
line-height:1.6;-webkit-font-smoothing:antialiased}}
.page{{max-width:1180px;margin:0 auto;padding:32px 20px 72px}}
h1{{font-size:26px;line-height:1.3;margin:0 0 6px}}
h2{{font-size:17px;margin:0 0 4px;letter-spacing:.01em}}
h3{{font-size:13px;margin:0 0 10px;color:var(--ink2);font-weight:600}}
p{{margin:0 0 10px}}
a{{color:var(--accent)}}
.sub{{color:var(--ink2);font-size:13.5px;margin:0}}
.meta{{color:var(--muted);font-size:12.5px;margin:6px 0 0;
font-variant-numeric:tabular-nums}}
.topbar{{display:flex;justify-content:space-between;align-items:flex-start;
gap:16px;flex-wrap:wrap;margin-bottom:20px}}
.note{{border-left:3px solid var(--axis);padding:8px 0 8px 14px;
color:var(--ink2);font-size:13px;margin:18px 0}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:20px 22px 18px;margin:16px 0}}
.card-head{{margin-bottom:14px}}
.card-note{{color:var(--muted);font-size:12.5px;margin:10px 0 0}}
.tiles{{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
margin:16px 0}}
.tile{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:14px 16px 15px}}
.tile.hero{{grid-column:span 2;min-width:260px}}
.tile-label{{color:var(--ink2);font-size:12.5px}}
.tile-value{{font-size:27px;font-weight:600;line-height:1.25;margin-top:2px;
display:flex;align-items:baseline;gap:9px}}
.tile.hero .tile-value{{font-size:50px;letter-spacing:-.02em}}
.tile-note{{color:var(--muted);font-size:12px;margin-top:2px}}
.delta{{font-size:13.5px;font-weight:600}}
.delta.up{{color:var(--delta_up)}}
.delta.down{{color:var(--delta_down)}}
.delta.flat{{color:var(--muted)}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 18px;margin:2px 0 14px;
color:var(--ink2);font-size:12.5px}}
.legend-item{{display:inline-flex;align-items:center;gap:7px}}
.key{{display:inline-block;flex:none}}
.key.rect{{width:12px;height:12px;border-radius:3px}}
.key.dot{{width:11px;height:11px;border-radius:50%}}
.key.line{{width:16px;height:2px;border-radius:1px}}
.filters{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:26px 0 4px}}
.filters .label{{color:var(--ink2);font-size:12.5px}}
button{{font:inherit;color:var(--ink2);background:var(--surface);cursor:pointer;
border:1px solid var(--border);border-radius:999px;padding:5px 14px;font-size:12.5px}}
button[aria-pressed="true"]{{background:var(--accent);border-color:var(--accent);
color:#fff}}
button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.chart{{max-width:100%;height:auto;display:block;overflow:visible}}
.chart-scroll{{overflow-x:auto;overflow-y:hidden;padding-bottom:2px}}
.panels{{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}}
.panel-card{{border:1px solid var(--border);border-radius:10px;padding:14px 14px 6px}}
text{{font-family:var(--font)}}
.row-label{{font-size:12.5px;fill:var(--ink)}}
.row-support{{font-size:11.5px;fill:var(--muted);font-variant-numeric:tabular-nums}}
.facet-title{{font-size:12.5px;font-weight:600;fill:var(--ink2)}}
.tick{{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}}
.axis-caption{{font-size:11px;fill:var(--muted)}}
.mk-label{{font-size:11px;fill:var(--ink2);font-variant-numeric:tabular-nums}}
.rule{{stroke:var(--grid);stroke-width:1}}
.zero{{stroke:var(--axis);stroke-width:1}}
.axis-rule{{stroke:var(--axis);stroke-width:1}}
.marker{{stroke:var(--ink2);stroke-width:1;stroke-dasharray:none;opacity:.55}}
.mk.tp{{fill:var(--tp)}}
.mk.fp{{fill:var(--fp)}}
.mk.fn{{fill:var(--fn)}}
.mk.accent{{fill:var(--accent)}}
.mk.context{{fill:var(--context)}}
.mk.improved,.mk.verified{{fill:var(--better)}}
.mk.worsened,.mk.error,.mk.worse{{fill:var(--worse)}}
.mk.unchanged,.mk.unverified,.mk.stale{{fill:var(--same)}}
.hit-area{{fill:transparent}}
.hit:hover .hit-area,.hit:focus-visible .hit-area{{fill:var(--hit)}}
.hit{{outline:none}}
.hit:focus-visible{{outline:2px solid var(--accent);outline-offset:-2px}}
.link{{stroke-width:2;stroke-linecap:round}}
.link.better{{stroke:var(--better)}}
.link.worse{{stroke:var(--worse)}}
.link.same{{stroke:var(--same)}}
.dot{{stroke:var(--surface);stroke-width:2}}
.dot.base{{fill:var(--same)}}
.dot.better{{fill:var(--better)}}
.dot.worse{{fill:var(--worse)}}
.dot.same{{fill:var(--same)}}
.cell.improved{{fill:var(--better)}}
.cell.worsened{{fill:var(--worse)}}
.cell.unchanged{{fill:var(--same)}}
.cell:hover{{stroke:var(--ink);stroke-width:2}}
.table-wrap{{overflow-x:auto;margin-top:6px}}
.section-break{{margin:38px 0 0}}
.section-break h2{{margin-bottom:2px}}
.share{{margin-bottom:2px}}
table.metrics{{border-collapse:collapse;font-size:12.5px;width:100%;
font-variant-numeric:tabular-nums}}
table.metrics caption{{caption-side:bottom;text-align:left;color:var(--muted);
font-size:12px;padding-top:10px}}
table.metrics th,table.metrics td{{padding:6px 9px;text-align:left;
border-bottom:1px solid var(--grid);white-space:nowrap}}
table.metrics thead th{{color:var(--ink2);font-weight:600;
border-bottom:1px solid var(--axis)}}
table.metrics .num{{text-align:right}}
table.metrics .grp{{border-left:1px solid var(--grid)}}
table.metrics tbody th{{font-weight:400}}
table.metrics tbody tr:hover{{background:var(--hit)}}
.cat{{display:block}}
.cat-key{{display:block;color:var(--muted);font-size:11px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}}
.empty{{color:var(--muted);font-size:13px}}
.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;
overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}}
#tip{{position:fixed;z-index:20;pointer-events:none;opacity:0;transition:opacity .1s;
background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:8px 11px;font-size:12.5px;line-height:1.55;color:var(--ink2);
box-shadow:0 6px 22px rgba(0,0,0,.16);max-width:330px}}
#tip.on{{opacity:1}}
#tip .t0{{color:var(--ink);font-weight:600}}
footer{{color:var(--muted);font-size:12.5px;margin-top:32px}}
footer li{{margin-bottom:3px}}
@media print{{.filters,#theme{{display:none}}.card{{break-inside:avoid}}}}
"""


def _js() -> str:
    return """
(function(){
 var root=document.documentElement;
 var saved=null;
 try{saved=localStorage.getItem('ecgagent-theme');}catch(e){}
 if(saved){root.setAttribute('data-theme',saved);}
 var toggle=document.getElementById('theme');
 if(toggle){toggle.addEventListener('click',function(){
   var dark=root.getAttribute('data-theme')==='dark'||
     (!root.getAttribute('data-theme')&&
      window.matchMedia('(prefers-color-scheme:dark)').matches);
   var next=dark?'light':'dark';
   root.setAttribute('data-theme',next);
   try{localStorage.setItem('ecgagent-theme',next);}catch(e){}
   toggle.setAttribute('aria-label',next==='dark'?'Switch to light theme':'Switch to dark theme');
 });}

 function relayout(scope){
  document.querySelectorAll('svg[data-chart="rows"]').forEach(function(svg){
   var rowH=parseFloat(svg.dataset.rowH);
   var top=parseFloat(svg.dataset.top);
   var bottom=parseFloat(svg.dataset.bottom);
   var visible=0;
   svg.querySelectorAll('g.row').forEach(function(row){
    var keep=scope==='all'||row.dataset.scope===scope;
    row.style.display=keep?'':'none';
    if(keep){row.setAttribute('transform','translate(0,'+(visible*rowH)+')');
      visible++;}
   });
   var rowsH=visible*rowH;
   svg.querySelectorAll('[data-rule="1"]').forEach(function(line){
    line.setAttribute('y2',rowsH);
   });
   var axis=svg.querySelector('g.xaxis');
   if(axis){axis.setAttribute('transform','translate(0,'+(top+rowsH)+')');}
   var box=svg.getAttribute('viewBox').split(/\\s+/);
   var height=top+rowsH+bottom;
   svg.setAttribute('viewBox',box[0]+' '+box[1]+' '+box[2]+' '+height);
   svg.setAttribute('height',height);
  });
  document.querySelectorAll('table.metrics tbody tr').forEach(function(row){
   row.style.display=(scope==='all'||row.dataset.scope===scope)?'':'none';
  });
 }
 document.querySelectorAll('#scope button').forEach(function(button){
  button.addEventListener('click',function(){
   document.querySelectorAll('#scope button').forEach(function(other){
    other.setAttribute('aria-pressed',String(other===button));
   });
   relayout(button.dataset.scope);
  });
 });

 var tip=document.getElementById('tip');
 function show(target,x,y){
  var text=target.dataset.tip;
  if(!text){return;}
  tip.textContent='';
  text.split('\\u2016').forEach(function(line,index){
   var div=document.createElement('div');
   div.className='t'+index;
   div.textContent=line;
   tip.appendChild(div);
  });
  tip.classList.add('on');
  var box=tip.getBoundingClientRect();
  var left=Math.min(Math.max(8,x+14),window.innerWidth-box.width-8);
  var topPx=Math.min(Math.max(8,y+14),window.innerHeight-box.height-8);
  tip.style.left=left+'px';
  tip.style.top=topPx+'px';
 }
 function hide(){tip.classList.remove('on');}
 document.addEventListener('pointermove',function(event){
  var target=event.target.closest('[data-tip]');
  if(target){show(target,event.clientX,event.clientY);}else{hide();}
 });
 document.addEventListener('pointerleave',hide);
 document.addEventListener('focusin',function(event){
  var target=event.target.closest('[data-tip]');
  if(!target){hide();return;}
  var box=target.getBoundingClientRect();
  show(target,box.left+box.width/2,box.top+box.height/2);
 });
 document.addEventListener('focusout',hide);
 document.addEventListener('keydown',function(event){
  if(event.key==='Escape'){hide();}
 });
})();
"""


OUTCOME_CLASSES = {
    "verified": "verified",
    "unverified": "unverified",
    "stale": "stale",
    "error": "error",
}
OUTCOME_TOKENS = {
    "verified": "better",
    "unverified": "same",
    "stale": "same",
    "error": "worse",
}


def _execution_section(summary: Mapping[str, Any]) -> tuple[str, str]:
    """Share bar plus legend for how the run's records ended up."""
    execution = summary.get("execution") or {}
    ordered = [
        (OUTCOME_LABELS.get(key, key), int(execution.get(key) or 0), key)
        for key in OUTCOME_LABELS
        if int(execution.get(key) or 0) > 0
    ]
    if not ordered:
        return "", ""
    bar = _share_bar(
        [
            (label, count, OUTCOME_CLASSES.get(key, "unverified"))
            for label, count, key in ordered
        ],
        aria="Record execution-outcome distribution",
    )
    legend = _legend(
        [
            (f"{label} {count}", f"var(--{OUTCOME_TOKENS.get(key, 'same')})", "rect")
            for label, count, key in ordered
        ]
    )
    return bar, legend


def _build_html(
    summary: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    *,
    record_rows: Sequence[Mapping[str, Any]] = (),
    code_counts: Sequence[Mapping[str, Any]] = (),
    source_dir: Path | None = None,
) -> str:
    rows = _family_rows(metrics)
    direct = summary.get("direct_label_agreement") or {}
    baseline = direct.get("baseline") or {}
    agent = direct.get("agent") or {}
    execution = summary.get("execution") or {}
    verified = int(execution.get("verified", 0) or 0)
    record_count = int(summary.get("record_count", 0) or 0)
    evaluated = int(summary.get("label_evaluated_records", 0) or 0)

    def delta_tile(label: str, key: str, note: str, *, hero: bool = False) -> str:
        agent_value = _number(agent.get(key))
        change, direction, arrow = _delta(agent_value, _number(baseline.get(key)))
        return _stat_tile(
            label,
            _fmt(agent_value),
            delta=f"{arrow} {_signed(change)}" if change is not None else "",
            direction=direction,
            note=note,
            hero=hero,
        )

    tiles = [
        delta_tile(
            "ECGAgent Direct-Family F1",
            "f1",
            f"Rule baseline {_fmt(baseline.get('f1'))} | semantic_scope=direct only",
            hero=True,
        ),
        delta_tile("Precision", "precision", f"Baseline {_fmt(baseline.get('precision'))}"),
        delta_tile("Recall", "recall", f"Baseline {_fmt(baseline.get('recall'))}"),
        _stat_tile(
            "Verified Records",
            f"{verified}/{record_count}",
            note=(
                f"Label-comparable {evaluated} | Coverage "
                f"{(verified / record_count * 100) if record_count else 0:.0f}%"
            ),
        ),
        _stat_tile(
            "Tool Calls / Revisions",
            f"{int(summary.get('total_tool_attempts', 0) or 0):,}"
            f" / {int(summary.get('total_revisions', 0) or 0):,}",
            note="Cumulative across verified records",
        ),
    ]

    runtimes = (summary.get("runtime_seconds_by_outcome") or {}).get("verified") or {}
    if runtimes.get("median") is not None:
        tiles.append(
            _stat_tile(
                "Median Runtime per Record",
                f"{_fmt(runtimes.get('median'), 0)}s",
                note=f"P90 {_fmt(runtimes.get('p90'), 0)}s"
                     f" | Maximum {_fmt(runtimes.get('max'), 0)}s",
            )
        )

    scopes_present = sorted({row["scope"] for row in rows if row["activity"] > 0})
    scope_buttons = ['<div id="scope" class="filters">'
                     '<span class="label">Semantic scope</span>'
                     '<button data-scope="all" aria-pressed="true">All</button>']
    for scope in scopes_present:
        scope_buttons.append(
            f'<button data-scope="{_esc(scope)}" aria-pressed="false">'
            f"{_esc(SCOPE_LABELS.get(scope, scope))}</button>"
        )
    scope_buttons.append(
        '<span class="label">(applies to family-level charts and the detail table below)</span></div>'
    )

    # Families neither source touched carry no information; naming them beats
    # drawing 24 empty rows, but only while the list is shorter than the chart.
    silent = [row for row in rows if row["activity"] == 0]
    drawn = len(rows) - len(silent)
    silent_note = ""
    if silent and drawn:
        names = [_esc(row["label"]) for row in silent[:12]]
        if len(silent) > 12:
            names.append(f"and {len(silent) - 12} more")
        silent_note = (
            f'<p class="card-note">Not plotted: {len(silent)} families with neither predictions nor reference positives: '
            + ", ".join(names) + ".</p>"
        )

    detection_legend = _legend(
        [
            ("True positive TP", "var(--tp)", "rect"),
            ("False positive FP", "var(--fp)", "rect"),
            ("False negative FN", "var(--fn)", "rect"),
        ]
    )
    f1_legend = _legend(
        [
            ("Rule baseline", "var(--same)", "dot"),
            ("ECGAgent improved", "var(--better)", "dot"),
            ("ECGAgent worsened", "var(--worse)", "dot"),
            ("Unchanged", "var(--same)", "line"),
        ]
    )

    body = [
        '<div class="page">',
        '<div class="topbar"><div>',
        "<h1>ECGAgent Label-Agreement Analysis</h1>",
        '<p class="sub">Diagnostic-family comparison between the rule baseline and ECGAgent '
        "(PTB-XL record-level reference labels)</p>",
        f'<p class="meta">{_esc(source_dir.name if source_dir else "")}'
        f' · Records {record_count} · Status '
        f'{_esc(str(summary.get("analysis_status", "partial")).upper())}'
        f' · Generated {_dt.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>',
        "</div>",
        '<button id="theme" type="button" aria-label="Switch color theme">◐ Theme</button>',
        "</div>",
        '<p class="note">For research and engineering use; not a medical device. PTB-XL provides record-level labels only. '
        "Label agreement is not clinical accuracy, and an unlabeled abnormality is not automatically a false positive.</p>",
        f'<div class="tiles">{"".join(tiles)}</div>',
    ]

    if summary.get("analysis_scope"):
        body.append(
            f'<p class="note">Analysis scope: {_esc(summary["analysis_scope"])}'
            + (
                f' (target {summary.get("trial_target_record_count")} records, '
                f'locked {summary.get("stopped_locked_record_count")} records)'
                if summary.get("trial_target_record_count") else ""
            )
            + ".</p>"
        )

    body.append("".join(scope_buttons))
    body.append(
        '<section class="card"><div class="card-head">'
        "<h2>Detection Structure by Diagnostic Family</h2>"
        '<h3>Reference positives missed are left of center; reported conclusions are right of center '
        "(true positives followed by false positives). Both facets share the same record-count scale.</h3></div>"
        + detection_legend
        + '<div class="chart-scroll">'
        + _detection_chart(rows)
        + "</div>"
        + silent_note
        + '<p class="card-note">n is the number of reference-positive records for the family. All values appear in the detail table below. '
        "Hover over or keyboard-focus a row to see complete TP/FP/FN and P/R/F1 values.</p></section>"
    )
    body.append(
        '<section class="card"><div class="card-head">'
        "<h2>Family-Level F1: Rule Baseline → ECGAgent</h2>"
        '<h3>Sorted by change magnitude; line color indicates direction, with reference-positive count and F1 difference on the right</h3>'
        "</div>"
        + f1_legend
        + '<div class="chart-scroll">'
        + _f1_chart(rows)
        + "</div>"
        + '<p class="card-note">Comparable families with zero hits are assigned F1=0; families with TP=FP=FN=0 are omitted. '
        "For very small families (n≤2), one interpretation can reverse F1, so do not interpret them in isolation.</p></section>"
    )
    body.append(
        '<section class="card"><div class="card-head"><h2>Detail Table</h2>'
        "<h3>Uses the same source as category_metrics.csv and follows the scope filter above</h3></div>"
        + _metrics_table(rows)
        + "</section>"
    )

    body.append(
        '<div class="section-break"><h2>Runtime and Record Level</h2>'
        '<p class="sub">The panels below cover all analyzed records and are unaffected by the scope filter above.</p>'
        "</div>"
    )

    waffle = _waffle_chart(record_rows)
    if waffle:
        effects = summary.get("label_effect_record_counts") or {}
        body.append(
            '<section class="card"><div class="card-head">'
            "<h2>Record-Level Label Effect</h2>"
            '<h3>Each cell is one record: whether ECGAgent agreement with the reference label improved, '
            "remained unchanged or worsened relative to the rule baseline</h3></div>"
            + _legend(
                [
                    (f'Improved {effects.get("improved", 0)}', "var(--better)", "rect"),
                    (f'Unchanged {effects.get("unchanged", 0)}', "var(--same)", "rect"),
                    (f'Worsened {effects.get("worsened", 0)}', "var(--worse)", "rect"),
                ]
            )
            + '<div class="chart-scroll">'
            + waffle
            + "</div>"
            + '<p class="card-note">Improved/worsened describes only distance from PTB-XL record-level labels, not clinical correctness. '
            "See record_results.csv for per-record details.</p></section>"
        )

    panels = []
    runtime_values = sorted(
        value for value in (
            _number(row.get("runtime_seconds")) for row in record_rows
            if str(row.get("outcome") or "") == "verified"
        ) if value is not None
    )
    if runtime_values:
        width = max(1.0, (runtime_values[-1] - runtime_values[0]) / 8)
        width = max(15.0, round(width / 15) * 15)
        start = math.floor(runtime_values[0] / width) * width
        buckets: dict[float, int] = {}
        while start <= runtime_values[-1]:
            buckets[start] = 0
            start += width
        for value in runtime_values:
            key = math.floor(value / width) * width
            buckets[key] = buckets.get(key, 0) + 1
        pairs = [
            (f"{int(edge)}–{int(edge + width)}", float(count))
            for edge, count in sorted(buckets.items())
        ]
        median = runtimes.get("median")
        panels.append(
            '<div class="panel-card"><h3>Verified-Record Runtime Distribution (seconds)</h3>'
            + _columns_chart(
                pairs,
                aria="Runtime distribution for verified records",
                caption="seconds / record",
            )
            + (
                f'<p class="card-note">Median {_fmt(median, 0)}s, '
                f'P90 {_fmt(runtimes.get("p90"), 0)}s.</p>'
                if median is not None else ""
            )
            + "</div>"
        )

    histogram = summary.get("verified_revision_histogram") or {}
    if histogram:
        panels.append(
            '<div class="panel-card"><h3>Revision Rounds</h3>'
            + _columns_chart(
                [(f"{key} rounds", float(value)) for key, value in
                 sorted(histogram.items(), key=lambda item: int(item[0]))],
                aria="Revision rounds required by verified records",
                caption="record count",
            )
            + "</div>"
        )

    grades = summary.get("outcome_by_record_grade") or {}
    if grades:
        panels.append(
            '<div class="panel-card"><h3>Record Quality Grade</h3>'
            + _columns_chart(
                [
                    (grade, float(sum(int(v or 0) for v in counts.values())))
                    for grade, counts in sorted(grades.items())
                ],
                aria="Record count by record-quality grade",
                caption="record count",
            )
            + "</div>"
        )

    errors = summary.get("error_category_counts") or {}
    if errors:
        panels.append(
            '<div class="panel-card"><h3>Error Categories (records)</h3>'
            + _hbars_chart(
                [(key, float(value)) for key, value in
                 sorted(errors.items(), key=lambda item: -int(item[1]))][:6],
                aria="Error categories for failed records",
                klass="worse",
            )
            + "</div>"
        )

    execution_bar, execution_legend = _execution_section(summary)
    if panels or execution_bar:
        body.append(
            '<section class="card"><div class="card-head"><h2>Runtime Health</h2>'
            '<h3>Execution outcomes, runtime and revision cost</h3></div>'
            + (execution_legend + execution_bar if execution_bar else "")
            + (f'<div class="panels" style="margin-top:18px">{"".join(panels)}</div>'
               if panels else "")
            + "</section>"
        )

    code_chart = _code_chart(code_counts)
    if code_chart:
        body.append(
            '<section class="card"><div class="card-head">'
            "<h2>Diagnosis-Code Frequency</h2>"
            '<h3>ECGAgent is the primary series and the rule baseline is the comparator; shows the 14 most frequent codes across both</h3>'
            "</div>"
            + _legend(
                [
                    ("ECGAgent", "var(--accent)", "rect"),
                    ("Rule baseline", "var(--context)", "rect"),
                ]
            )
            + '<div class="chart-scroll">'
            + code_chart
            + "</div>"
            + '<p class="card-note">This is output frequency, not accuracy. '
            "See diagnosis_code_counts.csv for the complete list.</p></section>"
        )

    body.extend([
        "<footer><p>Data sources: <code>analysis.json</code>, "
        "<code>category_metrics.csv</code>, <code>record_results.csv</code>, and "
        "<code>diagnosis_code_counts.csv</code>. This page is generated by "
        "<code>ecgagent.analysis_dashboard</code> and can be opened offline.</p>"
        "<ul>"
        "<li>PTB-XL provides record-level reference labels, not a waveform-by-waveform gold standard; missing labels and granularity differences affect agreement.</li>"
        "<li>Broad and screening families are excluded from direct label-agreement micro F1.</li>"
        "<li>All conclusions requiring human review or having a non-pass quality gate require qualified review.</li>"
        "</ul></footer>",
        "</div>",
        '<div id="tip" role="status" aria-live="polite"></div>',
    ])

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        "<title>ECGAgent Label-Agreement Analysis</title>"
        f"<style>{_css()}</style></head><body>"
        + "".join(body)
        + f"<script>{_js()}</script></body></html>\n"
    )


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def write_dashboard(
    path: Path,
    summary: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    *,
    record_rows: Sequence[Mapping[str, Any]] = (),
    code_counts: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write the visual dashboard next to the analysis CSVs."""
    from .batch import _text_dump_atomic

    html = _build_html(
        summary,
        metrics,
        record_rows=record_rows,
        code_counts=code_counts,
        source_dir=path.parent,
    )
    _text_dump_atomic(path, html)
    return path


def build_from_output_dir(output_dir: Path) -> Path:
    """Rebuild ``RESULTS.html`` from the files an analysis run left behind."""
    from .batch import _read_csv, _read_json

    output_dir = Path(output_dir)
    summary = _read_json(output_dir / "analysis.json")
    metrics = _read_csv(output_dir / "category_metrics.csv")
    records_path = output_dir / "record_results.csv"
    codes_path = output_dir / "diagnosis_code_counts.csv"
    return write_dashboard(
        output_dir / "RESULTS.html",
        summary,
        metrics,
        record_rows=_read_csv(records_path) if records_path.exists() else (),
        code_counts=_read_csv(codes_path) if codes_path.exists() else (),
    )


def main(argv: Sequence[str] | None = None) -> int:
    from .batch import DEFAULT_OUTPUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="*",
        type=Path,
        default=[DEFAULT_OUTPUT_DIR],
        help="Analysis output directory containing analysis.json and category_metrics.csv",
    )
    args = parser.parse_args(argv)
    for directory in args.output_dir or [DEFAULT_OUTPUT_DIR]:
        path = build_from_output_dir(directory)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
