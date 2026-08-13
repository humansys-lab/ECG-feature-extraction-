"""Shared rendering helpers.

Tool output shape matters as much as tool content: a 12xN markdown table is
far easier for a model to reason over than the equivalent nested JSON, and it
costs fewer tokens.  All tabular tools funnel through here so the formatting
stays uniform.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..evidence.pointer import display_precision

NULL_CELL = "-"


def format_cell(value: Any, unit: str | None) -> str:
    if value is None:
        return NULL_CELL
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return NULL_CELL
        text = f"{value:.{display_precision(unit)}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, (list, tuple)):
        if not value:
            return NULL_CELL
        return ",".join(str(item) for item in value[:4]) + ("" if len(value) <= 4 else ",...")
    if isinstance(value, dict):
        return NULL_CELL if not value else f"<{len(value)} keys>"
    text = str(value)
    return text if len(text) <= 28 else text[:25] + "..."


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a fixed-width markdown table. Empty rows yield an empty string."""
    if not rows:
        return ""
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(str(cell)))
    def line(cells: Sequence[Any]) -> str:
        padded = [str(cell).ljust(widths[index]) for index, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def truncate_lines(text: str, max_lines: int) -> tuple[str, bool]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, False
    return "\n".join(lines[:max_lines]), True


def bullet_list(items: Sequence[str], prefix: str = "  - ") -> str:
    return "\n".join(f"{prefix}{item}" for item in items)
