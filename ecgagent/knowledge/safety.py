"""Patient-planning safety filters for general knowledge excerpts."""
from __future__ import annotations

import re
from dataclasses import dataclass


_RECORD_ID = re.compile(
    r"\b(?:\d{4,}_(?:hr|lr)|(?:JS|PTBXL|ECG)\d{4,})\b",
    re.IGNORECASE,
)
_CASE_LINE = re.compile(
    r"(?:\brecord\b|\bpatient\b|\bsample\b|\bcase\b)"
    r"\s*(?:(?:id|no\.?|number)\s*)?[:=#-]?\s*[A-Z]*\d+"
    r"|(?:\u75c5\u4f8b|\u60a3\u8005|\u6837\u672c)\s*(?:(?:\u7f16\u53f7|ID)\s*)?[:\uff1a=#-]?\s*\d+",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?:record_id|patient_id|ecg_id|subject_id)\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)
_HAN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class SanitizedExcerpt:
    text: str
    removed_lines: int
    redacted_identifiers: int


def sanitize_runtime_label(text: str) -> str:
    """Redact identifiers from titles that may enter a model prompt."""

    value = _RECORD_ID.sub("[record-id-redacted]", str(text or ""))
    if _HAN.search(value):
        return "General ECG knowledge reference"
    return _ASSIGNMENT.sub("[record-id-redacted]", value)


def sanitize_runtime_excerpt(text: str) -> SanitizedExcerpt:
    """Remove case-specific examples while preserving general thresholds."""

    kept: list[str] = []
    removed = 0
    redacted = 0
    for line in str(text or "").splitlines():
        if _CASE_LINE.search(line) or _RECORD_ID.search(line) or _HAN.search(line):
            removed += 1
            continue
        line, n_ids = _RECORD_ID.subn("[record-id-redacted]", line)
        line, n_assignments = _ASSIGNMENT.subn("[record-id-redacted]", line)
        redacted += n_ids + n_assignments
        kept.append(line)
    return SanitizedExcerpt(
        text="\n".join(kept).strip(),
        removed_lines=removed,
        redacted_identifiers=redacted,
    )


__all__ = [
    "SanitizedExcerpt",
    "sanitize_runtime_excerpt",
    "sanitize_runtime_label",
]
