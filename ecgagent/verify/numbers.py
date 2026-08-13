"""Extraction and comparison of unit-bearing quantities in model text.

The verifier only reasons about numbers that carry an explicit unit.  That is
a deliberate precision/recall trade: a bare number in ECG prose is usually a
lead name, a rule id fragment, a priority level, a beat count or a ratio, and
flagging those would bury the real findings in noise.  A number written as
`126 ms` is unambiguously a measurement claim, and those are the ones that
must trace back to evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical unit -> the spellings a model actually produces.
_UNIT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "ms": ("ms", "msec", "msecs", "millisecond", "milliseconds"),
    "s": ("s", "sec", "secs", "second", "seconds"),
    "mV": ("mv", "millivolt", "millivolts"),
    "uV": ("uv", "µv", "microvolt", "microvolts"),
    "bpm": ("bpm", "beats/min", "beats per minute", "/min"),
    "deg": ("deg", "degree", "degrees", "°"),
    "mm": ("mm",),
    "mV*ms": ("mv*ms", "mv.ms", "mv·ms", "mv ms"),
}

_SPELLING_TO_UNIT: dict[str, str] = {
    spelling: unit
    for unit, spellings in _UNIT_SPELLINGS.items()
    for spelling in spellings
}

# Longest spellings first so `milliseconds` is not shortened to `ms`, and
# `mv*ms` is not split into `mv`.
_UNIT_ALTERNATION = "|".join(
    re.escape(spelling)
    for spelling in sorted(_SPELLING_TO_UNIT, key=len, reverse=True)
)

_QUANTITY_RE = re.compile(
    r"(?P<number>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>" + _UNIT_ALTERNATION + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

# A shared-unit range must be recognized before ordinary quantities.  Without
# this pass, ``114-116 ms`` is tokenized as ``-116 ms`` because the hyphen is
# mistaken for a unary minus.  The same representation is common with an en
# dash and in Chinese prose.  Keeping both endpoints as quantities lets the
# verifier pair them independently with two atomic evidence items.
_RANGE_RE = re.compile(
    r"(?<![\d.])"
    r"(?P<first>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?P<separator>[-–—~\u81f3\u5230])\s*"
    r"(?P<second>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>" + _UNIT_ALTERNATION + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_PLUS_MINUS_RE = re.compile(
    r"(?<![\d.])"
    r"(?P<center>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?:±|\+/-)\s*"
    r"(?P<spread>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>" + _UNIT_ALTERNATION + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

# Spans that must never be mined for quantities: citation tokens carry their
# own numbers (list indices), and identifiers such as CLIN-INTERVAL-PR-01 or
# a `P2` priority would otherwise read as measurements.
_MASK_PATTERNS = (
    re.compile(r"ev:/\S+"),
    re.compile(r"/[A-Za-z_][\w/\-.]*"),
    re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b"),
    # "V3 S wave" is a lead plus a wave label, not a three-second quantity.
    re.compile(r"\b(?:aVR|aVL|aVF|I{1,3}|V(?:[1-9]|1[0-8]))\b", re.IGNORECASE),
)

# Tolerances for calling two quantities the same measurement. Wider than the
# instrument's precision on purpose: the check is for invented numbers, not
# for rounding.
_TOLERANCE: dict[str, float] = {
    "ms": 1.0,
    "s": 0.05,
    "mV": 0.01,
    "uV": 10.0,
    "bpm": 1.0,
    "deg": 1.0,
    "mm": 0.5,
    "mV*ms": 1.0,
}

# Pairs convertible for comparison, with the factor from -> to.
_CONVERSIONS: dict[tuple[str, str], float] = {
    ("s", "ms"): 1000.0,
    ("ms", "s"): 0.001,
    ("mV", "uV"): 1000.0,
    ("uV", "mV"): 0.001,
}

# A guideline cutoff is not a measurement of this patient.  This helper lives
# beside quantity extraction so the free-text verifier and the structured
# diagnostic contract apply exactly the same rule.  Direction matters: a
# comparator must introduce the number, or a threshold noun must label it
# afterwards.  A symmetric window would incorrectly exempt a patient value in
# prose such as "QT is 446 ms, greater than ...".
_THRESHOLD_INTRODUCERS: tuple[str, ...] = (
    "above", "below", "over", "under", "exceeds", "exceed", "exceeding",
    "greater than", "less than", "more than", "at least", "at most",
    "limit of", "threshold of", "cutoff of", "cut-off of", "upper limit",
    "lower limit", "normal range", "reference range", "criterion of",
    ">=", "<=", ">", "<",
    "\u8d85\u8fc7", "\u4f4e\u4e8e", "\u5927\u4e8e", "\u5c0f\u4e8e", "\u9608\u503c\u4e3a", "\u4e0a\u9650\u4e3a", "\u4e0b\u9650\u4e3a", "\u6b63\u5e38\u8303\u56f4", "\u53c2\u8003\u8303\u56f4",
    "\u3092\u8d85\u3048", "\u4ee5\u4e0a", "\u4ee5\u4e0b", "\u95be\u5024",
)
_THRESHOLD_LABELS: tuple[str, ...] = (
    "limit", "threshold", "cutoff", "cut-off", "criterion", "criteria",
    "upper bound", "lower bound", "reference",
    "\u9608\u503c", "\u4e0a\u9650", "\u4e0b\u9650", "\u6807\u51c6\u503c", "\u53c2\u8003\u8303\u56f4", "\u95be\u5024", "\u57fa\u6e96",
)
_THRESHOLD_INTRODUCER_WINDOW = 28
_THRESHOLD_LABEL_WINDOW = 28


@dataclass(frozen=True)
class Quantity:
    """A number with a unit, as written in the text."""

    value: float
    unit: str
    text: str
    start: int
    end: int

    def render(self) -> str:
        return f"{self.text.strip()}"


def is_threshold_reference(claim: str, quantity: Quantity) -> bool:
    """Return whether a quantity is presented as a general cutoff/reference."""

    before = claim[
        max(0, quantity.start - _THRESHOLD_INTRODUCER_WINDOW) : quantity.start
    ].lower()
    after = claim[
        quantity.end : quantity.end + _THRESHOLD_LABEL_WINDOW
    ].lower()
    if any(marker in before for marker in _THRESHOLD_INTRODUCERS):
        return True
    return any(label in after for label in _THRESHOLD_LABELS)


def normalize_unit(spelling: str | None) -> str | None:
    if spelling is None:
        return None
    return _SPELLING_TO_UNIT.get(str(spelling).strip().lower(), None)


def _masked(text: str) -> str:
    """Blank out spans that look numeric but are not measurements."""
    masked = text
    for pattern in _MASK_PATTERNS:
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return masked


def extract_quantities(text: str) -> list[Quantity]:
    """Find every unit-bearing number in `text`, ignoring identifiers and pointers."""
    masked = _masked(text)
    found: list[Quantity] = []

    occupied: list[tuple[int, int]] = []

    def append_pair(
        match: re.Match[str],
        first_group: str,
        second_group: str,
    ) -> None:
        unit = normalize_unit(match.group("unit"))
        if unit is None:
            return
        first_text = match.group(first_group)
        second_text = match.group(second_group)
        found.extend(
            (
                Quantity(
                    value=float(first_text),
                    unit=unit,
                    text=f"{first_text} {match.group('unit')}",
                    start=match.start(first_group),
                    end=match.end(first_group),
                ),
                Quantity(
                    value=float(second_text),
                    unit=unit,
                    text=text[match.start(second_group) : match.end()],
                    start=match.start(second_group),
                    end=match.end(),
                ),
            )
        )
        occupied.append(match.span())

    for match in _RANGE_RE.finditer(masked):
        append_pair(match, "first", "second")
    for match in _PLUS_MINUS_RE.finditer(masked):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        append_pair(match, "center", "spread")

    for match in _QUANTITY_RE.finditer(masked):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        unit = normalize_unit(match.group("unit"))
        if unit is None:
            continue
        try:
            value = float(match.group("number"))
        except ValueError:  # pragma: no cover - regex guarantees a float
            continue
        found.append(
            Quantity(
                value=value,
                unit=unit,
                text=text[match.start() : match.end()],
                start=match.start(),
                end=match.end(),
            )
        )
    return sorted(found, key=lambda item: (item.start, item.end))


def convert(value: float, from_unit: str, to_unit: str) -> float | None:
    if from_unit == to_unit:
        return value
    factor = _CONVERSIONS.get((from_unit, to_unit))
    return None if factor is None else value * factor


def tolerance_for(unit: str) -> float:
    return _TOLERANCE.get(unit, 0.01)


def quantities_match(written: Quantity, value: float, unit: str | None) -> bool:
    """True when a written quantity is the same measurement as an evidence value."""
    if unit is None:
        # No unit recorded for the evidence: compare raw magnitudes only.
        return abs(written.value - value) <= tolerance_for(written.unit)
    converted = convert(value, unit, written.unit)
    if converted is None:
        return False
    return abs(written.value - converted) <= tolerance_for(written.unit)


def comparable(written_unit: str, evidence_unit: str | None) -> bool:
    """Whether the two units describe the same physical dimension."""
    if evidence_unit is None:
        return False
    return written_unit == evidence_unit or (evidence_unit, written_unit) in _CONVERSIONS
