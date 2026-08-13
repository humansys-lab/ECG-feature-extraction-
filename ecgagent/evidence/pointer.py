"""RFC 6901 JSON Pointer addressing, unit inference and clinical aliases.

Every value an LLM ever sees is addressed by a pointer into the features
payload.  The pointer string doubles as the citation token, so pointer syntax
is part of the output contract and must stay stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

POINTER_PREFIX = "ev:"


class PointerError(LookupError):
    """Raised when a pointer cannot be parsed or resolved."""


@dataclass(frozen=True)
class Pointer:
    """A parsed JSON Pointer. `raw` is the canonical citation form."""

    raw: str
    tokens: tuple[str, ...]

    @property
    def citation(self) -> str:
        return f"{POINTER_PREFIX}{self.raw}"

    @property
    def leaf(self) -> str:
        return self.tokens[-1] if self.tokens else ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw


def parse_pointer(text: str) -> Pointer:
    """Parse `/a/b/c`, `ev:/a/b/c` or `a.b.c` into a canonical Pointer."""
    if not isinstance(text, str) or not text.strip():
        raise PointerError("pointer must be a non-empty string")
    candidate = text.strip()
    if candidate.startswith(POINTER_PREFIX):
        candidate = candidate[len(POINTER_PREFIX) :]
    if not candidate.startswith("/"):
        # Accept dotted paths as a convenience; models produce them constantly.
        candidate = "/" + candidate.replace(".", "/")
    if candidate == "/":
        raise PointerError("pointer must address a value, not the document root")
    tokens = tuple(_unescape(tok) for tok in candidate.split("/")[1:])
    if any(tok == "" for tok in tokens):
        raise PointerError(f"pointer has an empty path segment: {text!r}")
    return Pointer(raw="/" + "/".join(_escape(tok) for tok in tokens), tokens=tokens)


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def resolve_in(document: Any, pointer: Pointer) -> Any:
    """Walk `document` along `pointer`. Raises PointerError if the path breaks."""
    node: Any = document
    for depth, token in enumerate(pointer.tokens):
        if isinstance(node, dict):
            if token in node:
                node = node[token]
                continue
            raise PointerError(
                f"no key {token!r} at /{'/'.join(pointer.tokens[:depth])}"
            )
        if isinstance(node, (list, tuple)):
            try:
                index = int(token)
            except ValueError as exc:
                raise PointerError(
                    f"list index expected at /{'/'.join(pointer.tokens[:depth + 1])}, got {token!r}"
                ) from exc
            if not -len(node) <= index < len(node):
                raise PointerError(
                    f"index {index} out of range at /{'/'.join(pointer.tokens[:depth])}"
                )
            node = node[index]
            continue
        raise PointerError(
            f"cannot descend into {type(node).__name__} at /{'/'.join(pointer.tokens[:depth])}"
        )
    return node


def walk_leaves(document: Any, prefix: str = "", max_depth: int = 6) -> Iterator[tuple[str, Any]]:
    """Yield (pointer, value) for every scalar leaf, bounded by `max_depth`."""
    if max_depth < 0:
        return
    if isinstance(document, dict):
        for key, value in document.items():
            child = f"{prefix}/{_escape(str(key))}"
            if isinstance(value, (dict, list)):
                yield from walk_leaves(value, child, max_depth - 1)
            else:
                yield child, value
    elif isinstance(document, list):
        for index, value in enumerate(document):
            child = f"{prefix}/{index}"
            if isinstance(value, (dict, list)):
                yield from walk_leaves(value, child, max_depth - 1)
            else:
                yield child, value


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
# Ordered longest-suffix-first: `_mv_ms` must win over `_ms` and `_mv`.
_UNIT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_mv_per_ms", "mV/ms"),
    ("_mv_per_ms2", "mV/ms^2"),
    ("_uv_ms", "uV*ms"),
    ("_mv_ms", "mV*ms"),
    ("_ms_sd", "ms"),
    ("_mv_sd", "mV"),
    ("_bpm", "bpm"),
    ("_deg", "deg"),
    ("_hz", "Hz"),
    ("_sec", "s"),
    ("_ms", "ms"),
    ("_mv", "mV"),
    ("_uv", "uV"),
)

_DIMENSIONLESS_SUFFIXES = (
    "_score",
    "_ratio",
    "_fraction",
    "_count",
    "_support",
    "_flag",
    "_index",
    "_corr",
    "_weight",
    "_sqi",
)


def infer_unit(field_name: str) -> str | None:
    """Infer a display unit from an ecgfeat field name.

    ecgfeat encodes units in field-name suffixes consistently, so this is
    reliable; anything unrecognised returns None rather than guessing.
    """
    name = str(field_name).lower()
    for suffix, unit in _UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    if any(name.endswith(suffix) for suffix in _DIMENSIONLESS_SUFFIXES):
        return None
    if name in {"rr_cv", "pnn50", "rr_entropy"}:
        return None
    return None


# Numeric tolerance used by the citation verifier, keyed by unit.
UNIT_TOLERANCE: dict[str, float] = {
    "ms": 1.0,
    "mV": 0.01,
    "uV": 10.0,
    "deg": 1.0,
    "bpm": 1.0,
    "s": 0.05,
}


def display_precision(unit: str | None) -> int:
    if unit in {"ms", "deg", "bpm", "uV"}:
        return 0
    if unit in {"mV"}:
        return 3
    if unit in {"mV*ms", "uV*ms"}:
        return 1
    if unit in {"mV/ms", "mV/ms^2"}:
        return 4
    return 3


# ---------------------------------------------------------------------------
# Clinical aliases
# ---------------------------------------------------------------------------
# Lets a model ask for "QTc(Bazett)" instead of memorising the payload layout.
# Output still carries the canonical pointer, so citations stay uniform.
_GLOBAL_ALIASES: dict[str, str] = {
    "hr": "/global_features/heart_rate_bpm",
    "heart rate": "/global_features/heart_rate_bpm",
    "ventricular rate": "/global_features/heart_rate_bpm",
    "atrial rate": "/global_features/atrial_rate_bpm",
    "pr": "/global_features/pr_ms",
    "pr interval": "/global_features/pr_ms",
    "qrs": "/global_features/qrs_ms",
    "qrs duration": "/global_features/qrs_ms",
    "qrs wide": "/global_features/qrs_wide_ms",
    "qt": "/global_features/qt_ms",
    "qtc": "/global_features/qtc_bazett_ms",
    "qtc bazett": "/global_features/qtc_bazett_ms",
    "qtc fridericia": "/global_features/qtc_fridericia_ms",
    "qtc framingham": "/global_features/qtc_framingham_ms",
    "qtc hodges": "/global_features/qtc_hodges_ms",
    "p axis": "/global_features/p_axis_deg",
    "qrs axis": "/global_features/qrs_axis_deg",
    "t axis": "/global_features/t_axis_deg",
    "st axis": "/global_features/st_axis_deg",
    "p duration": "/global_features/p_duration_ms",
    "qt dispersion": "/global_features/qt_dispersion_ms",
    "qrs-t angle": "/global_features/qrs_t_angle_deg",
    "transition zone": "/global_features/transition_zone",
    "ptf v1": "/global_features/ptf_v1_mv_ms",
    "rr mean": "/global_features/rr_mean_ms",
    "rr sd": "/global_features/rr_sd_ms",
    "rr cv": "/global_features/rr_cv",
    "rmssd": "/global_features/rmssd_ms",
    "pnn50": "/global_features/pnn50",
    "sd1": "/global_features/poincare_sd1_ms",
    "sd2": "/global_features/poincare_sd2_ms",
    "paced rhythm": "/global_features/paced_rhythm",
    "gate": "/metadata/diagnostic_gate/state",
    "record grade": "/metadata/record_quality/record_grade",
    "n beats": "/metadata/n_beats",
    "duration": "/metadata/duration_sec",
    "sampling rate": "/metadata/input_fs",
}

# Exact pointer-shaped aliases observed from otherwise valid tool-grounded
# verdicts. Each is a one-to-one naming variant of a canonical global field.
# Deliberately excluded: guessed rule paths, tool names, and aliases whose unit
# or clinical subject is ambiguous.
_POINTER_ALIASES: dict[str, str] = {
    "/global_features/ventricular_rate_bpm": "/global_features/heart_rate_bpm",
    "/global_features/ventricular_rate": "/global_features/heart_rate_bpm",
    "/global_features/hr_bpm": "/global_features/heart_rate_bpm",
    "/global_features/hr": "/global_features/heart_rate_bpm",
    "/hr_global_bpm": "/global_features/heart_rate_bpm",
    "/rr_mean_ms": "/global_features/rr_mean_ms",
    "/rr_cv": "/global_features/rr_cv",
    "/rmssd": "/global_features/rmssd_ms",
    "/pnn50": "/global_features/pnn50",
}


def resolve_alias(name: str) -> str | None:
    """Map a clinical name to a canonical pointer, or None if not an alias."""
    if not isinstance(name, str):
        return None
    raw = name.strip()
    lowered = raw.lower()
    if lowered in _POINTER_ALIASES:
        return _POINTER_ALIASES[lowered]
    if lowered.startswith("global."):
        suffix = raw.split(".", 1)[1].strip()
        if suffix:
            return "/global_features/" + suffix.replace(".", "/")
    if lowered.startswith("lead."):
        parts = raw.split(".")
        if len(parts) >= 3 and parts[1].strip() and all(part.strip() for part in parts[2:]):
            return (
                f"/representative_leads/{parts[1].strip()}/params/"
                + "/".join(part.strip() for part in parts[2:])
            )
    key = name.strip().lower().replace("_", " ")
    key = key.replace("(", " ").replace(")", " ")
    key = " ".join(key.split())
    return _GLOBAL_ALIASES.get(key)


def alias_table() -> dict[str, str]:
    """Copy of the alias map, for prompt rendering and tests."""
    return dict(_GLOBAL_ALIASES)
