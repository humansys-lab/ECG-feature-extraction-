"""Reliability caveats attached to every resolved measurement.

ecgfeat already publishes the reliability of each measurement, but it does so
in *companion fields* sitting next to the value (`qt_ms` next to
`qt_reliability`, `qt_reportable`, `qt_unreliable_reasons`, ...).  A model that
reads only the value silently loses that context, which is how a measurement
the extractor explicitly marked unreportable ends up quoted as fact.

This module makes that structurally impossible: resolving a value also
resolves its companions, so an unreliable number can never be handed over as a
bare number.  Nothing here decides anything clinical; it only relays what
ecgfeat already said.
"""
from __future__ import annotations

from typing import Any, Callable

# Companion fields consulted for a global measurement, keyed by the field the
# model asked for. Each entry is (companion_field, formatter).
_GLOBAL_COMPANIONS: dict[str, tuple[str, ...]] = {
    "qt_ms": (
        "qt_reliability",
        "qt_reportable",
        "qt_unreliable_reasons",
        "qt_path",
        "qt_source",
        "qt_excluded_leads",
        "qt_rejected",
        "qt_reject_reason",
        "t_fusion_reliable",
    ),
    "p_duration_ms": (
        "p_duration_reliability",
        "p_duration_source",
        "p_duration_support",
        "p_duration_spread_ms",
    ),
    "t_axis_deg": ("t_axis_reliable",),
    "qrs_ms": ("qrs_wide_ms",),
    "atrial_rate_bpm": ("p_duration_reliability", "p_duration_source"),
    "qt_dispersion_ms": ("qt_reliability", "qt_reportable"),
    "t_global_tpte_ms": ("t_fusion_reliable", "t_fusion_reliability_reasons"),
}

# QTc variants inherit the whole QT reliability story.
for _qtc in (
    "qtc_bazett_ms",
    "qtc_fridericia_ms",
    "qtc_framingham_ms",
    "qtc_hodges_ms",
):
    _GLOBAL_COMPANIONS[_qtc] = _GLOBAL_COMPANIONS["qt_ms"]

# Per-lead companion fields, keyed by the modality the requested field belongs to.
_LEAD_MODALITY_COMPANIONS: dict[str, tuple[str, ...]] = {
    "p": ("reliable_for_p", "p_measurement_suppressed", "p_confidence_mean"),
    "qrs": ("reliable_for_qrs", "qrs_confidence_mean", "qrs_terminal_confidence"),
    "t": ("reliable_for_t", "t_sqi_pass", "t_confidence_reason"),
    "qt": (
        "reliable_for_qt",
        "qt_confidence_mean",
        "t_offset_fusion_reliable",
        "t_offset_fusion_reliability_reason",
    ),
    "st": (
        "reliable_for_t",
        "st_j_reliable",
        "st_j_unreliable_reason",
        "st_hybrid_reliable",
        "st_hybrid_unreliable_reason",
        "st_j_source",
    ),
}

_MODALITY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("st_", "st"),
    ("twelve_sl_st", "st"),
    ("qt_", "qt"),
    ("jt_", "qt"),
    ("tpe_", "qt"),
    ("t_offset", "qt"),
    ("t_", "t"),
    ("p_", "p"),
    ("ptf_", "p"),
    ("pr_", "p"),
    ("q_", "qrs"),
    ("r_", "qrs"),
    ("s_", "qrs"),
    ("qrs_", "qrs"),
    ("vat_", "qrs"),
    ("initial_qrs", "qrs"),
    ("fqrs_", "qrs"),
    ("u_", "t"),
)

# A measurement whose lead is flagged unreliable for its own modality is the
# single most common way a wrong number reaches a conclusion.
_QUALITY_FLAG_BY_MODALITY = {
    "p": "reliable_for_p",
    "qrs": "reliable_for_qrs",
    "t": "reliable_for_t",
    "qt": "reliable_for_qt",
    "st": "reliable_for_t",
}


def classify_modality(field_name: str) -> str | None:
    """Which wave a per-lead field belongs to, for reliability lookup."""
    name = str(field_name).lower()
    for prefix, modality in _MODALITY_PREFIXES:
        if name.startswith(prefix):
            return modality
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "[" + ", ".join(_fmt(item) for item in value[:6]) + ("]" if len(value) <= 6 else ", ...]")
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = list(value.items())[:4]
        body = ", ".join(f"{k}={_fmt(v)}" for k, v in items)
        return "{" + body + ("}" if len(value) <= 4 else ", ...}")
    return str(value)


def _is_negative_signal(field: str, value: Any) -> bool:
    """True when a companion field indicates the value should not be trusted."""
    if value is None:
        return False
    if field.endswith("_reportable") or field.startswith("reliable_for"):
        return value is False
    if field.endswith("_reliable") or field == "t_axis_reliable":
        return value is False
    if field.endswith("_rejected") or field == "p_measurement_suppressed":
        return value is True
    if field.endswith("_reliability"):
        return str(value).lower() in {"unreliable", "unavailable", "low"}
    if field.endswith("_reasons") or field.endswith("_reason"):
        return bool(value)
    if field == "qt_excluded_leads":
        return bool(value)
    if field == "_sqi_pass" or field.endswith("_sqi_pass"):
        return value is False
    return False


def global_caveats(
    field: str,
    value: Any,
    global_features: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Caveats and companion pointers for `/global_features/<field>`."""
    notes: list[str] = []
    companions: list[str] = []
    for companion in _GLOBAL_COMPANIONS.get(field, ()):
        if companion not in global_features:
            continue
        companion_value = global_features[companion]
        companions.append(f"/global_features/{companion}")
        if _is_negative_signal(companion, companion_value):
            notes.append(f"{companion}={_fmt(companion_value)}")

    if field == "qrs_ms":
        wide = global_features.get("qrs_wide_ms")
        if isinstance(value, (int, float)) and isinstance(wide, (int, float)):
            delta = abs(float(wide) - float(value))
            if delta >= 8.0:
                notes.append(
                    f"consensus qrs_ms={_fmt(value)} vs qrs_wide_ms={_fmt(wide)} "
                    f"differ by {delta:.0f} ms; consensus can under-read near the "
                    "110-120 ms band (see FM-QRS-CONSENSUS-NARROW)"
                )

    if field == "atrial_rate_bpm":
        source = global_features.get("p_duration_source")
        if isinstance(source, str) and "suppress" in source:
            notes.append(
                "atrial rate falls back to the ventricular rate when P detection "
                "degrades; p_duration_source indicates suppressed atrial measurements"
            )
    return notes, companions


def lead_caveats(
    lead: str,
    field: str,
    params: dict[str, Any],
    lead_quality: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Caveats and companion pointers for a `/representative_leads/<lead>/params/<field>` value."""
    notes: list[str] = []
    companions: list[str] = []
    modality = classify_modality(field)
    if modality is None:
        return notes, companions

    for companion in _LEAD_MODALITY_COMPANIONS.get(modality, ()):
        if companion not in params:
            continue
        companion_value = params[companion]
        companions.append(f"/representative_leads/{lead}/params/{companion}")
        if _is_negative_signal(companion, companion_value):
            notes.append(f"{companion}={_fmt(companion_value)}")

    quality_flag = _QUALITY_FLAG_BY_MODALITY.get(modality)
    if lead_quality and quality_flag and lead_quality.get(quality_flag) is False:
        flags = lead_quality.get("flags") or []
        notes.append(
            f"lead {lead} quality {quality_flag}=false"
            + (f" (flags: {', '.join(str(f) for f in flags)})" if flags else "")
        )
    if lead_quality and lead_quality.get("missing"):
        notes.append(f"lead {lead} is missing from the record")
    return notes, companions


def record_caveats(document: dict[str, Any]) -> list[str]:
    """Record-level caveats that apply to any interpretation of this payload."""
    notes: list[str] = []
    metadata = document.get("metadata") or {}
    gate = metadata.get("diagnostic_gate") or {}
    state = gate.get("state")
    if state and state != "pass":
        reasons = gate.get("stop_reasons") or gate.get("partial_reasons") or []
        notes.append(
            f"diagnostic gate state={state}"
            + (f" ({', '.join(str(r) for r in reasons)})" if reasons else "")
        )
    record_quality = metadata.get("record_quality") or {}
    grade = record_quality.get("record_grade")
    if grade and grade not in {"Q0", "Q1"}:
        notes.append(f"record grade={grade}")
    rejected = record_quality.get("rejected_functions") or []
    if rejected:
        notes.append(f"rejected measurement functions: {', '.join(str(r) for r in rejected)}")
    return notes


CaveatFn = Callable[..., tuple[list[str], list[str]]]
