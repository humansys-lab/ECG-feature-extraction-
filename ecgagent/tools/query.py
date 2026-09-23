"""Measurement retrieval tools: global, per-lead, and per-beat values.

These answer "what was measured", never "what does it mean".  Interpretation
belongs to the diagnostic model.  ecgfeat is the measuring instrument.
"""
from __future__ import annotations

from typing import Any

from ..evidence.caveats import classify_modality
from ..evidence.pointer import PointerError, alias_table, infer_unit, resolve_alias
from ..evidence.store import EvidenceStore
from ._render import bullet_list, format_cell, markdown_table, truncate_lines
from .registry import ToolResult, ToolSpec

MAX_TABLE_FIELDS = 12
MAX_GLOBAL_FIELDS = 24
MAX_TABLE_LINES = 60

_QUALITY_FLAG_BY_MODALITY = {
    "p": "reliable_for_p",
    "qrs": "reliable_for_qrs",
    "t": "reliable_for_t",
    "qt": "reliable_for_qt",
    "st": "reliable_for_t",
}


def _citable_cell(
    store: EvidenceStore,
    pointer: str,
    value: Any,
    unit: str | None,
) -> tuple[str, str | None]:
    """Render a table cell with the exact citation token the model may reuse.

    `ToolResult.citations` is machine-readable provenance, but providers only
    show the model `ToolResult.text`.  Hiding the pointers from the rendered
    table caused live models to invent pseudo paths such as
    `ev:/get_beat_table/II/pr_ms`.  Put the canonical token next to the value
    and only whitelist it when the pointer really resolves.
    """
    cell = format_cell(value, unit)
    evidence = store.try_resolve(pointer)
    if evidence is None:
        return cell, None
    return f"{cell} [{evidence.citation}]", evidence.pointer


# ---------------------------------------------------------------------------
# get_global_table
# ---------------------------------------------------------------------------
def get_global_table(store: EvidenceStore, fields: list[str]) -> ToolResult:
    """Read a focused bundle of global measurements in one citable call."""
    if isinstance(fields, str):
        fields = [fields]
    if not fields:
        return ToolResult.error("fields must be a non-empty list of global field names or aliases")
    if len(fields) > MAX_GLOBAL_FIELDS:
        return ToolResult.error(
            f"at most {MAX_GLOBAL_FIELDS} fields per call; got {len(fields)}. "
            "Split the request by diagnostic domain."
        )

    rows: list[list[str]] = []
    citations: list[str] = []
    unknown: list[str] = []
    known_global = {pointer.rsplit("/", 1)[-1] for pointer in alias_table().values()
                    if pointer.startswith("/global_features/")} | {"qt_reportable", "qt_reliability", "qt_rejected", "qt_reject_reason"}
    for requested in fields:
        evidence = store.try_resolve(str(requested))
        if evidence is None or not evidence.pointer.startswith("/global_features/"):
            canonical = resolve_alias(str(requested)) or str(requested)
            field = canonical.removeprefix("/global_features/")
            if evidence is None and field in known_global:
                rows.append([str(requested), "", "unavailable", infer_unit(field) or "", "unavailable", "not exported"])
                continue
            unknown.append(str(requested))
            continue
        citations.extend((evidence.pointer, *evidence.companions))
        caveat = "; ".join(evidence.caveats) if evidence.caveats else ""
        rows.append(
            [
                str(requested),
                evidence.citation,
                evidence.format_value(),
                evidence.unit or "",
                "caution" if caveat else "usable",
                caveat,
            ]
        )

    if unknown:
        suggestions: list[str] = []
        for name in unknown[:4]:
            hits = [
                hit.field
                for hit in store.search(name, limit=5)
                if hit.pointer.startswith("/global_features/")
            ]
            if hits:
                suggestions.append(f"{name} -> {', '.join(hits)}")
        hint = ("\nClosest global fields:\n" + bullet_list(suggestions)) if suggestions else ""
        return ToolResult.error(
            f"unknown global field(s) or non-global pointer(s): {', '.join(unknown)}."
            f"{hint}\nUse search_measurements to discover exact field names."
        )

    text = markdown_table(
        ["requested", "pointer", "value", "unit", "reliability", "caveat"],
        rows,
    )
    return ToolResult(
        ok=True,
        text=text,
        citations=tuple(dict.fromkeys(citations)),
        note=(
            "A caution row remains informative as lower-weight evidence: state the "
            "limitation and triangulate it. Only null/not-produced values are unavailable."
            if any(row[4] == "caution" for row in rows)
            else None
        ),
    )


# ---------------------------------------------------------------------------
# get_measurement
# ---------------------------------------------------------------------------
def get_measurement(store: EvidenceStore, pointer: str) -> ToolResult:
    try:
        evidence = store.resolve(pointer)
    except PointerError as exc:
        hits = store.search(str(pointer).rsplit("/", 1)[-1], limit=6)
        hint = ""
        if hits:
            hint = "\nDid you mean:\n" + bullet_list([hit.pointer for hit in hits])
        return ToolResult.error(f"{exc}{hint}")

    lines = [
        f"{evidence.pointer} = {evidence.format_value()}"
        + (
            f" {evidence.unit}"
            if evidence.unit and not isinstance(evidence.value, (bool, str))
            else ""
        )
        + f" [{evidence.citation}]"
    ]
    if evidence.source:
        lines.append(f"source: {evidence.source}")
    if evidence.caveats:
        lines.append("caveats:")
        lines.append(bullet_list(list(evidence.caveats)))
    if evidence.companions:
        lines.append(
            "reliability companions (read separately before citing): "
            + ", ".join(evidence.companions)
        )

    citations = (evidence.pointer, *evidence.companions)
    return ToolResult(ok=True, text="\n".join(lines), citations=citations, payload=evidence.to_dict())


# ---------------------------------------------------------------------------
# get_lead_table
# ---------------------------------------------------------------------------
def get_lead_table(
    store: EvidenceStore,
    fields: list[str],
    leads: list[str] | None = None,
) -> ToolResult:
    if isinstance(fields, str):
        fields = [fields]
    if not fields:
        return ToolResult.error("fields must be a non-empty list of parameter names")
    if len(fields) > MAX_TABLE_FIELDS:
        return ToolResult.error(
            f"at most {MAX_TABLE_FIELDS} fields per call; got {len(fields)}. "
            "Split into focused calls."
        )

    from ..evidence.diagnostic_contract import QRS_MEASUREMENT_BUNDLE_FIELDS
    known = set(store.lead_param_fields()) | set(QRS_MEASUREMENT_BUNDLE_FIELDS) | set(_QUALITY_FLAG_BY_MODALITY.values())
    unknown = [name for name in fields if name not in known]
    if unknown:
        suggestions: list[str] = []
        for name in unknown[:3]:
            hits = [hit.field for hit in store.search(name, limit=4) if hit.field in known]
            if hits:
                suggestions.append(f"{name} -> {', '.join(hits)}")
        hint = ("\nClosest known fields:\n" + bullet_list(suggestions)) if suggestions else ""
        return ToolResult.error(
            f"unknown per-lead field(s): {', '.join(unknown)}."
            f"{hint}\nUse search_measurements to discover field names."
        )

    target_leads = [lead for lead in (leads or store.leads) if lead in store.leads]
    if not target_leads:
        return ToolResult.unavailable(
            f"no matching leads. Available: {', '.join(store.leads)}"
        )

    units = {name: infer_unit(name) for name in fields}
    headers = ["lead"] + [
        f"{name} ({units[name]})" if units[name] else name for name in fields
    ]

    citations: list[str] = []
    rows: list[list[str]] = []
    flagged: dict[str, set[str]] = {}
    for lead in target_leads:
        quality = store.lead_quality(lead)
        row = [lead]
        for name in fields:
            value = store.lead_param(lead, name)
            pointer = f"/representative_leads/{lead}/params/{name}"
            cell, citation = _citable_cell(store, pointer, value, units[name])
            if citation is not None:
                citations.append(citation)
            modality = classify_modality(name)
            flag = _QUALITY_FLAG_BY_MODALITY.get(modality or "")
            if flag and quality.get(flag) is False and value is not None:
                cell += "*"
                flagged.setdefault(lead, set()).add(modality or "")
            row.append(cell)
        rows.append(row)

    body = markdown_table(headers, rows)
    notes: list[str] = []
    if flagged:
        notes.append(
            "* marks a lower-confidence value, not a discarded value. State the "
            "limitation and triangulate across other leads/beats. Flagged waves: "
            + "; ".join(
                f"{lead} ({', '.join(sorted(mods))})" for lead, mods in sorted(flagged.items())
            )
        )
    missing_leads = [lead for lead in (leads or []) if lead not in store.leads]
    if missing_leads:
        notes.append(f"not in record, skipped: {', '.join(missing_leads)}")

    text = body + (("\n\n" + "\n".join(notes)) if notes else "")
    text, truncated = truncate_lines(text, MAX_TABLE_LINES)
    return ToolResult(
        ok=True,
        text=text,
        citations=tuple(citations),
        truncated=truncated,
        note=None if not flagged else "unreliable cells are marked with *",
    )


# ---------------------------------------------------------------------------
# get_beat_table
# ---------------------------------------------------------------------------
_BEAT_LEVEL_FIELDS = ("beat_id", "r_index", "paced", "group_id", "rr_prev_ms", "rr_next_ms")


def _beat_feature_index(store: EvidenceStore, lead: str) -> dict[int, tuple[int, dict[str, Any]]]:
    rows = store.document.get("beat_features") or []
    mapping: dict[int, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("lead") == lead:
            beat_id = row.get("beat_id")
            if isinstance(beat_id, int):
                mapping[beat_id] = (index, row)
    return mapping


def get_beat_table(
    store: EvidenceStore,
    fields: list[str],
    lead: str | None = None,
    beat_ids: list[int] | None = None,
) -> ToolResult:
    if isinstance(fields, str):
        fields = [fields]
    if not fields:
        return ToolResult.error("fields must be a non-empty list")
    if len(fields) > MAX_TABLE_FIELDS:
        return ToolResult.error(f"at most {MAX_TABLE_FIELDS} fields per call; got {len(fields)}")

    beats = store.document.get("beats") or []
    if not beats:
        return ToolResult.unavailable("this record has no per-beat data")

    per_lead: dict[int, tuple[int, dict[str, Any]]] = {}
    if lead is not None:
        if lead not in store.leads:
            return ToolResult.error(f"unknown lead {lead!r}. Available: {', '.join(store.leads)}")
        per_lead = _beat_feature_index(store, lead)
        if not per_lead:
            return ToolResult.unavailable(f"no per-beat features stored for lead {lead}")

    sample_lead_row = next(iter(per_lead.values()))[1] if per_lead else {}
    known = set(_BEAT_LEVEL_FIELDS) | set(sample_lead_row)
    unknown = [name for name in fields if name not in known]
    if unknown:
        scope = f"lead {lead}" if lead else "record level (pass `lead` for per-lead fields)"
        return ToolResult.error(
            f"unknown per-beat field(s) at {scope}: {', '.join(unknown)}. "
            f"Available here: {', '.join(sorted(known)[:24])}..."
        )

    selected = [
        beat for beat in beats
        if beat_ids is None or beat.get("beat_id") in set(beat_ids)
    ]
    if not selected:
        return ToolResult.unavailable(f"no beats matched beat_ids={beat_ids}")

    units = {name: infer_unit(name) for name in fields}
    headers = ["beat"] + [f"{n} ({units[n]})" if units[n] else n for n in fields]

    citations: list[str] = []
    rows: list[list[str]] = []
    for position, beat in enumerate(beats):
        if beat not in selected:
            continue
        beat_id = beat.get("beat_id", position)
        row = [str(beat_id)]
        for name in fields:
            if name in _BEAT_LEVEL_FIELDS:
                value = beat.get(name)
                pointer = f"/beats/{position}/{name}"
            else:
                entry = per_lead.get(beat_id)
                if entry is None:
                    value = None
                    pointer = ""
                else:
                    feature_index, feature_row = entry
                    value = feature_row.get(name)
                    pointer = f"/beat_features/{feature_index}/{name}"
            cell, citation = (
                _citable_cell(store, pointer, value, units[name])
                if pointer
                else (format_cell(value, units[name]), None)
            )
            if citation is not None:
                citations.append(citation)
            row.append(cell)
        rows.append(row)

    scope_line = f"lead {lead}" if lead else "record level (rhythm)"
    text = f"per-beat measurements, {scope_line}, {len(rows)} beats\n\n" + markdown_table(headers, rows)
    text, truncated = truncate_lines(text, MAX_TABLE_LINES)
    return ToolResult(ok=True, text=text, citations=tuple(citations), truncated=truncated)


# ---------------------------------------------------------------------------
# search_measurements
# ---------------------------------------------------------------------------
def search_measurements(store: EvidenceStore, query: str, limit: int = 15) -> ToolResult:
    limit = max(1, min(int(limit), 40))
    hits = store.search(query, limit=limit)
    if not hits:
        return ToolResult(
            ok=True,
            text=f"no measurement field matches {query!r}. "
                 "Try a shorter fragment such as 'qt', 'p_dur', 'st_j' or 'axis'.",
        )
    rows = [
        [
            hit.pointer,
            hit.unit or "",
            format_cell(hit.value, hit.unit),
        ]
        for hit in hits
    ]
    text = markdown_table(["pointer", "unit", "sample value"], rows)
    note = None
    if any("<LEAD>" in hit.pointer for hit in hits):
        note = (
            f"<LEAD> is a placeholder; substitute a lead name "
            f"({', '.join(store.leads[:4])}, ...) or use get_lead_table for all 12 at once. "
            "Sample values shown are from the first lead."
        )
    # Discovery only; a value seen here is not yet citable evidence.
    return ToolResult(ok=True, text=text, citations=(), note=note)


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_global_table",
        description=(
            "Read up to 24 global ECG measurements in one table. Use this first for "
            "rate, intervals, axes and rhythm summary fields. Fields may be exact names "
            "or clinical aliases. Every row exposes its canonical pointer, value, unit "
            "and reliability caveat; copy the exact ev:/ pointer into diagnostic evidence."
        ),
        parameters={
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Global field names or aliases, for example "
                    "['heart_rate_bpm','pr_ms','qrs_ms','QTc','qrs_axis_deg','rr_cv']."
                ),
            }
        },
        required=("fields",),
        handler=get_global_table,
    ),
    ToolSpec(
        name="get_measurement",
        description=(
            "Read one measurement by JSON pointer (e.g. /global_features/qtc_bazett_ms) "
            "or clinical alias (e.g. 'QTc Bazett', 'PR', 'QRS axis'). Returns the value "
            "with its unit and every reliability caveat ecgfeat recorded for it. "
            "Use this for single global numbers; use get_lead_table for per-lead values."
        ),
        parameters={
            "pointer": {
                "type": "string",
                "description": "JSON pointer such as /global_features/qt_ms, or a clinical alias such as 'QTc'.",
            }
        },
        required=("pointer",),
        handler=get_measurement,
    ),
    ToolSpec(
        name="get_lead_table",
        description=(
            "Read the same measurement fields across all 12 leads at once, as a table. "
            "This is the highest-value tool for morphology questions: one call gives the "
            "full territorial picture. Example fields: q_duration_ms, q_amp_mv, r_amp_mv, "
            "s_amp_mv, st_hybrid_j_mv, t_amp_mv, t_polarity, p_dur_ms, qrs_ms. "
            "Every cell includes its exact [ev:/...] citation; copy that token verbatim. "
            "Cells marked * are lower-weight observations from a lead flagged for that "
            "wave; they remain usable when qualified and cross-checked."
        ),
        parameters={
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Per-lead parameter names, at most 12 per call.",
            },
            "leads": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional lead subset, e.g. ['V1','V2','V3']. Defaults to all leads.",
            },
        },
        required=("fields",),
        handler=get_lead_table,
    ),
    ToolSpec(
        name="get_beat_table",
        description=(
            "Read measurements beat by beat. Without `lead`, returns rhythm-level fields "
            "(rr_prev_ms, rr_next_ms, paced, group_id) needed for regularity, pauses and "
            "ectopy. With `lead`, returns that lead's per-beat fields (pr_ms, qrs_ms, qt_ms, "
            "st_on_mv, t_amp_mv, ...) needed for Wenckebach, alternans and beat-to-beat drift. "
            "group_id is only a QRS morphology cluster; it does not establish P-QRS "
            "association, AV conduction, or rhythm stability. "
            "Every stored cell includes its exact [ev:/...] citation; copy it verbatim."
        ),
        parameters={
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Per-beat field names, at most 12 per call.",
            },
            "lead": {
                "type": "string",
                "description": "Optional lead for per-lead beat fields. Omit for rhythm-level fields.",
            },
            "beat_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional beat id subset. Defaults to all beats.",
            },
        },
        required=("fields",),
        handler=get_beat_table,
    ),
    ToolSpec(
        name="search_measurements",
        description=(
            "Find measurement field names and their pointers by keyword, when you do not "
            "know the exact field name. Discovery only: values shown here are samples and "
            "must be re-read with get_measurement or get_lead_table before being cited."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": "Fragment of a field name, e.g. 'qt', 'terminal_p', 'st_j', 'axis'.",
            },
            "limit": {"type": "integer", "description": "Max hits, 1-40 (default 15)."},
        },
        required=("query",),
        handler=search_measurements,
    ),
)
