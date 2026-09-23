"""Atomic, program-owned evidence views for model context.

Tool renderers are optimized for humans and may produce wide tables.  Local
model backends used to shorten those tables by character position, which can
split a value-bearing row away from its citation or drop a clinically central
row from the middle of the table.  This module converts the rendered result to
an explicit set of immutable evidence atoms *before* any context budgeting.

The same rendered view is used for model context, citation authorization and
the audit trace.  Consequently a pointer is either present together with its
program-resolved value, unit and reliability, or it is absent and cannot be
cited.  The model never has to transcribe a value into intermediate state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .ledger import visible_citations
from .store import EvidenceStore, EvidenceValue


MODEL_EVIDENCE_VIEW_VERSION = "ecgagent.model-evidence.v6"
DEFAULT_VIEW_CHAR_LIMIT = 6_000
OVERVIEW_VIEW_CHAR_LIMIT = 14_000
DEFAULT_MAX_ATOMS = 48
OVERVIEW_MAX_ATOMS = 72
MAX_CAVEATS_PER_ATOM = 2
MAX_CAVEAT_CHARS = 240
MAX_NON_EVIDENCE_NOTE_CHARS = 800

_INDEXED_GROUP_LIMIT = 8
_LEAD_GROUP_LIMIT = 12

_TOOL_FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    "get_diagnostic_overview": (
        "age", "sex", "state", "heart_rate_bpm", "pr_ms", "qrs_ms",
        "qt_reportable", "qtc_fridericia_ms", "qrs_axis_deg",
        "st_hybrid_j_mv", "t_amp_mv", "q_duration_ms", "r_amp_mv", "s_amp_mv",
        "background_rr_regular", "p_axis_deg", "qt_reliability",
    ),
    "get_rhythm_profile": (
        "atrial_rhythm_available",
        "background_rr_regular",
        "f_wave_multilead_consensus",
        "f_wave_confidence",
        "f_wave_supporting_leads",
        "f_wave_morphology_validated",
        "F_wave_multilead_consensus",
        "F_wave_confidence",
        "F_wave_supporting_leads",
        "F_wave_morphology_validated",
        "F_wave_rate_bpm",
        "short_pr_interval",
        "delta_lead_count",
        "delta_leads",
        "delta_beat_ids",
        "atrial_events_per_rr",
        "pr_series_ms",
        "localized_atrial_event_excess",
        "rr_cv",
    ),
    "get_p_assessment_table": (
        "accepted",
        "ta_ambiguous",
        "valid_leads",
        "onset_confidence",
        "offset_confidence",
        "morphology_cluster_id",
        "reject_reasons",
        "beat_id",
    ),
    "get_atrial_event_table": (
        "association_type",
        "confidence",
        "pr_ms",
        "source_leads",
        "associated_qrs_beat_id",
        "time_ms",
        "p_event_id",
        "axis_deg",
    ),
    "get_beat_table": (
        "rr_prev_ms",
        "rr_next_ms",
        "group_id",
        "paced",
        "qrs_ms",
        "pr_ms",
        "qt_ms",
    ),
    "get_morphology_groups": (
        "member_pct",
        "member_count",
        "longest_run",
        "mean_qrs_ms",
        "wide_qrs",
        "mean_rr_ms",
        "mean_pr_ms",
        "mean_qt_ms",
    ),
    "get_interval_waveform_context": (
        "qt_ms",
        "qt_reliability",
        "t_fusion_reliable",
        "t_amp_mv",
        "t_polarity",
        "reliable_for_t",
        "reliable_for_qt",
        "qtc_fridericia_ms",
        "qtc_bazett_ms",
        "pr_ms",
        "pr_reliability",
        "p_dur_ms",
        "p_amp_mv",
        "reliable_for_p",
        "reliable_for_pr",
    ),
}

_MORPHOLOGY_FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    "qrs": (
        "qrs_ms",
        "r_prime_amp_mv",
        "s_amp_mv",
        "r_amp_mv",
        "qrs_notch_count",
        "qrs_slur_flag",
        "vat_ms",
        "q_duration_ms",
        "q_amp_mv",
        "q_r_ratio",
        "s_prime_amp_mv",
    ),
    "st": (
        "st_hybrid_j_mv",
        "st_hybrid_reliable",
        "st_pattern_class",
        "st_hybrid_shape",
        "st_hybrid_slope_mv_per_ms",
        "st_hybrid_60ms_mv",
        "st_hybrid_80ms_mv",
        "st_hybrid_baseline_confidence",
        "st_hybrid_unreliable_reason",
    ),
    "t_u": (
        "t_amp_mv",
        "t_polarity",
        "t_sqi_score",
        "t_symmetry",
        "u_prominence_mv",
        "u_measurement_reliable",
        "t_dur_ms",
        "tpe_ms",
        "u_amp_signed_mv",
        "u_polarity",
        "u_dur_ms",
    ),
}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
    return value


def _clip_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _display_json_value(value: EvidenceValue) -> Any:
    """Preserve the exact numeric precision used by tool rendering."""

    if isinstance(value.value, float):
        rendered = value.format_value()
        try:
            return float(rendered) if "." in rendered else int(rendered)
        except ValueError:  # pragma: no cover - non-finite source value
            return _json_safe(value.display_value)
    return _json_safe(value.display_value)


def _atom(value: EvidenceValue) -> dict[str, Any]:
    reliability = (
        "unavailable"
        if value.is_null
        else "limited"
        if value.caveats
        else "reliable"
    )
    return {
        "citation": value.citation,
        "value": _display_json_value(value),
        "unit": value.unit,
        "reliability": reliability,
        "caveats": [
            _clip_text(caveat, MAX_CAVEAT_CHARS)
            for caveat in value.caveats[:MAX_CAVEATS_PER_ATOM]
        ],
    }


def _pointer_parts(pointer: str) -> list[str]:
    return [part for part in str(pointer).split("/") if part]


def _evidence_group(
    value: EvidenceValue,
    *,
    tool: str,
) -> tuple[str, str] | None:
    """Return a clinically meaningful row/lead group for fair sampling."""

    if tool == "get_diagnostic_overview":
        return None
    parts = _pointer_parts(value.pointer)
    if len(parts) == 3 and parts[:2] == ["measurement_bundles", "qrs_by_lead"]:
        return "lead", parts[2]
    if len(parts) >= 3 and parts[0] == "representative_leads":
        return "lead", parts[1]
    if len(parts) >= 2 and parts[0] == "p_wave_assessments":
        return "p_assessment", parts[1]
    if len(parts) >= 3 and parts[:2] == ["rhythm_inputs", "p_events"]:
        return "atrial_event", parts[2]
    if len(parts) >= 2 and parts[0] == "beat_features":
        return "beat", parts[1]
    if tool == "get_rhythm_profile" and len(parts) >= 2 and parts[0] == "rhythm_inputs":
        # The renderer combines background, record availability, atrial
        # detector and AV-association sections. Treating each section as one
        # group prevents a wide detector block from hiding AV evidence.
        return "rhythm_section", parts[1]
    return None


def _field_name(value: EvidenceValue) -> str:
    parts = _pointer_parts(value.pointer)
    return parts[-1] if parts else ""


def _evenly_spaced(values: list[Any], limit: int) -> list[Any]:
    if len(values) <= limit:
        return values
    if limit <= 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in dict.fromkeys(indices)]


def _field_priority(tool: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    if tool == "get_morphology_map":
        return _MORPHOLOGY_FIELD_PRIORITY.get(str(arguments.get("profile") or ""), ())
    return _TOOL_FIELD_PRIORITY.get(tool, ())


def _interval_contract_prefix(
    values: list[EvidenceValue], arguments: Mapping[str, Any]
) -> list[EvidenceValue]:
    """Put interval status and one residual waveform atom before wide tables.

    The rolling backend budget commonly retains only the first two atoms from
    each tool when many disease-specific views share a phase.  A limited
    interval must still expose both why the interval is limited and one P- or
    T-wave component; otherwise the required interval-context contract is
    impossible to complete even though the raw tool result contains the data.
    """

    interval = str(arguments.get("interval") or "").lower()
    field_groups = (
        (
            ("qt_reliability", "qt_reportable", "qt_ms"),
            ("t_amp_mv", "t_polarity", "reliable_for_t"),
        )
        if interval in {"qt", "qtc", "qt_qtc"}
        else (
            ("pr_ms", "pr_available", "pr_reliability"),
            ("p_amp_mv", "p_dur_ms", "reliable_for_p"),
        )
    )
    prefix: list[EvidenceValue] = []
    used: set[str] = set()
    for fields in field_groups:
        match = next(
            (
                value
                for field in fields
                for value in values
                if value.pointer not in used and _field_name(value) == field
            ),
            None,
        )
        if match is not None:
            prefix.append(match)
            used.add(match.pointer)
    return [*prefix, *(value for value in values if value.pointer not in used)]


def _fair_evidence_order(
    values: list[EvidenceValue],
    *,
    tool: str,
    arguments: Mapping[str, Any],
) -> list[EvidenceValue]:
    """Order atoms across leads/rows before applying the hard atom budget.

    Human tables are row-major. Taking their first N citations therefore
    showed every field from the first few leads/events and none from the rest.
    This function changes only model-view ordering: it preserves every exact
    candidate while making prefix-bounded views representative.
    """

    ungrouped: list[EvidenceValue] = []
    groups: dict[tuple[str, str], list[EvidenceValue]] = {}
    for value in values:
        key = _evidence_group(value, tool=tool)
        if key is None:
            ungrouped.append(value)
        else:
            groups.setdefault(key, []).append(value)
    priority = _field_priority(tool, arguments)
    if not groups:
        if not priority:
            return list(values)
        rank = {field: index for index, field in enumerate(priority)}
        return sorted(
            values,
            key=lambda value: rank.get(_field_name(value), len(rank)),
        )

    keys = list(groups)
    kind = keys[0][0]
    group_limit = _LEAD_GROUP_LIMIT if kind == "lead" else _INDEXED_GROUP_LIMIT
    keys = _evenly_spaced(keys, group_limit)
    ordered: list[EvidenceValue] = []
    used: set[str] = set()

    for field in priority:
        for value in ungrouped:
            if _field_name(value) == field and value.pointer not in used:
                ordered.append(value)
                used.add(value.pointer)
        for key in keys:
            for value in groups[key]:
                if _field_name(value) == field and value.pointer not in used:
                    ordered.append(value)
                    used.add(value.pointer)
                    break

    # Global status values remain available after the high-value fields have
    # been interleaved with grouped lead/beat observations.  This prevents the
    # first few global columns of a wide interval view from crowding all
    # residual P/T component evidence out of an 8K rolling packet.
    ordered.extend(value for value in ungrouped if value.pointer not in used)

    # Round-robin the remaining columns so generic tables and unprioritized
    # fields remain fair as well.
    remaining = [
        [value for value in groups[key] if value.pointer not in used]
        for key in keys
    ]
    max_remaining = max((len(group) for group in remaining), default=0)
    for index in range(max_remaining):
        for group in remaining:
            if index < len(group):
                ordered.append(group[index])
    if tool == "get_interval_waveform_context":
        return _interval_contract_prefix(ordered, arguments)
    return ordered


def _render(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


@dataclass(frozen=True)
class ModelEvidenceView:
    """One bounded view whose citations and values are inseparable."""

    text: str
    citations: tuple[str, ...]
    candidate_citations: tuple[str, ...]
    omitted_count: int
    atom_count: int
    original_chars: int
    required_evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_audit(self) -> dict[str, Any]:
        return {
            "contract": MODEL_EVIDENCE_VIEW_VERSION,
            "atom_count": self.atom_count,
            "candidate_count": len(self.candidate_citations),
            "omitted_count": self.omitted_count,
            "original_chars": self.original_chars,
            "model_context_chars": len(self.text),
            "candidate_citations": list(self.candidate_citations),
            "citations": list(self.citations),
        }


def minimum_evidence(
    values: list[EvidenceValue], *, tool: str, arguments: Mapping[str, Any],
) -> list[EvidenceValue]:
    """A per-view display floor preserves row identities and reliability atoms.

    Explicit field requests take precedence; profile views use their existing
    discriminating field ordering. Each cross-lead row gets up to three fields;
    beat/event samples retain up to three rows with the same field set.
    """
    if tool == "get_diagnostic_overview":
        keys = {"/metadata/patient_meta/age", "/metadata/patient_meta/age_days",
                "/metadata/patient_meta/sex", "/metadata/input_contract/amplitude_calibration",
                "/metadata/input_contract/amplitude_output_unit", "/metadata/diagnostic_gate/state"}
        return [value for value in values if value.pointer in keys]
    if tool == "get_waveform_review":
        # Keep provenance/status together with complete rows. An availability
        # flag alone must never be the only content of a morphology review.
        rows = {}
        summary = []
        for value in values:
            parts = value.pointer.split("/")
            if "/observations/" in value.pointer:
                key = "/".join(parts[:6])
                rows.setdefault(key, []).append(value)
            else:
                summary.append(value)
        profiles = set()
        for key, atoms in rows.items():
            profile = key.split("/")[3]
            if profile not in profiles:
                summary.extend(atoms)
                profiles.add(profile)
        return summary
    if tool == "get_interval_waveform_context":
        return _interval_contract_prefix(values, arguments)[:2]
    groups: dict[tuple[str, str], list[EvidenceValue]] = {}
    for value in values:
        group = _evidence_group(value, tool=tool) or ("summary", "summary")
        groups.setdefault(group, []).append(value)
    fields = tuple(arguments.get("fields") or _field_priority(tool, arguments))[:3]
    chosen: list[EvidenceValue] = []
    indexed = [key for key in groups if key[0] in {"beat", "atrial_event", "p_assessment"}]
    sampled = set(_evenly_spaced(indexed, 3))
    for key, group in groups.items():
        if key in indexed and key not in sampled:
            continue
        # QRS measurement bundles already contain an inseparable per-lead DTO.
        ordered = sorted(group, key=lambda item: fields.index(_field_name(item))
                         if _field_name(item) in fields else len(fields))
        chosen.extend(ordered[:3])
    return list({item.pointer: item for item in chosen}.values())


def build_model_evidence_view(
    store: EvidenceStore,
    *,
    tool: str,
    arguments: Mapping[str, Any] | None,
    rendered_text: str,
    citations: Iterable[str],
    char_limit: int | None = None,
    max_atoms: int | None = None,
    required_by: tuple[str, ...] = (),
) -> ModelEvidenceView:
    """Materialize a bounded atomic view from one successful tool result.

    Only pointers whose exact citation token occurred in the original rendered
    result are candidates.  Values are then resolved again from the immutable
    store, preventing stale or rounded prose in a tool table from becoming a
    second source of truth.  Budgeting happens one complete atom at a time.
    """

    original = str(rendered_text or "")
    candidate_pointers = visible_citations(original, citations)
    resolved: list[EvidenceValue] = []
    for pointer in candidate_pointers:
        evidence = store.try_resolve(pointer)
        if evidence is not None:
            resolved.append(evidence)

    is_overview = str(tool) == "get_diagnostic_overview"
    limit = int(
        char_limit
        if char_limit is not None
        else OVERVIEW_VIEW_CHAR_LIMIT
        if is_overview
        else DEFAULT_VIEW_CHAR_LIMIT
    )
    atom_limit = int(
        max_atoms
        if max_atoms is not None
        else OVERVIEW_MAX_ATOMS
        if is_overview
        else DEFAULT_MAX_ATOMS
    )
    if limit < 2:
        raise ValueError("char_limit must fit at least an empty JSON object")
    base: dict[str, Any] = {
        "contract": MODEL_EVIDENCE_VIEW_VERSION,
        "tool": str(tool),
        "arguments": dict(arguments or {}),
        "evidence": [],
        "omitted_atom_count": 0,
    }
    if not resolved:
        # Non-evidence notes preserve availability/error context but are
        # explicitly separated from citable patient evidence.
        base["non_evidence_note"] = _clip_text(
            original,
            MAX_NON_EVIDENCE_NOTE_CHARS,
        )

    ordered_evidence = _fair_evidence_order(
        resolved,
        tool=str(tool),
        arguments=dict(arguments or {}),
    )
    # These are display minima, not diagnostic criteria. Keep the leading
    # discriminating fields together for every requested lead / sampled row.
    minimum = minimum_evidence(ordered_evidence, tool=str(tool), arguments=dict(arguments or {})) if required_by else []
    required_pointers = {item.pointer for item in minimum}
    ordered_evidence = [*minimum, *(item for item in ordered_evidence if item.pointer not in required_pointers)]
    from .packing import json_size

    # Do not discard half a required cross-lead bundle before the shared
    # packet packer even sees it. Explicit caller limits remain hard limits.
    if minimum:
        if max_atoms is None:
            atom_limit = max(atom_limit, len(minimum))
        if char_limit is None:
            mandatory_atoms = [{**_atom(item), "required_by": list(required_by)} for item in minimum]
            minimum_size = json_size({**base, "evidence": mandatory_atoms,
                                     "omitted_atom_count": len(resolved) - len(minimum)})
            limit = min(24000, max(limit, minimum_size + 512))

    selected: list[EvidenceValue] = []
    base["omitted_atom_count"] = len(resolved)
    base["omission_policy"] = "Only displayed value-bearing atoms are citable. Missing required evidence leaves a node unknown."
    used = json_size(base)
    for evidence in ordered_evidence:
        if len(selected) >= atom_limit:
            break
        atom = _atom(evidence)
        if evidence.pointer in required_pointers:
            atom["required_by"] = list(required_by)
        old = base["omitted_atom_count"]
        cost = json_size(atom) + bool(selected) + len(str(old - 1)) - len(str(old))
        if used + cost > limit:
            continue
        base["evidence"].append(atom)
        base["omitted_atom_count"] -= 1
        used += cost
        selected.append(evidence)

    omitted = len(resolved) - len(selected)
    base["omitted_atom_count"] = omitted
    rendered = _render(base)
    if len(rendered) > limit:
        # A pathological tiny budget may not even fit the metadata shell.
        # Keep the tool result nonempty so backends never fall back to raw text.
        rendered = "{}"
        selected = []
        omitted = len(resolved)
    selected_pointers = tuple(evidence.pointer for evidence in selected)
    return ModelEvidenceView(
        text=rendered,
        citations=selected_pointers,
        candidate_citations=tuple(candidate_pointers),
        omitted_count=omitted,
        atom_count=len(selected),
        original_chars=len(original),
        required_evidence={node: tuple(item.pointer for item in minimum) for node in required_by},
    )


__all__ = [
    "DEFAULT_MAX_ATOMS",
    "DEFAULT_VIEW_CHAR_LIMIT",
    "MODEL_EVIDENCE_VIEW_VERSION",
    "ModelEvidenceView",
    "OVERVIEW_MAX_ATOMS",
    "OVERVIEW_VIEW_CHAR_LIMIT",
    "build_model_evidence_view",
]
