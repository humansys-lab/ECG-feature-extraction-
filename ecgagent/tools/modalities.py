"""High-level, diagnosis-neutral views over ecgfeat modality evidence.

These tools expose observations that already exist in the feature artifact.
They do not call ``clinical_rules``, Philips DXL, Glasgow interpretation,
knowledge documents, raw waveforms, or any remeasurement path.  Their purpose
is to stop the diagnostic model from reconstructing a rhythm or morphology
profile through dozens of low-level per-beat calls.
"""
from __future__ import annotations

from typing import Any, Iterable

from ..evidence.pointer import infer_unit
from ..evidence.store import EvidenceStore
from ._render import format_cell, markdown_table, truncate_lines
from .registry import ToolResult, ToolSpec

MAX_MODALITY_LINES = 80


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _cell(
    store: EvidenceStore,
    pointer: str,
    *,
    value: Any = None,
    unit: str | None = None,
) -> tuple[str, str | None, bool]:
    evidence = store.try_resolve(pointer)
    if evidence is None:
        return "-", None, False
    shown = evidence.value if value is None else value
    shown_unit = evidence.unit if unit is None else unit
    text = format_cell(shown, shown_unit)
    text += f" [{evidence.citation}]"
    if evidence.caveats:
        text += "*"
    return text, evidence.pointer, bool(evidence.caveats)


def _pointer_table(
    store: EvidenceStore,
    rows: Iterable[tuple[str, str]],
) -> ToolResult:
    rendered: list[list[str]] = []
    citations: list[str] = []
    cautioned = False
    for label, pointer in rows:
        evidence = store.try_resolve(pointer)
        if evidence is None:
            continue
        value = evidence.format_value()
        if evidence.unit and not isinstance(evidence.value, (bool, str, list, dict)):
            value += f" {evidence.unit}"
        value += f" [{evidence.citation}]"
        if evidence.caveats:
            value += "*"
            cautioned = True
        rendered.append([label, pointer, value])
        citations.extend((evidence.pointer, *evidence.companions))
    if not rendered:
        return ToolResult.error("the requested modality evidence is unavailable")
    text = markdown_table(["observation", "pointer", "value"], rendered)
    if cautioned:
        text += (
            "\n\n* marks lower-confidence evidence. Preserve the limitation and "
            "triangulate it before using it diagnostically."
        )
    return ToolResult(
        ok=True,
        text=text,
        citations=_unique(citations),
        note=(
            "Detector observations are evidence, not diagnoses. No clinical-rule "
            "or reference-engine conclusion is included."
        ),
    )


_RHYTHM_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "background": (
        ("clean background RR", "/rhythm_inputs/background/clean_rr_ms"),
        ("background RR regular", "/rhythm_inputs/background/background_rr_regular"),
        (
            "background ventricular rate",
            "/rhythm_inputs/background/background_ventricular_rate_bpm",
        ),
        (
            "background atrial rate",
            "/rhythm_inputs/background/background_atrial_rate_bpm",
        ),
        ("dominant morphology group", "/rhythm_inputs/background/dominant_group_id"),
        ("excluded beats", "/rhythm_inputs/background/excluded_beat_ids"),
        ("exclusion reasons", "/rhythm_inputs/background/exclusion_reasons"),
    ),
    "atrial_signal": (
        (
            "atrial rhythm measurements available",
            "/rhythm_inputs/record/availability/atrial_rhythm_available",
        ),
        ("PR measurements available", "/rhythm_inputs/record/availability/pr_available"),
        (
            "P-axis measurements available",
            "/rhythm_inputs/record/availability/p_axis_available",
        ),
        ("RR coefficient of variation", "/rhythm_inputs/af_afl/rr_cv"),
        ("RR RMSSD", "/rhythm_inputs/af_afl/rr_rmssd"),
        ("RR entropy", "/rhythm_inputs/af_afl/rr_entropy"),
        (
            "atrial detector indeterminate",
            "/rhythm_inputs/af_afl/af_afl_indeterminate",
        ),
        ("indeterminate reasons", "/rhythm_inputs/af_afl/indeterminate_reasons"),
        (
            "fine-wave candidate-detector confidence",
            "/rhythm_inputs/af_afl/f_wave_confidence",
        ),
        (
            "fine-wave candidate-detector multilead consensus",
            "/rhythm_inputs/af_afl/f_wave_multilead_consensus",
        ),
        (
            "flutter-wave candidate-detector confidence",
            "/rhythm_inputs/af_afl/F_wave_confidence",
        ),
        (
            "flutter-wave candidate-detector multilead consensus",
            "/rhythm_inputs/af_afl/F_wave_multilead_consensus",
        ),
        (
            "atrial signal stability",
            "/rhythm_inputs/af_afl/atrial_signal_stability",
        ),
        (
            "atrial signal repetitiveness",
            "/rhythm_inputs/af_afl/atrial_signal_repetitiveness",
        ),
        (
            "dominant atrial cycle",
            "/rhythm_inputs/af_afl/dominant_atrial_cycle_ms",
        ),
        (
            "QRST subtraction available",
            "/rhythm_inputs/af_afl/qrst_subtraction/available",
        ),
        ("QRST subtraction reason", "/rhythm_inputs/af_afl/qrst_subtraction/reason"),
        (
            "residual spectral entropy",
            "/rhythm_inputs/af_afl/qrst_subtraction_quality/spectral_entropy",
        ),
        (
            "residual organized spectral power",
            "/rhythm_inputs/af_afl/qrst_subtraction_quality/organized_spectral_power_ratio",
        ),
        (
            "residual fibrillatory-wave confidence",
            "/rhythm_inputs/af_afl/qrst_subtraction_quality/fibrillatory_wave_confidence",
        ),
    ),
    "av_association": (
        (
            "atrial events per RR",
            "/rhythm_inputs/av_block/evidence/atrial_events_per_rr",
        ),
        ("PR sequence", "/rhythm_inputs/av_block/evidence/pr_series_ms"),
        (
            "atrial-event excess indices",
            "/rhythm_inputs/av_block/evidence/atrial_event_excess_indices",
        ),
        (
            "localized atrial-event excess",
            "/rhythm_inputs/av_block/evidence/localized_atrial_event_excess",
        ),
        (
            "constant multiple events need validation",
            "/rhythm_inputs/av_block/evidence/constant_multiple_atrial_events_requires_validation",
        ),
        (
            "dropped-P interval indices",
            "/rhythm_inputs/av_block/evidence/dropped_p_interval_indices",
        ),
        (
            "dropped-P evidence",
            "/rhythm_inputs/av_block/evidence/dropped_p_evidence",
        ),
        (
            "atrial events per RR maximum",
            "/rhythm_inputs/av_block/atrial_events_per_rr_max",
        ),
        ("escape-origin observation", "/rhythm_inputs/av_block/escape_origin"),
    ),
    "preexcitation": (
        ("short PR interval observed", "/rhythm_inputs/preexcitation/short_pr_interval"),
        ("short PR segment observed", "/rhythm_inputs/preexcitation/short_pr_segment"),
        (
            "delta-wave candidate lead count",
            "/rhythm_inputs/preexcitation/delta_lead_count",
        ),
        ("delta-wave candidate leads", "/rhythm_inputs/preexcitation/delta_leads"),
        (
            "delta-wave candidate beat ids",
            "/rhythm_inputs/preexcitation/delta_beat_ids",
        ),
        (
            "delta candidate confidence by lead",
            "/rhythm_inputs/preexcitation/delta_confidence_by_lead",
        ),
        (
            "mean QRS duration in detector",
            "/rhythm_inputs/preexcitation/mean_qrs_duration_ms",
        ),
        (
            "initial QRS axis",
            "/rhythm_inputs/preexcitation/initial_qrs_axis_deg",
        ),
    ),
    "aberrancy": (
        (
            "post-pause or interpolated beats",
            "/rhythm_inputs/aberrancy/post_pause_or_interpolated_beats",
        ),
        ("escape candidates", "/rhythm_inputs/aberrancy/escape_candidates"),
        (
            "interpolated candidates",
            "/rhythm_inputs/aberrancy/interpolated_candidates",
        ),
    ),
}


def get_rhythm_profile(
    store: EvidenceStore,
    sections: list[str] | None = None,
) -> ToolResult:
    """Return organized rhythm observations without a rhythm diagnosis."""
    requested = sections or list(_RHYTHM_SECTIONS)
    unknown = [name for name in requested if name not in _RHYTHM_SECTIONS]
    if unknown:
        return ToolResult.error(
            f"unknown rhythm section(s): {', '.join(unknown)}. "
            f"Available: {', '.join(_RHYTHM_SECTIONS)}"
        )
    rows: list[tuple[str, str]] = []
    for section in requested:
        rows.extend(
            (f"{section}.{label}", pointer)
            for label, pointer in _RHYTHM_SECTIONS[section]
        )
    result = _pointer_table(store, rows)
    if not result.ok:
        return result

    safety_notes: list[str] = []
    if "atrial_signal" in requested:
        safety_notes.append(
            "f/F-wave confidence and multilead consensus are outputs of one "
            "candidate-detector chain. They are one evidence family, not "
            "independent confirmation of AF/flutter. Corroborate them with RR "
            "behavior, measured atrial-versus-ventricular organization, AV "
            "association, residual-signal quality and P-boundary quality."
        )
    if "preexcitation" in requested:
        preexcitation = (
            (store.document.get("rhythm_inputs") or {}).get("preexcitation") or {}
        )
        delta_count = preexcitation.get("delta_lead_count")
        short_pr = bool(
            preexcitation.get("short_pr_interval")
            or preexcitation.get("short_pr_segment")
        )
        if isinstance(delta_count, (int, float)) and delta_count > 0 and not short_pr:
            safety_notes.append(
                "EVIDENCE TIER: delta candidates are present without independent "
                "PR-shortening corroboration. Treat them as candidate-only: they "
                "may trigger focused review or a differential, but cannot by "
                "themselves establish ventricular pre-excitation."
            )
        else:
            safety_notes.append(
                "Delta lead/beat/confidence fields come from one candidate-detector "
                "chain. Count them as one evidence family and independently check "
                "PR timing plus repeated compatible initial-QRS morphology."
            )
    if safety_notes:
        result.note = " ".join(
            [
                result.note or "",
                *safety_notes,
            ]
        ).strip()
    return result


_ATRIAL_EVENT_FIELDS = (
    "p_event_id",
    "time_ms",
    "onset_ms",
    "offset_ms",
    "duration_ms",
    "amplitude_mv",
    "area_mv_ms",
    "signed_area_mv_ms",
    "template_similarity",
    "pp_ms",
    "confidence",
    "associated_qrs_beat_id",
    "association_type",
    "pr_ms",
    "axis_deg",
    "source",
    "source_leads",
    "boundary_source",
)


def get_atrial_event_table(
    store: EvidenceStore,
    fields: list[str] | None = None,
    association_types: list[str] | None = None,
    event_ids: list[int] | None = None,
    limit: int = 24,
) -> ToolResult:
    """Read the independent atrial-event stream as a citable table."""
    chosen = fields or [
        "time_ms",
        "confidence",
        "association_type",
        "associated_qrs_beat_id",
        "pr_ms",
        "axis_deg",
        "source_leads",
    ]
    unknown = [name for name in chosen if name not in _ATRIAL_EVENT_FIELDS]
    if unknown:
        return ToolResult.error(
            f"unknown atrial-event field(s): {', '.join(unknown)}. "
            f"Available: {', '.join(_ATRIAL_EVENT_FIELDS)}"
        )
    limit = max(1, min(int(limit), 40))
    wanted_ids = set(event_ids or [])
    wanted_types = {str(item) for item in (association_types or [])}
    events = store.document.get("rhythm_inputs", {}).get("p_events") or []
    selected: list[tuple[int, dict[str, Any]]] = []
    for position, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        event_id = event.get("p_event_id")
        if wanted_ids and event_id not in wanted_ids:
            continue
        if wanted_types and str(event.get("association_type")) not in wanted_types:
            continue
        selected.append((position, event))
        if len(selected) >= limit:
            break
    if not selected:
        return ToolResult.error("no atrial events matched the requested filters")

    headers = ["event"] + [
        f"{name} ({infer_unit(name)})" if infer_unit(name) else name
        for name in chosen
    ]
    rows: list[list[str]] = []
    citations: list[str] = []
    cautioned = False
    for position, event in selected:
        event_id = event.get("p_event_id", position)
        row = [str(event_id)]
        for name in chosen:
            pointer = f"/rhythm_inputs/p_events/{position}/{name}"
            cell, citation, caution = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
            cautioned = cautioned or caution
        rows.append(row)
    text = "independent atrial-event stream\n\n" + markdown_table(headers, rows)
    text, truncated = truncate_lines(text, MAX_MODALITY_LINES)
    return ToolResult(
        ok=True,
        text=text,
        citations=_unique(citations),
        truncated=truncated,
        note=(
            "Association labels are detector observations. Inspect their confidence, "
            "source leads and repeated pattern before inferring a rhythm. "
            "Rows whose `boundary_source` is `p_wave_assessment_fusion` carry the "
            "P assessment's own fused boundaries, so they are the same measurement "
            "as get_p_assessment_table and do not independently corroborate it. "
            "P duration, amplitude and shape for a chamber conclusion come from "
            "get_morphology_map, not from this stream."
        ),
    )


_P_ASSESSMENT_FIELDS = (
    "beat_id",
    "accepted",
    "reject_reasons",
    "strict_onset",
    "strict_offset",
    "robust_onset",
    "robust_offset",
    "onset_confidence",
    "offset_confidence",
    "valid_leads",
    "valid_lead_groups",
    "baseline_mode",
    "t_reconstructed",
    "ta_ambiguous",
    "global_detector_method",
    "morphology_cluster_id",
    "temporal_jitter_ms",
    "ci_calibration_status",
)


def get_p_assessment_table(
    store: EvidenceStore,
    fields: list[str] | None = None,
    beat_ids: list[int] | None = None,
    limit: int = 24,
) -> ToolResult:
    """Expose P-boundary acceptance and uncertainty without ``p_state`` labels."""
    chosen = fields or [
        "accepted",
        "reject_reasons",
        "onset_confidence",
        "offset_confidence",
        "valid_leads",
        "ta_ambiguous",
        "morphology_cluster_id",
    ]
    unknown = [name for name in chosen if name not in _P_ASSESSMENT_FIELDS]
    if unknown:
        return ToolResult.error(
            f"unknown P-assessment field(s): {', '.join(unknown)}. "
            f"Available: {', '.join(_P_ASSESSMENT_FIELDS)}"
        )
    limit = max(1, min(int(limit), 40))
    wanted = set(beat_ids or [])
    assessments = store.document.get("p_wave_assessments") or []
    selected: list[tuple[int, dict[str, Any]]] = []
    for position, assessment in enumerate(assessments):
        if not isinstance(assessment, dict):
            continue
        if wanted and assessment.get("beat_id") not in wanted:
            continue
        selected.append((position, assessment))
        if len(selected) >= limit:
            break
    if not selected:
        return ToolResult.error("no P-wave assessments matched the requested beat ids")

    show_diagnostic_use = "accepted" in chosen and "ta_ambiguous" in chosen
    headers = ["beat"] + [
        f"{name} ({infer_unit(name)})" if infer_unit(name) else name
        for name in chosen
    ]
    if show_diagnostic_use:
        headers.append("diagnostic_use")
    rows: list[list[str]] = []
    citations: list[str] = []
    for position, assessment in selected:
        row = [str(assessment.get("beat_id", position))]
        for name in chosen:
            pointer = f"/p_wave_assessments/{position}/{name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        if show_diagnostic_use:
            if not bool(assessment.get("accepted")):
                diagnostic_use = "rejected_boundary"
            elif bool(assessment.get("ta_ambiguous")):
                diagnostic_use = "candidate_only_ta_ambiguous"
            else:
                diagnostic_use = "clean_boundary_support"
            row.append(diagnostic_use)
        rows.append(row)
    text = "P-wave boundary and quality assessments\n\n" + markdown_table(headers, rows)
    text, truncated = truncate_lines(text, MAX_MODALITY_LINES)
    return ToolResult(
        ok=True,
        text=text,
        citations=_unique(citations),
        truncated=truncated,
        note=(
            "`accepted` means the boundary passed the extraction pipeline; it does "
            "not override `ta_ambiguous`. An accepted-but-ambiguous row and its "
            "morphology_cluster_id are candidate-only, not clean evidence of a "
            "distinct P-wave morphology. Rejected or uncertain rows remain useful "
            "as limitations and counterevidence."
        ),
    )


_GROUP_FIELDS = (
    "member_count",
    "member_pct",
    "longest_run",
    "mean_rr_ms",
    "mean_pr_ms",
    "mean_qrs_ms",
    "mean_qt_ms",
    "mean_ventr_rate_bpm",
)


def get_morphology_groups(
    store: EvidenceStore,
    include_beats: bool = True,
) -> ToolResult:
    """Describe QRS morphology families and their beat-level prevalence."""
    groups = store.document.get("groups") or {}
    if not groups:
        return ToolResult.error("no morphology groups are available")
    citations: list[str] = []
    rows: list[list[str]] = []
    headers = [
        "group",
        *_GROUP_FIELDS,
        "dominant",
        "wide_qrs",
        "template_members",
        "template_outliers",
        "template_used",
        "template_corr",
    ]
    for key, group in sorted(
        groups.items(),
        key=lambda item: int((item[1] or {}).get("group_id", item[0])),
    ):
        if not isinstance(group, dict):
            continue
        group_id = group.get("group_id", key)
        row = [str(group_id)]
        for field_name in _GROUP_FIELDS:
            pointer = f"/groups/{key}/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        for field_name in ("dominant_group", "wide_qrs"):
            pointer = f"/groups/{key}/flags/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        meta_root = f"/metadata/measurement_representative_beat_meta/{key}"
        if store.try_resolve(f"{meta_root}/member_count") is None:
            meta_root = f"/metadata/representative_beat_meta/{key}"
        for field_name in (
            "member_count",
            "outlier_count",
            "used_count",
            "mean_template_corr",
        ):
            pointer = f"{meta_root}/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        rows.append(row)

    text_parts = [markdown_table(headers, rows)]
    for label, pointer in (
        ("measurement representative group", "/metadata/representative_group_id"),
        ("initial measurement group", "/metadata/initial_measurement_group_id"),
        ("measurement beat ids", "/metadata/measurement_beat_ids"),
        (
            "measurement group reselected",
            "/metadata/measurement_group_reselected",
        ),
        (
            "measurement reselect reason",
            "/metadata/measurement_group_reselect_reason",
        ),
    ):
        evidence = store.try_resolve(pointer)
        if evidence is None:
            continue
        value = evidence.format_value() + f" [{evidence.citation}]"
        text_parts.append(f"{label}: {value}")
        citations.append(evidence.pointer)

    if include_beats:
        beat_rows: list[list[str]] = []
        for position, beat in enumerate(store.document.get("beats") or []):
            if not isinstance(beat, dict):
                continue
            beat_row = [str(beat.get("beat_id", position))]
            for field_name in ("group_id", "rr_prev_ms", "rr_next_ms", "paced"):
                pointer = f"/beats/{position}/{field_name}"
                cell, citation, _ = _cell(store, pointer)
                beat_row.append(cell)
                if citation:
                    citations.append(citation)
            beat_rows.append(beat_row)
        if beat_rows:
            text_parts.extend(
                [
                    "",
                    "beat membership",
                    markdown_table(
                        ["beat", "group_id", "rr_prev_ms", "rr_next_ms", "paced"],
                        beat_rows,
                    ),
                ]
            )

    text, truncated = truncate_lines("\n".join(text_parts), MAX_MODALITY_LINES)
    return ToolResult(
        ok=True,
        text=text,
        citations=_unique(citations),
        truncated=truncated,
        note=(
            "A single representative beat describes only its morphology family. "
            "Use member_pct, runs and beat membership before generalizing it to the ECG. "
            "Template member/outlier/used counts and correlation show how much repeated-"
            "beat support the representative waveform actually has."
        ),
    )


_MORPHOLOGY_PROFILES: dict[str, tuple[str, ...]] = {
    "p": (
        "p_dur_ms",
        "p_amp_mv",
        "p_notched",
        "p_biphasic",
        "p_notch_interval_ms",
        "p_initial_amp_mv",
        "p_terminal_duration_ms",
        "p_terminal_amp_mv",
        "p_confidence_mean",
        "p_informative_fraction",
    ),
    "qrs": (
        "qrs_ms",
        "q_duration_ms",
        "q_amp_mv",
        "q_r_ratio",
        "r_amp_mv",
        "r_prime_amp_mv",
        "s_amp_mv",
        "s_prime_amp_mv",
        "qrs_notch_count",
        "qrs_slur_flag",
        "vat_ms",
    ),
    "st": (
        "st_hybrid_j_mv",
        "st_hybrid_60ms_mv",
        "st_hybrid_80ms_mv",
        "st_hybrid_slope_mv_per_ms",
        "st_hybrid_shape",
        "st_pattern_class",
        "st_hybrid_baseline_confidence",
        "st_hybrid_reliable",
        "st_hybrid_unreliable_reason",
    ),
    "t_u": (
        "t_amp_mv",
        "t_polarity",
        "t_symmetry",
        "t_dur_ms",
        "tpe_ms",
        "t_sqi_score",
        "u_amp_signed_mv",
        "u_polarity",
        "u_dur_ms",
        "u_prominence_mv",
        "u_measurement_reliable",
    ),
}

_QUALITY_FIELDS = (
    "grade",
    "baseline_wander_score",
    "muscle_noise_score",
    "powerline_score",
    "clipping_score",
    "flatline_score",
    "reliable_for_p",
    "reliable_for_qrs",
    "reliable_for_t",
    "reliable_for_qt",
)


def get_morphology_map(
    store: EvidenceStore,
    profile: str,
    leads: list[str] | None = None,
) -> ToolResult:
    """Return a fixed, semantically grouped 12-lead morphology map."""
    if profile not in {*_MORPHOLOGY_PROFILES, "quality"}:
        return ToolResult.error(
            f"unknown morphology profile {profile!r}. Available: "
            f"{', '.join((*_MORPHOLOGY_PROFILES, 'quality'))}"
        )
    target_leads = [lead for lead in (leads or store.leads) if lead in store.leads]
    if not target_leads:
        return ToolResult.error(f"no matching leads. Available: {', '.join(store.leads)}")

    fields = (
        _QUALITY_FIELDS
        if profile == "quality"
        else _MORPHOLOGY_PROFILES[profile]
    )
    headers = ["lead"] + [
        f"{name} ({infer_unit(name)})" if infer_unit(name) else name
        for name in fields
    ]
    rows: list[list[str]] = []
    citations: list[str] = []
    cautioned = False
    for lead in target_leads:
        row = [lead]
        for field_name in fields:
            pointer = (
                f"/quality/{lead}/{field_name}"
                if profile == "quality"
                else f"/representative_leads/{lead}/params/{field_name}"
            )
            cell, citation, caution = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
            cautioned = cautioned or caution
        rows.append(row)
    text = f"{profile} morphology map\n\n" + markdown_table(headers, rows)
    if cautioned:
        text += (
            "\n\n* marks a lower-confidence measurement. It is not automatically "
            "discarded; qualify it and look for cross-lead or cross-beat support."
        )
    text, truncated = truncate_lines(text, MAX_MODALITY_LINES)
    if profile == "p":
        map_note = (
            "This is a P-morphology observation map, not a diagnosis. Before "
            "using P polarity, axis or morphology diagnostically, inspect "
            "get_p_assessment_table; accepted+ta_ambiguous boundaries cannot "
            "serve as clean morphology evidence."
        )
    elif profile == "t_u":
        map_note = (
            "This is a T/U observation map, not a diagnosis. State which leads "
            "support the proposed T-wave pattern, which leads oppose it, and how "
            "T SQI changes their weight. Do not infer a record-level T diagnosis "
            "from one lead or from amplitude/polarity without cross-lead review."
        )
    else:
        map_note = (
            "This is a morphology observation map, not a diagnostic-rule result. "
            "Check territorial consistency and morphology-group prevalence."
        )
    return ToolResult(
        ok=True,
        text=text,
        citations=_unique(citations),
        truncated=truncated,
        note=map_note,
    )


def get_qrs_measurement_bundle(
    store: EvidenceStore,
    leads: list[str] | None = None,
) -> ToolResult:
    """Return one coherent, citable QRS measurement object per lead.

    The values are the same representative-lead measurements exposed by
    :func:`get_morphology_map`; only their transport shape differs.  Keeping a
    complete lead in one small atom prevents a rolling model-context budget
    from showing QRS duration from every lead while silently omitting the
    terminal forces, notching and activation time needed for conduction
    morphology adjudication.
    """

    available = store.raw("/measurement_bundles/qrs_by_lead", {})
    available = available if isinstance(available, dict) else {}
    target_leads = [
        lead
        for lead in (leads or store.leads)
        if lead in available
    ]
    if not target_leads:
        return ToolResult.error(
            f"no matching QRS measurement bundles. Available: {', '.join(available)}"
        )
    result = _pointer_table(
        store,
        (
            (f"lead {lead} QRS measurements", f"/measurement_bundles/qrs_by_lead/{lead}")
            for lead in target_leads
        ),
    )
    if not result.ok:
        return result
    return ToolResult(
        ok=True,
        text="compact QRS measurement bundles\n\n" + result.text,
        citations=result.citations,
        note=(
            "Each row is a diagnosis-neutral bundle of the original per-lead "
            "QRS measurements. Component reliability caveats apply to the "
            "whole row; use cross-lead consistency and morphology-group "
            "prevalence before making a record-level interpretation."
        ),
    )


_QT_CONTEXT_GLOBAL_FIELDS = (
    "qt_ms",
    "qtc_bazett_ms",
    "qtc_fridericia_ms",
    "qt_reliability",
    "qt_reportable",
    "qt_unreliable_reasons",
    "qt_path",
    "qt_confidence_reason",
    "qt_excluded_leads",
    "qt_rejected",
    "qt_reject_reason",
    "qrs_wide_ms",
    "rr_cv",
    "t_fusion_support",
    "t_fusion_mad_ms",
    "t_fusion_ci_half_width_ms",
    "t_fusion_reliable",
    "t_fusion_reliability_reasons",
    "t_tail_incomplete_leads",
    "t_systematic_early_risk",
)

_QT_CONTEXT_LEAD_FIELDS = (
    "t_amp_mv",
    "t_polarity",
    "t_symmetry",
    "t_dur_ms",
    "tpe_ms",
    "t_sqi_score",
    "qt_confidence_mean",
    "t_confidence_reason",
    "t_offset_fusion_reliability_reason",
    "t_offset_systematic_early_risk",
    "t_offset_morphology_guard_pass",
)

_PR_CONTEXT_GLOBAL_ROWS = (
    ("PR interval", "/global_features/pr_ms"),
    ("P duration", "/global_features/p_duration_ms"),
    ("P-duration reliability", "/global_features/p_duration_reliability"),
    ("P axis", "/global_features/p_axis_deg"),
    (
        "atrial-rhythm measurements available",
        "/rhythm_inputs/record/availability/atrial_rhythm_available",
    ),
    (
        "PR measurements available",
        "/rhythm_inputs/record/availability/pr_available",
    ),
    (
        "P-axis measurements available",
        "/rhythm_inputs/record/availability/p_axis_available",
    ),
)

_PR_CONTEXT_LEAD_FIELDS = (
    "p_dur_ms",
    "p_amp_mv",
    "p_notched",
    "p_biphasic",
    "p_confidence_mean",
    "p_onset_confidence_mean",
    "p_offset_confidence_mean",
    "p_informative_fraction",
    "p_on_t_overlap_risk",
    "p_ta_overlap_risk",
    "p_baseline_confidence_mean",
)

_INTERVAL_FLAG_TOKENS = {
    "qt": (
        "flat_t_wave",
        "t_end",
        "st_t_confusion",
        "t_unreliable",
        "possible_t_u_fusion",
    ),
    "pr": (
        "p_unreliable",
        "p_isoelectric_uninformative",
        "p_no_tp_quiet_window",
        "p_on_t_overlap_risk",
        "p_ta_offset_risk",
        "p_candidate",
        "p_onset",
        "p_offset",
    ),
}


def _interval_lead_context(
    store: EvidenceStore,
    *,
    fields: tuple[str, ...],
    quality_fields: tuple[str, ...],
) -> tuple[str, list[str]]:
    headers = ["lead", *fields, *quality_fields]
    rows: list[list[str]] = []
    citations: list[str] = []
    for lead in store.leads:
        row = [lead]
        for field_name in fields:
            pointer = f"/representative_leads/{lead}/params/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        for field_name in quality_fields:
            pointer = f"/quality/{lead}/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        rows.append(row)
    return markdown_table(headers, rows), citations


def _interval_flag_context(
    store: EvidenceStore,
    *,
    interval: str,
    limit: int = 24,
) -> tuple[str, list[str]]:
    tokens = _INTERVAL_FLAG_TOKENS[interval]
    rows: list[list[str]] = []
    citations: list[str] = []
    for position, feature in enumerate(store.document.get("beat_features") or []):
        if not isinstance(feature, dict):
            continue
        flags = feature.get("flags") or []
        if not isinstance(flags, list):
            continue
        relevant = [
            str(flag)
            for flag in flags
            if any(token in str(flag) for token in tokens)
        ]
        if not relevant:
            continue
        row: list[str] = []
        for field_name in ("beat_id", "lead"):
            pointer = f"/beat_features/{position}/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        flags_pointer = f"/beat_features/{position}/flags"
        cell, citation, _ = _cell(store, flags_pointer, value=relevant)
        row.append(cell)
        if citation:
            citations.append(citation)
        rows.append(row)
        if len(rows) >= max(1, min(int(limit), 40)):
            break
    if not rows:
        return "No interval-related per-beat failure flags were stored.", citations
    return markdown_table(["beat", "lead", "relevant extraction flags"], rows), citations


def get_interval_waveform_context(
    store: EvidenceStore,
    interval: str,
    include_beat_flags: bool = True,
) -> ToolResult:
    """Link an unavailable interval back to its still-observable waveforms.

    This is a measurement-organization tool, not a diagnostic rule.  It makes
    extraction failure informative without claiming that a waveform pattern
    has one particular disease cause.
    """
    interval = str(interval or "").strip().lower()
    if interval not in {"qt", "pr"}:
        return ToolResult.error("interval must be 'qt' or 'pr'")

    sections: list[str] = []
    citations: list[str] = []
    if interval == "qt":
        status_rows = tuple(
            (field_name, f"/global_features/{field_name}")
            for field_name in _QT_CONTEXT_GLOBAL_FIELDS
        )
        status = _pointer_table(store, status_rows)
        if status.ok:
            sections.extend(["QT/QTc measurement status", status.text])
            citations.extend(status.citations)
        lead_table, lead_citations = _interval_lead_context(
            store,
            fields=_QT_CONTEXT_LEAD_FIELDS,
            quality_fields=("reliable_for_t", "reliable_for_qt"),
        )
        sections.extend(["Residual T-wave evidence by lead", lead_table])
        citations.extend(lead_citations)
        component_note = (
            "A null or non-reportable QT/QTc means the interval endpoint could not "
            "be used reliably; it does not mean that T-wave amplitude, polarity, "
            "distribution, ST/T confusion or U-wave observations are absent. "
            "Separate technical noise, T-end timing ambiguity, rhythm/conduction "
            "confounding and persistent T-wave morphology before concluding."
        )
    else:
        status = _pointer_table(store, _PR_CONTEXT_GLOBAL_ROWS)
        if status.ok:
            sections.extend(["PR measurement and atrial-availability status", status.text])
            citations.extend(status.citations)
        lead_table, lead_citations = _interval_lead_context(
            store,
            fields=_PR_CONTEXT_LEAD_FIELDS,
            quality_fields=("reliable_for_p",),
        )
        sections.extend(["Residual P-wave evidence by lead", lead_table])
        citations.extend(lead_citations)

        assessments = store.document.get("p_wave_assessments") or []
        assessment_rows: list[list[str]] = []
        for position, assessment in enumerate(assessments[:24]):
            if not isinstance(assessment, dict):
                continue
            row: list[str] = []
            for field_name in (
                "beat_id",
                "accepted",
                "reject_reasons",
                "onset_confidence",
                "offset_confidence",
                "valid_leads",
                "ta_ambiguous",
                "temporal_jitter_ms",
            ):
                pointer = f"/p_wave_assessments/{position}/{field_name}"
                cell, citation, _ = _cell(store, pointer)
                row.append(cell)
                if citation:
                    citations.append(citation)
            assessment_rows.append(row)
        if assessment_rows:
            sections.extend(
                [
                    "P-wave boundary and stability evidence",
                    markdown_table(
                        [
                            "beat",
                            "accepted",
                            "reject_reasons",
                            "onset_confidence",
                            "offset_confidence",
                            "valid_leads",
                            "ta_ambiguous",
                            "temporal_jitter_ms",
                        ],
                        assessment_rows,
                    ),
                ]
            )
        component_note = (
            "A null or unavailable PR interval does not prove that P waves are "
            "absent. Review P visibility and morphology separately from boundary "
            "stability and from P-QRS association; retained but ambiguous P evidence "
            "may support a limitation or differential without establishing a clean "
            "atrial mechanism."
        )

    if include_beat_flags:
        flag_table, flag_citations = _interval_flag_context(
            store,
            interval=interval,
        )
        sections.extend(["Per-beat extraction-failure observations", flag_table])
        citations.extend(flag_citations)

    text, truncated = truncate_lines("\n\n".join(sections), MAX_MODALITY_LINES)
    return ToolResult(
        ok=True,
        text=text,
        citations=_unique(citations),
        truncated=truncated,
        note=(
            component_note
            + " This tool exposes measurements and failure provenance only; it "
            "does not supply a clinical-rule or disease conclusion."
        ),
    )


_PACING_POINTERS = (
    ("detector enabled", "/rhythm_inputs/pacing/enabled"),
    ("pacing state", "/rhythm_inputs/pacing/state"),
    ("measurement state", "/rhythm_inputs/pacing/measurement_state"),
    ("detection state", "/rhythm_inputs/pacing/detection_state"),
    ("spike times", "/rhythm_inputs/pacing/spike_times"),
    ("spike count", "/rhythm_inputs/pacing/spike_count"),
    ("paced beat ids", "/rhythm_inputs/pacing/paced_beat_ids"),
    ("continuous pacing", "/rhythm_inputs/pacing/continuous_pacing"),
    ("intermittent pacing", "/rhythm_inputs/pacing/intermittent_pacing"),
    (
        "ventricular pacing detector",
        "/rhythm_inputs/pacing/ventricular_pacing_present",
    ),
    ("atrial pacing detector", "/rhythm_inputs/pacing/atrial_pacing_present"),
    (
        "dual-chamber pacing detector",
        "/rhythm_inputs/pacing/dual_chamber_pacing_present",
    ),
    ("evidence confidence state", "/rhythm_inputs/pacing/confidence_state"),
    ("evidence conflicted", "/rhythm_inputs/pacing/evidence_conflicted"),
    ("conflict reasons", "/rhythm_inputs/pacing/conflict_reasons"),
    (
        "supports measurement routing",
        "/rhythm_inputs/pacing/supports_measurement_routing",
    ),
    ("QRS count", "/rhythm_inputs/pacing/qrs_count"),
    (
        "QRS-associated spike count",
        "/rhythm_inputs/pacing/qrs_associated_spike_count",
    ),
    ("spikes per QRS", "/rhythm_inputs/pacing/spikes_per_qrs"),
    (
        "QRS-associated spike fraction",
        "/rhythm_inputs/pacing/qrs_associated_spike_fraction",
    ),
    (
        "associated beat fraction",
        "/rhythm_inputs/pacing/associated_beat_fraction",
    ),
    (
        "global capture alignment fraction",
        "/rhythm_inputs/pacing/capture_alignment_fraction",
    ),
    (
        "median absolute spike-QRS offset",
        "/rhythm_inputs/pacing/median_abs_spike_qrs_offset_ms",
    ),
    (
        "spike-QRS offset MAD",
        "/rhythm_inputs/pacing/spike_qrs_offset_mad_ms",
    ),
    (
        "legacy capture-alignment alias (artifact_confidence)",
        "/rhythm_inputs/pacing/artifact_confidence",
    ),
    (
        "capture-failure detector",
        "/rhythm_inputs/pacing/capture_failure_suspected",
    ),
    (
        "sensing-failure availability",
        "/rhythm_inputs/pacing/sensing_failure_suspected/available",
    ),
    (
        "sensing-failure reason",
        "/rhythm_inputs/pacing/sensing_failure_suspected/reason",
    ),
)


def get_pacing_profile(
    store: EvidenceStore,
    include_beats: bool = True,
) -> ToolResult:
    """Return pacing-detector observations plus paced beat membership."""
    result = _pointer_table(store, _PACING_POINTERS)
    if not result.ok or not include_beats:
        return result

    pacing = (store.document.get("rhythm_inputs") or {}).get("pacing") or {}
    if "evidence_conflicted" not in pacing:
        spike_count = pacing.get("spike_count")
        alignment = pacing.get(
            "capture_alignment_fraction",
            pacing.get("artifact_confidence"),
        )
        beat_count = len(store.document.get("beats") or [])
        paced_ids = pacing.get("paced_beat_ids") or []
        inferred_reasons: list[str] = []
        if (
            isinstance(spike_count, (int, float))
            and beat_count
            and float(spike_count) / float(beat_count) > 2.5
        ):
            inferred_reasons.append("excessive_spike_burden_per_qrs")
        if (
            isinstance(spike_count, (int, float))
            and float(spike_count) >= 4
            and isinstance(alignment, (int, float))
            and float(alignment) < 0.25
        ):
            inferred_reasons.append("low_global_capture_alignment")
        if (
            isinstance(spike_count, (int, float))
            and float(spike_count) >= 4
            and len(paced_ids) / float(spike_count) < 0.20
        ):
            inferred_reasons.append("sparse_qrs_proximal_spike_support")
        if inferred_reasons:
            result.text = (
                "PACING EVIDENCE CONFLICTED (legacy-artifact audit): "
                + ", ".join(inferred_reasons)
                + ". Candidate spikes remain informative, but this evidence "
                "must not by itself establish pacing or suppress native ST/T.\n\n"
                + result.text
            )
            for pointer in (
                "/rhythm_inputs/pacing/spike_count",
                "/rhythm_inputs/pacing/paced_beat_ids",
                "/rhythm_inputs/pacing/artifact_confidence",
            ):
                if store.try_resolve(pointer) is not None:
                    result.citations = _unique((*result.citations, pointer))

    rows: list[list[str]] = []
    citations = list(result.citations)
    for position, beat in enumerate(store.document.get("beats") or []):
        if not isinstance(beat, dict):
            continue
        row = [str(beat.get("beat_id", position))]
        for field_name in ("paced", "group_id", "rr_prev_ms", "rr_next_ms"):
            pointer = f"/beats/{position}/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            row.append(cell)
            if citation:
                citations.append(citation)
        rows.append(row)
    if rows:
        result.text += "\n\npaced-beat membership\n\n" + markdown_table(
            ["beat", "paced", "group_id", "rr_prev_ms", "rr_next_ms"],
            rows,
        )
        result.text, result.truncated = truncate_lines(
            result.text,
            MAX_MODALITY_LINES,
        )
        result.citations = _unique(citations)
    result.note = (
        "Pacing fields are detector observations, not a device diagnosis. "
        "The detector, spike-to-QRS matcher and paced-beat flags are one "
        "algorithmic chain, not independent corroboration. Positive pacing "
        "requires non-conflicted timing/burden evidence plus compatible repeated "
        "beat morphology. The legacy artifact_confidence field is actually a "
        "capture-alignment fraction."
    )
    return result


_NATIVE_BEAT_PROFILES: dict[str, tuple[str, ...]] = {
    "qrs_infarct": (
        "qrs_ms",
        "q_duration_ms",
        "q_amp_mv",
        "q_r_ratio",
        "r_amp_mv",
        "s_amp_mv",
        "initial_qrs_net_mv",
        "qrs_confidence",
        "beat_measurement_reliable",
    ),
    "repolarization": (
        "qt_ms",
        "qt_confidence",
        "st_hybrid_j_mv",
        "st_hybrid_60ms_mv",
        "st_hybrid_80ms_mv",
        "st_hybrid_slope_mv_per_ms",
        "st_hybrid_shape",
        "st_hybrid_reliable",
        "st_hybrid_unreliable_reason",
        "t_amp_mv",
        "t_polarity",
        "t_symmetry",
        "beat_measurement_reliable",
    ),
}


def _evenly_sampled_ids(beat_ids: list[int], limit: int) -> list[int]:
    ordered = sorted(dict.fromkeys(beat_ids))
    if len(ordered) <= limit:
        return ordered
    if limit <= 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / float(limit - 1))
        for index in range(limit)
    }
    return [ordered[index] for index in sorted(indices)]


def get_native_beat_profile(
    store: EvidenceStore,
    profile: str,
    leads: list[str] | None = None,
    max_beats: int = 3,
) -> ToolResult:
    """Read repeated non-paced beat morphology without a diagnostic rule."""
    if profile not in _NATIVE_BEAT_PROFILES:
        return ToolResult.error(
            f"unknown native-beat profile {profile!r}. Available: "
            f"{', '.join(_NATIVE_BEAT_PROFILES)}"
        )
    max_beats = max(1, min(int(max_beats), 5))
    target_leads = [lead for lead in (leads or store.leads) if lead in store.leads]
    if not target_leads:
        return ToolResult.error(f"no matching leads. Available: {', '.join(store.leads)}")

    compact = (
        (store.document.get("rhythm_inputs") or {})
        .get("native_beat_profiles", {})
    )
    compact_rows = compact.get("rows") if isinstance(compact, dict) else None
    row_refs: list[dict[str, Any]] = []
    if isinstance(compact_rows, list) and compact_rows:
        for position, row in enumerate(compact_rows):
            if not isinstance(row, dict):
                continue
            root = f"/rhythm_inputs/native_beat_profiles/rows/{position}"
            row_refs.append(
                {
                    "row": row,
                    "root": root,
                    "beat_pointer": f"{root}/beat_id",
                    "group_pointer": f"{root}/group_id",
                    "paced_pointer": f"{root}/paced",
                    "lead_pointer": f"{root}/lead",
                }
            )
    else:
        beat_positions = {
            int(beat.get("beat_id", position)): position
            for position, beat in enumerate(store.document.get("beats") or [])
            if isinstance(beat, dict)
        }
        for position, row in enumerate(store.document.get("beat_features") or []):
            if not isinstance(row, dict):
                continue
            beat_id = row.get("beat_id")
            try:
                beat_id = int(beat_id)
            except (TypeError, ValueError):
                continue
            beat_position = beat_positions.get(beat_id)
            beat = (
                (store.document.get("beats") or [])[beat_position]
                if beat_position is not None
                else {}
            )
            merged = {
                **row,
                "group_id": beat.get("group_id"),
                "paced": bool(beat.get("paced", False)),
            }
            root = f"/beat_features/{position}"
            row_refs.append(
                {
                    "row": merged,
                    "root": root,
                    "beat_pointer": f"{root}/beat_id",
                    "group_pointer": (
                        f"/beats/{beat_position}/group_id"
                        if beat_position is not None
                        else None
                    ),
                    "paced_pointer": (
                        f"/beats/{beat_position}/paced"
                        if beat_position is not None
                        else None
                    ),
                    "lead_pointer": f"{root}/lead",
                }
            )
    if not row_refs:
        return ToolResult.error(
            "native per-beat morphology is unavailable; regenerate the feature "
            "artifact with the compact native-beat profile contract or include "
            "beat_features"
        )

    beat_summary: dict[int, tuple[Any, bool]] = {}
    for ref in row_refs:
        row = ref["row"]
        try:
            beat_id = int(row.get("beat_id"))
        except (TypeError, ValueError):
            continue
        beat_summary[beat_id] = (row.get("group_id"), bool(row.get("paced", False)))
    native_beats = {
        beat_id: group_id
        for beat_id, (group_id, paced) in beat_summary.items()
        if not paced
    }
    used_paced_fallback = not native_beats
    eligible = native_beats or {
        beat_id: group_id
        for beat_id, (group_id, _paced) in beat_summary.items()
    }
    groups: dict[Any, list[int]] = {}
    for beat_id, group_id in eligible.items():
        groups.setdefault(group_id, []).append(beat_id)
    selected_group, group_beats = max(
        groups.items(),
        key=lambda item: (len(item[1]), -min(item[1])),
    )
    selected_beats = _evenly_sampled_ids(group_beats, max_beats)
    selected_set = set(selected_beats)

    fields = _NATIVE_BEAT_PROFILES[profile]
    headers = ["beat", "group", "lead", "paced"] + [
        f"{name} ({infer_unit(name)})" if infer_unit(name) else name
        for name in fields
    ]
    rendered: list[list[str]] = []
    citations: list[str] = []
    for ref in row_refs:
        row = ref["row"]
        try:
            beat_id = int(row.get("beat_id"))
        except (TypeError, ValueError):
            continue
        if beat_id not in selected_set or row.get("lead") not in target_leads:
            continue
        cells: list[str] = []
        for pointer_name in (
            "beat_pointer",
            "group_pointer",
            "lead_pointer",
            "paced_pointer",
        ):
            pointer = ref.get(pointer_name)
            if pointer is None:
                cells.append("-")
                continue
            cell, citation, _ = _cell(store, pointer)
            cells.append(cell)
            if citation:
                citations.append(citation)
        for field_name in fields:
            pointer = f"{ref['root']}/{field_name}"
            cell, citation, _ = _cell(store, pointer)
            cells.append(cell)
            if citation:
                citations.append(citation)
        rendered.append(cells)
    if not rendered:
        return ToolResult.error("no native-beat rows matched the requested leads")

    prefix = (
        "WARNING: no non-paced beat was available; showing the dominant paced "
        "family, which is not a native-beat fallback.\n\n"
        if used_paced_fallback
        else ""
    )
    text = (
        prefix
        + f"{profile} repeated-beat profile; selected group={selected_group}, "
        + f"beats={selected_beats}\n\n"
        + markdown_table(headers, rendered)
    )
    text, truncated = truncate_lines(text, MAX_MODALITY_LINES)
    note = (
        "This panel applies no infarct or ischemia rule. Compare Q duration/depth, "
        "R-wave progression and adjacent-lead consistency across repeated beats."
        if profile == "qrs_infarct"
        else
        "This panel preserves non-paced ST/T/QT observations when pacing is "
        "suspected. Unreliable values remain visible with their explicit "
        "reliability fields and must be weighted, not silently discarded."
    )
    return ToolResult(
        ok=True,
        text=text,
        citations=_unique(citations),
        truncated=truncated,
        note=note,
    )


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_rhythm_profile",
        description=(
            "Read organized rhythm-modality observations: background RR/rates, "
            "atrial residual signal, AV association sequences, pre-excitation "
            "candidate measurements and aberrancy candidates. Candidate-detector "
            "chains are explicitly identified and must be independently "
            "corroborated. Returns no rhythm diagnosis and no clinical-rule or "
            "reference-engine result."
        ),
        parameters={
            "sections": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(_RHYTHM_SECTIONS),
                },
                "description": (
                    "Optional subset: background, atrial_signal, av_association, "
                    "preexcitation, aberrancy. Defaults to all."
                ),
            }
        },
        required=(),
        handler=get_rhythm_profile,
    ),
    ToolSpec(
        name="get_atrial_event_table",
        description=(
            "Read independent atrial events with timing, confidence, P-QRS "
            "association, PR, axis and source leads. Use this instead of inferring "
            "atrial organization from one representative P wave."
        ),
        parameters={
            "fields": {
                "type": "array",
                "items": {"type": "string", "enum": list(_ATRIAL_EVENT_FIELDS)},
            },
            "association_types": {
                "type": "array",
                "items": {"type": "string"},
            },
            "event_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 40},
        },
        required=(),
        handler=get_atrial_event_table,
    ),
    ToolSpec(
        name="get_p_assessment_table",
        description=(
            "Read per-beat P-wave boundary acceptance, rejection reasons, confidence "
            "and valid lead support. Accepted+ta_ambiguous rows are explicitly marked "
            "candidate-only for morphology use. The AF-like detector label is "
            "deliberately absent."
        ),
        parameters={
            "fields": {
                "type": "array",
                "items": {"type": "string", "enum": list(_P_ASSESSMENT_FIELDS)},
            },
            "beat_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 40},
        },
        required=(),
        handler=get_p_assessment_table,
    ),
    ToolSpec(
        name="get_morphology_groups",
        description=(
            "Read QRS morphology-family prevalence, run length, group interval "
            "summaries, template correlation and beat membership. Use before "
            "generalizing a single beat to the whole ECG."
        ),
        parameters={
            "include_beats": {
                "type": "boolean",
                "description": "Include beat-to-group membership (default true).",
            }
        },
        required=(),
        handler=get_morphology_groups,
    ),
    ToolSpec(
        name="get_morphology_map",
        description=(
            "Read a fixed 12-lead observation map for one modality profile: P, QRS, "
            "ST, T/U or quality. Avoids field-name guessing and preserves per-wave "
            "reliability. This is not a diagnostic-rule result."
        ),
        parameters={
            "profile": {
                "type": "string",
                "enum": [*_MORPHOLOGY_PROFILES, "quality"],
            },
            "leads": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        required=("profile",),
        handler=get_morphology_map,
    ),
    ToolSpec(
        name="get_qrs_measurement_bundle",
        description=(
            "Read one compact, citable QRS measurement bundle per requested "
            "lead, including duration, Q/R/S/R-prime/S-prime amplitudes, "
            "notching, slurring and ventricular activation time. Use for "
            "bundle/fascicular block, IVCD and ventricular morphology paths "
            "that need complete cross-lead QRS evidence. This is not a rule result."
        ),
        parameters={
            "leads": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        required=(),
        handler=get_qrs_measurement_bundle,
    ),
    ToolSpec(
        name="get_interval_waveform_context",
        description=(
            "When QT/QTc or PR is null, non-reportable or unreliable, read the "
            "interval status together with the residual component-wave evidence "
            "that remains diagnostically informative. QT links to multi-lead T/U "
            "morphology, T-end failure provenance and T/QT quality; PR links to "
            "multi-lead P morphology, P-boundary stability and P-QRS availability. "
            "This is a measurement/failure-context view, not a clinical rule or "
            "disease conclusion."
        ),
        parameters={
            "interval": {
                "type": "string",
                "enum": ["qt", "pr"],
                "description": "The limited interval whose component waves must be reviewed.",
            },
            "include_beat_flags": {
                "type": "boolean",
                "description": "Include relevant per-beat extraction flags (default true).",
            },
        },
        required=("interval",),
        handler=get_interval_waveform_context,
    ),
    ToolSpec(
        name="get_pacing_profile",
        description=(
            "Read pacing detector state, spike evidence, artifact confidence, "
            "capture/sensing availability and paced-beat membership. Detector "
            "observations are not a device diagnosis."
        ),
        parameters={
            "include_beats": {
                "type": "boolean",
                "description": "Include per-beat paced flags (default true).",
            }
        },
        required=(),
        handler=get_pacing_profile,
    ),
    ToolSpec(
        name="get_native_beat_profile",
        description=(
            "Read repeated beats from the dominant non-paced morphology family. "
            "Use qrs_infarct for Q waves and R-wave progression, or repolarization "
            "for native ST/T/QT evidence when pacing may be a confounder. No "
            "diagnostic threshold or external knowledge is returned."
        ),
        parameters={
            "profile": {
                "type": "string",
                "enum": list(_NATIVE_BEAT_PROFILES),
            },
            "leads": {
                "type": "array",
                "items": {"type": "string"},
            },
            "max_beats": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
        },
        required=("profile",),
        handler=get_native_beat_profile,
    ),
)
