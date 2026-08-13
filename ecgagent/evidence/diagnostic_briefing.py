"""Neutral intake briefing for the diagnosis-first ECG agent.

It exposes record identity, acquisition metadata, data inventory and quality
limitations.  It intentionally contains no clinical-rule conclusions,
matched statement codes, reference labels, or ready-made diagnoses.
"""
from __future__ import annotations

from typing import Any

from .briefing import Briefing
from .store import EvidenceStore


def _shown(
    store: EvidenceStore,
    citations: list[str],
    pointer: str,
    label: str,
) -> str | None:
    evidence = store.try_resolve(pointer)
    if evidence is None:
        return None
    citations.append(evidence.pointer)
    return (
        f"{label}={evidence.format_value()}"
        f"{(' ' + evidence.unit) if evidence.unit else ''} "
        f"[{evidence.citation}]"
    )


def _refutable_flag_lines(
    store: EvidenceStore,
    citations: list[str],
) -> list[str]:
    """Render each raised gate flag as a claim plus the criteria it rests on.

    A technical flag reaches the model as a hypothesis about the recording, not
    as a finding it must adopt. Showing the criteria is what makes disagreeing
    with the flag a checkable act rather than an act of defiance: the model can
    read the same pointers the criteria name and see whether they hold.
    """
    flags = store.raw("/metadata/diagnostic_gate/refutable_flags", None)
    if not isinstance(flags, list) or not flags:
        return []
    lines: list[str] = []
    citations.append("/metadata/diagnostic_gate/refutable_flags")
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        reason = str(flag.get("reason") or "")
        if not reason:
            continue
        if not flag.get("refutable", True):
            lines.append(
                f"flag[{reason}] severity={flag.get('severity')} "
                "not refutable by patient measurement"
            )
            continue
        criteria = [
            item for item in (flag.get("criteria") or []) if isinstance(item, dict)
        ]
        rule = (
            "any failing criterion refutes it"
            if flag.get("refuted_when_any_fails")
            else "check every criterion"
        )
        lines.append(
            f"flag[{reason}] severity={flag.get('severity')} "
            f"claims: {flag.get('claim') or reason} ({rule})"
        )
        for item in criteria:
            pointers = " ".join(
                f"ev:{pointer}" for pointer in (item.get("pointers") or [])
            )
            lines.append(
                f"  criterion {item.get('id')}: {item.get('test')} "
                f"— {item.get('description')} {pointers}".rstrip()
            )
    return lines


def build_diagnostic_briefing(store: EvidenceStore) -> Briefing:
    citations: list[str] = []
    identity: list[str] = [f"record={store.record_id}"]
    for pointer, label in (
        ("/metadata/patient_meta/age", "age"),
        ("/metadata/patient_meta/sex", "sex"),
        ("/metadata/input_fs", "sampling_rate"),
        ("/metadata/duration_sec", "duration"),
        ("/metadata/n_beats", "detected_beats"),
    ):
        rendered = _shown(store, citations, pointer, label)
        if rendered:
            identity.append(rendered)

    quality: list[str] = []
    for pointer, label in (
        ("/metadata/record_quality/record_grade", "record_grade"),
        ("/metadata/diagnostic_gate/state", "quality_gate"),
        ("/metadata/diagnostic_gate/stop_reasons", "stop_reasons"),
        ("/metadata/diagnostic_gate/partial_reasons", "partial_reasons"),
        ("/metadata/diagnostic_gate/suppressed_domains", "suppressed_domains"),
    ):
        rendered = _shown(store, citations, pointer, label)
        if rendered:
            quality.append(rendered)
    quality.extend(_refutable_flag_lines(store, citations))

    unreliable: list[str] = []
    for lead in store.leads:
        flags: list[str] = []
        lead_quality = store.lead_quality(lead)
        for wave, key in (
            ("P", "reliable_for_p"),
            ("QRS", "reliable_for_qrs"),
            ("T", "reliable_for_t"),
            ("QT", "reliable_for_qt"),
        ):
            if lead_quality.get(key) is False:
                pointer = f"/quality/{lead}/{key}"
                if store.try_resolve(pointer) is not None:
                    citations.append(pointer)
                    flags.append(f"{wave}[ev:{pointer}]")
        if flags:
            unreliable.append(f"{lead}: {','.join(flags)}")

    global_fields = set(store.global_fields())
    bundles: dict[str, tuple[str, ...]] = {
        "rate_rhythm": (
            "heart_rate_bpm",
            "atrial_rate_bpm",
            "rr_mean_ms",
            "rr_cv",
            "rmssd_ms",
            "pnn50",
        ),
        "intervals": (
            "pr_ms",
            "qrs_ms",
            "qt_ms",
            "qtc_bazett_ms",
            "qtc_fridericia_ms",
        ),
        "axes": ("p_axis_deg", "qrs_axis_deg", "t_axis_deg", "st_axis_deg"),
    }
    inventory = [
        f"{name}: {', '.join(field for field in fields if field in global_fields) or 'none'}"
        for name, fields in bundles.items()
    ]
    inventory.extend(
        [
            "rhythm_modality: use get_rhythm_profile; use get_atrial_event_table "
            "and get_p_assessment_table for P-QRS organization",
            "morphology_families: use get_morphology_groups before generalizing "
            "a representative or single beat",
            "morphology_maps: use get_morphology_map(profile=p/qrs/st/t_u/quality) "
            "for fixed cross-lead views",
            "pacing_modality: use get_pacing_profile when pacing may confound "
            "rhythm, intervals or ST-T",
            "per_lead_detail: use get_lead_table to inspect custom fields across "
            + (", ".join(store.leads) if store.leads else "no leads"),
            "per_beat_detail: use get_beat_table for rhythm regularity, ectopy and "
            "beat-to-beat interval behavior",
        ]
    )

    lines = [
        "[record] " + " | ".join(identity),
        "[quality] " + (" | ".join(quality) if quality else "no quality metadata"),
        "[wave quality cautions] "
        + (
            "; ".join(f"{item} (lower-weight, not discarded)" for item in unreliable)
            if unreliable
            else "no lead-level wave caution reported"
        ),
        "[available measurement bundles]",
        *[f"  - {item}" for item in inventory],
        "",
        "A quality caution does not erase a measured waveform feature. Values that "
        "exist remain usable as explicitly limited, lower-weight evidence and should "
        "be triangulated across leads or beats; null/not-produced values are unavailable.",
        "No ecgfeat rule conclusion, Philips DXL/Glasgow reference "
        "interpretation, external knowledge-library text, raw waveform window or "
        "remeasurement result is included in this diagnostic view. Form the "
        "interpretation independently from measurement and modality tools.",
    ]
    return Briefing(
        text="\n".join(lines),
        citations=tuple(dict.fromkeys(citations)),
    )


def build_compact_diagnostic_briefing(store: EvidenceStore) -> Briefing:
    """Return routing context without a second patient-evidence namespace.

    The small-model workflow receives every citable value through the
    orchestrated overview packet as a short Qn atom.  Repeating full ev:/
    pointers in the briefing made Qwen reconcile two citation syntaxes and
    accounted for several otherwise inexplicable binding failures.  Quality
    reasons remain useful routing context, but are rendered here as
    non-citable limitations; the immutable source document remains available
    to the deterministic final renderer and audit trace.
    """

    gate = store.raw("/metadata/diagnostic_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    gate_state = str(gate.get("state") or "unknown")
    reasons = [
        str(value)
        for value in (
            gate.get("stop_reasons")
            or gate.get("partial_reasons")
            or []
        )
        if str(value).strip()
    ]
    suppressed = [
        str(value)
        for value in (gate.get("suppressed_domains") or [])
        if str(value).strip()
    ]
    lines = [
        f"record={store.record_id}",
        f"quality_gate={gate_state}",
        "quality_limitations=" + (", ".join(reasons) if reasons else "none"),
        "suppressed_domains=" + (
            ", ".join(suppressed) if suppressed else "none"
        ),
        "available_modalities=global, lead, beat, rhythm, atrial_event, "
        "P_assessment, morphology, interval_context, pacing, native_beat",
        "All citable patient measurements arrive in the following prefetched "
        "Qn evidence packet. This briefing contains routing context only.",
    ]
    return Briefing(text="\n".join(lines), citations=())
