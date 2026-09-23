"""Compact, diagnosis-neutral intake view for the Survey phase."""
from __future__ import annotations

from typing import Any
import math

from ..evidence.store import EvidenceStore
from ._render import markdown_table
from .registry import ToolResult, ToolSpec


_OVERVIEW_POINTERS: tuple[tuple[str, str], ...] = (
    ("patient age (years)", "/metadata/patient_meta/age"),
    ("patient age (days)", "/metadata/patient_meta/age_days"),
    ("patient sex", "/metadata/patient_meta/sex"),
    ("amplitude calibration", "/metadata/input_contract/amplitude_calibration"),
    ("amplitude output unit", "/metadata/input_contract/amplitude_output_unit"),
    ("sampling rate", "/metadata/input_fs"),
    ("record quality", "/metadata/record_quality/record_grade"),
    ("measurement gate", "/metadata/diagnostic_gate/state"),
    ("detected beats", "/metadata/n_beats"),
    ("heart rate", "/global_features/heart_rate_bpm"),
    ("atrial rate", "/global_features/atrial_rate_bpm"),
    ("mean RR", "/global_features/rr_mean_ms"),
    ("RR variability", "/global_features/rr_cv"),
    (
        "background RR regular",
        "/rhythm_inputs/background/background_rr_regular",
    ),
    (
        "atrial detector indeterminate",
        "/rhythm_inputs/af_afl/af_afl_indeterminate",
    ),
    (
        "fine-wave candidate-detector confidence",
        "/rhythm_inputs/af_afl/f_wave_confidence",
    ),
    (
        "fine-wave candidate-detector multilead consensus",
        "/rhythm_inputs/af_afl/f_wave_multilead_consensus",
    ),
    (
        "localized atrial-event excess",
        "/rhythm_inputs/av_block/evidence/localized_atrial_event_excess",
    ),
    ("PR", "/global_features/pr_ms"),
    ("QRS", "/global_features/qrs_ms"),
    ("QT", "/global_features/qt_ms"),
    ("QT reportable", "/global_features/qt_reportable"),
    ("QT reliability", "/global_features/qt_reliability"),
    ("QTc Bazett", "/global_features/qtc_bazett_ms"),
    ("QTc Fridericia", "/global_features/qtc_fridericia_ms"),
    ("P axis", "/global_features/p_axis_deg"),
    ("QRS axis", "/global_features/qrs_axis_deg"),
    ("T axis", "/global_features/t_axis_deg"),
    (
        "atrial rhythm available",
        "/rhythm_inputs/record/availability/atrial_rhythm_available",
    ),
    ("PR available", "/rhythm_inputs/record/availability/pr_available"),
    ("P axis available", "/rhythm_inputs/record/availability/p_axis_available"),
    ("pacing state", "/rhythm_inputs/pacing/state"),
    ("pacing evidence conflicted", "/rhythm_inputs/pacing/evidence_conflicted"),
)


def get_diagnostic_overview(store: EvidenceStore) -> ToolResult:
    """Return one bounded overview; detailed morphology remains pull-only."""

    rows: list[list[str]] = []
    citations: list[str] = []
    screening: list[tuple[str, str]] = []
    # Select existing extremes, without applying thresholds or creating
    # diagnostic flags. They raise questions; full cross-lead views must still
    # establish distribution, reliability and competing explanations.
    for field in ("st_hybrid_j_mv", "t_amp_mv", "q_duration_ms", "r_amp_mv", "s_amp_mv"):
        values = []
        for lead in store.leads:
            pointer = f"/representative_leads/{lead}/params/{field}"
            evidence = store.try_resolve(pointer)
            value = evidence.value if evidence else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.append((float(value), lead, pointer))
        if values:
            for value, lead, pointer in dict.fromkeys((min(values), max(values))):
                screening.append((f"screening extreme {lead}.{field}", pointer))
    for label, pointer in (*_OVERVIEW_POINTERS, *screening):
        evidence = store.try_resolve(pointer)
        if evidence is None:
            continue
        value = evidence.format_value()
        if evidence.unit and not isinstance(evidence.value, (bool, str, list, dict)):
            value += f" {evidence.unit}"
        caveat = "; ".join(evidence.caveats)
        rows.append(
            [
                label,
                f"{value} [{evidence.citation}]",
                "limited" if caveat else "usable",
                caveat,
            ]
        )
        citations.extend((evidence.pointer, *evidence.companions))

    if not rows:
        return ToolResult.unavailable("diagnostic overview measurements are unavailable")
    inventory = (
        "Screening extremes are individual measurements, not cross-lead patterns or diagnoses. "
        "Available pull views: rhythm/P-AV, interval component context, QRS and "
        "T/U morphology maps, morphology families, native-beat panels, pacing, "
        "focused per-lead/per-beat tables. Mark those domains not_assessed until "
        "a targeted view is read."
    )
    return ToolResult(
        ok=True,
        text=(
            markdown_table(
                ["observation", "value and citation", "reliability", "caveat"],
                rows,
            )
            + "\n\n"
            + inventory
        ),
        citations=tuple(dict.fromkeys(citations)),
        note=(
            "This overview contains measurements and availability only. It does "
            "not establish morphology patterns or diagnostic conclusions."
        ),
    )


SPECS = (
    ToolSpec(
        name="get_diagnostic_overview",
        description=(
            "Return one compact diagnosis-neutral overview of quality, rate, "
            "rhythm-screening observations, intervals, axes and modality "
            "availability and bounded measured morphology extremes. Full morphology must be "
            "queried after hypotheses are formed."
        ),
        parameters={},
        handler=get_diagnostic_overview,
    ),
)


__all__ = ["SPECS", "get_diagnostic_overview"]
