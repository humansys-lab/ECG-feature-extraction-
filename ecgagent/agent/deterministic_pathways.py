"""Program-owned measurement facts for compact diagnostic pathways.

The compact model should interpret conflicts and explain a diagnosis; it should
not count rows, calculate ratios, choose an interval threshold, or reconstruct a
sequence from a sampled table.  This module owns those reproducible operations.

Every resolver is deliberately three-state and conservative:

* ``pass`` requires enough directly measured support;
* ``fail`` requires an explicit definition-level contradiction;
* incomplete, borderline, or detector-only evidence remains ``unknown``.

No function creates a diagnosis.  It only resolves one already selected pathway
node from measurements returned by that node's fixed tool view.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..evidence.store import EvidenceStore
from .runtime import bool_env


DETERMINISTIC_PATHWAY_FACTS_VERSION = "ecgagent.deterministic-pathway-facts.v2"


@dataclass(frozen=True)
class DeterministicPathwayPolicy:
    """Versioned thresholds for reproducible measurement aggregation.

    Borderline zones are intentional.  They keep a deterministic helper from
    silently converting a debatable threshold into a positive diagnosis.
    """

    version: str = DETERMINISTIC_PATHWAY_FACTS_VERSION
    min_sequence_events: int = 5
    min_repeated_events: int = 2
    min_representative_beats: int = 3
    representative_member_pct: float = 50.0
    weak_member_pct: float = 20.0
    clean_p_min_count: int = 3
    clean_p_min_fraction: float = 0.50
    p_boundary_min_confidence: float = 0.50
    p_boundary_min_leads: int = 2
    atrial_event_min_confidence: float = 0.80
    atrial_event_min_leads: int = 2
    irregular_rr_pass_cv: float = 0.12
    regular_rr_fail_cv: float = 0.05
    sinus_p_axis_lower_deg: float = 0.0
    sinus_p_axis_upper_deg: float = 90.0
    sinus_rate_difference_fraction: float = 0.10
    premature_rr_ratio: float = 0.80
    compensatory_pause_ratio: float = 1.10
    # `background_rr_regular` is a whole-strip CV classification: one isolated
    # ectopic beat's short coupling interval and compensatory pause inflate
    # the CV enough to flip it to False even when the surrounding sinus
    # mechanism never wavered. These two constants gate a robust fallback,
    # symmetric with `premature_rr_ratio`/`compensatory_pause_ratio` above: a
    # single premature-plus-pause pair sits well outside +/-20% of the median
    # RR, so counting what fraction of intervals stays within that band
    # recovers "organized apart from a known outlier" without ever having to
    # look at ectopy detection directly.
    background_regular_rr_tolerance_fraction: float = 0.20
    background_regular_min_fraction: float = 0.80
    qrs_complete_bundle_pass_ms: float = 120.0
    qrs_complete_bundle_fail_below_ms: float = 110.0
    qrs_incomplete_rbbb_lower_ms: float = 110.0
    qrs_incomplete_rbbb_upper_ms: float = 120.0
    qtc_prolonged_pass_ms: float = 480.0
    qtc_prolonged_fail_below_ms: float = 450.0
    qtc_marked_prolonged_pass_ms: float = 500.0
    qtc_short_pass_ms: float = 340.0
    qtc_marked_short_pass_ms: float = 330.0
    qtc_short_borderline_upper_ms: float = 360.0
    qtc_short_fail_at_least_ms: float = 370.0
    low_voltage_limb_mv: float = 0.50
    low_voltage_precordial_mv: float = 1.00
    pacing_min_spikes: int = 3
    pacing_support_fraction: float = 0.50
    pacing_strong_capture_fraction: float = 0.80
    pacing_failure_fraction: float = 0.25
    # QRS duration at or above which ventricular activation is abnormal enough
    # that QRS-amplitude hypertrophy criteria stop measuring chamber mass.
    voltage_criteria_invalid_qrs_ms: float = 120.0
    # A single non-dominant complex this wide is a ventricular morphology on its
    # own. Requiring repetition for every PVC made an isolated PVC in a 10 s
    # strip structurally undiagnosable, which conflates two questions: whether
    # the morphology is real, and whether it recurs. Prematurity is established
    # separately by `premature_timing`; between 120 ms and this bound a lone
    # complex stays `unknown`.
    pvc_single_wide_qrs_ms: float = 140.0


DEFAULT_DETERMINISTIC_PATHWAY_POLICY = DeterministicPathwayPolicy()


@dataclass(frozen=True)
class DeterministicFact:
    status: str
    reason_code: str
    evidence: tuple[tuple[str, str], ...] = ()
    input_pointers: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)


_FULL_BBB_CODES = frozenset(
    {
        "right_bundle_branch_block",
        "rbbb_pattern",
        "probable_rbbb_pattern",
        "left_bundle_branch_block",
        "lbbb_pattern",
        "probable_lbbb_pattern",
    }
)
_ADVANCED_AV_SECOND_DEGREE = frozenset(
    {"second_degree_av_block", "second_degree_av_block_pattern"}
)
_ADVANCED_AV_COMPLETE = frozenset(
    {"complete_av_block", "complete_av_block_pattern"}
)
_PACING_CODES = frozenset(
    {
        "paced_rhythm",
        "ventricular_paced_rhythm",
        "intermittent_pacing",
        "pacing_failure_to_capture_suspected",
        "pacing_sensing_failure_suspected",
    }
)
_PVC_CODES = frozenset(
    {"premature_ventricular_complexes", "probable_premature_ventricular_complexes"}
)
# Bifascicular/trifascicular block necessarily includes a complete RBBB
# component (the rule engine's own `_bifascicular` only matches once
# `rbbb_matched` is true), so the same wide-QRS criterion that gates RBBB/LBBB
# applies here too.
_WIDE_FASCICULAR_CODES = frozenset(
    {
        "bifascicular_block_pattern",
        "probable_bifascicular_block_pattern",
        "trifascicular_block_pattern",
    }
)
# Isolated fascicular block is a diagnosis of exclusion made from axis and
# lead-morphology criteria; QRS duration is deliberately not part of the
# match criterion in `clinical_rules/conduction.py::_lafb` (see the comment
# there): a wide QRS from a concurrent RBBB is what makes the combination
# bifascicular, and gating the isolated statement on a narrow QRS would make
# that combination unreachable. This node exists to confirm the QRS
# measurement itself is available, not to threshold it.
_ISOLATED_FASCICULAR_CODES = frozenset(
    {"lafb_pattern", "probable_lafb_pattern", "lpfb_pattern"}
)


#: Shadow switch for every program-owned pathway node.  When set, the model is
#: shown the views again and its own label decides placement, while the program
#: answer is still computed and written to the audit as `program_status`.  That
#: makes each node's model-vs-program agreement measurable *before* the program
#: is trusted with it, and gives one lever to roll the whole migration back.
DETERMINISTIC_NODE_SHADOW_ENV = "ECG_AGENT_DETERMINISTIC_NODES_SHADOW"


def deterministic_nodes_shadowed() -> bool:
    return bool_env(DETERMINISTIC_NODE_SHADOW_ENV, False)


def program_owns_pathway_step(code: str, step_id: str) -> bool:
    """Return whether the compact model should skip this pathway node."""

    if deterministic_nodes_shadowed():
        return False

    always_program = {
        "rate_threshold",
        "axis_threshold",
        "preexcitation_components",
        "dominant_qrs_wide",
        "wide_qrs_representative",
        "preexcitation_excluded",
        "pr_criterion",
        "one_to_one_av",
        "repeated_blocked_atrial_events",
        "sequential_av_pattern",
        "short_pr_criterion",
        "pr_component",
        "interval_reportable",
        "qt_threshold",
        "component_endpoint_support",
        "organized_atrial_activity",
        "sinus_p_support",
        "sinus_mechanism_support",
        "sinus_candidate_stream_reconciled",
        "p_measurement_reliability",
        "organized_p_absent",
        "irregular_ventricular_response",
        "multilead_atrial_support",
        "pattern_representative",
        "premature_timing",
        "representative_event",
        "wide_complex_sequence",
        "territorial_qrs_voltage",
        "right_precordial_voltage",
        "precordial_progression",
        "voltage_criteria_qrs_valid",
        "voltage_criteria_pacing_valid",
    }
    if step_id in always_program:
        return True
    if step_id == "cross_lead_voltage_criterion" and code == "lvh_voltage_criteria":
        return True
    if step_id == "qrs_duration_support" and (
        code in _FULL_BBB_CODES
        or code == "incomplete_rbbb_pattern"
        or code in _WIDE_FASCICULAR_CODES
        or code in _ISOLATED_FASCICULAR_CODES
    ):
        return True
    if code in _PACING_CODES and step_id in {
        "pacing_marker_support",
        "capture_relation_support",
    }:
        return True
    if code in _PVC_CODES and step_id == "ectopic_morphology":
        return True
    return False


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fact(
    status: str,
    reason_code: str,
    evidence: Sequence[tuple[str, str]] = (),
    *,
    inputs: Sequence[str] = (),
    metrics: Mapping[str, Any] | None = None,
) -> DeterministicFact:
    return DeterministicFact(
        status=status,
        reason_code=reason_code,
        evidence=tuple(dict.fromkeys(evidence)),
        input_pointers=tuple(dict.fromkeys(inputs)),
        metrics=dict(metrics or {}),
    )


def _available_value(
    store: EvidenceStore,
    available: set[str],
    pointer: str,
) -> Any:
    return store.raw(pointer, None) if pointer in available else None


def _available_number(
    store: EvidenceStore,
    available: set[str],
    pointer: str,
) -> float | None:
    return _finite(_available_value(store, available, pointer))


def _dominant_group(
    store: EvidenceStore,
    available: set[str],
) -> tuple[str, Mapping[str, Any], dict[str, str]] | None:
    groups = store.raw("/groups", {})
    if not isinstance(groups, Mapping):
        return None
    dominant_id = ""
    for group_id, row in groups.items():
        if not isinstance(row, Mapping):
            continue
        pointer = f"/groups/{group_id}/flags/dominant_group"
        if pointer in available and row.get("flags", {}).get("dominant_group") is True:
            dominant_id = str(group_id)
            break
    if not dominant_id:
        candidate = str(store.raw("/metadata/representative_group_id", "") or "")
        if f"/groups/{candidate}/mean_qrs_ms" in available:
            dominant_id = candidate
    row = groups.get(dominant_id)
    if not dominant_id or not isinstance(row, Mapping):
        return None
    pointers = {
        "count": f"/groups/{dominant_id}/member_count",
        "pct": f"/groups/{dominant_id}/member_pct",
        "run": f"/groups/{dominant_id}/longest_run",
        "qrs": f"/groups/{dominant_id}/mean_qrs_ms",
        "rate": f"/groups/{dominant_id}/mean_ventr_rate_bpm",
    }
    return dominant_id, row, pointers


def _p_assessment_rows(
    store: EvidenceStore,
    available: set[str],
) -> list[tuple[int, Mapping[str, Any]]]:
    rows: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in enumerate(store.document.get("p_wave_assessments") or []):
        if not isinstance(row, Mapping):
            continue
        if f"/p_wave_assessments/{index}/accepted" not in available:
            continue
        rows.append((index, row))
    return rows


def _p_reliability_fact(
    store: EvidenceStore,
    available: set[str],
    *,
    absence_question: bool,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    rows = _p_assessment_rows(store, available)
    if not rows:
        return _fact("unknown", "p_assessment_rows_unavailable")
    clean: list[int] = []
    independent_boundaries: list[int] = []
    absence_analyzable: list[int] = []
    ambiguous = 0
    inputs: list[str] = []
    evidence: list[tuple[str, str]] = []
    for index, row in rows:
        accepted_pointer = f"/p_wave_assessments/{index}/accepted"
        ambiguous_pointer = f"/p_wave_assessments/{index}/ta_ambiguous"
        onset_pointer = f"/p_wave_assessments/{index}/onset_confidence"
        offset_pointer = f"/p_wave_assessments/{index}/offset_confidence"
        valid_leads_pointer = f"/p_wave_assessments/{index}/valid_leads"
        reasons_pointer = f"/p_wave_assessments/{index}/reject_reasons"
        inputs.extend(
            pointer
            for pointer in (
                accepted_pointer,
                ambiguous_pointer,
                onset_pointer,
                offset_pointer,
                valid_leads_pointer,
                reasons_pointer,
            )
            if pointer in available
        )
        accepted = row.get("accepted") is True
        ta_ambiguous = row.get("ta_ambiguous") is True
        onset_confidence = (
            _finite(row.get("onset_confidence"))
            if onset_pointer in available
            else None
        )
        offset_confidence = (
            _finite(row.get("offset_confidence"))
            if offset_pointer in available
            else None
        )
        valid_leads = (
            row.get("valid_leads")
            if valid_leads_pointer in available
            and isinstance(row.get("valid_leads"), list)
            else []
        )
        reject_reasons = {
            str(reason).upper()
            for reason in (
                row.get("reject_reasons")
                if reasons_pointer in available
                and isinstance(row.get("reject_reasons"), list)
                else []
            )
        }
        ambiguous += int(ta_ambiguous)
        if accepted and not ta_ambiguous:
            clean.append(index)
        # ``accepted`` is not an independent absence test: upstream may set it
        # false solely because an AF-like rhythm classifier fired.  Conversely,
        # a rejected row can still contain a strong, multilead P-boundary
        # candidate.  Count those measurements directly so the classifier
        # cannot erase visible organized atrial deflections and then use their
        # erasure to confirm itself.
        boundary_candidate = (
            not ta_ambiguous
            and len(valid_leads) >= policy.p_boundary_min_leads
            and onset_confidence is not None
            and offset_confidence is not None
            and onset_confidence >= policy.p_boundary_min_confidence
            and offset_confidence >= policy.p_boundary_min_confidence
        ) or "RHYTHM_ORGANIZED_ATRIAL_ACTIVITY" in reject_reasons
        if boundary_candidate:
            independent_boundaries.append(index)

        technically_limited = any(
            token in reason
            for reason in reject_reasons
            for token in (
                "INSUFFICIENT",
                "UNOBSERVABLE",
                "DESYNCHRONIZED",
                "OVERLAP",
                "DISAGREEMENT",
            )
        )
        if (
            not ta_ambiguous
            and len(valid_leads) >= policy.p_boundary_min_leads
            and not technically_limited
        ):
            absence_analyzable.append(index)
    clean_fraction = len(clean) / len(rows)
    metrics = {
        "analyzed_p_assessments": len(rows),
        "clean_p_count": len(clean),
        "clean_p_fraction": round(clean_fraction, 4),
        "ambiguous_p_count": ambiguous,
        "independent_p_boundary_count": len(independent_boundaries),
        "absence_analyzable_count": len(absence_analyzable),
    }
    for index in (clean[:2] if clean else [row[0] for row in rows[:2]]):
        evidence.append(
            (
                f"/p_wave_assessments/{index}/accepted",
                "Programmatic count of multi-beat P-wave boundary assessments that passed",
            )
        )
        ambiguous_pointer = f"/p_wave_assessments/{index}/ta_ambiguous"
        if ambiguous_pointer in available:
            evidence.append((ambiguous_pointer, "Programmatic exclusion of P waves with T/A boundary ambiguity"))
    enough_clean = (
        len(clean) >= policy.clean_p_min_count
        and clean_fraction >= policy.clean_p_min_fraction
    )
    if absence_question:
        enough_independent_boundaries = (
            len(independent_boundaries) >= policy.clean_p_min_count
            and len(independent_boundaries) / len(rows)
            >= policy.clean_p_min_fraction
        )
        if enough_clean or enough_independent_boundaries:
            boundary_evidence: list[tuple[str, str]] = []
            for index in independent_boundaries[:2]:
                for field_name in (
                    "onset_confidence",
                    "offset_confidence",
                    "valid_leads",
                ):
                    pointer = f"/p_wave_assessments/{index}/{field_name}"
                    if pointer in available:
                        boundary_evidence.append(
                            (
                                pointer,
                                "Programmatic detection of a repeated, nonambiguous multilead P-wave boundary candidate",
                            )
                        )
            return _fact(
                "fail",
                (
                    "repeated_clean_organized_p_present"
                    if enough_clean
                    else "repeated_independent_p_boundaries_present"
                ),
                boundary_evidence or evidence,
                inputs=inputs,
                metrics=metrics,
            )
        if (
            len(absence_analyzable) >= policy.min_sequence_events
            and not independent_boundaries
            and not clean
        ):
            return _fact(
                "pass",
                "no_independent_p_boundaries_in_analyzable_rows",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        return _fact(
            "unknown",
            "organized_p_absence_not_established",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if enough_clean:
        return _fact(
            "pass",
            "repeated_clean_p_boundaries",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if len(rows) >= policy.min_sequence_events and len(clean) == 0:
        return _fact(
            "unknown",
            "no_clean_p_boundaries_measurement_limited",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    return _fact(
        "unknown",
        "p_boundary_support_insufficient",
        evidence,
        inputs=inputs,
        metrics=metrics,
    )


def _atrial_event_rows(
    store: EvidenceStore,
    available: set[str],
) -> list[tuple[int, Mapping[str, Any]]]:
    rows: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in enumerate(
        store.document.get("rhythm_inputs", {}).get("p_events") or []
    ):
        if not isinstance(row, Mapping):
            continue
        if f"/rhythm_inputs/p_events/{index}/association_type" not in available:
            continue
        rows.append((index, row))
    return rows


def _qualified_atrial_events(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    policy: DeterministicPathwayPolicy,
) -> list[tuple[int, Mapping[str, Any]]]:
    result: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in rows:
        confidence = _finite(row.get("confidence"))
        leads = row.get("source_leads")
        lead_count = len({str(lead) for lead in leads}) if isinstance(leads, list) else 0
        if confidence is not None and confidence < policy.atrial_event_min_confidence:
            continue
        if leads is not None and lead_count < policy.atrial_event_min_leads:
            continue
        result.append((index, row))
    return result


def _one_to_one_av_fact(
    store: EvidenceStore,
    available: set[str],
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    rows = _qualified_atrial_events(_atrial_event_rows(store, available), policy)
    if len(rows) < policy.min_sequence_events:
        return _fact(
            "unknown",
            "insufficient_atrial_events_for_one_to_one",
            metrics={"analyzed_atrial_events": len(rows)},
        )
    types = [str(row.get("association_type") or "") for _, row in rows]
    conducted = [(index, row) for index, row in rows if row.get("association_type") == "conducted"]
    blocked = [(index, row) for index, row in rows if row.get("association_type") == "blocked"]
    retrograde = [(index, row) for index, row in rows if row.get("association_type") == "retrograde"]
    qrs_ids = [row.get("associated_qrs_beat_id") for _, row in conducted]
    unique_qrs = {value for value in qrs_ids if value is not None}
    conducted_fraction = len(conducted) / len(rows)
    inputs = [f"/rhythm_inputs/p_events/{index}/association_type" for index, _ in rows]
    evidence_rows = (blocked or retrograde or conducted)[:3]
    evidence = [
        (
            f"/rhythm_inputs/p_events/{index}/association_type",
            "Programmatic count of P-QRS association types among reliable atrial events",
        )
        for index, _ in evidence_rows
    ]
    metrics = {
        "analyzed_atrial_events": len(rows),
        "conducted_count": len(conducted),
        "blocked_count": len(blocked),
        "retrograde_count": len(retrograde),
        "conducted_fraction": round(conducted_fraction, 4),
        "unique_associated_qrs_count": len(unique_qrs),
    }
    if (
        not blocked
        and not retrograde
        and conducted_fraction >= 0.90
        and len(unique_qrs) == len(conducted)
    ):
        return _fact(
            "pass",
            "stable_unique_one_to_one_av_association",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if len(blocked) >= policy.min_repeated_events or len(retrograde) >= policy.min_repeated_events:
        return _fact(
            "fail",
            "repeated_non_one_to_one_av_association",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    return _fact(
        "unknown",
        "mixed_or_incomplete_av_association",
        evidence,
        inputs=inputs,
        metrics=metrics,
    )


def _repeated_blocked_fact(
    store: EvidenceStore,
    available: set[str],
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    rows = _qualified_atrial_events(_atrial_event_rows(store, available), policy)
    blocked = [(index, row) for index, row in rows if row.get("association_type") == "blocked"]
    inputs = [f"/rhythm_inputs/p_events/{index}/association_type" for index, _ in rows]
    evidence = [
        (
            f"/rhythm_inputs/p_events/{index}/association_type",
            "Programmatic localization of reliable atrial candidate events without an associated QRS",
        )
        for index, _ in blocked[:3]
    ]
    metrics = {
        "analyzed_atrial_events": len(rows),
        "reliable_blocked_event_count": len(blocked),
    }
    if len(blocked) >= policy.min_repeated_events:
        return _fact(
            "pass",
            "repeated_reliable_blocked_atrial_events",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if len(rows) >= policy.min_sequence_events and not blocked:
        return _fact(
            "fail",
            "no_blocked_event_in_analyzable_sequence",
            [
                (
                    f"/rhythm_inputs/p_events/{index}/association_type",
                    "Programmatic check that analyzable atrial events have no nonconducted marker",
                )
                for index, _ in rows[:2]
            ],
            inputs=inputs,
            metrics=metrics,
        )
    return _fact(
        "unknown",
        "blocked_event_repetition_not_established",
        evidence,
        inputs=inputs,
        metrics=metrics,
    )


def _sequential_av_fact(
    store: EvidenceStore,
    available: set[str],
    code: str,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    root = "/rhythm_inputs/av_block/evidence"
    ratio_pointer = f"{root}/atrial_events_per_rr"
    pr_pointer = f"{root}/pr_series_ms"
    dropped_pointer = f"{root}/dropped_p_evidence"
    indices_pointer = f"{root}/dropped_p_interval_indices"
    candidate_pointer = f"{root}/constant_multiple_atrial_events_requires_validation"
    ratios = _available_value(store, available, ratio_pointer)
    prs = _available_value(store, available, pr_pointer)
    dropped = _available_value(store, available, dropped_pointer)
    indices = _available_value(store, available, indices_pointer)
    candidate_only = _available_value(store, available, candidate_pointer)
    if not isinstance(ratios, list):
        return _fact("unknown", "av_sequence_summary_unavailable")
    numeric_ratios = [int(value) for value in ratios if isinstance(value, (int, float))]
    numeric_prs = [float(value) for value in prs if isinstance(value, (int, float))] if isinstance(prs, list) else []
    dropped_indices = [int(value) for value in indices if isinstance(value, (int, float))] if isinstance(indices, list) else []
    ratio_counts = {value: numeric_ratios.count(value) for value in sorted(set(numeric_ratios))}
    pr_mean = statistics.fmean(numeric_prs) if numeric_prs else None
    pr_sd = statistics.pstdev(numeric_prs) if len(numeric_prs) >= 2 else None
    inputs = [pointer for pointer in (ratio_pointer, pr_pointer, dropped_pointer, indices_pointer, candidate_pointer) if pointer in available]
    evidence = [
        (ratio_pointer, "Programmatic read of the complete per-RR atrial candidate-count sequence"),
    ]
    if dropped_pointer in available:
        evidence.append((dropped_pointer, "Programmatic read of independent nonconducted-P evidence status"))
    if indices_pointer in available:
        evidence.append((indices_pointer, "Programmatic read of RR intervals corresponding to nonconducted P waves"))
    metrics = {
        "analyzed_rr_intervals": len(numeric_ratios),
        "atrial_events_per_rr_histogram": ratio_counts,
        "dropped_interval_count": len(dropped_indices),
        "candidate_only_constant_multiple": candidate_only is True,
        "pr_count": len(numeric_prs),
        "pr_mean_ms": round(pr_mean, 2) if pr_mean is not None else None,
        "pr_sd_ms": round(pr_sd, 2) if pr_sd is not None else None,
    }
    if len(numeric_ratios) < policy.min_sequence_events:
        return _fact(
            "unknown",
            "insufficient_rr_intervals_for_av_sequence",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if code in _ADVANCED_AV_SECOND_DEGREE:
        if (
            dropped is True
            and len(dropped_indices) >= policy.min_repeated_events
            and candidate_only is not True
        ):
            return _fact(
                "pass",
                "validated_repeated_dropped_p_sequence",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        if all(value == 1 for value in numeric_ratios) and dropped is False:
            return _fact(
                "fail",
                "stable_one_atrial_event_per_rr_without_drop",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        return _fact(
            "unknown",
            "candidate_atrial_excess_without_validated_conduction_ratio",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if code in _ADVANCED_AV_COMPLETE:
        stable_one_to_one = (
            all(value == 1 for value in numeric_ratios)
            and dropped is False
            and pr_sd is not None
            and pr_sd <= 20.0
        )
        if stable_one_to_one:
            return _fact(
                "fail",
                "stable_pr_one_to_one_refutes_complete_av_dissociation",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        # The available detector summary does not directly localize AV
        # dissociation. Candidate count ratios must not manufacture complete AVB.
        return _fact(
            "unknown",
            "complete_av_dissociation_not_directly_measured",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    return _fact("unknown", "unsupported_advanced_av_code")


def _interval_status_fact(
    store: EvidenceStore,
    available: set[str],
) -> DeterministicFact:
    reportable_pointer = "/global_features/qt_reportable"
    reliability_pointer = "/global_features/qt_reliability"
    reportable = _available_value(store, available, reportable_pointer)
    reliability = str(_available_value(store, available, reliability_pointer) or "").lower()
    qtc_pointer = next(
        (
            pointer
            for pointer in (
                "/global_features/qtc_fridericia_ms",
                "/global_features/qtc_bazett_ms",
            )
            if _available_number(store, available, pointer) is not None
        ),
        "",
    )
    evidence = [
        (reportable_pointer, "Programmatic read of QT/QTc reportability"),
        (reliability_pointer, "Programmatic read of QT/QTc reliability grade"),
    ]
    if qtc_pointer:
        evidence.append((qtc_pointer, "Programmatic confirmation of an available corrected QT measurement"))
    inputs = [pointer for pointer, _ in evidence if pointer in available]
    metrics = {"qt_reportable": reportable, "qt_reliability": reliability, "qtc_pointer": qtc_pointer or None}
    bad = {"unreliable", "unavailable", "not_reportable", "low_confidence"}
    if reportable is True and reliability not in bad and qtc_pointer:
        return _fact("pass", "qt_interval_reportable", evidence, inputs=inputs, metrics=metrics)
    if reportable is False or reliability in bad:
        return _fact("unknown", "qt_interval_not_reportable", evidence, inputs=inputs, metrics=metrics)
    return _fact("unknown", "qt_reportability_incomplete", evidence, inputs=inputs, metrics=metrics)


def _qt_threshold_fact(
    store: EvidenceStore,
    available: set[str],
    code: str,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    reportable = _available_value(store, available, "/global_features/qt_reportable")
    reliability = str(_available_value(store, available, "/global_features/qt_reliability") or "").lower()
    bad = {"unreliable", "unavailable", "not_reportable", "low_confidence"}
    if reportable is not True or reliability in bad:
        return _fact("unknown", "qtc_threshold_requires_reportable_interval")
    qtc_pointer = "/global_features/qtc_fridericia_ms"
    qtc = _available_number(store, available, qtc_pointer)
    if qtc is None:
        qtc_pointer = "/global_features/qtc_bazett_ms"
        qtc = _available_number(store, available, qtc_pointer)
    if qtc is None:
        return _fact("unknown", "qtc_measurement_unavailable")
    inputs = [
        qtc_pointer,
        "/global_features/qt_reportable",
        "/global_features/qt_reliability",
    ]
    evidence = [
        (qtc_pointer, "Programmatic definition-level threshold test using a reportable corrected QT"),
        ("/global_features/qt_reportable", "Programmatic confirmation that QT/QTc is reportable"),
    ]
    metrics = {"qtc_ms": round(qtc, 3), "qtc_source": qtc_pointer.rsplit("/", 1)[-1]}
    status = "unknown"
    reason = "qtc_in_borderline_zone"
    if code == "prolonged_qt":
        if qtc >= policy.qtc_prolonged_pass_ms:
            status, reason = "pass", "qtc_meets_conservative_prolonged_threshold"
        elif qtc < policy.qtc_prolonged_fail_below_ms:
            status, reason = "fail", "qtc_below_prolonged_range"
    elif code == "markedly_prolonged_qt":
        if qtc >= policy.qtc_marked_prolonged_pass_ms:
            status, reason = "pass", "qtc_meets_marked_prolongation_threshold"
        elif qtc < policy.qtc_prolonged_pass_ms:
            status, reason = "fail", "qtc_below_marked_prolongation_range"
    elif code == "short_qt":
        if qtc <= policy.qtc_short_pass_ms:
            status, reason = "pass", "qtc_meets_short_qt_threshold"
        elif qtc >= policy.qtc_short_borderline_upper_ms:
            status, reason = "fail", "qtc_above_short_qt_range"
    elif code == "markedly_short_qt":
        if qtc <= policy.qtc_marked_short_pass_ms:
            status, reason = "pass", "qtc_meets_markedly_short_threshold"
        elif qtc > policy.qtc_short_pass_ms:
            status, reason = "fail", "qtc_above_markedly_short_range"
    elif code == "borderline_short_qt":
        if policy.qtc_short_pass_ms < qtc < policy.qtc_short_borderline_upper_ms:
            status, reason = "pass", "qtc_in_borderline_short_range"
        elif qtc <= policy.qtc_marked_short_pass_ms or qtc >= policy.qtc_short_fail_at_least_ms:
            status, reason = "fail", "qtc_outside_borderline_short_range"
    elif code == "possible_short_qt_pattern":
        if qtc < policy.qtc_short_borderline_upper_ms:
            status, reason = "pass", "qtc_supports_possible_short_qt_pattern"
        elif qtc >= policy.qtc_short_fail_at_least_ms:
            status, reason = "fail", "qtc_does_not_support_short_qt_pattern"
    return _fact(status, reason, evidence, inputs=inputs, metrics=metrics)


def _qrs_duration_fact(
    store: EvidenceStore,
    available: set[str],
    code: str,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    found = _dominant_group(store, available)
    if found is None:
        return _fact("unknown", "dominant_qrs_group_unavailable")
    _, _, pointers = found
    qrs = _available_number(store, available, pointers["qrs"])
    if qrs is None:
        return _fact("unknown", "dominant_qrs_duration_unavailable")
    evidence = [(pointers["qrs"], "Programmatic read of dominant morphology-group QRS duration")]
    metrics = {"dominant_mean_qrs_ms": round(qrs, 3)}
    if code in _FULL_BBB_CODES:
        if qrs >= policy.qrs_complete_bundle_pass_ms:
            return _fact("pass", "qrs_meets_complete_bundle_duration", evidence, inputs=[pointers["qrs"]], metrics=metrics)
        if qrs < policy.qrs_complete_bundle_fail_below_ms:
            return _fact("fail", "qrs_too_narrow_for_complete_bundle_block", evidence, inputs=[pointers["qrs"]], metrics=metrics)
        return _fact("unknown", "qrs_duration_in_bundle_borderline_zone", evidence, inputs=[pointers["qrs"]], metrics=metrics)
    if code == "incomplete_rbbb_pattern":
        if policy.qrs_incomplete_rbbb_lower_ms <= qrs < policy.qrs_incomplete_rbbb_upper_ms:
            return _fact("pass", "qrs_in_incomplete_rbbb_duration_range", evidence, inputs=[pointers["qrs"]], metrics=metrics)
        if qrs >= policy.qrs_incomplete_rbbb_upper_ms or qrs < 100.0:
            return _fact("fail", "qrs_outside_incomplete_rbbb_duration_range", evidence, inputs=[pointers["qrs"]], metrics=metrics)
        return _fact("unknown", "qrs_duration_borderline_for_incomplete_rbbb", evidence, inputs=[pointers["qrs"]], metrics=metrics)
    if code in _WIDE_FASCICULAR_CODES:
        if qrs >= policy.qrs_complete_bundle_pass_ms:
            return _fact("pass", "qrs_meets_complete_bundle_duration", evidence, inputs=[pointers["qrs"]], metrics=metrics)
        if qrs < policy.qrs_complete_bundle_fail_below_ms:
            return _fact("fail", "qrs_too_narrow_for_complete_bundle_block", evidence, inputs=[pointers["qrs"]], metrics=metrics)
        return _fact("unknown", "qrs_duration_in_bundle_borderline_zone", evidence, inputs=[pointers["qrs"]], metrics=metrics)
    if code in _ISOLATED_FASCICULAR_CODES:
        return _fact("pass", "qrs_duration_not_gating_for_isolated_fascicular_block", evidence, inputs=[pointers["qrs"]], metrics=metrics)
    return _fact("unknown", "qrs_duration_not_definition_level_for_code")


def _representative_group_fact(
    store: EvidenceStore,
    available: set[str],
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    found = _dominant_group(store, available)
    if found is None:
        return _fact("unknown", "representative_group_unavailable")
    _, _, pointers = found
    count = _available_number(store, available, pointers["count"])
    pct = _available_number(store, available, pointers["pct"])
    run = _available_number(store, available, pointers["run"])
    evidence = [
        (pointers["count"], "Programmatic read of representative morphology-group beat count"),
        (pointers["pct"], "Programmatic read of representative morphology-group prevalence"),
        (pointers["run"], "Programmatic read of the representative morphology group's longest run"),
    ]
    inputs = [pointer for pointer, _ in evidence if pointer in available]
    metrics = {"member_count": count, "member_pct": pct, "longest_run": run}
    if count is None or pct is None:
        return _fact("unknown", "representative_group_counts_unavailable", evidence, inputs=inputs, metrics=metrics)
    if count >= policy.min_representative_beats and pct >= policy.representative_member_pct:
        return _fact("pass", "morphology_group_is_representative", evidence, inputs=inputs, metrics=metrics)
    if count < 2 or pct < policy.weak_member_pct:
        return _fact("fail", "morphology_group_is_isolated_or_minor", evidence, inputs=inputs, metrics=metrics)
    return _fact("unknown", "morphology_group_representativeness_borderline", evidence, inputs=inputs, metrics=metrics)


def _premature_timing_fact(
    store: EvidenceStore,
    available: set[str],
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    rows: list[tuple[int, float, float | None]] = []
    inputs: list[str] = []
    for index, beat in enumerate(store.document.get("beats") or []):
        if not isinstance(beat, Mapping):
            continue
        prev_pointer = f"/beats/{index}/rr_prev_ms"
        next_pointer = f"/beats/{index}/rr_next_ms"
        if prev_pointer not in available:
            continue
        prev_rr = _finite(beat.get("rr_prev_ms"))
        next_rr = _finite(beat.get("rr_next_ms")) if next_pointer in available else None
        if prev_rr is None:
            continue
        rows.append((index, prev_rr, next_rr))
        inputs.append(prev_pointer)
        if next_pointer in available:
            inputs.append(next_pointer)
    if len(rows) < policy.min_sequence_events:
        return _fact("unknown", "insufficient_rr_intervals_for_prematurity", inputs=inputs, metrics={"rr_count": len(rows)})
    baseline = statistics.median([prev for _, prev, _ in rows])
    premature = [row for row in rows if row[1] < policy.premature_rr_ratio * baseline]
    supported = [row for row in premature if row[2] is not None and row[2] > policy.compensatory_pause_ratio * baseline]
    chosen = supported or premature
    evidence = [
        (f"/beats/{index}/rr_prev_ms", "Programmatic detection of a coupling interval premature relative to the local baseline")
        for index, _, _ in chosen[:2]
    ]
    for index, _, next_rr in supported[:1]:
        if next_rr is not None:
            evidence.append((f"/beats/{index}/rr_next_ms", "Programmatic detection of a prolonged interval after the premature beat"))
    metrics = {
        "rr_count": len(rows),
        "median_rr_ms": round(baseline, 3),
        "premature_event_count": len(premature),
        "premature_with_pause_count": len(supported),
    }
    if premature:
        return _fact("pass", "premature_rr_timing_detected", evidence, inputs=inputs, metrics=metrics)
    return _fact(
        "fail",
        "no_premature_rr_in_analyzable_sequence",
        [(f"/beats/{index}/rr_prev_ms", "Programmatic check showing no definition-level prematurity in beat-to-beat RR intervals") for index, _, _ in rows[:2]],
        inputs=inputs,
        metrics=metrics,
    )


def _robust_background_rr_regular_fact(
    store: EvidenceStore,
    available: set[str],
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact | None:
    """A `background_rr_regular is True` alternative tolerant of one outlier.

    Returns None (never "fail") when the sequence does not clearly support
    "organized apart from a known outlier" -- the caller falls back to its
    existing unknown/unavailable handling rather than treating this as a
    contradiction.
    """

    rows: list[tuple[int, float]] = []
    inputs: list[str] = []
    for index, beat in enumerate(store.document.get("beats") or []):
        if not isinstance(beat, Mapping):
            continue
        pointer = f"/beats/{index}/rr_prev_ms"
        if pointer not in available:
            continue
        prev_rr = _finite(beat.get("rr_prev_ms"))
        if prev_rr is None:
            continue
        rows.append((index, prev_rr))
        inputs.append(pointer)
    if len(rows) < policy.min_sequence_events:
        return None
    baseline = statistics.median([rr for _, rr in rows])
    if baseline <= 0:
        return None
    tolerance = policy.background_regular_rr_tolerance_fraction * baseline
    within = [(index, rr) for index, rr in rows if abs(rr - baseline) <= tolerance]
    fraction = len(within) / len(rows)
    if fraction < policy.background_regular_min_fraction:
        return None
    evidence = [
        (f"/beats/{index}/rr_prev_ms", "Programmatic beat-to-beat RR interval used in a median-anchored, outlier-tolerant background regularity check")
        for index, _ in within[:3]
    ]
    metrics = {
        "rr_count": len(rows),
        "median_rr_ms": round(baseline, 3),
        "within_tolerance_count": len(within),
        "within_tolerance_fraction": round(fraction, 3),
    }
    return _fact(
        "pass",
        "background_regular_apart_from_isolated_outlier_rr",
        evidence,
        inputs=inputs,
        metrics=metrics,
    )


def _pvc_morphology_fact(
    store: EvidenceStore,
    available: set[str],
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    groups = store.raw("/groups", {})
    if not isinstance(groups, Mapping):
        return _fact("unknown", "morphology_groups_unavailable")
    candidates: list[tuple[str, float, float, float]] = []
    inputs: list[str] = []
    for group_id, row in groups.items():
        if not isinstance(row, Mapping) or row.get("flags", {}).get("dominant_group") is True:
            continue
        qrs_pointer = f"/groups/{group_id}/mean_qrs_ms"
        count_pointer = f"/groups/{group_id}/member_count"
        pct_pointer = f"/groups/{group_id}/member_pct"
        qrs = _available_number(store, available, qrs_pointer)
        count = _available_number(store, available, count_pointer)
        pct = _available_number(store, available, pct_pointer)
        if qrs is None or count is None or pct is None:
            continue
        candidates.append((str(group_id), qrs, count, pct))
        inputs.extend((qrs_pointer, count_pointer, pct_pointer))
    wide = [row for row in candidates if row[1] >= 120.0 and row[2] >= 2]
    narrow = [row for row in candidates if row[1] < 110.0 and row[2] >= 2]
    lone_wide = [
        row
        for row in candidates
        if row[1] >= policy.pvc_single_wide_qrs_ms and row[2] < 2
    ]
    metrics = {
        "non_dominant_group_count": len(candidates),
        "repeated_wide_group_count": len(wide),
        "repeated_narrow_group_count": len(narrow),
        "lone_markedly_wide_group_count": len(lone_wide),
        "lone_wide_threshold_ms": policy.pvc_single_wide_qrs_ms,
    }
    if wide:
        group_id, _, _, _ = max(wide, key=lambda row: row[3])
        evidence = [
            (f"/groups/{group_id}/mean_qrs_ms", "Programmatic detection of a repeated nondominant wide-QRS morphology group"),
            (f"/groups/{group_id}/member_count", "Programmatic confirmation that the wide-QRS morphology is not an isolated beat"),
        ]
        return _fact("pass", "repeated_wide_ectopic_morphology", evidence, inputs=inputs, metrics=metrics)
    if lone_wide:
        group_id, qrs_ms, _, _ = max(lone_wide, key=lambda row: row[1])
        return _fact(
            "pass",
            "lone_markedly_wide_ectopic_morphology",
            [
                (
                    f"/groups/{group_id}/mean_qrs_ms",
                    "Programmatic detection of an isolated but markedly wide nondominant QRS morphology group",
                )
            ],
            inputs=inputs,
            metrics=metrics,
        )
    if candidates and narrow and len(narrow) == len(candidates):
        group_id = narrow[0][0]
        return _fact(
            "fail",
            "all_repeated_ectopic_groups_are_narrow",
            [(f"/groups/{group_id}/mean_qrs_ms", "Programmatic detection that repeated nondominant morphologies all have narrow QRS complexes")],
            inputs=inputs,
            metrics=metrics,
        )
    return _fact("unknown", "pvc_morphology_not_established", inputs=inputs, metrics=metrics)


def _wide_complex_sequence_fact(
    store: EvidenceStore,
    available: set[str],
    code: str,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    groups = store.raw("/groups", {})
    if not isinstance(groups, Mapping):
        return _fact("unknown", "morphology_groups_unavailable")
    rows: list[tuple[str, float, float, float]] = []
    inputs: list[str] = []
    for group_id, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        qrs_pointer = f"/groups/{group_id}/mean_qrs_ms"
        run_pointer = f"/groups/{group_id}/longest_run"
        rate_pointer = f"/groups/{group_id}/mean_ventr_rate_bpm"
        qrs = _available_number(store, available, qrs_pointer)
        run = _available_number(store, available, run_pointer)
        rate = _available_number(store, available, rate_pointer)
        if qrs is None or run is None:
            continue
        rows.append((str(group_id), qrs, run, rate or 0.0))
        inputs.extend(pointer for pointer in (qrs_pointer, run_pointer, rate_pointer) if pointer in available)
    qualifying = [row for row in rows if row[1] >= 120.0 and row[2] >= 3]
    if code == "wide_complex_tachycardia":
        qualifying = [row for row in qualifying if row[3] > 100.0]
    metrics = {
        "analyzed_group_count": len(rows),
        "qualifying_wide_run_count": len(qualifying),
    }
    if qualifying:
        group_id, qrs, run, rate = max(qualifying, key=lambda row: (row[2], row[3]))
        evidence = [
            (f"/groups/{group_id}/mean_qrs_ms", "Programmatic confirmation of QRS widening in a consecutive morphology group"),
            (f"/groups/{group_id}/longest_run", "Programmatic confirmation of the longest run of the wide-QRS morphology"),
        ]
        if code == "wide_complex_tachycardia":
            evidence.append((f"/groups/{group_id}/mean_ventr_rate_bpm", "Programmatic confirmation that the consecutive wide-QRS morphology group reaches tachycardia"))
        return _fact("pass", "representative_consecutive_wide_complex_sequence", evidence, inputs=inputs, metrics={**metrics, "qrs_ms": qrs, "longest_run": run, "rate_bpm": rate})
    if rows and all(row[1] < 110.0 or row[2] < 3 for row in rows):
        group_id = rows[0][0]
        return _fact(
            "fail",
            "no_representative_consecutive_wide_complex_sequence",
            [(f"/groups/{group_id}/mean_qrs_ms", "Programmatic check found no wide-QRS group meeting duration and continuity requirements")],
            inputs=inputs,
            metrics=metrics,
        )
    return _fact("unknown", "wide_complex_sequence_borderline", inputs=inputs, metrics=metrics)


def _territorial_low_voltage_fact(
    store: EvidenceStore,
    available: set[str],
    code: str,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    limb = "limb" in code
    leads = ["I", "II", "III", "aVR", "aVL", "aVF"] if limb else ["V1", "V2", "V3", "V4", "V5", "V6"]
    threshold = policy.low_voltage_limb_mv if limb else policy.low_voltage_precordial_mv
    amplitudes: dict[str, float] = {}
    inputs: list[str] = []
    evidence: list[tuple[str, str]] = []
    for lead in leads:
        pointers = [
            f"/representative_leads/{lead}/params/{field}"
            for field in ("r_amp_mv", "s_amp_mv", "q_amp_mv")
        ]
        values = [_available_number(store, available, pointer) for pointer in pointers]
        if values[0] is None or values[1] is None:
            continue
        positive = max(0.0, *(value for value in values if value is not None))
        negative = min(0.0, *(value for value in values if value is not None))
        amplitudes[lead] = positive - negative
        inputs.extend(pointer for pointer, value in zip(pointers, values) if value is not None)
    metrics = {
        "territory": "limb" if limb else "precordial",
        "analyzable_lead_count": len(amplitudes),
        "threshold_mv": threshold,
        "peak_to_peak_mv": {lead: round(value, 4) for lead, value in amplitudes.items()},
    }
    if len(amplitudes) < len(leads):
        return _fact("unknown", "incomplete_lead_set_for_low_voltage", inputs=inputs, metrics=metrics)
    for lead in leads[:2]:
        evidence.append((f"/representative_leads/{lead}/params/r_amp_mv", "Programmatic calculation of the positive component of target-lead peak-to-peak QRS voltage"))
        evidence.append((f"/representative_leads/{lead}/params/s_amp_mv", "Programmatic calculation of the negative component of target-lead peak-to-peak QRS voltage"))
    if all(value < threshold for value in amplitudes.values()):
        return _fact("pass", "all_territorial_leads_below_low_voltage_threshold", evidence, inputs=inputs, metrics=metrics)
    opposing = next(lead for lead, value in amplitudes.items() if value >= threshold)
    evidence = [
        (f"/representative_leads/{opposing}/params/r_amp_mv", "Programmatic finding that at least one target lead does not meet the low-voltage threshold"),
        (f"/representative_leads/{opposing}/params/s_amp_mv", "Programmatic calculation of peak-to-peak QRS voltage in that lead"),
    ]
    return _fact("fail", "territorial_lead_not_low_voltage", evidence, inputs=inputs, metrics=metrics)


def _pacing_fact(
    store: EvidenceStore,
    available: set[str],
    code: str,
    step_id: str,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    root = "/rhythm_inputs/pacing"
    pointers = {
        "state": f"{root}/state",
        "spikes": f"{root}/spike_count",
        "associated": f"{root}/qrs_associated_spike_fraction",
        "capture": f"{root}/capture_alignment_fraction",
        "conflicted": f"{root}/evidence_conflicted",
        "routing": f"{root}/supports_measurement_routing",
        "intermittent": f"{root}/intermittent_pacing",
        "capture_failure": f"{root}/capture_failure_suspected",
        "sensing_available": f"{root}/sensing_failure_suspected/available",
        "sensing_value": f"{root}/sensing_failure_suspected/value",
    }
    values = {name: _available_value(store, available, pointer) for name, pointer in pointers.items()}
    spike_count = _finite(values["spikes"])
    associated = _finite(values["associated"])
    capture = _finite(values["capture"])
    inputs = [pointer for pointer in pointers.values() if pointer in available]
    metrics = {
        "state": values["state"],
        "spike_count": spike_count,
        "qrs_associated_spike_fraction": associated,
        "capture_alignment_fraction": capture,
        "evidence_conflicted": values["conflicted"],
        "supports_measurement_routing": values["routing"],
    }
    if step_id == "pacing_marker_support":
        if spike_count == 0 or str(values["state"] or "").lower() in {"off", "none", "false"}:
            return _fact("fail", "no_pacing_markers", [(pointers["spikes"], "Programmatic check found no pacing spikes")], inputs=inputs, metrics=metrics)
        positive = (
            spike_count is not None
            and spike_count >= policy.pacing_min_spikes
            and values["conflicted"] is False
            and values["routing"] is True
        )
        if code == "intermittent_pacing":
            positive = positive and values["intermittent"] is True
        if positive:
            return _fact(
                "pass",
                "nonconflicted_repeated_pacing_markers",
                [(pointers["spikes"], "Programmatic confirmation of repeated pacing spikes"), (pointers["conflicted"], "Programmatic confirmation that pacing evidence is not marked as conflicted")],
                inputs=inputs,
                metrics=metrics,
            )
        return _fact("unknown", "pacing_markers_candidate_or_conflicted", inputs=inputs, metrics=metrics)
    if code == "pacing_sensing_failure_suspected":
        if values["sensing_available"] is not True:
            return _fact("unknown", "sensing_failure_detector_unavailable", inputs=inputs, metrics=metrics)
        status = "pass" if values["sensing_value"] is True else "fail"
        return _fact(status, "sensing_failure_flag_true" if status == "pass" else "sensing_failure_flag_false", [(pointers["sensing_value"], "Programmatic read of the sensing-failure detector result")], inputs=inputs, metrics=metrics)
    if code == "pacing_failure_to_capture_suspected":
        if values["capture_failure"] is True or (
            spike_count is not None
            and spike_count >= policy.pacing_min_spikes
            and capture is not None
            and capture < policy.pacing_failure_fraction
        ):
            return _fact("pass", "repeated_spikes_with_low_capture_alignment", [(pointers["capture_failure"], "Programmatic read of the failure-to-capture detector result"), (pointers["capture"], "Programmatic calculation of post-spike capture alignment")], inputs=inputs, metrics=metrics)
        if capture is not None and capture >= policy.pacing_strong_capture_fraction:
            return _fact("fail", "stable_capture_refutes_failure_to_capture", [(pointers["capture"], "Programmatic confirmation of a stable spike-QRS capture relationship")], inputs=inputs, metrics=metrics)
        return _fact("unknown", "capture_failure_not_established", inputs=inputs, metrics=metrics)
    if spike_count is None or associated is None or capture is None:
        return _fact("unknown", "pacing_capture_metrics_incomplete", inputs=inputs, metrics=metrics)
    if (
        spike_count >= policy.pacing_min_spikes
        and associated >= policy.pacing_support_fraction
        and capture >= policy.pacing_support_fraction
        and values["conflicted"] is False
    ):
        return _fact("pass", "stable_pacing_capture_relation", [(pointers["associated"], "Programmatic confirmation of the spike-to-QRS association rate"), (pointers["capture"], "Programmatic confirmation of spike-QRS capture alignment")], inputs=inputs, metrics=metrics)
    if spike_count >= policy.pacing_min_spikes and capture < policy.pacing_failure_fraction:
        return _fact("fail", "pacing_markers_without_stable_capture", [(pointers["capture"], "Programmatic finding of insufficient spike-QRS capture")], inputs=inputs, metrics=metrics)
    return _fact("unknown", "pacing_capture_relation_borderline", inputs=inputs, metrics=metrics)


def _voltage_validity_fact(
    store: EvidenceStore,
    available: set[str],
    step_id: str,
    policy: DeterministicPathwayPolicy,
) -> DeterministicFact:
    """Decide whether a QRS-voltage criterion is applicable to this record.

    Cornell/Peguero/Sokolow-Lyon all assume normal ventricular activation.
    Ventricular pacing, a complete bundle-branch block or a marked IVCD widen
    depolarization and inflate the same amplitudes the criteria measure, so a
    positive voltage sum carries no hypertrophy information under those
    conditions.  This runs as an ``invalidator``: only a demonstrated
    invalidating condition blocks the candidate, and an unmeasurable
    precondition never costs a true positive.
    """

    if step_id == "voltage_criteria_pacing_valid":
        pointers = {
            "state": "/rhythm_inputs/pacing/state",
            "ventricular": "/rhythm_inputs/pacing/ventricular_pacing_present",
        }
        values = {
            name: _available_value(store, available, pointer)
            for name, pointer in pointers.items()
        }
        inputs = [pointer for pointer in pointers.values() if pointer in available]
        metrics = {
            "pacing_state": values["state"],
            "ventricular_pacing_present": values["ventricular"],
        }
        state = str(values["state"] or "").strip().lower()
        if values["ventricular"] is True or state == "on":
            return _fact(
                "fail",
                "voltage_criteria_invalid_under_ventricular_pacing",
                [
                    (
                        pointers["ventricular"]
                        if pointers["ventricular"] in available
                        else pointers["state"],
                        "Programmatic confirmation of ventricular pacing, which invalidates QRS voltage criteria in this context",
                    )
                ],
                inputs=inputs,
                metrics=metrics,
            )
        if values["ventricular"] is False or state in {"off", "none"}:
            return _fact(
                "pass",
                "no_ventricular_pacing_detected",
                [(pointers["state"], "Programmatic confirmation that ventricular pacing was not detected")],
                inputs=inputs,
                metrics=metrics,
            )
        return _fact(
            "unknown",
            "pacing_state_unavailable",
            inputs=inputs,
            metrics=metrics,
        )

    dominant = _dominant_group(store, available)
    if dominant is None:
        return _fact("unknown", "dominant_group_unavailable")
    _, row, pointers = dominant
    qrs_ms = _finite(row.get("mean_qrs_ms")) if pointers["qrs"] in available else None
    inputs = [pointers["qrs"]] if pointers["qrs"] in available else []
    metrics = {
        "mean_qrs_ms": qrs_ms,
        "invalid_at_or_above_ms": policy.voltage_criteria_invalid_qrs_ms,
    }
    if qrs_ms is None:
        return _fact(
            "unknown",
            "dominant_qrs_duration_unavailable",
            inputs=inputs,
            metrics=metrics,
        )
    if qrs_ms >= policy.voltage_criteria_invalid_qrs_ms:
        return _fact(
            "fail",
            "voltage_criteria_invalid_under_wide_qrs",
            [(pointers["qrs"], "Programmatic confirmation of dominant QRS widening, which invalidates QRS voltage criteria in this context")],
            inputs=inputs,
            metrics=metrics,
        )
    return _fact(
        "pass",
        "narrow_qrs_keeps_voltage_criteria_valid",
        [(pointers["qrs"], "Programmatic confirmation that dominant QRS duration is not wide enough to invalidate voltage criteria")],
        inputs=inputs,
        metrics=metrics,
    )


def resolve_additional_deterministic_step(
    store: EvidenceStore,
    *,
    code: str,
    step_id: str,
    available_pointers: set[str],
    policy: DeterministicPathwayPolicy = DEFAULT_DETERMINISTIC_PATHWAY_POLICY,
) -> DeterministicFact | None:
    """Resolve program-owned nodes not handled by the legacy inline gates."""

    if step_id in {"pr_criterion", "short_pr_criterion", "pr_component"}:
        pointer = "/global_features/pr_ms"
        pr = _available_number(store, available_pointers, pointer)
        availability_pointer = "/rhythm_inputs/record/availability/pr_available"
        availability = _available_value(store, available_pointers, availability_pointer)
        if pr is None or availability is False:
            return _fact("unknown", "pr_measurement_unavailable", inputs=[p for p in (pointer, availability_pointer) if p in available_pointers])
        evidence = [(pointer, "Programmatic definition-level assessment using the global reportable PR interval")]
        inputs = [pointer] + ([availability_pointer] if availability_pointer in available_pointers else [])
        if step_id == "pr_component":
            return _fact("pass", "pr_component_reportable", evidence, inputs=inputs, metrics={"pr_ms": pr})
        cutoff = 200.0 if step_id == "pr_criterion" else 120.0
        passed = pr > cutoff if step_id == "pr_criterion" else pr < cutoff
        return _fact("pass" if passed else "fail", "pr_above_first_degree_threshold" if step_id == "pr_criterion" and passed else "pr_not_above_first_degree_threshold" if step_id == "pr_criterion" else "pr_below_short_pr_threshold" if passed else "pr_not_below_short_pr_threshold", evidence, inputs=inputs, metrics={"pr_ms": pr, "threshold_ms": cutoff})

    if step_id in {"voltage_criteria_qrs_valid", "voltage_criteria_pacing_valid"}:
        return _voltage_validity_fact(store, available_pointers, step_id, policy)
    if step_id == "one_to_one_av":
        return _one_to_one_av_fact(store, available_pointers, policy)
    if step_id == "repeated_blocked_atrial_events":
        return _repeated_blocked_fact(store, available_pointers, policy)
    if step_id == "sequential_av_pattern":
        return _sequential_av_fact(store, available_pointers, code, policy)
    if step_id in {"sinus_p_support", "p_measurement_reliability"}:
        return _p_reliability_fact(store, available_pointers, absence_question=False, policy=policy)
    if step_id == "organized_p_absent":
        return _p_reliability_fact(store, available_pointers, absence_question=True, policy=policy)
    if step_id == "organized_atrial_activity":
        regular_pointer = "/rhythm_inputs/background/background_rr_regular"
        available_pointer = "/rhythm_inputs/record/availability/atrial_rhythm_available"
        regular = _available_value(store, available_pointers, regular_pointer)
        atrial_available = _available_value(store, available_pointers, available_pointer)
        evidence = [(regular_pointer, "Programmatic read of background RR regularity"), (available_pointer, "Programmatic read of atrial-rhythm measurement availability")]
        inputs = [pointer for pointer, _ in evidence if pointer in available_pointers]
        if regular is True and atrial_available is True:
            return _fact("pass", "regular_background_with_available_atrial_measurements", evidence, inputs=inputs)
        if atrial_available is False:
            return _fact("unknown", "atrial_measurements_unavailable", evidence, inputs=inputs)
        if atrial_available is True:
            # `background_rr_regular` is a whole-strip CV classification: one
            # isolated ectopic beat's coupling interval and compensatory
            # pause can flip it to False even though the surrounding
            # mechanism never wavered. Before giving up, check whether the
            # RR sequence is organized apart from a small, tolerated set of
            # outliers -- that is a real precondition, not a relaxation of
            # it, so it stays a `pass`, not a new normality default.
            robust = _robust_background_rr_regular_fact(store, available_pointers, policy)
            if robust is not None:
                return _fact(
                    "pass",
                    "regular_background_apart_from_isolated_outlier_rr",
                    evidence + list(robust.evidence),
                    inputs=inputs + list(robust.input_pointers),
                    metrics=dict(robust.metrics),
                )
        return _fact("unknown", "organized_atrial_activity_not_established", evidence, inputs=inputs)
    if step_id == "sinus_mechanism_support":
        heart_pointer = "/global_features/heart_rate_bpm"
        atrial_pointer = "/global_features/atrial_rate_bpm"
        axis_pointer = "/global_features/p_axis_deg"
        heart_rate = _available_number(store, available_pointers, heart_pointer)
        atrial_rate = _available_number(store, available_pointers, atrial_pointer)
        p_axis = _available_number(store, available_pointers, axis_pointer)
        evidence = [
            (heart_pointer, "Programmatic read of global ventricular rate"),
            (atrial_pointer, "Programmatic read of directly measured atrial rate"),
            (axis_pointer, "Programmatic read of P-wave axis as component evidence for sinus origin"),
        ]
        inputs = [pointer for pointer, _ in evidence if pointer in available_pointers]
        metrics = {
            "heart_rate_bpm": heart_rate,
            "atrial_rate_bpm": atrial_rate,
            "p_axis_deg": p_axis,
        }
        if heart_rate is None or atrial_rate is None or p_axis is None:
            return _fact(
                "unknown",
                "sinus_mechanism_components_incomplete",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        rate_difference_fraction = abs(heart_rate - atrial_rate) / max(
            heart_rate, atrial_rate, 1.0
        )
        metrics["atrial_ventricular_rate_difference_fraction"] = round(
            rate_difference_fraction, 4
        )
        if (
            policy.sinus_p_axis_lower_deg
            <= p_axis
            <= policy.sinus_p_axis_upper_deg
            and rate_difference_fraction <= policy.sinus_rate_difference_fraction
        ):
            return _fact(
                "pass",
                "compatible_p_axis_and_atrial_ventricular_rates",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        # Atypical P axis or unequal rates can reflect AV block, ectopic atrial
        # activity, measurement error, or mixed mechanisms. It is not a safe
        # definition-level refutation of sinus-node origin.
        return _fact(
            "unknown",
            "sinus_mechanism_not_definition_level_established",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if step_id == "sinus_candidate_stream_reconciled":
        localized_pointer = (
            "/rhythm_inputs/av_block/evidence/localized_atrial_event_excess"
        )
        validation_pointer = (
            "/rhythm_inputs/av_block/evidence/"
            "constant_multiple_atrial_events_requires_validation"
        )
        localized = _available_value(
            store, available_pointers, localized_pointer
        )
        requires_validation = _available_value(
            store, available_pointers, validation_pointer
        )
        evidence = [
            (
                localized_pointer,
                "Programmatic read of whether focal additional atrial candidate events are present",
            ),
            (
                validation_pointer,
                "Programmatic read of whether multiple atrial candidate events still require independent validation",
            ),
        ]
        inputs = [
            pointer for pointer, _ in evidence if pointer in available_pointers
        ]
        metrics = {
            "localized_atrial_event_excess": localized,
            "constant_multiple_events_require_validation": requires_validation,
        }
        if localized is True and requires_validation is True:
            return _fact(
                "unknown",
                "unresolved_atrial_candidate_stream_excess",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        if localized is not None or requires_validation is not None:
            return _fact(
                "pass",
                "no_joint_unresolved_atrial_candidate_stream_flag",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        return _fact(
            "unknown",
            "atrial_candidate_stream_reconciliation_unavailable",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if step_id == "irregular_ventricular_response":
        pointer = "/rhythm_inputs/af_afl/rr_cv"
        rr_cv = _available_number(store, available_pointers, pointer)
        if rr_cv is None:
            return _fact("unknown", "rr_variability_unavailable")
        evidence = [(pointer, "Programmatic assessment of irregularity using the coefficient of variation of the complete RR sequence")]
        if rr_cv >= policy.irregular_rr_pass_cv:
            return _fact("pass", "rr_variability_supports_irregular_response", evidence, inputs=[pointer], metrics={"rr_cv": rr_cv})
        if rr_cv <= policy.regular_rr_fail_cv:
            return _fact("unknown", "rr_response_regular_but_not_an_af_exclusion", evidence, inputs=[pointer], metrics={"rr_cv": rr_cv})
        return _fact("unknown", "rr_variability_borderline", evidence, inputs=[pointer], metrics={"rr_cv": rr_cv})
    if step_id == "multilead_atrial_support":
        consensus_pointer = "/rhythm_inputs/af_afl/f_wave_multilead_consensus"
        confidence_pointer = "/rhythm_inputs/af_afl/f_wave_confidence"
        consensus = _available_value(store, available_pointers, consensus_pointer)
        confidence = _available_number(store, available_pointers, confidence_pointer)
        evidence = [(consensus_pointer, "Programmatic read of multilead consensus for fibrillatory-wave candidates"), (confidence_pointer, "Programmatic read of fibrillatory-wave candidate confidence")]
        inputs = [pointer for pointer, _ in evidence if pointer in available_pointers]
        if consensus is True and confidence is not None and confidence >= 0.50:
            return _fact("pass", "multilead_f_wave_candidate_support", evidence, inputs=inputs, metrics={"confidence": confidence})
        if consensus is False and confidence is not None and confidence <= 0.10:
            return _fact("fail", "no_multilead_f_wave_candidate_support", evidence, inputs=inputs, metrics={"confidence": confidence})
        return _fact("unknown", "f_wave_candidate_support_indeterminate", evidence, inputs=inputs, metrics={"confidence": confidence})
    if step_id == "interval_reportable":
        return _interval_status_fact(store, available_pointers)
    if step_id == "qt_threshold":
        return _qt_threshold_fact(store, available_pointers, code, policy)
    if step_id == "component_endpoint_support":
        # Endpoint support is deterministic only when the record-level interval
        # is already reportable or explicitly non-reportable. Morphologic cause
        # attribution remains outside this node.
        status = _interval_status_fact(store, available_pointers)
        return DeterministicFact(
            status=status.status,
            reason_code=("qt_endpoint_support_sufficient" if status.status == "pass" else "qt_endpoint_support_insufficient" if status.status == "fail" else "qt_endpoint_support_indeterminate"),
            evidence=status.evidence,
            input_pointers=status.input_pointers,
            metrics=status.metrics,
        )
    if step_id == "qrs_duration_support":
        return _qrs_duration_fact(store, available_pointers, code, policy)
    if step_id == "pattern_representative":
        return _representative_group_fact(store, available_pointers, policy)
    if step_id == "premature_timing":
        return _premature_timing_fact(store, available_pointers, policy)
    if step_id == "representative_event":
        return _representative_group_fact(store, available_pointers, policy)
    if step_id == "ectopic_morphology" and code in _PVC_CODES:
        return _pvc_morphology_fact(store, available_pointers, policy)
    if step_id == "wide_complex_sequence":
        return _wide_complex_sequence_fact(store, available_pointers, code, policy)
    if step_id == "territorial_qrs_voltage":
        return _territorial_low_voltage_fact(store, available_pointers, code, policy)
    if step_id == "right_precordial_voltage":
        leads = ("V1", "V2", "V5", "V6")
        ratios: dict[str, float] = {}
        inputs: list[str] = []
        for lead in leads:
            r_pointer = f"/representative_leads/{lead}/params/r_amp_mv"
            s_pointer = f"/representative_leads/{lead}/params/s_amp_mv"
            r_amp = _available_number(store, available_pointers, r_pointer)
            s_amp = _available_number(store, available_pointers, s_pointer)
            if r_amp is None or s_amp is None or abs(s_amp) < 1e-6:
                continue
            ratios[lead] = max(0.0, r_amp) / abs(s_amp)
            inputs.extend((r_pointer, s_pointer))
        metrics = {
            "analyzable_r_s_ratio_leads": len(ratios),
            "r_s_ratios": {lead: round(value, 4) for lead, value in ratios.items()},
        }
        if "V1" not in ratios or not ({"V5", "V6"} & ratios.keys()):
            return _fact(
                "unknown",
                "right_precordial_voltage_components_incomplete",
                inputs=inputs,
                metrics=metrics,
            )
        left_ratios = [ratios[lead] for lead in ("V5", "V6") if lead in ratios]
        evidence = [
            ("/representative_leads/V1/params/r_amp_mv", "Programmatic calculation of the R/S voltage ratio in V1"),
            ("/representative_leads/V1/params/s_amp_mv", "Programmatic calculation of the R/S voltage ratio in V1"),
        ]
        left_lead = "V6" if "V6" in ratios else "V5"
        evidence.extend(
            [
                (f"/representative_leads/{left_lead}/params/r_amp_mv", "Programmatic calculation of the R/S voltage ratio in a left precordial lead"),
                (f"/representative_leads/{left_lead}/params/s_amp_mv", "Programmatic calculation of the R/S voltage ratio in a left precordial lead"),
            ]
        )
        if ratios["V1"] > 1.0 and any(value < 1.0 for value in left_ratios):
            return _fact(
                "pass",
                "right_precordial_r_dominance_with_left_precordial_s_dominance",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        if ratios["V1"] < 0.5 and all(value > 1.0 for value in left_ratios):
            return _fact(
                "fail",
                "left_precordial_voltage_dominance_refutes_rvh_voltage_pattern",
                evidence,
                inputs=inputs,
                metrics=metrics,
            )
        return _fact(
            "unknown",
            "right_precordial_voltage_relation_borderline",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if step_id == "precordial_progression":
        leads = ("V1", "V2", "V3", "V4", "V5", "V6")
        r_values: dict[str, float] = {}
        ratios: dict[str, float] = {}
        inputs: list[str] = []
        for lead in leads:
            r_pointer = f"/representative_leads/{lead}/params/r_amp_mv"
            s_pointer = f"/representative_leads/{lead}/params/s_amp_mv"
            r_amp = _available_number(store, available_pointers, r_pointer)
            s_amp = _available_number(store, available_pointers, s_pointer)
            if r_amp is None or s_amp is None:
                continue
            r_values[lead] = max(0.0, r_amp)
            ratios[lead] = math.inf if abs(s_amp) < 1e-6 else max(0.0, r_amp) / abs(s_amp)
            inputs.extend((r_pointer, s_pointer))
        metrics: dict[str, Any] = {
            "analyzable_precordial_leads": len(ratios),
            "r_s_ratios": {
                lead: (round(value, 4) if math.isfinite(value) else "inf")
                for lead, value in ratios.items()
            },
        }
        if len(ratios) < len(leads):
            return _fact(
                "unknown",
                "incomplete_precordial_progression_lead_set",
                inputs=inputs,
                metrics=metrics,
            )
        transition = next((lead for lead in leads if ratios[lead] >= 1.0), None)
        transition_index = leads.index(transition) if transition is not None else None
        metrics["transition_lead"] = transition
        evidence_leads = [lead for lead in ("V1", "V3", "V4", "V6")]
        evidence = [
            (f"/representative_leads/{lead}/params/r_amp_mv", "Programmatic calculation of R-wave progression and R/S transition from V1 through V6")
            for lead in evidence_leads
        ]
        if code == "precordial_rotation":
            if transition_index is not None and transition_index in {0, 1, 4, 5}:
                return _fact("pass", "precordial_transition_outside_normal_v3_v4_zone", evidence, inputs=inputs, metrics=metrics)
            if transition_index in {2, 3}:
                return _fact("fail", "precordial_transition_in_normal_v3_v4_zone", evidence, inputs=inputs, metrics=metrics)
        elif code in {"clockwise_rotation"}:
            if transition_index is None or transition_index >= 4:
                return _fact("pass", "delayed_precordial_transition", evidence, inputs=inputs, metrics=metrics)
            if transition_index <= 2:
                return _fact("fail", "transition_not_delayed", evidence, inputs=inputs, metrics=metrics)
        elif code == "counterclockwise_rotation":
            if transition_index is not None and transition_index <= 1:
                return _fact("pass", "early_precordial_transition", evidence, inputs=inputs, metrics=metrics)
            if transition_index is None or transition_index >= 4:
                return _fact("fail", "transition_not_early", evidence, inputs=inputs, metrics=metrics)
        elif code == "poor_r_wave_progression":
            v3_r = r_values["V3"]
            if v3_r <= 0.30 and (transition_index is None or transition_index >= 4):
                return _fact("pass", "low_v3_r_with_delayed_transition", evidence, inputs=inputs, metrics={**metrics, "v3_r_mv": round(v3_r, 4)})
            if v3_r > 0.30 and transition_index is not None and transition_index <= 2:
                return _fact("fail", "preserved_r_progression", evidence, inputs=inputs, metrics={**metrics, "v3_r_mv": round(v3_r, 4)})
        elif code == "reversed_r_wave_progression":
            decreases = sum(
                r_values[right] < r_values[left]
                for left, right in zip(leads, leads[1:])
            )
            metrics["adjacent_r_decrease_count"] = decreases
            if r_values["V6"] < r_values["V1"] and decreases >= 3:
                return _fact("pass", "reversed_precordial_r_amplitude_trend", evidence, inputs=inputs, metrics=metrics)
            if r_values["V6"] > r_values["V1"] and decreases <= 1:
                return _fact("fail", "forward_precordial_r_amplitude_trend", evidence, inputs=inputs, metrics=metrics)
        # Dextrocardia and borderline/discordant transition patterns require
        # limb-lead and lead-placement evidence outside this measurement node.
        return _fact(
            "unknown",
            "precordial_progression_pattern_borderline_or_requires_context",
            evidence,
            inputs=inputs,
            metrics=metrics,
        )
    if code in _PACING_CODES and step_id in {"pacing_marker_support", "capture_relation_support"}:
        return _pacing_fact(store, available_pointers, code, step_id, policy)
    return None


__all__ = [
    "DEFAULT_DETERMINISTIC_PATHWAY_POLICY",
    "DETERMINISTIC_NODE_SHADOW_ENV",
    "DETERMINISTIC_PATHWAY_FACTS_VERSION",
    "DeterministicFact",
    "DeterministicPathwayPolicy",
    "deterministic_nodes_shadowed",
    "program_owns_pathway_step",
    "resolve_additional_deterministic_step",
]
