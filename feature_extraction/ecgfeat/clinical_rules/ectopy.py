from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite
from statistics import median

from .models import RuleEvaluation
from .rhythm import has_borderline_af_evidence
from .sources import SOURCE_AHA_RHYTHM


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _rhythm_context(context) -> tuple[bool, bool, bool, bool]:
    rhythm = _mapping(getattr(context.features, "metadata", {}).get("rhythm_analysis"))
    af_afl = _mapping(rhythm.get("af_afl_summary"))
    pacing = _mapping(rhythm.get("pacing_context"))
    confirmed_continuous_ventricular_pacing = bool(
        pacing.get("continuous_pacing")
        and (
            pacing.get("ventricular_pacing_present")
            or pacing.get("dual_chamber_pacing_present")
        )
    )
    unconfirmed_pacing_like_context = bool(
        (
            pacing.get("wide_qrs_pacing_like_context")
            or pacing.get("suppress_further_rhythm_interpretation")
        )
        and not confirmed_continuous_ventricular_pacing
    )
    return (
        bool(
            af_afl.get("probable_af")
            or has_borderline_af_evidence(af_afl)
        ),
        # Flutter remains reference-only in the unified layer. Preserve it as
        # a confidence confounder, but do not make the PAC rule not applicable.
        bool(af_afl.get("probable_flutter")),
        confirmed_continuous_ventricular_pacing,
        unconfirmed_pacing_like_context,
    )


def _beat_facts(context) -> tuple[list[dict], float | None]:
    beats = list(getattr(context.features, "beats", []) or [])
    group_counts = Counter(int(getattr(beat, "group_id", 1)) for beat in beats)
    dominant_group = group_counts.most_common(1)[0][0] if group_counts else None
    features_by_beat: dict[int, list] = defaultdict(list)
    for item in getattr(context.features, "beat_features", []) or []:
        if bool(getattr(item, "beat_measurement_reliable", True)):
            features_by_beat[int(item.beat_id)].append(item)

    rr_values = [
        value
        for value in (_finite(getattr(beat, "rr_prev_ms", None)) for beat in beats)
        if value is not None and value > 0.0
    ]
    background_rr = float(median(rr_values)) if len(rr_values) >= 3 else None
    rows: list[dict] = []
    for beat in beats:
        beat_id = int(beat.beat_id)
        items = features_by_beat.get(beat_id, [])
        qrs_values = [
            value
            for value in (_finite(getattr(item, "qrs_ms", None)) for item in items)
            if value is not None
        ]
        p_values = [
            value
            for value in (_finite(getattr(item, "p_confidence", None)) for item in items)
            if value is not None
        ]
        wide_qrs_lead_count = sum(value >= 120.0 for value in qrs_values)
        rows.append(
            {
                "beat_id": beat_id,
                "rr_prev_ms": _finite(getattr(beat, "rr_prev_ms", None)),
                "rr_next_ms": _finite(getattr(beat, "rr_next_ms", None)),
                "group_id": int(getattr(beat, "group_id", 1)),
                "morphology_outlier": bool(
                    dominant_group is not None
                    and int(getattr(beat, "group_id", 1)) != dominant_group
                ),
                "paced": bool(getattr(beat, "paced", False)),
                "qrs_duration_ms": float(median(qrs_values)) if qrs_values else None,
                "qrs_max_ms": max(qrs_values) if qrs_values else None,
                "wide_qrs_lead_count": wide_qrs_lead_count,
                "p_confidence": max(p_values) if p_values else None,
                "reliable_lead_count": len(items),
            }
        )
    return rows, background_rr


def _evaluation(
    *,
    code: str,
    rule_id: str,
    candidates: list[dict],
    analyzable: int,
    missing: list[str],
    not_applicable_by: str | None = None,
    confounders: list[str] | None = None,
    statement_code: str | None = None,
    confidence: str | None = None,
) -> RuleEvaluation:
    confounders = list(confounders or [])
    if not_applicable_by:
        status = "not_applicable"
    elif candidates:
        status = "matched"
    elif missing:
        status = "unavailable"
    else:
        status = "not_matched"
    count = len(candidates)
    burden = 100.0 * count / analyzable if analyzable else None
    is_ventricular = code == "premature_ventricular_complexes"
    label = "ventricular" if is_ventricular else "atrial"
    severity = (
        "abnormal"
        if status == "matched" and (count >= 3 or (burden or 0.0) >= 10.0)
        else "observation" if status == "matched" else "normal"
    )
    return RuleEvaluation(
        rule_id=rule_id,
        domain="ectopy",
        status=status,
        statement_code=(statement_code or code) if status == "matched" else None,
        statement=(
            f"Premature {label} complexes ({count}/{analyzable} analyzable beats, {burden:.1f}%)"
            if status == "matched" and burden is not None
            else None
        ),
        severity=severity,
        confidence=(
            confidence
            if status == "matched" and confidence is not None
            else "low" if status == "matched" and confounders
            else "moderate" if status == "matched" else None
        ),
        coverage="partial" if status == "matched" and confounders else None,
        required_inputs=["beats.rr_intervals", "beats.qrs_duration_ms", "beats.p_confidence"],
        missing_inputs=missing,
        evidence={
            "evaluates_code": code,
            "candidate_beats": candidates,
            "count": count,
            "analyzable_beats": analyzable,
            "burden_percent": burden,
            "not_applicable_by": not_applicable_by,
            "confounders": confounders,
            "manual_confirmation_required": status == "matched",
        },
        thresholds={
            "rr_shortening_percent": 15.0,
            "wide_qrs_ms": 120.0,
            "p_confidence_min": 0.30,
            "full_compensation_ratio_min": 1.80,
            "full_compensation_ratio_max": 2.20,
            "strict_probable_compensation_ratio_min": 1.75,
            "strict_probable_compensation_ratio_max": 2.30,
        },
        source=dict(SOURCE_AHA_RHYTHM),
        normality_required=False,
        normality_role="optional_screen",
    )


def evaluate_ectopy(context) -> list[RuleEvaluation]:
    (
        probable_af,
        reference_flutter,
        pacing_stop,
        unconfirmed_pacing_like_context,
    ) = _rhythm_context(context)
    rows, background_rr = _beat_facts(context)
    missing = []
    if len(rows) < 3 or background_rr is None:
        missing.append("at_least_three_analyzable_beats_with_rr")

    analyzable_rows = [
        row
        for row in rows
        if not row["paced"]
        and row["rr_prev_ms"] is not None
        and row["qrs_duration_ms"] is not None
        and row["reliable_lead_count"] >= 2
    ]
    if not analyzable_rows and not missing:
        missing.append("reliable_multilead_beat_measurements")

    premature: list[dict] = []
    if background_rr is not None:
        threshold = 0.85 * background_rr
        for row in analyzable_rows:
            if row["rr_prev_ms"] >= threshold:
                continue
            enriched = dict(row)
            rr_sum = None
            if row["rr_next_ms"] is not None:
                rr_sum = row["rr_prev_ms"] + row["rr_next_ms"]
            enriched["premature_threshold_ms"] = threshold
            enriched["compensatory_pause_ratio"] = (
                rr_sum / background_rr
                if rr_sum is not None and background_rr > 0.0
                else None
            )
            enriched["compensatory_pause_support"] = bool(
                rr_sum is not None and 1.8 * background_rr <= rr_sum <= 2.2 * background_rr
            )
            premature.append(enriched)

    nonpremature_qrs = [
        row["qrs_duration_ms"]
        for row in analyzable_rows
        if row not in premature and row["qrs_duration_ms"] is not None
    ]
    background_qrs = float(median(nonpremature_qrs)) if nonpremature_qrs else None
    pvc = [
        row
        for row in premature
        if row["qrs_duration_ms"] >= 120.0
        and (
            row["morphology_outlier"]
            or background_qrs is None
            or row["qrs_duration_ms"] >= background_qrs + 20.0
        )
    ]
    if probable_af or reference_flutter:
        pvc = [
            row
            for row in pvc
            if row["morphology_outlier"] and row["compensatory_pause_support"]
        ]
    probable_pvc = []
    if background_rr is not None and background_qrs is not None:
        for row in premature:
            relative_widening = row["qrs_duration_ms"] - background_qrs
            compensation_ratio = row["compensatory_pause_ratio"]
            strict_relative_candidate = bool(
                row["qrs_duration_ms"] < 120.0
                and row["rr_prev_ms"] <= 0.75 * background_rr
                and row["morphology_outlier"]
                and compensation_ratio is not None
                and 1.75 <= compensation_ratio <= 2.30
                and relative_widening >= 20.0
                and row["wide_qrs_lead_count"] >= 2
            )
            if strict_relative_candidate:
                enriched = dict(row)
                enriched["background_qrs_ms"] = background_qrs
                enriched["relative_qrs_widening_ms"] = relative_widening
                enriched["classification_basis"] = (
                    "strict_relative_widening_with_near_full_compensation"
                )
                probable_pvc.append(enriched)
    pvc_ids = {row["beat_id"] for row in pvc}
    probable_pvc = [
        row for row in probable_pvc if row["beat_id"] not in pvc_ids
    ]
    pvc_candidates = pvc + probable_pvc
    probable_pvc_ids = {row["beat_id"] for row in probable_pvc}
    pac = [
        row
        for row in premature
        if row["beat_id"] not in probable_pvc_ids
        if row["qrs_duration_ms"] < 120.0
        and row["p_confidence"] is not None
        and row["p_confidence"] >= 0.30
    ]
    atrial_not_applicable = (
        "atrial_fibrillation_pattern"
        if probable_af
        else None
    )
    pac_pacing_missing: list[str] = []
    if pacing_stop:
        pac = []
        pvc_candidates = []
        probable_pvc = []
        pacing_missing = ["nonpaced_rhythm_interpretation_available"]
    elif unconfirmed_pacing_like_context:
        # A pacing-like wide-QRS context can create false atrial deflections,
        # so PAC remains unavailable. PVC morphology candidates are retained
        # with a confounder because clearing them caused verified PVC misses.
        pac = []
        pacing_missing = []
        pac_pacing_missing = ["nonpaced_atrial_interpretation_available"]
    else:
        pacing_missing = []
    pacing_confounders = (
        ["unconfirmed_wide_qrs_pacing_like_context"]
        if unconfirmed_pacing_like_context
        else []
    )

    return [
        _evaluation(
            code="premature_atrial_complexes",
            rule_id="CLIN-RHYTHM-PAC-01",
            candidates=pac,
            analyzable=len(analyzable_rows),
            missing=missing + pacing_missing + pac_pacing_missing,
            not_applicable_by=atrial_not_applicable,
            confounders=(
                ["reference_only_atrial_flutter_candidate"]
                if reference_flutter else []
            ),
        ),
        _evaluation(
            code="premature_ventricular_complexes",
            rule_id="CLIN-RHYTHM-PVC-01",
            candidates=pvc_candidates,
            analyzable=len(analyzable_rows),
            missing=missing + pacing_missing,
            confounders=(
                ["reference_only_atrial_flutter_candidate"]
                if reference_flutter else []
            ) + pacing_confounders,
            statement_code=(
                "probable_premature_ventricular_complexes"
                if probable_pvc and not pvc
                else None
            ),
            confidence=(
                "probable_strict_relative_morphology"
                if probable_pvc and not pvc
                else None
            ),
        ),
    ]
