"""Atrial activation origin and multiplicity.

``basic_rhythm`` answers one question about the P wave -- is the atrial axis
consistent with a sinus origin.  It does not say what the rhythm *is* when the
answer is no.  Two named non-sinus atrial mechanisms are recognisable from the
P wave alone and are implemented here:

Inverted P waves in the inferior leads mean the atria depolarised from below.
The PR interval then separates the two mechanisms that produce that: a
junctional pacemaker conducts to the atria and ventricles nearly
simultaneously, so PR is short (< 120 ms), whereas a low ectopic atrial focus
still conducts down through the AV node, so PR stays normal.  This is a
duration test on top of a polarity test, and both must be measured -- an
inverted P with an unmeasurable PR names nothing.

Three or more distinct P-wave morphologies in one lead mean three or more
atrial pacemaker sites.  Below 100 bpm that is a multifocal (wandering) atrial
rhythm; at or above 100 bpm it is multifocal atrial tachycardia.  The trap
here is that beat-to-beat measurement noise manufactures "distinct"
morphologies out of one, so a cluster must hold at least two beats before it
counts, and the irregular PR/RR intervals that accompany genuine multifocal
activation are required as corroboration.

Reference: AHA/ACC/HRS 2009 ECG standardization; LITFL ECG Library
"Junctional rhythms", "Multifocal atrial tachycardia".
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import RuleEvaluation
from .sources import SOURCE_AHA_RHYTHM


INFERIOR_LEADS = ("II", "III", "aVF")

# Polarity gate. A P wave shallower than this is not confidently inverted.
INVERTED_P_MAX_MV = -0.05
# aVR is electrically opposite the inferior leads: a low atrial focus makes it
# upright. Corroborating, not required -- aVR is often the noisiest lead.
AVR_UPRIGHT_P_MIN_MV = 0.05
# Junctional vs ectopic atrial split.
JUNCTIONAL_MAX_PR_MS = 120.0
# Below this a "PR interval" is more likely a mis-assigned P wave than a real
# short PR; refuse rather than call a junctional rhythm on it.
MIN_PLAUSIBLE_PR_MS = 60.0

# ── Multifocal atrial activation ─────────────────────────────────────────────
# Two P waves are the same morphology when they agree within both tolerances.
# These are set above realistic beat-to-beat measurement noise (roughly
# +-0.03 mV and +-15 ms) so that noise alone cannot split one focus into two.
P_AMPLITUDE_TOLERANCE_MV = 0.06
P_DURATION_TOLERANCE_MS = 30.0
# A morphology seen once is an outlier or an artefact; seen twice it is a
# focus.
MIN_BEATS_PER_MORPHOLOGY = 2
MIN_DISTINCT_MORPHOLOGIES = 3
MIN_MEASURED_BEATS = 8
# Genuine multifocal activation varies the PR interval; a fixed PR across
# three "morphologies" points at measurement scatter in one morphology.
MIN_PR_RANGE_MS = 40.0
# Irregularly irregular. Same coefficient of variation the electrical-alternans
# screen uses to call a rhythm irregular.
MIN_RR_CV = 0.10
# If one morphology accounts for nearly everything, the rhythm is a dominant
# pacemaker with ectopics, not a multifocal rhythm.
MAX_DOMINANT_FRACTION = 0.75
MAT_RATE_BPM = 100.0
# Multifocal atrial tachycardia is the hardest rhythm to keep separate from
# atrial fibrillation: both are irregularly irregular, both are fast, and
# fibrillatory waves cluster into apparent "morphologies" as readily as real P
# waves do. The `probable_af` flag alone is not enough of a guard because it
# depends on a detector that can miss. What actually separates the two is that
# MAT has a discrete organised P wave before nearly every QRS and AF has none,
# so the organised-P ratio is required directly. The threshold sits below the
# sinus rule's 0.70 because a genuinely multifocal rhythm varies its P waves.
MIN_ORGANIZED_P_RATIO = 0.60


def _inverted_p_origin(context) -> RuleEvaluation:
    facts = {
        lead: {
            "p_amp_mv": context.lead_value(lead, "p_amp_mv", "reliable_for_p"),
            "p_dur_ms": context.lead_value(lead, "p_dur_ms", "reliable_for_p"),
        }
        for lead in (*INFERIOR_LEADS, "aVR")
    }
    measured = [
        lead for lead in INFERIOR_LEADS if facts[lead]["p_amp_mv"] is not None
    ]
    inverted = [
        lead
        for lead in INFERIOR_LEADS
        if facts[lead]["p_amp_mv"] is not None
        and facts[lead]["p_amp_mv"] <= INVERTED_P_MAX_MV
    ]
    avr_amp = facts["aVR"]["p_amp_mv"]
    avr_upright = bool(avr_amp is not None and avr_amp >= AVR_UPRIGHT_P_MIN_MV)

    pr_ms = context.global_value("pr_ms")
    if pr_ms is None:
        pr_ms = context.global_value("pr_consensus_ms")

    confounders: list[str] = []
    if getattr(context, "limb_reversal", False):
        # Limb-lead reversal inverts the inferior leads without any change of
        # atrial origin; this is the classic false positive.
        confounders.append("limb_lead_reversal")
    rhythm = getattr(context.features, "metadata", {}).get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    af_afl = rhythm.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    if af_afl.get("probable_af"):
        confounders.append("probable_atrial_fibrillation")
    if af_afl.get("probable_flutter"):
        confounders.append("probable_atrial_flutter")
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    if pacing.get("atrial_pacing_present") or pacing.get("continuous_pacing"):
        confounders.append("atrial_pacing")

    # Both the polarity and the interval must be in hand: polarity alone says
    # "not sinus" but cannot name the mechanism.
    available = len(measured) >= 2 and pr_ms is not None
    polarity_matched = len(measured) >= 2 and len(inverted) >= 2

    code: Optional[str] = None
    statement: Optional[str] = None
    if polarity_matched and pr_ms is not None:
        if pr_ms < MIN_PLAUSIBLE_PR_MS:
            code = None
        elif pr_ms < JUNCTIONAL_MAX_PR_MS:
            code = "junctional_rhythm_pattern"
            statement = (
                "Inverted inferior P waves with short PR interval; "
                "junctional (AV-nodal) origin"
            )
        else:
            code = "ectopic_atrial_rhythm_pattern"
            statement = (
                "Inverted inferior P waves with normal PR interval; "
                "low ectopic atrial origin"
            )
    matched = code is not None

    if not available:
        status = "unavailable"
    elif matched:
        status = "matched"
    elif polarity_matched:
        # Inverted P waves with an implausibly short PR: the polarity finding
        # is real but the mechanism is not resolvable.
        status = "indeterminate"
    else:
        status = "not_matched"

    missing: list[str] = []
    if len(measured) < 2:
        missing.append("two_reliable_inferior_lead_p_amplitudes")
    if pr_ms is None:
        missing.append("global.pr_ms")

    return RuleEvaluation(
        rule_id="CLIN-RHYTHM-P-ORIGIN-01",
        domain="rhythm",
        status=status,
        statement_code=code if matched else None,
        statement=statement if matched else None,
        severity="abnormal" if matched else "normal",
        confidence="moderate" if matched else None,
        priority="P3" if matched else None,
        required_inputs=[
            "lead.p_amp_mv",
            "lead.reliable_for_p",
            "global.pr_ms",
        ],
        missing_inputs=missing,
        evidence={
            "evaluates_code": code or "junctional_rhythm_pattern",
            "lead_facts": facts,
            "measured_inferior_leads": measured,
            "inverted_inferior_leads": inverted,
            "avr_p_upright": avr_upright,
            "pr_ms": pr_ms,
            "confounders": confounders,
        },
        thresholds={
            "inverted_p_le_mv": INVERTED_P_MAX_MV,
            "junctional_pr_lt_ms": JUNCTIONAL_MAX_PR_MS,
            "minimum_plausible_pr_ms": MIN_PLAUSIBLE_PR_MS,
            "avr_upright_ge_mv": AVR_UPRIGHT_P_MIN_MV,
            "minimum_inverted_leads": 2,
        },
        source=dict(SOURCE_AHA_RHYTHM),
        normality_required=False,
        # Supporting, not core: the sinus-mechanism rule already answers the
        # rhythm domain's normality question. An unmeasurable P wave here must
        # not make the whole rhythm domain read as unevaluated.
        normality_role="supporting",
        suppressed_by=confounders if matched else [],
    )


def _cluster_p_morphologies(
    samples: list[dict],
) -> list[list[dict]]:
    """Greedy single-pass clustering on (amplitude, duration).

    A beat joins the first cluster whose running centroid it matches within
    both tolerances, otherwise it seeds a new cluster. Greedy is adequate
    here because the decision downstream is only "how many clusters hold at
    least two beats", not which beat belongs where.
    """
    clusters: list[list[dict]] = []
    for sample in samples:
        placed = False
        for cluster in clusters:
            amp_centre = float(np.mean([item["amp"] for item in cluster]))
            dur_centre = float(np.mean([item["dur"] for item in cluster]))
            if (
                abs(sample["amp"] - amp_centre) <= P_AMPLITUDE_TOLERANCE_MV
                and abs(sample["dur"] - dur_centre) <= P_DURATION_TOLERANCE_MS
            ):
                cluster.append(sample)
                placed = True
                break
        if not placed:
            clusters.append([sample])
    return clusters


def _multifocal_atrial_rhythm(context) -> RuleEvaluation:
    features = getattr(context, "features", None)
    beats = list(getattr(features, "beats", []) or [])
    beat_by_id = {int(getattr(beat, "beat_id", -1)): beat for beat in beats}

    excluded: list[str] = []
    if any(bool(getattr(beat, "paced", False)) for beat in beats):
        excluded.append("paced_beats_present")
    metadata = getattr(features, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    rhythm = metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    af_afl = rhythm.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    # Fibrillatory and flutter waves have no discrete P morphology to count,
    # and clustering their baseline undulation would invent focus counts.
    if af_afl.get("probable_af"):
        excluded.append("probable_atrial_fibrillation")
    if af_afl.get("probable_flutter"):
        excluded.append("probable_atrial_flutter")
    try:
        organized_p = af_afl.get("organized_p_ratio")
        organized_p = float(organized_p) if organized_p is not None else None
    except (TypeError, ValueError):
        organized_p = None

    # Lead II carries the largest, most reliably delineated P wave.
    samples: list[dict] = []
    for feature in list(getattr(features, "beat_features", []) or []):
        if getattr(feature, "lead", None) != "II":
            continue
        beat = beat_by_id.get(int(getattr(feature, "beat_id", -1)))
        if beat is not None and bool(getattr(beat, "paced", False)):
            continue
        if not bool(getattr(feature, "beat_measurement_reliable", False)):
            continue
        amp = getattr(feature, "p_amp_mv", None)
        dur = getattr(feature, "p_dur_ms", None)
        if amp is None or dur is None:
            continue
        try:
            amp_value = float(amp)
            dur_value = float(dur)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(amp_value) and np.isfinite(dur_value)):
            continue
        pr = getattr(feature, "pr_ms", None)
        try:
            pr_value = float(pr) if pr is not None else None
        except (TypeError, ValueError):
            pr_value = None
        if pr_value is not None and not np.isfinite(pr_value):
            pr_value = None
        samples.append(
            {
                "beat_id": int(getattr(feature, "beat_id", -1)),
                "amp": amp_value,
                "dur": dur_value,
                "pr": pr_value,
            }
        )
    samples.sort(key=lambda item: item["beat_id"])

    clusters = _cluster_p_morphologies(samples)
    qualifying = [
        cluster for cluster in clusters if len(cluster) >= MIN_BEATS_PER_MORPHOLOGY
    ]
    dominant_fraction = (
        max((len(cluster) for cluster in clusters), default=0) / len(samples)
        if samples
        else None
    )

    pr_values = [item["pr"] for item in samples if item["pr"] is not None]
    pr_range = (
        float(max(pr_values) - min(pr_values)) if len(pr_values) >= 3 else None
    )

    rr_values = [
        float(beat.rr_prev_ms)
        for beat in beats
        if getattr(beat, "rr_prev_ms", None) is not None
        and float(beat.rr_prev_ms) > 0.0
    ]
    rr_cv = (
        float(np.std(rr_values) / np.mean(rr_values))
        if len(rr_values) >= 3 and float(np.mean(rr_values)) > 0.0
        else None
    )

    heart_rate = context.global_value("heart_rate_bpm")

    available = bool(
        len(samples) >= MIN_MEASURED_BEATS
        and pr_range is not None
        and rr_cv is not None
        and heart_rate is not None
        and organized_p is not None
    )
    criteria = {
        "distinct_morphologies": len(qualifying) >= MIN_DISTINCT_MORPHOLOGIES,
        "varying_pr": pr_range is not None and pr_range >= MIN_PR_RANGE_MS,
        "irregular_rr": rr_cv is not None and rr_cv >= MIN_RR_CV,
        "no_dominant_morphology": (
            dominant_fraction is not None
            and dominant_fraction <= MAX_DOMINANT_FRACTION
        ),
        "organized_p_waves": (
            organized_p is not None and organized_p >= MIN_ORGANIZED_P_RATIO
        ),
    }
    # Kept separate from ``matched`` so the confounded case still carries its
    # reasons: RuleEvaluation.__post_init__ turns matched + suppressed_by into
    # a suppressed row, which is how the exclusion reaches the report.
    criteria_met = bool(available and all(criteria.values()))
    matched = bool(criteria_met and not excluded)

    tachycardic = heart_rate is not None and heart_rate >= MAT_RATE_BPM
    code = (
        "multifocal_atrial_tachycardia" if tachycardic else "multifocal_atrial_rhythm"
    )
    statement = (
        "Multifocal atrial tachycardia; at least three P-wave morphologies "
        "with varying PR and irregular ventricular response"
        if tachycardic
        else "Multifocal (wandering) atrial rhythm; at least three P-wave "
        "morphologies with varying PR at a non-tachycardic rate"
    )

    return RuleEvaluation(
        rule_id="CLIN-RHYTHM-MULTIFOCAL-01",
        domain="rhythm",
        status=(
            "matched"
            if criteria_met
            else "not_matched"
            if available
            else "unavailable"
        ),
        statement_code=code if criteria_met else None,
        statement=statement if criteria_met else None,
        severity="abnormal" if criteria_met else "normal",
        confidence="low" if criteria_met else None,
        priority="P3" if criteria_met else None,
        required_inputs=[
            "beat_features.II.p_amp_mv",
            "beat_features.II.p_dur_ms",
            "beat_features.II.pr_ms",
            "global.heart_rate_bpm",
            "rhythm.organized_p_ratio",
        ],
        missing_inputs=(
            []
            if available
            else [
                "eight_beats_with_reliable_lead_II_p_measurements_and_organized_p_ratio"
            ]
        ),
        evidence={
            "evaluates_code": code,
            "analysis_lead": "II",
            "measured_beats": len(samples),
            "morphology_cluster_sizes": sorted(
                (len(cluster) for cluster in clusters), reverse=True
            ),
            "qualifying_morphology_count": len(qualifying),
            "dominant_morphology_fraction": dominant_fraction,
            "pr_range_ms": pr_range,
            "rr_cv": rr_cv,
            "organized_p_ratio": organized_p,
            "heart_rate_bpm": heart_rate,
            "criteria": criteria,
            "excluded_reasons": sorted(set(excluded)),
            "manual_confirmation_required": matched,
        },
        thresholds={
            "p_amplitude_tolerance_mv": P_AMPLITUDE_TOLERANCE_MV,
            "p_duration_tolerance_ms": P_DURATION_TOLERANCE_MS,
            "minimum_beats_per_morphology": MIN_BEATS_PER_MORPHOLOGY,
            "minimum_distinct_morphologies": MIN_DISTINCT_MORPHOLOGIES,
            "minimum_measured_beats": MIN_MEASURED_BEATS,
            "minimum_pr_range_ms": MIN_PR_RANGE_MS,
            "minimum_rr_cv": MIN_RR_CV,
            "minimum_organized_p_ratio": MIN_ORGANIZED_P_RATIO,
            "maximum_dominant_fraction": MAX_DOMINANT_FRACTION,
            "tachycardia_ge_bpm": MAT_RATE_BPM,
        },
        source=dict(SOURCE_AHA_RHYTHM),
        normality_required=False,
        # Supporting, not core: the sinus-mechanism rule already answers the
        # rhythm domain's normality question. An unmeasurable P wave here must
        # not make the whole rhythm domain read as unevaluated.
        normality_role="supporting",
        human_review_required=matched,
        suppressed_by=sorted(set(excluded)) if criteria_met else [],
    )


def evaluate_atrial_rhythm(context) -> list[RuleEvaluation]:
    return [_inverted_p_origin(context), _multifocal_atrial_rhythm(context)]
