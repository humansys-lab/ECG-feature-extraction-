from __future__ import annotations

from math import isfinite
from typing import Optional

from ..glasgow_rules.models import GlasgowConfig
from ..glasgow_rules.rate import bradycardia_limit, tachycardia_limit
from .models import RuleEvaluation


RHYTHM_SOURCE = {
    "authority": "AHA/ACC/HRS",
    "document": "ECG standardization and rhythm guidance",
    "section": "Rate and atrial rhythm",
    "version": "public-guideline projection",
}

PEDIATRIC_RATE_SOURCE = {
    "authority": "Glasgow ECG analysis program",
    "document": "Physician's Guide",
    "section": "Chapter 5 rate limits",
    "version": "age-continuous implementation",
}


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def has_borderline_af_evidence(summary) -> bool:
    """Return a conservative probable-AF tier below the definite detector."""
    af = summary if isinstance(summary, dict) else {}
    rr_cv = _finite_float(af.get("rr_cv"))
    organized_p_ratio = _finite_float(af.get("organized_p_ratio"))
    f_wave_confidence = _finite_float(af.get("f_wave_confidence"))
    return bool(
        not af.get("probable_af")
        and not af.get("probable_flutter")
        and not af.get("af_afl_indeterminate")
        and rr_cv is not None
        and rr_cv >= 0.09
        and organized_p_ratio is not None
        and organized_p_ratio < 0.50
        and af.get("f_wave_multilead_consensus")
        and f_wave_confidence is not None
        and f_wave_confidence >= 0.90
    )


def av_dissociation_corroborated(rule_summary) -> bool:
    """Whether an `av_dissociation` flag has support beyond PR scatter.

    `detect_av_block_availability_flags` raises `av_dissociation` from PR
    dispersion alone (range > 80 ms and sd > 30 ms across per-beat medians).
    That is exactly what a P wave associated to the wrong cycle in one lead
    produces, so on its own it is a statement about measurement noise rather
    than about AV conduction, and using it to withhold the PR interval cost 34
    PTB-XL records their PR class with no AF, flutter or pacing anywhere in
    sight.  Real dissociation shows an atrial rate running independently of the
    ventricular one, or presents as complete block; both are computed in the
    same dict and are required here before PR is withheld.
    """
    rs = rule_summary if isinstance(rule_summary, dict) else {}
    return bool(
        rs.get("atrial_faster_than_ventricular") or rs.get("complete_av_block")
    )


def disorganized_atrial_activity(summary, *, threshold: float) -> Optional[float]:
    """Return organized_p_ratio when it is too low to trust P-wave morphology.

    Any diagnosis read off the P wave -- atrial enlargement, PR-derived AV
    delay -- assumes the deflection in front of the QRS really is a sinus P.
    When atrial activity is disorganized that assumption fails: a flutter F
    wave, a fibrillatory wave or a T-wave residual gets measured as "the P",
    which then reads as a broad terminal force (left atrial abnormality) or a
    long PR (first-degree AV delay).

    This is deliberately separate from `probable_flutter` / `probable_af`.
    Those require the flutter/fibrillation call to have been *made*; this fires
    on the weaker and much commoner condition that organized atrial activity
    simply is not there, whether or not the mechanism was identified.

    Returns None when atrial activity is organized enough, or when the ratio
    was not measured at all -- an absent measurement is not evidence of
    disorganization and must not suppress anything.
    """
    af = summary if isinstance(summary, dict) else {}
    ratio = _finite_float(af.get("organized_p_ratio"))
    if ratio is None:
        return None
    return ratio if ratio < threshold else None


def _rate_observation(context) -> RuleEvaluation:
    heart_rate = context.global_value("heart_rate_bpm")
    age = context.age_years
    missing = []
    if heart_rate is None:
        missing.append("global.heart_rate_bpm")
    if age is None:
        missing.append("patient.age_years")
    if missing:
        return RuleEvaluation(
            rule_id="CLIN-RHYTHM-RATE-01",
            domain="rhythm",
            status="unavailable",
            required_inputs=["global.heart_rate_bpm", "patient.age_years"],
            missing_inputs=missing,
            evidence={
                "heart_rate_bpm": heart_rate,
                "age_years": age,
                "evaluates_code": "rate_abnormality",
            },
            source=dict(RHYTHM_SOURCE),
        )

    if age < 18.0:
        age_days = max(0.0, float(age) * 365.25)
        config = GlasgowConfig()
        tachy_limit = tachycardia_limit(age_days, config)
        brady_limit = bradycardia_limit(age_days, config)
        threshold_profile = "pediatric_age_continuous"
        source = PEDIATRIC_RATE_SOURCE
    else:
        tachy_limit = 100.0
        brady_limit = 60.0
        threshold_profile = "adult_observation"
        source = RHYTHM_SOURCE

    if heart_rate > tachy_limit:
        code, statement = "tachycardia", "Tachycardia"
    elif heart_rate < brady_limit:
        code, statement = "bradycardia", "Bradycardia"
    else:
        code = statement = None
    critical_rate = bool(heart_rate < 40.0 or heart_rate > 150.0)
    return RuleEvaluation(
        rule_id="CLIN-RHYTHM-RATE-01",
        domain="rhythm",
        status="matched" if code else "not_matched",
        statement_code=code,
        statement=statement,
        # Rate is an ECG observation.  Without rhythm mechanism or clinical
        # context it must not, by itself, make the whole tracing abnormal.
        severity="high" if critical_rate else "observation" if code else "normal",
        priority="P1" if critical_rate else None,
        human_review_required=critical_rate,
        required_inputs=["global.heart_rate_bpm", "patient.age_years"],
        evidence={
            "heart_rate_bpm": heart_rate,
            "age_years": age,
            "threshold_profile": threshold_profile,
            "evaluates_code": "rate_abnormality",
        },
        thresholds={
            "tachycardia_gt_bpm": tachy_limit,
            "bradycardia_lt_bpm": brady_limit,
        },
        source=dict(source),
    )


def project_rhythm_evidence(context) -> list[RuleEvaluation]:
    analysis = context.features.metadata.get("rhythm_analysis", {})
    analysis = analysis if isinstance(analysis, dict) else {}
    af = analysis.get("af_afl_summary", {})
    af = af if isinstance(af, dict) else {}
    residual = analysis.get("atrial_residual", {})
    residual = residual if isinstance(residual, dict) else {}
    validated = bool(residual.get("validated_qrst_subtraction"))
    probable_af = bool(af.get("probable_af"))
    probable_flutter = bool(af.get("probable_flutter"))
    af_afl_indeterminate = bool(af.get("af_afl_indeterminate"))
    borderline_af = has_borderline_af_evidence(af)
    af_positive = probable_af or borderline_af
    rr_cv = af.get("rr_cv")
    contradictory = probable_af and probable_flutter

    # The AF call itself is the 2014 AHA/ACC/HRS criterion (irregularly
    # irregular RR plus absence of organized atrial activity), already
    # computed as `probable_af`. QRST-template validation is a signal-quality
    # check on the residual used for flutter-wave/fibrillation-wave
    # morphology, not a precondition for the RR/P-wave criterion, so it is
    # exposed as a confidence qualifier rather than a hard gate.
    af_missing: list[str] = []
    if not af:
        af_missing.append("rhythm.af_afl_summary")
    if rr_cv is None:
        af_missing.append("rhythm.rr_cv")
    if contradictory:
        af_missing.append("rhythm.mutually_exclusive_af_afl_evidence")
    if af_afl_indeterminate:
        af_missing.append("rhythm.af_afl_indeterminate")
    af_result = RuleEvaluation(
        rule_id="CLIN-RHYTHM-AF-01",
        domain="rhythm",
        status=(
            "matched"
            if af_positive and not af_missing
            else ("not_matched" if not af_missing else "unavailable")
        ),
        statement_code=(
            (
                "atrial_fibrillation_pattern"
                if probable_af
                else "probable_atrial_fibrillation_pattern"
            )
            if af_positive
            else None
        ),
        statement=(
            (
                "ECG pattern consistent with atrial fibrillation "
                "(irregularly irregular RR without organized atrial activity)"
                if probable_af
                else (
                    "Probable atrial fibrillation pattern with borderline RR "
                    "irregularity, low organized-P support, and multilead "
                    "fibrillatory-wave evidence; rhythm-strip review required"
                )
            )
            if af_positive
            else None
        ),
        severity="abnormal" if af_positive else "normal",
        confidence=(
            None
            if af_missing
            else (
                "borderline_rr_p_f_wave_evidence"
                if borderline_af
                else (
                    "template_validated"
                    if validated
                    else "rr_and_p_wave_evidence"
                )
            )
        ),
        coverage="partial" if borderline_af else None,
        required_inputs=["rhythm.af_afl_summary", "rhythm.rr_cv"],
        missing_inputs=af_missing,
        evidence={
            **af,
            "borderline_probable_af": borderline_af,
            "borderline_af_thresholds": {
                "rr_cv_gte": 0.09,
                "organized_p_ratio_lt": 0.50,
                "f_wave_confidence_gte": 0.90,
                "multilead_f_wave_required": True,
            },
            "validated_qrst_subtraction": validated,
            "evaluates_code": "atrial_fibrillation_pattern",
        },
        source=dict(RHYTHM_SOURCE),
    )
    F_wave_morphology_validated = bool(
        residual.get("F_wave_morphology_validated")
        or (validated and af.get("F_wave_multilead_consensus"))
    )
    validation_missing = [] if validated else ["rhythm.validated_qrst_subtraction"]
    if not F_wave_morphology_validated:
        validation_missing.append("rhythm.multilead_F_wave_morphology")
    if contradictory:
        validation_missing.append("rhythm.mutually_exclusive_af_afl_evidence")
    if af_afl_indeterminate:
        validation_missing.append("rhythm.af_afl_indeterminate")
    F_wave_confidence = _finite_float(af.get("F_wave_confidence"))
    F_wave_rate_bpm = _finite_float(af.get("F_wave_rate_bpm"))
    strong_unvalidated_flutter_candidate = bool(
        probable_flutter
        and not contradictory
        and not af_afl_indeterminate
        and not validated
        and af.get("F_wave_multilead_consensus")
        and F_wave_confidence is not None
        and F_wave_confidence >= 0.80
        and F_wave_rate_bpm is not None
        and 180.0 <= F_wave_rate_bpm <= 400.0
    )
    validated_flutter_candidate = probable_flutter and not validation_missing
    validated_flutter = bool(
        validated_flutter_candidate
        and F_wave_confidence is not None
        and F_wave_confidence >= 0.80
    )
    possible_validated_flutter = bool(
        validated_flutter_candidate and not validated_flutter
    )
    flutter_matched = (
        validated_flutter
        or strong_unvalidated_flutter_candidate
        or possible_validated_flutter
    )
    flutter_result = RuleEvaluation(
        rule_id="CLIN-RHYTHM-AFL-01",
        domain="rhythm",
        status=(
            "matched"
            if flutter_matched
            else ("not_matched" if not validation_missing else "unavailable")
        ),
        statement_code=(
            (
                "atrial_flutter_pattern"
                if validated_flutter
                else (
                    "probable_atrial_flutter_pattern"
                    if strong_unvalidated_flutter_candidate
                    else "possible_atrial_flutter_pattern"
                )
            )
            if flutter_matched
            else None
        ),
        statement=(
            (
                "ECG pattern consistent with atrial flutter "
                "(validated organized multilead F-wave activity)"
                if validated_flutter
                else (
                    (
                        "Probable atrial flutter pattern with strong multilead "
                        "F-wave evidence; QRST-subtraction validation is incomplete "
                        "and rhythm-strip review is required"
                    )
                    if strong_unvalidated_flutter_candidate
                    else (
                        "Possible atrial flutter pattern with validated but "
                        "subthreshold multilead F-wave confidence; rhythm-strip "
                        "review is required"
                    )
                )
            )
            if flutter_matched
            else None
        ),
        severity=(
            "borderline"
            if possible_validated_flutter
            else "abnormal" if flutter_matched else "normal"
        ),
        confidence=(
            (
                "multilead_F_wave_validated"
                if validated_flutter
                else (
                    "strong_multilead_candidate_unvalidated_qrst"
                    if strong_unvalidated_flutter_candidate
                    else "validated_subthreshold_F_wave_review"
                )
            )
            if flutter_matched
            else None
        ),
        coverage="partial" if possible_validated_flutter else None,
        required_inputs=[
            "rhythm.validated_qrst_subtraction",
            "rhythm.multilead_F_wave_morphology",
        ],
        missing_inputs=(
            [] if flutter_matched else sorted(set(validation_missing))
        ),
        evidence={
            **af,
            "validated_qrst_subtraction": validated,
            "F_wave_morphology_validated": F_wave_morphology_validated,
            "definite_F_wave_confidence_min": 0.80,
            "possible_validated_flutter": possible_validated_flutter,
            "strong_unvalidated_flutter_candidate": (
                strong_unvalidated_flutter_candidate
            ),
            "candidate_limitations": (
                ["qrst_subtraction_not_validated"]
                if strong_unvalidated_flutter_candidate
                else ["F_wave_confidence_below_definite_threshold"]
                if possible_validated_flutter
                else []
            ),
            "evaluates_code": "atrial_flutter_pattern",
            "reference_detector_only": False,
        },
        source=dict(RHYTHM_SOURCE),
        normality_required=False,
    )
    indeterminate_result = RuleEvaluation(
        rule_id="CLIN-RHYTHM-AF-AFL-01",
        domain="rhythm",
        status="matched" if af_afl_indeterminate else "not_matched",
        statement_code=(
            "atrial_fibrillation_flutter_indeterminate"
            if af_afl_indeterminate
            else None
        ),
        statement=(
            "Atrial tachyarrhythmia pattern detected; AF versus atrial flutter "
            "is indeterminate and requires rhythm-strip review"
            if af_afl_indeterminate
            else None
        ),
        severity="abnormal" if af_afl_indeterminate else "normal",
        confidence="indeterminate" if af_afl_indeterminate else None,
        required_inputs=[
            "rhythm.af_afl_summary",
            "rhythm.multilead_atrial_activity",
        ],
        evidence={
            **af,
            "validated_qrst_subtraction": validated,
            "evaluates_code": "atrial_fibrillation_flutter_indeterminate",
        },
        source=dict(RHYTHM_SOURCE),
        normality_required=False,
    )
    return [
        af_result,
        flutter_result,
        indeterminate_result,
        _rate_observation(context),
    ]
