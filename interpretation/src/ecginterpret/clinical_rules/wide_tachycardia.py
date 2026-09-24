from __future__ import annotations

from .models import RuleEvaluation


def evaluate_wide_complex_tachycardia(context) -> RuleEvaluation:
    hr = context.global_value("heart_rate_bpm")
    qrs = context.global_value("qrs_ms")
    rhythm = context.features.metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    rules = rhythm.get("rule_summary", {})
    rules = rules if isinstance(rules, dict) else {}
    af_afl = rhythm.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    interpretation = context.features.interpretation
    ns_vt = bool(getattr(interpretation, "non_sustained_vt", False))
    av_dissociation_raw = bool(rules.get("av_dissociation"))
    av_dissociation_multilead_validated = bool(
        rules.get("validated_multilead_av_dissociation")
    )
    atrial_rhythm_confounders = []
    if bool(af_afl.get("probable_af")):
        atrial_rhythm_confounders.append("probable_atrial_fibrillation")
    if bool(af_afl.get("probable_flutter")):
        atrial_rhythm_confounders.append("probable_atrial_flutter")
    if bool(
        pacing.get("continuous_pacing")
        or pacing.get("ventricular_pacing_present")
        or pacing.get("dual_chamber_pacing_present")
        or pacing.get("wide_qrs_pacing_like_context")
        or pacing.get("suppress_further_rhythm_interpretation")
    ):
        atrial_rhythm_confounders.append("ventricular_pacing_or_pacing_like_context")
    av_dissociation_usable = bool(
        av_dissociation_raw
        and av_dissociation_multilead_validated
        and not atrial_rhythm_confounders
    )
    missing = [
        name
        for name, value in (
            ("global.heart_rate_bpm", hr),
            ("global.qrs_ms", qrs),
        )
        if value is None
    ]
    wide_tachy = bool(
        not missing and hr is not None and qrs is not None
        and hr > 100.0 and qrs >= 120.0
    )
    # PR variability alone cannot establish AV dissociation. Keep the P0
    # wide-complex tachycardia alert, but do not promote it to a definitive
    # VT-pattern code unless an independent ventricular run is present or
    # the atrial/ventricular relation has been explicitly validated across
    # multiple leads.
    vt_supported = wide_tachy and (ns_vt or av_dissociation_usable)
    code = (
        "ventricular_tachycardia_pattern"
        if vt_supported
        else "wide_complex_tachycardia" if wide_tachy else None
    )
    statement = (
        "Ventricular tachycardia pattern"
        if vt_supported
        else (
            "Wide-complex tachycardia; treat as ventricular tachycardia until proven otherwise"
            if wide_tachy
            else None
        )
    )
    return RuleEvaluation(
        rule_id="CLIN-RHYTHM-WCT-01",
        domain="high_risk_patterns",
        status="matched" if wide_tachy else (
            "unavailable" if missing else "not_matched"
        ),
        statement_code=code,
        statement=statement,
        severity="high" if wide_tachy else "normal",
        confidence="high" if vt_supported else "medium" if wide_tachy else None,
        priority="P0" if wide_tachy else None,
        required_inputs=["global.heart_rate_bpm", "global.qrs_ms"],
        missing_inputs=missing,
        evidence={
            "heart_rate_bpm": hr,
            "qrs_ms": qrs,
            "non_sustained_vt": ns_vt,
            "av_dissociation": av_dissociation_usable,
            "av_dissociation_raw": av_dissociation_raw,
            "av_dissociation_multilead_validated": (
                av_dissociation_multilead_validated
            ),
            "av_dissociation_usable": av_dissociation_usable,
            "atrial_rhythm_confounders": atrial_rhythm_confounders,
            "evaluates_code": "wide_complex_tachycardia",
        },
        thresholds={"heart_rate_gt_bpm": 100.0, "qrs_gte_ms": 120.0},
        human_review_required=wide_tachy,
    )
