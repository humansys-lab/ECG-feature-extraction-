from __future__ import annotations

from typing import Iterable, Optional

from .config import DEFAULT_DIAGNOSTIC_CONFIG
from .models import RuleEvaluation
from .sources import SOURCE_UDMI_2018


_ISCHEMIA = DEFAULT_DIAGNOSTIC_CONFIG.ischemia

STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONTIGUOUS_PAIRS = (
    ("I", "aVL"),
    ("V1", "V2"),
    ("V2", "V3"),
    ("V3", "V4"),
    ("V4", "V5"),
    ("V5", "V6"),
    ("III", "aVF"),
    ("II", "aVF"),
)
ST_DEPRESSION_PAIRS = (
    # V1-V3 depression is assessed by the dedicated posterior-ischaemia
    # screen, which also requires supporting T/R morphology.
    ("I", "aVL"),
    ("V3", "V4"),
    ("V4", "V5"),
    ("V5", "V6"),
    ("III", "aVF"),
    ("II", "aVF"),
)
Q_GROUPS = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anterior": ("V1", "V2", "V3", "V4"),
}


def _result(
    *,
    rule_id: str,
    domain: str,
    code: str,
    status: str,
    evidence: dict,
    statement: Optional[str] = None,
    severity: str = "abnormal",
    missing_inputs: Optional[Iterable[str]] = None,
    suppressed_by: Optional[Iterable[str]] = None,
    normality_required: bool = True,
    coverage: Optional[str] = None,
    confidence: Optional[str] = None,
    priority: Optional[str] = None,
    human_review_required: bool = False,
) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=rule_id,
        domain=domain,
        status=status,
        statement_code=code if status == "matched" else None,
        statement=statement if status == "matched" else None,
        severity=severity if status == "matched" else "normal",
        coverage=coverage,
        confidence=confidence,
        missing_inputs=list(missing_inputs or []),
        suppressed_by=list(suppressed_by or []),
        evidence={**evidence, "evaluates_code": code},
        source=dict(SOURCE_UDMI_2018),
        normality_required=normality_required,
        priority=priority,
        human_review_required=human_review_required,
    )


def _st_threshold(context, lead: str) -> Optional[float]:
    if lead not in {"V2", "V3"}:
        return 0.10
    if context.sex in {"female", "f"}:
        return 0.15
    if context.sex in {"male", "m"}:
        if context.age_years is None:
            return None
        return 0.25 if context.age_years < 40.0 else 0.20
    return None


def _st_value(context, lead: str) -> Optional[float]:
    """Return ST-J only when its measurement-specific guard did not reject it."""
    st_reliable = context.lead_raw_value(
        lead, "st_j_reliable", "reliable_for_qrs"
    )
    if st_reliable is False:
        return None
    return context.lead_value(lead, "st_on_mv", "reliable_for_qrs")


def _lead_variance_value(
    context,
    lead: str,
    name: str,
    reliability: str,
) -> Optional[float]:
    getter = getattr(context, "lead_variance_value", None)
    if callable(getter):
        return getter(lead, name, reliability)
    value = context.lead_raw_value(lead, name, reliability)
    return float(value) if isinstance(value, (int, float)) else None


def _matched_conduction_codes(
    conduction: Iterable[RuleEvaluation],
) -> set[str]:
    return {
        str(item.evidence.get("evaluates_code") or item.statement_code)
        for item in conduction
        if item.status == "matched"
        and (item.evidence.get("evaluates_code") or item.statement_code)
    }


def _pacing_present(context) -> bool:
    features = getattr(context, "features", None)
    metadata = getattr(features, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    rhythm = metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    pacing = rhythm.get("pacing_context", {})
    pacing = pacing if isinstance(pacing, dict) else {}
    failures = rhythm.get("pacing_failures", {})
    failures = failures if isinstance(failures, dict) else {}

    try:
        spike_count = int(
            failures.get("spike_count")
            or pacing.get("spike_count")
            or 0
        )
    except (TypeError, ValueError):
        spike_count = 0
    try:
        paced_fraction = float(
            metadata.get("paced_beat_fraction")
            if metadata.get("paced_beat_fraction") is not None
            else pacing.get("paced_fraction", 0.0)
        )
    except (TypeError, ValueError):
        paced_fraction = 0.0
    alignment_fraction = failures.get("capture_alignment_fraction")
    if alignment_fraction is None:
        failure_fraction = failures.get("capture_failure_fraction")
        try:
            alignment_fraction = (
                max(0.0, 1.0 - float(failure_fraction))
                if failure_fraction is not None
                else None
            )
        except (TypeError, ValueError):
            alignment_fraction = None
    try:
        capture_aligned = (
            alignment_fraction is not None
            and float(alignment_fraction) >= 0.60
        )
    except (TypeError, ValueError):
        capture_aligned = False

    return bool(
        str(metadata.get("measurement_pacing_state") or "off") == "on"
        and spike_count >= 3
        and paced_fraction >= 0.20
        and bool(pacing.get("ventricular_pacing_present"))
        and capture_aligned
    )


def _rhythm_template_confounders(context) -> list[str]:
    features = getattr(context, "features", None)
    metadata = getattr(features, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    rhythm = metadata.get("rhythm_analysis", {})
    rhythm = rhythm if isinstance(rhythm, dict) else {}
    af_afl = rhythm.get("af_afl_summary", {})
    af_afl = af_afl if isinstance(af_afl, dict) else {}
    gate = metadata.get("diagnostic_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    confounders = []
    if bool(af_afl.get("probable_af")):
        confounders.append("atrial_fibrillation")
    if bool(af_afl.get("probable_flutter")):
        confounders.append("atrial_flutter")
    if bool(af_afl.get("af_afl_indeterminate")):
        confounders.append("af_afl_indeterminate")
    global_value = getattr(context, "global_value", None)
    heart_rate_bpm = (
        global_value("heart_rate_bpm") if callable(global_value) else None
    )
    if heart_rate_bpm is not None and float(heart_rate_bpm) > 100.0:
        confounders.append("heart_rate_gt_100_bpm")
    if bool(gate.get("morphology_complex")):
        confounders.append("complex_beat_morphology")
    return confounders


def _st_depression_facts(context, lead: str) -> dict:
    st_j = _st_value(context, lead)
    st_mid = context.lead_value(lead, "st_mid_mv", "reliable_for_qrs")
    st_80 = context.lead_value(lead, "st_80ms_mv", "reliable_for_qrs")
    morphology = context.lead_raw_value(
        lead, "st_morphology", "reliable_for_qrs"
    )
    baseline_confidence = context.lead_value(
        lead, "st_hybrid_baseline_confidence", "reliable_for_qrs"
    )
    hybrid_reliable = context.lead_raw_value(
        lead, "st_hybrid_reliable", "reliable_for_qrs"
    )
    morphology_text = str(morphology or "").strip().lower()
    baseline_reliable = (
        hybrid_reliable is not False
        and (
            baseline_confidence is None
            or baseline_confidence >= 0.60
        )
    )
    measurements_available = all(
        value is not None for value in (st_j, st_mid, st_80)
    ) and bool(morphology_text)
    morphology_supported = morphology_text in {"horizontal", "downsloping"}
    borderline_persistent = bool(
        measurements_available
        and baseline_reliable
        and st_j <= -0.05
        and st_mid <= -0.05
        and st_80 <= -0.05
        and morphology_supported
    )
    persistent = bool(
        measurements_available
        and baseline_reliable
        and st_j <= -0.075
        and st_mid <= -0.075
        and st_80 <= -0.075
        and morphology_supported
    )
    delayed_persistent = bool(
        measurements_available
        and baseline_reliable
        and st_j <= 0.05
        and st_mid <= -0.075
        and st_80 <= -0.075
        and morphology_supported
    )
    return {
        "st_j_mv": st_j,
        "st_mid_mv": st_mid,
        "st_80ms_mv": st_80,
        "st_morphology": morphology,
        "st_pattern_class": context.lead_raw_value(
            lead, "st_pattern_class", "reliable_for_qrs"
        ),
        "st_hybrid_baseline_confidence": baseline_confidence,
        "st_hybrid_reliable": hybrid_reliable,
        "baseline_reliable": baseline_reliable,
        "measurements_available": measurements_available,
        "borderline_persistent_st_depression": borderline_persistent,
        "persistent_horizontal_or_downsloping_depression": persistent,
        "delayed_persistent_st_depression": delayed_persistent,
    }


def _acute_occlusion_pattern(
    context, conduction: Iterable[RuleEvaluation]
) -> RuleEvaluation:
    facts = {}
    for lead in STANDARD_LEADS:
        st_j = _st_value(context, lead)
        st_mid = context.lead_value(
            lead, "st_mid_mv", "reliable_for_qrs"
        )
        st_80 = context.lead_value(
            lead, "st_80ms_mv", "reliable_for_qrs"
        )
        baseline_confidence = context.lead_value(
            lead, "st_hybrid_baseline_confidence", "reliable_for_qrs"
        )
        hybrid_reliable = context.lead_raw_value(
            lead, "st_hybrid_reliable", "reliable_for_qrs"
        )
        facts[lead] = {
            "st_j_mv": st_j,
            "st_mid_mv": st_mid,
            "st_80ms_mv": st_80,
            "st_hybrid_baseline_confidence": baseline_confidence,
            "st_hybrid_reliable": hybrid_reliable,
            "baseline_reliable": bool(
                hybrid_reliable is not False
                and (
                    baseline_confidence is None
                    or baseline_confidence >= 0.60
                )
            ),
        }
    values = {lead: item["st_j_mv"] for lead, item in facts.items()}
    j_point_pairs = []
    qualifying_pairs = []
    threshold_unknown_leads = []
    for first, second in CONTIGUOUS_PAIRS:
        a, b = values.get(first), values.get(second)
        first_threshold = _st_threshold(context, first)
        second_threshold = _st_threshold(context, second)
        if a is not None and first_threshold is None:
            threshold_unknown_leads.append(first)
        if b is not None and second_threshold is None:
            threshold_unknown_leads.append(second)
        if (
            a is not None
            and b is not None
            and first_threshold is not None
            and second_threshold is not None
            and a >= first_threshold
            and b >= second_threshold
        ):
            pair = [first, second]
            j_point_pairs.append(pair)
            if all(
                facts[lead]["baseline_reliable"]
                and facts[lead]["st_mid_mv"] is not None
                and facts[lead]["st_mid_mv"] >= 0.05
                and facts[lead]["st_80ms_mv"] is not None
                and facts[lead]["st_80ms_mv"] >= 0.05
                for lead in pair
            ):
                qualifying_pairs.append(pair)
    measured = [lead for lead, value in values.items() if value is not None]
    missing = [] if len(measured) >= 2 else ["two_reliable_contiguous_st_leads"]
    if (
        not qualifying_pairs
        and threshold_unknown_leads
        and any(values.get(lead, -999.0) >= 0.15 for lead in threshold_unknown_leads)
    ):
        missing.append("patient.sex_and_age_for_V2_V3_st_threshold")
    qrs_ms = context.global_value("qrs_ms")
    confounders = []
    if qrs_ms is not None and float(qrs_ms) >= 120.0:
        confounders.append("qrs_duration_ge_120_ms")
    if _pacing_present(context):
        confounders.append("ventricular_pacing")
    conduction_codes = _matched_conduction_codes(conduction)
    if "lbbb_pattern" in conduction_codes:
        confounders.append("lbbb_pattern")
    rhythm_confounders = _rhythm_template_confounders(context)
    j_point_only_pairs = [
        pair for pair in j_point_pairs if pair not in qualifying_pairs
    ]
    if j_point_pairs and confounders:
        status = "suppressed"
    elif qualifying_pairs and rhythm_confounders:
        status = "indeterminate"
    else:
        status = (
            "matched"
            if qualifying_pairs
            else "indeterminate" if j_point_pairs
            else "unavailable" if missing else "not_matched"
        )
    return _result(
        rule_id="CLIN-ISCHEMIA-STE-01",
        domain="ischemia_infarction",
        code="acute_occlusion_pattern",
        status=status,
        evidence={
            "st_j_mv": values,
            "lead_facts": facts,
            "j_point_qualifying_pairs": j_point_pairs,
            "qualifying_contiguous_pairs": qualifying_pairs,
            "j_point_only_pairs": j_point_only_pairs,
            "sex": context.sex,
            "age_years": context.age_years,
            "qrs_ms": qrs_ms,
            "conduction_confounders": sorted(set(confounders)),
            "rhythm_confounders": rhythm_confounders,
            "required_persistent_st_levels_mv": {
                "st_mid_min": 0.05,
                "st_80ms_min": 0.05,
            },
        },
        missing_inputs=missing,
        suppressed_by=sorted(set(confounders)) if status == "suppressed" else [],
        statement="ECG pattern suggestive of acute ischemia or coronary occlusion; correlate urgently with symptoms, serial ECGs and biomarkers",
        coverage="partial" if status == "indeterminate" else "full",
        confidence=(
            "moderate" if status == "matched"
            else "low" if status == "indeterminate"
            else None
        ),
        priority="P1",
        human_review_required=bool(qualifying_pairs),
    )


def _st_depression_pattern(
    context, conduction: Iterable[RuleEvaluation]
) -> RuleEvaluation:
    facts = {
        lead: _st_depression_facts(context, lead)
        for lead in STANDARD_LEADS
    }
    qualifying_pairs = [
        [first, second]
        for first, second in ST_DEPRESSION_PAIRS
        if facts[first]["persistent_horizontal_or_downsloping_depression"]
        and facts[second]["persistent_horizontal_or_downsloping_depression"]
    ]
    borderline_pairs = [
        [first, second]
        for first, second in ST_DEPRESSION_PAIRS
        if facts[first]["borderline_persistent_st_depression"]
        and facts[second]["borderline_persistent_st_depression"]
        and [first, second] not in qualifying_pairs
    ]
    delayed_persistent_leads = [
        lead
        for lead in ("I", "II", "III", "aVL", "aVF", "V3", "V4", "V5", "V6")
        if facts[lead]["delayed_persistent_st_depression"]
    ]
    delayed_limb = any(
        lead in {"I", "II", "III", "aVL", "aVF"}
        for lead in delayed_persistent_leads
    )
    delayed_precordial = any(
        lead in {"V3", "V4", "V5", "V6"}
        for lead in delayed_persistent_leads
    )
    diffuse_delayed_pattern = bool(
        len(delayed_persistent_leads) >= 4
        and delayed_limb
        and delayed_precordial
    )
    available_pairs = [
        pair for pair in ST_DEPRESSION_PAIRS
        if facts[pair[0]]["measurements_available"]
        and facts[pair[1]]["measurements_available"]
    ]
    codes = _matched_conduction_codes(conduction)
    confounders = sorted(
        codes
        & {
            "lbbb_pattern",
            "nonspecific_ivcd",
            "preexcitation_pattern",
        }
    )
    if _pacing_present(context):
        confounders.append("ventricular_pacing")
    if (qualifying_pairs or diffuse_delayed_pattern) and confounders:
        status = "suppressed"
    elif qualifying_pairs:
        status = "matched"
    elif borderline_pairs or diffuse_delayed_pattern:
        status = "indeterminate"
    elif available_pairs:
        status = "not_matched"
    else:
        status = "unavailable"
    return _result(
        rule_id="CLIN-ISCHEMIA-STD-01",
        domain="ischemia_infarction",
        code="st_depression",
        status=status,
        evidence={
            "lead_facts": facts,
            "qualifying_contiguous_pairs": qualifying_pairs,
            "borderline_contiguous_pairs": borderline_pairs,
            "delayed_persistent_leads": delayed_persistent_leads,
            "diffuse_delayed_pattern": diffuse_delayed_pattern,
            "required_shape": ["horizontal", "downsloping"],
            "required_levels_mv": {
                "borderline_st_j_max": -0.05,
                "borderline_st_mid_max": -0.05,
                "borderline_st_80ms_max": -0.05,
                "matched_st_j_max": -0.075,
                "matched_st_mid_max": -0.075,
                "matched_st_80ms_max": -0.075,
                "delayed_st_j_max": 0.05,
                "delayed_st_mid_max": -0.075,
                "delayed_st_80ms_max": -0.075,
            },
        },
        missing_inputs=[] if available_pairs else ["reliable_contiguous_st_pair"],
        suppressed_by=confounders if status == "suppressed" else [],
        statement="Contiguous ST-segment depression; correlate for ischemia and secondary causes",
        coverage="partial" if status == "indeterminate" else "full",
        confidence=(
            "moderate" if status == "matched"
            else "low" if status == "indeterminate"
            else None
        ),
        priority="P2",
        human_review_required=bool(qualifying_pairs),
    )


def _acute_high_risk_templates(
    context, conduction: Iterable[RuleEvaluation]
) -> list[RuleEvaluation]:
    anterior = {}
    for lead in ("V2", "V3", "V4"):
        baseline_confidence = context.lead_value(
            lead, "st_hybrid_baseline_confidence", "reliable_for_qrs"
        )
        hybrid_reliable = context.lead_raw_value(
            lead, "st_hybrid_reliable", "reliable_for_qrs"
        )
        qrs_components = [
            value
            for value in (
                context.lead_value(lead, "q_amp_mv", "reliable_for_qrs"),
                context.lead_value(lead, "r_amp_mv", "reliable_for_qrs"),
                context.lead_value(lead, "s_amp_mv", "reliable_for_qrs"),
            )
            if value is not None
        ]
        qrs_peak_amplitude = (
            max(abs(value) for value in qrs_components)
            if qrs_components
            else None
        )
        t_amp = context.lead_value(lead, "t_amp_mv", "reliable_for_t")
        st_j_sd = _lead_variance_value(
            context, lead, "st_on_mv_sd", "reliable_for_qrs"
        )
        t_amp_sd = _lead_variance_value(
            context, lead, "t_amp_mv_sd", "reliable_for_t"
        )
        beat_support = context.lead_value(
            lead, "st_hybrid_beat_support", "reliable_for_qrs"
        )
        t_to_qrs_ratio = (
            abs(t_amp) / qrs_peak_amplitude
            if t_amp is not None
            and qrs_peak_amplitude is not None
            and qrs_peak_amplitude > 0.0
            else None
        )
        beat_consistent = bool(
            beat_support is not None
            and beat_support >= 3
            and st_j_sd is not None
            and st_j_sd <= 0.08
            and t_amp_sd is not None
            and t_amp is not None
            and t_amp_sd <= max(0.15, 0.35 * abs(t_amp))
        )
        anterior[lead] = {
            "st_j": _st_value(context, lead),
            "st_mid": context.lead_value(
                lead, "st_mid_mv", "reliable_for_qrs"
            ),
            "st_80": context.lead_value(
                lead, "st_80ms_mv", "reliable_for_qrs"
            ),
            "st_morphology": context.lead_raw_value(lead, "st_morphology"),
            "st_slope": context.lead_value(lead, "st_slope_mv_per_ms"),
            "t_amp": t_amp,
            "t_symmetry": context.lead_value(lead, "t_symmetry", "reliable_for_t"),
            "t_polarity": context.lead_raw_value(lead, "t_polarity", "reliable_for_t"),
            "qrs_peak_amplitude_mv": qrs_peak_amplitude,
            "t_to_qrs_ratio": t_to_qrs_ratio,
            "st_j_sd_mv": st_j_sd,
            "t_amp_sd_mv": t_amp_sd,
            "st_beat_support": beat_support,
            "beat_consistent": beat_consistent,
            "baseline_confidence": baseline_confidence,
            "baseline_reliable": (
                hybrid_reliable is not False
                and (
                    baseline_confidence is None
                    or baseline_confidence >= 0.60
                )
            ),
        }
    avr_st = _st_value(context, "aVR")
    de_winter_leads = [
        lead for lead, values in anterior.items()
        if values["st_j"] is not None
        and values["baseline_reliable"]
        # Very large apparent depression is more likely residual QRS tail or
        # a bad J fiducial than the classic de-Winter morphology.
        and values["st_j"] >= -0.40
        and values["st_j"] <= -0.10
        and values["st_mid"] is not None
        and values["st_mid"] <= 0.00
        and values["st_80"] is not None
        and values["st_80"] <= 0.05
        and (
            str(values["st_morphology"]).lower() == "upsloping"
            or (values["st_slope"] is not None and values["st_slope"] > 0.0)
        )
        and values["st_slope"] is not None
        and values["st_slope"] > 0.0
        and values["t_amp"] is not None
        and values["t_amp"] >= 0.50
        and values["t_symmetry"] is not None
        and 0.50 <= values["t_symmetry"] <= 1.50
        and str(values["t_polarity"]).strip().lower()
        in {"1", "positive", "upright"}
        and values["t_to_qrs_ratio"] is not None
        and values["t_to_qrs_ratio"] >= 0.40
        and values["beat_consistent"]
    ]
    de_winter_lead_set = set(de_winter_leads)
    de_winter_pairs = [
        [first, second]
        for first, second in (("V2", "V3"), ("V3", "V4"))
        if first in de_winter_lead_set and second in de_winter_lead_set
    ]
    de_winter_available = sum(
        values["st_j"] is not None
        and values["st_mid"] is not None
        and values["st_80"] is not None
        and values["t_amp"] is not None
        and values["t_symmetry"] is not None
        and values["t_to_qrs_ratio"] is not None
        and values["st_slope"] is not None
        and values["st_j_sd_mv"] is not None
        and values["t_amp_sd_mv"] is not None
        and values["st_beat_support"] is not None
        for values in anterior.values()
    ) >= 2
    de_winter = bool(
        de_winter_pairs and avr_st is not None and avr_st >= 0.05
    )

    wellens_leads = [
        lead for lead, values in anterior.items()
        if values["st_j"] is not None
        and values["baseline_reliable"]
        and abs(values["st_j"]) < 0.10
        and values["t_amp"] is not None
        and values["t_amp"] <= -0.10
        and values["t_symmetry"] is not None
        and 0.5 <= values["t_symmetry"] <= 1.5
    ]
    wellens_available = sum(
        values["st_j"] is not None
        and values["t_amp"] is not None
        and values["t_symmetry"] is not None
        for values in anterior.values()
    ) >= 2

    other_depression = {
        lead: _st_depression_facts(context, lead)
        for lead in STANDARD_LEADS if lead != "aVR"
    }
    depressed = [
        lead for lead, values in other_depression.items()
        if values["persistent_horizontal_or_downsloping_depression"]
    ]
    left_main_available = avr_st is not None and len(
        [
            values
            for values in other_depression.values()
            if values["measurements_available"]
        ]
    ) >= 6
    left_main = left_main_available and avr_st >= 0.10 and len(depressed) >= 6

    conduction_codes = _matched_conduction_codes(conduction)
    wide_qrs = (context.global_value("qrs_ms") or 0.0) >= 120.0
    repolarization_confounders = sorted(
        conduction_codes
        & {
            "rbbb_pattern",
            "lbbb_pattern",
            "nonspecific_ivcd",
            "preexcitation_pattern",
        }
    )
    if wide_qrs:
        repolarization_confounders.append("qrs_duration_ge_120_ms")
    if _pacing_present(context):
        repolarization_confounders.append("ventricular_pacing")
    repolarization_confounders = sorted(set(repolarization_confounders))
    rhythm_confounders = _rhythm_template_confounders(context)
    de_winter_confounders = sorted(
        set(repolarization_confounders + rhythm_confounders)
    )
    wellens_confounders = list(de_winter_confounders)
    left_main_confounders = list(de_winter_confounders)

    def template_status(
        matched: bool,
        available: bool,
        confounders: list[str],
    ) -> str:
        if confounders:
            return "suppressed"
        if matched:
            return "matched"
        return "not_matched" if available else "unavailable"

    return [
        _result(
            rule_id="CLIN-HIGHRISK-DEWINTER-01",
            domain="high_risk_patterns",
            code="de_winter_pattern",
            status=template_status(
                de_winter, de_winter_available, de_winter_confounders
            ),
            evidence={
                "anterior": anterior,
                "avr_st_j_mv": avr_st,
                "qualifying_leads": de_winter_leads,
                "qualifying_contiguous_pairs": de_winter_pairs,
                "criteria": {
                    "j_point_depression_mv": [-0.40, -0.10],
                    "st_mid_max_mv": 0.00,
                    "st_80ms_max_mv": 0.05,
                    "minimum_t_amp_mv": 0.50,
                    "minimum_t_to_qrs_ratio": 0.40,
                    "t_symmetry_range": [0.50, 1.50],
                    "minimum_beat_support": 3,
                    "maximum_st_j_sd_mv": 0.08,
                },
            },
            missing_inputs=[] if de_winter_available else ["anterior_st_t_morphology"],
            suppressed_by=de_winter_confounders,
            statement="de Winter pattern suggesting proximal LAD occlusion; urgent review",
            confidence="moderate" if de_winter else None,
            priority="P1",
            human_review_required=de_winter,
            normality_required=False,
        ),
        _result(
            rule_id="CLIN-HIGHRISK-WELLENS-01",
            domain="high_risk_patterns",
            code="wellens_pattern",
            status=template_status(
                len(wellens_leads) >= 2,
                wellens_available,
                wellens_confounders,
            ),
            evidence={"anterior": anterior, "qualifying_leads": wellens_leads},
            missing_inputs=[] if wellens_available else ["anterior_t_wave_symmetry"],
            suppressed_by=wellens_confounders,
            statement="Wellens-type anterior T-wave pattern; urgent clinical correlation",
            confidence="moderate" if len(wellens_leads) >= 2 else None,
            priority="P1",
            human_review_required=len(wellens_leads) >= 2,
            normality_required=False,
        ),
        _result(
            rule_id="CLIN-HIGHRISK-LEFTMAIN-01",
            domain="high_risk_patterns",
            code="left_main_pattern",
            status=template_status(
                left_main, left_main_available, left_main_confounders
            ),
            evidence={
                "avr_st_j_mv": avr_st,
                "diffusely_depressed_leads": depressed,
            },
            missing_inputs=[] if left_main_available else ["avr_and_six_other_st_leads"],
            suppressed_by=left_main_confounders,
            statement="Diffuse ST depression with aVR elevation; consider left-main or multivessel ischemia",
            confidence="moderate" if left_main else None,
            priority="P1",
            human_review_required=left_main,
            normality_required=False,
        ),
    ]


def _q_wave_pattern(context) -> RuleEvaluation:
    facts = {}
    matched_groups = {}
    territory_results = {}
    unresolved_territories = []
    for territory, leads in Q_GROUPS.items():
        qualifying = []
        unresolved_candidates = []
        for lead in leads:
            q_amp = context.lead_value(lead, "q_amp_mv")
            q_duration = context.lead_value(lead, "q_duration_ms")
            r_amp = context.lead_value(lead, "r_amp_mv")
            facts[lead] = {
                "q_amp_mv": q_amp,
                "q_duration_ms": q_duration,
                "r_amp_mv": r_amp,
            }
            if q_amp is None or q_amp >= 0:
                continue
            ratio = abs(q_amp) / abs(r_amp) if r_amp not in {None, 0.0} else None
            # When R amplitude is measurable, gate on the Q/R ratio rather than
            # absolute Q depth alone: a fixed mV threshold false-triggers in
            # leads with a large QRS (e.g. LVH), where a proportionally tiny Q
            # wave can still clear an absolute-depth cutoff. This mirrors the
            # Seattle-2 athlete-ECG criteria, which switched from absolute Q
            # amplitude to a Q/R ratio for the same reason. Absolute depth is
            # only used as a fallback when R amplitude isn't available to form
            # a ratio.
            if ratio is not None:
                amplitude_candidate = ratio >= _ISCHEMIA.pathological_q_r_ratio
            else:
                amplitude_candidate = abs(q_amp) >= _ISCHEMIA.pathological_q_amplitude_mv
            # A small physiological Q wave cannot become pathological merely
            # because its duration was not delineated.
            if not amplitude_candidate:
                continue
            if q_duration is None:
                unresolved_candidates.append(lead)
                continue
            if q_duration >= _ISCHEMIA.pathological_q_duration_ms:
                qualifying.append(lead)
            elif (
                lead in {"V2", "V3"}
                and q_duration >= _ISCHEMIA.pathological_q_duration_v2_v3_ms
            ):
                # Any Q wave in V2-V3 of at least 20 ms is abnormal under the
                # universal-definition morphology criteria; it need not wait
                # for a second noncontiguous lead.
                qualifying.append(lead)
        anterior_v23 = territory == "anterior" and any(
            lead in {"V2", "V3"} for lead in qualifying
        )
        if len(qualifying) >= 2 or anterior_v23:
            matched_groups[territory] = qualifying
        can_still_match = (
            not anterior_v23
            and len(qualifying) < 2
            and (
                len(qualifying) + len(unresolved_candidates) >= 2
                or (
                    territory == "anterior"
                    and any(lead in {"V2", "V3"} for lead in unresolved_candidates)
                )
            )
        )
        if can_still_match:
            unresolved_territories.append(territory)
        territory_results[territory] = {
            "qualifying_leads": qualifying,
            "unresolved_candidate_leads": unresolved_candidates,
            "status": (
                "matched"
                if len(qualifying) >= 2 or anterior_v23
                else "indeterminate" if can_still_match else "not_matched"
            ),
        }
    if matched_groups:
        status = "matched"
        missing = []
    elif unresolved_territories:
        status = "unavailable"
        missing = sorted(
            f"lead.{lead}.q_duration_ms"
            for territory in unresolved_territories
            for lead in territory_results[territory]["unresolved_candidate_leads"]
        )
    else:
        status = "not_matched"
        missing = []
    return _result(
        rule_id="CLIN-ISCHEMIA-Q-01",
        domain="ischemia_infarction",
        code="prior_infarct_q_wave_pattern",
        status=status,
        evidence={
            "lead_facts": facts,
            "matched_groups": matched_groups,
            "territory_results": territory_results,
            "criteria": {
                "q_duration_ms": _ISCHEMIA.pathological_q_duration_ms,
                "q_duration_v2_v3_ms": _ISCHEMIA.pathological_q_duration_v2_v3_ms,
                "q_amplitude_mv": _ISCHEMIA.pathological_q_amplitude_mv,
                "q_r_ratio": _ISCHEMIA.pathological_q_r_ratio,
            },
        },
        missing_inputs=missing,
        statement="Pathological Q-wave pattern compatible with prior or age-indeterminate infarction; review confounders and prior ECG",
    )


def _posterior_pattern(
    context, conduction: Iterable[RuleEvaluation]
) -> RuleEvaluation:
    code = "posterior_ischemia_screen"
    if context.precordial_reversal:
        return _result(
            rule_id="CLIN-ISCHEMIA-POST-01",
            domain="ischemia_infarction",
            code=code,
            status="suppressed",
            evidence={"recommended_leads": ["V7", "V8", "V9"]},
            suppressed_by=["precordial_lead_reversal"],
        )
    values = {
        lead: _st_value(context, lead)
        for lead in ("V1", "V2", "V3")
    }
    measured = [value for value in values.values() if value is not None]
    depressed = [lead for lead, value in values.items() if value is not None and value <= -0.05]
    morphology = {}
    supported = []
    for lead in depressed:
        t_amp = context.lead_value(lead, "t_amp_mv", "reliable_for_t")
        r_amp = context.lead_value(lead, "r_amp_mv", "reliable_for_qrs")
        s_amp = context.lead_value(lead, "s_amp_mv", "reliable_for_qrs")
        st_mid = context.lead_value(lead, "st_mid_mv", "reliable_for_qrs")
        st_80 = context.lead_value(lead, "st_80ms_mv", "reliable_for_qrs")
        st_morphology = context.lead_raw_value(
            lead, "st_morphology", "reliable_for_qrs"
        )
        positive_t = t_amp is not None and t_amp > 0.02
        dominant_r = (
            r_amp is not None and s_amp is not None and r_amp > abs(s_amp)
        )
        persistent_depression = (
            (st_mid is not None and st_mid <= -0.05)
            or (st_80 is not None and st_80 <= -0.05)
            or str(st_morphology or "").lower() in {"horizontal", "downsloping"}
        )
        morphology[lead] = {
            "t_amp_mv": t_amp,
            "r_amp_mv": r_amp,
            "s_amp_mv": s_amp,
            "st_mid_mv": st_mid,
            "st_80ms_mv": st_80,
            "st_morphology": st_morphology,
            "positive_terminal_t": positive_t,
            "dominant_r": dominant_r,
            "persistent_st_depression": persistent_depression,
        }
        if persistent_depression and (positive_t or dominant_r):
            supported.append(lead)
    missing = [] if len(measured) >= 2 else ["two_reliable_leads_among_V1_V2_V3"]
    confounders = sorted(
        {
            str(item.evidence.get("evaluates_code") or item.statement_code)
            for item in conduction
            if item.status == "matched"
            and (item.evidence.get("evaluates_code") or item.statement_code)
            in {"rbbb_pattern", "lbbb_pattern", "nonspecific_ivcd"}
        }
    )
    screen_positive = len(depressed) >= 2 and len(supported) >= 2
    if screen_positive and confounders:
        status = "indeterminate"
        coverage = "partial"
        confidence = "low"
    elif screen_positive:
        status = "matched"
        coverage = "full"
        confidence = "moderate"
    elif missing:
        status = "unavailable"
        coverage = "unavailable"
        confidence = "unavailable"
    else:
        status = "not_matched"
        coverage = "full"
        confidence = "moderate"
    return _result(
        rule_id="CLIN-ISCHEMIA-POST-01",
        domain="ischemia_infarction",
        code=code,
        status=status,
        evidence={
            "st_j_mv": values,
            "depressed_leads": depressed,
            "supported_leads": supported,
            "morphology": morphology,
            "conduction_confounders": confounders,
            "recommended_leads": ["V7", "V8", "V9"],
            "manual_confirmation_required": screen_positive,
        },
        missing_inputs=missing,
        statement="Posterior ischemia screening pattern; record V7-V9 and correlate clinically",
        coverage=coverage,
        confidence=confidence,
    )


def evaluate_sgarbossa(context, *, paced: bool) -> RuleEvaluation:
    qrs_ms = context.global_value("qrs_ms")
    eligible = bool(paced or (qrs_ms is not None and float(qrs_ms) >= 120.0))
    if not eligible:
        return _result(
            rule_id="CLIN-ISCHEMIA-SGARBOSSA-01",
            domain="ischemia_infarction",
            code="sgarbossa_positive",
            status="not_applicable",
            evidence={
                "score": 0,
                "criteria": [],
                "paced": paced,
                "qrs_ms": qrs_ms,
                "eligible_context": False,
            },
            missing_inputs=["definite_lbbb_or_ventricular_pacing"],
            normality_required=False,
        )
    criteria = []
    score = 0
    available = 0
    for lead in STANDARD_LEADS:
        st = _st_value(context, lead)
        qrs_area = context.lead_value(
            lead, "qrs_signed_area_uv_ms", "reliable_for_qrs"
        )
        if qrs_area is None:
            qrs_area = context.lead_value(
                lead, "qrs_signed_area", "reliable_for_qrs"
            )
        if st is None or qrs_area is None:
            continue
        available += 1
        if st >= 0.10 and qrs_area > 0:
            score += 5
            criteria.append({"lead": lead, "criterion": "concordant_st_elevation", "points": 5})
            break
    smith_positive = []
    for lead in STANDARD_LEADS:
        st = _st_value(context, lead)
        s_amp = context.lead_value(lead, "s_amp_mv", "reliable_for_qrs")
        if st is None or s_amp is None or s_amp >= -0.05 or st <= 0.0:
            continue
        ratio = st / abs(s_amp)
        if ratio >= 0.25:
            smith_positive.append({"lead": lead, "st_s_ratio": ratio})
    if smith_positive:
        score = max(score, 3)
        criteria.append({
            "criterion": "modified_sgarbossa_smith_ratio",
            "points": 3,
            "leads": smith_positive,
        })
    if not any(item["criterion"] == "concordant_st_elevation" for item in criteria):
        if sum(
            1
            for lead in ("V1", "V2", "V3")
            if (_st_value(context, lead) or 0.0) <= -0.10
        ) >= 1:
            score += 3
            criteria.append({"criterion": "V1_V3_st_depression", "points": 3})
        if any(
            (_st_value(context, lead) or 0.0) >= 0.50
            and (
                context.lead_value(
                    lead, "qrs_signed_area_uv_ms", "reliable_for_qrs"
                )
                or context.lead_value(
                    lead, "qrs_signed_area", "reliable_for_qrs"
                )
                or 0.0
            ) < 0
            for lead in STANDARD_LEADS
        ):
            score += 2
            criteria.append({"criterion": "excessively_discordant_st_elevation", "points": 2})
    missing = [] if available else ["reliable_st_and_qrs_polarity"]
    return _result(
        rule_id="CLIN-ISCHEMIA-SGARBOSSA-01",
        domain="ischemia_infarction",
        code="sgarbossa_positive",
        status="matched" if score >= 3 else ("unavailable" if missing else "not_matched"),
        evidence={
            "score": score,
            "criteria": criteria,
            "paced": paced,
            "qrs_ms": qrs_ms,
            "eligible_context": True,
        },
        missing_inputs=missing,
        statement="Positive Sgarbossa screening pattern for acute coronary occlusion",
        confidence="moderate" if score >= 3 else None,
        priority="P1",
        human_review_required=score >= 3,
    )


def _brugada_screen(context) -> RuleEvaluation:
    if context.precordial_reversal:
        return _result(
            rule_id="CLIN-HIGHRISK-BRUGADA-01",
            domain="high_risk_patterns",
            code="brugada_type1_screening",
            status="suppressed",
            evidence={},
            suppressed_by=["precordial_lead_reversal"],
            normality_required=False,
        )
    facts = {}
    high_st_without_morphology = []
    for lead in ("V1", "V2"):
        st = _st_value(context, lead)
        morphology = context.lead_raw_value(lead, "st_morphology", "reliable_for_qrs")
        t_amp = context.lead_value(lead, "t_amp_mv", "reliable_for_t")
        facts[lead] = {"st_j_mv": st, "st_morphology": morphology, "t_amp_mv": t_amp}
        if st is None or st < 0.20:
            continue
        if morphology is None or t_amp is None:
            high_st_without_morphology.append(lead)
            continue
        slope = context.lead_value(lead, "st_slope_mv_per_ms", "reliable_for_qrs")
        if (
            str(morphology).lower() in {"coved", "descending", "downsloping", "coved_type1"}
            or (slope is not None and slope < 0.0)
        ) and t_amp < 0:
            return _result(
                rule_id="CLIN-HIGHRISK-BRUGADA-01",
                domain="high_risk_patterns",
                code="brugada_type1_screening",
                status="matched",
                evidence={"lead_facts": facts, "matched_lead": lead},
                statement="Brugada type-1 ECG screening pattern; syndrome diagnosis requires clinical confirmation and appropriate lead placement",
                confidence="moderate",
                normality_required=False,
                priority="P1",
                human_review_required=True,
            )
    missing = [f"lead.{lead}.st_morphology_or_t_amp" for lead in high_st_without_morphology]
    measured = any(item["st_j_mv"] is not None for item in facts.values())
    if not measured:
        missing.append("reliable_V1_or_V2_st_measurement")
    return _result(
        rule_id="CLIN-HIGHRISK-BRUGADA-01",
        domain="high_risk_patterns",
        code="brugada_type1_screening",
        status="unavailable" if missing else "not_matched",
        evidence={"lead_facts": facts},
        missing_inputs=missing,
        normality_required=False,
    )


def evaluate_ischemia(context, conduction: list[RuleEvaluation]) -> list[RuleEvaluation]:
    results = [
        _acute_occlusion_pattern(context, conduction),
        _st_depression_pattern(context, conduction),
        _q_wave_pattern(context),
        _posterior_pattern(context, conduction),
        _brugada_screen(context),
        *_acute_high_risk_templates(context, conduction),
    ]
    lbbb = any(
        item.status == "matched" and item.evidence.get("evaluates_code") == "lbbb_pattern"
        for item in conduction
    )
    paced = _pacing_present(context)
    qrs_ms = context.global_value("qrs_ms")
    definite_lbbb_context = bool(
        lbbb and qrs_ms is not None and float(qrs_ms) >= 120.0
    )
    if definite_lbbb_context or paced:
        results.append(evaluate_sgarbossa(context, paced=paced))
    return results
