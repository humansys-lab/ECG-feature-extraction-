from feature_extraction.ecgfeat.clinical_rules.ischemia import (
    evaluate_ischemia,
    evaluate_sgarbossa,
)
from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation


class Context:
    def __init__(
        self,
        *,
        sex="female",
        age=39.0,
        qrs_ms=100.0,
        heart_rate_bpm=70.0,
        leads=None,
        precordial_reversal=False,
        metadata=None,
    ):
        self.sex = sex
        self.age_years = age
        self.precordial_reversal = precordial_reversal
        self.excluded_leads = frozenset(
            {"V1", "V2", "V3", "V4", "V5", "V6"}
            if precordial_reversal
            else set()
        )
        self._global = {
            "qrs_ms": qrs_ms,
            "heart_rate_bpm": heart_rate_bpm,
        }
        self._leads = leads or {}
        self.features = type(
            "Features",
            (),
            {
                "metadata": metadata or {},
                "global_features": type(
                    "GlobalFeatures",
                    (),
                    {"paced_rhythm": False},
                )(),
            },
        )()

    def global_value(self, name):
        return self._global.get(name)

    def lead_available(self, lead, reliability):
        return lead not in self.excluded_leads and lead in self._leads

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        if not self.lead_available(lead, reliability):
            return None
        value = self._leads[lead].get(name)
        return value if isinstance(value, (int, float)) else None

    def lead_raw_value(self, lead, name, reliability="reliable_for_qrs"):
        if not self.lead_available(lead, reliability):
            return None
        return self._leads[lead].get(name)

    def lead_variance_value(self, lead, name, reliability="reliable_for_qrs"):
        return self.lead_value(lead, name, reliability)


def by_code(evaluations, code):
    return next(
        item
        for item in evaluations
        if item.statement_code == code or item.evidence.get("evaluates_code") == code
    )


def matched_code(evaluations, code):
    return any(item.statement_code == code and item.status == "matched" for item in evaluations)


def test_noncontiguous_st_elevation_does_not_match() -> None:
    context = Context(
        leads={"V2": {"st_on_mv": 0.16}, "V5": {"st_on_mv": 0.11}}
    )

    assert not matched_code(evaluate_ischemia(context, []), "acute_occlusion_pattern")


def test_female_v2_v3_threshold_is_015_mv() -> None:
    context = Context(
        sex="female",
        age=39,
        leads={
            "V2": {
                "st_on_mv": 0.15,
                "st_mid_mv": 0.10,
                "st_80ms_mv": 0.08,
            },
            "V3": {
                "st_on_mv": 0.15,
                "st_mid_mv": 0.10,
                "st_80ms_mv": 0.08,
            },
        },
    )

    assert matched_code(evaluate_ischemia(context, []), "acute_occlusion_pattern")


def test_j_point_only_elevation_is_indeterminate_not_acute_occlusion() -> None:
    context = Context(
        leads={
            "II": {
                "st_on_mv": 0.12,
                "st_mid_mv": 0.01,
                "st_80ms_mv": 0.00,
            },
            "aVF": {
                "st_on_mv": 0.12,
                "st_mid_mv": 0.01,
                "st_80ms_mv": 0.00,
            },
        },
    )

    result = by_code(
        evaluate_ischemia(context, []), "acute_occlusion_pattern"
    )

    assert result.status == "indeterminate"
    assert result.statement_code is None
    assert result.evidence["j_point_only_pairs"] == [["II", "aVF"]]


def test_fast_rhythm_downgrades_persistent_st_elevation_to_indeterminate() -> None:
    context = Context(
        heart_rate_bpm=125.0,
        leads={
            "II": {
                "st_on_mv": 0.12,
                "st_mid_mv": 0.10,
                "st_80ms_mv": 0.08,
            },
            "aVF": {
                "st_on_mv": 0.12,
                "st_mid_mv": 0.10,
                "st_80ms_mv": 0.08,
            },
        },
    )

    result = by_code(
        evaluate_ischemia(context, []), "acute_occlusion_pattern"
    )

    assert result.status == "indeterminate"
    assert result.statement_code is None
    assert "heart_rate_gt_100_bpm" in result.evidence["rhythm_confounders"]


def test_wide_qrs_suppresses_standard_acute_occlusion_template() -> None:
    context = Context(
        qrs_ms=160.0,
        leads={"II": {"st_on_mv": 0.15}, "aVF": {"st_on_mv": 0.15}},
    )

    result = by_code(
        evaluate_ischemia(context, []), "acute_occlusion_pattern"
    )

    assert result.status == "suppressed"
    assert "qrs_duration_ge_120_ms" in result.suppressed_by


def test_missing_q_duration_cannot_match_old_mi() -> None:
    context = Context(
        leads={
            "II": {"q_amp_mv": -0.10, "q_duration_ms": None, "r_amp_mv": 0.30},
            "III": {"q_amp_mv": -0.10, "q_duration_ms": None, "r_amp_mv": 0.25},
            "aVF": {"q_amp_mv": -0.10, "q_duration_ms": None, "r_amp_mv": 0.20},
        }
    )

    result = by_code(
        evaluate_ischemia(context, []), "prior_infarct_q_wave_pattern"
    )
    assert result.status == "unavailable"


def test_tiny_q_with_missing_duration_does_not_make_rule_unavailable() -> None:
    context = Context(
        leads={
            "II": {"q_amp_mv": -0.02, "q_duration_ms": None, "r_amp_mv": 0.60},
            "III": {"q_amp_mv": 0.0, "q_duration_ms": None, "r_amp_mv": 0.50},
            "aVF": {"q_amp_mv": 0.0, "q_duration_ms": None, "r_amp_mv": 0.55},
        }
    )

    result = by_code(
        evaluate_ischemia(context, []), "prior_infarct_q_wave_pattern"
    )

    assert result.status == "not_matched"


def test_one_unresolved_q_candidate_cannot_form_pathological_pair() -> None:
    context = Context(
        leads={
            "II": {"q_amp_mv": -0.12, "q_duration_ms": None, "r_amp_mv": 0.50},
            "III": {"q_amp_mv": 0.0, "q_duration_ms": None, "r_amp_mv": 0.50},
            "aVF": {"q_amp_mv": 0.0, "q_duration_ms": None, "r_amp_mv": 0.50},
        }
    )

    result = by_code(
        evaluate_ischemia(context, []), "prior_infarct_q_wave_pattern"
    )

    assert result.status == "not_matched"


def test_lvh_high_amplitude_q_below_ratio_not_matched() -> None:
    # Large R amplitude (LVH-like) with a Q deep enough to clear the absolute
    # 0.10 mV cutoff, but the Q/R ratio (0.05) is far below the 0.25
    # pathological threshold. Should not be flagged as a pathological Q
    # pattern -- an absolute-amplitude-only check would false-trigger here.
    context = Context(
        leads={
            "II":  {"q_amp_mv": -0.15, "q_duration_ms": 35.0, "r_amp_mv": 3.0},
            "III": {"q_amp_mv": -0.15, "q_duration_ms": 35.0, "r_amp_mv": 3.0},
            "aVF": {"q_amp_mv": -0.15, "q_duration_ms": 35.0, "r_amp_mv": 3.0},
        }
    )

    result = by_code(
        evaluate_ischemia(context, []), "prior_infarct_q_wave_pattern"
    )

    assert result.status == "not_matched"


def test_ratio_qualifying_q_still_matches_with_measurable_r() -> None:
    # Sanity check that the ratio path itself still works: same absolute Q
    # depth as above, but with a normal (not LVH-sized) R amplitude the ratio
    # clears 0.25, so this should still match.
    context = Context(
        leads={
            "II":  {"q_amp_mv": -0.15, "q_duration_ms": 35.0, "r_amp_mv": 0.40},
            "III": {"q_amp_mv": -0.15, "q_duration_ms": 35.0, "r_amp_mv": 0.40},
            "aVF": {"q_amp_mv": -0.15, "q_duration_ms": 35.0, "r_amp_mv": 0.40},
        }
    )

    result = by_code(
        evaluate_ischemia(context, []), "prior_infarct_q_wave_pattern"
    )

    assert result.status == "matched"


def test_precordial_reversal_suppresses_posterior_pattern() -> None:
    context = Context(
        precordial_reversal=True,
        leads={
            "V1": {"st_on_mv": -0.12},
            "V2": {"st_on_mv": -0.12},
            "V3": {"st_on_mv": -0.12},
        },
    )

    result = by_code(evaluate_ischemia(context, []), "posterior_ischemia_screen")
    assert result.status == "suppressed"


def test_posterior_st_depression_without_supporting_morphology_does_not_match() -> None:
    context = Context(
        leads={
            "V1": {"st_on_mv": -0.10, "t_amp_mv": -0.05, "r_amp_mv": 0.2, "s_amp_mv": -0.6},
            "V2": {"st_on_mv": -0.12, "t_amp_mv": -0.04, "r_amp_mv": 0.3, "s_amp_mv": -0.7},
            "V3": {"st_on_mv": 0.00, "t_amp_mv": 0.10, "r_amp_mv": 0.8, "s_amp_mv": -0.3},
        }
    )

    result = by_code(evaluate_ischemia(context, []), "posterior_ischemia_screen")

    assert result.status == "not_matched"


def test_posterior_st_depression_with_positive_t_support_matches_screen() -> None:
    context = Context(
        leads={
            "V1": {"st_on_mv": -0.10, "st_mid_mv": -0.08, "st_80ms_mv": -0.06, "t_amp_mv": 0.08, "r_amp_mv": 0.2, "s_amp_mv": -0.6},
            "V2": {"st_on_mv": -0.12, "st_mid_mv": -0.09, "st_80ms_mv": -0.07, "t_amp_mv": 0.12, "r_amp_mv": 0.3, "s_amp_mv": -0.7},
            "V3": {"st_on_mv": 0.00, "t_amp_mv": 0.10, "r_amp_mv": 0.8, "s_amp_mv": -0.3},
        }
    )

    result = by_code(evaluate_ischemia(context, []), "posterior_ischemia_screen")

    assert result.status == "matched"
    assert result.confidence == "moderate"


def test_upsloping_j_point_depression_with_positive_t_does_not_match_posterior_screen() -> None:
    context = Context(
        leads={
            "V1": {"st_on_mv": -0.10, "st_mid_mv": 0.02, "st_80ms_mv": 0.05, "st_morphology": "upsloping", "t_amp_mv": 0.08, "r_amp_mv": 0.2, "s_amp_mv": -0.6},
            "V2": {"st_on_mv": -0.12, "st_mid_mv": 0.01, "st_80ms_mv": 0.06, "st_morphology": "upsloping", "t_amp_mv": 0.12, "r_amp_mv": 0.3, "s_amp_mv": -0.7},
        }
    )

    result = by_code(evaluate_ischemia(context, []), "posterior_ischemia_screen")

    assert result.status == "not_matched"


def test_rbbb_confounds_posterior_ischemia_screen() -> None:
    context = Context(
        leads={
            "V1": {"st_on_mv": -0.10, "st_mid_mv": -0.08, "t_amp_mv": 0.08, "r_amp_mv": 0.2, "s_amp_mv": -0.6},
            "V2": {"st_on_mv": -0.12, "st_mid_mv": -0.09, "t_amp_mv": 0.12, "r_amp_mv": 0.3, "s_amp_mv": -0.7},
        }
    )
    conduction = [
        RuleEvaluation(
            rule_id="TEST-RBBB",
            domain="conduction",
            status="matched",
            statement_code="rbbb_pattern",
            evidence={"evaluates_code": "rbbb_pattern"},
        )
    ]

    result = by_code(evaluate_ischemia(context, conduction), "posterior_ischemia_screen")

    assert result.status == "indeterminate"
    assert result.coverage == "partial"


def test_lbbb_uses_sgarbossa_instead_of_blanket_suppression() -> None:
    context = Context(
        qrs_ms=150,
        leads={"I": {"st_on_mv": 0.12, "qrs_signed_area_uv_ms": 100.0}},
    )

    result = evaluate_sgarbossa(context, paced=False)
    assert result.status == "matched"
    assert result.statement_code == "sgarbossa_positive"
    assert result.evidence["score"] >= 3


def test_sgarbossa_rejects_narrow_nonpaced_qrs() -> None:
    context = Context(
        qrs_ms=104,
        leads={"I": {"st_on_mv": 0.12, "qrs_signed_area_uv_ms": 100.0}},
    )

    result = evaluate_sgarbossa(context, paced=False)

    assert result.status == "not_applicable"
    assert "definite_lbbb_or_ventricular_pacing" in result.missing_inputs


def test_weak_pacing_candidate_does_not_enable_sgarbossa() -> None:
    context = Context(
        qrs_ms=104,
        leads={"I": {"st_on_mv": 0.12, "qrs_signed_area_uv_ms": 100.0}},
        metadata={
            "measurement_pacing_state": "on",
            "paced_beat_fraction": 0.55,
            "global_measurement_paced": True,
            "rhythm_analysis": {
                "pacing_context": {
                    "spike_count": 20,
                    "paced_fraction": 0.55,
                    "ventricular_pacing_present": True,
                },
                "pacing_failures": {
                    "spike_count": 20,
                    "capture_alignment_fraction": 0.20,
                },
            },
        },
    )
    context.features.global_features.paced_rhythm = True

    results = evaluate_ischemia(context, [])

    assert not any(
        item.evidence.get("evaluates_code") == "sgarbossa_positive"
        for item in results
    )


def test_capture_aligned_ventricular_pacing_enables_sgarbossa() -> None:
    context = Context(
        qrs_ms=104,
        leads={"I": {"st_on_mv": 0.12, "qrs_signed_area_uv_ms": 100.0}},
        metadata={
            "measurement_pacing_state": "on",
            "paced_beat_fraction": 0.55,
            "global_measurement_paced": True,
            "rhythm_analysis": {
                "pacing_context": {
                    "spike_count": 20,
                    "paced_fraction": 0.55,
                    "ventricular_pacing_present": True,
                },
                "pacing_failures": {
                    "spike_count": 20,
                    "capture_alignment_fraction": 0.85,
                },
            },
        },
    )
    context.features.global_features.paced_rhythm = True

    result = by_code(evaluate_ischemia(context, []), "sgarbossa_positive")

    assert result.status == "matched"
    assert result.evidence["paced"] is True


def test_brugada_requires_morphology_and_negative_t() -> None:
    context = Context(
        leads={
            "V1": {
                "st_on_mv": 0.22,
                "st_morphology": "coved",
                "t_amp_mv": -0.12,
            }
        }
    )

    result = by_code(evaluate_ischemia(context, []), "brugada_type1_screening")
    assert result.status == "matched"
    assert "screening" in result.statement.lower()


def test_st_depression_requires_persistent_horizontal_or_downsloping_st() -> None:
    context = Context(
        leads={
            "I": {
                "st_on_mv": -0.10,
                "st_mid_mv": -0.08,
                "st_80ms_mv": -0.08,
                "st_morphology": "horizontal",
                "st_hybrid_baseline_confidence": 0.90,
            },
            "aVL": {
                "st_on_mv": -0.09,
                "st_mid_mv": -0.08,
                "st_80ms_mv": -0.08,
                "st_morphology": "downsloping",
                "st_hybrid_baseline_confidence": 0.85,
            },
        }
    )

    result = by_code(evaluate_ischemia(context, []), "st_depression")

    assert result.status == "matched"
    assert result.evidence["qualifying_contiguous_pairs"] == [["I", "aVL"]]


def test_diffuse_delayed_st_depression_is_retained_as_indeterminate() -> None:
    leads = {
        lead: {
            "st_on_mv": 0.02,
            "st_mid_mv": -0.09,
            "st_80ms_mv": -0.08,
            "st_morphology": "horizontal",
            "st_hybrid_baseline_confidence": 0.90,
        }
        for lead in ("I", "II", "V5", "V6")
    }
    context = Context(leads=leads)

    result = by_code(evaluate_ischemia(context, []), "st_depression")

    assert result.status == "indeterminate"
    assert result.statement_code is None
    assert result.evidence["diffuse_delayed_pattern"] is True
    assert result.evidence["delayed_persistent_leads"] == ["I", "II", "V5", "V6"]


def test_shallow_persistent_st_depression_is_indeterminate_not_positive() -> None:
    context = Context(
        leads={
            "II": {
                "st_on_mv": -0.060,
                "st_mid_mv": -0.065,
                "st_80ms_mv": -0.060,
                "st_morphology": "horizontal",
                "st_hybrid_baseline_confidence": 0.90,
            },
            "aVF": {
                "st_on_mv": -0.059,
                "st_mid_mv": -0.065,
                "st_80ms_mv": -0.059,
                "st_morphology": "horizontal",
                "st_hybrid_baseline_confidence": 0.90,
            },
        }
    )

    result = by_code(evaluate_ischemia(context, []), "st_depression")

    assert result.status == "indeterminate"
    assert result.statement_code is None
    assert result.evidence["borderline_contiguous_pairs"] == [["II", "aVF"]]


def test_j_point_only_or_upsloping_depression_does_not_match_st_depression() -> None:
    context = Context(
        leads={
            "I": {
                "st_on_mv": -0.10,
                "st_mid_mv": -0.02,
                "st_80ms_mv": 0.01,
                "st_morphology": "upsloping",
            },
            "aVL": {
                "st_on_mv": -0.08,
                "st_mid_mv": -0.07,
                "st_80ms_mv": -0.06,
                "st_morphology": "horizontal",
            },
        }
    )

    result = by_code(evaluate_ischemia(context, []), "st_depression")

    assert result.status == "not_matched"


def test_wide_qrs_suppresses_wellens_template_before_matching() -> None:
    context = Context(
        qrs_ms=188.0,
        leads={
            "V2": {
                "st_on_mv": 0.02,
                "t_amp_mv": -0.30,
                "t_symmetry": 1.0,
            },
            "V3": {
                "st_on_mv": 0.01,
                "t_amp_mv": -0.25,
                "t_symmetry": 0.9,
            },
        },
    )

    result = by_code(evaluate_ischemia(context, []), "wellens_pattern")

    assert result.status == "suppressed"
    assert "qrs_duration_ge_120_ms" in result.suppressed_by


def test_fast_rhythm_suppresses_wellens_template() -> None:
    context = Context(
        heart_rate_bpm=155.0,
        leads={
            "V2": {
                "st_on_mv": 0.02,
                "t_amp_mv": -0.30,
                "t_symmetry": 1.0,
            },
            "V3": {
                "st_on_mv": 0.01,
                "t_amp_mv": -0.25,
                "t_symmetry": 0.9,
            },
        },
    )

    result = by_code(evaluate_ischemia(context, []), "wellens_pattern")

    assert result.status == "suppressed"
    assert "heart_rate_gt_100_bpm" in result.suppressed_by


def _de_winter_lead(*, st_j=-0.12, st_mid=-0.02, st_80=0.01):
    return {
        "st_on_mv": st_j,
        "st_mid_mv": st_mid,
        "st_80ms_mv": st_80,
        "st_morphology": "upsloping",
        "st_slope_mv_per_ms": 0.001,
        "t_amp_mv": 0.60,
        "t_symmetry": 1.0,
        "t_polarity": 1,
        "q_amp_mv": -0.02,
        "r_amp_mv": 1.20,
        "s_amp_mv": -0.30,
        "st_hybrid_baseline_confidence": 0.90,
        "st_hybrid_beat_support": 6,
        "st_on_mv_sd": 0.02,
        "t_amp_mv_sd": 0.05,
    }


def test_de_winter_requires_persistent_trajectory_and_t_qrs_ratio() -> None:
    context = Context(
        leads={
            "V2": _de_winter_lead(),
            "V3": _de_winter_lead(st_j=-0.14, st_mid=-0.03),
            "aVR": {"st_on_mv": 0.06},
        }
    )

    result = by_code(evaluate_ischemia(context, []), "de_winter_pattern")

    assert result.status == "matched"
    assert result.evidence["qualifying_leads"] == ["V2", "V3"]
    assert result.evidence["qualifying_contiguous_pairs"] == [["V2", "V3"]]


def test_rapidly_normalizing_upsloping_st_does_not_match_de_winter() -> None:
    context = Context(
        leads={
            "V2": _de_winter_lead(st_mid=0.03, st_80=0.08),
            "V3": _de_winter_lead(st_mid=0.02, st_80=0.07),
            "aVR": {"st_on_mv": 0.06},
        }
    )

    result = by_code(evaluate_ischemia(context, []), "de_winter_pattern")

    assert result.status == "not_matched"


def test_unstable_representative_st_t_does_not_match_de_winter() -> None:
    unstable_v2 = _de_winter_lead()
    unstable_v3 = _de_winter_lead()
    unstable_v2["st_on_mv_sd"] = 0.12
    unstable_v3["t_amp_mv_sd"] = 0.30
    context = Context(
        leads={
            "V2": unstable_v2,
            "V3": unstable_v3,
            "aVR": {"st_on_mv": 0.06},
        }
    )

    result = by_code(evaluate_ischemia(context, []), "de_winter_pattern")

    assert result.status == "not_matched"


def test_noncontiguous_anterior_leads_do_not_match_de_winter() -> None:
    v3 = _de_winter_lead()
    v3["st_on_mv"] = -0.05
    context = Context(
        leads={
            "V2": _de_winter_lead(),
            "V3": v3,
            "V4": _de_winter_lead(),
            "aVR": {"st_on_mv": 0.06},
        }
    )

    result = by_code(evaluate_ischemia(context, []), "de_winter_pattern")

    assert result.status == "not_matched"
