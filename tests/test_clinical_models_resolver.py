from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation
from feature_extraction.ecgfeat.clinical_rules.engine import _available_domains
from feature_extraction.ecgfeat.clinical_rules.quality import evaluate_quality
from feature_extraction.ecgfeat.clinical_rules.resolver import ClinicalStatementResolver
from feature_extraction.ecgfeat.clinical_rules.rhythm import project_rhythm_evidence


def test_missing_optional_input_does_not_erase_proven_match() -> None:
    result = RuleEvaluation(
        rule_id="CLIN-TEST-01",
        domain="intervals",
        status="matched",
        required_inputs=["global.qt_ms", "optional.residual"],
        missing_inputs=["optional.residual"],
        coverage="partial",
    )

    assert result.status == "matched"
    assert result.coverage == "partial"


def test_incomplete_required_domain_forbids_normal() -> None:
    analysis = ClinicalStatementResolver().resolve(
        evaluations=[],
        required_domains={"quality", "rhythm", "conduction", "intervals"},
        available_domains={"quality", "rhythm", "conduction"},
    )

    assert analysis.overall_status == "incomplete"
    assert "intervals" in analysis.unavailable_domains


def test_abnormal_finding_surfaces_despite_unavailable_domain() -> None:
    abnormal = RuleEvaluation(
        rule_id="CLIN-TEST-03",
        domain="ischemia_infarction",
        status="matched",
        statement_code="acute_occlusion_pattern",
        severity="abnormal",
    )
    analysis = ClinicalStatementResolver().resolve(
        evaluations=[abnormal],
        required_domains={"ischemia_infarction", "hypertrophy"},
        available_domains={"ischemia_infarction"},
    )

    assert analysis.overall_status == "abnormal_with_limited_coverage"
    assert "hypertrophy" in analysis.unavailable_domains
    assert analysis.summary["partial_evaluation"] is True


def test_reference_conflict_never_changes_final_status() -> None:
    abnormal = RuleEvaluation(
        rule_id="CLIN-TEST-02",
        domain="intervals",
        status="matched",
        statement_code="prolonged_qt",
        severity="abnormal",
    )
    analysis = ClinicalStatementResolver().resolve(
        evaluations=[abnormal],
        required_domains={"intervals"},
        available_domains={"intervals"},
        reference_interpretations={"legacy": {"summary": "normal"}},
    )

    assert analysis.overall_status == "abnormal"


def test_technical_quality_has_priority_over_normal() -> None:
    context = SimpleNamespace(
        excluded_leads=frozenset(),
        features=SimpleNamespace(metadata={"record_quality": {"record_grade": "Q2"}}),
    )

    evaluations = evaluate_quality(context)
    analysis = ClinicalStatementResolver().resolve(
        evaluations=evaluations,
        required_domains={"quality"},
        available_domains={"quality"},
    )

    assert analysis.overall_status == "technically_limited"


def test_technical_limitation_preserves_reliable_positive_finding() -> None:
    rows = [
        RuleEvaluation(
            rule_id="CLIN-QUALITY-TEST",
            domain="quality",
            status="matched",
            statement_code="technically_limited",
            severity="technical",
        ),
        RuleEvaluation(
            rule_id="CLIN-ABNORMAL-TEST",
            domain="conduction",
            status="matched",
            statement_code="rbbb_pattern",
            severity="abnormal",
        ),
    ]

    analysis = ClinicalStatementResolver().resolve(
        evaluations=rows,
        required_domains={"quality", "conduction"},
        available_domains={"quality", "conduction"},
    )

    assert analysis.overall_status == "technically_limited_with_findings"


def test_unknown_optional_screen_blocks_definitive_normal_result() -> None:
    rows = [
        RuleEvaluation(
            rule_id="CLIN-RHYTHM-CORE",
            domain="rhythm",
            status="not_matched",
            normality_role="core",
        ),
        RuleEvaluation(
            rule_id="CLIN-HIGHRISK-OPTIONAL",
            domain="high_risk_patterns",
            status="indeterminate",
            coverage="unavailable",
            normality_required=False,
            normality_role="optional_screen",
        ),
    ]

    available = _available_domains(rows)
    analysis = ClinicalStatementResolver().resolve(
        evaluations=rows,
        required_domains={"rhythm", "high_risk_patterns"},
        available_domains=available,
    )

    assert "high_risk_patterns" not in available
    assert analysis.overall_status == "incomplete"


def test_rate_observation_alone_does_not_make_overall_abnormal() -> None:
    rate = RuleEvaluation(
        rule_id="CLIN-RHYTHM-RATE-TEST",
        domain="rhythm",
        status="matched",
        statement_code="bradycardia",
        severity="observation",
    )

    analysis = ClinicalStatementResolver().resolve(
        evaluations=[rate],
        required_domains={"rhythm"},
        available_domains={"rhythm"},
    )

    assert analysis.overall_status == "normal_with_core_coverage"
    assert analysis.summary["observations"] == 1


def test_af_rule_unavailable_without_rr_variability_evidence() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {"validated_qrst_subtraction": False},
                    "af_afl_summary": {"probable_af": False},
                }
            }
        )
    )

    evaluation = project_rhythm_evidence(context)[0]

    assert evaluation.status == "unavailable"
    assert "rhythm.rr_cv" in evaluation.missing_inputs


def test_af_matches_on_rr_and_p_wave_evidence_without_validated_qrst_subtraction() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {"validated_qrst_subtraction": False},
                    "af_afl_summary": {
                        "probable_af": True,
                        "probable_flutter": False,
                        "rr_cv": 0.22,
                        "organized_p_ratio": 0.10,
                    },
                }
            }
        )
    )

    evaluation = project_rhythm_evidence(context)[0]

    assert evaluation.status == "matched"
    assert evaluation.statement_code == "atrial_fibrillation_pattern"
    assert evaluation.confidence == "rr_and_p_wave_evidence"


def test_af_not_matched_with_reliable_negative_rr_evidence_even_if_unvalidated() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {"validated_qrst_subtraction": False},
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": False,
                        "rr_cv": 0.02,
                        "organized_p_ratio": 0.95,
                    },
                }
            }
        )
    )

    evaluation = project_rhythm_evidence(context)[0]

    assert evaluation.status == "not_matched"


def test_adult_tachycardia_is_projected_as_a_final_rhythm_fact() -> None:
    context = SimpleNamespace(
        age_years=39.0,
        global_value=lambda name: 102.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {"validated_qrst_subtraction": True},
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": False,
                    },
                }
            }
        ),
    )

    evaluations = project_rhythm_evidence(context)

    assert any(
        item.status == "matched" and item.statement_code == "tachycardia"
        for item in evaluations
    )
    rate = next(item for item in evaluations if item.statement_code == "tachycardia")
    assert rate.severity == "observation"


def test_pediatric_rate_uses_age_continuous_thresholds() -> None:
    def context_for(heart_rate: float) -> SimpleNamespace:
        return SimpleNamespace(
            age_years=14.0,
            global_value=lambda name: (
                heart_rate if name == "heart_rate_bpm" else None
            ),
            features=SimpleNamespace(
                metadata={
                    "rhythm_analysis": {
                        "atrial_residual": {
                            "validated_qrst_subtraction": False
                        },
                        "af_afl_summary": {
                            "probable_af": False,
                            "probable_flutter": False,
                            "rr_cv": 0.02,
                        },
                    }
                }
            ),
        )

    normal_rate = next(
        item
        for item in project_rhythm_evidence(context_for(55.7))
        if item.rule_id == "CLIN-RHYTHM-RATE-01"
    )
    bradycardia = next(
        item
        for item in project_rhythm_evidence(context_for(45.0))
        if item.rule_id == "CLIN-RHYTHM-RATE-01"
    )

    assert normal_rate.status == "not_matched"
    assert normal_rate.missing_inputs == []
    assert normal_rate.thresholds["bradycardia_lt_bpm"] == 50.0
    assert normal_rate.evidence["threshold_profile"] == "pediatric_age_continuous"
    assert bradycardia.status == "matched"
    assert bradycardia.statement_code == "bradycardia"


def test_probable_flutter_requires_validated_qrst_subtraction() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {"validated_qrst_subtraction": False},
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": True,
                    },
                }
            }
        ),
    )

    flutter = next(
        item
        for item in project_rhythm_evidence(context)
        if item.evidence.get("evaluates_code") == "atrial_flutter_pattern"
    )

    assert flutter.status == "unavailable"
    assert "rhythm.multilead_F_wave_morphology" in flutter.missing_inputs


def test_strong_multilead_flutter_candidate_gets_probable_review_tier() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {
                        "validated_qrst_subtraction": False,
                    },
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": True,
                        "F_wave_multilead_consensus": True,
                        "F_wave_confidence": 0.87,
                        "F_wave_rate_bpm": 227.0,
                        "rr_cv": 0.09,
                    },
                }
            }
        ),
    )

    flutter = next(
        item
        for item in project_rhythm_evidence(context)
        if item.evidence.get("evaluates_code") == "atrial_flutter_pattern"
    )

    assert flutter.status == "matched"
    assert flutter.statement_code == "probable_atrial_flutter_pattern"
    assert flutter.confidence == "strong_multilead_candidate_unvalidated_qrst"
    assert flutter.missing_inputs == []
    assert flutter.evidence["candidate_limitations"] == [
        "qrst_subtraction_not_validated"
    ]


def test_probable_flutter_with_validated_multilead_F_waves_is_matched() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {
                        "validated_qrst_subtraction": True,
                        "F_wave_morphology_validated": True,
                    },
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": True,
                        "F_wave_multilead_consensus": True,
                        "F_wave_confidence": 0.91,
                        "rr_cv": 0.22,
                    },
                }
            }
        ),
    )

    flutter = next(
        item
        for item in project_rhythm_evidence(context)
        if item.evidence.get("evaluates_code") == "atrial_flutter_pattern"
    )

    assert flutter.status == "matched"
    assert flutter.statement_code == "atrial_flutter_pattern"
    assert flutter.confidence == "multilead_F_wave_validated"


def test_validated_but_subthreshold_flutter_is_review_observation() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {
                        "validated_qrst_subtraction": True,
                        "F_wave_morphology_validated": True,
                    },
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": True,
                        "F_wave_multilead_consensus": True,
                        "F_wave_confidence": 0.65,
                        "F_wave_rate_bpm": 280.0,
                        "rr_cv": 0.03,
                    },
                }
            }
        ),
    )

    flutter = next(
        item
        for item in project_rhythm_evidence(context)
        if item.evidence.get("evaluates_code") == "atrial_flutter_pattern"
    )

    assert flutter.status == "matched"
    assert flutter.statement_code == "possible_atrial_flutter_pattern"
    assert flutter.confidence == "validated_subthreshold_F_wave_review"
    assert flutter.coverage == "partial"
    assert flutter.evidence["definite_F_wave_confidence_min"] == 0.80


def test_borderline_multilead_af_evidence_gets_probable_tier() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 84.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {
                        "validated_qrst_subtraction": False,
                    },
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": False,
                        "af_afl_indeterminate": False,
                        "rr_cv": 0.105,
                        "organized_p_ratio": 0.23,
                        "f_wave_multilead_consensus": True,
                        "f_wave_confidence": 0.94,
                    },
                }
            }
        ),
    )

    af = next(
        item
        for item in project_rhythm_evidence(context)
        if item.evidence.get("evaluates_code") == "atrial_fibrillation_pattern"
    )

    assert af.status == "matched"
    assert af.statement_code == "probable_atrial_fibrillation_pattern"
    assert af.confidence == "borderline_rr_p_f_wave_evidence"
    assert af.coverage == "partial"
    assert af.evidence["borderline_probable_af"] is True


def test_borderline_af_tier_requires_rr_and_f_wave_thresholds() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 84.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {
                        "validated_qrst_subtraction": True,
                    },
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": False,
                        "af_afl_indeterminate": False,
                        "rr_cv": 0.08,
                        "organized_p_ratio": 0.20,
                        "f_wave_multilead_consensus": True,
                        "f_wave_confidence": 0.89,
                    },
                }
            }
        ),
    )

    af = next(
        item
        for item in project_rhythm_evidence(context)
        if item.evidence.get("evaluates_code") == "atrial_fibrillation_pattern"
    )

    assert af.status == "not_matched"
    assert af.statement_code is None


def test_conflicting_f_and_F_wave_evidence_surfaces_indeterminate_statement() -> None:
    context = SimpleNamespace(
        age_years=50.0,
        global_value=lambda name: 80.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "atrial_residual": {"validated_qrst_subtraction": True},
                    "af_afl_summary": {
                        "probable_af": False,
                        "probable_flutter": False,
                        "af_afl_indeterminate": True,
                        "rr_cv": 0.22,
                        "f_wave_confidence": 0.82,
                        "F_wave_confidence": 0.86,
                    },
                }
            }
        ),
    )

    evaluations = project_rhythm_evidence(context)
    result = next(
        item
        for item in evaluations
        if item.evidence.get("evaluates_code")
        == "atrial_fibrillation_flutter_indeterminate"
    )

    assert result.status == "matched"
    assert result.statement_code == "atrial_fibrillation_flutter_indeterminate"
