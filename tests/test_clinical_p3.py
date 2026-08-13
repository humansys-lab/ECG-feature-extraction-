from __future__ import annotations

from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.av_block import (
    evaluate_advanced_av_block,
)
from feature_extraction.ecgfeat.clinical_rules.ectopy import evaluate_ectopy
from feature_extraction.ecgfeat.clinical_rules.engine import analyze_clinical
from feature_extraction.ecgfeat.clinical_rules.models import RuleEvaluation
from feature_extraction.ecgfeat.clinical_rules.preexcitation import (
    evaluate_preexcitation,
)
from feature_extraction.ecgfeat.clinical_rules.repolarization import (
    evaluate_t_wave_abnormalities,
)
from feature_extraction.ecgfeat.clinical_rules.voltage import evaluate_low_voltage
from tests.test_clinical_export_consumers import _features


class LeadContext:
    def __init__(self, leads, *, age=50.0, metadata=None):
        self._leads = leads
        self.age_years = age
        self.features = SimpleNamespace(metadata=metadata or {})

    def lead_value(self, lead, name, reliability="reliable_for_qrs"):
        return self._leads.get(lead, {}).get(name)


def _by_code(rows, code):
    return next(
        row
        for row in rows
        if row.statement_code == code or row.evidence.get("evaluates_code") == code
    )


def test_low_voltage_requires_all_leads_for_positive_but_one_lead_can_disprove() -> None:
    leads = {
        lead: {"q_amp_mv": 0.0, "r_amp_mv": 0.20, "s_amp_mv": -0.20}
        for lead in ("I", "II", "III", "aVR", "aVL", "aVF")
    }
    context = LeadContext(leads)

    limb = _by_code(evaluate_low_voltage(context), "low_qrs_voltage_limb_leads")
    assert limb.status == "matched"

    del leads["III"]
    leads["II"]["r_amp_mv"] = 0.40
    leads["II"]["s_amp_mv"] = -0.20
    limb = _by_code(evaluate_low_voltage(context), "low_qrs_voltage_limb_leads")
    assert limb.status == "not_matched"


def test_preexcitation_projects_validated_summary_as_unified_screen() -> None:
    context = SimpleNamespace(
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "rule_summary": {
                        "preexcitation": {
                            "wpw_pattern": True,
                            "short_pr_interval": True,
                            "short_pr_segment": False,
                            "delta_leads": ["I", "II"],
                            "mean_qrs_duration_ms": 112.0,
                        }
                    },
                    "pacing_context": {},
                }
            },
            beat_features=[SimpleNamespace()],
            quality={
                lead: SimpleNamespace(reliable_for_qrs=True)
                for lead in ("I", "II", "V1")
            },
        )
    )

    result = evaluate_preexcitation(context)

    assert result.status == "matched"
    assert result.statement_code == "ventricular_preexcitation_pattern"
    assert result.normality_role == "optional_screen"


def test_preexcitation_negative_needs_measured_pr_or_pr_segment() -> None:
    context = SimpleNamespace(
        global_value=lambda name: None,
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "rule_summary": {
                        "preexcitation": {
                            "wpw_pattern": False,
                            "short_pr_interval": False,
                            "short_pr_segment": False,
                            "delta_leads": [],
                            "mean_qrs_duration_ms": 90.0,
                            "pr_segment_ms": None,
                        }
                    },
                    "pacing_context": {},
                }
            },
            beat_features=[SimpleNamespace()],
            quality={
                lead: SimpleNamespace(reliable_for_qrs=True)
                for lead in ("I", "II", "V1")
            },
        ),
    )

    result = evaluate_preexcitation(context)

    assert result.status == "unavailable"
    assert "global.pr_ms_or_pr_segment_ms" in result.missing_inputs


def test_short_pr_segment_alone_is_not_definite_preexcitation() -> None:
    context = SimpleNamespace(
        global_value=lambda name: {
            "pr_ms": 130.0,
            "qrs_ms": 108.0,
        }.get(name),
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "rule_summary": {
                        "preexcitation": {
                            "wpw_pattern": True,
                            "short_pr_interval": False,
                            "short_pr_segment": True,
                            "delta_leads": ["I", "II", "V2"],
                            "delta_beat_ids": [0, 1, 2, 3],
                            "mean_qrs_duration_ms": 108.0,
                        }
                    },
                    "pacing_context": {},
                }
            },
            beat_features=[SimpleNamespace()],
            quality={
                lead: SimpleNamespace(reliable_for_qrs=True)
                for lead in ("I", "II", "V1")
            },
        ),
    )

    result = evaluate_preexcitation(context)

    assert result.status == "indeterminate"
    assert result.statement_code is None
    assert "explicit_short_pr_or_strong_multilead_delta_evidence" in result.missing_inputs


def test_strong_multilead_delta_support_can_rescue_unmeasurable_pr() -> None:
    context = SimpleNamespace(
        global_value=lambda name: {
            "pr_ms": None,
            "qrs_ms": 150.0,
        }.get(name),
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "rule_summary": {
                        "preexcitation": {
                            "wpw_pattern": True,
                            "short_pr_interval": False,
                            "short_pr_segment": True,
                            "delta_leads": [
                                "I", "II", "III", "aVL",
                                "aVF", "V3", "V4", "V5",
                            ],
                            "delta_beat_ids": list(range(8)),
                            "mean_qrs_duration_ms": 150.0,
                        }
                    },
                    "pacing_context": {
                        "wide_qrs_pacing_like_context": True,
                        "suppress_further_rhythm_interpretation": True,
                    },
                }
            },
            beat_features=[SimpleNamespace()],
            quality={
                lead: SimpleNamespace(reliable_for_qrs=True)
                for lead in ("I", "II", "V1")
            },
        ),
    )

    result = evaluate_preexcitation(context)

    assert result.status == "matched"
    assert result.statement_code == "ventricular_preexcitation_pattern"
    assert result.evidence["strong_multilead_fallback"] is True


def _ectopy_context(*, qrs_ms=90.0, probable_af=False):
    beats = [
        SimpleNamespace(beat_id=0, rr_prev_ms=None, rr_next_ms=1000.0, group_id=1, paced=False),
        SimpleNamespace(beat_id=1, rr_prev_ms=1000.0, rr_next_ms=700.0, group_id=1, paced=False),
        SimpleNamespace(beat_id=2, rr_prev_ms=700.0, rr_next_ms=1300.0, group_id=2, paced=False),
        SimpleNamespace(beat_id=3, rr_prev_ms=1300.0, rr_next_ms=None, group_id=1, paced=False),
    ]
    beat_features = []
    for beat in beats:
        for lead in ("II", "V1"):
            beat_features.append(
                SimpleNamespace(
                    beat_id=beat.beat_id,
                    qrs_ms=qrs_ms if beat.beat_id == 2 else 90.0,
                    p_confidence=0.8,
                    beat_measurement_reliable=True,
                    lead=lead,
                )
            )
    return SimpleNamespace(
        features=SimpleNamespace(
            beats=beats,
            beat_features=beat_features,
            metadata={
                "rhythm_analysis": {
                    "af_afl_summary": {
                        "probable_af": probable_af,
                        "probable_flutter": False,
                    },
                    "pacing_context": {},
                }
            },
        )
    )


def test_ectopy_exports_structured_pac_burden_and_af_makes_pac_not_applicable() -> None:
    pac = _by_code(evaluate_ectopy(_ectopy_context()), "premature_atrial_complexes")
    assert pac.status == "matched"
    assert pac.evidence["count"] == 1
    assert pac.evidence["candidate_beats"][0]["beat_id"] == 2
    assert round(pac.evidence["burden_percent"], 1) == 33.3

    pac_during_af = _by_code(
        evaluate_ectopy(_ectopy_context(probable_af=True)),
        "premature_atrial_complexes",
    )
    assert pac_during_af.status == "not_applicable"
    assert pac_during_af.evidence["not_applicable_by"] == "atrial_fibrillation_pattern"


def test_wide_premature_beat_is_projected_as_pvc() -> None:
    pvc = _by_code(
        evaluate_ectopy(_ectopy_context(qrs_ms=140.0)),
        "premature_ventricular_complexes",
    )
    assert pvc.status == "matched"
    assert pvc.evidence["candidate_beats"][0]["compensatory_pause_support"] is True


def test_strict_relative_widening_and_full_pause_yields_probable_pvc() -> None:
    context = _ectopy_context(qrs_ms=90.0)
    leads = ("I", "II", "III", "aVR", "aVL", "V1")
    candidate_widths = (80.0, 90.0, 100.0, 110.0, 130.0, 150.0)
    beat_features = []
    for beat in context.features.beats:
        for index, lead in enumerate(leads):
            beat_features.append(
                SimpleNamespace(
                    beat_id=beat.beat_id,
                    qrs_ms=(
                        candidate_widths[index]
                        if beat.beat_id == 2
                        else 70.0
                    ),
                    p_confidence=0.9,
                    beat_measurement_reliable=True,
                    lead=lead,
                )
            )
    context.features.beat_features = beat_features

    evaluations = evaluate_ectopy(context)
    pvc = _by_code(evaluations, "premature_ventricular_complexes")
    pac = _by_code(evaluations, "premature_atrial_complexes")

    assert pvc.status == "matched"
    assert pvc.statement_code == "probable_premature_ventricular_complexes"
    assert pvc.confidence == "probable_strict_relative_morphology"
    assert pvc.evidence["candidate_beats"][0]["wide_qrs_lead_count"] == 2
    assert pac.status == "not_matched"


def test_strict_probable_pvc_accepts_borderline_near_full_compensation() -> None:
    context = _ectopy_context(qrs_ms=90.0)
    context.features.beats[2].rr_next_ms = 1550.0
    context.features.beats[3].rr_prev_ms = 1550.0
    leads = ("I", "II", "III", "aVR", "aVL", "V1")
    candidate_widths = (80.0, 90.0, 100.0, 110.0, 130.0, 150.0)
    context.features.beat_features = [
        SimpleNamespace(
            beat_id=beat.beat_id,
            qrs_ms=(
                candidate_widths[index]
                if beat.beat_id == 2
                else 70.0
            ),
            p_confidence=0.9,
            beat_measurement_reliable=True,
            lead=lead,
        )
        for beat in context.features.beats
        for index, lead in enumerate(leads)
    ]

    evaluations = evaluate_ectopy(context)
    pvc = _by_code(evaluations, "premature_ventricular_complexes")
    pac = _by_code(evaluations, "premature_atrial_complexes")

    assert pvc.status == "matched"
    candidate = pvc.evidence["candidate_beats"][0]
    assert candidate["compensatory_pause_support"] is False
    assert candidate["compensatory_pause_ratio"] == 2.25
    assert (
        candidate["classification_basis"]
        == "strict_relative_widening_with_near_full_compensation"
    )


def test_unconfirmed_pacing_like_context_retains_pvc_as_confounder() -> None:
    context = _ectopy_context(qrs_ms=140.0)
    context.features.metadata["rhythm_analysis"]["pacing_context"] = {
        "wide_qrs_pacing_like_context": True,
        "suppress_further_rhythm_interpretation": True,
        "continuous_pacing": False,
        "ventricular_pacing_present": False,
    }

    evaluations = evaluate_ectopy(context)
    pvc = _by_code(evaluations, "premature_ventricular_complexes")
    pac = _by_code(evaluations, "premature_atrial_complexes")

    assert pvc.status == "matched"
    assert pvc.confidence == "low"
    assert pvc.coverage == "partial"
    assert (
        "unconfirmed_wide_qrs_pacing_like_context"
        in pvc.evidence["confounders"]
    )
    assert pac.status == "unavailable"
    assert "nonpaced_atrial_interpretation_available" in pac.missing_inputs


def test_confirmed_continuous_ventricular_pacing_still_stops_pvc() -> None:
    context = _ectopy_context(qrs_ms=140.0)
    context.features.metadata["rhythm_analysis"]["pacing_context"] = {
        "continuous_pacing": True,
        "ventricular_pacing_present": True,
        "suppress_further_rhythm_interpretation": True,
    }

    pvc = _by_code(
        evaluate_ectopy(context), "premature_ventricular_complexes"
    )

    assert pvc.status == "unavailable"
    assert "nonpaced_rhythm_interpretation_available" in pvc.missing_inputs


def test_advanced_av_block_requires_dropped_p_and_invalidates_during_af() -> None:
    metadata = {
        "rhythm_analysis": {
            "af_afl_summary": {"probable_af": False, "probable_flutter": False},
            "pacing_context": {},
            "rule_summary": {
                "complete_av_block": False,
                "av_dissociation": False,
                "atrial_faster_than_ventricular": False,
                "pauses": {
                    "second_degree_avb": "mobitz_i",
                    "av_block_evidence": {"dropped_p_evidence": True},
                },
            },
        }
    }
    context = SimpleNamespace(features=SimpleNamespace(metadata=metadata))
    second = _by_code(
        evaluate_advanced_av_block(context), "second_degree_av_block_pattern"
    )
    assert second.status == "matched"

    metadata["rhythm_analysis"]["af_afl_summary"]["probable_af"] = True
    second = _by_code(
        evaluate_advanced_av_block(context), "second_degree_av_block_pattern"
    )
    assert second.status == "not_applicable"


def test_advanced_av_block_is_suppressed_when_pause_has_ectopy_confounder() -> None:
    metadata = {
        "rhythm_analysis": {
            "af_afl_summary": {},
            "pacing_context": {},
            "rule_summary": {
                "complete_av_block": False,
                "av_dissociation": False,
                "atrial_faster_than_ventricular": False,
                "pauses": {
                    "second_degree_avb": "second_degree_av_block",
                    "av_block_evidence": {"dropped_p_evidence": True},
                },
            },
        }
    }
    context = SimpleNamespace(
        features=SimpleNamespace(
            metadata=metadata,
            interpretation=SimpleNamespace(
                premature_complexes=["VPC×1"],
                bigeminy=None,
                trigeminy=False,
                non_sustained_vt=False,
            ),
        )
    )

    second = _by_code(
        evaluate_advanced_av_block(context), "second_degree_av_block_pattern"
    )

    assert second.status == "suppressed"
    assert "premature_complexes" in second.suppressed_by


def test_second_degree_av_block_is_not_applicable_in_fast_indeterminate_af_afl() -> None:
    metadata = {
        "rhythm_analysis": {
            "af_afl_summary": {
                "probable_af": False,
                "probable_flutter": False,
                "af_afl_indeterminate": True,
            },
            "pacing_context": {},
            "rule_summary": {
                "complete_av_block": False,
                "av_dissociation": False,
                "atrial_faster_than_ventricular": False,
                "pauses": {
                    "second_degree_avb": "second_degree_av_block",
                    "av_block_evidence": {"dropped_p_evidence": True},
                },
            },
        }
    }
    context = SimpleNamespace(
        global_value=lambda name: 130.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(metadata=metadata),
    )

    second = _by_code(
        evaluate_advanced_av_block(context), "second_degree_av_block_pattern"
    )

    assert second.status == "not_applicable"
    assert second.evidence["not_applicable_by"] == "af_afl_indeterminate"


def test_reference_only_flutter_candidate_does_not_hide_pac_or_av_block() -> None:
    context = _ectopy_context()
    rhythm = context.features.metadata["rhythm_analysis"]
    rhythm["af_afl_summary"].update(
        {"probable_af": False, "probable_flutter": True}
    )
    rhythm["atrial_residual"] = {"validated_qrst_subtraction": True}

    pac = _by_code(evaluate_ectopy(context), "premature_atrial_complexes")

    assert pac.status == "matched"
    assert pac.confidence == "low"
    assert pac.coverage == "partial"
    assert "reference_only_atrial_flutter_candidate" in pac.evidence["confounders"]


def test_complete_av_block_requires_independent_av_relation_evidence() -> None:
    metadata = {
        "rhythm_analysis": {
            "af_afl_summary": {},
            "pacing_context": {},
            "rule_summary": {
                "complete_av_block": True,
                "av_dissociation": True,
                "atrial_faster_than_ventricular": False,
                "pauses": {"av_block_evidence": {"dropped_p_evidence": False}},
            },
        }
    }
    context = SimpleNamespace(features=SimpleNamespace(metadata=metadata))
    complete = _by_code(
        evaluate_advanced_av_block(context), "complete_av_block_pattern"
    )
    assert complete.status == "matched"


def test_persistent_av_dissociation_is_indeterminate_without_faster_atrial_rate() -> None:
    metadata = {
        "rhythm_analysis": {
            "af_afl_summary": {},
            "pacing_context": {},
            "rule_summary": {
                "complete_av_block": False,
                "av_dissociation": True,
                "atrial_faster_than_ventricular": False,
                "pauses": {
                    "second_degree_avb": None,
                    "av_block_evidence": {
                        "atrial_events_per_rr": [1, 2, 2, 2, 2, 2, 3, 2, 2],
                        "pr_series_ms": [
                            315.0, 284.0, 193.0, 262.0, 263.0,
                            283.0, 308.0, 278.0, 308.0, 267.0,
                        ],
                        "localized_atrial_event_excess": True,
                        "dropped_p_evidence": False,
                    },
                },
            },
        }
    }
    context = SimpleNamespace(
        global_value=lambda name: 63.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(metadata=metadata),
    )

    complete = _by_code(
        evaluate_advanced_av_block(context), "complete_av_block_pattern"
    )

    assert complete.status == "indeterminate"
    assert complete.evidence["persistent_av_dissociation_candidate"] is True
    assert complete.evidence["atrial_event_excess_fraction"] > 0.80
    assert "validated_atrial_rate_faster_than_ventricular" in complete.missing_inputs


def test_constant_extra_atrial_events_do_not_create_complete_av_block_candidate() -> None:
    metadata = {
        "rhythm_analysis": {
            "af_afl_summary": {},
            "pacing_context": {},
            "rule_summary": {
                "complete_av_block": False,
                "av_dissociation": True,
                "atrial_faster_than_ventricular": False,
                "pauses": {
                    "second_degree_avb": None,
                    "av_block_evidence": {
                        "atrial_events_per_rr": [2, 2, 2, 2, 2, 2, 2, 2],
                        "pr_series_ms": [
                            310.0, 190.0, 280.0, 210.0,
                            300.0, 205.0, 295.0, 200.0,
                        ],
                        "localized_atrial_event_excess": False,
                        "dropped_p_evidence": False,
                    },
                },
            },
        }
    }
    context = SimpleNamespace(
        global_value=lambda name: 56.0 if name == "heart_rate_bpm" else None,
        features=SimpleNamespace(metadata=metadata),
    )

    complete = _by_code(
        evaluate_advanced_av_block(context), "complete_av_block_pattern"
    )

    assert complete.status == "not_matched"
    assert complete.evidence["persistent_av_dissociation_candidate"] is False


def test_engine_suppresses_sinus_statement_during_unresolved_complete_av_block() -> None:
    features = _features(heart_rate=63.0)
    features.metadata["rhythm_analysis"].update(
        {
            "pacing_context": {},
            "rule_summary": {
                "complete_av_block": False,
                "av_dissociation": True,
                "atrial_faster_than_ventricular": False,
                "pauses": {
                    "second_degree_avb": None,
                    "av_block_evidence": {
                        "atrial_events_per_rr": [1, 2, 2, 2, 2, 2, 3, 2, 2],
                        "pr_series_ms": [
                            315.0, 284.0, 193.0, 262.0, 263.0,
                            283.0, 308.0, 278.0, 308.0, 267.0,
                        ],
                        "localized_atrial_event_excess": True,
                        "dropped_p_evidence": False,
                    },
                },
            },
        }
    )

    result = analyze_clinical(features)

    assert not any(
        row.get("statement_code") == "sinus_rhythm"
        for row in result.final_statements
    )
    assert not any(
        row.get("statement_code") in {
            "first_degree_av_delay",
            "possible_first_degree_av_delay",
        }
        for row in result.final_statements
    )
    assert any(
        row.get("statement_code") == "sinus_rhythm"
        and "possible_complete_av_block_pattern" in row.get("suppressed_by", [])
        for row in result.suppressed_statements
    )
    complete = next(
        row
        for row in result.domains["advanced_av_block"]
        if row["evidence"].get("evaluates_code") == "complete_av_block_pattern"
    )
    assert complete["status"] == "indeterminate"


def _t_context():
    leads = {
        lead: {"t_amp_mv": 0.20, "q_amp_mv": 0.0, "r_amp_mv": 0.5, "s_amp_mv": -0.5}
        for lead in ("I", "II", "III", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6")
    }
    leads["V3"]["t_amp_mv"] = -0.20
    leads["V4"]["t_amp_mv"] = -0.25
    return LeadContext(leads, metadata={"rhythm_analysis": {"pacing_context": {}}})


def _no_preexcitation():
    return RuleEvaluation(
        rule_id="TEST-PREEXCITATION",
        domain="preexcitation",
        status="not_matched",
    )


def test_t_wave_rule_distinguishes_primary_and_secondary_patterns() -> None:
    primary = evaluate_t_wave_abnormalities(
        _t_context(), conduction=[], hypertrophy=[], preexcitation=_no_preexcitation()
    )
    assert primary.status == "matched"
    assert primary.statement_code == "primary_t_wave_abnormality"
    assert primary.evidence["inverted_contiguous_pairs"] == [["V3", "V4"]]

    rbbb = RuleEvaluation(
        rule_id="TEST-RBBB",
        domain="conduction",
        status="matched",
        statement_code="rbbb_pattern",
        evidence={"evaluates_code": "rbbb_pattern"},
    )
    secondary = evaluate_t_wave_abnormalities(
        _t_context(), conduction=[rbbb], hypertrophy=[], preexcitation=_no_preexcitation()
    )
    assert secondary.statement_code == "secondary_t_wave_abnormality"
    assert "rbbb_pattern" in secondary.evidence["secondary_by"]


def test_engine_exposes_p3_capability_coverage_and_wpw_suppression() -> None:
    features = _features(qt_ms=400.0)
    features.representative_leads["aVL"].params["r_amp_mv"] = 1.5
    features.representative_leads["V3"].params["s_amp_mv"] = -1.5
    features.metadata["rhythm_analysis"].update(
        {
            "rule_summary": {
                "preexcitation": {
                    "wpw_pattern": True,
                    "short_pr_interval": True,
                    "short_pr_segment": False,
                    "delta_leads": ["I", "II"],
                    "mean_qrs_duration_ms": 110.0,
                },
                "pauses": {"av_block_evidence": {"dropped_p_evidence": False}},
                "complete_av_block": False,
                "av_dissociation": False,
            },
            "pacing_context": {},
        }
    )

    result = analyze_clinical(features)

    assert any(
        row["statement_code"] == "ventricular_preexcitation_pattern"
        for row in result.final_statements
    )
    assert not any(
        row["statement_code"] == "lvh_voltage_criteria"
        for row in result.final_statements
    )
    assert any(
        row["statement_code"] == "lvh_voltage_criteria"
        and "ventricular_preexcitation_pattern" in row["suppressed_by"]
        for row in result.suppressed_statements
    )
    for domain in ("ectopy", "preexcitation", "advanced_av_block", "repolarization", "voltage"):
        assert domain in result.domains
        assert domain in result.summary["capability_coverage"]
