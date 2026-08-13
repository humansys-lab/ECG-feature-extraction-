import unittest
from types import SimpleNamespace

from feature_extraction.ecgfeat.features import estimate_initial_qrs_axis_deg
from feature_extraction.ecgfeat.models import RepresentativeLeadFeatures
from feature_extraction.ecgfeat.rhythm_rules import (
    build_measurement_availability,
    build_statement_evidence,
    classify_post_pause_or_interpolated_beats,
    detect_av_block_availability_flags,
    detect_pauses_and_av_block,
    detect_preexcitation,
)
from feature_extraction.ecgfeat.rhythm_statements import build_rhythm_statement_candidates


class RhythmRuleTests(unittest.TestCase):
    def test_detect_preexcitation_reduces_delta_threshold_when_pr_is_short(self) -> None:
        result = detect_preexcitation(
            short_pr_interval=True,
            short_pr_segment=True,
            delta_leads=["I", "II"],
            mean_qrs_duration_ms=108.0,
            initial_qrs_axis_deg=-55.0,
        )

        self.assertTrue(result["wpw_pattern"])
        self.assertEqual("left", result["accessory_pathway_side"])

    def test_detect_preexcitation_requires_short_pr_evidence(self) -> None:
        result = detect_preexcitation(
            short_pr_interval=False,
            short_pr_segment=False,
            delta_leads=["I", "II", "V1"],
            mean_qrs_duration_ms=108.0,
            initial_qrs_axis_deg=-55.0,
        )

        self.assertFalse(result["wpw_pattern"])
        self.assertIsNone(result["accessory_pathway_side"])

    def test_short_pr_segment_alone_requires_strong_multilead_delta_support(self) -> None:
        weak = detect_preexcitation(
            short_pr_interval=False,
            short_pr_segment=True,
            delta_leads=["I", "II", "V1"],
            mean_qrs_duration_ms=108.0,
            initial_qrs_axis_deg=-55.0,
        )
        strong = detect_preexcitation(
            short_pr_interval=False,
            short_pr_segment=True,
            delta_leads=["I", "II", "III", "aVL", "aVF", "V3", "V4", "V5"],
            mean_qrs_duration_ms=150.0,
            initial_qrs_axis_deg=-30.0,
        )

        self.assertFalse(weak["wpw_pattern"])
        self.assertTrue(strong["wpw_pattern"])
        self.assertTrue(strong["strong_multilead_fallback"])

    def test_detect_pauses_and_av_block_flags_mobitz_one(self) -> None:
        result = detect_pauses_and_av_block(
            rr_ms=[800.0, 810.0, 1180.0],
            pr_series_ms=[180.0, 220.0, None],
            atrial_events_per_rr=[1, 1, 2],
            qrs_duration_ms=90.0,
        )

        self.assertTrue(result["pauses_detected"])
        self.assertEqual("mobitz_i", result["second_degree_avb"])

    def test_detect_pauses_and_av_block_requires_dropped_evidence_at_missing_pr(self) -> None:
        result = detect_pauses_and_av_block(
            rr_ms=[800.0, 810.0, 1180.0],
            pr_series_ms=[180.0, 220.0, None],
            atrial_events_per_rr=[2, 1, 1],
            qrs_duration_ms=90.0,
        )

        self.assertIsNone(result["second_degree_avb"])

    def test_detect_pauses_and_av_block_requires_three_rr_intervals_for_pause(self) -> None:
        result = detect_pauses_and_av_block(
            rr_ms=[800.0, 3000.0],
            pr_series_ms=[180.0, 220.0],
            atrial_events_per_rr=[1, 1],
            qrs_duration_ms=90.0,
        )

        self.assertFalse(result["pauses_detected"])
        self.assertIsNone(result["pause_longest_ms"])

    def test_second_degree_av_block_uses_p_events_exceeding_qrs_count(self) -> None:
        result = detect_pauses_and_av_block(
            rr_ms=[800.0, 1600.0, 800.0],
            pr_series_ms=[160.0, None, 162.0, 161.0],
            atrial_events_per_rr=[1, 2, 1],
            qrs_duration_ms=90.0,
        )

        self.assertIn(
            result["second_degree_avb"],
            ("mobitz_i", "mobitz_ii", "second_degree_av_block"),
        )
        self.assertEqual(
            {
                "atrial_events_per_rr": [1, 2, 1],
                "pr_series_ms": [160.0, None, 162.0, 161.0],
                "atrial_event_excess_indices": [1],
                "localized_atrial_event_excess": True,
                "constant_multiple_atrial_events_requires_validation": False,
                "dropped_p_interval_indices": [1],
                "dropped_p_evidence": True,
                "requires_prolonged_ventricular_interval": True,
            },
            result["av_block_evidence"],
        )

    def test_second_degree_av_block_uses_extra_p_events_even_when_pr_series_has_no_placeholder(self) -> None:
        result = detect_pauses_and_av_block(
            rr_ms=[800.0, 1600.0, 800.0],
            pr_series_ms=[160.0, 162.0, 161.0],
            atrial_events_per_rr=[1, 2, 1],
            qrs_duration_ms=90.0,
        )

        self.assertEqual("second_degree_av_block", result["second_degree_avb"])
        self.assertTrue(result["av_block_evidence"]["dropped_p_evidence"])

    def test_extra_atrial_event_in_normal_rr_is_not_dropped_p_evidence(self) -> None:
        result = detect_pauses_and_av_block(
            rr_ms=[800.0, 820.0, 810.0, 790.0],
            pr_series_ms=[160.0, None, 162.0, 161.0],
            atrial_events_per_rr=[1, 2, 1, 1],
            qrs_duration_ms=90.0,
        )

        self.assertIsNone(result["second_degree_avb"])
        self.assertFalse(result["av_block_evidence"]["dropped_p_evidence"])

    def test_constant_multiple_atrial_events_need_independent_two_to_one_validation(self) -> None:
        result = detect_pauses_and_av_block(
            rr_ms=[800.0, 1600.0, 810.0, 790.0],
            pr_series_ms=[160.0, None, 162.0, 161.0],
            atrial_events_per_rr=[2, 2, 2, 2],
            qrs_duration_ms=90.0,
        )

        self.assertIsNone(result["second_degree_avb"])
        self.assertFalse(result["av_block_evidence"]["dropped_p_evidence"])
        self.assertTrue(
            result["av_block_evidence"][
                "constant_multiple_atrial_events_requires_validation"
            ]
        )

    def test_classify_post_pause_or_interpolated_beats_marks_escape_and_interpolated_candidates(self) -> None:
        events = classify_post_pause_or_interpolated_beats(
            beats=[
                {"beat_id": 3, "rr_prev_ms": 1500.0, "rr_next_ms": 800.0},
                {"beat_id": 4, "rr_prev_ms": 420.0, "rr_next_ms": 570.0},
            ],
            background_rr_ms=1000.0,
        )

        self.assertIn({"beat_id": 3, "type": "escape_candidate"}, events)
        self.assertIn({"beat_id": 4, "type": "interpolated_candidate"}, events)

    def test_build_statement_evidence_bypasses_remaining_algorithm_for_prewexcitation(self) -> None:
        evidence = build_statement_evidence(
            rhythm_summary={"primary_statement": "wpw_pattern", "bypass_remaining_algorithm": True},
            pacing_context={"suppress_further_rhythm_interpretation": False},
        )

        self.assertTrue(evidence["bypass_remaining_algorithm"])
        self.assertEqual("wpw_pattern", evidence["primary_statement"])

    def test_rhythm_statement_engine_prioritizes_wpw_bypass(self) -> None:
        evidence = build_rhythm_statement_candidates(
            rhythm_summary={"rr_irregularity_class": "regular"},
            preexcitation={"wpw_pattern": True},
            pacing_context={"continuous_pacing": False},
            availability={"pr_available": True, "p_axis_available": True},
        )

        self.assertEqual("wpw_pattern", evidence["primary_statement"])
        self.assertTrue(evidence["bypass_remaining_algorithm"])

    def test_rhythm_statement_evidence_carries_measurement_availability(self) -> None:
        availability = {
            "pr_available": False,
            "p_axis_available": False,
            "reasons": ["continuous_pacing"],
        }

        evidence = build_rhythm_statement_candidates(
            rhythm_summary={"primary_statement": None},
            preexcitation={"wpw_pattern": False},
            pacing_context={},
            availability=availability,
        )

        self.assertEqual(availability, evidence["availability"])

    def test_rhythm_statement_engine_stops_when_pacing_context_suppresses_rhythm(self) -> None:
        evidence = build_rhythm_statement_candidates(
            rhythm_summary={"primary_statement": None},
            preexcitation={"wpw_pattern": False},
            pacing_context={"suppress_further_rhythm_interpretation": True},
            availability={"reasons": ["wide_qrs_pacing_like_context"]},
        )

        self.assertTrue(evidence["stop_further_interpretation"])
        self.assertFalse(evidence["bypass_remaining_algorithm"])

    def test_build_measurement_availability_reports_top_level_av_block_reasons(self) -> None:
        availability = build_measurement_availability(
            pacing_context={},
            af_afl_summary={},
            rule_summary={"complete_av_block": True, "av_dissociation": True},
        )

        self.assertFalse(availability["pr_available"])
        self.assertIn("complete_av_block", availability["reasons"])
        self.assertIn("av_dissociation", availability["reasons"])

    def test_build_measurement_availability_reports_invalid_atrial_measurements(self) -> None:
        availability = build_measurement_availability(
            pacing_context={},
            af_afl_summary={},
            rule_summary={"atrial_measurements_invalid": True},
        )

        self.assertFalse(availability["atrial_rhythm_available"])
        self.assertFalse(availability["pr_available"])
        self.assertFalse(availability["p_axis_available"])
        self.assertIn("atrial_measurements_unavailable", availability["reasons"])

    def test_raw_flutter_confidence_does_not_invalidate_atrial_measurements(self) -> None:
        availability = build_measurement_availability(
            pacing_context={},
            af_afl_summary={
                "probable_flutter": False,
                "flutter_wave_confidence": 0.99,
            },
            rule_summary={},
        )

        self.assertTrue(availability["atrial_rhythm_available"])
        self.assertNotIn("probable_flutter", availability["reasons"])

    def test_af_afl_indeterminate_invalidates_pr_measurement_when_p_disorganized(self) -> None:
        availability = build_measurement_availability(
            pacing_context={},
            af_afl_summary={"af_afl_indeterminate": True, "organized_p_ratio": 0.10},
            rule_summary={},
        )

        self.assertFalse(availability["pr_available"])
        self.assertIn("af_afl_indeterminate", availability["reasons"])

    def test_af_afl_indeterminate_keeps_pr_when_atrial_activity_is_organized(self) -> None:
        # Borderline F-wave evidence on a record whose P waves are present and
        # organized says nothing about the P-to-QRS interval; withholding PR here
        # cost 54 PTB-XL records their PR class with no AF/flutter label.
        availability = build_measurement_availability(
            pacing_context={},
            af_afl_summary={"af_afl_indeterminate": True, "organized_p_ratio": 0.92},
            rule_summary={},
        )

        self.assertFalse(availability["atrial_rhythm_available"])
        self.assertTrue(availability["pr_available"])
        self.assertIn("af_afl_indeterminate", availability["reasons"])
        self.assertEqual(["af_afl_indeterminate"], availability["pr_soft_reasons"])

    def test_pr_scatter_alone_does_not_invalidate_pr_measurement(self) -> None:
        availability = build_measurement_availability(
            pacing_context={},
            af_afl_summary={},
            rule_summary={"av_dissociation": True},
        )

        self.assertFalse(availability["atrial_rhythm_available"])
        self.assertTrue(availability["pr_available"])
        self.assertEqual(["av_dissociation"], availability["pr_soft_reasons"])

    def test_corroborated_av_dissociation_invalidates_pr_measurement(self) -> None:
        availability = build_measurement_availability(
            pacing_context={},
            af_afl_summary={},
            rule_summary={
                "av_dissociation": True,
                "atrial_faster_than_ventricular": True,
            },
        )

        self.assertFalse(availability["pr_available"])
        self.assertEqual([], availability["pr_soft_reasons"])

    def test_build_measurement_availability_preserves_p_axis_for_p_synchronous_pacing_context(self) -> None:
        availability = build_measurement_availability(
            pacing_context={"wide_qrs_pacing_like_context": True},
            af_afl_summary={},
            rule_summary={},
        )

        self.assertFalse(availability["atrial_rhythm_available"])
        self.assertFalse(availability["pr_available"])
        self.assertTrue(availability["p_axis_available"])
        self.assertIn("wide_qrs_pacing_like_context", availability["reasons"])

    def test_build_measurement_availability_masks_p_axis_when_pacing_has_av_dissociation(self) -> None:
        availability = build_measurement_availability(
            pacing_context={"wide_qrs_pacing_like_context": True},
            af_afl_summary={},
            rule_summary={"av_dissociation": True},
        )

        self.assertFalse(availability["pr_available"])
        self.assertFalse(availability["p_axis_available"])
        self.assertIn("av_dissociation", availability["reasons"])

    def test_build_measurement_availability_tolerates_missing_inputs(self) -> None:
        availability = build_measurement_availability(None, None, None)

        self.assertTrue(availability["atrial_rhythm_available"])
        self.assertTrue(availability["pr_available"])
        self.assertTrue(availability["p_axis_available"])
        self.assertEqual([], availability["reasons"])

    def test_detect_av_block_availability_flags_uses_bradycardia_threshold(self) -> None:
        # A slow ventricular rate alone is not sufficient for complete AV
        # block -- it also requires independent AV-dissociation evidence,
        # here an atrial rate distinctly faster than the ventricular rate.
        result = detect_av_block_availability_flags(
            beats=[],
            beat_features=[],
            heart_rate_bpm=44.9,
            atrial_rate_bpm=75.0,
        )

        self.assertTrue(result["complete_av_block"])
        self.assertFalse(result["av_dissociation"])
        self.assertTrue(result["atrial_faster_than_ventricular"])

    def test_detect_av_block_availability_flags_slow_rate_alone_is_not_complete_avb(self) -> None:
        # Marked sinus bradycardia (slow rate, no AV dissociation evidence)
        # must not be classified as complete AV block.
        result = detect_av_block_availability_flags(
            beats=[],
            beat_features=[],
            heart_rate_bpm=44.9,
        )

        self.assertFalse(result["complete_av_block"])
        self.assertFalse(result["av_dissociation"])
        self.assertFalse(result["atrial_faster_than_ventricular"])

    def test_detect_av_block_availability_flags_uses_per_beat_pr_variability(self) -> None:
        beats = [
            SimpleNamespace(beat_id=0),
            SimpleNamespace(beat_id=1),
            SimpleNamespace(beat_id=2),
        ]
        beat_features = [
            SimpleNamespace(beat_id=0, pr_ms=100.0),
            SimpleNamespace(beat_id=0, pr_ms=110.0),
            SimpleNamespace(beat_id=1, pr_ms=190.0),
            SimpleNamespace(beat_id=2, pr_ms=260.0),
        ]

        result = detect_av_block_availability_flags(
            beats=beats,
            beat_features=beat_features,
            heart_rate_bpm=70.0,
        )

        self.assertFalse(result["complete_av_block"])
        self.assertTrue(result["av_dissociation"])

        availability = build_measurement_availability({}, {}, result)
        self.assertIn("av_dissociation", availability["reasons"])

    def test_apply_rule_summary_does_not_downgrade_existing_rhythm_positives(self) -> None:
        from feature_extraction.ecgfeat.interpret import _apply_rule_summary_to_interpretation

        interpretation = SimpleNamespace(
            wpw_pattern=True,
            pauses_detected=True,
            pause_longest_ms=1500.0,
            second_degree_avb=None,
            avb_grade=None,
        )

        _apply_rule_summary_to_interpretation(
            interpretation,
            {
                "preexcitation": {"wpw_pattern": False},
                "pauses": {"pauses_detected": False, "pause_longest_ms": None},
            },
            {"suppress_further_rhythm_interpretation": False},
        )

        self.assertTrue(interpretation.wpw_pattern)
        self.assertTrue(interpretation.pauses_detected)
        self.assertEqual(1500.0, interpretation.pause_longest_ms)

    def test_estimate_initial_qrs_axis_handles_malformed_params(self) -> None:
        result = estimate_initial_qrs_axis_deg({
            "I": RepresentativeLeadFeatures(
                lead="I",
                params={"qrs_signed_area": "bad", "r_amp_mv": "also_bad"},
                variance={},
            ),
            "II": RepresentativeLeadFeatures(
                lead="II",
                params={"qrs_signed_area": "bad", "r_amp_mv": "also_bad"},
                variance={},
            ),
        })

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
