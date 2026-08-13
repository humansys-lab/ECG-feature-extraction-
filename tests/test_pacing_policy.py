import unittest
from types import SimpleNamespace

from feature_extraction.ecgfeat.interpret import (
    _apply_measurement_availability_to_interpretation,
    _suppress_pacing_unreliable_rhythm_claims,
)
from feature_extraction.ecgfeat.rhythm_rules import (
    assess_pacing_evidence_quality,
    classify_pacing_context,
    detect_pacing_failures,
    select_measurement_beat_ids,
)
from feature_extraction.ecgfeat.clinical_rules.pacing import evaluate_pacing


class PacingPolicyTests(unittest.TestCase):
    def test_pacing_evidence_audit_rejects_noisy_cherry_picked_alignment(self) -> None:
        result = assess_pacing_evidence_quality(
            pacing_result={"paced": True, "state": "on"},
            spike_times=list(range(73)),
            qrs_count=10,
            pacing_spike_beat_ids=[2, 3, 5, 6, 7, 9],
            spike_offsets_samples=[-18, -26, -20, -9, -29, -22],
            pacing_failures={"capture_alignment_fraction": 14 / 73},
            fs=500,
        )

        self.assertTrue(result["evidence_conflicted"])
        self.assertFalse(result["supports_measurement_routing"])
        self.assertEqual("conflicted", result["confidence_state"])
        self.assertIn("excessive_spike_burden_per_qrs", result["conflict_reasons"])
        self.assertIn("low_global_capture_alignment", result["conflict_reasons"])
        self.assertIn(
            "sparse_qrs_proximal_spike_support",
            result["conflict_reasons"],
        )

    def test_pacing_evidence_audit_accepts_repeated_stable_qrs_alignment(self) -> None:
        result = assess_pacing_evidence_quality(
            pacing_result={"paced": True, "state": "on"},
            spike_times=[100, 600, 1100, 1600],
            qrs_count=4,
            pacing_spike_beat_ids=[0, 1, 2, 3],
            spike_offsets_samples=[-20, -18, -22, -19],
            pacing_failures={"capture_alignment_fraction": 1.0},
            fs=500,
        )

        self.assertFalse(result["evidence_conflicted"])
        self.assertTrue(result["supports_measurement_routing"])
        self.assertEqual("supported", result["confidence_state"])

    def test_conflicted_pacing_context_cannot_suppress_rhythm(self) -> None:
        pacing = classify_pacing_context(
            beats=[
                {"beat_id": 0, "paced": True, "qrs_duration_ms": 160.0},
                {"beat_id": 1, "paced": True, "qrs_duration_ms": 155.0},
            ],
            atrial_events=[],
            spike_times=[100, 600],
            evidence_quality={
                "evidence_conflicted": True,
                "supports_measurement_routing": False,
                "confidence_state": "conflicted",
            },
        )

        self.assertFalse(pacing["ventricular_pacing_present"])
        self.assertFalse(pacing["suppress_further_rhythm_interpretation"])

    def test_classify_pacing_context_marks_continuous_ventricular_pacing(self) -> None:
        pacing = classify_pacing_context(
            beats=[
                {"beat_id": 0, "paced": True, "qrs_duration_ms": 160.0},
                {"beat_id": 1, "paced": True, "qrs_duration_ms": 155.0},
            ],
            atrial_events=[],
            spike_times=[100, 600],
        )

        self.assertTrue(pacing["continuous_pacing"])
        self.assertTrue(pacing["ventricular_pacing_present"])
        self.assertTrue(pacing["suppress_further_rhythm_interpretation"])

    def test_select_measurement_beat_ids_prefers_non_paced_non_ectopic_family(self) -> None:
        beat_rows = [
            {"beat_id": 0, "paced": False, "group_id": 1, "is_ventricular_ectopic": False},
            {"beat_id": 1, "paced": False, "group_id": 1, "is_ventricular_ectopic": False},
            {"beat_id": 2, "paced": True, "group_id": 2, "is_ventricular_ectopic": False},
            {"beat_id": 3, "paced": False, "group_id": 3, "is_ventricular_ectopic": True},
        ]

        self.assertEqual([0, 1], select_measurement_beat_ids(beat_rows))

    def test_select_measurement_beat_ids_prefers_paced_family_when_pacing_is_majority(self) -> None:
        beat_rows = [
            {"beat_id": 0, "paced": False, "group_id": 2, "is_ventricular_ectopic": False},
            {"beat_id": 1, "paced": True, "group_id": 1, "is_ventricular_ectopic": False},
            {"beat_id": 2, "paced": True, "group_id": 1, "is_ventricular_ectopic": False},
            {"beat_id": 3, "paced": False, "group_id": 2, "is_ventricular_ectopic": False},
            {"beat_id": 4, "paced": True, "group_id": 1, "is_ventricular_ectopic": False},
        ]

        self.assertEqual([1, 2, 4], select_measurement_beat_ids(beat_rows))

    def test_classify_pacing_context_ignores_malformed_qrs_duration(self) -> None:
        pacing = classify_pacing_context(
            beats=[
                {"beat_id": 0, "paced": True, "qrs_duration_ms": "wide"},
                {"beat_id": 1, "paced": True, "qrs_duration_ms": 130.0},
            ],
            atrial_events=[],
            spike_times=[],
        )

        self.assertTrue(pacing["ventricular_pacing_present"])
        self.assertTrue(pacing["suppress_further_rhythm_interpretation"])

    def test_retrograde_atrial_event_is_not_mislabeled_as_atrial_pacing(self) -> None:
        pacing = classify_pacing_context(
            beats=[],
            atrial_events=[{"association_type": "retrograde"}],
            spike_times=[100],
        )

        self.assertFalse(pacing["atrial_pacing_present"])
        self.assertFalse(pacing["dual_chamber_pacing_present"])

    def test_select_measurement_beat_ids_ignores_malformed_beat_id(self) -> None:
        beat_rows = [
            {"beat_id": "bad", "paced": False, "group_id": 1, "is_ventricular_ectopic": False},
            {"beat_id": "also-bad", "paced": False, "group_id": 1, "is_ventricular_ectopic": False},
            {"beat_id": 2, "paced": False, "group_id": 2, "is_ventricular_ectopic": False},
        ]

        self.assertEqual([2], select_measurement_beat_ids(beat_rows))

    def test_pacing_suppression_preserves_pacemaker_artifact_evidence(self) -> None:
        interpretation = SimpleNamespace(
            pacemaker_like_artifact=True,
            complete_av_block=True,
            av_dissociation=True,
            avb_grade=3,
            probable_af=True,
            wpw_pattern=True,
            rvh_suspected=True,
            rvh_class="probable",
        )

        _suppress_pacing_unreliable_rhythm_claims(
            interpretation,
            {"suppress_further_rhythm_interpretation": True},
        )

        self.assertTrue(interpretation.pacemaker_like_artifact)
        self.assertFalse(interpretation.complete_av_block)
        self.assertFalse(interpretation.av_dissociation)
        self.assertIsNone(interpretation.avb_grade)
        self.assertFalse(interpretation.probable_af)
        self.assertFalse(interpretation.wpw_pattern)
        self.assertFalse(interpretation.rvh_suspected)
        self.assertIsNone(interpretation.rvh_class)

    def test_measurement_availability_masks_pr_and_p_axis_without_hiding_pacing_evidence(self) -> None:
        interpretation = SimpleNamespace(
            pr_class="normal",
            avb_grade=1,
            p_axis_normal=True,
            pacemaker_like_artifact=True,
        )

        _apply_measurement_availability_to_interpretation(
            interpretation,
            {"pr_available": False, "p_axis_available": False},
        )

        self.assertEqual("indeterminate", interpretation.pr_class)
        self.assertIsNone(interpretation.avb_grade)
        self.assertIsNone(interpretation.p_axis_normal)
        self.assertTrue(interpretation.pacemaker_like_artifact)

    def test_detect_capture_failure_when_spike_not_followed_by_qrs(self) -> None:
        result = detect_pacing_failures(
            spike_times=[100, 600, 1100, 1600],
            qrs_times=[130, 630],
            fs=500,
        )

        self.assertTrue(result["capture_failure_suspected"])
        self.assertEqual([1100, 1600], result["failure_spike_times"])
        self.assertEqual(4, result["spike_count"])
        self.assertEqual(2, result["captured_spike_count"])
        self.assertEqual(0.5, result["capture_failure_fraction"])

    def test_single_artifact_spike_cannot_trigger_capture_failure(self) -> None:
        result = detect_pacing_failures(
            spike_times=[600],
            qrs_times=[130, 1130],
            fs=500,
        )

        self.assertFalse(result["capture_failure_suspected"])
        self.assertEqual([600], result["failure_spike_times"])
        self.assertFalse(result["minimum_evidence_met"])

    def test_detect_pacing_failures_keeps_sensing_failure_unavailable_until_implemented(self) -> None:
        result = detect_pacing_failures(
            spike_times=[100],
            qrs_times=[130],
            fs=500,
        )

        self.assertEqual(
            {
                "available": False,
                "value": None,
                "reason": "sensing_failure_detector_not_implemented",
            },
            result["sensing_failure_suspected"],
        )

    def test_unavailable_sensing_detector_is_not_treated_as_true(self) -> None:
        context = SimpleNamespace(
            features=SimpleNamespace(
                global_features=SimpleNamespace(
                    pacing_spikes=[100],
                    paced_rhythm=False,
                ),
                metadata={
                    "measurement_pacing_state": "on",
                    "paced_beat_fraction": 0.20,
                    "rhythm_analysis": {
                        "pacing_context": {},
                        "pacing_failures": {
                            "capture_failure_suspected": False,
                            "sensing_failure_suspected": {
                                "available": False,
                                "value": None,
                                "reason": "not_implemented",
                            },
                        },
                    },
                },
            )
        )

        result = next(
            row for row in evaluate_pacing(context)
            if row.evidence["evaluates_code"]
            == "pacing_sensing_failure_suspected"
        )

        self.assertEqual("unavailable", result.status)
        self.assertIsNone(result.statement_code)

    def test_capture_alert_requires_established_recurrent_pacing_context(self) -> None:
        context = SimpleNamespace(
            features=SimpleNamespace(
                global_features=SimpleNamespace(
                    pacing_spikes=[600],
                    paced_rhythm=False,
                ),
                metadata={
                    "measurement_pacing_state": "unknown",
                    "paced_beat_fraction": 0.0,
                    "rhythm_analysis": {
                        "pacing_context": {
                            "spike_count": 1,
                            "paced_fraction": 0.0,
                            "ventricular_pacing_present": False,
                        },
                        "pacing_failures": {
                            "capture_failure_suspected": True,
                            "failure_spike_times": [600],
                            "spike_count": 1,
                            "capture_failure_count": 1,
                            "capture_failure_fraction": 1.0,
                            "minimum_evidence_met": False,
                            "sensing_failure_suspected": {
                                "available": False,
                                "value": None,
                            },
                        },
                    },
                },
            )
        )

        result = next(
            row for row in evaluate_pacing(context)
            if row.evidence["evaluates_code"]
            == "pacing_failure_to_capture_suspected"
        )

        self.assertEqual("indeterminate", result.status)
        self.assertIsNone(result.statement_code)
        self.assertIn("recurrent_confirmed_pacing_context", result.missing_inputs)

    def test_intermitent_pacing_requires_capture_alignment_support(self) -> None:
        context = SimpleNamespace(
            features=SimpleNamespace(
                global_features=SimpleNamespace(
                    pacing_spikes=list(range(20)),
                    paced_rhythm=True,
                ),
                metadata={
                    "measurement_pacing_state": "on",
                    "paced_beat_fraction": 0.55,
                    "rhythm_analysis": {
                        "pacing_context": {
                            "spike_count": 20,
                            "paced_fraction": 0.55,
                            "ventricular_pacing_present": True,
                        },
                        "pacing_failures": {
                            "capture_failure_suspected": True,
                            "failure_spike_times": list(range(16)),
                            "spike_count": 20,
                            "captured_spike_count": 4,
                            "capture_failure_count": 16,
                            "capture_failure_fraction": 0.8,
                            "minimum_evidence_met": True,
                            "sensing_failure_suspected": {
                                "available": False,
                                "value": None,
                            },
                        },
                    },
                },
            )
        )

        result = next(
            row for row in evaluate_pacing(context)
            if row.evidence["evaluates_code"] == "intermittent_pacing"
        )

        self.assertEqual("indeterminate", result.status)
        self.assertIsNone(result.statement_code)
