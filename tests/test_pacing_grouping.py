from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from feature_extraction.ecgfeat.api import (
    ECGFeatureExtractor,
    _pacing_qrs_rescue_evidence,
)
from feature_extraction.ecgfeat.quality import validate_pacing_spikes_against_qrs
from feature_extraction.ecgfeat.grouping import cluster_beats
from feature_extraction.ecgfeat.quality import detect_pacing_spikes, remove_pacing_spikes


class PacingGroupingTests(unittest.TestCase):
    def test_pacing_qrs_rescue_recovers_regular_pre_despike_beats(self) -> None:
        evidence = _pacing_qrs_rescue_evidence(
            pacing_result={
                "spike_times": [100, 600, 1100, 1600, 2100, 2600, 4100, 4600],
                "paced": True,
                "state": "on",
            },
            pre_despike_qrs_result=SimpleNamespace(
                r_locs=[105, 605, 1105, 1605, 2105, 2605, 3105, 3605, 4105, 4605],
            ),
            despiked_qrs_result=SimpleNamespace(
                r_locs=[125, 625, 2125, 3625, 4125],
            ),
            fs=500,
        )

        self.assertTrue(evidence["applied"])
        self.assertEqual("despiking_removed_regular_pacing_aligned_qrs", evidence["reason"])
        self.assertEqual(10, evidence["pre_despike_n_beats"])
        self.assertEqual(5, evidence["despiked_n_beats"])
        self.assertEqual(8, evidence["capture_beats"])
        self.assertAlmostEqual(0.8, evidence["pre_despike_capture_fraction"])
        self.assertAlmostEqual(1.0, evidence["spike_capture_fraction"])

    def test_pacing_qrs_rescue_rejects_irregular_pre_despike_candidates(self) -> None:
        evidence = _pacing_qrs_rescue_evidence(
            pacing_result={
                "spike_times": [100, 600, 1100, 1600, 2100, 2600],
                "paced": True,
                "state": "on",
            },
            pre_despike_qrs_result=SimpleNamespace(
                r_locs=[105, 605, 1105, 2105, 2605, 4105],
            ),
            despiked_qrs_result=SimpleNamespace(r_locs=[125, 1125, 2125]),
            fs=500,
        )

        self.assertFalse(evidence["applied"])
        self.assertGreater(
            evidence["pre_despike_max_relative_rr_deviation"],
            0.35,
        )

    def test_detect_pacing_spikes_reports_off_when_no_spikes(self) -> None:
        result = detect_pacing_spikes(np.zeros((12, 1000), dtype=float), fs=500)

        self.assertEqual([], result["spike_times"])
        self.assertFalse(result["paced"])
        self.assertEqual("off", result["state"])
        self.assertEqual(0, result["lead_vote_count"])

    def test_detect_pacing_spikes_reports_unknown_when_spikes_not_regular(self) -> None:
        ecg = np.zeros((12, 1000), dtype=float)
        for lead in range(4):
            ecg[lead, [100, 400]] = 10.0

        with patch("feature_extraction.ecgfeat.quality.highpass_filter", side_effect=lambda x, fs, cutoff_hz: x):
            result = detect_pacing_spikes(ecg, fs=500)

        self.assertEqual([100, 400], result["spike_times"])
        self.assertFalse(result["paced"])
        self.assertEqual("unknown", result["state"])
        self.assertEqual(4, result["lead_vote_count"])

    def test_detect_pacing_spikes_reports_on_for_regular_multilead_spikes(self) -> None:
        ecg = np.zeros((12, 1200), dtype=float)
        for lead in range(4):
            ecg[lead, [100, 300, 500]] = 10.0

        with patch("feature_extraction.ecgfeat.quality.highpass_filter", side_effect=lambda x, fs, cutoff_hz: x):
            result = detect_pacing_spikes(ecg, fs=500)

        self.assertEqual([100, 300, 500], result["spike_times"])
        self.assertTrue(result["paced"])
        self.assertEqual("on", result["state"])
        self.assertEqual(4, result["lead_vote_count"])

    def test_detect_pacing_spikes_uses_mad_threshold_for_low_amplitude_multilead_spikes(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1600), dtype=float)
        spike_times = [200, 600, 1000, 1400]
        rng = np.random.default_rng(123)
        ecg += rng.normal(0.0, 0.0008, size=ecg.shape)
        # 12SL low-spike tier: amplitude above the 250 microvolt absolute gate but
        # still small relative to a QRS, exercising the MAD-based relative path.
        for lead in range(4):
            for s in spike_times:
                ecg[lead, s] += 0.30

        result = detect_pacing_spikes(ecg, fs=fs, min_lead_votes=4)

        self.assertEqual(spike_times, result["spike_times"])
        self.assertTrue(result["paced"])
        self.assertEqual("on", result["state"])
        self.assertGreaterEqual(result["lead_vote_count"], 4)

    def test_detect_pacing_spikes_rejects_broad_multilead_qrs_like_transients(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1600), dtype=float)
        event_times = [200, 600, 1000, 1400]
        broad = np.hanning(9) * 0.08
        for lead in range(6):
            for s in event_times:
                ecg[lead, s - 4:s + 5] += broad

        result = detect_pacing_spikes(ecg, fs=fs, min_lead_votes=4)

        self.assertEqual([], result["spike_times"])
        self.assertFalse(result["paced"])
        self.assertEqual("off", result["state"])

    def test_remove_pacing_spikes_interpolates_narrow_spike_region(self) -> None:
        ecg = np.zeros((12, 500), dtype=float)
        ecg[:, 100] = 10.0
        ecg[:, 99] = 5.0
        ecg[:, 101] = 5.0

        cleaned = remove_pacing_spikes(ecg, spike_times=[100], fs=500, half_width_ms=4.0)

        self.assertTrue(np.all(np.abs(cleaned[:, 99:102]) < 0.1))
        self.assertTrue(np.all(ecg[:, 99:102] > cleaned[:, 99:102]))

    def test_validate_pacing_spikes_rejects_qrs_edge_artifacts(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1200), dtype=float)
        r_locs = np.asarray([200, 500, 800], dtype=int)
        for r in r_locs:
            for lead in range(8):
                ecg[lead, r - 2 : r + 3] = np.linspace(0.0, 1.0, 5)
                ecg[lead, r + 3 : r + 8] = np.linspace(1.0, 0.0, 5)

        result = validate_pacing_spikes_against_qrs(
            ecg,
            fs=fs,
            spike_times=[int(r - 2) for r in r_locs],
            r_locs=r_locs,
            pacing_result={"paced": True, "state": "on", "lead_vote_count": 8},
        )

        self.assertEqual([], result["spike_times"])
        self.assertFalse(result["paced"])
        self.assertEqual("off", result["state"])
        self.assertIn("qrs_edge_artifact", result["rejection_reason"])

    def test_validate_pacing_spikes_rejects_fixed_qrs_sync_cluster_despite_narrow_dominant_escape(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1800), dtype=float)
        r_locs = np.asarray([250, 550, 850, 1150, 1450], dtype=int)
        spike_times = [int(r - 10) for r in r_locs]
        for beat_idx, (r, spike) in enumerate(zip(r_locs, spike_times)):
            for lead in range(8):
                ecg[lead, r - 3 : r + 4] += np.linspace(0.0, 0.18, 7)
                ecg[lead, r + 4 : r + 11] += np.linspace(0.18, 0.0, 7)
                if beat_idx < 4:
                    ecg[lead, spike] += 1.4

        result = validate_pacing_spikes_against_qrs(
            ecg,
            fs=fs,
            spike_times=spike_times,
            r_locs=r_locs,
            pacing_result={"paced": True, "state": "on", "lead_vote_count": 8},
        )

        self.assertEqual([], result["spike_times"])
        self.assertFalse(result["paced"])
        self.assertEqual("off", result["state"])
        self.assertEqual("qrs_edge_artifact", result["rejection_reason"])

    def test_validate_pacing_spikes_removes_rejected_qrs_edges_before_recomputing_paced_state(self) -> None:
        fs = 500
        ecg = np.zeros((12, 3400), dtype=float)
        r_locs = np.asarray([250, 550, 850, 1150, 1450, 1750, 2050, 2350, 2650, 2950], dtype=int)
        qrs_edge_spikes = [int(r - 10) for r in r_locs[:5]]
        isolated_spike = 3300
        spike_times = qrs_edge_spikes + [isolated_spike]
        for spike, r in zip(qrs_edge_spikes, r_locs[:5]):
            for lead in range(8):
                ecg[lead, spike - 1 : spike + 2] += [0.03, 0.05, 0.03]
                ecg[lead, r - 3 : r + 4] += np.linspace(0.0, 0.50, 7)
                ecg[lead, r + 4 : r + 11] += np.linspace(0.50, 0.0, 7)
        for lead in range(8):
            ecg[lead, isolated_spike] += 1.5

        result = validate_pacing_spikes_against_qrs(
            ecg,
            fs=fs,
            spike_times=spike_times,
            r_locs=r_locs,
            pacing_result={"paced": True, "state": "on", "lead_vote_count": 8},
        )

        self.assertEqual([isolated_spike], result["spike_times"])
        self.assertEqual(qrs_edge_spikes, result["rejected_spike_times"])
        self.assertFalse(result["paced"])
        self.assertEqual("unknown", result["state"])

    def test_validate_pacing_spikes_preserves_dual_spike_pacing_context(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1800), dtype=float)
        r_locs = np.asarray([250, 550, 850, 1150, 1450], dtype=int)
        pre_qrs_spikes = [int(r_locs[0] - 56)] + [int(r - 120) for r in r_locs[1:]]
        qrs_edge_spikes = [int(r - 10) for r in r_locs]
        spike_times = sorted(pre_qrs_spikes + qrs_edge_spikes)
        for pre_spike, edge_spike, r in zip(pre_qrs_spikes, qrs_edge_spikes, r_locs):
            for lead in range(8):
                ecg[lead, pre_spike] += 1.5
                ecg[lead, edge_spike - 1 : edge_spike + 2] += [0.03, 0.05, 0.03]
                ecg[lead, r - 3 : r + 4] += np.linspace(0.0, 0.50, 7)
                ecg[lead, r + 4 : r + 11] += np.linspace(0.50, 0.0, 7)

        result = validate_pacing_spikes_against_qrs(
            ecg,
            fs=fs,
            spike_times=spike_times,
            r_locs=r_locs,
            pacing_result={"paced": True, "state": "on", "lead_vote_count": 8},
        )

        self.assertEqual(spike_times, result["spike_times"])
        self.assertTrue(result["paced"])
        self.assertEqual("on", result["state"])
        self.assertEqual(qrs_edge_spikes, result["rejected_spike_times"])

    def test_validate_pacing_spikes_keeps_atrial_pacing_metadata_only(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1800), dtype=float)
        r_locs = np.asarray([300, 700, 1100, 1500], dtype=int)
        spike_times = [int(r - 80) for r in r_locs]
        for spike in spike_times:
            for lead in range(5):
                ecg[lead, spike] += 0.8

        result = validate_pacing_spikes_against_qrs(
            ecg,
            fs=fs,
            spike_times=spike_times,
            r_locs=r_locs,
            pacing_result={"spike_times": spike_times, "paced": True, "state": "on", "lead_vote_count": 5},
        )

        self.assertEqual(spike_times, result["spike_times"])
        self.assertFalse(result["paced"])
        self.assertEqual("unknown", result["state"])
        self.assertEqual("atrial_or_nonventricular_pacing", result["rejection_reason"])
        self.assertEqual("atrial_or_nonventricular", result["pacing_event_class"])

    def test_validate_pacing_spikes_keeps_isolated_dominant_spikes(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1200), dtype=float)
        r_locs = np.asarray([200, 500, 800], dtype=int)
        spike_times = [int(r - 20) for r in r_locs]
        for r, spike in zip(r_locs, spike_times):
            for lead in range(8):
                ecg[lead, r - 2 : r + 3] += np.linspace(0.0, 0.25, 5)
                ecg[lead, r + 3 : r + 8] += np.linspace(0.25, 0.0, 5)
                ecg[lead, spike] += 2.0

        result = validate_pacing_spikes_against_qrs(
            ecg,
            fs=fs,
            spike_times=spike_times,
            r_locs=r_locs,
            pacing_result={"paced": True, "state": "on", "lead_vote_count": 8},
        )

        self.assertEqual(spike_times, result["spike_times"])
        self.assertTrue(result["paced"])
        self.assertEqual("on", result["state"])
        self.assertEqual("ventricular_capture", result["pacing_event_class"])

    def test_cluster_beats_keeps_paced_beats_together(self) -> None:
        ecg = np.zeros((12, 2000), dtype=float)
        r_locs = np.asarray([200, 600, 1000, 1400], dtype=int)

        groups = cluster_beats(ecg, r_locs, fs=500, paced_beat_ids=[1, 3])

        self.assertTrue(any(set(members) == {1, 3} for members in groups.values()))

    def test_extract_wires_pacing_state_and_spike_beat_ids(self) -> None:
        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={"spike_times": [100], "paced": False, "state": "unknown", "lead_vote_count": 4}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}) as mock_pacing_failures, \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[{"beat_id": 1, "type": "escape_candidate"}]) as mock_post_pause, \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1]}) as mock_cluster, \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features") as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [120, 320]
            mock_global.return_value.pacing_spikes = None
            mock_global.return_value.paced_rhythm = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 500), dtype=float),
                fs=500,
            )

        self.assertEqual("unknown", result.metadata["pacing_state"])
        self.assertEqual([0], result.metadata["pacing_spike_beat_ids"])
        self.assertEqual([-40.0], result.metadata["pacing_spike_offsets_ms"])
        self.assertFalse(result.metadata["pacing_capture_confirmed"])
        self.assertEqual([0], result.metadata["paced_beat_ids"])
        self.assertEqual([0], mock_cluster.call_args.kwargs["paced_beat_ids"])
        self.assertEqual(
            {"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False},
            result.metadata["rhythm_analysis"]["pacing_failures"],
        )
        self.assertEqual([100], mock_pacing_failures.call_args.kwargs["spike_times"])
        self.assertEqual([120, 320], mock_pacing_failures.call_args.kwargs["qrs_times"])
        self.assertEqual(500, mock_pacing_failures.call_args.kwargs["fs"])
        self.assertEqual(
            [{"beat_id": 1, "type": "escape_candidate"}],
            result.metadata["rhythm_analysis"]["rule_summary"]["post_pause_or_interpolated_beats"],
        )
        self.assertEqual([], mock_post_pause.call_args.kwargs["beats"])
        self.assertEqual(400.0, mock_post_pause.call_args.kwargs["background_rr_ms"])

    def test_extract_does_not_mark_non_paced_measurement_group_as_paced(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=None,
            qrs_ms=90.0,
            qt_ms=360.0,
            heart_rate_bpm=75.0,
            p_axis_deg=None,
            qrs_axis_deg=None,
            t_axis_deg=None,
            pacing_spikes=None,
            paced_rhythm=False,
        )
        beat_rows = [
            SimpleNamespace(beat_id=0, paced=True, group_id=1),
            SimpleNamespace(beat_id=1, paced=False, group_id=2),
        ]

        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={"spike_times": [100, 300, 500], "paced": True, "state": "on", "lead_vote_count": 4}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}), \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0], 2: [1]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=beat_rows), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({1: np.zeros((12, 300), dtype=float)}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", return_value=[]) as mock_delineate, \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features) as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [120, 700]
            mock_qrs.return_value.fallback_used = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 900), dtype=float),
                fs=500,
            )

        self.assertEqual([0], result.metadata["pacing_spike_beat_ids"])
        self.assertFalse(result.metadata["pacing_capture_confirmed"])
        self.assertEqual([], result.metadata["paced_beat_ids"])
        self.assertEqual([0], result.metadata["measurement_beat_ids"])
        self.assertEqual([], mock_delineate.call_args_list[0].kwargs["paced_beat_ids"])
        self.assertEqual([], mock_delineate.call_args_list[1].kwargs["paced_beat_ids"])
        self.assertFalse(mock_global.call_args.kwargs["paced"])

    def test_extract_does_not_pace_representative_when_selected_group_is_not_paced(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=None,
            qrs_ms=90.0,
            qt_ms=360.0,
            heart_rate_bpm=75.0,
            p_axis_deg=None,
            qrs_axis_deg=None,
            t_axis_deg=None,
            pacing_spikes=None,
            paced_rhythm=False,
        )
        beat_rows = [
            SimpleNamespace(beat_id=0, paced=True, group_id=1),
            SimpleNamespace(beat_id=1, paced=True, group_id=1),
            SimpleNamespace(beat_id=2, paced=False, group_id=2),
            SimpleNamespace(beat_id=3, paced=False, group_id=2),
            SimpleNamespace(beat_id=4, paced=False, group_id=2),
        ]
        beat_features = [
            SimpleNamespace(
                beat_id=beat_id,
                lead="II",
                qrs_ms=90.0,
                pr_ms=None,
                p_confidence=0.0,
                flags=[],
                p=SimpleNamespace(onset=None, peak=None, offset=None),
                qrs=SimpleNamespace(onset=None, peak=None, offset=None),
                t=SimpleNamespace(onset=None, peak=None, offset=None),
            )
            for beat_id in range(5)
        ]

        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={"spike_times": [100, 300], "paced": True, "state": "on", "lead_vote_count": 8}), \
             patch("feature_extraction.ecgfeat.api.validate_pacing_spikes_against_qrs", side_effect=lambda ecg, fs, spike_times, r_locs, pacing_result: pacing_result), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}), \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1], 2: [2, 3, 4]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=beat_rows), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({2: np.zeros((12, 300), dtype=float)}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", side_effect=[beat_features, []]) as mock_delineate, \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}) as mock_rep_leads, \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features) as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [120, 320, 520, 720, 920]
            mock_qrs.return_value.fallback_used = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 1100), dtype=float),
                fs=500,
            )

        self.assertEqual([2, 3, 4], result.metadata["measurement_beat_ids"])
        self.assertFalse(result.metadata["measurement_group_paced"])
        self.assertTrue(result.metadata["confirmed_pacing_context"])
        self.assertEqual("per_beat_segmentation_floor", result.metadata["pacing_segmentation_effect"])
        self.assertEqual("per_beat_segmentation_floor", result.metadata["pacing_measurement_effect"])
        self.assertTrue(result.metadata["pacing_evidence_by_beat"][0]["ventricular_capture_confirmed"])
        self.assertTrue(result.metadata["pacing_evidence_by_beat"][0]["used_for_segmentation"])
        self.assertFalse(result.metadata["pacing_evidence_by_beat"][0]["used_for_representative_segmentation"])
        self.assertFalse(result.metadata["pacing_evidence_by_beat"][0]["used_for_global_paced_route"])
        self.assertEqual([0, 1], mock_delineate.call_args_list[0].kwargs["paced_beat_ids"])
        self.assertEqual([], mock_delineate.call_args_list[1].kwargs["paced_beat_ids"])
        self.assertFalse(mock_global.call_args.kwargs["paced"])
        self.assertEqual([2, 3, 4], [bf.beat_id for bf in mock_rep_leads.call_args.args[0]])
        self.assertEqual([2, 3, 4], [bf.beat_id for bf in mock_global.call_args.args[1]])

    def test_extract_uses_paced_route_only_when_measurement_group_is_paced(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=None,
            qrs_ms=140.0,
            qt_ms=420.0,
            heart_rate_bpm=75.0,
            p_axis_deg=None,
            qrs_axis_deg=None,
            t_axis_deg=None,
            pacing_spikes=None,
            paced_rhythm=False,
        )
        beat_features = [
            SimpleNamespace(
                beat_id=beat_id,
                lead="II",
                qrs_ms=140.0,
                pr_ms=None,
                p_confidence=0.0,
                flags=["paced_beat"],
                p=SimpleNamespace(onset=None, peak=None, offset=None),
                qrs=SimpleNamespace(onset=None, peak=None, offset=None),
                t=SimpleNamespace(onset=None, peak=None, offset=None),
            )
            for beat_id in range(4)
        ]

        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={"spike_times": [100, 300, 500, 700], "paced": True, "state": "on", "lead_vote_count": 8}), \
             patch("feature_extraction.ecgfeat.api.validate_pacing_spikes_against_qrs", side_effect=lambda ecg, fs, spike_times, r_locs, pacing_result: pacing_result), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}), \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1, 2, 3]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({1: np.zeros((12, 300), dtype=float)}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", side_effect=[beat_features, []]) as mock_delineate, \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features) as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [120, 320, 520, 720]
            mock_qrs.return_value.fallback_used = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 900), dtype=float),
                fs=500,
            )

        self.assertTrue(result.metadata["measurement_group_paced"])
        self.assertEqual("segmentation_floor_and_global_paced_route", result.metadata["pacing_measurement_effect"])
        self.assertTrue(mock_global.call_args.kwargs["paced"])
        self.assertEqual([0], mock_delineate.call_args_list[1].kwargs["paced_beat_ids"])
        self.assertEqual([0], mock_delineate.call_args_list[1].kwargs["paced_qrs_floor_beat_ids"])

    def test_extract_reselects_stable_native_family_over_paced_majority(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=None,
            qrs_ms=88.0,
            qt_ms=430.0,
            heart_rate_bpm=75.0,
            p_axis_deg=None,
            qrs_axis_deg=None,
            t_axis_deg=None,
            pacing_spikes=None,
            paced_rhythm=False,
        )
        beat_rows = [
            SimpleNamespace(beat_id=beat_id, paced=beat_id < 7, group_id=1 if beat_id < 7 else 2)
            for beat_id in range(10)
        ]
        beat_features = []
        for beat_id in range(10):
            qrs_ms = 126.0 if beat_id < 7 else 86.0
            flags = ["paced_beat", "paced_floor_applied", "beat_unreliable"] if beat_id < 7 else []
            for lead in ("I", "II", "V5"):
                beat_features.append(
                    SimpleNamespace(
                        beat_id=beat_id,
                        lead=lead,
                        qrs_ms=qrs_ms,
                        pr_ms=None,
                        p_confidence=0.0,
                        qrs_confidence=0.8,
                        qrs_off_confidence=0.8,
                        flags=list(flags),
                        p=SimpleNamespace(onset=None, peak=None, offset=None),
                        qrs=SimpleNamespace(onset=None, peak=None, offset=None),
                        t=SimpleNamespace(onset=None, peak=None, offset=None),
                    )
                )

        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={"spike_times": [110, 310, 510, 710, 910, 1110, 1310], "paced": True, "state": "on", "lead_vote_count": 8}), \
             patch("feature_extraction.ecgfeat.api.validate_pacing_spikes_against_qrs", side_effect=lambda ecg, fs, spike_times, r_locs, pacing_result: pacing_result), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}), \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: list(range(7)), 2: [7, 8, 9]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=beat_rows), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({1: np.zeros((12, 300), dtype=float), 2: np.zeros((12, 300), dtype=float)}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", side_effect=[beat_features, []]) as mock_delineate, \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}) as mock_rep_leads, \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features) as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [120, 320, 520, 720, 920, 1120, 1320, 1520, 1720, 1920]
            mock_qrs.return_value.fallback_used = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 2200), dtype=float),
                fs=500,
            )

        self.assertEqual([7, 8, 9], result.metadata["measurement_beat_ids"])
        self.assertFalse(result.metadata["measurement_group_paced"])
        self.assertEqual([], mock_delineate.call_args_list[1].kwargs["paced_beat_ids"])
        self.assertEqual([7, 8, 9], sorted({bf.beat_id for bf in mock_rep_leads.call_args.args[0]}))
        self.assertFalse(mock_global.call_args.kwargs["paced"])

    def test_extract_rejects_pacing_morphology_when_spikes_are_too_far_before_qrs(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=None,
            qrs_ms=90.0,
            qt_ms=360.0,
            heart_rate_bpm=75.0,
            p_axis_deg=None,
            qrs_axis_deg=None,
            t_axis_deg=None,
            pacing_spikes=None,
            paced_rhythm=False,
        )
        beat_rows = [
            SimpleNamespace(beat_id=0, paced=False, group_id=1),
            SimpleNamespace(beat_id=1, paced=False, group_id=1),
            SimpleNamespace(beat_id=2, paced=False, group_id=1),
        ]

        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={"spike_times": [95, 595, 1095], "paced": True, "state": "on", "lead_vote_count": 8}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}), \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1, 2]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=beat_rows), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({1: np.zeros((12, 300), dtype=float)}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", return_value=[]) as mock_delineate, \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features) as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [125, 625, 1125]
            mock_qrs.return_value.fallback_used = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 1400), dtype=float),
                fs=500,
            )

        self.assertEqual([], result.metadata["paced_beat_ids"])
        self.assertEqual([0, 1, 2], result.metadata["pacing_spike_beat_ids"])
        self.assertEqual([], mock_delineate.call_args_list[0].kwargs["paced_beat_ids"])
        self.assertFalse(mock_global.call_args.kwargs["paced"])

    def test_extract_keeps_atrial_pacing_evidence_metadata_only(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=150.0,
            qrs_ms=88.0,
            qt_ms=390.0,
            heart_rate_bpm=75.0,
            p_axis_deg=40.0,
            qrs_axis_deg=20.0,
            t_axis_deg=30.0,
            pacing_spikes=None,
            paced_rhythm=False,
        )
        spike_times = [220, 620, 1020]

        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={"spike_times": spike_times, "paced": True, "state": "on", "lead_vote_count": 5}), \
             patch("feature_extraction.ecgfeat.api.validate_pacing_spikes_against_qrs", return_value={
                 "spike_times": spike_times,
                 "paced": False,
                 "state": "unknown",
                 "lead_vote_count": 5,
                 "rejection_reason": "atrial_or_nonventricular_pacing",
                 "pacing_event_class": "atrial_or_nonventricular",
             }), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}), \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1, 2]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({1: np.zeros((12, 300), dtype=float)}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", return_value=[]) as mock_delineate, \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features) as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [300, 700, 1100]
            mock_qrs.return_value.fallback_used = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 1300), dtype=float),
                fs=500,
            )

        self.assertEqual("unknown", result.metadata["pacing_state"])
        self.assertEqual([], result.metadata["pacing_spike_beat_ids"])
        self.assertEqual([], result.metadata["paced_beat_ids"])
        self.assertFalse(result.metadata["pacing_capture_confirmed"])
        self.assertEqual("metadata_only", result.metadata["pacing_measurement_effect"])
        self.assertEqual([], mock_delineate.call_args_list[0].kwargs["paced_qrs_floor_beat_ids"])
        self.assertFalse(mock_global.call_args.kwargs["paced"])

    def test_extract_downgrades_metadata_only_pacing_to_unknown_measurement_state(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=150.0,
            qrs_ms=92.0,
            qt_ms=392.0,
            heart_rate_bpm=75.0,
            p_axis_deg=40.0,
            qrs_axis_deg=20.0,
            t_axis_deg=30.0,
            pacing_spikes=None,
            paced_rhythm=False,
        )

        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_spikes", return_value={
                 "spike_times": [120],
                 "paced": True,
                 "state": "on",
                 "lead_vote_count": 6,
             }), \
             patch("feature_extraction.ecgfeat.api.validate_pacing_spikes_against_qrs", return_value={
                 "spike_times": [120],
                 "paced": True,
                 "state": "on",
                 "lead_vote_count": 6,
             }), \
             patch("feature_extraction.ecgfeat.api.detect_pacing_failures", return_value={"artifact_confidence": 1.0, "capture_failure_suspected": False, "sensing_failure_suspected": False}), \
             patch("feature_extraction.ecgfeat.api.classify_post_pause_or_interpolated_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({1: np.zeros((12, 300), dtype=float)}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", return_value=[]) as mock_delineate, \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features) as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [300, 700]
            mock_qrs.return_value.fallback_used = False

            result = ECGFeatureExtractor(enable_pacing=True, enable_lead_reversal=False).extract(
                np.zeros((12, 900), dtype=float),
                fs=500,
            )

        self.assertEqual("on", result.metadata["pacing_detection_state"])
        self.assertEqual("unknown", result.metadata["measurement_pacing_state"])
        self.assertEqual("unknown", result.metadata["pacing_state"])
        self.assertEqual([], result.metadata["paced_beat_ids"])
        self.assertFalse(result.metadata["pacing_capture_confirmed"])
        self.assertEqual("metadata_only", result.metadata["pacing_measurement_effect"])
        self.assertEqual([], mock_delineate.call_args_list[0].kwargs["paced_qrs_floor_beat_ids"])
        self.assertFalse(mock_global.call_args.kwargs["paced"])

    def test_extract_disabled_pacing_keeps_off_state(self) -> None:
        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1]}) as mock_cluster, \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features") as mock_global, \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [120, 320]
            mock_global.return_value.pacing_spikes = None
            mock_global.return_value.paced_rhythm = False

            result = ECGFeatureExtractor(enable_pacing=False, enable_lead_reversal=False).extract(
                np.zeros((12, 500), dtype=float),
                fs=500,
            )

        self.assertEqual("off", result.metadata["pacing_state"])
        self.assertEqual([], result.metadata["paced_beat_ids"])
        self.assertEqual([], mock_cluster.call_args.kwargs["paced_beat_ids"])

    def test_extract_metadata_global_qt_includes_consensus_vs_independent_map(self) -> None:
        global_features = SimpleNamespace(
            pr_ms=None,
            qrs_ms=130.0,
            qt_ms=None,
            heart_rate_bpm=75.0,
            p_axis_deg=None,
            pacing_spikes=None,
            paced_rhythm=False,
            qt_source="low_qt_support",
            qt_used_leads=["II"],
            qt_reliability="low",
            qt_path="low_qt_support",
            qt_confidence_reason="insufficient_reliable_qt_leads",
            qt_excluded_leads={"V1": "short_jt"},
            qt_lead_weights={"II": 0.75},
            consensus_vs_independent_per_lead={
                "II": {"qt_ms": 430.0, "qt_consensus_ms": None},
            },
        )
        with patch("feature_extraction.ecgfeat.api.compute_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_limb_lead_reversal", return_value={}), \
             patch("feature_extraction.ecgfeat.api.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.api.cluster_beats", return_value={1: [0, 1]}), \
             patch("feature_extraction.ecgfeat.api.build_beat_annotations", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", return_value=({}, {})), \
             patch("feature_extraction.ecgfeat.api.delineate_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch("feature_extraction.ecgfeat.api.compute_global_features", return_value=global_features), \
             patch("feature_extraction.ecgfeat.api.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [120, 320]

            result = ECGFeatureExtractor(enable_pacing=False, enable_lead_reversal=False).extract(
                np.zeros((12, 500), dtype=float),
                fs=500,
            )

        self.assertEqual(
            {"II": {"qt_ms": 430.0, "qt_consensus_ms": None}},
            result.metadata["global_qt"]["consensus_vs_independent_per_lead"],
        )
        global_features.consensus_vs_independent_per_lead["II"]["qt_ms"] = 999.0
        self.assertEqual(
            {"II": {"qt_ms": 430.0, "qt_consensus_ms": None}},
            result.metadata["global_qt"]["consensus_vs_independent_per_lead"],
        )


if __name__ == "__main__":
    unittest.main()
