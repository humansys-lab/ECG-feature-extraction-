from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.extract_patch_helpers import patch_calls

import numpy as np

from feature_extraction.ecgfeat.api import ECGFeatureExtractor
from feature_extraction.ecgfeat.models import LeadQuality, STANDARD_12_LEADS
from feature_extraction.ecgfeat.quality import summarize_record_quality
from tests.optional import needs_interpretation


def _make_quality(
    lead: str,
    *,
    flags: list[str] | None = None,
    reliable: bool = True,
    reliable_for_p: bool | None = None,
    reliable_for_qrs: bool | None = None,
    reliable_for_t: bool | None = None,
    reliable_for_qt: bool | None = None,
) -> LeadQuality:
    return LeadQuality(
        lead=lead,
        baseline_wander_score=0.0,
        muscle_noise_score=0.0,
        powerline_score=0.0,
        clipping_score=0.0,
        flatline_score=1.0,
        missing=False,
        reliable=reliable,
        flags=list(flags or []),
        reliable_for_p=reliable_for_p,
        reliable_for_qrs=reliable_for_qrs,
        reliable_for_t=reliable_for_t,
        reliable_for_qt=reliable_for_qt,
    )


class QualityContractTests(unittest.TestCase):
    def test_lead_quality_defaults_wave_specific_gates_from_reliable(self) -> None:
        quality = _make_quality("I", reliable=True)

        self.assertTrue(quality.reliable_for_p)
        self.assertTrue(quality.reliable_for_qrs)
        self.assertTrue(quality.reliable_for_t)
        self.assertTrue(quality.reliable_for_qt)

    def test_summarize_record_quality_downgrades_partial_map(self) -> None:
        qualities = {
            "I": _make_quality("I"),
            "II": _make_quality("II"),
        }

        summary = summarize_record_quality(qualities)

        self.assertEqual("Q3", summary["record_grade"])
        self.assertIn("partial_quality_map", summary["reason_codes"])
        self.assertIn("record", summary["rejected_functions"])
        self.assertEqual(2, summary["n_reliable_qrs_leads"])
        self.assertEqual(2, summary["n_reliable_p_leads"])
        self.assertEqual(2, summary["n_reliable_qt_leads"])

    def test_summarize_record_quality_marks_q2_when_p_leads_are_insufficient(self) -> None:
        qualities = {
            lead: _make_quality(lead, reliable_for_p=False)
            for lead in STANDARD_12_LEADS
        }
        qualities["V5"] = _make_quality("V5", flags=["baseline_wander"])

        summary = summarize_record_quality(qualities)

        self.assertEqual("Q2", summary["record_grade"])
        self.assertIn("p_measurement", summary["rejected_functions"])
        self.assertEqual(["baseline_wander"], summary["reason_codes"])
        self.assertEqual(12, summary["n_reliable_qrs_leads"])
        self.assertEqual(1, summary["n_reliable_p_leads"])
        self.assertEqual(12, summary["n_reliable_qt_leads"])

    def test_sparse_detector_sqi_advisories_do_not_downgrade_record(self) -> None:
        qualities = {
            lead: _make_quality(lead)
            for lead in STANDARD_12_LEADS
        }
        qualities["V1"].reason_codes = ["psqi_out_of_range"]
        qualities["V2"].reason_codes = ["low_bsqi"]

        summary = summarize_record_quality(qualities)

        self.assertEqual("Q0", summary["record_grade"])
        self.assertEqual([], summary["reason_codes"])
        self.assertEqual(
            {"low_bsqi": 1, "psqi_out_of_range": 1},
            summary["advisory_reason_counts"],
        )

    def test_multilead_bsqi_advisory_promotes_to_q1(self) -> None:
        qualities = {
            lead: _make_quality(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead in ("I", "II", "III", "aVR"):
            qualities[lead].reason_codes = ["low_bsqi"]

        summary = summarize_record_quality(qualities)

        self.assertEqual("Q1", summary["record_grade"])
        self.assertEqual(["low_bsqi"], summary["reason_codes"])

    def test_alternate_rhythm_anchors_prevent_whole_record_stop(self) -> None:
        qualities = {
            lead: _make_quality(lead)
            for lead in STANDARD_12_LEADS
        }
        qualities["II"] = _make_quality("II", reliable_for_qrs=False)
        qualities["V1"] = _make_quality("V1", reliable_for_qrs=False)

        summary = summarize_record_quality(qualities)

        self.assertEqual("Q0", summary["record_grade"])
        self.assertTrue(summary["required_anchor_available"])
        self.assertFalse(summary["preferred_anchor_available"])
        self.assertGreaterEqual(len(summary["fallback_anchor_leads"]), 2)
        self.assertNotIn("record", summary["rejected_functions"])

    @needs_interpretation
    def test_extract_adds_record_quality_summary_to_metadata(self) -> None:
        qualities = {
            lead: _make_quality(lead, reliable_for_p=False)
            for lead in STANDARD_12_LEADS
        }
        qualities["V5"] = _make_quality("V5", flags=["baseline_wander"])

        with patch("feature_extraction.ecgfeat.pipeline.stages.quality.compute_quality", return_value=qualities), \
             patch("feature_extraction.ecgfeat.pipeline.stages.quality.detect_limb_lead_reversal", return_value={}), \
             patch_calls("feature_extraction.ecgfeat.pipeline.stages.ventricular.detect_qrs_multilead_with_meta", "feature_extraction.ecgfeat.pipeline.policies.pacing.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.cluster_beats", return_value={1: [0, 1]}), \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.build_beat_annotations", return_value=[]), \
             patch_calls("feature_extraction.ecgfeat.pipeline.stages.beats.build_representative_beats_with_meta", "feature_extraction.ecgfeat.pipeline.stages.measurement.build_representative_beats_with_meta", return_value=({}, {})), \
             patch_calls("feature_extraction.ecgfeat.pipeline.stages.delineation.delineate_beats", "feature_extraction.ecgfeat.pipeline.stages.measurement.delineate_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.pipeline.stages.measurement.build_representative_lead_features", return_value={}), \
             patch("feature_extraction.ecgfeat.pipeline.stages.measurement.compute_group_features", return_value={}), \
             patch_calls("feature_extraction.ecgfeat.pipeline.stages.measurement.compute_global_features", "feature_extraction.ecgfeat.pipeline.policies.qt.compute_global_features") as mock_global, \
             patch("feature_extraction.ecgfeat.compat.interpretation_hooks.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [100, 300]
            mock_global.return_value.pacing_spikes = None
            mock_global.return_value.paced_rhythm = False

            result = ECGFeatureExtractor(enable_lead_reversal=False).extract(
                np.zeros((12, 500), dtype=float),
                fs=500,
            )

        self.assertIn("record_quality", result.metadata)
        self.assertEqual("Q2", result.metadata["record_quality"]["record_grade"])
        self.assertIn("p_measurement", result.metadata["record_quality"]["rejected_functions"])
        self.assertNotIn("glasgow_analysis", result.metadata)


if __name__ == "__main__":
    unittest.main()
