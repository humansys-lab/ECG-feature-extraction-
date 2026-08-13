from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.ecgfeat.features import compute_global_features
from feature_extraction.ecgfeat.measurement_paths import (
    build_qt_path_decision,
    classify_qt_path,
    qt_allowed_for_path,
)
from feature_extraction.ecgfeat.models import RepresentativeLeadFeatures


class MeasurementPathTests(unittest.TestCase):
    def test_late_qrs_consensus_does_not_override_stable_raw_core(self) -> None:
        raw_qrs_by_lead = {
            "I": 60.0,
            "II": 197.0,
            "III": 202.0,
            "aVR": 210.0,
            "aVL": 69.0,
            "aVF": None,
            "V1": 164.0,
            "V2": 142.0,
            "V3": 150.0,
            "V4": 88.0,
            "V5": 130.0,
            "V6": 206.0,
        }
        representative_leads = {}
        for lead, raw_qrs in raw_qrs_by_lead.items():
            representative_leads[lead] = RepresentativeLeadFeatures(
                lead=lead,
                params={
                    "qrs_ms": raw_qrs,
                    "qrs_consensus_ms": 210.0,
                    "qrs_wide_ms": 219.0,
                    "qt_ms": 500.0,
                    "qt_consensus_ms": 500.0,
                    "pr_ms": None,
                    "pr_consensus_ms": None,
                    "p_amp_mv": None,
                    "r_amp_mv": 0.2,
                    "t_amp_mv": 0.1,
                    "qrs_signed_area": 5.0,
                    "t_signed_area": 5.0,
                    "st_on_mv": 0.0,
                    "st_mid_mv": 0.0,
                    "st_80ms_mv": 0.0,
                    "st_t_confusion": lead in {"V5"},
                    "qt_confidence_mean": 0.8,
                    "reliable_for_global": True,
                    "reliable_for_qrs": True,
                    "reliable_for_qt": True,
                    "reliable_for_t": True,
                    "reliable_for_p": False,
                },
                variance={},
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([0, 500, 1000], dtype=int),
            fs=500,
            paced=False,
        )

        self.assertLessEqual(features.qrs_ms, 170.0)

    def test_wide_qrs_rejects_short_jt(self) -> None:
        self.assertFalse(qt_allowed_for_path(qt_ms=300.0, qrs_ms=170.0, path="wide_qrs_jt"))
        self.assertTrue(qt_allowed_for_path(qt_ms=430.0, qrs_ms=170.0, path="wide_qrs_jt"))

    def test_low_qrs_offset_confidence_does_not_make_normal_qrs_wide_path(self) -> None:
        self.assertEqual(
            "normal_qrs",
            classify_qt_path(qrs_ms=90.0, qrs_offset_confidence=0.1),
        )

    def test_low_qt_support_marks_low_confidence_without_projecting_consensus(self) -> None:
        decision = build_qt_path_decision(
            qrs_ms=100.0,
            reliable_qt_leads=["II"],
            lead_qt_values={"II": 420.0, "V5": None},
            lead_qrs_values={"II": 100.0},
            paced=False,
            qrs_offset_confidence=0.9,
        )

        self.assertEqual("low_qt_support", decision.path)
        self.assertEqual("low_confidence", decision.reliability)
        self.assertEqual(["II"], decision.used_leads)
        self.assertEqual("insufficient_reliable_qt_leads", decision.reason)
        self.assertEqual(420.0, decision.consensus_vs_independent_per_lead["II"]["qt_ms"])
        self.assertIn("qt_consensus_ms", decision.consensus_vs_independent_per_lead["II"])


if __name__ == "__main__":
    unittest.main()
