from __future__ import annotations

import unittest

from feature_extraction.ecgfeat.models import RepresentativeLeadFeatures
from feature_extraction.ecgfeat.vector_axis import compute_t_axis_from_cluster


def _rep(lead: str, **params: object) -> RepresentativeLeadFeatures:
    base = {
        "t_amp_mv": None,
        "t_area": None,
        "t_signed_area": None,
        "qt_confidence_mean": 0.0,
        "reliable_for_t": False,
        "st_t_confusion": False,
    }
    base.update(params)
    return RepresentativeLeadFeatures(lead=lead, params=base, variance={"t_amp_mv_sd": 0.0})


class VectorAxisTests(unittest.TestCase):
    def test_t_axis_excludes_conflicted_low_support_lead(self) -> None:
        reps = {
            "I": _rep("I", t_amp_mv=0.20, t_area=2.0, t_signed_area=2.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "II": _rep("II", t_amp_mv=0.18, t_area=2.0, t_signed_area=2.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "aVF": _rep("aVF", t_amp_mv=-0.22, t_area=2.0, t_signed_area=-2.0, qt_confidence_mean=0.8, reliable_for_t=True, st_t_confusion=True),
        }

        result = compute_t_axis_from_cluster(reps)

        self.assertIsNotNone(result.axis_deg)
        self.assertIn("aVF", result.excluded_leads)
        self.assertEqual("polarity_conflict_or_st_t_confusion", result.excluded_leads["aVF"])

    def test_t_axis_suppresses_when_support_is_too_sparse(self) -> None:
        reps = {
            "II": _rep("II", t_amp_mv=0.18, t_area=2.0, t_signed_area=2.0, qt_confidence_mean=0.8, reliable_for_t=True),
        }

        result = compute_t_axis_from_cluster(reps)

        self.assertIsNone(result.axis_deg)
        self.assertEqual("insufficient_t_axis_leads", result.reason)

    def test_t_axis_records_stable_reasons_for_excluded_limb_leads(self) -> None:
        reps = {
            "I": _rep("I", t_amp_mv=0.20, t_area=2.0, t_signed_area=2.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "II": _rep("II", t_amp_mv=0.01, t_area=2.0, t_signed_area=2.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "III": _rep("III", t_amp_mv=0.20, t_area=2.0, t_signed_area=2.0, qt_confidence_mean=0.2, reliable_for_t=True),
            "aVR": _rep("aVR", t_amp_mv=0.20, t_area=2.0, t_signed_area=None, qt_confidence_mean=0.8, reliable_for_t=True),
            "aVL": _rep("aVL", t_amp_mv=0.20, t_area=2.0, t_signed_area=2.0, qt_confidence_mean=0.8, reliable_for_t=False),
        }

        result = compute_t_axis_from_cluster(reps, min_leads=1)

        self.assertEqual("low_t_amplitude", result.excluded_leads["II"])
        self.assertEqual("low_t_confidence", result.excluded_leads["III"])
        self.assertEqual("missing_signed_area", result.excluded_leads["aVR"])
        self.assertEqual("not_reliable_for_t", result.excluded_leads["aVL"])
        self.assertEqual("missing", result.excluded_leads["aVF"])

    def test_t_axis_retains_physiologic_negative_avr_projection(self) -> None:
        reps = {
            "I": _rep("I", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "II": _rep("II", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "aVR": _rep("aVR", t_amp_mv=-0.20, t_area=1.0, t_signed_area=-1.0, qt_confidence_mean=0.8, reliable_for_t=True),
        }

        result = compute_t_axis_from_cluster(reps)

        self.assertIsNotNone(result.axis_deg)
        self.assertIn("I", result.used_leads)
        self.assertIn("II", result.used_leads)
        self.assertIn("aVR", result.used_leads)
        self.assertNotIn("aVR", result.excluded_leads)
        self.assertGreater(result.axis_deg, 20.0)
        self.assertLess(result.axis_deg, 40.0)

    def test_t_axis_excludes_lead_that_contradicts_provisional_vector(self) -> None:
        reps = {
            "I": _rep("I", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "II": _rep("II", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "aVR": _rep("aVR", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
        }

        result = compute_t_axis_from_cluster(reps)

        self.assertIsNotNone(result.axis_deg)
        self.assertIn("I", result.used_leads)
        self.assertIn("II", result.used_leads)
        self.assertNotIn("aVR", result.used_leads)
        self.assertEqual("polarity_conflict_or_st_t_confusion", result.excluded_leads["aVR"])

    def test_t_axis_honors_allowed_limb_leads(self) -> None:
        reps = {
            "I": _rep("I", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "II": _rep("II", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
            "aVF": _rep("aVF", t_amp_mv=0.20, t_area=1.0, t_signed_area=1.0, qt_confidence_mean=0.8, reliable_for_t=True),
        }

        result = compute_t_axis_from_cluster(reps, allowed_leads={"I", "II"})

        self.assertIsNotNone(result.axis_deg)
        self.assertEqual({"I", "II"}, set(result.used_leads))
        self.assertEqual("filtered_by_reporting", result.excluded_leads["aVF"])


if __name__ == "__main__":
    unittest.main()
