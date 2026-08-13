from __future__ import annotations

from typing import List, Optional
import unittest

import numpy as np

from feature_extraction.ecgfeat.repolarization import (
    TWaveCandidate,
    TWaveMeasurement,
    candidates_from_triplets,
    detect_t_wave,
    infer_cluster_polarity,
)


class RepolarizationTests(unittest.TestCase):
    def test_infer_cluster_polarity_uses_signed_area_majority(self) -> None:
        candidates = {
            "II": [TWaveCandidate(index=220, amp_mv=0.20, signed_area=2.0, abs_area=2.0, width_samples=30, time_from_qrs_off_ms=160.0, lead="II")],
            "aVF": [TWaveCandidate(index=222, amp_mv=0.16, signed_area=1.4, abs_area=1.4, width_samples=28, time_from_qrs_off_ms=164.0, lead="aVF")],
            "aVR": [TWaveCandidate(index=221, amp_mv=-0.10, signed_area=-0.7, abs_area=0.7, width_samples=20, time_from_qrs_off_ms=162.0, lead="aVR")],
        }

        self.assertEqual(1, infer_cluster_polarity(candidates, expected=None))

    def test_candidates_from_triplets_signs_area_and_marks_early_candidates(self) -> None:
        candidates = candidates_from_triplets(
            {"III": [(130, -0.2, 2.5), (190, 0.1, 1.0)]},
            qrs_off=100,
            fs=500,
        )

        lead_iii = candidates["III"]
        self.assertEqual(-2.5, lead_iii[0].signed_area)
        self.assertEqual(2.5, lead_iii[0].abs_area)
        self.assertEqual(60.0, lead_iii[0].time_from_qrs_off_ms)
        self.assertIn("early_st_trough_candidate", lead_iii[0].flags)
        self.assertEqual(1.0, lead_iii[1].signed_area)
        self.assertEqual(1.0, lead_iii[1].abs_area)
        self.assertEqual(180.0, lead_iii[1].time_from_qrs_off_ms)
        self.assertNotIn("early_st_trough_candidate", lead_iii[1].flags)

    def test_infer_cluster_polarity_prefers_later_t_over_early_st_troughs(self) -> None:
        candidates = {
            lead: [
                TWaveCandidate(
                    index=130,
                    amp_mv=-0.30,
                    signed_area=-4.0,
                    abs_area=4.0,
                    width_samples=20,
                    time_from_qrs_off_ms=60.0,
                    lead=lead,
                    flags=("early_st_trough_candidate",),
                ),
                TWaveCandidate(
                    index=190,
                    amp_mv=0.12,
                    signed_area=1.0,
                    abs_area=1.0,
                    width_samples=30,
                    time_from_qrs_off_ms=180.0,
                    lead=lead,
                ),
            ]
            for lead in ("II", "aVF", "III")
        }

        self.assertEqual(1, infer_cluster_polarity(candidates, expected=None))

    def test_detect_t_wave_prefers_later_signed_t_over_early_st_trough(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        qrs_off = 150
        sig[170:200] = -0.38 * np.hanning(30)
        sig[235:285] = 0.18 * np.hanning(50)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=160,
            search_hi=330,
            qrs_off=qrs_off,
            local_t_cap=360,
            fs=fs,
            lead="II",
            expected_polarity=1,
            cluster_polarity=1,
        )

        self.assertIsNotNone(result.peak)
        self.assertGreaterEqual(result.peak or 0, 245)
        self.assertTrue(result.st_t_confusion)
        self.assertEqual("later_signed_t_after_st_trough", result.confidence_reason)

    def test_detect_t_wave_respects_disabled_late_st_rescue(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        qrs_off = 150
        sig[170:215] = 0.32 * np.hanning(45)
        sig[255:310] = 0.16 * np.hanning(55)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=160,
            search_hi=340,
            qrs_off=qrs_off,
            local_t_cap=380,
            fs=fs,
            lead="V5",
            expected_polarity=1,
            cluster_polarity=1,
            allow_late_same_polarity_st_rescue=False,
        )

        self.assertIsNotNone(result.peak)
        self.assertLess(result.peak or 0, 230)
        self.assertFalse(result.st_t_confusion)
        self.assertIsNone(result.confidence_reason)

    def test_detect_t_wave_ignores_tiny_early_opposite_deflection(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        qrs_off = 150
        sig[170:175] = -0.08 * np.hanning(5)
        sig[235:285] = 0.18 * np.hanning(50)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=160,
            search_hi=330,
            qrs_off=qrs_off,
            local_t_cap=360,
            fs=fs,
            lead="II",
            expected_polarity=1,
            cluster_polarity=1,
        )

        self.assertIsNotNone(result.peak)
        self.assertGreaterEqual(result.peak or 0, 245)
        self.assertFalse(result.st_t_confusion)
        self.assertIsNone(result.confidence_reason)

    def test_detect_t_wave_uses_strong_earlier_opposite_component_over_late_polarity_candidate(self) -> None:
        fs = 500
        sig = np.zeros(520, dtype=float)
        qrs_off = 100
        sig[180:241] = -0.15 * np.hanning(61)
        sig[315:356] = 0.16 * np.hanning(41)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=150,
            search_hi=390,
            qrs_off=qrs_off,
            local_t_cap=430,
            fs=fs,
            lead="I",
            expected_polarity=1,
            cluster_polarity=1,
        )

        self.assertIsNotNone(result.peak)
        self.assertLessEqual(result.peak or 999, 225)
        self.assertEqual(-1, result.polarity_observed)

    def test_detect_t_wave_extends_offset_to_significant_t_prime_component(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        qrs_off = 150
        sig[230:271] = 0.22 * np.hanning(41)
        sig[300:331] = -0.12 * np.hanning(31)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=180,
            search_hi=350,
            qrs_off=qrs_off,
            local_t_cap=380,
            fs=fs,
            lead="II",
            expected_polarity=1,
            cluster_polarity=1,
        )

        self.assertIsNotNone(result.peak)
        self.assertGreaterEqual(result.peak or 0, 245)
        self.assertLessEqual(result.peak or 0, 255)
        self.assertIsNotNone(result.t_prime_peak)
        self.assertGreaterEqual(result.t_prime_peak or 0, 310)
        self.assertGreaterEqual(result.offset or 0, 328)

    def test_detect_t_wave_does_not_extend_t_prime_to_remote_late_wave(self) -> None:
        fs = 500
        sig = np.zeros(520, dtype=float)
        qrs_off = 100
        sig[180:241] = -0.15 * np.hanning(61)
        sig[315:356] = 0.16 * np.hanning(41)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=150,
            search_hi=390,
            qrs_off=qrs_off,
            local_t_cap=430,
            fs=fs,
            lead="I",
            expected_polarity=1,
            cluster_polarity=1,
        )

        self.assertIsNotNone(result.peak)
        self.assertLessEqual(result.peak or 999, 225)
        self.assertIsNone(result.t_prime_peak)
        self.assertLess(result.offset or 999, 300)

    def test_detect_t_wave_keeps_low_amplitude_t_prime_from_st_confusion(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        qrs_off = 150
        sig[175:230] = 0.030 * np.hanning(55)
        sig[245:306] = -0.030 * np.hanning(61)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=170,
            search_hi=340,
            qrs_off=qrs_off,
            local_t_cap=380,
            fs=fs,
            lead="II",
            expected_polarity=None,
            cluster_polarity=-1,
        )

        self.assertIsNotNone(result.peak)
        self.assertFalse(result.st_t_confusion)
        self.assertIsNone(result.confidence_reason)

    def test_detect_t_wave_prefers_broad_significant_t_over_narrow_early_spike(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        qrs_off = 150
        sig[178:185] = 0.65 * np.hanning(7)
        sig[245:306] = 0.14 * np.hanning(61)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=170,
            search_hi=340,
            qrs_off=qrs_off,
            local_t_cap=380,
            fs=fs,
            lead="II",
            expected_polarity=1,
            cluster_polarity=1,
        )

        self.assertIsNotNone(result.peak)
        self.assertGreaterEqual(result.peak or 0, 265)
        self.assertLessEqual(result.peak or 0, 285)
        self.assertIsNone(result.t_prime_peak)

    def test_detect_t_wave_unknown_polarity_ignores_early_same_polarity_deflection(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        qrs_off = 150
        sig[170:200] = 0.24 * np.hanning(30)
        sig[235:285] = 0.18 * np.hanning(50)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=160,
            search_hi=330,
            qrs_off=qrs_off,
            local_t_cap=360,
            fs=fs,
            lead="II",
            expected_polarity=None,
            cluster_polarity=None,
        )

        self.assertIsNotNone(result.peak)
        self.assertGreaterEqual(result.peak or 0, 245)
        self.assertFalse(result.st_t_confusion)
        self.assertIsNone(result.confidence_reason)

    def test_detect_t_wave_prefers_late_same_polarity_t_over_continuous_st_trough(self) -> None:
        fs = 500
        sig = np.zeros(560, dtype=float)
        qrs_off = 150
        sig[168:246] -= 0.015
        sig[178:235] += -0.22 * np.hanning(57)
        sig[235:430] += -0.055
        sig[320:410] += -0.08 * np.hanning(90)

        result = detect_t_wave(
            sig=sig,
            sig_sm=sig,
            baseline=0.0,
            search_lo=165,
            search_hi=450,
            qrs_off=qrs_off,
            local_t_cap=500,
            fs=fs,
            lead="II",
            expected_polarity=1,
            cluster_polarity=None,
        )

        self.assertIsNotNone(result.peak)
        self.assertGreaterEqual(result.peak or 0, 330)
        self.assertLessEqual(result.peak or 0, 390)
        self.assertTrue(result.st_t_confusion)
        self.assertEqual("later_signed_t_after_st_trough", result.confidence_reason)

    def test_delineate_uses_polarity_aware_t_detection_for_st_trough_competition(self) -> None:
        from unittest.mock import patch

        from feature_extraction.ecgfeat.delineate import delineate_beats
        from feature_extraction.ecgfeat.models import LeadQuality, STANDARD_12_LEADS

        fs = 500
        ecg = np.zeros((12, 600), dtype=float)
        r_locs = np.asarray([200], dtype=int)

        lead_ii = STANDARD_12_LEADS.index("II")
        ecg[lead_ii, 250:265] = -0.30 * np.hanning(15)
        ecg[lead_ii, 315:365] = 0.18 * np.hanning(50)

        quality = {
            lead: LeadQuality(
                lead=lead,
                baseline_wander_score=0.0,
                muscle_noise_score=0.0,
                powerline_score=0.0,
                clipping_score=0.0,
                flatline_score=1.0,
                missing=False,
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }

        with patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(180, 210, 0, 0.9, 0.9)):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        lead_ii_feature = next(item for item in features if item.lead == "II")
        self.assertIsNotNone(lead_ii_feature.t.peak)
        self.assertGreaterEqual(lead_ii_feature.t.peak or 0, 320)
        self.assertTrue(lead_ii_feature.st_t_confusion)
        self.assertIn("st_t_confusion", lead_ii_feature.flags)

    def test_delineate_full_t_window_overrides_early_fused_st_trough_anchor(self) -> None:
        from unittest.mock import patch

        from feature_extraction.ecgfeat.delineate import delineate_beats
        from feature_extraction.ecgfeat.models import LeadQuality, STANDARD_12_LEADS

        fs = 500
        ecg = np.zeros((12, 700), dtype=float)
        r_locs = np.asarray([200], dtype=int)

        lead_ii = STANDARD_12_LEADS.index("II")
        ecg[lead_ii, 235:300] -= 0.015
        ecg[lead_ii, 248:288] += -0.24 * np.hanning(40)
        ecg[lead_ii, 300:450] -= 0.055
        ecg[lead_ii, 340:420] += -0.08 * np.hanning(80)

        quality = {
            lead: LeadQuality(
                lead=lead,
                baseline_wander_score=0.0,
                muscle_noise_score=0.0,
                powerline_score=0.0,
                clipping_score=0.0,
                flatline_score=1.0,
                missing=False,
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }

        with patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(180, 210, 0, 0.9, 0.9)):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        lead_ii_feature = next(item for item in features if item.lead == "II")
        self.assertIsNotNone(lead_ii_feature.t.peak)
        self.assertGreaterEqual(lead_ii_feature.t.peak or 0, 350)
        self.assertTrue(lead_ii_feature.st_t_confusion)
        self.assertIn("st_t_confusion", lead_ii_feature.flags)

    def test_delineate_uses_beat_level_t_cluster_for_non_key_lead(self) -> None:
        from unittest.mock import patch

        from feature_extraction.ecgfeat.delineate import delineate_beats
        from feature_extraction.ecgfeat.models import LeadQuality, STANDARD_12_LEADS
        from feature_extraction.ecgfeat.repolarization import detect_t_wave as real_detect_t_wave

        fs = 500
        ecg = np.zeros((12, 700), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        lead_ii = STANDARD_12_LEADS.index("II")
        lead_avf = STANDARD_12_LEADS.index("aVF")
        lead_iii = STANDARD_12_LEADS.index("III")
        ecg[lead_ii, 350:400] = 0.25 * np.hanning(50)
        ecg[lead_avf, 350:400] = 0.24 * np.hanning(50)
        ecg[lead_iii, 290:315] = -0.30 * np.hanning(25)
        ecg[lead_iii, 350:400] = 0.16 * np.hanning(50)

        quality = {
            lead: LeadQuality(
                lead=lead,
                baseline_wander_score=0.0,
                muscle_noise_score=0.0,
                powerline_score=0.0,
                clipping_score=0.0,
                flatline_score=1.0,
                missing=False,
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }

        lead_iii_clusters: List[Optional[int]] = []

        def spy_detect_t_wave(**kwargs: object) -> TWaveMeasurement:
            if kwargs.get("lead") == "III":
                cluster_polarity = kwargs.get("cluster_polarity")
                lead_iii_clusters.append(cluster_polarity if isinstance(cluster_polarity, int) else None)
            return real_detect_t_wave(**kwargs)  # type: ignore[arg-type]

        with (
            patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(180, 210, 0, 0.9, 0.9)),
            patch("feature_extraction.ecgfeat.delineate.detect_t_wave", side_effect=spy_detect_t_wave),
        ):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        lead_iii_feature = next(item for item in features if item.lead == "III")
        self.assertIn(1, lead_iii_clusters)
        self.assertIsNotNone(lead_iii_feature.t.peak)
        self.assertGreaterEqual(lead_iii_feature.t.peak or 0, 360)
        self.assertEqual(1, lead_iii_feature.t_polarity_observed)


if __name__ == "__main__":
    unittest.main()
