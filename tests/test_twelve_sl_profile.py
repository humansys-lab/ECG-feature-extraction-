from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.ecgfeat.models import LeadBeatFeatures, RepresentativeLeadFeatures, WaveBounds
from feature_extraction.ecgfeat.twelve_sl import (
    TWELVE_SL_CONSTANTS,
    apply_twelve_sl_measurement_profile,
    special_t_amplitude_mv,
    twelve_sl_wave_measurements_from_signal,
)


class TwelveSLProfileTests(unittest.TestCase):
    def test_wave_measurements_use_qrs_onset_baseline_and_rr_scaled_st(self) -> None:
        fs = 500
        sig = np.full(260, 0.10, dtype=float)
        sig[105] = 0.80   # +700 uV from QRS-onset baseline
        sig[115] = -0.30  # -400 uV from QRS-onset baseline
        sig[120] = 0.13   # STJ +30 uV
        sig[151] = 0.05   # STM at QRS offset + 1000/16 ms => -50 uV
        sig[182] = 0.00   # STE at QRS offset + 1000/8 ms => -100 uV
        sig[190] = -0.20  # T peak -300 uV
        sig[230] = 0.11   # T offset +10 uV

        measurements = twelve_sl_wave_measurements_from_signal(
            sig=sig,
            qrs_on=100,
            qrs_off=120,
            t_peak=190,
            t_off=230,
            avg_rr_ms=1000.0,
            fs=fs,
        )

        self.assertAlmostEqual(0.03, measurements["stj_mv"], places=6)
        self.assertAlmostEqual(-0.05, measurements["stm_mv"], places=6)
        self.assertAlmostEqual(-0.10, measurements["ste_mv"], places=6)
        self.assertAlmostEqual(62.5, measurements["stm_offset_ms"], places=6)
        self.assertAlmostEqual(125.0, measurements["ste_offset_ms"], places=6)
        self.assertAlmostEqual(300.0, measurements["qrs_balance_uv"], places=6)
        self.assertAlmostEqual(1100.0, measurements["qrs_deflection_uv"], places=6)
        self.assertGreater(measurements["qrs_area_uv_ms"], TWELVE_SL_CONSTANTS["wave_significance_area_uv_ms"])
        self.assertTrue(measurements["qrs_significant"])
        self.assertAlmostEqual(-310.0, measurements["special_t_uv"], places=6)

    def test_wave_measurements_infer_significant_negative_t_prime_for_special_t(self) -> None:
        fs = 500
        sig = np.full(280, 0.10, dtype=float)
        sig[120] = 0.12
        sig[182] = 0.12
        sig[150:181] = 0.10 + 0.20 * np.hanning(31)
        sig[205:226] = 0.10 - 0.09 * np.hanning(21)
        sig[235] = 0.10

        measurements = twelve_sl_wave_measurements_from_signal(
            sig=sig,
            qrs_on=100,
            qrs_off=120,
            t_peak=165,
            t_off=235,
            avg_rr_ms=1000.0,
            fs=fs,
        )

        self.assertAlmostEqual(-90.0, measurements["t_prime_amp_uv"], places=6)
        self.assertGreaterEqual(
            measurements["t_prime_area_uv_ms"],
            TWELVE_SL_CONSTANTS["wave_significance_area_uv_ms"],
        )
        self.assertAlmostEqual(-90.0, measurements["special_t_uv"], places=6)

    def test_wave_measurements_ignore_small_negative_t_prime_when_positive_t_dominates(self) -> None:
        fs = 500
        sig = np.full(280, 0.10, dtype=float)
        sig[120] = 0.12
        sig[182] = 0.12
        sig[150:181] = 0.10 + 0.20 * np.hanning(31)
        sig[205:236] = 0.10 - 0.03 * np.hanning(31)
        sig[245] = 0.10

        measurements = twelve_sl_wave_measurements_from_signal(
            sig=sig,
            qrs_on=100,
            qrs_off=120,
            t_peak=165,
            t_off=245,
            avg_rr_ms=1000.0,
            fs=fs,
        )

        self.assertAlmostEqual(-30.0, measurements["t_prime_amp_uv"], places=6)
        self.assertAlmostEqual(180.0, measurements["special_t_uv"], places=6)
        self.assertAlmostEqual(62.5, measurements["stm_offset_ms"], places=6)
        self.assertAlmostEqual(125.0, measurements["ste_offset_ms"], places=6)

    def test_low_confidence_st_does_not_drive_special_t_amplitude(self) -> None:
        fs = 500
        sig = np.zeros(260, dtype=float)
        sig[105] = 0.80
        sig[115] = -0.30
        sig[120] = 0.18
        sig[151] = 0.18
        sig[182] = 0.18
        sig[190] = 0.20
        sig[230] = 0.00

        measurements = twelve_sl_wave_measurements_from_signal(
            sig=sig,
            qrs_on=100,
            qrs_off=120,
            t_peak=190,
            t_off=230,
            avg_rr_ms=1000.0,
            fs=fs,
            qrs_offset_confidence=0.20,
        )

        self.assertAlmostEqual(0.18, measurements["ste_mv"], places=6)
        self.assertLess(measurements["st_confidence"], 0.50)
        self.assertEqual("low_qrs_offset_confidence", measurements["st_confidence_reason"])
        self.assertAlmostEqual(200.0, measurements["special_t_uv"], places=6)

    def test_wave_measurements_support_numpy_without_trapezoid_alias(self) -> None:
        if not hasattr(np, "trapezoid"):
            self.skipTest("NumPy already lacks trapezoid in this environment")
        fs = 500
        sig = np.full(180, 0.10, dtype=float)
        sig[55] = 0.60
        sig[65] = -0.20
        original = np.trapezoid
        try:
            delattr(np, "trapezoid")
            measurements = twelve_sl_wave_measurements_from_signal(
                sig=sig,
                qrs_on=50,
                qrs_off=70,
                t_peak=None,
                t_off=None,
                avg_rr_ms=800.0,
                fs=fs,
            )
        finally:
            np.trapezoid = original

        self.assertGreater(measurements["qrs_area_uv_ms"], 0.0)

    def test_special_t_amplitude_branches(self) -> None:
        self.assertAlmostEqual(
            -0.22,
            special_t_amplitude_mv(t_amp_mv=-0.20, t_prime_amp_mv=None, ste_mv=-0.05, t_offset_amp_mv=0.02),
        )
        self.assertAlmostEqual(
            0.18,
            special_t_amplitude_mv(t_amp_mv=0.20, t_prime_amp_mv=-0.03, ste_mv=0.02, t_offset_amp_mv=0.0),
        )
        self.assertAlmostEqual(
            -0.08,
            special_t_amplitude_mv(t_amp_mv=0.20, t_prime_amp_mv=-0.08, ste_mv=0.02, t_offset_amp_mv=0.0),
        )
        self.assertAlmostEqual(
            0.10,
            special_t_amplitude_mv(t_amp_mv=0.02, t_prime_amp_mv=0.10, ste_mv=0.00, t_offset_amp_mv=0.0),
        )

    def test_profile_enriches_representative_params_and_metadata(self) -> None:
        feature = LeadBeatFeatures(
            lead="I",
            beat_id=0,
            p=WaveBounds(onset=60, peak=75, offset=86),
            qrs=WaveBounds(onset=90, peak=100, offset=120),
            t=WaveBounds(onset=150, peak=190, offset=230),
            qt_ms=280.0,
            pr_ms=80.0,
            qrs_ms=60.0,
            p_amp_mv=0.12,
            qrs_area=2.2,
            q_amp_mv=-0.05,
            r_amp_mv=0.70,
            s_amp_mv=-0.40,
            st_on_mv=0.01,
            st_mid_mv=0.02,
            st_80ms_mv=0.03,
            t_amp_mv=-0.30,
            j_index=120,
            twelve_sl_stj_mv=0.03,
            twelve_sl_stm_mv=-0.05,
            twelve_sl_ste_mv=-0.10,
            twelve_sl_stm_offset_ms=62.5,
            twelve_sl_ste_offset_ms=125.0,
            twelve_sl_qrs_area_uv_ms=2230.0,
            twelve_sl_qrs_signed_area_uv_ms=500.0,
            twelve_sl_qrs_balance_uv=300.0,
            twelve_sl_qrs_deflection_uv=1100.0,
            twelve_sl_minimum_st_uv=-50.0,
            twelve_sl_special_t_uv=-310.0,
            twelve_sl_t_prime_uv=-90.0,
            twelve_sl_t_prime_area_uv_ms=1800.0,
            twelve_sl_qrs_significant=True,
            beat_measurement_reliable=True,
            qrs_confidence=0.9,
        )
        reps = {
            "I": RepresentativeLeadFeatures(
                lead="I",
                params={"reliable_for_global": True, "reliable_for_qrs": True},
                variance={},
            )
        }

        profile = apply_twelve_sl_measurement_profile(
            representative_leads=reps,
            beat_features=[feature],
            r_locs=np.asarray([100, 600, 1150], dtype=int),
            fs=500,
        )

        params = reps["I"].params
        self.assertEqual("12sl_measurement_profile_v1", profile["profile_version"])
        self.assertAlmostEqual(57.142857142857146, profile["heart_rate_first_last_bpm"])
        self.assertEqual(1000.0, profile["constants"]["pace_spike_high_uv"])
        self.assertAlmostEqual(-0.05, params["twelve_sl_stm_mv"])
        self.assertAlmostEqual(-0.10, params["twelve_sl_ste_mv"])
        self.assertAlmostEqual(2230.0, params["twelve_sl_qrs_area_uv_ms"])
        self.assertAlmostEqual(-90.0, params["twelve_sl_t_prime_uv"])
        self.assertAlmostEqual(1800.0, params["twelve_sl_t_prime_area_uv_ms"])
        self.assertTrue(params["twelve_sl_qrs_significant"])
        self.assertAlmostEqual(-20.0, profile["global_fiducials"]["qrs_onset_offset_ms"])
        self.assertAlmostEqual(40.0, profile["global_fiducials"]["qrs_offset_offset_ms"])
        self.assertAlmostEqual(260.0, profile["global_fiducials"]["t_offset_offset_ms"])


if __name__ == "__main__":
    unittest.main()
