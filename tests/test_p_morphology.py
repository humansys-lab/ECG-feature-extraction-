from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.ecgfeat.p_morphology import measure_p_components


class PMorphologyTests(unittest.TestCase):
    def test_detects_notched_p_wave_from_two_positive_humps(self) -> None:
        fs = 500
        sig = np.zeros(120, dtype=float)
        samples = np.arange(sig.size)
        sig += 0.18 * np.exp(-0.5 * ((samples - 35) / 3.0) ** 2)
        sig += 0.16 * np.exp(-0.5 * ((samples - 50) / 3.0) ** 2)

        result = measure_p_components(
            sig,
            p_on=25,
            p_peak=35,
            p_off=60,
            baseline=0.0,
            fs=fs,
        )

        self.assertTrue(result["is_notched"])
        self.assertAlmostEqual(30.0, result["notch_interval_ms"], delta=2.0)
        self.assertFalse(result["is_biphasic"])

    def test_measures_terminal_negative_component_of_biphasic_p_wave(self) -> None:
        fs = 500
        sig = np.zeros(140, dtype=float)
        samples = np.arange(sig.size)
        sig += 0.14 * np.exp(-0.5 * ((samples - 40) / 5.0) ** 2)
        sig += -0.11 * np.exp(-0.5 * ((samples - 68) / 6.0) ** 2)

        result = measure_p_components(
            sig,
            p_on=28,
            p_peak=40,
            p_off=82,
            baseline=0.0,
            fs=fs,
        )

        self.assertTrue(result["is_biphasic"])
        self.assertGreater(result["initial_duration_ms"], 0.0)
        self.assertGreater(result["initial_amplitude_mV"], 0.0)
        self.assertGreaterEqual(result["terminal_duration_ms"], 30.0)
        self.assertLess(result["terminal_amplitude_mV"], 0.0)
        self.assertLess(result["terminal_area_mv_ms"], 0.0)

    def test_biphasic_detection_handles_abs_peak_at_terminal_negative_trough(self) -> None:
        fs = 500
        sig = np.zeros(150, dtype=float)
        samples = np.arange(sig.size)
        sig += 0.08 * np.exp(-0.5 * ((samples - 42) / 5.0) ** 2)
        sig += -0.16 * np.exp(-0.5 * ((samples - 72) / 7.0) ** 2)
        p_peak = int(np.argmax(np.abs(sig)))

        result = measure_p_components(
            sig,
            p_on=30,
            p_peak=p_peak,
            p_off=82,
            baseline=0.0,
            fs=fs,
        )

        self.assertTrue(result["is_biphasic"])
        self.assertLess(result["terminal_amplitude_mV"], 0.0)
        self.assertGreaterEqual(result["terminal_duration_ms"], 30.0)
        self.assertLess(result["terminal_area_mv_ms"], 0.0)

    def test_late_negative_noise_blip_does_not_replace_terminal_component(self) -> None:
        fs = 500
        sig = np.zeros(140, dtype=float)
        sig[32:48] = np.hanning(16) * 0.08
        sig[55:73] = -0.12
        sig[84:87] = -0.012

        result = measure_p_components(
            sig,
            p_on=28,
            p_peak=55,
            p_off=90,
            baseline=0.0,
            fs=fs,
        )

        self.assertTrue(result["is_biphasic"])
        self.assertLess(result["terminal_amplitude_mV"], -0.08)
        self.assertGreaterEqual(result["terminal_duration_ms"], 30.0)
        self.assertLess(result["terminal_area_mv_ms"], 0.0)

    def test_invalid_inputs_return_defaults_without_raising(self) -> None:
        cases = (
            {"sig": np.zeros(20), "baseline": 0.0, "fs": None},
            {"sig": [object()], "baseline": 0.0, "fs": 500},
            {"sig": np.zeros(20), "baseline": "bad", "fs": 500},
        )

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                result = measure_p_components(
                    kwargs["sig"],
                    p_on=1,
                    p_peak=5,
                    p_off=10,
                    baseline=kwargs["baseline"],
                    fs=kwargs["fs"],
                )

                self.assertFalse(result["is_notched"])
                self.assertFalse(result["is_biphasic"])
                self.assertIsNone(result["terminal_area_mv_ms"])


if __name__ == "__main__":
    unittest.main()
