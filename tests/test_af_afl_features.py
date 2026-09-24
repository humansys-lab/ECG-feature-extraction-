import unittest
from types import SimpleNamespace

import numpy as np

from feature_extraction.ecgfeat.atrial import (
    _aggregate_multilead_atrial_activity,
    _classify_af_afl,
    _qrst_subtracted_rhythm,
    _subtraction_high_frequency_ratio,
    _summarize_atrial_residual,
    _t_aligned_residual,
    build_qrst_subtracted_residual,
    compute_pr_dispersion_ms,
)
try:  # the interpretation rules ship in the separate ecginterpret distribution
    from feature_extraction.ecgfeat.rhythm_statements import build_rhythm_statement_candidates
except ImportError:
    build_rhythm_statement_candidates = None
    HAS_INTERPRETATION = False
else:
    HAS_INTERPRETATION = True
INTERPRET_SKIP = "needs the interpretation rules (separate ecginterpret distribution)"


class AfAflFeatureTests(unittest.TestCase):
    def test_summarize_atrial_residual_detects_repetitive_flutter_like_signal(self) -> None:
        fs = 500
        t = np.arange(0, 4.0, 1.0 / fs)
        residual = 0.08 * np.sin(2.0 * np.pi * 5.0 * t)

        summary = _summarize_atrial_residual(residual, fs)

        self.assertGreater(summary["repetitiveness"], 0.60)
        self.assertGreater(summary["stability"], 0.60)
        self.assertTrue(180.0 <= summary["dominant_cycle_ms"] <= 240.0)

    def test_spectral_flutter_detected_in_220_400_bpm_band(self) -> None:
        fs = 500
        t = np.arange(0, 6.0, 1.0 / fs)
        # 300 bpm (5 Hz) flutter residual with a harmonic and noise.
        residual = 0.10 * np.sign(np.sin(2.0 * np.pi * 5.0 * t))
        residual += 0.02 * np.sin(2.0 * np.pi * 10.0 * t)
        residual += np.random.default_rng(0).normal(0.0, 0.01, size=residual.shape)

        summary = _summarize_atrial_residual(residual, fs)

        self.assertTrue(summary["spectral_flutter_candidate"])
        self.assertTrue(220.0 <= summary["spectral_flutter_bpm"] <= 400.0)

        result = _classify_af_afl([300.0] * 5, summary, organized_p_ratio=0.05)
        self.assertTrue(result["probable_flutter"])
        self.assertFalse(result["probable_af"])
        self.assertIn("spectral", result["flutter_detection_source"])

    def test_spectral_flutter_not_triggered_by_sinus_rate_residual(self) -> None:
        fs = 500
        t = np.arange(0, 6.0, 1.0 / fs)
        # 75 bpm (1.25 Hz) sinus P-rate residual is far below the flutter band.
        residual = 0.05 * np.sin(2.0 * np.pi * 1.25 * t)
        residual += np.random.default_rng(1).normal(0.0, 0.01, size=residual.shape)

        summary = _summarize_atrial_residual(residual, fs)

        self.assertFalse(summary["spectral_flutter_candidate"])

    def test_classify_af_afl_prefers_af_for_irregular_rr_and_unstable_residual(self) -> None:
        rr_ms = [540.0, 920.0, 610.0, 860.0, 580.0, 940.0]
        residual_summary = {
            "available": True,
            "repetitiveness": 0.20,
            "stability": 0.25,
            "dominant_cycle_ms": 190.0,
            "residual_rms_mv": 0.05,
        }

        result = _classify_af_afl(rr_ms, residual_summary, organized_p_ratio=0.10)

        self.assertTrue(result["probable_af"])
        self.assertLess(result["flutter_wave_confidence"], 0.40)

    def test_classic_af_allows_partial_but_inconsistent_p_candidates(self) -> None:
        rr_ms = [540.0, 920.0, 610.0, 860.0, 580.0, 940.0]
        residual_summary = {
            "available": True,
            "repetitiveness": 0.20,
            "stability": 0.25,
            "dominant_cycle_ms": 190.0,
        }

        probable = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.59
        )
        organized = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.61
        )

        self.assertTrue(probable["probable_af"])
        self.assertFalse(organized["probable_af"])

    def _ambiguous_band_inputs(self):
        rr_ms = [540.0, 920.0, 610.0, 860.0, 580.0, 940.0]
        residual_summary = {
            "available": True,
            "repetitiveness": 0.20,
            "stability": 0.25,
            "dominant_cycle_ms": 190.0,
        }
        return rr_ms, residual_summary

    def test_pr_dispersion_adjudicates_inside_the_ambiguous_ratio_band(self) -> None:
        """In the band the ratio is ~a coin flip, so PR scatter decides.

        Same organized_p_ratio, opposite verdicts, driven only by whether the
        conducted P waves march with the ventricles at a fixed interval.
        """
        rr_ms, residual_summary = self._ambiguous_band_inputs()

        marching = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.55,
            pr_dispersion_ms=8.0,
        )
        scattered = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.55,
            pr_dispersion_ms=40.0,
        )

        self.assertFalse(marching["probable_af"])
        self.assertEqual("pr_dispersion", marching["p_absence_decided_by"])
        self.assertTrue(scattered["probable_af"])
        self.assertEqual("pr_dispersion", scattered["p_absence_decided_by"])

    def test_pr_dispersion_is_ignored_outside_the_ambiguous_band(self) -> None:
        """Outside the band the ratio is strong and keeps the decision."""
        rr_ms, residual_summary = self._ambiguous_band_inputs()

        # Clearly absent organized P: tight PR must not rescue it into sinus.
        low = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.20,
            pr_dispersion_ms=5.0,
        )
        # Clearly organized P: scattered PR must not force AF.
        high = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.95,
            pr_dispersion_ms=60.0,
        )

        self.assertTrue(low["probable_af"])
        self.assertEqual("organized_p_ratio", low["p_absence_decided_by"])
        self.assertFalse(high["probable_af"])
        self.assertEqual("organized_p_ratio", high["p_absence_decided_by"])

    def test_missing_pr_dispersion_falls_back_to_the_ratio(self) -> None:
        rr_ms, residual_summary = self._ambiguous_band_inputs()

        result = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.55,
            pr_dispersion_ms=None,
        )

        self.assertEqual("organized_p_ratio", result["p_absence_decided_by"])
        self.assertIsNone(result["pr_dispersion_ms"])
        self.assertTrue(result["probable_af"])

    def test_compute_pr_dispersion_uses_one_pr_per_beat(self) -> None:
        # Four beats, three of them seen by two leads each. The duplicates
        # must not widen the spread -- per-beat PR is 100/104/108/112.
        events = []
        for beat_id, prs in enumerate(
            [(100.0, 100.0), (104.0, 104.0), (108.0,), (112.0, 112.0)]
        ):
            for pr in prs:
                events.append(
                    {
                        "association_type": "conducted",
                        "confidence": 1.0,
                        "associated_qrs_beat_id": beat_id,
                        "pr_ms": pr,
                    }
                )

        dispersion = compute_pr_dispersion_ms(events)

        self.assertIsNotNone(dispersion)
        # MAD of [100,104,108,112] is 4 ms -> robust sigma 4*1.4826.
        self.assertAlmostEqual(4.0 * 1.4826, dispersion, places=3)

    def test_compute_pr_dispersion_needs_enough_beats(self) -> None:
        events = [
            {
                "association_type": "conducted",
                "confidence": 1.0,
                "associated_qrs_beat_id": beat_id,
                "pr_ms": 150.0,
            }
            for beat_id in range(3)
        ]

        self.assertIsNone(compute_pr_dispersion_ms(events))

    def test_compute_pr_dispersion_excludes_non_conducted_and_low_confidence(self) -> None:
        base = [
            {
                "association_type": "conducted",
                "confidence": 1.0,
                "associated_qrs_beat_id": beat_id,
                "pr_ms": 150.0,
            }
            for beat_id in range(4)
        ]
        noise = [
            {
                "association_type": "retrograde",
                "confidence": 1.0,
                "associated_qrs_beat_id": 9,
                "pr_ms": 300.0,
            },
            {
                "association_type": "conducted",
                "confidence": 0.10,
                "associated_qrs_beat_id": 10,
                "pr_ms": 300.0,
            },
        ]

        self.assertEqual(0.0, compute_pr_dispersion_ms(base + noise))

    def test_strong_F_wave_evidence_outranks_irregular_rr(self) -> None:
        rr_ms = [540.0, 920.0, 610.0, 860.0, 580.0, 940.0]
        residual_summary = {
            "available": True,
            "repetitiveness": 0.20,
            "stability": 0.25,
            "dominant_cycle_ms": 190.0,
            "spectral_flutter_candidate": True,
            "spectral_flutter_confidence": 0.90,
        }

        result = _classify_af_afl(
            rr_ms, residual_summary, organized_p_ratio=0.30
        )

        self.assertFalse(result["probable_af"])
        self.assertTrue(result["probable_flutter"])
        self.assertEqual("atrial_flutter", result["atrial_rhythm_classification"])

    def test_multilead_F_wave_consensus_requires_reproducible_rate(self) -> None:
        fs = 500
        t = np.arange(0, 8.0, 1.0 / fs)
        rng = np.random.default_rng(7)
        lead_ii = _summarize_atrial_residual(
            0.08 * np.sign(np.sin(2.0 * np.pi * 5.0 * t))
            + rng.normal(0.0, 0.01, size=t.shape),
            fs,
        )
        lead_v1 = _summarize_atrial_residual(
            0.08 * np.sin(2.0 * np.pi * 5.0 * t), fs
        )
        lead_v2 = _summarize_atrial_residual(
            rng.normal(0.0, 0.02, size=t.shape), fs
        )

        result = _aggregate_multilead_atrial_activity(
            {"II": lead_ii, "V1": lead_v1, "V2": lead_v2}
        )

        self.assertTrue(result["flutter_multilead_consensus"])
        self.assertEqual(["II", "V1"], result["flutter_supporting_leads"])
        self.assertAlmostEqual(300.0, result["F_wave_rate_bpm"], delta=5.0)
        self.assertGreater(result["F_wave_confidence"], 0.80)

    def test_multilead_broadband_residual_supports_lowercase_f_waves(self) -> None:
        fs = 500
        n = 8 * fs
        lead_ii = _summarize_atrial_residual(
            np.random.default_rng(10).normal(0.0, 0.025, size=n), fs
        )
        lead_v1 = _summarize_atrial_residual(
            np.random.default_rng(11).normal(0.0, 0.025, size=n), fs
        )
        lead_v2 = _summarize_atrial_residual(
            0.08 * np.sin(2.0 * np.pi * 1.2 * np.arange(n) / fs), fs
        )

        result = _aggregate_multilead_atrial_activity(
            {"II": lead_ii, "V1": lead_v1, "V2": lead_v2}
        )

        self.assertTrue(result["f_wave_multilead_consensus"])
        self.assertFalse(result["flutter_multilead_consensus"])
        self.assertGreater(result["f_wave_confidence"], 0.75)

    def test_conflicting_strong_f_and_F_wave_evidence_abstains(self) -> None:
        residual_summary = {
            "available": True,
            "validated_qrst_subtraction": True,
            "repetitiveness": 0.20,
            "stability": 0.20,
            "dominant_cycle_ms": 200.0,
            "flutter_multilead_consensus": True,
            "F_wave_confidence": 0.84,
            "f_wave_multilead_consensus": True,
            "f_wave_confidence": 0.80,
        }

        result = _classify_af_afl(
            [540.0, 920.0, 610.0, 860.0, 580.0, 940.0],
            residual_summary,
            organized_p_ratio=0.10,
        )

        self.assertFalse(result["probable_af"])
        self.assertFalse(result["probable_flutter"])
        self.assertTrue(result["af_afl_indeterminate"])
        self.assertEqual("af_afl_indeterminate", result["atrial_rhythm_classification"])

    def test_mild_rr_irregularity_with_absent_p_abstains(self) -> None:
        # rr_cv here is ~0.105, inside the abstain band. It used to be ~0.139,
        # but calibration against 150 AFIB records puts that value squarely
        # inside the AF distribution (p25 0.128, median 0.173) -- it is not
        # "mild" irregularity, and treating it as such was why 58 of 73 missed
        # AF records were missed. The property under test is unchanged: RR
        # irregularity too weak to call AF, with absent organized P, abstains
        # rather than committing either way.
        result = _classify_af_afl(
            [700.0, 900.0, 720.0, 880.0, 730.0, 870.0],
            {
                "available": True,
                "repetitiveness": 0.20,
                "stability": 0.20,
                "dominant_cycle_ms": None,
                "flutter_multilead_consensus": False,
                "F_wave_confidence": 0.0,
                "f_wave_multilead_consensus": True,
                "f_wave_confidence": 0.80,
            },
            organized_p_ratio=0.20,
        )

        self.assertTrue(result["af_afl_indeterminate"])
        self.assertIn(
            "mild_rr_irregularity_with_absent_organized_p",
            result["indeterminate_reasons"],
        )

    def test_borderline_multilead_F_wave_pattern_abstains(self) -> None:
        result = _classify_af_afl(
            [800.0] * 8,
            {
                "available": True,
                "validated_qrst_subtraction": False,
                "repetitiveness": 0.50,
                "stability": 0.50,
                "dominant_cycle_ms": 200.0,
                "flutter_multilead_consensus": True,
                "F_wave_confidence": 0.58,
                "f_wave_multilead_consensus": False,
                "f_wave_confidence": 0.0,
            },
            organized_p_ratio=0.20,
        )

        self.assertTrue(result["af_afl_indeterminate"])
        self.assertFalse(result["probable_flutter"])
        self.assertIn(
            "borderline_multilead_F_wave_pattern",
            result["indeterminate_reasons"],
        )

    def test_build_qrst_subtracted_residual_marks_scaffold_not_validated_subtraction(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1400), dtype=float)
        beat_features = [
            SimpleNamespace(
                lead="I",
                beat_id=beat_id,
                qrs=SimpleNamespace(onset=100 + 300 * beat_id),
                t=SimpleNamespace(offset=260 + 300 * beat_id),
            )
            for beat_id in range(4)
        ]
        quality = {"I": SimpleNamespace(reliable_for_p=True)}

        summary = build_qrst_subtracted_residual(
            ecg,
            fs,
            np.asarray([120, 420, 720, 1020]),
            beat_features,
            quality,
        )

        self.assertEqual("baseline_subtracted_qrst_window_scaffold", summary["method"])
        self.assertFalse(summary["validated_qrst_subtraction"])

    def test_qrst_template_subtraction_marks_validated_when_template_quality_is_high(self) -> None:
        fs = 500
        ecg = np.zeros((12, 2000), dtype=float)
        r_locs = np.asarray([300, 800, 1300, 1800])
        for r in r_locs:
            ecg[:, r - 10:r + 11] += np.hanning(21)

        result = build_qrst_subtracted_residual(
            ecg,
            fs,
            r_locs,
            beat_features=[],
            quality={},
        )

        self.assertTrue(result["validated_qrst_subtraction"])
        self.assertGreaterEqual(result["template_correlation"], 0.85)
        # The subtraction model is tapered at the QRST-window edges, so a
        # mean-centred template deliberately leaves a tiny smooth edge residual.
        self.assertLess(result["residual_rms_mv"], 0.02)
        self.assertLessEqual(result["template_high_frequency_ratio"], 1.25)
        self.assertEqual(
            "r_aligned_qrs_shifted_robust_affine_stt_cosine_taper",
            result["template_adaptation_method"],
        )

    def test_t_template_subtraction_adapts_smooth_beatwise_t_amplitude(self) -> None:
        fs = 500
        n = 180
        r_index = 60
        template = np.zeros(n, dtype=float)
        template[50:71] = np.hanning(21)
        template[105:166] = 0.20 * np.hanning(61)
        segment = template.copy()
        segment[105:166] = 0.28 * np.hanning(61)

        residual = _t_aligned_residual(segment, template, fs, r_index)
        fixed_residual = segment - template

        self.assertLess(
            float(np.sqrt(np.mean(residual[110:160] ** 2))),
            0.35 * float(np.sqrt(np.mean(fixed_residual[110:160] ** 2))),
        )
        self.assertLessEqual(
            _subtraction_high_frequency_ratio(segment, residual),
            1.25,
        )

    def test_qrst_rhythm_subtraction_refuses_high_rate_p_intrusion_geometry(self) -> None:
        fs = 500
        signal = np.random.default_rng(17).normal(0.0, 0.01, size=1200)
        r_locs = np.asarray([300, 500, 700, 900])

        residual, available = _qrst_subtracted_rhythm(signal, fs, r_locs)

        self.assertFalse(available)
        np.testing.assert_array_equal(signal, residual)

    def test_qrst_template_validation_is_lead_specific_for_opposite_polarity_leads(self) -> None:
        fs = 500
        ecg = np.zeros((12, 2000), dtype=float)
        r_locs = np.asarray([300, 800, 1300, 1800])
        template = np.hanning(21)
        for r in r_locs:
            ecg[0, r - 10:r + 11] += template
            ecg[1, r - 10:r + 11] -= template

        result = build_qrst_subtracted_residual(
            ecg,
            fs,
            r_locs,
            beat_features=[],
            quality={
                "I": SimpleNamespace(reliable_for_p=True),
                "II": SimpleNamespace(reliable_for_p=True),
            },
        )

        self.assertTrue(result["validated_qrst_subtraction"])
        self.assertGreaterEqual(result["template_correlation"], 0.85)

    def test_qrst_template_subtraction_rejects_high_residual_energy(self) -> None:
        fs = 500
        ecg = np.zeros((12, 2000), dtype=float)
        r_locs = np.asarray([300, 800, 1300, 1800])
        template = np.hanning(21)
        for r, amplitude in zip(r_locs, [1.0, 2.0, 1.0, 2.0]):
            ecg[:, r - 10:r + 11] += amplitude * template

        result = build_qrst_subtracted_residual(
            ecg,
            fs,
            r_locs,
            beat_features=[],
            quality={},
        )

        self.assertFalse(result["validated_qrst_subtraction"])
        self.assertGreater(result["template_residual_rms_ratio"], 0.25)

    def test_build_qrst_subtracted_residual_handles_missing_optional_inputs(self) -> None:
        summary = build_qrst_subtracted_residual(
            np.zeros((12, 1000), dtype=float),
            500,
            np.asarray([100]),
            None,
            None,
        )

        self.assertFalse(summary["available"])

    @unittest.skipUnless(HAS_INTERPRETATION, INTERPRET_SKIP)
    def test_af_statement_requires_validated_qrst_subtraction_for_final_status(self) -> None:
        unvalidated = build_rhythm_statement_candidates(
            rhythm_summary={"primary_statement": None, "statements": []},
            preexcitation={"wpw_pattern": False},
            pacing_context={"continuous_pacing": False},
            availability={"pr_available": False, "reasons": ["probable_af"]},
            af_afl_summary={"probable_af": True, "rr_cv": 0.24},
            atrial_residual={
                "available": True,
                "validated_qrst_subtraction": False,
                "reason": "template_correlation_below_threshold",
            },
        )

        self.assertEqual(["probable_af"], [item["code"] for item in unvalidated["unavailable"]])
        self.assertEqual([], unvalidated["final_statements"])
        self.assertEqual("qrst_subtraction_not_validated", unvalidated["unavailable"][0]["unavailable_inputs"][0])

        validated = build_rhythm_statement_candidates(
            rhythm_summary={"primary_statement": None, "statements": []},
            preexcitation={"wpw_pattern": False},
            pacing_context={"continuous_pacing": False},
            availability={"pr_available": False, "reasons": ["probable_af"]},
            af_afl_summary={"probable_af": True, "rr_cv": 0.24},
            atrial_residual={
                "available": True,
                "validated_qrst_subtraction": True,
                "template_correlation": 0.92,
                "template_residual_rms_ratio": 0.08,
            },
        )

        self.assertEqual(["probable_af"], [item["code"] for item in validated["final_statements"]])
        self.assertEqual([], validated["unavailable"])
