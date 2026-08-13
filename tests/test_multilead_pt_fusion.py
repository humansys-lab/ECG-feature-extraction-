"""Focused multi-lead P/T fusion coverage for Tasks 1 through 3."""
from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from feature_extraction.ecgfeat.delineate import (
    _apply_multilead_consensus,
    _candidate_triplets,
    _edge_preserving_moving_average,
    _fuse_peak_anchor,
    _refresh_p_boundary_diagnostics,
    delineate_beats,
)
from feature_extraction.ecgfeat.models import (
    LeadBeatFeatures,
    LeadQuality,
    STANDARD_12_LEADS,
    WaveBounds,
)


def _quality(*, reliable_for_p: bool = False, reliable_for_qt: bool = False) -> LeadQuality:
    return LeadQuality(
        lead="",
        baseline_wander_score=0.0,
        muscle_noise_score=0.0,
        powerline_score=0.0,
        clipping_score=0.0,
        flatline_score=1.0,
        missing=False,
        reliable=True,
        reliable_for_p=reliable_for_p,
        reliable_for_qrs=True,
        reliable_for_t=reliable_for_qt,
        reliable_for_qt=reliable_for_qt,
    )


class MultiLeadPTFusionTests(unittest.TestCase):
    @staticmethod
    def _p_feature(
        lead: str,
        *,
        p_on: int,
        p_peak: int,
        p_off: int,
        qrs_on: int,
    ) -> LeadBeatFeatures:
        return LeadBeatFeatures(
            lead=lead,
            beat_id=0,
            p=WaveBounds(p_on, p_peak, p_off),
            qrs=WaveBounds(qrs_on, qrs_on + 10, qrs_on + 25),
            t=WaveBounds(None, None, None),
            qt_ms=None,
            pr_ms=float(qrs_on - p_on),
            qrs_ms=50.0,
            p_amp_mv=0.08,
            qrs_area=1.0,
            q_amp_mv=-0.05,
            r_amp_mv=0.8,
            s_amp_mv=-0.2,
            st_on_mv=0.0,
            st_mid_mv=0.0,
            st_80ms_mv=0.0,
            t_amp_mv=None,
            j_index=qrs_on + 25,
            p_confidence=0.9,
            qrs_confidence=0.9,
        )

    def test_refresh_p_boundary_diagnostics_populates_local_snr_and_sigma(self) -> None:
        fs = 500
        rng = np.random.default_rng(7)
        ecg = rng.normal(0.0, 0.0025, size=(12, 500))
        p_wave = 0.09 * np.hanning(61)
        ecg[:, 170:231] += p_wave
        feature = self._p_feature(
            "II",
            p_on=170,
            p_peak=200,
            p_off=230,
            qrs_on=280,
        )

        _refresh_p_boundary_diagnostics([feature], ecg=ecg, fs=fs)

        self.assertTrue(feature.p_informative)
        self.assertGreater(feature.p_local_snr, 2.0)
        self.assertIn(
            feature.p_local_noise_source,
            {
                "pre_p_unverified_previous_t_out_of_context",
                "far_pre_p_unverified_previous_t_out_of_context",
            },
        )
        self.assertIsNone(feature.p_quiet_window_available)
        self.assertIsNone(feature.p_ta_overlap_risk)
        self.assertEqual(
            "robust_quadratic_overlap_context",
            feature.p_baseline_method,
        )
        self.assertIsNotNone(feature.p_onset_sigma_ms)
        self.assertIsNotNone(feature.p_offset_sigma_ms)
        self.assertEqual(
            "lp35_robust_quadratic_overlap_context_tangent_threshold_perturbation",
            feature.p_boundary_stability_method,
        )

    def test_refresh_p_boundary_diagnostics_marks_isoelectric_candidate(self) -> None:
        fs = 500
        axis = np.arange(500, dtype=float)
        noise = np.where((axis.astype(int) % 2) == 0, -0.010, 0.010)
        ecg = np.tile(noise, (12, 1))
        ecg[:, 200] += 0.004
        feature = self._p_feature(
            "II",
            p_on=185,
            p_peak=200,
            p_off=215,
            qrs_on=280,
        )

        _refresh_p_boundary_diagnostics([feature], ecg=ecg, fs=fs)

        self.assertFalse(feature.p_informative)
        self.assertEqual(0.9, feature.p_confidence)
        self.assertLessEqual(feature.p_onset_confidence, 0.20)
        self.assertLessEqual(feature.p_offset_confidence, 0.20)
        self.assertIn("p_isoelectric_uninformative", feature.flags)

    def test_refresh_p_boundary_diagnostics_uses_only_observed_tp_for_later_beat(self) -> None:
        fs = 500
        rng = np.random.default_rng(9)
        ecg = rng.normal(0.0, 0.002, size=(12, 600))
        ecg[:, 200:241] += 0.08 * np.hanning(41)
        previous = self._p_feature(
            "II",
            p_on=40,
            p_peak=60,
            p_off=80,
            qrs_on=100,
        )
        previous.t = WaveBounds(110, 130, 175)
        current = self._p_feature(
            "II",
            p_on=200,
            p_peak=220,
            p_off=240,
            qrs_on=290,
        )
        current.beat_id = 1

        _refresh_p_boundary_diagnostics([previous, current], ecg=ecg, fs=fs)

        self.assertEqual("tp_quiet", current.p_local_noise_source)
        self.assertTrue(current.p_quiet_window_available)
        self.assertAlmostEqual(50.0, current.p_tp_gap_ms)
        self.assertFalse(current.p_on_t_overlap_risk)
        self.assertEqual("tp_median", current.p_baseline_method)

    def test_refresh_p_boundary_diagnostics_fails_closed_when_tp_is_absent(self) -> None:
        fs = 500
        ecg = np.zeros((12, 600), dtype=float)
        ecg[:, 190:231] += 0.08 * np.hanning(41)
        previous = self._p_feature(
            "II",
            p_on=40,
            p_peak=60,
            p_off=80,
            qrs_on=100,
        )
        previous.t = WaveBounds(110, 145, 185)
        current = self._p_feature(
            "II",
            p_on=190,
            p_peak=210,
            p_off=230,
            qrs_on=280,
        )
        current.beat_id = 1

        _refresh_p_boundary_diagnostics([previous, current], ecg=ecg, fs=fs)

        self.assertEqual("no_tp_quiet_p_on_t", current.p_local_noise_source)
        self.assertIsNone(current.p_local_noise_rms_mv)
        self.assertIsNone(current.p_local_snr)
        self.assertIsNone(current.p_informative)
        self.assertFalse(current.p_quiet_window_available)
        self.assertTrue(current.p_on_t_overlap_risk)
        self.assertEqual(
            "robust_quadratic_overlap_context",
            current.p_baseline_method,
        )
        self.assertLessEqual(current.p_onset_confidence, 0.45)
        self.assertLessEqual(current.p_offset_confidence, 0.45)
        self.assertIn("p_no_tp_quiet_window", current.flags)
        self.assertIn("p_on_t_overlap_risk", current.flags)

    def test_refresh_p_boundary_diagnostics_detects_opposite_polarity_ta_risk(self) -> None:
        fs = 500
        rng = np.random.default_rng(19)
        ecg = rng.normal(0.0, 0.0015, size=(12, 650))
        ecg[:, 210:251] += 0.08 * np.hanning(41)
        ecg[:, 258:279] -= 0.025 * np.hanning(21)
        previous = self._p_feature(
            "II",
            p_on=40,
            p_peak=60,
            p_off=80,
            qrs_on=100,
        )
        previous.t = WaveBounds(120, 145, 180)
        current = self._p_feature(
            "II",
            p_on=210,
            p_peak=230,
            p_off=250,
            qrs_on=310,
        )
        current.beat_id = 1

        _refresh_p_boundary_diagnostics([previous, current], ecg=ecg, fs=fs)

        self.assertTrue(current.p_quiet_window_available)
        self.assertTrue(current.p_ta_overlap_risk)
        self.assertLessEqual(current.p_offset_confidence, 0.55)
        self.assertIn("p_ta_offset_risk", current.flags)

    def test_multilead_p_boundaries_use_protected_extremes_when_stable(self) -> None:
        fs = 500
        leads = ["I", "II", "aVF", "V1", "V5"]
        quality = {
            lead: _quality(reliable_for_p=True, reliable_for_qt=True)
            for lead in leads
        }
        for lead, lead_quality in quality.items():
            lead_quality.lead = lead
        features = []
        for lead, p_on, p_off in zip(
            leads,
            [20, 22, 24, 26, 28],
            [60, 62, 64, 66, 68],
        ):
            feature = self._p_feature(
                lead,
                p_on=p_on,
                p_peak=40,
                p_off=p_off,
                qrs_on=100,
            )
            feature.p_informative = True
            feature.p_quiet_window_available = True
            feature.p_on_t_overlap_risk = False
            feature.p_local_snr = 8.0
            feature.p_onset_sigma_ms = 4.0
            feature.p_offset_sigma_ms = 4.0
            feature.p_onset_confidence = 0.9
            feature.p_offset_confidence = 0.9
            features.append(feature)

        _apply_multilead_consensus(features, fs=fs, quality=quality)

        for feature in features:
            self.assertEqual(22, feature.p_onset_consensus_index)
            self.assertEqual(66, feature.p_offset_consensus_index)
            self.assertEqual(
                "confidence_gated_second_earliest",
                feature.p_onset_consensus_reason,
            )

    def test_multilead_p_boundaries_fall_back_to_median_without_tp_context(self) -> None:
        fs = 500
        leads = ["I", "II", "aVF", "V1", "V5"]
        quality = {
            lead: _quality(reliable_for_p=True, reliable_for_qt=True)
            for lead in leads
        }
        for lead, lead_quality in quality.items():
            lead_quality.lead = lead
        features = []
        for lead, p_on, p_off in zip(
            leads,
            [20, 22, 24, 26, 28],
            [60, 62, 64, 66, 68],
        ):
            feature = self._p_feature(
                lead,
                p_on=p_on,
                p_peak=40,
                p_off=p_off,
                qrs_on=100,
            )
            feature.p_informative = None
            feature.p_quiet_window_available = False
            feature.p_on_t_overlap_risk = True
            feature.p_onset_sigma_ms = 4.0
            feature.p_offset_sigma_ms = 4.0
            feature.p_onset_confidence = 0.45
            feature.p_offset_confidence = 0.45
            features.append(feature)

        _apply_multilead_consensus(features, fs=fs, quality=quality)

        for feature in features:
            self.assertEqual(24, feature.p_onset_consensus_index)
            self.assertEqual(64, feature.p_offset_consensus_index)
            self.assertNotIn(
                "second_earliest",
                feature.p_onset_consensus_reason,
            )

    def test_candidate_triplets_keeps_positive_and_negative_extrema(self) -> None:
        sig = np.zeros(120, dtype=float)
        sig[30:35] = np.array([0.0, -0.02, -0.05, -0.02, 0.0])
        sig[70:75] = np.array([0.0, 0.01, 0.04, 0.01, 0.0])

        candidates = _candidate_triplets(
            sig,
            lo=20,
            hi=90,
            baseline=0.0,
            min_amp=0.015,
            max_candidates=4,
        )

        peaks = [peak for peak, _amp, _area in candidates]
        self.assertIn(32, peaks)
        self.assertIn(72, peaks)

    def test_p_smoothing_does_not_create_a_zero_padding_edge_candidate(self) -> None:
        # A non-zero, gently declining beat boundary is ordinary ECG input.
        # mode="same" manufactures a local maximum at half the kernel width
        # because it treats samples before the fragment as zero.
        signal = np.linspace(0.12, 0.08, 200, dtype=float)
        kernel = np.ones(7, dtype=float) / 7.0
        zero_padded = np.convolve(signal, kernel, mode="same")
        legacy = _candidate_triplets(
            zero_padded,
            lo=0,
            hi=100,
            baseline=0.0,
            min_amp=0.008,
        )
        self.assertTrue(any(peak <= 4 for peak, _amp, _area in legacy))

        smoothed = _edge_preserving_moving_average(signal, 7)
        candidates = _candidate_triplets(
            smoothed,
            lo=0,
            hi=100,
            baseline=0.0,
            min_amp=0.008,
        )
        self.assertEqual(signal.shape, smoothed.shape)
        self.assertFalse(any(peak <= 4 for peak, _amp, _area in candidates))

    def test_delineate_beats_converts_r_relative_fusion_priors_to_beat_local(self) -> None:
        fs = 500
        ecg = np.zeros((12, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        priors = {
            1: {
                lead: {
                    "p_peak": -60,
                    "t_peak": 100,
                    "qrs_on": -20,
                    "qrs_off": 20,
                }
                for lead in STANDARD_12_LEADS
            }
        }
        observed_priors: dict[str, int | None] = {}

        def capture_fusion(
            _candidates_by_lead,
            _quality,
            reliable_attr,
            prior_peak,
            cluster_radius,
            **_kwargs,
        ):
            self.assertGreater(cluster_radius, 0)
            observed_priors[reliable_attr] = prior_peak
            return None, 0

        def observe_p_prior() -> int | None:
            observed_priors.clear()
            with patch(
                "feature_extraction.ecgfeat.delineate.build_group_priors",
                return_value=priors,
            ), patch(
                "feature_extraction.ecgfeat.delineate._fuse_peak_anchor",
                side_effect=capture_fusion,
            ):
                delineate_beats(
                    ecg,
                    fs=fs,
                    r_locs=r_locs,
                    rep_beats={1: np.zeros((12, 300), dtype=float)},
                )
            return observed_priors["reliable_for_p"]

        # beat_start=25, hence R is sample 175 in the beat-local frame.
        self.assertEqual(115, observe_p_prior())
        # A representative "P" after its QRS onset is invalid as a prior.
        for lead_prior in priors[1].values():
            lead_prior["p_peak"] = -10
        self.assertIsNone(observe_p_prior())
        # T prior behavior is outside this P-anchor correction.
        self.assertEqual(100, observed_priors["reliable_for_qt"])

    def test_fuse_peak_anchor_prefers_consistent_cluster_over_single_louder_outlier(self) -> None:
        quality = {
            "I": _quality(reliable_for_p=True),
            "II": _quality(reliable_for_p=True),
            "V1": _quality(reliable_for_p=True),
            "V2": _quality(reliable_for_p=True),
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        candidates = {
            "I": [(68, -0.03, 0.12)],
            "II": [(69, -0.04, 0.14)],
            "V1": [(70, -0.05, 0.13)],
            "V2": [(48, 0.10, 0.30)],
        }

        anchor, lead_support = _fuse_peak_anchor(
            candidates,
            quality,
            reliable_attr="reliable_for_p",
            prior_peak=69,
            cluster_radius=6,
        )

        self.assertEqual(69, anchor)
        # I/II/V1 form the winning cluster; V2's louder outlier is excluded.
        self.assertEqual(3, lead_support)

    def test_delineate_beats_uses_fused_p_anchor_on_weak_v1_lead(self) -> None:
        fs = 500
        ecg = np.zeros((12, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)

        lead_ii = STANDARD_12_LEADS.index("II")
        lead_v1 = STANDARD_12_LEADS.index("V1")

        ecg[lead_ii, 146:155] = np.array([0.0, 0.01, 0.03, 0.05, 0.08, 0.05, 0.03, 0.01, 0.0])
        ecg[lead_v1, 147:154] = np.array([0.0, -0.01, -0.02, -0.04, -0.02, -0.01, 0.0])
        ecg[lead_v1, 120] = 0.12

        quality = {
            lead: _quality(reliable_for_p=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        def _v1_p_peak(*, disable_fusion: bool = False) -> int:
            with ExitStack() as stack:
                if disable_fusion:
                    stack.enter_context(
                        patch(
                            "feature_extraction.ecgfeat.delineate._fuse_peak_anchor",
                            return_value=(None, 0),
                        )
                    )
                stack.enter_context(
                    patch(
                        "feature_extraction.ecgfeat.delineate._qrs_bounds",
                        return_value=(170, 190, 0, 0.9, 0.9),
                    )
                )
                stack.enter_context(
                    patch(
                        "feature_extraction.ecgfeat.delineate._t_onset_tangent",
                        side_effect=lambda sig, peak, lo, baseline, fs: None if peak is None else peak - 8,
                    )
                )
                stack.enter_context(
                    patch(
                        "feature_extraction.ecgfeat.delineate._p_offset_tangent",
                        side_effect=lambda sig, peak, hi, baseline, fs: None if peak is None else peak + 8,
                    )
                )
                stack.enter_context(
                    patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=None)
                )
                stack.enter_context(
                    patch(
                        "feature_extraction.ecgfeat.delineate._t_end_geometric",
                        return_value=(None, 0.0, "stub"),
                    )
                )
                features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

            v1 = next(bf for bf in features if bf.lead == "V1")
            self.assertIsNotNone(v1.p.peak)
            return v1.p.peak

        self.assertEqual(150, _v1_p_peak())
        # Without the cross-lead fused anchor, the single-sample spike at 120
        # still outscores the real (much smaller) deflection on amplitude
        # alone; the primary pick now uses the same raw-signal-refined
        # position as the alternatives list, landing exactly on the spike's
        # sample (120) instead of the smoothed-signal plateau's leftmost
        # index (117).
        self.assertEqual(120, _v1_p_peak(disable_fusion=True))

    def test_delineate_beats_uses_fused_t_anchor_to_ignore_late_noise_spike(self) -> None:
        fs = 500
        ecg = np.zeros((12, 600), dtype=float)
        r_locs = np.asarray([200], dtype=int)

        lead_ii = STANDARD_12_LEADS.index("II")
        lead_v5 = STANDARD_12_LEADS.index("V5")
        lead_v2 = STANDARD_12_LEADS.index("V2")

        ecg[lead_ii, 286:295] = np.array([0.0, 0.04, 0.09, 0.14, 0.18, 0.14, 0.09, 0.04, 0.0])
        ecg[lead_v5, 285:294] = np.array([0.0, 0.03, 0.08, 0.13, 0.16, 0.13, 0.08, 0.03, 0.0])
        ecg[lead_v2, 360] = 0.30

        quality = {
            lead: _quality(reliable_for_p=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        with patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(170, 190, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", return_value=None), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", return_value=None), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", side_effect=lambda sig, peak, lo, baseline, fs: None if peak is None else peak - 20), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", side_effect=lambda sig, peak, hi, baseline, fs: (None if peak is None else peak + 40, 0.9, "stub")):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        v2 = next(bf for bf in features if bf.lead == "V2")
        self.assertEqual(290, v2.t.peak)

    def test_fused_t_anchor_provenance_tracks_final_peak(self) -> None:
        fs = 500
        ecg = np.zeros((12, 600), dtype=float)
        r_locs = np.asarray([200], dtype=int)

        lead_ii = STANDARD_12_LEADS.index("II")
        lead_v5 = STANDARD_12_LEADS.index("V5")
        lead_v2 = STANDARD_12_LEADS.index("V2")

        ecg[lead_ii, 286:295] = np.array([0.0, 0.04, 0.09, 0.14, 0.18, 0.14, 0.09, 0.04, 0.0])
        ecg[lead_v5, 285:294] = np.array([0.0, 0.03, 0.08, 0.13, 0.16, 0.13, 0.08, 0.03, 0.0])
        ecg[lead_v2, 285:294] = 0.02 * np.array([0.0, 0.25, 0.55, 0.85, 1.0, 0.80, 0.45, 0.20, 0.0])

        quality = {
            lead: _quality(reliable_for_p=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        def _v2_feature(*, disable_fusion: bool = False):
            with ExitStack() as stack:
                if disable_fusion:
                    stack.enter_context(
                        patch(
                            "feature_extraction.ecgfeat.delineate._fuse_peak_anchor",
                            return_value=(None, 0),
                        )
                    )
                stack.enter_context(
                    patch(
                        "feature_extraction.ecgfeat.delineate._qrs_bounds",
                        return_value=(170, 190, 0, 0.9, 0.9),
                    )
                )
                stack.enter_context(
                    patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", return_value=None)
                )
                stack.enter_context(
                    patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", return_value=None)
                )
                stack.enter_context(
                    patch(
                        "feature_extraction.ecgfeat.delineate._t_onset_chord",
                        side_effect=lambda sig, peak, lo, baseline, fs: None if peak is None else peak - 20,
                    )
                )
                stack.enter_context(
                    patch(
                        "feature_extraction.ecgfeat.delineate._t_end_geometric",
                        side_effect=lambda sig, peak, hi, baseline, fs: (
                            None if peak is None else peak + 40,
                            0.9,
                            "stub",
                        ),
                    )
                )
                features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

            return next(bf for bf in features if bf.lead == "V2")

        without_fusion = _v2_feature(disable_fusion=True)
        with_fusion = _v2_feature()

        self.assertEqual(289, without_fusion.t.peak)
        self.assertEqual(290, with_fusion.t.peak)
        self.assertEqual("polarity_cluster_fused_anchor", with_fusion.t_peak_path)
        self.assertIsNone(with_fusion.t_polarity_expected)
        self.assertIsNone(with_fusion.t_polarity_observed)
        self.assertFalse(with_fusion.st_t_confusion)
        self.assertIsNone(with_fusion.t_confidence_reason)
