from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

from tests.extract_patch_helpers import patch_calls

import numpy as np

from feature_extraction.ecgfeat.api import ECGFeatureExtractor
from feature_extraction.ecgfeat.features import build_representative_lead_features
from feature_extraction.ecgfeat.interpret import (
    PRECORDIAL_LEADS,
    _classify_qtc,
    _lead_reversal_flags,
    _st_analysis,
)
from feature_extraction.ecgfeat.models import LeadBeatFeatures, LeadQuality, STANDARD_12_LEADS, WaveBounds
from feature_extraction.ecgfeat.quality import (
    compute_adjacent_precordial_correlations,
    detect_limb_lead_reversal,
    detect_precordial_reversal,
)


def _r_metrics(r_amps: dict[str, float]) -> dict[str, dict[str, float | None]]:
    return {
        lead: {"r_amp_mv": r_amp, "s_amp_mv": None, "q_amp_mv": None, "qrs_ms": None}
        for lead, r_amp in r_amps.items()
    }


def _st_rep(lead: str, st_on_mv: float):
    import types

    return types.SimpleNamespace(
        lead=lead,
        params={
            "st_on_mv": st_on_mv,
            "reliable_for_qt": True,
        },
    )


def _make_quality(lead: str) -> LeadQuality:
    return LeadQuality(
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


def _make_wave(onset: int | None, peak: int | None, offset: int | None) -> WaveBounds:
    return WaveBounds(onset=onset, peak=peak, offset=offset)


def _make_beat_feature(lead: str, beat_id: int, r_amp_mv: float) -> LeadBeatFeatures:
    return LeadBeatFeatures(
        lead=lead,
        beat_id=beat_id,
        p=_make_wave(10, 20, 30),
        qrs=_make_wave(40, 50, 60),
        t=_make_wave(70, 90, 120),
        qt_ms=380.0,
        pr_ms=150.0,
        qrs_ms=90.0,
        p_amp_mv=0.1,
        qrs_area=1.0,
        q_amp_mv=-0.1,
        r_amp_mv=r_amp_mv,
        s_amp_mv=-0.2,
        st_on_mv=0.0,
        st_mid_mv=0.0,
        st_80ms_mv=0.0,
        t_amp_mv=0.3,
        j_index=60,
        p_confidence=0.9,
        qrs_confidence=0.9,
        qt_confidence=0.9,
        beat_measurement_reliable=True,
    )


class LeadReversalTests(unittest.TestCase):
    def test_flat_limb_leads_do_not_emit_correlation_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = detect_limb_lead_reversal(
                np.zeros((12, 1000), dtype=float)
            )

        self.assertFalse(result["probable_ra_la"])
        self.assertFalse(result["probable_ra_ll"])
        self.assertFalse(result["probable_la_ll"])

    def test_detect_precordial_reversal_returns_structured_payload(self) -> None:
        result = detect_precordial_reversal(
            _r_metrics({"V1": 0.2, "V2": 0.9, "V3": 0.3, "V4": 1.0, "V5": 0.4, "V6": 0.8})
        )

        self.assertIsInstance(result, dict)
        self.assertIn("suspected", result)
        self.assertIn("progression_score", result)
        self.assertIn("best_adjacent_swap", result)

    def test_monotonic_improvement_never_confirms_precordial_reversal(self) -> None:
        result = detect_precordial_reversal(
            _r_metrics(
                {
                    "V1": 0.12,
                    "V2": 0.70,
                    "V3": 0.72,
                    "V4": 1.28,
                    "V5": 1.40,
                    "V6": 1.30,
                }
            )
        )

        self.assertIn(result["state"], {"not_suspected", "possible"})
        self.assertNotEqual(result["state"], "confirmed")

    def test_small_noise_level_dip_is_not_suspected(self) -> None:
        # A few-hundredths-of-a-mV dip is ordinary measurement noise, not a
        # lead-swap-sized discontinuity.
        result = detect_precordial_reversal(
            _r_metrics({"V1": 0.20, "V2": 0.85, "V3": 0.80, "V4": 1.00, "V5": 0.60, "V6": 0.50})
        )

        self.assertFalse(result["suspected"])

    def test_pathological_q_pattern_excludes_reversal_suspicion(self) -> None:
        # Poor R progression driven by anterior Q waves (old MI) must not be
        # relabeled as a lead-placement error.
        metrics = _r_metrics({"V1": 0.2, "V2": 0.9, "V3": 0.3, "V4": 1.0, "V5": 0.4, "V6": 0.8})
        metrics["V1"]["q_amp_mv"] = -0.30
        metrics["V2"]["q_amp_mv"] = -0.25

        result = detect_precordial_reversal(metrics)

        self.assertFalse(result["suspected"])
        self.assertEqual("excluded_known_poor_progression_cause", result.get("reason"))
        self.assertIn("pathological_q_pattern", result.get("excluded_causes", []))

    def test_adjacent_correlation_corroborates_candidate_swap(self) -> None:
        metrics = _r_metrics({"V1": 0.2, "V2": 0.9, "V3": 0.3, "V4": 1.0, "V5": 0.4, "V6": 0.8})
        correlations = {
            "V1_V2": 0.92,
            "V2_V3": 0.10,  # anomalously low relative to the other pairs
            "V3_V4": 0.90,
            "V4_V5": 0.93,
            "V5_V6": 0.91,
        }

        result = detect_precordial_reversal(metrics, correlations)

        self.assertTrue(result["suspected"])
        self.assertEqual(("V2", "V3"), result["best_adjacent_swap"])
        self.assertEqual("moderate", result["confidence"])
        self.assertTrue(result["continuity_corroboration"])

    def test_adjacent_correlation_contradicts_candidate_swap(self) -> None:
        metrics = _r_metrics({"V1": 0.2, "V2": 0.9, "V3": 0.3, "V4": 1.0, "V5": 0.4, "V6": 0.8})
        correlations = {
            "V1_V2": 0.90,
            "V2_V3": 0.88,  # just as continuous as the other adjacent pairs
            "V3_V4": 0.89,
            "V4_V5": 0.91,
            "V5_V6": 0.90,
        }

        result = detect_precordial_reversal(metrics, correlations)

        self.assertTrue(result["suspected"])
        self.assertEqual("low", result["confidence"])
        self.assertFalse(result["continuity_corroboration"])

    def test_compute_adjacent_precordial_correlations_high_for_similar_shapes(self) -> None:
        n = 200
        rep_beat = np.zeros((12, n), dtype=float)
        t = np.linspace(-3, 3, 40)
        bump = np.exp(-0.5 * t**2)
        lead_to_row = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
        bounds = {}
        for i, lead in enumerate(f"V{i}" for i in range(1, 7)):
            row = lead_to_row[lead]
            scale = 0.5 + 0.1 * i
            rep_beat[row, 80:120] = bump * scale
            bounds[lead] = (80, 119)

        correlations = compute_adjacent_precordial_correlations(rep_beat, bounds, lead_to_row)

        self.assertEqual(5, len(correlations))
        for pair, corr in correlations.items():
            self.assertGreater(corr, 0.95, f"{pair} should be highly correlated")

    def test_compute_adjacent_precordial_correlations_low_for_mismatched_shape(self) -> None:
        n = 200
        rep_beat = np.zeros((12, n), dtype=float)
        t = np.linspace(-3, 3, 40)
        bump = np.exp(-0.5 * t**2)
        ramp = np.linspace(-1.0, 1.0, 40)
        lead_to_row = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
        bounds = {}
        for lead in (f"V{i}" for i in range(1, 7)):
            row = lead_to_row[lead]
            shape = ramp if lead == "V3" else bump
            rep_beat[row, 80:120] = shape
            bounds[lead] = (80, 119)

        correlations = compute_adjacent_precordial_correlations(rep_beat, bounds, lead_to_row)

        self.assertLess(correlations["V2_V3"], 0.3)
        self.assertLess(correlations["V3_V4"], 0.3)
        self.assertGreater(correlations["V1_V2"], 0.95)
        self.assertGreater(correlations["V4_V5"], 0.95)
        self.assertGreater(correlations["V5_V6"], 0.95)

    def test_wide_qrs_excludes_reversal_suspicion(self) -> None:
        metrics = _r_metrics({"V1": 0.2, "V2": 0.9, "V3": 0.3, "V4": 1.0, "V5": 0.4, "V6": 0.8})
        for lead in metrics:
            metrics[lead]["qrs_ms"] = 140.0

        result = detect_precordial_reversal(metrics)

        self.assertFalse(result["suspected"])
        self.assertIn("wide_qrs", result.get("excluded_causes", []))

    def test_build_representative_lead_features_carries_reversal_detail(self) -> None:
        quality = {
            lead: _make_quality(lead)
            for lead in STANDARD_12_LEADS
        }
        beat_features = [
            _make_beat_feature("V1", 0, 0.2),
            _make_beat_feature("V2", 0, 0.9),
            _make_beat_feature("V3", 0, 0.3),
            _make_beat_feature("V4", 0, 1.0),
            _make_beat_feature("V5", 0, 0.4),
            _make_beat_feature("V6", 0, 0.8),
        ]

        reps = build_representative_lead_features(beat_features, quality)

        detail = reps["V1"].params.get("precordial_reversal_detail")
        self.assertTrue(reps["V1"].params.get("probable_precordial_reversal"))
        self.assertIsInstance(detail, dict)
        self.assertTrue(detail["suspected"])
        self.assertEqual(("V2", "V3"), detail["best_adjacent_swap"])

    def test_extract_stores_structured_lead_reversal_metadata(self) -> None:
        quality = {
            lead: _make_quality(lead)
            for lead in STANDARD_12_LEADS
        }
        representative = {
            lead: type("Rep", (), {"params": {}})()
            for lead in STANDARD_12_LEADS
        }
        for lead in [f"V{i}" for i in range(1, 7)]:
            representative[lead].params = {
                "probable_precordial_reversal": True,
                "precordial_reversal_detail": {
                    "suspected": True,
                    "progression_score": 0.4,
                    "best_adjacent_swap": ("V2", "V3"),
                    "best_swap_score": 0.8,
                },
            }

        with patch("feature_extraction.ecgfeat.pipeline.stages.quality.compute_quality", return_value=quality), \
             patch("feature_extraction.ecgfeat.pipeline.stages.quality.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.pipeline.stages.quality.detect_limb_lead_reversal", return_value={"probable_ra_la": True}), \
             patch_calls("feature_extraction.ecgfeat.pipeline.stages.ventricular.detect_qrs_multilead_with_meta", "feature_extraction.ecgfeat.pipeline.policies.pacing.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.cluster_beats", return_value={1: [0, 1]}), \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.build_beat_annotations", return_value=[]), \
             patch_calls("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", "feature_extraction.ecgfeat.pipeline.stages.beats.build_representative_beats_with_meta", return_value=({}, {})), \
             patch_calls("feature_extraction.ecgfeat.api.delineate_beats", "feature_extraction.ecgfeat.pipeline.stages.delineation.delineate_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value=representative), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch_calls("feature_extraction.ecgfeat.api.compute_global_features", "feature_extraction.ecgfeat.pipeline.policies.qt.compute_global_features") as mock_global, \
             patch_calls("feature_extraction.ecgfeat.api.interpret", "feature_extraction.ecgfeat.compat.interpretation_hooks.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [100, 300]
            mock_global.return_value.pacing_spikes = None
            mock_global.return_value.paced_rhythm = False

            result = ECGFeatureExtractor(enable_lead_reversal=True).extract(
                np.zeros((12, 500), dtype=float),
                fs=500,
            )

        self.assertEqual(
            {
                "limb": {"probable_ra_la": True},
                "precordial": {
                    "suspected": True,
                    "progression_score": 0.4,
                    "best_adjacent_swap": ("V2", "V3"),
                    "best_swap_score": 0.8,
                },
            },
            result.metadata["lead_reversal"],
        )

    def test_extract_passes_structured_lead_reversal_metadata_into_interpret(self) -> None:
        quality = {
            lead: _make_quality(lead)
            for lead in STANDARD_12_LEADS
        }
        representative = {
            lead: type("Rep", (), {"params": {}})()
            for lead in STANDARD_12_LEADS
        }
        representative["V1"].params = {
            "probable_precordial_reversal": True,
            "precordial_reversal_detail": {
                "suspected": True,
                "progression_score": 0.4,
                "best_adjacent_swap": ("V2", "V3"),
                "best_swap_score": 0.8,
            },
        }

        seen = {}

        def _capture_interpret(features):
            seen["lead_reversal"] = features.metadata.get("lead_reversal")
            return None

        with patch("feature_extraction.ecgfeat.pipeline.stages.quality.compute_quality", return_value=quality), \
             patch("feature_extraction.ecgfeat.pipeline.stages.quality.summarize_record_quality", return_value={}), \
             patch("feature_extraction.ecgfeat.pipeline.stages.quality.detect_limb_lead_reversal", return_value={"probable_ra_la": True}), \
             patch_calls("feature_extraction.ecgfeat.pipeline.stages.ventricular.detect_qrs_multilead_with_meta", "feature_extraction.ecgfeat.pipeline.policies.pacing.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.cluster_beats", return_value={1: [0, 1]}), \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.build_beat_annotations", return_value=[]), \
             patch_calls("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", "feature_extraction.ecgfeat.pipeline.stages.beats.build_representative_beats_with_meta", return_value=({}, {})), \
             patch_calls("feature_extraction.ecgfeat.api.delineate_beats", "feature_extraction.ecgfeat.pipeline.stages.delineation.delineate_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value=representative), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch_calls("feature_extraction.ecgfeat.api.compute_global_features", "feature_extraction.ecgfeat.pipeline.policies.qt.compute_global_features") as mock_global, \
             patch_calls("feature_extraction.ecgfeat.api.interpret", "feature_extraction.ecgfeat.compat.interpretation_hooks.interpret", side_effect=_capture_interpret):
            mock_qrs.return_value.r_locs = [100, 300]
            mock_global.return_value.pacing_spikes = None
            mock_global.return_value.paced_rhythm = False

            ECGFeatureExtractor(enable_lead_reversal=True).extract(
                np.zeros((12, 500), dtype=float),
                fs=500,
            )

        self.assertEqual(
            {
                "limb": {"probable_ra_la": True},
                "precordial": {
                    "suspected": True,
                    "progression_score": 0.4,
                    "best_adjacent_swap": ("V2", "V3"),
                    "best_swap_score": 0.8,
                },
            },
            seen["lead_reversal"],
        )

    def test_extract_disabled_lead_reversal_clears_precordial_flags_and_metadata(self) -> None:
        quality = {
            lead: _make_quality(lead)
            for lead in STANDARD_12_LEADS
        }
        representative = {
            lead: type("Rep", (), {"params": {}})()
            for lead in STANDARD_12_LEADS
        }
        for lead in [f"V{i}" for i in range(1, 7)]:
            representative[lead].params = {
                "probable_precordial_reversal": True,
                "precordial_reversal_detail": {
                    "suspected": True,
                    "progression_score": 0.4,
                    "best_adjacent_swap": ("V2", "V3"),
                    "best_swap_score": 0.8,
                },
            }

        with patch("feature_extraction.ecgfeat.pipeline.stages.quality.compute_quality", return_value=quality), \
             patch("feature_extraction.ecgfeat.pipeline.stages.quality.summarize_record_quality", return_value={}), \
             patch_calls("feature_extraction.ecgfeat.pipeline.stages.ventricular.detect_qrs_multilead_with_meta", "feature_extraction.ecgfeat.pipeline.policies.pacing.detect_qrs_multilead_with_meta") as mock_qrs, \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.cluster_beats", return_value={1: [0, 1]}), \
             patch("feature_extraction.ecgfeat.pipeline.stages.beats.build_beat_annotations", return_value=[]), \
             patch_calls("feature_extraction.ecgfeat.api.build_representative_beats_with_meta", "feature_extraction.ecgfeat.pipeline.stages.beats.build_representative_beats_with_meta", return_value=({}, {})), \
             patch_calls("feature_extraction.ecgfeat.api.delineate_beats", "feature_extraction.ecgfeat.pipeline.stages.delineation.delineate_beats", return_value=[]), \
             patch("feature_extraction.ecgfeat.api.build_representative_lead_features", return_value=representative), \
             patch("feature_extraction.ecgfeat.api.compute_group_features", return_value={}), \
             patch_calls("feature_extraction.ecgfeat.api.compute_global_features", "feature_extraction.ecgfeat.pipeline.policies.qt.compute_global_features") as mock_global, \
             patch_calls("feature_extraction.ecgfeat.api.interpret", "feature_extraction.ecgfeat.compat.interpretation_hooks.interpret", return_value=None):
            mock_qrs.return_value.r_locs = [100, 300]
            mock_global.return_value.pacing_spikes = None
            mock_global.return_value.paced_rhythm = False

            result = ECGFeatureExtractor(enable_lead_reversal=False).extract(
                np.zeros((12, 500), dtype=float),
                fs=500,
            )

        self.assertEqual({"limb": {}, "precordial": {}}, result.metadata["lead_reversal"])
        for lead in [f"V{i}" for i in range(1, 7)]:
            self.assertFalse(result.representative_leads[lead].params.get("probable_precordial_reversal", False))
            self.assertNotIn("precordial_reversal_detail", result.representative_leads[lead].params)

    def test_st_analysis_excludes_precordial_leads_suppressing_pmia(self) -> None:
        # V1/V2 ST depression >= 0.10 mV would fire the posterior (PMIA) STEMI
        # code; an inferior limb-lead depression is also present.
        reps = {
            "V1": _st_rep("V1", -0.12),
            "V2": _st_rep("V2", -0.13),
            "V3": _st_rep("V3", -0.09),
            "II": _st_rep("II", -0.07),
            "III": _st_rep("III", -0.05),
            "aVF": _st_rep("aVF", -0.07),
        }

        # Baseline: precordial-driven PMIA and posterior territory are present.
        _, dep, _, terr_dep, codes, _, _, _ = _st_analysis(reps)
        self.assertIn("PMIA", codes)
        self.assertIn("posterior_reciprocal", terr_dep)
        self.assertIn("V1", dep)

        # With precordial reversal the precordial leads are excluded: no PMIA,
        # no posterior territory, precordial leads gone — but the inferior
        # (limb-lead) territory is retained.
        _, dep_x, _, terr_dep_x, codes_x, _, _, _ = _st_analysis(
            reps, exclude_leads=tuple(PRECORDIAL_LEADS)
        )
        self.assertNotIn("PMIA", codes_x)
        self.assertNotIn("posterior_reciprocal", terr_dep_x)
        self.assertNotIn("V1", dep_x)
        self.assertNotIn("V2", dep_x)
        self.assertIn("inferior", terr_dep_x)

    def test_classify_qtc_only_confirmed_rvh_suppresses_prolongation(self) -> None:
        # 491 ms is above the 485 ms prolonged threshold.
        self.assertEqual("prolonged", _classify_qtc(491.0)[0])
        # A weak "consider"-level RVH must NOT mask a prolonged QTc.
        self.assertEqual("prolonged", _classify_qtc(491.0, rvh_class="consider")[0])
        # Confirmed RVH (probable/definitive) still suppresses.
        self.assertEqual("normal", _classify_qtc(491.0, rvh_class="probable")[0])
        self.assertEqual("normal", _classify_qtc(491.0, rvh_class="definitive")[0])

    def test_classify_qtc_borderline_ivcd_does_not_suppress_prolongation(self) -> None:
        # A borderline IVCD (QRS 100–110 ms) is too mild to mask a prolonged QTc.
        self.assertEqual(
            "prolonged",
            _classify_qtc(491.0, bbb="IVCD", qrs_width_class="borderline_ivcd")[0],
        )
        # A genuine VCD (non-specific IVCD / real BBB) still suppresses.
        self.assertEqual(
            "normal",
            _classify_qtc(491.0, bbb="IVCD", qrs_width_class="nonspecific_ivcd")[0],
        )
        self.assertEqual("normal", _classify_qtc(491.0, bbb="RBBB")[0])

    def test_lead_reversal_flags_accept_structured_and_legacy_metadata(self) -> None:
        structured_features = type(
            "Features",
            (),
            {
                "metadata": {
                    "lead_reversal": {
                        "limb": {"probable_ra_la": True, "probable_extremity_reversal": False},
                        "precordial": {"suspected": True},
                    }
                },
                "representative_leads": {},
            },
        )()
        legacy_features = type(
            "Features",
            (),
            {
                "metadata": {
                    "lead_reversal": {"probable_ra_la": True, "probable_extremity_reversal": False}
                },
                "representative_leads": {},
            },
        )()

        self.assertEqual(("probable_ra_la=True", True), _lead_reversal_flags(structured_features))
        self.assertEqual(("probable_ra_la=True", False), _lead_reversal_flags(legacy_features))


if __name__ == "__main__":
    unittest.main()
