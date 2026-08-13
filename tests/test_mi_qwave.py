from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.ecgfeat.delineate import _q_component_metrics
from feature_extraction.ecgfeat.interpret import _detect_premature_complexes, _pathological_q_waves, interpret
from feature_extraction.ecgfeat.mi import (
    build_culprit_artery_evidence,
    build_mi_evidence,
    build_mi_statement_candidates,
)
from feature_extraction.ecgfeat.models import ECGFeatures, GlobalFeatures, RepresentativeLeadFeatures
from feature_extraction.ecgfeat.rhythm_rules import classify_pacing_context


class QComponentMeasurementTests(unittest.TestCase):
    def test_q_duration_stops_when_wave_returns_to_baseline_before_r(self) -> None:
        fs = 1000
        sig = np.zeros(220, dtype=float)
        sig[100:111] = np.linspace(0.0, -0.20, 11)
        sig[111:122] = np.linspace(-0.20, 0.0, 11)
        sig[140:151] = np.linspace(0.0, 1.0, 11)
        sig[151:166] = np.linspace(1.0, 0.0, 15)

        metrics = _q_component_metrics(
            sig,
            qrs_on=100,
            r_pos=145,
            qrs_off=170,
            baseline=0.0,
            fs=fs,
            r_amp_mv=1.0,
        )

        self.assertEqual(100, metrics["q_onset"])
        self.assertLessEqual(metrics["q_offset"], 123)
        self.assertGreaterEqual(metrics["q_duration_ms"], 18.0)
        self.assertLessEqual(metrics["q_duration_ms"], 25.0)
        self.assertLess(metrics["q_area_mv_ms"], 4.0)

    def test_q_amplitude_can_exist_without_confident_duration_bounds(self) -> None:
        fs = 1000
        sig = np.zeros(180, dtype=float)
        sig[100:130] = -0.03
        sig[140:151] = np.linspace(0.0, 1.0, 11)

        metrics = _q_component_metrics(
            sig,
            qrs_on=100,
            r_pos=145,
            qrs_off=160,
            baseline=0.0,
            fs=fs,
            r_amp_mv=1.0,
        )

        self.assertIsNone(metrics["q_onset"])
        self.assertIsNone(metrics["q_offset"])
        self.assertIsNone(metrics["q_duration_ms"])
        self.assertIsNone(metrics["q_area_mv_ms"])
        self.assertAlmostEqual(0.03, metrics["q_r_ratio"], places=3)


def _rep(
    lead: str,
    *,
    q: float | None = None,
    r: float | None = None,
    q_dur: float | None = None,
    q_area: float | None = None,
    st: float | None = None,
    t: float | None = None,
) -> RepresentativeLeadFeatures:
    return RepresentativeLeadFeatures(
        lead=lead,
        params={
            "q_amp_mv": q,
            "r_amp_mv": r,
            "q_duration_ms": q_dur,
            "q_area_mv_ms": q_area,
            "st_on_mv": st,
            "t_amp_mv": t,
            "reliable_for_qrs": True,
            "reliable_for_t": True,
        },
        variance={},
    )


class MIEvidenceBuilderTests(unittest.TestCase):
    def test_empty_premature_complex_detection_returns_full_shape(self) -> None:
        result = _detect_premature_complexes([], {})

        self.assertEqual([], result["premature_complexes"])
        self.assertEqual([], result["beat_types"])
        self.assertFalse(result["nsVT"])

    def test_adult_inferior_q_wave_evidence_uses_ratio_and_duration(self) -> None:
        reps = {
            "II": _rep("II", q=-0.22, r=1.0, q_dur=32.0, q_area=3.0),
            "III": _rep("III", q=-0.25, r=0.9, q_dur=34.0, q_area=3.4),
            "aVF": _rep("aVF", q=-0.04, r=1.0, q_dur=20.0, q_area=0.4),
        }

        evidence = build_mi_evidence(
            representative_leads=reps,
            st_elevation_leads={},
            st_depression_leads={},
            stemi_codes=[],
            posterior_mi_suspected=False,
            r_progression_class="normal",
            is_pediatric=False,
        )

        inferior = evidence["territories"]["inferior"]
        self.assertEqual(["II", "III"], inferior["q_wave_leads"])
        self.assertEqual(2, inferior["q_wave_count"])
        self.assertTrue(inferior["q_wave_mi_pattern"])

    def test_anterior_q_next_to_large_lvh_r_wave_is_not_significant(self) -> None:
        # Regression for JS02086: V2/V4 had trivial, unmeasurable-duration Q
        # dips dwarfed by LVH-sized R waves (Q/R ~0.03-0.17) but were still
        # flagged "significant"/"old anterior MI" because the anterior branch
        # only checked absolute Q amplitude, never the Q/R ratio.
        reps = {
            "V2": _rep("V2", q=-0.077, r=0.455, q_dur=None, q_area=None),
            "V4": _rep("V4", q=-0.134, r=4.233, q_dur=None, q_area=None),
        }

        evidence = build_mi_evidence(
            representative_leads=reps,
            st_elevation_leads={},
            st_depression_leads={},
            stemi_codes=[],
            posterior_mi_suspected=False,
            r_progression_class="normal",
            is_pediatric=False,
        )

        self.assertFalse(evidence["q_by_lead"]["V2"]["significant_q"])
        self.assertFalse(evidence["q_by_lead"]["V4"]["significant_q"])
        anterior = evidence["territories"]["anterior"]
        self.assertEqual([], anterior["q_wave_leads"])
        self.assertFalse(anterior["q_wave_mi_pattern"])

    def test_pediatric_large_q_in_two_leads_creates_group_evidence(self) -> None:
        reps = {
            "V5": _rep("V5", q=-0.18, r=0.8, q_dur=30.0, q_area=2.2),
            "V6": _rep("V6", q=-0.19, r=0.9, q_dur=31.0, q_area=2.5),
        }

        evidence = build_mi_evidence(
            representative_leads=reps,
            st_elevation_leads={},
            st_depression_leads={},
            stemi_codes=[],
            posterior_mi_suspected=False,
            r_progression_class="normal",
            is_pediatric=True,
        )

        lateral = evidence["territories"]["lateral"]
        self.assertTrue(lateral["pediatric_large_q_group"])
        self.assertEqual(["V5", "V6"], lateral["q_wave_leads"])

    def test_posterior_evidence_keeps_r_dominant_flag(self) -> None:
        evidence = build_mi_evidence(
            representative_leads={},
            st_elevation_leads={},
            st_depression_leads={"V1": -0.12, "V2": -0.14},
            stemi_codes=[],
            posterior_mi_suspected=True,
            r_progression_class="normal",
            is_pediatric=False,
        )

        posterior = evidence["territories"]["posterior"]
        self.assertTrue(posterior["posterior_mi_suspected"])
        self.assertEqual(["V1", "V2"], posterior["reciprocal_depression_leads"])

    def test_culprit_artery_infers_rca_when_iii_exceeds_ii_and_reciprocal_lateral_depression(self) -> None:
        result = build_culprit_artery_evidence(
            st_elevation_leads={"III": 0.25, "II": 0.10, "aVF": 0.20},
            st_depression_leads={"I": -0.08, "aVL": -0.10},
            available_extended_leads=[],
        )

        self.assertTrue(result["available"])
        self.assertEqual("RCA", result["culprit"])
        self.assertIn("III_greater_than_II", result["criteria"])
        self.assertIn("no_V4R_V7_V8_V9", result["limitations"])

    def test_interpret_wires_initial_qrs_axis_into_mi_evidence(self) -> None:
        reps = {
            "I": _rep("I", q=-0.02, r=1.0),
            "aVF": _rep("aVF", q=-0.02, r=0.0),
        }
        features = ECGFeatures(
            fs=500,
            quality={},
            beats=[],
            beat_features=[],
            representative_leads=reps,
            groups={},
            global_features=GlobalFeatures(
                heart_rate_bpm=60.0,
                atrial_rate_bpm=60.0,
                pr_ms=160.0,
                qrs_ms=90.0,
                qt_ms=400.0,
                qtc_bazett_ms=400.0,
                qtc_fridericia_ms=400.0,
                p_axis_deg=50.0,
                qrs_axis_deg=0.0,
                t_axis_deg=40.0,
                st_axis_deg=None,
                qt_dispersion_ms=None,
            ),
            metadata={
                "rhythm_analysis": {
                    "premature_complexes": {
                        "premature_complexes": [],
                        "bigeminy": None,
                        "trigeminy": False,
                        "non_sustained_vt": False,
                    },
                    "pacing_context": {},
                    "rule_summary": {},
                    "availability": {},
                },
            },
        )

        result = interpret(features)

        self.assertIsNotNone(result.mi_evidence["initial_qrs_axis_deg"])


class MIStatementCandidateTests(unittest.TestCase):
    def test_adult_q_wave_mi_candidate_is_final_without_suppression(self) -> None:
        evidence = {
            "is_pediatric": False,
            "territories": {
                "inferior": {
                    "q_wave_mi_pattern": True,
                    "q_wave_leads": ["II", "III"],
                    "q_wave_mi_ratio_leads": ["II", "III"],
                    "st_elevation_leads": [],
                    "stemi_codes": [],
                },
                "posterior": {"posterior_mi_suspected": False, "reciprocal_depression_leads": []},
            },
        }

        statements = build_mi_statement_candidates(
            evidence,
            bundle_branch_block=None,
            pacing_context={},
        )

        self.assertEqual("old_mi_q_wave", statements[0]["code"])
        self.assertEqual("inferior", statements[0]["territory"])
        self.assertTrue(statements[0]["final"])
        self.assertEqual([], statements[0]["suppressed_by"])

    def test_lbbb_retains_but_suppresses_q_wave_mi_candidate(self) -> None:
        evidence = {
            "is_pediatric": False,
            "territories": {
                "lateral": {
                    "q_wave_mi_pattern": True,
                    "q_wave_leads": ["I", "aVL"],
                    "q_wave_mi_ratio_leads": ["I", "aVL"],
                    "st_elevation_leads": [],
                    "stemi_codes": [],
                },
                "posterior": {"posterior_mi_suspected": False, "reciprocal_depression_leads": []},
            },
        }

        statements = build_mi_statement_candidates(
            evidence,
            bundle_branch_block="LBBB",
            pacing_context={},
        )

        self.assertFalse(statements[0]["final"])
        self.assertEqual(["lbbb_secondary_repolarization"], statements[0]["suppressed_by"])

    def test_continuous_ventricular_pacing_suppresses_candidates(self) -> None:
        evidence = {
            "is_pediatric": False,
            "territories": {
                "anterior": {
                    "q_wave_mi_pattern": True,
                    "q_wave_leads": ["V2", "V3"],
                    "q_wave_mi_ratio_leads": ["V2", "V3"],
                    "st_elevation_leads": [],
                    "stemi_codes": [],
                },
                "posterior": {"posterior_mi_suspected": False, "reciprocal_depression_leads": []},
            },
        }

        statements = build_mi_statement_candidates(
            evidence,
            bundle_branch_block=None,
            pacing_context={"continuous_ventricular_pacing": True},
        )

        self.assertFalse(statements[0]["final"])
        self.assertEqual(["continuous_ventricular_pacing"], statements[0]["suppressed_by"])

    def test_production_continuous_pacing_context_suppresses_candidates(self) -> None:
        pacing_context = classify_pacing_context(
            beats=[
                {"beat_id": 0, "paced": True, "qrs_duration_ms": 160.0},
                {"beat_id": 1, "paced": True, "qrs_duration_ms": 150.0},
            ],
            atrial_events=[],
            spike_times=[100, 600],
        )
        evidence = {
            "is_pediatric": False,
            "territories": {
                "inferior": {
                    "q_wave_mi_pattern": True,
                    "q_wave_leads": ["II", "III"],
                    "q_wave_mi_ratio_leads": ["II", "III"],
                    "st_elevation_leads": [],
                    "stemi_codes": [],
                }
            },
        }

        statements = build_mi_statement_candidates(
            evidence,
            bundle_branch_block=None,
            pacing_context=pacing_context,
        )

        self.assertFalse(statements[0]["final"])
        self.assertEqual(["continuous_ventricular_pacing"], statements[0]["suppressed_by"])

    def test_pediatric_large_q_group_candidate_is_tagged(self) -> None:
        evidence = {
            "is_pediatric": True,
            "territories": {
                "lateral": {
                    "q_wave_mi_pattern": False,
                    "pediatric_large_q_group": True,
                    "q_wave_leads": ["V5", "V6"],
                    "q_wave_mi_ratio_leads": [],
                    "st_elevation_leads": [],
                    "stemi_codes": [],
                },
                "posterior": {"posterior_mi_suspected": False, "reciprocal_depression_leads": []},
            },
        }

        statements = build_mi_statement_candidates(
            evidence,
            bundle_branch_block=None,
            pacing_context={},
        )

        self.assertEqual("pediatric_q_wave_abnormality", statements[0]["code"])
        self.assertEqual("pediatric_morphology", statements[0]["category"])
        self.assertTrue(statements[0]["final"])


class InterpretMIIntegrationTests(unittest.TestCase):
    def test_pathological_q_waves_uses_true_q_duration_from_rep_params(self) -> None:
        # II fails on duration (12ms < 25ms); III and aVF both qualify, so the
        # >=2-lead territory corroboration threshold is met via III+aVF.
        reps = {
            "II": _rep("II", q=-0.22, r=1.0, q_dur=12.0, q_area=1.2),
            "III": _rep("III", q=-0.25, r=0.9, q_dur=34.0, q_area=3.4),
            "aVF": _rep("aVF", q=-0.25, r=0.9, q_dur=34.0, q_area=3.4),
        }

        path_q, territories = _pathological_q_waves(reps, {})

        self.assertFalse(path_q["II"])
        self.assertTrue(path_q["III"])
        self.assertTrue(path_q["aVF"])
        self.assertEqual(["inferior"], territories)

    def test_pathological_q_waves_ignores_anterior_q_dwarfed_by_lvh_r_wave(self) -> None:
        # Regression for JS02086: V2/V4 Q/R ratios of ~0.17 and ~0.03 next to
        # an LVH-sized R wave must not be flagged pathological even though
        # |Q| alone clears the 0.07 mV anterior amplitude floor.
        reps = {
            "V2": _rep("V2", q=-0.077, r=0.455, q_dur=None, q_area=None),
            "V4": _rep("V4", q=-0.134, r=4.233, q_dur=None, q_area=None),
        }

        path_q, territories = _pathological_q_waves(reps, {})

        self.assertFalse(path_q["V2"])
        self.assertFalse(path_q["V4"])
        self.assertEqual([], territories)

    def test_pathological_q_waves_suppresses_isolated_single_lead_hit(self) -> None:
        # A lone qualifying lead in a territory (no corroborating neighbour)
        # must not be reported -- isolated Q waves are a common normal variant.
        reps = {
            "III": _rep("III", q=-0.25, r=0.9, q_dur=34.0, q_area=3.4),
        }

        path_q, territories = _pathological_q_waves(reps, {})

        self.assertFalse(path_q["III"])
        self.assertEqual([], territories)


if __name__ == "__main__":
    unittest.main()
