import types
import unittest

from feature_extraction.ecgfeat.interpret import _peds_classify_qrs_width, interpret
from feature_extraction.ecgfeat.models import (
    ECGFeatures,
    GlobalFeatures,
    PatientMeta,
    RepresentativeLeadFeatures,
    resolve_patient_age,
)
from feature_extraction.ecgfeat.pediatric_rules import (
    build_pediatric_hypertrophy_evidence,
    build_pediatric_repolarization_candidates,
    pediatric_age_bin,
    pediatric_voltage_threshold,
    validate_pediatric_tables,
)


def _rep(lead: str, **params: object) -> object:
    return types.SimpleNamespace(lead=lead, params=params)


def _feature_rep(lead: str, **params: object) -> RepresentativeLeadFeatures:
    return RepresentativeLeadFeatures(
        lead=lead,
        params=params,
        variance={},
    )


def _minimal_pediatric_features(
    representative_leads: dict[str, RepresentativeLeadFeatures],
) -> ECGFeatures:
    return ECGFeatures(
        fs=500,
        quality={},
        beats=[],
        beat_features=[],
        representative_leads=representative_leads,
        groups={},
        global_features=GlobalFeatures(
            heart_rate_bpm=80.0,
            atrial_rate_bpm=80.0,
            pr_ms=140.0,
            qrs_ms=80.0,
            qt_ms=360.0,
            qtc_bazett_ms=400.0,
            qtc_fridericia_ms=390.0,
            p_axis_deg=50.0,
            qrs_axis_deg=50.0,
            t_axis_deg=35.0,
            st_axis_deg=None,
            qt_dispersion_ms=None,
        ),
        metadata={
            "patient_meta": PatientMeta(age=8.0, sex="M"),
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


class PediatricRulesTests(unittest.TestCase):
    def test_pediatric_age_bin_routes_infant_child_and_teen(self) -> None:
        self.assertEqual("0_23h", pediatric_age_bin(0.002))
        self.assertEqual("1_11m", pediatric_age_bin(0.5))
        self.assertEqual("8_11y", pediatric_age_bin(8.0))
        self.assertEqual("12_15y", pediatric_age_bin(15.0))

    def test_pediatric_voltage_threshold_returns_configured_rvh_limit(self) -> None:
        threshold = pediatric_voltage_threshold(
            age_years=8.0,
            sex="M",
            criterion="rvh_r_v1_98p_mv",
        )

        self.assertIsNotNone(threshold)
        self.assertGreater(threshold, 0.0)

    def test_pediatric_voltage_tables_cover_all_age_bins_and_required_criteria(self) -> None:
        validate_pediatric_tables()

    def test_appendix_a_values_are_converted_from_mm_to_mv(self) -> None:
        self.assertEqual(
            1.2,
            pediatric_voltage_threshold(8.0, "M", "rvh_r_v1_98p_mv"),
        )
        self.assertEqual(
            2.55,
            pediatric_voltage_threshold(8.0, "M", "lvh_r_v6_98p_mv"),
        )
        self.assertEqual(
            4.55,
            pediatric_voltage_threshold(8.0, "M", "lvh_sv1_rv6_98p_mv"),
        )

    def test_missing_dxl_appendix_a_columns_are_not_fabricated(self) -> None:
        self.assertIsNone(
            pediatric_voltage_threshold(8.0, "M", "rvh_r_v2_98p_mv")
        )
        self.assertIsNone(
            pediatric_voltage_threshold(8.0, "M", "lvh_r_i_98p_mv")
        )

    def test_unknown_voltage_criterion_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            pediatric_voltage_threshold(
                age_years=8.0,
                sex="M",
                criterion="rvh_typo",
            )

    def test_pediatric_rvh_evidence_uses_age_specific_voltage(self) -> None:
        reps = {
            "V1": _rep("V1", r_amp_mv=1.4, s_amp_mv=-0.1, reliable_for_qrs=True),
            "V6": _rep("V6", r_amp_mv=0.8, s_amp_mv=-0.9, reliable_for_qrs=True),
        }

        evidence = build_pediatric_hypertrophy_evidence(
            representative_leads=reps,
            age_years=8.0,
            sex="M",
            qrs_axis_class="RAD",
            bundle_branch_block=None,
        )

        self.assertIn(evidence["rvh"]["class"], ("consider", "probable", "definitive"))
        self.assertIn("rvh_r_v1_98p_mv", evidence["rvh"]["criteria"])

    def test_pediatric_rvh_is_bypassed_in_rbbb(self) -> None:
        evidence = build_pediatric_hypertrophy_evidence(
            representative_leads={},
            age_years=8.0,
            sex="M",
            qrs_axis_class="RAD",
            bundle_branch_block="RBBB",
        )

        self.assertIsNone(evidence["rvh"]["class"])
        self.assertEqual(["RBBB"], evidence["rvh"]["bypassed_by"])

    def test_interpret_pediatric_rvh_does_not_leak_adult_voltage_score(self) -> None:
        features = _minimal_pediatric_features(
            {
                "V1": _feature_rep(
                    "V1",
                    r_amp_mv=0.4,
                    s_amp_mv=-0.1,
                    reliable_for_qrs=True,
                ),
                "V6": _feature_rep(
                    "V6",
                    r_amp_mv=0.8,
                    s_amp_mv=-0.9,
                    reliable_for_qrs=True,
                ),
            }
        )

        result = interpret(features)

        self.assertIsNone(result.pediatric_hypertrophy_evidence["rvh"]["class"])
        self.assertIsNone(result.rvh_class)

    def test_age_days_overrides_conflicting_adult_age_in_interpretation(self) -> None:
        features = _minimal_pediatric_features({})
        features.metadata["patient_meta"] = PatientMeta(
            age=40.0,
            age_days=30.0,
            sex="M",
        )

        result = interpret(features)

        self.assertTrue(result.is_pediatric)
        self.assertTrue(result.pediatric_hypertrophy_evidence)

    def test_invalid_explicit_age_days_fails_closed_without_year_fallback(self) -> None:
        resolved = resolve_patient_age(
            PatientMeta(age=40.0, age_days=float("nan"), sex="M")
        )

        self.assertFalse(resolved.known)
        self.assertIsNone(resolved.age_years)
        self.assertEqual("invalid_age_days", resolved.source)

    def test_interpret_pediatric_bvh_matches_exported_evidence_for_rs_sum(self) -> None:
        features = _minimal_pediatric_features(
            {
                "V1": _feature_rep(
                    "V1",
                    r_amp_mv=0.4,
                    s_amp_mv=-0.1,
                    reliable_for_qrs=True,
                ),
                "V2": _feature_rep(
                    "V2",
                    r_amp_mv=3.2,
                    s_amp_mv=-3.1,
                    reliable_for_qrs=True,
                ),
                "V3": _feature_rep(
                    "V3",
                    r_amp_mv=3.3,
                    s_amp_mv=-3.0,
                    reliable_for_qrs=True,
                ),
                "V6": _feature_rep(
                    "V6",
                    r_amp_mv=0.8,
                    s_amp_mv=-0.9,
                    reliable_for_qrs=True,
                ),
            }
        )

        result = interpret(features)
        bvh_evidence = result.pediatric_hypertrophy_evidence["bvh"]

        self.assertEqual(bvh_evidence["suspected"], result.bvh_suspected)
        self.assertTrue(bvh_evidence["suspected"])
        self.assertIn("bvh_rs_sum_v2_v3_v4_mv", bvh_evidence["criteria"])

    def test_bvh_tall_r_v1_with_lvh_flags_biventricular(self) -> None:
        # R V1 = 1.1 mV (tall, but < 1.2 mV RVH limit) with an LVH-level R V6
        # (2.6 mV >= 2.55 mV) is biventricular.
        reps = {
            "V1": _rep("V1", r_amp_mv=1.1, s_amp_mv=-0.1, reliable_for_qrs=True),
            "V6": _rep("V6", r_amp_mv=2.6, s_amp_mv=-0.1, reliable_for_qrs=True),
        }

        evidence = build_pediatric_hypertrophy_evidence(
            representative_leads=reps,
            age_years=8.0,
            sex="M",
            qrs_axis_class="normal",
            bundle_branch_block=None,
        )

        self.assertIsNone(evidence["rvh"]["class"])          # V1 below RVH limit
        self.assertIsNotNone(evidence["lvh"]["class"])       # V6 meets LVH limit
        self.assertTrue(evidence["bvh"]["suspected"])
        self.assertIn("bvh_r_v1_with_lvh_mv", evidence["bvh"]["criteria"])
        self.assertNotIn("rvh_lvh_combination", evidence["bvh"]["criteria"])

    def test_bvh_isolated_tall_r_v6_without_right_sign_does_not_flag(self) -> None:
        # Tall R V6 (1.5 mV) but below the LVH limit, no right-sided sign,
        # no RS-sum leads -> must NOT flag BVH.
        reps = {
            "V1": _rep("V1", r_amp_mv=0.5, s_amp_mv=-0.1, reliable_for_qrs=True),
            "V6": _rep("V6", r_amp_mv=1.5, s_amp_mv=-0.1, reliable_for_qrs=True),
        }

        evidence = build_pediatric_hypertrophy_evidence(
            representative_leads=reps,
            age_years=8.0,
            sex="M",
            qrs_axis_class="normal",
            bundle_branch_block=None,
        )

        self.assertFalse(evidence["bvh"]["suspected"])
        self.assertEqual([], evidence["bvh"]["criteria"])

    def test_bvh_septal_q_v6_requires_amplitude_and_duration_with_right_sign(self) -> None:
        base = {
            "V1": _rep("V1", r_amp_mv=0.4, s_amp_mv=-0.1, reliable_for_qrs=True),
        }
        # Deep + wide septal Q in V6 with a right-axis (right) sign -> flags.
        reps_pos = {
            **base,
            "V6": _rep(
                "V6",
                r_amp_mv=0.2,
                s_amp_mv=-0.1,
                q_amp_mv=-0.09,
                q_duration_ms=12.0,
                reliable_for_qrs=True,
            ),
        }
        evidence_pos = build_pediatric_hypertrophy_evidence(
            representative_leads=reps_pos,
            age_years=8.0,
            sex="M",
            qrs_axis_class="RAD",
            bundle_branch_block=None,
        )
        self.assertIn("bvh_q_v6_septal", evidence_pos["bvh"]["criteria"])

        # Same amplitude but Q too narrow (8 ms < 10 ms) -> no septal criterion.
        reps_narrow = {
            **base,
            "V6": _rep(
                "V6",
                r_amp_mv=0.2,
                s_amp_mv=-0.1,
                q_amp_mv=-0.09,
                q_duration_ms=8.0,
                reliable_for_qrs=True,
            ),
        }
        evidence_narrow = build_pediatric_hypertrophy_evidence(
            representative_leads=reps_narrow,
            age_years=8.0,
            sex="M",
            qrs_axis_class="RAD",
            bundle_branch_block=None,
        )
        self.assertNotIn("bvh_q_v6_septal", evidence_narrow["bvh"]["criteria"])

    def test_peds_rbbb_requires_r_prime_duration_when_measured(self) -> None:
        reps = {
            "V1": _rep("V1", r_amp_mv=0.5, reliable_for_qrs=True),
            "I": _rep("I", r_amp_mv=0.0, s_amp_mv=-0.2, reliable_for_qrs=True),
            "V6": _rep("V6", r_amp_mv=0.2, s_amp_mv=-0.2, reliable_for_qrs=True),
        }
        r_prime = {"V1": 0.30}          # >= 0.15 mV amplitude
        qrs_ms = 160.0                  # comfortably in the BBB range for age 8

        # Duration measured and >= 20 ms -> RBBB.
        _, bbb_long = _peds_classify_qrs_width(
            qrs_ms, 8.0, reps, r_prime, {"V1": 25.0}
        )
        self.assertEqual("RBBB", bbb_long)

        # Duration unavailable -> fall back to amplitude-only -> RBBB.
        _, bbb_missing = _peds_classify_qrs_width(qrs_ms, 8.0, reps, r_prime, None)
        self.assertEqual("RBBB", bbb_missing)

        # Duration measured but < 20 ms -> RBBB suppressed (IVCD instead).
        _, bbb_short = _peds_classify_qrs_width(
            qrs_ms, 8.0, reps, r_prime, {"V1": 12.0}
        )
        self.assertEqual("IVCD", bbb_short)

    def test_pediatric_st_t_statement_is_suppressed_by_vcd_or_hypertrophy(self) -> None:
        statements = build_pediatric_repolarization_candidates(
            st_depression_leads={"V5": -0.12, "V6": -0.10},
            inverted_t_leads=["V5", "V6"],
            hypertrophy_evidence={
                "lvh": {"class": "probable"},
                "rvh": {"class": None},
            },
            bundle_branch_block="LBBB",
        )

        self.assertEqual("secondary_st_t_abnormality", statements[0]["code"])
        self.assertIn("LBBB", statements[0]["suppressed_by"])
        self.assertIn("LVH", statements[0]["suppressed_by"])


if __name__ == "__main__":
    unittest.main()
