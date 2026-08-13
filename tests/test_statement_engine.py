import unittest

from feature_extraction.ecgfeat.statement_engine import (
    CandidateStatement,
    resolve_statement_candidates,
)


class StatementEngineTests(unittest.TestCase):
    def test_resolution_exports_final_suppressed_bypassed_and_unavailable_buckets(self) -> None:
        resolution = resolve_statement_candidates(
            [
                CandidateStatement(
                    code="wpw_pattern",
                    category="preexcitation",
                    severity=4,
                    evidence={"delta_leads": ["I", "II"], "qrs_ms": 112.0},
                    confidence=0.95,
                    source="rhythm",
                    bypasses=["mi", "hypertrophy", "st_t"],
                ),
                CandidateStatement(
                    code="old_mi_q_wave",
                    category="mi",
                    severity=2,
                    evidence={"q_wave_leads": ["II", "III"]},
                    source="morphology",
                ),
                CandidateStatement(
                    code="sinus_rhythm",
                    category="sinus",
                    severity=0,
                    evidence={},
                    required_inputs=["p_axis"],
                    unavailable_inputs=["p_axis"],
                    source="rhythm",
                ),
            ]
        ).to_dict()

        self.assertTrue(resolution["available"])
        self.assertEqual(["wpw_pattern"], [item["code"] for item in resolution["final"]])
        self.assertEqual(["old_mi_q_wave"], [item["code"] for item in resolution["suppressed"]])
        self.assertEqual(["sinus_rhythm"], [item["code"] for item in resolution["unavailable"]])
        self.assertEqual("suppressed", resolution["suppressed"][0]["status"])
        self.assertIn("wpw_pattern", resolution["suppressed"][0]["suppressed_by"])
        self.assertEqual(
            ["hypertrophy", "mi", "pediatric_morphology", "st_t"],
            sorted(item["category"] for item in resolution["bypassed"]),
        )

    def test_lbbb_suppresses_mi_hypertrophy_and_st_t_candidates(self) -> None:
        resolution = resolve_statement_candidates(
            [
                CandidateStatement(
                    code="lbbb",
                    category="conduction",
                    severity=3,
                    evidence={"qrs_ms": 154.0},
                    source="morphology",
                ),
                CandidateStatement(
                    code="acute_mi_st_elevation",
                    category="mi",
                    severity=4,
                    evidence={"st_elevation_leads": ["V2", "V3"]},
                    source="mi",
                ),
                CandidateStatement(
                    code="lvh_voltage",
                    category="hypertrophy",
                    severity=2,
                    evidence={"criteria": ["sokolow_lyon"]},
                    source="morphology",
                ),
                CandidateStatement(
                    code="st_t_abnormality",
                    category="st_t",
                    severity=1,
                    evidence={"leads": ["I", "aVL"]},
                    source="morphology",
                ),
            ],
            context={"bundle_branch_block": "LBBB"},
        ).to_dict()

        self.assertEqual(["lbbb"], [item["code"] for item in resolution["final"]])
        self.assertEqual(
            ["acute_mi_st_elevation", "lvh_voltage", "st_t_abnormality"],
            sorted(item["code"] for item in resolution["suppressed"]),
        )
        for item in resolution["suppressed"]:
            self.assertIn("lbbb_secondary_repolarization", item["suppressed_by"])

    def test_continuous_ventricular_pacing_bypasses_dependent_categories(self) -> None:
        resolution = resolve_statement_candidates(
            [
                CandidateStatement(
                    code="paced_rhythm",
                    category="pacing",
                    severity=3,
                    evidence={"continuous_pacing": True, "ventricular_pacing_present": True},
                    source="rhythm",
                ),
                CandidateStatement(
                    code="probable_af",
                    category="af_afl",
                    severity=3,
                    evidence={"rr_cv": 0.22},
                    source="rhythm",
                ),
                CandidateStatement(
                    code="old_mi_q_wave",
                    category="mi",
                    severity=2,
                    evidence={"q_wave_leads": ["II"]},
                    source="mi",
                ),
            ],
            context={"pacing_context": {"continuous_pacing": True, "ventricular_pacing_present": True}},
        ).to_dict()

        self.assertEqual(["paced_rhythm"], [item["code"] for item in resolution["final"]])
        self.assertEqual(["old_mi_q_wave", "probable_af"], sorted(item["code"] for item in resolution["suppressed"]))
        self.assertIn("continuous_ventricular_pacing", resolution["suppressed"][0]["suppressed_by"])
        self.assertIn("rhythm", {item["category"] for item in resolution["bypassed"]})

    def test_resolution_exports_generic_summary_code(self) -> None:
        abnormal = resolve_statement_candidates(
            [
                CandidateStatement(
                    code="old_mi_q_wave",
                    category="mi",
                    severity="old_or_age_indeterminate",
                    evidence={"q_wave_leads": ["II", "III"]},
                    source="mi",
                )
            ]
        ).to_dict()

        self.assertEqual(
            {"code": 5, "label": "Abnormal ECG", "source": "candidate_resolver_derived"},
            abnormal["summary_code"],
        )

        borderline = resolve_statement_candidates(
            [
                CandidateStatement(
                    code="borderline_qtc",
                    category="qtc",
                    severity="borderline",
                    evidence={"qtc_ms": 470.0},
                    source="intervals",
                )
            ]
        ).to_dict()

        self.assertEqual(4, borderline["summary_code"]["code"])
        self.assertEqual("Borderline ECG", borderline["summary_code"]["label"])

        technical = resolve_statement_candidates(
            [
                CandidateStatement(
                    code="measurement_error",
                    category="technical",
                    severity="critical",
                    evidence={},
                    source="quality",
                )
            ]
        ).to_dict()

        self.assertEqual(6, technical["summary_code"]["code"])
        self.assertEqual("Technical error", technical["summary_code"]["label"])

    def test_rate_only_summary_code_is_normal_except_for_rate(self) -> None:
        resolution = resolve_statement_candidates(
            [
                CandidateStatement(
                    code="sinus_tachycardia",
                    category="rate",
                    severity="borderline",
                    evidence={"heart_rate_bpm": 105.0},
                    source="rhythm",
                )
            ]
        ).to_dict()

        self.assertEqual(2, resolution["summary_code"]["code"])
        self.assertEqual("Normal ECG except for rate", resolution["summary_code"]["label"])


if __name__ == "__main__":
    unittest.main()
