from __future__ import annotations

import unittest

from feature_extraction.ecgfeat.export import prepare_json_export


class PrepareJsonExportTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "beat_features": [{"lead": "I", "r_amp_mv": 0.123456789}],
            "representative_leads": {"I": {"r_amp_mv": 0.987654321}},
            "clinical_interpretation": {"overall_status": "abnormal"},
        }

    def test_drops_beat_features_by_default(self) -> None:
        result = prepare_json_export(self._payload())

        self.assertNotIn("beat_features", result)
        self.assertIn("representative_leads", result)
        self.assertIn("clinical_interpretation", result)

    def test_include_beat_features_keeps_it(self) -> None:
        result = prepare_json_export(
            self._payload(), include_beat_features=True, round_ndigits=None
        )

        self.assertIn("beat_features", result)
        self.assertEqual(
            [{"lead": "I", "r_amp_mv": 0.123456789}], result["beat_features"]
        )

    def test_debug_profile_keeps_beat_features(self) -> None:
        result = prepare_json_export(self._payload(), profile="debug")

        self.assertIn("beat_features", result)

    def test_summary_profile_removes_repeated_audit_details(self) -> None:
        payload = {
            **self._payload(),
            "p_wave_assessments": [{"beat_id": 1}],
            "metadata": {"rhythm_analysis": {"atrial_events": [1, 2]}},
            "morphology_inputs": {
                "native_beat_profiles": [{"beat_id": 1}],
                "statement_evidence": {"available": True},
            },
        }

        result = prepare_json_export(payload, profile="summary")

        self.assertNotIn("beat_features", result)
        self.assertNotIn("p_wave_assessments", result)
        self.assertNotIn("rhythm_analysis", result["metadata"])
        self.assertNotIn("native_beat_profiles", result["morphology_inputs"])
        self.assertIn("statement_evidence", result["morphology_inputs"])
        self.assertIn("p_wave_assessments", payload)

    def test_rejects_invalid_or_conflicting_profile(self) -> None:
        with self.assertRaises(ValueError):
            prepare_json_export(self._payload(), profile="unknown")
        with self.assertRaises(ValueError):
            prepare_json_export(
                self._payload(),
                profile="summary",
                include_beat_features=True,
            )

    def test_rounds_floats_to_default_precision(self) -> None:
        result = prepare_json_export(self._payload())

        self.assertEqual(0.987654, result["representative_leads"]["I"]["r_amp_mv"])

    def test_round_ndigits_none_disables_rounding(self) -> None:
        result = prepare_json_export(self._payload(), round_ndigits=None)

        self.assertEqual(
            0.987654321, result["representative_leads"]["I"]["r_amp_mv"]
        )

    def test_does_not_mutate_input_payload(self) -> None:
        payload = self._payload()
        prepare_json_export(payload)

        self.assertIn("beat_features", payload)
        self.assertEqual(0.987654321, payload["representative_leads"]["I"]["r_amp_mv"])

    def test_nested_structures_are_rounded(self) -> None:
        payload = {"a": [{"b": 1.23456789}, 2.3456789]}

        result = prepare_json_export(payload, round_ndigits=3)

        self.assertEqual(1.235, result["a"][0]["b"])
        self.assertEqual(2.346, result["a"][1])

    def test_nonfinite_floats_are_exported_as_null(self) -> None:
        payload = {
            "a": float("nan"),
            "nested": [float("inf"), -float("inf")],
        }

        rounded = prepare_json_export(payload, round_ndigits=3)
        unrounded = prepare_json_export(payload, round_ndigits=None)

        self.assertEqual({"a": None, "nested": [None, None]}, rounded)
        self.assertEqual({"a": None, "nested": [None, None]}, unrounded)

    def test_removes_retired_glasgow_interpretation_fields(self) -> None:
        payload = {
            **self._payload(),
            "glasgow": {"schema_version": "glasgow_rules.v2"},
            "metadata": {
                "glasgow_analysis": {"schema_version": "glasgow_rules.v2"},
                "input_fs": 500,
            },
        }

        result = prepare_json_export(payload, round_ndigits=None)

        self.assertNotIn("glasgow", result)
        self.assertNotIn("glasgow_analysis", result["metadata"])
        self.assertEqual(500, result["metadata"]["input_fs"])
        self.assertIn("glasgow", payload)
        self.assertIn("glasgow_analysis", payload["metadata"])


if __name__ == "__main__":
    unittest.main()
