from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

if "wfdb" not in sys.modules:
    sys.modules["wfdb"] = types.SimpleNamespace(rdrecord=None, rdann=None)

import compare_annotations as compare
from feature_extraction.ecgfeat.models import ECGFeatures, GlobalFeatures


class CompareAnnotationsReportTests(unittest.TestCase):
    def _features_with_profile(self) -> ECGFeatures:
        return ECGFeatures(
            fs=500,
            quality={},
            beats=[],
            beat_features=[],
            representative_leads={},
            groups={},
            global_features=GlobalFeatures(
                heart_rate_bpm=60.0,
                atrial_rate_bpm=60.0,
                pr_ms=150.0,
                qrs_ms=90.0,
                qt_ms=400.0,
                qtc_bazett_ms=410.0,
                qtc_fridericia_ms=405.0,
                p_axis_deg=40.0,
                qrs_axis_deg=55.0,
                t_axis_deg=35.0,
                st_axis_deg=20.0,
                qt_dispersion_ms=25.0,
            ),
            metadata={
                "twelve_sl_measurement_profile": {
                    "profile_version": "12sl_measurement_profile_v1",
                    "source": "parallel_profile_no_diagnostic_override",
                    "heart_rate_first_last_bpm": 59.4,
                    "global_fiducials": {
                        "p_onset_offset_ms": -102.5,
                        "qrs_onset_offset_ms": -40.0,
                        "qrs_offset_offset_ms": 50.0,
                        "t_offset_offset_ms": 260.0,
                    },
                }
            },
        )

    def test_choose_reference_gt_lead_prefers_ii(self) -> None:
        gt_by_lead = {
            "I": [{"r_sample": 100}],
            "II": [{"r_sample": 100}, {"r_sample": 200}],
            "V1": [{"r_sample": 100}, {"r_sample": 200}, {"r_sample": 300}],
        }

        self.assertEqual("II", compare.choose_reference_gt_lead(gt_by_lead))

    def test_build_annotation_derived_result_uses_annotation_intervals(self) -> None:
        t = np.linspace(0.0, 2.0, 1000, endpoint=False)
        ecg = np.vstack(
            [
                0.03 * np.sin(2.0 * np.pi * (1.0 + lead * 0.02) * t)
                for lead in range(12)
            ]
        )
        for lead_idx in range(12):
            ecg[lead_idx, 100] += 1.00
            ecg[lead_idx, 160] += 0.20
            ecg[lead_idx, 375] += 0.10
            ecg[lead_idx, 400] += 1.00
            ecg[lead_idx, 460] += 0.20

        gt_by_lead = {
            lead: [
                {
                    "r_sample": 100,
                    "qrs_on": 90,
                    "qrs_off": 110,
                    "t_on": 130,
                    "t_peak": 160,
                    "t_off": 210,
                    "p_on": 360,
                    "p_peak": 375,
                    "p_off": 385,
                },
                {
                    "r_sample": 400,
                    "qrs_on": 390,
                    "qrs_off": 410,
                    "t_on": 430,
                    "t_peak": 460,
                    "t_off": 510,
                    "p_on": None,
                    "p_peak": None,
                    "p_off": None,
                },
            ]
            for lead in compare.STANDARD_12_LEADS
        }

        result = compare.build_annotation_derived_result(ecg, 500, gt_by_lead)

        self.assertEqual(2, len(result.beats))
        self.assertAlmostEqual(60.0, result.global_features.pr_ms)
        self.assertAlmostEqual(40.0, result.global_features.qrs_ms)
        self.assertAlmostEqual(240.0, result.global_features.qt_ms)

    def test_build_annotation_derived_result_does_not_use_algorithm_global_fusion(self) -> None:
        t = np.linspace(0.0, 2.0, 1000, endpoint=False)
        ecg = np.vstack(
            [
                0.03 * np.sin(2.0 * np.pi * (1.0 + lead * 0.02) * t)
                for lead in range(12)
            ]
        )
        for lead_idx in range(12):
            ecg[lead_idx, 100] += 1.00
            ecg[lead_idx, 160] += 0.20
            ecg[lead_idx, 375] += 0.10
            ecg[lead_idx, 400] += 1.00
            ecg[lead_idx, 460] += 0.20

        gt_by_lead = {
            lead: [
                {
                    "r_sample": 100,
                    "qrs_on": 90,
                    "qrs_off": 110,
                    "t_on": 130,
                    "t_peak": 160,
                    "t_off": 210,
                    "p_on": 360,
                    "p_peak": 375,
                    "p_off": 385,
                },
                {
                    "r_sample": 400,
                    "qrs_on": 390,
                    "qrs_off": 410,
                    "t_on": 430,
                    "t_peak": 460,
                    "t_off": 510,
                    "p_on": None,
                    "p_peak": None,
                    "p_off": None,
                },
            ]
            for lead in compare.STANDARD_12_LEADS
        }

        with patch.object(compare, "compute_global_features", side_effect=AssertionError("algorithm fusion called")):
            result = compare.build_annotation_derived_result(ecg, 500, gt_by_lead)

        self.assertAlmostEqual(40.0, result.global_features.qrs_ms)
        self.assertAlmostEqual(240.0, result.global_features.qt_ms)

    def test_build_annotation_lead_feature_populates_signed_wave_areas(self) -> None:
        sig = np.zeros(80)
        sig[8:13] = np.array([-0.04, -0.05, -0.06, -0.05, -0.04])
        sig[20:25] = np.array([-0.15, 0.80, -0.20, -0.10, -0.05])
        sig[40:47] = np.array([-0.20, -0.20, 0.05, -0.20, -0.20, -0.20, -0.20])

        feature = compare._build_annotation_lead_feature(
            sig,
            fs=1000,
            lead="I",
            beat_id=0,
            canonical_r=21,
            annotation={
                "r_sample": 21,
                "qrs_on": 20,
                "qrs_off": 24,
                "t_on": 40,
                "t_peak": 42,
                "t_off": 46,
            },
            p_annotation={
                "p_on": 8,
                "p_peak": 10,
                "p_off": 12,
            },
        )

        self.assertIsNotNone(feature.p_signed_area)
        self.assertIsNotNone(feature.qrs_signed_area)
        self.assertIsNotNone(feature.t_signed_area)
        self.assertLess(feature.p_signed_area, 0.0)
        self.assertGreater(feature.qrs_signed_area, 0.0)
        self.assertLess(feature.t_signed_area, 0.0)

    def test_build_global_comparison_rows_reports_numeric_deltas(self) -> None:
        rows = compare.build_global_comparison_rows(
            {"HR": 60.0, "PR": 160.0},
            {"HR": 50.0, "PR": 180.0},
        )

        self.assertIn(("HR", 60.0, 50.0, 10.0), rows)
        self.assertIn(("PR", 160.0, 180.0, -20.0), rows)

    def test_result_global_metrics_can_select_hybrid_measurement_profile(self) -> None:
        result = self._features_with_profile()

        native = compare._result_global_metrics(result, measurement_profile="native")
        hybrid = compare._result_global_metrics(result, measurement_profile="hybrid")
        twelve_sl = compare._result_global_metrics(result, measurement_profile="12sl")

        self.assertEqual(60.0, native["HR"])
        self.assertEqual(400.0, native["QT"])
        self.assertEqual(59.4, hybrid["HR"])
        self.assertEqual(90.0, hybrid["QRS"])
        self.assertEqual(300.0, hybrid["QT"])
        self.assertEqual(25.0, hybrid["QT Dispersion"])
        self.assertEqual(40.0, hybrid["P Axis"])
        self.assertEqual(59.4, twelve_sl["HR"])
        self.assertEqual(90.0, twelve_sl["QRS"])
        self.assertEqual(300.0, twelve_sl["QT"])

    def test_build_batch_summary_row_records_measurement_profile(self) -> None:
        algorithm_result = self._features_with_profile()
        gt_result = self._features_with_profile()

        row = compare.build_batch_summary_row(
            "7",
            algorithm_result,
            gt_result,
            measurement_profile="hybrid",
        )

        self.assertEqual("hybrid", row["measurement_profile"])
        self.assertEqual(59.4, row["algorithm_hr"])
        self.assertEqual(300.0, row["algorithm_qt"])

    def test_build_batch_summary_row_adds_circular_axis_differences(self) -> None:
        algorithm_result = self._features_with_profile()
        gt_result = self._features_with_profile()
        algorithm_result.global_features.t_axis_deg = 176.9
        gt_result.global_features.t_axis_deg = -175.3

        row = compare.build_batch_summary_row(
            "61",
            algorithm_result,
            gt_result,
            measurement_profile="native",
        )

        self.assertAlmostEqual(352.2, row["diff_t_axis"], places=1)
        self.assertAlmostEqual(7.8, row["abs_t_axis_circular_diff"], places=1)
        self.assertAlmostEqual(0.0, row["abs_p_axis_circular_diff"], places=1)
        self.assertAlmostEqual(0.0, row["abs_qrs_axis_circular_diff"], places=1)

    def test_write_visual_comparison_bundle_calls_existing_plotters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with (
                patch.object(compare, "plot_single_lead") as plot_single_lead,
                patch.object(compare, "plot_all_leads_full_record") as plot_all_leads_full,
                patch.object(compare, "plot_single_beat") as plot_single_beat,
                patch.object(compare, "plot_all_leads_one_beat") as plot_all_leads,
            ):
                artifacts = compare.write_visual_comparison_bundle(
                    record_id="7",
                    out_dir=out_dir,
                    lead="V2",
                    beat_idx=3,
                    context_ms=140,
                )

        self.assertEqual(out_dir / "7_V2_compare.png", artifacts["full_record_plot_path"])
        self.assertEqual(
            out_dir / "7_all_leads_compare.png",
            artifacts["all_leads_full_record_plot_path"],
        )
        self.assertEqual(out_dir / "7_V2_beat3_compare.png", artifacts["beat_plot_path"])
        self.assertEqual(out_dir / "7_beat3_all_leads_compare.png", artifacts["grid_plot_path"])
        plot_single_lead.assert_called_once_with("7", "V2", str(out_dir / "7_V2_compare.png"))
        plot_all_leads_full.assert_called_once_with("7", str(out_dir / "7_all_leads_compare.png"))
        plot_single_beat.assert_called_once_with("7", "V2", 3, 140, str(out_dir / "7_V2_beat3_compare.png"))
        plot_all_leads.assert_called_once_with("7", 3, 140, str(out_dir / "7_beat3_all_leads_compare.png"))

    def test_resolve_batch_workers_auto_uses_cpu_count_bounded_by_records(self) -> None:
        with patch.object(compare.os, "cpu_count", return_value=16):
            self.assertEqual(3, compare._resolve_batch_workers(0, 3))
            self.assertEqual(16, compare._resolve_batch_workers(None, 40))
            self.assertEqual(1, compare._resolve_batch_workers(1, 40))
            self.assertEqual(5, compare._resolve_batch_workers(8, 5))

    def test_run_batch_report_compare_parallel_keeps_summary_order(self) -> None:
        class ImmediateFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class FakeExecutor:
            seen_max_workers = None

            def __init__(self, max_workers):
                FakeExecutor.seen_max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                return ImmediateFuture(fn(*args))

        def fake_job(record_id, *_args):
            return {
                "record_id": record_id,
                "summary_row": {"record_id": record_id, "diff_qrs": float(record_id)},
                "comparison_path": Path(record_id) / f"{record_id}_comparison.txt",
            }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with (
                patch.object(compare, "ProcessPoolExecutor", FakeExecutor),
                patch.object(compare, "as_completed", lambda futures: list(reversed(futures))),
                patch.object(compare, "_run_batch_report_job", side_effect=fake_job),
            ):
                compare.run_batch_report_compare(
                    out_dir=out_dir,
                    record_ids=["1", "2", "3"],
                    include_visuals=False,
                    workers=2,
                )

            summary_lines = (out_dir / "summary.csv").read_text(encoding="utf-8").splitlines()

        self.assertEqual(2, FakeExecutor.seen_max_workers)
        self.assertEqual(
            ["record_id,diff_qrs", "1,1.0", "2,2.0", "3,3.0"],
            summary_lines,
        )


if __name__ == "__main__":
    unittest.main()
