from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import demo_feature_extraction as demo


class ECGReportSheetTests(unittest.TestCase):
    @staticmethod
    def make_hdr() -> dict:
        return {
            "record": "rec001",
            "fs": 100,
            "n_samples": 1000,
            "age": 61,
            "sex": "Female",
            "dx": ["426783006"],
            "rx": "Unknown",
            "hx": "dataset=test",
            "sx": "unit-test",
        }

    @staticmethod
    def make_ecg() -> np.ndarray:
        t = np.linspace(0.0, 10.0, 1000, endpoint=False)
        return np.vstack([0.4 * np.sin(2.0 * np.pi * (1.0 + lead * 0.03) * t) for lead in range(12)])

    @staticmethod
    def make_result() -> SimpleNamespace:
        gf = SimpleNamespace(
            heart_rate_bpm=88.0,
            atrial_rate_bpm=88.0,
            pr_ms=154.0,
            qrs_ms=92.0,
            qt_ms=360.0,
            qtc_bazett_ms=410.0,
            qtc_fridericia_ms=392.0,
            p_axis_deg=50.0,
            qrs_axis_deg=35.0,
            t_axis_deg=20.0,
            st_axis_deg=None,
            qt_dispersion_ms=44.0,
            ptf_v1_mv_ms=None,
        )
        interp = SimpleNamespace(
            heart_rate_class="normal",
            rr_irregularity_class="regular",
            rr_cv=0.04,
            probable_af=False,
            pr_class="normal",
            avb_grade=None,
            qrs_width_class="normal",
            bundle_branch_block=None,
            qtc_class="normal",
            wpw_pattern=False,
            p_morphology_class="normal",
            rae_leads=[],
            lae_suspected=False,
            lae_definite=False,
            ptf_v1_class=None,
            pathological_q_leads={"II": False, "V1": True},
            q_wave_territories=["anterior"],
            r_progression_class="normal",
            r_s_transition_lead="V3",
            st_elevation_leads={"V2": 0.12},
            st_depression_leads={"III": -0.08},
            st_territories_elevated=["anterior"],
            st_territories_depressed=["inferior"],
            stemi_suspected_codes=[],
            reciprocal_change_detected=True,
            reciprocal_pairs=[["+V2", "-III"]],
            lvh_voltage_criteria=["Sokolow_Lyon"],
            lvh_class="probable",
            low_voltage_class=None,
            rvh_suspected=False,
            limb_reversal_suspected=None,
            precordial_reversal_suspected=False,
            qrs_axis_class="normal",
            p_axis_normal=True,
            t_axis_class="normal",
            qrs_t_angle_deg=15.0,
            tall_t_leads=[],
        )
        beats = [
            SimpleNamespace(rr_prev_ms=None),
            SimpleNamespace(rr_prev_ms=690.0),
            SimpleNamespace(rr_prev_ms=700.0),
            SimpleNamespace(rr_prev_ms=710.0),
        ]
        return SimpleNamespace(
            fs=100,
            global_features=gf,
            interpretation=interp,
            beats=beats,
            groups={},
            quality={},
            representative_leads={},
            beat_features=[],
            metadata={"reliable_qt_leads": ["II", "V5"]},
        )

    def test_build_report_summary_extracts_key_fields(self) -> None:
        summary = demo.build_report_summary("rec001", self.make_hdr(), self.make_ecg(), self.make_result())

        self.assertEqual("rec001", summary["record_id"])
        self.assertEqual("61", summary["patient_rows"]["Age"])
        self.assertEqual("426783006", summary["patient_rows"]["Dx Codes"])
        self.assertEqual("10.0 s @ 100 Hz -> resampled to 100 Hz", summary["patient_rows"]["Recording"])
        self.assertEqual("88 bpm", summary["measurements"][0]["value"])
        self.assertEqual("154 ms", summary["measurements"][1]["value"])
        self.assertTrue(any("Rhythm: normal" in line for line in summary["interpretation_lines"]))
        self.assertTrue(any("Q waves: anterior" in line for line in summary["interpretation_lines"]))

    def test_dxl_probable_af_is_visibly_superseded_by_authoritative_flutter(self) -> None:
        display, resolution = demo._dxl_af_reference_display(
            True, {"atrial_flutter_pattern"}
        )

        self.assertIn("REFERENCE CONFLICT", display)
        self.assertIn("NOT FINAL", display)
        self.assertEqual("Superseded by authoritative atrial flutter", resolution)

    def test_build_report_summary_uses_safe_fallbacks(self) -> None:
        hdr = {"fs": 100, "n_samples": 1000, "dx": []}
        result = self.make_result()
        result.global_features.heart_rate_bpm = None
        result.interpretation = None

        summary = demo.build_report_summary("rec002", hdr, self.make_ecg(), result)

        self.assertEqual("N/A", summary["patient_rows"]["Age"])
        self.assertEqual("N/A", summary["patient_rows"]["Dx Codes"])
        self.assertEqual("N/A", summary["measurements"][0]["value"])
        self.assertIn("Interpretation unavailable", summary["interpretation_lines"])

    def test_generate_ecg_report_sheet_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "rec001_ecg_report.png"

            demo.generate_ecg_report_sheet(
                "rec001",
                self.make_hdr(),
                self.make_ecg(),
                self.make_result(),
                out_path,
            )

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)

    def test_generate_ecg_report_sheet_draws_waveform_directly(self) -> None:
        source = inspect.getsource(demo.generate_ecg_report_sheet)

        self.assertNotIn("mpimg.imread", source)
        self.assertNotIn("TemporaryDirectory", source)
        self.assertNotIn("generate_ecg_paper_plot(", source)

    def test_generate_ecg_report_sheet_uses_readable_large_canvas(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "rec001_ecg_report.png"

            demo.generate_ecg_report_sheet(
                "rec001",
                self.make_hdr(),
                self.make_ecg(),
                self.make_result(),
                out_path,
            )

            with Image.open(out_path) as image:
                self.assertGreaterEqual(image.size[0], 4200)
                self.assertGreaterEqual(image.size[1], 3000)

    def test_generate_ecg_report_sheet_uses_larger_bold_text(self) -> None:
        panel_signature = inspect.signature(demo._draw_report_panel)

        self.assertGreaterEqual(panel_signature.parameters["title_fontsize"].default, 13.0)
        self.assertGreaterEqual(panel_signature.parameters["body_fontsize"].default, 10.5)
        self.assertEqual("bold", panel_signature.parameters["body_fontweight"].default)

        source = inspect.getsource(demo.generate_ecg_report_sheet)
        self.assertIn("lead_fontsize=13.5", source)
        self.assertIn("bottom_labelsize=10.5", source)
        self.assertIn('fontweight="bold"', source)

    def test_generate_report_masks_interval_columns_without_reliable_qt_leads(self) -> None:
        result = self.make_result()
        result.global_features.qt_path = "low_qt_support"
        result.global_features.qt_source = "low_qt_support"
        result.global_features.qt_reliability = "low_confidence"
        result.global_features.qt_used_leads = []
        result.metadata["reliable_qt_leads"] = []
        result.quality = {
            lead: SimpleNamespace(
                reliable=True,
                baseline_wander_score=0.01,
                muscle_noise_score=0.02,
                powerline_score=0.0,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
            )
            for lead in demo.STANDARD_12_LEADS
        }
        result.representative_leads = {
            lead: SimpleNamespace(
                params={
                    "pr_ms": 154.0,
                    "qrs_ms": 92.0,
                    "qt_ms": 360.0,
                    "qt_confidence_mean": 0.05,
                    "r_amp_mv": 0.91,
                    "t_amp_mv": -0.22,
                    "st_on_mv": -0.03,
                }
            )
            for lead in demo.STANDARD_12_LEADS
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "report.txt"
            demo.generate_report("rec001", self.make_hdr(), self.make_ecg(), result, out_path)
            report = out_path.read_text()

        self.assertIn("Reliable-QT leads        : none", report)
        self.assertIn("I      N/A      N/A      N/A      N/A      0.910", report)
        self.assertNotRegex(report, r"(?m)^\s*(?:I|II|III|aVR|aVL|aVF|V[1-6])\s+(?!N/A)\d")

    def test_generate_ecg_annotated_plot_writes_png(self) -> None:
        result = self.make_result()
        result.beat_features = [
            SimpleNamespace(
                lead="II",
                beat_id=1,
                p=SimpleNamespace(onset=85, peak=100, offset=115),
                qrs=SimpleNamespace(onset=145, peak=150, offset=158),
                t=SimpleNamespace(onset=190, peak=225, offset=265),
                j_index=158,
            ),
            SimpleNamespace(
                lead="V2",
                beat_id=1,
                p=SimpleNamespace(onset=85, peak=100, offset=115),
                qrs=SimpleNamespace(onset=145, peak=150, offset=158),
                t=SimpleNamespace(onset=190, peak=225, offset=265),
                j_index=158,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "rec001_ecg_annotated.png"

            demo.generate_ecg_annotated_plot(
                "rec001",
                self.make_hdr(),
                self.make_ecg(),
                result,
                out_path,
            )

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)

    def test_generate_ecg_annotated_plot_includes_detection_and_abnormality_overlays(self) -> None:
        source = inspect.getsource(demo.generate_ecg_annotated_plot)

        self.assertIn("p.onset", source)
        self.assertIn("qrs.onset", source)
        self.assertIn("t.offset", source)
        self.assertIn("st_elevation_leads", source)
        self.assertIn("st_depression_leads", source)

    def test_main_wires_annotated_ecg_output(self) -> None:
        source = inspect.getsource(demo.main)

        self.assertIn("_ecg_annotated.png", source)
        self.assertIn("generate_ecg_annotated_plot", source)


if __name__ == "__main__":
    unittest.main()
