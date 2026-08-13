import csv
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import wfdb

import evaluate_ludb
from compare_annotations import LEAD_TO_ANN_EXT, LUDB_DIR
from evaluate_ludb import parse_ludb_annotations
from feature_extraction.ecgfeat.api import ECGFeatureExtractor


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
COMPARE_TIMEOUT_SECONDS = 180


def _run_compare(records: list[str], out_dir: Path) -> dict[str, dict[str, str]]:
    try:
        subprocess.run(
            [
                str(PYTHON),
                "compare_annotations.py",
                "--batch-reports",
                "--records",
                *records,
                "--out-dir",
                str(out_dir),
                "--no-visuals",
            ],
            cwd=ROOT,
            check=True,
            timeout=COMPARE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "compare_annotations.py timed out after "
            f"{COMPARE_TIMEOUT_SECONDS} seconds for records: {', '.join(records)}"
        ) from exc

    with (out_dir / "summary.csv").open(newline="") as summary_file:
        return {
            row["record_id"]: row
            for row in csv.DictReader(summary_file)
        }


def _required_float(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        raise AssertionError(f"Missing required field {field!r}")
    return float(value)


def _real_wfdb_module():
    module = sys.modules.get("wfdb")
    if not callable(getattr(module, "rdrecord", None)):
        sys.modules.pop("wfdb", None)
        module = importlib.import_module("wfdb")
    globals()["wfdb"] = module
    evaluate_ludb.wfdb = module
    return module


class DxlRegressionTest(unittest.TestCase):
    def test_high_risk_ludb_records_stay_within_interval_gates(self) -> None:
        records = ["13", "74", "75", "95"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = _run_compare(records, Path(tmp_dir))

        self.assertLessEqual(abs(_required_float(rows["13"], "diff_qrs")), 25.0)
        self.assertLessEqual(abs(_required_float(rows["13"], "diff_qt")), 15.0)
        self.assertNotEqual("on", rows["13"]["pacing_state"])
        self.assertLessEqual(abs(_required_float(rows["74"], "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(rows["74"], "diff_hr")), 3.0)
        self.assertLessEqual(abs(_required_float(rows["75"], "diff_qt")), 10.0)
        self.assertLessEqual(abs(_required_float(rows["95"], "diff_qt")), 10.0)

    def test_ludb_127_brady_wide_qrs_measurements_do_not_use_early_outlier_onsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            row = _run_compare(["127"], Path(tmp_dir))["127"]

        self.assertLessEqual(abs(_required_float(row, "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qt")), 20.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qtcb")), 20.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qtcf")), 20.0)
        self.assertLessEqual(abs(_required_float(row, "diff_t_axis")), 30.0)

    def test_ludb_af_qt_aggregation_rejects_late_t_end_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = _run_compare(["109", "110"], Path(tmp_dir))

        self.assertLessEqual(abs(_required_float(rows["109"], "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(rows["109"], "diff_qt")), 60.0)
        self.assertLessEqual(abs(_required_float(rows["109"], "diff_qtcb")), 80.0)
        self.assertLessEqual(abs(_required_float(rows["109"], "diff_qtcf")), 80.0)
        self.assertLessEqual(abs(_required_float(rows["110"], "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(rows["110"], "diff_qt")), 20.0)
        self.assertLessEqual(abs(_required_float(rows["110"], "diff_qtcb")), 30.0)
        self.assertLessEqual(abs(_required_float(rows["110"], "diff_qtcf")), 30.0)

    def test_ludb_108_systematic_terminal_qrs_tail_reaches_st_plateau(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            row = _run_compare(["108"], Path(tmp_dir))["108"]

        self.assertLessEqual(abs(_required_float(row, "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qt")), 25.0)

    def test_ludb_45_paced_wide_qrs_uses_multilead_consensus_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            row = _run_compare(["45"], out_dir)["45"]
            report = (out_dir / "45" / "45_algorithm_report.txt").read_text()

        self.assertLessEqual(abs(_required_float(row, "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qt")), 30.0)
        self.assertIn("Probable AF         : No", report)
        self.assertIn("WPW pattern         : No", report)
        self.assertIn("RVH suspected       : No", report)

    def test_ludb_104_p_synchronous_pacing_suppresses_native_pr_and_secondary_morphology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            row = _run_compare(["104"], out_dir)["104"]
            report = (out_dir / "104" / "104_algorithm_report.txt").read_text()

        self.assertLessEqual(abs(_required_float(row, "diff_qrs")), 25.0)
        self.assertEqual("", row["algorithm_pr"])
        self.assertEqual("", row["algorithm_p_axis"])
        self.assertIn("PR Interval               N/A", report)
        self.assertIn("P Axis                    N/A", report)
        self.assertIn("P axis              : N/A", report)
        self.assertNotIn("AVB grade 1", report)
        self.assertIn("Pathological Q leads: —", report)
        self.assertIn("ST elevation        : None significant", report)
        self.assertIn("ST depression       : None significant", report)
        self.assertIn("RVH suspected       : No", report)

    def test_ludb_104_paced_branch_does_not_reuse_t_wave_as_retrograde_p(self) -> None:
        record = _real_wfdb_module().rdrecord(str(LUDB_DIR / "104"))
        features = ECGFeatureExtractor(fs_internal=500, mains_freq=50).extract(
            record.p_signal.T,
            float(record.fs),
        )

        self.assertEqual("on", features.metadata["pacing_state"])
        self.assertEqual(
            list(range(len(features.beats))),
            features.metadata["paced_beat_ids"],
        )

        def _overlap_samples(first: object, second: object) -> int:
            if (
                first.onset is None
                or first.offset is None
                or second.onset is None
                or second.offset is None
            ):
                return 0
            return max(0, min(first.offset, second.offset) - max(first.onset, second.onset))

        for beat_feature in features.beat_features:
            with self.subTest(lead=beat_feature.lead, beat_id=beat_feature.beat_id):
                self.assertFalse(
                    beat_feature.pr_ms is not None and beat_feature.pr_ms < 0.0,
                    "paced retrograde P must not produce a native PR interval",
                )
                self.assertEqual(
                    0,
                    _overlap_samples(beat_feature.p, beat_feature.qrs),
                    "paced P bounds must not overlap QRS bounds",
                )
                self.assertEqual(
                    0,
                    _overlap_samples(beat_feature.p, beat_feature.t),
                    "paced P bounds must not overlap T-wave bounds",
                )

    def test_ludb_111_near_wide_paced_av_dissociation_uses_pacing_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            row = _run_compare(["111"], out_dir)["111"]
            report = (out_dir / "111" / "111_algorithm_report.txt").read_text()

        self.assertLessEqual(abs(_required_float(row, "diff_qrs")), 25.0)
        self.assertEqual("", row["algorithm_pr"])
        self.assertIn("PR Interval               N/A", report)
        self.assertIn("WPW pattern         : No", report)
        self.assertIn("ST elevation        : None significant", report)
        self.assertIn("ST depression       : None significant", report)
        self.assertNotIn("*** STEMI codes", report)
        self.assertIn("RVH suspected       : No", report)

    def test_ludb_111_negative_dominant_qs_r_fiducial_is_not_late_s_trough(self) -> None:
        fs = 500
        tolerance_samples = int(0.075 * fs)
        record_id = "111"
        record = _real_wfdb_module().rdrecord(str(LUDB_DIR / record_id))
        features = ECGFeatureExtractor(fs_internal=fs, mains_freq=50).extract(
            record.p_signal.T,
            float(record.fs),
        )
        gt_lead_ii = parse_ludb_annotations(
            str(LUDB_DIR / record_id),
            LEAD_TO_ANN_EXT["II"],
        )

        global_errors_ms = []
        for gt in gt_lead_ii:
            nearest = min(features.beats, key=lambda beat: abs(beat.r_index - gt["r_sample"]))
            if abs(nearest.r_index - gt["r_sample"]) <= tolerance_samples:
                global_errors_ms.append((nearest.r_index - gt["r_sample"]) * 1000.0 / fs)

        lead_ii_features = [
            beat_feature
            for beat_feature in features.beat_features
            if beat_feature.lead == "II" and beat_feature.qrs.peak is not None
        ]
        per_lead_errors_ms = []
        used_detected: set[int] = set()
        for gt in gt_lead_ii:
            candidates = [
                (abs(int(det.qrs.peak) - gt["r_sample"]), idx, det)
                for idx, det in enumerate(lead_ii_features)
                if idx not in used_detected
            ]
            if not candidates:
                continue
            distance, idx, detected = min(candidates)
            if distance <= tolerance_samples:
                used_detected.add(idx)
                per_lead_errors_ms.append((int(detected.qrs.peak) - gt["r_sample"]) * 1000.0 / fs)

        self.assertGreaterEqual(len(global_errors_ms), 5)
        self.assertGreaterEqual(len(per_lead_errors_ms), 4)
        self.assertLessEqual(float(np.median(np.abs(global_errors_ms))), 20.0)
        self.assertLessEqual(float(np.median(np.abs(per_lead_errors_ms))), 20.0)

    def test_ludb_164_left_axis_deviation_not_flattened_by_initial_core_override(self) -> None:
        # Rec 164 is a single-morphology sinus record with genuine left-axis
        # deviation: the net QRS vector is negative in II/III/aVF because of
        # deep terminal S waves (GT QRS axis ~ -38 deg).  The representative
        # per-lead amplitudes are correct, so the net-amplitude axis lands near
        # GT.  The bug: the initial-core override (r+q only, dropping the S
        # wave) fires on this narrow QRS and pulls the reported axis back toward
        # normal (~+22 deg), a ~60 deg error.  Frontal QRS axis is defined by
        # the net deflection including S, so the override must not flatten it.
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            row = _run_compare(["164"], out_dir)["164"]

        self.assertLessEqual(abs(_required_float(row, "diff_qrs_axis")), 15.0)

    def test_ludb_121_low_voltage_lvh_uses_stable_limb_t_axis_and_suppresses_false_wpw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            row = _run_compare(["121"], out_dir)["121"]
            report = (out_dir / "121" / "121_algorithm_report.txt").read_text()

        self.assertNotEqual("on", row["pacing_state"])
        self.assertLessEqual(abs(_required_float(row, "diff_pr")), 25.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qt")), 15.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qrs_axis")), 15.0)
        self.assertLessEqual(abs(_required_float(row, "diff_t_axis")), 20.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qt_dispersion")), 45.0)
        self.assertIn("Precordial reversal : No", report)
        self.assertIn("WPW pattern         : No", report)
        self.assertIn("ST elevation        : None significant", report)
        self.assertIn("Reciprocal change   : No", report)

    def test_ludb_88_af_uncertain_t_axis_and_qtc_without_reliable_qt_leads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            row = _run_compare(["88"], out_dir)["88"]
            report = (out_dir / "88" / "88_algorithm_report.txt").read_text()

        self.assertLessEqual(abs(_required_float(row, "diff_qrs")), 15.0)
        self.assertLessEqual(abs(_required_float(row, "diff_qt")), 30.0)
        self.assertEqual("", row["algorithm_t_axis"])
        self.assertIn("T Axis                    N/A", report)
        self.assertRegex(report, r"QTc \(Bazett\)\s+: \d+ ms\s+→\s+indeterminate")
        self.assertNotIn("CRITICAL: Very prolonged QTcB", report)
        self.assertIn("WARNING: No reliable-QT leads found", report)
        self.assertNotRegex(
            report,
            r"(?m)^\s*(?:I|II|III|aVR|aVL|aVF|V[1-6])\s+(?!N/A)\d",
        )

    def test_ludb_133_normal_pr_does_not_display_second_degree_avb_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            _run_compare(["133"], out_dir)
            report = (out_dir / "133" / "133_algorithm_report.txt").read_text()

        self.assertRegex(report, r"PR interval\s+: \d+ ms\s+→\s+normal")
        self.assertNotIn("[AVB grade 2]", report)
        self.assertIn("Global QT source", report)
        self.assertIn("PER-LEAD MEASUREMENTS  (raw representative", report)

    def test_repolarization_refactor_targeted_records_have_bounded_major_errors_or_low_confidence(self) -> None:
        records = ["57", "67", "73", "76", "104", "111", "124", "125", "130", "133"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = _run_compare(records, Path(tmp_dir))

        for record_id, row in rows.items():
            with self.subTest(record_id=record_id):
                qt_error = row.get("abs_qt_diff", "")
                t_axis_error = row.get("abs_t_axis_circular_diff", "")
                reliable_count = int(row.get("algorithm_reliable_qt_lead_count") or 0)
                if qt_error:
                    self.assertLessEqual(float(qt_error), 70.0)
                if t_axis_error and reliable_count >= 3:
                    self.assertLessEqual(float(t_axis_error), 70.0)


if __name__ == "__main__":
    unittest.main()
