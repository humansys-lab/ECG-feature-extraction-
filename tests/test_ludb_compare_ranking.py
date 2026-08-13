from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import ludb_compare_ranking as ranking


COMPARISON_TEXT = """========================================================================
LUDB REPORT COMPARISON  —  Record 61
========================================================================

Global Measurements
Metric           Algorithm       GroundTruth     Diff(A-GT)
---------------  --------------  --------------  ------------
HR               62.4            62.5            -0.1
PR               172.0           187.0           -15.0
QRS              99.0            135.0           -36.0
QT               416.0           468.0           -52.0
QTcB             424.1           477.7           -53.5
QTcF             421.4           474.4           -53.0
P Axis           50.8            61.8            -11.0
QRS Axis         -10.6           -4.2            -6.4
T Axis           -168.6          176.1           -344.7
QT Dispersion    0.0             0.0             0.0

Beat / Missing Summary
Algorithm beats     : 11
Ground-truth beats  : 8
Algorithm missing   : P=40  T=12  QRS=0  Unreliable=11
Ground-truth missing: P=12  T=0  QRS=0  Unreliable=0
"""


class LudbCompareRankingTests(unittest.TestCase):
    def test_parse_comparison_report_extracts_metrics_and_circular_axis_error(self) -> None:
        row = ranking.parse_comparison_report(COMPARISON_TEXT, record_id=61)

        self.assertEqual(61, row["record_id"])
        self.assertEqual(11, row["algorithm_beats"])
        self.assertEqual(8, row["ground_truth_beats"])
        self.assertEqual(3, row["beat_diff"])
        self.assertAlmostEqual(-53.5, row["diff_qtcb"])
        self.assertAlmostEqual(344.7, row["abs_t_axis_raw_diff"])
        self.assertAlmostEqual(15.3, row["abs_t_axis_circular_diff"])

    def test_build_ranked_rows_sorts_best_to_worst_and_penalizes_algorithm_na(self) -> None:
        rows = [
            {
                "record_id": 182,
                "beat_diff": 2,
                "algorithm_metric_na_count": 0,
                "ground_truth_metric_na_count": 0,
                "available_metric_count": 10,
                "abs_hr_diff": 0.2,
                "abs_pr_diff": 1.0,
                "abs_qrs_diff": 4.0,
                "abs_qt_diff": 1.0,
                "abs_qtcb_diff": 1.8,
                "abs_qtcf_diff": 1.5,
                "abs_p_axis_circular_diff": 2.9,
                "abs_qrs_axis_circular_diff": 2.0,
                "abs_t_axis_circular_diff": 3.4,
                "abs_qt_dispersion_diff": 0.0,
            },
            ranking.parse_comparison_report(COMPARISON_TEXT, record_id=61),
            {
                "record_id": 13,
                "beat_diff": 2,
                "algorithm_metric_na_count": 3,
                "ground_truth_metric_na_count": 0,
                "available_metric_count": 7,
                "abs_hr_diff": 0.2,
                "abs_pr_diff": 67.0,
                "abs_qrs_diff": 38.0,
                "abs_qt_diff": None,
                "abs_qtcb_diff": None,
                "abs_qtcf_diff": None,
                "abs_p_axis_circular_diff": 6.9,
                "abs_qrs_axis_circular_diff": 11.1,
                "abs_t_axis_circular_diff": 93.8,
                "abs_qt_dispersion_diff": None,
            },
        ]

        ranked = ranking.build_ranked_rows(rows)

        self.assertEqual([182, 61, 13], [row["record_id"] for row in ranked])
        self.assertEqual([1, 2, 3], [row["rank"] for row in ranked])
        self.assertLess(ranked[0]["rank_score"], ranked[1]["rank_score"])
        self.assertLess(ranked[1]["rank_score"], ranked[2]["rank_score"])

    def test_summarize_directory_writes_ranked_csv_and_text_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_dir = root / "61"
            record_dir.mkdir()
            (record_dir / "61_comparison.txt").write_text(COMPARISON_TEXT)

            csv_path, table_path = ranking.summarize_directory(root)

            self.assertTrue(csv_path.exists())
            self.assertTrue(table_path.exists())
            table_text = table_path.read_text()
            self.assertIn("record_id", table_text)
            self.assertIn("61", table_text)


if __name__ == "__main__":
    unittest.main()
