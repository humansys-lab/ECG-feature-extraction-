from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.ecgfeat.models import QRSCandidateWindow
from feature_extraction.ecgfeat.qrs import _filter_edge_candidates, detect_qrs_multilead_with_meta


class QRSDetectionTests(unittest.TestCase):
    def test_filter_edge_candidates_removes_terminal_detection_without_measurement_window(self) -> None:
        fs = 500
        candidates = [
            QRSCandidateWindow(580, 555, 605, 619, 1.0, 0.9),
            QRSCandidateWindow(4380, 4355, 4405, 4370, 1.0, 0.9),
            QRSCandidateWindow(4970, 4945, 4995, 4994, 1.0, 1.0),
        ]

        kept = _filter_edge_candidates(candidates, n_samples=5000, fs=fs)

        self.assertEqual([619, 4370], [candidate.refined_r_index for candidate in kept])

    def test_robust_fallback_recovers_repeated_beats_when_one_peak_dominates(self) -> None:
        fs = 500
        n_samples = 5000
        ecg = np.zeros((12, n_samples), dtype=float)
        beat_locs = np.arange(600, 4600, 500)
        qrs_half_width = int(0.080 * fs)
        qrs_x = np.arange(-qrs_half_width, qrs_half_width + 1)

        for r_loc in beat_locs:
            amplitude = 8.0 if r_loc == 3600 else 1.0
            qrs = amplitude * np.exp(-0.5 * (qrs_x / (0.025 * fs)) ** 2)
            for lead_idx in (0, 1, 7, 8, 9, 10, 11):
                polarity = 1.0 if lead_idx in (0, 1, 9, 10, 11) else -1.0
                ecg[lead_idx, r_loc - qrs_half_width:r_loc + qrs_half_width + 1] += polarity * qrs

        result = detect_qrs_multilead_with_meta(ecg, fs)

        self.assertGreaterEqual(len(result.r_locs), len(beat_locs) - 1)
        self.assertTrue(result.fallback_used)
        self.assertEqual("robust_threshold_sparse_primary", result.fallback_reason)


if __name__ == "__main__":
    unittest.main()
