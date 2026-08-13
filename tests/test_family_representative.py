from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.ecgfeat.family_representative import (
    assign_morphology_families,
    select_family_medoid,
)
from feature_extraction.ecgfeat.representative import build_representative_beats_with_meta


def _beat(scale: float = 1.0, invert: bool = False) -> np.ndarray:
    template = np.zeros((12, 80), dtype=float)
    wave = np.hanning(24) * scale
    if invert:
        wave = -wave
    template[:, 28:52] = wave
    return template


class FamilyRepresentativeTests(unittest.TestCase):
    def test_assign_morphology_families_splits_paced_and_qrs_morphology(self) -> None:
        beats = [
            _beat(),
            _beat(scale=1.05),
            _beat(invert=True),
            _beat(scale=0.95),
        ]

        families = assign_morphology_families(
            beats,
            rr_prev_ms=[800.0, 810.0, 420.0, 790.0],
            paced_beat_ids={3},
        )

        normal_family = [item.family_id for item in families if item.beat_index in {0, 1}]
        self.assertEqual(1, len(set(normal_family)))
        self.assertNotEqual(families[2].family_id, normal_family[0])
        self.assertNotEqual(families[3].family_id, normal_family[0])
        self.assertTrue(families[3].paced)

    def test_assign_morphology_families_accepts_numpy_inputs_and_nonfinite_rr(self) -> None:
        beats = np.asarray([_beat(), _beat(scale=1.05)])

        families = assign_morphology_families(
            beats,
            rr_prev_ms=np.asarray([np.nan, np.inf]),
            paced_beat_ids=np.asarray([1]),
        )

        self.assertEqual("unknown", families[0].rr_context)
        self.assertEqual("unknown", families[1].rr_context)
        self.assertFalse(families[0].paced)
        self.assertTrue(families[1].paced)

    def test_assign_morphology_families_keeps_near_boundary_rr_regular(self) -> None:
        beats = [_beat() for _ in range(5)]

        families = assign_morphology_families(
            beats,
            rr_prev_ms=[790.0, 810.0, 1000.0, 1000.0, 1000.0],
        )

        self.assertEqual("regular_rr", families[0].rr_context)
        self.assertEqual("regular_rr", families[1].rr_context)
        self.assertEqual(families[0].family_id, families[1].family_id)

    def test_select_family_medoid_returns_existing_central_beat(self) -> None:
        beats = [
            _beat(scale=1.0),
            _beat(scale=1.02),
            _beat(scale=0.4),
        ]

        medoid = select_family_medoid(beats, members=[0, 1, 2])

        self.assertIn(medoid, {0, 1})

    def test_representative_with_meta_uses_family_medoid_for_boundaries(self) -> None:
        fs = 100
        r_locs = np.asarray([100, 300, 500], dtype=int)
        ecg = np.zeros((12, 700), dtype=float)
        shape = np.array([0.0, 1.0, 0.0])
        ecg[:, 100:103] = shape
        ecg[:, 300:303] = shape
        ecg[:, 500:503] = -shape

        reps, metas = build_representative_beats_with_meta(
            ecg,
            r_locs,
            {1: [0, 1, 2]},
            fs,
            left_ms=0,
            right_ms=120,
            outlier_corr_threshold=-1.0,
            max_lag_ms=0,
        )

        self.assertEqual(1, metas[1].used_count)
        np.testing.assert_allclose(reps[1][:, :3], ecg[:, 100:103])

    def test_representative_medoid_keeps_dominant_family_when_first_beat_is_outlier(self) -> None:
        fs = 100
        r_locs = np.asarray([100, 300, 500], dtype=int)
        ecg = np.zeros((12, 700), dtype=float)
        ecg[:, 100:103] = np.array([0.0, 1.0, 0.0])
        ecg[:, 306:309] = np.array([0.0, 1.0, 0.0])
        ecg[:, 506:509] = np.array([0.0, 1.0, 0.0])

        reps, metas = build_representative_beats_with_meta(
            ecg,
            r_locs,
            {1: [0, 1, 2]},
            fs,
            left_ms=0,
            right_ms=120,
            max_lag_ms=0,
        )

        self.assertEqual(1, metas[1].used_count)
        np.testing.assert_allclose(reps[1][:, 6:9], ecg[:, 306:309])

    def test_representative_with_meta_passes_paced_ids_to_family_assignment(self) -> None:
        fs = 100
        r_locs = np.asarray([100, 300, 500], dtype=int)
        ecg = np.zeros((12, 700), dtype=float)
        for r in r_locs:
            ecg[:, r:r + 3] = np.array([0.0, 1.0, 0.0])

        reps, _metas = build_representative_beats_with_meta(
            ecg,
            r_locs,
            {1: [0, 1, 2]},
            fs,
            left_ms=0,
            right_ms=120,
            max_lag_ms=0,
            paced_beat_ids={0},
        )

        np.testing.assert_allclose(reps[1][:, :3], ecg[:, 300:303])


if __name__ == "__main__":
    unittest.main()
