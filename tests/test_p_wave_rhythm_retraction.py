"""Regression tests for the P-wave contract surviving a retracted rhythm verdict.

`ECGFeatureExtractor.extract` finalizes the P-wave contract against the flutter
verdict long before that verdict can be retracted by the `regular_narrow_pr_core`
suppression further down.  When the retraction fires, every other consumer of
`af_afl_summary` reads the corrected copy, but the P contract used to keep the
dead verdict -- so a sinus record could report a numeric PR interval and
"0 of N beats have a usable P wave" simultaneously.
"""

import unittest
from pathlib import Path

from feature_extraction.ecgfeat.models import PWaveBeatAssessment
from feature_extraction.ecgfeat.p_wave_engine import (
    AF_LIKE,
    ORGANIZED_ATRIAL_ACTIVITY,
    OVERLAP_UNCERTAIN,
    P_PRESENT,
    finalize_p_wave_states,
)


def _assessment(beat_id: int = 0, accepted: bool = True) -> PWaveBeatAssessment:
    return PWaveBeatAssessment(
        beat_id=beat_id,
        strict_onset=10,
        strict_offset=40,
        robust_onset=10,
        robust_offset=40,
        accepted=accepted,
        p_state=P_PRESENT if accepted else OVERLAP_UNCERTAIN,
    )


def _snapshot(assessments):
    return [(a.accepted, list(a.reject_reasons), a.p_state) for a in assessments]


def _restore(assessments, snapshot):
    for assessment, (accepted, reasons, state) in zip(assessments, snapshot):
        assessment.accepted = accepted
        assessment.reject_reasons = list(reasons)
        assessment.p_state = state


class PWaveRhythmRetractionTests(unittest.TestCase):
    def test_finalize_is_destructive_and_cannot_be_repaired_by_recalling(self) -> None:
        """The reason the fix needs a snapshot rather than a second finalize call.

        `finalize` forces ``accepted=False`` with no pristine copy retained, so a
        later call carrying the corrected verdict cannot restore acceptance: the
        beat falls through to the ``else`` branch as OVERLAP_UNCERTAIN and keeps
        the stale reject reason.
        """
        assessments = [_assessment()]

        finalize_p_wave_states(assessments, {"probable_flutter": True})
        self.assertEqual(assessments[0].p_state, ORGANIZED_ATRIAL_ACTIVITY)
        self.assertFalse(assessments[0].accepted)

        finalize_p_wave_states(assessments, {"probable_flutter": False})

        self.assertFalse(assessments[0].accepted)
        self.assertEqual(assessments[0].p_state, OVERLAP_UNCERTAIN)
        self.assertIn("RHYTHM_ORGANIZED_ATRIAL_ACTIVITY", assessments[0].reject_reasons)

    def test_snapshot_restore_recovers_p_waves_after_flutter_retraction(self) -> None:
        assessments = [_assessment(beat_id=i) for i in range(11)]
        pristine = _snapshot(assessments)

        finalize_p_wave_states(assessments, {"probable_flutter": True})
        self.assertEqual(sum(1 for a in assessments if a.accepted), 0)

        # The retraction: probable_flutter goes False, flutter_suppressed_by set.
        _restore(assessments, pristine)
        finalize_p_wave_states(
            assessments,
            {
                "probable_flutter": False,
                "atrial_rhythm_classification": "none",
                "flutter_suppressed_by": "regular_narrow_pr_core",
            },
        )

        self.assertEqual(sum(1 for a in assessments if a.accepted), 11)
        for assessment in assessments:
            self.assertEqual(assessment.p_state, P_PRESENT)
            self.assertEqual(assessment.reject_reasons, [])

    def test_restore_preserves_boundary_quality_rejections(self) -> None:
        """A beat the delineator itself rejected must not be resurrected."""
        assessments = [_assessment(beat_id=0, accepted=True), _assessment(beat_id=1, accepted=False)]
        assessments[1].reject_reasons = ["INSUFFICIENT_INFORMATIVE_LEADS"]
        pristine = _snapshot(assessments)

        finalize_p_wave_states(assessments, {"probable_flutter": True})
        _restore(assessments, pristine)
        finalize_p_wave_states(assessments, {"probable_flutter": False})

        self.assertTrue(assessments[0].accepted)
        self.assertEqual(assessments[0].p_state, P_PRESENT)
        self.assertFalse(assessments[1].accepted)
        self.assertEqual(assessments[1].p_state, OVERLAP_UNCERTAIN)
        self.assertEqual(assessments[1].reject_reasons, ["INSUFFICIENT_INFORMATIVE_LEADS"])

    def test_restore_is_a_no_op_when_no_verdict_was_retracted(self) -> None:
        """The extract() restore runs unconditionally; it must be idempotent."""
        for verdict, expected_state in (
            ({"probable_af": True}, AF_LIKE),
            ({"probable_flutter": True}, ORGANIZED_ATRIAL_ACTIVITY),
            ({}, P_PRESENT),
        ):
            with self.subTest(verdict=verdict):
                assessments = [_assessment()]
                pristine = _snapshot(assessments)

                finalize_p_wave_states(assessments, verdict)
                before = (assessments[0].accepted, list(assessments[0].reject_reasons), assessments[0].p_state)

                _restore(assessments, pristine)
                finalize_p_wave_states(assessments, verdict)

                self.assertEqual(assessments[0].p_state, expected_state)
                self.assertEqual(
                    (assessments[0].accepted, list(assessments[0].reject_reasons), assessments[0].p_state),
                    before,
                )

    def test_genuine_af_still_invalidates_every_beat(self) -> None:
        """The retraction path must not loosen the real AF/flutter suppression."""
        assessments = [_assessment(beat_id=i) for i in range(8)]
        pristine = _snapshot(assessments)

        finalize_p_wave_states(assessments, {"probable_af": True})
        _restore(assessments, pristine)
        # No retraction happened, so the summary still carries the AF verdict.
        finalize_p_wave_states(assessments, {"probable_af": True})

        self.assertEqual(sum(1 for a in assessments if a.accepted), 0)
        for assessment in assessments:
            self.assertEqual(assessment.p_state, AF_LIKE)
            self.assertIn("RHYTHM_AF_LIKE", assessment.reject_reasons)


_RECORD = Path(__file__).resolve().parents[1] / "data" / "ptb-xl" / "09000" / "09009_hr"


@unittest.skipUnless(_RECORD.with_suffix(".hea").exists(), f"PTB-XL record {_RECORD.name} not available")
class ExtractRetractionIntegrationTests(unittest.TestCase):
    """Pins the invariant in `extract()` itself, not just the helper semantics.

    09009_hr is a sinus record (PTB-XL label IVCD, no atrial arrhythmia) on which
    the flutter suppression fires.  Before the fix it reported PR=193ms alongside
    0 of 11 usable P waves.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np
        import wfdb

        from feature_extraction.ecgfeat.api import ECGFeatureExtractor

        record = wfdb.rdrecord(str(_RECORD))
        signal = np.asarray(record.p_signal, dtype=float).T
        cls.features = ECGFeatureExtractor().extract(
            signal, float(record.fs), lead_names=[name.upper() for name in record.sig_name]
        )
        cls.af_afl = (cls.features.metadata.get("rhythm_analysis") or {}).get("af_afl_summary") or {}
        cls.contract = cls.features.metadata.get("p_wave_contract") or {}

    def test_this_record_still_exercises_the_retraction_path(self) -> None:
        # Guards the test itself: if the suppression stops firing here the
        # assertions below would pass vacuously.
        self.assertEqual(self.af_afl.get("flutter_suppressed_by"), "regular_narrow_pr_core")
        self.assertFalse(self.af_afl.get("probable_flutter"))

    def test_retracted_flutter_verdict_does_not_survive_in_the_p_contract(self) -> None:
        self.assertNotIn(
            ORGANIZED_ATRIAL_ACTIVITY, self.contract.get("state_counts") or {}
        )
        self.assertNotIn(
            "RHYTHM_ORGANIZED_ATRIAL_ACTIVITY", self.contract.get("reject_reason_counts") or {}
        )
        for assessment in self.features.p_wave_assessments:
            self.assertNotIn("RHYTHM_ORGANIZED_ATRIAL_ACTIVITY", assessment.reject_reasons)

    def test_a_record_reporting_a_pr_interval_reports_usable_p_waves(self) -> None:
        self.assertIsNotNone(self.features.global_features.pr_ms)
        self.assertGreater(self.contract.get("n_accepted", 0), 0)


if __name__ == "__main__":
    unittest.main()
