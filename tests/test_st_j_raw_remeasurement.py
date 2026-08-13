import unittest

import numpy as np

from feature_extraction.ecgfeat.delineate import (
    STJRepairTarget,
    _apply_st_j_raw_remeasurement,
    _apply_st_j_remeasurement,
    _st_j_remeasurement_candidate,
    _st_j_remeasurement_target,
    _st_j_sample_values,
)
from feature_extraction.ecgfeat.models import LeadBeatFeatures, LeadQuality, WaveBounds


def _quality(lead: str, *, qrs: bool = True) -> LeadQuality:
    return LeadQuality(
        lead=lead,
        baseline_wander_score=0.0,
        muscle_noise_score=0.0,
        powerline_score=0.0,
        clipping_score=0.0,
        flatline_score=1.0,
        missing=False,
        reliable=qrs,
        reliable_for_p=qrs,
        reliable_for_qrs=qrs,
        reliable_for_t=qrs,
        reliable_for_qt=qrs,
    )


def _beat(
    lead: str,
    *,
    qrs_on: int = 100,
    qrs_peak: int = 130,
    qrs_off: int = 160,
    st_on: float | None = 0.02,
    st_mid: float | None = 0.02,
    st_80: float | None = 0.02,
    qrs_conf: float = 0.90,
    qrs_off_conf: float = 0.90,
    flags: list[str] | None = None,
) -> LeadBeatFeatures:
    feature = LeadBeatFeatures(
        lead=lead,
        beat_id=0,
        p=WaveBounds(onset=None, peak=None, offset=None),
        qrs=WaveBounds(onset=qrs_on, peak=qrs_peak, offset=qrs_off),
        t=WaveBounds(onset=230, peak=300, offset=360),
        qt_ms=520.0,
        pr_ms=None,
        qrs_ms=(qrs_off - qrs_on) * 2.0,
        p_amp_mv=None,
        qrs_area=1.0,
        q_amp_mv=None,
        r_amp_mv=1.0,
        s_amp_mv=None,
        st_on_mv=st_on,
        st_mid_mv=st_mid,
        st_80ms_mv=st_80,
        t_amp_mv=0.3,
        j_index=qrs_off,
        qrs_confidence=qrs_conf,
        qrs_off_confidence=qrs_off_conf,
        st_slope_mv_per_ms=0.0,
        st_morphology="horizontal",
    )
    feature.flags = list(flags or [])
    return feature


def _synthetic_ecg(leads: int = 5, length: int = 500) -> np.ndarray:
    ecg = np.zeros((leads, length), dtype=float)
    ecg[:, 100:161] = np.hanning(61) * 1.0
    ecg[:, 160:260] = 0.02
    return ecg


class STJModelFieldTests(unittest.TestCase):
    def test_st_j_remeasurement_fields_default_to_none(self) -> None:
        feature = _beat("II")

        self.assertIsNone(feature.st_j_original_index)
        self.assertIsNone(feature.st_j_remeasured_index)
        self.assertIsNone(feature.st_j_remeasure_delta_ms)
        self.assertIsNone(feature.st_j_remeasure_reason)
        self.assertIsNone(feature.st_j_remeasure_confidence)
        self.assertIsNone(feature.st_j_remeasure_consensus_index)
        self.assertIsNone(feature.st_j_remeasure_support)
        self.assertIsNone(feature.st_j_remeasure_used_leads)
        self.assertIsNone(feature.st_j_remeasure_excluded_leads)


class STJTargetTests(unittest.TestCase):
    def test_st_j_remeasurement_target_uses_reliable_qrs_core(self) -> None:
        beats = [
            _beat("I", qrs_off=160),
            _beat("II", qrs_off=161),
            _beat("III", qrs_off=159),
            _beat("aVR", qrs_off=160),
            _beat("V5", qrs_off=205, qrs_conf=0.05, qrs_off_conf=0.05),
        ]
        quality = {beat.lead: _quality(beat.lead) for beat in beats}

        target = _st_j_remeasurement_target(beats, fs=500, quality=quality)

        self.assertIsNotNone(target)
        self.assertEqual(160, target.j_index)
        self.assertEqual(4, target.support)
        self.assertEqual("I,II,III,aVR", target.used_leads)
        self.assertEqual("V5", target.excluded_leads)

    def test_st_j_remeasurement_target_returns_none_with_low_support(self) -> None:
        beats = [
            _beat("I", qrs_off=160),
            _beat("II", qrs_off=161),
            _beat("V5", qrs_off=205, qrs_conf=0.05, qrs_off_conf=0.05),
        ]
        quality = {beat.lead: _quality(beat.lead) for beat in beats}

        self.assertIsNone(_st_j_remeasurement_target(beats, fs=500, quality=quality))


class STJCandidateTests(unittest.TestCase):
    def test_st_j_sample_values_recomputes_slope_and_morphology(self) -> None:
        fs = 500
        sig = np.zeros(400, dtype=float)
        sig[160] = 0.00
        sig[180] = 0.04
        sig[200] = 0.08

        st_on, st_mid, st_80, slope, morphology = _st_j_sample_values(
            sig,
            anchor=160,
            baseline=0.0,
            fs=fs,
        )

        self.assertAlmostEqual(0.00, st_on)
        self.assertAlmostEqual(0.04, st_mid)
        self.assertAlmostEqual(0.08, st_80)
        self.assertAlmostEqual(0.001, slope)
        self.assertEqual("upsloping", morphology)

    def test_st_j_remeasurement_candidate_selects_consensus_anchor_for_tail_jump(self) -> None:
        feature = _beat(
            "V5",
            qrs_off=205,
            st_on=0.55,
            st_mid=0.04,
            st_80=0.03,
            flags=["st_j_unreliable"],
        )
        target = STJRepairTarget(j_index=160, support=5, used_leads="I,II,III,aVR,V5", excluded_leads="")
        sig = np.zeros(500, dtype=float)
        sig[160:260] = 0.03

        anchor, reason = _st_j_remeasurement_candidate(
            feature,
            sig=sig,
            beat_start=0,
            fs=500,
            target=target,
            next_qrs_guard=None,
        )

        self.assertEqual(160, anchor)
        self.assertEqual("tail_guard", reason)

    def test_st_j_remeasurement_candidate_preserves_stable_st_elevation(self) -> None:
        feature = _beat(
            "V5",
            qrs_off=160,
            st_on=0.22,
            st_mid=0.23,
            st_80=0.21,
            flags=[],
        )
        target = STJRepairTarget(j_index=158, support=5, used_leads="I,II,III,aVR,V5", excluded_leads="")
        sig = np.zeros(500, dtype=float)
        sig[150:260] = 0.22

        anchor, reason = _st_j_remeasurement_candidate(
            feature,
            sig=sig,
            beat_start=0,
            fs=500,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNone(anchor)
        self.assertIsNone(reason)

    def test_apply_st_j_remeasurement_records_provenance_and_updates_values(self) -> None:
        fs = 500
        feature = _beat(
            "V5",
            qrs_off=205,
            st_on=0.55,
            st_mid=0.04,
            st_80=0.03,
            flags=["st_j_unreliable"],
        )
        sig = np.zeros(500, dtype=float)
        sig[160] = 0.01
        sig[180] = 0.02
        sig[200] = 0.08
        target = STJRepairTarget(j_index=160, support=5, used_leads="I,II,III,aVR,V5", excluded_leads="")

        _apply_st_j_remeasurement(
            feature,
            anchor_global=160,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            reason="tail_guard",
            target=target,
            confidence=0.80,
        )

        self.assertEqual(205, feature.j_index)
        self.assertEqual(205, feature.st_j_original_index)
        self.assertEqual(160, feature.st_j_remeasured_index)
        self.assertAlmostEqual(-90.0, feature.st_j_remeasure_delta_ms)
        self.assertEqual("tail_guard", feature.st_j_remeasure_reason)
        self.assertEqual(160, feature.st_j_remeasure_consensus_index)
        self.assertEqual(5, feature.st_j_remeasure_support)
        self.assertEqual("I,II,III,aVR,V5", feature.st_j_remeasure_used_leads)
        self.assertAlmostEqual(0.01, feature.st_on_mv)
        self.assertAlmostEqual(0.02, feature.st_mid_mv)
        self.assertAlmostEqual(0.08, feature.st_80ms_mv)
        self.assertGreater(feature.st_slope_mv_per_ms, 0.0)
        self.assertEqual("upsloping", feature.st_morphology)
        self.assertIn("st_j_remeasured_by_consensus", feature.flags)
        self.assertIn("st_j_tail_guard_remeasured", feature.flags)
        self.assertNotIn("st_j_unreliable", feature.flags)


class STJPassTests(unittest.TestCase):
    def test_raw_remeasurement_pass_repairs_only_tail_contaminated_lead(self) -> None:
        fs = 500
        ecg = _synthetic_ecg(leads=5)
        ecg[4, 205] = 0.55
        ecg[4, 180] = 0.02
        ecg[4, 200] = 0.03
        leads = ["I", "II", "III", "aVR", "V5"]
        beats = [
            _beat("I", qrs_off=160),
            _beat("II", qrs_off=161),
            _beat("III", qrs_off=159),
            _beat("aVR", qrs_off=160),
            _beat("V5", qrs_off=205, st_on=0.55, st_mid=0.04, st_80=0.03, flags=["st_j_unreliable"]),
        ]
        quality = {lead: _quality(lead) for lead in leads}

        out = _apply_st_j_raw_remeasurement(
            beats,
            ecg=ecg,
            fs=fs,
            quality=quality,
            r_locs=np.asarray([130], dtype=int),
        )

        repaired = next(feature for feature in out if feature.lead == "V5")
        untouched = [feature for feature in out if feature.lead != "V5"]
        self.assertIn("st_j_remeasured_by_consensus", repaired.flags)
        self.assertEqual(160, repaired.st_j_remeasured_index)
        self.assertLess(repaired.st_on_mv, 0.10)
        self.assertTrue(all("st_j_remeasured_by_consensus" not in feature.flags for feature in untouched))

    def test_raw_remeasurement_pass_preserves_stable_st_elevation(self) -> None:
        fs = 500
        ecg = _synthetic_ecg(leads=5)
        ecg[:, 160:260] = 0.22
        leads = ["I", "II", "III", "aVR", "V5"]
        beats = [
            _beat("I", qrs_off=160, st_on=0.22, st_mid=0.22, st_80=0.22),
            _beat("II", qrs_off=161, st_on=0.22, st_mid=0.22, st_80=0.22),
            _beat("III", qrs_off=159, st_on=0.22, st_mid=0.22, st_80=0.22),
            _beat("aVR", qrs_off=160, st_on=0.22, st_mid=0.22, st_80=0.22),
            _beat("V5", qrs_off=160, st_on=0.22, st_mid=0.23, st_80=0.21),
        ]
        quality = {lead: _quality(lead) for lead in leads}

        out = _apply_st_j_raw_remeasurement(
            beats,
            ecg=ecg,
            fs=fs,
            quality=quality,
            r_locs=np.asarray([130], dtype=int),
        )

        self.assertTrue(all("st_j_remeasured_by_consensus" not in feature.flags for feature in out))
        self.assertAlmostEqual(0.22, next(feature for feature in out if feature.lead == "V5").st_on_mv, places=6)


if __name__ == "__main__":
    unittest.main()
