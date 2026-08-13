import unittest
from unittest.mock import patch

import numpy as np

from feature_extraction.ecgfeat.delineate import (
    TDualEndpointEvidence,
    TDualRescueTarget,
    _apply_t_dual_method_raw_rescue,
    _apply_t_offset_dual_rescue,
    _t_dual_rescue_candidate,
    _t_dual_rescue_target,
    _t_offset_dual_evidence,
    _t_offset_slope_return,
    _t_offset_tangent,
)
from feature_extraction.ecgfeat.models import LeadBeatFeatures, LeadQuality, WaveBounds


def _quality(lead: str, *, qt: bool = True) -> LeadQuality:
    return LeadQuality(
        lead=lead,
        baseline_wander_score=0.0,
        muscle_noise_score=0.0,
        powerline_score=0.0,
        clipping_score=0.0,
        flatline_score=1.0,
        missing=False,
        reliable=qt,
        reliable_for_p=qt,
        reliable_for_qrs=qt,
        reliable_for_t=qt,
        reliable_for_qt=qt,
    )


def _beat(
    lead: str,
    *,
    beat_id: int = 0,
    qrs_on: int = 240,
    qrs_off: int = 260,
    t_on: int = 320,
    t_peak: int = 350,
    t_off: int | None = 400,
    qt_conf: float = 0.80,
    flags: list[str] | None = None,
    method: str = "dxl_chord",
) -> LeadBeatFeatures:
    feature = LeadBeatFeatures(
        lead=lead,
        beat_id=beat_id,
        p=WaveBounds(onset=None, peak=None, offset=None),
        qrs=WaveBounds(onset=qrs_on, peak=None, offset=qrs_off),
        t=WaveBounds(onset=t_on, peak=t_peak, offset=t_off),
        qt_ms=None if t_off is None else (t_off - qrs_on) * 2.0,
        pr_ms=None,
        qrs_ms=(qrs_off - qrs_on) * 2.0,
        p_amp_mv=None,
        qrs_area=None,
        q_amp_mv=None,
        r_amp_mv=None,
        s_amp_mv=None,
        st_on_mv=None,
        st_mid_mv=None,
        st_80ms_mv=None,
        t_amp_mv=0.45,
        j_index=qrs_off,
        qt_confidence=qt_conf,
        t_end_method=method,
    )
    feature.flags = list(flags or [])
    return feature


def _triangular_t_signal(
    *,
    length: int = 650,
    t_on: int = 320,
    peak: int = 350,
    end: int = 420,
    amp: float = 0.45,
) -> np.ndarray:
    sig = np.zeros(length, dtype=float)
    sig[t_on:peak + 1] = np.linspace(0.0, amp, peak - t_on + 1)
    sig[peak + 1:end + 1] = np.linspace(amp, 0.0, end - peak)
    return sig


def _ecg_with_t(
    leads: list[str],
    *,
    length: int = 650,
    t_on: int = 320,
    peak: int = 350,
    end: int = 420,
) -> np.ndarray:
    base = _triangular_t_signal(length=length, t_on=t_on, peak=peak, end=end)
    return np.vstack([base.copy() for _lead in leads])


class TDualModelFieldTests(unittest.TestCase):
    def test_t_offset_dual_fields_default_to_none(self) -> None:
        feature = _beat("II")

        self.assertIsNone(feature.t_offset_dual_original_index)
        self.assertIsNone(feature.t_offset_dual_rescued_index)
        self.assertIsNone(feature.t_offset_dual_delta_ms)
        self.assertIsNone(feature.t_offset_dual_reason)
        self.assertIsNone(feature.t_offset_dual_confidence)
        self.assertIsNone(feature.t_offset_dual_consensus_index)
        self.assertIsNone(feature.t_offset_dual_support)
        self.assertIsNone(feature.t_offset_dual_used_leads)
        self.assertIsNone(feature.t_offset_dual_excluded_leads)
        self.assertIsNone(feature.t_offset_dual_chord_index)
        self.assertIsNone(feature.t_offset_dual_slope_index)
        self.assertIsNone(feature.t_offset_dual_tangent_index)
        self.assertIsNone(feature.t_offset_dual_method_spread_ms)


class TDualEndpointTests(unittest.TestCase):
    def test_slope_return_finds_stable_baseline_after_t_tail(self) -> None:
        fs = 500
        sig = _triangular_t_signal(end=420)

        offset = _t_offset_slope_return(
            sig,
            t_peak=350,
            search_end=470,
            baseline=0.0,
            fs=fs,
        )

        self.assertIsNotNone(offset)
        self.assertGreaterEqual(offset, 410)
        self.assertLessEqual(offset, 430)

    def test_tangent_handles_upright_and_inverted_t(self) -> None:
        fs = 500
        upright = _triangular_t_signal(end=420)
        inverted = -upright

        upright_off = _t_offset_tangent(upright, t_peak=350, search_end=470, baseline=0.0, fs=fs)
        inverted_off = _t_offset_tangent(inverted, t_peak=350, search_end=470, baseline=0.0, fs=fs)

        self.assertIsNotNone(upright_off)
        self.assertIsNotNone(inverted_off)
        self.assertGreaterEqual(upright_off, 390)
        self.assertGreaterEqual(inverted_off, 390)
        self.assertLessEqual(abs(int(upright_off) - int(inverted_off)), 4)

    def test_dual_evidence_records_method_spread(self) -> None:
        fs = 500
        sig = _triangular_t_signal(end=430)

        evidence = _t_offset_dual_evidence(
            sig,
            t_peak=350,
            search_end=480,
            baseline=0.0,
            fs=fs,
        )

        self.assertIsNotNone(evidence.chord_index)
        self.assertIsNotNone(evidence.slope_index)
        self.assertIsNotNone(evidence.tangent_index)
        self.assertIsNotNone(evidence.method_spread_ms)
        self.assertGreaterEqual(evidence.best_index, 400)


class TDualTargetCandidateTests(unittest.TestCase):
    def test_target_marks_early_low_confidence_outlier(self) -> None:
        fs = 500
        beats = [
            _beat("I", t_off=420, qt_conf=0.80),
            _beat("II", t_off=422, qt_conf=0.80),
            _beat("III", t_off=418, qt_conf=0.80),
            _beat("aVR", t_off=421, qt_conf=0.80),
            _beat("V6", t_off=350, qt_conf=0.15, flags=["t_end_fallback"], method="threshold_fallback"),
        ]
        quality = {beat.lead: _quality(beat.lead) for beat in beats}

        target = _t_dual_rescue_target(beats, fs=fs, quality=quality)

        self.assertIsNotNone(target)
        self.assertEqual(421, target.offset)
        self.assertEqual(4, target.support)
        self.assertIn("I", target.used_leads)
        self.assertIn("V6", target.early_outlier_leads)

    def test_candidate_extends_early_truncated_t_when_methods_and_consensus_agree(self) -> None:
        fs = 500
        sig = _triangular_t_signal(end=420)
        feature = _beat("V6", t_off=350, qt_conf=0.15, flags=["t_end_fallback"], method="threshold_fallback")
        target = TDualRescueTarget(
            offset=421,
            support=4,
            used_leads="I,II,III,aVR",
            excluded_leads="V6",
            early_outlier_leads="V6",
        )

        rescued, reason, evidence = _t_dual_rescue_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNotNone(rescued)
        self.assertEqual("early_truncation_dual_method", reason)
        self.assertGreater(rescued, 390)
        self.assertIsNotNone(evidence.slope_index)
        self.assertIsNotNone(evidence.tangent_index)

    def test_candidate_preserves_high_confidence_true_short_t(self) -> None:
        fs = 500
        sig = _triangular_t_signal(end=360)
        feature = _beat("V5", t_peak=330, t_off=360, qt_conf=0.90)
        target = TDualRescueTarget(
            offset=421,
            support=4,
            used_leads="I,II,III,aVR",
            excluded_leads="V5",
            early_outlier_leads="V5",
        )

        rescued, reason, evidence = _t_dual_rescue_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNone(rescued)
        self.assertIsNone(reason)
        self.assertIsNotNone(evidence)

    def test_candidate_rejects_rescue_that_would_remain_short_qt(self) -> None:
        fs = 500
        sig = _triangular_t_signal(t_on=160, peak=190, end=230)
        feature = _beat(
            "V5",
            qrs_on=100,
            qrs_off=130,
            t_on=160,
            t_peak=190,
            t_off=200,
            qt_conf=0.15,
            flags=["t_end_fallback"],
            method="threshold_fallback",
        )
        target = TDualRescueTarget(
            offset=230,
            support=4,
            used_leads="I,II,III,aVR",
            excluded_leads="V5",
            early_outlier_leads="V5",
        )

        rescued, reason, _evidence = _t_dual_rescue_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNone(rescued)
        self.assertIsNone(reason)

    def test_candidate_rejects_high_method_disagreement(self) -> None:
        fs = 500
        sig = _triangular_t_signal(end=420)
        feature = _beat("V6", t_off=350, qt_conf=0.15, flags=["t_end_fallback"], method="threshold_fallback")
        target = TDualRescueTarget(
            offset=421,
            support=4,
            used_leads="I,II,III,aVR",
            excluded_leads="V6",
            early_outlier_leads="V6",
        )
        evidence = TDualEndpointEvidence(
            chord_index=360,
            slope_index=420,
            tangent_index=430,
            best_index=430,
            method_spread_ms=140.0,
            confidence=0.45,
            reason="method_disagreement",
        )

        with patch("feature_extraction.ecgfeat.delineate._t_offset_dual_evidence", return_value=evidence):
            rescued, reason, returned = _t_dual_rescue_candidate(
                feature,
                sig=sig,
                beat_start=0,
                baseline=0.0,
                fs=fs,
                target=target,
                next_qrs_guard=None,
            )

        self.assertIs(returned, evidence)
        self.assertIsNone(rescued)
        self.assertIsNone(reason)


class TDualApplyPassTests(unittest.TestCase):
    def test_apply_dual_rescue_records_provenance_and_recomputes_fields(self) -> None:
        fs = 500
        sig = _triangular_t_signal(end=420)
        feature = _beat("V6", t_off=350, qt_conf=0.15, flags=["t_end_fallback"], method="threshold_fallback")
        target = TDualRescueTarget(
            offset=421,
            support=4,
            used_leads="I,II,III,aVR",
            excluded_leads="V6",
            early_outlier_leads="V6",
        )
        evidence = _t_offset_dual_evidence(sig, t_peak=350, search_end=470, baseline=0.0, fs=fs)

        _apply_t_offset_dual_rescue(
            feature,
            new_off_global=420,
            sig_beat=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            reason="early_truncation_dual_method",
            target=target,
            evidence=evidence,
        )

        self.assertEqual(420, feature.t.offset)
        self.assertEqual(350, feature.t_offset_dual_original_index)
        self.assertEqual(420, feature.t_offset_dual_rescued_index)
        self.assertAlmostEqual(140.0, feature.t_offset_dual_delta_ms)
        self.assertEqual("early_truncation_dual_method", feature.t_offset_dual_reason)
        self.assertEqual(421, feature.t_offset_dual_consensus_index)
        self.assertEqual(4, feature.t_offset_dual_support)
        self.assertEqual("I,II,III,aVR", feature.t_offset_dual_used_leads)
        self.assertIsNotNone(feature.t_offset_dual_method_spread_ms)
        self.assertIn("t_offset_dual_method_rescued", feature.flags)
        self.assertIn("t_offset_early_truncation_dual_rescued", feature.flags)
        self.assertAlmostEqual(360.0, feature.qt_ms)
        self.assertAlmostEqual(320.0, feature.jt_ms)
        self.assertAlmostEqual(140.0, feature.tpe_ms)
        self.assertIsNotNone(feature.t_area)
        self.assertIsNotNone(feature.t_signed_area)
        self.assertIsNotNone(feature.t_dur_ms)

    def test_raw_rescue_pass_extends_only_early_truncated_lead(self) -> None:
        fs = 500
        leads = ["I", "II", "III", "aVR", "V6"]
        ecg = _ecg_with_t(leads, end=420)
        beats = [
            _beat("I", t_off=420, qt_conf=0.80),
            _beat("II", t_off=422, qt_conf=0.80),
            _beat("III", t_off=418, qt_conf=0.80),
            _beat("aVR", t_off=421, qt_conf=0.80),
            _beat("V6", t_off=350, qt_conf=0.15, flags=["t_end_fallback"], method="threshold_fallback"),
        ]
        quality = {lead: _quality(lead) for lead in leads}

        out = _apply_t_dual_method_raw_rescue(
            beats,
            ecg=ecg,
            fs=fs,
            quality=quality,
            r_locs=np.asarray([250], dtype=int),
        )

        repaired = next(feature for feature in out if feature.lead == "V6")
        untouched = [feature for feature in out if feature.lead != "V6"]
        self.assertIn("t_offset_dual_method_rescued", repaired.flags)
        self.assertGreater(repaired.t.offset, 390)
        self.assertTrue(all("t_offset_dual_method_rescued" not in feature.flags for feature in untouched))


if __name__ == "__main__":
    unittest.main()
