import unittest

import numpy as np

from feature_extraction.ecgfeat.delineate import (
    _repair_t_end_outliers,
    _t_end_repair_candidate,
    _t_end_repair_target,
)
from feature_extraction.ecgfeat.models import (
    STANDARD_12_LEADS,
    LeadBeatFeatures,
    LeadQuality,
    WaveBounds,
)


def _qt_reliable_quality(lead: str) -> LeadQuality:
    return LeadQuality(
        lead=lead, baseline_wander_score=0.0, muscle_noise_score=0.0,
        powerline_score=0.0, clipping_score=0.0, flatline_score=1.0,
        missing=False, reliable=True, reliable_for_p=True,
        reliable_for_qrs=True, reliable_for_t=True, reliable_for_qt=True,
    )


def _t_beat(lead, t_off, t_peak, t_on, qrs_on, qrs_off, qt_conf,
            t_signed_area=None):
    return LeadBeatFeatures(
        lead=lead, beat_id=0,
        p=WaveBounds(onset=None, peak=None, offset=None),
        qrs=WaveBounds(onset=qrs_on, peak=None, offset=qrs_off),
        t=WaveBounds(onset=t_on, peak=t_peak, offset=t_off),
        qt_ms=None, pr_ms=None, qrs_ms=None, p_amp_mv=None, qrs_area=None,
        q_amp_mv=None, r_amp_mv=None, s_amp_mv=None, st_on_mv=None,
        st_mid_mv=None, st_80ms_mv=None, t_amp_mv=0.5, j_index=None,
        t_signed_area=t_signed_area, qt_confidence=qt_conf,
    )


def _synthetic_ecg_with_t(n_leads=12, length=600, peak=350, foot=320, end=385):
    """12-lead array; every lead carries the same triangular positive T wave
    rising foot->peak and falling peak->end, flat (0) elsewhere."""
    ecg = np.zeros((n_leads, length), dtype=float)
    for s in range(foot, peak + 1):
        ecg[:, s] = 0.5 * (s - foot) / max(1, (peak - foot))
    for s in range(peak + 1, end + 1):
        ecg[:, s] = 0.5 * (end - s) / max(1, (end - peak))
    return ecg


def _synthetic_ecg_with_lead_specific_t(
    *,
    late_tail_leads,
    n_leads=12,
    length=600,
    peak=350,
    foot=320,
    early_end=360,
    late_end=420,
):
    ecg = np.zeros((n_leads, length), dtype=float)
    lead_to_idx = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
    for lead in STANDARD_12_LEADS[:n_leads]:
        end = late_end if lead in late_tail_leads else early_end
        idx = lead_to_idx[lead]
        for s in range(foot, peak + 1):
            ecg[idx, s] = 0.5 * (s - foot) / max(1, (peak - foot))
        for s in range(peak + 1, end + 1):
            ecg[idx, s] = 0.5 * (end - s) / max(1, (end - peak))
    return ecg


class ConsensusTargetTests(unittest.TestCase):
    def test_t_offset_repair_fields_default_to_none(self) -> None:
        feature = _t_beat("II", 386, 350, 320, 240, 260, 0.70)

        self.assertIsNone(feature.t_offset_original_index)
        self.assertIsNone(feature.t_offset_repaired_index)
        self.assertIsNone(feature.t_offset_repair_delta_ms)
        self.assertIsNone(feature.t_offset_repair_reason)
        self.assertIsNone(feature.t_offset_repair_confidence)
        self.assertIsNone(feature.t_offset_repair_consensus_index)
        self.assertIsNone(feature.t_offset_repair_support)
        self.assertIsNone(feature.t_offset_repair_used_leads)
        self.assertIsNone(feature.t_offset_repair_excluded_leads)

    def test_t_end_repair_target_uses_measurement_core_and_marks_outliers(self) -> None:
        fs = 500
        beats = [
            _t_beat("I", 400, 350, 320, 240, 260, 0.80),
            _t_beat("II", 402, 350, 320, 240, 260, 0.80),
            _t_beat("III", 398, 350, 320, 240, 260, 0.80),
            _t_beat("aVR", 401, 350, 320, 240, 260, 0.80),
            _t_beat("V5", 460, 350, 320, 240, 260, 0.15),
            _t_beat("V6", 342, 330, 310, 240, 260, 0.15),
        ]
        beats[4].flags.append("t_end_fallback")
        beats[5].flags.append("t_end_fallback")
        quality = {beat.lead: _qt_reliable_quality(beat.lead) for beat in beats}

        target = _t_end_repair_target(beats, fs=fs, quality=quality)

        self.assertIsNotNone(target)
        self.assertEqual(401, target.offset)
        self.assertEqual(4, target.support)
        self.assertIn("II", target.used_leads)
        self.assertIn("V5", target.excluded_leads)
        self.assertIn("V5", target.late_outlier_leads)
        self.assertIn("V6", target.early_outlier_leads)

    def test_t_end_repair_target_returns_none_with_too_few_reliable_leads(self) -> None:
        fs = 500
        beats = [
            _t_beat("I", 400, 350, 320, 240, 260, 0.80),
            _t_beat("II", 402, 350, 320, 240, 260, 0.80),
            _t_beat("V5", 460, 350, 320, 240, 260, 0.15),
        ]
        quality = {beat.lead: _qt_reliable_quality(beat.lead) for beat in beats}

        self.assertIsNone(_t_end_repair_target(beats, fs=fs, quality=quality))


class RepairPassTests(unittest.TestCase):
    def _quality(self):
        return {lead: _qt_reliable_quality(lead) for lead in STANDARD_12_LEADS}

    def _beats(self):
        return [
            _t_beat("I",   385, 350, 320, 240, 260, 0.70),
            _t_beat("II",  386, 350, 320, 240, 260, 0.70),
            _t_beat("III", 384, 350, 320, 240, 260, 0.70),
            _t_beat("aVR", 385, 350, 320, 240, 260, 0.70),
            _t_beat("aVL", 386, 350, 320, 240, 260, 0.70),
            _t_beat("aVF", 460, 350, 320, 240, 260, 0.15),  # gross-late, low conf
        ]

    def test_low_conf_late_outlier_is_pulled_toward_consensus(self):
        fs = 500
        r_locs = np.asarray([250])
        ecg = _synthetic_ecg_with_t()
        out = _repair_t_end_outliers(self._beats(), ecg, fs, self._quality(), r_locs)
        avf = next(b for b in out if b.lead == "aVF")
        self.assertLess(avf.t.offset, 460)
        self.assertLess(abs(avf.t.offset - 385), 20)
        self.assertIn("t_end_consensus_repaired", avf.flags)
        self.assertEqual("consensus_outlier", avf.t_end_repair_reason)
        self.assertIsNotNone(avf.t_signed_area)
        self.assertGreater(avf.t_signed_area, 0.0)  # positive T preserved
        self.assertAlmostEqual(
            avf.qt_ms, (avf.t.offset - 240) * 1000.0 / fs, places=6
        )

    def test_consistent_high_conf_lead_untouched(self):
        fs = 500
        r_locs = np.asarray([250])
        ecg = _synthetic_ecg_with_t()
        out = _repair_t_end_outliers(self._beats(), ecg, fs, self._quality(), r_locs)
        lead_ii = next(b for b in out if b.lead == "II")
        self.assertEqual(386, lead_ii.t.offset)
        self.assertNotIn("t_end_consensus_repaired", lead_ii.flags)

    def test_t_end_repair_candidate_extends_early_truncated_tail(self) -> None:
        fs = 500
        sig = np.zeros(600, dtype=float)
        sig[320:351] = np.linspace(0.0, 0.45, 31)
        sig[351:401] = np.linspace(0.45, 0.0, 50)
        feature = _t_beat("V6", 340, 350, 320, 240, 260, 0.15)
        feature.flags.append("t_end_fallback")
        target = type("Target", (), {
            "offset": 400,
            "support": 5,
            "early_outlier_leads": "V6",
            "late_outlier_leads": "",
        })()

        repaired, reason = _t_end_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNotNone(repaired)
        self.assertEqual("early_truncation", reason)
        self.assertLessEqual(abs(int(repaired) - 400), 20)

    def test_t_end_repair_candidate_pulls_late_low_confidence_offset(self) -> None:
        fs = 500
        sig = np.zeros(600, dtype=float)
        sig[320:351] = np.linspace(0.0, 0.45, 31)
        sig[351:401] = np.linspace(0.45, 0.0, 50)
        feature = _t_beat("aVF", 460, 350, 320, 240, 260, 0.15)
        feature.flags.append("t_end_fallback")
        target = type("Target", (), {
            "offset": 400,
            "support": 5,
            "early_outlier_leads": "",
            "late_outlier_leads": "aVF",
        })()

        repaired, reason = _t_end_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNotNone(repaired)
        self.assertEqual("late_outlier", reason)
        self.assertLessEqual(abs(int(repaired) - 400), 20)

    def test_t_end_repair_candidate_rejects_high_confidence_true_long_t(self) -> None:
        fs = 500
        sig = np.zeros(650, dtype=float)
        sig[320:381] = np.linspace(0.0, 0.45, 61)
        sig[381:481] = np.linspace(0.45, 0.0, 100)
        feature = _t_beat("V5", 480, 380, 320, 240, 260, 0.90)
        target = type("Target", (), {
            "offset": 400,
            "support": 5,
            "early_outlier_leads": "",
            "late_outlier_leads": "V5",
        })()

        repaired, reason = _t_end_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNone(repaired)
        self.assertIsNone(reason)

    def test_t_end_repair_candidate_rejects_short_qt(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        sig[180:221] = np.hanning(41) * 0.30
        feature = _t_beat("V5", 220, 200, 180, 150, 170, 0.15)
        feature.flags.append("t_end_fallback")
        target = type("Target", (), {
            "offset": 230,
            "support": 5,
            "early_outlier_leads": "",
            "late_outlier_leads": "V5",
        })()

        repaired, reason = _t_end_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNone(repaired)
        self.assertIsNone(reason)

    def test_t_end_repair_candidate_rejects_borderline_short_qt(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        sig[180:211] = np.linspace(0.0, 0.35, 31)
        sig[211:231] = np.linspace(0.35, 0.0, 20)
        feature = _t_beat("V5", 320, 210, 180, 100, 130, 0.15)
        feature.flags.append("t_end_fallback")
        target = type("Target", (), {
            "offset": 230,
            "support": 5,
            "early_outlier_leads": "",
            "late_outlier_leads": "V5",
        })()

        repaired, reason = _t_end_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            next_qrs_guard=None,
        )

        self.assertIsNone(repaired)
        self.assertIsNone(reason)

    def test_t_end_repair_candidate_bounds_geometric_search_to_target_not_outlier(self) -> None:
        # Regression for LUDB record 24 lead V2: a late_outlier's geometric
        # re-search window used to extend past the *old* (outlier) offset
        # rather than the consensus target, so it could re-find the same
        # distal artifact that produced the outlier in the first place.
        from unittest.mock import patch

        fs = 500
        sig = np.zeros(600, dtype=float)
        feature = _t_beat("V2", 460, 350, 320, 240, 260, 0.15)
        target = type("Target", (), {
            "offset": 400,
            "support": 5,
            "early_outlier_leads": "",
            "late_outlier_leads": "V2",
        })()

        captured = {}

        def fake_geometric(sig_arg, t_peak_local, search_end_local, baseline, fs_arg):
            captured["search_end_local"] = search_end_local
            return search_end_local, 0.5, "stub"

        with patch(
            "feature_extraction.ecgfeat.delineate._t_end_geometric",
            side_effect=fake_geometric,
        ):
            repaired, reason = _t_end_repair_candidate(
                feature,
                sig=sig,
                beat_start=0,
                baseline=0.0,
                fs=fs,
                target=target,
                next_qrs_guard=None,
            )

        margin_samples = int(round(30.0 * fs / 1000.0))
        self.assertEqual(400 + margin_samples, captured["search_end_local"])
        self.assertEqual("late_outlier", reason)
        self.assertIsNotNone(repaired)

    def test_apply_t_end_repair_records_provenance_and_recomputes_fields(self) -> None:
        from feature_extraction.ecgfeat.delineate import _apply_t_end_repair

        fs = 500
        sig = np.zeros(600, dtype=float)
        sig[320:351] = np.linspace(0.0, 0.45, 31)
        sig[351:401] = np.linspace(0.45, 0.0, 50)
        feature = _t_beat("aVF", 460, 350, 320, 240, 260, 0.15)
        target = type("Target", (), {
            "offset": 400,
            "support": 5,
            "used_leads": "I,II,III,aVR",
            "excluded_leads": "aVF",
        })()

        _apply_t_end_repair(
            feature,
            400,
            sig,
            0,
            0.0,
            fs,
            reason="late_outlier",
            target=target,
            confidence=0.80,
        )

        self.assertEqual(400, feature.t.offset)
        self.assertEqual(460, feature.t_offset_original_index)
        self.assertEqual(400, feature.t_offset_repaired_index)
        self.assertAlmostEqual(-120.0, feature.t_offset_repair_delta_ms)
        self.assertEqual("late_outlier", feature.t_offset_repair_reason)
        self.assertEqual(400, feature.t_offset_repair_consensus_index)
        self.assertEqual(5, feature.t_offset_repair_support)
        self.assertEqual("I,II,III,aVR", feature.t_offset_repair_used_leads)
        self.assertEqual("aVF", feature.t_offset_repair_excluded_leads)
        self.assertIn("t_offset_repaired_by_consensus", feature.flags)
        self.assertIn("t_offset_late_outlier_repaired", feature.flags)
        self.assertAlmostEqual(320.0, feature.qt_ms)
        self.assertAlmostEqual(280.0, feature.jt_ms)
        self.assertAlmostEqual(100.0, feature.tpe_ms)
        self.assertIsNotNone(feature.t_area)
        self.assertIsNotNone(feature.t_signed_area)
        self.assertIsNotNone(feature.t_dur_ms)

    def test_raw_repair_pass_extends_only_early_truncated_outlier(self) -> None:
        fs = 500
        r_locs = np.asarray([250])
        ecg = _synthetic_ecg_with_t(end=400)
        beats = [
            _t_beat("I", 400, 350, 320, 240, 260, 0.80),
            _t_beat("II", 402, 350, 320, 240, 260, 0.80),
            _t_beat("III", 398, 350, 320, 240, 260, 0.80),
            _t_beat("aVR", 401, 350, 320, 240, 260, 0.80),
            _t_beat("V6", 340, 350, 320, 240, 260, 0.15),
        ]
        beats[-1].flags.append("t_end_fallback")

        out = _repair_t_end_outliers(beats, ecg, fs, self._quality(), r_locs)

        repaired = next(feature for feature in out if feature.lead == "V6")
        untouched = [feature for feature in out if feature.lead != "V6"]
        self.assertIn("t_offset_repaired_by_consensus", repaired.flags)
        self.assertIn("t_offset_early_truncation_repaired", repaired.flags)
        self.assertGreater(repaired.t.offset, 340)
        self.assertLessEqual(abs(int(repaired.t.offset) - 400), 20)
        self.assertEqual(340, repaired.t_offset_original_index)
        self.assertEqual(401, repaired.t_offset_repair_consensus_index)
        self.assertTrue(all("t_offset_repaired_by_consensus" not in feature.flags for feature in untouched))

    def test_raw_repair_pass_extends_systematic_early_t_tail_cluster(self) -> None:
        fs = 500
        r_locs = np.asarray([250])
        ecg = _synthetic_ecg_with_t(end=420)
        beats = [
            _t_beat("I", 360, 350, 320, 220, 260, 0.30),
            _t_beat("II", 360, 350, 320, 220, 260, 0.30),
            _t_beat("III", 360, 350, 320, 220, 260, 0.30),
            _t_beat("aVR", 360, 350, 320, 220, 260, 0.30),
            _t_beat("aVL", 360, 350, 320, 220, 260, 0.30),
        ]

        out = _repair_t_end_outliers(beats, ecg, fs, self._quality(), r_locs)

        repaired = [feature for feature in out if "t_offset_systematic_early_tail_repaired" in feature.flags]
        self.assertGreaterEqual(len(repaired), 4)
        for feature in repaired:
            self.assertGreater(feature.t.offset, 390)
            self.assertLessEqual(abs(int(feature.t.offset) - 420), 20)
            self.assertEqual("systematic_early_tail", feature.t_offset_repair_reason)
            self.assertEqual(360, feature.t_offset_original_index)
            self.assertGreater(feature.qt_ms, 300.0)

    def test_raw_repair_pass_extends_high_confidence_systematic_early_t_tail_cluster(self) -> None:
        fs = 500
        r_locs = np.asarray([250])
        ecg = _synthetic_ecg_with_t(peak=300, foot=260, end=420)
        beats = [
            _t_beat("I", 360, 300, 260, 180, 220, 0.80),
            _t_beat("II", 360, 300, 260, 180, 220, 0.80),
            _t_beat("III", 360, 300, 260, 180, 220, 0.80),
            _t_beat("aVR", 360, 300, 260, 180, 220, 0.80),
            _t_beat("aVL", 360, 300, 260, 180, 220, 0.80),
        ]

        out = _repair_t_end_outliers(beats, ecg, fs, self._quality(), r_locs)

        repaired = [feature for feature in out if "t_offset_systematic_early_tail_repaired" in feature.flags]
        self.assertGreaterEqual(len(repaired), 4)
        for feature in repaired:
            self.assertGreater(feature.t.offset, 390)
            self.assertLessEqual(abs(int(feature.t.offset) - 420), 20)
            self.assertEqual("systematic_early_tail", feature.t_offset_repair_reason)
            self.assertEqual(360, feature.t_offset_original_index)

    def test_raw_repair_pass_extends_strong_three_lead_late_tail_cluster(self) -> None:
        fs = 500
        r_locs = np.asarray([250])
        late_tail_leads = {"V2", "V3", "V4"}
        ecg = _synthetic_ecg_with_lead_specific_t(late_tail_leads=late_tail_leads)
        beats = [
            _t_beat("I", 360, 350, 320, 220, 260, 0.80),
            _t_beat("II", 360, 350, 320, 220, 260, 0.80),
            _t_beat("III", 360, 350, 320, 220, 260, 0.80),
            _t_beat("V2", 360, 350, 320, 220, 260, 0.80),
            _t_beat("V3", 360, 350, 320, 220, 260, 0.80),
            _t_beat("V4", 360, 350, 320, 220, 260, 0.80),
        ]

        out = _repair_t_end_outliers(beats, ecg, fs, self._quality(), r_locs)

        repaired = [
            feature
            for feature in out
            if "t_offset_systematic_early_tail_repaired" in feature.flags
        ]
        self.assertEqual(late_tail_leads, {feature.lead for feature in repaired})
        for feature in repaired:
            self.assertGreater(feature.t.offset, 390)
            self.assertLessEqual(abs(int(feature.t.offset) - 420), 20)
            self.assertEqual(3, feature.t_offset_repair_support)
            self.assertEqual("systematic_early_tail", feature.t_offset_repair_reason)

    def test_raw_repair_pass_rejects_three_lead_late_tail_when_tpe_is_not_short(self) -> None:
        fs = 500
        r_locs = np.asarray([250])
        late_tail_leads = {"V2", "V3", "V4"}
        ecg = _synthetic_ecg_with_lead_specific_t(
            late_tail_leads=late_tail_leads,
            peak=330,
            early_end=390,
            late_end=420,
        )
        beats = [
            _t_beat("I", 390, 330, 320, 220, 260, 0.80),
            _t_beat("II", 390, 330, 320, 220, 260, 0.80),
            _t_beat("III", 390, 330, 320, 220, 260, 0.80),
            _t_beat("V2", 390, 330, 320, 220, 260, 0.80),
            _t_beat("V3", 390, 330, 320, 220, 260, 0.80),
            _t_beat("V4", 390, 330, 320, 220, 260, 0.80),
        ]

        out = _repair_t_end_outliers(beats, ecg, fs, self._quality(), r_locs)

        self.assertFalse(
            any("t_offset_systematic_early_tail_repaired" in feature.flags for feature in out)
        )
        self.assertTrue(all(feature.t.offset == 390 for feature in out))

    def test_raw_repair_pass_preserves_high_confidence_long_t(self) -> None:
        fs = 500
        r_locs = np.asarray([250])
        ecg = _synthetic_ecg_with_t(end=480)
        beats = [
            _t_beat("I", 400, 350, 320, 240, 260, 0.80),
            _t_beat("II", 402, 350, 320, 240, 260, 0.80),
            _t_beat("III", 398, 350, 320, 240, 260, 0.80),
            _t_beat("aVR", 401, 350, 320, 240, 260, 0.80),
            _t_beat("V5", 480, 380, 320, 240, 260, 0.90),
        ]

        out = _repair_t_end_outliers(beats, ecg, fs, self._quality(), r_locs)

        v5 = next(feature for feature in out if feature.lead == "V5")
        self.assertEqual(480, v5.t.offset)
        self.assertNotIn("t_offset_repaired_by_consensus", v5.flags)


if __name__ == "__main__":
    unittest.main()
