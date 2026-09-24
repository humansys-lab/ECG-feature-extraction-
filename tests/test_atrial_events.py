import unittest

import numpy as np

from feature_extraction.ecgfeat.atrial import (
    _associate_event_to_qrs,
    _beat_t_windows,
    _cluster_atrial_candidates,
    _composite_raw_p_detection,
    compute_organized_p_ratio,
    extract_atrial_events,
    select_atrial_leads,
    _scan_interbeat_atrial_candidates,
)
from feature_extraction.ecgfeat.export import _attach_assessment_boundaries
from feature_extraction.ecgfeat.models import LeadBeatFeatures, LeadQuality, WaveBounds
from tests.optional import needs_interpretation


def _good_quality(lead: str) -> LeadQuality:
    return LeadQuality(
        lead=lead,
        baseline_wander_score=0.0,
        muscle_noise_score=0.0,
        powerline_score=0.0,
        clipping_score=0.0,
        flatline_score=1.0,
        missing=False,
        reliable=True,
        reliable_for_p=True,
        reliable_for_qrs=True,
        reliable_for_t=True,
        reliable_for_qt=True,
    )


def _beat_feature(
    lead: str,
    beat_id: int,
    p_peak: int | None,
    qrs_onset: int,
    qrs_offset: int,
    p_confidence: float = 0.90,
) -> LeadBeatFeatures:
    return LeadBeatFeatures(
        lead=lead,
        beat_id=beat_id,
        p=WaveBounds(onset=None, peak=p_peak, offset=None),
        qrs=WaveBounds(onset=qrs_onset, peak=None, offset=qrs_offset),
        t=WaveBounds(onset=None, peak=None, offset=None),
        qt_ms=None,
        pr_ms=None,
        qrs_ms=None,
        p_amp_mv=None,
        qrs_area=None,
        q_amp_mv=None,
        r_amp_mv=None,
        s_amp_mv=None,
        st_on_mv=None,
        st_mid_mv=None,
        st_80ms_mv=None,
        t_amp_mv=None,
        j_index=None,
        p_confidence=p_confidence,
    )


class AtrialEventTests(unittest.TestCase):
    def test_cluster_atrial_candidates_merges_multilead_hits(self) -> None:
        events = _cluster_atrial_candidates(
            [
                {"sample": 400, "lead": "II", "confidence": 0.90},
                {"sample": 408, "lead": "V1", "confidence": 0.70},
                {"sample": 790, "lead": "II", "confidence": 0.60},
            ],
            merge_samples=20,
        )

        self.assertEqual(2, len(events))
        self.assertEqual(["II", "V1"], events[0]["source_leads"])
        self.assertEqual(404, events[0]["sample"])
        self.assertGreater(events[0]["confidence"], 0.75)

    def test_associate_event_marks_conducted_blocked_and_retrograde(self) -> None:
        qrs_onsets = np.asarray([480, 980], dtype=int)
        qrs_offsets = np.asarray([525, 1025], dtype=int)

        self.assertEqual(
            (0, "conducted", 160.0),
            _associate_event_to_qrs(400, qrs_onsets, qrs_offsets, fs=500),
        )
        self.assertEqual(
            (None, "blocked", None),
            _associate_event_to_qrs(790, qrs_onsets, qrs_offsets, fs=500),
        )
        self.assertEqual(
            (0, "retrograde", None),
            _associate_event_to_qrs(600, qrs_onsets, qrs_offsets, fs=500),
        )

    def test_associate_event_prefers_closest_boundary_in_overlap_window(self) -> None:
        qrs_onsets = np.asarray([480, 690], dtype=int)
        qrs_offsets = np.asarray([525, 735], dtype=int)

        self.assertEqual(
            (0, "retrograde", None),
            _associate_event_to_qrs(585, qrs_onsets, qrs_offsets, fs=500),
        )

    def test_deflection_inside_measured_t_wave_is_not_an_atrial_event(self) -> None:
        # The 0-240 ms post-QRS-offset "retrograde" window is the ST-T segment,
        # so an imperfectly cancelled T in the QRST-subtraction residual used to
        # be published as a retrograde P wave.
        qrs_onsets = np.asarray([480, 980], dtype=int)
        qrs_offsets = np.asarray([525, 1025], dtype=int)
        t_windows = {0: (580, 700)}

        self.assertEqual(
            (0, "retrograde", None),
            _associate_event_to_qrs(600, qrs_onsets, qrs_offsets, fs=500),
        )
        self.assertEqual(
            (None, "t_wave_artifact", None),
            _associate_event_to_qrs(
                600, qrs_onsets, qrs_offsets, fs=500, t_windows=t_windows
            ),
        )

    def test_t_wave_artifact_is_not_promoted_to_conducted(self) -> None:
        # A deflection can sit in one beat's T wave and simultaneously fall in
        # the next beat's conducted-P window. Suppressing only the retrograde
        # option would let the artifact through as an organised P wave, which is
        # worse than the retrograde label: it manufactures evidence against AF.
        qrs_onsets = np.asarray([480, 760], dtype=int)
        qrs_offsets = np.asarray([525, 805], dtype=int)
        # 210 ms after QRS offset 0 (retrograde window) and 260 ms before QRS
        # onset 1 (conducted window), so both associations are available and
        # proximity picks retrograde.
        event = 630

        self.assertEqual(
            (0, "retrograde", None),
            _associate_event_to_qrs(event, qrs_onsets, qrs_offsets, fs=500),
        )
        # Gated, it is rejected outright. Dropping only the retrograde candidate
        # would have left the conducted one and published a 260 ms PR.
        self.assertEqual(
            (None, "t_wave_artifact", None),
            _associate_event_to_qrs(
                event, qrs_onsets, qrs_offsets, fs=500, t_windows={0: (600, 700)}
            ),
        )

    def test_t_window_never_reaches_the_next_beats_p_wave(self) -> None:
        # An over-estimated T offset must not be able to delete a real conducted
        # P wave, so the window is clipped short of the conducted-P region.
        fs = 500
        beats = [
            _beat_feature("II", beat_id, p_peak=None, qrs_onset=on, qrs_offset=on + 45)
            for beat_id, on in enumerate((500, 1000))
        ]
        for beat in beats:
            # A T wave delineated absurdly late, running into the next P wave.
            beat.t = WaveBounds(onset=600, peak=700, offset=980)

        windows = _beat_t_windows(beats, fs, beat_count=2, qrs_onsets=np.asarray([500, 1000]))

        # 280 ms before QRS onset 1000 is sample 860; the window must stop there.
        self.assertLessEqual(windows[0][1], 860)

    def test_scan_interbeat_candidates_emits_blocked_p_wave(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1500), dtype=float)
        r_locs = np.asarray([500, 1000], dtype=int)
        ecg[1, 760:781] = 0.12 * np.hanning(21)
        ecg[7, 758:779] = 0.10 * np.hanning(21)
        quality = {lead: _good_quality(lead) for lead in ("I", "II", "III", "V1", "V2")}

        events = _scan_interbeat_atrial_candidates(
            ecg,
            fs=fs,
            r_locs=r_locs,
            qrs_offsets=np.asarray([525, 1025], dtype=int),
            quality=quality,
        )

        self.assertEqual(1, len(events))
        self.assertEqual("blocked", events[0]["association_type"])
        self.assertGreater(events[0]["confidence"], 0.30)

    @staticmethod
    def _synthetic_atrial_strip():
        fs = 500
        n = 5000
        rng = np.random.default_rng(0)
        ecg = rng.normal(0.0, 0.01, size=(12, n))
        t = np.arange(n)

        def gauss(center: int, width: float, amp: float) -> np.ndarray:
            return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

        r_times = [450, 1350, 2250, 3150, 4050]
        for p_time in range(300, n, 300):  # ~100 bpm atrial activity
            ecg[1] += gauss(p_time, 12, 0.12)  # lead II
            ecg[6] += gauss(p_time, 12, 0.08)  # lead V1
        for r_time in r_times:
            ecg[1] += gauss(r_time, 10, 1.2)
            ecg[6] += gauss(r_time, 10, -0.9)
        qrs_offsets = np.asarray([r + 40 for r in r_times], dtype=int)
        return ecg, fs, np.asarray(r_times, dtype=int), qrs_offsets

    def test_select_atrial_leads_picks_p_bearing_limb_and_precordial(self) -> None:
        ecg, fs, r_locs, _ = self._synthetic_atrial_strip()
        selection = select_atrial_leads(ecg, fs, r_locs, quality={})
        self.assertEqual("II", selection["limb_lead"])
        self.assertEqual("V1", selection["precordial_lead"])

    def test_composite_raw_p_detection_finds_interbeat_atrial_activity(self) -> None:
        ecg, fs, r_locs, qrs_offsets = self._synthetic_atrial_strip()
        events, meta = _composite_raw_p_detection(ecg, fs, r_locs, qrs_offsets, quality={})
        self.assertTrue(meta["available"])
        self.assertGreaterEqual(meta["event_count"], 6)
        self.assertTrue(all(ev["detection_method"] for ev in events))
        measured = max(events, key=lambda ev: float(ev["confidence"]))
        self.assertLess(measured["onset_sample"], measured["sample"])
        self.assertGreater(measured["offset_sample"], measured["sample"])
        self.assertAlmostEqual(
            measured["onset_sample"] * 1000.0 / fs,
            measured["onset_ms"],
        )
        self.assertAlmostEqual(
            measured["offset_sample"] * 1000.0 / fs,
            measured["offset_ms"],
        )
        self.assertGreater(measured["duration_ms"], 20.0)
        self.assertLess(measured["duration_ms"], 180.0)
        self.assertGreater(measured["area_mv_ms"], 0.0)
        self.assertIn("signed_area_mv_ms", measured)
        self.assertIn("amplitude_mv", measured)
        self.assertGreaterEqual(measured["template_similarity"], 0.0)
        self.assertLessEqual(measured["template_similarity"], 1.0)
        self.assertTrue(any(ev.get("pp_ms") is not None for ev in events))

    def test_composite_detection_requires_three_qrs(self) -> None:
        ecg, fs, _, _ = self._synthetic_atrial_strip()
        events, meta = _composite_raw_p_detection(
            ecg, fs, np.asarray([450, 1350], dtype=int), np.asarray([490, 1390], dtype=int), quality={}
        )
        self.assertFalse(meta["available"])
        self.assertEqual([], events)

    def test_extract_atrial_events_deduplicates_scanned_conducted_p_wave(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1500), dtype=float)
        r_locs = np.asarray([500, 1000], dtype=int)
        ecg[1, 890:911] = 0.12 * np.hanning(21)
        quality = {lead: _good_quality(lead) for lead in ("I", "II", "III", "V1", "V2")}
        beat_features = [
            _beat_feature("II", 0, None, qrs_onset=480, qrs_offset=525),
            _beat_feature("II", 1, 900, qrs_onset=980, qrs_offset=1025),
        ]

        events = extract_atrial_events(beat_features, quality, r_locs, fs, ecg=ecg)

        self.assertEqual(1, len(events))
        self.assertEqual("conducted", events[0]["association_type"])
        self.assertEqual(1, events[0]["associated_qrs_beat_id"])
        self.assertEqual(900, events[0]["sample"])

    def test_organized_p_ratio_counts_distinct_beats_not_raw_events(self) -> None:
        # Two qualifying events land on the same beat (e.g. an imperfectly
        # deduplicated per-lead detection plus a composite raw-scan hit);
        # the ratio must not exceed 1.0 by counting both.
        events = [
            {"association_type": "conducted", "confidence": 0.9, "associated_qrs_beat_id": 0, "pr_ms": 100.0},
            {"association_type": "conducted", "confidence": 0.8, "associated_qrs_beat_id": 0, "pr_ms": 105.0},
            {"association_type": "retrograde", "confidence": 0.7, "associated_qrs_beat_id": 1},
            {"association_type": "blocked", "confidence": 0.9, "associated_qrs_beat_id": None},
        ]

        ratio = compute_organized_p_ratio(events, beat_count=2)

        # Retrograde events do not establish an organized antegrade atrial
        # rhythm; only one of two QRS beats has a conducted P in the dominant
        # PR cluster.
        self.assertEqual(0.5, ratio)

    def test_organized_p_ratio_requires_consistent_pr_cluster(self) -> None:
        events = [
            {"association_type": "conducted", "confidence": 0.9, "associated_qrs_beat_id": 0, "pr_ms": 100.0},
            {"association_type": "conducted", "confidence": 0.9, "associated_qrs_beat_id": 1, "pr_ms": 108.0},
            {"association_type": "conducted", "confidence": 0.9, "associated_qrs_beat_id": 2, "pr_ms": 210.0},
            {"association_type": "conducted", "confidence": 0.9, "associated_qrs_beat_id": 3, "pr_ms": 280.0},
        ]

        ratio = compute_organized_p_ratio(events, beat_count=4)

        assert ratio == 0.5

    def test_organized_p_ratio_ignores_low_confidence_and_blocked_events(self) -> None:
        events = [
            {"association_type": "conducted", "confidence": 0.10, "associated_qrs_beat_id": 0},
            {"association_type": "blocked", "confidence": 0.95, "associated_qrs_beat_id": None},
        ]

        ratio = compute_organized_p_ratio(events, beat_count=4)

        self.assertEqual(0.0, ratio)

    def test_organized_p_ratio_zero_beats_is_zero(self) -> None:
        self.assertEqual(0.0, compute_organized_p_ratio([], beat_count=0))


class AtrialEventBoundaryJoinTests(unittest.TestCase):
    """`_attach_assessment_boundaries` joins p_events to p_wave_assessments."""

    FS = 500

    @staticmethod
    def _assessment(
        beat_id: int,
        *,
        accepted: bool = True,
        strict: tuple[int, int] = (571, 628),
        robust: tuple[int, int] = (575, 627),
    ) -> dict:
        return {
            "beat_id": beat_id,
            "accepted": accepted,
            "strict_onset": strict[0],
            "strict_offset": strict[1],
            "robust_onset": robust[0],
            "robust_offset": robust[1],
        }

    @staticmethod
    def _event(sample: int, *, beat_id: int, assoc: str = "conducted") -> dict:
        return {
            "sample": sample,
            "association_type": assoc,
            "associated_qrs_beat_id": beat_id,
            "onset_ms": None,
            "offset_ms": None,
            "duration_ms": None,
            "boundary_source": None,
        }

    @needs_interpretation
    def test_conducted_event_inside_the_envelope_gets_fused_boundaries(self) -> None:
        event = self._event(600, beat_id=1)

        _attach_assessment_boundaries([event], [self._assessment(1)], self.FS)

        self.assertEqual("p_wave_assessment_fusion", event["boundary_source"])
        self.assertAlmostEqual(1150.0, event["onset_ms"])
        self.assertAlmostEqual(1254.0, event["offset_ms"])
        # Robust bounds, not the wider strict envelope used for containment.
        self.assertAlmostEqual(104.0, event["duration_ms"])

    @needs_interpretation
    def test_event_sharing_a_beat_but_far_from_the_p_wave_is_not_joined(self) -> None:
        # Observed on 09017: the event is associated with the beat but sits
        # ~200 ms away, so it is a different deflection.
        event = self._event(700, beat_id=1)

        _attach_assessment_boundaries([event], [self._assessment(1)], self.FS)

        self.assertIsNone(event["boundary_source"])
        self.assertIsNone(event["onset_ms"])

    @needs_interpretation
    def test_small_boundary_rounding_still_joins(self) -> None:
        # Observed on 09009/09020: a few milliseconds outside the envelope.
        event = self._event(566, beat_id=1)

        _attach_assessment_boundaries([event], [self._assessment(1)], self.FS)

        self.assertEqual("p_wave_assessment_fusion", event["boundary_source"])

    @needs_interpretation
    def test_non_conducted_events_never_borrow_a_beats_boundaries(self) -> None:
        for assoc in ("retrograde", "blocked"):
            with self.subTest(assoc=assoc):
                event = self._event(600, beat_id=1, assoc=assoc)

                _attach_assessment_boundaries(
                    [event], [self._assessment(1)], self.FS
                )

                self.assertIsNone(event["boundary_source"])
                self.assertIsNone(event["onset_ms"])

    @needs_interpretation
    def test_rejected_assessment_leaves_the_event_bare(self) -> None:
        event = self._event(600, beat_id=1)

        _attach_assessment_boundaries(
            [event], [self._assessment(1, accepted=False)], self.FS
        )

        self.assertIsNone(event["boundary_source"])

    @needs_interpretation
    def test_existing_measured_boundaries_are_not_overwritten(self) -> None:
        event = self._event(600, beat_id=1)
        event["onset_ms"] = 1100.0
        event["offset_ms"] = 1200.0

        _attach_assessment_boundaries([event], [self._assessment(1)], self.FS)

        self.assertIsNone(event["boundary_source"])
        self.assertEqual(1100.0, event["onset_ms"])
