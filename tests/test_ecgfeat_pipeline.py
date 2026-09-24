from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from feature_extraction.ecgfeat.pipeline.policies import applicability as _applicability_mod
from feature_extraction.ecgfeat.pipeline.policies import pacing as _pacing_mod
from feature_extraction.ecgfeat.pipeline.policies import qt as _qt_mod
from feature_extraction.ecgfeat.pipeline.stages import beats as _beats_mod
from feature_extraction.ecgfeat.pipeline.stages import measurement as _measurement_mod
from feature_extraction.ecgfeat import delineate as delineate_mod
from feature_extraction.ecgfeat.delineate import (
    _apply_multilead_consensus,
    _apply_qrs_offset_raw_repair,
    _apply_st_j_raw_remeasurement,
    _apply_t_dual_method_raw_rescue,
    _apply_p_candidate_context,
    _apply_p_candidate_reselection,
    _measure_p_candidate_from_peak,
    _measure_st_j_with_guard,
    _p_context_corr,
    _p_context_robust_score,
    _p_context_snippet,
    _extend_p_candidate_edges,
    _p_offset_tangent,
    _p_should_reselect_candidate,
    _recompute_qrs_offset_dependent_fields,
    _qrs_bounds,
    _qrs_offset_consensus_target,
    _qrs_offset_repair_candidate,
    _qrs_tail_settling_candidate,
    _t_end_geometric,
    delineate_beats,
)
from feature_extraction.ecgfeat.features import (
    _filter_t_axis_values_for_reporting,
    _p_axis_values_with_polarity_consensus,
    _physiologic_pr_core_from_beats,
    _select_reliable_qt_leads,
    _stable_limb_signed_t_axis_deg,
    build_representative_lead_features,
    compute_global_features,
)
from feature_extraction.ecgfeat.models import BeatAnnotation, GlobalFeatures, LeadBeatFeatures, LeadQuality, RepresentativeLeadFeatures, STANDARD_12_LEADS, WaveBounds
from tests.optional import needs_interpretation


def _make_quality(
    reliable: bool = False,
    reliable_for_p: bool = False,
    reliable_for_qrs: bool = False,
    reliable_for_t: bool = False,
    reliable_for_qt: bool = False,
) -> LeadQuality:
    return LeadQuality(
        lead="",
        baseline_wander_score=0.0,
        muscle_noise_score=0.0,
        powerline_score=0.0,
        clipping_score=0.0,
        flatline_score=1.0,
        missing=False,
        reliable=reliable,
        reliable_for_p=reliable_for_p,
        reliable_for_qrs=reliable_for_qrs,
        reliable_for_t=reliable_for_t,
        reliable_for_qt=reliable_for_qt,
    )


def _make_rep(lead: str, **params: float | int | str | bool | None) -> RepresentativeLeadFeatures:
    base_params: dict[str, float | int | str | bool | None] = {
        "pr_ms": None,
        "pr_consensus_ms": None,
        "qrs_ms": None,
        "qt_ms": None,
        "qt_consensus_ms": None,
        "jt_ms": None,
        "p_amp_mv": None,
        "p_area": None,
        "p_signed_area": None,
        "q_amp_mv": None,
        "r_amp_mv": None,
        "s_amp_mv": None,
        "qrs_signed_area": None,
        "qrs_area": None,
        "st_on_mv": None,
        "st_mid_mv": None,
        "st_80ms_mv": None,
        "t_amp_mv": None,
        "t_area": None,
        "t_signed_area": None,
        "reliable_for_global": False,
        "reliable_for_p": False,
        "reliable_for_qrs": False,
        "reliable_for_t": False,
        "reliable_for_qt": False,
        "qt_confidence_mean": 0.0,
        "p_confidence_mean": 0.0,
        "p_onset_confidence_mean": None,
        "p_offset_confidence_mean": None,
        "qrs_confidence_mean": 0.0,
        "beat_count": 0,
        "representative_group_id": 1,
        "measurement_source": "test",
        "tpe_ms": None,
        "st_morphology": None,
        "fqrs_score": 0.0,
        "ptf_v1_mv_ms": None,
    }
    base_params.update(params)
    return RepresentativeLeadFeatures(
        lead=lead,
        params=base_params,
        variance={
            "pr_ms_sd": 5.0,
            "qrs_ms_sd": 5.0,
            "qt_ms_sd": 5.0,
            "jt_ms_sd": 5.0,
            "st_on_mv_sd": 0.0,
            "t_amp_mv_sd": 0.0,
        },
    )


def _make_wave(onset: int | None, peak: int | None, offset: int | None) -> WaveBounds:
    return WaveBounds(onset=onset, peak=peak, offset=offset)


def _make_beat_feature(
    lead: str,
    beat_id: int,
    pr_ms: float,
    qt_ms: float,
    *,
    qrs_ms: float = 90.0,
    pr_consensus_ms: float | None = None,
    qt_consensus_ms: float | None = None,
    p_amp_mv: float = 0.1,
    p_area: float = 1.0,
    p_signed_area: float | None = None,
    t_amp_mv: float = 0.3,
    t_area: float = 1.5,
    t_signed_area: float | None = None,
    t_polarity_observed: int | None = None,
    st_t_confusion: bool = False,
    t_confidence_reason: str | None = None,
) -> LeadBeatFeatures:
    feature = LeadBeatFeatures(
        lead=lead,
        beat_id=beat_id,
        p=_make_wave(10, 20, 30),
        qrs=_make_wave(40, 50, 60),
        t=_make_wave(70, 90, 120),
        qt_ms=qt_ms,
        pr_ms=pr_ms,
        qrs_ms=qrs_ms,
        p_amp_mv=p_amp_mv,
        qrs_area=1.0,
        q_amp_mv=-0.1,
        r_amp_mv=1.0,
        s_amp_mv=-0.2,
        st_on_mv=0.0,
        st_mid_mv=0.0,
        st_80ms_mv=0.0,
        t_amp_mv=t_amp_mv,
        j_index=60,
        p_confidence=0.9,
        qrs_confidence=0.9,
        qt_confidence=0.9,
        beat_measurement_reliable=True,
        p_area=p_area,
        p_signed_area=p_signed_area,
        t_area=t_area,
        t_signed_area=t_signed_area,
        pr_consensus_ms=pr_consensus_ms,
        qt_consensus_ms=qt_consensus_ms,
    )
    feature.t_polarity_observed = t_polarity_observed
    feature.st_t_confusion = st_t_confusion
    feature.t_confidence_reason = t_confidence_reason
    return feature


def _make_p_context_feature(
    lead: str,
    beat_id: int,
    *,
    p_on: int,
    p_peak: int,
    p_off: int,
    qrs_on: int = 200,
    p_confidence: float = 0.9,
    flags: list[str] | None = None,
) -> LeadBeatFeatures:
    feature = _make_beat_feature(
        lead,
        beat_id,
        pr_ms=float(qrs_on - p_on),
        qt_ms=360.0,
    )
    feature.p = WaveBounds(onset=p_on, peak=p_peak, offset=p_off)
    feature.qrs = WaveBounds(onset=qrs_on, peak=qrs_on + 20, offset=qrs_on + 50)
    feature.p_confidence = p_confidence
    feature.p_dur_ms = float(p_off - p_on)
    feature.flags = list(flags or [])
    return feature


class ECGFeaturePipelineTests(unittest.TestCase):
    def test_st_j_is_suppressed_when_j_point_is_on_qrs_tail_outlier(self) -> None:
        fs = 500
        sig = np.zeros(400, dtype=float)
        qrs_on = 100
        qrs_off = 205
        sig[qrs_on:qrs_off + 1] = np.linspace(0.0, 0.60, qrs_off - qrs_on + 1)
        sig[230:300] = np.hanning(70) * 0.15

        st_value, source, reliable = _measure_st_j_with_guard(
            sig=sig,
            qrs_on=qrs_on,
            qrs_off=qrs_off,
            baseline=0.0,
            fs=fs,
        )

        self.assertIsNone(st_value)
        self.assertEqual("qrs_tail_guard", source)
        self.assertFalse(reliable)

    def test_consensus_wide_qrs_suppresses_local_st_j_tail_outlier(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
            )
            for lead in ("I", "II", "III", "aVF")
        }
        features = [
            _make_beat_feature("I", 0, 160.0, 440.0, qrs_ms=186.0),
            _make_beat_feature("II", 0, 160.0, 440.0, qrs_ms=116.0),
            _make_beat_feature("III", 0, 160.0, 440.0, qrs_ms=186.0),
            _make_beat_feature("aVF", 0, 160.0, 440.0, qrs_ms=186.0),
        ]
        features[0].qrs = _make_wave(100, 150, 193)
        features[1].qrs = _make_wave(120, 150, 178)
        features[2].qrs = _make_wave(100, 150, 193)
        features[3].qrs = _make_wave(100, 150, 193)
        features[1].st_on_mv = 0.61
        features[1].st_mid_mv = 0.28
        features[1].st_80ms_mv = -0.03

        out = _apply_multilead_consensus(features, fs, quality)
        lead_ii = next(bf for bf in out if bf.lead == "II")

        self.assertIsNone(lead_ii.st_on_mv)
        self.assertIn("st_j_unreliable", lead_ii.flags)

    def test_multilead_qrs_consensus_uses_central_onset_when_early_leads_are_outliers(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        # Record 127-style wide-QRS morphology: aVR/V5/V6 vote much earlier
        # than the limb/precordial core and should not define measured QRS width.
        bounds_by_lead = {
            "I": (1365, 1422),
            "II": (1353, 1415),
            "III": (1378, 1410),
            "aVR": (1333, 1415),
            "aVL": (1359, 1400),
            "aVF": (1355, 1412),
            "V1": (1325, 1455),  # rejected from consensus: >250 ms
            "V2": (1355, 1424),
            "V3": (1362, 1403),
            "V4": (1370, 1414),
            "V5": (1335, 1449),
            "V6": (1337, 1418),
        }
        features = []
        for lead, (onset, offset) in bounds_by_lead.items():
            bf = _make_beat_feature(
                lead,
                0,
                pr_ms=160.0,
                qrs_ms=(offset - onset) * 1000.0 / fs,
                qt_ms=500.0,
            )
            bf.qrs = _make_wave(onset, (onset + offset) // 2, offset)
            bf.t = _make_wave(offset + 30, offset + 120, offset + 260)
            bf.qrs_confidence = 0.9
            bf.qt_confidence = 0.9
            features.append(bf)

        out = _apply_multilead_consensus(
            features,
            fs,
            quality,
            r_locs=np.asarray([1392, 2141], dtype=int),
        )

        consensus_values = {
            bf.qrs_consensus_ms
            for bf in out
            if bf.lead != "V1"
        }
        self.assertEqual(1, len(consensus_values))
        consensus_qrs = consensus_values.pop()
        self.assertIsNotNone(consensus_qrs)
        self.assertLessEqual(consensus_qrs, 145.0)

    def test_multilead_qrs_consensus_excludes_isolated_st_t_tail_offsets(self) -> None:
        fs = 1000
        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        late_tail_leads = {"V3", "V4", "V5", "V6"}
        features = []
        for lead in STANDARD_12_LEADS:
            offset = 290 if lead in late_tail_leads else 200
            bf = _make_beat_feature(
                lead,
                0,
                pr_ms=160.0,
                qrs_ms=float(offset - 100),
                qt_ms=430.0,
                st_t_confusion=lead in late_tail_leads,
            )
            bf.qrs = _make_wave(100, 140, offset)
            bf.t = _make_wave(offset + 30, offset + 100, offset + 180)
            bf.qrs_confidence = 0.9
            bf.qrs_off_confidence = 0.25 if lead in late_tail_leads else 0.9
            bf.qt_confidence = 0.9
            features.append(bf)

        out = _apply_multilead_consensus(features, fs, quality)

        consensus_values = {bf.qrs_consensus_ms for bf in out}
        self.assertEqual(1, len(consensus_values))
        consensus_qrs = consensus_values.pop()
        self.assertIsNotNone(consensus_qrs)
        self.assertLessEqual(consensus_qrs, 110.0)

    def test_multilead_qrs_consensus_keeps_supported_wide_terminal_offsets(self) -> None:
        fs = 1000
        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        late_terminal_leads = {"V3", "V4", "V5", "V6"}
        features = []
        for lead in STANDARD_12_LEADS:
            offset = 290 if lead in late_terminal_leads else 200
            bf = _make_beat_feature(
                lead,
                0,
                pr_ms=160.0,
                qrs_ms=float(offset - 100),
                qt_ms=430.0,
                st_t_confusion=False,
            )
            bf.qrs = _make_wave(100, 140, offset)
            bf.t = _make_wave(offset + 30, offset + 100, offset + 180)
            bf.qrs_confidence = 0.9
            bf.qrs_off_confidence = 0.85
            bf.qt_confidence = 0.9
            if lead in late_terminal_leads:
                bf.qrs_notch_count = 1
                bf.qrs_slur_flag = True
                bf.s_prime_amp_mv = -0.08
            features.append(bf)

        out = _apply_multilead_consensus(features, fs, quality)

        consensus_values = {bf.qrs_consensus_ms for bf in out}
        self.assertEqual(1, len(consensus_values))
        consensus_qrs = consensus_values.pop()
        self.assertIsNotNone(consensus_qrs)
        self.assertGreaterEqual(consensus_qrs, 180.0)

    def test_multilead_qrs_consensus_uses_median_when_terminal_support_is_sparse(self) -> None:
        fs = 1000
        reliable_leads = ("I", "II", "V1", "V2")
        quality = {
            lead: _make_quality(
                reliable=lead in reliable_leads,
                reliable_for_qrs=lead in reliable_leads,
                reliable_for_qt=lead in reliable_leads,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        offsets = {"I": 200, "II": 220, "V1": 240, "V2": 260}
        features = []
        for lead in reliable_leads:
            offset = offsets[lead]
            bf = _make_beat_feature(
                lead,
                0,
                pr_ms=160.0,
                qrs_ms=float(offset - 100),
                qt_ms=430.0,
            )
            bf.qrs = _make_wave(100, 140, offset)
            bf.t = _make_wave(offset + 30, offset + 100, offset + 180)
            bf.qrs_confidence = 0.9
            bf.qrs_off_confidence = 0.9
            bf.qt_confidence = 0.9
            features.append(bf)

        out = _apply_multilead_consensus(features, fs, quality)

        consensus_values = {bf.qrs_consensus_ms for bf in out}
        self.assertEqual(1, len(consensus_values))
        consensus_qrs = consensus_values.pop()
        self.assertIsNotNone(consensus_qrs)
        self.assertLessEqual(consensus_qrs, 135.0)

    def test_representative_lead_features_include_qrs_terminal_diagnostics(self) -> None:
        fs = 1000
        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        late_tail_leads = {"V3", "V4", "V5", "V6"}
        features = []
        for lead in STANDARD_12_LEADS:
            offset = 290 if lead in late_tail_leads else 200
            bf = _make_beat_feature(
                lead,
                0,
                pr_ms=160.0,
                qrs_ms=float(offset - 100),
                qt_ms=430.0,
                st_t_confusion=lead in late_tail_leads,
            )
            bf.qrs = _make_wave(100, 140, offset)
            bf.t = _make_wave(offset + 30, offset + 100, offset + 180)
            bf.qrs_confidence = 0.9
            bf.qrs_off_confidence = 0.25 if lead in late_tail_leads else 0.9
            bf.qt_confidence = 0.9
            features.append(bf)

        out = _apply_multilead_consensus(features, fs, quality)
        representatives = build_representative_lead_features(out, quality)

        lead_ii_params = representatives["II"].params
        lead_v5_params = representatives["V5"].params
        self.assertEqual("measurement", lead_ii_params["qrs_terminal_path"])
        self.assertEqual("classification_only", lead_v5_params["qrs_terminal_path"])
        self.assertEqual("st_t_tail", lead_v5_params["qrs_offset_exclusion_reason"])
        self.assertIn("II", lead_v5_params["qrs_offset_used_leads"])
        self.assertIn("V5", lead_v5_params["qrs_offset_excluded_leads"])
        self.assertEqual(0, lead_v5_params["qrs_late_cluster_support"])
        self.assertLess(lead_v5_params["qrs_terminal_confidence"], 0.5)

    def test_representative_t_values_use_pooled_consensus_when_template_t_is_tiny(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        pooled = []
        for beat_id, t_amp, t_area, t_signed_area in (
            (0, -0.027, 3.5, -3.2),
            (1, -0.022, 1.0, -0.3),
            (2, -0.030, 1.4, -0.4),
        ):
            bf = _make_beat_feature(
                "II",
                beat_id,
                pr_ms=160.0,
                qt_ms=430.0,
                t_amp_mv=t_amp,
                t_area=t_area,
                t_signed_area=t_signed_area,
            )
            bf.qt_confidence = 1.0
            pooled.append(bf)

        rep_feature = _make_beat_feature(
            "II",
            0,
            pr_ms=160.0,
            qt_ms=430.0,
            t_amp_mv=-0.006,
            t_area=0.8,
            t_signed_area=0.4,
        )
        rep_feature.qt_confidence = 1.0

        representatives = build_representative_lead_features(
            pooled,
            quality,
            representative_beat_features=[rep_feature],
            selected_beat_ids={0, 1, 2},
        )

        params = representatives["II"].params
        self.assertAlmostEqual(-0.027, params["t_amp_mv"])
        self.assertAlmostEqual(1.4, params["t_area"])
        self.assertAlmostEqual(-0.4, params["t_signed_area"])

    def test_extend_p_candidate_edges_keeps_physiologic_bounds(self) -> None:
        sig = np.zeros(220, dtype=float)
        sig[104:125] = np.linspace(0.01, 0.12, 21)
        sig[125:145] = np.linspace(0.11, 0.01, 20)
        extended_on, extended_off = _extend_p_candidate_edges(
            sig=sig,
            p_on=110,
            p_peak=124,
            p_off=140,
            baseline=0.0,
            search_lo=96,
            p_off_limit=179,
            qrs_on=180,
            fs=500,
        )
        self.assertEqual(104, extended_on)
        self.assertEqual(144, extended_off)

        sig_limited = np.zeros(240, dtype=float)
        sig_limited[124:146] = np.linspace(0.01, 0.12, 22)
        sig_limited[146:179] = np.linspace(0.11, 0.01, 33)
        _on, limited_off = _extend_p_candidate_edges(
            sig=sig_limited,
            p_on=130,
            p_peak=145,
            p_off=174,
            baseline=0.0,
            search_lo=100,
            p_off_limit=179,
            qrs_on=180,
            fs=500,
        )
        self.assertEqual(176, limited_off)

    def test_t_axis_filter_excludes_significant_t_prime_special_t_conflict(self) -> None:
        representatives = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representatives["aVR"] = _make_rep(
            "aVR",
            t_amp_mv=0.08,
            t_area=4.0,
            t_signed_area=3.5,
            twelve_sl_special_t_uv=-75.0,
            twelve_sl_t_prime_area_uv_ms=2200.0,
        )

        filtered = _filter_t_axis_values_for_reporting(
            representatives,
            {"aVR": 3.5},
        )

        self.assertIsNone(filtered["aVR"])

    def test_qrs_offset_repair_fields_default_to_none(self) -> None:
        feature = _make_beat_feature("II", 0, pr_ms=160.0, qt_ms=380.0)

        self.assertIsNone(feature.qrs_offset_original_index)
        self.assertIsNone(feature.qrs_offset_repaired_index)
        self.assertIsNone(feature.qrs_offset_repair_delta_ms)
        self.assertIsNone(feature.qrs_offset_repair_reason)
        self.assertIsNone(feature.qrs_offset_repair_confidence)
        self.assertIsNone(feature.qrs_offset_repair_consensus_index)
        self.assertIsNone(feature.qrs_offset_repair_support)

    def test_qrs_tail_settling_candidate_extends_slow_terminal_return(self) -> None:
        fs = 500
        signal = np.zeros(600, dtype=float)
        r = 250
        qrs_on = 225
        qrs_off = 270
        signal[qrs_on:r + 1] = np.linspace(0.0, 1.0, r - qrs_on + 1)
        signal[r + 1:260] = np.linspace(1.0, -0.5, 9)
        signal[260:305] = np.linspace(-0.5, 0.08, 45)
        signal[305:340] = 0.08
        smooth_window = max(3, int(round(0.008 * fs)))
        smoothed = np.convolve(
            signal,
            np.ones(smooth_window) / smooth_window,
            mode="same",
        )

        candidate = _qrs_tail_settling_candidate(
            smoothed,
            r=r,
            qrs_on=qrs_on,
            qrs_off=qrs_off,
            fs=fs,
        )

        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate, 295)
        self.assertLessEqual(candidate, 310)

    def test_qrs_tail_settling_candidate_keeps_already_settled_offset(self) -> None:
        fs = 500
        signal = np.zeros(600, dtype=float)
        r = 250
        qrs_on = 225
        qrs_off = 275
        signal[qrs_on:r + 1] = np.linspace(0.0, 1.0, r - qrs_on + 1)
        signal[r + 1:qrs_off + 1] = np.linspace(1.0, 0.05, qrs_off - r)
        signal[qrs_off + 1:340] = 0.05
        smooth_window = max(3, int(round(0.008 * fs)))
        smoothed = np.convolve(
            signal,
            np.ones(smooth_window) / smooth_window,
            mode="same",
        )

        candidate = _qrs_tail_settling_candidate(
            smoothed,
            r=r,
            qrs_on=qrs_on,
            qrs_off=qrs_off,
            fs=fs,
        )

        self.assertIsNone(candidate)

    def test_qrs_offset_consensus_target_uses_measurement_core_not_late_tail(self) -> None:
        fs = 500
        features: list[LeadBeatFeatures] = []
        offsets = {
            "I": 160,
            "II": 162,
            "III": 161,
            "aVR": 159,
            "V5": 205,
        }
        for lead, off in offsets.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=380.0)
            feature.qrs = WaveBounds(onset=100, peak=130, offset=off)
            feature.qrs_confidence = 0.9
            feature.qrs_off_confidence = 0.25 if lead == "V5" else 0.85
            feature.st_t_confusion = lead == "V5"
            features.append(feature)
        quality = {lead: _make_quality(reliable_for_qrs=True) for lead in offsets}

        target = _qrs_offset_consensus_target(features, fs=fs, quality=quality, paced=False)

        self.assertIsNotNone(target)
        self.assertEqual(161, target.offset)
        self.assertEqual(4, target.support)
        self.assertIn("V5", target.excluded_leads)
        self.assertIn("II", target.used_leads)

    def test_qrs_offset_consensus_target_returns_none_with_too_few_reliable_leads(self) -> None:
        fs = 500
        features: list[LeadBeatFeatures] = []
        for lead, off in {"I": 160, "II": 162, "V5": 205}.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=380.0)
            feature.qrs = WaveBounds(onset=100, peak=130, offset=off)
            feature.qrs_confidence = 0.9
            feature.qrs_off_confidence = 0.2
            features.append(feature)
        quality = {lead: _make_quality(reliable_for_qrs=True) for lead in ["I", "II", "V5"]}

        self.assertIsNone(_qrs_offset_consensus_target(features, fs=fs, quality=quality, paced=False))

    def test_qrs_offset_repair_candidate_accepts_late_low_confidence_tail(self) -> None:
        fs = 500
        sig = np.zeros(400, dtype=float)
        sig[100:161] = np.hanning(61) * 1.0
        sig[161:206] = np.linspace(0.04, 0.01, 45)
        feature = _make_beat_feature("V5", 0, pr_ms=160.0, qt_ms=380.0)
        feature.qrs = WaveBounds(onset=100, peak=130, offset=205)
        feature.qrs_off_confidence = 0.20
        feature.st_t_confusion = True
        target = SimpleNamespace(offset=160, support=5)

        repaired = _qrs_offset_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            paced=False,
        )

        self.assertIsNotNone(repaired)
        self.assertLessEqual(abs(repaired - 160), 13)

    def test_qrs_offset_repair_candidate_rejects_terminal_morphology(self) -> None:
        fs = 500
        sig = np.zeros(400, dtype=float)
        sig[100:206] = np.hanning(106) * 1.0
        feature = _make_beat_feature("V5", 0, pr_ms=160.0, qt_ms=380.0)
        feature.qrs = WaveBounds(onset=100, peak=130, offset=205)
        feature.qrs_off_confidence = 0.20
        feature.r_prime_amp_mv = 0.12
        target = SimpleNamespace(offset=160, support=5)

        self.assertIsNone(_qrs_offset_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            paced=False,
        ))

    def test_qrs_offset_repair_candidate_rejects_high_confidence_offset(self) -> None:
        fs = 500
        sig = np.zeros(400, dtype=float)
        sig[100:206] = np.hanning(106) * 1.0
        feature = _make_beat_feature("V5", 0, pr_ms=160.0, qt_ms=380.0)
        feature.qrs = WaveBounds(onset=100, peak=130, offset=205)
        feature.qrs_off_confidence = 0.85
        target = SimpleNamespace(offset=160, support=5)

        self.assertIsNone(_qrs_offset_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            paced=False,
        ))

    def test_qrs_offset_repair_candidate_rejects_excessive_shortening_of_plausible_qrs(self) -> None:
        fs = 500
        sig = np.zeros(300, dtype=float)
        sig[100:171] = np.hanning(71) * 1.0
        feature = _make_beat_feature("II", 0, pr_ms=160.0, qt_ms=380.0)
        feature.qrs = WaveBounds(onset=100, peak=130, offset=170)
        feature.qrs_off_confidence = 0.20
        target = SimpleNamespace(offset=135, support=5, excluded_leads="II")

        self.assertIsNone(_qrs_offset_repair_candidate(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
            target=target,
            paced=False,
        ))

    def test_recompute_qrs_offset_dependent_fields_updates_raw_intervals_and_st(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        sig[100:161] = np.hanning(61) * 1.0
        sig[161:260] = 0.05
        feature = _make_beat_feature("II", 0, pr_ms=160.0, qt_ms=380.0)
        feature.qrs = WaveBounds(onset=100, peak=130, offset=205)
        feature.t = WaveBounds(onset=250, peak=300, offset=360)
        feature.qrs_ms = 210.0
        feature.jt_ms = 310.0
        feature.qt_ms = 520.0
        feature.qrs.offset = 160

        _recompute_qrs_offset_dependent_fields(
            feature,
            sig=sig,
            beat_start=0,
            baseline=0.0,
            fs=fs,
        )

        self.assertAlmostEqual(120.0, feature.qrs_ms)
        self.assertEqual(160, feature.j_index)
        self.assertAlmostEqual(400.0, feature.jt_ms)
        self.assertAlmostEqual(520.0, feature.qt_ms)
        self.assertIsNotNone(feature.st_on_mv)
        self.assertIsNotNone(feature.qrs_area)
        self.assertIsNotNone(feature.qrs_signed_area)

    def test_qrs_offset_raw_repair_updates_only_late_low_confidence_outlier(self) -> None:
        fs = 500
        ecg = np.zeros((5, 500), dtype=float)
        leads = ["I", "II", "III", "aVR", "V5"]
        features: list[LeadBeatFeatures] = []
        for li, lead in enumerate(leads):
            ecg[li, 100:161] = np.hanning(61) * 1.0
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=380.0)
            off = 205 if lead == "V5" else 160
            feature.qrs = WaveBounds(onset=100, peak=130, offset=off)
            feature.t = WaveBounds(onset=240, peak=300, offset=360)
            feature.qrs_confidence = 0.9
            feature.qrs_off_confidence = 0.20 if lead == "V5" else 0.85
            feature.st_t_confusion = lead == "V5"
            features.append(feature)
        quality = {lead: _make_quality(reliable_for_qrs=True, reliable_for_qt=True) for lead in leads}

        out = _apply_qrs_offset_raw_repair(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            r_locs=np.asarray([130], dtype=int),
        )

        repaired = next(feature for feature in out if feature.lead == "V5")
        untouched = [feature for feature in out if feature.lead != "V5"]
        self.assertEqual(160, repaired.qrs.offset)
        self.assertIn("qrs_offset_repaired_by_consensus", repaired.flags)
        self.assertEqual(205, repaired.qrs_offset_original_index)
        self.assertEqual(160, repaired.qrs_offset_repaired_index)
        self.assertAlmostEqual(-90.0, repaired.qrs_offset_repair_delta_ms)
        self.assertEqual(160, repaired.qrs_offset_repair_consensus_index)
        self.assertEqual(4, repaired.qrs_offset_repair_support)
        self.assertTrue(all(feature.qrs.offset == 160 for feature in untouched))
        self.assertTrue(all("qrs_offset_repaired_by_consensus" not in feature.flags for feature in untouched))

    def test_qrs_offset_raw_repair_skips_paced_beat(self) -> None:
        fs = 500
        ecg = np.zeros((4, 500), dtype=float)
        leads = ["I", "II", "III", "V5"]
        features: list[LeadBeatFeatures] = []
        for li, lead in enumerate(leads):
            ecg[li, 100:161] = np.hanning(61) * 1.0
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=380.0)
            feature.qrs = WaveBounds(onset=100, peak=130, offset=205 if lead == "V5" else 160)
            feature.qrs_confidence = 0.9
            feature.qrs_off_confidence = 0.20
            feature.flags.append("paced_beat")
            features.append(feature)
        quality = {lead: _make_quality(reliable_for_qrs=True) for lead in leads}

        out = _apply_qrs_offset_raw_repair(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            r_locs=np.asarray([130], dtype=int),
        )

        self.assertEqual(205, next(feature for feature in out if feature.lead == "V5").qrs.offset)
        self.assertTrue(all("qrs_offset_repaired_by_consensus" not in feature.flags for feature in out))

    def test_delineate_beats_repairs_late_qrs_offset_before_consensus(self) -> None:
        fs = 500
        leads = ["I", "II", "III", "aVR", "V5"]
        ecg = np.zeros((len(leads), 700), dtype=float)
        r_locs = np.asarray([300], dtype=int)
        for li, lead in enumerate(leads):
            ecg[li, 260:321] = np.hanning(61) * 1.0
            if lead == "V5":
                ecg[li, 321:366] = np.linspace(0.04, 0.01, 45)
        quality = {lead: _make_quality(reliable_for_qrs=True, reliable_for_qt=True) for lead in leads}

        def fake_qrs_bounds(_sig, _local_r, _fs, *_args, **_kwargs):
            off = 240 if float(np.max(_sig[196:241])) > 0.03 else 195
            return 135, off, 0, 0.90, 0.20 if off == 240 else 0.90

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", leads), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", side_effect=fake_qrs_bounds), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", return_value=None), \
             patch("feature_extraction.ecgfeat.delineate.detect_t_wave", return_value=SimpleNamespace(
                 peak=None,
                 offset=None,
                 confidence=0.0,
                 method=None,
                 polarity_expected=None,
                 polarity_observed=None,
                 st_t_confusion=False,
                 confidence_reason=None,
             )):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        v5 = next(feature for feature in features if feature.lead == "V5")
        self.assertIn("qrs_offset_repaired_by_consensus", v5.flags)
        self.assertLessEqual(abs(int(v5.qrs.offset) - 320), 13)
        self.assertAlmostEqual((int(v5.qrs.offset) - int(v5.qrs.onset)) * 1000.0 / fs, v5.qrs_ms)
        self.assertEqual(v5.qrs.offset, v5.j_index)

    def test_delineate_beats_remeasures_st_j_after_qrs_raw_repair_pass(self) -> None:
        fs = 500
        leads = ["I", "II", "III", "aVR", "V5"]
        ecg = np.zeros((len(leads), 700), dtype=float)
        r_locs = np.asarray([300], dtype=int)
        for li, lead in enumerate(leads):
            ecg[li, 260:321] = np.hanning(61) * 1.0
            ecg[li, 321:430] = 0.02
            if lead == "V5":
                ecg[li, 365] = 0.55
                ecg[li, 385] = 0.03
                ecg[li, 405] = 0.02
        quality = {lead: _make_quality(reliable_for_qrs=True, reliable_for_qt=True) for lead in leads}

        def fake_qrs_bounds(_sig, _local_r, _fs, *_args, **_kwargs):
            off = 240 if float(np.max(_sig[196:242])) > 0.50 else 195
            return 135, off, 0, 0.90, 0.90

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", leads), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", side_effect=fake_qrs_bounds), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", return_value=None), \
             patch("feature_extraction.ecgfeat.delineate.detect_t_wave", return_value=SimpleNamespace(
                 peak=None,
                 offset=None,
                 confidence=0.0,
                 method=None,
                 polarity_expected=None,
                 polarity_observed=None,
                 st_t_confusion=False,
                 confidence_reason=None,
             )):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        v5 = next(feature for feature in features if feature.lead == "V5")
        self.assertNotIn("qrs_offset_repaired_by_consensus", v5.flags)
        self.assertIn("st_j_remeasured_by_consensus", v5.flags)
        self.assertEqual(v5.qrs.offset, v5.j_index)
        self.assertIsNotNone(v5.st_j_remeasured_index)
        self.assertLess(v5.st_j_remeasured_index, v5.j_index)
        self.assertLess(v5.st_on_mv or 0.0, 0.10)

    def test_delineate_beats_repairs_early_t_offset_before_consensus(self) -> None:
        fs = 500
        leads = ["I", "II", "III", "aVR", "V6"]
        ecg = np.zeros((len(leads), 700), dtype=float)
        r_locs = np.asarray([300], dtype=int)
        for li, _lead in enumerate(leads):
            ecg[li, 260:311] = np.hanning(51) * 1.0
            ecg[li, 445:471] = np.linspace(0.0, 0.35, 26)
            ecg[li, 471:521] = np.linspace(0.35, 0.0, 50)
        quality = {lead: _make_quality(reliable_for_qrs=True, reliable_for_qt=True) for lead in leads}

        def fake_qrs_bounds(_sig, _local_r, _fs, *_args, **_kwargs):
            return 135, 185, 0, 0.90, 0.90

        def fake_detect_t_wave(*_args, **kwargs):
            lead = kwargs["lead"]
            if lead == "V6":
                return SimpleNamespace(
                    peak=346,
                    offset=396,
                    confidence=0.15,
                    method="threshold_fallback",
                    polarity_expected=1,
                    polarity_observed=1,
                    st_t_confusion=False,
                    confidence_reason=None,
                )
            return SimpleNamespace(
                peak=346,
                offset=396,
                confidence=0.80,
                method="polarity_cluster",
                polarity_expected=1,
                polarity_observed=1,
                st_t_confusion=False,
                confidence_reason=None,
            )

        initial_geom_calls = 0

        def fake_t_end_geometric(_sig, _t_peak_local, search_end_local, _baseline, _fs):
            nonlocal initial_geom_calls
            if search_end_local > 430:
                initial_geom_calls += 1
                if initial_geom_calls == len(leads):
                    return 375, 0.15, "threshold_fallback"
                return 396, 0.80, "threshold_reference"
            return 396, 0.80, "dxl_chord"

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", leads), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", side_effect=fake_qrs_bounds), \
             patch("feature_extraction.ecgfeat.delineate.detect_t_wave", side_effect=fake_detect_t_wave), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", side_effect=fake_t_end_geometric):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        v6 = next(feature for feature in features if feature.lead == "V6")
        self.assertIn("t_offset_repaired_by_consensus", v6.flags)
        self.assertIn("t_offset_early_truncation_repaired", v6.flags)
        self.assertGreater(v6.t.offset, 500)
        self.assertAlmostEqual((int(v6.t.offset) - int(v6.qrs.onset)) * 1000.0 / fs, v6.qt_ms)

    def test_delineate_beats_dual_rescues_early_t_offset_before_consensus_repair(self) -> None:
        fs = 500
        leads = ["I", "II", "III", "aVR", "V6"]
        ecg = np.zeros((len(leads), 700), dtype=float)
        r_locs = np.asarray([300], dtype=int)
        for li, _lead in enumerate(leads):
            ecg[li, 260:311] = np.hanning(51) * 1.0
            ecg[li, 430:461] = np.linspace(0.0, 0.45, 31)
            ecg[li, 461:531] = np.linspace(0.45, 0.0, 70)
        quality = {lead: _make_quality(reliable_for_qrs=True, reliable_for_qt=True) for lead in leads}

        def fake_qrs_bounds(_sig, _local_r, _fs, *_args, **_kwargs):
            return 135, 185, 0, 0.90, 0.90

        def fake_detect_t_wave(*_args, **kwargs):
            lead = kwargs["lead"]
            if lead == "V6":
                return SimpleNamespace(
                    peak=346,
                    offset=351,
                    confidence=0.15,
                    method="threshold_fallback",
                    polarity_expected=1,
                    polarity_observed=1,
                    st_t_confusion=False,
                    confidence_reason=None,
                )
            return SimpleNamespace(
                peak=346,
                offset=406,
                confidence=0.80,
                method="polarity_cluster",
                polarity_expected=1,
                polarity_observed=1,
                st_t_confusion=False,
                confidence_reason=None,
            )

        geom_calls = 0

        def fake_t_end_geometric(_sig, _t_peak_local, _search_end_local, _baseline, _fs):
            nonlocal geom_calls
            geom_calls += 1
            if geom_calls == len(leads):
                return 351, 0.15, "threshold_fallback"
            if geom_calls < len(leads):
                return 406, 0.80, "dxl_chord"
            return 406, 0.80, "dxl_chord"

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", leads), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", side_effect=fake_qrs_bounds), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", return_value=None), \
             patch("feature_extraction.ecgfeat.delineate.detect_t_wave", side_effect=fake_detect_t_wave), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", side_effect=fake_t_end_geometric):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        v6 = next(feature for feature in features if feature.lead == "V6")
        self.assertIn("t_offset_dual_method_rescued", v6.flags)
        self.assertGreater(v6.t.offset, 500)
        self.assertIsNotNone(v6.t_offset_dual_slope_index)
        self.assertAlmostEqual((int(v6.t.offset) - int(v6.qrs.onset)) * 1000.0 / fs, v6.qt_ms)

    def test_delineate_beats_passes_t_prime_peak_to_twelve_sl_profile(self) -> None:
        fs = 500
        leads = ["II"]
        ecg = np.zeros((len(leads), 700), dtype=float)
        r_locs = np.asarray([300], dtype=int)
        quality = {lead: _make_quality(reliable_for_qrs=True, reliable_for_qt=True) for lead in leads}
        seen_t_prime_peaks: list[int | None] = []

        def fake_qrs_bounds(_sig, _local_r, _fs, *_args, **_kwargs):
            return 140, 190, 0, 0.90, 0.90

        def fake_detect_t_wave(*_args, **_kwargs):
            return SimpleNamespace(
                peak=260,
                offset=340,
                confidence=0.80,
                method="polarity_cluster",
                polarity_expected=1,
                polarity_observed=1,
                st_t_confusion=False,
                confidence_reason=None,
                flags=[],
                t_prime_peak=320,
            )

        def fake_twelve_sl_wave_measurements_from_signal(**kwargs):
            seen_t_prime_peaks.append(kwargs.get("t_prime_peak"))
            return {
                "stj_mv": 0.01,
                "stm_mv": 0.02,
                "ste_mv": 0.03,
                "stm_offset_ms": 62.5,
                "ste_offset_ms": 125.0,
                "qrs_area_uv_ms": 200.0,
                "qrs_signed_area_uv_ms": 100.0,
                "qrs_balance_uv": 50.0,
                "qrs_deflection_uv": 150.0,
                "minimum_st_uv": 10.0,
                "special_t_uv": -90.0,
                "t_prime_amp_uv": -90.0,
                "t_prime_area_uv_ms": 220.0,
                "qrs_significant": True,
            }

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", leads), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", side_effect=fake_qrs_bounds), \
             patch("feature_extraction.ecgfeat.delineate.detect_t_wave", side_effect=fake_detect_t_wave), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(340, 0.80, "dxl_chord")), \
             patch("feature_extraction.ecgfeat.delineate.twelve_sl_wave_measurements_from_signal", side_effect=fake_twelve_sl_wave_measurements_from_signal):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        self.assertEqual([320], seen_t_prime_peaks)
        lead_ii = next(feature for feature in features if feature.lead == "II")
        self.assertAlmostEqual(-90.0, lead_ii.twelve_sl_t_prime_uv)
        self.assertAlmostEqual(220.0, lead_ii.twelve_sl_t_prime_area_uv_ms)
        self.assertAlmostEqual(-90.0, lead_ii.twelve_sl_special_t_uv)

    def test_t_end_geometric_returns_measurement_for_descending_arm(self) -> None:
        sig = np.zeros(100, dtype=float)
        sig[20:61] = np.linspace(1.0, 0.2, 41)

        t_end, confidence, method = _t_end_geometric(
            sig,
            t_peak_local=20,
            search_end_local=60,
            baseline=0.0,
            fs=100,
        )

        self.assertEqual("dxl_chord", method)
        self.assertIsNotNone(t_end)
        self.assertGreaterEqual(t_end, 20)
        self.assertGreaterEqual(confidence, 0.0)

    def test_t_end_geometric_returns_low_confidence_for_flat_chord_deviation(self) -> None:
        sig = np.zeros(100, dtype=float)
        sig[20:61] = np.linspace(1.0, 0.2, 41)

        t_end, confidence, method = _t_end_geometric(
            sig,
            t_peak_local=20,
            search_end_local=60,
            baseline=0.0,
            fs=100,
        )

        self.assertEqual("dxl_chord", method)
        self.assertEqual(23, t_end)
        self.assertEqual(0.0, confidence)

    def test_p_offset_tangent_keeps_result_within_search_window(self) -> None:
        sig = np.zeros(100, dtype=float)
        sig[20:61] = np.linspace(0.8, 0.2, 41)

        p_off = _p_offset_tangent(
            sig,
            p_peak_local=20,
            search_end_local=60,
            baseline=0.0,
            fs=100,
        )

        self.assertIsNotNone(p_off)
        self.assertLessEqual(p_off, 60)

    def test_qrs_bounds_uses_nearby_energy_peak_when_fiducial_r_is_low_slope(self) -> None:
        fs = 500
        sig = np.zeros(400, dtype=float)
        sig[120:150] = np.linspace(0.0, 1.0, 30)
        sig[150:170] = 1.0
        sig[170:200] = np.linspace(1.0, 0.0, 30)

        qrs_on, qrs_off, _notch, _on_conf, _off_conf = _qrs_bounds(
            sig,
            r=160,
            fs=fs,
            enable_low_slope_guard=True,
        )

        self.assertGreaterEqual(qrs_on, 105)
        self.assertLessEqual(qrs_on, 130)
        self.assertGreaterEqual(qrs_off, 190)
        self.assertLessEqual(qrs_off, 215)

    def test_qrs_bounds_recovers_sustained_low_slope_qrs_foot(self) -> None:
        fs = 500
        true_onset = 150
        sig = np.zeros(500, dtype=float)
        sig[true_onset:180] = np.linspace(0.0, 0.08, 30, endpoint=False)
        sig[180:192] = np.linspace(0.08, 1.0, 12, endpoint=False)
        sig[192:218] = np.linspace(1.0, 0.0, 26)

        qrs_on, qrs_off, _notch, _on_conf, _off_conf = _qrs_bounds(
            sig,
            r=192,
            fs=fs,
        )

        self.assertGreaterEqual(qrs_on, true_onset - int(0.010 * fs))
        self.assertLessEqual(qrs_on, true_onset + int(0.014 * fs))
        self.assertGreaterEqual(qrs_off, 210)

    def test_delineate_beats_prefers_delayed_broad_low_amplitude_t_over_early_st_bump(self) -> None:
        fs = 500
        sig = np.zeros(800, dtype=float)
        sig[190:201] = np.linspace(0.0, 1.0, 11)
        sig[201:215] = np.linspace(1.0, 0.0, 14)
        sig[275:295] += np.hanning(20) * 0.08
        sig[330:460] += np.hanning(130) * 0.055
        ecg = np.vstack([sig for _ in STANDARD_12_LEADS])
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        beat_features = delineate_beats(ecg, fs, np.asarray([200]), quality=quality)
        lead_ii = next(bf for bf in beat_features if bf.lead == "II")

        self.assertIsNotNone(lead_ii.t.peak)
        self.assertGreaterEqual(lead_ii.t.peak, 360)
        self.assertGreaterEqual(lead_ii.qt_ms or 0.0, 330.0)

    def test_delineate_beats_clamps_t_end_to_tu_nadir_before_late_u_wave(self) -> None:
        fs = 500
        sig = np.zeros(900, dtype=float)
        sig[190:201] = np.linspace(0.0, 1.0, 11)
        sig[201:215] = np.linspace(1.0, 0.0, 14)
        sig[250:381] += np.hanning(131) * 0.08
        sig[420:520] += np.hanning(100) * 0.18
        ecg = np.vstack([sig for _ in STANDARD_12_LEADS])
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        beat_features = delineate_beats(ecg, fs, np.asarray([200]), quality=quality)
        lead_ii = next(bf for bf in beat_features if bf.lead == "II")

        self.assertIsNotNone(lead_ii.t.offset)
        self.assertLessEqual(lead_ii.t.offset, 405)
        self.assertTrue(lead_ii.u_wave_flag)

    def test_delineate_beats_extends_t_end_across_compact_biphasic_t_wave(self) -> None:
        fs = 500
        sig = np.zeros(900, dtype=float)
        sig[190:201] = np.linspace(0.0, 1.0, 11)
        sig[201:215] = np.linspace(1.0, 0.0, 14)
        sig[250:340] -= np.hanning(90) * 0.08
        sig[330:440] += np.hanning(110) * 0.055
        ecg = np.vstack([sig for _ in STANDARD_12_LEADS])
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        beat_features = delineate_beats(ecg, fs, np.asarray([200]), quality=quality)
        lead_ii = next(bf for bf in beat_features if bf.lead == "II")

        self.assertIsNotNone(lead_ii.t.offset)
        self.assertGreaterEqual(lead_ii.t.offset, 415)
        self.assertGreaterEqual(lead_ii.qt_ms or 0.0, 330.0)

    def test_lead_beat_features_expose_baseline_provenance(self) -> None:
        bf = LeadBeatFeatures(
            lead="II",
            beat_id=0,
            p=_make_wave(10, 20, 30),
            qrs=_make_wave(40, 50, 60),
            t=_make_wave(70, 90, 120),
            qt_ms=360.0,
            pr_ms=120.0,
            qrs_ms=90.0,
            p_amp_mv=0.1,
            qrs_area=1.0,
            q_amp_mv=-0.1,
            r_amp_mv=1.0,
            s_amp_mv=-0.2,
            st_on_mv=0.0,
            st_mid_mv=0.0,
            st_80ms_mv=0.0,
            t_amp_mv=0.3,
            j_index=60,
            baseline_source="pr_segment",
            baseline_confidence=0.8,
        )

        self.assertEqual("pr_segment", bf.baseline_source)
        self.assertEqual(0.8, bf.baseline_confidence)

    def test_delineate_beats_populates_baseline_provenance_and_signed_areas(self) -> None:
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))

        p_slice = slice(beat_start + 120, beat_start + 141)
        qrs_slice = slice(beat_start + 170, beat_start + 181)
        t_slice = slice(beat_start + 280, beat_start + 331)
        ecg[0, p_slice] = 0.2
        ecg[0, qrs_slice] = np.asarray([0.0, 0.1, 0.4, 0.8, 0.4, 0.1, 0.0, -0.05, -0.1, -0.05, 0.0])
        ecg[0, t_slice] = 0.3

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(170, 180, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", side_effect=[130, 300]), \
             patch("feature_extraction.ecgfeat.delineate._find_wave_bounds", return_value=(120, 140)), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=280), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(330, 0.7, "laguna_tangent")):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs)

        self.assertEqual(1, len(features))
        feature = features[0]
        self.assertEqual("pr_segment", feature.baseline_source)
        self.assertEqual(0.8, feature.baseline_confidence)
        self.assertGreater(feature.qrs_signed_area, 0.0)
        self.assertGreater(feature.p_signed_area, 0.0)
        self.assertGreater(feature.t_signed_area, 0.0)

    def test_delineate_beats_does_not_floor_narrow_false_paced_qrs(self) -> None:
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))

        ecg[0, beat_start + 170 : beat_start + 211] = np.hanning(41)
        ecg[0, beat_start + 300 : beat_start + 340] = np.hanning(40) * 0.2

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(170, 210, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", return_value=320), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=300), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(340, 0.7, "laguna_tangent")):
            features = delineate_beats(
                ecg,
                fs=fs,
                r_locs=r_locs,
                paced_beat_ids=[0],
            )

        feature = features[0]
        self.assertAlmostEqual(80.0, feature.qrs_ms)
        self.assertNotIn("paced_floor_applied", feature.flags)

    def test_delineate_beats_keeps_paced_floor_for_wide_qrs_candidate(self) -> None:
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))

        ecg[0, beat_start + 170 : beat_start + 227] = np.hanning(57)
        ecg[0, beat_start + 310 : beat_start + 350] = np.hanning(40) * 0.2

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(170, 226, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", return_value=330), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=310), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(350, 0.7, "laguna_tangent")):
            features = delineate_beats(
                ecg,
                fs=fs,
                r_locs=r_locs,
                paced_beat_ids=[0],
            )

        feature = features[0]
        self.assertAlmostEqual(120.0, feature.qrs_ms)
        self.assertIn("paced_floor_applied", feature.flags)

    def test_delineate_beats_allows_confirmed_paced_floor_for_narrow_initial_bounds(self) -> None:
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))

        ecg[0, beat_start + 170 : beat_start + 211] = np.hanning(41)
        ecg[0, beat_start + 300 : beat_start + 340] = np.hanning(40) * 0.2

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(170, 210, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", return_value=320), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=300), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(340, 0.7, "laguna_tangent")):
            features = delineate_beats(
                ecg,
                fs=fs,
                r_locs=r_locs,
                paced_beat_ids=[0],
                paced_qrs_floor_beat_ids=[0],
            )

        feature = features[0]
        self.assertAlmostEqual(120.0, feature.qrs_ms)
        self.assertIn("paced_floor_applied", feature.flags)

    def test_delineate_beats_keeps_long_pr_p_onset_when_p_wave_is_visible(self) -> None:
        fs = 500
        sig = np.zeros((12, 1200), dtype=float)
        r = 600
        for lead_idx in range(12):
            sig[lead_idx, r - 135:r - 115] = np.hanning(20) * 0.08
            sig[lead_idx, r - 25:r + 25] = np.r_[np.linspace(0, 1.0, 25), np.linspace(1.0, 0, 25)]
            sig[lead_idx, r + 120:r + 200] = np.hanning(80) * 0.20

        features = delineate_beats(sig, fs, np.asarray([r], dtype=int))
        lead_ii = next(bf for bf in features if bf.lead == "II")

        self.assertIsNotNone(lead_ii.pr_ms)
        self.assertIsNotNone(lead_ii.p.onset)
        self.assertLessEqual(abs(int(lead_ii.p.onset) - (r - 135)), 3)
        self.assertGreaterEqual(lead_ii.pr_ms, 155.0)
        self.assertLessEqual(lead_ii.pr_ms, 175.0)

    def test_delineate_beats_prefers_fused_p_peak_over_short_pr_abs_peak(self) -> None:
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))
        fused_p_peak_local = 113
        false_p_peak_local = 132
        fused_p_peak_global = beat_start + fused_p_peak_local

        ecg[0, fused_p_peak_global - 4 : fused_p_peak_global + 5] += np.hanning(9) * 0.08
        false_p_peak_global = beat_start + false_p_peak_local
        ecg[0, false_p_peak_global - 2 : false_p_peak_global + 3] += np.hanning(5) * 0.12
        ecg[0, 190:211] += np.hanning(21) * 1.0
        ecg[0, 270:330] += np.hanning(60) * 0.2

        def fake_find_peak(_sig: np.ndarray, lo: int, hi: int, mode: str = "abs") -> int | None:
            if lo <= false_p_peak_local < hi:
                return false_p_peak_local
            return None

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._fuse_peak_anchor",
                   side_effect=[(fused_p_peak_local, 1), (None, 0)]), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", side_effect=fake_find_peak), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(160, 195, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._detect_delta_wave", return_value=False):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs)

        feature = features[0]
        self.assertEqual(fused_p_peak_global, feature.p.peak)
        self.assertGreaterEqual(feature.pr_ms or 0.0, 90.0)

    def test_delineate_beats_fuses_p_peak_by_multilead_support_not_outlier_amplitude(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1000), dtype=float)
        r = 600
        beat_start = max(0, int(r - 0.35 * fs))
        true_p = r - 140
        false_p = r - 60
        qrs_on_local = r - beat_start - 20
        qrs_off_local = r - beat_start + 20

        for lead_idx in range(12):
            ecg[lead_idx, r - 12 : r + 13] += np.hanning(25) * 1.0
            ecg[lead_idx, r + 120 : r + 201] += np.hanning(81) * 0.2

        for lead_name in ["I", "II", "III", "aVL", "aVF", "V1"]:
            lead_idx = STANDARD_12_LEADS.index(lead_name)
            ecg[lead_idx, true_p - 10 : true_p + 11] += np.hanning(21) * 0.045

        for lead_name in ["V5", "V6"]:
            lead_idx = STANDARD_12_LEADS.index(lead_name)
            ecg[lead_idx, false_p - 4 : false_p + 5] += np.hanning(9) * 0.45

        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        with patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(qrs_on_local, qrs_off_local, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._detect_delta_wave", return_value=False):
            features = delineate_beats(ecg, fs=fs, r_locs=np.asarray([r], dtype=int), quality=quality)

        lead_ii = next(bf for bf in features if bf.lead == "II")
        self.assertIsNotNone(lead_ii.p.peak)
        self.assertLessEqual(abs(int(lead_ii.p.peak) - true_p), int(0.020 * fs))
        self.assertGreaterEqual(lead_ii.pr_ms or 0.0, 120.0)

    def test_delineate_beats_rejects_broad_short_qrs_adjacent_p_peak_cluster(self) -> None:
        fs = 500
        ecg = np.zeros((12, 1000), dtype=float)
        r = 600
        beat_start = max(0, int(r - 0.35 * fs))
        true_p = r - 150
        # The artifact has to sit where the P-peak/QRS-gap floor actually
        # calls it implausible. It used to be at r - 48, but that is ~76 ms
        # ahead of this beat's QRS reference, and LUDB's 16790 expert P
        # annotations put 15% of *genuine* P waves inside 80 ms (p1 52 ms,
        # p5 66 ms) -- so the old position is squarely inside the real P
        # distribution and rejecting it there costs real detections. The
        # property under test is unchanged: a narrow, higher-amplitude,
        # many-lead spike adjacent to the QRS must still lose to the broader
        # four-lead P further out.
        false_p = r - 35
        qrs_on_local = r - beat_start - 20
        qrs_off_local = r - beat_start + 20

        for lead_idx in range(12):
            ecg[lead_idx, r - 12 : r + 13] += np.hanning(25) * 1.0
            ecg[lead_idx, r + 120 : r + 201] += np.hanning(81) * 0.2

        for lead_name in ["I", "II", "aVF", "V1"]:
            lead_idx = STANDARD_12_LEADS.index(lead_name)
            ecg[lead_idx, true_p - 10 : true_p + 11] += np.hanning(21) * 0.055

        for lead_name in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4"]:
            lead_idx = STANDARD_12_LEADS.index(lead_name)
            ecg[lead_idx, false_p - 4 : false_p + 5] += np.hanning(9) * 0.09

        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        with patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(qrs_on_local, qrs_off_local, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._detect_delta_wave", return_value=False):
            features = delineate_beats(ecg, fs=fs, r_locs=np.asarray([r], dtype=int), quality=quality)

        lead_ii = next(bf for bf in features if bf.lead == "II")
        self.assertIsNotNone(lead_ii.p.peak)
        self.assertLessEqual(abs(int(lead_ii.p.peak) - true_p), int(0.020 * fs))
        self.assertGreaterEqual(lead_ii.pr_ms or 0.0, 125.0)

    def test_delineate_beats_computes_pr_for_wide_non_delta_qrs(self) -> None:
        # Wide QRS with P-peak clearly before QRS onset should still produce a
        # valid PR measurement; the geometry check only rejects pathological cases
        # (p_on >= p_peak, or p_peak >= qrs_on).
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))

        ecg[0, beat_start + 76:beat_start + 85] = np.hanning(9) * 0.08
        ecg[0, beat_start + 120:beat_start + 206] = np.hanning(86)
        ecg[0, beat_start + 300:beat_start + 340] = np.hanning(40) * 0.2

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(120, 205, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", side_effect=[80, 320]), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", return_value=60), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", return_value=100), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=280), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(340, 0.7, "dxl_chord")), \
             patch("feature_extraction.ecgfeat.delineate._detect_delta_wave", return_value=False):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs)

        feature = features[0]
        self.assertIsNotNone(feature.pr_ms)
        self.assertIsNotNone(feature.p.peak)
        self.assertNotIn("p_unreliable", feature.flags)

    def test_delineate_beats_preserves_short_pr_when_delta_wave_present(self) -> None:
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))

        ecg[0, beat_start + 76:beat_start + 85] = np.hanning(9) * 0.08
        ecg[0, beat_start + 120:beat_start + 206] = np.hanning(86)
        ecg[0, beat_start + 300:beat_start + 340] = np.hanning(40) * 0.2

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(120, 205, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", side_effect=[80, 320]), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", return_value=60), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", return_value=100), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=280), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(340, 0.7, "dxl_chord")), \
             patch("feature_extraction.ecgfeat.delineate._detect_delta_wave", return_value=True):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs)

        feature = features[0]
        self.assertIsNotNone(feature.pr_ms)
        self.assertAlmostEqual(120.0, feature.pr_ms)
        self.assertIsNotNone(feature.p.peak)
        self.assertNotIn("p_unreliable", feature.flags)

    def test_compute_global_features_uses_limb_leads_for_pr(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["I"] = _make_rep(
            "I",
            pr_ms=118.0,
            pr_consensus_ms=166.0,
            qrs_ms=88.0,
            qt_ms=350.0,
            qt_consensus_ms=420.0,
            qrs_area=1.0,
            t_amp_mv=0.2,
            reliable_for_global=True,
            reliable_for_p=True,
            reliable_for_qrs=True,
            reliable_for_t=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
            p_confidence_mean=0.8,
            qrs_confidence_mean=0.8,
        )
        representative_leads["II"] = _make_rep(
            "II",
            pr_ms=122.0,
            pr_consensus_ms=170.0,
            qrs_ms=90.0,
            qt_ms=360.0,
            qt_consensus_ms=430.0,
            qrs_area=1.1,
            t_amp_mv=0.3,
            reliable_for_global=True,
            reliable_for_p=True,
            reliable_for_qrs=True,
            reliable_for_t=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.9,
            p_confidence_mean=0.9,
            qrs_confidence_mean=0.9,
        )
        representative_leads["V5"] = _make_rep(
            "V5",
            pr_ms=126.0,
            pr_consensus_ms=174.0,
            qrs_ms=92.0,
            qt_ms=370.0,
            qt_consensus_ms=440.0,
            qrs_area=1.2,
            t_amp_mv=0.4,
            reliable_for_global=True,
            reliable_for_p=True,
            reliable_for_qrs=True,
            reliable_for_t=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.95,
            p_confidence_mean=0.95,
            qrs_confidence_mean=0.95,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(120.0, features.pr_ms)  # median of limb leads I(118) and II(122); V5 excluded
        self.assertEqual(90.0, features.qrs_ms)
        self.assertEqual(430.0, features.qt_ms)
        self.assertEqual(20.0, features.qt_dispersion_ms)

    def test_compute_global_features_uses_supported_pr_consensus_when_limb_raw_pr_is_split(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        raw_pr_by_lead = {
            "I": 129.0,
            "II": 180.0,
            "III": 78.0,
            "aVR": 130.0,
            "aVL": 142.0,
            "aVF": 210.0,
            "V3": 221.0,
            "V4": 222.0,
        }
        for lead, raw_pr in raw_pr_by_lead.items():
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=raw_pr,
                pr_consensus_ms=201.0,
                p_onset_consensus_support=5,
                p_onset_consensus_spread_ms=182.0,
                qrs_ms=92.0,
                qrs_consensus_ms=92.0,
                qt_ms=410.0,
                qt_consensus_ms=420.0,
                p_amp_mv=0.08,
                qrs_area=1.0,
                t_amp_mv=0.25,
                reliable_for_global=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                p_confidence_mean=0.85,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(201.0, features.pr_ms)

    def test_compute_global_features_uses_supported_p_onset_cluster_when_short_false_pr_outliers_pull_raw_median_down(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        raw_pr_by_lead = {
            "I": 154.0,
            "II": 187.0,
            "III": 197.0,
            "aVR": 94.0,
            "aVL": 140.0,
            "aVF": 192.0,
            "V1": 159.0,
            "V3": 88.0,
            "V5": 77.0,
            "V6": 76.0,
        }
        for lead, raw_pr in raw_pr_by_lead.items():
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=raw_pr,
                pr_consensus_ms=178.0,
                p_onset_consensus_support=6,
                p_onset_consensus_spread_ms=16.0,
                p_onset_cluster_support=6,
                p_onset_cluster_spread_ms=16.0,
                p_onset_cluster_pr_ms=178.0,
                p_onset_cluster_limb_support=5,
                p_onset_cluster_precordial_support=1,
                qrs_ms=92.0,
                qrs_consensus_ms=92.0,
                qt_ms=410.0,
                qt_consensus_ms=420.0,
                p_amp_mv=0.06,
                qrs_area=1.0,
                t_amp_mv=0.25,
                reliable_for_global=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                p_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(178.0, features.pr_ms)

    def test_compute_global_features_does_not_use_p_onset_cluster_without_short_raw_pr_outliers(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        raw_pr_by_lead = {
            "I": 154.0,
            "II": 160.0,
            "III": 168.0,
            "aVR": 170.0,
            "aVL": 174.0,
            "aVF": 230.0,
        }
        for lead, raw_pr in raw_pr_by_lead.items():
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=raw_pr,
                pr_consensus_ms=182.0,
                p_onset_consensus_support=6,
                p_onset_consensus_spread_ms=16.0,
                p_onset_cluster_support=6,
                p_onset_cluster_spread_ms=16.0,
                p_onset_cluster_pr_ms=182.0,
                p_onset_cluster_limb_support=5,
                p_onset_cluster_precordial_support=1,
                qrs_ms=92.0,
                qrs_consensus_ms=92.0,
                qt_ms=410.0,
                qt_consensus_ms=420.0,
                p_amp_mv=0.06,
                qrs_area=1.0,
                t_amp_mv=0.25,
                reliable_for_global=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                p_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(169.0, features.pr_ms)

    def test_compute_global_features_does_not_use_long_p_onset_cluster_without_raw_support(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        raw_pr_by_lead = {
            "I": 149.0,
            "II": 150.0,
            "III": 150.0,
            "aVR": 138.0,
            "aVL": 98.0,
            "aVF": 158.0,
            "V1": 141.0,
            "V2": 152.0,
            "V3": 151.0,
            "V4": 230.0,
        }
        for lead, raw_pr in raw_pr_by_lead.items():
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=raw_pr,
                pr_consensus_ms=144.0,
                p_onset_consensus_support=9,
                p_onset_consensus_spread_ms=12.0,
                p_onset_cluster_support=9,
                p_onset_cluster_spread_ms=12.0,
                p_onset_cluster_pr_ms=219.0,
                p_onset_cluster_limb_support=6,
                p_onset_cluster_precordial_support=3,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=410.0,
                qt_consensus_ms=420.0,
                p_amp_mv=0.08,
                qrs_area=1.0,
                t_amp_mv=0.25,
                reliable_for_global=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                p_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(149.5, features.pr_ms)

    def test_compute_global_features_uses_lower_p_onset_cluster_when_long_raw_outliers_pull_median_up(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        raw_pr_by_lead = {
            "I": 230.0,
            "II": 207.0,
            "III": 156.0,
            "aVR": 207.0,
            "aVL": 223.0,
            "aVF": 156.0,
            "V1": 135.0,
            "V3": 276.0,
            "V6": 308.0,
        }
        for lead, raw_pr in raw_pr_by_lead.items():
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=raw_pr,
                pr_consensus_ms=214.0,
                p_onset_consensus_support=8,
                p_onset_consensus_spread_ms=14.0,
                p_onset_cluster_support=8,
                p_onset_cluster_spread_ms=14.0,
                p_onset_cluster_pr_ms=130.0,
                p_onset_cluster_limb_support=6,
                p_onset_cluster_precordial_support=2,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=410.0,
                qt_consensus_ms=420.0,
                p_amp_mv=0.08,
                qrs_area=1.0,
                t_amp_mv=0.25,
                reliable_for_global=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                p_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(130.0, features.pr_ms)

    def test_compute_global_features_qt_dispersion_uses_raw_per_lead_qt_not_consensus_qt(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt in [("II", 410.0), ("V5", 447.0), ("V6", 433.0)]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=90.0,
                qrs_consensus_ms=90.0,
                qt_ms=raw_qt,
                qt_consensus_ms=430.0,
                t_amp_mv=0.25,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertEqual(430.0, features.qt_ms)
        self.assertEqual(37.0, features.qt_dispersion_ms)

    def test_compute_global_features_prefers_raw_qt_core_when_consensus_is_late_cluster(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt in [
            ("I", 328.0),
            ("III", 354.0),
            ("aVF", 356.0),
            ("V3", 356.0),
            ("V6", 366.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=90.0,
                qrs_consensus_ms=90.0,
                qt_ms=raw_qt,
                qt_consensus_ms=394.0,
                t_amp_mv=0.12,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.75,
                qrs_confidence_mean=0.9,
            )
        for lead, raw_qt in [
            ("V4", 406.0),
            ("V5", 422.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=90.0,
                qrs_consensus_ms=90.0,
                qt_ms=raw_qt,
                qt_consensus_ms=394.0,
                t_amp_mv=0.16,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 430, 770, 1090], dtype=int),
            fs=500,
        )

        self.assertEqual(356.0, features.qt_ms)

    def test_compute_global_features_prefers_raw_qt_core_when_consensus_is_early_and_no_qt_leads_reliable(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt, qt_sd, t_amp, conf in [
            ("I", 306.0, 90.0, -0.03, 0.35),
            ("II", 368.0, 94.0, -0.03, 0.50),
            ("aVF", 358.0, 54.0, -0.02, 0.45),
            ("V3", 458.0, 18.0, 0.06, 0.70),
            ("V4", 426.0, 20.0, -0.05, 0.75),
            ("V5", 436.0, 16.0, -0.04, 0.80),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=96.0,
                qrs_consensus_ms=96.0,
                qt_ms=raw_qt,
                qt_consensus_ms=372.0,
                t_amp_mv=t_amp,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=False,
                qt_confidence_mean=conf,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = qt_sd

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 700, 1300, 1900], dtype=int),
            fs=500,
        )

        self.assertEqual(372.0, features.qt_ms)
        self.assertIsNotNone(features.qtc_bazett_ms)
        self.assertIsNotNone(features.qtc_fridericia_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("low_confidence", features.qt_reliability)
        self.assertEqual("low_qt_support_fallback", features.qt_source)
        self.assertEqual(["V3"], features.qt_used_leads)

    def test_compute_global_features_allows_three_lead_low_amp_qt_core_when_rr_irregular(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt, t_amp in [
            ("III", 360.0, 0.076),
            ("aVL", 350.0, -0.139),
            ("aVF", 342.0, 0.038),
            ("V1", 506.0, -0.102),
            ("V5", 528.0, 0.139),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=96.0,
                qrs_consensus_ms=96.0,
                qt_ms=raw_qt,
                qt_consensus_ms=494.0,
                t_amp_mv=t_amp,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.75,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 430, 660, 1080], dtype=int),
            fs=500,
        )

        self.assertEqual(350.0, features.qt_ms)

    def test_compute_global_features_uses_irregular_beat_qt_core_to_drop_next_cycle_outliers(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt, t_amp in [
            ("II", 532.0, -0.06),
            ("V4", 532.0, 0.05),
            ("V5", 396.0, 0.03),
            ("V6", 444.0, 0.03),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=95.0,
                qrs_consensus_ms=95.0,
                qt_ms=raw_qt,
                qt_consensus_ms=406.0,
                t_amp_mv=t_amp,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.6,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = 80.0

        beat_qts = [332.0, 352.0, 410.0, 404.0, 348.0, 390.0, 542.0,
                    420.0, 414.0, 540.0, 408.0, 278.0, 390.0, 526.0]
        beat_features = [
            _make_beat_feature("II", beat_id, pr_ms=0.0, qt_ms=qt, qt_consensus_ms=qt)
            for beat_id, qt in enumerate(beat_qts)
        ]

        features = compute_global_features(
            representative_leads,
            beat_features=beat_features,
            r_locs=np.asarray([100, 420, 665, 997, 1348, 1605, 1896, 2393,
                               2719, 3035, 3526, 3840, 4168, 4520], dtype=int),
            fs=500,
        )

        self.assertEqual(397.0, features.qt_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("irregular_beat_qt_core", features.qt_source)
        self.assertEqual("low_confidence", features.qt_reliability)
        self.assertEqual("insufficient_reliable_qt_leads", features.qt_confidence_reason)

    def test_compute_global_features_uses_irregular_beat_core_when_paced_consensus_is_early(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, raw_qt, qt_consensus, t_amp, conf, qt_sd in [
            ("aVL", 449.0, 385.0, -0.10, 0.21, 7.0),
            ("V1", 420.0, 385.0, 0.08, 0.24, 21.0),
            ("V5", 436.0, 385.0, 0.12, 0.18, 24.0),
            ("V6", 435.0, 385.0, 0.13, 0.20, 23.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=104.0,
                qrs_consensus_ms=104.0,
                qt_ms=raw_qt,
                qt_consensus_ms=qt_consensus,
                t_amp_mv=t_amp,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=conf,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = qt_sd

        beat_qts = [412.0, 420.0, 424.0, 418.0, 426.0, 420.0]
        beat_features = [
            _make_beat_feature("II", beat_id, pr_ms=0.0, qt_ms=qt, qt_consensus_ms=qt)
            for beat_id, qt in enumerate(beat_qts)
        ]

        features = compute_global_features(
            representative_leads,
            beat_features=beat_features,
            r_locs=np.asarray([100, 530, 880, 1360, 1710, 2210], dtype=int),
            fs=500,
            paced=True,
        )

        self.assertEqual(420.0, features.qt_ms)
        self.assertEqual("irregular_beat_qt_core", features.qt_source)
        self.assertEqual("rescued", features.qt_reliability)

    def test_compute_global_features_uses_raw_beat_qt_core_when_low_support_consensus_is_late(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead in ("II", "V1"):
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=96.0,
                qrs_consensus_ms=96.0,
                qt_ms=510.0,
                qt_consensus_ms=510.0,
                t_amp_mv=0.08,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.5,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = 18.0

        beat_qts = [258.0, 353.0, 433.0, 499.0, 566.0, 249.0, 384.0,
                    523.0, 466.0, 539.0, 484.0, 299.0, 268.0, 429.0]
        beat_features = [
            _make_beat_feature("II", beat_id, pr_ms=0.0, qt_ms=qt, qt_consensus_ms=qt)
            for beat_id, qt in enumerate(beat_qts)
        ]

        features = compute_global_features(
            representative_leads,
            beat_features=beat_features,
            r_locs=np.asarray([100, 430, 650, 1110, 1380, 1900, 2200,
                               2600, 2930, 3500, 3810, 4300, 4600, 5200], dtype=int),
            fs=500,
        )

        self.assertEqual(283.5, features.qt_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("irregular_raw_beat_qt_core", features.qt_source)
        self.assertEqual("low_confidence", features.qt_reliability)
        self.assertEqual(299.0, representative_leads["II"].params["qt_ms"])
        self.assertEqual(299.0, representative_leads["II"].params["qt_consensus_ms"])
        self.assertEqual(203.0, representative_leads["II"].params["jt_ms"])
        self.assertEqual("lead_raw_beat_core", representative_leads["II"].params["representative_qt_core_scope"])
        self.assertEqual(283.5, representative_leads["V1"].params["qt_ms"])
        self.assertEqual(283.5, representative_leads["V1"].params["qt_consensus_ms"])
        self.assertEqual(187.5, representative_leads["V1"].params["jt_ms"])
        self.assertEqual("global_raw_beat_core", representative_leads["V1"].params["representative_qt_core_scope"])
        for lead in ("II", "V1"):
            self.assertEqual("irregular_raw_beat_qt_core", representative_leads[lead].params["representative_qt_source"])

    def test_compute_global_features_prefers_tight_reliable_raw_qt_cluster_when_consensus_is_late(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt, qt_sd, t_amp in [
            ("I", 466.0, 21.0, 0.06),
            ("V2", 366.0, 18.0, 0.21),
            ("V3", 378.0, 12.0, 0.13),
            ("V4", 382.0, 11.0, 0.20),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=106.0,
                qrs_consensus_ms=106.0,
                qt_ms=raw_qt,
                qt_consensus_ms=390.0,
                t_amp_mv=t_amp,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.65,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = qt_sd

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 410, 720, 1030], dtype=int),
            fs=500,
        )

        self.assertEqual(378.0, features.qt_ms)

    def test_compute_global_features_uses_tight_wide_qrs_raw_qt_pair_when_consensus_is_late(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt in [("V2", 454.0), ("V3", 458.0)]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=186.0,
                qrs_consensus_ms=186.0,
                qt_ms=raw_qt,
                qt_consensus_ms=470.0,
                t_amp_mv=0.16,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.75,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = 6.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 616, 1118, 1632], dtype=int),
            fs=500,
        )

        self.assertEqual(456.0, features.qt_ms)

    def test_compute_global_features_does_not_use_late_raw_core_rescue_in_wide_qrs(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, raw_qt in [
            ("II", 532.0),
            ("aVR", 558.0),
            ("aVF", 562.0),
            ("V1", 566.0),
            ("V5", 514.0),
            ("V6", 540.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=138.0,
                qrs_consensus_ms=138.0,
                qt_ms=raw_qt,
                qt_consensus_ms=501.0,
                t_amp_mv=0.05,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.6,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = 45.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 873, 1622, 2376], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.qt_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("unavailable", features.qt_reliability)
        self.assertEqual("insufficient_reliable_qt_leads", features.qt_confidence_reason)

    def test_compute_global_features_keeps_consensus_when_all_qt_leads_are_reliable(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=86.0,
                qrs_consensus_ms=86.0,
                qt_ms=raw_qt,
                qt_consensus_ms=434.0,
                t_amp_mv=0.15,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.70,
                qrs_confidence_mean=0.9,
            )
            for lead, raw_qt in zip(
                STANDARD_12_LEADS,
                [430.0, 408.0, 394.0, 440.0, 418.0, 406.0,
                 416.0, 410.0, 402.0, 408.0, 430.0, 418.0],
            )
        }
        for rep in representative_leads.values():
            rep.variance["qt_ms_sd"] = 12.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 582, 1064, 1546], dtype=int),
            fs=500,
        )

        self.assertEqual(434.0, features.qt_ms)

    def test_qt_dispersion_uses_raw_reliable_leads_even_when_global_qt_uses_rescue(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt in [("II", 430.0), ("V2", 454.0), ("V4", 460.0), ("V6", 480.0)]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=160.0,
                qrs_consensus_ms=160.0,
                qt_ms=qt,
                qt_consensus_ms=436.0,
                t_amp_mv=0.15,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
        representative_leads["V6"].variance["qt_ms_sd"] = 30.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(454.0, features.qt_ms)
        self.assertEqual(50.0, features.qt_dispersion_ms)

    def test_compute_global_features_pr_uses_limb_lead_pr_ms(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            pr_ms=152.0,
            pr_consensus_ms=126.0,
            qrs_ms=92.0,
            qt_ms=380.0,
            qt_consensus_ms=400.0,
            qrs_area=1.0,
            t_amp_mv=0.3,
            reliable_for_global=True,
            reliable_for_p=True,
            reliable_for_qrs=True,
            reliable_for_t=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.9,
            p_confidence_mean=0.9,
            qrs_confidence_mean=0.9,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(152.0, features.pr_ms)  # lead II is a limb lead; pr_ms=152 is used directly

    def test_compute_global_features_reports_multilead_p_duration_provenance(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, duration in (("I", 82.0), ("II", 84.0)):
            representative_leads[lead] = _make_rep(
                lead,
                p_amp_mv=0.12,
                p_confidence_mean=0.9,
                reliable_for_p=True,
                p_dur_ms=108.0,
                p_dur_consensus_ms=duration,
                p_onset_consensus_support=5,
                p_offset_consensus_support=4,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(83.0, features.p_duration_ms)
        self.assertEqual(
            "representative_multilead_consensus",
            features.p_duration_source,
        )
        self.assertEqual(["I", "II"], features.p_duration_used_leads)
        self.assertEqual(4, features.p_duration_support)
        self.assertEqual(2.0, features.p_duration_spread_ms)
        self.assertEqual("reliable", features.p_duration_reliability)

    def test_multilead_consensus_corrects_supported_p_offset_outlier_on_raw_bounds(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        offsets = {
            "I": 58,
            "II": 59,
            "aVF": 60,
            "V1": 90,
            "V5": 91,
        }
        beat_features = []
        for lead, p_offset in offsets.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=390.0)
            feature.p = _make_wave(20, 40, p_offset)
            feature.qrs = _make_wave(100, 110, 120)
            feature.p_confidence = 0.9
            feature.qrs_confidence = 0.9
            beat_features.append(feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality)

        late_v1 = next(feature for feature in beat_features if feature.lead == "V1")
        self.assertEqual(74, late_v1.p.offset)
        self.assertIn("p_offset_corrected_by_consensus", late_v1.flags)
        self.assertEqual(20, late_v1.p_onset_raw_index)
        self.assertEqual(90, late_v1.p_offset_raw_index)
        self.assertEqual(20, late_v1.p_onset_corrected_index)
        self.assertEqual(74, late_v1.p_offset_corrected_index)
        self.assertEqual(0.5, late_v1.p_offset_correction_fraction)
        self.assertEqual("lead_consensus_blend", late_v1.p_boundary_source)
        self.assertAlmostEqual(108.0, late_v1.p_dur_ms)
        self.assertEqual(71, late_v1.p_offset_consensus_index)
        self.assertEqual(3, late_v1.p_offset_consensus_support)
        self.assertAlmostEqual(102.0, late_v1.p_dur_consensus_ms)
        self.assertAlmostEqual(58.0, late_v1.pr_segment_consensus_ms)
        self.assertAlmostEqual(141.2, late_v1.p_duration_guard_target_ms)
        self.assertAlmostEqual(24.0, late_v1.p_duration_guard_delta_ms)
        self.assertEqual(5, late_v1.p_duration_guard_support)
        self.assertEqual(
            "reliable_paired_lead_duration_p90",
            late_v1.p_duration_guard_source,
        )
        self.assertEqual(
            "paired_lead_duration_guard",
            late_v1.p_boundary_consensus_source,
        )

    def test_multilead_consensus_populates_p_boundary_confidence(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        beat_features = []
        for lead in ["I", "II", "aVF", "V1"]:
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=390.0)
            feature.p = _make_wave(20, 40, 60)
            feature.qrs = _make_wave(100, 110, 120)
            feature.p_confidence = 0.9
            feature.qrs_confidence = 0.9
            beat_features.append(feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality)

        for feature in beat_features:
            self.assertIsNotNone(feature.p_onset_confidence)
            self.assertIsNotNone(feature.p_offset_confidence)
            self.assertGreater(feature.p_onset_confidence, 0.70)
            self.assertGreater(feature.p_offset_confidence, 0.70)

    def test_multilead_consensus_recomputes_p_morphology_after_offset_repair_with_ecg(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        ecg = np.zeros((len(STANDARD_12_LEADS), 200), dtype=float)
        lead_to_idx = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}
        for lead in ["I", "II", "aVF", "V1"]:
            ecg[lead_to_idx[lead], 20:61] = np.hanning(41) * 0.12
        ecg[lead_to_idx["V5"], 20:61] = np.hanning(41) * 0.12
        ecg[lead_to_idx["V5"], 70:96] = 0.04

        offsets = {"I": 60, "II": 60, "aVF": 60, "V1": 60, "V5": 95}
        beat_features = []
        for lead, p_offset in offsets.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=390.0)
            feature.p = _make_wave(20, 40, p_offset)
            feature.qrs = _make_wave(100, 110, 120)
            feature.p_confidence = 0.9
            feature.qrs_confidence = 0.9
            feature.p_area = 999.0
            feature.p_signed_area = 999.0
            beat_features.append(feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality, ecg=ecg)

        repaired = next(feature for feature in beat_features if feature.lead == "V5")
        self.assertEqual(78, repaired.p.offset)
        self.assertIn("p_offset_corrected_by_consensus", repaired.flags)
        self.assertLess(repaired.p_area, 999.0)
        self.assertLess(repaired.p_signed_area, 999.0)
        self.assertIs(repaired.p_notched, True)
        self.assertIs(repaired.p_biphasic, False)

    def test_multilead_consensus_selects_guarded_later_cross_family_p_offset_cluster(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        offsets = {
            "I": 60,
            "III": 60,
            "aVF": 60,
            "V1": 60,
            "II": 90,
            "V2": 90,
            "V3": 90,
        }
        beat_features = []
        for lead, p_offset in offsets.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=390.0)
            feature.p = _make_wave(20, 40, p_offset)
            feature.qrs = _make_wave(130, 140, 160)
            feature.p_confidence = 0.9
            feature.qrs_confidence = 0.9
            beat_features.append(feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality)

        for feature in beat_features:
            self.assertEqual(90, feature.p_offset_consensus_index)
            self.assertEqual(3, feature.p_offset_consensus_support)
            self.assertAlmostEqual(140.0, feature.p_dur_consensus_ms)
            self.assertEqual(
                "lead_boundary_late_supported_cluster",
                feature.p_boundary_consensus_source,
            )

    def test_multilead_consensus_does_not_force_ambiguous_supported_p_offset_clusters(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        offsets = {
            "I": 60,
            "II": 60,
            "aVF": 60,
            "V1": 90,
            "V2": 90,
            "V3": 90,
        }
        beat_features = []
        for lead, p_offset in offsets.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=390.0)
            feature.p = _make_wave(20, 40, p_offset)
            feature.qrs = _make_wave(120, 130, 150)
            feature.p_confidence = 0.95
            feature.qrs_confidence = 0.9
            beat_features.append(feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality)

        for feature in beat_features:
            self.assertEqual(offsets[feature.lead], feature.p.offset)
            self.assertNotIn("p_offset_corrected_by_consensus", feature.flags)
            self.assertIsNone(feature.p_offset_consensus_index)
            self.assertIsNone(feature.p_boundary_consensus_source)

    def test_multilead_consensus_corrects_supported_p_onset_outlier_on_raw_bounds(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        onset_by_lead = {
            "I": 10,
            "II": 11,
            "III": 12,
            "aVL": 13,
            "aVF": 14,
            "V1": 15,
            "V5": 48,
        }
        beat_features = []
        for lead, p_onset in onset_by_lead.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=390.0)
            feature.p = _make_wave(p_onset, p_onset + 7, p_onset + 27)
            feature.qrs = _make_wave(100, 110, 120)
            feature.p_confidence = 0.9
            feature.qrs_confidence = 0.9
            beat_features.append(feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality)

        late_v5 = next(feature for feature in beat_features if feature.lead == "V5")
        self.assertEqual(12, late_v5.p.onset)
        self.assertIn("p_onset_corrected_by_consensus", late_v5.flags)
        self.assertAlmostEqual(176.0, late_v5.pr_ms)
        self.assertEqual(12, late_v5.p_onset_cluster_center_index)
        self.assertEqual("physiologic_onset_cluster", late_v5.p_onset_consensus_reason)

    def test_multilead_consensus_suppresses_short_false_p_candidate_when_peak_cannot_match_cluster(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        beat_features = []
        for lead, p_onset in {
            "I": 10,
            "II": 11,
            "III": 12,
            "aVL": 13,
            "aVF": 14,
            "V1": 15,
        }.items():
            feature = _make_beat_feature(lead, 0, pr_ms=176.0, qt_ms=390.0)
            feature.p = _make_wave(p_onset, p_onset + 15, p_onset + 38)
            feature.qrs = _make_wave(100, 110, 120)
            feature.p_confidence = 0.9
            feature.qrs_confidence = 0.9
            beat_features.append(feature)

        false_feature = _make_beat_feature("V5", 0, pr_ms=56.0, qt_ms=390.0)
        false_feature.p = _make_wave(72, 84, 95)
        false_feature.qrs = _make_wave(100, 110, 120)
        false_feature.p_confidence = 0.9
        false_feature.qrs_confidence = 0.9
        false_feature.p_amp_mv = 0.05
        false_feature.p_area = 1.2
        false_feature.p_signed_area = 1.2
        false_feature.p_dur_ms = 46.0
        beat_features.append(false_feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality)

        self.assertIsNone(false_feature.p.onset)
        self.assertIsNone(false_feature.p.peak)
        self.assertIsNone(false_feature.p.offset)
        self.assertIsNone(false_feature.pr_ms)
        self.assertIsNone(false_feature.p_amp_mv)
        self.assertIsNone(false_feature.p_area)
        self.assertEqual(0.0, false_feature.p_confidence)
        self.assertIn("p_candidate_suppressed_by_consensus", false_feature.flags)
        self.assertIn("p_unreliable", false_feature.flags)

    def test_multilead_consensus_selects_physiologic_p_onset_cluster_over_short_false_cluster(self) -> None:
        fs = 500
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        onset_by_lead = {
            "I": 9,
            "II": 10,
            "III": 11,
            "aVL": 12,
            "aVF": 13,
            "V1": 14,
            "aVR": 52,
            "V3": 56,
            "V5": 60,
            "V6": 62,
        }
        beat_features = []
        for lead, p_onset in onset_by_lead.items():
            feature = _make_beat_feature(lead, 0, pr_ms=160.0, qt_ms=390.0)
            feature.p = _make_wave(p_onset, p_onset + 15, p_onset + 38)
            feature.qrs = _make_wave(100, 110, 120)
            feature.p_confidence = 0.9
            feature.qrs_confidence = 0.9
            beat_features.append(feature)

        _apply_multilead_consensus(beat_features, fs=fs, quality=quality)

        v5 = next(feature for feature in beat_features if feature.lead == "V5")
        self.assertEqual(12, v5.p_onset_cluster_center_index)
        self.assertEqual(6, v5.p_onset_cluster_support)
        self.assertAlmostEqual(10.0, v5.p_onset_cluster_spread_ms)
        self.assertAlmostEqual(176.0, v5.p_onset_cluster_pr_ms)
        self.assertEqual(5, v5.p_onset_cluster_limb_support)
        self.assertEqual(1, v5.p_onset_cluster_precordial_support)
        self.assertEqual("I,II,III,aVL,aVF,V1", v5.p_onset_cluster_leads)
        self.assertEqual("physiologic_onset_cluster", v5.p_onset_consensus_reason)
        self.assertEqual(6, v5.p_onset_consensus_support)
        self.assertAlmostEqual(10.0, v5.p_onset_consensus_spread_ms)

    def test_compute_global_features_falls_back_to_per_lead_qt_when_wide_qrs_consensus_qt_is_implausible(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=210.0,
                qrs_consensus_ms=216.0,
                qt_ms=460.0,
                qt_consensus_ms=313.0,
                t_amp_mv=0.12,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead in STANDARD_12_LEADS
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(216.0, features.qrs_ms)
        self.assertEqual(460.0, features.qt_ms)

    def test_compute_global_features_caps_qrs_consensus_when_it_exceeds_all_per_lead_widths(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=220.0,
                qt_ms=460.0,
                qt_consensus_ms=500.0,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [120.0, 132.0, 148.0, 160.0, 176.0, 180.0, 118.0, 130.0, 140.0, 150.0, 162.0, 170.0],
            )
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(180.0, features.qrs_ms)

    def test_compute_global_features_caps_qrs_consensus_when_only_slightly_beyond_raw_widths(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=206.0,
                qt_ms=458.0,
                qt_consensus_ms=458.0,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                t_amp_mv=0.20,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [172.0, 116.0, 164.0, 120.0, 186.0, 174.0, 126.0, 108.0, 94.0, 122.0, 118.0, 116.0],
            )
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(186.0, features.qrs_ms)

    def test_compute_global_features_caps_qrs_wide_for_direct_representative_leads(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=206.0,
                qrs_wide_ms=206.0,
                qt_ms=300.0,
                qt_consensus_ms=300.0,
                t_amp_mv=0.20,
                t_area=1.0 if lead in {"I", "II"} else None,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [90.0, 94.0, 98.0, 102.0, 106.0, 110.0, 92.0, 96.0, 100.0, 104.0, 108.0, 110.0],
            )
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(110.0, features.qrs_ms)
        self.assertEqual(300.0, features.qt_ms)
        self.assertIsNotNone(features.t_axis_deg)

    def test_compute_global_qrs_ignores_sparse_wide_outlier_leads(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=92.0,
                qrs_consensus_ms=130.0,
                qrs_confidence_mean=0.9,
                reliable_for_qrs=True,
            )
            for lead in STANDARD_12_LEADS
        }
        representative_leads["V4"].params["qrs_ms"] = 218.0
        representative_leads["V5"].params["qrs_ms"] = 182.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertLess(features.qrs_ms, 120.0)

    def test_compute_global_qrs_trusts_strong_wide_consensus_when_raw_widths_are_truncated(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=166.0,
                qrs_wide_ms=170.0,
                qt_ms=456.0,
                qt_consensus_ms=456.0,
                t_amp_mv=0.20,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [78.0, 198.0, 54.0, 67.0, 80.0, 80.0, 64.0, 84.0, 66.0, 122.0, 76.0, 76.0],
            )
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertEqual(166.0, features.qrs_ms)

    def test_compute_global_qrs_keeps_wide_consensus_when_raw_widths_have_wide_support(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=179.0,
                qrs_wide_ms=210.0,
                qt_ms=445.0,
                qt_consensus_ms=445.0,
                t_amp_mv=0.20,
                st_t_confusion=(lead == "III"),
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [None, 139.0, 123.0, 120.0, 121.0, 80.0, None, 198.0, None, 135.0, 81.0, 114.0],
            )
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 760, 1420], dtype=int),
            fs=500,
        )

        self.assertEqual(179.0, features.qrs_ms)

    def test_compute_global_qrs_allows_raw_core_for_confirmed_pacing_late_consensus(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=176.0,
                qrs_wide_ms=200.0,
                qt_ms=460.0,
                qt_consensus_ms=460.0,
                t_amp_mv=0.20,
                st_t_confusion=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [120.0, 126.0, 128.0, 130.0, 132.0, 134.0, 136.0, 138.0, 140.0, 142.0, 144.0, 146.0],
            )
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 760, 1420], dtype=int),
            fs=500,
            paced=True,
        )

        self.assertLess(features.qrs_ms or 999.0, 140.0)

    def test_compute_global_qrs_uses_raw_core_when_qrs_offset_marks_late_low_support(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=204.0,
                qrs_wide_ms=206.0,
                qt_ms=408.0,
                qt_consensus_ms=408.0,
                t_amp_mv=0.20,
                st_t_confusion=False,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [146.0, 144.0, 138.0, 96.0, 134.0, 158.0, None, 199.0, None, None, None, 110.0],
            )
        }
        representative_leads["V1"].params["qrs_offset_exclusion_reason"] = "late_offset_low_support"

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 760, 1420], dtype=int),
            fs=500,
        )

        self.assertLess(features.qrs_ms or 999.0, 160.0)

    def test_compute_global_qrs_uses_split_raw_core_when_limb_and_precordial_disagree(self) -> None:
        raw_qrs_by_lead = {
            "I": 54.0,
            "II": 60.0,
            "III": 56.0,
            "aVR": 94.0,
            "aVL": 54.0,
            "aVF": 58.0,
            "V1": 218.0,
            "V2": 164.0,
            "V3": None,
            "V4": 150.0,
            "V5": 198.0,
            "V6": 200.0,
        }
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=198.0,
                qrs_wide_ms=198.0,
                qt_ms=496.0,
                qt_consensus_ms=386.0,
                t_amp_mv=0.20,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                qt_confidence_mean=0.8,
                qrs_confidence_mean=0.8,
            )
            for lead, raw_qrs in raw_qrs_by_lead.items()
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 560, 1020, 1480], dtype=int),
            fs=500,
        )

        self.assertGreaterEqual(features.qrs_ms or 0.0, 120.0)
        self.assertLess(features.qrs_ms or 999.0, 170.0)

    def test_compute_global_features_uses_later_raw_qt_for_split_qrs_short_consensus(self) -> None:
        qt_by_lead = {
            "I": (54.0, 348.0, 386.0, 0.24, 0.97, False),
            "II": (60.0, 498.0, 386.0, -0.27, 0.09, False),
            "III": (56.0, 446.0, 386.0, -0.26, 0.32, False),
            "aVR": (94.0, 512.0, 386.0, -0.07, 0.63, True),
            "aVL": (54.0, 437.0, 386.0, 0.25, 0.36, False),
            "aVF": (58.0, 496.0, 386.0, -0.26, 0.31, True),
            "V1": (218.0, 468.0, 386.0, 0.20, 0.15, False),
            "V2": (164.0, 466.0, 386.0, 0.29, 0.14, False),
            "V3": (None, 482.0, 386.0, 0.23, 0.15, False),
            "V4": (150.0, 524.0, 386.0, 0.25, 0.45, False),
            "V5": (198.0, 530.0, 386.0, -0.15, 0.27, True),
            "V6": (200.0, 472.0, 386.0, -0.13, 0.27, True),
        }
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=qrs_ms,
                qrs_consensus_ms=198.0,
                qrs_wide_ms=198.0,
                qt_ms=qt_ms,
                qt_consensus_ms=qt_consensus_ms,
                t_amp_mv=t_amp_mv,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=reliable_for_qt,
                qt_confidence_mean=qt_confidence,
                qrs_confidence_mean=0.8,
            )
            for lead, (
                qrs_ms,
                qt_ms,
                qt_consensus_ms,
                t_amp_mv,
                qt_confidence,
                reliable_for_qt,
            ) in qt_by_lead.items()
        }

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 560, 1020, 1480], dtype=int),
            fs=500,
        )

        self.assertGreaterEqual(features.qt_ms or 0.0, 450.0)
        self.assertNotEqual("reliable_lead_median", features.qt_source)

    def test_compute_global_features_suppresses_pr_and_p_axis_without_p_support(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            pr_ms=206.0,
            qrs_ms=160.0,
            qt_ms=460.0,
            qt_consensus_ms=460.0,
            p_amp_mv=0.004,
            p_area=1.5,
            r_amp_mv=1.0,
            s_amp_mv=-0.2,
            t_amp_mv=0.2,
            reliable_for_global=True,
            reliable_for_p=True,
            reliable_for_qrs=True,
            reliable_for_t=True,
            reliable_for_qt=True,
            p_confidence_mean=0.9,
            qrs_confidence_mean=0.9,
            qt_confidence_mean=0.9,
        )
        representative_leads["aVR"] = _make_rep(
            "aVR",
            pr_ms=136.0,
            qrs_ms=162.0,
            qt_ms=462.0,
            qt_consensus_ms=462.0,
            p_amp_mv=0.010,
            p_area=2.0,
            r_amp_mv=0.8,
            s_amp_mv=-0.2,
            t_amp_mv=0.2,
            reliable_for_global=True,
            reliable_for_p=True,
            reliable_for_qrs=True,
            reliable_for_t=True,
            reliable_for_qt=True,
            p_confidence_mean=0.9,
            qrs_confidence_mean=0.9,
            qt_confidence_mean=0.9,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600, 1100], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.pr_ms)
        self.assertIsNone(features.p_axis_deg)

    def test_compute_global_features_suppresses_pr_and_p_axis_for_irregular_long_pr_false_p(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead in ("II", "aVF", "V2", "V3", "V4", "V5"):
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=300.0,
                p_amp_mv=0.03,
                p_area=0.2,
                reliable_for_global=True,
                reliable_for_p=True,
                p_confidence_mean=0.35,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 480, 930, 1210, 1800, 2050, 2500], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.pr_ms)
        self.assertIsNone(features.p_axis_deg)

    def test_compute_global_features_suppresses_pr_and_p_axis_for_af_threshold_irregular_long_pr(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead in ("II", "aVF", "V2", "V3", "V4", "V5"):
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=300.0,
                p_amp_mv=0.03,
                p_area=0.2,
                reliable_for_global=True,
                reliable_for_p=True,
                p_confidence_mean=0.35,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 373, 773, 1083, 1463, 1808, 2276], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.pr_ms)
        self.assertIsNone(features.p_axis_deg)

    def test_build_representative_lead_features_filters_implausible_intervals(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        items = [
            _make_beat_feature("II", 0, pr_ms=-48.0, qrs_ms=286.0, qt_ms=214.0),
            _make_beat_feature("II", 1, pr_ms=150.0, qrs_ms=100.0, qt_ms=420.0),
        ]

        representatives = build_representative_lead_features(items, quality)

        self.assertEqual(150.0, representatives["II"].params["pr_ms"])
        self.assertEqual(100.0, representatives["II"].params["qrs_ms"])
        self.assertEqual(420.0, representatives["II"].params["qt_ms"])

    def test_build_representative_lead_features_keeps_pooled_intervals_when_medoid_is_single_beat(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        items = [
            _make_beat_feature("II", 0, pr_ms=132.0, qrs_ms=94.0, qt_ms=398.0),
            _make_beat_feature("II", 1, pr_ms=136.0, qrs_ms=98.0, qt_ms=404.0),
            _make_beat_feature("II", 2, pr_ms=140.0, qrs_ms=102.0, qt_ms=410.0),
        ]
        medoid_feature = _make_beat_feature("II", 1, pr_ms=80.0, qrs_ms=124.0, qt_ms=500.0)

        representatives = build_representative_lead_features(
            items,
            quality,
            representative_beat_features=[medoid_feature],
            selected_beat_ids={0, 1, 2},
        )

        self.assertEqual(136.0, representatives["II"].params["pr_ms"])
        self.assertEqual(98.0, representatives["II"].params["qrs_ms"])
        self.assertEqual(404.0, representatives["II"].params["qt_ms"])

    def test_build_representative_lead_features_suppresses_pr_when_p_wave_is_tiny(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        items = [
            _make_beat_feature("II", 0, pr_ms=154.0, qt_ms=420.0, p_amp_mv=0.004),
            _make_beat_feature("II", 1, pr_ms=158.0, qt_ms=424.0, p_amp_mv=0.005),
        ]

        representatives = build_representative_lead_features(items, quality)

        self.assertIsNone(representatives["II"].params["pr_ms"])
        self.assertIsNone(representatives["II"].params["pr_consensus_ms"])

    def test_build_representative_lead_features_aggregates_p_fine_morphology(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        items = [
            _make_beat_feature("V1", 0, pr_ms=150.0, qt_ms=420.0),
            _make_beat_feature("V1", 1, pr_ms=152.0, qt_ms=422.0),
        ]
        items[0].p_biphasic = True
        items[0].p_terminal_duration_ms = 42.0
        items[0].p_terminal_amp_mv = -0.08
        items[0].p_terminal_area_mv_ms = -4.5
        items[1].p_notched = True
        items[1].p_notch_interval_ms = 34.0
        items[1].p_initial_duration_ms = 36.0
        items[1].p_initial_amp_mv = 0.11

        representatives = build_representative_lead_features(items, quality)

        params = representatives["V1"].params
        self.assertIs(params["p_biphasic"], True)
        self.assertIs(params["p_notched"], True)
        self.assertEqual(34.0, params["p_notch_interval_ms"])
        self.assertEqual(36.0, params["p_initial_duration_ms"])
        self.assertEqual(0.11, params["p_initial_amp_mv"])
        self.assertEqual(42.0, params["p_terminal_duration_ms"])
        self.assertEqual(-0.08, params["p_terminal_amp_mv"])
        self.assertEqual(-4.5, params["p_terminal_area_mv_ms"])

    def test_build_representative_lead_features_aggregates_qrs_offset_confidence(self) -> None:
        quality = {lead: _make_quality(reliable=True) for lead in STANDARD_12_LEADS}
        items = [
            _make_beat_feature("II", 0, pr_ms=150.0, qt_ms=420.0),
            _make_beat_feature("II", 1, pr_ms=150.0, qt_ms=420.0),
        ]
        items[0].qrs_off_confidence = 0.2
        items[1].qrs_off_confidence = 0.6

        representatives = build_representative_lead_features(items, quality)

        self.assertEqual(0.4, representatives["II"].params["qrs_off_confidence_mean"])

    def test_build_representative_lead_features_carries_t_wave_provenance(self) -> None:
        quality = {lead: _make_quality(reliable=True, reliable_for_t=True) for lead in STANDARD_12_LEADS}
        items = [
            _make_beat_feature(
                "III",
                0,
                pr_ms=150.0,
                qt_ms=420.0,
                t_signed_area=-2.0,
                t_polarity_observed=-1,
                st_t_confusion=True,
                t_confidence_reason="st_t_confusion",
            ),
            _make_beat_feature(
                "III",
                1,
                pr_ms=150.0,
                qt_ms=420.0,
                t_signed_area=-1.5,
                t_polarity_observed=-1,
                st_t_confusion=False,
                t_confidence_reason="polarity_cluster",
            ),
        ]

        representatives = build_representative_lead_features(items, quality)

        params = representatives["III"].params
        self.assertIs(params["st_t_confusion"], True)
        self.assertEqual(-1, params["t_polarity_observed"])
        self.assertEqual("st_t_confusion", params["t_confidence_reason"])

    def test_build_representative_lead_features_marks_unmeasured_p_booleans_unknown(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }

        representatives = build_representative_lead_features([], quality)

        self.assertIsNone(representatives["V1"].params["p_notched"])
        self.assertIsNone(representatives["V1"].params["p_biphasic"])

    def test_build_representative_lead_features_keeps_measured_false_p_booleans(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        items = [_make_beat_feature("V1", 0, pr_ms=150.0, qt_ms=420.0)]

        representatives = build_representative_lead_features(items, quality)

        self.assertIs(representatives["V1"].params["p_notched"], False)
        self.assertIs(representatives["V1"].params["p_biphasic"], False)

    def test_build_representative_lead_features_keeps_missing_p_bounds_booleans_unknown(self) -> None:
        quality = {
            lead: _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            for lead in STANDARD_12_LEADS
        }
        item = _make_beat_feature("V1", 0, pr_ms=150.0, qt_ms=420.0)
        item.p = WaveBounds(None, None, None)
        item.p_amp_mv = None
        item.p_dur_ms = None
        item.p_area = None
        item.p_notch_interval_ms = None
        item.p_initial_duration_ms = None
        item.p_initial_amp_mv = None
        item.p_terminal_duration_ms = None
        item.p_terminal_amp_mv = None
        item.p_terminal_area_mv_ms = None

        representatives = build_representative_lead_features([item], quality)

        self.assertIsNone(representatives["V1"].params["p_notched"])
        self.assertIsNone(representatives["V1"].params["p_biphasic"])

    @needs_interpretation
    def test_p_wave_morphology_keeps_unknown_v1_biphasic_terminal_neutral_for_rae(self) -> None:
        from feature_extraction.ecgfeat.interpret import _p_wave_morphology

        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            p_amp_mv=0.25,
            reliable_for_p=True,
        )
        representative_leads["V1"] = _make_rep(
            "V1",
            p_biphasic=True,
            reliable_for_p=True,
        )

        p_class, _rae_leads, _lae_suspected, _lae_definite, _ptf = _p_wave_morphology(
            representative_leads,
            p_dur_by_lead={},
            ptf_v1=None,
        )

        self.assertEqual("probable_rae", p_class)

    @needs_interpretation
    def test_p_wave_morphology_keeps_isolated_notched_p_neutral_for_lae(self) -> None:
        from feature_extraction.ecgfeat.interpret import _p_wave_morphology

        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            p_notched=True,
            reliable_for_p=True,
        )

        p_class, _rae_leads, lae_suspected, _lae_definite, _ptf = _p_wave_morphology(
            representative_leads,
            p_dur_by_lead={},
            ptf_v1=None,
        )

        self.assertEqual("normal", p_class)
        self.assertFalse(lae_suspected)

    @needs_interpretation
    def test_p_wave_morphology_uses_negative_v1_biphasic_terminal_for_lae(self) -> None:
        from feature_extraction.ecgfeat.interpret import _p_wave_morphology

        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["V1"] = _make_rep(
            "V1",
            p_biphasic=True,
            p_terminal_amp_mv=-0.08,
            p_terminal_area_mv_ms=-3.2,
            reliable_for_p=True,
        )

        p_class, _rae_leads, lae_suspected, _lae_definite, _ptf = _p_wave_morphology(
            representative_leads,
            p_dur_by_lead={},
            ptf_v1=None,
        )

        self.assertEqual("probable_lae", p_class)
        self.assertTrue(lae_suspected)

    @needs_interpretation
    def test_p_wave_morphology_downgrades_ptf_v1_without_terminal_component_support(self) -> None:
        from feature_extraction.ecgfeat.interpret import _p_wave_morphology

        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["V1"] = _make_rep(
            "V1",
            p_terminal_amp_mv=-0.03,
            p_terminal_duration_ms=12.0,
            p_terminal_area_mv_ms=-0.36,
            reliable_for_p=True,
        )

        p_class, _rae_leads, lae_suspected, lae_definite, ptf_v1_class = _p_wave_morphology(
            representative_leads,
            p_dur_by_lead={},
            ptf_v1=-8.0,
        )

        self.assertEqual("probable_lae", p_class)
        self.assertTrue(lae_suspected)
        self.assertFalse(lae_definite)
        self.assertEqual("probable_lae", ptf_v1_class)

    @needs_interpretation
    def test_p_wave_morphology_keeps_ptf_v1_definite_with_terminal_component_support(self) -> None:
        from feature_extraction.ecgfeat.interpret import _p_wave_morphology

        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["V1"] = _make_rep(
            "V1",
            p_terminal_amp_mv=-0.16,
            p_terminal_duration_ms=36.0,
            p_terminal_area_mv_ms=-5.76,
            reliable_for_p=True,
        )

        p_class, _rae_leads, lae_suspected, lae_definite, ptf_v1_class = _p_wave_morphology(
            representative_leads,
            p_dur_by_lead={},
            ptf_v1=-5.76,
        )

        self.assertEqual("lae", p_class)
        self.assertTrue(lae_suspected)
        self.assertTrue(lae_definite)
        self.assertEqual("definite_lae", ptf_v1_class)

    @needs_interpretation
    def test_p_wave_morphology_downgrades_ptf_v1_when_p_support_is_suppressed(self) -> None:
        from feature_extraction.ecgfeat.interpret import _p_wave_morphology

        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["V1"] = _make_rep(
            "V1",
            p_terminal_amp_mv=-0.16,
            p_terminal_duration_ms=40.0,
            p_terminal_area_mv_ms=-6.40,
            p_measurement_suppressed="low_p_support",
            reliable_for_p=True,
        )

        p_class, _rae_leads, lae_suspected, lae_definite, ptf_v1_class = _p_wave_morphology(
            representative_leads,
            p_dur_by_lead={},
            ptf_v1=-6.40,
        )

        self.assertEqual("probable_lae", p_class)
        self.assertTrue(lae_suspected)
        self.assertFalse(lae_definite)
        self.assertEqual("probable_lae", ptf_v1_class)

    def test_compute_global_features_filters_implausible_intervals(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            pr_ms=-48.0,
            qrs_ms=286.0,
            qrs_consensus_ms=286.0,
            qt_ms=214.0,
            qt_consensus_ms=214.0,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["V5"] = _make_rep(
            "V5",
            pr_ms=150.0,
            qrs_ms=100.0,
            qrs_consensus_ms=100.0,
            qt_ms=420.0,
            qt_consensus_ms=420.0,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([0, 500, 1000], dtype=int),
            fs=500,
        )

        self.assertEqual(150.0, features.pr_ms)
        self.assertEqual(100.0, features.qrs_ms)
        self.assertIsNone(features.qt_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("unavailable", features.qt_reliability)
        self.assertEqual("insufficient_reliable_qt_leads", features.qt_confidence_reason)

    def test_compute_global_features_uses_qrs_specific_gate(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["I"] = _make_rep(
            "I",
            qrs_ms=98.0,
            qrs_consensus_ms=96.0,
            reliable_for_global=False,
            reliable_for_qrs=True,
        )
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=104.0,
            reliable_for_global=False,
            reliable_for_qrs=True,
        )
        representative_leads["V5"] = _make_rep(
            "V5",
            pr_ms=150.0,
            qrs_ms=140.0,
            qrs_consensus_ms=140.0,
            reliable_for_global=True,
            reliable_for_qrs=False,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(100.0, features.qrs_ms)

    def test_build_representative_lead_features_can_pool_selected_beats_only(self) -> None:
        quality = {
            lead: _make_quality(reliable=True, reliable_for_p=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        beat_features = [
            _make_beat_feature("II", 0, 120.0, 360.0),
            _make_beat_feature("II", 1, 124.0, 364.0),
            _make_beat_feature("II", 2, 172.0, 420.0),
            _make_beat_feature("II", 3, 176.0, 424.0),
        ]

        representatives = build_representative_lead_features(
            beat_features,
            quality,
            selected_beat_ids={2, 3},
            dominant_group_id=2,
        )

        self.assertEqual(174.0, representatives["II"].params["pr_ms"])
        self.assertEqual(422.0, representatives["II"].params["qt_ms"])
        self.assertEqual(2, representatives["II"].params["beat_count"])

    @needs_interpretation
    def test_build_representative_lead_features_ignores_unreliable_st_j_values(self) -> None:
        from feature_extraction.ecgfeat.interpret import _st_j_lp

        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        guarded = _make_beat_feature("II", 1, 120.0, 400.0)
        guarded.st_on_mv = 0.60
        guarded.flags.append("st_j_unreliable")
        reliable = _make_beat_feature("II", 2, 120.0, 400.0)
        reliable.st_on_mv = 0.02

        representatives = build_representative_lead_features([guarded, reliable], quality)

        params = representatives["II"].params
        self.assertEqual(0.02, params["st_on_mv"])
        self.assertTrue(params["st_j_reliable"])
        self.assertFalse(params["st_j_unreliable"])
        self.assertEqual("mixed_reliable_beats", params["st_j_source"])
        self.assertIsNone(params["st_j_unreliable_reason"])
        self.assertEqual(0.02, _st_j_lp(representatives, "II"))

    @needs_interpretation
    def test_build_representative_lead_features_marks_st_j_unreliable_when_all_guarded(self) -> None:
        from feature_extraction.ecgfeat.interpret import _st_j_lp

        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        first = _make_beat_feature("II", 1, 120.0, 400.0)
        first.st_on_mv = 0.60
        first.flags.append("st_j_unreliable")
        second = _make_beat_feature("II", 2, 120.0, 400.0)
        second.st_on_mv = 0.50
        second.flags.append("st_j_unreliable")

        representatives = build_representative_lead_features([first, second], quality)

        params = representatives["II"].params
        self.assertIsNone(params["st_on_mv"])
        self.assertFalse(params["st_j_reliable"])
        self.assertTrue(params["st_j_unreliable"])
        self.assertEqual("qrs_tail_guard", params["st_j_source"])
        self.assertEqual("qrs_tail_guard", params["st_j_unreliable_reason"])
        self.assertIsNone(_st_j_lp(representatives, "II"))

    @needs_interpretation
    def test_build_representative_lead_features_does_not_rescue_all_guarded_st_j_with_rep_feature(self) -> None:
        from feature_extraction.ecgfeat.interpret import _st_j_lp

        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True, reliable_for_qt=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        first = _make_beat_feature("II", 1, 120.0, 400.0)
        first.st_on_mv = 0.60
        first.flags.append("st_j_unreliable")
        second = _make_beat_feature("II", 2, 120.0, 400.0)
        second.st_on_mv = 0.50
        second.flags.append("st_j_unreliable")
        rep_feature = _make_beat_feature("II", 99, 120.0, 400.0)
        rep_feature.st_on_mv = 0.02

        representatives = build_representative_lead_features(
            [first, second],
            quality,
            representative_beat_features=[rep_feature],
        )

        params = representatives["II"].params
        self.assertIsNone(params["st_on_mv"])
        self.assertTrue(params["st_j_unreliable"])
        self.assertEqual("qrs_tail_guard", params["st_j_source"])
        self.assertIsNone(_st_j_lp(representatives, "II"))

    def test_build_representative_lead_features_caps_qrs_consensus_to_raw_width_distribution(self) -> None:
        quality = {
            lead: _make_quality(reliable=True, reliable_for_qrs=True)
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name

        beat_features = [
            _make_beat_feature(
                lead,
                0,
                pr_ms=160.0,
                qrs_ms=raw_qrs,
                qt_ms=430.0,
            )
            for lead, raw_qrs in zip(
                STANDARD_12_LEADS,
                [172.0, 116.0, 164.0, 120.0, 186.0, 174.0, 126.0, 108.0, 94.0, 122.0, 118.0, 116.0],
            )
        ]
        for bf in beat_features:
            bf.qrs_consensus_ms = 206.0
            bf.qrs_wide_ms = 206.0

        representatives = build_representative_lead_features(beat_features, quality)

        self.assertEqual(186.0, representatives["II"].params["qrs_consensus_ms"])
        self.assertEqual(186.0, representatives["II"].params["qrs_wide_ms"])

    def test_build_representative_lead_features_carries_wave_areas_into_global_axes(self) -> None:
        quality = {
            lead: _make_quality()
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name
        for lead_name in ("I", "aVF"):
            quality[lead_name] = _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            quality[lead_name].lead = lead_name

        representatives = build_representative_lead_features(
            [
                _make_beat_feature("I", 0, 120.0, 360.0, p_amp_mv=0.2, p_area=1.0, t_amp_mv=0.3, t_area=1.5),
                _make_beat_feature("aVF", 0, 120.0, 360.0, p_amp_mv=0.2, p_area=1.0, t_amp_mv=0.3, t_area=1.5),
            ],
            quality,
        )

        self.assertEqual(1.0, representatives["I"].params["p_area"])
        self.assertEqual(1.5, representatives["I"].params["t_area"])

        features = compute_global_features(
            representatives,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertAlmostEqual(45.0, features.p_axis_deg)
        self.assertAlmostEqual(45.0, features.t_axis_deg)

    def test_compute_global_features_prefers_signed_qrs_area_for_qrs_axis(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["I"] = _make_rep(
            "I",
            qrs_ms=100.0,
            qrs_signed_area=1.0,
            r_amp_mv=1.0,
            q_amp_mv=0.0,
            s_amp_mv=0.0,
            qrs_confidence_mean=1.0,
            reliable_for_qrs=True,
        )
        representative_leads["aVF"] = _make_rep(
            "aVF",
            qrs_ms=100.0,
            qrs_signed_area=-5.0,
            r_amp_mv=1.0,
            q_amp_mv=0.0,
            s_amp_mv=0.0,
            qrs_confidence_mean=1.0,
            reliable_for_qrs=True,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertLess(features.qrs_axis_deg, -45.0)

    def test_compute_global_features_excludes_low_context_p_axis_polarity_conflict(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp, context in [
            ("I", 1.40, 0.08, 0.92),
            ("II", 1.50, 0.09, 0.94),
            ("aVF", 1.35, 0.08, 0.91),
            ("aVR", 14.00, 0.05, 0.18),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                p_amp_mv=amp,
                p_area=abs(signed_area),
                p_signed_area=signed_area,
                reliable_for_p=True,
                p_confidence_mean=0.90 if lead != "aVR" else 0.32,
                p_candidate_context_score=context,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 20.0)
        self.assertLess(features.p_axis_deg, 85.0)

    def test_p_axis_values_exclude_low_boundary_confidence_polarity_conflict(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp in [
            ("I", 1.40, 0.08),
            ("II", 1.50, 0.09),
            ("aVF", 1.35, 0.08),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=164.0,
                p_amp_mv=amp,
                p_area=abs(signed_area),
                p_signed_area=signed_area,
                reliable_for_p=True,
                p_confidence_mean=0.92,
                p_candidate_context_score=0.92,
                p_onset_confidence_mean=0.95,
                p_offset_confidence_mean=0.95,
            )
        representative_leads["aVR"] = _make_rep(
            "aVR",
            pr_ms=164.0,
            p_amp_mv=0.06,
            p_area=14.0,
            p_signed_area=14.0,
            reliable_for_p=True,
            p_confidence_mean=0.92,
            p_candidate_context_score=0.92,
            p_onset_confidence_mean=0.95,
            p_offset_confidence_mean=0.18,
        )

        values = _p_axis_values_with_polarity_consensus(representative_leads)

        self.assertIsNone(values["aVR"])
        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )
        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 20.0)
        self.assertLess(features.p_axis_deg, 85.0)

    def test_compute_global_features_excludes_physically_inconsistent_dominant_p_axis_outlier(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, area, amp in [
            ("I", 1.25, 0.08),
            ("II", 1.50, 0.09),
            ("aVF", 1.35, 0.08),
            ("aVR", 12.50, 0.07),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=164.0,
                pr_consensus_ms=164.0,
                p_amp_mv=amp,
                p_area=area,
                reliable_for_p=True,
                p_confidence_mean=0.92,
                p_candidate_context_score=0.92,
                p_onset_confidence_mean=0.95,
                p_offset_confidence_mean=0.95,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 10.0)
        self.assertLess(features.p_axis_deg, 95.0)

    def test_p_axis_boundary_filter_preserves_borderline_sinus_like_inversion_support(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, area, amp, confidence, onset_conf, offset_conf, pr_ms in [
            ("I", 1.16, -0.036, 0.27, 0.25, 0.22, 162.0),
            ("III", 0.88, 0.017, 0.95, 0.95, 0.94, None),
            ("aVR", 14.80, 0.026, 0.55, 0.53, 0.49, 135.0),
            ("aVL", 1.02, -0.064, 0.73, 0.71, 0.68, 146.0),
            ("aVF", 0.55, -0.021, 1.00, 0.91, 0.81, 138.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=pr_ms,
                pr_consensus_ms=142.0,
                p_amp_mv=amp,
                p_area=area,
                reliable_for_p=True,
                p_confidence_mean=confidence,
                p_candidate_context_score=0.82,
                p_onset_confidence_mean=onset_conf,
                p_offset_confidence_mean=offset_conf,
            )

        values = _p_axis_values_with_polarity_consensus(representative_leads)

        self.assertIsNotNone(values["I"])
        self.assertIsNotNone(values["aVR"])
        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )
        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 0.0)
        self.assertLess(features.p_axis_deg, 80.0)

    def test_compute_global_features_preserves_supported_negative_p_axis_vector(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp in [
            ("I", -1.40, -0.08),
            ("II", -1.50, -0.09),
            ("aVF", -1.35, -0.08),
            ("aVR", 1.20, 0.07),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                p_amp_mv=amp,
                p_area=abs(signed_area),
                p_signed_area=signed_area,
                reliable_for_p=True,
                p_confidence_mean=0.92,
                p_candidate_context_score=0.92,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.p_axis_deg)
        self.assertTrue(features.p_axis_deg < -90.0 or features.p_axis_deg > 150.0)

    def test_compute_global_features_keeps_peak_polarity_when_p_axis_is_already_physical(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, area, signed_area, amp in [
            ("I", 1.40, 0.25, 0.08),
            ("II", 1.40, 0.20, 0.08),
            ("aVF", 1.40, 2.80, 0.08),
            ("aVR", 1.10, -0.20, -0.07),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                p_amp_mv=amp,
                p_area=area,
                p_signed_area=signed_area,
                reliable_for_p=True,
                p_confidence_mean=0.92,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 35.0)
        self.assertLess(features.p_axis_deg, 75.0)

    def test_compute_global_features_flips_sinus_like_global_p_axis_inversion(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, area, amp, pr_ms in [
            ("I", 1.20, -0.05, 190.0),
            ("II", 3.90, -0.07, 190.0),
            ("III", 1.00, -0.02, None),
            ("aVR", 1.30, 0.05, 182.0),
            ("aVL", 1.50, -0.03, 180.0),
            ("aVF", 0.90, -0.04, 170.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=pr_ms,
                p_amp_mv=amp,
                p_area=area,
                reliable_for_p=True,
                p_confidence_mean=0.90,
                p_candidate_context_score=0.92,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 0.0)
        self.assertLess(features.p_axis_deg, 90.0)

    def test_compute_global_features_flips_mixed_seed_sinus_like_p_axis_inversion(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, area, amp, pr_ms in [
            ("I", 1.75, 0.04, 200.0),
            ("II", 2.82, -0.09, 204.0),
            ("III", 2.14, -0.06, 205.0),
            ("aVR", 2.16, 0.07, 194.0),
            ("aVL", 1.56, 0.06, 206.0),
            ("aVF", 3.26, -0.10, 205.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=pr_ms,
                pr_consensus_ms=197.0,
                p_amp_mv=amp,
                p_area=area,
                reliable_for_p=True,
                p_confidence_mean=1.0,
                p_candidate_context_score=0.85,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 40.0)
        self.assertLess(features.p_axis_deg, 130.0)

    def test_compute_global_features_suppresses_single_lead_p_axis(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["I"] = _make_rep(
            "I",
            pr_ms=150.0,
            p_amp_mv=-0.08,
            p_area=6.0,
            reliable_for_p=True,
            p_confidence_mean=0.95,
            p_candidate_context_score=0.90,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.p_axis_deg)

    def test_compute_global_features_allows_two_lead_p_axis_with_pr_support(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, area, amp, pr_ms in [
            ("I", 1.40, 0.08, 150.0),
            ("aVF", 1.35, 0.08, 152.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                pr_ms=pr_ms,
                pr_consensus_ms=151.0,
                p_amp_mv=amp,
                p_area=area,
                reliable_for_p=True,
                p_confidence_mean=0.95,
                p_candidate_context_score=0.90,
            )
        representative_leads["II"] = _make_rep(
            "II",
            pr_ms=151.0,
            p_amp_mv=0.01,
            p_area=0.10,
            reliable_for_p=True,
            p_confidence_mean=0.95,
            p_candidate_context_score=0.90,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.p_axis_deg)
        self.assertGreater(features.p_axis_deg, 10.0)
        self.assertLess(features.p_axis_deg, 95.0)

    def test_compute_global_features_uses_peak_net_qrs_axis_for_wide_qrs_when_consistent(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, qrs_signed_area, q_amp_mv, r_amp_mv, s_amp_mv in [
            ("I", -2.25, -0.275, 0.145, 0.145),
            ("II", -7.54, -0.008, 0.116, -0.476),
            ("III", -4.87, -0.014, 0.180, -0.440),
            ("aVR", 13.24, 0.025, 0.352, 0.024),
            ("aVL", 3.14, -0.212, 0.400, 0.400),
            ("aVF", -5.79, -0.012, 0.151, -0.455),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=185.0,
                qrs_consensus_ms=185.0,
                qrs_signed_area=qrs_signed_area,
                q_amp_mv=q_amp_mv,
                r_amp_mv=r_amp_mv,
                s_amp_mv=s_amp_mv,
                reliable_for_qrs=True,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertGreater(features.qrs_axis_deg, -95.0)
        self.assertLess(features.qrs_axis_deg, -70.0)

    def test_compute_global_features_prefers_signed_t_area_for_t_axis(self) -> None:
        quality = {
            lead: _make_quality()
            for lead in STANDARD_12_LEADS
        }
        for lead_name, lead_quality in quality.items():
            lead_quality.lead = lead_name
        for lead_name in ("I", "aVF"):
            quality[lead_name] = _make_quality(
                reliable=True,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
            )
            quality[lead_name].lead = lead_name

        representatives = build_representative_lead_features(
            [
                _make_beat_feature(
                    "I",
                    0,
                    120.0,
                    360.0,
                    t_amp_mv=0.3,
                    t_area=1.0,
                    t_signed_area=1.0,
                ),
                _make_beat_feature(
                    "aVF",
                    0,
                    120.0,
                    360.0,
                    t_amp_mv=0.3,
                    t_area=5.0,
                    t_signed_area=-5.0,
                ),
            ],
            quality,
        )

        features = compute_global_features(
            representatives,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertLess(features.t_axis_deg, -45.0)

    def test_compute_global_features_reports_t_axis_when_two_high_confidence_limb_t_waves_exist(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["I"] = _make_rep(
            "I",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["aVF"] = _make_rep(
            "aVF",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.t_axis_deg)

    def test_compute_global_features_filters_positive_avr_against_stable_limb_t_seed(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp in [
            ("I", 0.50, 0.10),
            ("II", 0.50, 0.10),
            ("III", 0.40, 0.10),
            ("aVR", 2.00, 0.10),
            ("aVL", 0.20, 0.10),
            ("aVF", 0.50, 0.10),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                reliable_for_qrs=True,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                qt_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.t_axis_deg)
        self.assertGreater(features.t_axis_deg, 35.0)
        self.assertLess(features.t_axis_deg, 80.0)

    def test_compute_global_features_does_not_fallback_to_st_confused_t_axis(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["I"] = _make_rep(
            "I",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
            st_t_confusion=True,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_does_not_fallback_to_missing_t_confidence_axis(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["I"] = _make_rep(
            "I",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=None,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_does_not_fallback_to_missing_t_amplitude_axis(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["I"] = _make_rep(
            "I",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=100.0,
            t_amp_mv=None,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_does_not_fallback_to_soft_excluded_st_confused_axis(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["I"] = _make_rep(
            "I",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=100.0,
            t_amp_mv=0.20,
            t_area=2.0,
            t_signed_area=2.0,
            reliable_for_t=True,
            qt_confidence_mean=None,
            st_t_confusion=True,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_relaxed_t_axis_keeps_cluster_exclusions(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead in ("I", "II", "III", "aVF"):
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                t_amp_mv=0.04,
                t_area=1.0,
                t_signed_area=1.0,
                reliable_for_t=True,
                qt_confidence_mean=0.8 if lead == "I" else 0.35,
                st_t_confusion=(lead == "aVF"),
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_stable_limb_t_axis_keeps_cluster_exclusions(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead in ("I", "II", "aVR", "aVL", "aVF"):
            polarity = -1.0 if lead == "aVR" else 1.0
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                t_amp_mv=0.03 * polarity,
                t_area=1.0,
                t_signed_area=polarity,
                reliable_for_t=True,
                reliable_for_qrs=True,
                qrs_confidence_mean=0.8,
                qt_confidence_mean=0.8 if lead == "I" else 0.0,
                st_t_confusion=(lead == "aVF"),
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_stable_limb_t_axis_rejects_low_amplitude_leads(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead in ("I", "II", "aVR", "aVL", "aVF"):
            polarity = -1.0 if lead == "aVR" else 1.0
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                t_amp_mv=0.01 * polarity,
                t_area=1.0,
                t_signed_area=polarity,
                reliable_for_t=True,
                reliable_for_qrs=True,
                qrs_confidence_mean=0.8,
                qt_confidence_mean=0.8 if lead == "I" else 0.0,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_stable_limb_t_axis_rejects_unstable_leads(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead in ("I", "II", "aVR", "aVL", "aVF"):
            polarity = -1.0 if lead == "aVR" else 1.0
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                t_amp_mv=0.20 * polarity,
                t_area=1.0,
                t_signed_area=polarity,
                reliable_for_t=True,
                reliable_for_qrs=True,
                qrs_confidence_mean=0.8,
                qt_confidence_mean=0.8 if lead == "I" else 0.0,
            )
            representative_leads[lead].variance["t_amp_mv_sd"] = 999.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_recovers_t_axis_coverage_from_stable_low_qt_support_limb_t(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp in [
            ("I", 1.00, 0.08),
            ("II", 1.10, 0.08),
            ("aVR", -0.95, -0.07),
            ("aVL", 0.70, 0.06),
            ("aVF", 1.00, 0.07),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=360.0,
                qt_consensus_ms=360.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.18,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.t_axis_deg)
        self.assertGreater(features.t_axis_deg, 10.0)
        self.assertLess(features.t_axis_deg, 95.0)

    def test_compute_global_features_recovers_t_axis_coverage_from_three_stable_detected_limb_t_waves(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp in [
            ("I", 1.20, 0.12),
            ("II", 1.30, 0.13),
            ("aVF", 1.10, 0.11),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=96.0,
                qrs_consensus_ms=96.0,
                qt_ms=360.0,
                qt_consensus_ms=360.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.24,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.t_axis_deg)
        self.assertGreater(features.t_axis_deg, 10.0)
        self.assertLess(features.t_axis_deg, 95.0)

    def test_compute_global_features_recovers_t_axis_from_st_supported_signed_area_pair(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp, st_mid in [
            ("III", -4.60, -0.09, -0.08),
            ("aVF", -6.50, -0.09, -0.12),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=110.0,
                qrs_consensus_ms=110.0,
                qt_ms=330.0,
                qt_consensus_ms=332.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                st_mid_mv=st_mid,
                st_t_confusion=True,
                twelve_sl_st_confidence=1.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.75,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertIsNotNone(features.t_axis_deg)
        self.assertGreater(features.t_axis_deg, -120.0)
        self.assertLess(features.t_axis_deg, -40.0)

    def test_compute_global_features_keeps_t_axis_suppressed_for_low_support_unstable_t(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp in [
            ("I", 1.00, 0.015),
            ("II", 1.10, 0.015),
            ("aVR", -0.95, -0.015),
            ("aVL", 0.70, 0.015),
            ("aVF", 1.00, 0.015),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=360.0,
                qt_consensus_ms=360.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.18,
                qrs_confidence_mean=0.90,
            )
            representative_leads[lead].variance["t_amp_mv_sd"] = 0.20

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_recovers_t_axis_when_st_confusion_has_stable_st_samples(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp, st_mid, st_confused in [
            ("I", 2.80, 0.14, -0.10, True),
            ("II", 1.90, 0.16, -0.09, True),
            ("III", -1.50, -0.06, 0.03, True),
            ("aVR", 1.30, 0.13, 0.04, False),
            ("aVL", 3.10, 0.15, -0.01, False),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=96.0,
                qrs_consensus_ms=96.0,
                qt_ms=377.0,
                qt_consensus_ms=377.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                st_mid_mv=st_mid,
                st_t_confusion=st_confused,
                twelve_sl_st_confidence=1.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.35,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 500, 900, 1300], dtype=int),
            fs=500,
        )

        self.assertIsNotNone(features.t_axis_deg)
        self.assertGreater(features.t_axis_deg, -60.0)
        self.assertLess(features.t_axis_deg, 40.0)

    def test_api_backfills_t_axis_after_profile_adds_st_context(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp, st_mid, st_confused in [
            ("I", 2.80, 0.14, -0.10, True),
            ("II", 1.90, 0.16, -0.09, True),
            ("III", -1.50, -0.06, 0.03, True),
            ("aVR", 1.30, 0.13, 0.04, False),
            ("aVL", 3.10, 0.15, -0.01, False),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=96.0,
                qrs_consensus_ms=96.0,
                qt_ms=377.0,
                qt_consensus_ms=377.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                st_mid_mv=st_mid,
                st_t_confusion=st_confused,
                twelve_sl_st_confidence=1.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.35,
                qrs_confidence_mean=0.90,
            )
        global_features = GlobalFeatures(
            heart_rate_bpm=75.0,
            atrial_rate_bpm=75.0,
            pr_ms=143.0,
            qrs_ms=96.0,
            qt_ms=377.0,
            qtc_bazett_ms=421.5,
            qtc_fridericia_ms=406.1,
            p_axis_deg=71.3,
            qrs_axis_deg=9.4,
            t_axis_deg=None,
            st_axis_deg=None,
            qt_dispersion_ms=39.0,
            qt_source="low_qt_support_fallback",
            qt_reliability="low_confidence",
            qt_confidence_reason="insufficient_reliable_qt_leads",
        )

        changed = _measurement_mod._backfill_t_axis_after_measurement_profile(
            global_features,
            representative_leads,
            np.asarray([100, 500, 900, 1300], dtype=int),
            fs=500,
        )

        self.assertTrue(changed)
        self.assertIsNotNone(global_features.t_axis_deg)
        self.assertGreater(global_features.t_axis_deg, -60.0)
        self.assertLess(global_features.t_axis_deg, 40.0)

    def test_api_backfill_keeps_t_axis_suppressed_without_st_context(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp, st_mid, st_confused in [
            ("I", 2.80, 0.14, -0.10, True),
            ("II", 1.90, 0.16, -0.09, True),
            ("III", -1.50, -0.06, 0.03, True),
            ("aVR", 1.30, 0.13, 0.04, False),
            ("aVL", 3.10, 0.15, -0.01, False),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=96.0,
                qrs_consensus_ms=96.0,
                qt_ms=377.0,
                qt_consensus_ms=377.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                st_mid_mv=st_mid,
                st_t_confusion=st_confused,
                twelve_sl_st_confidence=0.40 if st_confused else 1.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.35,
                qrs_confidence_mean=0.90,
            )
        global_features = GlobalFeatures(
            heart_rate_bpm=75.0,
            atrial_rate_bpm=75.0,
            pr_ms=143.0,
            qrs_ms=96.0,
            qt_ms=377.0,
            qtc_bazett_ms=421.5,
            qtc_fridericia_ms=406.1,
            p_axis_deg=71.3,
            qrs_axis_deg=9.4,
            t_axis_deg=None,
            st_axis_deg=None,
            qt_dispersion_ms=39.0,
            qt_source="low_qt_support_fallback",
            qt_reliability="low_confidence",
            qt_confidence_reason="insufficient_reliable_qt_leads",
        )

        changed = _measurement_mod._backfill_t_axis_after_measurement_profile(
            global_features,
            representative_leads,
            np.asarray([100, 500, 900, 1300], dtype=int),
            fs=500,
        )

        self.assertFalse(changed)
        self.assertIsNone(global_features.t_axis_deg)

    def test_api_backfill_recovers_st_supported_t_axis_after_profile_adds_st_context(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, signed_area, amp, st_mid in [
            ("III", -4.60, -0.09, -0.08),
            ("aVF", -6.50, -0.09, -0.12),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=110.0,
                qrs_consensus_ms=110.0,
                qt_ms=330.0,
                qt_consensus_ms=332.0,
                t_amp_mv=amp,
                t_area=abs(signed_area),
                t_signed_area=signed_area,
                st_mid_mv=st_mid,
                st_t_confusion=True,
                twelve_sl_st_confidence=1.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.75,
                qrs_confidence_mean=0.90,
            )
        global_features = GlobalFeatures(
            heart_rate_bpm=80.0,
            atrial_rate_bpm=80.0,
            pr_ms=143.0,
            qrs_ms=110.0,
            qt_ms=332.0,
            qtc_bazett_ms=383.0,
            qtc_fridericia_ms=365.0,
            p_axis_deg=60.0,
            qrs_axis_deg=-10.0,
            t_axis_deg=None,
            st_axis_deg=None,
            qt_dispersion_ms=20.0,
            qt_source="reliable_raw_qt_cluster_rescue",
            qt_reliability="rescued",
            qt_confidence_reason=None,
        )

        changed = _measurement_mod._backfill_t_axis_after_measurement_profile(
            global_features,
            representative_leads,
            np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertTrue(changed)
        self.assertIsNotNone(global_features.t_axis_deg)
        self.assertGreater(global_features.t_axis_deg, -120.0)
        self.assertLess(global_features.t_axis_deg, -40.0)

    def test_compute_global_features_uses_st_corrected_t_axis_for_wide_qrs_plateau(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        wide_qrs_ms = 196.0
        for lead, t_signed_area, t_amp_mv, st_mid_mv in [
            ("I", -5.03, -0.199, -0.196),
            ("II", 0.93, 0.212, -0.210),
            ("III", 4.92, 0.155, 0.135),
            ("aVR", -0.92, -0.177, 0.215),
            ("aVL", 0.56, 0.706, 0.698),
            ("aVF", 4.87, 0.133, -0.026),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=wide_qrs_ms,
                qrs_consensus_ms=wide_qrs_ms,
                t_amp_mv=t_amp_mv,
                t_area=abs(t_signed_area),
                t_signed_area=t_signed_area,
                st_mid_mv=st_mid_mv,
                reliable_for_qrs=True,
                reliable_for_t=True,
                qt_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertIsNotNone(features.t_axis_deg)
        self.assertGreater(features.t_axis_deg, 20.0)
        self.assertLess(features.t_axis_deg, 80.0)

    def test_compute_global_features_rescues_qt_from_late_tail_raw_pair(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms in [
            ("I", 318.0),
            ("II", 322.0),
            ("III", 320.0),
            ("aVR", 324.0),
            ("aVF", 319.0),
            ("V5", 394.0),
            ("V6", 402.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=90.0,
                qrs_consensus_ms=90.0,
                qt_ms=qt_ms,
                qt_consensus_ms=320.0,
                t_amp_mv=0.08,
                t_area=2.0,
                t_signed_area=2.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.70,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertEqual("tail_confirmed_late_raw_qt_rescue", features.qt_source)
        self.assertAlmostEqual(398.0, features.qt_ms)
        self.assertEqual(["V5", "V6"], features.qt_used_leads)

    def test_compute_global_features_rescues_low_support_qt_from_late_tail_raw_pair(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms in [
            ("I", 320.0),
            ("II", 322.0),
            ("III", 318.0),
            ("V5", 394.0),
            ("V6", 402.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=90.0,
                qrs_consensus_ms=90.0,
                qt_ms=qt_ms,
                qt_consensus_ms=320.0,
                t_amp_mv=0.08,
                t_area=2.0,
                t_signed_area=2.0,
                reliable_for_t=True,
                reliable_for_qt=False,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.70,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertEqual("tail_confirmed_late_raw_qt_rescue", features.qt_source)
        self.assertAlmostEqual(398.0, features.qt_ms)
        self.assertEqual(["V5", "V6"], features.qt_used_leads)

    def test_compute_global_features_does_not_overextend_qt_from_small_very_late_tail_cluster(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms, t_amp_mv in [
            ("I", 445.0, 0.08),
            ("II", 477.0, 0.08),
            ("III", 541.0, 0.08),
            ("aVR", 452.0, 0.08),
            ("aVL", 407.0, 0.08),
            ("aVF", 548.0, 0.08),
            ("V1", 506.0, 0.01),
            ("V2", 417.0, 0.08),
            ("V3", 402.0, 0.08),
            ("V4", 525.0, 0.08),
            ("V5", 427.0, 0.08),
            ("V6", 411.0, 0.08),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=qt_ms,
                qt_consensus_ms=441.0,
                t_amp_mv=t_amp_mv,
                t_area=2.0,
                t_signed_area=2.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.70,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertNotEqual("tail_confirmed_late_raw_qt_rescue", features.qt_source)
        self.assertLess(features.qt_ms, 500.0)

    def test_compute_global_features_does_not_rescue_qt_from_weak_precordial_tail_pair(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms, t_amp_mv in [
            ("I", 336.0, 0.08),
            ("II", 374.0, 0.08),
            ("III", 338.0, 0.08),
            ("V1", 442.0, 0.08),
            ("V2", 457.0, 0.023),
            ("V3", 422.0, 0.08),
            ("V4", 397.0, 0.08),
            ("V5", 379.0, 0.08),
            ("V6", 478.0, 0.081),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=92.0,
                qrs_consensus_ms=92.0,
                qt_ms=qt_ms,
                qt_consensus_ms=392.0,
                t_amp_mv=t_amp_mv,
                t_area=2.0,
                t_signed_area=2.0,
                reliable_for_t=True,
                reliable_for_qt=False,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.70,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertNotEqual("tail_confirmed_late_raw_qt_rescue", features.qt_source)
        self.assertLess(features.qt_ms, 430.0)

    def test_compute_global_features_does_not_rescue_qt_from_weak_precordial_tail_triplet(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms, t_amp_mv in [
            ("I", 336.0, 0.08),
            ("II", 374.0, 0.08),
            ("III", 338.0, 0.08),
            ("V1", 442.0, 0.041),
            ("V2", 457.0, 0.023),
            ("V3", 422.0, 0.08),
            ("V4", 397.0, 0.08),
            ("V5", 379.0, 0.08),
            ("V6", 470.0, 0.081),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=92.0,
                qrs_consensus_ms=92.0,
                qt_ms=qt_ms,
                qt_consensus_ms=392.0,
                t_amp_mv=t_amp_mv,
                t_area=2.0,
                t_signed_area=2.0,
                reliable_for_t=True,
                reliable_for_qt=False,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.70,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertNotEqual("tail_confirmed_late_raw_qt_rescue", features.qt_source)
        self.assertLess(features.qt_ms, 430.0)

    def test_compute_global_features_allows_moderate_precordial_tail_pair_with_one_low_amp_lead(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms, t_amp_mv in [
            ("aVR", 352.0, -0.10),
            ("aVF", 351.0, -0.13),
            ("V2", 434.0, 0.024),
            ("V6", 413.0, -0.069),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=83.0,
                qrs_consensus_ms=83.0,
                qt_ms=qt_ms,
                qt_consensus_ms=365.0,
                t_amp_mv=t_amp_mv,
                t_area=2.0,
                t_signed_area=t_amp_mv * 10.0,
                reliable_for_t=True,
                reliable_for_qt=False,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.80,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertEqual("tail_confirmed_late_raw_qt_rescue", features.qt_source)
        self.assertAlmostEqual(423.5, features.qt_ms)
        self.assertEqual(["V2", "V6"], features.qt_used_leads)

    def test_compute_global_features_allows_late_tail_cluster_with_one_near_delta_lead(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms, t_amp_mv in [
            ("aVR", 352.0, -0.10),
            ("aVF", 351.0, -0.13),
            ("V2", 434.0, 0.024),
            ("V6", 414.0, -0.069),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=83.0,
                qrs_consensus_ms=83.0,
                qt_ms=qt_ms,
                qt_consensus_ms=370.0,
                t_amp_mv=t_amp_mv,
                t_area=2.0,
                t_signed_area=t_amp_mv * 10.0,
                reliable_for_t=True,
                reliable_for_qt=False,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=0.80,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 850, 1600, 2350], dtype=int),
            fs=1000,
        )

        self.assertEqual("tail_confirmed_late_raw_qt_rescue", features.qt_source)
        self.assertAlmostEqual(424.0, features.qt_ms)
        self.assertEqual(["V2", "V6"], features.qt_used_leads)

    def test_compute_global_features_rescues_overlong_wide_qrs_qt_from_raw_lower_core(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms, t_amp_mv, confidence, st_t_confusion in [
            ("I", 536.0, -0.257, 0.51, False),
            ("II", 492.0, -0.094, 0.85, True),
            ("III", 468.0, -0.046, 0.75, False),
            ("aVR", 508.0, 0.199, 0.54, False),
            ("aVL", 520.0, -0.302, 0.60, False),
            ("aVF", 504.0, -0.037, 0.86, False),
            ("V1", 510.0, 0.194, 0.59, False),
            ("V2", 592.0, -0.091, 0.29, True),
            ("V3", 570.0, -0.133, 0.39, False),
            ("V4", 496.0, -0.595, 0.36, False),
            ("V5", 566.0, -0.249, 0.45, False),
            ("V6", 540.0, -0.189, 0.49, False),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=172.0,
                qrs_consensus_ms=172.0,
                qt_ms=qt_ms,
                qt_consensus_ms=530.0,
                t_amp_mv=t_amp_mv,
                t_area=2.0,
                t_signed_area=t_amp_mv * 10.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                st_t_confusion=st_t_confusion,
                qt_confidence_mean=confidence,
                qrs_confidence_mean=0.90,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1290, 2480, 3670], dtype=int),
            fs=1000,
        )

        self.assertEqual("wide_qrs_long_raw_qt_core_rescue", features.qt_source)
        self.assertLess(features.qt_ms, 510.0)
        self.assertIn("III", features.qt_used_leads)
        self.assertIn("V4", features.qt_used_leads)

    def test_compute_global_features_keeps_wide_qrs_raw_qt_rescue_from_irregular_core_reshortening(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, qt_ms, t_amp_mv, confidence, qt_sd in [
            ("I", 391.0, -0.048, 0.98, 42.0),
            ("II", 390.0, -0.031, 0.94, 11.0),
            ("III", 345.0, -0.016, 0.88, 43.0),
            ("aVR", 401.0, -0.026, 0.96, 15.0),
            ("aVL", 390.0, -0.031, 0.93, 14.0),
            ("aVF", 349.0, -0.033, 0.97, 21.0),
            ("V1", 425.0, 0.035, 0.75, 51.0),
            ("V2", 418.0, 0.021, 0.47, 72.0),
            ("V3", 465.0, 0.056, 0.51, 14.0),
            ("V4", 396.0, 0.033, 0.78, 37.0),
            ("V5", 397.0, -0.047, 0.99, 5.0),
            ("V6", 428.0, -0.056, 0.69, 11.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=126.0,
                qrs_consensus_ms=126.0,
                qt_ms=qt_ms,
                qt_consensus_ms=410.0,
                t_amp_mv=t_amp_mv,
                t_area=2.0,
                t_signed_area=t_amp_mv * 10.0,
                reliable_for_t=True,
                reliable_for_qt=True,
                reliable_for_global=True,
                reliable_for_qrs=True,
                qt_confidence_mean=confidence,
                qrs_confidence_mean=0.90,
            )
            representative_leads[lead].variance["qt_ms_sd"] = qt_sd

        features = compute_global_features(
            representative_leads,
            beat_features=[
                _make_beat_feature(
                    "V3",
                    beat_id,
                    pr_ms=160.0,
                    qrs_ms=126.0,
                    qt_ms=418.0,
                    qt_consensus_ms=418.0,
                    t_amp_mv=0.06,
                )
                for beat_id in range(8)
            ],
            r_locs=np.asarray([640, 1241, 1842, 2171, 2512, 3218, 3818, 4419], dtype=int),
            fs=500,
        )

        self.assertEqual("wide_qrs_raw_qt_rescue", features.qt_source)
        self.assertAlmostEqual(446.5, features.qt_ms)
        self.assertEqual(["V3", "V6"], features.qt_used_leads)

    def test_compute_global_features_excludes_polarity_conflicted_st_corrected_t_leads(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, t_signed_area, t_amp_mv, st_mid_mv in [
            ("I", -5.03, -0.199, -0.196),
            ("II", 0.93, 0.212, -0.210),
            ("III", 4.92, 0.155, 0.135),
            ("aVR", -0.92, -0.177, 0.215),
            ("aVL", 0.56, 0.706, 0.698),
            ("aVF", 4.87, -0.631, -0.026),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=196.0,
                qrs_consensus_ms=196.0,
                t_amp_mv=t_amp_mv,
                t_area=abs(t_signed_area),
                t_signed_area=t_signed_area,
                st_mid_mv=st_mid_mv,
                reliable_for_qrs=True,
                reliable_for_t=True,
                qt_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertGreater(features.t_axis_deg, 20.0)
        self.assertLess(features.t_axis_deg, 80.0)

    def test_compute_global_features_keeps_signed_t_axis_for_moderately_wide_qrs_when_st_correction_conflicts(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, t_signed_area, t_amp_mv, st_mid_mv, qt_confidence in [
            ("I", -5.79, -0.093, -0.029, 0.19),
            ("II", -3.46, 0.036, -0.201, 0.87),
            ("III", 3.18, 0.107, -0.085, 0.95),
            ("aVR", 19.04, 0.226, 0.219, 0.46),
            ("aVL", -4.31, -0.104, 0.035, 0.88),
            ("aVF", 0.22, 0.077, -0.138, 0.99),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=133.0,
                qrs_consensus_ms=133.0,
                t_amp_mv=t_amp_mv,
                t_area=abs(t_signed_area),
                t_signed_area=t_signed_area,
                st_mid_mv=st_mid_mv,
                reliable_for_qrs=True,
                reliable_for_t=True,
                qt_confidence_mean=qt_confidence,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertIsNotNone(features.t_axis_deg)
        self.assertTrue(features.t_axis_deg > 140.0 or features.t_axis_deg < -140.0)

    def test_compute_global_features_suppresses_t_axis_when_wide_qrs_t_is_st_plateau_only(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, t_signed_area, t_amp_mv, st_mid_mv in [
            ("I", -4.39, -0.077, -0.064),
            ("II", -2.41, -0.056, -0.047),
            ("III", 0.43, 0.024, -0.010),
            ("aVR", 4.76, 0.093, 0.071),
            ("aVL", -1.84, -0.036, -0.024),
            ("aVF", 0.51, 0.027, -0.017),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=180.0,
                qrs_consensus_ms=180.0,
                qrs_signed_area=1.0,
                t_amp_mv=t_amp_mv,
                t_area=abs(t_signed_area),
                t_signed_area=t_signed_area,
                st_mid_mv=st_mid_mv,
                reliable_for_qrs=True,
                reliable_for_t=True,
                qt_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_suppresses_t_axis_when_limb_t_confidence_is_weak(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        for lead, t_signed_area, t_amp_mv, st_mid_mv, qt_confidence in [
            ("I", -3.44, -0.353, -0.047, 0.34),
            ("II", 0.57, 0.249, -0.043, 0.57),
            ("III", 3.57, 0.410, 0.014, 0.39),
            ("aVR", 0.45, -0.196, 0.094, 0.52),
            ("aVL", -4.77, -0.431, -0.059, 0.36),
            ("aVF", 2.10, 0.346, -0.012, 0.58),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=108.0,
                qrs_consensus_ms=108.0,
                t_amp_mv=t_amp_mv,
                t_area=abs(t_signed_area),
                t_signed_area=t_signed_area,
                st_mid_mv=st_mid_mv,
                reliable_for_qrs=True,
                reliable_for_t=True,
                qt_confidence_mean=qt_confidence,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.t_axis_deg)

    def test_compute_global_features_suppresses_t_axis_for_irregular_low_p_support_rhythm(self) -> None:
        # Simulate AF-like irregular rhythm: all 12 leads have sub-threshold P-amplitude
        # (< _P_MIN_ABS_AMP_MV = 0.02 mV) → 0 leads with P-support → below threshold=3 → suppress.
        representative_leads = {
            lead: _make_rep(lead, p_amp_mv=0.01, reliable_for_p=True, p_confidence_mean=0.9)
            for lead in STANDARD_12_LEADS
        }
        for lead, t_signed_area, t_amp_mv in [
            ("I", 8.13, 0.452),
            ("II", 2.69, 0.298),
            ("III", -5.48, -0.199),
            ("aVR", -5.61, -0.402),
            ("aVL", 8.23, 0.448),
            ("aVF", 0.61, 0.180),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=104.0,
                qrs_consensus_ms=104.0,
                p_amp_mv=0.01,
                t_amp_mv=t_amp_mv,
                t_area=abs(t_signed_area),
                t_signed_area=t_signed_area,
                st_mid_mv=0.0,
                reliable_for_p=True,
                reliable_for_qrs=True,
                reliable_for_t=True,
                qt_confidence_mean=0.7,
                p_confidence_mean=0.9,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 410, 990, 1250, 1900, 2180], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.t_axis_deg)

    @needs_interpretation
    def test_rr_irregularity_treats_absent_reliable_p_candidates_as_probable_af(self) -> None:
        from feature_extraction.ecgfeat.interpret import _rr_irregularity

        representative_leads = {
            lead: _make_rep(lead, reliable_for_p=False, p_confidence_mean=0.0)
            for lead in STANDARD_12_LEADS
        }
        beats = [
            BeatAnnotation(
                beat_id=idx,
                r_index=idx * 500,
                paced=False,
                group_id=1,
                rr_prev_ms=None,
                rr_next_ms=rr_ms,
            )
            for idx, rr_ms in enumerate([540.0, 920.0, 610.0, 860.0, 580.0, 940.0])
        ]

        _rr_cv, rr_class, probable_af = _rr_irregularity(beats, representative_leads)

        self.assertEqual("irregular", rr_class)
        self.assertTrue(probable_af)

    @needs_interpretation
    def test_measurement_availability_masks_p_morphology_outputs(self) -> None:
        from feature_extraction.ecgfeat.interpret import _apply_measurement_availability_to_interpretation

        interpretation = SimpleNamespace(
            rr_irregularity_class="irregular",
            probable_af=False,
            pr_class="avb1",
            avb_grade=1,
            p_axis_normal=True,
            p_morphology_class="probable_lae",
            rae_leads=["II"],
            lae_suspected=True,
            lae_definite=False,
            ptf_v1_class="normal",
        )

        _apply_measurement_availability_to_interpretation(
            interpretation,
            {
                "atrial_rhythm_available": False,
                "pr_available": False,
                "p_axis_available": False,
                "reasons": ["atrial_measurements_unavailable"],
            },
        )

        self.assertEqual("indeterminate", interpretation.pr_class)
        self.assertIsNone(interpretation.avb_grade)
        self.assertIsNone(interpretation.p_axis_normal)
        self.assertIsNone(interpretation.p_morphology_class)
        self.assertEqual([], interpretation.rae_leads)
        self.assertFalse(interpretation.lae_suspected)
        self.assertFalse(interpretation.lae_definite)
        self.assertIsNone(interpretation.ptf_v1_class)
        self.assertTrue(interpretation.probable_af)

    def test_api_flags_invalid_atrial_measurements_for_availability_when_irregular(self) -> None:
        global_features = SimpleNamespace(pr_ms=None, p_axis_deg=None)

        self.assertTrue(
            _applicability_mod._atrial_measurements_invalid_for_availability(
                global_features,
                {"rr_cv": 0.22},
            )
        )

    def test_select_reliable_qt_leads_rejects_low_t_amplitude_and_qt_outlier(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            qt_ms=400.0,
            qt_consensus_ms=400.0,
            t_amp_mv=0.20,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["V5"] = _make_rep(
            "V5",
            qt_ms=404.0,
            qt_consensus_ms=404.0,
            t_amp_mv=0.22,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["aVF"] = _make_rep(
            "aVF",
            qt_ms=402.0,
            qt_consensus_ms=402.0,
            t_amp_mv=0.01,
            reliable_for_qt=True,
            qt_confidence_mean=0.9,
        )
        representative_leads["V6"] = _make_rep(
            "V6",
            qt_ms=560.0,
            qt_consensus_ms=560.0,
            t_amp_mv=0.25,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
        )

        self.assertEqual(["II", "V5"], _select_reliable_qt_leads(representative_leads))

    def test_select_reliable_qt_leads_respects_custom_confidence_threshold(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            qt_ms=400.0,
            qt_consensus_ms=400.0,
            t_amp_mv=0.20,
            reliable_for_qt=True,
            qt_confidence_mean=0.30,
        )
        representative_leads["V5"] = _make_rep(
            "V5",
            qt_ms=404.0,
            qt_consensus_ms=404.0,
            t_amp_mv=0.22,
            reliable_for_qt=True,
            qt_confidence_mean=0.70,
        )

        self.assertEqual(
            ["V5"],
            _select_reliable_qt_leads(
                representative_leads,
                qt_confidence_threshold=0.50,
            ),
        )

    def test_compute_global_features_rejects_short_jt_in_wide_qrs_mode(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=160.0,
            qrs_consensus_ms=160.0,
            qt_ms=300.0,
            qt_consensus_ms=300.0,
            jt_ms=140.0,
            t_amp_mv=0.20,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
        )
        representative_leads["V5"] = _make_rep(
            "V5",
            qrs_ms=160.0,
            qrs_consensus_ms=160.0,
            qt_ms=430.0,
            qt_consensus_ms=430.0,
            jt_ms=270.0,
            t_amp_mv=0.22,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.8,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertEqual(430.0, features.qt_ms)
        self.assertEqual(430.0, features.qtc_bazett_ms)
        self.assertEqual(430.0, features.qtc_fridericia_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("low_confidence", features.qt_reliability)
        self.assertEqual("low_qt_support_fallback", features.qt_source)
        self.assertEqual(["V5"], features.qt_used_leads)
        self.assertEqual("qt_outside_path_bounds", features.qt_excluded_leads["II"])
        self.assertIn("V5", features.qt_lead_weights)
        self.assertEqual(300.0, features.consensus_vs_independent_per_lead["II"]["qt_ms"])
        self.assertEqual(300.0, features.consensus_vs_independent_per_lead["II"]["qt_consensus_ms"])
        self.assertEqual(430.0, features.consensus_vs_independent_per_lead["V5"]["qt_ms"])
        self.assertEqual(430.0, features.consensus_vs_independent_per_lead["V5"]["qt_consensus_ms"])

    def test_compute_global_features_prefers_raw_reliable_qt_when_wide_qrs_consensus_is_early(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["V2"] = _make_rep(
            "V2",
            qrs_ms=186.0,
            qrs_consensus_ms=186.0,
            qt_ms=454.0,
            qt_consensus_ms=436.0,
            t_amp_mv=0.16,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.9,
        )
        representative_leads["V4"] = _make_rep(
            "V4",
            qrs_ms=186.0,
            qrs_consensus_ms=186.0,
            qt_ms=460.0,
            qt_consensus_ms=436.0,
            t_amp_mv=0.18,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.9,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1123], dtype=int),
            fs=1000,
        )

        self.assertEqual(457.0, features.qt_ms)
        self.assertEqual("wide_qrs_raw_qt_rescue", features.qt_source)
        self.assertEqual(["V2", "V4"], features.qt_used_leads)
        self.assertEqual("rescued", features.qt_reliability)

    def test_compute_global_features_uses_late_raw_qt_cluster_when_consensus_is_short(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, raw_qt in [
            ("I", 412.0),
            ("II", 413.0),
            ("III", 414.0),
            ("aVF", 411.0),
            ("V5", 415.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=raw_qt,
                qt_consensus_ms=375.0,
                t_amp_mv=0.12,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                reliable_for_t=True,
                qt_confidence_mean=0.85,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(413.0, features.qt_ms)
        self.assertEqual("late_raw_qt_cluster_rescue", features.qt_source)
        self.assertEqual("rescued", features.qt_reliability)
        self.assertEqual(["I", "II", "III", "aVF", "V5"], features.qt_used_leads)

    def test_compute_global_features_keeps_short_consensus_when_late_raw_qt_support_is_weak(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, raw_qt, conf in [
            ("I", 412.0, 0.85),
            ("II", 413.0, 0.85),
            ("V5", 455.0, 0.15),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=raw_qt,
                qt_consensus_ms=375.0,
                t_amp_mv=0.12,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                reliable_for_t=True,
                qt_confidence_mean=conf,
                qrs_confidence_mean=0.9,
            )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100], dtype=int),
            fs=1000,
        )

        self.assertEqual(375.0, features.qt_ms)
        self.assertNotEqual("late_raw_qt_cluster_rescue", features.qt_source)

    def test_compute_global_features_uses_low_support_late_raw_qt_cluster_when_consensus_is_short(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, raw_qt in [
            ("I", 414.0),
            ("II", 412.0),
            ("aVF", 416.0),
            ("V5", 413.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=raw_qt,
                qt_consensus_ms=330.0,
                t_amp_mv=0.06,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                reliable_for_t=True,
                qt_confidence_mean=0.18,
                qrs_confidence_mean=0.90,
            )
            representative_leads[lead].variance["qt_ms_sd"] = 35.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertEqual(413.5, features.qt_ms)
        self.assertEqual("low_support_late_raw_qt_cluster_rescue", features.qt_source)
        self.assertEqual("rescued", features.qt_reliability)
        self.assertEqual(["I", "II", "aVF", "V5"], features.qt_used_leads)

    def test_compute_global_features_ignores_low_support_late_raw_qt_cluster_when_spread_is_wide(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, raw_qt in [
            ("I", 410.0),
            ("II", 442.0),
            ("aVF", 480.0),
            ("V5", 520.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=100.0,
                qrs_consensus_ms=100.0,
                qt_ms=raw_qt,
                qt_consensus_ms=330.0,
                t_amp_mv=0.06,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                reliable_for_t=True,
                qt_confidence_mean=0.18,
                qrs_confidence_mean=0.90,
            )
            representative_leads[lead].variance["qt_ms_sd"] = 35.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1100, 2100, 3100], dtype=int),
            fs=1000,
        )

        self.assertEqual(330.0, features.qt_ms)
        self.assertNotEqual("low_support_late_raw_qt_cluster_rescue", features.qt_source)

    def test_compute_global_features_uses_paced_raw_qt_when_low_support_consensus_is_early(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        for lead, raw_qt, t_amp, conf, qt_sd in [
            ("I", 466.0, -0.07, 0.19, 12.0),
            ("II", 418.0, -0.07, 0.00, 80.0),
            ("III", 422.0, 0.05, 0.00, 80.0),
            ("aVL", 247.0, -0.04, 0.16, 7.0),
            ("aVF", 456.0, 0.07, 0.00, 80.0),
            ("V1", 500.0, 0.07, 0.15, 11.0),
            ("V2", 496.0, 0.15, 0.09, 9.0),
            ("V3", 392.0, 0.16, 0.83, 5.0),
            ("V4", 431.0, 0.18, 1.00, 43.0),
            ("V5", 494.0, 0.08, 0.00, 80.0),
            ("V6", 452.0, -0.06, 0.00, 80.0),
        ]:
            representative_leads[lead] = _make_rep(
                lead,
                qrs_ms=186.0,
                qrs_consensus_ms=198.0,
                qt_ms=raw_qt,
                qt_consensus_ms=408.0,
                t_amp_mv=t_amp,
                reliable_for_global=True,
                reliable_for_qrs=True,
                reliable_for_qt=True,
                qt_confidence_mean=conf,
                qrs_confidence_mean=0.9,
            )
            representative_leads[lead].variance["qt_ms_sd"] = qt_sd

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600, 1100, 1600], dtype=int),
            fs=500,
            paced=True,
        )

        self.assertEqual(456.0, features.qt_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("paced_low_support_raw_qt_fallback", features.qt_source)
        self.assertEqual("low_confidence", features.qt_reliability)

    def test_compute_global_features_marks_low_qt_support_with_provenance(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=100.0,
            qt_ms=420.0,
            t_amp_mv=0.20,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            reliable_for_global=True,
            qt_confidence_mean=0.8,
            qrs_confidence_mean=0.9,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("low_confidence", features.qt_reliability)
        self.assertEqual(["II"], features.qt_used_leads)
        self.assertEqual(420.0, features.qt_ms)
        self.assertEqual(420.0, features.qtc_bazett_ms)
        self.assertEqual(420.0, features.qtc_fridericia_ms)
        self.assertEqual("low_qt_support_fallback", features.qt_source)
        self.assertEqual("insufficient_reliable_qt_leads", features.qt_confidence_reason)
        self.assertEqual({}, features.qt_excluded_leads)
        self.assertIn("II", features.qt_lead_weights)
        self.assertEqual(420.0, features.consensus_vs_independent_per_lead["II"]["qt_ms"])
        self.assertIn("qt_consensus_ms", features.consensus_vs_independent_per_lead["II"])

    def test_compute_global_features_preserves_wide_qrs_single_lead_qt_as_low_confidence(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        representative_leads["III"] = _make_rep(
            "III",
            qrs_ms=138.0,
            qrs_consensus_ms=138.0,
            qrs_wide_ms=166.0,
            qt_ms=538.0,
            qt_consensus_ms=504.0,
            t_amp_mv=0.06,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=True,
            qt_confidence_mean=0.31,
            qrs_confidence_mean=0.9,
        )
        representative_leads["III"].variance["qt_ms_sd"] = 22.0

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 1000, 1900, 2800], dtype=int),
            fs=500,
        )

        self.assertEqual(504.0, features.qt_ms)
        self.assertEqual("low_qt_support", features.qt_path)
        self.assertEqual("wide_qrs_low_support_qt_fallback", features.qt_source)
        self.assertEqual("low_confidence", features.qt_reliability)
        self.assertEqual(["III"], features.qt_used_leads)

    def test_compute_global_features_does_not_rescue_short_jt_fallback_in_wide_qrs_mode(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=170.0,
            qrs_consensus_ms=170.0,
            qt_ms=300.0,
            jt_ms=130.0,
            t_amp_mv=0.20,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=False,
            qt_confidence_mean=0.0,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.qt_ms)

    def test_compute_global_features_uses_global_qrs_for_wide_qrs_jt_guard(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            qrs_ms=216.0,
            qrs_consensus_ms=216.0,
            qt_ms=313.0,
            jt_ms=200.0,
            t_amp_mv=0.20,
            reliable_for_global=True,
            reliable_for_qrs=True,
            reliable_for_qt=False,
            qt_confidence_mean=0.0,
        )

        features = compute_global_features(
            representative_leads,
            beat_features=[],
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.qt_ms)

    def test_compute_global_features_suppresses_pr_and_p_axis_when_all_beats_are_paced(self) -> None:
        representative_leads = {
            lead: _make_rep(lead)
            for lead in STANDARD_12_LEADS
        }
        representative_leads["II"] = _make_rep(
            "II",
            pr_ms=160.0,
            p_amp_mv=0.20,
            p_area=1.0,
            reliable_for_global=True,
            reliable_for_p=True,
            p_confidence_mean=0.9,
        )
        representative_leads["aVF"] = _make_rep(
            "aVF",
            pr_ms=162.0,
            p_amp_mv=0.20,
            p_area=1.0,
            reliable_for_global=True,
            reliable_for_p=True,
            p_confidence_mean=0.9,
        )
        beat_features = [
            _make_beat_feature("II", 0, 160.0, 420.0),
            _make_beat_feature("II", 1, 162.0, 422.0),
        ]
        for feature in beat_features:
            feature.flags.append("paced_beat")

        features = compute_global_features(
            representative_leads,
            beat_features=beat_features,
            r_locs=np.asarray([100, 600], dtype=int),
            fs=500,
        )

        self.assertIsNone(features.pr_ms)
        self.assertIsNone(features.p_axis_deg)

    def test_select_measurement_group_prefers_non_paced_beats_when_present(self) -> None:
        group_id, beat_ids = _beats_mod._select_measurement_group(
            beat_groups={1: [0], 2: [1, 2, 3]},
            paced_beat_ids=[0],
        )

        self.assertEqual(2, group_id)
        self.assertEqual([1, 2, 3], beat_ids)

    def test_select_measurement_group_prefers_paced_group_when_pacing_is_majority(self) -> None:
        group_id, beat_ids = _beats_mod._select_measurement_group(
            beat_groups={1: [1, 2, 4], 2: [0, 3]},
            paced_beat_ids=[1, 2, 4],
        )

        self.assertEqual(1, group_id)
        self.assertEqual([1, 2, 4], beat_ids)

    def test_dominant_paced_wide_qrs_override_uses_group_width_for_nearby_overwide_consensus(self) -> None:
        groups = {
            1: SimpleNamespace(mean_qrs_ms=186.0, flags={"dominant_group": True, "wide_qrs": True}),
            2: SimpleNamespace(mean_qrs_ms=113.0, flags={"dominant_group": False, "wide_qrs": False}),
        }

        self.assertEqual(
            186.0,
            _pacing_mod._dominant_paced_wide_qrs_override_ms(198.0, groups),
        )

    def test_dominant_paced_wide_qrs_override_ignores_large_group_consensus_gap(self) -> None:
        groups = {
            1: SimpleNamespace(mean_qrs_ms=125.0, flags={"dominant_group": True, "wide_qrs": True}),
            2: SimpleNamespace(mean_qrs_ms=166.0, flags={"dominant_group": False, "wide_qrs": True}),
        }

        self.assertIsNone(_pacing_mod._dominant_paced_wide_qrs_override_ms(180.0, groups))

    def test_intermittent_paced_wide_qrs_override_blends_wide_consensus_and_wide_group(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=80.0,
                qrs_consensus_ms=104.0,
                qrs_wide_ms=144.0,
                reliable_for_qrs=True,
            )
            for lead in STANDARD_12_LEADS
        }
        groups = {
            1: SimpleNamespace(mean_qrs_ms=None, flags={"dominant_group": True, "wide_qrs": False}),
            2: SimpleNamespace(mean_qrs_ms=82.0, flags={"dominant_group": False, "wide_qrs": False}),
            3: SimpleNamespace(mean_qrs_ms=212.0, flags={"dominant_group": False, "wide_qrs": True}),
        }

        self.assertEqual(
            178.0,
            _pacing_mod._intermittent_paced_wide_qrs_override_ms(104.0, representative_leads, groups),
        )

    def test_borderline_paced_qrs_override_uses_stable_wide_offset(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=120.0,
                qrs_wide_ms=132.0,
                reliable_for_qrs=True,
            )
            for lead in STANDARD_12_LEADS
        }

        self.assertEqual(
            132.0,
            _pacing_mod._borderline_paced_qrs_wide_offset_override_ms(120.0, representative_leads),
        )

    def test_secondary_paced_wide_group_override_rejects_overextended_consensus(self) -> None:
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=120.0,
                qrs_wide_ms=208.0,
                reliable_for_qrs=True,
            )
            for lead in STANDARD_12_LEADS
        }
        groups = {
            1: SimpleNamespace(mean_qrs_ms=125.0, flags={"dominant_group": True, "wide_qrs": True}),
            2: SimpleNamespace(mean_qrs_ms=166.0, flags={"dominant_group": False, "wide_qrs": True}),
        }

        self.assertEqual(
            166.0,
            _pacing_mod._secondary_paced_wide_group_qrs_override_ms(180.0, representative_leads, groups),
        )

    def test_secondary_paced_wide_group_override_recovers_floor_censored_consensus(self) -> None:
        raw_qrs_values = [
            120.0,
            None,
            120.0,
            120.0,
            120.0,
            208.0,
            120.0,
            120.0,
            120.0,
            220.0,
            212.0,
            194.0,
        ]
        representative_leads = {
            lead: _make_rep(
                lead,
                qrs_ms=raw_qrs,
                qrs_consensus_ms=186.0,
                qrs_wide_ms=210.0,
                reliable_for_qrs=True,
            )
            for lead, raw_qrs in zip(STANDARD_12_LEADS, raw_qrs_values)
        }
        groups = {
            1: SimpleNamespace(mean_qrs_ms=None, flags={"dominant_group": True, "wide_qrs": False}),
            2: SimpleNamespace(mean_qrs_ms=82.0, flags={"dominant_group": False, "wide_qrs": False}),
            3: SimpleNamespace(mean_qrs_ms=152.0, flags={"dominant_group": False, "wide_qrs": True}),
        }

        self.assertEqual(
            152.0,
            _pacing_mod._secondary_paced_wide_group_qrs_override_ms(
                120.0,
                representative_leads,
                groups,
            ),
        )

    def test_phys_pr_core_uses_sparse_plausible_candidates(self) -> None:
        beat_features = [
            SimpleNamespace(pr_ms=286.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(pr_ms=82.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(pr_ms=162.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(pr_ms=146.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(pr_ms=304.0, p_confidence=1.0, flags=[]),
        ]

        self.assertEqual(154.0, _physiologic_pr_core_from_beats(beat_features))

    def test_phys_pr_core_uses_upper_core_when_short_pr_candidates_dominate(self) -> None:
        beat_features = [
            SimpleNamespace(pr_ms=value, p_confidence=1.0, flags=[])
            for value in [48.0, 140.0, 134.0, 124.0, 132.0, 124.0, 134.0, 140.0, 56.0]
        ]

        self.assertAlmostEqual(138.08, _physiologic_pr_core_from_beats(beat_features), places=2)

    def test_phys_pr_core_counts_each_beat_once_when_one_beat_has_many_long_leads(self) -> None:
        beat_features = [
            SimpleNamespace(beat_id=0, lead="I", pr_ms=130.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(beat_id=0, lead="II", pr_ms=132.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(beat_id=1, lead="II", pr_ms=134.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(beat_id=2, lead="II", pr_ms=136.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(beat_id=3, lead="II", pr_ms=138.0, p_confidence=1.0, flags=[]),
            SimpleNamespace(beat_id=4, lead="II", pr_ms=140.0, p_confidence=1.0, flags=[]),
        ]
        beat_features.extend(
            SimpleNamespace(beat_id=5, lead=lead, pr_ms=202.0, p_confidence=1.0, flags=[])
            for lead in STANDARD_12_LEADS[:10]
        )

        self.assertAlmostEqual(139.8, _physiologic_pr_core_from_beats(beat_features), places=2)

    def test_phys_pr_core_uses_lower_cluster_when_beat_pr_distribution_is_bimodal(self) -> None:
        beat_features = [
            SimpleNamespace(beat_id=beat_id, lead="II", pr_ms=pr_ms, p_confidence=1.0, flags=[])
            for beat_id, pr_ms in enumerate([145.0, 157.0, 162.0, 200.0, 202.0, 207.0])
        ]

        self.assertAlmostEqual(159.5, _physiologic_pr_core_from_beats(beat_features), places=2)

    def test_p_context_corr_accepts_same_and_inverted_shape(self) -> None:
        template = np.asarray([0.0, 0.2, 1.0, 0.2, 0.0], dtype=float)
        same = np.asarray([0.0, 0.1, 0.5, 0.1, 0.0], dtype=float)
        inverted = -same

        self.assertGreater(_p_context_corr(same, template), 0.99)
        self.assertGreater(_p_context_corr(inverted, template), 0.99)

    def test_p_context_corr_returns_neutral_for_flat_snippet(self) -> None:
        template = np.asarray([0.0, 0.2, 1.0, 0.2, 0.0], dtype=float)
        flat = np.zeros(5, dtype=float)

        self.assertEqual(0.5, _p_context_corr(flat, template))

    def test_p_context_robust_score_is_neutral_without_context(self) -> None:
        self.assertEqual(0.5, _p_context_robust_score(None, 160.0, 20.0))
        self.assertEqual(0.5, _p_context_robust_score(160.0, None, 20.0))

    def test_p_context_robust_score_penalizes_large_timing_gap(self) -> None:
        self.assertGreater(_p_context_robust_score(162.0, 160.0, 20.0), 0.9)
        self.assertLess(_p_context_robust_score(230.0, 160.0, 20.0), 0.1)

    def test_p_context_snippet_extracts_baseline_normalized_window(self) -> None:
        sig = np.ones(100, dtype=float) * 0.2
        sig[48:53] += np.asarray([0.0, 0.1, 0.4, 0.1, 0.0])

        snippet = _p_context_snippet(sig, center=50, half_width=2)

        self.assertIsNotNone(snippet)
        self.assertEqual(5, len(snippet))
        self.assertAlmostEqual(1.0, float(np.max(np.abs(snippet))), places=6)

    def test_p_reselection_fields_default_to_none(self) -> None:
        feature = _make_p_context_feature("II", 0, p_on=140, p_peak=160, p_off=180)

        self.assertIsNone(feature.p_candidate_alternative_count)
        self.assertIsNone(feature.p_reselected_from_peak_index)
        self.assertIsNone(feature.p_reselected_to_peak_index)
        self.assertIsNone(feature.p_reselected_old_context_score)
        self.assertIsNone(feature.p_reselected_new_context_score)
        self.assertIsNone(feature.p_reselection_reason)

    def test_p_reselection_gate_accepts_clear_context_improvement(self) -> None:
        self.assertTrue(_p_should_reselect_candidate(
            old_score=0.30,
            new_score=0.72,
            old_support_score=0.25,
            new_support_score=0.75,
            old_template_corr=0.20,
            new_template_corr=0.78,
            old_pr_score=0.10,
            new_pr_score=0.85,
            old_pp_score=0.50,
            new_pp_score=0.50,
        ))

    def test_p_reselection_gate_rejects_small_score_improvement(self) -> None:
        self.assertFalse(_p_should_reselect_candidate(
            old_score=0.48,
            new_score=0.62,
            old_support_score=0.25,
            new_support_score=0.75,
            old_template_corr=0.20,
            new_template_corr=0.78,
            old_pr_score=0.10,
            new_pr_score=0.85,
            old_pp_score=0.50,
            new_pp_score=0.50,
        ))

    def test_p_reselection_gate_rejects_borderline_plausible_original(self) -> None:
        self.assertFalse(_p_should_reselect_candidate(
            old_score=0.53,
            new_score=0.78,
            old_support_score=0.25,
            new_support_score=1.0,
            old_template_corr=0.50,
            new_template_corr=0.90,
            old_pr_score=0.50,
            new_pr_score=1.0,
            old_pp_score=0.50,
            new_pp_score=0.50,
        ))

    def test_p_reselection_gate_protects_strong_multilead_supported_original(self) -> None:
        self.assertFalse(_p_should_reselect_candidate(
            old_score=0.48,
            new_score=0.80,
            old_support_score=1.0,
            new_support_score=0.25,
            old_template_corr=0.25,
            new_template_corr=0.90,
            old_pr_score=0.10,
            new_pr_score=0.90,
            old_pp_score=0.50,
            new_pp_score=0.50,
        ))

    def test_measure_p_candidate_from_peak_returns_consistent_measurement(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        sig[180:221] = np.hanning(41) * 0.12

        with patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", return_value=180), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", return_value=220):
            measured = _measure_p_candidate_from_peak(
                sig=sig,
                peak=200,
                search_lo=120,
                qrs_on=260,
                baseline=0.0,
                fs=fs,
                p_noise_floor=0.02,
                lead="II",
            )

        self.assertIsNotNone(measured)
        self.assertEqual(180, measured.onset)
        self.assertEqual(200, measured.peak)
        self.assertEqual(220, measured.offset)
        self.assertAlmostEqual(160.0, measured.pr_ms)
        self.assertAlmostEqual(80.0, measured.p_dur_ms)
        self.assertGreater(measured.p_area, 0.0)
        self.assertGreater(measured.p_confidence, 0.0)

    def test_measure_p_candidate_from_peak_rejects_invalid_geometry(self) -> None:
        fs = 500
        sig = np.zeros(500, dtype=float)
        sig[180:221] = np.hanning(41) * 0.12

        with patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", return_value=205), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", return_value=220):
            measured = _measure_p_candidate_from_peak(
                sig=sig,
                peak=200,
                search_lo=120,
                qrs_on=260,
                baseline=0.0,
                fs=fs,
                p_noise_floor=0.02,
                lead="II",
            )

        self.assertIsNone(measured)

    def test_p_candidate_reselection_replaces_weak_candidate_with_better_alternative(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1400), dtype=float)
        features: list[LeadBeatFeatures] = []
        alternatives: dict[tuple[int, str], list[object]] = {}
        for beat_id, r in enumerate([300, 600, 900, 1200]):
            good_peak = r - 90
            bad_peak = r - 38 if beat_id == 3 else good_peak
            ecg[0, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            ecg[1, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            if beat_id == 3:
                ecg[1, bad_peak - 2 : bad_peak + 3] += np.asarray([0.03, -0.08, 0.05, -0.02, 0.04])
            features.append(_make_p_context_feature("II", beat_id, p_on=good_peak - 20, p_peak=good_peak, p_off=good_peak + 20, qrs_on=r - 20))
            features.append(_make_p_context_feature("V1", beat_id, p_on=bad_peak - 20, p_peak=bad_peak, p_off=bad_peak + 20, qrs_on=r - 20, p_confidence=0.4 if beat_id == 3 else 0.9))
            alternatives[(beat_id, "V1")] = [SimpleNamespace(peak=good_peak, amp=0.12, area=1.0)]
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}
        _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=False,
        )

        with patch("feature_extraction.ecgfeat.delineate._apply_p_candidate_context", side_effect=lambda feats, **_kwargs: feats), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", side_effect=lambda _sig, peak, *_args: peak - 20), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", side_effect=lambda _sig, peak, *_args: peak + 20):
            out = _apply_p_candidate_reselection(
                features,
                ecg=ecg,
                fs=fs,
                quality=quality,
                beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
                alternatives_by_key=alternatives,
            )

        changed = [f for f in out if f.lead == "V1" and f.beat_id == 3][0]
        self.assertEqual(1110, changed.p.peak)
        self.assertIn("p_candidate_reselected_by_context", changed.flags)
        self.assertEqual(1162, changed.p_reselected_from_peak_index)
        self.assertEqual(1110, changed.p_reselected_to_peak_index)
        self.assertIsNotNone(changed.p_reselected_old_context_score)
        self.assertIsNotNone(changed.p_reselected_new_context_score)

    def test_p_candidate_reselection_uses_raw_atrial_event_sample_alternative(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1400), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([300, 600, 900, 1200]):
            good_peak = r - 90
            bad_peak = r - 38 if beat_id == 3 else good_peak
            ecg[0, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            ecg[1, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            if beat_id == 3:
                ecg[1, bad_peak - 2 : bad_peak + 3] += np.asarray([0.03, -0.08, 0.05, -0.02, 0.04])
            features.append(_make_p_context_feature("II", beat_id, p_on=good_peak - 20, p_peak=good_peak, p_off=good_peak + 20, qrs_on=r - 20))
            features.append(_make_p_context_feature("V1", beat_id, p_on=bad_peak - 20, p_peak=bad_peak, p_off=bad_peak + 20, qrs_on=r - 20, p_confidence=0.4 if beat_id == 3 else 0.9))
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}
        _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=False,
        )
        alternatives = {
            (3, "V1"): [
                {
                    "sample": 1110,
                    "confidence": 0.86,
                    "detection_method": "composite_qrst_residual_derivative",
                }
            ]
        }

        with patch("feature_extraction.ecgfeat.delineate._apply_p_candidate_context", side_effect=lambda feats, **_kwargs: feats), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", side_effect=lambda _sig, peak, *_args: peak - 20), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", side_effect=lambda _sig, peak, *_args: peak + 20):
            out = _apply_p_candidate_reselection(
                features,
                ecg=ecg,
                fs=fs,
                quality=quality,
                beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
                alternatives_by_key=alternatives,
            )

        changed = [f for f in out if f.lead == "V1" and f.beat_id == 3][0]
        self.assertEqual(1110, changed.p.peak)
        self.assertEqual("raw_atrial_event_context", changed.p_reselection_reason)

    def test_raw_atrial_event_reselection_restores_missing_p_candidate(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1400), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([300, 600, 900, 1200]):
            good_peak = r - 90
            ecg[0, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            ecg[1, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            features.append(_make_p_context_feature("II", beat_id, p_on=good_peak - 20, p_peak=good_peak, p_off=good_peak + 20, qrs_on=r - 20))
            v1_feature = _make_p_context_feature("V1", beat_id, p_on=good_peak - 20, p_peak=good_peak, p_off=good_peak + 20, qrs_on=r - 20)
            if beat_id == 3:
                v1_feature.p = WaveBounds(onset=None, peak=None, offset=None)
                v1_feature.pr_ms = None
                v1_feature.p_dur_ms = None
                v1_feature.p_amp_mv = None
                v1_feature.p_area = None
                v1_feature.p_signed_area = None
                v1_feature.p_confidence = 0.0
                v1_feature.p_candidate_context_score = 0.0
                v1_feature.p_multilead_support_score = 0.0
                v1_feature.p_template_corr = 0.0
                v1_feature.p_pr_consistency_score = 0.0
                v1_feature.p_pp_consistency_score = 0.50
                v1_feature.flags.append("p_unreliable")
            features.append(v1_feature)
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}
        alternatives = {
            (3, "V1"): [
                {
                    "sample": 1110,
                    "confidence": 0.90,
                    "detection_method": "composite_qrst_residual_derivative",
                }
            ]
        }

        with patch("feature_extraction.ecgfeat.delineate._apply_p_candidate_context", side_effect=lambda feats, **_kwargs: feats), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", side_effect=lambda _sig, peak, *_args: peak - 20), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", side_effect=lambda _sig, peak, *_args: peak + 20):
            out = _apply_p_candidate_reselection(
                features,
                ecg=ecg,
                fs=fs,
                quality=quality,
                beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
                alternatives_by_key=alternatives,
            )

        restored = [f for f in out if f.lead == "V1" and f.beat_id == 3][0]
        self.assertEqual(1110, restored.p.peak)
        self.assertEqual(1090, restored.p.onset)
        self.assertEqual(1130, restored.p.offset)
        self.assertEqual("raw_atrial_event_context", restored.p_reselection_reason)
        self.assertNotIn("p_unreliable", restored.flags)

    def test_raw_atrial_event_reselection_reviews_borderline_context_candidate(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1400), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([300, 600, 900, 1200]):
            good_peak = r - 90
            bad_peak = r - 38 if beat_id == 3 else good_peak
            ecg[0, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            ecg[1, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            if beat_id == 3:
                ecg[1, bad_peak - 2 : bad_peak + 3] += np.asarray([0.03, -0.08, 0.05, -0.02, 0.04])
            features.append(_make_p_context_feature("II", beat_id, p_on=good_peak - 20, p_peak=good_peak, p_off=good_peak + 20, qrs_on=r - 20))
            features.append(_make_p_context_feature("V1", beat_id, p_on=bad_peak - 20, p_peak=bad_peak, p_off=bad_peak + 20, qrs_on=r - 20, p_confidence=0.52 if beat_id == 3 else 0.9))
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}
        _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=False,
        )
        borderline = [f for f in features if f.lead == "V1" and f.beat_id == 3][0]
        borderline.p_candidate_context_score = 0.505
        borderline.p_multilead_support_score = 0.25
        borderline.p_template_corr = 0.20
        borderline.p_pr_consistency_score = 0.10
        borderline.p_pp_consistency_score = 0.50
        alternatives = {
            (3, "V1"): [
                {
                    "sample": 1110,
                    "confidence": 0.90,
                    "detection_method": "composite_qrst_residual_derivative",
                }
            ]
        }

        with patch("feature_extraction.ecgfeat.delineate._apply_p_candidate_context", side_effect=lambda feats, **_kwargs: feats), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", side_effect=lambda _sig, peak, *_args: peak - 20), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", side_effect=lambda _sig, peak, *_args: peak + 20):
            out = _apply_p_candidate_reselection(
                features,
                ecg=ecg,
                fs=fs,
                quality=quality,
                beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
                alternatives_by_key=alternatives,
            )

        changed = [f for f in out if f.lead == "V1" and f.beat_id == 3][0]
        self.assertEqual(1110, changed.p.peak)
        self.assertEqual("raw_atrial_event_context", changed.p_reselection_reason)

    def test_raw_atrial_events_are_added_as_per_lead_p_alternatives(self) -> None:
        fs = 500
        features = [
            _make_p_context_feature("II", 3, p_on=1090, p_peak=1110, p_off=1130, qrs_on=1180),
            _make_p_context_feature("V1", 3, p_on=1142, p_peak=1162, p_off=1178, qrs_on=1180, p_confidence=0.4),
        ]
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}
        alternatives: dict[tuple[int, str], list[object]] = {}
        raw_events = [
            {
                "sample": 1110,
                "confidence": 0.86,
                "associated_qrs_beat_id": 3,
                "association_type": "conducted",
                "detection_method": "composite_qrst_residual_derivative",
            }
        ]

        augment = getattr(delineate_mod, "_augment_p_alternatives_from_raw_atrial_events", None)
        self.assertIsNotNone(augment)
        augment(
            features,
            raw_events,
            fs=fs,
            quality=quality,
            alternatives_by_key=alternatives,
        )

        self.assertIn((3, "V1"), alternatives)
        self.assertEqual(1110, alternatives[(3, "V1")][0]["sample"])
        self.assertNotIn((3, "II"), alternatives)

    def test_p_candidate_reselection_rejects_invalid_alternative_geometry(self) -> None:
        fs = 500
        ecg = np.zeros((1, 600), dtype=float)
        ecg[0, 180:221] = np.hanning(41) * 0.12
        feature = _make_p_context_feature("II", 0, p_on=245, p_peak=265, p_off=285, qrs_on=320, p_confidence=0.4)
        feature.p_candidate_context_score = 0.30
        feature.p_multilead_support_score = 0.25
        feature.p_template_corr = 0.20
        feature.p_pr_consistency_score = 0.10
        feature.p_pp_consistency_score = 0.50
        quality = {"II": _make_quality(reliable_for_p=True)}
        alternatives = {(0, "II"): [SimpleNamespace(peak=200, amp=0.12, area=1.0)]}

        with patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", return_value=205), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", return_value=220):
            out = _apply_p_candidate_reselection(
                [feature],
                ecg=ecg,
                fs=fs,
                quality=quality,
                beat_to_group={0: 1},
                alternatives_by_key=alternatives,
            )

        self.assertEqual(265, out[0].p.peak)
        self.assertNotIn("p_candidate_reselected_by_context", out[0].flags)

    def test_p_candidate_context_scores_repeated_template_and_support(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1200), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([250, 550, 850, 1150]):
            for li, lead in enumerate(["II", "V1"]):
                peak = r - 80
                on = peak - 20
                off = peak + 20
                ecg[li, peak - 2 : peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
                features.append(_make_p_context_feature(lead, beat_id, p_on=on, p_peak=peak, p_off=off, qrs_on=r - 20))
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}

        out = _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=False,
        )

        self.assertTrue(all(f.p_template_corr is not None and f.p_template_corr > 0.9 for f in out))
        self.assertTrue(all(f.p_multilead_support == 2 for f in out))
        self.assertTrue(all(f.p_candidate_context_score is not None and f.p_candidate_context_score > 0.65 for f in out))

    def test_p_candidate_context_keeps_neutral_scores_when_template_is_sparse(self) -> None:
        fs = 500
        ecg = np.zeros((1, 400), dtype=float)
        ecg[0, 158:163] = np.asarray([0.0, 0.03, 0.10, 0.03, 0.0])
        feature = _make_p_context_feature("II", 0, p_on=140, p_peak=160, p_off=180)
        quality = {"II": _make_quality(reliable_for_p=True)}

        out = _apply_p_candidate_context(
            [feature],
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1},
            suppress=False,
        )

        self.assertEqual(0.5, out[0].p_template_corr)
        self.assertEqual(0.5, out[0].p_pr_consistency_score)
        self.assertEqual(0.5, out[0].p_pp_consistency_score)

    def test_p_candidate_context_suppresses_isolated_noise_like_candidate(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1500), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([300, 600, 900, 1200]):
            true_peak = r - 90
            ecg[0, true_peak - 2 : true_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
            features.append(_make_p_context_feature("II", beat_id, p_on=true_peak - 20, p_peak=true_peak, p_off=true_peak + 20, qrs_on=r - 20))
            v1_peak = true_peak if beat_id < 3 else r - 38
            if beat_id < 3:
                ecg[1, v1_peak - 2 : v1_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
                p_confidence = 0.9
            else:
                ecg[1, v1_peak - 2 : v1_peak + 3] += np.asarray([0.03, -0.08, 0.05, -0.02, 0.04])
                p_confidence = 0.38
            features.append(_make_p_context_feature("V1", beat_id, p_on=v1_peak - 20, p_peak=v1_peak, p_off=v1_peak + 20, qrs_on=r - 20, p_confidence=p_confidence))
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}

        out = _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=True,
        )
        bad = [f for f in out if f.lead == "V1" and f.beat_id == 3][0]
        good = [f for f in out if f.lead == "II" and f.beat_id == 3][0]

        self.assertIsNone(bad.p.peak)
        self.assertIn("p_candidate_suppressed_by_context", bad.flags)
        self.assertIsNotNone(good.p.peak)

    def test_p_candidate_context_does_not_suppress_low_amplitude_supported_p(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1200), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([250, 550, 850, 1150]):
            for li, lead in enumerate(["II", "V1"]):
                peak = r - 80
                ecg[li, peak - 2 : peak + 3] += np.asarray([0.0, 0.015, 0.04, 0.015, 0.0])
                features.append(_make_p_context_feature(lead, beat_id, p_on=peak - 20, p_peak=peak, p_off=peak + 20, qrs_on=r - 20, p_confidence=0.32))
        quality = {"II": _make_quality(reliable_for_p=True), "V1": _make_quality(reliable_for_p=True)}

        out = _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=True,
        )

        self.assertTrue(all(f.p.peak is not None for f in out))
        self.assertTrue(all("p_candidate_suppressed_by_context" not in f.flags for f in out))

    def test_p_candidate_context_does_not_suppress_strong_multilead_supported_candidate(self) -> None:
        fs = 500
        leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        ecg = np.zeros((len(leads), 1500), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([300, 600, 900, 1200]):
            for li, lead in enumerate(leads):
                true_peak = r - 90
                if beat_id < 3:
                    peak = true_peak
                    ecg[li, peak - 2 : peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
                    p_confidence = 0.9
                else:
                    peak = r - 38
                    ecg[li, peak - 2 : peak + 3] += np.asarray([0.03, -0.08, 0.05, -0.02, 0.04])
                    p_confidence = 0.0
                features.append(_make_p_context_feature(
                    lead,
                    beat_id,
                    p_on=peak - 20,
                    p_peak=peak,
                    p_off=peak + 20,
                    qrs_on=r,
                    p_confidence=p_confidence,
                ))
        quality = {lead: _make_quality(reliable_for_p=True) for lead in leads}

        out = _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=True,
        )
        supported = [f for f in out if f.beat_id == 3]

        self.assertTrue(all(f.p_multilead_support == len(leads) for f in supported))
        self.assertTrue(all(f.p.peak is not None for f in supported))
        self.assertTrue(all("p_candidate_suppressed_by_context" not in f.flags for f in supported))

    def test_p_candidate_context_suppresses_short_pr_noise_without_limb_support(self) -> None:
        fs = 500
        leads = ["V1", "V2", "V3"]
        ecg = np.zeros((len(leads), 1500), dtype=float)
        features: list[LeadBeatFeatures] = []
        for beat_id, r in enumerate([300, 600, 900, 1200]):
            for li, lead in enumerate(leads):
                if beat_id < 3:
                    peak = r - 90
                    ecg[li, peak - 2 : peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
                    p_confidence = 0.9
                else:
                    peak = r - 38
                    ecg[li, peak - 2 : peak + 3] += np.asarray([0.03, -0.08, 0.05, -0.02, 0.04])
                    p_confidence = 0.36
                features.append(_make_p_context_feature(
                    lead,
                    beat_id,
                    p_on=peak - 20,
                    p_peak=peak,
                    p_off=peak + 20,
                    qrs_on=r,
                    p_confidence=p_confidence,
                ))
        quality = {lead: _make_quality(reliable_for_p=True) for lead in leads}

        out = _apply_p_candidate_context(
            features,
            ecg=ecg,
            fs=fs,
            quality=quality,
            beat_to_group={0: 1, 1: 1, 2: 1, 3: 1},
            suppress=True,
        )
        short_pr_noise = [feature for feature in out if feature.beat_id == 3]

        self.assertTrue(all(feature.p_multilead_support == len(leads) for feature in short_pr_noise))
        self.assertTrue(all(feature.p.peak is None for feature in short_pr_noise))
        self.assertTrue(all("p_candidate_suppressed_by_context" in feature.flags for feature in short_pr_noise))

    def test_delineate_beats_populates_p_context_fields(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1800), dtype=float)
        r_locs = np.asarray([300, 700, 1100, 1500], dtype=int)
        for li in range(2):
            for r in r_locs:
                peak = int(r - 90)
                ecg[li, peak - 2 : peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
                ecg[li, int(r) - 3 : int(r) + 4] += np.asarray([0.0, 0.1, 0.4, 1.0, 0.4, 0.1, 0.0])
        quality = {
            "II": _make_quality(reliable_for_p=True, reliable_for_qrs=True, reliable_for_qt=True),
            "V1": _make_quality(reliable_for_p=True, reliable_for_qrs=True, reliable_for_qt=True),
        }

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II", "V1"]), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", side_effect=lambda _sig, peak, *_args: peak - 20), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", side_effect=lambda _sig, peak, *_args: peak + 20):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        p_features = [f for f in features if f.p.peak is not None]
        self.assertTrue(p_features)
        self.assertTrue(any(f.p_candidate_context_score is not None for f in p_features))

    def test_delineate_beats_reselects_better_local_p_candidate(self) -> None:
        fs = 500
        ecg = np.zeros((2, 1600), dtype=float)
        r_locs = np.asarray([300, 600, 900, 1200], dtype=int)
        for li in range(2):
            for r in r_locs:
                good_peak = int(r - 90)
                ecg[li, good_peak - 2 : good_peak + 3] += np.asarray([0.0, 0.04, 0.12, 0.04, 0.0])
                ecg[li, int(r) - 3 : int(r) + 4] += np.asarray([0.0, 0.1, 0.4, 1.0, 0.4, 0.1, 0.0])
        bad_peak = int(r_locs[-1] - 38)
        ecg[1, bad_peak - 2 : bad_peak + 3] += np.asarray([0.03, -0.08, 0.20, -0.02, 0.04])
        quality = {
            "II": _make_quality(reliable_for_p=True, reliable_for_qrs=True, reliable_for_qt=True),
            "V1": _make_quality(reliable_for_p=True, reliable_for_qrs=True, reliable_for_qt=True),
        }

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II", "V1"]), \
             patch("feature_extraction.ecgfeat.delineate._fuse_peak_anchor", return_value=(None, 0)), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_tangent", side_effect=lambda _sig, peak, *_args: peak - 20), \
             patch("feature_extraction.ecgfeat.delineate._p_offset_tangent", side_effect=lambda _sig, peak, *_args: peak + 20):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs, quality=quality)

        v1_last = [f for f in features if f.lead == "V1" and f.beat_id == 3][0]
        self.assertIn("p_candidate_reselected_by_context", v1_last.flags)
        self.assertEqual(1110, v1_last.p.peak)
        self.assertEqual(1090, v1_last.p.onset)
        self.assertEqual(1130, v1_last.p.offset)

    def test_stable_limb_signed_t_axis_uses_low_confidence_consistent_limb_vector(self) -> None:
        representative_leads = {lead: _make_rep(lead) for lead in STANDARD_12_LEADS}
        limb_values = {
            "I": (3.024, 0.075, 0.78),
            "II": (-8.161, -0.102, 0.37),
            "III": (-7.005, -0.110, 0.38),
            "aVR": (4.278, 0.051, 0.33),
            "aVL": (5.406, 0.096, 0.44),
            "aVF": (-7.983, -0.116, 0.34),
        }
        for lead, (t_area, t_amp, qt_conf) in limb_values.items():
            representative_leads[lead] = _make_rep(
                lead,
                t_signed_area=t_area,
                t_amp_mv=t_amp,
                qt_confidence_mean=qt_conf,
                reliable_for_t=True,
            )

        self.assertAlmostEqual(-72.4, _stable_limb_signed_t_axis_deg(representative_leads), places=1)

    def test_delineate_beats_populates_true_q_component_fields(self) -> None:
        fs = 500
        ecg = np.zeros((1, 500), dtype=float)
        r_locs = np.asarray([200], dtype=int)
        beat_start = max(0, int(r_locs[0] - 0.35 * fs))

        qrs_values = np.asarray([
            0.0, -0.05, -0.12, -0.20, -0.12, -0.04, 0.0,
            0.2, 0.7, 1.0, 0.7, 0.2, 0.0,
        ])
        ecg[0, beat_start + 170 : beat_start + 170 + len(qrs_values)] = qrs_values

        with patch("feature_extraction.ecgfeat.delineate.STANDARD_12_LEADS", ["II"]), \
             patch("feature_extraction.ecgfeat.delineate._qrs_bounds", return_value=(170, 182, 0, 0.9, 0.9)), \
             patch("feature_extraction.ecgfeat.delineate._find_peak", side_effect=[130, 300]), \
             patch("feature_extraction.ecgfeat.delineate._find_wave_bounds", return_value=(120, 140)), \
             patch("feature_extraction.ecgfeat.delineate._t_onset_chord", return_value=280), \
             patch("feature_extraction.ecgfeat.delineate._t_end_geometric", return_value=(330, 0.7, "laguna_tangent")):
            features = delineate_beats(ecg, fs=fs, r_locs=r_locs)

        feature = features[0]
        self.assertIsNotNone(feature.q_onset)
        self.assertIsNotNone(feature.q_offset)
        self.assertIsNotNone(feature.q_duration_ms)
        self.assertIsNotNone(feature.q_area_mv_ms)
        self.assertGreater(feature.q_area_mv_ms, 0.0)
        self.assertAlmostEqual(0.20, feature.q_r_ratio, places=2)


class QtRejectGateTests(unittest.TestCase):
    def _gf(self, *, qt_reliability, qt_ms=380.0, qtc_bazett_ms=410.0):
        return GlobalFeatures(
            heart_rate_bpm=70.0, atrial_rate_bpm=70.0, pr_ms=160.0, qrs_ms=90.0,
            qt_ms=qt_ms, qtc_bazett_ms=qtc_bazett_ms, qtc_fridericia_ms=400.0,
            p_axis_deg=50.0, qrs_axis_deg=50.0, t_axis_deg=40.0, st_axis_deg=None,
            qt_dispersion_ms=None, qt_reliability=qt_reliability,
        )

    def test_rejects_fallback_reliability_on_poor_record_grade(self) -> None:
        gf = self._gf(qt_reliability="fallback")
        _qt_mod._apply_qt_reject_gate(gf, "Q2")
        self.assertIsNone(gf.qt_ms)
        self.assertIsNone(gf.qtc_bazett_ms)
        self.assertIsNone(gf.qtc_fridericia_ms)
        self.assertTrue(gf.qt_rejected)
        self.assertIn("fallback", gf.qt_reject_reason)
        self.assertIn("Q2", gf.qt_reject_reason)

    def test_low_confidence_reliability_on_q3_is_rejected(self) -> None:
        gf = self._gf(qt_reliability="low_confidence")
        _qt_mod._apply_qt_reject_gate(gf, "Q3")
        self.assertIsNone(gf.qt_ms)
        self.assertTrue(gf.qt_rejected)

    def test_fallback_reliability_on_clean_record_is_not_rejected(self) -> None:
        # Weak QT evidence alone (fallback) on an otherwise clean record grade
        # shouldn't be nulled -- only the combination with poor overall
        # record quality should trigger rejection.
        gf = self._gf(qt_reliability="fallback")
        _qt_mod._apply_qt_reject_gate(gf, "Q0")
        self.assertEqual(380.0, gf.qt_ms)
        self.assertFalse(gf.qt_rejected)

    def test_reliable_qt_on_poor_record_grade_is_not_rejected(self) -> None:
        # A strong QT reliability tier should survive even a poor record
        # grade -- the reject gate targets the *combination* of weak QT
        # evidence and weak overall quality, not either alone.
        gf = self._gf(qt_reliability="reliable")
        _qt_mod._apply_qt_reject_gate(gf, "Q3")
        self.assertEqual(380.0, gf.qt_ms)
        self.assertFalse(gf.qt_rejected)

    def test_unavailable_reliability_is_left_alone(self) -> None:
        gf = self._gf(qt_reliability="unavailable", qt_ms=None, qtc_bazett_ms=None)
        _qt_mod._apply_qt_reject_gate(gf, "Q3")
        self.assertIsNone(gf.qt_ms)
        self.assertFalse(gf.qt_rejected)


if __name__ == "__main__":
    unittest.main()
