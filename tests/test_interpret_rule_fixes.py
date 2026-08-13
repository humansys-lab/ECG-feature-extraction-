from __future__ import annotations

import unittest
from types import SimpleNamespace

from feature_extraction.ecgfeat.interpret import (
    _classify_heart_rate,
    _classify_qrs_axis,
    _detect_avb_and_dissociation,
    _lvh_criteria,
    _p_wave_morphology,
    _resolve_qrs_width_class,
    _rvh_scored,
    _st_analysis,
)
from feature_extraction.ecgfeat.mi import build_mi_evidence
from feature_extraction.ecgfeat.models import RepresentativeLeadFeatures


def _rep(lead: str, **params: object) -> RepresentativeLeadFeatures:
    base = {
        "reliable_for_qrs": True,
        "reliable_for_p": True,
        "reliable_for_t": True,
        "reliable_for_qt": True,
    }
    base.update(params)
    return RepresentativeLeadFeatures(lead=lead, params=base, variance={})


class CompleteAvbRequiresDissociationTests(unittest.TestCase):
    def test_slow_rate_alone_is_not_complete_avb(self) -> None:
        gf = SimpleNamespace(heart_rate_bpm=40.0, atrial_rate_bpm=None)
        complete_avb, av_dissociation, _ = _detect_avb_and_dissociation([], {}, gf)
        self.assertFalse(complete_avb)
        self.assertFalse(av_dissociation)

    def test_slow_rate_with_faster_atrial_rate_is_complete_avb(self) -> None:
        gf = SimpleNamespace(heart_rate_bpm=40.0, atrial_rate_bpm=80.0)
        complete_avb, _, _ = _detect_avb_and_dissociation([], {}, gf)
        self.assertTrue(complete_avb)

    def test_atrial_rate_equal_to_ventricular_rate_does_not_count(self) -> None:
        # Guards against the atrial-rate estimator's fallback (atrial_rate_bpm
        # defaults to heart_rate_bpm when P-wave-based estimation fails)
        # being misread as "atrial faster than ventricular".
        gf = SimpleNamespace(heart_rate_bpm=40.0, atrial_rate_bpm=40.0)
        complete_avb, _, _ = _detect_avb_and_dissociation([], {}, gf)
        self.assertFalse(complete_avb)

    def test_normal_rate_is_never_complete_avb_even_with_dissociation_evidence(self) -> None:
        gf = SimpleNamespace(heart_rate_bpm=70.0, atrial_rate_bpm=120.0)
        complete_avb, _, _ = _detect_avb_and_dissociation([], {}, gf)
        self.assertFalse(complete_avb)


class QrsAxisLpfbWrapTests(unittest.TestCase):
    def test_lpfb_zone_wraps_through_negative_180(self) -> None:
        self.assertEqual("LPFB", _classify_qrs_axis(-160.0))
        self.assertEqual("LPFB", _classify_qrs_axis(-179.0))
        self.assertEqual("LPFB", _classify_qrs_axis(-150.0))  # inclusive boundary

    def test_beyond_lpfb_wrap_is_still_erad(self) -> None:
        self.assertEqual("ERAD", _classify_qrs_axis(-140.0))
        self.assertEqual("ERAD", _classify_qrs_axis(-100.0))

    def test_non_wrapped_lpfb_zone_unaffected(self) -> None:
        self.assertEqual("LPFB", _classify_qrs_axis(150.0))
        self.assertEqual("RAD", _classify_qrs_axis(100.0))


class PediatricHeartRateClassificationTests(unittest.TestCase):
    def test_newborn_normal_rate_is_not_flagged_tachycardia(self) -> None:
        # A 5-day-old with HR 130 bpm is physiologically normal; the adult
        # 100 bpm cutoff would wrongly call this tachycardia.
        self.assertEqual("normal", _classify_heart_rate(130.0, age_days=5.0))

    def test_same_rate_without_age_uses_adult_thresholds(self) -> None:
        self.assertEqual("tachycardia", _classify_heart_rate(130.0, age_days=None))

    def test_adult_age_reduces_to_adult_thresholds(self) -> None:
        adult_days = 40.0 * 365.25
        self.assertEqual("tachycardia", _classify_heart_rate(130.0, age_days=adult_days))
        self.assertEqual("normal", _classify_heart_rate(80.0, age_days=adult_days))


class PWaveMorphologyPediatricAndDeadConstantTests(unittest.TestCase):
    def test_pediatric_rae_threshold_is_lower_than_adult(self) -> None:
        p_dur_by_lead = {"I": 70.0, "II": 70.0, "III": 70.0, "aVL": 70.0, "aVF": 70.0}
        reps = {
            lead: _rep(lead, p_amp_mv=0.22)
            for lead in ("I", "II", "III", "aVL", "aVF")
        }
        p_class_peds, rae_leads_peds, *_ = _p_wave_morphology(
            reps, p_dur_by_lead, ptf_v1=None, is_pediatric=True
        )
        p_class_adult, rae_leads_adult, *_ = _p_wave_morphology(
            reps, p_dur_by_lead, ptf_v1=None, is_pediatric=False
        )
        # 0.22 mV clears the pediatric "consider" bar (0.20) in >=2 leads but
        # not the shared "confirmed" 2-lead bar (0.25), so pediatric routing
        # reaches "probable_rae" while adult routing (0.24 consider bar)
        # counts none of these leads at all.
        self.assertGreaterEqual(len(rae_leads_peds), 2)
        self.assertEqual("probable_rae", p_class_peds)
        self.assertEqual([], rae_leads_adult)
        self.assertEqual("normal", p_class_adult)

    def test_rae_two_lead_confirmation_requires_stricter_threshold(self) -> None:
        # 0.245 mV clears the "consider" bar (0.24) but not the "confirmed"
        # 2-lead bar (0.25); two such leads must not confirm RAE on their own.
        p_dur_by_lead = {"I": 70.0, "II": 70.0}
        reps = {
            "I": _rep("I", p_amp_mv=0.245),
            "II": _rep("II", p_amp_mv=0.245),
        }
        p_class, rae_leads, *_ = _p_wave_morphology(reps, p_dur_by_lead, ptf_v1=None)
        self.assertEqual(2, len(rae_leads))
        self.assertEqual("probable_rae", p_class)

    def test_p_duration_alone_is_not_bae(self) -> None:
        # 85 ms with a 0.05 mV P wave is an ordinary sinus P.  The 80 ms row in
        # DXL_Threshold_Reference.xlsx is the qualifying minimum for the BAE
        # combination criterion, not a criterion; treating it as sufficient made
        # BAE easier to reach than LAE and fired on 78% of PTB-XL records.
        p_dur_by_lead = {"II": 85.0}
        reps = {"II": _rep("II", p_amp_mv=0.05), "V1": _rep("V1", p_amp_mv=0.0)}
        p_class, *_ = _p_wave_morphology(reps, p_dur_by_lead, ptf_v1=None)
        self.assertEqual("normal", p_class)

    # A short V1 terminal component keeps the "RAE confirmed + LAE suspected"
    # route out of these two cases, so they exercise the direct V1+limb
    # criterion and nothing else.
    _BAE_DIRECT_REPS = {
        "I": dict(p_amp_mv=0.32),
        "II": dict(p_amp_mv=0.35),
        "V1": dict(p_terminal_amp_mv=-0.16, p_terminal_duration_ms=20.0),
    }

    def test_bae_direct_path_via_v1_terminal_and_limb_amplitude(self) -> None:
        reps = {lead: _rep(lead, **kw) for lead, kw in self._BAE_DIRECT_REPS.items()}
        p_class, _rae, lae_suspected, *_ = _p_wave_morphology(
            reps, {"I": 95.0, "II": 100.0}, ptf_v1=None
        )
        self.assertFalse(lae_suspected)
        self.assertEqual("bae", p_class)

    def test_bae_combination_requires_the_p_duration_minimum(self) -> None:
        # Same amplitudes, but every measured limb P is shorter than the 80 ms
        # qualifying minimum, so the combination must not be called BAE.
        reps = {lead: _rep(lead, **kw) for lead, kw in self._BAE_DIRECT_REPS.items()}
        p_class, *_ = _p_wave_morphology(reps, {"I": 70.0, "II": 74.0}, ptf_v1=None)
        self.assertEqual("rae", p_class)

    def test_plain_v1_biphasic_negative_terminal_is_not_lae(self) -> None:
        # The ordinary V1 P morphology: biphasic with a small negative terminal
        # component that clears neither the 0.09 mV nor the 30 ms DXL bar.
        reps = {
            "V1": _rep(
                "V1",
                p_biphasic=True,
                p_terminal_amp_mv=-0.02,
                p_terminal_duration_ms=20.0,
            ),
        }
        p_class, _rae, lae_suspected, lae_definite, _ptf = _p_wave_morphology(
            reps, {}, ptf_v1=None
        )
        self.assertFalse(lae_suspected)
        self.assertFalse(lae_definite)
        self.assertEqual("normal", p_class)

    def test_v1_biphasic_clearing_the_bars_still_suspects_lae(self) -> None:
        reps = {
            "V1": _rep(
                "V1",
                p_biphasic=True,
                p_terminal_amp_mv=-0.11,
                p_terminal_duration_ms=45.0,
                p_terminal_area_mv_ms=-2.5,
            ),
        }
        _p_class, _rae, lae_suspected, _lae_def, _ptf = _p_wave_morphology(
            reps, {}, ptf_v1=None
        )
        self.assertTrue(lae_suspected)

    def test_v1_amplitude_duration_lae_tier_is_wired(self) -> None:
        p_dur_by_lead: dict = {}
        reps = {
            "V1": _rep(
                "V1",
                p_terminal_amp_mv=-0.16,
                p_terminal_duration_ms=65.0,
            ),
        }
        p_class, _rae_leads, lae_suspected, lae_definite, _ptf_cls = _p_wave_morphology(
            reps, p_dur_by_lead, ptf_v1=None
        )
        self.assertTrue(lae_suspected)
        self.assertTrue(lae_definite)
        self.assertIn(p_class, ("lae",))


class LvhScoringTests(unittest.TestCase):
    def _one_voltage_criterion_leads(self) -> dict:
        # R aVL >= 1.2 mV (male) is the only voltage criterion satisfied.
        return {
            "aVL": _rep("aVL", r_amp_mv=1.3),
            "I": _rep("I", r_amp_mv=0.0),
            "III": _rep("III", s_amp_mv=0.0),
            "V5": _rep("V5", r_amp_mv=0.0),
            "V6": _rep("V6", r_amp_mv=0.0, vat_ms=30.0),
            "V1": _rep("V1", s_amp_mv=0.0),
            "V2": _rep("V2", s_amp_mv=0.0),
            "V3": _rep("V3", s_amp_mv=0.0),
        }

    def test_single_voltage_criterion_alone_is_by_voltage(self) -> None:
        criteria, lvh_class, secondary = _lvh_criteria(
            self._one_voltage_criterion_leads(), qrs_ms=90.0, is_female=False,
            age=50.0, qrs_axis_class="normal",
        )
        self.assertEqual(["R_aVL"], criteria)
        self.assertEqual("by_voltage", lvh_class)
        self.assertFalse(secondary)

    def test_voltage_plus_lad_is_probable_not_consider(self) -> None:
        _criteria, lvh_class, _secondary = _lvh_criteria(
            self._one_voltage_criterion_leads(), qrs_ms=90.0, is_female=False,
            age=50.0, qrs_axis_class="LAD",
        )
        self.assertEqual("probable", lvh_class)

    def test_multiple_voltage_plus_lad_is_definite(self) -> None:
        leads = self._one_voltage_criterion_leads()
        # Add a second independent voltage criterion: R I + S III >= 2.5 mV.
        leads["I"] = _rep("I", r_amp_mv=2.0)
        leads["III"] = _rep("III", s_amp_mv=-0.6)
        _criteria, lvh_class, _secondary = _lvh_criteria(
            leads, qrs_ms=90.0, is_female=False, age=50.0, qrs_axis_class="LAD",
        )
        self.assertEqual("definite", lvh_class)

    def test_anterolateral_strain_is_a_distinct_secondary_flag_not_a_score_bump(self) -> None:
        _criteria, lvh_class, secondary = _lvh_criteria(
            self._one_voltage_criterion_leads(), qrs_ms=90.0, is_female=False,
            age=50.0, qrs_axis_class="normal", lv_strain=True,
        )
        self.assertEqual("by_voltage", lvh_class)
        self.assertTrue(secondary)

    def test_cornell_product_alone_is_at_least_probable(self) -> None:
        leads = {
            "aVL": _rep("aVL", r_amp_mv=1.0),
            "V3": _rep("V3", s_amp_mv=-2.0),
            "I": _rep("I", r_amp_mv=0.0),
            "III": _rep("III", s_amp_mv=0.0),
            "V5": _rep("V5", r_amp_mv=0.0),
            "V6": _rep("V6", r_amp_mv=0.0),
            "V1": _rep("V1", s_amp_mv=0.0),
            "V2": _rep("V2", s_amp_mv=0.0),
        }
        _criteria, lvh_class, _secondary = _lvh_criteria(
            leads, qrs_ms=140.0, is_female=False, age=50.0, qrs_axis_class="normal",
        )
        self.assertIn(lvh_class, ("probable", "definite"))


class RvhDurationGateTests(unittest.TestCase):
    def _base_leads(self, **overrides: object) -> dict:
        leads = {
            "V1": _rep("V1", r_amp_mv=0.05, q_amp_mv=0.0, s_amp_mv=0.0),
            "V6": _rep("V6", r_amp_mv=1.0, s_amp_mv=0.0),
            "I": _rep("I", r_amp_mv=1.0, s_amp_mv=0.0),
        }
        leads.update(overrides)
        return leads

    def test_r_prime_below_duration_gate_does_not_count(self) -> None:
        leads = self._base_leads()
        r_prime_by_lead = {"V1": 0.35}
        short_dur = {"V1": 10.0}
        long_dur = {"V1": 25.0}

        self.assertIsNone(
            _rvh_scored(leads, r_prime_by_lead, rae_confirmed=False,
                        qrs_axis_class="normal", r_prime_dur_by_lead=short_dur)
        )
        self.assertEqual(
            "consider",
            _rvh_scored(leads, r_prime_by_lead, rae_confirmed=False,
                        qrs_axis_class="normal", r_prime_dur_by_lead=long_dur),
        )

    def test_r_prime_duration_unmeasured_does_not_block(self) -> None:
        leads = self._base_leads()
        r_prime_by_lead = {"V1": 0.35}
        self.assertEqual(
            "consider",
            _rvh_scored(leads, r_prime_by_lead, rae_confirmed=False,
                        qrs_axis_class="normal", r_prime_dur_by_lead=None),
        )

    def test_lateral_component_checks_q_and_s_prime_not_just_s(self) -> None:
        leads = self._base_leads(
            V6=_rep("V6", r_amp_mv=1.0, s_amp_mv=0.0, q_amp_mv=-0.25, q_duration_ms=45.0),
        )
        result = _rvh_scored(leads, {}, rae_confirmed=False, qrs_axis_class="normal")
        self.assertEqual("consider", result)

    def test_lateral_component_below_duration_gate_does_not_count(self) -> None:
        leads = self._base_leads(
            V6=_rep("V6", r_amp_mv=1.0, s_amp_mv=0.0, q_amp_mv=-0.25, q_duration_ms=10.0),
        )
        result = _rvh_scored(leads, {}, rae_confirmed=False, qrs_axis_class="normal")
        self.assertIsNone(result)


class StemiTUprightGateTests(unittest.TestCase):
    def _inferior_leads(self, t_amp: float) -> dict:
        return {
            "II": _rep("II", st_on_mv=0.12, t_amp_mv=t_amp),
            "III": _rep("III", st_on_mv=0.12, t_amp_mv=t_amp),
            "aVF": _rep("aVF", st_on_mv=0.0),
        }

    def test_inferior_acute_code_requires_upright_t(self) -> None:
        _ele, _dep, _terr_ele, _terr_dep, codes, *_ = _st_analysis(
            self._inferior_leads(t_amp=0.2)
        )
        self.assertIn("IMIA", codes)

    def test_inferior_downgrades_to_probable_without_upright_t(self) -> None:
        _ele, _dep, _terr_ele, _terr_dep, codes, *_ = _st_analysis(
            self._inferior_leads(t_amp=-0.1)
        )
        self.assertIn("IMIAP", codes)
        self.assertNotIn("IMIA", codes)

    def test_anterior_probable_code_requires_upright_t(self) -> None:
        leads = {
            "V2": _rep("V2", st_on_mv=0.16, t_amp_mv=-0.1),
            "V3": _rep("V3", st_on_mv=0.16, t_amp_mv=-0.1),
            "V4": _rep("V4", st_on_mv=0.0),
            "V5": _rep("V5", st_on_mv=0.0),
        }
        _ele, _dep, terr_ele, _terr_dep, codes, *_ = _st_analysis(leads)
        self.assertIn("anterior", terr_ele)
        self.assertNotIn("AMIAP", codes)


class MiQWaveDurationGateTests(unittest.TestCase):
    def test_any_mi_ratio_requires_duration_gate(self) -> None:
        reps_short = {"V1": _rep("V1", q_amp_mv=-0.3, r_amp_mv=1.0, q_duration_ms=20.0)}
        reps_long = {"V1": _rep("V1", q_amp_mv=-0.3, r_amp_mv=1.0, q_duration_ms=40.0)}

        evidence_short = build_mi_evidence(
            representative_leads=reps_short, st_elevation_leads={}, st_depression_leads={},
            stemi_codes=[], posterior_mi_suspected=False, r_progression_class="normal",
            is_pediatric=False,
        )
        evidence_long = build_mi_evidence(
            representative_leads=reps_long, st_elevation_leads={}, st_depression_leads={},
            stemi_codes=[], posterior_mi_suspected=False, r_progression_class="normal",
            is_pediatric=False,
        )
        self.assertFalse(evidence_short["q_by_lead"]["V1"]["q_wave_mi_ratio"])
        self.assertTrue(evidence_long["q_by_lead"]["V1"]["q_wave_mi_ratio"])


class QrsWidthWideConsensusEscalationTests(unittest.TestCase):
    # V1 R~0 (QS-like) + broad R in I/V6 is the classic LBBB amplitude
    # pattern used by _classify_qrs_width's >=120ms branch.
    _LBBB_LEADS = {
        "V1": _rep("V1", r_amp_mv=0.02),
        "I":  _rep("I", r_amp_mv=0.90),
        "V6": _rep("V6", r_amp_mv=1.30),
    }

    def test_wide_consensus_escalates_narrow_measurement_to_lbbb(self) -> None:
        # Measurement channel (P75, corroboration-gated) says 95ms -- narrow.
        # Classification channel (P90) independently reaches 125ms -- BBB
        # territory -- and morphology corroborates LBBB, so this should
        # escalate rather than report "normal".
        width_class, bbb, disagreement = _resolve_qrs_width_class(
            95.0, 125.0, self._LBBB_LEADS, {}
        )
        self.assertEqual("bbb", width_class)
        self.assertEqual("LBBB", bbb)
        self.assertTrue(disagreement)

    def test_no_escalation_when_channels_agree(self) -> None:
        # qrs_wide_ms falls back to qrs_ms when the P90 channel has no data of
        # its own (see features._global_qrs_wide) -- equal values must not be
        # treated as a disagreement.
        width_class, bbb, disagreement = _resolve_qrs_width_class(
            95.0, 95.0, self._LBBB_LEADS, {}
        )
        self.assertEqual("normal", width_class)
        self.assertIsNone(bbb)
        self.assertFalse(disagreement)

    def test_no_escalation_without_corroborating_morphology(self) -> None:
        # Wide P90 consensus alone, without matching BBB amplitude morphology,
        # is much weaker evidence (could just be a noisy terminal offset) and
        # must not escalate.
        non_bbb_leads = {
            "V1": _rep("V1", r_amp_mv=0.9),
            "I":  _rep("I", r_amp_mv=0.05),
            "V6": _rep("V6", r_amp_mv=0.05),
        }
        width_class, bbb, disagreement = _resolve_qrs_width_class(
            95.0, 125.0, non_bbb_leads, {}
        )
        self.assertNotEqual("bbb", width_class)
        self.assertFalse(disagreement)

    def test_no_escalation_when_measurement_channel_already_bbb(self) -> None:
        # Already-detected BBB via the primary channel should not be
        # re-derived or flagged as a disagreement.
        width_class, bbb, disagreement = _resolve_qrs_width_class(
            130.0, 130.0, self._LBBB_LEADS, {}
        )
        self.assertEqual("bbb", width_class)
        self.assertEqual("LBBB", bbb)
        self.assertFalse(disagreement)


class AtrialCriterionFidelityTests(unittest.TestCase):
    """The LAE paths must apply the published bars, not looser proxies."""

    @staticmethod
    def _notched(lead: str, *, interval_ms: float) -> RepresentativeLeadFeatures:
        return _rep(lead, p_notched=True, p_notch_interval_ms=interval_ms, p_amp_mv=0.08)

    def test_notched_p_needs_forty_millisecond_interpeak_interval(self) -> None:
        reps = {"II": self._notched("II", interval_ms=25.0)}
        _cls, _rae, lae_suspected, *_ = _p_wave_morphology(
            reps, {"II": 120.0}, ptf_v1=None
        )
        self.assertFalse(lae_suspected)

        reps = {"II": self._notched("II", interval_ms=45.0)}
        _cls, _rae, lae_suspected, *_ = _p_wave_morphology(
            reps, {"II": 120.0}, ptf_v1=None
        )
        self.assertTrue(lae_suspected)

    def test_notched_p_criterion_ignores_precordial_leads(self) -> None:
        # Scanning all 12 leads made aVL/III/V4 the usual triggers on a measure
        # whose per-lead MAE is 28.5 ms; the criterion is a limb-lead one.
        reps = {"V4": self._notched("V4", interval_ms=45.0)}
        _cls, _rae, lae_suspected, *_ = _p_wave_morphology(
            reps, {"V4": 120.0}, ptf_v1=None
        )
        self.assertFalse(lae_suspected)

    def test_ptf_between_the_bars_grades_but_does_not_call_lae(self) -> None:
        reps = {"V1": _rep("V1")}
        _cls, _rae, lae_suspected, _def, ptf_class = _p_wave_morphology(
            reps, {}, ptf_v1=-3.0
        )
        self.assertEqual("probable_lae", ptf_class)
        self.assertFalse(lae_suspected)

    def test_ptf_at_morris_bar_calls_lae(self) -> None:
        reps = {"V1": _rep("V1")}
        _cls, _rae, lae_suspected, *_ = _p_wave_morphology(reps, {}, ptf_v1=-5.0)
        self.assertTrue(lae_suspected)

    def test_p_duration_prefers_the_consensus_scale(self) -> None:
        from types import SimpleNamespace

        from feature_extraction.ecgfeat.interpret import _median_p_dur_per_lead

        beats = [SimpleNamespace(lead="II", p_dur_ms=78.0) for _ in range(4)]
        reps = {"II": _rep("II", p_dur_consensus_ms=112.0)}
        self.assertEqual({"II": 78.0}, _median_p_dur_per_lead(beats, 500))
        self.assertEqual({"II": 112.0}, _median_p_dur_per_lead(beats, 500, reps))

    def test_p_duration_falls_back_to_raw_when_no_consensus(self) -> None:
        from types import SimpleNamespace

        from feature_extraction.ecgfeat.interpret import _median_p_dur_per_lead

        beats = [SimpleNamespace(lead="II", p_dur_ms=78.0)]
        self.assertEqual(
            {"II": 78.0}, _median_p_dur_per_lead(beats, 500, {"II": _rep("II")})
        )


class PtfV1DriftGuardTests(unittest.TestCase):
    def test_monotonic_drift_is_not_a_terminal_force(self) -> None:
        import numpy as np

        from feature_extraction.ecgfeat.delineate import _ptf_v1

        # Baseline falling straight through the P window: the trough is the last
        # sample, so there is no terminal negative *deflection* to measure.
        sig = np.linspace(0.20, -0.05, 60)
        self.assertIsNone(_ptf_v1(sig, 0, len(sig) - 1, 0.0, 500))

    def test_real_terminal_deflection_is_still_measured(self) -> None:
        import numpy as np

        from feature_extraction.ecgfeat.delineate import _ptf_v1

        # Upright P, then a negative trough that returns toward baseline.
        up = np.concatenate([np.linspace(0.0, 0.10, 15), np.linspace(0.10, 0.0, 15)])
        down = np.concatenate([np.linspace(0.0, -0.12, 15), np.linspace(-0.12, -0.01, 15)])
        sig = np.concatenate([up, down])
        value = _ptf_v1(sig, 0, len(sig) - 1, 0.0, 500)
        self.assertIsNotNone(value)
        self.assertLess(value, -4.0)


class PIsoelectricReferenceTests(unittest.TestCase):
    """P amplitude must be referenced to tissue that is not the P wave."""

    @staticmethod
    def _beat(fs: int = 500):
        """A 100 ms upright P on a flat baseline, then a PR segment and QRS onset."""
        import numpy as np

        sig = np.zeros(int(0.6 * fs))
        p_on, p_off = int(0.10 * fs), int(0.20 * fs)
        hump = np.sin(np.linspace(0.0, np.pi, p_off - p_on + 1)) * 0.15
        sig[p_on:p_off + 1] = hump
        qrs_on = int(0.32 * fs)
        return sig, p_on, p_off, qrs_on, p_on + (p_off - p_on) // 2

    def test_reference_is_taken_off_the_p_wave(self) -> None:
        from feature_extraction.ecgfeat.delineate import _p_isoelectric_reference

        sig, p_on, p_off, qrs_on, _peak = self._beat()
        level, source = _p_isoelectric_reference(
            sig, p_on=p_on, p_off=p_off, qrs_on=qrs_on, fs=500, fallback=0.05
        )
        self.assertEqual("pr_and_tp_sidebands", source)
        self.assertAlmostEqual(0.0, level, places=6)

    def test_legacy_window_would_have_lost_amplitude(self) -> None:
        # The old reference is the median of a window the P wave sits inside, so
        # it is lifted toward the P and the measured amplitude shrinks.
        import numpy as np

        from feature_extraction.ecgfeat.delineate import _p_isoelectric_reference

        sig, p_on, p_off, qrs_on, peak = self._beat()
        r_index = qrs_on + int(0.02 * 500)
        lo, hi = max(0, r_index - int(0.22 * 500)), r_index - int(0.08 * 500)
        legacy = float(np.median(sig[lo:hi]))
        fixed, _source = _p_isoelectric_reference(
            sig, p_on=p_on, p_off=p_off, qrs_on=qrs_on, fs=500, fallback=legacy
        )
        self.assertGreater(legacy, fixed)
        self.assertGreater(float(sig[peak]) - fixed, float(sig[peak]) - legacy)

    def test_falls_back_when_no_isoelectric_stretch_exists(self) -> None:
        from feature_extraction.ecgfeat.delineate import _p_isoelectric_reference

        sig, p_on, p_off, _qrs_on, _peak = self._beat()
        level, source = _p_isoelectric_reference(
            sig, p_on=p_on, p_off=p_off, qrs_on=p_off + 2, fs=500, fallback=0.077
        )
        self.assertEqual("single_sideband", source)
        self.assertAlmostEqual(0.0, level, places=6)

        level, source = _p_isoelectric_reference(
            sig, p_on=None, p_off=None, qrs_on=None, fs=500, fallback=0.077
        )
        self.assertEqual("legacy_window_fallback", source)
        self.assertAlmostEqual(0.077, level, places=6)


class RaeSingleLeadCriterionTests(unittest.TestCase):
    def test_tall_p_in_lead_ii_alone_confirms_rae(self) -> None:
        # 0.26 mV in II is the classic 2.5 mm criterion; requiring a second limb
        # lead dropped it (record 09237) even alongside definite LAE.
        reps = {"II": _rep("II", p_amp_mv=0.26)}
        p_class, rae_leads, *_ = _p_wave_morphology(reps, {"II": 100.0}, ptf_v1=None)
        self.assertEqual(["II"], rae_leads)
        self.assertEqual("rae", p_class)

    def test_tall_p_in_ii_with_definite_lae_reads_biatrial(self) -> None:
        reps = {
            "II": _rep("II", p_amp_mv=0.26),
            "V1": _rep("V1", p_terminal_amp_mv=-0.16, p_terminal_duration_ms=65.0),
        }
        p_class, _rae, _susp, lae_definite, _ptf = _p_wave_morphology(
            reps, {"II": 100.0}, ptf_v1=None
        )
        self.assertTrue(lae_definite)
        self.assertEqual("bae", p_class)

    def test_a_wide_tall_p_is_not_right_atrial(self) -> None:
        # Tall *and* wide is left atrial (or biatrial), never RAE on its own.
        reps = {"II": _rep("II", p_amp_mv=0.26)}
        p_class, *_ = _p_wave_morphology(reps, {"II": 140.0}, ptf_v1=None)
        self.assertNotEqual("rae", p_class)


class LimbPAmplitudeAlignmentTests(unittest.TestCase):
    """The six limb-lead P amplitudes must form a valid instantaneous vector."""

    @staticmethod
    def _scene(peak_offsets):
        """Build I/II and the derived limb leads from one atrial dipole."""
        import numpy as np

        from feature_extraction.ecgfeat.models import LeadBeatFeatures, WaveBounds

        fs = 500
        n = int(0.6 * fs)
        t = np.arange(n)
        p_on, p_off = int(0.10 * fs), int(0.20 * fs)
        hump = np.zeros(n)
        hump[p_on:p_off + 1] = np.sin(np.linspace(0.0, np.pi, p_off - p_on + 1))
        lead_i = 0.08 * hump
        lead_ii = 0.15 * hump
        sigs = {
            "I": lead_i,
            "II": lead_ii,
            "III": lead_ii - lead_i,
            "aVR": -(lead_i + lead_ii) / 2.0,
            "aVL": lead_i - lead_ii / 2.0,
            "aVF": lead_ii - lead_i / 2.0,
        }
        order = ["I", "II", "III", "aVR", "aVL", "aVF"]
        ecg = np.vstack([sigs[k] for k in order])
        lead_to_index = {k: i for i, k in enumerate(order)}
        true_peak = p_on + (p_off - p_on) // 2
        beats = []
        for lead in order:
            beats.append(LeadBeatFeatures(
                lead=lead, beat_id=0,
                p=WaveBounds(onset=p_on, peak=true_peak + peak_offsets.get(lead, 0), offset=p_off),
                qrs=WaveBounds(onset=int(0.30 * fs), peak=int(0.33 * fs), offset=int(0.36 * fs)),
                t=WaveBounds(onset=None, peak=None, offset=None),
                qt_ms=None, pr_ms=None, qrs_ms=None,
                p_amp_mv=float(sigs[lead][true_peak + peak_offsets.get(lead, 0)]),
                qrs_area=None, q_amp_mv=None, r_amp_mv=None, s_amp_mv=None,
                st_on_mv=None, st_mid_mv=None, st_80ms_mv=None, t_amp_mv=None,
                j_index=None,
            ))
        return {0: beats}, ecg, lead_to_index, fs, t

    def test_amplitudes_satisfy_the_goldberger_identity(self) -> None:
        from feature_extraction.ecgfeat.delineate import _align_limb_p_amplitudes

        by_beat, ecg, idx, fs, _t = self._scene({"aVR": 8, "aVL": -6, "III": 5})
        _align_limb_p_amplitudes(by_beat, ecg, idx, fs)
        amps = {bf.lead: bf.p_amp_mv for bf in by_beat[0]}
        self.assertAlmostEqual(
            amps["aVR"], -(amps["I"] + amps["II"]) / 2.0, places=6
        )
        self.assertLess(amps["aVR"], 0.0)
        self.assertGreater(amps["II"], 0.0)
        self.assertTrue(all(bf.p_amp_source == "limb_simultaneous" for bf in by_beat[0]))

    def test_widely_disagreeing_peaks_keep_their_own_measurement(self) -> None:
        from feature_extraction.ecgfeat.delineate import _align_limb_p_amplitudes

        by_beat, ecg, idx, fs, _t = self._scene({"aVR": 45})  # 90 ms away at 500 Hz
        before = {bf.lead: bf.p_amp_mv for bf in by_beat[0]}
        _align_limb_p_amplitudes(by_beat, ecg, idx, fs)
        after = {bf.lead: bf.p_amp_mv for bf in by_beat[0]}
        self.assertEqual(before, after)
        self.assertTrue(all(bf.p_amp_source == "lead_local_peaks_disagree"
                            for bf in by_beat[0]))

    def test_too_few_limb_leads_is_left_alone(self) -> None:
        from feature_extraction.ecgfeat.delineate import _align_limb_p_amplitudes

        by_beat, ecg, idx, fs, _t = self._scene({})
        by_beat[0] = by_beat[0][:2]
        before = {bf.lead: bf.p_amp_mv for bf in by_beat[0]}
        _align_limb_p_amplitudes(by_beat, ecg, idx, fs)
        self.assertEqual(before, {bf.lead: bf.p_amp_mv for bf in by_beat[0]})
        self.assertTrue(all(bf.p_amp_source is None for bf in by_beat[0]))


class PAmplitudeMeasurabilityTests(unittest.TestCase):
    """A P no larger than the drift it sits on is not a measurement."""

    @staticmethod
    def _beats(amps, drifts):
        from types import SimpleNamespace

        return [SimpleNamespace(p_amp_mv=a, p_window_drift_mv=d)
                for a, d in zip(amps, drifts)]

    def test_p_below_the_drift_is_withheld(self) -> None:
        from feature_extraction.ecgfeat.features import _p_amplitude_if_measurable

        beats = self._beats([0.03, 0.028, 0.031, 0.029], [0.05] * 4)
        self.assertIsNone(_p_amplitude_if_measurable(beats[0], beats))

    def test_p_above_the_drift_is_reported(self) -> None:
        from feature_extraction.ecgfeat.features import _p_amplitude_if_measurable

        beats = self._beats([0.12, 0.13, 0.11, 0.12], [0.04] * 4)
        value = _p_amplitude_if_measurable(beats[0], beats)
        self.assertIsNotNone(value)
        self.assertAlmostEqual(0.12, value, places=3)

    def test_no_drift_estimate_leaves_the_amplitude_alone(self) -> None:
        from feature_extraction.ecgfeat.features import _p_amplitude_if_measurable

        beats = self._beats([0.03, 0.028, 0.031], [None, None, None])
        self.assertIsNotNone(_p_amplitude_if_measurable(beats[0], beats))

    def test_unmeasurable_amplitude_does_not_confirm_lae_on_duration_alone(self) -> None:
        # Withholding an unmeasurable amplitude must not read as "criterion met".
        reps = {"II": _rep("II", p_amp_mv=None)}
        _cls, _rae, lae_suspected, *_ = _p_wave_morphology(
            reps, {"II": 130.0}, ptf_v1=None
        )
        self.assertFalse(lae_suspected)

        reps = {"II": _rep("II", p_amp_mv=0.12)}
        _cls, _rae, lae_suspected, *_ = _p_wave_morphology(
            reps, {"II": 130.0}, ptf_v1=None
        )
        self.assertTrue(lae_suspected)

    def test_drift_estimate_ignores_the_p_bump(self) -> None:
        import numpy as np

        from feature_extraction.ecgfeat.delineate import _p_window_drift_mv

        fs = 500
        qrs_on = int(0.30 * fs)
        # Flat baseline with a 100 ms P bump: a least-squares slope would be
        # dragged by the bump, the robust one should read ~zero drift.
        sig = np.zeros(qrs_on + 50)
        p_on = qrs_on - int(0.20 * fs)
        sig[p_on:p_on + int(0.10 * fs)] = np.sin(
            np.linspace(0.0, np.pi, int(0.10 * fs))) * 0.15
        flat = _p_window_drift_mv(sig, qrs_on, fs)
        self.assertIsNotNone(flat)
        self.assertLess(flat, 0.02)
        # Same bump on a 200 µV ramp: the drift must now be seen.
        ramped = sig + np.linspace(0.0, 0.20, len(sig))
        sloped = _p_window_drift_mv(ramped, qrs_on, fs)
        self.assertIsNotNone(sloped)
        self.assertGreater(sloped, 0.03)


class PRepresentativePolarityConsensusTests(unittest.TestCase):
    """A record-level P amplitude must not be one beat's sample."""

    @staticmethod
    def _beats(values):
        from types import SimpleNamespace

        return [SimpleNamespace(p_amp_mv=v, p_area=abs(v) * 10.0) for v in values]

    def test_minority_sign_representative_beat_is_outvoted(self) -> None:
        from feature_extraction.ecgfeat.features import _p_polarity_consensus_value

        beats = self._beats([0.11, 0.12, 0.10, 0.13, -0.09])
        rep = beats[-1]          # the mis-located beat happens to be representative
        value = _p_polarity_consensus_value(rep, beats, "p_amp_mv")
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(0.115, value, places=3)

    def test_no_majority_falls_back_to_the_median_of_all_beats(self) -> None:
        from feature_extraction.ecgfeat.features import _p_polarity_consensus_value

        beats = self._beats([0.10, 0.11, -0.10, -0.11])
        value = _p_polarity_consensus_value(beats[0], beats, "p_amp_mv")
        self.assertAlmostEqual(0.0, value, places=6)

    def test_consensus_carries_over_to_a_paired_field(self) -> None:
        # p_area is aggregated over the same dominant-polarity beats, so it cannot
        # describe a different subset than p_amp_mv does.
        from feature_extraction.ecgfeat.features import _p_polarity_consensus_value

        beats = self._beats([0.11, 0.12, 0.10, 0.13, -0.09])
        area = _p_polarity_consensus_value(beats[-1], beats, "p_area")
        self.assertAlmostEqual(1.15, area, places=3)

    def test_single_beat_records_still_report_something(self) -> None:
        from feature_extraction.ecgfeat.features import _p_polarity_consensus_value

        beats = self._beats([0.09])
        self.assertAlmostEqual(
            0.09, _p_polarity_consensus_value(beats[0], beats, "p_amp_mv"), places=6
        )


if __name__ == "__main__":
    unittest.main()
