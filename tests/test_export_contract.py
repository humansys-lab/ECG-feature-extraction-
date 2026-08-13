from __future__ import annotations

import sys
import tempfile
import types
import unittest
from copy import deepcopy

if "wfdb" not in sys.modules:
    sys.modules["wfdb"] = types.SimpleNamespace(rdrecord=None, rdann=None)

import compare_annotations as compare

from feature_extraction.ecgfeat.models import (
    BeatAnnotation,
    ECGFeatures,
    GlobalFeatures,
    GroupFeatures,
    LeadBeatFeatures,
    LeadQuality,
    PatientMeta,
    RepresentativeLeadFeatures,
    WaveBounds,
)


def _make_features() -> ECGFeatures:
    return ECGFeatures(
        fs=500,
        quality={
            "I": LeadQuality(
                lead="I",
                baseline_wander_score=0.0,
                muscle_noise_score=0.0,
                powerline_score=0.0,
                clipping_score=0.0,
                flatline_score=0.0,
                missing=False,
                reliable=True,
                grade="Q1",
                reason_codes=["baseline_wander"],
            )
        },
        beats=[
            BeatAnnotation(
                beat_id=0,
                r_index=500,
                paced=False,
                group_id=1,
                rr_prev_ms=None,
                rr_next_ms=1000.0,
            ),
            BeatAnnotation(
                beat_id=1,
                r_index=1000,
                paced=False,
                group_id=1,
                rr_prev_ms=1000.0,
                rr_next_ms=None,
            ),
        ],
        beat_features=[
            LeadBeatFeatures(
                lead="I",
                beat_id=0,
                p=WaveBounds(onset=420, peak=440, offset=460),
                qrs=WaveBounds(onset=480, peak=500, offset=525),
                t=WaveBounds(onset=560, peak=650, offset=720),
                qt_ms=480.0,
                pr_ms=120.0,
                qrs_ms=90.0,
                p_amp_mv=0.12,
                qrs_area=1.5,
                q_amp_mv=-0.05,
                r_amp_mv=0.8,
                s_amp_mv=-0.2,
                st_on_mv=0.01,
                st_mid_mv=0.02,
                st_80ms_mv=0.03,
                t_amp_mv=0.2,
                j_index=525,
                qrs_signed_area=0.9,
                delta_present=True,
                p_confidence=0.9,
                qrs_confidence=0.95,
                beat_measurement_reliable=True,
            ),
            LeadBeatFeatures(
                lead="II",
                beat_id=0,
                p=WaveBounds(onset=418, peak=438, offset=458),
                qrs=WaveBounds(onset=480, peak=500, offset=525),
                t=WaveBounds(onset=560, peak=650, offset=720),
                qt_ms=480.0,
                pr_ms=122.0,
                qrs_ms=92.0,
                p_amp_mv=0.10,
                qrs_area=1.2,
                q_amp_mv=-0.04,
                r_amp_mv=0.7,
                s_amp_mv=-0.3,
                st_on_mv=0.01,
                st_mid_mv=0.02,
                st_80ms_mv=0.03,
                t_amp_mv=0.2,
                j_index=525,
                qrs_signed_area=-0.4,
                delta_present=False,
                p_confidence=0.7,
                qrs_confidence=0.9,
                beat_measurement_reliable=True,
            ),
        ],
        representative_leads={
            "I": RepresentativeLeadFeatures(
                lead="I",
                params={
                    "pr_ms": 120.0,
                    "qrs_ms": 90.0,
                    "p_confidence_mean": 0.9,
                    "r_amp_mv": 0.8,
                    "s_amp_mv": -0.2,
                    "q_amp_mv": -0.05,
                    "qrs_area": 1.5,
                    "reliable_for_p": True,
                    "reliable_for_qrs": True,
                    "reliable_for_t": True,
                    "reliable_for_qt": True,
                },
                variance={},
            )
        },
        groups={
            1: GroupFeatures(
                group_id=1,
                member_count=2,
                member_pct=100.0,
                longest_run=2,
                mean_rr_ms=1000.0,
                mean_pr_ms=150.0,
                mean_qrs_ms=90.0,
                mean_qt_ms=400.0,
                mean_ventr_rate_bpm=60.0,
            )
        },
        global_features=GlobalFeatures(
            heart_rate_bpm=60.0,
            atrial_rate_bpm=60.0,
            pr_ms=150.0,
            qrs_ms=90.0,
            qt_ms=400.0,
            qtc_bazett_ms=410.0,
            qtc_fridericia_ms=405.0,
            p_axis_deg=40.0,
            qrs_axis_deg=55.0,
            t_axis_deg=35.0,
            st_axis_deg=20.0,
            qt_dispersion_ms=25.0,
        ),
        metadata={
            "input_fs": 1000,
            "internal_fs": 500,
            "lead_order": ["I", "II"],
            "n_beats": 2,
            "lead_reversal": {"limb": {}, "precordial": {}},
            "pacing_state": "unknown",
            "representative_group_id": 1,
            "record_quality": {
                "record_grade": "Q2",
                "reason_codes": ["baseline_wander"],
                "rejected_functions": ["p_measurement"],
            },
        },
    )


class ExportContractTests(unittest.TestCase):
    def test_global_and_t_wave_provenance_fields_are_serializable(self) -> None:
        from dataclasses import asdict

        from feature_extraction.ecgfeat.models import GlobalFeatures, LeadBeatFeatures, WaveBounds

        beat = LeadBeatFeatures(
            lead="II",
            beat_id=0,
            p=WaveBounds(None, None, None),
            qrs=WaveBounds(100, 120, 140),
            t=WaveBounds(190, 240, 310),
            qt_ms=420.0,
            pr_ms=None,
            qrs_ms=80.0,
            p_amp_mv=None,
            qrs_area=1.0,
            q_amp_mv=0.0,
            r_amp_mv=1.0,
            s_amp_mv=-0.2,
            st_on_mv=-0.05,
            st_mid_mv=-0.04,
            st_80ms_mv=-0.02,
            t_amp_mv=0.18,
            j_index=140,
            t_peak_path="polarity_cluster",
            t_polarity_expected=1,
            t_polarity_observed=1,
            st_t_confusion=True,
            t_confidence_reason="later_signed_t_after_st_trough",
        )
        payload = asdict(beat)
        self.assertEqual("polarity_cluster", payload["t_peak_path"])
        self.assertEqual(1, payload["t_polarity_expected"])
        self.assertEqual(1, payload["t_polarity_observed"])
        self.assertTrue(payload["st_t_confusion"])
        self.assertEqual("later_signed_t_after_st_trough", payload["t_confidence_reason"])

        global_features = GlobalFeatures(
            heart_rate_bpm=70.0,
            atrial_rate_bpm=70.0,
            pr_ms=None,
            qrs_ms=130.0,
            qt_ms=None,
            qtc_bazett_ms=None,
            qtc_fridericia_ms=None,
            p_axis_deg=None,
            qrs_axis_deg=20.0,
            t_axis_deg=None,
            st_axis_deg=None,
            qt_dispersion_ms=None,
            qt_path="low_qt_support",
            qt_confidence_reason="one_independent_jt_checked_lead",
            qt_excluded_leads={"V1": "low_t_amplitude"},
            qt_lead_weights={"II": 0.82},
            consensus_vs_independent_per_lead={"II": {"qt_ms": 430.0, "qt_consensus_ms": None}},
        )
        global_payload = asdict(global_features)
        self.assertEqual("low_qt_support", global_payload["qt_path"])
        self.assertEqual(
            "one_independent_jt_checked_lead",
            global_payload["qt_confidence_reason"],
        )
        self.assertEqual("low_t_amplitude", global_payload["qt_excluded_leads"]["V1"])
        self.assertEqual(0.82, global_payload["qt_lead_weights"]["II"])
        self.assertEqual(
            {"qt_ms": 430.0, "qt_consensus_ms": None},
            global_payload["consensus_vs_independent_per_lead"]["II"],
        )

    def test_structured_payload_surfaces_qt_path_provenance(self) -> None:
        from feature_extraction.ecgfeat.export import build_structured_payload

        features = _make_features()
        features.global_features.qt_path = "wide_qrs_jt"
        features.global_features.qt_confidence_reason = "jt_interval_sanity_checked"
        features.global_features.qt_excluded_leads = {"V1": "short_jt"}
        features.global_features.qt_lead_weights = {"II": 0.8}
        features.global_features.consensus_vs_independent_per_lead = {
            "II": {
                "qt_ms": 430.0,
                "qt_consensus_ms": 420.0,
                "jt_ms": 330.0,
            }
        }

        payload = build_structured_payload(features)

        global_qt = payload["provenance"]["global_qt"]
        self.assertEqual("wide_qrs_jt", global_qt["path"])
        self.assertEqual("jt_interval_sanity_checked", global_qt["confidence_reason"])
        self.assertEqual({"V1": "short_jt"}, global_qt["excluded_leads"])
        self.assertEqual({"II": 0.8}, global_qt["lead_weights"])
        self.assertEqual(
            {"qt_ms": 430.0, "qt_consensus_ms": 420.0, "jt_ms": 330.0},
            global_qt["consensus_vs_independent_per_lead"]["II"],
        )

    def test_structured_payload_surfaces_global_p_duration_provenance(self) -> None:
        from feature_extraction.ecgfeat.export import build_structured_payload

        features = _make_features()
        features.global_features.p_duration_ms = 88.0
        features.global_features.p_duration_source = "representative_multilead_consensus"
        features.global_features.p_duration_used_leads = ["I", "II", "aVF"]
        features.global_features.p_duration_support = 5
        features.global_features.p_duration_spread_ms = 4.0
        features.global_features.p_duration_reliability = "reliable"

        payload = build_structured_payload(features)

        self.assertEqual(88.0, payload["global"]["p_duration_ms"])
        self.assertEqual(
            {
                "source": "representative_multilead_consensus",
                "used_leads": ["I", "II", "aVF"],
                "support": 5,
                "spread_ms": 4.0,
                "reliability": "reliable",
            },
            payload["provenance"]["global_p_duration"],
        )

    def test_full_feature_json_contains_rhythm_inputs_for_rule_engine(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        payload = to_dict(_make_features())

        self.assertIn("rhythm_inputs", payload)
        rhythm = payload["rhythm_inputs"]
        self.assertEqual(
            {
                "record",
                "beats",
                "native_beat_profiles",
                "p_events",
                "background",
                "pacing",
                "af_afl",
                "preexcitation",
                "av_block",
                "aberrancy",
                "statement_evidence",
            },
            set(rhythm.keys()),
        )
        self.assertEqual(1000.0, rhythm["background"]["clean_rr_ms"])
        self.assertEqual([0], rhythm["preexcitation"]["delta_beat_ids"])
        self.assertEqual(["I"], rhythm["beats"][0]["delta_leads"])
        self.assertEqual("positive", rhythm["beats"][0]["qrs_polarity_signature"]["I"]["polarity"])
        native_rows = rhythm["native_beat_profiles"]["rows"]
        self.assertEqual(2, len(native_rows))
        self.assertEqual("I", native_rows[0]["lead"])
        self.assertEqual(-0.05, native_rows[0]["q_amp_mv"])
        self.assertFalse(native_rows[0]["paced"])
        self.assertEqual(False, rhythm["af_afl"]["qrst_subtraction"]["available"])
        self.assertEqual(False, rhythm["af_afl"]["qrst_subtraction_quality"]["available"])
        self.assertEqual(False, rhythm["af_afl"]["atrial_residual_signal_summary"]["available"])
        self.assertEqual("not_implemented", rhythm["pacing"]["artifact_confidence"]["reason"])

    def test_rhythm_inputs_treat_missing_pacing_state_as_off_without_context(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata.pop("pacing_state", None)

        pacing = to_dict(features)["rhythm_inputs"]["pacing"]

        self.assertFalse(pacing["enabled"])
        self.assertEqual("off", pacing["state"])

    def test_rhythm_inputs_export_metadata_backed_pacing_failure_facts(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["rhythm_analysis"] = {
            "pacing_failures": {
                "artifact_confidence": 1.0,
                "capture_failure_suspected": True,
                "sensing_failure_suspected": False,
            }
        }

        pacing = to_dict(features)["rhythm_inputs"]["pacing"]
        self.assertEqual(1.0, pacing["artifact_confidence"])
        self.assertTrue(pacing["capture_failure_suspected"])
        self.assertFalse(pacing["sensing_failure_suspected"])

    def test_full_feature_json_contains_real_statement_evidence_shape(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["rhythm_analysis"] = {
            "rule_summary": {
                "statement_evidence": {
                    "available": True,
                    "primary_statement": "multiple_apc",
                    "additional_statements": ["pause"],
                    "statements": [{"code": "multiple_apc", "confidence": 0.8}],
                    "stop_further_interpretation": False,
                    "bypass_remaining_algorithm": False,
                }
            }
        }

        payload = to_dict(features)
        evidence = payload["rhythm_inputs"]["statement_evidence"]
        self.assertTrue(evidence["available"])
        self.assertEqual("multiple_apc", evidence["primary_statement"])

    def test_rhythm_inputs_export_av_block_and_aberrancy_evidence(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["rhythm_analysis"] = {
            "rule_summary": {
                "pauses": {
                    "second_degree_avb": "second_degree_av_block",
                    "escape_origin": "supraventricular",
                    "av_block_evidence": {
                        "atrial_events_per_rr": [1, 2, 1],
                        "pr_series_ms": [160.0, None, 162.0],
                        "dropped_p_evidence": True,
                    },
                },
                "post_pause_or_interpolated_beats": [
                    {"beat_id": 2, "type": "interpolated_candidate"}
                ],
            }
        }

        rhythm = to_dict(features)["rhythm_inputs"]
        self.assertEqual("second_degree_av_block", rhythm["av_block"]["second_degree_avb"])
        self.assertTrue(rhythm["av_block"]["evidence"]["dropped_p_evidence"])
        self.assertEqual(
            [{"beat_id": 2, "type": "interpolated_candidate"}],
            rhythm["aberrancy"]["post_pause_or_interpolated_beats"],
        )

    def test_rhythm_inputs_normalize_metadata_backed_p_events(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["rhythm_analysis"] = {
            "atrial_events": [
                {
                    "p_event_id": 7,
                    "sample": 438,
                    "time_ms": 876.0,
                    "confidence": 0.8,
                    "associated_qrs_beat_id": 0,
                    "association_type": "conducted",
                    "pr_ms": 84.0,
                    "source_leads": ["II", "V1"],
                    "duration_ms": 92.0,
                    "amplitude_mv": 0.08,
                    "area_mv_ms": 4.6,
                    "signed_area_mv_ms": 3.9,
                    "template_similarity": 0.91,
                    "pp_ms": 820.0,
                    "detection_method": "composite_qrst_residual_derivative",
                }
            ]
        }

        p_event = to_dict(features)["rhythm_inputs"]["p_events"][0]

        self.assertEqual(7, p_event["p_event_id"])
        self.assertEqual(438, p_event["sample"])
        self.assertEqual(876.0, p_event["time_ms"])
        self.assertEqual(0.8, p_event["confidence"])
        self.assertEqual(0, p_event["associated_qrs_beat_id"])
        self.assertEqual("conducted", p_event["association_type"])
        self.assertEqual(84.0, p_event["pr_ms"])
        self.assertEqual(["II", "V1"], p_event["source_leads"])
        self.assertIsNone(p_event["onset_ms"])
        self.assertIsNone(p_event["offset_ms"])
        self.assertIsNone(p_event["axis_deg"])
        self.assertIsNone(p_event["morphology"])
        self.assertEqual("independent_atrial_event_stream", p_event["source"])
        self.assertEqual(92.0, p_event["duration_ms"])
        self.assertEqual(0.08, p_event["amplitude_mv"])
        self.assertEqual(4.6, p_event["area_mv_ms"])
        self.assertEqual(3.9, p_event["signed_area_mv_ms"])
        self.assertEqual(0.91, p_event["template_similarity"])
        self.assertEqual(820.0, p_event["pp_ms"])
        self.assertEqual("composite_qrst_residual_derivative", p_event["detection_method"])

    def test_rhythm_inputs_do_not_report_scaffold_residual_as_validated_qrst_subtraction(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["rhythm_analysis"] = {
            "atrial_residual": {
                "available": True,
                "method": "baseline_subtracted_qrst_window_scaffold",
                "validated_qrst_subtraction": False,
                "residual_rms_mv": 0.04,
                "repetitiveness": 0.72,
                "stability": 0.68,
                "dominant_cycle_ms": 210.0,
            },
            "af_afl_summary": {
                "rr_cv": 0.22,
                "probable_af": False,
                "probable_flutter": True,
                "af_afl_indeterminate": False,
                "atrial_rhythm_classification": "atrial_flutter",
                "diagnostic_confidence": 0.88,
                "f_wave_confidence": 0.10,
                "f_wave_multilead_consensus": False,
                "F_wave_confidence": 0.88,
                "F_wave_multilead_consensus": True,
                "flutter_wave_confidence": 0.70,
            },
        }

        af_afl = to_dict(features)["rhythm_inputs"]["af_afl"]

        self.assertFalse(af_afl["qrst_subtraction"]["available"])
        self.assertTrue(af_afl["qrst_subtraction"]["scaffold_available"])
        self.assertEqual(
            "baseline_subtracted_qrst_window_scaffold",
            af_afl["qrst_subtraction"]["method"],
        )
        self.assertEqual(
            "qrst_template_subtraction_not_implemented",
            af_afl["qrst_subtraction"]["reason"],
        )
        self.assertEqual(0.68, af_afl["atrial_signal_stability"])
        self.assertEqual(0.72, af_afl["atrial_signal_repetitiveness"])
        self.assertEqual(210.0, af_afl["dominant_atrial_cycle_ms"])
        self.assertEqual(0.70, af_afl["flutter_wave_confidence"])
        self.assertTrue(af_afl["probable_flutter"])
        self.assertEqual("atrial_flutter", af_afl["atrial_rhythm_classification"])
        self.assertEqual(0.88, af_afl["diagnostic_confidence"])
        self.assertEqual(0.10, af_afl["f_wave_confidence"])
        self.assertEqual(0.88, af_afl["F_wave_confidence"])
        self.assertTrue(af_afl["F_wave_multilead_consensus"])
        self.assertTrue(af_afl["atrial_residual_signal_summary"]["available"])

    def test_rhythm_inputs_expose_measurement_availability_reasons(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["rhythm_analysis"] = {
            "pacing_context": {"suppress_further_rhythm_interpretation": True, "continuous_pacing": True},
            "af_afl_summary": {"probable_af": True, "flutter_wave_confidence": 0.2},
            "availability": {
                "atrial_rhythm_available": False,
                "pr_available": False,
                "p_axis_available": False,
                "reasons": ["continuous_pacing", "probable_af"],
            },
        }

        payload = to_dict(features)
        availability = payload["rhythm_inputs"]["record"]["availability"]
        self.assertFalse(availability["pr_available"])
        self.assertIn("probable_af", availability["reasons"])

    def test_full_feature_json_contains_morphology_inputs_for_rule_engine(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        payload = to_dict(_make_features())

        self.assertIn("morphology_inputs", payload)
        morphology = payload["morphology_inputs"]
        self.assertEqual(
            {
                "record",
                "global",
                "measurement_profiles",
                "leads",
                "derived_facts",
                "statement_evidence",
            },
            set(morphology.keys()),
        )
        self.assertEqual(500, morphology["record"]["fs"])
        self.assertEqual(["I", "II"], morphology["record"]["lead_order"])
        self.assertEqual(60.0, morphology["global"]["heart_rate_bpm"])
        self.assertEqual(90.0, morphology["global"]["qrs_duration_ms"])
        self.assertEqual("unknown", morphology["global"]["qrs_axis_horizontal_direction"])

        lead_i = morphology["leads"]["I"]
        self.assertEqual(80.0, lead_i["p"]["duration_ms"])
        self.assertEqual(0.12, lead_i["p"]["amplitude_mV"])
        self.assertIs(lead_i["p"]["is_notched"], False)
        self.assertEqual(-0.05, lead_i["qrs"]["q_amplitude_mV"])
        self.assertEqual(1.0, lead_i["qrs"]["peak_to_peak_mV"])
        self.assertEqual("positive", lead_i["qrs"]["area_sign"])
        self.assertEqual(0.01, lead_i["st"]["j_point_mV"])
        self.assertEqual("qrs_offset", lead_i["st"]["j_point_source"])
        self.assertTrue(lead_i["st"]["j_point_reliable"])
        self.assertIsNone(lead_i["st"]["unreliable_reason"])
        self.assertEqual("positive", lead_i["t"]["polarity"])

        facts = morphology["derived_facts"]
        self.assertIn("lvh_voltage_criteria", facts)
        self.assertEqual("interpretation_not_available", morphology["statement_evidence"]["reason"])

    def test_full_feature_json_excludes_glasgow_interpretation_layer(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.representative_leads["I"].params["glasgow_measurements"] = {
            "qrs_duration_ms": 90.0,
            "r_amp_mv": 0.8,
            "qrs_area_matrix": 75.0,
            "delta_confidence_pct": 80.0,
        }
        payload = to_dict(features)

        self.assertNotIn("glasgow", payload)
        self.assertNotIn("glasgow_analysis", payload["metadata"])

    def test_export_does_not_include_glasgow_qtc_statement_guard(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.global_features.heart_rate_bpm = 130.0
        features.global_features.qrs_ms = 126.0

        payload = to_dict(features)
        self.assertNotIn("glasgow", payload)

    def test_morphology_inputs_export_native_12sl_and_hybrid_measurement_profiles(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["twelve_sl_measurement_profile"] = {
            "profile_version": "12sl_measurement_profile_v1",
            "source": "parallel_profile_no_diagnostic_override",
            "constants": {"stm_rr_fraction": 1 / 16, "ste_rr_fraction": 1 / 8},
            "heart_rate_first_last_bpm": 59.4,
            "global_fiducials": {
                "p_onset_offset_ms": -102.5,
                "qrs_onset_offset_ms": -40.0,
                "qrs_offset_offset_ms": 50.0,
                "t_offset_offset_ms": 260.0,
            },
        }
        features.representative_leads["I"].params.update({
            "st_on_mv": 0.01,
            "st_mid_mv": 0.02,
            "st_80ms_mv": 0.03,
            "qrs_area": 1.5,
            "twelve_sl_stj_mv": 0.04,
            "twelve_sl_stm_mv": -0.05,
            "twelve_sl_ste_mv": -0.06,
            "twelve_sl_qrs_area_uv_ms": 240.0,
            "twelve_sl_qrs_balance_uv": 300.0,
            "twelve_sl_qrs_deflection_uv": 1100.0,
            "twelve_sl_minimum_st_uv": -50.0,
            "twelve_sl_special_t_uv": -120.0,
            "twelve_sl_qrs_significant": True,
        })

        morphology = to_dict(features)["morphology_inputs"]
        lead_i = morphology["leads"]["I"]

        self.assertEqual("hybrid", morphology["measurement_profiles"]["active"])
        self.assertEqual(
            "12sl_measurement_profile_v1",
            morphology["measurement_profiles"]["available"]["12sl"]["profile_version"],
        )
        native_global = morphology["measurement_profiles"]["available"]["native"]["global_measurements"]
        twelve_sl_global = morphology["measurement_profiles"]["available"]["12sl"]["global_measurements"]
        hybrid_global = morphology["measurement_profiles"]["available"]["hybrid"]["global_measurements"]
        self.assertEqual(60.0, native_global["heart_rate_bpm"])
        self.assertEqual("native", native_global["heart_rate_source"])
        self.assertEqual(59.4, twelve_sl_global["heart_rate_bpm"])
        self.assertEqual(90.0, twelve_sl_global["qrs_duration_ms"])
        self.assertEqual(300.0, twelve_sl_global["qt_ms"])
        self.assertEqual(62.5, twelve_sl_global["pr_ms"])
        self.assertEqual("12sl_first_last_qrs", twelve_sl_global["heart_rate_source"])
        self.assertEqual(59.4, hybrid_global["heart_rate_bpm"])
        self.assertEqual(90.0, hybrid_global["qrs_duration_ms"])
        self.assertEqual(300.0, hybrid_global["qt_ms"])
        self.assertEqual(25.0, hybrid_global["qt_dispersion_ms"])
        self.assertEqual(40.0, hybrid_global["p_axis_frontal_deg"])
        self.assertEqual("12sl_global_fiducials", hybrid_global["interval_source"])
        self.assertEqual("native", hybrid_global["axis_source"])
        self.assertEqual(0.02, lead_i["measurement_profiles"]["native"]["st"]["midpoint_mV"])
        self.assertEqual(-0.05, lead_i["measurement_profiles"]["12sl"]["st"]["stm_mV"])
        self.assertEqual(-0.05, lead_i["measurement_profiles"]["hybrid"]["st"]["midpoint_mV"])
        self.assertEqual("12sl_profile", lead_i["measurement_profiles"]["hybrid"]["st"]["source"])
        self.assertEqual(240.0, lead_i["measurement_profiles"]["hybrid"]["qrs"]["area_uV_ms"])
        self.assertEqual(True, lead_i["measurement_profiles"]["hybrid"]["qrs"]["significant"])
        self.assertEqual(-120.0, lead_i["measurement_profiles"]["hybrid"]["t"]["special_t_uV"])

    def test_morphology_inputs_hybrid_profile_falls_back_to_native_when_12sl_missing(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        morphology = to_dict(features)["morphology_inputs"]
        lead_i = morphology["leads"]["I"]

        self.assertEqual("hybrid", morphology["measurement_profiles"]["active"])
        self.assertFalse(morphology["measurement_profiles"]["available"]["12sl"]["available"])
        self.assertEqual(
            60.0,
            morphology["measurement_profiles"]["available"]["hybrid"]["global_measurements"]["heart_rate_bpm"],
        )
        self.assertEqual(
            "native",
            morphology["measurement_profiles"]["available"]["hybrid"]["global_measurements"]["interval_source"],
        )
        self.assertEqual(0.02, lead_i["measurement_profiles"]["hybrid"]["st"]["midpoint_mV"])
        self.assertEqual("native", lead_i["measurement_profiles"]["hybrid"]["st"]["source"])
        self.assertEqual(1.5, lead_i["measurement_profiles"]["hybrid"]["qrs"]["area_native"])

    def test_morphology_inputs_exports_robust_st_side_profile(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.representative_leads["I"].params.update({
            "st_hybrid_j_mv": 0.12,
            "st_hybrid_40ms_mv": 0.11,
            "st_hybrid_80ms_mv": 0.10,
            "st_hybrid_mean_mv": 0.105,
            "st_hybrid_area_mv_ms": 6.3,
            "st_hybrid_slope_mv_per_ms": -0.00025,
            "st_hybrid_trend": "horizontal",
            "st_hybrid_shape": "concave-up",
            "st_hybrid_baseline_mv": -0.01,
            "st_hybrid_baseline_source": "post_p_pr_segment",
            "st_hybrid_baseline_confidence": 0.91,
            "st_hybrid_j_method": "local_settling_consensus",
            "st_hybrid_j_confidence": 0.88,
            "st_hybrid_consensus_support": 9,
            "st_hybrid_beat_support": 6,
            "st_hybrid_source": "pooled_reliable_beats",
            "st_hybrid_reliable": True,
            "st_hybrid_unreliable_reason": None,
        })

        robust = to_dict(features)["morphology_inputs"]["leads"]["I"]["st"]["hybrid_robust"]

        self.assertEqual(0.12, robust["j_point_mV"])
        self.assertEqual(0.10, robust["j80_mV"])
        self.assertEqual("horizontal", robust["trend"])
        self.assertEqual("concave-up", robust["shape"])
        self.assertEqual("post_p_pr_segment", robust["baseline_source"])
        self.assertEqual(9.0, robust["consensus_support"])
        self.assertTrue(robust["reliable"])

    def test_morphology_inputs_statement_evidence_is_available_when_candidates_exist(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.interpretation = types.SimpleNamespace(
            mi_statement_candidates=[
                {
                    "code": "old_mi_q_wave",
                    "category": "mi",
                    "territory": "inferior",
                    "severity": "old_or_age_indeterminate",
                    "probability": "probable",
                    "evidence": {"q_wave_leads": ["II", "III"]},
                    "suppressed_by": [],
                    "final": True,
                }
            ],
            dextrocardia_suspected=False,
        )

        statement_evidence = to_dict(features)["morphology_inputs"]["statement_evidence"]

        self.assertTrue(statement_evidence["available"])
        self.assertEqual("old_mi_q_wave", statement_evidence["final_statements"][0]["code"])

    def test_morphology_inputs_export_culprit_artery_evidence(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["lead_order"] = ["I", "II", "III", "aVF", "aVL"]
        features.interpretation = types.SimpleNamespace(
            st_elevation_leads={"III": 0.25, "II": 0.10, "aVF": 0.20},
            st_depression_leads={"I": -0.08, "aVL": -0.10},
        )

        culprit = to_dict(features)["morphology_inputs"]["derived_facts"]["culprit_artery_criteria"]

        self.assertTrue(culprit["available"])
        self.assertEqual("RCA", culprit["culprit"])
        self.assertIn("III_greater_than_II", culprit["criteria"])
        self.assertEqual(["no_V4R_V7_V8_V9"], culprit["limitations"])

    def test_morphology_inputs_export_p_fine_morphology_values(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["lead_order"] = ["I", "II", "V1"]
        features.representative_leads["V1"] = RepresentativeLeadFeatures(
            lead="V1",
            params={
                "p_notched": True,
                "p_biphasic": True,
                "p_terminal_duration_ms": 42.0,
                "p_terminal_amplitude_mV": -0.08,
                "p_terminal_area_mv_ms": -3.2,
                "reliable_for_p": True,
            },
            variance={},
        )

        p = to_dict(features)["morphology_inputs"]["leads"]["V1"]["p"]

        self.assertIs(p["is_notched"], True)
        self.assertIs(p["is_biphasic"], True)
        self.assertEqual(42.0, p["terminal_duration_ms"])
        self.assertEqual(-0.08, p["terminal_amplitude_mV"])
        self.assertEqual(-3.2, p["terminal_area_ashman"])

    def test_morphology_inputs_keep_unmeasured_p_booleans_unavailable(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["lead_order"] = ["I", "II", "V1"]
        features.representative_leads["V1"] = RepresentativeLeadFeatures(
            lead="V1",
            params={
                "p_notched": None,
                "p_biphasic": None,
                "reliable_for_p": True,
            },
            variance={},
        )

        p = to_dict(features)["morphology_inputs"]["leads"]["V1"]["p"]

        self.assertIs(p["is_notched"]["available"], False)
        self.assertIs(p["is_biphasic"]["available"], False)

    def test_morphology_inputs_export_measured_false_p_booleans(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["lead_order"] = ["I", "II", "V1"]
        v1_beat = deepcopy(features.beat_features[0])
        v1_beat.lead = "V1"
        features.beat_features.append(v1_beat)

        p = to_dict(features)["morphology_inputs"]["leads"]["V1"]["p"]

        self.assertIs(p["is_notched"], False)
        self.assertIs(p["is_biphasic"], False)

    def test_morphology_inputs_keep_missing_p_bounds_booleans_unavailable(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["lead_order"] = ["I", "II", "V1"]
        v1_beat = deepcopy(features.beat_features[0])
        v1_beat.lead = "V1"
        v1_beat.p = WaveBounds(None, None, None)
        v1_beat.p_amp_mv = None
        v1_beat.p_dur_ms = None
        v1_beat.p_area = None
        v1_beat.p_notch_interval_ms = None
        v1_beat.p_initial_duration_ms = None
        v1_beat.p_initial_amp_mv = None
        v1_beat.p_terminal_duration_ms = None
        v1_beat.p_terminal_amp_mv = None
        v1_beat.p_terminal_area_mv_ms = None
        features.beat_features.append(v1_beat)

        p = to_dict(features)["morphology_inputs"]["leads"]["V1"]["p"]

        self.assertIs(p["is_notched"]["available"], False)
        self.assertIs(p["is_biphasic"]["available"], False)

    def test_morphology_inputs_prefer_representative_p_terminal_alias_before_raw(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.metadata["lead_order"] = ["I", "II", "V1"]
        v1_beat = deepcopy(features.beat_features[0])
        v1_beat.lead = "V1"
        v1_beat.p_terminal_amp_mv = 0.22
        features.beat_features.append(v1_beat)
        features.representative_leads["V1"] = RepresentativeLeadFeatures(
            lead="V1",
            params={
                "p_terminal_amplitude_mV": -0.08,
                "reliable_for_p": True,
            },
            variance={},
        )

        p = to_dict(features)["morphology_inputs"]["leads"]["V1"]["p"]

        self.assertEqual(-0.08, p["terminal_amplitude_mV"])

    def test_morphology_inputs_mark_unreliable_st_j_point_from_qrs_tail_guard(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.beat_features[0].st_on_mv = None
        features.beat_features[0].flags.append("st_j_unreliable")

        st = to_dict(features)["morphology_inputs"]["leads"]["I"]["st"]

        self.assertIsNone(st["j_point_mV"])
        self.assertEqual("qrs_tail_guard", st["j_point_source"])
        self.assertFalse(st["j_point_reliable"])
        self.assertEqual("qrs_tail_guard", st["unreliable_reason"])

    def test_morphology_inputs_do_not_fallback_to_guarded_st_j_raw_value(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.beat_features[0].st_on_mv = 0.60
        features.beat_features[0].flags.append("st_j_unreliable")
        features.representative_leads["I"].params.update({
            "st_on_mv": None,
            "st_j_source": "qrs_tail_guard",
            "st_j_reliable": False,
            "st_j_unreliable": True,
            "st_j_unreliable_reason": "qrs_tail_guard",
        })

        st = to_dict(features)["morphology_inputs"]["leads"]["I"]["st"]

        self.assertIsNone(st["j_point_mV"])
        self.assertEqual("qrs_tail_guard", st["j_point_source"])
        self.assertFalse(st["j_point_reliable"])
        self.assertEqual("qrs_tail_guard", st["unreliable_reason"])

    def test_morphology_inputs_export_mixed_reliable_st_j_provenance(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.beat_features[0].st_on_mv = 0.60
        features.beat_features[0].flags.append("st_j_unreliable")
        features.representative_leads["I"].params.update({
            "st_on_mv": 0.02,
            "st_j_source": "mixed_reliable_beats",
            "st_j_reliable": True,
            "st_j_unreliable": False,
            "st_j_unreliable_reason": None,
        })

        st = to_dict(features)["morphology_inputs"]["leads"]["I"]["st"]

        self.assertEqual(0.02, st["j_point_mV"])
        self.assertEqual("mixed_reliable_beats", st["j_point_source"])
        self.assertTrue(st["j_point_reliable"])
        self.assertIsNone(st["unreliable_reason"])

    def test_morphology_inputs_raw_st_j_fallback_ignores_guarded_beats_without_representative_status(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.representative_leads["I"].params.clear()
        features.beat_features[0].st_on_mv = 0.60
        features.beat_features[0].flags.append("st_j_unreliable")
        reliable_beat = deepcopy(features.beat_features[0])
        reliable_beat.beat_id = 1
        reliable_beat.st_on_mv = 0.02
        reliable_beat.flags = []
        features.beat_features.append(reliable_beat)

        st = to_dict(features)["morphology_inputs"]["leads"]["I"]["st"]

        self.assertEqual(0.02, st["j_point_mV"])
        self.assertEqual("mixed_reliable_beats", st["j_point_source"])
        self.assertTrue(st["j_point_reliable"])
        self.assertIsNone(st["unreliable_reason"])

    def test_morphology_inputs_route_adult_and_pediatric_contexts(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        adult_features = _make_features()
        adult_features.metadata["patient_meta"] = PatientMeta(age=35, sex="Female")
        adult = to_dict(adult_features)["morphology_inputs"]

        self.assertEqual("adult", adult["record"]["algorithm_age_group"])
        self.assertTrue(adult["derived_facts"]["adult"]["is_adult"])
        self.assertFalse(adult["derived_facts"]["pediatric"]["is_pediatric"])

        pediatric_features = _make_features()
        pediatric_features.metadata["patient_meta"] = PatientMeta(age=7, sex="Female")
        pediatric_features.global_features.qrs_ms = 100.0
        pediatric_features.global_features.qtc_bazett_ms = 460.0
        pediatric = to_dict(pediatric_features)["morphology_inputs"]
        pediatric_context = pediatric["derived_facts"]["pediatric"]

        self.assertEqual("pediatric", pediatric["record"]["algorithm_age_group"])
        self.assertTrue(pediatric_context["is_pediatric"])
        self.assertEqual("5_7_years", pediatric_context["age_bucket"])
        self.assertEqual(-10.0, pediatric_context["axis_limits"]["lad_threshold_deg"])
        self.assertEqual(1.0, pediatric_context["axis_limits"]["borderline_lad_upper_deg"])
        self.assertEqual(201.0, pediatric_context["axis_limits"]["rad_threshold_360_deg"])
        self.assertEqual(160.0, pediatric_context["axis_limits"]["borderline_rad_threshold_360_deg"])
        self.assertEqual(88.0, pediatric_context["qrs_duration_limits"]["normal_limit_ms"])
        self.assertEqual(96.8, pediatric_context["qrs_duration_limits"]["borderline_ivcd_ms"])
        self.assertEqual(105.6, pediatric_context["qrs_duration_limits"]["nonspecific_ivcd_ms"])
        self.assertEqual(340.0, pediatric_context["qtc_limits"]["short_ms"])
        self.assertEqual(454.0, pediatric_context["qtc_limits"]["borderline_prolonged_ms"])
        self.assertEqual(474.0, pediatric_context["qtc_limits"]["prolonged_ms"])
        self.assertEqual(520.0, pediatric_context["qtc_limits"]["severe_ms"])

    def test_build_structured_payload_contains_spec_sections(self) -> None:
        from feature_extraction.ecgfeat.export import build_structured_payload

        payload = build_structured_payload(_make_features())

        self.assertEqual(
            {
                "record",
                "signal",
                "quality",
                "pacing",
                "beats",
                "groups",
                "global",
                "rhythm_inputs",
                "morphology_inputs",
                "statement_engine",
                "clinical_interpretation",
                "reference_metadata",
                "provenance",
                "schema_version",
            },
            set(payload.keys()),
        )
        self.assertEqual("ecgfeat_structured_payload.v3", payload["schema_version"])
        self.assertEqual("Q2", payload["quality"]["record_grade"])
        self.assertEqual("unknown", payload["provenance"]["pacing_state"])
        self.assertEqual("unknown", payload["provenance"]["measurement_pacing_state"])
        self.assertEqual("unknown", payload["provenance"]["pacing_detection_state"])
        self.assertEqual(
            {"candidates", "final", "suppressed", "bypassed", "unavailable", "final_statements"},
            set(payload["statement_engine"].keys()) & {
                "candidates",
                "final",
                "suppressed",
                "bypassed",
                "unavailable",
                "final_statements",
            },
        )
        self.assertNotIn("glasgow", payload)

    def test_build_structured_payload_is_snapshot_not_live_view(self) -> None:
        from feature_extraction.ecgfeat.export import build_structured_payload

        features = _make_features()
        payload = build_structured_payload(features)

        features.metadata["lead_order"].append("III")
        features.metadata["record_quality"]["reason_codes"].append("muscle_noise")

        self.assertEqual(["I", "II"], payload["signal"]["lead_order"])
        self.assertEqual(["baseline_wander"], payload["quality"]["reason_codes"])

    def test_build_batch_summary_row_surfaces_record_contract_fields(self) -> None:
        algorithm_result = _make_features()
        gt_result = _make_features()

        row = compare.build_batch_summary_row("7", algorithm_result, gt_result)

        self.assertEqual("Q2", row["record_grade"])
        self.assertEqual("unknown", row["pacing_state"])
        self.assertEqual(
            [
                "record_id",
                "algorithm_beats",
                "ground_truth_beats",
                "reference_gt_lead",
                "record_grade",
                "pacing_state",
            ],
            list(row.keys())[:6],
        )

    def test_build_batch_summary_row_uses_measurement_pacing_state_and_keeps_detection_state(self) -> None:
        algorithm_result = _make_features()
        algorithm_result.metadata["pacing_state"] = "unknown"
        algorithm_result.metadata["measurement_pacing_state"] = "unknown"
        algorithm_result.metadata["pacing_detection_state"] = "on"
        algorithm_result.metadata["pacing_measurement_effect"] = "metadata_only"
        algorithm_result.metadata["pacing_capture_confirmed"] = False
        gt_result = _make_features()

        row = compare.build_batch_summary_row("7", algorithm_result, gt_result)

        self.assertEqual("unknown", row["pacing_state"])
        self.assertEqual("on", row["pacing_detection_state"])
        self.assertEqual("metadata_only", row["pacing_measurement_effect"])
        self.assertFalse(row["pacing_capture_confirmed"])

    def test_write_summary_csv_keeps_record_contract_columns(self) -> None:
        algorithm_result = _make_features()
        gt_result = _make_features()
        row = compare.build_batch_summary_row("7", algorithm_result, gt_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = compare.Path(tmpdir) / "summary.csv"
            compare._write_summary_csv([row], out_path)
            header = out_path.read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("record_grade", header)
        self.assertIn("pacing_state", header)

    def test_snapshot_regression_tracks_pr_ms(self) -> None:
        import snapshot_regression

        baseline = [
            {
                "record": 7,
                "lead": "II",
                "beat_id": 1,
                "qrs_on": 10.0,
                "qrs_off": 90.0,
                "t_off": 310.0,
                "qt_ms": 400.0,
                "pr_ms": 150.0,
                "p_on": 0.0,
                "p_off": 20.0,
            }
        ]
        current = [
            {
                "record": 7,
                "lead": "II",
                "beat_id": 1,
                "qrs_on": 10.0,
                "qrs_off": 90.0,
                "t_off": 310.0,
                "qt_ms": 400.0,
                "pr_ms": 162.0,
                "p_on": 0.0,
                "p_off": 20.0,
            }
        ]

        self.assertIn("pr_ms", snapshot_regression.TRACKED_FIELDS)
        stats = snapshot_regression.compare_snapshots(baseline, current)

        self.assertEqual(1, stats["pr_ms"]["n"])
        self.assertEqual(12.0, stats["pr_ms"]["mean_change"])
        self.assertTrue(stats["pr_ms"]["regression"])

    def test_morphology_inputs_export_true_q_component_fields(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        bf = features.beat_features[0]
        bf.q_onset = 480
        bf.q_offset = 492
        bf.q_duration_ms = 26.0
        bf.q_area_mv_ms = 2.6
        bf.q_r_ratio = 0.25
        bf.initial_qrs_area_mv_ms = -1.8
        bf.initial_qrs_net_mv = -0.08
        features.representative_leads["I"].params.update({
            "q_duration_ms": 26.0,
            "q_area_mv_ms": 2.6,
            "q_r_ratio": 0.25,
            "initial_qrs_area_mv_ms": -1.8,
            "initial_qrs_net_mv": -0.08,
        })

        lead_i = to_dict(features)["morphology_inputs"]["leads"]["I"]["qrs"]

        self.assertEqual(26.0, lead_i["q_duration_ms"])
        self.assertEqual(2.6, lead_i["q_area_mV_ms"])
        self.assertEqual(0.25, lead_i["q_r_ratio"])
        self.assertEqual(-1.8, lead_i["initial_qrs_area_mV_ms"])
        self.assertEqual(-0.08, lead_i["initial_qrs_net_mV"])
        self.assertEqual("measured_q_component", lead_i["q_duration_source"])

    def test_morphology_inputs_export_qrs_component_durations(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        bf = features.beat_features[0]
        bf.r_duration_ms = 32.0
        bf.r_prime_duration_ms = 18.0
        bf.s_duration_ms = 24.0
        bf.s_prime_duration_ms = 12.0
        features.representative_leads["I"].params.update({
            "r_duration_ms": 32.0,
            "r_prime_duration_ms": 18.0,
            "s_duration_ms": 24.0,
            "s_prime_duration_ms": 12.0,
        })

        lead_i = to_dict(features)["morphology_inputs"]["leads"]["I"]["qrs"]

        self.assertEqual(32.0, lead_i["r_duration_ms"])
        self.assertEqual(18.0, lead_i["r_prime_duration_ms"])
        self.assertEqual(24.0, lead_i["s_duration_ms"])
        self.assertEqual(12.0, lead_i["s_prime_duration_ms"])

        lead_ii = to_dict(features)["morphology_inputs"]["leads"]["II"]["qrs"]
        self.assertFalse(lead_ii["r_duration_ms"]["available"])
        self.assertEqual("r_duration_not_measured", lead_ii["r_duration_ms"]["reason"])

    def test_morphology_inputs_export_structured_mi_evidence(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.interpretation = types.SimpleNamespace(
            pathological_q_leads={"II": True, "III": True},
            q_wave_territories=["inferior"],
            posterior_mi_suspected=False,
            mi_evidence={
                "available": True,
                "territories": {
                    "inferior": {
                        "q_wave_leads": ["II", "III"],
                        "q_wave_count": 2,
                        "q_wave_mi_pattern": True,
                    }
                },
            },
            mi_statement_candidates=[
                {
                    "code": "old_mi_q_wave",
                    "territory": "inferior",
                    "final": True,
                    "suppressed_by": [],
                    "evidence": {"q_wave_leads": ["II", "III"]},
                }
            ],
        )

        mi = to_dict(features)["morphology_inputs"]["derived_facts"]["mi_territory_criteria"]

        self.assertEqual(["inferior"], mi["q_wave_territories"])
        self.assertTrue(mi["q_wave_evidence"]["available"])
        self.assertEqual(["II", "III"], mi["territory_evidence"]["inferior"]["q_wave_leads"])
        self.assertEqual("old_mi_q_wave", mi["statement_candidates"][0]["code"])

    def test_morphology_inputs_export_pediatric_hypertrophy_evidence(self) -> None:
        from feature_extraction.ecgfeat.export import to_dict

        features = _make_features()
        features.interpretation = types.SimpleNamespace(
            pediatric_hypertrophy_evidence={
                "rvh": {
                    "criteria": ["rvh_r_v1_98p_mv"],
                    "class": "consider",
                    "bypassed_by": [],
                },
                "lvh": {"criteria": [], "class": None, "bypassed_by": []},
                "bvh": {"suspected": False, "criteria": []},
            }
        )

        facts = to_dict(features)["morphology_inputs"]["derived_facts"]

        self.assertEqual(
            ["rvh_r_v1_98p_mv"],
            facts["pediatric_hypertrophy_evidence"]["rvh"]["criteria"],
        )


if __name__ == "__main__":
    unittest.main()
