from __future__ import annotations

from ecgagent.agent.diagnostic import DIAGNOSTIC_TOOLS, ECGDiagnosticAgent
from ecgagent.backends.mock import ScriptedBackend
from ecgagent.evidence.model_view import build_model_evidence_view
from ecgagent.evidence.store import EvidenceStore
from ecgagent.tools.registry import build_default_registry
from tests.test_ecgagent_tools import _payload


def _modality_payload() -> dict:
    payload = _payload()
    payload["metadata"]["clinical_interpretation"] = dict(
        payload["clinical_interpretation"]
    )
    payload["metadata"]["patient_meta"]["clinical_classifications"] = ["AFIB"]
    payload["metadata"]["representative_group_id"] = 1
    payload["metadata"]["initial_measurement_group_id"] = 1
    payload["metadata"]["measurement_beat_ids"] = [0, 1]
    payload["metadata"]["representative_beat_meta"] = {
        "1": {
            "group_id": 1,
            "member_count": 2,
            "outlier_count": 0,
            "mean_template_corr": 0.98,
            "used_count": 1,
        }
    }
    payload["metadata"]["measurement_representative_beat_meta"] = {
        "1": {
            "group_id": 1,
            "member_count": 2,
            "outlier_count": 0,
            "mean_template_corr": 0.98,
            "used_count": 1,
        }
    }
    payload["representative_leads"]["I"]["params"]["glasgow_measurements"] = {
        "stj_uv": 10
    }
    payload["beat_features"][0]["glasgow_measurements"] = {"stj_uv": 10}
    payload["groups"] = {
        "1": {
            "group_id": 1,
            "member_count": 2,
            "member_pct": 100.0,
            "longest_run": 2,
            "mean_rr_ms": 800.0,
            "mean_pr_ms": 160.0,
            "mean_qrs_ms": 97.0,
            "mean_qt_ms": 400.0,
            "mean_ventr_rate_bpm": 75.0,
            "flags": {"dominant_group": True, "wide_qrs": False},
        }
    }
    payload["p_wave_assessments"] = [
        {
            "beat_id": 0,
            "p_state": "AF_LIKE",
            "accepted": False,
            "reject_reasons": ["BASELINE_UNOBSERVABLE"],
            "onset_confidence": 0.2,
            "offset_confidence": 0.1,
            "valid_leads": ["I"],
            "valid_lead_groups": ["limb"],
            "ta_ambiguous": True,
            "morphology_cluster_id": None,
        }
    ]
    payload["rhythm_inputs"] = {
        "record": {
            "availability": {
                "atrial_rhythm_available": False,
                "pr_available": True,
                "p_axis_available": True,
                "reasons": ["limited_atrial_support"],
            }
        },
        "background": {
            "clean_rr_ms": 800.0,
            "background_rr_regular": True,
            "background_ventricular_rate_bpm": 75.0,
            "background_atrial_rate_bpm": 75.0,
            "dominant_group_id": 1,
            "excluded_beat_ids": [],
            "exclusion_reasons": [],
        },
        "p_events": [
            {
                "p_event_id": 0,
                "time_ms": 160.0,
                "confidence": 0.9,
                "associated_qrs_beat_id": 0,
                "association_type": "conducted",
                "pr_ms": 160.0,
                "axis_deg": 50.0,
                "source_leads": ["I", "II"],
            }
        ],
        "af_afl": {
            "rr_cv": 0.02,
            "rr_rmssd": 12.0,
            "rr_entropy": 0.1,
            "probable_af": False,
            "atrial_rhythm_classification": "sinus",
            "af_afl_indeterminate": False,
            "indeterminate_reasons": [],
            "f_wave_confidence": 0.0,
            "f_wave_multilead_consensus": False,
            "F_wave_confidence": 0.0,
            "F_wave_multilead_consensus": False,
            "atrial_signal_stability": 0.9,
            "atrial_signal_repetitiveness": 0.9,
            "dominant_atrial_cycle_ms": 800.0,
        },
        "preexcitation": {
            "short_pr_interval": False,
            "short_pr_segment": False,
            "delta_lead_count": 0,
            "delta_leads": [],
            "delta_beat_ids": [],
            "delta_confidence_by_lead": {},
            "mean_qrs_duration_ms": 97.0,
            "initial_qrs_axis_deg": 18.0,
            "wpw_pattern": "none",
        },
        "av_block": {
            "second_degree_avb": None,
            "evidence": {
                "atrial_events_per_rr": [1],
                "pr_series_ms": [160.0],
                "atrial_event_excess_indices": [],
                "localized_atrial_event_excess": False,
                "constant_multiple_atrial_events_requires_validation": False,
                "dropped_p_interval_indices": [],
                "dropped_p_evidence": False,
            },
            "atrial_events_per_rr_max": 1,
            "escape_origin": None,
        },
        "aberrancy": {
            "post_pause_or_interpolated_beats": [],
            "escape_candidates": [],
            "interpolated_candidates": [],
        },
        "pacing": {
            "enabled": True,
            "state": "off",
            "measurement_state": "off",
            "detection_state": "off",
            "spike_times": [],
            "spike_count": 0,
            "paced_beat_ids": [],
            "continuous_pacing": False,
            "intermittent_pacing": False,
            "ventricular_pacing_present": False,
            "atrial_pacing_present": False,
            "dual_chamber_pacing_present": False,
            "artifact_confidence": 0.0,
            "capture_failure_suspected": False,
            "sensing_failure_suspected": {
                "available": False,
                "value": None,
                "reason": "not_implemented",
            },
        },
        "statement_evidence": {"primary_statement": "sinus_rhythm"},
    }
    payload["rhythm_inputs"]["native_beat_profiles"] = {
        "available": True,
        "source": "ecgfeat_per_beat_compact_measurements",
        "profiles": {
            "qrs_infarct": [
                "qrs_ms",
                "q_duration_ms",
                "q_amp_mv",
                "q_r_ratio",
                "r_amp_mv",
                "s_amp_mv",
                "beat_measurement_reliable",
            ],
            "repolarization": [
                "qt_ms",
                "st_hybrid_j_mv",
                "st_hybrid_60ms_mv",
                "st_hybrid_reliable",
                "t_amp_mv",
                "t_polarity",
                "beat_measurement_reliable",
            ],
        },
        "rows": [
            {
                "beat_id": beat_id,
                "group_id": 1,
                "paced": False,
                "lead": lead,
                "qrs_ms": 96.0 + beat_id,
                "q_duration_ms": 24.0 if lead == "I" else 32.0,
                "q_amp_mv": -0.04 if lead == "I" else -0.12,
                "q_r_ratio": 0.05 if lead == "I" else 0.60,
                "r_amp_mv": 0.8 if lead == "I" else 0.2,
                "s_amp_mv": -0.2,
                "initial_qrs_net_mv": 0.3,
                "qrs_confidence": 0.9,
                "qt_ms": 400.0,
                "qt_confidence": 0.8,
                "st_hybrid_j_mv": 0.01,
                "st_hybrid_60ms_mv": -0.03,
                "st_hybrid_80ms_mv": -0.02,
                "st_hybrid_slope_mv_per_ms": 0.001,
                "st_hybrid_shape": "horizontal",
                "st_hybrid_reliable": lead == "I",
                "st_hybrid_unreliable_reason": (
                    None if lead == "I" else "baseline_unstable"
                ),
                "t_amp_mv": 0.30 if lead == "I" else -0.10,
                "t_polarity": 1 if lead == "I" else -1,
                "t_symmetry": 0.5,
                "beat_measurement_reliable": True,
            }
            for beat_id in (0, 1)
            for lead in ("I", "V2")
        ],
    }
    payload["morphology_inputs"] = {
        "record": {
            "age_years": 58,
            "sex": "M",
            "clinical_dx_codes": ["AFIB"],
            "rx_codes": ["NORM"],
        },
        "global": {"heart_rate_bpm": 62.0},
        "measurement_profiles": {"active": "native"},
        "leads": {},
        "derived_facts": {"axis_flags": {"normal": True}},
        "statement_evidence": {"summary_code": "normal"},
    }
    payload["statement_engine"] = {"summary_code": "normal"}
    payload["reference_metadata"] = {"interpretation": {"reference_only": True}}
    return payload


def test_diagnostic_view_physically_redacts_nonruntime_knowledge_and_references():
    full = EvidenceStore.from_dict(_modality_payload(), record_id="SAFE")
    safe = full.diagnostic_view()

    assert safe.access_profile == "diagnostic"
    for key in (
        "clinical_interpretation",
        "interpretation",
        "statement_engine",
        "reference_metadata",
    ):
        assert key not in safe.document
    assert "clinical_interpretation" not in safe.document["metadata"]
    assert "clinical_classifications" not in safe.document["metadata"]["patient_meta"]
    assert (
        "glasgow_measurements"
        not in safe.document["representative_leads"]["I"]["params"]
    )
    assert "glasgow_measurements" not in safe.document["beat_features"][0]
    assert "statement_evidence" not in safe.document["rhythm_inputs"]
    assert "probable_af" not in safe.document["rhythm_inputs"]["af_afl"]
    assert (
        "atrial_rhythm_classification"
        not in safe.document["rhythm_inputs"]["af_afl"]
    )
    assert "wpw_pattern" not in safe.document["rhythm_inputs"]["preexcitation"]
    assert "second_degree_avb" not in safe.document["rhythm_inputs"]["av_block"]
    assert "statement_evidence" not in safe.document["morphology_inputs"]
    assert "derived_facts" not in safe.document["morphology_inputs"]
    assert "clinical_dx_codes" not in safe.document["morphology_inputs"]["record"]
    assert "p_state" not in safe.document["p_wave_assessments"][0]

    # The caller's full store remains unchanged for legacy adjudication.
    assert full.resolve("/clinical_interpretation/review_required").value is True


def test_diagnostic_agent_rebinds_a_caller_registry_to_the_safe_view():
    full = EvidenceStore.from_dict(_modality_payload(), record_id="SAFE")
    caller_registry = build_default_registry(full, budget=None)
    agent = ECGDiagnosticAgent(
        store=full,
        backend=ScriptedBackend(script=[]),
        registry=caller_registry,
    )

    assert agent.store.access_profile == "diagnostic"
    assert agent.registry.store is agent.store
    leaked = agent.registry.call(
        "get_measurement",
        {"pointer": "/clinical_interpretation/review_required"},
    )
    assert not leaked.ok
    assert "get_rhythm_profile" in agent.registry.specs
    assert "list_findings" not in agent.registry.specs


def test_modality_tools_return_only_resolvable_citations():
    store = EvidenceStore.from_dict(_modality_payload(), record_id="MOD").diagnostic_view()
    registry = build_default_registry(store, budget=None)
    calls = [
        ("get_rhythm_profile", {"sections": ["background", "av_association"]}),
        ("get_atrial_event_table", {}),
        ("get_p_assessment_table", {}),
        ("get_morphology_groups", {"include_beats": True}),
        ("get_morphology_map", {"profile": "qrs"}),
        ("get_qrs_measurement_bundle", {"leads": ["I", "V1"]}),
        ("get_interval_waveform_context", {"interval": "qt"}),
        ("get_interval_waveform_context", {"interval": "pr"}),
        ("get_pacing_profile", {"include_beats": True}),
        ("get_native_beat_profile", {"profile": "qrs_infarct"}),
        ("get_native_beat_profile", {"profile": "repolarization"}),
    ]
    for name, arguments in calls:
        result = registry.call(name, arguments)
        assert result.ok, (name, result.text)
        assert result.citations, name
        assert all(store.try_resolve(pointer) is not None for pointer in result.citations)
        assert "clinical_interpretation" not in result.text
        assert "primary_statement" not in result.text
    groups = registry.call("get_morphology_groups", {"include_beats": False})
    assert "template_outliers" in groups.text
    assert (
        "/metadata/measurement_representative_beat_meta/1/mean_template_corr"
        in groups.citations
    )


def test_rhythm_profile_labels_detector_outputs_as_candidate_evidence():
    payload = _modality_payload()
    payload["rhythm_inputs"]["preexcitation"].update(
        {
            "delta_lead_count": 6,
            "delta_leads": ["I", "II", "V2", "V3", "V4", "V5"],
            "delta_beat_ids": [0, 1],
            "short_pr_interval": False,
            "short_pr_segment": False,
        }
    )
    store = EvidenceStore.from_dict(payload, record_id="CANDIDATE").diagnostic_view()
    registry = build_default_registry(store, budget=None)

    result = registry.call(
        "get_rhythm_profile",
        {"sections": ["atrial_signal", "preexcitation"]},
    )

    assert result.ok
    rendered = result.render()
    assert "candidate-detector" in rendered
    assert "EVIDENCE TIER" in rendered
    assert "candidate-only" in rendered
    assert "without independent PR-shortening corroboration" in rendered


def test_p_assessment_marks_accepted_ambiguous_cluster_candidate_only():
    payload = _modality_payload()
    payload["p_wave_assessments"][0].update(
        {
            "accepted": True,
            "reject_reasons": [],
            "ta_ambiguous": True,
            "morphology_cluster_id": 3,
        }
    )
    store = EvidenceStore.from_dict(payload, record_id="P-AMB").diagnostic_view()
    registry = build_default_registry(store, budget=None)

    result = registry.call("get_p_assessment_table", {})

    assert result.ok
    rendered = result.render()
    assert "diagnostic_use" in rendered
    assert "candidate_only_ta_ambiguous" in rendered
    assert "does not override `ta_ambiguous`" in rendered


def test_qt_failure_context_keeps_residual_t_wave_evidence_visible():
    payload = _modality_payload()
    payload["global_features"].update(
        {
            "qt_ms": None,
            "qtc_bazett_ms": None,
            "qt_reportable": False,
            "qt_reliability": "unavailable",
        }
    )
    payload["beat_features"][0]["flags"] = ["flat_t_wave", "t_end_fallback"]
    store = EvidenceStore.from_dict(payload, record_id="QT-CONTEXT").diagnostic_view()
    registry = build_default_registry(store, budget=None)

    result = registry.call("get_interval_waveform_context", {"interval": "qt"})

    assert result.ok
    rendered = result.render()
    assert "QT/QTc measurement status" in rendered
    assert "Residual T-wave evidence by lead" in rendered
    assert "flat_t_wave" in rendered
    assert "does not mean that T-wave" in rendered
    assert "/global_features/qt_reportable" in result.citations
    assert "/representative_leads/I/params/t_amp_mv" in result.citations
    assert "/beat_features/0/flags" in result.citations

    # Dense adjudication phases can retain only two atoms from this otherwise
    # wide view.  Those two must still make the required interval context
    # possible: one interval-status atom and one residual T-wave atom.
    view = build_model_evidence_view(
        store,
        tool="get_interval_waveform_context",
        arguments={"interval": "qt"},
        rendered_text=rendered,
        citations=result.citations,
        max_atoms=2,
    )
    assert len(view.citations) == 2
    assert "/global_features/qt_reliability" in view.citations
    assert any("/params/t_" in pointer for pointer in view.citations)


def test_pr_failure_context_separates_p_visibility_from_p_qrs_association():
    payload = _modality_payload()
    payload["global_features"]["pr_ms"] = None
    payload["rhythm_inputs"]["record"]["availability"]["pr_available"] = False
    payload["representative_leads"]["I"]["params"]["p_amp_mv"] = 0.08
    store = EvidenceStore.from_dict(payload, record_id="PR-CONTEXT").diagnostic_view()
    registry = build_default_registry(store, budget=None)

    result = registry.call("get_interval_waveform_context", {"interval": "pr"})

    assert result.ok
    rendered = result.render()
    assert "PR measurement and atrial-availability status" in rendered
    assert "Residual P-wave evidence by lead" in rendered
    assert "P-wave boundary and stability evidence" in rendered
    assert "does not prove that P waves are absent" in rendered
    assert "/rhythm_inputs/record/availability/pr_available" in result.citations
    assert "/representative_leads/I/params/p_amp_mv" in result.citations

    view = build_model_evidence_view(
        store,
        tool="get_interval_waveform_context",
        arguments={"interval": "pr"},
        rendered_text=rendered,
        citations=result.citations,
        max_atoms=2,
    )
    assert len(view.citations) == 2
    assert "/global_features/pr_ms" in view.citations
    assert any("/params/p_" in pointer for pointer in view.citations)


def test_diagnostic_tool_set_contains_modalities_but_no_knowledge_or_rule_tools():
    assert {
        "get_rhythm_profile",
        "get_atrial_event_table",
        "get_p_assessment_table",
        "get_morphology_groups",
        "get_morphology_map",
        "get_qrs_measurement_bundle",
        "get_interval_waveform_context",
        "get_pacing_profile",
        "get_native_beat_profile",
    } <= set(DIAGNOSTIC_TOOLS)
    assert "list_findings" not in DIAGNOSTIC_TOOLS
    assert "get_rule_detail" not in DIAGNOSTIC_TOOLS
    assert "lookup_criteria" not in DIAGNOSTIC_TOOLS
    assert "get_signal_window" not in DIAGNOSTIC_TOOLS
    assert "remeasure" not in DIAGNOSTIC_TOOLS


def test_qrs_measurement_bundle_keeps_each_requested_lead_coherent_and_citable():
    payload = _modality_payload()
    payload["measurement_bundles"] = {
        "qrs_by_lead": {"I": {"injected_diagnosis": "RBBB"}}
    }
    store = EvidenceStore.from_dict(payload, record_id="QRS-BUNDLE").diagnostic_view()
    registry = build_default_registry(store, budget=None)

    result = registry.call(
        "get_qrs_measurement_bundle",
        {"leads": ["I", "V2"]},
    )
    view = registry.model_evidence_view(
        result,
        tool="get_qrs_measurement_bundle",
        arguments={"leads": ["I", "V2"]},
    )

    assert result.ok
    assert view is not None
    assert view.atom_count == 2
    assert view.omitted_count == 0
    assert {
        "/measurement_bundles/qrs_by_lead/I",
        "/measurement_bundles/qrs_by_lead/V2",
    } <= set(result.citations)
    assert set(view.citations) == {
        "/measurement_bundles/qrs_by_lead/I",
        "/measurement_bundles/qrs_by_lead/V2",
    }
    bundle = store.resolve("/measurement_bundles/qrs_by_lead/I")
    assert bundle.source == "ecgagent.derived_measurement_bundle"
    assert bundle.value["qrs_ms"] == payload["representative_leads"]["I"]["params"]["qrs_ms"]
    assert "injected_diagnosis" not in bundle.value


def test_native_beat_profile_uses_repeated_nonpaced_rows_with_citations():
    store = EvidenceStore.from_dict(_modality_payload(), record_id="NATIVE").diagnostic_view()
    registry = build_default_registry(store, budget=None)

    result = registry.call(
        "get_native_beat_profile",
        {"profile": "qrs_infarct", "max_beats": 3},
    )

    assert result.ok
    assert "beats=[0, 1]" in result.text
    assert "q_duration_ms" in result.text
    assert "/rhythm_inputs/native_beat_profiles/rows/0/q_amp_mv" in result.citations
    assert all(store.try_resolve(pointer) is not None for pointer in result.citations)


def test_legacy_pacing_profile_surfaces_evidence_conflict_without_hiding_spikes():
    payload = _modality_payload()
    pacing = payload["rhythm_inputs"]["pacing"]
    pacing.update(
        {
            "state": "on",
            "measurement_state": "on",
            "detection_state": "on",
            "spike_times": list(range(73)),
            "spike_count": 73,
            "paced_beat_ids": [0],
            "artifact_confidence": 0.19,
        }
    )
    store = EvidenceStore.from_dict(payload, record_id="LEGACY").diagnostic_view()
    registry = build_default_registry(store, budget=None)

    result = registry.call("get_pacing_profile", {"include_beats": True})

    assert result.ok
    assert "PACING EVIDENCE CONFLICTED" in result.text
    assert "Candidate spikes remain informative" in result.text
    assert "/rhythm_inputs/pacing/spike_times" in result.citations
