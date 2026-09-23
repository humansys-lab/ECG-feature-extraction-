"""Regression contracts for program facts: missing evidence never adds support."""
from copy import deepcopy
import math

import pytest

from ecgagent.agent.deterministic_pathways import (
    DEFAULT_DETERMINISTIC_PATHWAY_POLICY,
    _qualified_atrial_events,
    resolve_additional_deterministic_step,
)
from ecgagent.evidence.store import EvidenceStore, STANDARD_LEAD_ORDER


def pointers(value, prefix=""):
    result = {prefix} if prefix else set()
    if isinstance(value, dict):
        for key, child in value.items():
            result |= pointers(child, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result |= pointers(child, f"{prefix}/{index}")
    return result


def resolve(payload, step="one_to_one_av", code="first_degree_av_block", available=None):
    fact = resolve_additional_deterministic_step(
        EvidenceStore.from_dict(payload), code=code, step_id=step,
        available_pointers=pointers(payload) if available is None else available,
    )
    assert fact is not None
    return fact


def atrial_payload(count=6):
    return {"rhythm_inputs": {
        "p_events": [
            {"p_event_id": i, "time_ms": 800.0 + i * 1000,
             "confidence": .95, "source_leads": ["II", "V1"],
             "associated_qrs_beat_id": i, "association_type": "conducted", "pr_ms": 214.0}
            for i in range(count)
        ],
        "beats": [{"beat_id": i, "r_time_ms": 1000.0 + i * 1000} for i in range(count)],
    }}


def events(payload):
    return payload["rhythm_inputs"]["p_events"]


BAD_CONFIDENCE = [None, False, True, "0.95", [], {}, math.nan, math.inf, -math.inf, -.1, 1.01, .79]
BAD_LEADS = [None, "II,V1", [], ["II"], ["II", "II"], ["bogus", "II"],
             ["II", None], ["II", 1], ["II", {}], ["II", "V1", "junk"]]


@pytest.mark.parametrize("field,bad", [("confidence", x) for x in BAD_CONFIDENCE] + [("source_leads", x) for x in BAD_LEADS])
def test_atrial_quality_fail_closed(field, bad):
    payload = atrial_payload()
    for event in events(payload):
        event[field] = bad
    assert _qualified_atrial_events(list(enumerate(events(payload))), DEFAULT_DETERMINISTIC_PATHWAY_POLICY) == []
    assert resolve(payload).status == "unknown"


@pytest.mark.parametrize("field", ["confidence", "source_leads", "p_event_id", "time_ms", "associated_qrs_beat_id", "pr_ms"])
def test_removing_required_event_value_or_pointer_cannot_pass(field):
    payload = atrial_payload()
    assert resolve(payload).status == "pass"
    original = deepcopy(payload)
    for event in events(payload):
        event.pop(field)
    assert resolve(payload).status == "unknown"
    visible = pointers(original) - {f"/rhythm_inputs/p_events/{i}/{field}" for i in range(6)}
    fact = resolve(original, available=visible)
    assert fact.status == "unknown"
    assert f"/rhythm_inputs/p_events/0/{field}" in fact.input_pointers


@pytest.mark.parametrize("field,bad", [
    ("p_event_id", True), ("p_event_id", 0.0), ("p_event_id", "0"),
    ("associated_qrs_beat_id", []), ("associated_qrs_beat_id", {}),
    ("associated_qrs_beat_id", True), ("associated_qrs_beat_id", 999),
    ("time_ms", -1), ("time_ms", math.nan), ("time_ms", True), ("time_ms", 1100),
    ("pr_ms", 0), ("pr_ms", -1), ("pr_ms", math.inf),
    ("association_type", "unknown"), ("association_type", {}),
])
def test_invalid_event_identity_reference_time_and_interval(field, bad):
    payload = atrial_payload()
    events(payload)[0][field] = bad
    assert resolve(payload).status == "unknown"


@pytest.mark.parametrize("field", ["p_event_id", "time_ms", "associated_qrs_beat_id"])
def test_duplicate_event_or_qrs_identity_is_not_one_to_one(field):
    payload = atrial_payload()
    events(payload)[1][field] = events(payload)[0][field]
    assert resolve(payload).status == "unknown"


@pytest.mark.parametrize("mutation", ["missing", "duplicate_id", "duplicate_time", "invalid_time", "unmatched", "reversed"])
def test_qrs_timeline_integrity(mutation):
    payload = atrial_payload()
    beats = payload["rhythm_inputs"]["beats"]
    if mutation == "missing":
        payload["rhythm_inputs"].pop("beats")
    elif mutation == "duplicate_id":
        beats[1]["beat_id"] = 0
    elif mutation == "duplicate_time":
        beats[1]["r_time_ms"] = beats[0]["r_time_ms"]
    elif mutation == "invalid_time":
        beats[1]["r_time_ms"] = math.inf
    elif mutation == "unmatched":
        beats.insert(1, {"beat_id": 100, "r_time_ms": 1500.0})
    else:
        events(payload).reverse()
    assert resolve(payload).status == "unknown"


def blocked_payload():
    payload = atrial_payload()
    for i in (1, 3):
        events(payload)[i].update(association_type="blocked", associated_qrs_beat_id=None, pr_ms=None)
    return payload


def test_blocked_events_contradict_one_to_one_and_remain_unknown_when_quality_removed():
    payload = blocked_payload()
    assert resolve(payload).status == "fail"
    assert resolve(payload, "repeated_blocked_atrial_events", "second_degree_av_block").status == "pass"
    for i in (1, 3):
        events(payload)[i].pop("confidence")
    assert resolve(payload).status == "unknown"
    assert resolve(payload, "repeated_blocked_atrial_events", "second_degree_av_block").status == "unknown"


@pytest.mark.parametrize("field", ["associated_qrs_beat_id", "pr_ms"])
def test_blocked_requires_explicit_null_association(field):
    payload = blocked_payload()
    for i in (1, 3):
        events(payload)[i].pop(field)
    assert resolve(payload, "repeated_blocked_atrial_events", "second_degree_av_block").status == "unknown"


def test_every_atrial_quality_identity_and_qrs_dependency_is_audited():
    fact = resolve(atrial_payload())
    assert fact.status == "pass"
    for i in range(6):
        for field in events(atrial_payload())[i]:
            assert f"/rhythm_inputs/p_events/{i}/{field}" in fact.input_pointers
        for field in ("beat_id", "r_time_ms"):
            assert f"/rhythm_inputs/beats/{i}/{field}" in fact.input_pointers
    assert len(fact.input_pointers) == len(set(fact.input_pointers))


def p_payload(accepted=True):
    return {"p_wave_assessments": [
        {"accepted": accepted, "ta_ambiguous": False, "onset_confidence": .9 if accepted else .1,
         "offset_confidence": .9 if accepted else .1, "valid_leads": ["II", "V1"], "reject_reasons": []}
        for _ in range(6)
    ]}


@pytest.mark.parametrize("field,bad", [
    ("accepted", None), ("accepted", "true"), ("ta_ambiguous", None),
    ("ta_ambiguous", "false"), ("onset_confidence", None), ("offset_confidence", math.nan),
    ("onset_confidence", 1.1), ("valid_leads", ["II", "II"]),
    ("valid_leads", ["x", "y"]), ("reject_reasons", None),
])
def test_p_quality_does_not_accept_missing_or_invalid_fields(field, bad):
    payload = p_payload()
    assert resolve(payload, "p_measurement_reliability").status == "pass"
    for row in payload["p_wave_assessments"]:
        row[field] = bad
    assert resolve(payload, "p_measurement_reliability").status == "unknown"


def test_p_absence_requires_all_rows_analyzable_and_does_not_self_confirm_from_detector_rejection():
    payload = p_payload(False)
    assert resolve(payload, "organized_p_absent", "atrial_fibrillation").status == "pass"
    row = payload["p_wave_assessments"][0]
    row.update(onset_confidence=.9, offset_confidence=.9, reject_reasons=["RHYTHM_AF_LIKE"])
    assert resolve(payload, "organized_p_absent", "atrial_fibrillation").status == "unknown"
    row.pop("onset_confidence")
    assert resolve(payload, "organized_p_absent", "atrial_fibrillation").status == "unknown"


@pytest.mark.parametrize("field", ["accepted", "ta_ambiguous", "valid_leads", "onset_confidence", "offset_confidence", "reject_reasons"])
def test_missing_p_absence_quality_never_becomes_absence(field):
    payload = p_payload(False)
    for row in payload["p_wave_assessments"]:
        row.pop(field)
    assert resolve(payload, "organized_p_absent", "atrial_fibrillation").status == "unknown"


def sequence_payload():
    return {"rhythm_inputs": {"av_block": {"evidence": {
        "atrial_events_per_rr": [1, 2, 1, 2, 1], "pr_series_ms": [210.] * 5,
        "dropped_p_evidence": True, "dropped_p_interval_indices": [1, 3],
        "constant_multiple_atrial_events_requires_validation": False,
    }}}}


@pytest.mark.parametrize("field,bad", [
    ("atrial_events_per_rr", [1, 2, math.nan, 2, 1]),
    ("atrial_events_per_rr", [1, 2, math.inf, 2, 1]),
    ("atrial_events_per_rr", [1, 2, True, 2, 1]),
    ("atrial_events_per_rr", [1, 2, 1.5, 2, 1]),
    ("pr_series_ms", [True] * 5), ("pr_series_ms", [math.nan] * 5),
    ("pr_series_ms", [0] * 5), ("dropped_p_interval_indices", [1, 1]),
    ("dropped_p_interval_indices", [-1, 3]), ("dropped_p_interval_indices", [1, 8]),
    ("dropped_p_interval_indices", [True, 3]), ("dropped_p_interval_indices", [1, 2]),
    ("dropped_p_evidence", "true"), ("constant_multiple_atrial_events_requires_validation", None),
])
def test_av_sequence_rejects_invalid_types_counts_and_internal_contradictions(field, bad):
    payload = sequence_payload()
    assert resolve(payload, "sequential_av_pattern", "second_degree_av_block").status == "pass"
    payload["rhythm_inputs"]["av_block"]["evidence"][field] = bad
    assert resolve(payload, "sequential_av_pattern", "second_degree_av_block").status == "unknown"


def test_complete_av_block_not_refuted_by_two_pr_values():
    payload = sequence_payload()
    payload["rhythm_inputs"]["av_block"]["evidence"].update(
        atrial_events_per_rr=[1] * 5, pr_series_ms=[210., 210.],
        dropped_p_evidence=False, dropped_p_interval_indices=[],
    )
    assert resolve(payload, "sequential_av_pattern", "complete_av_block").status == "unknown"


@pytest.mark.parametrize("grade", [None, "", "garbage", "fallback", "low_confidence", "unavailable", False, {}])
@pytest.mark.parametrize("step", ["interval_reportable", "qt_threshold", "component_endpoint_support"])
def test_qt_requires_positive_reliability_contract(grade, step):
    payload = {"global_features": {"qt_reportable": True, "qt_reliability": grade, "qtc_fridericia_ms": 510.}}
    assert resolve(payload, step, "markedly_prolonged_qt").status == "unknown"


@pytest.mark.parametrize("value,expected", [(449., "fail"), (450., "unknown"), (479.9, "unknown"), (480., "pass"), (0, "unknown"), (-1, "unknown"), (True, "unknown"), (math.inf, "unknown")])
def test_qtc_threshold_and_borderline_contract(value, expected):
    payload = {"global_features": {"qt_reportable": True, "qt_reliability": "reliable", "qtc_fridericia_ms": value}}
    assert resolve(payload, "qt_threshold", "prolonged_qt").status == expected


def test_qt_failure_audits_reportability_and_reliability_inputs():
    payload = {"global_features": {"qt_reportable": False, "qt_reliability": "reliable", "qtc_fridericia_ms": 510.}}
    fact = resolve(payload, "qt_threshold", "markedly_prolonged_qt")
    assert fact.status == "unknown"
    assert {"/global_features/qt_reportable", "/global_features/qt_reliability"} <= set(fact.input_pointers)


@pytest.mark.parametrize("availability", [None, False, 1, "true"])
def test_pr_requires_explicit_reportability(availability):
    payload = {"global_features": {"pr_ms": 214.}, "rhythm_inputs": {"record": {"availability": {"pr_available": availability}}}}
    assert resolve(payload, "pr_criterion").status == "unknown"


@pytest.mark.parametrize("value", [0, -1, True, math.nan, math.inf])
def test_pr_invalid_numbers_are_not_definition_level_contradictions(value):
    payload = {"global_features": {"pr_ms": value}, "rhythm_inputs": {"record": {"availability": {"pr_available": True}}}}
    assert resolve(payload, "pr_criterion").status == "unknown"


def group_payload():
    return {"groups": {"1": {"member_count": 6, "member_pct": 100., "longest_run": 6,
                             "mean_qrs_ms": 130., "mean_ventr_rate_bpm": 120., "flags": {"dominant_group": True}}}}


@pytest.mark.parametrize("field,bad", [("member_count", -1), ("member_count", 3.5), ("member_count", True), ("member_pct", 101.), ("member_pct", -1.), ("longest_run", 7), ("longest_run", 1.5)])
def test_representative_counts_are_validated(field, bad):
    payload = group_payload()
    assert resolve(payload, "pattern_representative").status == "pass"
    payload["groups"]["1"][field] = bad
    assert resolve(payload, "pattern_representative").status == "unknown"


def test_multiple_dominant_groups_are_unknown_and_selector_is_audited():
    payload = group_payload()
    fact = resolve(payload, "qrs_duration_support", "right_bundle_branch_block")
    assert fact.status == "pass"
    assert "/groups/1/flags/dominant_group" in fact.input_pointers
    payload["groups"]["2"] = deepcopy(payload["groups"]["1"])
    assert resolve(payload, "qrs_duration_support", "right_bundle_branch_block").status == "unknown"


@pytest.mark.parametrize("qrs,expected", [(0, "unknown"), (-1, "unknown"), (109., "fail"), (110., "unknown"), (119.9, "unknown"), (120., "pass")])
def test_bundle_qrs_duration_retains_borderline_zone(qrs, expected):
    payload = group_payload()
    payload["groups"]["1"]["mean_qrs_ms"] = qrs
    assert resolve(payload, "qrs_duration_support", "right_bundle_branch_block").status == expected


def test_zero_member_wide_group_is_not_an_isolated_pvc():
    payload = group_payload()
    payload["groups"]["1"].update(mean_qrs_ms=180., member_count=0, member_pct=10., flags={"dominant_group": False})
    assert resolve(payload, "ectopic_morphology", "premature_ventricular_complexes").status == "unknown"


def pacing_payload():
    return {"rhythm_inputs": {"pacing": {"state": "on", "spike_count": 6,
        "qrs_associated_spike_fraction": .9, "capture_alignment_fraction": .9,
        "evidence_conflicted": False, "supports_measurement_routing": True,
        "capture_failure_suspected": False}}}


@pytest.mark.parametrize("field,bad", [("spike_count", -1), ("spike_count", 3.5), ("spike_count", True), ("capture_alignment_fraction", 1.1), ("qrs_associated_spike_fraction", 1.1), ("evidence_conflicted", None), ("evidence_conflicted", True), ("state", "off"), ("capture_failure_suspected", True)])
def test_pacing_rejects_invalid_and_conflicting_support(field, bad):
    payload = pacing_payload()
    assert resolve(payload, "capture_relation_support", "paced_rhythm").status == "pass"
    payload["rhythm_inputs"]["pacing"][field] = bad
    assert resolve(payload, "capture_relation_support", "paced_rhythm").status == "unknown"


@pytest.mark.parametrize("value", [None, "false", 0, []])
def test_sensing_failure_missing_boolean_does_not_become_negative(value):
    payload = pacing_payload()
    payload["rhythm_inputs"]["pacing"]["sensing_failure_suspected"] = {"available": True, "value": value}
    assert resolve(payload, "capture_relation_support", "pacing_sensing_failure_suspected").status == "unknown"


def voltage_payload():
    return {"representative_leads": {lead: {"params": {
        "r_amp_mv": .2, "s_amp_mv": -.1, "q_amp_mv": -.05, "reliable_for_qrs": True,
    }} for lead in STANDARD_LEAD_ORDER}}


@pytest.mark.parametrize("field,bad", [("reliable_for_qrs", None), ("reliable_for_qrs", False), ("r_amp_mv", -.1), ("s_amp_mv", .1), ("q_amp_mv", .1), ("q_amp_mv", None)])
def test_low_voltage_requires_all_components_and_quality(field, bad):
    payload = voltage_payload()
    assert resolve(payload, "territorial_qrs_voltage", "low_qrs_voltage_limb_leads").status == "pass"
    payload["representative_leads"]["I"]["params"][field] = bad
    assert resolve(payload, "territorial_qrs_voltage", "low_qrs_voltage_limb_leads").status == "unknown"


def test_q_wave_is_included_in_low_voltage_peak_to_peak_and_audit():
    payload = voltage_payload()
    payload["representative_leads"]["II"]["params"]["q_amp_mv"] = -.6
    fact = resolve(payload, "territorial_qrs_voltage", "low_qrs_voltage_limb_leads")
    assert fact.status == "fail"
    assert "/representative_leads/II/params/q_amp_mv" in fact.input_pointers


def test_zero_over_zero_does_not_fabricate_early_transition():
    payload = voltage_payload()
    for lead in ("V1", "V2", "V3", "V4", "V5", "V6"):
        payload["representative_leads"][lead]["params"].update(r_amp_mv=0., s_amp_mv=0.)
    assert resolve(payload, "precordial_progression", "counterclockwise_rotation").status == "unknown"


def test_rvh_positive_s_amplitude_is_not_s_dominance():
    payload = voltage_payload()
    for lead in ("V5", "V6"):
        payload["representative_leads"][lead]["params"]["s_amp_mv"] = .8
    assert resolve(payload, "right_precordial_voltage", "right_ventricular_hypertrophy").status == "unknown"


@pytest.mark.parametrize("bad", [None, "false", 0, {}])
def test_reconciliation_missing_flag_is_not_reconciled(bad):
    payload = {"rhythm_inputs": {"av_block": {"evidence": {
        "localized_atrial_event_excess": True,
        "constant_multiple_atrial_events_requires_validation": bad,
    }}}}
    assert resolve(payload, "sinus_candidate_stream_reconciled").status == "unknown"


@pytest.mark.parametrize("rate", [0, -1, True, math.inf])
def test_invalid_matching_rates_do_not_prove_sinus_origin(rate):
    payload = {"global_features": {"heart_rate_bpm": rate, "atrial_rate_bpm": rate, "p_axis_deg": 45.}}
    assert resolve(payload, "sinus_mechanism_support").status == "unknown"


@pytest.mark.parametrize("confidence", [-1, 1.1, math.nan, True])
def test_af_support_fraction_must_be_a_probability(confidence):
    payload = {"rhythm_inputs": {"af_afl": {"f_wave_multilead_consensus": True, "f_wave_confidence": confidence}}}
    assert resolve(payload, "multilead_atrial_support", "atrial_fibrillation").status == "unknown"


@pytest.mark.parametrize("count,expected", [(0, "unknown"), (4, "unknown"), (5, "pass")])
def test_one_to_one_requires_minimum_unique_events(count, expected):
    assert resolve(atrial_payload(count)).status == expected


@pytest.mark.parametrize("confidence,expected", [(.7999, "unknown"), (.8, "pass"), (1., "pass")])
def test_atrial_confidence_threshold_includes_only_valid_probability(confidence, expected):
    payload = atrial_payload()
    for event in events(payload):
        event["confidence"] = confidence
    assert resolve(payload).status == expected


def test_top_level_beats_must_match_referenced_rhythm_timeline():
    payload = atrial_payload()
    payload["beats"] = [{"beat_id": i} for i in range(6)]
    assert resolve(payload).status == "pass"
    payload["beats"][2]["beat_id"] = 100
    fact = resolve(payload)
    assert fact.status == "unknown"
    assert "/beats/2/beat_id" in fact.input_pointers


@pytest.mark.parametrize("metadata", [{"n_beats": 7}, {"n_beats": True}, {"duration_sec": 2}, {"duration_sec": math.nan}])
def test_record_time_and_count_contradictions_are_unknown(metadata):
    payload = atrial_payload()
    payload["metadata"] = metadata
    assert resolve(payload).status == "unknown"


def rr_payload():
    return {"beats": [{"rr_prev_ms": rr, "rr_next_ms": 1000.}
                      for rr in (1000., 1000., 1000., 600., 1200., 1000.)]}


def test_prematurity_is_positive_but_cannot_be_inferred_from_negative_rr():
    payload = rr_payload()
    assert resolve(payload, "premature_timing", "premature_ventricular_complexes").status == "pass"
    payload["beats"][3]["rr_prev_ms"] = -1
    assert resolve(payload, "premature_timing", "premature_ventricular_complexes").status == "unknown"


def test_hidden_or_invalid_rr_cannot_make_background_regular():
    payload = {"beats": [{"rr_prev_ms": rr} for rr in (1000., 1000., 1000., 1000., 1000., 3000., 4000.)],
               "rhythm_inputs": {"background": {"background_rr_regular": False},
                                 "record": {"availability": {"atrial_rhythm_available": True}}}}
    assert resolve(payload, "organized_atrial_activity").status == "unknown"
    visible = pointers(payload) - {"/beats/5/rr_prev_ms", "/beats/6/rr_prev_ms"}
    assert resolve(payload, "organized_atrial_activity", available=visible).status == "unknown"
    payload["beats"][5]["rr_prev_ms"] = None
    payload["beats"][6]["rr_prev_ms"] = None
    assert resolve(payload, "organized_atrial_activity").status == "unknown"


def test_wide_sequence_requires_real_run_counts_and_tachycardia_rate():
    payload = group_payload()
    assert resolve(payload, "wide_complex_sequence", "wide_complex_tachycardia").status == "pass"
    payload["groups"]["1"].pop("mean_ventr_rate_bpm")
    assert resolve(payload, "wide_complex_sequence", "wide_complex_tachycardia").status == "unknown"
    payload["groups"]["1"]["longest_run"] = 7
    assert resolve(payload, "wide_complex_sequence", "ventricular_rhythm").status == "unknown"


def test_pacing_state_does_not_identify_ventricular_capture():
    payload = {"rhythm_inputs": {"pacing": {"state": "on"}}}
    assert resolve(payload, "voltage_criteria_pacing_valid", "lvh_voltage_criteria").status == "unknown"
    payload["rhythm_inputs"]["pacing"]["ventricular_pacing_present"] = True
    assert resolve(payload, "voltage_criteria_pacing_valid", "lvh_voltage_criteria").status == "fail"


@pytest.mark.parametrize("bad", [None, True, 1, "bad", {}])
@pytest.mark.parametrize("step,root", [
    ("one_to_one_av", "rhythm_inputs"), ("p_measurement_reliability", "p_wave_assessments"),
    ("premature_timing", "beats"), ("pattern_representative", "groups"),
    ("ectopic_morphology", "groups"), ("wide_complex_sequence", "groups"),
])
def test_malformed_containers_return_unknown_without_crashing(step, root, bad):
    assert resolve({root: bad}, step, "premature_ventricular_complexes").status == "unknown"


def test_unknown_fact_never_cites_nonexistent_fields():
    fact = resolve({"global_features": {"qt_reportable": True}}, "interval_reportable", "prolonged_qt")
    assert fact.status == "unknown"
    assert [pointer for pointer, _ in fact.evidence] == ["/global_features/qt_reportable"]
    assert "/global_features/qt_reliability" in fact.input_pointers


def test_deleting_conflicting_capture_measurement_does_not_confirm_capture_failure():
    payload = pacing_payload()
    pacing = payload["rhythm_inputs"]["pacing"]
    pacing["capture_failure_suspected"] = True
    assert resolve(payload, "capture_relation_support", "pacing_failure_to_capture_suspected").status == "unknown"
    pacing.pop("capture_alignment_fraction")
    assert resolve(payload, "capture_relation_support", "pacing_failure_to_capture_suspected").status == "unknown"
    pacing["capture_alignment_fraction"] = .1
    assert resolve(payload, "capture_relation_support", "pacing_failure_to_capture_suspected").status == "pass"


def test_deleting_conflicting_dominant_flag_does_not_choose_a_group():
    payload = group_payload()
    payload["groups"]["2"] = deepcopy(payload["groups"]["1"])
    assert resolve(payload, "qrs_duration_support", "right_bundle_branch_block").status == "unknown"
    payload["groups"]["2"].pop("flags")
    assert resolve(payload, "qrs_duration_support", "right_bundle_branch_block").status == "unknown"


def test_hidden_wide_group_does_not_refute_wide_sequence():
    payload = group_payload()
    payload["groups"]["2"] = deepcopy(payload["groups"]["1"])
    payload["groups"]["1"]["mean_qrs_ms"] = 90.
    assert resolve(payload, "wide_complex_sequence", "wide_complex_tachycardia").status == "pass"
    visible = pointers(payload) - {"/groups/2/mean_qrs_ms"}
    assert resolve(payload, "wide_complex_sequence", "wide_complex_tachycardia", visible).status == "unknown"


@pytest.mark.parametrize("code,value,expected", [
    ("markedly_prolonged_qt", 499., "unknown"), ("markedly_prolonged_qt", 500., "pass"),
    ("short_qt", 340., "pass"), ("short_qt", 341., "unknown"), ("short_qt", 360., "fail"),
    ("markedly_short_qt", 330., "pass"), ("markedly_short_qt", 335., "unknown"),
    ("markedly_short_qt", 345., "fail"), ("borderline_short_qt", 350., "pass"),
    ("borderline_short_qt", 365., "unknown"), ("borderline_short_qt", 370., "fail"),
    ("possible_short_qt_pattern", 355., "pass"), ("possible_short_qt_pattern", 365., "unknown"),
    ("possible_short_qt_pattern", 370., "fail"),
])
def test_disease_specific_qtc_boundaries(code, value, expected):
    payload = {"global_features": {"qt_reportable": True, "qt_reliability": "rescued", "qtc_fridericia_ms": value}}
    assert resolve(payload, "qt_threshold", code).status == expected


@pytest.mark.parametrize("step,value,expected", [
    ("pr_criterion", 200., "fail"), ("pr_criterion", 201., "pass"),
    ("short_pr_criterion", 119., "pass"), ("short_pr_criterion", 120., "fail"),
    ("pr_component", 180., "pass"),
])
def test_disease_specific_pr_boundaries(step, value, expected):
    payload = {"global_features": {"pr_ms": value}, "rhythm_inputs": {"record": {"availability": {"pr_available": True}}}}
    assert resolve(payload, step).status == expected
