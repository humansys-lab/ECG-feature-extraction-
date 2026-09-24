"""Phase 3 contract: api.py decomposed into pipeline stages and policies.

Byte parity of the staged pipeline against the frozen corpus is proven by the golden
harness (benchmarks/GOLDEN.md); these tests pin the structure that makes that proof
meaningful: context immutability, stage order and coverage of the legacy body, policy
values equal to the legacy helpers', the interpretation-hook seam, the input-contract
forwarding module, and the rollback route.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extraction.ecgfeat import api
from feature_extraction.ecgfeat.api import ECGFeatureExtractor
from feature_extraction.ecgfeat.pipeline import context as ctx
from feature_extraction.ecgfeat.pipeline import extractor as pipeline_extractor
from feature_extraction.ecgfeat.pipeline.policies import applicability, lead_integrity, pacing, qt
from feature_extraction.ecgfeat.pipeline.stages import (
    atrial,
    beats,
    delineation,
    finalize,
    input as input_stage,
    measurement,
    quality,
    ventricular,
)

PACKAGE = Path(api.__file__).parent
STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DECISION_KINDS = {"accept", "reject", "override", "rescue", "not_applicable", "no_change"}


# --------------------------------------------------------------------------- #
# synthetic signal
# --------------------------------------------------------------------------- #

def synthetic_12_lead(fs: int = 500, seconds: float = 10.0, rr_s: float = 0.85) -> np.ndarray:
    """A deterministic sinus-like 12-lead ECG in mV with Einthoven-consistent limb leads."""
    t = np.arange(int(fs * seconds)) / fs
    rng = np.random.default_rng(20260923)

    def beat_train(p: float, q: float, r: float, s: float, tw: float) -> np.ndarray:
        x = np.zeros_like(t)
        for center in np.arange(0.4, seconds - 0.4, rr_s):
            x += p * np.exp(-0.5 * ((t - (center - 0.16)) / 0.022) ** 2)
            x += q * np.exp(-0.5 * ((t - (center - 0.022)) / 0.008) ** 2)
            x += r * np.exp(-0.5 * ((t - center) / 0.010) ** 2)
            x += s * np.exp(-0.5 * ((t - (center + 0.026)) / 0.009) ** 2)
            x += tw * np.exp(-0.5 * ((t - (center + 0.30)) / 0.045) ** 2)
        return x + 0.01 * rng.standard_normal(t.size) + 0.03 * np.sin(2 * np.pi * 0.25 * t)

    lead_i = beat_train(0.08, -0.05, 0.7, -0.10, 0.20)
    lead_ii = beat_train(0.12, -0.08, 1.1, -0.15, 0.30)
    precordial = [
        beat_train(0.06, 0.0, 0.25, -0.9, -0.10),
        beat_train(0.07, 0.0, 0.45, -1.1, 0.25),
        beat_train(0.07, -0.02, 0.75, -0.8, 0.35),
        beat_train(0.08, -0.05, 1.20, -0.5, 0.40),
        beat_train(0.08, -0.08, 1.30, -0.3, 0.35),
        beat_train(0.08, -0.08, 1.00, -0.2, 0.30),
    ]
    limb = [
        lead_i,
        lead_ii,
        lead_ii - lead_i,
        -(lead_i + lead_ii) / 2.0,
        lead_i - lead_ii / 2.0,
        lead_ii - lead_i / 2.0,
    ]
    return np.vstack(limb + precordial)


_NORMALIZED = ("/clinical_interpretation/generated_at", "/metadata/clinical_interpretation/generated_at")


def legacy_payload_bytes(features) -> dict[str, bytes]:
    """The golden harness's legacy rendering (benchmarks/golden/runner.py)."""
    from feature_extraction.ecgfeat.export import prepare_json_export, to_dict

    outputs = {}
    for profile in ("summary", "audit", "debug", "full"):
        if profile == "full":
            payload = prepare_json_export(to_dict(features), profile="debug", round_ndigits=None)
        else:
            payload = prepare_json_export(to_dict(features, profile=profile), profile=profile, round_ndigits=None)
        for pointer in _NORMALIZED:
            node = payload
            tokens = pointer.split("/")[1:]
            for token in tokens[:-1]:
                node = node.get(token) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and tokens[-1] in node:
                node[tokens[-1]] = "<normalized>"
        outputs[profile] = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    return outputs


@pytest.fixture(scope="module")
def signal() -> np.ndarray:
    return synthetic_12_lead()


@pytest.fixture(scope="module")
def staged_run(signal):
    from feature_extraction.ecgfeat.compat.interpretation_hooks import LEGACY_INTERPRETATION_HOOKS

    return pipeline_extractor.run_legacy_pipeline(
        ECGFeatureExtractor(), np.array(signal, copy=True), 500.0, hooks=LEGACY_INTERPRETATION_HOOKS,
    )


# --------------------------------------------------------------------------- #
# rollback route
# --------------------------------------------------------------------------- #

def test_staged_and_rollback_routes_produce_identical_legacy_json(signal, monkeypatch):
    monkeypatch.delenv(pipeline_extractor.LEGACY_ORCHESTRATION_ENV, raising=False)
    staged = ECGFeatureExtractor().extract(np.array(signal, copy=True), 500.0)
    monkeypatch.setenv(pipeline_extractor.LEGACY_ORCHESTRATION_ENV, "1")
    assert pipeline_extractor.legacy_orchestration_requested()
    rolled_back = ECGFeatureExtractor().extract(np.array(signal, copy=True), 500.0)
    assert len(staged.beats) >= 8
    assert legacy_payload_bytes(staged) == legacy_payload_bytes(rolled_back)


def test_rollback_route_runs_the_preserved_orchestration(signal, monkeypatch):
    from feature_extraction.ecgfeat.compat import _api_v0_legacy_orchestration as rollback

    calls = []
    original = rollback.ECGFeatureExtractor.extract

    def spy(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rollback.ECGFeatureExtractor, "extract", spy)
    monkeypatch.setattr(pipeline_extractor, "run_legacy_pipeline",
                        lambda *a, **k: pytest.fail("staged pipeline used on the rollback route"))
    monkeypatch.setenv(pipeline_extractor.LEGACY_ORCHESTRATION_ENV, "1")
    ECGFeatureExtractor().extract(np.array(signal[:, :2500], copy=True), 500.0)
    assert calls == ["ECGFeatureExtractor"]


def test_rollback_module_is_the_verbatim_pre_decomposition_api():
    source = (PACKAGE / "compat" / "_api_v0_legacy_orchestration.py").read_text()
    tree = ast.parse(source)
    helpers = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(helpers) == 49
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    extract = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "extract")
    assert extract.end_lineno - extract.lineno > 1300
    for node in ast.walk(tree):
        # Phase 5: interpretation is reached through the compat package's own
        # lazy forwarders (level 1); every other relative import is adjusted.
        if isinstance(node, ast.ImportFrom) and node.level and node.module != "_interpretation":
            assert node.level >= 2, "relative imports are adjusted for the compat package"


# --------------------------------------------------------------------------- #
# api.py holds no orchestration
# --------------------------------------------------------------------------- #

def test_api_module_contains_no_orchestration():
    # Phase 4: api.py is only the deprecated spelling; the legacy extractor
    # lives in compat.api_v0 and delegates to the single routing point.
    source = Path(api.__file__).read_text()
    tree = ast.parse(source)
    top_level = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert [n.name for n in top_level] == ["ECGFeatureExtractor"]
    assert [n.name for n in top_level[0].body if isinstance(n, ast.FunctionDef)] == ["__init__"]
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__", "typing", "warnings", "_moved", "compat.api_v0", "errors"}
    assert len(source.splitlines()) < 60
    from feature_extraction.ecgfeat.compat import api_v0

    compat_tree = ast.parse(Path(api_v0.__file__).read_text())
    cls = next(n for n in compat_tree.body if isinstance(n, ast.ClassDef) and n.name == "ECGFeatureExtractor")
    extract = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "extract")
    assert len(extract.body) <= 3
    calls = {n.func.id for n in ast.walk(extract) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert calls == {"extract_legacy_features"}


def test_extractor_constructor_and_signature_are_unchanged():
    from feature_extraction.ecgfeat.compat import _api_v0_legacy_orchestration as rollback
    from feature_extraction.ecgfeat.compat.api_v0 import ECGFeatureExtractor as CompatExtractor

    for cls in (ECGFeatureExtractor, CompatExtractor):
        assert inspect.signature(cls.__init__) == inspect.signature(rollback.ECGFeatureExtractor.__init__)
        assert inspect.signature(cls.extract) == inspect.signature(rollback.ECGFeatureExtractor.extract)
    with pytest.warns(FutureWarning):  # ECGDeprecationWarning is user-visible by design
        new = ECGFeatureExtractor(fs_internal=500, mains_freq=60, input_mode="limited")
    old = rollback.ECGFeatureExtractor(fs_internal=500, mains_freq=60, input_mode="limited")
    assert vars(new) == vars(old) == vars(CompatExtractor(fs_internal=500, mains_freq=60, input_mode="limited"))


def test_deprecated_spelling_is_the_compat_class():
    from feature_extraction.ecgfeat.compat.api_v0 import ECGFeatureExtractor as CompatExtractor

    assert issubclass(ECGFeatureExtractor, CompatExtractor)


HELPER_DESTINATIONS = {
    "pipeline.stages.input": ["_resolve_mains_frequency"],
    "pipeline.stages.beats": ["_select_measurement_group", "_beat_qrs_profiles", "_group_qrs_profile",
                              "_refine_measurement_group_after_delineation"],
    "pipeline.stages.atrial": ["_av_block_evidence"],
    "pipeline.stages.measurement": ["_backfill_t_axis_after_measurement_profile", "_build_rhythm_beat_rows",
                                    "_delta_evidence", "_pr_series_ms", "_atrial_events_per_rr"],
    "pipeline.policies.lead_integrity": ["_clear_precordial_reversal_flags", "_probable_limb_lead_reversal"],
    "pipeline.policies.applicability": ["_apply_measurement_availability_to_representatives",
                                        "_atrial_measurements_invalid_for_availability"],
    "pipeline.policies.qt": ["_copy_qt_measurements", "_apply_qt_reject_gate",
                             "_rescue_intermittent_paced_qt_from_native", "_rescue_qt_after_qrs_tail_settling"],
    "pipeline.policies.pacing": [
        "_pacing_like_av_evidence", "_dominant_group_qrs_ms", "_wide_qrs_pacing_like_context",
        "_overwide_qrs_pacing_like_context", "_overwide_pacing_qrs_override_ms",
        "_raw_pacing_qrs_underestimate_override_ms", "_near_wide_pacing_qrs_override_ms",
        "_dominant_paced_wide_qrs_override_ms", "_intermittent_paced_wide_qrs_override_ms", "_qrs_wide_values",
        "_intermittent_pacing_wide_measurement_context", "_borderline_paced_qrs_wide_offset_override_ms",
        "_secondary_paced_wide_group_qrs_override_ms", "_near_wide_paced_qrs_offset_override_ms",
        "_overwide_paced_qrs_raw_consensus_override_ms", "_paced_beat_fraction", "_selected_group_paced_majority",
        "_pacing_capture_alignment_confirmed", "_confirmed_attenuated_pacing_rescue", "_pacing_capture_beat_ids",
        "_robust_rr_profile", "_pacing_qrs_rescue_evidence", "_pacing_qrs_cleaned_single_lead_rescue",
        "_try_attenuated_pacing_rescue", "_pacing_measurement_effect", "_pacing_segmentation_effect",
        "_measurement_pacing_state", "_build_pacing_evidence_by_beat",
    ],
    "_engine.foundation.numeric": ["_finite_float", "_median_or_none"],
}


def test_all_49_helpers_live_at_their_destinations_and_not_in_api():
    from importlib import import_module

    names = [name for helpers in HELPER_DESTINATIONS.values() for name in helpers]
    assert len(names) == len(set(names)) == 49
    for module, helpers in HELPER_DESTINATIONS.items():
        target = import_module(f"feature_extraction.ecgfeat.{module}")
        for name in helpers:
            function = getattr(target, name)
            assert function.__module__ == target.__name__, (name, function.__module__)
            assert not hasattr(api, name), name


# --------------------------------------------------------------------------- #
# stages: order, coverage, contract
# --------------------------------------------------------------------------- #

STAGE_MODULES = {
    "input": input_stage, "quality": quality, "ventricular": ventricular, "beats": beats,
    "delineation": delineation, "atrial": atrial, "measurement": measurement, "finalize": finalize,
}


def test_stage_sequence_is_fixed_and_follows_the_stage_order():
    assert pipeline_extractor.LEGACY_STAGE_SEQUENCE == (
        ("input", "input", "run"),
        ("quality", "quality", "run"),
        ("ventricular", "ventricular", "run"),
        ("beats", "beats", "run"),
        ("delineation", "delineation", "run"),
        ("beats.refine_measurement_group", "beats", "refine_measurement_group"),
        ("delineation.settle_qrs_tail_and_refine_t", "delineation", "settle_qrs_tail_and_refine_t"),
        ("atrial", "atrial", "run"),
        ("measurement", "measurement", "run"),
        ("finalize", "finalize", "run"),
    )
    first_seen = list(dict.fromkeys(module for _label, module, _function in pipeline_extractor.LEGACY_STAGE_SEQUENCE))
    assert tuple(first_seen) == ctx.STAGE_ORDER


def test_stages_cover_the_legacy_extract_body_contiguously_and_in_order():
    ranges = []
    for _label, module, function in pipeline_extractor.LEGACY_STAGE_SEQUENCE:
        doc = getattr(STAGE_MODULES[module], function).__doc__
        match = re.search(r"api\.py@e7c26f7 lines (\d+)-(\d+)", doc)
        assert match, (module, function)
        ranges.append((int(match[1]), int(match[2])))
    assert ranges[0][0] == 1808 and ranges[-1][1] == 3155
    for (_first, last), (next_first, _next_last) in zip(ranges, ranges[1:]):
        # line 1838 of the legacy body is blank; every other boundary is adjacent
        assert next_first == last + 1 or (last, next_first) == (1837, 1839)


def test_every_stage_run_rejects_a_missing_upstream_context():
    for module in STAGE_MODULES.values():
        with pytest.raises(ctx.StageUnavailableError):
            module.run(None)
        with pytest.raises(NotImplementedError):
            module.run(None)
    with pytest.raises(ctx.StageUnavailableError, match="upstream state"):
        quality.run(ctx.PipelineContext(config=ctx.ExtractorSettings.from_extractor(ECGFeatureExtractor())))


def test_stop_after_runs_a_prefix_of_the_pipeline(signal):
    run = pipeline_extractor.run_legacy_pipeline(ECGFeatureExtractor(), signal, 500.0, stop_after="quality")
    assert run.context.input is not None and run.context.quality is not None
    assert run.context.ventricular is None and run.features is None
    with pytest.raises(ValueError):
        pipeline_extractor.run_legacy_pipeline(ECGFeatureExtractor(), signal, 500.0, stop_after="nope")


FORBIDDEN_FOR_STAGES = ("interpret", "clinical_rules", "glasgow_rules", "statement_engine",
                        "rhythm_statements", "compat", "export")


def test_stages_and_policies_do_not_import_interpretation_or_compatibility_code():
    for folder in ("stages", "policies"):
        for path in sorted((PACKAGE / "pipeline" / folder).glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level:
                    root = (node.module or "").split(".")[0]
                    assert root not in FORBIDDEN_FOR_STAGES, (path.name, node.module)


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #

def test_pipeline_context_is_immutable_and_replaced_per_stage(staged_run):
    context = staged_run.context
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.quality = None
    with pytest.raises(TypeError):
        context.signals["native"] = None
    with pytest.raises(TypeError):
        context.field_states["x"] = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.input.fs_run = 1
    event = ctx.PolicyEvent("test", "no_change", "test_reason")
    extended = ctx.with_policy_event(context, event)
    assert extended is not context and extended.policy_events[-1] is event
    assert event not in context.policy_events
    assert context.native.samples.flags.writeable is False
    assert context.native.lead_names == tuple(STANDARD_12)


def test_policy_events_accumulate_in_context_but_not_in_legacy_output(staged_run):
    events = staged_run.policy_events
    assert events and all(isinstance(event, ctx.PolicyEvent) for event in events)
    assert all(event.decision in DECISION_KINDS for event in events)
    assert all(re.fullmatch(r"[a-z0-9_.=,]+", event.reason_code) for event in events)
    policies = [event.policy for event in events]
    assert policies.index("pacing.capture_alignment") < policies.index("pacing.measurement_effect")
    assert policies[-1] == "qt.reject_gate"
    json.dumps([dataclasses.asdict(event) for event in events])
    import pickle

    assert pickle.loads(pickle.dumps(events)) == events
    assert copy.deepcopy(events) == events
    with_details = next(event for event in events if event.details)
    with pytest.raises(TypeError):
        with_details.details["x"] = 1
    blob = b"".join(legacy_payload_bytes(staged_run.features).values())
    for event in events:
        assert event.policy.encode() not in blob


def test_interpretation_runs_only_through_injected_hooks(signal, staged_run):
    bare = pipeline_extractor.run_legacy_pipeline(ECGFeatureExtractor(), np.array(signal, copy=True), 500.0)
    assert "clinical_interpretation" not in bare.features.metadata
    assert "statement_evidence" not in bare.features.metadata["rhythm_analysis"]["rule_summary"]
    assert bare.features.interpretation is None
    hooked = staged_run.features
    assert "clinical_interpretation" in hooked.metadata
    assert list(hooked.metadata)[-1] == "clinical_interpretation"
    assert list(hooked.metadata["rhythm_analysis"]["rule_summary"])[-1] == "statement_evidence"


def test_record_path_uses_the_staged_pipeline(monkeypatch, signal):
    from feature_extraction.ecgfeat.pipeline.extractor import ecg_measure, ecg_prepare

    seen = []
    original = pipeline_extractor.run_legacy_pipeline

    def spy(*args, **kwargs):
        seen.append(kwargs.get("hooks"))
        return original(*args, **kwargs)

    monkeypatch.delenv(pipeline_extractor.LEGACY_ORCHESTRATION_ENV, raising=False)
    monkeypatch.setattr(pipeline_extractor, "run_legacy_pipeline", spy)
    prepared = ecg_prepare(signal[:, :2500], sampling_rate=500.0, lead_names=STANDARD_12)
    measured = ecg_measure(prepared)
    # Phase 5: the record path runs without interpretation hooks, so core-only
    # installations extract records without ecginterpret.
    assert len(seen) == 1 and seen[0] is None
    assert measured.legacy_features is not None


# --------------------------------------------------------------------------- #
# input stage owns the input contract; validation.py forwards
# --------------------------------------------------------------------------- #

def test_validation_module_forwards_every_name_with_identity():
    from feature_extraction.ecgfeat import errors, validation

    for name in ("INDEPENDENT_8_LEADS", "MIN_RECORD_DURATION_SECONDS", "MIN_SAMPLING_RATE_HZ",
                 "STANDARD_12_LEADS", "ECGInputError", "_assess_amplitude_calibration",
                 "_finite_sampling_rate", "validate_ecg_input"):
        assert getattr(validation, name) is getattr(input_stage, name), name
    assert validation.ECGInputError is errors.ECGInputError
    assert validation.validate_ecg_input.__module__ == input_stage.__name__
    assert input_stage.resolve_mains_frequency(np.zeros((12, 1000)), 500, 60) == 60


# --------------------------------------------------------------------------- #
# policies: decide() returns exactly the legacy helper's value, plus an event
# --------------------------------------------------------------------------- #

def _assert_event(decision, policy_prefix):
    assert isinstance(decision, ctx.PolicyDecision)
    assert decision.event.policy.startswith(policy_prefix)
    assert decision.event.decision in DECISION_KINDS
    assert decision.event.reason_code
    json.dumps(dict(decision.event.details))


def _rep(**params):
    return SimpleNamespace(params=dict(params))


def _group(qrs, **flags):
    return SimpleNamespace(mean_qrs_ms=qrs, flags=dict(flags))


def test_pacing_rescue_policies_return_the_helper_values():
    pacing_result = {"spike_times": [100, 600, 1100, 1600, 2100, 2600, 4100, 4600], "paced": True, "state": "on"}
    pre = SimpleNamespace(r_locs=[105, 605, 1105, 1605, 2105, 2605, 3105, 3605, 4105, 4605])
    post = SimpleNamespace(r_locs=[125, 625, 2125, 3625, 4125])
    decision = pacing.PACING_QRS_RESCUE_EVIDENCE.decide(
        pacing_result=pacing_result, pre_despike_qrs_result=pre, despiked_qrs_result=post, fs=500)
    expected = pacing._pacing_qrs_rescue_evidence(
        pacing_result=pacing_result, pre_despike_qrs_result=pre, despiked_qrs_result=post, fs=500)
    assert decision.value == expected and decision.value["applied"]
    assert decision.event.decision == "accept"
    _assert_event(decision, "pacing.")

    few = SimpleNamespace(r_locs=[10, 20])
    decision = pacing.CLEANED_SINGLE_LEAD_QRS_RESCUE.decide(
        ecg_detect=np.zeros((12, 100)), fs=500, quality={}, pacing_result=pacing_result, pre_despike_qrs_result=few)
    assert decision.value == pacing._pacing_qrs_cleaned_single_lead_rescue(
        ecg_detect=np.zeros((12, 100)), fs=500, quality={}, pacing_result=pacing_result, pre_despike_qrs_result=few)
    assert decision.event.decision == "reject"

    arrays = (np.zeros((12, 10)), np.zeros((12, 10)), np.zeros((12, 10)), np.zeros((12, 10)))
    qrs_result = SimpleNamespace(r_locs=[1, 2, 3])
    decision = pacing.ATTENUATED_PACING_RESCUE.decide(
        ecg_rs=arrays[0], ecg_an=arrays[1], fs_run=500, lp_hz=40.0, current_pacing_result=pacing_result,
        current_ecg_measure=arrays[2], current_ecg_detect=arrays[3], current_qrs_result=qrs_result,
        current_r_locs=np.asarray([1, 2, 3]))
    assert decision.value[0] is pacing_result and decision.value[3] is qrs_result
    assert decision.event.decision == "no_change"


def test_pacing_state_policies_return_the_helper_values():
    cases = [
        (pacing.PACING_CAPTURE_ALIGNMENT, pacing._pacing_capture_alignment_confirmed,
         ({"paced": True}, [-10, -12, -8], 500, None), {}),
        (pacing.PACING_CAPTURE_ALIGNMENT, pacing._pacing_capture_alignment_confirmed,
         ({"paced": False}, [], 500, None), {}),
        (pacing.PACED_MEASUREMENT_GROUP, pacing._selected_group_paced_majority, ([0, 1, 2], [0, 1]), {}),
        (pacing.INTERMITTENT_PACING_WIDE_CONTEXT, pacing._intermittent_pacing_wide_measurement_context,
         ({lead: _rep(reliable_for_qrs=True, qrs_ms=100.0, qrs_wide_ms=150.0) for lead in ("I", "II", "V5")},), {}),
        (pacing.PACING_MEASUREMENT_EFFECT, pacing._pacing_measurement_effect, (), dict(
            pacing_enabled=True, pacing_result={"spike_times": [1]}, measurement_group_paced=True,
            global_measurement_paced=True, pacing_capture_confirmed=True, paced_qrs_floor_beat_ids=[1])),
        (pacing.PACING_SEGMENTATION_EFFECT, pacing._pacing_segmentation_effect, (), dict(
            pacing_enabled=True, pacing_result={"spike_times": [1]}, measurement_group_paced=False,
            segmentation_paced_beat_ids=[1], paced_qrs_floor_beat_ids=[])),
        (pacing.MEASUREMENT_PACING_STATE, pacing._measurement_pacing_state, (), dict(
            pacing_enabled=True, detection_state="unknown", confirmed_pacing_context=False,
            pacing_measurement_effect="metadata_only")),
    ]
    for policy, helper, args, kwargs in cases:
        decision = policy.decide(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        assert decision.value == helper(*args, **kwargs), policy.name
        _assert_event(decision, "pacing.")


def test_pacing_like_context_and_qrs_override_policies_match_the_legacy_chains():
    rule_summary = {"av_dissociation": True, "atrial_measurements_invalid": True, "pauses": {}}
    groups = {1: _group(100.0, dominant_group=True), 2: _group(165.0, wide_qrs=True)}
    reps = {lead: _rep(reliable_for_qrs=True, qrs_ms=q, qrs_wide_ms=w, qrs_consensus_ms=q)
            for lead, q, w in (("I", 118.0, 192.0), ("II", 120.0, 195.0), ("V5", 121.0, 200.0), ("V6", 119.0, 196.0))}
    for qrs in (100.0, 118.0, 130.0, 165.0):
        legacy_context = bool(
            pacing._wide_qrs_pacing_like_context(qrs, rule_summary)
            or pacing._overwide_qrs_pacing_like_context(qrs, groups, rule_summary))
        decision = pacing.PACING_LIKE_CONTEXT.decide(qrs, groups, rule_summary)
        assert bool(decision.value) == legacy_context
        _assert_event(decision, "pacing.")

        legacy = pacing._raw_pacing_qrs_underestimate_override_ms(qrs, reps, rule_summary)
        if legacy is None:
            legacy = pacing._near_wide_pacing_qrs_override_ms(qrs, groups, rule_summary)
        if legacy is None:
            legacy = pacing._overwide_pacing_qrs_override_ms(qrs, reps, groups, rule_summary)
        decision = pacing.PACING_LIKE_QRS_OVERRIDE.decide(qrs, reps, groups, rule_summary)
        assert decision.value == legacy
        assert decision.event.decision == ("no_change" if legacy is None else "override")

    for qrs in (104.0, 120.0, 155.0, 165.0, 180.0, 200.0):
        legacy = pacing._overwide_paced_qrs_raw_consensus_override_ms(qrs, reps, groups)
        if legacy is None:
            legacy = pacing._dominant_paced_wide_qrs_override_ms(qrs, groups)
        if legacy is None:
            legacy = pacing._near_wide_paced_qrs_offset_override_ms(qrs, reps, groups)
        if legacy is None:
            legacy = pacing._intermittent_paced_wide_qrs_override_ms(qrs, reps, groups)
        if legacy is None:
            legacy = pacing._borderline_paced_qrs_wide_offset_override_ms(qrs, reps)
        if legacy is None:
            legacy = pacing._secondary_paced_wide_group_qrs_override_ms(qrs, reps, groups)
        decision = pacing.PACED_QRS_OVERRIDE.decide(qrs, reps, groups)
        assert decision.value == legacy, qrs
        if legacy is not None:
            assert decision.event.details["override_ms"] == legacy
            assert decision.event.details["measured_ms"] == qrs


def _qt_features(**overrides):
    values = dict(qt_ms=None, qrs_ms=100.0, qtc_bazett_ms=None, qtc_fridericia_ms=None, qt_source=None,
                  qt_used_leads=[], qt_reliability="unavailable", qt_reportable=False, qt_unreliable_reasons=["x"],
                  qt_path=None, qt_confidence_reason=None, qt_rejected=False, qt_reject_reason=None,
                  qtc_framingham_ms=None, qtc_hodges_ms=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qt_policies_apply_exactly_the_legacy_helper_effect():
    reps = {lead: _rep(reliable_for_global=True, qt_consensus_ms=400.0 + i) for i, lead in enumerate(STANDARD_12[:8])}
    rescue_state = {"applied": True}
    r_locs = np.asarray([0, 400, 800, 1200])

    legacy_gf, policy_gf = _qt_features(), _qt_features()
    expected = qt._rescue_qt_after_qrs_tail_settling(
        global_features=legacy_gf, representative_leads=copy.deepcopy(reps), r_locs=r_locs, fs=500,
        qrs_tail_settling_rescue=dict(rescue_state))
    decision = qt.QT_TAIL_SETTLING_RESCUE.decide(
        global_features=policy_gf, representative_leads=reps, r_locs=r_locs, fs=500,
        qrs_tail_settling_rescue=dict(rescue_state))
    assert decision.value == expected == "qrs_tail_settling_consensus_rescue"
    assert vars(policy_gf) == vars(legacy_gf)
    assert decision.event.decision == "rescue"

    decision = qt.INTERMITTENT_PACED_QT_RESCUE.decide(_qt_features(), reps, [], r_locs, 500, 0.9)
    assert decision.value is None and decision.event.decision == "no_change"

    for reliability, grade in (("fallback", "Q3"), ("low_confidence", "Q2"), ("fallback", "Q0"), ("consensus", "Q3")):
        legacy_gf = _qt_features(qt_ms=420.0, qtc_bazett_ms=440.0, qt_reliability=reliability)
        policy_gf = copy.deepcopy(legacy_gf)
        expected = qt._apply_qt_reject_gate(legacy_gf, grade)
        decision = qt.QT_REJECT_GATE.decide(policy_gf, grade)
        assert decision.value == expected
        assert vars(policy_gf) == vars(legacy_gf)
        assert decision.event.decision == ("reject" if legacy_gf.qt_rejected else "no_change")


def test_lead_integrity_and_applicability_policies_apply_exactly_the_legacy_helper_effect():
    flagged = {lead: _rep(probable_precordial_reversal=True, precordial_reversal_detail={"suspected": True})
               for lead in ("V1", "V2", "V3")}
    expected_reps = copy.deepcopy(flagged)
    lead_integrity._clear_precordial_reversal_flags(expected_reps)
    decision = lead_integrity.PRECORDIAL_REVERSAL_FLAGS.decide(flagged)
    assert decision.value is None
    assert {k: v.params for k, v in flagged.items()} == {k: v.params for k, v in expected_reps.items()}
    assert decision.event.decision == "reject"

    for evidence in ({"probable_ra_la": True}, {}, None):
        decision = lead_integrity.LIMB_LEAD_REVERSAL.decide(evidence)
        assert decision.value == lead_integrity._probable_limb_lead_reversal(evidence)

    for rr_cv, pr in ((0.2, None), (0.05, None), (0.2, 160.0), ("x", None)):
        gf = SimpleNamespace(pr_ms=pr, p_axis_deg=None)
        decision = applicability.ATRIAL_MEASUREMENT_VALIDITY.decide(gf, {"rr_cv": rr_cv})
        assert decision.value == applicability._atrial_measurements_invalid_for_availability(gf, {"rr_cv": rr_cv})

    for availability in ({"pr_available": False, "reasons": ["probable_af"]},
                         {"pr_available": False, "reasons": ["af_afl_indeterminate"]},
                         {"pr_available": True}):
        legacy_gf, policy_gf = SimpleNamespace(pr_ms=180.0), SimpleNamespace(pr_ms=180.0)
        legacy_reps = {"II": _rep(pr_ms=180.0, pr_consensus_ms=182.0)}
        policy_reps = copy.deepcopy(legacy_reps)
        applicability._apply_measurement_availability_to_representatives(legacy_reps, legacy_gf, availability, {})
        decision = applicability.PR_AVAILABILITY.decide(policy_reps, policy_gf, availability, {})
        assert decision.value is None
        assert vars(policy_gf) == vars(legacy_gf)
        assert policy_reps["II"].params == legacy_reps["II"].params
        assert decision.event.decision == ("reject" if legacy_gf.pr_ms is None else "no_change")
