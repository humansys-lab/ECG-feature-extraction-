from __future__ import annotations

import numpy as np

from feature_extraction.ecgfeat import acquisition_qc
from feature_extraction.ecgfeat.acquisition_qc import (
    apply_channel_delay_compensation,
    assess_acquisition_chain,
)
from feature_extraction.ecgfeat.models import PWaveLeadBoundary
from feature_extraction.ecgfeat.p_wave_engine import (
    AF_LIKE,
    BASELINE_UNOBSERVABLE,
    INSUFFICIENT_INFORMATIVE_LEADS,
    P_ON_T_UNRESOLVED,
    P_PRESENT,
    PWaveConfig,
    _fuse_beat,
    _reconcile_parallel_consensus_models,
    finalize_p_wave_states,
)


def _boundary(
    lead: str,
    onset: int,
    offset: int,
    *,
    baseline_mode: str = "tp_observed_state_space",
    on_t_overlap: bool = False,
    t_reconstruction_validated: bool = False,
) -> PWaveLeadBoundary:
    return PWaveLeadBoundary(
        lead=lead,
        onset=onset,
        peak=(onset + offset) // 2,
        offset=offset,
        onset_ci_low=onset - 2,
        onset_ci_high=onset + 2,
        offset_ci_low=offset - 2,
        offset_ci_high=offset + 2,
        onset_confidence=0.90,
        offset_confidence=0.88,
        onset_sigma_ms=4.0,
        offset_sigma_ms=5.0,
        quality_score=0.90,
        local_snr=6.0,
        informative=True,
        hard_quality_pass=True,
        baseline_mode=baseline_mode,
        baseline_confidence=0.95,
        on_t_overlap=on_t_overlap,
        t_reconstruction_validated=t_reconstruction_validated,
    )


def _fuse(rows: list[PWaveLeadBoundary]):
    return _fuse_beat(
        beat_id=0,
        per_lead={row.lead: row for row in rows},
        fs=500,
        coarse_start=80,
        coarse_stop=160,
        svd_dimension=1,
        acquisition_qc={"reject_reasons": []},
        t_reconstruction_available=False,
        config=PWaveConfig(),
    )


def test_fusion_requires_independent_lead_groups_and_exports_strict_robust() -> None:
    assessment = _fuse(
        [
            _boundary("I", 100, 140),
            _boundary("II", 101, 141),
            _boundary("V2", 102, 142),
        ]
    )

    assert assessment.accepted
    assert assessment.p_state == P_PRESENT
    assert assessment.strict_onset == 100
    assert assessment.robust_onset in {100, 101, 102}
    assert assessment.strict_offset == 142
    assert assessment.robust_offset in {140, 141, 142}
    assert assessment.onset_ci_low <= assessment.robust_onset
    assert assessment.offset_ci_high >= assessment.robust_offset
    assert set(assessment.valid_lead_groups) == {"limb", "right_precordial"}

    limb_only = _fuse(
        [
            _boundary("I", 100, 140),
            _boundary("II", 101, 141),
            _boundary("III", 102, 142),
        ]
    )
    assert not limb_only.accepted
    assert INSUFFICIENT_INFORMATIVE_LEADS in limb_only.reject_reasons


def test_overlap_and_unobservable_baseline_are_explicit_rejections() -> None:
    unresolved = _fuse(
        [
            _boundary("I", 100, 140, on_t_overlap=True),
            _boundary("II", 101, 141, on_t_overlap=True),
            _boundary("V2", 102, 142, on_t_overlap=True),
        ]
    )
    assert not unresolved.accepted
    assert P_ON_T_UNRESOLVED in unresolved.reject_reasons

    no_baseline = _fuse(
        [
            _boundary("I", 100, 140, baseline_mode="baseline_unobservable"),
            _boundary("II", 101, 141, baseline_mode="baseline_unobservable"),
            _boundary("V2", 102, 142, baseline_mode="baseline_unobservable"),
        ]
    )
    assert not no_baseline.accepted
    assert BASELINE_UNOBSERVABLE in no_baseline.reject_reasons


def test_rhythm_evidence_can_overrule_boundary_acceptance() -> None:
    assessment = _fuse(
        [
            _boundary("I", 100, 140),
            _boundary("II", 101, 141),
            _boundary("V2", 102, 142),
        ]
    )

    finalize_p_wave_states([assessment], {"probable_af": True})

    assert not assessment.accepted
    assert assessment.p_state == AF_LIKE
    assert "RHYTHM_AF_LIKE" in assessment.reject_reasons


def test_parallel_model_reconciliation_reduces_contraction_and_trusts_own_verdict_without_legacy() -> None:
    assessment = _fuse(
        [
            _boundary("I", 100, 140),
            _boundary("II", 101, 141),
            _boundary("V2", 102, 142),
        ]
    )
    original_onset = int(assessment.robust_onset)
    original_offset = int(assessment.robust_offset)

    _reconcile_parallel_consensus_models(
        assessment,
        legacy_onset=105,
        legacy_offset=150,
        fs=500,
    )

    assert assessment.robust_onset == round(0.5 * original_onset + 0.5 * 105)
    early = min(original_offset, 150)
    late = max(original_offset, 150)
    assert assessment.robust_offset == round(
        0.25 * (0.5 * (early + late)) + 0.75 * late
    )
    assert assessment.strict_onset <= assessment.robust_onset
    assert assessment.strict_offset >= assessment.robust_offset
    assert assessment.accepted

    # A missing legacy consensus is "no second opinion", not "disagreement":
    # the robust model's own cross-lead verdict from `_fuse` (already >=3
    # informative leads across >=2 independent groups here) stands as-is
    # instead of being force-rejected. Reconciliation only adjusts already
    # jointly-accepted assessments; with no legacy value it must be a no-op.
    missing_parallel_model = _fuse(
        [
            _boundary("I", 100, 140),
            _boundary("II", 101, 141),
            _boundary("V2", 102, 142),
        ]
    )
    accepted_before_reconciliation = missing_parallel_model.accepted
    onset_before_reconciliation = missing_parallel_model.robust_onset
    offset_before_reconciliation = missing_parallel_model.robust_offset
    _reconcile_parallel_consensus_models(
        missing_parallel_model,
        legacy_onset=100,
        legacy_offset=None,
        fs=500,
    )
    assert missing_parallel_model.accepted == accepted_before_reconciliation
    assert missing_parallel_model.robust_onset == onset_before_reconciliation
    assert missing_parallel_model.robust_offset == offset_before_reconciliation
    assert "MODEL_DISAGREEMENT" not in missing_parallel_model.reject_reasons


def test_acquisition_delay_compensation_is_guarded_and_fractional() -> None:
    fs = 500
    n_samples = 5000
    r_locs = np.arange(500, 4501, 500)
    positions = np.arange(n_samples, dtype=float)
    reference = np.zeros(n_samples, dtype=float)
    for r_sample in r_locs:
        reference += np.exp(-0.5 * ((positions - r_sample) / 8.0) ** 2)
        reference -= 0.4 * np.exp(
            -0.5 * ((positions - (r_sample + 18)) / 10.0) ** 2
        )
    rng = np.random.default_rng(20260728)
    ecg = np.vstack(
        [reference + 0.001 * rng.normal(size=n_samples) for _ in range(12)]
    )
    delayed = np.interp(
        positions - 5.0,
        positions,
        reference,
        left=reference[0],
        right=reference[-1],
    )
    ecg[7] = delayed + 0.001 * rng.normal(size=n_samples)  # V2, +10 ms.

    audit = assess_acquisition_chain(ecg, fs, r_locs)

    assert "V2" in audit["approved_delay_samples"]
    assert abs(audit["approved_delay_ms"]["V2"] - 10.0) < 1.0
    corrected, applied = apply_channel_delay_compensation(
        ecg,
        audit["approved_delay_samples"],
    )
    assert applied == ["V2"]
    before = np.corrcoef(ecg[7], reference)[0, 1]
    after = np.corrcoef(corrected[7], reference)[0, 1]
    assert after > before + 0.10


def test_zero_mad_large_fixed_delay_is_rejected(monkeypatch) -> None:
    call_index = 0

    def fake_delay_evidence(*_args, **_kwargs):
        nonlocal call_index
        lead_index = call_index
        call_index += 1
        if lead_index == 7:  # V2
            return {
                "available": True,
                "delay_samples": 15.0,
                "delay_ms": 30.0,
                "mad_ms": 0.0,
                "beat_count": 8,
                "raw_envelope_correlation": 0.60,
                "corrected_envelope_correlation": 0.95,
                "correlation_improvement": 0.35,
                "automatic_compensation_approved": False,
            }
        return {
            "available": True,
            "delay_samples": 0.0,
            "delay_ms": 0.0,
            "mad_ms": 0.0,
            "beat_count": 8,
            "raw_envelope_correlation": 0.95,
            "corrected_envelope_correlation": 0.95,
            "correlation_improvement": 0.0,
            "automatic_compensation_approved": False,
        }

    monkeypatch.setattr(acquisition_qc, "_lead_delay_evidence", fake_delay_evidence)
    monkeypatch.setattr(acquisition_qc, "_duplicate_channel_pairs", lambda _ecg: [])
    monkeypatch.setattr(
        acquisition_qc,
        "_gain_and_filter_outliers",
        lambda _ecg, _fs: ([], []),
    )
    rng = np.random.default_rng(20260823)
    audit = assess_acquisition_chain(
        rng.normal(size=(12, 1000)),
        500,
        np.arange(100, 901, 100),
    )

    assert audit["suspicious_delay_leads"] == ["V2"]
    assert audit["channel_desynchronized"] is True
    assert audit["fusion_allowed"] is False
    assert "CHANNEL_DESYNCHRONIZED" in audit["reject_reasons"]


def _ta_boundary(
    lead: str,
    *,
    ta_ambiguous: bool,
    branch_disagreement_ms: float,
) -> PWaveLeadBoundary:
    row = _boundary(lead, 100, 140)
    row.ta_ambiguous = ta_ambiguous
    row.branch_disagreement_ms = branch_disagreement_ms
    return row


def test_one_ta_flagged_lead_does_not_make_the_beat_ta_ambiguous() -> None:
    # A Ta deflection is present in the PR segment of every ECG, so an OR across
    # twelve leads flagged ~95% of beats. Consumers treat `ta_ambiguous` as
    # blocking for P-morphology conclusions, so it needs a quorum of leads whose
    # boundaries are actually unstable.
    stable = {"ta_ambiguous": False, "branch_disagreement_ms": 1.0}
    unstable = {"ta_ambiguous": True, "branch_disagreement_ms": 40.0}

    one_flagged = _fuse(
        [
            _ta_boundary("I", **unstable),
            _ta_boundary("II", **stable),
            _ta_boundary("III", **stable),
            _ta_boundary("V2", **stable),
        ]
    )
    assert one_flagged.ta_ambiguous is False

    majority_flagged = _fuse(
        [
            _ta_boundary("I", **unstable),
            _ta_boundary("II", **unstable),
            _ta_boundary("III", **stable),
            _ta_boundary("V2", **stable),
        ]
    )
    assert majority_flagged.ta_ambiguous is True


def test_ta_flagged_leads_with_stable_boundaries_do_not_block_the_beat() -> None:
    # Flagged but branch-stable leads carry no evidence that the offset moved.
    assessment = _fuse(
        [
            _ta_boundary("I", ta_ambiguous=True, branch_disagreement_ms=1.0),
            _ta_boundary("II", ta_ambiguous=True, branch_disagreement_ms=1.0),
            _ta_boundary("III", ta_ambiguous=True, branch_disagreement_ms=2.0),
            _ta_boundary("V2", ta_ambiguous=True, branch_disagreement_ms=1.5),
        ]
    )

    assert assessment.ta_ambiguous is False
    assert assessment.accepted
