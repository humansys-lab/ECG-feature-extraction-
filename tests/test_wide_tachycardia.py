from types import SimpleNamespace

from feature_extraction.ecgfeat.clinical_rules.wide_tachycardia import (
    evaluate_wide_complex_tachycardia,
)


def _context(
    *,
    probable_af=False,
    probable_flutter=False,
    ns_vt=False,
    validated_av_dissociation=False,
    pacing=None,
):
    values = {"heart_rate_bpm": 140.0, "qrs_ms": 188.0}
    return SimpleNamespace(
        global_value=lambda name: values.get(name),
        features=SimpleNamespace(
            metadata={
                "rhythm_analysis": {
                    "rule_summary": {
                        "av_dissociation": True,
                        "validated_multilead_av_dissociation": (
                            validated_av_dissociation
                        ),
                    },
                    "af_afl_summary": {
                        "probable_af": probable_af,
                        "probable_flutter": probable_flutter,
                    },
                    "pacing_context": pacing or {},
                }
            },
            interpretation=SimpleNamespace(non_sustained_vt=ns_vt),
        ),
    )


def test_pr_variability_alone_keeps_wide_complex_tachycardia_uncertain() -> None:
    result = evaluate_wide_complex_tachycardia(_context())

    assert result.status == "matched"
    assert result.statement_code == "wide_complex_tachycardia"
    assert result.confidence == "medium"
    assert result.evidence["av_dissociation_raw"]
    assert not result.evidence["av_dissociation_usable"]


def test_validated_multilead_av_dissociation_supports_vt_pattern() -> None:
    result = evaluate_wide_complex_tachycardia(
        _context(validated_av_dissociation=True)
    )

    assert result.statement_code == "ventricular_tachycardia_pattern"
    assert result.confidence == "high"
    assert result.evidence["av_dissociation_usable"]


def test_af_invalidates_pr_based_av_dissociation_but_keeps_wct_alert() -> None:
    result = evaluate_wide_complex_tachycardia(_context(probable_af=True))

    assert result.status == "matched"
    assert result.statement_code == "wide_complex_tachycardia"
    assert result.confidence == "medium"
    assert result.priority == "P0"
    assert result.evidence["av_dissociation_raw"]
    assert not result.evidence["av_dissociation_usable"]
    assert (
        "probable_atrial_fibrillation"
        in result.evidence["atrial_rhythm_confounders"]
    )


def test_independent_nsvt_evidence_can_support_vt_during_af() -> None:
    result = evaluate_wide_complex_tachycardia(
        _context(probable_af=True, ns_vt=True)
    )

    assert result.statement_code == "ventricular_tachycardia_pattern"
    assert result.confidence == "high"


def test_pacing_like_context_invalidates_av_dissociation_upgrade() -> None:
    result = evaluate_wide_complex_tachycardia(
        _context(
            validated_av_dissociation=True,
            pacing={"suppress_further_rhythm_interpretation": True},
        )
    )

    assert result.statement_code == "wide_complex_tachycardia"
    assert not result.evidence["av_dissociation_usable"]
    assert (
        "ventricular_pacing_or_pacing_like_context"
        in result.evidence["atrial_rhythm_confounders"]
    )
