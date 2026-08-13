from __future__ import annotations

from typing import Optional

from feature_extraction.ecgfeat.models import (
    BeatAnnotation,
    ECGFeatures,
    GlobalFeatures,
    GroupFeatures,
    LeadBeatFeatures,
    LeadQuality,
    PatientMeta,
    RepresentativeLeadFeatures,
    STANDARD_12_LEADS,
    WaveBounds,
)


def make_features(meta: Optional[PatientMeta] = None) -> ECGFeatures:
    quality = {
        lead: LeadQuality(
            lead=lead,
            baseline_wander_score=0.01,
            muscle_noise_score=0.01,
            powerline_score=0.0,
            clipping_score=0.0,
            flatline_score=0.0,
            missing=False,
            reliable=True,
            grade="Q1",
        )
        for lead in STANDARD_12_LEADS
    }
    beat_features = []
    representative_leads = {}
    for index, lead in enumerate(STANDARD_12_LEADS):
        qrs_area = 1.0 + index * 0.1
        beat_features.append(
            LeadBeatFeatures(
                lead=lead,
                beat_id=0,
                p=WaveBounds(onset=420, peak=440, offset=460),
                qrs=WaveBounds(onset=480, peak=500, offset=525),
                t=WaveBounds(onset=560, peak=650, offset=720),
                qt_ms=480.0,
                pr_ms=120.0,
                qrs_ms=90.0,
                p_amp_mv=0.12,
                qrs_area=qrs_area,
                q_amp_mv=-0.05,
                r_amp_mv=0.8,
                s_amp_mv=-0.2,
                st_on_mv=0.01,
                st_mid_mv=0.02,
                st_80ms_mv=0.03,
                t_amp_mv=0.2,
                j_index=525,
                qrs_signed_area=0.9,
                p_confidence=0.9,
                qrs_confidence=0.95,
                beat_measurement_reliable=True,
            )
        )
        representative_leads[lead] = RepresentativeLeadFeatures(
            lead=lead,
            params={
                "pr_ms": 150.0,
                "qrs_ms": 90.0,
                "qt_ms": 400.0,
                "p_amp_mv": 0.12,
                "q_amp_mv": -0.05,
                "r_amp_mv": 0.8,
                "s_amp_mv": -0.2,
                "st_on_mv": 0.01,
                "st_mid_mv": 0.02,
                "st_80ms_mv": 0.03,
                "t_amp_mv": 0.2,
                "qrs_area": qrs_area,
                "qrs_signed_area": 0.9,
                "reliable_for_p": True,
                "reliable_for_qrs": True,
                "reliable_for_t": True,
                "reliable_for_qt": True,
            },
            variance={},
        )

    metadata = {
        "input_fs": 500,
        "internal_fs": 500,
        "lead_order": list(STANDARD_12_LEADS),
        "n_beats": 2,
        "measurement_beat_ids": [0, 1],
        "representative_group_id": 1,
        "record_quality": {
            "record_grade": "Q1",
            "reason_codes": [],
            "rejected_functions": [],
        },
        "rhythm_analysis": {
            "availability": {
                "p_axis_available": True,
                "atrial_rhythm_available": True,
            },
            "pacing_context": {
                "continuous_pacing": False,
                "suppress_further_rhythm_interpretation": False,
            },
            "rule_summary": {
                "primary_statement": "sinus_rhythm",
                "preexcitation": {"wpw_pattern": False},
            },
        },
    }
    if meta is not None:
        metadata["patient_meta"] = meta

    return ECGFeatures(
        fs=500,
        quality=quality,
        beats=[
            BeatAnnotation(0, 500, False, 1, None, 1000.0),
            BeatAnnotation(1, 1000, False, 1, 1000.0, None),
        ],
        beat_features=beat_features,
        representative_leads=representative_leads,
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
                flags={"dominant_group": True},
            )
        },
        global_features=GlobalFeatures(
            heart_rate_bpm=60.0,
            atrial_rate_bpm=60.0,
            pr_ms=150.0,
            qrs_ms=90.0,
            qt_ms=400.0,
            qtc_bazett_ms=400.0,
            qtc_fridericia_ms=400.0,
            p_axis_deg=40.0,
            qrs_axis_deg=55.0,
            t_axis_deg=35.0,
            st_axis_deg=20.0,
            qt_dispersion_ms=25.0,
        ),
        metadata=metadata,
    )
