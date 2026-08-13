from __future__ import annotations

from analyze_ludb_p_boundaries import _match_by_r, summarize


def test_match_by_r_is_one_to_one_and_respects_tolerance() -> None:
    reference = [
        {"r_index": 100},
        {"r_index": 200},
        {"r_index": 500},
    ]
    algorithm = [
        {"r_index": 102},
        {"r_index": 198},
        {"r_index": 700},
    ]

    pairs = _match_by_r(reference, algorithm, fs=1000)

    assert [(pair[0]["r_index"], pair[1]["r_index"]) for pair in pairs] == [
        (100, 102),
        (200, 198),
    ]


def test_summary_decomposes_duration_as_offset_minus_onset() -> None:
    consensus_rows = [
        {
            "record_id": "1",
            "onset_error_ms": 20.0,
            "offset_error_ms": -5.0,
            "duration_error_ms": -25.0,
        },
        {
            "record_id": "1",
            "onset_error_ms": 10.0,
            "offset_error_ms": -15.0,
            "duration_error_ms": -25.0,
        },
    ]
    result = {
        "record_id": "1",
        "error": None,
        "record": {
            "record_id": "1",
            "algorithm_p_duration_ms": 75.0,
            "reference_p_duration_ms": 100.0,
            "duration_difference_ms": -25.0,
        },
        "consensus_pairs": consensus_rows,
        "per_lead_pairs": [],
    }

    summary, *_ = summarize([result])
    decomposition = summary["consensus_beat_level"]["decomposition"]

    assert decomposition["onset_contraction_component_ms"] == -15.0
    assert decomposition["offset_contraction_component_ms"] == -10.0
    assert decomposition["reconstructed_duration_error_ms"] == -25.0
    assert decomposition["direct_duration_error_ms"] == -25.0


def test_summary_separates_raw_and_corrected_per_lead_boundaries() -> None:
    result = {
        "record_id": "1",
        "error": None,
        "record": {
            "record_id": "1",
            "algorithm_p_duration_ms": 75.0,
            "reference_p_duration_ms": 100.0,
            "duration_difference_ms": -25.0,
        },
        "consensus_pairs": [
            {
                "record_id": "1",
                "onset_error_ms": 10.0,
                "offset_error_ms": -5.0,
                "duration_error_ms": -15.0,
            }
        ],
        "per_lead_pairs": [
            {
                "record_id": "1",
                "lead": "II",
                "onset_error_ms": 10.0,
                "offset_error_ms": -5.0,
                "duration_error_ms": -15.0,
                "raw_onset_error_ms": 4.0,
                "raw_offset_error_ms": 3.0,
                "raw_duration_error_ms": -1.0,
                "onset_correction_shift_ms": 6.0,
                "offset_correction_shift_ms": -8.0,
                "duration_correction_delta_ms": -14.0,
            }
        ],
    }

    summary, *_ = summarize([result])

    assert summary["per_lead_raw_boundaries"]["duration_error"]["mean_ms"] == -1.0
    assert summary["per_lead_final_boundaries"]["duration_error"]["mean_ms"] == -15.0
    correction = summary["per_lead_correction_effect"]
    assert correction["onset_shift"]["mean_ms"] == 6.0
    assert correction["offset_shift"]["mean_ms"] == -8.0
    assert correction["duration_delta"]["mean_ms"] == -14.0
