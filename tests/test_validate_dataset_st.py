from __future__ import annotations

from validate_dataset_st import (
    _confusion_metrics,
    _parse_header,
    _qualifying_leads,
)


def test_parse_header_separates_direct_st_and_exploratory_labels(tmp_path) -> None:
    header = tmp_path / "JS_TEST.hea"
    header.write_text(
        "JS_TEST 12 500 5000\n"
        "#Age: 71\n"
        "#Sex: Female\n"
        "#Dx: 429622005,428750005,164934002\n",
        encoding="utf-8",
    )

    result = _parse_header(header)

    assert result["record"] == "JS_TEST"
    assert result["label_st_depression"]
    assert not result["label_st_elevation"]
    assert result["label_nonspecific_st_t"]
    assert result["direct_st_label"]
    assert result["exploratory_st_context"]


def test_qualifying_leads_requires_two_leads_in_same_territory() -> None:
    rows = [
        {
            "lead": "II",
            "hybrid_st40_mv": -0.08,
            "hybrid_st_reliable": True,
            "hybrid_strict_support_pass": True,
        },
        {
            "lead": "III",
            "hybrid_st40_mv": -0.07,
            "hybrid_st_reliable": True,
            "hybrid_strict_support_pass": True,
        },
        {
            "lead": "V5",
            "hybrid_st40_mv": -0.09,
            "hybrid_st_reliable": True,
            "hybrid_strict_support_pass": True,
        },
    ]

    leads, territories = _qualifying_leads(
        rows,
        value_key="hybrid_st40_mv",
        reliable_key="hybrid_st_reliable",
        threshold_mv=0.05,
        direction="depression",
    )

    assert leads == ["II", "III", "V5"]
    assert territories == ["inferior"]


def test_strict_control_metrics_exclude_st_t_context_from_negatives() -> None:
    rows = [
        {
            "status": "ok",
            "label_st_depression": True,
            "prediction": True,
        },
        {
            "status": "ok",
            "label_st_depression": False,
            "prediction": True,
            "label_nonspecific_st_t": True,
        },
        {
            "status": "ok",
            "label_st_depression": False,
            "prediction": False,
        },
    ]

    all_labels = _confusion_metrics(
        rows,
        label_key="label_st_depression",
        prediction_key="prediction",
        strict_controls=False,
    )
    strict = _confusion_metrics(
        rows,
        label_key="label_st_depression",
        prediction_key="prediction",
        strict_controls=True,
    )

    assert (all_labels["tp"], all_labels["fp"], all_labels["tn"]) == (1, 1, 1)
    assert (strict["tp"], strict["fp"], strict["tn"]) == (1, 0, 1)
