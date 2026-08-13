from feature_extraction.ecgfeat.clinical_rules.validation import (
    binary_metrics,
    stratified_metrics,
)


def test_binary_metrics_exclude_unavailable_predictions() -> None:
    rows = [
        {"truth": True, "predicted": True},
        {"truth": True, "predicted": False},
        {"truth": False, "predicted": False},
        {"truth": False, "predicted": None},
    ]

    result = binary_metrics(rows)

    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["fp"] == 0
    assert result["unavailable"] == 1
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 1.0
    assert result["coverage"] == 0.75


def test_zero_denominator_metrics_are_none() -> None:
    result = binary_metrics([{"truth": False, "predicted": False}])

    assert result["sensitivity"] is None
    assert result["ppv"] is None
    assert result["specificity"] == 1.0


def test_stratification_keeps_exact_sex_and_quality_groups() -> None:
    rows = [
        {"truth": True, "predicted": True, "sex": "female", "quality": "reliable"},
        {"truth": False, "predicted": False, "sex": "male", "quality": "limited"},
        {"truth": True, "predicted": None, "sex": "female", "quality": "reliable"},
    ]

    result = stratified_metrics(rows, keys=("sex", "quality"))

    assert ("female", "reliable") in result
    assert result[("female", "reliable")]["unavailable"] == 1
    assert ("male", "limited") in result
