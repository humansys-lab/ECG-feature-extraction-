"""Named ST-segment morphology classes (st_localization.classify_st_pattern)."""
from __future__ import annotations

from feature_extraction.ecgfeat.st_localization import (
    ST_PATTERN_DOWNSLOPING,
    ST_PATTERN_HORIZONTAL,
    ST_PATTERN_J_POINT,
    ST_PATTERN_UNKNOWN,
    ST_PATTERN_UPSLOPING,
    classify_st_pattern,
)


def _classify(**overrides):
    kwargs = {
        "j_mv": 0.0,
        "trend": "horizontal",
        "shape": "straight",
        "notched_or_slurred": False,
        "reliable": True,
    }
    kwargs.update(overrides)
    return classify_st_pattern(**kwargs)


def test_flat_st_is_horizontal() -> None:
    assert _classify(trend="horizontal") == ST_PATTERN_HORIZONTAL


def test_descending_st_is_downsloping() -> None:
    assert _classify(trend="downsloping") == ST_PATTERN_DOWNSLOPING


def test_rising_st_is_slow_upsloping() -> None:
    assert _classify(trend="upsloping") == ST_PATTERN_UPSLOPING


def test_notched_elevated_j_point_is_the_early_repolarisation_class() -> None:
    assert (
        _classify(j_mv=0.15, trend="upsloping", notched_or_slurred=True)
        == ST_PATTERN_J_POINT
    )


def test_concave_up_elevated_j_point_also_qualifies_without_a_notch() -> None:
    assert (
        _classify(j_mv=0.15, trend="horizontal", shape="concave-up")
        == ST_PATTERN_J_POINT
    )


def test_j_point_elevation_alone_is_not_the_early_repolarisation_class() -> None:
    """Elevation without a notch or concave-up shape is just elevated ST."""
    assert (
        _classify(j_mv=0.15, trend="horizontal", shape="straight")
        == ST_PATTERN_HORIZONTAL
    )


def test_a_descending_st_is_never_called_early_repolarisation() -> None:
    """Downsloping is the most ischemia-specific shape and must win outright."""
    assert (
        _classify(j_mv=0.20, trend="downsloping", notched_or_slurred=True)
        == ST_PATTERN_DOWNSLOPING
    )


def test_small_j_elevation_does_not_reach_the_early_repolarisation_class() -> None:
    assert (
        _classify(j_mv=0.05, trend="upsloping", notched_or_slurred=True)
        == ST_PATTERN_UPSLOPING
    )


def test_unreliable_measurement_is_not_classified() -> None:
    assert _classify(trend="downsloping", reliable=False) == ST_PATTERN_UNKNOWN


def test_unknown_trend_is_not_classified() -> None:
    assert _classify(trend="unknown") == ST_PATTERN_UNKNOWN


def test_missing_j_amplitude_falls_back_to_the_trend_classes() -> None:
    assert (
        _classify(j_mv=None, trend="upsloping", notched_or_slurred=True)
        == ST_PATTERN_UPSLOPING
    )
