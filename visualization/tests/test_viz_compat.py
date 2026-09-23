"""Compatibility surface: ecgfeat.viz forwarding and the legacy ecgfeat.visualize alias."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

import ecgrecords_viz

SIX = {"plot_record", "plot_beat", "plot_beat_all_leads", "plot_representative_beat", "plot_quality_summary",
       "VisualizationInputError"}


def test_ecgfeat_viz_surface():
    import ecgfeat.viz

    assert set(ecgfeat.viz.__all__) == SIX
    assert SIX <= set(dir(ecgfeat.viz))
    for name in SIX:
        assert getattr(ecgfeat.viz, name) is getattr(ecgrecords_viz, name)
    with pytest.raises(AttributeError):
        ecgfeat.viz.plot_rep_beat  # noqa: B018 - legacy names are not forwarded here
    with pytest.raises(AttributeError):
        ecgfeat.viz.__version__  # noqa: B018


def test_ecgfeat_viz_is_not_a_top_level_reexport():
    import ecgfeat

    assert not SIX & (set(ecgfeat.__all__) - {"plot_beat", "plot_beat_all_leads", "plot_quality_summary"})
    assert "viz" not in ecgfeat.__all__


def test_legacy_alias_is_the_moved_module_and_warns_once():
    legacy = importlib.import_module("ecgrecords_viz.legacy")
    sys.modules.pop("ecgfeat.visualize", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = importlib.import_module("ecgfeat.visualize")
        again = importlib.import_module("ecgfeat.visualize")
    assert alias is legacy and again is legacy and sys.modules["ecgfeat.visualize"] is legacy
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
                and "ecgfeat.visualize" in str(w.message)]
    assert len(messages) == 1
    assert "ecgrecords_viz" in messages[0] and "no earlier than ecg-records 0.3.0" in messages[0]
    from ecgfeat import _moved

    assert f"ecg-records {_moved.REMOVAL_RELEASE}" in messages[0]


def test_legacy_module_keeps_its_public_functions():
    from ecgrecords_viz import legacy

    for name in ("plot_beat", "plot_rep_beat", "plot_beat_all_leads", "plot_quality_summary"):
        assert callable(getattr(legacy, name))
    from ecgfeat.compat.models_v0 import ECGFeatures, STANDARD_12_LEADS

    assert legacy.ECGFeatures is ECGFeatures and legacy.STANDARD_12_LEADS == STANDARD_12_LEADS


def test_legacy_helpers_still_plot_legacy_features(signal):
    import matplotlib.pyplot as plt
    import numpy as np

    from ecgfeat.api import ECGFeatureExtractor
    from ecgrecords_viz import legacy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = ECGFeatureExtractor().extract(signal, 500.0)
    try:
        ax = legacy.plot_beat(signal, features, lead="II", beat_id=3, show=False)
        assert ax.lines
        fig = legacy.plot_beat_all_leads(signal, features, beat_id=3, show=False)
        assert len(fig.axes) == 12
        fig = legacy.plot_quality_summary(features, show=False)
        assert fig.axes
        ax = legacy.plot_rep_beat({1: np.asarray(signal)[:, 100:550]}, features, group_id=1, lead="II", show=False)
        assert ax.lines
    finally:
        plt.close("all")
