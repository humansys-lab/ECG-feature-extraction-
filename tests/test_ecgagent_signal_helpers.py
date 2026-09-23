"""ecgagent's vendored raw-signal helpers stay identical to the engine's."""

import numpy as np
import pytest

from ecgagent.evidence import signal_helpers
from feature_extraction.ecgfeat._engine.atrial.validation import p_waveform_evidence as engine_p
from feature_extraction.ecgfeat._engine.preprocess import lowpass_filter as engine_lowpass


@pytest.mark.parametrize("fs", [250, 360, 500, 1000])
@pytest.mark.parametrize("seed", range(5))
def test_helpers_match_the_engine_bit_for_bit(fs, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(4 * fs) / fs
    signal = 0.15 * np.exp(-0.5 * ((t - 1.0) / 0.02) ** 2) + rng.normal(0, 0.01, t.size)
    ours, theirs = signal_helpers.lowpass_filter(signal, fs, 15.0), engine_lowpass(signal, fs, 15.0)
    np.testing.assert_array_equal(ours, theirs)
    for peak in (fs, fs // 2, 3 * fs, 5):
        a, b = signal_helpers.p_waveform_evidence(signal, ours, peak, fs), engine_p(signal, theirs, peak, fs)
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_array_equal(a.pop("snippet"), b.pop("snippet"))
            assert a == b


def test_short_signal_is_returned_unfiltered():
    x = np.arange(5.0)
    np.testing.assert_array_equal(signal_helpers.lowpass_filter(x, 500, 15.0), x)
