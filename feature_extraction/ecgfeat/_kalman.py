"""Optional compilation of the existing scalar recursion, without fast-math.

Numba is optional. No signal or patient result is cached, only machine code.
Set ECGFEAT_DISABLE_JIT=1 before extraction to use the Python reference path.
"""
from functools import lru_cache
from math import sqrt
import os
import warnings

import numpy as np


# This source can be imported as either ecgfeat._kalman (installed package)
# or feature_extraction.ecgfeat._kalman (repository tests). Numba serializes
# the import name in its disk cache: only the installed name may write it.
def kalman_recursion(y, mask, fs, initial, measurement_variance):
    n = y.size
    dt = 1.0 / float(fs)
    process_level = measurement_variance * 2e-3
    process_slope = measurement_variance * 2e-1
    state_level = initial
    state_slope = 0.0
    covariance_00 = measurement_variance * 4.0
    covariance_01 = 0.0
    covariance_10 = 0.0
    covariance_11 = measurement_variance
    levels = np.zeros(n, dtype=np.float64)
    variances = np.zeros(n, dtype=np.float64)
    for index in range(n):
        state_level = state_level + dt * state_slope
        predicted_00 = (
            covariance_00
            + dt * (covariance_10 + covariance_01)
            + dt * dt * covariance_11
            + process_level
        )
        predicted_01 = covariance_01 + dt * covariance_11
        predicted_10 = covariance_10 + dt * covariance_11
        predicted_11 = covariance_11 + process_slope
        covariance_00 = predicted_00
        covariance_01 = predicted_01
        covariance_10 = predicted_10
        covariance_11 = predicted_11
        if mask[index] and np.isfinite(y[index]):
            innovation = float(y[index] - state_level)
            innovation_variance = max(covariance_00 + measurement_variance, 1e-12)
            gain_level = covariance_00 / innovation_variance
            gain_slope = covariance_10 / innovation_variance
            limit = 3.0 * sqrt(innovation_variance)
            innovation = min(limit, max(-limit, innovation))
            state_level += gain_level * innovation
            state_slope += gain_slope * innovation
            old_00 = covariance_00
            old_01 = covariance_01
            covariance_00 = (1.0 - gain_level) * old_00
            covariance_01 = (1.0 - gain_level) * old_01
            covariance_10 = covariance_10 - gain_slope * old_00
            covariance_11 = covariance_11 - gain_slope * old_01
        levels[index] = state_level
        variances[index] = max(covariance_00, 0.0)
    # Keep the sqrt ufunc in the wrapper, matching NumPy's reference path.
    return levels, variances


@lru_cache(maxsize=1)
def _compiled_kernel():
    try:
        from numba import njit
    except ImportError:
        return kalman_recursion
    try:
        compiled = njit(cache=(__name__ == "ecgfeat._kalman"), fastmath=False)(kalman_recursion)
        # Compile both layouts before accepting the backend. Reverse passes
        # deliberately use negative strides, as in the Python reference.
        x, mask = np.zeros(2), np.zeros(2, dtype=bool)
        compiled(x, mask, 500, 0., 1e-8)
        compiled(x[::-1], mask[::-1], 500, 0., 1e-8)
        return compiled
    except Exception as exc:
        # Compilation/cache support varies across installations; scientific
        # extraction must remain available using only NumPy/SciPy.
        warnings.warn(f"ECG Kalman compilation unavailable; using Python ({type(exc).__name__})",
                      RuntimeWarning, stacklevel=2)
        return kalman_recursion


def run_kalman(y, mask, fs, initial, measurement_variance):
    kernel = kalman_recursion if os.environ.get("ECGFEAT_DISABLE_JIT") == "1" else _compiled_kernel()
    levels, variances = kernel(y, mask, fs, initial, measurement_variance)
    return levels, np.sqrt(variances)
