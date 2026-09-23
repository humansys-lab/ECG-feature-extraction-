import numpy as np
import pytest

from feature_extraction.ecgfeat import _kalman


@pytest.mark.parametrize("fs", [250, 500, 1000])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "all", "sparse"])
def test_compiled_kalman_exactly_matches_python_with_missing_and_outliers(fs, reverse, mask_kind):
    pytest.importorskip("numba")
    rng = np.random.default_rng(1923)
    y = rng.normal(0, .03, 521)
    y[17], y[319] = 20., np.nan
    mask = {"none": np.zeros(521, bool), "all": np.ones(521, bool),
            "sparse": rng.random(521) < .15}[mask_kind]
    if reverse:
        y, mask = y[::-1], mask[::-1]
    for variance in [1e-8, .001, 1.]:
        expected = _kalman.kalman_recursion(y, mask, fs, .05, variance)
        actual = _kalman._compiled_kernel()(y, mask, fs, .05, variance)
        for x, z in zip(expected, actual):
            np.testing.assert_array_equal(x, z)


def test_disabled_jit_never_imports_or_compiles_optional_backend(monkeypatch):
    monkeypatch.setenv("ECGFEAT_DISABLE_JIT", "1")
    def forbidden():
        raise AssertionError("disabled JIT must not be touched")
    monkeypatch.setattr(_kalman, "_compiled_kernel", forbidden)
    y, mask = np.arange(10, dtype=float), np.ones(10, bool)
    levels, variances = _kalman.kalman_recursion(y, mask, 500, 0., .001)
    result = _kalman.run_kalman(y, mask, 500, 0., .001)
    np.testing.assert_array_equal(result[0], levels)
    np.testing.assert_array_equal(result[1], np.sqrt(variances))


def test_missing_optional_numba_falls_back_without_installing(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def no_numba(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("optional dependency absent")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_numba)
    assert _kalman._compiled_kernel.__wrapped__() is _kalman.kalman_recursion


def test_empty_baseline_remains_supported():
    from feature_extraction.ecgfeat.p_wave_engine import _kalman_baseline_pass
    a, b = _kalman_baseline_pass(np.array([]), np.array([], bool), 500)
    assert a.size == b.size == 0


def test_installed_import_can_compile_after_repository_import(tmp_path):
    """Disk caches must not require the development checkout's import name."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    numba = pytest.importorskip("numba")
    if numba.config.DISABLE_JIT:
        pytest.skip("Numba JIT disabled by environment")
    _kalman._compiled_kernel()
    code = ("from ecgfeat._kalman import _compiled_kernel, kalman_recursion; "
            "k=_compiled_kernel(); assert k is not kalman_recursion; "
            "assert len(k.signatures) == 2")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "feature_extraction")}
    subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, check=True, capture_output=True)
