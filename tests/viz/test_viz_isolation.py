"""Import isolation, checked in fresh interpreters so no other test can mask it."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

INSTALL_HINT = 'plotting needs Matplotlib; install it with: pip install "ecg-records[viz]"'
try:
    import matplotlib  # noqa: F401
except ImportError:
    HAS_MATPLOTLIB = False
else:
    HAS_MATPLOTLIB = True
needs_matplotlib = pytest.mark.skipif(not HAS_MATPLOTLIB, reason='optional viz extra: pip install "ecg-records[viz]"')


def run_python(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                            env=env, timeout=300)
    assert result.returncode == 0, result.stderr
    return result


CALLS = {
    "plot_record": "plot_record(signal, record{figure}, annotations='all')",
    "plot_beat": "plot_beat(signal, record{figure}, beat=2, lead='V2')",
    "plot_beat_all_leads": "plot_beat_all_leads(signal, record{figure}, beat=2)",
    "plot_representative_beat": "plot_representative_beat(signal, record{figure}, lead='II')",
    "plot_quality_summary": "plot_quality_summary(signal, record{figure})",
}


@needs_matplotlib
@pytest.mark.parametrize("name", list(CALLS))
def test_plotting_never_imports_pyplot(reference_dir, name):
    result = run_python(f"""
        import io, sys
        import numpy as np
        from ecgfeat.record import load_record
        from ecgfeat.viz import {name}
        from matplotlib.figure import Figure

        record = load_record({str(reference_dir / "record.all.json")!r})
        with np.load({str(reference_dir / "signal.npz")!r}) as archive:
            signal = archive["signal"]
        figure, axes = {CALLS[name].format(figure="")}
        figure.savefig(io.BytesIO(), format="png")
        supplied = Figure()
        {CALLS[name].format(figure=", figure=supplied")}
        supplied.savefig(io.BytesIO(), format="png")
        assert "matplotlib.pyplot" not in sys.modules, "pyplot was imported"
        assert "ecgfeat.viz.legacy" not in sys.modules, "legacy helpers were imported"
        print("ok")
    """)
    assert result.stdout.strip() == "ok"


def test_core_imports_never_load_matplotlib_or_the_plotting_modules(reference_dir):
    run_python(f"""
        import sys
        import ecgfeat, ecgfeat.record, ecgfeat.viz
        from ecgfeat.record import load_record, query_measurement

        record = load_record({str(reference_dir / "record.all.json")!r})
        query_measurement(record, "r_peak", lead="II", beat=0)
        assert set(ecgfeat.viz.__all__) <= set(dir(ecgfeat.viz))
        loaded = sorted(m for m in sys.modules if m.split(".")[0] == "matplotlib"
                        or m in {{"ecgfeat.viz.plots", "ecgfeat.viz._inputs", "ecgfeat.viz.legacy"}})
        assert not loaded, loaded
    """)


@needs_matplotlib
def test_ecgfeat_viz_loads_the_plots_lazily():
    run_python("""
        import sys
        import ecgfeat.viz
        assert "matplotlib" not in sys.modules
        loaded = ecgfeat.viz.plot_record
        assert "matplotlib" in sys.modules and "matplotlib.pyplot" not in sys.modules
        from ecgfeat.viz import errors, plots
        assert loaded is plots.plot_record
        for name in ecgfeat.viz.__all__:
            source = errors if name == "VisualizationInputError" else plots
            assert getattr(ecgfeat.viz, name) is getattr(source, name), name
    """)


def test_ecgfeat_viz_without_the_extra_explains_how_to_install():
    run_python(f"""
        import sys
        sys.modules["matplotlib"] = None  # simulate the viz extra not being installed
        import ecgfeat, ecgfeat.viz  # still importable
        from ecgfeat.viz import VisualizationInputError  # needs no Matplotlib
        for access in (lambda: ecgfeat.viz.plot_beat,
                       lambda: __import__("ecgfeat.viz", fromlist=["plot_record"]).plot_record):
            try:
                access()
            except ImportError as exc:
                assert str(exc) == {INSTALL_HINT!r}, str(exc)
            else:
                raise AssertionError("expected ImportError")
        try:
            from ecgfeat.viz import plot_record
        except ImportError as exc:
            assert str(exc) == {INSTALL_HINT!r}, str(exc)
        else:
            raise AssertionError("expected ImportError")
    """)


def test_legacy_alias_warning_is_attributed_to_the_importer():
    result = run_python("import warnings; warnings.simplefilter('always'); import ecgfeat.visualize")
    # Other libraries may print their own deprecations (e.g. pyparsing under Matplotlib 3.7).
    ours = [line for line in result.stderr.splitlines() if "ecgfeat.visualize is deprecated" in line]
    assert ours and ours[0].startswith("<string>:1: DeprecationWarning: ecgfeat.visualize is deprecated"), result.stderr
    assert "ecgfeat.viz.legacy" in result.stderr
    assert "no earlier than ecg-records 0.3.0" in result.stderr


def test_legacy_alias_without_matplotlib_imports_and_explains_on_call():
    # The legacy module has always imported without Matplotlib and failed on use.
    run_python("""
        import sys, warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        sys.modules["matplotlib"] = None
        import ecgfeat.visualize as legacy
        try:
            legacy.plot_quality_summary(None, show=False)
        except ImportError as exc:
            assert "pip install" in str(exc), str(exc)
        else:
            raise AssertionError("expected ImportError")
    """)


def test_top_level_legacy_plot_exports_still_resolve():
    run_python("""
        import warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        import ecgfeat
        for name in ("plot_beat", "plot_beat_all_leads", "plot_rep_beat", "plot_quality_summary"):
            assert callable(getattr(ecgfeat, name)), name
    """)
