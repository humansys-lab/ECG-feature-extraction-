"""Phase 1 relocation contract: old engine paths alias the moved modules and warn."""

import importlib
import subprocess
import sys
import warnings

import pytest

import ecgfeat
from ecgfeat import _moved

MOVED = {
    "_kalman": "_engine.foundation.kalman",
    "numeric": "_engine.foundation.numeric",
    "preprocess": "_engine.preprocess",
    "acquisition_qc": "_engine.quality.acquisition",
    "quality": "_engine.quality.signal",
    "qrs": "_engine.detection.qrs",
    "adaptive_qrs": "_engine.detection.adaptive_qrs",
    "grouping": "_engine.beats.grouping",
    "representative": "_engine.beats.representative",
    "family_representative": "_engine.beats.families",
    "delineate": "_engine.delineation.core",
    "classical_candidates": "_engine.delineation.candidates",
    "boundary_refinement": "_engine.delineation.refinement",
    "wave_localization": "_engine.delineation.wave_localization",
    "r_localization": "_engine.delineation.r_localization",
    "repolarization": "_engine.delineation.repolarization",
    "t_wave_refinement": "_engine.delineation.t_refinement",
    "atrial": "_engine.atrial.core",
    "p_wave_engine": "_engine.atrial.p_wave",
    "p_morphology": "_engine.atrial.p_morphology",
    "atrial_validation": "_engine.atrial.validation",
    "features": "_engine.measurement.features",
    "dispersion": "_engine.measurement.dispersion",
    "measurement_paths": "_engine.measurement.paths",
    "st_baseline": "_engine.measurement.st_baseline",
    "st_localization": "_engine.measurement.st_localization",
    "u_wave": "_engine.measurement.u_wave",
    "vector_axis": "_engine.measurement.vector_axis",
    "calibration": "_engine.measurement.calibration",
    "twelve_sl": "_engine.measurement.profiles.twelve_sl",
    "glasgow_measurements": "_engine.measurement.profiles.glasgow",
    "rhythm_rules": "_engine.measurement.rhythm",
}


@pytest.mark.parametrize("old,new", sorted(MOVED.items()))
def test_old_path_is_same_module_and_warns(old, new):
    target = importlib.import_module(f"ecgfeat.{new}")
    sys.modules.pop(f"ecgfeat.{old}", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = importlib.import_module(f"ecgfeat.{old}")
    assert alias is target
    messages = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 1
    text = str(messages[0].message)
    assert f"ecgfeat.{new}" in text
    assert f"no earlier than ecg-records {_moved.REMOVAL_RELEASE}" in text


def test_warning_is_attributed_to_the_importer():
    code = "import warnings; warnings.simplefilter('always'); import ecgfeat.delineate"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stderr.startswith("<string>:1: DeprecationWarning: ecgfeat.delineate moved")


def test_package_code_never_imports_old_paths():
    # Interpretation modules left the package (Phase 5); their old paths are
    # deprecated aliases covered by tests/test_interpretation_split.py.
    code = ("import ecgfeat, ecgfeat.api, ecgfeat.export, ecgfeat.pipeline, ecgfeat.record, ecgfeat.cli, "
            "ecgfeat.compat.api_v0, ecgfeat.compat.export_v0, ecgfeat.compat.interpretation_hooks, "
            "ecginterpret, ecginterpret.clinical_rules.engine, ecginterpret.glasgow_rules.engine; "
            "from ecgfeat import PWaveConfig")
    subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", code], check=True)


def test_no_implementation_left_at_old_paths():
    from pathlib import Path

    root = Path(ecgfeat.__file__).parent
    for old in MOVED:
        text = (root / f"{old}.py").read_text()
        assert "alias_module(__name__," in text and len(text.splitlines()) < 15, old
