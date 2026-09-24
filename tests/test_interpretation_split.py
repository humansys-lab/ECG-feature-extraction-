"""Phase 5 integration: core and ecginterpret stay consistent where they overlap."""

import dataclasses
import subprocess
import sys
import textwrap

import pytest

from ecginterpret.clinical_rules import rhythm as rule_rhythm
from ecginterpret.clinical_rules.config import DEFAULT_DIAGNOSTIC_CONFIG
from feature_extraction.ecgfeat._engine.foundation.thresholds import ENGINE_THRESHOLDS
from feature_extraction.ecgfeat._engine.measurement import rhythm as engine_rhythm


def test_engine_thresholds_equal_the_rule_engine_defaults():
    assert dataclasses.asdict(ENGINE_THRESHOLDS.quality) == dataclasses.asdict(DEFAULT_DIAGNOSTIC_CONFIG.quality)
    for group in ("rhythm", "morphology"):
        mine = dataclasses.asdict(getattr(ENGINE_THRESHOLDS, group))
        theirs = dataclasses.asdict(getattr(DEFAULT_DIAGNOSTIC_CONFIG, group))
        assert mine == {key: theirs[key] for key in mine}, group


@pytest.mark.parametrize("summary", [
    {}, {"atrial_faster_than_ventricular": True}, {"complete_av_block": 1}, None,
    {"organized_p_ratio": 0.1}, {"organized_p_ratio": 0.5}, {"organized_p_ratio": float("nan")},
    {"organized_p_ratio": "x"},
])
def test_availability_predicates_match_the_rule_engine(summary):
    assert engine_rhythm.av_dissociation_corroborated(summary) == rule_rhythm.av_dissociation_corroborated(summary)
    for threshold in (0.2, 0.35, 0.6):
        assert engine_rhythm.disorganized_atrial_activity(summary, threshold=threshold) == \
            rule_rhythm.disorganized_atrial_activity(summary, threshold=threshold)


BLOCK_ECGINTERPRET = textwrap.dedent('''
    import sys, importlib.abc
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "ecginterpret" or name.startswith("ecginterpret."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    sys.meta_path.insert(0, Block())
''')


def _run(code):
    return subprocess.run([sys.executable, "-W", "ignore", "-c", BLOCK_ECGINTERPRET + textwrap.dedent(code)],
                          capture_output=True, text=True)


def test_records_are_extracted_without_the_interpretation_distribution():
    result = _run('''
        import numpy as np
        from ecgfeat import dumps_record, ecg_record
        signal = np.load("tests/fixtures/golden/reference_10s_12lead/signal.npz")["signal"]
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        data = dumps_record(ecg_record(signal, sampling_rate=500, lead_names=leads))
        assert data == open("tests/fixtures/golden/reference_10s_12lead/record.summary.json", "rb").read()
        assert not any(name.startswith("ecginterpret") for name in sys.modules)
    ''')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("code", [
    "import ecgfeat.interpret",
    "import ecgfeat.clinical_rules.engine",
    "from ecgfeat.export import to_dict; to_dict(object())",
    "import numpy as np; from ecgfeat.api import ECGFeatureExtractor; ECGFeatureExtractor().extract(np.zeros((12, 1000)), 500)",
])
def test_legacy_interpretation_entry_points_explain_how_to_install(code):
    result = _run(code)
    assert result.returncode != 0
    assert 'pip install "ecg-records[interpret]"' in result.stderr


def test_core_package_code_never_imports_ecginterpret_outside_compat():
    import ast
    from pathlib import Path

    root = Path("feature_extraction/ecgfeat")
    offenders = []
    for path in root.rglob("*.py"):
        if "compat" in path.parts or "__pycache__" in path.parts or path.name == "_moved.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            if any(name.split(".")[0] == "ecginterpret" for name in names):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []
