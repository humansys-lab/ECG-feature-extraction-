"""Shared fixtures: the pinned 10 s, 12-lead reference record and its signal."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = Path(
    os.environ.get("ECGRECORDS_VIZ_REFERENCE_DIR", REPO_ROOT / "tests" / "fixtures" / "golden" / "reference_10s_12lead")
)


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    assert (REFERENCE_DIR / "record.all.json").is_file(), f"reference fixture missing: {REFERENCE_DIR}"
    return REFERENCE_DIR


@pytest.fixture(scope="session")
def _document(reference_dir: Path) -> dict:
    return json.loads((reference_dir / "record.all.json").read_text(encoding="utf-8"))


@pytest.fixture()
def document(_document: dict) -> dict:
    """A fresh, mutable copy of the reference record mapping."""

    return copy.deepcopy(_document)


@pytest.fixture(scope="session")
def record(reference_dir: Path):
    from ecgfeat.record import load_record

    return load_record(reference_dir / "record.all.json")


@pytest.fixture(scope="session")
def _signal(reference_dir: Path) -> np.ndarray:
    with np.load(reference_dir / "signal.npz") as archive:
        return archive["signal"]


@pytest.fixture()
def signal(_signal: np.ndarray) -> np.ndarray:
    return _signal.copy()


@pytest.fixture(scope="session")
def sidecar_files(tmp_path_factory, record) -> Path:
    """The reference record serialized with every dense matrix in an NPZ sidecar."""

    from ecgfeat.record import serialize_record

    directory = tmp_path_factory.mktemp("sidecar")
    serialized = serialize_record(record, sidecar_uri="dense.npz")
    (directory / "record.json").write_bytes(serialized.json_bytes)
    for name, data in serialized.sidecars.items():
        (directory / name).write_bytes(data)
    return directory
