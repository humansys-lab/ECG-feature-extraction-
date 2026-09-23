"""Deterministic golden-corpus dataset adapters.

Each adapter turns one dataset record into the exact channel-major float64 mV
array handed to the extractor.  Windows, channel slots and unit conversions are
fixed here and recorded in every manifest case, so a changed adapter is visible
as a changed signal hash rather than as an unexplained measurement drift.

Adapters never consult annotations except where documented (QTDB window start),
and never interpolate, pad or relabel beyond the documented slot mapping.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

STANDARD_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
WINDOW_SECONDS = 10.0
# Two-channel datasets feed the standard 12-lead contract through the same
# sparse convention as scripts/evaluate_ecgfeat_external.py: channel 0 -> II,
# channel 1 -> V2, remaining slots zero.  No anatomical claim is made.
SPARSE_SLOTS = (1, 7)

DATASET_ROOTS = {
    "ludb": "lobachevsky-university-electrocardiography-database-1.0.1",
    "qtdb": "qtdb",
    "edb": "edb",
    "ptbxl": "ptb-xl",
    "but": "external_ecg/but-pdb",
    "nstdb": "external_ecg/nstdb",
    "gudb": "external_ecg/gudb/berndporr-ECG-GUDB-5d05f88/docs/experiment_data",
}

_LEAD_ALIASES = {"i": "I", "ii": "II", "iii": "III", "avr": "aVR", "avl": "aVL", "avf": "aVF",
                 **{f"v{i}": f"V{i}" for i in range(1, 7)}}


@dataclass(frozen=True)
class LoadedCase:
    """The exact extractor input plus the identities needed to freeze it."""

    signal: np.ndarray
    fs: float
    lead_names: tuple[str, ...]
    source_files: tuple[Path, ...]
    window: tuple[int, int]
    adapter: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signal_sha256(signal: np.ndarray) -> str:
    """SHA-256 of C-contiguous little-endian float64 bytes (document 05)."""

    canonical = np.ascontiguousarray(signal, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _wfdb():
    import wfdb

    return wfdb


def _wfdb_files(base: Path) -> tuple[Path, ...]:
    header = base.with_suffix(".hea")
    files = [header]
    for line in header.read_text().splitlines()[1:]:
        token = line.split()[0] if line.strip() and not line.startswith("#") else None
        if token:
            candidate = base.parent / token
            if candidate not in files:
                files.append(candidate)
    return tuple(files)


def _window(length: int, fs: float, start_s: float) -> tuple[int, int]:
    size = int(round(WINDOW_SECONDS * fs))
    start = int(round(start_s * fs))
    if start + size > length:
        start = max(0, length - size)
    if start + size > length:
        raise ValueError(f"record shorter than {WINDOW_SECONDS} s window")
    return start, start + size


def _sparse(values: np.ndarray) -> np.ndarray:
    output = np.zeros((12, values.shape[1]), dtype=np.float64)
    for row, slot in zip(values, SPARSE_SLOTS):
        output[slot] = row
    return output


def _named(names: list[str]) -> tuple[str, ...]:
    cleaned = [str(name).strip() for name in names]
    if len(set(cleaned)) != len(cleaned) or any(not name for name in cleaned):
        return tuple(f"channel_{i}" for i in range(len(names)))
    return tuple(cleaned)


def _read_wfdb(base: Path, window: tuple[int, int]):
    record = _wfdb().rdrecord(str(base), sampfrom=window[0], sampto=window[1], physical=True)
    values = np.asarray(record.p_signal, dtype=np.float64).T.copy()
    if not np.isfinite(values).all():
        raise ValueError("nonfinite physical samples")
    return record, values


def _header_length(base: Path) -> tuple[float, int]:
    fields = base.with_suffix(".hea").read_text().split("\n", 1)[0].split()
    fs = float(fields[2].split("/")[0])
    return fs, int(fields[3])


# --------------------------------------------------------------------------- #
# discovery (sorted, eligibility-filtered record ids)
# --------------------------------------------------------------------------- #

def discover(dataset: str, data_root: Path) -> list[str]:
    root = data_root / DATASET_ROOTS[dataset]
    if dataset == "ludb":
        return sorted((line.split("/")[-1] for line in (root / "RECORDS").read_text().split()), key=int)
    if dataset in {"qtdb", "edb"}:
        return sorted(path.stem for path in root.glob("*.hea"))
    if dataset == "ptbxl":
        return sorted(f"{path.parent.name}/{path.stem}" for path in root.glob("*/*_hr.hea"))
    if dataset == "but":
        return sorted(path.stem for path in root.glob("[0-9][0-9].hea"))
    if dataset == "nstdb":
        # Prebuilt noise-stress ECGs only; bw/em/ma are noise-only records.
        return sorted(path.stem for path in root.glob("11[89]e*.hea"))
    if dataset == "gudb":
        return sorted(f"{path.parent.parent.name}/{path.parent.name}"
                      for path in root.glob("subject_*/*/ECG.tsv")
                      if (path.parent / "annotation_cables.tsv").exists())
    raise KeyError(dataset)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def _load_12lead(base: Path, adapter: str, leads: tuple[str, ...] | None) -> LoadedCase:
    fs, length = _header_length(base)
    window = _window(length, fs, 0.0)
    record, values = _read_wfdb(base, window)
    names = [_LEAD_ALIASES.get(name.lower(), name) for name in record.sig_name]
    if sorted(names) != sorted(STANDARD_12):
        raise ValueError(f"not a standard 12-lead record: {record.sig_name}")
    if any(unit.lower() != "mv" for unit in record.units):
        raise ValueError(f"unexpected units {record.units}")
    order = leads or STANDARD_12
    signal = np.stack([values[names.index(name)] for name in order])
    return LoadedCase(signal, float(record.fs), tuple(order), _wfdb_files(base), window, adapter)


def _qtdb_start_seconds(base: Path, fs: float) -> float:
    q1c = base.with_suffix(".q1c")
    if not q1c.exists():
        return 0.0
    annotation = _wfdb().rdann(str(base), "q1c")
    if len(annotation.sample) == 0:
        return 0.0
    return max(0.0, float(annotation.sample[0]) / fs - 2.0)


def _load_two_channel(dataset: str, base: Path, mode: str, start_s: Callable[[float], float]) -> LoadedCase:
    fs, length = _header_length(base)
    window = _window(length, fs, start_s(fs))
    record, values = _read_wfdb(base, window)
    if record.n_sig != 2:
        raise ValueError(f"expected two channels, found {record.n_sig}")
    files = _wfdb_files(base)
    if dataset == "qtdb" and base.with_suffix(".q1c").exists():
        files = files + (base.with_suffix(".q1c"),)
    if mode == "standard":
        return LoadedCase(_sparse(values), float(record.fs), STANDARD_12, files, window,
                          f"{dataset}-sparse-II-V2-10s-v1")
    return LoadedCase(values, float(record.fs), _named(list(record.sig_name)), files, window,
                      f"{dataset}-limited-header-names-10s-v1")


GUDB_FS = 250.0
GUDB_START_S = 30.0


def _load_gudb(root: Path, record_id: str, mode: str) -> LoadedCase:
    directory = root / record_id
    raw = np.loadtxt(directory / "ECG.tsv")
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError("GUDB expects six finite raw columns")
    window = _window(raw.shape[0], GUDB_FS, GUDB_START_S)
    # Cables stream (columns 1 and 2), volts -> millivolts.
    values = raw[window[0]:window[1], [1, 2]].T * 1000.0
    files = (directory / "ECG.tsv",)
    if mode == "standard":
        return LoadedCase(_sparse(values), GUDB_FS, STANDARD_12, files, window, "gudb-cables-sparse-II-V2-10s-v1")
    return LoadedCase(values, GUDB_FS, ("cables_1", "cables_2"), files, window, "gudb-cables-limited-10s-v1")


LIMITED_LUDB_LEADS = ("I", "II", "V2")


def load_case(dataset: str, record_id: str, data_root: Path, *, mode: str) -> LoadedCase:
    """Load one case.  ``mode`` is ``standard`` or ``limited``."""

    if mode not in {"standard", "limited"}:
        raise ValueError(mode)
    root = data_root / DATASET_ROOTS[dataset]
    if dataset == "ludb":
        base = root / "data" / record_id
        if mode == "limited":
            return _load_12lead(base, "ludb-limited-I-II-V2-10s-v1", LIMITED_LUDB_LEADS)
        return _load_12lead(base, "ludb-12lead-10s-v1", None)
    if dataset == "ptbxl":
        if mode == "limited":
            raise ValueError("no limited PTB-XL adapter")
        return _load_12lead(root / record_id, "ptbxl-12lead-hr-10s-v1", None)
    if dataset == "qtdb":
        base = root / record_id
        return _load_two_channel(dataset, base, mode, lambda fs: _qtdb_start_seconds(base, fs))
    if dataset == "edb":
        return _load_two_channel(dataset, root / record_id, mode, lambda fs: 300.0)
    if dataset == "but":
        return _load_two_channel(dataset, root / record_id, mode, lambda fs: 0.0)
    if dataset == "nstdb":
        # t=300 s is where the prebuilt electrode-motion noise begins.
        return _load_two_channel(dataset, root / record_id, mode, lambda fs: 300.0)
    if dataset == "gudb":
        return _load_gudb(root, record_id, mode)
    raise KeyError(dataset)
