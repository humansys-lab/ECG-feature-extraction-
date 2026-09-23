"""Deterministic construction of harness-owned input corpora.

Two of the ten tools cannot be pointed at a raw PhysioNet directory:

* ``batch_extract_ecgfeat.py`` only reads ``<input>/<label>/<source>.pt``
  tensors.  Its production corpus (``all_diseases_pt100``: generated/real
  ECG tensors from the generation project) does not exist on this machine, so
  the harness builds a pinned stand-in from PTB-XL with a fixed, documented
  selection rule.
* ``score_measurement_verifiable.py`` scores an agent run directory
  (``diagnoses/*.json`` + ``features/*_features.json``).  The harness freezes
  the only locally available agent verdicts and lets the tree under test
  regenerate the ``features/`` half, so the gate measures whether ecgfeat's
  measurements still verify the same frozen calls.

The corpora live at a fixed location outside every tree under test, so a
baseline run and a candidate run read byte-identical inputs from the same
absolute path (``batch_extract_ecgfeat.py`` derives record ids from it).
Corpus builders run in the harness interpreter and never import ``ecgfeat``.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from .core import HarnessError, canonical_json, sha256_bytes

CORPUS_FORMAT_VERSION = 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _ptbxl_statements(metadata_dir: Path) -> dict[str, dict[str, str]]:
    return {row[""]: row for row in _read_csv(metadata_dir / "scp_statements.csv") if row.get("")}


def _active_codes(score_map: Mapping[str, float], statements: Mapping[str, Mapping[str, str]]) -> list[str]:
    """PTB-XL active-code policy shared by the repo's tools.

    Diagnostic statements need likelihood >= 50; form/rhythm statements are
    active by presence (evaluate_target_ecgfeat_diagnosis.py / ecgagent.batch).
    """

    active = []
    for code, score in score_map.items():
        statement = statements.get(code, {})
        if statement.get("diagnostic") == "1.0" and float(score) < 50.0:
            continue
        active.append(str(code))
    return active


def select_ptbxl_superclass_records(
    shard_dir: Path,
    metadata_dir: Path,
    *,
    classes: tuple[str, ...],
    per_class: int,
) -> dict[str, list[str]]:
    """First ``per_class`` records (ascending ecg_id) of the shard whose active
    diagnostic statements map to exactly one PTB-XL diagnostic superclass."""

    statements = _ptbxl_statements(metadata_dir)
    by_name: dict[str, dict[str, str]] = {}
    for row in _read_csv(metadata_dir / "ptbxl_database.csv"):
        for field in ("filename_lr", "filename_hr"):
            if row.get(field):
                by_name[Path(row[field]).name] = row
    chosen: dict[str, list[str]] = {label: [] for label in classes}
    for header in sorted(shard_dir.glob("*.hea")):
        row = by_name.get(header.stem)
        if row is None:
            continue
        score_map = ast.literal_eval(row.get("scp_codes") or "{}")
        superclasses = {
            statements[code].get("diagnostic_class")
            for code in _active_codes(score_map, statements)
            if statements.get(code, {}).get("diagnostic") == "1.0"
            and statements[code].get("diagnostic_class")
        }
        if len(superclasses) != 1:
            continue
        label = next(iter(superclasses))
        if label in chosen and len(chosen[label]) < per_class:
            chosen[label].append(header.stem)
        if all(len(items) >= per_class for items in chosen.values()):
            break
    short = {label: len(items) for label, items in chosen.items() if len(items) < per_class}
    if short:
        raise HarnessError(f"PTB-XL shard {shard_dir} lacks records for {short}")
    return chosen


def build_batch_pt_corpus(spec: Mapping[str, Any], data_root: Path, corpus_root: Path) -> dict[str, Any]:
    """Build (or verify) the pinned ``.pt`` corpus for batch_extract_ecgfeat.py.

    Layout: ``<corpus_root>/<variant>/<label>/real.pt`` + ``metadata.pt``.
    ``real.pt`` is a float32 tensor ``[n, 12, n_points]`` in the fixed lead
    order of the WFDB header; ``metadata.pt`` is ``{"dataset", "label"}``.
    Variants: native 500 Hz and a 100 Hz copy made with
    ``scipy.signal.resample_poly(x, 1, 5)`` (the tool's production rate is
    100 Hz; PTB-XL's own 100 Hz files are not present locally).
    """

    import numpy as np
    import torch
    import wfdb
    from scipy.signal import resample_poly

    recipe = {
        "format_version": CORPUS_FORMAT_VERSION,
        "source_shard": spec["source_shard"],
        "metadata_dir": spec["metadata_dir"],
        "classes": list(spec["classes"]),
        "per_class": int(spec["per_class"]),
        "variants": spec["variants"],
        "dtype": "float32",
    }
    recipe_hash = sha256_bytes(canonical_json(recipe))
    build_manifest_path = corpus_root / "corpus_build.json"
    if build_manifest_path.exists():
        existing = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        if existing.get("recipe_sha256") == recipe_hash:
            return existing
        raise HarnessError(
            f"{corpus_root} was built from a different recipe; refusing to overwrite a pinned corpus"
        )

    shard_dir = data_root / spec["source_shard"]
    metadata_dir = data_root / spec["metadata_dir"]
    selection = select_ptbxl_superclass_records(
        shard_dir, metadata_dir, classes=tuple(spec["classes"]), per_class=int(spec["per_class"])
    )
    tmp_root = corpus_root.with_name(corpus_root.name + ".building")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    arrays: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {}
    for label, records in selection.items():
        signals = []
        for record in records:
            rec = wfdb.rdrecord(str(shard_dir / record))
            if [name.lower() for name in rec.sig_name] != [
                "i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"
            ]:
                raise HarnessError(f"{record}: unexpected lead order {rec.sig_name}")
            if int(rec.fs) != 500:
                raise HarnessError(f"{record}: expected 500 Hz, got {rec.fs}")
            signals.append(np.asarray(rec.p_signal, dtype=np.float64).T)
            for suffix in (".hea", ".dat"):
                path = shard_dir / f"{record}{suffix}"
                source_files[f"{spec['source_shard']}/{record}{suffix}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        native = np.stack(signals)  # [n, 12, 5000]
        arrays[label] = {"records": records, "native": native}

    variant_rows = []
    for variant in spec["variants"]:
        name = variant["name"]
        fs = int(variant["sampling_rate"])
        for label, payload in arrays.items():
            native = payload["native"]
            if fs == 500:
                data = native.astype(np.float32)
            elif 500 % fs == 0:
                data = resample_poly(native, 1, 500 // fs, axis=-1).astype(np.float32)
            else:
                raise HarnessError(f"unsupported variant rate {fs}")
            out_dir = tmp_root / name / label.lower()
            out_dir.mkdir(parents=True, exist_ok=True)
            data = np.ascontiguousarray(data)
            torch.save(torch.from_numpy(data), out_dir / "real.pt")
            torch.save({"dataset": "ptb-xl", "label": label}, out_dir / "metadata.pt")
            variant_rows.append({
                "variant": name,
                "sampling_rate": fs,
                "label": label,
                "records": payload["records"],
                "shape": list(data.shape),
                "array_sha256_float32_le": hashlib.sha256(
                    data.astype("<f4", copy=False).tobytes(order="C")
                ).hexdigest(),
            })
    build = {
        "recipe": recipe,
        "recipe_sha256": recipe_hash,
        "selection_rule": (
            "first per_class records (ascending ecg_id) in the shard whose active "
            "diagnostic statements (likelihood>=50) map to exactly one PTB-XL "
            "diagnostic superclass"
        ),
        "selection": selection,
        "variants": variant_rows,
        "source_file_sha256": dict(sorted(source_files.items())),
        "library_versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "wfdb": wfdb.__version__ if hasattr(wfdb, "__version__") else None,
            "scipy": __import__("scipy").__version__,
        },
    }
    (tmp_root / "corpus_build.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
    if corpus_root.exists():
        shutil.rmtree(corpus_root)
    tmp_root.replace(corpus_root)
    return build


def build_smv_corpus(spec: Mapping[str, Any], repo_data_root: Path, corpus_root: Path,
                     workspace_root: Path) -> dict[str, Any]:
    """Freeze agent verdicts + stage their WFDB records for score_measurement_verifiable.py.

    Layout::

        <corpus_root>/wfdb/<record>.{hea,dat}          (copied source records)
        <corpus_root>/diagnoses/<record>.json          (frozen agent verdicts)
        <corpus_root>/record_results.csv               (PTB-XL reference codes)
    """

    build_manifest_path = corpus_root / "corpus_build.json"
    recipe = {"format_version": CORPUS_FORMAT_VERSION, **{k: spec[k] for k in (
        "frozen_verdicts", "metadata_dir")}}
    recipe_hash = sha256_bytes(canonical_json(recipe))
    if build_manifest_path.exists():
        existing = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        if existing.get("recipe_sha256") == recipe_hash:
            return existing
        raise HarnessError(f"{corpus_root} was built from a different recipe")

    tmp_root = corpus_root.with_name(corpus_root.name + ".building")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "wfdb").mkdir(parents=True)
    (tmp_root / "diagnoses").mkdir(parents=True)
    statements = _ptbxl_statements(repo_data_root / spec["metadata_dir"])
    by_name: dict[str, dict[str, str]] = {}
    for row in _read_csv(repo_data_root / spec["metadata_dir"] / "ptbxl_database.csv"):
        for field in ("filename_lr", "filename_hr"):
            if row.get(field):
                by_name[Path(row[field]).name] = row
    frozen_rows = []
    result_rows = []
    for item in spec["frozen_verdicts"]:
        record = item["record"]
        verdict_src = workspace_root / item["diagnosis_json"]
        if not verdict_src.is_file():
            raise HarnessError(f"frozen verdict source missing: {verdict_src}")
        verdict_bytes = verdict_src.read_bytes()
        verdict_hash = hashlib.sha256(verdict_bytes).hexdigest()
        if item.get("sha256") and item["sha256"] != verdict_hash:
            raise HarnessError(f"{verdict_src} changed since it was pinned")
        (tmp_root / "diagnoses" / f"{record}.json").write_bytes(verdict_bytes)
        for suffix in (".hea", ".dat"):
            shutil.copyfile(repo_data_root / f"{item['wfdb_record']}{suffix}",
                            tmp_root / "wfdb" / f"{record}{suffix}")
        payload = json.loads(verdict_bytes.decode("utf-8"))
        codes = [row.get("code") for row in (payload.get("verdict") or {}).get("diagnoses") or []]
        source = by_name.get(record)
        if source is None:
            raise HarnessError(f"{record} not in PTB-XL metadata")
        score_map = ast.literal_eval(source.get("scp_codes") or "{}")
        result_rows.append({"record": record,
                            "reference_codes": "|".join(sorted(_active_codes(score_map, statements)))})
        frozen_rows.append({"record": record, "source": item["diagnosis_json"],
                            "sha256": verdict_hash, "confirmed_codes": codes})
    with (tmp_root / "record_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record", "reference_codes"])
        writer.writeheader()
        writer.writerows(result_rows)
    build = {"recipe": recipe, "recipe_sha256": recipe_hash, "frozen_verdicts": frozen_rows,
             "record_results_rule": "PTB-XL active codes (diagnostic likelihood>=50; form/rhythm by presence)"}
    (tmp_root / "corpus_build.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
    if corpus_root.exists():
        shutil.rmtree(corpus_root)
    tmp_root.replace(corpus_root)
    return build
