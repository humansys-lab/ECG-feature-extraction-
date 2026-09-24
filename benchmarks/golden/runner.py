"""Run one golden case through a named extraction surface.

Surfaces are resolved from whatever ``ecgfeat`` is importable, so the same
runner freezes the pre-migration tree and checks every migrated tree.  The
legacy surface deliberately goes through the legacy public imports
(``ecgfeat.api.ECGFeatureExtractor`` and ``ecgfeat.export``): those forwarding
paths are exactly what old consumers use and what the parity gate protects.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .datasets import LoadedCase, load_case, sha256_file, signal_sha256

LEGACY_PROFILES = ("summary", "audit", "debug", "full")
RECORD_PROFILES = ("summary", "all", "debug")

# The only normalization permitted in legacy-bytes mode: wall-clock timestamps
# written by the interpretation layer.  Everything else is compared verbatim.
LEGACY_NORMALIZED_POINTERS = (
    "/clinical_interpretation/generated_at",
    "/metadata/clinical_interpretation/generated_at",
)
NORMALIZED_VALUE = "<normalized:generated_at>"

# Resolved extractor attributes that make up the frozen configuration.
EXTRACTOR_ATTRIBUTES = (
    "fs_internal", "mains_freq", "lp_hz", "enable_pacing", "enable_lead_reversal",
    "compute_grouping", "enable_hybrid_r_localization", "enable_hybrid_wave_localization",
    "enable_hybrid_st_measurement", "enable_t_wave_refinement", "st_amplitude_source",
    "input_mode",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def escape_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def iter_leaves(value: Any, pointer: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            yield pointer, {}
        for key, child in value.items():
            yield from iter_leaves(child, f"{pointer}/{escape_token(str(key))}")
    elif isinstance(value, list):
        if not value:
            yield pointer, []
        for index, child in enumerate(value):
            yield from iter_leaves(child, f"{pointer}/{index}")
    else:
        yield pointer, value


def leaf_digest(value: Any) -> str:
    """Digest of parsed leaves, independent of object key order."""

    lines = sorted(f"{pointer}\t{json.dumps(leaf, ensure_ascii=False, allow_nan=False)}"
                   for pointer, leaf in iter_leaves(value))
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def _reject(value: Any) -> Any:
    raise TypeError(f"non-JSON value in legacy export: {type(value).__name__}")


def legacy_bytes(payload: Any) -> bytes:
    """Exact legacy rendering: exporter key order, repr floats, no sorting."""

    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
                      default=_reject).encode("utf-8")


def normalize_legacy(payload: Any) -> Any:
    for pointer in LEGACY_NORMALIZED_POINTERS:
        node = payload
        tokens = pointer.split("/")[1:]
        for token in tokens[:-1]:
            node = node.get(token) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and tokens[-1] in node:
            node[tokens[-1]] = NORMALIZED_VALUE
    return payload


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

def refinement_from(spec: dict[str, Any]):
    from ecgfeat import RefinementConfig

    return RefinementConfig(**spec.get("refinement", {}))


def build_extractor(spec: dict[str, Any]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        from ecgfeat.api import ECGFeatureExtractor

        return ECGFeatureExtractor(
            fs_internal=spec["fs_internal"],
            mains_freq=spec["mains_freq"],
            input_mode=spec["input_mode"],
            st_amplitude_source=spec["st_amplitude_source"],
            refinement=refinement_from(spec),
        )


def resolved_config(spec: dict[str, Any]) -> dict[str, Any]:
    """Fully resolved extractor configuration, including every default."""

    extractor = build_extractor(spec)
    resolved = {name: getattr(extractor, name) for name in EXTRACTOR_ATTRIBUTES}
    resolved["refinement"] = asdict(extractor.refinement)
    return resolved


def record_config(spec: dict[str, Any]):
    from ecgfeat.config import ECGConfig

    return ECGConfig(
        input_mode="limited" if spec["input_mode"] == "limited" else "standard_12",
        fs_internal=spec["fs_internal"],
        mains_frequency_hz=spec["mains_freq"],
        refinement=refinement_from(spec),
        st_amplitude_source=spec["st_amplitude_source"],
    )


# --------------------------------------------------------------------------- #
# surfaces
# --------------------------------------------------------------------------- #

def run_legacy(loaded: LoadedCase, spec: dict[str, Any]) -> tuple[dict[str, bytes], Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        from ecgfeat.export import prepare_json_export, to_dict

        extractor = build_extractor(spec)
        kwargs = {"amplitude_unit": "mV"}
        if spec["input_mode"] == "limited":
            kwargs["lead_names"] = list(loaded.lead_names)
        features = extractor.extract(np.array(loaded.signal, copy=True), loaded.fs, **kwargs)
        outputs: dict[str, bytes] = {}
        debug_payload = None
        for profile in LEGACY_PROFILES:
            if profile == "full":
                payload = prepare_json_export(to_dict(features), profile="debug", round_ndigits=None)
            else:
                payload = prepare_json_export(to_dict(features, profile=profile), profile=profile,
                                              round_ndigits=None)
            payload = normalize_legacy(payload)
            outputs[profile] = legacy_bytes(payload)
            if profile == "debug":
                debug_payload = payload
    return outputs, debug_payload


def run_interpretation(loaded: LoadedCase, spec: dict[str, Any]) -> bytes:
    """Interpretation document from the legacy measurement state (ecginterpret)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        from ecginterpret import interpret_features

        extractor = build_extractor(spec)
        kwargs = {"amplitude_unit": "mV"}
        if spec["input_mode"] == "limited":
            kwargs["lead_names"] = list(loaded.lead_names)
        features = extractor.extract(np.array(loaded.signal, copy=True), loaded.fs, **kwargs)
        return canonical_json_bytes(interpret_features(features).as_dict())


def run_record(loaded: LoadedCase, spec: dict[str, Any]) -> dict[str, bytes]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        from ecgfeat import dumps_record, ecg_emit, ecg_measure, ecg_prepare

        prepared = ecg_prepare(
            np.array(loaded.signal, copy=True),
            sampling_rate=loaded.fs,
            lead_names=list(loaded.lead_names),
            amplitude_unit="mV",
            input_mode="limited" if spec["input_mode"] == "limited" else "standard_12",
        )
        measured = ecg_measure(prepared, config=record_config(spec))
        return {profile: dumps_record(ecg_emit(measured, profile=profile)) for profile in RECORD_PROFILES}


# --------------------------------------------------------------------------- #
# case identity and execution
# --------------------------------------------------------------------------- #

def input_identity(loaded: LoadedCase, data_root: Path) -> dict[str, Any]:
    meta = {
        "shape": list(loaded.signal.shape),
        "fs": loaded.fs,
        "units": "mV",
        "lead_names": list(loaded.lead_names),
        "dtype": "<f8",
    }
    return {
        "adapter": loaded.adapter,
        "window": list(loaded.window),
        "fs": loaded.fs,
        "units": "mV",
        "lead_names": list(loaded.lead_names),
        "shape": list(loaded.signal.shape),
        "source_files": [
            {"path": str(path.relative_to(data_root)), "sha256": sha256_file(path)}
            for path in loaded.source_files
        ],
        "signal_sha256": signal_sha256(loaded.signal),
        "signal_meta_sha256": sha256_bytes(canonical_json_bytes(meta)),
    }


def execute_case(case: dict[str, Any], config_spec: dict[str, Any], data_root: Path,
                 surfaces: tuple[str, ...], payload_dir: Path | None = None) -> dict[str, Any]:
    """Load, verify input identity, run the requested surfaces, and hash outputs."""

    mode = "limited" if config_spec["input_mode"] == "limited" else "standard"
    loaded = load_case(case["dataset"], case["record"], data_root, mode=mode)
    identity = input_identity(loaded, data_root)
    result: dict[str, Any] = {"case_id": case["case_id"], "input": identity, "outputs": {}}
    rendered = None
    if "legacy" in surfaces:
        try:
            rendered, debug_payload = run_legacy(loaded, config_spec)
        except Exception as exc:  # a frozen failure is behavior, compared exactly
            rendered = None
            result["outputs"]["legacy"] = {"error": f"{type(exc).__name__}: {exc}"}
    if rendered is not None:
        result["outputs"]["legacy"] = {
            profile: {"sha256": sha256_bytes(data), "bytes": len(data),
                      "leaf_digest": leaf_digest(json.loads(data))}
            for profile, data in rendered.items()
        }
        if payload_dir is not None:
            target = payload_dir / "legacy" / f"{case_file_stem(case['case_id'])}.debug.json.gz"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.compress(rendered["debug"], mtime=0))
    rendered_records = None
    if "record" in surfaces:
        try:
            rendered_records = run_record(loaded, config_spec)
        except Exception as exc:
            result["outputs"]["record"] = {"error": f"{type(exc).__name__}: {exc}"}
    if rendered_records is not None:
        result["outputs"]["record"] = {
            profile: {"sha256": sha256_bytes(data), "bytes": len(data),
                      "leaf_digest": leaf_digest(json.loads(data))}
            for profile, data in rendered_records.items()
        }
        if payload_dir is not None:
            for profile, data in rendered_records.items():
                target = payload_dir / "record" / f"{case_file_stem(case['case_id'])}.{profile}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    if "interpretation" in surfaces:
        try:
            document = run_interpretation(loaded, config_spec)
            result["outputs"]["interpretation"] = {"sha256": sha256_bytes(document), "bytes": len(document)}
            if payload_dir is not None:
                target = payload_dir / "interpretation" / f"{case_file_stem(case['case_id'])}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(document)
        except Exception as exc:
            result["outputs"]["interpretation"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def case_file_stem(case_id: str) -> str:
    return case_id.replace("/", "__")
