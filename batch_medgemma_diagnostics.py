#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from medgemma_ecg_core import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_MAX_LEN,
    DIAGNOSTIC_GATE_ENFORCEMENT_VERSION,
    MEDGEMMA_REASONING_PROTOCOL_VERSION,
    build_context_summary_from_paths,
    build_prompt,
    canonical_label_for_dir,
    diagnostic_gate_policy,
    parse_medgemma_output,
    run_layered_diagnosis,
)
from medgemma_runtime import build_generate_text_fn as _build_generate_text_fn


@dataclass(frozen=True)
class SampleJob:
    disease_dir_label: str
    split_name: str
    sample_id: str
    features_path: Path
    report_path: Path | None
    local_output_dir: Path
    mirror_output_dir: Path


MODEL_PATH = "./medgemma-27b"
MODEL_MAX_LEN = DEFAULT_MODEL_MAX_LEN
MODEL_MAX_OUTPUT_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS
EXECUTION_SIGNATURE_SCHEMA_VERSION = "medgemma_batch_execution.v1"


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_artifact_identity(model_path: str) -> dict[str, Any]:
    """Fingerprint a local model without hashing multi-gigabyte weight files."""
    requested = str(model_path)
    path = Path(model_path).expanduser()
    resolved = str(path.resolve(strict=False))
    if not path.exists():
        return {
            "requested": requested,
            "resolved": resolved,
            "kind": "model_identifier_or_missing_path",
            "inventory_sha256": None,
        }

    if path.is_file():
        stat = path.stat()
        inventory = {
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _file_sha256(path) if stat.st_size <= 16 * 1024 * 1024 else None,
        }
        return {
            "requested": requested,
            "resolved": resolved,
            "kind": "file",
            "inventory_sha256": _canonical_sha256(inventory),
        }

    inventory: list[dict[str, Any]] = []
    content_hash_names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        stat = item.stat()
        entry: dict[str, Any] = {
            "path": item.relative_to(path).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if item.is_symlink():
            entry["symlink_target"] = os.readlink(item)
        if (
            item.name in content_hash_names
            or item.name.endswith(".index.json")
        ) and stat.st_size <= 16 * 1024 * 1024:
            entry["sha256"] = _file_sha256(item)
        inventory.append(entry)
    return {
        "requested": requested,
        "resolved": resolved,
        "kind": "directory",
        "inventory_sha256": _canonical_sha256(inventory),
        "file_count": len(inventory),
    }


def _runtime_identity() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "vllm")
        },
        "environment": {
            "ECG_GEMMA_QUANTIZATION": os.getenv("ECG_GEMMA_QUANTIZATION"),
            "ECG_GEMMA_TP_SIZE": os.getenv("ECG_GEMMA_TP_SIZE"),
        },
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        runtime["cuda"] = {"available": cuda_available}
        if cuda_available:
            major, minor = torch.cuda.get_device_capability(0)
            runtime["cuda"].update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "compute_capability": f"{major}.{minor}",
                }
            )
    except (ImportError, RuntimeError) as exc:
        runtime["cuda"] = {"available": False, "inspection_error": type(exc).__name__}
    return runtime


def build_execution_signature(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    signature: dict[str, Any] = {
        "schema_version": EXECUTION_SIGNATURE_SCHEMA_VERSION,
        "model": _model_artifact_identity(str(args.model_path)),
        "prompt_and_code": {
            "medgemma_ecg_core_sha256": _file_sha256(root / "medgemma_ecg_core.py"),
            "batch_medgemma_diagnostics_sha256": _file_sha256(Path(__file__).resolve()),
            "reasoning_protocol_version": MEDGEMMA_REASONING_PROTOCOL_VERSION,
            "gate_enforcement_version": DIAGNOSTIC_GATE_ENFORCEMENT_VERSION,
        },
        "runtime": _runtime_identity(),
        "configuration": {
            "method": str(args.method),
            "language": str(args.language),
            "model_max_len": MODEL_MAX_LEN,
            "max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
            "dtype": "bfloat16",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_revision_rounds": 1,
            "legacy_non_independent_acknowledged": bool(
                getattr(args, "allow_non_independent_legacy", False)
            ),
        },
    }
    signature["signature_sha256"] = _canonical_sha256(signature)
    return signature


def _validate_execution_signature(
    persisted: Any, expected: dict[str, Any]
) -> None:
    if not isinstance(persisted, dict):
        raise ValueError("persisted result has no execution signature")
    claimed = persisted.get("signature_sha256")
    unsigned = dict(persisted)
    unsigned.pop("signature_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("persisted execution signature is malformed")
    if persisted != expected:
        raise ValueError(
            "persisted result model/prompt/code/gate/runtime/config signature is stale"
        )


def build_generate_text_fn(model_path: str = MODEL_PATH) -> Callable[[str], str]:
    """Compatibility wrapper around the shared MedGemma runtime factory."""
    return _build_generate_text_fn(
        model_path,
        model_max_len=MODEL_MAX_LEN,
        max_output_tokens=MODEL_MAX_OUTPUT_TOKENS,
    )


def iter_samples(input_root: Path, splits: Sequence[str]) -> Iterable[SampleJob]:
    for disease_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for split_name in splits:
            split_dir = disease_dir / split_name
            if not split_dir.exists():
                continue
            features_by_sample: dict[str, Path] = {}
            # Prefer an uncompressed duplicate when both forms exist, while
            # making extractor ``--gzip-json`` output a first-class input.
            for features_path in sorted(split_dir.glob("*_features.json")):
                sample_id = features_path.name[: -len("_features.json")]
                features_by_sample[sample_id] = features_path
            for features_path in sorted(split_dir.glob("*_features.json.gz")):
                sample_id = features_path.name[: -len("_features.json.gz")]
                features_by_sample.setdefault(sample_id, features_path)
            for sample_id, features_path in sorted(features_by_sample.items()):
                report_path = split_dir / f"{sample_id}_report.txt"
                yield SampleJob(
                    disease_dir_label=disease_dir.name,
                    split_name=split_name,
                    sample_id=sample_id,
                    features_path=features_path,
                    report_path=report_path if report_path.exists() else None,
                    local_output_dir=split_dir,
                    mirror_output_dir=input_root.parent / "all_diseases_medgemma" / disease_dir.name / split_name,
                )


def _read_feature_json(path: Path) -> dict[str, Any]:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def process_sample(
    sample: SampleJob,
    language: str,
    clinician_notes: str,
    generate_text_fn: Callable[[str], str],
    *,
    execution_signature: dict[str, Any] | None = None,
    allow_legacy_missing_gate: bool = False,
) -> dict[str, Any]:
    """Run the retained single-shot path.

    This path is intentionally marked non-independent.  Batch evaluation uses
    the gated layered method by default; callers must explicitly opt in to this
    compatibility path at the CLI.
    """
    feature_payload = _read_feature_json(sample.features_path)
    diagnostic_gate = diagnostic_gate_policy(
        feature_payload,
        allow_legacy_missing=allow_legacy_missing_gate,
    )
    if diagnostic_gate["state"] != "pass":
        gate_state = str(diagnostic_gate.get("state") or "stop").upper()
        gate_reasons = [
            *list(diagnostic_gate.get("stop_reasons") or []),
            *list(diagnostic_gate.get("partial_reasons") or []),
        ] or ["unspecified_quality_limitation"]
        raw_model_output = (
            "ABSTAIN — diagnostic quality gate is not PASS "
            f"({gate_state}); the legacy single-shot path cannot enforce "
            "domain-level restrictions, so no model inference was performed: "
            + ", ".join(gate_reasons)
        )
        parsed = {
            "top1_raw": None,
            "top1_normalized": None,
            "similar_raw": [],
            "similar_normalized": [],
            "diagnostic_rationale": "",
            "uncertainty": raw_model_output,
            "abstained": True,
            "abstention_reason": "diagnostic_gate_not_pass",
        }
        sanitized_context = ""
        prompt = ""
    else:
        sanitized_context = build_context_summary_from_paths(
            sample.features_path,
            sample.report_path,
            clinician_notes,
            language,
        )
        prompt = build_prompt(sanitized_context, language)
        raw_model_output = generate_text_fn(prompt)
        parsed = parse_medgemma_output(raw_model_output)
    result = {
        "sample_id": sample.sample_id,
        "split": sample.split_name,
        "ground_truth_dir_label": sample.disease_dir_label,
        "ground_truth_normalized_label": canonical_label_for_dir(sample.disease_dir_label),
        "relative_features_path": str(sample.features_path),
        "relative_report_path": str(sample.report_path) if sample.report_path is not None else None,
        "features_sha256": _file_sha256(sample.features_path),
        "report_sha256": _file_sha256(sample.report_path),
        "sanitized_context": sanitized_context,
        "prompt": prompt,
        "raw_model_output": raw_model_output,
        "parsed": parsed,
        "parse_status": (
            "abstained_diagnostic_gate"
            if parsed.get("abstained")
            else "ok"
            if parsed.get("top1_raw")
            else "missing_top1"
        ),
        "evaluation_method": "legacy_single_shot_non_independent",
        "independent_evaluation": False,
        "evaluation_warning": (
            "Compatibility path only; do not report these metrics as independent MedGemma evaluation."
        ),
        "diagnostic_gate": diagnostic_gate,
        "gate_enforcement_version": DIAGNOSTIC_GATE_ENFORCEMENT_VERSION,
        "execution_signature": execution_signature,
        "errors": [],
    }

    local_txt = sample.local_output_dir / f"{sample.sample_id}_medgemma.txt"
    local_json = sample.local_output_dir / f"{sample.sample_id}_medgemma.json"
    mirror_txt = sample.mirror_output_dir / f"{sample.sample_id}_medgemma.txt"
    mirror_json = sample.mirror_output_dir / f"{sample.sample_id}_medgemma.json"

    _write_text(local_txt, raw_model_output)
    _write_json(local_json, result)
    _write_text(mirror_txt, raw_model_output)
    _write_json(mirror_json, result)
    return result


def process_sample_layered(
    sample: SampleJob,
    language: str,
    clinician_notes: str,
    generate_text_fn: Callable[[str], str],
    *,
    execution_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layered = run_layered_diagnosis(
        sample.features_path,
        sample.report_path,
        clinician_notes,
        language,
        generate_text_fn,
    )
    parsed = layered["parsed"]
    result = {
        "sample_id": sample.sample_id,
        "split": sample.split_name,
        "ground_truth_dir_label": sample.disease_dir_label,
        "ground_truth_normalized_label": canonical_label_for_dir(sample.disease_dir_label),
        "relative_features_path": str(sample.features_path),
        "relative_report_path": str(sample.report_path) if sample.report_path is not None else None,
        "features_sha256": _file_sha256(sample.features_path),
        "report_sha256": _file_sha256(sample.report_path),
        "prompt": layered["prompt"],
        "raw_model_output": layered["raw_model_output"],
        "parsed": parsed,
        "guardrail_notes": layered["guardrail_notes"],
        "unaddressed_guardrail_notes": layered["unaddressed_guardrail_notes"],
        "revised": layered["revised"],
        "reference_only": layered["reference_only"],
        "reference_agreement": layered["reference_agreement"],
        "parse_status": (
            "abstained_diagnostic_gate"
            if layered.get("abstained")
            else "ok"
            if parsed.get("top1_raw")
            else "missing_top1"
        ),
        "evaluation_method": "layered_measurement_only_gated",
        "independent_evaluation": True,
        "gate_enforcement_version": layered.get(
            "gate_enforcement_version", DIAGNOSTIC_GATE_ENFORCEMENT_VERSION
        ),
        "reasoning_mode": layered.get("reasoning_mode"),
        "diagnostic_gate": layered.get("diagnostic_gate"),
        "blocked_layers": layered.get("blocked_layers", []),
        "abstained": layered.get("abstained", False),
        "abstention_reason": layered.get("abstention_reason"),
        "execution_signature": execution_signature,
        "errors": [],
    }

    local_txt = sample.local_output_dir / f"{sample.sample_id}_medgemma_layered.txt"
    local_json = sample.local_output_dir / f"{sample.sample_id}_medgemma_layered.json"
    mirror_txt = sample.mirror_output_dir / f"{sample.sample_id}_medgemma_layered.txt"
    mirror_json = sample.mirror_output_dir / f"{sample.sample_id}_medgemma_layered.json"

    _write_text(local_txt, layered["raw_model_output"])
    _write_json(local_json, result)
    _write_text(mirror_txt, layered["raw_model_output"])
    _write_json(mirror_json, result)
    return result


def _top3_hit(row: dict[str, Any]) -> bool:
    truth = row["ground_truth_normalized_label"]
    candidates = [row.get("top1_normalized")] + [item for item in row.get("similar_normalized", []) if item]
    return truth in candidates


def _summary_row_from_result(
    payload: dict[str, Any],
    sample: SampleJob,
    method: str,
    expected_execution_signature: dict[str, Any],
) -> dict[str, Any]:
    """Validate a persisted result before including it in cohort metrics."""
    expected_truth = canonical_label_for_dir(sample.disease_dir_label)
    identity = {
        "sample_id": sample.sample_id,
        "split": sample.split_name,
        "ground_truth_dir_label": sample.disease_dir_label,
        "ground_truth_normalized_label": expected_truth,
    }
    if not isinstance(payload, dict):
        raise ValueError("persisted result is not a JSON object")
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"persisted result identity mismatch for {key}: "
                f"expected {expected!r}, got {payload.get(key)!r}"
            )
    expected_method = (
        "layered_measurement_only_gated"
        if method == "layered"
        else "legacy_single_shot_non_independent"
    )
    if payload.get("evaluation_method") != expected_method:
        raise ValueError("persisted result was produced by a stale or different evaluation method")
    if method == "layered" and payload.get("independent_evaluation") is not True:
        raise ValueError("persisted layered result is not marked as independent evaluation")
    if method == "legacy" and payload.get("independent_evaluation") is not False:
        raise ValueError("persisted legacy result lacks its non-independent marker")
    if payload.get("features_sha256") != _file_sha256(sample.features_path):
        raise ValueError("persisted result does not match the current feature artifact")
    if payload.get("report_sha256") != _file_sha256(sample.report_path):
        raise ValueError("persisted result does not match the current report artifact")
    _validate_execution_signature(
        payload.get("execution_signature"), expected_execution_signature
    )
    parse_status = str(payload.get("parse_status") or "")
    if parse_status not in {"ok", "abstained_diagnostic_gate"}:
        raise ValueError(f"persisted result is incomplete: parse_status={parse_status or 'missing'}")
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError("persisted result has no parsed result object")
    similar_raw = parsed.get("similar_raw") or []
    similar_normalized = parsed.get("similar_normalized") or []
    if not isinstance(similar_raw, list) or not isinstance(similar_normalized, list):
        raise ValueError("persisted result has malformed similar-diagnosis fields")
    return {
        **identity,
        "top1_raw": parsed.get("top1_raw"),
        "top1_normalized": parsed.get("top1_normalized"),
        "similar_raw": similar_raw,
        "similar_normalized": similar_normalized,
        "operational_status": (
            "abstained" if parse_status == "abstained_diagnostic_gate" else "ok"
        ),
    }


def write_summary_outputs(rows: list[dict[str, Any]], summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "ground_truth_dir_label",
                "ground_truth_normalized_label",
                "operational_status",
                "top1_raw",
                "top1_normalized",
                "similar_raw",
                "similar_normalized",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "ground_truth_dir_label": row["ground_truth_dir_label"],
                    "ground_truth_normalized_label": row["ground_truth_normalized_label"],
                    "operational_status": row.get("operational_status", "ok"),
                    "top1_raw": row["top1_raw"],
                    "top1_normalized": row["top1_normalized"],
                    "similar_raw": json.dumps(row["similar_raw"], ensure_ascii=False),
                    "similar_normalized": json.dumps(row["similar_normalized"], ensure_ascii=False),
                }
            )

    raw_counts: dict[str, Any] = {"top1": {}, "similar": {}, "by_truth": {}}
    combined_top1 = Counter(row["top1_raw"] for row in rows if row.get("top1_raw"))
    combined_similar = Counter(item for row in rows for item in row.get("similar_raw", []) if item)
    raw_counts["top1"]["combined"] = dict(combined_top1)
    raw_counts["similar"]["combined"] = dict(combined_similar)
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        raw_counts["top1"][split] = dict(Counter(row["top1_raw"] for row in split_rows if row.get("top1_raw")))
        raw_counts["similar"][split] = dict(
            Counter(item for row in split_rows for item in row.get("similar_raw", []) if item)
        )
        truth_counts: dict[str, Any] = {}
        for truth in sorted({row["ground_truth_dir_label"] for row in split_rows}):
            truth_rows = [row for row in split_rows if row["ground_truth_dir_label"] == truth]
            truth_counts[truth] = {
                "top1": dict(Counter(row["top1_raw"] for row in truth_rows if row.get("top1_raw"))),
                "similar": dict(
                    Counter(item for row in truth_rows for item in row.get("similar_raw", []) if item)
                ),
            }
        raw_counts["by_truth"][split] = truth_counts
    _write_json(summary_dir / "raw_counts.json", raw_counts)

    by_split = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        total = len(split_rows)
        top1_correct = sum(
            1 for row in split_rows if row["ground_truth_normalized_label"] == row.get("top1_normalized")
        )
        top3_correct = sum(1 for row in split_rows if _top3_hit(row))
        by_split[split] = {
            "count": total,
            "top1_accuracy": (top1_correct / total) if total else 0.0,
            "top3_hit_rate": (top3_correct / total) if total else 0.0,
            "unknown_top1_count": sum(1 for row in split_rows if row.get("top1_normalized") is None),
        }
    total = len(rows)
    total_top1_correct = sum(1 for row in rows if row["ground_truth_normalized_label"] == row.get("top1_normalized"))
    total_top3_correct = sum(1 for row in rows if _top3_hit(row))
    by_truth = {}
    for truth in sorted({row["ground_truth_normalized_label"] for row in rows if row.get("ground_truth_normalized_label")}):
        truth_rows = [row for row in rows if row.get("ground_truth_normalized_label") == truth]
        truth_total = len(truth_rows)
        by_truth[truth] = {
            "count": truth_total,
            "top1_accuracy": (
                sum(1 for row in truth_rows if row.get("top1_normalized") == truth) / truth_total
            )
            if truth_total
            else 0.0,
            "top3_hit_rate": (sum(1 for row in truth_rows if _top3_hit(row)) / truth_total) if truth_total else 0.0,
            "unknown_top1_count": sum(1 for row in truth_rows if row.get("top1_normalized") is None),
        }
    _write_json(
        summary_dir / "normalized_metrics.json",
        {
            "operational": {
                "requested_count": total,
                "completed_count": sum(
                    1 for row in rows if row.get("operational_status", "ok") != "error"
                ),
                "failed_count": sum(
                    1 for row in rows if row.get("operational_status") == "error"
                ),
                "abstained_count": sum(
                    1 for row in rows if row.get("operational_status") == "abstained"
                ),
                "diagnostic_output_count": sum(
                    1 for row in rows if row.get("operational_status", "ok") == "ok"
                ),
                "operational_coverage": (
                    sum(
                        1
                        for row in rows
                        if row.get("operational_status", "ok") != "error"
                    )
                    / total
                )
                if total
                else 0.0,
                "all_samples_completed": not any(
                    row.get("operational_status") == "error" for row in rows
                ),
            },
            "combined": {
                "count": total,
                "top1_accuracy": (total_top1_correct / total) if total else 0.0,
                "top3_hit_rate": (total_top3_correct / total) if total else 0.0,
                "unknown_top1_count": sum(1 for row in rows if row.get("top1_normalized") is None),
            },
            "by_split": by_split,
            "by_truth": by_truth,
        },
    )

    labels = sorted(
        {row["ground_truth_normalized_label"] for row in rows if row.get("ground_truth_normalized_label")}
        | {row["top1_normalized"] for row in rows if row.get("top1_normalized")}
    )
    with (summary_dir / "confusion_top1.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["truth/pred"] + labels)
        for truth in labels:
            writer.writerow(
                [truth]
                + [
                    sum(
                        1
                        for row in rows
                        if row.get("ground_truth_normalized_label") == truth and row.get("top1_normalized") == pred
                    )
                    for pred in labels
                ]
            )

    with (summary_dir / "confusion_top3.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["truth/pred"] + labels)
        for truth in labels:
            writer.writerow(
                [truth]
                + [
                    sum(
                        1
                        for row in rows
                        if row.get("ground_truth_normalized_label") == truth
                        and pred in ([row.get("top1_normalized")] + [item for item in row.get("similar_normalized", []) if item])
                    )
                    for pred in labels
                ]
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch MedGemma diagnostics over all_diseases_ecgfeat.")
    parser.add_argument("--input-dir", type=Path, default=Path("all_diseases_ecgfeat"))
    parser.add_argument("--mirror-output-dir", type=Path, default=Path("all_diseases_medgemma"))
    parser.add_argument("--splits", type=str, default="generated,real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--method",
        type=str,
        default="layered",
        choices=["legacy", "layered"],
        help=(
            "layered (default): independent measurement-only, diagnostic-gate-enforced reasoning. "
            "legacy: NON-INDEPENDENT compatibility-only single-shot prompt; requires "
            "--allow-non-independent-legacy. "
            "The layered path uses bottom-up L0-L5 reasoning with guardrail-triggered revision "
            "(run_layered_diagnosis); withholds unified-clinical-rule conclusions "
            "from the reasoning prompt entirely."
        ),
    )
    parser.add_argument(
        "--allow-non-independent-legacy",
        action="store_true",
        help=(
            "Explicitly acknowledge that --method legacy is a compatibility path and its metrics "
            "must not be reported as an independent MedGemma evaluation."
        ),
    )
    return parser


def parse_splits(raw: str) -> tuple[str, ...]:
    parts = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not parts:
        raise ValueError("at least one split must be provided")
    return parts


def run_batch(args: argparse.Namespace) -> int:
    if args.method == "legacy" and not getattr(
        args, "allow_non_independent_legacy", False
    ):
        raise RuntimeError(
            "--method legacy is non-independent and disabled by default; pass "
            "--allow-non-independent-legacy to acknowledge this limitation"
        )
    if args.limit is not None and args.limit < 0:
        raise RuntimeError("--limit must be zero or greater")
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {args.input_dir}")

    splits = parse_splits(args.splits)
    samples = list(iter_samples(args.input_dir, splits))
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError(
            "no ECG feature samples matched the requested input directory, splits, and limit"
        )

    execution_signature = build_execution_signature(args)
    # Initialize the heavyweight model only when the first non-reusable sample
    # is reached. A zero-sample run and an all-valid --skip-existing run never
    # allocate model/GPU state.
    resolved_generate_text_fn: Callable[[str], str] | None = None

    def lazy_generate_text(prompt: str) -> str:
        nonlocal resolved_generate_text_fn
        if resolved_generate_text_fn is None:
            resolved_generate_text_fn = build_generate_text_fn(args.model_path)
        return resolved_generate_text_fn(prompt)

    rows: list[dict[str, Any]] = []
    sample_count = 0
    failure_count = 0

    output_suffix = "_medgemma_layered" if args.method == "layered" else "_medgemma"
    for sample in samples:
        concrete_sample = SampleJob(
            disease_dir_label=sample.disease_dir_label,
            split_name=sample.split_name,
            sample_id=sample.sample_id,
            features_path=sample.features_path,
            report_path=sample.report_path,
            local_output_dir=sample.local_output_dir,
            mirror_output_dir=args.mirror_output_dir / sample.disease_dir_label / sample.split_name,
        )
        existing_path = (
            concrete_sample.local_output_dir
            / f"{concrete_sample.sample_id}{output_suffix}.json"
        )
        if args.skip_existing and existing_path.exists():
            try:
                existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
                rows.append(
                    _summary_row_from_result(
                        existing_payload,
                        concrete_sample,
                        args.method,
                        execution_signature,
                    )
                )
                sample_count += 1
                continue
            except (OSError, json.JSONDecodeError, ValueError):
                # A corrupt, stale, or identity-mismatched artifact is not a
                # completed sample. Re-run it via the normal process function
                # below instead of silently excluding it from the cohort.
                pass
        try:
            if args.method == "layered":
                result = process_sample_layered(
                    concrete_sample,
                    args.language,
                    "",
                    lazy_generate_text,
                    execution_signature=execution_signature,
                )
            else:
                result = process_sample(
                    concrete_sample,
                    args.language,
                    "",
                    lazy_generate_text,
                    execution_signature=execution_signature,
                    allow_legacy_missing_gate=bool(
                        getattr(args, "allow_non_independent_legacy", False)
                    ),
                )
        except Exception as exc:
            error_payload = {
                "sample_id": concrete_sample.sample_id,
                "split": concrete_sample.split_name,
                "ground_truth_dir_label": concrete_sample.disease_dir_label,
                "ground_truth_normalized_label": canonical_label_for_dir(concrete_sample.disease_dir_label),
                "top1_raw": None,
                "top1_normalized": None,
                "similar_raw": [],
                "similar_normalized": [],
                "operational_status": "error",
                "error": str(exc),
            }
            error_json_payload = {
                "sample_id": concrete_sample.sample_id,
                "split": concrete_sample.split_name,
                "ground_truth_dir_label": concrete_sample.disease_dir_label,
                "ground_truth_normalized_label": canonical_label_for_dir(concrete_sample.disease_dir_label),
                "sanitized_context": "",
                "prompt": "",
                "raw_model_output": "",
                "parsed": {},
                "parse_status": "error",
                "evaluation_method": (
                    "layered_measurement_only_gated"
                    if args.method == "layered"
                    else "legacy_single_shot_non_independent"
                ),
                "independent_evaluation": args.method == "layered",
                "features_sha256": _file_sha256(concrete_sample.features_path),
                "report_sha256": _file_sha256(concrete_sample.report_path),
                "gate_enforcement_version": DIAGNOSTIC_GATE_ENFORCEMENT_VERSION,
                "execution_signature": execution_signature,
                "errors": [str(exc)],
            }
            _write_json(
                concrete_sample.local_output_dir / f"{concrete_sample.sample_id}{output_suffix}.json",
                error_json_payload,
            )
            _write_json(
                args.mirror_output_dir
                / concrete_sample.disease_dir_label
                / concrete_sample.split_name
                / f"{concrete_sample.sample_id}{output_suffix}.json",
                error_json_payload,
            )
            rows.append(error_payload)
            sample_count += 1
            failure_count += 1
            continue

        parse_status = result.get("parse_status")
        operational_status = (
            "ok"
            if parse_status == "ok"
            else "abstained"
            if parse_status == "abstained_diagnostic_gate"
            else "error"
        )
        rows.append(
            {
                "sample_id": result["sample_id"],
                "split": result["split"],
                "ground_truth_dir_label": result["ground_truth_dir_label"],
                "ground_truth_normalized_label": result["ground_truth_normalized_label"],
                "top1_raw": result["parsed"]["top1_raw"],
                "top1_normalized": result["parsed"]["top1_normalized"],
                "similar_raw": result["parsed"]["similar_raw"],
                "similar_normalized": result["parsed"]["similar_normalized"],
                "operational_status": operational_status,
            }
        )
        if operational_status == "error":
            failure_count += 1
        sample_count += 1

    write_summary_outputs(rows, args.mirror_output_dir / "summary")
    return 1 if failure_count else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
