"""Blinded record-level ablation evaluation for ECG interpretation arms.

Inputs are JSONL. Ground truth rows require ``record_id``, ``patient_id`` and
``labels``; each prediction arm requires ``record_id`` and ``labels``. The
tool assigns opaque arm aliases in the report and computes the same metrics for
rules, single-turn LLM and multi-stage Agent outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


EVALUATION_PROTOCOL_VERSION = "ecgagent.blind-ablation.v3"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _labels(row: Mapping[str, Any]) -> set[str]:
    return {str(value) for value in row.get("labels") or [] if str(value)}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _record_ids_sha256(record_ids: Sequence[str]) -> str:
    """Commit to a cohort without exposing labels or arm identities."""

    encoded = json.dumps(
        sorted(str(record_id) for record_id in record_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _arm_alias(key: str, role: str) -> str:
    digest = hmac.new(
        key.encode("utf-8"),
        role.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"arm_{digest[:12]}"


def evaluate_arms(
    truth_path: Path,
    arms: Mapping[str, Path],
    *,
    serious_labels: set[str] | None = None,
    unblind: bool = False,
    blinding_key: str | None = None,
    reviewed_report: Mapping[str, Any] | None = None,
    record_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    truth_rows = _read_jsonl(truth_path)
    truth = {str(row["record_id"]): row for row in truth_rows}
    if len(truth) != len(truth_rows):
        raise ValueError("truth JSONL contains duplicate record_id values")
    patients: dict[str, str] = {}
    for record_id, row in truth.items():
        patient_id = str(row.get("patient_id") or "")
        if not patient_id:
            raise ValueError(f"truth record {record_id} has no patient_id")
        patients[record_id] = patient_id
    serious = set(serious_labels or ())
    record_truth = {
        record_id: _labels(row) for record_id, row in truth.items()
    }
    patient_truth: dict[str, set[str]] = {}
    for record_id, expected in record_truth.items():
        patient_truth.setdefault(patients[record_id], set()).update(expected)
    key = str(blinding_key or secrets.token_hex(32))
    report_arms: dict[str, Any] = {}
    blind_map: dict[str, str] = {}
    for name, path in sorted(arms.items()):
        # The HMAC key is held outside the blinded report. Aliases therefore
        # remain opaque to reviewers but are stable across the later unblind.
        alias = _arm_alias(key, name)
        if alias in blind_map:
            raise ValueError("blinding key produced an arm alias collision")
        blind_map[alias] = name
        prediction_rows = _read_jsonl(path)
        predictions = {
            str(row["record_id"]): _labels(row) for row in prediction_rows
        }
        if len(predictions) != len(prediction_rows):
            raise ValueError(f"prediction arm {name!r} contains duplicate record_id values")
        missing = set(truth) - set(predictions)
        extra = set(predictions) - set(truth)
        if missing or extra:
            raise ValueError(
                f"prediction arm {name!r} does not match the locked truth cohort "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
        tp = fp = fn = exact = serious_cases = serious_misses = 0
        for record_id, expected in record_truth.items():
            predicted = predictions[record_id]
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            exact += int(expected == predicted)
            expected_serious = expected & serious
            if expected_serious:
                serious_cases += 1
                serious_misses += int(not expected_serious <= predicted)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        # Retain the historical patient-level union only as an explicitly
        # labelled secondary analysis. It must never replace the record-level
        # primary endpoint because doing so makes predictions swapped between
        # two ECGs from the same patient appear perfect.
        patient_predictions: dict[str, set[str]] = {
            patient_id: set() for patient_id in patient_truth
        }
        for record_id, predicted in predictions.items():
            patient_predictions[patients[record_id]].update(predicted)
        patient_tp = patient_fp = patient_fn = patient_exact = 0
        for patient_id, expected in patient_truth.items():
            predicted = patient_predictions[patient_id]
            patient_tp += len(expected & predicted)
            patient_fp += len(predicted - expected)
            patient_fn += len(expected - predicted)
            patient_exact += int(expected == predicted)
        patient_precision = (
            patient_tp / (patient_tp + patient_fp)
            if patient_tp + patient_fp
            else 0.0
        )
        patient_recall = (
            patient_tp / (patient_tp + patient_fn)
            if patient_tp + patient_fn
            else 0.0
        )
        patient_f1 = (
            2 * patient_precision * patient_recall
            / (patient_precision + patient_recall)
            if patient_precision + patient_recall
            else 0.0
        )
        report_arms[alias] = {
            "records": len(truth),
            "patients": len(patient_truth),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "exact_match_rate": exact / len(record_truth) if record_truth else 0.0,
            "serious_miss_rate": (
                serious_misses / serious_cases if serious_cases else 0.0
            ),
            "serious_cases": serious_cases,
            "prediction_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "patient_aggregate_secondary": {
                "evaluation_unit": "patient_id_union",
                "micro_precision": patient_precision,
                "micro_recall": patient_recall,
                "micro_f1": patient_f1,
                "exact_match_rate": (
                    patient_exact / len(patient_truth) if patient_truth else 0.0
                ),
            },
        }
    report = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "patient_level": False,
        "evaluation_unit": "record_id",
        "blinded": not unblind,
        "blinding_commitment": "sha256:" + hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest(),
        "truth_source_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "record_ids_sha256": _record_ids_sha256(list(truth)),
        "record_count": len(truth),
        "patient_count": len(set(patients.values())),
        "serious_labels": sorted(serious),
        "arms": report_arms,
    }
    if record_manifest is not None:
        manifest_rows = record_manifest.get("records")
        if not isinstance(manifest_rows, list):
            raise ValueError("record manifest has no records list")
        manifest_ids = [
            str(row.get("record_id") or row.get("record") or "")
            for row in manifest_rows
            if isinstance(row, Mapping)
        ]
        if (
            len(manifest_ids) != len(manifest_rows)
            or len(set(manifest_ids)) != len(manifest_ids)
            or set(manifest_ids) != set(truth)
        ):
            raise ValueError("record manifest does not match the locked truth cohort")
        report["record_manifest_sha256"] = _canonical_sha256(record_manifest)
    if unblind:
        # Release gating happens only after review.  The role map is therefore
        # opt-in and absent from the default evaluator-facing artifact.
        report["arm_roles"] = blind_map
        if reviewed_report is not None:
            if not isinstance(reviewed_report, Mapping):
                raise ValueError("reviewed report must be a JSON object")
            if reviewed_report.get("blinded") is not True:
                raise ValueError("reviewed report must be the blinded-stage artifact")
            for field in (
                "protocol_version",
                "truth_source_sha256",
                "record_ids_sha256",
                "record_manifest_sha256",
                "record_count",
                "patient_count",
                "serious_labels",
                "blinding_commitment",
                "arms",
            ):
                if reviewed_report.get(field) != report.get(field):
                    raise ValueError(
                        f"reviewed blinded report does not match unblind field {field!r}"
                    )
            report["reviewed_report_sha256"] = _canonical_sha256(reviewed_report)
    return report


def _read_blinding_key(path: Path) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("blinding key path is not a regular file")
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o077:
            raise ValueError(
                "blinding key file must not be accessible by group/other "
                f"(found mode {mode:04o})"
            )
        raw_key = os.read(descriptor, 4097)
        if len(raw_key) > 4096:
            raise ValueError("blinding key file is unexpectedly large")
        try:
            key = raw_key.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("blinding key file is not valid UTF-8") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not key:
        raise ValueError("blinding key file is empty")
    return key


def _load_or_create_blinding_key(path: Path, *, unblind: bool) -> str:
    try:
        return _read_blinding_key(path)
    except FileNotFoundError:
        if unblind:
            raise ValueError("unblind requires the existing blinding key file")

    key = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        # Another process won the atomic create race; trust it only after the
        # same permission and regular-file checks used on normal reads.
        return _read_blinding_key(path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(key + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return key


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=JSONL",
        help="repeat for rules, single_turn and agent arms",
    )
    parser.add_argument("--serious-label", action="append", default=[])
    parser.add_argument(
        "--blinding-key-file",
        type=Path,
        required=True,
        help="private key file reused for the blinded and unblinded exports",
    )
    parser.add_argument(
        "--reviewed-report",
        type=Path,
        help="exact blinded report reviewed by clinicians; required with --unblind",
    )
    parser.add_argument(
        "--record-manifest",
        type=Path,
        required=True,
        help="locked record_manifest.json whose exact cohort is being evaluated",
    )
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="include alias-to-role mapping after blinded review is complete",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    arms: dict[str, Path] = {}
    for item in args.arm:
        if "=" not in item:
            parser.error("--arm must be NAME=JSONL")
        name, raw_path = item.split("=", 1)
        arms[name] = Path(raw_path)
    required = {"rules", "single_turn", "agent"}
    if not required <= set(arms):
        parser.error("--arm must include rules, single_turn and agent")
    if args.unblind and args.reviewed_report is None:
        parser.error("--unblind requires --reviewed-report")
    try:
        blinding_key = _load_or_create_blinding_key(
            args.blinding_key_file,
            unblind=args.unblind,
        )
        reviewed_report = (
            json.loads(args.reviewed_report.read_text(encoding="utf-8"))
            if args.reviewed_report is not None
            else None
        )
        record_manifest = (
            json.loads(args.record_manifest.read_text(encoding="utf-8"))
            if args.record_manifest is not None
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    report = evaluate_arms(
        args.truth,
        arms,
        serious_labels=set(args.serious_label),
        unblind=args.unblind,
        blinding_key=blinding_key,
        reviewed_report=reviewed_report,
        record_manifest=record_manifest,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
