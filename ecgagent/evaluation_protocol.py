"""Blinded patient-level ablation evaluation for ECG interpretation arms.

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
import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence


EVALUATION_PROTOCOL_VERSION = "ecgagent.blind-ablation.v2"


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
    patient_truth: dict[str, set[str]] = {}
    for record_id, row in truth.items():
        patient_truth.setdefault(patients[record_id], set()).update(_labels(row))
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
        patient_predictions: dict[str, set[str]] = {
            patient_id: set() for patient_id in patient_truth
        }
        for record_id, predicted in predictions.items():
            patient_predictions[patients[record_id]].update(predicted)
        tp = fp = fn = exact = serious_cases = serious_misses = 0
        for patient_id, expected in patient_truth.items():
            predicted = patient_predictions[patient_id]
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
        report_arms[alias] = {
            "records": len(truth),
            "patients": len(patient_truth),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "exact_match_rate": (
                exact / len(patient_truth) if patient_truth else 0.0
            ),
            "serious_miss_rate": (
                serious_misses / serious_cases if serious_cases else 0.0
            ),
            "serious_cases": serious_cases,
            "prediction_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "patient_level": True,
        "evaluation_unit": "patient_id",
        "blinded": not unblind,
        "blinding_commitment": "sha256:" + hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest(),
        "truth_source_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "record_count": len(truth),
        "patient_count": len(set(patients.values())),
        "serious_labels": sorted(serious),
        "arms": report_arms,
    }
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


def _load_or_create_blinding_key(path: Path, *, unblind: bool) -> str:
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("blinding key file is empty")
        return key
    if unblind:
        raise ValueError("unblind requires the existing blinding key file")
    key = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8")
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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    report = evaluate_arms(
        args.truth,
        arms,
        serious_labels=set(args.serious_label),
        unblind=args.unblind,
        blinding_key=blinding_key,
        reviewed_report=reviewed_report,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
