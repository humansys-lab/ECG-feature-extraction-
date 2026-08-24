"""Release gate for a locked ECGAgent end-to-end cohort.

This module does not run a model. It evaluates persisted diagnosis artifacts
from the target backend and, by default, requires the post-review unblinded
export of a separate blinded clinical evaluation before declaring the protocol
acceptable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent.protocol import DIAGNOSTIC_AGENT_PROTOCOL_VERSION
from .evaluation_protocol import EVALUATION_PROTOCOL_VERSION


ACCEPTANCE_GATE_VERSION = "ecgagent.acceptance.v4"
_FINAL_CITATION = re.compile(r"ev:(/[^\s\]\[\"',;}]+)")
_SHA256_FINGERPRINT = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_REMOTE_MODEL_FINGERPRINT = re.compile(
    r"^remote-model-id:sha256:[0-9a-f]{64}$"
)


@dataclass(frozen=True)
class AcceptanceThresholds:
    min_records: int = 20
    min_execution_success_rate: float = 0.95
    min_verified_rate: float = 0.95
    max_p90_runtime_seconds: float = 300.0
    max_mean_revisions: float = 1.5
    max_phase_guard_failure_rate: float = 0.02
    min_visible_provenance_rate: float = 1.0
    min_clinical_f1_delta_vs_best_baseline: float = 0.0
    max_serious_miss_rate: float = 0.02
    disallowed_backends: tuple[str, ...] = ("mock",)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _finite_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _has_no_error(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _verified_artifact_is_usable(row: Mapping[str, Any]) -> bool:
    verdict = row.get("verdict")
    verification = row.get("verification")
    return (
        row.get("verified") is True
        and row.get("ok") is True
        and _has_no_error(row.get("error"))
        and isinstance(verdict, Mapping)
        and bool(verdict)
        and isinstance(verification, Mapping)
        and verification.get("passed") is True
    )


def _diagnosis_files(root: Path) -> list[Path]:
    direct = sorted(root.glob("*.json"))
    nested = sorted((root / "diagnoses").glob("*.json"))
    if nested:
        return nested
    non_diagnosis_names = {
        "record_manifest.json",
        "extraction_manifest.json",
        "diagnosis_manifest.json",
        "analysis.json",
        "summary.json",
    }
    return [path for path in direct if path.name not in non_diagnosis_names]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_fingerprint(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_FINGERPRINT.fullmatch(value))


def _is_model_fingerprint(value: Any) -> bool:
    return _is_sha256_fingerprint(value) or (
        isinstance(value, str)
        and bool(_REMOTE_MODEL_FINGERPRINT.fullmatch(value))
    )


def _record_ids_sha256(record_ids: Sequence[str]) -> str:
    return _canonical_sha256(sorted(str(record_id) for record_id in record_ids))


def _load_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in _diagnosis_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"unreadable diagnosis artifact {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"diagnosis artifact is not a JSON object: {path}")
            continue
        record_id = str(payload.get("record_id") or "").strip()
        if not record_id:
            failures.append(f"diagnosis artifact has no record_id: {path}")
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows, failures


def _load_manifest(root: Path) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    path = root / "record_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [], [f"locked record manifest is missing or unreadable: {exc}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return None, [], ["locked record manifest is not a valid manifest object"]
    record_ids: list[str] = []
    failures: list[str] = []
    for index, row in enumerate(payload["records"]):
        if not isinstance(row, Mapping):
            failures.append(f"record manifest row {index} is not an object")
            continue
        record_id = str(row.get("record_id") or row.get("record") or "").strip()
        if not record_id:
            failures.append(f"record manifest row {index} has no record identifier")
        else:
            record_ids.append(record_id)
    duplicates = sorted(
        record_id for record_id in set(record_ids) if record_ids.count(record_id) > 1
    )
    if duplicates:
        failures.append(
            "record manifest contains duplicate identifiers: " + ", ".join(duplicates)
        )
    declared_count = payload.get("record_count")
    if not isinstance(declared_count, int) or declared_count != len(record_ids):
        failures.append("record manifest record_count does not match its records")
    return payload, record_ids, failures


def _normalize_generation_config(value: Any) -> Any:
    """Remove per-record counters while retaining generation-affecting config."""

    runtime_only = {
        "citation_alias_count",
        "citation_aliases",
        "completed_phases",
        "retained_views",
        "scheduler",
        "phase_memory",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_generation_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in runtime_only
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_generation_config(item) for item in value]
    return value


def _execution_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    audit = row.get("audit") if isinstance(row.get("audit"), Mapping) else {}
    model_audit = audit.get("model") if isinstance(audit.get("model"), Mapping) else {}
    batch = row.get("batch") if isinstance(row.get("batch"), Mapping) else {}
    config = model_audit.get("config") if isinstance(model_audit.get("config"), Mapping) else {}
    stable_batch_keys = (
        "backend",
        "model",
        "thinking",
        "reasoning_effort",
        "knowledge_challenge",
        "knowledge_max_chunks",
        "diagnostic_workflow",
        "max_revisions",
        "record_retries",
        "generation_request",
        "request_signature",
        "runtime_fingerprint",
        "agent_code_fingerprint",
        "model_artifact_fingerprint",
    )
    runtime_fingerprints = {
        key: audit.get(key)
        for key in (
            "runtime_fingerprint",
            "agent_code_fingerprint",
            "model_artifact_fingerprint",
            "weights_sha256",
            "package_versions",
        )
        if audit.get(key) not in (None, "", {}, [])
    }
    for key in (
        "runtime_fingerprint",
        "agent_code_fingerprint",
        "model_artifact_fingerprint",
    ):
        if batch.get(key) not in (None, ""):
            runtime_fingerprints[key] = batch.get(key)
    normalized_config = _normalize_generation_config(config)
    normalized_batch = {
        key: _normalize_generation_config(batch.get(key))
        for key in stable_batch_keys
        if key in batch
    }
    return {
        "agent_protocol": audit.get("agent_protocol"),
        "prompt_fingerprint": audit.get("prompt_fingerprint"),
        "backend": model_audit.get("backend") or batch.get("backend"),
        "model_identifier": (
            normalized_config.get("model")
            or normalized_config.get("model_path")
            or normalized_batch.get("model")
        ),
        "generation_config": normalized_config,
        "batch_generation_config": normalized_batch,
        "runtime_fingerprints": _normalize_generation_config(runtime_fingerprints),
    }


def _clinical_gate(
    path: Path | None,
    thresholds: AcceptanceThresholds,
    *,
    required: bool,
    expected_record_ids_sha256: str | None = None,
    expected_record_count: int | None = None,
    expected_manifest_sha256: str | None = None,
    reviewed_report: Path | None = None,
    require_reviewed_artifact: bool = False,
    expected_truth_source_sha256: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return (
            {"status": "missing", "required": required},
            ["post-review clinical evaluation export is required"] if required else [],
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            {"status": "failed", "required": required, "source": str(path)},
            [f"clinical evaluation export is unreadable: {exc}"],
        )
    if not isinstance(payload, Mapping):
        return (
            {"status": "failed", "required": required, "source": str(path)},
            ["clinical evaluation export is not a JSON object"],
        )
    arms = payload.get("arms") if isinstance(payload, Mapping) else None
    arms = arms if isinstance(arms, Mapping) else {}
    role_map = (
        payload.get("arm_roles")
        if isinstance(payload, Mapping) and isinstance(payload.get("arm_roles"), Mapping)
        else {}
    )
    agent_key = next(
        (str(alias) for alias, role in role_map.items() if str(role) == "agent"),
        "agent",
    )
    agent = arms.get(agent_key) if isinstance(arms.get(agent_key), Mapping) else {}
    baselines = [
        row
        for name, row in arms.items()
        if name != agent_key and isinstance(row, Mapping)
    ]
    agent_f1 = _finite_number(agent.get("micro_f1"), 0.0)
    best_baseline = max(
        (_finite_number(row.get("micro_f1"), 0.0) for row in baselines),
        default=0.0,
    )
    raw_serious_miss_rate = agent.get("serious_miss_rate")
    serious_miss_rate = (
        _finite_number(raw_serious_miss_rate, 1.0)
        if isinstance(raw_serious_miss_rate, (int, float))
        else 1.0
    )
    failures: list[str] = []
    required_roles = {"rules", "single_turn", "agent"}
    if payload.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        failures.append("clinical report uses the wrong evaluation protocol")
    if payload.get("patient_level") is not False:
        failures.append("clinical report is not marked as record-level")
    if payload.get("evaluation_unit") != "record_id":
        failures.append("clinical report primary metrics are not evaluated by record_id")
    if payload.get("blinded") is not False:
        failures.append("clinical report has not reached the post-review unblinded stage")
    if not payload.get("blinding_commitment"):
        failures.append("clinical report has no blinding-key commitment")
    if not payload.get("reviewed_report_sha256"):
        failures.append("clinical report is not bound to a reviewed blinded artifact")
    if not required_roles <= {str(role) for role in role_map.values()}:
        failures.append("clinical report does not contain all three required arms")
    if set(map(str, role_map)) != set(map(str, arms)):
        failures.append("clinical report role map does not exactly cover its arms")
    if _integer(payload.get("record_count"), 0) < thresholds.min_records:
        failures.append("clinical evaluation cohort is below the minimum record count")
    if expected_record_count is not None and _integer(payload.get("record_count")) != expected_record_count:
        failures.append("clinical report record count does not match the locked run cohort")
    if expected_record_ids_sha256 is not None and payload.get("record_ids_sha256") != expected_record_ids_sha256:
        failures.append("clinical report record identifiers do not match the locked run cohort")
    if expected_manifest_sha256 is not None and payload.get("record_manifest_sha256") != expected_manifest_sha256:
        failures.append("clinical report is not bound to the exact locked record manifest")
    if required and expected_truth_source_sha256 is None:
        failures.append("a trusted truth source is required for clinical release gating")
    elif (
        expected_truth_source_sha256 is not None
        and payload.get("truth_source_sha256") != expected_truth_source_sha256
    ):
        failures.append("clinical report does not match the trusted truth source")
    if not payload.get("serious_labels"):
        failures.append("clinical evaluation declares no serious-label set")
    if not agent:
        failures.append(
            "clinical report is still blinded or has no agent arm; rerun the "
            "evaluation export with --unblind after review"
        )
    if agent_f1 - best_baseline < thresholds.min_clinical_f1_delta_vs_best_baseline:
        failures.append(
            "agent clinical F1 does not meet the configured delta over the best baseline"
        )
    if serious_miss_rate > thresholds.max_serious_miss_rate:
        failures.append("agent serious-miss rate exceeds the configured maximum")
    if _integer(agent.get("serious_cases"), 0) < 1:
        failures.append("clinical evaluation contains no serious positive case")
    if expected_record_count is not None and _integer(agent.get("records")) != expected_record_count:
        failures.append("agent arm record count does not match the locked run cohort")

    if required and require_reviewed_artifact and reviewed_report is None:
        failures.append("the exact reviewed blinded report is required for release gating")
    elif reviewed_report is not None:
        try:
            reviewed = json.loads(reviewed_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"reviewed blinded report is unreadable: {exc}")
        else:
            if not isinstance(reviewed, Mapping) or reviewed.get("blinded") is not True:
                failures.append("reviewed report is not the blinded-stage artifact")
            else:
                if payload.get("reviewed_report_sha256") != _canonical_sha256(reviewed):
                    failures.append("clinical report hash does not match the reviewed artifact")
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
                    if payload.get(field) != reviewed.get(field):
                        failures.append(
                            f"clinical report is not bound to reviewed field {field!r}"
                        )
    return (
        {
            "status": "passed" if not failures else "failed",
            "required": required,
            "agent_micro_f1": agent_f1,
            "best_baseline_micro_f1": best_baseline,
            "f1_delta": agent_f1 - best_baseline,
            "serious_miss_rate": serious_miss_rate,
            "source": str(path),
        },
        failures,
    )


def evaluate_acceptance(
    root: Path,
    *,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    clinical_report: Path | None = None,
    reviewed_report: Path | None = None,
    trusted_truth: Path | None = None,
    require_clinical: bool = True,
) -> dict[str, Any]:
    manifest, manifest_record_ids, manifest_failures = _load_manifest(root)
    rows, artifact_failures = _load_rows(root)
    count = len(rows)
    row_ids = [str(row.get("record_id") or "") for row in rows]
    duplicate_ids = sorted(
        record_id for record_id in set(row_ids) if row_ids.count(record_id) > 1
    )
    successful = [
        row
        for row in rows
        if row.get("ok") is True and _has_no_error(row.get("error"))
    ]
    declared_verified = [row for row in rows if row.get("verified") is True]
    verified_successful = [
        row for row in declared_verified if _verified_artifact_is_usable(row)
    ]
    inconsistent_verified_records = [
        str(row.get("record_id") or row.get("_path"))
        for row in declared_verified
        if not _verified_artifact_is_usable(row)
    ]
    runtimes: list[float] = []
    for row in rows:
        row_audit = row.get("audit") if isinstance(row.get("audit"), Mapping) else {}
        runtime_controls = (
            row_audit.get("runtime_controls")
            if isinstance(row_audit.get("runtime_controls"), Mapping)
            else {}
        )
        elapsed = runtime_controls.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and math.isfinite(float(elapsed)):
            runtimes.append(float(elapsed))
    revisions = [
        _integer(row.get("revisions"), 1_000_000) for row in rows
    ]
    phase_count = 0
    phase_guard_failures = 0
    missing_phase_records: list[str] = []
    invalid_phase_guards: list[str] = []
    protocol_mismatches: list[str] = []
    execution_signatures: dict[str, dict[str, Any]] = {}
    visible_final_citations = 0
    final_citations = 0
    for row in rows:
        audit = row.get("audit") if isinstance(row.get("audit"), Mapping) else {}
        if audit.get("agent_protocol") != DIAGNOSTIC_AGENT_PROTOCOL_VERSION:
            protocol_mismatches.append(str(row.get("record_id") or row.get("_path")))
        signature = _execution_signature(row)
        execution_signatures[_canonical_sha256(signature)] = signature
        phases = row.get("phases")
        if not isinstance(phases, list) or not phases:
            missing_phase_records.append(
                str(row.get("record_id") or row.get("_path"))
            )
            phases = []
        for phase_index, phase in enumerate(phases):
            phase_count += 1
            if (
                not isinstance(phase, Mapping)
                or not isinstance(phase.get("phase_guard_passed"), bool)
            ):
                invalid_phase_guards.append(
                    f"{row.get('record_id') or row.get('_path')}[{phase_index}]"
                )
                phase_guard_failures += 1
            elif phase.get("phase_guard_passed") is False:
                phase_guard_failures += 1
        tools = audit.get("tools") if isinstance(audit.get("tools"), Mapping) else {}
        whitelist = set()
        effective = tools.get("effective_evidence")
        if isinstance(effective, Mapping):
            for item in effective.get("items") or []:
                if not isinstance(item, Mapping) or not item.get("model_visible"):
                    continue
                pointer = item.get("pointer")
                if isinstance(pointer, str) and pointer:
                    whitelist.add(pointer)
        # Compatibility with pre-compaction result artifacts.
        provenance = tools.get("provenance")
        if isinstance(provenance, Mapping):
            visible_by_source = provenance.get("visible_by_source")
            if isinstance(visible_by_source, Mapping):
                for pointers in visible_by_source.values():
                    if isinstance(pointers, (list, tuple, set)):
                        whitelist.update(str(pointer) for pointer in pointers)
        verdict_text = json.dumps(row.get("verdict"), ensure_ascii=False, default=str)
        cited_pointers = [match.group(1) for match in _FINAL_CITATION.finditer(verdict_text)]
        final_citations += len(cited_pointers)
        visible_final_citations += sum(
            pointer in whitelist for pointer in cited_pointers
        )

    success_rate = len(successful) / count if count else 0.0
    verified_rate = len(declared_verified) / count if count else 0.0
    verified_success_rate = len(verified_successful) / count if count else 0.0
    guard_rate = phase_guard_failures / phase_count if phase_count else None
    provenance_rate = (
        min(1.0, visible_final_citations / final_citations)
        if final_citations
        else 0.0
    )
    mean_revisions = statistics.fmean(revisions) if revisions else 0.0
    p90 = _percentile(runtimes, 0.90)
    failures: list[str] = [*manifest_failures, *artifact_failures]
    trusted_truth_sha256: str | None = None
    if trusted_truth is not None:
        try:
            trusted_truth_sha256 = _file_sha256(trusted_truth)
        except OSError as exc:
            failures.append(f"trusted truth source is unreadable: {exc}")
    if duplicate_ids:
        failures.append(
            "diagnosis cohort contains duplicate record_id values: "
            + ", ".join(duplicate_ids)
        )
    if manifest is not None and not manifest_failures:
        missing = sorted(set(manifest_record_ids) - set(row_ids))
        extra = sorted(set(row_ids) - set(manifest_record_ids))
        if missing or extra or len(row_ids) != len(manifest_record_ids):
            failures.append(
                "diagnosis artifacts do not exactly match the locked record manifest "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
    if count < thresholds.min_records:
        failures.append(f"cohort has {count} records; requires {thresholds.min_records}")
    if protocol_mismatches:
        failures.append(
            f"{len(protocol_mismatches)} record(s) do not use {DIAGNOSTIC_AGENT_PROTOCOL_VERSION}"
        )
    if inconsistent_verified_records:
        failures.append(
            f"{len(inconsistent_verified_records)} record(s) declare verified=true "
            "without ok=true, no error, a verdict, and verification.passed=true: "
            + ", ".join(inconsistent_verified_records)
        )
    if missing_phase_records:
        failures.append(
            f"{len(missing_phase_records)} record(s) have no non-empty phase trace: "
            + ", ".join(missing_phase_records)
        )
    if invalid_phase_guards:
        failures.append(
            f"{len(invalid_phase_guards)} phase(s) have missing or invalid "
            "phase_guard_passed status: "
            + ", ".join(invalid_phase_guards)
        )
    if phase_count == 0:
        failures.append("cohort contains zero auditable phases")
    if len(execution_signatures) != 1:
        failures.append(
            "cohort does not have one locked prompt/backend/model execution signature"
        )
    if any(
        not _is_sha256_fingerprint(signature.get("prompt_fingerprint"))
        or not signature.get("backend")
        or not signature.get("model_identifier")
        or not isinstance(signature.get("generation_config"), Mapping)
        or not signature.get("generation_config")
        or not isinstance(signature.get("batch_generation_config"), Mapping)
        or not isinstance(
            signature.get("batch_generation_config", {}).get(
                "generation_request"
            ),
            Mapping,
        )
        or not signature.get("batch_generation_config", {}).get(
            "generation_request"
        )
        for signature in execution_signatures.values()
    ):
        failures.append("cohort execution signature is incomplete")
    required_runtime_fingerprints = {
        "runtime_fingerprint",
        "agent_code_fingerprint",
        "model_artifact_fingerprint",
    }
    for signature in execution_signatures.values():
        fingerprints = signature.get("runtime_fingerprints")
        fingerprints = fingerprints if isinstance(fingerprints, Mapping) else {}
        missing_fingerprints = sorted(
            key
            for key in required_runtime_fingerprints
            if key not in fingerprints or fingerprints.get(key) in (None, "")
        )
        if missing_fingerprints:
            failures.append(
                "cohort execution signature is missing required fingerprint(s): "
                + ", ".join(missing_fingerprints)
            )
            continue
        if not _is_sha256_fingerprint(fingerprints.get("runtime_fingerprint")):
            failures.append("cohort runtime fingerprint is malformed")
        if not _is_sha256_fingerprint(fingerprints.get("agent_code_fingerprint")):
            failures.append("cohort agent-code fingerprint is malformed")
        model_fingerprint = fingerprints.get("model_artifact_fingerprint")
        if (
            signature.get("backend") == "medgemma-local"
            and not _is_sha256_fingerprint(model_fingerprint)
        ):
            failures.append(
                "local cohort model-artifact fingerprint is not a weight hash"
            )
        elif not _is_model_fingerprint(model_fingerprint):
            failures.append("cohort model-artifact fingerprint is malformed")
    used_backends = {
        str(signature.get("backend") or "")
        for signature in execution_signatures.values()
    }
    if used_backends & set(thresholds.disallowed_backends):
        failures.append("cohort uses a backend disallowed for release acceptance")
    if success_rate < thresholds.min_execution_success_rate:
        failures.append("execution success rate is below threshold")
    if verified_rate < thresholds.min_verified_rate:
        failures.append("verified rate is below threshold")
    joint_threshold = max(
        thresholds.min_execution_success_rate,
        thresholds.min_verified_rate,
    )
    if verified_success_rate < joint_threshold:
        failures.append(
            "verified execution success rate is below the joint threshold"
        )
    if p90 is None or p90 > thresholds.max_p90_runtime_seconds:
        failures.append("P90 runtime is missing or above threshold")
    if len(runtimes) != count:
        failures.append("one or more diagnosis artifacts have no finite runtime")
    if mean_revisions > thresholds.max_mean_revisions:
        failures.append("mean revision count is above threshold")
    if (
        guard_rate is not None
        and guard_rate > thresholds.max_phase_guard_failure_rate
    ):
        failures.append("phase-guard failure rate is above threshold")
    if provenance_rate < thresholds.min_visible_provenance_rate:
        failures.append("not every final citation belongs to model-visible provenance")

    clinical, clinical_failures = _clinical_gate(
        clinical_report,
        thresholds,
        required=require_clinical,
        expected_record_ids_sha256=(
            _record_ids_sha256(manifest_record_ids)
            if manifest is not None and not manifest_failures
            else None
        ),
        expected_record_count=(
            len(manifest_record_ids)
            if manifest is not None and not manifest_failures
            else None
        ),
        expected_manifest_sha256=(
            _canonical_sha256(manifest)
            if manifest is not None and not manifest_failures
            else None
        ),
        reviewed_report=reviewed_report,
        require_reviewed_artifact=require_clinical,
        expected_truth_source_sha256=trusted_truth_sha256,
    )
    failures.extend(clinical_failures)
    return {
        "gate_version": ACCEPTANCE_GATE_VERSION,
        "protocol_version": DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
        "status": "passed" if not failures else "failed",
        "thresholds": asdict(thresholds),
        "metrics": {
            "records": count,
            "execution_success_rate": success_rate,
            "verified_rate": verified_rate,
            "verified_execution_success_rate": verified_success_rate,
            "median_runtime_seconds": statistics.median(runtimes) if runtimes else None,
            "p90_runtime_seconds": p90,
            "mean_revisions": mean_revisions,
            "phase_guard_failure_rate": guard_rate,
            "visible_final_citation_rate": provenance_rate,
        },
        "clinical": clinical,
        "protocol_mismatch_records": protocol_mismatches,
        "record_manifest": {
            "source": str(root / "record_manifest.json"),
            "sha256": _canonical_sha256(manifest) if manifest is not None else None,
            "record_ids_sha256": (
                _record_ids_sha256(manifest_record_ids) if manifest is not None else None
            ),
        },
        "trusted_truth": {
            "source": str(trusted_truth) if trusted_truth is not None else None,
            "sha256": trusted_truth_sha256,
        },
        "execution_signatures": [
            {"sha256": digest, **signature}
            for digest, signature in sorted(execution_signatures.items())
        ],
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--clinical-report", type=Path)
    parser.add_argument(
        "--trusted-truth",
        "--truth",
        dest="trusted_truth",
        type=Path,
        help="trusted truth JSONL used to create the clinical evaluation export",
    )
    parser.add_argument(
        "--reviewed-report",
        type=Path,
        help="exact blinded report reviewed before the clinical report was unblinded",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="do not require the post-review clinical evaluation export",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.runtime_only and args.trusted_truth is None:
        parser.error("clinical release gating requires --trusted-truth")
    report = evaluate_acceptance(
        args.run_dir,
        clinical_report=args.clinical_report,
        reviewed_report=args.reviewed_report,
        trusted_truth=args.trusted_truth,
        require_clinical=not args.runtime_only,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
