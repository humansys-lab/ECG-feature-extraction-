"""Recoverable PTB-XL extraction, ECGAgent diagnosis, and result analysis.

The reference labels are kept out of the feature payload and model context.
They are joined only after an agent run has finished, so label-agreement
analysis cannot leak ground truth into the diagnosis.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .agent.protocol import DEFAULT_DIAGNOSTIC_PROTOCOL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "feature_extraction"
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "ptb-xl" / "05000"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "ptb-xl-metadata"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ptbxl_05000_ecgagent"
DEFAULT_LOCAL_MAX_CONCURRENCY = 6


def _fatal_external_error(error: Any) -> bool:
    """Return whether retrying more records cannot fix this backend error."""
    text = str(error or "").lower()
    fatal_markers = (
        "error code: 401",
        "error code: 402",
        "error code: 403",
        "http 401",
        "http 402",
        "http 403",
        "authenticationerror",
        "permissiondeniederror",
        "insufficient balance",
        "insufficient_balance",
        "insufficient quota",
        "insufficient_quota",
        "billing hard limit",
        "invalid api key",
    )
    return any(marker in text for marker in fatal_markers)


def _json_dump_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=str,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _text_dump_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _sex(value: Any) -> str | None:
    parsed = _finite(value)
    if parsed is None:
        return None
    return "male" if int(round(parsed)) == 1 else "female"


def load_reference_index(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    """Index PTB-XL metadata by the basename of both signal variants."""
    statements = {
        row.get("", ""): row
        for row in _read_csv(metadata_dir / "scp_statements.csv")
        if row.get("")
    }
    index: dict[str, dict[str, Any]] = {}
    for row in _read_csv(metadata_dir / "ptbxl_database.csv"):
        try:
            score_map = ast.literal_eval(row.get("scp_codes") or "{}")
        except (SyntaxError, ValueError):
            score_map = {}
        active: list[str] = []
        for code, score in score_map.items():
            statement = statements.get(str(code), {})
            if statement.get("diagnostic") == "1.0" and float(score) < 50.0:
                continue
            active.append(str(code))
        item = {
            "ecg_id": row.get("ecg_id"),
            "age": _finite(row.get("age")),
            "sex": _sex(row.get("sex")),
            "reference_codes_raw": sorted(str(code) for code in score_map),
            "reference_codes_active": sorted(active),
            "reference_scores": {
                str(code): float(score) for code, score in score_map.items()
            },
            "reference_report": row.get("report") or "",
            "strat_fold": row.get("strat_fold"),
        }
        for field in ("filename_lr", "filename_hr"):
            filename = row.get(field)
            if filename:
                index[Path(filename).name] = item
    return index


def discover_records(
    dataset_dir: Path,
    metadata_dir: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    reference_index = load_reference_index(metadata_dir)
    records: list[dict[str, Any]] = []
    for header in sorted(dataset_dir.glob("*.hea")):
        reference = reference_index.get(header.stem)
        records.append(
            {
                "record": header.stem,
                "record_path": str(header.with_suffix("")),
                "header_path": str(header),
                "age": reference.get("age") if reference else None,
                "sex": reference.get("sex") if reference else None,
                "reference_available": reference is not None,
                "reference_codes_raw": (
                    reference.get("reference_codes_raw", []) if reference else []
                ),
                "reference_codes_active": (
                    reference.get("reference_codes_active", []) if reference else []
                ),
                "reference_scores": (
                    reference.get("reference_scores", {}) if reference else {}
                ),
                "reference_report": (
                    reference.get("reference_report", "") if reference else ""
                ),
                "strat_fold": reference.get("strat_fold") if reference else None,
            }
        )
        if limit is not None and len(records) >= max(0, limit):
            break
    return records


def _feature_path(output_dir: Path, record: str) -> Path:
    return output_dir / "features" / f"{record}_features.json"


def _diagnosis_path(output_dir: Path, record: str) -> Path:
    return output_dir / "diagnoses" / f"{record}.json"


def _diagnosis_report_path(output_dir: Path, record: str) -> Path:
    return output_dir / "diagnoses" / f"{record}.md"


def _brief_report_path(output_dir: Path, record: str) -> Path:
    return output_dir / "brief_reports" / f"{record}_brief.md"


def _agent_trace_path(output_dir: Path, record: str) -> Path:
    return output_dir / "agent_traces" / f"{record}_agent_trace.md"


def _agent_progress_path(output_dir: Path, record: str) -> Path:
    return output_dir / "agent_traces" / f"{record}_agent_progress.json"


def _agent_attempt_trace_path(
    output_dir: Path,
    record: str,
    attempt: int,
) -> Path:
    return output_dir / "agent_traces" / (
        f"{record}_agent_trace_attempt{int(attempt):02d}.md"
    )


def _load_wfdb_record(record_path: Path) -> tuple[Any, float]:
    import numpy as np
    import wfdb
    from ecgfeat.models import STANDARD_12_LEADS

    record = wfdb.rdrecord(str(record_path))
    by_name = {
        str(name).lower(): index
        for index, name in enumerate(record.sig_name)
    }
    missing = [
        lead for lead in STANDARD_12_LEADS if lead.lower() not in by_name
    ]
    if missing:
        raise ValueError(f"missing standard leads: {', '.join(missing)}")
    ecg = np.asarray(record.p_signal, dtype=float)[
        :,
        [by_name[lead.lower()] for lead in STANDARD_12_LEADS],
    ].T
    return ecg, float(record.fs)


def _extract_one(task: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    record, output_text, fs_internal = task
    output_dir = Path(output_text)
    started = time.perf_counter()
    feature_path = _feature_path(output_dir, str(record["record"]))
    try:
        from ecgfeat.api import ECGFeatureExtractor
        from ecgfeat.export import prepare_json_export, to_dict
        from ecgfeat.models import PatientMeta

        ecg, fs = _load_wfdb_record(Path(record["record_path"]))
        extractor = ECGFeatureExtractor(
            fs_internal=fs_internal,
            mains_freq=50,
        )
        features = extractor.extract(
            ecg,
            fs=fs,
            meta=PatientMeta(age=record.get("age"), sex=record.get("sex")),
        )
        payload = prepare_json_export(
            to_dict(features),
            include_beat_features=True,
        )
        _json_dump_atomic(feature_path, payload)
        return {
            "record": record["record"],
            "status": "ok",
            "runtime_seconds": time.perf_counter() - started,
            "feature_path": str(feature_path),
            "feature_bytes": feature_path.stat().st_size,
            "error": None,
        }
    except Exception as exc:
        return {
            "record": record["record"],
            "status": "failed",
            "runtime_seconds": time.perf_counter() - started,
            "feature_path": str(feature_path),
            "feature_bytes": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _valid_existing(path: Path, required_key: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = _read_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return required_key in payload


def _reusable_diagnosis(
    payload: Mapping[str, Any],
    feature_path: Path,
    *,
    model: str,
    thinking: bool,
    reasoning_effort: str,
    backend: str = "deepseek",
    knowledge_challenge: bool = False,
    knowledge_max_chunks: int = (
        DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.default_max_chunks
    ),
    diagnostic_workflow: str = "legacy",
) -> bool:
    """Whether a verified result matches the current code, prompt and input.

    A bare `verified=true` is insufficient after an Agent enhancement: it
    would silently preserve results produced under an older verifier or prompt.
    """
    if not payload.get("verified"):
        return False
    from .agent.diagnostic import (
        DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
        DIAGNOSTIC_PROMPT_FINGERPRINT,
    )

    audit = payload.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    if audit.get("agent_protocol") != DIAGNOSTIC_AGENT_PROTOCOL_VERSION:
        return False
    if audit.get("prompt_fingerprint") != DIAGNOSTIC_PROMPT_FINGERPRINT:
        return False
    saved_workflow = str(audit.get("diagnostic_workflow") or "")
    if saved_workflow and saved_workflow != diagnostic_workflow:
        return False
    if diagnostic_workflow == "compact" and not saved_workflow:
        return False
    navigation_audit = audit.get("knowledge_navigation")
    navigation_audit = (
        navigation_audit if isinstance(navigation_audit, Mapping) else {}
    )
    if diagnostic_workflow == "compact":
        if navigation_audit.get("enabled") is not False:
            return False
    else:
        if navigation_audit.get("enabled") is not True:
            return False
        if int(navigation_audit.get("max_chunks") or 0) != min(
            int(knowledge_max_chunks),
            DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.navigation_ceiling,
        ):
            return False
    knowledge_audit = audit.get("knowledge_challenge")
    knowledge_audit = (
        knowledge_audit if isinstance(knowledge_audit, Mapping) else {}
    )
    if bool(knowledge_audit.get("enabled")) != bool(knowledge_challenge):
        return False
    if knowledge_challenge and int(
        knowledge_audit.get("max_chunks") or 0
    ) != int(knowledge_max_chunks):
        return False

    batch = payload.get("batch")
    batch = batch if isinstance(batch, Mapping) else {}
    saved_backend = str(batch.get("backend") or "deepseek")
    if saved_backend != backend:
        return False
    if batch.get("model") != model or bool(batch.get("thinking")) != bool(thinking):
        return False
    if thinking and batch.get("reasoning_effort") != reasoning_effort:
        return False

    try:
        feature = _read_json(feature_path)
    except (OSError, ValueError, TypeError):
        return False
    from .evidence.store import EvidenceStore

    fingerprint = EvidenceStore.from_dict(feature).diagnostic_view().fingerprint()
    return audit.get("input_fingerprint") == fingerprint


def run_extraction(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    workers: int,
    fs_internal: int,
    reuse_existing: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    tasks: list[tuple[dict[str, Any], str, int]] = []
    for record in records:
        path = _feature_path(output_dir, str(record["record"]))
        if reuse_existing and _valid_existing(path, "clinical_interpretation"):
            results.append(
                {
                    "record": record["record"],
                    "status": "skipped",
                    "runtime_seconds": 0.0,
                    "feature_path": str(path),
                    "feature_bytes": path.stat().st_size,
                    "error": None,
                }
            )
        else:
            tasks.append((record, str(output_dir), int(fs_internal)))

    completed = len(results)
    total = len(records)
    if completed:
        print(f"[extract resume] {completed}/{total} feature files already complete")

    def persist(result: dict[str, Any]) -> None:
        nonlocal completed
        results.append(result)
        completed += 1
        print(
            f"[extract {completed}/{total}] {result['record']} "
            f"{result['status']} {float(result['runtime_seconds']):.2f}s",
            flush=True,
        )

    if workers <= 1:
        for task in tasks:
            persist(_extract_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_one, task) for task in tasks]
            for future in as_completed(futures):
                persist(future.result())
    results.sort(key=lambda row: str(row["record"]))
    _json_dump_atomic(
        output_dir / "extraction_manifest.json",
        {
            "record_count": len(records),
            "ok_or_skipped": sum(
                row["status"] in {"ok", "skipped"} for row in results
            ),
            "failed": sum(row["status"] == "failed" for row in results),
            "records": results,
        },
    )
    return results


def _source_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    clinical = document.get("clinical_interpretation")
    clinical = clinical if isinstance(clinical, Mapping) else {}
    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    gate = metadata.get("diagnostic_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    return {
        "baseline_codes": [
            str(item.get("statement_code"))
            for item in (clinical.get("final_statements") or [])
            if isinstance(item, Mapping) and item.get("statement_code")
        ],
        "baseline_rule_ids": [
            str(item.get("rule_id"))
            for item in (clinical.get("final_statements") or [])
            if isinstance(item, Mapping) and item.get("rule_id")
        ],
        "baseline_review_required": bool(clinical.get("review_required")),
        "baseline_review_reasons": list(clinical.get("review_reasons") or []),
        "gate_state": str(gate.get("state") or "unknown"),
        "record_grade": (
            (metadata.get("record_quality") or {}).get("record_grade")
            if isinstance(metadata.get("record_quality"), Mapping)
            else None
        ),
    }


def _diagnose_one(
    task: tuple[
        dict[str, Any], str, str, int, int, bool, str, bool, bool, int, str
    ],
    *,
    backend_name: str = "deepseek",
    backend_override: Any = None,
) -> dict[str, Any]:
    if len(task) == 10:
        # Private worker tuples from v31/tests did not carry a workflow. Keep
        # their historical five-stage behavior while every public batch entry
        # point now supplies the explicit compact default.
        task = (*task, "legacy")
    (
        record,
        output_text,
        model,
        max_revisions,
        record_retries,
        thinking,
        reasoning_effort,
        verbose,
        knowledge_challenge,
        knowledge_max_chunks,
        diagnostic_workflow,
    ) = task
    output_dir = Path(output_text)
    feature_path = _feature_path(output_dir, str(record["record"]))
    diagnosis_path = _diagnosis_path(output_dir, str(record["record"]))
    report_path = _diagnosis_report_path(output_dir, str(record["record"]))
    brief_report_path = _brief_report_path(output_dir, str(record["record"]))
    trace_path = _agent_trace_path(output_dir, str(record["record"]))
    progress_path = _agent_progress_path(output_dir, str(record["record"]))
    attempts: list[dict[str, Any]] = []
    attempt_trace_reports: list[tuple[int, str]] = []
    final_payload: dict[str, Any] | None = None
    final_result: Any = None
    started = time.perf_counter()

    for attempt in range(record_retries + 1):
        attempt_started = time.perf_counter()
        try:
            from .agent.diagnostic import ECGDiagnosticAgent
            from .backends import build_backend
            from .evidence.store import EvidenceStore

            store = EvidenceStore.from_path(
                feature_path,
                record_id=str(record["record"]),
            )
            if backend_override is not None:
                session_factory = getattr(backend_override, "new_session", None)
                if callable(session_factory):
                    backend = session_factory()
                else:
                    backend = backend_override
                    reset = getattr(backend, "reset_session", None)
                    if callable(reset):
                        reset()
            elif backend_name == "deepseek":
                backend = build_backend(
                    "deepseek",
                    model=model,
                    user_id="ecgagent_ptbxl_batch",
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                )
            else:
                backend = build_backend(backend_name, model=model)
            result = ECGDiagnosticAgent(
                store=store,
                backend=backend,
                max_revisions=max_revisions,
                knowledge_challenge=knowledge_challenge,
                knowledge_max_chunks=knowledge_max_chunks,
                workflow=diagnostic_workflow,
                on_event=(
                    (lambda line: print(
                        f"[{record['record']}] {line}",
                        flush=True,
                    ))
                    if verbose
                    else None
                ),
                on_checkpoint=(
                    lambda payload, attempt_number=attempt + 1: _json_dump_atomic(
                        progress_path,
                        {
                            **payload,
                            "attempt": attempt_number,
                            "progress_path": str(progress_path),
                        },
                    )
                ),
            ).run()
            final_result = result
            final_payload = result.to_dict()
            final_payload["source"] = _source_summary(store.document)
            final_payload["reference"] = {
                "available": bool(record.get("reference_available")),
                "codes_raw": list(record.get("reference_codes_raw") or []),
                "codes_active": list(record.get("reference_codes_active") or []),
                "scores": dict(record.get("reference_scores") or {}),
                "report": str(record.get("reference_report") or ""),
                "strat_fold": record.get("strat_fold"),
            }
            attempt_row = {
                "attempt": attempt + 1,
                "runtime_seconds": time.perf_counter() - attempt_started,
                "ok": result.ok,
                "verified": result.verified,
                "error": result.error,
            }
            attempts.append(attempt_row)
            attempt_payload = dict(final_payload)
            attempt_payload["batch"] = {
                "attempts": [attempt_row],
                "runtime_seconds": time.perf_counter() - attempt_started,
                "backend": backend_name,
                "model": model,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort if thinking else None,
            }
            attempt_trace_reports.append(
                (attempt + 1, result.render_trace(attempt_payload))
            )
            if result.ok and result.verified:
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "runtime_seconds": time.perf_counter() - attempt_started,
                    "ok": False,
                    "verified": False,
                    "error": error,
                    "traceback": traceback.format_exc(),
                }
            )
            if _fatal_external_error(error):
                break

    if final_payload is None:
        final_payload = {
            "record_id": record["record"],
            "ok": False,
            "verified": False,
            "error": attempts[-1]["error"] if attempts else "no attempt ran",
            "verdict": None,
            "verification": None,
            "phases": [],
            "audit": {},
            "source": {},
            "reference": {
                "available": bool(record.get("reference_available")),
                "codes_raw": list(record.get("reference_codes_raw") or []),
                "codes_active": list(record.get("reference_codes_active") or []),
                "scores": dict(record.get("reference_scores") or {}),
                "report": str(record.get("reference_report") or ""),
                "strat_fold": record.get("strat_fold"),
            },
        }
    final_payload["batch"] = {
        "attempts": attempts,
        "runtime_seconds": time.perf_counter() - started,
        "backend": backend_name,
        "model": model,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort if thinking else None,
        "knowledge_challenge": bool(knowledge_challenge),
        "knowledge_max_chunks": int(knowledge_max_chunks),
        "diagnostic_workflow": diagnostic_workflow,
    }
    from .trace import TRACE_FORMAT_VERSION, render_agent_trace

    final_payload["trajectory"] = {
        "format": TRACE_FORMAT_VERSION,
        "path": str(trace_path),
        "full_model_visible_tool_results": final_result is not None,
        "progress_path": str(progress_path),
    }
    report_text = str(final_payload.get("human_report") or "").strip()
    if not report_text:
        from .report import render_human_report

        report_text = render_human_report(
            (
                final_payload["verdict"]
                if isinstance(final_payload.get("verdict"), Mapping)
                else None
            ),
            record_id=str(record["record"]),
            verified=(
                bool(final_payload.get("verified"))
                if isinstance(final_payload.get("verdict"), Mapping)
                else None
            ),
            knowledge_navigation=(
                final_payload.get("knowledge_navigation")
                if isinstance(
                    final_payload.get("knowledge_navigation"), Mapping
                )
                else None
            ),
            knowledge_review=(
                final_payload.get("knowledge_review")
                if isinstance(final_payload.get("knowledge_review"), Mapping)
                else None
            ),
            error=(
                str(final_payload.get("error"))
                if final_payload.get("error")
                else None
            ),
        )
        final_payload["human_report"] = report_text
    brief_report_text = str(final_payload.get("brief_report") or "").strip()
    if not brief_report_text:
        from .report import render_brief_report

        brief_report_text = render_brief_report(
            (
                final_payload["verdict"]
                if isinstance(final_payload.get("verdict"), Mapping)
                else None
            ),
            record_id=str(record["record"]),
            verified=(
                bool(final_payload.get("verified"))
                if isinstance(final_payload.get("verdict"), Mapping)
                else None
            ),
            error=(
                str(final_payload.get("error"))
                if final_payload.get("error")
                else None
            ),
        )
        final_payload["brief_report"] = brief_report_text
    trace_text = (
        final_result.render_trace(final_payload)
        if final_result is not None
        else render_agent_trace(final_payload)
    )
    _json_dump_atomic(diagnosis_path, final_payload)
    if report_text:
        _text_dump_atomic(report_path, report_text)
    if brief_report_text:
        _text_dump_atomic(brief_report_path, brief_report_text)
    _text_dump_atomic(trace_path, trace_text)
    _json_dump_atomic(
        progress_path,
        {
            "record_id": record["record"],
            "status": "finished",
            "ok": bool(final_payload.get("ok")),
            "verified": bool(final_payload.get("verified")),
            "error": final_payload.get("error"),
            "attempts": attempts,
            "runtime_seconds": final_payload["batch"]["runtime_seconds"],
            "diagnosis_path": str(diagnosis_path),
            "report_path": str(report_path),
            "brief_report_path": str(brief_report_path),
            "trace_path": str(trace_path),
            "updated_at_epoch": time.time(),
        },
    )
    if len(attempt_trace_reports) > 1:
        for attempt_number, attempt_trace in attempt_trace_reports[:-1]:
            _text_dump_atomic(
                _agent_attempt_trace_path(
                    output_dir,
                    str(record["record"]),
                    attempt_number,
                ),
                attempt_trace,
            )
    return {
        "record": record["record"],
        "status": (
            "verified"
            if final_payload.get("verified")
            else "failed"
            if final_payload.get("error")
            else "unverified"
        ),
        "runtime_seconds": final_payload["batch"]["runtime_seconds"],
        "attempts": len(attempts),
        "error": final_payload.get("error"),
        "diagnosis_path": str(diagnosis_path),
        "report_path": str(report_path) if report_text else "",
        "brief_report_path": (
            str(brief_report_path) if brief_report_text else ""
        ),
        "trace_path": str(trace_path),
        "progress_path": str(progress_path),
    }


def run_diagnosis(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    workers: int,
    model: str,
    max_revisions: int,
    record_retries: int,
    thinking: bool,
    reasoning_effort: str,
    verbose: bool,
    reuse_existing: bool,
    retry_failed: bool,
    backend: str = "deepseek",
    model_max_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    qwen_base_url: str | None = None,
    knowledge_challenge: bool = False,
    knowledge_max_chunks: int = (
        DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.default_max_chunks
    ),
    diagnostic_workflow: str = "compact",
) -> list[dict[str, Any]]:
    if diagnostic_workflow not in {"compact", "legacy"}:
        raise ValueError("diagnostic_workflow must be `compact` or `legacy`")
    knowledge_max_chunks = max(
        1,
        min(
            int(knowledge_max_chunks),
            DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.challenge_ceiling,
        ),
    )
    backend = {
        "medgemma": "medgemma-local",
        "qwen": "qwen-local",
    }.get(backend, backend)
    if backend not in {"deepseek", "qwen-local", "medgemma-local"}:
        raise ValueError(
            "batch diagnosis backend must be `deepseek`, `qwen-local` or "
            "`medgemma-local`"
        )
    if backend == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set; provide it through the process environment"
        )
    if backend == "medgemma-local":
        raw_limit = os.getenv("ECG_GEMMA_MAX_CONCURRENCY", "").strip()
        try:
            local_limit = (
                int(raw_limit)
                if raw_limit
                else DEFAULT_LOCAL_MAX_CONCURRENCY
            )
        except ValueError as exc:
            raise RuntimeError(
                "ECG_GEMMA_MAX_CONCURRENCY must be a positive integer"
            ) from exc
        if local_limit < 1:
            raise RuntimeError(
                "ECG_GEMMA_MAX_CONCURRENCY must be a positive integer"
            )
        requested_workers = workers
        workers = min(workers, local_limit)
        print(
            f"[agent local] concurrent records={workers}, vLLM micro-batch "
            f"limit={workers}"
            + (
                f" (requested {requested_workers}, capped by "
                f"ECG_GEMMA_MAX_CONCURRENCY={local_limit})"
                if workers != requested_workers
                else ""
            ),
            flush=True,
        )

    results: list[dict[str, Any]] = []
    prioritized_tasks: list[
        tuple[
            int,
            tuple[
                dict[str, Any], str, str, int, int, bool, str, bool, bool, int, str
            ],
        ]
    ] = []
    for record in records:
        feature_path = _feature_path(output_dir, str(record["record"]))
        if not _valid_existing(feature_path, "clinical_interpretation"):
            results.append(
                {
                    "record": record["record"],
                    "status": "missing_feature",
                    "runtime_seconds": 0.0,
                    "attempts": 0,
                    "error": f"missing usable feature file: {feature_path}",
                    "diagnosis_path": str(
                        _diagnosis_path(output_dir, str(record["record"]))
                    ),
                }
            )
            continue
        diagnosis_path = _diagnosis_path(output_dir, str(record["record"]))
        existing: dict[str, Any] | None = None
        if reuse_existing and diagnosis_path.exists():
            try:
                existing = _read_json(diagnosis_path)
            except (OSError, ValueError, TypeError):
                existing = None
        reusable = (
            _reusable_diagnosis(
                existing,
                feature_path,
                model=model,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                backend=backend,
                knowledge_challenge=knowledge_challenge,
                knowledge_max_chunks=knowledge_max_chunks,
                diagnostic_workflow=diagnostic_workflow,
            )
            if existing is not None
            else False
        )
        if existing is not None and (reusable or not retry_failed):
            report_path = _diagnosis_report_path(
                output_dir,
                str(record["record"]),
            )
            brief_report_path = _brief_report_path(
                output_dir,
                str(record["record"]),
            )
            trace_path = _agent_trace_path(
                output_dir,
                str(record["record"]),
            )
            report_text = str(existing.get("human_report") or "").strip()
            if not report_text:
                from .report import render_human_report

                report_text = render_human_report(
                    (
                        existing["verdict"]
                        if isinstance(existing.get("verdict"), Mapping)
                        else None
                    ),
                    record_id=str(record["record"]),
                    verified=(
                        bool(existing.get("verified"))
                        if isinstance(existing.get("verdict"), Mapping)
                        else None
                    ),
                    knowledge_navigation=(
                        existing.get("knowledge_navigation")
                        if isinstance(
                            existing.get("knowledge_navigation"), Mapping
                        )
                        else None
                    ),
                    knowledge_review=(
                        existing.get("knowledge_review")
                        if isinstance(existing.get("knowledge_review"), Mapping)
                        else None
                    ),
                    error=(
                        str(existing.get("error"))
                        if existing.get("error")
                        else None
                    ),
                )
            if report_text and not report_path.exists():
                _text_dump_atomic(report_path, report_text)
            brief_report_text = str(existing.get("brief_report") or "").strip()
            if not brief_report_text:
                from .report import render_brief_report

                brief_report_text = render_brief_report(
                    (
                        existing["verdict"]
                        if isinstance(existing.get("verdict"), Mapping)
                        else None
                    ),
                    record_id=str(record["record"]),
                    verified=(
                        bool(existing.get("verified"))
                        if isinstance(existing.get("verdict"), Mapping)
                        else None
                    ),
                    error=(
                        str(existing.get("error"))
                        if existing.get("error")
                        else None
                    ),
                )
            if brief_report_text and not brief_report_path.exists():
                _text_dump_atomic(brief_report_path, brief_report_text)
            if not trace_path.exists():
                from .trace import render_agent_trace

                _text_dump_atomic(trace_path, render_agent_trace(existing))
            results.append(
                {
                    "record": record["record"],
                    "status": (
                        "skipped_verified"
                        if reusable
                        else "skipped_stale"
                        if existing.get("verified")
                        else "skipped_failed"
                    ),
                    "runtime_seconds": 0.0,
                    "attempts": 0,
                    "error": existing.get("error"),
                    "diagnosis_path": str(diagnosis_path),
                    "report_path": str(report_path),
                    "brief_report_path": str(brief_report_path),
                    "trace_path": str(trace_path),
                }
            )
            continue
        task = (
            record,
            str(output_dir),
            model,
            int(max_revisions),
            int(record_retries),
            bool(thinking),
            str(reasoning_effort),
            bool(verbose),
            bool(knowledge_challenge),
            int(knowledge_max_chunks),
            diagnostic_workflow,
        )
        # On resume, retry prior fatal account/API failures first. This turns
        # the first worker-width into a safe account-state probe and prevents
        # an unchanged 402 from overwriting valuable unverified candidates.
        prioritized_tasks.append(
            (
                0
                if existing is not None
                and _fatal_external_error(existing.get("error"))
                else 1,
                task,
            )
        )
    tasks = [task for _, task in sorted(
        prioritized_tasks,
        key=lambda item: (item[0], str(item[1][0]["record"])),
    )]

    completed = len(results)
    total = len(records)
    if completed:
        print(
            f"[agent prepare] {completed}/{total} records have no pending "
            "model call (reused, skipped, or missing features)",
            flush=True,
        )
    completed_tasks = 0
    total_tasks = len(tasks)

    def persist(result: dict[str, Any]) -> None:
        nonlocal completed, completed_tasks
        results.append(result)
        completed += 1
        completed_tasks += 1
        print(
            f"[agent {completed_tasks}/{total_tasks}] {result['record']} "
            f"{result['status']} {float(result['runtime_seconds']):.2f}s",
            flush=True,
        )

    fatal_error: str | None = None
    deferred_tasks: list[
        tuple[
            dict[str, Any], str, str, int, int, bool, str, bool, bool, int, str
        ]
    ] = []
    shared_backend: Any = None
    if backend in {"qwen-local", "medgemma-local"} and tasks:
        from .backends import build_backend

        backend_kwargs: dict[str, Any] = {"model": model}
        if backend == "medgemma-local":
            if model_max_len is not None:
                backend_kwargs["model_max_len"] = int(model_max_len)
            if gpu_memory_utilization is not None:
                backend_kwargs["gpu_memory_utilization"] = float(
                    gpu_memory_utilization
                )
            backend_kwargs["max_batch_size"] = workers
        else:
            backend_kwargs["thinking"] = bool(thinking)
            if qwen_base_url:
                backend_kwargs["base_url"] = qwen_base_url
        shared_backend = build_backend(backend, **backend_kwargs)

    def invoke(task: tuple[Any, ...]) -> dict[str, Any]:
        if shared_backend is not None:
            return _diagnose_one(
                task,
                backend_name=backend,
                backend_override=shared_backend,
            )
        return _diagnose_one(task)

    if workers <= 1:
        task_iterator = iter(tasks)
        for task in task_iterator:
            result = invoke(task)
            persist(result)
            if _fatal_external_error(result.get("error")):
                fatal_error = str(result["error"])
                deferred_tasks.extend(task_iterator)
                break
    else:
        # Keep only one worker-width of requests in flight. Besides reducing
        # memory use, this lets a fatal account/authentication error stop the
        # queue before hundreds of untouched records are overwritten.
        task_iterator = iter(tasks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: dict[Any, tuple[Any, ...]] = {}
            for _ in range(min(workers, len(tasks))):
                task = next(task_iterator, None)
                if task is not None:
                    pending[pool.submit(invoke, task)] = task

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    result = future.result()
                    persist(result)
                    if (
                        fatal_error is None
                        and _fatal_external_error(result.get("error"))
                    ):
                        fatal_error = str(result["error"])

                if fatal_error is None:
                    for _ in range(len(done)):
                        task = next(task_iterator, None)
                        if task is not None:
                            pending[pool.submit(invoke, task)] = task

            if fatal_error is not None:
                deferred_tasks.extend(task_iterator)

    for task in deferred_tasks:
        record = task[0]
        results.append(
            {
                "record": record["record"],
                "status": "deferred_external_error",
                "runtime_seconds": 0.0,
                "attempts": 0,
                "error": fatal_error,
                "diagnosis_path": str(
                    _diagnosis_path(output_dir, str(record["record"]))
                ),
            }
        )
    if deferred_tasks:
        completed += len(deferred_tasks)
        print(
            f"[agent circuit] deferred {len(deferred_tasks)} untouched records "
            "after a fatal external API error",
            flush=True,
        )

    close_backend = getattr(shared_backend, "close", None)
    if callable(close_backend):
        close_backend()

    results.sort(key=lambda row: str(row["record"]))
    manifest = {
        "backend": backend,
        "model": model,
        "diagnostic_workflow": diagnostic_workflow,
        "workers": workers,
        "record_count": len(records),
        "knowledge_navigation": {
            "enabled": diagnostic_workflow == "legacy",
            "max_chunks": min(
                int(knowledge_max_chunks),
                DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.navigation_ceiling,
            ),
            "runs_after_measurement_survey": True,
            "patient_evidence": False,
        },
        "knowledge_challenge": {
            "enabled": bool(knowledge_challenge),
            "max_chunks": int(knowledge_max_chunks),
            "runs_after_first_pass_verification": True,
        },
        "verified_or_skipped": sum(
            row["status"] in {"verified", "skipped_verified"}
            for row in results
        ),
        "failed_or_unverified": sum(
            row["status"]
            not in {"verified", "skipped_verified"}
            for row in results
        ),
        "circuit_breaker": {
            "tripped": fatal_error is not None,
            "error": fatal_error,
            "deferred_records": len(deferred_tasks),
        },
        "records": results,
    }
    _json_dump_atomic(
        output_dir / "diagnosis_manifest.json",
        manifest,
    )
    if fatal_error is not None:
        raise RuntimeError(
            "diagnosis stopped by external API circuit breaker: "
            f"{fatal_error}; {len(deferred_tasks)} records were not submitted"
        )
    return results


def _audit_whitelist(payload: Mapping[str, Any]) -> frozenset[str]:
    audit = payload.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    tools = audit.get("tools")
    tools = tools if isinstance(tools, Mapping) else {}
    pointers: set[str] = set()

    # Current artifacts persist only final-verdict evidence with concrete
    # values.  A row is provenance-authorised only when the runtime ledger
    # confirmed that its value-bearing citation was actually model-visible.
    effective = tools.get("effective_evidence")
    effective = effective if isinstance(effective, Mapping) else {}
    for item in effective.get("items") or []:
        if not isinstance(item, Mapping) or not item.get("model_visible"):
            continue
        pointer = item.get("pointer")
        if isinstance(pointer, str) and pointer:
            pointers.add(pointer)

    # Backward compatibility for older result files that embedded complete
    # pointer manifests in their provenance section.
    provenance = tools.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    visible_by_source = provenance.get("visible_by_source")
    visible_by_source = (
        visible_by_source if isinstance(visible_by_source, Mapping) else {}
    )
    for values in visible_by_source.values():
        if isinstance(values, (list, tuple, set)):
            pointers.update(str(value) for value in values)
    # Narrow fallback for early v25 artifacts that separated per-call visible
    # citations before aggregating the ledger. Raw touched citations are never
    # accepted here.
    for call in tools.get("calls") or []:
        if not isinstance(call, Mapping):
            continue
        pointers.update(
            str(value) for value in (call.get("visible_citations") or [])
        )
    return frozenset(pointers)


def reverify_results(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Re-run only deterministic contracts against saved candidate verdicts.

    This is intentionally model-free. It can recover a verdict after a
    verifier bug or safe pointer-alias addition, but never edits the verdict
    itself and never promotes a structurally invalid or stale candidate.
    """
    from .agent import diagnostic_prompts
    from .agent.diagnostic import (
        DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
        DIAGNOSTIC_PROMPT_FINGERPRINT,
    )
    from .evidence.store import EvidenceStore
    from .verify import VerificationPolicy, verify_structured

    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record["record"])
        diagnosis_path = _diagnosis_path(output_dir, record_id)
        feature_path = _feature_path(output_dir, record_id)
        row: dict[str, Any] = {
            "record": record_id,
            "status": "",
            "previous_verified": False,
            "verified": False,
            "blocking_findings": None,
        }
        if not diagnosis_path.exists():
            row["status"] = "missing"
            counts["missing"] += 1
            rows.append(row)
            continue
        try:
            payload = _read_json(diagnosis_path)
        except (OSError, ValueError, TypeError):
            row["status"] = "invalid_json"
            counts["invalid_json"] += 1
            rows.append(row)
            continue
        saved_audit = payload.get("audit")
        saved_audit = saved_audit if isinstance(saved_audit, Mapping) else {}
        if (
            saved_audit.get("agent_protocol")
            != DIAGNOSTIC_AGENT_PROTOCOL_VERSION
            or saved_audit.get("prompt_fingerprint")
            != DIAGNOSTIC_PROMPT_FINGERPRINT
        ):
            row["status"] = "stale"
            counts["stale"] += 1
            rows.append(row)
            continue
        row["previous_verified"] = bool(payload.get("verified"))
        if payload.get("verified"):
            row["status"] = "skipped_verified"
            row["verified"] = True
            counts["skipped_verified"] += 1
            rows.append(row)
            continue
        if payload.get("error"):
            row["status"] = "skipped_execution_error"
            counts["skipped_execution_error"] += 1
            rows.append(row)
            continue
        verdict = payload.get("verdict")
        if not isinstance(verdict, dict):
            row["status"] = "no_candidate"
            counts["no_candidate"] += 1
            rows.append(row)
            continue
        if not feature_path.exists():
            row["status"] = "missing"
            counts["missing"] += 1
            rows.append(row)
            continue
        try:
            feature = _read_json(feature_path)
        except (OSError, ValueError, TypeError):
            row["status"] = "invalid_json"
            counts["invalid_json"] += 1
            rows.append(row)
            continue
        audit = payload.get("audit")
        audit = audit if isinstance(audit, dict) else {}
        store = EvidenceStore.from_dict(
            feature,
            record_id=record_id,
        ).diagnostic_view()
        fingerprint = store.fingerprint()
        fresh = (
            audit.get("agent_protocol") == DIAGNOSTIC_AGENT_PROTOCOL_VERSION
            and audit.get("prompt_fingerprint") == DIAGNOSTIC_PROMPT_FINGERPRINT
            and audit.get("input_fingerprint") == fingerprint
        )
        if not fresh:
            row["status"] = "stale"
            counts["stale"] += 1
            rows.append(row)
            continue
        structural = diagnostic_prompts.validate_verdict(
            verdict,
            evidence_document=store.document,
            hypothesis_state=(
                audit.get("final_hypothesis_state")
                if isinstance(audit.get("final_hypothesis_state"), Mapping)
                else None
            ),
        )
        if structural:
            row["status"] = "structural_invalid"
            row["blocking_findings"] = len(structural)
            counts["structural_invalid"] += 1
            rows.append(row)
            continue

        previous_verification = payload.get("verification")
        report = verify_structured(
            verdict,
            store,
            whitelist=_audit_whitelist(payload),
            policy=VerificationPolicy.for_structured(),
        )
        payload["verification"] = report.to_dict()
        payload["verified"] = report.passed
        audit["local_reverification"] = {
            "agent_protocol": DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
            "prompt_fingerprint": DIAGNOSTIC_PROMPT_FINGERPRINT,
            "previous_counts": (
                previous_verification.get("counts")
                if isinstance(previous_verification, Mapping)
                else None
            ),
            "current_counts": report.counts(),
            "passed": report.passed,
            "verdict_edited": False,
        }
        payload["audit"] = audit
        batch = payload.get("batch")
        batch = dict(batch) if isinstance(batch, Mapping) else {}
        batch["local_reverification_attempted"] = True
        payload["batch"] = batch
        from .report import render_brief_report, render_human_report

        report_text = render_human_report(
            verdict,
            record_id=record_id,
            verified=report.passed,
            knowledge_navigation=(
                payload.get("knowledge_navigation")
                if isinstance(payload.get("knowledge_navigation"), Mapping)
                else None
            ),
            knowledge_review=(
                payload.get("knowledge_review")
                if isinstance(payload.get("knowledge_review"), Mapping)
                else None
            ),
        )
        payload["human_report"] = report_text
        brief_report_text = render_brief_report(
            verdict,
            record_id=record_id,
            verified=report.passed,
        )
        payload["brief_report"] = brief_report_text
        _json_dump_atomic(diagnosis_path, payload)
        _text_dump_atomic(
            _diagnosis_report_path(output_dir, record_id),
            report_text,
        )
        _text_dump_atomic(
            _brief_report_path(output_dir, record_id),
            brief_report_text,
        )
        from .trace import render_agent_trace

        _text_dump_atomic(
            _agent_trace_path(output_dir, record_id),
            render_agent_trace(payload),
        )

        row["verified"] = report.passed
        row["blocking_findings"] = len(report.blocking)
        row["status"] = "recovered" if report.passed else "still_unverified"
        counts[row["status"]] += 1
        rows.append(row)

    summary = {
        "record_count": len(records),
        "counts": dict(counts),
        "records": rows,
    }
    _json_dump_atomic(output_dir / "reverification_manifest.json", summary)
    return summary


def _diagnosis_codes(payload: Mapping[str, Any]) -> tuple[set[str], Counter[str]]:
    verdict = payload.get("verdict")
    verdict = verdict if isinstance(verdict, Mapping) else {}
    codes: set[str] = set()
    statuses: Counter[str] = Counter()
    for diagnosis in verdict.get("diagnoses") or []:
        if not isinstance(diagnosis, Mapping):
            continue
        # Diagnosis-first output has no baseline disposition. Keep legacy
        # statuses readable while counting new positive conclusions directly.
        status = str(diagnosis.get("status") or "diagnosed")
        code = str(diagnosis.get("code") or "")
        statuses[status] += 1
        if code and status != "withdrawn":
            codes.add(code)
    return codes, statuses


def _category_specs() -> tuple[Any, ...]:
    from evaluate_target_ecgfeat_diagnosis import CATEGORIES

    return CATEGORIES


def _category_set(codes: set[str], *, reference: bool) -> set[str]:
    output: set[str] = set()
    for spec in _category_specs():
        candidates = spec.ptbxl_refs if reference else spec.prediction_codes
        if codes & set(candidates):
            output.add(spec.key)
    return output


def _metric_rows(
    evaluated: Sequence[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _category_specs():
        tp = fp = fn = tn = 0
        for row in evaluated:
            reference_positive = spec.key in row["reference_categories"]
            predicted_positive = spec.key in row[f"{source}_categories"]
            if reference_positive and predicted_positive:
                tp += 1
            elif predicted_positive:
                fp += 1
            elif reference_positive:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        rows.append(
            {
                "source": source,
                "category": spec.key,
                "category_label": spec.key.replace("_", " ").title(),
                # Keep the legacy column name for historical CSV compatibility,
                # but emit an English label in all newly generated artifacts.
                "category_zh": spec.key.replace("_", " ").title(),
                "semantic_scope": spec.semantic_scope,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def _micro(rows: Sequence[Mapping[str, Any]], *, direct_only: bool) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if not direct_only or row.get("semantic_scope") == "direct"
    ]
    tp = sum(int(row["tp"]) for row in selected)
    fp = sum(int(row["fp"]) for row in selected)
    fn = sum(int(row["fn"]) for row in selected)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fmt_metric(value: Any) -> str:
    parsed = _finite(value)
    return "N/A" if parsed is None else f"{parsed:.3f}"


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    finite = sorted(
        number
        for value in values
        if (number := _finite(value)) is not None
    )
    if not finite:
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(finite)) - 1)
        return finite[index]

    middle = len(finite) // 2
    median = (
        finite[middle]
        if len(finite) % 2
        else (finite[middle - 1] + finite[middle]) / 2
    )
    return {
        "count": len(finite),
        "total": sum(finite),
        "mean": sum(finite) / len(finite),
        "median": median,
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "min": finite[0],
        "max": finite[-1],
    }


def _error_category(error: Any) -> str:
    text = str(error or "")
    lowered = text.lower()
    if "insufficient balance" in lowered or "error code: 402" in lowered:
        return "deepseek_insufficient_balance"
    if "deterministic contract validation" in lowered:
        return "output_contract_validation"
    if "parseable json" in lowered:
        return "output_json_parse"
    if "insufficient tool messages" in lowered:
        return "provider_tool_protocol"
    if "authentication" in lowered or "invalid api key" in lowered:
        return "provider_authentication"
    if "permission" in lowered or "error code: 403" in lowered:
        return "provider_permission"
    return text.split(":", 1)[0] if text else "none"


def analyze_results(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from .agent.diagnostic import (
        DIAGNOSTIC_AGENT_PROTOCOL_VERSION,
        DIAGNOSTIC_PROMPT_FINGERPRINT,
    )
    from .analysis_dashboard import write_dashboard

    by_record = {str(row["record"]): row for row in records}
    payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        path = _diagnosis_path(output_dir, str(record["record"]))
        if not path.exists():
            missing.append(str(record["record"]))
            continue
        try:
            payloads.append(_read_json(path))
        except (OSError, ValueError, TypeError):
            missing.append(str(record["record"]))

    execution: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    agent_code_counts: Counter[str] = Counter()
    baseline_code_counts: Counter[str] = Counter()
    all_baseline_code_counts: Counter[str] = Counter()
    change_code_counts: dict[str, Counter[str]] = {
        "added": Counter(),
        "withdrawn": Counter(),
        "downgraded": Counter(),
    }
    verification_findings: Counter[str] = Counter()
    human_review_counts: Counter[str] = Counter()
    label_effect_counts: Counter[str] = Counter()
    gate_outcomes: dict[str, Counter[str]] = {}
    grade_outcomes: dict[str, Counter[str]] = {}
    runtime_by_outcome: dict[str, list[float]] = {}
    revision_histogram: Counter[int] = Counter()
    tool_calls = revisions = 0
    usage: Counter[str] = Counter()
    source_recovered = 0
    record_rows: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []

    for payload in payloads:
        record_id = str(payload.get("record_id") or "")
        reference_row = by_record.get(record_id, {})
        audit = payload.get("audit")
        audit = audit if isinstance(audit, Mapping) else {}
        current_protocol = (
            audit.get("agent_protocol") == DIAGNOSTIC_AGENT_PROTOCOL_VERSION
            and audit.get("prompt_fingerprint") == DIAGNOSTIC_PROMPT_FINGERPRINT
        )
        verified = bool(payload.get("verified")) and current_protocol
        error = payload.get("error")
        outcome = (
            "stale"
            if not current_protocol
            else "verified"
            if verified
            else "error"
            if error
            else "unverified"
        )
        execution[outcome] += 1
        if error and current_protocol:
            error_counts[_error_category(error)] += 1

        candidate_agent_codes, candidate_statuses = _diagnosis_codes(payload)
        agent_codes = candidate_agent_codes if verified else set()
        source = payload.get("source")
        source = source if isinstance(source, Mapping) else {}
        if "baseline_codes" not in source:
            try:
                source = _source_summary(
                    _read_json(_feature_path(output_dir, record_id))
                )
                source_recovered += 1
            except (OSError, ValueError, TypeError):
                source = {}
        baseline_codes = set(source.get("baseline_codes") or [])
        all_baseline_code_counts.update(baseline_codes)
        if verified:
            status_counts.update(candidate_statuses)
            agent_code_counts.update(agent_codes)
            baseline_code_counts.update(baseline_codes)
            for diagnosis in (payload.get("verdict") or {}).get("diagnoses") or []:
                if not isinstance(diagnosis, Mapping):
                    continue
                status = str(diagnosis.get("status") or "")
                code = str(diagnosis.get("code") or "")
                if status in change_code_counts and code:
                    change_code_counts[status][code] += 1

        verification = payload.get("verification")
        verification = verification if isinstance(verification, Mapping) else {}
        verification_findings.update(verification.get("counts") or {})
        tools = audit.get("tools")
        tools = tools if isinstance(tools, Mapping) else {}
        tool_calls += int(tools.get("n_calls") or 0)
        revisions += int(payload.get("revisions") or 0)
        model = audit.get("model")
        model = model if isinstance(model, Mapping) else {}
        for key, value in (model.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] += value

        reference = payload.get("reference")
        reference = reference if isinstance(reference, Mapping) else {}
        reference_available = bool(
            reference.get(
                "available",
                reference_row.get("reference_available"),
            )
        )
        reference_codes = set(
            reference.get("codes_active")
            or reference_row.get("reference_codes_active")
            or []
        )
        reference_categories = _category_set(reference_codes, reference=True)
        baseline_categories = _category_set(baseline_codes, reference=False)
        agent_categories = _category_set(agent_codes, reference=False)
        label_effect = "not_evaluated"
        if reference_available and verified:
            comparison = {
                "record": record_id,
                "reference_categories": reference_categories,
                "baseline_categories": baseline_categories,
                "agent_categories": agent_categories,
            }
            evaluated.append(comparison)
            baseline_distance = len(reference_categories ^ baseline_categories)
            agent_distance = len(reference_categories ^ agent_categories)
            label_effect = (
                "improved"
                if agent_distance < baseline_distance
                else "worsened"
                if agent_distance > baseline_distance
                else "unchanged"
            )
            label_effect_counts[label_effect] += 1

        batch = payload.get("batch")
        batch = batch if isinstance(batch, Mapping) else {}
        runtime = _finite(batch.get("runtime_seconds"))
        if runtime is not None:
            runtime_by_outcome.setdefault(outcome, []).append(runtime)
        revision_count = int(payload.get("revisions") or 0)
        if verified:
            revision_histogram[revision_count] += 1
        verdict = payload.get("verdict")
        verdict = verdict if isinstance(verdict, Mapping) else {}
        human_review = verdict.get("human_review")
        human_review = (
            human_review if isinstance(human_review, Mapping) else {}
        )
        if verified:
            human_review_counts[
                "required" if human_review.get("required") else "not_required"
            ] += 1
        gate = str(source.get("gate_state") or "unknown")
        grade = str(source.get("record_grade") or "unknown")
        gate_outcomes.setdefault(gate, Counter())[outcome] += 1
        grade_outcomes.setdefault(grade, Counter())[outcome] += 1
        record_rows.append(
            {
                "record": record_id,
                "verified": verified,
                "outcome": outcome,
                "error": error or "",
                "error_category": (
                    _error_category(error)
                    if error and current_protocol
                    else ""
                ),
                "current_diagnostic_protocol": current_protocol,
                "runtime_seconds": runtime,
                "revisions": revision_count,
                "tool_calls": tools.get("n_calls"),
                "gate_state": gate,
                "record_grade": grade,
                "baseline_codes": "|".join(sorted(baseline_codes)),
                "agent_codes": "|".join(sorted(agent_codes)),
                "candidate_agent_codes": "|".join(
                    sorted(candidate_agent_codes)
                ),
                "added_codes": "|".join(
                    sorted(agent_codes - baseline_codes)
                ),
                "removed_codes": "|".join(
                    sorted(baseline_codes - agent_codes)
                ),
                "human_review_required": human_review.get("required"),
                "reference_available": reference_available,
                "reference_codes": "|".join(sorted(reference_codes)),
                "reference_categories": "|".join(sorted(reference_categories)),
                "baseline_categories": "|".join(sorted(baseline_categories)),
                "agent_categories": "|".join(sorted(agent_categories)),
                "label_effect": label_effect,
                "reference_report": reference.get(
                    "report",
                    reference_row.get("reference_report", ""),
                ),
            }
        )

    baseline_metrics = _metric_rows(evaluated, "baseline")
    agent_metrics = _metric_rows(evaluated, "agent")
    all_metrics = [*baseline_metrics, *agent_metrics]
    baseline_micro = _micro(baseline_metrics, direct_only=True)
    agent_micro = _micro(agent_metrics, direct_only=True)

    summary = {
        "schema_version": "ecgagent_batch_analysis.v2",
        "analysis_status": (
            "complete"
            if execution.get("verified", 0) == len(records)
            else "partial"
        ),
        "record_count": len(records),
        "diagnosis_file_count": len(payloads),
        "missing_or_invalid_results": missing,
        "reference_available_records": sum(
            bool(record.get("reference_available")) for record in records
        ),
        "reference_missing_records": [
            str(record["record"])
            for record in records
            if not record.get("reference_available")
        ],
        "execution": dict(execution),
        "verified_coverage": (
            execution.get("verified", 0) / len(records) if records else None
        ),
        "error_category_counts": dict(error_counts.most_common()),
        "fatal_external_error_records": sum(
            count
            for category, count in error_counts.items()
            if category
            in {
                "deepseek_insufficient_balance",
                "provider_authentication",
                "provider_permission",
            }
        ),
        "diagnosis_status_counts": dict(status_counts),
        "diagnosis_change_code_counts": {
            status: dict(counts.most_common())
            for status, counts in change_code_counts.items()
        },
        "verification_finding_counts": dict(verification_findings),
        "agent_code_counts": dict(agent_code_counts.most_common()),
        "baseline_code_counts": dict(baseline_code_counts.most_common()),
        "all_record_baseline_code_counts": dict(
            all_baseline_code_counts.most_common()
        ),
        "source_summaries_recovered_from_features": source_recovered,
        "total_tool_attempts": tool_calls,
        "total_revisions": revisions,
        "verified_revision_histogram": {
            str(key): value for key, value in sorted(revision_histogram.items())
        },
        "runtime_seconds_by_outcome": {
            outcome: _distribution(values)
            for outcome, values in sorted(runtime_by_outcome.items())
        },
        "human_review": dict(human_review_counts),
        "outcome_by_gate_state": {
            key: dict(value) for key, value in sorted(gate_outcomes.items())
        },
        "outcome_by_record_grade": {
            key: dict(value) for key, value in sorted(grade_outcomes.items())
        },
        "model_usage": dict(usage),
        "label_evaluated_records": len(evaluated),
        "label_effect_record_counts": dict(label_effect_counts),
        "direct_label_agreement": {
            "baseline": baseline_micro,
            "agent": agent_micro,
            "delta_f1": (
                agent_micro["f1"] - baseline_micro["f1"]
                if agent_micro["f1"] is not None
                and baseline_micro["f1"] is not None
                else None
            ),
        },
    }
    code_count_rows = [
        {
            "code": code,
            "agent_count": agent_code_counts.get(code, 0),
            "baseline_count": baseline_code_counts.get(code, 0),
            "delta": agent_code_counts.get(code, 0)
            - baseline_code_counts.get(code, 0),
        }
        for code in sorted(set(agent_code_counts) | set(baseline_code_counts))
    ]
    _json_dump_atomic(output_dir / "analysis.json", summary)
    _write_csv(output_dir / "record_results.csv", record_rows)
    _write_csv(output_dir / "category_metrics.csv", all_metrics)
    _write_csv(output_dir / "diagnosis_code_counts.csv", code_count_rows)
    # A diagnosis run can be interrupted before its final manifest write. The
    # analysis pass has already inventoried every on-disk result, so use that
    # observed state to replace any stale partial/smoke manifest.
    _json_dump_atomic(
        output_dir / "diagnosis_manifest.json",
        {
            "manifest_origin": "analysis_reconstruction",
            "record_count": len(records),
            "verified_or_skipped": execution.get("verified", 0),
            "failed_or_unverified": (
                len(records) - execution.get("verified", 0)
            ),
            "circuit_breaker": {
                "tripped": summary["fatal_external_error_records"] > 0,
                "error_category": (
                    "deepseek_insufficient_balance"
                    if error_counts.get("deepseek_insufficient_balance")
                    else None
                ),
                "deferred_records": 0,
            },
            "records": [
                {
                    "record": row["record"],
                    "status": row["outcome"],
                    "runtime_seconds": row["runtime_seconds"],
                    "attempts": None,
                    "error": row["error"] or None,
                    "diagnosis_path": str(
                        _diagnosis_path(output_dir, str(row["record"]))
                    ),
                }
                for row in record_rows
            ],
        },
    )
    _write_analysis_markdown(output_dir / "RESULTS.md", summary, all_metrics)
    write_dashboard(
        output_dir / "RESULTS.html",
        summary,
        all_metrics,
        record_rows=record_rows,
        code_counts=code_count_rows,
    )
    return summary


def _write_analysis_markdown(
    path: Path,
    summary: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    direct = summary["direct_label_agreement"]
    baseline = direct["baseline"]
    agent = direct["agent"]
    execution = summary["execution"]
    errors = summary.get("error_category_counts") or {}
    runtimes = summary.get("runtime_seconds_by_outcome") or {}
    verified_runtime = runtimes.get("verified") or {}
    usage = summary.get("model_usage") or {}
    lines = [
        "# PTB-XL ECGAgent Diagnostic Analysis",
        "",
        "> For research and engineering use; not a medical device. PTB-XL provides record-level labels.",
        "> Label agreement is not clinical accuracy, and an unlabeled abnormality is not automatically an Agent false positive.",
        "",
        "## Completion Status",
        "",
        (
            f"**{summary.get('analysis_status', 'partial').upper()}**: "
            f"{execution.get('verified', 0)}/{summary['record_count']} "
            "records passed the Agent's structure, evidence-source, numeric-consistency and reliability-qualification validation."
        ),
        "",
    ]
    if errors.get("deepseek_insufficient_balance"):
        lines.extend(
            [
                (
                    f"> {errors['deepseek_insufficient_balance']} records are incomplete because of "
                    "DeepSeek `402 Insufficient Balance`. The diagnostic frequencies and label comparisons below use only verified results; "
                    "unverified candidates and API errors are excluded."
                ),
                "",
            ]
        )
    if execution.get("stale"):
        lines.extend(
            [
                (
                    f"> {execution['stale']} historical results do not belong to the current "
                    "`ecgagent.diagnostic.v1` prompt/protocol and are excluded from diagnosis and label metrics."
                ),
                "",
            ]
        )
    lines.extend(
        [
        "## Execution Results",
        "",
        f"- Target records: {summary['record_count']}",
        f"- Diagnosis result files: {summary['diagnosis_file_count']}",
        f"- PTB-XL reference labels available: {summary['reference_available_records']}",
        f"- Verified: {execution.get('verified', 0)}",
        f"- Unverified: {execution.get('unverified', 0)}",
        f"- Execution errors: {execution.get('error', 0)}",
        f"- Legacy protocol/stale: {execution.get('stale', 0)}",
        f"- Label-comparable and verified: {summary['label_evaluated_records']}",
        f"- Tool-call attempts: {summary['total_tool_attempts']}",
        f"- Revision rounds: {summary['total_revisions']}",
        (
            f"- Verified-record runtime: median "
            f"{_fmt_metric(verified_runtime.get('median'))} seconds, P90 "
            f"{_fmt_metric(verified_runtime.get('p90'))} seconds"
        ),
        (
            f"- Model usage: prompt {usage.get('prompt_tokens', 0):,}, "
            f"completion {usage.get('completion_tokens', 0):,}, "
            f"cached {usage.get('cached_tokens', 0):,} tokens"
        ),
        "",
        "### Error Categories",
        "",
        "| Type | Records |",
        "|---|---:|",
    ])
    for category, count in errors.items():
        lines.append(f"| `{category}` | {count} |")
    if not errors:
        lines.append("| None | 0 |")
    lines.extend(
        [
        "",
        "## Independent Agent Diagnoses (Verified Records Only)",
        "",
        "| Status | Count |",
        "|---|---:|",
    ])
    for status in ("diagnosed", "unchanged", "downgraded", "withdrawn", "added"):
        lines.append(
            f"| {status} | "
            f"{summary.get('diagnosis_status_counts', {}).get(status, 0)} |"
        )
    lines.extend(
        [
            "",
            "The current diagnostic protocol does not expose the rule baseline to the model and does not use added/withdrawn as output semantics. "
            "Legacy statuses in the table are retained only for reading historical files.",
        ]
    )
    label_effects = summary.get("label_effect_record_counts") or {}
    human_review = summary.get("human_review") or {}
    lines.extend(
        [
        "",
        (
            f"Verified conclusions requiring human review: {human_review.get('required', 0)}; "
            f"not requiring human review: {human_review.get('not_required', 0)}."
        ),
        "",
        "## Post-Hoc Comparison of Independent Diagnoses and the Deterministic Rule Baseline",
        "",
        "| Source | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Rule baseline | {_fmt_metric(baseline['precision'])} | "
            f"{_fmt_metric(baseline['recall'])} | {_fmt_metric(baseline['f1'])} | "
            f"{baseline['tp']} | {baseline['fp']} | {baseline['fn']} |"
        ),
        (
            f"| ECGAgent | {_fmt_metric(agent['precision'])} | "
            f"{_fmt_metric(agent['recall'])} | {_fmt_metric(agent['f1'])} | "
            f"{agent['tp']} | {agent['fp']} | {agent['fn']} |"
        ),
        "",
        "Only diagnostic families with `semantic_scope=direct` are summarized here; broad/screening families are excluded from this F1.",
        (
            f"Per-record category change relative to the rule baseline: improved {label_effects.get('improved', 0)}, "
            f"worsened {label_effects.get('worsened', 0)}, "
            f"unchanged {label_effects.get('unchanged', 0)}."
        ),
        "This comparison covers only the verified subset and cannot be extrapolated to the whole dataset.",
        "",
        "## Metrics by Diagnostic Family",
        "",
        "| Source | Diagnostic family | Scope | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in metrics:
        lines.append(
            f"| {row['source']} | {row.get('category_label') or row['category']} | "
            f"{row['semantic_scope']} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {_fmt_metric(row['precision'])} | "
            f"{_fmt_metric(row['recall'])} | {_fmt_metric(row['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `RESULTS.html`: visual dashboard of the same metrics (opens offline and includes family-level detection-structure charts).",
            "- `analysis.json`: overall machine-readable statistics.",
            "- `record_results.csv`: per-record rule baseline, Agent change, labels and execution status.",
            "- `category_metrics.csv`: family-level label agreement for the rule baseline and Agent.",
            "- `diagnosis_code_counts.csv`: diagnostic-code frequency changes.",
            "- `diagnoses/*.json`: complete independent diagnoses, deterministic validation and audit traces.",
            "- `diagnoses/*.md`: English explanatory reports generated from the same structured verdict.",
            "- `brief_reports/*_brief.md`: brief reports retaining only conclusions, key evidence, limitations and human recommendations.",
            "- `agent_traces/*_agent_trace.md`: per-phase model output, complete tool results, knowledge navigation, revisions, validation and token/runtime traces.",
            "",
            "## Limitations",
            "",
        ]
    )
    if summary.get("fatal_external_error_records", 0):
        lines.append(
            "- External API/account errors left some records incomplete; the verified subset may have selection bias."
        )
    elif execution.get("error", 0):
        lines.append(
            "- Some records are incomplete because of Agent output or validation errors; label comparison covers only the verified subset."
        )
    lines.append(
        "- PTB-XL provides record-level reference labels, not a waveform-by-waveform gold standard; missing labels and granularity differences affect agreement."
    )
    missing_references = [
        str(record)
        for record in (summary.get("reference_missing_records") or [])
    ]
    if missing_references:
        preview = ", ".join(f"`{record}`" for record in missing_references[:8])
        suffix = ", and others," if len(missing_references) > 8 else ""
        lines.append(
            f"- {preview}{suffix} have no linkable PTB-XL metadata and are excluded from label metrics."
        )
    lines.extend(
        [
            "- All positive, downgraded or non-pass quality-gate conclusions require qualified human review.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("extract", "diagnose", "reverify", "analyze", "all"),
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extract-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--agent-workers",
        type=int,
        default=1,
        help=(
            "concurrent Agent records (default 1; local vLLM micro-batch size "
            "matches this value)"
        ),
    )
    parser.add_argument("--fs-internal", type=int, default=500)
    parser.add_argument(
        "--backend",
        choices=(
            "deepseek",
            "qwen-local",
            "qwen",
            "medgemma-local",
            "medgemma",
        ),
        default="deepseek",
        help=(
            "diagnostic LLM backend; qwen-local uses a native-tool vLLM "
            "OpenAI server; medgemma-local reuses one in-process vLLM engine"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "provider model id or local checkpoint path. Defaults to "
            "deepseek-v4-pro, qwen3.8-27b for qwen-local, or "
            "/workspace/ecg_gemma/medgemma-27b for medgemma-local"
        ),
    )
    parser.add_argument(
        "--qwen-base-url",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible vLLM endpoint for qwen-local (default "
            "http://127.0.0.1:8000/v1; or set QWEN_BASE_URL)"
        ),
    )
    parser.add_argument(
        "--diagnostic-workflow",
        choices=("compact", "legacy"),
        default="compact",
        help=(
            "diagnosis orchestration (default compact: two model decisions and "
            "program-prefetched targeted tools; legacy: five-stage workflow)"
        ),
    )
    parser.add_argument(
        "--model-max-len",
        type=int,
        default=None,
        help="vLLM context length for medgemma-local (default 131072)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory fraction for medgemma-local (default 0.90)",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default="high",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_revisions,
    )
    parser.add_argument(
        "--knowledge-challenge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after the independent verdict passes evidence verification, run "
            "the detached knowledge challenger and require any accepted "
            "revision to re-read ecgfeat measurements"
        ),
    )
    parser.add_argument(
        "--knowledge-max-chunks",
        type=int,
        default=DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.default_max_chunks,
        help=(
            "maximum governed excerpts for survey-to-hypothesis navigation "
            "and the optional post-hoc challenge "
            f"(1-{DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.challenge_ceiling}; "
            "navigation caps at "
            f"{DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.navigation_ceiling})"
        ),
    )
    parser.add_argument(
        "--record-retries",
        type=int,
        default=0,
        help="Whole-record retries after the backend and verifier retries are exhausted.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = discover_records(
        args.dataset_dir.resolve(),
        args.metadata_dir.resolve(),
        limit=args.limit,
    )
    if not records:
        raise SystemExit(f"no WFDB headers found under {args.dataset_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump_atomic(
        output_dir / "record_manifest.json",
        {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "metadata_dir": str(args.metadata_dir.resolve()),
            "record_count": len(records),
            "reference_available": sum(
                bool(row["reference_available"]) for row in records
            ),
            "reference_missing_records": [
                row["record"] for row in records if not row["reference_available"]
            ],
            "records": records,
        },
    )

    if args.command in {"extract", "all"}:
        extraction = run_extraction(
            records,
            output_dir,
            workers=max(1, args.extract_workers),
            fs_internal=args.fs_internal,
            reuse_existing=args.reuse_existing,
        )
        if any(row["status"] == "failed" for row in extraction):
            print("[warning] extraction has failures; diagnosis will skip those records")

    if args.command in {"diagnose", "all"}:
        backend_name = {
            "medgemma": "medgemma-local",
            "qwen": "qwen-local",
        }.get(args.backend, args.backend)
        model = args.model or (
            "/workspace/ecg_gemma/medgemma-27b"
            if backend_name == "medgemma-local"
            else "qwen3.8-27b"
            if backend_name == "qwen-local"
            else "deepseek-v4-pro"
        )
        thinking = (
            args.thinking
            if backend_name in {"deepseek", "qwen-local"}
            else False
        )
        run_diagnosis(
            records,
            output_dir,
            workers=max(1, args.agent_workers),
            model=model,
            max_revisions=max(0, args.max_revisions),
            record_retries=max(0, args.record_retries),
            thinking=thinking,
            reasoning_effort=args.reasoning_effort,
            verbose=args.verbose,
            reuse_existing=args.reuse_existing,
            retry_failed=args.retry_failed,
            backend=backend_name,
            model_max_len=args.model_max_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            qwen_base_url=args.qwen_base_url,
            knowledge_challenge=args.knowledge_challenge,
            knowledge_max_chunks=max(
                1,
                min(
                    args.knowledge_max_chunks,
                    DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.challenge_ceiling,
                ),
            ),
            diagnostic_workflow=args.diagnostic_workflow,
        )

    if args.command == "reverify":
        summary = reverify_results(records, output_dir)
        print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))

    if args.command in {"analyze", "all"}:
        summary = analyze_results(records, output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not summary["missing_or_invalid_results"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
