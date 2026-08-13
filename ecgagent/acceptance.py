"""Release gate for a locked ECGAgent end-to-end cohort.

This module does not run a model. It evaluates persisted diagnosis artifacts
from the target backend and, by default, requires the post-review unblinded
export of a separate blinded clinical evaluation before declaring the protocol
acceptable.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent.protocol import DIAGNOSTIC_AGENT_PROTOCOL_VERSION
from .evaluation_protocol import EVALUATION_PROTOCOL_VERSION


ACCEPTANCE_GATE_VERSION = "ecgagent.acceptance.v1"
_FINAL_CITATION = re.compile(r"ev:(/[^\s\]\[\"',;}]+)")


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


def _diagnosis_files(root: Path) -> list[Path]:
    direct = sorted(root.glob("*.json"))
    nested = sorted((root / "diagnoses").glob("*.json"))
    return nested or direct


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _diagnosis_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "record_id" in payload:
            payload["_path"] = str(path)
            rows.append(payload)
    return rows


def _clinical_gate(
    path: Path | None,
    thresholds: AcceptanceThresholds,
    *,
    required: bool,
) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return (
            {"status": "missing", "required": required},
            ["post-review clinical evaluation export is required"] if required else [],
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    agent_f1 = float(agent.get("micro_f1") or 0.0)
    best_baseline = max((float(row.get("micro_f1") or 0.0) for row in baselines), default=0.0)
    raw_serious_miss_rate = agent.get("serious_miss_rate")
    serious_miss_rate = (
        float(raw_serious_miss_rate)
        if isinstance(raw_serious_miss_rate, (int, float))
        else 1.0
    )
    failures: list[str] = []
    required_roles = {"rules", "single_turn", "agent"}
    if payload.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        failures.append("clinical report uses the wrong evaluation protocol")
    if payload.get("patient_level") is not True:
        failures.append("clinical report is not marked as patient-level")
    if payload.get("evaluation_unit") != "patient_id":
        failures.append("clinical report metrics are not aggregated by patient_id")
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
    if int(payload.get("patient_count") or 0) < thresholds.min_records:
        failures.append("clinical evaluation cohort is below the minimum patient count")
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
    if int(agent.get("serious_cases") or 0) < 1:
        failures.append("clinical evaluation contains no serious positive case")
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
    require_clinical: bool = True,
) -> dict[str, Any]:
    rows = _load_rows(root)
    count = len(rows)
    successful = [row for row in rows if bool(row.get("ok")) and not row.get("error")]
    verified = [row for row in rows if bool(row.get("verified"))]
    runtimes = [
        float((row.get("audit") or {}).get("runtime_controls", {}).get("elapsed_seconds"))
        for row in rows
        if isinstance((row.get("audit") or {}).get("runtime_controls", {}).get("elapsed_seconds"), (int, float))
    ]
    revisions = [int(row.get("revisions") or 0) for row in rows]
    phase_count = 0
    phase_guard_failures = 0
    protocol_mismatches: list[str] = []
    execution_signatures: set[tuple[str, str, str]] = set()
    visible_final_citations = 0
    final_citations = 0
    for row in rows:
        audit = row.get("audit") if isinstance(row.get("audit"), Mapping) else {}
        if audit.get("agent_protocol") != DIAGNOSTIC_AGENT_PROTOCOL_VERSION:
            protocol_mismatches.append(str(row.get("record_id") or row.get("_path")))
        model_audit = audit.get("model") if isinstance(audit.get("model"), Mapping) else {}
        model_config = (
            model_audit.get("config")
            if isinstance(model_audit.get("config"), Mapping)
            else {}
        )
        model_id = str(model_config.get("model") or model_config.get("model_path") or "")
        execution_signatures.add(
            (
                str(audit.get("prompt_fingerprint") or ""),
                str(model_audit.get("backend") or ""),
                model_id,
            )
        )
        for phase in row.get("phases") or []:
            if not isinstance(phase, Mapping):
                continue
            phase_count += 1
            if phase.get("phase_guard_passed") is False:
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
    verified_rate = len(verified) / count if count else 0.0
    guard_rate = phase_guard_failures / phase_count if phase_count else 0.0
    provenance_rate = (
        min(1.0, visible_final_citations / final_citations)
        if final_citations
        else (1.0 if verified else 0.0)
    )
    mean_revisions = statistics.fmean(revisions) if revisions else 0.0
    p90 = _percentile(runtimes, 0.90)
    failures: list[str] = []
    if count < thresholds.min_records:
        failures.append(f"cohort has {count} records; requires {thresholds.min_records}")
    if protocol_mismatches:
        failures.append(
            f"{len(protocol_mismatches)} record(s) do not use {DIAGNOSTIC_AGENT_PROTOCOL_VERSION}"
        )
    if len(execution_signatures) != 1:
        failures.append(
            "cohort does not have one locked prompt/backend/model execution signature"
        )
    if any(not prompt or not backend or not model for prompt, backend, model in execution_signatures):
        failures.append("cohort execution signature is incomplete")
    used_backends = {signature[1] for signature in execution_signatures}
    if used_backends & set(thresholds.disallowed_backends):
        failures.append("cohort uses a backend disallowed for release acceptance")
    if success_rate < thresholds.min_execution_success_rate:
        failures.append("execution success rate is below threshold")
    if verified_rate < thresholds.min_verified_rate:
        failures.append("verified rate is below threshold")
    if p90 is None or p90 > thresholds.max_p90_runtime_seconds:
        failures.append("P90 runtime is missing or above threshold")
    if mean_revisions > thresholds.max_mean_revisions:
        failures.append("mean revision count is above threshold")
    if guard_rate > thresholds.max_phase_guard_failure_rate:
        failures.append("phase-guard failure rate is above threshold")
    if provenance_rate < thresholds.min_visible_provenance_rate:
        failures.append("not every final citation belongs to model-visible provenance")

    clinical, clinical_failures = _clinical_gate(
        clinical_report,
        thresholds,
        required=require_clinical,
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
            "median_runtime_seconds": statistics.median(runtimes) if runtimes else None,
            "p90_runtime_seconds": p90,
            "mean_revisions": mean_revisions,
            "phase_guard_failure_rate": guard_rate,
            "visible_final_citation_rate": provenance_rate,
        },
        "clinical": clinical,
        "protocol_mismatch_records": protocol_mismatches,
        "execution_signatures": [
            {
                "prompt_fingerprint": prompt,
                "backend": backend,
                "model": model,
            }
            for prompt, backend, model in sorted(execution_signatures)
        ],
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--clinical-report", type=Path)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="do not require the post-review clinical evaluation export",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_acceptance(
        args.run_dir,
        clinical_report=args.clinical_report,
        require_clinical=not args.runtime_only,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
