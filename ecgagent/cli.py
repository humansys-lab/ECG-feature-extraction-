"""Command-line access to the agent tool layer.

Three uses:

    # what the model sees before it calls anything
    python -m ecgagent.cli --features X_features.json --briefing

    # drive one tool by hand
    python -m ecgagent.cli --features X_features.json \
        --tool get_lead_table --args '{"fields":["q_duration_ms","q_r_ratio"]}'

    # emit the tool schemas needed to wire this to a model
    python -m ecgagent.cli --features X_features.json --schema anthropic
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import secrets
import sys
from pathlib import Path

from .agent.protocol import DEFAULT_DIAGNOSTIC_PROTOCOL
from .evidence.briefing import build_chart_briefing
from .evidence.diagnostic_briefing import build_diagnostic_briefing
from .evidence.store import EvidenceStore
from .privacy import external_egress_error, resolve_backend_privacy
from .tools.registry import build_default_registry
from .verify import VerificationPolicy, audit_untraceable_numbers, verify_output


CLI_PSEUDONYM_STRATEGY = "hmac-sha256-ephemeral-cli-v1"


def _pseudonymized_agent_store(
    source: EvidenceStore,
) -> tuple[EvidenceStore, dict[str, str]]:
    """Build a one-run model view whose record identifier is unlinkable."""

    key = secrets.token_bytes(32)
    digest = hmac.new(
        key,
        f"ecgagent:cli-model-visible-record:v1\0{source.record_id}".encode(
            "utf-8"
        ),
        hashlib.sha256,
    ).hexdigest()[:24]
    pseudonym = f"ecg_{digest}"
    document = copy.deepcopy(source.document)
    for container in (document, document.get("metadata")):
        if not isinstance(container, dict):
            continue
        for field in ("record", "record_id"):
            if field in container:
                container[field] = pseudonym
    return EvidenceStore.from_dict(document, record_id=pseudonym), {
        "model_visible_record_id": pseudonym,
        "strategy": CLI_PSEUDONYM_STRATEGY,
        "key_scope": "single_cli_invocation",
    }


def _external_egress_error(
    backend: str,
    *,
    allowed: bool,
    qwen_base_url: str | None = None,
    anthropic_base_url: str | None = None,
) -> str | None:
    """Compatibility wrapper around the shared endpoint-aware policy."""
    return external_egress_error(
        backend,
        allowed=allowed,
        qwen_base_url=qwen_base_url,
        anthropic_base_url=anthropic_base_url,
    )


def _demo(store: EvidenceStore, registry, *, mode: str) -> None:
    """A scripted pull sequence, showing the shape of a real session."""
    briefing = (
        build_diagnostic_briefing(store)
        if mode == "diagnose"
        else build_chart_briefing(store)
    )
    print(briefing.text)
    print()
    if mode == "diagnose":
        script: list[tuple[str, dict]] = [
            ("get_diagnostic_overview", {}),
            ("get_rhythm_profile", {"sections": ["background", "atrial_signal"]}),
            ("get_morphology_groups", {"include_beats": True}),
            ("get_morphology_map", {"profile": "qrs"}),
        ]
    else:
        script = [
            ("list_findings", {"status": ["matched"]}),
            ("get_measurement", {"pointer": "QTc"}),
            (
                "get_lead_table",
                {"fields": ["q_duration_ms", "q_amp_mv", "r_amp_mv", "q_r_ratio"]},
            ),
            (
                "get_beat_table",
                {"fields": ["rr_prev_ms", "rr_next_ms", "group_id"]},
            ),
        ]
    for name, args in script:
        result = registry.call(name, args)
        print(f"--- {name}({json.dumps(args, ensure_ascii=False)}) "
              f"-> ok={result.ok}, {len(result.citations)} citations")
        print(result.render())
        print()
    audit = registry.audit()
    print(f"[audit] {audit['n_calls']} calls, {audit['whitelist_size']} citable pointers")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="path to a *_features.json payload")
    parser.add_argument(
        "--waveform-record", type=Path, default=None,
        help=("explicit matching original WFDB record (base path or .hea) for bounded "
              "raw measurement review; requires physical units and matching timebase"),
    )
    parser.add_argument("--record-id", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=["diagnose", "adjudicate"],
        default="diagnose",
        help="agent semantics (default: independent diagnosis; adjudicate is legacy)",
    )
    parser.add_argument("--briefing", action="store_true", help="print the one-page briefing")
    parser.add_argument("--tool", type=str, default=None, help="tool name to invoke")
    parser.add_argument("--args", type=str, default="{}", help="tool arguments as a JSON object")
    parser.add_argument("--list-tools", action="store_true", help="print the tool catalogue")
    parser.add_argument(
        "--schema",
        choices=["anthropic", "openai", "guided"],
        default=None,
        help="dump tool schemas for a model backend",
    )
    parser.add_argument("--budget", type=int, default=None, help="max tool calls (default unlimited)")
    parser.add_argument("--demo", action="store_true", help="run a scripted pull sequence")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--save-brief-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "save a concise Markdown diagnosis report for every --agent run "
            "(default enabled; use --no-save-brief-report to disable)"
        ),
    )
    parser.add_argument(
        "--brief-report-md",
        type=Path,
        default=None,
        help=(
            "concise report output path; default: a brief_reports directory "
            "beside the feature file"
        ),
    )
    parser.add_argument(
        "--save-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "save a Markdown Agent trajectory for every --agent run "
            "(default enabled; use --no-save-trace to disable)"
        ),
    )
    parser.add_argument(
        "--trace-md",
        type=Path,
        default=None,
        help=(
            "trajectory output path; default: an agent_traces directory beside "
            "the feature file"
        ),
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="verify a model-output text file whose claims carry ev:/... citations",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="audit an uncited model-output text file: can each number be traced to the payload?",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="run the full agent loop over this record using the selected LLM backend",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="medgemma-local",
        choices=[
            "anthropic",
            "deepseek",
            "qwen-local",
            "qwen",
            "medgemma-local",
            "medgemma",
        ],
        help=(
            "LLM provider (default medgemma-local; cloud backends require "
            "--allow-external-egress; deepseek reads DEEPSEEK_API_KEY; "
            "qwen-local connects to a vLLM OpenAI server with native tool "
            "calling; medgemma-local runs an in-process checkpoint)"
        ),
    )
    parser.add_argument(
        "--allow-external-egress",
        action="store_true",
        help=(
            "explicitly authorize sending a run-scoped pseudonymous record ID, "
            "demographics, and ECG measurements to a selected external endpoint"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "model id or local checkpoint path (qwen-local default served "
            "model: qwen3.8-27b; in-process local default: "
            "/workspace/ecg_gemma/medgemma-27b)"
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
        "--anthropic-base-url",
        type=str,
        default=None,
        help=(
            "Anthropic API endpoint (default https://api.anthropic.com; or set "
            "ANTHROPIC_BASE_URL). The resolved endpoint is pinned and audited."
        ),
    )
    parser.add_argument(
        "--qwen-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Qwen thinking and preserve it across native tool turns",
    )
    parser.add_argument(
        "--model-max-len",
        type=int,
        default=None,
        help=(
            "vLLM context length for medgemma-local (default 131072; can also "
            "set ECG_GEMMA_AGENT_MAX_LEN)"
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "vLLM GPU memory fraction for medgemma-local (default 0.90; can "
            "also set ECG_GEMMA_GPU_MEMORY_UTILIZATION)"
        ),
    )
    parser.add_argument(
        "--effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="reasoning effort (default high)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_revisions,
        help=(
            "how many times a failed verification may be sent back "
            f"(default {DEFAULT_DIAGNOSTIC_PROTOCOL.runtime.max_revisions})"
        ),
    )
    parser.add_argument(
        "--knowledge-challenge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after a diagnosis passes evidence verification, run the detached "
            "medical-knowledge challenge and re-read ecgfeat for any revision"
        ),
    )
    parser.add_argument(
        "--diagnostic-workflow",
        choices=["compact", "legacy"],
        default="compact",
        help=(
            "diagnosis orchestration: compact uses program-prefetched plan + "
            "one evidence/adjudication stage (default); legacy preserves the "
            "five-stage v31 research workflow"
        ),
    )
    parser.add_argument(
        "--knowledge-max-chunks",
        type=int,
        default=DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.default_max_chunks,
        help=(
            "maximum governed knowledge excerpts used for survey-to-hypothesis "
            "navigation and, when enabled, the post-hoc challenge "
            f"(1-{DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.challenge_ceiling}; "
            "navigation uses at most "
            f"{DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.navigation_ceiling})"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="suppress agent progress lines")
    args = parser.parse_args(argv)

    if not args.features.exists():
        parser.error(f"features file not found: {args.features}")

    source_store = EvidenceStore.from_path(args.features, record_id=args.record_id)
    from .evidence.waveform_review import (
        STANDARD_LEADS, build_waveform_review, unavailable_waveform_review,
    )

    document = dict(source_store.document)
    if args.waveform_record is not None:
        from .batch import _load_wfdb_record

        record_path = args.waveform_record
        if record_path.suffix == ".hea":
            record_path = record_path.with_suffix("")
        try:
            ecg, fs = _load_wfdb_record(record_path)
        except (OSError, ValueError, ImportError) as exc:
            parser.error(f"could not load --waveform-record: {exc}")
        document["waveform_review"] = build_waveform_review(
            ecg, fs, document, lead_names=STANDARD_LEADS, amplitude_unit="mV",
        )
    elif "waveform_review" not in document:
        document["waveform_review"] = unavailable_waveform_review()
    source_store = EvidenceStore.from_dict(document, record_id=source_store.record_id)
    store = source_store
    if args.mode == "diagnose":
        from .agent.diagnostic import DIAGNOSTIC_TOOLS

        store = store.diagnostic_view()
        registry = build_default_registry(
            store,
            budget=args.budget,
            include=DIAGNOSTIC_TOOLS,
        )
    else:
        registry = build_default_registry(store, budget=args.budget)

    if args.schema:
        if args.schema == "anthropic":
            payload = registry.anthropic_tools()
        elif args.schema == "openai":
            payload = registry.openai_tools()
        else:
            payload = registry.guided_json_schema()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.list_tools:
        print(registry.describe_for_prompt())
        return 0

    if args.demo:
        _demo(store, registry, mode=args.mode)
        return 0

    if args.agent:
        from .agent.diagnostic import ECGDiagnosticAgent
        from .agent.loop import ECGAgent
        from .backends import build_backend

        egress_policy = resolve_backend_privacy(
            args.backend,
            qwen_base_url=args.qwen_base_url,
            anthropic_base_url=args.anthropic_base_url,
        )
        egress_error = _external_egress_error(
            args.backend,
            allowed=bool(args.allow_external_egress),
            qwen_base_url=(
                egress_policy.effective_endpoint
                if args.backend in {"qwen-local", "qwen"}
                else None
            ),
            anthropic_base_url=(
                egress_policy.effective_endpoint
                if args.backend == "anthropic"
                else None
            ),
        )
        if egress_error:
            print(egress_error, file=sys.stderr)
            return 2

        true_record_id = source_store.record_id
        agent_source_store, record_identity = _pseudonymized_agent_store(
            source_store
        )
        if args.mode == "diagnose":
            agent_store = agent_source_store.diagnostic_view()
            agent_registry = build_default_registry(
                agent_store,
                budget=args.budget,
                include=DIAGNOSTIC_TOOLS,
            )
        else:
            agent_store = agent_source_store
            agent_registry = build_default_registry(
                agent_store,
                budget=args.budget,
            )

        backend_kwargs: dict = {}
        if args.model:
            backend_kwargs["model"] = args.model
        if args.backend in {"qwen-local", "qwen"}:
            backend_kwargs["base_url"] = egress_policy.effective_endpoint
            backend_kwargs["thinking"] = bool(args.qwen_thinking)
        if args.backend == "anthropic":
            backend_kwargs["base_url"] = egress_policy.effective_endpoint
        if args.backend in {"medgemma-local", "medgemma"}:
            if args.model_max_len is not None:
                backend_kwargs["model_max_len"] = args.model_max_len
            if args.gpu_memory_utilization is not None:
                backend_kwargs["gpu_memory_utilization"] = (
                    args.gpu_memory_utilization
                )
        # `effort` is an Anthropic-only control; local Qwen and DeepSeek use
        # their own thinking controls instead.
        if args.effort and args.backend == "anthropic":
            backend_kwargs["effort"] = args.effort
        elif args.effort:
            print(
                f"--effort is ignored on the {args.backend} backend (no equivalent control)",
                file=sys.stderr,
            )
        try:
            backend = build_backend(args.backend, **backend_kwargs)
        except Exception as exc:
            print(f"could not start the {args.backend} backend: {exc}", file=sys.stderr)
            return 2

        agent_class = ECGDiagnosticAgent if args.mode == "diagnose" else ECGAgent
        agent = agent_class(
            # ECGDiagnosticAgent captures a bounded rule second opinion before
            # constructing its own diagnosis-safe measurement view. Manual
            # diagnosis tools above still use the scrubbed registry/store.
            store=(
                agent_source_store
                if args.mode == "diagnose"
                else agent_store
            ),
            backend=backend,
            registry=agent_registry,
            max_revisions=args.max_revisions,
            **(
                {
                    "knowledge_challenge": args.knowledge_challenge,
                    "knowledge_max_chunks": max(
                        1,
                        min(
                            args.knowledge_max_chunks,
                            DEFAULT_DIAGNOSTIC_PROTOCOL.knowledge.challenge_ceiling,
                        ),
                    ),
                    "workflow": args.diagnostic_workflow,
                }
                if args.mode == "diagnose"
                else {}
            ),
            on_event=None if args.quiet else lambda line: print(line, file=sys.stderr, flush=True),
        )
        result = agent.run()
        # The model and its trace saw only the ephemeral pseudonym. Restore the
        # local identifier before rendering/saving user-facing artifacts.
        result.record_id = true_record_id
        result.audit["record_identity"] = record_identity
        result.audit["external_egress"] = egress_policy.audit(
            allowed=bool(args.allow_external_egress)
        )
        if args.save_brief_report:
            from .trace import trace_filename

            default_name = trace_filename(result.record_id).replace(
                "_agent_trace.md",
                "_brief.md",
            )
            brief_report_path = args.brief_report_md or (
                args.features.parent / "brief_reports" / default_name
            )
            brief_report_path.parent.mkdir(parents=True, exist_ok=True)
            brief_report_path.write_text(result.brief_report, encoding="utf-8")
            if not args.quiet:
                print(
                    f"[brief-report] saved {brief_report_path}",
                    file=sys.stderr,
                    flush=True,
                )
        if args.save_trace:
            from .trace import trace_filename

            trace_path = args.trace_md or (
                args.features.parent
                / "agent_traces"
                / trace_filename(result.record_id)
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(result.trace_report, encoding="utf-8")
            if not args.quiet:
                print(
                    f"[trace] saved {trace_path}",
                    file=sys.stderr,
                    flush=True,
                )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
        else:
            if args.mode == "diagnose" and result.verdict is not None:
                print(result.human_report)
            else:
                print(result.summary())
            if result.verification and not result.verification.passed:
                print("\n" + result.verification.summary())
        return 0 if result.ok and result.verified else 1

    if args.audit:
        if not args.audit.exists():
            parser.error(f"audit file not found: {args.audit}")
        report = audit_untraceable_numbers(args.audit.read_text(encoding="utf-8"), store)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.summary())
            detail = report.detail()
            if detail:
                print(detail)
        return 0 if not report.unsourced else 1

    if args.verify:
        if not args.verify.exists():
            parser.error(f"verify file not found: {args.verify}")
        # No tool session backs a file read from disk, so provenance is not
        # checkable here; the resolvability, consistency and qualification
        # checks still apply.
        report = verify_output(
            args.verify.read_text(encoding="utf-8"),
            store,
            whitelist=None,
            policy=VerificationPolicy(require_provenance=False),
        )
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.summary())
            for finding in report.findings:
                print("  " + finding.render())
            if not report.passed:
                print("\n" + report.feedback())
        return 0 if report.passed else 1

    if args.briefing:
        briefing = (
            build_diagnostic_briefing(store)
            if args.mode == "diagnose"
            else build_chart_briefing(store)
        )
        if args.json:
            print(json.dumps({"text": briefing.text, "citations": list(briefing.citations)},
                             ensure_ascii=False, indent=2))
        else:
            print(briefing.text)
        if not args.tool:
            return 0
        print()

    if args.tool:
        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError as exc:
            parser.error(f"--args is not valid JSON: {exc}")
        if not isinstance(tool_args, dict):
            parser.error("--args must be a JSON object")
        result = registry.call(args.tool, tool_args)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(result.render())
        return 0 if result.ok else 1

    parser.error(
        "nothing to do: pass --agent, --briefing, --tool, --demo, --list-tools, "
        "--schema, --verify or --audit"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
