"""Command-line entry point: ``python -m benchmarks.harness {run,compare,list}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import HarnessError
from .runner import DEFAULT_CORPUS_ROOT, DEFAULT_PYTHON, DEFAULT_SPEC, load_spec, renormalize_command, run_command


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_dirs

    report = compare_dirs(Path(args.baseline), Path(args.candidate), args.tools)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    s = report["summary"]
    print(f"compared {s['tools_compared']} tools: passed={s['tools_passed']} failed={s['tools_failed']} "
          f"ungated={s['tools_ungated']}")
    print(f"regressions={s['regressions']} new_failures={s['new_failures']} exact_diffs={s['exact_diffs']} "
          f"missing_metrics={s['missing_metrics']} became_unavailable={s['became_unavailable']} "
          f"manifest_mismatches={s['manifest_mismatches']}")
    for tool, row in report["tools"].items():
        if row.get("passed") is False:
            print(f"  FAIL {tool}: {row['status']} {row.get('reason', '')}")
            for item in row.get("failures", [])[:20]:
                print(f"       {item['status']:<18} {item['metric']}: baseline={item.get('baseline')} "
                      f"candidate={item.get('candidate')} allowed={item.get('allowed_regression')}")
    if not args.report:
        print(text)
    return 0 if report["passed"] else 1


def _cmd_list(args: argparse.Namespace) -> int:
    spec = load_spec(Path(args.spec))
    for tool in spec["tools"]:
        print(f"{tool['id']:<36} {tool['status']:<10} workers={tool.get('workers', '-')} "
              f"est={tool.get('estimated_minutes', '-')}min")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.harness",
                                     description="ecgfeat evaluator benchmark harness (doc 06 release gate)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run pinned tool invocations and write normalized results")
    run.add_argument("--repo-root", required=True, help="tree under test (e.g. a git worktree at a pinned commit)")
    run.add_argument("--baseline-id", required=True, help="result-set id, e.g. phase0-52a339c")
    run.add_argument("--out-dir", default=None,
                     help="normalized results dir (default benchmarks/baselines/<id>; use another dir for candidates)")
    run.add_argument("--raw-dir", default=None, help="raw tool outputs (default benchmarks/_raw/<id>)")
    run.add_argument("--tools", nargs="+", default=None, help="subset of tool ids (default: all)")
    run.add_argument("--repeat", type=int, default=2, help="runs per tool for the determinism check (default 2)")
    run.add_argument("--max-workers", type=int, default=None, help="total worker-process budget (default spec: 8)")
    run.add_argument("--allow-more-workers", action="store_true")
    run.add_argument("--python", default=str(DEFAULT_PYTHON), help="interpreter used to run the tools")
    run.add_argument("--spec", default=str(DEFAULT_SPEC))
    run.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT),
                     help="fixed location of harness-built corpora (shared by baseline and candidate runs)")
    run.add_argument("--force", action="store_true", help="replace existing per-tool result files")
    run.set_defaults(func=run_command)

    ren = sub.add_parser("renormalize", help="re-derive normalized results from existing raw runs (tools not re-run)")
    ren.add_argument("--repo-root", required=True)
    ren.add_argument("--baseline-id", required=True)
    ren.add_argument("--out-dir", default=None)
    ren.add_argument("--raw-dir", default=None)
    ren.add_argument("--tools", nargs="+", default=None)
    ren.add_argument("--python", default=str(DEFAULT_PYTHON))
    ren.add_argument("--spec", default=str(DEFAULT_SPEC))
    ren.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    ren.set_defaults(func=renormalize_command)

    cmp_ = sub.add_parser("compare", help="apply frozen tolerances; exit 1 on any regression")
    cmp_.add_argument("--baseline", required=True)
    cmp_.add_argument("--candidate", required=True)
    cmp_.add_argument("--tools", nargs="+", default=None)
    cmp_.add_argument("--report", default=None, help="write the JSON report here")
    cmp_.set_defaults(func=_cmd_compare)

    lst = sub.add_parser("list", help="list pinned tools")
    lst.add_argument("--spec", default=str(DEFAULT_SPEC))
    lst.set_defaults(func=_cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except HarnessError as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
