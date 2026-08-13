#!/usr/bin/env python3
"""Report who resolved each diagnostic pathway node, and how well.

The compact workflow's only model-facing degree of freedom is a pass/fail/
unknown label per pathway node, so node accounting is where a determinism
migration is won or lost.  `compare_gate_relaxation.py` scores the diagnoses
this produces; this script looks one level down, at the nodes themselves.

Three questions, one per section:

1. **Ownership** -- what share of nodes the program answers, and how the
   unknown rate differs between program-owned and model-owned nodes.  A node
   that stays `unknown` cannot confirm anything, so this is the direct measure
   of how much of the candidate pool the verification layer can actually
   resolve.
2. **Agreement** -- wherever both a `model_status` and a `program_status` are
   recorded, how often they agree.  Under
   `ECG_AGENT_DETERMINISTIC_NODES_SHADOW=1` the model is asked every node and
   this covers the whole pathway, which is how a node earns the right to be
   migrated.  Without it the model is not asked about program-owned nodes, so
   the sample is small and biased -- the script says so rather than quoting a
   misleading rate.
3. **Reason codes** -- which deterministic branch fired.  A branch that never
   fires is untested in production; one that always returns `unknown` is not
   yet paying for itself.

Usage::

    python3 analyze_deterministic_nodes.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

PROGRAM_OWNER = "deterministic_measurement_gate"
STATUSES = ("pass", "fail", "unknown")


def _walk_decisions(payload: Any) -> Iterator[Mapping[str, Any]]:
    """Yield every audit object that carries a pathway_steps list."""

    if isinstance(payload, Mapping):
        if isinstance(payload.get("pathway_steps"), list):
            yield payload
        for value in payload.values():
            yield from _walk_decisions(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_decisions(value)


def load_nodes(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((directory / "diagnoses").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for decision in _walk_decisions(payload):
            for step in decision["pathway_steps"]:
                if not isinstance(step, Mapping):
                    continue
                rows.append(
                    {
                        "record": path.stem,
                        "code": str(decision.get("code") or ""),
                        "placement": str(decision.get("final_placement") or ""),
                        "id": str(step.get("id") or ""),
                        "gate": str(step.get("gate") or ""),
                        "owner": str(step.get("resolution_owner") or ""),
                        "status": str(step.get("effective_status") or ""),
                        "model_status": step.get("model_status"),
                        "program_status": step.get("program_status"),
                        "reason": step.get("deterministic_reason_code"),
                    }
                )
    return rows


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:5.1f}%" if whole else "    --"


def _status_counts(rows: Iterable[Mapping[str, Any]]) -> collections.Counter:
    return collections.Counter(str(row["status"]) for row in rows)


def report_ownership(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    print(f"\n--- node ownership ({total} node instances) ---")
    print(f"{'owner':<34}{'n':>6}{'share':>8}{'pass':>7}{'fail':>7}{'unknown':>9}{'unk rate':>10}")
    by_owner = collections.defaultdict(list)
    for row in rows:
        by_owner[row["owner"] or "(unrecorded)"].append(row)
    for owner, group in sorted(by_owner.items(), key=lambda item: -len(item[1])):
        counts = _status_counts(group)
        print(
            f"{owner:<34}{len(group):>6}{_pct(len(group), total):>8}"
            f"{counts['pass']:>7}{counts['fail']:>7}{counts['unknown']:>9}"
            f"{_pct(counts['unknown'], len(group)):>10}"
        )

    gates = collections.Counter(row["gate"] for row in rows)
    print(f"\ngate mix: {dict(gates)}")
    blocked = [
        row for row in rows if row["gate"] == "invalidator" and row["status"] == "fail"
    ]
    if blocked:
        print(f"invalidator vetoes fired: {len(blocked)}")
        for row in blocked:
            print(f"  {row['record']:<14}{row['code']:<26}{row['id']:<32}{row['reason']}")


def report_agreement(rows: list[dict[str, Any]]) -> None:
    paired = [
        row
        for row in rows
        if row["program_status"] in STATUSES and row["model_status"] in STATUSES
    ]
    print(f"\n--- model vs program agreement ({len(paired)} nodes with both answers) ---")
    if not paired:
        print("  no node carried both a model and a program answer.")
        return

    # Outside shadow mode the model is never shown a program-owned node's view,
    # so its label there is a forced `unknown` rather than a judgement. Saying
    # so is the difference between a measurement and a misleading number.
    forced = [
        row
        for row in paired
        if row["owner"] == PROGRAM_OWNER and row["model_status"] == "unknown"
    ]
    if len(forced) > 0.5 * len(paired):
        print(
            f"  NOTE: {len(forced)}/{len(paired)} pairs are program-owned nodes whose "
            "model label is a forced `unknown` (the model never saw the view).\n"
            "  Re-run with ECG_AGENT_DETERMINISTIC_NODES_SHADOW=1 for a real "
            "agreement rate."
        )

    agree = sum(1 for row in paired if row["model_status"] == row["program_status"])
    print(f"  agreement: {agree}/{len(paired)} = {_pct(agree, len(paired)).strip()}")
    transitions = collections.Counter(
        (str(row["model_status"]), str(row["program_status"])) for row in paired
    )
    print(f"  {'model -> program':<26}{'n':>6}")
    for (model, program), count in transitions.most_common():
        mark = "" if model == program else "   <- disagreement"
        print(f"  {model + ' -> ' + program:<26}{count:>6}{mark}")

    per_step = collections.defaultdict(lambda: [0, 0])
    for row in paired:
        cell = per_step[row["id"]]
        cell[0] += 1
        cell[1] += int(row["model_status"] == row["program_status"])
    disputed = sorted(
        ((step, n, ok) for step, (n, ok) in per_step.items() if ok < n),
        key=lambda item: (item[2] - item[1], -item[1]),
    )
    if disputed:
        print(f"\n  {'step':<36}{'n':>5}{'agree':>7}")
        for step, n, ok in disputed:
            print(f"  {step:<36}{n:>5}{ok:>7}")


def report_reason_codes(rows: list[dict[str, Any]]) -> None:
    program = [row for row in rows if row["owner"] == PROGRAM_OWNER and row["reason"]]
    print(f"\n--- deterministic reason codes ({len(program)} program resolutions) ---")
    counts = collections.Counter(
        (str(row["reason"]), str(row["status"])) for row in program
    )
    print(f"{'reason code':<52}{'status':<9}{'n':>5}")
    for (reason, status), count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"{reason:<52}{status:<9}{count:>5}")


def report_placements(rows: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        seen[(row["record"], row["code"])] = row["placement"]
    counts = collections.Counter(seen.values())
    print(f"\n--- candidate placements ({len(seen)} candidates) ---")
    print(f"  {dict(counts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()

    for directory in args.run_dirs:
        rows = load_nodes(directory)
        print("=" * 78)
        print(f"{directory}  ({len({row['record'] for row in rows})} records)")
        print("=" * 78)
        if not rows:
            print("  no pathway audit found.")
            continue
        report_ownership(rows)
        report_placements(rows)
        report_agreement(rows)
        report_reason_codes(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
