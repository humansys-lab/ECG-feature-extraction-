#!/usr/bin/env python3
"""Analyze two ECGAgent batches on an exactly paired record cohort.

The official batch analysis deliberately scores every verified result available
in one run.  This helper adds an apples-to-apples comparison for regression
work: only records that verified in both runs are used for the paired metrics,
and reference-positive instances are followed through the compact candidate
and adjudication funnel.  PTB-XL labels are used only after inference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_target_ecgfeat_diagnosis import CATEGORIES


DIRECT_SPECS = tuple(spec for spec in CATEGORIES if spec.semantic_scope == "direct")
DIRECT_KEYS = frozenset(spec.key for spec in DIRECT_SPECS)
QRS_FAMILIES = frozenset(
    {
        "right_bundle_branch_block",
        "left_bundle_branch_block",
        "left_anterior_fascicular_block",
        "left_posterior_fascicular_block",
        "nonspecific_ivcd",
        "right_ventricular_hypertrophy",
    }
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _categories(codes: Iterable[str], *, reference: bool) -> set[str]:
    values = {str(code) for code in codes if code}
    return {
        spec.key
        for spec in CATEGORIES
        if values
        & set(spec.ptbxl_refs if reference else spec.prediction_codes)
    }


def _diagnosis_codes(payload: Mapping[str, Any]) -> set[str]:
    verdict = payload.get("verdict")
    verdict = verdict if isinstance(verdict, Mapping) else {}
    return {
        str(row.get("code"))
        for row in verdict.get("diagnoses") or []
        if isinstance(row, Mapping)
        and row.get("code")
        and str(row.get("status") or "diagnosed") != "withdrawn"
    }


def _load_batch(directory: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted((directory / "diagnoses").glob("*.json")):
        try:
            payload = _read(path)
        except (OSError, ValueError, TypeError):
            continue
        output[str(payload.get("record_id") or path.stem)] = payload
    return output


def _manifest_records(directory: Path) -> list[str]:
    manifest = _read(directory / "record_manifest.json")
    return [str(row["record"]) for row in manifest.get("records") or []]


def _reference_categories(payload: Mapping[str, Any]) -> set[str]:
    reference = payload.get("reference")
    reference = reference if isinstance(reference, Mapping) else {}
    return _categories(reference.get("codes_active") or [], reference=True)


def _baseline_categories(payload: Mapping[str, Any]) -> set[str]:
    source = payload.get("source")
    source = source if isinstance(source, Mapping) else {}
    return _categories(source.get("baseline_codes") or [], reference=False)


def _agent_categories(payload: Mapping[str, Any]) -> set[str]:
    return _categories(_diagnosis_codes(payload), reference=False)


def _differential_categories(payload: Mapping[str, Any]) -> set[str]:
    verdict = payload.get("verdict")
    verdict = verdict if isinstance(verdict, Mapping) else {}
    codes = {
        str(row.get("code") or "")
        for row in verdict.get("differential_diagnoses") or []
        if isinstance(row, Mapping)
    }
    return _categories(codes, reference=False)


def _score(
    rows: Sequence[tuple[set[str], set[str]]],
    *,
    specs: Sequence[Any] = DIRECT_SPECS,
) -> dict[str, Any]:
    tp = fp = fn = 0
    for truth, predicted in rows:
        for spec in specs:
            actual = spec.key in truth
            called = spec.key in predicted
            if actual and called:
                tp += 1
            elif called:
                fp += 1
            elif actual:
                fn += 1
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


def _per_family(
    truths: Mapping[str, set[str]],
    predictions: Mapping[str, Mapping[str, set[str]]],
    record_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in DIRECT_SPECS:
        row: dict[str, Any] = {
            "family": spec.key,
            "reference_positive": sum(
                spec.key in truths[record_id] for record_id in record_ids
            ),
        }
        for source, values in predictions.items():
            metric = _score(
                [(truths[rid], values[rid]) for rid in record_ids],
                specs=(spec,),
            )
            row[source] = metric
        rows.append(row)
    return rows


def _effect_counts(
    truths: Mapping[str, set[str]],
    baseline: Mapping[str, set[str]],
    candidate: Mapping[str, set[str]],
    record_ids: Sequence[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record_id in record_ids:
        baseline_distance = len(truths[record_id] ^ baseline[record_id])
        candidate_distance = len(truths[record_id] ^ candidate[record_id])
        counts[
            "improved"
            if candidate_distance < baseline_distance
            else "worsened"
            if candidate_distance > baseline_distance
            else "unchanged"
        ] += 1
    return dict(counts)


def _bootstrap_f1_delta(
    truths: Mapping[str, set[str]],
    left: Mapping[str, set[str]],
    right: Mapping[str, set[str]],
    record_ids: Sequence[str],
    *,
    iterations: int = 10_000,
    seed: int = 20_260_813,
) -> dict[str, Any]:
    """Record-bootstrap the micro-F1 difference, left minus right."""

    if not record_ids:
        return {"iterations": 0, "seed": seed}
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(record_ids) for _ in record_ids]
        left_metric = _score([(truths[rid], left[rid]) for rid in sampled])
        right_metric = _score([(truths[rid], right[rid]) for rid in sampled])
        deltas.append(float(left_metric["f1"] or 0.0) - float(right_metric["f1"] or 0.0))
    deltas.sort()
    lower_index = max(0, math.ceil(0.025 * iterations) - 1)
    upper_index = max(0, math.ceil(0.975 * iterations) - 1)
    return {
        "iterations": iterations,
        "seed": seed,
        "mean_delta": statistics.fmean(deltas),
        "percentile_95_interval": [
            deltas[lower_index],
            deltas[upper_index],
        ],
        "fraction_delta_le_zero": sum(value <= 0 for value in deltas)
        / iterations,
    }


def _rule_transitions(
    truths: Mapping[str, set[str]],
    baseline: Mapping[str, set[str]],
    agent: Mapping[str, set[str]],
    record_ids: Sequence[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record_id in record_ids:
        truth = truths[record_id]
        rule = baseline[record_id]
        result = agent[record_id]
        for family in DIRECT_KEYS:
            actual = family in truth
            before = family in rule
            after = family in result
            if before and actual:
                counts["rule_tp_retained" if after else "rule_tp_removed"] += 1
            elif before and not actual:
                counts["rule_fp_retained" if after else "rule_fp_removed"] += 1
            elif after and actual:
                counts["new_agent_tp"] += 1
            elif after:
                counts["new_agent_fp"] += 1
    return dict(counts)


def _decisions(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    audit = payload.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    compact = audit.get("compact_decision")
    compact = compact if isinstance(compact, Mapping) else {}
    return [
        row for row in compact.get("decisions") or [] if isinstance(row, Mapping)
    ]


def _decision_families(decision: Mapping[str, Any]) -> set[str]:
    return _categories({str(decision.get("code") or "")}, reference=False)


def _generated_families(payload: Mapping[str, Any]) -> set[str]:
    """Return families raised before final pathway placement.

    The compact plan contains routed candidates.  The rule audit additionally
    retains candidates deferred by the candidate cap or dropped by the exact
    view budget, which lets the FN funnel distinguish generation from routing.
    """

    audit = payload.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    compact_plan = audit.get("compact_plan")
    compact_plan = compact_plan if isinstance(compact_plan, Mapping) else {}
    rule = audit.get("rule_second_opinion")
    rule = rule if isinstance(rule, Mapping) else {}
    merge = rule.get("merge")
    merge = merge if isinstance(merge, Mapping) else {}
    codes = {
        str(row.get("code") or "")
        for row in compact_plan.get("candidates") or []
        if isinstance(row, Mapping)
    }
    codes.update(
        str(row.get("code") or "")
        for row in rule.get("candidate_rows") or []
        if isinstance(row, Mapping)
    )
    codes.update(str(code) for code in merge.get("rule_deferred_by_cap") or [])
    codes.update(
        str(row.get("code") or "")
        for row in merge.get("candidates_dropped_by_view_budget") or []
        if isinstance(row, Mapping)
    )
    return _categories(codes, reference=False)


def _funnel(
    payloads: Mapping[str, Mapping[str, Any]],
    truths: Mapping[str, set[str]],
    record_ids: Sequence[str],
) -> dict[str, Any]:
    overall: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = {
        spec.key: Counter() for spec in DIRECT_SPECS
    }
    rank = {"rejected": 1, "unresolved": 2, "confirmed": 3}
    for record_id in record_ids:
        placements: dict[str, str] = {}
        for decision in _decisions(payloads[record_id]):
            placement = str(decision.get("final_placement") or "unresolved")
            for family in _decision_families(decision) & DIRECT_KEYS:
                if rank.get(placement, 0) > rank.get(placements.get(family, ""), 0):
                    placements[family] = placement
        confirmed = _agent_categories(payloads[record_id])
        generated = _generated_families(payloads[record_id])
        for family in truths[record_id] & DIRECT_KEYS:
            stage = (
                "confirmed"
                if family in confirmed
                else placements.get(family)
                or (
                    "generated_not_routed"
                    if family in generated
                    else "not_generated"
                )
            )
            overall[stage] += 1
            by_family[family][stage] += 1
    return {
        "overall": dict(overall),
        "by_family": {
            key: dict(value)
            for key, value in by_family.items()
            if sum(value.values())
        },
    }


def _decision_level_metrics(
    payloads: Mapping[str, Mapping[str, Any]],
    truths: Mapping[str, set[str]],
    record_ids: Sequence[str],
) -> dict[str, Any]:
    predictions: dict[str, dict[str, set[str]]] = {
        "confirmed": {},
        "confirmed_or_differential": {},
        "all_routed_candidates": {},
        "all_generated_candidates": {},
    }
    for record_id in record_ids:
        confirmed = _agent_categories(payloads[record_id])
        differential = _differential_categories(payloads[record_id])
        routed = set().union(
            *(
                _decision_families(decision)
                for decision in _decisions(payloads[record_id])
            ),
            set(),
        )
        predictions["confirmed"][record_id] = confirmed
        predictions["confirmed_or_differential"][record_id] = (
            confirmed | differential
        )
        predictions["all_routed_candidates"][record_id] = routed
        predictions["all_generated_candidates"][record_id] = (
            _generated_families(payloads[record_id])
        )
    return {
        level: _score(
            [(truths[record_id], values[record_id]) for record_id in record_ids]
        )
        for level, values in predictions.items()
    }


def _routing_pressure(
    payloads: Mapping[str, Mapping[str, Any]], record_ids: Sequence[str]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    deferred_codes: Counter[str] = Counter()
    dropped_codes: Counter[str] = Counter()
    for record_id in record_ids:
        audit = payloads[record_id].get("audit")
        audit = audit if isinstance(audit, Mapping) else {}
        rule = audit.get("rule_second_opinion")
        rule = rule if isinstance(rule, Mapping) else {}
        merge = rule.get("merge")
        merge = merge if isinstance(merge, Mapping) else {}
        deferred = [
            str(code) for code in merge.get("rule_deferred_by_cap") or []
        ]
        dropped = [
            row
            for row in merge.get("candidates_dropped_by_view_budget") or []
            if isinstance(row, Mapping)
        ]
        if deferred:
            counts["records_with_cap_defer"] += 1
            deferred_codes.update(deferred)
        if dropped:
            counts["records_with_view_budget_drop"] += 1
            dropped_codes.update(str(row.get("code") or "") for row in dropped)
        counts["deferred_candidates"] += len(deferred)
        counts["view_budget_dropped_candidates"] += len(dropped)
        if int(merge.get("merged_candidate_count") or 0) >= int(
            merge.get("merged_candidate_cap") or 10**9
        ):
            counts["records_at_candidate_cap"] += 1
    return {
        **dict(counts),
        "deferred_codes": dict(deferred_codes.most_common()),
        "view_budget_dropped_codes": dict(dropped_codes.most_common()),
    }


def _pathway_stats(
    payloads: Mapping[str, Mapping[str, Any]], record_ids: Sequence[str]
) -> dict[str, Any]:
    placement: Counter[str] = Counter()
    status: Counter[str] = Counter()
    deterministic_status: Counter[str] = Counter()
    model_status: Counter[str] = Counter()
    model_candidate: Counter[str] = Counter()
    qrs_lead_pattern: Counter[str] = Counter()
    tool_calls = 0
    tool_elapsed_seconds = 0.0
    model_usage: Counter[str] = Counter()
    for record_id in record_ids:
        audit = payloads[record_id].get("audit")
        audit = audit if isinstance(audit, Mapping) else {}
        tools = audit.get("tools")
        tools = tools if isinstance(tools, Mapping) else {}
        tool_calls += int(tools.get("n_calls") or 0)
        tool_elapsed_seconds += sum(
            float(call.get("elapsed_ms") or 0.0) / 1000.0
            for call in tools.get("calls") or []
            if isinstance(call, Mapping)
        )
        model = audit.get("model")
        model = model if isinstance(model, Mapping) else {}
        for key, value in (model.get("usage") or {}).items():
            if isinstance(value, int):
                model_usage[key] += value
        for decision in _decisions(payloads[record_id]):
            placement[str(decision.get("final_placement") or "unknown")] += 1
            has_model_step = False
            for step in decision.get("pathway_steps") or []:
                if not isinstance(step, Mapping):
                    continue
                state = str(step.get("effective_status") or "unknown")
                owner = str(step.get("resolution_owner") or "unknown")
                status[state] += 1
                if owner.startswith("model_"):
                    has_model_step = True
                    model_status[state] += 1
                else:
                    deterministic_status[state] += 1
                if (
                    str(step.get("id") or "") == "required_lead_pattern"
                    and _decision_families(decision) & QRS_FAMILIES
                ):
                    qrs_lead_pattern[state] += 1
            if has_model_step:
                model_candidate[
                    "decision_present"
                    if decision.get("model_decision_present")
                    else "decision_missing"
                ] += 1
    return {
        "placements": dict(placement),
        "step_status": dict(status),
        "deterministic_step_status": dict(deterministic_status),
        "model_owned_step_status": dict(model_status),
        "model_candidate_completeness": dict(model_candidate),
        "qrs_required_lead_pattern_status": dict(qrs_lead_pattern),
        "tool_calls": tool_calls,
        "tool_elapsed_seconds": tool_elapsed_seconds,
        "model_usage": dict(model_usage),
    }


def _run_configs(
    payloads: Mapping[str, Mapping[str, Any]], record_ids: Sequence[str]
) -> list[dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for record_id in record_ids:
        payload = payloads[record_id]
        audit = payload.get("audit")
        audit = audit if isinstance(audit, Mapping) else {}
        model = audit.get("model")
        model = model if isinstance(model, Mapping) else {}
        config = model.get("config")
        config = config if isinstance(config, Mapping) else {}
        phase_config = [
            phase
            for phase in audit.get("phase_config") or []
            if isinstance(phase, Mapping)
        ]
        adjudicate = next(
            (
                phase
                for phase in phase_config
                if str(phase.get("key") or "") == "adjudicate"
            ),
            {},
        )
        row = {
            "backend": (payload.get("batch") or {}).get("backend"),
            "model": config.get("model") or (payload.get("batch") or {}).get("model"),
            "thinking": config.get("thinking"),
            "diagnostic_workflow": (payload.get("batch") or {}).get(
                "diagnostic_workflow"
            ),
            "agent_protocol": audit.get("agent_protocol"),
            "prompt_fingerprint": audit.get("prompt_fingerprint"),
            "tool_result_contract": (
                config.get("context_compaction") or {}
            ).get("tool_result_contract"),
            "adjudicate_max_tokens": adjudicate.get("max_tokens"),
        }
        key = json.dumps(row, sort_keys=True, default=str)
        configs.setdefault(key, {**row, "records": 0})["records"] += 1
    return list(configs.values())


def _tool_visibility(
    payloads: Mapping[str, Mapping[str, Any]], record_ids: Sequence[str]
) -> dict[str, Any]:
    by_tool: dict[str, Counter[str]] = {}
    for record_id in record_ids:
        audit = payloads[record_id].get("audit")
        audit = audit if isinstance(audit, Mapping) else {}
        tools = audit.get("tools")
        tools = tools if isinstance(tools, Mapping) else {}
        for call in tools.get("calls") or []:
            if not isinstance(call, Mapping) or not call.get("ok"):
                continue
            name = str(call.get("tool") or "unknown")
            manifest = call.get("citation_manifest")
            manifest = manifest if isinstance(manifest, Mapping) else {}
            row = by_tool.setdefault(name, Counter())
            row["calls"] += 1
            row["touched"] += int(manifest.get("touched_count") or 0)
            row["visible"] += int(manifest.get("visible_count") or 0)
            row["hidden"] += int(manifest.get("hidden_count") or 0)
    rendered = {name: dict(counts) for name, counts in sorted(by_tool.items())}
    total: Counter[str] = Counter()
    for counts in by_tool.values():
        total.update(counts)
    return {"total": dict(total), "by_tool": rendered}


def _af_false_positive_paths(
    payloads: Mapping[str, Mapping[str, Any]],
    truths: Mapping[str, set[str]],
    record_ids: Sequence[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    step_counts: dict[str, Counter[str]] = {}
    for record_id in record_ids:
        if "atrial_fibrillation" in truths[record_id]:
            continue
        if "atrial_fibrillation" not in _agent_categories(payloads[record_id]):
            continue
        for decision in _decisions(payloads[record_id]):
            if "atrial_fibrillation" not in _decision_families(decision):
                continue
            steps = {
                str(step.get("id") or "unknown"): str(
                    step.get("effective_status") or "unknown"
                )
                for step in decision.get("pathway_steps") or []
                if isinstance(step, Mapping)
            }
            for step_id, status in steps.items():
                step_counts.setdefault(step_id, Counter())[status] += 1
            rows.append(
                {
                    "record": record_id,
                    "model_decision_present": bool(
                        decision.get("model_decision_present")
                    ),
                    "steps": steps,
                }
            )
    return {
        "count": len(rows),
        "step_status": {
            key: dict(value) for key, value in sorted(step_counts.items())
        },
        "records": rows,
    }


def _runtime(
    payloads: Mapping[str, Mapping[str, Any]], record_ids: Sequence[str]
) -> dict[str, Any]:
    values = [
        float(runtime)
        for record_id in record_ids
        if isinstance(
            runtime := (payloads[record_id].get("batch") or {}).get(
                "runtime_seconds"
            ),
            (int, float),
        )
        and math.isfinite(float(runtime))
    ]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Earlier batch directory")
    parser.add_argument("candidate", type=Path, help="New batch directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally save the JSON analysis to this path",
    )
    args = parser.parse_args()

    old = _load_batch(args.baseline)
    new = _load_batch(args.candidate)
    cohort = _manifest_records(args.candidate)
    new_verified = [rid for rid in cohort if rid in new and new[rid].get("verified")]
    paired = [
        rid
        for rid in cohort
        if rid in old
        and rid in new
        and old[rid].get("verified")
        and new[rid].get("verified")
    ]

    truths = {rid: _reference_categories(new[rid]) for rid in new_verified}
    baseline = {rid: _baseline_categories(new[rid]) for rid in new_verified}
    new_agent = {rid: _agent_categories(new[rid]) for rid in new_verified}
    old_agent = {rid: _agent_categories(old[rid]) for rid in paired}

    new_metrics = {
        "baseline": _score([(truths[rid], baseline[rid]) for rid in new_verified]),
        "agent": _score([(truths[rid], new_agent[rid]) for rid in new_verified]),
    }
    paired_predictions = {
        "baseline": {rid: baseline[rid] for rid in paired},
        "old_agent": old_agent,
        "new_agent": {rid: new_agent[rid] for rid in paired},
    }
    paired_metrics = {
        source: _score([(truths[rid], values[rid]) for rid in paired])
        for source, values in paired_predictions.items()
    }

    output = {
        "cohort": {
            "requested": len(cohort),
            "new_result_files": sum(rid in new for rid in cohort),
            "new_verified": len(new_verified),
            "old_and_new_verified": len(paired),
            "new_failed_or_missing": [
                rid
                for rid in cohort
                if rid not in new or not new[rid].get("verified")
            ],
        },
        "new_verified_metrics": new_metrics,
        "strict_paired_metrics": paired_metrics,
        "bootstrap_f1_deltas": {
            "new_agent_minus_baseline_all_new_verified": _bootstrap_f1_delta(
                truths, new_agent, baseline, new_verified
            ),
            "new_agent_minus_old_agent_strict_pair": _bootstrap_f1_delta(
                truths,
                {rid: new_agent[rid] for rid in paired},
                old_agent,
                paired,
            ),
        },
        "new_verified_family_metrics": _per_family(
            truths,
            {"baseline": baseline, "agent": new_agent},
            new_verified,
        ),
        "strict_paired_family_metrics": _per_family(
            truths, paired_predictions, paired
        ),
        "new_vs_baseline_record_effect": _effect_counts(
            truths, baseline, new_agent, new_verified
        ),
        "old_vs_baseline_record_effect_on_pair": _effect_counts(
            truths, baseline, old_agent, paired
        ),
        "new_vs_old_record_effect_on_pair": _effect_counts(
            truths,
            old_agent,
            {rid: new_agent[rid] for rid in paired},
            paired,
        ),
        "new_vs_baseline_rule_transitions": _rule_transitions(
            truths, baseline, new_agent, new_verified
        ),
        "old_vs_baseline_rule_transitions_on_pair": _rule_transitions(
            truths, baseline, old_agent, paired
        ),
        "new_reference_positive_funnel": _funnel(
            new, truths, new_verified
        ),
        "old_reference_positive_funnel_on_pair": _funnel(old, truths, paired),
        "new_decision_level_metrics": _decision_level_metrics(
            new, truths, new_verified
        ),
        "old_decision_level_metrics_on_pair": _decision_level_metrics(
            old, truths, paired
        ),
        "new_routing_pressure": _routing_pressure(new, new_verified),
        "old_routing_pressure_on_pair": _routing_pressure(old, paired),
        "new_pathways": _pathway_stats(new, new_verified),
        "new_pathways_on_pair": _pathway_stats(new, paired),
        "old_pathways_on_pair": _pathway_stats(old, paired),
        "new_tool_visibility": _tool_visibility(new, new_verified),
        "old_tool_visibility_on_pair": _tool_visibility(old, paired),
        "new_af_false_positive_paths": _af_false_positive_paths(
            new, truths, new_verified
        ),
        "runtime_seconds": {
            "new": _runtime(new, new_verified),
            "new_on_pair": _runtime(new, paired),
            "old_on_pair": _runtime(old, paired),
        },
        "run_configs": {
            "new": _run_configs(new, new_verified),
            "old_on_pair": _run_configs(old, paired),
        },
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
