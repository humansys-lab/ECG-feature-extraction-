"""Bounded evidence batches for the compact diagnostic workflow.

The blind plan is preserved. Each adjudication has its own exact schema and
visible evidence ledger; at most one evidence-triggered candidate update is
allowed. Public responses remain in response_history, including local repairs.
"""
from __future__ import annotations

import copy
import json
import math
import time
from typing import Any, Mapping

from . import compact_diagnostic_prompts
from .diagnosis_catalog import DIAGNOSIS_CATALOG
from .diagnostic_pathways import attach_waveform_review, build_diagnostic_pathway, semantic_candidate_family
from .loop import PhaseRecord
from .protocol import DEFAULT_DIAGNOSTIC_PROTOCOL


def evidence_dependency(pointer: str) -> str:
    """Processing families, explicitly not counts of independent acquisitions."""
    if pointer.startswith("/waveform_review/"):
        return "raw_waveform_review_same_acquisition"
    if pointer.startswith(("/p_wave_assessments/", "/rhythm_inputs/p_events/")) or any(
        key in pointer for key in ("/pr_ms", "/p_axis", "/p_amp", "/atrial_rate")
    ):
        return "atrial_detection_and_delineation"
    if pointer.startswith("/rhythm_inputs/pacing/") or pointer.endswith("/paced"):
        return "pacing_detection"
    if "/af_afl/" in pointer:
        return "atrial_residual_detector"
    if pointer.startswith("/groups/"):
        return "qrs_detection_and_grouping"
    if any(key in pointer for key in ("/qt", "/t_", "/st_", "/u_")):
        return "repolarization_measurement"
    if pointer.startswith(("/beats/", "/rhythm_inputs/background/")) or any(
        key in pointer for key in ("/heart_rate", "/rr_")
    ):
        return "ventricular_detection_and_timing"
    return "ecgfeat_morphology_measurement"


def assess_evidence_strength(store, evidence, steps) -> dict[str, Any]:
    pointers = sorted({str(token).removeprefix("ev:") for item in evidence
                       for token in item.get("citations", [])})
    values = [store.try_resolve(pointer) for pointer in pointers]
    limited = [p for p, value in zip(pointers, values)
               if value is None or value.is_null or not value.reliable]
    unknown = [str(row["id"]) for row in steps
               if row.get("gate") == "invalidator" and row.get("effective_status") == "unknown"]
    return {
        "policy": "uncalibrated_evidence_strength_v1",
        "confidence": "MEDIUM",  # No clinical calibration supports HIGH yet.
        "diagnostic_probability": None,
        "calibrated": False,
        "processing_families": sorted({evidence_dependency(p) for p in pointers}),
        "independent_acquisitions_established": False,
        "limited_or_missing_pointers": limited,
        "unknown_applicability_nodes": unknown,
        "required_nodes_complete": all(row.get("effective_status") == "pass"
                                       for row in steps if row.get("gate") == "required"),
    }


class CompactExecutionMixin:
    def _pathway_with_raw_review(self, pathway):
        profiles = self.store.raw("/waveform_review/profiles", {})
        profiles = profiles if isinstance(profiles, Mapping) else {}
        relevant = set()
        for step in pathway.get("steps", []):
            tool, args = step.get("tool"), step.get("arguments") or {}
            fields = set(args.get("fields") or [])
            if tool in {"get_p_assessment_table", "get_atrial_event_table"} or (
                tool == "get_interval_waveform_context" and args.get("interval") == "pr"):
                relevant.add("p_av")
            if tool == "get_qrs_measurement_bundle" or (
                tool == "get_morphology_map" and args.get("profile") == "qrs") or (
                tool == "get_lead_table" and fields & {"r_amp_mv", "s_amp_mv", "q_amp_mv"}):
                relevant.add("qrs")
            if (tool == "get_morphology_map" and args.get("profile") == "st") or any(
                name.startswith("st_") for name in fields):
                relevant.add("st")
        for profile in sorted(relevant):
            if profiles.get(profile, {}).get("status") == "conflict":
                pathway = attach_waveform_review(pathway, profile)
        return pathway

    def _schedule_compact_candidates(self, candidates):
        """Retain complete paths across bounded batches instead of one packet."""
        policy = DEFAULT_DIAGNOSTIC_PROTOCOL.phases
        pool = list(candidates)
        batches, deferred = [], []
        retained_count = 0
        while pool and len(batches) < policy.compact_adjudication_batches:
            feasible, _ = self._compact_bound_candidates_by_view_budget(pool)
            if not feasible:
                break
            batch, nodes = [], 0
            for row in feasible:
                model_nodes = sum(step.get("owner", "model") != "program"
                                  for step in row.get("diagnostic_pathway", {}).get("steps", []))
                if batch and nodes + model_nodes > policy.compact_model_nodes_per_batch:
                    continue
                if retained_count >= compact_diagnostic_prompts.MERGED_CANDIDATE_MAX:
                    break
                batch.append(row)
                nodes += model_nodes
                retained_count += 1
            if not batch:
                break
            batches.append(batch)
            selected = {row["id"] for row in batch}
            pool = [row for row in pool if row["id"] not in selected]
        for row in pool:
            deferred.append({"id": row["id"], "code": row["code"],
                             "sources": list(row.get("sources", [])),
                             "reason": "bounded_adjudication_capacity",
                             "required_new_views": len(self._compact_candidate_pathway_calls(row))})
        self._compact_batches = [[row["id"] for row in batch] for batch in batches]
        return [row for batch in batches for row in batch], deferred

    def _evidence_update_candidates(self, proposals, plan, visible):
        """One governed update. A proposal is routing, never positive evidence."""
        existing_codes = {row["code"] for row in plan["candidates"]}
        existing_families = {semantic_candidate_family(code) for code in existing_codes}
        accepted, audit = [], []
        for proposal in proposals:
            code = str(proposal.get("code") or "")
            reason = None
            if code not in DIAGNOSIS_CATALOG or code not in compact_diagnostic_prompts._HYPOTHESIS_CODES:
                reason = "unknown_diagnostic_code"
            elif code in existing_codes or semantic_candidate_family(code) in existing_families:
                reason = "already_planned_family"
            pointers = []
            for token in proposal.get("support", []):
                pointer = self._compact_pointer(token)
                value = self.store.try_resolve(pointer) if pointer else None
                if pointer in visible and value is not None and not value.is_null and value.reliable:
                    pointers.append(pointer)
            if not pointers:
                reason = reason or "no_new_visible_reliable_measurement"
            if len(accepted) >= DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_candidate_updates:
                reason = reason or "candidate_update_cap"
            if reason:
                audit.append({"code": code, "status": "deferred", "reason": reason})
                continue
            pathway = self._pathway_with_raw_review(
                build_diagnostic_pathway(code, program_owned=self._compact_adult()))
            row = {
                "id": f"u{len(accepted) + 1}_{code}"[:32], "code": code,
                "domains": [], "support": ["ev:" + p for p in pointers[:4]],
                "counter": [], "checks": [], "sources": ["new_measurement_review"],
                "uncertainty": "Newly read measurements require a separate diagnostic pathway.",
                "diagnostic_pathway": pathway,
            }
            # Never confirm through a generic catch-all with no definition.
            accepted.append(row)
            existing_codes.add(code)
            existing_families.add(semantic_candidate_family(code))
            audit.append({"code": code, "status": "raised", "pointers": pointers})
        self._compact_update_audit.extend(audit)
        return accepted

    def _measurement_update_proposals(self, plan, visible):
        """Measured rate phenotypes and conflicting ST/QRS views may open checks.

        These routing thresholds never establish a disease; every proposed code
        must complete the same fixed, auditable pathway as a model proposal.
        """
        codes = {row["code"] for row in plan["candidates"]}
        proposals = []
        aliases = self.backend.citation_aliases()
        by_pointer = {pointer: token for token, pointer in aliases.items()}
        if not self._compact_adult():
            return proposals
        def number(pointer):
            if pointer not in visible:
                return None
            value = self.store.try_resolve(pointer)
            raw = value.value if value is not None and value.reliable else None
            return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw) else None
        # A rejected elevation hypothesis must not hide a directly observed
        # opposite polarity. This only raises a depression check, not ischemia.
        if "st_elevation" in codes and "st_depression" not in codes:
            negative = [pointer for pointer in visible if pointer.endswith(("/st_hybrid_j_mv", "/st_j_mv"))
                        and (number(pointer) is not None and number(pointer) < 0)]
            if negative:
                proposals.append({"code": "st_depression", "support": [by_pointer[p] for p in sorted(negative) if p in by_pointer][:4]})
        bbb = {"rbbb_pattern", "lbbb_pattern", "right_bundle_branch_block", "left_bundle_branch_block"}
        if codes & bbb and "nonspecific_ivcd" not in codes:
            wide = [p for p in visible if p.endswith(("/mean_qrs_ms", "/qrs_ms"))
                    and number(p) is not None and number(p) >= 120]
            if wide:
                proposals.append({"code": "nonspecific_ivcd", "support": [by_pointer[p] for p in sorted(wide) if p in by_pointer][:2]})
        return proposals

    def _run_phase(self, spec, messages, tools):
        if not (self._compact_workflow and spec.key == "adjudicate"
                and self._backend_capabilities.orchestrated_prefetch):
            return super()._run_phase(spec, messages, tools)
        started = time.perf_counter()
        full_plan = copy.deepcopy(self._compact_plan)
        self._compact_executed_candidates = set()
        all_candidates = {row["id"]: row for row in full_plan["candidates"]}
        queue = [list(ids) for ids in self._compact_batches] or [list(all_candidates)]
        records, decisions, contexts, proposals = [], {}, {}, []
        # The exact active plan below replaces the full validated plan in
        # each request. The original remains in the audit and caller context.
        prefix = [message for message in messages if not str(message.get("content") or "").startswith(
            "PROGRAM-VALIDATED COMPACT PLAN.")]
        update_used = False
        retried = set()
        template = next(phase for phase in self.phases if phase.key == "adjudicate")
        cap = DEFAULT_DIAGNOSTIC_PROTOCOL.phases.compact_adjudication_batches
        try:
            while queue and len(records) < cap:
                ids = queue.pop(0)
                active = [all_candidates[candidate_id] for candidate_id in ids]
                self._compact_plan = {**full_plan, "candidates": active,
                                      "review_tools": full_plan.get("review_tools", []) if not records else []}
                self._compact_batch_index = len(records)
                self._compact_current_call_start = len(self.registry.calls)
                reset = getattr(self.backend, "begin_evidence_batch", None)
                if callable(reset):
                    reset("adjudicate")
                runtime = self._compact_runtime_phase_spec(template)
                if update_used:
                    runtime.response_schema["properties"]["candidate_updates"]["maxItems"] = 0
                batch_messages = list(prefix)
                batch_messages.append(self.backend.user_turn(
                    "ACTIVE ADJUDICATION BATCH. Only the following candidates belong to this response schema; "
                    "other planned candidates are assessed separately.\n" + json.dumps(
                        self._compact_model_plan(self._compact_plan), separators=(",", ":"))))
                record = super()._run_phase(runtime, batch_messages, tools)
                records.append(record)
                visible = {p for call in self.registry.calls[self._compact_current_call_start:]
                           if call.ok for p in call.visible_citations}
                for candidate_id in ids:
                    self._compact_candidate_visibility[candidate_id] = set(visible)
                for response in record.response_history:
                    response["adjudication_batch"] = len(records)
                    response["candidate_ids"] = list(ids)
                self._compact_batch_audit.append({
                    "batch": len(records), "candidate_ids": list(ids),
                    "tool_budget": runtime.tool_budget, "model_turns": record.turns,
                    "visible_atom_count": len(visible), "elapsed_seconds": record.elapsed_s,
                })
                if record.phase_guard_passed is False or record.stop_reason == "refusal":
                    break
                missing = self._unmet_runtime_coverage(runtime)
                if missing:
                    record.phase_guard_passed = False
                    record.phase_guard_problems.extend(missing)
                    break
                parsed = self._parse_verdict(record.text)
                if parsed is None:
                    record.phase_guard_passed = False
                    record.phase_guard_problems.append("batch response is not JSON")
                    break
                self._compact_batch_audit[-1]["model_overall_status"] = parsed.get("overall_status")
                self._compact_executed_candidates.update(ids)
                for row in parsed.get("decisions", []):
                    decisions[row["id"]] = row
                for context in parsed.get("interval_contexts", []):
                    contexts[context["interval"]] = context
                if not update_used:
                    # Bind each proposal to evidence in its originating batch;
                    # a later packet cannot retroactively license a guessed Qn.
                    for proposal in parsed.get("candidate_updates", []):
                        proposals.append({**proposal, "support": [
                            token for token in proposal.get("support", [])
                            if self._compact_pointer(token) in visible]})
                    proposals.extend(self._measurement_update_proposals(full_plan, visible))
                # A packet compression failure gets one smaller-batch attempt.
                # Unavailable source measurements are never retried as if a
                # second model request could create them.
                omitted = [candidate_id for candidate_id in ids if candidate_id not in retried
                           and len(ids) > 1 and any(
                               node.startswith(candidate_id + ":") and coverage["status"] == "omitted"
                               for node, coverage in self._compact_evidence_coverage.items())]
                for candidate_id in omitted:
                    queue.append([candidate_id])
                    retried.add(candidate_id)
                if not queue and not update_used and proposals and len(records) < cap:
                    all_visible = {p for values in self._compact_candidate_visibility.values() for p in values}
                    updates = self._evidence_update_candidates(proposals, full_plan, all_visible)
                    update_used = True
                    if updates:
                        # Keep bounded total candidates as well as bounded calls.
                        slots = compact_diagnostic_prompts.MERGED_CANDIDATE_MAX - len(all_candidates)
                        for row in updates[max(0, slots):]:
                            self._compact_deferred.append({"code": row["code"], "reason": "candidate_cap"})
                        updates = updates[:max(0, slots)]
                        for row in updates:
                            full_plan["candidates"].append(row)
                            all_candidates[row["id"]] = row
                            queue.append([row["id"]])
            if proposals and not update_used:
                for proposal in proposals:
                    self._compact_update_audit.append({"code": proposal.get("code"),
                        "status": "deferred", "reason": "record_batch_cap_or_failed_batch"})
            for ids in queue:
                for candidate_id in ids:
                    self._compact_deferred.append({"code": all_candidates[candidate_id]["code"],
                                                   "reason": "required_evidence_retry_cap" if candidate_id in self._compact_executed_candidates else "record_batch_cap"})
            if not records:
                raise RuntimeError("compact adjudication did not execute")
            merged = copy.deepcopy(records[0])
            merged.elapsed_s = time.perf_counter() - started
            for name in ("turns", "tool_calls", "rejected_calls", "prefetched_tool_calls",
                         "phase_guard_attempts", "tool_budget"):
                setattr(merged, name, sum(getattr(row, name) for row in records))
            merged.phase_guard_passed = all(record.phase_guard_passed is not False for record in records)
            merged.phase_guard_problems = [problem for record in records for problem in record.phase_guard_problems]
            merged.stop_reason = records[-1].stop_reason
            merged.response_history = [row for record in records for row in record.response_history]
            merged.phase_sanitizations = [row for record in records for row in record.phase_sanitizations]
            merged.text = json.dumps({"overall_status": "insufficient_evidence",
                                      "decisions": list(decisions.values()),
                                      "interval_contexts": list(contexts.values())}, separators=(",", ":"))
            messages.append(self.backend.user_turn(
                "All bounded diagnostic batches completed; original public responses are retained in the trace."))
            return merged
        finally:
            self._compact_plan = full_plan

