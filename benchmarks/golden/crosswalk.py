"""``record-crosswalk`` comparator: frozen legacy payload vs. new ECG Record.

The rules come from ``schemas/ecg-record/1.0/crosswalk-legacy-v0.json`` and
are re-implemented here from that data.  This module deliberately imports
neither ``ecgfeat.pipeline.record_builder`` nor ``ecgfeat.compat.export_v0``
(document 05: the two parity gates must not share an oracle).  Mapped leaves
compare exactly; there are no tolerances.
"""

from __future__ import annotations

import json
import math
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

CROSSWALK_PATH = Path(__file__).resolve().parents[2] / "schemas" / "ecg-record" / "1.0" / "crosswalk-legacy-v0.json"
UNAVAILABLE, OUTSIDE, ORDER, NEGATIVE = ("measurement_unavailable", "sample_outside_acquisition",
                                         "fiducial_order_violation", "negative_interval")


def load_crosswalk() -> dict[str, Any]:
    return json.loads(CROSSWALK_PATH.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _half_even(value: float) -> int:
    # Python's round() is already half-even on binary floats; Decimal makes the
    # oracle independent of that implementation detail.
    return int(Decimal(repr(value)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def _member(cell: dict[str, Any], path: str) -> Any:
    node: Any = cell
    for token in path.split("/"):
        node = node.get(token) if isinstance(node, dict) else None
    return node


class _Context:
    def __init__(self, legacy: dict[str, Any], record: dict[str, Any]):
        self.legacy_fs = float(legacy["fs"])
        self.native_fs = float(record["acquisition"]["sample_rate_hz"])
        self.sample_count = int(record["acquisition"]["sample_count"])
        metadata = legacy.get("metadata", {})
        slot_map = (metadata.get("input_contract") or {}).get("lead_slot_map") \
            or (metadata.get("limited_lead_capabilities") or {}).get("channel_to_slot") or {}
        self.leads = list(record["axes"]["leads"])
        self.slots = [slot_map.get(lead, lead) for lead in self.leads]
        self.cells = {(int(c["beat_id"]), str(c["lead"])): c for c in legacy.get("beat_features", [])}
        self.beats = list(legacy.get("beats", []))
        self.limited = record["acquisition"]["input_mode"] == "limited"

    def transform(self, name: str, value: Any) -> tuple[int | None, str | None]:
        number = _finite(value)
        if number is None:
            return None, UNAVAILABLE
        if name == "native_sample":
            sample = _half_even(number * self.native_fs / self.legacy_fs)
            return (sample, None) if 0 <= sample < self.sample_count else (None, OUTSIDE)
        if name == "round_half_even":
            return _half_even(number), None
        if name == "mv_to_uv":
            return _half_even(number * 1000), None
        if name == "mv_internal_sample_to_uv_ms":
            return _half_even(number * 1_000_000 / self.legacy_fs), None
        raise ValueError(f"unknown crosswalk transform {name!r}")


def expected_dense(crosswalk: dict[str, Any], ctx: _Context, profile: str) -> dict[str, list[list[tuple[Any, Any]]]]:
    """Expected (value, reason) matrices keyed by record field name."""

    fields: dict[str, list[list[tuple[Any, Any]]]] = {}
    for rule in crosswalk["dense_fields"]:
        if profile not in rule["profiles"]:
            continue
        name = rule["record"].rsplit("/", 1)[1]
        matrix = []
        for beat in ctx.beats:
            row = []
            for slot in ctx.slots:
                cell = ctx.cells.get((int(beat["beat_id"]), slot))
                row.append(ctx.transform(rule["transform"], _member(cell, rule["legacy_cell_member"]))
                           if cell is not None else (None, UNAVAILABLE))
            matrix.append(row)
        fields[name] = matrix

    def value(name: str, i: int, j: int) -> Any:
        return fields[name][i][j][0]

    def withhold(name: str, i: int, j: int, reason: str) -> None:
        if name in fields and fields[name][i][j][0] is not None:
            fields[name][i][j] = (None, reason)

    for i in range(len(ctx.beats)):
        for j in range(len(ctx.slots)):
            if None not in (value("p_onset", i, j), value("p_offset", i, j)) and value("p_onset", i, j) > value("p_offset", i, j):
                for name in ("p_onset", "p_offset", "p_duration_ms"):
                    withhold(name, i, j, ORDER)
            onset, peak, offset = value("qrs_onset", i, j), value("r_peak", i, j), value("qrs_offset", i, j)
            if onset is not None and offset is not None and onset > offset:
                for name in ("qrs_onset", "qrs_offset", "qrs_duration_ms"):
                    withhold(name, i, j, ORDER)
            elif None not in (onset, peak, offset) and not onset <= peak <= offset:
                for name in ("qrs_onset", "r_peak", "qrs_offset", "qrs_duration_ms"):
                    withhold(name, i, j, ORDER)
    for rule in crosswalk["dense_fields"]:
        name = rule["record"].rsplit("/", 1)[1]
        if rule["record"].startswith("/measurements/intervals/") and name in fields:
            for i, row in enumerate(fields[name]):
                for j, (number, _) in enumerate(row):
                    if number is not None and number < 0:
                        withhold(name, i, j, NEGATIVE)
    return fields


def _record_cells(field: dict[str, Any], beat_ids: list[str], leads: list[str]) -> list[list[tuple[Any, Any]]]:
    absence = field.get("absence") or {}
    whole = absence.get("reason") if "kind" in absence else None
    sparse = absence.get("unmeasurable", {})
    default = (absence.get("default") or {}).get("reason")
    return [[(value, None if value is not None else (whole or sparse.get(f"{beat_ids[i]}|{leads[j]}") or default))
             for j, value in enumerate(row)] for i, row in enumerate(field["values"])]


def compare_with_crosswalk(legacy: dict[str, Any], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    crosswalk = load_crosswalk()
    differences: list[dict[str, Any]] = []
    for profile, record in records.items():
        ctx = _Context(legacy, record)
        beat_ids = [beat["id"] for beat in record["axes"]["beats"]]

        def diff(pointer: str, expected: Any, observed: Any) -> None:
            differences.append({"profile": profile, "pointer": pointer, "expected": expected, "observed": observed})

        if len(beat_ids) != len(ctx.beats):
            diff("/axes/beats", len(ctx.beats), len(beat_ids))
            continue
        for i, beat in enumerate(ctx.beats):
            want = ctx.transform("native_sample", beat.get("r_index"))[0]
            if record["axes"]["beats"][i].get("r_sample") != want:
                diff(f"/axes/beats/{i}/r_sample", want, record["axes"]["beats"][i].get("r_sample"))
        expected = expected_dense(crosswalk, ctx, profile)
        groups = {"fiducials": record["delineation"]["fiducials"], **record["measurements"]}
        for rule in crosswalk["dense_fields"]:
            group, name = rule["record"].split("/")[-2:]
            present = name in groups.get(group, {})
            if (profile in rule["profiles"]) != present:
                diff(rule["record"], "present" if profile in rule["profiles"] else "absent",
                     "present" if present else "absent")
                continue
            if not present:
                continue
            observed = _record_cells(groups[group][name], beat_ids, ctx.leads)
            for i, row in enumerate(expected[name]):
                for j, cell in enumerate(row):
                    if tuple(observed[i][j]) != tuple(cell):
                        diff(f"{rule['record']}/values/{i}/{j}", list(cell), list(observed[i][j]))
        global_fields = record["measurements"]["global"]
        for rule in crosswalk["scalar_fields"]:
            name = rule["record"].rsplit("/", 1)[1]
            present = name in global_fields
            if (profile in rule["profiles"]) != present:
                diff(rule["record"], profile in rule["profiles"], present)
                continue
            if not present:
                continue
            field = global_fields[name]
            node: Any = legacy
            for token in rule["legacy"].strip("/").split("/"):
                node = node.get(token) if isinstance(node, dict) else None
            if ctx.limited and "limited" in rule:
                want: tuple[Any, Any] = (None, rule["limited"])
                got: tuple[Any, Any] = (field["values"], field.get("absence"))
            else:
                number, reason = ctx.transform(rule["transform"], node)
                want = (number, None if number is not None else {"kind": "unmeasurable", "reason": reason})
                got = (field["values"], field.get("absence"))
            if want != got:
                diff(rule["record"], list(want), list(got))
        grade = (legacy.get("metadata", {}).get("record_quality") or {}).get("record_grade")
        if record["quality"]["record"].get("grade") != grade:
            diff("/quality/record/grade", grade, record["quality"]["record"].get("grade"))
        status = "usable" if grade in {"Q0", "Q1"} else "limited"
        if record["quality"]["record"].get("status") != status:
            diff("/quality/record/status", status, record["quality"]["record"].get("status"))
        for lead, slot in zip(ctx.leads, ctx.slots):
            reliable = bool((legacy.get("quality", {}).get(slot) or {}).get("reliable", False))
            want_q = {"status": "usable"} if reliable else {"status": "limited", "reason": "quality_gate"}
            if record["quality"]["leads"].get(lead) != want_q:
                diff(f"/quality/leads/{lead}", want_q, record["quality"]["leads"].get(lead))
        if float(record["provenance"]["internal_sample_rate_hz"]) != ctx.legacy_fs:
            diff("/provenance/internal_sample_rate_hz", ctx.legacy_fs, record["provenance"]["internal_sample_rate_hz"])
        mains = legacy.get("metadata", {}).get("mains_frequency_hz")
        if mains is not None and record["provenance"].get("mains_frequency_hz") != mains:
            diff("/provenance/mains_frequency_hz", mains, record["provenance"].get("mains_frequency_hz"))
        if profile == "debug":
            rule = next(r for r in crosswalk["other_mapped"] if r["record"].startswith("/debug/legacy_metadata"))
            metadata = legacy.get("metadata", {})
            want_debug = {key: metadata[key] for key in rule["keys"] if key in metadata}
            got_debug = json.loads(json.dumps(record.get("debug", {}).get("legacy_metadata", {})))
            for key, members in rule.get("call_dependent_members", {}).items():
                for side in (want_debug, got_debug):
                    if isinstance(side.get(key), dict):
                        side[key] = {k: v for k, v in side[key].items() if k not in members}
            if got_debug != want_debug:
                diff("/debug/legacy_metadata", want_debug, got_debug)
    return differences
