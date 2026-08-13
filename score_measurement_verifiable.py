#!/usr/bin/env python3
"""Score the measurement-derived diagnoses against measurements, not labels.

PTB-XL carries record-level labels chosen by a reporting cardiologist, and they
systematically omit findings that are true but not the point of the report --
a PR of 217 ms, a rate of 50 bpm, a QTc of 321 ms, uniformly low limb voltages.
Once the agent's threshold nodes became program-owned it started emitting
exactly those findings, so label agreement began charging it for being right.
The v38 run lost 0.09 F1 that way while 7 of its 15 "new false positives" were
arithmetically correct.

This script re-scores only the families whose definition *is* a measurement
comparison, using the record's own `*_features.json`:

    measurement-confirmed    the call satisfies its own definition
    measurement-contradicted the call violates its own definition -- a real error
    unlabelled-but-correct   confirmed by measurement, absent from the label set
    not-checkable            no reliable measurement, or a morphology family

`measurement-contradicted` is the number to drive to zero; it cannot be
explained away by label incompleteness.  Morphology and ischemia families are
deliberately out of scope -- their definitions are not a scalar comparison, so
labels remain the only available reference for them.

Usage::

    python3 score_measurement_verifiable.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

LIMB_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")
PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")

# Thresholds are the project's own, mirrored from
# feature_extraction/ecgfeat/clinical_rules/config.py and
# ecgagent/agent/safety_policy.py. They are duplicated rather than imported so a
# silent drift between engine and scorer shows up as a diff in this file.
PR_FIRST_DEGREE_MS = 200.0
PR_SHORT_MS = 120.0
HR_BRADY_BPM = 60.0
HR_TACHY_BPM = 100.0
QTC_PROLONGED_MALE_MS = 450.0
QTC_PROLONGED_FEMALE_MS = 460.0
QTC_MARKED_PROLONGED_MS = 480.0
QTC_SHORT_MS = 340.0
QTC_BORDERLINE_SHORT_MS = 390.0
LOW_VOLTAGE_LIMB_MV = 0.50
LOW_VOLTAGE_PRECORDIAL_MV = 1.00


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class Record:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.globals = payload.get("global_features") or {}
        self.leads = payload.get("representative_leads") or {}
        self.meta = (payload.get("metadata") or {}).get("patient_meta") or {}

    def g(self, name: str) -> float | None:
        return _number(self.globals.get(name))

    @property
    def female(self) -> bool:
        return str(self.meta.get("sex") or "").strip().lower() == "female"

    def qtc(self) -> float | None:
        for name in ("qtc_fridericia_ms", "qtc_bazett_ms"):
            value = self.g(name)
            if value is not None:
                return value
        return None

    def group_amplitudes(self, leads: tuple[str, ...]) -> dict[str, float]:
        out: dict[str, float] = {}
        for lead in leads:
            params = (self.leads.get(lead) or {}).get("params") or {}
            r = _number(params.get("r_amp_mv"))
            s = _number(params.get("s_amp_mv"))
            if r is None or s is None:
                continue
            out[lead] = abs(r) + abs(s)
        return out


def _all_below(amplitudes: Mapping[str, float], limit: float) -> bool | None:
    if len(amplitudes) < 4:
        return None
    return all(value < limit for value in amplitudes.values())


def _prolonged_qt(record: Record) -> bool | None:
    qtc = record.qtc()
    if qtc is None:
        return None
    limit = QTC_PROLONGED_FEMALE_MS if record.female else QTC_PROLONGED_MALE_MS
    return qtc > limit


def _axis_between(record: Record, low: float, high: float) -> bool | None:
    axis = record.g("qrs_axis_deg")
    return None if axis is None else low < axis < high


CHECKS: dict[str, Callable[[Record], bool | None]] = {
    "first_degree_av_block": lambda r: None
    if r.g("pr_ms") is None
    else r.g("pr_ms") > PR_FIRST_DEGREE_MS,
    "first_degree_av_delay": lambda r: None
    if r.g("pr_ms") is None
    else r.g("pr_ms") > PR_FIRST_DEGREE_MS,
    "possible_first_degree_av_delay": lambda r: None
    if r.g("pr_ms") is None
    else r.g("pr_ms") > PR_FIRST_DEGREE_MS,
    "short_pr_interval": lambda r: None
    if r.g("pr_ms") is None
    else r.g("pr_ms") < PR_SHORT_MS,
    "sinus_bradycardia": lambda r: None
    if r.g("heart_rate_bpm") is None
    else r.g("heart_rate_bpm") < HR_BRADY_BPM,
    "bradycardia": lambda r: None
    if r.g("heart_rate_bpm") is None
    else r.g("heart_rate_bpm") < HR_BRADY_BPM,
    "sinus_tachycardia": lambda r: None
    if r.g("heart_rate_bpm") is None
    else r.g("heart_rate_bpm") > HR_TACHY_BPM,
    "tachycardia": lambda r: None
    if r.g("heart_rate_bpm") is None
    else r.g("heart_rate_bpm") > HR_TACHY_BPM,
    "prolonged_qt": _prolonged_qt,
    "markedly_prolonged_qt": lambda r: None
    if r.qtc() is None
    else r.qtc() > QTC_MARKED_PROLONGED_MS,
    "short_qt": lambda r: None if r.qtc() is None else r.qtc() < QTC_SHORT_MS,
    "markedly_short_qt": lambda r: None
    if r.qtc() is None
    else r.qtc() < QTC_SHORT_MS,
    "borderline_short_qt": lambda r: None
    if r.qtc() is None
    else r.qtc() < QTC_BORDERLINE_SHORT_MS,
    "possible_short_qt_pattern": lambda r: None
    if r.qtc() is None
    else r.qtc() < QTC_BORDERLINE_SHORT_MS,
    "low_qrs_voltage_limb_leads": lambda r: _all_below(
        r.group_amplitudes(LIMB_LEADS), LOW_VOLTAGE_LIMB_MV
    ),
    "low_voltage_limb_leads": lambda r: _all_below(
        r.group_amplitudes(LIMB_LEADS), LOW_VOLTAGE_LIMB_MV
    ),
    "low_qrs_voltage_precordial_leads": lambda r: _all_below(
        r.group_amplitudes(PRECORDIAL_LEADS), LOW_VOLTAGE_PRECORDIAL_MV
    ),
    "low_voltage_precordial_leads": lambda r: _all_below(
        r.group_amplitudes(PRECORDIAL_LEADS), LOW_VOLTAGE_PRECORDIAL_MV
    ),
    "left_axis_deviation": lambda r: _axis_between(r, -90.0, -30.0),
    "right_axis_deviation": lambda r: None
    if r.g("qrs_axis_deg") is None
    else 90.0 < r.g("qrs_axis_deg") <= 180.0,
    "extreme_axis_deviation": lambda r: None
    if r.g("qrs_axis_deg") is None
    else -180.0 <= r.g("qrs_axis_deg") <= -90.0,
}

# PTB-XL codes that mean "this family was reported", used only to decide whether
# a measurement-confirmed call was also labelled.
LABEL_HINTS: dict[str, tuple[str, ...]] = {
    "first_degree_av_block": ("1AVB", "LPR"),
    "first_degree_av_delay": ("1AVB", "LPR"),
    "possible_first_degree_av_delay": ("1AVB", "LPR"),
    "short_pr_interval": ("SPRI", "WPW"),
    "sinus_bradycardia": ("SBRAD",),
    "bradycardia": ("SBRAD",),
    "sinus_tachycardia": ("STACH",),
    "tachycardia": ("STACH",),
    "prolonged_qt": ("LNGQT",),
    "markedly_prolonged_qt": ("LNGQT",),
    "short_qt": ("SQT",),
    "markedly_short_qt": ("SQT",),
    "borderline_short_qt": ("SQT",),
    "possible_short_qt_pattern": ("SQT",),
    "low_qrs_voltage_limb_leads": ("LVOLT",),
    "low_voltage_limb_leads": ("LVOLT",),
    "low_qrs_voltage_precordial_leads": ("LVOLT",),
    "low_voltage_precordial_leads": ("LVOLT",),
    "left_axis_deviation": ("LAD", "ALAD"),
    "right_axis_deviation": ("RAD", "ARAD"),
    "extreme_axis_deviation": ("RAD", "ARAD"),
}


def _confirmed_codes(payload: Mapping[str, Any]) -> Iterator[str]:
    verdict = payload.get("verdict")
    if not isinstance(verdict, Mapping):
        return
    for row in verdict.get("diagnoses") or []:
        if isinstance(row, Mapping) and row.get("code"):
            yield str(row["code"])


def load_reference(directory: Path) -> dict[str, set[str]]:
    path = directory / "record_results.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as handle:
        return {
            row["record"]: {
                code for code in (row.get("reference_codes") or "").split("|") if code
            }
            for row in csv.DictReader(handle)
        }


def report(directory: Path) -> None:
    reference = load_reference(directory)
    if not reference:
        for sibling in directory.parent.glob("*/record_results.csv"):
            reference = load_reference(sibling.parent)
            if reference:
                print(f"  (reference codes borrowed from {sibling.parent.name})")
                break

    buckets: collections.Counter[str] = collections.Counter()
    contradicted: list[tuple[str, str]] = []
    unlabelled: list[tuple[str, str]] = []
    for path in sorted((directory / "diagnoses").glob("*.json")):
        record_id = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        features = directory / "features" / f"{record_id}_features.json"
        if not features.exists():
            continue
        record = Record(json.loads(features.read_text(encoding="utf-8")))
        for code in _confirmed_codes(payload):
            check = CHECKS.get(code)
            if check is None:
                buckets["not-checkable (morphology family)"] += 1
                continue
            verdict = check(record)
            if verdict is None:
                buckets["not-checkable (no reliable measurement)"] += 1
            elif verdict is False:
                buckets["measurement-contradicted"] += 1
                contradicted.append((record_id, code))
            else:
                hints = set(LABEL_HINTS.get(code, ()))
                if hints & reference.get(record_id, set()):
                    buckets["measurement-confirmed and labelled"] += 1
                else:
                    buckets["unlabelled-but-correct"] += 1
                    unlabelled.append((record_id, code))

    total = sum(buckets.values())
    print(f"\n--- confirmed diagnoses: {total} ---")
    for name, count in buckets.most_common():
        share = f"{100.0 * count / total:5.1f}%" if total else "   --"
        print(f"  {name:<42}{count:>5}{share:>8}")

    checkable = (
        buckets["measurement-confirmed and labelled"]
        + buckets["unlabelled-but-correct"]
        + buckets["measurement-contradicted"]
    )
    if checkable:
        wrong = buckets["measurement-contradicted"]
        print(
            f"\n  measurement-verifiable calls: {checkable}, "
            f"of which wrong: {wrong} ({100.0 * wrong / checkable:.1f}%)"
        )
        print(
            "  label-only 'false positives' that are actually correct: "
            f"{buckets['unlabelled-but-correct']}"
        )
    if contradicted:
        print("\n  measurement-contradicted (real errors):")
        for record_id, code in contradicted:
            print(f"    {record_id:<14}{code}")
    if unlabelled:
        print("\n  correct but unlabelled (charged as FP by label scoring):")
        for record_id, code in unlabelled:
            print(f"    {record_id:<14}{code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.run_dirs:
        print("=" * 74)
        print(directory)
        print("=" * 74)
        report(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
