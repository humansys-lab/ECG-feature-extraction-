"""Comparison modes for the golden parity gate (document 05, section B).

``legacy-bytes``        exact legacy JSON bytes for every legacy profile.
``record-bytes``        exact canonical ECG Record bytes for every profile
                        (only meaningful while library/schema versions are fixed).
``record-crosswalk``    parsed legacy leaves vs. ECG Record through the reviewed
                        crosswalk table (see :mod:`benchmarks.golden.crosswalk`).
``canonical-document``  key-order-independent leaf equality of record documents.

No mode applies numeric tolerances.  Input identities (source-file hashes and
the canonical signal hash) are verified before any output is compared, so a
changed corpus fails as an integrity error rather than as a measurement diff.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from .runner import case_file_stem, iter_leaves

MODES = ("legacy-bytes", "record-bytes", "record-crosswalk", "canonical-document")


def input_mismatches(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    problems = []
    for key in ("adapter", "window", "fs", "units", "lead_names", "shape", "signal_sha256",
                "signal_meta_sha256", "source_files"):
        if expected.get(key) != observed.get(key):
            problems.append(key)
    return problems


def pointer_diff(before: Any, after: Any, limit: int = 25) -> dict[str, Any]:
    old = dict(iter_leaves(before))
    new = dict(iter_leaves(after))
    changed = sorted(key for key in old.keys() & new.keys() if old[key] != new[key])
    removed = sorted(old.keys() - new.keys())
    added = sorted(new.keys() - old.keys())
    return {
        "changed": len(changed), "removed": len(removed), "added": len(added),
        "examples": [{"pointer": key, "baseline": old[key], "candidate": new[key]} for key in changed[:limit]],
        "removed_examples": removed[:limit], "added_examples": added[:limit],
    }


def _load_legacy_payload(directory: Path | None, case_id: str) -> Any:
    if directory is None:
        return None
    path = directory / "legacy" / f"{case_file_stem(case_id)}.debug.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else None


def _load_record_payload(directory: Path | None, case_id: str, profile: str) -> Any:
    if directory is None:
        return None
    path = directory / "record" / f"{case_file_stem(case_id)}.{profile}.json"
    return json.loads(path.read_bytes()) if path.exists() else None


def compare_case(mode: str, expected: dict[str, Any], observed: dict[str, Any], *,
                 baseline_payloads: Path | None, candidate_payloads: Path | None) -> dict[str, Any]:
    case_id = expected["case_id"]
    report: dict[str, Any] = {"case_id": case_id, "status": "pass", "differences": []}
    integrity = input_mismatches(expected["input"], observed["input"])
    if integrity:
        report.update(status="integrity_error", integrity=integrity)
        return report

    if mode in {"legacy-bytes", "record-bytes", "canonical-document"}:
        surface = "legacy" if mode == "legacy-bytes" else "record"
        want_surface = expected["outputs"][surface]
        got_surface = observed["outputs"].get(surface, {})
        if "error" in want_surface or "error" in got_surface:
            if want_surface.get("error") != got_surface.get("error"):
                report["differences"].append({"surface": surface, "baseline_error": want_surface.get("error"),
                                              "candidate_error": got_surface.get("error")})
                report["status"] = "diff"
            return report

    if mode in {"legacy-bytes", "record-bytes"}:
        surface = "legacy" if mode == "legacy-bytes" else "record"
        for profile, want in expected["outputs"][surface].items():
            got = observed["outputs"].get(surface, {}).get(profile)
            if got is None or got["sha256"] != want["sha256"]:
                difference: dict[str, Any] = {"surface": surface, "profile": profile,
                                              "baseline": want, "candidate": got}
                if surface == "legacy" and profile == "debug":
                    before = _load_legacy_payload(baseline_payloads, case_id)
                    after = _load_legacy_payload(candidate_payloads, case_id)
                    if before is not None and after is not None:
                        difference["pointer_diff"] = pointer_diff(before, after)
                        if not any(difference["pointer_diff"][k] for k in ("changed", "removed", "added")):
                            difference["note"] = "leaves identical; key order or float rendering changed"
                if surface == "record":
                    before = _load_record_payload(baseline_payloads, case_id, profile)
                    after = _load_record_payload(candidate_payloads, case_id, profile)
                    if before is not None and after is not None:
                        difference["pointer_diff"] = pointer_diff(before, after)
                report["differences"].append(difference)
    elif mode == "canonical-document":
        for profile, want in expected["outputs"]["record"].items():
            got = observed["outputs"].get("record", {}).get(profile)
            if got is None or got["leaf_digest"] != want["leaf_digest"]:
                report["differences"].append({"surface": "record", "profile": profile,
                                              "baseline": want, "candidate": got})
    elif mode == "record-crosswalk":
        from .crosswalk import compare_with_crosswalk

        before = _load_legacy_payload(baseline_payloads, case_id)
        records = {profile: _load_record_payload(candidate_payloads, case_id, profile)
                   for profile in ("summary", "all", "debug")}
        if before is None or any(value is None for value in records.values()):
            report.update(status="missing_payload")
            return report
        report["differences"] = compare_with_crosswalk(before, records)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if report["differences"]:
        report["status"] = "diff"
    return report
