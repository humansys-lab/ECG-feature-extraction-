#!/usr/bin/env python3
"""Mechanical validation-tier gate (docs/library_design/06_testing_and_validation.md).

Fails when a field claims more validation than its evidence supports, when the
repository and packaged copies of the schema resources differ, or when a
known-misleading field is given a better tier or a published position.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT / "schemas" / "ecg-record" / "1.0"
PACKAGE_DIR = ROOT / "feature_extraction" / "ecgfeat" / "schemas" / "ecg-record" / "1.0"
EVIDENCE_DIR = ROOT / "benchmarks" / "evidence"
ORDER = ("known_problem", "unvalidated", "indirectly_validated", "benchmark_validated")
PUBLICATION = {"published_measurement", "provenance", "internal_debug"}
MANIFEST_KEYS = {"evidence_id", "tool", "dataset", "baseline_id", "metric", "acceptance_threshold",
                 "result", "validated_for", "schema_major", "code_revision", "approved_by"}


def problems(registry: dict[str, Any], evidence_dir: Path = EVIDENCE_DIR) -> list[str]:
    found: list[str] = []
    seen = set()
    for entry in registry["fields"] + registry["debug_caps"]:
        pointer = entry.get("pointer", "?")
        if pointer in seen:
            found.append(f"{pointer}: duplicate registry entry")
        seen.add(pointer)
        tier, ceiling = entry.get("validation_tier"), entry.get("validation_ceiling")
        if tier not in ORDER or ceiling not in ORDER:
            found.append(f"{pointer}: unknown validation tier/ceiling {tier!r}/{ceiling!r}")
            continue
        if entry.get("publication_tier") not in PUBLICATION:
            found.append(f"{pointer}: unknown publication tier")
        if ORDER.index(tier) > ORDER.index(ceiling):
            found.append(f"{pointer}: tier {tier} exceeds ceiling {ceiling}")
        if tier in {"indirectly_validated", "benchmark_validated"}:
            ids = entry.get("evidence_ids") or []
            if not ids:
                found.append(f"{pointer}: {tier} without evidence")
            if not entry.get("validated_for"):
                found.append(f"{pointer}: {tier} without a validated_for scope")
            for evidence_id in ids:
                path = evidence_dir / f"{evidence_id}.json"
                if not path.exists():
                    found.append(f"{pointer}: evidence manifest {evidence_id} missing")
                    continue
                manifest = json.loads(path.read_text(encoding="utf-8"))
                missing = MANIFEST_KEYS - manifest.keys()
                if missing:
                    found.append(f"{pointer}: evidence {evidence_id} lacks {sorted(missing)}")
                if not (manifest.get("result") or {}).get("passed"):
                    found.append(f"{pointer}: evidence {evidence_id} is not a passing result")
                if str(manifest.get("schema_major")) != registry["schema_family"].split(".")[0]:
                    found.append(f"{pointer}: evidence {evidence_id} is from another schema major")
        if tier == "indirectly_validated" and not entry.get("known_limitations"):
            found.append(f"{pointer}: indirectly_validated requires known_limitations")
        if entry.get("publication_tier") == "internal_debug" and entry.get("profiles"):
            found.append(f"{pointer}: internal-debug fields cannot be listed in published profiles")
    morphology = next((e for e in registry["debug_caps"] if e["pointer"] == "/debug/st_morphology"), None)
    if morphology is None or morphology["validation_ceiling"] != "known_problem" \
            or morphology["validation_tier"] not in {"known_problem", "unvalidated"} \
            or morphology["publication_tier"] != "internal_debug":
        found.append("/debug/st_morphology must stay internal_debug with ceiling known_problem")
    patterns = {p["pattern"]: p for p in registry.get("debug_patterns", [])}
    for pattern in ("st_hybrid_*", "twelve_sl_*"):
        if patterns.get(pattern, {}).get("publication_tier") != "internal_debug":
            found.append(f"{pattern} must be capped at internal_debug")
    for entry in registry["fields"]:
        name = entry["pointer"].rsplit("/", 1)[-1]
        if any(name.startswith(p[:-1]) for p in patterns) or name in {"st_morphology", "st_pattern_class", "st_t_confusion"}:
            found.append(f"{entry['pointer']}: debug-capped quantity listed as a published field")
    return found


def resource_mismatches() -> list[str]:
    found = []
    for name in ("schema.json", "validation-evidence.json"):
        repo, package = REPO_DIR / name, PACKAGE_DIR / name
        if not package.exists() or repo.read_bytes() != package.read_bytes():
            found.append(f"{name}: repository and packaged copies differ")
    return found


def main() -> int:
    registry = json.loads((REPO_DIR / "validation-evidence.json").read_text(encoding="utf-8"))
    issues = resource_mismatches() + problems(registry)
    for issue in issues:
        print(issue)
    counts: dict[str, int] = {}
    for entry in registry["fields"]:
        counts[entry["validation_tier"]] = counts.get(entry["validation_tier"], 0) + 1
    print(json.dumps({"fields": len(registry["fields"]), "tiers": counts, "problems": len(issues)}))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
