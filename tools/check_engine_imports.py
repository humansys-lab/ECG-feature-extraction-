#!/usr/bin/env python3
"""Reject imports of the private engine from consumer code (document 01).

``import-linter`` guards layers inside ``ecgfeat``; this AST check guards the
consumers outside it: ``ecgagent/``, the interpretation distribution, and
root-level tools.  They must use the ECG Record boundary.  (``ecgfeat.viz`` is
inside the package and guarded by the import-linter contracts instead.)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = ("ecgagent", "interpretation")
PRIVATE = ("_engine", "_kalman", "numeric", "preprocess", "acquisition_qc", "quality", "qrs",
           "adaptive_qrs", "grouping", "representative", "family_representative", "delineate",
           "classical_candidates", "boundary_refinement", "wave_localization", "r_localization",
           "repolarization", "t_wave_refinement", "atrial", "p_wave_engine", "p_morphology",
           "atrial_validation", "features", "dispersion", "measurement_paths", "st_baseline",
           "st_localization", "u_wave", "vector_axis", "calibration", "twelve_sl", "glasgow_measurements")
# The private engine, including its temporary aliases at the old module paths.
FORBIDDEN = tuple(f"{root}.{name}" for root in ("ecgfeat", "feature_extraction.ecgfeat") for name in PRIVATE)


def violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        for name in names:
            if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}: imports {name}")
                break
    return found


def main() -> int:
    problems = []
    for directory in CONSUMERS:
        base = ROOT / directory
        if base.exists():
            for path in sorted(base.rglob("*.py")):
                problems += violations(path)
    for line in problems:
        print(line)
    print(f"{len(problems)} private-engine import(s) in consumer code")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
