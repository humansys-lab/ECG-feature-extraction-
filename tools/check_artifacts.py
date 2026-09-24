#!/usr/bin/env python3
"""Release artifact inspection (docs/library_design/07_packaging_and_release.md, step 5).

For every wheel and sdist in a directory: required files present (py.typed,
shipped schemas and registry, README long description, CHANGELOG, LICENSE) and
nothing that must not ship (tests, caches, datasets, secrets, VCS metadata).
A missing LICENSE is a release blocker; ``--allow-missing-license`` exists only
for local rehearsal before the maintainer decision.
"""

from __future__ import annotations

import argparse
import email
import sys
import tarfile
import zipfile
from pathlib import Path

REQUIRED_WHEEL = {
    "ecg_records": ["ecgfeat/py.typed", "ecgfeat/schemas/ecg-record/1.0/schema.json",
                    "ecgfeat/schemas/ecg-record/1.0/validation-evidence.json", "ecgfeat/viz/plots.py"],
}
FORBIDDEN_PARTS = ("/tests/", "__pycache__", "/.git/", "/data/", "/dataset/", ".env", "credentials", "secrets")
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".npz", ".hea", ".dat", ".mat", ".edf", ".pem", ".key", ".pt", ".pth")
DISCLAIMER = "not a medical device"


def _project(path: Path) -> str:
    return path.name.split("-")[0]


def check(path: Path, allow_missing_license: bool) -> list[str]:
    problems: list[str] = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
            metadata = email.message_from_bytes(archive.read(metadata_name))
        for required in REQUIRED_WHEEL.get(_project(path), []):
            if required not in names:
                problems.append(f"missing {required}")
        if DISCLAIMER not in (metadata.get_payload() or "") + metadata.get("Summary", ""):
            problems.append("long description/summary lacks the medical disclaimer")
        has_license = bool(metadata.get_all("License-File") or metadata.get("License-Expression"))
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
        root = names[0].split("/")[0]
        for required in ("PKG-INFO", "pyproject.toml", "CHANGELOG.md"):
            if f"{root}/{required}" not in names:
                problems.append(f"sdist missing {required}")
        has_license = f"{root}/LICENSE" in names
    if not has_license and not allow_missing_license:
        problems.append("no LICENSE in the artifact (release blocker: maintainer license decision pending)")
    for name in names:
        lowered = "/" + name.lower()
        if any(part in lowered for part in FORBIDDEN_PARTS) or lowered.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"must not ship: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--allow-missing-license", action="store_true")
    args = parser.parse_args(argv)
    artifacts = sorted(p for p in args.dist.iterdir() if p.suffix in {".whl", ".gz"})
    failed = False
    for artifact in artifacts:
        problems = check(artifact, args.allow_missing_license)
        print(f"{'FAIL' if problems else 'ok  '} {artifact.name}")
        for problem in problems:
            print(f"     - {problem}")
        failed |= bool(problems)
    if not artifacts:
        print("no artifacts found")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
