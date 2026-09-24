#!/usr/bin/env python3
"""Fail unless the CHANGELOG has a section for the pyproject version."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTIONS = ("feature_extraction",)


def main() -> int:
    failed = False
    for directory in DISTRIBUTIONS:
        pyproject = (ROOT / directory / "pyproject.toml").read_text()
        version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
        changelog = (ROOT / directory / "CHANGELOG.md").read_text()
        ok = re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.M) is not None
        print(f"{'ok  ' if ok else 'FAIL'} {directory}: {version}")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
