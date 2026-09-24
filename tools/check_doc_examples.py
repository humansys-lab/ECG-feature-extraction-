#!/usr/bin/env python3
"""Execute every Python example in the documentation.

Each page under ``docs/site`` (except the generated ``reference/`` pages) runs
in its own scratch directory containing ``ecg.npy`` and ``ecg.npz`` (the 10 s,
12-lead reference signal from ``tests/fixtures/golden/reference_10s_12lead``)
and ``ecg.record.json`` (its ``summary`` record).  The ```` ```python ```` blocks
of one page share a namespace and run in order, so later blocks may use names
from earlier ones.  A block preceded by ``<!-- doc-check: skip (reason) -->``
is not run (it needs files the checker cannot provide).  Exit status 1 if any
block fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "site"
FIXTURE = ROOT / "tests" / "fixtures" / "golden" / "reference_10s_12lead"
BLOCK = re.compile(r"(?P<marker><!-- doc-check: skip[^>]*-->\n)?```python\n(?P<code>.*?)\n```", re.S)


def blocks(page: Path) -> list[tuple[int, str, bool]]:
    text = page.read_text(encoding="utf-8")
    found = []
    for match in BLOCK.finditer(text):
        line = text.count("\n", 0, match.start("code")) + 1
        found.append((line, match.group("code"), match.group("marker") is not None))
    return found


def prepare(directory: Path) -> None:
    import numpy as np

    with np.load(FIXTURE / "signal.npz") as archive:
        signal = archive["signal"]
    np.save(directory / "ecg.npy", signal)
    np.savez(directory / "ecg.npz", signal=signal)
    shutil.copyfile(FIXTURE / "record.summary.json", directory / "ecg.record.json")


def run_page(page: Path, verbose: bool) -> list[str]:
    failures = []
    found = blocks(page)
    if not found:
        return failures
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="ecg-doc-") as scratch:
        prepare(Path(scratch))
        os.chdir(scratch)
        namespace: dict = {"__name__": "__doc_example__"}
        try:
            with warnings.catch_warnings():  # a page's filters must not leak into the next page
                for line, code, skipped in found:
                    label = f"{page.relative_to(ROOT)}:{line}"
                    if skipped:
                        if verbose:
                            print(f"skip {label}")
                        continue
                    output = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(output):
                            exec(compile(code, label, "exec"), namespace)
                    except Exception:
                        failures.append(f"{label}\n{traceback.format_exc()}")
                        print(f"FAIL {label}")
                    else:
                        if verbose:
                            print(f"ok   {label}")
                            for text in output.getvalue().splitlines():
                                print(f"       {text}")
        finally:
            os.chdir(previous)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="*", type=Path, help="pages to check (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every block and its output")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    pages = [p.resolve() for p in args.pages] or sorted(
        p for p in DOCS.rglob("*.md") if "reference" not in p.relative_to(DOCS).parts)
    failures: list[str] = []
    total = 0
    for page in pages:
        total += sum(1 for *_, skipped in blocks(page) if not skipped)
        failures += run_page(page, args.verbose)
    for failure in failures:
        print("\n" + failure, file=sys.stderr)
    print(f"{total - len(failures)}/{total} documentation examples ran")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
