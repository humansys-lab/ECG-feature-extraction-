#!/usr/bin/env python3
"""Validate coverage and Markdown structure of localized documentation."""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

import localize_docs as localization


PLACEHOLDER_RE = re.compile(r"ZXQ(?:PH|CODE)\d+")
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\([^)\n]+\)")
BROKEN_HEADING_RE = re.compile(r"^＃+|^#{1,6}[^#\s]", re.MULTILINE)


def prose(text: str) -> str:
    return "".join(
        segment
        for is_code, segment in localization.split_prose_and_code(text)
        if not is_code
    )


def nonblank_prose_lines(text: str) -> int:
    return sum(
        1
        for line in prose(text).splitlines()
        if line.strip() and not line.startswith('<a id="')
    )


def main() -> int:
    issues: list[str] = []
    sources = localization.tracked_document_sources()

    for source in sources:
        original = localization.remove_navigation(source.read_text(encoding="utf-8"))
        original_code = [
            segment
            for is_code, segment in localization.split_prose_and_code(original)
            if is_code
        ]
        original_heading_levels = [
            marker for marker, _ in localization.markdown_headings(original)
        ]
        original_link_count = len(LINK_RE.findall(prose(original)))
        original_line_count = nonblank_prose_lines(original)

        for language in localization.LANGUAGES:
            output = localization.localized_path(source, language)
            label = f"{language}: {output.relative_to(localization.ROOT)}"
            if not output.exists():
                issues.append(f"missing translation: {label}")
                continue

            text = output.read_text(encoding="utf-8")
            if not text.startswith(localization.NAV_START):
                issues.append(f"missing language navigation: {label}")
            if PLACEHOLDER_RE.search(text):
                issues.append(f"unresolved placeholder: {label}")

            body = localization.remove_navigation(text)
            translated_code = [
                segment
                for is_code, segment in localization.split_prose_and_code(body)
                if is_code
            ]
            if translated_code != original_code:
                issues.append(f"code block changed: {label}")

            translated_heading_levels = [
                marker for marker, _ in localization.markdown_headings(body)
            ]
            if translated_heading_levels != original_heading_levels:
                issues.append(f"heading structure changed: {label}")
            if BROKEN_HEADING_RE.search(body):
                issues.append(f"invalid heading marker: {label}")

            translated_link_count = len(LINK_RE.findall(prose(body)))
            if translated_link_count != original_link_count:
                issues.append(
                    f"link count changed ({original_link_count} -> "
                    f"{translated_link_count}): {label}"
                )

            translated_line_count = nonblank_prose_lines(body)
            line_ratio = translated_line_count / max(1, original_line_count)
            if not 0.80 <= line_ratio <= 1.25:
                issues.append(
                    f"prose line coverage outside tolerance "
                    f"({original_line_count} -> {translated_line_count}): {label}"
                )

            nav = text.split(localization.NAV_END, 1)[0]
            for target in re.findall(r"\]\(([^)]+)\)", nav):
                destination = output.parent / urllib.parse.unquote(target)
                if not destination.exists():
                    issues.append(f"missing navigation target {target}: {label}")

            translated_prose = prose(body)
            if language == "en" and len(re.findall(r"\b[A-Za-z]{3,}\b", translated_prose)) < 20:
                issues.append(f"insufficient English language signal: {label}")
            if language == "ja" and len(re.findall(r"[\u3040-\u30ff]", translated_prose)) < 20:
                issues.append(f"insufficient Japanese language signal: {label}")

    if issues:
        print("Documentation localization checks failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print(
        f"Documentation localization checks passed: {len(sources)} sources, "
        f"{len(sources) * len(localization.LANGUAGES)} translations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
