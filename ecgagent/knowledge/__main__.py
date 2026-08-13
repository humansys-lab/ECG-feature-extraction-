"""CLI for the detached ECG knowledge library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index import KnowledgeBase


def _base(index: Path | None, project_root: Path | None) -> KnowledgeBase:
    return (
        KnowledgeBase.load(index)
        if index is not None
        else KnowledgeBase.from_project(project_root)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="workspace root (normally auto-detected)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="load an existing JSON index instead of building in memory",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build and persist the detached index")
    build.add_argument("--output", type=Path, required=True)

    search = sub.add_parser("search", help="search the detached knowledge corpus")
    search.add_argument("query", type=str)
    search.add_argument("--category", action="append", default=[])
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--json", action="store_true")

    manifest = sub.add_parser("manifest", help="show corpus categories and size")
    manifest.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build":
        base = KnowledgeBase.from_project(args.project_root)
        output = base.save(args.output)
        print(
            json.dumps(
                {**base.manifest(), "output": str(output)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    base = _base(args.index, args.project_root)
    if args.command == "manifest":
        payload = base.manifest()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"{payload['index_version']}: {payload['chunk_count']} chunks")
            for category, count in payload["categories"].items():
                print(f"- {category}: {count}")
            print("- agent_runtime_registered: false")
        return 0

    results = base.search(
        args.query,
        limit=args.limit,
        categories=args.category or None,
    )
    if args.json:
        print(
            json.dumps(
                [result.to_dict() for result in results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not results:
        print("no matching knowledge chunks")
        return 1
    for position, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"[{position}] score={result.score:.3f} [{chunk.category}] "
            f"{chunk.title}"
        )
        print(f"    {chunk.source}:{chunk.line_start}-{chunk.line_end}")
        excerpt = result.excerpt().replace("\n", "\n    ")
        print(f"    {excerpt}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
