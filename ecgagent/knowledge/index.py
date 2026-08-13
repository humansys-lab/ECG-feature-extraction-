"""Build and search a local, detached ECG knowledge index.

The corpus contains no patient records.  It indexes local diagnostic
references, failure-mode notes and the source definitions of the deterministic
clinical/DXL/Glasgow engines.  The index is deliberately independent from
``ecgagent.tools`` so querying it is an explicit out-of-band action, never an
implicit step in an Agent diagnosis.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

INDEX_VERSION = "ecg-knowledge.v1"
MAX_CHUNK_CHARS = 9000
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[_./+-][a-z0-9]+)*", re.I)
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    category: str
    source: str
    title: str
    line_start: int
    line_end: int
    text: str
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk: KnowledgeChunk

    def excerpt(self, max_chars: int = 1800) -> str:
        text = self.chunk.text.strip()
        return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"

    def to_dict(self, *, max_chars: int = 4000) -> dict[str, Any]:
        payload = self.chunk.to_dict()
        payload["text"] = self.excerpt(max_chars=max_chars)
        payload["score"] = round(self.score, 6)
        return payload


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(category: str, source: str, title: str, line_start: int) -> str:
    raw = f"{category}\0{source}\0{title}\0{line_start}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [token for token in _ASCII_TOKEN_RE.findall(lowered) if len(token) > 1]
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _split_text(
    *,
    category: str,
    source: str,
    title: str,
    lines: Sequence[str],
    line_start: int,
    source_sha256: str,
) -> list[KnowledgeChunk]:
    if not lines:
        return []
    chunks: list[KnowledgeChunk] = []
    part: list[str] = []
    part_start = line_start
    part_chars = 0
    for offset, line in enumerate(lines):
        next_chars = part_chars + len(line) + 1
        if part and next_chars > MAX_CHUNK_CHARS:
            part_title = title if not chunks else f"{title} (continued {len(chunks) + 1})"
            end = part_start + len(part) - 1
            text = "\n".join(part).strip()
            if text:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=_chunk_id(category, source, part_title, part_start),
                        category=category,
                        source=source,
                        title=part_title,
                        line_start=part_start,
                        line_end=end,
                        text=text,
                        source_sha256=source_sha256,
                    )
                )
            part = []
            part_start = line_start + offset
            part_chars = 0
        part.append(line)
        part_chars += len(line) + 1
    text = "\n".join(part).strip()
    if text:
        part_title = title if not chunks else f"{title} (continued {len(chunks) + 1})"
        chunks.append(
            KnowledgeChunk(
                chunk_id=_chunk_id(category, source, part_title, part_start),
                category=category,
                source=source,
                title=part_title,
                line_start=part_start,
                line_end=part_start + len(part) - 1,
                text=text,
                source_sha256=source_sha256,
            )
        )
    return chunks


def _markdown_chunks(
    path: Path,
    *,
    root: Path,
    category: str,
) -> list[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    source = str(path.relative_to(root))
    digest = _source_digest(text)
    chunks: list[KnowledgeChunk] = []
    hierarchy: list[str] = []
    section_lines: list[str] = []
    section_start = 1
    section_title = path.name

    def flush() -> None:
        nonlocal section_lines
        chunks.extend(
            _split_text(
                category=category,
                source=source,
                title=section_title,
                lines=section_lines,
                line_start=section_start,
                source_sha256=digest,
            )
        )
        section_lines = []

    for number, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            hierarchy[:] = hierarchy[: level - 1]
            hierarchy.append(heading)
            section_title = " > ".join(hierarchy)
            section_start = number
        section_lines.append(line)
    flush()
    return chunks


def _python_chunks(
    path: Path,
    *,
    root: Path,
    category: str,
) -> list[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    source = str(path.relative_to(root))
    digest = _source_digest(text)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _split_text(
            category=category,
            source=source,
            title=path.name,
            lines=lines,
            line_start=1,
            source_sha256=digest,
        )

    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    starts = [int(node.lineno) for node in nodes]
    chunks: list[KnowledgeChunk] = []
    if starts and starts[0] > 1:
        chunks.extend(
            _split_text(
                category=category,
                source=source,
                title=f"{path.name} module definitions",
                lines=lines[: starts[0] - 1],
                line_start=1,
                source_sha256=digest,
            )
        )
    elif not starts:
        return _split_text(
            category=category,
            source=source,
            title=path.name,
            lines=lines,
            line_start=1,
            source_sha256=digest,
        )

    for index, node in enumerate(nodes):
        start = int(node.lineno)
        end = (
            starts[index + 1] - 1
            if index + 1 < len(starts)
            else len(lines)
        )
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks.extend(
            _split_text(
                category=category,
                source=source,
                title=f"{path.name} {kind} {node.name}",
                lines=lines[start - 1 : end],
                line_start=start,
                source_sha256=digest,
            )
        )
    return chunks


def _corpus(root: Path) -> list[tuple[Path, str]]:
    fixed = [
        (
            root / "ecgagent" / "knowledge" / "english_diagnostic_reference.md",
            "diagnostic_reference",
        ),
        (
            root / "ecgagent" / "knowledge" / "english_diagnostic_reference.md",
            "morphology_reference",
        ),
        (
            root / "ecgagent" / "knowledge" / "english_measurement_reliability.md",
            "measurement_reliability_reference",
        ),
        (
            root / "ecgagent" / "knowledge" / "english_failure_modes.md",
            "failure_modes",
        ),
        (root / "docs" / "ecg_agent_architecture.md", "capability_blueprint"),
        (
            root / "feature_extraction" / "ecgfeat" / "interpret.py",
            "philips_dxl_reference",
        ),
        (
            root / "feature_extraction" / "ecgfeat" / "statement_engine.py",
            "philips_dxl_reference",
        ),
        (
            root / "feature_extraction" / "ecgfeat" / "glasgow.py",
            "glasgow_reference",
        ),
    ]
    clinical_dir = root / "feature_extraction" / "ecgfeat" / "clinical_rules"
    glasgow_dir = root / "feature_extraction" / "ecgfeat" / "glasgow_rules"
    fixed.extend(
        (path, "clinical_rules_reference")
        for path in sorted(clinical_dir.glob("*.py"))
        if path.name != "__init__.py"
    )
    fixed.extend(
        (path, "glasgow_reference")
        for path in sorted(glasgow_dir.glob("*.py"))
        if path.name != "__init__.py"
    )
    return [(path, category) for path, category in fixed if path.exists()]


class KnowledgeBase:
    """In-memory BM25-like index over the detached local corpus."""

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        *,
        project_root: str | Path | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self.project_root = Path(project_root or _project_root()).resolve()
        self._term_counts = [Counter(_tokens(chunk.title + "\n" + chunk.text)) for chunk in self.chunks]
        self._doc_lengths = [sum(counts.values()) for counts in self._term_counts]
        self._document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())
        self._average_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 1.0
        )

    @classmethod
    def from_project(
        cls,
        project_root: str | Path | None = None,
    ) -> "KnowledgeBase":
        root = Path(project_root or _project_root()).resolve()
        chunks: list[KnowledgeChunk] = []
        for path, category in _corpus(root):
            if path.suffix == ".py":
                chunks.extend(_python_chunks(path, root=root, category=category))
            else:
                chunks.extend(_markdown_chunks(path, root=root, category=category))
        return cls(chunks, project_root=root)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("index_version") != INDEX_VERSION:
            raise ValueError(
                f"unsupported knowledge index version {payload.get('index_version')!r}"
            )
        chunks = [KnowledgeChunk(**row) for row in payload.get("chunks") or []]
        return cls(
            chunks,
            project_root=payload.get("project_root") or _project_root(),
        )

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({chunk.category for chunk in self.chunks}))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_version": INDEX_VERSION,
            "project_root": str(self.project_root),
            "chunk_count": len(self.chunks),
            "categories": list(self.categories),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        categories: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        query_text = str(query).strip()
        query_terms = _tokens(query_text)
        if not query_terms:
            return []
        allowed = set(categories or [])
        unknown = allowed - set(self.categories)
        if unknown:
            raise ValueError(
                f"unknown categories: {', '.join(sorted(unknown))}; "
                f"available: {', '.join(self.categories)}"
            )

        n_documents = max(1, len(self.chunks))
        results: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            if allowed and chunk.category not in allowed:
                continue
            counts = self._term_counts[index]
            length = max(1, self._doc_lengths[index])
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency.get(term, 0)
                inverse = math.log(
                    1.0 + (n_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * length / max(1.0, self._average_length)
                )
                score += inverse * (frequency * 2.2) / denominator
            lowered_query = query_text.lower()
            if lowered_query and lowered_query in chunk.title.lower():
                score += 5.0
            elif lowered_query and lowered_query in chunk.text.lower():
                score += 2.0
            title_terms = set(_tokens(chunk.title))
            score += 0.7 * sum(term in title_terms for term in query_terms)
            if score > 0:
                results.append(SearchResult(score=score, chunk=chunk))
        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk.source,
                result.chunk.line_start,
            )
        )
        return results[: max(1, min(int(limit), 50))]

    def manifest(self) -> dict[str, Any]:
        counts = Counter(chunk.category for chunk in self.chunks)
        return {
            "index_version": INDEX_VERSION,
            "project_root": str(self.project_root),
            "chunk_count": len(self.chunks),
            "categories": dict(sorted(counts.items())),
            "agent_runtime_registered": False,
        }
