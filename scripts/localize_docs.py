#!/usr/bin/env python3
"""Generate English and Japanese companions for the repository documentation.

The script keeps fenced code blocks, inline code, URLs, local link targets, HTML
tags, and common ECG/product terms unchanged. Translations are cached locally so
an interrupted run can be resumed without retranslating completed chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / ".doc_translation_cache.json"
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
LOCAL_TRANSLATE_URLS = tuple(
    item.strip()
    for item in os.getenv("DOC_TRANSLATE_LOCAL_URL", "").split(",")
    if item.strip()
)
LOCAL_TRANSLATE_MODEL = os.getenv("DOC_TRANSLATE_LOCAL_MODEL", "qwen3.8-27b-docs")
LOCAL_URL_LOCK = threading.Lock()
LOCAL_URL_INDEX = 0
NAV_START = "<!-- i18n-nav -->"
NAV_END = "<!-- /i18n-nav -->"
LANGUAGES = {"en": "English", "ja": "日本語"}
NOTICES = {
    "en": (
        "> Translation note: This English version is provided for convenience. "
        "If it differs from the source document, the source document prevails."
    ),
    "ja": (
        "> 翻訳に関する注意：この日本語版は参考用です。原文と相違がある場合は、"
        "原文を優先します。"
    ),
}
MAX_CHARS = 3_500
PROTECTED_RE = re.compile(
    r"`[^`\n]+`"
    r"|https?://[^\s)>]+"
    r"|(?<=\]\()[^)\n]+(?=\))"
    r"|</?[^>\n]+>"
    r"|\$[^$\n]+\$"
    r"|\b[A-Za-z0-9_.+/-]+\.(?:py|json|md|csv|tsv|xlsx|toml|yaml|yml|sh|html|css|js)\b",
    re.IGNORECASE,
)
TECHNICAL_TERMS = (
    "ECGFeatureExtractor",
    "StructuredOutputsParams",
    "QRST subtraction",
    "representative beat",
    "MedGemma",
    "DeepSeek",
    "NeuroKit2",
    "BioSPPy",
    "PTB-XL",
    "ecgagent",
    "ecgfeat",
    "OpenAI",
    "Anthropic",
    "Matplotlib",
    "transformers",
    "PyTorch",
    "NumPy",
    "SciPy",
    "Gradio",
    "CUDA",
    "WFDB",
    "LUDB",
    "QTDB",
    "vLLM",
    "MedGemma",
    "QRS",
    "QTc",
    "AF/AFL",
    "ST-T",
    "JSON",
    "API",
    "DXL",
    "ECG",
)
TECHNICAL_RE = re.compile(
    "|".join(re.escape(term) for term in sorted(set(TECHNICAL_TERMS), key=len, reverse=True)),
    re.IGNORECASE,
)
LINK_RE = re.compile(r"(\]\()([^)\n]+)(\))")


class TranslationCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        try:
            self.data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}

    def get(self, key: str) -> str | None:
        with self.lock:
            return self.data.get(key)

    def put(self, key: str, value: str) -> None:
        with self.lock:
            self.data[key] = value
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.data, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def delete(self, key: str) -> None:
        with self.lock:
            if key not in self.data:
                return
            del self.data[key]
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.data, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.path)


def tracked_document_sources() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths = [ROOT / item.decode("utf-8") for item in raw.split(b"\0") if item]
    sources: list[Path] = []
    for path in paths:
        relative = path.relative_to(ROOT)
        localized = path.name.endswith((".en.md", ".ja.md"))
        is_markdown = path.suffix == ".md"
        is_docs_text = (
            relative.parts[0] == "docs"
            and (path.suffix == ".md" or path.suffix == "")
        )
        if not localized and (is_markdown or is_docs_text):
            sources.append(path)
    return sorted(sources, key=lambda item: item.as_posix())


def localized_path(source: Path, language: str) -> Path:
    if source.suffix == ".md":
        return source.with_name(f"{source.stem}.{language}.md")
    return source.with_name(f"{source.name}.{language}.md")


def remove_navigation(text: str) -> str:
    pattern = re.compile(
        rf"\A{re.escape(NAV_START)}.*?{re.escape(NAV_END)}\s*",
        re.DOTALL,
    )
    return pattern.sub("", text, count=1)


def encoded_name(path: Path) -> str:
    return urllib.parse.quote(path.name, safe="._-")


def navigation(source: Path) -> str:
    chinese = encoded_name(source)
    english = encoded_name(localized_path(source, "en"))
    japanese = encoded_name(localized_path(source, "ja"))
    return (
        f"{NAV_START}\n"
        f"[中文]({chinese}) | [English]({english}) | [日本語]({japanese})\n"
        f"{NAV_END}"
    )


def split_prose_and_code(text: str) -> list[tuple[bool, str]]:
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    in_code = False
    fence = ""

    def flush(is_code: bool) -> None:
        if buffer:
            segments.append((is_code, "".join(buffer)))
            buffer.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else ""
        if marker and not in_code:
            flush(False)
            in_code = True
            fence = marker
            buffer.append(line)
        elif in_code:
            buffer.append(line)
            if marker == fence:
                flush(True)
                in_code = False
                fence = ""
        else:
            buffer.append(line)
    flush(in_code)
    return segments


def chunks(text: str, limit: int = MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    result: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) <= limit:
            current += line
            continue
        if current:
            result.append(current)
            current = ""
        while len(line) > limit:
            split_at = line.rfind("。", 0, limit)
            if split_at < limit // 2:
                split_at = line.rfind(". ", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            else:
                split_at += 1
            result.append(line[:split_at])
            line = line[split_at:]
        current = line
    if current:
        result.append(current)
    return result


def protect(text: str) -> tuple[str, dict[str, str]]:
    values: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"ZXQPH{len(values):05d}QXZ"
        values[token] = match.group(0)
        return token

    protected = PROTECTED_RE.sub(replace, text)
    protected = TECHNICAL_RE.sub(replace, protected)
    return protected, values


def restore(text: str, values: dict[str, str]) -> str:
    missing = [token for token in values if token not in text]
    if missing:
        raise RuntimeError(f"translation dropped {len(missing)} protected placeholders")
    for token, value in values.items():
        text = text.replace(token, value)
    return text


def request_local_translation(text: str, language: str) -> str:
    global LOCAL_URL_INDEX
    language_name = {"en": "natural English", "ja": "natural Japanese"}[language]
    placeholders = sorted(set(re.findall(r"ZXQPH\d{5}QXZ", text)))
    prompt = (
        f"Translate the user text into {language_name}. Preserve Markdown structure, "
        "line breaks, table delimiters, and all content. Translate line by line without "
        "summarizing or omitting anything. Do not translate, alter, reorder, or omit any "
        f"placeholder. The exact placeholders that must each appear are: {placeholders}. "
        "Output only the complete translation with no commentary or Markdown wrapper."
    )
    body = json.dumps(
        {
            "model": LOCAL_TRANSLATE_MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 4_000,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    with LOCAL_URL_LOCK:
        url = LOCAL_TRANSLATE_URLS[LOCAL_URL_INDEX % len(LOCAL_TRANSLATE_URLS)]
        LOCAL_URL_INDEX += 1
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        data = json.loads(response.read().decode("utf-8"))
    if data["choices"][0].get("finish_reason") != "stop":
        raise RuntimeError(
            f"local translation was truncated: {data['choices'][0].get('finish_reason')}"
        )
    return data["choices"][0]["message"]["content"]


def request_translation(text: str, language: str, cache: TranslationCache) -> str:
    if not text.strip():
        return text
    protected, values = protect(text)
    digest = hashlib.sha256(f"v1\0{language}\0{protected}".encode("utf-8")).hexdigest()
    cached = cache.get(digest)
    if cached is not None:
        try:
            return restore_outer_whitespace(text, restore(html.unescape(cached), values))
        except RuntimeError:
            if not LOCAL_TRANSLATE_URLS:
                raise
            cache.delete(digest)
            cached = None
    if cached is None:
        if LOCAL_TRANSLATE_URLS:
            last_error: Exception | None = None
            for attempt in range(4):
                try:
                    cached = request_local_translation(protected, language)
                    restored = restore(cached, values)
                    cache.put(digest, cached)
                    return restore_outer_whitespace(text, restored)
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
                    last_error = exc
                    time.sleep(2**attempt)
            else:
                raise RuntimeError(f"local translation request failed: {last_error}")
        payload = urllib.parse.urlencode(
            {
                "client": "gtx",
                "sl": "auto",
                "tl": language,
                "dt": "t",
                "q": protected,
            }
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(7):
            try:
                request = urllib.request.Request(
                    TRANSLATE_URL,
                    data=payload,
                    headers={"User-Agent": "Mozilla/5.0 documentation-localizer"},
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = json.loads(response.read().decode("utf-8"))
                cached = "".join(part[0] or "" for part in data[0])
                cache.put(digest, cached)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 30))
        else:
            raise RuntimeError(f"translation request failed: {last_error}")
    return restore_outer_whitespace(text, restore(html.unescape(cached), values))


def restore_outer_whitespace(source: str, translated: str) -> str:
    """Keep boundary whitespace exact so prose cannot join adjacent code fences."""
    if not source:
        return translated
    leading_match = re.match(r"\s*", source)
    leading = leading_match.group(0) if leading_match else ""
    remaining = source[len(leading) :]
    trailing_match = re.search(r"\s*$", remaining)
    trailing = trailing_match.group(0) if trailing_match else ""
    core = translated.strip()
    return f"{leading}{core}{trailing}"


def translate_text(text: str, language: str, cache: TranslationCache) -> str:
    translated: list[str] = []
    for is_code, segment in split_prose_and_code(text):
        if is_code:
            translated.append(segment)
        else:
            prose_chunks = chunks(segment)
            if len(prose_chunks) == 1:
                translated.append(request_translation(prose_chunks[0], language, cache))
            else:
                with ThreadPoolExecutor(max_workers=min(6, len(prose_chunks))) as executor:
                    translated.extend(
                        executor.map(
                            lambda chunk: request_translation(chunk, language, cache),
                            prose_chunks,
                        )
                    )
    return "".join(translated)


def rewrite_local_links(
    text: str,
    source: Path,
    output: Path,
    language: str,
    source_lookup: dict[Path, Path],
) -> str:
    def replace(match: re.Match[str]) -> str:
        destination = match.group(2).strip()
        if destination.startswith(("#", "/", "http://", "https://", "mailto:")):
            return match.group(0)
        path_part, separator, fragment = destination.partition("#")
        decoded = urllib.parse.unquote(path_part)
        target = (source.parent / decoded).resolve()
        target_source = source_lookup.get(target)
        if target_source is None:
            project_target = (ROOT / decoded).resolve()
            target_source = source_lookup.get(project_target)
        if target_source is None:
            return match.group(0)
        translated_target = localized_path(target_source, language)
        relative = os.path.relpath(translated_target, start=output.parent)
        encoded = urllib.parse.quote(relative, safe="/._-")
        if separator:
            encoded += f"#{fragment}"
        return f"{match.group(1)}{encoded}{match.group(3)}"

    return LINK_RE.sub(replace, text)


def markdown_headings(text: str) -> list[tuple[str, str]]:
    headings: list[tuple[str, str]] = []
    for is_code, segment in split_prose_and_code(text):
        if is_code:
            continue
        for line in segment.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                headings.append((match.group(1), match.group(2)))
    return headings


def normalize_heading_markers(text: str) -> str:
    """Repair full-width or unspaced heading markers introduced by translators."""
    result: list[str] = []
    for is_code, segment in split_prose_and_code(text):
        if is_code:
            result.append(segment)
            continue
        lines: list[str] = []
        for line in segment.splitlines(keepends=True):
            line = re.sub(
                r"^(＃{1,6})\s*",
                lambda match: "#" * len(match.group(1)) + " ",
                line,
            )
            line = re.sub(r"^(#{1,6})(?=[^#\s])", r"\1 ", line)
            lines.append(line)
        result.append("".join(lines))
    return "".join(result)


def normalize_markdown_links(text: str) -> str:
    """Repair link punctuation/spacing that machine translation may localize."""
    result: list[str] = []
    for is_code, segment in split_prose_and_code(text):
        if is_code:
            result.append(segment)
            continue
        segment = re.sub(r"【([^】\n]+)】\(([^)\n]+)\)", r"[\1](\2)", segment)
        segment = re.sub(r"\]\s+\(([^)\n]+)\)", r"](\1)", segment)
        result.append(segment)
    return "".join(result)


def github_slugs(headings: list[tuple[str, str]]) -> list[str]:
    counts: dict[str, int] = {}
    slugs: list[str] = []
    for _, title in headings:
        title = re.sub(r"<[^>]+>", "", title)
        title = re.sub(r"[`*_~]", "", title).lower().strip()
        title = "".join(
            character
            for character in title
            if character == "-"
            or character.isspace()
            or character == "_"
            or character.isalnum()
        )
        base = re.sub(r"\s+", "-", title).strip("-")
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        slugs.append(base if occurrence == 0 else f"{base}-{occurrence}")
    return slugs


def add_source_heading_anchors(source_text: str, translated_text: str) -> str:
    source_headings = markdown_headings(source_text)
    translated_headings = markdown_headings(translated_text)
    if len(source_headings) != len(translated_headings):
        print(
            "warning: heading count changed during translation: "
            f"{len(source_headings)} -> {len(translated_headings)}; "
            "source anchors skipped",
            flush=True,
        )
        return translated_text
    source_slugs = github_slugs(source_headings)
    translated_slugs = github_slugs(translated_headings)
    anchors = [
        f'<a id="{slug}"></a>' if slug and slug != translated_slug else ""
        for slug, translated_slug in zip(source_slugs, translated_slugs)
    ]
    anchor_index = 0
    result: list[str] = []
    for is_code, segment in split_prose_and_code(translated_text):
        if is_code:
            result.append(segment)
            continue
        lines = segment.splitlines(keepends=True)
        for line in lines:
            if re.match(r"^#{1,6}\s+.+?\s*#*\s*$", line.rstrip("\r\n")):
                anchor = anchors[anchor_index]
                anchor_index += 1
                if anchor:
                    ending = "\r\n" if line.endswith("\r\n") else "\n"
                    result.append(f"{anchor}{ending}")
            result.append(line)
    return "".join(result)


def is_predominantly_english(text: str) -> bool:
    prose = "".join(segment for is_code, segment in split_prose_and_code(text) if not is_code)
    han_count = len(re.findall(r"[\u3400-\u9fff]", prose))
    latin_words = len(re.findall(r"\b[A-Za-z]{3,}\b", prose))
    return latin_words >= 20 and han_count < max(10, latin_words // 20)


def localize_one(
    source: Path,
    language: str,
    cache: TranslationCache,
    source_lookup: dict[Path, Path],
) -> tuple[Path, str]:
    original = remove_navigation(source.read_text(encoding="utf-8"))
    output = localized_path(source, language)
    if language == "en" and is_predominantly_english(original):
        translated = original
    else:
        translated = translate_text(original, language, cache)
    translated = normalize_heading_markers(translated)
    translated = normalize_markdown_links(translated)
    translated = rewrite_local_links(translated, source, output, language, source_lookup)
    translated = add_source_heading_anchors(original, translated)
    content = f"{navigation(source)}\n\n{NOTICES[language]}\n\n{translated.lstrip()}"
    if original.endswith("\n") and not content.endswith("\n"):
        content += "\n"
    return output, content


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sources = tracked_document_sources()
    source_lookup = {source.resolve(): source for source in sources}
    cache = TranslationCache(CACHE_PATH)
    jobs = [(source, language) for source in sources for language in LANGUAGES]
    outputs: dict[Path, str] = {}

    print(f"Localizing {len(sources)} documents into {len(LANGUAGES)} languages")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(localize_one, source, language, cache, source_lookup): (
                source,
                language,
            )
            for source, language in jobs
        }
        completed = 0
        for future in as_completed(futures):
            source, language = futures[future]
            output, content = future.result()
            outputs[output] = content
            atomic_write(output, content)
            completed += 1
            print(
                f"[{completed:03d}/{len(jobs):03d}] {language}: "
                f"{source.relative_to(ROOT)}",
                flush=True,
            )

    for source in sources:
        original = remove_navigation(source.read_text(encoding="utf-8"))
        atomic_write(source, f"{navigation(source)}\n\n{original.lstrip()}")

    print(f"Wrote {len(outputs)} translations and updated {len(sources)} source documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
