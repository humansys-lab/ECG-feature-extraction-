"""Chat web app for the local model checkpoints.

The models themselves are served by ``vllm serve`` (OpenAI-compatible API).
This process only owns the browser-facing part: it serves the single-page UI,
streams completions back as Server-Sent Events, and turns uploaded ECG
artifacts into the same context summary the batch pipeline feeds MedGemma.

One instance serves one model, selected by ``CHAT_PROFILE`` against
``webchat/models.json``.  Run several instances on different ports to keep
several models available at once.

    CHAT_PROFILE=qwen3.8 .venv/bin/python -m uvicorn webchat.server:app --port 7861

or just ``webchat/start_all.sh qwen3.8``.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_ecg_core import (  # noqa: E402  (path bootstrap must run first)
    build_context_summary_from_paths,
    sanitize_report_text,
)

WEBCHAT_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBCHAT_DIR / "static"
MODELS_FILE = WEBCHAT_DIR / "models.json"


def _load_profile() -> dict[str, Any]:
    """Resolve which model this instance talks to."""
    name = os.getenv("CHAT_PROFILE", "medgemma").strip()
    try:
        profiles = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 {MODELS_FILE}: {exc}") from exc
    if name not in profiles:
        raise RuntimeError(
            f"未知的模型档案 {name!r}；models.json 里可用的是: {', '.join(sorted(profiles))}"
        )
    profile = dict(profiles[name])
    profile["name"] = name
    return profile


PROFILE = _load_profile()

UPLOAD_DIR = Path(
    os.getenv("CHAT_UPLOAD_DIR", str(WEBCHAT_DIR / ".uploads" / PROFILE["name"]))
)
UPSTREAM_BASE_URL = os.getenv(
    "CHAT_BASE_URL", f"http://127.0.0.1:{PROFILE['port']}/v1"
).rstrip("/")
UPSTREAM_API_KEY = os.getenv("CHAT_API_KEY", "EMPTY")
CONFIGURED_MODEL = os.getenv("CHAT_MODEL", "").strip()

MAX_UPLOAD_BYTES = int(os.getenv("CHAT_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
MAX_ATTACHMENT_CHARS = int(os.getenv("CHAT_MAX_ATTACHMENT_CHARS", "60000"))
TEXT_SUFFIXES = {".txt", ".md", ".log", ".csv", ".tsv", ".yaml", ".yml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

SUPPORTS_IMAGES = bool(PROFILE.get("images"))
SUPPORTS_THINKING = bool(PROFILE.get("reasoning_parser"))
DEFAULT_SYSTEM_PROMPT = str(PROFILE.get("system_prompt") or "")
DEFAULT_MAX_TOKENS = int(PROFILE.get("default_max_tokens") or 2048)


class Attachment(BaseModel):
    id: str
    name: str
    kind: str
    chars: int = 0
    truncated: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str = ""
    attachments: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = DEFAULT_MAX_TOKENS
    repetition_penalty: float = 1.0
    thinking: bool = True


@dataclass
class StoredAttachment:
    """One uploaded file plus what is actually sent to the model."""

    id: str
    name: str
    kind: str
    text: str = ""
    data_url: str = ""
    truncated: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    def as_public(self) -> Attachment:
        # Deliberately without `data_url`: the browser already holds the file
        # it just uploaded and renders its own downscaled preview, so echoing
        # megabytes of base64 back would only bloat the response.
        return Attachment(
            id=self.id,
            name=self.name,
            kind=self.kind,
            chars=len(self.text),
            truncated=self.truncated,
        )


ATTACHMENTS: dict[str, StoredAttachment] = {}

app = FastAPI(title=f"{PROFILE['label']} Chat", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def revalidate_assets(request: Request, call_next):
    """Force the browser to revalidate the UI on every load.

    Without this a cached stylesheet or script survives an edit-and-reload
    cycle, which turns a fixed bug into a bug that looks unfixed.
    """
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=UPSTREAM_BASE_URL,
        headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}"},
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
    )


async def _resolve_model(client: httpx.AsyncClient) -> str:
    """Use the configured model id, else the first one the server advertises."""
    if CONFIGURED_MODEL:
        return CONFIGURED_MODEL
    response = await client.get("/models")
    response.raise_for_status()
    rows = response.json().get("data") or []
    if not rows:
        raise HTTPException(status_code=503, detail="模型服务没有返回任何可用模型")
    return str(rows[0].get("id") or "")


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _summarize_upload(path: Path, original_name: str, strip_dx: bool) -> tuple[str, str]:
    """Return ``(kind, text)`` for an uploaded text artifact.

    ECG feature JSON goes through the pipeline's own context builder so the
    chat sees exactly the measurements the batch runs see.  Reports are
    optionally stripped of ``Dx`` labels, matching the batch pipeline's
    anti-leakage handling.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        summary = build_context_summary_from_paths(path, None, "", "en")
        if summary.strip():
            recognized = not summary.lstrip().startswith("[Uploaded JSON was not recognized")
            return ("ecg_features" if recognized else "json", summary)
        return ("json", _decode(path.read_bytes()))

    text = _decode(path.read_bytes())
    if suffix in TEXT_SUFFIXES:
        if strip_dx and suffix in {".txt", ".md", ".log"}:
            cleaned = sanitize_report_text(text, char_limit=MAX_ATTACHMENT_CHARS)
            if cleaned.strip():
                return ("ecg_report", cleaned)
        return ("text", text)

    supported = sorted(TEXT_SUFFIXES | ({".json"} | IMAGE_SUFFIXES if SUPPORTS_IMAGES else {".json"}))
    raise HTTPException(
        status_code=415,
        detail=f"不支持的文件类型：{original_name}（支持 {', '.join(supported)}）",
    )


def _message_parts(message: ChatMessage) -> list[dict[str, Any]]:
    """Expand one message into OpenAI content parts."""
    stored = [ATTACHMENTS[key] for key in message.attachments if key in ATTACHMENTS]
    images = [item for item in stored if item.is_image]
    documents = [item for item in stored if not item.is_image]

    parts: list[dict[str, Any]] = []
    if documents:
        blocks = [
            f"<<<附件 {index}: {item.name} ({item.kind})>>>\n{item.text}\n<<<附件 {index} 结束>>>"
            for index, item in enumerate(documents, start=1)
        ]
        parts.append(
            {
                "type": "text",
                "text": (
                    "以下是用户随本条消息提供的资料，请把它当作已测量的客观数据来引用：\n\n"
                    + "\n\n".join(blocks)
                ),
            }
        )
    for item in images:
        parts.append({"type": "image_url", "image_url": {"url": item.data_url}})
    if message.content.strip():
        parts.append({"type": "text", "text": message.content.strip()})
    return parts


def _collapse(parts: Sequence[Mapping[str, Any]]) -> Any:
    """Use a plain string when there is nothing but text.

    Some chat templates (Gemma's among them) are markedly happier with a
    string than with a one-element part list.
    """
    if all(part.get("type") == "text" for part in parts):
        return "\n\n".join(str(part.get("text") or "") for part in parts).strip()
    return list(parts)


def _expand_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Inline attachments and enforce alternating roles.

    Gemma's bundled chat template raises if two same-role turns are adjacent,
    so consecutive turns from the same side are merged rather than rejected.
    """
    merged: list[tuple[str, list[dict[str, Any]]]] = []
    for message in messages:
        role = "assistant" if message.role in {"assistant", "model"} else "user"
        parts = _message_parts(message)
        if not parts:
            continue
        if merged and merged[-1][0] == role:
            merged[-1][1].extend(parts)
        else:
            merged.append((role, parts))
    return [{"role": role, "content": _collapse(parts)} for role, parts in merged]


def _sse(event: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "profile": PROFILE["name"],
        "label": PROFILE["label"],
        "icon": PROFILE.get("icon") or "🤖",
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "supports_images": SUPPORTS_IMAGES,
        "supports_thinking": SUPPORTS_THINKING,
        "upstream": UPSTREAM_BASE_URL,
        "max_attachment_chars": MAX_ATTACHMENT_CHARS,
        "accept": sorted(
            {".json"} | TEXT_SUFFIXES | (IMAGE_SUFFIXES if SUPPORTS_IMAGES else set())
        ),
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    async with _client() as client:
        try:
            response = await client.get("/models", timeout=5.0)
            response.raise_for_status()
        except (httpx.HTTPError, HTTPException) as exc:
            return {"ok": False, "upstream": UPSTREAM_BASE_URL, "error": str(exc)}
        rows = response.json().get("data") or []
        row = rows[0] if rows else {}
        return {
            "ok": bool(rows),
            "upstream": UPSTREAM_BASE_URL,
            "model": row.get("id"),
            "max_model_len": row.get("max_model_len"),
        }


@app.post("/api/upload", response_model=Attachment)
async def upload(
    file: UploadFile = File(...),
    strip_dx: str = Form("true"),
) -> Attachment:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传的文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
        )

    attachment_id = uuid.uuid4().hex
    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        if not SUPPORTS_IMAGES:
            raise HTTPException(
                status_code=415,
                detail=f"{PROFILE['label']} 是纯文本模型，无法读取图片",
            )
        mime = IMAGE_MIME.get(suffix, "image/png")
        stored = StoredAttachment(
            id=attachment_id,
            name=original_name,
            kind="image",
            data_url=f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
        )
        ATTACHMENTS[attachment_id] = stored
        return stored.as_public()

    target_dir = UPLOAD_DIR / attachment_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / original_name
    target.write_bytes(raw)

    kind, text = _summarize_upload(
        target,
        original_name,
        strip_dx.lower() not in {"false", "0", "no"},
    )
    truncated = len(text) > MAX_ATTACHMENT_CHARS
    if truncated:
        text = text[:MAX_ATTACHMENT_CHARS] + "\n[内容过长，已截断]"

    stored = StoredAttachment(
        id=attachment_id,
        name=original_name,
        kind=kind,
        text=text,
        truncated=truncated,
    )
    ATTACHMENTS[attachment_id] = stored
    return stored.as_public()


@app.get("/api/attachments/{attachment_id}")
async def attachment_preview(attachment_id: str) -> dict[str, Any]:
    stored = ATTACHMENTS.get(attachment_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="附件不存在或服务已重启")
    return {**stored.as_public().model_dump(), "text": stored.text}


@app.delete("/api/attachments/{attachment_id}")
async def attachment_delete(attachment_id: str) -> dict[str, bool]:
    ATTACHMENTS.pop(attachment_id, None)
    return {"ok": True}


async def _stream_chat(payload: ChatRequest, request: Request) -> AsyncIterator[str]:
    messages = _expand_messages(payload.messages)
    if not messages:
        yield _sse({"type": "error", "error": "没有可发送的消息"})
        return

    started = time.monotonic()
    first_token_at: float | None = None
    completion_tokens = 0
    usage: dict[str, Any] = {}

    async with _client() as client:
        try:
            model = await _resolve_model(client)
        except (httpx.HTTPError, HTTPException) as exc:
            detail = getattr(exc, "detail", None) or str(exc)
            yield _sse({"type": "error", "error": f"无法连接模型服务（{UPSTREAM_BASE_URL}）：{detail}"})
            return

        body: dict[str, Any] = {
            "model": model,
            "messages": (
                [{"role": "system", "content": payload.system}] if payload.system.strip() else []
            )
            + messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "max_tokens": payload.max_tokens,
        }
        if payload.top_k > 0:
            body["top_k"] = payload.top_k
        if payload.repetition_penalty != 1.0:
            body["repetition_penalty"] = payload.repetition_penalty
        if SUPPORTS_THINKING:
            # Qwen's template gates its <think> block on this flag.
            body["chat_template_kwargs"] = {"enable_thinking": bool(payload.thinking)}

        yield _sse({"type": "start", "model": model})
        try:
            async with client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    detail = _decode(await response.aread())[:2000]
                    yield _sse({"type": "error", "error": f"模型服务返回 {response.status_code}：{detail}"})
                    return
                async for line in response.aiter_lines():
                    if await request.is_disconnected():
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        # vLLM 0.24 streams the reasoning parser's output as
                        # `reasoning`; older builds and some servers use
                        # `reasoning_content`. Accept either.
                        thinking = delta.get("reasoning") or delta.get("reasoning_content") or ""
                        text = delta.get("content") or ""
                        if thinking:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            completion_tokens += 1
                            yield _sse({"type": "thinking", "text": thinking})
                        if text:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            completion_tokens += 1
                            yield _sse({"type": "delta", "text": text})
        except httpx.HTTPError as exc:
            yield _sse({"type": "error", "error": f"生成过程中断：{exc}"})
            return

    elapsed = time.monotonic() - started
    generated = int(usage.get("completion_tokens") or completion_tokens)
    yield _sse(
        {
            "type": "done",
            "elapsed_s": round(elapsed, 2),
            "ttft_s": round(first_token_at - started, 2) if first_token_at else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens") or completion_tokens,
            "tokens_per_s": round(generated / elapsed, 1) if elapsed > 0 and generated else None,
        }
    )


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat(payload, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
