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
import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
import warnings
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Mapping, Sequence
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_ecg_core import (  # noqa: E402  (path bootstrap must run first)
    build_context_summary_from_paths,
    diagnostic_gate_policy,
    is_current_features_schema,
    sanitize_report_text,
)
from ecgagent.privacy import (  # noqa: E402
    BackendPrivacyPolicy,
    endpoint_is_literal_loopback,
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
_CONFIGURED_UPSTREAM = os.getenv("CHAT_BASE_URL", "").strip()
UPSTREAM_BASE_URL = (
    _CONFIGURED_UPSTREAM or f"http://127.0.0.1:{PROFILE['port']}/v1"
).rstrip("/")
UPSTREAM_API_KEY = os.getenv("CHAT_API_KEY", "EMPTY")
CONFIGURED_MODEL = os.getenv("CHAT_MODEL", "").strip()


def _environment_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} 必须是 true/false 布尔值")


def _enforce_upstream_privacy(
    endpoint: str,
    *,
    allowed: bool,
    source: str = "configuration",
) -> BackendPrivacyPolicy:
    """Validate, redact and authorize the exact model-server endpoint."""

    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("CHAT_BASE_URL 不是有效 URL") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is not None and not (1 <= port <= 65535)
    ):
        raise RuntimeError(
            "CHAT_BASE_URL 必须是无凭据、无 query/fragment 的有效 HTTP(S) URL"
        )
    policy = BackendPrivacyPolicy(
        backend="webchat-upstream",
        effective_endpoint=endpoint,
        endpoint_source=source,
        external=not endpoint_is_literal_loopback(endpoint),
    )
    if policy.external and not allowed:
        raise RuntimeError(
            "CHAT_BASE_URL 指向外部模型服务；确认授权、数据驻留和隐私要求后，"
            "显式设置 CHAT_ALLOW_EXTERNAL_EGRESS=true"
        )
    return policy


CHAT_ALLOW_EXTERNAL_EGRESS = _environment_flag("CHAT_ALLOW_EXTERNAL_EGRESS")
UPSTREAM_PRIVACY = _enforce_upstream_privacy(
    UPSTREAM_BASE_URL,
    allowed=CHAT_ALLOW_EXTERNAL_EGRESS,
    source="environment" if _CONFIGURED_UPSTREAM else "profile_default",
)

MAX_UPLOAD_BYTES = int(os.getenv("CHAT_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
MAX_ATTACHMENT_CHARS = int(os.getenv("CHAT_MAX_ATTACHMENT_CHARS", "60000"))
MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "64"))
MAX_MESSAGE_CHARS = int(os.getenv("CHAT_MAX_MESSAGE_CHARS", "16000"))
MAX_TOTAL_MESSAGE_CHARS = int(os.getenv("CHAT_MAX_TOTAL_MESSAGE_CHARS", "120000"))
MAX_SYSTEM_CHARS = int(os.getenv("CHAT_MAX_SYSTEM_CHARS", "12000"))
MAX_ATTACHMENTS_PER_MESSAGE = int(os.getenv("CHAT_MAX_ATTACHMENTS_PER_MESSAGE", "8"))
MAX_ATTACHMENTS_PER_SESSION = int(os.getenv("CHAT_MAX_ATTACHMENTS_PER_SESSION", "24"))
MAX_ATTACHMENTS_GLOBAL = int(os.getenv("CHAT_MAX_ATTACHMENTS_GLOBAL", "256"))
MAX_SESSION_UPLOAD_BYTES = int(
    os.getenv("CHAT_MAX_SESSION_UPLOAD_BYTES", str(64 * 1024 * 1024))
)
MAX_TOTAL_UPLOAD_BYTES = int(
    os.getenv("CHAT_MAX_TOTAL_UPLOAD_BYTES", str(256 * 1024 * 1024))
)
ATTACHMENT_TTL_SECONDS = max(60, int(os.getenv("CHAT_ATTACHMENT_TTL_SECONDS", "3600")))
MAX_IMAGE_PIXELS = int(os.getenv("CHAT_MAX_IMAGE_PIXELS", str(25_000_000)))
MAX_IMAGE_EDGE = int(os.getenv("CHAT_MAX_IMAGE_EDGE", "8192"))
MAX_GENERATION_TOKENS = int(os.getenv("CHAT_MAX_GENERATION_TOKENS", "32768"))
MAX_CONTEXT_CHARS = int(os.getenv("CHAT_MAX_CONTEXT_CHARS", "180000"))
MAX_CONCURRENT_CHATS = max(1, int(os.getenv("CHAT_MAX_CONCURRENT_CHATS", "2")))
MAX_CONCURRENT_UPLOADS = max(1, int(os.getenv("CHAT_MAX_CONCURRENT_UPLOADS", "4")))
MAX_REQUESTS_PER_MINUTE = max(1, int(os.getenv("CHAT_MAX_REQUESTS_PER_MINUTE", "120")))
MAX_RATE_BUCKETS = max(64, int(os.getenv("CHAT_MAX_RATE_BUCKETS", "4096")))
CHAT_AUTH_TOKEN = os.getenv("CHAT_AUTH_TOKEN", "").strip()
CONFIGURED_UI_HOST = os.getenv("CHAT_UI_HOST", "127.0.0.1").strip() or "127.0.0.1"
CONFIGURED_ALLOWED_HOSTS = tuple(
    value.strip()
    for value in os.getenv("CHAT_ALLOWED_HOSTS", "").split(",")
    if value.strip()
)

if CHAT_AUTH_TOKEN and len(CHAT_AUTH_TOKEN) < 16:
    raise RuntimeError("CHAT_AUTH_TOKEN 至少需要 16 个字符")
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
IMAGE_FORMAT_SUFFIXES = {
    "PNG": {".png"},
    "JPEG": {".jpg", ".jpeg"},
    "WEBP": {".webp"},
    "GIF": {".gif"},
    "BMP": {".bmp"},
}
IMAGE_FORMAT_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}

SUPPORTS_IMAGES = bool(PROFILE.get("images"))
SUPPORTS_THINKING = bool(PROFILE.get("reasoning_parser"))
DEFAULT_SYSTEM_PROMPT = str(PROFILE.get("system_prompt") or "")
DEFAULT_MAX_TOKENS = int(PROFILE.get("default_max_tokens") or 2048)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Attachment(StrictModel):
    id: str
    name: str
    kind: str
    chars: int = 0
    truncated: bool = False
    diagnostic_gate_state: str | None = None


class ChatMessage(StrictModel):
    role: Literal["user", "assistant", "model"]
    content: str = Field(default="", max_length=MAX_MESSAGE_CHARS)
    attachments: list[str] = Field(
        default_factory=list,
        max_length=MAX_ATTACHMENTS_PER_MESSAGE,
    )

    @model_validator(mode="after")
    def validate_attachment_ids(self) -> "ChatMessage":
        if any(not re.fullmatch(r"[0-9a-f]{32}", value) for value in self.attachments):
            raise ValueError("附件 ID 格式无效")
        if len(set(self.attachments)) != len(self.attachments):
            raise ValueError("同一消息不能重复引用附件")
        return self


class ChatRequest(StrictModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    system: str = Field(default=DEFAULT_SYSTEM_PROMPT, max_length=MAX_SYSTEM_CHARS)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=64, ge=0, le=256)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=MAX_GENERATION_TOKENS)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    thinking: bool = True

    @model_validator(mode="after")
    def validate_total_size(self) -> "ChatRequest":
        total_chars = len(self.system) + sum(len(message.content) for message in self.messages)
        if total_chars > MAX_TOTAL_MESSAGE_CHARS:
            raise ValueError("消息正文总长度超过限制")
        attachment_count = sum(len(message.attachments) for message in self.messages)
        if attachment_count > MAX_ATTACHMENTS_PER_SESSION:
            raise ValueError("请求引用的附件过多")
        return self


@dataclass
class StoredAttachment:
    """One uploaded file plus what is actually sent to the model."""

    id: str
    name: str
    kind: str
    text: str = ""
    mime: str = ""
    file_path: Path | None = None
    size_bytes: int = 0
    session_id: str = ""
    diagnostic_gate_state: str | None = None
    diagnostic_gate_reasons: list[str] = field(default_factory=list)
    truncated: bool = False
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

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
            diagnostic_gate_state=self.diagnostic_gate_state,
        )


ATTACHMENTS: dict[str, StoredAttachment] = {}

_ATTACHMENT_LOCK = threading.RLock()
_RATE_LOCK = threading.Lock()
_RATE_BUCKETS: dict[str, deque[float]] = {}
_CHAT_LOCK = threading.Lock()
_ACTIVE_CHATS = 0
_UPLOAD_LOCK = threading.Lock()
_ACTIVE_UPLOADS = 0


def _is_loopback_host(host: str) -> bool:
    value = host.strip().strip("[]").lower()
    if value in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _canonical_host(host: str) -> str | None:
    value = str(host or "").strip().strip("[]").rstrip(".").lower()
    if not value or any(character.isspace() for character in value):
        return None
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", value):
            return None
        return value


def _parse_authority(value: str) -> tuple[str, int | None] | None:
    """Parse an HTTP authority without accepting userinfo, paths, or lists."""

    raw = str(value or "").strip()
    if not raw or any(token in raw for token in ("/", "?", "#", "@", ",", "\\")):
        return None
    try:
        parsed = urlsplit(f"//{raw}")
        if parsed.username is not None or parsed.password is not None:
            return None
        host = _canonical_host(parsed.hostname or "")
        port = parsed.port
    except ValueError:
        return None
    return (host, port) if host is not None else None


def _request_host_allowed(request: Request) -> bool:
    host_headers = request.headers.getlist("host")
    if len(host_headers) != 1:
        return False
    parsed = _parse_authority(host_headers[0])
    if parsed is None:
        return False
    host, _ = parsed
    configured_hosts = {
        candidate
        for candidate in (
            _canonical_host(CONFIGURED_UI_HOST),
            *(_canonical_host(value) for value in CONFIGURED_ALLOWED_HOSTS),
        )
        if candidate is not None
    }
    if host in configured_hosts:
        return True
    if _is_loopback_host(host):
        return True
    # Starlette/httpx's in-process test authority is not a deployable public
    # hostname. Limit the exception to an already-local ASGI peer.
    client_host = request.client.host if request.client else ""
    return host == "testserver" and _is_loopback_host(client_host)


def _default_origin_port(scheme: str) -> int | None:
    return 80 if scheme == "http" else 443 if scheme == "https" else None


def _effective_origin_port(port: int | None, scheme: str) -> int | None:
    return port if port is not None else _default_origin_port(scheme)


def _api_origin_allowed(request: Request) -> bool:
    origin_headers = request.headers.getlist("origin")
    if not origin_headers:
        return True
    if len(origin_headers) != 1:
        return False
    raw_origin = origin_headers[0]
    try:
        origin = urlsplit(raw_origin.strip())
        origin_host = _canonical_host(origin.hostname or "")
        origin_port = origin.port
    except ValueError:
        return False
    if (
        origin.scheme.lower() not in {"http", "https"}
        or origin_host is None
        or origin.username is not None
        or origin.password is not None
        or origin.path not in {"", "/"}
        or origin.query
        or origin.fragment
    ):
        return False
    host_headers = request.headers.getlist("host")
    if len(host_headers) != 1:
        return False
    request_authority = _parse_authority(host_headers[0])
    if request_authority is None:
        return False
    request_host, request_port = request_authority
    request_scheme = request.url.scheme.lower()
    return (
        origin.scheme.lower() == request_scheme
        and origin_host == request_host
        and _effective_origin_port(origin_port, origin.scheme.lower())
        == _effective_origin_port(request_port, request_scheme)
    )


if _canonical_host(CONFIGURED_UI_HOST) is None:
    raise RuntimeError("CHAT_UI_HOST 不是有效主机名或 IP 地址")
if any(_canonical_host(value) is None for value in CONFIGURED_ALLOWED_HOSTS):
    raise RuntimeError("CHAT_ALLOWED_HOSTS 包含无效主机名")
if not _is_loopback_host(CONFIGURED_UI_HOST) and not CHAT_AUTH_TOKEN:
    raise RuntimeError(
        "CHAT_UI_HOST 指向非回环地址时必须设置至少 16 字符的 CHAT_AUTH_TOKEN"
    )
if CONFIGURED_UI_HOST in {"0.0.0.0", "::", "[::]"} and not CONFIGURED_ALLOWED_HOSTS:
    raise RuntimeError(
        "CHAT_UI_HOST 使用通配监听地址时必须通过 CHAT_ALLOWED_HOSTS 明确列出浏览器 Host"
    )


def _remove_stored_file(stored: StoredAttachment) -> None:
    path = stored.file_path
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    finally:
        try:
            path.parent.rmdir()
        except OSError:
            pass


def _drop_attachment_locked(attachment_id: str) -> None:
    stored = ATTACHMENTS.pop(attachment_id, None)
    if stored is not None:
        _remove_stored_file(stored)


def _purge_expired_locked(now: float | None = None) -> None:
    cutoff = (time.time() if now is None else now) - ATTACHMENT_TTL_SECONDS
    for attachment_id, stored in list(ATTACHMENTS.items()):
        if stored.last_accessed < cutoff:
            _drop_attachment_locked(attachment_id)


def _usage_locked(session_id: str | None = None) -> tuple[int, int]:
    rows = [
        stored
        for stored in ATTACHMENTS.values()
        if session_id is None or stored.session_id == session_id
    ]
    return len(rows), sum(stored.size_bytes for stored in rows)


def _evict_lru_locked(session_id: str | None = None) -> bool:
    candidates = [
        stored
        for stored in ATTACHMENTS.values()
        if session_id is None or stored.session_id == session_id
    ]
    if not candidates:
        return False
    oldest = min(candidates, key=lambda item: (item.last_accessed, item.created_at))
    _drop_attachment_locked(oldest.id)
    return True


def _make_upload_room_locked(session_id: str, incoming_bytes: int) -> None:
    _purge_expired_locked()
    if incoming_bytes > MAX_SESSION_UPLOAD_BYTES or incoming_bytes > MAX_TOTAL_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过附件存储配额")
    while True:
        count, used = _usage_locked(session_id)
        if count < MAX_ATTACHMENTS_PER_SESSION and used + incoming_bytes <= MAX_SESSION_UPLOAD_BYTES:
            break
        if not _evict_lru_locked(session_id):
            raise HTTPException(status_code=413, detail="当前会话附件配额不足")
    while True:
        count, used = _usage_locked()
        if count < MAX_ATTACHMENTS_GLOBAL and used + incoming_bytes <= MAX_TOTAL_UPLOAD_BYTES:
            break
        if not _evict_lru_locked():
            raise HTTPException(status_code=503, detail="服务器附件存储配额不足")


def _clear_registered_attachments() -> None:
    with _ATTACHMENT_LOCK:
        for attachment_id in list(ATTACHMENTS):
            _drop_attachment_locked(attachment_id)


def _cleanup_orphan_uploads() -> None:
    try:
        UPLOAD_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        UPLOAD_DIR.chmod(0o700)
    except OSError:
        return
    cutoff = time.time() - ATTACHMENT_TTL_SECONDS
    for child in UPLOAD_DIR.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
        except OSError:
            continue


async def _attachment_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(min(60, ATTACHMENT_TTL_SECONDS))
        with _ATTACHMENT_LOCK:
            _purge_expired_locked()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _cleanup_orphan_uploads()
    cleanup_task = asyncio.create_task(_attachment_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        _clear_registered_attachments()


def _rate_allowed(key: str, now: float | None = None) -> bool:
    current = time.monotonic() if now is None else now
    with _RATE_LOCK:
        cutoff = current - 60.0
        for stale_key, stale_bucket in list(_RATE_BUCKETS.items()):
            while stale_bucket and stale_bucket[0] <= cutoff:
                stale_bucket.popleft()
            if not stale_bucket:
                _RATE_BUCKETS.pop(stale_key, None)
        if key not in _RATE_BUCKETS and len(_RATE_BUCKETS) >= MAX_RATE_BUCKETS:
            oldest_key = min(
                _RATE_BUCKETS,
                key=lambda candidate: _RATE_BUCKETS[candidate][-1],
            )
            _RATE_BUCKETS.pop(oldest_key, None)
        bucket = _RATE_BUCKETS.setdefault(key, deque())
        if len(bucket) >= MAX_REQUESTS_PER_MINUTE:
            return False
        bucket.append(current)
        return True


def _reserve_chat_slot() -> bool:
    global _ACTIVE_CHATS
    with _CHAT_LOCK:
        if _ACTIVE_CHATS >= MAX_CONCURRENT_CHATS:
            return False
        _ACTIVE_CHATS += 1
        return True


def _release_chat_slot() -> None:
    global _ACTIVE_CHATS
    with _CHAT_LOCK:
        _ACTIVE_CHATS = max(0, _ACTIVE_CHATS - 1)


def _reserve_upload_slot() -> bool:
    global _ACTIVE_UPLOADS
    with _UPLOAD_LOCK:
        if _ACTIVE_UPLOADS >= MAX_CONCURRENT_UPLOADS:
            return False
        _ACTIVE_UPLOADS += 1
        return True


def _release_upload_slot() -> None:
    global _ACTIVE_UPLOADS
    with _UPLOAD_LOCK:
        _ACTIVE_UPLOADS = max(0, _ACTIVE_UPLOADS - 1)

app = FastAPI(
    title=f"{PROFILE['label']} Chat",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _apply_browser_security_headers(response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


@app.middleware("http")
async def revalidate_assets(request: Request, call_next):
    """Force the browser to revalidate the UI on every load.

    Without this a cached stylesheet or script survives an edit-and-reload
    cycle, which turns a fixed bug into a bug that looks unfixed.
    """
    is_api_request = request.url.path.startswith("/api/")
    client_host = request.client.host if request.client else ""
    if is_api_request and not CHAT_AUTH_TOKEN and not _is_loopback_host(client_host):
        return _apply_browser_security_headers(
            JSONResponse(
                status_code=403,
                content={"detail": "未配置认证的服务只允许本机访问"},
            )
        )
    if not _request_host_allowed(request):
        return _apply_browser_security_headers(
            JSONResponse(
                status_code=400,
                content={"detail": "Host 不在本地聊天服务允许列表中"},
            )
        )
    if is_api_request:
        if not _api_origin_allowed(request):
            return _apply_browser_security_headers(
                JSONResponse(
                    status_code=403,
                    content={"detail": "浏览器 Origin 必须与聊天服务同源"},
                )
            )
        auth_identity = "local-no-token"
        if CHAT_AUTH_TOKEN:
            supplied = request.headers.get("X-Chat-Token", "")
            if not supplied or not hmac.compare_digest(supplied, CHAT_AUTH_TOKEN):
                return _apply_browser_security_headers(
                    JSONResponse(
                        status_code=401,
                        content={"detail": "需要有效的聊天访问令牌"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                )
            auth_identity = hashlib.sha256(CHAT_AUTH_TOKEN.encode("utf-8")).hexdigest()[:16]
        elif not _is_loopback_host(client_host):
            return _apply_browser_security_headers(
                JSONResponse(
                    status_code=403,
                    content={"detail": "未配置认证的服务只允许本机访问"},
                )
            )

        raw_session = request.headers.get("X-Chat-Session", "")
        if raw_session and not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", raw_session):
            return _apply_browser_security_headers(
                JSONResponse(status_code=400, content={"detail": "会话标识格式无效"})
            )
        request.state.chat_session = raw_session or f"local-{client_host or 'unknown'}"

        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            rate_key = f"client:{client_host}:auth:{auth_identity}"
            if not _rate_allowed(rate_key):
                return _apply_browser_security_headers(
                    JSONResponse(
                        status_code=429,
                        content={"detail": "请求过于频繁，请稍后再试"},
                        headers={"Retry-After": "60"},
                    )
                )

    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return _apply_browser_security_headers(response)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=UPSTREAM_BASE_URL,
        headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}"},
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
        # The exact endpoint above is the privacy boundary. Inheriting proxy
        # variables could silently turn an authorized loopback call into
        # external egress.
        trust_env=False,
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


def _uploaded_features_gate(
    path: Path,
    original_name: str | None = None,
) -> dict[str, Any] | None:
    lowered_name = str(original_name or path.name).lower()
    lowered_path = Path(lowered_name)
    feature_named = lowered_path.stem.endswith("_features") or lowered_name in {
        "ecg_input.json",
        "ecg_input.txt",
        "ecg_input.md",
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return diagnostic_gate_policy({}) if feature_named else None
    if not isinstance(data, dict):
        return diagnostic_gate_policy({}) if feature_named else None
    looks_like_features = feature_named or is_current_features_schema(data) or any(
        key in data
        for key in ("global_features", "clinical_interpretation", "interpretation")
    )
    if not looks_like_features:
        return None
    return diagnostic_gate_policy(data)


async def _stream_upload_to_path(file: UploadFile, target: Path) -> int:
    """Copy an upload to disk without buffering the full body in RAM."""
    total = 0
    try:
        with target.open("xb") as output:
            os.chmod(target, 0o600)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        return total
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _validate_image(path: Path, suffix: str) -> str:
    """Verify image structure, declared extension, dimensions, and pixel count."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                if image_format not in IMAGE_FORMAT_SUFFIXES:
                    raise HTTPException(status_code=415, detail="不支持的图片编码")
                if suffix not in IMAGE_FORMAT_SUFFIXES[image_format]:
                    raise HTTPException(status_code=415, detail="图片扩展名与实际编码不一致")
                if width < 1 or height < 1:
                    raise HTTPException(status_code=415, detail="图片尺寸无效")
                if width > MAX_IMAGE_EDGE or height > MAX_IMAGE_EDGE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"图片边长不能超过 {MAX_IMAGE_EDGE} 像素",
                    )
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"图片像素总数不能超过 {MAX_IMAGE_PIXELS}",
                    )
                image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(status_code=413, detail="图片像素规模过大") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=415, detail="文件不是有效图片或图片已损坏") from None
    return IMAGE_FORMAT_MIME[image_format]


def _session_id(request: Request) -> str:
    return str(getattr(request.state, "chat_session", "local-unknown"))


def _get_attachment(attachment_id: str, session_id: str) -> StoredAttachment:
    with _ATTACHMENT_LOCK:
        _purge_expired_locked()
        stored = ATTACHMENTS.get(attachment_id)
        # Return one indistinguishable response for missing and foreign
        # attachments so random identifiers cannot be used as an oracle.
        if stored is None or stored.session_id != session_id:
            raise HTTPException(status_code=404, detail="附件不存在、已过期或属于其他会话")
        stored.last_accessed = time.time()
        return stored


def _message_parts(message: ChatMessage, session_id: str) -> list[dict[str, Any]]:
    """Expand one message into OpenAI content parts."""
    stored = [_get_attachment(key, session_id) for key in message.attachments]
    gate_limited = [
        item
        for item in stored
        if item.diagnostic_gate_state is not None
        and item.diagnostic_gate_state != "pass"
    ]
    if gate_limited:
        states = sorted(
            {str(item.diagnostic_gate_state).upper() for item in gate_limited}
        )
        reasons = sorted(
            {
                reason
                for item in gate_limited
                for reason in (
                    item.diagnostic_gate_reasons
                    or ["unspecified_quality_limitation"]
                )
            }
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"ECG 诊断质量门为 {'/'.join(states)}；此通用聊天界面无法安全强制"
                "领域级限制，因此已在模型调用前阻止推理。请重新采集/修复 ECG，或使用"
                f"受控的分层诊断工作流。原因：{', '.join(reasons)}"
            ),
        )
    images = [item for item in stored if item.is_image]
    documents = [item for item in stored if not item.is_image]

    parts: list[dict[str, Any]] = []
    if documents:
        blocks = [
            f"<<<附件 {index} ({item.kind})>>>\n{item.text}\n<<<附件 {index} 结束>>>"
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
        if item.file_path is None:
            raise HTTPException(status_code=404, detail="图片附件文件已丢失")
        try:
            raw = item.file_path.read_bytes()
        except OSError:
            raise HTTPException(status_code=404, detail="图片附件文件已丢失") from None
        if len(raw) != item.size_bytes or len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=409, detail="图片附件在上传后发生变化")
        data_url = f"data:{item.mime};base64,{base64.b64encode(raw).decode('ascii')}"
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
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


def _expand_messages(messages: Sequence[ChatMessage], session_id: str) -> list[dict[str, Any]]:
    """Inline attachments and enforce alternating roles.

    Gemma's bundled chat template raises if two same-role turns are adjacent,
    so consecutive turns from the same side are merged rather than rejected.
    """
    merged: list[tuple[str, list[dict[str, Any]]]] = []
    for message in messages:
        role = "assistant" if message.role in {"assistant", "model"} else "user"
        parts = _message_parts(message, session_id)
        if not parts:
            continue
        if merged and merged[-1][0] == role:
            merged[-1][1].extend(parts)
        else:
            merged.append((role, parts))
    expanded = [{"role": role, "content": _collapse(parts)} for role, parts in merged]
    text_chars = 0
    for row in expanded:
        content = row["content"]
        if isinstance(content, str):
            text_chars += len(content)
        else:
            text_chars += sum(
                len(str(part.get("text") or ""))
                for part in content
                if part.get("type") == "text"
            )
    if text_chars > MAX_CONTEXT_CHARS:
        raise HTTPException(status_code=413, detail="消息和附件展开后的上下文过长")
    return expanded


def _sse(event: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
async def config() -> dict[str, Any]:
    upstream_audit = UPSTREAM_PRIVACY.audit(
        allowed=CHAT_ALLOW_EXTERNAL_EGRESS
    )
    return {
        "profile": PROFILE["name"],
        "label": PROFILE["label"],
        "icon": PROFILE.get("icon") or "🤖",
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "supports_images": SUPPORTS_IMAGES,
        "supports_thinking": SUPPORTS_THINKING,
        "upstream": upstream_audit["endpoint"]["display"],
        "upstream_external": upstream_audit["external"],
        "upstream_egress_authorized": upstream_audit["authorized"],
        "max_attachment_chars": MAX_ATTACHMENT_CHARS,
        "accept": sorted(
            {".json"} | TEXT_SUFFIXES | (IMAGE_SUFFIXES if SUPPORTS_IMAGES else set())
        ),
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    upstream_display = UPSTREAM_PRIVACY.endpoint_display
    async with _client() as client:
        try:
            response = await client.get("/models", timeout=5.0)
            response.raise_for_status()
        except (httpx.HTTPError, HTTPException) as exc:
            return {"ok": False, "upstream": upstream_display, "error": str(exc)}
        rows = response.json().get("data") or []
        row = rows[0] if rows else {}
        return {
            "ok": bool(rows),
            "upstream": upstream_display,
            "model": row.get("id"),
            "max_model_len": row.get("max_model_len"),
        }


@app.post("/api/upload", response_model=Attachment)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    strip_dx: str = Form("true"),
) -> Attachment:
    attachment_id = uuid.uuid4().hex
    original_name = Path(file.filename or "upload").name[:255]
    suffix = Path(original_name).suffix.lower()
    supported_suffixes = TEXT_SUFFIXES | {".json"} | IMAGE_SUFFIXES
    if suffix not in supported_suffixes:
        supported = sorted(TEXT_SUFFIXES | {".json"} | (IMAGE_SUFFIXES if SUPPORTS_IMAGES else set()))
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型：{original_name}（支持 {', '.join(supported)}）",
        )

    if suffix in IMAGE_SUFFIXES:
        if not SUPPORTS_IMAGES:
            raise HTTPException(
                status_code=415,
                detail=f"{PROFILE['label']} 是纯文本模型，无法读取图片",
            )

    if not _reserve_upload_slot():
        raise HTTPException(status_code=429, detail="上传任务已达到并发上限，请稍后再试")

    target_dir = UPLOAD_DIR / attachment_id
    target: Path | None = None
    registered = False
    try:
        UPLOAD_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        UPLOAD_DIR.chmod(0o700)
        target_dir.mkdir(mode=0o700, exist_ok=False)
        target = target_dir / f"upload{suffix}"
        size_bytes = await _stream_upload_to_path(file, target)

        session_id = _session_id(request)
        if suffix in IMAGE_SUFFIXES:
            mime = _validate_image(target, suffix)
            stored = StoredAttachment(
                id=attachment_id,
                name=original_name,
                kind="image",
                mime=mime,
                file_path=target,
                size_bytes=size_bytes,
                session_id=session_id,
            )
        else:
            kind, text = _summarize_upload(
                target,
                original_name,
                strip_dx.lower() not in {"false", "0", "no"},
            )
            truncated = len(text) > MAX_ATTACHMENT_CHARS
            if truncated:
                text = text[:MAX_ATTACHMENT_CHARS] + "\n[内容过长，已截断]"
            gate = _uploaded_features_gate(target, original_name)
            stored = StoredAttachment(
                id=attachment_id,
                name=original_name,
                kind=kind,
                text=text,
                file_path=target,
                size_bytes=size_bytes,
                session_id=session_id,
                truncated=truncated,
                diagnostic_gate_state=(str(gate.get("state")) if gate is not None else None),
                diagnostic_gate_reasons=(
                    [
                        str(reason)
                        for reason in (
                            list(gate.get("stop_reasons") or [])
                            + list(gate.get("partial_reasons") or [])
                        )
                    ]
                    if gate is not None
                    else []
                ),
            )

        with _ATTACHMENT_LOCK:
            _make_upload_room_locked(session_id, size_bytes)
            ATTACHMENTS[attachment_id] = stored
            registered = True
        return stored.as_public()
    finally:
        if not registered:
            if target is not None:
                target.unlink(missing_ok=True)
            try:
                target_dir.rmdir()
            except OSError:
                pass
        _release_upload_slot()


@app.get("/api/attachments/{attachment_id}")
async def attachment_preview(attachment_id: str, request: Request) -> dict[str, Any]:
    stored = _get_attachment(attachment_id, _session_id(request))
    return {**stored.as_public().model_dump(), "text": stored.text}


@app.delete("/api/attachments/{attachment_id}")
async def attachment_delete(attachment_id: str, request: Request) -> dict[str, bool]:
    session_id = _session_id(request)
    with _ATTACHMENT_LOCK:
        stored = ATTACHMENTS.get(attachment_id)
        if stored is not None and stored.session_id != session_id:
            raise HTTPException(status_code=404, detail="附件不存在、已过期或属于其他会话")
        _drop_attachment_locked(attachment_id)
    return {"ok": True}


async def _stream_chat(
    payload: ChatRequest,
    request: Request,
    messages: Sequence[Mapping[str, Any]],
) -> AsyncIterator[str]:
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
            yield _sse(
                {
                    "type": "error",
                    "error": (
                        "无法连接模型服务（"
                        f"{UPSTREAM_PRIVACY.endpoint_display}）：{detail}"
                    ),
                }
            )
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
    messages = _expand_messages(payload.messages, _session_id(request))
    if not messages:
        raise HTTPException(status_code=400, detail="没有可发送的消息")
    if not _reserve_chat_slot():
        raise HTTPException(status_code=429, detail="生成任务已达到并发上限，请稍后再试")

    async def guarded_stream() -> AsyncIterator[str]:
        try:
            async for event in _stream_chat(payload, request, messages):
                yield event
        finally:
            _release_chat_slot()

    return StreamingResponse(
        guarded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
