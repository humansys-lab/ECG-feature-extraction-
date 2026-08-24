"""Fail-closed privacy policy for model backends and network endpoints."""
from __future__ import annotations

import hashlib
import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit


DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_PROVIDER_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com/v1",
}
_ALIASES = {
    "qwen": "qwen-local",
    "medgemma": "medgemma-local",
}


def normalize_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    return _ALIASES.get(normalized, normalized)


def effective_qwen_base_url(explicit: str | None = None) -> tuple[str, str]:
    """Resolve Qwen's endpoint using the same precedence as its backend."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip(), "explicit"
    configured = os.getenv("QWEN_BASE_URL", "").strip()
    if configured:
        return configured, "environment"
    return DEFAULT_QWEN_BASE_URL, "default"


def effective_anthropic_base_url(explicit: str | None = None) -> tuple[str, str]:
    """Resolve and pin Anthropic's final endpoint before creating the SDK."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip(), "explicit"
    configured = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if configured:
        return configured, "environment"
    return DEFAULT_ANTHROPIC_BASE_URL, "default"


def endpoint_is_literal_loopback(endpoint: str) -> bool:
    """Return true only for a literal loopback host or local Unix socket."""
    try:
        parsed = urlsplit(endpoint)
        scheme = parsed.scheme.lower()
        if scheme in {"unix", "http+unix"}:
            if scheme == "unix":
                return not parsed.netloc and parsed.path.startswith("/")
            socket_path = unquote(parsed.netloc)
            return socket_path.startswith("/") and bool(parsed.path)
        if scheme not in {"http", "https"}:
            return False
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if hostname == "localhost":
            return True
        return ipaddress.ip_address(hostname).is_loopback
    except (ValueError, TypeError):
        return False


def _endpoint_display(endpoint: str) -> str:
    try:
        parsed = urlsplit(endpoint)
        scheme = parsed.scheme.lower()
        if scheme in {"unix", "http+unix"}:
            return f"{scheme}://local-socket"
        hostname = (parsed.hostname or "invalid-host").lower()
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{scheme or 'invalid'}://{hostname}{port}"
    except (ValueError, TypeError):
        return "invalid://redacted-endpoint"


def _endpoint_fingerprint(endpoint: str) -> str:
    return "sha256:" + hashlib.sha256(endpoint.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BackendPrivacyPolicy:
    backend: str
    effective_endpoint: str
    endpoint_source: str
    external: bool

    @property
    def endpoint_display(self) -> str:
        return _endpoint_display(self.effective_endpoint)

    @property
    def endpoint_fingerprint(self) -> str:
        return _endpoint_fingerprint(self.effective_endpoint)

    def audit(self, *, allowed: bool) -> dict[str, object]:
        return {
            "backend": self.backend,
            "external": self.external,
            "authorization_required": self.external,
            "authorized": bool(allowed),
            "endpoint": {
                "display": self.endpoint_display,
                "fingerprint": self.endpoint_fingerprint,
                "source": self.endpoint_source,
            },
        }


def resolve_backend_privacy(
    backend: str,
    *,
    qwen_base_url: str | None = None,
    anthropic_base_url: str | None = None,
) -> BackendPrivacyPolicy:
    normalized = normalize_backend(backend)
    if normalized == "qwen-local":
        endpoint, source = effective_qwen_base_url(qwen_base_url)
        return BackendPrivacyPolicy(
            backend=normalized,
            effective_endpoint=endpoint,
            endpoint_source=source,
            external=not endpoint_is_literal_loopback(endpoint),
        )
    if normalized == "medgemma-local":
        endpoint = "in-process://medgemma-local"
        return BackendPrivacyPolicy(normalized, endpoint, "in_process", False)
    if normalized == "anthropic":
        endpoint, source = effective_anthropic_base_url(anthropic_base_url)
        return BackendPrivacyPolicy(normalized, endpoint, source, True)
    if normalized in _PROVIDER_ENDPOINTS:
        endpoint = _PROVIDER_ENDPOINTS[normalized]
        return BackendPrivacyPolicy(normalized, endpoint, "provider_default", True)

    # Unknown backends fail closed. Callers may still explicitly authorize a
    # third-party backend, but it is never silently treated as local.
    endpoint = f"provider://{normalized or 'unknown'}"
    return BackendPrivacyPolicy(normalized or "unknown", endpoint, "unknown", True)


def external_egress_error(
    backend: str,
    *,
    allowed: bool,
    qwen_base_url: str | None = None,
    anthropic_base_url: str | None = None,
) -> str | None:
    policy = resolve_backend_privacy(
        backend,
        qwen_base_url=qwen_base_url,
        anthropic_base_url=anthropic_base_url,
    )
    if not policy.external or allowed:
        return None
    return (
        f"backend {policy.backend!r} would send model-visible ECG data to "
        f"{policy.endpoint_display}; rerun with --allow-external-egress only "
        "after confirming authorization, data residency, and record-ID handling"
    )
