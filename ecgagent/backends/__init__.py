"""LLM backends for the agent loop.

The loop is provider-neutral: it deals in `LLMResponse` and `ToolCall`, and
lets the backend own message construction, because provider message shapes
differ in ways that matter (Anthropic requires assistant content blocks to be
echoed back verbatim, thinking blocks included).
"""
from __future__ import annotations

from .base import BackendCapabilities, LLMBackend, LLMResponse, ToolCall
from .mock import ScriptedBackend

__all__ = [
    "BackendCapabilities",
    "LLMBackend",
    "LLMResponse",
    "ScriptedBackend",
    "ToolCall",
]


BACKENDS = (
    "anthropic",
    "deepseek",
    "qwen-local",
    "qwen",
    "medgemma-local",
    "medgemma",
    "mock",
)


def build_backend(name: str = "anthropic", **kwargs):
    """Construct a backend by name. Imported lazily so each SDK stays optional."""
    if name == "anthropic":
        from .anthropic_api import AnthropicBackend

        return AnthropicBackend(**kwargs)
    if name == "deepseek":
        from .deepseek import DeepSeekBackend

        return DeepSeekBackend(**kwargs)
    if name in {"qwen-local", "qwen"}:
        from .qwen_local import QwenLocalBackend

        return QwenLocalBackend(**kwargs)
    if name in {"medgemma-local", "medgemma"}:
        from .medgemma_local import MedGemmaLocalBackend

        return MedGemmaLocalBackend(**kwargs)
    if name == "mock":
        return ScriptedBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; available: {', '.join(BACKENDS)}")
