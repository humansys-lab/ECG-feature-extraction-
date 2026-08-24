"""Anthropic Messages API backend.

Three choices here are load-bearing and worth stating:

* **Streaming.** Agent turns combine a long cached prefix with adaptive
  thinking, so a non-streaming request can sit past the SDK's HTTP timeout.
  `messages.stream(...).get_final_message()` gives timeout safety without
  making the loop handle stream events.
* **Prompt caching on tools + system.** Render order is tools -> system ->
  messages, so one breakpoint on the last system block caches both. The seven
  tool schemas and the agent instructions are re-sent on every turn of every
  phase; without the breakpoint that prefix is re-billed each time.
* **Verbatim assistant echo.** `raw_content` carries the provider's own content
  blocks back. Thinking blocks must round-trip unmodified on these models, so
  the loop must never rebuild an assistant turn from its text.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import BackendCapabilities, LLMResponse, ToolCall, ToolOutcome

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "high"
DEFAULT_BASE_URL = "https://api.anthropic.com"

# Server-side refusal fallback: on a policy decline the API re-runs the request
# on a fallback model chosen by refusal category, instead of handing back a
# refusal the loop would have to abandon the record over.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AnthropicBackend:
    """Claude backend for the ECG agent loop."""

    model: str = DEFAULT_MODEL
    base_url: str | None = None
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    enable_fallbacks: bool = True
    enable_caching: bool = True
    client: Any = None
    name: str = "anthropic"
    max_tool_calls_per_turn: int = 4
    hard_phase_guards: bool = True
    capabilities: BackendCapabilities = field(
        default=BackendCapabilities(
            native_tool_calls=True,
            structured_output_level="schema",
            enforce_phase_coverage=True,
            max_parallel_tool_calls=4,
        ),
        init=False,
    )
    turns: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.base_url = str(
            self.base_url
            or os.getenv("ANTHROPIC_BASE_URL", "").strip()
            or DEFAULT_BASE_URL
        ).strip()
        self.capabilities = BackendCapabilities(
            native_tool_calls=True,
            structured_output_level=("schema" if self.hard_phase_guards else "json"),
            enforce_phase_coverage=True,
            max_parallel_tool_calls=max(1, int(self.max_tool_calls_per_turn)),
        )
        if self.client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "the `anthropic` package is required for the Anthropic backend "
                    f"({exc}); pip install anthropic"
                ) from exc
            # Zero-arg: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
            # `ant auth login` profile, in that order.
            # Pin the endpoint resolved before construction. This prevents a
            # hidden SDK/profile override from diverging from the endpoint
            # authorized and recorded by the caller.
            self.client = anthropic.Anthropic(base_url=self.base_url)

    # -- request ------------------------------------------------------------
    def _system_blocks(self, system: str) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if self.enable_caching:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        require_tool_call: bool = False,
    ) -> LLMResponse:
        output_config: dict[str, Any] = {"effort": self.effort}
        if response_schema is not None and not require_tool_call:
            output_config["format"] = {"type": "json_schema", "schema": response_schema}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": self._system_blocks(system),
            "messages": list(messages),
            "output_config": output_config,
            # Adaptive is the default on this model family; stated explicitly so
            # the intent survives a model swap. No temperature/top_p/top_k —
            # those are rejected on Opus 5 and Fable 5.
            "thinking": {"type": "adaptive"},
        }
        if tools:
            kwargs["tools"] = list(tools)
            if require_tool_call:
                kwargs["tool_choice"] = {"type": "any"}
        if self.enable_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            # extra_body rather than a typed param: the field is newer than some
            # installed SDK versions, and an unknown kwarg would TypeError.
            kwargs["extra_body"] = {"fallbacks": "default"}

        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except TypeError as exc:
            # The SDK resolves credentials lazily, so a missing key surfaces
            # here as a bare TypeError partway through a run rather than at
            # construction. Translate it into something actionable.
            if "authentication" not in str(exc).lower():
                raise
            raise RuntimeError(
                "no Anthropic credentials found. Set ANTHROPIC_API_KEY, or run "
                "`ant auth login` to store a profile the SDK picks up automatically."
            ) from exc

        self.turns.append(
            {
                "model": getattr(message, "model", self.model),
                "stop_reason": message.stop_reason,
                "usage": _usage_dict(message),
            }
        )
        return _to_response(message, fallback_model=self.model)

    # -- message construction ----------------------------------------------
    def assistant_turn(self, response: LLMResponse) -> dict[str, Any]:
        if response.raw_content is None:
            raise ValueError("assistant turn requires provider content blocks")
        return {"role": "assistant", "content": response.raw_content}

    def tool_result_turn(self, outcomes: Sequence[ToolOutcome]) -> list[dict[str, Any]]:
        # One message holding every result: splitting them across messages
        # trains the model out of requesting parallel tool calls.
        blocks: list[dict[str, Any]] = []
        for outcome in outcomes:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": outcome.call_id,
                "content": outcome.text,
            }
            if outcome.is_error:
                block["is_error"] = True
            blocks.append(block)
        return [{"role": "user", "content": blocks}]

    def user_turn(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    # -- reporting ----------------------------------------------------------
    def usage_total(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for turn in self.turns:
            for key, value in (turn.get("usage") or {}).items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        totals["turns"] = len(self.turns)
        return totals

    def cache_report(self) -> str:
        """One line on whether the tools+system prefix is actually being reused."""
        totals = self.usage_total()
        read = totals.get("cache_read_input_tokens", 0)
        written = totals.get("cache_creation_input_tokens", 0)
        uncached = totals.get("input_tokens", 0)
        if not (read or written):
            return "prompt cache: no cache activity (prefix likely under the minimum)"
        return (
            f"prompt cache: {read} tokens read, {written} written, {uncached} uncached "
            f"across {totals.get('turns', 0)} turns"
        )

    def audit_config(self) -> dict[str, Any]:
        """Non-secret provider settings needed to reproduce a run."""
        from ..privacy import resolve_backend_privacy

        endpoint = resolve_backend_privacy(
            "anthropic",
            anthropic_base_url=self.base_url,
        ).audit(allowed=True)["endpoint"]
        return {
            "model": self.model,
            "endpoint": endpoint,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "fallbacks_enabled": self.enable_fallbacks,
            "prompt_caching_enabled": self.enable_caching,
        }


def _usage_dict(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return {k: v for k, v in usage.model_dump().items() if isinstance(v, int)}
    return {}


def _to_response(message: Any, fallback_model: str) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in message.content or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            arguments = block.input if isinstance(block.input, dict) else {}
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))

    refusal = None
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        explanation = getattr(details, "explanation", None) if details else None
        refusal = explanation or f"declined (category={category})"

    return LLMResponse(
        text="\n".join(text_parts).strip(),
        tool_calls=tuple(tool_calls),
        stop_reason=message.stop_reason or "end_turn",
        model=getattr(message, "model", fallback_model),
        usage=_usage_dict(message),
        refusal=refusal,
        raw_content=message.content,
    )


def parse_json_text(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model reply.

    With `response_schema` the whole reply is valid JSON, but the synthesis turn
    can still be truncated by `max_tokens`; returning None lets the loop report
    that rather than raising. Some JSON-object providers have also emitted a
    complete object with a literal newline inside a quoted clinical note. A
    second parse with ``strict=False`` accepts only those unescaped control
    characters; contract validation and evidence verification still run on the
    resulting object.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.endswith("```"):
            candidate = candidate[: candidate.rfind("```")]
    parsed: Any = None
    for strict in (True, False):
        try:
            parsed = json.loads(candidate, strict=strict)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        extracted = candidate[start : end + 1]
        for strict in (True, False):
            try:
                parsed = json.loads(extracted, strict=strict)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            return None
    return parsed if isinstance(parsed, dict) else None
