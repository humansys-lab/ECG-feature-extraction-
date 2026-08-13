"""Scripted backend for tests and dry runs.

Lets the whole loop — phases, budgets, tool dispatch, verification, revision —
be exercised without an API key, so loop logic stays testable in CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .base import BackendCapabilities, LLMResponse, ToolCall, ToolOutcome

Script = Sequence[LLMResponse] | Callable[[list[dict[str, Any]]], LLMResponse]


@dataclass
class ScriptedBackend:
    """Replays a fixed sequence of responses, or computes them from the transcript."""

    script: Script = field(default_factory=list)
    name: str = "mock"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _index: int = 0
    capabilities: BackendCapabilities = field(
        default=BackendCapabilities(),
        init=False,
    )

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
        self.calls.append(
            {
                "system_chars": len(system),
                "n_messages": len(messages),
                "n_tools": len(tools),
                "tool_names": [str(tool.get("name") or "") for tool in tools],
                "structured": response_schema is not None,
                "require_tool_call": bool(require_tool_call),
            }
        )
        if callable(self.script):
            return self.script(list(messages))
        if self._index >= len(self.script):
            raise AssertionError(
                f"scripted backend exhausted after {self._index} responses; "
                "the loop asked for another turn"
            )
        response = self.script[self._index]
        self._index += 1
        return response

    def assistant_turn(self, response: LLMResponse) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for call in response.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        return {"role": "assistant", "content": content}

    def tool_result_turn(self, outcomes: Sequence[ToolOutcome]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": outcome.call_id,
                        "content": outcome.text,
                        **({"is_error": True} if outcome.is_error else {}),
                    }
                    for outcome in outcomes
                ],
            }
        ]

    def user_turn(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def usage_total(self) -> dict[str, int]:
        return {"turns": len(self.calls)}

    def cache_report(self) -> str:
        return "prompt cache: not applicable (mock backend)"


def text_response(text: str, stop_reason: str = "end_turn") -> LLMResponse:
    return LLMResponse(text=text, stop_reason=stop_reason, model="mock", raw_content=None)


def tool_response(*calls: tuple[str, dict[str, Any]], text: str = "") -> LLMResponse:
    """Build a tool-use response: `tool_response(("get_measurement", {...}))`."""
    tool_calls = tuple(
        ToolCall(id=f"toolu_mock_{index}", name=name, arguments=args)
        for index, (name, args) in enumerate(calls)
    )
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason="tool_use",
        model="mock",
        raw_content=None,
    )
