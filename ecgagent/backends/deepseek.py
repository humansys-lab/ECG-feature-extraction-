"""DeepSeek backend (OpenAI-compatible chat completions).

Four differences from the Anthropic path shape this file, all verified against
the live API rather than assumed:

* **No strict structured outputs.** `response_format: {"type": "json_schema"}`
  returns *"This response_format type is unavailable now"*. Only
  `{"type": "json_object"}` exists, which guarantees syntactically valid JSON
  and nothing about its shape — so the schema goes into the prompt and the
  result is validated locally. On the Anthropic path the API enforces it.
* **One message per tool result.** OpenAI semantics require a `role: "tool"`
  message per `tool_call_id`, where Anthropic wants them batched into one.
* **Tool arguments arrive as a JSON string**, not an object, and a model can
  emit a malformed one — that has to be an error the loop reports, not a crash.
* **`reasoning_content` must round-trip on tool turns.** DeepSeek V4 thinking
  mode requires the provider's reasoning content to be echoed with an
  assistant tool-call message. Dropping it makes the next request fail with a
  400. Non-tool reasoning is retained in the audit but need not be replayed.

Caching is automatic and has no breakpoint API, so there is nothing to place;
`usage.prompt_tokens_details.cached_tokens` reports what was reused.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..evidence.ledger import visible_citations
from ..evidence.model_view import MODEL_EVIDENCE_VIEW_VERSION
from .base import BackendCapabilities, LLMResponse, ToolCall, ToolOutcome

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 16384
# Reasoning turns over a long tool transcript are slow; the default SDK timeout
# is far too tight for the synthesis turn.
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_RETAINED_EVIDENCE_CHARS = 16_000

_PREFETCH_EVIDENCE_PREFIX = (
    "PROGRAM-PREFETCHED PATIENT EVIDENCE. Cite only the displayed Qn ids; "
    "do not construct ev:/ paths:\n"
)
_CITATION_SYSTEM_NOTE = """

# Compact evidence citation interface

Program-prefetched patient measurements use stable short citations such as
`Q1`. Copy the displayed Qn exactly into citation fields. Never construct an
`ev:/` path. The orchestrator expands authorized Qn ids before verification.
"""

_JSON_MODE_NOTE = """Return your answer as a single JSON object and nothing else — no prose, no markdown fence.

It must match this JSON Schema exactly. Every `required` key must be present; do not add keys that are not in `properties`.

{schema}"""


@dataclass
class DeepSeekBackend:
    """DeepSeek chat-completions backend for the ECG agent loop."""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT_S
    max_retries: int = 5
    thinking: bool = True
    reasoning_effort: str = "high"
    max_retained_evidence_chars: int = DEFAULT_MAX_RETAINED_EVIDENCE_CHARS
    user_id: str = "ecgagent"
    client: Any = None
    name: str = "deepseek"
    max_tool_calls_per_turn: int = 4
    # DeepSeek currently provides JSON-object mode, not schema/grammar
    # enforcement. Invalid phase state is repairable locally but is not a hard
    # backend guarantee.
    hard_phase_guards: bool = False
    capabilities: BackendCapabilities = field(
        default=BackendCapabilities(
            native_tool_calls=True,
            structured_output_level="json",
            citation_aliases=True,
            orchestrated_prefetch=True,
            enforce_phase_coverage=True,
            max_parallel_tool_calls=4,
        ),
        init=False,
    )
    turns: list[dict[str, Any]] = field(default_factory=list, repr=False)
    reasoning: list[str] = field(default_factory=list, repr=False)
    _citation_aliases: dict[str, str] = field(default_factory=dict, repr=False)
    _alias_pointers: dict[str, str] = field(default_factory=dict, repr=False)
    _citation_sequence: int = field(default=0, repr=False)
    _tool_context_trace: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.capabilities = BackendCapabilities(
            native_tool_calls=True,
            structured_output_level="json",
            citation_aliases=True,
            orchestrated_prefetch=True,
            enforce_phase_coverage=True,
            max_parallel_tool_calls=max(1, int(self.max_tool_calls_per_turn)),
        )
        if self.client is not None:
            return
        key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "no DeepSeek credentials found. Set DEEPSEEK_API_KEY in the "
                "environment, or pass api_key= to DeepSeekBackend."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                f"the `openai` package is required for the DeepSeek backend ({exc}); "
                "pip install openai"
            ) from exc
        self.client = OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    # -- request ------------------------------------------------------------
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
        request_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system + _CITATION_SYSTEM_NOTE},
            *messages,
        ]
        if response_schema is not None and not require_tool_call:
            # The schema note is request-local: it steers this turn without
            # entering the conversation history the loop carries forward.
            request_messages.append(
                {
                    "role": "user",
                    "content": _JSON_MODE_NOTE.format(
                        schema=json.dumps(response_schema, ensure_ascii=False, indent=2)
                    ),
                }
            )

        # The live API rejects thinking together with tool_choice=required.
        # Compact planning is a bounded routing/selection step rather than a
        # clinical adjudication: at its 1,200-token cap DeepSeek spent the
        # entire completion on hidden reasoning and returned no JSON in two
        # consecutive live probes. Preserve thinking for the evidence-backed
        # adjudication where it can improve clinical synthesis.
        compact_plan_turn = any(
            "COMPACT PLAN" in str(message.get("content") or "")
            for message in messages
            if isinstance(message, Mapping)
        )
        request_thinking = bool(
            self.thinking and not require_tool_call and not compact_plan_turn
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": request_messages,
            "extra_body": {
                "thinking": {
                    "type": "enabled" if request_thinking else "disabled"
                },
                "user_id": self.user_id,
            },
        }
        if request_thinking:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools:
            kwargs["tools"] = [_to_openai_tool(tool) for tool in tools]
            if require_tool_call:
                kwargs["tool_choice"] = "required"
        if response_schema is not None and not require_tool_call:
            # json_object guarantees parseable JSON, not the right JSON.
            kwargs["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message
        usage = _usage_dict(completion)

        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            self.reasoning.append(reasoning)

        self.turns.append(
            {
                "model": completion.model,
                "finish_reason": choice.finish_reason,
                "usage": usage,
            }
        )

        tool_calls: list[ToolCall] = []
        malformed: list[str] = []
        for call in message.tool_calls or []:
            arguments, error = _parse_arguments(call.function.arguments)
            if error:
                malformed.append(f"{call.function.name}: {error}")
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=tuple(tool_calls),
            stop_reason=_stop_reason(choice.finish_reason),
            model=completion.model,
            usage=usage,
            refusal=None,
            # Rebuilt rather than echoed: `reasoning_content` rides on the
            # provider object and is rejected if sent back.
            raw_content={
                "content": message.content or "",
                "reasoning_content": reasoning or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in (message.tool_calls or [])
                ],
                "malformed_arguments": malformed,
            },
        )

    # -- message construction ----------------------------------------------
    def assistant_turn(self, response: LLMResponse) -> dict[str, Any]:
        raw = response.raw_content or {}
        message: dict[str, Any] = {"role": "assistant", "content": raw.get("content", "")}
        if raw.get("tool_calls"):
            message["tool_calls"] = raw["tool_calls"]
            if raw.get("reasoning_content"):
                message["reasoning_content"] = raw["reasoning_content"]
        return message

    def tool_result_turn(self, outcomes: Sequence[ToolOutcome]) -> list[dict[str, Any]]:
        if outcomes and all(
            str(outcome.call_id).startswith("prefetch-") for outcome in outcomes
        ):
            packet = self._prefetch_packet(outcomes)
            return [{"role": "user", "content": packet}]

        # Native calls still require one role=tool message per call id.
        return [
            {
                "role": "tool",
                "tool_call_id": outcome.call_id,
                "content": self._alias_outcome_text(outcome),
            }
            for outcome in outcomes
        ]

    def user_turn(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def _citation_alias(self, pointer: str) -> str:
        pointer = str(pointer).removeprefix("ev:")
        existing = self._citation_aliases.get(pointer)
        if existing is not None:
            return existing
        self._citation_sequence += 1
        alias = f"Q{self._citation_sequence}"
        self._citation_aliases[pointer] = alias
        self._alias_pointers[alias] = pointer
        return alias

    @staticmethod
    def _citation_label(pointer: str) -> str:
        parts = [
            part
            for part in str(pointer).removeprefix("ev:").split("/")
            if part
        ]
        if not parts:
            return "evidence"
        if len(parts) >= 4 and parts[0] == "representative_leads":
            return f"{parts[1]}.{parts[-1]}"
        if len(parts) >= 3 and parts[0] == "p_wave_assessments":
            return f"p_assessment[{parts[1]}].{parts[-1]}"
        if len(parts) >= 4 and parts[:2] == ["rhythm_inputs", "p_events"]:
            return f"p_event[{parts[2]}].{parts[-1]}"
        if len(parts) >= 3 and parts[0] == "beat_features":
            return f"beat[{parts[1]}].{parts[-1]}"
        if parts[0] == "global_features":
            return f"global.{parts[-1]}"
        return ".".join(parts[-3:])[-72:]

    def _aliased_result(self, outcome: ToolOutcome) -> Any:
        raw = str(outcome.model_text or outcome.text or "")
        candidates = tuple(
            outcome.model_citations
            if outcome.model_text is not None
            else outcome.citations
        )
        present = visible_citations(raw, candidates)
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
            present_set = set(present)
            for atom in payload["evidence"]:
                if not isinstance(atom, dict):
                    continue
                pointer = str(atom.get("citation") or "").removeprefix("ev:")
                if pointer not in present_set:
                    continue
                atom["citation"] = self._citation_alias(pointer)
                atom["label"] = self._citation_label(pointer)
            return payload
        rendered = raw
        for pointer in sorted(present, key=len, reverse=True):
            rendered = rendered.replace(f"ev:{pointer}", self._citation_alias(pointer))
        return rendered

    def _alias_outcome_text(self, outcome: ToolOutcome) -> str:
        result = self._aliased_result(outcome)
        return (
            json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
            if isinstance(result, (dict, list))
            else str(result)
        )

    def _prefetch_packet(self, outcomes: Sequence[ToolOutcome]) -> str:
        """Pack complete evidence atoms fairly across exact prefetched views."""

        packed: list[dict[str, Any]] = []
        atom_sources: list[tuple[dict[str, Any], list[dict[str, Any]], int]] = []
        for outcome in outcomes:
            result = self._aliased_result(outcome)
            row = {
                "tool": outcome.name,
                "arguments": dict(outcome.arguments),
                "is_error": bool(outcome.is_error),
                "result": result,
            }
            if (
                isinstance(result, dict)
                and str(result.get("contract") or "").startswith(
                    "ecgagent.model-evidence."
                )
                and isinstance(result.get("evidence"), list)
            ):
                atoms = [
                    dict(atom)
                    for atom in result.get("evidence") or []
                    if isinstance(atom, dict)
                ]
                source_omitted = max(
                    0, int(result.get("omitted_atom_count") or 0)
                )
                shell = {
                    key: value
                    for key, value in result.items()
                    if key not in {"evidence", "omitted_atom_count", "omission_policy"}
                }
                shell["evidence"] = []
                shell["omitted_atom_count"] = len(atoms) + source_omitted
                row["result"] = shell
                atom_sources.append((row, atoms, source_omitted))
            packed.append(row)

        def size() -> int:
            return len(
                json.dumps(
                    packed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )

        max_atoms = max((len(atoms) for _, atoms, _ in atom_sources), default=0)
        for atom_index in range(max_atoms):
            for row, atoms, source_omitted in reversed(atom_sources):
                if atom_index >= len(atoms):
                    continue
                result = row["result"]
                selected = result["evidence"]
                selected.append(atoms[atom_index])
                result["omitted_atom_count"] = (
                    len(atoms) - len(selected) + source_omitted
                )
                if size() > self.max_retained_evidence_chars:
                    selected.pop()
                    result["omitted_atom_count"] = (
                        len(atoms) - len(selected) + source_omitted
                    )

        packet_json = json.dumps(
            packed,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        packet = _PREFETCH_EVIDENCE_PREFIX + packet_json
        for outcome, row in zip(outcomes, packed):
            candidates = tuple(
                outcome.model_citations
                if outcome.model_text is not None
                else outcome.citations
            )
            row_text = json.dumps(
                row, ensure_ascii=False, separators=(",", ":"), default=str
            )
            visible = visible_citations(
                row_text,
                candidates,
                aliases=self._alias_pointers,
                allow_exact=False,
            )
            parts = str(outcome.call_id).split("-", 3)
            self._tool_context_trace.append(
                {
                    "call_id": outcome.call_id,
                    "phase": parts[1] if len(parts) > 2 else "",
                    "tool": outcome.name,
                    "arguments": dict(outcome.arguments),
                    "model_context_text": row_text,
                    "original_chars": len(str(outcome.text or "")),
                    "model_context_chars": len(row_text),
                    "truncated_for_model": len(visible) < len(candidates),
                    "candidate_citations": list(candidates),
                    "visible_citations": list(visible),
                }
            )
        return packet

    def model_visible_citations(
        self,
        outcomes: Sequence[ToolOutcome],
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[str, ...]]:
        return {
            outcome.call_id: visible_citations(
                messages,
                outcome.model_citations
                if outcome.model_text is not None
                else outcome.citations,
                aliases=self._alias_pointers,
                allow_exact=False,
            )
            for outcome in outcomes
            if not outcome.is_error
        }

    def normalize_verdict(self, verdict: dict[str, Any]) -> dict[str, Any]:
        def expand(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): expand(item) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item) for item in value]
            if isinstance(value, tuple):
                return [expand(item) for item in value]
            if isinstance(value, str):
                token = value.strip()
                alias = token[3:] if token.startswith("ev:") else token
                pointer = self._alias_pointers.get(alias)
                if pointer is not None:
                    return f"ev:{pointer}"
            return value

        normalized = expand(verdict)
        return normalized if isinstance(normalized, dict) else verdict

    def citation_aliases(self) -> dict[str, str]:
        return dict(self._alias_pointers)

    def tool_context_trace(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._tool_context_trace]

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
        totals = self.usage_total()
        cached = totals.get("cached_tokens", 0)
        prompt = totals.get("prompt_tokens", 0)
        if not prompt:
            return "context cache: no usage recorded"
        share = cached / prompt if prompt else 0.0
        return (
            f"context cache: {cached}/{prompt} prompt tokens reused ({share:.0%}) "
            f"across {totals.get('turns', 0)} turns (DeepSeek caches automatically)"
        )

    def audit_config(self) -> dict[str, Any]:
        """Non-secret provider settings needed to reproduce a run."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout,
            "max_retries": self.max_retries,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort if self.thinking else None,
            "user_id": self.user_id,
            "context_compaction": {
                "tool_result_contract": MODEL_EVIDENCE_VIEW_VERSION,
                "max_retained_evidence_chars": self.max_retained_evidence_chars,
                "citation_format": "Q{sequence}",
                "citation_alias_count": len(self._alias_pointers),
            },
        }


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Accept either an Anthropic-shaped or an OpenAI-shaped tool definition."""
    if tool.get("type") == "function" and "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or tool.get("parameters") or {},
        },
    }


def _parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"arguments were not valid JSON ({exc})"
    if not isinstance(parsed, dict):
        return {}, f"arguments were {type(parsed).__name__}, expected an object"
    return parsed, None


def _stop_reason(finish_reason: str | None) -> str:
    return {
        "tool_calls": "tool_use",
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "refusal",
    }.get(finish_reason or "stop", finish_reason or "end_turn")


def _usage_dict(completion: Any) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    if isinstance(cached, int):
        out["cached_tokens"] = cached
    return out
