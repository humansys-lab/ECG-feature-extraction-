"""Native Qwen tool-calling backend for a local vLLM OpenAI server.

This backend is intentionally separate from :mod:`medgemma_local`.  Recent
Qwen chat templates already describe function calls and tool-result turns, and
vLLM can expose them as standard OpenAI ``tool_calls`` when the server starts
with ``--enable-auto-tool-choice --tool-call-parser qwen3_xml``.  Re-wrapping
those calls in the MedGemma JSON envelope wastes context and discards model
training that is directly useful to the agent.

The ECG agent remains responsible for executing tools, enforcing budgets,
recording evidence provenance and verifying the final diagnosis.  This module
only translates between the provider-neutral loop contract and the local
OpenAI-compatible API.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..evidence.ledger import visible_citations
from ..evidence.model_view import MODEL_EVIDENCE_VIEW_VERSION
from .base import BackendCapabilities, LLMResponse, ToolCall, ToolOutcome


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT_S = 600.0
# Qwen 27B can address a much larger physical context window, but exact
# state-machine work degrades well before that limit.  The canonical diagnostic
# ledger already carries every evidence atom that a committed phase selected,
# so the in-phase packet only needs enough room for the *current* decision.
# Citation aliasing below recovers much of the space previously spent copying
# long JSON pointers, allowing this smaller cap without proportionally dropping
# patient facts.
DEFAULT_MAX_RETAINED_EVIDENCE_CHARS = 12000

_INFLIGHT_LEDGER_PREFIX = "QWEN IN-PHASE EVIDENCE LEDGER."
_PHASE_MEMORY_PREFIX = "QWEN CUMULATIVE EVIDENCE MEMORY."
_REVISION_CANDIDATE_PREFIX = "QWEN CURRENT DIAGNOSIS CANDIDATE."
_PREFETCH_EVIDENCE_PREFIX = (
    "PROGRAM-PREFETCHED PATIENT EVIDENCE. Cite only the displayed Qn ids; "
    "do not construct ev:/ paths:\n"
)

_SCHEMA_NOTE = """Return one JSON object following this compact structural contract.
`properties.names` means every listed property uses the shape in `each`.
`allowed_values` refers to exact registered values named in the instructions or state.
Do not wrap the object in markdown and do not add prose outside it.

{schema}"""

_CITATION_SYSTEM_NOTE = """

# Qwen local citation interface

The local backend replaces every model-visible `ev:/...` evidence pointer with
a stable short id such as `Q1`. Treat each `Qn` as the exact citation: copy the
short id unchanged into citation fields and never construct or guess an
`ev:/...` path. The orchestrator expands valid short ids before ledger commit
and final verification. Any general instruction elsewhere about copying an exact
pointer is satisfied by copying its displayed `Qn` id.
"""


@dataclass
class QwenLocalBackend:
    """Qwen served locally by vLLM with its native tool-call parser enabled."""

    model: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT_S
    max_retries: int = 2
    thinking: bool = True
    preserve_thinking: bool = True
    temperature: float = 0.1
    top_p: float = 0.95
    max_retained_evidence_chars: int = DEFAULT_MAX_RETAINED_EVIDENCE_CHARS
    # Once a phase is committed, its selected evidence is materialised in the
    # program-owned ledger snapshot.  Retaining the raw tool packet beside that
    # snapshot duplicated the same facts and was the largest avoidable source
    # of Qwen context load.
    retain_committed_tool_views: bool = False
    client: Any = None
    name: str = "qwen-local"
    # Three views per native turn keeps one evidence packet cognitively small
    # enough for the 27B checkpoint while still allowing useful parallelism.
    max_tool_calls_per_turn: int = 3
    # Tool-enabled phase turns must be free to emit Qwen's native XML tool
    # syntax, so their first terminal candidate is prompt-steered. The agent's
    # hard phase guard validates that candidate and closes tools for a strict
    # json_schema repair if needed. Tool-free turns are constrained directly.
    capabilities: BackendCapabilities = field(
        default=BackendCapabilities(
            native_tool_calls=True,
            structured_output_level="schema",
            context_compaction=True,
            citation_aliases=True,
            phase_memory=True,
            orchestrated_prefetch=True,
            enforce_phase_coverage=True,
            max_parallel_tool_calls=3,
        ),
        init=False,
    )
    turns: list[dict[str, Any]] = field(default_factory=list, repr=False)
    reasoning: list[str] = field(default_factory=list, repr=False)
    _inflight_evidence: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict,
        repr=False,
    )
    _retained_evidence: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )
    _evidence_sequence: int = field(default=0, repr=False)
    _citation_aliases: dict[str, str] = field(default_factory=dict, repr=False)
    _alias_pointers: dict[str, str] = field(default_factory=dict, repr=False)
    _canonical_evidence_aliases: dict[str, str] = field(
        default_factory=dict,
        repr=False,
    )
    _citation_sequence: int = field(default=0, repr=False)
    _phase_history: list[str] = field(default_factory=list, repr=False)
    _current_phase_state: Any = field(default=None, repr=False)
    _tool_context_trace: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.base_url = str(
            self.base_url
            or os.getenv("QWEN_BASE_URL", "").strip()
            or DEFAULT_BASE_URL
        ).strip()
        self.capabilities = BackendCapabilities(
            native_tool_calls=True,
            structured_output_level="schema",
            context_compaction=True,
            citation_aliases=True,
            phase_memory=True,
            orchestrated_prefetch=True,
            enforce_phase_coverage=True,
            max_parallel_tool_calls=max(1, int(self.max_tool_calls_per_turn)),
        )
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"the `openai` package is required for the Qwen backend ({exc}); "
                "pip install openai"
            ) from exc
        # vLLM does not require a secret, but the OpenAI SDK requires a
        # non-empty value. A real key can still be supplied for a protected
        # reverse proxy through QWEN_API_KEY or the constructor.
        key = self.api_key or os.getenv("QWEN_API_KEY") or "EMPTY"
        self.client = OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
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
        request_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system + _CITATION_SYSTEM_NOTE},
            *messages,
        ]
        # Tool-free compact planning is a routing/selection task. Qwen's
        # hidden thinking otherwise consumed the entire short completion cap
        # before emitting the first JSON token (observed on v6). Preserve
        # thinking for the evidence adjudication where clinical synthesis
        # benefits from it.
        compact_plan_turn = any(
            "COMPACT PLAN" in str(message.get("content") or "")
            for message in messages
            if isinstance(message, dict)
        )
        request_thinking = bool(self.thinking and not compact_plan_turn)
        if response_schema is not None and not require_tool_call and tools:
            # With tools open, vLLM cannot apply the terminal JSON grammar
            # without disabling Qwen's native tool-call branch.  Supply only a
            # description-free structural outline here.  Once tools close the
            # real strict schema is enforced by ``response_format`` and does
            # not need to be repeated as another user-message-sized prompt.
            request_messages.append(
                {
                    "role": "user",
                    "content": _SCHEMA_NOTE.format(
                        schema=json.dumps(
                            _compact_schema(response_schema),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                }
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": request_thinking,
                    "preserve_thinking": bool(
                        request_thinking and self.preserve_thinking
                    ),
                }
            },
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(tool) for tool in tools]
            if require_tool_call:
                kwargs["tool_choice"] = "required"
        elif response_schema is not None and not require_tool_call:
            # vLLM's OpenAI server supports strict json_schema decoding. Do not
            # apply it while tools remain open: constraining generated content
            # to terminal JSON would make Qwen's native <tool_call> branch
            # unreachable.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ecgagent_response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message
        usage = _usage_dict(completion)
        reasoning = str(getattr(message, "reasoning_content", "") or "")
        if reasoning:
            self.reasoning.append(reasoning)

        native_calls = list(getattr(message, "tool_calls", None) or [])
        tool_calls: list[ToolCall] = []
        raw_calls: list[dict[str, Any]] = []
        malformed: list[str] = []
        for call in native_calls:
            function = call.function
            raw_arguments = str(getattr(function, "arguments", "") or "")
            arguments, error = _parse_arguments(raw_arguments)
            if error:
                malformed.append(f"{function.name}: {error}")
                # Every provider tool_call id still needs a corresponding tool
                # result. Preserve the call but make schema validation reject
                # it deterministically instead of accidentally executing a
                # no-argument tool with corrupted parameters.
                arguments = {
                    "__tool_argument_parse_error__": error,
                    "__raw_arguments__": raw_arguments[:500],
                }
            tool_calls.append(
                ToolCall(
                    id=str(call.id),
                    name=str(function.name),
                    arguments=arguments,
                )
            )
            raw_calls.append(
                {
                    "id": str(call.id),
                    "type": "function",
                    "function": {
                        "name": str(function.name),
                        "arguments": raw_arguments,
                    },
                }
            )

        refusal = getattr(message, "refusal", None)
        finish_reason = getattr(choice, "finish_reason", None)
        self.turns.append(
            {
                "model": getattr(completion, "model", self.model),
                "finish_reason": finish_reason,
                "usage": usage,
                "tool_calls": len(tool_calls),
                "thinking_enabled": request_thinking,
            }
        )
        return LLMResponse(
            text=str(getattr(message, "content", "") or "").strip(),
            tool_calls=tuple(tool_calls),
            stop_reason=_stop_reason(finish_reason),
            model=str(getattr(completion, "model", self.model) or self.model),
            usage=usage,
            refusal=str(refusal) if refusal else None,
            raw_content={
                "content": str(getattr(message, "content", "") or ""),
                "reasoning_content": reasoning,
                "tool_calls": raw_calls,
                "malformed_arguments": malformed,
            },
        )

    def assistant_turn(self, response: LLMResponse) -> dict[str, Any]:
        raw = response.raw_content or {}
        message: dict[str, Any] = {
            "role": "assistant",
            "content": str(raw.get("content") or ""),
        }
        if raw.get("tool_calls"):
            message["tool_calls"] = raw["tool_calls"]
            # Qwen's multi-step thinking/tool template can preserve this field
            # across a native tool round-trip. Terminal reasoning is omitted
            # from later phases to avoid carrying hidden scratch text forever.
            if raw.get("reasoning_content"):
                message["reasoning_content"] = raw["reasoning_content"]
        return message

    def tool_result_turn(
        self,
        outcomes: Sequence[ToolOutcome],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for outcome in outcomes:
            candidates = tuple(
                outcome.model_citations
                if outcome.model_text is not None
                else outcome.citations
            )
            content = self._alias_model_evidence_text(
                str(outcome.model_text or outcome.text or ""),
                candidates,
            )
            visible = visible_citations(
                content,
                candidates,
                aliases=self._alias_pointers,
                allow_exact=False,
            )
            prefetched = str(outcome.call_id).startswith("prefetch-")
            messages.append(
                {
                    "role": "user" if prefetched else "tool",
                    **(
                        {}
                        if prefetched
                        else {"tool_call_id": outcome.call_id}
                    ),
                    "content": (
                        _PREFETCH_EVIDENCE_PREFIX + content
                        if prefetched
                        else content
                    ),
                }
            )
            prefetched_phase = ""
            if prefetched:
                parts = str(outcome.call_id).split("-", 3)
                prefetched_phase = parts[1] if len(parts) > 2 else ""
            self._tool_context_trace.append(
                {
                    "call_id": outcome.call_id,
                    "phase": prefetched_phase,
                    "tool": outcome.name,
                    "arguments": dict(outcome.arguments),
                    "model_context_text": content,
                    "original_chars": len(str(outcome.text or "")),
                    "model_context_chars": len(content),
                    "truncated_for_model": bool(outcome.model_omitted_count),
                    "raw_touched_citations": list(outcome.citations),
                    "candidate_citations": list(
                        outcome.model_candidate_citations
                        if outcome.model_text is not None
                        else outcome.citations
                    ),
                    "atomic_view_citations": list(
                        outcome.model_citations
                        if outcome.model_text is not None
                        else outcome.citations
                    ),
                    "omitted_atom_count": int(outcome.model_omitted_count),
                    "visible_citations": list(visible),
                }
            )
        return messages

    def user_turn(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def reset_session(self) -> None:
        self.turns.clear()
        self.reasoning.clear()
        self._inflight_evidence.clear()
        self._retained_evidence.clear()
        self._evidence_sequence = 0
        self._citation_aliases.clear()
        self._alias_pointers.clear()
        self._canonical_evidence_aliases.clear()
        self._citation_sequence = 0
        self._phase_history.clear()
        self._current_phase_state = None
        self._tool_context_trace.clear()

    def new_session(self) -> "QwenLocalBackend":
        """Return isolated audit state while reusing the thread-safe client."""

        return QwenLocalBackend(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            thinking=self.thinking,
            preserve_thinking=self.preserve_thinking,
            temperature=self.temperature,
            top_p=self.top_p,
            max_retained_evidence_chars=self.max_retained_evidence_chars,
            retain_committed_tool_views=self.retain_committed_tool_views,
            client=self.client,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
        )

    @staticmethod
    def _call_signature(name: str, arguments: Any) -> str:
        return str(name) + ":" + json.dumps(
            arguments if isinstance(arguments, dict) else {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _citation_alias(self, pointer: str) -> str:
        """Return one short, stable model-visible id for an evidence pointer."""

        pointer = str(pointer).removeprefix("ev:")
        existing = self._citation_aliases.get(pointer)
        if existing is not None:
            return existing
        self._citation_sequence += 1
        alias = f"Q{self._citation_sequence}"
        self._citation_aliases[pointer] = alias
        self._alias_pointers[alias] = pointer
        return alias

    def _alias_model_evidence_text(
        self,
        text: str,
        citations: Sequence[str],
    ) -> str:
        """Replace only value-bearing citations actually present in a view."""

        rendered = str(text or "")
        present = visible_citations(rendered, citations)
        try:
            payload: Any = json.loads(rendered)
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
                # Removing a long JSON pointer must not remove the measurement
                # identity. Without this short label, a list of T amplitudes is
                # just anonymous numbers and Qwen can bind V2's value to lead II.
                atom["label"] = _citation_label(pointer)
            return json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        for pointer in sorted(present, key=len, reverse=True):
            rendered = rendered.replace(
                f"ev:{pointer}",
                self._citation_alias(pointer),
            )
        return rendered

    def _compact_ledger_state(self, state: Any) -> Any:
        """Project the canonical ledger into Qwen's small working-memory view.

        The reducer keeps the full snapshot and audit history.  Qwen needs the
        version, domain/hypothesis state and materialised evidence values, but
        not hashes, record provenance or two parallel citation namespaces.
        """

        if not isinstance(state, dict) or not str(
            state.get("ledger_schema") or ""
        ).startswith("diagnostic-ledger."):
            return state

        evidence = state.get("evidence")
        compact_evidence: dict[str, dict[str, Any]] = {}
        canonical_to_alias: dict[str, str] = {}
        if isinstance(evidence, dict):
            for canonical_id, raw_row in evidence.items():
                if not isinstance(raw_row, dict):
                    continue
                pointer = str(raw_row.get("pointer") or "")
                if not pointer:
                    continue
                alias = self._citation_alias(pointer)
                canonical_to_alias[str(canonical_id)] = alias
                self._canonical_evidence_aliases[str(canonical_id)] = alias
                display_value = raw_row.get("display_value")
                compact_evidence[alias] = _drop_empty(
                    {
                        "value": (
                            display_value
                            if display_value is not None
                            else raw_row.get("raw_value")
                        ),
                        "unit": raw_row.get("unit"),
                        "reliability": raw_row.get("reliability"),
                        "caveats": list(raw_row.get("caveats") or []),
                        "family": raw_row.get("evidence_family"),
                        "seen_phase": raw_row.get("first_seen_phase"),
                        "label": _citation_label(pointer),
                    },
                    keep_false=True,
                )

        def shorten(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): shorten(item) for key, item in value.items()}
            if isinstance(value, list):
                return [shorten(item) for item in value]
            if isinstance(value, tuple):
                return [shorten(item) for item in value]
            if isinstance(value, str):
                return canonical_to_alias.get(
                    value,
                    self._canonical_evidence_aliases.get(value, value),
                )
            return value

        def evidence_refs(value: Any) -> list[str]:
            """Flatten ledger provenance wrappers to the cited Q ids."""

            refs: list[str] = []
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                candidates: Sequence[Any]
                if isinstance(row, dict):
                    raw_candidates = row.get("evidence_ids") or row.get("citations")
                    candidates = (
                        raw_candidates
                        if isinstance(raw_candidates, (list, tuple))
                        else []
                    )
                elif isinstance(row, str):
                    candidates = [row]
                else:
                    candidates = []
                for candidate in candidates:
                    alias = canonical_to_alias.get(
                        str(candidate),
                        self._canonical_evidence_aliases.get(str(candidate)),
                    )
                    if alias is not None and alias not in refs:
                        refs.append(alias)
            return refs

        domains = shorten(state.get("domains") or {})
        hypotheses: list[dict[str, Any]] = []
        for raw_hypothesis in state.get("hypotheses") or []:
            if not isinstance(raw_hypothesis, dict):
                continue
            hypothesis = shorten(raw_hypothesis)
            hypothesis["support"] = evidence_refs(raw_hypothesis.get("support"))
            hypothesis["counterevidence"] = evidence_refs(
                raw_hypothesis.get("counterevidence")
            )
            hypotheses.append(
                _drop_empty(
                    {
                        key: hypothesis.get(key)
                        for key in (
                            "id",
                            "diagnosis_code",
                            "phenotype",
                            "kind",
                            "domains",
                            "status",
                            "support",
                            "counterevidence",
                            "required_evidence",
                            "next_tools",
                            "alternative_explanations",
                            "alternative_group",
                            "refuting_test",
                            "challenge_passed",
                        )
                    },
                    keep_false=True,
                )
            )

        compact = {
            "ledger_schema": state.get("ledger_schema"),
            "version": state.get("version"),
            "current_phase": state.get("current_phase"),
            "frozen": bool(state.get("frozen")),
            "domains": domains,
            "hypotheses": hypotheses,
            "evidence": compact_evidence,
            "targeted_checks": shorten(state.get("targeted_checks") or []),
            "detector_conflicts": list(state.get("detector_conflicts") or []),
            "unresolved": list(state.get("unresolved") or []),
            "allowed_final_placement": shorten(
                state.get("allowed_final_placement") or {}
            ),
        }
        return _drop_empty(compact, keep_false=True)

    def _shorten_known_citations(self, value: Any) -> Any:
        """Use aliases in carried candidates without blessing unknown paths."""

        if isinstance(value, dict):
            return {
                str(key): self._shorten_known_citations(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._shorten_known_citations(item) for item in value]
        if isinstance(value, tuple):
            return [self._shorten_known_citations(item) for item in value]
        if isinstance(value, str):
            token = value.strip()
            pointer = token[3:] if token.startswith("ev:") else token
            alias = self._citation_aliases.get(pointer)
            if alias is not None:
                return alias
            canonical_alias = self._canonical_evidence_aliases.get(token)
            if canonical_alias is not None:
                return canonical_alias
        return value

    def _collect_tool_evidence(
        self,
        phase: str,
        messages: Sequence[dict[str, Any]],
    ) -> None:
        calls: dict[str, tuple[str, dict[str, Any]]] = {}
        for message in messages:
            if str(message.get("role") or "") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                function = function if isinstance(function, dict) else {}
                raw_arguments = function.get("arguments")
                try:
                    arguments = json.loads(str(raw_arguments or "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                calls[str(call.get("id") or "")] = (
                    str(function.get("name") or "unknown"),
                    arguments,
                )

        phase_rows = self._inflight_evidence.setdefault(str(phase), {})
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            prefetched = role == "user" and content.startswith(
                _PREFETCH_EVIDENCE_PREFIX
            )
            if role != "tool" and not prefetched:
                continue
            call_id = str(message.get("tool_call_id") or "")
            self._evidence_sequence += 1
            if prefetched:
                content = content[len(_PREFETCH_EVIDENCE_PREFIX):]
            try:
                parsed_content: Any = json.loads(content)
            except json.JSONDecodeError:
                parsed_content = content
            if prefetched and isinstance(parsed_content, dict):
                name = str(parsed_content.get("tool") or "unknown")
                raw_arguments = parsed_content.get("arguments")
                arguments = (
                    dict(raw_arguments)
                    if isinstance(raw_arguments, Mapping)
                    else {}
                )
            else:
                name, arguments = calls.get(call_id, ("unknown", {}))
            signature = self._call_signature(name, arguments)
            if not isinstance(parsed_content, dict) or not str(
                parsed_content.get("contract") or ""
            ).startswith("ecgagent.model-evidence."):
                parsed_content = content
            existing = phase_rows.get(signature)
            existing_result = (
                existing.get("result")
                if isinstance(existing, dict)
                else None
            )
            existing_is_atomic = bool(
                isinstance(existing_result, dict)
                and str(existing_result.get("contract") or "").startswith(
                    "ecgagent.model-evidence."
                )
                and isinstance(existing_result.get("evidence"), list)
            )
            incoming_is_atomic = bool(
                isinstance(parsed_content, dict)
                and str(parsed_content.get("contract") or "").startswith(
                    "ecgagent.model-evidence."
                )
                and isinstance(parsed_content.get("evidence"), list)
            )
            if existing_is_atomic and not incoming_is_atomic:
                # A weak model may immediately repeat an already successful
                # call. The duplicate error is useful feedback, but must not
                # overwrite the earlier patient evidence in rolling memory.
                continue
            phase_rows[signature] = {
                "tool": name,
                "arguments": arguments,
                "result": parsed_content,
                "sequence": self._evidence_sequence,
            }
            for trace_row in reversed(self._tool_context_trace):
                if str(trace_row.get("call_id") or "") == call_id:
                    trace_row["phase"] = str(phase)
                    break

    def _bounded_evidence_rows(
        self,
        rows: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Pack complete atoms fairly across all retained evidence views.

        Context budgeting must not decide which clinical domain disappears.
        Every atomic tool view therefore gets a small shell first, after which
        complete citation/value/unit atoms are admitted round-robin.  No atom
        is ever split and one wide table cannot evict every other tool view.
        """

        def rendered_size(items: Sequence[dict[str, Any]]) -> int:
            return len(
                json.dumps(
                    items,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )

        ordered = sorted(rows, key=lambda item: int(item.get("sequence") or 0))
        packed: list[dict[str, Any]] = []
        atom_sources: list[tuple[dict[str, Any], list[dict[str, Any]], int]] = []
        raw_rows: list[dict[str, Any]] = []

        for row in ordered:
            public = {
                "tool": row.get("tool"),
                "arguments": row.get("arguments") or {},
                "result": row.get("result") or "",
            }
            result = public["result"]
            is_atomic = (
                isinstance(result, dict)
                and str(result.get("contract") or "").startswith(
                    "ecgagent.model-evidence."
                )
                and isinstance(result.get("evidence"), list)
            )
            if not is_atomic:
                raw_rows.append(public)
                continue

            atoms = [
                dict(atom)
                for atom in result.get("evidence") or []
                if isinstance(atom, dict)
            ]
            source_omitted = max(0, int(result.get("omitted_atom_count") or 0))
            shell_result = {
                key: value
                for key, value in result.items()
                if key not in {"evidence", "omitted_atom_count", "omission_policy"}
            }
            shell_result["evidence"] = []
            shell_result["omitted_atom_count"] = len(atoms) + source_omitted
            shell_result["omission_policy"] = (
                "Only listed atoms are model-visible and citable; use a narrower "
                "tool view for omitted evidence."
            )
            packed_row = {
                "tool": public["tool"],
                "arguments": public["arguments"],
                "result": shell_result,
            }
            packed.append(packed_row)
            atom_sources.append((packed_row, atoms, source_omitted))

        # Ordinary ECG tool arguments are schema-bounded and all shells fit.
        # For pathological third-party rows, retain the newest shells rather
        # than violating the hard context budget.
        while (
            len(packed) > 1
            and rendered_size(packed) > self.max_retained_evidence_chars
        ):
            removed = packed.pop(0)
            atom_sources = [
                source for source in atom_sources if source[0] is not removed
            ]

        max_atoms = max((len(atoms) for _, atoms, _ in atom_sources), default=0)
        for atom_index in range(max_atoms):
            # Recent views win only the tie inside a fair round-robin pass.
            for packed_row, atoms, source_omitted in reversed(atom_sources):
                if atom_index >= len(atoms):
                    continue
                result = packed_row["result"]
                selected_atoms = result["evidence"]
                selected_atoms.append(atoms[atom_index])
                result["omitted_atom_count"] = (
                    len(atoms) - len(selected_atoms) + source_omitted
                )
                if rendered_size(packed) > self.max_retained_evidence_chars:
                    selected_atoms.pop()
                    result["omitted_atom_count"] = (
                        len(atoms) - len(selected_atoms) + source_omitted
                    )

        # Raw compatibility/error rows are non-evidence notes. They consume
        # only space left after atomic patient data has been packed.
        for public in reversed(raw_rows):
            candidate = [*packed, public]
            if rendered_size(candidate) <= self.max_retained_evidence_chars:
                packed.append(public)

        for packed_row, _atoms, _source_omitted in atom_sources:
            result = packed_row["result"]
            if int(result.get("omitted_atom_count") or 0) == 0:
                result.pop("omission_policy", None)
        return packed

    def _record_packed_context(
        self,
        phase: str,
        packed_rows: Sequence[dict[str, Any]],
    ) -> None:
        """Update audit rows from the exact post-compaction evidence packet."""

        by_signature: dict[str, dict[str, Any]] = {}
        for row in packed_rows:
            if not isinstance(row, dict):
                continue
            signature = self._call_signature(
                str(row.get("tool") or "unknown"),
                row.get("arguments") or {},
            )
            by_signature[signature] = row
        for trace_row in self._tool_context_trace:
            if str(trace_row.get("phase") or "") != str(phase):
                continue
            signature = self._call_signature(
                str(trace_row.get("tool") or "unknown"),
                trace_row.get("arguments") or {},
            )
            packed = by_signature.get(signature)
            if packed is None:
                trace_row["model_context_text"] = ""
                trace_row["model_context_chars"] = 0
                trace_row["visible_citations"] = []
                trace_row["truncated_for_model"] = True
                continue
            result = packed.get("result")
            text = json.dumps(
                result,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            candidates = tuple(
                str(pointer)
                for pointer in trace_row.get("candidate_citations") or []
            )
            visible = visible_citations(
                text,
                candidates,
                aliases=self._alias_pointers,
                allow_exact=False,
            )
            omitted = (
                int(result.get("omitted_atom_count") or 0)
                if isinstance(result, dict)
                else max(0, len(candidates) - len(visible))
            )
            trace_row["model_context_text"] = text
            trace_row["model_context_chars"] = len(text)
            trace_row["visible_citations"] = list(visible)
            trace_row["omitted_atom_count"] = omitted
            trace_row["truncated_for_model"] = bool(omitted)

    def compact_phase_context(
        self,
        phase: str,
        transient_messages: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Collapse resolved native tool exchanges into one bounded ledger."""

        self._collect_tool_evidence(phase, transient_messages)
        kept: list[dict[str, Any]] = []
        for message in transient_messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role in {"assistant", "tool"}:
                continue
            if role == "user" and content.startswith(_PREFETCH_EVIDENCE_PREFIX):
                continue
            if content.startswith(_INFLIGHT_LEDGER_PREFIX):
                continue
            kept.append(dict(message))
        rows = list(self._inflight_evidence.get(str(phase), {}).values())
        packed_rows = self._bounded_evidence_rows(rows)
        if packed_rows:
            self._record_packed_context(str(phase), packed_rows)
        if packed_rows:
            kept.append(
                {
                    "role": "user",
                    "content": _INFLIGHT_LEDGER_PREFIX
                    + " Resolved native tool-call pairs were removed; exact full "
                    "results remain in the audit trace. Use only the retained "
                    "patient evidence. Cite its short Qn ids exactly; do not "
                    "construct ev:/ paths:\n"
                    + json.dumps(
                        {
                            "phase": str(phase),
                            "evidence_views": packed_rows,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        return kept

    def phase_memory_turn(
        self,
        phase: str,
        text: str,
        transient_messages: Sequence[dict[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Cache one compact orchestrator-owned ledger snapshot."""

        self._collect_tool_evidence(phase, transient_messages)
        phase_rows = self._inflight_evidence.pop(str(phase), {})
        if self.retain_committed_tool_views:
            self._retained_evidence.update(phase_rows)
        else:
            # The canonical snapshot below has already materialised every
            # selected evidence value. Keeping the source packet as well makes
            # Qwen reconcile two representations of the same patient facts.
            self._retained_evidence.clear()
        self._phase_history.append(str(phase))
        try:
            state: Any = json.loads(str(text or ""))
        except json.JSONDecodeError:
            state = {"conclusion": str(text or "").strip()}
        self._current_phase_state = self._compact_ledger_state(state)
        return [self._phase_memory_message(str(phase))]

    def _phase_memory_message(self, phase: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state_authority": "orchestrator",
            "phase_history": list(self._phase_history),
            "current_phase": str(phase),
            "current_phase_state": self._current_phase_state,
        }
        evidence = self._bounded_evidence_rows(list(self._retained_evidence.values()))
        if evidence:
            payload["uncommitted_evidence_views"] = evidence
        return {
            "role": "user",
            "content": _PHASE_MEMORY_PREFIX
            + " This replaces superseded phase transcripts and resolved "
            "tool-call messages. The state is canonical and its Qn evidence "
            "ids are exact citations. Re-query for omitted details; never "
            "invent an id or ev:/ path:\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }

    @staticmethod
    def is_phase_memory_turn(message: dict[str, Any]) -> bool:
        return str(message.get("content") or "").startswith(_PHASE_MEMORY_PREFIX)

    def compact_revision_context(
        self,
        messages: Sequence[dict[str, Any]],
        current_candidate: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Drop superseded candidate/revision transcripts before a repair."""

        kept: list[dict[str, Any]] = []
        if messages:
            kept.append(dict(messages[0]))
        # A structured revision can re-read measurements but has no commit
        # boundary. Preserve only these still-uncommitted views so a failed
        # revision can repair against the values it just requested.
        for phase_rows in self._inflight_evidence.values():
            self._retained_evidence.update(phase_rows)
        self._inflight_evidence.clear()
        memories = [
            dict(message)
            for message in messages
            if self.is_phase_memory_turn(message)
        ]
        if self._phase_history:
            memory = self._phase_memory_message(self._phase_history[-1])
        else:
            memory = memories[-1] if memories else None
        if memory is not None and memory not in kept:
            kept.append(memory)
        kept.append(
            {
                "role": "user",
                "content": _REVISION_CANDIDATE_PREFIX
                + " Edit only this latest candidate according to the upcoming "
                "deterministic feedback; old candidates were removed:\n"
                + json.dumps(
                    self._shorten_known_citations(dict(current_candidate or {})),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
            }
        )
        return kept

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
            return "vLLM prefix cache: no usage recorded"
        return (
            f"vLLM prefix cache: {cached}/{prompt} prompt tokens reported as "
            f"cached across {totals.get('turns', 0)} turns"
        )

    def audit_config(self) -> dict[str, Any]:
        """Return reproducible non-secret client and required server settings."""

        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout,
            "max_retries": self.max_retries,
            "thinking": self.thinking,
            "preserve_thinking": self.preserve_thinking,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_compaction": {
                "tool_result_contract": MODEL_EVIDENCE_VIEW_VERSION,
                "max_retained_evidence_chars": self.max_retained_evidence_chars,
                "retain_committed_tool_views": self.retain_committed_tool_views,
                "citation_format": "Q{sequence}",
                "citation_alias_count": len(self._alias_pointers),
                "retained_views": len(self._retained_evidence),
                "completed_phases": list(self._phase_history),
            },
            "required_vllm_flags": [
                "--enable-auto-tool-choice",
                "--tool-call-parser qwen3_xml",
                "--reasoning-parser qwen3",
            ],
        }

    def model_visible_citations(
        self,
        outcomes: Sequence[ToolOutcome],
        messages: Sequence[dict[str, Any]],
    ) -> dict[str, tuple[str, ...]]:
        """Authorize only atoms surviving the exact compacted request."""

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
        """Expand Qwen's short ids back to canonical evidence pointers."""

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
        """Return aliases used by deterministic provenance gates."""

        return dict(self._alias_pointers)

    def tool_context_trace(self) -> list[dict[str, Any]]:
        """Expose the exact atomic tool views sent to Qwen."""

        return [dict(row) for row in self._tool_context_trace]


def _citation_label(pointer: str) -> str:
    """Return a compact semantic identity for an aliased evidence pointer."""

    parts = [part for part in str(pointer).removeprefix("ev:").split("/") if part]
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
    if len(parts) >= 2 and parts[0] == "quality":
        return ".".join(("quality", *parts[-2:]))
    if parts[0] == "rhythm_inputs":
        return ".".join(("rhythm", *parts[1:]))[-72:]
    if parts[0] == "metadata":
        return ".".join(parts)[-72:]
    return ".".join(parts[-3:])[-72:]


_SCHEMA_ANNOTATION_KEYS = {
    "$comment",
    "default",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}
_PROMPT_SCHEMA_NOISE_KEYS = {
    "additionalProperties",
    "maxLength",
    "minLength",
    "pattern",
}
_MAX_INLINE_ENUM_VALUES = 24


def _compact_schema(value: Any) -> Any:
    """Build a small structural outline for prompt-only schema copies.

    The original schema is still passed unchanged to vLLM's strict decoder on
    tool-free turns. This projection is only the reminder shown on a native
    tool-enabled turn, where the deterministic phase guard will reject a bad
    terminal candidate and request one strict repair. Disabled array branches,
    repeated property shapes and very large enums therefore do not need to be
    copied verbatim into every exploratory request.
    """

    if isinstance(value, dict):
        if value.get("type") == "array" and value.get("maxItems") == 0:
            return {"type": "array", "maxItems": 0}

        compact: dict[str, Any] = {}
        properties = value.get("properties")
        if isinstance(properties, dict):
            compact_properties = {
                str(key): _compact_schema(item)
                for key, item in properties.items()
            }
            shapes = list(compact_properties.values())
            if (
                len(compact_properties) >= 3
                and shapes
                and all(shape == shapes[0] for shape in shapes[1:])
            ):
                compact["properties"] = {
                    "names": list(compact_properties),
                    "each": shapes[0],
                }
            else:
                compact["properties"] = compact_properties

        for key, item in value.items():
            key = str(key)
            if key == "properties":
                continue
            if key in _SCHEMA_ANNOTATION_KEYS or key in _PROMPT_SCHEMA_NOISE_KEYS:
                continue
            if (
                key == "enum"
                and isinstance(item, list)
                and len(item) > _MAX_INLINE_ENUM_VALUES
            ):
                compact["allowed_values"] = (
                    f"{len(item)} registered values; use the exact value named "
                    "in the instructions or current state"
                )
                continue
            compact[key] = _compact_schema(item)
        return compact
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    if isinstance(value, tuple):
        return [_compact_schema(item) for item in value]
    return value


def _drop_empty(value: dict[str, Any], *, keep_false: bool = False) -> dict[str, Any]:
    """Recursively remove empty presentation fields from model-only state."""

    compact: dict[str, Any] = {}
    for key, raw_item in value.items():
        if isinstance(raw_item, dict):
            item: Any = _drop_empty(raw_item, keep_false=keep_false)
        elif isinstance(raw_item, list):
            item = [
                _drop_empty(entry, keep_false=keep_false)
                if isinstance(entry, dict)
                else entry
                for entry in raw_item
            ]
        else:
            item = raw_item
        if item is None or item == "" or item == [] or item == {}:
            continue
        if item is False and not keep_false:
            continue
        compact[str(key)] = item
    return compact


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
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


__all__ = ["QwenLocalBackend"]
