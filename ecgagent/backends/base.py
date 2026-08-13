"""Provider-neutral contract between the agent loop and an LLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from ..evidence.ledger import visible_citations


StructuredOutputLevel = Literal["none", "json", "schema", "grammar"]


@dataclass(frozen=True)
class BackendCapabilities:
    """Explicit behavior contract used by the orchestrator.

    Optional methods remain available for provider-specific reporting, but
    control-flow decisions must be based on this object rather than a growing
    collection of implicit ``getattr`` feature flags.
    """

    native_tool_calls: bool = True
    structured_output_level: StructuredOutputLevel = "none"
    context_compaction: bool = False
    citation_aliases: bool = False
    phase_memory: bool = False
    orchestrated_prefetch: bool = False
    enforce_phase_coverage: bool = False
    max_parallel_tool_calls: int = 1

    @property
    def hard_phase_guards(self) -> bool:
        return self.structured_output_level in {"schema", "grammar"}


DEFAULT_BACKEND_CAPABILITIES = BackendCapabilities()


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolOutcome:
    """The result of executing a ToolCall, ready to hand back to the model."""

    call_id: str
    text: str
    is_error: bool = False
    name: str = ""
    # Exact evidence pointers touched by the raw tool remain available for
    # audit. Local backends may expose a smaller atomized set (and MedGemma may
    # additionally alias it) before deterministic authorization.
    citations: tuple[str, ...] = ()
    # Canonical local-model representation.  It is built from immutable store
    # values one complete evidence atom at a time; local backends use it in
    # place of human-oriented raw tables.  Raw ``text`` is retained for
    # provider compatibility and the audit trace.
    model_text: str | None = None
    model_citations: tuple[str, ...] = ()
    # Candidates are exact citation tokens found in the human renderer before
    # the atomic per-tool budget was applied. This keeps audit accounting
    # separate from the wider set of values merely touched by the tool.
    model_candidate_citations: tuple[str, ...] = ()
    model_omitted_count: int = 0
    # Local cumulative evidence memory uses arguments to retain distinct
    # modality/profile calls under the same tool name.
    arguments: dict[str, Any] = field(default_factory=dict)


def model_visible_outcome_citations(
    backend: Any,
    outcomes: Sequence[ToolOutcome],
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Return pointer sets visible in the exact next-request messages."""

    custom = getattr(backend, "model_visible_citations", None)
    if callable(custom):
        reported = custom(outcomes, messages)
        if isinstance(reported, Mapping):
            return {
                str(call_id): tuple(str(pointer) for pointer in pointers)
                for call_id, pointers in reported.items()
            }
    return {
        outcome.call_id: visible_citations(messages, outcome.citations)
        for outcome in outcomes
        if not outcome.is_error
    }


def backend_capabilities(backend: Any) -> BackendCapabilities:
    declared = getattr(backend, "capabilities", None)
    if isinstance(declared, BackendCapabilities):
        return declared
    # Compatibility for third-party backends while keeping all interpretation
    # of legacy flags in this one adapter.
    level: StructuredOutputLevel = (
        "schema" if bool(getattr(backend, "hard_phase_guards", False)) else "none"
    )
    return BackendCapabilities(
        structured_output_level=level,
        context_compaction=callable(getattr(backend, "compact_phase_context", None)),
        citation_aliases=callable(getattr(backend, "citation_aliases", None)),
        phase_memory=callable(getattr(backend, "phase_memory_turn", None)),
        orchestrated_prefetch=bool(
            getattr(backend, "supports_orchestrated_prefetch", False)
        ),
        enforce_phase_coverage=bool(
            getattr(backend, "enforce_phase_coverage", False)
        ),
        max_parallel_tool_calls=max(
            1, int(getattr(backend, "max_tool_calls_per_turn", 1) or 1)
        ),
    )


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    refusal: str | None = None
    # Provider-native content, echoed back verbatim on the next turn. On
    # Anthropic models thinking blocks must round-trip unmodified, so the loop
    # never reconstructs assistant turns from `text`.
    raw_content: Any = None

    @property
    def wants_tools(self) -> bool:
        # Providers occasionally return valid tool calls together with a
        # nonstandard finish reason such as ``length``/``max_tokens``. Once an
        # assistant message contains tool calls, every call id must receive a
        # tool result before any later user message; ignoring them makes the
        # next API request invalid.
        return bool(self.tool_calls)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


@runtime_checkable
class LLMBackend(Protocol):
    """What the agent loop needs from a model."""

    name: str
    capabilities: BackendCapabilities

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        require_tool_call: bool = False,
    ) -> LLMResponse:
        """One model turn.

        ``response_schema`` constrains the terminal reply.  When
        ``require_tool_call`` is true the orchestrator has not yet satisfied the
        phase's deterministic evidence coverage, so a backend must not offer a
        terminal-response branch on that turn.
        """
        ...

    def assistant_turn(self, response: LLMResponse) -> dict[str, Any]:
        """The message to append for `response`."""
        ...

    def tool_result_turn(self, outcomes: Sequence[ToolOutcome]) -> list[dict[str, Any]]:
        """Messages carrying the tool results for the preceding assistant turn.

        Returns a list because providers disagree on the shape: Anthropic wants
        every result in one user message (splitting them trains the model out
        of parallel tool calls), while OpenAI-compatible APIs require exactly
        one `role: "tool"` message per call id.
        """
        ...

    def user_turn(self, text: str) -> dict[str, Any]:
        """A plain user message, used for phase transitions and verifier feedback."""
        ...
