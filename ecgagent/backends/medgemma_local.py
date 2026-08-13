"""Local MedGemma backend backed by the in-process vLLM engine.

The bundled MedGemma chat template supports ordinary user/model turns but not
provider-native function-calling messages.  This backend therefore uses a
small, schema-constrained envelope for tool-enabled turns:

```
{"action":"tool_calls","tool_calls":[{"name":"...","arguments":{...}}],"text":""}
{"action":"finish","tool_calls":[],"text":"phase conclusion"}
```

The ECGAgent loop still owns dispatch, budgets, evidence provenance,
verification and revisions.  MedGemma only chooses tools and reasons over
their rendered results.  No clinical rules or labels are introduced here.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..evidence.ledger import visible_citations
from .base import BackendCapabilities, LLMResponse, ToolCall, ToolOutcome


DEFAULT_MODEL = "/workspace/ecg_gemma/medgemma-27b"
DEFAULT_MODEL_MAX_LEN = 131072
DEFAULT_GENERATION_MAX_TOKENS = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_BATCH_WAIT_MS = 50.0
MAX_RETAINED_TOOL_EVIDENCE_CHARS = 28000
MAX_RETAINED_TOOL_RESULT_CHARS = 6000
MAX_TRANSIENT_TOOL_RESULT_CHARS = 9000
MAX_INFLIGHT_TOOL_EVIDENCE_CHARS = 36000
CONTEXT_SAFETY_TOKENS = 2048

_TOOL_RESULTS_PREFIX = "ECGFEAT TOOL RESULTS."
_INFLIGHT_LEDGER_PREFIX = "IN-PHASE EVIDENCE LEDGER."
_CUMULATIVE_MEMORY_PREFIX = "CUMULATIVE EVIDENCE MEMORY."

_DOMAIN_KEYS: tuple[str, ...] = (
    "quality",
    "rhythm_rate",
    "p_av",
    "intervals",
    "axis",
    "conduction_preexcitation",
    "ectopy_pauses",
    "voltage_chamber_r_progression",
    "q_st_t_u",
    "pacing_high_risk",
)

_DOMAIN_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["assessed", "limited", "not_assessed"],
        },
        "summary": {"type": "string", "maxLength": 500},
        "citations": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "counterevidence": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 260},
        },
    },
    "required": ["status", "summary", "citations", "counterevidence"],
    "additionalProperties": False,
}

_PHASE_MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string", "maxLength": 2400},
        "domains": {
            "type": "object",
            "properties": {
                domain: _DOMAIN_ENTRY_SCHEMA for domain in _DOMAIN_KEYS
            },
            "required": list(_DOMAIN_KEYS),
            "additionalProperties": False,
        },
        "findings": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "maxLength": 600},
                    "citations": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "reliability": {
                        "type": "string",
                        "enum": ["reliable", "caution", "candidate_only"],
                    },
                },
                "required": ["claim", "citations", "reliability"],
                "additionalProperties": False,
            },
        },
        "counterevidence": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "maxLength": 600},
                    "citations": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                },
                "required": ["claim", "citations"],
                "additionalProperties": False,
            },
        },
        "unresolved": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 500},
        },
    },
    "required": [
        "conclusion",
        "domains",
        "findings",
        "counterevidence",
        "unresolved",
    ],
    "additionalProperties": False,
}

_TOOL_PROTOCOL = """# Local tool-call response protocol

This local model does not use provider-native function calls. For this turn,
return exactly one JSON object with all three keys:

- To request measurements:
  {"action":"tool_calls","tool_calls":[{"name":"TOOL_NAME","arguments":{...}}],"text":""}
- When this phase is complete:
  {"action":"finish","tool_calls":[],"text":{"conclusion":"concise conclusion","domains":{"quality":{"status":"assessed","summary":"quality conclusion","citations":["E1"],"counterevidence":[]},"rhythm_rate":{"status":"assessed","summary":"rhythm/rate conclusion","citations":["E2"],"counterevidence":[]},"p_av":{"status":"limited","summary":"P/AV conclusion or limitation","citations":[],"counterevidence":[]},"intervals":{"status":"assessed","summary":"interval conclusion","citations":[],"counterevidence":[]},"axis":{"status":"assessed","summary":"axis conclusion","citations":[],"counterevidence":[]},"conduction_preexcitation":{"status":"assessed","summary":"conduction conclusion","citations":[],"counterevidence":[]},"ectopy_pauses":{"status":"assessed","summary":"ectopy/pause conclusion","citations":[],"counterevidence":[]},"voltage_chamber_r_progression":{"status":"assessed","summary":"voltage/chamber/R progression conclusion","citations":[],"counterevidence":[]},"q_st_t_u":{"status":"assessed","summary":"Q/ST/T/U conclusion","citations":[],"counterevidence":[]},"pacing_high_risk":{"status":"assessed","summary":"pacing/high-risk conclusion","citations":[],"counterevidence":[]}},"findings":[{"claim":"supported observation","citations":["E1"],"reliability":"reliable"}],"counterevidence":[],"unresolved":[]}}

Rules:
- Use only tool names and arguments from AVAILABLE TOOLS below.
- Request at most four tools in one turn. Multiple fields in one table call
  are preferred over many tiny calls.
- Do not invent a tool result. Request it and wait for the next user message.
- A detector candidate is not a diagnosis; preserve candidate/ambiguity notes
  returned by the tools.
- Local tool results use short citation aliases such as E1. Copy those
  aliases exactly into phase memories and final `citations` arrays. Never
  expand, shorten or guess a JSON Pointer; the backend expands aliases before
  deterministic verification.
- A finish response must use the structured phase-memory object shown above.
  Complete all ten domain entries. Preserve decisive support, counterevidence,
  reliability caveats and unresolved questions; do not silently drop a domain
  just to shorten the response.

AVAILABLE TOOLS:
{tools}
"""

_STRUCTURED_OUTPUT_NOTE = """# Required final response

Return only one JSON object matching the supplied JSON Schema. Do not wrap it
in a markdown fence and do not add prose outside the object.
All human-facing narrative strings must be written in clear English. Keep
registered diagnosis codes and citation aliases unchanged.

JSON Schema:
{schema}
"""

_STRUCTURED_TOOL_PROTOCOL = """# Local tool-call and final-response protocol

This is a structured diagnosis/revision turn. Return exactly one JSON object:

- To request measurements, return:
  {"action":"tool_calls","tool_calls":[{"name":"TOOL_NAME","arguments":{...}}],"text":""}
- When the diagnosis is complete, return the final diagnosis object directly,
  matching FINAL JSON SCHEMA. Do not put it inside `action`, `text`,
  `conclusion`, or a markdown fence.

Use only AVAILABLE TOOLS. Request at most four tools in one turn. Tool results
use citation aliases such as E1; copy them exactly into `citations` arrays.
All human-facing narrative strings in the final diagnosis must be written in
clear English. Keep registered diagnosis codes and citation aliases unchanged.

AVAILABLE TOOLS:
{tools}

FINAL JSON SCHEMA:
{schema}
"""

_TOOL_ONLY_PROTOCOL = """# Required ecgfeat measurement turn

The orchestrator has not yet completed this phase's minimum patient-evidence
coverage.  This turn cannot finish the phase and cannot return a diagnosis or
phase-state object.  Return exactly one tool-call envelope:

{"action":"tool_calls","tool_calls":[{"name":"TOOL_NAME","arguments":{...}}],"text":""}

Use only AVAILABLE TOOLS and request at most four distinct, clinically useful
views in this turn. Prefer one table call containing several related fields over
many narrow calls. Do not repeat an identical tool and argument set.

AVAILABLE TOOLS:
{tools}
"""


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not 0.0 < value <= 1.0:
        raise RuntimeError(f"{name} must be greater than 0 and at most 1")
    return value


def _nonnegative_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative number") from exc
    if value < 0.0:
        raise RuntimeError(f"{name} must be a non-negative number")
    return value


def _openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
        function = tool["function"]
        return {
            "name": str(function.get("name") or ""),
            "description": str(function.get("description") or ""),
            "parameters": function.get("parameters") or {},
        }
    return {
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "parameters": tool.get("input_schema") or tool.get("parameters") or {},
    }


def _tool_call_schema(tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in (_openai_tool(tool) for tool in tools) if row["name"]]
    branches = [
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": row["name"]},
                "arguments": row["parameters"],
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for row in rows
    ]
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "const": "tool_calls"},
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"oneOf": branches},
            },
            "text": {"type": "string", "const": ""},
        },
        "required": ["action", "tool_calls", "text"],
        "additionalProperties": False,
    }


def _tool_turn_schema(tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tool_calls = _tool_call_schema(tools)
    finish = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "const": "finish"},
            "tool_calls": {
                "type": "array",
                "maxItems": 0,
                "items": {"type": "object"},
            },
            "text": _PHASE_MEMORY_SCHEMA,
        },
        "required": ["action", "tool_calls", "text"],
        "additionalProperties": False,
    }
    return {"oneOf": [tool_calls, finish]}


def _tool_or_structured_schema(
    tools: Sequence[Mapping[str, Any]],
    response_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Allow either a tool request or the direct final structured object.

    Revision phases can still re-read evidence, but their terminal response is
    the diagnosis contract itself rather than the compact phase-memory
    envelope used by evidence-gathering phases.
    """

    tool_branch = _tool_call_schema(tools)
    return {"oneOf": [tool_branch, dict(response_schema)]}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("content") is not None:
                    parts.append(str(item["content"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _normalize_chat(
    system: str,
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Convert provider-shaped history to Gemma's alternating chat contract."""
    normalized: list[dict[str, str]] = [{"role": "system", "content": system}]
    for message in messages:
        role = str(message.get("role") or "user")
        if role in {"developer", "system"}:
            role = "user"
        elif role in {"model", "assistant"}:
            role = "assistant"
        elif role == "tool":
            role = "user"
        else:
            role = "user"
        content = _content_text(message.get("content"))
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = (
                normalized[-1]["content"].rstrip()
                + "\n\n"
                + content.lstrip()
            ).strip()
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline >= 0:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _parse_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_json_fence(text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _canonical_tool_key(tool: str, arguments: Mapping[str, Any]) -> str:
    return str(tool) + ":" + json.dumps(
        dict(arguments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _compact_tool_excerpt(text: str, limit: int) -> tuple[str, int]:
    """Keep representative table rows and both ends of long tool output.

    Full output remains in ``ToolRegistry`` for the Markdown audit trace.  This
    function only controls what is copied into the model conversation.  Most
    large ecgfeat returns are Markdown tables; retaining the first and last
    rows preserves cross-beat variation better than a head-only character cut.
    """

    original = str(text or "")
    hard_limit = max(1000, int(limit))
    if len(original) <= hard_limit:
        return original, 0

    lines = original.splitlines()
    data_rows = [
        index
        for index, line in enumerate(lines)
        if line.startswith("|")
        and not set(line.replace("|", "").strip()) <= {"-", ":"}
    ]
    # The first table row is normally the header, so never count it as a
    # representative patient row.
    patient_rows = data_rows[1:] if data_rows else []
    if len(patient_rows) >= 8:
        keep_count = min(len(patient_rows), 24)
        while keep_count >= 4:
            first_count = (keep_count + 1) // 2
            last_count = keep_count // 2
            selected = set(patient_rows[:first_count]) | set(
                patient_rows[-last_count:] if last_count else []
            )
            rebuilt: list[str] = []
            omission_written = False
            for index, line in enumerate(lines):
                if index in patient_rows and index not in selected:
                    if not omission_written:
                        rebuilt.append(
                            "[model-context excerpt: intermediate table rows "
                            "omitted; full rows remain in the audit trace]"
                        )
                        omission_written = True
                    continue
                rebuilt.append(line)
            candidate = "\n".join(rebuilt)
            if len(candidate) <= hard_limit:
                return candidate, len(original) - len(candidate)
            keep_count -= 2

    # Non-tabular or exceptionally wide output: preserve the beginning,
    # closing caveats and final rows rather than cutting off only the tail.
    marker = (
        "\n[model-context excerpt truncated; full result remains in the "
        "agent trace]\n"
    )
    head = max(400, int((hard_limit - len(marker)) * 0.65))
    tail = max(300, hard_limit - len(marker) - head)
    candidate = original[:head].rstrip() + marker + original[-tail:].lstrip()
    return candidate[:hard_limit], max(0, len(original) - len(candidate[:hard_limit]))


def _parse_prefixed_json(content: str, prefix: str) -> Any:
    if not str(content).startswith(prefix) or "\n" not in str(content):
        return None
    try:
        return json.loads(str(content).split("\n", 1)[1])
    except (json.JSONDecodeError, TypeError):
        return None


@dataclass
class _PendingChat:
    messages: list[dict[str, str]]
    sampling_params: Any
    done: threading.Event = field(default_factory=threading.Event)
    output: Any = None
    error: BaseException | None = None


class _VLLMChatBatcher:
    """Micro-batch synchronous Agent turns onto one offline vLLM engine.

    Each ECG Agent still runs as an ordinary blocking worker.  Their model
    turns are queued here for a short window and submitted together through
    ``LLM.chat``.  This preserves the simple Agent state machine while letting
    vLLM schedule several independent records in one GPU batch.
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_batch_size: int,
        wait_ms: float,
    ) -> None:
        self.llm = llm
        self.max_batch_size = max(1, int(max_batch_size))
        self.wait_s = max(0.0, float(wait_ms)) / 1000.0
        self._queue: queue.Queue[_PendingChat | None] = queue.Queue()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._batches = 0
        self._requests = 0
        self._max_observed_batch = 0
        self._thread = threading.Thread(
            target=self._run,
            name="medgemma-vllm-batcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, messages: list[dict[str, str]], sampling_params: Any) -> Any:
        if self._closed:
            raise RuntimeError("local MedGemma batch scheduler is closed")
        pending = _PendingChat(messages=messages, sampling_params=sampling_params)
        self._queue.put(pending)
        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        return pending.output

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            stop_after_batch = False
            deadline = time.monotonic() + self.wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._closed = True
                    stop_after_batch = True
                    break
                batch.append(item)
            try:
                with self._stats_lock:
                    self._batches += 1
                    self._requests += len(batch)
                    self._max_observed_batch = max(
                        self._max_observed_batch,
                        len(batch),
                    )
                outputs = self.llm.chat(
                    [item.messages for item in batch],
                    sampling_params=[item.sampling_params for item in batch],
                    use_tqdm=False,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        "vLLM returned a different number of outputs than "
                        f"batched inputs ({len(outputs)} != {len(batch)})"
                    )
                for item, output in zip(batch, outputs):
                    item.output = output
            except BaseException as exc:
                for item in batch:
                    item.error = exc
            finally:
                for item in batch:
                    item.done.set()
            if stop_after_batch:
                return

    def report(self) -> dict[str, Any]:
        with self._stats_lock:
            batches = self._batches
            requests = self._requests
            max_observed = self._max_observed_batch
        return {
            "configured_max_batch_size": self.max_batch_size,
            "batch_wait_ms": round(self.wait_s * 1000.0, 3),
            "batches": batches,
            "requests": requests,
            "mean_batch_size": (
                round(requests / batches, 3) if batches else 0.0
            ),
            "max_observed_batch_size": max_observed,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10.0)


@dataclass
class MedGemmaLocalBackend:
    """In-process vLLM backend for the local MedGemma-27B checkpoint."""

    model: str = DEFAULT_MODEL
    model_max_len: int = DEFAULT_MODEL_MAX_LEN
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 64
    seed: int = 0
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    enforce_eager: bool = False
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS
    enforce_phase_coverage: bool = True
    # This backend carries tool observations as ordinary bounded text, so an
    # invariant Survey packet can be dispatched before the first generation
    # without violating a provider-native tool-call-id protocol.
    supports_orchestrated_prefetch: bool = True
    max_tool_calls_per_turn: int = 4
    # Phase-state JSON is generated under a real vLLM grammar, so unlike a
    # provider/mock that may only offer best-effort JSON, an invalid state can
    # safely be treated as a hard transition failure.
    hard_phase_guards: bool = True
    capabilities: BackendCapabilities = field(
        default=BackendCapabilities(
            native_tool_calls=False,
            structured_output_level="grammar",
            context_compaction=True,
            citation_aliases=True,
            phase_memory=True,
            orchestrated_prefetch=True,
            enforce_phase_coverage=True,
            max_parallel_tool_calls=4,
        ),
        init=False,
    )
    llm: Any = None
    sampling_params_factory: Callable[..., Any] | None = None
    structured_outputs_factory: Callable[..., Any] | None = None
    scheduler: Any = field(default=None, repr=False)
    name: str = "medgemma-local"
    turns: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _runtime_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _call_sequence: int = field(default=0, repr=False)
    _citation_sequence: int = field(default=0, repr=False)
    _citation_aliases: dict[str, str] = field(default_factory=dict, repr=False)
    _alias_pointers: dict[str, str] = field(default_factory=dict, repr=False)
    _compression: dict[str, int] = field(default_factory=dict, repr=False)
    _phase_memories: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )
    _phase_memory_history: list[str] = field(default_factory=list, repr=False)
    _retained_tool_evidence: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )
    _retained_sequence: int = field(default=0, repr=False)
    _inflight_tool_evidence: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict,
        repr=False,
    )
    _inflight_warnings: dict[str, list[str]] = field(
        default_factory=dict,
        repr=False,
    )
    _inflight_sequence: int = field(default=0, repr=False)
    _preflight_checks: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    _tool_context_trace: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    _owns_scheduler: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.capabilities = BackendCapabilities(
            native_tool_calls=False,
            structured_output_level=("grammar" if self.hard_phase_guards else "json"),
            context_compaction=True,
            citation_aliases=True,
            phase_memory=True,
            orchestrated_prefetch=self.supports_orchestrated_prefetch,
            enforce_phase_coverage=self.enforce_phase_coverage,
            max_parallel_tool_calls=max(1, int(self.max_tool_calls_per_turn)),
        )
        self.model_max_len = _positive_int_env(
            "ECG_GEMMA_AGENT_MAX_LEN",
            int(self.model_max_len),
        )
        self.gpu_memory_utilization = _float_env(
            "ECG_GEMMA_GPU_MEMORY_UTILIZATION",
            float(self.gpu_memory_utilization),
        )
        self.max_batch_size = _positive_int_env(
            "ECG_GEMMA_MAX_CONCURRENCY",
            int(self.max_batch_size),
        )
        self.batch_wait_ms = _nonnegative_float_env(
            "ECG_GEMMA_BATCH_WAIT_MS",
            float(self.batch_wait_ms),
        )
        if self.llm is not None:
            self._ensure_scheduler()
            return

        model_path = Path(self.model).expanduser().resolve()
        if not model_path.exists():
            raise RuntimeError(f"local MedGemma model path does not exist: {model_path}")
        if not (model_path / "config.json").exists():
            raise RuntimeError(
                f"local MedGemma model is missing config.json: {model_path}"
            )
        try:
            model_config = json.loads(
                (model_path / "config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"could not read local model config.json: {model_path} ({exc})"
            ) from exc
        model_type = str(model_config.get("model_type") or "").lower()
        if model_type.startswith("qwen"):
            raise RuntimeError(
                "Qwen checkpoints must not use the MedGemma JSON-envelope "
                "backend. Start vLLM with --enable-auto-tool-choice "
                "--tool-call-parser qwen3_xml --reasoning-parser qwen3, then "
                "run ecgagent with --backend qwen-local and --model set to the "
                "served model name."
            )
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"vllm is required for the local MedGemma backend ({exc})"
            ) from exc

        from medgemma_runtime import infer_runtime_config

        runtime = infer_runtime_config()
        self._runtime_config = dict(runtime)
        llm_kwargs: dict[str, Any] = {
            "model": str(model_path),
            "dtype": "bfloat16",
            "max_model_len": self.model_max_len,
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": True,
            "enforce_eager": self.enforce_eager,
        }
        if runtime.get("quantization"):
            llm_kwargs["quantization"] = runtime["quantization"]
        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as exc:  # pragma: no cover - requires a large GPU model
            raise RuntimeError(
                "could not initialize local MedGemma with vLLM. Check "
                "CUDA_VISIBLE_DEVICES, ECG_GEMMA_TP_SIZE, free GPU memory and "
                f"ECG_GEMMA_AGENT_MAX_LEN. Original error: {exc}"
            ) from exc
        self.sampling_params_factory = SamplingParams
        self.structured_outputs_factory = StructuredOutputsParams
        self.model = str(model_path)
        self._ensure_scheduler()

    def _ensure_scheduler(self) -> None:
        if self.scheduler is not None or self.max_batch_size <= 1:
            return
        self.scheduler = _VLLMChatBatcher(
            self.llm,
            max_batch_size=self.max_batch_size,
            wait_ms=self.batch_wait_ms,
        )
        self._owns_scheduler = True

    def new_session(self) -> "MedGemmaLocalBackend":
        """Return isolated per-record state backed by the shared vLLM engine."""
        session = MedGemmaLocalBackend(
            model=self.model,
            model_max_len=self.model_max_len,
            generation_max_tokens=self.generation_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            max_batch_size=self.max_batch_size,
            batch_wait_ms=self.batch_wait_ms,
            enforce_phase_coverage=self.enforce_phase_coverage,
            supports_orchestrated_prefetch=self.supports_orchestrated_prefetch,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            hard_phase_guards=self.hard_phase_guards,
            llm=self.llm,
            sampling_params_factory=self.sampling_params_factory,
            structured_outputs_factory=self.structured_outputs_factory,
            scheduler=self.scheduler,
        )
        session._runtime_config = dict(self._runtime_config)
        return session

    def close(self) -> None:
        if self._owns_scheduler and self.scheduler is not None:
            self.scheduler.close()
            self._owns_scheduler = False

    def reset_session(self) -> None:
        """Reset per-record audit state without unloading the shared model."""
        self.turns.clear()
        self._call_sequence = 0
        self._citation_sequence = 0
        self._citation_aliases.clear()
        self._alias_pointers.clear()
        self._compression.clear()
        self._phase_memories.clear()
        self._phase_memory_history.clear()
        self._retained_tool_evidence.clear()
        self._retained_sequence = 0
        self._inflight_tool_evidence.clear()
        self._inflight_warnings.clear()
        self._inflight_sequence = 0
        self._preflight_checks.clear()
        self._tool_context_trace.clear()

    def _structured_outputs(self, schema: dict[str, Any]) -> Any:
        factory = self.structured_outputs_factory
        if factory is None:
            from vllm.sampling_params import StructuredOutputsParams

            factory = StructuredOutputsParams
        return factory(json=schema)

    def _sampling_params(
        self,
        *,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> Any:
        factory = self.sampling_params_factory
        if factory is None:
            from vllm import SamplingParams

            factory = SamplingParams
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": min(
                max(1, int(max_tokens)),
                int(self.generation_max_tokens),
            ),
        }
        if schema is not None:
            kwargs["structured_outputs"] = self._structured_outputs(schema)
        return factory(**kwargs)

    def _estimate_chat_tokens(self, chat: Sequence[Mapping[str, Any]]) -> int:
        """Estimate the fully templated prompt, preferring the real tokenizer."""

        getter = getattr(self.llm, "get_tokenizer", None)
        tokenizer = None
        if callable(getter):
            try:
                tokenizer = getter()
            except Exception:
                tokenizer = None
        if tokenizer is not None:
            try:
                templated = tokenizer.apply_chat_template(
                    list(chat),
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(templated, Sequence) and not isinstance(
                    templated, (str, bytes)
                ):
                    return len(templated)
            except Exception:
                try:
                    rendered = tokenizer.apply_chat_template(
                        list(chat),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    return len(tokenizer.encode(rendered, add_special_tokens=False))
                except Exception:
                    pass
        # UTF-8 bytes/3 is conservative for Chinese (roughly one token per
        # character) while still tracking the denser JSON/English sections.
        serialized = json.dumps(
            list(chat),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return max(1, (len(serialized.encode("utf-8")) + 2) // 3 + 256)

    @staticmethod
    def _shrink_evidence_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        per_result_chars: int,
        total_chars: int,
    ) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        used = 0
        for source in rows:
            row = dict(source)
            result, _ = _compact_tool_excerpt(
                str(row.get("result") or ""),
                per_result_chars,
            )
            row["result"] = result
            size = len(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            if rendered and used + size > total_chars:
                continue
            rendered.append(row)
            used += size
        return rendered

    def _shrink_context_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        per_result_chars: int,
        total_chars: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Shrink only auditable evidence excerpts, never clinical state text."""

        output: list[dict[str, Any]] = []
        saved = 0
        for message in messages:
            cloned = dict(message)
            content = str(cloned.get("content") or "")
            prefix: str | None = None
            rows_key: str | None = None
            if content.startswith(_INFLIGHT_LEDGER_PREFIX):
                prefix = _INFLIGHT_LEDGER_PREFIX
                rows_key = "evidence_views"
            elif content.startswith(_CUMULATIVE_MEMORY_PREFIX):
                prefix = _CUMULATIVE_MEMORY_PREFIX
                rows_key = "retained_key_tool_evidence"
            if prefix is not None and rows_key is not None:
                payload = _parse_prefixed_json(content, prefix)
                rows = payload.get(rows_key) if isinstance(payload, Mapping) else None
                if isinstance(rows, list):
                    before = len(content)
                    payload = dict(payload)
                    payload[rows_key] = self._shrink_evidence_rows(
                        [row for row in rows if isinstance(row, Mapping)],
                        per_result_chars=per_result_chars,
                        total_chars=total_chars,
                    )
                    introduction = content.split("\n", 1)[0]
                    cloned["content"] = introduction + "\n" + json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    saved += max(0, before - len(str(cloned["content"])))
            output.append(cloned)
        return output, saved

    def _prepare_chat_with_budget(
        self,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        requested_output_tokens: int,
    ) -> list[dict[str, str]]:
        """Preflight every local request and compact evidence before overflow."""

        input_budget = max(
            1,
            int(self.model_max_len)
            - max(1, int(requested_output_tokens))
            - CONTEXT_SAFETY_TOKENS,
        )
        working = [dict(message) for message in messages]
        chat = _normalize_chat(system, working)
        initial = self._estimate_chat_tokens(chat)
        soft_limit = min(input_budget, max(1, int(self.model_max_len * 0.80)))
        saved_chars = 0
        level = "none"

        if initial > soft_limit:
            working, saved = self._shrink_context_messages(
                working,
                per_result_chars=4000,
                total_chars=28000,
            )
            saved_chars += saved
            chat = _normalize_chat(system, working)
            level = "soft"

        estimated = self._estimate_chat_tokens(chat)
        if estimated > input_budget:
            working, saved = self._shrink_context_messages(
                working,
                per_result_chars=2000,
                total_chars=14000,
            )
            saved_chars += saved
            chat = _normalize_chat(system, working)
            estimated = self._estimate_chat_tokens(chat)
            level = "hard"

        check = {
            "model_max_len": int(self.model_max_len),
            "requested_output_tokens": int(requested_output_tokens),
            "safety_tokens": CONTEXT_SAFETY_TOKENS,
            "input_budget_tokens": input_budget,
            "initial_estimated_prompt_tokens": initial,
            "final_estimated_prompt_tokens": estimated,
            "compaction_level": level,
            "saved_chars": saved_chars,
            "status": "pass" if estimated <= input_budget else "overflow",
        }
        self._preflight_checks.append(check)
        if estimated > input_budget:
            raise RuntimeError(
                "preflight context budget exceeded after evidence compaction: "
                f"estimated_prompt_tokens={estimated}, input_budget={input_budget}, "
                f"requested_output_tokens={requested_output_tokens}, "
                f"model_max_len={self.model_max_len}. Narrow the requested "
                "measurement view or reduce retained evidence; vLLM was not called."
            )
        return chat

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
        tool_schema: dict[str, Any] | None = None
        system_sections = [system]
        if tools:
            tool_rows = [_openai_tool(tool) for tool in tools]
            rendered_tools = json.dumps(tool_rows, ensure_ascii=False, indent=2)
            if require_tool_call:
                system_sections.append(
                    _TOOL_ONLY_PROTOCOL.replace("{tools}", rendered_tools)
                )
                tool_schema = _tool_call_schema(tools)
            elif response_schema is not None:
                system_sections.append(
                    _STRUCTURED_TOOL_PROTOCOL.replace(
                        "{tools}",
                        rendered_tools,
                    ).replace(
                        "{schema}",
                        json.dumps(
                            response_schema,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                )
                tool_schema = _tool_or_structured_schema(tools, response_schema)
            else:
                system_sections.append(
                    _TOOL_PROTOCOL.replace(
                        "{tools}",
                        rendered_tools,
                    )
                )
                tool_schema = _tool_turn_schema(tools)
        elif response_schema is not None:
            system_sections.append(
                _STRUCTURED_OUTPUT_NOTE.format(
                    schema=json.dumps(
                        response_schema,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            )

        guided_schema = tool_schema if tools else response_schema
        requested_output_tokens = min(
            max(1, int(max_tokens or self.generation_max_tokens)),
            int(self.generation_max_tokens),
        )
        chat = self._prepare_chat_with_budget(
            "\n\n".join(system_sections),
            messages,
            requested_output_tokens=requested_output_tokens,
        )
        sampling_params = self._sampling_params(
            max_tokens=requested_output_tokens,
            schema=guided_schema,
        )
        try:
            if self.scheduler is not None:
                request_output = self.scheduler.submit(chat, sampling_params)
            else:
                outputs = self.llm.chat(
                    chat,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                if not outputs:
                    raise RuntimeError("local MedGemma returned no completion")
                request_output = outputs[0]
        except Exception as exc:
            detail = str(exc)
            lowered = detail.lower()
            if "grammar error" in lowered or "unimplemented keys" in lowered:
                prefix = (
                    "local structured-output grammar was rejected by vLLM; "
                    "this is a schema/backend compatibility error, not a context "
                    "overflow"
                )
            elif "out of memory" in lowered or "cuda oom" in lowered:
                prefix = "local vLLM generation ran out of GPU memory"
            elif "maximum context" in lowered or "context length" in lowered:
                prefix = "local vLLM generation exceeded the model context length"
            else:
                prefix = "local vLLM generation failed"
            raise RuntimeError(f"{prefix}. Original error: {detail}") from exc
        if not getattr(request_output, "outputs", None):
            raise RuntimeError("local MedGemma returned no completion")

        completion = request_output.outputs[0]
        generated = str(getattr(completion, "text", "") or "").strip()
        prompt_tokens = len(getattr(request_output, "prompt_token_ids", None) or [])
        completion_tokens = len(getattr(completion, "token_ids", None) or [])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        tool_calls: list[ToolCall] = []
        response_text = generated
        assistant_content = generated
        stop_reason = (
            "max_tokens"
            if str(getattr(completion, "finish_reason", "") or "").lower()
            in {"length", "max_tokens"}
            else "end_turn"
        )
        if tools:
            envelope = _parse_object(generated) or {}
            action = str(envelope.get("action") or "")
            rows = envelope.get("tool_calls")
            rows = rows if isinstance(rows, list) else []
            if action == "tool_calls":
                normalized_rows: list[dict[str, Any]] = []
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    name = str(row.get("name") or "")
                    arguments = row.get("arguments")
                    if not name or not isinstance(arguments, Mapping):
                        continue
                    self._call_sequence += 1
                    call_id = f"medgemma_tool_{self._call_sequence}"
                    argument_dict = dict(arguments)
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            name=name,
                            arguments=argument_dict,
                        )
                    )
                    normalized_rows.append(
                        {"id": call_id, "name": name, "arguments": argument_dict}
                    )
                assistant_content = json.dumps(
                    {
                        "action": "tool_calls",
                        "tool_calls": normalized_rows,
                        "text": "",
                    },
                    ensure_ascii=False,
                )
                response_text = ""
                stop_reason = "tool_use" if tool_calls else "end_turn"
            elif response_schema is not None:
                # Structured synthesis/revision responses are the diagnosis
                # object itself.  Do not reinterpret a valid verdict as a
                # phase-memory conclusion merely because revision tools were
                # available in this turn.
                response_text = generated
                assistant_content = generated
            else:
                phase_memory = envelope.get("text")
                if isinstance(phase_memory, Mapping):
                    response_text = json.dumps(
                        dict(phase_memory),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                else:
                    conclusion = str(phase_memory or generated).strip()
                    response_text = json.dumps(
                        {
                            "conclusion": conclusion,
                            "findings": [],
                            "counterevidence": [],
                            "unresolved": [],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                assistant_content = response_text

        self.turns.append(
            {
                "model": self.model,
                "stop_reason": stop_reason,
                "usage": usage,
                "tool_calls": len(tool_calls),
            }
        )
        return LLMResponse(
            text=response_text,
            tool_calls=tuple(tool_calls),
            stop_reason=stop_reason,
            model=self.model,
            usage=usage,
            raw_content=assistant_content,
        )

    def assistant_turn(self, response: LLMResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": str(response.raw_content or response.text or ""),
        }

    def _citation_alias(self, pointer: str) -> str:
        pointer = str(pointer)
        existing = self._citation_aliases.get(pointer)
        if existing is not None:
            return existing
        self._citation_sequence += 1
        alias = f"E{self._citation_sequence}"
        self._citation_aliases[pointer] = alias
        self._alias_pointers[alias] = pointer
        return alias

    def _compress_tool_text(self, outcome: ToolOutcome) -> str:
        original = str(outcome.model_text or outcome.text or "")
        # Alias only citations whose value-bearing token is present before
        # compression. Raw ToolResult citations may include companions or rows
        # already truncated by the tool renderer.
        rendered_candidates = visible_citations(
            original,
            outcome.model_citations
            if outcome.model_text is not None
            else outcome.citations,
        )
        aliases = {
            pointer: self._citation_alias(pointer)
            for pointer in rendered_candidates
        }
        compressed = original
        # A pointer may be a prefix of another pointer. Replace the longest
        # first so every visible citation remains an exact, atomic alias.
        for pointer in sorted(aliases, key=len, reverse=True):
            compressed = compressed.replace(
                f"ev:{pointer}",
                aliases[pointer],
            )
        transient_limit = (
            MAX_TRANSIENT_TOOL_RESULT_CHARS
            if outcome.name
            in {
                "get_native_beat_profile",
                "get_atrial_event_table",
                "get_p_assessment_table",
                "get_beat_table",
            }
            else MAX_RETAINED_TOOL_RESULT_CHARS
        )
        compressed, omitted = _compact_tool_excerpt(
            compressed,
            transient_limit,
        )
        visible_after_compression = visible_citations(
            compressed,
            rendered_candidates,
            aliases={alias: pointer for pointer, alias in aliases.items()},
            allow_exact=False,
        )
        self._compression["outcomes"] = self._compression.get("outcomes", 0) + 1
        self._compression["original_chars"] = (
            self._compression.get("original_chars", 0) + len(original)
        )
        self._compression["context_chars"] = (
            self._compression.get("context_chars", 0) + len(compressed)
        )
        if omitted:
            self._compression["truncated_outcomes"] = (
                self._compression.get("truncated_outcomes", 0) + 1
            )
            self._compression["omitted_chars"] = (
                self._compression.get("omitted_chars", 0) + omitted
            )
        self._tool_context_trace.append(
            {
                "call_id": outcome.call_id,
                "phase": "",
                "tool": outcome.name,
                "arguments": dict(outcome.arguments),
                "model_context_text": compressed,
                "original_chars": len(str(outcome.text or "")),
                "model_context_chars": len(compressed),
                "truncated_for_model": bool(omitted),
                "candidate_citations": list(outcome.citations),
                "visible_citations": list(visible_after_compression),
            }
        )
        return compressed

    def tool_result_turn(
        self,
        outcomes: Sequence[ToolOutcome],
    ) -> list[dict[str, Any]]:
        rows = [
            {
                "tool_call_id": outcome.call_id,
                "tool": outcome.name,
                "arguments": dict(outcome.arguments),
                "is_error": bool(outcome.is_error),
                "result": self._compress_tool_text(outcome),
            }
            for outcome in outcomes
        ]
        return [
            {
                "role": "user",
                "content": (
                    "ECGFEAT TOOL RESULTS. Use these observations in the current "
                    "phase and preserve every reliability/candidate caveat:\n"
                    + json.dumps(
                        rows,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            }
        ]

    def _collect_inflight_tool_evidence(
        self,
        phase: str,
        transient_messages: Sequence[Mapping[str, Any]],
    ) -> None:
        phase_key = str(phase)
        evidence = self._inflight_tool_evidence.setdefault(phase_key, {})
        warnings = self._inflight_warnings.setdefault(phase_key, [])
        for message in transient_messages:
            content = str(message.get("content") or "")
            rows = _parse_prefixed_json(content, _TOOL_RESULTS_PREFIX)
            if not isinstance(rows, list):
                continue
            for source in rows:
                if not isinstance(source, Mapping):
                    continue
                tool = str(source.get("tool") or "")
                call_id = str(source.get("tool_call_id") or "")
                if call_id:
                    for trace_row in reversed(self._tool_context_trace):
                        if trace_row.get("call_id") == call_id:
                            trace_row["phase"] = phase_key
                            break
                arguments = source.get("arguments")
                arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
                if bool(source.get("is_error")):
                    warning = str(source.get("result") or "").strip()
                    if warning and warning not in warnings:
                        warnings.append(warning[:1000])
                    continue
                self._inflight_sequence += 1
                key = _canonical_tool_key(tool, arguments)
                evidence[key] = {
                    "tool": tool,
                    "arguments": arguments,
                    "result": str(source.get("result") or ""),
                    "sequence": self._inflight_sequence,
                }

    @staticmethod
    def _tool_evidence_priority(tool: str) -> int:
        return {
            "get_global_table": 0,
            "get_rhythm_profile": 1,
            "get_p_assessment_table": 1,
            "get_atrial_event_table": 1,
            "get_morphology_groups": 2,
            "get_morphology_map": 2,
            "get_interval_waveform_context": 2,
            "get_native_beat_profile": 3,
            "get_pacing_profile": 3,
            "get_beat_table": 4,
            "get_lead_table": 4,
            "get_measurement": 4,
        }.get(str(tool), 9)

    def _render_inflight_tool_evidence(self, phase: str) -> list[dict[str, Any]]:
        rows = sorted(
            self._inflight_tool_evidence.get(str(phase), {}).values(),
            key=lambda row: (
                self._tool_evidence_priority(str(row.get("tool") or "")),
                int(row.get("sequence") or 0),
            ),
        )
        rendered: list[dict[str, Any]] = []
        used = 0
        for row in rows:
            public = {
                "tool": row["tool"],
                "arguments": row["arguments"],
                "result": row["result"],
            }
            size = len(
                json.dumps(
                    public,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            if rendered and used + size > MAX_INFLIGHT_TOOL_EVIDENCE_CHARS:
                continue
            rendered.append(public)
            used += size
        return rendered

    def compact_phase_context(
        self,
        phase: str,
        transient_messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replace local tool exchanges with one bounded in-phase ledger.

        Provider-native tool histories cannot always be rewritten safely, so
        the generic loop calls this optional method only for backends that
        implement it.  vLLM/Gemma uses plain alternating text and can compact
        immediately after every tool batch without breaking call-id semantics.
        """

        self._collect_inflight_tool_evidence(phase, transient_messages)
        kept: list[dict[str, Any]] = []
        for message in transient_messages:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
            if role == "assistant":
                continue
            if content.startswith((_TOOL_RESULTS_PREFIX, _INFLIGHT_LEDGER_PREFIX)):
                continue
            kept.append(dict(message))
        payload = {
            "phase": str(phase),
            "distinct_evidence_views": len(
                self._inflight_tool_evidence.get(str(phase), {})
            ),
            "evidence_views": self._render_inflight_tool_evidence(phase),
            "suppressed_or_rejected_calls": list(
                self._inflight_warnings.get(str(phase), [])[-6:]
            ),
        }
        kept.append(
            {
                "role": "user",
                "content": (
                    _INFLIGHT_LEDGER_PREFIX
                    + " Exact repeat calls are suppressed. Full raw results "
                    "remain in the Markdown trace; reason from these bounded "
                    "patient-evidence excerpts and citation aliases:\n"
                    + json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                ),
            }
        )
        return kept

    def phase_memory_turn(
        self,
        phase: str,
        text: str,
        transient_messages: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        memory = _parse_object(text)
        if memory is None:
            memory = {
                "conclusion": str(text or "").strip(),
                "domains": {
                    domain: {
                        "status": "not_assessed",
                        "summary": "",
                        "citations": [],
                        "counterevidence": [],
                    }
                    for domain in _DOMAIN_KEYS
                },
                "findings": [],
                "counterevidence": [],
                "unresolved": [],
            }
        phase_key = str(phase)
        previous_memory = next(iter(self._phase_memories.values()), None)
        if isinstance(previous_memory, Mapping) and isinstance(memory, Mapping):
            previous_domains = previous_memory.get("domains")
            domain_updates = memory.get("domains")
            if isinstance(previous_domains, Mapping):
                # Later phase contracts omit domains entirely. Preserve the
                # single compact Survey ledger without making the model
                # regenerate it. The optional merge keeps older/custom phase
                # contracts backward compatible.
                merged_domains = {
                    str(key): dict(value) if isinstance(value, Mapping) else value
                    for key, value in previous_domains.items()
                }
                if isinstance(domain_updates, Mapping):
                    for key, value in domain_updates.items():
                        merged_domains[str(key)] = (
                            dict(value) if isinstance(value, Mapping) else value
                        )
                memory = dict(memory)
                memory["domains"] = merged_domains
        self._phase_memory_history.append(phase_key)
        # Keep only the newest structured clinical state. Earlier phases remain
        # in the persisted trace; accumulating them in every subsequent prompt
        # repeats superseded hypotheses and encourages copy-forward errors.
        self._phase_memories.clear()
        self._phase_memories[phase_key] = memory
        self._retain_key_tool_evidence(transient_messages)
        payload = {
            "state_authority": "orchestrator",
            "phase_history": list(self._phase_memory_history),
            "current_phase": phase_key,
            "current_phase_state": memory,
            "retained_key_tool_evidence": self._render_retained_tool_evidence(),
        }
        self._inflight_tool_evidence.pop(phase_key, None)
        self._inflight_warnings.pop(phase_key, None)
        return [
            {
                "role": "user",
                "content": (
                    "CUMULATIVE EVIDENCE MEMORY. This contains the current "
                    "program-owned diagnostic ledger snapshot and replaces superseded phase "
                    "states, while preserving all ten diagnostic domains plus "
                    "selected compressed raw results from decisive tools. Treat "
                    "citation aliases as exact previously observed evidence; "
                    "re-query when the retained excerpt is insufficient:\n"
                    + json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            }
        ]

    @staticmethod
    def is_phase_memory_turn(message: Mapping[str, Any]) -> bool:
        return str(message.get("content") or "").startswith(
            "CUMULATIVE EVIDENCE MEMORY."
        )

    def compact_revision_context(
        self,
        messages: Sequence[Mapping[str, Any]],
        current_candidate: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Keep one clinical state and one candidate during verdict repair.

        Previously every complete synthesis and every complete failed revision
        remained in history.  A third repair therefore prefixed several copies
        of the largest object in the protocol.  The audit trace already stores
        those outputs, so the next model turn only needs the latest phase memory,
        the current canonical candidate and the new deterministic feedback.
        """

        kept: list[dict[str, Any]] = []
        if messages:
            kept.append(dict(messages[0]))
        phase_memories = [
            dict(message)
            for message in messages
            if self.is_phase_memory_turn(message)
        ]
        if phase_memories:
            if not kept or phase_memories[-1] != kept[-1]:
                kept.append(phase_memories[-1])

        def shorten_citations(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(key): shorten_citations(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [shorten_citations(item) for item in value]
            if isinstance(value, tuple):
                return [shorten_citations(item) for item in value]
            if isinstance(value, str):
                pointer = value.removeprefix("ev:")
                alias = self._citation_aliases.get(pointer)
                return alias if alias is not None else value
            return value

        candidate = (
            shorten_citations(dict(current_candidate))
            if isinstance(current_candidate, Mapping)
            else None
        )
        kept.append(
            {
                "role": "user",
                "content": (
                    "CURRENT DIAGNOSIS CANDIDATE FOR DETERMINISTIC REPAIR. "
                    "This is the only candidate version to edit; superseded "
                    "synthesis/revision outputs were removed from model context. "
                    "Preserve valid fields and change only fields named by the "
                    "following revision feedback:\n"
                    + json.dumps(
                        candidate,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                ),
            }
        )
        return kept

    def _retain_key_tool_evidence(
        self,
        transient_messages: Sequence[Mapping[str, Any]],
    ) -> None:
        retained_tools = {
            "get_global_table",
            "get_rhythm_profile",
            "get_atrial_event_table",
            "get_p_assessment_table",
            "get_morphology_groups",
            "get_morphology_map",
            "get_interval_waveform_context",
            "get_native_beat_profile",
            "get_pacing_profile",
        }
        for message in transient_messages:
            content = str(message.get("content") or "")
            rows = _parse_prefixed_json(content, _TOOL_RESULTS_PREFIX)
            if rows is None:
                payload = _parse_prefixed_json(content, _INFLIGHT_LEDGER_PREFIX)
                rows = (
                    payload.get("evidence_views")
                    if isinstance(payload, Mapping)
                    else None
                )
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                tool = str(row.get("tool") or "")
                if tool not in retained_tools or bool(row.get("is_error")):
                    continue
                arguments = row.get("arguments")
                arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
                key = tool + ":" + json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                self._retained_sequence += 1
                result = str(row.get("result") or "")
                if len(result) > MAX_RETAINED_TOOL_RESULT_CHARS:
                    result = (
                        result[:MAX_RETAINED_TOOL_RESULT_CHARS]
                        + "\n[retained excerpt truncated; re-query for full detail]"
                    )
                self._retained_tool_evidence[key] = {
                    "tool": tool,
                    "arguments": arguments,
                    "result": result,
                    "sequence": self._retained_sequence,
                }

    def _render_retained_tool_evidence(self) -> list[dict[str, Any]]:
        rows = sorted(
            self._retained_tool_evidence.values(),
            key=lambda row: (
                self._tool_evidence_priority(str(row.get("tool") or "")),
                -int(row.get("sequence") or 0),
            ),
        )
        rendered: list[dict[str, Any]] = []
        used = 0
        for row in rows:
            public = {
                "tool": row["tool"],
                "arguments": row["arguments"],
                "result": row["result"],
            }
            size = len(
                json.dumps(
                    public,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if rendered and used + size > MAX_RETAINED_TOOL_EVIDENCE_CHARS:
                continue
            rendered.append(public)
            used += size
        return rendered

    def normalize_verdict(self, verdict: dict[str, Any]) -> dict[str, Any]:
        """Expand exact citation aliases without asking the model to copy paths."""

        def expand(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): expand(item) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item) for item in value]
            if isinstance(value, tuple):
                return [expand(item) for item in value]
            if isinstance(value, str):
                alias = value.strip()
                if alias.startswith("ev:"):
                    alias = alias[3:]
                pointer = self._alias_pointers.get(alias)
                if pointer is not None:
                    return f"ev:{pointer}"
            return value

        normalized = expand(verdict)
        return normalized if isinstance(normalized, dict) else verdict

    def citation_aliases(self) -> dict[str, str]:
        """Return the session alias-to-pointer map for phase provenance gates."""

        return dict(self._alias_pointers)

    def model_visible_citations(
        self,
        outcomes: Sequence[ToolOutcome],
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[str, ...]]:
        """Report citations surviving the exact compacted next-turn context."""

        return {
            outcome.call_id: visible_citations(
                messages,
                outcome.model_citations
                if outcome.model_text is not None
                else outcome.citations,
                aliases=self._alias_pointers,
                # Local value citations were deterministically replaced by
                # aliases. An exact ev:/ token in a retained call argument is
                # not proof that the value-bearing result survived compaction.
                allow_exact=False,
            )
            for outcome in outcomes
            if not outcome.is_error
        }

    def tool_context_trace(self) -> list[dict[str, Any]]:
        """Return model-visible excerpts for the separate Markdown trace."""

        return [dict(row) for row in self._tool_context_trace]

    def user_turn(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def usage_total(self) -> dict[str, int]:
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "turns": len(self.turns),
        }
        for turn in self.turns:
            usage = turn.get("usage")
            usage = usage if isinstance(usage, Mapping) else {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    totals[key] += value
        return totals

    def cache_report(self) -> str:
        return (
            "vLLM prefix caching enabled; per-request cache-token accounting is "
            "not exposed by the offline engine"
        )

    def audit_config(self) -> dict[str, Any]:
        scheduler_report = (
            self.scheduler.report()
            if self.scheduler is not None and hasattr(self.scheduler, "report")
            else {
                "configured_max_batch_size": 1,
                "batch_wait_ms": 0.0,
                "batches": len(self.turns),
                "requests": len(self.turns),
                "mean_batch_size": 1.0 if self.turns else 0.0,
                "max_observed_batch_size": 1 if self.turns else 0,
            }
        )
        return {
            "model_path": self.model,
            "model_max_len": self.model_max_len,
            "generation_max_tokens": self.generation_max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_batch_size": self.max_batch_size,
            "batch_wait_ms": self.batch_wait_ms,
            "enforce_phase_coverage": self.enforce_phase_coverage,
            "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            "hard_phase_guards": self.hard_phase_guards,
            "scheduler": scheduler_report,
            "tensor_parallel_size": self._runtime_config.get(
                "tensor_parallel_size"
            ),
            "quantization": self._runtime_config.get("quantization"),
            "device_name": self._runtime_config.get("device_name"),
            "citation_alias_count": len(self._alias_pointers),
            "citation_aliases": dict(sorted(self._alias_pointers.items())),
            "phase_memory": {
                "completed_phases": list(self._phase_memory_history),
                "current_state_phase": (
                    next(iter(self._phase_memories), None)
                ),
                "domain_count": len(_DOMAIN_KEYS),
                "retained_tool_results": len(self._retained_tool_evidence),
                "retained_context_chars": len(
                    json.dumps(
                        self._render_retained_tool_evidence(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
                "retained_context_cap_chars": MAX_RETAINED_TOOL_EVIDENCE_CHARS,
                "inflight_context_cap_chars": MAX_INFLIGHT_TOOL_EVIDENCE_CHARS,
                "per_transient_result_cap_chars": MAX_TRANSIENT_TOOL_RESULT_CHARS,
            },
            "tool_context_compression": {
                **self._compression,
                "saved_chars": max(
                    0,
                    self._compression.get("original_chars", 0)
                    - self._compression.get("context_chars", 0),
                ),
            },
            "context_preflight": {
                "safety_tokens": CONTEXT_SAFETY_TOKENS,
                "checks": list(self._preflight_checks),
                "peak_initial_estimated_prompt_tokens": max(
                    (
                        int(row.get("initial_estimated_prompt_tokens") or 0)
                        for row in self._preflight_checks
                    ),
                    default=0,
                ),
                "peak_final_estimated_prompt_tokens": max(
                    (
                        int(row.get("final_estimated_prompt_tokens") or 0)
                        for row in self._preflight_checks
                    ),
                    default=0,
                ),
            },
        }
