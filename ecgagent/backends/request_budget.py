"""One absolute monotonic deadline shared by HTTP attempts and backoff."""
from __future__ import annotations

import time
from typing import Any, Callable


def remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("model request deadline exhausted")
    return remaining


def openai_completion(
    client: Any, kwargs: dict[str, Any], *, deadline: float | None,
    timeout: float, max_retries: int,
) -> Any:
    if deadline is None:
        return client.chat.completions.create(**kwargs)
    for attempt in range(max(0, max_retries) + 1):
        remaining = remaining_seconds(deadline)
        # Disable hidden SDK retries: each attempt must recalculate the same
        # record deadline, rather than receiving a fresh full timeout.
        options = getattr(client, "with_options", None)
        scoped = options(timeout=min(timeout, remaining), max_retries=0) if callable(options) else client
        request = dict(kwargs)
        if not callable(options):
            request["timeout"] = min(timeout, remaining)
        try:
            result = scoped.chat.completions.create(**request)
            remaining_seconds(deadline)
            return result
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status in {408, 409, 429} or (isinstance(status, int) and status >= 500)
            retryable = retryable or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
            if not retryable or attempt >= max_retries:
                raise
            delay = min(0.25 * 2 ** attempt, 2.0)
            if remaining_seconds(deadline) <= delay:
                raise TimeoutError("model request deadline exhausted before retry") from exc
            time.sleep(delay)
    raise AssertionError("unreachable")
