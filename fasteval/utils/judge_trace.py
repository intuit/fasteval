"""
Record LLM judge calls made while a metric evaluates.

The evaluator opens a capture around each ``metric.evaluate()``; metrics
wrap their client with :func:`traced`. Every ``invoke()`` made through the
wrapper is logged (judge identity, prompt messages, raw response, latency,
error) and the evaluator attaches the log to the metric result as
``details["judge_calls"]``. Outside a capture the wrapper is a pass-through.

This is what lets reports and the UI show exactly which model judged an
output, the prompt it was given, and what it answered on every attempt.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

# Large inputs (documents, transcripts) are kept but bounded so reports
# stay loadable in a browser.
MAX_TEXT_CHARS = 20_000

_calls: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "fasteval_judge_calls", default=None
)


@contextmanager
def capture() -> Iterator[List[Dict[str, Any]]]:
    """Collect judge calls made in this context (task-local under asyncio)."""
    calls: List[Dict[str, Any]] = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(token)


def describe_judge(client: Any) -> Dict[str, Any]:
    """Best-effort identity of an LLM client: provider class and model."""
    inner = getattr(client, "_fasteval_inner", client)
    info: Dict[str, Any] = {"provider": type(inner).__name__}
    for attr in ("model", "model_name", "deployment"):
        value = getattr(inner, attr, None)
        if isinstance(value, str) and value:
            info["model"] = value
            break
    temperature = getattr(inner, "temperature", None)
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        info["temperature"] = temperature
    return info


class TracedClient:
    """Pass-through LLM client wrapper that records ``invoke()`` calls."""

    def __init__(self, inner: Any) -> None:
        self._fasteval_inner = inner

    async def invoke(self, messages: List[Dict[str, Any]]) -> str:
        calls = _calls.get()
        if calls is None:
            return await self._fasteval_inner.invoke(messages)

        entry: Dict[str, Any] = {
            "judge": describe_judge(self._fasteval_inner),
            "messages": _sanitize_messages(messages),
        }
        start = time.perf_counter()
        try:
            response = await self._fasteval_inner.invoke(messages)
            entry["response"] = (
                _truncate(response) if isinstance(response, str) else response
            )
            return response
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            entry["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            calls.append(entry)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fasteval_inner, name)

    def __repr__(self) -> str:
        return repr(self._fasteval_inner)


def traced(client: Any) -> Any:
    """Wrap ``client`` so its judge calls are recorded during a capture."""
    if client is None or isinstance(client, TracedClient):
        return client
    return TracedClient(client)


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + f"\n… [{len(text) - MAX_TEXT_CHARS} chars truncated]"


def _sanitize_messages(messages: Any) -> Any:
    """Copy messages with long text truncated and binary/image data elided."""
    if not isinstance(messages, list):
        return messages
    out: List[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(str(message))
            continue
        copy = dict(message)
        content = copy.get("content")
        if isinstance(content, str):
            copy["content"] = _truncate(content)
        elif isinstance(content, list):
            copy["content"] = [_sanitize_part(p) for p in content]
        out.append(copy)
    return out


def _sanitize_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    kind = part.get("type", "")
    if kind == "text" and isinstance(part.get("text"), str):
        return {**part, "text": _truncate(part["text"])}
    if kind in ("image_url", "image", "input_audio", "audio", "document"):
        return {"type": kind, "omitted": "binary content not recorded"}
    return part
