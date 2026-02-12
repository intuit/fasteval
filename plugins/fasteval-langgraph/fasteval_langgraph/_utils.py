"""Internal utilities: deep merge and auto-detection helpers."""

from __future__ import annotations

from typing import Any


def deep_merge(*dicts: dict[str, Any] | None) -> dict[str, Any]:
    """Merge dicts left-to-right. Later values win. Nested dicts merge recursively.

    ``None`` values are skipped (no-op layer).

    Examples:
        >>> deep_merge({"a": 1}, {"b": 2})
        {'a': 1, 'b': 2}
        >>> deep_merge({"c": {"x": 1}}, {"c": {"y": 2}})
        {'c': {'x': 1, 'y': 2}}
        >>> deep_merge({"c": {"x": 1}}, {"c": {"x": 9}})
        {'c': {'x': 9}}
        >>> deep_merge(None, {"a": 1}, None)
        {'a': 1}
    """
    result: dict[str, Any] = {}
    for d in dicts:
        if d is None:
            continue
        for key, value in d.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def has_messages_channel(compiled_graph: Any) -> bool:
    """Detect whether a compiled LangGraph uses MessagesState.

    Checks for the presence of a ``messages`` key in the graph's channel
    definitions, which indicates a ``MessagesState``-based graph.
    """
    # CompiledStateGraph stores channels as a dict
    channels = getattr(compiled_graph, "channels", None)
    if channels and isinstance(channels, dict):
        return "messages" in channels

    # Fallback: inspect the builder's schema if available
    builder = getattr(compiled_graph, "builder", None)
    if builder is not None:
        schema = getattr(builder, "schema", None)
        if schema is not None:
            annotations = getattr(schema, "__annotations__", {})
            return "messages" in annotations

    return False


def extract_last_ai_message(state: dict[str, Any]) -> str:
    """Extract the content of the last AI message from state['messages'].

    Returns empty string if no AI message is found.
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        # Support both LangChain message objects and plain dicts
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content or ""
        if isinstance(msg, dict) and msg.get("type") == "ai":
            return msg.get("content", "")
    return ""


def default_state_filter_messages(state: dict[str, Any]) -> dict[str, Any]:
    """Filter state by excluding the ``messages`` key."""
    return {k: v for k, v in state.items() if k != "messages"}


def default_state_filter_identity(state: dict[str, Any]) -> dict[str, Any]:
    """Identity filter -- return all keys."""
    return dict(state)
