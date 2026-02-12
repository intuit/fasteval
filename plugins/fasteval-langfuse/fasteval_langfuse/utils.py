"""Utility functions for trace processing."""

from typing import Any, Dict, List, Optional


def extract_context_from_trace(trace: Dict[str, Any]) -> Optional[List[str]]:
    """
    Extract context from trace metadata.

    Looks for common context keys in metadata and returns as list of strings.

    Args:
        trace: Trace dictionary

    Returns:
        List of context strings, or None if not found
    """
    metadata = trace.get("metadata", {})

    # Common context keys
    context_keys = [
        "context",
        "retrieved_docs",
        "documents",
        "retrieval_context",
        "docs",
    ]

    for key in context_keys:
        if key in metadata:
            context = metadata[key]

            # Normalize to list of strings
            if context is None:
                continue
            elif isinstance(context, list):
                return [str(item) for item in context]
            elif isinstance(context, str):
                return [context]
            else:
                return [str(context)]

    return None


def parse_time_range(time_range: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse time range string to from/to timestamps.

    Supports formats like:
    - "last_24h", "last_7d", "last_30d"
    - "2026-02-01 to 2026-02-05"

    Args:
        time_range: Time range string

    Returns:
        Tuple of (from_timestamp, to_timestamp) in ISO 8601 format
    """
    from datetime import datetime, timedelta

    if not time_range:
        return None, None

    # Handle "last_Xh" or "last_Xd" format
    if time_range.startswith("last_"):
        parts = time_range.split("_")
        if len(parts) == 2:
            duration_str = parts[1]

            # Parse duration
            if duration_str.endswith("h"):
                hours = int(duration_str[:-1])
                delta = timedelta(hours=hours)
            elif duration_str.endswith("d"):
                days = int(duration_str[:-1])
                delta = timedelta(days=days)
            else:
                raise ValueError(f"Invalid time range format: {time_range}")

            to_timestamp = datetime.utcnow().isoformat() + "Z"
            from_timestamp = (datetime.utcnow() - delta).isoformat() + "Z"
            return from_timestamp, to_timestamp

    # Handle "YYYY-MM-DD to YYYY-MM-DD" format
    if " to " in time_range:
        from_str, to_str = time_range.split(" to ")
        from_timestamp = datetime.fromisoformat(from_str.strip()).isoformat() + "Z"
        to_timestamp = datetime.fromisoformat(to_str.strip()).isoformat() + "Z"
        return from_timestamp, to_timestamp

    raise ValueError(f"Unsupported time range format: {time_range}")


def format_sampling_stats(
    sampled_count: int, total_count: int, strategy_name: str
) -> str:
    """
    Format sampling statistics for display.

    Args:
        sampled_count: Number of sampled traces
        total_count: Total number of traces
        strategy_name: Name of sampling strategy

    Returns:
        Formatted statistics string
    """
    if total_count == 0:
        percentage = 0.0
    else:
        percentage = (sampled_count / total_count) * 100

    return (
        f"Evaluating {sampled_count:,}/{total_count:,} traces "
        f"({percentage:.1f}% sample, strategy={strategy_name})"
    )
