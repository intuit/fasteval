"""Tests for utility functions."""

from fasteval_langfuse.utils import (
    extract_context_from_trace,
    format_sampling_stats,
    parse_time_range,
)


def test_extract_context_from_trace():
    """Test context extraction from trace metadata."""
    # Test with context key
    trace = {"metadata": {"context": ["doc1", "doc2"]}}
    context = extract_context_from_trace(trace)
    assert context == ["doc1", "doc2"]

    # Test with retrieved_docs key
    trace = {"metadata": {"retrieved_docs": "single doc"}}
    context = extract_context_from_trace(trace)
    assert context == ["single doc"]

    # Test with no context
    trace = {"metadata": {"other_key": "value"}}
    context = extract_context_from_trace(trace)
    assert context is None


def test_parse_time_range_last_format():
    """Test parsing 'last_Xh' and 'last_Xd' formats."""
    from_ts, to_ts = parse_time_range("last_24h")
    assert from_ts is not None
    assert to_ts is not None
    assert from_ts.endswith("Z")
    assert to_ts.endswith("Z")

    from_ts, to_ts = parse_time_range("last_7d")
    assert from_ts is not None
    assert to_ts is not None


def test_parse_time_range_to_format():
    """Test parsing 'YYYY-MM-DD to YYYY-MM-DD' format."""
    from_ts, to_ts = parse_time_range("2026-02-01 to 2026-02-05")
    assert from_ts == "2026-02-01T00:00:00Z"
    assert to_ts == "2026-02-05T00:00:00Z"


def test_format_sampling_stats():
    """Test sampling statistics formatting."""
    stats = format_sampling_stats(200, 1000, "RandomSamplingStrategy")
    assert "200" in stats
    assert "1,000" in stats
    assert "20.0%" in stats
    assert "RandomSamplingStrategy" in stats
