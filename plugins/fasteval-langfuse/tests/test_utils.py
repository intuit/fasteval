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


def test_parse_time_range_empty():
    """Test empty time range returns None."""
    from_ts, to_ts = parse_time_range("")
    assert from_ts is None
    assert to_ts is None


def test_parse_time_range_invalid():
    """Test invalid time range format raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unsupported"):
        parse_time_range("invalid_format")


def test_parse_time_range_invalid_duration():
    """Test invalid duration suffix raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Invalid time range"):
        parse_time_range("last_5m")


def test_extract_context_non_list_value():
    """Test context extraction with non-list, non-string value."""
    trace = {"metadata": {"context": 42}}
    context = extract_context_from_trace(trace)
    assert context == ["42"]


def test_extract_context_none_value():
    """Test context extraction skips None values."""
    trace = {"metadata": {"context": None, "retrieved_docs": ["doc1"]}}
    context = extract_context_from_trace(trace)
    assert context == ["doc1"]


def test_format_sampling_stats():
    """Test sampling statistics formatting."""
    stats = format_sampling_stats(200, 1000, "RandomSamplingStrategy")
    assert "200" in stats
    assert "1,000" in stats
    assert "20.0%" in stats
    assert "RandomSamplingStrategy" in stats


def test_format_sampling_stats_zero_total():
    """Test formatting with zero total traces."""
    stats = format_sampling_stats(0, 0, "NoSamplingStrategy")
    assert "0.0%" in stats
