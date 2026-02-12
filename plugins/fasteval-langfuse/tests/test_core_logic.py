"""Tests for core logic without external dependencies."""

from fasteval_langfuse.utils import extract_context_from_trace, format_sampling_stats


def test_context_extraction():
    """Test context extraction logic."""
    # Test with context key
    trace = {"metadata": {"context": ["doc1", "doc2"]}}
    assert extract_context_from_trace(trace) == ["doc1", "doc2"]

    # Test with string context
    trace = {"metadata": {"retrieved_docs": "single doc"}}
    assert extract_context_from_trace(trace) == ["single doc"]

    # Test with no context
    trace = {"metadata": {}}
    assert extract_context_from_trace(trace) is None


def test_sampling_stats_formatting():
    """Test sampling statistics formatting."""
    stats = format_sampling_stats(200, 1000, "RandomSamplingStrategy")
    assert "200" in stats
    assert "1,000" in stats
    assert "20.0%" in stats
    assert "RandomSamplingStrategy" in stats

    # Test with 100% sampling
    stats = format_sampling_stats(1000, 1000, "NoSamplingStrategy")
    assert "100.0%" in stats


def test_sampling_strategies_imported():
    """Test that sampling strategies can be imported."""
    from fasteval_langfuse.sampling import (
        BaseSamplingStrategy,
        NoSamplingStrategy,
        RandomSamplingStrategy,
        ScoreBasedSamplingStrategy,
        StratifiedSamplingStrategy,
    )

    # Verify they're classes
    assert isinstance(NoSamplingStrategy, type)
    assert isinstance(RandomSamplingStrategy, type)
    assert isinstance(ScoreBasedSamplingStrategy, type)
    assert isinstance(StratifiedSamplingStrategy, type)
    assert isinstance(BaseSamplingStrategy, type)
