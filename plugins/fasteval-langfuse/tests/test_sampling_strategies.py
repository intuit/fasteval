"""Tests for sampling strategies."""

import pytest

from fasteval_langfuse.sampling import (
    NoSamplingStrategy,
    RandomSamplingStrategy,
    ScoreBasedSamplingStrategy,
    StratifiedSamplingStrategy,
)


def test_no_sampling_strategy(sample_traces):
    """Test NoSamplingStrategy returns all traces."""
    strategy = NoSamplingStrategy()
    sampled = strategy.sample(sample_traces)
    assert len(sampled) == len(sample_traces)
    assert sampled == sample_traces


def test_random_sampling_strategy(sample_traces):
    """Test RandomSamplingStrategy samples correct number."""
    strategy = RandomSamplingStrategy(sample_size=2, seed=42)
    sampled = strategy.sample(sample_traces)
    assert len(sampled) == 2
    assert all(trace in sample_traces for trace in sampled)

    # Test reproducibility with seed
    sampled2 = RandomSamplingStrategy(sample_size=2, seed=42).sample(sample_traces)
    assert sampled == sampled2


def test_random_sampling_strategy_handles_small_input():
    """Test RandomSamplingStrategy with fewer traces than sample_size."""
    strategy = RandomSamplingStrategy(sample_size=10)
    traces = [{"id": "1"}, {"id": "2"}]
    sampled = strategy.sample(traces)
    assert len(sampled) == 2


def test_score_based_sampling_strategy(sample_traces):
    """Test ScoreBasedSamplingStrategy oversamples low scores."""
    strategy = ScoreBasedSamplingStrategy(
        score_name="user_rating",
        low_score_threshold=3.0,
        low_score_rate=1.0,
        high_score_rate=0.0,
        seed=42,
    )
    sampled = strategy.sample(sample_traces)

    # Should only include low-scoring trace (trace-2 with rating 2.0)
    assert len(sampled) == 1
    assert sampled[0]["id"] == "trace-2"


def test_score_based_sampling_handles_missing_scores():
    """Test ScoreBasedSamplingStrategy handles traces without scores."""
    traces = [
        {"id": "1", "scores": [{"name": "other_score", "value": 5.0}]},
        {"id": "2", "scores": []},
    ]

    strategy = ScoreBasedSamplingStrategy(
        score_name="user_rating",
        low_score_threshold=3.0,
        low_score_rate=1.0,
        high_score_rate=0.5,
        seed=42,
    )

    # Should not crash, treats missing scores as high scores
    sampled = strategy.sample(traces)
    assert isinstance(sampled, list)


def test_stratified_sampling_strategy(sample_traces):
    """Test StratifiedSamplingStrategy samples evenly across strata."""
    strategy = StratifiedSamplingStrategy(
        strata_key="metadata.user_type", samples_per_stratum=1, seed=42
    )
    sampled = strategy.sample(sample_traces)

    # Should have 1 from each user type (free, paid)
    assert len(sampled) == 2
    user_types = {trace["metadata"]["user_type"] for trace in sampled}
    assert "free" in user_types
    assert "paid" in user_types


def test_stratified_sampling_nested_key():
    """Test StratifiedSamplingStrategy with nested metadata key."""
    traces = [
        {"id": "1", "metadata": {"nested": {"key": "A"}}},
        {"id": "2", "metadata": {"nested": {"key": "B"}}},
        {"id": "3", "metadata": {"nested": {"key": "A"}}},
    ]

    strategy = StratifiedSamplingStrategy(
        strata_key="metadata.nested.key", samples_per_stratum=1, seed=42
    )
    sampled = strategy.sample(traces)

    # Should have 1 from each group (A, B)
    assert len(sampled) == 2


def test_sampling_strategy_validation():
    """Test validation of strategy parameters."""
    # Invalid sample_size
    with pytest.raises(ValueError):
        RandomSamplingStrategy(sample_size=0)

    # Invalid rates
    with pytest.raises(ValueError):
        ScoreBasedSamplingStrategy(
            score_name="test", low_score_threshold=3.0, low_score_rate=1.5
        )

    # Invalid samples_per_stratum
    with pytest.raises(ValueError):
        StratifiedSamplingStrategy(strata_key="test", samples_per_stratum=-1)
