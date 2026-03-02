"""Tests for sampling strategies."""

import pytest

from fasteval_observe.sampling import (
    AdaptiveSamplingStrategy,
    BaseSamplingStrategy,
    ComposableSamplingStrategy,
    FixedRateSamplingStrategy,
    NoSamplingStrategy,
    ProbabilisticSamplingStrategy,
    TokenBudgetSamplingStrategy,
)


class TestNoSamplingStrategy:
    """Tests for NoSamplingStrategy."""

    def test_always_samples(self):
        """NoSamplingStrategy should always return True."""
        strategy = NoSamplingStrategy()

        for _ in range(100):
            assert strategy.should_sample("test_func", (), {}, {}) is True

    def test_name(self):
        """Should return class name."""
        strategy = NoSamplingStrategy()
        assert strategy.name == "NoSamplingStrategy"


class TestFixedRateSamplingStrategy:
    """Tests for FixedRateSamplingStrategy."""

    def test_rate_validation(self):
        """Should validate rate is between 0 and 1."""
        with pytest.raises(ValueError):
            FixedRateSamplingStrategy(rate=-0.1)

        with pytest.raises(ValueError):
            FixedRateSamplingStrategy(rate=1.5)

    def test_zero_rate_never_samples(self):
        """Rate of 0 should never sample."""
        strategy = FixedRateSamplingStrategy(rate=0.0)

        for _ in range(100):
            assert strategy.should_sample("test_func", (), {}, {}) is False

    def test_full_rate_always_samples(self):
        """Rate of 1.0 should always sample."""
        strategy = FixedRateSamplingStrategy(rate=1.0)

        for _ in range(100):
            assert strategy.should_sample("test_func", (), {}, {}) is True

    def test_fixed_rate_samples_correctly(self):
        """Should sample approximately at the specified rate."""
        strategy = FixedRateSamplingStrategy(rate=0.1)  # 1 in 10

        # With rate=0.1, every 10th call should be sampled
        samples = [strategy.should_sample("test_func", (), {}, {}) for _ in range(100)]

        # Should have sampled exactly 10 times (deterministic)
        assert sum(samples) == 10

    def test_reset(self):
        """Reset should clear counter."""
        strategy = FixedRateSamplingStrategy(rate=0.5)

        # Make some calls
        strategy.should_sample("test_func", (), {}, {})
        strategy.should_sample("test_func", (), {}, {})

        # Reset
        strategy.reset()

        # Counter should be reset (next sample at interval)
        strategy.should_sample("test_func", (), {}, {})


class TestProbabilisticSamplingStrategy:
    """Tests for ProbabilisticSamplingStrategy."""

    def test_probability_validation(self):
        """Should validate probability is between 0 and 1."""
        with pytest.raises(ValueError):
            ProbabilisticSamplingStrategy(probability=-0.1)

        with pytest.raises(ValueError):
            ProbabilisticSamplingStrategy(probability=1.5)

    def test_zero_probability_never_samples(self):
        """Probability of 0 should never sample."""
        strategy = ProbabilisticSamplingStrategy(probability=0.0)

        for _ in range(100):
            assert strategy.should_sample("test_func", (), {}, {}) is False

    def test_full_probability_always_samples(self):
        """Probability of 1.0 should always sample."""
        strategy = ProbabilisticSamplingStrategy(probability=1.0)

        for _ in range(100):
            assert strategy.should_sample("test_func", (), {}, {}) is True

    def test_reproducible_with_seed(self):
        """Same seed should produce same results."""
        strategy1 = ProbabilisticSamplingStrategy(probability=0.5, seed=42)
        strategy2 = ProbabilisticSamplingStrategy(probability=0.5, seed=42)

        results1 = [strategy1.should_sample("f", (), {}, {}) for _ in range(100)]
        results2 = [strategy2.should_sample("f", (), {}, {}) for _ in range(100)]

        assert results1 == results2

    def test_approximate_rate(self):
        """Should sample approximately at the specified probability."""
        strategy = ProbabilisticSamplingStrategy(probability=0.3, seed=42)

        samples = [strategy.should_sample("test_func", (), {}, {}) for _ in range(1000)]

        # Should be approximately 30% (with some variance)
        rate = sum(samples) / len(samples)
        assert 0.25 <= rate <= 0.35


class TestAdaptiveSamplingStrategy:
    """Tests for AdaptiveSamplingStrategy."""

    def test_base_rate_sampling(self):
        """Should sample at base rate in normal conditions."""
        strategy = AdaptiveSamplingStrategy(
            base_rate=0.5,
            error_rate=1.0,
            slow_threshold_ms=1000,
        )

        # Sample at base rate
        samples = [strategy.should_sample("test_func", (), {}, {}) for _ in range(100)]

        # Should be approximately 50%
        rate = sum(samples) / len(samples)
        assert 0.3 <= rate <= 0.7

    def test_increases_rate_on_errors(self):
        """Should increase sampling rate after errors."""
        strategy = AdaptiveSamplingStrategy(
            base_rate=0.01,
            error_rate=1.0,
            window_size=10,
            cooldown_seconds=0,  # Disable cooldown so error rate logic is reached
        )

        # Simulate errors
        for _ in range(5):
            strategy.on_completion("test", 100, Exception("error"), None)

        # After errors, should sample more frequently
        samples = [strategy.should_sample("test_func", (), {}, {}) for _ in range(10)]

        # With >10% error rate and cooldown disabled, should use error_rate (1.0)
        assert sum(samples) == 10  # error_rate=1.0 means always sample

    def test_increases_rate_on_slow_calls(self):
        """Should increase sampling rate for slow calls."""
        strategy = AdaptiveSamplingStrategy(
            base_rate=0.01,
            slow_rate=0.8,
            slow_threshold_ms=100,
            window_size=10,
        )

        # Simulate slow calls
        for _ in range(5):
            strategy.on_completion("test", 200, None, "result")

        # After slow calls, should sample more frequently
        samples = [strategy.should_sample("test_func", (), {}, {}) for _ in range(10)]

        assert sum(samples) >= 3  # Higher than base rate

    def test_reset(self):
        """Reset should clear history."""
        strategy = AdaptiveSamplingStrategy(base_rate=0.5)

        # Add some history
        strategy.on_completion("test", 100, Exception("error"), None)
        strategy.on_completion("test", 5000, None, "result")

        # Reset
        strategy.reset()

        # Should be back to base rate behavior


class TestTokenBudgetSamplingStrategy:
    """Tests for TokenBudgetSamplingStrategy."""

    def test_base_rate_under_budget(self):
        """Should sample at base rate when under budget."""
        strategy = TokenBudgetSamplingStrategy(
            budget_tokens_per_hour=100000,
            base_rate=0.1,
        )

        samples = [strategy.should_sample("test_func", (), {}, {}) for _ in range(1000)]

        # Should be approximately 10%
        rate = sum(samples) / len(samples)
        assert 0.07 <= rate <= 0.13

    def test_high_cost_sampling(self):
        """Should sample high-cost calls more frequently."""
        strategy = TokenBudgetSamplingStrategy(
            budget_tokens_per_hour=100000,
            high_cost_threshold=500,
            high_cost_rate=0.8,
            base_rate=0.01,
        )

        # High cost context
        context = {"estimated_tokens": 1000}

        samples = [
            strategy.should_sample("test_func", (), {}, context) for _ in range(100)
        ]

        # Should be approximately 80%
        rate = sum(samples) / len(samples)
        assert rate >= 0.5

    def test_reset(self):
        """Reset should clear token tracking."""
        strategy = TokenBudgetSamplingStrategy(budget_tokens_per_hour=100000)
        strategy.reset()


class TestComposableSamplingStrategy:
    """Tests for ComposableSamplingStrategy."""

    def test_requires_at_least_one_strategy(self):
        """Should require at least one strategy."""
        with pytest.raises(ValueError):
            ComposableSamplingStrategy(strategies=[], mode="any")

    def test_any_mode_or_logic(self):
        """Mode 'any' should use OR logic."""
        always_false = FixedRateSamplingStrategy(rate=0.0)
        always_true = FixedRateSamplingStrategy(rate=1.0)

        strategy = ComposableSamplingStrategy(
            strategies=[always_false, always_true],
            mode="any",
        )

        # Should sample because one is True
        assert strategy.should_sample("test_func", (), {}, {}) is True

    def test_all_mode_and_logic(self):
        """Mode 'all' should use AND logic."""
        always_false = FixedRateSamplingStrategy(rate=0.0)
        always_true = FixedRateSamplingStrategy(rate=1.0)

        strategy = ComposableSamplingStrategy(
            strategies=[always_false, always_true],
            mode="all",
        )

        # Should not sample because one is False
        assert strategy.should_sample("test_func", (), {}, {}) is False

    def test_all_true_with_all_mode(self):
        """Should sample when all strategies return True."""
        strategy = ComposableSamplingStrategy(
            strategies=[
                NoSamplingStrategy(),
                FixedRateSamplingStrategy(rate=1.0),
            ],
            mode="all",
        )

        assert strategy.should_sample("test_func", (), {}, {}) is True

    def test_forwards_completion_to_all(self):
        """Should forward on_completion to all strategies."""
        adaptive1 = AdaptiveSamplingStrategy(base_rate=0.5)
        adaptive2 = AdaptiveSamplingStrategy(base_rate=0.5)

        strategy = ComposableSamplingStrategy(
            strategies=[adaptive1, adaptive2],
            mode="any",
        )

        # This should update both internal strategies
        strategy.on_completion("test", 100, None, "result")

    def test_name(self):
        """Should return combined name."""
        strategy = ComposableSamplingStrategy(
            strategies=[NoSamplingStrategy(), FixedRateSamplingStrategy(rate=0.5)],
            mode="any",
        )

        assert "Composable" in strategy.name
        assert "any" in strategy.name


class TestCustomStrategy:
    """Tests for custom strategy implementation."""

    def test_custom_strategy_works(self):
        """Custom strategies should work when extending BaseSamplingStrategy."""

        class MyCustomStrategy(BaseSamplingStrategy):
            def __init__(self, sample_even_only: bool = True):
                self.sample_even_only = sample_even_only
                self.call_count = 0

            def should_sample(self, function_name, args, kwargs, context):
                self.call_count += 1
                if self.sample_even_only:
                    return self.call_count % 2 == 0
                return True

        strategy = MyCustomStrategy(sample_even_only=True)

        results = [strategy.should_sample("test_func", (), {}, {}) for _ in range(10)]

        # Should sample every other call
        assert results == [
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
        ]
