"""Integration tests with fasteval decorators."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add fasteval to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fasteval_observe import (
    ObserveConfig,
    configure_observe,
    observe,
)
from fasteval_observe.async_handler import ObservationQueue
from fasteval_observe.config import reset_config
from fasteval_observe.sampling import (
    FixedRateSamplingStrategy,
    NoSamplingStrategy,
)


@pytest.fixture(autouse=True)
def reset_all():
    """Reset all singletons before each test."""
    ObservationQueue.reset_instance()
    reset_config()
    yield
    ObservationQueue.reset_instance()
    reset_config()


class TestFastevalDecoratorIntegration:
    """Tests for integration with fasteval decorators."""

    @pytest.mark.asyncio
    async def test_observe_with_fasteval_correctness(self):
        """Should work alongside @fe.correctness decorator."""
        try:
            import fasteval as fe

            @observe(sampling=NoSamplingStrategy())
            @fe.correctness(threshold=0.8)
            async def my_agent(query: str) -> str:
                return f"Answer to: {query}"

            # Function should work
            result = await my_agent("What is Python?")
            assert "Answer to" in result

            # Should have fasteval metrics attached
            assert hasattr(my_agent, "_fasteval_metrics")

        except ImportError:
            pytest.skip("fasteval not available")

    @pytest.mark.asyncio
    async def test_observe_with_multiple_fasteval_decorators(self):
        """Should work with multiple fasteval decorators."""
        try:
            import fasteval as fe

            @observe(sampling=NoSamplingStrategy())
            @fe.correctness(threshold=0.8)
            @fe.relevance(threshold=0.7)
            async def evaluated_agent(query: str) -> str:
                return f"Response: {query}"

            result = await evaluated_agent("test query")
            assert "Response" in result

            # Should have multiple metrics attached
            assert hasattr(evaluated_agent, "_fasteval_metrics")
            assert len(evaluated_agent._fasteval_metrics) >= 2

        except ImportError:
            pytest.skip("fasteval not available")

    @pytest.mark.asyncio
    async def test_run_evaluations_with_fasteval(self):
        """Should run fasteval evaluations when run_evaluations=True."""
        try:
            import fasteval as fe

            @observe(
                sampling=NoSamplingStrategy(),
                run_evaluations=True,
            )
            @fe.correctness(threshold=0.8)
            async def agent_with_eval(query: str) -> str:
                return f"Response: {query}"

            # This should trigger evaluation in background
            result = await agent_with_eval("test")
            assert result == "Response: test"

        except ImportError:
            pytest.skip("fasteval not available")

    @pytest.mark.asyncio
    async def test_observe_order_matters(self):
        """Observe should be outermost decorator."""
        try:
            import fasteval as fe

            # Correct order: @observe on top
            @observe(sampling=NoSamplingStrategy())
            @fe.correctness(threshold=0.8)
            async def correct_order():
                return "result"

            # Should have both attributes
            assert hasattr(correct_order, "_fasteval_metrics")

            result = await correct_order()
            assert result == "result"

        except ImportError:
            pytest.skip("fasteval not available")


class TestObservationFlow:
    """Tests for the complete observation flow."""

    @pytest.mark.asyncio
    async def test_full_observation_flow(self, capsys):
        """Test complete flow from decorator to log output."""
        from fasteval_observe import initialize_observer, shutdown_observer

        # Initialize observer
        initialize_observer()

        @observe(sampling=NoSamplingStrategy())
        async def observed_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        # Call function
        result = await observed_func(5)
        assert result == 10

        # Shutdown to flush logs
        shutdown_observer()

        # Verify log output
        captured = capsys.readouterr()
        assert "observed_func" in captured.out or "observed_func" in captured.err

    @pytest.mark.asyncio
    async def test_sampling_reduces_observations(self, capsys):
        """Sampling should reduce number of observations."""
        from fasteval_observe import initialize_observer, shutdown_observer

        initialize_observer()

        # Very low sampling rate
        @observe(sampling=FixedRateSamplingStrategy(rate=0.1))
        async def sampled_func():
            return "result"

        # Call many times
        for _ in range(100):
            await sampled_func()

        shutdown_observer()

        # Should have approximately 10 observations (10%)
        captured = capsys.readouterr()
        # Count "sampled_func" occurrences in output
        count = captured.out.count("sampled_func")
        assert count <= 20  # Should be around 10, allow some margin

    @pytest.mark.asyncio
    async def test_error_handling_in_observed_function(self):
        """Should handle errors in observed functions."""
        from fasteval_observe import initialize_observer, shutdown_observer

        initialize_observer()

        @observe(sampling=NoSamplingStrategy())
        async def error_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await error_func()

        shutdown_observer()

    @pytest.mark.asyncio
    async def test_metadata_appears_in_logs(self, capsys):
        """Custom metadata should appear in logs."""
        from fasteval_observe import initialize_observer, shutdown_observer

        initialize_observer()

        @observe(
            sampling=NoSamplingStrategy(),
            metadata={"custom_key": "custom_value"},
        )
        async def func_with_metadata():
            return "result"

        await func_with_metadata()
        shutdown_observer()

        captured = capsys.readouterr()
        assert "custom_key" in captured.out
        assert "custom_value" in captured.out
