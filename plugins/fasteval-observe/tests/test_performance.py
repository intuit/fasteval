"""Performance benchmarks for fasteval-observe."""

import asyncio
import statistics
import time
from typing import List

import pytest

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


def measure_time(func, *args, **kwargs) -> float:
    """Measure execution time in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000, result


async def measure_time_async(coro) -> float:
    """Measure async execution time in milliseconds."""
    start = time.perf_counter()
    result = await coro
    end = time.perf_counter()
    return (end - start) * 1000, result


class TestDecoratorOverhead:
    """Benchmark decorator overhead."""

    @pytest.mark.asyncio
    async def test_unsampled_overhead_under_1ms(self):
        """
        Decorator overhead when NOT sampled should be < 1ms.

        This is the critical path - most calls won't be sampled.
        """
        # Use 0% sampling rate - never samples
        @observe(sampling=FixedRateSamplingStrategy(rate=0.0))
        async def fast_func():
            return "result"

        # Baseline without decorator
        async def baseline():
            return "result"

        # Warmup
        for _ in range(100):
            await fast_func()
            await baseline()

        # Measure decorated
        decorated_times: List[float] = []
        for _ in range(1000):
            t, _ = await measure_time_async(fast_func())
            decorated_times.append(t)

        # Measure baseline
        baseline_times: List[float] = []
        for _ in range(1000):
            t, _ = await measure_time_async(baseline())
            baseline_times.append(t)

        # Calculate overhead
        avg_decorated = statistics.mean(decorated_times)
        avg_baseline = statistics.mean(baseline_times)
        overhead = avg_decorated - avg_baseline

        print(f"\nUnsampled overhead: {overhead:.4f}ms")
        print(f"Decorated avg: {avg_decorated:.4f}ms")
        print(f"Baseline avg: {avg_baseline:.4f}ms")

        # Overhead should be < 1ms (ideally < 0.1ms)
        assert overhead < 1.0, f"Overhead {overhead}ms exceeds 1ms target"

    @pytest.mark.asyncio
    async def test_sampled_overhead_under_1ms(self):
        """
        Decorator overhead when sampled should be < 1ms.

        The actual observation is enqueued asynchronously.
        """
        # Use 100% sampling rate
        @observe(sampling=NoSamplingStrategy())
        async def observed_func():
            return "result"

        # Baseline
        async def baseline():
            return "result"

        # Warmup
        for _ in range(100):
            await observed_func()
            await baseline()

        # Measure
        decorated_times: List[float] = []
        for _ in range(1000):
            t, _ = await measure_time_async(observed_func())
            decorated_times.append(t)

        baseline_times: List[float] = []
        for _ in range(1000):
            t, _ = await measure_time_async(baseline())
            baseline_times.append(t)

        avg_decorated = statistics.mean(decorated_times)
        avg_baseline = statistics.mean(baseline_times)
        overhead = avg_decorated - avg_baseline

        print(f"\nSampled overhead: {overhead:.4f}ms")
        print(f"Decorated avg: {avg_decorated:.4f}ms")
        print(f"Baseline avg: {avg_baseline:.4f}ms")

        # Overhead should be < 1ms
        assert overhead < 1.0, f"Overhead {overhead}ms exceeds 1ms target"

    @pytest.mark.asyncio
    async def test_disabled_config_minimal_overhead(self):
        """Disabled config should have minimal overhead."""
        configure_observe(ObserveConfig(enabled=False))

        @observe(sampling=NoSamplingStrategy())
        async def disabled_func():
            return "result"

        async def baseline():
            return "result"

        # Warmup
        for _ in range(100):
            await disabled_func()

        # Measure
        times: List[float] = []
        for _ in range(1000):
            t, _ = await measure_time_async(disabled_func())
            times.append(t)

        baseline_times: List[float] = []
        for _ in range(1000):
            t, _ = await measure_time_async(baseline())
            baseline_times.append(t)

        overhead = statistics.mean(times) - statistics.mean(baseline_times)

        print(f"\nDisabled overhead: {overhead:.4f}ms")

        # Should be negligible when disabled
        assert overhead < 0.5


class TestQueuePerformance:
    """Benchmark queue operations."""

    def test_enqueue_performance(self):
        """Enqueue should be fast (non-blocking)."""
        from fasteval_observe.async_handler import get_observation_queue
        from fasteval_observe.metrics import Observation, ObservationMetrics

        queue = get_observation_queue()

        obs = Observation(
            function_name="test",
            sampling_strategy="Test",
            metrics=ObservationMetrics(latency_ms=100, success=True),
        )

        # Warmup
        for _ in range(100):
            queue.enqueue(obs)

        # Measure
        times: List[float] = []
        for _ in range(10000):
            start = time.perf_counter()
            queue.enqueue(obs)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = statistics.mean(times)
        p99_time = sorted(times)[int(len(times) * 0.99)]

        print(f"\nEnqueue avg: {avg_time:.4f}ms")
        print(f"Enqueue p99: {p99_time:.4f}ms")

        # Enqueue should be < 0.1ms on average
        assert avg_time < 0.1, f"Enqueue avg {avg_time}ms exceeds 0.1ms"


class TestSamplingStrategyPerformance:
    """Benchmark sampling strategy decision time."""

    def test_fixed_rate_decision_time(self):
        """FixedRateSamplingStrategy decision should be fast."""
        strategy = FixedRateSamplingStrategy(rate=0.05)

        # Warmup
        for _ in range(100):
            strategy.should_sample("test", (), {}, {})

        # Measure
        times: List[float] = []
        for _ in range(10000):
            start = time.perf_counter()
            strategy.should_sample("test", (), {}, {})
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = statistics.mean(times)

        print(f"\nFixed rate sampling decision avg: {avg_time:.6f}ms")

        # Should be < 0.01ms (10 microseconds)
        assert avg_time < 0.01

    def test_no_sampling_decision_time(self):
        """NoSamplingStrategy decision should be instant."""
        strategy = NoSamplingStrategy()

        times: List[float] = []
        for _ in range(10000):
            start = time.perf_counter()
            strategy.should_sample("test", (), {}, {})
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = statistics.mean(times)

        print(f"\nNo sampling decision avg: {avg_time:.6f}ms")

        # Should be nearly instant
        assert avg_time < 0.01


class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_queue_bounded_memory(self):
        """Queue should have bounded memory usage."""
        configure_observe(ObserveConfig(max_queue_size=100))
        ObservationQueue.reset_instance()

        from fasteval_observe.async_handler import get_observation_queue
        from fasteval_observe.metrics import Observation, ObservationMetrics

        queue = get_observation_queue()

        obs = Observation(
            function_name="test",
            sampling_strategy="Test",
            metrics=ObservationMetrics(latency_ms=100, success=True),
        )

        # Try to enqueue more than max size
        for _ in range(500):
            queue.enqueue(obs)

        # Queue should not exceed max size
        assert queue.stats["queue_size"] <= 100
        assert queue.stats["dropped"] > 0

        print(f"\nQueue stats: {queue.stats}")
