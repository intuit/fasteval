"""Tests for async handler (queue and circuit breaker)."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from fasteval_observe.async_handler import (
    CircuitBreaker,
    ObservationQueue,
    get_observation_queue,
)
from fasteval_observe.config import ObserveConfig, configure_observe, reset_config
from fasteval_observe.metrics import Observation, ObservationMetrics


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test."""
    ObservationQueue.reset_instance()
    reset_config()
    yield
    ObservationQueue.reset_instance()
    reset_config()


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_starts_closed(self):
        """Circuit breaker should start in closed state."""
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.is_open is False

    def test_opens_after_threshold_failures(self):
        """Should open after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb.is_open is False

        cb.record_failure()
        assert cb.is_open is False

        cb.record_failure()
        assert cb.is_open is True

    def test_success_resets_failures(self):
        """Success should reset failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        # Should not open yet
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False

    def test_recovers_after_timeout(self):
        """Should recover after recovery time."""
        cb = CircuitBreaker(failure_threshold=2, recovery_time=0.1)

        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        # Wait for recovery
        time.sleep(0.15)

        # Should be half-open now (allows operations)
        assert cb.is_open is False

    def test_reset(self):
        """Reset should clear all state."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        cb.reset()
        assert cb.is_open is False


class TestObservationQueue:
    """Tests for ObservationQueue."""

    def create_observation(self) -> Observation:
        """Create a test observation."""
        return Observation(
            function_name="test_func",
            sampling_strategy="TestStrategy",
            metrics=ObservationMetrics(latency_ms=100.0, success=True),
        )

    def test_singleton_pattern(self):
        """Should return same instance."""
        q1 = get_observation_queue()
        q2 = get_observation_queue()
        assert q1 is q2

    def test_enqueue_observation(self):
        """Should enqueue observations."""
        queue = get_observation_queue()
        obs = self.create_observation()

        result = queue.enqueue(obs)

        assert result is True
        assert queue.stats["enqueued"] == 1

    def test_respects_disabled_config(self):
        """Should not enqueue when disabled."""
        configure_observe(ObserveConfig(enabled=False))
        queue = get_observation_queue()
        obs = self.create_observation()

        result = queue.enqueue(obs)

        assert result is False

    def test_drops_when_queue_full(self):
        """Should drop observations when queue is full."""
        configure_observe(ObserveConfig(max_queue_size=2))
        ObservationQueue.reset_instance()

        queue = get_observation_queue()

        # Fill the queue
        queue.enqueue(self.create_observation())
        queue.enqueue(self.create_observation())

        # This should be dropped
        result = queue.enqueue(self.create_observation())

        assert result is False
        assert queue.stats["dropped"] == 1

    def test_worker_flushes_batch(self):
        """Worker should flush batches to callback."""
        callback = MagicMock()
        queue = get_observation_queue()
        queue.set_flush_callback(callback)
        queue.start_worker()

        # Enqueue some observations
        for _ in range(5):
            queue.enqueue(self.create_observation())

        # Wait for flush
        time.sleep(0.5)

        # Should have been flushed
        assert callback.called
        assert queue.stats["flushed"] >= 1

        queue.shutdown()

    def test_graceful_shutdown(self):
        """Should drain queue on shutdown."""
        callback = MagicMock()
        queue = get_observation_queue()
        queue.set_flush_callback(callback)
        queue.start_worker()

        # Enqueue observations
        for _ in range(10):
            queue.enqueue(self.create_observation())

        # Shutdown
        queue.shutdown(timeout=5.0)

        # All should have been flushed
        assert queue.stats["enqueued"] == queue.stats["flushed"]

    def test_stats(self):
        """Should track statistics correctly."""
        queue = get_observation_queue()

        for _ in range(5):
            queue.enqueue(self.create_observation())

        stats = queue.stats
        assert stats["enqueued"] == 5
        assert stats["queue_size"] == 5
        assert stats["dropped"] == 0
