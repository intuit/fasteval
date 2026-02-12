"""Async queue handler and background worker for observations."""

import atexit
import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional

from fasteval_observe.config import get_config
from fasteval_observe.metrics import Observation

# Use module logger for internal logging (not observations)
logger = logging.getLogger("fasteval_observe.internal")


class CircuitBreaker:
    """
    Circuit breaker for queue overflow protection.

    Prevents resource exhaustion when the logging system can't keep up.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_time: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time

        self._lock = threading.Lock()
        self._failures = 0
        self._state = "closed"  # closed, open, half-open
        self._last_failure_time: Optional[float] = None

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking operations)."""
        with self._lock:
            if self._state == "closed":
                return False

            if self._state == "open":
                # Check if recovery time has passed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_time:
                        self._state = "half-open"
                        return False
                return True

            # half-open state allows operations
            return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._failures = 0
            if self._state == "half-open":
                self._state = "closed"
                logger.info("Circuit breaker closed - system recovered")

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._failures >= self.failure_threshold:
                if self._state != "open":
                    self._state = "open"
                    logger.warning(
                        f"Circuit breaker opened after {self._failures} failures"
                    )

    def reset(self) -> None:
        """Reset circuit breaker state."""
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._last_failure_time = None


class ObservationQueue:
    """
    Thread-safe async queue for observations.

    Single background worker flushes observations to the logger.
    Includes circuit breaker for overflow protection.
    """

    _instance: Optional["ObservationQueue"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ObservationQueue":
        """Singleton pattern for global queue."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize queue (only once due to singleton)."""
        if getattr(self, "_initialized", False):
            return

        config = get_config()
        self._queue: Queue[Observation] = Queue(maxsize=config.max_queue_size)
        self._flush_callback: Optional[Callable[[List[Observation]], None]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_time=config.circuit_breaker_recovery_time,
        )

        # Stats
        self._enqueued = 0
        self._dropped = 0
        self._flushed = 0
        self._stats_lock = threading.Lock()

        self._initialized = True

        # Register shutdown handler
        atexit.register(self.shutdown)

    def set_flush_callback(
        self,
        callback: Callable[[List[Observation]], None],
    ) -> None:
        """Set the callback for flushing observations."""
        self._flush_callback = callback

    def enqueue(self, observation: Observation) -> bool:
        """
        Non-blocking enqueue of an observation.

        Args:
            observation: Observation to enqueue

        Returns:
            True if enqueued, False if dropped (queue full or circuit open)
        """
        config = get_config()

        # Check if observability is disabled
        if not config.enabled:
            return False

        # Check circuit breaker
        if config.circuit_breaker_enabled and self._circuit_breaker.is_open:
            with self._stats_lock:
                self._dropped += 1
            return False

        try:
            self._queue.put_nowait(observation)
            with self._stats_lock:
                self._enqueued += 1
            return True
        except Full:
            with self._stats_lock:
                self._dropped += 1
            self._circuit_breaker.record_failure()
            logger.warning("Observation queue full, dropping observation")
            return False

    def start_worker(self) -> None:
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="fasteval-observe-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.debug("Observation queue worker started")

    def _worker_loop(self) -> None:
        """Background worker that batches and flushes observations."""
        config = get_config()

        while not self._shutdown_event.is_set():
            try:
                batch = self._dequeue_batch(
                    max_size=config.batch_size,
                    timeout=config.flush_interval_seconds,
                )
                if batch:
                    self._flush_batch(batch)
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                self._circuit_breaker.record_failure()

        # Final flush on shutdown
        self._drain()

    def _dequeue_batch(
        self,
        max_size: int,
        timeout: float,
    ) -> List[Observation]:
        """
        Dequeue a batch of observations.

        Args:
            max_size: Maximum batch size
            timeout: Timeout in seconds

        Returns:
            List of observations (may be empty)
        """
        batch: List[Observation] = []
        deadline = time.time() + timeout

        while len(batch) < max_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                observation = self._queue.get(timeout=min(remaining, 0.1))
                batch.append(observation)
                self._queue.task_done()
            except Empty:
                # Check if we should continue waiting
                if time.time() >= deadline:
                    break

        return batch

    def _flush_batch(self, batch: List[Observation]) -> None:
        """Flush a batch of observations to the callback."""
        if not batch or not self._flush_callback:
            return

        try:
            self._flush_callback(batch)
            with self._stats_lock:
                self._flushed += len(batch)
            self._circuit_breaker.record_success()
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            self._circuit_breaker.record_failure()

    def _drain(self) -> None:
        """Drain remaining observations on shutdown."""
        batch: List[Observation] = []
        while True:
            try:
                observation = self._queue.get_nowait()
                batch.append(observation)
                self._queue.task_done()
            except Empty:
                break

        if batch:
            self._flush_batch(batch)
            logger.info(f"Drained {len(batch)} observations on shutdown")

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Graceful shutdown with observation drain.

        Args:
            timeout: Maximum time to wait for drain
        """
        if self._worker_thread is None:
            return

        logger.info("Shutting down observation queue...")
        self._shutdown_event.set()

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        self._worker_thread = None
        logger.info(
            f"Observation queue shutdown complete. "
            f"Enqueued: {self._enqueued}, Flushed: {self._flushed}, "
            f"Dropped: {self._dropped}"
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._stats_lock:
            return {
                "enqueued": self._enqueued,
                "flushed": self._flushed,
                "dropped": self._dropped,
                "queue_size": self._queue.qsize(),
                "circuit_breaker_open": self._circuit_breaker.is_open,
            }

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None


# Global queue instance
def get_observation_queue() -> ObservationQueue:
    """Get the global observation queue instance."""
    return ObservationQueue()
