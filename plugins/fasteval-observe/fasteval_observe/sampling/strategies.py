"""Built-in sampling strategies for fasteval-observe."""

import random
import threading
import time
from collections import deque
from typing import Any, Dict, List, Literal, Optional

from fasteval_observe.sampling.base import BaseSamplingStrategy


class NoSamplingStrategy(BaseSamplingStrategy):
    """
    Sample every invocation (no sampling).

    Use for development, debugging, or critical paths where
    every call must be monitored.

    Example:
        @observe(sampling=NoSamplingStrategy())
        async def critical_agent(query: str) -> str:
            return await process(query)
    """

    def should_sample(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
    ) -> bool:
        """Always returns True - sample everything."""
        return True


class FixedRateSamplingStrategy(BaseSamplingStrategy):
    """
    Sample at a fixed rate (e.g., 1 in N calls).

    Uses a counter-based approach for deterministic sampling.

    Args:
        rate: Sampling rate between 0.0 and 1.0 (e.g., 0.05 for 5%)

    Example:
        # Sample 5% of calls
        @observe(sampling=FixedRateSamplingStrategy(rate=0.05))
        async def my_agent(query: str) -> str:
            return await process(query)
    """

    def __init__(self, rate: float = 0.05):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Rate must be between 0.0 and 1.0, got {rate}")
        self.rate = rate
        self._counter = 0
        self._lock = threading.Lock()

        # Calculate interval (e.g., rate=0.05 means 1 in 20)
        self._interval = int(1 / rate) if rate > 0 else 0

    def should_sample(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
    ) -> bool:
        """Sample based on fixed rate using counter."""
        if self.rate <= 0:
            return False
        if self.rate >= 1.0:
            return True

        with self._lock:
            self._counter += 1
            return self._counter % self._interval == 0

    def reset(self) -> None:
        """Reset the counter."""
        with self._lock:
            self._counter = 0


class ProbabilisticSamplingStrategy(BaseSamplingStrategy):
    """
    Random probability-based sampling.

    Each call has an independent probability of being sampled.

    Args:
        probability: Probability of sampling each call (0.0-1.0)
        seed: Optional random seed for reproducibility

    Example:
        # 10% probability for each call
        @observe(sampling=ProbabilisticSamplingStrategy(probability=0.1))
        async def my_agent(query: str) -> str:
            return await process(query)
    """

    def __init__(self, probability: float = 0.05, seed: Optional[int] = None):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability must be between 0.0 and 1.0, got {probability}"
            )
        self.probability = probability
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def should_sample(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
    ) -> bool:
        """Sample based on random probability."""
        if self.probability <= 0:
            return False
        if self.probability >= 1.0:
            return True

        with self._lock:
            return self._rng.random() < self.probability


class AdaptiveSamplingStrategy(BaseSamplingStrategy):
    """
    Dynamically adjust sampling rate based on latency and errors.

    Increases sampling rate during incidents (high latency, errors)
    and reduces during normal operation.

    Args:
        base_rate: Base sampling rate for normal operation (0.0-1.0)
        error_rate: Sampling rate when errors occur (0.0-1.0)
        slow_threshold_ms: Latency threshold for "slow" calls
        slow_rate: Sampling rate for slow calls (0.0-1.0)
        window_size: Number of recent calls to consider
        cooldown_seconds: Time before reducing from elevated rates

    Example:
        @observe(
            sampling=AdaptiveSamplingStrategy(
                base_rate=0.01,        # 1% baseline
                error_rate=1.0,        # 100% on errors
                slow_threshold_ms=2000,
                slow_rate=0.5,         # 50% of slow calls
            )
        )
        async def my_agent(query: str) -> str:
            return await process(query)
    """

    def __init__(
        self,
        base_rate: float = 0.01,
        error_rate: float = 1.0,
        slow_threshold_ms: float = 2000.0,
        slow_rate: float = 0.5,
        window_size: int = 100,
        cooldown_seconds: float = 60.0,
    ):
        self.base_rate = base_rate
        self.error_rate = error_rate
        self.slow_threshold_ms = slow_threshold_ms
        self.slow_rate = slow_rate
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds

        self._lock = threading.Lock()
        self._recent_latencies: deque = deque(maxlen=window_size)
        self._recent_errors: deque = deque(maxlen=window_size)
        self._last_incident_time: Optional[float] = None
        self._current_rate = base_rate
        self._rng = random.Random()

    def should_sample(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
    ) -> bool:
        """Sample based on current adaptive rate."""
        with self._lock:
            rate = self._calculate_current_rate()
            if rate <= 0:
                return False
            if rate >= 1.0:
                return True
            return self._rng.random() < rate

    def on_completion(
        self,
        function_name: str,
        execution_time_ms: float,
        error: Optional[Exception],
        result: Any,
    ) -> None:
        """Track completion for adaptive rate adjustment."""
        with self._lock:
            self._recent_latencies.append(execution_time_ms)
            self._recent_errors.append(error is not None)

            # Mark incident time if slow or error
            if error is not None or execution_time_ms > self.slow_threshold_ms:
                self._last_incident_time = time.time()

    def _calculate_current_rate(self) -> float:
        """Calculate current sampling rate based on recent history."""
        now = time.time()

        # Check if we're in cooldown period after incident
        if self._last_incident_time:
            time_since_incident = now - self._last_incident_time
            if time_since_incident < self.cooldown_seconds:
                # Still in elevated rate period
                return max(self.base_rate, self.slow_rate)

        # Check recent error rate
        if self._recent_errors:
            error_count = sum(1 for e in self._recent_errors if e)
            error_rate = error_count / len(self._recent_errors)
            if error_rate > 0.1:  # >10% errors
                return self.error_rate

        # Check recent latencies
        if self._recent_latencies:
            slow_count = sum(
                1 for lat in self._recent_latencies if lat > self.slow_threshold_ms
            )
            slow_rate = slow_count / len(self._recent_latencies)
            if slow_rate > 0.1:  # >10% slow
                return self.slow_rate

        return self.base_rate

    def reset(self) -> None:
        """Reset adaptive state."""
        with self._lock:
            self._recent_latencies.clear()
            self._recent_errors.clear()
            self._last_incident_time = None
            self._current_rate = self.base_rate


class TokenBudgetSamplingStrategy(BaseSamplingStrategy):
    """
    Sample based on token cost budget.

    Tracks cumulative token usage and samples high-cost calls
    or when approaching budget limits.

    Args:
        budget_tokens_per_hour: Token budget per hour
        high_cost_threshold: Token count threshold for "high cost" calls
        high_cost_rate: Sampling rate for high-cost calls
        base_rate: Base sampling rate for normal calls

    Example:
        @observe(
            sampling=TokenBudgetSamplingStrategy(
                budget_tokens_per_hour=100000,
                high_cost_threshold=1000,
                high_cost_rate=0.5,
            )
        )
        async def my_agent(query: str) -> str:
            return await process(query)
    """

    def __init__(
        self,
        budget_tokens_per_hour: int = 100000,
        high_cost_threshold: int = 1000,
        high_cost_rate: float = 0.5,
        base_rate: float = 0.05,
    ):
        self.budget_tokens_per_hour = budget_tokens_per_hour
        self.high_cost_threshold = high_cost_threshold
        self.high_cost_rate = high_cost_rate
        self.base_rate = base_rate

        self._lock = threading.Lock()
        self._hourly_tokens: deque = deque()  # (timestamp, tokens)
        self._rng = random.Random()

    def should_sample(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
    ) -> bool:
        """Sample based on token budget."""
        # Check if this looks like a high-cost call from context
        estimated_tokens = context.get("estimated_tokens", 0)

        with self._lock:
            # Clean old entries (older than 1 hour)
            cutoff = time.time() - 3600
            while self._hourly_tokens and self._hourly_tokens[0][0] < cutoff:
                self._hourly_tokens.popleft()

            # Calculate current hourly usage
            current_usage = sum(tokens for _, tokens in self._hourly_tokens)
            usage_ratio = current_usage / self.budget_tokens_per_hour

            # Determine sampling rate
            if estimated_tokens > self.high_cost_threshold:
                rate = self.high_cost_rate
            elif usage_ratio > 0.9:
                # Near budget limit, sample more to monitor
                rate = min(1.0, self.base_rate * 5)
            elif usage_ratio > 0.7:
                rate = min(1.0, self.base_rate * 2)
            else:
                rate = self.base_rate

            if rate <= 0:
                return False
            if rate >= 1.0:
                return True
            return self._rng.random() < rate

    def on_completion(
        self,
        function_name: str,
        execution_time_ms: float,
        error: Optional[Exception],
        result: Any,
    ) -> None:
        """Track token usage from completion."""
        # Try to extract tokens from result if available
        tokens = 0
        if hasattr(result, "usage"):
            usage = result.usage
            if hasattr(usage, "total_tokens"):
                tokens = usage.total_tokens
            elif isinstance(usage, dict):
                tokens = usage.get("total_tokens", 0)

        if tokens > 0:
            with self._lock:
                self._hourly_tokens.append((time.time(), tokens))

    def reset(self) -> None:
        """Reset token tracking."""
        with self._lock:
            self._hourly_tokens.clear()


class ComposableSamplingStrategy(BaseSamplingStrategy):
    """
    Combine multiple sampling strategies with AND/OR logic.

    Args:
        strategies: List of strategies to combine
        mode: "any" (OR) or "all" (AND) logic

    Example:
        # Sample if error OR high latency OR 1% random
        @observe(
            sampling=ComposableSamplingStrategy(
                strategies=[
                    ProbabilisticSamplingStrategy(probability=0.01),
                    AdaptiveSamplingStrategy(base_rate=0.0, error_rate=1.0),
                ],
                mode="any"
            )
        )
        async def my_agent(query: str) -> str:
            return await process(query)
    """

    def __init__(
        self,
        strategies: List[BaseSamplingStrategy],
        mode: Literal["any", "all"] = "any",
    ):
        if not strategies:
            raise ValueError("At least one strategy required")
        self.strategies = strategies
        self.mode = mode

    def should_sample(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
    ) -> bool:
        """Sample based on combined strategies."""
        results = [
            s.should_sample(function_name, args, kwargs, context)
            for s in self.strategies
        ]

        if self.mode == "any":
            return any(results)
        else:  # mode == "all"
            return all(results)

    def on_completion(
        self,
        function_name: str,
        execution_time_ms: float,
        error: Optional[Exception],
        result: Any,
    ) -> None:
        """Forward completion to all strategies."""
        for strategy in self.strategies:
            strategy.on_completion(function_name, execution_time_ms, error, result)

    def reset(self) -> None:
        """Reset all strategies."""
        for strategy in self.strategies:
            strategy.reset()

    @property
    def name(self) -> str:
        """Return combined strategy name."""
        strategy_names = [s.name for s in self.strategies]
        return f"Composable({self.mode}:{','.join(strategy_names)})"
