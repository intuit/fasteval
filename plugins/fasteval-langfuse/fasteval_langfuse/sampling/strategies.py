"""Built-in sampling strategies for trace evaluation."""

import random
from typing import Any, Dict, List, Optional

from fasteval_langfuse.sampling.base import BaseSamplingStrategy


class NoSamplingStrategy(BaseSamplingStrategy):
    """
    Return all traces (no sampling).

    Use when you want to evaluate all matching traces without sampling.

    Example:
        @langfuse_traces(
            project="prod",
            sampling=NoSamplingStrategy()
        )
        def test_all_traces(trace_id, input, output, context, metadata):
            fe.score(output, input=input)
    """

    def sample(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return all traces unchanged."""
        return traces


class RandomSamplingStrategy(BaseSamplingStrategy):
    """
    Random sampling of traces.

    Provides unbiased random sampling with optional seed for reproducibility.

    Args:
        sample_size: Number of traces to sample
        seed: Optional random seed for reproducible sampling

    Example:
        # Sample 200 random traces
        @langfuse_traces(
            project="prod",
            sampling=RandomSamplingStrategy(sample_size=200, seed=42)
        )
        def test_traces(trace_id, input, output, context, metadata):
            fe.score(output, input=input)
    """

    def __init__(self, sample_size: int, seed: Optional[int] = None):
        if sample_size <= 0:
            raise ValueError(f"sample_size must be > 0, got {sample_size}")

        self.sample_size = sample_size
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample random subset of traces."""
        if len(traces) <= self.sample_size:
            return traces

        return self._rng.sample(traces, self.sample_size)


class StratifiedSamplingStrategy(BaseSamplingStrategy):
    """
    Stratified sampling - even distribution across groups.

    Ensures representative sampling across different categories (tags, users, etc).

    Args:
        strata_key: Key to group by (supports dot notation for nested keys)
        samples_per_stratum: Number of samples from each group
        seed: Optional random seed for reproducibility

    Example:
        # Sample 30 traces from each user type
        @langfuse_traces(
            project="prod",
            sampling=StratifiedSamplingStrategy(
                strata_key="metadata.user_type",
                samples_per_stratum=30
            )
        )
        def test_across_segments(trace_id, input, output, context, metadata):
            fe.score(output, input=input)
    """

    def __init__(
        self,
        strata_key: str,
        samples_per_stratum: int,
        seed: Optional[int] = None,
    ):
        if samples_per_stratum <= 0:
            raise ValueError(
                f"samples_per_stratum must be > 0, got {samples_per_stratum}"
            )

        self.strata_key = strata_key
        self.samples_per_stratum = samples_per_stratum
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample evenly across strata."""
        # Group traces by strata
        strata: Dict[str, List[Dict[str, Any]]] = {}

        for trace in traces:
            key_value = self._get_nested_value(trace, self.strata_key)
            strata_name = str(key_value) if key_value is not None else "unknown"

            if strata_name not in strata:
                strata[strata_name] = []
            strata[strata_name].append(trace)

        # Sample from each stratum
        sampled = []
        for stratum_traces in strata.values():
            sample_size = min(self.samples_per_stratum, len(stratum_traces))
            sampled.extend(self._rng.sample(stratum_traces, sample_size))

        return sampled

    def _get_nested_value(self, obj: Dict[str, Any], key_path: str) -> Any:
        """Get nested value using dot notation (e.g., 'metadata.user_type')."""
        keys = key_path.split(".")
        value = obj

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value


class ScoreBasedSamplingStrategy(BaseSamplingStrategy):
    """
    Sample traces based on existing score values, oversampling low scores.

    Useful for debugging: focus evaluation budget on traces that already
    have poor user ratings or low quality scores.

    Args:
        score_name: Name of the score to filter on (e.g., "user_rating")
        low_score_threshold: Threshold below which scores are considered "low"
        low_score_rate: Sampling rate for low scores (0.0-1.0)
        high_score_rate: Sampling rate for high scores (0.0-1.0)
        seed: Optional random seed for reproducibility

    Example:
        # Evaluate 100% of low ratings, 5% of high ratings
        @langfuse_traces(
            project="prod",
            sampling=ScoreBasedSamplingStrategy(
                score_name="user_rating",
                low_score_threshold=3.0,
                low_score_rate=1.0,
                high_score_rate=0.05
            )
        )
        def test_failures(trace_id, input, output, context, metadata):
            fe.score(output, input=input)
    """

    def __init__(
        self,
        score_name: str,
        low_score_threshold: float,
        low_score_rate: float = 1.0,
        high_score_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        if not 0.0 <= low_score_rate <= 1.0:
            raise ValueError(f"low_score_rate must be 0.0-1.0, got {low_score_rate}")
        if not 0.0 <= high_score_rate <= 1.0:
            raise ValueError(f"high_score_rate must be 0.0-1.0, got {high_score_rate}")

        self.score_name = score_name
        self.low_score_threshold = low_score_threshold
        self.low_score_rate = low_score_rate
        self.high_score_rate = high_score_rate
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sample traces based on their score values.

        Separates traces into low-score and high-score groups,
        then samples each group at different rates.
        """
        if not traces:
            return []

        low_score_traces = []
        high_score_traces = []
        no_score_traces = []

        # Partition traces by score
        for trace in traces:
            score_value = self._get_score(trace)

            if score_value is None:
                # Traces without this score - treat as high score (don't oversample)
                no_score_traces.append(trace)
            elif score_value <= self.low_score_threshold:
                low_score_traces.append(trace)
            else:
                high_score_traces.append(trace)

        # Sample each group
        sampled = []

        # Low scores - oversample
        if low_score_traces:
            low_sample_size = max(1, int(len(low_score_traces) * self.low_score_rate))
            low_sample_size = min(low_sample_size, len(low_score_traces))
            sampled.extend(self._rng.sample(low_score_traces, low_sample_size))

        # High scores - undersample
        if high_score_traces:
            high_sample_size = max(1, int(len(high_score_traces) * self.high_score_rate))
            high_sample_size = min(high_sample_size, len(high_score_traces))
            sampled.extend(self._rng.sample(high_score_traces, high_sample_size))

        # No score - same rate as high scores
        if no_score_traces:
            no_score_sample_size = max(
                1, int(len(no_score_traces) * self.high_score_rate)
            )
            no_score_sample_size = min(no_score_sample_size, len(no_score_traces))
            sampled.extend(self._rng.sample(no_score_traces, no_score_sample_size))

        return sampled

    def _get_score(self, trace: Dict[str, Any]) -> Optional[float]:
        """
        Extract score value from trace.

        Langfuse stores scores in trace.scores as a list of score objects.
        Each score has: name, value, timestamp, etc.
        """
        scores = trace.get("scores", [])

        for score in scores:
            if score.get("name") == self.score_name:
                return score.get("value")

        return None


class RecentFirstSamplingStrategy(BaseSamplingStrategy):
    """
    Prioritize recent traces with exponential decay.

    Useful for focusing on recent production issues while still
    including some historical context.

    Args:
        sample_size: Number of traces to sample
        decay_factor: Exponential decay factor (0.0-1.0), higher = more recent bias
        seed: Optional random seed

    Example:
        # Sample 500 traces, heavily favoring recent ones
        @langfuse_traces(
            project="prod",
            time_range="last_30d",
            sampling=RecentFirstSamplingStrategy(
                sample_size=500,
                decay_factor=0.8
            )
        )
        def test_recent(trace_id, input, output, context, metadata):
            fe.score(output, input=input)
    """

    def __init__(
        self,
        sample_size: int,
        decay_factor: float = 0.8,
        seed: Optional[int] = None,
    ):
        if sample_size <= 0:
            raise ValueError(f"sample_size must be > 0, got {sample_size}")
        if not 0.0 <= decay_factor <= 1.0:
            raise ValueError(f"decay_factor must be 0.0-1.0, got {decay_factor}")

        self.sample_size = sample_size
        self.decay_factor = decay_factor
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample with preference for recent traces."""
        if len(traces) <= self.sample_size:
            return traces

        # Assume traces are ordered by timestamp (most recent first)
        # Assign weights with exponential decay
        weights = []
        for i in range(len(traces)):
            weight = self.decay_factor**i
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Weighted random sampling
        sampled_indices = self._rng.choices(
            range(len(traces)), weights=probabilities, k=self.sample_size
        )

        return [traces[i] for i in sampled_indices]
