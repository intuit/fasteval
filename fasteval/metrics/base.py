"""Base metric interface."""

from abc import ABC, abstractmethod
from typing import Optional

from fasteval.models.evaluation import EvalInput, MetricResult


class Metric(ABC):
    """
    Base class for all evaluation metrics.

    Subclasses must implement the `evaluate` method.

    Example:
        class MyMetric(Metric):
            async def evaluate(self, eval_input: EvalInput) -> MetricResult:
                # Your evaluation logic
                score = compute_score(eval_input)
                return MetricResult(
                    metric_name=self.name,
                    score=score,
                    passed=self._determine_pass(score),
                    threshold=self.threshold,
                )
    """

    def __init__(
        self,
        name: str,
        threshold: float = 0.5,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize metric.

        Args:
            name: Unique name for this metric instance
            threshold: Minimum score to pass (0.0 to 1.0)
            weight: Weight for aggregation (default 1.0)
        """
        self.name = name
        self.threshold = threshold
        self.weight = weight

    @abstractmethod
    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """
        Evaluate the input and return a MetricResult.

        Args:
            eval_input: The evaluation input containing actual/expected output

        Returns:
            MetricResult with score, pass/fail status, and optional reasoning
        """
        ...

    def _determine_pass(self, score: float) -> bool:
        """Determine if score meets threshold."""
        return score >= self.threshold

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, threshold={self.threshold})"
        )
