"""Tests for fasteval.metrics.base."""

import pytest

from fasteval.metrics.base import Metric
from fasteval.models.evaluation import EvalInput, MetricResult


class ConcreteMetric(Metric):
    """Concrete implementation for testing."""

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=0.8,
            passed=self._determine_pass(0.8),
            threshold=self.threshold,
        )


class TestMetric:
    def test_init(self):
        m = ConcreteMetric(name="test", threshold=0.7, weight=2.0)
        assert m.name == "test"
        assert m.threshold == 0.7
        assert m.weight == 2.0

    def test_determine_pass_above(self):
        m = ConcreteMetric(name="test", threshold=0.5)
        assert m._determine_pass(0.6) is True

    def test_determine_pass_equal(self):
        m = ConcreteMetric(name="test", threshold=0.5)
        assert m._determine_pass(0.5) is True

    def test_determine_pass_below(self):
        m = ConcreteMetric(name="test", threshold=0.5)
        assert m._determine_pass(0.4) is False

    def test_repr(self):
        m = ConcreteMetric(name="test", threshold=0.7)
        assert repr(m) == "ConcreteMetric(name='test', threshold=0.7)"

    @pytest.mark.asyncio
    async def test_evaluate(self):
        m = ConcreteMetric(name="test", threshold=0.5)
        result = await m.evaluate(EvalInput(actual_output="hello"))
        assert result.score == 0.8
        assert result.passed is True

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Metric(name="test")  # type: ignore[abstract]
