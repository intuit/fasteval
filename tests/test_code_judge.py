"""Tests for the Code-as-Judge feature (fe.judge decorator)."""

import asyncio

import pytest

import fasteval as fe
from fasteval.core.decorators import fasteval_METRICS_ATTR
from fasteval.metrics.code_judge import CodeJudgeMetric
from fasteval.models.evaluation import EvalInput, MetricResult

# ---------------------------------------------------------------------------
# Scoring functions used by tests
# ---------------------------------------------------------------------------


def score_by_length(actual_output: str) -> float:
    words = len((actual_output or "").split())
    if words < 5:
        return 0.2
    if words > 200:
        return 0.3
    return 1.0


def score_with_reasoning(actual_output: str, expected_output: str) -> tuple:
    if actual_output == expected_output:
        return (1.0, "Exact match")
    return (0.3, f"Mismatch: got {actual_output!r}")


def score_returning_dict(actual_output: str) -> dict:
    has_greeting = "hello" in (actual_output or "").lower()
    return {
        "score": 1.0 if has_greeting else 0.0,
        "reasoning": "Greeting found" if has_greeting else "No greeting",
        "details": {"has_greeting": has_greeting},
    }


def score_returning_metric_result(actual_output: str) -> MetricResult:
    score = 1.0 if actual_output else 0.0
    return MetricResult(
        metric_name="custom_mr",
        score=score,
        passed=score >= 0.5,
        threshold=0.5,
        reasoning="Returned MetricResult directly",
    )


async def async_scorer(actual_output: str) -> float:
    return 1.0 if actual_output else 0.0


def score_with_context(actual_output: str, expected_output: str, input: str) -> float:
    if expected_output and expected_output.lower() in (actual_output or "").lower():
        return 1.0
    return 0.0


def score_full_eval_input(eval_input: EvalInput) -> float:
    if eval_input.actual_output and eval_input.expected_output:
        return 1.0 if eval_input.expected_output in eval_input.actual_output else 0.0
    return 0.0


def score_with_kwargs(actual_output: str, **kwargs) -> float:
    return 1.0 if actual_output else 0.0


# ---------------------------------------------------------------------------
# Tests: CodeJudgeMetric directly
# ---------------------------------------------------------------------------


class TestCodeJudgeMetricDirect:
    """Test CodeJudgeMetric without the decorator, via evaluate()."""

    def test_float_return(self):
        metric = CodeJudgeMetric(func=score_by_length, name="length", threshold=0.5)
        inp = EvalInput(actual_output="This is a decent length response for testing")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0
        assert result.passed is True
        assert result.metric_name == "length"

    def test_float_return_fail(self):
        metric = CodeJudgeMetric(func=score_by_length, name="length", threshold=0.5)
        inp = EvalInput(actual_output="Short")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 0.2
        assert result.passed is False

    def test_tuple_return(self):
        metric = CodeJudgeMetric(func=score_with_reasoning, name="match", threshold=0.5)
        inp = EvalInput(actual_output="hello", expected_output="hello")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0
        assert result.reasoning == "Exact match"

    def test_tuple_return_fail(self):
        metric = CodeJudgeMetric(func=score_with_reasoning, name="match", threshold=0.5)
        inp = EvalInput(actual_output="hello", expected_output="world")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 0.3
        assert "Mismatch" in result.reasoning

    def test_dict_return(self):
        metric = CodeJudgeMetric(
            func=score_returning_dict, name="greeting", threshold=0.5
        )
        inp = EvalInput(actual_output="Hello, how can I help?")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0
        assert result.details["has_greeting"] is True

    def test_dict_return_fail(self):
        metric = CodeJudgeMetric(
            func=score_returning_dict, name="greeting", threshold=0.5
        )
        inp = EvalInput(actual_output="Goodbye")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 0.0
        assert result.passed is False

    def test_metric_result_return(self):
        metric = CodeJudgeMetric(
            func=score_returning_metric_result, name="direct_mr", threshold=0.5
        )
        inp = EvalInput(actual_output="some text")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0
        assert result.metric_name == "custom_mr"

    def test_async_function(self):
        metric = CodeJudgeMetric(func=async_scorer, name="async", threshold=0.5)
        inp = EvalInput(actual_output="hello")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0

    def test_full_eval_input_param(self):
        metric = CodeJudgeMetric(func=score_full_eval_input, name="full", threshold=0.5)
        inp = EvalInput(actual_output="Paris is great", expected_output="Paris")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0

    def test_kwargs_param(self):
        metric = CodeJudgeMetric(func=score_with_kwargs, name="kw", threshold=0.5)
        inp = EvalInput(actual_output="hello", expected_output="world")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0

    def test_multi_field_signature(self):
        metric = CodeJudgeMetric(func=score_with_context, name="ctx", threshold=0.5)
        inp = EvalInput(
            actual_output="Paris is the capital",
            expected_output="Paris",
            input="What is the capital of France?",
        )
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0

    def test_unsupported_return_type(self):
        def bad_return(actual_output: str):
            return [1, 2, 3]

        metric = CodeJudgeMetric(func=bad_return, name="bad", threshold=0.5)
        inp = EvalInput(actual_output="hello")
        with pytest.raises(TypeError, match="unsupported type"):
            asyncio.run(metric.evaluate(inp))

    def test_int_return_coerced_to_float(self):
        def returns_int(actual_output: str) -> int:
            return 1

        metric = CodeJudgeMetric(func=returns_int, name="int_ret", threshold=0.5)
        inp = EvalInput(actual_output="hello")
        result = asyncio.run(metric.evaluate(inp))
        assert result.score == 1.0
        assert isinstance(result.score, float)

    def test_threshold_and_weight(self):
        metric = CodeJudgeMetric(
            func=score_by_length, name="len", threshold=0.9, weight=2.0
        )
        assert metric.threshold == 0.9
        assert metric.weight == 2.0

    def test_name_defaults(self):
        metric = CodeJudgeMetric(func=score_by_length, name="score_by_length")
        assert metric.name == "score_by_length"


# ---------------------------------------------------------------------------
# Tests: @fe.judge decorator wiring
# ---------------------------------------------------------------------------


class TestJudgeDecorator:
    """Test that @fe.judge attaches MetricConfig correctly."""

    def test_attaches_metric_config(self):
        @fe.judge(score_by_length, threshold=0.8)
        def my_test():
            pass

        metrics = getattr(my_test, fasteval_METRICS_ATTR, [])
        assert len(metrics) == 1
        assert metrics[0].metric_type == "custom"
        assert metrics[0].name == "score_by_length"
        assert metrics[0].threshold == 0.8
        assert isinstance(metrics[0].config["instance"], CodeJudgeMetric)

    def test_custom_name(self):
        @fe.judge(score_by_length, name="word_count", threshold=0.7)
        def my_test():
            pass

        metrics = getattr(my_test, fasteval_METRICS_ATTR, [])
        assert metrics[0].name == "word_count"

    def test_default_threshold(self):
        @fe.judge(score_by_length)
        def my_test():
            pass

        metrics = getattr(my_test, fasteval_METRICS_ATTR, [])
        assert metrics[0].threshold == 0.5

    def test_weight_passed_through(self):
        @fe.judge(score_by_length, weight=3.0)
        def my_test():
            pass

        metrics = getattr(my_test, fasteval_METRICS_ATTR, [])
        assert metrics[0].weight == 3.0

    def test_multiple_judges(self):
        @fe.judge(score_by_length, threshold=0.8)
        @fe.judge(score_returning_dict, threshold=0.6)
        def my_test():
            pass

        metrics = getattr(my_test, fasteval_METRICS_ATTR, [])
        assert len(metrics) == 2
        names = {m.name for m in metrics}
        assert names == {"score_by_length", "score_returning_dict"}

    def test_combined_with_builtin(self):
        @fe.judge(score_by_length, threshold=0.8)
        @fe.contains()
        def my_test():
            pass

        metrics = getattr(my_test, fasteval_METRICS_ATTR, [])
        assert len(metrics) == 2
        types = {m.metric_type for m in metrics}
        assert "custom" in types
        assert "contains" in types


# ---------------------------------------------------------------------------
# Tests: End-to-end with fe.score()
# ---------------------------------------------------------------------------


class TestJudgeEndToEnd:
    """Test @fe.judge with fe.score() for actual evaluation."""

    def test_passing_evaluation(self):
        @fe.judge(score_by_length, threshold=0.5)
        def _test():
            return fe.score("This is a decent length response for testing")

        result = _test()
        assert result.passed
        assert result.metric_results[0].score == 1.0

    def test_failing_evaluation(self):
        @fe.judge(score_by_length, threshold=0.5)
        def _test():
            return fe.score("Short")

        with pytest.raises(fe.EvaluationFailedError):
            _test()

    def test_with_expected_output(self):
        @fe.judge(score_with_reasoning, threshold=0.5)
        def _test():
            return fe.score("hello", "hello")

        result = _test()
        assert result.passed
        assert result.metric_results[0].reasoning == "Exact match"

    def test_combined_with_contains(self):
        @fe.judge(score_by_length, threshold=0.5)
        @fe.contains()
        def _test():
            return fe.score(
                "The capital of France is Paris, a beautiful city",
                "Paris",
            )

        result = _test()
        assert result.passed
        assert len(result.metric_results) == 2

    def test_dict_return_in_evaluation(self):
        @fe.judge(score_returning_dict, threshold=0.5)
        def _test():
            return fe.score("Hello, how are you?")

        result = _test()
        assert result.passed
        assert result.metric_results[0].details["has_greeting"] is True
