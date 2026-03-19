"""Tests for fasteval.core.decorators."""

import csv
import os
import tempfile

import pytest
from pydantic import BaseModel

# Import the public decorator functions
import fasteval.core.decorators as dec
from fasteval.core.decorators import (
    _attach_metric,
    _metric_decorator_factory,
    fasteval_HUMAN_REVIEW_ATTR,
    fasteval_METRICS_ATTR,
)
from fasteval.models.config import MetricConfig

# ── _attach_metric ───────────────────────────────────────────────────────────


class TestAttachMetric:
    def test_creates_attribute(self):
        def my_func():
            pass

        config = MetricConfig(metric_type="test", name="test")
        _attach_metric(my_func, config)
        assert hasattr(my_func, fasteval_METRICS_ATTR)
        assert len(getattr(my_func, fasteval_METRICS_ATTR)) == 1

    def test_appends_to_existing(self):
        def my_func():
            pass

        config1 = MetricConfig(metric_type="t1", name="n1")
        config2 = MetricConfig(metric_type="t2", name="n2")
        _attach_metric(my_func, config1)
        _attach_metric(my_func, config2)
        assert len(getattr(my_func, fasteval_METRICS_ATTR)) == 2


# ── _metric_decorator_factory ────────────────────────────────────────────────


class TestMetricDecoratorFactory:
    def test_creates_working_decorator(self):
        decorator_fn = _metric_decorator_factory("test_type", "test_name")

        @decorator_fn()
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert len(configs) == 1
        assert configs[0].metric_type == "test_type"
        assert configs[0].name == "test_name"

    def test_threshold_override(self):
        decorator_fn = _metric_decorator_factory("test", "test", default_threshold=0.5)

        @decorator_fn(threshold=0.9)
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].threshold == 0.9

    def test_weight_override(self):
        decorator_fn = _metric_decorator_factory("test", "test")

        @decorator_fn(weight=2.5)
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].weight == 2.5

    def test_name_override(self):
        decorator_fn = _metric_decorator_factory("test", "default_name")

        @decorator_fn(name="custom_name")
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].name == "custom_name"

    def test_model_override(self):
        decorator_fn = _metric_decorator_factory("test", "test")

        @decorator_fn(model="gpt-4o")
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].llm_config == {"model": "gpt-4o"}

    def test_llm_client_passthrough(self):
        class FakeClient:
            async def invoke(self, messages):
                return ""

        client = FakeClient()
        decorator_fn = _metric_decorator_factory("test", "test")

        @decorator_fn(llm_client=client)
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].llm_client is client


# ── All metric decorators ────────────────────────────────────────────────────


class TestAllMetricDecorators:
    """Test that each public decorator attaches the correct metric_type."""

    # Simple decorators that take no required positional args
    _simple_decorators = {
        "correctness": dec.correctness,
        "hallucination": dec.hallucination,
        "relevance": dec.relevance,
        "toxicity": dec.toxicity,
        "bias": dec.bias,
        "conciseness": dec.conciseness,
        "coherence": dec.coherence,
        "completeness": dec.completeness,
        "helpfulness": dec.helpfulness,
        "faithfulness": dec.faithfulness,
        "contextual_precision": dec.contextual_precision,
        "contextual_recall": dec.contextual_recall,
        "answer_correctness": dec.answer_correctness,
        "rouge": dec.rouge,
        "exact_match": dec.exact_match,
        "contains": dec.contains,
        "tool_call_accuracy": dec.tool_call_accuracy,
        "tool_sequence": dec.tool_sequence,
        "tool_args_match": dec.tool_args_match,
        "context_retention": dec.context_retention,
        "consistency": dec.consistency,
        "topic_drift": dec.topic_drift,
    }

    @pytest.mark.parametrize(
        "metric_type,decorator_fn", list(_simple_decorators.items())
    )
    def test_simple_decorator(self, metric_type, decorator_fn):
        @decorator_fn()
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert len(configs) == 1
        assert configs[0].metric_type == metric_type

    def test_regex_decorator(self):
        @dec.regex(pattern=r"\d+")
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].metric_type == "regex"

    def test_criteria_decorator(self):
        @dec.criteria("Is the response helpful?")
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].metric_type == "criteria"

    def test_geval_decorator(self):
        # geval is an alias for criteria
        @dec.geval(criteria="Is the response helpful?")
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].metric_type == "criteria"

    def test_instruction_following_decorator(self):
        @dec.instruction_following(instructions=["Be concise", "Use examples"])
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].metric_type == "instruction_following"

    def test_json_decorator(self):
        class User(BaseModel):
            name: str

        @dec.json(model=User)
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].metric_type == "json"


# ── Decorator stacking ───────────────────────────────────────────────────────


class TestDecoratorStacking:
    def test_multiple_decorators(self):
        @dec.correctness()
        @dec.relevance()
        @dec.contains()
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert len(configs) == 3
        types = [c.metric_type for c in configs]
        assert "correctness" in types
        assert "relevance" in types
        assert "contains" in types


# ── Data decorators ──────────────────────────────────────────────────────────


class TestCsvDecorator:
    def test_csv_decorator(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["input", "expected"])
            writer.writerow(["q1", "a1"])
            writer.writerow(["q2", "a2"])
            csv_path = f.name

        try:

            @dec.csv(csv_path)
            def my_func(input, expected):
                pass

            assert callable(my_func)
            assert hasattr(my_func, dec.fasteval_DATA_ATTR)
        finally:
            os.unlink(csv_path)


# ── Human review decorator ───────────────────────────────────────────────────


class TestHumanReviewDecorator:
    def test_attaches_config(self):
        @dec.human_review(prompt="Review this", required=True, threshold=0.8)
        def my_func():
            pass

        assert hasattr(my_func, fasteval_HUMAN_REVIEW_ATTR)
        config = getattr(my_func, fasteval_HUMAN_REVIEW_ATTR)
        assert config["prompt"] == "Review this"
        assert config["required"] is True
        assert config["threshold"] == 0.8


# ── Stack decorator ──────────────────────────────────────────────────────────


class TestConversationDecorator:
    def test_sync_conversation(self):
        results = []

        @dec.conversation(
            [
                {"query": "Hello", "expected": "Hi"},
                {"query": "Bye", "expected": "Goodbye"},
            ]
        )
        def my_func(query, expected, history):
            results.append({"query": query, "expected": expected, "history": history})
            return None

        my_func()  # type: ignore[call-arg]
        assert len(results) == 2
        assert results[0]["query"] == "Hello"
        assert results[0]["history"] == []
        assert results[1]["query"] == "Bye"

    @pytest.mark.asyncio
    async def test_async_conversation(self):
        results = []

        @dec.conversation(
            [
                {"query": "Hello"},
                {"query": "Bye"},
            ]
        )
        async def my_func(query, expected, history):
            results.append({"query": query, "history": history})

        await my_func()  # type: ignore[call-arg]
        assert len(results) == 2

    def test_conversation_preserves_metrics(self):
        @dec.correctness()
        @dec.conversation([{"query": "Hi"}])
        def my_func(query, expected, history):
            pass

        assert hasattr(my_func, fasteval_METRICS_ATTR)


class TestCriteriaWithEvaluationSteps:
    def test_criteria_with_steps(self):
        @dec.criteria("Be formal", evaluation_steps=["Step1", "Step2"])
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert configs[0].config["criteria"] == "Be formal"
        assert configs[0].config["evaluation_steps"] == ["Step1", "Step2"]


class TestStackDecorator:
    def test_stack_combines_metrics(self):
        # @fe.stack() goes at the TOP, captures decorators below it
        @dec.stack()
        @dec.correctness()
        @dec.relevance()
        def my_stack():
            pass

        # my_stack is now a decorator itself
        @my_stack
        def my_func():
            pass

        configs = getattr(my_func, fasteval_METRICS_ATTR)
        assert len(configs) == 2
        types = [c.metric_type for c in configs]
        assert "correctness" in types
        assert "relevance" in types
