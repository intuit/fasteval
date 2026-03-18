"""Tests for fasteval.core.evaluator."""

import json

import pytest

from typing import Any

from fasteval.core.evaluator import (
    METRIC_REGISTRY,
    Evaluator,
    EvaluatorConfig,
    create_evaluator,
)
from fasteval.metrics.base import Metric
from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import EvalInput, MetricResult


class MockLLMClient:
    async def invoke(self, messages):
        return json.dumps({"score": 0.9, "reasoning": "Mock"})


# ── EvaluatorConfig ──────────────────────────────────────────────────────────


class TestEvaluatorConfig:
    def test_defaults(self):
        config = EvaluatorConfig()
        assert config.fail_fast is False
        assert config.parallel is True
        assert config.cache_enabled is True

    def test_custom(self):
        config = EvaluatorConfig(fail_fast=True, parallel=False)
        assert config.fail_fast is True
        assert config.parallel is False


# ── Evaluator._create_metric ─────────────────────────────────────────────────


class TestEvaluatorCreateMetric:
    def test_standard_deterministic_metric(self):
        evaluator = Evaluator()
        config = MetricConfig(metric_type="exact_match", name="em", threshold=1.0)
        metric = evaluator._create_metric(config)
        assert metric.name == "em"
        assert metric.threshold == 1.0

    def test_custom_metric_with_instance(self):
        class MyMetric(Metric):
            async def evaluate(self, eval_input):
                return MetricResult(
                    metric_name="custom", score=1.0, passed=True, threshold=0.5
                )

        instance = MyMetric(name="custom")
        evaluator = Evaluator()
        config = MetricConfig(
            metric_type="custom",
            name="custom",
            config={"instance": instance},
        )
        metric = evaluator._create_metric(config)
        assert metric is instance

    def test_unknown_metric_type(self):
        evaluator = Evaluator()
        config = MetricConfig(metric_type="nonexistent", name="bad")
        with pytest.raises(ValueError, match="Unknown metric type"):
            evaluator._create_metric(config)

    def test_json_metric_pydantic_model(self):
        from pydantic import BaseModel

        class User(BaseModel):
            name: str

        evaluator = Evaluator()
        config = MetricConfig(
            metric_type="json",
            name="json_check",
            config={"pydantic_model": User},
        )
        metric: Any = evaluator._create_metric(config)
        assert metric.model is User

    def test_criteria_metric(self):
        evaluator = Evaluator()
        config = MetricConfig(
            metric_type="criteria",
            name="criteria_check",
            config={"criteria": "Be formal"},
            llm_client=MockLLMClient(),
        )
        metric: Any = evaluator._create_metric(config)
        assert metric.criteria == "Be formal"

    def test_criteria_with_evaluation_steps(self):
        evaluator = Evaluator()
        config = MetricConfig(
            metric_type="geval",
            name="geval_check",
            config={
                "criteria": "test",
                "evaluation_steps": ["Step 1"],
            },
            llm_client=MockLLMClient(),
        )
        metric: Any = evaluator._create_metric(config)
        assert metric.evaluation_steps == ["Step 1"]

    def test_instruction_following_metric(self):
        evaluator = Evaluator()
        config = MetricConfig(
            metric_type="instruction_following",
            name="if_check",
            config={"instructions": "Be concise"},
            llm_client=MockLLMClient(),
        )
        metric: Any = evaluator._create_metric(config)
        assert metric.instructions == "Be concise"

    def test_with_llm_client(self):
        client = MockLLMClient()
        evaluator = Evaluator()
        config = MetricConfig(
            metric_type="correctness",
            name="corr",
            llm_client=client,
        )
        metric: Any = evaluator._create_metric(config)
        assert metric._llm_client is client

    def test_with_llm_config_model(self):
        evaluator = Evaluator()
        config = MetricConfig(
            metric_type="correctness",
            name="corr",
            llm_config={"model": "gpt-4o"},
        )
        metric: Any = evaluator._create_metric(config)
        assert metric._model_override == "gpt-4o"


# ── Evaluator.evaluate ───────────────────────────────────────────────────────


class TestEvaluatorEvaluate:
    @pytest.mark.asyncio
    async def test_single_metric_pass(self):
        evaluator = Evaluator()
        result = await evaluator.evaluate(
            eval_input=EvalInput(actual_output="hello", expected_output="hello"),
            metrics=[MetricConfig(metric_type="exact_match", name="em", threshold=1.0)],
        )
        assert result.passed is True
        assert result.aggregate_score == 1.0
        assert len(result.metric_results) == 1

    @pytest.mark.asyncio
    async def test_single_metric_fail(self):
        evaluator = Evaluator()
        result = await evaluator.evaluate(
            eval_input=EvalInput(actual_output="yes", expected_output="no"),
            metrics=[MetricConfig(metric_type="exact_match", name="em", threshold=1.0)],
        )
        assert result.passed is False
        assert result.aggregate_score == 0.0

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        evaluator = Evaluator(EvaluatorConfig(parallel=True))
        result = await evaluator.evaluate(
            eval_input=EvalInput(
                actual_output="hello world", expected_output="hello world"
            ),
            metrics=[
                MetricConfig(metric_type="exact_match", name="em1", threshold=1.0),
                MetricConfig(metric_type="contains", name="contains1", threshold=1.0),
            ],
        )
        assert result.passed is True
        assert len(result.metric_results) == 2

    @pytest.mark.asyncio
    async def test_sequential_fail_fast(self):
        evaluator = Evaluator(EvaluatorConfig(parallel=False, fail_fast=True))
        result = await evaluator.evaluate(
            eval_input=EvalInput(actual_output="yes", expected_output="no"),
            metrics=[
                MetricConfig(metric_type="exact_match", name="em1", threshold=1.0),
                MetricConfig(metric_type="exact_match", name="em2", threshold=1.0),
            ],
        )
        # fail_fast should stop after first failure
        assert result.passed is False
        assert len(result.metric_results) == 1

    @pytest.mark.asyncio
    async def test_weighted_aggregate(self):
        evaluator = Evaluator()
        result = await evaluator.evaluate(
            eval_input=EvalInput(actual_output="hello", expected_output="hello"),
            metrics=[
                MetricConfig(
                    metric_type="exact_match", name="em", threshold=1.0, weight=2.0
                ),
                MetricConfig(
                    metric_type="contains", name="ct", threshold=1.0, weight=1.0
                ),
            ],
        )
        # Both pass with score 1.0: (1.0*2 + 1.0*1) / 3 = 1.0
        assert result.aggregate_score == 1.0

    @pytest.mark.asyncio
    async def test_execution_time_recorded(self):
        evaluator = Evaluator()
        result = await evaluator.evaluate(
            eval_input=EvalInput(actual_output="a", expected_output="a"),
            metrics=[MetricConfig(metric_type="exact_match", name="em")],
        )
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_reference_id_preserved(self):
        evaluator = Evaluator()
        result = await evaluator.evaluate(
            eval_input=EvalInput(
                actual_output="a",
                expected_output="a",
                reference_id="ref-123",
            ),
            metrics=[MetricConfig(metric_type="exact_match", name="em")],
        )
        assert result.reference_id == "ref-123"


# ── Evaluator._evaluate_metric ───────────────────────────────────────────────


class TestEvaluatorEvaluateMetric:
    @pytest.mark.asyncio
    async def test_error_handling(self):
        class BrokenMetric(Metric):
            async def evaluate(self, eval_input):
                raise RuntimeError("metric broke")

        evaluator = Evaluator()
        metric = BrokenMetric(name="broken", threshold=0.5)
        result = await evaluator._evaluate_metric(
            metric, EvalInput(actual_output="test")
        )
        assert result.score == 0.0
        assert result.passed is False
        assert result.reasoning is not None and "metric broke" in result.reasoning


# ── Evaluator.evaluate_batch ─────────────────────────────────────────────────


class TestEvaluatorBatch:
    @pytest.mark.asyncio
    async def test_batch(self):
        evaluator = Evaluator()
        inputs = [
            EvalInput(actual_output="a", expected_output="a"),
            EvalInput(actual_output="b", expected_output="b"),
        ]
        results = await evaluator.evaluate_batch(
            inputs,
            [MetricConfig(metric_type="exact_match", name="em")],
        )
        assert len(results) == 2
        assert all(r.passed for r in results)


# ── create_evaluator ─────────────────────────────────────────────────────────


class TestCreateEvaluator:
    def test_factory(self):
        evaluator = create_evaluator(fail_fast=True, parallel=False)
        assert evaluator.config.fail_fast is True
        assert evaluator.config.parallel is False


# ── METRIC_REGISTRY ──────────────────────────────────────────────────────────


class TestMetricRegistry:
    def test_core_metrics_registered(self):
        expected = [
            "correctness",
            "hallucination",
            "relevance",
            "criteria",
            "geval",
            "toxicity",
            "bias",
            "conciseness",
            "coherence",
            "completeness",
            "helpfulness",
            "instruction_following",
            "faithfulness",
            "contextual_precision",
            "contextual_recall",
            "answer_correctness",
            "rouge",
            "exact_match",
            "contains",
            "json",
            "regex",
            "tool_call_accuracy",
            "tool_sequence",
            "tool_args_match",
            "context_retention",
            "consistency",
            "topic_drift",
        ]
        for name in expected:
            assert name in METRIC_REGISTRY, f"{name} not in METRIC_REGISTRY"
