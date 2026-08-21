"""Tests for fasteval.metrics.conversation."""

import json

import pytest

from fasteval import metric
from fasteval.metrics.conversation import (
    ConsistencyMetric,
    ContextRetentionMetric,
    SkillQualityMetric,
    TopicDriftMetric,
)
from fasteval.models.evaluation import EvalInput


class MockLLMClient:
    def __init__(self, score=0.85):
        self.response = json.dumps(
            {"score": score, "reasoning": "Mock conversation eval"}
        )

    async def invoke(self, messages):
        return self.response


class TestContextRetentionMetric:
    def test_default_name(self):
        metric = ContextRetentionMetric(llm_client=MockLLMClient())
        assert metric.name == "context_retention"

    def test_prompt_includes_history(self):
        metric = ContextRetentionMetric(llm_client=MockLLMClient())
        prompt = metric.get_evaluation_prompt(
            EvalInput(
                actual_output="Yes, I remember",
                history=[
                    {"role": "user", "content": "My name is Alice"},
                    {"role": "assistant", "content": "Nice to meet you, Alice"},
                    {"role": "user", "content": "What is my name?"},
                ],
            )
        )
        assert "Alice" in prompt

    @pytest.mark.asyncio
    async def test_evaluation(self):
        metric = ContextRetentionMetric(llm_client=MockLLMClient(0.9), threshold=0.5)
        result = await metric.evaluate(
            EvalInput(
                actual_output="Your name is Alice",
                history=[
                    {"role": "user", "content": "My name is Alice"},
                    {"role": "assistant", "content": "Hello Alice"},
                ],
            )
        )
        assert result.score == 0.9
        assert result.passed is True


class TestConsistencyMetric:
    def test_default_name(self):
        metric = ConsistencyMetric(llm_client=MockLLMClient())
        assert metric.name == "consistency"

    def test_default_binary_scoring(self):
        metric = ConsistencyMetric(llm_client=MockLLMClient())
        assert metric.scoring_type == "binary"

    @pytest.mark.asyncio
    async def test_evaluation(self):
        # ConsistencyMetric uses binary scoring by default
        # Score 0.8 >= 0.5 → binary 1.0
        metric = ConsistencyMetric(llm_client=MockLLMClient(0.8), threshold=0.5)
        result = await metric.evaluate(
            EvalInput(
                actual_output="Paris is the capital",
                history=[
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital is Paris"},
                ],
            )
        )
        assert result.score == 1.0  # binary: 0.8 >= 0.5 → 1.0
        assert result.passed is True


class TestTopicDriftMetric:
    def test_default_name(self):
        metric = TopicDriftMetric(llm_client=MockLLMClient())
        assert metric.name == "topic_drift"

    @pytest.mark.asyncio
    async def test_evaluation(self):
        metric = TopicDriftMetric(llm_client=MockLLMClient(0.7), threshold=0.5)
        result = await metric.evaluate(
            EvalInput(
                actual_output="About cooking",
                history=[
                    {"role": "user", "content": "Let's discuss cooking"},
                    {"role": "assistant", "content": "Sure, what dish?"},
                ],
            )
        )
        assert result.score == 0.7
        assert result.passed is True


class TestSkillQualityMetric:
    def test_defaults(self):
        metric = SkillQualityMetric(skill="Sample skill", llm_client=MockLLMClient())
        assert metric.threshold == 0.8
        assert metric.name == "skill_quality"

    def test_get_evaluation_prompt(self):
        eval_input = EvalInput(
            history=[
                {"role": "user", "content": "Let's discuss cooking"},
                {"role": "assistant", "content": "Sure, what dish?"},
            ]
        )
        metric = SkillQualityMetric(skill="Sample skill", llm_client=MockLLMClient())
        metric.get_evaluation_prompt(eval_input)
