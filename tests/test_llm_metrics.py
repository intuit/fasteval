"""Tests for fasteval.metrics.llm (with mocked LLM client)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from fasteval.metrics.llm import (
    AnswerCorrectnessMetric,
    BaseLLMMetric,
    BiasMetric,
    CoherenceMetric,
    CompletenessMetric,
    ConcisenessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    CorrectnessMetric,
    CriteriaMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    HelpfulnessMetric,
    InstructionFollowingMetric,
    LLMEvalResponse,
    RelevanceMetric,
    ToxicityMetric,
)
from fasteval.models.evaluation import EvalInput


class MockLLMClient:
    """Mock LLM client that returns configurable responses."""

    def __init__(self, response_text=None):
        self.response_text = response_text or json.dumps(
            {"score": 0.85, "reasoning": "Mock evaluation"}
        )
        self.call_count = 0

    async def invoke(self, messages):
        self.call_count += 1
        return self.response_text


class FailingLLMClient:
    """Mock client that always raises."""

    async def invoke(self, messages):
        raise RuntimeError("LLM error")


# ── LLMEvalResponse ─────────────────────────────────────────────────────────


class TestLLMEvalResponse:
    def test_valid(self):
        resp = LLMEvalResponse(score=0.8, reasoning="Good")
        assert resp.score == 0.8
        assert resp.reasoning == "Good"

    def test_score_bounds(self):
        with pytest.raises(Exception):
            LLMEvalResponse(score=1.5, reasoning="Out of range")

    def test_default_reasoning(self):
        resp = LLMEvalResponse(score=0.5)
        assert resp.reasoning == ""


# ── BaseLLMMetric ────────────────────────────────────────────────────────────


class TestBaseLLMMetric:
    def test_get_client_explicit(self):
        client = MockLLMClient()
        metric = CorrectnessMetric(llm_client=client)
        assert metric._get_client() is client

    def test_get_client_model_override(self):
        metric = CorrectnessMetric(model="gpt-4o")
        client = metric._get_client()
        # Should create an OpenAI client
        from fasteval.providers.openai import OpenAIClient

        assert isinstance(client, OpenAIClient)

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        client = MockLLMClient()
        metric = CorrectnessMetric(llm_client=client, threshold=0.5)
        result = await metric.evaluate(
            EvalInput(
                actual_output="4",
                expected_output="4",
                input="What is 2+2?",
            )
        )
        assert result.score == 0.85
        assert result.passed is True
        assert result.reasoning == "Mock evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_retry_then_success(self):
        call_count = 0

        class RetryClient:
            async def invoke(self, messages):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "not json"
                return json.dumps({"score": 0.9, "reasoning": "OK"})

        metric = CorrectnessMetric(llm_client=RetryClient(), max_retries=3)
        result = await metric.evaluate(
            EvalInput(actual_output="answer", expected_output="answer")
        )
        assert result.score == 0.9
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_all_retries_fail(self):
        metric = CorrectnessMetric(llm_client=FailingLLMClient(), max_retries=2)
        result = await metric.evaluate(
            EvalInput(actual_output="answer", expected_output="answer")
        )
        assert result.score == 0.0
        assert result.passed is False
        assert "LLM error" in result.reasoning

    @pytest.mark.asyncio
    async def test_binary_scoring(self):
        client = MockLLMClient(json.dumps({"score": 0.7, "reasoning": "Partial"}))
        metric = CorrectnessMetric(
            llm_client=client, scoring_type="binary", threshold=0.5
        )
        result = await metric.evaluate(
            EvalInput(actual_output="answer", expected_output="answer")
        )
        assert result.score == 1.0  # 0.7 >= 0.5 → 1.0

    @pytest.mark.asyncio
    async def test_binary_scoring_below(self):
        client = MockLLMClient(json.dumps({"score": 0.3, "reasoning": "Low"}))
        metric = CorrectnessMetric(
            llm_client=client, scoring_type="binary", threshold=0.5
        )
        result = await metric.evaluate(
            EvalInput(actual_output="answer", expected_output="answer")
        )
        assert result.score == 0.0  # 0.3 < 0.5 → 0.0


# ── Specific Metric Prompt Tests ─────────────────────────────────────────────


class TestCorrectnessMetric:
    def test_default_name(self):
        metric = CorrectnessMetric(llm_client=MockLLMClient())
        assert metric.name == "correctness"

    def test_prompt_contains_inputs(self):
        metric = CorrectnessMetric(llm_client=MockLLMClient())
        prompt = metric.get_evaluation_prompt(
            EvalInput(
                actual_output="4",
                expected_output="4",
                input="What is 2+2?",
            )
        )
        assert "What is 2+2?" in prompt
        assert "Semantic Equivalence" in prompt


class TestHallucinationMetric:
    def test_default_threshold(self):
        metric = HallucinationMetric(llm_client=MockLLMClient())
        assert metric.threshold == 0.9

    def test_prompt_includes_context(self):
        metric = HallucinationMetric(llm_client=MockLLMClient())
        prompt = metric.get_evaluation_prompt(
            EvalInput(
                actual_output="The earth is flat",
                context=["The earth is round"],
            )
        )
        assert "The earth is round" in prompt


class TestRelevanceMetric:
    def test_default_name(self):
        metric = RelevanceMetric(llm_client=MockLLMClient())
        assert metric.name == "relevance"

    @pytest.mark.asyncio
    async def test_evaluation(self):
        client = MockLLMClient()
        metric = RelevanceMetric(llm_client=client, threshold=0.5)
        result = await metric.evaluate(
            EvalInput(actual_output="answer", input="question")
        )
        assert result.score == 0.85


class TestCriteriaMetric:
    def test_default_name(self):
        metric = CriteriaMetric(llm_client=MockLLMClient(), criteria="Be concise")
        assert metric.name == "criteria"

    def test_prompt_includes_criteria(self):
        metric = CriteriaMetric(
            llm_client=MockLLMClient(), criteria="Answer must be formal"
        )
        prompt = metric.get_evaluation_prompt(EvalInput(actual_output="yo what up"))
        assert "Answer must be formal" in prompt

    def test_with_evaluation_steps(self):
        metric = CriteriaMetric(
            llm_client=MockLLMClient(),
            criteria="test",
            evaluation_steps=["Step 1", "Step 2"],
        )
        prompt = metric.get_evaluation_prompt(EvalInput(actual_output="test"))
        assert "Step 1" in prompt


class TestToxicityMetric:
    def test_default_name(self):
        metric = ToxicityMetric(llm_client=MockLLMClient())
        assert metric.name == "toxicity"

    @pytest.mark.asyncio
    async def test_evaluation(self):
        client = MockLLMClient()
        metric = ToxicityMetric(llm_client=client, threshold=0.5)
        result = await metric.evaluate(
            EvalInput(actual_output="Hello, how can I help?")
        )
        assert result.passed is True


class TestBiasMetric:
    def test_default_name(self):
        metric = BiasMetric(llm_client=MockLLMClient())
        assert metric.name == "bias"


class TestQualityMetrics:
    def test_conciseness_name(self):
        metric = ConcisenessMetric(llm_client=MockLLMClient())
        assert metric.name == "conciseness"

    def test_coherence_name(self):
        metric = CoherenceMetric(llm_client=MockLLMClient())
        assert metric.name == "coherence"

    def test_completeness_name(self):
        metric = CompletenessMetric(llm_client=MockLLMClient())
        assert metric.name == "completeness"

    def test_helpfulness_name(self):
        metric = HelpfulnessMetric(llm_client=MockLLMClient())
        assert metric.name == "helpfulness"

    def test_instruction_following_name(self):
        metric = InstructionFollowingMetric(llm_client=MockLLMClient())
        assert metric.name == "instruction_following"

    def test_instruction_following_prompt(self):
        metric = InstructionFollowingMetric(
            llm_client=MockLLMClient(),
            instructions=["Always respond in French"],
        )
        prompt = metric.get_evaluation_prompt(EvalInput(actual_output="Bonjour"))
        assert "Always respond in French" in prompt


class TestRAGMetrics:
    def test_faithfulness_name(self):
        metric = FaithfulnessMetric(llm_client=MockLLMClient())
        assert metric.name == "faithfulness"

    def test_faithfulness_prompt_includes_context(self):
        metric = FaithfulnessMetric(llm_client=MockLLMClient())
        prompt = metric.get_evaluation_prompt(
            EvalInput(
                actual_output="answer",
                context=["doc1", "doc2"],
            )
        )
        assert "doc1" in prompt
        assert "doc2" in prompt

    def test_contextual_precision_name(self):
        metric = ContextualPrecisionMetric(llm_client=MockLLMClient())
        assert metric.name == "contextual_precision"

    def test_contextual_recall_name(self):
        metric = ContextualRecallMetric(llm_client=MockLLMClient())
        assert metric.name == "contextual_recall"

    def test_answer_correctness_name(self):
        metric = AnswerCorrectnessMetric(llm_client=MockLLMClient())
        assert metric.name == "answer_correctness"

    def test_contextual_precision_prompt(self):
        metric = ContextualPrecisionMetric(llm_client=MockLLMClient())
        prompt = metric.get_evaluation_prompt(
            EvalInput(
                actual_output="answer",
                input="question",
                retrieval_context=["doc1", "doc2"],
            )
        )
        assert "doc1" in prompt

    def test_contextual_recall_prompt(self):
        metric = ContextualRecallMetric(llm_client=MockLLMClient())
        prompt = metric.get_evaluation_prompt(
            EvalInput(
                actual_output="answer",
                expected_output="expected",
                context=["doc1"],
            )
        )
        assert "doc1" in prompt

    def test_answer_correctness_prompt(self):
        metric = AnswerCorrectnessMetric(llm_client=MockLLMClient())
        prompt = metric.get_evaluation_prompt(
            EvalInput(
                actual_output="answer",
                expected_output="expected",
                input="question",
            )
        )
        assert "question" in prompt


class TestLLMMetricEdgeCases:
    def test_get_client_default_provider(self):
        """Test _get_client falls back to default provider."""
        mock_client = MockLLMClient()
        from fasteval.providers.registry import (
            clear_default_provider,
            set_default_provider,
        )

        set_default_provider(mock_client)
        try:
            metric = CorrectnessMetric()
            client = metric._get_client()
            assert client is mock_client
        finally:
            clear_default_provider()

    def test_parse_response(self):
        metric = CorrectnessMetric(llm_client=MockLLMClient())
        result = metric._parse_response('{"score": 0.7, "reasoning": "OK"}')
        assert result.score == 0.7

    @pytest.mark.asyncio
    async def test_conciseness_evaluation(self):
        client = MockLLMClient()
        metric = ConcisenessMetric(llm_client=client, threshold=0.5)
        result = await metric.evaluate(
            EvalInput(actual_output="Short answer", input="question")
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_coherence_evaluation(self):
        client = MockLLMClient()
        metric = CoherenceMetric(llm_client=client, threshold=0.5)
        result = await metric.evaluate(EvalInput(actual_output="Coherent text"))
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_completeness_evaluation(self):
        client = MockLLMClient()
        metric = CompletenessMetric(llm_client=client, threshold=0.5)
        result = await metric.evaluate(
            EvalInput(actual_output="Complete answer", input="q")
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_helpfulness_evaluation(self):
        client = MockLLMClient()
        metric = HelpfulnessMetric(llm_client=client, threshold=0.5)
        result = await metric.evaluate(
            EvalInput(actual_output="Helpful response", input="q")
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_instruction_following_evaluation(self):
        client = MockLLMClient()
        metric = InstructionFollowingMetric(
            llm_client=client,
            instructions=["Be formal"],
            threshold=0.5,
        )
        result = await metric.evaluate(EvalInput(actual_output="Formal response"))
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_instruction_following_no_instructions(self):
        client = MockLLMClient()
        metric = InstructionFollowingMetric(llm_client=client, threshold=0.5)
        prompt = metric.get_evaluation_prompt(EvalInput(actual_output="test"))
        assert "No instructions" in prompt
