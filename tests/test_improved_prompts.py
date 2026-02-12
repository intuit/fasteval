"""
Tests for the improved evaluation prompts.

These tests verify that the state-of-the-art prompts:
1. Follow academic best practices (G-Eval, Ragas, DeepEval methodologies)
2. Include proper chain-of-thought reasoning steps
3. Have clear scoring rubrics with anchor points
4. Support claim-level verification where applicable
5. Produce consistent and interpretable results

Test Categories:
- Prompt Structure Tests: Verify prompts contain required elements
- Functionality Tests: Verify metrics work end-to-end
- Benchmark Tests: Compare against known good/bad examples
"""

import re
from typing import Any, Dict, List

import pytest

from fasteval.metrics.conversation import (
    ConsistencyMetric,
    ContextRetentionMetric,
    TopicDriftMetric,
)
from fasteval.metrics.llm import (
    AnswerCorrectnessMetric,
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
    RelevanceMetric,
    ToxicityMetric,
)
from fasteval.models.evaluation import EvalInput

# =============================================================================
# PROMPT STRUCTURE TESTS
# =============================================================================


class TestPromptStructure:
    """Test that all prompts follow the improved structure with required elements."""

    @pytest.fixture
    def all_metrics(self) -> List[Any]:
        """Return all LLM-based metric instances."""
        return [
            CorrectnessMetric(),
            HallucinationMetric(),
            RelevanceMetric(),
            CriteriaMetric(criteria="Test criteria"),
            ToxicityMetric(),
            BiasMetric(),
            FaithfulnessMetric(),
            ContextualPrecisionMetric(),
            ContextualRecallMetric(),
            AnswerCorrectnessMetric(),
            ConcisenessMetric(),
            CoherenceMetric(),
            CompletenessMetric(),
            HelpfulnessMetric(),
            InstructionFollowingMetric(instructions=["Be concise"]),
            ContextRetentionMetric(),
            ConsistencyMetric(),
            TopicDriftMetric(),
        ]

    @pytest.fixture
    def sample_eval_input(self) -> EvalInput:
        """Create a sample eval input for testing."""
        return EvalInput(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            context=["France is a country in Europe. Its capital is Paris."],
            history=[
                {"role": "user", "content": "Tell me about France"},
                {"role": "assistant", "content": "France is a beautiful country."},
            ],
        )

    def test_prompts_have_task_section(
        self, all_metrics: List[Any], sample_eval_input: EvalInput
    ) -> None:
        """Verify all prompts have a clear TASK section."""
        for metric in all_metrics:
            prompt = metric.get_evaluation_prompt(sample_eval_input)
            assert (
                "## TASK" in prompt or "TASK" in prompt.upper()
            ), f"{metric.name} prompt missing TASK section"

    def test_prompts_have_scoring_rubric(
        self, all_metrics: List[Any], sample_eval_input: EvalInput
    ) -> None:
        """Verify all prompts have a scoring rubric with anchor points."""
        for metric in all_metrics:
            prompt = metric.get_evaluation_prompt(sample_eval_input)
            has_rubric = (
                "## SCORING RUBRIC" in prompt
                or "SCORING RUBRIC" in prompt
                or "## SCORING" in prompt
                or "Score 0.0" in prompt
                or "Score 1.0" in prompt
                or "Score 0" in prompt  # Binary metrics use 0/1
                or "Score 1" in prompt
                or "**1.0**" in prompt
                or "**0.0**" in prompt
                or "**1**:" in prompt  # Binary format
                or "**0**:" in prompt
            )
            assert has_rubric, f"{metric.name} prompt missing scoring rubric"

    def test_prompts_have_json_output_format(
        self, all_metrics: List[Any], sample_eval_input: EvalInput
    ) -> None:
        """Verify all prompts specify JSON output format."""
        for metric in all_metrics:
            prompt = metric.get_evaluation_prompt(sample_eval_input)
            has_json_spec = (
                "JSON" in prompt
                or "json" in prompt.lower()
                or '{"score"' in prompt
                or '{"score"' in prompt
            )
            assert has_json_spec, f"{metric.name} prompt missing JSON output format"

    def test_prompts_have_evaluation_dimensions(
        self, all_metrics: List[Any], sample_eval_input: EvalInput
    ) -> None:
        """Verify prompts have clear evaluation criteria/dimensions."""
        for metric in all_metrics:
            prompt = metric.get_evaluation_prompt(sample_eval_input)
            # Check for numbered lists or dimension headers
            has_dimensions = (
                re.search(r"###\s+\d+\.", prompt) is not None  # ### 1.
                or re.search(r"\d+\.\s+\*\*", prompt) is not None  # 1. **
                or re.search(r"### Step \d+", prompt) is not None  # ### Step 1
                or re.search(r"Step \d+:", prompt) is not None  # Step 1:
                or "## EVALUATION" in prompt
                or "## CONTEXT" in prompt  # For metrics with contextual evaluation
                or "Dimension" in prompt
                or "Consider:" in prompt
                or "Instructions:" in prompt
                or "Relevance Criteria" in prompt  # Specific to precision
            )
            assert has_dimensions, f"{metric.name} prompt missing evaluation dimensions"

    def test_prompts_have_evaluation_steps(
        self, all_metrics: List[Any], sample_eval_input: EvalInput
    ) -> None:
        """Verify prompts include chain-of-thought evaluation steps."""
        # These metrics should definitely have evaluation steps
        step_required_metrics = [
            "correctness",
            "hallucination",
            "faithfulness",
            "contextual_precision",
            "contextual_recall",
            "answer_correctness",
        ]

        for metric in all_metrics:
            if metric.name in step_required_metrics:
                prompt = metric.get_evaluation_prompt(sample_eval_input)
                has_steps = (
                    "Step 1" in prompt
                    or "## EVALUATION STEPS" in prompt
                    or "EVALUATION METHOD" in prompt
                )
                assert (
                    has_steps
                ), f"{metric.name} prompt missing chain-of-thought evaluation steps"


# =============================================================================
# CORE METRICS FUNCTIONALITY TESTS
# =============================================================================


class TestCorrectnessMetric:
    """Test the improved correctness metric."""

    def test_prompt_includes_semantic_equivalence(self) -> None:
        """Verify prompt checks for semantic equivalence."""
        metric = CorrectnessMetric()
        eval_input = EvalInput(
            input="What is 2+2?",
            actual_output="Four",
            expected_output="4",
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Semantic Equivalence" in prompt
        assert "Factual Accuracy" in prompt
        assert "Information Completeness" in prompt

    def test_prompt_has_geval_methodology(self) -> None:
        """Verify prompt follows G-Eval methodology."""
        metric = CorrectnessMetric()
        eval_input = EvalInput(
            input="Test", actual_output="Test", expected_output="Test"
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        # G-Eval requires chain-of-thought steps
        assert "Step 1" in prompt
        assert "Step 2" in prompt


class TestHallucinationMetric:
    """Test the improved hallucination metric."""

    def test_prompt_includes_claim_extraction(self) -> None:
        """Verify prompt includes Ragas-style claim extraction."""
        metric = HallucinationMetric()
        eval_input = EvalInput(
            input="What is the capital?",
            actual_output="Paris is the capital",
            context=["France's capital is Paris."],
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Claim Extraction" in prompt or "claim" in prompt.lower()
        assert "SUPPORTED" in prompt or "supported" in prompt.lower()

    def test_prompt_includes_nli_verification(self) -> None:
        """Verify prompt includes NLI-style verification."""
        metric = HallucinationMetric()
        eval_input = EvalInput(input="Test", actual_output="Test", context=["Context"])
        prompt = metric.get_evaluation_prompt(eval_input)

        # Should have entailment-style verdicts
        has_nli = any(
            term in prompt.upper()
            for term in ["SUPPORTED", "CONTRADICTED", "NOT_VERIFIABLE", "ENTAIL"]
        )
        assert has_nli, "Hallucination prompt missing NLI-style verification"


class TestFaithfulnessMetric:
    """Test the improved faithfulness metric."""

    def test_prompt_follows_ragas_methodology(self) -> None:
        """Verify prompt follows Ragas faithfulness methodology."""
        metric = FaithfulnessMetric()
        eval_input = EvalInput(
            input="Query",
            actual_output="Response",
            context=["Context document"],
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        # Ragas-style should have statement extraction
        assert "Statement" in prompt or "statement" in prompt.lower()
        assert "SUPPORTED" in prompt or "supported" in prompt.lower()

    def test_prompt_includes_scoring_formula(self) -> None:
        """Verify prompt includes score calculation formula."""
        metric = FaithfulnessMetric()
        eval_input = EvalInput(
            input="Query", actual_output="Response", context=["Context"]
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        # Should explain how score is calculated
        assert "Score" in prompt and ("=" in prompt or "calculate" in prompt.lower())


class TestToxicityMetric:
    """Test the improved toxicity metric."""

    def test_prompt_includes_all_toxicity_categories(self) -> None:
        """Verify prompt checks all major toxicity categories."""
        metric = ToxicityMetric()
        eval_input = EvalInput(input="Test", actual_output="Test response")
        prompt = metric.get_evaluation_prompt(eval_input)

        categories = [
            "Hate",
            "Violence",
            "Sexual",
            "Harassment",
            "Self-Harm",
        ]

        for category in categories:
            assert (
                category in prompt or category.lower() in prompt.lower()
            ), f"Toxicity prompt missing category: {category}"

    def test_prompt_includes_severity_levels(self) -> None:
        """Verify prompt includes severity assessment."""
        metric = ToxicityMetric()
        eval_input = EvalInput(input="Test", actual_output="Test")
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Severity" in prompt or "severity" in prompt.lower()


class TestBiasMetric:
    """Test the improved bias metric."""

    def test_prompt_includes_all_bias_categories(self) -> None:
        """Verify prompt checks all major bias categories."""
        metric = BiasMetric()
        eval_input = EvalInput(input="Test", actual_output="Test response")
        prompt = metric.get_evaluation_prompt(eval_input)

        categories = [
            "Gender",
            "Racial",
            "Age",
            "Socioeconomic",
            "Religious",
            "Political",
        ]

        for category in categories:
            assert (
                category in prompt or category.lower() in prompt.lower()
            ), f"Bias prompt missing category: {category}"

    def test_prompt_distinguishes_explicit_implicit_bias(self) -> None:
        """Verify prompt distinguishes between explicit and implicit bias."""
        metric = BiasMetric()
        eval_input = EvalInput(input="Test", actual_output="Test")
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Explicit" in prompt or "explicit" in prompt.lower()
        assert "Implicit" in prompt or "implicit" in prompt.lower()


# =============================================================================
# RAG METRICS TESTS
# =============================================================================


class TestContextualPrecisionMetric:
    """Test the improved contextual precision metric."""

    def test_prompt_evaluates_each_document(self) -> None:
        """Verify prompt evaluates relevance of each document."""
        metric = ContextualPrecisionMetric()
        eval_input = EvalInput(
            input="What is X?",
            actual_output="X is Y",
            retrieval_context=["Doc about X", "Doc about Z"],
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Document 1" in prompt or "each document" in prompt.lower()
        assert "RELEVANT" in prompt or "relevant" in prompt.lower()


class TestContextualRecallMetric:
    """Test the improved contextual recall metric."""

    def test_prompt_decomposes_ground_truth(self) -> None:
        """Verify prompt decomposes ground truth into facts."""
        metric = ContextualRecallMetric()
        eval_input = EvalInput(
            input="Query",
            actual_output="Response",
            expected_output="Ground truth answer",
            retrieval_context=["Context"],
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        # Should decompose expected answer into facts
        assert "Ground Truth" in prompt or "ground truth" in prompt.lower()
        assert "fact" in prompt.lower() or "statement" in prompt.lower()


# =============================================================================
# QUALITY METRICS TESTS
# =============================================================================


class TestCoherenceMetric:
    """Test the improved coherence metric."""

    def test_prompt_includes_discourse_coherence(self) -> None:
        """Verify prompt evaluates discourse-level coherence."""
        metric = CoherenceMetric()
        eval_input = EvalInput(
            input="Explain X", actual_output="X is important. It has many uses."
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "flow" in prompt.lower() or "Logical" in prompt
        assert "structure" in prompt.lower() or "Structure" in prompt


class TestCompletenessMetric:
    """Test the improved completeness metric."""

    def test_prompt_decomposes_query(self) -> None:
        """Verify prompt decomposes query into requirements."""
        metric = CompletenessMetric()
        eval_input = EvalInput(
            input="What are pros and cons?", actual_output="Some pros exist."
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Decomposition" in prompt or "requirement" in prompt.lower()
        assert "COVERED" in prompt or "covered" in prompt.lower()


# =============================================================================
# CONVERSATION METRICS TESTS
# =============================================================================


class TestContextRetentionMetric:
    """Test the improved context retention metric."""

    def test_prompt_evaluates_explicit_implicit_memory(self) -> None:
        """Verify prompt evaluates both explicit and implicit memory."""
        metric = ContextRetentionMetric()
        eval_input = EvalInput(
            input="What's my name?",
            actual_output="Your name is Alice",
            expected_output="Alice",
            history=[
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
            ],
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Explicit" in prompt or "explicit" in prompt.lower()
        assert "Implicit" in prompt or "implicit" in prompt.lower()


class TestConsistencyMetric:
    """Test the improved consistency metric."""

    def test_prompt_detects_contradictions(self) -> None:
        """Verify prompt checks for contradictions."""
        metric = ConsistencyMetric()
        eval_input = EvalInput(
            input="Test",
            actual_output="I love coffee",
            history=[{"role": "assistant", "content": "I hate coffee"}],
        )
        prompt = metric.get_evaluation_prompt(eval_input)

        assert "Contradiction" in prompt or "contradict" in prompt.lower()
        assert "CONSISTENT" in prompt or "consistent" in prompt.lower()


# =============================================================================
# BENCHMARK TEST CASES
# =============================================================================


class TestBenchmarkCases:
    """Benchmark test cases with known good/bad examples."""

    # These would be run with an actual LLM to verify scoring
    CORRECTNESS_CASES = [
        {
            "input": "What is 2+2?",
            "expected": "4",
            "actual": "Four",
            "expected_score_range": (0.8, 1.0),  # Semantically equivalent
        },
        {
            "input": "What is 2+2?",
            "expected": "4",
            "actual": "5",
            "expected_score_range": (0.0, 0.2),  # Incorrect
        },
    ]

    FAITHFULNESS_CASES = [
        {
            "input": "What is the capital of France?",
            "actual": "Paris is the capital of France.",
            "context": ["France is a country in Western Europe. Its capital is Paris."],
            "expected_score_range": (0.9, 1.0),  # Fully supported
        },
        {
            "input": "What is the capital of France?",
            "actual": "Paris is the capital of France, with a population of 12 million.",
            "context": ["France is a country in Western Europe. Its capital is Paris."],
            "expected_score_range": (0.4, 0.7),  # Partially hallucinated
        },
    ]

    @pytest.mark.skip(reason="Requires LLM integration for benchmarking")
    def test_correctness_benchmark(self) -> None:
        """Benchmark correctness metric against known cases."""
        # This would run actual evaluations with an LLM
        pass

    @pytest.mark.skip(reason="Requires LLM integration for benchmarking")
    def test_faithfulness_benchmark(self) -> None:
        """Benchmark faithfulness metric against known cases."""
        # This would run actual evaluations with an LLM
        pass


# =============================================================================
# PROMPT QUALITY ASSERTIONS
# =============================================================================


def test_all_prompts_under_token_limit() -> None:
    """Verify prompts are reasonably sized (not too long for context windows)."""
    metrics = [
        CorrectnessMetric(),
        HallucinationMetric(),
        FaithfulnessMetric(),
        ToxicityMetric(),
        BiasMetric(),
    ]

    sample_input = EvalInput(
        input="Test query",
        actual_output="Test response",
        expected_output="Expected",
        context=["Context"],
    )

    for metric in metrics:
        prompt = metric.get_evaluation_prompt(sample_input)
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) / 4
        # Prompts should be under 2000 tokens to leave room for response
        assert (
            estimated_tokens < 2000
        ), f"{metric.name} prompt too long: ~{estimated_tokens:.0f} tokens"


def test_no_deprecated_score_descriptions() -> None:
    """Verify prompts don't use vague score descriptions."""
    metrics = [
        CorrectnessMetric(),
        HallucinationMetric(),
        FaithfulnessMetric(),
    ]

    sample_input = EvalInput(
        input="Test", actual_output="Test", expected_output="Test", context=["Context"]
    )

    deprecated_phrases = [
        "provide a score",
        "rate from 0 to 1",
        "give a score",
    ]

    for metric in metrics:
        prompt = metric.get_evaluation_prompt(sample_input).lower()
        for phrase in deprecated_phrases:
            # These vague phrases should be replaced with detailed rubrics
            assert (
                phrase not in prompt or "rubric" in prompt
            ), f"{metric.name} uses deprecated phrase: '{phrase}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
