"""Example tests demonstrating fasteval usage."""

import pytest
from pydantic import BaseModel

import fasteval as fe
from fasteval.models.evaluation import EvalInput

# === Example: Basic Correctness Test ===


@fe.correctness(threshold=0.8)
def test_basic_qa():
    """
    Basic test showing decorator + score pattern.

    In real usage, you'd call your LLM/agent here.
    This test is skipped since it requires an LLM.
    """
    pytest.skip("Requires LLM - example only")

    # In actual usage with async agent:
    # async def test_basic_qa():
    #     response = await your_agent("What is 2+2?")
    #     fe.score(response, "4", input="What is 2+2?")


# === Example: Multiple Metrics ===


@fe.correctness(threshold=0.7)
@fe.relevance(threshold=0.8)
@fe.toxicity(threshold=0.95)
def test_multiple_metrics():
    """
    Test with multiple metrics stacked.
    All metrics must pass for the test to pass.
    """
    pytest.skip("Requires LLM - example only")


# === Example: Deterministic Metrics (No LLM Required) ===


@fe.exact_match(case_sensitive=False)
def test_exact_match_deterministic():
    """Test using exact_match - no LLM required."""
    # Simulate getting a response
    actual = "Hello World"
    expected = "hello world"

    # Evaluation happens immediately - raises EvaluationFailedError if fails
    result = fe.score(actual, expected)
    assert result.passed


@fe.contains()
def test_contains_deterministic():
    """Test using contains metric."""
    actual = "The capital of France is Paris"
    expected = "Paris"

    # Evaluation happens immediately
    result = fe.score(actual, expected)
    assert result.passed


# === Example: JSON Schema Validation ===


class UserResponse(BaseModel):
    """Expected response schema."""

    name: str
    age: int
    email: str


@fe.json(model=UserResponse)
def test_json_output():
    """Test that output matches Pydantic schema."""
    # Simulated LLM output
    output = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'

    # Evaluation happens immediately
    result = fe.score(output)
    assert result.passed


# === Example: RAG Evaluation with Hallucination Detection ===


@fe.hallucination(threshold=0.9)
def test_rag_no_hallucination():
    """
    Test RAG response for hallucinations.

    Requires context to be provided.
    """
    pytest.skip("Requires LLM - example only")

    # Example context for RAG:
    # context = [
    #     "Paris is the capital of France.",
    #     "France is located in Western Europe.",
    # ]
    # response = await your_rag_agent(query, context=context)
    # fe.score(response, context=context)


# === Example: Custom Criteria Evaluation ===


@fe.criteria(
    "Response should be professional, helpful, and use formal language",
    threshold=0.7,
)
def test_professional_tone():
    """Test with custom criteria evaluation."""
    pytest.skip("Requires LLM - example only")


# === Example: Traces Evaluation (EvalOps) ===


# Note: @traces() decorated functions must be generators that yield EvalInput.
# pytest doesn't allow 'yield' in test functions, so traces generators should
# be defined separately and called from tests.


def _production_traces_generator():
    """
    Generator that yields EvalInput objects from production traces.

    This would typically fetch traces from a database or observability platform.
    """
    # Simulated production traces
    traces = [
        {"input": "What is 2+2?", "output": "4", "expected": "4"},
        {"input": "What is 3+3?", "output": "6", "expected": "6"},
    ]

    for trace in traces:
        yield EvalInput(
            input=trace["input"],
            actual_output=trace["output"],
            expected_output=trace["expected"],
        )


@fe.correctness(threshold=0.8)
@fe.traces()
def get_production_traces():
    """
    Trace generator decorated with @traces().

    Note: This is NOT a test function (no 'test_' prefix).
    Call it from a test or use 'pyvet run' to evaluate.
    """
    return _production_traces_generator()


def test_production_traces_example():
    """
    Example showing how to use traces evaluation.

    In practice, you'd use 'pyvet run' CLI or call the generator
    from a test with manual evaluation.
    """
    pytest.skip("Requires LLM - example only; see get_production_traces()")


# === Example: Custom LLM Client ===


class MockLLMClient:
    """Example custom LLM client implementation."""

    async def invoke(self, messages):
        # Return a mock evaluation response
        return '{"score": 0.85, "reasoning": "Mock evaluation"}'

    async def get_embedding(self, text):
        return [0.1, 0.2, 0.3]  # Mock embedding


@fe.correctness(threshold=0.7, llm_client=MockLLMClient())
def test_with_custom_client():
    """Test using a custom LLM client for evaluation."""
    pytest.skip("Example only")

    # fe.score(response, expected)


# === Example: Set Global Default Provider ===


def test_configure_provider():
    """Example of configuring the default LLM provider."""
    # Set a global default
    # fe.set_default_provider(fe.OpenAIClient(model="gpt-4o-mini"))

    # Or auto-detect from environment:
    # OPENAI_API_KEY=... pytest tests/

    pass


# === Example: Manual Evaluation (without pytest plugin) ===


def test_manual_evaluation():
    """Example of running evaluation manually."""
    import asyncio

    async def _run_evaluation():
        evaluator = fe.create_evaluator()

        eval_input = EvalInput(
            actual_output="The capital of France is Paris",
            expected_output="Paris",
            input="What is the capital of France?",
        )

        # Use deterministic metric (no LLM needed)
        from fasteval.models.config import MetricConfig

        result = await evaluator.evaluate(
            eval_input=eval_input,
            metrics=[
                MetricConfig(
                    metric_type="contains",
                    name="contains",
                    threshold=1.0,
                ),
            ],
        )

        assert result.passed is True
        assert result.metric_results[0].score == 1.0

    asyncio.run(_run_evaluation())


# === Example: Metric Stacks ===


# Define a reusable metric stack
# Note: @fe.stack() goes at the TOP to capture metrics below
@fe.stack()
@fe.exact_match(case_sensitive=False)
@fe.contains()
def deterministic_quality():
    """Stack of deterministic metrics for quick quality checks."""
    pass


# Define another stack
@fe.stack()
@fe.contains()
def contains_stack():
    """Simple contains check stack."""
    pass


def test_stack_basic_usage():
    """Test using a metric stack on a test function."""

    # Define a function and apply the stack
    @deterministic_quality
    def _inner_test():
        pass

    # Check that metrics were attached
    from fasteval.core.decorators import fasteval_METRICS_ATTR

    metrics = getattr(_inner_test, fasteval_METRICS_ATTR, [])
    assert len(metrics) == 2
    metric_types = [m.metric_type for m in metrics]
    assert "exact_match" in metric_types
    assert "contains" in metric_types


def test_stack_combined_with_additional_metric():
    """Test stack combined with additional metrics."""

    @fe.rouge(threshold=0.5)  # Additional metric
    @deterministic_quality  # Stack
    def _inner_test():
        pass

    from fasteval.core.decorators import fasteval_METRICS_ATTR

    metrics = getattr(_inner_test, fasteval_METRICS_ATTR, [])
    assert len(metrics) == 3
    metric_types = [m.metric_type for m in metrics]
    assert "exact_match" in metric_types
    assert "contains" in metric_types
    assert "rouge" in metric_types


def test_stack_multiple_stacks_combined():
    """Test combining multiple stacks."""

    @contains_stack
    @deterministic_quality
    def _inner_test():
        pass

    from fasteval.core.decorators import fasteval_METRICS_ATTR

    metrics = getattr(_inner_test, fasteval_METRICS_ATTR, [])
    # deterministic_quality has 2 metrics, contains_stack has 1
    assert len(metrics) == 3


def test_stack_introspection():
    """Test that stack metadata is accessible."""
    # Check stack name
    assert deterministic_quality.__name__ == "deterministic_quality"

    # Check it's marked as a stack
    assert getattr(deterministic_quality, "_is_fasteval_stack", False) is True

    # Check captured metrics
    captured = getattr(deterministic_quality, "_captured_metrics", [])
    assert len(captured) == 2


def test_stack_actual_evaluation():
    """Test that a stack actually evaluates correctly."""

    @deterministic_quality
    def _test_func():
        actual = "Hello World"
        expected = "hello world"
        return fe.score(actual, expected)

    # Run the test function
    result = _test_func()
    assert result.passed
    assert len(result.metric_results) == 2
