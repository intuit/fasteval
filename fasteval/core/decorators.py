"""Metric and data decorators for fasteval."""

import csv as csv_module
import functools
import inspect
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import EvalInput
from fasteval.providers.base import LLMClient

F = TypeVar("F", bound=Callable[..., Any])

# Marker attribute for fasteval-decorated functions
fasteval_METRICS_ATTR = "_fasteval_metrics"
fasteval_DATA_ATTR = "_fasteval_data"
fasteval_TRACES_ATTR = "_fasteval_traces"


def _attach_metric(func: F, config: MetricConfig) -> F:
    """Attach a metric config to a function."""
    if not hasattr(func, fasteval_METRICS_ATTR):
        setattr(func, fasteval_METRICS_ATTR, [])
    getattr(func, fasteval_METRICS_ATTR).append(config)
    return func


def _metric_decorator_factory(
    metric_type: str,
    default_name: str,
    default_threshold: float = 0.5,
    **default_config: Any,
) -> Callable[..., Callable[[F], F]]:
    """
    Factory for creating metric decorators.

    Args:
        metric_type: Type identifier (e.g., "correctness")
        default_name: Default metric name
        default_threshold: Default pass threshold
        **default_config: Default metric-specific config
    """

    def decorator(
        threshold: float = default_threshold,
        weight: float = 1.0,
        name: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **config: Any,
    ) -> Callable[[F], F]:
        """
        The actual decorator.

        Args:
            threshold: Pass threshold (0.0-1.0)
            weight: Weight for aggregate scoring
            name: Custom metric name
            llm_client: Custom LLM client
            model: Model override
            **config: Additional metric config
        """
        merged_config = {**default_config, **config}
        if model:
            merged_config["model"] = model

        def wrapper(func: F) -> F:
            metric_config = MetricConfig(
                metric_type=metric_type,
                name=name or default_name,
                threshold=threshold,
                weight=weight,
                config=merged_config,
                llm_client=llm_client,
                llm_config={"model": model} if model else None,
            )
            return _attach_metric(func, metric_config)

        return wrapper

    return decorator


# === LLM Metric Decorators ===


def correctness(
    threshold: float = 0.5,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach correctness metric to a test function.

    Evaluates if actual output is semantically correct vs expected.

    Example:
        @fe.correctness(threshold=0.8)
        async def test_qa():
            response = await agent("What is 2+2?")
            fe.score(response, "4")
    """
    return _metric_decorator_factory(
        "correctness", "correctness", default_threshold=0.5
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def hallucination(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach hallucination detection metric.

    Detects information not supported by provided context.

    Example:
        @fe.hallucination(threshold=0.9)
        async def test_rag():
            response = await agent(query, context=docs)
            fe.score(response, context=docs)
    """
    return _metric_decorator_factory(
        "hallucination", "hallucination", default_threshold=0.9
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def relevance(
    threshold: float = 0.5,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach relevance metric.

    Evaluates if response is relevant to the input.

    Example:
        @fe.relevance(threshold=0.7)
        async def test_relevance():
            response = await agent(question)
            fe.score(response, input=question)
    """
    return _metric_decorator_factory("relevance", "relevance", default_threshold=0.5)(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def criteria(
    criteria_text: str,
    evaluation_steps: Optional[List[str]] = None,
    threshold: float = 0.5,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Evaluate using custom criteria in plain English.

    Use this when built-in metrics don't fit your use case. Define any
    evaluation criteria and fasteval will use an LLM to judge against it.

    Example:
        @fe.criteria("Is the response empathetic and professional?")
        def test_tone():
            response = agent("I'm frustrated!")
            fe.score(response)

        @fe.criteria(
            "Does the response include a disclaimer?",
            threshold=0.9
        )
        def test_compliance():
            response = agent("Can I break my lease?")
            fe.score(response)

    Args:
        criteria_text: What to evaluate in plain English
        evaluation_steps: Optional chain-of-thought steps for consistency
        threshold: Minimum score to pass (0.0-1.0)
        weight: Weight for aggregate scoring
        name: Custom metric name (defaults to truncated criteria)
        llm_client: Custom LLM client
        model: Model override
    """
    config["criteria"] = criteria_text
    if evaluation_steps:
        config["evaluation_steps"] = evaluation_steps

    # Default name to first 30 chars of criteria if not provided
    default_name = name or f"criteria:{criteria_text[:30]}..."

    return _metric_decorator_factory("criteria", default_name, default_threshold=0.5)(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# Aliases for backwards compatibility and DeepEval familiarity
def geval(
    criteria: str,  # Keep 'criteria' for backwards compatibility
    evaluation_steps: Optional[List[str]] = None,
    threshold: float = 0.5,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """Alias for criteria(). Provided for DeepEval compatibility."""
    # Call the criteria function using globals to avoid shadowing
    return globals()["criteria"](
        criteria_text=criteria,
        evaluation_steps=evaluation_steps,
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# Also provide g_eval alias (snake_case version)
g_eval = geval


def toxicity(
    threshold: float = 0.95,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach toxicity detection metric.

    Example:
        @fe.toxicity(threshold=0.95)
        async def test_safe():
            response = await agent(input)
            fe.score(response)
    """
    return _metric_decorator_factory("toxicity", "toxicity", default_threshold=0.95)(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def bias(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach bias detection metric.

    Example:
        @fe.bias(threshold=0.9)
        async def test_unbiased():
            response = await agent(input)
            fe.score(response)
    """
    return _metric_decorator_factory("bias", "bias", default_threshold=0.9)(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === Quality Metric Decorators ===


def conciseness(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach conciseness metric.

    Evaluates brevity and information density of the response.

    Example:
        @fe.conciseness(threshold=0.7)
        def test_summary():
            response = agent("Summarize this article")
            fe.score(response, input=query)
    """
    return _metric_decorator_factory(
        "conciseness", "conciseness", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def coherence(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach coherence metric.

    Evaluates readability, logical flow, and language quality.

    Example:
        @fe.coherence(threshold=0.8)
        def test_writing_quality():
            response = agent("Explain quantum computing")
            fe.score(response)
    """
    return _metric_decorator_factory("coherence", "coherence", default_threshold=0.8)(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def completeness(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach completeness metric.

    Evaluates if the response covers all aspects of the query.

    Example:
        @fe.completeness(threshold=0.8)
        def test_comprehensive():
            response = agent("What are the pros and cons of remote work?")
            fe.score(response, input=query)
    """
    return _metric_decorator_factory(
        "completeness", "completeness", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def helpfulness(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach helpfulness metric.

    Evaluates the practical utility of the response.

    Example:
        @fe.helpfulness(threshold=0.7)
        def test_useful():
            response = agent("How do I fix a leaky faucet?")
            fe.score(response, input=query)
    """
    return _metric_decorator_factory(
        "helpfulness", "helpfulness", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def instruction_following(
    instructions: List[str],
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach instruction following metric.

    Evaluates if the response adheres to specific instructions.

    Example:
        @fe.instruction_following(
            instructions=["Be concise", "Use bullet points", "Include examples"],
            threshold=0.8
        )
        def test_format():
            response = agent(query)
            fe.score(response, input=query)

    Args:
        instructions: List of instructions the response should follow
        threshold: Minimum score to pass (0.0-1.0)
        weight: Weight for aggregate scoring
        name: Custom metric name
        llm_client: Custom LLM client
        model: Model override
    """
    config["instructions"] = instructions

    return _metric_decorator_factory(
        "instruction_following", "instruction_following", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === RAG Metric Decorators ===


def faithfulness(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach faithfulness metric for RAG evaluation.

    Evaluates if the response is grounded/faithful to the provided context.
    Every claim in the response should be supported by the context documents.

    Example:
        @fe.faithfulness(threshold=0.8)
        def test_rag():
            response = agent(query, context=docs)
            fe.score(response, context=docs, input=query)
    """
    return _metric_decorator_factory(
        "faithfulness", "faithfulness", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def contextual_precision(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach contextual precision metric for RAG evaluation.

    Evaluates if retrieved documents are relevant to the query.
    Measures retrieval quality by checking document relevance.

    Example:
        @fe.contextual_precision(threshold=0.7)
        def test_retrieval():
            docs = retriever.get_relevant_documents(query)
            response = llm(query, context=docs)
            fe.score(response, retrieval_context=docs, input=query)
    """
    return _metric_decorator_factory(
        "contextual_precision", "contextual_precision", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def contextual_recall(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach contextual recall metric for RAG evaluation.

    Evaluates if the retrieved context covers all information needed.
    Compares context against expected answer to check coverage.

    Example:
        @fe.contextual_recall(threshold=0.7)
        def test_retrieval_coverage():
            docs = retriever.get_relevant_documents(query)
            response = llm(query, context=docs)
            fe.score(response, expected_answer, retrieval_context=docs, input=query)
    """
    return _metric_decorator_factory(
        "contextual_recall", "contextual_recall", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def answer_correctness(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach answer correctness metric for RAG evaluation.

    Evaluates factual correctness of the answer against ground truth.
    Combines semantic similarity with factual accuracy.

    Example:
        @fe.answer_correctness(threshold=0.7)
        def test_answer():
            response = agent(query, context=docs)
            fe.score(response, expected_answer, context=docs, input=query)
    """
    return _metric_decorator_factory(
        "answer_correctness", "answer_correctness", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === Deterministic Metric Decorators ===


def rouge(
    rouge_type: str = "rougeL",
    use_stemmer: bool = True,
    threshold: float = 0.5,
    weight: float = 1.0,
    name: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach ROUGE similarity metric.

    Example:
        @fe.rouge(rouge_type="rougeL", threshold=0.5)
        async def test_summary():
            response = await agent(doc)
            fe.score(response, expected_summary)
    """
    config["rouge_type"] = rouge_type
    config["use_stemmer"] = use_stemmer

    return _metric_decorator_factory("rouge", "rouge", default_threshold=0.5)(
        threshold=threshold, weight=weight, name=name, **config
    )


def exact_match(
    normalize: bool = True,
    case_sensitive: bool = False,
    threshold: float = 1.0,
    weight: float = 1.0,
    name: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach exact match metric.

    Example:
        @fe.exact_match(case_sensitive=False)
        async def test_exact():
            response = await agent(query)
            fe.score(response, expected)
    """
    config["normalize"] = normalize
    config["case_sensitive"] = case_sensitive

    return _metric_decorator_factory(
        "exact_match", "exact_match", default_threshold=1.0
    )(threshold=threshold, weight=weight, name=name, **config)


def contains(
    case_sensitive: bool = False,
    threshold: float = 1.0,
    weight: float = 1.0,
    name: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach contains metric.

    Example:
        @fe.contains(case_sensitive=False)
        async def test_contains():
            response = await agent(query)
            fe.score(response, "expected substring")
    """
    config["case_sensitive"] = case_sensitive

    return _metric_decorator_factory("contains", "contains", default_threshold=1.0)(
        threshold=threshold, weight=weight, name=name, **config
    )


def json(
    model: Type[BaseModel],
    threshold: float = 1.0,
    weight: float = 1.0,
    name: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach JSON schema validation metric using Pydantic model.

    Example:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        @fe.json(model=User)
        async def test_json_output():
            response = await agent("Generate a user")
            fe.score(response)
    """
    config["pydantic_model"] = model

    return _metric_decorator_factory("json", "json", default_threshold=1.0)(
        threshold=threshold, weight=weight, name=name, **config
    )


def regex(
    pattern: str,
    flags: int = 0,
    full_match: bool = True,
    threshold: float = 1.0,
    weight: float = 1.0,
    name: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach regex pattern matching metric.

    Validates that the output matches a regular expression pattern.
    Returns 1.0 if pattern matches, 0.0 otherwise.

    Example:
        # Match phone number format (full match)
        @fe.regex(pattern=r"^\d{3}-\d{4}$")
        def test_phone_format():
            fe.score(response)

        # Case-insensitive matching
        import re
        @fe.regex(pattern=r"^yes|no$", flags=re.IGNORECASE)
        def test_yes_no():
            fe.score(response)

        # Search for pattern anywhere in output
        @fe.regex(pattern=r"\b\d{4}\b", full_match=False)
        def test_contains_year():
            fe.score(response)

    Args:
        pattern: Regular expression pattern to match
        flags: Regex flags (e.g., re.IGNORECASE, re.MULTILINE)
        full_match: If True, pattern must match entire output.
                    If False, pattern can match anywhere in output.
        threshold: Score threshold (typically 1.0 for pass/fail)
        weight: Weight for aggregate scoring
        name: Custom metric name
    """
    config["pattern"] = pattern
    config["flags"] = flags
    config["full_match"] = full_match

    return _metric_decorator_factory("regex", "regex", default_threshold=1.0)(
        threshold=threshold, weight=weight, name=name, **config
    )


# === Tool Trajectory Metric Decorators ===


def tool_call_accuracy(
    threshold: float = 0.8,
    ignore_extra: bool = False,
    weight: float = 1.0,
    name: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach tool call accuracy metric (deterministic).

    Measures if the correct tools were called.
    Score = matched_tools / max(expected_required, actual_count)

    Example:
        from fasteval import ExpectedTool

        @fe.tool_call_accuracy(threshold=0.8)
        def test_booking():
            response, tool_calls = agent("Book flight to NYC")
            expected = [
                ExpectedTool(name="search_flights"),
                ExpectedTool(name="book_flight"),
            ]
            fe.score(response, tool_calls=tool_calls, expected_tools=expected)

    Args:
        threshold: Minimum score to pass (0.0 - 1.0)
        ignore_extra: If True, don't penalize extra tool calls
        weight: Weight for score aggregation
        name: Custom metric name
    """
    config["ignore_extra"] = ignore_extra

    return _metric_decorator_factory(
        "tool_call_accuracy", "tool_call_accuracy", default_threshold=0.8
    )(threshold=threshold, weight=weight, name=name, **config)


def tool_sequence(
    threshold: float = 0.8,
    strict: bool = False,
    weight: float = 1.0,
    name: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach tool sequence metric (deterministic).

    Measures if tools were called in the correct order.
    - Strict mode: Exact sequence match (score 0 or 1)
    - Non-strict: Longest Common Subsequence ratio

    Example:
        @fe.tool_sequence(threshold=0.8, strict=False)
        def test_workflow():
            response, tool_calls = agent("Process order")
            expected = [
                ExpectedTool(name="validate_order"),
                ExpectedTool(name="charge_payment"),
                ExpectedTool(name="send_confirmation"),
            ]
            fe.score(response, tool_calls=tool_calls, expected_tools=expected)

    Args:
        threshold: Minimum score to pass (0.0 - 1.0)
        strict: If True, require exact sequence match
        weight: Weight for score aggregation
        name: Custom metric name
    """
    config["strict"] = strict

    return _metric_decorator_factory(
        "tool_sequence", "tool_sequence", default_threshold=0.8
    )(threshold=threshold, weight=weight, name=name, **config)


def tool_args_match(
    threshold: float = 0.8,
    ignore_extra: bool = True,
    partial_match: bool = True,
    fuzzy_match: bool = False,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach tool arguments match metric (deterministic, or LLM with fuzzy_match).

    Measures if tool arguments match expected values.
    Score = matched_args / total_expected_args

    Example:
        @fe.tool_args_match(threshold=0.9)
        def test_search():
            response, tool_calls = agent("Search for flights to NYC")
            expected = [
                ExpectedTool(name="search_flights", args={"destination": "NYC"}),
            ]
            fe.score(response, tool_calls=tool_calls, expected_tools=expected)

    Args:
        threshold: Minimum score to pass (0.0 - 1.0)
        ignore_extra: If True, don't penalize extra arguments
        partial_match: If True, allow subset of expected args
        fuzzy_match: If True, use LLM for semantic comparison (e.g., "NYC" ≈ "New York")
        weight: Weight for score aggregation
        name: Custom metric name
        llm_client: LLM client for fuzzy matching (only used if fuzzy_match=True)
        model: Model name for fuzzy matching
    """
    config["ignore_extra"] = ignore_extra
    config["partial_match"] = partial_match
    config["fuzzy_match"] = fuzzy_match

    return _metric_decorator_factory(
        "tool_args_match", "tool_args_match", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === Conversation Metric Decorators ===


def context_retention(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach context retention metric for multi-turn evaluation.

    Example:
        @fe.context_retention(threshold=0.8)
        @fe.conversation([...])
        async def test_memory():
            ...
    """
    return _metric_decorator_factory(
        "context_retention", "context_retention", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def consistency(
    threshold: float = 1.0,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach consistency metric for multi-turn evaluation.

    Example:
        @fe.consistency(threshold=1.0)
        @fe.conversation([...])
        async def test_consistent():
            ...
    """
    return _metric_decorator_factory(
        "consistency", "consistency", default_threshold=1.0
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def topic_drift(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach topic drift detection metric.

    Example:
        @fe.topic_drift(threshold=0.8)
        @fe.conversation([...])
        async def test_on_topic():
            ...
    """
    return _metric_decorator_factory(
        "topic_drift", "topic_drift", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === Vision Metric Decorators ===


def image_understanding(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach image understanding metric for VLM evaluation.

    Evaluates if the model correctly understands and describes image content.

    Example:
        @fe.image_understanding(threshold=0.8)
        def test_vision_qa():
            response = gpt4v.chat(image="chart.png", prompt="What trend?")
            fe.score(response, expected="Revenue increased", image="chart.png")
    """
    return _metric_decorator_factory(
        "image_understanding", "image_understanding", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def ocr_accuracy(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach OCR accuracy metric for text extraction evaluation.

    Example:
        @fe.ocr_accuracy(threshold=0.95)
        def test_extraction():
            text = vision_model.extract("invoice.pdf")
            fe.score(text, expected_fields={"total": "$1,234"})
    """
    return _metric_decorator_factory(
        "ocr_accuracy", "ocr_accuracy", default_threshold=0.9
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def chart_interpretation(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach chart interpretation metric.

    Evaluates correct reading of charts, graphs, and visualizations.

    Example:
        @fe.chart_interpretation(threshold=0.8)
        def test_chart():
            response = model.analyze(image="sales_chart.png", query="Q3 revenue?")
            fe.score(response, expected="$4.2M", image="sales_chart.png")
    """
    return _metric_decorator_factory(
        "chart_interpretation", "chart_interpretation", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def visual_grounding(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach visual grounding metric.

    Evaluates if response correctly references specific image regions.

    Example:
        @fe.visual_grounding(threshold=0.8)
        def test_location():
            response = model.locate(image="room.jpg", query="Where is the lamp?")
            fe.score(response, expected="upper right corner", image="room.jpg")
    """
    return _metric_decorator_factory(
        "visual_grounding", "visual_grounding", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def image_faithfulness(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach image faithfulness metric.

    Evaluates if response is grounded in what the image shows (no hallucinations).

    Example:
        @fe.image_faithfulness(threshold=0.85)
        def test_faithful():
            response = model.describe(image="photo.jpg")
            fe.score(response, image="photo.jpg")
    """
    return _metric_decorator_factory(
        "image_faithfulness", "image_faithfulness", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def image_quality(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach image quality metric for image generation evaluation.

    Evaluates visual quality, coherence, and aesthetic appeal.

    Example:
        @fe.image_quality(threshold=0.7)
        def test_gen_quality():
            image = dalle.generate("A sunset over mountains")
            fe.score(generated_image=image, input="A sunset over mountains")
    """
    return _metric_decorator_factory(
        "image_quality", "image_quality", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def prompt_adherence(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach prompt adherence metric for image generation.

    Evaluates if generated image matches the prompt description.

    Example:
        @fe.prompt_adherence(threshold=0.8)
        def test_prompt_match():
            image = dalle.generate("A red sports car on a mountain road")
            fe.score(generated_image=image, input="A red sports car on a mountain road")
    """
    return _metric_decorator_factory(
        "prompt_adherence", "prompt_adherence", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def safety_check(
    threshold: float = 0.95,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach safety check metric for image generation.

    Checks generated images for NSFW or harmful content.

    Example:
        @fe.safety_check(threshold=0.95)
        def test_safe_gen():
            image = model.generate(prompt)
            fe.score(generated_image=image, input=prompt)
    """
    return _metric_decorator_factory(
        "safety_check", "safety_check", default_threshold=0.95
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === Audio/Speech Metric Decorators ===


def word_error_rate(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    normalize_text: bool = True,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach Word Error Rate (WER) metric for ASR evaluation.

    Standard industry metric for speech recognition. Score = 1.0 - WER.

    Example:
        @fe.word_error_rate(threshold=0.95)  # WER < 5%
        def test_asr():
            transcript = whisper.transcribe("audio.mp3")
            fe.score(transcript, expected=ground_truth)

    Args:
        threshold: Minimum accuracy (1.0 - WER) to pass
        normalize_text: If True, normalize text before comparison
    """
    config["normalize_text"] = normalize_text
    return _metric_decorator_factory(
        "word_error_rate", "word_error_rate", default_threshold=0.9
    )(threshold=threshold, weight=weight, name=name, **config)


def character_error_rate(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    normalize_text: bool = True,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach Character Error Rate (CER) metric for ASR evaluation.

    Useful for character-level ASR evaluation. Score = 1.0 - CER.

    Example:
        @fe.character_error_rate(threshold=0.95)
        def test_asr():
            transcript = model.transcribe(audio)
            fe.score(transcript, expected=reference)
    """
    config["normalize_text"] = normalize_text
    return _metric_decorator_factory(
        "character_error_rate", "character_error_rate", default_threshold=0.9
    )(threshold=threshold, weight=weight, name=name, **config)


def match_error_rate(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    normalize_text: bool = True,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach Match Error Rate (MER) metric for ASR evaluation.

    Variant of WER bounded between 0 and 1.

    Example:
        @fe.match_error_rate(threshold=0.9)
        def test_asr():
            transcript = model.transcribe(audio)
            fe.score(transcript, expected=reference)
    """
    config["normalize_text"] = normalize_text
    return _metric_decorator_factory(
        "match_error_rate", "match_error_rate", default_threshold=0.9
    )(threshold=threshold, weight=weight, name=name, **config)


def transcription_accuracy(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach transcription accuracy metric (LLM-based).

    Evaluates semantic correctness of transcription, not just character match.

    Example:
        @fe.transcription_accuracy(threshold=0.9)
        def test_semantic():
            transcript = whisper.transcribe("meeting.mp3")
            fe.score(transcript, expected=ground_truth)
    """
    return _metric_decorator_factory(
        "transcription_accuracy", "transcription_accuracy", default_threshold=0.9
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def speaker_diarization(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach speaker diarization accuracy metric.

    Evaluates if speakers are correctly identified.

    Example:
        @fe.speaker_diarization(threshold=0.8)
        def test_speakers():
            result = model.transcribe_with_speakers("meeting.mp3")
            fe.score(result, expected=annotated_transcript)
    """
    return _metric_decorator_factory(
        "speaker_diarization", "speaker_diarization", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def audio_sentiment(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach audio sentiment/emotion detection metric.

    Evaluates tone/emotion detection accuracy.

    Example:
        @fe.audio_sentiment(threshold=0.8)
        def test_sentiment():
            sentiment = model.detect_sentiment("call.mp3")
            fe.score(sentiment, expected="frustrated, but cooperative")
    """
    return _metric_decorator_factory(
        "audio_sentiment", "audio_sentiment", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === Multi-Modal Metric Decorators ===


def multimodal_faithfulness(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach multi-modal faithfulness metric.

    Evaluates if response is faithful to mixed text+image context.

    Example:
        @fe.multimodal_faithfulness(threshold=0.8)
        def test_doc_qa():
            response = rag.query("Q3 revenue?", docs=["report.pdf"])
            fe.score(response, context=text_chunks, images=["chart.png"])
    """
    return _metric_decorator_factory(
        "multimodal_faithfulness", "multimodal_faithfulness", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def table_extraction(
    threshold: float = 0.9,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach table extraction accuracy metric.

    Evaluates accuracy of table data extraction from images.

    Example:
        @fe.table_extraction(threshold=0.9)
        def test_table():
            answer = model.answer(image="table.png", q="Total for 2024?")
            fe.score(answer, expected="$150,000", image="table.png")
    """
    return _metric_decorator_factory(
        "table_extraction", "table_extraction", default_threshold=0.9
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def figure_reference(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach figure reference metric.

    Evaluates if response correctly references figures/charts.

    Example:
        @fe.figure_reference(threshold=0.8)
        def test_figures():
            response = model.analyze(images=["doc.png"], query="Summarize trends")
            fe.score(response, images=["doc.png"])
    """
    return _metric_decorator_factory(
        "figure_reference", "figure_reference", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def cross_modal_coherence(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach cross-modal coherence metric.

    Evaluates coherence between text and visual elements.

    Example:
        @fe.cross_modal_coherence(threshold=0.8)
        def test_caption():
            caption = model.caption(image="photo.jpg")
            fe.score(caption, image="photo.jpg")
    """
    return _metric_decorator_factory(
        "cross_modal_coherence", "cross_modal_coherence", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def document_understanding(
    threshold: float = 0.8,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach document understanding metric.

    Evaluates understanding of complex documents with mixed content.

    Example:
        @fe.document_understanding(threshold=0.8)
        def test_doc():
            response = model.query(doc="contract.pdf", q="Termination clause?")
            fe.score(response, expected="30-day notice", image="contract.pdf")
    """
    return _metric_decorator_factory(
        "document_understanding", "document_understanding", default_threshold=0.8
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def clip_score(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach CLIP-style image-text alignment metric.

    Evaluates semantic alignment between image and text.

    Example:
        @fe.clip_score(threshold=0.7)
        def test_alignment():
            image = dalle.generate("A red sports car")
            fe.score(generated_image=image, input="A red sports car")
    """
    return _metric_decorator_factory("clip_score", "clip_score", default_threshold=0.7)(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


def aesthetic_score(
    threshold: float = 0.7,
    weight: float = 1.0,
    name: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    **config: Any,
) -> Callable[[F], F]:
    """
    Attach aesthetic score metric for image generation.

    Evaluates visual appeal, composition, and artistic merit.

    Example:
        @fe.aesthetic_score(threshold=0.7)
        def test_aesthetics():
            image = model.generate("A beautiful sunset")
            fe.score(generated_image=image, input="A beautiful sunset")
    """
    return _metric_decorator_factory(
        "aesthetic_score", "aesthetic_score", default_threshold=0.7
    )(
        threshold=threshold,
        weight=weight,
        name=name,
        llm_client=llm_client,
        model=model,
        **config,
    )


# === Generic Metric Decorator ===


def metric(
    metric_instance: Any,
    weight: float = 1.0,
) -> Callable[[F], F]:
    """
    Attach a custom metric instance to a test function.

    Example:
        from fasteval.metrics import BaseLLMMetric

        class MyMetric(BaseLLMMetric):
            ...

        @fe.metric(MyMetric(threshold=0.7))
        async def test_custom():
            ...
    """

    def decorator(func: F) -> F:
        config = MetricConfig(
            metric_type="custom",
            name=getattr(metric_instance, "name", "custom"),
            threshold=getattr(metric_instance, "threshold", 0.5),
            weight=weight,
            config={"instance": metric_instance},
        )
        return _attach_metric(func, config)

    return decorator


# === Code-as-Judge Decorator ===


def judge(
    func: Callable[..., Any],
    threshold: float = 0.5,
    weight: float = 1.0,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Use a plain function as an evaluation metric (Code as a Judge).

    The function can accept any subset of EvalInput fields as named
    parameters, or the full EvalInput via an ``eval_input`` parameter.
    It must return a score (float 0-1), a (score, reasoning) tuple,
    a dict with a ``score`` key, or a MetricResult.

    Example:
        def check_tone(actual_output: str, input: str) -> float:
            return 1.0 if "thank you" in actual_output.lower() else 0.3

        @fe.judge(check_tone, threshold=0.8)
        def test_support():
            response = my_agent("Help me")
            fe.score(response, input="Help me")
    """
    from fasteval.metrics.code_judge import CodeJudgeMetric

    metric_name = name or func.__name__
    instance = CodeJudgeMetric(
        func=func,
        name=metric_name,
        threshold=threshold,
        weight=weight,
    )

    def decorator(test_func: F) -> F:
        config = MetricConfig(
            metric_type="custom",
            name=metric_name,
            threshold=threshold,
            weight=weight,
            config={"instance": instance},
        )
        return _attach_metric(test_func, config)

    return decorator


# === Data Decorators ===


def csv(
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> Callable[[F], F]:
    """
    Load test cases from CSV file.

    CSV columns become keyword arguments to the test function.

    Example:
        # test_data.csv:
        # query,expected
        # "What is 2+2?","4"
        # "What is 3+3?","6"

        @fe.correctness()
        @fe.csv("test_data.csv")
        async def test_qa(query, expected):
            response = await agent(query)
            fe.score(response, expected, input=query)
    """

    def decorator(func: F) -> F:
        filepath = Path(path)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> None:
            with open(filepath, "r", encoding=encoding) as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    # Call the test function with row data as kwargs
                    if inspect.iscoroutinefunction(func):
                        await func(**{**kwargs, **row})
                    else:
                        func(**{**kwargs, **row})

        # Mark as data-decorated and preserve metrics
        if hasattr(func, fasteval_METRICS_ATTR):
            setattr(
                wrapper, fasteval_METRICS_ATTR, getattr(func, fasteval_METRICS_ATTR)
            )

        setattr(wrapper, fasteval_DATA_ATTR, {"type": "csv", "path": str(filepath)})
        return wrapper  # type: ignore

    return decorator


def conversation(
    turns: List[Dict[str, Any]],
) -> Callable[[F], F]:
    """
    Run multi-turn conversation test.

    Each turn can specify: query, expected, and optional metadata.
    History accumulates across turns.

    Example:
        @fe.context_retention()
        @fe.conversation([
            {"query": "My name is Alice"},
            {"query": "What's my name?", "expected": "Alice"},
        ])
        async def test_memory(query, expected, history):
            response = await agent(query, history=history)
            fe.score(response, expected, input=query, history=history)
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        wrapper: Callable[..., Any]

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> None:
                from fasteval.core.scoring import get_last_score_result

                history: List[Dict[str, str]] = []

                for turn in turns:
                    query = turn.get("query", "")
                    expected = turn.get("expected")

                    # Call the test function
                    result = await func(
                        query=query,
                        expected=expected,
                        history=history.copy(),
                        **kwargs,
                    )

                    # Get result from return value or from last score
                    if result is None or not hasattr(result, "eval_input"):
                        result = get_last_score_result()

                    # Extract actual_output to build conversation history
                    if result is not None and hasattr(result, "eval_input"):
                        last_response = result.eval_input.actual_output or ""
                        history.append({"role": "user", "content": query})
                        history.append({"role": "assistant", "content": last_response})

            wrapper = async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> None:
                from fasteval.core.scoring import get_last_score_result

                history: List[Dict[str, str]] = []

                for turn in turns:
                    query = turn.get("query", "")
                    expected = turn.get("expected")

                    # Call the test function
                    result = func(
                        query=query,
                        expected=expected,
                        history=history.copy(),
                        **kwargs,
                    )

                    # Get result from return value or from last score
                    if result is None or not hasattr(result, "eval_input"):
                        result = get_last_score_result()

                    # Extract actual_output to build conversation history
                    if result is not None and hasattr(result, "eval_input"):
                        last_response = result.eval_input.actual_output or ""
                        history.append({"role": "user", "content": query})
                        history.append({"role": "assistant", "content": last_response})

            wrapper = sync_wrapper

        # Preserve metrics
        if hasattr(func, fasteval_METRICS_ATTR):
            setattr(
                wrapper, fasteval_METRICS_ATTR, getattr(func, fasteval_METRICS_ATTR)
            )

        setattr(wrapper, fasteval_DATA_ATTR, {"type": "conversation", "turns": turns})
        return wrapper  # type: ignore

    return decorator


# === Traces Decorator (EvalOps) ===


def traces() -> Callable[[F], F]:
    """
    Mark a generator function as a trace source for EvalOps.

    The generator yields EvalInput objects. Metrics stacked on this function
    are evaluated for each yielded input.

    Example:
        @fe.correctness(threshold=0.8)
        @fe.traces()
        def get_production_traces():
            for trace in fetch_traces_from_db():
                yield EvalInput(
                    actual_output=trace.response,
                    expected_output=trace.expected,
                    input=trace.query,
                )

        # Run evaluation:
        # pytest test_traces.py
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Generator[EvalInput, None, None]:
            gen = func(*args, **kwargs)
            for eval_input in gen:
                if not isinstance(eval_input, EvalInput):
                    raise TypeError(
                        f"@traces generator must yield EvalInput objects, got {type(eval_input)}"
                    )
                yield eval_input

        # Preserve metrics
        if hasattr(func, fasteval_METRICS_ATTR):
            setattr(
                wrapper, fasteval_METRICS_ATTR, getattr(func, fasteval_METRICS_ATTR)
            )

        setattr(wrapper, fasteval_TRACES_ATTR, True)
        return wrapper  # type: ignore

    return decorator


# === Human Review Decorator ===

# Marker attribute for human review configuration
fasteval_HUMAN_REVIEW_ATTR = "_fasteval_human_review"


def human_review(
    prompt: str = "Please review this response",
    required: bool = False,
    threshold: float = 0.5,
) -> Callable[[F], F]:
    """
    Add human-in-the-loop review step after automated evaluation.

    When the test runs, after automated metrics are evaluated, the terminal
    will display the evaluation context and prompt the reviewer for a score.

    The reviewer can:
    - Enter a score 1-5 (normalized to 0.0-1.0)
    - Type 'p' or 'pass' for score = 1.0
    - Type 'f' or 'fail' for score = 0.0
    - Type 's' or 'skip' to skip review

    In CI/CD environments, human review is automatically skipped when:
    - --no-interactive flag is passed to pytest
    - FASTEVAL_NO_INTERACTIVE=1 environment variable is set
    - stdin is not a TTY

    Args:
        prompt: Question to ask the reviewer
        required: If True, test fails when reviewer skips (default: False)
        threshold: Minimum human score to pass (default: 0.5)

    Example:
        @fe.human_review(prompt="Is this response helpful and accurate?")
        @fe.correctness(threshold=0.8)
        def test_qa():
            response = agent(query)
            fe.score(response, expected, input=query)

    Multi-turn conversation example:
        @fe.human_review(prompt="Is context maintained correctly?")
        @fe.context_retention(threshold=0.8)
        @fe.conversation([
            {"query": "My name is Alice"},
            {"query": "What's my name?", "expected": "Alice"},
        ])
        def test_memory(query, expected, history):
            response = agent(query, history=history)
            return fe.score(response, expected, input=query, history=history)
    """

    def decorator(func: F) -> F:
        # Store human review config on the function
        setattr(
            func,
            fasteval_HUMAN_REVIEW_ATTR,
            {
                "prompt": prompt,
                "required": required,
                "threshold": threshold,
            },
        )
        return func

    return decorator


# === Metric Stack Decorator ===


def stack(name: Optional[str] = None) -> Callable[[F], Callable[[F], F]]:
    """
    Transform a metric-decorated function into a reusable decorator.

    Place @fe.stack() at the TOP of your decorator stack. It captures all
    metrics attached by decorators below it and creates a reusable decorator
    that applies those metrics to any function it decorates.

    Example:
        # Define a reusable metric stack
        # Note: @fe.stack() goes at the TOP
        @fe.stack()
        @fe.correctness(threshold=0.8)
        @fe.relevance(threshold=0.7)
        @fe.toxicity(threshold=0.95)
        def quality_metrics():
            pass

        # Use the stack on multiple tests
        @quality_metrics
        def test_chatbot_response():
            response = agent("What is Python?")
            fe.score(response, expected="A programming language")

        @quality_metrics
        def test_another_response():
            response = agent("What is JavaScript?")
            fe.score(response, expected="A programming language")

        # Stacks can be combined with other stacks or metrics
        @fe.bias(threshold=0.9)  # Additional metric
        @quality_metrics         # Reusable stack
        def test_combined():
            response = agent("Tell me about programming")
            fe.score(response, expected="Programming is...")

    Args:
        name: Optional name for the stack (defaults to function name)

    Returns:
        A decorator that transforms the function into a reusable metric decorator
    """

    def outer(func: F) -> Callable[[F], F]:
        # Capture metrics from the placeholder function
        # These were attached by decorators that ran before us (below us in the source)
        captured_metrics: List[MetricConfig] = list(
            getattr(func, fasteval_METRICS_ATTR, [])
        )
        stack_name = name or func.__name__

        def stack_decorator(target_func: F) -> F:
            # Copy captured metrics to the target function
            for metric_config in captured_metrics:
                _attach_metric(target_func, metric_config)
            return target_func

        # Store metadata for introspection
        stack_decorator.__name__ = stack_name
        stack_decorator.__doc__ = (
            f"Metric stack '{stack_name}' with {len(captured_metrics)} metrics"
        )
        setattr(stack_decorator, "_is_fasteval_stack", True)
        setattr(stack_decorator, "_captured_metrics", captured_metrics)

        return stack_decorator

    return outer
