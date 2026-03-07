"""Score and evaluate API for fasteval."""

import inspect
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fasteval.core.decorators import fasteval_HUMAN_REVIEW_ATTR, fasteval_METRICS_ATTR
from fasteval.core.evaluator import Evaluator
from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import (
    EvalInput,
    EvalResult,
    EvaluationFailedError,
    ExpectedTool,
    HumanReviewResult,
    ToolCall,
)
from fasteval.models.multimodal import (
    AudioInput,
    GeneratedImage,
    ImageInput,
    MultiModalContext,
)
from fasteval.utils import format_evaluation_report, prompt_human_review, run_async

# Shared evaluator instance
_evaluator = Evaluator()


def _normalize_image(
    image: Union[ImageInput, str, Path],
) -> Union[ImageInput, str]:
    """Normalize image input to ImageInput or string."""
    if isinstance(image, ImageInput):
        return image
    if isinstance(image, Path):
        return str(image)
    return image


def _normalize_audio(
    audio: Union[AudioInput, str, Path],
) -> Union[AudioInput, str]:
    """Normalize audio input to AudioInput or string."""
    if isinstance(audio, AudioInput):
        return audio
    if isinstance(audio, Path):
        return str(audio)
    return audio


def _normalize_generated_image(
    image: Union[GeneratedImage, ImageInput, str, Path],
) -> Union[GeneratedImage, ImageInput, str]:
    """Normalize generated image input."""
    if isinstance(image, (GeneratedImage, ImageInput)):
        return image
    if isinstance(image, Path):
        return str(image)
    return image


# Context-local storage for the last score result
# Using ContextVar ensures thread-safety and async-safety
_last_score_result: ContextVar[Optional[EvalResult]] = ContextVar(
    "fasteval_last_score_result", default=None
)


def get_last_score_result() -> Optional[EvalResult]:
    """
    Get the most recent EvalResult from fe.score() in the current context.

    Used internally by the @conversation decorator to build history
    without requiring the test function to return the result.

    This is context-local (thread-safe and async-safe).

    Returns:
        The last EvalResult, or None if no score has been recorded
    """
    return _last_score_result.get()


def clear_last_score_result() -> None:
    """Clear the last score result in the current context."""
    _last_score_result.set(None)


def _has_fasteval_decorator(func: Any) -> bool:
    """Check if a function has any fasteval decorator attribute."""
    return hasattr(func, fasteval_METRICS_ATTR) or hasattr(
        func, fasteval_HUMAN_REVIEW_ATTR
    )


def _get_decorated_func_from_caller() -> Optional[Any]:
    """
    Walk the call stack to find the decorated test function.

    Returns the function object if found, None otherwise.
    """
    for frame_info in inspect.stack():
        frame = frame_info.frame
        func_name = frame_info.function

        # First, try direct lookup by function name
        func = frame.f_locals.get(func_name) or frame.f_globals.get(func_name)
        if func is not None and _has_fasteval_decorator(func):
            return func

        # For nested functions (defined inside test methods), scan all locals
        for name, val in frame.f_locals.items():
            if callable(val) and _has_fasteval_decorator(val):
                return val

    return None


def _get_metrics_from_caller() -> List[MetricConfig]:
    """
    Walk the call stack to find metrics from a decorated test function.

    Looks for functions with the _fasteval_metrics attribute set by
    decorators like @correctness, @relevance, etc.

    Returns:
        List of MetricConfig objects, or empty list if none found
    """
    func = _get_decorated_func_from_caller()
    if func is not None:
        return getattr(func, fasteval_METRICS_ATTR, [])
    return []


def _get_human_review_config_from_caller() -> Optional[Dict[str, Any]]:
    """
    Walk the call stack to find human review config from a decorated function.

    Returns:
        Human review config dict or None if not found
    """
    func = _get_decorated_func_from_caller()
    if func is not None and hasattr(func, fasteval_HUMAN_REVIEW_ATTR):
        return getattr(func, fasteval_HUMAN_REVIEW_ATTR)
    return None


def _get_test_name_from_caller() -> str:
    """Get the name of the calling test function."""
    for frame_info in inspect.stack():
        func_name = frame_info.function
        if func_name.startswith("test_"):
            return func_name
    return "unknown_test"


def score(
    actual_output: Optional[str] = None,
    expected_output: Optional[str] = None,
    *,
    input: Optional[str] = None,
    input_kwargs: Optional[Dict[str, Any]] = None,
    context: Optional[List[str]] = None,
    retrieval_context: Optional[List[str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None,
    expected_tools: Optional[List[Union[ExpectedTool, Dict[str, Any]]]] = None,
    # Multi-modal inputs
    image: Optional[Union[ImageInput, str, Path]] = None,
    images: Optional[List[Union[ImageInput, str, Path]]] = None,
    audio: Optional[Union[AudioInput, str, Path]] = None,
    audios: Optional[List[Union[AudioInput, str, Path]]] = None,
    generated_image: Optional[Union[GeneratedImage, ImageInput, str, Path]] = None,
    multimodal_context: Optional[List[MultiModalContext]] = None,
    expected_fields: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> EvalResult:
    """
    Evaluate actual output against expected output using decorated metrics.

    This function:
    1. Creates an EvalInput from the provided arguments
    2. Detects metrics from the calling function's decorators
    3. Runs evaluation immediately
    4. Raises EvaluationFailedError if any metric fails
    5. Returns the EvalResult on success

    Example:
        @fe.correctness(threshold=0.8)
        @fe.relevance(threshold=0.7)
        def test_qa():
            response = agent("What is 2+2?")
            result = fe.score(response, "4", input="What is 2+2?")
            # result.passed is True if all metrics passed
            # result.aggregate_score is the weighted average

    Tool Trajectory Example:
        from fasteval import ExpectedTool

        @fe.tool_call_accuracy(threshold=0.8)
        def test_agent():
            response, tool_calls = agent("Book flight to NYC")
            expected = [ExpectedTool(name="search_flights", args={"dest": "NYC"})]
            fe.score(response, tool_calls=tool_calls, expected_tools=expected)

    Multi-Modal Example:
        @fe.image_understanding(threshold=0.8)
        def test_vision():
            response = gpt4v.chat(image="chart.png", prompt="What trend?")
            fe.score(
                actual_output=response,
                expected_output="Revenue increased 25%",
                image="chart.png"
            )

    Image Generation Example:
        @fe.prompt_adherence(threshold=0.8)
        def test_dalle():
            image = dalle.generate("A red sports car")
            fe.score(
                generated_image=image,
                input="A red sports car"
            )

    Args:
        actual_output: The actual output from the LLM/agent
        expected_output: The expected/ground truth output
        input: The input that produced the output
        input_kwargs: Dict of input kwargs (for structured inputs)
        context: Context documents (for RAG evaluation)
        retrieval_context: Retrieved documents (for retrieval evaluation)
        history: Conversation history for multi-turn
        tool_calls: Actual tool calls made by the agent
        expected_tools: Expected tool calls for trajectory evaluation
        image: Single image input (path, URL, or ImageInput)
        images: Multiple images for evaluation
        audio: Single audio input (path, URL, or AudioInput)
        audios: Multiple audio files for evaluation
        generated_image: Generated image for image generation evaluation
        multimodal_context: Multi-modal context (text + images + audio)
        expected_fields: Expected fields for structured extraction (OCR, etc.)
        metadata: Additional metadata
        **kwargs: Extra fields for EvalInput

    Returns:
        EvalResult with all metric scores and pass/fail status

    Raises:
        EvaluationFailedError: If any metric fails its threshold
    """
    # Convert tool_calls to ToolCall models if needed
    normalized_tool_calls: List[ToolCall] = []
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, ToolCall):
                normalized_tool_calls.append(tc)
            elif isinstance(tc, dict):
                # Support both "args" and "arguments" keys for compatibility
                args = tc.get("args") or tc.get("arguments", {})
                normalized_tool_calls.append(
                    ToolCall(
                        name=tc.get("name", ""),
                        arguments=args,
                        result=tc.get("result"),
                        timestamp=tc.get("timestamp"),
                    )
                )

    # Convert expected_tools to ExpectedTool models if needed
    normalized_expected_tools: List[ExpectedTool] = []
    if expected_tools:
        for et in expected_tools:
            if isinstance(et, ExpectedTool):
                normalized_expected_tools.append(et)
            elif isinstance(et, dict):
                normalized_expected_tools.append(
                    ExpectedTool(
                        name=et.get("name", ""),
                        args=et.get("args", {}),
                        required=et.get("required", True),
                    )
                )

    # Normalize image inputs
    normalized_image = _normalize_image(image) if image else None
    normalized_images = [_normalize_image(img) for img in (images or [])]

    # Normalize audio inputs
    normalized_audio = _normalize_audio(audio) if audio else None
    normalized_audios = [_normalize_audio(aud) for aud in (audios or [])]

    # Normalize generated image
    normalized_generated_image = (
        _normalize_generated_image(generated_image) if generated_image else None
    )

    # Create the evaluation input
    eval_input = EvalInput(
        actual_output=actual_output,
        expected_output=expected_output,
        input=input,
        input_kwargs=input_kwargs,
        context=context,
        retrieval_context=retrieval_context,
        history=history,
        tool_calls=normalized_tool_calls,
        expected_tools=normalized_expected_tools,
        # Multi-modal fields
        image=normalized_image,
        images=normalized_images,
        audio=normalized_audio,
        audios=normalized_audios,
        generated_image=normalized_generated_image,
        multimodal_context=multimodal_context,
        expected_fields=expected_fields,
        metadata=metadata or {},
        **kwargs,
    )

    # Get metrics from the calling function's decorators
    metrics = _get_metrics_from_caller()

    # Get human review config if present
    human_review_config = _get_human_review_config_from_caller()

    if not metrics:
        # No metrics found - create a base result
        # This allows score() to be called without decorators for testing
        result = EvalResult(
            eval_input=eval_input,
            metric_results=[],
            passed=True,
            aggregate_score=1.0,
        )
    else:
        # Run automated evaluation immediately
        result = run_async(_evaluator.evaluate(eval_input, metrics))

    # Run human review if configured (after automated metrics, or as standalone)
    if human_review_config:
        result = _run_human_review(result, eval_input, human_review_config)

    # Store result for conversation decorator to access (context-local)
    _last_score_result.set(result)

    # Auto-collect result for reporting
    from fasteval.collectors.collector import get_collector

    test_name = _get_test_name_from_caller()
    get_collector().collect(result, test_name=test_name)

    # If evaluation failed, raise an exception with the formatted report
    if not result.passed:
        report = format_evaluation_report(test_name, [result], [eval_input])
        raise EvaluationFailedError(report, result)

    return result


def _run_human_review(
    result: EvalResult,
    eval_input: EvalInput,
    config: Dict[str, Any],
) -> EvalResult:
    """
    Run human review and integrate score into result.

    Args:
        result: The automated evaluation result
        eval_input: The evaluation input
        config: Human review config (prompt, required, threshold)

    Returns:
        Updated EvalResult with human review
    """
    prompt = config.get("prompt", "Please review this response")
    required = config.get("required", False)
    threshold = config.get("threshold", 0.5)

    # Prompt for human review
    human_score = prompt_human_review(
        prompt=prompt,
        input_text=eval_input.input,
        expected=eval_input.expected_output,
        actual=eval_input.actual_output,
        history=eval_input.history,
        metric_results=result.metric_results,
    )

    # Create human review result
    human_review_result = HumanReviewResult(
        score=human_score.score,
        raw_input=human_score.raw_input,
        skipped=human_score.skipped,
        prompt=prompt,
        timestamp=datetime.now().isoformat(),
    )

    # Determine if human review passes
    human_passed = True
    if human_score.skipped:
        if required:
            human_passed = False  # Required review was skipped
    else:
        human_passed = human_score.score >= threshold

    # Update result with human review
    # Create a new result with human review fields
    updated_result = EvalResult(
        eval_input=result.eval_input,
        metric_results=result.metric_results,
        passed=result.passed and human_passed,  # Both auto and human must pass
        aggregate_score=result.aggregate_score,
        execution_time_ms=result.execution_time_ms,
        error=result.error,
        reference_id=result.reference_id,
        human_review=human_review_result,
        human_review_required=required,
    )

    return updated_result
