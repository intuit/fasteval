"""The @observe decorator for runtime monitoring."""

import asyncio
import functools
import inspect
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from fasteval_observe.async_handler import get_observation_queue
from fasteval_observe.config import get_config
from fasteval_observe.logger import log_observations
from fasteval_observe.metrics import (
    EvaluationMetrics,
    Observation,
    ObservationMetrics,
)
from fasteval_observe.sampling.base import BaseSamplingStrategy
from fasteval_observe.sampling.strategies import NoSamplingStrategy

# Internal logger for debug/error messages (not observations)
logger = logging.getLogger("fasteval_observe.internal")

F = TypeVar("F", bound=Callable[..., Any])

# Context variables for distributed tracing
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)

# Marker attribute for fasteval metrics (from core fasteval)
FASTEVAL_METRICS_ATTR = "_fasteval_metrics"


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    return _trace_id.get()


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID for the current context."""
    _trace_id.set(trace_id)


def get_span_id() -> Optional[str]:
    """Get the current span ID."""
    return _span_id.get()


def set_span_id(span_id: str) -> None:
    """Set the span ID for the current context."""
    _span_id.set(span_id)


def _generate_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())


def _generate_span_id() -> str:
    """Generate a new span ID."""
    return str(uuid.uuid4())[:16]


def _extract_args_for_logging(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> Dict[str, Any]:
    """Extract function arguments as a dictionary for logging."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _build_context(
    func: Callable,
    args: tuple,
    kwargs: dict,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build context dictionary for sampling decision."""
    return {
        "trace_id": get_trace_id(),
        "span_id": get_span_id(),
        "metadata": metadata,
        "function_name": func.__name__,
        "module": func.__module__,
    }


def _build_input_string(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Build a string representation of function inputs for evaluation.

    Extracts all arguments and formats them as a readable input string.
    Long values (>500 chars) are truncated to keep the input manageable.
    """
    MAX_VALUE_LENGTH = 500

    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Build input string from all arguments
        input_parts = []
        for param_name, value in bound.arguments.items():
            str_val = str(value)
            # Truncate long values
            if len(str_val) > MAX_VALUE_LENGTH:
                str_val = str_val[:MAX_VALUE_LENGTH] + "..."
            input_parts.append(f"{param_name}: {str_val}")

        return "\n".join(input_parts) if input_parts else ""
    except Exception:
        # Fallback to simple representation
        return f"args: {args}, kwargs: {kwargs}"


async def _run_fasteval_evaluations(
    func: Callable,
    result: Any,
    args: tuple,
    kwargs: dict,
    context_fn: Optional[Callable[[tuple, dict], Optional[Union[str, List[str]]]]] = None,
) -> Optional[EvaluationMetrics]:
    """
    Run fasteval evaluations if the function has @fe.* decorators.

    Uses all function arguments as input and the return value as output.
    Optionally extracts context using context_fn callback for RAG metrics.

    This is called in the background for sampled calls.

    Args:
        func: The decorated function
        result: Function return value
        args: Positional arguments passed to function
        kwargs: Keyword arguments passed to function
        context_fn: Optional callback to extract context from args/kwargs
    """
    try:
        # Check if function has fasteval metrics
        if not hasattr(func, FASTEVAL_METRICS_ATTR):
            return None

        metrics_config = getattr(func, FASTEVAL_METRICS_ATTR)
        if not metrics_config:
            return None

        # Import fasteval components lazily
        from fasteval.core.evaluator import Evaluator
        from fasteval.models.evaluation import EvalInput

        evaluator = Evaluator()

        # Build actual_output from result
        if result is None:
            actual_output = None
        elif isinstance(result, str):
            actual_output = result
        elif hasattr(result, "model_dump"):
            # Pydantic model
            actual_output = str(result.model_dump())
        else:
            actual_output = str(result)

        # Build input from all function arguments
        input_str = _build_input_string(func, args, kwargs)

        # Extract context using callback if provided
        context = None
        if context_fn:
            try:
                extracted = context_fn(args, kwargs)
                if extracted is not None:
                    # Normalize to list of strings
                    if isinstance(extracted, list):
                        context = [str(item) for item in extracted]
                    elif isinstance(extracted, str):
                        context = [extracted]
                    else:
                        context = [str(extracted)]
            except Exception as e:
                logger.warning(f"context_fn failed: {e}")
                context = None

        # Build EvalInput with extracted context
        eval_input = EvalInput(
            actual_output=actual_output,
            input=input_str,
            input_kwargs=dict(kwargs),
            context=context,
        )

        # Run evaluation
        eval_result = await evaluator.evaluate(eval_input, metrics_config)

        return EvaluationMetrics(
            metrics_evaluated=[mr.metric_name for mr in eval_result.metric_results],
            aggregate_score=eval_result.aggregate_score,
            passed=eval_result.passed,
            metric_scores={
                mr.metric_name: mr.score for mr in eval_result.metric_results
            },
        )

    except ImportError:
        logger.debug("fasteval not available, skipping evaluations")
        return None
    except Exception as e:
        logger.warning(f"Failed to run fasteval evaluations: {e}")
        return None


def observe(
    sampling: Optional[BaseSamplingStrategy] = None,
    run_evaluations: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    log_inputs: Optional[bool] = None,
    log_outputs: Optional[bool] = None,
    context_fn: Optional[Callable[[tuple, dict], Optional[Union[str, List[str]]]]] = None,
) -> Callable[[F], F]:
    """
    Decorator for runtime monitoring of agent functions.

    Collects execution metrics and optionally runs fasteval evaluations
    on sampled calls. Metrics are logged asynchronously to avoid
    impacting function latency.

    Args:
        sampling: Sampling strategy instance (default: NoSamplingStrategy)
        run_evaluations: If True, run any @fe.* metrics in background
        metadata: Additional metadata to include in observations
        log_inputs: Override config.include_inputs for this function
        log_outputs: Override config.include_outputs for this function
        context_fn: Optional callback to extract context from function args/kwargs.
                   Receives (args, kwargs) and should return List[str], str, or None.
                   Used for RAG metrics like faithfulness that require context.
                   Example: context_fn=lambda args, kwargs: kwargs.get("docs")

    Example:
        from fasteval_observe import observe
        from fasteval_observe.sampling import FixedRateSamplingStrategy

        @observe(sampling=FixedRateSamplingStrategy(rate=0.05))
        async def my_agent(query: str) -> str:
            return await llm.invoke(query)

        # With fasteval evaluation
        import fasteval as fe

        @observe(
            sampling=FixedRateSamplingStrategy(rate=0.05),
            run_evaluations=True
        )
        @fe.correctness(threshold=0.8)
        async def evaluated_agent(query: str) -> str:
            return await llm.invoke(query)

        # With context extraction for RAG metrics
        @observe(
            sampling=FixedRateSamplingStrategy(rate=0.05),
            run_evaluations=True,
            context_fn=lambda args, kwargs: kwargs.get("docs")
        )
        @fe.faithfulness(threshold=0.85)
        async def rag_agent(query: str, docs: list[str]) -> str:
            return await llm.invoke(query, context=docs)
    """
    if sampling is None:
        sampling = NoSamplingStrategy()

    if metadata is None:
        metadata = {}

    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _execute_with_observation(
                func=func,
                args=args,
                kwargs=kwargs,
                sampling=sampling,
                run_evaluations=run_evaluations,
                metadata=metadata,
                log_inputs=log_inputs,
                log_outputs=log_outputs,
                is_async=True,
                context_fn=context_fn,
            )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Run in event loop for consistency
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    _execute_with_observation(
                        func=func,
                        args=args,
                        kwargs=kwargs,
                        sampling=sampling,
                        run_evaluations=run_evaluations,
                        metadata=metadata,
                        log_inputs=log_inputs,
                        log_outputs=log_outputs,
                        is_async=False,
                        context_fn=context_fn,
                    )
                )
            finally:
                loop.close()

        wrapper = async_wrapper if is_async else sync_wrapper

        # Preserve fasteval decorators
        if hasattr(func, FASTEVAL_METRICS_ATTR):
            setattr(wrapper, FASTEVAL_METRICS_ATTR, getattr(func, FASTEVAL_METRICS_ATTR))

        return wrapper  # type: ignore

    return decorator


async def _execute_with_observation(
    func: Callable,
    args: tuple,
    kwargs: dict,
    sampling: BaseSamplingStrategy,
    run_evaluations: bool,
    metadata: Dict[str, Any],
    log_inputs: Optional[bool],
    log_outputs: Optional[bool],
    is_async: bool,
    context_fn: Optional[Callable[[tuple, dict], Optional[Union[str, List[str]]]]] = None,
) -> Any:
    """Execute function with observation tracking."""
    config = get_config()

    # Quick exit if disabled
    if not config.enabled:
        if is_async:
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    # Build context for sampling decision
    context = _build_context(func, args, kwargs, metadata)

    # Sampling decision (fast path)
    should_sample = sampling.should_sample(func.__name__, args, kwargs, context)

    if not should_sample:
        # Execute without observation overhead
        start = time.perf_counter()
        try:
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            execution_time_ms = (time.perf_counter() - start) * 1000
            sampling.on_completion(func.__name__, execution_time_ms, None, result)
            return result
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start) * 1000
            sampling.on_completion(func.__name__, execution_time_ms, e, None)
            raise

    # Sampled execution - full observation
    trace_id = get_trace_id() or _generate_trace_id()
    span_id = _generate_span_id()
    parent_span_id = get_span_id()

    # Set context for nested calls
    set_trace_id(trace_id)
    set_span_id(span_id)

    start = time.perf_counter()
    error: Optional[Exception] = None
    result: Any = None

    try:
        if is_async:
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
    except Exception as e:
        error = e
        raise
    finally:
        execution_time_ms = (time.perf_counter() - start) * 1000

        # Notify sampling strategy
        sampling.on_completion(func.__name__, execution_time_ms, error, result)

        # Build observation
        observation = _build_observation(
            func=func,
            args=args,
            kwargs=kwargs,
            result=result,
            error=error,
            execution_time_ms=execution_time_ms,
            sampling=sampling,
            metadata=metadata,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            log_inputs=log_inputs,
            log_outputs=log_outputs,
        )

        # Run evaluations if requested (async, non-blocking)
        if run_evaluations and error is None:
            try:
                eval_results = await _run_fasteval_evaluations(
                    func, result, args, kwargs, context_fn
                )
                if eval_results:
                    observation.evaluation_results = eval_results
            except Exception as eval_error:
                logger.debug(f"Evaluation failed: {eval_error}")

        # Enqueue observation (non-blocking)
        queue = get_observation_queue()
        queue.enqueue(observation)

    return result


def _build_observation(
    func: Callable,
    args: tuple,
    kwargs: dict,
    result: Any,
    error: Optional[Exception],
    execution_time_ms: float,
    sampling: BaseSamplingStrategy,
    metadata: Dict[str, Any],
    trace_id: str,
    span_id: str,
    parent_span_id: Optional[str],
    log_inputs: Optional[bool],
    log_outputs: Optional[bool],
) -> Observation:
    """Build an Observation from execution data."""
    config = get_config()

    # Determine whether to log inputs/outputs
    should_log_inputs = log_inputs if log_inputs is not None else config.include_inputs
    should_log_outputs = (
        log_outputs if log_outputs is not None else config.include_outputs
    )

    # Build metrics
    metrics = ObservationMetrics(
        latency_ms=execution_time_ms,
        success=error is None,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
    )

    # Build input data if requested
    input_data = None
    if should_log_inputs:
        try:
            input_data = _extract_args_for_logging(func, args, kwargs)
        except Exception:
            input_data = {"args": str(args), "kwargs": str(kwargs)}

    # Build output data if requested
    output_data = None
    if should_log_outputs and error is None:
        try:
            # Try to serialize result, fall back to string
            if hasattr(result, "model_dump"):
                output_data = result.model_dump()
            else:
                output_data = str(result)
        except Exception:
            output_data = str(result)

    return Observation(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        function_name=func.__name__,
        function_module=func.__module__,
        sampling_strategy=sampling.name,
        metrics=metrics,
        metadata=metadata,
        input_data=input_data,
        output_data=output_data,
    )


def initialize_observer() -> None:
    """
    Initialize the observer system.

    Call this at application startup to start the background worker.
    """
    queue = get_observation_queue()

    # Set up flush callback to log observations
    queue.set_flush_callback(log_observations)

    # Start worker
    queue.start_worker()

    logger.info("fasteval-observe initialized")


def shutdown_observer() -> None:
    """
    Gracefully shutdown the observer system.

    Call this at application shutdown to flush pending observations.
    """
    queue = get_observation_queue()
    queue.shutdown()

    logger.info("fasteval-observe shutdown complete")
