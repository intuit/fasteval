"""Decorators for Langfuse integration."""

import functools
import inspect
from datetime import datetime
from typing import Any, Callable, List, Optional, TypeVar

from fasteval_langfuse.client import LangfuseClient
from fasteval_langfuse.config import get_config
from fasteval_langfuse.sampling.base import BaseSamplingStrategy
from fasteval_langfuse.score_reporter import ScoreReporter
from fasteval_langfuse.trace_fetcher import TraceFetcher
from fasteval_langfuse.utils import format_sampling_stats

F = TypeVar("F", bound=Callable[..., Any])

# Marker attributes from fasteval core
FASTEVAL_METRICS_ATTR = "_fasteval_metrics"
FASTEVAL_DATA_ATTR = "_fasteval_data"


def langfuse_traces(
    project: Optional[str] = None,
    filter_tags: Optional[List[str]] = None,
    time_range: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
    sampling: Optional[BaseSamplingStrategy] = None,
    auto_push_scores: Optional[bool] = None,
) -> Callable[[F], F]:
    """
    Fetch and evaluate traces from Langfuse.

    Fetches traces, applies sampling, and calls the test function once per trace
    with trace data as parameters: trace_id, input, output, context, metadata.

    Args:
        project: Project name to filter traces
        filter_tags: Tags to filter by
        time_range: Time range (e.g., "last_24h", "last_7d")
        user_id: Filter by user ID
        session_id: Filter by session ID
        limit: Max traces to fetch before sampling
        sampling: Sampling strategy (default: evaluate all)
        auto_push_scores: Override config auto_push_scores

    Example:
        from fasteval_langfuse import langfuse_traces
        from fasteval_langfuse.sampling import RandomSamplingStrategy
        import fasteval as fe

        @fe.correctness(threshold=0.8)
        @langfuse_traces(
            project="production",
            filter_tags=["customer-support"],
            time_range="last_24h",
            sampling=RandomSamplingStrategy(sample_size=200)
        )
        def test_support_traces(trace_id, input, output, context, metadata):
            fe.score(output, input=input)

        # Run with pytest:
        # pytest test_production.py -v
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> None:
            _execute_trace_evaluation(
                func=func,
                is_async=False,
                project=project,
                filter_tags=filter_tags,
                time_range=time_range,
                user_id=user_id,
                session_id=session_id,
                limit=limit,
                sampling=sampling,
                auto_push_scores=auto_push_scores,
                args=args,
                kwargs=kwargs,
            )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> None:
            _execute_trace_evaluation(
                func=func,
                is_async=True,
                project=project,
                filter_tags=filter_tags,
                time_range=time_range,
                user_id=user_id,
                session_id=session_id,
                limit=limit,
                sampling=sampling,
                auto_push_scores=auto_push_scores,
                args=args,
                kwargs=kwargs,
            )

        wrapper = async_wrapper if is_async else sync_wrapper

        # Preserve fasteval metrics
        if hasattr(func, FASTEVAL_METRICS_ATTR):
            setattr(
                wrapper, FASTEVAL_METRICS_ATTR, getattr(func, FASTEVAL_METRICS_ATTR)
            )

        # Mark as data-decorated
        setattr(
            wrapper,
            FASTEVAL_DATA_ATTR,
            {
                "type": "langfuse_traces",
                "project": project,
                "sampling": sampling.name if sampling else "NoSamplingStrategy",
            },
        )

        return wrapper  # type: ignore

    return decorator


def _execute_trace_evaluation(
    func: Callable,
    is_async: bool,
    project: Optional[str],
    filter_tags: Optional[List[str]],
    time_range: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    limit: Optional[int],
    sampling: Optional[BaseSamplingStrategy],
    auto_push_scores: Optional[bool],
    args: tuple,
    kwargs: dict,
) -> None:
    """Execute trace evaluation (internal implementation)."""
    # Initialize components
    client = LangfuseClient()
    fetcher = TraceFetcher(client)
    reporter = ScoreReporter(client)

    # Fetch and sample traces
    traces, total_count = fetcher.fetch_and_sample(
        project=project,
        filter_tags=filter_tags,
        time_range=time_range,
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        sampling=sampling,
    )

    # Print sampling stats
    if sampling:
        stats = format_sampling_stats(len(traces), total_count, sampling.name)
        print(f"\n{stats}\n")

    # Import here to avoid circular dependency
    from fasteval.core.scoring import get_last_score_result

    # Evaluate each trace
    for trace in traces:
        # Map trace to function parameters
        params = fetcher.map_trace_to_params(trace)

        # Call the test function
        if is_async:
            from fasteval.utils import run_async

            run_async(func(*args, **{**kwargs, **params}))
        else:
            func(*args, **{**kwargs, **params})

        # Get evaluation result
        result = get_last_score_result()

        # Push scores back to Langfuse
        if result and auto_push_scores is not False:
            reporter.push_evaluation_result(
                trace_id=params["trace_id"],
                metric_results=result.metric_results,
                aggregate_score=result.aggregate_score,
            )

    # Flush scores
    reporter.flush()


def langfuse_dataset(
    name: str,
    version: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Fetch and evaluate dataset items from Langfuse.

    Similar to @fe.csv(), but fetches data from Langfuse datasets.
    All dataset columns are passed as kwargs - declare what you need in the function signature.

    Args:
        name: Dataset name in Langfuse
        version: Optional dataset version

    Example:
        from fasteval_langfuse import langfuse_dataset
        import fasteval as fe

        # Basic usage - standard fields
        @fe.correctness(threshold=0.8)
        @langfuse_dataset(name="qa-golden-set", version="v2")
        def test_qa_dataset(input, expected_output):
            response = my_agent(input)
            fe.score(response, expected_output, input=input)

        # With custom metadata fields
        @fe.correctness(threshold=0.8)
        @langfuse_dataset(name="qa-golden-set", version="v2")
        def test_with_metadata(input, expected_output, user_type, complexity):
            # user_type and complexity come from item.metadata
            response = my_agent(input)
            fe.score(response, expected_output, input=input)

        # Only what you need
        @fe.correctness(threshold=0.8)
        @langfuse_dataset(name="qa-golden-set")
        def test_minimal(input):
            # Only declare input, ignore other fields
            response = my_agent(input)
            fe.score(response, input=input)
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> None:
            _execute_dataset_evaluation(
                func=func,
                is_async=False,
                name=name,
                version=version,
                args=args,
                kwargs=kwargs,
            )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> None:
            _execute_dataset_evaluation(
                func=func,
                is_async=True,
                name=name,
                version=version,
                args=args,
                kwargs=kwargs,
            )

        wrapper = async_wrapper if is_async else sync_wrapper

        # Preserve fasteval metrics
        if hasattr(func, FASTEVAL_METRICS_ATTR):
            setattr(
                wrapper, FASTEVAL_METRICS_ATTR, getattr(func, FASTEVAL_METRICS_ATTR)
            )

        # Mark as data-decorated
        setattr(
            wrapper,
            FASTEVAL_DATA_ATTR,
            {"type": "langfuse_dataset", "name": name, "version": version},
        )

        return wrapper  # type: ignore

    return decorator


def _execute_dataset_evaluation(
    func: Callable,
    is_async: bool,
    name: str,
    version: Optional[str],
    args: tuple,
    kwargs: dict,
) -> None:
    """Execute dataset evaluation with Langfuse Experiments integration.

    Uses item.run() to create linked traces in Langfuse, enabling the
    Experiments comparison UI for side-by-side run analysis.
    """
    from fasteval.core.scoring import get_last_score_result

    client = LangfuseClient()
    config = get_config()
    raw_items = client.fetch_dataset_raw(name=name, version=version)
    run_name = f"fasteval-{func.__name__}-{datetime.now().isoformat()}"

    print(f"\nEvaluating {len(raw_items)} items from dataset '{name}' (run: {run_name})\n")

    for item in raw_items:
        with item.run(run_name=run_name) as root_span:
            params: dict[str, Any] = {}

            if item.input is not None:
                params["input"] = item.input
            if item.expected_output is not None:
                params["expected_output"] = item.expected_output
            if item.id is not None:
                params["item_id"] = item.id

            metadata = getattr(item, "metadata", None) or {}
            if metadata:
                params.update(metadata)

            if is_async:
                from fasteval.utils import run_async

                run_async(func(*args, **{**kwargs, **params}))
            else:
                func(*args, **{**kwargs, **params})
            result = get_last_score_result()
            if result:
                for mr in result.metric_results:
                    root_span.score_trace(
                        name=f"{config.score_name_prefix}{mr.metric_name}",
                        value=mr.score,
                        comment=getattr(mr, "reasoning", None),
                    )
                root_span.score_trace(
                    name=f"{config.score_name_prefix}aggregate",
                    value=result.aggregate_score,
                )

    client.flush()
