"""Decorators for Langfuse integration.

Under pytest, ``@langfuse_traces`` and ``@langfuse_dataset`` expand into one
test case per trace / dataset item (via fasteval-core's dataset support), so
each trace passes or fails on its own, shows up individually in reports and
the fasteval UI, and can be re-run (``--lf``) or parallelized (``-n``).
Scores are pushed to Langfuse after every case, including failing ones.

Called directly (outside pytest), a decorated function evaluates every trace
in turn, pushes all scores, and raises once at the end if any trace failed.
"""

import functools
import hashlib
import inspect
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from fasteval.core.datasets import Case, Dataset
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

# Hidden per-case parameter carrying what the wrapper needs (trace id, dataset
# item, run name). Never passed to the user's function.
CASE_PARAM = "_fe_langfuse"

# How long xdist workers wait for the worker that fetches shared traces.
SHARED_FETCH_TIMEOUT_S = 300.0


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
    Fetch and evaluate traces from Langfuse, one test case per trace.

    Fetches traces (at pytest collection time), applies sampling, and runs the
    test function once per trace with trace data as parameters: trace_id,
    input, output, context, metadata. Declare the ones you need.

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

        # pytest test_production.py -v   ->  test_support_traces[<trace_id>] ...
    """
    query: Dict[str, Any] = {
        "project": project,
        "filter_tags": filter_tags,
        "time_range": time_range,
        "user_id": user_id,
        "session_id": session_id,
        "limit": limit,
    }

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        state: Dict[str, Any] = {}

        def client() -> LangfuseClient:
            if "client" not in state:
                state["client"] = LangfuseClient()
            return state["client"]

        def fetch() -> List[Dict[str, Any]]:
            """Fetched + sampled trace params (identical across xdist workers)."""

            def do_fetch() -> List[Dict[str, Any]]:
                fetcher = TraceFetcher(client())
                traces, total_count = fetcher.fetch_and_sample(
                    sampling=sampling, **query
                )
                if sampling:
                    stats = format_sampling_stats(
                        len(traces), total_count, sampling.name
                    )
                    print(f"\n{stats}\n")
                return [fetcher.map_trace_to_params(t) for t in traces]

            key = {
                "func": f"{func.__module__}.{func.__qualname__}",
                "sampling": sampling.name if sampling else None,
                **query,
            }
            return _shared_across_xdist_workers(key, do_fetch)

        def cases() -> Iterator[Any]:
            for params in fetch():
                # Case(...) rather than case(**...): trace/item fields may be
                # named "id" or "tags", which case() would treat as options.
                yield Case(
                    values={**params, CASE_PARAM: {"trace_id": params["trace_id"]}},
                    id=str(params["trace_id"]),
                )

        def push(trace_id: Any, result: Any) -> None:
            if result is None or auto_push_scores is False:
                return
            reporter = ScoreReporter(client())
            reporter.push_evaluation_result(
                trace_id=trace_id,
                metric_results=result.metric_results,
                aggregate_score=result.aggregate_score,
            )
            reporter.flush()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            payload = kwargs.pop(CASE_PARAM, None)
            if payload is None:
                # Called directly (not by pytest): evaluate every trace.
                _execute_trace_evaluation(
                    func=func,
                    is_async=is_async,
                    sampling=sampling,
                    auto_push_scores=auto_push_scores,
                    args=args,
                    kwargs=kwargs,
                    **query,
                )
                return
            _run_case(
                func,
                is_async,
                args,
                kwargs,
                on_result=lambda r: push(payload["trace_id"], r),
            )

        _finish(
            wrapper,
            func,
            Dataset(
                cases,
                name=func.__name__,
                info={
                    "type": "langfuse_traces",
                    "project": project,
                    "sampling": sampling.name if sampling else "NoSamplingStrategy",
                },
            ),
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
    """Evaluate every trace in one call (direct, non-pytest use).

    Every trace is evaluated and its scores pushed, even after a failure;
    scores are flushed; then the first failure is raised, listing them all.
    """
    client = LangfuseClient()
    fetcher = TraceFetcher(client)
    reporter = ScoreReporter(client)

    traces, total_count = fetcher.fetch_and_sample(
        project=project,
        filter_tags=filter_tags,
        time_range=time_range,
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        sampling=sampling,
    )

    if sampling:
        stats = format_sampling_stats(len(traces), total_count, sampling.name)
        print(f"\n{stats}\n")

    def push(trace_id: Any, result: Any) -> None:
        if result and auto_push_scores is not False:
            reporter.push_evaluation_result(
                trace_id=trace_id,
                metric_results=result.metric_results,
                aggregate_score=result.aggregate_score,
            )

    failures: List[tuple] = []
    try:
        for trace in traces:
            params = fetcher.map_trace_to_params(trace)
            try:
                _run_case(
                    func,
                    is_async,
                    args,
                    {**kwargs, **params},
                    on_result=functools.partial(push, params["trace_id"]),
                )
            except Exception as e:  # keep evaluating the remaining traces
                failures.append((params["trace_id"], e))
    finally:
        reporter.flush()
    _raise_failures(failures, len(traces), "traces")


def langfuse_dataset(
    name: str,
    version: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Fetch and evaluate dataset items from Langfuse, one test case per item.

    Similar to @fe.csv(), but fetches data from Langfuse datasets. Dataset
    fields (input, expected_output, item_id, and item metadata keys) are
    passed as parameters - declare what you need in the function signature.
    Every item is linked to one Langfuse experiment run (item.run()), so runs
    can be compared side by side in Langfuse.

    Args:
        name: Dataset name in Langfuse
        version: Optional dataset version

    Example:
        from fasteval_langfuse import langfuse_dataset
        import fasteval as fe

        @fe.correctness(threshold=0.8)
        @langfuse_dataset(name="qa-golden-set", version="v2")
        def test_qa_dataset(input, expected_output):
            response = my_agent(input)
            fe.score(response, expected_output, input=input)

        # With custom metadata fields (absent ones are passed as None)
        @fe.correctness(threshold=0.8)
        @langfuse_dataset(name="qa-golden-set", version="v2")
        def test_with_metadata(input, expected_output, user_type, complexity):
            response = my_agent(input)
            fe.score(response, expected_output, input=input)
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        state: Dict[str, Any] = {}

        def client() -> LangfuseClient:
            if "client" not in state:
                state["client"] = LangfuseClient()
            return state["client"]

        def cases() -> Iterator[Any]:
            raw_items = client().fetch_dataset_raw(name=name, version=version)
            run_name = _experiment_run_name(func.__name__)
            print(
                f"\nEvaluating {len(raw_items)} items from dataset '{name}' "
                f"(run: {run_name})\n"
            )
            rows = [(item, _dataset_item_params(item)) for item in raw_items]
            # Every case must provide every field some item has (metadata
            # keys vary between items); missing ones are passed as None.
            fields = sorted({k for _, params in rows for k in params})
            for index, (item, params) in enumerate(rows):
                values = {k: params.get(k) for k in fields}
                values[CASE_PARAM] = {"item": item, "run_name": run_name}
                yield Case(
                    values=values,
                    id=str(item.id if item.id is not None else f"{name}-{index}"),
                )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            payload = kwargs.pop(CASE_PARAM, None)
            if payload is None:
                _execute_dataset_evaluation(
                    func=func,
                    is_async=is_async,
                    name=name,
                    version=version,
                    args=args,
                    kwargs=kwargs,
                )
                return
            try:
                _run_dataset_item(
                    func, is_async, payload["item"], payload["run_name"], args, kwargs
                )
            finally:
                client().flush()

        _finish(
            wrapper,
            func,
            Dataset(
                cases,
                name=func.__name__,
                info={"type": "langfuse_dataset", "name": name, "version": version},
            ),
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
    """Evaluate every dataset item in one call (direct, non-pytest use).

    Uses item.run() to create linked traces in Langfuse, enabling the
    Experiments comparison UI for side-by-side run analysis. Every item is
    evaluated and scored even after a failure; failures are raised at the end.
    """
    client = LangfuseClient()
    raw_items = client.fetch_dataset_raw(name=name, version=version)
    run_name = _experiment_run_name(func.__name__)

    print(
        f"\nEvaluating {len(raw_items)} items from dataset '{name}' (run: {run_name})\n"
    )

    # Same as under pytest: fields only some items have are passed as None.
    fields = {k for item in raw_items for k in _dataset_item_params(item)}
    defaults = {k: None for k in fields}

    failures: List[tuple] = []
    try:
        for item in raw_items:
            try:
                _run_dataset_item(
                    func, is_async, item, run_name, args, kwargs, defaults
                )
            except Exception as e:
                failures.append((item.id, e))
    finally:
        client.flush()
    _raise_failures(failures, len(raw_items), "dataset items")


# ── helpers ──────────────────────────────────────────────────────────────────


def _finish(wrapper: Callable, func: Callable, dataset: Dataset) -> None:
    """Attach dataset/metric markers and the pytest-facing signature."""
    dataset(wrapper)  # sets _fasteval_dataset and _fasteval_data
    if hasattr(func, FASTEVAL_METRICS_ATTR):
        setattr(wrapper, FASTEVAL_METRICS_ATTR, getattr(func, FASTEVAL_METRICS_ATTR))
    # pytest sees the test's own parameters plus the hidden case parameter,
    # so dataset fields are parametrized and other params stay fixtures.
    sig = inspect.signature(func)
    hidden = inspect.Parameter(CASE_PARAM, inspect.Parameter.KEYWORD_ONLY)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    setattr(wrapper, "__signature__", sig.replace(parameters=[*params, hidden]))


def _run_case(
    func: Callable,
    is_async: bool,
    args: tuple,
    kwargs: dict,
    on_result: Callable[[Any], None],
) -> None:
    """Call the test function; hand its fe.score() result to on_result even if it failed."""
    from fasteval.core.scoring import clear_last_score_result, get_last_score_result

    clear_last_score_result()
    if not is_async:
        try:
            func(*args, **_accepted(func, kwargs))
        finally:
            on_result(get_last_score_result())
        return

    # fe.score() records its result in a context variable. The coroutine runs
    # in its own asyncio task with a *copy* of the context, so read the result
    # inside that task; reading it afterwards here would always see None.
    from fasteval.utils import run_async

    captured: Dict[str, Any] = {}

    async def run_and_capture() -> None:
        try:
            await func(*args, **_accepted(func, kwargs))
        finally:
            captured["result"] = get_last_score_result()

    try:
        run_async(run_and_capture())
    finally:
        on_result(captured.get("result"))


def _run_dataset_item(
    func: Callable,
    is_async: bool,
    item: Any,
    run_name: str,
    args: tuple,
    kwargs: dict,
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    config = get_config()
    params = {**(defaults or {}), **_dataset_item_params(item)}
    with item.run(run_name=run_name) as root_span:

        def score(result: Any) -> None:
            if not result:
                return
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

        _run_case(func, is_async, args, {**params, **kwargs}, on_result=score)


def _dataset_item_params(item: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if item.input is not None:
        params["input"] = item.input
    if item.expected_output is not None:
        params["expected_output"] = item.expected_output
    if item.id is not None:
        params["item_id"] = item.id
    metadata = getattr(item, "metadata", None) or {}
    if isinstance(metadata, dict):
        params.update(metadata)
    return params


def _accepted(func: Callable, kwargs: dict) -> dict:
    """Keep only kwargs the function accepts (all of them with **kwargs)."""
    params = inspect.signature(func).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _raise_failures(failures: List[tuple], total: int, what: str) -> None:
    if not failures:
        return
    ids = ", ".join(str(i) for i, _ in failures[:10])
    more = f" (+{len(failures) - 10} more)" if len(failures) > 10 else ""
    raise AssertionError(
        f"{len(failures)} of {total} {what} failed: {ids}{more}"
    ) from failures[0][1]


def _experiment_run_name(func_name: str) -> str:
    """One Langfuse experiment run per test session, shared by xdist workers."""
    run = os.environ.get("FASTEVAL_RUN_ID") or os.environ.get("PYTEST_XDIST_TESTRUNUID")
    return f"fasteval-{func_name}-{run or datetime.now().isoformat()}"


def _shared_across_xdist_workers(
    key: Dict[str, Any], produce: Callable[[], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Under pytest-xdist every worker collects separately, and all must collect
    the same cases. With relative time windows or unseeded sampling, separate
    fetches can differ, so the first worker fetches and the rest reuse it.
    Outside xdist this is a plain call.
    """
    run_uid = os.environ.get("PYTEST_XDIST_TESTRUNUID")
    if not run_uid:
        return produce()
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    base = os.path.join(
        tempfile.gettempdir(), f"fasteval-langfuse-{run_uid[:16]}-{digest}"
    )
    data_path, lock_path = base + ".json", base + ".lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        deadline = time.monotonic() + SHARED_FETCH_TIMEOUT_S
        while not os.path.exists(data_path):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for another pytest-xdist worker to fetch "
                    f"Langfuse traces ({data_path})"
                )
            time.sleep(0.1)
        with open(data_path, encoding="utf-8") as f:
            result: List[Dict[str, Any]] = json.load(f)
        return result
    os.close(fd)
    produced = produce()
    tmp = data_path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(produced, f, default=str)
    os.replace(tmp, data_path)
    # Workers that load the file see JSON-normalized values; reload it here
    # too so this worker's cases are identical to theirs.
    with open(data_path, encoding="utf-8") as f:
        result = json.load(f)
    return result
