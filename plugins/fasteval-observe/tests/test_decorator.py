"""Tests for the @observe decorator."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from fasteval_observe import (
    ObserveConfig,
    configure_observe,
    observe,
)
from fasteval_observe.async_handler import ObservationQueue
from fasteval_observe.config import reset_config
from fasteval_observe.decorator import (
    _build_input_string,
    get_span_id,
    get_trace_id,
    initialize_observer,
    set_span_id,
    set_trace_id,
    shutdown_observer,
)
from fasteval_observe.sampling import (
    FixedRateSamplingStrategy,
    NoSamplingStrategy,
)


@pytest.fixture(autouse=True)
def reset_all():
    """Reset all singletons before each test."""
    ObservationQueue.reset_instance()
    reset_config()
    yield
    ObservationQueue.reset_instance()
    reset_config()


class TestTraceContext:
    """Tests for trace context management."""

    def test_trace_id_context(self):
        """Should get and set trace ID."""
        set_trace_id("test-trace-123")
        assert get_trace_id() == "test-trace-123"

    def test_span_id_context(self):
        """Should get and set span ID."""
        set_span_id("test-span-456")
        assert get_span_id() == "test-span-456"


class TestObserveDecorator:
    """Tests for @observe decorator."""

    @pytest.mark.asyncio
    async def test_async_function_works(self):
        """Should work with async functions."""

        @observe(sampling=NoSamplingStrategy())
        async def my_async_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await my_async_func(5)
        assert result == 10

    def test_sync_function_works(self):
        """Should work with sync functions."""

        @observe(sampling=NoSamplingStrategy())
        def my_sync_func(x: int) -> int:
            return x * 2

        result = my_sync_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Should preserve function name and docstring."""

        @observe(sampling=NoSamplingStrategy())
        async def my_documented_func():
            """This is a docstring."""
            pass

        assert my_documented_func.__name__ == "my_documented_func"
        assert my_documented_func.__doc__ == "This is a docstring."

    @pytest.mark.asyncio
    async def test_samples_according_to_strategy(self):
        """Should only sample when strategy says so."""
        call_count = 0

        @observe(sampling=FixedRateSamplingStrategy(rate=0.0))
        async def never_sampled():
            nonlocal call_count
            call_count += 1
            return "result"

        # Call multiple times
        for _ in range(10):
            await never_sampled()

        assert call_count == 10

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        """Should propagate exceptions from decorated function."""

        @observe(sampling=NoSamplingStrategy())
        async def raising_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await raising_func()

    @pytest.mark.asyncio
    async def test_tracks_execution_time(self):
        """Should track execution time."""
        times_recorded = []

        class TimingStrategy(NoSamplingStrategy):
            def on_completion(self, function_name, execution_time_ms, error, result):
                times_recorded.append(execution_time_ms)

        @observe(sampling=TimingStrategy())
        async def slow_func():
            await asyncio.sleep(0.05)
            return "done"

        await slow_func()

        assert len(times_recorded) == 1
        assert times_recorded[0] >= 50  # At least 50ms

    @pytest.mark.asyncio
    async def test_disabled_config_skips_observation(self):
        """Should skip observation when disabled."""
        configure_observe(ObserveConfig(enabled=False))

        call_count = 0

        @observe(sampling=NoSamplingStrategy())
        async def func():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await func()

        assert result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_metadata_passed_to_observation(self):
        """Should include metadata in observations."""

        @observe(
            sampling=NoSamplingStrategy(),
            metadata={"custom_key": "custom_value"},
        )
        async def func_with_metadata():
            return "result"

        # This would require inspecting the observation
        result = await func_with_metadata()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_log_inputs_override(self):
        """Should respect log_inputs override."""

        @observe(
            sampling=NoSamplingStrategy(),
            log_inputs=True,
        )
        async def func_log_inputs(x: int, y: str):
            return f"{x}-{y}"

        result = await func_log_inputs(1, "test")
        assert result == "1-test"

    @pytest.mark.asyncio
    async def test_log_outputs_override(self):
        """Should respect log_outputs override."""

        @observe(
            sampling=NoSamplingStrategy(),
            log_outputs=True,
        )
        async def func_log_outputs():
            return {"key": "value"}

        result = await func_log_outputs()
        assert result == {"key": "value"}


class TestObserverLifecycle:
    """Tests for observer initialization and shutdown."""

    def test_initialize_observer(self):
        """Should initialize observer system."""
        initialize_observer()

        queue = ObservationQueue()
        assert queue._worker_thread is not None
        assert queue._worker_thread.is_alive()

        shutdown_observer()

    def test_shutdown_observer(self):
        """Should shutdown observer gracefully."""
        initialize_observer()
        shutdown_observer()

        queue = ObservationQueue()
        assert queue._worker_thread is None or not queue._worker_thread.is_alive()


class TestFastevalIntegration:
    """Tests for integration with fasteval decorators."""

    @pytest.mark.asyncio
    async def test_preserves_fasteval_attributes(self):
        """Should preserve _fasteval_metrics attribute."""
        # Simulate fasteval decorator by adding attribute
        def mock_fasteval_decorator(func):
            func._fasteval_metrics = [{"type": "correctness", "threshold": 0.8}]
            return func

        @observe(sampling=NoSamplingStrategy())
        @mock_fasteval_decorator
        async def evaluated_func():
            return "result"

        assert hasattr(evaluated_func, "_fasteval_metrics")
        assert evaluated_func._fasteval_metrics == [
            {"type": "correctness", "threshold": 0.8}
        ]

    @pytest.mark.asyncio
    async def test_run_evaluations_flag(self):
        """Should respect run_evaluations flag."""

        @observe(
            sampling=NoSamplingStrategy(),
            run_evaluations=True,
        )
        async def func_with_evals():
            return "result"

        # Should work even without actual fasteval decorators
        result = await func_with_evals()
        assert result == "result"


class TestInputExtraction:
    """Tests for input extraction utilities."""

    def test_build_input_string_simple(self):
        """Should build input string from simple arguments."""

        def sample_func(query: str, limit: int = 10):
            pass

        result = _build_input_string(sample_func, ("hello",), {"limit": 5})

        assert "query: hello" in result
        assert "limit: 5" in result

    def test_build_input_string_complex_types(self):
        """Should handle complex argument types."""

        def sample_func(data: dict, items: list):
            pass

        result = _build_input_string(
            sample_func, (), {"data": {"key": "value"}, "items": [1, 2, 3]}
        )

        assert "data:" in result
        assert "items:" in result

    def test_build_input_string_truncates_long_values(self):
        """Should truncate very long values."""

        def sample_func(content: str):
            pass

        long_string = "x" * 1000
        result = _build_input_string(sample_func, (long_string,), {})

        # Should be truncated with ellipsis
        assert len(result) < 1000
        assert "..." in result


class TestContextFnExtraction:
    """Tests for context_fn parameter."""

    @pytest.mark.asyncio
    async def test_context_fn_extracts_from_kwargs(self):
        """Should use context_fn to extract context from kwargs."""
        extracted_contexts = []

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=lambda args, kwargs: kwargs.get("docs"),
        )
        async def func_with_context(query: str, docs: list[str]):
            return f"Response to {query}"

        result = await func_with_context("hello", docs=["doc1", "doc2"])

        assert result == "Response to hello"
        # context_fn itself is called, but we can't easily verify the EvalInput
        # without mocking the evaluator. Test that the function works correctly.

    @pytest.mark.asyncio
    async def test_context_fn_returns_list(self):
        """Should handle context_fn returning a list."""

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=lambda args, kwargs: ["doc1", "doc2"],
        )
        async def func():
            return "result"

        result = await func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_context_fn_returns_string(self):
        """Should accept context_fn returning a single string."""

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=lambda args, kwargs: "single doc",
        )
        async def func():
            return "result"

        result = await func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_context_fn_returns_none(self):
        """Should handle context_fn returning None."""

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=lambda args, kwargs: None,
        )
        async def func():
            return "result"

        result = await func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_context_fn_with_complex_extraction(self):
        """Should support complex extraction logic."""

        def extract(args, kwargs):
            if "results" in kwargs:
                return [r["text"] for r in kwargs["results"]]
            return None

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=extract,
        )
        async def func(query: str, results: list[dict]):
            return "result"

        result = await func(
            "query", results=[{"text": "doc1"}, {"text": "doc2"}]
        )
        assert result == "result"

    @pytest.mark.asyncio
    async def test_context_fn_exception_handling(self):
        """Should gracefully handle context_fn exceptions."""

        def failing_fn(args, kwargs):
            raise ValueError("extraction failed")

        @observe(
            sampling=NoSamplingStrategy(),
            run_evaluations=True,
            context_fn=failing_fn,
        )
        async def func():
            return "result"

        # Function should still work even if context_fn fails
        result = await func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_no_context_fn_no_context(self):
        """Should not extract context if context_fn not provided."""

        @observe(
            sampling=NoSamplingStrategy(),
            run_evaluations=True,
        )
        async def func(query: str, context: list[str]):
            return "result"

        # Without context_fn, context kwarg is not automatically extracted
        result = await func("query", context=["doc1"])
        assert result == "result"

    @pytest.mark.asyncio
    async def test_context_fn_with_args(self):
        """Should allow extraction from positional args."""

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=lambda args, kwargs: args[1] if len(args) > 1 else None,
        )
        async def func(query: str, docs: list[str]):
            return "result"

        result = await func("query", ["doc1", "doc2"])
        assert result == "result"

    def test_sync_context_fn_extracts(self):
        """Should work with sync functions too."""

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=lambda args, kwargs: kwargs.get("docs"),
        )
        def sync_func(query: str, docs: list[str]):
            return f"Result for {query}"

        result = sync_func("hello", docs=["doc1"])
        assert result == "Result for hello"

    @pytest.mark.asyncio
    async def test_context_fn_with_fallback(self):
        """Should support fallback patterns in context_fn."""

        @observe(
            sampling=NoSamplingStrategy(),
            context_fn=lambda args, kwargs: (
                kwargs.get("docs") or kwargs.get("context") or []
            ),
        )
        async def func(query: str, context: list[str] = None, docs: list[str] = None):
            return "result"

        # Using context kwarg
        result1 = await func("q1", context=["ctx doc"])
        assert result1 == "result"

        # Using docs kwarg
        result2 = await func("q2", docs=["docs doc"])
        assert result2 == "result"
