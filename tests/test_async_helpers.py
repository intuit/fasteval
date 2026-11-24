"""Tests for fasteval.utils.async_helpers."""

import asyncio

from fasteval.utils.async_helpers import run_async


class TestRunAsync:
    def test_run_async_no_running_loop(self):
        async def coro():
            return 42

        result = run_async(coro())
        assert result == 42

    def test_run_async_with_running_loop(self):
        async def inner():
            return "from_inner"

        async def outer():
            # run_async called from within a running event loop
            return run_async(inner())

        result = asyncio.run(outer())
        assert result == "from_inner"

    def test_run_async_with_async_sleep(self):
        async def coro():
            await asyncio.sleep(0.01)
            return "done"

        result = run_async(coro())
        assert result == "done"

    def test_loop_reused_across_calls(self):
        """Async SDK clients bind to the loop they first ran on; a new loop
        per call made the next call fail with "Event loop is closed"."""

        async def current():
            return asyncio.get_running_loop()

        first, second = run_async(current()), run_async(current())
        assert first is second and not first.is_closed()

    def test_running_loop_uses_one_background_loop(self):
        async def current():
            return asyncio.get_running_loop()

        async def outer():
            return run_async(current()), run_async(current())

        a, b = asyncio.run(outer())
        assert a is b

    def test_client_bound_to_first_loop_keeps_working(self):
        """Mimics an httpx/AsyncOpenAI pool: a future created on the first
        call's loop is awaited again on later calls."""
        state = {}

        async def use_client():
            loop = asyncio.get_running_loop()
            state.setdefault("loop", loop)
            fut = state["loop"].create_future()
            loop.call_soon(fut.set_result, "ok")
            return await fut

        assert [run_async(use_client()) for _ in range(3)] == ["ok"] * 3

    def test_exceptions_propagate(self):
        async def boom():
            raise ValueError("nope")

        import pytest

        with pytest.raises(ValueError, match="nope"):
            run_async(boom())
        assert run_async(asyncio.sleep(0, result=1)) == 1  # loop still usable
