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
