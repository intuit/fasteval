"""Async utility functions."""

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from synchronous code, whether or not an event
    loop is already running (e.g. Jupyter, Colab, IPython).

    When no loop is running, delegates to ``asyncio.run()``.
    When called inside a running loop, spins up a worker thread with its
    own loop so the caller never blocks the outer loop and ``asyncio.run()``
    is never nested.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Example:
        async def fetch_data():
            return await some_async_call()

        result = run_async(fetch_data())
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside a running loop — run in a dedicated thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
