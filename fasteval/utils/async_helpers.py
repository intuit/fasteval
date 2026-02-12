"""Async utility functions."""

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine safely, handling event loop issues in Python 3.10+.

    Uses asyncio.run() which creates a new event loop and closes it when done.
    This avoids issues with get_event_loop() deprecation warnings and errors.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Example:
        async def fetch_data():
            return await some_async_call()

        result = run_async(fetch_data())
    """
    return asyncio.run(coro)
