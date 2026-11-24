"""Async utility functions."""

import asyncio
import threading
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")

# One event loop per thread, reused by every run_async() call. A fresh loop per
# call (asyncio.run) breaks async SDK clients that outlive a call: AsyncOpenAI,
# AsyncAnthropic and httpx keep connections bound to the loop they first ran
# on, and closing them after that loop is gone fails with
# "Event loop is closed". A reused loop also reuses those connections.
_thread_state = threading.local()

# Used when run_async() is called while a loop is already running (Jupyter,
# Colab, IPython): the coroutine runs on this background loop instead, so the
# caller's loop is never nested or blocked by asyncio.run().
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_background_lock = threading.Lock()


def _thread_loop() -> asyncio.AbstractEventLoop:
    loop: Optional[asyncio.AbstractEventLoop] = getattr(_thread_state, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_state.loop = loop
    return loop


def _get_background_loop() -> asyncio.AbstractEventLoop:
    global _background_loop
    with _background_lock:
        if _background_loop is None or _background_loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="fasteval-async", daemon=True
            ).start()
            _background_loop = loop
        return _background_loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from synchronous code, whether or not an event
    loop is already running (e.g. Jupyter, Colab, IPython).

    When no loop is running, the coroutine runs on this thread's own event
    loop, which is reused across calls. When called inside a running loop, it
    runs on a shared background loop thread, so the caller's loop is never
    nested. Either way async SDK clients keep working across calls.

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
        return _thread_loop().run_until_complete(coro)

    # Already inside a running loop: run on the background loop.
    return asyncio.run_coroutine_threadsafe(coro, _get_background_loop()).result()
