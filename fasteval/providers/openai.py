"""OpenAI LLM client implementation."""

import asyncio
from typing import Any, Dict, List, Optional


class OpenAIClient:
    """
    OpenAI LLM client for fasteval evaluation.

    Uses the openai package to call OpenAI's chat completions API.

    Example:
        from fasteval.providers import OpenAIClient

        client = OpenAIClient(model="gpt-4o-mini")
        response = await client.invoke([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Sampling temperature (0.0 = deterministic)
            **kwargs: Additional arguments passed to AsyncOpenAI
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._kwargs = kwargs
        self._client: Optional[Any] = None
        self._client_loop: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazy initialization of OpenAI client."""
        # The SDK's connection pool is bound to the event loop it first ran
        # on, and each eval may run on a new loop (asyncio.run per test), so
        # keep one SDK client per loop instead of failing with
        # "Event loop is closed" on the next test.
        loop = _running_loop()
        if self._client is not None and self._client_loop is None:
            self._client_loop = loop  # first use of a client set from outside
        if self._client is None or self._client_loop is not loop:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIClient. "
                    "Install it with: pip install openai"
                )
            self._client = AsyncOpenAI(api_key=self.api_key, **self._kwargs)
            self._client_loop = loop
        return self._client

    async def invoke(self, messages: List[Dict[str, Any]]) -> str:
        """
        Call OpenAI chat completions API.

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            Response content string
        """
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self.model!r})"


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
