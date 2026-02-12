"""OpenAI LLM client implementation."""

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

    def _get_client(self) -> Any:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIClient. "
                    "Install it with: pip install openai"
                )
            self._client = AsyncOpenAI(api_key=self.api_key, **self._kwargs)
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
