"""Anthropic LLM client implementation."""

from typing import Any, Dict, List, Optional


class AnthropicClient:
    """
    Anthropic LLM client for fasteval evaluation.

    Uses the anthropic package to call Anthropic's messages API.

    Example:
        from fasteval.providers import AnthropicClient

        client = AnthropicClient(model="claude-sonnet-4-6")
        response = await client.invoke([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Anthropic client.

        Args:
            model: Model name (e.g., "claude-sonnet-4-6", "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in the response
            **kwargs: Additional arguments passed to AsyncAnthropic
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._kwargs = kwargs
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicClient. "
                    "Install it with: pip install anthropic"
                )
            self._client = AsyncAnthropic(api_key=self.api_key, **self._kwargs)
        return self._client

    async def invoke(self, messages: List[Dict[str, Any]]) -> str:
        """
        Call Anthropic messages API.

        Args:
            messages: List of message dicts with "role" and "content".
                      System messages are extracted and passed via the
                      dedicated system parameter.

        Returns:
            Response content string
        """
        client = self._get_client()

        system_parts: List[str] = []
        non_system: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                non_system.append(msg)

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=non_system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = await client.messages.create(**kwargs)
        return response.content[0].text

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r})"
