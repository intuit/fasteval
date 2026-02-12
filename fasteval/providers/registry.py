"""Provider registry for managing default LLM client."""

import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fasteval.providers.base import LLMClient

# Global default provider storage
_default_provider: ContextVar[Optional["LLMClient"]] = ContextVar(
    "fasteval_default_provider", default=None
)


def set_default_provider(provider: "LLMClient") -> None:
    """
    Set the default LLM provider for all metrics.

    Args:
        provider: Any object with an async invoke(messages) method.

    Example:
        from fasteval.providers import OpenAIClient, set_default_provider

        set_default_provider(OpenAIClient(model="gpt-4o"))
    """
    _default_provider.set(provider)


def get_default_provider() -> "LLMClient":
    """
    Get the default LLM provider.

    Resolution order:
    1. Explicitly set provider via set_default_provider()
    2. Auto-detect from environment variables

    Returns:
        LLMClient instance

    Raises:
        ValueError: If no provider configured and no env vars found
    """
    provider = _default_provider.get()
    if provider is not None:
        return provider

    # Auto-detect from environment
    return _create_provider_from_env()


def _create_provider_from_env() -> "LLMClient":
    """Create provider from environment variables."""
    # Check for OpenAI
    if os.getenv("OPENAI_API_KEY"):
        from fasteval.providers.openai import OpenAIClient

        return OpenAIClient(model="gpt-4o-mini")

    # Check for Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from fasteval.providers.anthropic import AnthropicClient

            return AnthropicClient(model="claude-3-5-sonnet-20241022")
        except ImportError:
            pass  # Anthropic not installed

    raise ValueError(
        "No LLM provider configured. Either:\n"
        "  1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable\n"
        "  2. Call fasteval.set_default_provider(your_client)"
    )


def clear_default_provider() -> None:
    """Clear the default provider (mainly for testing)."""
    _default_provider.set(None)


def create_provider_for_model(model_name: str) -> "LLMClient":
    """
    Create an LLM provider for a specific model name.

    Used for per-metric model overrides.

    Args:
        model_name: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")

    Returns:
        LLMClient instance configured for the model

    Example:
        @fe.correctness(model="gpt-4o")  # Uses this function
        async def test_qa():
            ...
    """
    # OpenAI models
    if model_name.startswith(("gpt-", "o1-", "o3-")):
        from fasteval.providers.openai import OpenAIClient

        return OpenAIClient(model=model_name)

    # Anthropic models
    if model_name.startswith("claude"):
        try:
            from fasteval.providers.anthropic import AnthropicClient

            return AnthropicClient(model=model_name)
        except ImportError:
            raise ImportError(
                f"Anthropic provider not installed. Install with: pip install fasteval[anthropic]"
            )

    raise ValueError(
        f"Unknown model: {model_name}. " "Supported prefixes: gpt-, o1-, o3-, claude-"
    )
