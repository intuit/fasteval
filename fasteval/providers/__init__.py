"""fasteval LLM provider module."""

from fasteval.providers.base import LLMClient
from fasteval.providers.openai import OpenAIClient
from fasteval.providers.anthropic import AnthropicClient
from fasteval.providers.registry import (
    create_provider_for_model,
    get_default_provider,
    set_default_provider,
)

__all__ = [
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "set_default_provider",
    "get_default_provider",
    "create_provider_for_model",
]


def __getattr__(name: str) -> object:
    if name == "AnthropicClient":
        from fasteval.providers.anthropic import AnthropicClient

        return AnthropicClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
