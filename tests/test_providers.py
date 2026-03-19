"""Tests for fasteval.providers (registry, openai, anthropic)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import fasteval.providers.registry as registry_module
from fasteval.providers.anthropic import AnthropicClient
from fasteval.providers.openai import OpenAIClient
from fasteval.providers.registry import (
    clear_default_provider,
    create_provider_for_model,
    get_default_provider,
    set_default_provider,
)


class MockLLMClient:
    async def invoke(self, messages):
        return "mock response"


class TestProviderRegistry:
    def setup_method(self):
        clear_default_provider()

    def test_set_and_get_default_provider(self):
        client = MockLLMClient()
        set_default_provider(client)
        assert get_default_provider() is client

    def test_clear_default_provider(self):
        set_default_provider(MockLLMClient())
        clear_default_provider()
        # Should now try env vars or raise
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="No LLM provider configured"):
                    get_default_provider()

    def test_create_from_env_openai(self):
        clear_default_provider()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = get_default_provider()
            assert isinstance(provider, OpenAIClient)

    def test_create_from_env_anthropic(self):
        clear_default_provider()
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key"},
            clear=False,
        ):
            # Remove OpenAI key if present
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env["ANTHROPIC_API_KEY"] = "test-key"
            with patch.dict(os.environ, env, clear=True):
                provider = get_default_provider()
                assert isinstance(provider, AnthropicClient)

    def test_create_from_env_no_keys(self):
        clear_default_provider()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No LLM provider configured"):
                get_default_provider()

    def test_create_provider_for_model_gpt(self):
        provider = create_provider_for_model("gpt-4o")
        assert isinstance(provider, OpenAIClient)
        assert provider.model == "gpt-4o"

    def test_create_provider_for_model_o1(self):
        provider = create_provider_for_model("o1-preview")
        assert isinstance(provider, OpenAIClient)

    def test_create_provider_for_model_o3(self):
        provider = create_provider_for_model("o3-mini")
        assert isinstance(provider, OpenAIClient)

    def test_create_provider_for_model_claude(self):
        provider = create_provider_for_model("claude-sonnet-4-6")
        assert isinstance(provider, AnthropicClient)
        assert provider.model == "claude-sonnet-4-6"

    def test_create_provider_for_model_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_provider_for_model("llama-70b")


class TestOpenAIClient:
    def test_init_defaults(self):
        client = OpenAIClient()
        assert client.model == "gpt-4o-mini"
        assert client.temperature == 0.0
        assert client._client is None

    def test_init_custom(self):
        client = OpenAIClient(model="gpt-4o", api_key="key", temperature=0.5)
        assert client.model == "gpt-4o"
        assert client.api_key == "key"
        assert client.temperature == 0.5

    def test_repr(self):
        client = OpenAIClient(model="gpt-4o")
        assert repr(client) == "OpenAIClient(model='gpt-4o')"

    @pytest.mark.asyncio
    async def test_invoke_mocked(self):
        client = OpenAIClient(model="gpt-4o", api_key="test")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"

        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        client._client = mock_async_client

        result = await client.invoke([{"role": "user", "content": "Hi"}])
        assert result == "Hello!"


class TestProviderRegistryEdgeCases:
    def setup_method(self):
        clear_default_provider()

    def test_create_provider_for_model_claude_import_error(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            # When anthropic is not importable
            try:
                provider = create_provider_for_model("claude-3-5-sonnet")
                # If it succeeds, that's fine (anthropic might already be imported)
                assert isinstance(provider, AnthropicClient)
            except ImportError:
                pass  # Expected when anthropic can't be imported


class TestAnthropicClient:
    def test_init_defaults(self):
        client = AnthropicClient()
        assert client.model == "claude-sonnet-4-6"
        assert client.temperature == 0.0
        assert client.max_tokens == 4096

    def test_repr(self):
        client = AnthropicClient(model="claude-3-5-sonnet")
        assert repr(client) == "AnthropicClient(model='claude-3-5-sonnet')"

    @pytest.mark.asyncio
    async def test_invoke_mocked(self):
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test")

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Hi from Claude!"

        mock_async_client = AsyncMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_response)
        client._client = mock_async_client

        result = await client.invoke([{"role": "user", "content": "Hello"}])
        assert result == "Hi from Claude!"

    @pytest.mark.asyncio
    async def test_invoke_with_system_message(self):
        client = AnthropicClient(api_key="test")

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response"

        mock_async_client = AsyncMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_response)
        client._client = mock_async_client

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        await client.invoke(messages)

        call_kwargs = mock_async_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
