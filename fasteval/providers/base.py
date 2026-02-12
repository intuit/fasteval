"""Base LLM client protocol."""

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM clients.

    Users can implement their own client - fasteval just calls invoke().
    Any class with an async invoke() method satisfies this protocol.

    Example:
        class MyCustomClient:
            async def invoke(self, messages: List[Dict[str, Any]]) -> str:
                # Call your LLM endpoint
                response = await my_api.chat(messages)
                return response.content

        # Use it
        fe.set_default_provider(MyCustomClient())

        # Or per-metric
        @fe.correctness(llm=MyCustomClient())
        async def test_something(): ...
    """

    async def invoke(self, messages: List[Dict[str, Any]]) -> str:
        """
        Call LLM with messages, return response content.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Content may be a string or a list of content parts
                     for multimodal messages (vision, audio).
                     Example: [{"role": "user", "content": "Hello"}]

        Returns:
            Response content string from the LLM.
        """
        ...
