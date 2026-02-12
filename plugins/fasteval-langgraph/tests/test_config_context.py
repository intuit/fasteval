"""Tests for config/context layered override system."""

from __future__ import annotations

import pytest

from fasteval_langgraph import harness
from fasteval_langgraph._utils import deep_merge


# ---------------------------------------------------------------------------
# deep_merge unit tests
# ---------------------------------------------------------------------------
class TestDeepMerge:
    """Test the deep_merge utility function."""

    def test_simple_merge(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_merge(self):
        result = deep_merge(
            {"configurable": {"thread_id": "t1", "model": "gpt-4"}},
            {"configurable": {"model": "gpt-3.5"}},
        )
        assert result == {"configurable": {"thread_id": "t1", "model": "gpt-3.5"}}

    def test_none_skipped(self):
        assert deep_merge(None, {"a": 1}, None) == {"a": 1}

    def test_all_none(self):
        assert deep_merge(None, None) == {}

    def test_later_wins(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_deep_nested(self):
        result = deep_merge(
            {"a": {"b": {"c": 1, "d": 2}}},
            {"a": {"b": {"c": 99}}},
        )
        assert result == {"a": {"b": {"c": 99, "d": 2}}}


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------
class TestConfigFactory:
    """Test config_factory and per-call config overrides."""

    @pytest.mark.asyncio
    async def test_default_config_has_thread_id(self, echo_graph):
        graph = harness(echo_graph)
        # Just ensure it works -- the default config_factory adds thread_id
        result = await graph.chat("Test")
        assert isinstance(result.response, str)

    @pytest.mark.asyncio
    async def test_custom_config_factory(self, echo_graph):
        captured_configs = []

        def my_config_factory(tid):
            config = {"configurable": {"thread_id": tid, "custom_key": "factory_val"}}
            captured_configs.append(config)
            return config

        graph = harness(echo_graph, config_factory=my_config_factory)
        await graph.chat("Test")
        assert len(captured_configs) == 1
        assert captured_configs[0]["configurable"]["custom_key"] == "factory_val"

    @pytest.mark.asyncio
    async def test_per_call_config_override(self, echo_graph):
        graph = harness(echo_graph)
        # Per-call config should deep-merge with factory defaults
        result = await graph.chat(
            "Test",
            config={"configurable": {"extra_key": "call_val"}},
        )
        assert isinstance(result.response, str)


# ---------------------------------------------------------------------------
# Context factory
# ---------------------------------------------------------------------------
class TestContextFactory:
    """Test context_factory and per-call context overrides."""

    @pytest.mark.asyncio
    async def test_no_context_by_default(self, echo_graph):
        graph = harness(echo_graph)
        # Should work fine without context
        result = await graph.chat("Test")
        assert isinstance(result.response, str)


# ---------------------------------------------------------------------------
# Session-level overrides
# ---------------------------------------------------------------------------
class TestSessionOverrides:
    """Test session-level config/context overrides."""

    @pytest.mark.asyncio
    async def test_session_config_applied(self, echo_graph):
        graph = harness(echo_graph)
        async with graph.session(
            config={"configurable": {"session_key": "sess_val"}}
        ) as s:
            result = await s.chat("Hello")
            assert isinstance(result.response, str)

    @pytest.mark.asyncio
    async def test_per_turn_override_in_session(self, echo_graph):
        graph = harness(echo_graph)
        async with graph.session() as s:
            result = await s.chat(
                "Hello",
                config={"configurable": {"turn_key": "turn_val"}},
            )
            assert isinstance(result.response, str)


# ---------------------------------------------------------------------------
# Custom input_fn / output_fn / state_filter
# ---------------------------------------------------------------------------
class TestCustomHooks:
    """Test custom input_fn, output_fn, state_filter."""

    @pytest.mark.asyncio
    async def test_custom_input_fn(self, calc_graph):
        graph = harness(
            calc_graph,
            input_fn=lambda msg: {"input": f"CUSTOM: {msg}"},
        )
        result = await graph.chat("5+3")
        assert "CUSTOM: 5+3" in result.state.get("input", "")

    @pytest.mark.asyncio
    async def test_custom_output_fn(self, echo_graph):
        graph = harness(
            echo_graph,
            output_fn=lambda state: "CUSTOM OUTPUT",
        )
        result = await graph.chat("Hello")
        assert result.response == "CUSTOM OUTPUT"

    @pytest.mark.asyncio
    async def test_custom_state_filter(self, echo_graph):
        graph = harness(
            echo_graph,
            state_filter=lambda state: {"only_count": state.get("call_count")},
        )
        result = await graph.chat("Hello")
        assert "only_count" in result.state
        assert "call_count" not in result.state


# ---------------------------------------------------------------------------
# Recursion limit
# ---------------------------------------------------------------------------
class TestRecursionLimit:
    """Test recursion_limit parameter."""

    def test_default_recursion_limit(self, echo_graph):
        graph = harness(echo_graph)
        config = graph._build_config("test-thread")
        assert config["recursion_limit"] == 25

    def test_custom_recursion_limit(self, echo_graph):
        graph = harness(echo_graph, recursion_limit=50)
        config = graph._build_config("test-thread")
        assert config["recursion_limit"] == 50
