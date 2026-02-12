"""Tests for GraphHarness.chat() -- single message execution."""

from __future__ import annotations

import pytest

from fasteval_langgraph import GraphHarness, harness
from fasteval_langgraph.models import ChatResult, NodeStep


# ---------------------------------------------------------------------------
# MessagesState graph (echo)
# ---------------------------------------------------------------------------
class TestChatMessagesState:
    """Test .chat() with a MessagesState-based graph."""

    @pytest.mark.asyncio
    async def test_basic_chat(self, echo_graph):
        graph = harness(echo_graph)
        result = await graph.chat("Hello world")
        assert isinstance(result, ChatResult)
        assert "Echo: Hello world" in result.response

    @pytest.mark.asyncio
    async def test_state_excludes_messages(self, echo_graph):
        graph = harness(echo_graph)
        result = await graph.chat("Hi")
        assert "messages" not in result.state
        assert "call_count" in result.state
        assert result.state["call_count"] == 1

    @pytest.mark.asyncio
    async def test_trace_captured(self, echo_graph):
        graph = harness(echo_graph)
        result = await graph.chat("Test")
        assert len(result.trace) >= 1
        assert result.trace[0].node == "echo"
        assert isinstance(result.trace[0], NodeStep)

    @pytest.mark.asyncio
    async def test_nodes_ran(self, echo_graph):
        graph = harness(echo_graph)
        result = await graph.chat("Trace test")
        assert "echo" in result.nodes_ran

    @pytest.mark.asyncio
    async def test_each_chat_new_thread(self, echo_graph):
        graph = harness(echo_graph)
        r1 = await graph.chat("First")
        r2 = await graph.chat("Second")
        # Each call starts fresh, so call_count should be 1 both times
        assert r1.state["call_count"] == 1
        assert r2.state["call_count"] == 1


# ---------------------------------------------------------------------------
# Plain TypedDict graph (calculator)
# ---------------------------------------------------------------------------
class TestChatPlainState:
    """Test .chat() with a plain TypedDict graph (no messages)."""

    @pytest.mark.asyncio
    async def test_basic_chat(self, calc_graph):
        graph = harness(calc_graph)
        result = await graph.chat("5 + 3")
        assert isinstance(result, ChatResult)
        # output_fn default for non-messages graph: str(state)
        assert "Calculated: 5 + 3" in result.response

    @pytest.mark.asyncio
    async def test_state_has_all_keys(self, calc_graph):
        graph = harness(calc_graph)
        result = await graph.chat("10 * 2")
        # Plain state: identity filter keeps all keys
        assert "input" in result.state
        assert "result" in result.state

    @pytest.mark.asyncio
    async def test_trace_captured(self, calc_graph):
        graph = harness(calc_graph)
        result = await graph.chat("1 + 1")
        assert len(result.trace) >= 1
        assert result.trace[0].node == "calc"


# ---------------------------------------------------------------------------
# Multi-node router graph
# ---------------------------------------------------------------------------
class TestChatRouter:
    """Test .chat() with a multi-node routing graph."""

    @pytest.mark.asyncio
    async def test_faq_route(self, router_graph):
        graph = harness(router_graph)
        result = await graph.chat("What is OAuth?")
        assert "FAQ" in result.state.get("intent", "")
        assert len(result.nodes_ran) >= 3  # classifier -> rag -> responder
        assert "classifier" in result.nodes_ran
        assert "rag" in result.nodes_ran
        assert "responder" in result.nodes_ran

    @pytest.mark.asyncio
    async def test_troubleshoot_route(self, router_graph):
        graph = harness(router_graph)
        result = await graph.chat("I need to troubleshoot my app")
        assert "TROUBLESHOOTING" in result.state.get("intent", "")
        assert "planner" in result.nodes_ran
        assert "responder" in result.nodes_ran


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------
class TestAutoDetection:
    """Test auto-detection of MessagesState vs plain state."""

    def test_messages_state_detected(self, echo_graph):
        graph = harness(echo_graph)
        assert graph.has_messages_state is True

    def test_plain_state_detected(self, calc_graph):
        graph = harness(calc_graph)
        assert graph.has_messages_state is False


# ---------------------------------------------------------------------------
# Inspection properties
# ---------------------------------------------------------------------------
class TestInspection:
    """Test graph inspection properties."""

    def test_nodes_property(self, router_graph):
        graph = harness(router_graph)
        assert graph.nodes == {"classifier", "rag", "planner", "responder"}

    def test_has_messages_state_property(self, echo_graph, calc_graph):
        assert harness(echo_graph).has_messages_state is True
        assert harness(calc_graph).has_messages_state is False


# ---------------------------------------------------------------------------
# harness() factory function
# ---------------------------------------------------------------------------
class TestHarnessFactory:
    """Test that harness() produces the same result as GraphHarness()."""

    def test_returns_graph_harness(self, echo_graph):
        graph = harness(echo_graph)
        assert isinstance(graph, GraphHarness)

    @pytest.mark.asyncio
    async def test_equivalent_to_class(self, echo_graph):
        g1 = harness(echo_graph)
        g2 = GraphHarness(echo_graph)
        r1 = await g1.chat("test")
        r2 = await g2.chat("test")
        assert r1.response == r2.response
