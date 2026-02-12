"""Tests for .node().run() -- single node execution."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from fasteval_langgraph import harness
from fasteval_langgraph.models import NodeResult


class TestNodeExecution:
    """Test single node execution via .node().run()."""

    @pytest.mark.asyncio
    async def test_basic_node_run(self, router_graph):
        graph = harness(router_graph)
        result = await graph.node("classifier").run(
            messages=[HumanMessage(content="What is OAuth?")]
        )
        assert isinstance(result, NodeResult)
        assert result.node_name == "classifier"
        assert result.updates.get("intent") == "FAQ"
        assert result.goto == "rag"

    @pytest.mark.asyncio
    async def test_troubleshoot_classification(self, router_graph):
        graph = harness(router_graph)
        result = await graph.node("classifier").run(
            messages=[HumanMessage(content="I need to troubleshoot this")]
        )
        assert result.updates.get("intent") == "TROUBLESHOOTING"
        assert result.goto == "planner"

    @pytest.mark.asyncio
    async def test_node_with_ai_response(self, router_graph):
        graph = harness(router_graph)
        result = await graph.node("responder").run(
            messages=[HumanMessage(content="Test")],
            docs=["some doc"],
        )
        assert result.response is not None
        assert "some doc" in result.response

    @pytest.mark.asyncio
    async def test_execution_time_measured(self, router_graph):
        graph = harness(router_graph)
        result = await graph.node("classifier").run(
            messages=[HumanMessage(content="Hello")]
        )
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_nonexistent_node_raises(self, router_graph):
        graph = harness(router_graph)
        with pytest.raises(ValueError, match="not found"):
            await graph.node("nonexistent").run(x=1)


class TestNodePlainState:
    """Test node execution with plain TypedDict graph."""

    @pytest.mark.asyncio
    async def test_calc_node(self, calc_graph):
        graph = harness(calc_graph)
        result = await graph.node("calc").run(input="5 + 3")
        assert result.node_name == "calc"
        assert "Calculated: 5 + 3" in result.updates.get("result", "")
