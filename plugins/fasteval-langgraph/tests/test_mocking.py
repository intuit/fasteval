"""Tests for node mocking: mock(), with_mocks(), mocked()."""

from __future__ import annotations

import pytest

from fasteval_langgraph import harness, mock
from fasteval_langgraph.mocking import NodeMock


class TestMockBuilder:
    """Test NodeMock fluent builder."""

    def test_creates_node_mock(self):
        m = mock("rag")
        assert isinstance(m, NodeMock)
        assert m.node_name == "rag"

    def test_fluent_chaining(self):
        m = mock("rag").updates({"docs": ["fake"]}).goto("planner")
        assert m._state_updates == {"docs": ["fake"]}
        assert m._goto_target == "planner"

    def test_fn_handler(self):
        handler = lambda state: {"result": "ok"}
        m = mock("analyzer").fn(handler)
        assert m._handler is handler

    def test_repr(self):
        m = mock("rag").updates({"docs": ["x"]}).goto("planner")
        r = repr(m)
        assert "rag" in r
        assert "updates" in r
        assert "goto" in r


class TestWithMocks:
    """Test graph.with_mocks() -- returns new GraphHarness with mocks applied."""

    @pytest.mark.asyncio
    async def test_with_mocks_overrides_node(self, router_graph):
        graph = harness(router_graph)
        test_graph = graph.with_mocks(
            mock("rag").updates({"docs": ["mocked doc"]}).goto("responder"),
        )
        result = await test_graph.chat("What is OAuth?")
        assert "mocked doc" in str(result.state.get("docs", []))
        # Clean up
        test_graph.reset_mocks()

    @pytest.mark.asyncio
    async def test_with_mocks_doesnt_affect_original(self, router_graph):
        graph = harness(router_graph)
        test_graph = graph.with_mocks(
            mock("rag").updates({"docs": ["mocked"]}).goto("responder"),
        )
        # Reset test mocks
        test_graph.reset_mocks()
        # Original should still work normally
        result = await graph.chat("What is OAuth?")
        assert "mocked" not in str(result.state.get("docs", []))


class TestMockedContextManager:
    """Test graph.mocked() context manager."""

    @pytest.mark.asyncio
    async def test_mocked_scope(self, router_graph):
        graph = harness(router_graph)
        with graph.mocked(
            mock("rag").updates({"docs": ["scoped mock"]}).goto("responder"),
        ):
            result = await graph.chat("What is OAuth?")
            assert "scoped mock" in str(result.state.get("docs", []))

        # After scope, mocks should be restored
        result2 = await graph.chat("What is OAuth?")
        assert "scoped mock" not in str(result2.state.get("docs", []))


class TestMockInPlace:
    """Test graph.mock() -- mutate in place."""

    @pytest.mark.asyncio
    async def test_mock_in_place(self, router_graph):
        graph = harness(router_graph)
        graph.mock("rag").updates({"docs": ["in-place mock"]}).goto("responder")
        result = await graph.chat("What is OAuth?")
        assert "in-place mock" in str(result.state.get("docs", []))
        graph.reset_mocks()

    @pytest.mark.asyncio
    async def test_reset_mocks(self, router_graph):
        graph = harness(router_graph)
        graph.mock("rag").updates({"docs": ["temp mock"]}).goto("responder")
        graph.reset_mocks()
        result = await graph.chat("What is OAuth?")
        assert "temp mock" not in str(result.state.get("docs", []))


class TestMockWithDynamicFn:
    """Test mock with fn() handler."""

    @pytest.mark.asyncio
    async def test_fn_handler(self, router_graph):
        graph = harness(router_graph)
        with graph.mocked(
            mock("rag").fn(
                lambda state: {"docs": [f"dynamic for {state.get('intent', '?')}"]}
            ).goto("responder"),
        ):
            result = await graph.chat("What is OAuth?")
            docs = result.state.get("docs", [])
            assert len(docs) >= 1
            assert "dynamic for FAQ" in docs[0]


class TestMockValidation:
    """Test mock error handling."""

    def test_mock_nonexistent_node_raises(self, router_graph):
        graph = harness(router_graph)
        with pytest.raises(ValueError, match="not found"):
            graph.with_mocks(mock("nonexistent").updates({"x": 1}))
