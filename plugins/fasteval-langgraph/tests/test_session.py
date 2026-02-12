"""Tests for Session -- multi-turn conversations."""

from __future__ import annotations

import pytest

from fasteval_langgraph import harness
from fasteval_langgraph.models import ChatResult


class TestSession:
    """Test multi-turn session with same thread_id."""

    @pytest.mark.asyncio
    async def test_session_reuses_thread(self, echo_graph):
        graph = harness(echo_graph)
        async with graph.session() as s:
            r1 = await s.chat("Hello")
            r2 = await s.chat("World")
            # Same thread: call_count should accumulate
            assert r1.state["call_count"] == 1
            assert r2.state["call_count"] == 2

    @pytest.mark.asyncio
    async def test_session_history(self, echo_graph):
        graph = harness(echo_graph)
        async with graph.session() as s:
            await s.chat("First")
            await s.chat("Second")
            assert len(s.history) == 2
            assert s.history[0].response == "Echo: First"
            assert s.history[1].response == "Echo: Second"

    @pytest.mark.asyncio
    async def test_session_thread_id(self, echo_graph):
        graph = harness(echo_graph)
        s = graph.session()
        assert s.thread_id is not None
        assert len(s.thread_id) > 0

    @pytest.mark.asyncio
    async def test_session_without_context_manager(self, echo_graph):
        graph = harness(echo_graph)
        s = graph.session()
        r1 = await s.chat("Hello")
        assert isinstance(r1, ChatResult)
        assert "Echo: Hello" in r1.response


class TestSessionInterrupt:
    """Test session with interrupt/resume."""

    @pytest.mark.asyncio
    async def test_interrupt_and_resume(self, interrupt_graph):
        graph = harness(interrupt_graph)
        async with graph.session() as s:
            # First turn: triggers the ask node, then interrupt at wait
            r1 = await s.chat("Start")
            assert "Do you approve?" in r1.response

            # Second turn: should auto-detect interrupt and resume
            r2 = await s.chat("yes")
            assert "Approved" in r2.response or "Proceeding" in r2.response

    @pytest.mark.asyncio
    async def test_interrupt_reject(self, interrupt_graph):
        graph = harness(interrupt_graph)
        async with graph.session() as s:
            r1 = await s.chat("Start")
            r2 = await s.chat("no")
            assert "Rejected" in r2.response


class TestSessionUpdateState:
    """Test session.update_state() for state seeding."""

    @pytest.mark.asyncio
    async def test_update_state(self, echo_graph):
        graph = harness(echo_graph)
        async with graph.session() as s:
            # First, send a message to establish thread
            r1 = await s.chat("Hello")
            assert r1.state["call_count"] == 1
