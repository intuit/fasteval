"""Node mocking: fluent builder and replacement logic."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from langchain_core.runnables import RunnableLambda
from langgraph.types import Command


class NodeMock:
    """Fluent builder for a single node mock.

    Example::

        mock("rag").updates({"docs": ["fake"]}).goto("planner")
        mock("analyzer").fn(lambda state: {"verdict": "OK"}).goto("END")
    """

    def __init__(self, node_name: str) -> None:
        self.node_name = node_name
        self._state_updates: Dict[str, Any] | None = None
        self._goto_target: str | None = None
        self._handler: Callable[[dict], dict] | None = None

    def updates(self, state_updates: dict[str, Any]) -> NodeMock:
        """Set the static state updates the mock returns."""
        self._state_updates = state_updates
        return self

    def goto(self, target: str) -> NodeMock:
        """Set the routing target (goto) for the mock."""
        self._goto_target = target
        return self

    def fn(self, handler: Callable[[dict], dict]) -> NodeMock:
        """Set a dynamic handler ``(state) -> dict`` that produces updates.

        The handler may return a dict with optional ``"updates"`` and ``"goto"``
        keys, or a plain dict which is treated as state updates.
        """
        self._handler = handler
        return self

    def _build_fake_node(self) -> Callable:
        """Build the replacement node function.

        The fake node reads from ``self`` at invocation time so that
        chained ``.updates()`` / ``.goto()`` calls after mock registration
        are picked up correctly.
        """
        mock_ref = self

        def _fake_node(state: dict) -> Command | dict:
            if mock_ref._handler is not None:
                result = mock_ref._handler(state)
                # Handler may return {"updates": {...}, "goto": "..."}
                if "updates" in result or "goto" in result:
                    upd = result.get("updates", {})
                    gt = result.get("goto", mock_ref._goto_target)
                else:
                    upd = result
                    gt = mock_ref._goto_target
            else:
                upd = mock_ref._state_updates or {}
                gt = mock_ref._goto_target

            if gt is not None:
                return Command(update=upd, goto=gt)
            return upd

        return _fake_node

    def __repr__(self) -> str:
        parts = [f"NodeMock({self.node_name!r})"]
        if self._state_updates is not None:
            parts.append(f".updates({self._state_updates!r})")
        if self._goto_target is not None:
            parts.append(f".goto({self._goto_target!r})")
        if self._handler is not None:
            parts.append(f".fn({self._handler!r})")
        return "".join(parts)


def mock(node_name: str) -> NodeMock:
    """Create a standalone ``NodeMock`` for use with ``with_mocks()`` or ``mocked()``.

    Example::

        from fasteval_langgraph import mock

        test_graph = graph.with_mocks(
            mock("rag").updates({"docs": ["fake"]}).goto("planner"),
        )
    """
    return NodeMock(node_name)


def _user_node_names(compiled_graph: Any) -> set[str]:
    """Return the set of user-defined node names (excluding internal nodes)."""
    return {
        k
        for k in compiled_graph.nodes.keys()
        if not k.startswith("__")
    }


def apply_mocks(
    compiled_graph: Any,
    mocks: list[NodeMock],
) -> dict[str, Any]:
    """Apply mocks to a compiled graph, returning the original bound runnables.

    Replaces both ``PregelNode.bound`` and ``PregelNode.node.steps[0]``
    (the ``RunnableSeq`` step that actually executes), preserving triggers,
    channels, and writers.

    Returns a dict mapping ``node_name -> (original_bound, original_step0)``
    for restoration.
    """
    originals: dict[str, Any] = {}
    nodes_dict = compiled_graph.nodes

    for m in mocks:
        if m.node_name not in nodes_dict:
            available = ", ".join(sorted(_user_node_names(compiled_graph)))
            raise ValueError(
                f"Cannot mock node {m.node_name!r}: not found in graph. "
                f"Available nodes: {available}"
            )
        pregel_node = nodes_dict[m.node_name]
        original_bound = pregel_node.bound
        original_step0 = (
            pregel_node.node.steps[0]
            if hasattr(pregel_node, "node")
            and hasattr(pregel_node.node, "steps")
            else None
        )
        originals[m.node_name] = (original_bound, original_step0)

        fake = RunnableLambda(m._build_fake_node())
        pregel_node.bound = fake
        if original_step0 is not None:
            pregel_node.node.steps[0] = fake

    return originals


def restore_mocks(compiled_graph: Any, originals: dict[str, Any]) -> None:
    """Restore original bound runnables after mocking."""
    nodes_dict = compiled_graph.nodes
    for name, (original_bound, original_step0) in originals.items():
        if name in nodes_dict:
            pregel_node = nodes_dict[name]
            pregel_node.bound = original_bound
            if (
                original_step0 is not None
                and hasattr(pregel_node, "node")
                and hasattr(pregel_node.node, "steps")
            ):
                pregel_node.node.steps[0] = original_step0


@contextmanager
def mocked_scope(compiled_graph: Any, mocks: list[NodeMock]):
    """Context manager that applies mocks and restores on exit."""
    originals = apply_mocks(compiled_graph, mocks)
    try:
        yield
    finally:
        restore_mocks(compiled_graph, originals)
