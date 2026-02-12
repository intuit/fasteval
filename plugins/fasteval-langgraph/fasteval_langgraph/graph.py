"""GraphHarness: the core wrapper for testing compiled LangGraph agents."""

from __future__ import annotations

import copy
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from fasteval_langgraph._utils import (
    deep_merge,
    default_state_filter_identity,
    default_state_filter_messages,
    extract_last_ai_message,
    has_messages_channel,
)
from fasteval_langgraph.mocking import (
    NodeMock,
    apply_mocks,
    mocked_scope,
    restore_mocks,
)
from fasteval_langgraph.models import ChatResult, NodeResult, NodeStep


# ---------------------------------------------------------------------------
# NodeHandle -- returned by graph.node("name")
# ---------------------------------------------------------------------------
class NodeHandle:
    """Handle for executing a single node in isolation."""

    def __init__(self, graph_harness: GraphHarness, node_name: str) -> None:
        self._harness = graph_harness
        self._node_name = node_name

    async def run(self, **state_fields: Any) -> NodeResult:
        """Execute this node with the given state fields.

        Args:
            **state_fields: Keyword arguments become the input state dict.

        Returns:
            ``NodeResult`` with updates, goto, response, and timing.
        """
        graph = self._harness._graph
        nodes_dict = graph.nodes
        if self._node_name not in nodes_dict:
            available = ", ".join(sorted(nodes_dict.keys()))
            raise ValueError(
                f"Node {self._node_name!r} not found. Available: {available}"
            )

        node_fn = nodes_dict[self._node_name]
        config = self._harness._build_config(str(uuid4()))

        t0 = time.perf_counter()
        # Node functions can be sync or async
        if hasattr(node_fn, "ainvoke"):
            raw_result = await node_fn.ainvoke(state_fields, config)
        elif callable(node_fn):
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(node_fn):
                raw_result = await node_fn(state_fields, config)
            else:
                raw_result = await asyncio.get_event_loop().run_in_executor(
                    None, node_fn, state_fields
                )
        else:
            raw_result = node_fn.invoke(state_fields, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Parse result
        goto = None
        updates: dict[str, Any] = {}
        if isinstance(raw_result, Command):
            goto = raw_result.goto if hasattr(raw_result, "goto") else None
            updates = raw_result.update if hasattr(raw_result, "update") else {}
            if isinstance(goto, str):
                pass
            elif goto is not None:
                goto = str(goto)
        elif isinstance(raw_result, dict):
            updates = raw_result
        else:
            updates = {}

        # Extract AI message if present
        response = None
        msgs = updates.get("messages", [])
        for msg in reversed(msgs if isinstance(msgs, list) else []):
            if hasattr(msg, "type") and msg.type == "ai":
                response = msg.content or ""
                break
            if isinstance(msg, dict) and msg.get("type") == "ai":
                response = msg.get("content", "")
                break

        return NodeResult(
            node_name=self._node_name,
            updates=updates,
            goto=goto,
            response=response,
            execution_time_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Session -- multi-turn conversation
# ---------------------------------------------------------------------------
class Session:
    """Multi-turn conversation session (holds a single thread_id).

    Supports async context manager usage::

        async with graph.session() as s:
            r1 = await s.chat("Hello")
            r2 = await s.chat("Follow up")
    """

    def __init__(
        self,
        harness: GraphHarness,
        thread_id: str,
        session_config: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
    ) -> None:
        self._harness = harness
        self._thread_id = thread_id
        self._session_config = session_config
        self._session_context = session_context
        self._history: list[ChatResult] = []
        self._is_interrupted = False

    @property
    def thread_id(self) -> str:
        """The thread_id used for this session."""
        return self._thread_id

    @property
    def history(self) -> list[ChatResult]:
        """All ``ChatResult`` objects from this session, in order."""
        return list(self._history)

    async def chat(
        self,
        message: str,
        *,
        config: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Send a message in this session.

        Auto-detects interrupts and uses ``resume_fn`` when needed.

        Args:
            message: User message string.
            config: Per-turn config override (deep-merged with session + factory).
            context: Per-turn context override (deep-merged with session + factory).
        """
        result = await self._harness._execute_chat(
            message=message,
            thread_id=self._thread_id,
            session_config=self._session_config,
            session_context=self._session_context,
            call_config=config,
            call_context=context,
        )
        self._history.append(result)
        return result

    async def update_state(
        self,
        values: dict[str, Any],
        as_node: str | None = None,
    ) -> None:
        """Seed / update the graph state for this session.

        Calls ``compiled_graph.update_state()`` which writes to the checkpoint.
        The next ``chat()`` call will continue from this updated state.

        Args:
            values: State updates to apply.
            as_node: Optional node name to attribute the update to.
        """
        config = self._harness._build_config(self._thread_id)
        kwargs: dict[str, Any] = {"config": config, "values": values}
        if as_node is not None:
            kwargs["as_node"] = as_node
        await self._harness._graph.aupdate_state(**kwargs)

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# GraphHarness -- the main wrapper
# ---------------------------------------------------------------------------
class GraphHarness:
    """Wrap a compiled LangGraph for testing.

    Args:
        compiled_graph: A compiled ``StateGraph`` (result of ``graph.compile()``).
        input_fn: ``(str) -> dict`` converts a user message into graph input.
        output_fn: ``(dict) -> str`` extracts a string response from state.
        state_filter: ``(dict) -> dict`` filters the raw state for ``ChatResult.state``.
        config_factory: ``(thread_id) -> dict`` generates ``RunnableConfig``.
        context_factory: ``() -> dict | None`` generates runtime context.
        resume_fn: ``(str) -> Any`` produces input for resuming after interrupt.
        recursion_limit: Max recursion depth for graph execution.
    """

    def __init__(
        self,
        compiled_graph: Any,
        *,
        input_fn: Callable[[str], dict] | None = None,
        output_fn: Callable[[dict], str] | None = None,
        state_filter: Callable[[dict], dict] | None = None,
        config_factory: Callable[[str], dict] | None = None,
        context_factory: Callable[[], dict | None] | None = None,
        resume_fn: Callable[[str], Any] | None = None,
        recursion_limit: int = 25,
    ) -> None:
        self._graph = compiled_graph
        self._recursion_limit = recursion_limit

        # Detect graph type
        self._has_messages = has_messages_channel(compiled_graph)

        # Set defaults based on auto-detection
        if input_fn is not None:
            self._input_fn = input_fn
        elif self._has_messages:
            self._input_fn = lambda msg: {"messages": [HumanMessage(content=msg)]}
        else:
            self._input_fn = lambda msg: {"input": msg}

        if output_fn is not None:
            self._output_fn = output_fn
        elif self._has_messages:
            self._output_fn = extract_last_ai_message
        else:
            self._output_fn = lambda state: str(state)

        if state_filter is not None:
            self._state_filter = state_filter
        elif self._has_messages:
            self._state_filter = default_state_filter_messages
        else:
            self._state_filter = default_state_filter_identity

        self._config_factory = config_factory or (
            lambda tid: {"configurable": {"thread_id": tid}}
        )

        self._context_factory = context_factory

        if resume_fn is not None:
            self._resume_fn = resume_fn
        else:
            self._resume_fn = lambda msg: Command(resume=msg)

        # Mocking state
        self._active_mocks: list[NodeMock] = []
        self._mock_originals: dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------
    @property
    def nodes(self) -> set[str]:
        """Set of all user-defined node names in the graph."""
        return {k for k in self._graph.nodes.keys() if not k.startswith("__")}

    @property
    def entry(self) -> str:
        """Entry point node name."""
        # LangGraph stores the first node after START
        first_node = getattr(self._graph, "first_node", None)
        if first_node:
            return first_node
        # Fallback: look at the graph structure
        nodes_list = list(self._graph.nodes.keys())
        return nodes_list[0] if nodes_list else ""

    @property
    def has_messages_state(self) -> bool:
        """Whether the graph uses MessagesState (auto-detected)."""
        return self._has_messages

    # -------------------------------------------------------------------
    # Chat (single message)
    # -------------------------------------------------------------------
    async def chat(
        self,
        message: str,
        *,
        config: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Send a single message to the graph (new thread per call).

        Args:
            message: User message string.
            config: Per-call config override (deep-merged with factory defaults).
            context: Per-call context override (deep-merged with factory defaults).
        """
        thread_id = str(uuid4())
        return await self._execute_chat(
            message=message,
            thread_id=thread_id,
            session_config=None,
            session_context=None,
            call_config=config,
            call_context=context,
        )

    # -------------------------------------------------------------------
    # Session (multi-turn)
    # -------------------------------------------------------------------
    def session(
        self,
        *,
        config: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Session:
        """Create a multi-turn session with a shared thread_id.

        Args:
            config: Session-level config override.
            context: Session-level context override.
        """
        thread_id = str(uuid4())
        return Session(
            harness=self,
            thread_id=thread_id,
            session_config=config,
            session_context=context,
        )

    # -------------------------------------------------------------------
    # Node (single node execution)
    # -------------------------------------------------------------------
    def node(self, name: str) -> NodeHandle:
        """Get a handle for executing a single node.

        Args:
            name: Node name.

        Returns:
            ``NodeHandle`` with a ``.run()`` method.
        """
        return NodeHandle(self, name)

    # -------------------------------------------------------------------
    # Mocking
    # -------------------------------------------------------------------
    def mock(self, node_name: str) -> NodeMock:
        """Create and register a mock for a node (mutates in place).

        Returns a ``NodeMock`` builder for fluent chaining::

            graph.mock("rag").updates({"docs": ["fake"]}).goto("planner")
        """
        nm = NodeMock(node_name)
        self._active_mocks.append(nm)
        # Apply immediately
        originals = apply_mocks(self._graph, [nm])
        self._mock_originals.update(originals)
        return nm

    def with_mocks(self, *mocks: NodeMock) -> GraphHarness:
        """Return a new ``GraphHarness`` with mocks applied. Original is unchanged.

        Args:
            *mocks: ``NodeMock`` instances (created via ``mock()``).
        """
        # We need to create a shallow copy that shares the compiled graph
        # but has its own mock state. We apply mocks to the shared graph
        # and track them for cleanup.
        new_harness = GraphHarness.__new__(GraphHarness)
        new_harness._graph = self._graph
        new_harness._recursion_limit = self._recursion_limit
        new_harness._has_messages = self._has_messages
        new_harness._input_fn = self._input_fn
        new_harness._output_fn = self._output_fn
        new_harness._state_filter = self._state_filter
        new_harness._config_factory = self._config_factory
        new_harness._context_factory = self._context_factory
        new_harness._resume_fn = self._resume_fn
        new_harness._active_mocks = list(mocks)
        new_harness._mock_originals = apply_mocks(self._graph, list(mocks))
        return new_harness

    def mocked(self, *mocks: NodeMock):
        """Context manager that applies mocks and restores on exit.

        Example::

            with graph.mocked(mock("rag").updates({"docs": ["fake"]})):
                result = await graph.chat("...")
        """
        return mocked_scope(self._graph, list(mocks))

    def reset_mocks(self) -> None:
        """Restore all original node functions."""
        if self._mock_originals:
            restore_mocks(self._graph, self._mock_originals)
            self._mock_originals.clear()
            self._active_mocks.clear()

    # -------------------------------------------------------------------
    # Internal: build config
    # -------------------------------------------------------------------
    def _build_config(
        self,
        thread_id: str,
        session_config: dict[str, Any] | None = None,
        call_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build effective config by deep-merging layers."""
        factory_config = self._config_factory(thread_id)
        merged = deep_merge(factory_config, session_config, call_config)
        # Inject recursion_limit if not overridden
        if "recursion_limit" not in merged:
            merged["recursion_limit"] = self._recursion_limit
        return merged

    def _build_context(
        self,
        session_context: dict[str, Any] | None = None,
        call_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Build effective context by deep-merging layers."""
        factory_context = self._context_factory() if self._context_factory else None
        # If all layers are None, return None
        if factory_context is None and session_context is None and call_context is None:
            return None
        return deep_merge(factory_context, session_context, call_context)

    # -------------------------------------------------------------------
    # Internal: execute chat (used by both .chat() and Session.chat())
    # -------------------------------------------------------------------
    async def _execute_chat(
        self,
        message: str,
        thread_id: str,
        session_config: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        call_config: dict[str, Any] | None = None,
        call_context: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Core chat execution with trace capture via streaming."""
        config = self._build_config(thread_id, session_config, call_config)
        context = self._build_context(session_context, call_context)

        # Detect if we need to resume from interrupt
        is_resuming = False
        try:
            current_state = await self._graph.aget_state(config)
            if current_state and current_state.next:
                is_resuming = True
        except Exception:
            # No prior state -- fresh thread
            pass

        # Build input
        if is_resuming:
            graph_input = self._resume_fn(message)
        elif message == "" or message is None:
            graph_input = None
        else:
            graph_input = self._input_fn(message)

        # Stream with trace capture
        trace: list[NodeStep] = []
        invoke_kwargs: dict[str, Any] = {}
        if context is not None:
            invoke_kwargs["context"] = context

        async for chunk in self._graph.astream(
            graph_input,
            config,
            stream_mode="updates",
            **invoke_kwargs,
        ):
            # chunk is a dict {node_name: updates_dict}
            if isinstance(chunk, dict):
                for node_name, updates in chunk.items():
                    if node_name == "__interrupt__":
                        continue
                    step = NodeStep(
                        node=node_name,
                        updates=updates if isinstance(updates, dict) else {},
                    )
                    trace.append(step)

        # Get final state
        final_state = await self._graph.aget_state(config)
        raw_state = (
            dict(final_state.values) if final_state and final_state.values else {}
        )

        # Extract response and filtered state
        response = self._output_fn(raw_state)
        filtered_state = self._state_filter(raw_state)
        nodes_ran = [step.node for step in trace]

        return ChatResult(
            response=response,
            state=filtered_state,
            trace=trace,
            nodes_ran=nodes_ran,
        )


# ---------------------------------------------------------------------------
# harness() factory function
# ---------------------------------------------------------------------------
def harness(
    compiled_graph: Any,
    *,
    input_fn: Callable[[str], dict] | None = None,
    output_fn: Callable[[dict], str] | None = None,
    state_filter: Callable[[dict], dict] | None = None,
    config_factory: Callable[[str], dict] | None = None,
    context_factory: Callable[[], dict | None] | None = None,
    resume_fn: Callable[[str], Any] | None = None,
    recursion_limit: int = 25,
) -> GraphHarness:
    """Create a ``GraphHarness`` for testing a compiled LangGraph.

    This is the **preferred entry point** -- a thin convenience function
    that mirrors the ``GraphHarness`` constructor.

    Args:
        compiled_graph: A compiled ``StateGraph`` (result of ``graph.compile()``).
        input_fn: ``(str) -> dict`` converts a user message into graph input.
        output_fn: ``(dict) -> str`` extracts a string response from state.
        state_filter: ``(dict) -> dict`` filters raw state for ``ChatResult.state``.
        config_factory: ``(thread_id) -> dict`` generates ``RunnableConfig``.
        context_factory: ``() -> dict | None`` generates runtime context.
        resume_fn: ``(str) -> Any`` produces input for resuming after interrupt.
        recursion_limit: Max recursion depth. Default: 25.

    Returns:
        A configured ``GraphHarness`` instance.
    """
    return GraphHarness(
        compiled_graph,
        input_fn=input_fn,
        output_fn=output_fn,
        state_filter=state_filter,
        config_factory=config_factory,
        context_factory=context_factory,
        resume_fn=resume_fn,
        recursion_limit=recursion_limit,
    )
