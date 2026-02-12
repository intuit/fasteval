"""Shared fixtures: simple test graphs for plugin unit tests."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict


# ---------------------------------------------------------------------------
# 1. Simple MessagesState graph (echo bot)
# ---------------------------------------------------------------------------
class EchoState(MessagesState):
    """Messages + a simple counter."""

    call_count: int


def echo_node(state: EchoState) -> dict:
    last_msg = state["messages"][-1].content
    return {
        "messages": [AIMessage(content=f"Echo: {last_msg}")],
        "call_count": state.get("call_count", 0) + 1,
    }


@pytest.fixture()
def echo_graph():
    """Compiled MessagesState graph that echoes user input."""
    builder = StateGraph(EchoState)
    builder.add_node("echo", echo_node)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 2. Plain TypedDict graph (calculator)
# ---------------------------------------------------------------------------
class CalcState(TypedDict):
    input: str
    result: str


def calc_node(state: CalcState) -> dict:
    return {"result": f"Calculated: {state['input']}"}


@pytest.fixture()
def calc_graph():
    """Compiled plain TypedDict graph (no messages)."""
    builder = StateGraph(CalcState)
    builder.add_node("calc", calc_node)
    builder.add_edge(START, "calc")
    builder.add_edge("calc", END)
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 3. Multi-node routing graph with Command
# ---------------------------------------------------------------------------
class RouterState(MessagesState):
    intent: str
    docs: list
    plan: str


def classifier_node(state: RouterState) -> Command:
    last_msg = state["messages"][-1].content
    if "troubleshoot" in last_msg.lower():
        intent = "TROUBLESHOOTING"
    else:
        intent = "FAQ"
    return Command(
        update={"intent": intent},
        goto="rag" if intent == "FAQ" else "planner",
    )


def rag_node(state: RouterState) -> Command:
    return Command(
        update={"docs": [f"doc for: {state['intent']}"]},
        goto="responder",
    )


def planner_node(state: RouterState) -> Command:
    return Command(
        update={"plan": f"plan for: {state['intent']}"},
        goto="responder",
    )


def responder_node(state: RouterState) -> dict:
    context = state.get("docs") or [state.get("plan", "")]
    return {
        "messages": [AIMessage(content=f"Response based on: {context[0]}")],
    }


@pytest.fixture()
def router_graph():
    """Multi-node graph with Command-based routing."""
    builder = StateGraph(RouterState)
    builder.add_node("classifier", classifier_node)
    builder.add_node("rag", rag_node)
    builder.add_node("planner", planner_node)
    builder.add_node("responder", responder_node)
    builder.add_edge(START, "classifier")
    builder.add_edge("responder", END)
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 4. Interrupt graph (human-in-the-loop)
# ---------------------------------------------------------------------------
class InterruptState(MessagesState):
    approved: bool


def ask_node(state: InterruptState) -> dict:
    return {"messages": [AIMessage(content="Do you approve?")]}


def wait_node(state: InterruptState) -> dict:
    answer = interrupt("Waiting for approval")
    return {"approved": answer == "yes"}


def final_node(state: InterruptState) -> dict:
    if state.get("approved"):
        return {"messages": [AIMessage(content="Approved! Proceeding.")]}
    return {"messages": [AIMessage(content="Rejected.")]}


@pytest.fixture()
def interrupt_graph():
    """Graph with an interrupt for human-in-the-loop testing."""
    builder = StateGraph(InterruptState)
    builder.add_node("ask", ask_node)
    builder.add_node("wait", wait_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", "wait")
    builder.add_edge("wait", "final")
    builder.add_edge("final", END)
    return builder.compile(checkpointer=MemorySaver())
