"""
fasteval-langgraph: LangGraph testing plugin for fasteval

Wrap any compiled LangGraph StateGraph with a minimal harness for
chat, session, mocking, and node-level testing. Results feed directly
into fe.score() and fasteval decorators.

Install: pip install fasteval-langgraph

Example:
    from fasteval_langgraph import harness, mock

    graph = harness(compiled_graph)
    result = await graph.chat("Hello")
    fe.score(result.response, "Expected greeting", input="Hello")
"""

from fasteval_langgraph.graph import GraphHarness, harness
from fasteval_langgraph.mocking import NodeMock, mock
from fasteval_langgraph.models import ChatResult, NodeResult, NodeStep

__version__ = "0.1.0"

__all__ = [
    # Factory function (preferred entry point)
    "harness",
    # Core class
    "GraphHarness",
    # Mocking
    "mock",
    "NodeMock",
    # Data models
    "ChatResult",
    "NodeResult",
    "NodeStep",
]
