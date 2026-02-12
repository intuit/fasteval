"""Data models for fasteval-langgraph plugin."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NodeStep(BaseModel):
    """One node's execution in a trace."""

    node: str
    """Node name that executed."""

    updates: Dict[str, Any] = Field(default_factory=dict)
    """State updates written by this node (not full state)."""

    duration_ms: Optional[float] = None
    """Execution time in milliseconds (if measured)."""


class ChatResult(BaseModel):
    """Result of a single ``.chat()`` call."""

    response: str = ""
    """Last AI message content (via ``output_fn``)."""

    state: Dict[str, Any] = Field(default_factory=dict)
    """Filtered state snapshot (via ``state_filter``)."""

    trace: List[NodeStep] = Field(default_factory=list)
    """Per-node execution trace, in order."""

    nodes_ran: List[str] = Field(default_factory=list)
    """Node names in execution order."""


class NodeResult(BaseModel):
    """Result of running a single node via ``.node().run()``."""

    node_name: str
    """Name of the node that was executed."""

    updates: Dict[str, Any] = Field(default_factory=dict)
    """State updates from this node."""

    goto: Optional[str] = None
    """Routing target from Command, if any."""

    response: Optional[str] = None
    """AI message content, if any."""

    execution_time_ms: float = 0.0
    """Execution time in milliseconds."""
