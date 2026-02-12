"""Pydantic models for observation data."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ObservationMetrics(BaseModel):
    """Metrics collected during function execution."""

    latency_ms: float = Field(description="Execution time in milliseconds")
    success: bool = Field(description="Whether execution completed without error")
    error_type: Optional[str] = Field(
        default=None,
        description="Exception type if error occurred",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Exception message if error occurred",
    )

    # Token metrics (optional, populated if available)
    input_tokens: Optional[int] = Field(
        default=None,
        description="Number of input tokens (if tracked)",
    )
    output_tokens: Optional[int] = Field(
        default=None,
        description="Number of output tokens (if tracked)",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens used",
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        description="Estimated cost in USD (if tracked)",
    )


class EvaluationMetrics(BaseModel):
    """Metrics from fasteval evaluations (if run_evaluations=True)."""

    metrics_evaluated: List[str] = Field(
        default_factory=list,
        description="Names of metrics that were evaluated",
    )
    aggregate_score: Optional[float] = Field(
        default=None,
        description="Aggregate evaluation score (0.0-1.0)",
    )
    passed: Optional[bool] = Field(
        default=None,
        description="Whether all metrics passed their thresholds",
    )
    metric_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual metric scores",
    )


class Observation(BaseModel):
    """
    Complete observation record for a single function call.

    This is the structured log format written to files/stdout.
    """

    # Identifiers
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of observation",
    )
    source: str = Field(
        default="fasteval_observe",
        description="Source identifier",
    )
    version: str = Field(
        default="0.1.0",
        description="fasteval-observe version",
    )
    event_type: str = Field(
        default="agent_observation",
        description="Type of observation event",
    )

    # Tracing
    trace_id: Optional[str] = Field(
        default=None,
        description="Distributed trace ID",
    )
    span_id: Optional[str] = Field(
        default=None,
        description="Span ID within trace",
    )
    parent_span_id: Optional[str] = Field(
        default=None,
        description="Parent span ID",
    )

    # Function info
    function_name: str = Field(description="Name of the observed function")
    function_module: Optional[str] = Field(
        default=None,
        description="Module containing the function",
    )

    # Sampling info
    sampling_strategy: str = Field(description="Name of sampling strategy used")
    sampling_rate: Optional[float] = Field(
        default=None,
        description="Sampling rate at time of observation",
    )

    # Core metrics
    metrics: ObservationMetrics = Field(description="Execution metrics")

    # Evaluation results (optional)
    evaluation_results: Optional[EvaluationMetrics] = Field(
        default=None,
        description="Evaluation results if run_evaluations=True",
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-provided metadata (use for environment, service_name, etc.)",
    )

    # Input/Output (privacy sensitive, controlled by config)
    input_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Function inputs (if include_inputs=True)",
    )
    output_data: Optional[Any] = Field(
        default=None,
        description="Function output (if include_outputs=True)",
    )

    model_config = {"extra": "allow"}


class ObservationBatch(BaseModel):
    """Batch of observations for efficient processing."""

    observations: List[Observation] = Field(default_factory=list)
    batch_id: str = Field(description="Unique batch identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    count: int = Field(default=0, description="Number of observations in batch")

    def add(self, observation: Observation) -> None:
        """Add an observation to the batch."""
        self.observations.append(observation)
        self.count = len(self.observations)
