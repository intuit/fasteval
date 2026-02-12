"""Configuration models for fasteval using Pydantic."""

from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from fasteval.providers.base import LLMClient


class MetricConfig(BaseModel):
    """
    Configuration for a metric attached via decorator.

    Stored on func._fasteval_metrics by metric decorators.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric_type: str  # "correctness", "hallucination", "geval", etc.
    name: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    config: Dict[str, Any] = Field(default_factory=dict)  # Metric-specific config
    llm_config: Optional[Dict[str, Any]] = None  # Model, temperature, etc.
    llm_client: Optional[Any] = None  # Custom LLM client (Any to avoid import issues)
