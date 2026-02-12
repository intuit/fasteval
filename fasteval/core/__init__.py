"""fasteval core module - decorators and scoring API."""

from fasteval.core.decorators import (
    bias,
    consistency,
    contains,
    context_retention,
    conversation,
    correctness,
    csv,
    exact_match,
    geval,
    hallucination,
    human_review,
    json,
    metric,
    relevance,
    rouge,
    topic_drift,
    toxicity,
    traces,
)
from fasteval.core.scoring import score

__all__ = [
    # Score API
    "score",
    # LLM Metric Decorators
    "correctness",
    "hallucination",
    "relevance",
    "geval",
    "toxicity",
    "bias",
    # Deterministic Metric Decorators
    "rouge",
    "exact_match",
    "contains",
    "json",
    # Conversation Decorators
    "context_retention",
    "consistency",
    "topic_drift",
    # Data Decorators
    "csv",
    "conversation",
    "traces",
    # Human Review
    "human_review",
    # Generic
    "metric",
]
