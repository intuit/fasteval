"""
fasteval-langfuse: Langfuse integration for fasteval

Evaluate production traces from Langfuse with fasteval metrics.

Install: pip install fasteval-langfuse

Example:
    from fasteval_langfuse import langfuse_traces
    from fasteval_langfuse.sampling import RandomSamplingStrategy
    import fasteval as fe

    @fe.correctness(threshold=0.8)
    @langfuse_traces(
        project="production",
        filter_tags=["customer-support"],
        time_range="last_24h",
        sampling=RandomSamplingStrategy(sample_size=200)
    )
    def test_production_traces(trace_id, input, output, context, metadata):
        fe.score(output, input=input)

    # Run with pytest:
    # pytest test_production.py -v
"""

from fasteval_langfuse.client import LangfuseClient
from fasteval_langfuse.config import LangfuseConfig, configure_langfuse, get_config
from fasteval_langfuse.decorators import langfuse_dataset, langfuse_traces
from fasteval_langfuse.sampling import (
    BaseSamplingStrategy,
    NoSamplingStrategy,
    RandomSamplingStrategy,
    RecentFirstSamplingStrategy,
    ScoreBasedSamplingStrategy,
    StratifiedSamplingStrategy,
)

__version__ = "0.1.0"

__all__ = [
    # Decorators
    "langfuse_traces",
    "langfuse_dataset",
    # Configuration
    "LangfuseConfig",
    "configure_langfuse",
    "get_config",
    # Client
    "LangfuseClient",
    # Sampling strategies
    "BaseSamplingStrategy",
    "NoSamplingStrategy",
    "RandomSamplingStrategy",
    "StratifiedSamplingStrategy",
    "ScoreBasedSamplingStrategy",
    "RecentFirstSamplingStrategy",
]
