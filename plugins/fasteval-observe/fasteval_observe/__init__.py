"""
fasteval-observe: Runtime monitoring for fasteval

A production-ready plugin for runtime agent monitoring with async
sampling and structured logging. Install separately from fasteval.

Install: pip install fasteval-observe

Example:
    from fasteval_observe import observe
    from fasteval_observe.sampling import FixedRateSamplingStrategy

    @observe(sampling=FixedRateSamplingStrategy(rate=0.05))
    async def my_agent(query: str) -> str:
        return await llm.invoke(query)
"""

from fasteval_observe.config import ObserveConfig, configure_observe, get_config
from fasteval_observe.decorator import (
    initialize_observer,
    observe,
    shutdown_observer,
)
from fasteval_observe.logger import get_logger, set_logger
from fasteval_observe.sampling import (
    AdaptiveSamplingStrategy,
    BaseSamplingStrategy,
    ComposableSamplingStrategy,
    FixedRateSamplingStrategy,
    NoSamplingStrategy,
    ProbabilisticSamplingStrategy,
    TokenBudgetSamplingStrategy,
)

__version__ = "0.1.0"

__all__ = [
    # Decorator
    "observe",
    # Lifecycle
    "initialize_observer",
    "shutdown_observer",
    # Configuration
    "ObserveConfig",
    "configure_observe",
    "get_config",
    # Logging
    "get_logger",
    "set_logger",
    # Sampling strategies
    "BaseSamplingStrategy",
    "NoSamplingStrategy",
    "FixedRateSamplingStrategy",
    "ProbabilisticSamplingStrategy",
    "AdaptiveSamplingStrategy",
    "TokenBudgetSamplingStrategy",
    "ComposableSamplingStrategy",
]
