"""Configuration for fasteval-observe."""

from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel, Field


class ObserveConfig(BaseModel):
    """
    Configuration for the observe decorator.

    Example:
        from fasteval_observe import configure_observe, ObserveConfig

        configure_observe(ObserveConfig(
            enabled=True,
            include_inputs=False,
            max_queue_size=10000,
        ))

    For custom logging, use set_logger() instead:
        from fasteval_observe import set_logger
        import logging

        my_logger = logging.getLogger("my_app.observations")
        set_logger(my_logger)
    """

    # General settings
    enabled: bool = Field(
        default=True,
        description="Enable/disable observation globally",
    )

    # Privacy settings
    include_inputs: bool = Field(
        default=False,
        description="Include function inputs in logs (privacy sensitive)",
    )
    include_outputs: bool = Field(
        default=False,
        description="Include function outputs in logs (privacy sensitive)",
    )

    # Queue settings
    max_queue_size: int = Field(
        default=10000,
        description="Maximum observations in queue before dropping",
    )
    flush_interval_seconds: float = Field(
        default=5.0,
        description="Interval between queue flushes",
    )
    batch_size: int = Field(
        default=100,
        description="Maximum observations per batch flush",
    )

    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker for queue overflow protection",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        description="Consecutive failures before opening circuit",
    )
    circuit_breaker_recovery_time: float = Field(
        default=30.0,
        description="Seconds before attempting to close circuit",
    )

    model_config = {"extra": "forbid"}


# Global configuration using ContextVar for thread-safety
_config: ContextVar[ObserveConfig] = ContextVar(
    "fasteval_observe_config",
    default=ObserveConfig(),
)


def configure_observe(config: ObserveConfig) -> None:
    """
    Set the global observe configuration.

    Args:
        config: ObserveConfig instance with desired settings

    Example:
        configure_observe(ObserveConfig(
            enabled=True,
            include_inputs=False,
            max_queue_size=10000,
        ))
    """
    _config.set(config)


def get_config() -> ObserveConfig:
    """
    Get the current global observe configuration.

    Returns:
        Current ObserveConfig instance
    """
    return _config.get()


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config.set(ObserveConfig())
