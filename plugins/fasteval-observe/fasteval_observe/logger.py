"""Structured JSON logger for observations."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fasteval_observe.config import get_config
from fasteval_observe.metrics import Observation

# Default logger - logs JSON to console
_logger: logging.Logger = logging.getLogger("fasteval_observe")

# Configure default handler if none exists
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def get_logger() -> logging.Logger:
    """
    Get the current observation logger.

    Returns:
        The logger instance used for observations
    """
    return _logger


def set_logger(logger: logging.Logger) -> None:
    """
    Set a custom logger for observations.

    Use this to integrate with your existing logging infrastructure.

    Example:
        import logging
        from fasteval_observe import set_logger

        # Use your own logger
        my_logger = logging.getLogger("my_app.observations")
        my_logger.setLevel(logging.INFO)

        # Add your handlers (file, cloud logging, etc.)
        handler = logging.FileHandler("/var/log/observations.jsonl")
        my_logger.addHandler(handler)

        # Set as the observe logger
        set_logger(my_logger)

    Args:
        logger: A configured logging.Logger instance
    """
    global _logger
    _logger = logger


def _serialize_observation(observation: Observation) -> str:
    """
    Serialize an observation to JSON string.

    Args:
        observation: Observation to serialize

    Returns:
        JSON string representation
    """
    data = observation.model_dump(mode="json", exclude_none=True)

    # Ensure timestamp is ISO format string
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat() + "Z"

    return json.dumps(data, default=str, ensure_ascii=False)


def log_observation(observation: Observation) -> None:
    """
    Log a single observation as JSON.

    The observation is logged at INFO level to the configured logger.

    Args:
        observation: Observation to log
    """
    config = get_config()
    if not config.enabled:
        return

    try:
        json_str = _serialize_observation(observation)
        _logger.info(json_str)
    except Exception as e:
        _logger.error(f"Failed to log observation: {e}")


def log_observations(observations: List[Observation]) -> None:
    """
    Log multiple observations.

    Args:
        observations: List of observations to log
    """
    for obs in observations:
        log_observation(obs)
