"""Tests for the observation logger."""

import json
import logging

import pytest

from fasteval_observe.config import ObserveConfig, configure_observe, reset_config
from fasteval_observe.logger import (
    get_logger,
    log_observation,
    set_logger,
)
from fasteval_observe.metrics import Observation, ObservationMetrics


@pytest.fixture(autouse=True)
def reset_all():
    """Reset config and logger before each test."""
    import sys

    from fasteval_observe import logger as logger_module

    reset_config()

    # Reset logger to default
    logger_module._logger = logging.getLogger("fasteval_observe")
    logger_module._logger.handlers.clear()
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger_module._logger.addHandler(_handler)
    logger_module._logger.setLevel(logging.INFO)
    logger_module._logger.propagate = True  # Allow caplog to capture

    yield

    reset_config()


class TestLoggerConfiguration:
    """Tests for logger configuration."""

    def test_default_logger_exists(self):
        """Should have a default logger configured."""
        logger = get_logger()
        assert logger is not None
        assert logger.name == "fasteval_observe"

    def test_set_custom_logger(self):
        """Should allow setting a custom logger."""
        custom_logger = logging.getLogger("my_custom_logger")
        set_logger(custom_logger)

        assert get_logger() == custom_logger

    def test_custom_logger_receives_observations(self, caplog):
        """Custom logger should receive observations."""
        custom_logger = logging.getLogger("test_observations")
        custom_logger.setLevel(logging.INFO)
        set_logger(custom_logger)

        obs = Observation(
            function_name="test_func",
            sampling_strategy="TestStrategy",
            metrics=ObservationMetrics(latency_ms=100, success=True),
        )

        with caplog.at_level(logging.INFO, logger="test_observations"):
            log_observation(obs)

        assert len(caplog.records) == 1
        assert "test_func" in caplog.records[0].message


class TestLogObservation:
    """Tests for log_observation function."""

    def create_observation(self) -> Observation:
        """Create a test observation."""
        return Observation(
            function_name="test_func",
            function_module="test_module",
            sampling_strategy="TestStrategy",
            metrics=ObservationMetrics(
                latency_ms=123.45,
                success=True,
            ),
            trace_id="trace-123",
            span_id="span-456",
        )

    def test_logs_to_default_logger(self, caplog):
        """Should log to default logger (stdout)."""
        with caplog.at_level(logging.INFO, logger="fasteval_observe"):
            obs = self.create_observation()
            log_observation(obs)

        assert len(caplog.records) == 1
        log_message = caplog.records[0].message
        assert "test_func" in log_message
        assert "TestStrategy" in log_message

    def test_logs_valid_json(self, caplog):
        """Should log valid JSON."""
        with caplog.at_level(logging.INFO, logger="fasteval_observe"):
            obs = self.create_observation()
            log_observation(obs)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)

        assert data["function_name"] == "test_func"
        assert data["sampling_strategy"] == "TestStrategy"
        assert data["metrics"]["latency_ms"] == 123.45

    def test_respects_disabled_config(self, caplog):
        """Should not log when disabled."""
        configure_observe(ObserveConfig(enabled=False))

        with caplog.at_level(logging.INFO, logger="fasteval_observe"):
            obs = self.create_observation()
            log_observation(obs)

        assert len(caplog.records) == 0

    def test_excludes_none_values(self, caplog):
        """Should exclude None values from JSON."""
        obs = Observation(
            function_name="test_func",
            sampling_strategy="TestStrategy",
            metrics=ObservationMetrics(latency_ms=100, success=True),
            trace_id=None,  # Should be excluded
        )

        with caplog.at_level(logging.INFO, logger="fasteval_observe"):
            log_observation(obs)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)

        assert "trace_id" not in data

    def test_timestamp_is_iso_format(self, caplog):
        """Should format timestamp as ISO string."""
        obs = Observation(
            function_name="test_func",
            sampling_strategy="TestStrategy",
            metrics=ObservationMetrics(latency_ms=100, success=True),
        )

        with caplog.at_level(logging.INFO, logger="fasteval_observe"):
            log_observation(obs)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)

        assert isinstance(data["timestamp"], str)
        assert "T" in data["timestamp"]


class TestCustomLoggerIntegration:
    """Tests for custom logger integration scenarios."""

    def test_file_handler_integration(self, tmp_path):
        """Should work with file handlers."""
        log_file = tmp_path / "observations.jsonl"

        # Create logger with file handler
        file_logger = logging.getLogger("file_test")
        file_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        file_logger.addHandler(handler)
        file_logger.propagate = False

        set_logger(file_logger)

        obs = Observation(
            function_name="file_test_func",
            sampling_strategy="FileTest",
            metrics=ObservationMetrics(latency_ms=50, success=True),
        )

        log_observation(obs)
        handler.flush()

        # Verify file contents
        content = log_file.read_text()
        data = json.loads(content.strip())
        assert data["function_name"] == "file_test_func"

    def test_multiple_handlers(self, tmp_path):
        """Should work with multiple handlers."""
        log_file = tmp_path / "multi.jsonl"

        multi_logger = logging.getLogger("multi_handler_test")
        multi_logger.setLevel(logging.INFO)
        multi_logger.propagate = False

        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        multi_logger.addHandler(file_handler)

        set_logger(multi_logger)

        obs = Observation(
            function_name="multi_func",
            sampling_strategy="MultiTest",
            metrics=ObservationMetrics(latency_ms=75, success=True),
        )

        log_observation(obs)
        file_handler.flush()

        # File should have the log
        file_content = log_file.read_text()
        assert "multi_func" in file_content

        # Verify it's valid JSON
        data = json.loads(file_content.strip())
        assert data["function_name"] == "multi_func"
