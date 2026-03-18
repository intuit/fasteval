"""Tests for fasteval.testing.plugin."""

import os
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from fasteval.collectors.summary import EvalRunSummary, MetricAggregate, TestCaseSummary
from fasteval.testing.plugin import (
    _print_console_summary,
    pytest_addoption,
    pytest_configure,
    pytest_sessionfinish,
    pytest_sessionstart,
    pytest_unconfigure,
)

# ── pytest_addoption ─────────────────────────────────────────────────────────


class TestPytestAddoption:
    def test_adds_options(self):
        mock_parser = MagicMock()
        pytest_addoption(mock_parser)
        assert mock_parser.addoption.call_count == 3
        option_names = [call[0][0] for call in mock_parser.addoption.call_args_list]
        assert "--no-interactive" in option_names
        assert "--fe-output" in option_names
        assert "--fe-summary" in option_names


# ── pytest_configure ─────────────────────────────────────────────────────────


class TestPytestConfigure:
    def test_sets_env_var(self):
        mock_config = MagicMock()
        mock_config.getoption.return_value = True
        with patch.dict(os.environ, {}, clear=False):
            pytest_configure(mock_config)
            assert os.environ.get("FASTEVAL_NO_INTERACTIVE") == "1"
            # Cleanup
            os.environ.pop("FASTEVAL_NO_INTERACTIVE", None)

    def test_no_env_var_when_false(self):
        mock_config = MagicMock()
        mock_config.getoption.return_value = False
        initial = os.environ.get("FASTEVAL_NO_INTERACTIVE")
        pytest_configure(mock_config)
        assert os.environ.get("FASTEVAL_NO_INTERACTIVE") == initial


# ── pytest_sessionstart ──────────────────────────────────────────────────────


class TestPytestSessionStart:
    def test_resets_collector(self):
        with patch("fasteval.collectors.collector.reset_collector") as mock_reset:
            pytest_sessionstart(MagicMock())


# ── pytest_sessionfinish ─────────────────────────────────────────────────────


class TestPytestSessionFinish:
    def test_no_results_noop(self):
        mock_session = MagicMock()
        mock_collector = MagicMock()
        mock_collector.results = []

        with patch(
            "fasteval.collectors.collector.get_collector",
            return_value=mock_collector,
        ):
            pytest_sessionfinish(mock_session, 0)
            mock_collector.report.assert_not_called()

    def test_fe_output_with_path(self):
        mock_session = MagicMock()
        mock_session.config.getoption.side_effect = lambda opt, **kwargs: {
            "--fe-summary": False,
            "--fe-output": ["json:output.json"],
        }.get(opt, kwargs.get("default"))

        mock_collector = MagicMock()
        mock_collector.results = [MagicMock()]

        with patch(
            "fasteval.collectors.collector.get_collector",
            return_value=mock_collector,
        ):
            pytest_sessionfinish(mock_session, 0)
            mock_collector.report.assert_called_once_with("json", path="output.json")

    def test_fe_output_without_path(self):
        mock_session = MagicMock()
        mock_session.config.getoption.side_effect = lambda opt, **kwargs: {
            "--fe-summary": False,
            "--fe-output": ["json"],
        }.get(opt, kwargs.get("default"))

        mock_collector = MagicMock()
        mock_collector.results = [MagicMock()]
        mock_collector.report.return_value = '{"test": true}'

        with patch(
            "fasteval.collectors.collector.get_collector",
            return_value=mock_collector,
        ):
            pytest_sessionfinish(mock_session, 0)
            mock_collector.report.assert_called_once_with("json")

    def test_fe_summary(self):
        mock_session = MagicMock()
        mock_session.config.getoption.side_effect = lambda opt, **kwargs: {
            "--fe-summary": True,
            "--fe-output": [],
        }.get(opt, kwargs.get("default"))

        mock_collector = MagicMock()
        mock_collector.results = [MagicMock()]

        with (
            patch(
                "fasteval.collectors.collector.get_collector",
                return_value=mock_collector,
            ),
            patch("fasteval.testing.plugin._print_console_summary") as mock_print,
        ):
            pytest_sessionfinish(mock_session, 0)
            mock_print.assert_called_once()


# ── pytest_unconfigure ───────────────────────────────────────────────────────


class TestPytestUnconfigure:
    def test_cleans_up_env_var(self):
        mock_config = MagicMock()
        mock_config.getoption.return_value = True
        with patch.dict(os.environ, {"FASTEVAL_NO_INTERACTIVE": "1"}):
            pytest_unconfigure(mock_config)
            assert "FASTEVAL_NO_INTERACTIVE" not in os.environ

    def test_no_cleanup_when_not_set(self):
        mock_config = MagicMock()
        mock_config.getoption.return_value = False
        env_copy = os.environ.copy()
        env_copy.pop("FASTEVAL_NO_INTERACTIVE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            pytest_unconfigure(mock_config)


# ── _print_console_summary ───────────────────────────────────────────────────


class TestPrintConsoleSummary:
    def test_output_format(self, capsys):
        summary = EvalRunSummary(
            total_tests=3,
            passed_tests=2,
            failed_tests=1,
            pass_rate=2 / 3,
            avg_aggregate_score=0.75,
            total_execution_time_ms=100.0,
            metric_aggregates=[
                MetricAggregate(
                    metric_name="correctness",
                    count=3,
                    pass_count=2,
                    fail_count=1,
                    pass_rate=2 / 3,
                    avg_score=0.75,
                    min_score=0.3,
                    max_score=1.0,
                )
            ],
            test_summaries=[
                TestCaseSummary(
                    test_name="test_pass",
                    passed=True,
                    aggregate_score=1.0,
                    metric_count=1,
                    execution_time_ms=10.0,
                ),
                TestCaseSummary(
                    test_name="test_fail",
                    passed=False,
                    aggregate_score=0.3,
                    metric_count=1,
                    execution_time_ms=20.0,
                    error="low score",
                ),
            ],
        )
        _print_console_summary(summary)
        captured = capsys.readouterr()
        assert "FastEval Summary" in captured.out
        assert "3 total" in captured.out
        assert "2 passed" in captured.out
        assert "1 failed" in captured.out
        assert "correctness" in captured.out
        assert "test_fail" in captured.out

    def test_no_metrics(self, capsys):
        summary = EvalRunSummary(
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            pass_rate=1.0,
            avg_aggregate_score=1.0,
            total_execution_time_ms=5.0,
        )
        _print_console_summary(summary)
        captured = capsys.readouterr()
        assert "FastEval Summary" in captured.out
