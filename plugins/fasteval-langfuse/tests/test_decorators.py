"""Tests for fasteval_langfuse.decorators."""

from unittest.mock import MagicMock, patch

import pytest

from fasteval_langfuse.decorators import (
    FASTEVAL_DATA_ATTR,
    FASTEVAL_METRICS_ATTR,
    langfuse_dataset,
    langfuse_traces,
)


class TestLangfuseTracesDecorator:
    def test_attaches_data_attr(self):
        @langfuse_traces(project="prod", filter_tags=["support"])
        def my_func(trace_id, input, output, context, metadata):
            pass

        assert hasattr(my_func, FASTEVAL_DATA_ATTR)
        data = getattr(my_func, FASTEVAL_DATA_ATTR)
        assert data["type"] == "langfuse_traces"
        assert data["project"] == "prod"

    def test_preserves_metrics(self):
        def my_func():
            pass

        setattr(my_func, FASTEVAL_METRICS_ATTR, ["metric1"])

        decorated = langfuse_traces(project="prod")(my_func)
        assert getattr(decorated, FASTEVAL_METRICS_ATTR) == ["metric1"]

    def test_default_sampling_name(self):
        @langfuse_traces()
        def my_func():
            pass

        data = getattr(my_func, FASTEVAL_DATA_ATTR)
        assert data["sampling"] == "NoSamplingStrategy"

    def test_custom_sampling_name(self):
        mock_sampling = MagicMock()
        mock_sampling.name = "CustomStrategy"

        @langfuse_traces(sampling=mock_sampling)
        def my_func():
            pass

        data = getattr(my_func, FASTEVAL_DATA_ATTR)
        assert data["sampling"] == "CustomStrategy"

    def test_async_function(self):
        @langfuse_traces()
        async def my_func():
            pass

        assert hasattr(my_func, FASTEVAL_DATA_ATTR)


class TestLangfuseDatasetDecorator:
    def test_attaches_data_attr(self):
        @langfuse_dataset(name="qa-set", version="v2")
        def my_func(input, expected_output):
            pass

        data = getattr(my_func, FASTEVAL_DATA_ATTR)
        assert data["type"] == "langfuse_dataset"
        assert data["name"] == "qa-set"
        assert data["version"] == "v2"

    def test_preserves_metrics(self):
        def my_func():
            pass

        setattr(my_func, FASTEVAL_METRICS_ATTR, ["metric1"])

        decorated = langfuse_dataset(name="ds")(my_func)
        assert getattr(decorated, FASTEVAL_METRICS_ATTR) == ["metric1"]

    def test_async_function(self):
        @langfuse_dataset(name="ds")
        async def my_func():
            pass

        assert hasattr(my_func, FASTEVAL_DATA_ATTR)


class TestExecuteTraceEvaluation:
    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.format_sampling_stats")
    @patch("fasteval_langfuse.decorators.ScoreReporter")
    @patch("fasteval_langfuse.decorators.TraceFetcher")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_basic_execution(
        self,
        mock_client_cls,
        mock_fetcher_cls,
        mock_reporter_cls,
        mock_format_stats,
        mock_get_score,
    ):
        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher
        mock_fetcher.fetch_and_sample.return_value = (
            [{"id": "t-1", "input": "q", "output": "a", "metadata": {}}],
            1,
        )
        mock_fetcher.map_trace_to_params.return_value = {
            "trace_id": "t-1",
            "input": "q",
            "output": "a",
            "context": None,
            "metadata": {},
        }

        mock_reporter = MagicMock()
        mock_reporter_cls.return_value = mock_reporter
        mock_get_score.return_value = None

        from fasteval_langfuse.decorators import _execute_trace_evaluation

        called_with = {}

        def my_func(**kwargs):
            called_with.update(kwargs)

        _execute_trace_evaluation(
            func=my_func,
            is_async=False,
            project="prod",
            filter_tags=None,
            time_range=None,
            user_id=None,
            session_id=None,
            limit=None,
            sampling=None,
            auto_push_scores=True,
            args=(),
            kwargs={},
        )

        assert called_with["trace_id"] == "t-1"
        mock_reporter.flush.assert_called_once()

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.format_sampling_stats")
    @patch("fasteval_langfuse.decorators.ScoreReporter")
    @patch("fasteval_langfuse.decorators.TraceFetcher")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_pushes_scores_when_result_exists(
        self,
        mock_client_cls,
        mock_fetcher_cls,
        mock_reporter_cls,
        mock_format_stats,
        mock_get_score,
    ):
        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher
        mock_fetcher.fetch_and_sample.return_value = (
            [{"id": "t-1", "input": "q", "output": "a", "metadata": {}}],
            1,
        )
        mock_fetcher.map_trace_to_params.return_value = {
            "trace_id": "t-1",
            "input": "q",
            "output": "a",
            "context": None,
            "metadata": {},
        }

        mock_reporter = MagicMock()
        mock_reporter_cls.return_value = mock_reporter

        mock_result = MagicMock()
        mock_result.metric_results = [MagicMock()]
        mock_result.aggregate_score = 0.9
        mock_get_score.return_value = mock_result

        from fasteval_langfuse.decorators import _execute_trace_evaluation

        def my_func(**kwargs):
            pass

        _execute_trace_evaluation(
            func=my_func,
            is_async=False,
            project=None,
            filter_tags=None,
            time_range=None,
            user_id=None,
            session_id=None,
            limit=None,
            sampling=None,
            auto_push_scores=True,
            args=(),
            kwargs={},
        )

        mock_reporter.push_evaluation_result.assert_called_once()

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.format_sampling_stats")
    @patch("fasteval_langfuse.decorators.ScoreReporter")
    @patch("fasteval_langfuse.decorators.TraceFetcher")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_skips_push_when_auto_push_false(
        self,
        mock_client_cls,
        mock_fetcher_cls,
        mock_reporter_cls,
        mock_format_stats,
        mock_get_score,
    ):
        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher
        mock_fetcher.fetch_and_sample.return_value = (
            [{"id": "t-1", "input": "q", "output": "a", "metadata": {}}],
            1,
        )
        mock_fetcher.map_trace_to_params.return_value = {
            "trace_id": "t-1",
            "input": "q",
            "output": "a",
            "context": None,
            "metadata": {},
        }

        mock_reporter = MagicMock()
        mock_reporter_cls.return_value = mock_reporter

        mock_result = MagicMock()
        mock_get_score.return_value = mock_result

        from fasteval_langfuse.decorators import _execute_trace_evaluation

        def my_func(**kwargs):
            pass

        _execute_trace_evaluation(
            func=my_func,
            is_async=False,
            project=None,
            filter_tags=None,
            time_range=None,
            user_id=None,
            session_id=None,
            limit=None,
            sampling=None,
            auto_push_scores=False,
            args=(),
            kwargs={},
        )

        mock_reporter.push_evaluation_result.assert_not_called()

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.format_sampling_stats")
    @patch("fasteval_langfuse.decorators.ScoreReporter")
    @patch("fasteval_langfuse.decorators.TraceFetcher")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_prints_stats_when_sampling(
        self,
        mock_client_cls,
        mock_fetcher_cls,
        mock_reporter_cls,
        mock_format_stats,
        mock_get_score,
    ):
        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher
        mock_fetcher.fetch_and_sample.return_value = ([], 0)

        mock_reporter = MagicMock()
        mock_reporter_cls.return_value = mock_reporter
        mock_get_score.return_value = None
        mock_format_stats.return_value = "stats"

        from fasteval_langfuse.decorators import _execute_trace_evaluation

        mock_sampling = MagicMock()
        mock_sampling.name = "TestStrategy"

        _execute_trace_evaluation(
            func=lambda **kw: None,
            is_async=False,
            project=None,
            filter_tags=None,
            time_range=None,
            user_id=None,
            session_id=None,
            limit=None,
            sampling=mock_sampling,
            auto_push_scores=True,
            args=(),
            kwargs={},
        )

        mock_format_stats.assert_called_once()
