"""Tests for Langfuse Experiments integration in dataset evaluation."""

from unittest.mock import MagicMock, call, patch

import pytest


@pytest.fixture
def mock_dataset_items():
    """Create mock DatasetItemClient objects with .run() context manager support."""
    items = []
    for i in range(3):
        item = MagicMock()
        item.id = f"item-{i}"
        item.input = f"question {i}"
        item.expected_output = f"answer {i}"
        item.metadata = {"difficulty": "easy"} if i == 0 else {}
        item.version = "v1"

        root_span = MagicMock()
        root_span.score_trace = MagicMock()
        item.run.return_value.__enter__ = MagicMock(return_value=root_span)
        item.run.return_value.__exit__ = MagicMock(return_value=False)
        item._root_span = root_span

        items.append(item)
    return items


@pytest.fixture
def mock_score_result():
    """Create a mock EvalResult with metric_results."""
    result = MagicMock()
    result.aggregate_score = 0.85

    mr1 = MagicMock()
    mr1.metric_name = "correctness"
    mr1.score = 0.9
    mr1.reasoning = "Good answer"

    mr2 = MagicMock()
    mr2.metric_name = "relevance"
    mr2.score = 0.8
    mr2.reasoning = None

    result.metric_results = [mr1, mr2]
    return result


def _make_client_with_mock(mock_dataset_items):
    """Build a LangfuseClient whose _client is a MagicMock returning the given items."""
    from fasteval_langfuse.client import LangfuseClient

    mock_langfuse = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.items = mock_dataset_items
    mock_langfuse.get_dataset.return_value = mock_dataset

    client = LangfuseClient.__new__(LangfuseClient)
    client.public_key = "pk-test"
    client.secret_key = "sk-test"
    client.host = "https://cloud.langfuse.com"
    client._client = mock_langfuse

    return client, mock_langfuse


class TestFetchDatasetRaw:
    """Tests for LangfuseClient.fetch_dataset_raw()."""

    def test_returns_raw_item_objects(self, mock_dataset_items):
        client, mock_langfuse = _make_client_with_mock(mock_dataset_items)

        raw = client.fetch_dataset_raw(name="test-dataset")

        assert raw is mock_dataset_items
        assert len(raw) == 3
        assert hasattr(raw[0], "run")
        mock_langfuse.get_dataset.assert_called_once_with("test-dataset")

    def test_filters_by_version(self, mock_dataset_items):
        mock_dataset_items[1].version = "v2"
        client, _ = _make_client_with_mock(mock_dataset_items)

        raw = client.fetch_dataset_raw(name="test-dataset", version="v1")

        assert len(raw) == 2
        assert all(item.version == "v1" for item in raw)

    def test_no_version_filter_returns_all(self, mock_dataset_items):
        client, _ = _make_client_with_mock(mock_dataset_items)

        raw = client.fetch_dataset_raw(name="test-dataset")

        assert len(raw) == 3


class TestDatasetExperimentsIntegration:
    """Tests for the Experiments integration in _execute_dataset_evaluation."""

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.get_config")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_item_run_called_with_run_name(
        self, mock_client_cls, mock_get_config, mock_get_score, mock_dataset_items
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_dataset_raw.return_value = mock_dataset_items

        config = MagicMock()
        config.score_name_prefix = "fasteval_"
        mock_get_config.return_value = config
        mock_get_score.return_value = None

        from fasteval_langfuse.decorators import _execute_dataset_evaluation

        def my_test_func(input, expected_output, **kw):
            pass

        _execute_dataset_evaluation(
            func=my_test_func,
            is_async=False,
            name="test-dataset",
            version=None,
            args=(),
            kwargs={},
        )

        for item in mock_dataset_items:
            item.run.assert_called_once()
            run_name_arg = item.run.call_args
            actual_run_name = run_name_arg[1].get("run_name") or run_name_arg[0][0]
            assert actual_run_name.startswith("fasteval-my_test_func-")

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.get_config")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_scores_pushed_via_root_span(
        self,
        mock_client_cls,
        mock_get_config,
        mock_get_score,
        mock_dataset_items,
        mock_score_result,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_dataset_raw.return_value = mock_dataset_items

        config = MagicMock()
        config.score_name_prefix = "fasteval_"
        mock_get_config.return_value = config
        mock_get_score.return_value = mock_score_result

        from fasteval_langfuse.decorators import _execute_dataset_evaluation

        def my_test_func(**kw):
            pass

        _execute_dataset_evaluation(
            func=my_test_func,
            is_async=False,
            name="test-dataset",
            version=None,
            args=(),
            kwargs={},
        )

        for item in mock_dataset_items:
            root_span = item._root_span
            score_calls = root_span.score_trace.call_args_list

            assert call(name="fasteval_correctness", value=0.9, comment="Good answer") in score_calls
            assert call(name="fasteval_relevance", value=0.8, comment=None) in score_calls
            assert call(name="fasteval_aggregate", value=0.85) in score_calls

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.get_config")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_flush_called_after_evaluation(
        self, mock_client_cls, mock_get_config, mock_get_score, mock_dataset_items
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_dataset_raw.return_value = mock_dataset_items

        config = MagicMock()
        config.score_name_prefix = "fasteval_"
        mock_get_config.return_value = config
        mock_get_score.return_value = None

        from fasteval_langfuse.decorators import _execute_dataset_evaluation

        def my_test_func(**kw):
            pass

        _execute_dataset_evaluation(
            func=my_test_func,
            is_async=False,
            name="test-dataset",
            version=None,
            args=(),
            kwargs={},
        )

        mock_client.flush.assert_called_once()

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.get_config")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_metadata_flattened_into_params(
        self, mock_client_cls, mock_get_config, mock_get_score, mock_dataset_items
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_dataset_raw.return_value = [mock_dataset_items[0]]

        config = MagicMock()
        config.score_name_prefix = "fasteval_"
        mock_get_config.return_value = config
        mock_get_score.return_value = None

        from fasteval_langfuse.decorators import _execute_dataset_evaluation

        received_params = {}

        def my_test_func(**kw):
            received_params.update(kw)

        _execute_dataset_evaluation(
            func=my_test_func,
            is_async=False,
            name="test-dataset",
            version=None,
            args=(),
            kwargs={},
        )

        assert received_params["input"] == "question 0"
        assert received_params["expected_output"] == "answer 0"
        assert received_params["item_id"] == "item-0"
        assert received_params["difficulty"] == "easy"

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.get_config")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_no_scores_pushed_when_result_is_none(
        self, mock_client_cls, mock_get_config, mock_get_score, mock_dataset_items
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_dataset_raw.return_value = mock_dataset_items

        config = MagicMock()
        config.score_name_prefix = "fasteval_"
        mock_get_config.return_value = config
        mock_get_score.return_value = None

        from fasteval_langfuse.decorators import _execute_dataset_evaluation

        def my_test_func(**kw):
            pass

        _execute_dataset_evaluation(
            func=my_test_func,
            is_async=False,
            name="test-dataset",
            version=None,
            args=(),
            kwargs={},
        )

        for item in mock_dataset_items:
            item._root_span.score_trace.assert_not_called()

    @patch("fasteval.core.scoring.get_last_score_result")
    @patch("fasteval_langfuse.decorators.get_config")
    @patch("fasteval_langfuse.decorators.LangfuseClient")
    def test_all_items_share_same_run_name(
        self, mock_client_cls, mock_get_config, mock_get_score, mock_dataset_items
    ):
        """All items in one evaluation share the same run_name for Experiments grouping."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.fetch_dataset_raw.return_value = mock_dataset_items

        config = MagicMock()
        config.score_name_prefix = "fasteval_"
        mock_get_config.return_value = config
        mock_get_score.return_value = None

        from fasteval_langfuse.decorators import _execute_dataset_evaluation

        def my_test_func(**kw):
            pass

        _execute_dataset_evaluation(
            func=my_test_func,
            is_async=False,
            name="test-dataset",
            version=None,
            args=(),
            kwargs={},
        )

        run_names = set()
        for item in mock_dataset_items:
            run_call = item.run.call_args
            rn = run_call[1].get("run_name") or run_call[0][0]
            run_names.add(rn)

        assert len(run_names) == 1, "All items should share the same run_name"
