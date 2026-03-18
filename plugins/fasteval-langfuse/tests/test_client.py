"""Tests for fasteval_langfuse.client."""

from unittest.mock import MagicMock, patch

import pytest

from fasteval_langfuse.client import LangfuseClient


def _make_mock_client():
    """Create a LangfuseClient with mocked internals."""
    client = LangfuseClient.__new__(LangfuseClient)
    client.public_key = "pk-test"
    client.secret_key = "sk-test"
    client.host = "https://cloud.langfuse.com"
    client._client = MagicMock()
    return client


class TestLangfuseClientInit:
    def test_missing_credentials(self):
        with (
            patch.dict(
                "os.environ",
                {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""},
                clear=False,
            ),
            patch(
                "fasteval_langfuse.config.get_config",
                return_value=MagicMock(public_key=None, secret_key=None, host="h"),
            ),
        ):
            with pytest.raises(ValueError, match="Langfuse credentials required"):
                LangfuseClient()


class TestFetchTraces:
    def test_fetch_traces_basic(self):
        client = _make_mock_client()
        mock_trace = MagicMock()
        mock_trace.id = "t-1"
        mock_trace.timestamp = "2026-01-01"
        mock_trace.name = "query"
        mock_trace.user_id = "u1"
        mock_trace.session_id = "s1"
        mock_trace.tags = ["prod"]
        mock_trace.metadata = {"key": "val"}
        mock_trace.input = "hello"
        mock_trace.output = "world"
        mock_trace.scores = []

        client._client.fetch_traces.return_value = MagicMock(data=[mock_trace])

        result = client.fetch_traces(project="test")
        assert len(result) == 1
        assert result[0]["id"] == "t-1"
        assert result[0]["input"] == "hello"
        assert result[0]["output"] == "world"

    def test_fetch_traces_with_filters(self):
        client = _make_mock_client()
        client._client.fetch_traces.return_value = MagicMock(data=[])

        client.fetch_traces(
            project="prod",
            tags=["support"],
            user_id="u1",
            session_id="s1",
            from_timestamp="2026-01-01",
            to_timestamp="2026-01-02",
            limit=10,
        )

        call_kwargs = client._client.fetch_traces.call_args
        assert call_kwargs[1]["tags"] == ["support"]
        assert call_kwargs[1]["user_id"] == "u1"
        assert call_kwargs[1]["session_id"] == "s1"
        assert call_kwargs[1]["limit"] == 10


class TestPushScore:
    def test_push_score(self):
        client = _make_mock_client()
        client.push_score(trace_id="t-1", name="correctness", value=0.9, comment="Good")
        client._client.score.assert_called_once_with(
            trace_id="t-1", name="correctness", value=0.9, comment="Good"
        )


class TestFetchDataset:
    def test_fetch_dataset(self):
        client = _make_mock_client()
        mock_item = MagicMock()
        mock_item.id = "item-1"
        mock_item.input = "question"
        mock_item.expected_output = "answer"
        mock_item.metadata = {}
        mock_item.version = "v1"

        mock_dataset = MagicMock()
        mock_dataset.items = [mock_item]
        client._client.get_dataset.return_value = mock_dataset

        result = client.fetch_dataset(name="test-ds")
        assert len(result) == 1
        assert result[0]["id"] == "item-1"
        assert result[0]["input"] == "question"
        assert result[0]["expected_output"] == "answer"


class TestFlush:
    def test_flush(self):
        client = _make_mock_client()
        client.flush()
        client._client.flush.assert_called_once()
