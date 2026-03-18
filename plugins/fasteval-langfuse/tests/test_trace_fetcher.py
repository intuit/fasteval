"""Tests for fasteval_langfuse.trace_fetcher."""

from unittest.mock import MagicMock

from fasteval_langfuse.trace_fetcher import TraceFetcher


def _make_fetcher_with_mock(traces):
    """Create a TraceFetcher with a mocked client."""
    mock_client = MagicMock()
    mock_client.fetch_traces.return_value = traces
    fetcher = TraceFetcher.__new__(TraceFetcher)
    fetcher.client = mock_client
    return fetcher


class TestFetchAndSample:
    def test_basic_fetch(self, sample_traces):
        fetcher = _make_fetcher_with_mock(sample_traces)
        result, total = fetcher.fetch_and_sample()
        assert total == 3
        assert len(result) == 3

    def test_with_time_range(self, sample_traces):
        fetcher = _make_fetcher_with_mock(sample_traces)
        result, total = fetcher.fetch_and_sample(time_range="last_24h")
        assert total == 3

    def test_with_sampling(self, sample_traces):
        from fasteval_langfuse.sampling import RandomSamplingStrategy

        fetcher = _make_fetcher_with_mock(sample_traces)
        sampling = RandomSamplingStrategy(sample_size=1, seed=42)
        result, total = fetcher.fetch_and_sample(sampling=sampling)
        assert total == 3
        assert len(result) == 1

    def test_with_filters(self, sample_traces):
        fetcher = _make_fetcher_with_mock(sample_traces)
        fetcher.fetch_and_sample(
            project="prod",
            filter_tags=["tag1"],
            user_id="u1",
            session_id="s1",
            limit=10,
        )
        call_kwargs = fetcher.client.fetch_traces.call_args[1]
        assert call_kwargs["project"] == "prod"
        assert call_kwargs["tags"] == ["tag1"]
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["limit"] == 10


class TestMapTraceToParams:
    def test_basic_mapping(self, sample_traces):
        fetcher = _make_fetcher_with_mock([])
        params = fetcher.map_trace_to_params(sample_traces[0])
        assert params["trace_id"] == "trace-1"
        assert params["input"] == "What is Python?"
        assert params["output"] == "Python is a programming language"
        assert params["metadata"] == {"user_type": "free"}

    def test_dict_input(self):
        fetcher = _make_fetcher_with_mock([])
        trace = {
            "id": "t-1",
            "input": {"query": "hello"},
            "output": {"response": "world"},
            "metadata": {},
        }
        params = fetcher.map_trace_to_params(trace)
        assert params["input"] == "hello"
        assert params["output"] == "world"

    def test_dict_input_fallback(self):
        fetcher = _make_fetcher_with_mock([])
        trace = {
            "id": "t-1",
            "input": {"custom_key": "value"},
            "output": {"custom_key": "value"},
            "metadata": {},
        }
        params = fetcher.map_trace_to_params(trace)
        assert "custom_key" in params["input"]

    def test_context_extraction(self, sample_traces):
        fetcher = _make_fetcher_with_mock([])
        # trace-3 has context in metadata
        params = fetcher.map_trace_to_params(sample_traces[2])
        assert params["context"] == ["RAG combines retrieval with generation"]

    def test_no_context(self, sample_traces):
        fetcher = _make_fetcher_with_mock([])
        params = fetcher.map_trace_to_params(sample_traces[0])
        assert params["context"] is None
