"""Tests for fasteval_langfuse.score_reporter."""

from unittest.mock import MagicMock, patch

from fasteval_langfuse.score_reporter import ScoreReporter


def _make_reporter():
    """Create a ScoreReporter with mocked client and config."""
    mock_client = MagicMock()
    reporter = ScoreReporter.__new__(ScoreReporter)
    reporter.client = mock_client
    reporter.config = MagicMock()
    reporter.config.auto_push_scores = True
    reporter.config.score_name_prefix = "fasteval_"
    return reporter


class TestPushEvaluationResult:
    def test_pushes_metric_scores(self):
        reporter = _make_reporter()

        mr1 = MagicMock()
        mr1.metric_name = "correctness"
        mr1.score = 0.9
        mr1.reasoning = "Good"

        mr2 = MagicMock()
        mr2.metric_name = "relevance"
        mr2.score = 0.8
        mr2.reasoning = None

        reporter.push_evaluation_result(
            trace_id="t-1", metric_results=[mr1, mr2], aggregate_score=0.85
        )

        calls = reporter.client.push_score.call_args_list
        assert len(calls) == 3  # 2 metrics + 1 aggregate
        assert calls[0][1]["name"] == "fasteval_correctness"
        assert calls[0][1]["value"] == 0.9
        assert calls[1][1]["name"] == "fasteval_relevance"
        assert calls[2][1]["name"] == "fasteval_aggregate"
        assert calls[2][1]["value"] == 0.85

    def test_skips_when_auto_push_disabled(self):
        reporter = _make_reporter()
        reporter.config.auto_push_scores = False

        reporter.push_evaluation_result(
            trace_id="t-1", metric_results=[MagicMock()], aggregate_score=0.5
        )

        reporter.client.push_score.assert_not_called()


class TestFlush:
    def test_flush(self):
        reporter = _make_reporter()
        reporter.flush()
        reporter.client.flush.assert_called_once()
