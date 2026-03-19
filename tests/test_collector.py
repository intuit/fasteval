"""Tests for fasteval.collectors.collector and fasteval.collectors.summary."""

import json

import pytest

import fasteval.collectors.collector as collector_module
from fasteval.collectors.collector import (
    ResultCollector,
    get_collector,
    reset_collector,
)
from fasteval.collectors.summary import (
    EvalRunSummary,
    MetricAggregate,
    TestCaseSummary,
)
from fasteval.models.evaluation import EvalInput, EvalResult, MetricResult


def _make_result(
    passed=True, aggregate_score=1.0, metrics=None, execution_time_ms=10.0, error=None
):
    return EvalResult(
        eval_input=EvalInput(actual_output="test"),
        metric_results=metrics or [],
        passed=passed,
        aggregate_score=aggregate_score,
        execution_time_ms=execution_time_ms,
        error=error,
    )


def _make_metric(name="m1", score=0.8, passed=True, threshold=0.5):
    return MetricResult(
        metric_name=name, score=score, passed=passed, threshold=threshold
    )


# ── ResultCollector ──────────────────────────────────────────────────────────


class TestResultCollector:
    def test_init(self):
        collector = ResultCollector()
        assert len(collector.results) == 0

    def test_collect_and_results(self):
        collector = ResultCollector()
        r1 = _make_result()
        r2 = _make_result(passed=False, aggregate_score=0.0)
        collector.collect(r1, "test1")
        collector.collect(r2, "test2")
        assert len(collector.results) == 2

    def test_results_is_copy(self):
        collector = ResultCollector()
        collector.collect(_make_result(), "test1")
        results = collector.results
        results.clear()
        assert len(collector.results) == 1

    def test_summary(self):
        collector = ResultCollector()
        collector.collect(_make_result(passed=True), "test1")
        collector.collect(_make_result(passed=False, aggregate_score=0.0), "test2")
        summary = collector.summary()
        assert summary.total_tests == 2
        assert summary.passed_tests == 1
        assert summary.failed_tests == 1

    def test_report_json(self):
        collector = ResultCollector()
        collector.collect(_make_result(), "test1")
        content = collector.report("json")
        parsed = json.loads(content)
        assert "summary" in parsed
        assert "results" in parsed

    def test_report_unknown_format(self):
        collector = ResultCollector()
        collector.collect(_make_result(), "test1")
        with pytest.raises(ValueError, match="Unknown format"):
            collector.report("xml")

    def test_register_reporter(self):
        from fasteval.collectors.reporters.base import OutputReporter

        class CustomReporter(OutputReporter):
            def generate(self, summary, results):
                return "custom"

        collector = ResultCollector()
        collector.register_reporter("custom", CustomReporter)
        collector.collect(_make_result(), "test1")
        content = collector.report("custom")
        assert content == "custom"

    def test_reset(self):
        collector = ResultCollector()
        collector.collect(_make_result(), "test1")
        collector.reset()
        assert len(collector.results) == 0

    def test_report_to_file(self, tmp_path):
        collector = ResultCollector()
        collector.collect(_make_result(), "test1")
        filepath = str(tmp_path / "report.json")
        collector.report("json", path=filepath)
        with open(filepath) as f:
            parsed = json.loads(f.read())
        assert "summary" in parsed

    def test_report_html(self):
        collector = ResultCollector()
        mr = _make_metric()
        collector.collect(_make_result(metrics=[mr]), "test1")
        content = collector.report("html")
        assert "FastEval" in content
        assert "<html" in content


# ── Global collector ─────────────────────────────────────────────────────────


class TestGlobalCollector:
    def setup_method(self):
        collector_module._collector = None

    def test_get_collector_singleton(self):
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2

    def test_reset_collector(self):
        collector = get_collector()
        collector.collect(_make_result(), "test1")
        reset_collector()
        assert len(collector.results) == 0

    def test_reset_collector_when_none(self):
        collector_module._collector = None
        reset_collector()  # Should not raise


# ── EvalRunSummary ───────────────────────────────────────────────────────────


class TestEvalRunSummary:
    def test_empty_results(self):
        summary = EvalRunSummary.from_results([], [])
        assert summary.total_tests == 0
        assert summary.timestamp != ""

    def test_single_result(self):
        mr = _make_metric(name="m1", score=0.8, passed=True)
        result = _make_result(
            passed=True, aggregate_score=0.8, metrics=[mr], execution_time_ms=15.0
        )
        summary = EvalRunSummary.from_results([result], ["test1"])
        assert summary.total_tests == 1
        assert summary.passed_tests == 1
        assert summary.failed_tests == 0
        assert summary.pass_rate == 1.0
        assert summary.avg_aggregate_score == 0.8
        assert summary.total_execution_time_ms == 15.0

    def test_multiple_results_with_metrics(self):
        mr1 = _make_metric(name="m1", score=0.9, passed=True)
        mr2 = _make_metric(name="m1", score=0.3, passed=False)
        mr3 = _make_metric(name="m2", score=0.7, passed=True)

        r1 = _make_result(
            passed=True, aggregate_score=0.9, metrics=[mr1], execution_time_ms=10.0
        )
        r2 = _make_result(
            passed=False,
            aggregate_score=0.3,
            metrics=[mr2, mr3],
            execution_time_ms=20.0,
        )

        summary = EvalRunSummary.from_results([r1, r2], ["t1", "t2"])

        assert summary.total_tests == 2
        assert summary.passed_tests == 1
        assert summary.failed_tests == 1
        assert summary.pass_rate == 0.5
        assert summary.total_execution_time_ms == 30.0

        # Check metric aggregates
        assert len(summary.metric_aggregates) == 2  # m1 and m2

        m1_agg = next(m for m in summary.metric_aggregates if m.metric_name == "m1")
        assert m1_agg.count == 2
        assert m1_agg.pass_count == 1
        assert m1_agg.fail_count == 1
        assert m1_agg.pass_rate == 0.5
        assert m1_agg.min_score == 0.3
        assert m1_agg.max_score == 0.9
        assert m1_agg.std_score > 0

        m2_agg = next(m for m in summary.metric_aggregates if m.metric_name == "m2")
        assert m2_agg.count == 1
        assert m2_agg.std_score == 0.0  # Single value

    def test_test_summaries(self):
        result = _make_result(
            passed=False,
            aggregate_score=0.4,
            execution_time_ms=5.0,
            error="failed",
        )
        summary = EvalRunSummary.from_results([result], ["test_func"])
        assert len(summary.test_summaries) == 1
        ts = summary.test_summaries[0]
        assert ts.test_name == "test_func"
        assert ts.passed is False
        assert ts.aggregate_score == 0.4
        assert ts.error == "failed"
