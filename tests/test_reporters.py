"""Tests for fasteval.collectors.reporters (json + html)."""

import json

import pytest

from fasteval.collectors.reporters.html_reporter import HtmlReporter
from fasteval.collectors.reporters.json_reporter import JsonReporter
from fasteval.collectors.summary import EvalRunSummary
from fasteval.models.evaluation import EvalInput, EvalResult, MetricResult


def _make_metric(name="m1", score=0.8, passed=True, threshold=0.5, reasoning="OK"):
    return MetricResult(
        metric_name=name,
        score=score,
        passed=passed,
        threshold=threshold,
        reasoning=reasoning,
    )


def _make_result(
    passed=True, aggregate_score=0.8, metrics=None, execution_time_ms=10.0
):
    return EvalResult(
        eval_input=EvalInput(actual_output="test output", expected_output="expected"),
        metric_results=metrics or [],
        passed=passed,
        aggregate_score=aggregate_score,
        execution_time_ms=execution_time_ms,
    )


def _make_summary_and_results():
    mr1 = _make_metric(name="correctness", score=0.9, passed=True)
    mr2 = _make_metric(
        name="relevance", score=0.3, passed=False, reasoning="Not relevant"
    )
    r1 = _make_result(passed=True, metrics=[mr1])
    r2 = _make_result(passed=False, aggregate_score=0.3, metrics=[mr2])
    summary = EvalRunSummary.from_results([r1, r2], ["test_pass", "test_fail"])
    return summary, [r1, r2]


# ── JsonReporter ─────────────────────────────────────────────────────────────


class TestJsonReporter:
    def test_generates_valid_json(self):
        summary, results = _make_summary_and_results()
        reporter = JsonReporter()
        output = reporter.generate(summary, results)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_includes_summary_and_results(self):
        summary, results = _make_summary_and_results()
        reporter = JsonReporter()
        parsed = json.loads(reporter.generate(summary, results))
        assert "summary" in parsed
        assert "results" in parsed
        assert len(parsed["results"]) == 2

    def test_test_name_assigned(self):
        summary, results = _make_summary_and_results()
        reporter = JsonReporter()
        parsed = json.loads(reporter.generate(summary, results))
        assert parsed["results"][0]["test_name"] == "test_pass"
        assert parsed["results"][1]["test_name"] == "test_fail"

    def test_include_inputs_false(self):
        summary, results = _make_summary_and_results()
        reporter = JsonReporter(include_inputs=False)
        parsed = json.loads(reporter.generate(summary, results))
        for r in parsed["results"]:
            assert "eval_input" not in r

    def test_custom_indent(self):
        summary, results = _make_summary_and_results()
        reporter = JsonReporter(indent=4)
        output = reporter.generate(summary, results)
        # 4-space indent should produce more whitespace than 2-space
        assert "    " in output

    def test_empty_results(self):
        summary = EvalRunSummary.from_results([], [])
        reporter = JsonReporter()
        output = reporter.generate(summary, [])
        parsed = json.loads(output)
        assert parsed["results"] == []


# ── HtmlReporter ─────────────────────────────────────────────────────────────


class TestHtmlReporter:
    def test_generates_html(self):
        summary, results = _make_summary_and_results()
        reporter = HtmlReporter()
        output = reporter.generate(summary, results)
        assert "<!DOCTYPE html>" in output
        assert "<html" in output
        assert "</html>" in output

    def test_contains_key_sections(self):
        summary, results = _make_summary_and_results()
        reporter = HtmlReporter()
        output = reporter.generate(summary, results)
        assert "FastEval Evaluation Report" in output
        assert "Metric Breakdown" in output
        assert "Test Results" in output

    def test_cards_section(self):
        summary, results = _make_summary_and_results()
        reporter = HtmlReporter()
        output = reporter.generate(summary, results)
        assert "Total Tests" in output
        assert "Passed" in output
        assert "Failed" in output

    def test_metric_table(self):
        summary, results = _make_summary_and_results()
        reporter = HtmlReporter()
        output = reporter.generate(summary, results)
        assert "correctness" in output
        assert "relevance" in output

    def test_pass_fail_badges(self):
        summary, results = _make_summary_and_results()
        reporter = HtmlReporter()
        output = reporter.generate(summary, results)
        assert "PASS" in output
        assert "FAIL" in output

    def test_empty_results(self):
        summary = EvalRunSummary.from_results([], [])
        reporter = HtmlReporter()
        output = reporter.generate(summary, [])
        assert "No test results" in output

    def test_empty_metrics(self):
        r = _make_result(metrics=[])
        summary = EvalRunSummary.from_results([r], ["test1"])
        reporter = HtmlReporter()
        output = reporter.generate(summary, [r])
        assert "No metrics recorded" in output

    def test_reasoning_displayed(self):
        mr = _make_metric(reasoning="Detailed reason here", passed=False)
        r = _make_result(metrics=[mr], passed=False)
        summary = EvalRunSummary.from_results([r], ["test1"])
        reporter = HtmlReporter()
        output = reporter.generate(summary, [r])
        assert "Detailed reason here" in output
