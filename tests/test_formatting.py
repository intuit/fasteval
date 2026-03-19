"""Tests for fasteval.utils.formatting."""

from fasteval.models.evaluation import EvalInput, EvalResult, MetricResult
from fasteval.utils.formatting import format_evaluation_report


def _make_metric_result(
    name="test_metric", score=0.5, passed=True, threshold=0.5, reasoning=""
):
    return MetricResult(
        metric_name=name,
        score=score,
        passed=passed,
        threshold=threshold,
        reasoning=reasoning,
    )


def _make_eval_result(metric_results=None, passed=True, aggregate_score=1.0):
    return EvalResult(
        eval_input=EvalInput(actual_output="actual", expected_output="expected"),
        metric_results=metric_results or [],
        passed=passed,
        aggregate_score=aggregate_score,
    )


class TestFormatEvaluationReport:
    def test_basic_report_structure(self):
        mr = _make_metric_result(score=0.3, passed=False, reasoning="Bad output")
        result = _make_eval_result(metric_results=[mr], passed=False)
        report = format_evaluation_report("test_func", [result])

        assert "FASTEVAL EVALUATION FAILED" in report
        assert "test_func" in report
        assert "0/1 metrics passed" in report

    def test_failed_metric_shows_reasoning(self):
        mr = _make_metric_result(
            score=0.1, passed=False, threshold=0.7, reasoning="Not correct"
        )
        result = _make_eval_result(metric_results=[mr], passed=False)
        report = format_evaluation_report("test_func", [result])

        assert "Not correct" in report
        assert "0.10 / 0.70" in report

    def test_passed_metric_no_reasoning(self):
        mr = _make_metric_result(score=0.9, passed=True)
        result = _make_eval_result(metric_results=[mr])
        report = format_evaluation_report("test_func", [result])

        assert "1/1 metrics passed" in report

    def test_with_eval_inputs(self):
        result = _make_eval_result()
        eval_input = EvalInput(
            actual_output="my actual",
            expected_output="my expected",
            input="my question",
        )
        report = format_evaluation_report("test_func", [result], [eval_input])

        assert "my question" in report
        assert "my expected" in report
        assert "my actual" in report

    def test_without_eval_inputs(self):
        result = _make_eval_result()
        report = format_evaluation_report("test_func", [result])
        # Should still generate without errors
        assert "FASTEVAL EVALUATION FAILED" in report

    def test_empty_metrics(self):
        result = _make_eval_result(metric_results=[])
        report = format_evaluation_report("test_func", [result])
        assert "0/0 metrics passed" in report

    def test_mixed_pass_fail(self):
        mr1 = _make_metric_result(name="m1", score=0.9, passed=True)
        mr2 = _make_metric_result(
            name="m2", score=0.2, passed=False, reasoning="fail reason"
        )
        result = _make_eval_result(metric_results=[mr1, mr2], passed=False)
        report = format_evaluation_report("test_func", [result])

        assert "1/2 metrics passed" in report
        assert "fail reason" in report
