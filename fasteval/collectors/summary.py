"""Aggregate data models for evaluation run summaries."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Dict, List

from pydantic import BaseModel, Field

from fasteval.models.evaluation import EvalResult


class MetricAggregate(BaseModel):
    """Aggregate statistics for a single metric type across all results."""

    metric_name: str
    count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    pass_rate: float = 0.0
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    std_score: float = 0.0


class TestCaseSummary(BaseModel):
    """Summary for a single test case."""

    test_name: str
    passed: bool
    aggregate_score: float
    metric_count: int
    execution_time_ms: float
    error: str | None = None


class EvalRunSummary(BaseModel):
    """Complete summary of an evaluation run."""

    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    pass_rate: float = 0.0
    avg_aggregate_score: float = 0.0
    total_execution_time_ms: float = 0.0

    metric_aggregates: List[MetricAggregate] = Field(default_factory=list)
    test_summaries: List[TestCaseSummary] = Field(default_factory=list)

    timestamp: str = ""

    @classmethod
    def from_results(
        cls, results: List[EvalResult], test_names: List[str]
    ) -> EvalRunSummary:
        """Compute summary from a list of EvalResult objects."""
        if not results:
            return cls(timestamp=datetime.now(timezone.utc).isoformat())

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        scores = [r.aggregate_score for r in results]
        total_time = sum(r.execution_time_ms for r in results)

        # Per-test summaries
        test_summaries = []
        for result, name in zip(results, test_names):
            test_summaries.append(
                TestCaseSummary(
                    test_name=name,
                    passed=result.passed,
                    aggregate_score=result.aggregate_score,
                    metric_count=len(result.metric_results),
                    execution_time_ms=result.execution_time_ms,
                    error=result.error,
                )
            )

        # Per-metric aggregates
        metric_scores: Dict[str, List[float]] = {}
        metric_passed: Dict[str, int] = {}
        metric_count: Dict[str, int] = {}

        for result in results:
            for mr in result.metric_results:
                name = mr.metric_name
                if name not in metric_scores:
                    metric_scores[name] = []
                    metric_passed[name] = 0
                    metric_count[name] = 0
                metric_scores[name].append(mr.score)
                metric_count[name] += 1
                if mr.passed:
                    metric_passed[name] += 1

        metric_aggregates = []
        for name in sorted(metric_scores.keys()):
            s = metric_scores[name]
            count = metric_count[name]
            pc = metric_passed[name]
            metric_aggregates.append(
                MetricAggregate(
                    metric_name=name,
                    count=count,
                    pass_count=pc,
                    fail_count=count - pc,
                    pass_rate=pc / count if count > 0 else 0.0,
                    avg_score=statistics.mean(s),
                    min_score=min(s),
                    max_score=max(s),
                    std_score=statistics.stdev(s) if len(s) > 1 else 0.0,
                )
            )

        return cls(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            pass_rate=passed / total if total > 0 else 0.0,
            avg_aggregate_score=statistics.mean(scores),
            total_execution_time_ms=total_time,
            metric_aggregates=metric_aggregates,
            test_summaries=test_summaries,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
