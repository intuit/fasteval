"""
FastEval output collectors and reporters.

Provides result collection, aggregation, and export in multiple formats.
"""

from fasteval.collectors.collector import (
    ResultCollector,
    get_collector,
    reset_collector,
)
from fasteval.collectors.reporters.base import OutputReporter
from fasteval.collectors.summary import EvalRunSummary, MetricAggregate, TestCaseSummary

__all__ = [
    "ResultCollector",
    "get_collector",
    "reset_collector",
    "EvalRunSummary",
    "MetricAggregate",
    "TestCaseSummary",
    "OutputReporter",
]
