"""Core result collector for fasteval evaluation runs."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from fasteval.collectors.reporters.html_reporter import HtmlReporter
from fasteval.collectors.reporters.json_reporter import JsonReporter
from fasteval.collectors.summary import EvalRunSummary

if TYPE_CHECKING:
    from fasteval.collectors.reporters.base import OutputReporter
    from fasteval.models.evaluation import EvalResult


class ResultCollector:
    """
    Collects EvalResult objects during an evaluation run.

    Thread-safe. Each call to fe.score() automatically registers
    its result here via the global singleton.

    Usage (standalone):
        collector = get_collector()
        # ... run evaluations with fe.score() ...
        summary = collector.summary()
        collector.report("json", path="results.json")
    """

    def __init__(self) -> None:
        self._results: List[EvalResult] = []
        self._test_names: List[str] = []
        self._reporters: Dict[str, Type[OutputReporter]] = {}
        self._lock = threading.Lock()
        self._register_builtins()

    def collect(self, result: EvalResult, test_name: str = "unknown") -> None:
        """Add a result. Called automatically by fe.score()."""
        with self._lock:
            self._results.append(result)
            self._test_names.append(test_name)

    @property
    def results(self) -> List[EvalResult]:
        """All collected results (read-only copy)."""
        return list(self._results)

    def summary(self) -> EvalRunSummary:
        """Compute aggregate statistics from collected results."""
        return EvalRunSummary.from_results(self._results, self._test_names)

    def report(
        self,
        format: str,
        *,
        path: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        """
        Generate a report in the given format.

        Args:
            format: "json", "html", or a custom registered name.
            path: Optional file path to write to. If None, returns string only.
            **kwargs: Format-specific options passed to reporter constructor.

        Returns:
            The report content as a string.
        """
        reporter_cls = self._reporters.get(format)
        if not reporter_cls:
            available = list(self._reporters.keys())
            raise ValueError(f"Unknown format: {format!r}. Available: {available}")
        reporter = reporter_cls(**kwargs)
        content = reporter.generate(self.summary(), self._results)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding="utf-8")
        return content

    def register_reporter(self, name: str, reporter_cls: Type[OutputReporter]) -> None:
        """Register a custom reporter format."""
        self._reporters[name] = reporter_cls

    def reset(self) -> None:
        """Clear all collected results."""
        with self._lock:
            self._results.clear()
            self._test_names.clear()

    def _register_builtins(self) -> None:
        self._reporters = {
            "json": JsonReporter,
            "html": HtmlReporter,
        }


_collector: Optional[ResultCollector] = None


def get_collector() -> ResultCollector:
    """Get (or create) the global result collector."""
    global _collector
    if _collector is None:
        _collector = ResultCollector()
    return _collector


def reset_collector() -> None:
    """Reset the global collector. Used between test sessions."""
    global _collector
    if _collector is not None:
        _collector.reset()
