"""Base class for output reporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fasteval.collectors.summary import EvalRunSummary
    from fasteval.models.evaluation import EvalResult


class OutputReporter(ABC):
    """
    Base class for output format reporters.

    Subclass and implement `generate()` to create custom reporters:

        class SlackReporter(OutputReporter):
            def generate(self, summary, results):
                return f"Eval: {summary.pass_rate:.0%} pass"

        get_collector().register_reporter("slack", SlackReporter)
    """

    @abstractmethod
    def generate(self, summary: EvalRunSummary, results: List[EvalResult]) -> str:
        """Generate report content as a string."""
        ...
