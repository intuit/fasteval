"""JSON output reporter."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fasteval.collectors.reporters.base import OutputReporter
from fasteval.collectors.summary import EvalRunSummary
from fasteval.models.evaluation import EvalResult


class JsonReporter(OutputReporter):
    """Generates a JSON report with summary and per-test results."""

    def __init__(self, *, indent: int = 2, include_inputs: bool = True) -> None:
        self.indent = indent
        self.include_inputs = include_inputs

    def generate(
        self, summary: EvalRunSummary, results: List[EvalResult]
    ) -> str:
        output: Dict[str, Any] = {
            "summary": summary.model_dump(mode="json"),
            "results": [],
        }

        for i, result in enumerate(results):
            test_name = (
                summary.test_summaries[i].test_name
                if i < len(summary.test_summaries)
                else "unknown"
            )
            result_dict = result.model_dump(mode="json")
            if not self.include_inputs:
                result_dict.pop("eval_input", None)
            result_dict["test_name"] = test_name
            output["results"].append(result_dict)

        return json.dumps(output, indent=self.indent, default=str)
