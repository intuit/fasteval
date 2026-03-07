"""Code-as-Judge metric: wrap a plain function as an evaluation metric."""

import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

from fasteval.metrics.base import Metric
from fasteval.models.evaluation import EvalInput, MetricResult


class CodeJudgeMetric(Metric):
    """
    Wraps a user-defined function as a Metric.

    The function can accept any subset of EvalInput fields as parameters
    (matched by name), or the full EvalInput object via an ``eval_input``
    parameter. Supports sync and async functions.

    Return types accepted:
        - float: score only (0.0 to 1.0)
        - tuple[float, str]: (score, reasoning)
        - dict: {"score": float, "reasoning": str, "details": dict}
        - MetricResult: returned as-is

    Example:
        def check_length(actual_output: str) -> float:
            return 1.0 if len(actual_output.split()) < 100 else 0.5

        metric = CodeJudgeMetric(func=check_length, name="check_length", threshold=0.8)
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        threshold: float = 0.5,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, threshold=threshold, weight=weight)
        self.func = func
        self._is_async = inspect.iscoroutinefunction(func)
        self._sig = inspect.signature(func)

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        kwargs = self._build_kwargs(eval_input)

        if self._is_async:
            raw = await self.func(**kwargs)
        else:
            raw = self.func(**kwargs)

        return self._normalize_result(raw)

    def _build_kwargs(self, eval_input: EvalInput) -> Dict[str, Any]:
        """Map EvalInput fields to the function's declared parameters."""
        params = self._sig.parameters
        input_dict = eval_input.model_dump()

        if "eval_input" in params:
            return {"eval_input": eval_input}

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        if has_var_keyword:
            kwargs: Dict[str, Any] = {}
            for pname, param in params.items():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    continue
                if pname in input_dict:
                    kwargs[pname] = input_dict[pname]
            remaining = {k: v for k, v in input_dict.items() if k not in kwargs}
            kwargs.update(remaining)
            return kwargs

        kwargs = {}
        for pname in params:
            if pname in input_dict:
                kwargs[pname] = input_dict[pname]
        return kwargs

    def _normalize_result(
        self,
        raw: Union[float, int, Tuple[float, str], Dict[str, Any], MetricResult],
    ) -> MetricResult:
        """Convert the function's return value into a MetricResult."""
        if isinstance(raw, MetricResult):
            return raw

        if isinstance(raw, (int, float)):
            score = float(raw)
            return MetricResult(
                metric_name=self.name,
                score=score,
                passed=self._determine_pass(score),
                threshold=self.threshold,
            )

        if isinstance(raw, tuple):
            score = float(raw[0])
            reasoning = str(raw[1]) if len(raw) > 1 else None
            return MetricResult(
                metric_name=self.name,
                score=score,
                passed=self._determine_pass(score),
                threshold=self.threshold,
                reasoning=reasoning,
            )

        if isinstance(raw, dict):
            score = float(raw["score"])
            return MetricResult(
                metric_name=self.name,
                score=score,
                passed=self._determine_pass(score),
                threshold=self.threshold,
                reasoning=raw.get("reasoning"),
                details=raw.get("details", {}),
            )

        raise TypeError(
            f"Judge function {self.func.__name__!r} returned unsupported type "
            f"{type(raw).__name__}. Expected float, tuple, dict, or MetricResult."
        )
