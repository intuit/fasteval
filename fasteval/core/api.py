"""
Programmatic evaluation outside pytest, and custom metric registration.

    result = fe.evaluate(
        actual_output=response,
        expected_output="Paris",
        input="What is the capital of France?",
        metrics=["correctness", "relevance"],
    )
    result.passed, result.aggregate_score

Unlike ``fe.score()``, these never raise on a failing result and don't
need decorators: metrics are named (or given as MetricConfig/Metric
instances), which suits batch jobs, notebooks and scripts.
"""

from __future__ import annotations

import keyword
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from fasteval.metrics.base import Metric
from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import EvalInput, EvalResult

MetricSpec = Union[str, MetricConfig, Metric]

# Names fe.register_metric() created, so they can be re-registered.
_registered: Dict[str, Any] = {}


def _metric_configs(
    metrics: Sequence[MetricSpec],
    threshold: Optional[float],
    llm_client: Any,
    model: Optional[str],
) -> List[MetricConfig]:
    import fasteval

    if isinstance(metrics, (str, MetricConfig, Metric)):
        metrics = [metrics]
    if not metrics:
        raise ValueError("Pass at least one metric, e.g. metrics=['correctness']")
    configs: List[MetricConfig] = []
    for spec in metrics:
        if isinstance(spec, MetricConfig):
            configs.append(spec)
        elif isinstance(spec, Metric):
            configs.append(
                MetricConfig(
                    metric_type="custom",
                    name=spec.name,
                    threshold=spec.threshold,
                    weight=getattr(spec, "weight", 1.0),
                    config={"instance": spec},
                )
            )
        elif isinstance(spec, str):
            decorator = getattr(fasteval, spec, None)
            if decorator is None or not callable(decorator) or spec.startswith("_"):
                raise ValueError(f"Unknown metric {spec!r}. {_available_hint()}")
            kwargs: Dict[str, Any] = {}
            if threshold is not None:
                kwargs["threshold"] = threshold
            if llm_client is not None:
                kwargs["llm_client"] = llm_client
            if model is not None:
                kwargs["model"] = model

            # Build the config through the metric's own decorator so defaults
            # (threshold, settings) match @fe.<name>() exactly.
            def _target() -> None:
                return None

            try:
                decorator(**kwargs)(_target)
            except TypeError as e:
                raise ValueError(
                    f"{spec!r} can't be used by name ({e}); pass a MetricConfig "
                    "or a Metric instance instead"
                ) from e
            attached = getattr(_target, "_fasteval_metrics", None)
            if not attached:
                raise ValueError(f"{spec!r} is not a metric. {_available_hint()}")
            configs.extend(attached)
        else:
            raise TypeError(
                f"metrics entries must be names, MetricConfig or Metric, got {type(spec).__name__}"
            )
    return configs


def _available_hint() -> str:
    from fasteval.core.evaluator import METRIC_REGISTRY

    names = sorted(n for n in METRIC_REGISTRY if n not in ("custom", "prompt_judge"))
    return f"Available: {', '.join(names)}"


async def evaluate_async(
    actual_output: Optional[str] = None,
    expected_output: Optional[str] = None,
    *,
    metrics: Sequence[MetricSpec],
    threshold: Optional[float] = None,
    llm_client: Any = None,
    model: Optional[str] = None,
    **fields: Any,
) -> EvalResult:
    """
    Evaluate one output against named metrics (async). Never raises on a
    failing result; inspect ``result.passed``.

    Args:
        actual_output: The output being evaluated
        expected_output: Optional reference answer
        metrics: Metric names (``"correctness"``), MetricConfig or Metric objects
        threshold: Override every named metric's threshold
        llm_client: Judge client for LLM metrics (default: the default provider)
        model: Judge model name for LLM metrics
        **fields: Other EvalInput fields: input, context, retrieval_context,
            history, tool_calls, expected_tools, metadata, ...
    """
    from fasteval.core.scoring import _evaluator

    eval_input = EvalInput(
        actual_output=actual_output, expected_output=expected_output, **fields
    )
    configs = _metric_configs(metrics, threshold, llm_client, model)
    return await _evaluator.evaluate(eval_input, configs)


def evaluate(
    actual_output: Optional[str] = None,
    expected_output: Optional[str] = None,
    *,
    metrics: Sequence[MetricSpec],
    threshold: Optional[float] = None,
    llm_client: Any = None,
    model: Optional[str] = None,
    **fields: Any,
) -> EvalResult:
    """Synchronous :func:`evaluate_async`. Don't call it from a running event loop."""
    from fasteval.utils.async_helpers import run_async

    return run_async(
        evaluate_async(
            actual_output,
            expected_output,
            metrics=metrics,
            threshold=threshold,
            llm_client=llm_client,
            model=model,
            **fields,
        )
    )


def register_metric(
    name: str,
    metric_class: Type[Metric],
    *,
    default_threshold: float = 0.5,
    overwrite: bool = False,
) -> Any:
    """
    Register a custom Metric class under ``name`` and create its decorator.

        fe.register_metric("legal_compliance", LegalComplianceMetric)

        @fe.legal_compliance(jurisdiction="US", threshold=0.85)
        def test_legal():
            fe.score(agent("Can I sue my landlord?"))

    Extra decorator keyword arguments are passed to the metric's
    constructor. The name also works in ``fe.evaluate(metrics=[...])``.
    Returns the decorator.
    """
    import fasteval
    from fasteval.core.decorators import _metric_decorator_factory
    from fasteval.core.evaluator import METRIC_REGISTRY

    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"Invalid metric name {name!r}: use a Python identifier")
    if not (isinstance(metric_class, type) and issubclass(metric_class, Metric)):
        raise TypeError(
            "metric_class must be a subclass of fasteval.metrics.base.Metric"
        )
    clashes = name in METRIC_REGISTRY or hasattr(fasteval, name)
    if clashes and name not in _registered and not overwrite:
        raise ValueError(
            f"{name!r} is already a fasteval name; choose another or pass overwrite=True"
        )
    METRIC_REGISTRY[name] = metric_class
    decorator = _metric_decorator_factory(name, name, default_threshold)
    _registered[name] = decorator
    setattr(fasteval, name, decorator)
    return decorator
