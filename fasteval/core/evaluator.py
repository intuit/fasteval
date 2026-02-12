"""Evaluator that runs metrics on EvalInputs."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from fasteval.metrics.base import Metric
from fasteval.metrics.conversation import (
    ConsistencyMetric,
    ContextRetentionMetric,
    TopicDriftMetric,
)
from fasteval.metrics.deterministic import (  # Tool Trajectory Metrics
    ContainsMetric,
    ExactMatchMetric,
    JsonMetric,
    RegexMetric,
    RougeMetric,
    ToolArgsMatchMetric,
    ToolCallAccuracyMetric,
    ToolSequenceMetric,
)
from fasteval.metrics.llm import (  # RAG Metrics; Quality Metrics
    AnswerCorrectnessMetric,
    BaseLLMMetric,
    BiasMetric,
    CoherenceMetric,
    CompletenessMetric,
    ConcisenessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    CorrectnessMetric,
    CriteriaMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    HelpfulnessMetric,
    InstructionFollowingMetric,
    RelevanceMetric,
    ToxicityMetric,
)
from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import EvalInput, EvalResult, MetricResult

logger = logging.getLogger(__name__)


# Metric type to class mapping
METRIC_REGISTRY: Dict[str, Type[Metric]] = {
    # LLM-based
    "correctness": CorrectnessMetric,
    "hallucination": HallucinationMetric,
    "relevance": RelevanceMetric,
    "criteria": CriteriaMetric,
    "geval": CriteriaMetric,  # Alias for backwards compatibility
    "toxicity": ToxicityMetric,
    "bias": BiasMetric,
    # Quality Metrics
    "conciseness": ConcisenessMetric,
    "coherence": CoherenceMetric,
    "completeness": CompletenessMetric,
    "helpfulness": HelpfulnessMetric,
    "instruction_following": InstructionFollowingMetric,
    # RAG Metrics
    "faithfulness": FaithfulnessMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
    "answer_correctness": AnswerCorrectnessMetric,
    # Deterministic
    "rouge": RougeMetric,
    "exact_match": ExactMatchMetric,
    "contains": ContainsMetric,
    "json": JsonMetric,
    "regex": RegexMetric,
    # Tool Trajectory
    "tool_call_accuracy": ToolCallAccuracyMetric,
    "tool_sequence": ToolSequenceMetric,
    "tool_args_match": ToolArgsMatchMetric,
    # Conversation
    "context_retention": ContextRetentionMetric,
    "consistency": ConsistencyMetric,
    "topic_drift": TopicDriftMetric,
}

# Register vision metrics if available
try:
    from fasteval.metrics.vision import (
        ChartInterpretationMetric,
        ImageFaithfulnessMetric,
        ImageQualityMetric,
        ImageUnderstandingMetric,
        OCRAccuracyMetric,
        PromptAdherenceMetric,
        SafetyCheckMetric,
        VisualGroundingMetric,
    )

    METRIC_REGISTRY.update({
        "image_understanding": ImageUnderstandingMetric,
        "ocr_accuracy": OCRAccuracyMetric,
        "chart_interpretation": ChartInterpretationMetric,
        "visual_grounding": VisualGroundingMetric,
        "image_faithfulness": ImageFaithfulnessMetric,
        "image_quality": ImageQualityMetric,
        "prompt_adherence": PromptAdherenceMetric,
        "safety_check": SafetyCheckMetric,
    })
except ImportError:
    pass  # Vision metrics not available

# Register audio metrics if available
try:
    from fasteval.metrics.audio import (
        AudioSentimentMetric,
        CharacterErrorRateMetric,
        MatchErrorRateMetric,
        SpeakerDiarizationMetric,
        TranscriptionAccuracyMetric,
        WordErrorRateMetric,
    )

    METRIC_REGISTRY.update({
        "word_error_rate": WordErrorRateMetric,
        "character_error_rate": CharacterErrorRateMetric,
        "match_error_rate": MatchErrorRateMetric,
        "transcription_accuracy": TranscriptionAccuracyMetric,
        "speaker_diarization": SpeakerDiarizationMetric,
        "audio_sentiment": AudioSentimentMetric,
    })
except ImportError:
    pass  # Audio metrics not available

# Register multi-modal metrics if available
try:
    from fasteval.metrics.multimodal import (
        AestheticScoreMetric,
        CLIPScoreMetric,
        CrossModalCoherenceMetric,
        DocumentUnderstandingMetric,
        FigureReferenceMetric,
        MultiModalFaithfulnessMetric,
        TableExtractionMetric,
    )

    METRIC_REGISTRY.update({
        "multimodal_faithfulness": MultiModalFaithfulnessMetric,
        "table_extraction": TableExtractionMetric,
        "figure_reference": FigureReferenceMetric,
        "cross_modal_coherence": CrossModalCoherenceMetric,
        "document_understanding": DocumentUnderstandingMetric,
        "clip_score": CLIPScoreMetric,
        "aesthetic_score": AestheticScoreMetric,
    })
except ImportError:
    pass  # Multi-modal metrics not available


class EvaluatorConfig(BaseModel):
    """Configuration for the evaluator."""

    fail_fast: bool = False  # Stop on first failure
    parallel: bool = True  # Run metrics in parallel
    cache_enabled: bool = True


class Evaluator:
    """
    Runs metrics on EvalInput and aggregates results.

    Example:
        evaluator = Evaluator()
        result = await evaluator.evaluate(
            eval_input=EvalInput(actual_output="...", expected_output="..."),
            metrics=[MetricConfig(metric_type="correctness", name="correctness", threshold=0.8)]
        )
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()
        self._metric_cache: Dict[str, Metric] = {}

    def _create_metric(self, config: MetricConfig) -> Metric:
        """Create a metric instance from config."""
        # Check for custom metric instance
        if config.metric_type == "custom" and "instance" in config.config:
            return config.config["instance"]

        # Get metric class
        metric_cls = METRIC_REGISTRY.get(config.metric_type)
        if not metric_cls:
            raise ValueError(f"Unknown metric type: {config.metric_type}")

        # Build kwargs
        kwargs: Dict[str, Any] = {
            "name": config.name,
            "threshold": config.threshold,
            "weight": config.weight,
        }

        # Add LLM client if specified
        if config.llm_client:
            kwargs["llm_client"] = config.llm_client
        elif config.llm_config and config.llm_config.get("model"):
            kwargs["model"] = config.llm_config["model"]

        # Add metric-specific config
        metric_config = config.config.copy()

        # Handle special cases
        if config.metric_type == "json" and "pydantic_model" in metric_config:
            kwargs["model"] = metric_config.pop("pydantic_model")
        elif config.metric_type in ("criteria", "geval"):
            kwargs["criteria"] = metric_config.pop("criteria", "")
            if "evaluation_steps" in metric_config:
                kwargs["evaluation_steps"] = metric_config.pop("evaluation_steps")
        elif config.metric_type == "instruction_following":
            if "instructions" in metric_config:
                kwargs["instructions"] = metric_config.pop("instructions")

        kwargs.update(metric_config)

        return metric_cls(**kwargs)

    async def _evaluate_metric(
        self,
        metric: Metric,
        eval_input: EvalInput,
    ) -> MetricResult:
        """Run a single metric evaluation."""
        try:
            return await metric.evaluate(eval_input)
        except Exception as e:
            logger.error(f"Metric {metric.name} failed: {e}")
            return MetricResult(
                metric_name=metric.name,
                score=0.0,
                passed=False,
                threshold=metric.threshold,
                reasoning=f"Evaluation error: {e}",
                details={"error": str(e)},
            )

    async def evaluate(
        self,
        eval_input: EvalInput,
        metrics: List[MetricConfig],
    ) -> EvalResult:
        """
        Evaluate an input against all configured metrics.

        Args:
            eval_input: The input to evaluate
            metrics: List of metric configurations

        Returns:
            EvalResult with all metric results
        """
        start_time = time.time()

        # Create metric instances
        metric_instances = [self._create_metric(m) for m in metrics]

        # Run evaluations
        if self.config.parallel and len(metric_instances) > 1:
            # Run in parallel
            tasks = [self._evaluate_metric(m, eval_input) for m in metric_instances]
            metric_results = await asyncio.gather(*tasks)
        else:
            # Run sequentially
            metric_results = []
            for metric in metric_instances:
                result = await self._evaluate_metric(metric, eval_input)
                metric_results.append(result)
                if self.config.fail_fast and not result.passed:
                    break

        # Compute aggregate
        total_weight = sum(m.weight for m in metric_instances)
        weighted_score = sum(
            r.score * metric_instances[i].weight for i, r in enumerate(metric_results)
        )
        aggregate_score = weighted_score / total_weight if total_weight > 0 else 0.0

        all_passed = all(r.passed for r in metric_results)

        execution_time_ms = (time.time() - start_time) * 1000

        return EvalResult(
            eval_input=eval_input,
            metric_results=list(metric_results),
            passed=all_passed,
            aggregate_score=aggregate_score,
            execution_time_ms=execution_time_ms,
            reference_id=eval_input.reference_id,
        )

    async def evaluate_batch(
        self,
        eval_inputs: List[EvalInput],
        metrics: List[MetricConfig],
    ) -> List[EvalResult]:
        """
        Evaluate multiple inputs.

        Args:
            eval_inputs: List of inputs to evaluate
            metrics: List of metric configurations (applied to all inputs)

        Returns:
            List of EvalResults
        """
        results = []
        for eval_input in eval_inputs:
            result = await self.evaluate(eval_input, metrics)
            results.append(result)
        return results


def create_evaluator(
    fail_fast: bool = False,
    parallel: bool = True,
    cache_enabled: bool = True,
) -> Evaluator:
    """
    Create an evaluator instance.

    Args:
        fail_fast: Stop evaluation on first metric failure
        parallel: Run metrics in parallel
        cache_enabled: Enable caching

    Returns:
        Configured Evaluator instance
    """
    config = EvaluatorConfig(
        fail_fast=fail_fast,
        parallel=parallel,
        cache_enabled=cache_enabled,
    )
    return Evaluator(config)
