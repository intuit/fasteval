"""
fasteval - LLM Evaluation Library

A decorator-first evaluation library for LLMs and AI agents.
Compatible with pytest for test discovery and execution.

Example:
    import fasteval as fe

    @fe.correctness(threshold=0.8)
    def test_qa():
        response = agent("What is 2+2?")
        fe.score(response, "4", input="What is 2+2?")
        # Evaluation runs immediately - raises EvaluationFailedError if fails

Run with pytest:
    pytest tests/

"""

__version__ = "0.1.0a0"

# === Core API ===

from fasteval.cache.memory import MemoryCache, clear_cache, get_cache
from fasteval.core.decorators import (  # LLM Metrics; Quality Metrics; RAG Metrics; Deterministic Metrics; Tool Trajectory; Conversation Metrics; Generic; Data Decorators; Human Review; Stack; Vision; Audio; Multi-Modal
    answer_correctness,
    bias,
    coherence,
    completeness,
    conciseness,
    consistency,
    contains,
    context_retention,
    contextual_precision,
    contextual_recall,
    conversation,
    correctness,
    criteria,
    csv,
    exact_match,
    faithfulness,
    g_eval,
    geval,
    hallucination,
    helpfulness,
    human_review,
    instruction_following,
    json,
    metric,
    regex,
    relevance,
    rouge,
    stack,
    tool_args_match,
    tool_call_accuracy,
    tool_sequence,
    topic_drift,
    toxicity,
    traces,
    # Vision metrics
    image_understanding,
    ocr_accuracy,
    chart_interpretation,
    visual_grounding,
    image_faithfulness,
    image_quality,
    prompt_adherence,
    safety_check,
    # Audio metrics
    word_error_rate,
    character_error_rate,
    match_error_rate,
    transcription_accuracy,
    speaker_diarization,
    audio_sentiment,
    # Multi-modal metrics
    multimodal_faithfulness,
    table_extraction,
    figure_reference,
    cross_modal_coherence,
    document_understanding,
    clip_score,
    aesthetic_score,
)
from fasteval.core.evaluator import Evaluator, create_evaluator
from fasteval.core.scoring import score
from fasteval.metrics.base import Metric
from fasteval.metrics import LLMMetric
from fasteval.metrics.conversation import (
    ConsistencyMetric,
    ContextRetentionMetric,
    TopicDriftMetric,
)
from fasteval.metrics.deterministic import (
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
    GEvalMetric,
    HallucinationMetric,
    HelpfulnessMetric,
    InstructionFollowingMetric,
    RelevanceMetric,
    ToxicityMetric,
)
from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import (
    EvalInput,
    EvalResult,
    EvaluationFailedError,
    ExpectedTool,
    HumanReviewResult,
    MetricResult,
    ToolCall,
)
from fasteval.models.multimodal import (
    AudioInput,
    GeneratedImage,
    ImageInput,
    MultiModalContext,
)
from fasteval.providers.base import LLMClient
from fasteval.providers.openai import OpenAIClient
from fasteval.providers.registry import (
    create_provider_for_model,
    get_default_provider,
    set_default_provider,
)

# === Metric Decorators ===


# === Models ===


# === Metrics (for custom metrics) ===


# === Providers ===


# === Cache ===


# === Evaluator ===


__all__ = [
    # Version
    "__version__",
    # Core API
    "score",
    # Metric Decorators
    "correctness",
    "hallucination",
    "relevance",
    "criteria",
    "geval",  # Alias for criteria
    "g_eval",  # Alias for criteria
    "toxicity",
    "bias",
    # Quality Decorators
    "conciseness",
    "coherence",
    "completeness",
    "helpfulness",
    "instruction_following",
    # RAG Decorators
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "answer_correctness",
    # Deterministic Decorators
    "rouge",
    "exact_match",
    "contains",
    "json",
    "regex",
    # Tool Trajectory Decorators
    "tool_call_accuracy",
    "tool_sequence",
    "tool_args_match",
    # Conversation Decorators
    "context_retention",
    "consistency",
    "topic_drift",
    "metric",
    # Data Decorators
    "csv",
    "conversation",
    "traces",
    # Human Review
    "human_review",
    # Metric Stack
    "stack",
    # Vision Metric Decorators
    "image_understanding",
    "ocr_accuracy",
    "chart_interpretation",
    "visual_grounding",
    "image_faithfulness",
    "image_quality",
    "prompt_adherence",
    "safety_check",
    # Audio Metric Decorators
    "word_error_rate",
    "character_error_rate",
    "match_error_rate",
    "transcription_accuracy",
    "speaker_diarization",
    "audio_sentiment",
    # Multi-Modal Metric Decorators
    "multimodal_faithfulness",
    "table_extraction",
    "figure_reference",
    "cross_modal_coherence",
    "document_understanding",
    "clip_score",
    "aesthetic_score",
    # Models
    "EvalInput",
    "EvalResult",
    "EvaluationFailedError",
    "ExpectedTool",
    "HumanReviewResult",
    "MetricResult",
    "MetricConfig",
    "ToolCall",
    # Multi-Modal Models
    "ImageInput",
    "AudioInput",
    "MultiModalContext",
    "GeneratedImage",
    # Metrics (base classes)
    "Metric",
    "BaseLLMMetric",
    "LLMMetric",
    "CorrectnessMetric",
    "HallucinationMetric",
    "RelevanceMetric",
    "CriteriaMetric",
    "GEvalMetric",  # Alias for CriteriaMetric
    "ToxicityMetric",
    "BiasMetric",
    # Quality Metric Classes
    "ConcisenessMetric",
    "CoherenceMetric",
    "CompletenessMetric",
    "HelpfulnessMetric",
    "InstructionFollowingMetric",
    # RAG Metric Classes
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "AnswerCorrectnessMetric",
    # Deterministic Metric Classes
    "RougeMetric",
    "ExactMatchMetric",
    "ContainsMetric",
    "JsonMetric",
    "RegexMetric",
    # Tool Trajectory Metric Classes
    "ToolCallAccuracyMetric",
    "ToolSequenceMetric",
    "ToolArgsMatchMetric",
    # Conversation Metric Classes
    "ContextRetentionMetric",
    "ConsistencyMetric",
    "TopicDriftMetric",
    # Providers
    "LLMClient",
    "set_default_provider",
    "get_default_provider",
    "create_provider_for_model",
    "OpenAIClient",
    # Cache
    "MemoryCache",
    "get_cache",
    "clear_cache",
    # Evaluator
    "Evaluator",
    "create_evaluator",
]
