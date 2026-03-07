"""fasteval metrics module."""

from fasteval.metrics.base import Metric
from fasteval.metrics.code_judge import CodeJudgeMetric
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
    GEvalMetric,
    HallucinationMetric,
    HelpfulnessMetric,
    InstructionFollowingMetric,
    RelevanceMetric,
    ToxicityMetric,
)

# Alias for BaseLLMMetric
LLMMetric = BaseLLMMetric

# Vision metrics (lazy import to avoid dependency issues)
try:
    from fasteval.metrics.vision import (
        BaseVisionMetric,
        ChartInterpretationMetric,
        ImageFaithfulnessMetric,
        ImageQualityMetric,
        ImageUnderstandingMetric,
        OCRAccuracyMetric,
        PromptAdherenceMetric,
        SafetyCheckMetric,
        VisualGroundingMetric,
    )

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

# Audio metrics (lazy import to avoid dependency issues)
try:
    from fasteval.metrics.audio import (
        AudioSentimentMetric,
        CharacterErrorRateMetric,
        MatchErrorRateMetric,
        SpeakerDiarizationMetric,
        TranscriptionAccuracyMetric,
        WordErrorRateMetric,
    )

    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

# Multi-modal metrics
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

    _MULTIMODAL_AVAILABLE = True
except ImportError:
    _MULTIMODAL_AVAILABLE = False

__all__ = [
    # Base
    "Metric",
    "CodeJudgeMetric",
    "LLMMetric",
    # LLM Metrics
    "CorrectnessMetric",
    "HallucinationMetric",
    "RelevanceMetric",
    "CriteriaMetric",
    "GEvalMetric",  # Alias for CriteriaMetric
    "ToxicityMetric",
    "BiasMetric",
    # Quality Metrics
    "ConcisenessMetric",
    "CoherenceMetric",
    "CompletenessMetric",
    "HelpfulnessMetric",
    "InstructionFollowingMetric",
    # RAG Metrics
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "AnswerCorrectnessMetric",
    # Deterministic
    "RougeMetric",
    "ExactMatchMetric",
    "ContainsMetric",
    "JsonMetric",
    "RegexMetric",
    # Tool Trajectory
    "ToolCallAccuracyMetric",
    "ToolSequenceMetric",
    "ToolArgsMatchMetric",
    # Conversation
    "ContextRetentionMetric",
    "ConsistencyMetric",
    "TopicDriftMetric",
    # Vision Metrics (when available)
    "BaseVisionMetric",
    "ImageUnderstandingMetric",
    "OCRAccuracyMetric",
    "ChartInterpretationMetric",
    "VisualGroundingMetric",
    "ImageFaithfulnessMetric",
    "ImageQualityMetric",
    "PromptAdherenceMetric",
    "SafetyCheckMetric",
    # Audio Metrics (when available)
    "WordErrorRateMetric",
    "CharacterErrorRateMetric",
    "MatchErrorRateMetric",
    "TranscriptionAccuracyMetric",
    "SpeakerDiarizationMetric",
    "AudioSentimentMetric",
    # Multi-Modal Metrics (when available)
    "MultiModalFaithfulnessMetric",
    "TableExtractionMetric",
    "FigureReferenceMetric",
    "CrossModalCoherenceMetric",
    "DocumentUnderstandingMetric",
    "CLIPScoreMetric",
    "AestheticScoreMetric",
]
