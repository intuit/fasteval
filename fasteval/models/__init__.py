"""fasteval data models."""

from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import (
    EvalInput,
    EvalResult,
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

__all__ = [
    "EvalInput",
    "EvalResult",
    "ExpectedTool",
    "HumanReviewResult",
    "MetricResult",
    "MetricConfig",
    "ToolCall",
    # Multi-modal models
    "ImageInput",
    "AudioInput",
    "MultiModalContext",
    "GeneratedImage",
]
