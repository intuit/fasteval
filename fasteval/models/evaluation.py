"""Evaluation data models using Pydantic."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool call made during agent execution."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    timestamp: Optional[str] = None


class ExpectedTool(BaseModel):
    """
    Expected tool call for trajectory evaluation.

    Used to define what tools an agent SHOULD have called,
    for comparison against actual tool_calls.

    Example:
        expected = [
            ExpectedTool(name="search_flights", args={"destination": "NYC"}),
            ExpectedTool(name="validate_user", required=False),  # Optional
            ExpectedTool(name="book_flight"),
        ]
    """

    name: str  # Tool name (supports wildcards like "search_*")
    args: Dict[str, Any] = Field(default_factory=dict)  # Expected arguments
    required: bool = True  # If False, tool is optional in trajectory


class EvalInput(BaseModel):
    """
    Standard input for all metrics.

    Metrics receive this object and can use any field
    for prompt template placeholders.

    Example:
        eval_input = EvalInput(
            actual_output="The answer is 4",
            expected_output="4",
            input="What is 2+2?",
            context=["Basic arithmetic"]
        )

    Multi-modal example:
        from fasteval.models.multimodal import ImageInput

        eval_input = EvalInput(
            actual_output="The chart shows 25% growth",
            input="What trend does this chart show?",
            image=ImageInput(source="chart.png")
        )
    """

    # Core fields
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None

    # Input that produced the output
    input: Optional[str] = None
    input_kwargs: Optional[Dict[str, Any]] = None

    # Context for RAG
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None

    # Multi-turn conversation
    history: Optional[List[Dict[str, str]]] = None

    # Agent/tool info
    tool_calls: List[ToolCall] = Field(default_factory=list)
    expected_tools: List[ExpectedTool] = Field(default_factory=list)

    # === Multi-modal fields ===
    # Single image input (convenience for common case)
    image: Optional[Any] = None  # ImageInput or str
    # Multiple images
    images: List[Any] = Field(default_factory=list)  # List[ImageInput | str]
    # Audio input
    audio: Optional[Any] = None  # AudioInput or str
    # Multiple audio files
    audios: List[Any] = Field(default_factory=list)  # List[AudioInput | str]
    # Generated image (for image generation evaluation)
    generated_image: Optional[Any] = None  # GeneratedImage, ImageInput, or str
    # Multi-modal context (text + images + audio)
    multimodal_context: Optional[List[Any]] = None  # List[MultiModalContext]
    # Expected fields for structured extraction (OCR, document extraction)
    expected_fields: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reference_id: Optional[str] = None  # trace_id, test_id, etc.

    model_config = {"extra": "allow"}

    def has_images(self) -> bool:
        """Check if this input contains any images."""
        return self.image is not None or len(self.images) > 0

    def has_audio(self) -> bool:
        """Check if this input contains any audio."""
        return self.audio is not None or len(self.audios) > 0

    def is_multimodal(self) -> bool:
        """Check if this input contains multi-modal content."""
        return (
            self.has_images()
            or self.has_audio()
            or self.generated_image is not None
            or self.multimodal_context is not None
        )

    def get_all_images(self) -> List[Any]:
        """Get all images as a list."""
        result: List[Any] = []
        if self.image is not None:
            result.append(self.image)
        result.extend(self.images)
        return result

    def get_all_audio(self) -> List[Any]:
        """Get all audio files as a list."""
        result: List[Any] = []
        if self.audio is not None:
            result.append(self.audio)
        result.extend(self.audios)
        return result


class MetricResult(BaseModel):
    """Result from a single metric evaluation."""

    metric_name: str
    score: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    passed: bool
    threshold: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HumanReviewResult(BaseModel):
    """Result from human-in-the-loop review."""

    score: float = Field(ge=0.0, le=1.0)  # Normalized 0-1
    raw_input: str  # Original input (e.g., "4", "pass", "p")
    skipped: bool = False  # True if reviewer skipped
    prompt: Optional[str] = None  # The question asked
    reviewer_id: Optional[str] = None  # Optional reviewer identifier
    timestamp: Optional[str] = None  # When review was completed


class EvalResult(BaseModel):
    """Complete evaluation result for a single test case."""

    eval_input: EvalInput
    metric_results: List[MetricResult] = Field(default_factory=list)
    passed: bool  # All metrics passed
    aggregate_score: float = Field(ge=0.0, le=1.0)  # Weighted average
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    reference_id: Optional[str] = None

    # Human review fields
    human_review: Optional[HumanReviewResult] = None
    human_review_required: bool = False  # If True and skipped, test fails


class EvaluationFailedError(AssertionError):
    """
    Raised when evaluation metrics fail their thresholds.

    This exception is raised immediately when fe.score() is called
    and one or more metrics fail to meet their threshold.

    Attributes:
        result: The EvalResult containing all metric results

    Example:
        try:
            fe.score(response, expected)
        except EvaluationFailedError as e:
            print(f"Failed with score: {e.result.aggregate_score}")
            for mr in e.result.metric_results:
                if not mr.passed:
                    print(f"  {mr.metric_name}: {mr.score} < {mr.threshold}")
    """

    def __init__(self, message: str, result: "EvalResult") -> None:
        super().__init__(message)
        self.result = result
