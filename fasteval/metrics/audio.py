"""Audio and speech evaluation metrics.

This module provides metrics for evaluating speech-to-text (ASR),
voice assistants, and audio understanding systems.

Requires: pip install fasteval[audio]
"""

import logging
from typing import Any, Optional

from fasteval.metrics.base import Metric
from fasteval.metrics.llm import BaseLLMMetric
from fasteval.models.evaluation import EvalInput, MetricResult
from fasteval.providers.base import LLMClient

logger = logging.getLogger(__name__)

# Check for audio dependencies
try:
    from fasteval.utils.audio import (
        JIWER_AVAILABLE,
        calculate_cer,
        calculate_mer,
        calculate_wer,
        convert_cer_to_accuracy,
        convert_wer_to_accuracy,
    )

    AUDIO_AVAILABLE = JIWER_AVAILABLE
except ImportError:
    AUDIO_AVAILABLE = False


def _check_audio_available() -> None:
    """Check if audio dependencies are available."""
    if not AUDIO_AVAILABLE:
        raise ImportError(
            "Audio metrics require the 'audio' extra. "
            "Install with: pip install fasteval[audio]"
        )


class WordErrorRateMetric(Metric):
    """
    Calculate Word Error Rate (WER) for speech recognition evaluation.

    WER is the standard industry metric for ASR systems. It measures the
    minimum number of word-level edits (insertions, deletions, substitutions)
    needed to transform the hypothesis into the reference.

    Note: This metric uses inverted scoring where:
    - score = 1.0 - WER (so higher scores are better)
    - A WER of 0.05 becomes score of 0.95

    Example:
        @fe.word_error_rate(threshold=0.95)  # WER < 5%
        def test_asr():
            transcript = whisper.transcribe("audio.mp3")
            fe.score(actual_output=transcript, expected_output=ground_truth)
    """

    def __init__(
        self,
        name: str = "word_error_rate",
        threshold: float = 0.9,
        weight: float = 1.0,
        normalize_text: bool = True,
    ) -> None:
        """
        Initialize WER metric.

        Args:
            name: Metric name
            threshold: Minimum accuracy score (1.0 - WER) to pass
            weight: Weight for aggregation
            normalize_text: If True, normalize text before comparison
        """
        super().__init__(name=name, threshold=threshold, weight=weight)
        self.normalize_text = normalize_text

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Calculate WER and return result."""
        _check_audio_available()

        reference = eval_input.expected_output or ""
        hypothesis = eval_input.actual_output or ""

        if not reference:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="No reference transcript provided",
            )

        if not hypothesis:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="No hypothesis transcript provided",
            )

        wer = calculate_wer(reference, hypothesis, normalize=self.normalize_text)
        score = convert_wer_to_accuracy(wer)

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning=f"WER: {wer:.2%} (Accuracy: {score:.2%})",
            details={
                "wer": wer,
                "accuracy": score,
                "reference_words": len(reference.split()),
                "hypothesis_words": len(hypothesis.split()),
            },
        )


class CharacterErrorRateMetric(Metric):
    """
    Calculate Character Error Rate (CER) for speech recognition evaluation.

    CER is useful for character-level ASR evaluation, especially for:
    - Languages without clear word boundaries (Chinese, Japanese)
    - Name/entity recognition
    - Detailed accuracy analysis

    Note: This metric uses inverted scoring where:
    - score = 1.0 - CER (so higher scores are better)

    Example:
        @fe.character_error_rate(threshold=0.95)  # CER < 5%
        def test_asr_detailed():
            transcript = model.transcribe(audio)
            fe.score(actual_output=transcript, expected_output=reference)
    """

    def __init__(
        self,
        name: str = "character_error_rate",
        threshold: float = 0.9,
        weight: float = 1.0,
        normalize_text: bool = True,
    ) -> None:
        """
        Initialize CER metric.

        Args:
            name: Metric name
            threshold: Minimum accuracy score (1.0 - CER) to pass
            weight: Weight for aggregation
            normalize_text: If True, normalize text before comparison
        """
        super().__init__(name=name, threshold=threshold, weight=weight)
        self.normalize_text = normalize_text

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Calculate CER and return result."""
        _check_audio_available()

        reference = eval_input.expected_output or ""
        hypothesis = eval_input.actual_output or ""

        if not reference:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="No reference transcript provided",
            )

        if not hypothesis:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="No hypothesis transcript provided",
            )

        cer = calculate_cer(reference, hypothesis, normalize=self.normalize_text)
        score = convert_cer_to_accuracy(cer)

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning=f"CER: {cer:.2%} (Accuracy: {score:.2%})",
            details={
                "cer": cer,
                "accuracy": score,
                "reference_chars": len(reference),
                "hypothesis_chars": len(hypothesis),
            },
        )


class MatchErrorRateMetric(Metric):
    """
    Calculate Match Error Rate (MER) for speech recognition evaluation.

    MER is a variant of WER that accounts for the total number of
    words in both reference and hypothesis. It's bounded between 0 and 1.

    Example:
        @fe.match_error_rate(threshold=0.9)
        def test_asr():
            transcript = model.transcribe(audio)
            fe.score(actual_output=transcript, expected_output=reference)
    """

    def __init__(
        self,
        name: str = "match_error_rate",
        threshold: float = 0.9,
        weight: float = 1.0,
        normalize_text: bool = True,
    ) -> None:
        super().__init__(name=name, threshold=threshold, weight=weight)
        self.normalize_text = normalize_text

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Calculate MER and return result."""
        _check_audio_available()

        reference = eval_input.expected_output or ""
        hypothesis = eval_input.actual_output or ""

        if not reference or not hypothesis:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="Missing reference or hypothesis transcript",
            )

        mer = calculate_mer(reference, hypothesis, normalize=self.normalize_text)
        score = max(0.0, 1.0 - mer)

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning=f"MER: {mer:.2%} (Accuracy: {score:.2%})",
            details={"mer": mer, "accuracy": score},
        )


class TranscriptionAccuracyMetric(BaseLLMMetric):
    """
    Evaluate semantic correctness of transcription using LLM.

    Unlike WER/CER which are purely character-based, this metric
    evaluates if the transcription captures the semantic meaning
    correctly, even if exact words differ slightly.

    Example:
        @fe.transcription_accuracy(threshold=0.9)
        def test_semantic_accuracy():
            transcript = whisper.transcribe("meeting.mp3")
            fe.score(
                actual_output=transcript,
                expected_output=ground_truth_transcript
            )
    """

    def __init__(
        self,
        name: str = "transcription_accuracy",
        threshold: float = 0.9,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        return f"""Evaluate the semantic accuracy of a speech transcription.

Reference Transcript (Ground Truth):
{eval_input.expected_output}

Hypothesis Transcript (ASR Output):
{eval_input.actual_output}

Consider:
1. Are the key words and phrases captured correctly?
2. Is the meaning preserved even if exact wording differs?
3. Are names, numbers, and technical terms transcribed accurately?
4. Are any important details missing or misheard?

Provide your evaluation as JSON:
{{"score": <0.0 to 1.0>, "reasoning": "<your explanation>"}}

Score 0.0 = Completely incorrect transcription, meaning is lost
Score 0.5 = Partially correct, some meaning captured but errors
Score 1.0 = Semantically accurate, captures all intended meaning"""


class SpeakerDiarizationMetric(BaseLLMMetric):
    """
    Evaluate speaker diarization (who spoke when) accuracy.

    Checks if speakers are correctly identified and attributed
    in multi-speaker transcriptions.

    Example:
        @fe.speaker_diarization(threshold=0.8)
        def test_speaker_id():
            result = model.transcribe_with_speakers("meeting.mp3")
            fe.score(
                actual_output=result,
                expected_output=annotated_transcript
            )
    """

    def __init__(
        self,
        name: str = "speaker_diarization",
        threshold: float = 0.8,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        return f"""Evaluate speaker diarization accuracy.

Expected (with speaker labels):
{eval_input.expected_output}

Actual (with speaker labels):
{eval_input.actual_output}

Consider:
1. Are speakers correctly identified and labeled?
2. Are speaker turn boundaries accurate?
3. Is the same speaker consistently labeled throughout?
4. Are overlapping speech segments handled correctly?

Provide your evaluation as JSON:
{{"score": <0.0 to 1.0>, "reasoning": "<your explanation>"}}

Score 0.0 = Speakers completely misidentified
Score 0.5 = Some speakers correct but inconsistent labeling
Score 1.0 = All speakers correctly identified with accurate turn boundaries"""


class AudioSentimentMetric(BaseLLMMetric):
    """
    Evaluate tone/emotion detection accuracy from audio.

    Checks if detected sentiment/emotion matches the expected
    tone in speech recordings.

    Example:
        @fe.audio_sentiment(threshold=0.8)
        def test_tone_detection():
            sentiment = model.detect_sentiment("customer_call.mp3")
            fe.score(
                actual_output=sentiment,
                expected_output="frustrated, but cooperative"
            )
    """

    def __init__(
        self,
        name: str = "audio_sentiment",
        threshold: float = 0.8,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        return f"""Evaluate audio sentiment/emotion detection accuracy.

Expected Sentiment/Tone: {eval_input.expected_output}
Detected Sentiment/Tone: {eval_input.actual_output}

Consider:
1. Does the detected sentiment match the expected emotion?
2. Is the intensity/strength of emotion captured correctly?
3. Are multiple emotions identified if present?
4. Is the overall tone (positive/negative/neutral) correct?

Provide your evaluation as JSON:
{{"score": <0.0 to 1.0>, "reasoning": "<your explanation>"}}

Score 0.0 = Completely wrong sentiment detected
Score 0.5 = Partially correct, some emotions missed
Score 1.0 = Perfectly accurate sentiment detection"""
