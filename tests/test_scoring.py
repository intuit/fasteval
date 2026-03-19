"""Tests for fasteval.core.scoring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fasteval.core.scoring import (
    _get_test_name_from_caller,
    _last_score_result,
    _normalize_audio,
    _normalize_generated_image,
    _normalize_image,
    clear_last_score_result,
    get_last_score_result,
    score,
)
from fasteval.models.evaluation import (
    EvalInput,
    EvalResult,
    EvaluationFailedError,
    ExpectedTool,
    MetricResult,
    ToolCall,
)
from fasteval.models.multimodal import AudioInput, GeneratedImage, ImageInput

# ── Normalization helpers ────────────────────────────────────────────────────


class TestNormalizeImage:
    def test_image_input_passthrough(self):
        img = ImageInput(source="test.png")
        assert _normalize_image(img) is img

    def test_path_to_str(self):
        result = _normalize_image(Path("/tmp/test.png"))
        assert result == "/tmp/test.png"
        assert isinstance(result, str)

    def test_string_passthrough(self):
        assert _normalize_image("http://img.png") == "http://img.png"


class TestNormalizeAudio:
    def test_audio_input_passthrough(self):
        aud = AudioInput(source="test.wav")
        assert _normalize_audio(aud) is aud

    def test_path_to_str(self):
        result = _normalize_audio(Path("/tmp/test.wav"))
        assert result == "/tmp/test.wav"

    def test_string_passthrough(self):
        assert _normalize_audio("http://audio.wav") == "http://audio.wav"


class TestNormalizeGeneratedImage:
    def test_generated_image_passthrough(self):
        img = GeneratedImage(image=ImageInput(source="test.png"), prompt="A test image")
        assert _normalize_generated_image(img) is img

    def test_image_input_passthrough(self):
        img = ImageInput(source="test.png")
        assert _normalize_generated_image(img) is img

    def test_path_to_str(self):
        result = _normalize_generated_image(Path("/tmp/gen.png"))
        assert result == "/tmp/gen.png"

    def test_string_passthrough(self):
        assert _normalize_generated_image("gen.png") == "gen.png"


# ── Context variable storage ─────────────────────────────────────────────────


class TestLastScoreResult:
    def test_default_is_none(self):
        clear_last_score_result()
        assert get_last_score_result() is None

    def test_set_and_get(self):
        result = EvalResult(
            eval_input=EvalInput(actual_output="test"),
            metric_results=[],
            passed=True,
            aggregate_score=1.0,
        )
        _last_score_result.set(result)
        assert get_last_score_result() is result
        clear_last_score_result()

    def test_clear(self):
        _last_score_result.set(
            EvalResult(
                eval_input=EvalInput(actual_output="test"),
                metric_results=[],
                passed=True,
                aggregate_score=1.0,
            )
        )
        clear_last_score_result()
        assert get_last_score_result() is None


# ── _get_test_name_from_caller ───────────────────────────────────────────────


class TestGetTestName:
    def test_from_test_function(self):
        # This is called from a function starting with "test_"
        name = _get_test_name_from_caller()
        assert name.startswith("test_")

    def _helper_non_test(self):
        return _get_test_name_from_caller()

    def test_from_non_test_via_helper(self):
        # The helper doesn't start with test_, but the caller does
        name = self._helper_non_test()
        assert name.startswith("test_")


# ── score() function ─────────────────────────────────────────────────────────


class TestScoreFunction:
    def test_score_no_decorators(self):
        """score() without decorators returns a base result."""
        result = score("actual output", "expected output", input="question")
        assert result.passed is True
        assert result.aggregate_score == 1.0
        assert len(result.metric_results) == 0

    def test_score_stores_last_result(self):
        clear_last_score_result()
        result = score("output")
        assert get_last_score_result() is result

    def test_score_normalizes_tool_calls_dict(self):
        result = score(
            "output",
            tool_calls=[
                {"name": "search", "args": {"q": "test"}, "result": "found"},
            ],
        )
        assert result.eval_input.tool_calls[0].name == "search"
        assert result.eval_input.tool_calls[0].arguments == {"q": "test"}

    def test_score_normalizes_tool_calls_with_arguments_key(self):
        result = score(
            "output",
            tool_calls=[
                {"name": "search", "arguments": {"q": "test"}},
            ],
        )
        assert result.eval_input.tool_calls[0].arguments == {"q": "test"}

    def test_score_normalizes_tool_call_model(self):
        tc = ToolCall(name="search", arguments={"q": "test"})
        result = score("output", tool_calls=[tc])
        assert result.eval_input.tool_calls[0] is tc

    def test_score_normalizes_expected_tools_dict(self):
        result = score(
            "output",
            expected_tools=[
                {"name": "search", "args": {"q": "test"}, "required": False},
            ],
        )
        assert result.eval_input.expected_tools[0].name == "search"
        assert result.eval_input.expected_tools[0].required is False

    def test_score_normalizes_expected_tool_model(self):
        et = ExpectedTool(name="search")
        result = score("output", expected_tools=[et])
        assert result.eval_input.expected_tools[0] is et

    def test_score_with_context(self):
        result = score(
            "output",
            context=["doc1", "doc2"],
            retrieval_context=["ret1"],
        )
        assert result.eval_input.context == ["doc1", "doc2"]
        assert result.eval_input.retrieval_context == ["ret1"]

    def test_score_with_metadata(self):
        result = score("output", metadata={"key": "value"})
        assert result.eval_input.metadata == {"key": "value"}
