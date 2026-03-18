"""Tests for fasteval.metrics.deterministic."""

import re

import pytest
from pydantic import BaseModel

from fasteval.metrics.deterministic import (
    ContainsMetric,
    ExactMatchMetric,
    JsonMetric,
    RegexMetric,
    RougeMetric,
    ToolArgsMatchMetric,
    ToolCallAccuracyMetric,
    ToolSequenceMetric,
    _match_tool_name,
)
from fasteval.models.evaluation import EvalInput, ExpectedTool, ToolCall


class UserModel(BaseModel):
    name: str
    age: int


# ── RougeMetric ──────────────────────────────────────────────────────────────


class TestRougeMetric:
    @pytest.mark.asyncio
    async def test_high_similarity(self):
        metric = RougeMetric(threshold=0.5)
        result = await metric.evaluate(
            EvalInput(
                actual_output="the cat sat on the mat",
                expected_output="the cat sat on the mat",
            )
        )
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_low_similarity(self):
        metric = RougeMetric(threshold=0.9)
        result = await metric.evaluate(
            EvalInput(
                actual_output="completely different text",
                expected_output="the cat sat on the mat",
            )
        )
        assert result.score < 0.9
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_missing_actual(self):
        metric = RougeMetric()
        result = await metric.evaluate(
            EvalInput(actual_output=None, expected_output="expected")
        )
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_missing_expected(self):
        metric = RougeMetric()
        result = await metric.evaluate(
            EvalInput(actual_output="actual", expected_output=None)
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_details_include_precision_recall(self):
        metric = RougeMetric(threshold=0.3)
        result = await metric.evaluate(
            EvalInput(
                actual_output="the cat sat on the mat",
                expected_output="the cat sat on the mat",
            )
        )
        assert "precision" in result.details
        assert "recall" in result.details
        assert "fmeasure" in result.details

    def test_custom_name(self):
        metric = RougeMetric(name="my_rouge", rouge_type="rouge1")
        assert metric.name == "my_rouge"
        assert metric.rouge_type == "rouge1"


# ── ExactMatchMetric ─────────────────────────────────────────────────────────


class TestExactMatchMetric:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        metric = ExactMatchMetric()
        result = await metric.evaluate(
            EvalInput(actual_output="Hello World", expected_output="hello world")
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_case_sensitive(self):
        metric = ExactMatchMetric(case_sensitive=True)
        result = await metric.evaluate(
            EvalInput(actual_output="Hello", expected_output="hello")
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_normalize_whitespace(self):
        metric = ExactMatchMetric()
        result = await metric.evaluate(
            EvalInput(actual_output="hello   world", expected_output="hello world")
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        metric = ExactMatchMetric()
        result = await metric.evaluate(
            EvalInput(actual_output="yes", expected_output="no")
        )
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_missing_output(self):
        metric = ExactMatchMetric()
        result = await metric.evaluate(
            EvalInput(actual_output=None, expected_output="expected")
        )
        assert result.score == 0.0


# ── ContainsMetric ───────────────────────────────────────────────────────────


class TestContainsMetric:
    @pytest.mark.asyncio
    async def test_contains(self):
        metric = ContainsMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="The answer is 42 indeed",
                expected_output="42",
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_does_not_contain(self):
        metric = ContainsMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="The answer is unknown",
                expected_output="42",
            )
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        metric = ContainsMetric(case_sensitive=False)
        result = await metric.evaluate(
            EvalInput(actual_output="HELLO WORLD", expected_output="hello")
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_case_sensitive(self):
        metric = ContainsMetric(case_sensitive=True)
        result = await metric.evaluate(
            EvalInput(actual_output="HELLO WORLD", expected_output="hello")
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_missing_output(self):
        metric = ContainsMetric()
        result = await metric.evaluate(
            EvalInput(actual_output=None, expected_output="test")
        )
        assert result.score == 0.0


# ── JsonMetric ───────────────────────────────────────────────────────────────


class TestJsonMetric:
    @pytest.mark.asyncio
    async def test_valid_json(self):
        metric = JsonMetric(model=UserModel)
        result = await metric.evaluate(
            EvalInput(actual_output='{"name": "Alice", "age": 30}')
        )
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_json_syntax(self):
        metric = JsonMetric(model=UserModel)
        result = await metric.evaluate(EvalInput(actual_output="{not valid json}"))
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_schema_validation_failure(self):
        metric = JsonMetric(model=UserModel)
        result = await metric.evaluate(EvalInput(actual_output='{"name": "Alice"}'))
        assert result.score == 0.0
        assert "validation" in result.details.get("error_type", "")

    @pytest.mark.asyncio
    async def test_missing_output(self):
        metric = JsonMetric(model=UserModel)
        result = await metric.evaluate(EvalInput(actual_output=None))
        assert result.score == 0.0


# ── RegexMetric ──────────────────────────────────────────────────────────────


class TestRegexMetric:
    @pytest.mark.asyncio
    async def test_full_match(self):
        metric = RegexMetric(pattern=r"\d{3}-\d{4}")
        result = await metric.evaluate(EvalInput(actual_output="123-4567"))
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_full_match_fails(self):
        metric = RegexMetric(pattern=r"\d{3}-\d{4}")
        result = await metric.evaluate(EvalInput(actual_output="phone: 123-4567"))
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_search_match(self):
        metric = RegexMetric(pattern=r"\d{3}-\d{4}", full_match=False)
        result = await metric.evaluate(EvalInput(actual_output="phone: 123-4567"))
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        metric = RegexMetric(pattern=r"\d+", full_match=False)
        result = await metric.evaluate(EvalInput(actual_output="no digits here"))
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_flags_ignorecase(self):
        metric = RegexMetric(pattern=r"^yes$", flags=re.IGNORECASE)
        result = await metric.evaluate(EvalInput(actual_output="YES"))
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_missing_output(self):
        metric = RegexMetric(pattern=r"\d+")
        result = await metric.evaluate(EvalInput(actual_output=None))
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_match_details(self):
        metric = RegexMetric(pattern=r"\d+", full_match=False)
        result = await metric.evaluate(EvalInput(actual_output="abc123def"))
        assert result.details["match"] == "123"
        assert result.details["match_start"] == 3


# ── _match_tool_name ─────────────────────────────────────────────────────────


class TestMatchToolName:
    def test_exact_match(self):
        assert _match_tool_name("search_flights", "search_flights") is True

    def test_wildcard_prefix(self):
        assert _match_tool_name("search_flights", "search_*") is True

    def test_wildcard_suffix(self):
        assert _match_tool_name("search_flights", "*_flights") is True

    def test_no_match(self):
        assert _match_tool_name("search_flights", "book_*") is False


# ── ToolCallAccuracyMetric ───────────────────────────────────────────────────


class TestToolCallAccuracyMetric:
    @pytest.mark.asyncio
    async def test_all_tools_match(self):
        metric = ToolCallAccuracyMetric(threshold=0.8)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search"), ToolCall(name="book")],
                expected_tools=[
                    ExpectedTool(name="search"),
                    ExpectedTool(name="book"),
                ],
            )
        )
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_missing_required_tool(self):
        metric = ToolCallAccuracyMetric(threshold=0.8)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search")],
                expected_tools=[
                    ExpectedTool(name="search"),
                    ExpectedTool(name="book"),
                ],
            )
        )
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_extra_tools_penalized(self):
        metric = ToolCallAccuracyMetric(threshold=0.8, ignore_extra=False)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[
                    ToolCall(name="search"),
                    ToolCall(name="book"),
                    ToolCall(name="cancel"),
                ],
                expected_tools=[
                    ExpectedTool(name="search"),
                    ExpectedTool(name="book"),
                ],
            )
        )
        # 2 matched / max(2 required, 3 actual) = 2/3
        assert abs(result.score - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_ignore_extra(self):
        metric = ToolCallAccuracyMetric(threshold=0.8, ignore_extra=True)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[
                    ToolCall(name="search"),
                    ToolCall(name="book"),
                    ToolCall(name="cancel"),
                ],
                expected_tools=[
                    ExpectedTool(name="search"),
                    ExpectedTool(name="book"),
                ],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_wildcard_matching(self):
        metric = ToolCallAccuracyMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search_flights")],
                expected_tools=[ExpectedTool(name="search_*")],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_expected_tools_no_actual(self):
        metric = ToolCallAccuracyMetric()
        result = await metric.evaluate(
            EvalInput(actual_output="result", tool_calls=[], expected_tools=[])
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_expected_with_actual(self):
        metric = ToolCallAccuracyMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search")],
                expected_tools=[],
            )
        )
        assert result.score == 0.0


# ── ToolSequenceMetric ───────────────────────────────────────────────────────


class TestToolSequenceMetric:
    @pytest.mark.asyncio
    async def test_strict_exact_match(self):
        metric = ToolSequenceMetric(strict=True)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="a"), ToolCall(name="b"), ToolCall(name="c")],
                expected_tools=[
                    ExpectedTool(name="a"),
                    ExpectedTool(name="b"),
                    ExpectedTool(name="c"),
                ],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_strict_mismatch(self):
        metric = ToolSequenceMetric(strict=True)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="b"), ToolCall(name="a")],
                expected_tools=[
                    ExpectedTool(name="a"),
                    ExpectedTool(name="b"),
                ],
            )
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_lcs_scoring(self):
        metric = ToolSequenceMetric(strict=False)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[
                    ToolCall(name="a"),
                    ToolCall(name="c"),
                    ToolCall(name="b"),
                ],
                expected_tools=[
                    ExpectedTool(name="a"),
                    ExpectedTool(name="b"),
                    ExpectedTool(name="c"),
                ],
            )
        )
        # LCS of [a, c, b] vs [a, b, c] -> LCS len 2 / max(3, 3) = 2/3
        assert result.score > 0.0
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_no_expected(self):
        metric = ToolSequenceMetric()
        result = await metric.evaluate(
            EvalInput(actual_output="result", tool_calls=[], expected_tools=[])
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_required_tools(self):
        metric = ToolSequenceMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="a")],
                expected_tools=[
                    ExpectedTool(name="a", required=False),
                ],
            )
        )
        assert result.score == 1.0


# ── ToolArgsMatchMetric ──────────────────────────────────────────────────────


class TestToolArgsMatchMetric:
    @pytest.mark.asyncio
    async def test_all_args_match(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[
                    ToolCall(
                        name="search", arguments={"dest": "NYC", "date": "2024-01-01"}
                    )
                ],
                expected_tools=[
                    ExpectedTool(
                        name="search", args={"dest": "NYC", "date": "2024-01-01"}
                    )
                ],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_value_mismatch(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search", arguments={"dest": "LAX"})],
                expected_tools=[ExpectedTool(name="search", args={"dest": "NYC"})],
            )
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_missing_arg(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search", arguments={})],
                expected_tools=[ExpectedTool(name="search", args={"dest": "NYC"})],
            )
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_tool_not_called(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[],
                expected_tools=[ExpectedTool(name="search", args={"dest": "NYC"})],
            )
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_expected_args(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search", arguments={"dest": "NYC"})],
                expected_tools=[ExpectedTool(name="search", args={})],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_expected_tools(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(actual_output="result", tool_calls=[], expected_tools=[])
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_numeric_comparison(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="calc", arguments={"value": 3.14})],
                expected_tools=[ExpectedTool(name="calc", args={"value": 3.14})],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_string_case_insensitive(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search", arguments={"dest": "nyc"})],
                expected_tools=[ExpectedTool(name="search", args={"dest": "NYC"})],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_none_expected_value(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="search", arguments={"dest": "anything"})],
                expected_tools=[ExpectedTool(name="search", args={"dest": None})],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_numeric_tolerance(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[ToolCall(name="calc", arguments={"value": 3.14000001})],
                expected_tools=[ExpectedTool(name="calc", args={"value": 3.14})],
            )
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_multiple_tools_partial_match(self):
        metric = ToolArgsMatchMetric()
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[
                    ToolCall(name="a", arguments={"x": 1}),
                    ToolCall(name="b", arguments={"y": "wrong"}),
                ],
                expected_tools=[
                    ExpectedTool(name="a", args={"x": 1}),
                    ExpectedTool(name="b", args={"y": "correct"}),
                ],
            )
        )
        assert result.score == 0.5


# ── RougeMetric additional ───────────────────────────────────────────────────


class TestRougeMetricAdditional:
    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        metric = RougeMetric(threshold=0.3)
        result = await metric.evaluate(
            EvalInput(
                actual_output="the quick brown fox",
                expected_output="the slow brown dog",
            )
        )
        assert 0.0 < result.score < 1.0
        assert result.passed is True

    def test_default_name(self):
        metric = RougeMetric()
        assert metric.name == "rouge"


# ── ExactMatch additional ────────────────────────────────────────────────────


class TestExactMatchAdditional:
    @pytest.mark.asyncio
    async def test_no_normalize(self):
        metric = ExactMatchMetric(normalize=False, case_sensitive=False)
        result = await metric.evaluate(
            EvalInput(actual_output="hello   world", expected_output="hello world")
        )
        assert result.score == 0.0  # Extra spaces not normalized

    def test_default_name(self):
        metric = ExactMatchMetric()
        assert metric.name == "exact_match"


# ── RegexMetric additional ───────────────────────────────────────────────────


class TestRegexMetricAdditional:
    @pytest.mark.asyncio
    async def test_multiline(self):
        metric = RegexMetric(pattern=r"hello", flags=re.MULTILINE, full_match=False)
        result = await metric.evaluate(EvalInput(actual_output="foo\nhello\nbar"))
        assert result.score == 1.0

    def test_default_name(self):
        metric = RegexMetric(pattern=r"\d+")
        assert metric.name == "regex"


# ── ToolSequence additional ──────────────────────────────────────────────────


class TestToolSequenceAdditional:
    @pytest.mark.asyncio
    async def test_empty_actual_tools(self):
        metric = ToolSequenceMetric(strict=True)
        result = await metric.evaluate(
            EvalInput(
                actual_output="result",
                tool_calls=[],
                expected_tools=[ExpectedTool(name="a")],
            )
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_lcs_empty_sequences(self):
        metric = ToolSequenceMetric(strict=False)
        # Test internal LCS with empty sequences
        assert metric._longest_common_subsequence([], ["a"]) == 0
        assert metric._longest_common_subsequence(["a"], []) == 0
