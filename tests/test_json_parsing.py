"""Tests for fasteval.utils.json_parsing."""

import pytest
from pydantic import BaseModel

from fasteval.utils.json_parsing import extract_json_from_text, parse_json_response


class SampleModel(BaseModel):
    score: float
    reasoning: str = ""


class TestExtractJsonFromText:
    def test_direct_json(self):
        result = extract_json_from_text('{"score": 0.8, "reasoning": "Good"}')
        assert result == {"score": 0.8, "reasoning": "Good"}

    def test_markdown_code_block(self):
        text = 'Here is the result:\n```json\n{"score": 0.5, "reasoning": "OK"}\n```'
        result = extract_json_from_text(text)
        assert result == {"score": 0.5, "reasoning": "OK"}

    def test_markdown_code_block_no_json_tag(self):
        text = '```\n{"score": 0.9}\n```'
        result = extract_json_from_text(text)
        assert result == {"score": 0.9}

    def test_embedded_json_with_score(self):
        text = 'The evaluation shows {"score": 0.7, "reasoning": "decent"} overall.'
        result = extract_json_from_text(text)
        assert result == {"score": 0.7, "reasoning": "decent"}

    def test_score_only_fallback(self):
        text = "The score is score: 0.85 based on analysis"
        result = extract_json_from_text(text)
        assert result is not None
        assert result["score"] == 0.85

    def test_score_clamping_above_one(self):
        text = "score: 1.5"
        result = extract_json_from_text(text)
        assert result is not None
        assert result["score"] == 1.0

    def test_no_match_for_negative_score(self):
        # Regex only matches positive numbers, so negative scores don't match
        text = "score: -0.5"
        result = extract_json_from_text(text)
        # Falls through to "score" fallback but regex captures "0.5" from "-0.5"
        if result is not None:
            assert 0.0 <= result["score"] <= 1.0

    def test_empty_string(self):
        assert extract_json_from_text("") is None

    def test_no_json_found(self):
        assert extract_json_from_text("no json here at all") is None

    def test_invalid_json_in_code_block(self):
        text = "```json\n{invalid json}\n```"
        # Should fall through to other strategies
        result = extract_json_from_text(text)
        assert result is None

    def test_score_with_equals(self):
        text = "score=0.6"
        result = extract_json_from_text(text)
        assert result is not None
        assert result["score"] == 0.6


class TestParseJsonResponse:
    def test_valid_model(self):
        result = parse_json_response('{"score": 0.8, "reasoning": "Good"}', SampleModel)
        assert isinstance(result, SampleModel)
        assert result.score == 0.8
        assert result.reasoning == "Good"

    def test_extraction_failure(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            parse_json_response("no json here", SampleModel)

    def test_validation_failure(self):
        with pytest.raises(ValueError, match="JSON validation failed"):
            parse_json_response('{"score": "not_a_number"}', SampleModel)

    def test_from_markdown_code_block(self):
        text = '```json\n{"score": 0.9, "reasoning": "Excellent"}\n```'
        result = parse_json_response(text, SampleModel)
        assert result.score == 0.9


class TestExtractJsonEdgeCases:
    def test_embedded_json_with_invalid_inner(self):
        # JSON object found but has invalid content when parsed
        text = 'Result: {"score": "bad"} end'
        result = extract_json_from_text(text)
        # It should still extract the JSON dict
        assert result is not None
        assert result["score"] == "bad"

    def test_score_value_extraction_with_float(self):
        text = "Based on analysis, score: 0.75 out of 1.0"
        result = extract_json_from_text(text)
        assert result is not None
        assert result["score"] == 0.75

    def test_score_extraction_with_single_quotes(self):
        text = "score': 0.6"
        result = extract_json_from_text(text)
        assert result is not None
        assert result["score"] == 0.6
