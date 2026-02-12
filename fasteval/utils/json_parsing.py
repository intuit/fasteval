"""JSON parsing utilities for LLM responses."""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from text that may contain other content.

    Handles multiple formats:
    1. Clean JSON string
    2. JSON in markdown code blocks (```json ... ```)
    3. JSON embedded in prose text
    4. Fallback: extract score value from text

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON as dict, or None if no valid JSON found

    Example:
        >>> extract_json_from_text('{"score": 0.8, "reasoning": "Good"}')
        {'score': 0.8, 'reasoning': 'Good'}

        >>> extract_json_from_text('Here is the result: ```json\\n{"score": 0.5}\\n```')
        {'score': 0.5}
    """
    if not text:
        return None

    # Try 1: Direct JSON parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try 2: Extract JSON from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 3: Find JSON object in text (look for {"score": ...})
    json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 4: Extract just the score value as fallback
    score_match = re.search(r"score[\"']?\s*[:=]\s*([0-9.]+)", text, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            return {"score": min(1.0, max(0.0, score)), "reasoning": text}
        except ValueError:
            pass

    return None


def parse_json_response(text: str, model_class: Type[T]) -> T:
    """
    Parse LLM response text and validate against a Pydantic model.

    Extracts JSON from the text and validates it against the provided
    Pydantic model class.

    Args:
        text: Text containing JSON response
        model_class: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If JSON cannot be extracted or validation fails

    Example:
        class EvalResponse(BaseModel):
            score: float
            reasoning: str

        response = parse_json_response('{"score": 0.8, "reasoning": "Good"}', EvalResponse)
        print(response.score)  # 0.8
    """
    data = extract_json_from_text(text)

    if data is None:
        raise ValueError(f"Could not extract JSON from response: {text[:200]}")

    try:
        return model_class(**data)
    except Exception as e:
        raise ValueError(f"JSON validation failed: {e}. Data: {data}")
