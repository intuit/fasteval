"""Deterministic (non-LLM) evaluation metrics."""

import fnmatch
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, ValidationError

from fasteval.metrics.base import Metric
from fasteval.models.evaluation import EvalInput, ExpectedTool, MetricResult, ToolCall

logger = logging.getLogger(__name__)


class RougeMetric(Metric):
    """
    ROUGE similarity metric.

    Measures overlap between actual and expected output.

    Example:
        metric = RougeMetric(rouge_type="rougeL", threshold=0.5)
    """

    def __init__(
        self,
        rouge_type: str = "rougeL",
        use_stemmer: bool = True,
        threshold: float = 0.5,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name or "rouge", threshold=threshold, weight=weight)
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        self._scorer: Optional[Any] = None

    def _get_scorer(self) -> Any:
        """Lazy initialization of ROUGE scorer."""
        if self._scorer is None:
            try:
                from rouge_score import rouge_scorer  # type: ignore[import-untyped]

                self._scorer = rouge_scorer.RougeScorer(
                    [self.rouge_type], use_stemmer=self.use_stemmer
                )
            except ImportError:
                raise ImportError(
                    "rouge-score package is required for RougeMetric. "
                    "Install it with: pip install rouge-score"
                )
        return self._scorer

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Evaluate ROUGE score between actual and expected output."""
        if not eval_input.actual_output or not eval_input.expected_output:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="Missing actual_output or expected_output",
            )

        try:
            scorer = self._get_scorer()
            scores = scorer.score(eval_input.expected_output, eval_input.actual_output)
            score = scores[self.rouge_type].fmeasure

            return MetricResult(
                metric_name=self.name,
                score=score,
                passed=self._determine_pass(score),
                threshold=self.threshold,
                reasoning=f"{self.rouge_type} F1 score: {score:.3f}",
                details={
                    "rouge_type": self.rouge_type,
                    "precision": scores[self.rouge_type].precision,
                    "recall": scores[self.rouge_type].recall,
                    "fmeasure": scores[self.rouge_type].fmeasure,
                },
            )
        except Exception as e:
            logger.error(f"Error in ROUGE metric: {e}")
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning=f"Error: {e}",
            )


class ExactMatchMetric(Metric):
    """
    Exact string match metric.

    Returns 1.0 if actual matches expected (optionally normalized),
    0.0 otherwise.

    Example:
        metric = ExactMatchMetric(normalize=True, case_sensitive=False)
    """

    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        threshold: float = 1.0,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name or "exact_match", threshold=threshold, weight=weight)
        self.normalize = normalize
        self.case_sensitive = case_sensitive

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not self.case_sensitive:
            text = text.lower()
        if self.normalize:
            text = " ".join(text.split())  # Normalize whitespace
        return text.strip()

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Check if actual exactly matches expected."""
        if not eval_input.actual_output or not eval_input.expected_output:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="Missing actual_output or expected_output",
            )

        actual = self._normalize_text(eval_input.actual_output)
        expected = self._normalize_text(eval_input.expected_output)

        score = 1.0 if actual == expected else 0.0

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning="Exact match" if score == 1.0 else "No match",
            details={
                "normalized": self.normalize,
                "case_sensitive": self.case_sensitive,
            },
        )


class ContainsMetric(Metric):
    """
    Check if expected is contained in actual output.

    Returns 1.0 if expected is found in actual, 0.0 otherwise.

    Example:
        metric = ContainsMetric(case_sensitive=False)
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        threshold: float = 1.0,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name or "contains", threshold=threshold, weight=weight)
        self.case_sensitive = case_sensitive

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Check if expected is contained in actual."""
        if not eval_input.actual_output or not eval_input.expected_output:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="Missing actual_output or expected_output",
            )

        actual = eval_input.actual_output
        expected = eval_input.expected_output

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        score = 1.0 if expected in actual else 0.0

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning=(
                "Contains expected" if score == 1.0 else "Does not contain expected"
            ),
            details={"case_sensitive": self.case_sensitive},
        )


class JsonMetric(Metric):
    """
    JSON schema validation using Pydantic models.

    Validates if actual output is valid JSON conforming to a Pydantic model.

    Example:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        metric = JsonMetric(model=User)
    """

    def __init__(
        self,
        model: Type[BaseModel],
        threshold: float = 1.0,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize JSON metric.

        Args:
            model: Pydantic model class to validate against
            threshold: Score threshold (typically 1.0 for pass/fail)
            name: Metric name (defaults to "json")
            weight: Weight for aggregation
        """
        super().__init__(name=name or "json", threshold=threshold, weight=weight)
        self.model = model

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Validate actual output against Pydantic model."""
        if not eval_input.actual_output:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="Missing actual_output",
            )

        try:
            # Use Pydantic's model_validate_json for parsing + validation
            self.model.model_validate_json(eval_input.actual_output)

            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                reasoning=f"Valid {self.model.__name__} JSON",
                details={"model": self.model.__name__},
            )

        except json.JSONDecodeError as e:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning=f"Invalid JSON: {e.msg}",
                details={"error_type": "json_decode", "error": str(e)},
            )

        except ValidationError as e:
            # Get first error for concise message
            error_list = e.errors()
            if error_list:
                first_error = error_list[0]
                loc = first_error.get("loc", ())
                field = ".".join(str(part) for part in loc)
                msg = str(first_error.get("msg", "Validation failed"))
            else:
                field = ""
                msg = "Validation failed"

            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning=f"Schema validation failed: {field} - {msg}",
                details={
                    "error_type": "validation",
                    "errors": [
                        {
                            "field": ".".join(
                                str(loc_part) for loc_part in err.get("loc", ())
                            ),
                            "message": str(err.get("msg", "")),
                        }
                        for err in error_list
                    ],
                },
            )


class RegexMetric(Metric):
    """
    Regex pattern matching for structured outputs.

    Returns 1.0 if the pattern matches the actual output, 0.0 otherwise.
    Supports full regex syntax and optional flags.

    Example:
        # Match phone number format
        metric = RegexMetric(pattern=r"^\d{3}-\d{4}$")

        # Case-insensitive matching
        metric = RegexMetric(pattern=r"^yes|no$", flags=re.IGNORECASE)

        # Match anywhere in output (not just full match)
        metric = RegexMetric(pattern=r"\d{4}", full_match=False)
    """

    def __init__(
        self,
        pattern: str,
        flags: Union[int, re.RegexFlag] = 0,
        full_match: bool = True,
        threshold: float = 1.0,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize regex metric.

        Args:
            pattern: Regular expression pattern to match
            flags: Regex flags (e.g., re.IGNORECASE, re.MULTILINE)
            full_match: If True, pattern must match entire output.
                        If False, pattern can match anywhere in output.
            threshold: Score threshold (typically 1.0 for pass/fail)
            name: Metric name (defaults to "regex")
            weight: Weight for aggregation
        """
        super().__init__(name=name or "regex", threshold=threshold, weight=weight)
        self.pattern = pattern
        self.flags = flags
        self.full_match = full_match
        self._compiled: Optional[re.Pattern[str]] = None

    def _get_compiled_pattern(self) -> re.Pattern[str]:
        """Lazy compilation of regex pattern."""
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, self.flags)
        return self._compiled

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Evaluate if actual output matches the regex pattern."""
        if not eval_input.actual_output:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning="Missing actual_output",
            )

        try:
            compiled = self._get_compiled_pattern()
            text = eval_input.actual_output

            if self.full_match:
                match = compiled.fullmatch(text)
            else:
                match = compiled.search(text)

            if match:
                return MetricResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    threshold=self.threshold,
                    reasoning=f"Pattern matched: '{match.group()}'",
                    details={
                        "pattern": self.pattern,
                        "match": match.group(),
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "full_match": self.full_match,
                    },
                )
            else:
                return MetricResult(
                    metric_name=self.name,
                    score=0.0,
                    passed=False,
                    threshold=self.threshold,
                    reasoning=f"Pattern '{self.pattern}' did not match",
                    details={
                        "pattern": self.pattern,
                        "full_match": self.full_match,
                        "actual_output_preview": text[:100]
                        + ("..." if len(text) > 100 else ""),
                    },
                )

        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                reasoning=f"Invalid regex pattern: {e}",
                details={"error": str(e), "pattern": self.pattern},
            )


# =============================================================================
# TOOL TRAJECTORY METRICS
# =============================================================================


def _match_tool_name(actual_name: str, expected_pattern: str) -> bool:
    """
    Match tool name against expected pattern (supports wildcards).

    Examples:
        _match_tool_name("search_flights", "search_flights") -> True
        _match_tool_name("search_flights", "search_*") -> True
        _match_tool_name("search_flights", "*_flights") -> True
    """
    return fnmatch.fnmatch(actual_name, expected_pattern)


class ToolCallAccuracyMetric(Metric):
    """
    Measures if the correct tools were called.

    Score = matched_tools / max(expected_required, actual_count)

    Example:
        metric = ToolCallAccuracyMetric(threshold=0.8, ignore_extra=False)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        ignore_extra: bool = False,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize tool call accuracy metric.

        Args:
            threshold: Minimum score to pass
            ignore_extra: If True, don't penalize extra tool calls
            name: Metric name
            weight: Weight for aggregation
        """
        super().__init__(
            name=name or "tool_call_accuracy", threshold=threshold, weight=weight
        )
        self.ignore_extra = ignore_extra

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Evaluate tool call accuracy."""
        actual_tools = eval_input.tool_calls
        expected_tools = eval_input.expected_tools

        if not expected_tools:
            return MetricResult(
                metric_name=self.name,
                score=1.0 if not actual_tools else 0.0,
                passed=not actual_tools,
                threshold=self.threshold,
                reasoning="No expected tools defined",
            )

        # Get actual tool names
        actual_names = [tc.name for tc in actual_tools]

        # Match expected tools against actual
        matched: List[str] = []
        missing: List[str] = []
        required_count = 0

        for expected in expected_tools:
            if expected.required:
                required_count += 1

            # Check if any actual tool matches this expected tool
            found = False
            for actual_name in actual_names:
                if _match_tool_name(actual_name, expected.name):
                    if actual_name not in matched:
                        matched.append(actual_name)
                    found = True
                    break

            if not found and expected.required:
                missing.append(expected.name)

        # Calculate extra tools (not matching any expected pattern)
        extra: List[str] = []
        for actual_name in actual_names:
            matches_any = any(
                _match_tool_name(actual_name, exp.name) for exp in expected_tools
            )
            if not matches_any:
                extra.append(actual_name)

        # Calculate score
        if self.ignore_extra:
            # Score based only on required tools found
            denominator = required_count if required_count > 0 else 1
            score = len(matched) / denominator
        else:
            # Penalize extra tools
            denominator = max(required_count, len(actual_names))
            if denominator == 0:
                score = 1.0
            else:
                score = len(matched) / denominator

        score = min(1.0, max(0.0, score))

        # Build reasoning
        parts = [f"{len(matched)}/{required_count} required tools called"]
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        if extra and not self.ignore_extra:
            parts.append(f"Extra: {', '.join(extra)}")

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning="; ".join(parts),
            details={
                "matched": matched,
                "missing": missing,
                "extra": extra,
                "required_count": required_count,
                "actual_count": len(actual_names),
            },
        )


class ToolSequenceMetric(Metric):
    """
    Measures if tools were called in the correct order.

    Strict mode: Exact sequence match required (score 0 or 1)
    Non-strict mode: Longest Common Subsequence ratio

    Example:
        metric = ToolSequenceMetric(threshold=0.8, strict=False)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        strict: bool = False,
        name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize tool sequence metric.

        Args:
            threshold: Minimum score to pass
            strict: If True, require exact sequence match
            name: Metric name
            weight: Weight for aggregation
        """
        super().__init__(
            name=name or "tool_sequence", threshold=threshold, weight=weight
        )
        self.strict = strict

    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate LCS length using dynamic programming."""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _match_sequences(
        self, actual: List[str], expected: List[ExpectedTool]
    ) -> List[str]:
        """
        Match actual tool names to expected patterns, preserving order.
        Returns matched expected names in order found.
        """
        matched = []
        for actual_name in actual:
            for exp in expected:
                if _match_tool_name(actual_name, exp.name) and exp.name not in matched:
                    matched.append(exp.name)
                    break
        return matched

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Evaluate tool call sequence."""
        actual_tools = eval_input.tool_calls
        expected_tools = eval_input.expected_tools

        if not expected_tools:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                reasoning="No expected tools defined",
            )

        # Get required expected tool names in order
        expected_names = [exp.name for exp in expected_tools if exp.required]
        actual_names = [tc.name for tc in actual_tools]

        if not expected_names:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                reasoning="No required tools in expected sequence",
            )

        # Map actual to expected patterns (for wildcard support)
        matched_sequence = self._match_sequences(actual_names, expected_tools)

        if self.strict:
            # Exact sequence match
            score = 1.0 if matched_sequence == expected_names else 0.0
            reasoning = (
                "Exact sequence match"
                if score == 1.0
                else f"Sequence mismatch: expected {' → '.join(expected_names)}, got {' → '.join(actual_names)}"
            )
        else:
            # LCS-based scoring
            lcs_len = self._longest_common_subsequence(matched_sequence, expected_names)
            denominator = max(len(expected_names), len(matched_sequence))
            score = lcs_len / denominator if denominator > 0 else 1.0
            reasoning = f"Sequence similarity: {lcs_len}/{denominator} tools in order"

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning=reasoning,
            details={
                "expected_sequence": expected_names,
                "actual_sequence": actual_names,
                "matched_sequence": matched_sequence,
                "strict": self.strict,
            },
        )


class ToolArgsMatchMetric(Metric):
    """
    Measures if tool arguments match expected values.

    Score = matched_args / total_expected_args

    Supports fuzzy matching using an LLM for semantic comparison
    (e.g., "NYC" matches "New York City").

    Example:
        metric = ToolArgsMatchMetric(threshold=0.8, ignore_extra=True)
        metric = ToolArgsMatchMetric(fuzzy_match=True)  # Use LLM for semantic comparison
    """

    def __init__(
        self,
        threshold: float = 0.8,
        ignore_extra: bool = True,
        partial_match: bool = True,
        fuzzy_match: bool = False,
        name: Optional[str] = None,
        weight: float = 1.0,
        llm_client: Optional[Any] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize tool args match metric.

        Args:
            threshold: Minimum score to pass
            ignore_extra: If True, don't penalize extra arguments
            partial_match: If True, allow subset of expected args
            fuzzy_match: If True, use LLM for semantic comparison
            name: Metric name
            weight: Weight for aggregation
            llm_client: LLM client for fuzzy matching
            model: Model name for fuzzy matching
        """
        super().__init__(
            name=name or "tool_args_match", threshold=threshold, weight=weight
        )
        self.ignore_extra = ignore_extra
        self.partial_match = partial_match
        self.fuzzy_match = fuzzy_match
        self._llm_client = llm_client
        self._model_override = model

    def _get_llm_client(self) -> Any:
        """Get the LLM client for fuzzy matching."""
        if self._llm_client:
            return self._llm_client

        if self._model_override:
            from fasteval.providers.registry import create_provider_for_model

            return create_provider_for_model(self._model_override)

        from fasteval.providers.registry import get_default_provider

        return get_default_provider()

    def _compare_values(self, actual: Any, expected: Any) -> bool:
        """Compare two values for equality (non-fuzzy)."""
        # Handle None
        if expected is None:
            return True  # None means "don't care"

        # Direct equality
        if actual == expected:
            return True

        # String comparison (case-insensitive)
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.lower().strip() == expected.lower().strip()

        # Numeric comparison with tolerance
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) < 0.0001

        return False

    async def _fuzzy_compare_values(
        self, actual: Any, expected: Any, arg_name: str
    ) -> tuple[bool, float, str]:
        """
        Use LLM for semantic comparison of values.

        Returns: (is_match, confidence, reasoning)
        """
        # First try exact match
        if self._compare_values(actual, expected):
            return True, 1.0, "Exact match"

        # Use LLM for semantic comparison
        prompt = f"""Compare these two argument values semantically and determine if they represent the same intent/meaning.

Argument name: {arg_name}
Expected value: {expected}
Actual value: {actual}

Consider:
- Synonyms and equivalent expressions (e.g., "NYC" = "New York City")
- Different formats representing the same data (e.g., "2024-01-15" = "January 15, 2024")
- Semantic equivalence in context

Return JSON:
{{"match": true, "confidence": 0.95, "reasoning": "brief explanation"}}
or
{{"match": false, "confidence": 0.1, "reasoning": "brief explanation"}}"""

        try:
            client = self._get_llm_client()
            # Use invoke() with message format per LLMClient protocol
            messages = [{"role": "user", "content": prompt}]
            response = await client.invoke(messages)

            from fasteval.utils.json_parsing import extract_json_from_text

            result = extract_json_from_text(response)
            if result is None:
                return False, 0.0, f"Could not parse LLM response: {response[:100]}"

            is_match = result.get("match", False)
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "LLM comparison")

            return is_match, confidence, reasoning
        except Exception as e:
            # Fallback to exact match on error
            return False, 0.0, f"Fuzzy match failed: {e}"

    def _match_args(
        self, actual_args: Dict[str, Any], expected_args: Dict[str, Any]
    ) -> tuple[int, int, List[str], List[str]]:
        """
        Match actual args against expected (synchronous, non-fuzzy).

        Returns: (matched_count, expected_count, matched_keys, missing_keys)
        """
        if not expected_args:
            return 0, 0, [], []

        matched_keys: List[str] = []
        missing_keys: List[str] = []

        for key, expected_value in expected_args.items():
            if key in actual_args:
                if self._compare_values(actual_args[key], expected_value):
                    matched_keys.append(key)
                else:
                    missing_keys.append(f"{key} (value mismatch)")
            else:
                missing_keys.append(f"{key} (missing)")

        return len(matched_keys), len(expected_args), matched_keys, missing_keys

    async def _match_args_fuzzy(
        self, actual_args: Dict[str, Any], expected_args: Dict[str, Any]
    ) -> tuple[float, int, List[str], List[str], List[str]]:
        """
        Match actual args against expected with fuzzy matching.

        Returns: (total_score, expected_count, matched_keys, partial_keys, missing_keys)
        """
        if not expected_args:
            return 0.0, 0, [], [], []

        matched_keys: List[str] = []
        partial_keys: List[str] = []  # Fuzzy matches with confidence < 1.0
        missing_keys: List[str] = []
        total_score = 0.0

        for key, expected_value in expected_args.items():
            if key in actual_args:
                is_match, confidence, reasoning = await self._fuzzy_compare_values(
                    actual_args[key], expected_value, key
                )
                if is_match:
                    if confidence >= 0.9:
                        matched_keys.append(key)
                        total_score += 1.0
                    else:
                        partial_keys.append(
                            f"{key} (fuzzy: {confidence:.0%}, {reasoning})"
                        )
                        total_score += confidence
                else:
                    missing_keys.append(f"{key} (value mismatch: {reasoning})")
            else:
                missing_keys.append(f"{key} (missing)")

        return total_score, len(expected_args), matched_keys, partial_keys, missing_keys

    async def evaluate(self, eval_input: EvalInput) -> MetricResult:
        """Evaluate tool arguments match."""
        actual_tools = eval_input.tool_calls
        expected_tools = eval_input.expected_tools

        if not expected_tools:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                reasoning="No expected tools defined",
            )

        total_expected_args = 0
        total_matched_score = 0.0  # Use float for fuzzy matching
        details_per_tool: List[Dict[str, Any]] = []

        for expected in expected_tools:
            if not expected.args:
                continue  # No args to check for this tool

            # Find matching actual tool
            matching_actual = None
            for actual in actual_tools:
                if _match_tool_name(actual.name, expected.name):
                    matching_actual = actual
                    break

            if matching_actual is None:
                # Tool not called - count all expected args as missing
                total_expected_args += len(expected.args)
                details_per_tool.append(
                    {
                        "tool": expected.name,
                        "status": "not_called",
                        "expected_args": list(expected.args.keys()),
                        "matched": [],
                        "missing": list(expected.args.keys()),
                    }
                )
                continue

            # Compare arguments - use fuzzy matching if enabled
            if self.fuzzy_match:
                (
                    matched_score,
                    expected_count,
                    matched_keys,
                    partial_keys,
                    missing_keys,
                ) = await self._match_args_fuzzy(
                    matching_actual.arguments, expected.args
                )
                total_expected_args += expected_count
                total_matched_score += matched_score

                details_per_tool.append(
                    {
                        "tool": expected.name,
                        "status": "called",
                        "expected_args": list(expected.args.keys()),
                        "actual_args": list(matching_actual.arguments.keys()),
                        "matched": matched_keys,
                        "partial": partial_keys,
                        "missing": missing_keys,
                        "fuzzy_match": True,
                    }
                )
            else:
                matched, expected_count, matched_keys, missing_keys = self._match_args(
                    matching_actual.arguments, expected.args
                )
                total_expected_args += expected_count
                total_matched_score += matched

                details_per_tool.append(
                    {
                        "tool": expected.name,
                        "status": "called",
                        "expected_args": list(expected.args.keys()),
                        "actual_args": list(matching_actual.arguments.keys()),
                        "matched": matched_keys,
                        "missing": missing_keys,
                    }
                )

        # Calculate score
        if total_expected_args == 0:
            score = 1.0
            reasoning = "No expected arguments to check"
        else:
            score = total_matched_score / total_expected_args
            if self.fuzzy_match:
                reasoning = f"{total_matched_score:.1f}/{total_expected_args} expected arguments matched (fuzzy)"
            else:
                reasoning = f"{int(total_matched_score)}/{total_expected_args} expected arguments matched"

        # Add details about mismatches
        mismatches = []
        for detail in details_per_tool:
            issues = []
            if detail.get("partial"):
                issues.extend(detail["partial"])
            if detail.get("missing"):
                issues.extend(detail["missing"])
            if issues:
                mismatches.append(f"{detail['tool']}: {', '.join(issues)}")

        if mismatches:
            reasoning += f"; Issues: {'; '.join(mismatches)}"

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=self._determine_pass(score),
            threshold=self.threshold,
            reasoning=reasoning,
            details={
                "total_expected_args": total_expected_args,
                "total_matched_score": total_matched_score,
                "fuzzy_match": self.fuzzy_match,
                "tools": details_per_tool,
            },
        )
