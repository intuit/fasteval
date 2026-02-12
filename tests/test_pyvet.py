"""Tests for fasteval library."""

import pytest
from pydantic import BaseModel

import fasteval as fe
from fasteval.models.evaluation import EvalInput, MetricResult

# === Unit Tests for Models ===


class TestEvalInput:
    """Tests for EvalInput model."""

    def test_basic_creation(self):
        """Test basic EvalInput creation."""
        eval_input = EvalInput(
            actual_output="4",
            expected_output="4",
            input="What is 2+2?",
        )
        assert eval_input.actual_output == "4"
        assert eval_input.expected_output == "4"
        assert eval_input.input == "What is 2+2?"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed (Pydantic extra='allow')."""
        # EvalInput has model_config = {"extra": "allow"}
        # Using model_construct to bypass type checking for extra fields
        eval_input = EvalInput.model_construct(
            actual_output="test",
            custom_field="custom_value",
        )
        assert getattr(eval_input, "custom_field") == "custom_value"

    def test_context_fields(self):
        """Test context and retrieval_context."""
        eval_input = EvalInput(
            actual_output="Paris",
            context=["France is a country in Europe", "Paris is the capital"],
            retrieval_context=["Paris is the capital of France"],
        )
        assert eval_input.context is not None and len(eval_input.context) == 2
        assert (
            eval_input.retrieval_context is not None
            and len(eval_input.retrieval_context) == 1
        )


class TestMetricResult:
    """Tests for MetricResult model."""

    def test_basic_creation(self):
        """Test basic MetricResult creation."""
        result = MetricResult(
            metric_name="correctness",
            score=0.85,
            passed=True,
            threshold=0.8,
            reasoning="Good answer",
        )
        assert result.metric_name == "correctness"
        assert result.score == 0.85
        assert result.passed is True

    def test_score_validation(self):
        """Test that score must be between 0 and 1."""
        with pytest.raises(ValueError):
            MetricResult(
                metric_name="test",
                score=1.5,  # Invalid
                passed=True,
                threshold=0.5,
            )


# === Unit Tests for Deterministic Metrics ===


class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""

    @pytest.mark.asyncio
    async def test_exact_match_success(self):
        """Test exact match succeeds with identical strings."""
        metric = fe.ExactMatchMetric(case_sensitive=False)
        eval_input = EvalInput(
            actual_output="Hello World",
            expected_output="hello world",
        )
        result = await metric.evaluate(eval_input)
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_exact_match_failure(self):
        """Test exact match fails with different strings."""
        metric = fe.ExactMatchMetric()
        eval_input = EvalInput(
            actual_output="Hello World",
            expected_output="Goodbye World",
        )
        result = await metric.evaluate(eval_input)
        assert result.score == 0.0
        assert result.passed is False


class TestContainsMetric:
    """Tests for ContainsMetric."""

    @pytest.mark.asyncio
    async def test_contains_success(self):
        """Test contains succeeds when expected is in actual."""
        metric = fe.ContainsMetric()
        eval_input = EvalInput(
            actual_output="The answer is 42",
            expected_output="42",
        )
        result = await metric.evaluate(eval_input)
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_contains_failure(self):
        """Test contains fails when expected is not in actual."""
        metric = fe.ContainsMetric()
        eval_input = EvalInput(
            actual_output="The answer is 42",
            expected_output="100",
        )
        result = await metric.evaluate(eval_input)
        assert result.score == 0.0
        assert result.passed is False


class TestJsonMetric:
    """Tests for JsonMetric with Pydantic validation."""

    class UserSchema(BaseModel):
        name: str
        age: int

    @pytest.mark.asyncio
    async def test_valid_json(self):
        """Test JSON validation with valid data."""
        metric = fe.JsonMetric(model=self.UserSchema)
        eval_input = EvalInput(
            actual_output='{"name": "Alice", "age": 30}',
        )
        result = await metric.evaluate(eval_input)
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_json_syntax(self):
        """Test JSON validation with invalid syntax."""
        metric = fe.JsonMetric(model=self.UserSchema)
        eval_input = EvalInput(
            actual_output="{invalid json}",
        )
        result = await metric.evaluate(eval_input)
        assert result.score == 0.0
        assert result.passed is False
        # Error type is "validation" for all schema validation failures
        assert result.details.get("error_type") == "validation"

    @pytest.mark.asyncio
    async def test_invalid_schema(self):
        """Test JSON validation with schema violation."""
        metric = fe.JsonMetric(model=self.UserSchema)
        eval_input = EvalInput(
            actual_output='{"name": "Alice", "age": "not a number"}',
        )
        result = await metric.evaluate(eval_input)
        assert result.score == 0.0
        assert result.passed is False
        assert "validation" in result.details.get("error_type", "")


# === Unit Tests for Cache ===


class TestMemoryCache:
    """Tests for MemoryCache."""

    def test_basic_get_set(self):
        """Test basic cache operations."""
        cache = fe.MemoryCache(max_size=100)
        cache.set("key1", {"value": 1})

        result = cache.get("key1")
        assert result == {"value": 1}

        result = cache.get("nonexistent")
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = fe.MemoryCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to make it most recently used
        cache.get("a")

        # Add new item, should evict "b" (least recently used)
        cache.set("d", 4)

        assert cache.get("a") == 1  # Still there
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3  # Still there
        assert cache.get("d") == 4  # New item

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = fe.MemoryCache(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1


# === Unit Tests for Score API ===


class TestScoreAPI:
    """Tests for score() API with immediate evaluation."""

    def test_score_returns_eval_result(self):
        """Test that score() returns EvalResult with eval_input."""
        # Without decorators, score returns a passing result
        result = fe.score(
            actual_output="4",
            expected_output="4",
            input="What is 2+2?",
        )

        # With no metrics, returns a passing EvalResult
        from fasteval.models.evaluation import EvalResult

        assert isinstance(result, EvalResult)
        assert result.passed is True
        assert result.eval_input.actual_output == "4"
        assert result.eval_input.expected_output == "4"

    def test_score_with_decorator_evaluates_immediately(self):
        """Test that score() evaluates immediately with decorated function."""

        @fe.contains()
        def decorated_func():
            result = fe.score("The answer is Paris", "Paris", input="Capital?")
            return result

        result = decorated_func()
        assert result.passed is True
        assert len(result.metric_results) == 1
        assert result.metric_results[0].metric_name == "contains"

    def test_score_raises_on_failure(self):
        """Test that score() raises EvaluationFailedError on failure."""

        @fe.exact_match()  # threshold=1.0 by default
        def failing_func():
            fe.score("wrong answer", "correct answer", input="test")

        with pytest.raises(fe.EvaluationFailedError) as exc_info:
            failing_func()

        assert exc_info.value.result.passed is False
        assert exc_info.value.result.aggregate_score == 0.0


# === Unit Tests for Decorators ===


class TestDecorators:
    """Tests for metric decorators."""

    def test_correctness_decorator_attaches_config(self):
        """Test that @correctness attaches MetricConfig."""

        @fe.correctness(threshold=0.9)
        async def my_test():
            pass

        configs = getattr(my_test, "_fasteval_metrics", [])
        assert len(configs) == 1
        assert configs[0].metric_type == "correctness"
        assert configs[0].threshold == 0.9

    def test_multiple_decorators_stack(self):
        """Test that multiple metric decorators stack."""

        @fe.correctness(threshold=0.8)
        @fe.hallucination(threshold=0.9)
        async def my_test():
            pass

        configs = getattr(my_test, "_fasteval_metrics", [])
        assert len(configs) == 2
        types = {c.metric_type for c in configs}
        assert types == {"correctness", "hallucination"}

    def test_json_decorator_with_pydantic_model(self):
        """Test @json decorator with Pydantic model."""

        class ResponseSchema(BaseModel):
            status: str
            data: dict

        @fe.json(model=ResponseSchema)
        async def my_test():
            pass

        configs = getattr(my_test, "_fasteval_metrics", [])
        assert len(configs) == 1
        assert configs[0].metric_type == "json"
        assert configs[0].config["pydantic_model"] == ResponseSchema
