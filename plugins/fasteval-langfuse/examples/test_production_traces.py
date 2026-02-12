"""Example: Evaluating production traces from Langfuse."""

import fasteval as fe
from fasteval_langfuse import langfuse_traces
from fasteval_langfuse.sampling import (
    RandomSamplingStrategy,
    ScoreBasedSamplingStrategy,
    StratifiedSamplingStrategy,
)


# Example 1: Random sampling for cost-effective monitoring
@fe.correctness(threshold=0.8)
@fe.hallucination(threshold=0.9)
@langfuse_traces(
    project="production",
    filter_tags=["customer-support"],
    time_range="last_24h",
    sampling=RandomSamplingStrategy(sample_size=200, seed=42),
)
def test_production_quality(trace_id, input, output, context, metadata):
    """Evaluate 200 random traces from the last 24 hours."""
    # Get expected answer from metadata if available
    expected = metadata.get("expected_answer")

    # Evaluate
    fe.score(output, expected, context=context, input=input)


# Example 2: Focus on failures with score-based sampling
@fe.relevance(threshold=0.7)
@fe.faithfulness(threshold=0.8)
@langfuse_traces(
    project="production",
    filter_tags=["rag"],
    time_range="last_7d",
    sampling=ScoreBasedSamplingStrategy(
        score_name="user_rating",
        low_score_threshold=3.0,
        low_score_rate=1.0,  # 100% of low ratings
        high_score_rate=0.05,  # 5% of high ratings
    ),
)
def test_low_rated_interactions(trace_id, input, output, context, metadata):
    """Focus evaluation budget on traces with poor user ratings."""
    # Context auto-extracted from metadata
    if not context:
        context = metadata.get("docs") or metadata.get("retrieval_context")

    expected = metadata.get("expected_answer")
    fe.score(output, expected, context=context, input=input)


# Example 3: Ensure quality across user segments
@fe.correctness(threshold=0.8)
@fe.relevance(threshold=0.75)
@langfuse_traces(
    project="production",
    filter_tags=["customer-query"],
    time_range="last_7d",
    sampling=StratifiedSamplingStrategy(
        strata_key="metadata.user_type",
        samples_per_stratum=30,
    ),
)
def test_quality_across_user_types(trace_id, input, output, context, metadata):
    """Sample evenly across free/paid users to ensure fair quality."""
    user_type = metadata.get("user_type", "unknown")

    # Evaluate
    fe.score(output, input=input)

    print(f"Evaluated trace {trace_id} for user_type: {user_type}")


# Run with pytest:
# pytest examples/test_production_traces.py -v
#
# Output example:
# Evaluating 200/5,432 traces (3.7% sample, strategy=RandomSamplingStrategy)
# test_production_quality PASSED
#
# Scores are automatically pushed to Langfuse with prefix "fasteval_"
