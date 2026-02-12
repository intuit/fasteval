"""Example: Evaluating Langfuse datasets."""

import fasteval as fe
from fasteval_langfuse import langfuse_dataset


# Example 1: Basic dataset evaluation
@fe.correctness(threshold=0.85)
@fe.relevance(threshold=0.8)
@langfuse_dataset(name="qa-golden-set", version="v3")
def test_new_agent_version(input, expected_output):
    """Test new agent version against golden dataset."""
    # Run with new agent version
    response = my_new_agent_v2(input)

    # Evaluate
    fe.score(response, expected_output, input=input)


# Example 2: Using custom metadata fields
@fe.correctness(threshold=0.85)
@langfuse_dataset(name="qa-golden-set", version="v3")
def test_with_metadata(input, expected_output, difficulty, category):
    """
    Test using custom metadata fields from dataset.

    Dataset items have metadata like:
    {
        "difficulty": "hard",
        "category": "technical"
    }
    """
    response = my_new_agent_v2(input)

    # Can use metadata in evaluation logic
    print(f"Testing {category} question with difficulty: {difficulty}")

    fe.score(response, expected_output, input=input)


# Example 3: Only using what you need
@fe.correctness(threshold=0.8)
@langfuse_dataset(name="qa-inputs-only")
def test_minimal(input):
    """Test when dataset only has inputs, no expected outputs."""
    response = my_new_agent_v2(input)

    # Evaluate without expected output (judge-only metrics)
    fe.score(response, input=input)


# Example 4: A/B testing comparison
@fe.correctness(threshold=0.85)
@langfuse_dataset(name="qa-golden-set", version="v3")
def test_baseline_agent(input, expected_output):
    """Test baseline agent version for comparison."""
    response = my_baseline_agent_v1(input)
    fe.score(response, expected_output, input=input)


# Mock agents for example
def my_new_agent_v2(query: str) -> str:
    """New agent version."""
    return f"Response to: {query}"


def my_baseline_agent_v1(query: str) -> str:
    """Baseline agent version."""
    return f"Baseline response to: {query}"


# Run comparison:
# pytest examples/test_datasets.py -v --html=comparison.html
#
# Compare results in Langfuse dashboard to see:
# - Which version has higher scores
# - Which specific test cases improved/degraded
# - Cost differences between versions
