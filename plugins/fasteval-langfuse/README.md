# fasteval-langfuse

Langfuse integration for [fasteval](https://github.com/intuit/fasteval) - evaluate production traces with fasteval's research-backed metrics.

## Installation

```bash
pip install fasteval fasteval-langfuse
```

## Quick Start

### Evaluate Production Traces

Fetch traces from Langfuse and evaluate them with fasteval metrics:

```python
from fasteval_langfuse import langfuse_traces
from fasteval_langfuse.sampling import RandomSamplingStrategy
import fasteval as fe

@fe.correctness(threshold=0.8)
@fe.hallucination(threshold=0.9)
@langfuse_traces(
    project="production",
    filter_tags=["customer-support"],
    time_range="last_24h",
    sampling=RandomSamplingStrategy(sample_size=200)
)
def test_production_traces(trace_id, input, output, context, metadata):
    # Evaluate the trace
    fe.score(output, input=input)

# Run with pytest - scores automatically pushed to Langfuse
# pytest test_production.py -v
```

### Sampling Strategies

Reduce evaluation costs with intelligent sampling:

```python
from fasteval_langfuse.sampling import (
    RandomSamplingStrategy,
    StratifiedSamplingStrategy,
    ScoreBasedSamplingStrategy,
)

# Random sampling - 200 random traces
@langfuse_traces(
    project="prod",
    sampling=RandomSamplingStrategy(sample_size=200, seed=42)
)
def test_random_sample(trace_id, input, output, context, metadata):
    fe.score(output, input=input)

# Stratified sampling - even distribution across user types
@langfuse_traces(
    project="prod",
    sampling=StratifiedSamplingStrategy(
        strata_key="metadata.user_type",
        samples_per_stratum=30
    )
)
def test_across_segments(trace_id, input, output, context, metadata):
    fe.score(output, input=input)

# Score-based sampling - focus on failures
@langfuse_traces(
    project="prod",
    sampling=ScoreBasedSamplingStrategy(
        score_name="user_rating",
        low_score_threshold=3.0,
        low_score_rate=1.0,      # 100% of low ratings
        high_score_rate=0.05     # 5% of high ratings
    )
)
def test_failures(trace_id, input, output, context, metadata):
    fe.score(output, input=input)
```

## Built-in Sampling Strategies

- **NoSamplingStrategy**: Evaluate all matching traces (default)
- **RandomSamplingStrategy**: Unbiased random sampling
- **StratifiedSamplingStrategy**: Even distribution across groups
- **ScoreBasedSamplingStrategy**: Oversample low-scoring traces
- **RecentFirstSamplingStrategy**: Prioritize recent traces

## Dataset Integration

Evaluate against Langfuse datasets. All dataset columns are passed as parameters - declare what you need:

```python
from fasteval_langfuse import langfuse_dataset

# Basic usage
@fe.correctness(threshold=0.8)
@langfuse_dataset(name="qa-golden-set", version="v2")
def test_qa_dataset(input, expected_output):
    response = my_agent(input)
    fe.score(response, expected_output, input=input)

# Using custom metadata fields
@fe.correctness(threshold=0.8)
@langfuse_dataset(name="qa-golden-set", version="v2")
def test_with_metadata(input, expected_output, difficulty, category):
    # difficulty and category come from item.metadata
    response = my_agent(input)
    fe.score(response, expected_output, input=input)

# Only what you need
@fe.correctness(threshold=0.8)
@langfuse_dataset(name="inputs-only")
def test_minimal(input):
    # Only declare input, ignore other fields
    response = my_agent(input)
    fe.score(response, input=input)
```

## Configuration

```python
from fasteval_langfuse import configure_langfuse, LangfuseConfig

configure_langfuse(LangfuseConfig(
    public_key="pk-...",                # Or from LANGFUSE_PUBLIC_KEY env
    secret_key="sk-...",                # Or from LANGFUSE_SECRET_KEY env
    host="https://cloud.langfuse.com",  # Or self-hosted
    default_project="production",
    auto_push_scores=True,              # Push scores back automatically
    score_name_prefix="fasteval_",      # Prefix for score names
))
```

## RAG Evaluation with Context

The decorator automatically extracts context from trace metadata:

```python
@fe.faithfulness(threshold=0.8)
@fe.contextual_precision(threshold=0.7)
@langfuse_traces(
    project="prod",
    filter_tags=["rag"]
)
def test_rag_quality(trace_id, input, output, context, metadata):
    # context is auto-extracted from metadata keys:
    # - "context", "retrieved_docs", "documents", "retrieval_context"
    
    # Or manually extract if needed:
    if not context:
        context = metadata.get("custom_docs_key")
    
    fe.score(output, context=context, input=input)
```

## Benefits

- 💰 **Cost Reduction**: Reduce LLM evaluation costs by 90%+ with sampling
- ⚡ **Faster Feedback**: Evaluate in minutes vs hours
- 📊 **Research-Backed Metrics**: Use fasteval's validated evaluation metrics
- 🎯 **Focus on Issues**: Oversample failures with ScoreBasedSamplingStrategy
- ✅ **Zero Instrumentation**: Evaluate existing traces without code changes
- 🔄 **Automatic Scoring**: Evaluation results automatically sync to Langfuse

## License

MIT
