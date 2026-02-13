# fasteval

[![PyPI version](https://img.shields.io/pypi/v/fasteval-core.svg)](https://pypi.org/project/fasteval-core/)
![Python versions](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12_|_3.13_|_3.14-blue?logo=python)
[![CI](https://github.com/intuit/fasteval/actions/workflows/ci.yml/badge.svg)](https://github.com/intuit/fasteval/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A **decorator-first LLM evaluation library** for testing AI agents and LLMs. Stack decorators to define evaluation criteria, run with pytest.

## Features

- **Decorator-based metrics** -- stack `@fe.correctness`, `@fe.relevance`, `@fe.hallucination`, and 30+ more
- **pytest native** -- run evaluations with `pytest`, get familiar pass/fail output
- **LLM-as-judge + deterministic** -- semantic LLM metrics alongside ROUGE, exact match, JSON schema, regex
- **Multi-modal** -- evaluate vision, audio, and image generation models
- **Conversation metrics** -- context retention, topic drift, consistency for multi-turn agents
- **RAG metrics** -- faithfulness, contextual precision, contextual recall, answer correctness
- **Tool trajectory** -- verify agent tool calls, argument matching, call sequences
- **Pluggable providers** -- OpenAI (default), Anthropic, Azure OpenAI, Ollama

## Quick Start

```bash
pip install fasteval-core
```

Set your LLM provider key:

```bash
export OPENAI_API_KEY=sk-your-key-here
```

Write your first evaluation test:

```python
import fasteval as fe

@fe.correctness(threshold=0.8)
@fe.relevance(threshold=0.7)
def test_qa_agent():
    response = my_agent("What is the capital of France?")
    fe.score(response, expected_output="Paris", input="What is the capital of France?")
```

Run it:

```bash
pytest test_qa_agent.py -v
```

## Installation

```bash
# pip
pip install fasteval-core

# uv
uv add fasteval-core
```

### Optional Extras

```bash
# Anthropic provider
pip install fasteval-core[anthropic]

# Vision-language evaluation (GPT-4V, Claude Vision)
pip install fasteval-core[vision]

# Audio/speech evaluation (Whisper, ASR)
pip install fasteval-core[audio]

# Image generation evaluation (DALL-E, Stable Diffusion)
pip install fasteval-core[image-gen]

# All multi-modal features
pip install fasteval-core[multimodal]
```

## Usage Examples

### Deterministic Metrics

```python
import fasteval as fe

@fe.contains()
def test_keyword_present():
    fe.score("The answer is 42", expected_output="42")

@fe.rouge(threshold=0.6, rouge_type="rougeL")
def test_summary_quality():
    fe.score(actual_output=summary, expected_output=reference)
```

### RAG Evaluation

```python
@fe.faithfulness(threshold=0.8)
@fe.contextual_precision(threshold=0.7)
def test_rag_pipeline():
    result = rag_pipeline("How does photosynthesis work?")
    fe.score(
        actual_output=result.answer,
        context=result.retrieved_docs,
        input="How does photosynthesis work?",
    )
```

### Tool Trajectory

```python
@fe.tool_call_accuracy(threshold=0.9)
def test_agent_tools():
    result = agent.run("Book a flight to Paris")
    fe.score(
        actual_tools=result.tool_calls,
        expected_tools=[
            {"name": "search_flights", "args": {"destination": "Paris"}},
            {"name": "book_flight"},
        ],
    )
```

### Metric Stacks

```python
@fe.correctness(threshold=0.8, weight=2.0)
@fe.relevance(threshold=0.7, weight=1.0)
@fe.coherence(threshold=0.6, weight=1.0)
def test_comprehensive():
    response = agent("Explain quantum computing")
    fe.score(response, expected_output=reference_answer, input="Explain quantum computing")
```

## Plugins

| Plugin | Description | Install |
|--------|-------------|---------|
| [fasteval-langfuse](./plugins/fasteval-langfuse/) | Evaluate Langfuse production traces with fasteval metrics | `pip install fasteval-langfuse` |
| [fasteval-langgraph](./plugins/fasteval-langgraph/) | Test harness for LangGraph agents | `pip install fasteval-langgraph` |
| [fasteval-observe](./plugins/fasteval-observe/) | Runtime monitoring with async sampling | `pip install fasteval-observe` |

## Local Development

```bash
# Install uv
brew install uv

# Create virtual environment and install dependencies
uv sync --all-extras

# Run the test suite
uv run tox

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .
```

## Documentation

Full documentation is available in the [docs/](./docs/) directory, covering:

- [Getting Started](./docs/getting-started/) -- installation, quickstart
- [Core Concepts](./docs/core-concepts/) -- decorators, metrics, scoring, data sources
- [LLM Metrics](./docs/llm-metrics/) -- correctness, relevance, hallucination, and more
- [Deterministic Metrics](./docs/deterministic-metrics/) -- ROUGE, exact match, regex, JSON schema
- [RAG Metrics](./docs/rag-metrics/) -- faithfulness, contextual precision/recall
- [Conversation Metrics](./docs/conversation-metrics/) -- context retention, consistency
- [Multi-Modal](./docs/multimodal/) -- vision, audio, image generation evaluation
- [Plugins](./docs/plugins/) -- Langfuse, LangGraph, Observe
- [API Reference](./docs/api-reference/) -- decorators, evaluator, models, score

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, coding standards, and how to submit pull requests.

## License

Apache License 2.0 -- see [LICENSE](./LICENSE) for details.
