# Contributing to fasteval

Thank you for your interest in contributing to fasteval! Whether it's fixing a bug, improving documentation, or adding a new metric, all contributions are welcome.

- [Reporting Issues](#reporting-issues)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Reporting Issues

- **Bugs**: Open a [GitHub Issue](https://github.com/intuit/fasteval/issues) with reproduction steps, expected vs actual behavior, and your environment (Python version, fasteval version, OS).
- **Feature requests**: Open a [GitHub Issue](https://github.com/intuit/fasteval/issues) describing the problem, your proposed solution, and any alternatives you considered.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Getting Started

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/fasteval.git
   cd fasteval
   ```

2. Install dependencies:

   ```bash
   uv sync --all-extras
   ```

3. Verify everything works:

   ```bash
   uv run tox
   ```

## Making Changes

1. Create a branch from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, including tests and documentation updates.

3. Run the full test and lint suite:

   ```bash
   uv run tox
   ```

4. Commit with a clear message describing what changed and why:

   ```bash
   git commit -m "Add support for custom metric weights in stacks"
   ```

5. Push and open a pull request:

   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

This project uses the following tools for consistent code style:

- **[Black](https://black.readthedocs.io/)** for code formatting (line length 88)
- **[isort](https://pycqa.github.io/isort/)** for import sorting (Black-compatible profile)
- **[mypy](https://mypy.readthedocs.io/)** for static type checking

Format your code before committing:

```bash
uv run black .
uv run isort .
uv run mypy .
```

### Conventions

- Use type hints for all function signatures
- Prefer Pydantic models over raw dictionaries for input validation
- Use `async def` for asynchronous operations, `def` for pure functions
- Use descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_context`)
- Write docstrings for public functions and classes

## Testing

- All new functionality must have corresponding tests
- Maintain code coverage at or above 80%
- Tests live in `tests/` for the core package and `plugins/*/tests/` for plugins

Run tests:

```bash
# Full test suite across Python versions
uv run tox

# Quick single-version test
uv run pytest tests/ -v --cov=fasteval

# Run a specific test
uv run pytest tests/test_example.py::test_name -v
```

## Pull Request Process

1. Ensure all tests pass and linting is clean.
2. Update documentation if your change affects user-facing behavior (see `docs/`).
3. Open a pull request against `main` with a clear description of your changes.
4. A maintainer will review your PR, typically within a few business days.
5. Once approved, a maintainer will merge your contribution.

### What We Look For

- Tests covering new functionality
- Type hints on all new functions
- Documentation updates where applicable
- Adherence to the existing code style
- Clear, focused commits (one logical change per commit)

## Project Structure

```
fasteval/
├── core/           # Decorators, scoring engine, evaluator
├── metrics/        # Metric implementations (LLM, deterministic, conversation)
├── models/         # Pydantic models (EvalInput, EvalResult, MetricResult)
├── providers/      # LLM provider clients (OpenAI, Anthropic)
├── cache/          # Caching utilities
├── utils/          # Helpers (formatting, JSON parsing, async)
└── testing/        # pytest plugin

plugins/
├── fasteval-langfuse/   # Langfuse production trace evaluation
├── fasteval-langgraph/  # LangGraph agent testing
└── fasteval-observe/    # Runtime monitoring

docs/                    # MDX documentation
tests/                   # Core package tests
```

## Questions?

Open a [GitHub Discussion](https://github.com/intuit/fasteval/discussions) or reach out to the project [code owners](./.github/CODEOWNERS).
