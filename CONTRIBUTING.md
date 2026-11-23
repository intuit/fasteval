# Contributing to fasteval

Thank you for your interest in contributing to fasteval! Whether it's fixing a bug, improving documentation, or adding a new metric, all contributions are welcome.

- [Reporting Issues](#reporting-issues)
- [Development Setup](#development-setup)
- [Building the UI](#building-the-ui)
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

2. Install all dependencies (including dev and test groups):

   ```bash
   uv sync --all-extras --group dev --group test
   ```

3. Verify everything works:

   ```bash
   uv run tox
   ```

4. Optionally, build the UI bundle (see [Building the UI](#building-the-ui)) if you'll use `fasteval ui`, the live view of `fasteval run`, or HTML run reports.

## Building the UI

The local UI (`fasteval ui`, the live view of `fasteval run`, and the self-contained HTML reports) is a React app in `ui/`. It's compiled into `fasteval/ui/static/dist/`, which ships inside the `fasteval-core` wheel.

The built bundle is **not committed**: `fasteval/ui/static/dist/` (and `ui/node_modules/`) are in `.gitignore`. CI builds it and checks that the wheel contains it, and the release workflow builds it before publishing. From a source checkout, build it yourself with Node.js (CI uses Node 20):

```bash
cd ui
npm ci
npm run build    # typechecks, then writes ../fasteval/ui/static/dist/
```

Without the bundle, the Python side still works: `fasteval ui` and the UI server show a "build the UI" message, and `fasteval run` falls back to headless mode without writing an HTML report.

For UI development with hot reload, run the API from a project and the Vite dev server side by side:

```bash
fasteval ui --port 8484 --no-open   # in a project with evals
cd ui && npm run dev                 # proxies /api to localhost:8484
```

`npm run typecheck` runs the TypeScript checks without building.

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
- Maintain code coverage at or above **85%**
- Tests live in `tests/` for the core package and `plugins/*/tests/` for plugins
- Coverage configuration is in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]` -- models, vision/audio/multimodal metrics, and other non-logic files are excluded from measurement

Run tests:

```bash
# Full test suite across Python versions
uv run tox

# Quick single-version test with coverage
uv run --group test pytest tests/ --cov=fasteval --cov-report=term -v

# Run a specific test
uv run --group test pytest tests/test_example.py::test_name -v

# Run plugin tests (from plugin directory)
cd plugins/fasteval-langgraph
uv run pytest tests/ -v
```

> **Note**: The project includes a custom pytest plugin (`fasteval.testing.plugin`). When running tests with coverage, the plugin is automatically disabled via `addopts` in `pyproject.toml` (`-p no:fasteval`) to ensure accurate coverage tracking.

## Pull Request Process

1. Ensure all tests pass and linting is clean.
2. Update documentation if your change affects user-facing behavior. Docs are published at [fasteval.io](https://fasteval.io) and source lives in `docs/`.
3. Open a pull request against `main` with a clear description of your changes.
4. A maintainer will review your PR, typically within a few business days.
5. Once approved, a maintainer will merge your contribution.

### What We Look For

- Tests covering new functionality
- Type hints on all new functions
- Documentation updates where applicable
- Adherence to the existing code style
- Clear, focused commits (one logical change per commit)

### Writing Custom Metrics

If you're contributing a new metric, see the [Custom Metrics guide](https://fasteval.io/docs/advanced/custom-metrics) for the expected patterns. All metrics should:
- Extend `Metric` (deterministic) or `BaseLLMMetric` (LLM-based)
- Include a corresponding decorator in `fasteval/core/decorators.py`
- Be registered in `METRIC_REGISTRY` in `fasteval/core/evaluator.py`
- Have tests with >85% coverage

## Project Structure

```
fasteval/
├── core/           # Decorators, scoring engine, evaluator
├── metrics/        # Metric implementations (LLM, deterministic, conversation)
├── models/         # Pydantic models (EvalInput, EvalResult, MetricResult)
├── providers/      # LLM provider clients (OpenAI, Anthropic)
├── cache/          # In-memory LRU caching
├── collectors/     # Result collection and reporting
│   └── reporters/  # Output reporters (JSON, HTML)
├── utils/          # Helpers (formatting, JSON parsing, async, judge call tracing)
├── testing/        # pytest plugin (--fe-only, --fe-output, gates, datasets, xdist, events)
├── cli/            # `fasteval` command: run, list, profiles, report, ui, init; [tool.fasteval] config
├── ui/             # Local UI server, run history, static HTML report (bundle in ui/static/dist/)
└── judges.py       # Prompt judges (judges/*.toml) and run-time judge swaps

ui/                      # React source of the local UI (built into fasteval/ui/static/dist/)

plugins/
├── fasteval-langfuse/   # Langfuse production trace evaluation
├── fasteval-langgraph/  # LangGraph agent testing
└── fasteval-observe/    # Runtime monitoring

docs/                    # MDX documentation (published at fasteval.io)
tests/                   # Core package tests
```

## Questions?

Open a [GitHub Discussion](https://github.com/intuit/fasteval/discussions) or reach out to the project [code owners](./.github/CODEOWNERS).
