# Changelog

All notable changes to this project will be documented in this file.

## fasteval-langfuse (unreleased)

### Fixes

- `@langfuse_traces` and `@langfuse_dataset` work under pytest again; every documented use failed with `fixture 'trace_id' not found`.
- A failing trace no longer stops the remaining traces, and its scores are now pushed to Langfuse (they were skipped, and scores could be left unflushed).
- Async test functions push their scores; they never reached Langfuse before.

### Changes

- Each trace / dataset item is its own pytest case (`test_x[<trace_id>]`): per-trace pass/fail, `--lf`, `-k <id>`, and `pytest -n` across traces, with all xdist workers evaluating the same sampled traces.
- Dataset items missing a metadata field some other item has get `None` for it.
- Calling a decorated function directly evaluates every trace and raises one `AssertionError` at the end listing the failures.
- Requires `fasteval-core>=1.3.0`.

## Unreleased

### Features

- `@fe.traces()` now runs: each yielded `EvalInput` becomes its own test case, evaluated against the stacked metrics (it was documented but never executed).
- `fe.evaluate()` / `fe.evaluate_async()`: score an output against named metrics outside pytest; failing results are returned, not raised.
- `fe.register_metric(name, MetricClass)`: register a custom metric and get its `@fe.<name>()` decorator.
- `llm=` is accepted as an alias for `llm_client=` on metric decorators (documented before, but crashed with a TypeError).
- Relative `fe.csv` / `fe.jsonl` paths are looked up next to the test file first, then in the working directory.

- **`fasteval` CLI** (also `python -m fasteval`): `fasteval` (project status), `run`, `list`/`ls`, `profiles`, `report`, `ui` and `init`. `fasteval run` runs only fasteval evals by default (`--all` includes plain tests), selects with `-t` tag expressions and `-k`, supports `-n` (pytest-xdist), `--lf`, `-x`, `--dry-run`, quality gates (`--min-score`, `--min-pass-rate`), judge swaps (`--judge METRIC=JUDGE`), `--json` and `--html`, and passes unknown flags (and anything after `--`) to pytest.
- **Live UI mode**: from a terminal, `fasteval run` opens the run in the browser and keeps the local UI open until Ctrl+C. **Headless mode** (`--headless`, `--ci`, profile `headless = true`, `$CI`, or no terminal) writes a self-contained HTML report to `.fasteval/report.html`.
- **Run history**: every run is recorded under `.fasteval/runs/<id>/` (gitignored automatically; the 50 most recent are kept) and can be re-run, re-run for failures only, compared and exported.
- **Profiles** in `[tool.fasteval]` or `fasteval.toml`: `default_profile`, a tag registry, and `profiles.<name>` with `paths`, `tags`, `keyword`, `workers`, `outputs`, `review_prompts`, `headless`, `min_score`, `min_pass_rate`, `env`, `pytest_args` and `extends` inheritance. Unknown keys are rejected with "did you mean" suggestions.
- **`@fe.tag(...)`**: tags are pytest markers (a tag alone doesn't make a test an eval), registered automatically (they work with `--strict-markers`), selectable with `fasteval run -t` or `pytest -m`, and usable on classes.
- **Datasets**: `@fe.dataset` (generator, function or iterable), `fe.case(id=..., tags=[...], **values)` and the new `@fe.jsonl(path, id_field=...)`. Every item becomes its own pytest test case with a readable id. `fe.Dataset` and `fe.Case` are exported.
- **Prompt judges**: LLM judges defined as `judges/<name>.toml` templates with `{{variable}}` placeholders, used with `@fe.prompt_judge("name")` or swapped in at run time with `fasteval run --judge metric=name` (pytest: `--fe-judge`). A swap changes the prompt; the eval keeps its own threshold and judge model unless the judge file sets `threshold` or `model`.
- **`judge_client`** in `[tool.fasteval]`: the project's default LLM judge, loaded by the pytest plugin unless code calls `fe.set_default_provider()`, and used by the UI judge playground.
- **Judge call recording**: LLM metrics record every judge call (provider, model, prompt messages, raw response, latency, error) in `MetricResult.details["judge_calls"]`.
- **Local UI** (`fasteval ui`): run history, live run view with per-case input/expected/actual, metric reasoning, judge prompts and raw responses, quality gates, re-run comparisons, an evals catalog, saved judges, and a judge playground for tuning a judge prompt against real, optionally labeled cases. Binds to localhost and requires a session token for state-changing requests.
- **pytest plugin options**: `--fe-only`, `--fe-min-score`, `--fe-min-pass-rate`, `--fe-judge`, `--fe-inventory` and `--fe-events`. Every eval gets an automatic `fasteval` marker.

### Changes

- **`@fe.csv` now creates one test case per row** (at collection time) instead of looping over all rows inside a single test, which didn't work under pytest. Columns are passed to parameters of the same name, or the whole row to a `case`/`row` parameter; an `id` column or `id_field=` names the cases. `fe.csv()` returns a `fe.Dataset`.
- Result test names include the parametrize id (`test_x[case]`), so cases are reported separately.

### Fixes

- Results are aggregated across pytest-xdist workers; previously the summary and `--fe-output` reports were empty under `-n`.

### Documentation

- New "CLI & UI" section (CLI overview, running evals, profiles, tags, local UI, prompt judges) and a CLI and pytest options reference; data sources, pytest integration, CI/CD and configuration pages updated for the CLI, datasets and the new `[tool.fasteval]` keys.

---

## v1.0.0a1 (2026-02-12)

### Changes

- Version reset to `1.0.0a1` for all packages (core + plugins)
- PR-label-driven release process (replaces manual tag-based releases)
- Dev preview releases via `release:*:dev` labels on open PRs
- Dynamic `__version__` via `importlib.metadata`
- Aligned `uv_build` across all packages

---

## Previous Versions (pre-reset)

## v3.0.0 (2026-01-30)

### Breaking Changes

- Initial open source release
- Package renamed to `fasteval` on PyPI
- Criteria documentation updated

## v2.6.1 (2026-01-30)

- Metric stacks implemented

## v2.5.1 (2026-01-29)

- Cookbook examples added
- Refactored `geval` to `criteria` metric

## v2.4.1 (2026-01-28)

- Cache documentation removed

## v2.4.0 (2026-01-28)

### Enhancements

- pytest mocking for tool calls and nodes added

## v2.2.1 (2026-01-25)

- pytest patterns documentation added

## v2.1.0 (2026-01-24)

### Enhancements

- Tool trajectory evaluation added
- Human review moved to main section

## v1.0.0 (2026-01-23)

### Breaking Changes

- Initial release
- Documentation added (LLM setup, RAG context, human-in-the-loop)
