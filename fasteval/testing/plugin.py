"""
FastEval pytest plugin.

Provides pytest integration for fasteval, including:
- --no-interactive flag to skip human review prompts in CI/CD
- --fe-output flag to export evaluation results (json, html)
- --fe-summary flag to print a console summary after the run
- --fe-only to run only fasteval evals (tests using fasteval decorators)
- --fe-min-score / --fe-min-pass-rate quality gates that fail the run
- --fe-inventory to write a JSON inventory of collected evals
- a ``fasteval`` marker added to every eval, so ``-m fasteval`` works
- tag registration from ``[tool.fasteval.tags]``
- aggregation of results across pytest-xdist workers

These are the primitives the ``fasteval`` CLI builds on; everything the CLI
does can also be done with plain ``pytest`` flags.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

FASTEVAL_MARKER = "fasteval"

# Attributes fasteval decorators set on test functions.
_FASTEVAL_ATTRS = (
    "_fasteval_metrics",
    "_fasteval_data",
    "_fasteval_human_review",
    "_fasteval_traces",
    "_fasteval_dataset",
)  # @fe.tag alone doesn't make a test an eval; it only tags it

# Markers that are pytest plumbing, not user tags.
_NON_TAG_MARKERS = {
    FASTEVAL_MARKER,
    "parametrize",
    "usefixtures",
    "filterwarnings",
    "skip",
    "skipif",
    "xfail",
    "asyncio",
    "anyio",
    "tryfirst",
    "trylast",
}

_WORKER_OUTPUT_KEY = "fasteval_results"

_EVENTS_KEY: "pytest.StashKey[Any]" = pytest.StashKey()

# logstart/logreport hooks don't receive config; remember the active one.
_current_config: Optional[Any] = None


def pytest_addoption(parser: Any) -> None:
    """Add fasteval-specific command line options to pytest."""
    parser.addoption(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Skip human review prompts (for CI/CD pipelines)",
    )
    parser.addoption(
        "--fe-output",
        action="append",
        default=[],
        help=(
            "Export evaluation results. Format: type:path "
            "(e.g., json:results.json, html:report.html). "
            "Can be specified multiple times."
        ),
    )
    parser.addoption(
        "--fe-summary",
        action="store_true",
        default=False,
        help="Print evaluation summary to console after test run",
    )
    parser.addoption(
        "--fe-only",
        action="store_true",
        default=False,
        help="Only run fasteval evals (tests that use fasteval decorators)",
    )
    parser.addoption(
        "--fe-min-score",
        type=float,
        default=None,
        help="Fail the run if the average aggregate score is below this (0-1)",
    )
    parser.addoption(
        "--fe-min-pass-rate",
        type=float,
        default=None,
        help="Fail the run if the eval pass rate is below this (0-1)",
    )
    parser.addoption(
        "--fe-inventory",
        default=None,
        help="Write a JSON inventory of collected evals to this path",
    )
    parser.addoption(
        "--fe-judge",
        action="append",
        default=[],
        metavar="METRIC=JUDGE",
        help="Swap metric METRIC for saved prompt judge JUDGE (judges/JUDGE.toml)",
    )
    parser.addoption(
        "--fe-events",
        default=None,
        metavar="DIR",
        help="Write a live JSONL event log of the run to DIR (used by fasteval ui)",
    )


def pytest_configure(config: Any) -> None:
    """Configure fasteval based on pytest options."""
    global _current_config
    _current_config = config
    # Set environment variable if --no-interactive flag is passed
    if config.getoption("--no-interactive", default=False):
        os.environ["FASTEVAL_NO_INTERACTIVE"] = "1"

    config.addinivalue_line(
        "markers", f"{FASTEVAL_MARKER}: test uses fasteval (added automatically)"
    )
    from fasteval.cli.config import load_tag_registry

    rootpath = getattr(config, "rootpath", None)
    if isinstance(rootpath, Path):
        for tag, description in load_tag_registry(rootpath).items():
            config.addinivalue_line("markers", f"{tag}: {description}")

    _configure_judge_client(config)
    _configure_judge_overrides(config)

    events_dir = config.getoption("--fe-events", default=None)
    if isinstance(events_dir, str) and events_dir:
        from fasteval.testing.events import EventWriter

        worker_id = config.workerinput["workerid"] if _is_xdist_worker(config) else None
        config.stash[_EVENTS_KEY] = EventWriter(Path(events_dir), worker_id=worker_id)


def _configure_judge_client(config: Any) -> None:
    """
    ``[tool.fasteval] judge_client`` is the project's default LLM judge, the
    same one the UI playground uses, unless code already called
    ``fe.set_default_provider(...)`` (explicit setup always wins).
    """
    rootpath = getattr(config, "rootpath", None)
    if not isinstance(rootpath, Path):
        return
    from fasteval.cli.config import ConfigError, find_config

    try:
        spec = find_config(rootpath).judge_client
    except ConfigError:
        return
    if not spec:
        return
    from fasteval.providers import registry

    if registry._default_provider is not None:
        return
    from fasteval import judges

    try:
        registry.set_default_provider(judges.load_judge_client(spec, rootpath))
    except judges.JudgeError as e:
        raise pytest.UsageError(f"[tool.fasteval] {e}") from e


def _configure_judge_overrides(config: Any) -> None:
    specs = config.getoption("--fe-judge", default=None)
    if not isinstance(specs, list):
        return
    from fasteval import judges

    overrides = {}
    root = (
        Path(config.rootpath)
        if isinstance(getattr(config, "rootpath", None), Path)
        else None
    )
    for text in specs:
        try:
            metric, judge = judges.parse_override(text)
            overrides[metric] = judges.load_judge(judge, root)
        except judges.JudgeError as e:
            raise pytest.UsageError(f"--fe-judge: {e}") from e
    judges.set_overrides(overrides)


def _events(config: Any) -> Any:
    """The run's EventWriter, or None when --fe-events wasn't given."""
    stash = getattr(config, "stash", None)
    if stash is None or not hasattr(stash, "get"):
        return None
    return stash.get(_EVENTS_KEY, None)


def pytest_sessionstart(session: Any) -> None:
    """Reset collector at the start of each test session."""
    from fasteval.collectors.collector import reset_collector

    reset_collector()

    events = _events(session.config)
    if events is not None and not _is_xdist_worker(session.config):
        from fasteval.testing.events import now_iso

        params = getattr(session.config, "invocation_params", None)
        events.emit(
            "run_start",
            started_at=now_iso(),
            pid=os.getpid(),
            rootdir=str(session.config.rootpath),
            args=list(params.args) if params else [],
            workers=_xdist_worker_count(session.config),
        )


def is_fasteval_item(item: Any) -> bool:
    """True if a collected item is a fasteval eval."""
    func = getattr(item, "function", None) or getattr(item, "obj", None)
    return func is not None and any(hasattr(func, a) for a in _FASTEVAL_ATTRS)


def pytest_generate_tests(metafunc: Any) -> None:
    """Expand @fe.dataset / @fe.csv / @fe.jsonl into one test case per row."""
    ds = getattr(metafunc.function, "_fasteval_dataset", None)
    if ds is None:
        return
    from fasteval.core.datasets import parametrize_spec

    try:
        argnames, rows = parametrize_spec(metafunc.function, ds)
    except (ValueError, TypeError, OSError) as e:
        raise pytest.UsageError(f"{metafunc.function.__name__}: {e}") from e
    _register_tags(metafunc.config, {t for _, _, tags in rows for t in tags})
    metafunc.parametrize(
        argnames,
        [
            pytest.param(
                *values, id=case_id, marks=[getattr(pytest.mark, t) for t in tags]
            )
            for values, case_id, tags in rows
        ],
    )


def _register_tags(config: Any, names: Any) -> None:
    registered = {
        line.split(":", 1)[0].split("(", 1)[0].strip()
        for line in config.getini("markers")
    }
    for name in sorted(set(names) - registered):
        config.addinivalue_line("markers", f"{name}: fasteval tag")


def _fe_tags(item: Any) -> List[str]:
    """Tags from @fe.tag on the test function and its class."""
    tags: List[str] = []
    for owner in (getattr(item, "cls", None), getattr(item, "function", None)):
        for name in getattr(owner, "_fasteval_tags", ()) or ():
            if name not in tags:
                tags.append(name)
    return tags


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """
    Mark fasteval evals and optionally deselect everything else.

    Runs before pytest's own ``-m``/``-k`` filtering so ``-m fasteval``
    and tag expressions see the marker.
    """
    keep, drop = [], []
    # Register @fe.tag names before use: no unknown-mark warning, and
    # --strict-markers accepts them.
    _register_tags(config, {name for item in items for name in _fe_tags(item)})
    for item in items:
        for name in _fe_tags(item):
            item.add_marker(name)
        if is_fasteval_item(item):
            item.add_marker(FASTEVAL_MARKER)
            keep.append(item)
        else:
            drop.append(item)

    if config.getoption("--fe-only", default=False) and drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_collection_finish(session: Any) -> None:
    """Write the eval inventory and/or the ``collected`` event if requested."""
    config = session.config
    path = config.getoption("--fe-inventory", default=None)
    events = _events(config)
    worker = _is_xdist_worker(config)
    # Under xdist every worker collects the same items; gw0 reports them.
    emit_collected = events is not None and (
        not worker or config.workerinput.get("workerid") == "gw0"
    )
    if not emit_collected and (not path or worker):
        return
    root = Path(config.rootpath)
    inventory = [_inventory_entry(item, root) for item in session.items]
    if emit_collected:
        events.emit("collected", total=len(inventory), items=inventory)
    if not path or worker:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"root": str(root), "items": inventory}, indent=2, default=str),
        encoding="utf-8",
    )


def _inventory_entry(item: Any, root: Path) -> Dict[str, Any]:
    func = getattr(item, "function", None) or getattr(item, "obj", None)
    metrics = [
        {
            "name": m.name,
            "type": m.metric_type,
            "threshold": m.threshold,
            "config": _json_safe_config(m.config),
            "model": (m.llm_config or {}).get("model"),
        }
        for m in getattr(func, "_fasteval_metrics", []) or []
    ]
    tags = sorted(
        {m.name for m in item.iter_markers() if m.name not in _NON_TAG_MARKERS}
    )
    try:
        rel_file = str(Path(item.path).relative_to(root))
    except (ValueError, AttributeError):
        rel_file = str(getattr(item, "path", item.nodeid.split("::")[0]))
    data = getattr(func, "_fasteval_data", None)
    if isinstance(data, dict) and data.get("type") == "conversation":
        data = {"type": "conversation", "turns": len(data.get("turns", []))}
    return {
        "nodeid": item.nodeid,
        "file": rel_file,
        "function": getattr(item, "originalname", None) or item.name,
        "name": item.name,
        "tags": tags,
        "metrics": metrics,
        "data": data,
        "human_review": hasattr(func, "_fasteval_human_review"),
        "skipped": any(m.name == "skip" for m in item.iter_markers()),
    }


def pytest_runtest_logstart(nodeid: str, location: Any) -> None:
    """Emit ``test_start`` (on xdist this fires on the controller too)."""
    config = _current_config
    events = _events(config) if config is not None else None
    if events is not None and not _is_xdist_worker(config):
        events.emit("test_start", nodeid=nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Any) -> Any:
    """Emit an ``eval`` event for every fe.score() result the test produced."""
    events = _events(item.config)
    if events is None:
        yield
        return
    from fasteval.collectors.collector import get_collector

    collector = get_collector()
    before = len(collector.results)
    yield
    results = collector.results[before:]
    names = collector.test_names[before:]
    for index, (result, name) in enumerate(zip(results, names)):
        events.emit(
            "eval",
            nodeid=item.nodeid,
            index=index,
            test_name=name,
            result=result.model_dump(mode="json"),
        )


def pytest_runtest_logreport(report: Any) -> None:
    """Emit ``test_end`` once per test phase that decides its outcome."""
    config = _current_config
    events = _events(config) if config is not None else None
    if events is None or _is_xdist_worker(config):
        return
    when = report.when
    if when == "call" or (when == "setup" and not report.passed):
        outcome = report.outcome
        if when == "setup" and report.failed:
            outcome = "error"
    elif when == "teardown" and report.failed:
        outcome = "error"
    else:
        return
    if hasattr(report, "wasxfail"):
        outcome = "xfailed" if report.skipped else "xpassed"
    longrepr = None
    if not report.passed and report.longrepr is not None:
        longrepr = getattr(report, "longreprtext", None) or str(report.longrepr)
    node = getattr(report, "node", None)
    worker = getattr(node, "workerinput", {}).get("workerid") if node else None
    events.emit(
        "test_end",
        nodeid=report.nodeid,
        when=when,
        outcome=outcome,
        duration=round(report.duration, 4),
        longrepr=longrepr,
        stdout=report.capstdout or None,
        stderr=report.capstderr or None,
        worker_id=worker,
    )


def _json_safe_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Metric config values that survive JSON (criteria, steps, ...)."""
    safe: Dict[str, Any] = {}
    for key, value in (config or {}).items():
        if key == "instance":
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe[key] = value
    return safe


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    """xdist controller: merge results sent back by a finished worker."""
    payload = getattr(node, "workeroutput", {}).get(_WORKER_OUTPUT_KEY)
    if not payload:
        return
    from fasteval.collectors.collector import get_collector
    from fasteval.models.evaluation import EvalResult

    collector = get_collector()
    for entry in payload:
        try:
            result = EvalResult.model_validate(entry["result"])
        except Exception:  # pragma: no cover - defensive, depends on payload
            continue
        collector.collect(result, test_name=entry["test_name"])


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Generate reports and apply quality gates after all tests complete."""
    from fasteval.collectors.collector import get_collector

    collector = get_collector()
    config = session.config

    if _is_xdist_worker(config):
        # Ship results to the controller; it does the reporting.
        config.workeroutput[_WORKER_OUTPUT_KEY] = [
            {"test_name": name, "result": result.model_dump(mode="json")}
            for result, name in zip(collector.results, collector.test_names)
        ]
        return

    gate_failures: List[str] = []
    try:
        gate_failures = _report_and_gate(session, collector)
    finally:
        _emit_run_end(session, collector, gate_failures)


def _report_and_gate(session: Any, collector: Any) -> List[str]:
    config = session.config
    min_score = config.getoption("--fe-min-score", default=None)
    min_pass_rate = config.getoption("--fe-min-pass-rate", default=None)

    if not collector.results:
        if min_score is not None or min_pass_rate is not None:
            _write_line(
                "\nFastEval: no eval results were recorded, quality gates not applied."
            )
        return []

    summary = collector.summary()

    # Print console summary if requested
    if config.getoption("--fe-summary", default=False):
        _print_console_summary(summary)

    # Process --fe-output flags
    outputs = config.getoption("--fe-output", default=[])
    for output_spec in outputs:
        if ":" in output_spec:
            fmt, path = output_spec.split(":", 1)
            collector.report(fmt.strip(), path=path.strip())
        else:
            # No path = print to stdout
            content = collector.report(output_spec.strip())
            sys.stdout.write(content + "\n")

    failures = check_gates(summary, min_score=min_score, min_pass_rate=min_pass_rate)
    for message in failures:
        _write_line(f"FastEval gate failed: {message}")
    if failures and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    return failures


def _emit_run_end(session: Any, collector: Any, gate_failures: List[str]) -> None:
    events = _events(session.config)
    if events is None:
        return
    summary = collector.summary().model_dump(mode="json")
    payload = {
        "exit_status": int(session.exitstatus),
        "summary": summary,
        "gate_failures": gate_failures,
        "gates": {
            "min_score": session.config.getoption("--fe-min-score", default=None),
            "min_pass_rate": session.config.getoption(
                "--fe-min-pass-rate", default=None
            ),
        },
    }
    events.emit("run_end", **payload)
    events.write_result(payload)


def check_gates(
    summary: Any,
    *,
    min_score: Optional[float] = None,
    min_pass_rate: Optional[float] = None,
) -> List[str]:
    """Return a message per failed quality gate."""
    failures = []
    if min_score is not None and summary.avg_aggregate_score < min_score:
        failures.append(
            f"average score {summary.avg_aggregate_score:.3f} < {min_score:.3f}"
        )
    if min_pass_rate is not None and summary.pass_rate < min_pass_rate:
        failures.append(f"pass rate {summary.pass_rate:.0%} < {min_pass_rate:.0%}")
    return failures


def pytest_unconfigure(config: Any) -> None:
    """Clean up after pytest run."""
    global _current_config
    events = _events(config)
    if events is not None:
        events.close()
    if _current_config is config:
        _current_config = None
    # Remove the environment variable if we set it
    if "FASTEVAL_NO_INTERACTIVE" in os.environ:
        if config.getoption("--no-interactive", default=False):
            del os.environ["FASTEVAL_NO_INTERACTIVE"]


def _xdist_worker_count(config: Any) -> int:
    value = getattr(config.option, "numprocesses", None)
    return value if isinstance(value, int) else 0


def _is_xdist_worker(config: Any) -> bool:
    # pytest-xdist sets ``config.workerinput`` (a dict) on worker processes.
    return isinstance(getattr(config, "workerinput", None), dict)


def _write_line(text: str) -> None:
    sys.stdout.write(text + "\n")


def _print_console_summary(summary: Any) -> None:
    """Print a compact console summary of evaluation results."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  FastEval Summary")
    print(sep)
    print(
        f"  Tests:  {summary.total_tests} total, "
        f"{summary.passed_tests} passed, "
        f"{summary.failed_tests} failed"
    )

    print(f"  Pass Rate:  {summary.pass_rate:.0%}")
    print(f"  Avg Score:  {summary.avg_aggregate_score:.3f}")
    print(f"  Time:       {summary.total_execution_time_ms:.0f}ms")

    if summary.metric_aggregates:
        print(f"\n  {'Metric':<25} {'Pass Rate':>10} {'Avg':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-' * 25} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8}")
        for m in summary.metric_aggregates:
            print(
                f"  {m.metric_name:<25} {m.pass_rate:>9.0%} "
                f"{m.avg_score:>8.3f} {m.min_score:>8.3f} {m.max_score:>8.3f}"
            )

    if summary.failed_tests > 0:
        print(f"\n  Failed tests:")
        for t in summary.test_summaries:
            if not t.passed:
                err = f" - {t.error}" if t.error else ""
                print(f"    x {t.test_name} (score: {t.aggregate_score:.3f}){err}")

    print(sep + "\n")
