"""
FastEval pytest plugin.

Provides pytest integration for fasteval, including:
- --no-interactive flag to skip human review prompts in CI/CD
- --fe-output flag to export evaluation results (json, html)
- --fe-summary flag to print a console summary after the run
"""

import os
import sys
from typing import Any


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


def pytest_configure(config: Any) -> None:
    """Configure fasteval based on pytest options."""
    # Set environment variable if --no-interactive flag is passed
    if config.getoption("--no-interactive", default=False):
        os.environ["FASTEVAL_NO_INTERACTIVE"] = "1"


def pytest_sessionstart(session: Any) -> None:
    """Reset collector at the start of each test session."""
    from fasteval.collectors.collector import reset_collector

    reset_collector()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Generate reports after all tests complete."""
    from fasteval.collectors.collector import get_collector

    collector = get_collector()

    if not collector.results:
        return

    # Print console summary if requested
    if session.config.getoption("--fe-summary", default=False):
        summary = collector.summary()
        _print_console_summary(summary)

    # Process --fe-output flags
    outputs = session.config.getoption("--fe-output", default=[])
    for output_spec in outputs:
        if ":" in output_spec:
            fmt, path = output_spec.split(":", 1)
            collector.report(fmt.strip(), path=path.strip())
        else:
            # No path = print to stdout
            content = collector.report(output_spec.strip())
            sys.stdout.write(content + "\n")


def pytest_unconfigure(config: Any) -> None:
    """Clean up after pytest run."""
    # Remove the environment variable if we set it
    if "FASTEVAL_NO_INTERACTIVE" in os.environ:
        if config.getoption("--no-interactive", default=False):
            del os.environ["FASTEVAL_NO_INTERACTIVE"]


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
