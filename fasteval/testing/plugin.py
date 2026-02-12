"""
FastEval pytest plugin.

Provides pytest integration for fasteval, including:
- --no-interactive flag to skip human review prompts in CI/CD
"""

import os
from typing import Any


def pytest_addoption(parser: Any) -> None:
    """Add fasteval-specific command line options to pytest."""
    parser.addoption(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Skip human review prompts (for CI/CD pipelines)",
    )


def pytest_configure(config: Any) -> None:
    """Configure fasteval based on pytest options."""
    # Set environment variable if --no-interactive flag is passed
    if config.getoption("--no-interactive", default=False):
        os.environ["FASTEVAL_NO_INTERACTIVE"] = "1"


def pytest_unconfigure(config: Any) -> None:
    """Clean up after pytest run."""
    # Remove the environment variable if we set it
    if "FASTEVAL_NO_INTERACTIVE" in os.environ:
        if config.getoption("--no-interactive", default=False):
            del os.environ["FASTEVAL_NO_INTERACTIVE"]
