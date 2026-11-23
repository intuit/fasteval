"""Minimal terminal styling for the fasteval CLI (no third-party deps)."""

from __future__ import annotations

import os
import sys
from typing import TextIO

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}


def color_enabled(stream: TextIO = sys.stdout) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def style(text: str, *names: str, stream: TextIO = sys.stdout) -> str:
    if not names or not color_enabled(stream):
        return text
    codes = ";".join(_CODES[n] for n in names)
    return f"\033[{codes}m{text}\033[0m"


def error(message: str) -> None:
    sys.stderr.write(
        style("error: ", "red", "bold", stream=sys.stderr) + message + "\n"
    )


def warn(message: str) -> None:
    sys.stderr.write(style("warning: ", "yellow", stream=sys.stderr) + message + "\n")


def pct(value: float) -> str:
    return f"{value:.0%}"


def score_style(value: float, threshold: float = 0.7) -> str:
    return "green" if value >= threshold else "red"
