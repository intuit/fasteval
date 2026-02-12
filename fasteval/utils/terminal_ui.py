"""Terminal UI utilities for human-in-the-loop evaluation."""

import os
import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HumanScore(BaseModel):
    """Human reviewer score."""

    score: float = Field(ge=0.0, le=1.0)  # Normalized 0-1
    raw_input: str  # Original input (e.g., "4", "pass", "p")
    skipped: bool = False


# Box drawing characters
BOX_TOP_LEFT = "┌"
BOX_TOP_RIGHT = "┐"
BOX_BOTTOM_LEFT = "└"
BOX_BOTTOM_RIGHT = "┘"
BOX_HORIZONTAL = "─"
BOX_VERTICAL = "│"
BOX_SEPARATOR = "├"
BOX_SEPARATOR_RIGHT = "┤"
SEPARATOR_DOUBLE = "═"


def is_interactive() -> bool:
    """
    Check if we're in an interactive terminal.

    Returns False if:
    - --no-interactive flag is set (checked via env var)
    - FASTEVAL_NO_INTERACTIVE env var is set
    - Not running in a TTY
    """
    # Check environment variable (set by pytest flag or manually)
    if os.getenv("FASTEVAL_NO_INTERACTIVE", "").lower() in ("1", "true", "yes"):
        return False

    # Check if stdin is a TTY
    if not sys.stdin.isatty():
        return False

    return True


def render_separator(width: int = 80, char: str = SEPARATOR_DOUBLE) -> str:
    """Render a horizontal separator line."""
    return char * width


def render_box_line(content: str, width: int = 78) -> str:
    """Render a line inside a box, padded to width."""
    # Truncate if too long
    if len(content) > width - 4:
        content = content[: width - 7] + "..."
    padded = content.ljust(width - 4)
    return f"{BOX_VERTICAL} {padded} {BOX_VERTICAL}"


def render_conversation_history(
    history: Optional[List[Dict[str, str]]],
    current_input: Optional[str] = None,
    current_expected: Optional[str] = None,
    current_actual: Optional[str] = None,
    width: int = 80,
) -> str:
    """
    Render conversation history with box drawing.

    Args:
        history: List of {"role": "user"|"assistant", "content": "..."}
        current_input: Current turn's user input
        current_expected: Expected output for current turn
        current_actual: Actual output for current turn
        width: Terminal width

    Returns:
        Formatted string with box-drawn conversation
    """
    lines = []
    inner_width = width - 2  # Account for box borders

    # Top border
    lines.append(f"{BOX_TOP_LEFT}{BOX_HORIZONTAL * (width - 2)}{BOX_TOP_RIGHT}")

    # Render history turns
    if history:
        turn_num = 1
        i = 0
        while i < len(history):
            # Get user message
            user_msg = history[i] if i < len(history) else None
            # Get assistant message (next one)
            assistant_msg = history[i + 1] if i + 1 < len(history) else None

            if user_msg and user_msg.get("role") == "user":
                lines.append(render_box_line(f"[Turn {turn_num}]", width))
                lines.append(
                    render_box_line(
                        f"  User:      \"{user_msg.get('content', '')}\"", width
                    )
                )

                if assistant_msg and assistant_msg.get("role") == "assistant":
                    lines.append(
                        render_box_line(
                            f"  Assistant: \"{assistant_msg.get('content', '')}\"",
                            width,
                        )
                    )
                    i += 2
                else:
                    i += 1

                turn_num += 1

                # Add separator if not last and there's a current turn
                if i < len(history) or current_input:
                    lines.append(
                        f"{BOX_SEPARATOR}{BOX_HORIZONTAL * (width - 2)}{BOX_SEPARATOR_RIGHT}"
                    )
            else:
                i += 1

    # Render current turn
    if current_input is not None:
        turn_num = (len(history) // 2 + 1) if history else 1
        lines.append(render_box_line(f"[Turn {turn_num}] ← Current", width))
        lines.append(render_box_line(f'  User:      "{current_input}"', width))
        if current_expected:
            lines.append(render_box_line(f'  Expected:  "{current_expected}"', width))
        if current_actual:
            lines.append(render_box_line(f'  Actual:    "{current_actual}"', width))

    # Bottom border
    lines.append(f"{BOX_BOTTOM_LEFT}{BOX_HORIZONTAL * (width - 2)}{BOX_BOTTOM_RIGHT}")

    return "\n".join(lines)


def render_metrics_summary(
    metric_results: List[Any],  # List[MetricResult]
) -> str:
    """Render auto-metric results summary."""
    if not metric_results:
        return ""

    lines = ["Auto-metrics:"]

    for mr in metric_results:
        status_icon = "✓" if mr.passed else "✗"
        name = mr.metric_name.ljust(20)
        score_str = f"{mr.score:.2f} / {mr.threshold:.2f}"
        lines.append(f"  {status_icon} {name} {score_str}")

    return "\n".join(lines)


def render_human_review(
    prompt: str,
    input_text: Optional[str] = None,
    expected: Optional[str] = None,
    actual: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    metric_results: Optional[List[Any]] = None,
    width: int = 80,
) -> str:
    """
    Render the complete human review prompt.

    Args:
        prompt: The question to ask the reviewer
        input_text: The input/query
        expected: Expected output
        actual: Actual output
        history: Conversation history for multi-turn
        metric_results: Auto-metric results to display
        width: Terminal width

    Returns:
        Formatted string for terminal display
    """
    lines = []

    # Header
    lines.append("")
    lines.append(render_separator(width))
    lines.append("HUMAN REVIEW REQUESTED")
    lines.append(render_separator(width))
    lines.append("")

    # Conversation history or single-turn display
    if history or input_text:
        lines.append("Conversation History:" if history else "Evaluation Context:")
        lines.append(
            render_conversation_history(
                history=history,
                current_input=input_text,
                current_expected=expected,
                current_actual=actual,
                width=width,
            )
        )
        lines.append("")

    # Auto-metrics summary
    if metric_results:
        lines.append(render_metrics_summary(metric_results))
        lines.append("")

    # Review prompt
    lines.append(f"Question: {prompt}")

    return "\n".join(lines)


def get_human_score(
    prompt: str = "Score [1-5] or [p]ass/[f]ail/[s]kip: ",
) -> Optional[HumanScore]:
    """
    Prompt user for score input.

    Accepts:
    - Numbers 1-5 (normalized to 0.2, 0.4, 0.6, 0.8, 1.0)
    - 'p' or 'pass' (score = 1.0)
    - 'f' or 'fail' (score = 0.0)
    - 's' or 'skip' (returns HumanScore with skipped=True)

    Returns:
        HumanScore object, or None if input is invalid
    """
    if not is_interactive():
        return HumanScore(score=0.0, raw_input="skipped", skipped=True)

    try:
        user_input = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return HumanScore(score=0.0, raw_input="interrupted", skipped=True)

    if not user_input:
        return None

    # Handle skip
    if user_input in ("s", "skip"):
        return HumanScore(score=0.0, raw_input=user_input, skipped=True)

    # Handle pass/fail
    if user_input in ("p", "pass"):
        return HumanScore(score=1.0, raw_input=user_input, skipped=False)
    if user_input in ("f", "fail"):
        return HumanScore(score=0.0, raw_input=user_input, skipped=False)

    # Handle numeric 1-5
    try:
        num = int(user_input)
        if 1 <= num <= 5:
            # Normalize to 0-1 scale
            normalized = (num - 1) / 4.0  # 1->0, 2->0.25, 3->0.5, 4->0.75, 5->1.0
            return HumanScore(score=normalized, raw_input=user_input, skipped=False)
    except ValueError:
        pass

    # Handle decimal 0.0-1.0
    try:
        score = float(user_input)
        if 0.0 <= score <= 1.0:
            return HumanScore(score=score, raw_input=user_input, skipped=False)
    except ValueError:
        pass

    # Invalid input
    return None


def prompt_human_review(
    prompt: str,
    input_text: Optional[str] = None,
    expected: Optional[str] = None,
    actual: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    metric_results: Optional[List[Any]] = None,
    width: int = 80,
    max_retries: int = 3,
) -> HumanScore:
    """
    Display human review UI and get score from user.

    This is the main entry point for human review.

    Args:
        prompt: Question to ask the reviewer
        input_text: The input/query
        expected: Expected output
        actual: Actual output
        history: Conversation history
        metric_results: Auto-metric results
        width: Terminal width
        max_retries: Max attempts for invalid input

    Returns:
        HumanScore with the reviewer's score
    """
    # Check if interactive mode is disabled
    if not is_interactive():
        return HumanScore(score=0.0, raw_input="auto-skipped", skipped=True)

    # Render and print the review UI
    ui_text = render_human_review(
        prompt=prompt,
        input_text=input_text,
        expected=expected,
        actual=actual,
        history=history,
        metric_results=metric_results,
        width=width,
    )
    print(ui_text)

    # Get score with retries for invalid input
    for attempt in range(max_retries):
        score = get_human_score()
        if score is not None:
            print("")  # Blank line after input
            return score

        if attempt < max_retries - 1:
            print("Invalid input. Use 1-5, p/pass, f/fail, or s/skip.")

    # After max retries, skip
    print("Max retries reached, skipping...")
    return HumanScore(score=0.0, raw_input="max-retries", skipped=True)
