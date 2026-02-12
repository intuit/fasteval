"""Output formatting utilities for evaluation reports."""

from typing import List, Optional

from fasteval.models.evaluation import EvalInput, EvalResult
from fasteval.utils.text import truncate


def format_evaluation_report(
    test_name: str,
    results: List[EvalResult],
    eval_inputs: Optional[List[EvalInput]] = None,
    width: int = 80,
) -> str:
    """
    Format evaluation results as a rich block report.

    Shows all metrics (pass/fail) with reasoning for failures,
    plus input/expected/actual context.

    Args:
        test_name: Name of the test that was evaluated
        results: List of EvalResult objects from the evaluation
        eval_inputs: Optional list of EvalInput objects for context display
        width: Width of the separator lines (default: 80)

    Returns:
        Formatted report string

    Example output:
        ════════════════════════════════════════════════════════════════════════════════
        FASTEVAL EVALUATION FAILED
        ════════════════════════════════════════════════════════════════════════════════

        Test: test_my_function
        Overall: FAILED (1/3 metrics passed)

        Metrics:
          ✗ correctness    0.10 / 0.70
            └─ The response doesn't match...

          ✓ relevance      0.85 / 0.70

        Input:    "What is the answer?"
        Expected: "42"
        Actual:   "I don't know"

        ════════════════════════════════════════════════════════════════════════════════
    """
    lines = []
    separator = "═" * width

    # Header
    lines.append("")
    lines.append(separator)
    lines.append("FASTEVAL EVALUATION FAILED")
    lines.append(separator)
    lines.append("")

    # Test name
    lines.append(f"Test: {test_name}")

    # Aggregate pass/fail counts across all results
    total_passed = 0
    total_metrics = 0
    all_metric_results = []

    for result in results:
        for mr in result.metric_results:
            total_metrics += 1
            if mr.passed:
                total_passed += 1
            all_metric_results.append(mr)

    lines.append(f"Overall: FAILED ({total_passed}/{total_metrics} metrics passed)")
    lines.append("")

    # Metrics section
    lines.append("Metrics:")

    # Find max metric name length for alignment
    max_name_len = (
        max(len(mr.metric_name) for mr in all_metric_results)
        if all_metric_results
        else 15
    )

    for mr in all_metric_results:
        status_icon = "✓" if mr.passed else "✗"
        name_padded = mr.metric_name.ljust(max_name_len)
        score_str = f"{mr.score:.2f} / {mr.threshold:.2f}"

        lines.append(f"  {status_icon} {name_padded}  {score_str}")

        # Show full reasoning for failed metrics
        if not mr.passed and mr.reasoning:
            # Normalize whitespace but keep full reasoning
            reasoning_clean = mr.reasoning.replace("\n", " ").strip()
            lines.append(f"    └─ {reasoning_clean}")

        lines.append("")  # Blank line between metrics

    # Show input/expected/actual from first eval_input if available
    if eval_inputs and len(eval_inputs) > 0:
        first_input = eval_inputs[0]

        if first_input.input:
            lines.append(f'Input:    "{truncate(first_input.input, 60)}"')
        if first_input.expected_output:
            lines.append(f'Expected: "{truncate(first_input.expected_output, 60)}"')
        if first_input.actual_output:
            lines.append(f'Actual:   "{truncate(first_input.actual_output, 60)}"')

        lines.append("")

    lines.append(separator)

    return "\n".join(lines)
