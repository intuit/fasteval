"""HTML output reporter."""

from __future__ import annotations

from html import escape
from typing import List

from fasteval.collectors.reporters.base import OutputReporter
from fasteval.collectors.summary import EvalRunSummary
from fasteval.models.evaluation import EvalResult

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 24px; }
h1 { font-size: 24px; margin-bottom: 16px; }
h2 { font-size: 18px; margin: 24px 0 12px; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.card { background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; }
.card .label { font-size: 12px; text-transform: uppercase; color: #888; margin-bottom: 4px; }
.card .value { font-size: 28px; font-weight: 600; }
.card .value.pass { color: #16a34a; }
.card .value.fail { color: #dc2626; }
.card .value.neutral { color: #2563eb; }
table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }
th { background: #f8f8f8; text-align: left; padding: 10px 14px; font-size: 13px; text-transform: uppercase; color: #666; border-bottom: 2px solid #eee; }
td { padding: 10px 14px; border-bottom: 1px solid #eee; font-size: 14px; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.badge.pass { background: #dcfce7; color: #16a34a; }
.badge.fail { background: #fee2e2; color: #dc2626; }
details { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 8px; }
details summary { padding: 12px 16px; cursor: pointer; font-size: 14px; }
details summary:hover { background: #f8f8f8; }
details .detail-body { padding: 0 16px 16px; }
.reasoning { background: #f8f8f8; padding: 8px 12px; border-radius: 4px; font-size: 13px; margin-top: 6px; white-space: pre-wrap; }
.bar { height: 8px; border-radius: 4px; background: #eee; margin-top: 4px; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-fill.pass { background: #16a34a; }
.bar-fill.fail { background: #dc2626; }
.timestamp { font-size: 12px; color: #999; margin-top: 24px; }
"""


class HtmlReporter(OutputReporter):
    """Generates a self-contained HTML report with inline CSS."""

    def generate(
        self, summary: EvalRunSummary, results: List[EvalResult]
    ) -> str:
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            "<title>FastEval Report</title>",
            f"<style>{_CSS}</style>",
            "</head><body>",
            "<h1>FastEval Evaluation Report</h1>",
            self._render_cards(summary),
            "<h2>Metric Breakdown</h2>",
            self._render_metric_table(summary),
            "<h2>Test Results</h2>",
            self._render_test_details(summary, results),
            f'<p class="timestamp">Generated: {escape(summary.timestamp)}</p>',
            "</body></html>",
        ]
        return "\n".join(parts)

    def _render_cards(self, summary: EvalRunSummary) -> str:
        pass_rate_pct = f"{summary.pass_rate:.0%}"
        avg_score = f"{summary.avg_aggregate_score:.2f}"
        time_s = f"{summary.total_execution_time_ms / 1000:.1f}s"
        return f"""<div class="cards">
<div class="card"><div class="label">Total Tests</div><div class="value neutral">{summary.total_tests}</div></div>
<div class="card"><div class="label">Passed</div><div class="value pass">{summary.passed_tests}</div></div>
<div class="card"><div class="label">Failed</div><div class="value fail">{summary.failed_tests}</div></div>
<div class="card"><div class="label">Pass Rate</div><div class="value {'pass' if summary.pass_rate >= 0.5 else 'fail'}">{pass_rate_pct}</div></div>
<div class="card"><div class="label">Avg Score</div><div class="value neutral">{avg_score}</div></div>
<div class="card"><div class="label">Total Time</div><div class="value neutral">{time_s}</div></div>
</div>"""

    def _render_metric_table(self, summary: EvalRunSummary) -> str:
        if not summary.metric_aggregates:
            return "<p>No metrics recorded.</p>"
        rows = []
        for m in summary.metric_aggregates:
            badge_cls = "pass" if m.pass_rate >= 0.5 else "fail"
            bar_pct = f"{m.pass_rate * 100:.0f}%"
            rows.append(
                f"<tr><td>{escape(m.metric_name)}</td>"
                f"<td>{m.count}</td>"
                f'<td><span class="badge {badge_cls}">{m.pass_rate:.0%}</span></td>'
                f"<td>{m.avg_score:.3f}</td>"
                f"<td>{m.min_score:.3f}</td>"
                f"<td>{m.max_score:.3f}</td>"
                f'<td><div class="bar"><div class="bar-fill {badge_cls}" style="width:{bar_pct}"></div></div></td>'
                f"</tr>"
            )
        return f"""<table>
<thead><tr><th>Metric</th><th>Count</th><th>Pass Rate</th><th>Avg</th><th>Min</th><th>Max</th><th>Distribution</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>"""

    def _render_test_details(
        self, summary: EvalRunSummary, results: List[EvalResult]
    ) -> str:
        if not results:
            return "<p>No test results.</p>"
        parts = []
        for i, result in enumerate(results):
            test_name = (
                summary.test_summaries[i].test_name
                if i < len(summary.test_summaries)
                else "unknown"
            )
            badge = "pass" if result.passed else "fail"
            label = "PASS" if result.passed else "FAIL"
            metric_rows = []
            for mr in result.metric_results:
                m_badge = "pass" if mr.passed else "fail"
                reasoning_html = ""
                if mr.reasoning:
                    reasoning_html = (
                        f'<div class="reasoning">{escape(mr.reasoning)}</div>'
                    )
                metric_rows.append(
                    f"<tr><td>{escape(mr.metric_name)}</td>"
                    f"<td>{mr.score:.3f}</td>"
                    f"<td>{mr.threshold:.2f}</td>"
                    f'<td><span class="badge {m_badge}">{"PASS" if mr.passed else "FAIL"}</span></td>'
                    f"<td>{reasoning_html}</td></tr>"
                )
            metrics_table = f"""<table>
<thead><tr><th>Metric</th><th>Score</th><th>Threshold</th><th>Status</th><th>Reasoning</th></tr></thead>
<tbody>{"".join(metric_rows)}</tbody>
</table>""" if metric_rows else "<p>No metrics.</p>"

            parts.append(
                f'<details><summary><span class="badge {badge}">{label}</span> '
                f"{escape(test_name)} "
                f"(score: {result.aggregate_score:.3f}, "
                f"{result.execution_time_ms:.0f}ms)</summary>"
                f'<div class="detail-body">{metrics_table}</div></details>'
            )
        return "\n".join(parts)
