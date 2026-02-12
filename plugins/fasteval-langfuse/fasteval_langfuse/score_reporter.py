"""Score reporting to Langfuse."""

from typing import Any, Dict, Optional

from fasteval_langfuse.client import LangfuseClient
from fasteval_langfuse.config import get_config


class ScoreReporter:
    """
    Handles pushing evaluation scores back to Langfuse.

    Attributes:
        client: LangfuseClient instance
    """

    def __init__(self, client: Optional[LangfuseClient] = None):
        """
        Initialize score reporter.

        Args:
            client: Optional LangfuseClient instance (creates default if None)
        """
        self.client = client or LangfuseClient()
        self.config = get_config()

    def push_evaluation_result(
        self,
        trace_id: str,
        metric_results: list,
        aggregate_score: float,
    ) -> None:
        """
        Push evaluation results to Langfuse trace.

        Creates scores for each metric and an aggregate score.

        Args:
            trace_id: Langfuse trace ID
            metric_results: List of metric result objects
            aggregate_score: Overall aggregate score
        """
        if not self.config.auto_push_scores:
            return

        # Push individual metric scores
        for metric_result in metric_results:
            score_name = f"{self.config.score_name_prefix}{metric_result.metric_name}"
            self.client.push_score(
                trace_id=trace_id,
                name=score_name,
                value=metric_result.score,
                comment=getattr(metric_result, "reasoning", None),
            )

        # Push aggregate score
        self.client.push_score(
            trace_id=trace_id,
            name=f"{self.config.score_name_prefix}aggregate",
            value=aggregate_score,
        )

    def flush(self) -> None:
        """Flush pending scores to Langfuse."""
        self.client.flush()
