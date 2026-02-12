"""Trace fetching and mapping functionality."""

from typing import Any, Dict, List, Optional

from fasteval_langfuse.client import LangfuseClient
from fasteval_langfuse.sampling.base import BaseSamplingStrategy
from fasteval_langfuse.sampling.strategies import NoSamplingStrategy
from fasteval_langfuse.utils import extract_context_from_trace, parse_time_range


class TraceFetcher:
    """
    Handles fetching traces from Langfuse and applying sampling.

    Attributes:
        client: LangfuseClient instance
    """

    def __init__(self, client: Optional[LangfuseClient] = None):
        """
        Initialize trace fetcher.

        Args:
            client: Optional LangfuseClient instance (creates default if None)
        """
        self.client = client or LangfuseClient()

    def fetch_and_sample(
        self,
        project: Optional[str] = None,
        filter_tags: Optional[List[str]] = None,
        time_range: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        sampling: Optional[BaseSamplingStrategy] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Fetch traces and apply sampling strategy.

        Args:
            project: Project name
            filter_tags: Tags to filter by
            time_range: Time range string (e.g., "last_24h")
            user_id: Filter by user ID
            session_id: Filter by session ID
            limit: Max traces to fetch before sampling
            sampling: Sampling strategy (default: NoSamplingStrategy)

        Returns:
            Tuple of (sampled_traces, total_count)
        """
        # Parse time range
        from_timestamp, to_timestamp = None, None
        if time_range:
            from_timestamp, to_timestamp = parse_time_range(time_range)

        # Fetch traces
        traces = self.client.fetch_traces(
            project=project,
            tags=filter_tags,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

        total_count = len(traces)

        # Apply sampling
        if sampling is None:
            sampling = NoSamplingStrategy()

        sampled_traces = sampling.sample(traces)

        return sampled_traces, total_count

    def map_trace_to_params(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map Langfuse trace to test function parameters.

        Extracts: trace_id, input, output, context, metadata

        Args:
            trace: Trace dictionary from Langfuse

        Returns:
            Dictionary of function parameters
        """
        # Extract context from metadata
        context = extract_context_from_trace(trace)

        # Extract input and output
        input_value = trace.get("input", "")
        if isinstance(input_value, dict):
            # If input is dict, try to get common keys
            input_value = (
                input_value.get("query")
                or input_value.get("input")
                or input_value.get("prompt")
                or str(input_value)
            )

        output_value = trace.get("output", "")
        if isinstance(output_value, dict):
            # If output is dict, try to get common keys
            output_value = (
                output_value.get("response")
                or output_value.get("output")
                or output_value.get("answer")
                or str(output_value)
            )

        return {
            "trace_id": trace["id"],
            "input": str(input_value) if input_value else "",
            "output": str(output_value) if output_value else "",
            "context": context,
            "metadata": trace.get("metadata", {}),
        }
