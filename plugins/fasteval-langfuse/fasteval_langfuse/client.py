"""Langfuse SDK wrapper for fasteval integration."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fasteval_langfuse.config import get_config

if TYPE_CHECKING:
    from langfuse import Langfuse


class LangfuseClient:
    """
    Wrapper around Langfuse SDK for trace fetching and score reporting.

    Handles authentication, API calls, and error handling.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize Langfuse client.

        Args:
            public_key: Override config public key
            secret_key: Override config secret key
            host: Override config host

        Raises:
            ValueError: If credentials are missing
        """
        config = get_config()

        self.public_key = public_key or config.public_key
        self.secret_key = secret_key or config.secret_key
        self.host = host or config.host

        if not self.public_key or not self.secret_key:
            raise ValueError(
                "Langfuse credentials required. Set via config or environment variables "
                "(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)"
            )

        # Lazy import to avoid dependency issues in tests
        try:
            from langfuse import Langfuse
        except ImportError as e:
            raise ImportError(
                "Langfuse SDK not installed. Install with: pip install langfuse>=2.0.0"
            ) from e

        self._client = Langfuse(
            public_key=self.public_key,
            secret_key=self.secret_key,
            host=self.host,
        )

    def fetch_traces(
        self,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch traces from Langfuse.

        Args:
            project: Project name to filter by
            tags: List of tags to filter by
            from_timestamp: Start timestamp (ISO 8601)
            to_timestamp: End timestamp (ISO 8601)
            user_id: Filter by user ID
            session_id: Filter by session ID
            limit: Maximum number of traces to fetch

        Returns:
            List of trace dictionaries
        """
        # Build filter parameters
        filters = {}
        if tags:
            filters["tags"] = tags
        if user_id:
            filters["user_id"] = user_id
        if session_id:
            filters["session_id"] = session_id

        # Fetch traces using Langfuse SDK
        traces = self._client.fetch_traces(
            name=project,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            limit=limit,
            **filters,
        )

        # Convert to dict format
        return [self._trace_to_dict(trace) for trace in traces.data]

    def push_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """
        Push an evaluation score to a Langfuse trace.

        Args:
            trace_id: Langfuse trace ID
            name: Score name
            value: Score value (0.0-1.0)
            comment: Optional score comment/reasoning
        """
        self._client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )

    def fetch_dataset(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch dataset items from Langfuse.

        Args:
            name: Dataset name
            version: Optional dataset version

        Returns:
            List of dataset item dictionaries
        """
        dataset = self._client.get_dataset(name)

        # Filter by version if specified
        items = dataset.items
        if version:
            items = [item for item in items if item.version == version]

        return [self._dataset_item_to_dict(item) for item in items]

    def _trace_to_dict(self, trace: Any) -> Dict[str, Any]:
        """Convert Langfuse trace object to dictionary."""
        return {
            "id": trace.id,
            "timestamp": trace.timestamp,
            "name": trace.name,
            "user_id": getattr(trace, "user_id", None),
            "session_id": getattr(trace, "session_id", None),
            "tags": getattr(trace, "tags", []),
            "metadata": getattr(trace, "metadata", {}),
            "input": getattr(trace, "input", None),
            "output": getattr(trace, "output", None),
            "scores": getattr(trace, "scores", []),
        }

    def _dataset_item_to_dict(self, item: Any) -> Dict[str, Any]:
        """Convert Langfuse dataset item to dictionary."""
        return {
            "id": item.id,
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": getattr(item, "metadata", {}),
        }

    def flush(self) -> None:
        """Flush pending scores to Langfuse."""
        self._client.flush()
