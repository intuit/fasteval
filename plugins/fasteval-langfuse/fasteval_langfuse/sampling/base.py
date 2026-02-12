"""Base class for trace sampling strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSamplingStrategy(ABC):
    """
    Base class for trace sampling strategies.

    Unlike fasteval-observe which samples in real-time, Langfuse sampling
    operates on batches of already-fetched traces from Langfuse.

    Example:
        class CustomStrategy(BaseSamplingStrategy):
            def sample(self, traces):
                # Custom sampling logic
                return [t for t in traces if self.should_include(t)]

            def should_include(self, trace):
                return trace.get("metadata", {}).get("priority") == "high"
    """

    @abstractmethod
    def sample(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sample from a list of traces.

        Args:
            traces: List of trace dictionaries from Langfuse

        Returns:
            Sampled subset of traces
        """
        pass

    @property
    def name(self) -> str:
        """
        Strategy name for reporting.

        Returns:
            Strategy class name by default
        """
        return self.__class__.__name__
