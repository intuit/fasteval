"""Base class for sampling strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseSamplingStrategy(ABC):
    """
    Base class for sampling strategies.

    Extend this class to create custom sampling strategies for the @observe decorator.
    Implement the should_sample() method to define your sampling logic.

    Example:
        class MyCustomStrategy(BaseSamplingStrategy):
            def __init__(self, sample_production_only: bool = True):
                self.sample_production_only = sample_production_only

            def should_sample(self, function_name, args, kwargs, context):
                if self.sample_production_only:
                    return os.getenv("ENV") == "production"
                return True

        @observe(sampling=MyCustomStrategy())
        async def my_agent(query: str) -> str:
            return await process(query)
    """

    @abstractmethod
    def should_sample(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        context: Dict[str, Any],
    ) -> bool:
        """
        Decide whether to sample this invocation.

        This method is called BEFORE the function executes. Return True to sample
        (collect metrics), or False to skip sampling for this call.

        Args:
            function_name: Name of the decorated function
            args: Positional arguments passed to the function
            kwargs: Keyword arguments passed to the function
            context: Additional context dictionary containing:
                - trace_id: Optional trace ID for distributed tracing
                - span_id: Optional span ID
                - metadata: User-provided metadata
                - function_name: Name of the function
                - module: Module containing the function

        Returns:
            True to sample this invocation, False to skip
        """
        pass

    def on_completion(
        self,
        function_name: str,
        execution_time_ms: float,
        error: Optional[Exception],
        result: Any,
    ) -> None:
        """
        Called after function execution completes (for adaptive strategies).

        Override this method to track execution statistics for adaptive
        sampling decisions. This is called regardless of whether the call
        was sampled.

        Args:
            function_name: Name of the function that completed
            execution_time_ms: Execution time in milliseconds
            error: Exception if the function raised one, None otherwise
            result: Return value of the function (None if error)
        """
        pass

    def reset(self) -> None:
        """
        Reset the strategy's internal state.

        Override this method if your strategy maintains state that should
        be periodically reset (e.g., counters, sliding windows).
        """
        pass

    @property
    def name(self) -> str:
        """
        Return the strategy's name for logging/metrics.

        Returns:
            Strategy class name by default
        """
        return self.__class__.__name__
