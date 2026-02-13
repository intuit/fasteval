# fasteval-observe

Runtime monitoring plugin for [fasteval](https://github.com/intuit/fasteval) with async sampling and structured logging.

## Installation

```bash
pip install fasteval-core fasteval-observe
```

## Quick Start

```python
import fasteval as fe
from fasteval_observe import observe
from fasteval_observe.sampling import FixedRateSamplingStrategy

# Monitor 5% of agent calls with structured logging
@observe(sampling=FixedRateSamplingStrategy(rate=0.05))
async def my_agent(query: str) -> str:
    response = await llm.invoke(query)
    return response

# Combine with fasteval evaluation metrics
@observe(
    sampling=FixedRateSamplingStrategy(rate=0.05),
    run_evaluations=True  # Run @fe.* metrics in background
)
@fe.correctness(threshold=0.8)
async def evaluated_agent(query: str) -> str:
    response = await llm.invoke(query)
    return response
```

## Sampling Strategies

### Built-in Strategies

- **NoSamplingStrategy** - Sample every call (development/debugging)
- **FixedRateSamplingStrategy** - Fixed rate sampling (e.g., 1 in N calls)
- **ProbabilisticSamplingStrategy** - Random probability-based sampling
- **AdaptiveSamplingStrategy** - Adjust rate based on latency/errors
- **TokenBudgetSamplingStrategy** - Sample based on token cost
- **ComposableSamplingStrategy** - Combine multiple strategies

### Custom Strategies

```python
from fasteval_observe.sampling import BaseSamplingStrategy

class MyCustomStrategy(BaseSamplingStrategy):
    def should_sample(self, function_name, args, kwargs, context):
        # Custom logic
        return True

@observe(sampling=MyCustomStrategy())
async def my_agent(query: str) -> str:
    return await process(query)
```

## Configuration

```python
from fasteval_observe import configure_observe, ObserveConfig, set_logger
import logging

configure_observe(ObserveConfig(
    enabled=True,
    flush_interval_seconds=5.0,
    max_queue_size=10000,
    include_inputs=False,  # Privacy: don't log inputs
    include_outputs=False,
))

# Optional: Use custom logger for file output, cloud logging, etc.
my_logger = logging.getLogger("my_app.observations")
my_logger.addHandler(logging.FileHandler("/var/log/observations.jsonl"))
set_logger(my_logger)
```

## Structured Logging

Observations are logged as JSON for easy collection by Kubernetes sidecars, Fluentd, etc:

```json
{
  "timestamp": "2026-02-03T10:15:30.123Z",
  "source": "fasteval_observe",
  "function_name": "my_agent",
  "execution_time_ms": 1234.56,
  "sampling_strategy": "FixedRateSamplingStrategy",
  "metrics": {
    "latency_ms": 1234.56,
    "success": true
  }
}
```

## License

MIT
