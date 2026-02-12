# fasteval-langgraph

LangGraph testing plugin for [fasteval](https://github.com/intuit/fasteval). Wraps any compiled `StateGraph` with a minimal test harness for chat, session, mocking, and node-level testing.

## Install

```bash
pip install fasteval-langgraph
```

## Quick Start

```python
from fasteval_langgraph import harness, mock
import fasteval as fe

graph = harness(compiled_graph)

@fe.correctness(threshold=0.8)
async def test_agent():
    result = await graph.chat("How do I configure OAuth?")
    fe.score(result.response, "Use OAuth 2.0...", input="How do I configure OAuth?")
```

See the [full documentation](../../docs/plugins/langgraph/) for details.
