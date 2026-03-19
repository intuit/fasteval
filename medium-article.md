# We Got Tired of Writing the Same LLM Evaluation Code Over and Over, So We Open-Sourced Ours

## Introducing fasteval, a decorator-first library that makes LLM testing feel like regular pytest

![The evaluation journey: from non-deterministic LLM outputs to reliable engineering metrics](hero-evaluation-journey.png)

---

I need to tell you about a problem that drove us crazy for months before we finally did something about it.

Every team building with LLMs eventually hits the same wall. You ship your first agent or chatbot, it works great in demos, and then someone asks the obvious question: "How do we know this is actually good?" Not vibes-good. Measurably good. Good enough to not embarrass us in production.

That question sent us down a rabbit hole. What we found at the bottom was a mess.

## The testing problem that kept showing up

Every team we talked to was solving this the same way, and badly. Someone would write a one-off script that calls the model a hundred times and dumps results into a spreadsheet. Someone else would build a custom evaluator with hard-coded prompts for LLM-as-judge. A third person would slap together some ROUGE scores and call it a day.

None of it talked to each other. None of it ran in CI. And none of it caught the regression that shipped to production last Thursday.

Testing LLMs is fundamentally weird compared to testing normal software. You can't just assert that output equals expected. Ask a model "What's the capital of France?" and you'll get "Paris" one time and "The capital of France is Paris, a city in Western Europe" the next. Both correct. Completely different strings.

So teams end up building custom evaluation frameworks. We did too. Multiple times, actually. And each time we found ourselves writing the same patterns: prompt templates for LLM-as-judge, score parsing, threshold logic, result aggregation. Over and over, in slightly different shapes.

At some point we looked at each other and said: why are we doing this?

## What we actually wanted

We had a pretty clear picture of the ideal tool. Something that felt like pytest, not like a new platform to learn. We wanted to stack evaluation criteria the way you stack decorators: readable, composable, obvious at a glance. Both LLM-based and deterministic evaluation in the same framework, because sometimes you need semantic judgment and sometimes you just need to check if the output is valid JSON.

Nothing we found checked all those boxes. So we built it.

## How fasteval works

![How fasteval works: Decorate → Test → Score → Evaluate → Result](fasteval-overview.png)

You decorate a test function with the metrics you care about, then call `fe.score()` with your model's output.

```python
import fasteval as fe

@fe.correctness(threshold=0.8)
@fe.relevance(threshold=0.7)
def test_qa_agent():
    response = my_agent("What is the capital of France?")
    fe.score(response, expected_output="Paris", input="What is the capital of France?")
```

Run it with `pytest -v` and you get pass/fail for each metric, with scores and reasoning. No config files. No dashboard setup. No new CLI to learn.

We went with decorators because they make the evaluation criteria visible right where the test is defined. When someone new joins the team and opens the test file, they can immediately tell what quality bar each test enforces. People mention this more than anything else when they try the library, so I think we got that decision right.

## Stacking metrics is where it gets interesting

Real-world evaluation is never one-dimensional. You don't just care about correctness. You also care about relevance, whether the response is toxic, whether it follows instructions. fasteval lets you stack all of that:

```python
@fe.correctness(threshold=0.8, weight=2.0)
@fe.relevance(threshold=0.7, weight=1.0)
@fe.toxicity(threshold=0.95)
def test_customer_support_bot():
    response = support_bot("I want to cancel my subscription")
    fe.score(
        response,
        expected_output="Acknowledge the request and provide cancellation steps",
        input="I want to cancel my subscription"
    )
```

Each metric evaluates independently. Weights let you prioritize what matters most. The test fails if any metric drops below its threshold.

We've got over 30 built-in metrics at this point: correctness, hallucination, coherence, conciseness, bias, instruction following, and a bunch more. We kept adding them because every time we thought "okay that's enough," someone on the team would need one more.

## Not everything needs an LLM to evaluate

This was an important design decision. LLM-as-judge is powerful but it's slow and it costs money. For a lot of checks you genuinely don't need it. Does the output contain a required keyword? Is it valid JSON? Does it match a regex pattern? You don't need GPT-4 to tell you that.

We built deterministic metrics right into the same decorator system:

```python
from pydantic import BaseModel

class UserResponse(BaseModel):
    name: str
    age: int
    email: str

@fe.json(model=UserResponse)
def test_structured_output():
    output = my_agent("Create a user profile for Alice, age 30")
    result = fe.score(output)
    assert result.passed
```

No API key needed. Runs instantly. We use these for fast sanity checks on every commit, and save the heavier LLM evaluations for nightly runs. Having fast deterministic tests and thorough semantic tests living in the same framework, sharing the same decorator API, was something we didn't find in other tools.

There's also `@fe.exact_match`, `@fe.contains`, `@fe.rouge`, and `@fe.regex`. Mix and match with LLM metrics however you want.

## RAG evaluation was a big motivator

Half the teams we work with are building RAG pipelines, and RAG is especially tricky to test. You need to evaluate retrieval quality and generation quality at the same time. Is the model sticking to the retrieved context or making stuff up? Are the right documents being pulled in the first place?

```python
@fe.faithfulness(threshold=0.8)
@fe.contextual_precision(threshold=0.7)
def test_rag_pipeline():
    result = rag_pipeline("How does photosynthesis work?")
    fe.score(
        actual_output=result.answer,
        context=result.retrieved_docs,
        input="How does photosynthesis work?",
    )
```

Faithfulness measures whether the answer is grounded in the retrieved context. Contextual precision checks whether the retriever pulled the right documents. Throw `@fe.hallucination` on top and you've got a solid RAG evaluation suite in about ten lines of code.

We've actually seen teams catch retrieval regressions with this setup that they'd been missing for weeks. One team had a broken chunking config that degraded recall by 15%, and their existing tests never flagged it because they were only checking the final answer.

## Testing agent tool calls

If your agent is supposed to search for flights and then book one, you need to verify it actually called those tools, in the right order, with the right arguments. Doing that by hand gets old fast.

```python
@fe.tool_call_accuracy(threshold=0.9)
def test_booking_agent():
    result = agent.run("Book a flight to Paris")
    fe.score(
        actual_tools=result.tool_calls,
        expected_tools=[
            {"name": "search_flights", "args": {"destination": "Paris"}},
            {"name": "book_flight"},
        ],
    )
```

Tool name matching, argument validation, sequence verification. We also have `@fe.tool_sequence` and `@fe.tool_args_match` for more granular control.

## Testing LangGraph agents without losing your mind

![Testing pyramid for agents: unit tests, integration tests, and end-to-end evaluation layers](testing-pyramid-agents.png)

Okay, this one is close to my heart.

If you're building agents with LangGraph, you know the pain. You've got a state graph with five or six nodes, a classifier, a retriever, a responder, maybe some routing logic, and testing the whole thing end-to-end is slow, flaky, and expensive. But testing nodes individually means ripping apart the graph and manually wiring up state. Nobody wants to do that.

We built a test harness specifically for this. It wraps any compiled `StateGraph` and gives you a clean API to test the full flow, individual nodes, or anything in between.

Full conversation testing is just `.chat()`:

```python
from fasteval_langgraph import harness
import fasteval as fe

graph = harness(compiled_graph)

@fe.correctness(threshold=0.8)
async def test_support_agent():
    result = await graph.chat("How do I configure OAuth?")
    fe.score(result.response, "Use OAuth 2.0...", input="How do I configure OAuth?")
```

The harness auto-detects whether your graph uses `MessagesState` or plain `TypedDict` and sets up sensible defaults. You don't configure anything for the common case.

Where it gets really useful is node-level testing. Say you want to test just your classifier node without running the entire graph:

```python
from langchain_core.messages import HumanMessage

result = await graph.node("classifier").run(
    messages=[HumanMessage(content="What is OAuth?")]
)

assert result.updates.get("intent") == "FAQ"
assert result.goto == "rag"  # Where the classifier routes to
assert result.execution_time_ms < 500  # Performance check
```

State updates, routing decision, execution timing. All from running one node in isolation. We use this pattern constantly when iterating on individual node logic because waiting for the whole graph to execute every time just kills your feedback loop.

For conversational agents, multi-turn sessions keep state across messages:

```python
async with graph.session() as s:
    r1 = await s.chat("I need help with billing")
    r2 = await s.chat("Actually, make that a refund")

    # State persists across the session
    assert r2.state["call_count"] == 2
    assert len(s.history) == 2
```

And mocking. If you need to test a node without its dependencies hitting real APIs:

```python
from fasteval_langgraph import mock

with graph.mocked(
    mock("rag").updates({"docs": ["fake retrieval result"]}).goto("responder"),
):
    result = await graph.chat("What is OAuth?")
    # RAG node is mocked, everything else runs normally
```

Mocks auto-restore when the context manager exits. No cleanup code.

One more thing worth mentioning: the harness captures a full execution trace. Which nodes ran, in what order, what each one produced, how long each took. When a test fails and you're trying to figure out which node in the graph screwed up, that trace is the first thing you'll reach for.

Install it separately with `pip install fasteval-langgraph`.

## Production monitoring and the other plugins

Building fasteval as a standalone library was step one. But we knew people would need it in production too, not just in test suites.

**fasteval-langfuse** lets you evaluate production traces from Langfuse. Pull traces, run them through your metrics, push scores back. We added smart sampling so you're not burning through API credits evaluating every single request. You can sample by percentage, token budget, or adaptive strategies.

**fasteval-observe** does async runtime monitoring with configurable sampling. Lightweight way to keep an eye on quality in production without adding latency.

Each plugin is a separate pip install, so you only pull in what you need.

## Getting started

```bash
pip install fasteval-core
export OPENAI_API_KEY=sk-your-key-here
```

Write a test file:

```python
import fasteval as fe

@fe.correctness(threshold=0.8)
def test_my_llm():
    response = call_your_model("What is 2 + 2?")
    fe.score(response, expected_output="4", input="What is 2 + 2?")
```

Run it:

```bash
pytest test_my_llm.py -v
```

Want HTML reports? `--fe-output=report.html`. Aggregate statistics? `--fe-summary`. Anthropic instead of OpenAI? Set `ANTHROPIC_API_KEY`. Local Ollama model? Also works.

We support Python 3.10 through 3.14. Apache 2.0 licensed.

## Why we open-sourced it

Because we kept seeing the same pain everywhere. Every team we talked to was reinventing evaluation from scratch. Smart people spending weeks on infrastructure that already existed, except everyone's version was trapped inside their own codebase and nobody could benefit from each other's work.

We figured open-sourcing ours would save people time. And honestly, we wanted the feedback. The library has gotten meaningfully better since we opened it up because people file issues about use cases we never thought of.

The repo is active and we ship regularly. If you try it and something's broken or missing, open an issue. We read them.

---

**GitHub**: [github.com/intuit/fasteval](https://github.com/intuit/fasteval) | **Docs**: [fasteval.io](https://fasteval.io) | **Install**: `pip install fasteval-core`
