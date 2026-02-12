"""Conversation/multi-turn evaluation metrics.

Based on multi-turn dialogue evaluation research including:
- Knowledge-grounded conversation evaluation (Dinan et al., 2019)
- Dialogue coherence assessment (Dziri et al., 2019)
- Multi-turn consistency checking (Welleck et al., 2019)
"""

from typing import Any, Optional

from fasteval.metrics.llm import BaseLLMMetric
from fasteval.models.evaluation import EvalInput


class ContextRetentionMetric(BaseLLMMetric):
    """
    Evaluates if the response demonstrates understanding of conversation history.

    Based on knowledge retention research in dialogue systems and
    memory-augmented neural networks evaluation methodology.
    Measures explicit and implicit context utilization.

    Use with @conversation decorator for multi-turn tests.

    Example:
        @fe.context_retention(threshold=0.8)
        @fe.conversation([
            {"query": "My name is Alice"},
            {"query": "What's my name?", "expected": "Alice"},
        ])
        async def test_memory(query, expected):
            response = await agent(query)
            fe.score(response, expected)
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "context_retention")
        super().__init__(**kwargs)

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        history_str = ""
        if eval_input.history:
            history_str = "\n".join(
                f"**{turn.get('role', 'unknown').title()}**: {turn.get('content', '')}"
                for turn in eval_input.history
            )
            history_str = f"\n{history_str}\n"

        expected_str = ""
        if eval_input.expected_output:
            expected_str = (
                f"\n**Expected Response** (reference): {eval_input.expected_output}"
            )

        return f"""You are an expert evaluator assessing context retention in multi-turn conversations.

## TASK
Evaluate whether the CURRENT RESPONSE demonstrates proper retention and utilization
of information from the CONVERSATION HISTORY.

## INPUTS
**Conversation History**:
{history_str if history_str else "(No prior history)"}

**Current User Input**: {eval_input.input}
**Current Response**: {eval_input.actual_output}{expected_str}

## CONTEXT RETENTION DIMENSIONS

### 1. Explicit Memory (Direct References)
- Does the response correctly recall specific facts from history?
- Are names, numbers, preferences, and details accurately remembered?
- Does it correctly reference previous statements when relevant?

### 2. Implicit Memory (Contextual Understanding)
- Does the response reflect understanding of the conversation's context?
- Are implicit assumptions from earlier turns honored?
- Does it maintain awareness of the user's situation/goals?

### 3. Anaphora Resolution
- Are pronouns and references correctly resolved?
- Does "it", "that", "this" refer to the right entities?
- Is there confusion about what's being discussed?

### 4. Temporal Coherence
- Is the conversation timeline understood?
- Are "before", "after", "then" relationships maintained?
- Does the response respect the sequence of events discussed?

### 5. User Model Maintenance
- Does it remember user preferences, constraints, or characteristics?
- Are user-specific details from earlier turns retained?
- Does it adapt based on information learned about the user?

## EVALUATION STEPS
Step 1: Identify key information from conversation history that should be retained
Step 2: Analyze the current input to determine what context is needed
Step 3: Check if the response correctly uses/references historical context
Step 4: Identify any context that was forgotten, misremembered, or ignored
Step 5: Assess overall context retention quality

## RETENTION ISSUES TO DETECT
- Forgetting explicitly stated facts
- Asking for information already provided
- Contradicting previous context
- Failing to connect related topics
- Ignoring relevant background information

## SCORING RUBRIC
- **1.0**: Perfect retention - all relevant context correctly used, no memory lapses
- **0.8**: Strong retention - minor context underutilization, no errors
- **0.6**: Moderate retention - some context used but gaps present
- **0.4**: Weak retention - significant context missed, but not contradicted
- **0.2**: Poor retention - most context ignored or misremembered
- **0.0**: No retention - acts as if no history exists, completely ignores context

## OUTPUT FORMAT
Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<identify key context items, assess retention of each, note any memory failures>"}}"""


class ConsistencyMetric(BaseLLMMetric):
    """
    Evaluates if responses are consistent and don't contradict previous statements.

    Based on dialogue contradiction detection research (Welleck et al., 2019)
    and natural language inference for conversational consistency.
    Uses entailment-based contradiction checking.

    Checks that the agent doesn't contradict itself across conversation turns.

    Example:
        @fe.consistency(threshold=1.0)
        @fe.conversation([...])
        async def test_consistent(query, expected):
            ...
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "consistency")
        kwargs.setdefault("scoring_type", "binary")
        kwargs.setdefault("threshold", 1.0)
        super().__init__(**kwargs)

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        history_str = ""
        if eval_input.history:
            history_str = "\n".join(
                f"**{turn.get('role', 'unknown').title()}**: {turn.get('content', '')}"
                for turn in eval_input.history
            )
            history_str = f"\n{history_str}\n"

        return f"""You are an expert evaluator detecting contradictions in multi-turn conversations.

## TASK
Determine if the CURRENT RESPONSE is CONSISTENT with or CONTRADICTS statements
made earlier in the CONVERSATION HISTORY, **taking into account the CURRENT USER INPUT**
that prompted the response.

## INPUTS
**Conversation History**:
{history_str if history_str else "(No prior history)"}

**Current User Input**: {eval_input.input}
**Current Response to Check**: {eval_input.actual_output}

## CONSISTENCY METHODOLOGY (NLI-Based Contradiction Detection)

### Types of Contradictions to Detect

#### 1. Direct Contradictions
- Explicit statements that negate previous claims
- Example: Previously "I love coffee" → Now "I hate coffee"

#### 2. Factual Inconsistencies
- Different facts about the same entity/topic
- Example: Previously "Meeting is at 3pm" → Now "Meeting is at 4pm"

#### 3. Logical Contradictions
- Statements that can't both be true simultaneously
- Example: "I've never been to Paris" → "When I was in Paris last year..."

#### 4. Entity Attribute Conflicts
- Contradictory attributes for the same entity
- Example: "The car is red" → "The blue car I mentioned"

#### 5. Stance/Opinion Reversals
- Unexplained changes in position or preference
- Example: "I agree with A" → "A is wrong"

### What is NOT a Contradiction
- Clarifications or elaborations
- Legitimate changes of mind (if acknowledged)
- Different aspects of the same topic
- Additional information that's compatible
- Corrections with acknowledgment ("I was wrong earlier...")
- **Responding to user corrections** (e.g., user says "Actually it is X" and the agent updates accordingly)
- **Adapting to new information provided by the user** in the current input

## IMPORTANT: Consider the Current User Input
The current user input provides essential context for why the response may differ
from previous conversation turns. If the user corrects, updates, or changes the
topic, the agent's response should adapt accordingly — this is NOT a contradiction.

## EVALUATION STEPS
Step 1: Read the current user input to understand what was asked or stated
Step 2: Extract all claims/statements from conversation history
Step 3: Extract all claims/statements from current response
Step 4: For each current claim that differs from history, check if the current user input justifies the change
Step 5: Only flag as contradiction if the response contradicts history WITHOUT justification from the user input
Step 6: Binary decision: Any unjustified contradiction = FAIL

## SCORING (Binary)
- **1**: CONSISTENT - No unjustified contradictions detected, all statements compatible or justified by user input
- **0**: INCONSISTENT - One or more unjustified contradictions detected

## OUTPUT FORMAT
Respond with JSON only:
{{"score": <0 or 1>, "reasoning": "<list any contradictions found with specific examples from history vs current response, note if user input justifies any changes, or confirm consistency>"}}"""


class TopicDriftMetric(BaseLLMMetric):
    """
    Evaluates if the agent stays appropriately on topic.

    Based on topic modeling and discourse coherence research.
    Implements topic anchor tracking and relevance decay measurement.

    Detects when responses drift away from the conversation topic.

    Example:
        @fe.topic_drift(threshold=0.8)
        @fe.conversation([...])
        async def test_on_topic(query, expected):
            ...
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "topic_drift")
        kwargs.setdefault("threshold", 0.8)
        super().__init__(**kwargs)

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        history_str = ""
        if eval_input.history:
            history_str = "\n".join(
                f"**{turn.get('role', 'unknown').title()}**: {turn.get('content', '')}"
                for turn in eval_input.history
            )
            history_str = f"\n{history_str}\n"

        return f"""You are an expert evaluator assessing topic coherence in multi-turn conversations.

## TASK
Evaluate whether the RESPONSE stays appropriately on-topic with the conversation
or drifts into unrelated territory.

## INPUTS
**Conversation History**:
{history_str if history_str else "(No prior history)"}

**Current User Input**: {eval_input.input}
**Response to Evaluate**: {eval_input.actual_output}

## TOPIC DRIFT DIMENSIONS

### 1. Topic Anchor Tracking
- What is the main topic established in the conversation?
- What sub-topics have been introduced?
- Is the current input asking about an established topic?

### 2. Response Relevance
- Does the response address the current input's topic?
- Is the content related to established conversation themes?
- Are new topics introduced appropriately (bridged from existing)?

### 3. Tangent Detection
- Does the response go off on unnecessary tangents?
- Is unrelated information injected without connection?
- Are digressions excessive or disruptive?

### 4. Topic Transition Quality
- If topic changes, is it smooth and appropriate?
- Does the user's input justify a topic shift?
- Are transitions acknowledged or abrupt?

### 5. Focus Maintenance
- Does the response stay focused on the question asked?
- Is there appropriate depth vs. breadth?
- Does it avoid rambling or unfocused content?

## EVALUATION STEPS
Step 1: Identify the main topic(s) from conversation history
Step 2: Determine the topic/focus of the current input
Step 3: Analyze the response for topic alignment
Step 4: Identify any off-topic content or unnecessary tangents
Step 5: Assess if any topic shifts are appropriate or inappropriate

## TOPIC DRIFT INDICATORS
**On-topic signs**: Directly addresses query, builds on established themes, relevant examples
**Off-topic signs**: Unrequested information, unconnected tangents, ignores query focus, random pivots

## ACCEPTABLE TOPIC EVOLUTION
- Natural expansion within the topic domain
- User-initiated topic changes (follow the user)
- Brief relevant context that supports the main topic
- Appropriate clarifying questions

## UNACCEPTABLE TOPIC DRIFT
- Unsolicited topic changes
- Excessive tangents unrelated to user need
- Ignoring the user's actual question
- Random information injection

## SCORING RUBRIC
- **1.0**: Perfectly on-topic - focused, relevant, no drift
- **0.8**: Mostly on-topic - minor tangents but returns to focus
- **0.6**: Moderately on-topic - noticeable drift but core topic addressed
- **0.4**: Somewhat off-topic - significant drift, partially addresses topic
- **0.2**: Mostly off-topic - major drift, barely addresses the topic
- **0.0**: Completely off-topic - ignores topic entirely, irrelevant response

## OUTPUT FORMAT
Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<identify main topic, assess response alignment, note any drift or tangents>"}}"""
