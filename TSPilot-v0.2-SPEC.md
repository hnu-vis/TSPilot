# TSPilot v0.2 SPEC

## Goal

Build one simple and strict execution model for time-series data work:

- one outer `data_agent`
- one outer ReAct loop
- one action per turn
- typed tool contracts
- deterministic core modules behind tools
- evidence-first analysis
- final answers grounded in verified facts

## Canonical request flow

1. API receives `ChatRequest`
2. request is normalized into `RequestStateModel` and `ConversationStateModel`
3. `data_agent` produces one standard ReAct turn
4. runtime validates the turn and invokes at most one action
5. the action returns a normalized `ToolObservation`
6. runtime updates state and loops
7. when evidence is sufficient, `format_answer` assembles `FinalAnswer`
8. the runtime finalizes the request after a terminal payload exists

## Main action space

- `todowrite`
- `sql_query`
- `code_interpreter`
- `forecast`
- `anomaly`
- `rag`
- `skill`
- `format_answer`

## Core design rules

- `sql_query` returns evidence only; schema linking and query generation are LLM-driven
- analysis tools convert evidence into request-scoped facts and analysis artifacts
- `forecast` and `anomaly` only consume time-series evidence
- `format_answer` only assembles verified outputs
- request termination is runtime-owned after a final answer or error terminal state exists
- prompt context is bounded; older context is compacted deterministically
- the runtime never executes domain logic directly

## Architectural layers

### Control

- API layer
- runtime loop
- tool executor
- trace handling

### State

- request state
- conversation state
- database context
- tool call / observation wrappers

### Evidence

- database evidence

### Analysis

- analysis artifacts
- forecast result
- anomaly result

### Presentation

- final answer
- references
- visualizations

## File groups

The canonical file groups are:

- `app/*` for HTTP and streaming
- `runtime/*` for loop and execution control
- `agents/*` for orchestration
- `tools/*` for visible capabilities
- `core/*` for deterministic helpers
- `schemas/*` for typed contracts
- `prompts/*` for model-facing output format

## Acceptance criteria

- the model emits exactly one ReAct block per turn
- each action input matches the selected tool contract
- the runtime rejects malformed actions
- evidence and analysis are separated from presentation
- the final answer is traceable to evidence and verified facts

## Non-goals for v0.2

- multi-agent orchestration
- graph routing
- long-term memory
- broad skill ecosystem
- hidden prompt-driven business logic
