# runtime/react_loop.py SPEC

## Purpose

Execute the single outer ReAct loop for `data_agent`.

## Inputs

- `RequestStateModel`
- `ConversationStateModel`
- `ToolExecutor`
- prompt builder
- model client
- `max_iterations`

## Outputs

- updated `RequestStateModel`
- terminal payload: final answer or error
- trace events

## Model-visible context

The prompt builder may include only a bounded, runtime-curated view of state:

- current user message
- normalized `database_context`
- legacy database aliases only when present in the incoming request
- current control state
- latest evidence / analysis / visualization payloads
- bounded recent messages
- bounded tool history and observations
- `prompt_context_summary` when older context has been compacted

The model must not assume it can see:

- full raw trace data
- full historical tool payloads beyond the bounded window
- hidden runtime internals

## ReAct turn contract

Each model turn must be parsed as one `ReActTurn` with:

- `Thought`
- `Action`
- `Action Input`

`Observation` is runtime-owned and must never be emitted by the model.

## Allowed actions

- `todowrite`
- `sql_query`
- `code_interpreter`
- `forecast`
- `anomaly`
- `rag`
- `skill`
- `terminate`

## Runtime behavior

1. load request and conversation state
2. build the prompt context
3. ask the model for one ReAct turn
4. parse the turn into `schemas.agent_turn.ReActTurn`
5. validate `action` and `action_input`
6. if `action` is an outer tool, invoke exactly one tool through `ToolExecutor`
7. normalize the result into `ToolObservation`
8. append trace events
9. update request and conversation state
10. if the executed tool has `produces_terminal_payload = true`, persist the assembled final answer, emit a runtime `terminate` event, and end the loop
11. otherwise re-enter the loop until a stop condition is hit

## Action dispatch rules

- all allowed actions must resolve through `ToolExecutor`
- one loop iteration may emit at most one `ToolCall` and one `ToolObservation`
- a failed parse or validation must not create a partial `ToolCall`

## Current fact-routing behavior

The outer agent does not yet plan fact execution at the level of
"query-native fact" versus "analysis-native fact".

Current practical routing is:

1. retrieve evidence through `sql_query`
2. inspect the returned evidence family
3. if the family is non-timeseries (`statistics`, `table`, `schema`,
   `metric_list`), prefer `terminate`
4. if the family is `timeseries`, use `code_interpreter`, `anomaly`, or
   `forecast` only when the current ReAct gap assessment requires them, then `terminate`

This means the main routing boundary for many requests currently depends on the
evidence family selected by `sql_query`, not on a dedicated fact execution
plan.

Design note:

- mixed requests such as "先给均值，再分析趋势" should eventually be split by a
  dedicated planning layer into query-native facts and analysis-native facts
- that planning layer is not yet implemented in the current runtime contract
- the future planning layer must remain backend-agnostic: runtime should reason
  over logical query plans and evidence shapes, not over SQL-only assumptions
- the actual backend query may be SQL, Flux, PromQL, or another adapter-native
  request format, and traces should preserve that distinction explicitly

## Validation rules

- a malformed turn must not trigger partial tool execution
- unknown actions must be rejected
- `Action Input` must match the selected action contract
- if validation fails, emit an error trace and stop or request correction

## Observation visibility rules

After each successful tool call, runtime must store:

- the full `ToolObservation`
- a prompt-visible observation block derived from it

The prompt-visible observation block may include:

- `tool_name`
- `success`
- `summary`
- `payload`
- `payload_truncated`
- `payload_ref`

The visible `payload` must obey the request context budget. Runtime should trim
in this order:

1. oversized raw rows or points inside the payload
2. verbose diagnostics
3. old observation payloads already superseded by newer state

The runtime must not truncate:

- `tool_name`
- `success`
- `summary`
- stable ids needed for follow-up actions, such as `evidence_id`, `fact_id`,
  `visualization_id`, `forecast_id`, and `anomaly_id`

## Prompt budget alignment

Prompt assembly and observation truncation must use the same budget policy.

Recommended runtime fields:

- `max_prompt_tokens`
- `max_history_messages`
- `max_tool_history_items`
- `max_observation_chars`
- `max_visible_rows`
- `max_visible_points`

## Stop conditions

- terminal payload completed and final answer is ready
- max iterations reached
- context budget overflow after compaction
- unrecoverable error

## Circuit breaker

Before each model turn, the runtime must enforce a prompt context budget.

When the assembled prompt would exceed budget:

1. compact older conversation turns into `prompt_context_summary`
2. drop or summarize older tool observations and evidence payloads first
3. keep the newest user message, latest evidence, and latest analysis payloads
4. if the prompt still cannot fit, stop with a controlled overflow error

The circuit breaker must be deterministic and must not depend on model judgment.

## State update rules

After each tool observation, runtime-owned state update logic should route
results by `ToolSpec.result_target`:

- `todo` updates `todo_list`
- `evidence` updates `latest_database_evidence`
- `analysis` updates the matching analysis slot and any derived `visualizations`
- `presentation` updates `final_answer_draft`

The executor must not perform these mutations directly.

## Reads

- request state
- conversation state
- model output
- tool registry

## Writes

- request state
- conversation state
- trace events

## Responsibilities

- implement the outer ReAct control flow
- ensure one action per turn
- keep the model on a strict action contract
- enforce prompt budget and compaction behavior

## Forbidden responsibilities

- domain-specific business logic
- database execution logic
- analysis logic
