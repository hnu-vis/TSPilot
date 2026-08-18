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
- current control state
- compact artifact references and verified Insight state
- latest Action Output observation
- bounded recent trajectory receipts

The model must not assume it can see:

- full raw trace data
- full historical tool payloads beyond the bounded window
- hidden runtime internals

## ReAct turn contract

Each model turn must be parsed as one JSON `ReActTurn` with:

- `thought`
- optional `task_contract`
- optional `previous_observation_assessment`
- `action`
- `action_input`

`Observation` is runtime-owned and must never be emitted by the model.

## Allowed actions

- `todowrite`
- `sql_query`
- `code_interpreter`
- `forecast`
- `anomaly`
- `visualization`
- `rag`
- `skill`
- `terminate`

## Runtime behavior

1. load request and conversation state
2. build the prompt context
3. emit the round placeholder, then ask the model for one ReAct turn while
   streaming a distinct child span for every concrete model invocation; the
   decision phase uses `iteration-N:decision`, while any selected tool uses
   `iteration-N`, so the placeholder/decision can never become the tool node
4. parse the turn into `schemas.agent_turn.ReActTurn`
5. validate the selected action against runtime policy; visualization Thoughts
   that omit verified targets, an inspectable relation, or complete context,
   and visualization Actions that omit target Insight refs, are repaired by the
   LLM before reaching this step; related artifacts are resolved from Insight
   lineage inside the visualization tool
6. normalize and validate the input once through `ToolExecutor.prepare`
7. emit Action from the prepared semantic input, excluding runtime-owned
   context and diagnostics
8. invoke that exact prepared action once
9. normalize the result into `ToolObservation`, update canonical state, then
   compute its coverage receipt
10. append the minimal Thought / Action / Observation transcript and trace
11. update request and conversation state
12. if the executed tool has `produces_terminal_payload = true`, persist the assembled final answer, emit a runtime `terminate` event, and end the loop
13. otherwise re-enter the loop until a stop condition is hit

## Action dispatch rules

- all allowed actions must resolve through `ToolExecutor`
- one loop iteration may emit at most one `ToolCall` and one `ToolObservation`
- a failed parse or validation must not create a partial `ToolCall`
- a rejected or input-invalid attempted action may still be shown in the trace,
  but it is closed as a failed policy decision and never recorded as an
  executed `ToolCall`

## Insight and visualization routing

1. `sql_query` owns grounded database Evidence and atomic query-native Insights.
2. `code_interpreter` owns calculated Insights and reusable derived Evidence;
   generated computation and independent LLM semantic binding are separate.
3. `anomaly` and `forecast` own their specialized artifacts; code may derive
   requested conclusions from them but cannot replace them.
4. `visualization` starts from verified Insights, selects an inspectable
   relation, projects complete contextual artifacts, materializes the chart, and
   passes semantic and production-render audits before publication.
5. Completion is evaluated from the current task contract, Insights, and
   artifacts on demand. A broad artifact does not cover a missing exact output.

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
- stable ids needed for follow-up actions, such as `evidence_id`, `insight_id`,
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
