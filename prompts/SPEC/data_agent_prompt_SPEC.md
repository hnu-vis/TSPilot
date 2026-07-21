# prompts/data_agent.py SPEC

## Purpose

Define the system prompt and prompt builder for the single outer `data_agent`.

## Prompt contract

The model must produce one ReAct block per turn:

- `Thought:`
- `Action:`
- `Action Input:`

`Observation:` is runtime-owned and must not be emitted by the model.

`Action Input` must be a single JSON object.

## Model-visible context

The prompt builder may expose only a bounded view of runtime state:

- current user message
- normalized `database_context`
- optional legacy `selected_database` / `selected_database_type`
- time range and constraints
- bounded recent conversation messages
- current control state
- current todo list
- latest observation summaries
- latest evidence / analysis payloads
- verified facts
- visualizations
- `prompt_context_summary` when older context has been compacted

The model must treat anything outside that window as unavailable.

## Allowed actions

- `todowrite`
- `query_database`
- `insight`
- `forecast`
- `anomaly`
- `rag`
- `skill`
- `terminate`

## Prompt responsibilities

- define the outer ReAct format
- describe the allowed action space
- provide the action input contract for each action
- instruct the model to prefer evidence-grounded output
- instruct the model to rely on verified facts only
- instruct the model to prefer deterministic recovery over guessing

## Example ReAct turn

```text
Thought: I need evidence before deciding the facts.
Action: query_database
Action Input: {"message":"最近7天CPU有什么趋势？","database_context":{"database_id":"prometheus-prod","database_type":"prometheus","display_name":"Prometheus Prod"},"time_range":{"start":"2026-07-07T00:00:00Z","end":"2026-07-14T00:00:00Z"},"constraints":{"max_points":100},"history":[]}
```

## Action input contracts

### `todowrite`

Required fields:

- `message`
- `current_intent`
- `requested_fact_types`
- `focus`
- `todos`

Optional fields:

- `evidence_summary`

### `query_database`

Required fields:

- `message`
- `database_context`
- `time_range`
- `constraints`

Optional fields:

- `selected_database`
- `selected_database_type`
- `history`

### `insight`

Required fields:

- `database_evidence`
- `requested_fact_types`
- `focus`

Optional fields:

- `constraints`

### `forecast`

Required fields:

- `database_evidence`

Optional fields:

- `horizon`
- `constraints`

### `anomaly`

Required fields:

- `database_evidence`

Optional fields:

- `constraints`

### `rag`

Required fields:

- `query`

Optional fields:

- `database_context`
- `database_evidence`
- `filters`

### `skill`

Required fields:

- `skill_name`
- `task_context`

Optional fields:

- `parameters`

### `terminate`

Required fields:

- none

Optional fields:

- `result`
- `summary_goal`
- `direct_answer`
- `include_fact_ids`
- `include_analysis_ids`
- `include_visualization_ids`
- `section_plan`

## Observation contract

After the runtime executes one action, it appends an `Observation` block for the next turn.

The model-visible observation is a bounded rendering of the selected tool result and may include:

- `tool_name`
- `success`
- `summary`
- a bounded `payload` view when small enough
- a payload reference or digest when the payload was truncated

The model must rely on the observation summary and any visible payload fields only.

## Prompt-required structured fields

The prompt must explicitly expose these fields to the model:

### request context fields

- `message`
- `database_context`
- `selected_database`
- `selected_database_type`
- `time_range`
- `constraints`
- `history`

### request-state fields

- `current_intent`
- `requested_fact_types`
- `focus`
- `todo_list`
- `database_context`
- `latest_database_evidence`
- `latest_insight`
- `latest_forecast`
- `latest_anomaly`
- `latest_rag`
- `latest_skill`
- `verified_facts`
- `visualizations`
- `prompt_context_summary`

### database evidence fields

- `evidence_id`
- `result_type`
- `database`
- `query_language`
- `query`
- `summary`
- `data`
- `columns`
- `metadata`
- `diagnostics`

Valid `result_type` values:

- `schema`
- `metric_list`
- `statistics`
- `table`
- `timeseries`

### insight result fields

- `insight_id`
- `requested_fact_types`
- `supported_fact_types`
- `fact_candidates`
- `completed_facts`
- `verified_facts`
- `rejected_facts`
- `summary_blocks`
- `visualizations`

### final answer fields

- `title`
- `summary`
- `sections`
- `references`
- `visualizations`

### time-series result fields

- `forecast_id`
- `anomaly_id`

### observation fields

- `tool_name`
- `success`
- `summary`
- `payload`
- `payload_truncated`
- `payload_ref`

## ReAct guidance

The prompt should guide the model with this order:

1. inspect the request context and determine the current intent
2. decide whether the request needs `todowrite`
3. decide whether the request needs `query_database`
4. decide requested fact families before calling `insight`
5. call `forecast` or `anomaly` only when the evidence is time-series shaped
6. call `rag` or `skill` only when the request genuinely needs extension capability
7. call `terminate` when enough verified outputs exist and the ReAct loop should end; do not pass full runtime state back to the tool
8. let the runtime stop the loop after the terminal payload has been produced

## Prompt rules

- one model turn must produce exactly one ReAct block
- `Action` must name exactly one allowed action
- `Action Input` must satisfy the selected action contract
- `Observation` is not model output
- prefer deterministic recovery or a failed observation over guessing when required fields are missing
- treat `query_database` output as evidence, not as a final answer
- use only `verified_facts` in final narration
- prefer `linechart` for time-indexed or ratio/comparison facts when the evidence supports it
- do not emit hidden state changes outside the structured block
- do not emit `Observation`
- treat `terminate` as the final model-visible action
- do not assume access to truncated context or hidden runtime state

## Must not do

- contain backend-specific business logic
- contain tool execution code
