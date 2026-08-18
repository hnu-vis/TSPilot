# prompts/data_agent.py SPEC

## Purpose

Define the system prompt and prompt builder for the single outer `data_agent`.

## Prompt contract

The model must produce one JSON ReAct turn with:

- `thought`
- `task_contract`
- `previous_observation_assessment`
- `action`
- `action_input`

`Observation:` is runtime-owned and must not be emitted by the model.

`action_input` must be a single JSON object. `thought` is the only decision
explanation; separate intention and reason fields are not part of the contract.

## Model-visible context

The prompt builder may expose only a bounded view of runtime state:

- current user message
- normalized `database_context`
- time range and constraints
- bounded recent conversation messages
- minimal execution state
- current todo list
- a compact artifact reference inventory
- bounded immutable artifact facts needed to quote exact values in the final
  answer; these facts are presentation receipts, not inputs for new calculation
- the latest Action Output observation
- verified Insight state
- a bounded recent Thought / Action / Observation trajectory
- `prompt_context_summary` when older context has been compacted

The model-visible context must not expose runtime completion or repair state as
global guidance, including:

- `completion_state`
- `latest_gap_assessment`
- `latest_goal`
- `decision_frame`
- SQL `task_coverage` as a top-level or evidence-level routing signal

The model must treat anything outside that window as unavailable.

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

## Prompt responsibilities

- define the outer ReAct format
- describe the allowed action space
- provide the action input contract for each action
- instruct the model to prefer evidence-grounded output
- instruct the model to rely on verified insights only
- instruct the model to use grounded LLM-guided repair or return structured
  unavailability instead of guessing
- instruct the model to identify user-visible task outputs before choosing tools
- prevent fixed tool-chain templates for database-computable statistics

## Task-first evidence rule

- First translate the request into a task output contract: required measures, dimensions, time scope, grouping, comparisons, derived quantities, model outputs, and evidence quality notes.
- After each observation, write `previous_observation_assessment` as the gap
  between the task output contract and current evidence. Do not duplicate this
  decision as separate intention/reason fields. The assessment accepts only the
  immediately active Todo; runtime computes Todo transitions.
- Choose tools to produce missing contract fields from grounded evidence; do not hard-code a tool chain for a task phrase.
- Database-computable work should be satisfied by `sql_query` when returned columns or result shape explicitly cover the contract.
- Use `code_interpreter` after SQL evidence is grounded for requested derived or analytical Key Insights.
- Every call must carry exact, non-empty `insight_requests`; Python code is optional because generation is internal.
- Generated Python receives canonical variables `df`, `time`, `value`, `time_col`, `value_col`, `series`, and `analysis_context`, plus compatibility variables. It emits only `computed_insights` and optional `derived_evidence`.
- Formal statements and Insight semantics are added by the independent LLM Insight Binder without changing computed values.
- A user-facing conclusion is one atomic Insight. Its calculation method and
  provenance belong in `calculation_trace` and `evidence_refs`; method/basis and
  display-context carrier Insights must not be requested separately.
- A visualization Action explicitly cites the verified target Insight. The
  visualization tool parses its evidence lineage and loads the related complete
  data; the outer Action does not duplicate contextual artifact refs.
- A line or area is meaningful only with at least two grounded points per
  series. Scalar method/boundary receipts cannot become decorative line layers.
- `missing` is only for explicitly requested core outputs that cannot be answered from current evidence. Do not put optional drill-downs, caveats, nicer formatting, or quality notes in `missing`.
- Do not set `can_answer=true` while `missing` contains core requested outputs. If the core request is answerable, set `can_answer=true` and keep `missing` empty. Terminate with truly missing core outputs only when the terminal input explicitly includes `unavailable_outputs` and `unavailable_reason`; both fields must remain visible in the compact terminate action contract.
- Final citations prefer exact semantic Insight keys. Opaque Insight IDs are
  copied verbatim when needed and are never reconstructed by the model.
- Machine-readable timestamps remain unchanged in structured artifacts,
  citations, locators, JSON, and code. User-facing prose, titles, and chart
  annotations render absolute times naturally in the response language while
  retaining the precision and timezone required by the evidence.

## Example ReAct turn

```json
{"thought":"当前没有数据库证据；需要先取得完整区间序列，再判断可计算的趋势 Insight。","task_contract":null,"previous_observation_assessment":null,"action":"sql_query","action_input":{"message":"查询最近7天 CPU 的完整时序数据，供趋势计算和可视验证使用。","time_range":{"start":"2026-07-07T00:00:00Z","end":"2026-07-14T00:00:00Z"}}}
```

## Action input contracts

### `todowrite`

Required fields:

- `message`
- `current_intent`
- `focus`
- `todos`

Optional fields:

- `evidence_summary`

### `sql_query`

Required fields:

- `message`

Optional fields:

- `purpose`
- `time_range`
- `constraints`
- `insight_requests`

Runtime database context, intent profile, selected-database metadata, and
conversation history are injected internally and are not Action fields.

### `forecast`

Required fields:

- `database_evidence`

Optional fields:

- `horizon` as explicit steps or a duration-like user phrase
- `model_name`
- `series_name`
- `constraints`

Contract notes:

- a `forecast` observation with `status=succeeded` provides direct forecast points
- a `forecast` observation with `status=requires_rolling` provides a forecast plan and is answerable unless the user explicitly requested executing every rolling chunk
- `code_interpreter` may compute supporting statistics, but must not replace the registered `forecast` tool for a requested forecast

### `anomaly`

Required fields:

- `database_evidence`

Contract notes:

- `code_interpreter` may compute supporting statistics, but must not replace the registered `anomaly` tool for requested anomaly detection

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
- `include_insight_ids`
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
- `focus`
- `todo_list`
- `database_context`
- `latest_database_evidence`
- `latest_forecast`
- `latest_anomaly`
- `latest_rag`
- `latest_skill`
- `insight_set`
- `insight_events`
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
3. decide whether the request needs `sql_query`
4. decide whether grounded evidence needs `code_interpreter`
5. call `forecast` or `anomaly` only when the evidence is time-series shaped
6. call `rag` or `skill` only when the request genuinely needs extension capability
7. call `terminate` when enough verified outputs exist and the ReAct loop should end; do not pass full runtime state back to the tool
8. let the runtime stop the loop after the terminal payload has been produced

## Prompt rules

- one model turn must produce exactly one ReAct block
- `Action` must name exactly one allowed action
- `Action Input` must satisfy the selected action contract
- `Observation` is not model output
- use the structured LLM repair path or a failed observation instead of guessing
- treat `sql_query` output as evidence, not as a final answer
- use only selected `insight_set` insights from the current request flow in final narration
- let visualization choose a grounded renderer-native mark from the verified
  relationship and its related data
- do not emit hidden state changes outside the structured block
- do not emit `Observation`
- treat `terminate` as the final model-visible action
- do not assume access to truncated context or hidden runtime state

## Must not do

- contain backend-specific business logic
- contain tool execution code
