# schemas/agent_turn.py SPEC

## Purpose

Define the parsed structure of one outer ReAct turn produced by `data_agent`.

## Models

- `ReActTurn`
- `ReActTurnParseError`

## `ReActTurn`

Fields:

- `thought: str`
- `task_contract: TaskContract | None`
- `previous_observation_assessment: PreviousObservationAssessment | None`
- `action: str`
- `action_input: dict`

Contract notes:

- `action` must be one allowed outer action name
- `action_input` must satisfy the corresponding tool contract
- `Observation` is not part of the model-emitted turn
- the runtime may attach observation after tool execution
- `thought` is the single decision explanation; there are no duplicate
  `action_intention` or `action_reason` fields
- `PreviousObservationAssessment` contains only the prior artifact acceptance
  receipt needed by completion state (`completed_active_todo`, `reason`,
  `evidence_refs`, `missing`, and answerability); runtime owns Todo transitions,
  so completed-Todo lists and next-Todo selectors are not model fields
  and the next-action explanation belongs in `thought`, not the assessment
- a visualization Thought must name the exact verified Insight keys, the
  inspectable verification question, and why the upstream lineage is sufficient;
  malformed visual decisions are sent through LLM contract repair before trace
  emission
- a visualization Action cites at least one verified `insight:` target. It does
  not repeat that Insight's Evidence/Analysis refs; the visualization tool
  resolves related data through the Insight's canonical lineage

Allowed outer action names:

- `todowrite`
- `sql_query`
- `code_interpreter`
- `forecast`
- `anomaly`
- `visualization`
- `rag`
- `skill`
- `terminate`

## `ReActTurnParseError`

Fields:

- `error_code: str`
- `message: str`
- `raw_turn: str`

## Responsibilities

- define the parse target for one standard ReAct turn
- keep prompt and runtime aligned on the same output shape

## Must not do

- define business logic
- execute tools
