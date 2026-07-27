# schemas/agent_turn.py SPEC

## Purpose

Define the parsed structure of one outer ReAct turn produced by `data_agent`.

## Models

- `ReActTurn`
- `ReActTurnParseError`

## `ReActTurn`

Fields:

- `thought: str`
- `action: str`
- `action_input: dict`

Contract notes:

- `action` must be one allowed outer action name
- `action_input` must satisfy the corresponding tool contract
- `Observation` is not part of the model-emitted turn
- the runtime may attach observation after tool execution

Allowed outer action names:

- `todowrite`
- `sql_query`
- `code_interpreter`
- `forecast`
- `anomaly`
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
