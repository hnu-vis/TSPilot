# tools/registry.py SPEC

## Purpose

Map outer action names to tool implementations and schemas.

## Registry model

The registry should expose one `ToolSpec` record per prompt-visible action.

### `ToolSpec`

Fields:

- `tool_name: str`
- `description: str`
- `input_contract_ref: str`
- `implementation_ref: str`
- `prompt_visible: bool`
- `runtime_access: Literal["none", "request_state_read", "request_and_conversation_read"]`
- `result_target: Literal["todo", "evidence", "analysis", "presentation"]`
- `produces_terminal_payload: bool`
- `supports_streaming: bool`

## Required registry entries

The registry must define exactly one `ToolSpec` for each prompt-visible action:

- `todowrite`
- `query_database`
- `insight`
- `forecast`
- `anomaly`
- `rag`
- `skill`
- `format_answer`

## Resolution rules

- one action name resolves to exactly one `ToolSpec`
- `prompt_visible = false` entries must never appear in the model action space
- runtime-owned finalization must not appear as a registry entry
- the prompt builder should derive tool metadata from the same registry used by
  the executor

## Responsibilities

- register available tools
- expose tool metadata to the runtime and prompt builder
- resolve one action name to one tool implementation
- keep runtime-owned finalizers out of the prompt-visible action set
- declare which tools may read runtime state
- declare which layer each tool writes into

## Must not do

- execute tool logic
- choose the next action
- contain business rules
