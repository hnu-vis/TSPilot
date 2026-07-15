# runtime/tool_executor.py SPEC

## Purpose

Resolve and invoke one outer action by name.

## Inputs

- action name
- action input payload
- current `RequestStateModel`
- current `ConversationStateModel`

## Outputs

- normalized `ToolObservation`

## Reads

- tool registry
- request state
- conversation state when required by the selected action

## Writes

- tool history
- observations
- trace hooks

## Execution contract

The executor should work in five deterministic stages:

1. resolve `ToolSpec` from the registry
2. validate `action_input` against `input_contract_ref`
3. build the allowed runtime context for the selected tool
4. invoke exactly one implementation
5. normalize the raw result into one `ToolObservation`

## Runtime context injection

The executor must inject runtime context only according to `ToolSpec.runtime_access`.

### `runtime_access = "none"`

- pass only the validated `action_input`
- applies to tools that can run as pure capability calls

### `runtime_access = "request_state_read"`

- pass the validated `action_input`
- pass a read-only view of `RequestStateModel`
- do not pass `ConversationStateModel`

### `runtime_access = "request_and_conversation_read"`

- pass the validated `action_input`
- pass a read-only view of `RequestStateModel`
- pass a read-only view of `ConversationStateModel`

The executor must not pass mutable state objects into tools. Tool results are
applied by runtime-owned state update logic after observation normalization.

## Output normalization

`ToolObservation` must be built from two payload views:

- `full_payload`: the complete normalized tool result kept in runtime state
- `visible_payload`: the bounded prompt-visible projection

Normalization rules:

- `tool_name` must equal the selected action name
- `success` is `true` only when the implementation returns a valid normalized result
- `summary` must be short and human-readable
- `payload` stores only `visible_payload`
- `payload_truncated = true` when `visible_payload` omits fields or rows from `full_payload`
- `payload_ref` must be a stable runtime reference when truncation occurs, such as
  `obs:<request_id>:<iteration>:<tool_name>`

## Failure rules

- unknown tool name: reject before invocation
- schema validation failure: reject before invocation
- registry/configuration error: emit non-retryable tool error
- tool implementation error: normalize into failed `ToolObservation`
- state-injection violation: fail closed and emit error trace

## Responsibilities

- resolve the selected action from the registry
- validate the action input against the tool contract
- invoke exactly one tool
- normalize success or failure into `ToolObservation`
- enforce read-only runtime context injection
- separate full payload storage from prompt-visible payloads

## Forbidden responsibilities

- choose the next action
- implement domain logic
- mutate analysis state directly

## Contract notes

- the executor must reject unknown actions
- the executor must reject inputs that do not match the selected contract
- the executor may pass through runtime context only when the selected tool requires it
- the executor must mark large payloads as truncated when they exceed the prompt budget
- the executor must preserve the full payload in runtime state even when the visible prompt view is truncated
