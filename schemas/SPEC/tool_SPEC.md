# schemas/tool.py SPEC

## Purpose

Define the normalized wrapper types for outer action execution.

## Models

- `ToolCall`
- `ToolObservation`
- `ReActTranscriptStep`
- `ToolError`

## `ToolCall`

Fields:

- `tool_name: str`
- `tool_input: dict`
- `iteration: int`

Contract notes:

- `tool_name` must resolve to exactly one registered outer action
- `tool_input` must match the selected action contract
- `ToolCall` is an execution record, not the model output itself
- decision reasoning lives once in `ReActTranscriptStep.thought`; it is not
  duplicated on each `ToolCall`

## `ReActTranscriptStep`

Fields:

- `iteration: int`
- `thought: str | null`
- `action: str`
- `action_input: dict`
- `observation: ToolObservation | null`

The transcript does not repeat the user question, phase, action intention, or
action reason.

## `ToolObservation`

Fields:

- `tool_name: str`
- `success: bool`
- `summary: str`
- `payload: dict`
- `error: str | null`
- `payload_truncated: bool`
- `payload_ref: str | null`

Contract notes:

- `payload` is the normalized runtime result of the action
- `summary` must be human-readable for trace/debug use
- the runtime may truncate or summarize payloads for prompt visibility
- the full payload remains available to runtime state and trace storage

## `ToolError`

Fields:

- `tool_name: str`
- `error_code: str`
- `message: str`
- `retryable: bool`

## Responsibilities

- normalize all outer action execution records
- keep runtime/tool boundaries stable

## Must not do

- execute business logic
- choose the next action
