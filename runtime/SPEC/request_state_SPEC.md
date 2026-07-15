# runtime/request_state.py SPEC

## Purpose

Define how runtime reads and mutates one request's state.

## Input

- normalized `RequestStateModel`

## Output

- updated `RequestStateModel`

## Responsibilities

- initialize request state from API payload
- update control state after each outer ReAct turn
- keep evidence, analysis, and presentation state in sync with tool output
- preserve traceable history of tool calls and observations
- enforce the prompt context budget before each model turn
- store compacted context summaries when older turns are truncated

## Reads

- request context
- current tool observation
- model turn parse result

## Writes

- status
- current intent
- requested fact types
- focus
- todo list
- latest evidence / analysis / presentation payloads
- tool history
- observations
- prompt context summary
- context status / overflow reason

## Must not do

- infer business conclusions
- execute tools
