# runtime/trace.py SPEC

## Purpose

Define trace events for one request execution.

## Event types

- `thought`
- `action`
- `observation`
- `todo_update`
- `final_answer`
- `terminate`
- `error`

Contract note:

- `terminate` is a runtime-owned completion event emitted after the terminal payload is persisted.

## Responsibilities

- normalize trace event format
- support debugging and replay
- preserve the sequence of outer ReAct turns

## Must not do

- store business state itself
- change execution flow
