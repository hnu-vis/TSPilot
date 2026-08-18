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
- canonical `thought` contains only `iteration` and the decision record
- canonical `action` contains only `iteration`, the selected action, and the
  compact runtime-validated semantic input; SSE exposes the same data through
  `step.meta` / `tool_call` compatibility events
- canonical `observation` carries the result summary, grounded artifact receipt,
  and coverage result; it does not repeat Thought or Action metadata
- runtime-only database context, history, retrieval diagnostics, and duplicate
  question/phase/intention/reason fields are not part of the ReAct transcript

## Responsibilities

- normalize trace event format
- support debugging and replay
- preserve the sequence of outer ReAct turns

## Must not do

- store business state itself
- change execution flow
