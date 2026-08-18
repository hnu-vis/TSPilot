# runtime/trace.py SPEC

## Purpose

Define trace events for one request execution.

## Event types

- `thought`
- `agent_decision_start`
- `trace_span_start`
- `trace_span_end`
- `action_proposed`
- `tool_boundary`
- `action`
- `action_output`
- `observation`
- `todo_update`
- `final_answer`
- `terminate`
- `error`

Contract note:

- `terminate` is a runtime-owned completion event emitted after the terminal payload is persisted.
- every ReAct round emits `agent_decision_start` before awaiting the outer model,
  so the public UI can create a stable placeholder immediately; the decision
  uses `iteration-N:decision` and must never be merged into the later
  `iteration-N` tool execution node
- every request-scoped LLM invocation emits one correlated
  `trace_span_start` / `trace_span_end` pair under its owning ReAct round;
  retries are separate invocations and therefore separate pairs
- LLM spans are first-class selectable execution nodes: the stable `span_id`
  is upserted from running to complete/error, while `parent_id` preserves the
  owning ReAct/tool relationship
- `trace_span_start` exposes structural input metadata plus a bounded textual
  message preview; `trace_span_end` adds structural output metadata, token
  counts, and a bounded textual output preview. Binary values, multimodal
  payloads, and data URLs are replaced by descriptive markers. Provider names
  and model identifiers remain outside trace presentation data
- canonical `thought` contains only `iteration` and the decision record
- `action_proposed` records a model proposal that was rejected before tool
  execution; it must not produce public `step.meta` or `tool_call` events
- `tool_boundary` proves that an allowed tool entered preparation/execution and
  is the only event that may open the public `step.meta` / `tool_call` lifecycle
- canonical `action` contains only `iteration`, the selected action, and the
  compact runtime-validated semantic input produced by preparation; it may
  update public step metadata but must not open a second tool lifecycle
- a rejected policy `action_output` maps to `policy_decision` followed by
  `step.done(status=failed)`, never to `tool_result`
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
