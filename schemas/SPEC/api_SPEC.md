# schemas/api.py SPEC

## Purpose

Define the external request and response contract for TSPilot v0.2.

## Models

- `Message`
- `ChatRequest`
- `ChatResponse`

## `Message`

Fields:

- `role: Literal["user", "assistant", "system"]`
- `content: str`
- `timestamp: str | null`

## `ChatRequest`

Required fields:

- `message: str`

Optional fields:

- `conversation_id: str | null`
- `database_context: DatabaseContext | null`
- `selected_database: str | null`
  - legacy alias for `database_context.database_id`
- `selected_database_type: str | null`
  - legacy alias for `database_context.database_type`
- `time_range: dict | null`
- `constraints: dict`
- `history: list[Message]`
- `stream: bool`

Example:

```json
{
  "message": "最近7天CPU有什么趋势？",
  "conversation_id": "conv_123",
  "database_context": {
    "database_id": "prometheus-prod",
    "database_type": "prometheus",
    "display_name": "Prometheus Prod"
  },
  "time_range": {
    "start": "2026-07-07T00:00:00Z",
    "end": "2026-07-14T00:00:00Z"
  },
  "constraints": {
    "max_points": 100,
    "language": "zh"
  },
  "history": [],
  "stream": true
}
```

## `ChatResponse`

Required fields:

- `conversation_id: str`
- `request_id: str`
- `status: Literal["completed", "failed"]`
- `response_kind: Literal["final_answer", "error"]`
- `used_tools: list[str]`

Optional fields:

- `answer: FinalAnswer | null`
- `trace: list[TraceEventModel] | null`
- `error: str | null`

Example:

```json
{
  "conversation_id": "conv_123",
  "request_id": "req_456",
  "status": "completed",
  "response_kind": "final_answer",
  "used_tools": ["query_database", "insight", "format_answer"],
  "answer": {
    "title": "CPU趋势分析",
    "summary": "最近7天CPU整体上升。",
    "sections": [],
    "references": [],
    "visualizations": []
  }
}
```

## Streaming event payloads

Suggested event types:

- `thought`
- `action`
- `observation`
- `final_answer`
- `error`
- `terminate`

Contract note:

- `terminate` here is a runtime-owned completion event, not a prompt-visible model action.

## Responsibilities

- validate the API boundary
- normalize legacy database aliases into `database_context`
- carry the terminal payload and trace data

## Must not do

- embed ReAct logic
- embed tool execution
