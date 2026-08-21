# Chat API Contract

The canonical implementation is defined in `schemas/api.py`. This document
describes the stable request and response boundary used by chat clients.

## `Message`

| Field | Type | Required |
|---|---|:---:|
| `role` | `user \| assistant \| system` | yes |
| `content` | string | yes |
| `timestamp` | string or null | no |

## `ChatRequest`

| Field | Type | Required | Description |
|---|---|:---:|---|
| `message` | string | yes | Current user request. |
| `conversation_id` | string or null | no | Existing conversation identifier. |
| `model_id` | string or null | no | Model connection selected for this request. |
| `database_context` | object or null | no | Selected database identifier, type, and display metadata. |
| `selected_database` | string or null | no | Legacy alias for the database identifier. |
| `selected_database_type` | string or null | no | Legacy alias for the database type. |
| `time_range` | object or null | no | Optional request-level time boundary. |
| `constraints` | object | no | Additional request constraints. |
| `history` | `Message[]` | no | Client-provided conversation context. |
| `stream` | boolean | no | Enables streaming when true. |

Example:

```json
{
  "message": "How did CPU usage change over the last seven days?",
  "conversation_id": "conv_123",
  "model_id": "primary-model",
  "database_context": {
    "database_id": "prometheus-prod",
    "database_type": "prometheus",
    "display_name": "Prometheus Prod"
  },
  "time_range": {
    "start": "2026-08-15T00:00:00Z",
    "end": "2026-08-22T00:00:00Z"
  },
  "constraints": {},
  "history": [],
  "stream": true
}
```

## `ChatResponse`

| Field | Type | Required | Description |
|---|---|:---:|---|
| `conversation_id` | string | yes | Conversation identifier. |
| `request_id` | string | yes | Request identifier. |
| `status` | `completed \| partial \| failed` | yes | Terminal request status. |
| `response_kind` | `final_answer \| error` | yes | Response payload kind. |
| `used_tools` | string array | yes | Tools used during the request. |
| `answer` | `FinalAnswer` or null | no | Grounded terminal answer. |
| `trace` | trace event array or null | no | Optional execution trace. |
| `token_usage` | object or null | no | Aggregated model token usage. |
| `error` | string or null | no | Terminal error description. |

## Streaming

Streaming responses expose runtime events for thoughts, actions, observations,
the final answer, errors, and termination. `terminate` is emitted by the runtime;
it is not a model-visible action.

## Boundary rules

- The API validates and normalizes transport data but does not orchestrate tools.
- Legacy database aliases are normalized into `database_context`.
- Final answers, references, and visualization identifiers are carried by `FinalAnswer`.
- Business calculations belong to tools and authoritative artifacts, not the API layer.
