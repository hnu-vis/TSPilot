# schemas/database_context.py SPEC

## Purpose

Define the normalized database selection context used across API, state, tools,
and prompts.

## Model

### `DatabaseContext`

Fields:

- `database_id: str`
- `database_type: str`
- `display_name: str | null`
- `connection_hint: str | null`
- `selected_at: str | null`

## Contract notes

- `database_context` is the primary database selection object
- `database_id` identifies the selected database instance or datasource
- `database_type` identifies the backend family or dialect, such as
  `prometheus`, `iotdb`, `clickhouse`, or another supported backend
- `display_name` is for human-readable UI/debug labels only
- `connection_hint` is optional backend routing metadata
- `selected_at` records when the selection was made
- legacy fields such as `selected_database` and `selected_database_type` may be
  accepted during transition, but they must be normalized into `DatabaseContext`

## Responsibilities

- carry one normalized database choice across the stack
- eliminate repeated inference of database id/type in downstream modules

## Must not do

- infer adapter behavior
- execute queries
- contain request logic
