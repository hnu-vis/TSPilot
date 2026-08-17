# schemas/database.py SPEC

## Purpose

Define the evidence contract returned by `sql_query`.

## Models

- `DatabaseEvidence`
- `SchemaEvidence`
- `MetricListEvidence`
- `StatisticsEvidence`
- `TableEvidence`
- `TimeSeriesEvidence`

## `DatabaseEvidence`

Common fields:

- `evidence_id: str`
- `result_type: Literal["schema", "metric_list", "statistics", "table", "timeseries"]`
- `database: str`
- `query_language: str | null`
- `query: str | null`
- `summary: str`
- `data: dict`
- `columns: list[str]`
- `metadata: dict`
- `diagnostics: dict`

## Evidence families

### `schema`

Use when the question is about structure, fields, measurements, tags, or query planning.

`data` fields:

- `tables_or_measurements: list[dict]`
- `fields: list[dict]`
- `labels_or_tags: list[dict]`
- `time_columns: list[str]`

### `metric_list`

Use when the question is about available metrics or metric catalog lookup.

`data` fields:

- `metrics: list[dict]`

Each metric item should support:

- `name`
- `description`
- `labels`
- `source`

### `statistics`

Use when the question can be answered by scalar aggregates or grouped statistics.

`data` fields:

- `statistics: dict`
- `rows: list[dict]`
- `statistics_functions: list[str]`

Suggested statistic keys:

- `min`
- `max`
- `mean`
- `median`
- `sum`
- `count`
- `std`
- `latest`
- `change_percent`

### `table`

Use when the question requires grouped rows or row-level detail.

`data` fields:

- `rows: list[dict]`

### `timeseries`

Use when the question requires time-indexed points, trend analysis, forecast input, anomaly input, or chartable output.

`data` fields:

- `points: list[dict]`
- `time_field: str`
- `value_field: str`
- `series_name: str | null`
- `labels: dict`

## Contract notes

- `result_type` must always be present
- `evidence_id` must be stable within one request and usable by references, verified insights, and visualizations
- `summary` must always be human-readable
- `data` must match the chosen evidence family
- `sql_query` should return the smallest evidence family that answers the request
- if the user asks for trend / forecast / anomaly / chartable change, prefer `timeseries`
- if the user asks for a direct scalar answer, prefer `statistics`
- if the user asks for structure or available metrics, prefer `schema` or `metric_list`

## Responsibilities

- keep query results stable and typed
- make downstream analysis deterministic

## Must not do

- perform analysis narration
- produce final answer text
