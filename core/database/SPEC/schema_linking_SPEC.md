# Schema Linking Specification

## Purpose

`schema linking` is the query pipeline feature that determines which datasource
objects a user request is allowed to query before any backend query is rendered
or executed.

It turns a natural-language request plus a real datasource schema into grounded
query constraints:

- source: table, measurement, metric, or equivalent datasource object
- time field: timestamp column, `_time`, or metric time axis
- value field: numeric field such as price, value, energy, or rate
- dimensions: tags, labels, category columns, grouping keys
- value-domain filters: concrete filter values such as `code = USD` or
  `crypto = bitcoin`
- ambiguity diagnostics: candidate objects when the request cannot be grounded
  confidently

## Root Problem

The model may generate syntactically valid queries that are semantically wrong:

- using the right measurement but losing entity filters
- mixing currencies, devices, regions, or labels
- selecting an aggregate when the user asked for raw records
- guessing field names from user-facing text instead of schema metadata

`schema linking` addresses this at the input side. It identifies the semantic
constraints that must survive into the query, then downstream validators can
reject or repair rendered queries that drop those constraints.

## Inputs

- `QueryRequestContext`
  - user message
  - datasource id and type
  - time range
  - constraints
  - intent profile
- `DatabaseSchema`
  - tables, measurements, metrics, or views
  - columns, fields, tags, or labels
  - column data types and units when available
  - `metadata.value_domains` for known dimension values

## Outputs

- `SchemaLinkingPipelineResult`
  - `linking`
  - `field_mappings`
  - `plan`
  - `required_filters`
- `SchemaLinkingResult`
  - linked sources
  - linked columns
  - inferred time columns
  - inferred value columns
  - join or alignment keys
  - ambiguous terms
  - confidence and evidence strings
- `DatabaseQueryPlan.schema_linking`
  - a traceable copy of the linking result used to build the logical plan
- required filters in `DatabaseQueryPlan.filters`
  - filters inferred from matched value domains in the user request

## Flow

1. Load real schema through `SchemaCatalog` or the connector's `get_schema`.
2. Match request terms against source names and column names.
3. Infer time, value, and dimension roles from names, data types, and units.
4. Match user terms against `metadata.value_domains`.
5. Convert matched value domains into required logical filters.
6. Build a conservative backend-agnostic `DatabaseQueryPlan`.
7. Validate rendered or explicit queries against the plan before execution.
8. Record linking output in diagnostics and query trace.

## Guarantees

- Query rendering must preserve linked filters, dimensions, and time bounds.
- Explicit model-written queries are still validated against linked filters.
- If a required filter is missing, execution must fail closed before querying.
- Backend syntax belongs in renderers and connectors, not in schema linking.
- The linking result must be inspectable in traces and tests.

## Current Implementation

- [core/database/schema_linking.py](/home/feilvvl/TSPilot-v0.2/core/database/schema_linking.py)
  is the single orchestration entrypoint. It links schema objects, builds field
  mappings, applies value-domain filters, and returns a grounded
  `DatabaseQueryPlan` plus required filters.
- [core/database/schema_linker.py](/home/feilvvl/TSPilot-v0.2/core/database/schema_linker.py)
  links schema sources and columns, infers roles, and builds a conservative
  plan seed used by the pipeline.
- [core/database/query_flow.py](/home/feilvvl/TSPilot-v0.2/core/database/query_flow.py)
  calls `SchemaLinkingPipeline` once in the default path, then renders and
  validates the returned plan.
- [tools/sql_query.py](/home/feilvvl/TSPilot-v0.2/tools/sql_query.py)
  uses `SchemaLinkingPipeline` to validate explicit read-only queries against
  required linked filters before execution.

## Example

Request:

```text
查询当前数据源中比特币 USD 价格的最晚一条原始记录
```

Schema domains:

```json
{
  "coindesk": {
    "code": ["EUR", "GBP", "USD"],
    "crypto": ["bitcoin"]
  }
}
```

Linked requirements:

- source: `coindesk`
- value field: `price`
- filters:
  - `code = "USD"`
  - `crypto = "bitcoin"`
- query shape: raw latest record

Any rendered or explicit query that omits `code = "USD"` or
`crypto = "bitcoin"` is semantically invalid even if the backend accepts it.
