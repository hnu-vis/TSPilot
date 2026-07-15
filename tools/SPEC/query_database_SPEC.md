# tools/query_database.py SPEC

## Purpose

Retrieve normalized database evidence for the current request.

## Input

- `message: str`
- `database_context: DatabaseContext | null`
- `selected_database: str | null`
- `selected_database_type: str | null`
- `time_range: dict | null`
- `constraints: dict`
- optional `history: list[Message]`

## Output

- `DatabaseEvidence`

## Reads

- request context
- conversation context
- database adapter registry
- `database_context` first, legacy aliases second

## Writes

- evidence payload
- diagnostics

## Internal pipeline

1. resolve database context
2. inspect datasource capabilities and schema catalog
3. map user intent into a backend-agnostic query plan
4. render the plan into the selected backend query language
5. execute query
6. validate, repair, and retry when safe
7. normalize result into one evidence family

## Backend-agnostic query architecture

`query_database` must be designed around a logical query-planning layer rather
than around any single query language such as Flux or SQL.

This design may borrow from systems such as DB-GPT at the architectural level:

- separate schema grounding from query generation
- separate logical planning from backend dialect rendering
- expose datasource adapters behind stable interfaces
- validate and repair generated queries before surfacing analytical results

The goal is to reuse the good separation of concerns, not to couple this tool
to DB-GPT-specific classes, prompts, or runtime assumptions.

The core abstraction is:

1. schema grounding
2. intent parsing
3. logical query planning
4. dialect rendering
5. execution validation

This tool must support multiple datasource families, for example:

- SQL databases
- InfluxDB / Flux-style time-series databases
- Prometheus / PromQL-style metric stores
- other adapters that can expose a deterministic capability contract

The runtime-visible execution trace should expose both:

- the logical query plan
- the rendered backend query text or structured request

Examples:

- for Postgres or DuckDB, show generated SQL
- for InfluxDB, show generated Flux
- for Prometheus, show generated PromQL
- for REST-style adapters, show the normalized request payload or endpoint call

The product requirement is "show the actual database query sent to the
backend", not "always show SQL".

## Decoupled implementation modules

The implementation should remain decomposed into small replaceable components.

Recommended module boundaries:

- `SchemaCatalog`
  - introspect tables, measurements, fields, tags, dimensions, and capabilities
- `SchemaLinker`
  - rank candidate tables, measurements, fields, and labels against the request
  - expose why one candidate beat another
- `IntentInterpreter`
  - convert the user request into requested fact families, filters, and
    evidence-shape hints
- `FieldMapper`
  - resolve user-facing concepts onto grounded backend objects
- `LogicalQueryPlanner`
  - build a backend-agnostic query plan
- `DialectRenderer`
  - render that plan into SQL, Flux, PromQL, or another adapter-native request
- `QueryExecutor`
  - execute the rendered query through the adapter
- `QueryValidator`
  - compare rendered/executed behavior against the logical plan
- `QueryRepairPolicy`
  - define safe retries when grounding or rendering was close but wrong
- `QueryResultSnapshotStore`
  - persist full raw results or execution snapshots outside the prompt context
- `EvidenceNormalizer`
  - map raw backend results into `DatabaseEvidence`

Dependency rule:

- upstream planning modules must not depend on backend-specific syntax
- renderers and executors may depend on backend dialect details
- normalizers must depend on backend result shapes but not on model prompts
- the outer runtime should orchestrate these interfaces, not reimplement them

This allows:

- swapping one renderer without rewriting planning
- adding a new database backend without changing intent parsing
- testing field mapping independently from execution
- testing repair logic independently from normalization
- keeping full result storage independent from prompt compaction

## Schema grounding and field mapping

Before drafting a backend query, the tool should ground the request against the
real datasource schema or metric catalog.

Grounding should identify, when available:

- database / datasource id
- namespace such as catalog, schema, bucket, or project
- table / measurement / metric family
- timestamp column or time field
- numeric value fields
- dimensions or tags
- valid filter fields and filter values
- supported aggregate functions

Field mapping must not rely purely on free-form model guessing. The planning
layer should map user-facing concepts such as "Bitcoin USD", "price", "过去一周",
"每天", or "按地区分组" onto verified backend objects.

Minimum mapping responsibilities:

- map user metric names to concrete fields or measurements
- map time-range constraints into backend-native time predicates
- map grouping requests into backend-native group-by constructs
- map "raw series" versus "aggregate summary" requests into different query
  shapes

If grounding is ambiguous, the tool should either:

- choose the best verified candidate and record diagnostics, or
- fail closed with diagnostics before executing a likely-wrong query

## Logical query plan

The planning layer should produce a backend-agnostic plan before rendering.

Recommended plan fields:

- `target_entities`
- `selected_metrics`
- `time_range`
- `filters`
- `group_by`
- `requested_fact_families`
- `evidence_family`
- `query_shape`
- `requires_raw_points`
- `requires_full_fidelity`
- `sampling_policy`

Recommended `query_shape` values:

- `raw_timeseries`
- `scalar_aggregate`
- `grouped_aggregate`
- `row_detail`
- `schema_introspection`
- `metric_discovery`

Planning rule examples:

- "平均值/总和/最大值" usually implies `scalar_aggregate` or
  `grouped_aggregate`
- "趋势/周期性/异常/预测" implies `raw_timeseries`
- "有哪些字段/有哪些指标" implies introspection or metric discovery

For mixed requests, the planner should prefer the richest evidence shape needed
to satisfy the full request. If the request asks for both summary aggregates and
time-series analysis, the long-term preferred behavior is to either:

- issue multiple coordinated queries, or
- issue one raw query that preserves downstream analysis fidelity and derive
  simpler facts later when safe

## Dialect rendering

Rendering is backend-specific, but it must preserve the semantics of the
logical query plan.

Required rendering guarantees:

- the user-provided absolute time range must survive into the rendered query
- aggregation functions must only appear when the logical plan requires them
- dimensions, filters, and grouping keys must be preserved
- the renderer must not silently downgrade `raw_timeseries` into aggregated
  evidence

Examples of backend-specific renderings from the same logical plan:

- SQL: `SELECT ts, price FROM btc_usd WHERE ts BETWEEN ...`
- Flux: `from(bucket: ...) |> range(start: ..., stop: ...) |> ...`
- PromQL: `metric_name{label="value"}[window]`

The exact syntax differs by backend; the logical plan semantics must not.

## Query validation and repair

After rendering and execution, `query_database` should validate that the query
matched the intended plan.

Minimum validation checks:

- did the rendered query include the intended absolute time range
- did the query shape match the requested evidence family
- was an aggregate added unexpectedly
- did the query reference existing tables / measurements / fields
- did the query return empty results

Safe repair behavior may include:

- retrying with alternate grounded field candidates
- retrying with a corrected time predicate
- removing an accidental aggregate when the request requires raw points
- trying a different verified measurement or metric alias

Repair must not silently change the analytical meaning of the request. If the
tool cannot repair safely, it should return a failed observation with the
attempted queries and the observed failure reason.

## Implementation guidance inspired by DB-GPT-style systems

Practical design choices worth following:

- keep datasource metadata access explicit instead of hidden inside prompts
- keep schema-linking output inspectable, not buried inside one prompt
- keep query-language rendering in adapter-specific modules
- keep generated-query validation deterministic
- keep execution traces inspectable for debugging and E2E testing
- keep full query results outside the prompt when they are large, and reference
  them through stable artifact ids or snapshot paths

Practical design choices to avoid:

- one monolithic prompt that performs schema linking, planning, rendering, and
  repair in a single opaque step
- mixing full raw result transport with model-visible observation text
- embedding backend-specific assumptions directly into runtime control flow
- coupling `insight` or answer formatting to one query language
- treating SQL generation as the universal path when some backends are not SQL

## Evidence selection guidance

- `schema` for structure or query-planning questions
- `metric_list` for available metrics discovery
- `statistics` for scalar aggregates or summary values
- `table` for grouped rows or row-level detail
- `timeseries` for trend, forecast, anomaly, and chartable requests

## Fact execution boundary

`query_database` is the preferred execution layer for query-native facts that can
be answered directly by the datasource or by a deterministic reference-dataset
aggregate without requiring downstream time-series analysis.

Typical query-native fact families:

- `aggregation`
- `extreme`
- some `difference`
- some grouped `rank`
- some grouped `proportion`

Implications:

- if a request can be answered by scalar or grouped aggregate evidence, prefer
  `statistics` or `table` over `timeseries`
- if the user mixes aggregate and trend-style requests in one message, the
  runtime may still choose `timeseries` evidence to preserve raw rows/points for
  downstream analysis
- this tool should return the richest evidence family required by the full
  request, not merely the cheapest family implied by one keyword
- do not assume these facts are rendered in SQL only; the same boundary applies
  across SQL, Flux, PromQL, and other backend adapters

Current limitation:

- evidence-family routing is still message-keyword driven rather than fact-plan
  driven, so mixed requests such as "先给均值，再分析趋势" may require a future
  planning layer to split query-native facts from analysis-native facts more
  precisely

## Trace and testability contract

For end-to-end testing and debugging, the tool output or trace should make the
database execution process inspectable.

At minimum, traces should expose:

- resolved datasource id and adapter type
- schema summary and grounded field candidates
- logical query plan
- rendered backend query text or structured request
- whether the query was original or repaired
- raw result summary such as row count / point count
- stable references to persisted full results when prompt compaction applies
- normalized evidence family

This contract exists so a reviewer can tell whether a result was truly grounded
in datasource execution, and whether the backend query matched the user request.

## Contract notes

- return the smallest evidence family that answers the request
- if the request is chartable or time-based, prefer `timeseries`
- if the request can be answered by a scalar aggregate, prefer `statistics`
- `summary` must be human-readable
- `data` must match the chosen evidence family

## Must not do

- produce final fact narration
- produce final answer text
- decide charts or presentation layout
