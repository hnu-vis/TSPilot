# Query Architecture Specification

## Purpose

Define the decoupled, backend-agnostic architecture for `query_database`.

## Design goals

- support multiple backends without SQL-only assumptions
- separate schema grounding from query rendering
- make database execution inspectable in traces and E2E tests
- keep repair and normalization deterministic
- let runtime reason over logical plans rather than prompt-only query strings

## Contract layers

1. `SchemaCatalog`
2. `IntentInterpreter`
3. `FieldMapper`
4. `LogicalQueryPlanner`
5. `DialectRenderer`
6. `QueryExecutor`
7. `QueryValidator`
8. `QueryRepairPolicy`
9. `QueryResultSnapshotStore`
10. `EvidenceNormalizer`

## DB-GPT-inspired boundaries

This architecture should borrow DB-GPT's separation of concerns, but not depend
on DB-GPT classes, prompts, or runtime objects.

Borrow the following ideas:

- treat schema metadata as a first-class resource rather than hidden prompt text
- run schema linking before query rendering
- keep full query results outside the model prompt when they are large
- inject only a bounded prompt-visible projection of observations
- compact old observations and keep stable references to full snapshots

Do not borrow the following coupling patterns:

- DB-GPT-specific message formats
- DB-GPT-specific agent classes or middleware
- a SQL-only assumption in logical planning
- a requirement that visualization or answer layers understand backend syntax

## Primary code contracts

The canonical interface skeletons live in:

- [core/database/contracts.py](/home/feilvvl/TSPilot-v0.2/core/database/contracts.py)

The current reusable plan model lives in:

- [core/database/query_plan.py](/home/feilvvl/TSPilot-v0.2/core/database/query_plan.py)

The schema analysis and determination flow is captured as the `schema linking`
feature:

- [core/database/SPEC/schema_linking_SPEC.md](/home/feilvvl/TSPilot-v0.2/core/database/SPEC/schema_linking_SPEC.md)

## Runtime orchestration

Recommended orchestration inside `query_database`:

1. build `QueryRequestContext`
2. load schema through `SchemaCatalog`
3. interpret request into `QueryIntent`
4. ground fields through `FieldMapper`
5. build `DatabaseQueryPlan`
6. render the plan into backend query text or structured request
7. validate the rendered query against the logical plan
8. execute the query
9. persist the full raw result through `QueryResultSnapshotStore` when needed
10. validate the actual result shape
11. repair and retry when deterministic and safe
12. normalize into `DatabaseEvidence`
13. emit `QueryExecutionTrace`

## Current code mapping

Existing files already provide partial building blocks:

- schema grounding:
  - [core/database/schema.py](/home/feilvvl/TSPilot-v0.2/core/database/schema.py)
  - [core/database/schema_linker.py](/home/feilvvl/TSPilot-v0.2/core/database/schema_linker.py)
  - [core/database/SPEC/schema_linking_SPEC.md](/home/feilvvl/TSPilot-v0.2/core/database/SPEC/schema_linking_SPEC.md)
- logical plan:
  - [core/database/query_plan.py](/home/feilvvl/TSPilot-v0.2/core/database/query_plan.py)
- rendering:
  - [core/database/query_compiler.py](/home/feilvvl/TSPilot-v0.2/core/database/query_compiler.py)
  - [core/database/query_translator.py](/home/feilvvl/TSPilot-v0.2/core/database/query_translator.py)
- execution:
  - [core/database/engine.py](/home/feilvvl/TSPilot-v0.2/core/database/engine.py)
  - [core/database/connector.py](/home/feilvvl/TSPilot-v0.2/core/database/connector.py)
- repair:
  - [core/database/repair.py](/home/feilvvl/TSPilot-v0.2/core/database/repair.py)
- snapshot persistence:
  - [core/database/query_flow.py](/home/feilvvl/TSPilot-v0.2/core/database/query_flow.py)
- result shaping:
  - [core/database/result_processor.py](/home/feilvvl/TSPilot-v0.2/core/database/result_processor.py)
  - [core/database/engine.py](/home/feilvvl/TSPilot-v0.2/core/database/engine.py)

## Migration plan

### Phase 1

- keep existing `query_database` runtime behavior
- replace ad-hoc local routing helpers with `QueryRequestContext` and
  `QueryIntent`
- route schema grounding through `SchemaLinker`
- surface logical plan and rendered query in traces

### Phase 2

- promote `QueryCompiler` into a real `DialectRenderer` abstraction
- restrict `QueryTranslator` to optional NL fallback or dialect-specific
  synthesis, not as the only planner
- add non-SQL renderers for Flux and PromQL

### Phase 3

- add deterministic `QueryValidator`
- add repair policies for time-range loss, wrong aggregation, and empty-result
  retries
- normalize repair attempts into traceable execution events
- persist large raw results via a stable snapshot reference instead of pushing
  them into model-visible observations

### Phase 4

- split mixed requests into multi-query plans when needed
- preserve both aggregate evidence and raw-series evidence
- let downstream `insight` consume raw evidence without losing query-native
  facts
- formalize prompt projection rules so the model sees summaries while tools read
  full artifacts

## Constraints

- upstream planning modules must not depend on SQL, Flux, or PromQL syntax
- backend-specific syntax belongs only in renderers or connector adapters
- `insight` must not query databases directly
- answer formatting must consume evidence, not infer backend query semantics
- full raw results must be storable independently from prompt-visible summaries
