# tools/format_answer.py SPEC

## Purpose

Validate and assemble the structured `FinalResponsePlan` authored by the outer
ReAct model's terminal call. Visualization planning and materialization are not
part of this tool.

## Input

- `response_plan.title`, summary, and grounded sections
- `visualization_ids[]` returned by successful `visualization` tool calls

## Pipeline

1. Resolve every section reference against the request-scoped
   `PresentationCatalog` or the selected visualization ids.
2. Select only existing visualization descriptors and validate their durable
   `data_ref` and source lineage.
3. Expand Data View lineage into answer references and claim-to-visual links.
4. Assemble the final answer without invoking an internal LLM.
5. Route a missing/broken visualization artifact to `visualization`; input-only
   response-plan errors remain terminal-plan repairs.

## Invariants

- No database query, analysis execution, visualization planning/materialization,
  or internal LLM call.
- No deterministic visualization fallback.
- The formatter never truncates, copies, or recomputes visualization data.
- Full visualization data remains behind the descriptor's `data_ref`.
