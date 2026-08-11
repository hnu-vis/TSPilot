# tools/format_answer.py SPEC

## Purpose

Validate and materialize the structured `FinalResponsePlan` authored by the
outer ReAct model's terminal call.

## Input

- `response_plan.title`
- grounded prose in `summary` and `sections[]`
- semantic `visual_intents[]` containing template IDs, source refs, Fact refs,
  and optional source-field encodings

## Pipeline

1. Resolve every referenced Fact, Evidence, Analysis, Forecast, and Anomaly
   against the request-scoped `PresentationCatalog`.
2. Reject invented references, incompatible templates, and duplicate primary
   views for one purpose.
3. Read complete canonical artifacts at this final boundary.
4. Materialize typed V2 datasets, semantic layers, sampling, bindings, and
   scale/facet layout without invoking an LLM.
5. Assemble sections, references, claims, and visualizations into `FinalAnswer`.

## Invariants

- No database query, analysis execution, or internal LLM call.
- No silent visualization fallback or swallowed materialization error.
- No tool-produced forecast/anomaly chart is reused; those tools produce data
  artifacts and the final formatter owns presentation.
- The outer model decides what the user needs to see; deterministic code only
  performs renderer mechanics such as sampling and legibility.
