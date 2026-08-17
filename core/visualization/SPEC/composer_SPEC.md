# core/visualization/composer.py SPEC

## Purpose

Plan visualization semantics from selected verified Insights and materialize chart data exclusively from request-scoped artifacts.

## Contract

- The LLM sees only bounded Insight, Evidence, and existing-visualization inventories, never the full data arrays.
- Plans may reference only listed Insight IDs, Evidence IDs, visualization IDs, fields, renderers, and marks.
- Materialization reads full `database_evidence_artifacts`, applies bounded visual sampling, and creates explicit bindings.
- Existing forecast/anomaly views are composed in place when compatible Insights need highlights; duplicate contextual charts are rejected.
- Forecast presentation plans select a bounded historical context window and may request explicit independent scales when supplied series ranges cannot share a readable scale.
- Insight marks remain separate from sampled context data and retain stable Insight/item bindings.
- Context lines do not require point-level bindings; prediction, anomaly, and Insight interaction bindings are retained only for visible interactive marks.
- Invalid plans are repaired through the LLM planning contract; the composer does not use a heuristic chart fallback.
- The composer must not query a database or invent data absent from an artifact.

## Supported renderers

- `metric`
- `table`
- `linechart`
- `barchart`

## Supported marks

- `line`
- `point`
- `bar`
- `rule`
- `band`
- `label`

Rules may be vertical (`x_field`) or horizontal (`y_field`). Bands use
`x_field`/`x2_field` for temporal intervals or `y_field`/`y2_field` for value ranges.
