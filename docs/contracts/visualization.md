# LineChart Visualization V4 Contract

The canonical schema is defined in `schemas/visualization.py`. Each
`VisualizationPayload` represents one renderer-independent, evidence-grounded
LineChart.

## Payload structure

A payload contains:

- identity, title, summary, purpose, and primary/supporting priority;
- verification metadata, `source_refs`, and a persistent `data_ref`;
- typed `data_views` with fields and materialized records;
- one x-axis and one or more semantically compatible y-axes;
- `lines`, `points`, `bands`, `intervals`, `reference_lines`, and `annotations`;
- legend, tooltip, zoom, evidence bindings, and accessibility metadata.

Every visual component references a real `view_id` and source field. Component
bindings preserve Insight and Evidence lineage for frontend hover and click
interactions.

## Semantic rules

- `chart_type` is `line` and `schema_version` is `4`.
- A line requires more than one positionable record.
- Lines sharing an axis must use compatible measures and units.
- Reference lines must use a real value with the same measure and unit as their target y-axis.
- Other scalar results are represented as annotations instead of misleading reference lines.
- Annotation targets are limited to chart, x, xy, and interval targets.
- Required insights must be covered by a visible component and a valid binding.
- A supporting chart is used only when data domains or metric semantics cannot share the primary chart.

## Persistence and hydration

The artifact store persists full records. Final answers carry dehydrated
descriptors: records and accessibility table rows are removed, while field
metadata, row count, time range, source references, and `data_ref` remain. The
frontend hydrates the descriptor from the artifact endpoint before rendering.

Old V3 and pre-LineChart V4 artifacts are not migrated or interpreted. They are
handled by the generic unsupported-schema path so historical sessions do not
break the current renderer.
