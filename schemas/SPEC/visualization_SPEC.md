# schemas/visualization.py SPEC

## Purpose

Define the breaking V3 renderer-independent visualization contract. V3
represents a user-visible goal as grounded datasets and semantic layers; it has
no template registry.

## Contract

`VisualizationPayload` contains:

- `schema_version: "3"`
- `purpose`, `priority`, `title`, and optional summary
- `required_roles` copied from the semantic visual goal
- one or more typed `datasets`, each with its own `source_ref`
- a durable `data_ref`; descriptors also expose `row_count` and `time_range`
- composable `layers`; every layer declares `mark`, semantic `role`, one
  `source_ref`, one `dataset_id`, and field `encoding`
- provenance bindings for interactive Insight and Insight-item marks
- computed layout (`overlay` or explicit `facets`)
- an accessible description and bounded data table

The visualization tool persists the full payload as an artifact and returns a
lightweight descriptor. The frontend retrieves the complete payload lazily from
`GET /api/v1/visualizations/{visualization_id}/data` when the chart is rendered.

Supported marks are `line`, `point`, `bar`, `area`, `band`, `rule`, `rect`,
`text`, `boxplot`, and `table`.

## Invariants

- Payloads never contain ECharts options, JavaScript formatters, SVG, or
  model-generated renderer configuration.
- Every layer resolves to a request-scoped source and dataset.
- Every required semantic role has a non-empty materialized layer.
- Data View encodings name real schema fields and their lineage resolves to
  canonical artifacts.
- Unverified Insights cannot produce visual marks.
- Passive context series do not carry one binding per row; semantic decision
  marks retain Insight-item identity.
- Full time-series datasets are retained in the artifact so global patterns are
  visible. Only prompt previews and accessibility tables are bounded.
- Semantic extrema retain their locator row (including timestamp) and are
  materialized as explicit marks alongside the full base series.
- Analysis filtering and business calculations happen before presentation and
  are published as typed Data Views.
