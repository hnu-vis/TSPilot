# schemas/visualization.py SPEC

## Purpose

Define the breaking V3 renderer-independent visualization contract. V3
represents a user-visible goal as grounded datasets and semantic layers; it has
no template registry.

Visualization planning has two LLM-owned stages. Semantic projection first
interprets the request together with raw Evidence and verified Insights, then
publishes request-scoped semantic views by selecting, renaming, and reorganizing
existing nested values. Chart planning consumes only those semantic views and
chooses the layers, encodings, composition, and renderer-native presentation.
The materializer executes both plans but does not independently infer business
field roles or substitute a deterministic chart when either plan fails.

## Contract

`VisualizationPayload` contains:

- `schema_version: "3"`
- `purpose`, `priority`, `title`, and optional summary
- `required_roles` authored by chart planning as a description of the completed
  visual expression; no separate business-role matcher reinterprets them
- one or more typed `datasets`, each with its own `source_ref`
- a durable `data_ref`; descriptors also expose `row_count` and `time_range`
- composable `layers`; every layer declares an open renderer-native `mark`,
  semantic `role`, one `source_ref`, one `dataset_id`, field `encoding`, and an
  optional data-free `presentation` object
- provenance bindings for interactive Insight and Insight-item marks
- computed layout (`overlay` or explicit `facets`)
- an accessible description and bounded data table

The visualization tool persists the full payload as an artifact and returns a
lightweight descriptor. The frontend retrieves the complete payload lazily from
`GET /api/v1/visualizations/{visualization_id}/data` when the chart is rendered.

The backend does not maintain a closed mark registry. Graphical ECharts series
types can pass through without backend changes; `text` and `table` remain
excluded because they are answer content rather than graphical visualization.

## Invariants

- Payloads may contain JSON-only renderer presentation options, but those
  options cannot contain `data`, `source`, `dataset`, `dimensions`, `series`, or
  `encode`. The materializer and frontend inject those properties from grounded
  sources after presentation options are applied.
- Every layer resolves to a request-scoped source and dataset.
- Every required semantic role has a non-empty materialized layer.
- Semantic-view encodings name real projected fields and their recursive lineage
  resolves to canonical artifacts.
- Encoding channel names and multi-field bindings are renderer-native and open;
  every referenced field is still validated against the selected source.
- Unverified Insights cannot produce visual marks.
- Passive context series do not carry one binding per row; semantic decision
  marks retain Insight-item identity.
- Full time-series datasets are retained in the artifact so global patterns are
  visible. Only prompt previews and accessibility tables are bounded.
- Semantic extrema retain their locator row (including timestamp) and are
  materialized as explicit marks alongside the full base series.
- Analysis filtering and business calculations happen before visualization.
  Semantic projection may compose existing meanings, but cannot calculate,
  aggregate, predict, rescale, or manufacture values.
