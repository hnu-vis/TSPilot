# schemas/visualization.py SPEC

## Purpose

Define the V2 renderer-independent presentation contract used by the final
answer boundary and the ECharts template registry.

## Contract

`VisualizationPayload` contains:

- `schema_version: "2"`
- a supported analytical `template_id`
- `purpose` and `priority` (`primary` or `supporting`)
- grounded `source_refs` and `fact_refs`
- a typed dataset of dimensions, series, points, rows, or one metric
- composable semantic layers (`line`, `bar`, `point`, `rule`, `area`, `band`,
  `boxplot`, `scatter`)
- bindings only for interactive semantic marks
- computed `layout` (`overlay` or explicit `facets`)
- an accessible description and bounded data table

## Invariants

- Payloads never contain ECharts options, JavaScript formatters, SVG, or model-
  generated data arrays.
- Every point binding must resolve to a binding entry.
- Passive context series do not carry one binding per row.
- Fact, anomaly, prediction, interval, and boundary marks retain provenance.
- Source arrays are read from canonical artifacts and sampled mechanically;
  semantic marks are never removed by sampling.
- Incompatible or visually unreadable scales use labelled facets instead of
  independent scales overlaid in one plot.

## Supported templates

Metric, detail table, TopK ranking, time-series trend/highlight/interval,
forecast, anomaly, category/time-series comparison, histogram, boxplot, and
scatter.
