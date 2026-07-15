# schemas/visualization.py SPEC

## Purpose

Define the structured payload used to render charts and other visual artifacts.

## Model

### `VisualizationPayload`

Fields:

- `visualization_id: str`
- `visualization_type: Literal["chart", "table", "metric_card", "annotation"]`
- `visualization_kind: str`
- `renderer: str`
- `title: str`
- `summary: str | null`
- `chart: dict | null`
- `annotations: list[dict]`
- `binding_fact_ids: list[str]`
- `binding_evidence_ids: list[str]`
- `requested_fact_types: list[str]`
- `subject: dict`
- `presentation: dict`
- `row_count: int | null`
- `columns: list[str]`
- `rows: list[dict]`
- `display_rows: list[dict]`
- `time_column: str | null`
- `primary_measure: str | null`
- `legend: list[dict]`
- `display_priority: int`
- `render_hints: dict`

## Contract notes

- `renderer` should default to `linechart` when the evidence is time-indexed
- ratio, comparison, and trend facts should prefer `linechart` when possible
- grouped non-time evidence may render as `table`, `bar`, or another faithful view
- `binding_fact_ids` and `binding_evidence_ids` must point to grounded facts/evidence
- `chart` may be null for non-chart visualizations such as metric cards or tables

## Responsibilities

- carry chart-ready structured data
- let the frontend render without re-inferring facts

## Must not do

- contain business analysis rules
- invent evidence not present in state
