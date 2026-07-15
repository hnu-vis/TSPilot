# schemas/output.py SPEC

## Purpose

Define the final user-facing answer contract.

## Models

- `FinalAnswer`
- `AnswerSection`
- `AnswerReference`

## `FinalAnswer`

Fields:

- `title: str | null`
- `summary: str`
- `sections: list[AnswerSection]`
- `references: list[AnswerReference]`
- `visualizations: list[VisualizationPayload]`

## `AnswerSection`

Fields:

- `section_type: str`
- `heading: str | null`
- `content: str`
- `structured_payload: dict | null`

## `AnswerReference`

Fields:

- `source_type: Literal["query", "statistics", "fact", "forecast", "anomaly", "rag", "skill"]`
- `source_id: str | null`
- `label: str`
- `evidence: dict | null`

## Contract notes

- `summary` should be concise and directly answer the user
- `sections` may carry structured payloads such as verified fact ids and visualization ids
- `references` must point to grounded evidence or facts
- `source_id` should point to a stable runtime artifact id such as `evidence_id`, `fact_id`, `forecast_id`, or `anomaly_id`
- `visualizations` are the only backend payload for chart rendering

## Responsibilities

- define the answer assembly target for `format_answer`
- keep the final response grounded and structured

## Must not do

- perform analysis
- query databases
