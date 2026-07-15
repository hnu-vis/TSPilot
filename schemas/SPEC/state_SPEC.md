# schemas/state.py SPEC

## Purpose

Define the mutable request and conversation state used by runtime and tools.

## Models

- `RequestStateModel`
- `ConversationStateModel`

This schema also depends on:

## `RequestStateModel`

### Identity and request context

- `request_id: str`
- `conversation_id: str | null`
- `message: str`
- `database_context: DatabaseContext | null`
- `selected_database: str | null`
- `selected_database_type: str | null`
- `time_range: dict | null`
- `constraints: dict`
- `history: list[Message]`

### Control state

- `status: Literal["running", "completed", "failed"]`
- `current_intent: str | null`
- `requested_fact_types: list[str]`
- `focus: str | null`
- `todo_list: list[dict]`
- `iteration: int`
- `max_iterations: int`
- `context_budget: dict`
- `context_status: Literal["ok", "summarized", "truncated", "overflowed"]`
- `context_overflow_reason: str | null`

`context_budget` should carry the runtime prompt-window limits, such as:

- `max_prompt_tokens`
- `max_history_messages`
- `max_tool_history_items`
- `max_observation_chars`
- `overflow_policy`

### Evidence state

- `latest_database_evidence: DatabaseEvidence | null`

### Analysis state

- `latest_insight: InsightResult | null`
- `latest_forecast: ForecastResult | null`
- `latest_anomaly: AnomalyResult | null`
- `latest_rag: dict | null`
- `latest_skill: dict | null`
- `verified_facts: list[VerifiedFact]`
- `rejected_facts: list[RejectedFact]`

### Presentation state

- `final_answer_draft: FinalAnswer | null`
- `visualizations: list[VisualizationPayload]`

### Trace / runtime state

- `tool_history: list[ToolCall]`
- `observations: list[ToolObservation]`
- `errors: list[dict]`
- `prompt_context_summary: str | null`

## `ConversationStateModel`

Fields:

- `conversation_id: str`
- `database_context: DatabaseContext | null`
- `recent_messages: list[Message]`
- `session_summary: str | null`
- `latest_database_evidence: DatabaseEvidence | null`
- `latest_insight: InsightResult | null`
- `latest_forecast: ForecastResult | null`
- `latest_anomaly: AnomalyResult | null`
- `latest_rag: dict | null`
- `latest_skill: dict | null`
- `recent_visualizations: list[VisualizationPayload]`
- `updated_at: str | null`
- `context_budget: dict | null`

`context_budget`, when present, should mirror the runtime limits used for compaction and prompt assembly.

## Contract notes

- request state is mutable per request and may be updated every loop turn
- conversation state is short session context, not long-term memory
- state objects must keep control, evidence, analysis, and presentation concerns separate
- legacy database aliases should be normalized into `database_context`
- terminal response assembly should read from presentation state instead of mirroring internal analysis fields into the external API

## Responsibilities

- provide a shared mutable contract for runtime and tools
- prevent hidden state from living in prompts alone

## Must not do

- execute runtime logic
- contain business rules
