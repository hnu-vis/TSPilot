# agents/data_agent.py SPEC

## Purpose

Implement the single outer `data_agent`.

## Inputs

- `ChatRequest`
- `RequestStateModel`
- `ConversationStateModel`

## Outputs

- one parsed ReAct turn per model step
- updated states
- terminal response payload via the runtime loop
- trace events

## Responsibilities

- perform lightweight intent recognition
- decide whether the request needs todo planning
- decide whether the request needs query, analysis, forecast, anomaly, rag, or skill
- choose one outer action per turn
- keep the prompt aligned with the schema and runtime contracts

## Reads

- request state
- conversation state
- normalized database context
- allowed action space

## Writes

- request state
- conversation state
- action plan
- trace events

## Forbidden responsibilities

- low-level database execution
- deterministic verification rules
- final presentation assembly
- multiple-agent orchestration

## Prompt-contract alignment

The implementation must ensure the prompt sees the same structured fields
defined in:

- `schemas.api`
- `schemas.state`
- `schemas.database_context`
- `schemas.database`
- `schemas.analysis`
- `schemas.timeseries`
- `schemas.output`
- `schemas.agent_turn`
